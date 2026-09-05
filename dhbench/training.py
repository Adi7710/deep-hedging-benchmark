"""The training loop — where deep hedging actually happens.

    for each batch:
        spot  = world(batch)                 fresh paths, never reused
        delta = agent.hedge_path(spot)       the rollout, INSIDE the tape
        pnl   = terminal_pnl(...)            dhbench.pnl, the single source of truth
        loss  = objective(pnl)               one scalar from the whole distribution
        backpropagate through every hedging decision, step

That is the entire method. Everything else in this repository exists so that the number
coming out of it can be believed.

Three design choices, each with consequences worth knowing.

**Fresh paths every batch.** The model never sees the same path twice, so classical
overfitting is not the failure mode and "epoch" is meaningless -- budget is counted in
gradient steps and paths consumed. Evaluation uses a *disjoint seed stream* rather than a
held-out split. See ``docs/05-stage-1-2-plan.md`` §2.4.

**The world is injected as a callable.** ``world(n_paths, generator) -> spot``. Swapping
GBM for Heston is a change of argument, not of loop. It is also what makes vol-robust
training (``docs/03``) a one-line variant: a world that draws ``sigma`` per batch rather
than fixing it, so the policy never learns which market it is in.

**The objective may carry its own variables.** CVaR in Rockafellar-Uryasev form has a
trainable ``w`` that must be optimised jointly with the network. Omitting it is the classic
CVaR deep hedging bug: training runs, the loss falls a little, then flatlines, and it reads
as a learning-rate problem rather than a formulation one. :func:`trainable_variables`
collects both so a caller cannot forget.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Protocol

import tensorflow as tf
from tensorflow import keras

from dhbench.pnl import terminal_pnl
from dhbench.seeding import make_generator

__all__ = ["train", "trainable_variables", "TrainingResult"]


class _Objective(Protocol):
    def __call__(self, pnl: tf.Tensor) -> tf.Tensor: ...


def trainable_variables(agent: keras.Model, objective: Any) -> list[tf.Variable]:
    """Every variable the optimiser must see: the network's, plus the objective's.

    Objectives without state (entropic, mean-variance) contribute nothing. CVaR
    contributes its Rockafellar-Uryasev auxiliary ``w``, which is *not* optional --
    see the module docstring.
    """
    return list(agent.trainable_variables) + list(
        getattr(objective, "trainable_variables", [])
    )


class TrainingResult(dict):
    """Loss history plus the diagnostics needed to tell training from thrashing."""

    @property
    def losses(self) -> list[float]:
        return self["losses"]

    @property
    def improved(self) -> bool:
        """Did the last decile beat the first? The weakest useful convergence check."""
        n = max(1, len(self.losses) // 10)
        return sum(self.losses[-n:]) / n < sum(self.losses[:n]) / n


def train(
    agent: keras.Model,
    objective: _Objective,
    world: Callable[[int, tf.random.Generator], tf.Tensor],
    payoff: Callable[[tf.Tensor], tf.Tensor],
    *,
    maturity: float,
    strike: float,
    cost_rate: float = 0.0,
    premium: float = 0.0,
    rate: float = 0.0,
    batch_size: int = 1024,
    n_gradient_steps: int = 500,
    learning_rate: float = 5e-3,
    seed: int = 0,
    stream: str = "train",
    log_every: int = 0,
    compile_step: bool = True,
) -> TrainingResult:
    """Train a hedging policy by direct policy optimisation.

    agent:     provides ``hedge_path(spot, maturity, strike) -> (n_paths, n_steps)``
    objective: ``(n_paths,) -> scalar``, lower is better
    world:     ``(n_paths, generator) -> (n_paths, n_steps + 1)``
    payoff:    ``(spot) -> (n_paths,)``, the claim we are SHORT

    Returns a :class:`TrainingResult` carrying the loss history, paths consumed, wall
    time, and the seed actually used.

    compile_step: wrap the inner step in ``tf.function``. On by default -- measured
        16.9x faster (92.4 ms -> 5.5 ms per gradient step on the reference contract).
        Turn it off when debugging: a tracing error is much harder to read than an
        eager one.

    The gradient path runs objective -> P&L -> every hedging decision -> weights. The
    rollout must therefore stay inside the tape; if it drifts outside, ``tape.gradient``
    returns ``None`` silently and nothing trains. ``tests/test_training.py`` pins that.
    """
    generator = make_generator(seed, stream)
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    losses: list[float] = []
    started = time.perf_counter()

    # --- first step runs EAGERLY -------------------------------------------------
    # Two jobs: it creates the weights (Keras builds lazily, so the variable list does
    # not exist until something has flowed through), and it is where the missing-gradient
    # diagnostic can still produce a readable error. Inside a tf.function the same failure
    # surfaces as a tracing error, which is far harder to read -- hence docs/05's rule to
    # compile only once it works.
    spot = world(batch_size, generator)
    with tf.GradientTape() as tape:
        delta = agent.hedge_path(spot, maturity, strike)
        pnl = terminal_pnl(spot, delta, payoff(spot), cost_rate, premium, rate, maturity)
        loss = objective(pnl)

    variables = trainable_variables(agent, objective)
    if not variables:
        raise ValueError("nothing to train: no trainable variables found")

    grads = tape.gradient(loss, variables)
    missing = [v.name for g, v in zip(grads, variables) if g is None]
    if missing:
        raise ValueError(
            f"no gradient reached {missing}. The usual cause is the rollout sitting "
            f"outside the GradientTape, or delta_prev being detached inside it -- "
            f"both leave a model that appears to train and does not learn."
        )
    optimizer.apply_gradients(zip(grads, variables))
    losses.append(float(loss))
    if log_every:
        print(f"  step {0:>5}   loss {float(loss):>12.6f}")

    # --- remaining steps run COMPILED --------------------------------------------
    # Measured 16.9x faster on the reference contract (92.4 ms -> 5.5 ms per step),
    # which is the difference between a 15-hour grid and a 1-hour one.
    #
    # `variables` is captured by CLOSURE, never passed as an argument. Passing tf.Variable
    # objects into a tf.function converts them to symbolic tensors, and the optimiser then
    # fails with "'SymbolicTensor' object has no attribute '_unique_id'" -- which is
    # exactly the unreadable tracing error the eager-first-step rule exists to keep out of
    # the diagnostic path. Defining the closure here, after the variables are resolved, is
    # what makes that safe.
    def one_step(spot: tf.Tensor) -> tf.Tensor:
        with tf.GradientTape() as tape:
            delta = agent.hedge_path(spot, maturity, strike)
            pnl = terminal_pnl(
                spot, delta, payoff(spot), cost_rate, premium, rate, maturity
            )
            loss = objective(pnl)
        optimizer.apply_gradients(zip(tape.gradient(loss, variables), variables))
        return loss

    stepper = (
        tf.function(one_step, reduce_retracing=True) if compile_step else one_step
    )
    for step in range(1, n_gradient_steps):
        loss = stepper(world(batch_size, generator))
        losses.append(float(loss))
        if log_every and (step % log_every == 0 or step == n_gradient_steps - 1):
            print(f"  step {step:>5}   loss {float(loss):>12.6f}")

    return TrainingResult(
        losses=losses,
        n_gradient_steps=n_gradient_steps,
        batch_size=batch_size,
        paths_consumed=n_gradient_steps * batch_size,
        seconds=time.perf_counter() - started,
        seed=seed,
        stream=stream,
        n_variables=len(variables),
        compiled=compile_step,
    )
