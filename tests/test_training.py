"""The training loop.

Step 5 of ``docs/05-stage-1-2-plan.md``. By construction everything beneath this is
already verified — the simulator (rung 1), the accounting (rung 2), the risk measures,
the network, and the rollout against its analytic cross-check — so a failure here
localises to the loop itself.

Deliberately small and fast. These check that training *works*, not that it produces a
good policy; whether the learned hedge recovers ``Phi(d1)`` is rung 4 and lives in
``test_rungs_3_to_6.py``.
"""

from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

from dhbench.agents.feedforward import FeedforwardAgent
from dhbench.objectives.cvar import CVaRRisk
from dhbench.objectives.entropic import EntropicRisk
from dhbench.objectives.mean_variance import MeanVarianceRisk
from dhbench.seeding import seed_keras
from dhbench.training import train, trainable_variables
from dhbench.worlds.gbm import simulate_gbm

S0 = STRIKE = 100.0
MATURITY, RATE, SIGMA, N_STEPS = 1.0, 0.0, 0.2, 8


def world(n_paths, generator):
    return simulate_gbm(n_paths, N_STEPS, S0, RATE, SIGMA, MATURITY, generator)


def payoff(spot):
    return tf.maximum(spot[:, -1] - STRIKE, 0.0)


def _agent(seed=0, hidden=(16, 16)):
    seed_keras(seed, "init")
    return FeedforwardAgent(hidden)


def _train(agent=None, objective=None, steps=40, **kwargs):
    return train(
        agent or _agent(),
        objective or EntropicRisk(1.0),
        world,
        payoff,
        maturity=MATURITY,
        strike=STRIKE,
        batch_size=128,
        n_gradient_steps=steps,
        **kwargs,
    )


# --------------------------------------------------------------------------------------
# It trains
# --------------------------------------------------------------------------------------

def test_loss_decreases():
    """The weakest useful check, and the first thing to fail if the tape is broken."""
    result = _train(steps=60)
    assert result.improved, f"loss did not fall: {result.losses[:3]} -> {result.losses[-3:]}"


def test_result_records_the_budget_in_paths_not_epochs():
    """Fresh paths every batch means "epoch" is meaningless (docs/05 §2.4).

    The reportable training budget is gradient steps and paths consumed, and the protocol
    controls equal-budget comparisons on those.
    """
    result = _train(steps=25)
    assert result["paths_consumed"] == 25 * 128
    assert result["n_gradient_steps"] == 25
    assert result["seconds"] > 0.0


# --------------------------------------------------------------------------------------
# Reproducibility — rung 6 depends on this
# --------------------------------------------------------------------------------------

def test_same_seed_reproduces_the_loss_history():
    """Same replicate index, same weights, same paths, same trajectory.

    Not `approx` — the reproducibility contract is bit-identical. If this becomes flaky,
    something is drawing from global random state instead of the derived streams.
    """
    a = _train(agent=_agent(seed=0), steps=30)
    b = _train(agent=_agent(seed=0), steps=30)
    np.testing.assert_array_equal(a.losses, b.losses)


def test_different_replicates_give_different_trajectories():
    a = _train(agent=_agent(seed=0), steps=30)
    b = _train(agent=_agent(seed=1), steps=30, seed=1)
    assert not np.allclose(a.losses, b.losses)


# --------------------------------------------------------------------------------------
# The objective's own variables — the classic CVaR bug
# --------------------------------------------------------------------------------------

def test_trainable_variables_includes_the_objectives_own():
    """CVaR's Rockafellar-Uryasev ``w`` must reach the optimiser.

    Omitting it is the classic CVaR deep hedging bug: training runs, the loss falls a
    little, then flatlines — and it reads as a learning-rate problem rather than a
    formulation one.
    """
    agent = _agent()
    agent(tf.zeros((1, 3)))

    assert len(trainable_variables(agent, EntropicRisk(1.0))) == len(
        agent.trainable_variables
    )
    assert len(trainable_variables(agent, MeanVarianceRisk(1.0))) == len(
        agent.trainable_variables
    )

    cvar = CVaRRisk(0.95)
    combined = trainable_variables(agent, cvar)
    assert len(combined) == len(agent.trainable_variables) + 1
    assert any(v is cvar.w for v in combined)


def test_cvar_auxiliary_variable_actually_moves_during_training():
    """It must be optimised jointly, not left at its initial value.

    ``w`` converges toward the value-at-risk, so a ``w`` that never moves is a free
    diagnostic that it was excluded from the optimiser.
    """
    objective = CVaRRisk(0.95)
    before = float(objective.w)
    _train(objective=objective, steps=40)
    assert abs(float(objective.w) - before) > 1e-4, "w never moved; it is not being trained"


@pytest.mark.parametrize(
    "objective",
    [EntropicRisk(1.0), CVaRRisk(0.95), MeanVarianceRisk(1.0)],
    ids=["entropic", "cvar", "meanvar"],
)
def test_every_objective_trains(objective):
    assert _train(objective=objective, steps=50).improved


# --------------------------------------------------------------------------------------
# The failure this loop exists to make loud
# --------------------------------------------------------------------------------------

def test_a_detached_rollout_raises_instead_of_silently_not_learning():
    """**The failure mode that costs people days.**

    Detaching the rollout — ``stop_gradient``, or a NumPy round trip mid-loop — leaves a
    model that runs, converges, and has learned nothing, because no gradient ever reached
    the weights. ``train`` checks for ``None`` gradients on the first (eager) step and
    fails with a message naming the likely cause.
    """
    class _DetachedAgent(FeedforwardAgent):
        def hedge_path(self, spot, maturity, strike):
            return tf.stop_gradient(super().hedge_path(spot, maturity, strike))

    seed_keras(0, "init")
    with pytest.raises(ValueError, match="no gradient reached"):
        _train(agent=_DetachedAgent((16, 16)), steps=3)


# --------------------------------------------------------------------------------------
# Compilation
# --------------------------------------------------------------------------------------

def test_compiled_and_eager_training_agree():
    """``tf.function`` is a 16.9x speedup and must not change the answer.

    Compiled by default; the eager path stays available because a tracing error is far
    harder to read than an eager one, and that is where the missing-gradient diagnostic
    has to remain legible.
    """
    a = _train(agent=_agent(seed=0), steps=25, compile_step=True)
    b = _train(agent=_agent(seed=0), steps=25, compile_step=False)
    np.testing.assert_allclose(a.losses, b.losses, rtol=1e-4, atol=1e-4)


def test_training_reports_whether_it_compiled():
    """Recorded in the result so a run's provenance includes it."""
    assert _train(steps=5, compile_step=True)["compiled"] is True
    assert _train(steps=5, compile_step=False)["compiled"] is False
