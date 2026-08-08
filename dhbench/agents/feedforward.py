"""Feedforward hedging agent — the vanilla Buehler et al. (2019) architecture.

The reference point of the whole grid. A small MLP maps the state at each timestep to a
hedge position:

    delta_i = pi_theta( t_i, S_i, delta_{i-1} )

The **same network** is applied at every timestep — weights are shared across time, with
``t`` passed in as a feature. This is what makes the parameter count independent of
``n_steps``, and it is a detail worth being deliberate about: a per-timestep network is a
different (and much larger) model, and some papers quietly use one.

Stage 2. Depends on ``tf.GradientTape`` and custom training loops — learn those from the
`current TensorFlow docs <https://www.tensorflow.org/guide/basic_training_loops>`_ in
Stage 1.

**Why this cannot use ``model.fit``.** There are no labels. The loss is a risk measure of
terminal P&L, which is a functional of the *entire* simulated path, and gradients flow back
through every hedging decision to every timestep's network call. The training loop
therefore looks like:

    with tf.GradientTape() as tape:
        delta = agent.hedge_path(spot)      # roll forward, collecting positions
        pnl   = terminal_pnl(spot, delta, payoff, cost_rate, premium)
        loss  = objective(pnl)
    grads = tape.gradient(loss, agent.trainable_variables + objective_vars)

That structure — roll forward, reduce to a scalar risk measure, backpropagate through the
whole roll — is the core idea of deep hedging, and everything else is detail.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow import keras

__all__ = ["FeedforwardAgent"]


class FeedforwardAgent(keras.Model):
    """Time-shared MLP hedging policy.

    Args:
        hidden_units: units per hidden layer. Buehler et al. use two or three layers of
            around 32. Keep this in the config — the protocol fixes parameter count across
            agents so that capacity is not a hidden confound.
        activation: hidden activation. ``"relu"`` is standard; ``"tanh"`` sometimes trains
            more stably here because the outputs are bounded.
        output_activation: ``None`` for an unconstrained position, or ``"sigmoid"`` to
            constrain a call hedge to ``[0, 1]``.

            Constraining is tempting and **is a modelling choice, not a free improvement**:
            it hard-codes knowledge of the correct answer for a call, which would be
            cheating in the rung-4 recovery test. Default to unconstrained and let the
            network find ``[0, 1]`` on its own — that it does so is itself evidence the
            setup is right.
    """

    def __init__(
        self,
        hidden_units: tuple[int, ...] = (32, 32),
        activation: str = "relu",
        output_activation: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.hidden_units = hidden_units
        self.activation = activation
        self.output_activation = output_activation
        raise NotImplementedError(
            "Build the Dense stack here: one Dense(u, activation) per entry in "
            "hidden_units, then Dense(1, output_activation). Store them on self so Keras "
            "tracks the weights."
        )

    def call(self, features: tf.Tensor) -> tf.Tensor:
        """Map state features to a hedge position for a single timestep.

        Args:
            features: ``(n_paths, n_features)`` state. Minimally
                ``(t, S, delta_prev)``; see ``docs/03-benchmark-protocol.md`` for the open
                question of what else may be included.

        Returns:
            ``(n_paths,)`` hedge position for this step.

        Note:
            Normalise the inputs. Raw spot around 100 and ``t`` in ``[0, 1]`` differ by two
            orders of magnitude, and the network will train badly. Either normalise here or
            in :meth:`hedge_path`, but do it in exactly one place and record where.
        """
        raise NotImplementedError

    def hedge_path(self, spot: tf.Tensor, maturity: float) -> tf.Tensor:
        """Roll the policy forward along whole paths.

        The bridge between a per-step network and the path-level P&L. Loops over timesteps,
        feeding each step's ``delta_prev`` from the previous step's output, and stacks the
        results.

        Args:
            spot: ``(n_paths, n_steps + 1)`` simulated prices.
            maturity: ``T`` in years, for constructing the time feature.

        Returns:
            ``(n_paths, n_steps)`` hedge positions, ready for
            :func:`dhbench.pnl.terminal_pnl`.

        Warning:
            This loop must stay inside the ``GradientTape``. The whole point is that
            gradients flow back through every step of it — detaching ``delta_prev`` (with
            ``tf.stop_gradient``, or by converting to NumPy mid-loop) silently turns this
            into a myopic one-step policy that trains, converges, and is wrong.

        Note:
            A Python ``for`` over ``n_steps`` is fine and much more readable than
            ``tf.while_loop``. ``n_steps`` is tens; ``n_paths`` is thousands. Wrap the
            whole training step in ``@tf.function`` once it works, not before — tracing
            errors are much harder to read than eager ones.
        """
        raise NotImplementedError
