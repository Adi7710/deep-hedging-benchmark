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
        self.hidden_layers = [
            keras.layers.Dense(units, activation=activation, name=f"hidden_{i}")
            for i, units in enumerate(hidden_units)
        ]
        # Dense(1) with NO activation by default: the network must FIND that a call
        # delta lives in [0, 1] rather than being handed the range. See the class
        # docstring and docs/05-stage-1-2-plan.md section 2.3.
        self.output_layer = keras.layers.Dense(1, activation=output_activation,
                                               name="position")

    def call(self, features: tf.Tensor) -> tf.Tensor:
        """Map state features to a hedge position for a single timestep.

        Args:
            features: ``(n_paths, n_features)`` state. Minimally
                ``(t, S, delta_prev)``; see ``docs/03-benchmark-protocol.md`` for the open
                question of what else may be included.

        Returns:
            ``(n_paths,)`` hedge position for this step.

        Note:
            Normalisation happens in :meth:`hedge_path`, not here, and is done by feature
            choice rather than by a scaling layer: moneyness ``S/K`` and ``tau/T`` are both
            order 1, as is ``delta_prev``. See docs/05-stage-1-2-plan.md section 2.2 --
            batch statistics are disqualified because they couple paths within a batch and
            break bit-reproducibility.
        """
        h = features
        for layer in self.hidden_layers:
            h = layer(h)
        return tf.squeeze(self.output_layer(h), axis=-1)   # (n_paths, 1) -> (n_paths,)

    def hedge_path(
        self, spot: tf.Tensor, maturity: float, strike: float
    ) -> tf.Tensor:
        """Roll the policy forward along whole paths.

        The bridge between a per-step network and the path-level P&L. Loops over timesteps,
        feeding each step's ``delta_prev`` from the previous step's output, and stacks the
        results.

        Args:
            spot: ``(n_paths, n_steps + 1)`` simulated prices.
            maturity: ``T`` in years, for constructing the time feature.
            strike: ``K``. Needed because moneyness is measured against the strike, not
                the initial spot — without it the policy cannot locate the payoff, and
                would be solving a different problem on every contract.

        Returns:
            ``(n_paths, n_steps)`` hedge positions, ready for
            :func:`dhbench.pnl.terminal_pnl`.

        Features, in order, all dimensionless by construction (docs/05 §2.2):

            tau / T      time remaining as a fraction, 1.0 down to 1/n_steps
            S / K        moneyness
            delta_prev   position carried in, delta_{-1} = 0

        Feature choice *is* the normalisation here — no scaling layer, and no batch
        statistics, which would couple paths within a batch and break reproducibility.
        A side benefit: the policy is scale-invariant, so one trained at S0 = 100
        transfers to any spot level.

        Warning:
            This loop must stay inside the ``GradientTape``. The whole point is that
            gradients flow back through every step of it — detaching ``delta_prev`` (with
            ``tf.stop_gradient``, or by converting to NumPy mid-loop) silently turns this
            into a myopic one-step policy that trains, converges, and is wrong.

        Note:
            A Python ``for`` over ``n_steps`` is fine and much more readable than
            ``tf.while_loop``. ``n_steps`` is tens; ``n_paths`` is thousands. Wrap the
            whole training step in ``@tf.function`` once it works, not before — tracing
            errors are much harder to read than eager ones. The Python loop does require
            a statically known ``n_steps``, which is a config value, so we fail loudly
            rather than silently if the time axis is dynamic.
        """
        n_columns = spot.shape[-1]
        if n_columns is None:
            raise ValueError(
                "hedge_path needs a statically known number of timesteps: the rollout is "
                "a Python loop. Give the time axis a concrete size in the input signature "
                "(the batch axis may stay dynamic)."
            )
        n_steps = int(n_columns) - 1
        dtype = spot.dtype

        positions = []
        delta_prev = tf.zeros_like(spot[:, 0])  # delta_{-1} = 0, start flat

        for i in range(n_steps):
            # tau/T = 1 - i/n_steps. Matches the grid in bs_delta.delta_hedge_positions,
            # which is what makes the analytic cross-check in tests exact.
            tau_fraction = tf.constant(1.0 - i / n_steps, dtype=dtype)
            features = tf.stack(
                [
                    tf.ones_like(delta_prev) * tau_fraction,
                    spot[:, i] / tf.constant(strike, dtype=dtype),
                    delta_prev,
                ],
                axis=-1,
            )                                    # (n_paths, 3)
            # Keras autocasts inputs to the model's own dtype policy, so the output may
            # come back narrower than the price path. Cast once, here: positions are
            # multiplied by prices in pnl.py and must share their dtype, and a silent
            # float32/float64 collision would otherwise surface on the NEXT iteration
            # as an opaque Mul error rather than here.
            delta_prev = tf.cast(self(features), dtype)  # carried forward, NOT detached
            positions.append(delta_prev)

        return tf.stack(positions, axis=-1)      # (n_paths, n_steps)
