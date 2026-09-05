"""The learned policy — construction and shape contract.

Tests for the network itself, deliberately separate from tests of training. A wrong shape
or a non-deterministic initialisation would be attributed to the training loop otherwise,
which is the failure mode the staged build in ``docs/05-stage-1-2-plan.md`` exists to
prevent.

The rollout (``hedge_path``) is tested separately, because it has a free and decisive
check available: swap the network for ``Phi(d1)`` and it must reproduce
``delta_hedge_positions``.
"""

from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

from dhbench.agents.feedforward import FeedforwardAgent
from dhbench.baselines.bs_delta import delta_hedge_positions
from dhbench.seeding import make_generator, seed_keras
from dhbench.worlds.gbm import simulate_gbm

N_FEATURES = 3  # (tau/T, S/K, delta_prev) -- see docs/05 section 2.1


def _weights(model) -> np.ndarray:
    return np.concatenate([v.numpy().ravel() for v in model.trainable_variables])


def _build(hidden=(32, 32), seed=0, **kwargs) -> FeedforwardAgent:
    seed_keras(seed, "init")
    agent = FeedforwardAgent(hidden, **kwargs)
    agent(tf.zeros((1, N_FEATURES)))  # force weight creation
    return agent


def test_maps_features_to_one_position_per_path():
    """(n_paths, n_features) -> (n_paths,). One number per path, not (n_paths, 1)."""
    agent = _build()
    out = agent(tf.zeros((7, N_FEATURES)))
    assert out.shape == (7,), f"expected (7,), got {out.shape}"


def test_parameter_count_is_independent_of_the_number_of_timesteps():
    """**Weight sharing across time**, which is the architectural claim being made.

    One network is applied at every timestep, with time passed in as a feature. A
    per-timestep network would be a different and much larger model, and some papers
    quietly use one — so the property is pinned rather than assumed.

    The agent has no ``n_steps`` argument at all; the count depends only on the feature
    width and the hidden sizes.
    """
    agent = _build((32, 32))
    expected = (N_FEATURES * 32 + 32) + (32 * 32 + 32) + (32 * 1 + 1)
    assert agent.count_params() == expected == 1217


def test_initialisation_is_reproducible_from_the_seed():
    """Same replicate index and stream -> identical weights.

    Weight initialisation is the one place this project cannot avoid global random state,
    since Keras initialisers draw from it rather than from an injected generator. It is
    therefore routed through ``seed_keras`` so it still derives from the config's replicate
    index, and rung 6 stays satisfiable.
    """
    assert np.array_equal(_weights(_build(seed=0)), _weights(_build(seed=0)))


def test_different_replicates_give_different_initialisations():
    assert not np.allclose(_weights(_build(seed=0)), _weights(_build(seed=1)))


def test_output_is_unconstrained_by_default():
    """**The rung-4 gate depends on this.**

    A sigmoid output would confine a call hedge to [0, 1], which happens to be the correct
    range — and would hand the network the answer. That the policy *finds* [0, 1] unaided
    is itself evidence the setup is sound, and constraining it forfeits that evidence.

    See docs/05-stage-1-2-plan.md section 2.3, which rejects both the sigmoid and the
    residual parameterisation for the same underlying reason.
    """
    agent = _build()
    wild = tf.random.stateless_normal((4_000, N_FEATURES), seed=(1, 2)) * 5.0
    assert float(tf.reduce_max(tf.abs(agent(wild)))) > 1.0, (
        "output never left [-1, 1]; check that output_activation is None"
    )


def test_sigmoid_output_is_available_but_constrains():
    """The alternative is supported, and pinned as constraining so the choice is visible.

    Recorded rather than hidden: if a later experiment uses it, the effect on the rung-4
    interpretation must be stated in the protocol.
    """
    agent = _build(output_activation="sigmoid")
    wild = tf.random.stateless_normal((2_000, N_FEATURES), seed=(3, 4)) * 5.0
    out = agent(wild)
    assert float(tf.reduce_min(out)) >= 0.0
    assert float(tf.reduce_max(out)) <= 1.0


def test_gradients_reach_every_weight():
    """Every trainable variable must receive gradient from the output.

    A layer that is constructed but never applied, or one detached by a stray conversion,
    would show up here as a None gradient rather than as a mysteriously slow training run.
    """
    agent = _build()
    features = tf.random.stateless_normal((64, N_FEATURES), seed=(5, 6))
    with tf.GradientTape() as tape:
        loss = tf.reduce_mean(agent(features) ** 2)
    grads = tape.gradient(loss, agent.trainable_variables)

    assert len(grads) == len(agent.trainable_variables)
    for grad, var in zip(grads, agent.trainable_variables):
        assert grad is not None, f"no gradient path to {var.name}"
        assert float(tf.reduce_max(tf.abs(grad))) > 0.0, f"zero gradient for {var.name}"


@pytest.mark.parametrize("hidden", [(16,), (32, 32), (64, 64, 64)])
def test_arbitrary_depth_is_supported(hidden):
    """Capacity is a config axis: the protocol fixes parameter count across agents so that
    capacity does not become a hidden confound."""
    agent = _build(hidden)
    assert agent(tf.zeros((3, N_FEATURES))).shape == (3,)
    assert len(agent.hidden_layers) == len(hidden)


# --------------------------------------------------------------------------------------
# The rollout — validated against machinery already verified in rung 2
# --------------------------------------------------------------------------------------

class _AnalyticAgent(FeedforwardAgent):
    """A ``FeedforwardAgent`` whose ``call`` returns Phi(d1) instead of a learned value.

    Exists so the *real* ``hedge_path`` can be exercised with a policy whose output is
    already known, isolating the loop from the learning. Everything except ``call`` is
    the production code path.
    """

    def __init__(self, maturity: float, rate: float, sigma: float):
        # float64 so the comparison against the SciPy reference can be tight rather than
        # limited by the model's default single precision.
        super().__init__(hidden_units=(1,), dtype="float64")
        self._maturity, self._rate, self._sigma = maturity, rate, sigma

    def call(self, features: tf.Tensor) -> tf.Tensor:
        tau_fraction, moneyness, _ = tf.unstack(features, axis=-1)
        tau = tf.maximum(tau_fraction * self._maturity, 1e-12)
        vol_sqrt_tau = self._sigma * tf.sqrt(tau)
        d1 = (
            tf.math.log(moneyness)
            + (self._rate + 0.5 * self._sigma**2) * tau
        ) / vol_sqrt_tau
        return 0.5 * (1.0 + tf.math.erf(d1 / tf.sqrt(tf.constant(2.0, features.dtype))))


def test_rollout_reproduces_the_analytic_delta_hedge():
    """**The free decisive test of the rollout** (docs/05 §3.1).

    Swap the network for Phi(d1) and ``hedge_path`` must reproduce
    ``delta_hedge_positions`` -- the same object computed by an entirely different route,
    already verified in rung 2. This validates the indexing, the time-feature grid, the
    delta_prev carry and the off-by-one at maturity, all before any learning exists.

    If it fails, the fault is in the rollout and nowhere else. That localisation is the
    whole reason to run it.
    """
    maturity, rate, sigma, strike = 1.0, 0.0, 0.2, 100.0
    spot = simulate_gbm(
        256, 24, 100.0, rate, sigma, maturity,
        make_generator(0, "test"), dtype=tf.float64,
    )

    rolled = _AnalyticAgent(maturity, rate, sigma).hedge_path(spot, maturity, strike)
    reference = delta_hedge_positions(spot.numpy(), strike, maturity, rate, sigma)

    assert rolled.shape == reference.shape == (256, 24)
    np.testing.assert_allclose(rolled.numpy(), reference, atol=1e-9)


def test_rollout_shape_and_flat_start():
    """(n_paths, n_steps+1) in -> (n_paths, n_steps) out, one decision per gap."""
    agent = _build()
    spot = simulate_gbm(16, 10, 100.0, 0.0, 0.2, 1.0, make_generator(1, "test"))
    assert agent.hedge_path(spot, 1.0, 100.0).shape == (16, 10)


def test_gradients_flow_back_through_every_step_of_the_rollout():
    """**The failure mode that silently produces a myopic policy.**

    Detaching delta_prev -- via stop_gradient or a NumPy round trip inside the loop --
    leaves a model that trains and converges and is wrong, because the policy can no
    longer see that today's trade changes tomorrow's cost.

    Gradient must reach the weights, and the rollout must be inside the tape for it to
    exist at all.
    """
    agent = _build()
    spot = simulate_gbm(64, 12, 100.0, 0.0, 0.2, 1.0, make_generator(2, "test"))

    with tf.GradientTape() as tape:
        delta = agent.hedge_path(spot, 1.0, 100.0)
        loss = tf.reduce_mean(delta[:, -1] ** 2)      # depends on the LAST step only
    grads = tape.gradient(loss, agent.trainable_variables)

    for grad, var in zip(grads, agent.trainable_variables):
        assert grad is not None, f"no gradient path to {var.name}"
        assert float(tf.reduce_max(tf.abs(grad))) > 0.0, f"zero gradient for {var.name}"


def test_the_last_step_depends_on_the_first_decision():
    """Backpropagation through time: step 0's weights must influence step n-1's output.

    A myopic policy would show zero sensitivity here. This is the property that makes the
    problem a control problem rather than a sequence of independent regressions.
    """
    agent = _build()
    spot = simulate_gbm(64, 12, 100.0, 0.0, 0.2, 1.0, make_generator(3, "test"))

    with tf.GradientTape() as tape:
        delta = agent.hedge_path(spot, 1.0, 100.0)
        last_only = tf.reduce_mean(delta[:, -1])
    grad = tape.gradient(last_only, agent.trainable_variables[0])
    assert float(tf.reduce_max(tf.abs(grad))) > 0.0


def test_rollout_rejects_a_dynamic_time_axis():
    """The Python loop needs a static n_steps. Fail loudly rather than at trace time."""
    agent = _build()

    @tf.function(input_signature=[tf.TensorSpec([None, None], tf.float32)])
    def compiled(spot):
        return agent.hedge_path(spot, 1.0, 100.0)

    with pytest.raises(ValueError, match="statically known"):
        compiled(tf.ones((4, 6)))
