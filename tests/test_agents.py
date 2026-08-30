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
from dhbench.seeding import seed_keras

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
