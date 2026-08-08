"""Regime-switching GBM — the out-of-distribution *test* world.

Markov-modulated GBM: a hidden state ``k_t`` follows a discrete-time Markov chain, and the
drift and volatility take regime-dependent values ``(mu_k, sigma_k)``. The canonical setup
is two regimes — calm and stressed — with the stressed regime having higher volatility and
a low probability of persisting.

This world exists for **§8, the regime-fragility result**. Agents are trained on Heston and
evaluated here. The 2026 paper *What Does Deep Hedging Actually Learn?* reports that
learned policies fail when conditions shift beyond the training distribution; §8 quantifies
that across the whole method set rather than one model.

Design constraint worth stating explicitly: the regime path is generated but **not exposed
to the agent**. That is the point — the agent must cope with a shift it cannot observe.
Handing it the regime label would test something else entirely, and would be an easy
mistake to make while debugging.
"""

from __future__ import annotations

import tensorflow as tf

__all__ = ["simulate_regime_switching"]


def simulate_regime_switching(
    n_paths: int,
    n_steps: int,
    s0: float,
    mus: tf.Tensor,
    sigmas: tf.Tensor,
    transition_matrix: tf.Tensor,
    initial_regime_probs: tf.Tensor,
    maturity: float,
    generator: tf.random.Generator,
    dtype: tf.DType = tf.float32,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Simulate Markov-modulated GBM paths.

    Args:
        n_paths: number of independent paths.
        n_steps: number of time steps.
        s0: initial spot.
        mus: ``(n_regimes,)`` drift per regime.
        sigmas: ``(n_regimes,)`` volatility per regime.
        transition_matrix: ``(n_regimes, n_regimes)`` row-stochastic, per time step.
            **Per step, not per year** — rescale if you calibrate in annual terms, and
            record which convention the config uses.
        initial_regime_probs: ``(n_regimes,)`` distribution of the regime at ``t=0``.
        maturity: ``T`` in years.
        generator: explicit ``tf.random.Generator``.
        dtype: float dtype.

    Returns:
        ``(spot, regimes)`` — spot is ``(n_paths, n_steps + 1)``, regimes is
        ``(n_paths, n_steps + 1)`` of integer regime indices.

        The regime path is returned **for analysis only**. Never pass it to an agent as a
        state feature; that would defeat the purpose of the experiment.

    Implementation note:
        Sampling the chain is inherently sequential in time, but vectorised across paths.
        Either use ``tf.while_loop`` / ``tf.scan``, or accept a Python loop over
        ``n_steps`` — ``n_steps`` is small (tens), ``n_paths`` is large (thousands), so the
        Python loop is fine and much easier to read. Prefer the readable version until
        profiling says otherwise.
    """
    raise NotImplementedError
