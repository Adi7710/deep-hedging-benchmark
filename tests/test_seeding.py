"""Seeding — the foundation of the reproducibility claim.

Bit-reproducibility from config + seed is a headline contribution, and "multiple seeds
with dispersion reported" is one of the controls §6 criticises other papers for omitting.
Both rest on replicate seeds producing genuinely independent streams.

They do not, if the replicate index reaches TensorFlow unhashed. ``from_seed(0..19)``
priced an ATM call at +0.025 (9.3 SE) above truth with 20/20 replicates high, and a
cross-seed dispersion 2.5x narrower than the analytic standard error. Narrow error bars
turn inconclusive comparisons into significant ones, which is a worse failure than the
ones this benchmark is criticising.

The dispersion test below is the one that matters: it would fail on raw ``from_seed``.
"""

from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

from dhbench.baselines.bs_delta import bs_call_price
from dhbench.seeding import derive_seed, make_generator, seed_keras
from dhbench.worlds.gbm import simulate_gbm

S0 = STRIKE = 100.0
MATURITY, RATE, SIGMA = 1.0, 0.0, 0.2


def test_derive_seed_is_deterministic():
    """Same inputs, same seed — across processes and platforms.

    Uses SHA-256 rather than ``hash()``, which is salted per process unless
    PYTHONHASHSEED is pinned and would silently break reproducibility between runs.
    """
    assert derive_seed(7, "train") == derive_seed(7, "train")
    assert derive_seed(0) == derive_seed(0)


def test_streams_are_disjoint():
    """Same replicate, different stream label -> independent seeds.

    This is what makes train/eval splits disjoint by construction rather than by an
    ad-hoc numeric offset that someone can forget to apply.
    """
    assert derive_seed(0, "train") != derive_seed(0, "eval")
    assert derive_seed(0, "train") != derive_seed(0, "init")
    assert len({derive_seed(k) for k in range(64)}) == 64


def test_consecutive_replicates_give_far_apart_seeds():
    """Adjacent replicate indices must not give adjacent seeds — that is the bug."""
    seeds = [derive_seed(k) for k in range(8)]
    assert all(abs(b - a) > 2**32 for a, b in zip(seeds, seeds[1:]))


def test_cross_seed_dispersion_matches_analytic_standard_error():
    """**The test that catches the real bug.**

    Monte Carlo theory fixes the spread of an estimator across independent replicates:
    it must be the payoff standard deviation over sqrt(n_paths). If replicate streams are
    correlated the observed spread comes out too small, and every error bar in the results
    grid is over-confident.

    Acceptance: observed cross-seed dispersion within [0.6, 1.7] of analytic. Raw
    ``tf.random.Generator.from_seed(0..k)`` scores about 0.41 here and fails.
    """
    n_paths, n_seeds = 60_000, 16

    prices = []
    for k in range(n_seeds):
        paths = simulate_gbm(
            n_paths, 25, S0, RATE, SIGMA, MATURITY, make_generator(k, "test")
        )
        payoff = tf.maximum(paths[:, -1] - STRIKE, 0.0)
        prices.append(float(tf.reduce_mean(payoff)))
        if k == 0:
            analytic_se = float(tf.math.reduce_std(payoff)) / np.sqrt(n_paths)

    observed = float(np.std(prices, ddof=1))
    ratio = observed / analytic_se
    assert 0.6 < ratio < 1.7, (
        f"Cross-seed dispersion {observed:.5f} vs analytic SE {analytic_se:.5f} "
        f"(ratio {ratio:.2f}). Far below 1 means replicate streams are correlated and "
        f"every reported error bar is too narrow. Are seeds being hashed?"
    )


def test_replicates_are_unbiased_against_the_closed_form():
    """The mean across replicates must sit within Monte Carlo error of the true price.

    ``from_seed(0..19)`` lands +0.025 high at 9.3 SE with 20/20 replicates above truth,
    because its samples carry a systematically high realised volatility.
    """
    n_paths, n_seeds = 60_000, 16
    truth = float(bs_call_price(S0, STRIKE, MATURITY, RATE, SIGMA))

    prices = np.array([
        float(tf.reduce_mean(tf.maximum(
            simulate_gbm(n_paths, 25, S0, RATE, SIGMA, MATURITY,
                         make_generator(k, "unbiased"))[:, -1] - STRIKE, 0.0)))
        for k in range(n_seeds)
    ])

    sem = prices.std(ddof=1) / np.sqrt(n_seeds)
    z = (prices.mean() - truth) / sem
    assert abs(z) < 4.0, (
        f"Mean replicate price {prices.mean():.5f} vs closed form {truth:.5f} "
        f"= {z:.1f} SE. A systematic offset means the replicate streams are not "
        f"sampling the intended distribution."
    )
    n_above = int((prices > truth).sum())
    assert 2 <= n_above <= n_seeds - 2, (
        f"{n_above}/{n_seeds} replicates above truth — replicates should straddle it."
    )


def test_narrowed_seeds_stay_on_the_same_stream():
    """``bits`` truncates the digest; it does not move you to a different stream."""
    assert derive_seed(3, "init", bits=32) == derive_seed(3, "init", bits=63) & 0xFFFFFFFF
    assert derive_seed(3, "init", bits=32) < 2**32


def test_seed_keras_fits_numpys_range():
    """**Found by trying it.**

    ``keras.utils.set_random_seed`` forwards to ``numpy.random.seed``, which rejects any
    seed >= 2**32. The 63-bit default from :func:`derive_seed` raises ValueError there, so
    weight initialisation is routed through :func:`seed_keras`, which narrows to 32 bits.
    """
    used = seed_keras(0, "init")
    assert 0 <= used < 2**32
    assert used == derive_seed(0, "init", bits=32)


def test_seed_keras_is_deterministic_and_stream_separated():
    assert seed_keras(4, "init") == seed_keras(4, "init")
    assert seed_keras(4, "init") != seed_keras(4, "train")


def test_derive_seed_rejects_impossible_widths():
    for bad in (0, 65, -1):
        with pytest.raises(ValueError):
            derive_seed(0, "x", bits=bad)
