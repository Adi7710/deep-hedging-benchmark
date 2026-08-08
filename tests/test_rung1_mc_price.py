"""Rung 1 — Monte Carlo price must match closed-form Black-Scholes.

The first gate. Verifies the GBM simulator produces the right distribution: if the
simulated terminal prices are wrong, every hedging result downstream is wrong too, and no
amount of network debugging will find it.

Nothing here involves a neural network. Get this passing before writing any TensorFlow
training code.
"""

from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

from dhbench.baselines.bs_delta import bs_call_price
from dhbench.worlds.gbm import simulate_gbm

# Reference parameters. Chosen so the analytic price is a round-ish number that is easy to
# eyeball: bs_call_price(100, 100, 1.0, 0.0, 0.2) = 7.9656
S0 = 100.0
STRIKE = 100.0
MATURITY = 1.0
RATE = 0.0
SIGMA = 0.2
N_PATHS = 500_000
N_STEPS = 50


def test_bs_reference_price_is_stable():
    """The analytic reference itself, pinned.

    Guards against an accidental edit to bs_delta.py silently moving the target that every
    other test in this file is measured against.
    """
    price = float(bs_call_price(S0, STRIKE, MATURITY, RATE, SIGMA))
    assert price == pytest.approx(7.9656, abs=1e-4)


def test_mc_price_matches_black_scholes():
    """Monte Carlo European call price ≈ closed form, within Monte Carlo error.

    The tolerance is three standard errors of the MC estimate, computed from the sample
    rather than hard-coded. A fixed tolerance would either be too loose to catch a real bug
    or too tight and flaky as ``N_PATHS`` changes.
    """
    generator = tf.random.Generator.from_seed(42)
    paths = simulate_gbm(
        n_paths=N_PATHS,
        n_steps=N_STEPS,
        s0=S0,
        mu=RATE,
        sigma=SIGMA,
        maturity=MATURITY,
        generator=generator,
    )

    terminal = paths[:, -1].numpy()
    payoffs = np.maximum(terminal - STRIKE, 0.0)

    mc_price = payoffs.mean() * np.exp(-RATE * MATURITY)
    std_error = payoffs.std(ddof=1) / np.sqrt(N_PATHS)
    analytic = float(bs_call_price(S0, STRIKE, MATURITY, RATE, SIGMA))

    assert mc_price == pytest.approx(analytic, abs=3.0 * std_error), (
        f"MC price {mc_price:.4f} vs analytic {analytic:.4f}, "
        f"3 s.e. = {3 * std_error:.4f}"
    )


def test_paths_start_at_s0_exactly():
    """Column 0 is ``s0`` exactly, on every path.

    Not pedantry: if the first column comes from a floating-point round trip rather than
    being prepended, an off-by-one has crept into the path construction, and it will show
    up later as a small, baffling hedging bias.
    """
    generator = tf.random.Generator.from_seed(0)
    paths = simulate_gbm(
        n_paths=100,
        n_steps=10,
        s0=S0,
        mu=0.0,
        sigma=SIGMA,
        maturity=MATURITY,
        generator=generator,
    )
    np.testing.assert_array_equal(paths[:, 0].numpy(), np.full(100, S0, dtype=np.float32))


def test_output_shape():
    """``(n_paths, n_steps + 1)`` — the shape contract the whole package relies on."""
    generator = tf.random.Generator.from_seed(0)
    paths = simulate_gbm(
        n_paths=37,
        n_steps=11,
        s0=S0,
        mu=0.0,
        sigma=SIGMA,
        maturity=MATURITY,
        generator=generator,
    )
    assert paths.shape == (37, 12)


def test_paths_are_strictly_positive():
    """GBM cannot go negative. Catches a build in price space rather than log space."""
    generator = tf.random.Generator.from_seed(7)
    paths = simulate_gbm(
        n_paths=10_000,
        n_steps=100,
        s0=S0,
        mu=0.0,
        sigma=0.8,  # high vol: the regime where a naive implementation breaks
        maturity=MATURITY,
        generator=generator,
    )
    assert bool(tf.reduce_all(paths > 0.0))


def test_same_seed_reproduces_exactly():
    """Bit-reproducibility from an explicit generator.

    This is a headline claim of the benchmark, not a nicety. If it fails, no published
    number from this repository can be trusted.
    """
    kwargs = dict(
        n_paths=1_000,
        n_steps=20,
        s0=S0,
        mu=0.0,
        sigma=SIGMA,
        maturity=MATURITY,
    )
    a = simulate_gbm(generator=tf.random.Generator.from_seed(123), **kwargs)
    b = simulate_gbm(generator=tf.random.Generator.from_seed(123), **kwargs)
    np.testing.assert_array_equal(a.numpy(), b.numpy())


def test_different_seeds_differ():
    """The trivial converse — guards against a generator that is silently ignored."""
    kwargs = dict(
        n_paths=1_000,
        n_steps=20,
        s0=S0,
        mu=0.0,
        sigma=SIGMA,
        maturity=MATURITY,
    )
    a = simulate_gbm(generator=tf.random.Generator.from_seed(1), **kwargs)
    b = simulate_gbm(generator=tf.random.Generator.from_seed(2), **kwargs)
    assert not np.allclose(a.numpy(), b.numpy())
