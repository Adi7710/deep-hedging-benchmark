"""Seeding — the foundation of the reproducibility claim.

Bit-reproducibility from config + seed is a headline contribution, and "multiple seeds
with dispersion reported" is one of the controls §6 criticises other papers for omitting.
Both rest on replicate seeds producing genuinely independent streams.

They do not, if the replicate index reaches TensorFlow unhashed. ``from_seed(k)`` writes
``k`` into the Philox counter with the key fixed at zero, so consecutive seeds produce the
*same stream shifted by four draws*. Replicates built that way are copies of one sample.
Their dispersion collapses to ~0.41x the true sampling error, and an ordinary +0.85 SE
fluctuation gets reported as +9.3 SE. That is false significance, not bias -- and false
significance is a worse failure than any this benchmark criticises, because it makes noise
look like a result.

The dispersion test below is the one that matters: it would fail on raw ``from_seed``.
``test_consecutive_integer_seeds_are_shifted_copies`` pins the mechanism itself, so a
change in TensorFlow's seeding shows up here rather than silently in the paper.
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

    Under ``from_seed(0..19)`` the mean lands +0.025 high, which reads as +9.3 SE with
    20/20 replicates above truth. It is not a bias: the replicates are shifted copies of
    one sample that happened to sit +0.85 proper SE high, and their collapsed dispersion
    inflates that into apparent significance. Independent replicates straddle the truth.
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
        f"= {z:.1f} SE. If this is large while each replicate looks individually "
        f"plausible, the replicates are probably not independent -- check whether "
        f"their streams overlap before suspecting a bias."
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


# --------------------------------------------------------------------------------------
# The mechanism -- pinned so the paper's claim fails loudly if TensorFlow changes
# --------------------------------------------------------------------------------------

def _shift_between(a: np.ndarray, b: np.ndarray, max_lag: int = 64) -> int | None:
    """Smallest lag k with b[:m] == a[k:k+m], or None if no lag up to max_lag matches."""
    m = len(a) - max_lag
    for k in range(max_lag):
        if np.array_equal(a[k:k + m], b[:m]):
            return k
    return None


def test_integer_seed_is_written_into_the_philox_counter():
    """``from_seed(k)`` sets state [k, 0, 0]: the counter moves, the key never does.

    This is the whole mechanism. Pinned against TensorFlow's current behaviour; if a
    future release changes seed handling, this fails and the claim in the paper must be
    re-checked rather than silently going stale.
    """
    for k in (0, 1, 2, 7):
        state = tf.random.Generator.from_seed(k).state.numpy().tolist()
        assert state == [k, 0, 0], f"from_seed({k}).state == {state}"


def test_consecutive_integer_seeds_are_shifted_copies():
    """**The defect.** ``from_seed(k)`` reproduces ``from_seed(0)`` offset by ``4k`` draws.

    Philox yields four outputs per counter increment, so bumping the counter by one skips
    exactly one block. Replicates built from seeds 0, 1, 2, ... are therefore not samples
    at all -- they are one sample read from slightly different starting points.
    """
    base = tf.random.Generator.from_seed(0).normal((4_000,)).numpy()
    for k in (1, 2, 3):
        other = tf.random.Generator.from_seed(k).normal((4_000,)).numpy()
        assert _shift_between(base, other) == 4 * k


def test_lag_zero_correlation_cannot_see_the_overlap():
    """Why this survives review: the obvious independence check passes.

    Two streams that share every single draw, offset by four, have lag-0 correlation near
    zero -- shifting an i.i.d. sequence decorrelates it pointwise while leaving the *set*
    of numbers identical. A reviewer checking corrcoef(stream_0, stream_1) sees
    independence. Only the dispersion of a downstream estimator, or a lag search, reveals it.
    """
    a = tf.random.Generator.from_seed(0).normal((200_000,)).numpy()
    b = tf.random.Generator.from_seed(1).normal((200_000,)).numpy()

    assert abs(np.corrcoef(a, b)[0, 1]) < 0.02, "lag-0 check looks independent"
    assert np.array_equal(a[4:], b[:-4]), "...while every draw is shared"


def test_hashed_replicates_are_not_shifted_copies():
    """The fix: SHA-256 scatters counters across 2^63, so no small offset exists."""
    a = make_generator(0).normal((4_000,)).numpy()
    b = make_generator(1).normal((4_000,)).numpy()
    assert _shift_between(a, b) is None
    assert _shift_between(b, a) is None


def test_a_list_seed_lands_in_the_key():
    """The inconsistency the paper reports, and the one-character alternative fix.

    A list seed is left-padded, so ``from_seed([k])`` sets state [0, 0, k]: k is the KEY,
    and distinct keys give distinct streams. TensorFlow's ``_make_1d_state`` pads on the
    left precisely so that "a small seed" is not "used as the 'counter' while the 'key' is
    always zero". An int seed is first chopped into ``state_size`` words, arrives full
    length, and is never padded -- so the guard never fires for the most natural call.
    """
    for k in (1, 2, 7):
        assert tf.random.Generator.from_seed([k]).state.numpy().tolist() == [0, 0, k]
    a = tf.random.Generator.from_seed([0]).normal((4_000,)).numpy()
    b = tf.random.Generator.from_seed([1]).normal((4_000,)).numpy()
    assert _shift_between(a, b) is None
    assert _shift_between(b, a) is None
