"""Black-Scholes reference implementation.

**These tests should pass immediately** — ``dhbench/baselines/bs_delta.py`` is written in
full because it is reference material that everything else is checked against.

If any of these fail, the environment is broken (bad SciPy, wrong NumPy ABI) rather than
the maths. Run this file first to confirm the install before debugging anything else.
"""

from __future__ import annotations

import numpy as np
import pytest

from dhbench.baselines.bs_delta import (
    bs_call_price,
    bs_delta,
    bs_gamma,
    d1_d2,
    delta_hedge_positions,
)

S0 = 100.0
STRIKE = 100.0
MATURITY = 1.0
RATE = 0.0
SIGMA = 0.2


# --------------------------------------------------------------------------------------
# Pinned values
# --------------------------------------------------------------------------------------

def test_atm_call_price():
    """ATM call, 1y, 20% vol, zero rate."""
    assert float(bs_call_price(100.0, 100.0, 1.0, 0.0, 0.2)) == pytest.approx(7.9656, abs=1e-4)


def test_atm_delta_is_slightly_above_half():
    """ATM delta exceeds 0.5 because ``d1`` carries the ``+sigma^2/2`` term."""
    delta = float(bs_delta(100.0, 100.0, 1.0, 0.0, 0.2))
    assert delta == pytest.approx(0.5398, abs=1e-4)
    assert delta > 0.5


def test_d1_d2_relationship():
    """``d2 = d1 - sigma*sqrt(tau)`` by definition."""
    d1, d2 = d1_d2(105.0, 100.0, 0.5, 0.03, 0.25)
    assert float(d1 - d2) == pytest.approx(0.25 * np.sqrt(0.5))


# --------------------------------------------------------------------------------------
# Limiting behaviour
# --------------------------------------------------------------------------------------

def test_deep_itm_delta_approaches_one():
    """Deep in the money, the call behaves like the stock."""
    assert float(bs_delta(1000.0, 100.0, 1.0, 0.0, 0.2)) == pytest.approx(1.0, abs=1e-6)


def test_deep_otm_delta_approaches_zero():
    """Deep out of the money, the call is worthless and unhedged."""
    assert float(bs_delta(1.0, 100.0, 1.0, 0.0, 0.2)) == pytest.approx(0.0, abs=1e-6)


def test_at_expiry_delta_is_a_step_function():
    """``tau -> 0``: delta becomes the indicator of finishing in the money.

    Exercises the ``tau`` clipping in :func:`d1_d2`. Without it this produces ``nan``,
    which then propagates silently through an entire hedging path.
    """
    assert float(bs_delta(110.0, 100.0, 0.0, 0.0, 0.2)) == pytest.approx(1.0, abs=1e-6)
    assert float(bs_delta(90.0, 100.0, 0.0, 0.0, 0.2)) == pytest.approx(0.0, abs=1e-6)


def test_no_nan_at_expiry():
    """Explicit guard: the singular case must not produce ``nan``."""
    for fn in (bs_call_price, bs_delta, bs_gamma):
        assert np.isfinite(float(fn(100.0, 100.0, 0.0, 0.0, 0.2))), fn.__name__


def test_price_bounded_by_no_arbitrage():
    """``max(S - K e^{-r tau}, 0) <= C <= S``."""
    spots = np.array([50.0, 80.0, 100.0, 130.0, 200.0])
    prices = bs_call_price(spots, STRIKE, MATURITY, 0.03, SIGMA)
    lower = np.maximum(spots - STRIKE * np.exp(-0.03 * MATURITY), 0.0)
    assert np.all(prices >= lower - 1e-9)
    assert np.all(prices <= spots + 1e-9)


def test_delta_is_monotone_in_spot():
    """Delta increases with spot — the call is convex in the underlying."""
    spots = np.linspace(50.0, 150.0, 100)
    deltas = bs_delta(spots, STRIKE, MATURITY, RATE, SIGMA)
    assert np.all(np.diff(deltas) > 0)


def test_gamma_peaks_at_the_analytic_location():
    """Gamma peaks slightly **below** the strike, at ``K exp(-(r + 3 sigma^2 / 2) tau)``.

    Maximising ``phi(d1) / (S sigma sqrt(tau))`` over ``S`` gives ``d1 = -sigma sqrt(tau)``,
    hence that location — 94.18 for the reference parameters, not 100.

    Worth pinning precisely rather than asserting "near the money": the Whalley-Wilmott
    band half-width scales as ``Gamma^(2/3)``, so the band is widest *here*, and a learned
    band should peak in the same place. That comparison is rung 5, and it only means
    something if the reference location is exact.
    """
    spots = np.linspace(60.0, 160.0, 1001)
    gammas = bs_gamma(spots, STRIKE, MATURITY, RATE, SIGMA)

    analytic_peak = STRIKE * np.exp(-(RATE + 1.5 * SIGMA**2) * MATURITY)
    assert analytic_peak == pytest.approx(94.176, abs=1e-3)

    empirical_peak = spots[int(np.argmax(gammas))]
    assert empirical_peak == pytest.approx(analytic_peak, abs=0.2)


def test_gamma_is_non_negative():
    """A long call has non-negative gamma everywhere."""
    spots = np.linspace(10.0, 400.0, 200)
    assert np.all(bs_gamma(spots, STRIKE, MATURITY, RATE, SIGMA) >= 0.0)


# --------------------------------------------------------------------------------------
# delta_hedge_positions shape contract
# --------------------------------------------------------------------------------------

def test_delta_hedge_positions_shape():
    """``(n_paths, n_steps + 1)`` in, ``(n_paths, n_steps)`` out.

    The output has one fewer column because no decision is made at maturity. Getting this
    wrong is an off-by-one that shows up as a small persistent hedging loss.
    """
    paths = np.full((7, 21), S0)
    positions = delta_hedge_positions(paths, STRIKE, MATURITY, RATE, SIGMA)
    assert positions.shape == (7, 20)


def test_delta_hedge_positions_in_unit_interval():
    """Call deltas live in ``[0, 1]``."""
    rng = np.random.default_rng(0)
    paths = S0 * np.exp(np.cumsum(rng.normal(0, 0.05, (50, 31)), axis=1))
    positions = delta_hedge_positions(paths, STRIKE, MATURITY, RATE, SIGMA)
    assert np.all(positions >= 0.0) and np.all(positions <= 1.0)


def test_delta_hedge_first_position_is_the_t0_delta():
    """The first position is chosen at ``t=0`` from ``S_0`` with full time remaining."""
    paths = np.full((3, 11), S0)
    positions = delta_hedge_positions(paths, STRIKE, MATURITY, RATE, SIGMA)
    expected = float(bs_delta(S0, STRIKE, MATURITY, RATE, SIGMA))
    assert positions[0, 0] == pytest.approx(expected)
