"""The arithmetic behind the market calibration, pinned on synthetic data.

``experiments/market_calibration.py`` needs vendor data that may not be committed, so its
published numbers cannot be regenerated in CI. What can be pinned is everything between
the data and the numbers: the realised-volatility windows, the cost conversion, the
bootstrap's determinism, and the refusal to cache raw data where git would commit it.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from experiments.market_calibration import (
    REPO_ROOT,
    _assert_cache_is_untracked,
    block_bootstrap_ci,
    forward_realised_vol,
    forward_realised_vol_calendar,
    half_spread_bp,
)


@pytest.fixture(scope="module")
def closes():
    rng = np.random.default_rng(7)
    return 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, 400)))


def test_forward_realised_vol_matches_a_brute_force_loop(closes):
    """RV_i uses the 21 returns AFTER day i -- the window a hedge struck at i experiences."""
    rv = forward_realised_vol(closes, horizon=21, periods_per_year=252)
    r = np.diff(np.log(closes))
    for i in range(closes.size - 21):
        assert rv[i] == pytest.approx(np.sqrt(252 / 21 * np.sum(r[i:i + 21] ** 2)), rel=1e-12)
    assert np.all(np.isnan(rv[closes.size - 21:])), "a window past the sample must be NaN"


def test_calendar_realised_vol_matches_a_brute_force_loop(closes):
    start = dt.date(2020, 1, 1)
    business = [start + dt.timedelta(days=k) for k in range(700)]
    days = [d for d in business if d.weekday() < 5][: closes.size]
    dates = np.array(days, dtype="datetime64[D]")
    rv = forward_realised_vol_calendar(dates, closes, days=30)
    r = np.diff(np.log(closes))
    for i in range(0, len(days) - 40, 7):
        inside = [j for j in range(i + 1, len(days)) if days[j] <= days[i] + dt.timedelta(days=30)]
        expected = np.sqrt(365 / 30 * np.sum(r[np.array(inside) - 1] ** 2))
        assert rv[i] == pytest.approx(expected, rel=1e-12)


def test_one_tick_half_spread_in_basis_points():
    """The paper's 0.16bp: E-mini S&P tick 0.25 points, half of it, over the index level."""
    assert half_spread_bp(0.25, 7743.41) == pytest.approx(0.1614, abs=5e-5)
    assert half_spread_bp(0.25, 1000.0) == pytest.approx(1.25)


def test_block_bootstrap_is_deterministic(closes):
    x = closes / closes.mean()
    a = block_bootstrap_ci(x, block=21, reps=200, seed=1)
    b = block_bootstrap_ci(x, block=21, reps=200, seed=1)
    assert a == b


def test_raw_data_is_never_cached_in_a_tracked_path(tmp_path):
    """One careless ``git add .`` must not be able to publish vendor data."""
    with pytest.raises(SystemExit):
        _assert_cache_is_untracked(REPO_ROOT / "paper")
    _assert_cache_is_untracked(REPO_ROOT / "experiments" / "runs" / "market_data")  # ignored
    _assert_cache_is_untracked(tmp_path)                                            # outside
