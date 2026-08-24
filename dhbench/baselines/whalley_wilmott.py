"""Whalley-Wilmott asymptotic no-transaction band -- the baseline that matters.

Beating naive delta hedging under costs is trivial (it rebalances every step regardless
of trade size, so it bleeds cost). Whalley & Wilmott (1997) give the asymptotically
optimal answer under proportional costs, and that is the real bar.

Band half-width around the BS delta, in the small-cost limit:

    H = ( (3/2) * c * S * Gamma^2 * exp(-r * tau) / lambda )^(1/3)

Hold while |delta_held - delta_BS| <= H. Otherwise trade to the nearest band EDGE.

Three scalings a learned agent should rediscover -- use them to sanity-check it:

    H ~ c^(1/3)        tripling cost widens the band only ~44%
    H ~ Gamma^(2/3)    wide at the money and near expiry, where delta moves fastest
    H ~ lambda^(-1/3)  more risk aversion -> tighter band -> more trading
"""

from __future__ import annotations

import numpy as np

from dhbench.baselines.bs_delta import bs_delta, bs_gamma

__all__ = ["whalley_wilmott_band", "band_hedge_positions"]


def whalley_wilmott_band(
    spot: float | np.ndarray,
    strike: float,
    time_to_maturity: float | np.ndarray,
    rate: float,
    sigma: float,
    cost_rate: float,
    risk_aversion: float,
) -> np.ndarray:
    """Band half-width H (see module docstring). Non-negative, broadcast to common shape.

    risk_aversion: must match the lambda of the objective the learned agent is trained
        against, or the comparison is not like-for-like. Easy mistake, invalidates the
        result.

    Reuse :func:`dhbench.baselines.bs_delta.bs_gamma`; do not re-derive gamma here.
    The formula is asymptotic in small c -- at large costs it is a heuristic, and that
    belongs in the paper rather than being quietly ignored.

    At c = 0 this is identically zero, so the band rule degenerates to delta hedging --
    which is correct: with no friction there is no reason to tolerate tracking error.

    Near expiry at the money gamma diverges, so H does too and the rule stops trading.
    That is the formula being honest rather than broken: when gamma is unbounded no
    finite rebalancing helps, and pnl.py liquidates at T regardless.
    """
    spot = np.asarray(spot, dtype=np.float64)
    tau = np.maximum(np.asarray(time_to_maturity, dtype=np.float64), 1e-12)

    gamma = bs_gamma(spot, strike, tau, rate, sigma)
    inner = 1.5 * cost_rate * spot * gamma**2 * np.exp(-rate * tau) / risk_aversion
    return np.cbrt(inner)


def band_hedge_positions(
    spot_paths: np.ndarray,
    strike: float,
    maturity: float,
    rate: float,
    sigma: float,
    cost_rate: float,
    risk_aversion: float,
) -> np.ndarray:
    """Hedge positions under the band rule.

    spot_paths: (n_paths, n_steps + 1)
    ->          (n_paths, n_steps)

    Per step: compute delta_BS and H. If the held position is inside
    [delta_BS - H, delta_BS + H], hold. Otherwise trade to the nearest EDGE.

    Trading all the way back to delta_BS is the single most common error in band
    hedging. It throws away most of the cost saving the band exists to capture, so the
    baseline underperforms -- which makes deep hedging look better than it is, biasing
    the headline comparison in our favour. Precisely what this benchmark should not do.

    The whole rule is a clip. Inside the band, np.clip returns the held position
    unchanged (no trade); outside, it returns the nearer bound (trade to that edge).
    Writing it this way makes the trade-to-the-centre bug unrepresentable.

    Inherently sequential: step i depends on step i-1. Vectorise across paths, loop over
    steps. NumPy is fine -- baselines never sit inside a training graph.
    """
    spot_paths = np.asarray(spot_paths, dtype=np.float64)
    n_paths, n_cols = spot_paths.shape
    n_steps = n_cols - 1

    times = np.linspace(0.0, maturity, n_cols)
    tau = maturity - times[:-1]  # decision times only: none is made at T

    positions = np.empty((n_paths, n_steps), dtype=np.float64)
    held = np.zeros(n_paths, dtype=np.float64)  # delta_{-1} = 0, start flat

    for i in range(n_steps):
        spot_i = spot_paths[:, i]
        target = bs_delta(spot_i, strike, tau[i], rate, sigma)
        half_width = whalley_wilmott_band(
            spot_i, strike, tau[i], rate, sigma, cost_rate, risk_aversion
        )
        held = np.clip(held, target - half_width, target + half_width)
        positions[:, i] = held

    return positions
