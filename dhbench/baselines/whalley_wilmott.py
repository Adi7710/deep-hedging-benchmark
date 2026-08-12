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
    """
    raise NotImplementedError


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

    Inherently sequential: step i depends on step i-1. Vectorise across paths, loop over
    steps. NumPy is fine -- baselines never sit inside a training graph.
    """
    raise NotImplementedError