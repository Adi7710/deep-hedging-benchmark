"""Zakamouline band — the second classical baseline.

Zakamouline (2006) improves on Whalley-Wilmott in two ways that matter for this benchmark:

1. It handles **fixed costs** as well as proportional ones, so it remains a valid baseline
   in the non-convex cell of the grid where Whalley-Wilmott does not apply.
2. Its band is **asymmetric** and centred on a delta that is itself adjusted away from
   ``delta_BS``, rather than a symmetric band around the Black-Scholes delta. Empirically
   this outperforms Whalley-Wilmott at realistic (non-asymptotic) cost levels — which is
   the regime the benchmark actually runs in.

The form is

    delta_centre = delta_BS + H_0 * (adjustment term)
    band         = [ delta_centre - H_1, delta_centre + H_1 ]

with ``H_0`` and ``H_1`` given by fitted expressions in the paper's §5.

**Implement this after Whalley-Wilmott, not before.** The coefficients are fiddly, the
paper's notation differs from ours, and the marginal benchmark value is lower than getting
the first band right. Stage 4 work, not Stage 0.

Translate the paper's symbols into our notation in ``docs/00-problem-statement.md`` before
writing any of it — the crosswalk table exists for exactly this.
"""

from __future__ import annotations

import numpy as np

__all__ = ["zakamouline_band", "zakamouline_hedge_positions"]


def zakamouline_band(
    spot: float | np.ndarray,
    strike: float,
    time_to_maturity: float | np.ndarray,
    rate: float,
    sigma: float,
    cost_rate: float,
    fixed_cost: float,
    risk_aversion: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Centre and half-width of the Zakamouline band.

    Args:
        spot: underlying price.
        strike: strike ``K``.
        time_to_maturity: ``tau`` in years.
        rate: risk-free rate.
        sigma: volatility.
        cost_rate: proportional cost ``c``.
        fixed_cost: fixed per-trade cost ``f``. Zero reduces to the proportional case.
        risk_aversion: ``lambda``, matching the learned agent's objective.

    Returns:
        ``(centre, half_width)``, each broadcast to the common shape. Note the centre is
        **not** ``delta_BS`` — that adjustment is a substantive part of the method, and
        dropping it silently reduces this to a worse Whalley-Wilmott.
    """
    raise NotImplementedError


def zakamouline_hedge_positions(
    spot_paths: np.ndarray,
    strike: float,
    maturity: float,
    rate: float,
    sigma: float,
    cost_rate: float,
    fixed_cost: float,
    risk_aversion: float,
) -> np.ndarray:
    """Hedge positions under the Zakamouline band rule.

    Same sequential structure as
    :func:`dhbench.baselines.whalley_wilmott.band_hedge_positions`, and the same warning
    applies: **trade to the band edge, not to the centre.**

    Args:
        spot_paths: ``(n_paths, n_steps + 1)`` simulated prices.
        strike: strike ``K``.
        maturity: ``T`` in years.
        rate: risk-free rate.
        sigma: hedging volatility.
        cost_rate: proportional cost ``c``.
        fixed_cost: fixed per-trade cost ``f``.
        risk_aversion: ``lambda``.

    Returns:
        ``(n_paths, n_steps)`` hedge positions.
    """
    raise NotImplementedError
