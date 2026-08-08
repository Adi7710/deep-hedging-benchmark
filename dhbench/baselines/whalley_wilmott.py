"""Whalley-Wilmott asymptotic no-transaction band.

**This is the baseline that matters.** Beating naive delta hedging under transaction costs
is trivial — delta hedging rebalances on every step regardless of trade size, so it bleeds
cost. Any policy that trades less will beat it, which proves nothing about learning.
Whalley-Wilmott is the asymptotically optimal answer under proportional costs, and it is
the bar deep hedging has to clear to be interesting.

Whalley & Wilmott (1997) derive, in the small-cost limit, a band around the Black-Scholes
delta with half-width

    H = ( (3/2) * c * S * Gamma^2 * exp(-r * tau) / lambda )^(1/3)

Trade back to ``delta_BS`` only when ``|delta_held - delta_BS| > H``; otherwise hold.

Three properties worth internalising, because they are what a learned agent should
rediscover and they are how you sanity-check its behaviour:

- ``H ~ c^(1/3)``. Band width grows slowly in cost — tripling costs widens the band by
  only ~44%.
- ``H ~ Gamma^(2/3)``. Wide where gamma is high (at the money, near expiry) — precisely
  where delta moves fastest and naive hedging is most expensive.
- ``H ~ lambda^(-1/3)``. More risk aversion, tighter band, more trading.

Rung 5 of the correctness ladder: a learned policy trained with small proportional costs
should produce a band that approximates this one. It is the strongest available evidence
that the agent has learned the right *structure* rather than merely a good number.
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
    """Half-width ``H`` of the no-transaction band.

    Args:
        spot: underlying price.
        strike: strike ``K``.
        time_to_maturity: ``tau`` in years.
        rate: risk-free rate.
        sigma: volatility.
        cost_rate: proportional cost ``c``.
        risk_aversion: ``lambda``. **Must match the risk aversion of the objective the
            learned agent is trained against**, or the comparison is not like-for-like —
            an easy and invalidating mistake.

    Returns:
        Band half-width, broadcast to the common shape. Non-negative.

    Note:
        Reuse :func:`dhbench.baselines.bs_delta.bs_gamma` — do not re-derive gamma here.
        The formula is asymptotic in small ``c``; at large costs it is a heuristic, and
        that limitation belongs in the paper rather than being quietly ignored.
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
    """Hedge positions under the Whalley-Wilmott band rule.

    At each step: compute ``delta_BS`` and the band ``H``. If the currently held position
    is inside ``[delta_BS - H, delta_BS + H]``, hold. Otherwise trade **to the nearest band
    edge**, not to ``delta_BS``.

    Args:
        spot_paths: ``(n_paths, n_steps + 1)`` simulated prices.
        strike: strike ``K``.
        maturity: ``T`` in years.
        rate: risk-free rate.
        sigma: hedging volatility.
        cost_rate: proportional cost ``c``.
        risk_aversion: ``lambda``.

    Returns:
        ``(n_paths, n_steps)`` hedge positions.

    Warning:
        **Trade to the band edge, not to the delta.** This is the single most common
        implementation error in band hedging. Rebalancing all the way to ``delta_BS``
        throws away most of the cost saving the band exists to capture, and the resulting
        strategy underperforms — which then makes deep hedging look better than it should,
        biasing the paper's headline comparison in our favour. That failure mode is
        exactly what this benchmark is meant to eliminate.

    Note:
        Inherently sequential: the position at step ``i`` depends on the position at
        ``i-1``. Vectorise across paths, loop over steps. NumPy is fine — baselines are
        never inside a training graph.
    """
    raise NotImplementedError
