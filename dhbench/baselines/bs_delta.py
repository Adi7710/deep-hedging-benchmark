"""Black-Scholes pricing and delta hedging.

**Written in full deliberately** -- reference material, not a learning exercise. Rungs 1,
2 and 4 all check against it, so it has to be right from day one.

    d1 = ( log(S/K) + (r + sigma^2/2) tau ) / ( sigma sqrt(tau) )
    d2 = d1 - sigma sqrt(tau)

    C     = S Phi(d1) - K exp(-r tau) Phi(d2)
    Delta = Phi(d1)
    Gamma = phi(d1) / ( S sigma sqrt(tau) )

Gamma is here because the Whalley-Wilmott band width depends on it.

NumPy rather than TensorFlow: these are reference values, never part of a training
graph, and keeping them out of TF means the tests can trust them independently of any
TF-side bug.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

__all__ = ["d1_d2", "bs_call_price", "bs_delta", "bs_gamma", "delta_hedge_positions"]

ArrayLike = float | np.ndarray


def d1_d2(
    spot: ArrayLike,
    strike: float,
    time_to_maturity: ArrayLike,
    rate: float,
    sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    """The d1 and d2 terms, broadcast to the common shape.

    tau is clipped to a tiny positive floor: at expiry both terms are singular, and we
    want d1 -> +-inf cleanly so Phi(d1) saturates to 1 or 0 rather than emitting a nan
    that propagates through a whole path.
    """
    spot = np.asarray(spot, dtype=np.float64)
    tau = np.maximum(np.asarray(time_to_maturity, dtype=np.float64), 1e-12)

    vol_sqrt_tau = sigma * np.sqrt(tau)
    d1 = (np.log(spot / strike) + (rate + 0.5 * sigma**2) * tau) / vol_sqrt_tau
    d2 = d1 - vol_sqrt_tau
    return d1, d2


def bs_call_price(
    spot: ArrayLike,
    strike: float,
    time_to_maturity: ArrayLike,
    rate: float,
    sigma: float,
) -> np.ndarray:
    """European call price. The reference value for rung 1.

    Example:
        >>> round(float(bs_call_price(100.0, 100.0, 1.0, 0.0, 0.2)), 4)
        7.9656
    """
    d1, d2 = d1_d2(spot, strike, time_to_maturity, rate, sigma)
    tau = np.maximum(np.asarray(time_to_maturity, dtype=np.float64), 1e-12)
    spot = np.asarray(spot, dtype=np.float64)
    return spot * norm.cdf(d1) - strike * np.exp(-rate * tau) * norm.cdf(d2)


def bs_delta(
    spot: ArrayLike,
    strike: float,
    time_to_maturity: ArrayLike,
    rate: float,
    sigma: float,
) -> np.ndarray:
    """Call delta, Phi(d1), in [0, 1].

    The single most important reference value in the project: rung 4 requires a deep
    hedging agent trained under GBM at zero cost to reproduce this function. If the
    learned policy does not overlay Phi(d1), every downstream number is meaningless.
    """
    d1, _ = d1_d2(spot, strike, time_to_maturity, rate, sigma)
    return norm.cdf(d1)


def bs_gamma(
    spot: ArrayLike,
    strike: float,
    time_to_maturity: ArrayLike,
    rate: float,
    sigma: float,
) -> np.ndarray:
    """Gamma, phi(d1) / (S sigma sqrt(tau)). Non-negative.

    Needed by the Whalley-Wilmott band, whose half-width scales as Gamma^(2/3).
    """
    d1, _ = d1_d2(spot, strike, time_to_maturity, rate, sigma)
    tau = np.maximum(np.asarray(time_to_maturity, dtype=np.float64), 1e-12)
    spot = np.asarray(spot, dtype=np.float64)
    return norm.pdf(d1) / (spot * sigma * np.sqrt(tau))


def delta_hedge_positions(
    spot_paths: np.ndarray,
    strike: float,
    maturity: float,
    rate: float,
    sigma: float,
) -> np.ndarray:
    """Delta hedge positions along paths, rebalanced every step.

    spot_paths: (n_paths, n_steps + 1)
    ->          (n_paths, n_steps)

    Optimal at zero cost; deliberately bad under costs, since it rebalances every step
    however small the trade. That is why whalley_wilmott.py exists.

    sigma is the volatility used *for hedging* -- deliberately separate from the
    volatility used to simulate, since hedging at the wrong vol is its own experiment.

    The position at index i is chosen at t_i from S_i and held over [t_i, t_{i+1}), so
    the final spot column is never used to *choose* a position, only to value one.
    """
    spot_paths = np.asarray(spot_paths, dtype=np.float64)
    n_steps = spot_paths.shape[1] - 1

    times = np.linspace(0.0, maturity, n_steps + 1)
    tau = maturity - times[:-1]  # drop the last column: no decision is made at T

    return bs_delta(spot_paths[:, :-1], strike, tau[None, :], rate, sigma)
