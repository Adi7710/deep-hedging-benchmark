"""Black-Scholes pricing and delta hedging.

**Written in full deliberately.** This is reference material, not a learning exercise —
every other component in the repository is checked against it, so it needs to be correct
from day one rather than correct eventually. Rungs 1, 2 and 4 of the correctness ladder
all depend on these functions being right.

The closed forms, for a European call on a non-dividend-paying underlying:

    d1 = ( log(S/K) + (r + sigma^2/2) tau ) / ( sigma sqrt(tau) )
    d2 = d1 - sigma sqrt(tau)

    C     = S Phi(d1) - K exp(-r tau) Phi(d2)
    Delta = Phi(d1)
    Gamma = phi(d1) / ( S sigma sqrt(tau) )

Gamma is here because the Whalley-Wilmott band width depends on it — see
:mod:`dhbench.baselines.whalley_wilmott`.

Implemented in NumPy rather than TensorFlow: these are reference values and analysis
utilities, never part of a training graph. Keeping them out of TF also means the tests can
trust them independently of any TF-side bug.
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
    """The Black-Scholes ``d1`` and ``d2`` terms.

    Args:
        spot: underlying price. Scalar or array.
        strike: strike ``K``.
        time_to_maturity: ``tau = T - t`` in years. Scalar or array, broadcast against
            ``spot``.
        rate: risk-free rate ``r``.
        sigma: volatility, annualised.

    Returns:
        ``(d1, d2)`` as arrays broadcast to the common shape.

    Note:
        At expiry ``tau = 0`` both terms are singular. We clip ``tau`` to a tiny positive
        floor so ``d1 -> +-inf`` cleanly and ``Phi(d1)`` saturates to the correct 1 or 0,
        rather than producing a ``nan`` that propagates through an entire path.
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
    """Black-Scholes European call price.

    The reference value for rung 1 of the correctness ladder: a Monte Carlo price from
    :func:`dhbench.worlds.gbm.simulate_gbm` must match this within Monte Carlo error.

    Args:
        spot: underlying price.
        strike: strike ``K``.
        time_to_maturity: ``tau`` in years.
        rate: risk-free rate.
        sigma: volatility.

    Returns:
        Call price, broadcast to the common shape.

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
    """Black-Scholes call delta, ``Phi(d1)``.

    **The single most important reference value in the project.** Rung 4 of the ladder is
    the requirement that a deep hedging agent trained under GBM with zero transaction costs
    reproduces this function. If the learned policy does not overlay ``Phi(d1)``, every
    downstream number is meaningless.

    Args:
        spot: underlying price.
        strike: strike ``K``.
        time_to_maturity: ``tau`` in years.
        rate: risk-free rate.
        sigma: volatility.

    Returns:
        Delta in ``[0, 1]``, broadcast to the common shape.
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
    """Black-Scholes gamma, ``phi(d1) / (S sigma sqrt(tau))``.

    Needed by the Whalley-Wilmott band, whose half-width scales as ``Gamma^(2/3)``.

    Args:
        spot: underlying price.
        strike: strike ``K``.
        time_to_maturity: ``tau`` in years.
        rate: risk-free rate.
        sigma: volatility.

    Returns:
        Gamma, broadcast to the common shape. Always non-negative.
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
    """Delta hedge positions along simulated paths, rebalanced every step.

    The zero-cost-optimal strategy, and the baseline every learned agent is measured
    against. Under transaction costs it is deliberately *bad* — it rebalances on every
    step regardless of how small the trade — which is why
    :mod:`dhbench.baselines.whalley_wilmott` exists and is the baseline that actually
    matters under frictions.

    Args:
        spot_paths: ``(n_paths, n_steps + 1)`` simulated prices.
        strike: strike ``K``.
        maturity: ``T`` in years.
        rate: risk-free rate.
        sigma: volatility used *for hedging*. Deliberately a separate argument from the
            volatility used to simulate: hedging at the wrong vol is itself an
            experiment worth running.

    Returns:
        ``(n_paths, n_steps)`` hedge positions, ready for
        :func:`dhbench.pnl.terminal_pnl`.

    Note:
        The position at index ``i`` is chosen at time ``t_i`` using ``S_i``, and is held
        over ``[t_i, t_{i+1})``. So the final spot column is never used to *choose* a
        position — only to value one. Off-by-one errors here look like small persistent
        hedging losses and are easy to miss; the shape contract is what protects you.
    """
    spot_paths = np.asarray(spot_paths, dtype=np.float64)
    n_steps = spot_paths.shape[1] - 1

    times = np.linspace(0.0, maturity, n_steps + 1)
    tau = maturity - times[:-1]  # drop the last column: no decision is made at T

    return bs_delta(spot_paths[:, :-1], strike, tau[None, :], rate, sigma)
