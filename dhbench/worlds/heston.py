"""Heston stochastic volatility — the realistic training world.

    dS = mu S dt + sqrt(v) S dW_S
    dv = kappa (theta - v) dt + xi sqrt(v) dW_v,     corr(dW_S, dW_v) = rho

Stage 3. The world agents are *trained* on for the regime-fragility experiment; they are
then *tested* on ``regime_switching`` and ``jumps``.

Two things make this harder than GBM and both bite in practice:

**Negative variance.** A naive Euler step on ``v`` goes negative whenever the Feller
condition ``2 kappa theta > xi^2`` is violated — and realistic calibrations routinely
violate it. Use the **full truncation** scheme (Lord et al. 2010): keep ``v`` as-is but use
``max(v, 0)`` everywhere it appears under a square root. It is the standard fix, is a
one-line change, and has the least bias of the simple schemes. Do not simply clip ``v``
itself to zero — that biases the variance process upward.

**Correlated increments.** Draw two independent standard normals and combine:
``W_v = Z_1``, ``W_S = rho Z_1 + sqrt(1 - rho^2) Z_2``. Getting the correlation sign wrong
silently inverts the volatility skew, and the resulting prices look plausible enough that
you won't notice without rung 3.

Rung 3 of the ladder checks the simulator against semi-analytic characteristic-function
prices, which is why :func:`heston_price_cf` lives here rather than in a test file — it is
a component, not a test fixture.
"""

from __future__ import annotations

import tensorflow as tf

__all__ = ["simulate_heston", "heston_price_cf"]


def simulate_heston(
    n_paths: int,
    n_steps: int,
    s0: float,
    mu: float,
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    maturity: float,
    generator: tf.random.Generator,
    dtype: tf.DType = tf.float32,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Simulate Heston price and variance paths.

    Args:
        n_paths: number of independent paths.
        n_steps: number of time steps.
        s0: initial spot.
        mu: drift of the spot process.
        v0: initial variance (**not** volatility — ``v0 = sigma_0**2``).
        kappa: mean-reversion speed of the variance.
        theta: long-run variance.
        xi: volatility of volatility.
        rho: correlation between the spot and variance Brownian motions. Typically
            negative for equities — that is what produces the leverage/skew effect.
        maturity: ``T`` in years.
        generator: explicit ``tf.random.Generator``.
        dtype: float dtype.

    Returns:
        ``(spot, variance)``, each ``(n_paths, n_steps + 1)``.

        The variance path is returned because an agent may legitimately observe it (or a
        proxy for it) as a state feature. Whether it *should* is an open protocol
        question — see ``docs/03-benchmark-protocol.md``.
    """
    raise NotImplementedError


def heston_price_cf(
    s0: float,
    strike: float,
    maturity: float,
    r: float,
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
) -> float:
    """Semi-analytic European call price under Heston, via the characteristic function.

    The reference value that rung 3 checks the Monte Carlo simulator against. Uses the
    Heston (1993) / Lewis formulation with numerical integration.

    Implement with ``scipy.integrate.quad`` — this is a one-off reference calculation, not
    something in a training loop, so readability beats speed and there is no reason for it
    to be a tensor op.

    Args:
        s0: initial spot.
        strike: strike ``K``.
        maturity: ``T`` in years.
        r: risk-free rate.
        v0, kappa, theta, xi, rho: Heston parameters, as in :func:`simulate_heston`.

    Returns:
        European call price.

    Warning:
        Use the **Little Heston Trap** formulation (Albrecher et al. 2007). The original
        Heston branch of the complex logarithm is discontinuous and produces wrong prices
        for longer maturities — the failure is intermittent and looks like a
        Monte Carlo error, which makes it genuinely nasty to debug.
    """
    raise NotImplementedError
