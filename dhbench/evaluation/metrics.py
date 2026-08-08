"""Benchmark metrics.

Every number that appears in a results table comes from here. Centralised for the same
reason :mod:`dhbench.pnl` is: a metric computed slightly differently in two places is a
comparability bug, which is precisely the class of problem this benchmark exists to
eliminate.

The metric set, and why each earns its place:

    cvar_95              the tail is what hedging is FOR
    pnl_std              comparability with older literature
    pnl_mean             detects a "hedge" that is really a directional bet
    turnover             is the edge just trading more?
    total_cost           separates gross skill from net
    indifference_price   economic interpretation; checkable against Black-Scholes

**Report the whole set, always.** A method that improves CVaR by trading three times as
much has not improved anything a desk would deploy, and reporting CVaR alone hides that.
"""

from __future__ import annotations

import tensorflow as tf

__all__ = ["summarise", "degradation_ratio"]


def summarise(
    pnl: tf.Tensor,
    delta: tf.Tensor,
    spot: tf.Tensor,
    cost_rate: float,
    alpha: float = 0.95,
) -> dict[str, float]:
    """Compute the full metric set for one experimental cell.

    Args:
        pnl: ``(n_paths,)`` terminal P&L.
        delta: ``(n_paths, n_steps)`` hedge positions.
        spot: ``(n_paths, n_steps + 1)`` price paths.
        cost_rate: proportional cost used, for the cost breakdown.
        alpha: CVaR confidence level.

    Returns:
        Plain Python floats keyed by metric name — ``cvar_95``, ``pnl_mean``, ``pnl_std``,
        ``turnover_mean``, ``total_cost_mean``. Python floats rather than tensors so the
        result serialises straight to JSON/YAML without a conversion step.

    Note:
        Reuse :func:`dhbench.objectives.cvar.cvar_empirical`,
        :func:`dhbench.pnl.turnover`, and :func:`dhbench.pnl.transaction_costs`. Do not
        re-derive any of them here.
    """
    raise NotImplementedError


def degradation_ratio(
    metric_in_distribution: float,
    metric_out_of_distribution: float,
) -> float:
    """Relative worsening of a metric under regime shift. **The §8 headline number.**

    Args:
        metric_in_distribution: metric on test paths from the *training* world.
        metric_out_of_distribution: same metric on paths from a *different* world.

    Returns:
        Relative degradation. ``0`` means no degradation; ``1`` means the metric doubled
        in badness. Larger is worse.

    Note:
        Both arguments must be **lower-is-better** metrics (CVaR, std). Passing a
        higher-is-better metric silently inverts the interpretation, which is why this
        takes floats and not a metric name — the caller has to think about it.

        Guard the denominator: an in-distribution CVaR near zero makes the ratio explode.
        Report the raw pair alongside the ratio so a reader can see when that happens.

    Warning:
        The in-distribution figure must come from **held-out test paths**, not training
        paths. Comparing training-set performance against out-of-distribution performance
        conflates overfitting with regime fragility, and they are different phenomena with
        different remedies. Disjoint seeds for train/validation/test, per the protocol.
    """
    raise NotImplementedError
