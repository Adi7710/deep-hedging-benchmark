"""Mean-variance objective.

    rho(X) = -E[X] + (lambda / 2) * Var(X)

**Not a coherent risk measure**, and not even monotone — it penalises upside deviation as
heavily as downside, so a strategy that occasionally makes a large profit is scored as
risky. Nobody would choose it on theoretical grounds.

It is here because a large fraction of the deep hedging and RL-hedging literature reports
it, and the 2023 *Mathematics* review found mean-variance objectives "prevail" in reward
formulations. Excluding it would make this benchmark incomparable with the very work it is
trying to make comparable.

Its presence also serves sub-question 5 directly: **if the ranking of methods changes
between mean-variance and CVaR, that is evidence that cross-paper comparison is unsound**,
which is a result in its own right and one of the cheaper ones to obtain.
"""

from __future__ import annotations

import tensorflow as tf

__all__ = ["MeanVarianceRisk"]


class MeanVarianceRisk:
    """Mean-variance objective.

    Args:
        risk_aversion: ``lambda >= 0``. Zero recovers pure expected-return maximisation,
            which will not hedge at all — a useful degenerate check.
    """

    def __init__(self, risk_aversion: float = 1.0) -> None:
        if risk_aversion < 0:
            raise ValueError(f"risk_aversion must be non-negative, got {risk_aversion}")
        self.risk_aversion = risk_aversion

    def __call__(self, pnl: tf.Tensor) -> tf.Tensor:
        """Evaluate the objective. This is the training loss.

        Args:
            pnl: ``(n_paths,)`` terminal P&L.

        Returns:
            Scalar. Lower is better.

        Note:
            Use ``tf.math.reduce_variance``. It is the **population** variance (divides by
            ``n``), not the sample variance — with thousands of paths the difference is
            negligible, but state which one you used, because a reviewer comparing against
            a paper that used the other will notice.
        """
        lam = tf.constant(self.risk_aversion, dtype=pnl.dtype)
        # POPULATION variance (divides by n). Recorded in the protocol, not left implicit.
        return -tf.reduce_mean(pnl) + 0.5 * lam * tf.math.reduce_variance(pnl)

    @property
    def name(self) -> str:
        """Identifier used in config files and results tables."""
        return f"meanvar_{self.risk_aversion}"
