"""Proportional transaction costs — the standard friction model.

Cost of trading ``|dq|`` shares at price ``S``:

    cost = c * S * |dq|

which is a stand-in for paying half the bid-ask spread on each trade. It is the friction
used in Buehler et al. (2019) and in essentially all of the deep hedging literature, which
makes it the one that matters for comparability.

**Why proportional costs are the interesting case.** With zero costs the optimal hedge is
the Black-Scholes delta and there is nothing to learn. With proportional costs, continuous
rebalancing becomes infinitely expensive, the optimal policy becomes a **no-transaction
band** around the delta, and *that* band is what a learned agent must discover — see
``baselines/whalley_wilmott.py`` for the closed-form asymptotic answer it is judged
against.

Note the actual arithmetic lives in :func:`dhbench.pnl.transaction_costs`. This module
exists to hold the *rate*, so that a cost model is a config-level object with a name rather
than a float threaded through call signatures.
"""

from __future__ import annotations

from dataclasses import dataclass

import tensorflow as tf

__all__ = ["ProportionalCost"]


@dataclass(frozen=True)
class ProportionalCost:
    """Proportional transaction cost model.

    Attributes:
        rate: ``c``, the proportional cost. E.g. ``0.005`` is 50 basis points.

    Note:
        Frozen so a cost model can't be mutated mid-experiment — a config object that
        changes underneath a run would break the reproducibility contract.
    """

    rate: float

    def __call__(self, spot: tf.Tensor, traded: tf.Tensor) -> tf.Tensor:
        """Cost charged per trade.

        Args:
            spot: ``(n_paths, n_trades)`` price at which each trade executes.
            traded: ``(n_paths, n_trades)`` signed traded quantity ``delta_i - delta_{i-1}``.

        Returns:
            ``(n_paths, n_trades)`` non-negative cost per trade.
        """
        raise NotImplementedError

    @property
    def name(self) -> str:
        """Identifier used in config files and results tables."""
        return f"proportional_{self.rate}"
