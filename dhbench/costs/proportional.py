"""Proportional transaction costs -- the standard friction model.

    cost = c * S * |dq|          (a stand-in for paying half the bid-ask spread)

Used by Buehler et al. (2019) and essentially all of the deep hedging literature, which
is what makes it the comparable choice.

Why it is the interesting case: at zero cost the optimal hedge is just the BS delta and
there is nothing to learn. With proportional costs, continuous rebalancing becomes
infinitely expensive and the optimal policy becomes a no-transaction *band* around the
delta -- see ``baselines/whalley_wilmott.py`` for the closed form it is judged against.

The arithmetic itself lives in :func:`dhbench.pnl.transaction_costs`. This module only
holds the rate, so a cost model is a named config object rather than a loose float.
"""

from __future__ import annotations

from dataclasses import dataclass

import tensorflow as tf

__all__ = ["ProportionalCost"]


@dataclass(frozen=True)
class ProportionalCost:
    """Proportional cost model. ``rate`` is c, e.g. 0.005 for 50bp.

    Frozen: a config object that mutates mid-run would break reproducibility.
    """

    rate: float

    def __call__(self, spot: tf.Tensor, traded: tf.Tensor) -> tf.Tensor:
        """Cost per trade: ``c * spot * |traded|``.

        spot:   (n_paths, n_trades) price at which each trade executes
        traded: (n_paths, n_trades) signed quantity ``delta_i - delta_{i-1}``
        ->      (n_paths, n_trades), non-negative
        """
        raise NotImplementedError

    @property
    def name(self) -> str:
        """Identifier used in config files and results tables."""
        return f"proportional_{self.rate}"
