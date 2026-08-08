"""Fixed-plus-proportional transaction costs.

    cost = (f + c * S * |dq|) * 1{dq != 0}

A per-trade fee ``f`` charged whenever a trade happens at all, on top of the proportional
component. Zakamouline (2006) handles exactly this case, which is why his band is the
second baseline.

**This is the non-convex cell of the grid, and it matters more than its size suggests.**
The indicator ``1{dq != 0}`` is discontinuous at zero, so the objective is no longer convex
in the policy. *Deep Hedging Under Non-Convexity* (arXiv:2510.01874) shows gradient-based
deep hedging depends on convexity and converges to local optima without it — proposing
MCTS instead. Including this cost model is what lets the benchmark **test that claim
directly** rather than cite it, and a confirmed failure here is one of the more publishable
things in the grid.

Practical note: the indicator has zero gradient almost everywhere and is undefined at zero,
so it cannot be backpropagated through as written. Options, to be decided and **recorded in
the protocol**:

1. A smooth surrogate during training (e.g. ``tanh(|dq| / eps)``), with the true cost used
   at evaluation. Standard, but the surrogate is a modelling choice that affects results.
2. Train with proportional costs only, evaluate with fixed costs included. Honest but
   mismatched.
3. Accept the zero gradient and see what happens. Arguably the most informative option
   given what the paper is testing.

Whichever is chosen, it is a protocol decision, not an implementation detail — see
``docs/03-benchmark-protocol.md``.
"""

from __future__ import annotations

from dataclasses import dataclass

import tensorflow as tf

__all__ = ["FixedPlusProportionalCost"]


@dataclass(frozen=True)
class FixedPlusProportionalCost:
    """Fixed fee plus proportional transaction cost.

    Attributes:
        fixed: ``f``, charged once per non-zero trade.
        rate: ``c``, the proportional component.
        smoothing: ``eps`` for the differentiable surrogate of the indicator. ``None``
            uses the exact, non-differentiable indicator.
    """

    fixed: float
    rate: float
    smoothing: float | None = None

    def __call__(self, spot: tf.Tensor, traded: tf.Tensor) -> tf.Tensor:
        """Cost charged per trade.

        Args:
            spot: ``(n_paths, n_trades)`` execution prices.
            traded: ``(n_paths, n_trades)`` signed traded quantities.

        Returns:
            ``(n_paths, n_trades)`` non-negative cost per trade.

        Note:
            When ``smoothing`` is set, replace the indicator with
            ``tanh(|traded| / smoothing)``, which → 1 for trades large relative to ``eps``
            and → 0 for negligible ones, and is differentiable everywhere. Record the
            chosen ``eps`` in the config — it is a result-affecting parameter, not a
            numerical detail.
        """
        raise NotImplementedError

    @property
    def name(self) -> str:
        """Identifier used in config files and results tables."""
        return f"fixed_{self.fixed}_prop_{self.rate}"
