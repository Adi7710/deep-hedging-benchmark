"""Transaction cost models.

A cost model is a frozen dataclass that is callable as ``(spot, traded) -> cost`` and
carries a ``name`` used in configs and results tables. Frozen because a config object that
mutates mid-run would break the reproducibility contract.

    proportional  c * S * |dq|                      -- the standard friction, convex
    fixed         (f + c * S * |dq|) * 1{dq != 0}   -- non-convex; tests the AlphaZero claim

The actual P&L arithmetic lives in :mod:`dhbench.pnl`, not here. These objects hold the
*parameters* so a cost model can be named and configured rather than threaded through call
signatures as bare floats.
"""
