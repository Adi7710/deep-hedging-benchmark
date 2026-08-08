"""dhbench -- a reproducible benchmark for deep hedging.

See ``docs/00-problem-statement.md`` for the notation every module in this package uses,
and ``README.md`` for the correctness ladder that gates the work.

Array shape conventions, used everywhere without exception:

    spot   (n_paths, n_steps + 1)
    delta  (n_paths, n_steps)
    payoff (n_paths,)
    pnl    (n_paths,)

Sub-packages:
    worlds      market simulators (GBM, Heston, regime-switching, jumps)
    costs       transaction cost models
    objectives  convex risk measures (entropic, CVaR, mean-variance)
    agents      learned hedging policies
    baselines   classical hedges (BS delta, Whalley-Wilmott, Zakamouline)
    evaluation  metrics and regime-shift stress tests

Module ``pnl`` is the single source of truth for P&L accounting.
"""

__version__ = "0.1.0-dev"
