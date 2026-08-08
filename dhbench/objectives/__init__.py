"""Convex risk measures — the training objectives.

Deep hedging has no labels. The loss is a functional of the terminal P&L distribution over
the whole simulated path, and that functional is what these classes implement.

Every objective is callable as ``(pnl) -> scalar``, lower-is-better, and carries a ``name``
for configs and results tables.

    entropic       (1/lam) log E[exp(-lam X)]        -- Buehler's default; coherent
    cvar           Rockafellar-Uryasev form           -- coherent; targets the tail
    mean_variance  -E[X] + (lam/2) Var(X)             -- NOT coherent; included for
                                                         comparability with the literature

**One asymmetry to be aware of:** :class:`~dhbench.objectives.cvar.CVaRRisk` carries a
trainable auxiliary variable and therefore exposes ``trainable_variables``; the others do
not. Training loops should always write::

    variables = model.trainable_variables + getattr(objective, "trainable_variables", [])

so that swapping the objective in a config does not silently drop a variable from the
optimiser. Forgetting this is the standard CVaR bug — training plateaus and looks like a
learning-rate problem.
"""
