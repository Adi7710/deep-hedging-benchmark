"""Classical hedging strategies — the bar deep hedging must clear.

    bs_delta         Phi(d1), rebalanced every step. Optimal at zero cost
    whalley_wilmott  asymptotic no-transaction band. THE baseline under frictions
    zakamouline      asymmetric band; also handles fixed costs

All in NumPy, never TensorFlow — baselines are reference values and analysis utilities, and
never appear inside a training graph. Keeping them out of TF means the tests can trust them
independently of any TF-side bug.

**Why the band baselines matter so much.** Beating naive delta hedging under transaction
costs is trivial: delta hedging rebalances every step regardless of trade size, so anything
that trades less will win. A paper that reports "deep hedging beats delta hedging under
costs" has shown nothing. Whalley-Wilmott is the real bar, and a substantial part of this
benchmark's value is simply insisting on it.

``bs_delta`` is written in full — it is reference material that everything else is checked
against, so it has to be correct from the start rather than eventually.
"""
