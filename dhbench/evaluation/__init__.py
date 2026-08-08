"""Evaluation — metrics and regime-shift stress testing.

    metrics  the full metric set; every number in a results table comes from here
    stress   train-on-one-world, test-on-another. The machinery behind PAPER.md §8

Two rules that the benchmark's credibility rests on:

**Report the whole metric set, always.** A method that improves CVaR by trading three times
as much has improved nothing a desk would deploy. Reporting CVaR alone hides that, and
several papers in the comparison set do exactly this.

**Evaluation seeds are disjoint from training seeds.** Comparing training-set performance
against out-of-distribution performance conflates overfitting with regime fragility. They
are different phenomena with different remedies, and mixing them would undermine the one
result this project most wants to be right about.
"""
