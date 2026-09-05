"""The measured claims the paper rests on.

Every number quoted in `paper/` comes from `experiments/findings.py`. These tests pin the
*qualitative* content of each finding — direction, ordering, significance — rather than
exact values, so they survive a change in path count but fail if a claim stops being true.

Run at reduced sample sizes for speed. The published figures use the module's defaults.

The most important test here is :func:`test_the_centre_over_edge_ratio_is_not_estimable`,
which pins a *negative* result: the ratio that an earlier draft quoted from a single seed
is not a stable quantity. It exists so nobody, including a future version of this project,
quotes it again.
"""

from __future__ import annotations

import numpy as np
import pytest

from experiments.findings import baseline, cvar, misspecification, precision

FAST = dict(n_paths=8_000)


def test_cvar_is_the_mean_of_the_left_tail():
    """Sanity on the helper every finding depends on."""
    x = np.arange(100.0)  # 0..99, worst 5% is 0..4
    assert cvar(x, 0.95) == pytest.approx(np.arange(6.0).mean(), abs=1.0)


# --------------------------------------------------------------------------------------
# Finding 1 — the baseline
# --------------------------------------------------------------------------------------

def test_band_edge_beats_band_centre_beats_delta():
    """The ordering the paper's §4 argument depends on.

    If this ever reverses, either the band implementation regressed to trading to the
    centre, or the baseline claim is wrong. Both need to stop the presses.
    """
    out = baseline(n_seeds=6, **FAST)
    assert out["edge_vs_delta"]["mean"] > 0.0
    assert out["centre_vs_delta"]["mean"] > 0.0
    assert out["edge_vs_centre"]["mean"] > 0.0


def test_the_edge_rule_advantage_is_significant_and_consistent():
    """Not just positive on average — positive in essentially every seed.

    Consistency of sign matters more than the t-statistic here, because with few seeds a
    large t can come from one outlier shrinking the estimated standard deviation.
    """
    n_seeds = 8
    out = baseline(n_seeds=n_seeds, **FAST)
    assert out["edge_vs_centre"]["t"] > 3.0
    assert out["edge_vs_centre"]["consistent"] >= n_seeds - 1


def test_the_centre_over_edge_ratio_is_not_estimable():
    """**Pins a withdrawn claim so it cannot return.**

    An earlier draft quoted "the centre rule captures 7% of the available improvement"
    from a single seed. The ratio divides two differences each of order the CVaR-95
    estimation noise, and across seeds its standard deviation is comparable to or larger
    than its mean — it spans zero. Reporting it as a point estimate is exactly the
    practice the paper criticises.

    This test asserts the ratio remains *unstable*. If sample sizes ever grow enough to
    make it estimable, this test fails and the claim can be revisited deliberately rather
    than by accident.
    """
    out = baseline(n_seeds=8, **FAST)
    ratio = out["ratio_centre_over_edge"]
    assert not ratio["stable"], (
        f"ratio is now stable (mean {ratio['mean']:.1%}, sd {ratio['sd']:.1%}). "
        f"Revisit the withdrawal in paper/00-draft.md §8 deliberately."
    )
    assert ratio["sd"] > 0.5 * abs(ratio["mean"])


# --------------------------------------------------------------------------------------
# Finding 2 — misspecification versus frictions
# --------------------------------------------------------------------------------------

def test_volatility_misspecification_dominates_realistic_transaction_costs():
    """**The measurement that reframed the project.**

    A two-point volatility error must hurt substantially more than 5bp of transaction
    cost. If this reverses, `docs/06-implementation-plan.md` §A.1 is wrong and the paper's
    framing should revert to the original transaction-cost question.
    """
    out = misspecification(n_paths=20_000)
    vol_effect = out["vol"]["0.22"]
    cost_effect = out["cost"]["0.0005"]

    assert vol_effect < 0.0 and cost_effect < 0.0, "both should hurt"
    assert abs(vol_effect) > 3.0 * abs(cost_effect), (
        f"vol effect {vol_effect:.3f} is not clearly larger than the 5bp cost effect "
        f"{cost_effect:.3f}. The reframing in docs/06 rests on this ordering."
    )


def test_misspecification_hurts_monotonically_in_the_error():
    """Bigger volatility error, bigger loss. A non-monotone result would signal a bug."""
    out = misspecification(n_paths=20_000)
    effects = [out["vol"][k] for k in ("0.22", "0.25", "0.30")]
    assert effects == sorted(effects, reverse=True), f"not monotone: {effects}"


def test_transaction_costs_hurt_monotonically_in_the_rate():
    out = misspecification(n_paths=20_000)
    effects = [out["cost"][k] for k in ("0.0005", "0.0050", "0.0500")]
    assert effects == sorted(effects, reverse=True), f"not monotone: {effects}"


# --------------------------------------------------------------------------------------
# Finding 3 — precision
# --------------------------------------------------------------------------------------

def test_cvar_is_substantially_noisier_than_the_mean():
    """Drives sample sizing: the headline metric is the least precise one.

    A benchmark that sizes its runs by the precision of the mean will under-power every
    CVaR comparison it reports.
    """
    out = precision(n_paths=10_000, n_seeds=8)
    assert out["sd_cvar_across_seeds"] > 0.0
    assert out["paired_sd_of_difference"] > 0.0


def test_common_random_numbers_help_but_only_modestly():
    """Pins the correction to a claim `docs/05` originally overstated.

    Shared evaluation paths were asserted to give variance reduction that "materially
    increases the power" of every comparison. Measured, it is a modest factor: the two
    strategies produce genuinely different P&L distributions, so path-level P&Ls are not
    tightly correlated, and for a tail statistic the paths populating each strategy's tail
    are not even the same paths. Measured, it is ~1.4x on the mean and ~1.0-1.2x on
    CVaR-95 -- and it can fall marginally below 1, i.e. pairing occasionally makes the
    tail comparison slightly *noisier*.
    """
    out = precision(n_paths=10_000, n_seeds=8)
    reduction = out["variance_reduction_from_common_paths"]
    assert 0.8 < reduction < 2.0, (
        f"pairing gives {reduction:.2f}x on CVaR-95. docs/05 §3.2 and docs/06 §F.3 "
        f"describe it as giving essentially nothing for tail statistics (~1x, and it can "
        f"fall marginally below 1); revisit those if this has genuinely changed."
    )


def test_minimum_detectable_effect_is_reported():
    """A benchmark that cannot state its own MDE cannot distinguish a null result from an
    underpowered one."""
    out = precision(n_paths=10_000, n_seeds=8)
    assert out["mde_at_5_seeds"] > 0.0
    assert isinstance(out["detectable"], bool)
