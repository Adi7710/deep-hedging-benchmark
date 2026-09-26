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
    reduction = out["sd_ratio_from_pairing_cvar"]  # an SD ratio, not a variance ratio
    assert 0.8 < reduction < 2.0, (
        f"pairing gives {reduction:.2f}x on CVaR-95. docs/05 §3.2 and docs/06 §F.3 "
        f"describe it as giving essentially nothing for tail statistics (~1x, and it can "
        f"fall marginally below 1); revisit those if this has genuinely changed."
    )


def test_resolution_is_reported():
    """A benchmark that cannot state its own resolution cannot distinguish a null result
    from an underpowered one."""
    out = precision(n_paths=10_000, n_seeds=8)
    assert out["ci_halfwidth_5_seeds"] > 0.0
    assert isinstance(out["detectable"], bool)


# --------------------------------------------------------------------------------------
# Finding 4 — the training-seed noise floor
# --------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def floor():
    from experiments.findings import noise_floor
    return noise_floor(n_seeds=4, n_gradient_steps=200, batch_size=128, n_eval=4_000)


def test_noise_floor_reports_a_positive_spread(floor):
    """Replicates must actually differ; a zero spread would mean the seed does nothing."""
    assert floor["cvar_95_sd"] > 0.0
    assert floor["cvar_95_min"] < floor["cvar_95_max"]


def test_ci_halfwidth_uses_the_t_critical_value(floor):
    """**Pins the first of two errors in this calculation.**

    An early version used a normal critical value of 2.0; at replicate counts a compute
    budget permits the t value is much larger -- t(4) = 2.776 is 39% above 2.0.
    """
    normal_approximation = 2.0 * floor["cvar_95_sd"] / np.sqrt(5)
    assert floor["ci_halfwidth"]["5"] > normal_approximation


def test_the_ci_halfwidth_is_not_an_80pct_power_mde():
    """**Pins the second error: a CI half-width detects an effect of its own size only
    about half the time.**

    A later version reported t * sd / sqrt(k) as a "minimum detectable effect" and
    concluded six seeds sufficed. That quantity is the half-width of a 95% CI; an effect
    exactly that large is detected with ~50% power, not the conventional 80%.
    """
    from experiments.findings import _power

    sd = 0.1218
    for k in (5, 6, 10):
        from scipy.stats import t as student_t
        halfwidth = student_t.ppf(0.975, k - 1) * sd / np.sqrt(k)
        assert 0.4 < _power(k, sd, halfwidth, "fixed") < 0.65


def test_power_reproduces_an_independent_calculation():
    """With the 8-replicate sd, 80% power needs 9 seeds vs a fixed comparator and 14 per
    arm learned-vs-learned; the sd's own 95% CI spans 6 to 29 seeds. Checked independently
    against scipy's noncentral t before being pinned here.

    The 30-replicate sd (0.1439, full scale) needs 12 and 20, with the sd's CI spanning 8
    to 19 -- checked by simulating 200,000 t-tests at each count (power 0.797 at 11 seeds,
    0.837 at 12; 0.819 at 20 per arm)."""
    from experiments.findings import _seed_power_report

    r = _seed_power_report(0.1218, 8, 0.1343)
    assert r["seeds_80pct_vs_fixed"] == 9
    assert r["seeds_80pct_per_arm"] == 14
    assert r["seeds_80pct_vs_fixed_sd_ci"] == [6, 29]

    r30 = _seed_power_report(0.14388, 30, 0.1343)
    assert r30["seeds_80pct_vs_fixed"] == 12
    assert r30["seeds_80pct_per_arm"] == 20
    assert r30["seeds_80pct_vs_fixed_sd_ci"] == [8, 19]


def test_learned_vs_learned_needs_more_seeds_than_vs_a_fixed_comparator(floor):
    """Two independently trained arms double the variance of the difference."""
    assert floor["seeds_80pct_per_arm"] > floor["seeds_80pct_vs_fixed"]


def test_power_rises_with_replicates(floor):
    powers = [floor["power_vs_fixed"][k] for k in ("5", "6", "8", "9", "10", "14", "20")]
    assert powers == sorted(powers)


# --------------------------------------------------------------------------------------
# Finding 5 — the misspecification ordering, swept
# --------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sweep():
    from experiments.findings import misspecification_sweep
    return misspecification_sweep(n_paths=8_000, n_seeds=2)


def test_adverse_vol_error_dominates_index_like_costs_in_every_contract(sweep):
    """The headline ordering, across strikes, maturities and rebalancing frequencies.

    A +10% relative volatility error must hurt more than 5bp of proportional cost in every
    contract. Measured at full scale: min 2.7x, median 4.2x, max 7.1x over 12 contracts.
    """
    assert sweep["adverse_vol_dominates_5bp_everywhere"]


def test_favourable_vol_error_is_a_gain(sweep):
    """The effect is directional, not 'risk'.

    A short-gamma hedger whose realised volatility comes in BELOW the hedging volatility
    collects the difference. Stating the ordering without the direction would overstate it.
    """
    assert sweep["favourable_vol_error_is_a_gain_everywhere"]


def test_the_ordering_flips_at_illiquid_underlying_costs(sweep):
    """Which failure mode dominates depends on the instrument's cost regime.

    At 50bp costs hurt more than a +10% volatility error in every contract. That is
    small- and micro-cap territory, not "single stocks": quoted half-spreads are ~1.6bp
    for a median S&P 500 constituent and ~0.16bp for E-mini futures (market_calibration).
    At 25bp the split falls exactly on rebalancing frequency: the volatility error wins
    every weekly contract and costs win every daily one. The ordering the paper reports is
    conditional on both the cost level and how often the hedger trades.
    """
    for row in sweep["rows"]:
        vol = abs(row["vol_effect"]["1.1"][0])
        cost_50 = abs(row["cost_effect"]["0.005"][0])
        cost_25 = abs(row["cost_effect"]["0.0025"][0])
        where = f"K={row['strike']} T={row['maturity']} {row['rebalancing']}"
        assert cost_50 > vol, f"{where}: 50bp cost {cost_50:.3f} <= vol effect {vol:.3f}"
        if row["rebalancing"] == "daily":
            assert cost_25 > vol, f"{where}: 25bp cost {cost_25:.3f} <= vol effect {vol:.3f}"
        else:
            assert cost_25 < vol, f"{where}: 25bp cost {cost_25:.3f} >= vol effect {vol:.3f}"


def test_daily_rebalancing_narrows_the_ratio(sweep):
    """More trades means more cost while the vol effect barely moves, so the ratio falls."""
    by_contract = {}
    for row in sweep["rows"]:
        ratio = abs(row["vol_effect"]["1.1"][0]) / abs(row["cost_effect"]["0.0005"][0])
        by_contract.setdefault((row["strike"], row["maturity"]), {})[row["rebalancing"]] = ratio
    for key, r in by_contract.items():
        assert r["daily"] < r["weekly"], f"{key}: daily {r['daily']:.1f} >= weekly {r['weekly']:.1f}"


# --------------------------------------------------------------------------------------
# Finding 10 — the friction ordering, decomposed
# --------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def mechanism():
    from experiments.findings import friction_mechanism
    return friction_mechanism(n_paths=8_000, n_seeds=1)


def test_mean_effects_match_closed_forms(mechanism):
    """An analytic check of the simulator and the scorekeeper, contract by contract.

    With mu = r = 0 the mean effect of hedging at the wrong volatility is exactly
    V(sigma_h) - V(sigma_r), and the mean cost is Leland's rebalancing cost plus opening
    and closing the hedge (full scale: within 0.3% and 3.5%). The prediction includes the
    terminal liquidation, so an accounting that dropped it would miss by about a third on
    the short-dated in-the-money contracts.
    """
    s = mechanism["summary"]
    assert s["max_rel_error_vol_mean"] < 0.03
    assert s["max_rel_error_cost_mean"] < 0.06


def test_the_tail_metric_amplifies_vol_error_more_than_cost(mechanism):
    """CVaR-95 is not a rescaled mean: in every contract it amplifies a volatility error
    (1.6-3.0x its mean effect at full scale) more than a proportional cost (1.1-2.1x)."""
    for r in mechanism["rows"]:
        five = r["cost_0.0005"]
        where = f"K={r['strike']} T={r['maturity']} {r['rebalancing']}"
        assert r["amplification_vol"] > five["amplification"] > 1.0, where


def test_which_friction_dominates_depends_on_the_risk_measure(mechanism):
    """At 25bp costs dominate the MEAN effect in 10 of 12 contracts, but the CVaR-95 effect
    only in the 6 daily ones. Which friction a benchmark finds first-order is a property
    of its risk measure as well as of the market."""
    d = mechanism["summary"]["cost_dominates"]["0.0025"]
    assert d["mean"] - d["cvar"] >= 3, d
