"""Regenerate every measured number quoted in the paper.

``paper/STRUCTURE.md`` §6 commits to "every number traces to a config plus a seed".
Measurements taken interactively do not satisfy that: they are unreproducible by a
reviewer and unverifiable by us after the fact. This module is the fix -- each finding is
a function, seeded through :mod:`dhbench.seeding`, printing the table that appears in the
paper.

Run one::

    python -m experiments.findings baseline

Run all and write a machine-readable record::

    python -m experiments.findings --all --json paper/findings.json

Numbers here are *load-bearing for the argument*, not diagnostics, so
``tests/test_findings.py`` pins the headline value of each against a tolerance. If a
refactor changes one, that is either a bug or a result -- and both need noticing.
"""

from __future__ import annotations

import argparse
import json
from typing import Callable

import numpy as np
import tensorflow as tf
from scipy.stats import t as student_t

from dhbench.baselines.bs_delta import bs_call_price, bs_delta, delta_hedge_positions
from dhbench.baselines.whalley_wilmott import band_hedge_positions, whalley_wilmott_band
from dhbench.objectives.cvar import cvar_empirical
from dhbench.pnl import terminal_pnl
from dhbench.seeding import make_generator
from dhbench.worlds.gbm import simulate_gbm

__all__ = ["FINDINGS", "cvar", "run_finding"]

# Reference contract. Fixed here rather than per-finding so the numbers are comparable
# across findings, which is the whole point of quoting them side by side.
S0 = STRIKE = 100.0
MATURITY = 1.0
RATE = 0.0
SIGMA = 0.2
N_STEPS = 50
RISK_AVERSION = 1.0


def cvar(pnl: np.ndarray, alpha: float = 0.95) -> float:
    """P&L-oriented CVaR: mean of the worst tail, **higher is better**.

    Exactly ``-cvar_empirical(pnl)``. One implementation, two orientations, and the
    relation stated rather than reimplemented -- a metric computed slightly differently in
    two places is the comparability bug this whole project exists to remove.

    The findings below are quoted in this orientation because they are differences in
    P&L; :mod:`dhbench.evaluation.metrics` reports the loss orientation, matching
    :class:`~dhbench.objectives.cvar.CVaRRisk`.
    """
    return -float(cvar_empirical(tf.constant(pnl), alpha))


def _run(sigma_realised, sigma_hedge, cost_rate, strategy, seed, n_paths):
    """One evaluation: simulate, hedge, price the premium at the HEDGING vol, tally P&L.

    The premium is set at ``sigma_hedge`` deliberately -- it is what the desk charged,
    which is what it believed. Setting it at the realised vol would hide the
    misspecification inside the premium and make the effect vanish by construction.
    """
    spot = simulate_gbm(
        n_paths, N_STEPS, S0, RATE, sigma_realised, MATURITY, make_generator(seed, "eval")
    )
    spot_np = spot.numpy()

    if strategy == "delta":
        pos = delta_hedge_positions(spot_np, STRIKE, MATURITY, RATE, sigma_hedge)
    elif strategy == "band":
        pos = band_hedge_positions(
            spot_np, STRIKE, MATURITY, RATE, sigma_hedge, cost_rate, RISK_AVERSION
        )
    elif strategy == "band_centre":
        pos = _band_to_centre(spot_np, sigma_hedge, cost_rate)
    else:
        raise ValueError(f"unknown strategy {strategy!r}")

    premium = float(bs_call_price(S0, STRIKE, MATURITY, RATE, sigma_hedge))
    return terminal_pnl(
        spot,
        tf.constant(pos, dtype=tf.float32),
        tf.maximum(spot[:, -1] - STRIKE, 0.0),
        cost_rate,
        premium,
    ).numpy()


def _band_to_centre(spot_paths, sigma, cost_rate):
    """The centre rule: on exit, trade back to delta_BS rather than to the nearest edge.

    Implemented here rather than in ``dhbench/`` on purpose -- it is a wrong strategy kept
    only to quantify how wrong, and it must not be importable as if it were a baseline.
    """
    n_steps = spot_paths.shape[1] - 1
    tau = MATURITY - np.linspace(0.0, MATURITY, n_steps + 1)[:-1]
    out = np.empty((spot_paths.shape[0], n_steps))
    held = np.zeros(spot_paths.shape[0])
    for i in range(n_steps):
        target = bs_delta(spot_paths[:, i], STRIKE, tau[i], RATE, sigma)
        width = whalley_wilmott_band(
            spot_paths[:, i], STRIKE, tau[i], RATE, sigma, cost_rate, RISK_AVERSION
        )
        held = np.where(np.abs(held - target) > width, target, held)  # <- to the CENTRE
        out[:, i] = held
    return out


# ======================================================================================
# Finding 1 -- the band baseline: nearest edge versus centre
# ======================================================================================

def baseline(n_paths: int = 20_000, n_seeds: int = 20) -> dict:
    """Rebalancing to the band centre recovers much less improvement than the edge rule.

    Supports §4 of the paper: a benchmark whose Whalley-Wilmott baseline uses the centre
    rule reports a bar much closer to naive delta hedging than it should, and therefore
    overstates any learned policy's advantage.

    **Reported across seeds, deliberately.** An earlier single-seed run of this finding
    gave "the centre rule captures 7% of the available improvement". That is a ratio of
    two differences each of order the CVaR noise floor (0.048), and across 20 seeds it
    has mean 24% with standard deviation 27% and a range spanning zero. The ratio is not
    an estimable quantity at this sample size and must not be quoted as a point estimate.

    The defensible statement is the paired difference in levels, which is stable and
    highly significant.
    """
    cost = 0.005
    per_seed = {
        name: np.array([
            cvar(_run(SIGMA, SIGMA, cost, name, s, n_paths)) for s in range(n_seeds)
        ])
        for name in ("delta", "band_centre", "band")
    }
    d, c, e = per_seed["delta"], per_seed["band_centre"], per_seed["band"]

    def paired(x, label):
        t = float(x.mean() / (x.std(ddof=1) / np.sqrt(n_seeds)))
        return {
            "label": label,
            "mean": float(x.mean()),
            "sd": float(x.std(ddof=1)),
            "t": t,
            "consistent": int((np.sign(x) == np.sign(x.mean())).sum()),
        }

    out = {
        "n_seeds": n_seeds,
        "edge_vs_delta": paired(e - d, "edge   vs delta "),
        "centre_vs_delta": paired(c - d, "centre vs delta "),
        "edge_vs_centre": paired(e - c, "edge   vs centre"),
    }
    ratio = (c - d) / (e - d)
    out["ratio_centre_over_edge"] = {
        "mean": float(ratio.mean()),
        "sd": float(ratio.std(ddof=1)),
        "min": float(ratio.min()),
        "max": float(ratio.max()),
        "stable": bool(abs(ratio.mean()) > 2 * ratio.std(ddof=1)),
    }

    print(f"\n  {n_seeds} seeds, paired (common paths within each seed)\n")
    print(f"  {'comparison':<18}{'mean':>9}{'sd':>9}{'t':>8}{'consistent':>13}")
    for key in ("edge_vs_delta", "centre_vs_delta", "edge_vs_centre"):
        r = out[key]
        print(f"  {r['label']:<18}{r['mean']:>+9.4f}{r['sd']:>9.4f}{r['t']:>8.2f}"
              f"{r['consistent']:>9}/{n_seeds}")
    rr = out["ratio_centre_over_edge"]
    print(f"\n  ratio centre/edge : mean {rr['mean']:.1%}  sd {rr['sd']:.1%}  "
          f"range [{rr['min']:.1%}, {rr['max']:.1%}]")
    print(f"  stable enough to quote as a point estimate: {rr['stable']}")
    print(f"\n  DEFENSIBLE: the edge rule delivers {out['edge_vs_centre']['mean']:.3f} more")
    print(f"  CVaR improvement than the centre rule, in "
          f"{out['edge_vs_centre']['consistent']}/{n_seeds} seeds (t="
          f"{out['edge_vs_centre']['t']:.1f}).")
    return out


# ======================================================================================
# Finding 2 -- misspecification versus frictions
# ======================================================================================

def misspecification(n_paths: int = 40_000) -> dict:
    """Which hurts more: transaction costs, or hedging at the wrong volatility?

    The measurement that reframed the project (docs/06 §A.1). Both effects are reported
    against the same zero-cost, correctly-specified baseline so they are directly
    comparable, which is the entire point.
    """
    base = cvar(_run(SIGMA, SIGMA, 0.0, "delta", seed=7, n_paths=n_paths))
    out = {"baseline_cvar95": base, "cost": {}, "vol": {}}

    print(f"\n  baseline (true vol, zero cost)          CVaR-95 = {base:8.3f}\n")
    for c in (0.0005, 0.005, 0.05):
        v = cvar(_run(SIGMA, SIGMA, c, "delta", seed=7, n_paths=n_paths))
        out["cost"][f"{c:.4f}"] = v - base
        print(f"  cost {c * 1e4:5.0f}bp, hedged at true vol      "
              f"CVaR-95 = {v:8.3f}   effect {v - base:+8.3f}")
    print()
    for sv in (0.22, 0.25, 0.30):
        v = cvar(_run(sv, SIGMA, 0.0, "delta", seed=7, n_paths=n_paths))
        out["vol"][f"{sv:.2f}"] = v - base
        print(f"  zero cost, realised {sv:.2f} vs hedged 0.20  "
              f"CVaR-95 = {v:8.3f}   effect {v - base:+8.3f}")

    out["ratio_vol22_over_cost5bp"] = out["vol"]["0.22"] / out["cost"]["0.0005"]
    print(f"\n  a 2-point vol error costs "
          f"{out['ratio_vol22_over_cost5bp']:.1f}x what 5bp of cost does")
    return out


# ======================================================================================
# Finding 3 -- estimator precision and detectable effect size
# ======================================================================================

def precision(n_paths: int = 20_000, n_seeds: int = 12) -> dict:
    """How precise is CVaR-95, and what does pairing on common paths buy?

    Supports §F of docs/06. A benchmark that cannot state its own resolution cannot
    distinguish a null result from an underpowered one.

    The pairing factor is a ratio of STANDARD DEVIATIONS of the band-minus-delta
    difference, unpaired over paired; square it for a variance ratio. It is reported for
    the mean P&L and for CVaR-95 because common random numbers help them very differently.
    """
    cost = 0.005
    runs = [
        (_run(SIGMA, SIGMA, cost, "delta", s, n_paths), _run(SIGMA, SIGMA, cost, "band", s, n_paths))
        for s in range(n_seeds)
    ]

    def spreads(stat):
        a = np.array([stat(delta_pnl) for delta_pnl, _ in runs])
        b = np.array([stat(band_pnl) for _, band_pnl in runs])
        return a, b, float(np.sqrt(a.var(ddof=1) + b.var(ddof=1))), float((b - a).std(ddof=1))

    a, b, unpaired, paired = spreads(cvar)
    _, _, unpaired_mean, paired_mean = spreads(np.mean)
    d = b - a
    halfwidth5 = float(student_t.ppf(0.975, 4) * paired / np.sqrt(5))

    out = {
        "sd_cvar_across_seeds": float(a.std(ddof=1)),
        "unpaired_sd_of_difference": unpaired,
        "paired_sd_of_difference": paired,
        "sd_ratio_from_pairing_cvar": unpaired / paired,
        "sd_ratio_from_pairing_mean": unpaired_mean / paired_mean,
        "effect": float(d.mean()),
        "ci_halfwidth_5_seeds": halfwidth5,
        "detectable": bool(abs(d.mean()) > halfwidth5),
    }
    print(f"\n  SD of CVaR-95 across {n_seeds} seeds        : {out['sd_cvar_across_seeds']:.4f}")
    print(f"  unpaired SD of the difference       : {unpaired:.4f}")
    print(f"  paired SD (common random numbers)   : {paired:.4f}")
    print(f"  SD ratio from pairing, CVaR-95      : {out['sd_ratio_from_pairing_cvar']:.2f}x"
          f"   (NOT an order of magnitude)")
    print(f"  SD ratio from pairing, mean P&L     : {out['sd_ratio_from_pairing_mean']:.2f}x")
    print(f"\n  measured band-vs-delta effect       : {out['effect']:+.4f}")
    print(f"  95% CI half-width at 5 seeds (~50% power): {halfwidth5:.4f}")
    print(f"  detectable at 5 seeds               : {out['detectable']}")
    return out


# ======================================================================================
# Finding 4 -- the training-seed noise floor
# ======================================================================================

def noise_floor(
    n_seeds: int = 30,
    n_gradient_steps: int = 2_000,
    batch_size: int = 512,
    n_eval: int = 20_000,
    effect: float = 0.1343,
) -> dict:
    """How much does a *trained* policy's score vary from the training seed alone?

    Step 6 of ``docs/05-stage-1-2-plan.md``, and it is scheduled before any method
    comparison on purpose. This spread is the **resolution limit of every result in the
    paper**: a difference between two methods smaller than it is not a finding, whatever
    the point estimates say.

    Two sources of variation are deliberately separated. Each replicate gets its own
    weight initialisation *and* its own training paths, because that is what a replicate
    means in practice. But every replicate is scored on the **same** evaluation paths, so
    what is measured here is training variability alone and not evaluation noise on top.

    Reported, and not to be confused:

      ci_halfwidth   t_{0.975,k-1} * sd / sqrt(k): the half-width of a 95% CI for a mean
                     difference. An effect this size is detected only ~50% of the time.
      power          from the noncentral t, two-sided alpha = 0.05, for an effect of
                     ``effect`` (default: band-vs-delta, findings.baseline). Two designs:
                     vs a FIXED comparator (one-sample t on k differences) and learned vs
                     learned (independent training, two-sample t, k per arm).
      sd CI          chi-square 95% interval for the noise-floor sd itself -- the seed
                     requirement inherits its uncertainty.

    Correction history, kept because each is instructive: an early version used a normal
    critical value (2.0) and concluded four seeds sufficed; a later version called the CI
    half-width an "MDE", which quietly assumed 50% power and implied six seeds; and it
    estimated sd from only 8 replicates, whose 95% CI [0.081, 0.248] spans 6 to 29 seeds.
    Hence 30 replicates by default, and power stated explicitly. At 30 replicates the sd
    is 0.144 [0.115, 0.193]: 12 seeds for 80% power against a fixed comparator (8 to 19
    across the sd's CI) and 20 per arm learned against learned; five seeds give 36%.

    The sd assumes a well-behaved replicate distribution, so two diagnostics sit beside
    it: a robust spread (IQR / 1.349) with robust z-scores flagging outlying replicates,
    and each run's loss history (``_loss_history_diagnostics``), which tells a run that
    DESTABILISED from one that was merely unlucky. At 8,000 steps (``noise_floor_long``)
    one replicate in 30 did: its training loss rose from 0.65 to 1.4 after step 3,000
    and never recovered, and it alone more than doubles the sd. The project's first
    convergence check (last decile below first) passes that run; final-versus-best-window
    regression flags it. Flagged runs are reported, never dropped.
    """
    from dhbench.agents.feedforward import FeedforwardAgent
    from dhbench.objectives.entropic import EntropicRisk
    from dhbench.seeding import seed_keras
    from dhbench.training import evaluate, train

    def world(n_paths, generator):
        return simulate_gbm(
            n_paths, N_STEPS, S0, RATE, SIGMA, MATURITY, generator
        )

    def payoff(spot):
        return tf.maximum(spot[:, -1] - STRIKE, 0.0)

    premium = float(bs_call_price(S0, STRIKE, MATURITY, RATE, SIGMA))
    scores, histories = [], []
    for replicate in range(n_seeds):
        seed_keras(replicate, "init")
        agent = FeedforwardAgent((32, 32))
        result = train(
            agent, EntropicRisk(RISK_AVERSION), world, payoff,
            maturity=MATURITY, strike=STRIKE, premium=premium,
            batch_size=batch_size, n_gradient_steps=n_gradient_steps,
            seed=replicate, stream="train",
        )
        histories.append(_loss_history_diagnostics(result.losses))
        # SAME evaluation paths for every replicate: isolates training variability.
        scores.append(
            evaluate(
                agent, world, payoff, maturity=MATURITY, strike=STRIKE,
                premium=premium, n_paths=n_eval, seed=0, stream="eval",
            )
        )

    cvars = np.array([s["cvar_95"] for s in scores])
    sd = float(cvars.std(ddof=1))
    shape = _replicate_distribution(cvars)
    regression = np.array([h["regression_from_best"] for h in histories])
    out = {**_seed_power_report(sd, n_seeds, effect),
           **shape,
           "n_gradient_steps": n_gradient_steps,
           "cvar_95_mean": float(cvars.mean()),
           "cvar_95_min": float(cvars.min()),
           "cvar_95_max": float(cvars.max()),
           "cvar_95_all": [float(c) for c in cvars],
           "loss_history": histories}
    order = np.argsort(regression)[::-1][:3]
    print(f"\n  replicate distribution: median {shape['cvar_95_median']:.4f}, robust sd (IQR/1.349) "
          f"{shape['cvar_95_robust_sd']:.4f}; outlying replicates (|robust z| > 5): "
          + (", ".join(f"#{i} (z = {z:+.1f})" for i, z in shape["outliers"]) or "none"))
    print("  largest final-vs-best-window loss regressions: "
          + ", ".join(f"#{i} {regression[i]:+.0%}" for i in order)
          + f"; last decile below first in {sum(h['last_decile_beats_first'] for h in histories)}"
          f"/{n_seeds} runs")
    return out


def _loss_history_diagnostics(losses: list[float], n_windows: int = 16) -> dict:
    """Did a run converge, or rise again after finding a good region?

    The loss history is cut into ``n_windows`` equal windows. ``regression_from_best`` is
    final-window mean over best-window mean, minus one: 0 for a run still at its best, and
    large for a run that found a good region and left it. ``last_decile_beats_first`` is the
    weaker check the project used first -- it passes a run that improved from its
    initialisation spike and then regressed, which is exactly the case that matters.
    """
    arr = np.asarray(losses, dtype=float)
    usable = arr.size // n_windows * n_windows
    windows = arr[:usable].reshape(n_windows, -1).mean(axis=1)
    tenth = max(1, arr.size // 10)
    return {"final_window_loss": float(windows[-1]),
            "best_window_loss": float(windows.min()),
            "best_window": int(windows.argmin()),
            "regression_from_best": float(windows[-1] / windows.min() - 1.0),
            "last_decile_beats_first": bool(arr[-tenth:].mean() < arr[:tenth].mean())}


def _replicate_distribution(x: np.ndarray, z_flag: float = 5.0) -> dict:
    """Median, robust spread and outlying replicates of a set of replicate scores.

    Robust z uses the median absolute deviation scaled by 1.4826, so one extreme run
    cannot hide itself by inflating the yardstick it is measured against. The sd without
    flagged runs is a DIAGNOSTIC of how much they carry, not a replacement for the sd.

    A flag is a screen, not a verdict: a MAD from ~30 runs is itself noisy, and a random
    normal sample of 30 can put a legitimate run near z = 5. A flagged run is explained by
    its loss history (``_loss_history_diagnostics``) or not at all.
    """
    median = float(np.median(x))
    mad_sd = float(1.4826 * np.median(np.abs(x - median)))
    q1, q3 = np.percentile(x, [25, 75])
    z = (x - median) / mad_sd if mad_sd > 0 else np.zeros_like(x)
    flagged = [(int(i), float(z[i])) for i in np.flatnonzero(np.abs(z) > z_flag)]
    kept = np.delete(x, [i for i, _ in flagged])
    return {"cvar_95_median": median,
            "cvar_95_robust_sd": float((q3 - q1) / 1.349),
            "outliers": flagged,
            "cvar_95_sd_without_outliers": float(kept.std(ddof=1)) if kept.size > 1 else None}


def noise_floor_long() -> dict:
    """The noise floor at 8,000 gradient steps: does longer training shrink it?

    Same replicates, same evaluation paths, four times the training budget. The typical
    spread shrinks by more than half, but one run in 30 destabilises; the two numbers
    that matter are then the robust spread and the failure count, not the sd alone.
    """
    return noise_floor(n_gradient_steps=8_000)


def _power(k: int, sd: float, effect: float, design: str, alpha: float = 0.05) -> float:
    """Two-sided t-test power from the noncentral t distribution.

    design "fixed": one-sample t on k differences against a deterministic comparator.
    design "two_arm": independent training runs, two-sample t with k per arm.
    """
    from scipy.stats import nct

    if design == "fixed":
        df, scale = k - 1, sd / np.sqrt(k)
    else:
        df, scale = 2 * k - 2, sd * np.sqrt(2.0 / k)
    crit = float(student_t.ppf(1 - alpha / 2, df))
    ncp = effect / scale
    return float(1 - nct.cdf(crit, df, ncp) + nct.cdf(-crit, df, ncp))


def _seeds_for_power(sd: float, effect: float, design: str, target: float = 0.8) -> int | None:
    return next((k for k in range(3, 2_000) if _power(k, sd, effect, design) >= target), None)


def _seed_power_report(sd: float, n_used: int, effect: float) -> dict:
    """Seed-count requirements implied by a measured noise-floor sd, with its uncertainty."""
    from scipy.stats import chi2

    df = n_used - 1
    sd_lo = float(sd * np.sqrt(df / chi2.ppf(0.975, df)))
    sd_hi = float(sd * np.sqrt(df / chi2.ppf(0.025, df)))
    halfwidth = {k: float(student_t.ppf(0.975, k - 1) * sd / np.sqrt(k)) for k in (5, 6, 10)}
    power_fixed = {k: _power(k, sd, effect, "fixed") for k in (5, 6, 8, 9, 10, 14, 20)}
    power_two = {k: _power(k, sd, effect, "two_arm") for k in (5, 6, 8, 9, 10, 14, 20)}
    out = {
        "n_seeds": n_used, "cvar_95_sd": sd, "cvar_95_sd_ci95": [sd_lo, sd_hi],
        "effect": effect,
        "ci_halfwidth": {str(k): v for k, v in halfwidth.items()},
        "power_vs_fixed": {str(k): v for k, v in power_fixed.items()},
        "power_learned_vs_learned": {str(k): v for k, v in power_two.items()},
        "seeds_80pct_vs_fixed": _seeds_for_power(sd, effect, "fixed"),
        "seeds_80pct_per_arm": _seeds_for_power(sd, effect, "two_arm"),
        "seeds_80pct_vs_fixed_sd_ci": [_seeds_for_power(sd_lo, effect, "fixed"),
                                       _seeds_for_power(sd_hi, effect, "fixed")],
    }
    print(f"\n  noise floor from {n_used} replicates: sd {sd:.4f}, 95% CI [{sd_lo:.4f}, {sd_hi:.4f}]")
    print(f"  effect to detect: {effect:.4f} (band vs delta)")
    print("\n  95% CI half-width (detects an effect this size only ~50% of the time):")
    print("   " + "   ".join(f"k={k}: {v:.4f}" for k, v in halfwidth.items()))
    print("\n  power, two-sided alpha 0.05 (noncentral t)")
    print(f"  {'k':>4}{'vs fixed comparator':>22}{'learned vs learned':>21}")
    for k in power_fixed:
        print(f"  {k:>4}{power_fixed[k]:>22.2f}{power_two[k]:>21.2f}")
    lo_n, hi_n = out["seeds_80pct_vs_fixed_sd_ci"]
    print(f"\n  80% power: {out['seeds_80pct_vs_fixed']} seeds vs a fixed comparator "
          f"(sd CI implies {lo_n} to {hi_n}); {out['seeds_80pct_per_arm']} per arm learned vs learned")
    return out


# ======================================================================================
# Finding 5 -- is the misspecification ordering robust? Swept, both directions
# ======================================================================================

def misspecification_sweep(n_paths: int = 40_000, n_seeds: int = 3) -> dict:
    """Does volatility error dominate transaction cost across contracts -- and which way?

    ``misspecification`` measured the ordering on ONE contract in ONE direction. This
    sweeps strike (90/100/110), maturity (3m, 1y) and rebalancing (weekly, daily), and
    measures volatility error in BOTH directions, because the effect is not symmetric: a
    short-gamma hedger whose realised volatility comes in *below* the hedging volatility
    collects the difference. Only the adverse direction is a risk.

    Design: within each (contract, seed), every variant is driven by the SAME normal draws
    -- common random numbers across volatility levels and cost rates -- so each effect is a
    paired difference against the correctly specified, zero-cost delta hedge. The premium
    is charged at the hedging volatility, which is what the desk believed; charging it at
    the realised volatility would hide the misspecification inside the premium.

    Effects are CVaR-95 in P&L orientation: negative means worse.
    """
    sigma_h, s0 = 0.2, 100.0
    strikes = (90.0, 100.0, 110.0)
    maturities = (0.25, 1.0)
    frequencies = {"weekly": 52, "daily": 252}
    vol_ratios = (0.8, 0.9, 1.1, 1.25)
    costs = (0.0001, 0.0005, 0.0025, 0.005)

    def evaluate_contract(strike, maturity, n_steps, seed):
        premium = float(bs_call_price(s0, strike, maturity, RATE, sigma_h))

        def pnl_of(sigma_r, cost):
            spot = simulate_gbm(
                n_paths, n_steps, s0, RATE, sigma_r, maturity,
                make_generator(seed, "sweep"),        # same draws for every variant
            )
            delta = delta_hedge_positions(spot.numpy(), strike, maturity, RATE, sigma_h)
            return terminal_pnl(
                spot, tf.constant(delta, tf.float32),
                tf.maximum(spot[:, -1] - strike, 0.0), cost, premium,
            ).numpy()

        base = cvar(pnl_of(sigma_h, 0.0))
        return (
            {rho: cvar(pnl_of(rho * sigma_h, 0.0)) - base for rho in vol_ratios},
            {c: cvar(pnl_of(sigma_h, c)) - base for c in costs},
        )

    rows = []
    for strike in strikes:
        for maturity in maturities:
            for freq, per_year in frequencies.items():
                n_steps = max(2, round(maturity * per_year))
                per_seed = [evaluate_contract(strike, maturity, n_steps, s)
                            for s in range(n_seeds)]
                vol = {rho: np.array([p[0][rho] for p in per_seed]) for rho in vol_ratios}
                cst = {c: np.array([p[1][c] for p in per_seed]) for c in costs}
                rows.append({
                    "strike": strike, "maturity": maturity, "rebalancing": freq,
                    "n_steps": n_steps,
                    "vol_effect": {f"{r}": [float(vol[r].mean()), float(vol[r].std(ddof=1))]
                                   for r in vol_ratios},
                    "cost_effect": {f"{c}": [float(cst[c].mean()), float(cst[c].std(ddof=1))]
                                    for c in costs},
                })

    def effect(row, kind, key):
        return row[f"{kind}_effect"][key][0]

    ratio_adverse = [abs(effect(r, "vol", "1.1")) / abs(effect(r, "cost", "0.0005"))
                     for r in rows]
    favourable_is_gain = [effect(r, "vol", "0.9") > 0 for r in rows]

    out = {
        "n_contracts": len(rows),
        "n_seeds": n_seeds,
        "rows": rows,
        "ratio_vol10pct_over_cost5bp": {
            "min": float(min(ratio_adverse)), "median": float(np.median(ratio_adverse)),
            "max": float(max(ratio_adverse)),
        },
        "adverse_vol_dominates_5bp_everywhere": bool(min(ratio_adverse) > 1.0),
        "favourable_vol_error_is_a_gain_everywhere": bool(all(favourable_is_gain)),
    }

    print(f"\n  {len(rows)} contracts x {n_seeds} seeds, {n_paths:,} paths, paired draws")
    print(f"  effect on CVaR-95 (P&L orientation, negative = worse) vs correct-vol zero-cost\n")
    print(f"  {'K':>5}{'T':>6}{'rebal':>8} |{'vol x0.9':>10}{'vol x1.1':>10}{'vol x1.25':>10}"
          f" |{'1bp':>8}{'5bp':>8}{'25bp':>8}{'50bp':>8} | ratio")
    for r, ratio in zip(rows, ratio_adverse):
        v = [effect(r, "vol", k) for k in ("0.9", "1.1", "1.25")]
        c = [effect(r, "cost", k) for k in ("0.0001", "0.0005", "0.0025", "0.005")]
        print(f"  {r['strike']:>5.0f}{r['maturity']:>6.2f}{r['rebalancing']:>8} |"
              + "".join(f"{x:>+10.3f}" for x in v) + " |"
              + "".join(f"{x:>+8.3f}" for x in c) + f" | {ratio:5.1f}x")
    q = out["ratio_vol10pct_over_cost5bp"]
    print(f"\n  |effect of +10% vol| / |effect of 5bp cost|:  "
          f"min {q['min']:.1f}x   median {q['median']:.1f}x   max {q['max']:.1f}x")
    print(f"  adverse vol error dominates 5bp in every contract: "
          f"{out['adverse_vol_dominates_5bp_everywhere']}")
    print(f"  favourable vol error (x0.9) is a GAIN in every contract: "
          f"{out['favourable_vol_error_is_a_gain_everywhere']}")
    return out


# ======================================================================================
# Finding 6 -- pseudo-replication: the mechanism and its consequence, regenerable
# ======================================================================================

def seeding(n_paths: int = 200_000, n_replicates: int = 20) -> dict:
    """Consecutive integer seeds are shifted copies of one stream -- and what that costs.

    Mechanism: ``tf.random.Generator.from_seed(k)`` writes ``k`` into the Philox counter
    with the key fixed at zero, so seed ``k`` reproduces seed 0 offset by ``4k`` draws.
    Consequence: replicates built that way re-use the same random numbers, their dispersion
    collapses, and an ordinary fluctuation is reported as highly significant.

    Two standard errors are reported and they must not be confused:

      proper SE   the sampling error of ONE n_paths estimate (payoff sd / sqrt(n_paths))
      replicate   the standard error of the replicate mean, sd_across / sqrt(n_replicates),
      SEM         which is only valid if the replicates are independent

    Hashed replicates use ``make_generator(k)`` (default stream), matching the original fix.
    """
    truth = float(bs_call_price(S0, STRIKE, MATURITY, RATE, SIGMA))

    states = {k: tf.random.Generator.from_seed(k).state.numpy().tolist() for k in (0, 1, 2, 7)}
    base = tf.random.Generator.from_seed(0).normal((4_000,)).numpy()
    shifts = {}
    for k in (1, 2, 3):
        other = tf.random.Generator.from_seed(k).normal((4_000,)).numpy()
        shifts[k] = next(
            (lag for lag in range(64) if np.array_equal(base[lag:lag + 3_000], other[:3_000])),
            None,
        )
    a = tf.random.Generator.from_seed(0).normal((200_000,)).numpy()
    b = tf.random.Generator.from_seed(1).normal((200_000,)).numpy()
    shared = float(np.mean(a[4:] == b[:-4]))
    lag0 = float(np.corrcoef(a, b)[0, 1])

    # The inconsistency: a LIST seed is left-padded into the KEY, as a comment in
    # stateful_random_ops._make_1d_state intends ("Padding with zeros on the *left* ...
    # Padding on the right would cause a small seed to be used as the 'counter' while the
    # 'key' is always zero ... two RNGs with two different small seeds may generate
    # overlapping outputs"). An INT seed is first chopped into state_size little-endian
    # chunks, so it arrives full length, padding_size is 0, and the protection never fires.
    list_states = {k: tf.random.Generator.from_seed([k]).state.numpy().tolist() for k in (1, 2)}
    la = tf.random.Generator.from_seed([0]).normal((4_000,)).numpy()
    lb = tf.random.Generator.from_seed([1]).normal((4_000,)).numpy()
    list_overlap = any(np.array_equal(la[lag:lag + 3_000], lb[:3_000]) for lag in range(64))

    def replicate_prices(make):
        prices, proper_se, log_st = [], None, []
        for k in range(n_replicates):
            paths = simulate_gbm(n_paths, N_STEPS, S0, RATE, SIGMA, MATURITY, make(k))
            payoff = tf.maximum(paths[:, -1] - STRIKE, 0.0)
            prices.append(float(tf.reduce_mean(payoff)))
            if k == 0:
                proper_se = float(tf.math.reduce_std(payoff)) / np.sqrt(n_paths)
            if k < 2:
                log_st.append(np.log(paths[:, -1].numpy().astype(np.float64) / S0))
        return np.array(prices), proper_se, float(np.corrcoef(*log_st)[0, 1])

    def summarise(prices, proper_se, path_corr):
        sd = float(prices.std(ddof=1))
        sem = sd / np.sqrt(n_replicates)
        dev = float(prices.mean() - truth)
        return {
            "mean": float(prices.mean()), "deviation": dev, "sd_across": sd,
            "proper_se": proper_se, "replicate_sem": sem,
            "deviation_in_proper_se": dev / proper_se,
            "deviation_in_replicate_sem": dev / sem,
            "dispersion_ratio": sd / proper_se,
            "above_truth": int((prices > truth).sum()),
            "path_corr_replicate_0_vs_1": path_corr,
            "first_replicate": float(prices[0]),
        }

    raw = summarise(*replicate_prices(lambda k: tf.random.Generator.from_seed(k)))
    hashed = summarise(*replicate_prices(lambda k: make_generator(k)))

    out = {
        "truth": truth, "n_paths": n_paths, "n_steps": N_STEPS, "n_replicates": n_replicates,
        "mechanism": {"states": {str(k): v for k, v in states.items()},
                      "shift_vs_seed0": {str(k): v for k, v in shifts.items()},
                      "share_of_seed1_draws_that_are_seed0_draws": shared,
                      "lag0_correlation": lag0,
                      "list_seed_states": {str(k): v for k, v in list_states.items()},
                      "list_seeds_overlap": bool(list_overlap),
                      "tensorflow_version": tf.__version__},
        "from_seed": raw, "hashed": hashed,
    }

    print("\n  mechanism")
    for k, v in states.items():
        print(f"    from_seed({k}).state = {v}")
    for k, v in shifts.items():
        print(f"    from_seed({k}) == from_seed(0) shifted by {v} draws")
    print(f"    share of seed 1's 200k draws that are seed 0's draws: {shared:.6f}")
    print(f"    lag-0 correlation: {lag0:+.4f}")
    for k, v in list_states.items():
        print(f"    from_seed([{k}]).state = {v}   <- a LIST seed lands in the key")
    print(f"    list seeds [0] and [1] overlap at a small shift: {list_overlap}")
    print(f"\n  consequence: ATM call, truth {truth:.5f}, {n_paths:,} paths x {n_replicates}")
    print(f"  {'':<22}{'mean':>9}{'dev':>9}{'sd':>8}{'proper SE':>11}{'SEM':>9}"
          f"{'dev/SE':>8}{'dev/SEM':>9}{'disp':>7}{'above':>7}{'corr':>7}")
    for name, r in (("from_seed(0..19)", raw), ("hashed", hashed)):
        print(f"  {name:<22}{r['mean']:>9.5f}{r['deviation']:>+9.5f}{r['sd_across']:>8.5f}"
              f"{r['proper_se']:>11.5f}{r['replicate_sem']:>9.5f}"
              f"{r['deviation_in_proper_se']:>+8.2f}{r['deviation_in_replicate_sem']:>+9.2f}"
              f"{r['dispersion_ratio']:>7.2f}{r['above_truth']:>5}/{n_replicates}"
              f"{r['path_corr_replicate_0_vs_1']:>7.3f}")
    return out


# ======================================================================================
# Finding 7 -- estimator precision by path count, from enough replicates to trust
# ======================================================================================

def precision_by_n(n_replicates: int = 100, sizes: tuple[int, ...] = (5_000, 20_000, 100_000)) -> dict:
    """Standard error of mean P&L versus CVaR-95 at equal path count.

    Supersedes two measurements that had too few replicates to estimate a standard
    deviation. With 8, SE(mean) barely fell from 5,000 to 20,000 paths (0.0078 vs 0.0071)
    when theory says it should halve. With 32, SE(mean) at 5,000 paths came out at 0.72x
    its theoretical value -- a 2.2-sigma fluctuation, since an sd from 32 replicates is
    only good to ~13% -- which inflated the CVaR/mean ratio to 5.7x; 200 replicates gave
    1.00x and 4.3x. So the ratio is taken against the THEORETICAL SE of the mean, sd/sqrt(N)
    pooled over replicates, and the replicate-based SE of CVaR-95 carries its chi-square CI.
    """
    from scipy.stats import chi2

    df = n_replicates - 1
    lo_f, hi_f = np.sqrt(df / chi2.ppf(0.975, df)), np.sqrt(df / chi2.ppf(0.025, df))
    out = {"n_replicates": n_replicates, "rows": []}
    print(f"\n  zero-cost delta hedge, ATM 1y, {N_STEPS} steps, {n_replicates} independent replicates\n")
    print(f"  {'N':>8}{'SE(mean)':>11}{'theory':>9}{'obs/th':>8}{'SE(CVaR95)':>12}{'95% CI':>18}{'CVaR/theory':>13}")
    for n in sizes:
        means, cvars, sds = [], [], []
        for k in range(n_replicates):
            pnl = _run(SIGMA, SIGMA, 0.0, "delta", k, n)
            means.append(float(pnl.mean()))
            cvars.append(cvar(pnl))
            sds.append(float(pnl.std()))
        theory = float(np.mean(sds)) / np.sqrt(n)
        se_mean = float(np.std(means, ddof=1))
        se_cvar = float(np.std(cvars, ddof=1))
        row = {"n_paths": n, "se_mean": se_mean, "se_mean_theory": theory,
               "se_cvar95": se_cvar, "se_cvar95_ci95": [se_cvar * lo_f, se_cvar * hi_f],
               "ratio_cvar_to_mean_theory": se_cvar / theory}
        out["rows"].append(row)
        print(f"  {n:>8,}{se_mean:>11.5f}{theory:>9.5f}{se_mean / theory:>8.2f}{se_cvar:>12.5f}"
              f"   [{se_cvar * lo_f:.4f}, {se_cvar * hi_f:.4f}]{se_cvar / theory:>12.2f}x")
    return out


# ======================================================================================
# Finding 8 -- rung 2: the n^{-1/2} law, regenerable
# ======================================================================================

def convergence(n_paths: int = 40_000, step_counts: tuple[int, ...] = (10, 40, 160, 640)) -> dict:
    """Zero-cost delta-hedging error std should scale as n_steps^{-1/2}.

    The joint check on simulator, pricing reference and accounting: std(PL) * sqrt(n)
    must stay roughly constant, and mean P&L near zero when the premium is fair.
    """
    premium = float(bs_call_price(S0, STRIKE, MATURITY, RATE, SIGMA))
    out = {"n_paths": n_paths, "rows": []}
    print(f"\n  {'n_steps':>8}{'mean PL':>10}{'std PL':>9}{'std*sqrt(n)':>13}")
    for n in step_counts:
        spot = simulate_gbm(n_paths, n, S0, RATE, SIGMA, MATURITY, make_generator(n, "convergence"))
        delta = delta_hedge_positions(spot.numpy(), STRIKE, MATURITY, RATE, SIGMA)
        pnl = terminal_pnl(spot, tf.constant(delta, tf.float32),
                           tf.maximum(spot[:, -1] - STRIKE, 0.0), 0.0, premium).numpy()
        row = {"n_steps": n, "mean": float(pnl.mean()), "std": float(pnl.std()),
               "std_sqrt_n": float(pnl.std() * np.sqrt(n))}
        out["rows"].append(row)
        print(f"  {n:>8}{row['mean']:>+10.4f}{row['std']:>9.4f}{row['std_sqrt_n']:>13.3f}")
    return out


# ======================================================================================
# Finding 9 -- an underspecified acceptance gate
# ======================================================================================

def evaluation_design(n_seeds: int = 6, n_gradient_steps: int = 8_000) -> dict:
    """Does a pre-registered accuracy gate's verdict depend on unreported evaluation choices?

    Each replicate trains a feedforward policy under GBM at zero cost (own init, own
    training paths). The SAME trained policy is then scored against Phi(d1) under three
    designs over one region R (moneyness 0.8-1.2, tau 0.1-1.0):

      grid_prev0    constructed grid, previous-position input = 0
      grid_prevPhi  constructed grid, previous-position input = Phi(d1)
      visited       states visited on the policy's own rollout, inside R

    Gate (rung 4, as written): MAD < 0.05 and max abs deviation < 0.15 on R. It specifies
    neither the conditioning input nor the weighting over states.

    Invariance diagnostic: at zero cost the optimal hedge is Markov in (t, S), so its output
    cannot depend on the previous position. We move ONLY that input over {0,.25,.5,.75,1} at
    each grid point and report the output range. It should be zero.
    """
    from dhbench.agents.feedforward import FeedforwardAgent
    from dhbench.objectives.entropic import EntropicRisk
    from dhbench.seeding import seed_keras
    from dhbench.training import train

    n_steps, mad_gate, max_gate = 30, 0.05, 0.15

    def world(n, g):
        return simulate_gbm(n, n_steps, S0, RATE, SIGMA, MATURITY, g)

    def payoff(s):
        return tf.maximum(s[:, -1] - STRIKE, 0.0)

    m_grid, t_grid = np.linspace(0.8, 1.2, 17), np.linspace(0.1, 1.0, 10)
    mm, tt = (a.ravel() for a in np.meshgrid(m_grid, t_grid, indexing="ij"))
    truth = bs_delta(mm * STRIKE, STRIKE, tt, RATE, SIGMA)
    premium = float(bs_call_price(S0, STRIKE, MATURITY, RATE, SIGMA))

    spot = simulate_gbm(20_000, n_steps, S0, RATE, SIGMA, MATURITY, make_generator(0, "eval"))
    s = spot.numpy()[:, :-1]
    tau = np.broadcast_to(MATURITY - np.linspace(0, MATURITY, n_steps + 1)[:-1], s.shape)
    in_r = (s / STRIKE >= 0.8) & (s / STRIKE <= 1.2) & (tau >= 0.1 - 1e-9)
    truth_v = bs_delta(s, STRIKE, tau, RATE, SIGMA)

    rows = []
    for seed in range(n_seeds):
        seed_keras(seed, "init")
        agent = FeedforwardAgent((32, 32))
        train(agent, EntropicRisk(RISK_AVERSION), world, payoff, maturity=MATURITY,
              strike=STRIKE, premium=premium, batch_size=512,
              n_gradient_steps=n_gradient_steps, learning_rate=5e-3, seed=seed)

        def grid(prev):
            feats = np.stack([tt / MATURITY, mm, prev], axis=-1).astype("float32")
            return agent(tf.constant(feats)).numpy()

        errs = {
            "grid_prev0": np.abs(grid(np.zeros_like(mm)) - truth),
            "grid_prevPhi": np.abs(grid(truth) - truth),
            "visited": np.abs(agent.hedge_path(spot, MATURITY, STRIKE).numpy() - truth_v)[in_r],
        }
        outputs = np.stack([grid(np.full_like(mm, p)) for p in (0.0, 0.25, 0.5, 0.75, 1.0)])
        spread = outputs.max(axis=0) - outputs.min(axis=0)
        row = {"seed": seed, "prev_sensitivity_mean": float(spread.mean()),
               "prev_sensitivity_max": float(spread.max())}
        for name, e in errs.items():
            row[name] = {"mad": float(e.mean()), "max": float(e.max()),
                         "passes_gate": bool(e.mean() < mad_gate and e.max() < max_gate)}
        rows.append(row)

    summary = {}
    for name in ("grid_prev0", "grid_prevPhi", "visited"):
        mads = np.array([r[name]["mad"] for r in rows])
        maxs = np.array([r[name]["max"] for r in rows])
        summary[name] = {"mad_mean": float(mads.mean()), "mad_sd": float(mads.std(ddof=1)),
                         "max_mean": float(maxs.mean()), "max_sd": float(maxs.std(ddof=1)),
                         "gate_passes": int(sum(r[name]["passes_gate"] for r in rows))}
    sens = np.array([r["prev_sensitivity_mean"] for r in rows])
    summary["prev_sensitivity"] = {"mean": float(sens.mean()), "sd": float(sens.std(ddof=1))}
    out = {"n_seeds": n_seeds, "n_gradient_steps": n_gradient_steps,
           "gate": {"mad": mad_gate, "max": max_gate}, "rows": rows, "summary": summary}

    print(f"\n  {n_seeds} replicates x {n_gradient_steps} gradient steps; gate MAD<{mad_gate} and max<{max_gate}\n")
    print(f"  {'design':<14}{'MAD mean +- sd':>18}{'max mean +- sd':>18}{'passes':>9}")
    for name in ("grid_prev0", "grid_prevPhi", "visited"):
        q = summary[name]
        print(f"  {name:<14}{q['mad_mean']:>9.4f} +- {q['mad_sd']:.4f}"
              f"{q['max_mean']:>9.4f} +- {q['max_sd']:.4f}{q['gate_passes']:>6}/{n_seeds}")
    print(f"\n  output range when ONLY the previous-position input moves 0 -> 1 "
          f"(0 for the optimal policy): {sens.mean():.3f} +- {sens.std(ddof=1):.3f}")
    return out


# ======================================================================================
# Finding 10 -- the friction ordering, decomposed: closed forms on the mean, the tail apart
# ======================================================================================

def friction_mechanism(n_paths: int = 40_000, n_seeds: int = 3) -> dict:
    """Why does a +10% volatility error outrank a 5bp cost -- and is the ordering the metric's?

    Leland (1985) prices a one-way proportional cost c, paid at rebalances dt apart, as extra
    variance: sigma^2 (1 + Le), Le = c sqrt(8/pi) / (sigma sqrt(dt)). With mu = r = 0 and
    G = E[1/2 int Gamma S^2 sigma^2 dt] = sigma Vega / 2, the MEAN effects of the sweep
    (same contracts, same draws as ``misspecification_sweep``) have closed forms:

      vol error, sigma_r = 1.1 sigma_h:   V(sigma_h) - V(sigma_r)                    (exact)
      cost c:                            -(Le G + c S0 (Delta_0 + N(d1)))      (first order)

    The second cost term is opening the hedge at t0 and liquidating it at T, which
    dhbench/pnl.py charges (cost sum to n) and Leland's rebalancing count omits; with
    mu = 0, E[S_n delta_{n-1}] ~ S0 N(d1). Agreement on the mean checks simulator and
    accounting against closed forms, contract by contract.

    The paper's ordering is on CVaR-95, and the tail metric amplifies the two effects by
    different factors, so which friction dominates can depend on the risk measure as well
    as the cost level. Rebalancing-only Leland (ratio ((1.1)^2 - 1) / Le) tracks the CVaR
    ratios only because the two things it omits -- the opening and closing trades, and the
    tail amplification -- roughly offset. Agreement for the wrong reason, recorded as such.
    """
    from scipy.stats import norm

    from dhbench.pnl import transaction_costs

    sigma_h, s0, x = 0.2, 100.0, 0.10
    costs = (0.0005, 0.0025, 0.005)

    def leland(c, dt):
        return c * np.sqrt(8.0 / np.pi) / (sigma_h * np.sqrt(dt))

    rows = []
    for strike in (90.0, 100.0, 110.0):
        for maturity in (0.25, 1.0):
            for freq, per_year in (("weekly", 52), ("daily", 252)):
                n_steps = max(2, round(maturity * per_year))
                dt = maturity / n_steps
                premium = float(bs_call_price(s0, strike, maturity, RATE, sigma_h))
                d1 = (np.log(s0 / strike) + 0.5 * sigma_h**2 * maturity) / (sigma_h * np.sqrt(maturity))
                g = sigma_h * s0 * np.sqrt(maturity) * norm.pdf(d1) / 2.0

                acc: dict[str, list[float]] = {"vol_mean": [], "vol_cvar": [], "rebalancing_share": []}
                for c in costs:
                    acc[f"cost_mean_{c}"], acc[f"cost_cvar_{c}"] = [], []
                for seed in range(n_seeds):
                    def hedged(sigma_r):
                        spot = simulate_gbm(n_paths, n_steps, s0, RATE, sigma_r, maturity,
                                            make_generator(seed, "sweep"))
                        delta = delta_hedge_positions(spot.numpy(), strike, maturity, RATE, sigma_h)
                        return spot, tf.constant(delta, tf.float32)

                    def pnl(spot, delta, cost):
                        payoff = tf.maximum(spot[:, -1] - strike, 0.0)
                        return terminal_pnl(spot, delta, payoff, cost, premium).numpy()

                    spot, delta = hedged(sigma_h)
                    base = pnl(spot, delta, 0.0)
                    high = pnl(*hedged((1.0 + x) * sigma_h), 0.0)
                    acc["vol_mean"].append(high.mean() - base.mean())
                    acc["vol_cvar"].append(cvar(high) - cvar(base))
                    for c in costs:
                        charged = pnl(spot, delta, c)
                        acc[f"cost_mean_{c}"].append(charged.mean() - base.mean())
                        acc[f"cost_cvar_{c}"].append(cvar(charged) - cvar(base))
                    # total from the scorekeeper; opening + closing from their definition
                    total = transaction_costs(spot, delta, costs[0]).numpy().mean()
                    s, d = spot.numpy(), delta.numpy()
                    ends = costs[0] * np.mean(s[:, 0] * np.abs(d[:, 0]) + s[:, -1] * np.abs(d[:, -1]))
                    acc["rebalancing_share"].append(1.0 - ends / total)

                m = {k: float(np.mean(v)) for k, v in acc.items()}
                vol_exact = premium - float(bs_call_price(s0, strike, maturity, RATE, (1.0 + x) * sigma_h))
                row = {"strike": strike, "maturity": maturity, "rebalancing": freq, "n_steps": n_steps,
                       "vol_mean": m["vol_mean"], "vol_mean_exact": vol_exact,
                       "vol_cvar": m["vol_cvar"], "rebalancing_share_5bp": m["rebalancing_share"],
                       "amplification_vol": m["vol_cvar"] / m["vol_mean"],
                       "leland_rebalancing_only_ratio_5bp": ((1.0 + x) ** 2 - 1.0) / leland(costs[0], dt)}
                for c in costs:
                    pred = -(leland(c, dt) * g + c * s0 * 2.0 * norm.cdf(d1))
                    row[f"cost_{c}"] = {"mean": m[f"cost_mean_{c}"], "mean_predicted": pred,
                                        "cvar": m[f"cost_cvar_{c}"],
                                        "amplification": m[f"cost_cvar_{c}"] / m[f"cost_mean_{c}"],
                                        "ratio_mean": m["vol_mean"] / m[f"cost_mean_{c}"],
                                        "ratio_cvar": m["vol_cvar"] / m[f"cost_cvar_{c}"]}
                rows.append(row)

    def rel(a, b):
        return abs(a / b - 1.0)

    five = f"cost_{costs[0]}"
    summary = {
        "max_rel_error_vol_mean": max(rel(r["vol_mean"], r["vol_mean_exact"]) for r in rows),
        "max_rel_error_cost_mean": max(rel(r[f"cost_{c}"]["mean"], r[f"cost_{c}"]["mean_predicted"])
                                       for r in rows for c in costs),
        "amplification_vol": [min(r["amplification_vol"] for r in rows),
                              max(r["amplification_vol"] for r in rows)],
        "amplification_cost_5bp": [min(r[five]["amplification"] for r in rows),
                                   max(r[five]["amplification"] for r in rows)],
        "rebalancing_share_5bp": [min(r["rebalancing_share_5bp"] for r in rows),
                                  max(r["rebalancing_share_5bp"] for r in rows)],
        "leland_rebalancing_only_vs_cvar_ratio_5bp": [
            min(r[five]["ratio_cvar"] / r["leland_rebalancing_only_ratio_5bp"] - 1.0 for r in rows),
            max(r[five]["ratio_cvar"] / r["leland_rebalancing_only_ratio_5bp"] - 1.0 for r in rows)],
        "cost_dominates": {f"{c}": {"mean": sum(r[f"cost_{c}"]["ratio_mean"] < 1.0 for r in rows),
                                    "cvar": sum(r[f"cost_{c}"]["ratio_cvar"] < 1.0 for r in rows)}
                           for c in costs},
    }
    out = {"n_paths": n_paths, "n_seeds": n_seeds, "rows": rows, "summary": summary}

    print(f"\n  {len(rows)} contracts x {n_seeds} seeds, {n_paths:,} paths, sweep draws; mean vs CVaR-95\n")
    print(f"  {'K':>5}{'T':>6}{'rebal':>8} | {'vol mean':>9}{'exact':>8} | {'5bp mean':>9}{'pred':>8}"
          f"{'rebal%':>7} | {'amp vol':>8}{'amp 5bp':>8} | {'ratio mean':>10}{'CVaR':>6}{'Leland':>7}")
    for r in rows:
        c5 = r[five]
        print(f"  {r['strike']:>5.0f}{r['maturity']:>6.2f}{r['rebalancing']:>8} | "
              f"{r['vol_mean']:>+9.3f}{r['vol_mean_exact']:>+8.3f} | {c5['mean']:>+9.4f}{c5['mean_predicted']:>+8.4f}"
              f"{r['rebalancing_share_5bp']:>7.0%} | {r['amplification_vol']:>8.2f}{c5['amplification']:>8.2f} | "
              f"{c5['ratio_mean']:>10.2f}{c5['ratio_cvar']:>6.2f}{r['leland_rebalancing_only_ratio_5bp']:>7.2f}")
    q = summary
    print(f"\n  mean effects vs closed forms: vol within {q['max_rel_error_vol_mean']:.1%}, "
          f"cost within {q['max_rel_error_cost_mean']:.1%} (all contracts, 5/25/50bp)")
    print(f"  tail amplification (CVaR / mean): vol {q['amplification_vol'][0]:.2f}-{q['amplification_vol'][1]:.2f}x, "
          f"5bp cost {q['amplification_cost_5bp'][0]:.2f}-{q['amplification_cost_5bp'][1]:.2f}x")
    for c, v in q["cost_dominates"].items():
        print(f"  cost dominates at {float(c) * 1e4:.0f}bp: {v['mean']}/{len(rows)} on the mean, "
              f"{v['cvar']}/{len(rows)} on CVaR-95")
    return out


FINDINGS: dict[str, Callable[..., dict]] = {
    "baseline": baseline,
    "misspecification": misspecification,
    "precision": precision,
    "noise_floor": noise_floor,
    "noise_floor_long": noise_floor_long,
    "misspecification_sweep": misspecification_sweep,
    "seeding": seeding,
    "precision_by_n": precision_by_n,
    "convergence": convergence,
    "evaluation_design": evaluation_design,
    "friction_mechanism": friction_mechanism,
}


def run_finding(name: str) -> dict:
    print(f"\n{'=' * 78}\n  {name.upper()}\n{'=' * 78}")
    return FINDINGS[name]()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("finding", nargs="?", choices=sorted(FINDINGS))
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--json", metavar="PATH", help="write results as JSON")
    args = parser.parse_args()

    if not args.all and args.finding is None:
        parser.error("give a finding name or --all")

    names = sorted(FINDINGS) if args.all else [args.finding]
    results = {n: run_finding(n) for n in names}

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2, sort_keys=True)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
