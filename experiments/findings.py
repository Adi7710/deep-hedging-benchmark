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

from dhbench.baselines.bs_delta import bs_call_price, bs_delta, delta_hedge_positions
from dhbench.baselines.whalley_wilmott import band_hedge_positions, whalley_wilmott_band
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
    """Mean of the worst ``1 - alpha`` fraction. Higher is better, so the left tail."""
    return float(pnl[pnl <= np.quantile(pnl, 1.0 - alpha)].mean())


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
    """The common band-hedging error: on exit, trade back to delta_BS not to the edge.

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
# Finding 1 -- the baseline that most of the literature gets wrong
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
    """How precise is CVaR-95, and what effect size can 5 seeds resolve?

    Supports §F of docs/06. A benchmark that cannot state its own minimum detectable
    effect cannot distinguish a null result from an underpowered one.
    """
    cost = 0.005
    pairs = [
        (
            cvar(_run(SIGMA, SIGMA, cost, "delta", s, n_paths)),
            cvar(_run(SIGMA, SIGMA, cost, "band", s, n_paths)),
        )
        for s in range(n_seeds)
    ]
    a = np.array([p[0] for p in pairs])
    b = np.array([p[1] for p in pairs])
    d = b - a

    unpaired = float(np.sqrt(a.var(ddof=1) + b.var(ddof=1)))
    paired = float(d.std(ddof=1))
    mde5 = 2.776 * paired / np.sqrt(5)  # t_{0.975, 4}

    out = {
        "sd_cvar_across_seeds": float(a.std(ddof=1)),
        "unpaired_sd_of_difference": unpaired,
        "paired_sd_of_difference": paired,
        "variance_reduction_from_common_paths": unpaired / paired,
        "effect": float(d.mean()),
        "mde_at_5_seeds": float(mde5),
        "detectable": bool(abs(d.mean()) > mde5),
    }
    print(f"\n  SD of CVaR-95 across {n_seeds} seeds        : {out['sd_cvar_across_seeds']:.4f}")
    print(f"  unpaired SD of the difference       : {unpaired:.4f}")
    print(f"  paired SD (common random numbers)   : {paired:.4f}")
    print(f"  variance reduction from pairing     : {unpaired / paired:.1f}x"
          f"   (NOT an order of magnitude)")
    print(f"\n  measured band-vs-delta effect       : {out['effect']:+.4f}")
    print(f"  minimum detectable effect, 5 seeds  : {mde5:.4f}")
    print(f"  detectable at 5 seeds               : {out['detectable']}")
    return out


FINDINGS: dict[str, Callable[..., dict]] = {
    "baseline": baseline,
    "misspecification": misspecification,
    "precision": precision,
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
