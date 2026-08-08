# Paper plan

**Working title:** *A Reproducible Benchmark for Deep Hedging: Do Learned Policies Beat
Classical Transaction-Cost Bands?*

**Target:** arXiv preprint → workshop paper (ICAIF, NeurIPS FinML workshops) or thesis chapter.

**Status:** outline only. Sections fill in as stages complete.

---

## The claim

Not "we invented a better hedging algorithm." The claim is narrower and more defensible:

> Published deep hedging results are not comparable, because each paper fixes a different
> simulator, cost model, risk measure, and metric. Under a single protocol, we measure
> which methods actually win, by how much, and where they break.

This is a benchmark-and-honest-evaluation paper, in the spirit of what LOBFrame did for
limit order book forecasting. Contribution is rigour and reproducibility, not new
mathematics — which is exactly what makes it achievable alongside learning the tooling.

---

## Section plan

| § | Section | Content | Ready after |
|:--|:--|:--|:--|
| 1 | Introduction | the comparability gap, quoted and cited | now |
| 2 | Related work | Buehler 2019 → the 2025–26 wave | now |
| 3 | Problem setup | market, claim, P&L, convex risk measures, indifference price | Stage 0 |
| 4 | Benchmark protocol | worlds, costs, risk measures, metrics, controls | Stage 4 |
| 5 | Baselines | BS delta, Whalley–Wilmott, Zakamouline | Stage 0 |
| 6 | Agents | feedforward, recurrent, band, robust | Stages 2–3, 5 |
| 7 | Results | the full grid, **including null results** | Stage 5 |
| 8 | Regime fragility | train Heston → test regime-switching | Stage 5 |
| 9 | Limitations | simulated-only, compute budget, seed variance | — |

---

## §3 — Problem setup

Discrete times $t_0 < t_1 < \dots < t_T$, tradeable asset $S_t$, contingent claim $Z$
(European call payoff $(S_T - K)^+$ for the base case).

The agent chooses a hedge position $\delta_t = \pi_\theta(I_t)$ where $I_t$ is the
information available at $t$ — at minimum $(t, S_t, \delta_{t-1})$, optionally more.

Terminal P&L:

$$
\mathrm{PL}_T = -Z + \sum_{t=0}^{T-1} \delta_t (S_{t+1} - S_t) - \sum_{t=0}^{T-1} c_t(\delta_t, \delta_{t-1}, S_t) + p_0
$$

Training minimises a convex risk measure $\rho$:

$$
\min_\theta \ \rho\big(\mathrm{PL}_T(\theta)\big)
$$

and the indifference price is $p_0 = \rho(\mathrm{PL} \text{ with claim}) - \rho(\mathrm{PL}\text{ without})$.

**The key structural point** — and the reason this is not ordinary supervised learning —
is that there are no labels. The loss is a functional of the whole simulated path, and
gradients flow back through every timestep of the hedging decision.

**Sanity result to state and prove empirically:** with zero costs, GBM dynamics, and a
quadratic or entropic objective, the optimiser's solution is the Black–Scholes delta. §5
of the paper shows the overlay.

---

## §4 — Benchmark protocol

The contribution. Everything below is frozen and published as config files.

**Worlds** — GBM · Heston · regime-switching · jump-diffusion. Fixed parameter sets, fixed
seeds.

**Cost models** — zero · proportional at three levels · proportional + fixed.

**Risk measures** — entropic at two risk-aversion levels · CVaR₉₅ · mean-variance.

**Metrics**

| Metric | Why |
|:--|:--|
| CVaR₉₅ of terminal P&L | the tail is what hedging is for |
| P&L standard deviation | comparability with older literature |
| Turnover | is the advantage just trading more? |
| Total cost paid | separates gross skill from net |
| Indifference price vs. BS price | economic interpretation |
| **Degradation ratio** under train/test mismatch | §8, the regime fragility result |

**Controls the literature usually omits.** Equal parameter count. Equal training budget.
Multiple seeds with error bars. A stated compute budget. Most cross-paper comparisons
control for none of these, which is a large part of why the numbers don't line up.

---

## §7 — Results, and the null-result commitment

The grid is worlds × costs × risk measures × agents. Report all of it, including where
deep hedging **ties or loses** to Whalley–Wilmott.

This is stated as a commitment up front because the incentive to quietly drop unflattering
cells is exactly the failure mode the paper is criticising. Null results are the most
valuable and least published part of this work.

---

## §8 — Regime fragility

Train on one world, evaluate on another. The 2026 paper *What Does Deep Hedging Actually
Learn?* reports that learned policies "fail when market conditions shift beyond training
distribution." This section quantifies that across the full method set rather than one
model, which is what makes it a benchmark result rather than a replication.

If the benchmark angle gets scooped, **this section is the fallback standalone paper.**

---

## §9 — Limitations, stated honestly

- **Simulated markets only** (until the Stage 5+ real-data section). Any conclusion is
  conditional on the simulator being a reasonable proxy.
- **Compute budget** caps the hyperparameter search; a better-tuned agent may exist.
- **Seed variance** in deep hedging is large. Error bars, not point estimates.
- **Re-implementation risk** — reproduced variants may underperform their published
  versions through implementation error rather than method weakness. Mitigation: verify
  each against its paper's headline number before entering it in the grid.

---

## Literature-tracking discipline

Four directly relevant papers appeared in the last twelve months. Re-run the search
**before Stage 4 and again before Stage 6**. If someone publishes this benchmark first,
narrow to §8 as a focused paper. Log each search in
[docs/04-progress-log.md](docs/04-progress-log.md).
