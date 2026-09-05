# Paper structure

**Working title:** *Do Learned Hedging Policies Beat Classical Transaction-Cost Bands? A
Controlled Comparison*

**Decided 2026-08-25.** Supersedes the section plan in [PAPER.md](../PAPER.md), which
remains the roadmap document. This file governs the paper's shape; PAPER.md governs the
project's stages.

**Target:** arXiv preprint → ICAIF / NeurIPS FinML workshop, or thesis chapter.

---

## 1. The framing decision

Two shapes were considered.

| | A — benchmark-first | B — finding-first ✅ |
|:--|:--|:--|
| Lead claim | "we built a benchmark" | "published results are not comparable, and here is what changes when you fix that" |
| Benchmark's role | the contribution | the *instrument* that produces the contribution |
| Results | one section | three research questions, each with a stated answer |
| Failure mode | cited, not read | none obvious |
| Survives being scooped on tooling? | no | yes — the findings stand |

**B is adopted.** A benchmark is a hygiene contribution: valuable, and rarely read. The
same work, reorganised around three answerable questions, produces a findings paper in
which the benchmark is the apparatus. It also degrades gracefully — if another group
publishes a deep hedging benchmark first, §8 (regime fragility) remains a standalone paper,
which §7 of the roadmap already identifies as the fallback.

Two structural consequences follow.

- **Results split into three research questions (§7–9)**, each stated as a question with an
  answer, *including* "no measurable difference" answers. That is what distinguishes a
  findings paper from a tool paper.
- **Baselines (§4) precede the protocol (§5).** The reader must understand why the bar
  matters — and that most of the literature clears a lower one — before seeing the protocol
  constructed around it.

---

## 2. Section plan

| § | Title | Ready after | Status |
|:--|:--|:--|:--|
| 1 | Introduction | — | draftable now |
| 2 | Related work | — | draftable now |
| 3 | Problem setup and P&L accounting | Stage 0 | **written** |
| 4 | Baselines, and why the bar matters | Stage 0 | **written** |
| 5 | The benchmark protocol | Stage 4 | pending |
| 6 | Agents and re-implementation | Stage 5 | pending |
| 7 | **RQ1** — does deep hedging beat the classical band? | Stage 5 | pending |
| 8 | **RQ2** — what survives a regime shift? | Stage 5+ | pending |
| 9 | **RQ3** — does the ranking change with the risk measure? | Stage 5 | pending |
| 10 | What verification revealed | Stage 0 | **written** |
| 11 | Limitations | — | outlined |
| A–C | Appendices: full grid, configs, ladder | Stage 5 | pending |

### §1 Introduction

The comparability gap, quoted and cited from the 2023 *Mathematics* review. The
delta-hedging strawman: beating naive delta hedging under costs is close to trivial, since
any policy that trades less will do it, so the comparison carries almost no information
about learning.

**Contributions are stated as findings, not artefacts.** The benchmark is named as the
instrument. Three questions are posed and their answers previewed.

### §2 Related work

Buehler et al. (2019) as the formulation inherited wholesale. Whalley–Wilmott (1997) and
Zakamouline (2006) as the classical bar. The 2025–26 wave as the comparison set: adversarial
training, band-structured priors, ambiguity aversion, uncertainty quantification,
non-convexity, hedging valuation adjustment. *What Does Deep Hedging Actually Learn?* (2026)
as the direct antecedent for §8.

LOBFrame as the structural model: a framework paired with an honest evaluation showing that
strong in-sample performance need not imply economic value.

### §3 Problem setup and P&L accounting — **written**

Market, claim, and the shape contract implied by predictability. The P&L functional
**derived** from the self-financing condition rather than asserted. Terminal liquidation and
why the cost sum runs to `n`. Sign convention. The change of numéraire, and the two
independent facts that make it free. Risk measures and cash-invariance. Indifference pricing.

Includes §3.7, the four-part argument for why this cannot be posed as supervised learning —
which is also the justification for the rung-4 gate having diagnostic power.

### §4 Baselines, and why the bar matters — **written**

Black–Scholes delta as the zero-cost ground truth and a deliberately poor cost baseline.
Whalley–Wilmott with the cube-root scaling argument. Zakamouline.

**Carries the trade-to-centre measurement**, which is doing real argumentative work rather
than serving as an implementation note:

```
20 seeds, paths shared within each seed

comparison            mean       sd       t   consistent
edge   vs delta    +0.1343   0.0446   13.46      20/20
centre vs delta    +0.0354   0.0327    4.84      17/20
edge   vs centre   +0.0989   0.0405   10.93      20/20
```

A baseline implemented with the centre rule delivers 0.099 less CVaR improvement than a
correct one, consistently in every seed, and therefore sits much closer to the naive
strawman than to a real band. Any paper comparing against such a baseline overstates its
advantage. This is evidence that the field's baselines may be soft, and it belongs early.

Report the **difference in levels, not the ratio of improvements**: the ratio divides two
quantities of order the CVaR noise floor and is not estimable at this sample size (mean 24%,
sd 27%, range spanning zero). An earlier single-seed ratio was withdrawn — see
`experiments/findings.py`.

### §5 The benchmark protocol

Worlds, cost models, risk measures, rebalancing frequencies, claim set, metrics, controls,
and the tiered grid design (§4 below). Everything frozen and published as configuration
files. Resolution of the open decisions in §3 of this document must appear here explicitly.

### §6 Agents and re-implementation

Feedforward, recurrent (Carbonneau), band-structured (Arzel–Lehdili), robust (He et al.).
Equal parameter count and equal training budget stated and enforced.

**The two-stage validation protocol** (§3.1 below) is described here, because it is the
answer to the most serious methodological objection the paper faces.

### §7 RQ1 — Does deep hedging beat the classical band?

The headline grid. Effect sizes with dispersion across seeds, never point estimates.
Comparisons that do not survive seed variation are reported as **inconclusive**, not as wins.
Null results retained by advance commitment.

### §8 RQ2 — What survives a regime shift?

Train on one world, evaluate on another; degradation ratio for every method. Extends the
single-model finding of the 2026 fragility paper across the full method set, which is what
makes it a benchmark result rather than a replication.

**Includes the simulator → realised SPX evaluation.** Simulator-to-simulator fragility is a
weak claim, since both distributions were chosen by us. Simulator-to-reality is the version
that gets read, and reality lies in none of the simulator families.

### §9 RQ3 — Does the ranking change with the risk measure?

Re-rank all methods under entropic, CVaR₉₅ and mean–variance. **If the ranking changes, that
is itself the finding**: it means cross-paper comparison is meaningless even in principle,
because papers using different objectives cannot be placed on a common scale. A negative
result here is as publishable as a positive one.

### §10 What verification revealed — **written**

Short, and it earns the reader's trust in every other number. The correctness ladder. The
seeding defect (§5.4 of the technical report): unhashed replicate seeds biased a Monte Carlo
price by 9.3 standard errors and produced cross-seed error bars 2.5× too narrow. The three
specification errors caught during Stage 0, all of which biased results *toward* deep hedging.

This section supports the paper's thesis rather than being housekeeping: it is direct
evidence that plausible-looking numbers in this area can be wrong in ways no single run
reveals.

### §11 Limitations

Simulated markets and the conditionality that implies. Market incompleteness under Heston
and what stock-only hedging can and cannot address. The single borrow/lend rate assumed by
the numéraire. Compute budget and its effect on hyperparameter search. Seed variance
relative to effect sizes. Re-implementation risk, with the two-stage mitigation and its
residual.

---

## 3. Protocol decisions required before Stage 4

These emerged from a quant-perspective review on 2026-08-25. Each is unresolved, each
affects results, and each becomes far more expensive to change once the protocol is frozen
or results exist. **Resolve and record all seven in §5 before Stage 4 closes.**

### 3.1 🔴 Re-implementation validation is currently impossible as specified

The stated mitigation — *"validate each re-implementation against its source paper's
headline number"* — cannot be executed. That number exists in the source paper's protocol:
its simulator, cost rate, risk measure and rebalancing frequency. This benchmark replaces all
of them. There is nothing to validate against.

The consequence is the standard objection that kills benchmark papers. When a re-implemented
method underperforms, a reviewer cannot distinguish *"the method is worse under a fair
protocol"* from *"this implementation is worse than the original."*

**Resolution: two-stage validation.**

1. Reproduce the method's published headline number **in its own published setting**. Record
   the match, or the failure to match, in an appendix.
2. Only then port the method into the benchmark protocol.

Any method that fails stage 1 enters the grid labelled **unvalidated**, and its results are
reported with that label attached rather than silently pooled.

**Cost:** roughly doubles implementation work per method. Must be budgeted at Stage 5
planning, not discovered inside it. It is also a reason to prefer *fewer, properly validated*
methods over broad coverage.

### 3.2 🔴 No Whalley–Wilmott band exists for a CVaR objective

Whalley–Wilmott is derived under exponential utility. The grid has three risk measures:

```
entropic         -> WW band exists; the comparison is well posed
CVaR-95          -> no corresponding classical band
mean-variance    -> no corresponding classical band
```

For two of three columns, *"does deep hedging beat the classical bar"* has no well-posed
answer. The failure mode is silent: the comparison quietly reverts to delta hedging, which is
the strawman this project exists to avoid, or to a band tuned to a different objective, which
violates the like-for-like requirement the code already warns about.

**Options.**

- **(a)** Restrict the RQ1 classical comparison to the entropic column, and state that
  plainly. Cheapest, honest, and narrows the headline claim.
- **(b)** Use Zakamouline as the general-purpose bar where it admits the relevant
  calibration.
- **(c)** Derive or adopt a CVaR-appropriate band from the literature. Most work, strongest
  result.

**Recommendation: (a) for the first version, with (b) explored if time permits.** A narrow
well-posed claim beats a broad ill-posed one. RQ3 is unaffected — it compares methods against
*each other* within each risk measure, which remains well posed throughout.

### 3.3 🔴 Heston is an incomplete market; stock-only hedging confounds the result

Volatility risk cannot be hedged with the underlying alone. Buehler et al. hedge with options
as well as stock, precisely for this reason.

If Heston cells permit stock-only hedging, residual risk is dominated by unhedgeable vega and
every method's performance converges toward the irreducible variance. Measured differences
between methods shrink toward noise **for reasons unrelated to the methods**, and the paper
would report a null result caused by its own design.

**Options.**

- **(a)** Stock-only, with the incompleteness stated prominently in §11 and the Heston cells
  interpreted as *"how well does each method handle risk it cannot remove"* rather than as a
  hedging-quality comparison.
- **(b)** Add a second hedging instrument (a liquid option). Correct, and it changes
  `pnl.py`: `delta` gains an instrument axis, `(n_paths, n_steps)` becomes
  `(n_paths, n_steps, n_instruments)`.

**Recommendation: (a) for the first version, but the `pnl.py` signature should be designed
now so that (b) is a later addition rather than a rewrite** of the single-source-of-truth
module. See §3.8.

### 3.4 🟠 The cost grid is not calibrated to the instrument

50 bp proportional cost is roughly a single stock. Nobody delta-hedges an SPX option in cash
equities; it is hedged in ES futures at round-trip costs closer to **0.5–1 bp**.

This matters because the cost level determines whether the headline question is interesting
at all. The measured band advantage was `CVaR +0.134` at 50 bp. At 1 bp it will be a small
fraction of that, and plausibly inside seed noise — in which case RQ1's answer under
realistic costs is *"neither method matters much"*, which is a legitimate finding but a very
different paper.

**Resolution:** calibrate the cost axis from realised SPX bid-ask spreads via WRDS, and span
the range from futures-like (≈1 bp) to single-stock-like (≈50 bp) rather than picking round
numbers. **This promotes WRDS cost calibration from optional to required at Stage 4.**

### 3.5 🟠 Rebalancing frequency must be an axis, not a fixed constant

`n_steps = 30` versus `250` changes hedging error by roughly a factor of three and changes
the cost/risk trade-off qualitatively. It is currently fixed in one configuration.

Leaving it fixed makes every result conditional on an unexamined choice — precisely the
criticism this paper levels at others. **Add rebalancing frequency to the protocol as a
first-class axis**, at minimum {monthly, weekly, daily} equivalents.

### 3.6 🟠 The claim set is the easiest possible case

A single at-the-money European call on one underlying is where delta hedging performs best
and where a learned policy has least room to add value. The natural objection is that the
benchmark tests the case that does not need the method.

**Recommendation: add one path-dependent claim — a down-and-out barrier call.** Its delta is
discontinuous at the barrier, which is exactly where band-structured and recurrent policies
should separate from feedforward ones, and where the classical bands have no clean analogue.
Cost is one payoff function plus a pricing reference (closed form exists for the
Black–Scholes case). A portfolio with offsetting gammas is a stronger test still, and a
reasonable second addition if time permits.

### 3.7 🟠 Real-data evaluation should be required, not optional

Currently deferred to "Stage 5+", which in practice means it may not happen. It is the answer
to the paper's fundamental limitation, WRDS access is confirmed, and simulator-to-reality is
the version of §8 that gets read.

**Promote it to a required section of §8.** Use SPX (European, cash-settled — *not*
single-name equity options, which are American and would change the problem). Note the
implied code change: SPX pays dividends, so `bs_delta` and `bs_gamma` need a continuous
dividend yield `q` before any real-data section, changing `d1` and multiplying delta by
`exp(-q*tau)`.

### 3.8 🟡 Carried over: the cost-model interface

`transaction_costs` hardcodes proportional costs and cannot express
`costs/fixed.py`'s fixed-plus-proportional model. That is the **non-convex** cell of the
grid — the one that tests arXiv:2510.01874's claim directly, and the most publishable single
cell in the design.

The signature of the single-source-of-truth module must change **before** the protocol is
frozen and results exist. Combine this change with the instrument-axis design in §3.3 so
`pnl.py` is touched once rather than twice.

---

## 4. Grid design and compute budget

The naive full factorial is infeasible:

```
4 worlds x 4 cost levels x 3 risk measures x 5 agents x 5 seeds  ~=  1,200 runs
```

That is not achievable on free Colab, and discovering it during Stage 5 would be
disqualifying. **Design a tiered grid at Stage 4:**

- **Core (full factorial).** GBM and Heston × two cost levels × entropic and CVaR × all
  agents × 5 seeds. This carries RQ1 and RQ3.
- **Periphery (one factor at a time).** Vary jumps, regime-switching, mean–variance,
  additional cost levels and rebalancing frequencies against a fixed core configuration.
- **Fragility (§8).** All train/test world pairs at a single fixed cost and risk measure.

Every omitted cell is **logged as omitted** with its reason. Silent truncation reads as full
coverage when it is not, which is itself a form of the reporting failure this paper
criticises.

**State the compute budget in §11.** A stated constraint is a limitation; an unstated one is
a flaw.

---

## 5. What is already written

`paper/00-draft.md` covers §1–4 and §10 in publishable prose, with an explicit status
section recording that no empirical results exist. `paper/technical-report.pdf` (22pp)
documents Stage 0 in full and plans Stages 1–6.

**Reusable immediately:** the problem setup and accounting derivation, the baselines section
including the trade-to-centre measurement, the verification section including the seeding
defect, and the limitations outline.

**Needs writing after results exist:** §5–9 and the appendices.

---

## 6. Standing commitments

- **Null results are reported.** Cells where deep hedging ties or loses to the classical band
  appear in the tables. The incentive to drop unflattering cells is the specific failure this
  paper criticises.
- **Dispersion, not point estimates.** Comparisons that do not survive seed variation are
  reported as inconclusive.
- **Omissions are logged.** Any bounded coverage — top-N, sampling, no-retry — is stated.
- **Every number traces to a config plus a seed.** Nothing in the paper comes from a
  notebook.
