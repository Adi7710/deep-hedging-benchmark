# 06 — Master implementation plan

**Written 2026-09-05**, after a quantitative reassessment that changed the framing. This is
the top-level planning document: why the paper exists, what it claims, what data it uses,
how it will be built, and — stated honestly — what it is and is not worth.

Supersedes nothing; it sits above `paper/STRUCTURE.md` (paper shape),
`docs/03-benchmark-protocol.md` (frozen axes) and `docs/05-stage-1-2-plan.md` (near-term
build). Where this document and those disagree, this one is newer.

---

# Part A — Why this paper exists

## A.1 The reframing, and the measurement that forced it

The original question was:

> *Do learned hedging policies beat classical transaction-cost bands?*

On 2026-09-05 the relative size of the effects involved was measured for the first time,
on an ATM one-year call, 50 rebalances, 40,000 paths:

```
effect on CVaR-95

  transaction cost   5bp   (index options hedged in futures -- realistic)   -0.217
  transaction cost  50bp   (single stock)                                   -2.257
  transaction cost 500bp   (implausible)                                   -24.570

  realised vol 22% when hedged at 20%                                       -1.386
  realised vol 25% when hedged at 20%                                       -3.635
  realised vol 30% when hedged at 20%                                       -7.700
```

**At realistic hedging costs, a two-point volatility error costs 6.4x what transaction
costs do.** The benchmark as originally scoped made transaction cost the central axis while
holding volatility known and correct — that is, it made the second-order friction central
and treated the first-order one as solved.

This is not a small correction. It is the difference between a paper about a rounding error
and a paper about the thing that actually drives hedging P&L.

**Adopted question:**

> Under **model misspecification** and **market frictions**, do learned hedging policies
> beat classical rules — and **which of the two failure modes dominates**?

## A.2 Why this question, and why it is answerable

Three claims underpin it, in order of how contestable they are.

**Claim 1 — published deep hedging results are not comparable.** Uncontested; a 2023 review
in *Mathematics* states it directly. Each paper fixes its own simulator, cost model, risk
measure, rebalancing frequency and metric.

**Claim 2 — many are measured against a baseline that is too weak.** Beating naive delta
hedging under costs is close to trivial: it rebalances every step regardless of trade size,
so any policy that trades less wins. The informative comparison is against a cost-aware
band. We have *measured* how much this matters, across 20 seeds with paths shared within
each seed:

```
comparison            mean       sd       t   consistent
edge   vs delta    +0.1343   0.0446   13.46      20/20
centre vs delta    +0.0354   0.0327    4.84      17/20
edge   vs centre   +0.0989   0.0405   10.93      20/20
```

A Whalley–Wilmott implementation that rebalances to the centre of the band rather than its
nearest edge — a common error — delivers **0.099 less CVaR improvement**, consistently in
every seed. Its baseline sits much closer to the naive strawman than to a correct band.

> **Withdrawn.** An earlier draft quoted "the centre rule captures 7% of the available
> improvement" from a single seed. That ratio divides two differences each of order the
> CVaR noise floor (0.048); across 20 seeds it has mean 24%, standard deviation 27%, and a
> range spanning zero. It is not estimable at this sample size and must not be quoted. The
> paired difference in levels above is the defensible statement. This is exactly the error
> the paper criticises — a point estimate from one seed, dispersion unchecked — and it was
> caught only by moving the measurement into `experiments/findings.py`.

**Claim 3 — the field is optimising the wrong friction.** New, and the measurement above is
the evidence. If it survives scrutiny it reorders what the subfield should work on, and it
is the most valuable thing this project can say.

All three are answerable with simulation alone, which is what makes the project tractable.

## A.3 What this paper does *not* claim

Stated up front because over-claiming is the failure mode being criticised.

- It does not claim deep hedging is good or bad. It measures, under one protocol.
- It does not claim results transfer to a trading desk. Every world is simulated, save one
  bounded real-data section.
- It does not claim to have implemented published methods as well as their authors. The
  two-stage validation protocol (§E.5) exists precisely because that cannot be assumed.
- It does not claim novelty in method. The contribution is measurement.

---

# Part B — Value assessment

Asked directly: **does this create value for the quant world?** Segmented honestly, because
the answer differs sharply by audience.

## B.1 To the deep hedging research literature — **high, and specific**

| Contribution | Strength of evidence | Would it change behaviour? |
|:--|:--|:--|
| Published baselines may be materially too weak | **measured**, 20 seeds: the centre rule delivers 0.099 less CVaR improvement than the edge rule, t = 10.9, consistent 20/20 | Yes — a referee can now ask "which band rule did you implement?" |
| Misspecification dominates frictions at realistic costs | **measured**: 6.4x at 5bp | Yes, if it holds — it reorders the subfield's priorities |
| Cross-seed error bars can be 2.5x too narrow | **measured**: 9.3 SE bias, dispersion 0.34 of analytic | Yes, and it is cheap to fix |
| CVaR-95 is ~5x noisier than the mean at equal N | **measured**: SE 0.038 vs 0.0071 at N=20k | Yes — it changes required sample sizes |
| An open, verified reference implementation | correctness ladder, 85 tests | Moderate — reduces duplicated effort |

That is a real contribution set. Note that **four of five are measurements, not assertions**,
and three were discovered rather than planned.

## B.2 To practitioner quants — **low, with one exception**

Honest assessment: **no trading desk will change its hedging because of this paper.**

- Deep hedging is not in production at most institutions.
- Every result is conditional on a simulator, and a desk's model risk is precisely that the
  world is not in the simulator's class.
- The claim set — one ATM European call on one underlying — is the easiest case, not a book.

**The exception**: the misspecification-versus-friction ordering (§A.1) is a statement about
hedging economics, not about neural networks, and it holds regardless of whether anyone uses
deep hedging. A practitioner would find that ordering unsurprising but would find it useful
to see it *quantified* against the frictions literature, which routinely treats costs as the
central problem.

## B.3 To ML methodology beyond finance — **narrow but transferable**

The seeding defect is not specific to hedging. Any study drawing replicate seeds as
consecutive small integers from `tf.random.Generator.from_seed` inherits it, and the symptom
— error bars too narrow — is invisible in any single run and undetectable by a same-seed
reproducibility check. That is a short, citable methodological note in its own right, and
arguably worth publishing separately.

## B.4 To the author — **high, and worth stating plainly**

The artefact that signals capability is not the paper; it is the reasoning trace. Concretely,
what this project has demonstrated so far:

- Finding a real defect nobody was looking for (seeding), and quantifying it rather than
  patching it.
- Measuring rather than asserting — the trade-to-centre cost, the misspecification ordering,
  the CVaR precision, the minimum detectable effect.
- Correcting one's own claims when measurement contradicts them (see §F.3 on common random
  numbers).
- Designing verification so that failures localise — the correctness ladder, the free
  analytic test on the rollout.
- Refusing constructions that would make a test vacuous — the sigmoid output, the residual
  parameterisation, model-derived state features.

These are the behaviours a quantitative research group hires for, and they are visible in the
commit history whether or not the paper lands well.

## B.5 The honest summary

> The paper's contribution to the quant *world* is **modest and specific**: it is a
> measurement paper that corrects three things the deep hedging literature gets wrong, one of
> which — the misspecification ordering — may reorder the subfield's priorities. Its
> contribution to *trading practice* is near zero, and claiming otherwise would be
> dishonest. Its contribution as evidence of research capability is substantial.

Those are three different things and should not be conflated in the abstract.

---

# Part C — Data

## C.1 Primary: simulated

There is no dataset to acquire, and this is methodologically correct rather than a
compromise. Ground truth is known under GBM at zero cost, which gives a hard correctness
gate no real-data project can have; and fixing the data-generating process is the entire
point of a controlled comparison.

| World | Module | Purpose | Status |
|:--|:--|:--|:--|
| GBM | `worlds/gbm.py` | ground truth, rung 4 | **done** |
| Heston | `worlds/heston.py` | stochastic vol, incomplete market | Stage 3 |
| Regime-switching | `worlds/regime_switching.py` | the fragility test's target | Stage 3 |
| Jump-diffusion (Merton) | `worlds/jumps.py` | unhedgeable jump risk | Stage 3 |

All driven by explicit `tf.random.Generator` instances seeded through
`dhbench.seeding.make_generator`, never global state.

## C.2 Secondary: WRDS / OptionMetrics (IvyDB) — access confirmed

**Use SPX index options.** European and cash-settled, matching the fixed-maturity and
cash-settlement conventions already built into `pnl.py`. US single-name equity options are
**American**; early exercise makes the payoff date a stopping time and would silently change
the problem being solved.

Three uses, now reordered by the reassessment:

| Priority | Use | Why it moved |
|:--|:--|:--|
| **1 (Stage 4, required)** | Calibrate the **cost axis** from realised SPX bid-ask | The measurements show results depend critically on cost level. 50bp is a single stock; index hedging is nearer 1–5bp, where the friction effect is an order of magnitude smaller. Picking round numbers would make the headline conditional on an arbitrary choice |
| **2 (Stage 4, required)** | Calibrate the **misspecification axis**: realised-vs-implied vol spread from history | Newly first-class. The realistic range of `sigma_realised − sigma_implied` is an empirical question, and the whole reframing rests on it |
| 3 (Stage 5+) | Hedge realised SPX paths with simulator-trained policies | The strongest form of the fragility result: reality is in none of the simulator families |

**Constraints.** OptionMetrics is not redistributable: no raw data in the repository, and
the committed retrieval script plus its date is the reproducibility artefact. A code change
is implied — SPX pays dividends, so `bs_delta` and `bs_gamma` need a continuous yield `q`
before any real-data section.

## C.3 Deliberately not used

Deribit, OptionsDX, CBOE samples, yfinance option chains. Each was considered; none adds
anything OptionMetrics does not, and each adds a data-cleaning burden. Recorded so the choice
is visible.

---

# Part D — Protocol axes

Updated. **The change from previous versions is that misspecification is promoted to a
first-class axis and the cost axis is recalibrated downward.**

| Axis | Levels | Note |
|:--|:--|:--|
| World | GBM, Heston, regime-switching, jumps | |
| **Misspecification** | `sigma_hedge / sigma_realised` in {0.8, 0.9, 1.0, 1.1, 1.25} | **new, and the dominant effect** |
| Cost | 0, 1bp, 5bp, 25bp, 50bp | recalibrated: 1–5bp is index-realistic, 50bp single-stock |
| Cost model | proportional; proportional + fixed; **proportional + quadratic** | quadratic is a crude impact proxy; linear cost understates exactly the fast trades that matter |
| Risk measure | entropic (2 levels), CVaR-95, mean-variance | |
| Rebalancing | monthly, weekly, daily equivalents | promoted from a fixed constant |
| Claim | ATM call; **down-and-out barrier** | barrier has discontinuous delta — where methods should separate |
| Agent | feedforward, recurrent, band, robust, + baselines | |

The full factorial is not runnable (§F.4). The tiered design is in `paper/STRUCTURE.md` §4.

---

# Part E — Implementation sequence

## E.1 Done

```
worlds/gbm.py            exact solution, rung 1 green
pnl.py                   rung 2 green 13/13, numeraire-invariant, graph-safe
baselines/bs_delta.py    reference
baselines/whalley_wilmott.py   rung-5 baseline half green
seeding.py               audited; dispersion enforced by test
objectives/*             entropic, CVaR (RU, w trainable), mean-variance
agents/feedforward.py    construction + forward pass
```

85 tests passing, 7 skipped, 0 failing.

## E.2 Immediate — the reframing, before more machinery

| # | Task | Cost | Rationale |
|:--|:--|:--|:--|
| 1 | Misspecification axis in configs + evaluation | ~1 session | `delta_hedge_positions` already takes a separate hedging sigma; this is mostly plumbing and config |
| 2 | Power analysis: MDE per metric per seed count | ~1 session | Decides which comparisons are askable at all before any are run |
| 3 | Cost-model interface in `pnl.py` | ~1 session | Blocks the quadratic/fixed cells; must change before Stage 4 freezes, and should be done once alongside a possible instrument axis |

## E.3 Stage 1–2 — to rung 4

Steps 4–7 of `docs/05-stage-1-2-plan.md`, unchanged: the rollout (with its free analytic
test), the training loop, the seed noise floor, then rung 4 on a constructed moneyness grid.

## E.4 Stage 3 — richer worlds and agents

Heston with full truncation, validated against characteristic-function prices with a
Feller-violating parameter set (rung 3). Recurrent agent. Rung 5's learned-band half.

## E.5 Stage 5 — the grid, with two-stage validation

**Re-implementation validation cannot be done as originally specified.** A published headline
number lives in that paper's protocol, which this benchmark deliberately replaces, so there
is nothing to validate against. The resolution:

```
stage 1   reproduce the method's published number IN ITS OWN SETTING
          record the match, or the failure, in an appendix
stage 2   only then port it into this protocol
```

Any method failing stage 1 enters the grid labelled **unvalidated**, with results reported
under that label rather than silently pooled. This roughly doubles per-method work and is the
strongest argument for **fewer methods, properly validated** over broad coverage.

---

# Part F — Statistical design

The part most benchmark papers omit, and the part a quantitative reviewer will look at first.

## F.1 Metric precision — measured

```
N          SE(mean)    SE(CVaR-95)    ratio
5,000        0.0078        0.0515      6.6x
20,000       0.0071        0.0382      5.3x
100,000      0.0024        0.0128      5.2x
```

**CVaR-95 is roughly five times noisier than the mean at equal path count**, because a tail
statistic uses only the tail: 20,000 paths yields a CVaR estimated from 1,000. The headline
metric is the least precise one, and sample sizes must be set by *it*, not by the mean.

## F.2 Minimum detectable effect — measured

```
paired sd of the CVaR-95 difference across seeds : 0.0480
MDE at 5 seeds (t_0.975,4)                       : 0.0597
measured band-vs-delta effect                    : 0.1308   detectable, ~2.2x MDE
```

**Consequence for the grid.** Any learned-versus-band difference smaller than the
band-versus-delta difference will be marginal or undetectable at five seeds. Either the seed
count rises for the headline cells, or comparisons below the MDE are reported as
**inconclusive** — which is a legitimate and publishable answer, and one the field rarely
gives.

## F.3 Correction to a previous claim

`docs/05-stage-1-2-plan.md` §3.2 asserted that common random numbers give "free variance
reduction" on every A-versus-B comparison. **Measured, it is 1.4x on the mean and
essentially nothing on the headline metric.**

```
metric      unpaired sd   paired sd   reduction     (n = 20,000, 12 seeds)
mean P&L        0.01158     0.00816       1.4x
CVaR-95         0.05693     0.04800       1.2x
std P&L         0.01347     0.01243       1.1x

CVaR-95         0.06380     0.06670       0.96x     (n = 10,000, 8 seeds)
```

The two strategies produce genuinely different P&L distributions — the band trades far less
— so path-level P&Ls are not tightly correlated, and for a tail statistic the paths
populating each strategy's tail **are not even the same paths**. The CVaR-95 reduction
measures between 1.0x and 1.2x depending on sample size and can fall marginally *below* 1,
meaning pairing occasionally makes the tail comparison slightly noisier.

For the headline metric the honest statement is that common random numbers buy essentially
nothing. They remain worth using — they cost nothing, and they do help the mean — but they
must not be counted on in the power budget. `tests/test_findings.py` pins the measured range
so the claim cannot silently drift back.

## F.4 Sample-size and compute budget

Path count per evaluation is set by F.1; seed count per cell by F.2. The naive factorial is
roughly 1,200 runs and is not feasible on free Colab. Tiered design per
`paper/STRUCTURE.md` §4, with **every omitted cell logged as omitted with its reason** —
silent truncation reads as full coverage.

If cost forces a cut, **narrow the method set rather than the seed count.** Seeds are what
make the error bars honest.

---

# Part G — Risks and kill criteria

| Risk | Early signal | Response |
|:--|:--|:--|
| Rung 4 fails | §4.3 checklist of `docs/05` | Fixed diagnostic order; then reproduce Buehler's own setup and bisect |
| Effects below the MDE | power analysis (E.2 task 2) | Report inconclusive; narrow to paired within-method fragility, which is far less noise-limited |
| Misspecification result does not replicate | wider parameter sweep | It is currently a single measurement on one claim; **treat as provisional until swept** |
| Re-implementation cannot be validated | stage-1 reproduction fails | Label unvalidated; do not pool silently |
| Compute infeasible | per-run timing at step 5 | Narrow methods, not seeds |
| Scooped on the benchmark framing | literature search before Stages 4 and 6 | The misspecification-ordering result stands alone |

**Kill criteria** are in `docs/05-stage-1-2-plan.md` §7 and stand unchanged. Each has a
defensible smaller paper attached, which is the point of writing them down before they are
needed.

---

# Part H — Immediate actions

```
1. add the misspecification axis                      REFRAMES THE PAPER, ~free
2. power analysis: MDE per metric per seed count      decides what is askable
3. correct the CRN claim in docs/05                   honesty maintenance
4. cost-model interface in pnl.py                     unblocks 2 grid cells
5. resume step 4: the rollout, with its analytic test as planned
```

Items 1–3 come before more machinery, because they determine what the machinery is for.

**The single most important open empirical question** is whether the misspecification
ordering in §A.1 survives a proper sweep across moneyness, maturity, rebalancing frequency
and cost level. It is currently one measurement on one contract. If it holds, it is the
paper's headline. If it does not, the original framing was right and this document should be
revised back.
