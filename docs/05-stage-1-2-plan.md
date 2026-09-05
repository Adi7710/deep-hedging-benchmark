# 05 — Stage 1–2 execution plan

**Written 2026-08-29.** Covers the path from the current state to rung 4, the gate that
licenses every downstream number. Complements two existing documents: `paper/STRUCTURE.md`
governs the paper's shape and lists seven protocol decisions due before Stage 4;
`docs/03-benchmark-protocol.md` governs the frozen axes. This file governs *how the first
learned result is produced and what would make it trustworthy*.

---

## 1. Position

| Layer | State |
|:--|:--|
| Simulator (GBM) | verified, rung 1 green |
| P&L accounting | verified, rung 2 green, 13/13 |
| Black–Scholes reference | complete |
| Whalley–Wilmott baseline | complete, baseline half of rung 5 green |
| Seed derivation | audited and fixed; dispersion enforced by test |
| Risk measures | complete, 22 tests |
| **Everything learned** | **not started** |

71 tests passing. Nothing that follows can be blamed on the layers beneath it, which is the
entire point of having spent Stage 0 the way we did.

---

## 2. Design decisions to settle before writing code

These are modelling choices, not implementation details. Each affects what the results
*mean*, each is expensive to revisit once runs exist, and each will be asked about.

### 2.1 The information set — the leakage tension

The state fed to the policy must be **rich enough that the optimal policy is representable**
and **poor enough that it does not contain the answer**. Those pull against each other, and
the boundary needs a stated rule rather than case-by-case judgement.

**Adopted rule.** The information set may contain market observables and the agent's own
position. It may **not** contain any quantity derived from the pricing model the agent is
being tested against.

Applying it:

| Feature | Admitted? | Reasoning |
|:--|:--|:--|
| time to maturity `tau` | yes | a calendar fact |
| moneyness `S/K` | yes | observable price, scaled |
| `delta_prev` | **required** | without it a band is not representable (§2.3 of the technical report) |
| realised volatility over a trailing window | yes | computable from observed prices alone |
| `Phi(d1)`, BS gamma, `sigma*sqrt(tau)` | **no** | these *are* the answer, or trivially yield it |
| implied volatility surface features | deferred | observable in principle, but not in our simulated worlds; revisit only with real data |
| Heston variance `v_t` | yes, **in Heston worlds only** | the optimal policy genuinely depends on it, so excluding it would handicap every method equally but make the comparison uninformative |

**Consequence worth stating in the paper.** Rung 4 — "the learned policy recovers
`Phi(d1)`" — is a statement about the **GBM world only**. Under Heston the state necessarily
includes variance, the optimal policy is not `Phi(d1)`, and no analogous closed-form gate
exists. Rung 3 (characteristic-function prices) is what carries verification there instead.

### 2.2 Normalisation — fixed constants, never batch statistics

Raw spot near 100 and `tau` in `[0,1]` differ by two orders of magnitude and will train
badly. Two rules.

**Prefer dimensionless features over learned scaling.** Using `S/K` (order 1) and `tau/T`
(order 1) makes the inputs naturally comparable, and `delta_prev` is already order 1. With
that feature choice, no additional normalisation layer is required at all — *feature choice
is the normalisation*, and it has the further benefit of making the policy scale-invariant,
so a policy trained at `S0 = 100` transfers to any spot level.

**Never normalise using batch statistics.** Batch normalisation or per-batch standardisation
is disqualified for two independent reasons:

- It couples paths within a batch, so one path's hedge depends on other paths' prices. That
  is a mild but real information leak and is indefensible in a hedging context.
- Batch composition depends on shuffling, which breaks bit-reproducibility — the claim the
  whole seeding audit exists to protect.

Any fixed normalisation constants are recorded in the config, not computed at runtime.

### 2.3 Output parameterisation — absolute, unconstrained

The network outputs `delta` directly, with **no output activation**.

Two tempting alternatives are both rejected, and for the same underlying reason.

**A sigmoid output** would constrain a call hedge to `[0,1]`, which happens to be correct.
It hands the network the answer's range. That the network *finds* `[0,1]` unaided is itself
evidence the setup is sound, and constraining it forfeits that evidence.

**A residual parameterisation**, `delta = delta_BS + correction`, is more subtle and more
dangerous. §8 of the paper tests the claim from arXiv:2605.21696 that learned policies are
largely *delta corrections* rather than novel strategies. **Parameterising the output as a
correction assumes that finding rather than testing it**, and would make §8 vacuous. It also
smuggles `Phi(d1)` into the architecture, violating §2.1.

Recorded as a modelling choice with its reasoning, per `docs/03`.

### 2.4 Training-data regime — fresh paths per batch

Buehler et al. draw new paths every batch. We follow, and the consequences need stating
because they change what standard vocabulary means here.

```
consequence 1   the model never sees the same path twice
                -> classical overfitting is not the failure mode
consequence 2   "epoch" is meaningless; count GRADIENT STEPS
                -> the training budget control in the protocol is
                   measured in gradient steps and paths consumed
consequence 3   there is no held-out split
                -> evaluation uses a DISJOINT SEED STREAM, which
                   dhbench.seeding already provides via stream labels
consequence 4   total paths consumed = batch_size x gradient_steps
                -> this is the number to report, not "epochs"
```

Streams to use: `"init"` for weight initialisation, `"train"` for training paths, `"eval"`
for evaluation paths. Disjointness then holds by construction rather than by an additive
offset a caller can forget.

### 2.5 What constitutes a run

For rung 6 to be satisfiable later, this must be pinned now rather than retrofitted.

A run is `(config file, replicate index)` → deterministic outputs. Requires: weight
initialisation seeded from `derive_seed(k, "init")`; training and evaluation paths from
their own streams; TensorFlow op determinism enabled; and no dependence on wall-clock,
process ID, dict ordering, or unpinned library versions. The reproducibility test is
bit-identical metrics on a repeat run — identical, not `approx`.

---

## 3. Build sequence

Seven steps. Each has an acceptance criterion checkable **without** the step that follows,
so a failure localises to one place. This is the discipline that made Stage 0's ladder work.

| # | Step | Acceptance criterion |
|:--|:--|:--|
| 1 | Risk measures | **done** — 22 tests, incl. the overflow proof |
| 2 | `GradientTape` on a toy problem | recovers the analytic minimum of a quadratic to 1e-6 |
| 3 | `FeedforwardAgent` construction and `call` | correct shapes; parameter count matches the config; deterministic given `"init"` seed |
| 4 | `hedge_path` rollout | **see §3.1 — the free decisive test** |
| 5 | Training loop | loss decreases; gradients non-zero for every trainable variable including CVaR's `w` |
| 6 | **Noise floor** | **see §3.2 — do this before any comparison** |
| 7 | Rung 4 | **see §4** |

### 3.1 Step 4 has a free, decisive test

Replace the network with a function returning `Phi(d1)`. The rollout must then reproduce
`delta_hedge_positions` **to floating-point tolerance**, since both compute the same object
by different routes.

This validates the loop mechanics completely — indexing, the `delta_prev` carry, the time
feature, the off-by-one at maturity — using machinery already verified in rung 2, and
entirely before any learning is involved. If it fails, the fault is in the rollout and
nowhere else.

A second check on the same step: assert `tape.gradient` returns non-`None` for the network's
variables when the rollout is differentiated. Detaching `delta_prev` — via `stop_gradient`,
or by a NumPy round trip inside the loop — silently converts the policy to a myopic one-step
rule that trains, converges, and is wrong.

### 3.2 Step 6 — establish the noise floor before measuring anything

**Train the same configuration under five different replicate seeds and measure the spread
of the final evaluation metric.**

That spread is the **resolution limit** of every comparison in the paper. A method
difference smaller than it is not a finding, whatever the point estimates say.

Doing this at Stage 2 rather than Stage 5 is a deliberate scheduling choice, because the
answer determines whether the planned grid is viable at all:

```
if seed spread << method differences   -> the grid works as designed
if comparable                          -> more seeds per cell, or common random
                                          numbers (see below), or fewer cells
if seed spread >> method differences   -> the headline question cannot be answered
                                          at this sample size. Better to know in
                                          Stage 2 than after 1,200 runs.
```

**Common random numbers.** Every method is evaluated on the *same* evaluation paths, drawn
from a fixed `"eval"` stream, and recorded in the protocol as a stated control.

> **Corrected 2026-09-05.** This section originally claimed shared paths give "free variance
> reduction" that "materially increases the power" of every comparison. Measured, the
> reduction is **1.1–1.4x, not an order of magnitude**: 1.4x on mean P&L, 1.2x on CVaR-95,
> 1.1x on P&L standard deviation. The two strategies produce genuinely different P&L
> distributions — the band trades far less — so path-level P&Ls are not tightly correlated,
> and for a tail statistic the paths populating each tail differ between strategies. Shared
> paths remain worth using and cost nothing, but they do **not** rescue the precision
> problem. See `docs/06-implementation-plan.md` §F.3.

---

## 4. Rung 4 protocol

The gate. `configs/gbm_zerocost_entropic.yaml`, zero transaction cost, GBM.

### 4.1 Evaluate on a constructed grid, not on sampled paths

Simulated paths concentrate near the money. Evaluating the learned hedge only where paths
happened to go **under-tests the wings** — deep in- and out-of-the-money, which is precisely
where the policy is least constrained by training data and most likely to be wrong.

The acceptance grid is therefore **constructed**: a mesh over moneyness `S/K` in `[0.8, 1.2]`
crossed with `tau` in `[0.1, 1.0]`, evaluated directly. Some of that mesh is off the training
distribution, and the report must say so — a policy that is accurate on-distribution and
wrong in the wings is a *finding about extrapolation*, not a failure of the test, and it
connects directly to the fragility question in §8.

### 4.2 Acceptance

```
mean absolute deviation from bs_delta  < 0.05
maximum deviation at any grid point    < 0.15
plus the overlay plot, which is more convincing than either threshold
```

Report the deviation surface, not only the summary statistics. Where the error concentrates
is informative: near-expiry at-the-money error is expected and benign, since gamma diverges
there and the discrete-time optimum genuinely departs from `Phi(d1)`; error in the wings is
not benign and indicates under-training or poor extrapolation.

### 4.3 If it fails

Diagnostic order is fixed in advance so that debugging does not become a search:

```
1. rungs 1 and 2 still green?          (the foundation)
2. step 4's analytic-policy test?      (the rollout)
3. is the rollout inside the tape?     (gradients exist at all)
4. is delta_prev detached?             (myopic policy)
5. are inputs dimensionless?           (conditioning)
6. does the entropic loss use logsumexp? (already tested, but check)
7. is the training budget simply too small? (loss still descending)
```

Only after all seven does the architecture come into question.

### 4.4 The second Stage 2 gate

The **indifference price** — from two training runs, with and without the claim — must
approximate the Black–Scholes price of 7.9656 in the frictionless limit, within 5%.

This is an economic check rather than a numerical one and is genuinely independent of rung
4: a policy can have approximately the right *shape* while being mispriced. Slow, since it
needs two full runs, so mark it `slow` and exclude it from the default suite.

---

## 5. Compute budgeting

**Measure wall-clock per training run as soon as step 5 works**, and record it. Every
downstream scoping decision depends on that single number and nothing else can be planned
without it.

```
runs = worlds x costs x risk measures x agents x seeds
     = 4 x 4 x 3 x 5 x 5  ~=  1,200          not feasible on free Colab
```

The tiered design in `paper/STRUCTURE.md` §4 is the response: full factorial on a small core
carrying RQ1 and RQ3, one-factor-at-a-time on the periphery, and every omitted cell **logged
as omitted with its reason**. Silent truncation reads as full coverage when it is not.

If the measured per-run cost makes even the tiered grid infeasible, the correct response is
to **narrow the method set rather than to reduce seeds**. Seeds are what make the error bars
honest, and §5.4 of the technical report is precisely about not compromising them.

---

## 6. Critical path and risks

```
step 2 ──> step 3 ──> step 4 ──> step 5 ──> step 6 ──> RUNG 4 ──> Stage 3
                        ^                      ^
                        |                      |
              free analytic test        noise floor: decides
              localises loop bugs       whether the grid is viable
```

| Risk | Early signal | Response |
|:--|:--|:--|
| Rollout has a subtle indexing bug | step 4's analytic test fails | localised by construction; fix before proceeding |
| Gradients severed by a detach | step 5 shows zero or `None` gradients | inspect the loop; `pnl.py` is already guarded by test |
| Training too slow for the grid | step 5 timing | narrow the method set, not the seed count |
| Seed noise swamps method differences | step 6 | more seeds, common random numbers, or narrow the claim |
| Rung 4 fails on the wings only | §4.1 deviation surface | a finding about extrapolation, feeds §8 |
| Rung 4 fails everywhere | §4.3 checklist | work stops until resolved |

---

## 7. Kill criteria

Stated in advance, because the time to decide what counts as failure is before it happens.

- **Rung 4 unresolved after the §4.3 checklist is exhausted twice.** Stop feature work. The
  fault is a modelling assumption, not a bug, and the correct next move is to reproduce
  Buehler's own published setup exactly — including their architecture and hyperparameters —
  and bisect toward ours.
- **Seed noise exceeds plausible method differences at step 6.** The headline comparison
  cannot be made at the available sample size. Narrow to the regime-fragility question
  (§8), which compares a method against *itself* across distributions and is therefore
  paired and far less noise-limited.
- **Per-run cost makes the tiered grid infeasible.** Reduce the method set to the two or
  three best-validated implementations, and say so.

None of these is failure of the project. Each has a defensible paper attached to it, which
is the point of writing them down now.

---

## 8. Timeline

Estimates assume part-time work and are deliberately conservative.

| Steps | Content | Estimate |
|:--|:--|:--|
| 2–4 | tape, agent, rollout | 1–2 weeks |
| 5 | training loop, first convergence | 1–2 weeks |
| 6 | noise floor | 3–5 days |
| 7 | rung 4, overlay plot, indifference price | 1–2 weeks |
| | **Stage 1–2 total** | **5–8 weeks** |
| | Stage 3 (Heston, recurrent, CVaR, rung 5) | 6–10 weeks |
| | Stage 4 (freeze protocol, resolve the seven decisions) | 3–4 weeks |
| | Stage 5 (re-implementations with two-stage validation, grid) | 3–4 months |
| | Stage 6 (writeup) | 4–6 weeks |

**Roughly 9–12 months to submission at part-time pace.** The dominant cost is Stage 5, and
within it the two-stage validation protocol (`paper/STRUCTURE.md` §3.1), which roughly
doubles per-method work and is the strongest argument for committing to fewer methods,
properly validated, rather than broad coverage.

---

## 9. Immediate next action

Step 2: `tf.GradientTape` on a toy quadratic. No finance, no hedging, ten lines. The point
is to make the tool boring before it is load-bearing, so that when step 5 misbehaves the
tape is not among the suspects.
