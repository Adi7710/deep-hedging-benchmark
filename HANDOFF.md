# Handoff

Coordination file between two Claude Code sessions:

- **Course session** — `TensorFlow-ML-DL`, working through Bourke's course
- **Research session** — this repo, building the deep hedging benchmark

Each session reads this at the start and updates it at the end. It works whether or not
live cross-session messaging is available, and it survives restarts.

**Keep it short.** Current state only. History belongs in
[docs/04-progress-log.md](docs/04-progress-log.md).

---

## The two repos are independent

**This project is not gated on the course.** TensorFlow needed here is learned here, from
the [current TF/Keras documentation](https://www.tensorflow.org/api_docs), at the point it
is needed. The course is a parallel track for breadth, not a prerequisite for depth.

That matters because the course covers only `model.fit`, and deep hedging fundamentally
cannot use it — there are no labels, and the loss is a functional of the whole simulated
path. `tf.GradientTape` and custom training loops are learned in Stage 1 of *this* repo,
directly from the docs.

Where the course genuinely helps: shared vocabulary (tensors, shapes, layers) and the RNN
material that maps onto the recurrent agent. Where it doesn't: everything structural about
deep hedging. Don't wait on it.

---

## Current state

**Date:** 2026-08-19
**Research stage:** 0 — **complete.** Next: Stage 1 (GradientTape, custom training loops).
**Ladder rungs green:** **1 and 2**, plus the rung-5 baseline check.
Suite: 49 passed, 7 skipped, 0 failed. The skips are rungs 3, 4, 6 and the
learned-band half of rung 5 — all need components that do not exist yet.

---

## Stage 0 checklist

Turn the red tests green, in this order. No neural networks in any of it.

- [x] `dhbench/pnl.py` — hedging gains, transaction costs, turnover, terminal P&L
- [x] `dhbench/worlds/gbm.py` — GBM simulator (exact solution, not Euler)
- [x] Rung 1 green: `pytest tests/test_rung1_mc_price.py`
- [x] Rung 2 green: `pytest tests/test_rung2_pnl_accounting.py` — 13/13
- [x] `dhbench/baselines/whalley_wilmott.py` — the band that actually matters

**Stage 0 is done.** All four classical components implemented and verified.

**Audited 2026-08-25.** `gbm.py` and `pnl.py` reviewed for publication readiness. Fixed:
seed derivation (see below), graph-mode shape handling in `terminal_pnl`, missing test
coverage of the discounting path, missing input validation. Cleared on inspection: float32
precision (matches float64 to 6 dp), the Ito correction, terminal-liquidation accounting.

**Use `dhbench.seeding.make_generator(k, stream)`, never `tf.random.Generator.from_seed(k)`.**
Small consecutive seeds are not independent: they biased an MC call price by 9.3 SE and
made cross-seed error bars 2.5x too narrow. Enforced by `tests/test_seeding.py`.

**Paper shape decided 2026-08-25:** finding-first, not benchmark-first. Three research
questions replace a single results section. See `paper/STRUCTURE.md`.

**Stage 1-2 execution plan written 2026-08-29:** `docs/05-stage-1-2-plan.md`. Settles four
modelling decisions before any code (information set, normalisation, output
parameterisation, fresh-paths regime), and schedules the seed noise floor at step 6 —
before any method comparison — because it decides whether the grid is viable at all.

**REFRAMED 2026-09-05.** Measurement showed that at realistic hedging costs (5bp, index
options in futures) a two-point volatility error costs 6.4x what transaction costs do:
CVaR-95 effect -1.386 for realised 22% vs hedged 20%, against -0.217 for 5bp cost. The
benchmark had made the second-order friction central while treating the first-order one as
solved. Question is now "under model misspecification AND frictions, do learned policies
beat classical rules -- and which failure mode dominates?" Master plan, including the value
assessment and data sources: `docs/06-implementation-plan.md`.

**Measured statistical design** (docs/06 Part F): CVaR-95 is ~5x noisier than the mean at
equal path count (SE 0.038 vs 0.0071 at N=20k). MDE at 5 seeds is 0.060 on CVaR-95 against
a band-vs-delta effect of 0.131, so anything smaller than that effect is marginal. Common
random numbers give only 1.1-1.4x variance reduction, not the order of magnitude docs/05
originally claimed -- now corrected there.

**RETRACTED 2026-09-05:** the "centre rule captures 7% of the available improvement"
figure was a single-seed point estimate of a ratio that is not estimable -- across 20 seeds
it is mean 24%, sd 27%, range spanning zero. Defensible replacement: the edge rule delivers
0.099 more CVaR improvement than the centre rule (t=10.9, consistent 20/20). Corrected in
docs/06, paper/00-draft.md, paper/STRUCTURE.md and the technical report PDF. Caught only by
moving measurements into `experiments/findings.py` -- exactly the error the paper criticises.

**Every paper number now regenerates:** `python -m experiments.findings --all`.
`tests/test_findings.py` pins each claim, including a NEGATIVE test that the withdrawn
ratio stays unquotable.

**Steps 1-3 of the Stage 1-2 plan done.** Objectives (entropic, CVaR, mean-variance);
GradientTape verified on a toy quadratic to 4.8e-7; FeedforwardAgent construction and
forward pass. Suite 85 passed, 7 skipped. **Next: step 4, the rollout** -- and it has a
free decisive test, swap the network for Phi(d1) and it must reproduce
delta_hedge_positions.

**Use `seeding.seed_keras(k, "init")` for weight initialisation**, not `set_random_seed`
directly: Keras forwards to numpy.random.seed, which rejects seeds >= 2**32, and
`derive_seed` returns 63 bits.

**Seven protocol decisions must be resolved before Stage 4 freezes** — three of them serious.
Re-implementation validation as currently specified is logically impossible; there is no
Whalley-Wilmott band for a CVaR objective, so two of three risk-measure columns have no
well-posed classical comparison; and Heston is an incomplete market, so stock-only hedging
would confound the result. Full list with recommendations in `paper/STRUCTURE.md` section 3.

**Decided this session:** the benchmark is specified in **discounted (time-0) units**.
The P&L functional is form-invariant under the change of numéraire, so `hedging_gains`
and `transaction_costs` take no rate argument; `terminal_pnl` is the sole site of
discounting. Stage 0–2 configs stay at `r = 0` regardless, so a discount-factor bug and a
simulator bug stay distinguishable. See `paper/00-draft.md` §3.2.1.

**Corrected this session:** `docs/00`, `docs/01` and `PAPER.md` each contained a spec
error that would have biased results toward deep hedging — trading back to the delta
instead of the band edge, and a cost sum stopping at `T-1`.

## Where the maths lives

`notebooks/01-the-maths.ipynb` — every equation, typeset, plus a section that checks them
against the code. The terminal cannot render LaTeX; that notebook can.

## TensorFlow concepts, as they land

Ticked when used and understood **in this repo**. Not a dependency list — a record.

- [x] Tensor creation, shapes, dtypes
- [x] `tf.random.Generator` — explicit seeding *(Stage 0)*
- [x] Aggregation and `axis` semantics — `reduce_mean`, `reduce_sum`, `reduce_logsumexp` *(Stage 0)*
- [x] `tf.GradientTape` *(Stage 1)* — custom training loop still to come at step 5
- [x] Model subclassing (`keras.Model`) *(Stage 1)*
- [ ] `@tf.function` and graph mode *(Stage 2, after it works eagerly)*
- [ ] `tf.data` pipelines *(Stage 3)*
- [ ] LSTM/GRU cells, stepped manually *(Stage 3)*

---

## Blocked

| What | Blocked on | Workaround |
|:--|:--|:--|
| Local GPU training | AMD 860M unsupported by TF | Colab from Stage 3 |
| Cross-session messaging | WSL2 — Virtual Machine Platform component not enabled | this file |

Nothing is blocked on the course.

---

## Open questions

- ~~Does Stevens provide **WRDS / OptionMetrics** access?~~ ✅ **Confirmed 2026-08-12.**
  Use **SPX** (European) — *not* single-name equity options, which are American and would
  silently change the problem. Three staged uses and the licensing constraints are in
  [docs/02-data-sources.md](docs/02-data-sources.md#wrds--optionmetrics--access-confirmed-2026-08-12).
  **Not before Stage 0 is done** — it expands what the paper can claim, not what to build next.
- Confirm whether the AMD 860M (gfx1152) works with ROCm on WSL2 in practice. `librocdxg`
  1.2 added the GFX target in May 2026, but AMD's official matrix lists discrete cards
  only. Low priority; Colab is the plan regardless.

---

## Course session → research session

_Record here if the course covers something worth reusing — a plotting helper, a debugging
trick. Not a gate on anything._
