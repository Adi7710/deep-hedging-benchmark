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
- [ ] `tf.random.Generator` — explicit seeding *(Stage 0)*
- [ ] Aggregation and `axis` semantics — `reduce_mean`, `reduce_sum`, `reduce_logsumexp` *(Stage 0)*
- [ ] `tf.GradientTape` and custom training loops *(Stage 1 — learn from the docs, here)*
- [ ] Model subclassing (`keras.Model`) *(Stage 1)*
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
