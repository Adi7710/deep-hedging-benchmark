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

**Date:** 2026-08-08
**Research stage:** 0 — classical foundation
**Ladder rungs green:** none yet (15 Black-Scholes reference tests pass; rungs 1–2 red by design)

---

## Stage 0 checklist

Turn the red tests green, in this order. No neural networks in any of it.

- [ ] `dhbench/pnl.py` — hedging gains, transaction costs, turnover, terminal P&L
- [ ] `dhbench/worlds/gbm.py` — GBM simulator
- [ ] Rung 1 green: `pytest tests/test_rung1_mc_price.py`
- [ ] Rung 2 green: `pytest tests/test_rung2_pnl_accounting.py`
- [ ] `dhbench/baselines/whalley_wilmott.py` — the band that actually matters

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

- Does Stevens provide **WRDS / OptionMetrics** access? Highest-leverage non-coding task —
  it materially strengthens the paper's real-data section. Ask the library.
- Confirm whether the AMD 860M (gfx1152) works with ROCm on WSL2 in practice. `librocdxg`
  1.2 added the GFX target in May 2026, but AMD's official matrix lists discrete cards
  only. Low priority; Colab is the plan regardless.

---

## Course session → research session

_Record here if the course covers something worth reusing — a plotting helper, a debugging
trick. Not a gate on anything._
