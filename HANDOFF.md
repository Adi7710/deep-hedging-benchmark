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

**Date:** 2026-09-26
**Research stage:** 1-2. Steps 1-6 of `docs/05-stage-1-2-plan.md` done; step 7 (rung 4) waits on
its re-specified gate (below). **Target now: Agenthon 2026 workshop paper** (NeurIPS 2026
satellite, Atlanta, Sat 12 Dec; due **Wed 30 Sep 23:59 AoE**, non-archival, in-person poster).
Draft: `paper/workshop/main.tex` -> `main.pdf`. Longer term: NeurIPS 2027 Evaluations & Datasets.
**Ladder:** rungs 1 and 2 green; rung-5 baseline half green; rung 4 NOT claimed either way.
Suite: 149 passed, 7 skipped, 0 failed.

**Author decisions outstanding for the workshop submission:** attend Atlanta in person (required
for acceptance); approve the AI-assistance disclosure wording (draft in the paper's ack); title;
whether to add a short section on errors the AI-assisted workflow made and the checks caught;
whether to file the TensorFlow issue (text drafted, NOT filed). Submission is the author's, via
the form in the CFP.

### Standing rules (unchanged)

- **Seeds:** `dhbench.seeding.make_generator(k, stream)`, never `tf.random.Generator.from_seed(k)`
  with an int. An int seed lands in the Philox COUNTER under key 0, so seed k is seed 0 shifted by
  4k draws: false significance (+0.85 SE reads as +9.3), not bias. `from_seed([k])` (a list) puts k
  in the KEY and is safe. TF's own `_make_1d_state` pads left to avoid exactly this; ints bypass it.
  Pinned by `tests/test_seeding.py`.
- **Weight init:** `seeding.seed_keras(k, "init")`, not `set_random_seed` (NumPy rejects >= 2**32).
- **Units:** discounted (time-0); `terminal_pnl` is the sole site of discounting; configs at r = 0.
- **tf.function** is worth 16.9x (92.4 -> 5.5 ms/step); the first step stays eager on purpose.
- **Every paper number regenerates** from `python -m experiments.findings <name>` (or `--all`), and
  `tests/test_findings.py` pins each claim, including negative tests for withdrawn ones.
- **No claim about another paper** unless recorded with page references in
  `docs/07-literature-audit.md`. No claims about "the literature" in general.

### What the measurements now say (corrected 2026-09-26)

- **Pseudo-replicates** (above). TF's guide does warn about overlapping streams from `from_seed`;
  the contribution is the deterministic int path and its measured consequence.
- **Band baseline:** edge beats centre by 0.099 CVaR-95 (t = 10.9, 20/20). Rehedging to delta is
  Whalley-Wilmott's own "market movement" rule, "commonly used in practice" -- a distinct strategy,
  not "a common error": every implementation audited trades to the edge. The centre/edge ratio is
  not estimable (withdrawn "7%").
- **Friction ordering** (12 contracts): +10% adverse vol error beats 5bp cost 2.7-7.1x, flips at
  25-50bp. On the MEAN, closed forms hold (vol exact to 0.3%; Leland + open/close cost to 3.5%);
  CVaR-95 amplifies vol 1.6-3.0x vs cost 1.1-2.1x, so at 25bp costs dominate the mean in 10/12 but
  CVaR in 6/12 -- the ordering depends on the risk measure. Calibrated: realised S&P vol >= 1.1 x
  VIX in 8.8% of months (1990-2026); ES half-spread 0.16bp, median S&P 500 stock 1.6bp; 25-50bp is
  small/micro-cap (NOT "single stock"). `experiments/market_calibration.py`; raw data never in git.
- **Seeds:** noise floor from 30 replicates, sd 0.144 [0.115, 0.193] on CVaR-95 (2,000 steps).
  Five seeds: power 0.36 for the band-vs-delta effect. **80% power: 12 seeds vs a fixed baseline
  (8-19 across the sd CI), 20 per arm learned vs learned** -- not "six minimum, ten headline",
  which came from calling a CI half-width an MDE. CVaR-95 SE is ~4x the mean's (not ~5x).
  Pairing helps the mean 1.42x, CVaR-95 only 1.19x.
- **At 8,000 steps** (`noise_floor_long`): typical spread more than halves (robust sd 0.160 ->
  0.063) but ONE replicate in 30 (#20) destabilised -- loss 0.65 -> 1.4 after step 3,000, CVaR-95
  2.833, robust z 11 -- and alone lifts the sd to 0.134 (0.059 without it). "Last decile beats
  first" passes that run; final-vs-best-window loss regression (+55%) flags it. Failed runs are
  reported, never dropped; seed counts must be set at the budget actually used.
- **Rung 4:** the old gate was underspecified. Same trained networks pass 1/6 (grid, previous
  position = 0), 6/6 (grid, previous position = Phi(d1)), 6/6 (visited states). Networks depend on
  the previous-position input (0.052 +- 0.019) though the zero-cost optimum cannot -- a shortcut to
  rule out before any rung-5 learned-band claim. Gate re-specified in
  `tests/test_rungs_3_to_6.py`; not yet re-run.
- **Vol-robust training is prior art** (Lutkebohmert, Schmidt & Sester 2022; He et al. 2025 train
  across parameter uncertainty). Our contribution there would be the comparison, not the method.

**Seven protocol decisions must be resolved before Stage 4 freezes** — full list in
`paper/STRUCTURE.md` section 3 (seed counts in section 4 now use the power numbers above).

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
