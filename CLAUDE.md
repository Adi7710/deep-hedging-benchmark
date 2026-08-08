# Deep Hedging Benchmark — instructions for Claude

Research repository. The goal is a **publishable benchmark**, not a working demo.
Rigour and reproducibility are the contribution; correctness beats speed everywhere.

## Working relationship

**Aditya implements; Claude teaches, scaffolds, and reviews.**

- **Do not fill in `NotImplementedError` bodies unless explicitly asked.** Scaffold
  signatures, state the maths, write the tests that define correctness.
- When he's stuck, explain the concept and point at the equation — don't hand over code.
- Two exceptions, already written in full because they're reference material rather than
  learning material: `dhbench/baselines/bs_delta.py` and everything in `tests/`.

**Teach whatever TensorFlow the task needs, from the current docs.** This project is *not*
limited to what the parallel Bourke course covers. Deep hedging cannot use `model.fit` at
all — there are no labels, and the loss is a functional of the entire simulated path — so
`tf.GradientTape`, custom training loops, and `keras.Model` subclassing are learned here,
when Stage 1 needs them. Cite <https://www.tensorflow.org/api_docs> rather than course
material, and prefer current Keras 3 idioms over anything from the TF 2.7-era tutorials.

**Go step by step, and interleave the reading.** Before implementing a component, work
through the relevant paper section with him — see `docs/01-papers.md` for what to read and
when. Understanding the equation comes before typing it. Don't run ahead to the next
component while the current one is still unclear.

## Non-negotiables

- **`dhbench/pnl.py` is the single source of truth for P&L accounting.** Never compute
  P&L anywhere else. Most failed deep hedging reproductions trace to bookkeeping bugs
  hidden inside a training loop.
- **The correctness ladder gates everything.** Rungs 1–3 (`tests/`) must pass before any
  neural network result is meaningful. Do not chase a training bug before the simulator is
  verified.
- **Every experiment is defined by a YAML config in `configs/` plus a seed.** No
  hyperparameters hard-coded in scripts. Bit-reproducibility is a headline claim of the
  paper — breaking it breaks the contribution.
- **Report null results.** Where deep hedging ties or loses to Whalley–Wilmott, that goes
  in the table. Quietly dropping unflattering cells is the exact failure this paper
  criticises.

## Conventions

- TensorFlow 2.16+ / Keras 3. Plain TF + NumPy + SciPy only.
- **Do not depend on `hansbuehler/deephedging`.** Read it as a reference; it needs
  `tensorflow_probability` version-matched to TF and we are on TF 2.21.
- Simulators return shape `(n_paths, n_steps + 1)`. Hedge positions are `(n_paths, n_steps)`.
- Seed via `keras.utils.set_random_seed`, and pass explicit `tf.random.Generator`s into
  simulators — never rely on global state for a published number.
- Notebooks in `notebooks/` are for exploration and figures only. Anything that produces a
  number in the paper lives in `dhbench/` or `experiments/`.
- Type-hint public functions. Docstrings state the equation being implemented.

## Commit messages

```
<area>: <what changed>
```
Areas: `worlds`, `costs`, `objectives`, `agents`, `baselines`, `eval`, `docs`, `tests`,
`configs`, `paper`.

## Companion repository

`TensorFlow-ML-DL` — Bourke's course, running in parallel. Sibling checkout:
`c:\Users\adity\TensorFlow-ML-DL` on Windows, `~/projects/TensorFlow-ML-DL` under WSL2.

**It is a parallel track, not a prerequisite.** Nothing here waits on it. Read
`@HANDOFF.md` at the start of a session and update it at the end — it carries current
stage, ladder status, and open questions.

## Compute

CPU is fine through Stage 2. Stage 3+ needs a GPU — use Colab. The local machine has an
AMD Radeon 860M (gfx1152); it is **not** a supported TensorFlow GPU path, so don't suggest
local GPU training.
