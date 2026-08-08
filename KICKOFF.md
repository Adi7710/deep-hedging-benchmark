# Starting a working session

Open a Claude Code session **with this repository as the working directory**:

```
c:\Users\adity\deep-hedging-benchmark
```

Claude reads [CLAUDE.md](CLAUDE.md) and [HANDOFF.md](HANDOFF.md) automatically at session
start, so it will already know the conventions and the current stage. The prompt below
sets the *working mode* — how you want to be taught — which is the part it can't infer.

Paste everything between the rules.

---

I'm building this deep hedging benchmark as a research project, aiming at an arXiv
preprint or thesis chapter. Read `README.md`, `CLAUDE.md`, `HANDOFF.md`, and
`docs/00-problem-statement.md` first, then confirm you understand the shape of the project
before we start.

**How I want to work:**

1. **Go step by step, and wait for me.** One component at a time. Don't move to the next
   until I've implemented the current one and its tests pass. Don't produce a plan for the
   whole stage and then execute it — stop at each step and check I'm with you.

2. **I write the code, you teach.** The `NotImplementedError` bodies are mine to fill.
   Explain the concept, derive the equation, tell me the shape contract and the specific
   way it tends to go wrong — then let me write it. Review what I write and be blunt about
   bugs. If I'm stuck after a genuine attempt, give me a hint before giving me code.

3. **Teach me the paper before the code.** I need to actually understand this literature,
   not just get tests passing. Before each component, walk me through the relevant section
   of the relevant paper — the setup, the equation, why it's that equation and not another
   one. `docs/01-papers.md` says what to read and when. Assume I have a quantitative
   background but have not read these papers.

4. **Use current TensorFlow, not a course.** I'm doing Bourke's TF course in parallel, but
   this project is not limited to it and is not waiting on it. Teach me whatever the task
   needs — `tf.GradientTape`, custom training loops, `keras.Model` subclassing — from the
   current TF/Keras 3 docs, when we get there. Don't hold back an API because a beginner
   course hasn't covered it.

5. **Tell me when I'm wrong about the finance, not just the code.** If my reasoning about
   hedging, risk measures, or the experimental design is off, say so directly.

**Where we are:** Stage 0. Rungs 1 and 2 of the correctness ladder are red; the 15
Black–Scholes reference tests pass. Nothing is implemented yet.

**Start with:** the theory behind `dhbench/pnl.py` — the P&L functional in
`docs/00-problem-statement.md`. Explain where each term comes from, in particular why the
transaction cost sum runs to `n` rather than `n-1`, and why we're short the claim. Then let
me implement `hedging_gains` and we'll check it against
`tests/test_rung2_pnl_accounting.py`.

Before any of that, tell me roughly how long Stage 0 should take and what I should read
tonight.

---

## What the session should cover, in order

Rough shape, so you can tell if it's drifting. Each numbered item is a stop-and-check point.

**Stage 0 — classical foundation, no neural networks**

| # | Component | Read alongside |
|:--|:--|:--|
| 1 | `pnl.py` — the P&L functional | `docs/00-problem-statement.md` |
| 2 | `worlds/gbm.py` — GBM simulator | any stochastic calculus refresher; the exact-solution form |
| 3 | rung 1 green: MC price ≈ Black–Scholes | — |
| 4 | rung 2 green: P&L accounting + convergence | — |
| 5 | `baselines/whalley_wilmott.py` | Whalley & Wilmott (1997) |

**Stage 1 — the TensorFlow you actually need**

`tf.GradientTape`, custom training loops, `keras.Model` subclassing, `@tf.function`.
Learned from the docs, on a toy problem, before touching the real one.

**Stage 2 — vanilla deep hedging**

Buehler et al. (2019) §1–3 properly, then `agents/feedforward.py` and the training loop.
The gate is rung 4: **the learned hedge must overlay Φ(d₁)**. Don't go past it until it does.

## Reading order

Don't read everything up front — it's a lot, and most of it only makes sense once you've
built the thing it describes. In order of when you need it:

1. **`docs/00-problem-statement.md`** — tonight. Our notation, ten minutes.
2. **Whalley & Wilmott (1997)** — before component 5. The band formula and why `H ~ c^(1/3)`.
3. **Buehler et al. (2019) §1–3** — before Stage 2. The core paper; §4 onward can wait.
4. **Föllmer & Schied ch. 4** — skim, alongside `objectives/`. Enough to know what convexity
   buys you.
5. **Carbonneau (2020)** — before the recurrent agent, Stage 3.
6. **The 2025–26 wave** — Stage 4 onward, when you're choosing what to benchmark.

Full annotated list with links: [docs/01-papers.md](docs/01-papers.md).

## Keeping the two sessions in sync

Update [HANDOFF.md](HANDOFF.md) at the end of each session — stage, which ladder rungs are
green, what you're stuck on. Longer notes go in
[docs/04-progress-log.md](docs/04-progress-log.md). Both sessions read `HANDOFF.md` at
start, so it's the only thing that has to stay current.
