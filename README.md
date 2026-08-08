# Deep Hedging Benchmark

> A reproducible benchmark for deep hedging: do learned policies actually beat classical
> transaction-cost bands, and how much of any advantage survives a regime shift?

**Status:** 🟨 Stage 0 — classical foundation. Nothing implemented yet.
**Author:** Aditya Bhatia, Stevens Institute of Technology
**Started:** 2026-08-08

---

## Why this exists

[Deep hedging](docs/01-papers.md) (Buehler et al., 2019) reframes derivative hedging as a
learning problem: a neural network outputs a hedge position at each timestep, trained to
minimise a convex risk measure of terminal P&L under transaction costs. The idea works,
and the subfield has moved fast — 2025–26 alone brought distributional adversarial
training (NeurIPS 2025), ambiguity-averse hedging, no-transaction band priors, AlphaZero
search for non-convex costs, and robust hedging valuation adjustment.

Every one of those papers uses **its own simulator, its own cost model, its own risk
measure, and its own metrics.** A 2023 review in *Mathematics* names the consequence:

> the lack of a standardized testing dataset or universal benchmark in the RL hedging
> space makes it difficult to compare results across different studies

That is still true. This repository is an attempt to fix it.

## Research question

> Under one fixed protocol — same simulators, same cost models, same risk measures, same
> seeds, same metrics — do deep hedging agents outperform classical transaction-cost-aware
> hedging, and how much of any advantage survives a regime shift between training and test?

Five sub-questions, each a section of [PAPER.md](PAPER.md):

1. **Recovery** — at zero cost under Black–Scholes, does the agent recover the BS delta?
2. **Frictions** — does the learned policy reproduce the Whalley–Wilmott band, and beat it?
3. **Architecture** — feedforward vs. recurrent vs. band-structured, at equal parameters
   and equal training budget.
4. **Regime fragility** — train on Heston, test on regime-switching. Quantify the damage.
5. **Risk measure sensitivity** — does the ranking of methods change under entropic vs.
   CVaR₉₅ vs. mean-variance? If it does, cross-paper comparison is meaningless.

## What makes this tractable

**There is no dataset to acquire.** All training and evaluation data is simulated — no
vendor cost, no licensing, no NDA. More importantly, the ground truth is *known*, so
correctness is checkable rather than guessable. See [docs/02-data-sources.md](docs/02-data-sources.md).

---

## Structure

```
dhbench/
  pnl.py          P&L accounting — the single source of truth
  worlds/         market simulators: GBM, Heston, regime-switching, jumps
  costs/          transaction cost models
  objectives/     convex risk measures: entropic, CVaR, mean-variance
  agents/         learned policies: feedforward, recurrent, band, robust
  baselines/      BS delta, Whalley-Wilmott, Zakamouline
  evaluation/     metrics and stress tests
configs/          one YAML per experiment — the reproducibility layer
tests/            the correctness ladder (see below)
```

## The correctness ladder

Each rung must pass before the next one means anything. These live in [tests/](tests/) and
are the project's spine — a deep hedging result is worthless if the P&L accounting beneath
it is wrong, and that is the most common way these reproductions fail.

| # | Test | Gate |
|:--|:--|:--|
| 1 | MC European call price ≈ closed-form Black–Scholes | simulator + pricing |
| 2 | Zero-cost, fine-grid delta hedge → P&L std → 0 | P&L accounting |
| 3 | Heston reproduces characteristic-function prices | stochastic vol simulator |
| 4 | **Learned agent ≈ BS delta at zero cost** | **the headline gate** |
| 5 | Learned band ≈ Whalley–Wilmott at small costs | frictions |
| 6 | Every experiment bit-reproducible from config + seed | the benchmark claim |

Rungs 1–3 need no neural networks at all. Do them first.

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pytest                             # expect failures — that is the point
```

Tests fail until you implement the bodies. The scaffold defines *what* must be true;
filling it in is the work.

**Compute.** Stages 0–2 run on a CPU. Stage 3 onward wants a GPU — use Google Colab.
Local AMD/integrated GPUs are not a supported TensorFlow path; see
[docs/02-data-sources.md](docs/02-data-sources.md#compute).

## Roadmap

| Stage | Content | Status |
|:--|:--|:--|
| 0 | Classical foundation — GBM, MC pricing, delta hedging, Whalley–Wilmott | 🟨 in progress |
| 1 | `GradientTape`, custom training loops, model subclassing | ⬜ |
| 2 | Vanilla deep hedging — **gate: learned δ overlays BS delta** | ⬜ |
| 3 | Frictions, Heston, recurrent agent | ⬜ |
| 4 | Freeze the benchmark protocol | ⬜ |
| 5 | Re-implement published variants; run the grid; report nulls | ⬜ |
| 6 | Writeup | ⬜ |

Full detail in [PAPER.md](PAPER.md). Session-by-session notes in
[docs/04-progress-log.md](docs/04-progress-log.md).

## Related

Companion repository: [TensorFlow-ML-DL](https://github.com/Adi7710/TensorFlow-ML-DL) —
where the TensorFlow fundamentals are being learned. Coordination between the two runs
through [HANDOFF.md](HANDOFF.md).

## License

MIT — see [LICENSE](LICENSE).
