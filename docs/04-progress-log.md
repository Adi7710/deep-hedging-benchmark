# 04 — Progress log

Dated session notes. Raw and unfiltered — what was done, what broke, what's still unclear.
The polished version lives in [PAPER.md](../PAPER.md); current state lives in
[HANDOFF.md](../HANDOFF.md). This file is the history neither of those keeps.

Also the audit trail for §9's honesty commitments: every literature search, every
abandoned approach, every result that didn't reproduce.

---

## 2026-08-08 — Repository created

**Stage:** 0 (classical foundation)
**Done by:** Claude, scaffolding only

### What happened

Project scoped and repository scaffolded. Research direction settled after surveying the
2025–26 deep hedging literature.

**Direction chosen:** a reproducible benchmark, not a new hedging algorithm. Rationale —
the robustness angle is already crowded (adversarial training at NeurIPS 2025,
ambiguity-averse hedging, band priors, AlphaZero, robust HVA all within twelve months),
but *none* of those papers share a protocol. The comparability gap is real, confirmed by
the 2023 *Mathematics* review, and closing it needs engineering rigour rather than new
mathematics — which is the right shape of contribution while still learning the tooling.

### Scaffolded

- Root: `README.md`, `PAPER.md`, `CLAUDE.md`, `HANDOFF.md`, `LICENSE`, `CITATION.cff`,
  `.gitignore`, `requirements.txt`
- Docs: problem statement, paper set, data sources, protocol placeholder, this log
- `dhbench/` package — signatures and docstrings only, bodies raise `NotImplementedError`
- `tests/` — the correctness ladder, failing by design until implementations land

### Decisions worth recording

| Decision | Reasoning |
|:--|:--|
| Separate repo, not a folder in the course repo | research artifact needs its own README, license, citation; the course repo is organised around per-module study logs |
| Don't depend on `hansbuehler/deephedging` | needs `tensorflow_probability` matched to TF; we're on TF 2.21. Reference reading only |
| Simulated data primary | ground truth is known, so correctness is checkable rather than guessable |
| `pnl.py` as single source of truth | most failed reproductions in this field are P&L bookkeeping bugs hidden in a training loop |
| Baseline is Whalley–Wilmott, not naive delta | beating naive delta under costs is trivial and proves nothing |

### Environment notes

- TensorFlow 2.21 / Keras 3.15 / NumPy 2.2.6, Python 3.12, **CPU only**
- Local GPU is not available: AMD Radeon 860M (gfx1152) isn't on AMD's supported ROCm WSL
  matrix. Colab from Stage 3
- WSL2 blocked on the Virtual Machine Platform component (needs admin + reboot). Doesn't
  block any repo work

### Literature search

Logged in [01-papers.md](01-papers.md#search-log). Re-run before Stage 4 and Stage 6.

### Next

Stage 0, step 1 — GBM simulator in TensorFlow, then Monte Carlo price a European call and
check it against closed-form Black–Scholes. That's rung 1 of the ladder.

**Not written by Claude** — the implementations are Aditya's. The scaffold defines what
must be true; filling it in is the work.

---

## Template for future entries

```markdown
## YYYY-MM-DD — <title>

**Stage:**
**Ladder rungs passing:**

### What I did

### What broke

### What I'm still unsure about

### Next
```
