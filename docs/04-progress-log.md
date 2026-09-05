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

---

## 2026-09-05 — The training loop works; policy converges toward Phi(d1)

**Stage:** 1–2 (steps 4 and 5 of `docs/05-stage-1-2-plan.md`)
**Suite:** 100 → 112 passing, 7 skipped, 0 failing

### What happened

`dhbench/training.py` written. Deep hedging now runs end to end: simulate, roll the policy
forward, tally P&L through `pnl.py`, score with a risk measure, backpropagate through every
hedging decision, step.

The world is **injected as a callable** rather than hard-coded. That was chosen so
vol-robust training (`docs/03`) becomes a change of argument, not a change of loop — a
world that draws sigma per batch instead of fixing it.

### Audit as it was built

**Does it train?** Loss falls 72 to 2 over 300 gradient steps. But loss decreasing proves
nothing about correctness, so:

**Does it learn the right thing?** Evaluated against Phi(d1) on a *constructed* moneyness
grid, not on sampled paths:

```
steps    sec    loss     MAD     max     ATM   wings
  500      7   1.538  0.1324  0.5030  0.1017  0.1499
 2000     19   0.861  0.1153  0.3743  0.0455  0.1505
 8000     67   0.863  0.0740  0.2193  0.0284  0.0880

rung-4 acceptance: MAD < 0.05, max < 0.15   -- NOT YET MET
```

Converging, monotonically, in the right direction. Rung 4 is not passed and is not claimed.

**The error is concentrated in the wings** — 0.088 against 0.028 at the money. This was
*predicted in advance* in `docs/05` section 4.1: simulated paths concentrate near the money,
so the wings are the least-trained region, and evaluating on sampled paths would have hidden
it. The constructed grid caught exactly what it was designed to catch. Whether that gap
closes with budget or is a genuine extrapolation limit is a step-7 question and feeds
section 8.

**How fast?** 92.4 ms per gradient step eagerly. This matters: `docs/06` section F.4 could
not size the grid without it.

### Impact: tf.function is worth 16.9x

```
eager         92.4 ms / gradient step
@tf.function   5.5 ms / gradient step

2000-step run   eager  185 s     compiled   11 s
300-run grid    eager 15.4 h     compiled  0.9 h
```

That is the difference between a grid that needs allocation and one that runs on a laptop
overnight. Compilation is now on by default, with the **first step deliberately left
eager** — it creates the weights and is where the missing-gradient diagnostic can still
produce a readable message. `docs/05` said to compile only once it works; that rule earned
its keep twice in one session.

### What broke

**Variables passed as a tf.function argument** became symbolic tensors, and the optimiser
failed with an AttributeError about SymbolicTensor lacking a unique id — an unreadable
error pointing nowhere near the cause. Fixed by capturing them by closure, which requires
defining the step *after* the variables are resolved. Written into the code comment rather
than just fixed, since it is exactly the class of tracing error the eager-first-step rule
exists to keep out of the diagnostic path.

**Keras autocasting** (found in step 4): with a float64 price path the float32 network
output collided with the float64 time grid on the *next* loop iteration, surfacing as an
opaque Mul error. `hedge_path` now casts once, where the reason can be documented.

### Pinned by test

Bit-identical loss history from the same replicate index. CVaR's auxiliary w actually
moves during training — a w that never moves is a free diagnostic that it was excluded
from the optimiser. A detached rollout raises with a message naming the likely cause rather
than silently training nothing. Compiled and eager agree to 1e-4.

### Next

Step 6, the seed noise floor — before any comparison, because it decides whether the grid
is viable. Then step 7, rung 4, which needs a budget sweep and possibly a learning-rate
schedule; the wing gap is the open question.
