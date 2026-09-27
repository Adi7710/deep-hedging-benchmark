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

---

## 2026-09-07 — Step 6: the noise floor says five seeds is not enough

**Stage:** 1–2 (step 6 of `docs/05-stage-1-2-plan.md`)
**Suite:** 112 → 127 passing, 7 skipped, 0 failing

### What happened

Built the metrics layer (`cvar_empirical`, `summarise`, `degradation_ratio`) and an
`evaluate` counterpart to `train`, then measured the training-seed noise floor — scheduled
before any method comparison because it decides whether the grid is answerable at all.

### Sign convention, fixed before it spread

`cvar_empirical` reports a **loss** (positive = bad, matching `CVaRRisk`), while
`experiments/findings.py` quotes P&L differences (higher = better). Two conventions for one
metric is precisely the comparability bug this project exists to remove, so `findings.cvar`
now delegates: `cvar(pnl) = -cvar_empirical(pnl)`. One implementation, the relation stated.

Verified the two CVaR routes agree exactly at the optimum — `cvar_empirical`, the NumPy
reference, and `CVaRRisk` at `w = VaR` all give 20.6004. That agreement doubles as a free
convergence diagnostic in real runs: a persistent gap means training stopped early.

### The measurement

8 replicates, 2,000 gradient steps each. Each replicate gets its own weight initialisation
*and* its own training paths — that is what a replicate means — but all are scored on
**identical** evaluation paths, so the figure isolates training variability rather than
mixing in evaluation noise.

```
CVaR-95 (loss)   mean 2.4498   sd 0.1218   range [2.2891, 2.6312]

     k   t_.975,k-1      MDE    resolves band-vs-delta (0.1343)?
     4        3.182   0.1938    no
     5        2.776   0.1512    NO     <- the count assumed throughout
     6        2.571   0.1278    yes    <- minimum
    10        2.262   0.0871    yes
```

**Five seeds cannot resolve an effect the size of band-versus-delta.** Six is the bare
minimum, and a learned-versus-band difference is plausibly *smaller* than that, so headline
cells take ten.

### Two regimes, previously conflated

`docs/06` §F.2 reported a paired sd of 0.0480 and an MDE of 0.0597. That figure is for
comparisons between **two deterministic policies**, where the only randomness is the
evaluation sample. It does not apply when a policy is trained:

```
evaluation noise (band vs delta)   sd 0.0480
training noise   (learned policy)  sd 0.1218      2.5x larger
```

Conflating them made every comparison involving a learned policy look 2.5× more precise
than it is. `docs/06` §F.2 now separates the regimes explicitly.

### What broke — in my own reporting

The first run printed `seeds needed to resolve it: 4` on the line directly below
`MDE at 5 seeds: 0.1512` against an effect of `0.1343`. Those contradict: if five seeds
cannot resolve it, four certainly cannot.

The cause was computing the required count with a normal critical value of 2.0 while
reporting the MDE with a `t` value. At the replicate counts a compute budget actually
permits, `t(4) = 2.776` against `2.0` understates the requirement by roughly 40%. Now uses
`scipy.stats.t`, and `test_noise_floor_seed_requirement_is_self_consistent` asserts the two
outputs cannot diverge again.

### Impact on the grid

1,200 runs → **2,400**. Affordable only because of the 16.9× compile speedup measured two
days ago: ~11 s per run makes 2,400 runs about 7 hours rather than 5 days. The performance
work and the statistical requirement turned out to be connected — the first is what makes
the second payable. `paper/STRUCTURE.md` §4 updated.

### Next

Step 7, rung 4. The learned policy reached MAD 0.074 against `Phi(d1)` at 8,000 gradient
steps, against an acceptance of 0.05. Needs a budget sweep, and the open question is
whether the wing error (0.088 against 0.028 at the money) closes with budget or is a real
extrapolation limit.

---

## 2026-09-26 — Research mode: five pitfalls, verified, and a workshop paper

Target set: the Agenthon 2026 workshop at NeurIPS (due 30 Sep, non-archival). Work split
across subagents (NeurIPS finance survey, prior art, impact and ideas, data engineering, a
paper writer); **every claim an agent made was re-checked against its source or re-measured
before it entered the repository.** Three agent claims failed that check (below).

### What happened

- **Seeding mechanism corrected** (8f9c60a): shifted streams and false significance, not
  bias. Traced to source: TF's `_make_1d_state` left-pads seeds precisely so a small seed is
  not "used as the counter while the key is always zero"; an int seed is pre-split to full
  length and never padded. `from_seed([k])` lands in the key. TF's guide already warns about
  overlapping streams from `from_seed`, so the claim is the deterministic int path and its
  measured consequence, never "undocumented".
- **Five new regenerable findings:** `seeding`, `precision_by_n`, `convergence`,
  `evaluation_design`, `friction_mechanism`; plus `experiments/market_calibration.py`.
- **Workshop paper** `paper/workshop/main.tex`, 14 pp., five pitfalls, every table sourced
  to a command and every cited test checked to exist.

### The measurements

```
noise floor, 30 replicates      sd 0.144 [0.115, 0.193]   (8 replicates had said 0.122)
power for band-vs-delta, k=5    0.36 vs fixed baseline, 0.26 learned vs learned
80% power                       12 seeds vs fixed (8-19 across the sd CI); 20 per arm
SE(CVaR-95) / SE(mean)          3.9-4.3x over 100 replicates (not ~5x)
friction, mean vs closed form   vol 0.3%, cost (Leland + open/close) 3.5%, all 12 contracts
tail amplification              vol 1.6-3.0x, cost 1.1-2.1x
cost dominates at 25bp          10/12 on the mean, 6/12 on CVaR-95
realised S&P vol >= 1.1 x VIX   8.8% of 21-day windows 1990-2026 (CI 6.3-11.4%)
one-tick half-spread            ES 0.16bp, SPY 0.065bp, median S&P 500 stock 1.6bp
rung-4 gate, same networks      1/6, 6/6, 6/6 under three unstated evaluation designs
```

### What broke — in my own reporting, again

1. **"MDE" was a 95% CI half-width** (~50% power). "Six seeds minimum, ten headline" becomes
   12 and 20. Caught by the impact agent; verified with the noncentral t and 200,000
   simulated t-tests; pinned.
2. **"CVaR is ~5x noisier"** rested on 8, then 32 replicates; at 32 the SE of the mean read
   0.72x theory (a 2.2-sigma fluctuation). 100 replicates: 0.98-1.04x theory and a 4x ratio.
3. **An SD ratio was labelled "variance reduction."** Renamed; now reported for the mean too.
4. **"Rung 4 not met (MAD 0.074)"** was one of three equally defensible designs. The gate,
   not the network, was underspecified; re-specified before re-running.
5. **"A common error" / "most of the literature"** (trade-to-centre): no audited
   implementation does it; our own design documents did. Wording corrected everywhere.
6. **"Single stock = 50bp"**: large caps are ~1.6bp; 50bp is small/micro-cap.

### Agent claims that failed verification

- The prior-art agent: "Leland reproduces our ratios contract by contract." On the CVaR
  ratios that is agreement for the wrong reason: the rebalancing-only formula omits opening
  and closing costs (13-75% of cost) and the tail amplification, which roughly cancel. On the
  mean, which Leland models, the closed forms hold once opening and closing are included.
- The paper writer's draft quoted "MAD 0.074 not met" and the stale seed counts; replaced.
- A machine summary had reversed two facts about He et al. (earlier today, docs/07).

### Next

8,000-step noise floor (convergence status of the floor); regenerate `paper/findings.json`
with `--all` and confirm bit-reproducibility against the individual runs; the author's
decisions on the submission (see HANDOFF).

### Addendum, 2026-09-27 — the noise floor at 8,000 steps has a failed run

Same 30 replicates, four times the budget (`python -m experiments.findings noise_floor_long`).
The typical spread more than halves (robust sd 0.160 -> 0.063), yet the sd barely moves
(0.144 -> 0.134): replicate #20 destabilised after step 3,000 (training loss 0.65 -> 1.4,
settling near 1.0) and ended at CVaR-95 2.833, 11 robust sds from the median. Without it the
sd is 0.059. Retraining #20 alone reproduced 2.8328 exactly.

**What broke — in our own tooling:** `TrainingResult.improved` ("last decile beats first")
passes this run, because the first decile holds the initialisation spike. The new
`_loss_history_diagnostics` compares the final window with the best window (+55% for #20,
~0 for converged runs) and is pinned by a test on a synthetic regressed history.

**And in the new diagnostic:** its first test drew a random normal sample and flagged a
legitimate point at robust z = 5.35 -- a MAD from 30 runs is itself noisy. So a robust-z flag
is a screen to be explained by the loss history, not a verdict; the test now uses fixed
normal quantiles, and the docstring says so.
