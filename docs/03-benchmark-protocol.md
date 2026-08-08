# 03 — The benchmark protocol

**Status: NOT YET FROZEN.** This document is written at Stage 4, once Stages 0–3 have
shown which choices are defensible. Drafting it now would be guessing.

Everything below is a placeholder specifying *what must be decided*, not what has been.

---

## Why this document is the contribution

The paper's claim is that published deep hedging results aren't comparable. The remedy is
a protocol that is:

- **Complete** — every degree of freedom fixed, nothing left to the implementer
- **Published** — as YAML in [`../configs/`](../configs/), not as prose
- **Bit-reproducible** — config + seed → identical numbers, on any machine

Once frozen, **changing it invalidates every result already collected.** That's the point;
it's also why it doesn't get written until the foundations are verified.

---

## Axes to freeze

### Worlds

| World | Parameters to pin | Source of values |
|:--|:--|:--|
| GBM | $S_0, \mu, \sigma, T, n_{\text{steps}}$ | conventional; state them |
| Heston | $v_0, \kappa, \theta, \xi, \rho$ | calibrate once to a real SPX surface, then freeze |
| Regime-switching | per-regime $(\mu,\sigma)$, transition matrix | calm/stressed, from literature |
| Jump-diffusion | $\lambda_{\text{jump}}, \mu_J, \sigma_J$ | Merton, from literature |

Also fix: number of paths for training, validation, and test — and **use disjoint seeds
for each**. Evaluating on training paths is an easy and fatal mistake.

### Cost models

Zero · proportional at three levels (something like 0.1%, 0.5%, 1%) · proportional + fixed.

Justify the levels against real bid-ask spreads rather than picking round numbers.

### Risk measures

Entropic at two risk-aversion levels · CVaR₉₅ · mean-variance.

**Sub-question 5 asks whether the method ranking changes across these.** If it does, that
alone shows cross-paper comparison is unsound — a result worth having regardless of how
the rest lands.

### Agents

Feedforward (Buehler) · recurrent (Carbonneau) · band-structured (Arzel & Lehdili) ·
adversarially robust (He et al.). Baselines: BS delta, Whalley–Wilmott, Zakamouline.

---

## Metrics

| Metric | Why it's in |
|:--|:--|
| CVaR₉₅ of terminal P&L | the tail is what hedging is *for* |
| P&L standard deviation | comparability with older literature |
| Mean P&L | detects a policy that's just taking directional risk |
| Turnover | is the edge just trading more? |
| Total cost paid | separates gross skill from net |
| Indifference price vs. BS price | economic interpretation |
| **Degradation ratio** under train/test regime mismatch | §8, the fragility result |

---

## Controls (what makes this a benchmark rather than a bake-off)

Most published comparisons control for none of these. Fixing them is a large part of why
the numbers in the literature don't line up.

- [ ] **Equal parameter count** across agents — or report it and treat capacity as an axis
- [ ] **Equal training budget** — same epochs, same path count, same optimiser settings
- [ ] **Multiple seeds** — at least 5. Report mean ± std, never a single run. Seed variance
      in deep hedging is large enough to swamp real differences
- [ ] **Disjoint train / validation / test seeds**
- [ ] **Stated compute budget** — hyperparameter search is bounded, and the bound is a
      stated limitation, not a hidden one
- [ ] **Identical evaluation paths** across all methods within a cell

---

## Reproducibility contract

```bash
python -m experiments.run --config configs/<name>.yaml --seed 0
```

Must produce bit-identical results on re-run. Enforced by rung 6 of the correctness ladder
in [`../tests/`](../tests/).

Every published number traces to exactly one config file plus one seed. No exceptions, no
hand-tuned one-off runs in the tables.

---

## Open decisions

Resolve before freezing. Record the reasoning, not just the choice — reviewers will ask.

- Which claim? European call only, or add digital / barrier for a convexity contrast?
- Hedge instrument: underlying only, or underlying + a liquid vanilla option?
- Rebalancing frequency: fixed, or an axis in its own right?
- State features: minimal $(t, S, \delta_{t-1})$, or richer (realised vol, implied vol,
  time-to-maturity)? The 2024 IV-surface-feedback paper argues richer helps — is that a
  fair comparison or an unfair advantage?
- How is the indifference price computed for methods that don't naturally produce one?
