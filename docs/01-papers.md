# 01 — The paper set

Every paper this project builds on, replicates, or benchmarks against. Read in the order
of the first section.

**Reading discipline:** before implementing anything from a paper, write its core equation
into [00-problem-statement.md](00-problem-statement.md) in our notation. Papers use
inconsistent symbols and half the reproduction bugs in this field come from silently
mismatched conventions.

---

## Tier 1 — Read before writing any code

### Buehler, Gonon, Teichmann, Wood (2019), *Deep Hedging*
*Quantitative Finance* 19(8). [arXiv:1802.03042](https://arxiv.org/abs/1802.03042)

The paper being benchmarked. Formulates hedging under frictions and convex risk measures
as a machine learning problem. **Read §1–3 properly** — that's the problem setup we
inherit wholesale. §4 onward is their specific network and experiments.

Key things to extract: the P&L functional, the convex risk measure formulation, and the
indifference price definition.

### Whalley & Wilmott (1997), *An asymptotic analysis of an optimal hedging model for option pricing with transaction costs*
*Mathematical Finance* 7(3).

**The baseline that actually matters.** Derives the asymptotic no-transaction band under
proportional costs. Beating naive delta hedging under costs is trivial and proves nothing;
beating Whalley–Wilmott is the real bar, and most papers that skip it are overstating
their result.

Band half-width, to implement in `baselines/whalley_wilmott.py`:

$$
H = \left( \frac{3}{2} \frac{ c\, S\, \Gamma^2 e^{-r(T-t)} }{ \lambda } \right)^{1/3}
$$

with $c$ the proportional cost rate, $\Gamma$ the Black–Scholes gamma, and $\lambda$ risk
aversion. Hedge to the delta only when $|\delta_t - \delta^{BS}_t| > H$.

### Zakamouline (2006), *European option pricing and hedging with both fixed and proportional transaction costs*
*Journal of Economic Dynamics and Control* 30(1).

A better empirical band than Whalley–Wilmott, and handles fixed costs. Second baseline.

### Föllmer & Schied, *Stochastic Finance* — Ch. 4
Convex and coherent risk measures. The theory behind `objectives/`. You need enough to
know *why* entropic utility and CVaR are the right objectives and what "convex" buys you —
which turns out to matter, see the non-convexity paper below.

---

## Tier 2 — Reference implementation

### `hansbuehler/deephedging`
[github.com/hansbuehler/deephedging](https://github.com/hansbuehler/deephedging)

Buehler's own TensorFlow implementation. Components: `SimpleWorld_Spot_ATM` (a
Black–Scholes and a toy stochastic-vol world), agents (feedforward / recurrent /
iterative), and `objectives.py` (entropic utilities, CVaR).

**Read it; do not depend on it.** It requires `tensorflow_probability` version-matched to
TensorFlow and we are on TF 2.21 — a known setup trap. It also states it "is not optimized
for speed" and is educational rather than production code. Writing our own is the point.

Useful as a cross-check when our Stage 2 result disagrees with the paper.

### Carbonneau (2020), *Deep hedging of long-term financial derivatives*
*Insurance: Mathematics and Economics* 98. [arXiv:2007.15128](https://arxiv.org/pdf/2007.15128)

LSTM agent. The recurrent variant in our grid. Has an accompanying
[notebook](https://github.com/alexandrecarbonneau/Deep-Hedging-of-Long-Term-Financial-Derivatives).

---

## Tier 3 — The 2025–26 wave (the comparison set)

These are what §7 of the paper benchmarks against each other. Each introduced a method;
none used a common protocol.

| Paper | Contribution | Grid entry |
|:--|:--|:--|
| He et al., *Distributional Adversarial Attacks and Training in Deep Hedging* — NeurIPS 2025, [arXiv:2508.14757](https://arxiv.org/abs/2508.14757) | standard deep hedging is highly vulnerable to small input-distribution perturbations; adversarial training over a Wasserstein ball fixes much of it | `agents/robust.py` |
| Arzel & Lehdili (2026), *Bridging Stochastic Control and Deep Hedging: Structural Priors for No-Transaction Band Networks* — [arXiv:2603.29994](https://arxiv.org/pdf/2603.29994) | imposes band structure as an architectural prior instead of hoping it's learned | `agents/band.py` |
| Jones, Horvath, Reisinger, Wood et al., *Ambiguity-Averse Deep Hedging with Feature Clustering* — [SSRN 5390563](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5390563) | control over *which* distributional shifts to be robust to | optional |
| *Uncertainty-Aware Deep Hedging* — [arXiv:2603.10137](https://arxiv.org/html/2603.10137v1) | deep hedging gives hedge ratios with no confidence measure; a deployment barrier | optional |
| *Deep Hedging Under Non-Convexity: Limitations and a Case for AlphaZero* — [arXiv:2510.01874](https://arxiv.org/html/2510.01874) | gradient-based deep hedging depends on convexity; fails on non-convex costs / capital constraints. MCTS alternative | context for §9 |
| Sakuma (2026), *Robust Hedging Valuation Adjustment for Deep Hedging Policies under Market Frictions* — [arXiv:2607.25258](https://arxiv.org/abs/2607.25258) | reserve cost and residual risk under liquidity stress | context |

### The one that shapes §8

**What Does Deep Hedging Actually Learn? Delta Corrections, Regime Fragility, and Symbolic
Distillation** — [arXiv:2605.21696](https://arxiv.org/pdf/2605.21696)

Findings: learned policies are mostly delta-*corrections* rather than novel strategies;
they exhibit **regime fragility**, failing when conditions shift outside the training
distribution; and symbolic distillation can extract human-readable rules from them.

§8 of our paper quantifies that fragility **across the whole method set** rather than one
model — which is what turns a replication into a benchmark result. This is also the
fallback standalone paper if the benchmark angle gets scooped.

---

## Tier 4 — Motivating the gap

### *Deep Reinforcement Learning for Dynamic Stock Option Hedging: A Review*
*Mathematics* 11(24), 2023. [MDPI](https://www.mdpi.com/2227-7390/11/24/4943)

The citation for §1. States plainly:

> the lack of a standardized testing dataset or universal benchmark in the RL hedging
> space makes it difficult to compare results across different studies

Also documents the variation this project standardises: no consensus on state-space
variables, mean-variance objectives dominant but CVaR and episodic return also used, GBM
the primary generator with SABR and Heston as alternatives.

### LOBFrame / *Deep limit order book forecasting: a microstructural guide*
*Quantitative Finance*, 2025. [PMC12315853](https://pmc.ncbi.nlm.nih.gov/articles/PMC12315853/)

Different domain, same shape of contribution — a framework plus an honest evaluation
showing high forecasting accuracy doesn't imply economic value. **The structural model for
this paper.** Worth reading for how they framed the contribution.

---

## Search log

Re-run before Stage 4 and before Stage 6. Record here.

| Date | Query | New papers found |
|:--|:--|:--|
| 2026-08-08 | deep hedging robustness / benchmark / open problems | initial set above |
