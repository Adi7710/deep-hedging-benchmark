# 07 — Literature audit

What published deep hedging papers **actually report**, recorded from a full reading of the
paper text. Started 2026-09-26.

## Why this file exists

On 2026-09-26 a machine summary of He, Sutter & Gonon (NeurIPS 2025) reported "no explicit
mention of seeds, error bars, or confidence intervals" and stated the paper "compares
against delta hedging". A full reading of the text showed **both statements were false**.
Had the workshop paper's introduction been written from the summary, it would have
accused a NeurIPS paper of two failings it does not have — the exact carelessness this
project criticises.

**Rule:** no claim about what another paper does or does not report enters any draft
unless it is recorded here, with page references, from the paper's own text.

---

## He, Sutter & Gonon (2025) — *Distributional Adversarial Attacks and Training in Deep Hedging*

NeurIPS 2025 · arXiv:2508.14757 · 33 pages including checklist · read in full 2026-09-26

| Item | What the paper reports | Where |
|:--|:--|:--|
| Research question | Robustness of deep hedging to distributional shift; adversarial training over a Wasserstein ball | abstract, §5 |
| Baseline | "We adopt the standard deep hedging methodology [Buehler et al.] as baseline" — i.e. **clean deep hedging**, not a classical rule | §5.2 |
| Classical hedging rules | **None.** "delta hedg": 0 occurrences. Whalley, Wilmott, Zakamouline: 0 occurrences | full text |
| Other comparator | Robust deep hedging of Lütkebohmert et al. (2022), "Clean training on ROBUST [11]" | §5, Table 3 |
| Replicates | "For clean training, we average the results over **three independent runs** with identical settings" | Appendix, p.31 |
| Error bars (checklist Q7) | **[Yes]** — "out-of-sample and out-of-distribution performance in Fig 1 containing **min-max ranges across training partitions**" | checklist, p.16 |
| Compute (checklist Q8) | [Yes] — Appendix D | checklist |
| Worlds | Black–Scholes and Heston; transaction costs in an appendix extension under Heston | §5, pp.29–30 |
| Risk measure | Entropic; CVaR also discussed | pp.3–4, 21–23 |
| Real data | AAPL, AMZN, BRK-B, GOOGL, MSFT via yfinance; tested price trajectory from 9 March 2020; rolling-window REAL dataset 7 Mar 2020 – 30 Sep 2021, normalised to start at 10 | §5, Tables 3 and 8, p.31 |
| Finding on parameter-interval training | "explicitly incorporating robust parameter intervals into the data generation process may be unnecessary—or even counterproductive" | appendix |

### What may fairly be said about it

- It answers a **robustness** question by comparing learned policies against learned
  policies. It does not claim, and is not designed to establish, an advantage over a
  classical cost-aware hedging rule. Our question is **complementary**, not a correction.
- It reports variability — min-max ranges, three runs — and answers the NeurIPS checklist
  honestly. It is **not** an example of missing error bars and must not be cited as one.
- A general point about replicate counts may be made using *our* measured noise floor, as
  guidance, without imputing insufficiency to any specific paper whose effect sizes and
  noise we have not measured.

### What may NOT be said

- That it lacks error bars. False.
- That it compares against delta hedging. False.
- That three runs is insufficient *for its results*. Unknown — depends on its effect sizes.

---

## Lütkebohmert, Schmidt & Sester (2022) — *Robust deep hedging*

*Quantitative Finance* 22(8):1465–1480 · **not yet read in full**

Known only through He et al.'s description: training on paths generated across parameter
uncertainty intervals estimated from historical data. Prior art for the parameter-robust
grid entry in `docs/03`. Must be read before any claim about it, or about the
conservative-vol desk heuristic, appears in a draft.

---

## Not yet audited

Buehler et al. (2019); Carbonneau (2020); Imaki et al. (2021, no-transaction band network);
Arzel & Lehdili (2026); the 2026 "What does deep hedging actually learn?" paper.

Until they are, **no claim about "the literature" in general** — e.g. "most papers compare
against delta hedging", "papers rarely report seed dispersion" — may appear in a draft.
Claims must be scoped to papers recorded here, or to our own measurements.
