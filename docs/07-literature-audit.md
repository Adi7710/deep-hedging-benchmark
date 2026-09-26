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

## Chen, Cai, Qin & An (2025) — *OPHR: Mastering Volatility Trading with Multi-Agent Deep Reinforcement Learning*

NeurIPS 2025 main track · 31 pages · surveyed in full by a research subagent 2026-09-26;
**the three passages below re-verified from the proceedings PDF by the orchestrator**

| Item | What the paper reports | Where |
|:--|:--|:--|
| Baseline hedger | "The Delta threshold hedging strategy is a commonly used rule-based method that monitors the Delta exposure of the current portfolio and trades the hedging instruments to make the Delta exposure 0 when the Delta exposure exceeds the threshold." | p.6 |
| Its specification | "When the absolute ∆ of the position exceeds this threshold, \|∆t\| > ∆thres, a full Delta hedge is executed to neutralize the position's ∆ exposure." | App. B.2, p.23 |
| Applied to all baselines | "For consistency across all baselines, we apply a unified Delta-based hedging scheme as described in Section B.2, where the hedging threshold is set to ∆thres = 0.1." | App. C, p.26 |

### What may fairly be said about it

- It documents, in its own words, that threshold rules re-hedging fully to the centre
  (zero delta) are "commonly used". That is primary-source support for describing
  trade-to-centre as a common design choice in practice.
- Its setting — hedging the delta of a volatility-trading book with a fixed threshold —
  differs from ours, a Whalley–Wilmott band for a short call under proportional costs.
  Our measurement says nothing about OPHR's results.

### What may NOT be said

- That OPHR implemented a band "wrongly" or that its results are affected. Untested, and a
  re-hedge-to-zero rule is a legitimate practitioner choice in its context.
- Anything about other observations the survey agent recorded as internal.

---

## Prior-art pass, 2026-09-26 — quotes re-verified by the orchestrator

A prior-art subagent read ~50 sources in full (report and PDFs kept outside the repo).
Every passage below was then **re-located in the source text by the orchestrator** before
being allowed into a draft. Page numbers are PDF pages of the version named.

### Seeding and pseudo-replication

| Source | Verified passage | Where |
|:--|:--|:--|
| TensorFlow guide, *Random number generation* (last updated 2024-08-15) | "You will also run the risk that you may accidentally create two generators with the same seed or with seeds that lead to overlapping random-number streams." | section "Creating independent random-number streams", Note |
| TensorFlow 2.21 `stateful_random_ops._make_1d_state` | "Padding on the right would cause a small seed to be used as the 'counter' while the 'key' is always zero ... two RNGs with two different small seeds may generate overlapping outputs." An int seed is chopped to full length first, so the left-padding never runs. | source, verified 2026-09-26 by reading the installed file and by `from_seed([k])` → `[0,0,k]` (pinned in `tests/test_seeding.py`) |
| Salmon et al. (2011), SC'11 | "there is a danger that the generated streams are not statistically independent" | p.1 |
| Hurlbert (1984), *Ecol. Monogr.* 54(2) | "Pseudoreplication is defined as the use of inferential statistics to test for treatment effects with data from experiments where either treatments are not replicated (though samples may be) or replicates are not statistically independent." | abstract, p.187 |
| Bouthillier et al. (2021), MLSys | "this conditioning to arbitrary ξ induces a correlation between the trainings which in turns increases the variance of the estimator" (Eq. 7); and "model initialization generally is less than 50% of the variance of bootstrap" | p.6; p.4 |

**Consequence for drafts:** TensorFlow *documents the risk* in general terms. Our claim is
that the integer path realises it deterministically for every pair of small seeds, and what
that does to a replicate study. Never "undocumented", never "a TensorFlow bug".

### Band rebalancing

| Source | Verified passage | Where |
|:--|:--|:--|
| Whalley & Wilmott (1997), working-paper version (rev. 7 Mar 1997; journal *Math. Finance* 7(3)) | optimal band: "he must trade so as to just stay inside"; the rehedge-to-delta rule is their separate 'market movement' model, "rehedged to the delta value ... This models a strategy commonly used in practice." | WP p.6; WP p.18 |
| Whalley & Wilmott, "The best hedging strategy" (OFRC) | "If costs are entirely proportional to volume traded then shares are bought or sold to remain at the edge." | p.1 |
| Imaki et al. (arXiv:2103.01775v1) | optimal strategy is `clamp(δ, b_l, b_u)`; Remark 4.2 gives the Whalley–Wilmott width | p.10 |
| Sakuma (2026), arXiv:2607.25258v1 | "The trigger rule trades to the center rather than the nearest boundary, and this increases turnover and reserve cost." Table 12: +3.1% to +25.6% total reserve | App. E, p.15; main rule eq. (12), p.4 |
| Kumar (2026), arXiv:2608.29025v1 | band "rebalanced only to the nearest band edge, not the exact target"; abstract asks whether advantages "survive a genuinely fair comparison" | p.7; p.1 |

**Consequence:** the centre rule is a *named practitioner strategy*, not a typo. The error
we measure is implementing it and calling it Whalley–Wilmott. Every implementation checked
(Imaki, Kumar, Sakuma; pfhedge per the subagent) trades to the edge, so **no draft may imply
that a published study used the centre rule.** Sakuma (2026) already shows the execution
rule matters for reserve cost; our contribution is the paired CVaR-95 size for the baseline
a benchmark is judged against, and the non-estimable ratio.

### Evaluation statistics

| Source | Verified passage | Where |
|:--|:--|:--|
| Henderson et al. (2018), AAAI | "the variance between runs is enough to create statistically different distributions just from varying random seeds" | abstract / p.1 |
| Colas et al. (2018), arXiv:1806.08295v2 | "β = 0.51 for N = 5 ... To meet the requirement β = 0.2, N should be increased to N = 10" | p.13 |
| Agarwal et al. (2021), NeurIPS | "Statistical concerns cannot be satisfactorily addressed with few runs ... closer to 50–100 runs in Atari 100k" | p.5 |
| Ruf & Wang (2020), *J. Comput. Finance* | benchmark choice decides conclusions; random partitioning "introduces information leakage and underestimates the generalization error" | pp.3, 18 |

### Hedging theory — originals NOT read (paywalled); attribution is secondary

- **Leland (1985)**, *J. Finance* 40(5). The adjusted variance σ²(1 + Le),
  Le = √(2/π)·k/(σ√dt) with round-trip cost k, is taken from Whalley & Wilmott (1997 WP,
  p.18, eq. 22) and Fukasawa (2012, arXiv:1103.2013v2, p.14). Cite for the formula only.
- **El Karoui, Jeanblanc-Picqué & Shreve (1998)** and **Hobson (1998)**: overestimated
  volatility super-hedges a convex claim. Known here from Bosserhoff & Stadje (p.2) and
  Hobson's abstract. The one-line argument (P&L rate ½ΓS²(σ_h² − σ_r²) ≥ 0 when Γ ≥ 0) is
  given in the draft so the claim does not rest on an unread source.
- **Matsumoto et al. (2007)**: cited only for the defect class, as characterised by
  Salmon et al. (p.1) and Blackman & Vigna (arXiv:1805.01407v3, p.11).

### A subagent claim corrected by measurement

The prior-art report stated that Leland's formula "reproduces the repo's measured ratios
contract by contract". Checked (`python -m experiments.findings friction_mechanism`), that is
**agreement for the wrong reason**. Leland's *rebalancing-only* ratio (1.21 − 1)/Le tracks
the CVaR-95 ratios (−28% to −2%) because two things it omits roughly offset:

- opening and closing the hedge are 13–75% of the mean cost (our P&L liquidates at T);
- CVaR-95 amplifies the volatility effect 1.6–3.0× its mean, the cost effect only 1.1–2.1×.

On the **mean**, which Leland actually models, closed forms hold: the volatility effect
matches V(σ_h) − V(σ_r) within 0.3%, and the cost matches Leland plus the opening and
closing trades within 3.5%, in all 12 contracts. The mean ratio is then 1.5–5.3, not
3.3–7.3. And at 25bp costs dominate the mean in 10/12 contracts but CVaR-95 in 6/12: which
friction dominates depends on the risk measure. Drafts cite Leland for the mechanism and
the direction, not as a contract-by-contract prediction of CVaR ratios.

### Market-calibration sources

Recorded with page locators in `experiments/market_calibration.py` (`LITERATURE`), opened by
the data-engineering subagent. The headline (realised S&P 500 volatility ≥ 1.1 × VIX in 8.8%
of 21-day windows, 1990–2026; median ratio 0.73) was recomputed independently by the
orchestrator from Cboe's own VIX and SPX files and matched to the stated precision.

---

## Not yet audited

Buehler et al. (2019); Carbonneau (2020); Arzel & Lehdili (2026); Poddar (2026); the 2026
"What does deep hedging actually learn?" paper.

Until they are, **no claim about "the literature" in general** — e.g. "most papers compare
against delta hedging", "papers rarely report seed dispersion" — may appear in a draft.
Claims must be scoped to papers recorded here, or to our own measurements.
