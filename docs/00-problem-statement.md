# 00 — Problem statement

The maths this repository implements, in **one consistent notation**. Papers in this field
use inconsistent symbols; half the reproduction bugs come from silently mismatched
conventions. When you read a new paper, translate its equations into this notation here
before writing any code.

---

## Setup

Discrete trading times $t_0 = 0 < t_1 < \dots < t_T$, uniformly spaced with
$\Delta t = T/n_{\text{steps}}$.

| Symbol | Meaning | Array shape |
|:--|:--|:--|
| $S_t$ | underlying price | `(n_paths, n_steps + 1)` |
| $\delta_t$ | hedge position held over $[t_i, t_{i+1})$ | `(n_paths, n_steps)` |
| $Z$ | contingent claim payoff at $T$ | `(n_paths,)` |
| $c$ | proportional transaction cost rate | scalar |
| $p_0$ | premium received at $t=0$ | scalar |
| $\rho$ | convex risk measure | scalar-valued functional |

**Convention:** $\delta_{-1} = 0$ (start flat), and the position is liquidated at $T$.
`dhbench/pnl.py` owns this convention — nothing else may re-derive it.

---

## The P&L functional

$$
\mathrm{PL}_T = p_0 - Z + \underbrace{\sum_{i=0}^{n-1} \delta_i \,(S_{i+1} - S_i)}_{\text{hedging gains}} - \underbrace{\sum_{i=0}^{n} c\, S_i \,|\delta_i - \delta_{i-1}|}_{\text{transaction costs}}
$$

Three things to be careful about, each a classic bug:

1. **The cost sum runs to $n$, not $n-1$** — liquidating the final position costs money.
   Forgetting this makes deep hedging look better than it is.
2. **Costs are charged on the *traded* amount** $|\delta_i - \delta_{i-1}|$, not the held
   amount.
3. **Sign convention:** we are short the claim ($-Z$) and receive the premium ($+p_0$).
   Buehler et al. write this with the opposite sign in places. Ours is fixed here.

---

## The objective

There are no labels. Training minimises a convex risk measure of the terminal P&L:

$$
\min_\theta \ \rho\big( \mathrm{PL}_T(\theta) \big), \qquad \delta_i = \pi_\theta(I_i)
$$

where $I_i$ is the information available at $t_i$ — at minimum $(t_i, S_i, \delta_{i-1})$.

This is the structural reason deep hedging is not ordinary supervised learning: the loss
is a functional of the **entire simulated path**, and gradients flow back through every
hedging decision. It's why `model.fit` doesn't apply and Stage 1 needs `GradientTape`.

### Risk measures implemented

**Entropic** (exponential utility), $\lambda > 0$ risk aversion:

$$
\rho_{\text{ent}}(X) = \frac{1}{\lambda} \log \mathbb{E}\left[ e^{-\lambda X} \right]
$$

Numerically delicate — implement via `reduce_logsumexp`, never `log(mean(exp(·)))`, which
overflows.

**CVaR** at level $\alpha$ (Rockafellar–Uryasev form, which is what makes it
differentiable):

$$
\rho_{\text{CVaR}}(X) = \min_{w \in \mathbb{R}} \ \left\{ w + \frac{1}{1-\alpha}\, \mathbb{E}\big[(-X - w)^+\big] \right\}
$$

The inner $w$ is optimised **jointly with the network weights** — it's an extra trainable
scalar, not a separate optimisation. This is the standard trick and the usual place people
go wrong.

**Mean-variance**, for comparability with older literature:

$$
\rho_{\text{mv}}(X) = -\mathbb{E}[X] + \tfrac{\lambda}{2}\,\mathrm{Var}(X)
$$

Not a coherent risk measure. Included because much of the field reports it.

### Indifference price

$$
p_0 = \rho\big(\mathrm{PL} \text{ with claim}\big) - \rho\big(\mathrm{PL} \text{ without claim}\big)
$$

Requires two training runs. Reduces to the Black–Scholes price in the zero-cost complete-market limit — another checkable gate.

---

## The result that gates everything

> Under GBM, zero transaction costs, and continuous rebalancing, the risk-minimising hedge
> is the Black–Scholes delta $\delta^{BS}_t = \Phi(d_1)$.

So a correctly implemented agent trained in that setting **must** reproduce $\Phi(d_1)$.
Plot the learned $\delta$ against $\Phi(d_1)$ across moneyness and time-to-maturity; they
should overlay.

**If they don't, stop.** Every downstream number is meaningless. This is rung 4 of the
correctness ladder and the single most important plot in the project — it belongs in §5 of
the paper.

---

## Baselines

The bar deep hedging must clear. Implemented in `dhbench/baselines/`.

**Black–Scholes delta** — $\delta^{BS} = \Phi(d_1)$, rebalanced every step. Optimal at zero
cost; badly suboptimal with costs because it trades constantly. Not a serious baseline
under frictions, but it is the zero-cost ground truth.

**Whalley–Wilmott band** — the asymptotically optimal no-transaction band under
proportional costs:

$$
H = \left( \frac{3}{2}\, \frac{c\, S\, \Gamma^2 e^{-r(T-t)}}{\lambda} \right)^{1/3}
$$

Trade back to $\delta^{BS}$ only when $|\delta - \delta^{BS}| > H$. **This is the baseline
that matters** — beating naive delta hedging under costs is trivial and proves nothing.

**Zakamouline** — a better empirical band; also handles fixed costs.

---

## Notation crosswalk

Fill in as you read. Prevents the silent-mismatch class of bug.

| Ours | Buehler et al. 2019 | Whalley–Wilmott 1997 | Carbonneau 2020 |
|:--|:--|:--|:--|
| $\delta_i$ | | | |
| $\rho$ | | | |
| $c$ | | | |
| $\lambda$ | | | |
