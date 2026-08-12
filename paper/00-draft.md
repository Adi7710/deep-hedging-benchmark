# A Reproducible Benchmark for Deep Hedging: Do Learned Policies Beat Classical Transaction-Cost Bands?

**Aditya Bhatia**
Stevens Institute of Technology

**Draft v0.1 — 2026-08-12**
**Status: methods draft. No empirical results yet.** Stage 0 of 6; see §8.

---

> **Reader's note on status.** This document specifies the problem, the accounting
> convention, the baselines, and the verification protocol, and reports the implementation
> completed so far. It contains **no experimental results**, because none have been
> produced. Sections 6 and 7 describe what will be measured and how; Section 8 states
> precisely what is and is not implemented at the time of writing. Nothing here should be
> read as a finding.

---

## Abstract

Deep hedging (Buehler et al., 2019) recasts derivative hedging under market frictions as a
stochastic control problem solved by direct policy optimisation: a neural network emits a
hedge position at each rebalancing date, and its parameters are chosen to minimise a convex
risk measure of terminal profit and loss. The approach has generated a substantial and
rapidly growing literature. It has not generated a common evaluation protocol. Published
studies differ simultaneously in market simulator, transaction-cost model, risk measure,
rebalancing frequency, network capacity, training budget, and reported metric, so that
cross-study comparison of headline numbers is not meaningful. A 2023 review of the area
states the problem directly: "the lack of a standardized testing dataset or universal
benchmark in the RL hedging space makes it difficult to compare results across different
studies."

This work proposes a fixed benchmark protocol and evaluates published method families
within it. Two questions are asked. First, whether deep hedging policies outperform
*classical transaction-cost-aware* hedging — specifically the Whalley–Wilmott asymptotic
no-transaction band — rather than the naive delta-hedging strawman against which they are
usually compared. Second, how much of any measured advantage survives a regime shift
between the training and evaluation distributions. The contribution is methodological
rigour and reproducibility rather than a new hedging algorithm, and the design commits in
advance to reporting null results.

---

## 1. Introduction

The Black–Scholes–Merton framework establishes that under continuous, frictionless trading
a European claim is perfectly replicable and the replicating strategy holds $\Phi(d_1)$
units of the underlying. The result is foundational, and its assumptions are false. Two
failures matter for practice: trading occurs at discrete dates, and each trade incurs cost.
Taken together they are not a small perturbation. A strategy that rebalances continuously
incurs unbounded cumulative cost under any proportional cost rate, so the frictionless
optimum is not merely inaccurate under frictions — it is inadmissible.

The correct problem under frictions is a trade-off. Rebalancing more finely reduces the
variance of the replication error and increases cost paid; rebalancing less finely does the
reverse. Classical treatments solve special cases of this trade-off asymptotically. Whalley
and Wilmott (1997), working in the small-cost limit under exponential utility, show that
the optimal policy is not to track the delta but to maintain a *no-transaction band* around
it, trading only on exit from the band and only as far as the nearest boundary. Zakamouline
(2006) extends the analysis to fixed plus proportional costs and gives empirically superior
band widths.

Deep hedging replaces derivation with optimisation. Rather than solving a
Hamilton–Jacobi–Bellman equation for the optimal control, one parameterises the control by
a neural network, simulates a large batch of price paths, evaluates the resulting terminal
P&L distribution under a convex risk measure, and differentiates that scalar with respect to
the network parameters. The method requires only the ability to *simulate and evaluate*, not
the ability to *label*, and therefore extends immediately to settings where no closed form
exists: stochastic volatility, jumps, incomplete markets, path-dependent claims, and
non-trivial cost structures.

The approach has been productive. The 2025–26 period alone produced distributional
adversarial training (He et al., NeurIPS 2025), band-structured architectural priors (Arzel
and Lehdili, 2026), ambiguity-averse formulations with feature clustering, uncertainty
quantification for learned hedge ratios, an analysis of failure under non-convex costs with
an AlphaZero-style alternative, and a robust hedging valuation adjustment under liquidity
stress.

**The gap.** Each of these contributions is evaluated in a setting of its authors' choosing.
The simulator differs; the cost rate differs; the risk measure differs; the rebalancing
frequency differs; the reported metric differs. Controls that would make numbers comparable
— equal parameter count, equal training budget, multiple seeds with dispersion reported, a
stated compute budget — are typically absent. The consequence is that the field's headline
claims cannot be ranked against one another, and in many cases cannot be ranked against the
classical baselines either, because the comparison is drawn against delta hedging rather
than against a cost-aware band.

Beating delta hedging under transaction costs is close to trivial. Delta hedging rebalances
at every date regardless of trade size, so any policy that trades less will reduce cost, and
under a moderate cost rate that alone can dominate. A demonstration that a learned policy
beats delta hedging under costs is therefore weak evidence of learning. The informative
comparison is against Whalley–Wilmott.

**Contributions.**

1. A single frozen evaluation protocol — simulators, cost models, risk measures, metrics,
   seeds, and training-budget controls — published as configuration files (§6).
2. A verified P&L accounting layer with an explicit and tested treatment of terminal
   liquidation, isolated as the sole source of truth for the objective (§4, §5).
3. A staged correctness ladder that establishes the simulator, pricing reference, and
   accounting are mutually consistent *before* any learned result is interpreted (§7).
4. Evaluation of published method families within the protocol, with an advance commitment
   to reporting cells in which deep hedging ties or loses to the classical band.
5. Quantification of regime fragility across the method set rather than a single model.

---

## 2. Related work

**Foundational.** Buehler, Gonon, Teichmann and Wood (2019) formulate hedging under
frictions and convex risk measures as a machine learning problem and supply the objective
this work inherits. Whalley and Wilmott (1997) supply the asymptotic band that constitutes
the principal baseline. Zakamouline (2006) supplies a second, empirically stronger band that
also handles fixed costs. Föllmer and Schied provide the convex-risk-measure theory
underlying the objectives, in particular cash-invariance and its consequences (§4.5).

**Reference implementation.** Buehler's `hansbuehler/deephedging` is read as reference but
not taken as a dependency: it requires `tensorflow_probability` version-matched to
TensorFlow, and this project targets TensorFlow 2.21. It is used only as a cross-check
should a Stage 2 result disagree with the published one.

**Recurrent policies.** Carbonneau (2020) applies an LSTM agent to long-dated derivatives
and supplies the recurrent entry in the method grid.

**The comparison set.** He et al. (2025) show standard deep hedging policies are highly
sensitive to small perturbations of the input distribution and propose adversarial training
over a Wasserstein ball. Arzel and Lehdili (2026) impose no-transaction band structure as an
architectural prior rather than relying on it being learned. Further 2026 work addresses
ambiguity aversion, uncertainty quantification, non-convex cost structures, and hedging
valuation adjustment under liquidity stress.

**Interpretation and fragility.** *What Does Deep Hedging Actually Learn? Delta Corrections,
Regime Fragility, and Symbolic Distillation* (2026) reports that learned policies function
largely as corrections to the delta rather than as qualitatively novel strategies, and that
they degrade when market conditions move outside the training distribution. §7.2 of the
present work quantifies that degradation across the full method set, which is what
distinguishes a benchmark result from a replication.

**Framing.** The structural model for this paper is LOBFrame (*Deep limit order book
forecasting: a microstructural guide*, 2025), which pairs a reproducible framework with an
honest evaluation demonstrating that strong predictive accuracy need not imply economic
value.

---

## 3. Problem formulation

### 3.1 Market and claim

Fix a maturity $T$ and rebalancing dates $t_0 = 0 < t_1 < \dots < t_n = T$, uniformly spaced
with $\Delta t = T/n$. A single risky asset has price process $S$, observed on the grid. The
agent is **short** one European contingent claim with payoff $Z$ at $T$; the base case is a
call, $Z = (S_n - K)^+$. A premium $p_0$ is received at inception.

| Symbol | Meaning | Array shape |
|:--|:--|:--|
| $S_i$ | underlying price at $t_i$ | `(n_paths, n_steps + 1)` |
| $\delta_i$ | position held over $[t_i, t_{i+1})$ | `(n_paths, n_steps)` |
| $Z$ | claim payoff at $T$ | `(n_paths,)` |
| $c$ | proportional cost rate | scalar |
| $p_0$ | premium received at $t_0$ | scalar |
| $\rho$ | convex risk measure | functional |

Two boundary conventions are fixed once and enforced in code: $\delta_{-1} = 0$, the agent
starts flat, and $\delta_n = 0$, the position is liquidated at maturity.

The shape convention encodes a **predictability** requirement. The position $\delta_i$ is
chosen at $t_i$ using information available at $t_i$ and is held across
$[t_i, t_{i+1})$; consequently there are $n+1$ observed prices, $n$ increments, and $n$
decisions. The terminal price $S_n$ is used to *value* a position but never to *choose* one.
A policy indexed so that $\delta_i$ multiplies $(S_i - S_{i-1})$ is trading on an already
observed increment, which is a look-ahead. This failure mode does not raise a shape error;
it presents as a hedge that performs implausibly well.

### 3.2 The P&L functional

The terminal P&L is

$$
\mathrm{PL}_T \;=\; p_0 \;-\; Z \;+\; \underbrace{\sum_{i=0}^{n-1} \delta_i\,(S_{i+1} - S_i)}_{\text{hedging gains}} \;-\; \underbrace{\sum_{i=0}^{n} c\,S_i\,\lvert \delta_i - \delta_{i-1}\rvert}_{\text{transaction costs}}
\tag{1}
$$

This is a consequence of the self-financing condition rather than a definition, and the
derivation is worth stating because it determines how (1) must be extended later.

Let $X_i$ denote the cash balance at $t_i$ and $\delta_i$ the share holding. Immediately
before rebalancing at $t_{i+1}$, wealth is $\delta_i S_{i+1} + X_i(1 + r\Delta t)$.
Rebalancing exchanges stock for cash and does not itself change wealth, except for the cost
incurred:

$$
X_{i+1} \;=\; X_i(1 + r\Delta t) \;-\; (\delta_{i+1} - \delta_i)S_{i+1} \;-\; c\,S_{i+1}\lvert \delta_{i+1} - \delta_i\rvert .
\tag{2}
$$

That rebalancing is wealth-neutral up to cost *is* the self-financing condition: no capital
enters or leaves the strategy after inception. Setting $r = 0$ and telescoping (2) from
$i = 0$ to $n$, the $\delta_i S_i$ terms cancel pairwise and what remains is the discrete
stochastic integral $\sum_i \delta_i (S_{i+1} - S_i)$, the discrete analogue of
$\int_0^T \delta_t \, dS_t$. Subtracting the claim and adding the premium gives (1).

### 3.2.1 Numéraire

The telescoping above used $r = 0$; for $r \neq 0$ the cash account does not cancel, and its
balance depends on the entire trading history. Two resolutions are available: carry an
explicit cash-account state, or change numéraire. **This benchmark is specified in
discounted (time-0) units**, for the reason that (1) is form-invariant under that change and
therefore requires no additional state.

Write $\tilde S_i = e^{-r t_i} S_i$ and $\tilde Z = e^{-rT} Z$. Two facts combine. First,
the discounted wealth of a self-financing strategy is the discrete stochastic integral of
$\delta$ against the *discounted* price, so the gains term becomes
$\sum_i \delta_i (\tilde S_{i+1} - \tilde S_i)$. Second, a cost incurred at $t_i$ is
$c\,S_i\lvert\Delta\delta_i\rvert$ in time-$t_i$ money, and discounting it yields
$c\,\tilde S_i \lvert\Delta\delta_i\rvert$ — the cost term discounts by the same factor as
the price it is levied on. Hence

$$
\widetilde{\mathrm{PL}}_T = p_0 - \tilde Z + \sum_{i=0}^{n-1}\delta_i(\tilde S_{i+1} - \tilde S_i) - \sum_{i=0}^{n} c\,\tilde S_i \lvert \delta_i - \delta_{i-1}\rvert,
\tag{1'}
$$

identical in form to (1). The premium requires no adjustment: $p_0$ is received at $t_0$ and
is already time-0 money, as is the Black–Scholes price it is typically set to. That this
consistency holds is not incidental — it is why the change of numéraire is free.

Three consequences are stated rather than left implicit. **(a)** All reported P&L is in
time-0 money. **(b)** Collapsing the cash account into a single factor assumes one rate for
both borrowing and lending; funding spreads, collateral, or asymmetric borrow/lend rates
would break the collapse and require explicit cash flows (§10). **(c)** The band half-width
(4) is a statement about a *number of shares* and is computed in real units from real $S$,
$\Gamma$ and $r$; accounting units and band units are separate. Relatedly, the risk aversion
$\lambda$ is defined against a numéraire — Whalley–Wilmott's is over terminal wealth,
whereas $\rho$ here acts on discounted P&L, and the two differ by $e^{rT}$ in the exponent.
At $r = 0$ the distinction vanishes; at $r > 0$ it must be reconciled before any
agent-versus-band comparison, or the like-for-like requirement of §4.2 is violated through a
side door.

Operationally, `terminal_pnl` is the sole site of discounting: callers pass real prices and
a real payoff together with $r$ and $T$, and the conversion happens once, in the module that
owns the convention. `hedging_gains` and `transaction_costs` take no rate argument and are
numéraire-agnostic, which is exactly the content of (1'). Stage 0–2 configurations
nevertheless set $r = 0$, so that discounting is the identity while the accounting is still
being verified; a discount-factor error and a simulator error are otherwise
indistinguishable in rung 2.

### 3.3 Why the cost sum runs to $n$

The upper limit in the cost term of (1) is $n$, not $n-1$. With $\delta_{-1} = \delta_n = 0$
the sequence of traded quantities is

$$
\underbrace{\delta_0 - 0}_{\text{open}},\;\; \delta_1 - \delta_0,\;\; \dots,\;\; \delta_{n-1} - \delta_{n-2},\;\; \underbrace{0 - \delta_{n-1}}_{\text{liquidate}}
$$

which is $n+1$ trades against $n$ decisions. Both boundary trades are real: the position
must be established and it must be unwound.

Two arguments make the terminal term non-optional. The accounting argument is that
$\mathrm{PL}_T$ is a cash quantity, whereas a share still held at $T$ is a mark-to-market
value; omitting the liquidation charge asserts that the position can be unwound at mid.
The incentive argument is sharper and is the reason this is emphasised: the terminal cost
is the *only* term in (1) penalising a large position at maturity. Without it, a trained
policy can carry an arbitrarily large hedge into expiry at no charge, improving the reported
risk measure through a bookkeeping artefact. The resulting strategy is not merely
mismeasured but fictitious, and no internal consistency check detects it, because the
accounting is self-consistent — merely wrong. Under a 50 bp rate on a call whose delta tends
to unity on in-the-money paths, the omitted charge is of order $c\,S_n \approx 0.5$ on a
spot of 100, a material fraction of a typical premium.

The convention assumes **cash settlement**. Physical settlement would deliver the share
against the strike and require a different terminal term.

### 3.4 Sign convention

The agent is short the claim ($-Z$) and receives the premium ($+p_0$). Buehler et al. adopt
the opposite orientation in places; the convention is fixed here and enforced by test. Three
reasons motivate it. It is the sell-side problem, which is the problem the literature
addresses. It places the risk where a convex risk measure has purchase: a short call has
bounded upside and unbounded downside, whereas a long claim has payoff bounded below by
zero. And it orients the indifference price (§3.6) as an ask rather than a bid, which is the
side reported throughout the literature.

### 3.5 Objective

Training minimises a convex risk measure of terminal P&L,

$$
\min_\theta \; \rho\big(\mathrm{PL}_T(\theta)\big), \qquad \delta_i = \pi_\theta(I_i),
\tag{3}
$$

with $I_i$ the information set at $t_i$, at minimum $(t_i, S_i, \delta_{i-1})$.

Three measures are implemented, spanning current practice.

**Entropic** (exponential utility), risk aversion $\lambda > 0$:

$$
\rho_{\text{ent}}(X) = \tfrac{1}{\lambda} \log \mathbb{E}\big[e^{-\lambda X}\big].
$$

Implemented via `reduce_logsumexp`; the naive $\log(\text{mean}(\exp(\cdot)))$ overflows.

**CVaR** at level $\alpha$, in Rockafellar–Uryasev form, which is what renders it
differentiable:

$$
\rho_{\text{CVaR}}(X) = \min_{w \in \mathbb{R}} \Big\{ w + \tfrac{1}{1-\alpha}\,\mathbb{E}\big[(-X - w)^+\big] \Big\}.
$$

The auxiliary $w$ is a trainable scalar optimised jointly with the network parameters, not
by an inner loop.

**Mean-variance**, for comparability with older literature:

$$
\rho_{\text{mv}}(X) = -\mathbb{E}[X] + \tfrac{\lambda}{2}\mathrm{Var}(X),
$$

which is not coherent and is included only because much of the field reports it.

**Cash-invariance.** All three satisfy $\rho(X + m) = \rho(X) - m$. The premium $p_0$ is
therefore a translation of the objective and does not affect the optimal policy $\delta^\star$.
It matters for reporting and for indifference pricing, not for optimisation. This is worth
stating explicitly because the apparent non-effect of $p_0$ on a learned policy is
frequently mistaken for a bug.

### 3.6 Indifference price

$$
p_0 = \rho\big(\mathrm{PL} \text{ with claim}\big) - \rho\big(\mathrm{PL} \text{ without claim}\big),
$$

requiring two training runs. In the frictionless complete-market limit it reduces to the
Black–Scholes price, providing a further checkable gate.

### 3.7 Why this is not supervised learning

Under GBM with zero costs the optimal control is known in closed form, and it is reasonable
to ask why the policy is not simply regressed onto $\Phi(d_1)$. Four reasons, in increasing
order of consequence.

**(i) Labels exist only where the answer is already known.** $\Phi(d_1)$ is optimal under
GBM, zero cost, continuous rebalancing and market completeness. Under proportional costs the
optimum is a band known only asymptotically; under stochastic volatility or jumps the market
is incomplete and no closed form exists. A supervised policy can reproduce only what is
already available. Deep hedging requires the ability to *evaluate* a strategy — simulate,
accumulate P&L, score it — which is available wherever simulation is available. This
asymmetry between labelling and evaluation is the methodological content of the approach.

**(ii) The surrogate loss is not the objective.** Squared error on the control weights all
deviations equally. A delta error of fixed size is nearly costless when gamma is small and
maturity distant, and expensive at the money near expiry. Mean squared error is a pointwise
loss on the control; (3) is a functional of the induced P&L distribution. They do not share
a minimiser away from the idealised case.

**(iii) A pointwise label cannot represent the optimal policy under costs.** Once trading is
costly the current position enters the state: the optimum is a function of
$(t, S, \delta_{i-1})$, not $(t, S)$. The label $\Phi(d_1)$ contains no dependence on
$\delta_{i-1}$, so a policy supervised on it cannot express a no-transaction band even in
principle. The band is the object of interest in this literature.

**(iv) It destroys the verification protocol.** Rung 4 of the correctness ladder (§7)
requires that an agent trained at zero cost recovers $\Phi(d_1)$. That test has diagnostic
power precisely because the delta is never shown to the network: the policy arrives at
$\Phi(d_1)$ by numerical optimisation of a risk functional, along a route entirely
independent of the Black–Scholes derivation, and agreement is therefore joint evidence for
the simulator, the accounting, and the training loop simultaneously. Supervising on
$\Phi(d_1)$ reduces the test to confirming that a regression fits its own labels.

A related subtlety justifies the "$\approx$" in rung 4. $\Phi(d_1)$ is optimal in continuous
time; at finite $n$ the risk-minimising discrete position differs from it, and the
difference depends on the risk measure chosen. A correctly trained agent solves the discrete
problem actually posed, for which no closed form is available, and should therefore be
expected to approach but not equal $\Phi(d_1)$, converging as $n$ grows.

---

## 4. Baselines

### 4.1 Black–Scholes delta

$\delta^{BS} = \Phi(d_1)$, rebalanced at every date. Optimal at zero cost and the zero-cost
ground truth; deliberately poor under costs, since it trades at every date irrespective of
trade size. Reported, but not treated as a serious competitor under frictions.

### 4.2 Whalley–Wilmott band

The principal baseline. In the small-cost limit the optimal policy maintains a band of
half-width

$$
H = \left( \frac{3}{2}\, \frac{c\,S\,\Gamma^2 e^{-r(T-t)}}{\lambda} \right)^{1/3}
\tag{4}
$$

about $\delta^{BS}$, and trades **to the nearest boundary of the band**, not to
$\delta^{BS}$, upon exit. This distinction is material and is the most common
implementation error in band hedging: rebalancing to the centre discards most of the cost
saving the band exists to capture, weakening the baseline and biasing the headline
comparison in favour of the learned policy. Since that bias is precisely the failure mode
this benchmark exists to eliminate, the convention is stated here and enforced in the
implementation.

The exponent in (4) admits a short heuristic derivation that is useful as a diagnostic.
Writing the cost rate per unit time as decreasing in band width and the risk carried as
increasing in it,

$$
\text{total} \;\approx\; \frac{a\,c}{H} + b\,H^2, \qquad \frac{d}{dH}: \; -\frac{a c}{H^2} + 2bH = 0 \;\Longrightarrow\; H \sim c^{1/3},
$$

the cube root arising because the risk penalty grows quadratically in $H$ while the cost
saving grows only linearly. The resulting scalings,

$$
H \sim c^{1/3}, \qquad H \sim \Gamma^{2/3}, \qquad H \sim \lambda^{-1/3},
$$

are used in §7 as structural checks on learned policies: an agent whose band fails to
respond to the cost rate approximately as a cube root has not recovered the correct
structure, independent of whether its risk number is competitive.

### 4.3 Zakamouline

An empirically superior band that additionally accommodates fixed costs. Second baseline.

---

## 5. Implementation

TensorFlow 2.21 / Keras 3, with NumPy and SciPy. No dependency on
`hansbuehler/deephedging`. Simulators return `(n_paths, n_steps + 1)`; hedge positions are
`(n_paths, n_steps)`. Randomness is supplied by explicit `tf.random.Generator` instances
passed into simulators; global random state is not used anywhere that produces a reported
number, since bit-reproducibility from configuration plus seed is a claimed contribution.

### 5.1 The accounting layer

`dhbench/pnl.py` is the sole implementation of (1). No other module computes P&L. The
rationale is empirical: the dominant failure mode in reproductions of this literature is a
bookkeeping error concealed inside a training loop, where it presents as a convergence
problem. Isolating the arithmetic makes it testable independently of anything stochastic,
which is what gives rung 2 of the ladder its force.

Reference values (Black–Scholes price, delta, gamma) are implemented in NumPy rather than
TensorFlow. They are never part of a training graph, and keeping them outside TensorFlow
means the tests can validate TensorFlow-side code against a reference that shares no
machinery with it.

The module exposes four functions. Only the first is implemented at the time of writing.

```python
def hedging_gains(spot: tf.Tensor, delta: tf.Tensor) -> tf.Tensor:
    """Hedge gains before costs: ``sum_{i=0}^{n-1} delta_i (S_{i+1} - S_i)``.

    spot:  (n_paths, n_steps + 1)
    delta: (n_paths, n_steps)
    ->     (n_paths,)

    delta has one fewer column than spot on purpose: one decision per *gap*, not
    per price. Stay in TensorFlow -- a ``.numpy()`` here kills Stage 2 gradients.
    """
    moves = spot[:, 1:] - spot[:, :-1]      # (n_paths, n_steps): S_{i+1} - S_i
    return tf.reduce_sum(delta * moves, axis=-1)   # collapse time, keep paths
```

Two properties of this implementation are load-bearing. The reduction is over the **last**
axis, so that the result is one value per path; reducing over axis 0 yields one value per
timestep, which is dimensionally plausible, silently wrong, and undetectable on a
single-path test. And the computation remains entirely within TensorFlow: any conversion to
NumPy severs the gradient path from the objective back to the policy parameters, which at
Stage 2 manifests as a network that does not train rather than as an error.

The remaining three signatures, with the conventions they must satisfy:

```python
def transaction_costs(spot, delta, cost_rate) -> tf.Tensor:
    """Proportional cost: ``sum_{i=0}^{n} c S_i |delta_i - delta_{i-1}|``.
    -> (n_paths,), always >= 0.
    Sum runs to n, not n-1: unwinding at T is a real trade. Pad delta with a
    leading AND a trailing zero before differencing -> n_steps + 1 trades."""

def turnover(delta) -> tf.Tensor:
    """``sum_i |delta_i - delta_{i-1}|``, same padding. Prices do not appear.
    Reported to test whether an agent's advantage is merely higher turnover."""

def terminal_pnl(spot, delta, payoff, cost_rate=0.0, premium=0.0) -> tf.Tensor:
    """premium - payoff + hedging_gains - transaction_costs.  -> (n_paths,)"""
```

`transaction_costs` consumes all $n+1$ columns of `spot` — trades occur at $n+1$ dates —
whereas `hedging_gains` consumes $n$ increments. The asymmetry is a direct consequence of
§3.3 and is the sharpest available test that the terminal liquidation has not been dropped.

### 5.2 Worked example

One path, four dates, three decisions, $c = 0.01$:

```
spot  = [[100., 104., 102., 109.]]        (1, 4)
delta = [[  2.,   2.,   -1.]]             (1, 3)

hedging_gains
  later prices    [[104., 102., 109.]]
  earlier prices  [[100., 104., 102.]]
  increments      [[  4.,  -2.,   7.]]
  gain per gap    [[  8.,  -4.,  -7.]]
  total                        [-3.00]

transaction_costs
  padded delta    [[0., 2., 2., -1., 0.]]      (1, 5)
  traded          [[2., 0., -3., 1.]]          (1, 4)   <- four trades
  |traded|        [[2., 0.,  3., 1.]]
  c * S * |dq|    [[2.00, 0.00, 3.06, 1.09]]
  total                                [6.15]

turnover                                [6.00]
```

The second trade is zero: the position is unchanged from $\delta_0$ to $\delta_1$ and no
charge accrues. Cost is levied on the traded quantity, never the held quantity.

---

## 6. Benchmark protocol

Frozen and published as configuration files; every experiment is defined by a YAML config
plus a seed, with no hyperparameters in scripts.

**Worlds.** GBM; Heston; regime-switching; jump-diffusion. Fixed parameter sets, fixed
seeds.

**Cost models.** Zero; proportional at three levels; proportional plus fixed.

**Risk measures.** Entropic at two risk-aversion levels; CVaR$_{95}$; mean-variance.

**Metrics.**

| Metric | Rationale |
|:--|:--|
| CVaR$_{95}$ of terminal P&L | the tail is what hedging is for |
| Standard deviation of P&L | comparability with older literature |
| Turnover | tests whether the advantage is merely more trading |
| Total cost paid | separates gross skill from net outcome |
| Indifference price vs. BS price | economic interpretation |
| Degradation ratio under train/test mismatch | the regime-fragility result |

**Controls.** Equal parameter count across architectures; equal training budget; multiple
seeds with dispersion reported rather than point estimates; a stated compute budget. These
are routinely absent from cross-paper comparison and are a substantial part of why published
numbers do not reconcile.

---

## 7. Verification protocol

Results are not interpreted until the layers beneath them are verified. The ladder is
ordered so that each rung is meaningful only if its predecessors hold.

| Rung | Test | Gate | Status |
|:--|:--|:--|:--|
| 1 | Monte Carlo European call price matches closed form | simulator and pricing | red |
| 2 | Zero-cost fine-grid delta hedge: P&L std $\to 0$ as $n$ grows | P&L accounting | partial |
| 3 | Heston reproduces characteristic-function prices | stochastic vol simulator | red |
| 4 | Learned agent recovers $\Phi(d_1)$ at zero cost | **the headline gate** | not started |
| 5 | Learned band approximates Whalley–Wilmott at small cost | frictions | not started |
| 6 | Every experiment bit-reproducible from config plus seed | the benchmark claim | not started |

Rungs 1–3 require no neural network and are completed first.

Rung 2 is the load-bearing one for the present draft. A short call, delta-hedged at zero
cost with the premium set to the Black–Scholes price, must produce terminal P&L concentrated
near zero with residual dispersion scaling as $n^{-1/2}$. The test is strong because it is
*joint*: the simulator, the Black–Scholes reference, and the accounting must be mutually
consistent. Each could be individually wrong; all three being wrong in a manner that still
exhibits the correct convergence rate is improbable.

### 7.1 Null-result commitment

The full grid is worlds $\times$ costs $\times$ risk measures $\times$ agents, and all of it
is reported, including cells in which deep hedging ties or loses to Whalley–Wilmott. This is
stated in advance because selective reporting of favourable cells is the specific failure
this benchmark is intended to correct, and a benchmark that succumbs to it has no value.

### 7.2 Regime fragility

Train on one world, evaluate on another, and report the degradation ratio for every method.
Quantifying this across the method set rather than a single model is what distinguishes a
benchmark result from a replication, and this section is the fallback standalone
contribution should the benchmark framing be pre-empted.

---

## 8. Current status

Reported honestly and in full, at draft time.

**Test suite.** 42 collected: **18 passing, 16 failing, 8 skipped.** Failures are
`NotImplementedError` from unimplemented components and are expected at this stage; the
scaffold is written test-first, so red tests are the specification rather than a defect.

**Passing.** 15 Black–Scholes reference tests (price, delta, gamma, put–call parity,
boundary behaviour, and the $\tau \to 0$ limit), plus 3 tests of `hedging_gains` covering a
unit position over a single step, a flat position across arbitrary price moves, and the sign
of a short position into a rising market.

**Implemented.** `dhbench/baselines/bs_delta.py` in full (reference material, written
complete by design); `hedging_gains` in `dhbench/pnl.py`.

**Not implemented.** `transaction_costs`, `turnover`, `terminal_pnl`; the GBM simulator; all
stochastic-volatility, regime-switching and jump worlds; all three risk measures; both band
baselines; every agent; the evaluation and stress-testing layer.

**Therefore.** No rung of the correctness ladder is green. No experiment has been run. No
number in this document is an empirical result. Sections 6 and 7 are specifications of
intent.

---

## 9. Open specification items

Recorded rather than deferred silently, since each would produce a silent inconsistency if
left unresolved.

**9.1 Interest rates in the P&L functional.** Equation (1) is derived under $r = 0$, while
the Whalley–Wilmott band (4) carries $e^{-r(T-t)}$ and so reintroduces $r$ at the final
Stage 0 component. **Resolved 2026-08-12:** the benchmark is specified in discounted
(time-0) units, under which the functional is form-invariant and no cash-account state is
required; see §3.2.1. `terminal_pnl` gained `rate` and `maturity` arguments and is the sole
site of discounting, with `discount_factors` exposed so that evaluation code shares one time
grid rather than re-deriving it. Two items remain live and are tracked here rather than
closed: the single-rate assumption (§3.2.1(b), §10), and reconciliation of $\lambda$ across
numéraires before any agent-versus-band comparison at $r > 0$ (§3.2.1(c)).

**9.2 Band rebalancing target.** `docs/00-problem-statement.md` and `docs/01-papers.md`
describe trading back to $\delta^{BS}$ on band exit; `baselines/whalley_wilmott.py`
specifies trading to the nearest band boundary. The latter is correct — the optimal policy
under proportional costs is a singular control that executes the minimal trade returning the
state to the no-transaction region. Recorded because the error weakens the baseline and
would bias the headline comparison in the favourable direction.
**Resolved 2026-08-12:** both documents corrected to the band-edge rule.

**9.3 Cost-sum upper limit in the paper plan.** `PAPER.md` §3 wrote the cost sum with upper
limit $T-1$, omitting terminal liquidation, inconsistent with (1) and with the
implementation. This is precisely the error §3.3 identifies as material.
**Resolved 2026-08-12:** `PAPER.md` corrected to run to $n$, with the boundary conventions
stated inline.

**9.4 Settlement convention.** Cash settlement is assumed throughout (§3.3). Physical
settlement would alter the terminal cost term. To be stated explicitly in the final text.

---

## 10. Limitations

**Simulated markets only** for the present scope. Every conclusion is conditional on the
simulator being an adequate proxy, and the regime-shift experiments are internal to the
simulator family rather than a test against realised market data.

**Compute budget** constrains hyperparameter search. A better-tuned instance of any method
in the grid may exist; the budget is stated so that the constraint is legible rather than
hidden.

**Seed variance** in deep hedging is large relative to the effect sizes under study.
Dispersion across seeds is reported rather than a single point estimate, and comparisons
that do not survive seed variation are reported as inconclusive rather than as wins.

**Re-implementation risk.** A reproduced method may underperform its published version
through implementation error rather than method weakness. Mitigation: each re-implementation
is validated against its source paper's headline number before entry into the grid, and any
method that cannot be so validated is reported as unvalidated rather than as a loss.

---

## References

Arzel, P. and Lehdili, N. (2026). Bridging Stochastic Control and Deep Hedging: Structural
Priors for No-Transaction Band Networks. arXiv:2603.29994.

Buehler, H., Gonon, L., Teichmann, J. and Wood, B. (2019). Deep Hedging. *Quantitative
Finance* 19(8), 1271–1291. arXiv:1802.03042.

Carbonneau, A. (2020). Deep hedging of long-term financial derivatives. *Insurance:
Mathematics and Economics* 98, 327–340. arXiv:2007.15128.

Föllmer, H. and Schied, A. *Stochastic Finance: An Introduction in Discrete Time*, Ch. 4.

He, X. et al. (2025). Distributional Adversarial Attacks and Training in Deep Hedging.
NeurIPS 2025. arXiv:2508.14757.

Jones, D., Horvath, B., Reisinger, C., Wood, B. et al. Ambiguity-Averse Deep Hedging with
Feature Clustering. SSRN 5390563.

Prata, M. et al. (2025). Deep limit order book forecasting: a microstructural guide.
*Quantitative Finance*.

Pickard, R. and Lawryshyn, Y. (2023). Deep Reinforcement Learning for Dynamic Stock Option
Hedging: A Review. *Mathematics* 11(24), 4943.

Sakuma, T. (2026). Robust Hedging Valuation Adjustment for Deep Hedging Policies under
Market Frictions. arXiv:2607.25258.

Whalley, A. E. and Wilmott, P. (1997). An asymptotic analysis of an optimal hedging model
for option pricing with transaction costs. *Mathematical Finance* 7(3), 307–324.

Zakamouline, V. (2006). European option pricing and hedging with both fixed and
proportional transaction costs. *Journal of Economic Dynamics and Control* 30(1), 1–25.

*What Does Deep Hedging Actually Learn? Delta Corrections, Regime Fragility, and Symbolic
Distillation* (2026). arXiv:2605.21696.

*Deep Hedging Under Non-Convexity: Limitations and a Case for AlphaZero* (2025).
arXiv:2510.01874.

*Uncertainty-Aware Deep Hedging* (2026). arXiv:2603.10137.
