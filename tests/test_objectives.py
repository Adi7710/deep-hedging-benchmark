"""Convex risk measures — the training objective.

These turn a distribution of terminal P&L into the single scalar that training minimises.
Like :mod:`dhbench.pnl` they are pure functions of an array, so they are checkable by hand
with no neural network present. That is deliberate: a wrong objective produces a network
that converges beautifully to the wrong policy, and nothing downstream detects it.

The motivating fact, pinned first: **maximising average P&L is not a risk measure and
would tell an agent not to hedge at all.** Hedging has negative expected profit — you pay
a spread on every trade — and its entire value is in reducing dispersion.
"""

from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

from dhbench.objectives.cvar import CVaRRisk
from dhbench.objectives.entropic import EntropicRisk
from dhbench.objectives.mean_variance import MeanVarianceRisk

# Factories, not instances: CVaR carries mutable state (w), and sharing one object
# across parametrised cases would let one test's assignment leak into another.
MEASURE_FACTORIES = [
    ("entropic_1.0", lambda: EntropicRisk(1.0)),
    ("meanvar_1.0", lambda: MeanVarianceRisk(1.0)),
    ("cvar_0.95", lambda: CVaRRisk(0.95)),
]


def _score(measure, pnl):
    """Evaluate a measure, putting CVaR at its optimal w first.

    CVaR in Rockafellar-Uryasev form is a *minimum* over w, so the defining properties of
    a risk measure -- rho(constant) = -c, cash invariance -- hold at that minimum and not
    at an arbitrary w. In training the optimiser finds w; in a unit test we place it,
    which is why the shared property tests route through this helper.
    """
    if isinstance(measure, CVaRRisk):
        measure.w.assign(float(np.quantile(-np.asarray(pnl, dtype=np.float64), measure.alpha)))
    return float(measure(pnl))


def _coinflip(n: int = 20_000, size: float = 100.0) -> tf.Tensor:
    """Half +size, half -size. Mean zero, maximally risky."""
    return tf.constant(
        np.r_[np.full(n // 2, size), np.full(n // 2, -size)], dtype=tf.float32
    )


# --------------------------------------------------------------------------------------
# The property that motivates the whole module
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("name,factory", MEASURE_FACTORIES, ids=[n for n, _ in MEASURE_FACTORIES])
def test_measures_separate_equal_mean_distributions(name, factory):
    """A certain zero and a coin flip on +/-100 have the same mean and different risk.

    The mean cannot tell them apart, which is precisely why it is unusable as a training
    objective: an agent scored on mean P&L would decline to hedge, since hedging costs
    money and its benefit is entirely in dispersion.
    """
    certain = tf.zeros(20_000)
    gamble = _coinflip()

    assert float(tf.reduce_mean(certain)) == pytest.approx(
        float(tf.reduce_mean(gamble)), abs=1e-3
    ), "precondition: the two distributions must have the same mean"

    assert _score(factory(), gamble) > _score(factory(), certain) + 1.0, (
        f"{name} scored the coin flip no worse than the certain outcome; "
        f"it is not measuring risk."
    )


@pytest.mark.parametrize("name,factory", MEASURE_FACTORIES, ids=[n for n, _ in MEASURE_FACTORIES])
def test_certain_outcome_scores_as_its_negative(name, factory):
    """rho(constant c) = -c. Higher P&L is lower risk; the sign convention is fixed here.

    For CVaR this holds at the optimal w -- see :func:`_score`.
    """
    for c in (-5.0, 0.0, 7.5):
        got = _score(factory(), tf.fill((5_000,), c))
        assert got == pytest.approx(-c, abs=1e-3), (
            f"{name}: rho({c}) = {got}, expected {-c}"
        )


# --------------------------------------------------------------------------------------
# Entropic
# --------------------------------------------------------------------------------------

def test_entropic_is_cash_invariant():
    """rho(X + c) = rho(X) - c.

    Adding a guaranteed amount to every outcome improves the score by exactly that
    amount. This is why the premium cannot change the optimal hedge — it translates the
    objective. A learned policy that appears indifferent to the premium is correct, not
    broken, and this test is the reason we can say so.
    """
    rng = np.random.default_rng(0)
    pnl = tf.constant(rng.normal(0.0, 5.0, 50_000), dtype=tf.float32)
    measure = EntropicRisk(1.0)

    base = float(measure(pnl))
    for c in (3.0, -2.5):
        shifted = float(measure(pnl + c))
        assert shifted == pytest.approx(base - c, abs=1e-3)


def test_entropic_survives_where_the_naive_form_overflows():
    """**The trap this module exists to avoid.**

    ``log(mean(exp(-lambda * X)))`` overflows float32 for perfectly ordinary P&L. The
    failure is not loud: you get inf, then nan gradients, and it presents as a divergent
    training run rather than an arithmetic bug. ``reduce_logsumexp`` subtracts the max
    before exponentiating, so the largest term is exp(0).
    """
    rng = np.random.default_rng(0)
    pnl = tf.constant(rng.normal(-30.0, 12.0, 50_000), dtype=tf.float32)

    naive = float(tf.math.log(tf.reduce_mean(tf.exp(-2.0 * pnl))) / 2.0)
    assert not np.isfinite(naive), (
        "precondition: this input must overflow the naive form, or the test proves nothing"
    )

    stable = float(EntropicRisk(2.0)(pnl))
    assert np.isfinite(stable)
    assert stable > 0.0


def test_entropic_approaches_negative_mean_as_risk_aversion_vanishes():
    """As lambda -> 0 the measure degenerates to -E[X], the risk-neutral objective."""
    rng = np.random.default_rng(1)
    pnl = tf.constant(rng.normal(2.0, 4.0, 100_000), dtype=tf.float32)
    expected = -float(tf.reduce_mean(pnl))

    assert float(EntropicRisk(1e-4)(pnl)) == pytest.approx(expected, abs=1e-2)


def test_entropic_increases_with_risk_aversion():
    """More risk-averse means a worse score for the same risky distribution."""
    pnl = _coinflip(20_000, 10.0)
    scores = [float(EntropicRisk(lam)(pnl)) for lam in (0.1, 0.5, 1.0, 2.0)]
    assert scores == sorted(scores), f"not monotone in lambda: {scores}"


def test_entropic_is_monotone():
    """If X dominates Y pathwise, X cannot be the riskier of the two."""
    rng = np.random.default_rng(2)
    y = tf.constant(rng.normal(0.0, 5.0, 20_000), dtype=tf.float32)
    x = y + 1.0  # strictly better on every path
    assert float(EntropicRisk(1.0)(x)) < float(EntropicRisk(1.0)(y))


def test_entropic_is_convex():
    """rho(aX + (1-a)Y) <= a*rho(X) + (1-a)*rho(Y).

    Convexity is what makes the training problem well posed and is the property the
    non-convexity literature (arXiv:2510.01874) shows gradient methods depend on.
    """
    rng = np.random.default_rng(3)
    x = tf.constant(rng.normal(0.0, 6.0, 40_000), dtype=tf.float32)
    y = tf.constant(rng.normal(1.0, 3.0, 40_000), dtype=tf.float32)
    measure = EntropicRisk(1.0)

    for a in (0.25, 0.5, 0.75):
        mixed = float(measure(a * x + (1.0 - a) * y))
        bound = a * float(measure(x)) + (1.0 - a) * float(measure(y))
        assert mixed <= bound + 1e-4


def test_entropic_rejects_non_positive_risk_aversion():
    """lambda <= 0 divides by zero or flips the sign; fail at construction."""
    with pytest.raises(ValueError):
        EntropicRisk(0.0)
    with pytest.raises(ValueError):
        EntropicRisk(-1.0)


# --------------------------------------------------------------------------------------
# Mean-variance
# --------------------------------------------------------------------------------------

def test_mean_variance_matches_its_definition_with_population_variance():
    """-E[X] + (lambda/2)*Var(X), with the POPULATION variance (ddof=0).

    Which variance is used is a protocol decision, not a numerical detail: a reviewer
    comparing against a paper that used the sample variance will notice, so it is pinned.
    """
    rng = np.random.default_rng(4)
    arr = rng.normal(1.5, 3.0, 10_000).astype(np.float32)
    expected = -arr.mean() + 0.5 * 2.0 * arr.var(ddof=0)
    assert float(MeanVarianceRisk(2.0)(tf.constant(arr))) == pytest.approx(
        expected, rel=1e-4
    )


def test_mean_variance_is_cash_invariant():
    rng = np.random.default_rng(5)
    pnl = tf.constant(rng.normal(0.0, 4.0, 20_000), dtype=tf.float32)
    measure = MeanVarianceRisk(1.5)
    assert float(measure(pnl + 3.0)) == pytest.approx(float(measure(pnl)) - 3.0, abs=1e-3)


def test_mean_variance_at_zero_risk_aversion_is_negative_mean():
    """The degenerate case: pure expected-return maximisation, which will not hedge."""
    rng = np.random.default_rng(6)
    pnl = tf.constant(rng.normal(2.0, 5.0, 20_000), dtype=tf.float32)
    assert float(MeanVarianceRisk(0.0)(pnl)) == pytest.approx(
        -float(tf.reduce_mean(pnl)), abs=1e-4
    )


def test_mean_variance_penalises_upside_too():
    """Documented defect, pinned so it is a known property rather than a surprise.

    Mean-variance is not monotone: adding a chance of a large *gain* increases variance and
    therefore worsens the score. Nobody would choose it on theoretical grounds; it is in the
    benchmark because much of the literature reports it, and because if the method ranking
    changes between mean-variance and CVaR that is itself a result (RQ3).
    """
    base = tf.zeros(10_000)
    upside = tf.constant(np.r_[np.zeros(9_500), np.full(500, 50.0)], dtype=tf.float32)
    measure = MeanVarianceRisk(1.0)
    assert float(measure(upside)) > float(measure(base)), (
        "mean-variance should penalise upside dispersion; if it does not, the variance "
        "term is missing"
    )


# --------------------------------------------------------------------------------------
# CVaR
# --------------------------------------------------------------------------------------

def test_cvar_at_the_optimal_w_equals_the_empirical_cvar():
    """Rockafellar-Uryasev: the minimiser is the VaR, and the minimum is the CVaR.

    This is the identity that licenses replacing a sort-and-average metric — whose
    gradient sees only the tail samples — with a smooth objective every path contributes to.
    """
    rng = np.random.default_rng(3)
    arr = rng.normal(0.0, 10.0, 200_000).astype(np.float32)
    alpha = 0.95

    loss = -arr
    var = float(np.quantile(loss, alpha))
    empirical_cvar = float(loss[loss >= var].mean())

    measure = CVaRRisk(alpha)
    measure.w.assign(var)
    assert float(measure(tf.constant(arr))) == pytest.approx(empirical_cvar, rel=1e-3)


def test_cvar_is_minimised_at_the_var_so_other_w_are_upper_bounds():
    """The formula is a minimum over w, so any other w over-states the risk.

    Practical consequence: a training run whose w has not converged reports a CVaR that is
    too high, which looks like a policy that is not learning.
    """
    rng = np.random.default_rng(3)
    arr = tf.constant(rng.normal(0.0, 10.0, 100_000), dtype=tf.float32)
    var = float(np.quantile(-arr.numpy(), 0.95))

    measure = CVaRRisk(0.95)
    measure.w.assign(var)
    at_optimum = float(measure(arr))

    for offset in (-5.0, -1.0, 1.0, 5.0):
        measure.w.assign(var + offset)
        assert float(measure(arr)) >= at_optimum - 1e-4


def test_cvar_auxiliary_variable_is_trainable_and_receives_gradient():
    """**The classic CVaR deep hedging bug.**

    ``w`` must be handed to the optimiser alongside the network weights. Omitting it lets
    training run, decrease a little, then flatline — a symptom that reads as a
    learning-rate problem rather than a formulation problem.
    """
    measure = CVaRRisk(0.95)
    assert measure.trainable_variables == [measure.w]
    assert measure.w.trainable

    rng = np.random.default_rng(7)
    pnl = tf.constant(rng.normal(0.0, 8.0, 20_000), dtype=tf.float32)
    with tf.GradientTape() as tape:
        loss = measure(pnl)
    grad = tape.gradient(loss, measure.w)
    assert grad is not None, "no gradient path to w -- CVaR cannot be optimised jointly"
    assert float(tf.abs(grad)) > 0.0


def test_cvar_is_cash_invariant_at_the_optimum():
    """rho(X + c) = rho(X) - c, once w tracks the shift.

    w is a VaR level and lives in the same units as the P&L, so shifting the distribution
    shifts the optimal w by the same amount. Tested at the optimum because the identity is
    a property of the minimisation, not of an arbitrary w.
    """
    rng = np.random.default_rng(8)
    arr = tf.constant(rng.normal(0.0, 7.0, 100_000), dtype=tf.float32)
    alpha, shift = 0.95, 4.0
    measure = CVaRRisk(alpha)

    var = float(np.quantile(-arr.numpy(), alpha))
    measure.w.assign(var)
    base = float(measure(arr))

    measure.w.assign(var - shift)  # optimal w for the shifted distribution
    shifted = float(measure(arr + shift))
    assert shifted == pytest.approx(base - shift, abs=1e-3)


def test_cvar_rejects_alpha_outside_the_unit_interval():
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            CVaRRisk(bad)
