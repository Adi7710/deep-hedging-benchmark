"""Benchmark metrics — every number in a results table comes from here.

Centralised for the same reason ``dhbench.pnl`` is: a metric computed slightly differently
in two places is a comparability bug, which is the class of problem this benchmark exists
to eliminate. These tests exist mainly to pin the *orientations*, since a sign convention
that drifts silently corrupts every table downstream.
"""

from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

from dhbench.evaluation.metrics import degradation_ratio, summarise
from dhbench.objectives.cvar import CVaRRisk, cvar_empirical
from dhbench.pnl import transaction_costs, turnover

METRIC_KEYS = {"cvar_95", "pnl_mean", "pnl_std", "turnover_mean", "total_cost_mean"}


# --------------------------------------------------------------------------------------
# cvar_empirical — the reporting estimator
# --------------------------------------------------------------------------------------

def test_empirical_cvar_matches_the_training_objective_at_its_optimum():
    """**The cross-check the two implementations exist for.**

    ``CVaRRisk`` (Rockafellar-Uryasev, differentiable, used for training) and
    ``cvar_empirical`` (sort-and-average, used for reporting) must agree at the optimal
    ``w``. A persistent gap in a real run means training stopped early -- which makes this
    a free convergence diagnostic, not just a unit test.
    """
    rng = np.random.default_rng(3)
    pnl = tf.constant(rng.normal(0.0, 10.0, 200_000), dtype=tf.float32)

    objective = CVaRRisk(0.95)
    objective.w.assign(float(np.quantile(-pnl.numpy(), 0.95)))

    assert float(cvar_empirical(pnl)) == pytest.approx(float(objective(pnl)), rel=1e-3)


def test_empirical_cvar_is_positive_for_a_risky_position():
    """Loss orientation: positive means expected loss in the tail, lower is better.

    Matches ``CVaRRisk``. The P&L-oriented view (higher is better) is obtained by negating
    -- see ``experiments.findings.cvar`` -- rather than by a second implementation.
    """
    rng = np.random.default_rng(0)
    assert float(cvar_empirical(tf.constant(rng.normal(0.0, 5.0, 50_000)))) > 0.0


def test_empirical_cvar_of_a_certain_outcome_is_its_negative():
    assert float(cvar_empirical(tf.fill((1_000,), 4.0))) == pytest.approx(-4.0, abs=1e-4)


def test_empirical_cvar_worsens_as_the_tail_gets_thinner():
    """A more extreme confidence level looks at a worse tail, so the number rises."""
    rng = np.random.default_rng(1)
    pnl = tf.constant(rng.normal(0.0, 8.0, 100_000), dtype=tf.float32)
    values = [float(cvar_empirical(pnl, a)) for a in (0.80, 0.90, 0.95, 0.99)]
    assert values == sorted(values), f"not monotone in alpha: {values}"


# --------------------------------------------------------------------------------------
# summarise
# --------------------------------------------------------------------------------------

def _fixture():
    spot = tf.constant([[100.0, 104.0, 102.0, 109.0], [100.0, 90.0, 95.0, 85.0]])
    delta = tf.constant([[2.0, 2.0, -1.0], [0.0, 1.0, 1.0]])
    pnl = tf.constant([-3.0, 7.0])
    return spot, delta, pnl


def test_summarise_reports_the_whole_metric_set():
    """Report the whole set, always.

    A method that improves CVaR by trading three times as much has improved nothing a desk
    would deploy, and reporting CVaR alone hides exactly that.
    """
    spot, delta, pnl = _fixture()
    assert set(summarise(pnl, delta, spot, 0.01)) == METRIC_KEYS


def test_summarise_returns_plain_floats_for_serialisation():
    """Results go straight to JSON; tensors would need a conversion step per call site."""
    spot, delta, pnl = _fixture()
    assert all(type(v) is float for v in summarise(pnl, delta, spot, 0.01).values())


def test_summarise_does_not_re_derive_turnover_or_cost():
    """Values must equal the single-source-of-truth functions exactly, not approximately."""
    spot, delta, pnl = _fixture()
    got = summarise(pnl, delta, spot, 0.01)

    assert got["turnover_mean"] == pytest.approx(
        float(tf.reduce_mean(turnover(delta))), rel=1e-6
    )
    assert got["total_cost_mean"] == pytest.approx(
        float(tf.reduce_mean(transaction_costs(spot, delta, 0.01))), rel=1e-6
    )
    assert got["pnl_mean"] == pytest.approx(2.0, abs=1e-6)


# --------------------------------------------------------------------------------------
# degradation_ratio — the §8 headline number
# --------------------------------------------------------------------------------------

def test_degradation_ratio_is_zero_when_nothing_degrades():
    assert degradation_ratio(2.0, 2.0) == pytest.approx(0.0)


def test_degradation_ratio_of_one_means_the_metric_doubled_in_badness():
    """Both inputs are lower-is-better, so 2 -> 4 is a doubling of loss."""
    assert degradation_ratio(2.0, 4.0) == pytest.approx(1.0)


def test_degradation_ratio_is_negative_when_the_shift_helps():
    """Occasionally a policy does better out of distribution. Report it, do not clamp it."""
    assert degradation_ratio(4.0, 2.0) == pytest.approx(-0.5)


def test_degradation_ratio_refuses_a_near_zero_denominator():
    """An in-distribution metric near zero makes the ratio explode.

    That is a property of the cell, not a degradation of infinity, so it fails loudly and
    the caller reports the raw pair instead.
    """
    with pytest.raises(ValueError, match="unbounded"):
        degradation_ratio(1e-15, 5.0)
