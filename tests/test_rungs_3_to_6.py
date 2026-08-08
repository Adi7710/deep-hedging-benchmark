"""Rungs 3-6 — specifications for tests that cannot be written yet.

These rungs depend on components that don't exist. Rather than leave them undocumented,
each is recorded as a skipped test carrying the acceptance criterion, so the ladder is
visible in ``pytest -v`` output from day one and nothing quietly goes unchecked.

**Replace each skip with a real test when you implement the component.** A skipped rung is
a rung that isn't gating anything.
"""

from __future__ import annotations

import pytest

# ======================================================================================
# Rung 3 — Heston simulator vs. characteristic-function prices
# ======================================================================================


@pytest.mark.skip(reason="Rung 3: simulate_heston and heston_price_cf not implemented")
def test_heston_mc_matches_characteristic_function():
    """Monte Carlo Heston price ≈ semi-analytic CF price.

    Acceptance: for at least three strikes (ITM, ATM, OTM) and two maturities, the MC
    price is within 3 standard errors of :func:`heston_price_cf`.

    Use a parameter set that **violates the Feller condition** (``2*kappa*theta < xi^2``)
    in at least one case — that is the regime where a naive variance scheme goes negative,
    and it is the whole reason full truncation is specified.

    Watch for: the Little Heston Trap. If longer maturities disagree while short ones pass,
    it is the complex-logarithm branch, not the simulator.
    """


@pytest.mark.skip(reason="Rung 3: simulate_heston not implemented")
def test_heston_variance_stays_non_negative():
    """Under full truncation, the variance used under any square root is ``>= 0``.

    Acceptance: no ``nan`` in the spot paths, and ``max(v, 0)`` used consistently. Test
    with a deliberately Feller-violating parameter set — with a benign one this passes
    even when the scheme is wrong.
    """


# ======================================================================================
# Rung 4 — the headline gate
# ======================================================================================


@pytest.mark.skip(reason="Rung 4: FeedforwardAgent and the training loop not implemented")
def test_learned_hedge_recovers_black_scholes_delta():
    """**The most important test in the project.**

    Train :class:`~dhbench.agents.feedforward.FeedforwardAgent` under GBM with zero
    transaction costs (``configs/gbm_zerocost_entropic.yaml``). The learned policy must
    reproduce ``Phi(d1)``.

    Acceptance: over a moneyness grid ``S/K`` in ``[0.8, 1.2]`` and times-to-maturity in
    ``[0.1, 1.0]``, mean absolute deviation from :func:`bs_delta` is below 0.05, with no
    single point off by more than 0.15.

    Also produce the overlay plot — it belongs in §5 of the paper and is far more
    convincing to a reader than the numeric threshold.

    **If this fails, stop.** Every downstream number is meaningless until it passes. Check,
    in order: rungs 1-2 still green; the roll-forward stays inside the ``GradientTape``;
    ``delta_prev`` is not detached; inputs are normalised; the entropic loss uses
    ``reduce_logsumexp``.
    """


@pytest.mark.skip(reason="Rung 4: training loop not implemented")
def test_learned_indifference_price_approximates_bs_price():
    """The indifference price recovers the Black-Scholes price in the frictionless limit.

    Acceptance: within 5% of 7.9656 for the reference config. Requires two training runs
    (with and without the claim), so it is slow — mark it ``@pytest.mark.slow``.
    """


# ======================================================================================
# Rung 5 — learned band vs. Whalley-Wilmott
# ======================================================================================


@pytest.mark.skip(reason="Rung 5: whalley_wilmott and trained agents not implemented")
def test_learned_band_approximates_whalley_wilmott():
    """With small proportional costs, the learned no-transaction band matches theory.

    Acceptance: the learned half-width from
    :meth:`~dhbench.agents.band.BandAgent.learned_band` correlates above 0.8 with the
    analytic ``H`` across the moneyness/time grid, and — more importantly — has the same
    *shape*: widest where gamma is highest.

    Shape agreement matters more than level agreement. Whalley-Wilmott is asymptotic in
    small ``c``, so at realistic cost levels an exact level match would actually be
    suspicious.
    """


@pytest.mark.skip(reason="Rung 5: band_hedge_positions not implemented")
def test_band_hedging_beats_naive_delta_under_costs():
    """Sanity check on the baseline itself, before it judges anything.

    Acceptance: under proportional costs, Whalley-Wilmott achieves strictly better CVaR-95
    **and** strictly lower turnover than every-step delta hedging.

    If it does not, the band implementation is wrong — most likely trading to the delta
    instead of to the band edge. A broken baseline would make deep hedging look better than
    it is, which is exactly the failure this benchmark exists to prevent.
    """


# ======================================================================================
# Rung 6 — reproducibility contract
# ======================================================================================


@pytest.mark.skip(reason="Rung 6: experiments/run.py not implemented")
def test_experiment_is_bit_reproducible():
    """Same config plus same seed produces bit-identical results.

    Acceptance: running ``configs/gbm_zerocost_entropic.yaml`` twice at seed 0 yields
    identical metrics to full float precision — not ``approx``, identical.

    This is a headline claim of the paper. If it fails, find the source of nondeterminism:
    global random state instead of an explicit generator, dict/set iteration order, or
    non-deterministic GPU kernels (``tf.config.experimental.enable_op_determinism()``).
    """


@pytest.mark.skip(reason="Rung 6: experiments/run.py not implemented")
def test_train_and_eval_seeds_are_disjoint():
    """Evaluation never touches training paths.

    Acceptance: the seed sets are provably disjoint by construction, via
    ``evaluation.seed_offset`` in the config.

    Not a nicety. Overlapping seeds would conflate overfitting with regime fragility, and
    those are different phenomena with different remedies — it would undermine §8, the
    result this project most wants to be right about.
    """
