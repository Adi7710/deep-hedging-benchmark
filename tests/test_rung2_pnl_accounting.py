"""Rung 2 — P&L accounting.

The most important tests in the repository, and the least glamorous. Most failed deep
hedging reproductions are P&L bookkeeping bugs hidden inside a training loop, where they
present as "the network won't converge" and cost days.

Every test here is checkable by hand. That is deliberate: the whole point of a
single-source-of-truth :mod:`dhbench.pnl` is that its correctness can be established
without reference to anything stochastic.

The convergence test at the bottom is the headline: a delta-hedged short call with zero
costs should have terminal P&L concentrating on zero as rebalancing gets finer.
"""

from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

from dhbench.baselines.bs_delta import bs_call_price, delta_hedge_positions
from dhbench.pnl import (
    discount_factors,
    hedging_gains,
    terminal_pnl,
    transaction_costs,
    turnover,
)
from dhbench.worlds.gbm import simulate_gbm

S0 = 100.0
STRIKE = 100.0
MATURITY = 1.0
RATE = 0.0
SIGMA = 0.2


# --------------------------------------------------------------------------------------
# Hand-checkable cases
# --------------------------------------------------------------------------------------

def test_hedging_gains_single_step():
    """One step, one unit held, price moves +10 -> gain of 10."""
    spot = tf.constant([[100.0, 110.0]])
    delta = tf.constant([[1.0]])
    assert float(hedging_gains(spot, delta)[0]) == pytest.approx(10.0)


def test_hedging_gains_flat_position_is_zero():
    """Hold nothing, earn nothing, whatever the price does."""
    spot = tf.constant([[100.0, 150.0, 60.0, 100.0]])
    delta = tf.zeros((1, 3))
    assert float(hedging_gains(spot, delta)[0]) == pytest.approx(0.0)


def test_hedging_gains_short_position_sign():
    """Short one unit into a rising market loses money. Guards a flipped sign."""
    spot = tf.constant([[100.0, 110.0]])
    delta = tf.constant([[-1.0]])
    assert float(hedging_gains(spot, delta)[0]) == pytest.approx(-10.0)


def test_transaction_costs_includes_final_liquidation():
    """**The classic bug.** The cost sum runs to ``n``, not ``n-1``.

    Buy 1 unit at 100, hold one step, unwind at 110. Two trades, not one:

        open:  c * 100 * |1 - 0| = 1.0
        close: c * 110 * |0 - 1| = 1.1
        total = 2.1

    Omitting the liquidation gives 1.0 and makes every hedging strategy look cheaper than
    it is -- which biases the whole benchmark in favour of high-turnover policies.
    """
    spot = tf.constant([[100.0, 110.0]])
    delta = tf.constant([[1.0]])
    cost = float(transaction_costs(spot, delta, cost_rate=0.01)[0])
    assert cost == pytest.approx(2.1), (
        "Expected 2.1 (open at 100 + close at 110). Getting 1.0 means the final "
        "liquidation is missing from the cost sum."
    )


def test_transaction_costs_are_never_negative():
    """Costs are paid on ``|traded|``. A negative cost means a missing absolute value."""
    generator = tf.random.Generator.from_seed(3)
    spot = simulate_gbm(200, 20, S0, 0.0, SIGMA, MATURITY, generator)
    delta = generator.normal((200, 20))
    assert bool(tf.reduce_all(transaction_costs(spot, delta, 0.01) >= 0.0))


def test_zero_cost_rate_gives_zero_cost():
    """Degenerate check: no cost rate, no cost, regardless of turnover."""
    generator = tf.random.Generator.from_seed(4)
    spot = simulate_gbm(100, 10, S0, 0.0, SIGMA, MATURITY, generator)
    delta = generator.normal((100, 10))
    assert float(tf.reduce_max(tf.abs(transaction_costs(spot, delta, 0.0)))) == pytest.approx(0.0)


def test_turnover_counts_open_and_close():
    """Same padding convention as costs: go to 1, come back to 0, turnover is 2."""
    delta = tf.constant([[1.0]])
    assert float(turnover(delta)[0]) == pytest.approx(2.0)


def test_turnover_is_price_independent():
    """Turnover is a quantity, not a value -- it must not depend on spot."""
    delta = tf.constant([[0.5, 0.7, 0.2]])
    assert float(turnover(delta)[0]) == pytest.approx(0.5 + 0.2 + 0.5 + 0.2)


# --------------------------------------------------------------------------------------
# terminal_pnl composition
# --------------------------------------------------------------------------------------

def test_terminal_pnl_sign_convention():
    """We are SHORT the claim and RECEIVE the premium.

    No hedge, no costs: P&L = premium - payoff. Buehler et al. flip this in places, so it
    is pinned here.
    """
    spot = tf.constant([[100.0, 120.0]])
    delta = tf.zeros((1, 1))
    payoff = tf.constant([20.0])
    pnl = float(terminal_pnl(spot, delta, payoff, cost_rate=0.0, premium=8.0)[0])
    assert pnl == pytest.approx(8.0 - 20.0)


def test_terminal_pnl_equals_its_parts():
    """``terminal_pnl`` is exactly ``premium - payoff + gains - costs``.

    Catches a term dropped or double-counted during composition.
    """
    generator = tf.random.Generator.from_seed(11)
    spot = simulate_gbm(500, 12, S0, 0.0, SIGMA, MATURITY, generator)
    delta = generator.normal((500, 12)) * 0.5
    payoff = tf.maximum(spot[:, -1] - STRIKE, 0.0)
    cost_rate, premium = 0.002, 8.0

    combined = terminal_pnl(spot, delta, payoff, cost_rate, premium)
    expected = (
        premium
        - payoff
        + hedging_gains(spot, delta)
        - transaction_costs(spot, delta, cost_rate)
    )
    np.testing.assert_allclose(combined.numpy(), expected.numpy(), rtol=1e-5)


# --------------------------------------------------------------------------------------
# The headline test
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("n_steps", [10, 40, 160])
def test_delta_hedge_error_shrinks_with_rebalancing(n_steps):
    """Zero-cost delta hedging converges as rebalancing gets finer.

    A short call, delta hedged, with the premium set to the Black-Scholes price, should
    have terminal P&L concentrated near zero -- and the residual should shrink roughly as
    ``1/sqrt(n_steps)``.

    This is the strongest available evidence that the simulator, the Black-Scholes
    reference, and the P&L accounting are **mutually** consistent. Each could be wrong
    alone; all three being wrong in a way that still converges is very unlikely.
    """
    generator = tf.random.Generator.from_seed(99)
    spot = simulate_gbm(20_000, n_steps, S0, RATE, SIGMA, MATURITY, generator)

    delta = tf.constant(
        delta_hedge_positions(spot.numpy(), STRIKE, MATURITY, RATE, SIGMA),
        dtype=tf.float32,
    )
    payoff = tf.maximum(spot[:, -1] - STRIKE, 0.0)
    premium = float(bs_call_price(S0, STRIKE, MATURITY, RATE, SIGMA))

    pnl = terminal_pnl(spot, delta, payoff, cost_rate=0.0, premium=premium)

    # Mean near zero: the premium was fair.
    assert float(tf.reduce_mean(pnl)) == pytest.approx(0.0, abs=0.15)

    # Residual scales as ~1/sqrt(n). The constant is loose on purpose -- this is a
    # convergence-rate check, not a precision claim.
    std = float(tf.math.reduce_std(pnl))
    assert std < 8.0 / np.sqrt(n_steps), (
        f"P&L std {std:.4f} too large for n_steps={n_steps}. Discrete delta hedging "
        f"error should scale as 1/sqrt(n_steps)."
    )


# --------------------------------------------------------------------------------------
# Discounting — the numeraire path
# --------------------------------------------------------------------------------------

def test_discount_factors_start_at_one():
    """t_0 = 0, so the first factor is exactly 1 — not 0.9999998."""
    factors = discount_factors(4, 0.05, 1.0)
    assert float(factors[0]) == 1.0


def test_discount_factors_match_closed_form():
    """exp(-r t_i) on the grid t_i = i * maturity / n_steps."""
    got = discount_factors(4, 0.05, 2.0).numpy()
    expected = np.exp(-0.05 * np.linspace(0.0, 2.0, 5))
    np.testing.assert_allclose(got, expected, rtol=1e-6)


def test_zero_rate_discounting_is_the_identity():
    """r = 0 must be exactly 1.0 everywhere, so the default path is untouched."""
    np.testing.assert_array_equal(
        discount_factors(6, 0.0, 1.0).numpy(), np.ones(7, dtype=np.float32)
    )


def test_terminal_pnl_at_zero_rate_ignores_maturity():
    """The rate=0 default must not depend on maturity being supplied."""
    spot = tf.constant([[100.0, 104.0, 102.0, 109.0]])
    delta = tf.constant([[2.0, 2.0, -1.0]])
    payoff = tf.constant([9.0])
    a = terminal_pnl(spot, delta, payoff, 0.01, 8.0)
    b = terminal_pnl(spot, delta, payoff, 0.01, 8.0, rate=0.0, maturity=1.0)
    np.testing.assert_allclose(a.numpy(), b.numpy(), rtol=1e-6)


def test_terminal_pnl_discounts_exactly_like_discounting_the_inputs():
    """The numeraire claim, tested rather than asserted.

    (1) is form-invariant under S -> exp(-rt) S, so discounting the inputs by hand and
    calling the r=0 path must equal passing real inputs with a rate. If these disagree,
    either the cost term is discounting on the wrong grid or the payoff is not being
    discounted to T.
    """
    generator = tf.random.Generator.from_seed(21)
    spot = simulate_gbm(400, 12, S0, 0.0, SIGMA, MATURITY, generator)
    delta = generator.normal((400, 12)) * 0.5
    payoff = tf.maximum(spot[:, -1] - STRIKE, 0.0)
    rate, cost_rate, premium = 0.05, 0.002, 8.0

    factors = discount_factors(12, rate, MATURITY, dtype=spot.dtype)
    by_hand = terminal_pnl(
        spot * factors, delta, payoff * factors[-1], cost_rate, premium
    )
    automatic = terminal_pnl(
        spot, delta, payoff, cost_rate, premium, rate=rate, maturity=MATURITY
    )
    np.testing.assert_allclose(automatic.numpy(), by_hand.numpy(), rtol=1e-5)


def test_nonzero_rate_without_maturity_raises():
    """The time grid cannot be inferred from spot alone — fail loudly, never guess.

    A silent default of maturity=1.0 would be wrong for every option that is not
    one year, and wrong quietly.
    """
    spot = tf.constant([[100.0, 110.0]])
    with pytest.raises(ValueError, match="maturity"):
        terminal_pnl(spot, tf.constant([[1.0]]), tf.constant([5.0]), rate=0.05)


def test_terminal_pnl_works_under_tf_function():
    """Stage 2 wraps this in @tf.function with a dynamic batch.

    ``int(spot.shape[-1])`` raises there because the static shape is None. Pinned so the
    graph-mode path cannot regress unnoticed.
    """
    @tf.function(input_signature=[
        tf.TensorSpec([None, None], tf.float32),
        tf.TensorSpec([None, None], tf.float32),
        tf.TensorSpec([None], tf.float32),
    ])
    def compiled(spot, delta, payoff):
        return terminal_pnl(spot, delta, payoff, 0.01, 8.0, rate=0.05, maturity=1.0)

    spot = tf.constant([[100.0, 104.0, 102.0, 109.0]])
    delta = tf.constant([[2.0, 2.0, -1.0]])
    payoff = tf.constant([9.0])
    np.testing.assert_allclose(
        compiled(spot, delta, payoff).numpy(),
        terminal_pnl(spot, delta, payoff, 0.01, 8.0, rate=0.05, maturity=1.0).numpy(),
        rtol=1e-6,
    )


def test_gradients_flow_through_terminal_pnl():
    """Stage 2 differentiates the objective back to the policy through this function.

    A .numpy() anywhere inside pnl.py severs the tape silently: no error, zero gradient,
    a network that trains without learning. Pinned here so it cannot creep in.
    """
    spot = tf.constant([[100.0, 104.0, 102.0, 109.0]])
    delta = tf.Variable([[2.0, 2.0, -1.0]])
    payoff = tf.constant([9.0])
    with tf.GradientTape() as tape:
        loss = tf.reduce_sum(terminal_pnl(spot, delta, payoff, 0.01, 8.0))
    grad = tape.gradient(loss, delta)
    assert grad is not None, "gradient is None -- the tape was severed inside pnl.py"
    assert float(tf.reduce_max(tf.abs(grad))) > 0.0, "gradient is identically zero"
