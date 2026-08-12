"""Terminal P&L accounting -- the single source of truth. Nothing else computes P&L.

    PL_T = p_0 - Z + sum_{i=0}^{n-1} delta_i (S_{i+1} - S_i)     <- hedging gains
                   - sum_{i=0}^{n}   c S_i |delta_i - delta_{i-1}|  <- costs

Conventions: delta_{-1} = 0 (start flat), delta_n = 0 (liquidate at T).
We are SHORT the claim (-Z) and RECEIVE the premium (+p_0).

NUMERAIRE
---------
The functional above is the r = 0 form: it comes from telescoping the self-financing
recursion, and with r != 0 the cash account does not cancel. Rather than carry an
explicit cash-account state machine, the benchmark is specified in **discounted (time-0)
units**, where the functional keeps exactly this shape:

    S~_i = exp(-r t_i) S_i          discounted price
    Z~   = exp(-r T)   Z            discounted payoff

    PL~ = p_0 - Z~ + sum_i delta_i (S~_{i+1} - S~_i)
                   - sum_i c S~_i |delta_i - delta_{i-1}|

Two facts make this work. The discounted wealth of a self-financing strategy is the
discrete stochastic integral of delta against the *discounted* price. And a cost paid at
t_i is c S_i |dq| in time-i money, so discounting it gives c S~_i |dq| -- the cost term
discounts with the same factor as the price it is charged on. So (1) is form-invariant
and ``hedging_gains`` / ``transaction_costs`` need no rate argument at all.

p_0 needs no adjustment: the premium is received at t_0 and is already time-0 money, as
is the Black-Scholes price it is usually set to.

Assumes ONE rate for borrowing and lending. Funding spreads, collateral or asymmetric
borrow/lend would break the collapse into a single factor and require real cash flows.
Stated as a boundary in the paper, not discovered later.

Pass ``terminal_pnl`` REAL prices and a REAL payoff, with ``rate`` and ``maturity``; it
converts. Do not discount before calling, or you will discount twice.
"""

from __future__ import annotations

import tensorflow as tf

__all__ = [
    "terminal_pnl",
    "hedging_gains",
    "transaction_costs",
    "turnover",
    "discount_factors",
]


def discount_factors(
    n_steps: int,
    rate: float,
    maturity: float,
    dtype: tf.DType = tf.float32,
) -> tf.Tensor:
    """``exp(-rate * t_i)`` on the trading grid ``t_i = i * maturity / n_steps``.

    -> (n_steps + 1,), matching the column count of ``spot``. Entry 0 is exactly 1.

    Public so evaluation code discounts on the same grid rather than re-deriving it;
    an off-by-one in the time grid is invisible in the output and fatal in the numbers.
    """
    times = tf.linspace(
        tf.constant(0.0, dtype=dtype),
        tf.constant(maturity, dtype=dtype),
        n_steps + 1,
    )
    return tf.exp(-tf.constant(rate, dtype=dtype) * times)


def _traded(delta: tf.Tensor) -> tf.Tensor:
    """Signed quantity traded at each of the n_steps + 1 trade dates.

    delta: (n_paths, n_steps)  ->  (n_paths, n_steps + 1)

    Pad with delta_{-1} = 0 and delta_n = 0, then difference: n decisions become
    n + 1 trades, because the position must be opened and must be unwound. The
    result has the same column count as ``spot``, which is the check that both
    boundary trades survived -- pad only one end and the multiply by spot raises
    a shape error instead of silently dropping the liquidation.

    Shared by transaction_costs and turnover so the padding convention is
    enforced in one place rather than restated in two.
    """
    zeros = tf.zeros_like(delta[:, :1])
    padded = tf.concat([zeros, delta, zeros], axis=-1)   # (n_paths, n_steps + 2)
    return padded[:, 1:] - padded[:, :-1]                # (n_paths, n_steps + 1)


def hedging_gains(spot: tf.Tensor, delta: tf.Tensor) -> tf.Tensor:
    """Hedge gains before costs: ``sum_{i=0}^{n-1} delta_i (S_{i+1} - S_i)``.

    spot:  (n_paths, n_steps + 1)
    delta: (n_paths, n_steps)
    ->     (n_paths,)

    delta has one fewer column than spot on purpose: one decision per *gap*, not
    per price. Stay in TensorFlow -- a ``.numpy()`` here kills Stage 2 gradients.

    Numeraire-agnostic: feed it discounted prices and you get discounted gains.
    """
    moves = spot[:, 1:] - spot[:, :-1]      # (n_paths, n_steps): S_{i+1} - S_i
    return tf.reduce_sum(delta * moves, axis=-1)   # collapse time, keep paths


def transaction_costs(
    spot: tf.Tensor,
    delta: tf.Tensor,
    cost_rate: float,
) -> tf.Tensor:
    """Proportional cost: ``sum_{i=0}^{n} c S_i |delta_i - delta_{i-1}|``.

    spot:  (n_paths, n_steps + 1)
    delta: (n_paths, n_steps)
    ->     (n_paths,), always >= 0

    The sum runs to n, not n-1: unwinding the final position at T is a real trade
    and costs real money. Pad delta with a leading AND a trailing zero before
    differencing, giving n_steps + 1 traded amounts.

    Numeraire-agnostic, same as hedging_gains: a cost paid at t_i discounts by the
    same factor as the S_i it is charged on.
    """
    traded = _traded(delta)                                    # (n_paths, n_steps + 1)
    return cost_rate * tf.reduce_sum(spot * tf.abs(traded), axis=-1)


def turnover(delta: tf.Tensor) -> tf.Tensor:
    """Total traded quantity: ``sum_i |delta_i - delta_{i-1}|``, padding as above.

    delta: (n_paths, n_steps)
    ->     (n_paths,)

    Prices deliberately do not appear -- so this is numeraire-free by construction.
    Reported to answer "is the agent's edge just trading more?"
    """
    return tf.reduce_sum(tf.abs(_traded(delta)), axis=-1)


def terminal_pnl(
    spot: tf.Tensor,
    delta: tf.Tensor,
    payoff: tf.Tensor,
    cost_rate: float = 0.0,
    premium: float = 0.0,
    rate: float = 0.0,
    maturity: float | None = None,
) -> tf.Tensor:
    """``premium - payoff + hedging_gains - transaction_costs``, in time-0 money.

    spot:   (n_paths, n_steps + 1) -- REAL prices, not pre-discounted
    delta:  (n_paths, n_steps)
    payoff: (n_paths,) -- the claim Z, which we are SHORT. Real, not pre-discounted.
    rate:   risk-free r. Zero (the default) makes discounting the identity.
    maturity: T in years. Required when rate != 0; ignored when rate == 0.
    ->      (n_paths,), higher is better

    Everything the objectives measure, and what gradients flow back through.

    This is the ONLY place discounting happens. Callers pass real quantities; see the
    module docstring for why the functional is form-invariant under the change of
    numeraire, and why premium needs no adjustment.
    """
    if rate != 0.0:
        if maturity is None:
            raise ValueError(
                "terminal_pnl: maturity is required when rate != 0 -- discount factors "
                "are exp(-rate * t_i) and the time grid cannot be inferred from spot "
                "alone. Pass maturity=T, or rate=0.0 to work in undiscounted units."
            )
        n_steps = int(spot.shape[-1]) - 1
        factors = discount_factors(n_steps, rate, maturity, dtype=spot.dtype)
        spot = spot * factors           # (n_paths, n_steps+1) * (n_steps+1,)
        payoff = payoff * factors[-1]   # exp(-rate * T)

    return (
        premium
        - payoff
        + hedging_gains(spot, delta)
        - transaction_costs(spot, delta, cost_rate)
    )
