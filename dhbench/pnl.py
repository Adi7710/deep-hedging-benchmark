"""Terminal P&L accounting -- the single source of truth. Nothing else computes P&L.

    PL_T = p_0 - Z + sum_{i=0}^{n-1} delta_i (S_{i+1} - S_i)     <- hedging gains
                   - sum_{i=0}^{n}   c S_i |delta_i - delta_{i-1}|  <- costs

Conventions: delta_{-1} = 0 (start flat), delta_n = 0 (liquidate at T).
We are SHORT the claim (-Z) and RECEIVE the premium (+p_0).
"""

from __future__ import annotations

import tensorflow as tf

__all__ = ["terminal_pnl", "hedging_gains", "transaction_costs", "turnover"]


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
    """
    raise NotImplementedError


def turnover(delta: tf.Tensor) -> tf.Tensor:
    """Total traded quantity: ``sum_i |delta_i - delta_{i-1}|``, padding as above.

    delta: (n_paths, n_steps)
    ->     (n_paths,)

    Prices deliberately do not appear. Reported to answer "is the agent's edge
    just trading more?"
    """
    raise NotImplementedError


def terminal_pnl(
    spot: tf.Tensor,
    delta: tf.Tensor,
    payoff: tf.Tensor,
    cost_rate: float = 0.0,
    premium: float = 0.0,
) -> tf.Tensor:
    """``premium - payoff + hedging_gains - transaction_costs``.

    spot:   (n_paths, n_steps + 1)
    delta:  (n_paths, n_steps)
    payoff: (n_paths,) -- the claim Z, which we are SHORT
    ->      (n_paths,), higher is better

    Everything the objectives measure, and what gradients flow back through.
    """
    raise NotImplementedError