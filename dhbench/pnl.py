"""Terminal P&L accounting.

**This module is the single source of truth for P&L.** Nothing else in the codebase may
compute it. Most failed deep hedging reproductions trace to bookkeeping bugs hidden inside
a training loop, and centralising the arithmetic here is what makes rung 2 of the
correctness ladder meaningful.

The functional (see ``docs/00-problem-statement.md``):

    PL_T = p_0 - Z + sum_{i=0}^{n-1} delta_i (S_{i+1} - S_i)
                   - sum_{i=0}^{n}   c S_i |delta_i - delta_{i-1}|

Three conventions, each a classic bug if you get it wrong:

1. ``delta_{-1} = 0`` -- start flat.
2. ``delta_n = 0`` -- the position is liquidated at T, and **that liquidation costs
   money**. The cost sum therefore runs to ``n``, not ``n-1``. Omitting it makes deep
   hedging look better than it is.
3. Costs are charged on the *traded* amount ``|delta_i - delta_{i-1}|``, never the held
   amount.

Sign convention: we are **short** the claim (``-Z``) and **receive** the premium
(``+p_0``). Buehler et al. flip this in places; ours is fixed here and everywhere.
"""

from __future__ import annotations

import tensorflow as tf

__all__ = ["terminal_pnl", "hedging_gains", "transaction_costs", "turnover"]


def hedging_gains(spot: tf.Tensor, delta: tf.Tensor) -> tf.Tensor:
    """Gains from the hedge portfolio, before costs.

    Implements ``sum_{i=0}^{n-1} delta_i (S_{i+1} - S_i)``.

    Args:
        spot: ``(n_paths, n_steps + 1)`` underlying price paths.
        delta: ``(n_paths, n_steps)`` hedge position held over ``[t_i, t_{i+1})``.

    Returns:
        ``(n_paths,)`` total hedging gain per path.

    Note:
        ``delta`` has one fewer column than ``spot``. The position at index ``i`` is held
        across the increment from ``i`` to ``i+1``, so it multiplies ``diff(spot)``, which
        also has ``n_steps`` columns. If the shapes fight you, that's the check working.
    """
    raise NotImplementedError


def transaction_costs(
    spot: tf.Tensor,
    delta: tf.Tensor,
    cost_rate: float,
) -> tf.Tensor:
    """Total proportional transaction cost paid over the path.

    Implements ``sum_{i=0}^{n} c S_i |delta_i - delta_{i-1}|`` with ``delta_{-1} = 0`` and
    ``delta_n = 0``.

    Args:
        spot: ``(n_paths, n_steps + 1)`` underlying price paths.
        delta: ``(n_paths, n_steps)`` hedge positions.
        cost_rate: proportional cost ``c``, e.g. ``0.005`` for 50bp.

    Returns:
        ``(n_paths,)`` total cost paid per path. Always non-negative.

    Warning:
        The sum runs to ``n``, not ``n-1`` -- unwinding the final position at ``T`` is a
        real trade and costs real money. Pad ``delta`` with a leading and a trailing zero
        before differencing, so the traded amounts have ``n_steps + 1`` entries.
    """
    raise NotImplementedError


def turnover(delta: tf.Tensor) -> tf.Tensor:
    """Total absolute traded quantity, ignoring prices.

    ``sum_i |delta_i - delta_{i-1}|`` with the same padding convention as
    :func:`transaction_costs`.

    Reported as a benchmark metric to answer "is the agent's edge just trading more?" --
    a policy that beats a baseline purely by higher turnover under a low cost rate has not
    learned anything transferable.

    Args:
        delta: ``(n_paths, n_steps)`` hedge positions.

    Returns:
        ``(n_paths,)`` turnover per path.
    """
    raise NotImplementedError


def terminal_pnl(
    spot: tf.Tensor,
    delta: tf.Tensor,
    payoff: tf.Tensor,
    cost_rate: float = 0.0,
    premium: float = 0.0,
) -> tf.Tensor:
    """Terminal profit and loss of the hedged short-claim position.

    The quantity every objective in :mod:`dhbench.objectives` is a functional of, and the
    thing gradients flow back through during training.

    Args:
        spot: ``(n_paths, n_steps + 1)`` underlying price paths.
        delta: ``(n_paths, n_steps)`` hedge positions.
        payoff: ``(n_paths,)`` claim payoff ``Z`` at maturity. We are short this.
        cost_rate: proportional transaction cost ``c``. Zero recovers the frictionless case.
        premium: ``p_0``, received at inception.

    Returns:
        ``(n_paths,)`` terminal P&L. Higher is better; the objectives minimise risk of the
        *negative* tail.

    Example:
        Perfect hedge, zero costs, zero premium -- P&L concentrates near zero, and its
        standard deviation shrinks as ``n_steps`` grows. That is rung 2 of the ladder.
    """
    raise NotImplementedError
