"""Geometric Brownian motion — the base world.

The setting where ground truth is known: under GBM with zero transaction costs, the
risk-minimising hedge **is** the Black-Scholes delta. That makes this world the correctness
gate for the entire project (rung 4 of the ladder), not merely the simplest case.

Exact solution of ``dS = mu S dt + sigma S dW``:

    S_{i+1} = S_i * exp( (mu - sigma^2/2) dt + sigma sqrt(dt) Z_i ),   Z_i ~ N(0,1)

Use the exact form above, not an Euler step. Euler introduces discretisation bias that will
show up as a small, confusing, persistent error in rung 1 and cost you an afternoon.

For pricing and hedging under the risk-neutral measure, ``mu`` is the risk-free rate ``r``.
Keep them as separate arguments anyway -- the real-world drift matters for the
regime-shift experiments in ``evaluation/stress.py``.
"""

from __future__ import annotations

import tensorflow as tf

__all__ = ["simulate_gbm"]


def simulate_gbm(
    n_paths: int,
    n_steps: int,
    s0: float,
    mu: float,
    sigma: float,
    maturity: float,
    generator: tf.random.Generator,
    dtype: tf.DType = tf.float32,
) -> tf.Tensor:
    """Simulate GBM price paths.

    Args:
        n_paths: number of independent paths.
        n_steps: number of time steps; the output has ``n_steps + 1`` columns.
        s0: initial spot price.
        mu: drift. Set to the risk-free rate ``r`` for risk-neutral pricing.
        sigma: volatility, annualised.
        maturity: ``T`` in years. Step size is ``dt = maturity / n_steps``.
        generator: explicit ``tf.random.Generator``. **Required, not optional** — global
            random state would break the bit-reproducibility claim that the benchmark
            rests on.
        dtype: float dtype for the output.

    Returns:
        ``(n_paths, n_steps + 1)`` price paths. Column 0 is ``s0`` exactly, on every path.

    Implementation notes:
        - Draw all normals at once: ``generator.normal((n_paths, n_steps))``. One call is
          faster and, more importantly, reproducible in a way that a per-step loop is not.
        - Build the path with ``tf.cumsum`` on the log-increments, then ``exp``. Working in
          logs avoids the negative prices a naive cumulative product can produce.
        - Prepend the ``s0`` column with ``tf.concat`` so column 0 is exact rather than the
          result of a floating-point round trip.
    """
    raise NotImplementedError
