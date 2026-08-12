"""Geometric Brownian motion -- the base world, where ground truth is known.

Exact solution of ``dS = mu S dt + sigma S dW``:

    S_{i+1} = S_i * exp( (mu - sigma^2/2) dt + sigma sqrt(dt) Z_i ),   Z_i ~ N(0,1)

Use the exact form, not an Euler step -- Euler adds a discretisation bias that shows up
as a small persistent error in rung 1.
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
    """Simulate GBM price paths. ``dt = maturity / n_steps``.

    -> (n_paths, n_steps + 1), with column 0 exactly s0 on every path.

    mu: set to the risk-free rate r for risk-neutral pricing. Kept separate from r
        because real-world drift matters for the regime-shift experiments.
    generator: required, not optional -- global random state would break the
        bit-reproducibility claim the benchmark rests on.

    Recipe: draw all normals in one call, ``tf.cumsum`` the log-increments, ``exp``,
    then ``tf.concat`` the s0 column on the front. Working in logs avoids the negative
    prices a naive cumulative product can produce.
    """
    raise NotImplementedError