"""Geometric Brownian motion -- the base world, where ground truth is known.

Exact solution of ``dS = mu S dt + sigma S dW``:

    S_{i+1} = S_i * exp( (mu - sigma^2/2) dt + sigma sqrt(dt) Z_i ),   Z_i ~ N(0,1)

Use the exact form, not an Euler step -- Euler adds a discretisation bias that shows up
as a small persistent error in rung 1.
"""

from __future__ import annotations

import math

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
    dt = maturity / n_steps

    # -sigma^2/2 is the Ito correction. Without it E[S_{i+1}] = S_i exp(mu dt + sigma^2 dt/2)
    # by Jensen, the stock drifts faster than mu, and the rung-1 MC price comes out high.
    drift = (mu - 0.5 * sigma * sigma) * dt
    vol_step = sigma * math.sqrt(dt)

    # ONE draw, not n_steps of them: a generator's output depends on its call pattern,
    # so a per-step loop is a different sequence and does not reproduce.
    z = generator.normal((n_paths, n_steps), dtype=dtype)

    log_increments = drift + vol_step * z                    # (n_paths, n_steps)
    paths = s0 * tf.exp(tf.cumsum(log_increments, axis=-1))  # (n_paths, n_steps)

    # Concat rather than compute: exp(0)*s0 is exact in theory but a float round trip.
    s0_column = tf.fill((n_paths, 1), tf.cast(s0, dtype))
    return tf.concat([s0_column, paths], axis=-1)            # (n_paths, n_steps + 1)