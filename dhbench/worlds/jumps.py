"""Merton jump-diffusion — the second out-of-distribution test world.

    dS/S = (mu - lambda_j k) dt + sigma dW + (Y - 1) dN

with ``N`` a Poisson process of intensity ``lambda_j``, jump sizes ``log Y ~ N(mu_J,
sigma_J^2)``, and ``k = E[Y] - 1 = exp(mu_J + sigma_J^2/2) - 1`` the compensator that keeps
the drift correct.

Complements ``regime_switching`` in §8. The two break a learned hedge in genuinely
different ways, which is the reason for having both:

- **Regime switching** shifts the *volatility level* — gradually, and persistently.
- **Jumps** break the *continuity assumption* underlying delta hedging altogether. No
  continuous rebalancing strategy can hedge a jump, so this is a harder and more
  interesting failure.

Expect every method, learned and classical alike, to do badly here. **That is a result,
not a bug** — and reporting it is exactly the null-result commitment in ``PAPER.md`` §7.
The interesting question is not who wins but whether the *ranking* is preserved.
"""

from __future__ import annotations

import tensorflow as tf

__all__ = ["simulate_merton_jump_diffusion"]


def simulate_merton_jump_diffusion(
    n_paths: int,
    n_steps: int,
    s0: float,
    mu: float,
    sigma: float,
    jump_intensity: float,
    jump_mean: float,
    jump_std: float,
    maturity: float,
    generator: tf.random.Generator,
    dtype: tf.DType = tf.float32,
) -> tf.Tensor:
    """Simulate Merton jump-diffusion price paths.

    Args:
        n_paths: number of independent paths.
        n_steps: number of time steps.
        s0: initial spot.
        mu: drift before compensation.
        sigma: diffusive volatility.
        jump_intensity: ``lambda_j``, expected jumps per year.
        jump_mean: ``mu_J``, mean of the **log** jump size. Negative for crash risk.
        jump_std: ``sigma_J``, standard deviation of the log jump size.
        maturity: ``T`` in years.
        generator: explicit ``tf.random.Generator``.
        dtype: float dtype.

    Returns:
        ``(n_paths, n_steps + 1)`` price paths.

    Implementation notes:
        - Per step, draw the jump count ``N_i ~ Poisson(lambda_j * dt)``. Several jumps in
          one step is rare but must be handled — given ``N_i`` jumps, the aggregate log
          jump is ``N(N_i * mu_J, N_i * sigma_J^2)``, so you can draw it in one go rather
          than looping.
        - **Subtract the compensator** ``lambda_j * k * dt`` from the drift. Without it the
          process has the wrong mean and prices come out biased — a quiet error that rung 1
          on a jump world would catch, but nothing else will.
        - Build in log space and ``exp`` at the end, as in :mod:`dhbench.worlds.gbm`.
    """
    raise NotImplementedError
