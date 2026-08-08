"""Band-structured agent — no-transaction band as an architectural prior.

Following Arzel & Lehdili (2026), *Bridging Stochastic Control and Deep Hedging: Structural
Priors for No-Transaction Band Networks*.

Instead of hoping the network discovers band behaviour, **build it in**. The network
outputs a band, and the trading rule is imposed:

    centre_i, width_i = f_theta( t_i, S_i, ... )
    delta_i = clip( delta_{i-1}, centre_i - width_i, centre_i + width_i )

The ``clip`` is the trading rule: if the inherited position is already inside the band,
it passes through unchanged (no trade); if outside, it is moved to the nearest edge.

**Why this is a good benchmark entry.** Theory says the optimal policy under proportional
costs *is* a band (Whalley-Wilmott). A vanilla network has to spend capacity rediscovering
that structure from scratch. Baking it in should mean faster training, fewer parameters,
and better sample efficiency — and if it *doesn't*, that is a genuinely interesting
negative result about how much structure these problems actually need.

It also sharpens §8: a structural prior derived from theory that holds across regimes
ought to be **more robust to regime shift** than a freely-learned policy that may have
overfitted the training dynamics. That is a concrete, falsifiable prediction, which is
exactly what a benchmark should be testing.

Implementation note: ``clip_by_value`` is differentiable, with zero gradient outside the
clip range — which is correct here. Inside the band no trade happens, so there is nothing
for the gradient to say about ``delta_prev``; the gradient flows through ``centre`` and
``width`` instead.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow import keras

__all__ = ["BandAgent"]


class BandAgent(keras.Model):
    """No-transaction band network.

    Args:
        hidden_units: units per hidden layer of the band network.
        activation: hidden activation.
        parameterise_around_delta: if ``True``, the network outputs a *correction* to
            ``delta_BS`` rather than an absolute centre.

            This is a strong prior and cuts both ways. It should train far faster, and it
            matches the 2026 finding that deep hedging mostly learns "delta corrections."
            But it **hands the agent the analytic answer**, so it cannot be used for the
            rung-4 recovery test, and its wins are not comparable with agents that started
            from nothing. Report both variants, and say which is which.
    """

    def __init__(
        self,
        hidden_units: tuple[int, ...] = (32, 32),
        activation: str = "relu",
        parameterise_around_delta: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.hidden_units = hidden_units
        self.activation = activation
        self.parameterise_around_delta = parameterise_around_delta
        raise NotImplementedError(
            "Build the Dense stack with a 2-unit output: (centre, raw_width). Apply "
            "softplus to raw_width so the half-width is strictly positive -- a negative "
            "width would invert the clip and produce nonsense that still trains."
        )

    def call(self, features: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        """Produce the band for a single timestep.

        Args:
            features: ``(n_paths, n_features)`` state.

        Returns:
            ``(centre, half_width)``, each ``(n_paths,)``. ``half_width`` is strictly
            positive.
        """
        raise NotImplementedError

    def hedge_path(self, spot: tf.Tensor, maturity: float) -> tf.Tensor:
        """Roll the band policy forward along whole paths.

        Args:
            spot: ``(n_paths, n_steps + 1)`` simulated prices.
            maturity: ``T`` in years.

        Returns:
            ``(n_paths, n_steps)`` hedge positions.

        Note:
            Per step: compute the band, then
            ``delta_i = tf.clip_by_value(delta_prev, centre - width, centre + width)``.

            At ``i = 0``, ``delta_prev = 0`` — so the first position is the band edge
            nearest zero, not the centre. That is correct behaviour and worth checking
            explicitly, since it is the one step where the clip does something
            counter-intuitive.
        """
        raise NotImplementedError

    def learned_band(
        self, spot: tf.Tensor, maturity: float
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """Expose the band directly, for comparison against Whalley-Wilmott.

        This is what makes rung 5 of the correctness ladder checkable for this agent: plot
        the learned half-width against the analytic ``H`` across moneyness and
        time-to-maturity. They should have the same shape — widest where gamma is highest.

        Args:
            spot: ``(n_paths, n_steps + 1)`` simulated prices.
            maturity: ``T`` in years.

        Returns:
            ``(centres, half_widths)``, each ``(n_paths, n_steps)``.
        """
        raise NotImplementedError
