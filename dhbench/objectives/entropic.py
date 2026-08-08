"""Entropic risk measure (exponential utility).

    rho(X) = (1 / lambda) * log E[ exp(-lambda * X) ]

The default objective in Buehler et al. (2019), and a genuine convex risk measure. Larger
``lambda`` means more risk aversion; as ``lambda -> 0`` it degenerates to ``-E[X]``.

**The numerical trap, and it is not optional.** Never write this as
``log(reduce_mean(exp(-lam * x)))``. P&L values times ``lambda`` land in a range where
``exp`` overflows ``float32`` routinely, and the failure is silent-ish: you get ``inf``,
then ``nan`` gradients, and it looks like a divergent training run rather than an
arithmetic bug. People lose days to this.

Use ``tf.reduce_logsumexp``, which internally subtracts the max before exponentiating:

    rho(X) = (1/lam) * ( logsumexp(-lam * X) - log(n) )

Same value, stable everywhere.

The entropic measure also has the property that its indifference price has a closed form in
simple cases, which makes it useful for validation beyond just being the field's default.
"""

from __future__ import annotations

import tensorflow as tf

__all__ = ["EntropicRisk"]


class EntropicRisk:
    """Entropic (exponential utility) risk measure.

    Args:
        risk_aversion: ``lambda > 0``. Buehler et al. use values around 1.0; the protocol
            fixes two levels so risk-aversion sensitivity is an axis rather than a
            confound.
    """

    def __init__(self, risk_aversion: float = 1.0) -> None:
        if risk_aversion <= 0:
            raise ValueError(f"risk_aversion must be positive, got {risk_aversion}")
        self.risk_aversion = risk_aversion

    def __call__(self, pnl: tf.Tensor) -> tf.Tensor:
        """Evaluate the risk measure. This is the training loss.

        Args:
            pnl: ``(n_paths,)`` terminal P&L from :func:`dhbench.pnl.terminal_pnl`.

        Returns:
            Scalar. Lower is better — minimise this directly.

        Warning:
            Implement via ``tf.reduce_logsumexp``. See the module docstring; this is the
            single most common numerical failure in deep hedging implementations.
        """
        raise NotImplementedError

    @property
    def name(self) -> str:
        """Identifier used in config files and results tables."""
        return f"entropic_{self.risk_aversion}"
