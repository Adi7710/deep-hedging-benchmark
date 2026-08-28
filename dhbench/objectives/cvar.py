"""Conditional Value at Risk, in Rockafellar-Uryasev form.

    rho(X) = min_w { w + (1 / (1 - alpha)) * E[ (-X - w)^+ ] }

CVaR at level ``alpha`` is the expected loss conditional on being in the worst
``(1 - alpha)`` tail. It is coherent, it is what risk managers actually report, and it
targets exactly the tail that hedging exists to control.

**The thing to get right: ``w`` is a trainable variable.**

The naive route — sort the P&L, take the worst 5%, average — is correct as a *metric* but
its gradient is nearly useless for training: it depends only on the tail samples, so most
of the batch contributes nothing to the update.

Rockafellar & Uryasev (2000) showed the minimisation above has the same optimum and is
smooth in ``w``. So you make ``w`` an extra ``tf.Variable``, hand it to the optimiser
alongside the network weights, and minimise jointly. Every path then contributes gradient
through the hinge. At the optimum ``w`` converges to the value-at-risk, which is a free
diagnostic — if ``w`` drifts far from the empirical VaR of your P&L, training has not
converged.

This joint-optimisation detail is the usual place CVaR deep hedging implementations go
wrong, and the symptom (training that plateaus early) looks like a learning-rate problem
rather than a formulation problem.
"""

from __future__ import annotations

import tensorflow as tf

__all__ = ["CVaRRisk"]


class CVaRRisk:
    """CVaR risk measure with the auxiliary variable optimised jointly.

    Args:
        alpha: confidence level, e.g. ``0.95`` for CVaR-95 — the mean of the worst 5%.
        initial_w: starting value for the auxiliary variable.

    Attributes:
        w: the Rockafellar-Uryasev auxiliary variable. **Must be included in the
            optimiser's variable list**, e.g.
            ``tape.gradient(loss, model.trainable_variables + [objective.w])``.
            Forgetting it is the classic bug — training runs, loss decreases a little,
            then flatlines.
    """

    def __init__(self, alpha: float = 0.95, initial_w: float = 0.0) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self.alpha = alpha
        self.w = tf.Variable(initial_w, dtype=tf.float32, name="cvar_w", trainable=True)

    def __call__(self, pnl: tf.Tensor) -> tf.Tensor:
        """Evaluate the risk measure. This is the training loss.

        Args:
            pnl: ``(n_paths,)`` terminal P&L.

        Returns:
            Scalar. Lower is better.

        Note:
            ``w + mean(relu(-pnl - w)) / (1 - alpha)``. Use ``tf.nn.relu`` for the
            positive part — it has the correct subgradient at zero.
        """
        w = tf.cast(self.w, pnl.dtype)
        tail = tf.reduce_mean(tf.nn.relu(-pnl - w))
        return w + tail / tf.constant(1.0 - self.alpha, dtype=pnl.dtype)

    @property
    def trainable_variables(self) -> list[tf.Variable]:
        """The auxiliary variable, to be concatenated with the model's.

        Provided so the training loop can write
        ``model.trainable_variables + objective.trainable_variables`` and not have to know
        whether a given objective carries state.
        """
        return [self.w]

    @property
    def name(self) -> str:
        """Identifier used in config files and results tables."""
        return f"cvar_{self.alpha}"


def cvar_empirical(pnl: tf.Tensor, alpha: float = 0.95) -> tf.Tensor:
    """Empirical CVaR by sorting — for **reporting**, not training.

    The direct estimator: sort, take the worst ``(1 - alpha)`` fraction, average. Correct
    and easy to verify, but a poor training signal, which is why :class:`CVaRRisk` exists
    separately.

    Used in :mod:`dhbench.evaluation.metrics` to report final numbers, and as a
    cross-check that :class:`CVaRRisk` converged — the two should agree closely at the
    optimum. A persistent gap means training stopped early.

    Args:
        pnl: ``(n_paths,)`` terminal P&L.
        alpha: confidence level.

    Returns:
        Scalar CVaR. Positive numbers mean expected loss in the tail.
    """
    raise NotImplementedError
