"""Adversarially robust agent — He et al., NeurIPS 2025.

*Distributional Adversarial Attacks and Training in Deep Hedging*
([arXiv:2508.14757](https://arxiv.org/abs/2508.14757)).

The finding this implements: standard deep hedging policies are **highly vulnerable to
small perturbations of the input distribution**, degrading sharply under shifts that ought
to be benign. The fix is adversarial training lifted from the pointwise setting to the
distributional one, over a Wasserstein ball around the training measure:

    min_theta  max_{Q in B_eps(P)}  rho_Q( PL_T(theta) )

The inner maximisation is intractable in general; the paper's contribution is a
computationally tractable reformulation over the Wasserstein ball.

This is a **Stage 5** entry, and the most involved agent in the grid. Do not attempt it
before the vanilla agent passes rung 4 — an adversarial training loop that fails is nearly
impossible to debug if you cannot trust the non-adversarial one underneath it.

**Its role in the benchmark.** This is the current state of the art for robustness, and §8
asks whether robustness training genuinely helps under a *structural* regime shift (Heston
→ regime-switching, or → jumps) or only under the perturbation family it was trained
against. Those are different claims, and the paper reports the former while demonstrating
the latter. Distinguishing them is a real contribution — and is precisely the kind of
question only a common protocol can answer.

Read the paper before implementing. The docstrings below deliberately do not restate its
algorithm; deriving it from the source is part of the work, and paraphrasing it here would
risk baking in a misreading.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow import keras

__all__ = ["RobustAgent"]


class RobustAgent(keras.Model):
    """Adversarially trained hedging policy.

    Wraps a base agent (feedforward, recurrent, or band) and changes only the *training
    procedure*, not the architecture. Keeping that separation explicit matters: it means
    any robustness gain is attributable to the training, not to a different network.

    Args:
        base_agent: the policy being made robust. Composition, not inheritance — so
            "adversarial training" is an axis of the grid rather than a fourth architecture.
        epsilon: Wasserstein ball radius. The key hyperparameter; ``0`` recovers standard
            training exactly, which is a useful degenerate check that the wrapper is inert
            when it should be.
        n_inner_steps: ascent steps for the inner maximisation per outer update.
    """

    def __init__(
        self,
        base_agent: keras.Model,
        epsilon: float = 0.05,
        n_inner_steps: int = 1,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.base_agent = base_agent
        self.epsilon = epsilon
        self.n_inner_steps = n_inner_steps

    def hedge_path(self, spot: tf.Tensor, maturity: float) -> tf.Tensor:
        """Delegate to the base agent. Robustness lives in training, not inference.

        Args:
            spot: ``(n_paths, n_steps + 1)`` simulated prices.
            maturity: ``T`` in years.

        Returns:
            ``(n_paths, n_steps)`` hedge positions.
        """
        return self.base_agent.hedge_path(spot, maturity)

    def adversarial_paths(
        self,
        spot: tf.Tensor,
        payoff: tf.Tensor,
        objective,
        maturity: float,
    ) -> tf.Tensor:
        """Solve the inner maximisation: perturb the path distribution adversarially.

        Args:
            spot: ``(n_paths, n_steps + 1)`` nominal simulated prices.
            payoff: ``(n_paths,)`` claim payoff.
            objective: the risk measure being maximised over the ball.
            maturity: ``T`` in years.

        Returns:
            ``(n_paths, n_steps + 1)`` perturbed paths, within the ``epsilon`` ball.

        Note:
            Follow the paper's tractable reformulation rather than inventing one. Two
            things to verify explicitly, because both fail silently:

            - the perturbation genuinely respects the ball constraint (measure it);
            - ``epsilon = 0`` reproduces standard training **bit-for-bit**. If it does not,
              the wrapper has a bug and every robustness number is contaminated.
        """
        raise NotImplementedError

    def train_step_adversarial(
        self,
        spot: tf.Tensor,
        payoff: tf.Tensor,
        objective,
        optimizer: keras.optimizers.Optimizer,
        maturity: float,
        cost_rate: float,
        premium: float,
    ) -> dict[str, tf.Tensor]:
        """One outer minimisation step over adversarially perturbed paths.

        Args:
            spot: ``(n_paths, n_steps + 1)`` nominal prices.
            payoff: ``(n_paths,)`` claim payoff.
            objective: risk measure.
            optimizer: Keras optimizer.
            maturity: ``T`` in years.
            cost_rate: proportional transaction cost.
            premium: ``p_0``.

        Returns:
            Metrics for logging — at minimum the nominal loss, the adversarial loss, and
            the realised perturbation size. **Log all three.** The gap between nominal and
            adversarial loss is the diagnostic that tells you the inner problem is actually
            doing something; if they track each other exactly, the attack is inert.
        """
        raise NotImplementedError
