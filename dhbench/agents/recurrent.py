"""Recurrent hedging agent — the Carbonneau (2020) variant.

An LSTM or GRU cell carries hidden state across timesteps, so the policy can in principle
condition on the whole price history rather than just the current state:

    h_i, delta_i = RNN( [t_i, S_i, delta_{i-1}], h_{i-1} )

Stage 3. See the `Keras RNN guide <https://www.tensorflow.org/guide/keras/working_with_rnns>`_
for cell-level APIs — note we step a ``Cell`` manually rather than using the ``Layer``,
because ``delta_prev`` feeds the next input.

**The honest question this variant exists to answer.** Under GBM and Heston the state is
Markov — everything relevant is in ``(t, S, v, delta_prev)`` — so a recurrent policy has
*no theoretical advantage*. If it wins anyway, the interesting explanations are:

- it is implicitly estimating realised volatility from the path (genuinely useful under
  Heston, where ``v`` may not be observable);
- it has more parameters and the comparison is not capacity-controlled;
- the extra state acts as a regulariser or an optimisation aid.

Distinguishing these is worth a paragraph in §6. **Control the parameter count** —
otherwise "the LSTM won" is an uninformative result and reviewers will say so.

Where it should genuinely help is the regime-switching world of §8: there the state is
*not* Markov in ``(t, S)``, because the hidden regime matters and the agent cannot observe
it. Inferring it from recent path behaviour is exactly what recurrence is for. That is the
sharpest prediction this benchmark can make about when recurrence pays.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow import keras

__all__ = ["RecurrentAgent"]


class RecurrentAgent(keras.Model):
    """LSTM/GRU hedging policy.

    Args:
        cell_type: ``"lstm"`` or ``"gru"``. GRU has fewer parameters, which makes
            capacity-matching against the feedforward agent easier.
        units: hidden units in the recurrent cell.
        dense_units: optional Dense stack between the cell output and the position.

    Note:
        Match the parameter count to :class:`~dhbench.agents.feedforward.FeedforwardAgent`
        before drawing any conclusion. ``model.count_params()`` after building; record both
        numbers in the results table so a reader can check.
    """

    def __init__(
        self,
        cell_type: str = "lstm",
        units: int = 32,
        dense_units: tuple[int, ...] = (),
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if cell_type not in ("lstm", "gru"):
            raise ValueError(f"cell_type must be 'lstm' or 'gru', got {cell_type}")
        self.cell_type = cell_type
        self.units = units
        self.dense_units = dense_units
        raise NotImplementedError(
            "Build an LSTMCell or GRUCell (the *Cell*, not the Layer -- we step it "
            "manually because delta_prev feeds back into the input), plus the Dense head."
        )

    def hedge_path(self, spot: tf.Tensor, maturity: float) -> tf.Tensor:
        """Roll the recurrent policy forward along whole paths.

        Args:
            spot: ``(n_paths, n_steps + 1)`` simulated prices.
            maturity: ``T`` in years.

        Returns:
            ``(n_paths, n_steps)`` hedge positions.

        Note:
            Use ``keras.layers.LSTMCell`` and step it yourself rather than
            ``keras.layers.LSTM`` over a pre-built sequence. The input at step ``i``
            includes ``delta_{i-1}``, which is the *previous output* — a genuine recurrence
            through the action, not just through the hidden state, so the sequence cannot
            be assembled in advance.

            Initialise the hidden state with ``cell.get_initial_state``.

        Warning:
            Same as the feedforward agent: the entire roll stays inside the
            ``GradientTape``, and nothing in the loop may detach the gradient. With an RNN
            there is more state to accidentally detach, and the failure is silent.
        """
        raise NotImplementedError
