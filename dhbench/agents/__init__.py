"""Learned hedging policies.

Every agent exposes ``hedge_path(spot, maturity) -> (n_paths, n_steps)``. That single
method is the entire interface the training loop and the evaluation code need, which is
what lets the benchmark treat architectures as an interchangeable axis.

    feedforward  time-shared MLP           -- Buehler et al. 2019; the reference point
    recurrent    LSTM/GRU                  -- Carbonneau 2020
    band         no-transaction band prior -- Arzel & Lehdili 2026
    robust       adversarial training      -- He et al., NeurIPS 2025

``robust`` **wraps** one of the others rather than being a fourth architecture. That keeps
"adversarial training" an axis of the grid, so its effect is separable from the choice of
network.

**The rule that matters more than any of the above:** the roll-forward inside
``hedge_path`` must stay within the caller's ``tf.GradientTape``, and nothing in it may
detach the gradient. ``delta_prev`` feeding back into the next step is a genuine recurrence
through the action; ``tf.stop_gradient`` on it (or a NumPy round-trip mid-loop) silently
converts the policy into a myopic one-step rule that still trains, still converges, and is
wrong. This is the most expensive mistake available in this codebase.
"""
