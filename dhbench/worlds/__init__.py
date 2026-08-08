"""Market simulators.

Every simulator returns ``(n_paths, n_steps + 1)`` price paths whose first column is
exactly ``s0``, and every one takes an explicit ``tf.random.Generator``. Global random
state is never used — bit-reproducibility from config plus seed is a headline claim of the
benchmark, and global state silently breaks it.

Worlds, in implementation order:

    gbm               base case; ground truth is the Black-Scholes delta
    heston            stochastic volatility; the realistic training world
    regime_switching  the *test* world for the regime-fragility result (§8)
    jumps             Merton jump-diffusion; a second out-of-distribution test
"""
