"""Experiment runner — the entry point every published number comes from.

    python -m experiments.run --config configs/gbm_zerocost_entropic.yaml --seed 0

**This is the reproducibility layer**, and it is what rung 6 of the correctness ladder
tests. The contract:

    same config + same seed  ->  bit-identical results

Not "approximately equal" — identical. That is a headline claim of the paper, and it only
holds if *every* source of randomness is threaded explicitly from the seed. The usual
leaks, in order of how often they bite:

1. Global random state instead of an explicit ``tf.random.Generator``
2. Weight initialisation not seeded
3. Shuffling / batching order
4. Non-deterministic GPU kernels — ``tf.config.experimental.enable_op_determinism()``

There is a corollary that is easy to forget: **no hand-tuned one-off runs may appear in a
results table.** If a number is in the paper, it traces to a committed config file plus a
seed. Anything else silently reintroduces the problem this benchmark exists to solve.

Stage 4 work. Build it after the vanilla agent passes rung 4 — a runner around a training
loop you don't yet trust is premature.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

__all__ = ["load_config", "build_from_config", "run_experiment", "main"]


def load_config(path: str | Path) -> dict[str, Any]:
    """Load and validate an experiment config.

    Args:
        path: path to a YAML file in ``configs/``.

    Returns:
        The parsed config.

    Note:
        **Validate rather than trust.** A typo'd key that silently defaults is worse than a
        crash: it produces a plausible number under the wrong settings, and nothing in the
        results will look wrong. Fail loudly on unknown keys.
    """
    raise NotImplementedError


def build_from_config(config: dict[str, Any], seed: int) -> dict[str, Any]:
    """Instantiate world, cost model, objective, and agent from a config.

    Args:
        config: parsed config.
        seed: master seed. Every generator in the experiment derives from this one —
            training paths, weight init, and (offset by ``evaluation.seed_offset``)
            evaluation paths.

    Returns:
        The constructed components, keyed by role.

    Note:
        Derive sub-seeds deterministically from the master seed, e.g.
        ``train_seed = seed``, ``init_seed = seed + 1``, ``eval_seed = seed +
        config["evaluation"]["seed_offset"]``. Deterministic derivation is what keeps the
        train/eval seed sets provably disjoint, which rung 6 checks.
    """
    raise NotImplementedError


def run_experiment(config: dict[str, Any], seed: int) -> dict[str, float]:
    """Train, evaluate, and return the metric set for one cell of the grid.

    Args:
        config: parsed config.
        seed: master seed.

    Returns:
        The metric dict from :func:`dhbench.evaluation.metrics.summarise`.

    Note:
        Write results to ``experiments/runs/<config_name>/seed_<n>.json`` — gitignored, but
        keep the **raw per-seed** numbers, not just aggregates. Reviewers ask for error
        bars, and regenerating a full grid to answer that is expensive.

        Record the environment alongside: TensorFlow version, NumPy version, and whether it
        ran on CPU or GPU. Numerical results can shift between them, and a reader trying to
        reproduce needs to know which they are comparing against.
    """
    raise NotImplementedError


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", required=True, help="path to a YAML config in configs/")
    parser.add_argument("--seed", type=int, default=0, help="master seed")
    args = parser.parse_args()

    config = load_config(args.config)
    results = run_experiment(config, args.seed)
    for key, value in results.items():
        print(f"{key:24s} {value:.6f}")


if __name__ == "__main__":
    main()
