"""Regime-shift stress testing — the machinery behind §8.

Train in one world, evaluate in another. This is the experiment that turns the benchmark
from a leaderboard into a finding, and it is the fallback standalone paper if the
benchmark angle gets scooped.

The motivating result is from *What Does Deep Hedging Actually Learn?*
([arXiv:2605.21696](https://arxiv.org/pdf/2605.21696)): learned policies exhibit **regime
fragility**, failing when conditions shift beyond the training distribution. That paper
shows it for one model. §8 measures it **across the whole method set** under a common
protocol — which is what makes it a benchmark result rather than a replication, and lets it
answer the question the original cannot: *does any method's advantage survive?*

The shift matrix:

                        test on ->
                        GBM    Heston   regime-switch   jumps
    train on  GBM        .        x           x           x
              Heston     x       (.)          x           x

Diagonal cells are the in-distribution reference. Off-diagonal cells are the result.

Two distinct kinds of shift, deliberately:

- **Heston -> regime-switching** shifts the *volatility level*, gradually and persistently.
  A well-generalising policy might cope.
- **Heston -> jumps** breaks the *continuity assumption* that delta hedging rests on. No
  continuous rebalancing strategy can hedge a jump, so everything should do badly.

**Expect broad failure in the jump column. That is a result, not a bug.** The interesting
question is not who wins but whether the *ranking* is preserved — if the ordering of
methods reverses under shift, then in-distribution benchmarks (i.e. the entire existing
literature) are not predictive of robustness, which is a strong claim and a publishable one.
"""

from __future__ import annotations

from typing import Any, Callable

__all__ = ["run_shift_matrix", "evaluate_on_world"]


def evaluate_on_world(
    agent: Any,
    world_fn: Callable[..., Any],
    world_params: dict[str, Any],
    eval_config: dict[str, Any],
    seed: int,
) -> dict[str, float]:
    """Evaluate a trained agent on paths from a given world.

    Args:
        agent: a trained agent exposing ``hedge_path``. Also accepts a baseline wrapped to
            the same interface, so classical strategies enter the same table.
        world_fn: simulator from :mod:`dhbench.worlds`.
        world_params: parameters for that simulator.
        eval_config: strike, maturity, cost rate, risk measure, path count.
        seed: **evaluation** seed. Must be disjoint from every training seed.

    Returns:
        The metric dict from :func:`dhbench.evaluation.metrics.summarise`.

    Warning:
        No training happens here. If the agent has any state that adapts at evaluation
        time, that is a different experiment and must be labelled as one.
    """
    raise NotImplementedError


def run_shift_matrix(
    agents: dict[str, Any],
    train_world: str,
    test_worlds: dict[str, dict[str, Any]],
    eval_config: dict[str, Any],
    seeds: list[int],
) -> dict[str, dict[str, dict[str, float]]]:
    """Evaluate every agent on every test world. The §8 results table.

    Args:
        agents: name -> trained agent. Include the classical baselines: whether
            Whalley-Wilmott degrades *less* than every learned method is one of the most
            interesting things this table can show, and it is a real possibility.
        train_world: name of the world the agents were trained on, for labelling the
            in-distribution reference column.
        test_worlds: name -> world parameters.
        eval_config: shared evaluation settings, identical across all cells.
        seeds: evaluation seeds. **At least 5** — seed variance in deep hedging is large
            enough to swamp genuine differences, and a single-seed shift matrix is not
            evidence of anything.

    Returns:
        Nested ``{agent: {world: {metric: value}}}``, with mean and std across seeds.

    Note:
        Every cell uses **identical evaluation paths** across agents. Same world, same
        seed, same paths — otherwise you are comparing agents *and* path draws at once, and
        the comparison is worthless.

        Persist raw per-seed results, not just the aggregates. Reviewers ask for error
        bars, and regenerating a full matrix to answer that is expensive.
    """
    raise NotImplementedError
