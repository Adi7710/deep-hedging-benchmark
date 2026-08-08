"""Shared pytest configuration and fixtures.

Also makes the repo root importable so ``pytest`` works from a bare checkout without an
editable install. Once the package is installed with ``pip install -e .`` this becomes
redundant but stays harmless.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: requires model training; excluded by default"
    )


# Reference parameters shared across the ladder. Kept in one place so a change here
# propagates to every test rather than drifting between files.
REFERENCE = dict(
    s0=100.0,
    strike=100.0,
    maturity=1.0,
    rate=0.0,
    sigma=0.2,
)


@pytest.fixture(scope="session")
def reference_params() -> dict[str, float]:
    """ATM 1-year call, 20% vol, zero rate. Black-Scholes price 7.9656."""
    return dict(REFERENCE)


@pytest.fixture(scope="session")
def bs_reference_price() -> float:
    """The analytic price for :func:`reference_params`."""
    return 7.9656


@pytest.fixture
def generator():
    """A seeded ``tf.random.Generator``.

    Imported lazily so that collection of the non-TensorFlow tests stays fast, and so a
    broken TensorFlow install produces a clear error inside the tests that need it rather
    than at collection time.
    """
    import tensorflow as tf

    return tf.random.Generator.from_seed(20260808)
