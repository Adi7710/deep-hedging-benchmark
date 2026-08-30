"""Seed derivation -- the reproducibility layer's foundation.

``tf.random.Generator.from_seed(k)`` for **small consecutive integers** does not yield
independent streams. Measured on this project: pricing an ATM call (true value 7.96557)
with 200k paths across seeds 0..19 gave mean 7.99070, a +0.025 bias at 9.3 standard
errors with 20/20 replicates above truth, and a cross-seed dispersion of 0.012 against a
theoretical standard error of 0.0296 -- error bars 2.5x too narrow.

The mechanism is visible in the raw draws. Sample means of 200k normals from seeds 0-5:

    [-0.00204, -0.00204, -0.00204, -0.00206, -0.00204, -0.00204]

identical to five decimals, where independent streams should scatter with sd 0.00224.
That systematic offset propagates into realised volatility (sd(log S_T) ~ 0.2004 rather
than 0.2000), and an ATM call's vega turns +0.0005 of sigma into +0.02 of price.

For a benchmark whose stated controls include "multiple seeds with dispersion reported",
2.5x-narrow error bars would make inconclusive comparisons read as significant. So
replicate indices are hashed to well-separated seeds before reaching TensorFlow.

**SHA-256, not** ``hash()``. Python's built-in hash is salted per process unless
PYTHONHASHSEED is pinned, which would break bit-reproducibility across runs -- the exact
claim this module exists to protect.
"""

from __future__ import annotations

import hashlib

import tensorflow as tf

__all__ = ["derive_seed", "make_generator", "seed_keras"]

_MASK63 = (1 << 63) - 1


def derive_seed(seed: int, stream: str = "", bits: int = 63) -> int:
    """Map a small replicate index to a well-separated seed.

    seed:   the replicate index as written in a config, e.g. 0, 1, 2.
    stream: a label separating independent uses of the same replicate --
            ``"train"``, ``"eval"``, ``"init"``. Different labels give
            independent streams, which is what makes train/eval splits disjoint
            by construction rather than by an ad-hoc numeric offset.
    bits:   width of the returned seed. 63 suits ``tf.random.Generator``; use 32
            for anything routed through NumPy, which rejects seeds >= 2**32.
            The digest is the same either way, so narrowing does not change which
            stream you are on, only how much of it is used.

    Deterministic across processes, platforms and Python versions.
    """
    if not 1 <= bits <= 64:
        raise ValueError(f"bits must be in [1, 64], got {bits}")
    payload = f"dhbench/{stream}/{seed}".encode("utf-8")
    digest = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return digest & ((1 << bits) - 1)


def make_generator(seed: int, stream: str = "") -> tf.random.Generator:
    """A ``tf.random.Generator`` seeded via :func:`derive_seed`.

    Use this everywhere instead of ``tf.random.Generator.from_seed`` directly. Passing a
    raw replicate index to TensorFlow is the bug this module exists to prevent.
    """
    return tf.random.Generator.from_seed(derive_seed(seed, stream))


def seed_keras(seed: int, stream: str = "init") -> int:
    """Seed Keras/NumPy/Python global state for reproducible weight initialisation.

    Returns the 32-bit seed actually used, so a run can record it.

    Weight initialisation is the one place the benchmark cannot avoid global random
    state: Keras layers draw their initialisers from it rather than from an injected
    generator. Everything else in the project takes an explicit ``tf.random.Generator``.

    Narrowed to 32 bits because ``keras.utils.set_random_seed`` forwards to
    ``numpy.random.seed``, which rejects anything >= 2**32. Discovered by trying it --
    ``derive_seed`` returns 63 bits and raises ValueError there.
    """
    import keras

    seed32 = derive_seed(seed, stream, bits=32)
    keras.utils.set_random_seed(seed32)
    return seed32
