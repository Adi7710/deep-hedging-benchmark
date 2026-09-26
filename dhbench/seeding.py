"""Seed derivation -- the reproducibility layer's foundation.

``tf.random.Generator.from_seed(k)`` for **small consecutive integers** does not yield
independent streams. It yields *the same stream, shifted*.

**The mechanism.** The integer seed is written into the Philox *counter*, and the key is
left at zero for every seed::

    from_seed(0).state == [0, 0, 0]
    from_seed(1).state == [1, 0, 0]
    from_seed(7).state == [7, 0, 0]

Each Philox block yields four outputs, so ``from_seed(k)`` reproduces ``from_seed(0)``
offset by ``4k`` draws. At 200,000 draws, 100% of ``from_seed(1)``'s draws are
``from_seed(0)``'s draws. A lag-0 correlation check reports +0.005 -- apparently
independent -- which is why this passes every naive test. TensorFlow documents no
guarantee here: "two different seeds are *likely* to produce two independent generators
(but no guarantee)". The general defect class -- related seeds producing correlated
streams through careless initialisation -- is long established (Matsumoto et al., ACM
TOMACS 2007). This is one concrete instance of it.

**The consequence is false significance, not bias.** Pricing an ATM call (truth 7.96557)
with 200k paths under seeds 0..19, each replicate's paths share 46 of 50 increments with
the next (path-level correlation 0.92). The mean of the 20 replicates, 7.99070, sits
+0.85 *proper* standard errors above truth -- an ordinary sampling fluctuation. But the
cross-replicate dispersion collapses to 0.41x the true sampling error, so treating the
replicates as independent reports that same fluctuation as **+9.3 standard errors**.

An earlier version of this docstring called the +0.025 a "bias" propagated through a
"systematic offset" in realised volatility. That was the right numbers with the wrong
mechanism: nothing is biased. Twenty pseudo-replicates of one sample were counted as
twenty samples. Corrected 2026-09-26 after reading the generator state directly.

For a benchmark whose stated controls include "multiple seeds with dispersion reported",
this is the worst available failure mode: it turns noise into a significant result. So
replicate indices are hashed to well-separated seeds before reaching TensorFlow.

**Why hashing works.** It does not change the key -- hashed seeds also land in the counter,
also with key zero. It works because SHA-256 scatters the counters across a 2^63 range, so
two streams would overlap only if their counters fell within (draws / 4) of each other.
At any realistic draw count that probability is negligible. A stricter fix would derive
distinct *keys*; recorded here as a hardening option rather than a need.

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
