# DRAFT — TensorFlow GitHub issue (NOT filed)

Filing is the author's decision. Considerations before filing: it is public and under the
author's name (relevant to double-blind review in 2027); TensorFlow may reasonably answer
that seed-to-state conversion is documented as unspecified; checked 2026-09-26 that no
existing issue reports this (searches listed in `docs/07-literature-audit.md`).

---

**Title:** `tf.random.Generator.from_seed(int)`: consecutive integer seeds give shifted copies
of one Philox stream — the left-padding guard in `_make_1d_state` never runs for Python ints

**System information**
- TensorFlow 2.21.0 (CPU), Python 3.12, Windows 11. Same code on current `master`.

**Describe the current behavior**

`tf.random.Generator.from_seed(k)` with a Python `int` writes `k` into the Philox **counter**
and leaves the **key** at zero for every seed:

```python
>>> tf.random.Generator.from_seed(1).state.numpy()
array([1, 0, 0])
>>> tf.random.Generator.from_seed([1]).state.numpy()   # list seed: k goes to the KEY
array([0, 0, 1])
```

Because Philox produces four 32-bit outputs per counter increment, the stream for seed `k`
is the stream for seed `0` with its first `4k` draws removed. Seeds `0, 1, 2, ...` — the
natural choice for replicate experiments — therefore share almost every draw.

`_make_1d_state` in `tensorflow/python/ops/stateful_random_ops.py` already anticipates this:

> Padding with zeros on the *left* if too short. Padding on the right would cause a small
> seed to be used as the "counter" while the "key" is always zero (for counter-based RNG
> algorithms), because in the current memory layout counter is stored before key. In such a
> situation two RNGs with two different small seeds may generate overlapping outputs.

But for a Python `int` the preceding branch chops the integer into `state_size` words
(low word first), so the seed already has full length, `padding_size == 0`, and the guard
never runs. The int path produces exactly the layout the comment was written to prevent;
list and tensor seeds do not.

**Minimal reproduction**

```python
import numpy as np, tensorflow as tf

a = tf.random.Generator.from_seed(0).normal((200_000,)).numpy()
b = tf.random.Generator.from_seed(1).normal((200_000,)).numpy()
print(np.array_equal(a[4:], b[:-4]))          # True: seed 1 == seed 0 shifted by 4 draws
print(np.corrcoef(a, b)[0, 1])                 # 0.0051: a lag-0 check sees "independence"

c = tf.random.Generator.from_seed([0]).normal((100_000,)).numpy()
d = tf.random.Generator.from_seed([1]).normal((100_000,)).numpy()
print(any(np.array_equal(c[k:k + 1000], d[:1000]) for k in range(64)))   # False
```

**Why it matters**

In a Monte Carlo study with 20 replicates seeded `0..19` (200,000 paths each, an at-the-money
option price), the replicates' dispersion is 0.41× the true standard error, so a deviation of
+0.85 standard errors from the closed form is reported as +9.3 standard errors of the
replicate mean. Every replicate reproduces bit for bit and the lag-0 correlation between
streams is ~0.005, so routine checks do not reveal it.

**Describe the expected behavior**

Distinct small integer seeds should not give overlapping streams — as the comment in
`_make_1d_state` intends — or the `from_seed` docstring should say plainly that they do.

**Possible fixes** (each has costs; offered for discussion)

1. Route `int` seeds through the same left-padding as list seeds, so a small int lands in the
   key. Changes the stream produced by every existing integer seed — a reproducibility break
   for existing users.
2. Hash integer seeds (as NumPy's `SeedSequence` does before keying Philox). Same
   compatibility cost.
3. At minimum, document in the `from_seed` docstring that consecutive integer seeds produce
   overlapping streams, and recommend `from_seed([k])` or `Generator.split` for independent
   replicates. The guide's note on overlapping streams is general; this case is
   deterministic and hits the most common usage.

We recognise the docstring says the seed-to-state conversion is unspecified and gives no
guarantee of independence, and the guide recommends `split`. The report is that the most
natural call bypasses a guard the source already contains.
