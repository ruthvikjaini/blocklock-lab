"""Tiled reductions: the rounding must depend on the padded length (the
mechanism), and be exactly reproducible for a fixed padded length."""

import numpy as np

from blocklock_lab.reductions import chunk_boundaries, tiled_sum, tiled_weighted_sum


def test_chunk_boundaries():
    assert chunk_boundaries(16, 16) == [0]
    assert chunk_boundaries(17, 16) == [0, 16]
    assert chunk_boundaries(24, 8) == [0, 8, 16]


def test_tiled_sum_matches_plain_sum_in_fp32():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(4, 24)).astype(np.float32)
    got = tiled_sum(x, axis=-1, tile=8, dtype="fp32")
    want = x.sum(axis=-1)
    assert np.allclose(got, want, rtol=1e-6, atol=1e-6)


def test_tiled_sum_shape_dependence_bf16():
    # Same logical values, left-padded to different lengths whose offsets
    # mod tile differ (16 vs 20 with tile 8): the tile grouping of the real
    # values shifts, so bf16 rounding must change for at least some inputs.
    # (Padding to 16 vs 24 would keep the same grouping — 10 values split
    # 2+8 in both cases — and show no difference.)
    rng = np.random.default_rng(2)
    changed = 0
    trials = 0
    for _ in range(200):
        vals = rng.normal(size=10).astype(np.float32)
        a = np.concatenate([np.zeros(6, dtype=np.float32), vals])   # len 16
        b = np.concatenate([np.zeros(10, dtype=np.float32), vals])  # len 20
        sa = tiled_sum(a, axis=-1, tile=8, dtype="bf16")
        sb = tiled_sum(b, axis=-1, tile=8, dtype="bf16")
        trials += 1
        if float(sa) != float(sb):
            changed += 1
    # The effect is real but not universal: regrouping only changes the
    # rounding when a partial sum crosses a rounding boundary.
    assert changed > 0, "no bf16 regrouping difference observed in 200 trials"
    assert changed < trials, "every trial differed; suspicious"


def test_tiled_sum_repeat_stable():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(3, 24)).astype(np.float32)
    assert np.array_equal(tiled_sum(x, axis=-1, tile=8, dtype="bf16"),
                          tiled_sum(x, axis=-1, tile=8, dtype="bf16"))


def test_tiled_weighted_sum_matches_matmul():
    rng = np.random.default_rng(4)
    w = rng.normal(size=(2, 3, 5, 12)).astype(np.float32)
    v = rng.normal(size=(2, 3, 12, 7)).astype(np.float32)
    got = tiled_weighted_sum(w, v, tile=4, dtype="fp32")
    want = np.matmul(w, v)
    assert got.shape == want.shape
    assert np.allclose(got, want, rtol=1e-5, atol=1e-5)
