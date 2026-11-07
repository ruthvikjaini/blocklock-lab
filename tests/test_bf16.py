"""bf16 emulation: rounding must be round-to-nearest-even, and every op
must return exactly bf16-representable values."""

import numpy as np

from blocklock_lab.bf16 import bf16_add, bf16_matmul, bf16_mul, is_bf16, to_bf16


def test_round_to_nearest_even():
    # bf16 has 7 explicit mantissa bits: spacing is 2^-7 near 1.0, so the
    # neighbours of 1.0 are 1.0 and 1.0078125, with midpoint 1 + 2^-8.
    assert float(to_bf16(np.float32(1.0))) == 1.0
    # 1 + 2^-9 is below the midpoint -> rounds to 1.0.
    assert float(to_bf16(np.float32(1.0 + 2.0 ** -9))) == 1.0
    # 1 + 3*2^-9 is above the midpoint -> rounds to 1 + 2^-7.
    assert float(to_bf16(np.float32(1.0 + 3.0 * 2.0 ** -9))) == 1.0 + 2.0 ** -7
    # Exactly at the midpoint, ties round to even (1.0, mantissa 0).
    assert float(to_bf16(np.float32(1.0 + 2.0 ** -8))) == 1.0
    # Just above the midpoint rounds up; just below rounds down.
    assert float(to_bf16(np.float32(1.0 + 2.0 ** -8 + 2.0 ** -20))) == 1.0 + 2.0 ** -7
    assert float(to_bf16(np.float32(1.0 + 2.0 ** -8 - 2.0 ** -20))) == 1.0
    # Negative values round symmetrically.
    assert float(to_bf16(np.float32(-1.0 - 2.0 ** -8))) == -1.0
    assert float(to_bf16(np.float32(-1.0 - 3.0 * 2.0 ** -9))) == -1.0 - 2.0 ** -7


def test_values_are_bf16():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(50, 50)).astype(np.float32)
    assert is_bf16(to_bf16(x))
    assert is_bf16(bf16_matmul(x, x))
    assert is_bf16(bf16_add(x, x))
    assert is_bf16(bf16_mul(x, x))


def test_matmul_matches_fp32_within_one_ulp():
    rng = np.random.default_rng(1)
    a = rng.normal(size=(8, 16)).astype(np.float32)
    b = rng.normal(size=(16, 8)).astype(np.float32)
    got = bf16_matmul(a, b)
    want = (a @ b).astype(np.float32)
    # bf16 relative error per element is at most 2^-8 (one ulp is 2^-8).
    rel = np.abs(got - want) / np.maximum(np.abs(want), 1e-6)
    assert np.all(rel < 2.0 ** -7)
