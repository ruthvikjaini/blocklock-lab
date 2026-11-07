"""Real bfloat16 arithmetic, emulated with NumPy.

Every value is stored as float32 whose low 16 mantissa bits are zero, i.e.
exactly a bf16 value. Operations are computed in float32 and rounded back
to bf16 with round-to-nearest-even, which is the standard model of bf16
tensor-core arithmetic (fp32 accumulate, bf16 store).

The rounding is done on the raw bit pattern: adding 0x7FFF plus the least
significant kept bit implements round-to-nearest-even on the truncated
mantissa. This is exact for finite values, which is all this lab uses.
"""

import numpy as np


def to_bf16(x):
    """Round float32 array to bf16 (round-to-nearest-even), returned as float32."""
    x = np.asarray(x, dtype=np.float32)
    u = x.view(np.uint32).astype(np.uint64)
    # 0x7FFF is half of the discarded 16-bit range; adding the lsb of the
    # kept half makes ties round to even.
    r = u + np.uint64(0x7FFF) + ((u >> np.uint64(16)) & np.uint64(1))
    r = (r & np.uint64(0xFFFF0000)).astype(np.uint32)
    return r.view(np.float32)


def is_bf16(x):
    """True if every value is exactly representable in bf16."""
    x = np.asarray(x, dtype=np.float32)
    return bool(np.all((x.view(np.uint32) & np.uint32(0xFFFF)) == 0))


def bf16_matmul(a, b):
    """Matrix product computed in fp32, result rounded to bf16."""
    return to_bf16(np.matmul(np.asarray(a, dtype=np.float32),
                             np.asarray(b, dtype=np.float32)))


def bf16_add(a, b):
    """Elementwise add computed in fp32, result rounded to bf16."""
    return to_bf16(np.asarray(a, dtype=np.float32) + np.asarray(b, dtype=np.float32))


def bf16_mul(a, b):
    """Elementwise multiply computed in fp32, result rounded to bf16."""
    return to_bf16(np.asarray(a, dtype=np.float32) * np.asarray(b, dtype=np.float32))
