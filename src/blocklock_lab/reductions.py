"""Shape-dependent split/tiled reductions.

This is the mechanism the BlockLock paper's argument rests on, made explicit
and measurable: GPU kernels split a reduction across tiles / thread blocks,
and the tile boundaries follow the *padded* tensor shape. When the padded
length changes (different companions), the same logical values are grouped
differently, and because bf16 addition is not associative the rounded result
changes — even though every individual value is identical.

A stock NumPy GEMM is row-independent, so it cannot show this effect; that
is why the reduction is implemented here directly. The model is:

* ``tiled_sum`` splits ``axis`` into consecutive chunks of ``tile``,
  sums each chunk in fp32, rounds the partial to bf16, then combines the
  partials left-to-right in bf16. The chunking is a function of the padded
  length, so two runs that differ only in padding regroup the same values.
* ``tiled_weighted_sum`` does the same for an attention-weighted sum
  ``sum_k p_k * V_k`` over the key axis.

With right padding the target's real keys always occupy the first ``Lx``
positions, so regrouping never happens and the drift is exactly zero — which
is why the lab (like the paper's Figure 3a, where the target's *padding
offset* shifts) uses left padding for the padded baseline. With left
padding the real keys occupy the last ``Lx`` positions and their chunk
membership shifts with the padded length.

In fp32 the same regrouping changes results only at ~1e-7, below any
flip threshold; in bf16 it changes them at ~1e-2 relative, which is what
flips near-tie logits. Both regimes are measured in the experiments.
"""

import numpy as np

from .bf16 import to_bf16


def tiled_sum(x, axis, tile, dtype="bf16"):
    """Sum along ``axis`` in consecutive chunks of ``tile``.

    Chunk partials are summed in fp32 and (for bf16) rounded to bf16 before
    being combined left-to-right in bf16. For fp32 the partials are combined
    in fp32. Returns an array with ``axis`` removed.
    """
    x = np.asarray(x, dtype=np.float32)
    n = x.shape[axis]
    moved = np.moveaxis(x, axis, -1)
    flat = moved.reshape(-1, n)
    out = np.zeros(flat.shape[0], dtype=np.float32)
    for start in range(0, n, tile):
        chunk = flat[:, start:start + tile]
        partial = chunk.sum(axis=1)  # fp32 pairwise sum inside the chunk
        if dtype == "bf16":
            partial = to_bf16(partial)
            out = to_bf16(out + partial)
        else:
            out = out + partial
    return out.reshape(moved.shape[:-1])


def tiled_weighted_sum(weights, values, tile, dtype="bf16"):
    """Attention-weighted sum ``sum_k weights[..., q, k] * values[..., k, d]``.

    ``weights`` is [..., Tq, Tk] and ``values`` is [..., Tk, D] with matching
    leading dims. The sum over Tk is split into consecutive chunks of
    ``tile``; each chunk is summed in fp32 (one matmul) and rounded to bf16
    before the chunks are combined left-to-right in bf16. The chunking is
    a function of the padded length Tk, so the rounding of the same logical
    values changes when the padding changes.
    """
    w = np.asarray(weights, dtype=np.float32)
    v = np.asarray(values, dtype=np.float32)
    tq, tk = w.shape[-2], w.shape[-1]
    lead = w.shape[:-2]
    w2 = w.reshape(-1, tq, tk)
    v2 = v.reshape(-1, tk, v.shape[-1])
    out = np.zeros((w2.shape[0], tq, v2.shape[-1]), dtype=np.float32)
    for start in range(0, tk, tile):
        stop = min(start + tile, tk)
        chunk = np.matmul(w2[:, :, start:stop], v2[:, start:stop, :])
        if dtype == "bf16":
            out = to_bf16(out + to_bf16(chunk))
        else:
            out = out + chunk
    return out.reshape(lead + (tq, v2.shape[-1]))


def chunk_boundaries(n, tile):
    """Chunk start indices for a length-``n`` axis split into ``tile`` chunks."""
    return list(range(0, n, tile))
