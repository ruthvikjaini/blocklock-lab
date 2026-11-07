"""A tiny causal transformer, forward-only, in NumPy.

One layer, two heads, d_model = 64, vocab = 128, learned absolute position
embeddings. All weights are random (fixed seed); the lab studies inference
numerics, not training. Every linear projection is computed in fp32 and
rounded to bf16 (in bf16 mode); the attention softmax denominator and the
attention-weighted sum use the shape-dependent tiled reductions from
``reductions`` — that is where batch geometry enters the numerics.

The model supports a KV cache: ``prefill`` processes a padded prompt and
returns the cache; ``decode_step`` appends one token per row and attends
over the full (padded) cache, so the decode-time reduction geometry follows
the same policy width as prefill. Position ids are logical (0-based per
request) by default; the position-shift extension passes
``positions="padded"`` to index positions by padded coordinate instead.
"""

import numpy as np

from .bf16 import bf16_add, bf16_matmul, to_bf16
from .reductions import tiled_sum, tiled_weighted_sum


def init_params(seed=0, d=64, heads=2, vocab=128, ff=128, max_len=256):
    """Random weights, fixed seed. All values are bf16-representable."""
    rng = np.random.default_rng(seed)
    scale = 1.0 / np.sqrt(d)

    def w(*shape):
        return to_bf16(rng.normal(0.0, scale, size=shape).astype(np.float32))

    params = {
        "d": d, "heads": heads, "vocab": vocab, "ff": ff, "max_len": max_len,
        "tok_emb": w(vocab, d),
        "pos_emb": w(max_len, d),
        "w_q": w(d, d), "w_k": w(d, d), "w_v": w(d, d), "w_o": w(d, d),
        "w1": w(d, ff), "w2": w(ff, d),
        "ln1_g": to_bf16(np.ones(d, dtype=np.float32)),
        "ln2_g": to_bf16(np.ones(d, dtype=np.float32)),
        "lnf_g": to_bf16(np.ones(d, dtype=np.float32)),
        "lm_head": w(d, vocab),
    }
    return params


def _layernorm(x, gain, dtype):
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    y = (x - mean) / np.sqrt(var + 1e-5)
    y = y * gain
    return to_bf16(y) if dtype == "bf16" else y.astype(np.float32)


def _linear(x, wmat, dtype):
    if dtype == "bf16":
        return bf16_matmul(x, wmat)
    return x @ wmat


def _attention(q, k, v, mask, tile, dtype):
    """Attention with tiled reductions over the key axis.

    q, k, v: [B, H, Tq, Dh] / [B, H, Tk, Dh]. mask: [B, Tq, Tk] bool
    (True = attend). Scores are fp32 -> bf16 per element; the softmax
    denominator and the weighted sum are tiled reductions over Tk, so their
    rounding depends on the padded length Tk.
    """
    dh = q.shape[-1]
    scores = np.matmul(q, np.swapaxes(k, -1, -2)) / np.sqrt(dh)
    if dtype == "bf16":
        scores = to_bf16(scores)
    with np.errstate(invalid="ignore"):
        scores = np.where(mask[:, None, :, :], scores, -np.inf)
        m = np.max(scores, axis=-1, keepdims=True)
        e = np.exp(scores - m)
    e = np.where(mask[:, None, :, :], e, 0.0)
    denom = tiled_sum(e, axis=-1, tile=tile, dtype=dtype)
    p = np.divide(e, denom[..., None], out=np.zeros_like(e),
                  where=denom[..., None] != 0)
    if dtype == "bf16":
        p = to_bf16(p)
    return tiled_weighted_sum(p, v, tile=tile, dtype=dtype)


def forward(params, tokens, mask, positions, tile=16, dtype="bf16",
            kv_cache=None):
    """Forward pass.

    tokens: [B, Tq] int ids of the *new* tokens (prefill: whole prompt;
    decode: one token per row). mask: [B, Tq, Tk] bool over the full key
    length Tk (cache + new tokens). positions: [B, Tq] int position ids.
    kv_cache: None, or {"k": [B, H, Tc, Dh], "v": [...]} from a previous call.

    Returns (logits [B, Tq, V], new_cache {"k","v"} covering all Tk keys).
    Logits are bf16-rounded in bf16 mode.
    """
    d = params["d"]
    h = params["heads"]
    dh = d // h
    b, tq = tokens.shape
    tk = mask.shape[-1]

    x = params["tok_emb"][tokens] + params["pos_emb"][positions]
    if dtype == "bf16":
        x = to_bf16(x)
    x = _layernorm(x, params["ln1_g"], dtype)
    q = _linear(x, params["w_q"], dtype).reshape(b, tq, h, dh).transpose(0, 2, 1, 3)
    k_new = _linear(x, params["w_k"], dtype).reshape(b, tq, h, dh).transpose(0, 2, 1, 3)
    v_new = _linear(x, params["w_v"], dtype).reshape(b, tq, h, dh).transpose(0, 2, 1, 3)
    if kv_cache is None:
        k_full = k_new
        v_full = v_new
    else:
        k_full = np.concatenate([kv_cache["k"], k_new], axis=2)
        v_full = np.concatenate([kv_cache["v"], v_new], axis=2)
    assert k_full.shape[2] == tk, (k_full.shape, tk)

    att = _attention(q, k_full, v_full, mask, tile, dtype)
    att = att.transpose(0, 2, 1, 3).reshape(b, tq, d)
    att = _linear(att, params["w_o"], dtype)
    x = bf16_add(x, att) if dtype == "bf16" else x + att
    h2 = _layernorm(x, params["ln2_g"], dtype)
    ff = _linear(h2, params["w1"], dtype)
    ff = np.maximum(ff, 0)  # ReLU; exact in both dtypes
    ff = _linear(ff, params["w2"], dtype)
    x = bf16_add(x, ff) if dtype == "bf16" else x + ff
    x = _layernorm(x, params["lnf_g"], dtype)
    logits = _linear(x, params["lm_head"], dtype)
    return logits, {"k": k_full, "v": v_full}


def causal_mask(lengths, total, segment=None):
    """Mask [B, Tq, Tk] for left-padded rows (prefill, Tq == Tk == total).

    lengths: [B] real lengths; the real tokens occupy the last ``L`` padded
    indices of each row. A query attends to keys in its own real span with
    key index <= query index, and only within its own segment when
    ``segment`` (an [B, Tk] int segment id per key position) is given.
    """
    b = len(lengths)
    idx = np.arange(total)
    real = idx[None, :] >= (total - np.asarray(lengths))[:, None]
    causal = idx[None, None, :] <= idx[None, :, None]
    mask = real[:, None, :] & causal & real[:, :, None]
    if segment is not None:
        seg = np.asarray(segment)
        mask = mask & (seg[:, None, :] == seg[:, :, None])
    return mask


def decode_mask(lengths, total):
    """Mask [B, 1, Tk] for one decode step.

    The new query is the last real token of its row, so it attends to all
    real keys: the last ``L`` padded indices of a cache of width ``total``.
    """
    idx = np.arange(total)
    real = idx[None, :] >= (total - np.asarray(lengths))[:, None]
    return real[:, None, :]
