"""Model: determinism, cache consistency, and padding invariance."""

import numpy as np

from blocklock_lab import model


def _params():
    return model.init_params(seed=0)


def test_forward_deterministic():
    params = _params()
    tokens = np.asarray([[3, 17, 42, 9, 100, 5]])
    mask = model.causal_mask([6], 6)
    pos = np.arange(6)[None, :]
    a, _ = model.forward(params, tokens, mask, pos, dtype="bf16")
    b, _ = model.forward(params, tokens, mask, pos, dtype="bf16")
    assert np.array_equal(a, b)


def test_prefill_then_decode_matches_direct_prefill():
    # B = 1: prefill [a, b] then decode c must equal a direct prefill of
    # [a, b, c] (same tile grouping, same positions).
    params = _params()
    prompt = np.asarray([[3, 17, 42]])
    mask = model.causal_mask([3], 3)
    pos = np.arange(3)[None, :]
    logits, cache = model.forward(params, prompt, mask, pos, dtype="bf16")
    assert logits.shape == (1, 3, 128)
    nxt = np.asarray([[99]])
    dmask = model.decode_mask([4], 4)
    dpos = np.asarray([[3]])
    dlogits, _ = model.forward(params, nxt, dmask, dpos, dtype="bf16",
                               kv_cache=cache)
    direct, _ = model.forward(params, np.asarray([[3, 17, 42, 99]]),
                              model.causal_mask([4], 4),
                              np.arange(4)[None, :], dtype="bf16")
    assert np.array_equal(dlogits[0, 0, :], direct[0, -1, :])


def test_left_padding_does_not_change_real_token_logits():
    # Same logical prompt, left-padded to different widths that stay within
    # one reduction tile: real-token logits must be identical.
    params = _params()
    prompt = [3, 17, 42, 9]
    for width in (4, 8, 16):
        tokens = np.asarray([[0] * (width - 4) + prompt])
        mask = model.causal_mask([4], width)
        pos = np.zeros((1, width), dtype=np.int64)
        pos[0, width - 4:] = np.arange(4)
        logits, _ = model.forward(params, tokens, mask, pos, dtype="bf16")
        if width == 4:
            ref = logits[0, -1, :].copy()
        else:
            assert np.array_equal(logits[0, -1, :], ref)


def test_fp32_forward_finite():
    params = _params()
    tokens = np.asarray([[3, 17, 42, 9, 100, 5, 77, 12]])
    mask = model.causal_mask([8], 8)
    pos = np.arange(8)[None, :]
    logits, _ = model.forward(params, tokens, mask, pos, dtype="fp32")
    assert np.all(np.isfinite(logits))
