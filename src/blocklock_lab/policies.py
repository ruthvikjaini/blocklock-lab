"""Batching policies: Padded, BlockLock-Rows, BlockLock-Pack, CoRun-style.

All policies share the same tiny transformer and the same tiled bf16
reductions; only the *geometry* differs. That is the whole point: the
numerical path of the protected (target) request is a function of the
geometry it sees.

Conventions: prompts are left-padded (the paper's Figure 3a: the target's
padding offset shifts with its companions). Position ids are logical
(0-based within a request) unless ``positions="padded"`` is requested, in
which case they are the padded coordinate — the position-shift audit of
extension E3.

* Padded: one tensor [B, T_max], T_max = longest request in the batch.
  The target's padding offset — and therefore the tile grouping of its
  real keys — shifts with the companions.
* Rows: one tensor [B, W] where W is the deployment's fixed class width
  (a multiple of ``block``). The target keeps the same row and the same
  logical positions whatever the companions are.
* Pack: one tensor [1, P] with P = B*W. Segments in order: companions,
  an isolated dummy segment absorbing the slack, then the target anchored
  at the terminal interval [P - Lx, P). A segment-causal mask isolates the
  segments. After prefill the target's KV (last Lx entries) is extracted
  and decode continues target-only.
* CoRun: isolated prefill, one request at a time (B = 1), and decode in a
  fixed envelope of ``envelope`` slots. The target's geometry never
  depends on the companions; the price is serial prefill and the fixed
  decode envelope.
"""

import time

import numpy as np

from .model import causal_mask, decode_mask, forward


def class_width(length, block=8):
    """Smallest multiple of ``block`` that fits ``length``."""
    return ((length + block - 1) // block) * block


def _left_pad(seqs, width, pad_id=0):
    rows = []
    for s in seqs:
        s = list(s)
        rows.append([pad_id] * (width - len(s)) + s)
    return np.asarray(rows, dtype=np.int64)


def _logical_positions(lengths, width):
    pos = np.zeros((len(lengths), width), dtype=np.int64)
    for i, L in enumerate(lengths):
        pos[i, width - L:] = np.arange(L)
    return pos


def _padded_positions(width, batch):
    return np.tile(np.arange(width, dtype=np.int64), (batch, 1))


class PrefillSpec:
    """Everything one prefill call needs, plus cost bookkeeping."""

    def __init__(self, tokens, positions, mask, lengths, width, target_row,
                 target_last_index, policy, prefill_positions,
                 prefill_attn_work, decode_width_fn, extra=None):
        self.tokens = tokens
        self.positions = positions
        self.mask = mask
        self.lengths = lengths
        self.width = width
        self.target_row = target_row
        self.target_last_index = target_last_index
        self.policy = policy
        self.prefill_positions = prefill_positions
        self.prefill_attn_work = prefill_attn_work
        self.decode_width_fn = decode_width_fn  # t -> cache width at decode step t
        self.extra = extra or {}


def build_padded(target, companions, pad_id=0, positions="logical"):
    seqs = [list(target)] + [list(c) for c in companions]
    lengths = [len(s) for s in seqs]
    width = max(lengths)
    tokens = _left_pad(seqs, width, pad_id)
    pos = (_logical_positions(lengths, width) if positions == "logical"
           else _padded_positions(width, len(seqs)))
    mask = causal_mask(lengths, width)
    return PrefillSpec(
        tokens, pos, mask, lengths, width, target_row=0,
        target_last_index=width - 1, policy="padded",
        prefill_positions=len(seqs) * width,
        prefill_attn_work=len(seqs) * width * width,
        decode_width_fn=lambda t: width + t,
    )


def build_rows(target, companions, width, pad_id=0, positions="logical"):
    seqs = [list(target)] + [list(c) for c in companions]
    lengths = [len(s) for s in seqs]
    assert all(L <= width for L in lengths), "companion exceeds class width"
    tokens = _left_pad(seqs, width, pad_id)
    pos = (_logical_positions(lengths, width) if positions == "logical"
           else _padded_positions(width, len(seqs)))
    mask = causal_mask(lengths, width)
    return PrefillSpec(
        tokens, pos, mask, lengths, width, target_row=0,
        target_last_index=width - 1, policy="rows",
        prefill_positions=len(seqs) * width,
        prefill_attn_work=len(seqs) * width * width,
        decode_width_fn=lambda t: width + t,
    )


def build_pack(target, companions, width, pad_id=0, positions="logical"):
    seqs = [list(c) for c in companions] + [list(target)]
    lengths = [len(s) for s in seqs]
    assert all(L <= width for L in lengths), "request exceeds class width"
    n_req = len(seqs)
    total_width = n_req * width
    used = sum(lengths)
    dummy_len = total_width - used
    assert dummy_len >= 0
    # Segments in order: companions, dummy, target (anchored at the end).
    seg_tokens, seg_pos, seg_id = [], [], []
    seg = 0
    for s in seqs[:-1]:
        seg_tokens.extend(s)
        seg_pos.extend(range(len(s)))
        seg_id.extend([seg] * len(s))
        seg += 1
    seg_tokens.extend([pad_id] * dummy_len)
    seg_pos.extend(range(dummy_len))
    seg_id.extend([seg] * dummy_len)
    seg += 1
    seg_tokens.extend(seqs[-1])
    seg_pos.extend(range(len(seqs[-1])))
    seg_id.extend([seg] * len(seqs[-1]))
    target_len = len(target)
    tokens = np.asarray([seg_tokens], dtype=np.int64)
    if positions == "logical":
        pos = np.asarray([seg_pos], dtype=np.int64)
    else:
        pos = np.arange(total_width, dtype=np.int64)[None, :]
    mask = causal_mask([total_width], total_width,
                       segment=np.asarray([seg_id], dtype=np.int64))
    return PrefillSpec(
        tokens, pos, mask, [total_width], total_width, target_row=0,
        target_last_index=total_width - 1, policy="pack",
        prefill_positions=total_width,
        prefill_attn_work=total_width * total_width,
        decode_width_fn=lambda t: target_len + t,
        extra={"target_len": target_len, "dummy_len": dummy_len,
               "segment": np.asarray([seg_id], dtype=np.int64)},
    )


def build_corun(target, companions, envelope=8, positions="logical"):
    # Isolated prefill: the target alone, B = 1. Companions are prefetched
    # separately; only their cost is counted, not their numerics.
    lengths = [len(target)]
    tokens = np.asarray([list(target)], dtype=np.int64)
    pos = (np.arange(len(target), dtype=np.int64)[None, :] if positions == "logical"
           else np.arange(len(target), dtype=np.int64)[None, :])
    mask = causal_mask(lengths, len(target))
    comp_lens = [len(c) for c in companions]
    return PrefillSpec(
        tokens, pos, mask, lengths, len(target), target_row=0,
        target_last_index=len(target) - 1, policy="corun",
        prefill_positions=len(target) + sum(comp_lens),
        prefill_attn_work=len(target) ** 2 + sum(L * L for L in comp_lens),
        decode_width_fn=lambda t: len(target) + t,
        extra={"envelope": envelope, "target_len": len(target)},
    )


BUILDERS = {
    "padded": lambda t, c, **kw: build_padded(t, c, **kw),
    "rows": lambda t, c, **kw: build_rows(t, c, width=kw.pop("width"), **kw),
    "pack": lambda t, c, **kw: build_pack(t, c, width=kw.pop("width"), **kw),
    "corun": lambda t, c, **kw: build_corun(t, c, **kw),
}


class Generation:
    """Greedy generation of the target request under one policy."""

    def __init__(self, tokens, logits_trace, wall_s, cost, row_tokens=None):
        self.tokens = tokens              # [n_steps] generated token ids
        self.logits_trace = logits_trace  # [n_steps, V] raw logits per step
        self.wall_s = wall_s
        self.cost = cost                  # dict of modelled work
        # Generated tokens for every decoded row (companions included for
        # batched policies); [target] only for pack/corun.
        self.row_tokens = row_tokens if row_tokens is not None else [tokens]


def generate(params, spec, n_steps=16, tile=16, dtype="bf16"):
    """Prefill + greedy decode of the target under ``spec``.

    For padded/rows the whole batch decodes together (companions generate
    their own greedy tokens); for pack the target KV is extracted after
    prefill and decode continues target-only; for corun prefill and decode
    are already single-request.
    """
    t0 = time.perf_counter()
    logits, cache = forward(params, spec.tokens, spec.mask, spec.positions,
                            tile=tile, dtype=dtype)
    target_len = spec.lengths[spec.target_row]
    if spec.policy == "pack":
        # Faithful handoff (contract C3): extract the target's KV state —
        # the last target_len entries of the packed cache — and decode
        # target-only from it.
        lx = spec.extra["target_len"]
        cache = {"k": cache["k"][:, :, -lx:, :].copy(),
                 "v": cache["v"][:, :, -lx:, :].copy()}
        lengths = [lx]
        width = lx
    else:
        lengths = list(spec.lengths)
        width = spec.width
    row = spec.target_row
    last = spec.target_last_index
    step_logits = [logits[row, last, :].astype(np.float64)]
    generated = [int(np.argmax(step_logits[0]))]
    # Per-row generated tokens (companions included for batched policies).
    if spec.policy in ("padded", "rows"):
        row_generated = [[int(np.argmax(logits[r, spec.width - 1, :]))]
                         for r in range(len(lengths))]
    else:
        row_generated = [list(generated)]  # copy: appended separately below
    decode_positions = 0
    decode_attn = 0
    for t in range(n_steps):
        if spec.policy in ("padded", "rows"):
            # Batched decode: every row appends its own greedy token.
            # The target's row is row 0; companions decode in parallel.
            batch_tokens = []
            for r in range(len(lengths)):
                if t == 0:
                    prev = logits[r, spec.width - 1, :]
                else:
                    prev = step_logits_all[r]
                batch_tokens.append(int(np.argmax(prev)))
            tokens = np.asarray([batch_tokens], dtype=np.int64).T  # [B, 1]
            # lengths grow by one per row; cache width grows by one.
            lengths_t = [L + t + 1 for L in spec.lengths]
            total = spec.width + t + 1
            mask = decode_mask(lengths_t, total)
            pos = np.asarray([[L + t] for L in spec.lengths], dtype=np.int64)
            logits, cache = forward(params, tokens, mask, pos, tile=tile,
                                    dtype=dtype, kv_cache=cache)
            step_logits_all = [logits[r, 0, :].astype(np.float64)
                               for r in range(len(lengths))]
            for r in range(len(lengths)):
                row_generated[r].append(int(np.argmax(step_logits_all[r])))
        else:
            # Target-only decode (pack after extraction, corun).
            tokens = np.asarray([[generated[-1]]], dtype=np.int64)
            lengths_t = [lengths[0] + t + 1]
            total = width + t + 1
            mask = decode_mask(lengths_t, total)
            pos = np.asarray([[lengths[0] + t]], dtype=np.int64)
            logits, cache = forward(params, tokens, mask, pos, tile=tile,
                                    dtype=dtype, kv_cache=cache)
            step_logits_all = [logits[0, 0, :].astype(np.float64)]
            row_generated[0].append(int(np.argmax(step_logits_all[0])))
        step_logits.append(step_logits_all[0])
        generated.append(int(np.argmax(step_logits_all[0])))
        decode_positions += (len(lengths) if spec.policy in ("padded", "rows")
                             else 1) * (width + t + 1 if spec.policy in ("padded", "rows") else total)
        decode_attn += (len(lengths) if spec.policy in ("padded", "rows")
                        else 1) * (width + t + 1 if spec.policy in ("padded", "rows") else total)
    wall = time.perf_counter() - t0
    # The loop above generated n_steps + 1 tokens (prefill token + n steps);
    # keep exactly n_steps generated tokens and their logits.
    tokens_out = np.asarray(generated[:n_steps], dtype=np.int64)
    logits_trace = np.stack(step_logits[:n_steps])
    row_tokens = [np.asarray(rg[:n_steps], dtype=np.int64) for rg in row_generated]
    if spec.policy == "corun":
        env = spec.extra["envelope"]
        decode_positions = env * sum(width + t + 1 for t in range(n_steps))
        decode_attn = env * sum(width + t + 1 for t in range(n_steps))
    cost = {
        "prefill_positions": spec.prefill_positions,
        "prefill_attn_work": spec.prefill_attn_work,
        "decode_positions": int(decode_positions),
        "decode_attn_work": int(decode_attn),
    }
    return Generation(tokens_out, logits_trace, wall, cost,
                      row_tokens=row_tokens)
