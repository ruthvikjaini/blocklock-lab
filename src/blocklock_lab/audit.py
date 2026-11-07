"""Auditing a serving policy: does the target's output survive companion
changes?

Metrics follow the paper's definitions (Section 4.2):

* ``sequence_agreement`` — fraction of repetition pairs whose generated
  token sequences are identical.
* ``topk_agreement`` — fraction of pairs whose top-k token ids *and* raw
  logits are bitwise identical at every generation step.
* ``max_logit_delta`` — largest absolute raw-logit difference for the
  target across any pair of repetitions, at any step.
* ``flip_rate`` — fraction of repetitions whose generated sequence differs
  from the canonical (companion-free, B = 1) run.

``contract_check`` verifies the paper's Section 3.4 contract on the lab's
own terms: C1 canonical geometry (the target's padded width / terminal
interval is identical across repetitions), C2 repeat-stable arithmetic
(the same geometry twice gives bitwise-identical logits), C3 faithful
handoff (Pack's extracted KV equals a standalone prefill's KV, bitwise).
"""

import itertools

import numpy as np


def pairwise_agreement(generations, topk=20):
    """Agreement over all repetition pairs.

    ``generations`` is a list of Generation objects for the same target.
    Returns (sequence_agreement, topk_agreement, max_logit_delta).
    """
    n = len(generations)
    pairs = list(itertools.combinations(range(n), 2))
    seq_ok = 0
    topk_ok = 0
    max_delta = 0.0
    for i, j in pairs:
        gi, gj = generations[i], generations[j]
        if np.array_equal(gi.tokens, gj.tokens):
            seq_ok += 1
        li, lj = gi.logits_trace, gj.logits_trace
        same_ids = True
        same_bits = True
        for a, b in zip(li, lj):
            ki = np.argsort(-a)[:topk]
            kj = np.argsort(-b)[:topk]
            if not np.array_equal(ki, kj):
                same_ids = False
            if not np.array_equal(a, b):
                same_bits = False
        if same_ids and same_bits:
            topk_ok += 1
        max_delta = max(max_delta, float(np.max(np.abs(li - lj))))
    return seq_ok / len(pairs), topk_ok / len(pairs), max_delta


def flip_rate(generations, canonical):
    """Fraction of repetitions whose sequence differs from the canonical run."""
    flips = sum(0 if np.array_equal(g.tokens, canonical.tokens) else 1
                for g in generations)
    return flips / len(generations)


def first_step_delta(generations):
    """Largest absolute raw-logit difference at the first (prefill) step.

    This is the pure prefill drift, before autoregressive amplification can
    send flipped sequences down different paths and inflate later deltas.
    """
    first = [g.logits_trace[0] for g in generations]
    max_delta = 0.0
    for i, j in itertools.combinations(range(len(first)), 2):
        max_delta = max(max_delta, float(np.max(np.abs(first[i] - first[j]))))
    return max_delta


def top2_margin(logits):
    """Top-1 minus top-2 logit gap for a single logit vector."""
    s = np.sort(np.asarray(logits, dtype=np.float64))
    return float(s[-1] - s[-2])


def contract_check(params, spec_a, spec_b, tile=16, dtype="bf16"):
    """Check C1–C3 for two prefill specs of the same target.

    Returns a dict: c1_geometry (same width and same target interval),
    c2_repeat (same spec twice -> bitwise-identical logits), c3_handoff
    (only meaningful for pack: extracted KV == standalone KV).
    """
    from .model import forward
    from .policies import build_corun

    c1 = (spec_a.width == spec_b.width
          and spec_a.target_last_index == spec_b.target_last_index)
    logits_a, _ = forward(params, spec_a.tokens, spec_a.mask,
                          spec_a.positions, tile=tile, dtype=dtype)
    logits_a2, _ = forward(params, spec_a.tokens, spec_a.mask,
                           spec_a.positions, tile=tile, dtype=dtype)
    c2 = bool(np.array_equal(logits_a, logits_a2))
    out = {"c1_geometry": bool(c1), "c2_repeat_stable": c2,
           "c3_handoff": None}
    if spec_a.policy == "pack":
        # Standalone prefill of the target alone.
        target_tokens = spec_a.tokens[0, -spec_a.extra["target_len"]:]
        solo = build_corun(target_tokens, [])
        _, solo_cache = forward(params, solo.tokens, solo.mask,
                                solo.positions, tile=tile, dtype=dtype)
        _, pack_cache = forward(params, spec_a.tokens, spec_a.mask,
                                spec_a.positions, tile=tile, dtype=dtype)
        lx = spec_a.extra["target_len"]
        extracted_k = pack_cache["k"][:, :, -lx:, :]
        extracted_v = pack_cache["v"][:, :, -lx:, :]
        out["c3_handoff"] = bool(
            np.array_equal(extracted_k, solo_cache["k"])
            and np.array_equal(extracted_v, solo_cache["v"]))
    return out
