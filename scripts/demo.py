"""Demo: one near-tie target, four policies, eight companion sets.

Shows the headline in under ten seconds: padded batching lets the target's
output drift when the companions change, while BlockLock-Rows, BlockLock-Pack
and CoRun reproduce it bitwise.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from blocklock_lab import audit, model, policies

VOCAB = 128


def find_flipping_target(params, rng, width=16, pool=100):
    """Scan a pool of prompts, take the smallest first-step margins (the
    same selection as the main experiment), and return the first whose
    padded sequences disagree across 8 companion sets. Most near-tie
    targets drift only at the logit level; the demo wants one that flips."""
    cands = []
    for _ in range(pool):
        cand = [int(x) for x in rng.integers(0, VOCAB, size=12)]
        logits, _ = model.forward(
            params, np.asarray([cand]),
            model.causal_mask([len(cand)], len(cand)),
            np.arange(len(cand))[None, :], dtype="bf16")
        cands.append((audit.top2_margin(logits[0, -1, :]), cand))
    cands.sort(key=lambda t: t[0])
    for margin, cand in cands[:16]:
        gens = []
        for _ in range(8):
            comps = [[int(x) for x in rng.integers(0, VOCAB, size=int(rng.integers(8, width + 1)))]
                     for _ in range(3)]
            spec = policies.build_padded(cand, comps)
            gens.append(policies.generate(params, spec, n_steps=16, dtype="bf16"))
        if len({tuple(g.tokens.tolist()) for g in gens}) > 1:
            return cand, float(margin)
    return None, None


def main():
    params = model.init_params(seed=0)
    rng = np.random.default_rng(20261107)
    # A near-tie target of 12 tokens in class width 16: companion lengths
    # 8..16 make the padded width vary across sets, so the geometry (and
    # hence the bf16 rounding) varies too. A 16-token target would pin
    # T_max at 16 and show no drift at all.
    target, margin = find_flipping_target(params, rng)
    if target is None:
        print("no flipping target found in the search budget")
        return
    width = policies.class_width(len(target))
    print(f"target: {len(target)} tokens, class width {width}, "
          f"first-step margin {margin:.4f}")
    print(f"companions: 3 per set, lengths 8..{width}, 8 sets, 16 steps, bf16")
    for policy in ("padded", "rows", "pack", "corun"):
        gens = []
        for _ in range(8):
            comps = [[int(x) for x in rng.integers(0, VOCAB, size=int(rng.integers(8, width + 1)))]
                     for _ in range(3)]
            kw = {"width": width} if policy in ("rows", "pack") else {}
            spec = policies.BUILDERS[policy](target, comps, **kw)
            gens.append(policies.generate(params, spec, n_steps=16, dtype="bf16"))
        seq_ag, topk_ag, maxd = audit.pairwise_agreement(gens)
        n_seq = len({tuple(g.tokens.tolist()) for g in gens})
        print(f"{policy:7s} sequence agreement {seq_ag:.3f}  "
              f"top-20 agreement {topk_ag:.3f}  max |dlogit| {maxd:.4f}  "
              f"distinct sequences {n_seq}/8")


if __name__ == "__main__":
    main()
