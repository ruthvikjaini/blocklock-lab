"""Extensions — five checks beyond the paper's own evaluation.

E1  Margin vs flip: flip probability as a function of the target's first-step
    top-2 margin (the paper's flips happen at near-ties; here the curve is
    measured, not assumed).
E2  Decode-length amplification: sequence agreement vs generation length
    (the paper's step-280 flip is the same phenomenon at real scale).
E3  Position-shift audit: a second, independent drift class — position ids
    derived from padded coordinates instead of logical positions. It shows
    up even in fp32, and Rows fixes it too.
E4  Per-request protection: the companion in row 1 is measured as well —
    under padded batching it drifts like the target; under Rows it does not.
    Protection is a property of the geometry, not of which row you watch.
E5  Tile sensitivity: drift vs reduction tile size (4/8/16/32). Drift
    appears once the sequence spans more than one reduction tile.

Outputs: results/extension.json.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from blocklock_lab import audit, model, policies

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

SEED = 20261107
N_POOL = 200
N_TARGETS = 16
REPS = 8
TILE = 16
VOCAB = 128


def rand_prompt(rng, lo=8, hi=24):
    return [int(x) for x in rng.integers(0, VOCAB, size=int(rng.integers(lo, hi + 1)))]


def first_margin(params, prompt):
    logits, _ = model.forward(
        params, np.asarray([prompt]),
        model.causal_mask([len(prompt)], len(prompt)),
        np.arange(len(prompt))[None, :], dtype="bf16")
    return audit.top2_margin(logits[0, -1, :])


def near_tie_targets(params):
    rng = np.random.default_rng(SEED)
    pool = [rand_prompt(rng) for _ in range(N_POOL)]
    scored = []
    for i, p in enumerate(pool):
        scored.append({"idx": i, "prompt": p, "margin": first_margin(params, p),
                       "width": policies.class_width(len(p))})
    near = [s for s in sorted(scored, key=lambda s: s["margin"])
            if s["width"] >= 16][:N_TARGETS]
    return near, rng


def main():
    t_start = time.time()
    params = model.init_params(seed=0)
    ext = {}

    # --- E1: margin vs flip ---
    # First pass: first-step margin vs any flip (as originally planned).
    # Second pass: the margin AT the step where a sequence first diverges
    # from rep 0, vs the margin distribution at all steps. The first-step
    # margin turns out to be a poor predictor (prefill drift is usually
    # exactly zero in the single-tile regime); the step-local margin is
    # the quantity the near-tie story is actually about.
    rng1 = np.random.default_rng(555)
    pool = [rand_prompt(rng1) for _ in range(120)]
    bins = [(0.0, 0.0156), (0.0156, 0.0312), (0.0312, 0.0625),
            (0.0625, 0.125), (0.125, 0.25), (0.25, 10.0)]
    bin_stats = {f"{lo}-{hi}": {"n": 0, "flipped": 0} for lo, hi in bins}
    flip_step_margins = []
    all_step_margins = []
    for p in pool:
        m = first_margin(params, p)
        width = policies.class_width(len(p))
        comps_sets = [[rand_prompt(rng1, lo=8, hi=width) for _ in range(3)]
                      for _ in range(4)]
        gens = []
        for comps in comps_sets:
            spec = policies.build_padded(p, comps)
            gens.append(policies.generate(params, spec, n_steps=16,
                                          tile=TILE, dtype="bf16"))
        seq_ag, _, _ = audit.pairwise_agreement(gens)
        flipped = 1 if seq_ag < 1.0 else 0
        for lo, hi in bins:
            if lo <= m < hi or (lo == 0.0 and m == 0.0):
                key = f"{lo}-{hi}"
                bin_stats[key]["n"] += 1
                bin_stats[key]["flipped"] += flipped
                break
        ref = gens[0]
        all_step_margins.extend(
            audit.top2_margin(ref.logits_trace[s]) for s in range(16))
        for g in gens[1:]:
            div = [s for s in range(16) if g.tokens[s] != ref.tokens[s]]
            if div:
                flip_step_margins.append(
                    audit.top2_margin(ref.logits_trace[div[0]]))
    ext["E1_margin_vs_flip"] = {
        "by_first_step_margin": {
            k: {"n": v["n"], "flipped": v["flipped"],
                "flip_fraction": (v["flipped"] / v["n"] if v["n"] else None)}
            for k, v in bin_stats.items()},
        "flip_step_margin_median": (float(np.median(flip_step_margins))
                                     if flip_step_margins else None),
        "flip_step_margin_n": len(flip_step_margins),
        "all_step_margin_median": float(np.median(all_step_margins)),
        "all_step_margin_q10": float(np.quantile(all_step_margins, 0.1)),
    }

    # --- E2–E5 use the main grid's near-tie targets ---
    near, rng = near_tie_targets(params)

    # --- E2: decode-length amplification ---
    e2 = {}
    for n_steps in (8, 16, 32, 64):
        row = {}
        for policy in ("padded", "rows"):
            seqs = []
            for t in near:
                comp_sets = [[rand_prompt(rng, lo=8, hi=t["width"])
                              for _ in range(3)] for _ in range(REPS)]
                gens = []
                for comps in comp_sets:
                    kw = {"width": t["width"]} if policy == "rows" else {}
                    spec = policies.BUILDERS[policy](t["prompt"], comps, **kw)
                    gens.append(policies.generate(params, spec, n_steps=n_steps,
                                                  tile=TILE, dtype="bf16"))
                seq_ag, _, _ = audit.pairwise_agreement(gens)
                seqs.append(seq_ag)
            row[policy] = {"seq_agreement_mean": float(np.mean(seqs)),
                           "targets_with_any_flip": int(sum(1 for s in seqs if s < 1.0))}
        e2[f"steps_{n_steps}"] = row
    ext["E2_decode_length"] = e2

    # --- E3: position-shift audit ---
    e3 = {}
    for dtype in ("bf16", "fp32"):
        for policy in ("padded", "rows"):
            seqs, deltas = [], []
            for t in near:
                comp_sets = [[rand_prompt(rng, lo=8, hi=t["width"])
                              for _ in range(3)] for _ in range(REPS)]
                gens = []
                for comps in comp_sets:
                    kw = {"width": t["width"], "positions": "padded"} \
                        if policy == "rows" else {"positions": "padded"}
                    spec = policies.BUILDERS[policy](t["prompt"], comps, **kw)
                    gens.append(policies.generate(params, spec, n_steps=16,
                                                  tile=TILE, dtype=dtype))
                seq_ag, _, maxd = audit.pairwise_agreement(gens)
                seqs.append(seq_ag)
                deltas.append(maxd)
            e3[f"{policy}_{dtype}"] = {
                "seq_agreement_mean": float(np.mean(seqs)),
                "max_logit_delta_median": float(np.median(deltas))}
    ext["E3_position_shift"] = e3

    # --- E4: per-request protection (row-1 companion) ---
    # The companion in row 1 is FIXED across repetitions; only the other
    # companions vary. Under padded batching its output drifts with the
    # batch; under Rows its geometry is fixed and it does not.
    e4 = {}
    for policy in ("padded", "rows"):
        seqs = []
        for t in near:
            fixed_comp = rand_prompt(rng, lo=8, hi=t["width"])
            comp_sets = [[fixed_comp]
                         + [rand_prompt(rng, lo=8, hi=t["width"])
                            for _ in range(2)] for _ in range(REPS)]
            gens = []
            for comps in comp_sets:
                kw = {"width": t["width"]} if policy == "rows" else {}
                spec = policies.BUILDERS[policy](t["prompt"], comps, **kw)
                gens.append(policies.generate(params, spec, n_steps=16,
                                              tile=TILE, dtype="bf16"))
            # Row 1 is the fixed companion; its tokens across reps.
            import itertools as _it
            pairs = list(_it.combinations(range(len(gens)), 2))
            ok = sum(1 for i, j in pairs
                     if np.array_equal(gens[i].row_tokens[1],
                                       gens[j].row_tokens[1]))
            seqs.append(ok / len(pairs))
        e4[policy] = {"row1_seq_agreement_mean": float(np.mean(seqs)),
                      "targets_with_row1_flip": int(sum(1 for s in seqs if s < 1.0))}
    ext["E4_per_request_protection"] = e4

    # --- E5: tile sensitivity ---
    e5 = {}
    for tile in (4, 8, 16, 32):
        seqs, d1s, dmaxs = [], [], []
        for t in near:
            comp_sets = [[rand_prompt(rng, lo=8, hi=t["width"])
                          for _ in range(3)] for _ in range(REPS)]
            gens = []
            for comps in comp_sets:
                spec = policies.build_padded(t["prompt"], comps)
                gens.append(policies.generate(params, spec, n_steps=16,
                                              tile=tile, dtype="bf16"))
            seq_ag, _, maxd = audit.pairwise_agreement(gens)
            seqs.append(seq_ag)
            d1s.append(audit.first_step_delta(gens))
            dmaxs.append(maxd)
        e5[f"tile_{tile}"] = {
            "seq_agreement_mean": float(np.mean(seqs)),
            "first_step_max_logit_delta_median": float(np.median(d1s)),
            "max_logit_delta_median": float(np.median(dmaxs))}
    ext["E5_tile_sensitivity"] = e5

    ext["meta"] = {"seed": SEED, "reps": REPS, "tile": TILE,
                   "elapsed_s": round(time.time() - t_start, 1)}
    (RESULTS / "extension.json").write_text(json.dumps(ext, indent=2))
    print(json.dumps(ext, indent=2))
    print(f"elapsed {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
