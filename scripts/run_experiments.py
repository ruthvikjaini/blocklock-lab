"""Main experiment grid: does the target's output survive companion changes?

Protocol (mirrors the paper's Section 4.1, at our scale):
* 200 random prompts (lengths 8..24, vocab 128); first-step top-2 margin
  measured for each. Two slices: near-tie (smallest margins, class width
  >= 16 so companion lengths can vary) and typical (margin >= 0.125).
* For each target: canonical run (target alone, B = 1) plus, for
  B in {2, 4, 8} and 8 companion sets, one generation per policy
  (padded / rows / pack / corun). Companion sets are identical across
  policies; for rows/pack companions are drawn from the target's length
  class (length <= W), as the paper's bucketing requires.
* Greedy decode, 16 steps, tile = 16, bf16; plus an fp32 control for the
  padded policy on the near-tie slice at B = 4.

Outputs: results/measured.json (headline aggregates) and
results/experiment.json (per-target, per-condition records).
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
RESULTS.mkdir(exist_ok=True)

SEED = 20261107
N_POOL = 200
N_TARGETS = 16
B_VALUES = (2, 4, 8)
REPS = 8
N_STEPS = 16
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


def run_condition(params, target, companions_list, policy, width, dtype):
    """Run one policy over a list of companion sets; return generations."""
    gens = []
    for comps in companions_list:
        kw = {"width": width} if policy in ("rows", "pack") else {}
        spec = policies.BUILDERS[policy](target, comps, **kw)
        gens.append(policies.generate(params, spec, n_steps=N_STEPS,
                                      tile=TILE, dtype=dtype))
    return gens


def main():
    t_start = time.time()
    params = model.init_params(seed=0)
    rng = np.random.default_rng(SEED)

    # --- prompt pool and target selection ---
    pool = [rand_prompt(rng) for _ in range(N_POOL)]
    scored = []
    for i, p in enumerate(pool):
        m = first_margin(params, p)
        scored.append({"idx": i, "prompt": p, "margin": m,
                       "width": policies.class_width(len(p))})
    near = [s for s in sorted(scored, key=lambda s: s["margin"])
            if s["width"] >= 16][:N_TARGETS]
    typical_pool = [s for s in scored if s["margin"] >= 0.125 and s["width"] >= 16]
    typical = [typical_pool[i] for i in
               rng.choice(len(typical_pool), size=N_TARGETS, replace=False)]
    print(f"pool margins: min={min(s['margin'] for s in scored):.4f} "
          f"median={float(np.median([s['margin'] for s in scored])):.4f}")
    print(f"near-tie margins: {[round(s['margin'], 4) for s in near]}")
    print(f"typical margins: {[round(s['margin'], 4) for s in typical]}")

    records = []
    headline = {}
    for label, targets in (("near_tie", near), ("typical", typical)):
        for t in targets:
            target = t["prompt"]
            width = t["width"]
            canon_spec = policies.build_corun(target, [])
            canon = policies.generate(params, canon_spec, n_steps=N_STEPS,
                                      tile=TILE, dtype="bf16")
            canon_fp32 = policies.generate(params, canon_spec, n_steps=N_STEPS,
                                           tile=TILE, dtype="fp32")
            for B in B_VALUES:
                comp_sets = []
                for _ in range(REPS):
                    comps = [rand_prompt(rng, lo=8, hi=width)
                             for _ in range(B - 1)]
                    comp_sets.append(comps)
                for policy in ("padded", "rows", "pack", "corun"):
                    gens = run_condition(params, target, comp_sets, policy,
                                         width, "bf16")
                    seq_ag, topk_ag, maxd = audit.pairwise_agreement(gens)
                    # flip_vs_isolated compares against the isolated B=1
                    # run in the same dtype. It measures the one-time
                    # geometry shift from isolated to batched execution,
                    # NOT companion sensitivity — the paper's contract is
                    # per-deployment, not cross-B.
                    fr = audit.flip_rate(gens, canon)
                    wall = float(np.median([g.wall_s for g in gens]))
                    cost = {k: int(sum(g.cost[k] for g in gens) / len(gens))
                            for k in gens[0].cost}
                    records.append({
                        "slice": label, "target_idx": t["idx"],
                        "target_len": len(target), "margin": t["margin"],
                        "width": width, "B": B, "policy": policy,
                        "dtype": "bf16", "seq_agreement": seq_ag,
                        "top20_agreement": topk_ag, "max_logit_delta": maxd,
                        "first_step_max_logit_delta": audit.first_step_delta(gens),
                        "flip_vs_isolated": fr, "wall_s_median": wall, "cost": cost,
                        "rep_tokens": [g.tokens.tolist() for g in gens],
                        "canon_tokens": canon.tokens.tolist(),
                    })
                    key = (label, B, policy)
                    headline.setdefault(key, []).append(records[-1])
                # fp32 control: padded only, near-tie slice, B = 4
                if label == "near_tie" and B == 4:
                    gens = run_condition(params, target, comp_sets, "padded",
                                         width, "fp32")
                    seq_ag, topk_ag, maxd = audit.pairwise_agreement(gens)
                    fr = audit.flip_rate(gens, canon_fp32)
                    records.append({
                        "slice": label, "target_idx": t["idx"],
                        "target_len": len(target), "margin": t["margin"],
                        "width": width, "B": B, "policy": "padded",
                        "dtype": "fp32", "seq_agreement": seq_ag,
                        "top20_agreement": topk_ag, "max_logit_delta": maxd,
                        "first_step_max_logit_delta": audit.first_step_delta(gens),
                        "flip_vs_isolated": fr, "wall_s_median": None, "cost": None,
                        "rep_tokens": [g.tokens.tolist() for g in gens],
                        "canon_tokens": canon_fp32.tokens.tolist(),
                    })

    # --- headline aggregates ---
    measured = {"slices": {}, "fp32_control": {}, "meta": {
        "seed": SEED, "pool": N_POOL, "targets_per_slice": N_TARGETS,
        "B_values": list(B_VALUES), "reps": REPS, "n_steps": N_STEPS,
        "tile": TILE, "elapsed_s": round(time.time() - t_start, 1)}}
    slices = {}
    for (label, B, policy), recs in sorted(headline.items()):
        agg = {
            "n_targets": len(recs),
            "seq_agreement_mean": float(np.mean([r["seq_agreement"] for r in recs])),
            "seq_agreement_min": float(np.min([r["seq_agreement"] for r in recs])),
            "top20_agreement_mean": float(np.mean([r["top20_agreement"] for r in recs])),
            "max_logit_delta_median": float(np.median([r["max_logit_delta"] for r in recs])),
            "max_logit_delta_max": float(np.max([r["max_logit_delta"] for r in recs])),
            "first_step_max_logit_delta_median": float(np.median([r["first_step_max_logit_delta"] for r in recs])),
            "first_step_max_logit_delta_max": float(np.max([r["first_step_max_logit_delta"] for r in recs])),
            "flip_vs_isolated_mean": float(np.mean([r["flip_vs_isolated"] for r in recs])),
            "flip_vs_isolated_max": float(np.max([r["flip_vs_isolated"] for r in recs])),
            "targets_with_any_flip": int(sum(1 for r in recs if r["seq_agreement"] < 1.0)),
            "wall_s_median": float(np.median([r["wall_s_median"] for r in recs])),
            "cost_mean": {k: float(np.mean([r["cost"][k] for r in recs]))
                          for k in recs[0]["cost"]},
        }
        slices.setdefault(label, {}).setdefault(f"B{B}", {})[policy] = agg
    measured["slices"] = slices
    # Cost ratios vs padded per (slice, B).
    for label in slices:
        for bkey in slices[label]:
            pad = slices[label][bkey]["padded"]["cost_mean"]
            for policy, agg in slices[label][bkey].items():
                c = agg["cost_mean"]
                agg["cost_ratio_vs_padded"] = {
                    k: round(c[k] / pad[k], 3) for k in c}
    fp32 = [r for r in records if r["dtype"] == "fp32"]
    measured["fp32_control"] = {
        "n_targets": len(fp32),
        "seq_agreement_mean": float(np.mean([r["seq_agreement"] for r in fp32])),
        "top20_agreement_mean": float(np.mean([r["top20_agreement"] for r in fp32])),
        "max_logit_delta_median": float(np.median([r["max_logit_delta"] for r in fp32])),
        "max_logit_delta_max": float(np.max([r["max_logit_delta"] for r in fp32])),
        "first_step_max_logit_delta_median": float(np.median([r["first_step_max_logit_delta"] for r in fp32])),
        "flip_vs_isolated_mean": float(np.mean([r["flip_vs_isolated"] for r in fp32])),
    }

    (RESULTS / "measured.json").write_text(json.dumps(measured, indent=2))
    (RESULTS / "experiment.json").write_text(json.dumps(records, indent=2))
    print(json.dumps(measured, indent=2))
    print(f"elapsed {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
