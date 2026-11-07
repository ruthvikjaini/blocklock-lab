"""Cross-foot every number in README.md against the raw run output.

Recomputes the headline aggregates from results/experiment.json (the
per-condition records) and the E1 fractions from results/extension.json,
compares them with results/measured.json, and checks that README.md
contains the formatted values. Exits nonzero on any mismatch.
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def load(name):
    return json.loads((RESULTS / name).read_text())


def recompute_slices(records):
    """Recompute slice aggregates from the per-condition records."""
    from collections import defaultdict
    groups = defaultdict(list)
    for r in records:
        if r["dtype"] == "bf16":
            groups[(r["slice"], r["B"], r["policy"])].append(r)
    out = {}
    for (label, b, policy), recs in groups.items():
        agg = {
            "seq_agreement_mean": float(np.mean([r["seq_agreement"] for r in recs])),
            "seq_agreement_min": float(np.min([r["seq_agreement"] for r in recs])),
            "top20_agreement_mean": float(np.mean([r["top20_agreement"] for r in recs])),
            "max_logit_delta_median": float(np.median([r["max_logit_delta"] for r in recs])),
            "max_logit_delta_max": float(np.max([r["max_logit_delta"] for r in recs])),
            "first_step_max_logit_delta_median": float(np.median([r["first_step_max_logit_delta"] for r in recs])),
            "flip_vs_isolated_mean": float(np.mean([r["flip_vs_isolated"] for r in recs])),
            "targets_with_any_flip": int(sum(1 for r in recs if r["seq_agreement"] < 1.0)),
            "cost_mean": {k: float(np.mean([r["cost"][k] for r in recs]))
                          for k in recs[0]["cost"]},
        }
        out[(label, b, policy)] = agg
    return out


def check():
    failures = []
    measured = load("measured.json")
    records = load("experiment.json")
    ext = load("extension.json")
    readme = (ROOT / "README.md").read_text()

    recomputed = recompute_slices(records)
    for (label, b, policy), agg in recomputed.items():
        stored = measured["slices"][label][f"B{b}"][policy]
        for key, val in agg.items():
            if key == "cost_mean":
                for k, v in val.items():
                    if abs(stored[key][k] - v) > 1e-9:
                        failures.append(f"{label} B{b} {policy} cost {k}: "
                                        f"stored {stored[key][k]} != recomputed {v}")
                continue
            if abs(stored[key] - val) > 1e-9:
                failures.append(f"{label} B{b} {policy} {key}: stored "
                                f"{stored[key]} != recomputed {val}")
        # Cost ratios vs padded, same (slice, B).
        pad = recomputed[(label, b, "padded")]["cost_mean"]
        for k, v in agg["cost_mean"].items():
            ratio = round(v / pad[k], 3)
            if abs(stored["cost_ratio_vs_padded"][k] - ratio) > 5e-4:
                failures.append(f"{label} B{b} {policy} cost ratio {k}: "
                                f"stored {stored['cost_ratio_vs_padded'][k]} "
                                f"!= recomputed {ratio}")

    # fp32 control.
    fp32 = [r for r in records if r["dtype"] == "fp32"]
    fc = measured["fp32_control"]
    checks = {
        "seq_agreement_mean": float(np.mean([r["seq_agreement"] for r in fp32])),
        "top20_agreement_mean": float(np.mean([r["top20_agreement"] for r in fp32])),
        "max_logit_delta_median": float(np.median([r["max_logit_delta"] for r in fp32])),
        "flip_vs_isolated_mean": float(np.mean([r["flip_vs_isolated"] for r in fp32])),
    }
    for key, val in checks.items():
        if abs(fc[key] - val) > 1e-9:
            failures.append(f"fp32 {key}: stored {fc[key]} != recomputed {val}")

    # E1 fractions.
    for key, v in ext["E1_margin_vs_flip"]["by_first_step_margin"].items():
        if v["n"]:
            frac = v["flipped"] / v["n"]
            if abs(v["flip_fraction"] - frac) > 1e-9:
                failures.append(f"E1 {key}: stored {v['flip_fraction']} "
                                f"!= recomputed {frac}")

    # README must contain the headline values (3-decimal formatting).
    headline_values = [
        ("near-tie B4 padded seq agreement",
         measured["slices"]["near_tie"]["B4"]["padded"]["seq_agreement_mean"]),
        ("near-tie B4 padded top-20 agreement",
         measured["slices"]["near_tie"]["B4"]["padded"]["top20_agreement_mean"]),
        ("near-tie B4 rows seq agreement",
         measured["slices"]["near_tie"]["B4"]["rows"]["seq_agreement_mean"]),
        ("near-tie B8 padded seq agreement",
         measured["slices"]["near_tie"]["B8"]["padded"]["seq_agreement_mean"]),
        ("typical B4 padded seq agreement",
         measured["slices"]["typical"]["B4"]["padded"]["seq_agreement_mean"]),
        ("E1 flip-step margin median",
         ext["E1_margin_vs_flip"]["flip_step_margin_median"]),
        ("E1 all-step margin median",
         ext["E1_margin_vs_flip"]["all_step_margin_median"]),
        ("E2 64-step padded seq agreement",
         ext["E2_decode_length"]["steps_64"]["padded"]["seq_agreement_mean"]),
        ("E3 padded bf16 seq agreement",
         ext["E3_position_shift"]["padded_bf16"]["seq_agreement_mean"]),
        ("E3 padded fp32 seq agreement",
         ext["E3_position_shift"]["padded_fp32"]["seq_agreement_mean"]),
        ("E4 row-1 padded agreement",
         ext["E4_per_request_protection"]["padded"]["row1_seq_agreement_mean"]),
        ("E5 tile-4 seq agreement",
         ext["E5_tile_sensitivity"]["tile_4"]["seq_agreement_mean"]),
        ("E5 tile-32 seq agreement",
         ext["E5_tile_sensitivity"]["tile_32"]["seq_agreement_mean"]),
    ]
    for label, val in headline_values:
        s = f"{val:.3f}"
        if s not in readme:
            failures.append(f"README missing {label}: '{s}'")
    # fp32 drift is ~1e-6; check the scientific-notation string.
    if "7.15e-07" not in readme and "7.2e-07" not in readme:
        failures.append("README missing fp32 max logit delta median (~7.15e-07)")
    # Exact-zero claims: rows/pack/corun max logit delta must be 0.0.
    for label in ("near_tie", "typical"):
        for b in ("B2", "B4", "B8"):
            for policy in ("rows", "pack", "corun"):
                agg = measured["slices"][label][b][policy]
                if agg["max_logit_delta_max"] != 0.0:
                    failures.append(f"{label} {b} {policy} max delta "
                                    f"{agg['max_logit_delta_max']} != 0.0")
                if agg["seq_agreement_mean"] != 1.0:
                    failures.append(f"{label} {b} {policy} seq agreement "
                                    f"{agg['seq_agreement_mean']} != 1.0")
    return failures


def main():
    failures = check()
    if failures:
        print(f"CROSSFOOT FAIL ({len(failures)}):")
        for f in failures:
            print(" -", f)
        sys.exit(1)
    print("CROSSFOOT PASS")


if __name__ == "__main__":
    main()
