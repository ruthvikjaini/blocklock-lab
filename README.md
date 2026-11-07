# blocklock-lab

Reproducing **BlockLock** (Qi, Reviriego, Conde & Liu, arXiv:2611.06977, 4 Nov 2026):
the same LLM request can produce a *different* answer depending on which
other requests share its batch — and fixing the batch geometry fixes the
answer, bitwise.

## The problem

Floating-point addition is not associative, so a sum's rounded result depends
on the order the terms are added. In a serving system that order depends on
tensor shapes, and the shapes depend on the lengths of the other requests in
the batch. A small logit change flips a near-tie between two tokens, and from
there the rest of the generation diverges. The paper measures this on real
GPUs and real 4–14B models: padded batching reaches only **52.59%**
generated-sequence agreement and **35.93%** Top-20 logit agreement across
companion changes, while both BlockLock schemes reach **100%**.

This lab reproduces the mechanism and the fix on CPU, with real bf16
arithmetic, on a tiny causal transformer (1 layer, 2 heads, d = 64,
vocab = 128, NumPy only, no model weights, no GPU, no LLM calls).

## What is reproduced, and what is modelled

A stock NumPy GEMM is row-independent, so it cannot show the effect — that is
why the lab implements the reduction the way GPU kernels do it: attention
softmax denominators and attention-weighted sums are computed as **split /
tiled reductions over the padded key axis** (tile = 16), with bf16-rounded
partials combined in order. When the padded length changes, the same logical
values are regrouped across tile boundaries and the bf16 rounding changes.
All drift numbers below are measured from this model, on real bf16
arithmetic (fp32 accumulate, bf16 store, round-to-nearest-even). They are
simulation numbers, not GPU numbers; the paper's numbers are quoted
separately and never mixed in.

Left padding is used throughout, as in the paper's Figure 3a (the target's
*padding offset* shifts with its companions). With right padding the
target's real keys always occupy the first positions, the tile grouping never
changes, and the drift is exactly zero — which is itself a check of the
mechanism. Four policies are implemented: **Padded** (`[B, T_max]`),
**BlockLock-Rows** (`[B, W]`, fixed class width, a multiple of block 8),
**BlockLock-Pack** (`[1, P = BW]`, segment-causal mask, target anchored at
the terminal interval, dummy segment absorbing the slack, KV extraction,
target-only decode), and **CoRun-style** (isolated B = 1 prefill, fixed
decode envelope of 8 slots).

## Measured results

Protocol: 200 random prompts (8–24 tokens); two slices of 16 targets —
near-tie (smallest first-step top-2 margins) and typical (margin ≥ 0.125).
For each target, B ∈ {2, 4, 8}, 8 companion sets, greedy decode, 16 steps,
bf16, tile 16. Agreement is the paper's definition: fraction of repetition
pairs with identical generated sequences (Top-20 agreement additionally
requires identical token ids *and* bitwise-identical raw logits at every
step). Full records in `results/experiment.json`; aggregates in
`results/measured.json`.

| slice / B | policy | sequence agreement | Top-20 agreement | median max \|Δlogit\| |
|---|---|---|---|---|
| near-tie, B=4 | Padded | 0.850 | 0.529 | 0.023 |
| near-tie, B=4 | Rows | 1.000 | 1.000 | 0.000 |
| near-tie, B=4 | Pack | 1.000 | 1.000 | 0.000 |
| near-tie, B=4 | CoRun | 1.000 | 1.000 | 0.000 |
| near-tie, B=8 | Padded | 0.848 | 0.478 | 0.023 |
| near-tie, B=2 | Padded | 0.984 | 0.757 | 0.010 |
| typical, B=4 | Padded | 0.891 | 0.386 | 0.023 |

- **Padded batching drifts.** At B=4 on near-tie targets, only 0.850 of
  repetition pairs agree on the generated sequence (worst target 0.429),
  and Top-20 logit agreement is 0.529. 6 of 16 near-tie targets flip on at
  least one companion set. The median largest logit difference is 0.023 —
  one to two bf16 ulps, which is all a near-tie needs.
- **Rows, Pack and CoRun reproduce bitwise.** Sequence agreement 1.000,
  Top-20 agreement 1.000, max |Δlogit| exactly 0.000 in every condition
  (both slices, B = 2/4/8). The fix is exact, not approximate.
- **fp32 control.** The same padded grid in fp32: sequence agreement 1.000,
  median max |Δlogit| 7.15e-07, no flips. The regrouping still changes bits
  (Top-20 agreement 0.529 even in fp32 — fp32 is not bitwise stable under
  regrouping either), but the drift is ~30,000x smaller than in bf16 and
  far below any flip threshold.
- **One subtlety, stated plainly.** Rows/Pack reproduce exactly *across
  companion changes within a fixed deployment*, but their logits differ
  from the isolated B = 1 run (flip-vs-isolated 0.062 for Rows/Pack on
  near-tie targets, 0.125 on typical targets; CoRun is 0.000 because its
  prefill *is* the isolated run). That is the paper's contract, not a bug:
  BlockLock fixes the batch size and geometry of the compared runs; it
  does not promise equality across different batch sizes.

## Cost (modelled token work, normalised to Padded = 1.00, B=4, near-tie)

| policy | prefill positions | prefill attention work | decode positions |
|---|---|---|---|
| Padded | 1.000 | 1.000 | 1.000 |
| Rows | 1.073 | 1.143 | 1.048 |
| Pack | 1.073 | 4.573 | 0.236 |
| CoRun | 0.786 | 0.655 | 1.885 |

Rows costs a little padding (7–14% at this tiny scale, where prompts are
8–24 tokens and the class width rounds up to a multiple of 8; the paper's
0.5–0.8% latency overhead is at production scale, where the same rounding
is a much smaller fraction of a hundreds-of-tokens prompt). Pack pays
quadratically in prefill (the dense `P²` segment mask: 4.573x at B=4,
8.746x at B=8) but decodes target-only at 0.236x. CoRun processes the
fewest prefill positions (no padding waste) but pays for it in latency —
its prefill is serial, and the fixed 8-slot decode envelope costs 3.929x
at B=2 when mostly empty, falling to 0.928x at B=8 when full. Positions
are not latency for a serial policy; the paper's latency numbers
(CoRun 3.12–3.28x shared batching; Rows 2.15–2.22x faster than vLLM
Batch-Invariant at equal delivered work) are quoted separately.

## Extensions (results/extension.json)

- **E1 — the near-tie that matters is step-local, not first-step.** The
  first-step top-2 margin does *not* predict flips (flip fractions 0.000
  in the two smallest-margin bins, 0.111–0.250 in larger ones, n = 120).
  But the margin at the step where a sequence first diverges has median
  0.008, against 0.219 for margins at all steps — 28x smaller. Flips
  happen at near-ties; the relevant near-tie is the one at the step where
  the drift is live, and prefill drift is usually exactly zero (see E5).
- **E2 — decode-length amplification.** Padded sequence agreement falls
  from 0.931 at 8 steps to 0.759 at 64 steps (Rows stays 1.000): a small
  per-step drift is a fresh near-tie lottery at every step, so longer
  generations diverge more. This is the paper's step-280 flip at small
  scale.
- **E3 — position-shift audit: a second drift class.** When position ids
  are derived from padded coordinates instead of logical positions,
  padded agreement collapses to 0.536 in bf16 and 0.496 in fp32, with
  median max |Δlogit| ~3.70 in both dtypes. Rows fixes it too (1.000 in
  both dtypes), because the fixed width pins the coordinates. This drift
  is *not* a rounding effect — it is a different bug class that the same
  geometry fix removes.
- **E4 — protection is per-request, not per-row.** With one companion
  held fixed across repetitions and the others varied, that companion's
  own sequence agreement is 0.949 under padded batching and 1.000 under
  Rows. Any row in a fixed-width batch is protected, not just the target.
- **E5 — tile sensitivity confirms the mechanism.** Padded sequence
  agreement is 0.757 at tile 4, 0.817 at tile 8, 0.882 at tile 16 and
  0.967 at tile 32; the first-step drift median is 0.016 at tiles 4–8
  and 0.000 at tiles 16–32. Drift appears once a sequence spans more
  than one reduction tile — which is why prefill drift is usually zero
  at tile 16 (prompts ≤ 24 tokens fit in one or two tiles) and appears
  during decode as the cache grows across the boundary.

## Limitations

- The kernel-level mechanism (split/tiled bf16 reductions) is modelled,
  not observed on a GPU; a stock NumPy GEMM cannot show the effect.
  All drift numbers are from that model.
- Tiny model (1 layer, 2 heads, d = 64), random weights, synthetic
  prompts — the near-tie regime is constructed by margin selection, the
  way the paper's BFCL flips are the near-tie queries of a real benchmark.
- Cost is a token-work model plus NumPy CPU wall-clock, not GPU latency.
- The paper's numbers (52.59% / 35.93% padded agreement, 100% BlockLock,
  0.5–0.8% Rows overhead, 2.15–2.22x vs vLLM Batch-Invariant) are on
  Qwen3/Llama/Gemma 4–14B on A6000/A100/H100; this lab's numbers are on
  the toy model above and are not comparable in magnitude — only the
  ordering and the mechanism are.

## Run it

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/demo.py                 # headline in ~10 s
.venv/bin/python scripts/run_experiments.py      # full grid -> results/
.venv/bin/python scripts/run_extensions.py       # E1-E5 -> results/
.venv/bin/python scripts/crossfoot.py            # README vs results
.venv/bin/python -m pytest tests/ -q             # test suite
```

## Layout

- `src/blocklock_lab/` — `bf16.py` (bf16 emulation), `reductions.py`
  (shape-dependent tiled reductions), `model.py` (tiny transformer + KV
  cache), `policies.py` (Padded / Rows / Pack / CoRun + greedy decode),
  `audit.py` (agreement metrics, Section 3.4 contract check)
- `scripts/` — `demo.py`, `run_experiments.py`, `run_extensions.py`,
  `crossfoot.py`
- `tests/` — pytest suite (bf16, reductions, model, policies, audit,
  branding, crossfoot)
- `docs/RESEARCH.md` — trending check, paper verification, rejected
  candidates, design notes
- `results/` — `measured.json`, `experiment.json`, `extension.json`

## Sources

- Qi, Reviriego, Conde & Liu, "BlockLock: Reproducible LLM Inference with
  Fixed Batch Geometry", arXiv:2611.06977, 4 Nov 2026.
  https://arxiv.org/abs/2611.06977
- Trending check, 7 Nov 2026: GitHub daily/weekly trending and
  star-velocity trackers (gitdiscover, findarepo, topopensource.ai,
  github.locker) — dominated by skill/prompt packs, agent frameworks and
  full platforms; no repo with a measurable one-night CPU weakness, so the
  paper fallback was used. Details in `docs/RESEARCH.md`.
