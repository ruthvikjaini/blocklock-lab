# Research notes — blocklock-lab (Day 14, 2026-11-07)

## Trending-first check (2026-11-07, ~00:00 ET)

GitHub daily/weekly trending and star-velocity trackers checked first, per the
standing rule:

- gitdiscover daily picks (Nov 7): Anil-matcha/Open-Generative-AIJavaScript
  (32.8K stars, image/video generation studio), pipeshub-ai/pipeshub-ai
  (context-layer platform), tokio-rs/tokio, tang1693/KIKUmap — plus
  NVIDIA/auto-ontology (+2,005 stars/24h, 5,526 stars, 436 forks, Apache-2.0).
- findarepo papers-with-code (Nov 6): google-research/timesfm (35K, +423/7d),
  LearningCircuit/local-deep-research (9.5K), chenxiachan/textfold.
- findarepo AI agents (Nov 6): Ebony-Vinyl/dsh-our-free-model (+11K/7d),
  thedotmack/claude-mem (114K, +8.5K/7d), deepseek-ai/deepseek-harness (274K),
  stablyai/orca (117K).
- github.locker LLM agents (Nov 5): agentscope (35.5K), next-ai-draw-io
  (36.7K), titanwings/distilly (26.4K), EverMind-AI/Raven (6.1K).

Judgement, same as the last eight nights: the trending list is dominated by
skill/prompt packs, agent frameworks/platforms, and full applications. The
code repos with real implementations (auto-ontology, timesfm, local-deep-
research, deepseek-harness) need NIM API keys, a 1.5 TB-class checkpoint, a
real LLM + eval harness, or a GPU fleet — none has a genuine, measurable,
one-night CPU weakness. **No trending repo picked; paper fallback per the
standing rule.**

## Paper chosen

Qi, Reviriego, Conde & Liu, "BlockLock: Reproducible LLM Inference with
Fixed Batch Geometry" (arXiv:2611.06977, submitted 4 Nov 2026).
Verified on the arXiv abstract page and full HTML on 2026-11-07.

Core claims (kept separate from our numbers; paper numbers are on real GPUs
and real 4–14B models, ours are on a tiny NumPy model with a modelled
reduction — see "What we reproduce and what we model" below):

- Reproducible inference = the same request produces the same output
  regardless of the other requests in the batch. Floating-point addition is
  not associative, so the result depends on the order of the sums, and the
  order depends on tensor shapes, which depend on companion lengths.
- Padded batching: only **52.59%** pair-weighted generated-sequence agreement
  and **35.93%** Top-20 logit agreement across companion changes (single-GPU
  subset); on 2-GPU tensor-parallel, padded agreement degrades 100% → 72.4%
  → 43.6% → 21.8% at B = 1/2/4/8.
- BlockLock-Rows (fixed-width rows `[B, W]`, class width a multiple of block
  size L = 64) and BlockLock-Pack (one `[1, BW]` block, target anchored at
  the terminal interval `[P − Lx, P)`, isolated dummy segment absorbing length
  variation, then target-only KV extraction and decode) both reach **100%**
  sequence and Top-20 agreement, single- and multi-GPU.
- Efficiency: BlockLock-Rows costs only **0.5–0.8%** latency overhead over
  standard PyTorch batching; at equal delivered work it is **2.15–2.22x**
  faster than vLLM Batch-Invariant, reducing latency 53.5–55.0% and improving
  throughput 115.2–122.2% vs vLLM BI. CoRun-style isolated prefill costs
  3.12–3.28x the latency of shared batched execution.
- Contract (Sec. 3.4): bitwise-identical target logits at every prediction
  index under three sufficient conditions — C1 canonical geometry, C2
  repeat-stable arithmetic, C3 faithful cache handoff. Backend policy
  excludes non-repeat-stable components (cuDNN SDPA, TF32); at B = 1 the
  default backend alone produced divergent sequences for an identical prompt
  (a logit changed 24.625 → 24.5 at step 280, flipping a near-tie).
- Mechanism: the paper does not pin a single kernel-level mechanism
  ("identifying an internal mechanism such as atomic accumulation or split
  reduction requires kernel-level evidence", Sec. 2.2). We model it as
  shape-dependent split/tiled reductions (see below).

## Candidates considered and rejected

- Liu, "There Is No Free Lunch in RLVR" (arXiv:2610.39295, 29 Oct 2026) —
  held a fifth night. Strong paper (exponential-tilt model, Stiemke price,
  sign-variation crossing bound, entropy-performance law, diagnostic from any
  eval log, verification package 34/34 checks). Rejected again: it is the
  RLVR cluster (Days 1, 7, 10, 12, 13) and tonight's pick breaks that run
  with a fresh inference-systems cluster. It remains the first fallback for
  a future night.
- KAPPO sharpening (arXiv:2611.03146) — same RLVR cluster, held.
- LegoEnv (arXiv, 6 Nov 2026; 5,465 self-evolving agent envs, +2.7pp on
  AutomationBench) — needs a real LLM (Qwen3-8B) + verifier runs; not a
  one-night CPU build.
- TraceSpec (6 Nov 2026, speculative decoding for LLM agents) — needs a real
  model + draft/target pair.
- "A Night at the LLM Zoo" (flow matching for generating pretrained LLM
  weights) — needs real checkpoints to be honest; a toy version would be
  decorative.
- BlockLock (arXiv:2611.06977) — **chosen**: Nov 4, fresh cluster
  (inference systems / determinism), a mechanism that is exactly reproducible
  on CPU with real bf16 arithmetic, and a fix (Rows/Pack) whose correctness
  is a bitwise identity — the strongest possible verification.

## What we reproduce and what we model

The lab is an independent implementation; no paper code was read or used
(the paper ships no public code link on its arXiv page at check time).

- Tiny causal transformer (1 layer, 2 heads, d = 64, vocab 128, learned
  absolute positions), forward-only, NumPy, CPU-only, no LLM calls.
- Real bf16 arithmetic: every linear/attention result is computed in fp32
  and rounded to bf16 with round-to-nearest-even (bit-level emulation,
  tested against known values). This is the standard bf16 tensor-core model
  (fp32 accumulate, bf16 store).
- The kernel-level mechanism is modelled, not observed on a GPU: attention
  softmax denominators and attention-weighted sums are computed as
  **split/tiled reductions over the padded key axis** (tile = 16), with
  bf16-rounded partials combined in order. A stock NumPy GEMM is
  row-independent, so it cannot show the effect; the tiled reduction is
  what GPU kernels do (split-K / tile decomposition), and it is the
  mechanism the paper's argument rests on. All drift numbers are measured
  from this model, on real bf16 arithmetic — they are simulation numbers,
  not GPU numbers, and the README says so.
- Why left padding: the paper's Figure 3(a) has the target's *padding
  offset* shifting with its companions. With right padding, a tiled
  reduction over the key axis groups the target's real keys identically
  whatever the padded length (extra zero terms do not change a float sum),
  so no drift appears. With left padding, the target's real keys shift
  across tile boundaries as the padded length changes, which is exactly
  the shape-dependent regrouping that changes bf16 rounding. Rows fixes
  the width W, Pack fixes the terminal interval — both pin the grouping.
- Policies implemented: Padded (left-padded `[B, T_max]`), BlockLock-Rows
  (`[B, W]`, W a multiple of block L = 8 at our scale), BlockLock-Pack
  (`[1, P = BW]`, segment-causal mask, dummy segment, KV extraction,
  target-only decode), CoRun-style (isolated B = 1 prefill, fixed-envelope
  decode).
- Greedy decode, 16 steps unless the extension varies it. Agreement is the
  paper's definition: fraction of repetition pairs with identical generated
  sequences; Top-20 agreement requires identical token IDs *and*
  bitwise-identical raw logits at every step.
- Cost is a token-work model (positions processed + attention work),
  normalised to Padded = 1.00, plus measured NumPy CPU wall-clock for the
  same phases (labelled as CPU wall-clock, not GPU latency).

## Bugs / design fixes during the night

1. **bf16 test expectation, not code.** A pilot assertion expected
   `to_bf16(1 + 3·2⁻⁹) == 1 + 2⁻⁸`; the value is exactly halfway between
   `1 + 2⁻⁸` (odd mantissa) and `1 + 2⁻⁷` (even), so round-to-nearest-even
   correctly returns `1 + 2⁻⁷`. Test fixed; the bit-trick rounding was
   right.
2. **`tiled_weighted_sum` axis mismatch.** The first version assumed
   weights and values share a reduction axis; attention weights reduce
   over −1 and values over −2. Rewritten with per-chunk matmuls.
3. **Flip metric conflated two effects.** The first run compared every
   policy against a bf16 B = 1 canonical, so Rows/Pack showed a 0.062–
   0.125 "flip rate" that was really the one-time geometry shift from
   isolated to batched execution, and the fp32 control showed 0.5625
   because it was compared against a bf16 canonical. Fixed: canonical
   per dtype, metric renamed `flip_vs_isolated`, and the per-deployment
   reading documented in the README.
4. **E4 measured the wrong thing first.** The row-1 companion varied
   across repetitions, so its tokens differed trivially (agreement 0.0
   for both policies). Fixed: hold the row-1 companion fixed, vary only
   the other companions.
5. **`row_generated` aliasing.** The target-only branch aliased the
   `generated` list, which would have double-appended. Fixed with a copy.
6. **First-step drift is usually zero — a property, not a bug.** At
   tile 16, prefill drift needs the sequence to span two reduction tiles
   (padded length > 16); most prompts fit in one tile, so prefill logits
   are bitwise identical and drift appears only during decode as the
   cache grows across the boundary. Confirmed by E5 (tile 4/8 show
   first-step drift, tile 32 shows almost none) and documented in the
   README/E1. All pre-fix numbers from the first experiment run were
   discarded after the flip-metric fix; the shipped results are from the
   corrected rerun (identical seeds, 48 s).
7. **Cost model caveat.** Positions processed understate CoRun's latency
   (serial prefill) and overstate nothing else; the README reports
   positions and attention work separately and quotes the paper's latency
   numbers apart from them.
