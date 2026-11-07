"""Policies: the fix must be bitwise exact; the baseline must drift."""

import numpy as np

from blocklock_lab import audit, model, policies

VOCAB = 128


def _setup(seed=5):
    params = model.init_params(seed=0)
    rng = np.random.default_rng(seed)
    # Target length 12 in class width 16: companion lengths 8..16 make the
    # padded width vary across sets, so the geometry (and hence the bf16
    # rounding) varies too. A 16-token target would pin T_max at 16 and
    # show no drift at all.
    target = [int(x) for x in rng.integers(0, VOCAB, size=12)]
    width = policies.class_width(len(target))
    comp_sets = []
    for _ in range(6):
        comps = [[int(x) for x in rng.integers(0, VOCAB, size=int(rng.integers(8, width + 1)))]
                 for _ in range(3)]
        comp_sets.append(comps)
    return params, target, width, comp_sets


def _run(params, target, width, comp_sets, policy, dtype="bf16"):
    gens = []
    for comps in comp_sets:
        kw = {"width": width} if policy in ("rows", "pack") else {}
        spec = policies.BUILDERS[policy](target, comps, **kw)
        gens.append(policies.generate(params, spec, n_steps=16, dtype=dtype))
    return gens


def test_rows_pack_corun_bitwise_identical():
    params, target, width, comp_sets = _setup()
    for policy in ("rows", "pack", "corun"):
        gens = _run(params, target, width, comp_sets, policy)
        seq_ag, topk_ag, maxd = audit.pairwise_agreement(gens)
        assert seq_ag == 1.0, policy
        assert topk_ag == 1.0, policy
        assert maxd == 0.0, policy


def test_padded_drifts_in_bf16():
    params, target, width, comp_sets = _setup()
    gens = _run(params, target, width, comp_sets, "padded")
    _, topk_ag, maxd = audit.pairwise_agreement(gens)
    # Logits must differ bitwise across companion sets (the problem exists).
    assert topk_ag < 1.0
    assert maxd > 0.0


def test_padded_barely_drifts_in_fp32():
    params, target, width, comp_sets = _setup()
    gens = _run(params, target, width, comp_sets, "padded", dtype="fp32")
    seq_ag, _, maxd = audit.pairwise_agreement(gens)
    assert seq_ag == 1.0
    assert maxd < 1e-5


def test_pack_target_anchored_at_end():
    params, target, width, comp_sets = _setup()
    comps = comp_sets[0]
    spec = policies.build_pack(target, comps, width)
    assert spec.width == 4 * width
    assert spec.target_last_index == spec.width - 1
    assert spec.extra["target_len"] == len(target)
    # The dummy segment absorbs exactly the slack.
    used = sum(len(c) for c in comps) + len(target)
    assert spec.extra["dummy_len"] == spec.width - used


def test_pack_kv_extraction_matches_standalone():
    params, target, width, comp_sets = _setup()
    spec = policies.build_pack(target, comp_sets[0], width)
    check = audit.contract_check(params, spec, spec)
    assert check["c3_handoff"] is True
    assert check["c1_geometry"] is True
    assert check["c2_repeat_stable"] is True


def test_class_width():
    assert policies.class_width(1) == 8
    assert policies.class_width(8) == 8
    assert policies.class_width(9) == 16
    assert policies.class_width(16) == 16
    assert policies.class_width(17) == 24


def test_generate_returns_n_steps():
    params, target, width, comp_sets = _setup()
    spec = policies.build_padded(target, comp_sets[0])
    gen = policies.generate(params, spec, n_steps=16, dtype="bf16")
    assert gen.tokens.shape == (16,)
    assert gen.logits_trace.shape == (16, VOCAB)
    assert len(gen.row_tokens) == 4  # target + 3 companions
