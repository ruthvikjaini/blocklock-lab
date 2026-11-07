"""Audit metrics and the Section 3.4 contract check."""

import numpy as np

from blocklock_lab import audit, model, policies


def _fake_gen(tokens, logits):
    class G:
        pass
    g = G()
    g.tokens = np.asarray(tokens)
    g.logits_trace = np.asarray(logits, dtype=np.float64)
    return g


def test_pairwise_agreement_identical():
    g = _fake_gen([1, 2, 3], [[0.1, 0.5, 0.2], [0.3, 0.1, 0.9]])
    seq, topk, maxd = audit.pairwise_agreement([g, g, g])
    assert seq == 1.0 and topk == 1.0 and maxd == 0.0


def test_pairwise_agreement_detects_flip():
    g1 = _fake_gen([1, 2, 3], [[0.1, 0.5], [0.3, 0.9]])
    g2 = _fake_gen([1, 2, 4], [[0.1, 0.5], [0.3, 0.9]])
    seq, _, _ = audit.pairwise_agreement([g1, g2])
    assert seq == 0.0


def test_top2_margin():
    assert audit.top2_margin([1.0, 3.0, 2.0]) == 1.0
    assert audit.top2_margin([5.0, 5.0, 1.0]) == 0.0


def test_first_step_delta_zero_when_identical():
    g = _fake_gen([1, 2], [[0.1, 0.5], [0.3, 0.9]])
    assert audit.first_step_delta([g, g]) == 0.0


def test_contract_check_rows():
    params = model.init_params(seed=0)
    rng = np.random.default_rng(9)
    target = [int(x) for x in rng.integers(0, 128, size=12)]
    width = policies.class_width(len(target))
    comps_a = [[int(x) for x in rng.integers(0, 128, size=10)] for _ in range(3)]
    comps_b = [[int(x) for x in rng.integers(0, 128, size=14)] for _ in range(3)]
    spec_a = policies.build_rows(target, comps_a, width)
    spec_b = policies.build_rows(target, comps_b, width)
    check = audit.contract_check(params, spec_a, spec_b)
    assert check["c1_geometry"] is True
    assert check["c2_repeat_stable"] is True


def test_contract_check_padded_geometry_varies():
    params = model.init_params(seed=0)
    rng = np.random.default_rng(10)
    target = [int(x) for x in rng.integers(0, 128, size=12)]
    comps_a = [[int(x) for x in rng.integers(0, 128, size=10)] for _ in range(3)]
    comps_b = [[int(x) for x in rng.integers(0, 128, size=20)] for _ in range(3)]
    spec_a = policies.build_padded(target, comps_a)
    spec_b = policies.build_padded(target, comps_b)
    check = audit.contract_check(params, spec_a, spec_b)
    assert check["c1_geometry"] is False  # widths 12 vs 20
    assert check["c2_repeat_stable"] is True
