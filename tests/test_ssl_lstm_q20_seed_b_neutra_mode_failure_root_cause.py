from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / (
    "docs/benchmarks/"
    "run_ssl_lstm_q20_seed_b_neutra_mode_failure_root_cause_2026_08_10.py"
)
PLAN = ROOT / (
    "docs/plans/"
    "bayesfilter-ssl-lstm-q20-seed-b-neutra-mode-failure-root-cause-plan-2026-08-10.md"
)


def _module():
    spec = importlib.util.spec_from_file_location("neutra_mode_root_cause", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_numeric_contract_and_xla_cpu_diagnostic_boundaries() -> None:
    module = _module()
    source = RUNNER.read_text(encoding="utf-8")
    assert module.FLOW_SAMPLE_COUNT == 100_000
    assert module.PATH_POINT_COUNT == 65
    assert module.CANARY_CHAINS_PER_REGION == 16
    assert module.CANARY_TRANSITIONS == 8
    assert module.MATERIAL_CHAINS_PER_REGION == 32
    assert module.MATERIAL_TRANSITIONS == 64
    assert module.STATIONARY_CHAINS_PER_REGION == 8
    assert module.STATIONARY_TRANSITIONS == 4
    assert module.STATIONARY_CONTROL_STEP_SIZE == 0.1
    assert source.index('os.environ["CUDA_VISIBLE_DEVICES"] = "-1"') < source.index(
        "from bayesfilter.inference.neutra_artifacts import"
    )
    assert "jit_compile=True" in source
    assert 'kernel.get("mass_policy") != "fixed_identity_z"' in source
    assert "ORIGINAL_TARGET_SCOPE" in source
    assert "ORIGINAL_VALIDATION_PLAN" in source
    assert "_archived_value_score_compatibility" in source
    assert "PARITY_VALUE_TOLERANCE = 5.0e-7" in source
    assert "PARITY_SCORE_TOLERANCE = 5.0e-7" in source
    assert '"stationary-canary"' in source
    assert '"stationary-step-control"' in source
    assert '"curvature_derived_explanatory_control"' in source
    assert '"proposed_target_log_prob"' in source


def test_zero_count_bound_is_exact_and_not_posterior_mass() -> None:
    module = _module()
    bound = module.zero_count_upper_bound(100_000)
    assert bound == pytest.approx(1.0 - 0.05 ** (1.0 / 100_000), rel=0.0, abs=1e-15)
    assert bound == pytest.approx(2.9956874e-5, rel=1e-6)
    with pytest.raises(ValueError):
        module.zero_count_upper_bound(0)
    source = RUNNER.read_text(encoding="utf-8")
    assert '"role": "learned_reverse_kl_proposal_q_phi_only"' in source
    assert '"nonclaim": "not posterior basin mass"' in source


def test_four_dimensional_kinetic_survival_formula() -> None:
    module = _module()
    assert module.identity_mass_kinetic_survival_4d(0.0) == 1.0
    assert module.identity_mass_kinetic_survival_4d(10.0) == pytest.approx(
        math.exp(-10.0) * 11.0
    )
    with pytest.raises(ValueError):
        module.identity_mass_kinetic_survival_4d(-1.0)


def test_optimizer_metadata_accepts_scalar_and_batched_tensors() -> None:
    module = _module()

    class FakeTF:
        @staticmethod
        def convert_to_tensor(value):
            return value

        @staticmethod
        def reshape(value, _shape):
            return value

    class Scalar:
        class Shape:
            rank = 0

        shape = Shape()

    class Vector(list):
        class Shape:
            rank = 1

        shape = Shape()

    scalar = Scalar()
    assert module.optimizer_result_item(FakeTF, scalar, 1) is scalar
    assert module.optimizer_result_item(FakeTF, Vector([3, 5]), 1) == 5


def test_representative_selection_is_score_and_sign_based() -> None:
    module = _module()
    rows = [
        {"position": [0.0, 0.0, 0.5, 0.0], "log_prob": -2.0, "score_inf_norm": 1e-7, "start_index": 1},
        {"position": [0.0, 0.0, 0.6, 0.0], "log_prob": -1.0, "score_inf_norm": 1e-7, "start_index": 2},
        {"position": [0.0, 0.0, -0.5, 0.0], "log_prob": -1.5, "score_inf_norm": 1e-7, "start_index": 3},
        {"position": [0.0, 0.0, -0.7, 0.0], "log_prob": 0.0, "score_inf_norm": 1e-2, "start_index": 4},
    ]
    selected = module.select_sign_representatives(rows)
    assert selected["plus"]["start_index"] == 2
    assert selected["minus"]["start_index"] == 3


def test_transition_summary_counts_directions_from_initial_state() -> None:
    module = _module()
    result = module.transition_summary(0, [0, 1, 1, 0, 1])
    assert result["plus_to_minus"] == 2
    assert result["minus_to_plus"] == 1
    assert result["any_opposite_region"] is True
    assert result["terminal_sign"] == 1


def test_plan_audit_does_not_overclaim_paths_or_laplace() -> None:
    text = PLAN.read_text(encoding="utf-8")
    assert "PASS AFTER TWO REPAIRS" in text
    assert "finite grid\n   can miss a narrow peak" in text
    assert "heuristics with no bound claim" in text
    assert "local Laplace" in text
    assert "does not create an independent posterior authority" in text
    assert "Exact posterior weights remain unanswered" in text
