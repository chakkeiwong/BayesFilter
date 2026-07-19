from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = (
    ROOT
    / "docs/benchmarks/"
    "run_ssl_lstm_neutra_dsge_parity_material_training_2026_07_15.py"
)


def _load_runner():
    name = "ssl_lstm_neutra_dsge_parity_material_training_runner"
    spec = importlib.util.spec_from_file_location(name, RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


runner = _load_runner()


def test_seed_roles_are_prospectively_independent() -> None:
    assert [spec.label for spec in runner.SEEDS] == ["seed-a", "seed-b"]
    roles = [
        seed
        for spec in runner.SEEDS
        for seed in (
            spec.initialization_seed,
            spec.training_seed,
            spec.validation_seed,
        )
    ]
    assert len(roles) == len(set(roles)) == 6
    assert (20260715, 4099) not in roles
    assert (20260714, 3301) not in roles
    runner._assert_seed_contract()


def test_material_config_uses_only_strict_source_parity_preset() -> None:
    class Target:
        def target_signature(self):
            return "a" * 64

        def adapter_signature(self):
            return "b" * 64

    config = runner._config(Target(), runner.SEEDS[0])
    assert config.family == runner.DSGE_PAPER_NEUTRA_FAMILY
    assert config.hidden_layers == (4, 4)
    assert config.activation == "elu"
    assert config.stages == 3
    assert config.fixed_translation == tuple(runner.PRIOR_CENTER_VALUES)
    assert config.learning_rate == pytest.approx(0.01)
    assert config.learning_rate_schedule == "paper_piecewise"
    assert config.epsilon == pytest.approx(1.0e-7)
    assert config.gradient_clip_mode == "per_variable"
    assert config.gradient_clip_norm == pytest.approx(10.0)
    assert config.jit_compile is True
    source = RUNNER_PATH.read_text(encoding="utf-8")
    assert "NeuTraTrainerConfig(" not in source
    assert "--learning-rate" not in source
    assert "--steps" not in source


def _validation(*, saturation=0.0):
    return {"saturation_fraction": float(saturation)}


def _probes(**overrides):
    values = {
        "all_finite": True,
        "roundtrip_max_abs": 1.0e-12,
        "original_neighborhood_max_inverse_radius": 3.0,
        "moderate_shell_max_inverse_radius": 4.0,
    }
    values.update(overrides)
    return values


def test_candidate_decision_separates_hard_and_promotion_vetoes() -> None:
    decision, hard, promotion = runner.candidate_decision(
        final_validation=_validation(),
        loss_interval={"one_sided_95_upper": -0.1},
        probes=_probes(),
        reload_exact=True,
    )
    assert decision == "VIABLE_FROZEN_CANDIDATE"
    assert hard == []
    assert promotion == []

    decision, hard, promotion = runner.candidate_decision(
        final_validation=_validation(saturation=0.1),
        loss_interval={"one_sided_95_upper": 0.1},
        probes=_probes(
            roundtrip_max_abs=1.0e-3,
            original_neighborhood_max_inverse_radius=5.0,
        ),
        reload_exact=False,
    )
    assert decision == "INVALID_HARD_VETO"
    assert "frozen_reload_mismatch" in hard
    assert "roundtrip_residual_above_threshold" in promotion
    assert "original_neighborhood_missing_support" in promotion
    assert "heldout_loss_improvement_not_established" in promotion
    assert "dense_scale_saturation_above_cap" in promotion


def test_seed_a_candidate_rejection_does_not_suppress_seed_b() -> None:
    assert runner.should_continue_after_candidate("VIABLE_FROZEN_CANDIDATE") is True
    assert runner.should_continue_after_candidate("CANDIDATE_NOT_NOMINATED") is True
    assert runner.should_continue_after_candidate("INVALID_HARD_VETO") is False
    with pytest.raises(ValueError, match="unknown candidate"):
        runner.should_continue_after_candidate("DESCRIPTIVELY_WORSE")


def test_shared_budget_reserves_a_complete_next_seed() -> None:
    assert runner.PER_SEED_GPU_SECONDS == 18_000.0
    assert runner.SHARED_GPU_SECONDS == 36_000.0
    assert runner.can_start_next_seed(17_999.9) is True
    assert runner.can_start_next_seed(18_000.0) is True
    assert runner.can_start_next_seed(18_000.1) is False
    with pytest.raises(ValueError, match="finite and nonnegative"):
        runner.can_start_next_seed(float("nan"))


def test_program_classification_is_fail_closed() -> None:
    pass_row = {"decision": "VIABLE_FROZEN_CANDIDATE"}
    reject_row = {"decision": "CANDIDATE_NOT_NOMINATED"}
    hard_row = {"decision": "INVALID_HARD_VETO"}
    assert runner.classify_program([pass_row, pass_row], 35_999.0) == (
        True,
        "TWO_INDEPENDENT_CANDIDATES_NOMINATED",
    )
    assert runner.classify_program([pass_row, reject_row], 35_999.0) == (
        True,
        "SEED_INSTABILITY_REPAIR_REQUIRED",
    )
    assert runner.classify_program([reject_row, reject_row], 35_999.0) == (
        True,
        "SOURCE_MATCHED_CANDIDATE_REJECTED_UNDER_DECLARED_GATES",
    )
    for rows, wall in (
        ([pass_row], 100.0),
        ([pass_row, hard_row], 100.0),
        ([pass_row, pass_row], 36_000.1),
    ):
        assert runner.classify_program(rows, wall) == (
            False,
            "PROGRAM_STOPPED_BY_CONTINUATION_VETO",
        )
    with pytest.raises(ValueError, match="finite and nonnegative"):
        runner.classify_program([], float("nan"))


def test_program_root_must_be_fresh(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    output = Path("material")
    (tmp_path / output).mkdir()
    (tmp_path / output / "existing.json").write_text("{}", encoding="utf-8")
    with pytest.raises(runner.MaterialTrainingError, match="not fresh"):
        runner._require_fresh_root(output)


def test_resource_stop_preserves_exact_latest_state(tmp_path, monkeypatch) -> None:
    class Step:
        def numpy(self):
            return 137

    class Trainer:
        step = Step()

        def state_payload(self):
            return {"schema": "state", "step": 137}

    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(
        runner,
        "_check_budget",
        lambda **_kwargs: (_ for _ in ()).throw(runner.ResourceStop("cap")),
    )
    output = Path("candidate")
    with pytest.raises(runner.ResourceStop, match="cap"):
        runner._check_budget_and_preserve_state(
            Trainer(),
            output_dir=output,
            candidate_started=0.0,
            program_started=0.0,
        )
    payload = (tmp_path / output / "resource-stop-state-0137.json").read_text(
        encoding="utf-8"
    )
    assert '"step":137' in payload


def test_failure_and_result_claim_boundaries_are_explicit() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    assert "candidate_rejection_is_research_direction_rejection" in source
    assert "statistically_supported_ranking" in source
    assert "failure_receipt_only_gpu_provenance_not_established" in source
    assert "per_seed_gpu_cap_overrun" in source
    assert "shared_gpu_cap_overrun" in source
    assert "no posterior, HMC, predictive, superiority, or readiness claim" in source
