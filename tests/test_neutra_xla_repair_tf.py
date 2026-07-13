from __future__ import annotations

import json
import os

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

from bayesfilter.testing.neutra_xla_repair_tf import (
    NEUTRA_XLA_REPAIR_NONCLAIMS,
    NeuTraXLARepairConfig,
    NeuTraXLARepairError,
    XLA_REPAIR_ROUTE_LEGACY_AUTODIFF,
    XLA_REPAIR_ROUTE_LGSSM_WHILE_LOOP,
    phase13_error_payload,
    run_neutra_xla_repair_diagnostic,
)


def test_phase15_config_records_xla_compile_only_boundary(tmp_path) -> None:
    config = NeuTraXLARepairConfig(output_path=tmp_path / "xla.json")
    payload = config.normalized()

    assert payload["phase"] == "phase15_manual_score_xla_compile_gate"
    assert payload["route"] == XLA_REPAIR_ROUTE_LGSSM_WHILE_LOOP
    assert payload["jit_compile"] is True
    assert payload["jit_compile_false_runtime_allowed"] is False
    assert payload["training_execution_target"] == (
        "not_training_xla_compile_diagnostic_only"
    )
    assert payload["optimizer_update_executed"] is False
    assert payload["cpu_training_fallback_policy"] == "forbidden"
    assert payload["external_sample_generation_policy"] == (
        "not_run_phase12_separate_boundary"
    )
    assert payload["hmc_policy"] == "not_run_not_authorized_for_phase15"
    assert payload["nonclaims"] == NEUTRA_XLA_REPAIR_NONCLAIMS


def test_phase15_rejects_cpu_hidden_gpu_xla_diagnostic(tmp_path) -> None:
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"

    with pytest.raises(NeuTraXLARepairError, match="CUDA_VISIBLE_DEVICES=-1"):
        run_neutra_xla_repair_diagnostic(
            NeuTraXLARepairConfig(output_path=tmp_path / "xla.json")
        )


def test_phase15_rejects_legacy_autodiff_route(tmp_path) -> None:
    with pytest.raises(NeuTraXLARepairError, match="unknown Phase 15 route"):
        run_neutra_xla_repair_diagnostic(
            NeuTraXLARepairConfig(
                route=XLA_REPAIR_ROUTE_LEGACY_AUTODIFF,
                output_path=tmp_path / "xla.json",
            )
        )


def test_phase15_error_payload_preserves_nonclaims(tmp_path) -> None:
    config = NeuTraXLARepairConfig(output_path=tmp_path / "xla.json")
    payload = phase13_error_payload(RuntimeError("example"), config=config)

    assert payload["passed"] is False
    assert payload["decision"] == "BLOCK_PHASE15_MANUAL_SCORE_XLA_COMPILE_GATE"
    assert payload["hmc_executed"] is False
    assert payload["training_executed"] is False
    assert payload["optimizer_update_executed"] is False
    assert payload["external_sample_generation_executed"] is False
    assert payload["jit_compile"] is True
    assert payload["jit_compile_false_runtime_executed"] is False
    assert payload["nonclaims"] == NEUTRA_XLA_REPAIR_NONCLAIMS


def test_phase15_artifact_if_present_preserves_boundary() -> None:
    path = (
        "docs/plans/"
        "bayesfilter-lgssm-first-neutra-hmc-phase15-manual-score-xla-compile-"
        "diagnostic-2026-07-08.json"
    )
    if not os.path.exists(path):
        pytest.skip("Phase 15 trusted GPU/XLA artifact has not been generated yet")
    payload = json.loads(open(path, encoding="utf-8").read())

    assert payload["schema"] == "bayesfilter.neutra.manual_score_xla_compile_result.v1"
    assert payload["phase"] == "phase15_manual_score_xla_compile_gate"
    assert payload["config"]["jit_compile"] is True
    assert payload["config"]["jit_compile_false_runtime_allowed"] is False
    assert payload["hmc_executed"] is False
    assert payload["training_executed"] is False
    assert payload["optimizer_update_executed"] is False
    assert payload["external_sample_generation_executed"] is False
    assert payload["jit_compile_false_runtime_executed"] is False
    assert payload["nonclaims"] == list(NEUTRA_XLA_REPAIR_NONCLAIMS)
    if payload["passed"]:
        assert payload["decision"] == "PASS_PHASE15_MANUAL_SCORE_XLA_COMPILE_GATE"
        assert payload["route_result"]["route"] == XLA_REPAIR_ROUTE_LGSSM_WHILE_LOOP
        assert payload["route_result"]["jit_compile"] is True
        assert payload["route_result"]["jit_compile_false_runtime_executed"] is False
        assert payload["route_result"]["finite_checks"] == {
            "first_loss_finite": True,
            "first_raw_scale_gradient_finite": True,
            "first_shift_gradient_finite": True,
            "second_loss_finite": True,
            "second_raw_scale_gradient_finite": True,
            "second_shift_gradient_finite": True,
        }
        assert payload["route_result"]["device_checks"]["all_outputs_on_gpu"] is True
        assert payload["route_result"]["first_call_wall_seconds"] >= 0.0
        assert payload["route_result"]["second_call_wall_seconds"] >= 0.0
        assert payload["route_result"]["compile_time_proxy_seconds"] >= 0.0
    else:
        assert payload["decision"] == "BLOCK_PHASE15_MANUAL_SCORE_XLA_COMPILE_GATE"
