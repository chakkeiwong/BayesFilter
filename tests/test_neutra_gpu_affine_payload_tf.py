from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import pytest

from bayesfilter.ssm import stable_ssm_posterior_adapter_signature, stable_ssm_target_signature
from bayesfilter.testing.lgssm_generic_target_adapter_tf import (
    make_lgssm_generic_target_fixture,
)
from bayesfilter.testing.neutra_gpu_affine_payload_tf import (
    EXPECTED_PHASE16_ADAPTER_SIGNATURE,
    EXPECTED_PHASE16_TARGET_SIGNATURE,
    EXPECTED_PHASE16_TRAINING_STATE_FILE_SHA256,
    PHASE16_TRAINING_STATE_PATH,
    PHASE17_PAYLOAD_FILENAME,
    PHASE17_VALIDATION_FILENAME,
    NeuTraGPUAffinePayloadConfig,
    NeuTraGPUAffinePayloadError,
    package_and_validate_gpu_affine_payload,
)
from bayesfilter.testing.neutra_gpu_bounded_training_tf import DEFAULT_PHASE16_ARTIFACT_DIR


OLD_PHASE10_TRAINING_STATE_PATH = DEFAULT_PHASE16_ARTIFACT_DIR / (
    "lgssm_static_qr_exact_kalman_affine_neutra_gpu_training_state_seed20260707.json"
)


def _phase16_state_or_skip() -> dict:
    if not PHASE16_TRAINING_STATE_PATH.exists():
        pytest.skip("Phase 16 trusted GPU/XLA training-state artifact is unavailable")
    payload = json.loads(PHASE16_TRAINING_STATE_PATH.read_text(encoding="utf-8"))
    if payload.get("passed") is not True:
        pytest.skip("Phase 16 trusted GPU/XLA pass artifact is unavailable")
    return payload


def test_phase17_default_config_uses_phase16_xla_source_and_outputs() -> None:
    config = NeuTraGPUAffinePayloadConfig()
    payload = config.normalized()

    assert config.phase16_training_state_path == PHASE16_TRAINING_STATE_PATH
    assert config.artifact_dir == DEFAULT_PHASE16_ARTIFACT_DIR
    assert config.payload_path.name == PHASE17_PAYLOAD_FILENAME
    assert config.validation_path.name == PHASE17_VALIDATION_FILENAME
    assert "gpu_xla_training_state" in payload["phase16_training_state_path"]
    assert "gpu_xla_frozen_payload" in payload["payload_path"]
    assert "gpu_xla_payload_validation" in payload["validation_path"]
    assert payload["phase"] == "phase17_frozen_gpu_xla_trained_affine_payload"
    assert payload["training_execution_target"] == "not_run_phase16_gpu_xla_artifact_only"
    assert payload["packaging_execution_target"] == "cpu_hidden_artifact_loader_reference"
    assert payload["hmc_policy"] == "not_run_mechanics_compile_deferred_to_phase18"


def test_phase17_packages_phase16_xla_state_and_records_boundaries(
    tmp_path: Path,
) -> None:
    state = _phase16_state_or_skip()

    result = package_and_validate_gpu_affine_payload(
        NeuTraGPUAffinePayloadConfig(
            phase16_training_state_path=PHASE16_TRAINING_STATE_PATH,
            artifact_dir=tmp_path,
        )
    )

    assert result.passed is True
    assert result.payload_path.name == PHASE17_PAYLOAD_FILENAME
    assert result.validation_path.name == PHASE17_VALIDATION_FILENAME
    assert result.phase16_training_state["artifact_hash"] == state["artifact_hash"]

    payload = json.loads(result.payload_path.read_text(encoding="utf-8"))
    validation = json.loads(result.validation_path.read_text(encoding="utf-8"))
    assert payload["phase"] == "phase17_frozen_gpu_xla_trained_affine_payload"
    assert payload["source_training_state_path"] == str(PHASE16_TRAINING_STATE_PATH)
    assert payload["training_executed"] is False
    assert payload["hmc_executed"] is False
    assert payload["external_sample_generation_executed"] is False
    assert payload["source_jit_compile"] is True
    assert payload["fixed_transport_hmc_mechanics_executed"] is False
    assert payload["fixed_transport_hmc_mechanics_deferred_to_phase18"] is True
    assert payload["jit_compile_runtime_executed"] is False
    assert payload["jit_compile_false_runtime_executed"] is False
    assert validation["decision"] == "PASS_PHASE17_FROZEN_GPU_XLA_AFFINE_PAYLOAD"
    assert validation["source_training_state_path"] == str(PHASE16_TRAINING_STATE_PATH)
    assert validation["source_training_state_file_sha256"] == (
        EXPECTED_PHASE16_TRAINING_STATE_FILE_SHA256
    )
    assert validation["target_signature"] == EXPECTED_PHASE16_TARGET_SIGNATURE
    assert validation["adapter_signature"] == EXPECTED_PHASE16_ADAPTER_SIGNATURE
    assert validation["boundary_checks"]["source_jit_compile_true"] is True
    assert validation["boundary_checks"]["fixed_transport_hmc_mechanics_executed"] is False
    assert (
        validation["boundary_checks"]["fixed_transport_hmc_mechanics_deferred_to_phase18"]
        is True
    )
    assert validation["boundary_checks"]["jit_compile_runtime_executed"] is False
    assert validation["boundary_checks"]["jit_compile_false_runtime_executed"] is False
    assert validation["training_executed"] is False
    assert validation["hmc_executed"] is False
    assert validation["external_sample_generation_executed"] is False
    assert validation["mechanics_manifest"] is None
    assert validation["mechanics_value"] is None
    assert validation["mechanics_score"] is None


def test_phase17_old_phase10_gpu_state_is_stale_after_manual_score_policy(
    tmp_path: Path,
) -> None:
    if not OLD_PHASE10_TRAINING_STATE_PATH.exists():
        pytest.skip("Old Phase 10 trusted GPU training-state artifact is unavailable")
    state = json.loads(OLD_PHASE10_TRAINING_STATE_PATH.read_text(encoding="utf-8"))
    if state.get("passed") is not True:
        pytest.skip("Old Phase 10 trusted GPU pass artifact is unavailable")

    fixture = make_lgssm_generic_target_fixture()
    current_target_signature = stable_ssm_target_signature(fixture.contract)
    current_adapter_signature = stable_ssm_posterior_adapter_signature(fixture.adapter)

    assert current_target_signature == EXPECTED_PHASE16_TARGET_SIGNATURE
    assert current_adapter_signature == EXPECTED_PHASE16_ADAPTER_SIGNATURE
    assert state["target_signature"] != EXPECTED_PHASE16_TARGET_SIGNATURE
    assert state["adapter_signature"] != EXPECTED_PHASE16_ADAPTER_SIGNATURE
    assert state["config"]["jit_compile"] is False

    with pytest.raises(NeuTraGPUAffinePayloadError, match="file sha256 mismatch"):
        package_and_validate_gpu_affine_payload(
            NeuTraGPUAffinePayloadConfig(
                phase16_training_state_path=OLD_PHASE10_TRAINING_STATE_PATH,
                artifact_dir=tmp_path,
            )
        )


def test_phase17_rejects_unhidden_cpu_policy(tmp_path: Path, monkeypatch) -> None:
    _phase16_state_or_skip()
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    with pytest.raises(NeuTraGPUAffinePayloadError, match="CUDA_VISIBLE_DEVICES=-1"):
        package_and_validate_gpu_affine_payload(
            NeuTraGPUAffinePayloadConfig(artifact_dir=tmp_path)
        )


def test_phase17_rejects_target_signature_mismatch(tmp_path: Path) -> None:
    state = _phase16_state_or_skip()
    bad_state = dict(state)
    bad_state["schema"] = "bayesfilter.neutra.gpu_xla_bounded_training_state.v1"
    bad_state["phase"] = "phase16_bounded_gpu_xla_neutra_training"
    bad_state["decision"] = "PASS_PHASE16_BOUNDED_GPU_XLA_NEUTRA_TRAINING"
    bad_state["adapter_signature"] = EXPECTED_PHASE16_ADAPTER_SIGNATURE
    bad_state["target_signature"] = "0" * 64
    bad_state_path = tmp_path / "bad-phase16-state.json"
    bad_state_path.write_text(json.dumps(bad_state), encoding="utf-8")

    bad_state_hash = hashlib.sha256(bad_state_path.read_bytes()).hexdigest()

    with pytest.raises(NeuTraGPUAffinePayloadError, match="target_signature"):
        package_and_validate_gpu_affine_payload(
            NeuTraGPUAffinePayloadConfig(
                phase16_training_state_path=bad_state_path,
                artifact_dir=tmp_path,
                training_state_file_sha256=bad_state_hash,
            )
        )
