from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from bayesfilter.testing.neutra_gpu_bounded_training_tf import (
    DEFAULT_PHASE10_ARTIFACT_DIR,
    LGSSM_QR_ROUTE_ID,
    NEUTRA_GPU_BOUNDED_TRAINING_NONCLAIMS,
    NeuTraGPUBoundedTrainingConfig,
    NeuTraGPUBoundedTrainingError,
    phase10_error_payload,
    run_neutra_gpu_bounded_training,
)
from bayesfilter.testing.neutra_gpu_training_preflight_tf import (
    MODEL_B_SVD_UKF_ROUTE_ID,
    SIMPLE_NONLINEAR_SVD_CUT4_FILTER_ID,
)


PHASE10_TRAINING_STATE = DEFAULT_PHASE10_ARTIFACT_DIR / (
    "lgssm_static_qr_exact_kalman_affine_neutra_gpu_training_state_seed20260707.json"
)


def test_phase10_config_records_gpu_training_policy(tmp_path) -> None:
    config = NeuTraGPUBoundedTrainingConfig(artifact_dir=tmp_path)
    payload = config.normalized()

    assert payload["schema"] == "bayesfilter.neutra.gpu_xla_bounded_training_config.v1"
    assert payload["phase"] == "phase16_bounded_gpu_xla_neutra_training"
    assert payload["route_id"] == LGSSM_QR_ROUTE_ID
    assert payload["training_execution_target"] == "gpu_required"
    assert payload["cpu_training_fallback_policy"] == "forbidden"
    assert payload["external_sample_generation_policy"] == (
        "multicore_cpu_separate_phase_not_run_here"
    )
    assert payload["hmc_policy"] == "not_run_not_authorized_for_phase16"
    assert payload["jit_compile"] is True
    assert payload["xla_readiness_policy"] == "jit_compile_true_required_no_fallback"
    assert payload["gradient_policy"] == "manual_score_no_gradienttape"
    assert payload["full_neutra_training_executed"] is False
    assert payload["bounded_optimizer_training_executed"] is True
    assert payload["hmc_executed"] is False
    assert payload["external_sample_generation_executed"] is False
    assert payload["nonclaims"] == NEUTRA_GPU_BOUNDED_TRAINING_NONCLAIMS
    assert "gpu_xla_training_state" in payload["training_state_path"]


def test_phase10_rejects_cpu_hidden_training(tmp_path) -> None:
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"

    with pytest.raises(NeuTraGPUBoundedTrainingError, match="CUDA_VISIBLE_DEVICES=-1"):
        run_neutra_gpu_bounded_training(
            NeuTraGPUBoundedTrainingConfig(artifact_dir=tmp_path)
        )


def test_phase10_rejects_non_xla_runtime(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    with pytest.raises(NeuTraGPUBoundedTrainingError, match="jit_compile=true"):
        run_neutra_gpu_bounded_training(
            NeuTraGPUBoundedTrainingConfig(
                jit_compile=False,
                artifact_dir=tmp_path,
            )
        )


def test_phase10_rejects_non_lgssm_admitted_route(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    with pytest.raises(NeuTraGPUBoundedTrainingError, match="selects only the LGSSM"):
        run_neutra_gpu_bounded_training(
            NeuTraGPUBoundedTrainingConfig(
                route_id=MODEL_B_SVD_UKF_ROUTE_ID,
                artifact_dir=tmp_path,
            )
        )


def test_phase10_rejects_deferred_route(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    with pytest.raises(NeuTraGPUBoundedTrainingError, match="route is deferred"):
        run_neutra_gpu_bounded_training(
            NeuTraGPUBoundedTrainingConfig(
                route_id=SIMPLE_NONLINEAR_SVD_CUT4_FILTER_ID,
                artifact_dir=tmp_path,
            )
        )


def test_phase10_error_payload_preserves_nonclaims(tmp_path) -> None:
    config = NeuTraGPUBoundedTrainingConfig(artifact_dir=tmp_path)
    payload = phase10_error_payload(RuntimeError("example"), config=config)

    assert payload["passed"] is False
    assert payload["decision"] == "BLOCK_PHASE16_BOUNDED_GPU_XLA_NEUTRA_TRAINING"
    assert payload["optimizer_steps_executed"] == 0
    assert payload["full_neutra_training_executed"] is False
    assert payload["hmc_executed"] is False
    assert payload["external_sample_generation_executed"] is False
    assert payload["jit_compile"] is True
    assert payload["jit_compile_false_runtime_executed"] is False
    assert payload["runtime_autodiff_executed"] is False
    assert payload["keras_optimizer_gradient_route_executed"] is False
    assert payload["nonclaims"] == NEUTRA_GPU_BOUNDED_TRAINING_NONCLAIMS


def test_phase10_old_non_xla_training_artifact_is_stale_if_present() -> None:
    if not PHASE10_TRAINING_STATE.exists():
        pytest.skip("Phase 10 trusted GPU artifact has not been generated yet")
    payload = json.loads(PHASE10_TRAINING_STATE.read_text(encoding="utf-8"))
    if payload.get("passed") is not True:
        pytest.skip("Phase 10 trusted GPU pass artifact has not been generated yet")
    normalized = dict(payload)
    normalized.pop("artifact_hash", None)
    normalized.pop("artifact_hash_semantics", None)
    stable_hash = hashlib.sha256(
        json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    assert payload["route"]["route_id"] == LGSSM_QR_ROUTE_ID
    assert payload["config"]["training_execution_target"] == "gpu_required"
    assert payload["config"]["cpu_training_fallback_policy"] == "forbidden"
    assert payload["config"]["jit_compile"] is False
    assert payload["target_signature"] == (
        "290a91d2a8f90d5b29243965b258b1ec6fd965aa46ffca69dcb78f7fa1ecabcb"
    )
    assert payload["artifact_hash"] == f"sha256:{stable_hash}"
    assert payload["artifact_hash_semantics"] == (
        "stable_json_sha256_excluding_artifact_hash_fields"
    )
    assert payload["decision"] == "PASS_PHASE10_BOUNDED_GPU_NEUTRA_TRAINING"
