from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import pytest

from bayesfilter.testing.neutra_cpu_sample_boundary import (
    EXPECTED_PHASE10_ADAPTER_SIGNATURE,
    EXPECTED_PHASE10_TARGET_SIGNATURE,
    NEUTRA_CPU_SAMPLE_BOUNDARY_NONCLAIMS,
    NeuTraCPUSampleBoundaryConfig,
    NeuTraCPUSampleBoundaryError,
    generate_cpu_multicore_external_sample_boundary,
)


def _payload() -> dict:
    return {
        "schema": "bayesfilter.neutra.frozen_affine_diag.v1",
        "phase": "phase11_frozen_gpu_trained_affine_payload",
        "transport_id": "lgssm-gpu-trained-affine-diag-neutra-seed20260707",
        "dimension": 2,
        "target_signature": EXPECTED_PHASE10_TARGET_SIGNATURE,
        "log_jacobian_available": True,
        "shift": [-0.02, -1.37],
        "raw_scale": [-1.03, -1.03],
        "training_state_hash": "sha256:" + "1" * 64,
        "source_adapter_signature": EXPECTED_PHASE10_ADAPTER_SIGNATURE,
        "source_training_state_artifact_hash": "sha256:" + "2" * 64,
        "source_training_execution_target": "gpu_required",
        "training_executed": False,
        "hmc_executed": False,
        "external_sample_generation_executed": False,
        "jit_compile": False,
    }


def _write_payload(tmp_path: Path, payload: dict | None = None) -> Path:
    path = tmp_path / "payload.json"
    path.write_text(json.dumps(_payload() if payload is None else payload), encoding="utf-8")
    return path


def test_phase12_cpu_multicore_boundary_writes_diagnostic_samples(tmp_path) -> None:
    payload_path = _write_payload(tmp_path)
    output_path = tmp_path / "samples.json"

    result = generate_cpu_multicore_external_sample_boundary(
        NeuTraCPUSampleBoundaryConfig(
            payload_path=payload_path,
            output_path=output_path,
            sample_count=6,
            worker_count=2,
        )
    )

    assert result.passed is True
    assert output_path.exists()
    assert result.artifact["schema"] == (
        "bayesfilter.neutra.cpu_multicore_sample_boundary.v1"
    )
    assert result.artifact["sample_count"] == 6
    assert result.artifact["worker_count"] == 2
    assert len(result.artifact["base_samples"]) == 6
    assert len(result.artifact["transported_samples"]) == 6
    assert result.artifact["finite_checks"] == {
        "base_samples_finite": True,
        "sample_count_match": True,
        "sample_dimension_match": True,
        "transported_samples_finite": True,
    }
    assert result.artifact["boundary_checks"]["cpu_hidden"] is True
    assert result.artifact["boundary_checks"]["worker_count_recorded"] is True
    assert result.artifact["boundary_checks"]["source_gpu_training_artifact"] is True
    assert result.artifact["training_executed"] is False
    assert result.artifact["cpu_neutra_training_executed"] is False
    assert result.artifact["hmc_executed"] is False
    assert result.artifact["gpu_sample_generation_executed"] is False
    assert result.artifact["jit_compile"] is False
    assert result.artifact["external_sample_generation_executed"] is True
    assert result.artifact["nonclaims"] == NEUTRA_CPU_SAMPLE_BOUNDARY_NONCLAIMS


def test_phase12_boundary_is_deterministic_for_fixed_seed(tmp_path) -> None:
    payload_path = _write_payload(tmp_path)
    left = generate_cpu_multicore_external_sample_boundary(
        NeuTraCPUSampleBoundaryConfig(
            payload_path=payload_path,
            output_path=tmp_path / "left.json",
            sample_count=5,
            worker_count=2,
            seed=20260707,
        )
    )
    right = generate_cpu_multicore_external_sample_boundary(
        NeuTraCPUSampleBoundaryConfig(
            payload_path=payload_path,
            output_path=tmp_path / "right.json",
            sample_count=5,
            worker_count=2,
            seed=20260707,
        )
    )

    assert left.artifact["base_samples"] == right.artifact["base_samples"]
    assert left.artifact["transported_samples"] == right.artifact["transported_samples"]


def test_phase12_rejects_unhidden_cpu_policy(tmp_path, monkeypatch) -> None:
    payload_path = _write_payload(tmp_path)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    with pytest.raises(NeuTraCPUSampleBoundaryError, match="CUDA_VISIBLE_DEVICES=-1"):
        generate_cpu_multicore_external_sample_boundary(
            NeuTraCPUSampleBoundaryConfig(
                payload_path=payload_path,
                output_path=tmp_path / "samples.json",
            )
        )


@pytest.mark.parametrize(
    "flag_name",
    [
        "allow_neutra_training",
        "allow_cpu_neutra_training",
        "allow_hmc_sampling_or_tuning",
        "allow_gpu_sample_generation",
        "allow_xla",
    ],
)
def test_phase12_rejects_forbidden_capability_flags(tmp_path, flag_name) -> None:
    payload_path = _write_payload(tmp_path)

    with pytest.raises(NeuTraCPUSampleBoundaryError, match="forbids"):
        generate_cpu_multicore_external_sample_boundary(
            NeuTraCPUSampleBoundaryConfig(
                payload_path=payload_path,
                output_path=tmp_path / "samples.json",
                **{flag_name: True},
            )
        )


def test_phase12_rejects_payload_not_sourced_from_gpu_training(tmp_path) -> None:
    payload = _payload()
    payload["source_training_execution_target"] = "cpu"
    payload_path = _write_payload(tmp_path, payload)

    with pytest.raises(NeuTraCPUSampleBoundaryError, match="GPU training"):
        generate_cpu_multicore_external_sample_boundary(
            NeuTraCPUSampleBoundaryConfig(
                payload_path=payload_path,
                output_path=tmp_path / "samples.json",
            )
        )
