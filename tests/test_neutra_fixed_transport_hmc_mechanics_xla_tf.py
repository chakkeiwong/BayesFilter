from __future__ import annotations

import json
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import pytest

from bayesfilter.testing.neutra_fixed_transport_hmc_mechanics_xla_tf import (
    DEFAULT_PHASE17_PAYLOAD_PATH,
    NEUTRA_FIXED_TRANSPORT_HMC_MECHANICS_XLA_NONCLAIMS,
    PHASE18_ROUTE,
    NeuTraFixedTransportHMCMechanicsXLAConfig,
    NeuTraFixedTransportHMCMechanicsXLAError,
    phase18_error_payload,
    run_fixed_transport_hmc_mechanics_xla_diagnostic,
)


def test_phase18_config_records_compile_only_boundary(tmp_path) -> None:
    config = NeuTraFixedTransportHMCMechanicsXLAConfig(
        output_path=tmp_path / "phase18.json"
    )
    payload = config.normalized()

    assert payload["phase"] == PHASE18_ROUTE
    assert payload["payload_path"] == str(DEFAULT_PHASE17_PAYLOAD_PATH)
    assert payload["use_xla"] is True
    assert payload["jit_compile"] is True
    assert payload["jit_compile_false_runtime_allowed"] is False
    assert payload["training_execution_target"] == (
        "not_run_phase18_mechanics_compile_only"
    )
    assert payload["hmc_policy"] == "mechanics_compile_only_no_sampling_no_tuning"
    assert payload["external_sample_generation_policy"] == "not_run"
    assert payload["nonclaims"] == NEUTRA_FIXED_TRANSPORT_HMC_MECHANICS_XLA_NONCLAIMS


def test_phase18_rejects_cpu_hidden_trusted_gpu_diagnostic(tmp_path) -> None:
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"

    with pytest.raises(NeuTraFixedTransportHMCMechanicsXLAError, match="CUDA_VISIBLE_DEVICES"):
        run_fixed_transport_hmc_mechanics_xla_diagnostic(
            NeuTraFixedTransportHMCMechanicsXLAConfig(
                output_path=tmp_path / "phase18.json"
            )
        )


def test_phase18_error_payload_preserves_nonclaims(tmp_path) -> None:
    config = NeuTraFixedTransportHMCMechanicsXLAConfig(
        output_path=tmp_path / "phase18.json"
    )
    payload = phase18_error_payload(RuntimeError("example"), config=config)

    assert payload["passed"] is False
    assert payload["decision"] == "BLOCK_PHASE18_FIXED_TRANSPORT_HMC_MECHANICS_XLA_COMPILE"
    assert payload["training_executed"] is False
    assert payload["hmc_sampling_or_tuning_executed"] is False
    assert payload["external_sample_generation_executed"] is False
    assert payload["use_xla"] is True
    assert payload["jit_compile"] is True
    assert payload["jit_compile_false_runtime_executed"] is False
    assert payload["nonclaims"] == NEUTRA_FIXED_TRANSPORT_HMC_MECHANICS_XLA_NONCLAIMS


def test_phase18_artifact_if_present_preserves_compile_only_boundary() -> None:
    path = (
        "docs/plans/artifacts/lgssm-first-neutra-gpu-training-2026-07-07/"
        "lgssm_static_qr_exact_kalman_affine_neutra_"
        "phase18_fixed_transport_hmc_mechanics_xla_compile_seed20260707.json"
    )
    if not os.path.exists(path):
        pytest.skip("Phase 18 trusted GPU/XLA artifact has not been generated yet")
    payload = json.loads(open(path, encoding="utf-8").read())

    assert payload["schema"] == (
        "bayesfilter.neutra.fixed_transport_hmc_mechanics_xla_result.v1"
    )
    assert payload["phase"] == PHASE18_ROUTE
    assert payload["use_xla"] is True
    assert payload["jit_compile"] is True
    assert payload["jit_compile_false_runtime_executed"] is False
    assert payload["training_executed"] is False
    assert payload["hmc_sampling_or_tuning_executed"] is False
    assert payload["external_sample_generation_executed"] is False
    assert payload["nonclaims"] == list(
        NEUTRA_FIXED_TRANSPORT_HMC_MECHANICS_XLA_NONCLAIMS
    )
    if payload["passed"]:
        assert payload["decision"] == (
            "PASS_PHASE18_FIXED_TRANSPORT_HMC_MECHANICS_XLA_COMPILE"
        )
        assert payload["finite_checks"]["mechanics_value_finite"] is True
        assert payload["finite_checks"]["mechanics_score_finite"] is True
        assert payload["base_value_score_capability"][
            "accepted_xla_hmc_authority"
        ] is True
        assert payload["fixed_transport_value_score_capability"][
            "accepted_xla_hmc_authority"
        ] is True
        assert payload["compile_time_proxy_seconds"] >= 0.0
    else:
        assert payload["decision"] == (
            "BLOCK_PHASE18_FIXED_TRANSPORT_HMC_MECHANICS_XLA_COMPILE"
        )
