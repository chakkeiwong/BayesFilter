from __future__ import annotations

import json
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import pytest

from bayesfilter.testing.neutra_cpu_multicore_hmc_chain_harness_tf import (
    DEFAULT_PHASE19_OUTPUT_PATH,
    NEUTRA_CPU_MULTICORE_HMC_CHAIN_HARNESS_NONCLAIMS,
    PHASE19_ROUTE,
    NeuTraCPUMulticoreHMCChainHarnessConfig,
    NeuTraCPUMulticoreHMCChainHarnessError,
    phase19_error_payload,
    run_cpu_multicore_hmc_chain_harness,
)


def test_phase19_config_records_cpu_hidden_jit_boundary(tmp_path) -> None:
    default_config = NeuTraCPUMulticoreHMCChainHarnessConfig()
    config = NeuTraCPUMulticoreHMCChainHarnessConfig(
        output_path=tmp_path / "phase19.json"
    )
    payload = config.normalized()

    assert default_config.output_path.name == DEFAULT_PHASE19_OUTPUT_PATH.name
    assert payload["output_path"] == str(tmp_path / "phase19.json")
    assert payload["phase"] == PHASE19_ROUTE
    assert payload["execution_target"] == "cpu_hidden_multicore_worker_harness"
    assert payload["hmc_policy"] == "worker_value_score_compile_smoke_no_transition"
    assert payload["posterior_validation_policy"] == "not_run_deferred_to_phase20"
    assert payload["jit_compile"] is True
    assert payload["jit_compile_false_runtime_allowed"] is False
    assert payload["allow_hmc_transition"] is False
    assert payload["training_execution_target"] == "not_run"
    assert payload["gpu_sample_generation_policy"] == "forbidden"
    assert payload["nonclaims"] == NEUTRA_CPU_MULTICORE_HMC_CHAIN_HARNESS_NONCLAIMS


def test_phase19_rejects_jit_compile_false(tmp_path) -> None:
    with pytest.raises(NeuTraCPUMulticoreHMCChainHarnessError, match="jit_compile=false"):
        run_cpu_multicore_hmc_chain_harness(
            NeuTraCPUMulticoreHMCChainHarnessConfig(
                output_path=tmp_path / "phase19.json",
                jit_compile=False,
            )
        )


def test_phase19_rejects_hmc_transition_flag(tmp_path) -> None:
    with pytest.raises(NeuTraCPUMulticoreHMCChainHarnessError, match="forbids HMC"):
        run_cpu_multicore_hmc_chain_harness(
            NeuTraCPUMulticoreHMCChainHarnessConfig(
                output_path=tmp_path / "phase19.json",
                allow_hmc_transition=True,
            )
        )


def test_phase19_rejects_unhidden_cpu_policy(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    with pytest.raises(NeuTraCPUMulticoreHMCChainHarnessError, match="CUDA_VISIBLE_DEVICES=-1"):
        run_cpu_multicore_hmc_chain_harness(
            NeuTraCPUMulticoreHMCChainHarnessConfig(
                output_path=tmp_path / "phase19.json",
            )
        )


def test_phase19_error_payload_preserves_nonclaims(tmp_path) -> None:
    config = NeuTraCPUMulticoreHMCChainHarnessConfig(
        output_path=tmp_path / "phase19.json"
    )
    payload = phase19_error_payload(RuntimeError("example"), config=config)

    assert payload["passed"] is False
    assert payload["decision"] == "BLOCK_PHASE19_CPU_MULTICORE_HMC_CHAIN_HARNESS"
    assert payload["jit_compile"] is True
    assert payload["jit_compile_false_runtime_executed"] is False
    assert payload["training_executed"] is False
    assert payload["hmc_transition_executed"] is False
    assert payload["hmc_sampling_or_tuning_executed"] is False
    assert payload["posterior_validation_executed"] is False
    assert payload["gpu_sample_generation_executed"] is False
    assert payload["nonclaims"] == NEUTRA_CPU_MULTICORE_HMC_CHAIN_HARNESS_NONCLAIMS


def test_phase19_artifact_if_present_preserves_harness_boundary() -> None:
    path = (
        "docs/plans/artifacts/lgssm-first-neutra-gpu-training-2026-07-07/"
        "lgssm_static_qr_exact_kalman_affine_neutra_"
        "phase19_cpu_multicore_hmc_chain_harness_seed20260707.json"
    )
    if not os.path.exists(path):
        pytest.skip("Phase 19 artifact has not been generated yet")
    payload = json.loads(open(path, encoding="utf-8").read())

    assert payload["schema"] == (
        "bayesfilter.neutra.cpu_multicore_hmc_chain_harness_result.v1"
    )
    assert payload["phase"] == PHASE19_ROUTE
    assert payload["cuda_visible_devices"] == "-1"
    assert payload["jit_compile"] is True
    assert payload["jit_compile_false_runtime_executed"] is False
    assert payload["training_executed"] is False
    assert payload["hmc_transition_executed"] is False
    assert payload["hmc_sampling_or_tuning_executed"] is False
    assert payload["posterior_validation_executed"] is False
    assert payload["gpu_sample_generation_executed"] is False
    assert payload["nonclaims"] == list(
        NEUTRA_CPU_MULTICORE_HMC_CHAIN_HARNESS_NONCLAIMS
    )
    if payload["passed"]:
        assert payload["decision"] == "PASS_PHASE19_CPU_MULTICORE_HMC_CHAIN_HARNESS"
        assert payload["boundary_checks"]["worker_return_codes_zero"] is True
        assert payload["boundary_checks"]["worker_seeds_distinct"] is True
        assert payload["boundary_checks"]["fixed_transport_adapter_signature_match"] is True
        assert payload["boundary_checks"]["jit_compile_false_runtime_not_executed"] is True
        assert payload["boundary_checks"]["posterior_validation_not_executed"] is True
        assert payload["boundary_checks"]["hmc_transition_not_executed"] is True
        assert payload["workers"]
    else:
        assert payload["decision"] == "BLOCK_PHASE19_CPU_MULTICORE_HMC_CHAIN_HARNESS"
