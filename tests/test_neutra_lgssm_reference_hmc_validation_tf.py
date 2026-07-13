from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import pytest

from bayesfilter.testing.neutra_lgssm_reference_hmc_validation_tf import (
    DEFAULT_PHASE20_OUTPUT_PATH,
    NEUTRA_LGSSM_REFERENCE_HMC_VALIDATION_NONCLAIMS,
    PHASE20_ROUTE,
    PHASE20_TARGET_SCOPE,
    NeuTraLGSSMReferenceHMCValidationConfig,
    NeuTraLGSSMReferenceHMCValidationError,
    _aggregate_worker_samples,
    _worker_chain_count,
    phase20_error_payload,
    run_lgssm_reference_hmc_validation,
)


def test_phase20_config_records_reference_and_jit_boundary(tmp_path) -> None:
    default_config = NeuTraLGSSMReferenceHMCValidationConfig()
    config = NeuTraLGSSMReferenceHMCValidationConfig(
        output_path=tmp_path / "phase20.json"
    )
    payload = config.normalized()

    assert default_config.output_path.name == DEFAULT_PHASE20_OUTPUT_PATH.name
    assert payload["output_path"] == str(tmp_path / "phase20.json")
    assert payload["phase"] == PHASE20_ROUTE
    assert payload["target_scope"] == PHASE20_TARGET_SCOPE
    assert payload["execution_target"] == "cpu_hidden_multicore_full_chain_hmc"
    assert payload["training_execution_target"] == "not_run"
    assert payload["gpu_sample_generation_policy"] == "forbidden"
    assert payload["jit_compile"] is True
    assert payload["jit_compile_false_runtime_allowed"] is False
    assert payload["reference_posterior_policy"] == (
        "deterministic_2d_quadrature_over_exact_lgssm_likelihood"
    )
    assert payload["full_chain_xla_diagnostic_authority_scope"] == "phase20_only"
    assert payload["nonclaims"] == NEUTRA_LGSSM_REFERENCE_HMC_VALIDATION_NONCLAIMS


def test_phase20_rejects_jit_compile_false(tmp_path) -> None:
    with pytest.raises(NeuTraLGSSMReferenceHMCValidationError, match="jit_compile=false"):
        run_lgssm_reference_hmc_validation(
            NeuTraLGSSMReferenceHMCValidationConfig(
                output_path=tmp_path / "phase20.json",
                jit_compile=False,
            )
        )


def test_phase20_rejects_unhidden_cpu_policy(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    with pytest.raises(
        NeuTraLGSSMReferenceHMCValidationError,
        match="CUDA_VISIBLE_DEVICES=-1",
    ):
        run_lgssm_reference_hmc_validation(
            NeuTraLGSSMReferenceHMCValidationConfig(
                output_path=tmp_path / "phase20.json",
            )
        )


def test_phase20_rejects_worker_count_above_chain_count(tmp_path) -> None:
    with pytest.raises(
        NeuTraLGSSMReferenceHMCValidationError,
        match="worker_count must not exceed chain_count",
    ):
        run_lgssm_reference_hmc_validation(
            NeuTraLGSSMReferenceHMCValidationConfig(
                output_path=tmp_path / "phase20.json",
                worker_count=3,
                chain_count=2,
            )
        )


def test_phase20_worker_chain_counts_partition_chains() -> None:
    counts = [
        _worker_chain_count(worker_index=i, worker_count=3, chain_count=8)
        for i in range(3)
    ]

    assert counts == [3, 3, 2]
    assert sum(counts) == 8


def test_phase20_error_payload_preserves_nonclaims(tmp_path) -> None:
    config = NeuTraLGSSMReferenceHMCValidationConfig(
        output_path=tmp_path / "phase20.json"
    )
    payload = phase20_error_payload(RuntimeError("example"), config=config)

    assert payload["passed"] is False
    assert payload["decision"] == "BLOCK_PHASE20_LGSSM_REFERENCE_HMC_VALIDATION"
    assert payload["jit_compile"] is True
    assert payload["jit_compile_false_runtime_executed"] is False
    assert payload["training_executed"] is False
    assert payload["gpu_sample_generation_executed"] is False
    assert payload["posterior_validation_executed"] is False
    assert payload["full_chain_hmc_executed"] is False
    assert payload["hmc_tuning_executed"] is False
    assert payload["nonclaims"] == NEUTRA_LGSSM_REFERENCE_HMC_VALIDATION_NONCLAIMS
    assert payload["artifact_hash"].startswith("sha256:")


def test_phase20_aggregate_worker_samples_uses_public_summaries() -> None:
    aggregate = _aggregate_worker_samples(
        [
            {
                "return_code": 0,
                "retained_sample_count": 2,
                "sample_mean": [1.0, 2.0],
                "sample_covariance": [[0.5, 0.0], [0.0, 0.5]],
            },
            {
                "return_code": 0,
                "retained_sample_count": 2,
                "sample_mean": [3.0, 4.0],
                "sample_covariance": [[0.5, 0.0], [0.0, 0.5]],
            },
        ]
    )

    assert aggregate["total_retained_sample_count"] == 4
    assert aggregate["posterior_sample_summary"]["mean"] == [2.0, 3.0]
    assert aggregate["posterior_sample_summary"]["sample_source"] == (
        "weighted_public_worker_summaries"
    )


def test_phase20_helper_source_has_no_forbidden_runtime_tape_tokens() -> None:
    path = Path("bayesfilter/testing/neutra_lgssm_reference_hmc_validation_tf.py")
    source = path.read_text(encoding="utf-8")

    for forbidden in ("GradientTape", "batch_jacobian", "tape.", "jit_compile=False"):
        assert forbidden not in source
    for required in (
        "CUDA_VISIBLE_DEVICES",
        "jit_compile",
        "full_chain",
        "reference_posterior",
        "posterior_validation_executed",
    ):
        assert required in source


def test_phase20_artifact_if_present_preserves_validation_boundary() -> None:
    path = (
        "docs/plans/artifacts/lgssm-first-neutra-gpu-training-2026-07-07/"
        "lgssm_static_qr_exact_kalman_affine_neutra_"
        "phase20_lgssm_reference_hmc_validation_seed20260707.json"
    )
    if not os.path.exists(path):
        pytest.skip("Phase 20 artifact has not been generated yet")
    payload = json.loads(open(path, encoding="utf-8").read())

    assert payload["schema"] == (
        "bayesfilter.neutra.lgssm_reference_hmc_validation_result.v1"
    )
    assert payload["phase"] == PHASE20_ROUTE
    assert payload["cuda_visible_devices"] == "-1"
    assert payload["jit_compile"] is True
    assert payload["jit_compile_false_runtime_executed"] is False
    assert payload["training_executed"] is False
    assert payload["gpu_sample_generation_executed"] is False
    assert payload["posterior_validation_executed"] is True
    assert payload["reference_posterior"]["reference_type"] == (
        "deterministic_2d_quadrature"
    )
    assert payload["reference_posterior"]["analytic_exact_posterior"] is False
    assert payload["sample_diagnostics"]
    assert payload["nonclaims"] == list(
        NEUTRA_LGSSM_REFERENCE_HMC_VALIDATION_NONCLAIMS
    )
    assert payload["decision"] in {
        "PASS_PHASE20_LGSSM_REFERENCE_HMC_VALIDATION",
        "BLOCK_PHASE20_LGSSM_REFERENCE_HMC_VALIDATION",
    }
