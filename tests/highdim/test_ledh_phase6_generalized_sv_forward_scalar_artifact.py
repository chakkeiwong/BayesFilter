from __future__ import annotations

import json
import math
from pathlib import Path

from bayesfilter.highdim.ledh_forward_contract import (
    GENERALIZED_SV_ROW_ID,
    LEDH_FORWARD_ADMISSION_STATUS_ADMITTED,
    LEDH_TARGET_SCALAR_OBSERVED_DATA_LOG_LIKELIHOOD,
    validate_ledh_forward_scalar_artifact,
)


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_PATH = (
    ROOT / "docs/plans/ledh-phase6-generalized-sv-forward-scalar-artifact-2026-07-07.json"
)


def test_phase6_generalized_sv_full_forward_scalar_artifact_replays_admission_gate() -> None:
    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))

    normalized = validate_ledh_forward_scalar_artifact(
        artifact,
        expected_row_id=GENERALIZED_SV_ROW_ID,
        require_admitted=True,
    )

    assert normalized["row_id"] == GENERALIZED_SV_ROW_ID
    assert normalized["target_scalar"] == LEDH_TARGET_SCALAR_OBSERVED_DATA_LOG_LIKELIHOOD
    assert normalized["target_output_tensor_field"] == "log_likelihood"
    assert normalized["admission_status"] == LEDH_FORWARD_ADMISSION_STATUS_ADMITTED
    assert normalized["num_particles"] == 10000
    assert normalized["time_steps"] == 1008
    assert normalized["batch_seeds"] == [81120, 81121, 81122, 81123, 81124]
    assert normalized["theta_coordinate_system"] == "source_route_active_transformed_prior_mean"
    assert normalized["theta_values"] == [1.0824113944610982, -2.076793740349318, 0.0]
    assert normalized["target_observation_policy"] == "source_route_prior_mean_generalized_sv"
    assert normalized["flow_observation_policy"] == "log_square_gaussian_surrogate_for_ledh_flow_only"
    assert normalized["flow_observation_policy"] != normalized["target_observation_policy"]
    assert normalized["target_density_used_for_correction"] is True
    assert len(normalized["log_likelihood_by_seed"]) == 5
    assert all(math.isfinite(value) for value in normalized["log_likelihood_by_seed"])

    semantics = artifact["generalized_sv_semantics"]
    assert artifact["run_scope"] == "full-row-admission"
    assert semantics["row_id"] == GENERALIZED_SV_ROW_ID
    assert semantics["target_observation_density"] == (
        "raw_zero_mean_generalized_sv_prior_mean_normal_log_density"
    )
    assert semantics["flow_observation_transform"] == "log(y_t^2 + 1e-6)"
    assert semantics["flow_is_proposal_surface_only"] is True
    assert semantics["actual_sv_evidence_used"] is False
    assert semantics["ksc_mixture_used"] is False
    assert semantics["native_generalized_sv_dense_fixture_used"] is False
    assert semantics["sp500_returns_used_as_benchmark_observations"] is False
    assert semantics["author_defaults_used_as_truth"] is False
    assert artifact["transport"]["plan_mode"] == "streaming"
    assert artifact["transport"]["dense_transport_matrix_materialized"] is False
    assert artifact["normalization_checks"]["exact_full_row_requested"] is True
    assert artifact["normalization_checks"]["full_row_admission_scope"] is True
    assert artifact["output_devices"]
    assert all("GPU" in device.upper() for device in artifact["output_devices"])


def test_phase6_generalized_sv_full_artifact_nonclaims_exclude_score_and_substitutions() -> None:
    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    nonclaims = set(artifact["nonclaims"])
    validator_nonclaims = set(artifact["validator_normalized_core"]["nonclaims"])

    assert validator_nonclaims == nonclaims
    assert "not full generalized-SV row admission" not in nonclaims
    assert "not full generalized-SV row admission" not in validator_nonclaims
    assert "not score admission" in nonclaims
    assert "not score correctness" in nonclaims
    assert "not actual-SV admission" in nonclaims
    assert "not KSC admission" in nonclaims
    assert "not KSC surrogate likelihood evidence" in nonclaims
    assert "not native generalized-SV dense fixture evidence" in nonclaims
    assert "not SP500 benchmark-observation evidence" in nonclaims
    assert "not author-default truth evidence" in nonclaims
    assert "not HMC readiness evidence" in nonclaims
    assert "not posterior correctness evidence" in nonclaims
    assert "not scientific superiority evidence" in nonclaims
    assert "not runtime ranking evidence" in nonclaims
