from __future__ import annotations

import pytest

from bayesfilter.highdim.ledh_forward_contract import (
    ACTUAL_SV_ROW_ID,
    KSC_SV_ROW_ID,
    LEDH_FORWARD_ADMISSION_STATUS_ADMITTED,
    LEDH_FORWARD_ADMISSION_STATUS_HISTORICAL_RAW,
    LEDH_FORWARD_ADMISSION_STATUS_TINY,
    LEDH_FORWARD_SCALAR_ARTIFACT_SCHEMA_VERSION,
    LEDH_TARGET_SCALAR_OBSERVED_DATA_LOG_LIKELIHOOD,
    LEDHForwardLikelihoodContract,
    LGSSM_M3_T50_ROW_ID,
    make_actual_sv_forward_contract,
    make_ksc_sv_forward_contract,
    make_lgssm_m3_t50_forward_contract,
    validate_ledh_forward_contract_manifest,
    validate_ledh_forward_scalar_artifact,
)


def _lgssm_contract(*, full_leaderboard_row: bool = True) -> dict[str, object]:
    return make_lgssm_m3_t50_forward_contract(
        truth_theta=[0.72, 0.55, 0.35, 0.35, 0.45],
        time_steps=50,
        num_particles=10000,
        batch_seeds=[81120, 81121],
        full_leaderboard_row=full_leaderboard_row,
    ).to_manifest()


def _canonical_artifact(
    *,
    contract: dict[str, object] | None = None,
    admission_status: str = LEDH_FORWARD_ADMISSION_STATUS_ADMITTED,
) -> dict[str, object]:
    contract = _lgssm_contract() if contract is None else contract
    theta_contract = contract["theta_contract"]
    assert isinstance(theta_contract, dict)
    return {
        "schema_version": LEDH_FORWARD_SCALAR_ARTIFACT_SCHEMA_VERSION,
        "row_id": contract["row_id"],
        "forward_contract": contract,
        "target_scalar": contract["target_scalar"],
        "target_output_tensor_field": contract["output_tensor_field"],
        "target_density_fields": contract["target_density_fields"],
        "proposal_flow_fields": contract["proposal_flow_fields"],
        "correction_formula": contract["correction_formula"],
        "theta_values": theta_contract["truth_theta"],
        "theta_coordinate_system": theta_contract["theta_coordinate_system"],
        "flow_observation_policy": "identity_lgssm_observation_flow",
        "target_observation_policy": "lgssm_gaussian_observation_density",
        "target_density_used_for_correction": True,
        "batch_seeds": [81120, 81121],
        "num_particles": 10000,
        "time_steps": 50,
        "log_likelihood_by_seed": [-135.9, -136.1],
        "average_log_likelihood_by_seed": [-2.718, -2.722],
        "finite_output": True,
        "admission_status": admission_status,
        "nonclaims": [
            "not score admission",
            "not score correctness",
        ],
    }


def test_v1_forward_scalar_artifact_is_readable_but_canonically_ineligible() -> None:
    artifact = _canonical_artifact()

    normalized = validate_ledh_forward_scalar_artifact(
        artifact,
        expected_row_id=LGSSM_M3_T50_ROW_ID,
        require_admitted=False,
    )

    assert normalized["row_id"] == LGSSM_M3_T50_ROW_ID
    assert (
        LEDH_TARGET_SCALAR_OBSERVED_DATA_LOG_LIKELIHOOD
        == "observed_data_log_likelihood_estimator"
    )
    assert normalized["target_scalar"] == LEDH_TARGET_SCALAR_OBSERVED_DATA_LOG_LIKELIHOOD
    assert normalized["target_output_tensor_field"] == "log_likelihood"
    assert normalized["log_likelihood_by_seed"] == [-135.9, -136.1]
    assert normalized["admission_status"] == LEDH_FORWARD_ADMISSION_STATUS_HISTORICAL_RAW
    assert normalized["canonical_admission_eligible"] is False
    with pytest.raises(ValueError, match="lacks factory-issued Contract E"):
        validate_ledh_forward_scalar_artifact(
            artifact,
            expected_row_id=LGSSM_M3_T50_ROW_ID,
            require_admitted=True,
        )


def test_metadata_only_forward_contract_cannot_be_value_admission() -> None:
    contract = _lgssm_contract()

    assert validate_ledh_forward_contract_manifest(contract) == contract
    with pytest.raises(ValueError, match="schema_version"):
        validate_ledh_forward_scalar_artifact(contract, require_admitted=True)


def test_callback_only_artifact_cannot_be_value_admission() -> None:
    artifact = _canonical_artifact()
    artifact.pop("log_likelihood_by_seed")
    artifact["target_density_callbacks"] = {
        "transition_log_density_fn": "present",
        "observation_log_density_fn": "present",
    }

    with pytest.raises(ValueError, match="log_likelihood_by_seed"):
        validate_ledh_forward_scalar_artifact(artifact, require_admitted=True)


def test_missing_average_log_likelihood_vector_is_rejected() -> None:
    artifact = _canonical_artifact()
    artifact.pop("average_log_likelihood_by_seed")

    with pytest.raises(ValueError, match="average_log_likelihood_by_seed"):
        validate_ledh_forward_scalar_artifact(artifact, require_admitted=True)


def test_proposal_scalar_cannot_replace_target_scalar_in_executable_artifact() -> None:
    artifact = _canonical_artifact()
    artifact["target_scalar"] = "proposal_log_likelihood"

    with pytest.raises(ValueError, match="target_scalar"):
        validate_ledh_forward_scalar_artifact(artifact, require_admitted=True)


def test_theta_values_must_match_forward_contract_truth_theta() -> None:
    artifact = _canonical_artifact()
    artifact["theta_values"] = [0.73, 0.55, 0.35, 0.35, 0.45]

    with pytest.raises(ValueError, match="truth_theta"):
        validate_ledh_forward_scalar_artifact(artifact, require_admitted=True)


def test_target_density_must_be_used_for_correction() -> None:
    artifact = _canonical_artifact()
    artifact["target_density_used_for_correction"] = False

    with pytest.raises(ValueError, match="target_density_used_for_correction"):
        validate_ledh_forward_scalar_artifact(artifact, require_admitted=True)


def test_flow_and_target_observation_policies_must_not_be_ambiguous() -> None:
    artifact = _canonical_artifact()
    artifact["target_observation_policy"] = artifact["flow_observation_policy"]

    with pytest.raises(ValueError, match="flow_observation_policy"):
        validate_ledh_forward_scalar_artifact(artifact, require_admitted=True)


def test_tiny_executed_artifact_validates_but_cannot_be_required_admission() -> None:
    contract = make_lgssm_m3_t50_forward_contract(
        truth_theta=[0.72, 0.55, 0.35, 0.35, 0.45],
        time_steps=2,
        num_particles=4,
        batch_seeds=[81120],
        full_leaderboard_row=False,
    ).to_manifest()
    artifact = _canonical_artifact(
        contract=contract,
        admission_status=LEDH_FORWARD_ADMISSION_STATUS_TINY,
    )
    artifact["batch_seeds"] = [81120]
    artifact["num_particles"] = 4
    artifact["time_steps"] = 2
    artifact["log_likelihood_by_seed"] = [-9.17]
    artifact["average_log_likelihood_by_seed"] = [-4.58]

    normalized = validate_ledh_forward_scalar_artifact(artifact)
    assert normalized["admission_status"] == LEDH_FORWARD_ADMISSION_STATUS_HISTORICAL_RAW
    with pytest.raises(ValueError, match="not admitted"):
        validate_ledh_forward_scalar_artifact(artifact, require_admitted=True)


def test_admitted_status_requires_full_row_scale() -> None:
    contract = make_lgssm_m3_t50_forward_contract(
        truth_theta=[0.72, 0.55, 0.35, 0.35, 0.45],
        time_steps=50,
        num_particles=4,
        batch_seeds=[81120],
        full_leaderboard_row=True,
    ).to_manifest()
    artifact = _canonical_artifact(contract=contract)
    artifact["batch_seeds"] = [81120]
    artifact["num_particles"] = 4
    artifact["log_likelihood_by_seed"] = [-135.9]
    artifact["average_log_likelihood_by_seed"] = [-2.718]

    with pytest.raises(ValueError, match="at least 10000 particles"):
        validate_ledh_forward_scalar_artifact(artifact, require_admitted=True)


def test_actual_sv_and_ksc_target_policies_cannot_be_cross_used() -> None:
    actual_contract = make_actual_sv_forward_contract(
        time_steps=1000,
        num_particles=10000,
        batch_seeds=[81120],
        full_leaderboard_row=True,
    ).to_manifest()
    artifact = _canonical_artifact(contract=actual_contract)
    artifact["row_id"] = ACTUAL_SV_ROW_ID
    artifact["theta_values"] = [0.2533471031357997, -0.916290731874155]
    artifact["theta_coordinate_system"] = "synthetic_unconstrained"
    artifact["flow_observation_policy"] = "log_square_actual_sv_flow_observation"
    artifact["target_observation_policy"] = "ksc_log_chi_square_gaussian_mixture_surrogate"
    artifact["batch_seeds"] = [81120]
    artifact["time_steps"] = 1000
    artifact["log_likelihood_by_seed"] = [-1400.0]
    artifact["average_log_likelihood_by_seed"] = [-1.4]

    with pytest.raises(ValueError, match="target_observation_policy"):
        validate_ledh_forward_scalar_artifact(
            artifact,
            expected_row_id=ACTUAL_SV_ROW_ID,
            require_admitted=True,
        )

    ksc_contract = make_ksc_sv_forward_contract(
        time_steps=1000,
        num_particles=10000,
        batch_seeds=[81120],
        full_leaderboard_row=True,
    ).to_manifest()
    artifact["row_id"] = ACTUAL_SV_ROW_ID
    artifact["forward_contract"] = ksc_contract
    artifact["target_observation_policy"] = "ksc_log_chi_square_gaussian_mixture_surrogate"

    with pytest.raises(ValueError, match="row_id"):
        validate_ledh_forward_scalar_artifact(
            artifact,
            expected_row_id=ACTUAL_SV_ROW_ID,
            require_admitted=True,
        )


def test_ksc_contract_rejects_actual_sv_target_policy() -> None:
    ksc = make_ksc_sv_forward_contract()

    with pytest.raises(ValueError, match=KSC_SV_ROW_ID):
        LEDHForwardLikelihoodContract(
            row_id=KSC_SV_ROW_ID,
            row_scope="main_observed_data_filtering_row",
            theta_contract=ksc.theta_contract,
            metadata={"target_observation_policy": "transformed_actual_sv_log_y_square"},
        )
