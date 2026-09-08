from __future__ import annotations

from dataclasses import replace

import pytest

from bayesfilter.highdim.sqmc_tuning_scope import (
    SQMCOrderingScope,
    SQMCTuningArtifact,
    SQMCTuningControls,
    issue_sqmc_tuning_artifact,
    require_sqmc_tuning_artifact,
    scope_from_mapping,
)


def _scope() -> SQMCOrderingScope:
    return SQMCOrderingScope(
        model_id="lgssm",
        target_id="lgssm_kalman",
        arm_id="ordered_rqmc_hybrid",
        experiment_scope="all_innovations",
        horizon=6,
        particle_count=48,
        state_dimension=3,
        parameter_count=5,
        dtype="float32",
        jit_compile=True,
        reset_contract_id="contract_e_chol_v1",
        score_route_id="standard_backward_filtering_score_v1",
        prepared_data_id="claim_data_sha",
        calibration_partition_sha256="calibration_sha",
        state_map_id="componentwise_logistic_calibrated_v1",
        state_map_location=(0.0, 0.0, 0.0),
        state_map_scale=(1.0, 1.0, 1.0),
        saturation_policy_id="fixed_threshold_v1",
        hilbert_implementation_id="skilling_transpose_tf_int32_v1",
        hilbert_bits=12,
        hilbert_tie_policy_id="stable_original_index_v1",
        point_set_id="tfp_owen2017_randomized_halton_v1",
        point_set_dimension=4,
        row_sort_policy_id="stable_ancestor_coordinate_v1",
        endpoint_policy_id="dtype_nextafter_open_unit_interval_v1",
        ancestor_cdf_policy_id="empirical_inverse_cdf_right_open_v1",
    )


def test_scope_round_trip_and_artifact_match() -> None:
    scope = _scope()
    assert scope_from_mapping(scope.as_dict()) == scope
    controls = SQMCTuningControls(2.0, 8, 8, 1.0e-5)
    artifact = issue_sqmc_tuning_artifact(
        scope,
        controls,
        calibration_metric=1.0,
        candidate_count=3,
        calibration_seed_count=4,
    )
    assert require_sqmc_tuning_artifact(artifact, scope) == controls


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("arm_id", "full_sqmc_halton_mechanics"),
        ("experiment_scope", "initial_only"),
        ("hilbert_bits", 10),
        ("state_map_scale", (1.0, 2.0, 1.0)),
        ("point_set_dimension", 3),
        ("prepared_data_id", "different_claim_data"),
    ),
)
def test_material_scope_changes_fail_closed(field: str, value) -> None:
    scope = _scope()
    artifact = issue_sqmc_tuning_artifact(
        scope,
        SQMCTuningControls(2.0, 8, 8, 1.0e-5),
        calibration_metric=1.0,
        candidate_count=3,
        calibration_seed_count=4,
    )
    with pytest.raises(ValueError, match="does not match"):
        require_sqmc_tuning_artifact(artifact, replace(scope, **{field: value}))


def test_caller_stamped_artifact_is_rejected() -> None:
    scope = _scope()
    artifact = SQMCTuningArtifact(
        scope,
        SQMCTuningControls(2.0, 8, 8, 1.0e-5),
        1.0,
        3,
        4,
        "caller",
    )
    with pytest.raises(TypeError, match="caller-stamped"):
        require_sqmc_tuning_artifact(artifact, scope)
