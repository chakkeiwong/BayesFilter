"""Static policy and replay-role contracts for ordinary HMC.

These tests intentionally stop before target evaluation, tuning, or HMC.  They
verify that policy identity and authority failures are visible at construction
boundaries.
"""

from __future__ import annotations

import pytest

from bayesfilter.inference import (
    HMCKernelTuningConfig,
    ORDINARY_BROAD_FIXED_METRIC_POLICY_ID,
    ORDINARY_BROAD_PRIMARY_L_GRID,
    ORDINARY_ENGINEERING_JOINT_L_EPSILON_POLICY_ID,
    ORDINARY_LEGACY_JOINT_L_EPSILON_POLICY_ID,
    ORDINARY_SHARED_EPSILON_SCREEN_POLICY_ID,
    REPLAY_ROLE_MECHANICS_ONLY,
    build_claim_bearing_retained_frozen_kernel_hmc_adapter_from_mechanics_payload,
    build_claim_bearing_retained_frozen_kernel_hmc_adapter_from_tuning_payload,
    build_retained_frozen_kernel_hmc_adapter_from_mechanics_payload,
    resolve_ordinary_hmc_selection_policy,
)
from bayesfilter.hmc_route_contract import LEGACY_JOINT_L_EPSILON_ALGORITHM_ID
from bayesfilter.inference.hmc_kernel_tuning import stable_config_hash


def test_ordinary_policy_resolver_describes_shared_epsilon_route() -> None:
    policy = resolve_ordinary_hmc_selection_policy(
        "operational_paired_fixed_trajectory_selection_v3"
    )

    assert policy["policy_id"] == ORDINARY_SHARED_EPSILON_SCREEN_POLICY_ID
    assert policy["epsilon_l_treatment"] == (
        "shared_frozen_epsilon_screen_then_exact_l_retune"
    )
    assert policy["mass_signature_frozen_during_selection"] is True
    assert "shared_epsilon_screen_not_joint_pair_selection" in policy[
        "claim_bearing_blockers"
    ]


def test_ordinary_policy_resolver_distinguishes_legacy_and_probe_routes() -> None:
    legacy = resolve_ordinary_hmc_selection_policy(
        LEGACY_JOINT_L_EPSILON_ALGORITHM_ID
    )
    probe = resolve_ordinary_hmc_selection_policy(
        "operational_paired_fixed_trajectory_selection_v3",
        engineering_probe_covariance_multiplier_configured=True,
    )

    assert legacy["policy_id"] == ORDINARY_LEGACY_JOINT_L_EPSILON_POLICY_ID
    assert legacy["authority_status"] == "diagnostic_only_non_promoting"
    assert probe["policy_id"] == ORDINARY_ENGINEERING_JOINT_L_EPSILON_POLICY_ID
    assert probe["authority_status"] == "engineering_only_non_promoting"


def test_config_payload_carries_policy_identity() -> None:
    payload = HMCKernelTuningConfig.standard().payload()
    policy = payload["ordinary_selection_policy"]

    assert policy["policy_id"] == ORDINARY_BROAD_FIXED_METRIC_POLICY_ID
    assert tuple(policy["primary_l_grid"]) == ORDINARY_BROAD_PRIMARY_L_GRID
    assert policy["epsilon_l_treatment"] == (
        "independent_epsilon_ladder_for_every_l"
    )
    assert policy["refinement_rounds"] == 1
    assert policy["claim_bearing_blockers"] == ()


def test_claim_bearing_tuning_replay_rejects_numpy_blocker_before_validation() -> None:
    payload = {
        "schema": "bayesfilter.hmc_kernel_tuning_result.v1",
        "resolved_policy": {
            "claim_bearing_artifact_authority": False,
            "claim_bearing_blockers": ["ordinary_runtime_numpy_policy_pending"],
            "claim_bearing_blocker": "ordinary_runtime_numpy_policy_pending",
        },
    }

    with pytest.raises(ValueError, match="ordinary_runtime_numpy_policy_pending"):
        build_claim_bearing_retained_frozen_kernel_hmc_adapter_from_tuning_payload(
            adapter=object(),
            tuning_payload=payload,
            initial_position=[0.0],
        )


def test_claim_bearing_replay_cannot_clear_repository_policy_in_serialized_fields() -> None:
    config = HMCKernelTuningConfig.standard().payload()
    # Simulate a caller copying a valid ordinary policy and then deleting the
    # blocker fields.  The guard must recompute the policy from config rather
    # than accepting the caller's apparently clear authority declaration.
    payload = {
        "schema": "bayesfilter.hmc_kernel_tuning_result.v1",
        "config": config,
        "resolved_policy": {
            "algorithm_id": config["algorithm_id"],
            "ordinary_selection_policy": config["ordinary_selection_policy"],
            "claim_bearing_artifact_authority": True,
            "claim_bearing_blockers": [],
            "claim_bearing_blocker": None,
        },
    }

    with pytest.raises(ValueError, match="ordinary_runtime_numpy_policy_pending"):
        build_claim_bearing_retained_frozen_kernel_hmc_adapter_from_tuning_payload(
            adapter=object(),
            tuning_payload=payload,
            initial_position=[0.0],
        )


def test_mechanics_builder_rejects_claim_role_on_mechanics_only_boundary() -> None:
    mechanics = {
        "schema": "bayesfilter.admitted_hmc_kernel_mechanics.v1",
        "replay_role": "claim_bearing_retained",
        "claim_bearing_artifact_authority": True,
        "resolved_policy": {
            "claim_bearing_artifact_authority": True,
            "claim_bearing_blockers": [],
            "claim_bearing_blocker": None,
        },
    }
    payload = {
        "schema": "bayesfilter.admitted_hmc_kernel_replay_artifact.v1",
        "mechanics": mechanics,
        "mechanics_sha256": stable_config_hash(mechanics),
    }

    with pytest.raises(ValueError, match="replay role mismatch"):
        build_retained_frozen_kernel_hmc_adapter_from_mechanics_payload(
            adapter=object(),
            mechanics_payload=payload,
            initial_position=[0.0],
            target_signature="target",
            target_scope="scope",
            execution={},
            target_accept_prob=0.7,
            acceptance_band=(0.65, 0.75),
        )


def test_mechanics_role_identifier_is_explicit() -> None:
    assert REPLAY_ROLE_MECHANICS_ONLY == "mechanics_only"
