"""Bounded local driver for the September 17 filter execution repair.

This is host orchestration, not a numerical implementation. The command prefix
is deliberately stable. Only registered test groups and fixtures can execute;
there is no arbitrary command, network, installation, merge, or push action.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import io
import json
import os
import signal
import subprocess
import sys
import tarfile
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

from enforce_filter_gradient_policy import verify as verify_source_policy
from filter_repair_additional_fixtures import FIXTURES as ADDITIONAL_FIXTURES
from filter_repair_batched_locator_fixtures import FIXTURES as BATCHED_LOCATOR_FIXTURES
from filter_repair_centered_fixtures import FIXTURES as CENTERED_FIXTURES
from filter_repair_centered_training_fixtures import (
    FIXTURES as CENTERED_TRAINING_FIXTURES,
)
from filter_repair_endpoint_fixtures import FIXTURES as ENDPOINT_FIXTURES
from filter_repair_forecast_fixtures import FIXTURES as FORECAST_FIXTURES
from filter_repair_forecast_pool_fixtures import FIXTURES as FORECAST_POOL_FIXTURES
from filter_repair_gpu_selection import check_gpu_available
from filter_repair_initialization_fixtures import FIXTURES as INITIALIZATION_FIXTURES
from filter_repair_locator_fixtures import FIXTURES as LOCATOR_FIXTURES
from filter_repair_preparation_fixtures import FIXTURES as PREPARATION_FIXTURES
from filter_repair_source_fixtures import FIXTURES as SOURCE_FIXTURES
from filter_repair_stochastic_fixtures import FIXTURES as STOCHASTIC_FIXTURES
from filter_repair_training_fixtures import FIXTURES as TRAINING_FIXTURES

ROOT = Path(__file__).resolve().parents[1]


def campaign_output_root(root):
    """Share artifacts and the cumulative budget across linked worktrees."""
    common = subprocess.check_output([
        "git", "rev-parse", "--path-format=absolute", "--git-common-dir",
    ], cwd=root, text=True).strip()
    return Path(common).parent / "docs/plans/artifacts/filter-gradient-repair-20260917"


OUTPUT = campaign_output_root(ROOT)
PLAN = "docs/plans/filter_gradient_repair_master_20260917.md"
BASELINE = "3582b4ac5fea67fb5da7fa60a1cbfaf19df35adf"
BASELINE_ROOT = Path("/tmp/bayesfilter-filter-repair-baseline-3582b4ac")
BASELINE_PARENT_PACKAGES = (
    "experiments/__init__.py", "experiments/dpf_implementation/__init__.py",
)
# September 19 owner authorization adds 24 CPU / 48 GPU process-hours to
# the original cumulative 8 CPU / 4 GPU hours; prior charges remain counted.
# Owner added 24 CPU hours on 2026-09-25; GPU allocation is unchanged.
BUDGET_SECONDS = {"CPU": 56 * 3600, "GPU": 52 * 3600}
TEST_TIMEOUT_SECONDS = (60, 120, 300, 900)
# The original eager full fitter takes 294 s for two replicates (run 01303).
# Reserve the same bounded ceiling for both source arms at either extent.
MEASUREMENT_TIMEOUT_SECONDS = {"fixed_fitting": 900}
SEQUENTIAL_PUBLIC_CONSUMERS = (
    (
        'tests/test_sequential_map_covariance.py::test_score_fit_retains_best_exact_cloud_row',
        'tests/test_sequential_map_covariance.py::test_proposal_score_policy_default_and_positional_prefix_are_compatible',
        'tests/test_sequential_map_covariance.py::test_pair_disjoint_holdout_is_opt_in_and_preserves_whole_antithetic_pairs',
        'tests/test_sequential_map_covariance.py::test_pair_disjoint_holdout_rejects_odd_fit_counts_before_target_use',
        'tests/test_sequential_map_covariance.py::test_pair_disjoint_score_fit_recovers_quadratic_with_honest_holdout',
        'tests/test_sequential_map_covariance.py::test_proposal_score_policy_validates_before_target_evaluation',
        'tests/test_sequential_map_covariance.py::test_resolvable_score_gate_repairs_completed_ccma_false_rejections',
        'tests/test_sequential_map_covariance.py::test_resolvable_score_gate_rejects_equal_worse_and_subfloor_changes',
        'tests/test_sequential_map_covariance.py::test_proposal_acceptance_conjunction_remains_fail_closed',
        'tests/test_sequential_map_covariance.py::test_nonfinite_score_norm_fails_both_active_policies',
        'tests/test_sequential_map_covariance.py::test_disabled_score_gate_is_policy_inert',
    ),
    (
        'tests/test_sequential_map_covariance.py::test_disabled_score_gate_has_identical_integrated_behavior',
        'tests/test_sequential_map_covariance.py::test_omitted_policy_matches_explicit_fractional_behavior',
        'tests/test_sequential_map_covariance.py::test_policy_switch_preserves_transactional_center_and_radius',
        'tests/test_sequential_map_covariance.py::test_refinement_movement_diagnostics_are_default_off',
        'tests/test_sequential_map_covariance.py::test_rotated_quadratic_recovers_mode_and_fresh_covariance',
    ),
    (
        'tests/test_sequential_map_covariance.py::test_nonstationary_locator_fails_closed_at_evaluation_budget',
        'tests/test_sequential_map_covariance.py::test_budget_rejection_reports_highest_exact_candidate',
        'tests/test_sequential_map_covariance.py::test_malformed_score_fails_closed',
        'tests/test_sequential_map_covariance.py::test_nonlinear_canary_recovers_after_truncated_locator',
        'tests/test_sequential_map_covariance.py::test_rank_deficient_terminal_fit_fails_closed',
    ),
    (
        'tests/test_sequential_map_covariance.py::test_terminal_cloud_saddle_winner_prevents_curvature_emission',
        'tests/test_sequential_map_covariance.py::test_no_finite_start_fails_closed',
        'tests/test_sequential_map_covariance.py::test_locator_uses_start_centered_standardized_coordinates',
        'tests/test_sequential_map_covariance.py::test_batched_cloud_route_matches_scalar_result',
        'tests/test_sequential_map_covariance.py::test_native_batched_locator_recovers_same_quadratic_mode',
    ),
    (
        'tests/test_sequential_map_covariance.py::test_locator_gradient_tolerance_is_explicit_and_recorded',
        'tests/test_sequential_map_covariance.py::test_locator_gradient_tolerance_must_be_positive_finite',
        'tests/test_sequential_map_covariance.py::test_progress_callback_records_locator_and_terminal_stages',
        'tests/test_sequential_map_covariance.py::test_locator_stopping_condition_is_explicit_and_validated',
        'tests/test_sequential_map_covariance.py::test_center_first_stationary_center_skips_locator_and_fits_geometry',
    ),
    (
        'tests/test_sequential_map_covariance.py::test_center_first_nonstationary_center_uses_local_refinement',
        'tests/test_sequential_map_covariance.py::test_center_first_requires_one_center_and_policy_is_validated',
        'tests/test_sequential_map_covariance.py::test_default_locator_policy_remains_multistart',
        'tests/test_sequential_map_covariance.py::test_terminal_fit_attempt_cap_is_enforced',
        'tests/test_sequential_map_covariance.py::test_terminal_fit_attempt_cap_must_be_positive',
    ),
)
BLOCK_PUBLIC_LEGACY_NAMES = (
    "test_coupled_quadratic_runs_ordered_gauss_seidel_then_detects_reversal",
    "test_scalar_and_batched_routes_match_on_coupled_quadratic",
    "test_default_payload_is_array_free_and_private_payload_is_explicit",
    "test_new_sweep_policy_defaults_are_behavior_and_payload_compatible",
    "test_material_reversal_can_be_recorded_without_stopping_full_sweep",
)
TEST_GROUPS = {
    "driver_history_semantics_cpu": ("tests/test_filter_repair_driver_history.py", "-k", "not fresh_driver_history_memory"),
    **{f"driver_history_memory_{arm}_cpu": (
        f"tests/test_filter_repair_driver_history.py::test_fresh_driver_history_memory[{arm}]",)
        for arm in ("prior", "streamed")},
    "remote_integration_hermite_cpu": ("tests/test_filter_repair_remote_integration.py",),
    "remote_integration_hermite_consumers_cpu": (
        "tests/highdim/test_pair_block_tt_remedy.py",
        "tests/highdim/test_observation_guided_tt_tf.py",
        "tests/highdim/test_c2_ukf_guided_tt_dmis_tf.py"),
    "remote_integration_eigen_cpu": ("tests/test_principal_sqrt_eigen_refinement_tf.py",),
    "remote_integration_genut_primitives_cpu": ("tests/highdim/test_genut_shape_lm_tf.py",),
    "remote_integration_neutra_cpu": (
        "tests/test_neutra_single_authority.py::test_author_free_bias_is_outside_cap_and_has_unrestricted_gradient",
        "tests/test_neutra_single_authority.py::test_hoffman_profile_masks_match_author_tfp_blocks",
        "tests/test_neutra_single_authority.py::test_conditional_map_jacobian_score_inverse_and_logdet[iaf]",
        "tests/test_neutra_single_authority.py::test_layerwise_vaitl_score_matches_density_score[iaf]",
        "tests/test_neutra_single_authority.py::test_frozen_loader_roundtrip_and_tamper_rejection[iaf]",
        "tests/test_neutra_single_authority.py::test_existing_iaf_facades_cannot_contain_numerical_forks"),
    "remaining_svd_cost_analysis_cpu": ("tests/test_filter_repair_remaining_svd_cost_analysis.py",),
    **{f"remaining_svd_cost_{arm}_{dimension}_{device}": (
        f"tests/test_filter_repair_remaining_svd_cost.py::test_remaining_consumer_svd_cost[{arm}-{dimension}]",)
        for device in ("cpu", "gpu") for dimension in (3, 5)
        for arm in ("prior_graph", "prior_xla", "after_graph", "after_xla")},
    **{f"remaining_svd_requalification_{device}": (
        "tests/test_filter_repair_remaining_svd_scale.py",
        "tests/test_filter_repair_accurate_svd.py", "tests/test_filter_repair_dense_svd.py",
        "tests/test_filter_repair_dense_derivatives.py", "tests/test_filter_repair_qr.py",
        "tests/test_filter_repair_svd_scale_qualification.py", "tests/test_filter_repair_srukf_scale.py",
        "tests/test_rectangular_factor_tf.py", "tests/test_rectangular_srukf_tf.py",
        "tests/test_compiled_kalman_ukf_runtime.py") for device in ("cpu", "gpu")},
    **{f"remaining_svd_ownership_{device}": (
        "tests/test_filter_repair_block_capture.py::test_conditional_owner_releases_callback_dependencies",
        "tests/test_filter_repair_geometry_fit.py::test_fit_target_changes_and_resource_ownership",
        "tests/test_filter_repair_geometry_fit.py::test_fit_cache_binds_identity_numerical_settings_and_releases_old_target")
        for device in ("cpu", "gpu")},
    **{f"remaining_svd_scale_{device}": ("tests/test_filter_repair_remaining_svd_scale.py",)
        for device in ("cpu", "gpu")},
    **{f"remaining_svd_derivatives_{device}": ("tests/test_filter_repair_dense_svd.py",
        "tests/test_filter_repair_dense_derivatives.py", "tests/test_filter_repair_qr.py")
        for device in ("cpu", "gpu")},
    **{f"remaining_svd_endpoints_{device}": ("tests/test_filter_repair_block_score_geometry.py",
        "tests/test_block_score_geometry.py", "tests/test_quadratic_geometry.py",
        "tests/test_filter_repair_block_capture.py::test_conditional_owner_releases_callback_dependencies",
        "tests/test_filter_repair_geometry_fit.py::test_fit_target_changes_and_resource_ownership",
        "tests/test_filter_repair_geometry_fit.py::test_fit_cache_binds_identity_numerical_settings_and_releases_old_target")
        for device in ("cpu", "gpu")},
    "remaining_svd_endpoint_crash_cpu": ("-s", "tests/test_quadratic_geometry.py::test_center_refinement_accepts_nearby_mode"),
    "remaining_svd_endpoint_sequence_cpu": ("-s", "-vv",
        "tests/test_filter_repair_block_score_geometry.py", "tests/test_block_score_geometry.py",
        "tests/test_quadratic_geometry.py",
        "tests/test_filter_repair_block_capture.py::test_conditional_owner_releases_callback_dependencies",
        "tests/test_filter_repair_geometry_fit.py::test_fit_target_changes_and_resource_ownership",
        "tests/test_filter_repair_geometry_fit.py::test_fit_cache_binds_identity_numerical_settings_and_releases_old_target"),
    "svd_graph_attribution_gpu": ("tests/test_filter_repair_svd_graph_attribution.py",),
    **{f"dz5_score_oracle_{mode}_{device}": (
        f"tests/test_filter_repair_dz5_score_oracle.py::test_dz5_fresh_score_oracle[{jit}]",)
        for mode, jit in (("graph", "False"), ("xla", "True")) for device in ("cpu", "gpu")},
    "dz5_score_replay_localize_cpu": (
        "tests/test_filter_repair_dz5_score_oracle.py::test_dz5_graph_replay_thread_localization",),
    "dz5_snapshot_import_cpu": ("tests/test_filter_repair_dz5_snapshot.py",),
    "dz5_merged_import_cpu": ("tests/test_filter_repair_dz5_merged.py::test_merged_dz5_snapshot_import",),
    **{f"merged_ledh_boundary_{device}": ("tests/test_filter_repair_merged_ledh_boundary.py",)
        for device in ("cpu", "gpu")},
    **{f"merged_ledh_models_{device}": ("tests/test_filter_repair_merged_ledh_models.py",)
        for device in ("cpu", "gpu")},
    **{f"ledh_seed_compatibility_{device}": ("tests/test_filter_repair_ledh_seed_compatibility.py",)
        for device in ("cpu", "gpu")},
    **{f"ledh_random_native_{device}": ("tests/test_filter_repair_ledh_random_native.py",)
        for device in ("cpu", "gpu")},
    **{f"ledh_value_native_{device}": ("-k", "not localization and not precision_reference and not isolated_cost",
        "tests/test_filter_repair_ledh_value_native.py", "tests/test_filter_repair_slogdet.py")
        for device in ("cpu", "gpu")},
    **{f"ledh_score_native_{device}": ("tests/test_filter_repair_ledh_score_native.py",)
        for device in ("cpu", "gpu")},
    "ledh_value_localize_cpu": ("tests/test_filter_repair_ledh_value_native.py::test_dual_reset_localization",),
    "ledh_value_precision_cpu": ("tests/test_filter_repair_ledh_value_native.py::test_dual_reset_precision_reference",),
    "ledh_flow_localize_gpu": ("tests/test_filter_repair_ledh_value_native.py::test_flow_fraction_localization",),
    **{f"ledh_native_components_{device}": ("-k", "not dual_trust and not localization and not precision_reference and not isolated_cost",
        "tests/test_filter_repair_ledh_value_native.py", "tests/test_filter_repair_slogdet.py",
        "tests/test_filter_repair_ledh_random_native.py") for device in ("cpu", "gpu")},
    **{f"ledh_flow_cost_{arm}_{device}": (
        f"tests/test_filter_repair_ledh_value_native.py::test_flow_isolated_cost[{arm}]",)
        for arm in ("prior_graph", "native_graph", "native_xla") for device in ("cpu", "gpu")},
    "pruned_enclosing_gpu": (
        "tests/test_filter_repair_pruned_enclosing.py", "tests/test_pruned_srukf_tf.py",
        "tests/test_filter_repair_campaign.py", "tests/test_filter_repair_policy.py",
        "tests/test_filter_repair_gpu_selection.py", "tests/test_filter_repair_cost_provenance.py"),
    **{f"dz5_merged_target_{batch}_{device}": (
        f"tests/test_filter_repair_dz5_merged.py::test_merged_dz5_target_graph_xla[{batch}]",)
        for batch in (1, 4, 46, 68) for device in ("cpu", "gpu")},
    **{f"dz5_archived_target_{batch}_{device}": (
        f"tests/test_filter_repair_dz5_merged.py::test_archived_dz5_target_graph_xla[{batch}]",)
        for batch in (1, 4, 46, 68) for device in ("cpu", "gpu")},
    "dense_execution_reporting_cpu": ("tests/test_filter_repair_dense_execution_reporting.py",),
    "svd_cost_analysis_cpu": ("tests/test_filter_repair_svd_cost_analysis.py",),
    "dense_isotropic_initialization_cpu": ("tests/test_filter_repair_dense_isotropic_initialization.py",),
    **{f"svd_cost_{arm}_{horizon}_{device}": (
        f"tests/test_filter_repair_svd_cost.py::test_complete_srukf_svd_cost[{arm}-{horizon}]",)
        for arm in ("prior_graph", "prior_xla", "after_graph", "after_xla")
        for horizon in (1, 3) for device in ("cpu", "gpu")},
    "dense_seeded_attribution_cpu": ("tests/test_filter_repair_dense_seeded.py",),
    **{f"dense_seeded_{dimension}_{case}_{device}": (
        f"tests/test_filter_repair_dense_controller.py::test_complete_original_seeded_controller[{case}-{dimension}]",)
        for dimension in (1, 3) for case in ("healthy", "invalid_locator", "invalid_cloud", "score_veto", "fit_rejected")
        for device in ("cpu", "gpu")},
    **{f"accurate_svd_{device}": ("tests/test_filter_repair_accurate_svd.py",)
        for device in ("cpu", "gpu")},
    **{f"svd_endpoints_{device}": ("tests/test_filter_repair_svd_scale_qualification.py",
        "tests/test_filter_repair_srukf_scale.py", "tests/test_rectangular_factor_tf.py",
        "tests/test_rectangular_srukf_tf.py", "tests/test_compiled_kalman_ukf_runtime.py")
        for device in ("cpu", "gpu")},
    **{f"svd_scale_audit_{device}": ("tests/test_filter_repair_svd_scale_audit.py",)
        for device in ("cpu", "gpu")},
    **{f"dense_rng_{dimension}_{device}": (
        f"tests/test_filter_repair_dense_rng.py::test_original_dense_cloud_stream[{dimension}]",)
        for dimension in (1, 3, 23) for device in ("cpu", "gpu")},
    **{f"dense_controller_{dimension}_{case}_{device}": (
        f"tests/test_filter_repair_dense_controller.py::test_complete_original_dense_attempt_controller[{case}-{dimension}]",)
        for dimension, case in ((1, "moved_retry"), (1, "exhausted"),
            (1, "invalid_second_cloud"), (3, "invalid_second_cloud"),
            (1, "invalid_rank"), (3, "invalid_rank"), (1, "overlap"), (3, "overlap"))
        for device in ("cpu", "gpu")},
    **{f"precision_operator_{device}": ("tests/test_filter_repair_precision_operator.py",
        "tests/test_filter_repair_fixed_stability.py",
        "tests/test_filter_repair_block_score_geometry.py::test_stability_caps_preserve_both_sides_of_boundary")
        for device in ("cpu", "gpu")},
    **{f"dense_controller_svd_{device}": ("tests/test_filter_repair_dense_controller_svd.py",)
        for device in ("cpu", "gpu")},
    **{f"dense_controller_{dimension}_{case}_{device}": (
        f"tests/test_filter_repair_dense_controller.py::test_complete_original_dense_attempt_controller[{case}-{dimension}]",)
        for dimension in (1, 3) for case in ("healthy", "invalid_locator", "invalid_cloud", "score_veto", "fit_rejected")
        for device in ("cpu", "gpu")},
    "tensor_npz_cpu": ("tests/test_filter_repair_tensor_npz.py",),
    "dense_attempt_condition_gpu": ("tests/test_filter_repair_dense_attempt_condition.py",),
    **{f"dense_attempt_{dimension}_{case}_{device}": (
        "tests/test_filter_repair_dense_attempt_composition.py", "-k", f"{case} and {dimension}")
        for dimension in (1, 3) for case in ("centered", "moved", "invalid", "fit_rejected")
        for device in ("cpu", "gpu")},
    **{f"dense_validated_fit_{dimension}_{case}_{device}": (
        "tests/test_filter_repair_dense_validated_fit.py", "-k", f"{case} and {dimension}")
        for dimension, case in ((1, "healthy"), (3, "healthy"), (3, "audit"), (3, "incomplete"), (3, "rank"))
        for device in ("cpu", "gpu")},
    **{f"dense_fit_error_order_{device}": (
        "tests/test_filter_repair_dense_validated_fit.py::test_fitting_error_precedence_is_tensor_native",
        "tests/test_filter_repair_dense_validated_fit.py::test_enclosing_validation_recurrence_and_frozen_derivatives")
        for device in ("cpu", "gpu")},
    "block_buffer_attribution_gpu": ("tests/test_filter_repair_block_buffer_attribution.py",),
    "dense_partition_validation_cpu": ("tests/test_filter_repair_dense_partition_validation.py",),
    "dense_partition_validation_gpu": ("tests/test_filter_repair_dense_partition_validation.py",),
    "dense_initializer_cloud_localization_cpu": (
        "tests/test_filter_repair_dense_initializer_cloud.py", "-k", "healthy and 1"),
    **{f"dense_initializer_cloud_{dimension}_{device}": (
        "tests/test_filter_repair_dense_initializer_cloud.py", "-k", f"{dimension}")
        for dimension in (1, 3) for device in ("cpu", "gpu")},
    "staged_center_rounding_cpu": (
        "tests/test_filter_repair_staged_center_rounding.py",),
    **{f"staged_center_cost_{arm}_{dimension}_{device}": (
        f"tests/test_filter_repair_staged_center_cost.py::test_staged_center_complete_costs[{arm}-{dimension}]",)
        for arm in ("prior", "graph", "xla") for dimension in (1, 3) for device in ("cpu", "gpu")},
    **{f"staged_center_{case}_{dimension}_{device}": (
        f"tests/test_filter_repair_staged_center.py::test_staged_original_records[{dimension}-{case}]",)
        for dimension, case in ((1, "quadratic"), (3, "quadratic"), (1, "quartic"), (3, "quartic"),
            (3, "constant"), (3, "invalid"), (3, "cap"), (3, "cap_after"), (3, "reject"), (3, "validator_error"))
        for device in ("cpu", "gpu")},
    **{f"staged_center_edges_{device}": (
        "tests/test_filter_repair_staged_center.py", "-k", "not original_records and not construction_failure and not execution_label and not supplied_state and not native_compilation and not nested_validator")
        for device in ("cpu", "gpu")},
    **{f"staged_center_failure_isolation_{device}": (
        "tests/test_filter_repair_staged_center.py", "-k", "native_compilation or nested_validator")
        for device in ("cpu", "gpu")},
    **{f"staged_center_state_{device}": (
        "tests/test_filter_repair_staged_center.py", "-k", "supplied_state") for device in ("cpu", "gpu")},
    "staged_center_config_cpu": ("tests/test_filter_repair_staged_center.py", "-k", "execution_label"),
    **{f"staged_center_construction_{device}": (
        "tests/test_filter_repair_staged_center.py", "-k", "construction_failure") for device in ("cpu", "gpu")},
    **{f"target_failure_endpoint_{device}": (
        "tests/test_filter_repair_target_failure_endpoint.py",
        "tests/test_common_inference_runtime_contracts.py", "-k", "target_failure",
        "tests/test_linear_kalman_svd_tf.py::test_target_failure_policy_does_not_activate_on_valid_lgssm_value",
    ) for device in ("cpu", "gpu")},
    **{f"consensus_endpoint_{device}": (
        "tests/test_filter_repair_consensus_endpoint.py",
        "tests/test_fixed_center_curvature.py::test_consensus_shrinkage_and_geometry_metrics_are_spd_and_oriented",
        "tests/test_fixed_center_curvature.py::test_consensus_tensorflow_kernel_has_xla_value_and_gradient_parity",)
        for device in ("cpu", "gpu")},
    **{f"process_containment_{device}": (
        "tests/test_filter_repair_process_containment.py",) for device in ("cpu", "gpu")},
    "dz5_current_deadline_cpu": ("tests/test_filter_repair_dz5_supervision.py",),
    **{f"batched_center_enclosing_{device}": (
        "tests/test_filter_repair_batched_center_reuse.py::test_batched_locator_resets_inside_enclosing_xla_recurrence",)
        for device in ("cpu", "gpu")},
    **{f"batched_center_reuse_{case}_{batch}_{device}": (
        f"tests/test_filter_repair_batched_center_reuse.py::test_batched_locator_reuses_operand_starts[{case}-{batch}]",)
        for case in ("quadratic", "nonquadratic", "flat", "invalid_rows")
        for batch in (1, 3) for device in ("cpu", "gpu")},
    "block_public_cost_analysis": ("tests/test_filter_repair_block_cost_analysis.py",),
    **{f"block_public_cost_{arm}_{dimension}_{device}": (
        f"tests/test_filter_repair_block_public_memory.py::test_complete_public_block_costs[{arm}-{dimension}]",)
        for arm in ("prior", "graph", "xla") for dimension in (3, 5) for device in ("cpu", "gpu")},
    **{f"block_public_churn_{device}": (
        "tests/test_filter_repair_block_public_churn.py::test_bounded_public_block_callback_and_shape_churn",)
        for device in ("cpu", "gpu")},
    "initializer_native_residual_gpu": (
        "tests/test_filter_repair_initializer_arithmetic.py::test_initializer_residual_correction_candidate",),
    **{f"block_public_options_{case}_{device}": (
        f"tests/test_filter_repair_block_public_options.py::test_public_block_configured_dependencies[{case}]",)
        for case in ("scalar_locator", "factor_one", "factor_two_reuse", "paired", "scaled_search", "score_disabled")
        for device in ("cpu", "gpu")},
    **{f"initializer_coefficient_reference_{batched}_{device}": (
        f"tests/test_filter_repair_initializer_coefficient_reference.py::test_identical_arrays_high_precision_coefficient_reference[{batched}]",)
        for batched in (False, True) for device in ("cpu", "gpu")},
    **{f"block_public_{case}_{batched}_{device}": (
        f"tests/test_filter_repair_block_public.py::test_complete_public_block_matches_original[{case}-{batched}]",)
        for case in ("coupled", "record_reversal", "partial_heterogeneous")
        for batched in (False, True) for device in ("cpu", "gpu")},
    **{f"block_public_edges_{device}": ("tests/test_filter_repair_block_public.py",
        "-k", "not complete_public_block") for device in ("cpu", "gpu")},
    **{f"block_public_legacy_real_{index}_{device}": (
        f"tests/test_block_coordinate_center.py::{name}",)
        for index, name in enumerate(BLOCK_PUBLIC_LEGACY_NAMES) for device in ("cpu", "gpu")},
    **{f"block_public_legacy_misc_{device}": ("tests/test_block_coordinate_center.py",
        "-k", "not (" + " or ".join(BLOCK_PUBLIC_LEGACY_NAMES) + ")") for device in ("cpu", "gpu")},
    **{f"block_public_legacy_reference_{coupling}_{stop}_{batched}_{device}": (
        f"tests/test_filter_repair_block_center.py::test_complete_public_and_private_records_match_original[{coupling}-{stop}-{batched}]",)
        for coupling, stop in ((0.0, True), (3.0, True), (3.0, False))
        for batched in (False, True) for device in ("cpu", "gpu")},
    **{f"block_public_legacy_helpers_{device}": ("tests/test_filter_repair_block_center.py",
        "-k", "not complete_public_and_private") for device in ("cpu", "gpu")},
    **{f"objective_resolution_controlled_{batched}_{device}": (
        f"tests/test_filter_repair_objective_resolution.py::test_controlled_completed_error_skips_mass_and_block_handoff[{batched}]",)
        for batched in (False, True) for device in ("cpu", "gpu")},
    **{f"resolution_backend_{batched}_{device}": (
        f"tests/test_filter_repair_resolution_backend_diagnostic.py::test_backend_resolution_attribution[{batched}]",)
        for batched in (False, True) for device in ("cpu", "gpu")},
    **{f"objective_resolution_boundaries_{device}": (
        "tests/test_filter_repair_objective_resolution.py::test_exact_resolution_boundaries",)
        for device in ("cpu", "gpu")},
    **{f"objective_resolution_sequential_{batched}_{device}": (
        f"tests/test_filter_repair_objective_resolution.py::test_completed_error_preserves_lifecycle_and_skips_mass[{batched}]",)
        for batched in (False, True) for device in ("cpu", "gpu")},
    **{f"objective_resolution_block_{batched}_{device}": (
        f"tests/test_filter_repair_objective_resolution.py::test_public_block_error_prevents_replay_and_later_blocks[{batched}]",)
        for batched in (False, True) for device in ("cpu", "gpu")},
    "objective_resolution_diagnostic_cpu": ("tests/test_filter_repair_objective_resolution_diagnostic.py",),
    **{f"block_boundaries_{device}": (
        "tests/test_filter_repair_block_boundaries.py::test_transaction_veto_stops_later_blocks",)
        for device in ("cpu", "gpu")},
    **{f"block_outer_ownership_{device}": (
        "tests/test_filter_repair_block_boundaries.py::test_outer_owner_release_preserves_retained_compiled_handle",)
        for device in ("cpu", "gpu")},
    "block_rounding_diagnostic_cpu": ("tests/test_filter_repair_block_rounding_diagnostic.py::test_partial_block_rounding_attribution",),
    "block_target_arithmetic_cpu": ("tests/test_filter_repair_block_rounding_diagnostic.py::test_identical_point_target_arithmetic",),
    **{f"block_controller_{case}_{batched}_{device}": (
        f"tests/test_filter_repair_block_controller.py::test_complete_ordered_block_matches_original[{case}-{batched}]",)
        for case in ("coupled", "record_reversal", "partial_heterogeneous")
        for batched in (False, True) for device in ("cpu", "gpu")},
    **{f"block_dependency_derivatives_{device}": (
        "tests/test_filter_repair_eigen_consumers.py",
        "tests/test_filter_repair_independent_primitives.py",
        "tests/test_filter_repair_structured_preparation.py::test_preparation_pullbacks_preserve_original_tensors",
        "tests/test_filter_repair_mass_matrix.py::test_public_mass_records_preserve_projection_and_floor_decisions",
        "tests/test_filter_repair_mass_matrix.py::test_public_mass_keeps_fail_closed_validation",
        "tests/test_filter_repair_mass_matrix.py::test_matrix_pullback_retains_original_frozen_floor_boundary",
        "tests/test_filter_repair_mass_matrix.py::test_compiled_precision_gradient_matches_inverse_identity_and_finite_difference")
        for device in ("cpu", "gpu")},
    "block_graph_registry_diagnostic_cpu": (
        "tests/test_filter_repair_block_graph_diagnostic.py",),
    **{f"block_capture_{batched}_{device}": (
        f"tests/test_filter_repair_block_capture.py::test_conditional_full_center_is_dynamic_and_matches_original[{batched}]",)
        for batched in (False, True) for device in ("cpu", "gpu")},
    **{f"block_capture_ownership_{device}": (
        "tests/test_filter_repair_block_capture.py::test_conditional_owner_releases_callback_dependencies",
        "tests/test_filter_repair_block_capture.py::test_hlo_metadata_normalization_preserves_executable_and_source_changes")
        for device in ("cpu", "gpu")},
    **{f"sequential_residency_{primed}_{dimension}_{device}": (
        f"tests/test_filter_repair_sequential_residency.py::test_sequential_startup_and_reuse_residency[{primed}-{dimension}]",)
        for primed in (False, True) for dimension in (3, 5) for device in ("cpu", "gpu")},
    **{f"sequential_residency_observer_{device}": (
        "tests/test_filter_repair_sequential_residency.py::test_sequential_residency_observer_retention_control",)
        for device in ("cpu", "gpu")},
    "sequential_public_cost_analysis": ("tests/test_filter_repair_sequential_cost_analysis.py",),
    **{f"sequential_public_cost_{arm}_{dimension}_{device}": (
        f"tests/test_filter_repair_sequential_public_memory.py::test_complete_public_sequential_costs[{arm}-{dimension}]",)
        for arm in ("prior", "graph", "xla") for dimension in (3, 5) for device in ("cpu", "gpu")},
    "program_ownership_scope_cpu": ("tests/test_filter_repair_program_ownership.py", "-k", "factory_scope"),
    **{f"program_ownership_{case}_{device}": (
        f"tests/test_filter_repair_program_ownership.py::test_actual_owner_releases_nested_callbacks_and_keeps_retained_handle[{case}]",)
        for case in ("terminal", "factor_one", "batched_locator") for device in ("cpu", "gpu")},
    **{f"program_ownership_identity_{device}": (
        "tests/test_filter_repair_program_ownership.py::test_distinct_public_target_identities_do_not_reuse_stale_graphs",)
        for device in ("cpu", "gpu")},
    **{f"posterior_residency_observer_{device}": (
        "tests/test_filter_repair_posterior_residency.py::test_posterior_residency_observer_retention_control",)
        for device in ("cpu", "gpu")},
    **{f"sequential_public_{case}_{device}": (
        f"tests/test_filter_repair_sequential_public.py::test_public_original_records_and_target_order[{case}]",)
        for case in ("terminal", "terminal_reject", "symmetric", "recenter", "fit_reject",
            "factor_one", "factor_two", "factor_two_reuse", "stationary_budget", "moving_budget",
            "nonfinite", "paired", "scaled_search", "score_disabled", "resolvable", "scalar_locator",
            "batched_locator", "locator_budget") for device in ("cpu", "gpu")},
    **{f"sequential_public_edges_{device}": (
        "tests/test_filter_repair_sequential_public.py", "-k", "not original_records")
        for device in ("cpu", "gpu")},
    **{f"sequential_public_consumers_{index}_{device}": targets
        for index, targets in enumerate(SEQUENTIAL_PUBLIC_CONSUMERS, start=1) for device in ("cpu", "gpu")},
    **{f"sequential_public_consumers_factor_{device}": ("tests/test_factor_correlation_geometry.py",)
        for device in ("cpu", "gpu")},
    **{f"sequential_public_consumers_boundary_{device}": ("tests/test_filter_repair_locator_frozen.py",
        "tests/test_filter_repair_lifecycle_original.py::test_original_full_lifecycle_records_and_target_order[symmetric]")
        for device in ("cpu", "gpu")},
    **{f"posterior_residency_{primed}_{dimension}_{device}": (
        f"tests/test_filter_repair_posterior_residency.py::test_posterior_startup_and_reuse_residency[{primed}-{dimension}]",)
        for primed in (False, True) for dimension in (3, 5) for device in ("cpu", "gpu")},
    **{f"sequential_controller_{case}_{device}": (
        f"tests/test_filter_repair_sequential_controller.py::test_complete_original_records_and_target_order[{case}]",)
        for case in ("terminal", "terminal_reject", "symmetric", "recenter", "fit_reject",
            "factor_one", "factor_two", "factor_two_reuse", "stationary_budget", "moving_budget",
            "nonfinite", "paired", "scaled_search", "score_disabled", "resolvable", "scalar_locator",
            "batched_locator", "locator_budget") for device in ("cpu", "gpu")},
    **{f"sequential_controller_edges_{device}": (
        "tests/test_filter_repair_sequential_controller.py", "-k", "not complete_original")
        for device in ("cpu", "gpu")},
    **{f"posterior_public_memory_{arm}_{dimension}_{device}": (
        f"tests/test_filter_repair_posterior_public_memory.py::test_complete_public_posterior_costs[{arm}-{dimension}]",)
        for arm in ("prior", "graph", "xla") for dimension in (3, 5) for device in ("cpu", "gpu")},
    **{f"posterior_public_{dimension}_{device}": (
        "tests/test_filter_repair_posterior_public.py", "-k", f"original_records and {dimension}")
        for dimension in (1, 3, 5) for device in ("cpu", "gpu")},
    **{f"posterior_public_edges_{device}": (
        "tests/test_filter_repair_posterior_public.py", "-k", "not original_records")
        for device in ("cpu", "gpu")},
    **{f"posterior_public_regression_{device}": (
        "tests/test_posterior_curvature_refinement.py", "tests/test_posterior_curvature_refinement_regressions.py")
        for device in ("cpu", "gpu")},
    **{f"posterior_public_dense_{device}": (
        "tests/test_filter_repair_dense_condition.py", "-k", "posterior_consumer")
        for device in ("cpu", "gpu")},
    "initializer_native_residual_cpu": (
        "tests/test_filter_repair_initializer_arithmetic.py::test_initializer_residual_correction_candidate",),
    "initializer_native_lstsq_cpu": (
        "tests/test_filter_repair_initializer_arithmetic.py::test_initializer_least_squares_attribution",),
    **{f"initializer_controller_{dimension}_{batched}_{device}": (
        "tests/test_filter_repair_initializer_native.py", "-k", f"controller_records and {dimension} and {batched}")
        for dimension in (1, 3) for batched in (False, True) for device in ("cpu", "gpu")},
    **{f"initializer_controller_edges_{device}": (
        "tests/test_filter_repair_initializer_native.py", "-k", "typed_target or reuses_changed")
        for device in ("cpu", "gpu")},
    "initializer_native_affine_cpu": (
        "tests/test_filter_repair_initializer_native.py::test_initializer_affine_boundary_attribution",),
    "initializer_native_roundoff_cpu": (
        "tests/test_filter_repair_initializer_native.py::test_initializer_roundoff_attribution",),
    **{f"initializer_native_{dimension}_{batched}_{device}": (
        "tests/test_filter_repair_initializer_native.py", "-k", f"original_records and {dimension} and {batched}")
        for dimension in (1, 3) for batched in (False, True) for device in ("cpu", "gpu")},
    "initializer_native_smoke_cpu": ("tests/test_filter_repair_initializer_native.py", "-k",
        "original_records and 1 and False and (iterative or single)"),
    **{f"joint_native_failures_{device}": (
        "tests/test_filter_repair_joint_center_native.py", "-k", "synthetic_failures")
        for device in ("cpu", "gpu")},
    "initializer_iterative_gpu": (
        "tests/test_quadratic_map_covariance.py::test_iterative_initializer_reaches_terminal_center_with_bounded_steps", "-s"),
    "joint_native_rounding_cpu": ("tests/test_filter_repair_joint_center_native.py", "-k",
        "affine_rounding or (original_records and 1 and (quadratic or quartic))"),
    **{f"joint_native_{dimension}_{device}": (
        "tests/test_filter_repair_joint_center_native.py", "-k", f"original_records and {dimension}")
        for dimension in (1, 3) for device in ("cpu", "gpu")},
    **{f"joint_native_edges_{device}": (
        "tests/test_filter_repair_joint_center_native.py", "-k", "not original_records")
        for device in ("cpu", "gpu")},
    "initializer_small_cpu": ("tests/test_quadratic_map_covariance.py", "-k",
        "requires_constrained or fails_closed_at_refinement_budget or locator_exception or nonfinite_initial or insufficient_samples or payload_nonclaims"),
    "initializer_gaussian_cpu": ("tests/test_quadratic_map_covariance.py::test_quadratic_initializer_recovers_gaussian_mode_and_covariance",),
    "initializer_scaled_cpu": ("tests/test_quadratic_map_covariance.py::test_scaled_quadratic_initializer_returns_original_coordinate_mass",),
    "initializer_moved_cpu": ("tests/test_quadratic_map_covariance.py::test_one_shot_wrapper_never_emits_covariance_at_a_moved_candidate",),
    "initializer_batch_cpu": ("tests/test_quadratic_map_covariance.py::test_initializer_forwards_batched_design_callback_without_changing_result",),
    "initializer_iterative_cpu": ("tests/test_quadratic_map_covariance.py::test_iterative_initializer_reaches_terminal_center_with_bounded_steps",),
    "initializer_enabled_cpu": ("tests/test_quadratic_map_covariance.py::test_enabled_locator_is_finite_locator_only_not_covariance_authority",),
    "initializer_helpers_cpu": ("tests/test_filter_repair_quadratic_initializer.py", "-k", "not existing_initializer_outcomes"),
    "initializer_current_clouds_cpu": ("tests/test_filter_repair_quadratic_initializer.py", "-k", "existing_initializer_outcomes"),
    "geometry_public_first_cpu": ("tests/test_quadratic_geometry.py", "-k",
        "undersampled or spd_and_condition or nonfinite_values or bad_holdout or holdout_gate or center_refinement_accepts_nearby"),
    "geometry_public_second_cpu": ("tests/test_quadratic_geometry.py", "-k",
        "center_refinement_rejects_out or seed_and_payload or malformed_batched or spd_quadratic_trust_region or constrained_refinement or default_refinement or retains_best"),
    "geometry_public_parity_cpu": ("tests/test_filter_repair_geometry_parity.py",),
    "geometry_public_memory_attribution_cpu": (
        "tests/test_quadratic_geometry.py", "tests/test_filter_repair_geometry_parity.py", "-s", "-k",
        "not synthetic_low_rank_spd and not batched_design_route_matches"),
    "geometry_public_boundary_cpu": (
        "tests/test_quadratic_geometry.py::test_constrained_refinement_accepts_exact_boundary_quadratic_step", "-s"),
    "geometry_full_prior_attribution_cpu": (
        "tests/test_filter_repair_geometry_full_memory.py::test_prior_geometry_spectral_attribution",),
    **{f"geometry_public_capacity_{device}": (
        "tests/test_quadratic_geometry.py::test_synthetic_low_rank_spd_quadratic_recovers_precision",
        "tests/test_quadratic_geometry.py::test_batched_design_route_matches_scalar_geometry")
        for device in ("cpu", "gpu")},
    **{f"geometry_public_remaining_{device}": (
        "tests/test_quadratic_geometry.py", "tests/test_filter_repair_geometry_parity.py", "-k",
        "not synthetic_low_rank_spd and not batched_design_route_matches")
        for device in ("cpu", "gpu")},
    **{f"geometry_full_memory_{arm}_{capacity}_{device}": (
        f"tests/test_filter_repair_geometry_full_memory.py::test_full_geometry_costs[{arm}-{capacity}]",)
        for arm in ("prior", "prior_refined", "graph", "xla") for capacity in (24, 120) for device in ("cpu", "gpu")},
    "geometry_full_public_cpu": ("tests/test_filter_repair_geometry_full.py", "-k", "public_geometry"),
    "geometry_full_legacy_cpu": (
        "tests/test_filter_repair_geometry_parity.py::test_complete_geometry_matches_baseline_on_identical_clouds",
        "tests/test_quadratic_geometry.py::test_geometry_retains_best_exact_design_row",),
    "geometry_full_shapes_cpu": ("tests/test_filter_repair_geometry_full.py::test_full_geometry_output_shapes",),
    "geometry_full_smoke_cpu": (
        "tests/test_filter_repair_geometry_full.py::test_full_geometry_original_records[gaussian-False-3]",
        "tests/test_filter_repair_geometry_full.py::test_full_geometry_original_records[gaussian-True-3]"),
    **{f"geometry_full_{dimension}_{batched}_{device}": (
        "tests/test_filter_repair_geometry_full.py", "-k", f"original_records and {batched}-{dimension}")
        for dimension in (1, 3, 5) for batched in ("False", "True") for device in ("cpu", "gpu")},
    **{f"geometry_full_edges_{device}": (
        "tests/test_filter_repair_geometry_full.py", "-k", "not original_records")
        for device in ("cpu", "gpu")},
    "geometry_active_pilot_attribution_cpu": (
        "tests/test_filter_repair_geometry_active_pilot.py::test_active_pilot_degenerate_basis_and_hlo_attribution",),
    **{f"geometry_active_pilot_{dimension}_{device}": (
        "tests/test_filter_repair_geometry_active_pilot.py", "-k", f"complete_original and {dimension}")
        for dimension in (3, 5) for device in ("cpu", "gpu")},
    **{f"geometry_active_pilot_edges_{device}": (
        "tests/test_filter_repair_geometry_active_pilot.py", "-k", "invalid_counts or guard_relative_margin")
        for device in ("cpu", "gpu")},
    **{f"geometry_active_memory_{arm}_{capacity}_{device}": (
        f"tests/test_filter_repair_geometry_active_memory.py::test_active_fit_costs[{arm}-{capacity}]",)
        for arm in ("compact", "graph", "xla") for capacity in (16, 24) for device in ("cpu", "gpu")},
    **{f"geometry_public_guard_{device}": (
        "tests/test_quadratic_geometry.py", "tests/test_filter_repair_geometry_parity.py")
        for device in ("cpu", "gpu")},
    "geometry_active_rank_diagnostic_cpu": (
        "tests/test_filter_repair_geometry_active_rows.py::test_single_row_and_rank_cutoff_use_active_extent",),
    **{f"geometry_active_{dimension}_{device}": (
        f"tests/test_filter_repair_geometry_active_rows.py::test_active_fit_preserves_complete_original_records[gaussian-{dimension}]",
        f"tests/test_filter_repair_geometry_active_rows.py::test_active_fit_preserves_complete_original_records[nonquadratic-{dimension}]",
        f"tests/test_filter_repair_geometry_active_rows.py::test_active_fit_preserves_complete_original_records[zero_design-{dimension}]",
        f"tests/test_filter_repair_geometry_active_rows.py::test_active_fit_preserves_complete_original_records[reject_holdout-{dimension}]",)
        for dimension in (1, 3, 5) for device in ("cpu", "gpu")},
    **{f"geometry_active_edges_{device}": (
        "tests/test_filter_repair_geometry_active_rows.py", "-k", "not complete_original") for device in ("cpu", "gpu")},
    "gap_ownership_supervisor_cpu": ("tests/test_filter_repair_gap_followup.py",),
    **{f"gap_enclosure_diagnostics_{device}": (
        "tests/test_filter_repair_gap_diagnostics.py", "-k", "not compiler_memory")
        for device in ("cpu", "gpu")},
    **{f"gap_compiler_memory_{arm}_{device}": (
        f"tests/test_filter_repair_gap_diagnostics.py::test_fit_compiler_memory_release[{jit}]",)
        for arm, jit in (("graph", False), ("xla", True)) for device in ("cpu", "gpu")},
    **{f"geometry_preparation_reuse_{kind}_{dimension}_{device}": (
        f"tests/test_filter_repair_geometry_preparation_memory.py::test_preparation_components_and_changing_input_reuse[{kind}-{dimension}]",)
        for kind in ("directions", "partition") for dimension in (3, 5) for device in ("cpu", "gpu")},
    **{f"geometry_preparation_boundary_{device}": (
        "tests/test_filter_repair_geometry_preparation.py::test_extreme_direction_and_holdout_rounding_boundaries",)
        for device in ("cpu", "gpu")},
    **{f"geometry_preparation_memory_{kind}_{arm}_{dimension}_{device}": (
        f"tests/test_filter_repair_geometry_preparation_memory.py::test_preparation_costs[{kind}-{arm}-{dimension}]",)
        for kind in ("directions", "partition", "design_scalar", "design_batch") for arm in ("before", "graph", "xla")
        for dimension in (3, 5) for device in ("cpu", "gpu")},
    "geometry_preparation_smoke": ("tests/test_filter_repair_geometry_preparation.py::test_original_finite_partition[ordinary-3]",),
    **{f"geometry_preparation_{device}": ("tests/test_filter_repair_geometry_preparation.py",) for device in ("cpu", "gpu")},
    **{f"geometry_pilot_memory_{lane}_{arm}_{dimension}_{device}": (
        f"tests/test_filter_repair_geometry_pilot_memory.py::test_complete_pilot_costs[{batched}-{arm}-{dimension}]",)
        for lane, batched in (("scalar", False), ("batch", True)) for arm in ("before", "graph", "xla")
        for dimension in (3, 5) for device in ("cpu", "gpu")},
    **{f"geometry_pilot_extras_{device}": (
        "tests/test_filter_repair_geometry_pilot.py", "-k", "resources or failure") for device in ("cpu", "gpu")},
    "geometry_pilot_smoke": (
        "tests/test_filter_repair_geometry_pilot.py::test_original_pilot_records[gaussian-False-3]",),
    **{f"geometry_pilot_{dimension}_{device}": (
        "tests/test_filter_repair_geometry_pilot.py", "-k", f"original_pilot and {dimension}")
        for dimension in (1, 3, 5) for device in ("cpu", "gpu")},
    **{f"geometry_pilot_operands_{device}": (
        "tests/test_filter_repair_geometry_pilot.py", "-k", "runtime_inputs") for device in ("cpu", "gpu")},
    **{f"geometry_fit_lifetime_{device}": (
        "tests/test_filter_repair_geometry_fit.py::test_fit_target_changes_and_resource_ownership",)
        for device in ("cpu", "gpu")},
    **{f"geometry_fit_cache_{device}": (
        "tests/test_filter_repair_geometry_fit.py::test_fit_cache_binds_identity_numerical_settings_and_releases_old_target",)
        for device in ("cpu", "gpu")},
    **{f"geometry_fit_memory_{arm}_{dimension}_{device}": (
        f"tests/test_filter_repair_geometry_fit_memory.py::test_complete_geometry_fit_costs[{arm}-{dimension}]",)
        for arm in ("before", "graph", "xla") for dimension in (3, 5) for device in ("cpu", "gpu")},
    "geometry_fit_smoke": (
        "tests/test_filter_repair_geometry_fit.py::test_full_original_geometry_fit_suffix[gaussian-3]",),
    **{f"geometry_fit_{dimension}_{device}": (
        "tests/test_filter_repair_geometry_fit.py", "-k", f"full_original and {dimension}")
        for dimension in (1, 3, 5) for device in ("cpu", "gpu")},
    **{f"geometry_fit_operands_{device}": (
        "tests/test_filter_repair_geometry_fit.py", "-k", "not full_original") for device in ("cpu", "gpu")},
    **{f"geometry_control_callbacks_{device}": (
        "tests/test_filter_repair_geometry_callback_contract.py",)
        for device in ("cpu", "gpu")},
    **{f"geometry_control_memory_{arm}_{dimension}_{device}": (
        f"tests/test_filter_repair_geometry_control_memory.py::test_complete_center_proposal_costs[{arm}-{dimension}]",)
        for arm in ("before", "graph", "xla") for dimension in (3, 5) for device in ("cpu", "gpu")},
    "geometry_control_scalar_operands": (
        "tests/test_filter_repair_geometry_control.py::test_full_center_proposal_records_against_original[interior-1]",
        "tests/test_filter_repair_geometry_control.py::test_changed_inputs_are_runtime_operands_and_enclosing_xla"),
    "geometry_control_scalar_attribution": (
        "tests/test_filter_repair_geometry_control.py::test_scalar_graph_and_lu_attribution",),
    "geometry_control_eigen_attribution": (
        "tests/test_filter_repair_geometry_control.py::test_trust_eigensystem_attribution",),
    "geometry_control_smoke": (
        "tests/test_filter_repair_geometry_control.py::test_full_center_proposal_records_against_original[interior-3]",),
    **{f"geometry_control_{device}": ("tests/test_filter_repair_geometry_control.py", "-k", "not attribution")
        for device in ("cpu", "gpu")},
    **{f"posterior_curvature_ill_conditioned_{device}": (
        "tests/test_filter_repair_posterior_curvature_extras.py::test_deficient_design_preserves_full_rejection[ill_conditioned]",)
        for device in ("cpu", "gpu")},
    **{f"posterior_curvature_growth_{replicates}_{device}": (
        f"tests/test_filter_repair_posterior_growth.py::test_native_posterior_capacity_and_warm_memory[{replicates}]",)
        for replicates in (2, 4, 8) for device in ("cpu", "gpu")},
    "posterior_curvature_zero_diagnostic_gpu": ("tests/test_filter_repair_posterior_zero_diagnostic.py",),
    "posterior_curvature_zero_gpu": (
        "tests/test_filter_repair_posterior_curvature_extras.py::test_deficient_design_preserves_full_rejection[zero]",),
    **{f"posterior_curvature_numerical_vetoes_{device}": (
        "tests/test_filter_repair_posterior_numerical_vetoes.py",) for device in ("cpu", "gpu")},
    **{f"posterior_curvature_memory_{arm}_{dimension}_{device}": (
        f"tests/test_filter_repair_posterior_curvature_memory.py::test_complete_native_posterior_costs[{arm}-{dimension}]",)
        for arm in ("before", "graph", "xla") for dimension in (3, 5) for device in ("cpu", "gpu")},
    **{f"posterior_curvature_extras_qualified_{device}": (
        "tests/test_filter_repair_posterior_curvature_extras.py",) for device in ("cpu", "gpu")},
    "posterior_curvature_condition_diagnostic": ("tests/test_filter_repair_posterior_condition_diagnostic.py",),
    "cod_tail_cpu": ("tests/test_filter_repair_qr.py", "tests/test_filter_repair_active_cod_runtime.py"),
    "cod_tail_gpu": ("tests/test_filter_repair_qr.py", "tests/test_filter_repair_active_cod_runtime.py"),
    "posterior_curvature_rank_one_cpu": (
        "tests/test_filter_repair_posterior_curvature_extras.py::test_deficient_design_preserves_full_rejection[rank_one]",),
    "posterior_curvature_rank_diagnostic": ("tests/test_filter_repair_posterior_rank_diagnostic.py",),
    **{f"posterior_curvature_extras_{device}": ("tests/test_filter_repair_posterior_curvature_extras.py",)
        for device in ("cpu", "gpu")},
    "posterior_curvature_streams_cpu": ("tests/test_filter_repair_posterior_curvature.py", "-k", "streams or consensus"),
    "posterior_curvature_streams_gpu": ("tests/test_filter_repair_posterior_curvature.py", "-k", "streams or consensus"),
    "posterior_curvature_smoke_cpu": ("tests/test_filter_repair_posterior_curvature.py::test_complete_original_controller_records[gaussian-3]",),
    "posterior_curvature_smoke_gpu": ("tests/test_filter_repair_posterior_curvature.py::test_complete_original_controller_records[gaussian-3]",),
    **{f"posterior_curvature_{dimension}_{device}": tuple(
        f"tests/test_filter_repair_posterior_curvature.py::test_complete_original_controller_records[{case}-{dimension}]"
        for case in ("gaussian", "nonquadratic", "nonspd", "transformed", "nonfinite_position",
            "ineligible_center", "ineligible_training", "ineligible_selection", "ineligible_audit", "ineligible_proposal",
            "value_center", "score_training", "score_selection", "value_audit", "score_proposal", "reject_audit", "reject_proposal"))
        for dimension in (1, 3, 5) for device in ("cpu", "gpu")},
    **{f"posterior_curvature_operands_{device}": (
        "tests/test_filter_repair_posterior_curvature.py::test_ball_replicates_and_changed_inputs_keep_runtime_operands",)
        for device in ("cpu", "gpu")},
    **{f"uniform_public_{device}": ("tests/test_filter_repair_uniform_public.py",) for device in ("cpu", "gpu")},
    **{f"uniform_public_memory_{arm}_{dimension}_{device}": (
        f"tests/test_filter_repair_uniform_public_memory.py::test_public_uniform_refinement_costs[{arm}-{dimension}]",)
        for arm in ("before", "graph", "xla") for dimension in (3, 5) for device in ("cpu", "gpu")},
    **{f"uniform_extras_{device}": ("tests/test_filter_repair_uniform_rounds_extras.py",) for device in ("cpu", "gpu")},
    **{f"uniform_round_growth_{rounds}_{device}": (
        f"tests/test_filter_repair_uniform_round_growth.py::test_complete_controller_capacity_and_warm_allocations[{rounds}]",)
        for rounds in (1, 4, 8) for device in ("cpu", "gpu")},
    "dense_svd_cpu": ("tests/test_filter_repair_dense_svd.py",),
    "dense_svd_gpu": ("tests/test_filter_repair_dense_svd.py",),
    **{f"uniform_resources_{device}": ("tests/test_filter_repair_uniform_resources.py",) for device in ("cpu", "gpu")},
    "uniform_rounds_smoke": ("tests/test_filter_repair_uniform_rounds.py::test_complete_uniform_original_records[move-3]",),
    **{f"uniform_rounds_{dimension}_{device}": tuple(
        f"tests/test_filter_repair_uniform_rounds.py::test_complete_uniform_original_records[{case}-{dimension}]"
        for case in ("centered", "move", "nonquadratic", "round_limit", "invalid_initial", "invalid_replay",
            "invalid_fit", "invalid_check", "invalid_proposal", "invalid_final_replay", "replay_value",
            "initial_evidence", "bad_evidence", "nonspd", "nonfinite_scaled_score", "factor_failure"))
        for dimension in (1, 3, 5) for device in ("cpu", "gpu")},
    **{f"uniform_operands_{device}": (
        "tests/test_filter_repair_uniform_rounds.py::test_uniform_changed_inputs_partial_batches_and_hlo",
        "tests/test_filter_repair_uniform_rounds.py::test_uniform_cache_keeps_seed_dynamic_and_configuration_distinct")
        for device in ("cpu", "gpu")},
    **{f"uniform_round_memory_{arm}_{dimension}_{device}": (
        f"tests/test_filter_repair_uniform_round_memory.py::test_complete_uniform_controller_costs[{arm}-{dimension}]",)
        for arm in ("before", "graph", "xla") for dimension in (3, 5) for device in ("cpu", "gpu")},
    "dense_extreme_comparison": ("tests/test_filter_repair_dense_extreme_comparison.py",),
    "quadratic_numerics_cpu": ("tests/test_filter_repair_quadratic_numerics.py",),
    "dense_boundaries_cpu": ("tests/test_filter_repair_dense_boundaries.py",),
    "dense_boundaries_gpu": ("tests/test_filter_repair_dense_boundaries.py",),
    "dense_components_cpu": ("tests/test_filter_repair_dense_cost_components.py",),
    "dense_components_gpu": ("tests/test_filter_repair_dense_cost_components.py",),
    "dense_extreme_cpu": ("tests/test_filter_repair_dense_extreme_diagnostic.py",),
    "dense_extreme_gpu": ("tests/test_filter_repair_dense_extreme_diagnostic.py",),
    "dense_derivatives_cpu": ("tests/test_filter_repair_dense_derivatives.py",),
    "dense_derivatives_gpu": ("tests/test_filter_repair_dense_derivatives.py",),
    "factor_record_comparison": ("tests/test_filter_repair_record_comparison.py",),
    "factor_equivalence": ("tests/test_filter_repair_factor_equivalence.py",),
    "dense_condition": ("tests/test_filter_repair_dense_condition.py",),
    "factor_equivalence_gpu": ("tests/test_filter_repair_factor_equivalence.py",),
    "dense_condition_gpu": ("tests/test_filter_repair_dense_condition.py",),
    **{f"dense_numerics_memory_{arm}_{dimension}_{device}": (
        f"tests/test_filter_repair_quadratic_numerics_memory.py::test_numerical_dependency_costs[{arm}-dense-{dimension}]",)
        for arm in ("before", "graph", "xla") for dimension in (3, 5) for device in ("cpu", "gpu")},
    **{f"quadratic_public_memory_{arm}_{dimension}_{device}": (
        f"tests/test_filter_repair_quadratic_public_memory.py::test_public_paired_refinement_costs[{arm}-{dimension}]",)
        for arm in ("before", "graph", "xla") for dimension in (3, 5) for device in ("cpu", "gpu")},
    "quadratic_round_cache_cpu": ("tests/test_filter_repair_quadratic_round_cache.py",),
    "quadratic_round_cache_gpu": ("tests/test_filter_repair_quadratic_round_cache.py",),
    "quadratic_round_lifetime_cpu": ("tests/test_filter_repair_quadratic_round_lifetime.py",),
    "quadratic_round_lifetime_gpu": ("tests/test_filter_repair_quadratic_round_lifetime.py",),
    **{f"quadratic_round_growth_{rounds}_{device}": (
        f"tests/test_filter_repair_quadratic_round_growth.py::test_complete_controller_capacity_and_warm_allocations[{rounds}]",)
        for rounds in (1, 4, 8) for device in ("cpu", "gpu")},
    **{f"quadratic_round_memory_{arm}_{dimension}_{device}": (
        f"tests/test_filter_repair_quadratic_round_memory.py::test_complete_paired_controller_costs[{arm}-{dimension}]",)
        for arm in ("before", "graph", "xla") for dimension in (3, 5) for device in ("cpu", "gpu")},
    "quadratic_rounds_smoke": ("tests/test_filter_repair_quadratic_rounds.py::test_complete_original_round_records[move-3]",),
    "quadratic_rounds_operands": ("tests/test_filter_repair_quadratic_rounds.py::test_round_controller_changed_inputs_and_enclosing_xla",),
    **{f"quadratic_rounds_{dimension}_{device}": tuple(
        f"tests/test_filter_repair_quadratic_rounds.py::test_complete_original_round_records[{case}-{dimension}]"
        for case in ("centered", "move", "nonquadratic", "round_limit", "invalid_initial", "invalid_replay",
            "invalid_large", "invalid_small", "invalid_check", "invalid_proposal", "invalid_final_replay",
            "replay_value", "initial_evidence", "bad_evidence", "nonspd", "nonfinite_scaled_score", "factor_failure"))
        for dimension in (1, 3, 5) for device in ("cpu", "gpu")},
    **{f"quadratic_probe_growth_{device}": (
        "tests/test_filter_repair_quadratic_probe_memory.py::test_paired_probe_sparse_allocation_growth",)
        for device in ("cpu", "gpu")},
    **{f"quadratic_probe_memory_{arm}_{dimension}_{device}": (
        f"tests/test_filter_repair_quadratic_probe_memory.py::test_paired_probe_costs[{arm}-{dimension}]",)
        for arm in ("before", "graph", "xla") for dimension in (3, 5) for device in ("cpu", "gpu")},
    "quadratic_probes_cpu": ("tests/test_filter_repair_quadratic_probes.py",),
    "quadratic_probes_gpu": ("tests/test_filter_repair_quadratic_probes.py",),
    "quadratic_probe_operands": ("tests/test_filter_repair_quadratic_probes.py::test_changed_probe_inputs_remain_runtime_operands",),
    **{f"quadratic_numerics_memory_{arm}_{kind}_{dimension}_{device}": (
        f"tests/test_filter_repair_quadratic_numerics_memory.py::test_numerical_dependency_costs[{arm}-{kind}-{dimension}]",)
        for arm in ("before", "graph", "xla") for kind in ("paired", "trust")
        for dimension in (3, 5) for device in ("cpu", "gpu")},
    "quadratic_numerics": ("tests/test_filter_repair_quadratic_numerics.py",),
    "quadratic_trust_paired": ("tests/test_filter_repair_quadratic_numerics.py", "-k", "not dense"),
    "quadratic_trust_paired_gpu": ("tests/test_filter_repair_quadratic_numerics.py", "-k", "not dense"),
    "quadratic_trust_paired_cpu": ("tests/test_filter_repair_quadratic_numerics.py", "-k", "not dense"),
    "quadratic_center_public_cpu": ("tests/test_batched_quadratic_center.py",),
    "quadratic_paired_public_cpu": ("tests/test_paired_score_pilot.py",),
    "quadratic_batches_cpu": ("tests/test_filter_repair_quadratic_batches.py",),
    **{f"quadratic_batch_long_growth_{device}": (
        "tests/test_filter_repair_quadratic_batch_growth.py::test_long_sparse_xla_allocation_growth",)
        for device in ("cpu", "gpu")},
    **{f"quadratic_batch_growth_{arm}_{device}": (
        f"tests/test_filter_repair_quadratic_batch_growth.py::test_sparse_fixed_shape_allocation_growth[{arm}]",)
        for arm in ("before", "xla") for device in ("cpu", "gpu")},
    "refinement_original_symmetric_cpu": ("tests/test_filter_repair_refinement_original.py", "-k", "symmetric"),
    **{f"refinement_original_{case}_cpu": (
        f"tests/test_filter_repair_refinement_original.py::test_original_complete_refinement_records[{case}]",)
        for case in ("factor_one", "factor_two", "factor_two_reuse")},
    "refinement_original_gpu": ("tests/test_filter_repair_refinement_original.py",),
    **{f"quadratic_batch_memory_{arm}_{dimension}_cpu": (
        f"tests/test_filter_repair_quadratic_batch_memory.py::test_ordered_batch_costs[{arm}-{dimension}-{capacity}]",)
        for arm in ("before", "graph", "xla") for dimension, capacity in ((3, 32), (5, 128))},
    **{f"quadratic_batch_memory_{arm}_{dimension}": (
        f"tests/test_filter_repair_quadratic_batch_memory.py::test_ordered_batch_costs[{arm}-{dimension}-{capacity}]",)
        for arm in ("before", "graph", "xla") for dimension, capacity in ((3, 32), (5, 128))},
    "quadratic_batches": ("tests/test_filter_repair_quadratic_batches.py",),
    "quadratic_center_public": ("tests/test_batched_quadratic_center.py",),
    "quadratic_paired_public": ("tests/test_paired_score_pilot.py",),
    "factor_input_sensitivity": ("tests/test_filter_repair_factor_sensitivity.py",),
    **{f"factor_trajectory_{data}": (
        f"tests/test_filter_repair_factor_trajectory.py::test_crossed_factor_objective_trajectories[{data}]",)
        for data in ("original", "current")},
    "factor_row_decode": ("tests/test_filter_repair_factor_row_decode.py",),
    "lifecycle_factor_initial": ("tests/test_filter_repair_lifecycle_factor_initial.py",),
    "lifecycle_factor_inputs": ("tests/test_filter_repair_lifecycle_factor_inputs.py",),
    **{f"lifecycle_original_runtime_{dimension}_cpu": (
        f"tests/test_filter_repair_lifecycle_original_runtime.py::test_original_lifecycle_runtime_inputs_and_resource_lifetime[{dimension}]",)
        for dimension in (3, 5)},
    "lifecycle_original_runtime_gpu": ("tests/test_filter_repair_lifecycle_original_runtime.py",),
    **{f"lifecycle_original_symmetric_{device}": (
        "tests/test_filter_repair_lifecycle_original.py", "-k", "not factor")
        for device in ("cpu", "gpu")},
    **{f"lifecycle_original_{case}_{device}": (
        f"tests/test_filter_repair_lifecycle_original.py::test_original_full_lifecycle_records_and_target_order[{case}]",)
        for case in ("factor_one", "factor_two", "factor_two_reuse") for device in ("cpu", "gpu")},
    "terminal_original_cpu": ("tests/test_filter_repair_terminal_original.py",),
    "terminal_original_gpu": ("tests/test_filter_repair_terminal_original.py",),
    "sequential_eigen_consumers_cpu": ("tests/test_filter_repair_eigen_consumers.py",),
    "sequential_eigen_consumers_gpu": ("tests/test_filter_repair_eigen_consumers.py",),
    **{f"lifecycle_profile_{dimension}": (
        f"tests/test_filter_repair_lifecycle_profile.py::test_lifecycle_compilation_memory_stages[{dimension}-{search_count}]",)
        for dimension, search_count in ((3, 4), (5, 32))},
    "lifecycle_terminal_modes": (
        "tests/test_filter_repair_lifecycle_modes.py::test_terminal_mode_failure_source_and_eigensystem",),
    "lifecycle_factor_modes": (
        "tests/test_filter_repair_lifecycle_modes.py::test_factor_condition_failure_on_identical_optimizer_state",),
    **{f"lifecycle_memory_{arm}_{dimension}": (
        f"tests/test_filter_repair_lifecycle_memory.py::test_complete_lifecycle_memory[{arm}-{dimension}-{search_count}]",)
        for arm in ("before", "graph", "xla") for dimension, search_count in ((3, 4), (5, 32))},
    "lifecycle_specialization": ("tests/test_filter_repair_lifecycle_specialization.py", "-s"),
    **{f"lifecycle_runtime_{dimension}_cpu": (
        f"tests/test_filter_repair_lifecycle_runtime.py::test_actual_lifecycle_runtime_inputs_and_resource_lifetime[{dimension}]",)
        for dimension in (3, 5)},
    "lifecycle_runtime_gpu": ("tests/test_filter_repair_lifecycle_runtime.py",),
    "lifecycle_actual_symmetric_cpu": ("tests/test_filter_repair_lifecycle_actual.py", "-k", "not factor"),
    **{f"lifecycle_actual_{case}_{device}": (
        f"tests/test_filter_repair_lifecycle_actual.py::test_actual_lifecycle_complete_records_and_calls[{case}]",)
        for case in ("factor_one", "factor_two", "factor_two_reuse") for device in ("cpu", "gpu")},
    "lifecycle_actual_symmetric_gpu": ("tests/test_filter_repair_lifecycle_actual.py", "-k", "not factor"),
    "refinement_symmetric_cpu": ("tests/test_filter_repair_sequential_refinement.py", "-k", "symmetric"),
    **{f"refinement_{case}_cpu": (
        f"tests/test_filter_repair_sequential_refinement.py::test_complete_refinement_records[{case}]",)
        for case in ("factor_one", "factor_two", "factor_two_reuse")},
    "refinement_gpu": ("tests/test_filter_repair_sequential_refinement.py",),
    "sequential_terminal_cpu": ("tests/test_filter_repair_sequential_terminal.py",),
    "sequential_terminal_gpu": ("tests/test_filter_repair_sequential_terminal.py",),
    "sequential_lifecycle_cpu": ("tests/test_filter_repair_sequential_lifecycle.py",),
    "sequential_lifecycle_gpu": ("tests/test_filter_repair_sequential_lifecycle.py",),
    **{f"attempts_capacity_{arm}_{capacity}": (
        f"tests/test_filter_repair_attempts_memory.py::test_attempts_enclosure_capacity[{arm}-{capacity}]",)
        for arm in ("before", "xla") for capacity in (4, 32)},
    "attempts_dense_trust_modes": ("tests/test_filter_repair_attempts_modes.py",),
    **{f"attempts_memory_{arm}_{dimension}": (
        f"tests/test_filter_repair_attempts_memory.py::test_attempts_memory[{arm}-{dimension}-{capacity}]",)
        for arm in ("before", "graph", "xla") for dimension, capacity in ((3, 4), (5, 32))},
    "attempts_public_actual_cpu": ("tests/test_filter_repair_attempts_public.py",),
    "attempts_public_actual_gpu": ("tests/test_filter_repair_attempts_public.py",),
    "sequential_attempts_edges": ("tests/test_filter_repair_sequential_attempts.py", "-k", "single_factor or better_first"),
    "sequential_attempts_inputs": ("tests/test_filter_repair_sequential_attempts.py::test_single_factor_skips_second_callback_and_retains_graph_inputs",),
    "sequential_attempts_cpu": ("tests/test_filter_repair_sequential_attempts.py",),
    "sequential_attempts_gpu": ("tests/test_filter_repair_sequential_attempts.py",),
    "factor_decisions_cpu": ("tests/test_filter_repair_factor_decisions.py",),
    "factor_decisions_gpu": ("tests/test_filter_repair_factor_decisions.py",),
    **{f"proposal_memory_{arm}_{dimension}": (f"tests/test_filter_repair_proposal_memory.py::test_proposal_memory[{arm}-{dimension}]",)
        for arm in ("before", "graph", "xla") for dimension in (3, 5)},
    "sequential_proposal_public_cpu": ("tests/test_filter_repair_sequential_proposal.py", "-k", "public_history"),
    "sequential_proposal_public_gpu": ("tests/test_filter_repair_sequential_proposal.py", "-k", "public_history"),
    "sequential_proposal_cpu": ("tests/test_filter_repair_sequential_proposal.py",),
    "sequential_proposal_gpu": ("tests/test_filter_repair_sequential_proposal.py",),
    "structured_inherited_modes": (
        "tests/test_filter_repair_structured_memory.py::test_inherited_structured_fitter_modes",),
    **{f"structured_eligibility_{arm}": (
        f"tests/test_filter_repair_structured_memory.py::test_structured_changing_eligibility_memory[{arm}]",)
        for arm in ("before", "after")},
    "structured_graph_xla_localization": (
        "tests/test_filter_repair_structured_memory.py::test_structured_graph_xla_localization",),
    "structured_record_boundary": (
        "tests/test_factor_correlation_geometry.py::test_reuse_filter_includes_radius_boundary_and_rejects_zero_outside_nonfinite",
        "tests/test_filter_repair_structured_fit.py", "-k", "three or radius_boundary"),
    **{f"structured_memory_{arm}_{dimension}": (
        f"tests/test_filter_repair_structured_memory.py::test_structured_preparation_fit_memory[{arm}-{dimension}-{capacity}]",)
        for arm in ("before", "after", "graph", "xla") for dimension, capacity in ((3, 4), (5, 32))},
    "structured_fit_enclosure": ("tests/test_filter_repair_structured_fit.py::test_enclosing_preparation_fit_has_runtime_inputs_and_frozen_geometry",
        "tests/test_filter_repair_structured_preparation.py::test_preparation_pullbacks_preserve_original_tensors"),
    "structured_preparation_stages": ("tests/test_filter_repair_structured_fit.py::test_preparation_stage_localization",),
    "structured_fit_arithmetic": ("tests/test_filter_repair_structured_fit.py::test_structured_preparation_arithmetic_localization",),
    "structured_fit_cpu": ("tests/test_filter_repair_structured_fit.py", "-k", "not localization"),
    "structured_fit_gpu": ("tests/test_filter_repair_structured_fit.py", "-k", "not localization"),
    "structured_preparation_inputs": ("tests/test_filter_repair_structured_preparation.py::test_runtime_eligibility_keeps_one_trace_all_operands_and_same_hlo",),
    "structured_preparation_cpu": ("tests/test_filter_repair_structured_preparation.py",),
    "structured_preparation_gpu": ("tests/test_filter_repair_structured_preparation.py",),
    "factor_guard_cpu_fixed": ("tests/test_filter_repair_fixed_fitting.py",),
    "factor_guard_cpu_padded": ("tests/test_filter_repair_padded_factor.py", "-k", "not localization"),
    "factor_guard_cpu_domain": ("tests/test_filter_repair_factor_domain.py", "-k", "runtime"),
    "factor_guard_gpu_lifetime": ("tests/test_filter_repair_factor_resource_lifetime.py", "-s"),
    **{f"factor_guard_memory_{arm}_{mode}_{dimension}": (
        f"tests/test_filter_repair_factor_guard_memory.py::test_factor_guard_memory[{arm}-{mode == 'xla'}-{dimension}]", "-s")
        for arm in ("checkpoint", "candidate") for mode in ("graph", "xla") for dimension in (3, 5)},
    "factor_guard_mapping_release": ("tests/test_filter_repair_fixed_fitting.py",
        "tests/test_filter_repair_padded_factor.py",
        "tests/test_filter_repair_factor_domain.py", "-k", "not localization and not comparison and not guard_trial and not evaluations_trial",
        "-s", "-p", "no:faulthandler", "-p", "tests.filter_repair_memory_plugin"),
    "factor_guard_mapping_probe": ("tests/test_filter_repair_fixed_fitting.py",
        "tests/test_filter_repair_padded_factor.py",
        "tests/test_filter_repair_factor_domain.py", "-k", "not localization and not comparison and not guard_trial and not evaluations_trial",
        "-s", "-p", "no:faulthandler"),
    "factor_guard_resource_lifetime": ("tests/test_filter_repair_factor_resource_lifetime.py", "-s"),
    "factor_guard_crash_replay": ("tests/test_filter_repair_padded_factor.py",
        "tests/test_filter_repair_factor_domain.py::test_runtime_fixed_fitter_blocks_any_invalid_covariance_evaluation",
        "-k", "not localization", "-s"),
    "factor_guard_fixed_replay": ("tests/test_filter_repair_fixed_fitting.py",
        "tests/test_filter_repair_factor_domain.py", "-k", "not comparison and not trial", "-s"),
    "factor_guard_native_stack": ("tests/test_filter_repair_fixed_fitting.py",
        "tests/test_filter_repair_padded_factor.py",
        "tests/test_filter_repair_factor_domain.py", "-k", "not localization and not comparison and not guard_trial and not evaluations_trial",
        "-s", "-p", "no:faulthandler"),
    "factor_guard_qualification": ("tests/test_filter_repair_fixed_fitting.py",
        "tests/test_filter_repair_padded_factor.py",
        "tests/test_filter_repair_factor_domain.py::test_runtime_covariance_guard_rejects_invalid_xla_inputs",
        "tests/test_filter_repair_factor_domain.py::test_runtime_factor_rejects_invalid_trial_and_resets_checked_state",
        "tests/test_filter_repair_factor_domain.py::test_runtime_fixed_fitter_blocks_any_invalid_covariance_evaluation",
        "-k", "not localization"),
    "factor_domain_runtime": ("tests/test_filter_repair_factor_domain.py", "-k", "runtime", "-s"),
    "factor_domain_localization": ("tests/test_filter_repair_factor_domain.py", "-k", "comparison", "-s"),
    "factor_domain_guard_trial": ("tests/test_filter_repair_factor_domain.py::test_factor_domain_guard_trial", "-s"),
    "factor_domain_events": ("tests/test_filter_repair_factor_domain.py::test_factor_domain_all_evaluations_trial", "-s"),
    "active_cod_compact_step": ("tests/test_filter_repair_active_cod.py::test_compact_cod_step_complete_fit", "-k", "step and not cpqr", "-s"),
    "active_cod_compact_cpqr": ("tests/test_filter_repair_active_cod.py::test_compact_cod_step_complete_fit", "-k", "cpqr", "-s"),
    "active_cod_steps": ("tests/test_filter_repair_active_cod.py::test_active_cod_step_localization", "-s"),
    "active_cod_enclosure": ("tests/test_filter_repair_active_cod.py::test_active_cod_enclosure_localization", "-s"),
    "active_cod_runtime": ("tests/test_filter_repair_active_cod_runtime.py", "tests/test_filter_repair_qr.py"),
    "factor_capacity_shared_cod": ("tests/test_filter_repair_active_cod.py::test_shared_cod_row_arithmetic_localization", "-s"),
    **{f"factor_shared_memory_{arm}": (
        f"tests/test_filter_repair_active_cod.py::test_shared_cod_fresh_process_memory[{arm}]", "-s")
        for arm in ("compact", "shared")},
    "factor_enclosure_localization": ("tests/test_filter_repair_factor_enclosure.py", "-s"),
    "padded_jacobian_localization": ("tests/test_filter_repair_padded_jacobian.py::test_partial_occupancy_jacobian_factorization_localization", "-s"),
    "padded_dynamic_qr_localization": ("tests/test_filter_repair_padded_jacobian.py::test_bounded_dynamic_qr_localization", "-s"),
    "padded_dynamic_qr_output": ("tests/test_filter_repair_padded_jacobian.py::test_bounded_dynamic_qr_localization[True]", "-s"),
    "padded_shape_dispatch": ("tests/test_filter_repair_padded_jacobian.py::test_active_shape_qr_dispatch_localization", "-s"),
    "factor_capacity_initial": ("tests/test_filter_repair_factor_capacity.py::test_capacity_initial_arithmetic_localization", "-s"),
    "factor_capacity_initial_fixed": ("tests/test_filter_repair_factor_capacity.py::test_capacity_initial_arithmetic_localization[True]", "-s"),
    "factor_capacity_cod": ("tests/test_filter_repair_factor_capacity.py::test_compact_initialization_dispatch_localization", "-s"),
    **{f"factor_capacity_{capacity}": (
        f"tests/test_filter_repair_factor_capacity.py::test_factor_capacity_memory_and_records[{capacity}]", "-s")
        for capacity in (0, 4, 32)},
    **{f"factor_capacity_graph_{capacity}": (
        f"tests/test_filter_repair_factor_capacity.py::test_factor_capacity_graph_memory_and_records[{capacity}]", "-s")
        for capacity in (0, 32)},
    **{f"factor_capacity_short_{mode}_{capacity}": (
        f"tests/test_filter_repair_factor_capacity.py::test_factor_capacity_short_memory_and_records[{mode == 'xla'}-{capacity}]", "-s")
        for mode in ("graph", "xla") for capacity in (0, 32)},
    "public_pullbacks": ("tests/test_filter_repair_public_pullbacks.py",),
    "tensor_program": ("tests/test_compiled_tensor_program_tf.py",),
    "source_sequential_captures": ("tests/test_filter_repair_source_sequential.py::test_repeated_transport_keeps_all_captured_core_frame_and_callback_derivatives",),
    "source_sequential": ("tests/test_filter_repair_source_sequential.py",
        "tests/highdim/test_p57_m6_sequential_fixed_hmc_source_loop.py"),
    "source_runtime": ("tests/test_filter_repair_source_runtime.py",
        "tests/highdim/test_p57_m6_sequential_fixed_hmc_source_loop.py"),
    "density_enclosing": ("tests/test_filter_repair_squared_density.py::test_public_density_and_previous_marginal_enclose_full_xla_and_input_gradient",
        "tests/test_filter_repair_squared_density.py::test_heterogeneous_marginal_preserves_query_and_core_pullbacks"),
    "ttsirt_enclosing": ("tests/test_filter_repair_ttsirt.py::test_public_transport_complete_enclosing_graph_and_xla_with_invalid_status",
        "tests/test_filter_repair_ttsirt.py::test_transport_vetoes_and_bisection_graph_are_preserved"),
    "ttsirt_coordinates": ("tests/test_filter_repair_ttsirt_coordinates.py",),
    "source_numerics": ("tests/test_filter_repair_source_numerics.py",
        "tests/highdim/test_p49_source_route_sample_proposal.py",
        "tests/highdim/test_p49_source_route_recenter_normalizer.py",
        "tests/highdim/test_p55_source_route_target_transport.py"),
    "stochastic_training": ("tests/test_filter_repair_stochastic_training.py",
        "tests/highdim/test_p75_stochastic_density_training.py",
        "tests/highdim/test_p76_corrected_heldout_metric.py"),
    "lane_b_training": ("tests/test_filter_repair_lane_b_training.py",
        "tests/highdim/test_zhao_cui_austria_sir_lane_b_tf.py::test_exact_core_rescale_recovers_requested_normalizer",
        "tests/highdim/test_zhao_cui_austria_sir_lane_b_tf.py::test_compiled_training_kernel_matches_eager_training_base_update",
        "tests/highdim/test_zhao_cui_austria_sir_lane_b_tf.py::test_artifact_reload_identity_and_tensor_tamper_rejection"),
    "core_tangents": ("tests/test_filter_repair_core_tangents.py",),
    "centered_random": ("tests/test_filter_repair_centered_random.py",),
    "prefix_scores": ("tests/test_filter_repair_prefix_scores.py",),
    "centered_initializers": ("tests/test_filter_repair_centered_initializers.py",),
    "centered_callbacks": ("tests/test_filter_repair_centered_tt.py::test_centered_training_callback_has_stable_signature_and_preserves_update",),
    "centered_updates": ("tests/test_filter_repair_centered_updates.py",),
    "centered_batch_objectives": ("tests/test_filter_repair_centered_updates.py::test_public_batch_metrics_and_absolute_loss_preserve_external_pullbacks",),
    "centered_preparation": ("tests/test_filter_repair_centered_preparation.py",
        "tests/highdim/test_zhao_cui_austria_sir_parameter_density_training_tf.py::test_rotating_prefix_schedule_covers_each_epoch_exactly"),
    "centered_gpu": ("tests/test_filter_repair_centered_tt.py::test_complete_centered_child_scores_preserve_pinned_consumer",
        "tests/test_filter_repair_centered_solver.py"),
    "centered_solver": ("tests/test_filter_repair_centered_solver.py",
        "tests/highdim/test_zhao_cui_austria_sir_parameter_density_training_tf.py::test_matrix_free_conjugate_gradient_solves_quadratic_callback"),
    "centered_consumer": ("tests/test_filter_repair_centered_tt.py::test_complete_centered_child_scores_preserve_pinned_consumer",),
    "centered_tt": ("tests/test_filter_repair_centered_tt.py",),
    "sealed_sir": ("tests/test_filter_repair_sealed_sir.py",),
    "austria_consumer": ("tests/highdim/test_zhao_cui_austria_sir_parameter_density_training_tf.py::test_batch_native_target_and_score_match_scalar_theta_authority",),
    "austria_preparation": ("tests/test_filter_repair_austria_preparation.py",
        "tests/highdim/test_zhao_cui_austria_sir_parameter_density_training_tf.py::test_batch_native_target_and_score_match_scalar_theta_authority"),
    "gamma_random_gpu": ("tests/test_filter_repair_gamma_random.py",
        "tests/test_filter_repair_hermite_proposal.py::test_complete_proposal_random_inputs_preserve_existing_draws_and_hlo",
        "tests/test_filter_repair_student_proposal.py"),
    "student_proposal": ("tests/test_filter_repair_student_proposal.py",
        "tests/highdim/test_c2_transformed_observation_student_proposal_tf.py"),
    "gamma_random": ("tests/test_filter_repair_gamma_random.py",),
    "source_preparation_numerics": ("tests/test_filter_repair_source_preparation.py",),
    "source_guard_preparation": ("tests/test_filter_repair_source_guard_preparation.py",
        "tests/test_filter_repair_source_gates.py",
        "tests/highdim/test_p72_support_certified_lower_gate.py"),
    "source_guard_localization": (
        "tests/test_filter_repair_source_guard_preparation.py::test_guard_lines_preserve_selection_order_duplicate_keys_and_frozen_design[duplicate]",),
    "source_preparation_localization": (
        "tests/highdim/test_p59_author_sir_step_spec_assembly.py::test_p59_9b_assembles_two_author_sir_36d_step_specs",
        "--capture=no", "-p", "no:faulthandler"),
    "source_preparation": ("tests/test_filter_repair_source_preparation.py",
        "tests/highdim/test_p49_source_route_recenter_normalizer.py",
        "tests/highdim/test_p55_source_route_target_transport.py",
        "tests/highdim/test_p57_m6_sequential_fixed_hmc_source_loop.py",
        "tests/highdim/test_p59_author_sir_step_spec_assembly.py"),
    "predator_tp": ("tests/test_filter_repair_predator_tp.py", "tests/highdim/test_ledh_contract_e_tp_predator_prey.py",
        "tests/test_filter_repair_predator_tp_localization.py"),
    "predator_tp_residual": ("tests/test_filter_repair_predator_tp.py::test_complete_predator_tp_same_mode_value_gradient_and_history[True-4-2]",),
    "predator_tp_difference": ("tests/test_filter_repair_predator_tp_localization.py::test_gpu_finite_difference_step_localization",),
    "predator_tp_localization": ("tests/test_filter_repair_predator_tp_localization.py",),
    "predator_tp_continuation": ("tests/test_filter_repair_predator_tp_localization.py::test_continuation_rounding_breakdown",),
    "predator_tp_continuation_windows": ("tests/test_filter_repair_predator_tp_localization.py::test_dynamic_continuation_windows_keep_values_and_total_gradients",),
    "predator_tp_fixed_features": ("tests/test_filter_repair_predator_tp_localization.py::test_first_tp_projection_breakdown[features-False]",
        "tests/test_filter_repair_predator_tp_localization.py::test_first_tp_projection_breakdown[features-True]"),
    "predictive": ("tests/test_filter_repair_predictive.py", "tests/test_ssl_lstm_predictive_tf.py", "tests/test_ssl_lstm_complexity_predictive_tf.py"),
    "forecast_shards": ("tests/test_filter_repair_forecast_pool.py",),
    "forecast_pool": ("tests/test_cpu_forecast_pool.py",),
    "cpu_cloud": ("tests/test_cpu_xla_cloud.py",),
    "complexity_target": ("tests/test_ssl_lstm_complexity_target_tf.py",),
    "hermite_proposal": ("tests/test_filter_repair_hermite_proposal.py", "tests/highdim/test_c2_gaussian_hermite_proposal_tf.py"),
    "teacher_identity": ("tests/test_filter_repair_dispatch_identity.py",
        "tests/highdim/test_ledh_contract_e_schema_v2_factory.py",
        "tests/highdim/test_zhao_cui_moment_teacher_integration.py::test_factory_identity_binds_teacher_particle_controls_and_source",
        "tests/highdim/test_zhao_cui_moment_teacher_nonlinear.py::test_repository_factory_binds_nonlinear_model_and_prepared_program",
        "tests/highdim/test_zhao_cui_moment_teacher_actual_sv.py::test_factory_binds_actual_sv_and_rejects_cross_model_substitution"),
    "teacher_consumers": ("tests/highdim/test_zhao_cui_moment_teacher_integration.py",
        "tests/highdim/test_zhao_cui_moment_teacher_nonlinear.py",
        "tests/highdim/test_zhao_cui_moment_teacher_actual_sv.py"),
    "moment_teacher": ("tests/test_filter_repair_moment_teacher.py",),
    "ukf_initializer": ("tests/test_filter_repair_ukf_initializer.py", "tests/highdim/test_p76_ukf_initializer.py"),
    "tp_recursions": ("tests/test_filter_repair_tp_recursions.py",),
    "information_recursions": ("tests/test_filter_repair_information_recursions.py",),
    "remaining_routes": ("tests/test_filter_repair_remaining_routes.py",),
    "tt_preparation": ("tests/test_filter_repair_tt_preparation.py",),
    "filtering_wrappers": ("tests/test_filter_repair_filtering.py", "tests/highdim/test_filtering_kalman_exact.py", "tests/highdim/test_zhao_cui_hmc_default_route_policy.py"),
    "sv_sgqf": ("tests/test_filter_repair_sv_sgqf.py",),
    "start_bank": ("tests/test_filter_repair_start_bank.py", *tuple(
        "tests/test_hmc_warmup.py::" + name for name in (
            "test_start_bank_selector_is_byte_identical_to_frozen_oracle",
            "test_start_bank_selector_preserves_transform_scaling_and_greedy_order",
            "test_start_bank_selector_keeps_less_than_or_equal_tolerance_boundary",
            "test_start_bank_endpoint_exclusion_precedes_prior_eligible_exclusion",
            "test_start_bank_failures_match_oracle_and_carry_no_public_diagnostic",
            "test_start_bank_combined_interpretations_are_shadow_decision_inert",
            "test_start_bank_shadow_failures_are_fixed_bounded_codes",
            "test_start_bank_diagnostic_schema_is_finite_fixed_and_private_safe",
            "test_start_bank_failure_carrier_requires_concrete_validated_type",
        ))),
    "signatures": ("tests/test_filter_repair_signatures.py", "tests/test_kalman_covariance_derivatives_tf.py", "tests/test_linear_correlated_kalman_tf.py", "tests/highdim/test_zhao_cui_moment_teacher_xla.py"),
    "contract_e_reset": ("tests/test_filter_repair_contract_e_reset.py",),
    "student_t_selection": ("tests/highdim/test_c2_student_t_floor.py::test_student_t_margin_vs_dense_grid_max", "tests/highdim/test_c2_student_t_floor.py::test_student_t_nu_criterion_well_posed", "tests/test_filter_repair_student_t_selection.py"),
    "moment_hints": ("tests/test_filter_repair_moment_hints.py",),
    "qr": ("tests/test_filter_repair_qr.py",),
    "tt_scalar": ("tests/test_filter_repair_scalar_tt.py", "tests/highdim/test_zhao_cui_fixed_adjacent_tt_tf.py"),
    "tt_scalar_retained": ("tests/test_filter_repair_scalar_retained.py", "tests/highdim/test_p30_sv_short_sequential_tt_value_path.py", "tests/highdim/test_fixed_branch_derivatives.py::test_scalar_fixed_design_tt_score_path_matches_same_branch_fd_for_exact_transformed_sv", "tests/highdim/test_fixed_branch_derivatives.py::test_scalar_fixed_design_tt_score_path_rejects_missing_manual_model_score_method"),
    "tt_panel_retained": ("tests/test_filter_repair_panel_retained.py",),
    "fixed_fit": ("tests/highdim/test_fixed_branch_fit.py",),
    "fixed_fit_cache": ("tests/highdim/test_fixed_branch_fit.py::test_public_native_fit_matches_original_als_and_reuses_signature",),
    "fixed_fit_pullbacks": ("tests/test_filter_repair_fixed_fit_pullbacks.py",),
    "basis_graph": ("tests/test_filter_repair_basis_graph.py",),
    "model_simulation": ("tests/test_filter_repair_model_simulation.py",),
    "squared_density": ("tests/highdim/test_squared_tt_density.py", "tests/highdim/test_failure_exits.py", "tests/test_filter_repair_squared_density.py"),
    "ttsirt": ("tests/highdim/test_zhao_cui_frozen_ttsirt_apf_compiler.py", "tests/highdim/test_p57_m2_fixed_ttsirt_transport_contract.py", "tests/test_filter_repair_ttsirt.py"),
    "tt_algebra": ("tests/highdim/test_tt_algebra.py", "tests/highdim/test_fixed_branch_derivatives.py"),
    "tt_scalar_smoke": ("tests/test_filter_repair_scalar_tt.py::test_complete_value_score_and_fit_veto_parity[False-True]",),
    "factor_geometry": ("tests/test_factor_correlation_geometry.py",),
    "fixed_geometry": ("tests/test_fixed_center_curvature.py", "tests/test_filter_repair_host_io.py", "tests/test_posterior_curvature_refinement.py"),
    "block_geometry": ("tests/test_block_score_geometry.py",),
    "sequential_geometry": ("tests/test_sequential_map_covariance.py",),
    "sequential_score_fit": ("tests/test_filter_repair_sequential_score_fit.py",),
    "sequential_preparation": ("tests/test_filter_repair_sequential_preparation.py",),
    "block_center": ("tests/test_block_coordinate_center.py", "tests/test_filter_repair_block_center.py"),
    "quadratic_geometry": ("tests/test_quadratic_geometry.py", "tests/test_filter_repair_geometry_parity.py"),
    "quadratic_initializer": ("tests/test_quadratic_map_covariance.py", "tests/test_filter_repair_quadratic_initializer.py"),
    "quadratic_initializer_localization": ("tests/test_filter_repair_quadratic_initializer.py",),
    "quadratic_locator": ("tests/test_filter_repair_quadratic_initializer.py::test_real_locator_retains_selected_quadratic_center_with_xla",),
    "posterior_initializer": ("tests/test_posterior_local_initializer.py",),
    "primitives": ("tests/highdim/test_retained_moments.py", "tests/test_filter_repair_primitives.py"),
    "kalman": ("tests/test_compiled_filter_parity_tf.py", "tests/test_compiled_kalman_ukf_runtime.py"),
    "sgqf": ("tests/test_fixed_sgqf_tf.py", "tests/test_fixed_sgqf_scores_tf.py", "tests/test_fixed_sgqf_integration_tf.py", "tests/test_predator_prey_sgqf_neutra_target.py"),
    "dense_ledh": ("tests/test_experimental_batched_ledh_pfpf_ot_tf.py",),
    "particle": ("tests/test_filter_repair_particles.py",),
    "sinkhorn": ("tests/test_filter_repair_sinkhorn.py",),
    "annealed": ("tests/test_filter_repair_annealed.py",),
    "alg1_ukf": ("tests/test_ledh_pfpf_alg1_ukf_tf.py", "tests/test_filter_repair_slogdet.py"),
    "native_execution": ("tests/test_filter_repair_native_execution.py",),
    "ot_endpoints": ("tests/test_filter_repair_ot_endpoints.py",),
    "random": ("tests/test_filter_repair_random.py::test_uniform_words_preserve_non_xla_double_stream",),
    "random_gpu": ("tests/test_filter_repair_random.py::test_gpu_categorical_engine_diagnostic",),
    "geometry_random": ("tests/test_filter_repair_geometry_random.py",),
    "contract_e": ("tests/highdim/test_ledh_contract_e_canonical_lgssm_phase5.py",),
    "contract_e_strict": tuple("tests/highdim/test_ledh_contract_e_canonical_lgssm_phase5.py::" + name for name in (
        "test_one_batch_one_step_active_reset_and_all_parameter_sensitivity",
        "test_exact_chunk_mixed_reset_full_graph_matches_baseline_ad_rounding",
        "test_float32_shared_core_manual_jvp_matches_forward_autodiff",
    )),
    "cpu_pool": ("tests/test_ssl_lstm_process_parallel.py",),
    "cpu_target": ("tests/test_filter_repair_cpu_target.py",),
    "identity": ("tests/test_hmc_identity.py", "tests/test_filter_repair_host_io.py"),
    "target_failure": tuple("tests/test_common_inference_runtime_contracts.py::" + name for name in (
        "test_target_failure_policy_valid_gaussian_does_not_use_fallback",
        "test_target_failure_policy_declared_support_error_gets_finite_fallback",
        "test_target_failure_policy_nonfinite_value_gradient_is_ambiguous",
        "test_target_failure_policy_does_not_mask_programmer_or_shape_errors",
        "test_target_failure_policy_labels_are_bounded_and_backend_breakdown_is_separate",
        "test_target_failure_policy_classifies_sampler_energy_error_after_valid_target",
    )) + ("tests/test_linear_kalman_svd_tf.py::test_target_failure_policy_does_not_activate_on_valid_lgssm_value",),
    "exact_incumbent": ("tests/test_filter_repair_exact_incumbent.py", "tests/test_exact_incumbent.py"),
    "mass_matrix": ("tests/test_filter_repair_mass_matrix.py", "tests/test_hmc_mass_matrix.py",
        "tests/test_structured_empirical_mass.py"),
    "mass_matrix_localization": ("tests/test_filter_repair_mass_matrix.py::test_eigensolver_residual_localization", "-s"),
    "block_score_geometry": ("tests/test_filter_repair_block_score_geometry.py", "tests/test_block_score_geometry.py"),
    "block_score_localization": ("tests/test_filter_repair_block_score_geometry.py::test_pair_eigensystem_localization", "-s"),
    "fixed_stability": ("tests/test_filter_repair_fixed_stability.py",),
    "fixed_selection": ("tests/test_filter_repair_fixed_selection.py",),
    "fixed_fitting": ("tests/test_filter_repair_fixed_fitting.py",),
    "fixed_fitting_consumers": ("tests/test_fixed_center_curvature.py", "tests/test_factor_correlation_geometry.py",
        "tests/test_posterior_local_initializer.py", "tests/test_filter_repair_fixed_selection.py",
        "tests/test_filter_repair_fixed_stability.py", "tests/test_filter_repair_host_io.py",
        "tests/test_posterior_curvature_refinement.py"),
    "fixed_fitting_original": ("tests/test_filter_repair_fixed_fitting.py::test_original_factor_and_dense_lifecycle_fields",),
    "fixed_fitting_two_factor": (
        "tests/test_filter_repair_fixed_fitting.py::test_complete_fit_records_preserve_pre_enclosure[5-2-holdout]",
        "tests/test_filter_repair_fixed_fitting.py::test_complete_fit_records_preserve_pre_enclosure[5-2-explicit_second]"),
    "fixed_fitting_frozen": ("tests/test_filter_repair_fixed_fitting.py::test_public_fitted_geometry_preserves_frozen_derivative_boundary",),
    "fixed_fitting_fields": ("tests/test_filter_repair_fixed_fitting_localization.py::test_original_fit_field_breakdown", "-s"),
    "fixed_fitting_initializer": ("tests/test_filter_repair_initializer_rounding.py", "-s"),
    "fixed_fitting_initializer_stages": ("tests/test_filter_repair_initializer_stages.py", "-s"),
    "padded_factor": ("tests/test_filter_repair_padded_factor.py", "-k", "not localization", "-s"),
    "padded_factor_full": ("tests/test_filter_repair_padded_factor.py::test_padded_fit_keeps_complete_public_numerics[4-4-5-2]", "-s"),
    "padded_factor_localization": ("tests/test_filter_repair_padded_factor.py::test_full_occupancy_initializer_and_same_state_loss_localization", "-s"),
    "padded_factor_arithmetic": ("tests/test_filter_repair_padded_factor.py::test_full_occupancy_loss_arithmetic_localization", "-s"),
    "factor_specialization": ("tests/test_filter_repair_padded_factor.py::test_factor_compiler_input_specialization_localization", "-s"),
    "factor_runtime_inputs": ("tests/test_filter_repair_padded_factor.py::test_complete_fitter_retains_runtime_inputs",),
    "factor_runtime_consumers": ("tests/test_filter_repair_qr.py", "tests/test_factor_correlation_geometry.py"),
    **{f"factor_memory_{arm}_{mode}": (
        f"tests/test_filter_repair_factor_compilation_memory.py::test_changing_training_cloud_compilation_memory[{mode == 'xla'}-{arm}]", "-s")
        for arm in ("checkpoint", "candidate") for mode in ("graph", "xla")},
    **{f"locator_memory_{count}_{capacity}": (
        f"tests/test_filter_repair_locator_memory.py::test_buffer_capacity_memory_and_complete_observations[{count}-{capacity}]", "-s")
        for count, capacity in ((2, 128), (2, 4096), (4, 128), (4, 4096))},
    "sequential_selection": ("tests/test_filter_repair_sequential_selection.py",),
    "sequential_locator": ("tests/test_filter_repair_sequential_locator.py",),
    "batched_locator": ("tests/test_filter_repair_batched_locator.py",),
    "locator_frozen": ("tests/test_filter_repair_locator_frozen.py",),
    "locator_completion": ("tests/test_filter_repair_sequential_locator.py",
        "tests/test_filter_repair_batched_locator.py", "tests/test_filter_repair_locator_frozen.py"),
    "fixed_fitting_localization": ("tests/test_filter_repair_fixed_fitting_localization.py", "-s"),
    "joint_center": ("tests/test_exact_incumbent.py", "tests/test_joint_center.py"),
    "apf": ("tests/highdim/test_zhao_cui_frozen_proposal_apf_tf.py", "tests/highdim/test_c2_sv_frozen_proposal_apf_tf.py"),
    "preparation": ("tests/test_backend_readiness.py", "tests/highdim/test_bases.py", "tests/highdim/test_c2_hermite_basis.py", "tests/highdim/test_p86_lagrangep_mass_integral.py", "tests/highdim/test_retained_moments.py", "tests/test_filter_repair_primitives.py", "tests/test_filter_repair_consumers.py"),
    "tt": ("tests/highdim/test_squared_tt_density.py", "tests/highdim/test_zhao_cui_actual_sv_batched_tt_tf.py", "tests/highdim/test_zhao_cui_frozen_proposal_apf_tf.py"),
    "tt_actual": ("tests/highdim/test_zhao_cui_actual_sv_batched_tt_tf.py",),
    "tt_contractions": ("tests/highdim/test_p1a_retained_quadratic_form.py", "tests/highdim/test_squared_tt_density.py", "tests/test_filter_repair_tt.py"),
    "tt_value": ("tests/highdim/test_p3_xla_value_parity.py",),
    "tt_maps": ("tests/test_filter_repair_tt_maps.py",),
    "tt_gaussian_consumers": ("tests/highdim/test_c2_gaussian_frozen_target_diagnostics.py", "tests/highdim/test_c2_student_t_floor.py::test_student_t_floor_lane_parity"),
    "tt_adjoint_nodes": ("tests/highdim/test_p2_adjoint_nodes.py", "tests/highdim/test_p2_adjoint_vs_forward_jvp.py"),
    "tt_adjoint": ("tests/highdim/test_p2_adjoint_engine_fd.py",),
    "consumers": ("tests/test_filter_repair_consumers.py",),
    "policy": ("tests/test_filter_repair_campaign.py", "tests/test_filter_repair_policy.py",
        "tests/test_filter_repair_gpu_selection.py", "tests/test_filter_repair_cost_provenance.py"),
}
TEST_GROUPS.update({f"remaining_svd_{part}_{device}": TEST_GROUPS[original]
    for part, original in (("block", "block_score_geometry"),
        ("public_first", "geometry_public_first_cpu"),
        ("public_second", "geometry_public_second_cpu"),
        ("public_capacity", "geometry_public_capacity_cpu"))
    for device in ("cpu", "gpu")})
TEST_GROUPS.update({name.removesuffix("_cpu") + "_gpu": cases
    for name, cases in tuple(TEST_GROUPS.items()) if name.startswith("remote_integration_")})

# Exact intermediate comparisons remain callable historical diagnostics. Each
# has the same-scope original-source authority as a mandatory replacement.
# cfbc32d2's demonstrated eigensystem error is preserved, not a precision gate.
ORIGINAL_AUTHORITY_REPLACEMENTS = {
    **{f"lifecycle_actual_{case}_{device}": f"lifecycle_original_{case}_{device}"
       for case in ("symmetric", "factor_one", "factor_two", "factor_two_reuse")
       for device in ("cpu", "gpu")},
    **{f"lifecycle_runtime_{dimension}_cpu": f"lifecycle_original_runtime_{dimension}_cpu"
       for dimension in (3, 5)},
    "lifecycle_runtime_gpu": "lifecycle_original_runtime_gpu",
    **{f"refinement_{case}_cpu": f"refinement_original_{case}_cpu"
       for case in ("symmetric", "factor_one", "factor_two", "factor_two_reuse")},
    "refinement_gpu": "refinement_original_gpu",
}
# These exact jobs explain preserved failures or compare diagnostic source
# trials. They are callable, but never substitutes for current runtime gates.
# New/unlisted groups remain mandatory; names and historical pass/fail outcomes
# do not classify a job. See the master program's terminal-role review.
EXPLANATORY_TEST_GROUPS = {
    **{f"driver_history_memory_{arm}_cpu": "Fresh-process supervisor record loading only; not target XLA or device memory evidence."
        for arm in ("prior", "streamed")},
    "dz5_score_replay_localize_cpu": "Short-horizon thread/replay attribution; cannot qualify the full DZ5 score oracle.",
    **{f"ledh_flow_cost_{arm}_{device}": "Isolated qualified flow dependency costs; full public value/score integration and CPU compiler RSS remain separate gates."
        for arm in ("prior_graph", "native_graph", "native_xla") for device in ("cpu", "gpu")},
    "ledh_value_localize_cpu": "Frozen-reset attribution for a preserved full-value numerical veto; not admission.",
    "ledh_value_precision_cpu": "Independent FP64 reset precision diagnosis; no runtime dtype or tolerance change.",
    "ledh_flow_localize_gpu": "Preserved cast-pair excess-precision localization, superseded by exact fraction regression.",
    **{f"remaining_svd_cost_{arm}_{dimension}_{device}": "Matched affected-consumer costs; independent numerical veto, caller collection and separate capacity disposition required."
        for device in ("cpu", "gpu") for dimension in (3, 5)
        for arm in ("prior_graph", "prior_xla", "after_graph", "after_xla")},
    **{group: "Preserved combined-suite LLVM memory failure/localization; all unchanged endpoint cases require the registered process shards."
        for group in ("remaining_svd_endpoints_cpu", "remaining_svd_endpoints_gpu",
            "remaining_svd_endpoint_crash_cpu", "remaining_svd_endpoint_sequence_cpu")},
    "svd_graph_attribution_gpu": "Interleaved graph identity/timing attribution only; independent numerical and unshared-device checks cannot waive public/default cost gates.",
    "dense_isotropic_initialization_cpu": "Original isotropic factor initialization and stopping-condition attribution; no equivalence waiver.",
    **{f"svd_cost_{arm}_{horizon}_{device}": "Matched SVD repair costs; numerical failure forbids speed ranking, separate provenance and resource disposition required."
        for arm in ("prior_graph", "prior_xla", "after_graph", "after_xla")
        for horizon in (1, 3) for device in ("cpu", "gpu")},
    "dense_seeded_attribution_cpu": "Isotropic original/current fitter attribution on identical and RNG-rounded clouds; diagnostic execution is not numerical qualification.",
    **{f"svd_scale_audit_{device}": "Actual Kalman/SRUKF SVD scale attribution; execution does not qualify numerical outputs."
        for device in ("cpu", "gpu")},
    **{f"dense_controller_svd_{device}": "Identical small-matrix SVD attribution of run03469; no runtime or comparison waiver."
        for device in ("cpu", "gpu")},
    "dense_attempt_condition_gpu": "Same-data/state attribution of run03450; cannot close complete-record qualification.",
    **{f"block_public_cost_{arm}_{dimension}_{device}":
        "Complete public block costs; repeated original-record/provenance comparisons and separate ledger disposition required."
        for arm in ("prior", "graph", "xla") for dimension in (3, 5) for device in ("cpu", "gpu")},
    **{f"block_public_churn_{device}":
        "Bounded public callback/shape churn attribution; cannot establish native eviction or waive complete original-record/cost gates."
        for device in ("cpu", "gpu")},
    "initializer_native_residual_gpu":
        "GPU attribution of the diagnostic residual-correction candidate; no runtime installation or clipping-count waiver.",
    **{f"initializer_coefficient_reference_{batched}_{device}":
        "Identical-array100/70-digit coefficient attribution only; no runtime correction, clipping-count waiver or healthy initializer admission."
        for batched in (False, True) for device in ("cpu", "gpu")},
    **{f"resolution_backend_{batched}_{device}": "Backend predicate and preguard lifecycle attribution; original public differences remain diagnostic and cannot waive qualification."
        for batched in (False, True) for device in ("cpu", "gpu")},
    "objective_resolution_diagnostic_cpu": "Historical prototype representation-resolution diagnostic; actual error/no-use qualification requires the installed guard tests.",
    "block_target_arithmetic_cpu": "Identical predecessor/proposal target arithmetic and100-digit reference; no equivalence or predicate waiver.",
    "block_rounding_diagnostic_cpu": "Partial-block strict incumbent arithmetic attribution; original uninstrumented gate remains mandatory.",
    "block_graph_registry_diagnostic_cpu": "Trace-only custom-gradient closure ancestry attribution; no numerical qualification or memory-cost claim.",
    **{f"sequential_residency_{primed}_{dimension}_{device}":
        "Sequential-specific startup/reuse/release allocation attribution; mandatory original-record and cost gates remain unchanged."
        for primed in (False, True) for dimension in (3, 5) for device in ("cpu", "gpu")},
    **{f"sequential_residency_observer_{device}":
        "Sequential retained mapping observer control without numerical calls; cannot waive cost or numerical gates."
        for device in ("cpu", "gpu")},
    **{f"sequential_public_cost_{arm}_{dimension}_{device}":
        "Complete public sequential costs; repeated original-record/provenance comparisons and separate ledger disposition required."
        for arm in ("prior", "graph", "xla") for dimension in (3, 5) for device in ("cpu", "gpu")},
    **{f"posterior_residency_observer_{device}":
        "Retained mapping-observer allocation control with no numerical calls between snapshots; cannot waive numerical or cost gates."
        for device in ("cpu", "gpu")},
    **{f"posterior_residency_{primed}_{dimension}_{device}":
        "First-XLA startup and reuse residency attribution; complete original public numerical/cost groups remain mandatory."
        for primed in (False, True) for dimension in (3, 5) for device in ("cpu", "gpu")},
    "initializer_native_residual_cpu":
        "Rejected residual-correction diagnostic candidate; mandatory initializer_native original-record groups retain the unmodified runtime gate.",
    "initializer_native_lstsq_cpu":
        "Identical-array least-squares arithmetic attribution; not original initializer equivalence.",
    "initializer_native_affine_cpu":
        "Diagnostic affine rounding adaptations; not an installed runtime repair or parity waiver.",
    "initializer_native_roundoff_cpu":
        "Same-center attribution of the preserved original initializer mismatch; original-record groups remain mandatory.",
    "geometry_active_pilot_attribution_cpu":
        "Rank-deficient pilot and HLO attribution; does not replace mandatory complete pilot records.",
    **{f"geometry_active_memory_{arm}_{capacity}_{device}":
        "Single-process active-count fit costs; complete original-record qualification is mandatory separately, and terminal repeats remain required."
        for arm in ("compact", "graph", "xla") for capacity in (16, 24) for device in ("cpu", "gpu")},
    "geometry_active_rank_diagnostic_cpu":
        "Inherited near-rank failure attribution; the identical case and rejection guard remain mandatory in geometry_active_edges_cpu/gpu.",
    **{f"geometry_preparation_reuse_{kind}_{dimension}_{device}":
        "Native/report component attribution and 3000 alternating-input calls; no original raw-kernel ranking or general memory bound."
        for kind in ("directions", "partition") for dimension in (3, 5) for device in ("cpu", "gpu")},
    **{f"geometry_preparation_memory_{kind}_{arm}_{dimension}_{device}":
        "Matched original direction/design/partition dependency with common complete outputs; enclosing initializer and terminal costs remain open."
        for kind in ("directions", "partition", "design_scalar", "design_batch") for arm in ("before", "graph", "xla")
        for dimension in (3, 5) for device in ("cpu", "gpu")},
    **{f"geometry_pilot_memory_{lane}_{arm}_{dimension}_{device}":
        "Matched prepared-direction pilot and full records; normalization/prefix and whole-initializer/terminal evidence remain open."
        for lane in ("scalar", "batch") for arm in ("before", "graph", "xla")
        for dimension in (3, 5) for device in ("cpu", "gpu")},
    **{f"geometry_fit_memory_{arm}_{dimension}_{device}":
        "Matched original suffix body and complete result/hash/report costs; whole-initializer and final repeated evidence remain required."
        for arm in ("before", "graph", "xla") for dimension in (3, 5) for device in ("cpu", "gpu")},
    **{f"geometry_control_memory_{arm}_{dimension}_{device}":
        "Complete proposal dependency costs; public full-initializer and terminal repeated evidence remain required."
        for arm in ("before", "graph", "xla") for dimension in (3, 5) for device in ("cpu", "gpu")},
    "geometry_control_eigen_attribution": "Localize inherited raw eigensystem error; original full-record tests remain mandatory.",
    "geometry_control_scalar_attribution": "Localize scalar graph empty outputs and check native LU compatibility; no runtime gate waiver.",
    "posterior_curvature_zero_diagnostic_gpu": "Isolated vector-versus-row center projection attribution for a rejected zero design; full runtime record checks remain mandatory.",
    "posterior_curvature_condition_diagnostic": "Rejected ill-conditioned dense records, original one-ULP sensitivity and100/160-digit reference; explanatory only, no tolerance waiver.",
    "posterior_curvature_rank_diagnostic": "Isolated COD pivot/threshold and source-order attribution; not a numerical gate waiver or runtime admission.",
    **{f"dense_components_{device}": "Separate component timings explain the preserved whole-D5 trigger; mandatory numerical and full-controller gates remain separate."
        for device in ("cpu", "gpu")},
    **{f"dense_extreme_{device}": "Original one-ULP self-sensitivity is explanatory; approved complete spectral comparison plus independent references remain mandatory."
        for device in ("cpu", "gpu")},
    **{f"quadratic_numerics_memory_{arm}_{kind}_{dimension}_{device}":
        "Fresh-process paired/trust dependency costs; mandatory numerical groups and complete terminal comparisons remain separate."
        for arm in ("before", "graph", "xla") for kind in ("paired", "trust")
        for dimension in (3, 5) for device in ("cpu", "gpu")},
    **{f"quadratic_batch_long_growth_{device}":
        "Bounded 10,000-call XLA allocator follow-up to small continued CPU RSS growth; not timing or general leak-freedom evidence."
        for device in ("cpu", "gpu")},
    **{f"quadratic_batch_growth_{arm}_{device}":
        "Sparse fixed-shape allocation observations after 20 warm calls; no runtime cleanup, performance ranking or general leak-freedom claim."
        for arm in ("before", "xla") for device in ("cpu", "gpu")},
    **{historical: f"Historical cfbc32d2 precision comparison; complete original-source replacement {original} remains mandatory at unchanged tolerance."
       for historical, original in ORIGINAL_AUTHORITY_REPLACEMENTS.items()},
    **{f"quadratic_batch_memory_{arm}_{dimension}_cpu":
        "Explicit CPU reference costs of the fixed-batch dependency; no default-GPU or complete-outer claim."
        for arm in ("before", "graph", "xla") for dimension in (3, 5)},
    **{f"quadratic_batch_memory_{arm}_{dimension}":
        "Descriptive fixed-batch dependency cost at a source-pinned checkpoint; complete outer costs and terminal repeats remain required."
        for arm in ("before", "graph", "xla") for dimension in (3, 5)},
    "factor_input_sensitivity": "Frozen original one-ULP sensitivity and analytic fixture reference; diagnostic only, no tolerance waiver.",
    **{f"factor_trajectory_{data}": "Diagnostic objective trajectory attribution; instrumentation must preserve all archived public fields; full-record gates remain unchanged."
        for data in ("original", "current")},
    "factor_row_decode": "Native translation of original per-row decoder as a diagnostic injection; full-record gates still required before runtime use.",
    "lifecycle_factor_initial": "Frozen original initializer intervention and identical-state objectives; diagnostic only, no runtime substitute or tolerance waiver.",
    "lifecycle_factor_inputs": "Crossed original/current prepared inputs and fitters; attribution cannot waive original full-record numerical failure.",
    **{f"lifecycle_profile_{dimension}":
        "Stage attribution of lifecycle host/allocator compilation growth; not comparative timing or terminal qualification."
        for dimension in (3, 5)},
    "lifecycle_terminal_modes": "Exact saved terminal inputs and frozen-source eigensystem attribution; strict mode failure remains open.",
    "lifecycle_factor_modes": "Crossed actual optimizer states and frozen/current Jacobian diagnostics; no full-record tolerance waiver.",
    **{f"lifecycle_memory_{arm}_{dimension}":
        "Descriptive complete-lifecycle costs; original full records and terminal repeats remain separate mandatory gates."
        for arm in ("before", "graph", "xla") for dimension in (3, 5)},
    "lifecycle_specialization": "Standalone terminal/refinement compiler operand localization; full runtime gate remains mandatory.",
    **{f"attempts_capacity_{arm}_{capacity}": "Descriptive compilation-capacity localization of enclosing second fit; no memory cap or terminal timing ranking."
        for arm in ("before", "xla") for capacity in (4, 32)},
    "attempts_dense_trust_modes": "Frozen/current source and eigensolver attribution of the D3 graph/XLA mismatch; failed numerical comparison is not waived.",
    **{f"attempts_memory_{arm}_{dimension}": "Single-process exact attempt-block cost; comparison and terminal repeated evidence remain separate."
        for arm in ("before", "graph", "xla") for dimension in (3, 5)},
    **{f"proposal_memory_{arm}_{dimension}": "Single-process proposal dependency cost; full endpoint comparisons and terminal repeats remain required."
        for arm in ("before", "graph", "xla") for dimension in (3, 5)},
    "structured_inherited_modes": "Frozen-source attribution of a failed compiler comparison; no parity threshold is waived.",
    **{f"structured_eligibility_{arm}": "Single-process changing active-size cost; no terminal performance claim."
        for arm in ("before", "after")},
    "structured_graph_xla_localization": "Crossed prepared inputs isolate a failed graph/XLA comparison; complete record gates remain unchanged.",
    **{f"structured_memory_{arm}_{dimension}":
        "Single-process checkpoint memory/performance observation; complete comparisons and terminal repeats required separately."
        for arm in ("before", "after", "graph", "xla") for dimension in (3, 5)},
    "structured_preparation_stages": "Instrumented arithmetic boundaries only; no runtime record gate is replaced.",
    "structured_fit_arithmetic": "Instrumented preparation versus same-input padding breakdown; complete runtime records remain mandatory.",
    "factor_guard_mapping_probe": "01598 CPU LLVM mapping exhaustion reproducer; required coverage is CPU modules plus combined GPU qualification.",
    "factor_guard_mapping_release": "01599 rejected cache-release trial; TensorFlow retains device executables without eviction.",
    "factor_guard_native_stack": "01597 native failure localization, not a runtime acceptance suite.",
    "factor_guard_crash_replay": "Subset of padded/domain qualification used only to localize call history.",
    "factor_guard_fixed_replay": "Subset of fixed/domain qualification used only to localize call history.",
    "factor_domain_localization": "Frozen 085baaaa eager/graph/XLA domain disagreement; actual guard checks remain mandatory.",
    "factor_domain_guard_trial": "Rejected frozen-source final-value-only guard; it misses invalid intermediate evaluations.",
    "factor_domain_events": "Frozen-source instrumentation used to design the guard; runtime reset/concurrency checks remain mandatory.",
    "active_cod_compact_step": "Rejected pinned solver trial; actual compact CPQR primal/pullback and enclosing fit tests remain mandatory.",
    "active_cod_compact_cpqr": "Pinned candidate experiment superseded by active_cod_runtime and padded_factor runtime gates.",
    "active_cod_steps": "Intermediate arithmetic observation of pinned candidate solvers, not enclosing fit qualification.",
    "active_cod_enclosure": "Instrumented initializer/loss localization; complete uninstrumented fitter is mandatory.",
    "factor_capacity_shared_cod": "Rejected shared-body solver arithmetic trial; no runtime admission.",
    "factor_shared_memory_compact": "Pinned solver-comparison resource diagnostic; not terminal before/after evidence.",
    "factor_shared_memory_shared": "Rejected shared-body solver resource diagnostic; not terminal before/after evidence.",
    "factor_enclosure_localization": "Instrumented initial weights/loss breakdown; full records and input reuse remain mandatory.",
    "padded_jacobian_localization": "Pinned compact/padded QR discrepancy observation; actual padded Jacobian regression remains mandatory.",
    "padded_dynamic_qr_localization": "Rejected bounded dynamic-shape QR trial with preserved compilation failure.",
    "padded_dynamic_qr_output": "Rejected dynamic-shape output-metadata variant, not implemented runtime.",
    "padded_shape_dispatch": "Pinned shape-dispatch trial; runtime shape dispatch and complete records remain mandatory.",
    "factor_capacity_initial": "Instrumented initializer arithmetic across capacities; uninstrumented records remain mandatory.",
    "factor_capacity_initial_fixed": "Instrumented same-initial-state arithmetic diagnostic.",
    "factor_capacity_cod": "Pinned full-COD cloning trial, not the accepted compact-CPQR implementation.",
    "factor_capacity_0": "Single-process descriptive capacity cost on an invalid full-optimizer trajectory; not admission evidence.",
    "factor_capacity_4": "Single-process descriptive capacity cost on an invalid full-optimizer trajectory; not admission evidence.",
    "factor_capacity_32": "Single-process descriptive capacity cost on an invalid full-optimizer trajectory; not admission evidence.",
    "factor_capacity_graph_0": "Preserved original graph-domain assertion failure; no valid warm timing exists.",
    "factor_capacity_graph_32": "Preserved original graph-domain assertion failure; no valid warm timing exists.",
    "factor_capacity_short_graph_0": "Four-iteration explanatory control, not full-optimizer qualification.",
    "factor_capacity_short_graph_32": "Four-iteration explanatory control, not full-optimizer qualification.",
    "factor_capacity_short_xla_0": "Four-iteration explanatory control, not full-optimizer qualification.",
    "factor_capacity_short_xla_32": "Four-iteration explanatory control, not full-optimizer qualification.",
    "fixed_fitting_fields": "Instrumented frozen fitter field breakdown; current full-record checks remain mandatory.",
    "fixed_fitting_initializer": "Decimal/COD and rejected correction-trial comparison; runtime solver/record checks remain mandatory.",
    "fixed_fitting_initializer_stages": "Instrumented identical-predecessor arithmetic, not fitter admission.",
    "padded_factor_localization": "Instrumented initial-state and loss breakdown; padded_factor remains mandatory.",
    "padded_factor_arithmetic": "Instrumented loss/pullback stages, not uninstrumented fitter qualification.",
    "factor_specialization": "Instrumented compiler-input localization; factor_runtime_inputs remains mandatory.",
    **{f"factor_guard_memory_{arm}_{mode}_{dimension}":
        "Fresh-process descriptive guard overhead; comparisons and ledger disposition required separately."
        for arm in ("checkpoint", "candidate") for mode in ("graph", "xla") for dimension in (3, 5)},
}
TEST_BATCHES = {
    "dz5_remaining_target_cpu": tuple(f"dz5_{source}_target_{batch}_cpu"
        for batch in (4, 46, 68) for source in ("merged", "archived")),
    "dz5_target_gpu": tuple(f"dz5_{source}_target_{batch}_gpu"
        for batch in (1, 4, 46, 68) for source in ("merged", "archived")),
    **{f"remote_integration_{device}": (
        f"remote_integration_hermite_{device}",
        f"remote_integration_hermite_consumers_{device}",
        f"remote_integration_neutra_{device}",
        f"remote_integration_eigen_{device}",
        f"remote_integration_genut_primitives_{device}",
        f"remaining_svd_requalification_{device}",
        "policy") for device in ("cpu", "gpu")},
    **{f"remaining_svd_cost_{device}": tuple(f"remaining_svd_cost_{arm}_{dimension}_{device}"
        for dimension in (3, 5) for arm in ("prior_graph", "prior_xla", "after_graph", "after_xla"))
        for device in ("cpu", "gpu")},
    **{f"remaining_svd_qualification_{device}": tuple(f"remaining_svd_{part}_{device}"
        for part in ("ownership", "requalification", "block", "public_first", "public_second", "public_capacity"))
        for device in ("cpu", "gpu")},
    **{f"svd_cost_{device}": tuple(f"svd_cost_{arm}_{horizon}_{device}"
        for horizon in (1, 3) for arm in ("prior_graph", "prior_xla", "after_graph", "after_xla"))
        for device in ("cpu", "gpu")},
    **{f"dense_reference_renewal_{device}": (
        *(f"dense_controller_{dimension}_{case}_{device}"
            for dimension in (1, 3) for case in ("healthy", "invalid_locator", "invalid_cloud", "score_veto", "fit_rejected")),
        *(f"dense_controller_{dimension}_{case}_{device}"
            for dimension, case in ((1, "moved_retry"), (1, "exhausted"),
                (1, "invalid_second_cloud"), (3, "invalid_second_cloud"),
                (1, "invalid_rank"), (3, "invalid_rank"), (1, "overlap"), (3, "overlap"))),
        *(f"dense_validated_fit_{dimension}_{case}_{device}"
            for dimension, case in ((1, "healthy"), (3, "healthy"), (3, "audit"), (3, "incomplete"), (3, "rank"))),
        f"dense_fit_error_order_{device}",
        *(f"dense_attempt_{dimension}_{case}_{device}"
            for dimension in (1, 3) for case in ("centered", "moved", "invalid", "fit_rejected")))
        for device in ("cpu", "gpu")},
    **{f"dense_seeded_{device}": tuple(f"dense_seeded_{dimension}_{case}_{device}"
        for dimension in (1, 3) for case in ("healthy", "invalid_locator", "invalid_cloud", "score_veto", "fit_rejected"))
        for device in ("cpu", "gpu")},
    **{f"svd_repair_{device}": (f"accurate_svd_{device}", f"svd_endpoints_{device}")
        for device in ("cpu", "gpu")},
    **{f"dense_rng_{device}": tuple(f"dense_rng_{dimension}_{device}" for dimension in (1, 3, 23))
        for device in ("cpu", "gpu")},
    **{f"dense_controller_edges_{device}": tuple(f"dense_controller_{dimension}_{case}_{device}"
        for dimension, case in ((1, "moved_retry"), (1, "exhausted"),
            (1, "invalid_second_cloud"), (3, "invalid_second_cloud"),
            (1, "invalid_rank"), (3, "invalid_rank"), (1, "overlap"), (3, "overlap")))
        for device in ("cpu", "gpu")},
    **{f"dense_controller_{device}": tuple(f"dense_controller_{dimension}_{case}_{device}"
        for dimension in (1, 3) for case in ("healthy", "invalid_locator", "invalid_cloud", "score_veto", "fit_rejected"))
        for device in ("cpu", "gpu")},
    "dense_attempt_unaffected": ("dense_attempt_3_moved_cpu", "dense_attempt_3_invalid_cpu",
        "dense_attempt_3_moved_gpu", "dense_attempt_3_invalid_gpu", "tensor_npz_cpu", "policy"),
    "dense_attempt_cpu": tuple(f"dense_attempt_{dimension}_{case}_cpu"
        for dimension in (1, 3) for case in ("centered", "moved", "invalid", "fit_rejected")),
    "dense_attempt_gpu": tuple(f"dense_attempt_{dimension}_{case}_gpu"
        for dimension in (1, 3) for case in ("centered", "moved", "invalid", "fit_rejected")),
    "dense_validated_fit_finish_cpu": (
        "dense_validated_fit_3_audit_cpu", "dense_validated_fit_3_incomplete_cpu",
        "dense_validated_fit_3_rank_cpu", "dense_fit_error_order_cpu", "fixed_geometry", "policy"),
    "dense_validated_fit_gpu": (
        "dense_validated_fit_1_healthy_gpu", "dense_validated_fit_3_healthy_gpu",
        "dense_validated_fit_3_audit_gpu", "dense_validated_fit_3_incomplete_gpu",
        "dense_validated_fit_3_rank_gpu", "dense_fit_error_order_gpu", "fixed_geometry"),
    "staged_center_completion_gpu": ("staged_center_edges_gpu", "staged_center_construction_gpu",
        "staged_center_state_gpu", "staged_center_failure_isolation_gpu"),
    **{f"staged_center_cost_{device}": tuple(f"staged_center_cost_{arm}_{dimension}_{device}"
        for arm in ("prior", "graph", "xla") for dimension in (1, 3)) for device in ("cpu", "gpu")},
    **{f"staged_center_{device}": tuple(f"staged_center_{case}_{dimension}_{device}"
        for dimension, case in ((3, "quadratic"), (1, "quartic"), (3, "quartic"),
            (3, "constant"), (3, "invalid"), (3, "cap"), (3, "cap_after"), (3, "reject"), (3, "validator_error")))
        for device in ("cpu", "gpu")},
    **{f"batched_center_reuse_{device}": tuple(f"batched_center_reuse_{case}_{batch}_{device}"
        for case in ("quadratic", "nonquadratic", "flat", "invalid_rows") for batch in (1, 3))
        for device in ("cpu", "gpu")},
    **{f"block_public_cost_{device}": tuple(f"block_public_cost_{arm}_{dimension}_{device}"
        for dimension in (3, 5) for arm in ("prior", "graph", "xla")) for device in ("cpu", "gpu")},
    **{f"block_public_options_{device}": tuple(f"block_public_options_{case}_{device}"
        for case in ("scalar_locator", "factor_one", "factor_two_reuse", "paired", "scaled_search", "score_disabled"))
        for device in ("cpu", "gpu")},
    **{f"block_public_guard_{device}": tuple(f"objective_resolution_{kind}_{batched}_{device}"
        for kind in ("block", "controlled") for batched in (False, True))
        for device in ("cpu", "gpu")},
    **{f"block_public_{device}": tuple(f"block_public_{case}_{batched}_{device}"
        for case in ("coupled", "record_reversal", "partial_heterogeneous") for batched in (False, True))
        + (f"block_public_edges_{device}",) for device in ("cpu", "gpu")},
    **{f"block_public_legacy_{device}": (f"block_public_legacy_misc_{device}",)
        + tuple(f"block_public_legacy_real_{index}_{device}" for index in range(len(BLOCK_PUBLIC_LEGACY_NAMES)))
        + tuple(f"block_public_legacy_reference_{coupling}_{stop}_{batched}_{device}"
            for coupling, stop in ((0.0, True), (3.0, True), (3.0, False)) for batched in (False, True))
        + (f"block_public_legacy_helpers_{device}",) for device in ("cpu", "gpu")},
    **{f"objective_resolution_{device}": (f"objective_resolution_boundaries_{device}",
        *(f"objective_resolution_controlled_{batched}_{device}" for batched in (False, True)),
        *(f"objective_resolution_sequential_{batched}_{device}" for batched in (False, True)),
        *(f"objective_resolution_block_{batched}_{device}" for batched in (False, True)),
        f"block_boundaries_{device}") for device in ("cpu", "gpu")},
    **{f"block_controller_{device}": tuple(f"block_controller_{case}_{batched}_{device}"
        for case in ("coupled", "record_reversal", "partial_heterogeneous") for batched in (False, True))
        for device in ("cpu", "gpu")},
    **{f"block_capture_{device}": (f"block_capture_False_{device}", f"block_capture_True_{device}",
        f"block_capture_ownership_{device}") for device in ("cpu", "gpu")},
    **{f"sequential_residency_{device}": (
        *(f"sequential_residency_{primed}_{dimension}_{device}"
          for primed in (False, True) for dimension in (3, 5)), f"sequential_residency_observer_{device}")
        for device in ("cpu", "gpu")},
    **{f"sequential_public_cost_{device}": tuple(f"sequential_public_cost_{arm}_{dimension}_{device}"
        for dimension in (3, 5) for arm in ("prior", "graph", "xla")) for device in ("cpu", "gpu")},
    **{f"program_ownership_{device}": (
        *(f"program_ownership_{case}_{device}" for case in ("terminal", "factor_one", "batched_locator", "identity")),
        *(f"sequential_public_{case}_{device}" for case in ("terminal", "factor_one", "factor_two",
            "scalar_locator", "batched_locator", "moving_budget", "edges"))) for device in ("cpu", "gpu")},
    **{f"sequential_public_{device}": tuple(f"sequential_public_{case}_{device}" for case in
        ("terminal", "terminal_reject", "symmetric", "recenter", "fit_reject", "factor_one", "factor_two",
         "factor_two_reuse", "stationary_budget", "moving_budget", "nonfinite", "paired", "scaled_search",
         "score_disabled", "resolvable", "scalar_locator", "batched_locator", "locator_budget", "edges"))
        for device in ("cpu", "gpu")},
    **{f"sequential_public_consumers_{device}": tuple(f"sequential_public_consumers_{group}_{device}"
        for group in (1, 2, 3, 4, 5, 6, "factor", "boundary")) for device in ("cpu", "gpu")},
    **{f"posterior_residency_{device}": tuple(f"posterior_residency_{primed}_{dimension}_{device}"
        for primed in (False, True) for dimension in (3, 5)) for device in ("cpu", "gpu")},
    **{f"sequential_controller_{device}": tuple(f"sequential_controller_{case}_{device}" for case in
        ("terminal", "terminal_reject", "symmetric", "recenter", "fit_reject", "factor_one", "factor_two",
         "factor_two_reuse", "stationary_budget", "moving_budget", "nonfinite", "paired", "scaled_search",
         "score_disabled", "resolvable", "scalar_locator", "batched_locator", "locator_budget", "edges"))
        for device in ("cpu", "gpu")},
    **{f"posterior_public_memory_{device}": tuple(f"posterior_public_memory_{arm}_{dimension}_{device}"
        for arm in ("prior", "graph", "xla") for dimension in (3, 5)) for device in ("cpu", "gpu")},
    **{f"posterior_public_{device}": (f"posterior_public_1_{device}", f"posterior_public_3_{device}",
        f"posterior_public_5_{device}", f"posterior_public_edges_{device}",
        f"posterior_public_dense_{device}")
        for device in ("cpu", "gpu")},
    **{f"initializer_controller_{device}": (*tuple(
        f"initializer_controller_{dimension}_{batched}_{device}" for dimension in (1, 3) for batched in (False, True)),
        f"initializer_controller_edges_{device}") for device in ("cpu", "gpu")},
    **{f"initializer_native_{device}": tuple(
        f"initializer_native_{dimension}_{batched}_{device}" for dimension in (1, 3) for batched in (False, True))
        for device in ("cpu", "gpu")},
    **{f"joint_native_{device}": (f"joint_native_1_{device}", f"joint_native_3_{device}",
        f"joint_native_edges_{device}") for device in ("cpu", "gpu")},
    "initializer_public_cpu": ("initializer_small_cpu", "initializer_gaussian_cpu",
        "initializer_scaled_cpu", "initializer_moved_cpu", "initializer_batch_cpu",
        "initializer_iterative_cpu", "initializer_enabled_cpu", "initializer_helpers_cpu",
        "initializer_current_clouds_cpu"),
    "geometry_public_cpu_finish": (
        "geometry_public_first_cpu", "geometry_public_second_cpu", "geometry_public_parity_cpu", "policy"),
    **{f"geometry_full_memory_{device}": tuple(f"geometry_full_memory_{arm}_{capacity}_{device}"
        for capacity in (24, 120) for arm in ("prior_refined", "graph", "xla")) for device in ("cpu", "gpu")},
    **{f"geometry_full_{device}": (
        *(f"geometry_full_{dimension}_{batched}_{device}" for dimension in (1, 3, 5) for batched in ("False", "True")),
        f"geometry_full_edges_{device}") for device in ("cpu", "gpu")},
    **{f"geometry_active_pilot_{device}": (
        f"geometry_active_pilot_3_{device}", f"geometry_active_pilot_5_{device}",
        f"geometry_active_pilot_edges_{device}",
        *(f"geometry_pilot_{dimension}_{device}" for dimension in (1, 3, 5)),
        f"geometry_pilot_operands_{device}", f"geometry_pilot_extras_{device}")
        for device in ("cpu", "gpu")},
    **{f"geometry_active_qualification_{device}": (
        *(f"geometry_active_{dimension}_{device}" for dimension in (1, 3, 5)),
        f"geometry_active_edges_{device}",
        *(f"geometry_fit_{dimension}_{device}" for dimension in (1, 3, 5)),
        f"geometry_fit_operands_{device}", f"geometry_public_guard_{device}")
        for device in ("cpu", "gpu")},
    **{f"geometry_active_memory_{device}": tuple(f"geometry_active_memory_{arm}_{capacity}_{device}"
        for arm in ("compact", "graph", "xla") for capacity in (16, 24)) for device in ("cpu", "gpu")},
    **{f"geometry_preparation_followup_{device}": (
        *(f"geometry_preparation_memory_directions_{arm}_{dimension}_{device}"
          for arm in ("before", "graph", "xla") for dimension in (3, 5)),
        *(f"geometry_preparation_reuse_{kind}_{dimension}_{device}"
          for kind in ("directions", "partition") for dimension in (3, 5))) for device in ("cpu", "gpu")},
    **{f"geometry_preparation_memory_{device}": tuple(f"geometry_preparation_memory_{kind}_{arm}_{dimension}_{device}"
        for kind in ("directions", "partition", "design_scalar", "design_batch") for arm in ("before", "graph", "xla")
        for dimension in (3, 5)) for device in ("cpu", "gpu")},
    **{f"geometry_pilot_memory_{device}": tuple(f"geometry_pilot_memory_{lane}_{arm}_{dimension}_{device}"
        for lane in ("scalar", "batch") for arm in ("before", "graph", "xla")
        for dimension in (3, 5)) for device in ("cpu", "gpu")},
    **{f"geometry_pilot_{device}": (f"geometry_pilot_1_{device}", f"geometry_pilot_3_{device}",
        f"geometry_pilot_5_{device}", f"geometry_pilot_operands_{device}") for device in ("cpu", "gpu")},
    **{f"geometry_fit_memory_{device}": tuple(f"geometry_fit_memory_{arm}_{dimension}_{device}"
        for arm in ("before", "graph", "xla") for dimension in (3, 5)) for device in ("cpu", "gpu")},
    **{f"geometry_fit_{device}": (f"geometry_fit_1_{device}", f"geometry_fit_3_{device}",
        f"geometry_fit_5_{device}", f"geometry_fit_operands_{device}") for device in ("cpu", "gpu")},
    **{f"geometry_control_memory_{device}": tuple(f"geometry_control_memory_{arm}_{dimension}_{device}"
        for arm in ("before", "graph", "xla") for dimension in (3, 5)) for device in ("cpu", "gpu")},
    **{f"posterior_curvature_growth_{device}": tuple(f"posterior_curvature_growth_{replicates}_{device}"
        for replicates in (2, 4, 8)) for device in ("cpu", "gpu")},
    **{f"posterior_curvature_qualified_{device}": (f"posterior_curvature_streams_{device}",
        f"posterior_curvature_1_{device}", f"posterior_curvature_3_{device}", f"posterior_curvature_5_{device}",
        f"posterior_curvature_operands_{device}", f"posterior_curvature_extras_qualified_{device}",
        f"posterior_curvature_numerical_vetoes_{device}", "policy") for device in ("cpu", "gpu")},
    "cod_tail_followup_gpu": ("cod_tail_gpu", "dense_condition_gpu", "dense_boundaries_gpu",
        "dense_derivatives_gpu", "factor_equivalence_gpu", "lifecycle_original_runtime_gpu",
        "fixed_fitting_consumers", "policy"),
    **{f"posterior_curvature_memory_{device}": tuple(f"posterior_curvature_memory_{arm}_{dimension}_{device}"
        for arm in ("before", "graph", "xla") for dimension in (3, 5)) for device in ("cpu", "gpu")},
    "cod_tail_consumer_cpu": ("posterior_curvature_1_cpu", "posterior_curvature_3_cpu", "posterior_curvature_5_cpu",
        "posterior_curvature_operands_cpu", "posterior_curvature_extras_qualified_cpu",
        "uniform_public_cpu", "quadratic_center_public_cpu", "quadratic_paired_public_cpu", "quadratic_batches_cpu",
        "lifecycle_original_runtime_3_cpu", "lifecycle_original_runtime_5_cpu", "fixed_fitting_original", "policy"),
    "cod_tail_cpu": ("cod_tail_cpu", "posterior_curvature_rank_one_cpu", "dense_condition", "dense_boundaries_cpu",
        "dense_derivatives_cpu", "factor_equivalence", "policy"),
    **{f"posterior_curvature_{device}": (f"posterior_curvature_streams_{device}",
        f"posterior_curvature_1_{device}", f"posterior_curvature_3_{device}",
        f"posterior_curvature_5_{device}", f"posterior_curvature_operands_{device}", f"posterior_curvature_extras_{device}")
        for device in ("cpu", "gpu")},
    **{f"uniform_public_{device}": (f"uniform_public_{device}",
        "quadratic_center_public_cpu" if device == "cpu" else "quadratic_center_public",
        "quadratic_paired_public_cpu" if device == "cpu" else "quadratic_paired_public",
        "quadratic_batches_cpu" if device == "cpu" else "quadratic_batches", "policy")
        for device in ("cpu", "gpu")},
    **{f"uniform_public_memory_{device}": tuple(f"uniform_public_memory_{arm}_{dimension}_{device}"
        for arm in ("before", "graph", "xla") for dimension in (3, 5)) for device in ("cpu", "gpu")},
    "uniform_cpu_finish": ("uniform_extras_cpu", "dense_derivatives_cpu", "dense_components_cpu", "quadratic_numerics_cpu", "policy"),
    "uniform_svd_cpu": ("dense_svd_cpu", "uniform_operands_cpu", "uniform_resources_cpu", "dense_boundaries_cpu", "dense_derivatives_cpu", "dense_condition", "policy"),
    "uniform_svd_gpu": ("uniform_extras_gpu", "dense_svd_gpu", "uniform_operands_gpu", "uniform_resources_gpu", "dense_boundaries_gpu", "dense_derivatives_gpu", "dense_condition_gpu", "dense_components_gpu", "policy"),
    **{f"uniform_native_{device}": (f"uniform_rounds_1_{device}", f"uniform_rounds_3_{device}",
        f"uniform_rounds_5_{device}", f"uniform_operands_{device}",
        f"quadratic_rounds_3_{device}", f"quadratic_rounds_5_{device}",
        f"quadratic_probes_{device}", f"quadratic_round_cache_{device}", "policy")
        for device in ("cpu", "gpu")},
    **{f"uniform_round_memory_{device}": tuple(f"uniform_round_memory_{arm}_{dimension}_{device}"
        for arm in ("before", "graph", "xla") for dimension in (3, 5)) for device in ("cpu", "gpu")},
    "dense_complete_numerics": ("dense_extreme_comparison", "dense_boundaries_cpu", "dense_boundaries_gpu",
        "dense_extreme_cpu", "dense_extreme_gpu", "dense_derivatives_cpu", "dense_derivatives_gpu",
        "dense_condition", "dense_condition_gpu", "quadratic_numerics_cpu", "quadratic_numerics",
        "dense_components_cpu", "dense_components_gpu", "policy"),
    "dense_reduced_svd": ("dense_condition", "dense_condition_gpu", "dense_derivatives_cpu",
        "dense_derivatives_gpu", "policy"),
    "dense_investigation": ("dense_extreme_gpu", "dense_components_cpu", "dense_components_gpu",
        "dense_derivatives_cpu", "dense_derivatives_gpu", "dense_condition", "dense_condition_gpu", "policy"),
    "approved_numerics_cpu": (
        "factor_trajectory_original", "factor_trajectory_current",
        "lifecycle_original_symmetric_cpu", "lifecycle_original_factor_one_cpu",
        "lifecycle_original_factor_two_cpu", "lifecycle_original_factor_two_reuse_cpu",
        "refinement_original_symmetric_cpu", "refinement_original_factor_one_cpu",
        "refinement_original_factor_two_cpu", "refinement_original_factor_two_reuse_cpu",
        "terminal_original_cpu", "fixed_fitting_original", "policy"),
    "approved_numerics_gpu": (
        "factor_equivalence_gpu", "dense_condition_gpu", "quadratic_numerics",
        "lifecycle_original_runtime_gpu", "lifecycle_original_symmetric_gpu",
        "lifecycle_original_factor_one_gpu", "lifecycle_original_factor_two_gpu",
        "lifecycle_original_factor_two_reuse_gpu", "refinement_original_gpu",
        "terminal_original_gpu", "factor_geometry", "fixed_fitting_consumers", "policy"),
    **{f"dense_numerics_memory_{device}": tuple(
        f"dense_numerics_memory_{arm}_{dimension}_{device}"
        for arm in ("before", "graph", "xla") for dimension in (3, 5))
        for device in ("cpu", "gpu")},
    "quadratic_public_memory": tuple(f"quadratic_public_memory_{arm}_{dimension}_{device}"
        for device in ("cpu", "gpu") for arm in ("before", "graph", "xla") for dimension in (3, 5)),
    "quadratic_round_consumers": ("quadratic_probes_cpu", "quadratic_probes_gpu",
        "quadratic_center_public_cpu", "quadratic_center_public", "quadratic_paired_public_cpu",
        "quadratic_paired_public", "quadratic_batches_cpu", "quadratic_batches", "policy"),
    "quadratic_round_cache": ("quadratic_round_cache_cpu", "quadratic_round_cache_gpu"),
    "quadratic_round_lifetime": ("quadratic_round_lifetime_cpu", "quadratic_round_lifetime_gpu"),
    "quadratic_round_growth": tuple(f"quadratic_round_growth_{rounds}_{device}"
        for device in ("cpu", "gpu") for rounds in (1, 4, 8)),
    "quadratic_round_memory": tuple(f"quadratic_round_memory_{arm}_{dimension}_{device}"
        for device in ("cpu", "gpu") for arm in ("before", "graph", "xla") for dimension in (3, 5)),
    "quadratic_rounds": ("quadratic_rounds_1_cpu", "quadratic_rounds_3_cpu", "quadratic_rounds_5_cpu",
        "quadratic_rounds_operands", "quadratic_rounds_1_gpu", "quadratic_rounds_3_gpu", "quadratic_rounds_5_gpu", "policy"),
    "quadratic_probe_consumers": ("quadratic_probes_cpu", "quadratic_probes_gpu",
        "quadratic_trust_paired_cpu", "quadratic_trust_paired_gpu",
        "quadratic_center_public_cpu", "quadratic_center_public", "quadratic_paired_public_cpu",
        "quadratic_paired_public", "quadratic_batches_cpu", "quadratic_batches", "policy"),
    "quadratic_probe_memory": tuple(f"quadratic_probe_memory_{arm}_{dimension}_{device}"
        for device in ("cpu", "gpu") for arm in ("before", "graph", "xla") for dimension in (3, 5)),
    "quadratic_numerics_consumers": ("quadratic_trust_paired_gpu", "quadratic_center_public",
        "quadratic_paired_public", "quadratic_batches", "quadratic_center_public_cpu",
        "quadratic_paired_public_cpu", "quadratic_batches_cpu", "policy"),
    **{f"quadratic_numerics_memory_{device}": tuple(
        f"quadratic_numerics_memory_{arm}_{kind}_{dimension}_{device}"
        for arm in ("before", "graph", "xla") for kind in ("paired", "trust") for dimension in (3, 5))
        for device in ("cpu", "gpu")},
    "quadratic_batch_long_growth": ("quadratic_batch_long_growth_cpu", "quadratic_batch_long_growth_gpu"),
    "quadratic_batch_growth": tuple(f"quadratic_batch_growth_{arm}_{device}"
        for arm in ("before", "xla") for device in ("cpu", "gpu")),
    "refinement_original": ("refinement_original_symmetric_cpu", "refinement_original_factor_one_cpu",
        "refinement_original_factor_two_cpu", "refinement_original_factor_two_reuse_cpu",
        "refinement_original_gpu", "policy"),
    "quadratic_batch_consumers": ("quadratic_paired_public", "policy"),
    "quadratic_batch_memory": tuple(f"quadratic_batch_memory_{arm}_{dimension}"
        for arm in ("before", "graph", "xla") for dimension in (3, 5)),
    "quadratic_batch_memory_cpu": tuple(f"quadratic_batch_memory_{arm}_{dimension}_cpu"
        for arm in ("before", "graph", "xla") for dimension in (3, 5)),
    "sequential_eigen_public": ("sequential_preparation", "sequential_score_fit", "sequential_geometry",
        "block_center", "locator_completion", "policy"),
    "lifecycle_original_runtime": ("lifecycle_original_runtime_3_cpu", "lifecycle_original_runtime_5_cpu",
        "lifecycle_original_runtime_gpu", "sequential_geometry", "block_center", "locator_completion", "policy"),
    "lifecycle_original": ("lifecycle_original_symmetric_cpu", "lifecycle_original_factor_one_cpu",
        "lifecycle_original_factor_two_cpu", "lifecycle_original_factor_two_reuse_cpu",
        "lifecycle_original_symmetric_gpu", "lifecycle_original_factor_one_gpu",
        "lifecycle_original_factor_two_gpu", "lifecycle_original_factor_two_reuse_gpu", "policy"),
    "terminal_original": ("terminal_original_cpu", "terminal_original_gpu", "policy"),
    "sequential_eigen_dependencies": ("sequential_preparation", "sequential_score_fit",
        "sequential_terminal_cpu", "sequential_terminal_gpu", "policy"),
    "sequential_eigen_consumers": ("sequential_eigen_consumers_cpu", "sequential_eigen_consumers_gpu", "policy"),
    "lifecycle_profile": ("lifecycle_profile_3", "lifecycle_profile_5", "policy"),
    "lifecycle_investigation": ("lifecycle_profile_3", "lifecycle_profile_5",
        "lifecycle_terminal_modes", "lifecycle_factor_modes", "policy"),
    "lifecycle_memory": (*(f"lifecycle_memory_{arm}_{dimension}"
        for arm in ("before", "graph", "xla") for dimension in (3, 5)), "policy"),
    "lifecycle_runtime_post_index": ("lifecycle_runtime_3_cpu", "lifecycle_runtime_5_cpu", "lifecycle_runtime_gpu", "policy"),
    "lifecycle_index_consumers": ("sequential_selection", "sequential_score_fit", "locator_completion",
        "sequential_geometry", "block_center", "policy"),
    "lifecycle_runtime": ("sequential_lifecycle_cpu", "sequential_lifecycle_gpu",
        "lifecycle_runtime_3_cpu", "lifecycle_runtime_5_cpu", "lifecycle_runtime_gpu", "policy"),
    "lifecycle_actual": ("lifecycle_actual_symmetric_cpu", "lifecycle_actual_factor_one_cpu",
        "lifecycle_actual_factor_two_cpu", "lifecycle_actual_factor_two_reuse_cpu",
        "lifecycle_actual_symmetric_gpu", "lifecycle_actual_factor_one_gpu",
        "lifecycle_actual_factor_two_gpu", "lifecycle_actual_factor_two_reuse_gpu", "policy"),
    "sequential_refinement": ("refinement_symmetric_cpu", "refinement_factor_one_cpu",
        "refinement_factor_two_cpu", "refinement_factor_two_reuse_cpu", "refinement_gpu", "policy"),
    "sequential_terminal": ("sequential_terminal_cpu", "sequential_terminal_gpu", "policy"),
    "sequential_lifecycle": ("sequential_lifecycle_cpu", "sequential_lifecycle_gpu", "policy"),
    "attempts_capacity": tuple(f"attempts_capacity_{arm}_{capacity}"
        for arm in ("before", "xla") for capacity in (4, 32)),
    "attempts_cost_followup": ("attempts_dense_trust_modes", *(f"attempts_memory_{arm}_{dimension}"
        for arm in ("before", "graph", "xla") for dimension in (3, 5)), "policy"),
    "attempts_reporting": ("sequential_proposal_public_cpu", "sequential_proposal_public_gpu",
        "attempts_public_actual_cpu", "attempts_public_actual_gpu", "policy"),
    "attempts_memory": tuple(f"attempts_memory_{arm}_{dimension}"
        for arm in ("before", "graph", "xla") for dimension in (3, 5)),
    "attempts_public_actual": ("attempts_public_actual_cpu", "attempts_public_actual_gpu", "policy"),
    "attempts_public": ("sequential_proposal_public_cpu", "sequential_proposal_public_gpu", "factor_geometry", "sequential_geometry", "block_center", "policy"),
    "sequential_attempts_expanded": ("sequential_attempts_edges", "sequential_attempts_gpu", "policy"),
    "sequential_attempts": ("sequential_attempts_cpu", "sequential_attempts_gpu", "policy"),
    "factor_decisions": ("factor_decisions_cpu", "factor_decisions_gpu", "factor_guard_cpu_fixed", "fixed_fitting_consumers", "policy"),
    "proposal_final": ("sequential_proposal_cpu", "sequential_proposal_gpu", "policy"),
    "proposal_memory": tuple(f"proposal_memory_{arm}_{dimension}" for arm in ("before", "graph", "xla") for dimension in (3, 5)),
    "proposal_followup": ("sequential_proposal_public_cpu", "sequential_proposal_public_gpu", "block_center", "factor_geometry", "policy"),
    "proposal_consumers": ("sequential_preparation", "sequential_geometry", "block_center", "factor_geometry", "policy"),
    "structured_cost_investigation": ("structured_inherited_modes", "structured_eligibility_before", "structured_eligibility_after"),
    "structured": ("structured_preparation_cpu", "structured_fit_cpu", "structured_preparation_gpu",
        "structured_fit_gpu", "fixed_fitting_consumers", "sequential_geometry", "factor_guard_cpu_fixed", "policy"),
    "structured_memory": tuple(f"structured_memory_{arm}_{dimension}"
        for arm in ("before", "after", "graph", "xla") for dimension in (3, 5)),
    "factor_guard": ("factor_guard_gpu_lifetime", "factor_guard_qualification",
        "fixed_fitting_consumers", "factor_guard_resource_lifetime", "factor_guard_cpu_fixed",
        "factor_guard_cpu_padded", "factor_guard_cpu_domain", "policy"),
    "factor_guard_memory": tuple(f"factor_guard_memory_{arm}_{mode}_{dimension}"
        for arm in ("checkpoint", "candidate") for mode in ("graph", "xla") for dimension in (3, 5)),
}
TEST_BATCHES.update({f"dense_controller_complete_{device}": (
    f"precision_operator_{device}", *TEST_BATCHES[f"dense_rng_{device}"],
    *TEST_BATCHES[f"dense_controller_{device}"], *TEST_BATCHES[f"dense_controller_edges_{device}"])
    for device in ("cpu", "gpu")})
TEST_BATCHES.update({f"ledh_flow_cost_{device}": tuple(
    f"ledh_flow_cost_{arm}_{device}" for arm in ("prior_graph", "native_graph", "native_xla"))
    for device in ("cpu", "gpu")})


def mandatory_test_groups():
    for historical, original in ORIGINAL_AUTHORITY_REPLACEMENTS.items():
        if historical in TEST_GROUPS and (original not in TEST_GROUPS or original in EXPLANATORY_TEST_GROUPS):
            raise ValueError(f"Missing mandatory original-source replacement for {historical}: {original}")
    return tuple(group for group in TEST_GROUPS if group not in EXPLANATORY_TEST_GROUPS)


FIXTURES = ("rectangular", "factor", "covariance", "sqmc", "dns", "retained_moments", "sgqf_derivatives", "joint_target", "contract_e", "tt", "tt_adapted", "tt_gaussian", "tt_actual", "tt_adjoint", "tt_scalar", "apf", "particle", "particle_alg1", "cpu_pool", "squared_density", "ttsirt_preparation", "simulation_sv", "simulation_sir", "simulation_predator_prey", "tt_scalar_retained", "tt_panel_retained", "tt_panel_ksc", *ENDPOINT_FIXTURES, *FORECAST_POOL_FIXTURES)


TEST_DEVICES = {
    "dz5_score_oracle_graph_gpu": "GPU",
    "dz5_score_oracle_xla_gpu": "GPU",
    "merged_ledh_boundary_gpu": "GPU",
    "merged_ledh_models_gpu": "GPU",
    "ledh_seed_compatibility_gpu": "GPU",
    "ledh_random_native_gpu": "GPU",
    "ledh_value_native_gpu": "GPU",
    "ledh_score_native_gpu": "GPU",
    "ledh_flow_localize_gpu": "GPU",
    "ledh_native_components_gpu": "GPU",
    **{group: "GPU" for group in TEST_BATCHES["ledh_flow_cost_gpu"]},
    "pruned_enclosing_gpu": "GPU",
    **{group: "GPU" for group in TEST_BATCHES["dz5_target_gpu"]},
    **{name: "GPU" for name in TEST_GROUPS if name.startswith("remote_integration_") and name.endswith("_gpu")},
    **{group: "GPU" for group in TEST_BATCHES["remaining_svd_cost_gpu"]},
    **{group: "GPU" for group in TEST_BATCHES["remaining_svd_qualification_gpu"]},
    "remaining_svd_scale_gpu": "GPU",
    "remaining_svd_derivatives_gpu": "GPU",
    "remaining_svd_endpoints_gpu": "GPU",
    "svd_graph_attribution_gpu": "GPU",
    **{group: "GPU" for group in TEST_BATCHES["svd_cost_gpu"]},
    **{group: "GPU" for group in TEST_BATCHES["dense_seeded_gpu"]},
    "accurate_svd_gpu": "GPU",
    "svd_endpoints_gpu": "GPU",
    "svd_scale_audit_gpu": "GPU",
    **{group: "GPU" for group in TEST_BATCHES["dense_rng_gpu"]},
    **{group: "GPU" for group in TEST_BATCHES["dense_controller_edges_gpu"]},
    "precision_operator_gpu": "GPU",
    "dense_controller_svd_gpu": "GPU",
    **{group: "GPU" for group in TEST_BATCHES["dense_controller_gpu"]},
    "dense_attempt_condition_gpu": "GPU",
    **{f"dense_attempt_{dimension}_{case}_gpu": "GPU"
        for dimension in (1, 3) for case in ("centered", "moved", "invalid", "fit_rejected")},
    **{f"dense_validated_fit_{dimension}_{case}_gpu": "GPU"
        for dimension, case in ((1, "healthy"), (3, "healthy"), (3, "audit"), (3, "incomplete"), (3, "rank"))},
    "dense_fit_error_order_gpu": "GPU",
    "block_buffer_attribution_gpu": "GPU",
    "dense_partition_validation_gpu": "GPU",
    "dense_initializer_cloud_1_gpu": "GPU",
    "dense_initializer_cloud_3_gpu": "GPU",
    **{group: "GPU" for group in TEST_BATCHES["staged_center_cost_gpu"]},
    "staged_center_failure_isolation_gpu": "GPU",
    "staged_center_state_gpu": "GPU",
    "staged_center_construction_gpu": "GPU",
    "staged_center_edges_gpu": "GPU",
    **{f"staged_center_{case}_{dimension}_gpu": "GPU"
        for dimension, case in ((1, "quadratic"), (3, "quadratic"), (1, "quartic"), (3, "quartic"),
            (3, "constant"), (3, "invalid"), (3, "cap"), (3, "cap_after"), (3, "reject"), (3, "validator_error"))},
    "target_failure_endpoint_gpu": "GPU",
    "process_containment_gpu": "GPU",
    "consensus_endpoint_gpu": "GPU",
    "batched_center_enclosing_gpu": "GPU",
    **{group: "GPU" for group in TEST_BATCHES["batched_center_reuse_gpu"]},
    **{group: "GPU" for group in TEST_BATCHES["block_public_cost_gpu"]},
    "block_public_churn_gpu": "GPU", "initializer_native_residual_gpu": "GPU",
    **{group: "GPU" for group in TEST_BATCHES["block_public_options_gpu"]},
    **{f"initializer_coefficient_reference_{batched}_gpu": "GPU" for batched in (False, True)},
    **{group: "GPU" for group in TEST_BATCHES["block_public_gpu"]},
    **{group: "GPU" for group in TEST_BATCHES["block_public_legacy_gpu"]},
    **{f"resolution_backend_{batched}_gpu": "GPU" for batched in (False, True)},
    **{group: "GPU" for group in TEST_BATCHES["objective_resolution_gpu"]},
    "block_boundaries_gpu": "GPU", "block_outer_ownership_gpu": "GPU",
    **{group: "GPU" for group in TEST_BATCHES["block_controller_gpu"]},
    "block_dependency_derivatives_cpu": "CPU",
    "block_dependency_derivatives_gpu": "GPU",
    "block_graph_registry_diagnostic_cpu": "CPU",
    **{group: "GPU" for group in TEST_BATCHES["block_capture_gpu"]},
    **{group: "GPU" for group in TEST_BATCHES["sequential_residency_gpu"]},
    **{f"geometry_full_memory_prior_{capacity}_gpu": "GPU" for capacity in (24, 120)},
    **{group: "GPU" for group in TEST_BATCHES["sequential_public_cost_gpu"]},
    **{group: "GPU" for group in TEST_BATCHES["program_ownership_gpu"]},
    "posterior_residency_observer_gpu": "GPU",
    **{group: "GPU" for group in TEST_BATCHES["sequential_public_gpu"]},
    **{group: "GPU" for group in TEST_BATCHES["sequential_public_consumers_gpu"]},
    **{group: "GPU" for group in TEST_BATCHES["posterior_public_gpu"]},
    "posterior_public_regression_gpu": "GPU",
    **{group: "GPU" for group in TEST_BATCHES["posterior_residency_gpu"]},
    **{group: "GPU" for group in TEST_BATCHES["sequential_controller_gpu"]},
    **{group: "GPU" for group in TEST_BATCHES["posterior_public_memory_gpu"]},
    **{group: "GPU" for group in TEST_BATCHES["initializer_controller_gpu"]},
    **{group: "GPU" for group in TEST_BATCHES["initializer_native_gpu"]},
    "joint_native_failures_gpu": "GPU", "initializer_iterative_gpu": "GPU",
    "joint_native_1_gpu": "GPU", "joint_native_3_gpu": "GPU", "joint_native_edges_gpu": "GPU",
    "geometry_public_capacity_gpu": "GPU",
    "geometry_public_remaining_gpu": "GPU",
    **{group: "GPU" for group in TEST_BATCHES["geometry_full_memory_gpu"]},
    **{group: "GPU" for group in TEST_BATCHES["geometry_full_gpu"]},
    **{f"geometry_active_pilot_{dimension}_gpu": "GPU" for dimension in (3, 5)},
    "geometry_active_pilot_edges_gpu": "GPU",
    **{group: "GPU" for group in TEST_BATCHES["geometry_active_memory_gpu"]},
    "geometry_public_guard_gpu": "GPU",
    **{f"geometry_active_{dimension}_gpu": "GPU" for dimension in (1, 3, 5)},
    "geometry_active_edges_gpu": "GPU",
    "gap_enclosure_diagnostics_gpu": "GPU",
    **{f"gap_compiler_memory_{arm}_gpu": "GPU" for arm in ("graph", "xla")},
    **{group: "GPU" for group in TEST_BATCHES["geometry_preparation_followup_gpu"]},
    "geometry_preparation_boundary_gpu": "GPU",
    **{group: "GPU" for group in TEST_BATCHES["geometry_preparation_memory_gpu"]},
    "geometry_preparation_gpu": "GPU",
    **{group: "GPU" for group in TEST_BATCHES["geometry_pilot_memory_gpu"]},
    "geometry_pilot_extras_gpu": "GPU",
    **{f"geometry_pilot_{dimension}_gpu": "GPU" for dimension in (1, 3, 5)},
    "geometry_pilot_operands_gpu": "GPU",
    "geometry_fit_lifetime_gpu": "GPU",
    "geometry_fit_cache_gpu": "GPU",
    **{group: "GPU" for group in TEST_BATCHES["geometry_fit_memory_gpu"]},
    **{f"geometry_fit_{dimension}_gpu": "GPU" for dimension in (1, 3, 5)},
    "geometry_fit_operands_gpu": "GPU",
    "geometry_control_callbacks_gpu": "GPU",
    **{group: "GPU" for group in TEST_BATCHES["geometry_control_memory_gpu"]},
    "geometry_control_gpu": "GPU","sequential_attempts_gpu": "GPU", "factor_decisions_gpu": "GPU", **{group: "GPU" for group in TEST_BATCHES["proposal_memory"]}, "sequential_proposal_public_gpu": "GPU", "factor_geometry": "GPU", "sequential_proposal_gpu": "GPU", **{group: "GPU" for group in TEST_BATCHES["structured_cost_investigation"]}, "structured_record_boundary": "GPU", "sequential_geometry": "GPU", **{group: "GPU" for group in TEST_BATCHES["structured_memory"]}, "structured_fit_gpu": "GPU", "structured_preparation_gpu": "GPU", "random_gpu": "GPU", "gamma_random_gpu": "GPU", "austria_preparation": "GPU", "centered_gpu": "GPU",
    "cod_tail_gpu": "GPU",
    "posterior_curvature_smoke_gpu": "GPU",
    "posterior_curvature_numerical_vetoes_gpu": "GPU",
    "posterior_curvature_zero_diagnostic_gpu": "GPU", "posterior_curvature_zero_gpu": "GPU",
    "posterior_curvature_ill_conditioned_gpu": "GPU",
    **{group: "GPU" for group in TEST_BATCHES["posterior_curvature_growth_gpu"]},
    **{group: "GPU" for group in TEST_BATCHES["posterior_curvature_gpu"]},
    "posterior_curvature_extras_qualified_gpu": "GPU",
    **{group: "GPU" for group in TEST_BATCHES["posterior_curvature_memory_gpu"]},
    "factor_equivalence_gpu": "GPU", "dense_condition_gpu": "GPU",
    **{f"uniform_rounds_{dimension}_gpu": "GPU" for dimension in (1, 3, 5)},
    "uniform_extras_gpu": "GPU", "uniform_public_gpu": "GPU",
    **{group: "GPU" for group in TEST_BATCHES["uniform_public_memory_gpu"]},
    **{f"uniform_round_growth_{rounds}_gpu": "GPU" for rounds in (1, 4, 8)},
    "uniform_operands_gpu": "GPU", "uniform_resources_gpu": "GPU", "dense_svd_gpu": "GPU",
    **{group: "GPU" for group in TEST_BATCHES["uniform_round_memory_gpu"]},
    "dense_boundaries_gpu": "GPU",
    "dense_components_gpu": "GPU",
    "dense_extreme_gpu": "GPU",
    "dense_derivatives_gpu": "GPU",
    **{group: "GPU" for group in TEST_BATCHES["dense_numerics_memory_gpu"]},
    "factor_guard_gpu_lifetime": "GPU", "fixed_fitting_consumers": "GPU",
    **{group: "GPU" for group in TEST_BATCHES["factor_guard_memory"]},
    "factor_guard_qualification": "GPU",
    "factor_domain_runtime": "GPU", "factor_domain_localization": "GPU",
    "factor_domain_guard_trial": "GPU", "factor_domain_events": "GPU",
    "active_cod_runtime": "GPU",
    "factor_capacity_shared_cod": "GPU",
    **{f"factor_shared_memory_{arm}": "GPU" for arm in ("compact", "shared")},
    **{f"factor_capacity_{capacity}": "GPU" for capacity in (0, 4, 32)},
    **{f"factor_capacity_graph_{capacity}": "GPU" for capacity in (0, 32)},
    **{f"factor_capacity_short_{mode}_{capacity}": "GPU"
       for mode in ("graph", "xla") for capacity in (0, 32)},
    **{group: "GPU" for group in ("padded_jacobian_localization", "padded_dynamic_qr_localization",
        "padded_dynamic_qr_output", "padded_shape_dispatch", "factor_capacity_initial",
        "factor_capacity_initial_fixed", "factor_capacity_cod")},
    "quadratic_batches": "GPU", "quadratic_center_public": "GPU", "quadratic_paired_public": "GPU",
    "quadratic_numerics": "GPU", "quadratic_trust_paired": "GPU",
    "quadratic_trust_paired_gpu": "GPU",
    "quadratic_probes_gpu": "GPU",
    "quadratic_rounds_1_gpu": "GPU", "quadratic_rounds_3_gpu": "GPU", "quadratic_rounds_5_gpu": "GPU",
    "quadratic_round_lifetime_gpu": "GPU",
    "quadratic_round_cache_gpu": "GPU",
    **{group: "GPU" for group in TEST_BATCHES["quadratic_public_memory"] if group.endswith("_gpu")},
    **{group: "GPU" for group in TEST_BATCHES["quadratic_round_memory"] if group.endswith("_gpu")},
    **{group: "GPU" for group in TEST_BATCHES["quadratic_round_growth"] if group.endswith("_gpu")},
    "quadratic_probe_growth_gpu": "GPU",
    **{group: "GPU" for group in TEST_BATCHES["quadratic_probe_memory"] if group.endswith("_gpu")},
    **{group: "GPU" for group in TEST_BATCHES["quadratic_numerics_memory_gpu"]},
    "quadratic_batch_long_growth_gpu": "GPU",
    **{f"quadratic_batch_growth_{arm}_gpu": "GPU" for arm in ("before", "xla")},
    **{group: "GPU" for group in TEST_BATCHES["quadratic_batch_memory"]},
    "sequential_preparation": "GPU", "sequential_score_fit": "GPU", "block_center": "GPU",
    "quadratic_initializer": "GPU", "joint_center": "GPU", "predator_tp": "GPU", "exact_incumbent": "GPU",
    "mass_matrix": "GPU", "block_score_geometry": "GPU", "fixed_stability": "GPU", "fixed_selection": "GPU", "fixed_fitting": "GPU",
    "sequential_selection": "GPU", "sequential_locator": "GPU", "batched_locator": "GPU", "locator_frozen": "GPU",
    "locator_completion": "GPU", "padded_factor": "GPU", "factor_runtime_inputs": "GPU",
    "factor_runtime_consumers": "GPU", "attempts_public_actual_gpu": "GPU",
    "attempts_dense_trust_modes": "GPU",
    "sequential_lifecycle_gpu": "GPU",
    "sequential_terminal_gpu": "GPU",
    "refinement_gpu": "GPU",
    "refinement_original_gpu": "GPU",
    "lifecycle_runtime_gpu": "GPU",
    "lifecycle_original_runtime_gpu": "GPU",
    "sequential_eigen_consumers_gpu": "GPU",
    "terminal_original_gpu": "GPU",
    **{group: "GPU" for group in TEST_BATCHES["lifecycle_original"] if group.endswith("_gpu")},
    **{group: "GPU" for group in TEST_BATCHES["lifecycle_investigation"] if group != "policy"},
    **{group: "GPU" for group in TEST_BATCHES["lifecycle_memory"] if group != "policy"},
    **{group: "GPU" for group in TEST_BATCHES["lifecycle_actual"] if group.endswith("_gpu")},
    **{group: "GPU" for group in TEST_BATCHES["attempts_capacity"]},
    **{group: "GPU" for group in TEST_BATCHES["attempts_memory"]},
    **{f"factor_memory_{arm}_{mode}": "GPU"
        for arm in ("checkpoint", "candidate") for mode in ("graph", "xla")},
    **{f"locator_memory_{count}_{capacity}": "GPU"
        for count, capacity in ((2, 128), (2, 4096), (4, 128), (4, 4096))}}
FIXTURES += ADDITIONAL_FIXTURES
FIXTURES += FORECAST_FIXTURES
FIXTURES += PREPARATION_FIXTURES
FIXTURES += CENTERED_FIXTURES
FIXTURES += TRAINING_FIXTURES
FIXTURES += STOCHASTIC_FIXTURES
FIXTURES += INITIALIZATION_FIXTURES
FIXTURES += CENTERED_TRAINING_FIXTURES
FIXTURES += SOURCE_FIXTURES
FIXTURES += LOCATOR_FIXTURES
FIXTURES += BATCHED_LOCATOR_FIXTURES


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def measurement_harness(fixture):
    names = ("filter_repair_benchmark_worker.py", "measure_filter_xla_memory.py",
             "filter_repair_endpoint_fixtures.py")
    if fixture in ADDITIONAL_FIXTURES:
        names += ("filter_repair_additional_worker.py", "filter_repair_additional_fixtures.py")
    if fixture in FORECAST_FIXTURES:
        names += ("filter_repair_forecast_worker.py", "filter_repair_forecast_fixtures.py")
    if fixture in FORECAST_POOL_FIXTURES:
        names += ("filter_repair_forecast_pool_worker.py", "filter_repair_forecast_pool_fixtures.py")
    if fixture in PREPARATION_FIXTURES:
        names += ("filter_repair_preparation_worker.py", "filter_repair_preparation_fixtures.py")
    if fixture in CENTERED_FIXTURES:
        names += ("filter_repair_centered_worker.py", "filter_repair_centered_fixtures.py")
    if fixture in TRAINING_FIXTURES:
        names += ("filter_repair_training_worker.py", "filter_repair_training_fixtures.py",
                  "filter_repair_centered_fixtures.py")
    if fixture in STOCHASTIC_FIXTURES:
        names += ("filter_repair_stochastic_worker.py", "filter_repair_stochastic_fixtures.py")
    if fixture in INITIALIZATION_FIXTURES:
        names += ("filter_repair_initialization_worker.py", "filter_repair_initialization_fixtures.py",
                  "filter_repair_centered_fixtures.py")
    if fixture in CENTERED_TRAINING_FIXTURES:
        names += ("filter_repair_centered_training_worker.py", "filter_repair_centered_training_fixtures.py",
                  "filter_repair_centered_fixtures.py")
    if fixture in SOURCE_FIXTURES:
        names += ("filter_repair_source_worker.py", "filter_repair_source_fixtures.py")
    if fixture in LOCATOR_FIXTURES:
        names += ("filter_repair_locator_worker.py", "filter_repair_locator_fixtures.py")
    if fixture in BATCHED_LOCATOR_FIXTURES:
        names += ("filter_repair_batched_locator_worker.py", "filter_repair_batched_locator_fixtures.py")
    return {name: sha(ROOT / "scripts" / name) for name in names}


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def source_hashes():
    paths = git("ls-files", "--cached", "--others", "--exclude-standard", "bayesfilter", *BASELINE_PARENT_PACKAGES, "experiments/dpf_implementation/tf_tfp", "scripts", "tests", "docs/benchmarks").splitlines()
    return {p: sha(ROOT / p) for p in sorted(set(paths))
            if (p.endswith(".py") or p == "scripts/filter_gradient_runtime_policy.json")
            and (ROOT / p).is_file()}


def records():
    """Read history one record at a time, including its full source identity."""
    return (json.loads(p.read_text()) for p in sorted(OUTPUT.glob("run-*/run.json")))


def history_summary(rows, key=None, hashes=None):
    """Retain budget fields and count exact attempts without retaining sources.

    Historical manifests carry thousands of source hashes each. Holding all of
    them while a numerical worker runs wastes gigabytes of supervisor memory.
    Compare the complete source dictionary before discarding each record.
    """
    charges = []
    attempts = 0
    for row in rows:
        charges.append({field: row[field] for field in
            ("device", "timeout_seconds", "elapsed_seconds") if field in row})
        if key is not None and row["key"] == key and row["source_sha256"] == hashes:
            attempts += 1
    return charges, attempts


def latest_record():
    """Read the final record without retaining the rest of the history."""
    latest = None
    for row in records():
        latest = row
    if latest is None:
        raise RuntimeError("No campaign run record exists")
    return latest


def charged_seconds(rows, device):
    # A crashed/unfinished run is charged its whole reserved timeout on resume.
    recorded = sum(row.get("elapsed_seconds", row["timeout_seconds"]) for row in rows if row["device"] == device)
    supplemental = [json.loads(path.read_text()) for path in OUTPUT.glob("supplemental-compute-*.json")]
    return recorded + sum(row["charged_seconds"] for row in supplemental if row["device"] == device)


def test_evidence(run):
    """A pytest exit alone cannot certify that required checks executed."""
    path = Path(run["result"]).with_name("junit.xml")
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError):
        return {"passed": False, "reason": "missing_or_invalid_junit"}
    cases = list(root.iter("testcase"))
    counts = {name: sum(len(case.findall(name)) for case in cases)
              for name in ("failure", "error", "skipped")}
    return dict(passed=bool(cases) and not any(counts.values()), tests=len(cases), **counts)


def save_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def require_unshared_cost_preflight(args):
    """Decline new public-cost workers before charging shared-device timing."""
    if (args.action != "test" or args.device != "GPU"
            or args.group not in ("svd_graph_attribution_gpu",
                                 *TEST_BATCHES["ledh_flow_cost_gpu"],
                                 *TEST_BATCHES["remaining_svd_cost_gpu"],
                                 *TEST_BATCHES["posterior_public_memory_gpu"],
                                 *TEST_BATCHES["sequential_public_cost_gpu"],
                                 *TEST_BATCHES["block_public_cost_gpu"],
                                 *TEST_BATCHES["staged_center_cost_gpu"],
                                 *TEST_BATCHES["svd_cost_gpu"])):
        return
    samples = args.gpu_preflight
    if len(samples) >= 2 and all(sample["performance_preflight_uncontended"]
                               for sample in samples[-2:]):
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    record = OUTPUT / f"cost-preflight-declined-{stamp}.json"
    save_json(record, {"schema": "filter_repair_cost_preflight.v1", "worker_launched": False,
        "group": args.group, "gpu_uuid": args.gpu_uuid, "samples": samples,
        "reason": "Shared-device preflight cannot qualify public-cost timing"})
    raise RuntimeError(f"GPU cost worker declined before launch: {record}")


def ensure_baseline():
    marker = BASELINE_ROOT / "source-manifest.json"
    if marker.exists():
        manifest = json.loads(marker.read_text())
        if manifest["commit"] != BASELINE or any(sha(BASELINE_ROOT / p) != digest for p, digest in manifest["files"].items()):
            raise RuntimeError("Baseline snapshot changed; comparison is invalid")
        paths = tuple(path for path in BASELINE_PARENT_PACKAGES if path not in manifest["files"])
        if not paths:
            return
        if any((BASELINE_ROOT / path).exists() for path in paths):
            raise RuntimeError("Unrecorded baseline package marker; inspect before recovery")
    else:
        BASELINE_ROOT.mkdir(exist_ok=False)
        manifest = {"commit": BASELINE, "files": {}}
        paths = ("bayesfilter", *BASELINE_PARENT_PACKAGES, "experiments/dpf_implementation/tf_tfp")
    # Parent markers keep Python from resolving the live regular package in
    # preference to an incomplete baseline namespace package.
    archive = subprocess.check_output(["git", "archive", BASELINE, *paths], cwd=ROOT, timeout=60)
    with tarfile.open(fileobj=io.BytesIO(archive)) as handle:
        files = tuple(member.name for member in handle.getmembers() if member.isfile())
        handle.extractall(BASELINE_ROOT, filter="data")
    manifest["files"].update({path: sha(BASELINE_ROOT / path) for path in files})
    save_json(marker, manifest)


def run_job(args):
    device = args.device
    timeout = getattr(args, "test_timeout_seconds", 900) if args.action == "test" else 300
    if args.action == "measure":
        timeout = MEASUREMENT_TIMEOUT_SECONDS.get(args.fixture, 300)
    if args.action == "test" and timeout not in TEST_TIMEOUT_SECONDS:
        raise ValueError("Test timeout must be one of the bounded registered limits")
    key = [args.action, args.group, args.arm, args.fixture, args.jit, args.size, args.repeat, device]
    hashes = source_hashes()
    rows, attempts = history_summary(records(), key, hashes)
    if charged_seconds(rows, device) + timeout > BUDGET_SECONDS[device]:
        raise RuntimeError(f"{device} campaign budget exhausted")
    if attempts >= 3:
        raise RuntimeError("Three attempts consumed for this exact job; inspect/repair scope before retry")
    if device == "GPU" and getattr(args, "gpu_preflight", None) is None:
        prepare_gpu(args, "test_gpu_index" if args.action == "test" else "measurement_gpu_index")
    require_unshared_cost_preflight(args)
    directory = OUTPUT / f"run-{len(rows) + 1:05d}"
    directory.mkdir(exist_ok=False)
    result = directory / "result.json"
    env = os.environ.copy()
    env.update({"CUDA_VISIBLE_DEVICES": args.gpu_uuid if device == "GPU" else "-1", "TF_FORCE_GPU_ALLOW_GROWTH": "true", "BAYESFILTER_TEST_DEVICE_SCOPE": "visible" if device == "GPU" else "cpu", "TF_NUM_INTRAOP_THREADS": "2", "TF_NUM_INTEROP_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MPLBACKEND": "Agg", "PYTHONHASHSEED": "0"})
    if args.action == "test":
        if args.group == "block_buffer_attribution_gpu":
            env["XLA_FLAGS"] = (env.get("XLA_FLAGS", "") +
                f" --xla_dump_to={directory / 'xla'} --xla_dump_hlo_as_text"
                " --xla_dump_hlo_module_re=inference_execute").strip()
        if args.group == "source_preparation_localization":
            env["FILTER_REPAIR_REPEAT_STACKS"] = "1"
        if args.arm == "before":
            ensure_baseline()
            env["FILTER_REPAIR_SOURCE_ROOT"] = str(BASELINE_ROOT)
        command = [sys.executable, "scripts/filter_repair_test_worker.py", "-q", *TEST_GROUPS[args.group], f"--junitxml={directory / 'junit.xml'}"]
        if args.group == "factor_guard_native_stack":
            if device != "CPU":
                raise ValueError("The factor crash stack diagnostic is CPU-only")
            command = ["gdb", "-batch", "-return-child-result", "-ex", "set debuginfod enabled off",
                "-ex", "set auto-solib-add off", "-ex", "set auto-load python-scripts off",
                "-ex", "set pagination off", "-ex", "run",
                "-ex", "sharedlibrary libtensorflow", "-ex", "thread apply all bt 24",
                "--args", *command]
    elif args.action == "measure":
        ensure_baseline()
        source = BASELINE_ROOT if args.arm == "before" else ROOT
        worker = "filter_repair_additional_worker.py" if args.fixture in ADDITIONAL_FIXTURES else "filter_repair_benchmark_worker.py"
        if args.fixture in FORECAST_FIXTURES:
            worker = "filter_repair_forecast_worker.py"
        if args.fixture in FORECAST_POOL_FIXTURES:
            worker = "filter_repair_forecast_pool_worker.py"
        if args.fixture in PREPARATION_FIXTURES:
            worker = "filter_repair_preparation_worker.py"
        if args.fixture in CENTERED_FIXTURES:
            worker = "filter_repair_centered_worker.py"
        if args.fixture in TRAINING_FIXTURES:
            worker = "filter_repair_training_worker.py"
        if args.fixture in STOCHASTIC_FIXTURES:
            worker = "filter_repair_stochastic_worker.py"
        if args.fixture in INITIALIZATION_FIXTURES:
            worker = "filter_repair_initialization_worker.py"
        if args.fixture in CENTERED_TRAINING_FIXTURES:
            worker = "filter_repair_centered_training_worker.py"
        if args.fixture in SOURCE_FIXTURES:
            worker = "filter_repair_source_worker.py"
        if args.fixture in LOCATOR_FIXTURES:
            worker = "filter_repair_locator_worker.py"
        if args.fixture in BATCHED_LOCATOR_FIXTURES:
            worker = "filter_repair_batched_locator_worker.py"
        command = [sys.executable, str(ROOT / "scripts" / worker), "--source-root", str(source), "--fixture", args.fixture, "--jit", args.jit, "--size", str(args.size), "--device", device, "--output", str(result)]
    elif args.action == "audit":
        command = [sys.executable, "scripts/audit_filter_gradient_policy.py", "--output", str(directory / "audit.json.gz"), "--markdown", str(directory / "audit.md")]
    elif args.action == "compare":
        command = [sys.executable, "scripts/compare_filter_repair_campaign.py", "--output", str(result)]
    else:
        raise ValueError(args.action)
    record = {"schema": "filter_repair_run.v1", "key": key, "started_utc": datetime.now(timezone.utc).isoformat(), "state": "running", "device": device, "timeout_seconds": timeout, "command": command, "cwd": str(ROOT), "environment": {k: env[k] for k in ("CUDA_VISIBLE_DEVICES", "TF_FORCE_GPU_ALLOW_GROWTH", "BAYESFILTER_TEST_DEVICE_SCOPE", "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS", "OPENBLAS_NUM_THREADS", "PYTHONHASHSEED")}, "git_head": git("rev-parse", "HEAD"), "git_diff_stat": git("diff", "--stat"), "source_sha256": hashes, "plan": PLAN, "result": str(result), "log": str(directory / "process.log")}
    if args.action == "test" and args.group == "block_buffer_attribution_gpu":
        record["environment"]["XLA_FLAGS"] = env["XLA_FLAGS"]
        record["diagnostic_xla_dump"] = {"path": str(directory / "xla"),
            "module_pattern": "inference_execute", "timing_eligible": False}
    if getattr(args, "gpu_preflight", None) is not None:
        record["gpu_preflight"] = args.gpu_preflight
        record["gpu_uuid"] = args.gpu_uuid
        record["gpu_performance_preflight_uncontended"] = all(
            sample["performance_preflight_uncontended"] for sample in args.gpu_preflight[-2:])
    if env.get("FILTER_REPAIR_REPEAT_STACKS") == "1":
        record["diagnostic_stack_interval_seconds"] = 45
    save_json(directory / "run.json", record)
    started = time.monotonic()
    print(json.dumps({"run": str(directory), "command": command}), flush=True)
    with (directory / "process.log").open("x") as log:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            if args.action == "test" and args.group in (
                    "factor_guard_mapping_probe", "factor_guard_mapping_release",
                    "geometry_public_memory_attribution_cpu", "remaining_svd_endpoint_sequence_cpu"):
                from filter_repair_process_memory import wait_observing_memory

                code = wait_observing_memory(process, timeout, directory)
            else:
                code = process.wait(timeout=timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt) as exc:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            code = 130 if isinstance(exc, KeyboardInterrupt) else 124
    if args.action == "test":
        record["test_evidence"] = test_evidence(record)
        if code == 0 and not record["test_evidence"]["passed"]:
            code = 1
    record.update(state="passed" if code == 0 else "failed", exit_code=code, elapsed_seconds=time.monotonic() - started)
    save_json(directory / "run.json", record)
    print(json.dumps({"state": record["state"], "elapsed_seconds": record["elapsed_seconds"], "log": record["log"]}), flush=True)
    if code:
        print((directory / "process.log").read_text()[-14000:])
    return code


def gate():
    try:
        policy = verify_source_policy(ROOT, ROOT / "scripts/filter_gradient_runtime_policy.json")
    except (OSError, ValueError, KeyError, SyntaxError) as exc:
        policy = {"passed": False, "error": str(exc)}
    ledger = json.loads((ROOT / "docs/plans/filter_gradient_repair_ledger_20260917.json").read_text())
    pending = [item["id"] for item in ledger["findings"] if item["status"] != "closed" or not item.get("evidence") or any(not (ROOT / path).is_file() for path in item.get("evidence", []))]
    if {item["id"] for item in ledger["findings"]} != {f"F{i:02d}" for i in range(1, 21)}:
        pending.append("incomplete_finding_inventory")
    current = source_hashes()
    qualified = set()
    comparisons = False
    for row in records():
        if row["state"] != "passed" or row["source_sha256"] != current:
            continue
        if row["key"][0] == "compare":
            comparisons = True
        elif row["key"][0] == "test" and row["key"][2] == "after":
            group = row["key"][1]
            if row["device"] == TEST_DEVICES.get(group, "CPU") and test_evidence(row)["passed"]:
                qualified.add(group)
    missing = [group for group in mandatory_test_groups() if group not in qualified]
    complete = not pending and not missing and bool(comparisons) and policy["passed"]
    print(json.dumps({"merge_allowed": complete, "open_findings": pending, "missing_current_tests": missing, "current_comparison": bool(comparisons), "source_policy": policy}, indent=2))
    return 0 if complete else 1


def prepare_gpu(args, option):
    """Recheck availability while preserving a matrix's physical device."""
    samples = check_gpu_available(getattr(args, option, None))
    selected = samples[-1]
    if getattr(args, "gpu_uuid", selected["selected_uuid"]) != selected["selected_uuid"]:
        raise RuntimeError("Physical GPU identity changed during the campaign matrix")
    setattr(args, option, selected["selected_gpu_index"])
    args.gpu_uuid = selected["selected_uuid"]
    args.gpu_preflight = samples


def same_gpu(run, args):
    """Absent physical identity cannot qualify a resumed GPU comparison."""
    return run.get("gpu_uuid") == args.gpu_uuid


def check_matrix_state(frozen):
    if (OUTPUT / "pause-request.json").exists():
        raise RuntimeError("Campaign paused between workers; restart the matrix to resume")
    if source_hashes() != frozen:
        raise RuntimeError("Source changed during the campaign matrix")


def measurement_modes(name):
    """Keep complete public replay timing alongside numerical compilation arms."""
    if name == "cpu_forecast_pool":
        return ("eager",)
    return (("off", "on", "eager") if name in ("source_route_sequence", "source_guard_gates", "cpu_forecast_shard",
            "exact_incumbent", "sequential_score_fit", "mass_precision", "mass_structured", "block_score_geometry",
            "fixed_stability", "fixed_selection", "fixed_fitting", "sequential_replay", "sequential_search_selection",
            "sequential_scalar_locator", "sequential_batched_locator", "sequential_batched_locator_progress")
            else ("off", "on"))


def measurement_device(name):
    """Keep external CPU generation separate from the default GPU kernels."""
    return "CPU" if name in ("cpu_pool", *FORECAST_POOL_FIXTURES) else "GPU"


def declared_test_device(group):
    """Fail on omitted GPU metadata rather than silently launch a CPU reference."""
    device = TEST_DEVICES.get(group, "CPU")
    if group.endswith("_gpu") and device != "GPU":
        raise ValueError(f"GPU-labeled test group lacks GPU registration: {group}")
    return device


def run_matrix(args):
    """Resume registered jobs sequentially; failures retain their original evidence."""
    from compare_filter_repair_campaign import (
        ONE_SIZE,
        baseline_compilation_failure,
        compare_pair,
        current_provenance,
    )

    frozen = source_hashes()
    if args.stage == "tests":
        batch = getattr(args, "test_batch", "all")
        groups = mandatory_test_groups() if batch == "all" else TEST_BATCHES[batch]
        unknown = set(groups) - TEST_GROUPS.keys()
        if unknown:
            raise ValueError(f"Unknown test groups in batch {batch}: {sorted(unknown)}")
        devices = {group: declared_test_device(group) for group in groups}
        if any(device == "GPU" for device in devices.values()):
            check_matrix_state(frozen)
            prepare_gpu(args, "test_gpu_index")
        for group in groups:
            check_matrix_state(frozen)
            device = devices[group]
            if any(row["key"][:3] == ["test", group, "after"] and row["state"] == "passed"
                   and row["source_sha256"] == frozen and row["device"] == device
                   and (device != "GPU" or same_gpu(row, args))
                   and row["key"][6] == getattr(args, "repeat", 0)
                   and test_evidence(row)["passed"] for row in records()):
                continue
            job = argparse.Namespace(**vars(args))
            job.gpu_preflight = None
            job.action, job.group, job.arm, job.device = "test", group, "after", device
            if device == "GPU":
                prepare_gpu(job, "test_gpu_index")
            code = run_job(job)
            if code:
                return code
        return 0

    ensure_baseline()
    fixtures = ((args.fixture,) if args.selection == "fixture" else
                ADDITIONAL_FIXTURES if args.selection == "additional" else
                ENDPOINT_FIXTURES if args.selection == "new" else FIXTURES)
    if any(measurement_device(name) == "GPU" for name in fixtures):
        check_matrix_state(frozen)
        prepare_gpu(args, "measurement_gpu_index")
    marker = json.loads((BASELINE_ROOT / "source-manifest.json").read_text())
    available = {}
    for row in records():
        if row["key"][0] != "measure" or not Path(row["result"]).is_file():
            continue
        if row["device"] == "GPU" and (not hasattr(args, "gpu_uuid") or not same_gpu(row, args)):
            continue
        if (row["device"] == "GPU" and args.stage == "repeat"
                and row.get("gpu_performance_preflight_uncontended") is not True):
            continue
        value = json.loads(Path(row["result"]).read_text())
        try:
            current_provenance(row, value, row["key"][2], measurement_harness(row["key"][3]), marker["files"])
        except (ValueError, KeyError):
            continue
        available[tuple(row["key"][2:])] = (row, value)

    def execute(name, size, repeat, arm, mode):
        check_matrix_state(frozen)
        device = measurement_device(name)
        key = (arm, name, mode, size, repeat, device)
        if key in available:
            row, value = available[key]
        else:
            job = argparse.Namespace(**vars(args))
            job.gpu_preflight = None
            job.action, job.fixture, job.size, job.repeat = "measure", name, size, repeat
            job.arm, job.jit, job.device = arm, mode, device
            if device == "GPU":
                prepare_gpu(job, "measurement_gpu_index")
            code = run_job(job)
            row = latest_record()
            if not Path(row["result"]).is_file():
                raise RuntimeError(f"Missing measurement artifact after exit {code}: {row['log']}")
            value = json.loads(Path(row["result"]).read_text())
            current_provenance(row, value, arm, measurement_harness(name), marker["files"])
            available[key] = row, value
        if value["status"] != "passed" and (arm != "before" or not baseline_compilation_failure(value)):
            raise RuntimeError(f"Measurement failure requires repair: {row['result']}")
        return row, value

    for name in fixtures:
        for size in ((1,) if name in ONE_SIZE else (1, 2)):
            for repeat in (range(1) if args.stage == "qualify" else range(3)):
                for mode in measurement_modes(name):
                    before_run, before = execute(name, size, repeat, "before", mode)
                    baseline_attempt = before_run["result"]
                    if before["status"] != "passed" and mode == "on":
                        before_run, before = execute(name, size, repeat, "before", "off")
                    if before["status"] != "passed":
                        before_run, before = execute(name, size, repeat, "before", "eager")
                    after_run, after = execute(name, size, repeat, "after", mode)
                    error = compare_pair(before, after)
                    print(json.dumps({"parity": "passed", "fixture": name, "size": size,
                        "repeat": repeat, "mode": mode, "max_absolute_error": error,
                        "before": before_run["result"], "after": after_run["result"],
                        "baseline_attempt": baseline_attempt}), flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("status", "test", "measure", "matrix", "pause", "audit", "compare", "gate"))
    parser.add_argument("--stage", choices=("qualify", "repeat", "tests"), default="qualify")
    parser.add_argument("--test-batch", choices=("all", *TEST_BATCHES), default="all")
    parser.add_argument("--selection", choices=("fixture", "new", "additional", "all"), default="all")
    parser.add_argument("--group", choices=tuple(TEST_GROUPS), default="policy")
    parser.add_argument("--fixture", choices=FIXTURES, default="covariance")
    parser.add_argument("--arm", choices=("before", "after"), default="after")
    parser.add_argument("--jit", choices=("on", "off", "eager"), default="on")
    parser.add_argument("--size", type=int, choices=(1, 2), default=1)
    parser.add_argument("--repeat", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--device", choices=("CPU", "GPU"), default="GPU")
    parser.add_argument("--test-timeout-seconds", type=int, choices=TEST_TIMEOUT_SECONDS, default=900,
                        help="Smaller focused-test reservation; cumulative caps and 900-second ceiling remain fixed")
    parser.add_argument("--test-gpu-index", type=int, choices=(0, 1, 2, 3), default=None,
                        help="Optional physical GPU for tests; default selects an available non-desktop GPU")
    parser.add_argument("--measurement-gpu-index", type=int, choices=(0, 1, 2, 3), default=None,
                        help="Optional physical GPU for matched costs; default auto-selects and pins one device")
    args = parser.parse_args()
    if args.test_batch != "all" and not (args.action == "matrix" and args.stage == "tests"):
        parser.error("--test-batch is only available for a test matrix")
    if args.test_timeout_seconds != 900 and not (args.action == "test" or
                                               (args.action == "matrix" and args.stage == "tests")):
        parser.error("--test-timeout-seconds is only available for correctness tests")
    if args.test_gpu_index is not None and not (args.action == "test" or
                                       (args.action == "matrix" and args.stage == "tests")):
        parser.error("--test-gpu-index is only available for correctness tests")
    if args.measurement_gpu_index is not None and not (args.action == "measure" or
            (args.action == "matrix" and args.stage in ("qualify", "repeat"))):
        parser.error("--measurement-gpu-index is only available for measurements")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if args.action == "pause":
        save_json(OUTPUT / "pause-request.json", {"requested_utc": datetime.now(timezone.utc).isoformat()})
        print(json.dumps({"state": "pause_requested", "active_worker": "allowed_to_finish"}))
        return 0
    with (OUTPUT / "campaign.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.action == "status":
            rows, _ = history_summary(records())
            print(json.dumps({"branch": git("branch", "--show-current"), "baseline": BASELINE, "runs": len(rows), "budget_seconds": BUDGET_SECONDS, "charged_seconds": {d: charged_seconds(rows, d) for d in BUDGET_SECONDS}, "artifact_root": str(OUTPUT)}, indent=2))
            return 0
        if args.action == "gate":
            return gate()
        if args.action == "matrix":
            (OUTPUT / "pause-request.json").unlink(missing_ok=True)
            return run_matrix(args)
        return run_job(args)


if __name__ == "__main__":
    raise SystemExit(main())
