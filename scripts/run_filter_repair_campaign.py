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
from filter_repair_endpoint_fixtures import FIXTURES as ENDPOINT_FIXTURES
from filter_repair_additional_fixtures import FIXTURES as ADDITIONAL_FIXTURES
from filter_repair_forecast_fixtures import FIXTURES as FORECAST_FIXTURES
from filter_repair_forecast_pool_fixtures import FIXTURES as FORECAST_POOL_FIXTURES
from filter_repair_preparation_fixtures import FIXTURES as PREPARATION_FIXTURES
from filter_repair_centered_fixtures import FIXTURES as CENTERED_FIXTURES
from filter_repair_centered_training_fixtures import FIXTURES as CENTERED_TRAINING_FIXTURES
from filter_repair_initialization_fixtures import FIXTURES as INITIALIZATION_FIXTURES
from filter_repair_training_fixtures import FIXTURES as TRAINING_FIXTURES
from filter_repair_stochastic_fixtures import FIXTURES as STOCHASTIC_FIXTURES
from filter_repair_source_fixtures import FIXTURES as SOURCE_FIXTURES
from filter_repair_locator_fixtures import FIXTURES as LOCATOR_FIXTURES
from filter_repair_batched_locator_fixtures import FIXTURES as BATCHED_LOCATOR_FIXTURES

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
BUDGET_SECONDS = {"CPU": 32 * 3600, "GPU": 52 * 3600}
TEST_TIMEOUT_SECONDS = (60, 120, 300, 900)
# The original eager full fitter takes 294 s for two replicates (run 01303).
# Reserve the same bounded ceiling for both source arms at either extent.
MEASUREMENT_TIMEOUT_SECONDS = {"fixed_fitting": 900}
TEST_GROUPS = {
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
        "tests/test_filter_repair_posterior_curvature_extras.py", "-k", "not ill_conditioned") for device in ("cpu", "gpu")},
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
    "genut": ("tests/highdim/test_cubature_genut_batch.py", "tests/highdim/test_genut_batch_primal_parity.py", "tests/highdim/test_genut_batch_general_route_parity.py", "tests/highdim/test_ledh_contract_e_canonical_lgssm_phase5.py"),
    "genut_graph": ("tests/highdim/test_genut_batch_primal_parity.py::test_public_score_traces_in_graph_and_xla",),
    "genut_targets": ("tests/test_genut_neutra_targets.py",),
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
    "policy": ("tests/test_filter_repair_campaign.py", "tests/test_filter_repair_policy.py"),
}
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


def mandatory_test_groups():
    for historical, original in ORIGINAL_AUTHORITY_REPLACEMENTS.items():
        if historical in TEST_GROUPS and (original not in TEST_GROUPS or original in EXPLANATORY_TEST_GROUPS):
            raise ValueError(f"Missing mandatory original-source replacement for {historical}: {original}")
    return tuple(group for group in TEST_GROUPS if group not in EXPLANATORY_TEST_GROUPS)


FIXTURES = ("rectangular", "factor", "covariance", "sinkhorn_jvp", "sqmc", "dns", "retained_moments", "sgqf_derivatives", "joint_target", "genut", "contract_e", "tt", "tt_adapted", "tt_gaussian", "tt_actual", "tt_adjoint", "tt_scalar", "apf", "particle", "particle_alg1", "cpu_pool", "squared_density", "ttsirt_preparation", "simulation_sv", "simulation_sir", "simulation_predator_prey", "tt_scalar_retained", "tt_panel_retained", "tt_panel_ksc", *ENDPOINT_FIXTURES, *FORECAST_POOL_FIXTURES)


TEST_DEVICES = {
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
    "sequential_preparation": "GPU", "sequential_geometry": "GPU", "sequential_score_fit": "GPU", "block_center": "GPU",
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
    return [json.loads(p.read_text()) for p in sorted(OUTPUT.glob("run-*/run.json"))]


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
        return dict(passed=False, reason="missing_or_invalid_junit")
    cases = list(root.iter("testcase"))
    counts = {name: sum(len(case.findall(name)) for case in cases)
              for name in ("failure", "error", "skipped")}
    return dict(passed=bool(cases) and not any(counts.values()), tests=len(cases), **counts)


def save_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


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
    rows = records()
    device = args.device
    timeout = getattr(args, "test_timeout_seconds", 900) if args.action == "test" else 300
    if args.action == "measure":
        timeout = MEASUREMENT_TIMEOUT_SECONDS.get(args.fixture, 300)
    if args.action == "test" and timeout not in TEST_TIMEOUT_SECONDS:
        raise ValueError("Test timeout must be one of the bounded registered limits")
    if charged_seconds(rows, device) + timeout > BUDGET_SECONDS[device]:
        raise RuntimeError(f"{device} campaign budget exhausted")
    key = [args.action, args.group, args.arm, args.fixture, args.jit, args.size, args.repeat, device]
    hashes = source_hashes()
    attempts = [row for row in rows if row["key"] == key and row["source_sha256"] == hashes]
    if len(attempts) >= 3:
        raise RuntimeError("Three attempts consumed for this exact job; inspect/repair scope before retry")
    gpu_index = (getattr(args, "test_gpu_index", 2) if args.action == "test"
                 else getattr(args, "measurement_gpu_index", 2) if args.action == "measure" else 2)
    if device == "GPU" and getattr(args, "gpu_preflight", None) is None:
        args.gpu_preflight = check_gpu_idle(gpu_index)
    directory = OUTPUT / f"run-{len(rows) + 1:05d}"
    directory.mkdir(exist_ok=False)
    result = directory / "result.json"
    env = os.environ.copy()
    env.update({"CUDA_VISIBLE_DEVICES": str(gpu_index) if device == "GPU" else "-1", "TF_FORCE_GPU_ALLOW_GROWTH": "true", "TF_NUM_INTRAOP_THREADS": "2", "TF_NUM_INTEROP_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MPLBACKEND": "Agg", "PYTHONHASHSEED": "0"})
    if args.action == "test":
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
    record = {"schema": "filter_repair_run.v1", "key": key, "started_utc": datetime.now(timezone.utc).isoformat(), "state": "running", "device": device, "timeout_seconds": timeout, "command": command, "cwd": str(ROOT), "environment": {k: env[k] for k in ("CUDA_VISIBLE_DEVICES", "TF_FORCE_GPU_ALLOW_GROWTH", "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS", "OPENBLAS_NUM_THREADS", "PYTHONHASHSEED")}, "git_head": git("rev-parse", "HEAD"), "git_diff_stat": git("diff", "--stat"), "source_sha256": hashes, "plan": PLAN, "result": str(result), "log": str(directory / "process.log")}
    if getattr(args, "gpu_preflight", None) is not None:
        record["gpu_preflight"] = args.gpu_preflight
    if env.get("FILTER_REPAIR_REPEAT_STACKS") == "1":
        record["diagnostic_stack_interval_seconds"] = 45
    save_json(directory / "run.json", record)
    started = time.monotonic()
    print(json.dumps({"run": str(directory), "command": command}), flush=True)
    with (directory / "process.log").open("x") as log:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            if args.action == "test" and args.group in (
                    "factor_guard_mapping_probe", "factor_guard_mapping_release"):
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
    rows = records()
    current = source_hashes()
    missing = []
    for group in mandatory_test_groups():
        candidates = [row for row in rows if row["key"][:3] == ["test", group, "after"]
                      and row["state"] == "passed" and row["source_sha256"] == current
                      and row["device"] == TEST_DEVICES.get(group, "CPU")
                      and test_evidence(row)["passed"]]
        if not candidates:
            missing.append(group)
    comparisons = [row for row in rows if row["key"][0] == "compare" and row["state"] == "passed" and row["source_sha256"] == current]
    complete = not pending and not missing and bool(comparisons) and policy["passed"]
    print(json.dumps({"merge_allowed": complete, "open_findings": pending, "missing_current_tests": missing, "current_comparison": bool(comparisons), "source_policy": policy}, indent=2))
    return 0 if complete else 1


def check_gpu_idle(gpu_index=2):
    if gpu_index not in (2, 3):
        raise ValueError("Campaign GPU index must be 2 or 3")
    samples, consecutive_idle = [], 0
    # Utilization is sampled over an interval and can outlive the prior worker.
    for attempt in range(6):
        output = subprocess.check_output([
            "nvidia-smi", "-i", str(gpu_index), "--query-gpu=memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ], text=True, timeout=5)
        memory, utilization = (int(value.strip()) for value in output.strip().split(","))
        samples.append(dict(memory_mib=memory, utilization_percent=utilization))
        consecutive_idle = consecutive_idle + 1 if memory <= 100 and utilization <= 5 else 0
        if consecutive_idle == 2:
            return samples
        if attempt < 5:
            time.sleep(2)
    raise RuntimeError(f"GPU{gpu_index} contention veto after bounded recheck: {samples}")


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


def run_matrix(args):
    """Resume registered jobs sequentially; failures retain their original evidence."""
    from compare_filter_repair_campaign import (
        ONE_SIZE, baseline_compilation_failure, compare_pair, current_provenance,
    )

    frozen = source_hashes()
    if args.stage == "tests":
        batch = getattr(args, "test_batch", "all")
        groups = mandatory_test_groups() if batch == "all" else TEST_BATCHES[batch]
        unknown = set(groups) - TEST_GROUPS.keys()
        if unknown:
            raise ValueError(f"Unknown test groups in batch {batch}: {sorted(unknown)}")
        for group in groups:
            check_matrix_state(frozen)
            device = TEST_DEVICES.get(group, "CPU")
            if any(row["key"][:3] == ["test", group, "after"] and row["state"] == "passed"
                   and row["source_sha256"] == frozen and row["device"] == device
                   and row["key"][6] == getattr(args, "repeat", 0)
                   and test_evidence(row)["passed"] for row in records()):
                continue
            job = argparse.Namespace(**vars(args))
            job.action, job.group, job.arm, job.device = "test", group, "after", device
            if device == "GPU":
                job.gpu_preflight = check_gpu_idle(getattr(args, "test_gpu_index", 2))
            code = run_job(job)
            if code:
                return code
        return 0

    ensure_baseline()
    marker = json.loads((BASELINE_ROOT / "source-manifest.json").read_text())
    available = {}
    for row in records():
        if row["key"][0] != "measure" or not Path(row["result"]).is_file():
            continue
        if (row["device"] == "GPU" and row.get("environment", {}).get("CUDA_VISIBLE_DEVICES")
                != str(getattr(args, "measurement_gpu_index", 2))):
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
            job.action, job.fixture, job.size, job.repeat = "measure", name, size, repeat
            job.arm, job.jit, job.device = arm, mode, device
            if device == "GPU":
                job.gpu_preflight = check_gpu_idle(getattr(args, "measurement_gpu_index", 2))
            code = run_job(job)
            row = records()[-1]
            if not Path(row["result"]).is_file():
                raise RuntimeError(f"Missing measurement artifact after exit {code}: {row['log']}")
            value = json.loads(Path(row["result"]).read_text())
            current_provenance(row, value, arm, measurement_harness(name), marker["files"])
            available[key] = row, value
        if value["status"] != "passed" and (arm != "before" or not baseline_compilation_failure(value)):
            raise RuntimeError(f"Measurement failure requires repair: {row['result']}")
        return row, value

    fixtures = ((args.fixture,) if args.selection == "fixture" else
                ADDITIONAL_FIXTURES if args.selection == "additional" else
                ENDPOINT_FIXTURES if args.selection == "new" else FIXTURES)
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
    parser.add_argument("--test-gpu-index", type=int, choices=(2, 3), default=2,
                        help="Physical GPU for correctness tests only")
    parser.add_argument("--measurement-gpu-index", type=int, choices=(2, 3), default=2,
                        help="Physical GPU for fresh matched before/after groups; never mix repeat devices")
    args = parser.parse_args()
    if args.test_batch != "all" and not (args.action == "matrix" and args.stage == "tests"):
        parser.error("--test-batch is only available for a test matrix")
    if args.test_timeout_seconds != 900 and not (args.action == "test" or
                                               (args.action == "matrix" and args.stage == "tests")):
        parser.error("--test-timeout-seconds is only available for correctness tests")
    if args.test_gpu_index != 2 and not (args.action == "test" or
                                       (args.action == "matrix" and args.stage == "tests")):
        parser.error("--test-gpu-index is only available for correctness tests")
    if args.measurement_gpu_index != 2 and not (args.action == "measure" or
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
            rows = records()
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
