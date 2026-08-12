#!/usr/bin/env python3
"""Run bounded Phase 6 scalar Zhao-Cui comparator certification diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-contract-e-tp-phase6")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tensorflow as tf

import bayesfilter.highdim as highdim
from bayesfilter.highdim import ledh_contract_e_tp_scalar_sv_tf as tp_scalar
from bayesfilter.ledh_fd_policy import evaluate_ledh_fd_policy
from bayesfilter.highdim.sv_mixture_cut4 import (
    ExactTransformedSVSSM,
    KSCMixtureTransformedSVSSM,
    exact_transformed_sv_observations,
    transformed_sv_observations,
)
from bayesfilter.highdim import ukf_initializer as p76
from bayesfilter.highdim.ukf_scout import ukf_scout_result_from_paths
from scripts.filtering_value_gradient_benchmark_generate_p8_datasets import (
    _generalized_sv_prior_mean_dataset,
    _sv_dataset,
)


PLAN = (
    "docs/plans/"
    "bayesfilter-contract-e-tp-phase6-zhao-cui-comparator-certification-plan-2026-07-15.md"
)
DEFAULT_OUTPUT_ROOT = ROOT / (
    "docs/benchmarks/artifacts/contract_e_tp_all_models_2026_07_15/"
    "phase6_zhao_cui_comparators"
)
FD_STEPS = (1e-2, 3e-3, 1e-3, 3e-4)


ROW_METADATA = {
    "actual_sv": {
        "row_id": "zhao_cui_sv_actual_nongaussian_T1000",
        "classification": "extension_or_invention",
        "transition_before_first_observation": False,
        "target_observation_policy": "exact_log_y_square_log_chi_square",
        "parameter_names": ("gamma_unconstrained", "log_beta"),
        "source_anchors": (
            "Zhao-Cui 2024 equations 15-16 and Algorithm 2",
            "third_party/audit/tensor-ssm-paper-demo/models/full_sol.m:72",
            "third_party/audit/tensor-ssm-paper-demo/models/full_sol.m:101",
            "third_party/audit/tensor-ssm-paper-demo/models/full_sol.m:124",
            "third_party/audit/tensor-ssm-paper-demo/models/sv/transition.m:1",
            "third_party/audit/tensor-ssm-paper-demo/models/sv/like.m:1",
        ),
    },
    "ksc_sv": {
        "row_id": "zhao_cui_sv_ksc_gaussian_mixture_surrogate_T1000",
        "classification": "extension_or_invention",
        "transition_before_first_observation": False,
        "target_observation_policy": "offset_log_y_square_ksc_mixture",
        "parameter_names": ("gamma_unconstrained", "log_beta"),
        "source_anchors": (
            "Zhao-Cui 2024 equations 15-16 and Algorithm 2 for parent recursion",
            "KSC observation mixture is a BayesFilter target extension",
        ),
    },
    "generalized_sv": {
        "row_id": "zhao_cui_generalized_sv_synthetic_from_estimated_values",
        "classification": "extension_or_invention",
        "transition_before_first_observation": True,
        "target_observation_policy": "raw_zero_mean_normal_generalized_sv",
        "parameter_names": ("gamma_unconstrained", "log_tau", "mu_over_tau"),
        "source_anchors": (
            "Zhao-Cui 2024 equations 15-16 and Algorithm 2 for parent recursion",
            "generalized-SV model operations are not author-source operations",
        ),
    },
}


def _json_value(value: Any) -> Any:
    if isinstance(value, tf.Tensor):
        array = value.numpy()
        return array.item() if array.ndim == 0 else array.tolist()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    return value


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()


def _tensor_sha256(value: tf.Tensor) -> str:
    return hashlib.sha256(tf.io.serialize_tensor(value).numpy()).hexdigest()


def _convention() -> highdim.MeasureConvention:
    return highdim.MeasureConvention(
        density_measure=highdim.DensityMeasure.REFERENCE_MEASURE,
        mass_measure=highdim.MassMeasure.REFERENCE_MEASURE,
        reference_weight_name="omega",
    )


def _coordinate_map(half_width: float = 8.0) -> highdim.AffineCoordinateMap:
    if not math.isfinite(half_width) or half_width <= 0.0:
        raise ValueError("coordinate half-width must be finite and positive")
    return highdim.AffineCoordinateMap(
        offset=tf.constant([0.0], dtype=tf.float64),
        matrix=tf.constant([[half_width]], dtype=tf.float64),
    )


def _basis(dimension: int, degree: int) -> highdim.ProductBasis:
    return highdim.ProductBasis(
        [
            highdim.LegendreBasis1D(
                highdim.BoundedInterval(-1.0, 1.0),
                degree,
            )
            for _ in range(dimension)
        ],
        _convention(),
    )


def _filter_config(
    *,
    dimension: int,
    degree: int,
    order: int,
    rank: int,
    seed: str,
    coordinate_half_width: float,
    density_tau: float = 0.0,
    initial_cores=None,
    initialization_rule: str = "orthonormal_mode_diagonal_norm_balanced_v1",
) -> highdim.FixedBranchFilterConfig:
    product = _basis(dimension, degree)
    ranks = (1, 1) if dimension == 1 else (1, rank, 1)
    sweep_order = (0,) if dimension == 1 else (0, 1, 1, 0)
    if initial_cores is None:
        initial_cores = highdim.norm_balanced_initial_cores(product, ranks)
    row_count = order**dimension
    return highdim.FixedBranchFilterConfig(
        fit_config=highdim.FixedTTFitConfig(
            ranks=ranks,
            ridge=1e-10,
            max_sweeps=2,
            sweep_order=sweep_order,
            row_budget=max(1024, row_count),
            column_budget=256,
            dense_matrix_byte_budget=16_000_000,
            normal_matrix_byte_budget=1_000_000,
            condition_number_warning=1e12,
            condition_number_veto=1e16,
            holdout_tolerance=1.0,
        ),
        density_tau=density_tau,
        normalizer_floor=1e-14,
        denominator_floor=1e-14,
        retained_storage_byte_budget=10_000_000,
        coordinate_maps=(_coordinate_map(coordinate_half_width),),
        measure_convention=_convention(),
        deterministic_seed=seed,
        product_basis=product,
        initial_cores=initial_cores,
        fit_quadrature_order=order,
        initialization_rule=initialization_rule,
    )


def _comparator_config(
    *,
    degree: int,
    order: int,
    rank: int,
    seed: str,
    transition_before_first_observation: bool,
    coordinate_half_width: float,
    density_tau: float = 0.0,
    initial_cores=None,
    adjacent_initial_cores=None,
    initialization_rule: str = "orthonormal_mode_diagonal_norm_balanced_v1",
) -> highdim.ScalarAdjacentTTConfig:
    return highdim.ScalarAdjacentTTConfig(
        initial=_filter_config(
            dimension=1,
            degree=degree,
            order=order,
            rank=1,
            seed=f"{seed}:initial",
            coordinate_half_width=coordinate_half_width,
            density_tau=density_tau,
            initial_cores=initial_cores,
            initialization_rule=initialization_rule,
        ),
        adjacent=_filter_config(
            dimension=2,
            degree=degree,
            order=order,
            rank=rank,
            seed=f"{seed}:adjacent",
            coordinate_half_width=coordinate_half_width,
            density_tau=density_tau,
            initial_cores=adjacent_initial_cores,
            initialization_rule=initialization_rule,
        ),
        scalar_coordinate_map=_coordinate_map(coordinate_half_width),
        transition_before_first_observation=transition_before_first_observation,
        initializer_id=initialization_rule,
    )


def _ukf_initial_cores(
    *,
    model: object,
    theta: tf.Tensor,
    raw_observations: tf.Tensor,
    degree: int,
    order: int,
    rank: int,
    coordinate_half_width: float,
) -> tuple[tuple[object, ...], tuple[object, ...], Mapping[str, object]]:
    """Build SVX-ZC initial and adjacent cores through the existing UKF route."""

    parameters = model.physical_parameters(theta)
    ukf_result = highdim.actual_transformed_sv_independent_panel_augmented_noise_ukf_filter(
        raw_observations,
        gamma=parameters["gamma"],
        beta=parameters["beta"],
        sigma=parameters["sigma"],
    )
    scout = ukf_scout_result_from_paths(
        ukf_result.mean_path,
        ukf_result.covariance_path,
        sigma_point_count=5,
        source="actual_transformed_sv_augmented_noise_gaussian_closure_ukf",
    )
    convention = _convention()
    initial_product = _basis(1, degree)
    adjacent_product = _basis(2, degree)
    projection_order = max(int(order), 2 * int(degree) + 4)
    initial_config = p76.P76UKFInitializerConfig(
        product_basis=initial_product,
        ranks=(1, 1),
        time_index=0,
        quadrature_order=projection_order,
    )
    adjacent_config = p76.P76UKFInitializerConfig(
        product_basis=adjacent_product,
        ranks=(1, int(rank), 1),
        time_index=1,
        quadrature_order=projection_order,
    )
    del convention
    initial_result = p76.p76_build_ukf_initializer(
        scout,
        initial_config,
        reference_offset=tf.zeros([1], dtype=tf.float64),
        reference_matrix=tf.constant([[coordinate_half_width]], dtype=tf.float64),
    )
    adjacent_result = p76.p76_build_ukf_initializer(
        scout,
        adjacent_config,
        reference_offset=tf.zeros([2], dtype=tf.float64),
        reference_matrix=tf.eye(2, dtype=tf.float64) * coordinate_half_width,
    )
    manifest = {
        "initializer_rule": p76.P76_UKF_INITIALIZER_RULE,
        "initializer_default": True,
        "claim_class": p76.P52_UKF_SCOUT_CLAIM if hasattr(p76, "P52_UKF_SCOUT_CLAIM") else "scout_not_truth",
        "ukf_source": "actual_transformed_sv_augmented_noise_gaussian_closure_ukf",
        "ukf_target_nonclaim": "not exact transformed same-target admission",
        "projection_order": projection_order,
        "initial": dict(initial_result.manifest),
        "adjacent": dict(adjacent_result.manifest),
        "initial_core_hash": _tensor_sha256(tf.concat([core.values for core in initial_result.cores], axis=0)),
        "adjacent_core_hash": _tensor_sha256(tf.concat([tf.reshape(core.values, [-1]) for core in adjacent_result.cores], axis=0)),
        "ukf_mean_path": _json_value(scout.mean_path),
        "ukf_covariance_path": _json_value(scout.covariance_path),
        "ukf_diagnostics": dict(ukf_result.diagnostics),
    }
    return initial_result.cores, adjacent_result.cores, manifest


def _row_inputs(
    row: str,
    horizon: int,
    target_preparation: Path | None = None,
) -> tuple[object, tf.Tensor, tf.Tensor]:
    if target_preparation is not None:
        preparation = json.loads(target_preparation.read_text(encoding="utf-8"))
        metadata = ROW_METADATA[row]
        target = preparation["target"]
        if preparation.get("row_id") != metadata["row_id"]:
            raise ValueError("target preparation row mismatch")
        if int(target["time_steps"]) != horizon:
            raise ValueError("target preparation horizon mismatch")
        if target["target_observation_policy"] != metadata["target_observation_policy"]:
            raise ValueError("target preparation observation policy mismatch")
        if "target_observations" not in target:
            raise ValueError("target preparation does not embed observations")
        theta = tf.convert_to_tensor(target["theta"], dtype=tf.float64)
        observations = tf.convert_to_tensor(
            target["target_observations"], dtype=tf.float64
        )
        if _tensor_sha256(observations) != target["target_observations_sha256"]:
            raise ValueError("target preparation observation hash mismatch")
        spec = tp_scalar.make_scalar_sv_spec(metadata["row_id"])
        return spec.model, theta, observations
    if row in ("actual_sv", "ksc_sv"):
        dataset = _sv_dataset(81101)
        theta = tf.convert_to_tensor(dataset["truth_theta"], dtype=tf.float64)
        raw = tf.convert_to_tensor(dataset["observations"], dtype=tf.float64)
        row_id = ROW_METADATA[row]["row_id"]
        spec = tp_scalar.make_scalar_sv_spec(row_id)
        model = spec.model
        observations, _flow_observations = tp_scalar.target_and_flow_observations(
            spec, raw
        )
        return model, theta, observations[:horizon]
    if row == "generalized_sv":
        dataset = _generalized_sv_prior_mean_dataset(81105)
        theta = tf.convert_to_tensor(dataset["truth_theta"], dtype=tf.float64)
        raw = tf.convert_to_tensor(dataset["observations"], dtype=tf.float64)
        spec = tp_scalar.make_scalar_sv_spec(ROW_METADATA[row]["row_id"])
        model = spec.model
        observations, _flow_observations = tp_scalar.target_and_flow_observations(
            spec, raw
        )
        return model, theta, observations[:horizon]
    raise ValueError(f"unknown scalar row: {row}")


def _fd_payload(
    result: highdim.FixedBranchScoreResult,
    parameter_names: tuple[str, ...],
) -> dict[str, Any]:
    rows = []
    grouped: dict[int, list[Any]] = {}
    for row in result.finite_difference_table.rows:
        grouped.setdefault(row.parameter_index, []).append(row)
        rows.append(
            {
                "parameter_index": row.parameter_index,
                "parameter": parameter_names[row.parameter_index],
                "h": float(row.h.numpy()),
                "score": float(row.analytic_gradient.numpy()),
                "finite_difference": float(row.centered_difference.numpy()),
                "absolute_error": float(row.abs_error.numpy()),
                "legacy_row_relative_error": float(row.rel_error.numpy()),
                "branch_hash_base": row.branch_hash_base,
                "branch_hash_plus": row.branch_hash_plus,
                "branch_hash_minus": row.branch_hash_minus,
                "row_status": row.row_status.value,
            }
        )

    p = len(parameter_names)
    threshold = 0.05 * math.sqrt(p)
    stable_windows = []
    selected_fd = []
    for parameter_index, parameter_name in enumerate(parameter_names):
        parameter_rows = sorted(
            grouped.get(parameter_index, []),
            key=lambda row: float(row.h.numpy()),
            reverse=True,
        )
        relative = []
        for row in parameter_rows:
            score = float(row.analytic_gradient.numpy())
            finite_difference = float(row.centered_difference.numpy())
            error = abs(score - finite_difference) / max(
                abs(score),
                abs(finite_difference),
                1e-12,
            )
            relative.append((row, error))
        windows = []
        for (left, left_error), (right, right_error) in zip(
            relative[:-1],
            relative[1:],
        ):
            compatible = (
                left.row_status.value == "VALID"
                and right.row_status.value == "VALID"
                and left_error <= threshold
                and right_error <= threshold
            )
            shape_ok = (
                right_error <= left_error
                or abs(right_error - left_error) <= 0.1 * threshold
            )
            windows.append(
                {
                    "h_large": float(left.h.numpy()),
                    "h_small": float(right.h.numpy()),
                    "relative_error_large": left_error,
                    "relative_error_small": right_error,
                    "pass": compatible and shape_ok,
                }
            )
        passing = [window for window in windows if window["pass"]]
        stable_windows.append(
            {
                "parameter": parameter_name,
                "threshold": threshold,
                "windows": windows,
                "status": "pass" if passing else "fail",
            }
        )
        if passing:
            selected_h = passing[-1]["h_small"]
            selected = next(
                row
                for row in parameter_rows
                if math.isclose(float(row.h.numpy()), selected_h)
            )
        else:
            selected = min(
                parameter_rows,
                key=lambda row: abs(
                    float(row.analytic_gradient.numpy())
                    - float(row.centered_difference.numpy())
                ),
            )
        selected_fd.append(float(selected.centered_difference.numpy()))

    policy = evaluate_ledh_fd_policy(
        [float(value) for value in result.score.numpy()],
        selected_fd,
        parameter_names,
    )
    stable_pass = all(entry["status"] == "pass" for entry in stable_windows)
    return {
        "steps": list(FD_STEPS),
        "rows": rows,
        "stable_windows": stable_windows,
        "selected_fd_for_policy": selected_fd,
        "owner_fd_only_policy": policy,
        "status": "pass" if stable_pass and policy["status"] == "pass" else "fail",
    }


def _fit_diagnostics(result: highdim.FixedBranchScoreResult) -> dict[str, Any]:
    return {
        "log_increments": _json_value(result.diagnostics["log_increments"]),
        "compatibility_hash": result.diagnostics["compatibility_hash"],
        "score_backend": result.diagnostics["score_backend"],
        "previous_marginal_derivative_included": result.diagnostics[
            "previous_marginal_derivative_included"
        ],
        "steps": _json_value(result.diagnostics["step_evidence"]),
    }


def _run(args: argparse.Namespace) -> dict[str, Any]:
    metadata = ROW_METADATA[args.row]
    target_preparation = args.target_preparation
    if target_preparation is not None and not target_preparation.is_absolute():
        target_preparation = ROOT / target_preparation
    model, theta, observations = _row_inputs(
        args.row, args.horizon, target_preparation
    )
    raw_observations = None
    ukf_manifest = None
    initial_cores = None
    adjacent_initial_cores = None
    initialization_rule = "orthonormal_mode_diagonal_norm_balanced_v1"
    if args.row == "actual_sv" and target_preparation is None:
        raw_observations = tf.convert_to_tensor(_sv_dataset(81101)["observations"][: args.horizon], dtype=tf.float64)
        initial_cores, adjacent_initial_cores, ukf_manifest = _ukf_initial_cores(
            model=model,
            theta=theta,
            raw_observations=raw_observations,
            degree=args.degree,
            order=args.order,
            rank=args.rank,
            coordinate_half_width=args.coordinate_half_width,
        )
        initialization_rule = p76.P76_UKF_INITIALIZER_RULE
    effective_density_tau = (
        0.0 if args.row == "actual_sv" and target_preparation is None else args.density_tau
    )
    config = _comparator_config(
        degree=args.degree,
        order=args.order,
        rank=args.rank,
        seed=args.seed,
        transition_before_first_observation=bool(
            metadata["transition_before_first_observation"]
        ),
        coordinate_half_width=args.coordinate_half_width,
        density_tau=effective_density_tau,
        initial_cores=initial_cores,
        adjacent_initial_cores=adjacent_initial_cores,
        initialization_rule=initialization_rule,
    )
    started = time.perf_counter()
    result = highdim.scalar_adjacent_state_fixed_tt_score(
        model,
        theta,
        observations,
        config,
        finite_difference_h=FD_STEPS,
        fixture_id=(
            f"contract-e-tp.phase6.{args.row}.t{args.horizon}."
            f"degree{args.degree}.order{args.order}.rank{args.rank}.v1"
        ),
        branch_seed_prefix=args.seed,
    )
    wall_time = time.perf_counter() - started
    fd = _fd_payload(result, metadata["parameter_names"])
    step_evidence = tuple(result.diagnostics["step_evidence"])
    first_step = step_evidence[0]
    if bool(metadata["transition_before_first_observation"]):
        first_step_time_order_valid = bool(
            int(first_step["fit_dimension"]) == 2
            and tuple(first_step["axis_order"]) == ("x_t", "x_t_minus_1")
            and tuple(first_step["integrated_axes"]) == (1,)
            and first_step["target_kind"]
            == "transitioned_initial_adjacent_state_update"
        )
    else:
        first_step_time_order_valid = bool(
            int(first_step["fit_dimension"]) == 1
            and tuple(first_step["axis_order"]) == ("x_0",)
            and tuple(first_step["integrated_axes"]) == ()
            and first_step["target_kind"] == "initial_state_observation"
        )
    adjacent_steps = step_evidence[1:]
    previous_state_axis_present = all(
        int(step["fit_dimension"]) == 2
        and tuple(step["axis_order"]) == ("x_t", "x_t_minus_1")
        and tuple(step["integrated_axes"]) == (1,)
        for step in adjacent_steps
    )
    float64_epsilon = float(tf.experimental.numpy.finfo(tf.float64.as_numpy_dtype).eps)
    marginal_mass_tolerance = (
        256.0
        * float64_epsilon
        * max(1, 65 * args.degree * args.rank)
    )
    marginal_mass_errors = tuple(
        abs(float(tf.convert_to_tensor(step["marginal_mass"]).numpy()) - 1.0)
        for step in step_evidence
    )
    marginal_mass_valid = all(
        error <= marginal_mass_tolerance for error in marginal_mass_errors
    )
    finite_value_and_score = bool(
        math.isfinite(float(result.log_likelihood.numpy()))
        and all(math.isfinite(float(value)) for value in result.score.numpy())
    )
    engineering_pass = bool(
        finite_value_and_score
        and fd["status"] == "pass"
        and first_step_time_order_valid
        and previous_state_axis_present
        and marginal_mass_valid
    )
    status = (
        "certified_extension_or_invention"
        if engineering_pass
        else "blocked_phase6_engineering_gate_failure"
    )
    return {
        "schema_version": "contract_e_tp.phase6.zhao_cui_comparator.v1",
        "metadata_date": "2026-07-15",
        "status": status,
        "row": args.row,
        "row_id": metadata["row_id"],
        "route_id": highdim.ZHAO_CUI_FIXED_ADJACENT_ROUTE_ID,
        "route_classification": metadata["classification"],
        "route_subtype": highdim.ZHAO_CUI_FIXED_ADJACENT_ROUTE_SUBTYPE,
        "source_anchors": metadata["source_anchors"],
        "horizon": args.horizon,
        "theta": _json_value(theta),
        "parameter_names": metadata["parameter_names"],
        "target": {
            "time_steps": args.horizon,
            "target_observations_sha256": _tensor_sha256(observations),
            "target_observation_policy": metadata["target_observation_policy"],
            "transition_before_first_observation": bool(
                metadata["transition_before_first_observation"]
            ),
            "preparation": (
                {
                    "path": str(target_preparation.relative_to(ROOT)),
                    "sha256": hashlib.sha256(target_preparation.read_bytes()).hexdigest(),
                }
                if target_preparation is not None
                else None
            ),
        },
        "value": float(result.log_likelihood.numpy()),
        "score": _json_value(result.score),
        "own_scalar_fd": fd,
        "finite_program": _fit_diagnostics(result),
        "config": {
            "degree": args.degree,
            "quadrature_order_per_axis": args.order,
            "adjacent_rank": args.rank,
            "ridge": 1e-10,
            "max_sweeps": 2,
            "axis_order": highdim.ZHAO_CUI_FIXED_ADJACENT_AXIS_ORDER,
            "density_tau": effective_density_tau,
            "initializer_id": initialization_rule,
            "parameter_treatment": "theta_external_fixed_query_not_tt_coordinate",
            "coordinate_half_width": args.coordinate_half_width,
            "transition_before_first_observation": bool(
                metadata["transition_before_first_observation"]
            ),
            "status": "ukf_initializer_default" if initialization_rule == p76.P76_UKF_INITIALIZER_RULE else "historical_comparator",
            "ukf_initializer": ukf_manifest,
        },
        "run_manifest": {
            "git_commit": _git_commit(),
            "command": (
                "CUDA_VISIBLE_DEVICES=-1 "
                "MPLCONFIGDIR=/tmp/matplotlib-contract-e-tp-phase6 python "
                + " ".join(sys.argv)
            ),
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV", "not_detected"),
            "cpu_gpu_status": "CPU-only; CUDA_VISIBLE_DEVICES=-1",
            "jit_compile": False,
            "jit_status": "explicit_nondefault_cpu_reference_diagnostic",
            "seed": args.seed,
            "data_seed": 81101 if args.row != "generalized_sv" else 81105,
            "wall_time_seconds": wall_time,
            "plan": PLAN,
        },
        "hard_vetoes": {
            "finite_value_and_score": finite_value_and_score,
            "own_scalar_fd": fd["status"],
            "first_step_time_order_valid": first_step_time_order_valid,
            "transition_before_first_observation": bool(
                metadata["transition_before_first_observation"]
            ),
            "previous_state_axis_present_from_t1": previous_state_axis_present,
            "carried_marginal_mass_valid": marginal_mass_valid,
            "carried_marginal_mass_errors": marginal_mass_errors,
            "carried_marginal_mass_tolerance": marginal_mass_tolerance,
            "carried_marginal_mass_tolerance_rule": (
                "256 * float64_epsilon * max(1, 65 * degree * adjacent_rank)"
            ),
            "forbidden_retained_grid_route_used": False,
            "oracle_alias_used": False,
        },
        "what_is_not_concluded": (
            "not adaptive TT-cross or TTSIRT reproduction",
            "not source-faithful",
            "not exact filtering",
            "not cross-method equivalence",
            "not superiority or ranking",
            "not HMC, default, leaderboard, GPU, or full-horizon readiness",
        ),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--row",
        choices=tuple(ROW_METADATA),
        required=True,
    )
    parser.add_argument("--horizon", type=int, choices=(1, 2, 10), required=True)
    parser.add_argument("--degree", type=int, default=8)
    parser.add_argument("--order", type=int, default=17)
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--coordinate-half-width", type=float, default=8.0)
    parser.add_argument(
        "--density-tau",
        type=float,
        default=0.0,
        help="frozen defensive mass; zero is retained only for historical diagnostics",
    )
    parser.add_argument("--seed", default="contract-e-tp-phase6-zhaocui-fixed")
    parser.add_argument("--target-preparation", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    artifact = _run(args)
    output = args.output
    if output is None:
        output = DEFAULT_OUTPUT_ROOT / (
            f"{args.row}_t{args.horizon}_degree{args.degree}_order{args.order}_"
            f"rank{args.rank}_result.json"
        )
    output = output if output.is_absolute() else ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {output}")
    artifact["run_manifest"]["output_artifact"] = str(output.relative_to(ROOT))
    output.write_text(
        json.dumps(_json_value(artifact), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": artifact["status"], "output": str(output)}))


if __name__ == "__main__":
    main()
