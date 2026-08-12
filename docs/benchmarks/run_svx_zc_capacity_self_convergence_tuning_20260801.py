#!/usr/bin/env python3
"""Run bounded SVX-ZC likelihood-value capacity self-convergence tuning."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("MPLCONFIGDIR", "/tmp")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tensorflow as tf

from bayesfilter.highdim.capacity_tuning import (
    SignificantPlacePolicy,
    assert_frozen_scope_equal,
    compare_likelihood_values,
    nominate_capacity,
)
from bayesfilter.highdim.filtering import legendre_gauss_nodes_weights
from bayesfilter.highdim.zhao_cui_fixed_adjacent_tt_tf import (
    ROUTE_CLASSIFICATION,
    ROUTE_ID,
    scalar_adjacent_state_fixed_tt_value,
)
import docs.benchmarks.run_contract_e_tp_phase6_zhao_cui_comparator as comparator


PLAN = "docs/plans/bayesfilter-svx-zc-capacity-self-convergence-tuning-execution-plan-2026-08-01.md"
RESULT_NOTE = "docs/plans/bayesfilter-svx-zc-capacity-self-convergence-tuning-result-2026-08-01.md"
ROW = "actual_sv"
DATA_SEED = 81101
FULL_HORIZON = 10
SMOKE_HORIZON = 3
DEGREES = (4, 6, 8, 10, 12)
RANKS = (2, 4, 6, 8)
CALIBRATION_ORDER = 25
ORDER_CONFIRMATIONS = (29, 33)
COORDINATE_HALF_WIDTH = 8.0
MASS_TOLERANCE = 1.0e-10
DENSITY_TOLERANCE = -1.0e-14
CONDITION_TOLERANCE = 1.0e10
DENSITY_QUERY_ORDER = 9
FULL_CELL_BUDGET = 35
FULL_WALL_TIME_BUDGET_SECONDS = 30.0 * 60.0
POLICY = SignificantPlacePolicy()
VALIDATION_OFFSETS = (
    (-0.05, 0.0),
    (0.05, 0.0),
    (0.0, -0.05),
    (0.0, 0.05),
)


class CampaignBudgetExceeded(RuntimeError):
    """Raised before an evaluation that would exceed a campaign limit."""


def _tensor_python(value: tf.Tensor) -> Any:
    tensor = tf.convert_to_tensor(value)
    if tensor.shape.rank == 0:
        if tensor.dtype == tf.bool:
            return bool(tensor)
        if tensor.dtype.is_integer:
            return int(tensor)
        if tensor.dtype.is_floating:
            scalar = float(tensor)
            return scalar if math.isfinite(scalar) else str(scalar)
        if tensor.dtype == tf.string:
            raw = bytes(tensor).decode("utf-8")
            return raw
        return str(tensor)
    if tensor.shape.rank is None:
        raise ValueError("artifact tensors require a statically known rank")
    return [_tensor_python(item) for item in tf.unstack(tensor)]


def _json_value(value: Any) -> Any:
    if isinstance(value, tf.Tensor):
        return _tensor_python(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    enum_value = getattr(value, "value", None)
    if isinstance(enum_value, (str, int, float, bool)):
        return _json_value(enum_value)
    return str(value)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(_json_value(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256_tensor(value: tf.Tensor) -> str:
    return hashlib.sha256(tf.io.serialize_tensor(value).numpy()).hexdigest()


def _stable_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        _json_value(payload), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _git_status() -> str:
    return subprocess.check_output(
        ["git", "status", "--short"], cwd=ROOT, text=True
    ).strip()


def _as_finite_float(value: Any) -> float | None:
    if value is None or value == "unavailable":
        return None
    if isinstance(value, tf.Tensor):
        value = float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _density_query(step: Any) -> dict[str, Any]:
    basis = step.density.sqrt_tt.product_basis
    dimension = int(basis.dimension)
    nodes, _weights = legendre_gauss_nodes_weights(DENSITY_QUERY_ORDER)
    if dimension == 1:
        points = nodes[:, tf.newaxis]
    elif dimension == 2:
        grid = tf.stack(tf.meshgrid(nodes, nodes, indexing="ij"), axis=-1)
        points = tf.reshape(grid, (-1, 2))
    else:
        raise ValueError("SVX-ZC density invariant expects one or two axes")
    keep_axes = tuple(range(dimension))
    values = step.density.normalized_marginal_density_values(keep_axes, points)
    finite = bool(tf.reduce_all(tf.math.is_finite(values)))
    minimum = float(tf.reduce_min(values))
    return {
        "time_index": int(step.time_index),
        "dimension": dimension,
        "query_order": DENSITY_QUERY_ORDER,
        "query_count": int(points.shape[0]),
        "finite": finite,
        "minimum": minimum,
        "nonnegative_with_tolerance": finite and minimum >= DENSITY_TOLERANCE,
    }


def _step_diagnostics(result: Any) -> dict[str, Any]:
    mass_rows: list[dict[str, Any]] = []
    density_rows: list[dict[str, Any]] = []
    residuals: list[float] = []
    holdout_residuals: list[float] = []
    conditions: list[float] = []
    condition_available = True
    all_fit_finite = True
    for step in result.steps:
        mass_error = abs(float(step.marginal_mass) - 1.0)
        mass_rows.append(
            {
                "time_index": int(step.time_index),
                "mass": float(step.marginal_mass),
                "absolute_error": mass_error,
                "pass": mass_error <= MASS_TOLERANCE,
            }
        )
        density_rows.append(_density_query(step))
        residual = float(step.fit_result.fit_residual)
        residuals.append(residual)
        all_fit_finite = all_fit_finite and math.isfinite(residual)
        holdout = step.fit_result.holdout_residual
        if holdout is not None:
            holdout_value = float(holdout)
            holdout_residuals.append(holdout_value)
            all_fit_finite = all_fit_finite and math.isfinite(holdout_value)
        for update in step.fit_result.core_update_statuses:
            condition = _as_finite_float(update.get("condition_number"))
            if condition is None:
                condition_available = False
            else:
                conditions.append(condition)
    maximum_condition = max(conditions) if conditions else float("inf")
    finite_value = math.isfinite(float(result.log_likelihood))
    finite_increments = bool(tf.reduce_all(tf.math.is_finite(result.log_increments)))
    mass_pass = all(row["pass"] for row in mass_rows)
    density_pass = all(
        row["finite"] and row["nonnegative_with_tolerance"] for row in density_rows
    )
    condition_pass = (
        condition_available
        and math.isfinite(maximum_condition)
        and maximum_condition <= CONDITION_TOLERANCE
    )
    invariant_pass = all(
        (
            finite_value,
            finite_increments,
            all_fit_finite,
            mass_pass,
            density_pass,
            condition_pass,
        )
    )
    return {
        "invariant_pass": invariant_pass,
        "finite_value": finite_value,
        "finite_increments": finite_increments,
        "all_fit_residuals_finite": all_fit_finite,
        "mass_pass": mass_pass,
        "density_pass": density_pass,
        "condition_pass": condition_pass,
        "condition_available_for_every_update": condition_available,
        "maximum_condition_number": maximum_condition,
        "condition_tolerance": CONDITION_TOLERANCE,
        "maximum_fit_residual": max(residuals) if residuals else None,
        "fit_residuals": tuple(residuals),
        "holdout_residuals": tuple(holdout_residuals),
        "mass_rows": tuple(mass_rows),
        "density_rows": tuple(density_rows),
        "mass_tolerance": MASS_TOLERANCE,
        "density_minimum_tolerance": DENSITY_TOLERANCE,
        "fit_residual_role": "explanatory_capacity_diagnostic_not_promotion_veto",
    }


def _frozen_scope(
    *,
    model: Any,
    theta: tf.Tensor,
    observations: tf.Tensor,
    horizon: int,
    point_id: str,
) -> dict[str, Any]:
    return {
        "schema": "bayesfilter.svx_zc.capacity_tuning.frozen_scope.v1",
        "row": ROW,
        "route_id": ROUTE_ID,
        "route_classification": ROUTE_CLASSIFICATION,
        "model": _json_value(model.manifest_payload()),
        "theta": _tensor_python(theta),
        "theta_coordinate": ("gamma_unconstrained", "log_beta"),
        "point_id": point_id,
        "observation_sha256": _sha256_tensor(observations),
        "horizon": int(horizon),
        "data_seed": DATA_SEED,
        "coordinate_half_width": COORDINATE_HALF_WIDTH,
        "dtype": "float64",
        "backend": "tensorflow_cpu_diagnostic",
        "jit_compile": False,
        "density_tau": 0.0,
        "ridge": 1.0e-10,
        "max_sweeps": 2,
        "initial_sweep_order": (0,),
        "adjacent_sweep_order": (0, 1, 1, 0),
        "fit_condition_warning": 1.0e12,
        "fit_condition_veto": 1.0e16,
        "campaign_condition_tolerance": CONDITION_TOLERANCE,
        "mass_tolerance": MASS_TOLERANCE,
        "density_minimum_tolerance": DENSITY_TOLERANCE,
        "measure": "reference_measure",
        "transition_before_first_observation": False,
        "initializer_family": "repository_ukf_initializer",
    }


def _executed_noncapacity_config(config: Any) -> dict[str, Any]:
    def filter_payload(filter_config: Any) -> dict[str, Any]:
        fit = filter_config.fit_config
        if fit is None:
            raise ValueError("capacity tuning requires fixed fit configuration")
        return {
            "density_tau": float(filter_config.density_tau),
            "normalizer_floor": float(filter_config.normalizer_floor),
            "denominator_floor": float(filter_config.denominator_floor),
            "retained_storage_byte_budget": int(
                filter_config.retained_storage_byte_budget
            ),
            "coordinate_maps": _json_value(
                tuple(item.manifest_payload() for item in filter_config.coordinate_maps)
            ),
            "measure_convention": {
                "density_measure": filter_config.measure_convention.density_measure.value,
                "mass_measure": filter_config.measure_convention.mass_measure.value,
                "reference_weight_name": (
                    filter_config.measure_convention.reference_weight_name
                ),
            },
            "dtype": filter_config.dtype.name,
            "initialization_rule": filter_config.initialization_rule,
            "fit": {
                "ridge": float(fit.ridge),
                "max_sweeps": int(fit.max_sweeps),
                "sweep_order": tuple(int(value) for value in fit.sweep_order),
                "dense_matrix_byte_budget": int(fit.dense_matrix_byte_budget),
                "normal_matrix_byte_budget": int(fit.normal_matrix_byte_budget),
                "condition_number_warning": float(fit.condition_number_warning),
                "condition_number_veto": float(fit.condition_number_veto),
                "holdout_tolerance": float(fit.holdout_tolerance),
                "stabilization_policy_id": fit.stabilization_policy_id,
                "solver_backend": fit.solver_backend,
                "column_scale_floor": float(fit.column_scale_floor),
            },
        }

    return {
        "initial": filter_payload(config.initial),
        "adjacent": filter_payload(config.adjacent),
        "scalar_coordinate_map": _json_value(
            config.scalar_coordinate_map.manifest_payload()
        ),
        "transition_before_first_observation": bool(
            config.transition_before_first_observation
        ),
        "initializer_id": config.initializer_id,
    }


def _validate_executed_noncapacity_config(payload: Mapping[str, Any]) -> None:
    expected_common = {
        "density_tau": 0.0,
        "normalizer_floor": 1.0e-14,
        "denominator_floor": 1.0e-14,
        "retained_storage_byte_budget": 10_000_000,
        "coordinate_maps": [
            {
                "family": "AffineCoordinateMap",
                "offset": [0.0],
                "matrix": [[COORDINATE_HALF_WIDTH]],
            }
        ],
        "measure_convention": {
            "density_measure": "REFERENCE_MEASURE",
            "mass_measure": "REFERENCE_MEASURE",
            "reference_weight_name": "omega",
        },
        "dtype": "float64",
        "initialization_rule": payload["initializer_id"],
    }
    expected_fit_common = {
        "ridge": 1.0e-10,
        "max_sweeps": 2,
        "dense_matrix_byte_budget": 16_000_000,
        "normal_matrix_byte_budget": 1_000_000,
        "condition_number_warning": 1.0e12,
        "condition_number_veto": 1.0e16,
        "holdout_tolerance": 1.0,
        "stabilization_policy_id": "objective_preserving_column_scaled_augmented_ridge_v1",
        "solver_backend": "tensorflow.linalg.lstsq(fast=False)",
    }
    for lane, sweep_order in (("initial", (0,)), ("adjacent", (0, 1, 1, 0))):
        actual = dict(payload[lane])
        fit = dict(actual.pop("fit"))
        if actual != expected_common:
            raise ValueError(f"executed {lane} non-capacity configuration drifted")
        column_scale_floor = float(fit.pop("column_scale_floor"))
        if not math.isfinite(column_scale_floor) or column_scale_floor <= 0.0:
            raise ValueError("column_scale_floor must remain finite and positive")
        if tuple(fit.pop("sweep_order")) != sweep_order:
            raise ValueError(f"executed {lane} sweep order drifted")
        if fit != expected_fit_common:
            raise ValueError(f"executed {lane} fit policy drifted")
    expected_map = {
        "family": "AffineCoordinateMap",
        "offset": [0.0],
        "matrix": [[COORDINATE_HALF_WIDTH]],
    }
    if payload["scalar_coordinate_map"] != expected_map:
        raise ValueError("executed scalar coordinate map drifted")
    if payload["transition_before_first_observation"] is not False:
        raise ValueError("executed transition timing drifted")


def _cell_filename(
    *, point_id: str, degree: int, rank: int, order: int
) -> str:
    safe_point = point_id.replace("+", "plus").replace("-", "minus")
    return f"cell-{safe_point}-d{degree}-r{rank}-o{order}.json"


def _run_cell(
    *,
    output_root: Path,
    model: Any,
    theta: tf.Tensor,
    observations: tf.Tensor,
    raw_observations: tf.Tensor,
    horizon: int,
    point_id: str,
    degree: int,
    rank: int,
    order: int,
    deadline: float,
    cell_counter: list[int],
    cell_budget: int,
) -> dict[str, Any]:
    if time.perf_counter() >= deadline:
        raise CampaignBudgetExceeded("wall-time budget exhausted before next cell")
    if cell_counter[0] >= cell_budget:
        raise CampaignBudgetExceeded("cell budget exhausted before next cell")
    cell_counter[0] += 1
    started = time.perf_counter()
    seed = f"svx-zc-capacity-20260801:{point_id}:d{degree}:r{rank}:o{order}"
    frozen_scope = _frozen_scope(
        model=model,
        theta=theta,
        observations=observations,
        horizon=horizon,
        point_id=point_id,
    )
    path = output_root / _cell_filename(
        point_id=point_id, degree=degree, rank=rank, order=order
    )
    try:
        initial_cores, adjacent_cores, ukf_manifest = comparator._ukf_initial_cores(
            model=model,
            theta=theta,
            raw_observations=raw_observations,
            degree=degree,
            order=order,
            rank=rank,
            coordinate_half_width=COORDINATE_HALF_WIDTH,
        )
        config = comparator._comparator_config(
            degree=degree,
            order=order,
            rank=rank,
            seed=seed,
            transition_before_first_observation=False,
            coordinate_half_width=COORDINATE_HALF_WIDTH,
            density_tau=0.0,
            initial_cores=initial_cores,
            adjacent_initial_cores=adjacent_cores,
            initialization_rule=str(ukf_manifest["initializer_rule"]),
        )
        executed_noncapacity_config = _executed_noncapacity_config(config)
        _validate_executed_noncapacity_config(executed_noncapacity_config)
        frozen_scope["executed_noncapacity_config"] = executed_noncapacity_config
        result = scalar_adjacent_state_fixed_tt_value(
            model,
            theta,
            observations,
            config,
            fixture_id=f"svx-zc-capacity.{point_id}.d{degree}.r{rank}.o{order}",
            branch_seed_prefix=seed,
        )
        diagnostics = _step_diagnostics(result)
        record = {
            "schema": "bayesfilter.svx_zc.capacity_tuning.cell.v1",
            "execution_status": "completed",
            "point_id": point_id,
            "capacity": {"degree": degree, "rank": rank, "order": order},
            "value": float(result.log_likelihood),
            "increments": _tensor_python(result.log_increments),
            "invariant_pass": diagnostics["invariant_pass"],
            "diagnostics": diagnostics,
            "frozen_scope": frozen_scope,
            "frozen_scope_sha256": _stable_hash(frozen_scope),
            "capacity_manifest": {
                "degree": degree,
                "rank": rank,
                "quadrature_order": order,
                "seed": seed,
                "ukf_projection_order": ukf_manifest["projection_order"],
                "initial_core_hash": ukf_manifest["initial_core_hash"],
                "adjacent_core_hash": ukf_manifest["adjacent_core_hash"],
                "config": _json_value(config.manifest_payload()),
            },
            "ukf_initializer": _json_value(ukf_manifest),
            "branch_identity_sha256": result.branch_identity.hash.value,
            "compatibility_hash": result.compatibility_hash,
            "wall_time_seconds": time.perf_counter() - started,
        }
    except ValueError as exc:
        record = {
            "schema": "bayesfilter.svx_zc.capacity_tuning.cell.v1",
            "execution_status": "numerically_invalid",
            "point_id": point_id,
            "capacity": {"degree": degree, "rank": rank, "order": order},
            "value": None,
            "increments": (),
            "invariant_pass": False,
            "frozen_scope": frozen_scope,
            "frozen_scope_sha256": _stable_hash(frozen_scope),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "wall_time_seconds": time.perf_counter() - started,
        }
    _write_json(path, record)
    return record


def _calibration_cells() -> tuple[tuple[int, int], ...]:
    return tuple(
        (degree, rank)
        for degree in DEGREES
        for rank in RANKS
        if rank <= degree + 1
    )


def _validation_points(center: tf.Tensor) -> tuple[tuple[str, tf.Tensor], ...]:
    rows = []
    for index, offset in enumerate(VALIDATION_OFFSETS):
        theta = center + tf.constant(offset, dtype=tf.float64)
        rows.append((f"validation{index + 1}", theta))
    return tuple(rows)


def _comparison(
    low: Mapping[str, Any], high: Mapping[str, Any]
) -> dict[str, Any]:
    assert_frozen_scope_equal(low["frozen_scope"], high["frozen_scope"])
    return compare_likelihood_values(
        low_value=float(low["value"]),
        high_value=float(high["value"]),
        low_increments=low["increments"],
        high_increments=high["increments"],
        policy=POLICY,
    )


def _manifest(
    *, output_root: Path, horizon: int, smoke: bool, started: float
) -> dict[str, Any]:
    return {
        "schema": "bayesfilter.svx_zc.capacity_tuning.run_manifest.v1",
        "git_commit": _git_commit(),
        "git_status_short": _git_status(),
        "command": " ".join(sys.argv),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "tensorflow_version": tf.__version__,
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV", "not_detected"),
        "device": "CPU-only; CUDA_VISIBLE_DEVICES=-1",
        "gpu_devices_intentionally_hidden": True,
        "dtype": "float64",
        "jit_compile": False,
        "mode": "smoke" if smoke else "full",
        "data_seed": DATA_SEED,
        "horizon": horizon,
        "policy": POLICY.manifest_payload(),
        "degrees": DEGREES,
        "ranks": RANKS,
        "calibration_order": CALIBRATION_ORDER,
        "order_confirmations": ORDER_CONFIRMATIONS,
        "cell_budget": 2 if smoke else FULL_CELL_BUDGET,
        "wall_time_budget_seconds": (
            None if smoke else FULL_WALL_TIME_BUDGET_SECONDS
        ),
        "plan": PLAN,
        "result_note": RESULT_NOTE,
        "output_root": str(output_root.relative_to(ROOT)),
        "started_monotonic_seconds": started,
        "trust_basis": "deliberate_cpu_only_diagnostic_exception",
    }


def run(output_root: Path, *, smoke: bool) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite existing root: {output_root}")
    output_root.mkdir(parents=True)
    started = time.perf_counter()
    horizon = SMOKE_HORIZON if smoke else FULL_HORIZON
    deadline = (
        float("inf") if smoke else started + FULL_WALL_TIME_BUDGET_SECONDS
    )
    cell_budget = 2 if smoke else FULL_CELL_BUDGET
    cell_counter = [0]
    manifest = _manifest(
        output_root=output_root, horizon=horizon, smoke=smoke, started=started
    )
    _write_json(output_root / "run_manifest.json", manifest)

    model, center, observations = comparator._row_inputs(ROW, horizon)
    dataset = comparator._sv_dataset(DATA_SEED)
    raw_observations = tf.convert_to_tensor(
        dataset["observations"], dtype=tf.float64
    )[:horizon]
    records: dict[str, dict[str, Any]] = {}
    status = "HARNESS_INVALID"
    decision_phase = "setup"
    nomination: dict[str, Any] | None = None
    order_comparisons: dict[str, Any] = {}
    validation_results: dict[str, Any] = {}
    budget_error: str | None = None

    try:
        if smoke:
            for degree, rank in ((4, 2), (4, 4)):
                record = _run_cell(
                    output_root=output_root,
                    model=model,
                    theta=center,
                    observations=observations,
                    raw_observations=raw_observations,
                    horizon=horizon,
                    point_id="smoke",
                    degree=degree,
                    rank=rank,
                    order=CALIBRATION_ORDER,
                    deadline=deadline,
                    cell_counter=cell_counter,
                    cell_budget=cell_budget,
                )
                records[f"smoke:d{degree}:r{rank}:o{CALIBRATION_ORDER}"] = record
            status = (
                "SMOKE_COMPLETE"
                if all(row["invariant_pass"] for row in records.values())
                else "SMOKE_NUMERICALLY_INVALID"
            )
            decision_phase = "smoke"
        else:
            decision_phase = "calibration"
            calibration: dict[tuple[int, int], dict[str, Any]] = {}
            for degree, rank in _calibration_cells():
                record = _run_cell(
                    output_root=output_root,
                    model=model,
                    theta=center,
                    observations=observations,
                    raw_observations=raw_observations,
                    horizon=horizon,
                    point_id="calibration",
                    degree=degree,
                    rank=rank,
                    order=CALIBRATION_ORDER,
                    deadline=deadline,
                    cell_counter=cell_counter,
                    cell_budget=cell_budget,
                )
                calibration[(degree, rank)] = record
                records[f"calibration:d{degree}:r{rank}:o{CALIBRATION_ORDER}"] = record
            nomination = nominate_capacity(
                calibration, degrees=DEGREES, ranks=RANKS, policy=POLICY
            )
            if nomination["status"] != "nominated":
                any_invalid = any(
                    not bool(row["invariant_pass"]) for row in calibration.values()
                )
                axis_summary = nomination["axis_summary"]
                if any_invalid:
                    status = "NUMERICALLY_INVALID"
                elif (
                    axis_summary["rank_all_stable"]
                    and not axis_summary["degree_all_stable"]
                ):
                    status = "UNDER_RESOLVED_DEGREE"
                elif (
                    axis_summary["degree_all_stable"]
                    and not axis_summary["rank_all_stable"]
                ):
                    status = "UNDER_RESOLVED_RANK"
                else:
                    status = "UNDER_RESOLVED_DEGREE_AND_OR_RANK"
            else:
                nominee = nomination["nominee"]
                degree = int(nominee["degree"])
                rank = int(nominee["rank"])
                decision_phase = "quadrature_confirmation"
                order_records = {CALIBRATION_ORDER: calibration[(degree, rank)]}
                for order in ORDER_CONFIRMATIONS:
                    record = _run_cell(
                        output_root=output_root,
                        model=model,
                        theta=center,
                        observations=observations,
                        raw_observations=raw_observations,
                        horizon=horizon,
                        point_id="calibration",
                        degree=degree,
                        rank=rank,
                        order=order,
                        deadline=deadline,
                        cell_counter=cell_counter,
                        cell_budget=cell_budget,
                    )
                    order_records[order] = record
                    records[f"calibration:d{degree}:r{rank}:o{order}"] = record
                if not all(row["invariant_pass"] for row in order_records.values()):
                    status = "NUMERICALLY_INVALID"
                else:
                    order_comparisons = {
                        "25_to_29": _comparison(order_records[25], order_records[29]),
                        "29_to_33": _comparison(order_records[29], order_records[33]),
                    }
                    if not all(row["stable"] for row in order_comparisons.values()):
                        status = "UNDER_RESOLVED_ORDER"
                    else:
                        decision_phase = "validation"
                        validation_pass = True
                        validation_numeric_pass = True
                        higher_degree = int(nominee["degree_neighbor"])
                        higher_rank = int(nominee["rank_neighbor"])
                        for point_id, theta in _validation_points(center):
                            point_cells: dict[tuple[int, int, int], dict[str, Any]] = {}
                            for cell_degree, cell_rank, cell_order in (
                                (degree, rank, 25),
                                (higher_degree, rank, 25),
                                (degree, higher_rank, 25),
                                (degree, rank, 33),
                            ):
                                key = (cell_degree, cell_rank, cell_order)
                                if key in point_cells:
                                    continue
                                record = _run_cell(
                                    output_root=output_root,
                                    model=model,
                                    theta=theta,
                                    observations=observations,
                                    raw_observations=raw_observations,
                                    horizon=horizon,
                                    point_id=point_id,
                                    degree=cell_degree,
                                    rank=cell_rank,
                                    order=cell_order,
                                    deadline=deadline,
                                    cell_counter=cell_counter,
                                    cell_budget=cell_budget,
                                )
                                point_cells[key] = record
                                records[
                                    f"{point_id}:d{cell_degree}:r{cell_rank}:o{cell_order}"
                                ] = record
                            point_numeric = all(
                                row["invariant_pass"] for row in point_cells.values()
                            )
                            validation_numeric_pass = validation_numeric_pass and point_numeric
                            if point_numeric:
                                base = point_cells[(degree, rank, 25)]
                                point_comparisons = {
                                    "degree": _comparison(
                                        base, point_cells[(higher_degree, rank, 25)]
                                    ),
                                    "rank": _comparison(
                                        base, point_cells[(degree, higher_rank, 25)]
                                    ),
                                    "order": _comparison(
                                        base, point_cells[(degree, rank, 33)]
                                    ),
                                }
                                point_pass = all(
                                    row["stable"] for row in point_comparisons.values()
                                )
                            else:
                                point_comparisons = {}
                                point_pass = False
                            validation_pass = validation_pass and point_pass
                            validation_results[point_id] = {
                                "theta": _tensor_python(theta),
                                "numeric_pass": point_numeric,
                                "value_stability_pass": point_pass,
                                "comparisons": point_comparisons,
                            }
                        if not validation_numeric_pass:
                            status = "NUMERICALLY_INVALID"
                        elif not validation_pass:
                            status = "VALIDATION_FAILED"
                        else:
                            status = "SELF_CONVERGED_VALUE"
    except CampaignBudgetExceeded as exc:
        status = "CAMPAIGN_BUDGET_EXHAUSTED"
        budget_error = str(exc)

    manifest["completed_wall_time_seconds"] = time.perf_counter() - started
    manifest["executed_cell_count"] = cell_counter[0]
    manifest["terminal_status"] = status
    _write_json(output_root / "run_manifest.json", manifest)
    result_payload = {
        "schema": "bayesfilter.svx_zc.capacity_tuning.result.v1",
        "status": status,
        "decision_phase": decision_phase,
        "cell_id": "SVX-ZC",
        "route_id": ROUTE_ID,
        "mode": "smoke" if smoke else "full",
        "nomination": nomination,
        "order_comparisons": order_comparisons,
        "validation": validation_results,
        "records": records,
        "invalid_cells": tuple(
            key for key, row in records.items() if not bool(row["invariant_pass"])
        ),
        "budget_error": budget_error,
        "run_manifest": manifest,
        "decision": {
            "primary_criterion": "total_likelihood_self_convergence_at_three_significant_digits",
            "score_delta_computed": False,
            "dense_reference_required_or_computed": False,
            "fit_residual_role": "explanatory_only",
            "increment_role": "explanatory_cancellation_warning_only",
        },
        "nonclaims": (
            "not exact likelihood accuracy",
            "not score accuracy",
            "not HMC validity or convergence",
            "not posterior agreement",
            "not GPU/XLA or production readiness",
            "not transferable beyond this frozen scope",
        ),
    }
    _write_json(output_root / "result.json", result_payload)
    return result_payload


def reinterpret_preserved_run(
    source_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Recompute capacity decisions from saved cells without filter calls."""

    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite existing root: {output_root}")
    source_root = source_root.resolve()
    source_result = json.loads((source_root / "result.json").read_text(encoding="utf-8"))
    source_manifest = json.loads(
        (source_root / "run_manifest.json").read_text(encoding="utf-8")
    )
    if source_manifest.get("mode") != "full":
        raise ValueError("reinterpretation requires a full preserved run")
    records: dict[str, dict[str, Any]] = {}
    calibration: dict[tuple[int, int], dict[str, Any]] = {}
    for path in sorted(source_root.glob("cell-calibration-d*-r*-o25.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        capacity = record.get("capacity", {})
        degree = int(capacity["degree"])
        rank = int(capacity["rank"])
        if not record.get("invariant_pass", False):
            raise ValueError(f"preserved cell is not valid: {path.name}")
        calibration[(degree, rank)] = record
        records[f"calibration:d{degree}:r{rank}:o25"] = record
    if len(calibration) != 17:
        raise ValueError("preserved run does not contain all 17 calibration cells")
    nomination = nominate_capacity(
        calibration, degrees=DEGREES, ranks=RANKS, policy=POLICY
    )
    output_root.mkdir(parents=True)
    started = time.perf_counter()
    manifest = dict(source_manifest)
    manifest.update(
        {
            "schema": "bayesfilter.svx_zc.capacity_tuning.reinterpret_manifest.v1",
            "mode": "read_only_reinterpretation",
            "source_run_root": str(source_root),
            "source_run_status": source_result.get("status"),
            "command": " ".join(sys.argv),
            "policy": POLICY.manifest_payload(),
            "executed_cell_count": 0,
            "completed_wall_time_seconds": time.perf_counter() - started,
            "terminal_status": "CENTER_NOMINATION_ONLY",
            "filter_calls": 0,
            "score_calls": 0,
            "dense_reference_calls": 0,
        }
    )
    result_payload = {
        "schema": "bayesfilter.svx_zc.capacity_tuning.reinterpret_result.v1",
        "status": "CENTER_NOMINATION_ONLY",
        "decision_phase": "read_only_reinterpretation",
        "cell_id": "SVX-ZC",
        "route_id": ROUTE_ID,
        "nomination": nomination,
        "records": records,
        "source_artifact": str((source_root / "result.json").relative_to(ROOT)),
        "run_manifest": manifest,
        "decision": {
            "primary_criterion": "leading_two_of_three_significant_digits_equal",
            "score_delta_computed": False,
            "dense_reference_required_or_computed": False,
            "filter_calls": 0,
            "center_nomination_only": True,
            "quadrature_and_validation_deferred": True,
        },
        "nonclaims": (
            "not final capacity promotion",
            "not quadrature confirmation",
            "not validation-neighborhood confirmation",
            "not exact likelihood accuracy",
            "not score accuracy or HMC readiness",
        ),
    }
    _write_json(output_root / "run_manifest.json", manifest)
    _write_json(output_root / "result.json", result_payload)
    return result_payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--reinterpret-source", type=Path)
    args = parser.parse_args()
    output_root = (
        args.output_root
        if args.output_root.is_absolute()
        else ROOT / args.output_root
    )
    if args.reinterpret_source is not None:
        source_root = (
            args.reinterpret_source
            if args.reinterpret_source.is_absolute()
            else ROOT / args.reinterpret_source
        )
        result = reinterpret_preserved_run(source_root, output_root)
    else:
        result = run(output_root, smoke=bool(args.smoke))
    print(
        json.dumps(
            {
                "status": result["status"],
                "output_root": str(output_root),
                "executed_cell_count": result["run_manifest"]["executed_cell_count"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
