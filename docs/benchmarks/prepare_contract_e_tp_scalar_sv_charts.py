#!/usr/bin/env python3
"""Prepare fixed center-only Contract E--TP charts for one scalar SV row."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import numpy as np
import tensorflow as tf
from numpy.polynomial.hermite import hermgauss
from numpy.polynomial.legendre import leggauss
from scipy.optimize import Bounds, LinearConstraint, least_squares, linprog, minimize


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bayesfilter.highdim import ledh_contract_e_tp_scalar_sv_tf as model
from bayesfilter.highdim import ledh_contract_e_tp_tf as tp
from bayesfilter.highdim.ledh_forward_contract import (
    ACTUAL_SV_ROW_ID,
    GENERALIZED_SV_ROW_ID,
    KSC_SV_ROW_ID,
)
from scripts.filtering_value_gradient_benchmark_generate_p8_datasets import (
    _generalized_sv_prior_mean_dataset,
    _sv_dataset,
)


DTYPE = tf.float64
ROW_IDS = (ACTUAL_SV_ROW_ID, KSC_SV_ROW_ID, GENERALIZED_SV_ROW_ID)
PARAMETER_NAMES = {
    ACTUAL_SV_ROW_ID: ("gamma_unconstrained", "log_beta"),
    KSC_SV_ROW_ID: ("gamma_unconstrained", "log_beta"),
    GENERALIZED_SV_ROW_ID: ("gamma_unconstrained", "log_tau", "mu_over_tau"),
}


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--row-id", choices=ROW_IDS, required=True)
    parser.add_argument(
        "--time-steps", type=int, choices=(1, 2, 3, 10, 100, 1000), required=True
    )
    parser.add_argument("--teacher-order", type=int, choices=(13, 25, 41, 65), required=True)
    parser.add_argument("--continuation-order", type=int, choices=(33, 65, 129, 257), required=True)
    parser.add_argument("--continuation-radius", type=float, required=True)
    parser.add_argument("--lookahead-steps", type=int, required=True)
    parser.add_argument(
        "--feature-lookaheads",
        default=None,
        help="strictly increasing comma-separated progressive horizons",
    )
    parser.add_argument(
        "--chart-mode",
        choices=("auto", "fixed_square", "fixed_overcomplete_kkt"),
        default="auto",
    )
    parser.add_argument("--extra-anchors", type=int, default=2)
    parser.add_argument(
        "--anchor-strategy",
        choices=(
            "positive_lp_max_margin",
            "weighted_quantile_voronoi",
            "positive_basis_plus_quantile_qp",
            "positive_basis_plus_quantile_kl",
            "positive_basis_quantile_analytic_center",
        ),
        default="positive_lp_max_margin",
    )
    parser.add_argument("--anchor-count", type=int, default=None)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _row_data(row_id: str, time_steps: int) -> tuple[tf.Tensor, tf.Tensor]:
    if row_id in (ACTUAL_SV_ROW_ID, KSC_SV_ROW_ID):
        dataset = _sv_dataset(81101)
    else:
        dataset = _generalized_sv_prior_mean_dataset(81105)
    return (
        tf.constant(dataset["truth_theta"], DTYPE),
        tf.convert_to_tensor(dataset["observations"][:time_steps], DTYPE),
    )


def _hermite_rule(order: int) -> tuple[tf.Tensor, tf.Tensor]:
    nodes, weights = hermgauss(order)
    return (
        tf.constant(np.sqrt(2.0) * nodes, DTYPE),
        tf.constant(weights / np.sqrt(np.pi), DTYPE),
    )


def _legendre_rule(order: int, radius: float) -> tuple[tf.Tensor, tf.Tensor]:
    nodes, weights = leggauss(order)
    return (
        tf.constant(radius * nodes, DTYPE),
        tf.constant(radius * weights, DTYPE),
    )


def _tensor_sha256(value: tf.Tensor) -> str:
    return hashlib.sha256(tf.io.serialize_tensor(value).numpy()).hexdigest()


def main() -> None:
    args = _parse()
    started = time.perf_counter()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not np.isfinite(args.continuation_radius) or args.continuation_radius <= 0.0:
        raise ValueError("continuation radius must be finite and positive")
    if args.lookahead_steps < 1:
        raise ValueError("lookahead steps must be positive")
    if args.extra_anchors < 0:
        raise ValueError("extra anchors must be nonnegative")
    if args.anchor_count is not None and args.anchor_count < 1:
        raise ValueError("anchor count must be positive")
    if args.anchor_strategy in (
        "weighted_quantile_voronoi",
        "positive_basis_plus_quantile_qp",
        "positive_basis_plus_quantile_kl",
        "positive_basis_quantile_analytic_center",
    ):
        if args.chart_mode == "fixed_square":
            raise ValueError("weighted quantile anchors require a KKT chart")
        if args.anchor_count is None:
            raise ValueError("weighted quantile anchors require --anchor-count")
    requested_lookaheads = (
        tuple(int(item) for item in args.feature_lookaheads.split(","))
        if args.feature_lookaheads
        else (args.lookahead_steps,)
    )
    model.effective_progressive_lookaheads(requested_lookaheads, args.time_steps)
    chart_mode = args.chart_mode
    if chart_mode == "auto":
        chart_mode = (
            "fixed_overcomplete_kkt" if args.feature_lookaheads else "fixed_square"
        )
    spec = model.make_scalar_sv_spec(args.row_id)
    theta, raw_observations = _row_data(args.row_id, args.time_steps)
    target_observations, flow_observations = model.target_and_flow_observations(
        spec, raw_observations
    )
    nodes, weights = _hermite_rule(args.teacher_order)
    grid, grid_weights = _legendre_rule(
        args.continuation_order, args.continuation_radius
    )
    parents, parent_log_weights, standard_nodes, standard_log_weights = (
        model.initial_rule(spec, theta, nodes, weights)
    )
    active_indices: list[list[int]] = []
    row_scales: list[list[float]] = []
    reference_weights: list[list[float]] = []
    precisions: list[list[list[float]]] = []
    realized_lookaheads: list[list[int]] = []
    charts: list[dict[str, object]] = []
    for time_index in range(args.time_steps - 1):
        teacher = model._teacher_step(
            spec,
            theta,
            parents,
            parent_log_weights,
            standard_nodes,
            standard_log_weights,
            target_observations[time_index],
            flow_observations[time_index],
            time_index,
        )
        future = target_observations[time_index + 1 :]
        realized = model.effective_progressive_lookaheads(
            requested_lookaheads, int(future.shape[0])
        )
        if chart_mode == "fixed_square":
            features_tensor = model._features(
                spec,
                theta,
                teacher["particles"],
                future[: realized[0]],
                grid,
                grid_weights,
                first_future_time_index=time_index + 1,
            )
        else:
            features_tensor = model.progressive_features(
                spec,
                theta,
                teacher["particles"],
                future,
                grid,
                grid_weights,
                first_future_time_index=time_index + 1,
                requested_lookaheads=requested_lookaheads,
            )
        features = features_tensor.numpy()
        normalized = tf.nn.softmax(teacher["log_unnormalized_weights"]).numpy()
        target = features @ normalized
        scale = np.maximum(
            1.0e-12,
            np.maximum(np.max(np.abs(features), axis=1), np.abs(target)),
        )
        scaled_features = features / scale[:, None]
        scaled_target = target / scale
        row_rank = int(np.linalg.matrix_rank(scaled_features))
        feature_count = int(features.shape[0])
        if row_rank != feature_count:
            raise RuntimeError(
                f"time {time_index}: feature rank {row_rank} != {feature_count}"
            )
        if args.anchor_strategy in (
            "weighted_quantile_voronoi",
            "positive_basis_plus_quantile_qp",
            "positive_basis_plus_quantile_kl",
            "positive_basis_quantile_analytic_center",
        ):
            quantile_count = int(args.anchor_count)
            if (
                args.anchor_strategy == "weighted_quantile_voronoi"
                and quantile_count < feature_count
            ):
                raise RuntimeError(
                    f"time {time_index}: anchor count {quantile_count} is smaller "
                    f"than feature count {feature_count}"
                )
            values = teacher["particles"][:, 0].numpy()
            ordering = np.argsort(values, kind="stable")
            cumulative = np.cumsum(normalized[ordering])
            quantiles = (
                np.arange(quantile_count, dtype=float) + 0.5
            ) / quantile_count
            sorted_positions = np.searchsorted(cumulative, quantiles, side="left")
            sorted_positions = np.minimum(sorted_positions, ordering.size - 1)
            indices = np.unique(ordering[sorted_positions])
            if indices.size != quantile_count:
                unused_by_mass = [
                    int(index)
                    for index in np.argsort(-normalized, kind="stable")
                    if int(index) not in set(indices.tolist())
                ]
                needed = quantile_count - int(indices.size)
                indices = np.asarray(
                    [*indices.tolist(), *unused_by_mass[:needed]], dtype=int
                )
                if indices.size != quantile_count:
                    raise RuntimeError(
                        f"time {time_index}: insufficient distinct teacher anchors"
                    )
            if args.anchor_strategy in (
                "positive_basis_plus_quantile_qp",
                "positive_basis_plus_quantile_kl",
                "positive_basis_quantile_analytic_center",
            ):
                feasible = linprog(
                    np.zeros(features.shape[1]),
                    A_eq=scaled_features,
                    b_eq=scaled_target,
                    bounds=(0.0, None),
                    method="highs",
                )
                basis = (
                    np.flatnonzero(feasible.x > 1.0e-10)
                    if feasible.success
                    else np.array([], dtype=int)
                )
                if not feasible.success or basis.size != feature_count:
                    raise RuntimeError(
                        f"time {time_index}: no positive basis for quantile union; "
                        f"status={feasible.message!r}, active={basis.size}"
                    )
                indices = np.asarray(
                    list(dict.fromkeys([*basis.tolist(), *indices.tolist()])),
                    dtype=int,
                )
            anchor_count = int(indices.size)
            anchor_values = values[indices]
            assignments = np.argmin(
                np.abs(values[:, None] - anchor_values[None, :]), axis=1
            )
            selected_weights = np.bincount(
                assignments, weights=normalized, minlength=anchor_count
            )
            if np.min(selected_weights) <= 0.0:
                raise RuntimeError(
                    f"time {time_index}: Voronoi aggregation produced a nonpositive mass"
                )
            if args.anchor_strategy == "positive_basis_plus_quantile_qp":
                anchor_matrix = scaled_features[:, indices]
                inverse_reference = 1.0 / selected_weights

                def objective(value: np.ndarray) -> float:
                    difference = value - selected_weights
                    return float(0.5 * np.dot(difference * inverse_reference, difference))

                def gradient(value: np.ndarray) -> np.ndarray:
                    return (value - selected_weights) * inverse_reference

                projected = minimize(
                    objective,
                    selected_weights,
                    jac=gradient,
                    method="SLSQP",
                    bounds=Bounds(np.zeros(anchor_count), np.full(anchor_count, np.inf)),
                    constraints=LinearConstraint(anchor_matrix, scaled_target, scaled_target),
                    options={"ftol": 1.0e-13, "maxiter": 2000},
                )
                equality_residual = np.max(
                    np.abs(anchor_matrix @ projected.x - scaled_target)
                )
                if (
                    not projected.success
                    or equality_residual > 1.0e-10
                    or np.min(projected.x) <= 0.0
                ):
                    raise RuntimeError(
                        f"time {time_index}: positive basis/quantile QP failed; "
                        f"status={projected.message!r}, residual={equality_residual}, "
                        f"minimum={np.min(projected.x)}"
                    )
                selected_weights = projected.x
            elif args.anchor_strategy == "positive_basis_plus_quantile_kl":
                anchor_matrix = scaled_features[:, indices]
                reference = selected_weights

                def kl_residual(multiplier: np.ndarray) -> np.ndarray:
                    logits = -(anchor_matrix.T @ multiplier)
                    if not np.all(np.isfinite(logits)) or np.max(logits) > 700.0:
                        return np.full(feature_count, np.finfo(float).max ** 0.25)
                    weights_at_multiplier = reference * np.exp(logits)
                    return anchor_matrix @ weights_at_multiplier - scaled_target

                def kl_jacobian(multiplier: np.ndarray) -> np.ndarray:
                    logits = -(anchor_matrix.T @ multiplier)
                    weights_at_multiplier = reference * np.exp(logits)
                    return -(
                        anchor_matrix * weights_at_multiplier[None, :]
                    ) @ anchor_matrix.T

                projected = least_squares(
                    kl_residual,
                    np.zeros(feature_count),
                    jac=kl_jacobian,
                    method="trf",
                    x_scale="jac",
                    ftol=1.0e-14,
                    xtol=1.0e-14,
                    gtol=1.0e-14,
                    max_nfev=5000,
                )
                logits = -(anchor_matrix.T @ projected.x)
                candidate_weights = reference * np.exp(logits)
                residual_norm = float(
                    np.max(np.abs(anchor_matrix @ candidate_weights - scaled_target))
                )
                if not projected.success or residual_norm > 1.0e-12:
                    raise RuntimeError(
                        f"time {time_index}: positive basis/quantile KL projection "
                        f"did not converge; status={projected.message!r}, "
                        f"residual={residual_norm}"
                    )
                if np.min(candidate_weights) <= 0.0:
                    raise RuntimeError(
                        f"time {time_index}: KL projection was not strictly positive"
                    )
                selected_weights = candidate_weights
            elif args.anchor_strategy == "positive_basis_quantile_analytic_center":
                basis_set = set(basis.tolist())
                extra_positions = np.asarray(
                    [
                        position
                        for position, index in enumerate(indices.tolist())
                        if index not in basis_set
                    ],
                    dtype=int,
                )
                basis_positions = np.asarray(
                    [
                        indices.tolist().index(index)
                        for index in basis.tolist()
                    ],
                    dtype=int,
                )
                basis_matrix = scaled_features[:, basis]
                basis_weights = np.linalg.solve(basis_matrix, scaled_target)
                if np.min(basis_weights) <= 0.0:
                    raise RuntimeError(
                        f"time {time_index}: source LP basis is not strictly positive"
                    )
                if extra_positions.size == 0:
                    selected_weights = basis_weights
                    indices = basis
                else:
                    extra_profile = selected_weights[extra_positions]
                    extra_matrix = scaled_features[:, indices[extra_positions]]
                    basis_correction = np.linalg.solve(
                        basis_matrix, extra_matrix @ extra_profile
                    )
                    positive_correction = basis_correction > 0.0
                    if np.any(positive_correction):
                        insertion_supremum = float(
                            np.min(
                                basis_weights[positive_correction]
                                / basis_correction[positive_correction]
                            )
                        )
                    else:
                        insertion_supremum = float("inf")
                    if insertion_supremum > 1.0:
                        insertion = 1.0
                    else:
                        # Maximizes log(alpha) + log(alpha_max - alpha).
                        insertion = 0.5 * insertion_supremum
                    selected_weights = np.zeros(len(indices))
                    selected_weights[extra_positions] = insertion * extra_profile
                    selected_weights[basis_positions] = (
                        basis_weights - insertion * basis_correction
                    )
                    equality_residual = float(
                        np.max(
                            np.abs(
                                scaled_features[:, indices] @ selected_weights
                                - scaled_target
                            )
                        )
                    )
                    if equality_residual > 1.0e-12 or np.min(selected_weights) <= 0.0:
                        raise RuntimeError(
                            f"time {time_index}: analytic-center construction failed; "
                            f"residual={equality_residual}, "
                            f"minimum={np.min(selected_weights)}"
                        )
            precision = np.diag(1.0 / selected_weights)
            projection = tp._contract_e_tp_dense_kkt_forward_core(
                teacher["particles"],
                teacher["log_unnormalized_weights"],
                features_tensor,
                tf.constant(indices, tf.int32),
                tf.constant(scale, DTYPE),
                tf.constant(selected_weights, DTYPE),
                tf.constant(precision, DTYPE),
            )
        else:
            solution = linprog(
                np.zeros(features.shape[1]),
                A_eq=scaled_features,
                b_eq=scaled_target,
                bounds=(0.0, None),
                method="highs",
            )
            indices = (
                np.flatnonzero(solution.x > 1.0e-10)
                if solution.success
                else np.array([], dtype=int)
            )
            if not solution.success or indices.size != feature_count:
                raise RuntimeError(
                    f"time {time_index}: no positive square chart; "
                    f"status={solution.message!r}, active={indices.size}"
                )
        if (
            args.anchor_strategy == "positive_lp_max_margin"
            and chart_mode == "fixed_overcomplete_kkt"
        ):
            outside = [
                int(index)
                for index in np.argsort(-normalized)
                if int(index) not in set(indices.tolist())
            ]
            extras = np.asarray(outside[: args.extra_anchors], dtype=int)
            if extras.size != args.extra_anchors:
                raise RuntimeError(f"time {time_index}: insufficient extra KKT anchors")
            indices = np.concatenate([indices, extras])
            anchor_features = scaled_features[:, indices]
            anchor_count = int(indices.size)
            objective = np.concatenate([np.zeros(anchor_count), np.array([-1.0])])
            lower_bound = np.concatenate([-np.eye(anchor_count), np.ones((anchor_count, 1))], axis=1)
            max_margin = linprog(
                objective,
                A_ub=lower_bound,
                b_ub=np.zeros(anchor_count),
                A_eq=np.concatenate([anchor_features, np.zeros((feature_count, 1))], axis=1),
                b_eq=scaled_target,
                bounds=[(0.0, None)] * (anchor_count + 1),
                method="highs",
            )
            if not max_margin.success or max_margin.x[-1] <= 1.0e-12:
                raise RuntimeError(
                    f"time {time_index}: no strictly positive overcomplete chart; "
                    f"status={max_margin.message!r}, margin={max_margin.x[-1] if max_margin.success else None}"
                )
            selected_weights = max_margin.x[:-1]
            precision = np.diag(1.0 / selected_weights)
            projection = tp._contract_e_tp_dense_kkt_forward_core(
                teacher["particles"],
                teacher["log_unnormalized_weights"],
                features_tensor,
                tf.constant(indices, tf.int32),
                tf.constant(scale, DTYPE),
                tf.constant(selected_weights, DTYPE),
                tf.constant(precision, DTYPE),
            )
        elif args.anchor_strategy == "positive_lp_max_margin":
            selected_weights = np.linalg.solve(
                scaled_features[:, indices], scaled_target
            )
            precision = np.zeros((0, 0))
            projection = tp._contract_e_tp_dense_square_forward_core(
                teacher["particles"],
                teacher["log_unnormalized_weights"],
                features_tensor,
                tf.constant(indices, tf.int32),
                tf.constant(scale, DTYPE),
            )
        if np.min(selected_weights) <= 0.0:
            raise RuntimeError(f"time {time_index}: chart is not strictly positive")
        parents = tf.reshape(projection["student_points"], [-1])
        parent_log_weights = tf.math.log(projection["student_weights"])
        active_indices.append(indices.tolist())
        row_scales.append(scale.tolist())
        reference_weights.append(selected_weights.tolist())
        precisions.append(precision.tolist())
        realized_lookaheads.append(list(realized))
        charts.append(
            {
                "time_index_zero_based": time_index,
                "teacher_count": int(features.shape[1]),
                "feature_rank": row_rank,
                "feature_names": [
                    "mass",
                    "state",
                    "state_square",
                    *(f"target_continuation_likelihood_h{value}" for value in realized),
                ],
                "realized_progressive_lookaheads": list(realized),
                "active_indices": indices.tolist(),
                "minimum_weight": float(projection["minimum_weight"].numpy()),
                "condition_number": float(projection["condition_number"].numpy()),
                "maximum_feature_residual_abs": float(
                    tf.reduce_max(tf.abs(projection["feature_residual"])).numpy()
                ),
            }
        )
    payload = {
        "schema": (
            "bayesfilter.contract_e_tp.scalar_sv_preparation.v2"
            if chart_mode == "fixed_overcomplete_kkt"
            else "bayesfilter.contract_e_tp.scalar_sv_preparation.v1"
        ),
        "status": "PASS_CENTER_ONLY_PREPARATION",
        "algorithm_id": model.ALGORITHM_ID,
        "row_id": args.row_id,
        "scope": "center_only_preparation_not_parameter_region_certificate",
        "target": {
            "theta": theta.numpy().tolist(),
            "parameter_names": list(PARAMETER_NAMES[args.row_id]),
            "time_steps": args.time_steps,
            "raw_observations": raw_observations.numpy().tolist(),
            "target_observations": target_observations.numpy().tolist(),
            "flow_observations": flow_observations.numpy().tolist(),
            "raw_observations_sha256": _tensor_sha256(raw_observations),
            "target_observations_sha256": _tensor_sha256(target_observations),
            "flow_observations_sha256": _tensor_sha256(flow_observations),
            "target_observation_policy": spec.target_observation_policy,
            "flow_observation_policy": spec.flow_observation_policy,
            "transition_before_first_observation": spec.transition_before_first_observation,
            "target_and_flow_are_distinct": bool(
                not np.array_equal(
                    target_observations.numpy(), flow_observations.numpy()
                )
            ),
        },
        "feature_contract": {
            "base_feature_names": ["mass", "state", "state_square"],
            "progressive_feature_name_template": "target_continuation_likelihood_h{lookahead}",
            "continuation_measure": "target_transition_times_target_likelihood",
            "ledh_proposal_likelihood_used_in_continuation": False,
            "lookahead_steps": args.lookahead_steps,
            "requested_progressive_lookaheads": list(requested_lookaheads),
            "realized_progressive_lookaheads": realized_lookaheads,
            "common_reference_scaling_retains_total_derivative": True,
        },
        "chart_contract": {
            "mode": chart_mode,
            "anchor_strategy": args.anchor_strategy,
            "anchor_count": args.anchor_count,
            "extra_anchor_count": args.extra_anchors if chart_mode == "fixed_overcomplete_kkt" else 0,
            "reference_metric": (
                "fixed_center_pearson_chi_square_precision_diag_inverse_reference_weight"
                if chart_mode == "fixed_overcomplete_kkt"
                else "not_applicable"
            ),
            "square_equivalent_singleton_feasible_set": bool(
                chart_mode == "fixed_overcomplete_kkt" and args.extra_anchors == 0
            ),
            "runtime_active_set_switching": False,
            "duplicate_quantile_fill_rule": (
                "highest_normalized_mass_unused_teacher_points_stable_order"
                if args.anchor_strategy in (
                    "weighted_quantile_voronoi",
                    "positive_basis_plus_quantile_qp",
                    "positive_basis_plus_quantile_kl",
                    "positive_basis_quantile_analytic_center",
                )
                else "not_applicable"
            ),
            "reference_weight_provenance": (
                "positive_basis_union_weighted_quantiles_then_chi_square_projection_of_voronoi_teacher_mass"
                if args.anchor_strategy == "positive_basis_plus_quantile_qp"
                else (
                    "positive_basis_union_weighted_quantiles_then_kl_information_projection_of_voronoi_teacher_mass"
                    if args.anchor_strategy == "positive_basis_plus_quantile_kl"
                    else (
                        "positive_basis_with_voronoi_extra_profile_at_full_feasible_insertion_or_log_barrier_analytic_center"
                        if args.anchor_strategy == "positive_basis_quantile_analytic_center"
                        else (
                            "teacher_mass_aggregated_to_fixed_weighted_quantile_voronoi_cells"
                            if args.anchor_strategy == "weighted_quantile_voronoi"
                            else "positive_center_maximum_minimum_weight_linear_program"
                        )
                    )
                )
            ),
        },
        "teacher_quadrature": {
            "family": "gauss_hermite_standard_normal",
            "order": args.teacher_order,
            "nodes": nodes.numpy().tolist(),
            "weights": weights.numpy().tolist(),
        },
        "continuation_quadrature": {
            "family": "legendre_fixed_interval",
            "order": args.continuation_order,
            "radius": args.continuation_radius,
            "points": grid.numpy().tolist(),
            "weights": grid_weights.numpy().tolist(),
        },
        "active_indices": active_indices,
        "row_scales": row_scales,
        "reference_weights": reference_weights,
        "precisions": precisions,
        "charts": charts,
        "summary": {
            "chart_count": len(charts),
            "minimum_weight": min((row["minimum_weight"] for row in charts), default=None),
            "maximum_condition_number": max((row["condition_number"] for row in charts), default=None),
            "maximum_feature_residual_abs": max(
                (row["maximum_feature_residual_abs"] for row in charts), default=0.0
            ),
            "minimum_feature_count": min(
                (row["feature_rank"] for row in charts), default=0
            ),
            "maximum_feature_count": max(
                (row["feature_rank"] for row in charts), default=0
            ),
        },
        "execution": {
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "backend": "TensorFlow_float64_CPU_reference_exception_plus_SciPy_chart_selection",
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "wall_time_seconds": time.perf_counter() - started,
            "command": " ".join(sys.argv),
        },
        "nonclaims": [
            "not off-center chart validity",
            "not target accuracy or score agreement",
            "not full-horizon evidence",
            "not cross-method equivalence",
            "not HMC or default readiness",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"row_id": args.row_id, "summary": payload["summary"]}, indent=2))


if __name__ == "__main__":
    main()
