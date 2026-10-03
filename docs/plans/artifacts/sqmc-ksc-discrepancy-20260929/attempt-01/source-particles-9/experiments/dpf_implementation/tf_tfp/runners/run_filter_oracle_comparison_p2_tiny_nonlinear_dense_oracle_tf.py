"""Run P2 tiny nonlinear dense-oracle value and gradient comparison.

P2 certifies at least one tiny nonlinear dense/refined reference row before
later DPF statistical comparison.  P0 marks DPF rows as P5-eligible, not
P2-eligible, so this runner records DPF deferral rather than executing DPF.
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/bayesfilter-dpf-mpl")
PRE_IMPORT_CUDA_VISIBLE_DEVICES = os.environ["CUDA_VISIBLE_DEVICES"]

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tensorflow as tf
import tensorflow_probability as tfp

import bayesfilter.highdim as highdim
from bayesfilter.nonlinear.sigma_points_tf import tf_svd_sigma_point_filter
from bayesfilter.nonlinear.svd_cut_tf import tf_svd_cut4_filter
from bayesfilter.structural import StatePartition, StructuralFilterConfig
from bayesfilter.structural_tf import TFStructuralStateSpace
from experiments.dpf_implementation.tf_tfp.runners.common_tf import (
    OUTPUT_DIR,
    REPORT_DIR,
    REPO_ROOT,
    environment_manifest,
    load_json,
    scalar,
    stable_digest,
    tensor_to_json,
    utc_now,
    write_json,
    write_text,
)
from experiments.dpf_implementation.tf_tfp.runners.run_filter_oracle_comparison_p0_registry_tf import (
    REGISTRY_PATH,
)


DTYPE = tf.float64
MODULE_PATH = (
    "experiments.dpf_implementation.tf_tfp.runners."
    "run_filter_oracle_comparison_p2_tiny_nonlinear_dense_oracle_tf"
)
PLAN_PATH = (
    "docs/plans/"
    "bayesfilter-dpf-filter-oracle-comparison-p2-tiny-nonlinear-dense-oracle-subplan-2026-06-08.md"
)
RESULT_PATH = (
    "docs/plans/"
    "bayesfilter-dpf-filter-oracle-comparison-p2-tiny-nonlinear-dense-oracle-result-2026-06-08.md"
)
REVIEW_LEDGER_PATH = (
    "docs/plans/"
    "bayesfilter-dpf-filter-oracle-comparison-p2-claude-review-ledger-2026-06-08.md"
)
JSON_PATH = OUTPUT_DIR / "dpf_filter_oracle_comparison_p2_tiny_nonlinear_dense_oracle_2026-06-08.json"
REPORT_PATH = REPORT_DIR / "dpf-filter-oracle-comparison-p2-tiny-nonlinear-dense-oracle-2026-06-08.md"

TARGET_ID = "p44_m2_cubic_additive_gaussian_panel"
LOW_DENSE_ORDER = 161
HIGH_DENSE_ORDER = 241
DIMS = (1, 2, 3)
VALUE_REFINEMENT_TOL = 1e-8
DIRECTIONAL_REFINEMENT_TOL = 1e-7
FD_STEPS = (2e-4, 1e-4, 5e-5)
FD_REGRESSION_TOL = 5e-5
FD_MAX_STEP_SPREAD_TOL = 2e-4


class P2ValidationError(ValueError):
    """Raised when a P2 artifact violates the phase contract."""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    if args.validate_only:
        _validate_payload(load_json(JSON_PATH))
        print("P2_TINY_NONLINEAR_DENSE_ORACLE_VALIDATED")
        return 0

    start = time.perf_counter()
    payload = _run()
    payload["run_manifest"]["wall_time_seconds"] = time.perf_counter() - start
    payload["reproducibility_digest"] = _digest_payload(payload)
    markdown = _markdown(payload)
    write_json(JSON_PATH, payload)
    write_text(REPORT_PATH, markdown)
    write_text(REPO_ROOT / RESULT_PATH, markdown)
    _validate_payload(payload)
    print(payload["decision"])
    return 0


def _run() -> dict[str, Any]:
    registry = load_json(REGISTRY_PATH)
    routes = registry["route_matrix"][TARGET_ID]
    target = next(row for row in registry["targets"] if row["target_id"] == TARGET_ID)
    theta0 = _theta0()

    target_rows = []
    for dim in DIMS:
        target_rows.append(_target_row(theta0, dim, routes))

    route_summaries = _route_summaries(target_rows)
    veto = _veto_diagnostics(routes, target_rows, route_summaries)
    decision = (
        "PASS_P2_TINY_NONLINEAR_DENSE_ORACLE_PENDING_CLAUDE_REVIEW"
        if not any(bool(value) for value in veto.values())
        else "P2_TINY_NONLINEAR_DENSE_ORACLE_VETO_PENDING_REVIEW"
    )
    manifest = environment_manifest(
        command="CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m " + MODULE_PATH,
        pre_import_cuda_visible_devices=PRE_IMPORT_CUDA_VISIBLE_DEVICES,
    )
    manifest.update(
        {
            "plan_path": PLAN_PATH,
            "result_path": RESULT_PATH,
            "review_ledger_path": REVIEW_LEDGER_PATH,
            "json_path": str(JSON_PATH.relative_to(REPO_ROOT)),
            "report_path": str(REPORT_PATH.relative_to(REPO_ROOT)),
            "data_version": "deterministic P44-M2 cubic additive-Gaussian fixture",
            "target_id": TARGET_ID,
            "dense_orders": [LOW_DENSE_ORDER, HIGH_DENSE_ORDER],
            "dims": list(DIMS),
            "fd_steps": list(FD_STEPS),
            "seeds": "N/A: deterministic dense and sigma-point routes only",
            "particle_counts": "N/A: DPF routes are deferred to P5 by P0 registry",
        }
    )
    return {
        "metadata_date": "2026-06-08",
        "created_at_utc": utc_now(),
        "phase": "P2",
        "execution_route": "VISIBLE_IN_DIALOGUE",
        "decision": decision,
        "question": (
            "For the P44-M2 tiny cubic additive-Gaussian target, do dense "
            "refinement and deterministic approximation routes provide "
            "same-target value and gradient evidence without promoting "
            "non-oracles or running DPF outside P0 eligibility?"
        ),
        "target": {
            "target_id": TARGET_ID,
            "registry_target": target,
            "theta": tensor_to_json(theta0),
            "gradient_parameterization": (
                "theta = (rho_raw, log_q, log_r, raw_initial_mean, cubic_raw)"
            ),
            "observations": tensor_to_json(_observations(max(DIMS))),
            "model_definition": {
                "state_law": "x_t = rho x_{t-1} + eta_t",
                "observation_law": "y_t = x_t + cubic * x_t^3 + epsilon_t",
                "panel_convention": "independent scalar panels summed over dims",
                "horizon": 2,
            },
        },
        "routes": {
            "dense_refined_quadrature": routes["dense_refined_quadrature"],
            "ukf": routes["ukf"],
            "svd_sigma_point": routes["svd_sigma_point"],
            "cut4": routes["cut4"],
            "zhao_cui_fixed_design_tt": routes["zhao_cui_fixed_design_tt"],
            "dpf_bootstrap_ot": {
                **routes["dpf_bootstrap_ot"],
                "p2_execution": "deferred_to_p5_by_p0_registry",
            },
            "dpf_ledh_pfpf_ot": {
                **routes["dpf_ledh_pfpf_ot"],
                "p2_execution": "deferred_to_p5_by_p0_registry",
            },
        },
        "evidence_contract": {
            "baseline_comparator": (
                "dense order-241 fixed quadrature after order-161 refinement "
                "passes value and directional score tolerances"
            ),
            "primary_criterion": (
                "at least one dim has a promoted dense-oracle row; all dims "
                "record dense refinement and deterministic route gaps"
            ),
            "gradient_object": "reference_score for dense; diagnostic_fixed_branch_score or fixed_branch_score for deterministic approximations",
            "dpf_policy": "DPF not run in P2; P0 marks DPF rows p2=false and p5=true",
        },
        "tolerances": {
            "dense_value_refinement_abs": VALUE_REFINEMENT_TOL,
            "dense_directional_score_refinement_abs": DIRECTIONAL_REFINEMENT_TOL,
            "fd_regression_abs": FD_REGRESSION_TOL,
            "fd_step_spread_abs": FD_MAX_STEP_SPREAD_TOL,
        },
        "rows": target_rows,
        "route_summaries": route_summaries,
        "veto_diagnostics": veto,
        "explanatory_diagnostics": {
            "finite_difference_is_diagnostic_only": True,
            "ukf_svd_cut4_are_not_oracles": True,
            "dpf_deferred_to_p5": True,
            "dense_reference_refinement_recorded": True,
        },
        "run_manifest": manifest,
        "nonclaims": _nonclaims(),
    }


def _target_row(theta: tf.Tensor, dim: int, routes: dict[str, Any]) -> dict[str, Any]:
    initial_law_alignment = _initial_law_alignment(theta, dim)
    low_value, low_score = _value_and_score(
        lambda current: _dense_panel_value(current, dim, LOW_DENSE_ORDER),
        theta,
    )
    high_value, high_score = _value_and_score(
        lambda current: _dense_panel_value(current, dim, HIGH_DENSE_ORDER),
        theta,
    )
    refinement = {
        "low_order": LOW_DENSE_ORDER,
        "high_order": HIGH_DENSE_ORDER,
        "value_gap": scalar(tf.abs(low_value - high_value)),
        "max_directional_score_gap": scalar(_directional_abs_max(low_score, high_score)),
        "reference_score_norm": scalar(tf.linalg.norm(high_score)),
        "near_stationary_guardrail": _near_stationary_guardrail(high_score),
        "directional_fd_checks": _directional_fd_checks(
            lambda current: _dense_panel_value(current, dim, HIGH_DENSE_ORDER),
            theta,
            high_score,
        ),
    }
    route_rows = []
    for route_id, value_fn in (
        ("ukf", lambda current: _ukf_panel_value(current, dim)),
        ("svd_sigma_point", lambda current: _svd_panel_value(current, dim)),
        ("cut4", lambda current: _cut4_panel_value(current, dim)),
        ("zhao_cui_fixed_design_tt", lambda current: _tt_panel_value(current, dim)),
    ):
        value, score = _value_and_score(value_fn, theta)
        route_rows.append(
            _route_row(
                route_id=route_id,
                route=routes[route_id],
                value=value,
                score=score,
                reference_value=high_value,
                reference_score=high_score,
                dim=dim,
            )
        )
    return {
        "target_id": TARGET_ID,
        "dim": int(dim),
        "dense_reference": {
            "claim_class": routes["dense_refined_quadrature"]["claim_class"],
            "route_status": routes["dense_refined_quadrature"]["route_status"],
            "value": scalar(high_value),
            "score": tensor_to_json(high_score),
            "finite": _finite_scalar_vector(high_value, high_score),
            "initial_law_alignment": initial_law_alignment,
            "refinement": refinement,
            "promotion_status": (
                "promoted_dense_oracle_for_p2"
                if _refinement_passes(refinement)
                else "dense_oracle_veto"
            ),
        },
        "deterministic_routes": route_rows,
        "blocked_or_deferred_routes": [
            {
                "route_id": "kalman_exact",
                "status": "blocked_nonlinear_target",
                "reason": "P0 marks Kalman exact blocked for this nonlinear target.",
            },
            {
                "route_id": "dpf_bootstrap_ot",
                "status": "deferred_to_p5",
                "reason": "P0 marks DPF p2=false and p5=true after dense reference pass.",
            },
            {
                "route_id": "dpf_ledh_pfpf_ot",
                "status": "deferred_to_p5",
                "reason": "P0 marks DPF p2=false and p5=true after dense reference pass.",
            },
        ],
    }


def _route_row(
    *,
    route_id: str,
    route: dict[str, Any],
    value: tf.Tensor,
    score: tf.Tensor,
    reference_value: tf.Tensor,
    reference_score: tf.Tensor,
    dim: int,
) -> dict[str, Any]:
    value_error = value - reference_value
    score_error = score - reference_score
    diagnostics = {}
    if route_id in {"ukf", "svd_sigma_point", "cut4"}:
        result = _sigma_route_result(route_id, _theta0(), dim)
        diagnostics = _diagnostics_to_json(result.diagnostics.as_dict())
    return {
        "route_id": route_id,
        "claim_class": route["claim_class"],
        "route_status": route["route_status"],
        "phase_eligibility": route["phase_eligibility"],
        "primary_gradient_statistic": route["primary_gradient_statistic"],
        "value": scalar(value),
        "score": tensor_to_json(score),
        "value_error": scalar(value_error),
        "per_observation_value_error": scalar(value_error / tf.constant(2 * dim, DTYPE)),
        "abs_value_error": scalar(tf.abs(value_error)),
        "directional_score_gap": scalar(_directional_abs_max(score, reference_score)),
        "score_error_norm": scalar(tf.linalg.norm(score_error)),
        "relative_score_error": scalar(_relative_score_error(score, reference_score)),
        "finite": _finite_scalar_vector(value, score),
        "diagnostics": diagnostics,
        "interpretation": _route_interpretation(route_id, route),
    }


def _route_interpretation(route_id: str, route: dict[str, Any]) -> str:
    if route["claim_class"] == "CERTIFIED_APPROXIMATION":
        return f"{route_id}_same_target_certified_approximation_not_oracle"
    return f"{route_id}_diagnostic_only_not_oracle"


def _route_summaries(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summaries: dict[str, Any] = {
        "dense_refined_quadrature": {
            "all_refinement_pass": all(
                _refinement_passes(row["dense_reference"]["refinement"]) for row in rows
            ),
            "max_value_refinement_gap": max(
                row["dense_reference"]["refinement"]["value_gap"] for row in rows
            ),
            "max_directional_score_refinement_gap": max(
                row["dense_reference"]["refinement"]["max_directional_score_gap"]
                for row in rows
            ),
            "promoted_dims": [
                row["dim"]
                for row in rows
                if row["dense_reference"]["promotion_status"]
                == "promoted_dense_oracle_for_p2"
            ],
        }
    }
    for route_id in ("ukf", "svd_sigma_point", "cut4", "zhao_cui_fixed_design_tt"):
        route_rows = [
            route
            for row in rows
            for route in row["deterministic_routes"]
            if route["route_id"] == route_id
        ]
        summaries[route_id] = {
            "claim_class": route_rows[0]["claim_class"],
            "max_abs_value_error": max(route["abs_value_error"] for route in route_rows),
            "max_directional_score_gap": max(
                route["directional_score_gap"] for route in route_rows
            ),
            "max_relative_score_error": max(
                route["relative_score_error"] for route in route_rows
            ),
            "all_finite": all(route["finite"] for route in route_rows),
            "interpretation": route_rows[0]["interpretation"],
        }
    return summaries


def _veto_diagnostics(
    routes: dict[str, Any],
    rows: list[dict[str, Any]],
    route_summaries: dict[str, Any],
) -> dict[str, bool]:
    dpf_p2_eligible = any(
        bool(routes[route_id]["phase_eligibility"].get("p2"))
        for route_id in ("dpf_bootstrap_ot", "dpf_ledh_pfpf_ot")
    )
    fd_bad = any(
        check["max_abs_error_vs_ad"] > FD_REGRESSION_TOL
        or check["max_step_spread"] > FD_MAX_STEP_SPREAD_TOL
        for row in rows
        for check in row["dense_reference"]["refinement"]["directional_fd_checks"]
    )
    return {
        "dense_reference_lacks_refinement_pass": not bool(
            route_summaries["dense_refined_quadrature"]["all_refinement_pass"]
        ),
        "no_promoted_dense_oracle_row": not bool(
            route_summaries["dense_refined_quadrature"]["promoted_dims"]
        ),
        "dpf_executed_or_p2_eligible": dpf_p2_eligible,
        "non_p2_route_executed": any(
            not bool(route["phase_eligibility"].get("p2"))
            for row in rows
            for route in row["deterministic_routes"]
        ),
        "ukf_svd_cut4_called_oracle": any(
            route["claim_class"] == "EXACT_ORACLE"
            for row in rows
            for route in row["deterministic_routes"]
            if route["route_id"] in {"ukf", "svd_sigma_point", "cut4"}
        ),
        "route_row_nonfinite": any(
            not bool(route["finite"])
            for row in rows
            for route in row["deterministic_routes"]
        ),
        "dense_reference_nonfinite": any(
            not bool(row["dense_reference"]["finite"]) for row in rows
        ),
        "single_step_finite_difference_used_as_promotion": False,
        "directional_fd_diagnostic_large": fd_bad,
        "gradient_parameterization_mismatch": any(
            not bool(row["dense_reference"]["initial_law_alignment"]["aligned"])
            for row in rows
        ),
    }


def _convention() -> highdim.MeasureConvention:
    return highdim.MeasureConvention(
        density_measure=highdim.DensityMeasure.REFERENCE_MEASURE,
        mass_measure=highdim.MassMeasure.REFERENCE_MEASURE,
        reference_weight_name="omega",
    )


def _observations(dim: int) -> tf.Tensor:
    values = tf.constant(
        [
            [0.10, -0.04, 0.07],
            [-0.02, 0.06, -0.05],
        ],
        dtype=DTYPE,
    )
    return values[:, : int(dim)]


def _theta0() -> tf.Tensor:
    return tf.constant([0.25, math.log(0.14), math.log(0.10), 0.04, 0.35], dtype=DTYPE)


def _physical_parts(theta: tf.Tensor, dim: int) -> dict[str, tf.Tensor]:
    theta = tf.convert_to_tensor(theta, dtype=DTYPE)
    rho_scale = tf.constant([1.00, 0.86, 0.72], dtype=DTYPE)[:dim]
    q_scale = tf.constant([0.90, 1.15, 1.35], dtype=DTYPE)[:dim]
    r_scale = tf.constant([1.00, 1.20, 0.85], dtype=DTYPE)[:dim]
    mean_scale = tf.constant([1.00, -0.50, 0.25], dtype=DTYPE)[:dim]
    cubic_scale = tf.constant([1.00, 0.80, 1.20], dtype=DTYPE)[:dim]
    raw_initial_variance = tf.constant([0.55, 0.70, 0.90], dtype=DTYPE)[:dim]
    rho = 0.45 * tf.tanh(theta[0]) * rho_scale
    transition_variance = tf.exp(theta[1]) * q_scale
    observation_variance = tf.exp(theta[2]) * r_scale
    raw_initial_mean = theta[3] * mean_scale
    cubic = 0.04 * tf.tanh(theta[4]) * cubic_scale
    initial_mean = rho * raw_initial_mean
    initial_variance = tf.square(rho) * raw_initial_variance + transition_variance
    return {
        "rho": rho,
        "transition_variance": transition_variance,
        "observation_variance": observation_variance,
        "raw_initial_mean": raw_initial_mean,
        "raw_initial_variance": raw_initial_variance,
        "initial_mean": initial_mean,
        "initial_variance": initial_variance,
        "cubic": cubic,
    }


def _axis_part(theta: tf.Tensor, axis: int) -> dict[str, tf.Tensor]:
    parts = _physical_parts(theta, axis + 1)
    return {key: value[axis] for key, value in parts.items()}


def _rows(values: tf.Tensor, width: int, name: str) -> tf.Tensor:
    tensor = tf.convert_to_tensor(values, dtype=DTYPE)
    if tensor.shape.rank == 1:
        tensor = tensor[tf.newaxis, :]
    if tensor.shape.rank != 2 or tensor.shape[1] != int(width):
        raise ValueError(f"{name} has wrong shape")
    return tensor


@dataclass(frozen=True)
class _ScalarCubicAdditiveGaussianSSM:
    axis: int

    def parameter_dim(self) -> int:
        return 5

    def state_dim(self) -> int:
        return 1

    def observation_dim(self) -> int:
        return 1

    def initial_log_density(self, theta: tf.Tensor, x0: tf.Tensor) -> tf.Tensor:
        points = _rows(x0, 1, "x0")[:, 0]
        parts = _axis_part(theta, self.axis)
        return tfp.distributions.Normal(
            loc=parts["initial_mean"],
            scale=tf.sqrt(parts["initial_variance"]),
        ).log_prob(points)

    def transition_log_density(
        self,
        theta: tf.Tensor,
        x_prev: tf.Tensor,
        x_next: tf.Tensor,
        t: int,
    ) -> tf.Tensor:
        del t
        previous = _rows(x_prev, 1, "x_prev")[:, 0]
        current = _rows(x_next, 1, "x_next")[:, 0]
        parts = _axis_part(theta, self.axis)
        return tfp.distributions.Normal(
            loc=parts["rho"] * previous,
            scale=tf.sqrt(parts["transition_variance"]),
        ).log_prob(current)

    def observation_log_density(
        self,
        theta: tf.Tensor,
        x_t: tf.Tensor,
        y_t: tf.Tensor,
        t: int,
    ) -> tf.Tensor:
        del t
        points = _rows(x_t, 1, "x_t")[:, 0]
        observation = tf.reshape(tf.convert_to_tensor(y_t, dtype=DTYPE), [1])[0]
        parts = _axis_part(theta, self.axis)
        loc = points + parts["cubic"] * tf.pow(points, 3)
        return tfp.distributions.Normal(
            loc=loc,
            scale=tf.sqrt(parts["observation_variance"]),
        ).log_prob(observation)

    def manifest_payload(self) -> dict[str, object]:
        return {
            "family": "p44_m2_scalar_cubic_additive_gaussian",
            "axis": int(self.axis),
            "parameterization": "(rho_raw, log_q, log_r, raw_initial_mean, cubic_raw)",
        }


def _structural_model(theta: tf.Tensor, dim: int) -> TFStructuralStateSpace:
    parts = _physical_parts(theta, dim)
    padding_dim = max(0, 3 - (2 * dim))
    innovation_dim = dim + padding_dim

    def transition_fn(previous_state: tf.Tensor, innovation: tf.Tensor) -> tf.Tensor:
        previous = _rows(previous_state, dim, "previous_state")
        innovation_points = _rows(innovation, innovation_dim, "innovation")[:, :dim]
        next_points = (
            parts["rho"][tf.newaxis, :] * previous
            + tf.sqrt(parts["transition_variance"])[tf.newaxis, :] * innovation_points
        )
        return next_points[0] if tf.convert_to_tensor(previous_state).shape.rank == 1 else next_points

    def observation_fn(state_points: tf.Tensor) -> tf.Tensor:
        points = _rows(state_points, dim, "state_points")
        observations = points + parts["cubic"][tf.newaxis, :] * tf.pow(points, 3)
        return observations[0] if tf.convert_to_tensor(state_points).shape.rank == 1 else observations

    return TFStructuralStateSpace(
        partition=StatePartition(
            state_names=tuple(f"x{axis}" for axis in range(dim)),
            stochastic_indices=tuple(range(dim)),
            deterministic_indices=(),
            innovation_dim=innovation_dim,
        ),
        config=StructuralFilterConfig(
            integration_space="innovation",
            deterministic_completion="none",
            approximation_label="p44_m2_cubic_additive_gaussian_p2_same_target",
        ),
        initial_mean=parts["initial_mean"],
        initial_covariance=tf.linalg.diag(parts["initial_variance"]),
        innovation_covariance=tf.eye(innovation_dim, dtype=DTYPE),
        observation_covariance=tf.linalg.diag(parts["observation_variance"]),
        transition_fn=transition_fn,
        observation_fn=observation_fn,
        name=f"p44_m2_cubic_additive_gaussian_p2_dim_{dim}",
    )


def _initial_law_alignment(theta: tf.Tensor, dim: int) -> dict[str, Any]:
    parts = _physical_parts(theta, dim)
    structural = _structural_model(theta, dim)
    dense_mean = parts["initial_mean"]
    dense_covariance = tf.linalg.diag(parts["initial_variance"])
    mean_gap = tf.reduce_max(tf.abs(structural.initial_mean - dense_mean))
    covariance_gap = tf.reduce_max(
        tf.abs(structural.initial_covariance - dense_covariance)
    )
    return {
        "policy": (
            "dense scalar route and structural sigma-point routes use the same "
            "predictive initial law for x_0"
        ),
        "dense_initial_mean": tensor_to_json(dense_mean),
        "structural_initial_mean": tensor_to_json(structural.initial_mean),
        "dense_initial_covariance": tensor_to_json(dense_covariance),
        "structural_initial_covariance": tensor_to_json(structural.initial_covariance),
        "max_mean_gap": scalar(mean_gap),
        "max_covariance_gap": scalar(covariance_gap),
        "aligned": scalar(mean_gap) <= 1e-14 and scalar(covariance_gap) <= 1e-14,
    }


def _dense_config(axis: int, order: int) -> highdim.FixedBranchFilterConfig:
    return highdim.FixedBranchFilterConfig(
        fit_config=None,
        density_tau=0.0,
        normalizer_floor=1e-14,
        denominator_floor=1e-14,
        retained_storage_byte_budget=20_000_000,
        coordinate_maps=(
            highdim.AffineCoordinateMap(
                offset=tf.constant([0.0], dtype=DTYPE),
                matrix=tf.constant([[6.0]], dtype=DTYPE),
            ),
        ),
        measure_convention=_convention(),
        deterministic_seed=f"p2-p44-m2-dense-axis-{axis}-order-{order}",
        fit_quadrature_order=int(order),
    )


def _tt_config(axis: int) -> highdim.FixedBranchFilterConfig:
    convention = _convention()
    product_basis = highdim.ProductBasis(
        [highdim.LegendreBasis1D(highdim.BoundedInterval(-1.0, 1.0), 48)],
        convention,
    )
    return highdim.FixedBranchFilterConfig(
        fit_config=highdim.FixedTTFitConfig(
            ranks=(1, 1),
            ridge=1e-12,
            max_sweeps=2,
            sweep_order=(0,),
            row_budget=512,
            column_budget=128,
            dense_matrix_byte_budget=400_000,
            normal_matrix_byte_budget=200_000,
            condition_number_warning=1e10,
            condition_number_veto=1e14,
            holdout_tolerance=5e-4,
        ),
        density_tau=0.0,
        normalizer_floor=1e-14,
        denominator_floor=1e-14,
        retained_storage_byte_budget=20_000_000,
        coordinate_maps=(
            highdim.AffineCoordinateMap(
                offset=tf.constant([0.0], dtype=DTYPE),
                matrix=tf.constant([[6.0]], dtype=DTYPE),
            ),
        ),
        measure_convention=convention,
        deterministic_seed=f"p2-p44-m2-tt-axis-{axis}",
        product_basis=product_basis,
        initial_cores=(
            highdim.TTCore(tf.ones([1, product_basis.bases[0].basis_dim, 1], dtype=DTYPE)),
        ),
        fit_quadrature_order=161,
    )


def _dense_scalar_value(theta: tf.Tensor, axis: int, order: int) -> tf.Tensor:
    result = highdim.FixedBranchSquaredTTFilter(_dense_config(axis, order)).log_likelihood(
        _ScalarCubicAdditiveGaussianSSM(axis),
        theta,
        _observations(axis + 1)[:, axis : axis + 1],
    )
    return result.log_likelihood


def _dense_panel_value(theta: tf.Tensor, dim: int, order: int) -> tf.Tensor:
    return tf.reduce_sum(
        tf.stack([_dense_scalar_value(theta, axis, order) for axis in range(dim)])
    )


def _sigma_route_result(route_id: str, theta: tf.Tensor, dim: int):
    model = _structural_model(theta, dim)
    if route_id == "ukf":
        return tf_svd_sigma_point_filter(
            _observations(dim),
            model,
            backend="tf_svd_ukf",
            innovation_floor=tf.constant(1e-12, dtype=DTYPE),
            return_filtered=True,
        )
    if route_id == "svd_sigma_point":
        return tf_svd_sigma_point_filter(
            _observations(dim),
            model,
            backend="tf_svd_cubature",
            innovation_floor=tf.constant(1e-12, dtype=DTYPE),
            return_filtered=True,
        )
    if route_id == "cut4":
        return tf_svd_cut4_filter(
            _observations(dim),
            model,
            innovation_floor=tf.constant(1e-12, dtype=DTYPE),
            return_filtered=True,
        )
    raise ValueError(f"unknown sigma route {route_id}")


def _ukf_panel_value(theta: tf.Tensor, dim: int) -> tf.Tensor:
    return _sigma_route_result("ukf", theta, dim).log_likelihood


def _svd_panel_value(theta: tf.Tensor, dim: int) -> tf.Tensor:
    return _sigma_route_result("svd_sigma_point", theta, dim).log_likelihood


def _cut4_panel_value(theta: tf.Tensor, dim: int) -> tf.Tensor:
    return _sigma_route_result("cut4", theta, dim).log_likelihood


def _tt_panel_value(theta: tf.Tensor, dim: int) -> tf.Tensor:
    terms = []
    for axis in range(dim):
        result = highdim.scalar_nonlinear_fixed_design_tt_value_path(
            _ScalarCubicAdditiveGaussianSSM(axis),
            theta,
            _observations(axis + 1)[:, axis : axis + 1],
            _tt_config(axis),
            fixture_id=f"p2.p44.m2.cubic-additive-gaussian.axis-{axis}.v1",
            branch_seed_prefix=f"p2-p44-m2-cubic-additive-gaussian-axis-{axis}",
            retained_moment_order=241,
            retained_propagation_order=241,
        )
        terms.append(result.log_likelihood)
    return tf.reduce_sum(tf.stack(terms))


def _value_and_score(value_fn, theta: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    theta = tf.convert_to_tensor(theta, dtype=DTYPE)
    with tf.GradientTape() as tape:
        tape.watch(theta)
        value = value_fn(theta)
    score = tape.gradient(value, theta)
    if score is None:
        raise P2ValidationError("GradientTape returned None")
    return value, score


def _directions(size: int) -> tf.Tensor:
    eye = tf.eye(size, dtype=DTYPE)
    mixed_a = tf.cast(tf.range(1, size + 1), DTYPE)
    mixed_a = mixed_a / tf.linalg.norm(mixed_a)
    mixed_b = tf.where(
        tf.math.floormod(tf.range(size), 2) == 0,
        tf.ones([size], dtype=DTYPE),
        -tf.ones([size], dtype=DTYPE),
    )
    mixed_b = mixed_b / tf.linalg.norm(mixed_b)
    mixed_c = tf.reverse(mixed_a, axis=[0])
    return tf.concat(
        [eye, mixed_a[tf.newaxis, :], mixed_b[tf.newaxis, :], mixed_c[tf.newaxis, :]],
        axis=0,
    )


def _directional_abs_max(candidate_score: tf.Tensor, reference_score: tf.Tensor) -> tf.Tensor:
    residual = tf.cast(candidate_score, DTYPE) - tf.cast(reference_score, DTYPE)
    directional = tf.linalg.matvec(_directions(int(residual.shape[0])), residual)
    return tf.reduce_max(tf.abs(directional))


def _relative_score_error(candidate_score: tf.Tensor, reference_score: tf.Tensor) -> tf.Tensor:
    return tf.linalg.norm(tf.cast(candidate_score, DTYPE) - tf.cast(reference_score, DTYPE)) / tf.maximum(
        tf.constant(1.0, dtype=DTYPE),
        tf.linalg.norm(tf.cast(reference_score, DTYPE)),
    )


def _directional_fd_checks(value_fn, theta: tf.Tensor, score: tf.Tensor) -> list[dict[str, Any]]:
    checks = []
    for index, direction in enumerate(tf.unstack(_directions(int(theta.shape[0])), axis=0)):
        ad_directional = tf.tensordot(score, direction, axes=1)
        step_estimates = []
        for step in FD_STEPS:
            h = tf.constant(step, dtype=DTYPE)
            plus = value_fn(theta + h * direction)
            minus = value_fn(theta - h * direction)
            step_estimates.append((plus - minus) / (2.0 * h))
        estimates = tf.stack(step_estimates)
        mean_estimate = tf.reduce_mean(estimates)
        checks.append(
            {
                "direction_index": int(index),
                "direction": tensor_to_json(direction),
                "ad_directional": scalar(ad_directional),
                "fd_step_estimates": tensor_to_json(estimates),
                "fd_regression_mean": scalar(mean_estimate),
                "max_abs_error_vs_ad": scalar(tf.reduce_max(tf.abs(estimates - ad_directional))),
                "max_step_spread": scalar(tf.reduce_max(estimates) - tf.reduce_min(estimates)),
                "policy": "multi_step_directional_fd_diagnostic_only",
            }
        )
    return checks


def _near_stationary_guardrail(score: tf.Tensor) -> dict[str, Any]:
    norm = scalar(tf.linalg.norm(score))
    floor = 1e-12
    # No score-covariance estimate exists for this fixed-data P2 row, so the
    # P42 guardrail treats relative score error as insufficient by itself.
    return {
        "score_norm": norm,
        "score_scale_available": False,
        "floor": floor,
        "treated_as_potentially_near_stationary": True,
        "required_supplements": [
            "absolute vector error",
            "coordinatewise absolute errors",
            "deterministic directional derivative checks",
        ],
    }


def _refinement_passes(refinement: dict[str, Any]) -> bool:
    if refinement["value_gap"] > VALUE_REFINEMENT_TOL:
        return False
    if refinement["max_directional_score_gap"] > DIRECTIONAL_REFINEMENT_TOL:
        return False
    for check in refinement["directional_fd_checks"]:
        if check["max_abs_error_vs_ad"] > FD_REGRESSION_TOL:
            return False
        if check["max_step_spread"] > FD_MAX_STEP_SPREAD_TOL:
            return False
    return True


def _finite_scalar_vector(value: tf.Tensor, score: tf.Tensor) -> bool:
    return bool(
        tf.math.is_finite(value).numpy()
        and tf.reduce_all(tf.math.is_finite(tf.cast(score, DTYPE))).numpy()
    )


def _diagnostics_to_json(values: Any) -> Any:
    if isinstance(values, dict):
        return {key: _diagnostics_to_json(value) for key, value in values.items()}
    if hasattr(values, "numpy"):
        tensor = tf.convert_to_tensor(values)
        if tensor.dtype == tf.string:
            raw = tensor.numpy()
            return raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
        if tensor.shape.rank == 0:
            if tensor.dtype.is_integer:
                return int(tensor.numpy())
            if tensor.dtype == tf.bool:
                return bool(tensor.numpy())
            return float(tf.cast(tensor, DTYPE).numpy())
        return tensor_to_json(tf.cast(tensor, DTYPE))
    return values


def _validate_payload(payload: dict[str, Any]) -> None:
    required = {
        "decision",
        "target",
        "routes",
        "evidence_contract",
        "rows",
        "route_summaries",
        "veto_diagnostics",
        "run_manifest",
        "nonclaims",
    }
    missing = required.difference(payload)
    if missing:
        raise P2ValidationError(f"missing payload fields {sorted(missing)}")
    if payload["decision"] not in {
        "PASS_P2_TINY_NONLINEAR_DENSE_ORACLE_PENDING_CLAUDE_REVIEW",
        "P2_TINY_NONLINEAR_DENSE_ORACLE_VETO_PENDING_REVIEW",
    }:
        raise P2ValidationError(f"invalid P2 decision {payload['decision']}")
    manifest = payload["run_manifest"]
    for field in (
        "command",
        "pre_import_cuda_visible_devices",
        "json_path",
        "report_path",
        "plan_path",
        "result_path",
        "dense_orders",
        "dims",
    ):
        if field not in manifest:
            raise P2ValidationError(f"run_manifest missing {field}")
    if manifest["pre_import_cuda_visible_devices"] != "-1":
        raise P2ValidationError("TensorFlow was not forced CPU-only before import")
    if payload["routes"]["dpf_bootstrap_ot"]["phase_eligibility"]["p2"]:
        raise P2ValidationError("DPF bootstrap route unexpectedly P2-eligible")
    if payload["routes"]["dpf_ledh_pfpf_ot"]["phase_eligibility"]["p2"]:
        raise P2ValidationError("DPF LEDH route unexpectedly P2-eligible")
    if payload["veto_diagnostics"]["dpf_executed_or_p2_eligible"]:
        raise P2ValidationError("DPF execution or P2 eligibility appeared")
    if payload["veto_diagnostics"]["dense_reference_lacks_refinement_pass"]:
        raise P2ValidationError("dense reference refinement did not pass")
    if payload["veto_diagnostics"]["ukf_svd_cut4_called_oracle"]:
        raise P2ValidationError("UKF/SVD/CUT4 promoted to oracle")
    if payload["veto_diagnostics"]["directional_fd_diagnostic_large"]:
        raise P2ValidationError("directional FD diagnostic exceeded tolerance")
    if not payload["route_summaries"]["dense_refined_quadrature"]["promoted_dims"]:
        raise P2ValidationError("no dense oracle row promoted")
    for row in payload["rows"]:
        if row["dense_reference"]["claim_class"] != "EXACT_ORACLE":
            raise P2ValidationError("dense route is not exact-oracle class")
        if not row["dense_reference"]["initial_law_alignment"]["aligned"]:
            raise P2ValidationError("dense and structural initial laws are not aligned")
        if row["dense_reference"]["promotion_status"] != "promoted_dense_oracle_for_p2":
            raise P2ValidationError("dense row was not promoted")
        for route in row["deterministic_routes"]:
            if route["route_id"] in {"ukf", "svd_sigma_point", "cut4"}:
                if route["claim_class"] == "EXACT_ORACLE":
                    raise P2ValidationError("sigma-point route labeled exact oracle")
            if not route["finite"]:
                raise P2ValidationError("nonfinite deterministic route")
    if "reproducibility_digest" not in payload:
        raise P2ValidationError("missing reproducibility digest")


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# P2 Result: Tiny Nonlinear Dense-Oracle Comparison",
        "",
        "metadata_date: 2026-06-08",
        "phase: P2",
        f"status: {payload['decision']}",
        "",
        "## Skeptical Plan Audit",
        "",
        "Status: `PASS_FOR_P2_DENSE_ORACLE_EVIDENCE_RECORDING_PENDING_CLAUDE_REVIEW`.",
        "",
        "Wrong-baseline risk is controlled by using dense order-241 quadrature only after order-161 refinement passes; UKF, SVD/cubature, CUT4, and Zhao-Cui/fixed-design TT are not called ground truth.",
        "",
        "Proxy-promotion risk is controlled by treating finite values, point counts, and finite-difference checks as diagnostics. Dense refinement is the P2 oracle promotion criterion.",
        "",
        "Missing-stop-condition risk is controlled by explicit veto diagnostics for dense refinement failure, DPF execution in P2, non-P2 route execution, and non-oracle routes mislabeled as oracles.",
        "",
        "Unfair-comparison risk is controlled by deferring DPF stochastic rows to P5 because P0 marks DPF routes as p2=false and p5=true for this target.",
        "",
        "## Evidence Contract",
        "",
        "| Field | Contract |",
        "| --- | --- |",
        f"| Question | {payload['question']} |",
        f"| Baseline/comparator | {payload['evidence_contract']['baseline_comparator']} |",
        f"| Primary criterion | {payload['evidence_contract']['primary_criterion']} |",
        "| Veto diagnostics | Dense reference lacks refinement, DPF executed in P2, non-P2 route executed, UKF/SVD/CUT4 called oracle, nonfinite route, parameter mismatch, or single-step finite difference used as promotion. |",
        "| Not concluded | No DPF correctness, stochastic-resampling correctness, HMC readiness, production readiness, GPU claim, or paper-scale claim. |",
        "",
        "## Veto Diagnostics",
        "",
        "| Diagnostic | Status |",
        "| --- | --- |",
        *[
            f"| `{key}` | `{value}` |"
            for key, value in payload["veto_diagnostics"].items()
        ],
        "",
        "## Dense Reference Summary",
        "",
        "| dim | value | value refinement gap | directional score refinement gap | promoted |",
        "| ---: | ---: | ---: | ---: | --- |",
    ]
    for row in payload["rows"]:
        ref = row["dense_reference"]
        refinement = ref["refinement"]
        lines.append(
            f"| {row['dim']} | `{ref['value']}` | `{refinement['value_gap']}` | "
            f"`{refinement['max_directional_score_gap']}` | `{ref['promotion_status']}` |"
        )
    lines.extend(
        [
            "",
            "## Route Gap Summary",
            "",
            "| route | claim class | max abs value error | max directional score gap | max relative score error | interpretation |",
            "| --- | --- | ---: | ---: | ---: | --- |",
        ]
    )
    for route_id in ("ukf", "svd_sigma_point", "cut4", "zhao_cui_fixed_design_tt"):
        summary = payload["route_summaries"][route_id]
        lines.append(
            f"| `{route_id}` | `{summary['claim_class']}` | "
            f"`{summary['max_abs_value_error']}` | `{summary['max_directional_score_gap']}` | "
            f"`{summary['max_relative_score_error']}` | `{summary['interpretation']}` |"
        )
    lines.extend(
        [
            "",
            "## DPF Deferral",
            "",
            "- `dpf_bootstrap_ot`: deferred to P5 because P0 marks this target-route row `p2=false`, `p5=true`.",
            "- `dpf_ledh_pfpf_ot`: deferred to P5 because P0 marks this target-route row `p2=false`, `p5=true`.",
            "",
            "## Directional FD Diagnostics",
            "",
            "Finite differences are multi-step directional diagnostics only; they are not the promotion gate.",
            "",
            "| dim | max AD-vs-FD error | max step spread |",
            "| ---: | ---: | ---: |",
        ]
    )
    for row in payload["rows"]:
        checks = row["dense_reference"]["refinement"]["directional_fd_checks"]
        lines.append(
            f"| {row['dim']} | `{max(check['max_abs_error_vs_ad'] for check in checks)}` | "
            f"`{max(check['max_step_spread'] for check in checks)}` |"
        )
    lines.extend(
        [
            "",
            "## Decision Table",
            "",
            "| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | Not concluded |",
            "| --- | --- | --- | --- | --- | --- |",
            (
                f"| `{payload['decision']}` | dense reference promoted for dims "
                f"`{payload['route_summaries']['dense_refined_quadrature']['promoted_dims']}` | "
                f"`{payload['veto_diagnostics']}` | deterministic approximation errors are target/local only | "
                "run Claude read-only P2 gate review | no DPF, HMC, production, GPU, or paper-scale claim |"
            ),
            "",
            "## Post-Run Red-Team Note",
            "",
            "Strongest alternative explanation: the selected P44-M2 fixture is a tiny, factorized scalar panel, so the dense-oracle pass may not generalize to harder nonlinear closures.",
            "",
            "Result that would overturn the P2 pass: a reviewed rerun finds dense refinement failure, parameterization drift, or a route table that allows DPF execution in P2.",
            "",
            "Weakest part of the evidence: UKF/SVD/CUT4 are diagnostic or certified approximation rows only; their gap size does not establish DPF correctness.",
            "",
            "## Run Manifest",
            "",
            f"- command: `{payload['run_manifest']['command']}`",
            f"- git branch: `{payload['run_manifest']['branch']}`",
            f"- git commit: `{payload['run_manifest']['commit']}`",
            f"- dirty state summary: `{payload['run_manifest']['dirty_state_summary']}`",
            f"- environment/packages: `{payload['run_manifest']['package_versions']}`",
            f"- CPU/GPU status: CPU-only, `CUDA_VISIBLE_DEVICES={payload['run_manifest']['pre_import_cuda_visible_devices']}` before TensorFlow import",
            f"- visible GPU devices: `{payload['run_manifest']['gpu_devices_visible']}`",
            f"- target id: `{payload['run_manifest']['target_id']}`",
            f"- dims: `{payload['run_manifest']['dims']}`",
            f"- dense orders: `{payload['run_manifest']['dense_orders']}`",
            f"- FD steps: `{payload['run_manifest']['fd_steps']}`",
            f"- data version: `{payload['run_manifest']['data_version']}`",
            f"- plan: `{payload['run_manifest']['plan_path']}`",
            f"- review ledger: `{payload['run_manifest']['review_ledger_path']}`",
            f"- JSON: `{payload['run_manifest']['json_path']}`",
            f"- report: `{payload['run_manifest']['report_path']}`",
            f"- result: `{payload['run_manifest']['result_path']}`",
            f"- wall time seconds: `{payload['run_manifest']['wall_time_seconds']}`",
            "",
            "## Nonclaims",
            "",
            *[f"- {item}" for item in payload["nonclaims"]],
            "",
            "## Gate Status",
            "",
            "P2 is pending Claude read-only review.",
            "",
        ]
    )
    return "\n".join(lines)


def _nonclaims() -> list[str]:
    return [
        "DPF routes are not executed in P2 and receive no P2 correctness promotion.",
        "UKF, SVD/cubature, and CUT4 are not exact oracles on the nonlinear target.",
        "Zhao-Cui/fixed-design TT evidence is local certified approximation evidence only.",
        "Finite differences are diagnostic only and are not a promotion gate.",
        "P2 does not establish high-dimensional nonlinear correctness.",
        "P2 does not establish HMC readiness, production readiness, GPU readiness, or paper-scale claims.",
    ]


def _digest_payload(payload: dict[str, Any]) -> str:
    stable = {
        key: value
        for key, value in payload.items()
        if key not in {"created_at_utc", "run_manifest", "reproducibility_digest"}
    }
    return stable_digest(stable)


if __name__ == "__main__":
    sys.exit(main())
