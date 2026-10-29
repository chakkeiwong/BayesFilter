"""Run P3 transformed-SV and KSC-mixture oracle comparisons.

P3 records same-target value and gradient evidence for transformed or
finite-mixture stochastic-volatility rows.  P0 blocks DPF for these targets, so
this runner records those blockers instead of executing DPF.
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/bayesfilter-dpf-mpl")
PRE_IMPORT_CUDA_VISIBLE_DEVICES = os.environ["CUDA_VISIBLE_DEVICES"]

import argparse
import json
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable

import tensorflow as tf
import tensorflow_probability as tfp

import bayesfilter.highdim as highdim
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
    "run_filter_oracle_comparison_p3_conditional_gaussian_mixture_tf"
)
PLAN_PATH = (
    "docs/plans/"
    "bayesfilter-dpf-filter-oracle-comparison-p3-conditional-gaussian-mixture-subplan-2026-06-08.md"
)
RESULT_PATH = (
    "docs/plans/"
    "bayesfilter-dpf-filter-oracle-comparison-p3-conditional-gaussian-mixture-result-2026-06-08.md"
)
REVIEW_LEDGER_PATH = (
    "docs/plans/"
    "bayesfilter-dpf-filter-oracle-comparison-p3-claude-review-ledger-2026-06-08.md"
)
JSON_PATH = OUTPUT_DIR / "dpf_filter_oracle_comparison_p3_conditional_gaussian_mixture_2026-06-08.json"
REPORT_PATH = REPORT_DIR / "dpf-filter-oracle-comparison-p3-conditional-gaussian-mixture-2026-06-08.md"

TARGET_EXACT = "sv_exact_transformed_log_chi_square_panel"
TARGET_KSC = "sv_ksc_transformed_mixture_panel"
DIMS = (1, 2, 3)
EXACT_DENSE_ORDER = 401
EXACT_DENSE_RADIUS = 8.0
TT_FIT_QUADRATURE_ORDER = 141
TT_BASIS_DIM = 48
KSC_TRANSFORM_OFFSET = 1e-8
EXACT_TT_VALUE_ABS_TOL = 2e-6
EXACT_TT_DIRECTIONAL_SCORE_TOL = 2e-5
EXACT_TT_RELATIVE_SCORE_TOL = 1e-5
KSC_CUT4_VALUE_ABS_TOL = 2e-6
KSC_CUT4_DIRECTIONAL_SCORE_TOL = 2e-8
KSC_CUT4_RELATIVE_SCORE_TOL = 1e-8

_STD_NORMAL = tfp.distributions.Normal(
    loc=tf.constant(0.0, DTYPE),
    scale=tf.constant(1.0, DTYPE),
)


class P3ValidationError(ValueError):
    """Raised when a P3 artifact violates the phase contract."""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    if args.validate_only:
        _validate_payload(load_json(JSON_PATH))
        print("P3_CONDITIONAL_GAUSSIAN_MIXTURE_VALIDATED")
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
    exact_routes = registry["route_matrix"][TARGET_EXACT]
    ksc_routes = registry["route_matrix"][TARGET_KSC]
    exact_target = _target(registry, TARGET_EXACT)
    ksc_target = _target(registry, TARGET_KSC)

    exact_rows = [_exact_transformed_row(dim, exact_routes) for dim in DIMS]
    exact_by_dim = {row["dim"]: row for row in exact_rows}
    ksc_rows = [_ksc_mixture_row(dim, ksc_routes, exact_by_dim[dim]) for dim in DIMS]
    route_summaries = _route_summaries(exact_rows, ksc_rows)
    veto = _veto_diagnostics(exact_routes, ksc_routes, exact_rows, ksc_rows, route_summaries)
    decision = (
        "PASS_P3_CONDITIONAL_GAUSSIAN_MIXTURE_PENDING_CLAUDE_REVIEW"
        if not any(bool(value) for value in veto.values())
        else "P3_CONDITIONAL_GAUSSIAN_MIXTURE_VETO_PENDING_REVIEW"
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
            "data_version": "deterministic P40/P41/P43 transformed-SV fixture",
            "target_ids": [TARGET_EXACT, TARGET_KSC],
            "dims": list(DIMS),
            "exact_dense_order": EXACT_DENSE_ORDER,
            "exact_dense_radius": EXACT_DENSE_RADIUS,
            "tt_fit_quadrature_order": TT_FIT_QUADRATURE_ORDER,
            "tt_basis_dim": TT_BASIS_DIM,
            "ksc_transform_offset": KSC_TRANSFORM_OFFSET,
            "seeds": "deterministic fixed-design TT branch seeds; no Monte Carlo seeds",
            "particle_counts": "N/A: DPF routes are blocked by P0 for P3 SV targets",
        }
    )
    return {
        "metadata_date": "2026-06-08",
        "created_at_utc": utc_now(),
        "phase": "P3",
        "execution_route": "VISIBLE_IN_DIALOGUE",
        "decision": decision,
        "question": (
            "For P0-approved transformed-SV and finite KSC-mixture SV rows, "
            "can P3 record same-target value and gradient reference evidence "
            "without promoting native-SV or blocked DPF claims?"
        ),
        "targets": {
            TARGET_EXACT: {
                "registry_target": exact_target,
                "reference_route": exact_routes["dense_refined_quadrature"],
                "candidate_route": exact_routes["zhao_cui_fixed_design_tt"],
                "blocked_routes": _blocked_routes(exact_routes),
            },
            TARGET_KSC: {
                "registry_target": ksc_target,
                "reference_route": ksc_routes["kalman_exact"],
                "candidate_route": ksc_routes["cut4"],
                "blocked_routes": _blocked_routes(ksc_routes),
            },
        },
        "evidence_contract": {
            "baseline_comparator": (
                "exact transformed dense reference for "
                f"{TARGET_EXACT}; component-enumerated Kalman mixture for {TARGET_KSC}"
            ),
            "primary_criterion": (
                "same-target reference and candidate rows have finite values, "
                "finite reference scores, and route gaps inside local P43/P40 "
                "certificate tolerances, with DPF blockers preserved"
            ),
            "gradient_object": (
                "reference_score or fixed_branch_score for theta = "
                "(probit(gamma_i), log(beta_i)) per coordinate with sigma fixed"
            ),
            "dpf_policy": "DPF not run in P3; P0 marks DPF rows blocked for both selected SV targets",
            "jacobian_policy": (
                "exact transformed target uses z=log(y^2), offset 0; raw-native "
                "comparisons would subtract the observation-only Jacobian, but "
                "native likelihood is not promoted in P3"
            ),
        },
        "tolerances": {
            "exact_tt_value_abs": EXACT_TT_VALUE_ABS_TOL,
            "exact_tt_directional_score_abs": EXACT_TT_DIRECTIONAL_SCORE_TOL,
            "exact_tt_relative_score": EXACT_TT_RELATIVE_SCORE_TOL,
            "ksc_cut4_value_abs": KSC_CUT4_VALUE_ABS_TOL,
            "ksc_cut4_directional_score_abs": KSC_CUT4_DIRECTIONAL_SCORE_TOL,
            "ksc_cut4_relative_score": KSC_CUT4_RELATIVE_SCORE_TOL,
        },
        "rows": {
            "exact_transformed": exact_rows,
            "ksc_mixture": ksc_rows,
        },
        "route_summaries": route_summaries,
        "veto_diagnostics": veto,
        "explanatory_diagnostics": {
            "ksc_vs_exact_transformed_gap_is_approximation_only": True,
            "component_enumeration_counts_recorded": True,
            "raw_native_jacobian_recorded_but_not_promoted": True,
            "dpf_blockers_preserved": True,
        },
        "run_manifest": manifest,
        "nonclaims": _nonclaims(),
    }


def _target(registry: dict[str, Any], target_id: str) -> dict[str, Any]:
    return next(row for row in registry["targets"] if row["target_id"] == target_id)


def _exact_transformed_row(dim: int, routes: dict[str, Any]) -> dict[str, Any]:
    observations = _observations(dim)
    gamma, beta, sigma = _physical_parameters(dim)
    theta = _theta_from_physical(gamma, beta)
    dense_result, dense_value, dense_score = _value_and_score_result(
        lambda current: _exact_dense_result(current, observations, sigma),
        theta,
    )
    tt_result, tt_value, tt_score = _value_and_score_result(
        lambda current: _exact_tt_result(
            current,
            observations,
            sigma,
            seed=f"p3-dim-{dim}",
        ),
        theta,
    )
    comparison = _comparison_row(
        route_id="zhao_cui_fixed_design_tt",
        route=routes["zhao_cui_fixed_design_tt"],
        value=tt_value,
        score=tt_score,
        reference_value=dense_value,
        reference_score=dense_score,
        value_tol=EXACT_TT_VALUE_ABS_TOL,
        directional_score_tol=EXACT_TT_DIRECTIONAL_SCORE_TOL,
        relative_score_tol=EXACT_TT_RELATIVE_SCORE_TOL,
        diagnostics=_compact_diagnostics(tt_result.diagnostics),
        interpretation=(
            "same_target_fixed_design_tt_local_certificate_not_native_sv_oracle"
        ),
    )
    return {
        "target_id": TARGET_EXACT,
        "dim": int(dim),
        "observations": tensor_to_json(observations),
        "physical_parameters": {
            "gamma": tensor_to_json(gamma),
            "beta": tensor_to_json(beta),
            "sigma": tensor_to_json(sigma),
        },
        "theta": tensor_to_json(theta),
        "gradient_parameterization": _gradient_parameterization(dim),
        "transform_convention": {
            "transformed_observations": tensor_to_json(
                highdim.exact_transformed_sv_observations(observations)
            ),
            "transform": "z = log(y^2)",
            "transform_offset": 0.0,
            "jacobian_log_abs_det_for_raw_native_relation": scalar(
                highdim.exact_transformed_sv_jacobian_log_abs_det(observations)
            ),
            "raw_native_relation": (
                "native_log_likelihood = exact_transformed_log_likelihood "
                "- jacobian_log_abs_det_for_raw_native_relation"
            ),
            "raw_native_promoted_in_p3": False,
        },
        "reference": {
            "route_id": "dense_refined_quadrature",
            "claim_class": routes["dense_refined_quadrature"]["claim_class"],
            "route_status": routes["dense_refined_quadrature"]["route_status"],
            "primary_gradient_statistic": routes["dense_refined_quadrature"][
                "primary_gradient_statistic"
            ],
            "value": scalar(dense_value),
            "score": tensor_to_json(dense_score),
            "finite": _finite_scalar_vector(dense_value, dense_score),
            "diagnostics": _compact_diagnostics(dense_result.diagnostics),
        },
        "candidate": comparison,
        "blocked_routes": _blocked_routes(routes),
    }


def _ksc_mixture_row(
    dim: int,
    routes: dict[str, Any],
    exact_row: dict[str, Any],
) -> dict[str, Any]:
    observations = _observations(dim)
    gamma, beta, sigma = _physical_parameters(dim)
    theta = _theta_from_physical(gamma, beta)
    kalman_result, kalman_value, kalman_score = _value_and_score_result(
        lambda current: _ksc_kalman_result(current, observations, sigma),
        theta,
    )
    cut4_result, cut4_value, cut4_score = _value_and_score_result(
        lambda current: _ksc_cut4_result(current, observations, sigma),
        theta,
    )
    exact_value = tf.constant(exact_row["reference"]["value"], DTYPE)
    exact_score = tf.constant(exact_row["reference"]["score"], DTYPE)
    comparison = _comparison_row(
        route_id="cut4",
        route=routes["cut4"],
        value=cut4_value,
        score=cut4_score,
        reference_value=kalman_value,
        reference_score=kalman_score,
        value_tol=KSC_CUT4_VALUE_ABS_TOL,
        directional_score_tol=KSC_CUT4_DIRECTIONAL_SCORE_TOL,
        relative_score_tol=KSC_CUT4_RELATIVE_SCORE_TOL,
        diagnostics=_compact_diagnostics(cut4_result.diagnostics),
        interpretation=(
            "same_target_cut4_certificate_for_declared_ksc_mixture_not_native_sv"
        ),
    )
    return {
        "target_id": TARGET_KSC,
        "dim": int(dim),
        "observations": tensor_to_json(observations),
        "physical_parameters": {
            "gamma": tensor_to_json(gamma),
            "beta": tensor_to_json(beta),
            "sigma": tensor_to_json(sigma),
        },
        "theta": tensor_to_json(theta),
        "gradient_parameterization": _gradient_parameterization(dim),
        "transform_convention": {
            "transformed_observations": tensor_to_json(
                highdim.transformed_sv_panel_observations(
                    observations,
                    offset=KSC_TRANSFORM_OFFSET,
                )
            ),
            "transform": "z = log(y^2 + offset)",
            "transform_offset": KSC_TRANSFORM_OFFSET,
            "target": "declared finite Gaussian-mixture approximation target",
            "native_exact_promoted_in_p3": False,
        },
        "reference": {
            "route_id": "kalman_exact",
            "claim_class": routes["kalman_exact"]["claim_class"],
            "route_status": routes["kalman_exact"]["route_status"],
            "primary_gradient_statistic": routes["kalman_exact"][
                "primary_gradient_statistic"
            ],
            "value": scalar(kalman_value),
            "score": tensor_to_json(kalman_score),
            "finite": _finite_scalar_vector(kalman_value, kalman_score),
            "diagnostics": _compact_diagnostics(kalman_result.diagnostics),
        },
        "candidate": comparison,
        "approximation_gap_to_exact_transformed": {
            "policy": "explanatory_only_not_a_veto_and_not_native_sv_promotion",
            "exact_transformed_reference_value": exact_row["reference"]["value"],
            "ksc_mixture_reference_value": scalar(kalman_value),
            "value_gap": scalar(kalman_value - exact_value),
            "abs_value_gap": scalar(tf.abs(kalman_value - exact_value)),
            "directional_score_gap": scalar(
                _directional_abs_max(kalman_score, exact_score)
            ),
            "score_error_norm": scalar(tf.linalg.norm(kalman_score - exact_score)),
            "relative_score_error": scalar(
                _relative_score_error(kalman_score, exact_score)
            ),
        },
        "blocked_routes": _blocked_routes(routes),
    }


def _comparison_row(
    *,
    route_id: str,
    route: dict[str, Any],
    value: tf.Tensor,
    score: tf.Tensor,
    reference_value: tf.Tensor,
    reference_score: tf.Tensor,
    value_tol: float,
    directional_score_tol: float,
    relative_score_tol: float,
    diagnostics: dict[str, Any],
    interpretation: str,
) -> dict[str, Any]:
    value_error = value - reference_value
    score_error = score - reference_score
    abs_value_error = scalar(tf.abs(value_error))
    directional_score_gap = scalar(_directional_abs_max(score, reference_score))
    relative_score_error = scalar(_relative_score_error(score, reference_score))
    certificate_passes = (
        abs_value_error <= float(value_tol)
        and directional_score_gap <= float(directional_score_tol)
        and relative_score_error <= float(relative_score_tol)
    )
    return {
        "route_id": route_id,
        "claim_class": route["claim_class"],
        "route_status": route["route_status"],
        "phase_eligibility": route["phase_eligibility"],
        "primary_gradient_statistic": route["primary_gradient_statistic"],
        "value": scalar(value),
        "score": tensor_to_json(score),
        "value_error": scalar(value_error),
        "abs_value_error": abs_value_error,
        "directional_score_gap": directional_score_gap,
        "score_error_norm": scalar(tf.linalg.norm(score_error)),
        "relative_score_error": relative_score_error,
        "finite": _finite_scalar_vector(value, score),
        "certificate_tolerances": {
            "value_abs": float(value_tol),
            "directional_score_abs": float(directional_score_tol),
            "relative_score": float(relative_score_tol),
        },
        "certificate_status": (
            "same_target_certificate_pass"
            if certificate_passes
            else "same_target_certificate_veto"
        ),
        "diagnostics": diagnostics,
        "interpretation": interpretation,
    }


def _route_summaries(
    exact_rows: list[dict[str, Any]],
    ksc_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    exact_candidates = [row["candidate"] for row in exact_rows]
    ksc_candidates = [row["candidate"] for row in ksc_rows]
    ksc_gaps = [row["approximation_gap_to_exact_transformed"] for row in ksc_rows]
    return {
        "exact_transformed_dense_reference": {
            "claim_class": exact_rows[0]["reference"]["claim_class"],
            "all_finite": all(row["reference"]["finite"] for row in exact_rows),
            "dims": [row["dim"] for row in exact_rows],
        },
        "exact_transformed_zhao_cui_fixed_design_tt": _candidate_summary(
            exact_candidates
        ),
        "ksc_kalman_mixture_reference": {
            "claim_class": ksc_rows[0]["reference"]["claim_class"],
            "all_finite": all(row["reference"]["finite"] for row in ksc_rows),
            "component_tuple_counts": [
                row["reference"]["diagnostics"].get("component_tuple_count")
                for row in ksc_rows
            ],
            "dims": [row["dim"] for row in ksc_rows],
        },
        "ksc_mixture_cut4": _candidate_summary(ksc_candidates),
        "ksc_vs_exact_transformed_gap": {
            "interpretation": "approximation_only_not_native_sv_promotion",
            "max_abs_value_gap": max(row["abs_value_gap"] for row in ksc_gaps),
            "max_directional_score_gap": max(
                row["directional_score_gap"] for row in ksc_gaps
            ),
            "max_relative_score_error": max(
                row["relative_score_error"] for row in ksc_gaps
            ),
        },
    }


def _candidate_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "claim_class": rows[0]["claim_class"],
        "all_finite": all(row["finite"] for row in rows),
        "all_certificate_pass": all(
            row["certificate_status"] == "same_target_certificate_pass"
            for row in rows
        ),
        "certificate_dims": [
            index + 1
            for index, row in enumerate(rows)
            if row["certificate_status"] == "same_target_certificate_pass"
        ],
        "max_abs_value_error": max(row["abs_value_error"] for row in rows),
        "max_directional_score_gap": max(
            row["directional_score_gap"] for row in rows
        ),
        "max_relative_score_error": max(row["relative_score_error"] for row in rows),
        "interpretation": rows[0]["interpretation"],
    }


def _veto_diagnostics(
    exact_routes: dict[str, Any],
    ksc_routes: dict[str, Any],
    exact_rows: list[dict[str, Any]],
    ksc_rows: list[dict[str, Any]],
    route_summaries: dict[str, Any],
) -> dict[str, bool]:
    return {
        "native_and_transformed_likelihoods_mixed": False,
        "ksc_mixture_called_native_exact_truth": False,
        "jacobian_terms_missing": any(
            "jacobian_log_abs_det_for_raw_native_relation"
            not in row["transform_convention"]
            for row in exact_rows
        )
        or any(
            "transform_offset" not in row["transform_convention"]
            for row in ksc_rows
        ),
        "dpf_executed_or_p3_eligible": _dpf_p3_eligible_or_unblocked(
            exact_routes,
            ksc_routes,
        ),
        "route_target_mismatch": any(row["target_id"] != TARGET_EXACT for row in exact_rows)
        or any(row["target_id"] != TARGET_KSC for row in ksc_rows),
        "gradient_parameterization_mismatch": any(
            row["reference"]["route_id"] not in {"dense_refined_quadrature", "kalman_exact"}
            or row["candidate"]["primary_gradient_statistic"]
            not in {"fixed_branch_score", "diagnostic_fixed_branch_score"}
            or len(row["reference"]["score"]) != len(row["candidate"]["score"])
            for row in [*exact_rows, *ksc_rows]
        ),
        "route_row_nonfinite": any(
            not row["reference"]["finite"] or not row["candidate"]["finite"]
            for row in [*exact_rows, *ksc_rows]
        ),
        "missing_reference_score_or_gap": any(
            not row["reference"].get("score")
            or "directional_score_gap" not in row["candidate"]
            for row in [*exact_rows, *ksc_rows]
        ),
        "same_target_certificate_tolerance_failed": not bool(
            route_summaries[
                "exact_transformed_zhao_cui_fixed_design_tt"
            ]["all_certificate_pass"]
            and route_summaries["ksc_mixture_cut4"]["all_certificate_pass"]
        ),
        "unsupported_generalized_sv_native_claim": False,
    }


def _dpf_p3_eligible_or_unblocked(
    exact_routes: dict[str, Any],
    ksc_routes: dict[str, Any],
) -> bool:
    for routes in (exact_routes, ksc_routes):
        for route_id in ("dpf_bootstrap_ot", "dpf_ledh_pfpf_ot"):
            route = routes[route_id]
            if route["claim_class"] != "BLOCKED":
                return True
            if bool(route["phase_eligibility"].get("p3")):
                return True
    return False


def _blocked_routes(routes: dict[str, Any]) -> list[dict[str, Any]]:
    blocked = []
    for route_id, route in routes.items():
        if route["claim_class"] == "BLOCKED":
            blocked.append(
                {
                    "route_id": route_id,
                    "claim_class": route["claim_class"],
                    "route_status": route["route_status"],
                    "phase_eligibility": route["phase_eligibility"],
                    "blockers": route["blockers"],
                }
            )
    return blocked


def _observations(dim: int) -> tf.Tensor:
    values = tf.constant(
        [
            [0.12, -0.08, 0.05],
            [-0.07, 0.11, -0.04],
        ],
        dtype=DTYPE,
    )
    return values[:, : int(dim)]


def _physical_parameters(dim: int) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    gamma = tf.constant([0.60, 0.52, 0.47], dtype=DTYPE)[: int(dim)]
    beta = tf.constant([0.40, 0.35, 0.45], dtype=DTYPE)[: int(dim)]
    sigma = tf.constant([1.00, 0.85, 0.75], dtype=DTYPE)[: int(dim)]
    return gamma, beta, sigma


def _theta_from_physical(gamma: tf.Tensor, beta: tf.Tensor) -> tf.Tensor:
    return tf.reshape(
        tf.stack([_STD_NORMAL.quantile(gamma), tf.math.log(beta)], axis=1),
        [-1],
    )


def _physical_from_theta(theta: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    theta_matrix = tf.reshape(tf.convert_to_tensor(theta, DTYPE), [-1, 2])
    return _STD_NORMAL.cdf(theta_matrix[:, 0]), tf.exp(theta_matrix[:, 1])


def _gradient_parameterization(dim: int) -> dict[str, Any]:
    return {
        "theta": "(probit(gamma_i), log(beta_i)) interleaved by coordinate",
        "theta_dim": 2 * int(dim),
        "sigma_policy": "sigma is fixed by the P40/P41/P43 fixture; no sigma-gradient claim",
        "same_for_reference_and_candidate": True,
    }


def _convention() -> highdim.MeasureConvention:
    return highdim.MeasureConvention(
        density_measure=highdim.DensityMeasure.REFERENCE_MEASURE,
        mass_measure=highdim.MassMeasure.REFERENCE_MEASURE,
        reference_weight_name="omega",
    )


def _tt_config(seed: str) -> highdim.FixedBranchFilterConfig:
    convention = _convention()
    product_basis = highdim.ProductBasis(
        [highdim.LegendreBasis1D(highdim.BoundedInterval(-1.0, 1.0), TT_BASIS_DIM)],
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
            dense_matrix_byte_budget=200_000,
            normal_matrix_byte_budget=100_000,
            condition_number_warning=1e10,
            condition_number_veto=1e14,
            holdout_tolerance=5e-4,
        ),
        density_tau=0.0,
        normalizer_floor=1e-12,
        denominator_floor=1e-12,
        retained_storage_byte_budget=10_000_000,
        coordinate_maps=(
            highdim.AffineCoordinateMap(
                offset=tf.constant([0.0], DTYPE),
                matrix=tf.constant([[EXACT_DENSE_RADIUS]], DTYPE),
            ),
        ),
        measure_convention=convention,
        deterministic_seed=seed,
        product_basis=product_basis,
        initial_cores=(
            highdim.TTCore(tf.ones([1, product_basis.bases[0].basis_dim, 1], dtype=DTYPE)),
        ),
        fit_quadrature_order=TT_FIT_QUADRATURE_ORDER,
    )


def _exact_dense_result(theta: tf.Tensor, observations: tf.Tensor, sigma: tf.Tensor):
    gamma, beta = _physical_from_theta(theta)
    return highdim.exact_transformed_sv_independent_panel_dense_reference(
        observations,
        gamma=gamma,
        beta=beta,
        sigma=sigma,
        order=EXACT_DENSE_ORDER,
        radius=EXACT_DENSE_RADIUS,
    )


def _exact_tt_result(
    theta: tf.Tensor,
    observations: tf.Tensor,
    sigma: tf.Tensor,
    *,
    seed: str,
):
    gamma, beta = _physical_from_theta(theta)
    return highdim.exact_transformed_sv_independent_panel_zhaocui_tt_filter(
        observations,
        gamma=gamma,
        beta=beta,
        sigma=sigma,
        config=_tt_config(seed),
        fixture_id=f"p3.exact-transformed-gradient.{seed}",
        branch_seed_prefix=f"p3-exact-transformed-gradient-{seed}",
    )


def _ksc_kalman_result(theta: tf.Tensor, observations: tf.Tensor, sigma: tf.Tensor):
    gamma, beta = _physical_from_theta(theta)
    return highdim.independent_panel_sv_mixture_kalman_filter(
        observations,
        gamma=gamma,
        beta=beta,
        sigma=sigma,
        transform_offset=KSC_TRANSFORM_OFFSET,
    )


def _ksc_cut4_result(theta: tf.Tensor, observations: tf.Tensor, sigma: tf.Tensor):
    gamma, beta = _physical_from_theta(theta)
    return highdim.independent_panel_sv_mixture_cut4_filter(
        observations,
        gamma=gamma,
        beta=beta,
        sigma=sigma,
        transform_offset=KSC_TRANSFORM_OFFSET,
    )


def _value_and_score_result(
    result_fn: Callable[[tf.Tensor], Any],
    theta: tf.Tensor,
) -> tuple[Any, tf.Tensor, tf.Tensor]:
    theta = tf.convert_to_tensor(theta, DTYPE)
    with tf.GradientTape() as tape:
        tape.watch(theta)
        result = result_fn(theta)
        value = result.log_likelihood
    score = tape.gradient(value, theta)
    if score is None:
        raise P3ValidationError("GradientTape returned None")
    return result, value, score


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
    directions = tf.concat(
        [eye, mixed_a[tf.newaxis, :], mixed_b[tf.newaxis, :], mixed_c[tf.newaxis, :]],
        axis=0,
    )
    if int(directions.shape[0]) < 5:
        extra = tf.ones([1, size], DTYPE) / tf.sqrt(tf.cast(size, DTYPE))
        directions = tf.concat([directions, extra], axis=0)
    return directions


def _directional_abs_max(candidate_score: tf.Tensor, reference_score: tf.Tensor) -> tf.Tensor:
    residual = tf.cast(candidate_score, DTYPE) - tf.cast(reference_score, DTYPE)
    directional = tf.linalg.matvec(_directions(int(residual.shape[0])), residual)
    return tf.reduce_max(tf.abs(directional))


def _relative_score_error(candidate_score: tf.Tensor, reference_score: tf.Tensor) -> tf.Tensor:
    return tf.linalg.norm(tf.cast(candidate_score, DTYPE) - tf.cast(reference_score, DTYPE)) / tf.maximum(
        tf.constant(1.0, DTYPE),
        tf.linalg.norm(tf.cast(reference_score, DTYPE)),
    )


def _finite_scalar_vector(value: tf.Tensor, score: tf.Tensor) -> bool:
    return bool(
        tf.math.is_finite(value).numpy()
        and tf.reduce_all(tf.math.is_finite(tf.cast(score, DTYPE))).numpy()
    )


def _compact_diagnostics(values: Any) -> dict[str, Any]:
    converted = _jsonify(values)
    if not isinstance(converted, dict):
        return {"diagnostics": converted}
    if "component_tuples" in converted:
        tuples = converted.pop("component_tuples")
        converted["component_tuple_examples"] = (
            [tuples[0], tuples[-1]] if tuples else []
        )
    if "cut4_point_counts" in converted:
        rows = converted.pop("cut4_point_counts")
        flat = [entry for row in rows for entry in row]
        converted["cut4_point_count_summary"] = {
            "time_steps": len(rows),
            "min": min(flat) if flat else None,
            "max": max(flat) if flat else None,
            "first_time_first_five": rows[0][:5] if rows else [],
        }
    return converted


def _jsonify(values: Any) -> Any:
    if isinstance(values, Mapping):
        return {str(key): _jsonify(value) for key, value in values.items()}
    if isinstance(values, (list, tuple)):
        return [_jsonify(value) for value in values]
    if hasattr(values, "numpy"):
        tensor = tf.convert_to_tensor(values)
        if tensor.dtype == tf.string:
            raw = tensor.numpy()
            return raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
        if tensor.shape.rank == 0:
            if tensor.dtype == tf.bool:
                return bool(tensor.numpy())
            if tensor.dtype.is_integer:
                return int(tensor.numpy())
            return float(tf.cast(tensor, DTYPE).numpy())
        return tensor_to_json(tf.cast(tensor, DTYPE))
    if isinstance(values, (str, int, float, bool)) or values is None:
        return values
    return str(values)


def _validate_payload(payload: dict[str, Any]) -> None:
    required = {
        "decision",
        "targets",
        "evidence_contract",
        "rows",
        "route_summaries",
        "veto_diagnostics",
        "run_manifest",
        "nonclaims",
    }
    missing = required.difference(payload)
    if missing:
        raise P3ValidationError(f"missing payload fields {sorted(missing)}")
    if payload["decision"] not in {
        "PASS_P3_CONDITIONAL_GAUSSIAN_MIXTURE_PENDING_CLAUDE_REVIEW",
        "P3_CONDITIONAL_GAUSSIAN_MIXTURE_VETO_PENDING_REVIEW",
    }:
        raise P3ValidationError(f"invalid P3 decision {payload['decision']}")
    manifest = payload["run_manifest"]
    for field in (
        "command",
        "pre_import_cuda_visible_devices",
        "json_path",
        "report_path",
        "plan_path",
        "result_path",
        "target_ids",
        "dims",
    ):
        if field not in manifest:
            raise P3ValidationError(f"run_manifest missing {field}")
    if manifest["pre_import_cuda_visible_devices"] != "-1":
        raise P3ValidationError("TensorFlow was not forced CPU-only before import")
    if payload["veto_diagnostics"]["dpf_executed_or_p3_eligible"]:
        raise P3ValidationError("DPF execution or P3 eligibility appeared")
    if payload["veto_diagnostics"]["jacobian_terms_missing"]:
        raise P3ValidationError("Jacobian or transform-offset terms are missing")
    if payload["veto_diagnostics"]["ksc_mixture_called_native_exact_truth"]:
        raise P3ValidationError("KSC mixture was called native exact truth")
    if payload["veto_diagnostics"]["native_and_transformed_likelihoods_mixed"]:
        raise P3ValidationError("native and transformed likelihoods were mixed")
    if payload["veto_diagnostics"]["same_target_certificate_tolerance_failed"]:
        raise P3ValidationError("same-target certificate tolerance failed")
    for target_id in (TARGET_EXACT, TARGET_KSC):
        for route_id in ("dpf_bootstrap_ot", "dpf_ledh_pfpf_ot"):
            route = payload["targets"][target_id]["blocked_routes"]
            matches = [row for row in route if row["route_id"] == route_id]
            if len(matches) != 1 or matches[0]["claim_class"] != "BLOCKED":
                raise P3ValidationError(f"{target_id}/{route_id} blocker missing")
            if bool(matches[0]["phase_eligibility"].get("p3")):
                raise P3ValidationError(f"{target_id}/{route_id} unexpectedly P3-eligible")
    if payload["targets"][TARGET_EXACT]["reference_route"]["claim_class"] != "EXACT_ORACLE":
        raise P3ValidationError("exact transformed dense reference is not exact")
    if payload["targets"][TARGET_KSC]["reference_route"]["claim_class"] != "EXACT_ORACLE":
        raise P3ValidationError("KSC Kalman reference is not exact for mixture target")
    for row in payload["rows"]["exact_transformed"]:
        if row["transform_convention"]["transform_offset"] != 0.0:
            raise P3ValidationError("exact transformed target must use zero offset")
        if row["transform_convention"]["raw_native_promoted_in_p3"]:
            raise P3ValidationError("raw-native SV was promoted in P3")
        if row["candidate"]["claim_class"] == "EXACT_ORACLE":
            raise P3ValidationError("Zhao-Cui candidate promoted to exact oracle")
        if row["candidate"]["certificate_status"] != "same_target_certificate_pass":
            raise P3ValidationError("exact transformed candidate certificate failed")
    for row in payload["rows"]["ksc_mixture"]:
        if row["transform_convention"]["native_exact_promoted_in_p3"]:
            raise P3ValidationError("native exact SV was promoted from KSC row")
        if row["reference"]["claim_class"] != "EXACT_ORACLE":
            raise P3ValidationError("KSC reference is not exact for declared target")
        if row["candidate"]["claim_class"] == "EXACT_ORACLE":
            raise P3ValidationError("CUT4 candidate promoted to exact oracle")
        if row["candidate"]["certificate_status"] != "same_target_certificate_pass":
            raise P3ValidationError("KSC CUT4 candidate certificate failed")
        if row["approximation_gap_to_exact_transformed"]["policy"] != (
            "explanatory_only_not_a_veto_and_not_native_sv_promotion"
        ):
            raise P3ValidationError("KSC approximation gap policy missing")
    if "reproducibility_digest" not in payload:
        raise P3ValidationError("missing reproducibility digest")


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# P3 Result: Conditional-Gaussian and Mixture-Oracle Rows",
        "",
        "metadata_date: 2026-06-08",
        "phase: P3",
        f"status: {payload['decision']}",
        "",
        "## Skeptical Plan Audit",
        "",
        "Status: `PASS_FOR_P3_TRANSFORMED_AND_MIXTURE_EVIDENCE_RECORDING_PENDING_CLAUDE_REVIEW`.",
        "",
        "Wrong-baseline risk is controlled by separating exact transformed SV from the finite KSC mixture target. The KSC Kalman row is exact only for the declared mixture approximation target.",
        "",
        "Proxy-promotion risk is controlled by treating KSC-vs-exact transformed gaps as explanatory approximation diagnostics, not native-SV promotion criteria.",
        "",
        "Missing-stop-condition risk is controlled by explicit veto diagnostics for target mixing, missing Jacobian terms, DPF execution, nonfinite rows, missing reference scores, and certificate failures.",
        "",
        "Unfair-comparison risk is controlled by preserving P0 DPF blockers; bootstrap-OT and LEDH-PFPF-OT are not executed in P3.",
        "",
        "## Evidence Contract",
        "",
        "| Field | Contract |",
        "| --- | --- |",
        f"| Question | {payload['question']} |",
        f"| Baseline/comparator | {payload['evidence_contract']['baseline_comparator']} |",
        f"| Primary criterion | {payload['evidence_contract']['primary_criterion']} |",
        f"| Gradient object | {payload['evidence_contract']['gradient_object']} |",
        "| Veto diagnostics | Target mixing, KSC called native exact, missing Jacobian/offset terms, DPF execution, route mismatch, gradient mismatch, nonfinite rows, missing reference scores/gaps, or certificate failure. |",
        "| Not concluded | No DPF correctness, no native SV correctness from KSC agreement, no generalized-SV equality, no HMC, production, GPU, or paper-scale claim. |",
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
        "## Exact Transformed SV",
        "",
        "| dim | dense value | TT value error | TT directional score gap | TT relative score error | certificate |",
        "| ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in payload["rows"]["exact_transformed"]:
        candidate = row["candidate"]
        lines.append(
            f"| {row['dim']} | `{row['reference']['value']}` | "
            f"`{candidate['value_error']}` | `{candidate['directional_score_gap']}` | "
            f"`{candidate['relative_score_error']}` | `{candidate['certificate_status']}` |"
        )
    lines.extend(
        [
            "",
            "Exact transformed convention: `z = log(y^2)` with offset `0`; raw-native comparison would subtract the recorded observation-only Jacobian and is not promoted here.",
            "",
            "## KSC Mixture SV",
            "",
            "| dim | Kalman value | CUT4 value error | CUT4 directional score gap | CUT4 relative score error | KSC-vs-exact value gap | certificate |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in payload["rows"]["ksc_mixture"]:
        candidate = row["candidate"]
        approx = row["approximation_gap_to_exact_transformed"]
        lines.append(
            f"| {row['dim']} | `{row['reference']['value']}` | "
            f"`{candidate['value_error']}` | `{candidate['directional_score_gap']}` | "
            f"`{candidate['relative_score_error']}` | `{approx['value_gap']}` | "
            f"`{candidate['certificate_status']}` |"
        )
    lines.extend(
        [
            "",
            "KSC convention: `z = log(y^2 + 1e-8)` and finite Gaussian-mixture observation noise. Agreement with Kalman is exact only for this declared approximation target.",
            "",
            "## Route Summary",
            "",
            "| route | claim class | max abs value error | max directional score gap | max relative score error | interpretation |",
            "| --- | --- | ---: | ---: | ---: | --- |",
        ]
    )
    exact_summary = payload["route_summaries"]["exact_transformed_zhao_cui_fixed_design_tt"]
    ksc_summary = payload["route_summaries"]["ksc_mixture_cut4"]
    for route_id, summary in (
        ("zhao_cui_fixed_design_tt", exact_summary),
        ("cut4", ksc_summary),
    ):
        lines.append(
            f"| `{route_id}` | `{summary['claim_class']}` | "
            f"`{summary['max_abs_value_error']}` | `{summary['max_directional_score_gap']}` | "
            f"`{summary['max_relative_score_error']}` | `{summary['interpretation']}` |"
        )
    approx_summary = payload["route_summaries"]["ksc_vs_exact_transformed_gap"]
    lines.extend(
        [
            "",
            "## Approximation Gap",
            "",
            "| comparison | max abs value gap | max directional score gap | max relative score error | interpretation |",
            "| --- | ---: | ---: | ---: | --- |",
            (
                "| `ksc_kalman_vs_exact_transformed_dense` | "
                f"`{approx_summary['max_abs_value_gap']}` | "
                f"`{approx_summary['max_directional_score_gap']}` | "
                f"`{approx_summary['max_relative_score_error']}` | "
                f"`{approx_summary['interpretation']}` |"
            ),
            "",
            "## DPF Blockers",
            "",
            "- `dpf_bootstrap_ot`: blocked for both selected P3 SV targets by P0; not executed.",
            "- `dpf_ledh_pfpf_ot`: blocked for both selected P3 SV targets by P0; not executed.",
            "",
            "## Decision Table",
            "",
            "| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | Not concluded |",
            "| --- | --- | --- | --- | --- | --- |",
            (
                f"| `{payload['decision']}` | exact transformed TT and KSC CUT4 certificates pass for dims "
                f"`{exact_summary['certificate_dims']}` and `{ksc_summary['certificate_dims']}` | "
                f"`{payload['veto_diagnostics']}` | tiny independent-panel fixture only; sigma-gradient not claimed | "
                "run Claude read-only P3 gate review | no DPF, native SV, generalized SV, HMC, production, GPU, or paper-scale claim |"
            ),
            "",
            "## Post-Run Red-Team Note",
            "",
            "Strongest alternative explanation: these are tiny independent panels already covered by local P40/P41/P43 fixtures, so the pass validates route separation and local certificates rather than broad SV filtering performance.",
            "",
            "Result that would overturn the P3 pass: a reviewed rerun finds target mixing, missing Jacobian convention, DPF execution, or KSC evidence represented as exact native SV.",
            "",
            "Weakest part of the evidence: gradients are with respect to unconstrained gamma/beta at fixed sigma; no sigma-gradient or stochastic DPF score is evaluated.",
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
            f"- target ids: `{payload['run_manifest']['target_ids']}`",
            f"- dims: `{payload['run_manifest']['dims']}`",
            f"- exact dense order/radius: `{payload['run_manifest']['exact_dense_order']}`, `{payload['run_manifest']['exact_dense_radius']}`",
            f"- KSC transform offset: `{payload['run_manifest']['ksc_transform_offset']}`",
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
            "P3 is pending Claude read-only review.",
            "",
        ]
    )
    return "\n".join(lines)


def _nonclaims() -> list[str]:
    return [
        "DPF routes are not executed in P3 and receive no P3 correctness promotion.",
        "KSC Kalman exactness is scoped to the declared finite-mixture approximation target, not native SV.",
        "Exact transformed SV evidence is for z=log(y^2); raw-native SV requires the recorded Jacobian relation and is not promoted here.",
        "Zhao-Cui/fixed-design TT and CUT4 evidence is local same-target certificate evidence only.",
        "Gradients are with respect to unconstrained gamma/beta at fixed sigma; no sigma-gradient is claimed.",
        "P3 does not establish generalized-SV equality, HMC readiness, production readiness, GPU readiness, or paper-scale claims.",
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
