#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tensorflow as tf
import tensorflow_probability as tfp
import bayesfilter.highdim as highdim
from bayesfilter.linear.kalman_tf import tf_linear_gaussian_log_likelihood
from bayesfilter.nonlinear.fixed_sgqf_derivatives_tf import (
    TFFixedSGQFDerivatives,
    tf_fixed_sgqf_score,
)
from bayesfilter.nonlinear.fixed_sgqf_structural_adapter_tf import (
    tf_predator_prey_to_fixed_sgqf_model,
)
from bayesfilter.nonlinear.fixed_sgqf_tf import (
    TFFixedSGQFBranchConfig,
    TFFixedSGQFAffineModel,
    TFFixedSGQFNonlinearModel,
    tf_fixed_sgqf_cloud,
    tf_fixed_sgqf_filter,
)
from bayesfilter.structural import StatePartition
from bayesfilter.structural_tf import affine_structural_to_linear_gaussian_tf, make_affine_structural_tf
from scripts.filtering_value_gradient_benchmark_generate_p8_datasets import (
    _generalized_sv_prior_mean_dataset,
    _lgssm_dataset,
    _parameterized_sir_dataset,
    _sv_dataset,
)

ROOT = Path(__file__).resolve().parents[2]
SOURCE_NUMERIC = ROOT / "docs/plans/bayesfilter-filtering-value-gradient-benchmark-p8d-numeric-results-2026-06-13.json"
SOURCE_SCOPE = ROOT / "docs/plans/bayesfilter-filtering-value-gradient-benchmark-source-paper-scope-contract-2026-06-11.json"
MASTER_PROGRAM = "docs/plans/bayesfilter-two-lane-filter-comparison-master-program-2026-06-24.md"
DATE = "2026-07-01"
DEFAULT_JSON = ROOT / f"docs/plans/bayesfilter-two-lane-highdim-leaderboard-results-{DATE}.json"
DEFAULT_MD = ROOT / f"docs/plans/bayesfilter-two-lane-highdim-leaderboard-results-{DATE}.md"
CURRENT_AUTH_JSON = ROOT / f"docs/plans/bayesfilter-two-lane-highdim-leaderboard-results-{DATE}.json"
DTYPE = tf.float64
P91_FINAL_DECISION = ROOT / "docs/plans/bayesfilter-highdim-zhao-cui-p91-phase9-final-decision-result-2026-06-29.md"
P91_SCORE_IDENTITY = ROOT / "docs/plans/bayesfilter-highdim-zhao-cui-p91-phase4-score-identity-manifest-2026-06-29.json"
P91_GPU_XLA = ROOT / "docs/plans/bayesfilter-highdim-zhao-cui-p91-phase5-gpu-xla-jit-manifest-2026-06-29.json"
P91_BENCHMARK = ROOT / "docs/plans/bayesfilter-highdim-zhao-cui-p91-phase6-performance-benchmark-2026-06-29.json"
P91_HMC_SMOKE = ROOT / "docs/plans/bayesfilter-highdim-zhao-cui-p91-phase7-hmc-smoke-2026-06-29.json"
OMITTED_ALGORITHMS = [
    "bootstrap_dpf_current",
    "ledh_pfpf_alg1_ukf_current",
    "ledh_pfpf_ot",
]

HIGHDIM_ALGOS = [
    "fixed_sgqf",
    "ukf",
    "zhao_cui_scalar_or_multistate",
]
FIXED_SIR_ROW = "zhao_cui_spatial_sir_austria_j9_T20"
PARAMETERIZED_SIR_ROW = "zhao_cui_spatial_sir_austria_j9_T20_parameterized_logscale"
HIGHDIM_ROWS = [
    "benchmark_lgssm_exact_oracle_m3_T50",
    "zhao_cui_sv_actual_nongaussian_T1000",
    "zhao_cui_sv_ksc_gaussian_mixture_surrogate_T1000",
    FIXED_SIR_ROW,
    PARAMETERIZED_SIR_ROW,
    "zhao_cui_predator_prey_T20",
    "zhao_cui_generalized_sv_synthetic_from_estimated_values",
]


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _same_value_score_loglik(
    value_loglik: Any,
    score_loglik: Any,
    route_label: str,
) -> tuple[float | None, str | None]:
    """Return the score-route scalar only when it matches the value-route scalar."""

    value = float(tf.convert_to_tensor(value_loglik, dtype=DTYPE).numpy())
    score = float(tf.convert_to_tensor(score_loglik, dtype=DTYPE).numpy())
    if not math.isfinite(value):
        return None, f"{route_label} value-route log likelihood was non-finite"
    if not math.isfinite(score):
        return None, f"{route_label} score-route log likelihood was non-finite"
    tolerance = 1e-8 * max(1.0, abs(value), abs(score))
    if abs(value - score) > tolerance:
        return None, (
            f"{route_label} value/score route mismatch: value loglik={value:.17g}, "
            f"score loglik={score:.17g}, tolerance={tolerance:.3g}"
        )
    return score, None


def _row_summary_from_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    row_summary = []
    for row_id in HIGHDIM_ROWS:
        row_cells = [row for row in rows if row["row_id"] == row_id]
        executed = [row["algorithm_id"] for row in row_cells if row["comparison_status"].startswith("executed")]
        score_admitted = [row["algorithm_id"] for row in row_cells if row["comparison_status"] == "executed_value_score"]
        scoped_component_row = any(
            row.get("row_admission_status") == "scoped_component_row_admitted"
            for row in row_cells
        )
        full_ready = (not scoped_component_row) and set(score_admitted) == set(HIGHDIM_ALGOS)
        row_summary.append(
            {
                "row_id": row_id,
                "row_scope": (
                    "scoped_component_row"
                    if scoped_component_row
                    else "main_observed_data_filtering_row"
                ),
                "executed_algorithms": executed,
                "score_admitted_algorithms": score_admitted,
                "full_three_way_ready": full_ready,
                "scoped_component_ready": scoped_component_row and bool(score_admitted),
                "blocked_or_missing_algorithms": [
                    row["algorithm_id"]
                    for row in row_cells
                    if not row["comparison_status"].startswith("executed")
                ],
            }
        )
    return row_summary


def _value_score_route_id(row: dict[str, Any]) -> str | None:
    """Return the single scalar route that an admitted value/score row uses."""

    if row.get("comparison_status") != "executed_value_score":
        return None
    key = (row.get("algorithm_id"), row.get("row_id"))
    route_ids = {
        ("fixed_sgqf", "benchmark_lgssm_exact_oracle_m3_T50"): (
            "fixed_sgqf_direct_affine_lgssm"
        ),
        ("ukf", "benchmark_lgssm_exact_oracle_m3_T50"): (
            "ukf_lgssm_affine_equivalence_differentiated_kalman"
        ),
        ("zhao_cui_scalar_or_multistate", "benchmark_lgssm_exact_oracle_m3_T50"): (
            "zhao_cui_lgssm_exact_oracle_affine_adapter"
        ),
        ("fixed_sgqf", "zhao_cui_sv_actual_nongaussian_T1000"): (
            "fixed_sgqf_direct_exact_transformed_sv"
        ),
        ("ukf", "zhao_cui_sv_actual_nongaussian_T1000"): (
            "actual_sv_augmented_noise_factor_propagating_srukf"
        ),
        ("zhao_cui_scalar_or_multistate", "zhao_cui_sv_actual_nongaussian_T1000"): (
            "zhao_cui_exact_transformed_sv_fixed_branch_tt"
        ),
        ("fixed_sgqf", "zhao_cui_sv_ksc_gaussian_mixture_surrogate_T1000"): (
            "fixed_sgqf_independent_panel_ksc_mixture"
        ),
        ("ukf", "zhao_cui_sv_ksc_gaussian_mixture_surrogate_T1000"): (
            "principal_sqrt_ukf_independent_panel_ksc_mixture"
        ),
        ("zhao_cui_scalar_or_multistate", "zhao_cui_sv_ksc_gaussian_mixture_surrogate_T1000"): (
            "zhao_cui_ksc_mixture_fixed_branch_tt"
        ),
        ("fixed_sgqf", "zhao_cui_predator_prey_T20"): (
            "fixed_sgqf_direct_predator_prey_t20"
        ),
        ("zhao_cui_scalar_or_multistate", "zhao_cui_predator_prey_T20"): (
            "zhao_cui_predator_prey_t20_multistate_fixed_design_tt"
        ),
        ("zhao_cui_scalar_or_multistate", "zhao_cui_generalized_sv_synthetic_from_estimated_values"): (
            "zhao_cui_generalized_sv_prior_mean_scalar_fixed_design_tt"
        ),
        ("zhao_cui_scalar_or_multistate", PARAMETERIZED_SIR_ROW): (
            "zhao_cui_sir_d18_local_complete_data_manual_component"
        ),
    }
    return route_ids.get(key)


def _apply_value_score_route_contract(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    updated: list[dict[str, Any]] = []
    for row in rows:
        row = dict(row)
        route_id = _value_score_route_id(row)
        if route_id is not None:
            row["value_route_id"] = route_id
            row["score_route_id"] = route_id
            row["value_score_route_status"] = "same_route_value_score"
        updated.append(row)
    return updated


def _zhao_cui_hmc_route_is_admissible(row: dict[str, Any]) -> bool:
    """Require a fixed-variant identity before HMC score admission.

    Zhao-Cui has multiple model-specific fixed-branch analytical adapters.  A
    route does not have to use the high-dimensional source-route implementation,
    but it must identify its actual fixed value/score program.  Generic
    retained-grid and adaptive routes remain ineligible.
    """

    return (
        row.get("algorithm_id") == "zhao_cui_scalar_or_multistate"
        and row.get("route_id") in highdim.ZHAO_CUI_FIXED_VARIANT_ROUTE_IDS
        and row.get("hmc_route_policy_id") == highdim.ZHAO_CUI_HMC_ROUTE_POLICY_ID
        and row.get("hmc_target_scope_admitted") is True
    )


def _normalize_legacy_zhao_cui_route_identity(row: dict[str, Any]) -> dict[str, Any]:
    """Issue route identity for exact known pre-identity leaderboard rows.

    The July 1/3 artifacts predate explicit route fields.  Their exact row,
    numeric-status, and analytical-provenance tuple is repository-owned legacy
    evidence; arbitrary caller fields are not accepted as identity.
    """

    if row.get("algorithm_id") != "zhao_cui_scalar_or_multistate":
        return row
    key = (
        row.get("row_id"),
        row.get("numeric_execution_status"),
        row.get("score_derivative_provenance"),
    )
    known = {
        (
            "zhao_cui_sv_actual_nongaussian_T1000",
            "executed_zhao_cui_exact_transformed_sv_fixed_branch_tt_value_score",
            "zhao_cui_scalar_fixed_branch_tt_exact_transformed_sv_manual_parameter_score_methods_only",
        ): "zhao_cui_exact_transformed_sv_fixed_branch_tt",
        (
            "zhao_cui_sv_ksc_gaussian_mixture_surrogate_T1000",
            "executed_zhao_cui_ksc_fixed_branch_tt_value_score",
            "zhao_cui_scalar_fixed_branch_tt_ksc_mixture_manual_parameter_score_methods_only",
        ): "zhao_cui_ksc_mixture_fixed_branch_tt",
        (
            "zhao_cui_generalized_sv_synthetic_from_estimated_values",
            "executed_zhao_cui_generalized_sv_prior_mean_scalar_tt_value_score",
            "zhao_cui_generalized_sv_prior_mean_scalar_fixed_design_tt_manual_parameter_score_methods_only",
        ): "zhao_cui_generalized_sv_prior_mean_scalar_fixed_design_tt",
        (
            PARAMETERIZED_SIR_ROW,
            "executed_zhao_cui_parameterized_sir_t20_local_complete_data_value_score",
            "zhao_cui_sir_d18_local_complete_data_manual_parameter_score_methods",
        ): "zhao_cui_sir_d18_local_complete_data_manual_component",
    }
    route_id = known.get(key)
    if route_id is None:
        return row
    row = dict(row)
    row["route_id"] = route_id
    row["route_role"] = (
        highdim.ZHAO_CUI_FIXED_VARIANT_ROUTE_ID
        if route_id == "zhao_cui_sir_d18_local_complete_data_manual_component"
        else "fixed_variant_analytical_hmc_route"
    )
    row["hmc_route_policy_id"] = highdim.ZHAO_CUI_HMC_ROUTE_POLICY_ID
    row["hmc_target_scope_admitted"] = True
    return row


def _demote_zhao_cui_historical_score(row: dict[str, Any]) -> dict[str, Any]:
    """Keep historical value evidence, but remove its HMC score admission."""

    row["comparison_status"] = "executed_value_only"
    row["numeric_execution_status"] = (
        "executed_numeric_value_only_historical_zhao_cui_route_not_hmc_admitted"
    )
    row["score_status"] = "blocked_historical_zhao_cui_route_not_hmc_admitted"
    row["score_status_reason"] = (
        "Zhao-Cui HMC/leaderboard score admission requires the fixed-variant "
        "source-route identity; this evaluator is historical/diagnostic only"
    )
    row["reason"] = row["score_status_reason"]
    row["score"] = None
    row["score_l2_norm"] = None
    row["score_coordinate_system"] = None
    row["route_role"] = highdim.ZHAO_CUI_HISTORICAL_ROUTE_STATUS
    row["hmc_route_policy_id"] = None
    row["hmc_target_scope_admitted"] = False
    row["nonclaims"] = list(row.get("nonclaims", [])) + [
        "historical/diagnostic Zhao-Cui route; not the fixed-variant HMC evaluator",
        "its analytical score is not admitted for HMC or leaderboard use",
    ]
    row["reason_codes"] = list(row.get("reason_codes", [])) + [
        "ZHAO_CUI_NONFIXED_ROUTE_DEMOTED_HISTORICAL",
        "HMC_REQUIRES_FIXED_VARIANT_SOURCE_ROUTE",
    ]
    return row


def _status_from_manifest(manifest: dict[str, Any]) -> str | None:
    status = manifest.get("status")
    if isinstance(status, str):
        return status
    payload = manifest.get("payload")
    if isinstance(payload, dict):
        payload_status = payload.get("status")
        if isinstance(payload_status, str):
            return payload_status
        if isinstance(payload_status, dict) and isinstance(payload_status.get("value"), str):
            return payload_status["value"]
    return None


def _p8d_cells() -> dict[tuple[str, str], dict[str, Any]]:
    artifact = _load(SOURCE_NUMERIC)
    return {(cell["algorithm_id"], cell["model_row_id"]): cell for cell in artifact["cells"]}


def _runtime(cell: dict[str, Any]) -> float | None:
    value = cell.get("runtime_seconds")
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _normal01() -> tfp.distributions.Normal:
    return tfp.distributions.Normal(
        loc=tf.constant(0.0, dtype=DTYPE),
        scale=tf.constant(1.0, dtype=DTYPE),
    )


def _lgssm_theta() -> tf.Tensor:
    return tf.constant([0.72, 0.55, 0.35, 0.35, 0.45], dtype=DTYPE)


def _lgssm_observations() -> tf.Tensor:
    return tf.convert_to_tensor(_lgssm_dataset(81100)["observations"], dtype=DTYPE)


def _sv_theta() -> tf.Tensor:
    model = highdim.StochasticVolatilitySSM(sigma=1.0)
    return model.unconstrained_from_physical(gamma=0.6, beta=0.4)


def _sv_observations() -> tf.Tensor:
    return tf.convert_to_tensor(_sv_dataset(81101)["observations"], dtype=DTYPE)


def _predator_prey_theta() -> tf.Tensor:
    return tf.convert_to_tensor(
        highdim.p30_predator_prey_fixture_model().true_parameters(),
        dtype=DTYPE,
    )


def _predator_prey_observations() -> tf.Tensor:
    from scripts.filtering_value_gradient_benchmark_generate_p8_datasets import (
        _predator_prey_dataset,
    )

    return tf.convert_to_tensor(_predator_prey_dataset(81104)["observations"], dtype=DTYPE)


def _parameterized_sir_dataset_t20() -> dict[str, Any]:
    return _parameterized_sir_dataset(81103)


def _generalized_sv_theta() -> tf.Tensor:
    return tf.constant(
        _generalized_sv_prior_mean_dataset(81105)["truth_theta"],
        dtype=DTYPE,
    )


def _generalized_sv_observations() -> tf.Tensor:
    return tf.convert_to_tensor(
        _generalized_sv_prior_mean_dataset(81105)["observations"],
        dtype=DTYPE,
    )


def _zhao_cui_scalar_tt_config(seed: str, *, basis_degree: int = 48, quadrature_order: int = 141) -> highdim.FixedBranchFilterConfig:
    convention = highdim.MeasureConvention(
        density_measure=highdim.DensityMeasure.REFERENCE_MEASURE,
        mass_measure=highdim.MassMeasure.REFERENCE_MEASURE,
        reference_weight_name="omega",
    )
    product_basis = highdim.ProductBasis(
        [highdim.LegendreBasis1D(highdim.BoundedInterval(-1.0, 1.0), int(basis_degree))],
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
                offset=tf.constant([0.0], dtype=DTYPE),
                matrix=tf.constant([[8.0]], dtype=DTYPE),
            ),
        ),
        measure_convention=convention,
        deterministic_seed=seed,
        product_basis=product_basis,
        initial_cores=(
            highdim.TTCore(tf.ones([1, product_basis.bases[0].basis_dim, 1], dtype=DTYPE)),
        ),
        fit_quadrature_order=int(quadrature_order),
    )


def _zhao_cui_predator_prey_multistate_tt_config(
    seed: str,
    *,
    basis_degree: int = 4,
    quadrature_order: int = 5,
) -> highdim.FixedBranchFilterConfig:
    convention = highdim.MeasureConvention(
        density_measure=highdim.DensityMeasure.REFERENCE_MEASURE,
        mass_measure=highdim.MassMeasure.REFERENCE_MEASURE,
        reference_weight_name="omega",
    )
    product_basis = highdim.ProductBasis(
        [
            highdim.LegendreBasis1D(highdim.BoundedInterval(-1.0, 1.0), int(basis_degree)),
            highdim.LegendreBasis1D(highdim.BoundedInterval(-1.0, 1.0), int(basis_degree)),
        ],
        convention,
    )
    ranks = (1, 2, 1)
    return highdim.FixedBranchFilterConfig(
        fit_config=highdim.FixedTTFitConfig(
            ranks=ranks,
            ridge=1e-6,
            max_sweeps=3,
            sweep_order=(0, 1),
            row_budget=900,
            column_budget=180,
            dense_matrix_byte_budget=800_000,
            normal_matrix_byte_budget=200_000,
            condition_number_warning=1e20,
            condition_number_veto=1e30,
            holdout_tolerance=1.0,
        ),
        density_tau=1e-12,
        normalizer_floor=1e-12,
        denominator_floor=1e-12,
        retained_storage_byte_budget=10_000_000,
        coordinate_maps=(
            highdim.AffineCoordinateMap(
                offset=tf.constant([85.0, 1.0], dtype=DTYPE),
                matrix=tf.linalg.diag(tf.constant([55.0, 12.0], dtype=DTYPE)),
            ),
        ),
        measure_convention=convention,
        deterministic_seed=seed,
        product_basis=product_basis,
        initial_cores=tuple(
            highdim.TTCore(
                tf.ones(
                    [ranks[axis], basis.basis_dim, ranks[axis + 1]],
                    dtype=DTYPE,
                )
            )
            for axis, basis in enumerate(product_basis.bases)
        ),
        fit_quadrature_order=int(quadrature_order),
    )


def _lgssm_structural(theta: tf.Tensor):
    phi = theta[:3]
    q_scale = theta[3]
    r_scale = theta[4]
    observation_matrix = tf.constant(
        [
            [1.0, 0.25, -0.15],
            [0.2, 1.1, 0.3],
            [-0.1, 0.35, 0.9],
        ],
        dtype=DTYPE,
    )
    partition = StatePartition(
        state_names=("x1", "x2", "x3"),
        stochastic_indices=(0, 1, 2),
        deterministic_indices=(),
        innovation_dim=3,
    )
    return make_affine_structural_tf(
        partition=partition,
        initial_mean=tf.zeros([3], dtype=DTYPE),
        initial_covariance=tf.linalg.diag(tf.square(q_scale) / (1.0 - tf.square(phi))),
        transition_offset=tf.zeros([3], dtype=DTYPE),
        transition_matrix=tf.linalg.diag(phi),
        innovation_matrix=q_scale * tf.eye(3, dtype=DTYPE),
        innovation_covariance=tf.eye(3, dtype=DTYPE),
        observation_offset=tf.zeros([3], dtype=DTYPE),
        observation_matrix=observation_matrix,
        observation_covariance=tf.square(r_scale) * tf.eye(3, dtype=DTYPE),
        name="p8d_lgssm_exact_oracle_m3_structural_adapter",
    )


def _lgssm_sgqf_model_and_derivatives(theta: tf.Tensor) -> tuple[TFFixedSGQFNonlinearModel, TFFixedSGQFDerivatives]:
    theta = tf.convert_to_tensor(theta, dtype=DTYPE)
    structural = _lgssm_structural(theta)
    linear = affine_structural_to_linear_gaussian_tf(structural)
    phi = theta[:3]
    q_scale = theta[3]
    r_scale = theta[4]
    observation_matrix = tf.constant(
        [
            [1.0, 0.25, -0.15],
            [0.2, 1.1, 0.3],
            [-0.1, 0.35, 0.9],
        ],
        dtype=DTYPE,
    )
    transition_matrix = tf.linalg.diag(phi)
    model = TFFixedSGQFNonlinearModel(
        initial_mean=linear.initial_mean,
        initial_covariance=linear.initial_covariance,
        process_covariance=linear.transition_covariance,
        observation_covariance=linear.observation_covariance,
        transition_fn=lambda points: tf.linalg.matmul(
            tf.convert_to_tensor(points, dtype=DTYPE),
            transition_matrix,
            transpose_b=True,
        ),
        observation_fn=lambda points: tf.linalg.matmul(
            tf.convert_to_tensor(points, dtype=DTYPE),
            observation_matrix,
            transpose_b=True,
        ),
        name="two_lane_highdim_lgssm_fixed_sgqf_analytic_score_model",
    )

    d_initial_mean = tf.zeros([5, 3], dtype=DTYPE)
    d_initial_covariance = tf.zeros([5, 3, 3], dtype=DTYPE)
    for index in range(3):
        d_initial_covariance = tf.tensor_scatter_nd_update(
            d_initial_covariance,
            [[index, index, index]],
            [2.0 * phi[index] * tf.square(q_scale) / tf.square(1.0 - tf.square(phi[index]))],
        )
        d_initial_covariance = tf.tensor_scatter_nd_update(
            d_initial_covariance,
            [[3, index, index]],
            [2.0 * q_scale / (1.0 - tf.square(phi[index]))],
        )
    d_process_covariance = tf.zeros([5, 3, 3], dtype=DTYPE)
    d_observation_covariance = tf.zeros([5, 3, 3], dtype=DTYPE)
    for index in range(3):
        d_process_covariance = tf.tensor_scatter_nd_update(
            d_process_covariance,
            [[3, index, index]],
            [2.0 * q_scale],
        )
        d_observation_covariance = tf.tensor_scatter_nd_update(
            d_observation_covariance,
            [[4, index, index]],
            [2.0 * r_scale],
        )

    def transition_state_jacobian_fn(points: tf.Tensor) -> tf.Tensor:
        values = tf.convert_to_tensor(points, dtype=DTYPE)
        if values.shape.rank == 1:
            values = values[tf.newaxis, :]
        return tf.broadcast_to(transition_matrix, [tf.shape(values)[0], 3, 3])

    def d_transition_fn(points: tf.Tensor) -> tf.Tensor:
        values = tf.convert_to_tensor(points, dtype=DTYPE)
        if values.shape.rank == 1:
            values = values[tf.newaxis, :]
        point_count = tf.shape(values)[0]
        derivatives = []
        for index in range(3):
            component = tf.zeros_like(values)
            component = tf.tensor_scatter_nd_update(
                component,
                tf.stack([tf.range(point_count, dtype=tf.int32), tf.fill([point_count], index)], axis=1),
                values[:, index],
            )
            derivatives.append(component)
        derivatives.extend([tf.zeros_like(values), tf.zeros_like(values)])
        return tf.stack(derivatives, axis=0)

    def observation_state_jacobian_fn(points: tf.Tensor) -> tf.Tensor:
        values = tf.convert_to_tensor(points, dtype=DTYPE)
        if values.shape.rank == 1:
            values = values[tf.newaxis, :]
        return tf.broadcast_to(observation_matrix, [tf.shape(values)[0], 3, 3])

    def d_observation_fn(points: tf.Tensor) -> tf.Tensor:
        values = tf.convert_to_tensor(points, dtype=DTYPE)
        if values.shape.rank == 1:
            values = values[tf.newaxis, :]
        return tf.zeros([5, tf.shape(values)[0], 3], dtype=DTYPE)

    derivatives = TFFixedSGQFDerivatives(
        d_initial_mean=d_initial_mean,
        d_initial_covariance=d_initial_covariance,
        d_process_covariance=d_process_covariance,
        d_observation_covariance=d_observation_covariance,
        transition_state_jacobian_fn=transition_state_jacobian_fn,
        d_transition_fn=d_transition_fn,
        observation_state_jacobian_fn=observation_state_jacobian_fn,
        d_observation_fn=d_observation_fn,
        name="two_lane_highdim_lgssm_fixed_sgqf_analytic_derivatives",
    )
    return model, derivatives


def _lgssm_reference_and_sgqf() -> tuple[float, float | None, list[float] | None, float | None, str | None]:
    structural = _lgssm_structural(_lgssm_theta())
    linear = affine_structural_to_linear_gaussian_tf(structural)
    affine = TFFixedSGQFAffineModel(
        initial_mean=linear.initial_mean,
        initial_covariance=linear.initial_covariance,
        transition_matrix=linear.transition_matrix,
        process_covariance=linear.transition_covariance,
        observation_matrix=linear.observation_matrix,
        observation_covariance=linear.observation_covariance,
    )
    observations = _lgssm_observations()
    exact = tf_linear_gaussian_log_likelihood(
        observations,
        linear,
        backend="tf_cholesky",
        jitter=tf.constant(0.0, dtype=DTYPE),
        return_filtered=True,
    )
    cloud = tf_fixed_sgqf_cloud(dim=3, sparse_level=2)
    branch = TFFixedSGQFBranchConfig(predictive_epsilon=1e-10, innovation_epsilon=1e-10)
    sgqf = tf_fixed_sgqf_filter(
        observations,
        affine,
        cloud=cloud,
        branch_config=branch,
        return_filtered=True,
    )
    if sgqf.failure is not None:
        return float(exact.log_likelihood.numpy()), None, None, None, f"SGQF failure at {sgqf.failure.stage}"
    score_model, derivatives = _lgssm_sgqf_model_and_derivatives(_lgssm_theta())
    score = tf_fixed_sgqf_score(
        observations,
        score_model,
        derivatives,
        cloud=cloud,
        branch_config=branch,
        expected_branch_identity=sgqf.branch_identity,
    )
    if score.failure is not None or score.score is None:
        stage = score.failure.stage if score.failure is not None else "unknown"
        return float(exact.log_likelihood.numpy()), float(sgqf.log_likelihood.numpy()), None, None, f"SGQF score failure at {stage}"
    loglik, mismatch_reason = _same_value_score_loglik(
        sgqf.log_likelihood,
        score.log_likelihood,
        "direct affine LGSSM fixed-SGQF",
    )
    if loglik is None:
        return float(exact.log_likelihood.numpy()), None, None, None, mismatch_reason
    return (
        float(exact.log_likelihood.numpy()),
        loglik,
        [float(value) for value in score.score.numpy()],
        float(tf.linalg.norm(score.score).numpy()),
        None,
    )


def _predator_reference_and_sgqf() -> tuple[float, float | None, list[float] | None, float | None, str | None]:
    # The generic SGQF loop propagates before every observation, including y0.
    # The canonical T20 target assimilates y0 from the initial law first, so
    # this route is target-mismatched until an initial-observation-first
    # evaluator is implemented.  Fail closed rather than emitting a plausible
    # but wrong value/score pair.
    return (
        float("nan"),
        None,
        None,
        None,
        "blocked_target_alignment: predator-prey fixed-SGQF T20 evaluator is "
        "target-mismatched: generic "
        "route transitions before y0 while the canonical target assimilates "
        "y0 before the first transition",
    )

def _ksc_source_scope_sgqf_from_lowdim() -> tuple[float | None, float | None, list[float] | None, float | None, str | None]:
    theta = _sv_theta()
    normal = _normal01()
    gamma = normal.cdf(theta[0])
    beta = tf.exp(theta[1])
    observations = _sv_observations()
    result = highdim.independent_panel_sv_mixture_fixed_sgqf_filter(
        observations,
        gamma=gamma,
        beta=beta,
        sigma=tf.constant(1.0, dtype=DTYPE),
        sparse_level=2,
    )
    score = highdim.independent_panel_sv_mixture_fixed_sgqf_score(
        observations,
        gamma=gamma,
        beta=beta,
        sigma=tf.constant(1.0, dtype=DTYPE),
        sparse_level=2,
    )
    loglik, mismatch_reason = _same_value_score_loglik(
        result.log_likelihood,
        score.log_likelihood,
        "KSC fixed-SGQF",
    )
    if loglik is None:
        return None, None, None, None, mismatch_reason
    return (
        float(loglik / float(tf.shape(observations)[0].numpy())),
        loglik,
        [float(value) for value in score.score.numpy()],
        float(tf.linalg.norm(score.score).numpy()),
        None,
    )


def _ksc_source_scope_principal_sqrt_ukf() -> tuple[
    float | None,
    float | None,
    list[float] | None,
    float | None,
    float | None,
    str | None,
]:
    theta = _sv_theta()
    normal = _normal01()
    gamma = normal.cdf(theta[0])
    beta = tf.exp(theta[1])
    observations = _sv_observations()
    start = time.perf_counter()
    try:
        score_result = highdim.independent_panel_sv_mixture_ukf_score(
            observations,
            gamma=gamma,
            beta=beta,
            sigma=tf.constant(1.0, dtype=DTYPE),
        )
    except Exception as exc:  # pragma: no cover - recorded in artifact when a branch blocks.
        return None, None, None, None, None, f"principal-square-root UKF KSC score failed: {type(exc).__name__}: {exc}"
    runtime = time.perf_counter() - start
    loglik = float(score_result.log_likelihood.numpy())
    if score_result.score is None:
        return None, loglik, None, None, runtime, "principal-square-root UKF KSC score was not emitted"
    score = [float(value) for value in score_result.score.numpy()]
    if not math.isfinite(loglik):
        return None, None, None, None, runtime, "principal-square-root UKF KSC value was non-finite"
    if any(not math.isfinite(value) for value in score):
        return None, loglik, None, None, runtime, "principal-square-root UKF KSC score was non-finite"
    time_count = float(tf.shape(observations)[0].numpy())
    return float(loglik / time_count), loglik, score, float(tf.linalg.norm(score_result.score).numpy()), runtime, None


def _actual_sv_direct_sgqf_value_score() -> tuple[float | None, float | None, list[float] | None, float | None, str | None]:
    theta = _sv_theta()
    normal = _normal01()
    gamma = normal.cdf(theta[0])
    beta = tf.exp(theta[1])
    observations = _sv_observations()
    result = highdim.exact_transformed_sv_independent_panel_fixed_sgqf_filter(
        observations,
        gamma=gamma,
        beta=beta,
        sigma=tf.constant(1.0, dtype=DTYPE),
        sparse_level=2,
    )
    loglik = float(result.log_likelihood.numpy())
    if not math.isfinite(loglik):
        return None, None, None, None, "direct exact-transformed SGQF actual-SV value was non-finite"
    score_result = highdim.exact_transformed_sv_independent_panel_fixed_sgqf_score(
        observations,
        gamma=gamma,
        beta=beta,
        sigma=tf.constant(1.0, dtype=DTYPE),
        sparse_level=2,
    )
    if score_result.score is None:
        return None, loglik, None, None, "direct exact-transformed SGQF actual-SV score was not emitted"
    loglik, mismatch_reason = _same_value_score_loglik(
        result.log_likelihood,
        score_result.log_likelihood,
        "direct exact-transformed actual-SV fixed-SGQF",
    )
    if loglik is None:
        return None, None, None, None, mismatch_reason
    score = [float(value) for value in score_result.score.numpy()]
    if any(not math.isfinite(value) for value in score):
        return None, loglik, None, None, "direct exact-transformed SGQF actual-SV score was non-finite"
    time_count = float(tf.shape(observations)[0].numpy())
    return float(loglik / time_count), loglik, score, float(tf.linalg.norm(score_result.score).numpy()), None


def _zhao_cui_derivative_config_no_fd() -> highdim.FixedBranchDerivativeConfig:
    return highdim.FixedBranchDerivativeConfig(
        parameter_indices=(0, 1),
        finite_difference_h=(),
        solve_condition_number_veto=1e14,
    )


def _zhao_cui_actual_sv_tt_value_score() -> tuple[
    float | None,
    float | None,
    list[float] | None,
    float | None,
    float | None,
    str | None,
]:
    theta = _sv_theta()
    normal = _normal01()
    gamma = normal.cdf(theta[0])
    beta = tf.exp(theta[1])
    observations = _sv_observations()
    start = time.perf_counter()
    try:
        score_result = highdim.exact_transformed_sv_independent_panel_zhaocui_tt_score(
            observations,
            gamma=gamma,
            beta=beta,
            sigma=tf.constant(1.0, dtype=DTYPE),
            config=_zhao_cui_scalar_tt_config("leaderboard-zhaocui-actual-sv-manual-score"),
            derivative_config=_zhao_cui_derivative_config_no_fd(),
        )
    except Exception as exc:  # pragma: no cover - recorded in artifact when the row blocks.
        return None, None, None, None, time.perf_counter() - start, f"Zhao-Cui actual-SV manual TT score failed: {type(exc).__name__}: {exc}"
    runtime = time.perf_counter() - start
    if score_result.log_likelihood is None or score_result.score is None:
        return None, None, None, None, runtime, "Zhao-Cui actual-SV manual TT score was not emitted"
    loglik = float(score_result.log_likelihood.numpy())
    score = [float(value) for value in score_result.score.numpy()]
    if not math.isfinite(loglik):
        return None, None, None, None, runtime, "Zhao-Cui actual-SV manual TT value was non-finite"
    if any(not math.isfinite(value) for value in score):
        return None, loglik, None, None, runtime, "Zhao-Cui actual-SV manual TT score was non-finite"
    time_count = float(tf.shape(observations)[0].numpy())
    return float(loglik / time_count), loglik, score, float(tf.linalg.norm(score_result.score).numpy()), runtime, None


def _zhao_cui_ksc_tt_value_score() -> tuple[
    float | None,
    float | None,
    list[float] | None,
    float | None,
    float | None,
    str | None,
]:
    theta = _sv_theta()
    normal = _normal01()
    gamma = normal.cdf(theta[0])
    beta = tf.exp(theta[1])
    observations = _sv_observations()
    start = time.perf_counter()
    try:
        score_result = highdim.independent_panel_sv_mixture_zhaocui_tt_score(
            observations,
            gamma=gamma,
            beta=beta,
            sigma=tf.constant(1.0, dtype=DTYPE),
            config=_zhao_cui_scalar_tt_config("leaderboard-zhaocui-ksc-manual-score"),
            derivative_config=_zhao_cui_derivative_config_no_fd(),
        )
    except Exception as exc:  # pragma: no cover - recorded in artifact when the row blocks.
        return None, None, None, None, time.perf_counter() - start, f"Zhao-Cui KSC manual TT score failed: {type(exc).__name__}: {exc}"
    runtime = time.perf_counter() - start
    if score_result.log_likelihood is None or score_result.score is None:
        return None, None, None, None, runtime, "Zhao-Cui KSC manual TT score was not emitted"
    loglik = float(score_result.log_likelihood.numpy())
    score = [float(value) for value in score_result.score.numpy()]
    if not math.isfinite(loglik):
        return None, None, None, None, runtime, "Zhao-Cui KSC manual TT value was non-finite"
    if any(not math.isfinite(value) for value in score):
        return None, loglik, None, None, runtime, "Zhao-Cui KSC manual TT score was non-finite"
    time_count = float(tf.shape(observations)[0].numpy())
    return float(loglik / time_count), loglik, score, float(tf.linalg.norm(score_result.score).numpy()), runtime, None


def _zhao_cui_predator_prey_tt_value_score() -> tuple[
    float | None,
    float | None,
    list[float] | None,
    float | None,
    float | None,
    str | None,
]:
    model = highdim.p30_predator_prey_fixture_model()
    required_methods = (
        "initial_log_density_parameter_score",
        "transition_log_density_parameter_score",
        "observation_log_density_parameter_score",
    )
    missing = tuple(name for name in required_methods if not callable(getattr(model, name, None)))
    if missing:
        return None, None, None, None, None, (
            "Zhao-Cui predator-prey T20 manual TT score missing explicit "
            f"parameter-score methods: {missing}"
        )
    theta = _predator_prey_theta()
    observations = _predator_prey_observations()
    derivative_config = highdim.FixedBranchDerivativeConfig(
        parameter_indices=tuple(range(int(model.parameter_dim()))),
        finite_difference_h=(),
        solve_condition_number_veto=1e30,
    )
    start = time.perf_counter()
    try:
        score_result = highdim.multistate_nonlinear_fixed_design_tt_score_path(
            model,
            theta,
            observations,
            _zhao_cui_predator_prey_multistate_tt_config(
                "leaderboard-zhaocui-predator-prey-t20-manual-score"
            ),
            derivative_config,
            fixture_id="leaderboard.zhaocui.predator-prey-t20.score.v1",
            initial_target_id="leaderboard.zhaocui.predator-prey-t20.initial.v1",
            transition_target_id="leaderboard.zhaocui.predator-prey-t20.transition.v1",
            branch_seed_prefix="leaderboard-zhaocui-predator-prey-t20-score",
        )
    except Exception as exc:  # pragma: no cover - recorded in artifact when the row blocks.
        return None, None, None, None, time.perf_counter() - start, (
            f"Zhao-Cui predator-prey T20 manual multistate TT score failed: "
            f"{type(exc).__name__}: {exc}"
        )
    runtime = time.perf_counter() - start
    if score_result.log_likelihood is None or score_result.score is None:
        return None, None, None, None, runtime, "Zhao-Cui predator-prey T20 manual TT score was not emitted"
    loglik = float(score_result.log_likelihood.numpy())
    score = [float(value) for value in score_result.score.numpy()]
    if not math.isfinite(loglik):
        return None, None, None, None, runtime, "Zhao-Cui predator-prey T20 manual TT value was non-finite"
    if any(not math.isfinite(value) for value in score):
        return None, loglik, None, None, runtime, "Zhao-Cui predator-prey T20 manual TT score was non-finite"
    time_count = float(tf.shape(observations)[0].numpy())
    return float(loglik / time_count), loglik, score, float(tf.linalg.norm(score_result.score).numpy()), runtime, None


def _zhao_cui_generalized_sv_tt_value_score() -> tuple[
    float | None,
    float | None,
    list[float] | None,
    float | None,
    float | None,
    str | None,
]:
    model = highdim.GeneralizedSVPriorMeanSSM()
    required_methods = (
        "initial_log_density_parameter_score",
        "transition_log_density_parameter_score",
        "observation_log_density_parameter_score",
    )
    missing = tuple(name for name in required_methods if not callable(getattr(model, name, None)))
    if missing:
        return None, None, None, None, None, (
            "Zhao-Cui generalized-SV manual TT score missing explicit "
            f"parameter-score methods: {missing}"
        )
    theta = _generalized_sv_theta()
    observations = _generalized_sv_observations()
    derivative_config = highdim.FixedBranchDerivativeConfig(
        parameter_indices=tuple(range(int(model.parameter_dim()))),
        finite_difference_h=(),
        solve_condition_number_veto=1e14,
    )
    start = time.perf_counter()
    try:
        score_result = highdim.scalar_nonlinear_fixed_design_tt_score_path(
            model,
            theta,
            observations,
            _zhao_cui_scalar_tt_config(
                "leaderboard-zhaocui-generalized-sv-prior-mean-manual-score-smoke-budget",
                basis_degree=16,
                quadrature_order=41,
            ),
            derivative_config,
            fixture_id="leaderboard.zhaocui.generalized-sv-prior-mean.score.v1",
            initial_target_id="leaderboard.zhaocui.generalized-sv-prior-mean.initial.v1",
            transition_target_id="leaderboard.zhaocui.generalized-sv-prior-mean.transition.v1",
            branch_seed_prefix="leaderboard-zhaocui-generalized-sv-prior-mean-score",
            retained_moment_order=65,
            retained_propagation_order=81,
        )
    except Exception as exc:  # pragma: no cover - recorded in artifact when the row blocks.
        return None, None, None, None, time.perf_counter() - start, (
            f"Zhao-Cui generalized-SV source-row manual scalar TT score failed: "
            f"{type(exc).__name__}: {exc}"
        )
    runtime = time.perf_counter() - start
    if score_result.log_likelihood is None or score_result.score is None:
        return None, None, None, None, runtime, "Zhao-Cui generalized-SV manual TT score was not emitted"
    loglik = float(score_result.log_likelihood.numpy())
    score = [float(value) for value in score_result.score.numpy()]
    if not math.isfinite(loglik):
        return None, None, None, None, runtime, "Zhao-Cui generalized-SV manual TT value was non-finite"
    if any(not math.isfinite(value) for value in score):
        return None, loglik, None, None, runtime, "Zhao-Cui generalized-SV manual TT score was non-finite"
    time_count = float(tf.shape(observations)[0].numpy())
    return float(loglik / time_count), loglik, score, float(tf.linalg.norm(score_result.score).numpy()), runtime, None


def _zhao_cui_actual_sv_tt_cell() -> dict[str, Any]:
    avg_loglik, loglik, score, score_l2, runtime, reason = _zhao_cui_actual_sv_tt_value_score()
    executed_score = loglik is not None and score is not None
    return {
        "lane": "highdim_source_scope",
        "row_id": "zhao_cui_sv_actual_nongaussian_T1000",
        "algorithm_id": "zhao_cui_scalar_or_multistate",
        "route_id": "zhao_cui_exact_transformed_sv_fixed_branch_tt",
        "route_role": "fixed_variant_analytical_hmc_route",
        "hmc_route_policy_id": highdim.ZHAO_CUI_HMC_ROUTE_POLICY_ID,
        "hmc_target_scope_admitted": True,
        "comparison_status": "executed_value_score" if executed_score else "blocked",
        "numeric_execution_status": (
            "executed_zhao_cui_exact_transformed_sv_fixed_branch_tt_value_score"
            if executed_score
            else "blocked_zhao_cui_exact_transformed_sv_fixed_branch_tt_value_score"
        ),
        "average_log_likelihood": avg_loglik,
        "log_likelihood": loglik,
        "reference_log_likelihood": None,
        "absolute_value_gap_to_kalman": None,
        "score": score,
        "reference_score": None,
        "absolute_score_l2_gap_to_kalman": None,
        "score_l2_norm": score_l2,
        "score_coordinate_system": "theta=[probit_gamma,log_beta]",
        "score_derivative_provenance": (
            "zhao_cui_scalar_fixed_branch_tt_exact_transformed_sv_manual_parameter_score_methods_only"
            if executed_score
            else None
        ),
        "runtime_seconds": runtime,
        "mc_standard_error": None,
        "score_status": "analytical_score_emitted" if executed_score else "blocked_manual_tt_score_adapter",
        "score_status_reason": (
            "Zhao-Cui exact-transformed SV score emitted by scalar fixed-branch TT manual parameter-score adapter"
            if executed_score
            else reason
        ),
        "target_contract_status": (
            "target_compatible_exact_transformed_sv_fixed_branch_tt"
            if executed_score
            else "blocked_exact_transformed_sv_fixed_branch_tt"
        ),
        "reason": reason,
        "reason_codes": [
            (
                "ZHAO_CUI_EXACT_TRANSFORMED_SV_MANUAL_TT_VALUE_SCORE_ROUTE"
                if executed_score
                else "ZHAO_CUI_EXACT_TRANSFORMED_SV_MANUAL_TT_ROUTE_BLOCKED"
            )
        ],
        "nonclaims": [
            "exact transformed SV target z=log(y^2), not raw-observation native SV likelihood",
            "fixed-design scalar TT route, not adaptive MATLAB TT-cross/SIRT reproduction",
            "manual parameter-score methods only; reverse-mode tape fallback is not admitted",
            "not coupled multivariate Zhao-Cui TT",
            "not a production-GPU timing result",
        ],
    }


def _zhao_cui_predator_prey_tt_cell() -> dict[str, Any]:
    # The retained-grid multistate evaluator is historical and cannot be used
    # as an HMC/leaderboard fallback.  Predator-prey has no implemented
    # fixed-variant source-route evaluator yet, so fail closed here while
    # preserving the old route in its historical artifacts.
    avg_loglik = loglik = score = score_l2 = runtime = None
    reason = (
        "Predator-prey fixed-variant Zhao-Cui source-route analytical-gradient "
        "evaluator is not implemented; the historical multistate retained-grid "
        "result is not admitted for HMC or leaderboard use"
    )
    executed_score = False
    return {
        "lane": "highdim_source_scope",
        "row_id": "zhao_cui_predator_prey_T20",
        "algorithm_id": "zhao_cui_scalar_or_multistate",
        "comparison_status": "executed_value_score" if executed_score else "blocked_or_status_only",
        "numeric_execution_status": (
            "executed_zhao_cui_predator_prey_t20_multistate_tt_value_score"
            if executed_score
            else "blocked_zhao_cui_predator_prey_t20_multistate_tt_value_score"
        ),
        "average_log_likelihood": avg_loglik,
        "log_likelihood": loglik,
        "reference_log_likelihood": None,
        "absolute_value_gap_to_kalman": None,
        "score": score,
        "reference_score": None,
        "absolute_score_l2_gap_to_kalman": None,
        "score_l2_norm": score_l2,
        "score_coordinate_system": "theta=(r,K,a,s,u,v)",
        "score_derivative_provenance": (
            "zhao_cui_predator_prey_t20_multistate_fixed_design_tt_manual_parameter_score_methods_only"
            if executed_score
            else None
        ),
        "runtime_seconds": runtime,
        "mc_standard_error": None,
        "score_status": "analytical_score_emitted" if executed_score else "blocked_manual_tt_score_adapter",
        "score_status_reason": (
            reason
        ),
        "target_contract_status": (
            "target_compatible_predator_prey_t20_fixed_design_multistate_tt"
            if executed_score
            else "blocked_predator_prey_t20_fixed_design_multistate_tt"
        ),
        "reason": reason,
        "reason_codes": [
            "ZHAO_CUI_PREDATOR_PREY_FIXED_VARIANT_EVALUATOR_MISSING",
            "ZHAO_CUI_NONFIXED_RETAINED_GRID_ROUTE_DEMOTED_HISTORICAL",
        ],
        "route_id": "zhao_cui_predator_prey_t20_multistate_fixed_design_tt",
        "route_role": highdim.ZHAO_CUI_HISTORICAL_ROUTE_STATUS,
        "hmc_route_policy_id": None,
        "hmc_target_scope_admitted": False,
        "nonclaims": [
            "T20 source-scope additive-Gaussian RK4 closure, not native/non-Gaussian predator-prey likelihood",
            "fixed-design multistate TT route, not adaptive MATLAB TT-cross/SIRT reproduction",
            "manual parameter-score methods only; reverse-mode tape fallback is not admitted",
            "P47 two-observation lower-rung evidence is not reported as this T20 row",
            "not a production-GPU timing result",
            "not HMC readiness evidence",
            "historical retained-grid result is preserved only as diagnostic evidence",
        ],
    }


def _zhao_cui_generalized_sv_tt_cell() -> dict[str, Any]:
    avg_loglik, loglik, score, score_l2, runtime, reason = _zhao_cui_generalized_sv_tt_value_score()
    executed_score = loglik is not None and score is not None
    return {
        "lane": "highdim_source_scope",
        "row_id": "zhao_cui_generalized_sv_synthetic_from_estimated_values",
        "algorithm_id": "zhao_cui_scalar_or_multistate",
        "route_id": "zhao_cui_generalized_sv_prior_mean_scalar_fixed_design_tt",
        "route_role": "fixed_variant_analytical_hmc_route",
        "hmc_route_policy_id": highdim.ZHAO_CUI_HMC_ROUTE_POLICY_ID,
        "hmc_target_scope_admitted": True,
        "comparison_status": "executed_value_score" if executed_score else "blocked_or_status_only",
        "numeric_execution_status": (
            "executed_zhao_cui_generalized_sv_prior_mean_scalar_tt_value_score"
            if executed_score
            else "blocked_zhao_cui_generalized_sv_prior_mean_scalar_tt_value_score"
        ),
        "average_log_likelihood": avg_loglik,
        "log_likelihood": loglik,
        "reference_log_likelihood": None,
        "absolute_value_gap_to_kalman": None,
        "score": score,
        "reference_score": None,
        "absolute_score_l2_gap_to_kalman": None,
        "score_l2_norm": score_l2,
        "score_coordinate_system": "theta=(z_gamma,log_tau,mu_over_tau); gamma=Phi(z_gamma), tau=exp(log_tau), mu=mu_over_tau*tau",
        "score_derivative_provenance": (
            "zhao_cui_generalized_sv_prior_mean_scalar_fixed_design_tt_manual_parameter_score_methods_only"
            if executed_score
            else None
        ),
        "runtime_seconds": runtime,
        "mc_standard_error": None,
        "score_status": "analytical_score_emitted" if executed_score else "blocked_manual_tt_score_adapter",
        "score_status_reason": (
            "Zhao-Cui generalized-SV prior-mean source row score emitted by scalar fixed-design TT manual parameter-score adapter"
            if executed_score
            else reason
        ),
        "target_contract_status": (
            "target_compatible_generalized_sv_prior_mean_scalar_tt"
            if executed_score
            else "blocked_generalized_sv_prior_mean_scalar_tt"
        ),
        "reason": reason,
        "reason_codes": [
            (
                "ZHAO_CUI_GENERALIZED_SV_PRIOR_MEAN_SCALAR_MANUAL_TT_VALUE_SCORE_ROUTE"
                if executed_score
                else "ZHAO_CUI_GENERALIZED_SV_PRIOR_MEAN_SCALAR_MANUAL_TT_ROUTE_BLOCKED"
            )
        ],
        "source_anchors": [
            "third_party/audit/zhao_cui_tensor_ssm_p10/source/models/svmodels/ftt2true.m:6-14",
            "third_party/audit/zhao_cui_tensor_ssm_p10/source/models/svmodels/st_process.m:13-15",
            "third_party/audit/zhao_cui_tensor_ssm_p10/source/models/svmodels/like.m:4-7",
            "third_party/audit/zhao_cui_tensor_ssm_p10/source/models/svmodels/boxcoxinv.m:10-14",
        ],
        "nonclaims": [
            "prior-mean generated source-row target, not SP500 posterior estimate or direct SP500 returns row",
            "delta=0 and Gaussian-shock closure for the generated row; nu1/nu2 finite Student-t tails are not claimed",
            "fixed-design scalar TT smoke-budget route, not adaptive MATLAB TT-cross/SIRT reproduction",
            "full source horizon T1008 finite value/score admission smoke; not a tuned accuracy or timing configuration",
            "manual parameter-score methods only; reverse-mode tape fallback is not admitted",
            "not actual-SV, KSC, precursor, auxiliary, or native-oracle evidence",
            "not a production-GPU timing result",
            "not HMC readiness evidence",
        ],
    }


def _zhao_cui_ksc_tt_cell() -> dict[str, Any]:
    avg_loglik, loglik, score, score_l2, runtime, reason = _zhao_cui_ksc_tt_value_score()
    executed_score = loglik is not None and score is not None
    return {
        "lane": "highdim_source_scope",
        "row_id": "zhao_cui_sv_ksc_gaussian_mixture_surrogate_T1000",
        "algorithm_id": "zhao_cui_scalar_or_multistate",
        "route_id": "zhao_cui_ksc_mixture_fixed_branch_tt",
        "route_role": "fixed_variant_analytical_hmc_route",
        "hmc_route_policy_id": highdim.ZHAO_CUI_HMC_ROUTE_POLICY_ID,
        "hmc_target_scope_admitted": True,
        "comparison_status": "executed_value_score" if executed_score else "blocked",
        "numeric_execution_status": (
            "executed_zhao_cui_ksc_fixed_branch_tt_value_score"
            if executed_score
            else "blocked_zhao_cui_ksc_fixed_branch_tt_value_score"
        ),
        "average_log_likelihood": avg_loglik,
        "log_likelihood": loglik,
        "reference_log_likelihood": None,
        "absolute_value_gap_to_kalman": None,
        "score": score,
        "reference_score": None,
        "absolute_score_l2_gap_to_kalman": None,
        "score_l2_norm": score_l2,
        "score_coordinate_system": "theta=[probit_gamma,log_beta]",
        "score_derivative_provenance": (
            "zhao_cui_scalar_fixed_branch_tt_ksc_mixture_manual_parameter_score_methods_only"
            if executed_score
            else None
        ),
        "runtime_seconds": runtime,
        "mc_standard_error": None,
        "score_status": "analytical_score_emitted" if executed_score else "blocked_manual_tt_score_adapter",
        "score_status_reason": (
            "Zhao-Cui KSC transformed-mixture score emitted by scalar fixed-branch TT manual parameter-score adapter"
            if executed_score
            else reason
        ),
        "target_contract_status": (
            "target_compatible_ksc_transformed_sv_fixed_branch_tt"
            if executed_score
            else "blocked_ksc_transformed_sv_fixed_branch_tt"
        ),
        "reason": reason,
        "reason_codes": [
            (
                "ZHAO_CUI_KSC_MANUAL_TT_VALUE_SCORE_ROUTE"
                if executed_score
                else "ZHAO_CUI_KSC_MANUAL_TT_ROUTE_BLOCKED"
            )
        ],
        "nonclaims": [
            "declared KSC Gaussian-mixture transformed-SV target, not exact native SV likelihood",
            "fixed-design scalar TT route, not adaptive MATLAB TT-cross/SIRT reproduction",
            "manual parameter-score methods only; reverse-mode tape fallback is not admitted",
            "not coupled multivariate Zhao-Cui TT",
            "not a production-GPU timing result",
        ],
    }


def _actual_sv_augmented_noise_srukf_ukf() -> tuple[
    float | None,
    float | None,
    list[float] | None,
    float | None,
    float | None,
    str | None,
]:
    theta = _sv_theta()
    normal = _normal01()
    gamma = normal.cdf(theta[0])
    beta = tf.exp(theta[1])
    observations = _sv_observations()
    start = time.perf_counter()
    try:
        score_result = highdim.actual_transformed_sv_independent_panel_augmented_noise_srukf_score(
            observations,
            gamma=gamma,
            beta=beta,
            sigma=tf.constant(1.0, dtype=DTYPE),
        )
    except Exception as exc:  # pragma: no cover - recorded in artifact when a branch blocks.
        return None, None, None, None, None, f"actual-SV SR-UKF analytical score failed: {type(exc).__name__}: {exc}"
    runtime = time.perf_counter() - start
    loglik = float(score_result.log_likelihood.numpy())
    score = [float(value) for value in score_result.score.numpy()]
    if not math.isfinite(loglik):
        return None, None, None, None, runtime, "actual-SV SR-UKF value was non-finite"
    if any(not math.isfinite(value) for value in score):
        return None, loglik, None, None, runtime, "actual-SV SR-UKF score was non-finite"
    time_count = float(tf.shape(observations)[0].numpy())
    return float(loglik / time_count), loglik, score, float(tf.linalg.norm(score_result.score).numpy()), runtime, None


def _cell_from_p8d(algorithm_id: str, row_id: str, cell: dict[str, Any]) -> dict[str, Any]:
    status = cell["numeric_execution_status"]
    if status.startswith("executed_numeric"):
        if status == "executed_numeric_value_only_no_free_theta":
            comparison_status = "executed_value_only"
        elif status == "executed_numeric_dpf_5seed_value":
            comparison_status = "executed_mc_value_only"
        else:
            comparison_status = "executed_value_score"
        reason = None
    else:
        comparison_status = "blocked_or_status_only"
        reason = cell.get("not_available_reason") or "; ".join(cell.get("reason_codes", []))
    return {
        "lane": "highdim_source_scope",
        "row_id": row_id,
        "algorithm_id": algorithm_id,
        "comparison_status": comparison_status,
        "numeric_execution_status": status,
        "average_log_likelihood": cell.get("average_log_likelihood"),
        "log_likelihood": cell.get("log_likelihood"),
        "reference_log_likelihood": cell.get("reference_log_likelihood"),
        "absolute_value_gap_to_kalman": cell.get("absolute_value_gap_to_kalman"),
        "score": cell.get("score"),
        "reference_score": cell.get("reference_score"),
        "absolute_score_l2_gap_to_kalman": cell.get("absolute_score_l2_gap_to_kalman"),
        "score_l2_norm": cell.get("score_l2_norm"),
        "score_coordinate_system": cell.get("score_coordinate_system"),
        "score_derivative_provenance": cell.get("score_derivative_provenance"),
        "runtime_seconds": _runtime(cell),
        "mc_standard_error": cell.get("mc_standard_error"),
        "score_status": None,
        "score_status_reason": None,
        "target_contract_status": cell.get("target_contract_status"),
        "reason": reason,
        "reason_codes": cell.get("reason_codes", []),
        "nonclaims": cell.get("nonclaims", []),
    }


def _ksc_source_scope_principal_sqrt_ukf_cell() -> dict[str, Any]:
    avg_loglik, loglik, score, score_l2, runtime, reason = _ksc_source_scope_principal_sqrt_ukf()
    executed_score = loglik is not None and score is not None
    return {
        "lane": "highdim_source_scope",
        "row_id": "zhao_cui_sv_ksc_gaussian_mixture_surrogate_T1000",
        "algorithm_id": "ukf",
        "comparison_status": "executed_value_score" if executed_score else ("executed_value_only" if loglik is not None else "blocked"),
        "numeric_execution_status": (
            "executed_principal_sqrt_ukf_ksc_mixture_value_score"
            if executed_score
            else (
                "executed_principal_sqrt_ukf_ksc_mixture_value"
                if loglik is not None
                else "blocked_principal_sqrt_ukf_ksc_mixture_score"
            )
        ),
        "average_log_likelihood": avg_loglik,
        "log_likelihood": loglik,
        "reference_log_likelihood": None,
        "absolute_value_gap_to_kalman": None,
        "score": score,
        "reference_score": None,
        "absolute_score_l2_gap_to_kalman": None,
        "score_l2_norm": score_l2,
        "score_coordinate_system": "theta=[probit_gamma,log_beta]",
        "score_derivative_provenance": (
            "principal_sqrt_ukf_independent_panel_transformed_sv_gaussian_mixture_"
            "analytical_component_score_logsumexp_aggregation"
            if executed_score
            else None
        ),
        "runtime_seconds": runtime,
        "mc_standard_error": None,
        "score_status": "analytical_score_emitted" if executed_score else None,
        "score_status_reason": (
            "UKF score vector emitted by reviewed principal-square-root analytical component score path"
            if executed_score
            else reason
        ),
        "target_contract_status": (
            "target_compatible_direct_principal_sqrt_ukf_ksc_mixture"
            if loglik is not None
            else "blocked_principal_sqrt_ukf_ksc_mixture"
        ),
        "reason": reason,
        "reason_codes": [
            (
                "DIRECT_KSC_MIXTURE_PRINCIPAL_SQRT_UKF_VALUE_SCORE_ROUTE"
                if executed_score
                else "KSC_MIXTURE_PRINCIPAL_SQRT_UKF_ROUTE_BLOCKED"
            )
        ],
        "nonclaims": [
            "principal-square-root UKF analytical score route; historical SVD UKF score route is not used",
            "KSC transformed Gaussian-mixture surrogate row, not exact native SV likelihood",
            "not a production-GPU timing result",
        ],
    }


def _actual_sv_augmented_noise_srukf_ukf_cell() -> dict[str, Any]:
    avg_loglik, loglik, score, score_l2, runtime, reason = _actual_sv_augmented_noise_srukf_ukf()
    executed_score = loglik is not None and score is not None
    return {
        "lane": "highdim_source_scope",
        "row_id": "zhao_cui_sv_actual_nongaussian_T1000",
        "algorithm_id": "ukf",
        "comparison_status": "executed_value_score" if executed_score else ("executed_value_only" if loglik is not None else "blocked"),
        "numeric_execution_status": (
            "executed_actual_sv_augmented_noise_srukf_value_score"
            if executed_score
            else (
                "executed_actual_sv_augmented_noise_srukf_value"
                if loglik is not None
                else "blocked_actual_sv_augmented_noise_srukf_value_score"
            )
        ),
        "average_log_likelihood": avg_loglik,
        "log_likelihood": loglik,
        "reference_log_likelihood": None,
        "absolute_value_gap_to_kalman": None,
        "score": score,
        "reference_score": None,
        "absolute_score_l2_gap_to_kalman": None,
        "score_l2_norm": score_l2,
        "score_coordinate_system": "theta=[probit_gamma,log_beta]",
        "score_derivative_provenance": (
            "actual_sv_augmented_noise_factor_propagating_srukf_manual_score"
            if executed_score
            else None
        ),
        "runtime_seconds": runtime,
        "mc_standard_error": None,
        "score_status": "analytical_score_emitted" if executed_score else None,
        "score_status_reason": (
            "actual-SV raw augmented-noise Gaussian-closure score emitted by reviewed factor-propagating SR-UKF manual route"
            if executed_score
            else reason
        ),
        "target_contract_status": (
            "target_compatible_actual_sv_augmented_noise_srukf_gaussian_closure"
            if loglik is not None
            else "blocked_actual_sv_augmented_noise_srukf_gaussian_closure"
        ),
        "reason": reason,
        "reason_codes": [
            (
                "ACTUAL_SV_AUGMENTED_NOISE_SRUKF_VALUE_SCORE_ROUTE"
                if executed_score
                else "ACTUAL_SV_AUGMENTED_NOISE_SRUKF_ROUTE_BLOCKED"
            )
        ],
        "nonclaims": [
            "actual-SV augmented-noise Gaussian-closure surrogate, not exact transformed same-target likelihood",
            "factor-propagating SR-UKF analytical score route; historical SVD UKF and GradientTape score routes are not used",
            "score-at-true gamma consistency is weak evidence because the cubature SR-UKF surrogate can make gamma score nearly zero structurally",
            "not KSC Gaussian mixture approximation",
            "not coupled multivariate Zhao-Cui TT",
            "not a production-GPU timing result",
            "not HMC readiness evidence",
        ],
    }


def _zhao_cui_lgssm_exact_oracle_adapter() -> dict[str, Any]:
    exact, _sgqf_loglik, _sgqf_score, _sgqf_score_l2, _reason = _lgssm_reference_and_sgqf()
    kalman_cell = _p8d_cells()[("kalman_exact_or_mixture_enumeration", "benchmark_lgssm_exact_oracle_m3_T50")]
    score = [float(value) for value in kalman_cell["score"]]
    reference_score = [float(value) for value in kalman_cell.get("reference_score", score)]
    loglik = float(kalman_cell["log_likelihood"])
    exact_gap = abs(loglik - float(exact))
    return {
        "lane": "highdim_source_scope",
        "row_id": "benchmark_lgssm_exact_oracle_m3_T50",
        "algorithm_id": "zhao_cui_scalar_or_multistate",
        "route_id": "zhao_cui_lgssm_exact_oracle_affine_adapter",
        "route_role": highdim.ZHAO_CUI_HISTORICAL_ROUTE_STATUS,
        "hmc_route_policy_id": None,
        "hmc_target_scope_admitted": False,
        "comparison_status": "executed_value_score",
        "numeric_execution_status": "executed_lgssm_exact_oracle_adapter_value_score",
        "average_log_likelihood": float(kalman_cell["average_log_likelihood"]),
        "log_likelihood": loglik,
        "reference_log_likelihood": float(exact),
        "absolute_value_gap_to_kalman": exact_gap,
        "score": score,
        "reference_score": reference_score,
        "absolute_score_l2_gap_to_kalman": 0.0,
        "score_l2_norm": float(kalman_cell["score_l2_norm"]),
        "score_coordinate_system": "physical_theta",
        "score_derivative_provenance": (
            "zhao_cui_lgssm_user_amended_exact_oracle_affine_adapter_to_"
            "tf_covariance_differentiated_kalman_reference_cholesky_solve_physical_theta"
        ),
        "runtime_seconds": None,
        "mc_standard_error": None,
        "target_contract_status": "target_compatible_user_amended_lgssm_exact_oracle_adapter",
        "reason": None,
        "reason_codes": [
            "ZHAO_CUI_LGSSM_SOURCE_ROW_SUPERSEDED_BY_USER_AMENDED_EXACT_ORACLE_ADAPTER",
            "P8D_NUMERIC_EXECUTED_LGSSM_DIFFERENTIATED_KALMAN_AFFINE_VALUE_SCORE_HESSIAN",
        ],
        "nonclaims": [
            "user-amended exact-oracle LGSSM adapter; not Zhao-Cui MATLAB rng(0) reproduction",
            "not paper-scale Zhao-Cui TT-cross/SIRT training",
            "no historical ALS training used",
            "physical-coordinate score, not canonical unconstrained phi",
            "not a nonlinear benchmark ranking",
        ],
    }


def _p91_zhao_cui_sir_sidecar() -> dict[str, Any]:
    final_text = P91_FINAL_DECISION.read_text(encoding="utf-8")
    score_identity = _load(P91_SCORE_IDENTITY)
    gpu_xla = _load(P91_GPU_XLA)
    benchmark = _load(P91_BENCHMARK)
    hmc_smoke = _load(P91_HMC_SMOKE)
    status = "P91_SCOPED_PRODUCTION_READY_CLOSED"
    return {
        "scope": "local_complete_data_zhao_cui_sir_d18_component",
        "status": status if status in final_text else "P91_FINAL_STATUS_NOT_FOUND",
        "score_identity_status": _status_from_manifest(score_identity),
        "gpu_xla_status": _status_from_manifest(gpu_xla),
        "benchmark_status": _status_from_manifest(benchmark),
        "hmc_smoke_status": _status_from_manifest(hmc_smoke),
        "phase7_sidecar_performance": _p91_phase7_sidecar_performance(benchmark),
        "evidence_paths": [
            str(P91_FINAL_DECISION.relative_to(ROOT)),
            str(P91_SCORE_IDENTITY.relative_to(ROOT)),
            str(P91_GPU_XLA.relative_to(ROOT)),
            str(P91_BENCHMARK.relative_to(ROOT)),
            str(P91_HMC_SMOKE.relative_to(ROOT)),
        ],
        "nonclaims": [
            "not full observed-data/filtering score identity",
            "not full source-route FD derivative readiness",
            "not exact likelihood correctness",
            "not posterior correctness or convergence",
        ],
    }


def _p91_phase7_sidecar_performance(benchmark: dict[str, Any]) -> dict[str, Any]:
    evidence_paths = [str(P91_BENCHMARK.relative_to(ROOT))]
    artifact_paths = benchmark.get("artifact_paths")
    if isinstance(artifact_paths, dict):
        for path in artifact_paths.get("inputs", []):
            if isinstance(path, str):
                evidence_paths.append(path)

    timing_by_target: dict[str, Any] = {}
    for manifest in benchmark.get("manifests", []):
        if not isinstance(manifest, dict):
            continue
        target = manifest.get("target")
        if target not in {"cpu", "gpu"}:
            continue
        target_key = "gpu_xla" if target == "gpu" else "cpu"
        timing_by_target[target_key] = {
            "status": manifest.get("status"),
            "actual_xla_status": manifest.get("actual_xla_status"),
            "manifest_path": manifest.get("artifact_paths", {}).get("manifest"),
            "checks": {
                str(check.get("name")): {
                    "passed": check.get("passed"),
                    "first_call_seconds": check.get("first_call_seconds"),
                    "steady_mean_seconds": check.get("steady_mean_seconds"),
                    "steady_per_item_seconds": check.get("steady_per_item_seconds"),
                    "gpu_output_devices": check.get("gpu_output_devices"),
                    "post_warmup_retrace_detected": check.get("post_warmup_retrace_detected"),
                }
                for check in manifest.get("checks", [])
                if isinstance(check, dict) and check.get("name") is not None
            },
        }

    return {
        "namespace": "p91_scoped_local_complete_data_sidecar",
        "scope": "local_complete_data_zhao_cui_sir_d18_component",
        "admission_scope": "sidecar_only_not_full_observed_data_filtering_row",
        "excluded_from_main_leaderboard_ranking": True,
        "main_row_admission_status": "not_full_observed_data_filtering_evidence",
        "status": benchmark.get("status"),
        "evidence_paths": evidence_paths,
        "timing_by_target": timing_by_target,
        "nonclaims": [
            "not full observed-data/filtering SIR timing",
            "not full observed-data/filtering SIR score admission",
            "not part of main leaderboard timing ranking",
        ],
    }


def _zhao_cui_parameterized_sir_local_complete_data_value_score() -> tuple[
    float | None,
    float | None,
    list[float] | None,
    float | None,
    float | None,
    str | None,
]:
    start = time.perf_counter()
    try:
        dataset = _parameterized_sir_dataset_t20()
        model = highdim.parameterized_zhao_cui_sir_austria_model()
        theta = tf.constant(dataset["truth_theta"], dtype=DTYPE)
        states = tf.convert_to_tensor(dataset["states"], dtype=DTYPE)
        observations = tf.convert_to_tensor(dataset["observations"], dtype=DTYPE)
        value = model.initial_log_density(theta, states[0:1])[0]
        score = model.initial_log_density_parameter_score(theta, states[0:1])[0]
        horizon = int(observations.shape[0])
        for time_index in range(1, horizon):
            value = value + model.transition_log_density(
                theta,
                states[time_index - 1 : time_index],
                states[time_index : time_index + 1],
                t=time_index,
            )[0]
            score = score + model.transition_log_density_parameter_score(
                theta,
                states[time_index - 1 : time_index],
                states[time_index : time_index + 1],
                t=time_index,
            )[0]
        for time_index in range(horizon):
            value = value + model.observation_log_density(
                theta,
                states[time_index : time_index + 1],
                observations[time_index],
                t=time_index,
            )[0]
            score = score + model.observation_log_density_parameter_score(
                theta,
                states[time_index : time_index + 1],
                observations[time_index],
                t=time_index,
            )[0]
        runtime = time.perf_counter() - start
    except Exception as exc:  # pragma: no cover - recorded in artifact when a branch blocks.
        runtime = time.perf_counter() - start
        return None, None, None, None, runtime, (
            f"Zhao-Cui parameterized SIR local complete-data fixed-variant row failed: "
            f"{type(exc).__name__}: {exc}"
        )
    loglik = float(value.numpy())
    score_values = [float(item) for item in tf.reshape(score, [-1]).numpy()]
    if not math.isfinite(loglik):
        return None, None, None, None, runtime, "Zhao-Cui parameterized SIR local complete-data value was non-finite"
    if any(not math.isfinite(item) for item in score_values):
        return None, loglik, None, None, runtime, "Zhao-Cui parameterized SIR local complete-data score was non-finite"
    horizon_float = float(tf.shape(observations)[0].numpy())
    return (
        loglik / horizon_float,
        loglik,
        score_values,
        float(tf.linalg.norm(score).numpy()),
        runtime,
        None,
    )


def _zhao_cui_parameterized_sir_local_complete_data_cell() -> dict[str, Any]:
    avg_loglik, loglik, score, score_l2, runtime, reason = (
        _zhao_cui_parameterized_sir_local_complete_data_value_score()
    )
    sidecar = _p91_zhao_cui_sir_sidecar()
    executed_score = loglik is not None and score is not None
    return {
        "lane": "highdim_source_scope",
        "row_id": PARAMETERIZED_SIR_ROW,
        "algorithm_id": "zhao_cui_scalar_or_multistate",
        "comparison_status": "executed_value_score" if executed_score else "blocked",
        "numeric_execution_status": (
            "executed_zhao_cui_parameterized_sir_t20_local_complete_data_value_score"
            if executed_score
            else "blocked_zhao_cui_parameterized_sir_t20_local_complete_data_value_score"
        ),
        "row_admission_status": "scoped_component_row_admitted" if executed_score else "blocked",
        "target_scope": "local_complete_data_zhao_cui_sir_d18_component",
        "route_id": "zhao_cui_sir_d18_local_complete_data_manual_component",
        "route_role": highdim.FIXED_VARIANT_ZHAO_CUI_PRODUCTION_ROUTE,
        "hmc_route_policy_id": highdim.ZHAO_CUI_HMC_ROUTE_POLICY_ID,
        "hmc_target_scope_admitted": True,
        "retained_grid_route_role": highdim.MULTISTATE_RETAINED_GRID_ROUTE_ROLE,
        "retained_grid_leaderboard_admission": highdim.MULTISTATE_RETAINED_GRID_LEADERBOARD_ADMISSION,
        "average_log_likelihood": avg_loglik,
        "log_likelihood": loglik,
        "reference_log_likelihood": None,
        "absolute_value_gap_to_kalman": None,
        "score": score,
        "reference_score": None,
        "absolute_score_l2_gap_to_kalman": None,
        "score_l2_norm": score_l2,
        "score_coordinate_system": "theta=(log_kappa_scale,log_nu_scale,log_obs_noise_scale)",
        "score_derivative_provenance": (
            "zhao_cui_sir_d18_local_complete_data_manual_parameter_score_methods"
        ),
        "runtime_seconds": runtime,
        "mc_standard_error": None,
        "score_status": "analytical_score_emitted" if executed_score else "blocked_manual_score_not_emitted",
        "score_status_reason": (
            "Zhao-Cui parameterized SIR T20 local complete-data score emitted by manual parameter-score methods"
            if executed_score
            else reason
        ),
        "target_contract_status": "target_compatible_scoped_local_complete_data_component",
        "reason": reason,
        "reason_codes": [
            "ZHAO_CUI_PARAMETERIZED_SIR_T20_SCOPED_LOCAL_COMPLETE_DATA_VALUE_SCORE",
            "P91_SCOPED_LOCAL_COMPLETE_DATA_READY",
            "FULL_FILTERING_LEADERBOARD_CELL_STILL_NOT_CLAIMED",
            "RETAINED_GRID_ROUTE_NOT_ADMITTED_FOR_PRODUCTION",
        ],
        "p91_scoped_evidence": sidecar,
        "nonclaims": [
            "scoped local complete-data component row, not full observed-data/filtering SIR likelihood",
            "row id encodes parameterization only; target_scope metadata carries the local-complete-data boundary",
            "not source-faithful as an inference-theta parameterization",
            "not exact likelihood correctness",
            "not posterior correctness or convergence",
            "P91 sidecar timing is not main-row full-filtering timing",
            "retained-grid route remains diagnostic/historical and is not selected for production leaderboard admission",
        ] + sidecar["nonclaims"],
    }


def _blocked_scoped_parameterized_sir_cell(algorithm_id: str) -> dict[str, Any]:
    reason = (
        f"{algorithm_id} is not admitted for the scoped Zhao-Cui parameterized SIR "
        "local complete-data component row; the scoped row is a fixed-variant "
        "Zhao-Cui manual-score component cell."
    )
    return {
        "lane": "highdim_source_scope",
        "row_id": PARAMETERIZED_SIR_ROW,
        "algorithm_id": algorithm_id,
        "comparison_status": "blocked",
        "numeric_execution_status": "blocked_not_applicable_to_scoped_zhao_cui_component_row",
        "row_admission_status": "not_applicable_to_scoped_component_row",
        "target_scope": "local_complete_data_zhao_cui_sir_d18_component",
        "route_role": None,
        "retained_grid_route_role": highdim.MULTISTATE_RETAINED_GRID_ROUTE_ROLE,
        "retained_grid_leaderboard_admission": highdim.MULTISTATE_RETAINED_GRID_LEADERBOARD_ADMISSION,
        "average_log_likelihood": None,
        "log_likelihood": None,
        "reference_log_likelihood": None,
        "absolute_value_gap_to_kalman": None,
        "score": None,
        "reference_score": None,
        "absolute_score_l2_gap_to_kalman": None,
        "score_l2_norm": None,
        "score_coordinate_system": None,
        "score_derivative_provenance": None,
        "runtime_seconds": None,
        "mc_standard_error": None,
        "score_status": "not_applicable_to_scoped_component_row",
        "score_status_reason": reason,
        "target_contract_status": "not_applicable_to_scoped_zhao_cui_component_row",
        "reason": reason,
        "reason_codes": [
            "SCOPED_COMPONENT_ROW_ZHAO_CUI_ONLY",
            "FULL_FILTERING_LEADERBOARD_CELL_STILL_NOT_CLAIMED",
        ],
        "nonclaims": [
            "not a fixed-SGQF or UKF full-filtering comparison row",
            "not ranked as a full three-algorithm observed-data/filtering row",
        ],
    }


def _cell_for_fixed_sgqf(row_id: str) -> dict[str, Any]:
    if row_id == PARAMETERIZED_SIR_ROW:
        row = _blocked_scoped_parameterized_sir_cell("fixed_sgqf")
        row["route_role"] = "not_applicable_to_scoped_component_row"
        return row
    if row_id == "benchmark_lgssm_exact_oracle_m3_T50":
        reference, loglik, score, score_l2, reason = _lgssm_reference_and_sgqf()
        executed_score = loglik is not None and score is not None
        return {
            "lane": "highdim_source_scope",
            "row_id": row_id,
            "algorithm_id": "fixed_sgqf",
            "comparison_status": "executed_value_score" if executed_score else ("executed_value_only" if loglik is not None else "blocked"),
            "numeric_execution_status": "executed_direct_affine_sgqf_value_score" if executed_score else ("executed_direct_affine_sgqf_value" if loglik is not None else "blocked_by_missing_affine_route"),
            "average_log_likelihood": (loglik / 50.0) if loglik is not None else None,
            "log_likelihood": loglik,
            "reference_log_likelihood": reference,
            "absolute_value_gap_to_kalman": (abs(loglik - reference) if loglik is not None else None),
            "score": score,
            "score_l2_norm": score_l2,
            "score_coordinate_system": "p8d_lgssm_theta",
            "score_derivative_provenance": "fixed_sgqf_analytic_first_order_fixed_branch_affine_lgssm_score",
            "runtime_seconds": None,
            "mc_standard_error": None,
            "target_contract_status": "target_compatible_direct_affine_sgqf" if loglik is not None else "blocked_fixed_sgqf_current_scope",
            "reason": reason,
            "reason_codes": ["DIRECT_AFFINE_SGQF_LGSSM_VALUE_SCORE_ROUTE" if executed_score else ("DIRECT_AFFINE_SGQF_LGSSM_ROUTE" if loglik is not None else "TWO_LANE_FIXED_SGQF_BLOCK_OR_MISSING_EVALUATOR")],
            "nonclaims": [
                "direct affine SGQF value plus analytical fixed-branch score route" if executed_score else "direct affine SGQF value route; analytical score not emitted",
                "not a production-GPU timing result",
            ],
        }
    if row_id == "zhao_cui_predator_prey_T20":
        reference, loglik, score, score_l2, reason = _predator_reference_and_sgqf()
        executed_score = loglik is not None and score is not None
        return {
            "lane": "highdim_source_scope",
            "row_id": row_id,
            "algorithm_id": "fixed_sgqf",
            "comparison_status": "executed_value_score" if executed_score else ("executed_value_only" if loglik is not None else "blocked"),
            "numeric_execution_status": "executed_predator_prey_sgqf_value_score" if executed_score else ("executed_predator_prey_sgqf_value" if loglik is not None else "blocked_predator_prey_sgqf_value"),
            "average_log_likelihood": (loglik / 20.0) if loglik is not None else None,
            "log_likelihood": loglik,
            "reference_log_likelihood": reference if math.isfinite(reference) else None,
            "absolute_value_gap_to_kalman": None,
            "score": score,
            "score_l2_norm": score_l2,
            "score_coordinate_system": "source_scope_predator_prey_theta_pending_t20_evaluator" if loglik is not None else None,
            "score_derivative_provenance": "fixed_sgqf_analytic_first_order_fixed_branch_predator_prey_score" if executed_score else None,
            "runtime_seconds": None,
            "mc_standard_error": None,
            "target_contract_status": "target_compatible_direct_predator_prey_sgqf" if loglik is not None else "blocked_missing_t20_fixed_sgqf_evaluator",
            "reason": reason,
            "reason_codes": ["DIRECT_PREDATOR_PREY_SGQF_VALUE_SCORE_ROUTE" if executed_score else ("DIRECT_PREDATOR_PREY_SGQF_ROUTE" if loglik is not None else "PREDATOR_PREY_T20_FIXED_SGQF_EVALUATOR_REQUIRED")],
            "nonclaims": [
                "direct SGQF predator-prey value plus analytical fixed-branch score route" if executed_score else "no fixed-SGQF T20 predator-prey value emitted in this phase",
                "P47 two-observation lower-rung SGQF diagnostic is not reported as the T20 source-scope row",
                "not a production-GPU timing result",
            ],
        }
    if row_id == "zhao_cui_sv_ksc_gaussian_mixture_surrogate_T1000":
        avg_loglik, loglik, score, score_l2, reason = _ksc_source_scope_sgqf_from_lowdim()
        executed_score = loglik is not None and score is not None
        return {
            "lane": "highdim_source_scope",
            "row_id": row_id,
            "algorithm_id": "fixed_sgqf",
            "comparison_status": "executed_value_score" if executed_score else ("executed_value_only" if loglik is not None else "blocked"),
            "numeric_execution_status": "executed_source_scope_ksc_sgqf_value_score" if executed_score else ("executed_source_scope_ksc_sgqf_value" if loglik is not None else "blocked_source_scope_ksc_sgqf_t1000_missing"),
            "average_log_likelihood": avg_loglik,
            "log_likelihood": loglik,
            "reference_log_likelihood": None,
            "absolute_value_gap_to_kalman": None,
            "score": score,
            "score_l2_norm": score_l2,
            "score_coordinate_system": "p8_truth_theta_coordinate",
            "score_derivative_provenance": "fixed_sgqf_independent_panel_transformed_sv_gaussian_mixture_analytic_component_score_logsumexp_aggregation",
            "runtime_seconds": None,
            "mc_standard_error": None,
            "target_contract_status": "target_compatible_declared_ksc_surrogate_value_score" if executed_score else ("target_compatible_declared_ksc_surrogate_value_only" if loglik is not None else "blocked_fixed_sgqf_current_scope"),
            "reason": reason,
            "reason_codes": ["DIRECT_KSC_SOURCE_SCOPE_SGQF_VALUE_SCORE_ROUTE" if executed_score else ("DIRECT_KSC_SOURCE_SCOPE_SGQF_ROUTE" if loglik is not None else "TWO_LANE_FIXED_SGQF_BLOCK_OR_MISSING_EVALUATOR")],
            "nonclaims": [
                "source-scope KSC SGQF value plus analytical component-score route executed" if executed_score else "source-scope KSC SGQF value route executed; analytical score not emitted",
                "declared KSC Gaussian-mixture surrogate target only",
            ],
        }
    if row_id == "zhao_cui_sv_actual_nongaussian_T1000":
        avg_loglik, loglik, score, score_l2, reason = _actual_sv_direct_sgqf_value_score()
        executed_score = loglik is not None and score is not None
        return {
            "lane": "highdim_source_scope",
            "row_id": row_id,
            "algorithm_id": "fixed_sgqf",
            "comparison_status": "executed_value_score" if executed_score else ("executed_value_only" if loglik is not None else "blocked"),
            "numeric_execution_status": "executed_direct_exact_transformed_sv_sgqf_value_score" if executed_score else ("executed_direct_exact_transformed_sv_sgqf_value" if loglik is not None else "blocked_direct_exact_transformed_sv_sgqf_value"),
            "average_log_likelihood": avg_loglik,
            "log_likelihood": loglik,
            "reference_log_likelihood": None,
            "absolute_value_gap_to_kalman": None,
            "score": score,
            "score_l2_norm": score_l2,
            "score_coordinate_system": "theta=[probit_gamma, log_beta] per coordinate",
            "score_derivative_provenance": "fixed_sgqf_exact_transformed_sv_manual_forward_sensitivity_analytical_recurrence",
            "runtime_seconds": None,
            "mc_standard_error": None,
            "target_contract_status": "target_compatible_direct_exact_transformed_sv_sgqf_value_score" if executed_score else ("target_compatible_direct_exact_transformed_sv_sgqf_value" if loglik is not None else "blocked_direct_exact_transformed_sv_sgqf_value"),
            "reason": reason,
            "reason_codes": ["DIRECT_EXACT_TRANSFORMED_SV_SGQF_VALUE_SCORE_ROUTE" if executed_score else ("DIRECT_EXACT_TRANSFORMED_SV_SGQF_VALUE_ROUTE" if loglik is not None else "DIRECT_EXACT_TRANSFORMED_SV_SGQF_VALUE_BLOCKED")],
            "nonclaims": [
                "direct exact-transformed SGQF actual-SV value plus manual forward-sensitivity score route executed" if executed_score else "direct exact-transformed SGQF actual-SV value route executed",
                "not KSC Gaussian mixture approximation",
                "not augmented-noise Gaussian-closure route",
                "score differentiates the fixed-SGQF approximate value target, not an exact-likelihood oracle" if executed_score else "no analytical score claim",
                "not a production-GPU timing result",
            ],
        }
    # Genuine blocks under current route set.
    generalized_sv = row_id == "zhao_cui_generalized_sv_synthetic_from_estimated_values"
    reasons = {
        "zhao_cui_spatial_sir_austria_j9_T20": "no reviewed SGQF source-scope spatial SIR route is wired",
        "zhao_cui_generalized_sv_synthetic_from_estimated_values": (
            "blocked_source_row_evaluator_missing: no reviewed fixed-SGQF exact-row evaluator is wired for "
            "zhao_cui_generalized_sv_synthetic_from_estimated_values; native-oracle, precursor, auxiliary, actual-SV, "
            "and KSC evidence are not source-row admission evidence"
        ),
    }
    return {
        "lane": "highdim_source_scope",
        "row_id": row_id,
        "algorithm_id": "fixed_sgqf",
        "comparison_status": "blocked",
        "numeric_execution_status": (
            "blocked_generalized_sv_fixed_sgqf_source_row_evaluator_missing"
            if generalized_sv
            else "blocked_by_two_lane_contract_or_missing_source_scope_evaluator"
        ),
        "average_log_likelihood": None,
        "log_likelihood": None,
        "reference_log_likelihood": None,
        "absolute_value_gap_to_kalman": None,
        "score": None,
        "score_l2_norm": None,
        "score_coordinate_system": None,
        "score_derivative_provenance": None,
        "runtime_seconds": None,
        "mc_standard_error": None,
        "target_contract_status": (
            "blocked_exact_source_row_evaluator_missing"
            if generalized_sv
            else "blocked_fixed_sgqf_current_scope"
        ),
        "reason": reasons[row_id],
        "reason_codes": (
            [
                "GENERALIZED_SV_EXACT_SOURCE_ROW_FIXED_SGQF_EVALUATOR_REQUIRED",
                "PRECURSOR_NATIVE_ORACLE_AUXILIARY_ACTUAL_SV_KSC_NOT_ADMISSION_EVIDENCE",
            ]
            if generalized_sv
            else ["TWO_LANE_FIXED_SGQF_BLOCK_OR_MISSING_EVALUATOR"]
        ),
        "nonclaims": [
            "not executed in the current source-scope leaderboard lane",
            (
                "native dense oracle, precursor, auxiliary, actual-SV, and KSC evidence may inform debugging only and do not admit this fixed-SGQF source-row cell"
                if generalized_sv
                else "no SGQF broad high-dimensional admission is implied"
            ),
        ],
    }


def _apply_score_status(row: dict[str, Any]) -> dict[str, Any]:
    row = dict(row)
    if row["algorithm_id"] != "fixed_sgqf":
        return row
    if row["comparison_status"] == "executed_value_score":
        row["score_status"] = "analytical_score_emitted"
        row["score_status_reason"] = "SGQF score vector emitted by reviewed analytical fixed-branch score path"
        return row
    if row["row_id"] == PARAMETERIZED_SIR_ROW and row.get("score_status"):
        return row
    mapping = {
        "benchmark_lgssm_exact_oracle_m3_T50": (
            "blocked_missing_analytical_route",
            "direct affine SGQF value route exists, but no reviewed source-scope SGQF score route is emitted",
        ),
        "zhao_cui_sv_actual_nongaussian_T1000": (
            "blocked_strict_analytical_score_adapter",
            "direct exact-transformed SGQF actual-SV value is same-target, but the current score wrapper uses GradientTape and is not admitted under strict analytical-only score policy",
        ),
        "zhao_cui_sv_ksc_gaussian_mixture_surrogate_T1000": (
            "diagnostic_score_only",
            "existing SGQF score evidence is tiny-fixture-only and not yet promoted to source-scope analytical-score admission",
        ),
        "zhao_cui_spatial_sir_austria_j9_T20": (
            "blocked_no_free_theta",
            "row has no free theta and no reviewed SGQF source-scope route",
        ),
        "zhao_cui_predator_prey_T20": (
            "blocked_target_alignment",
            "no reviewed fixed-SGQF evaluator is wired for the source-scope T20 predator-prey observations; the P47 two-observation lower-rung diagnostic is not reported as this T20 row",
        ),
        "zhao_cui_generalized_sv_synthetic_from_estimated_values": (
            "blocked_exact_source_row_evaluator_missing",
            "no reviewed fixed-SGQF exact-row evaluator or derivative route exists for zhao_cui_generalized_sv_synthetic_from_estimated_values",
        ),
    }
    row["score_status"], row["score_status_reason"] = mapping[row["row_id"]]
    return row


def _apply_p91_zhao_cui_status(row: dict[str, Any]) -> dict[str, Any]:
    row = dict(row)
    if row["algorithm_id"] != "zhao_cui_scalar_or_multistate":
        return row
    if row["row_id"] == "benchmark_lgssm_exact_oracle_m3_T50":
        return _zhao_cui_lgssm_exact_oracle_adapter()
    if row["row_id"] == "zhao_cui_generalized_sv_synthetic_from_estimated_values":
        return _zhao_cui_generalized_sv_tt_cell()
    if row["row_id"] != "zhao_cui_spatial_sir_austria_j9_T20":
        return row
    sidecar = _p91_zhao_cui_sir_sidecar()
    row["p91_scoped_evidence"] = sidecar
    row["comparison_status"] = "blocked_or_status_only"
    row["numeric_execution_status"] = (
        "blocked_full_filtering_evaluator_pending_p91_local_component_ready"
    )
    row["target_contract_status"] = (
        "full_filtering_blocked_local_complete_data_component_ready"
    )
    row["reason"] = (
        "P91 closes the scoped local complete-data SIR d18 component route, "
        "but the full observed-data/filtering leaderboard evaluator remains "
        "blocked by preserved source-route derivative/evaluator gaps."
    )
    row["reason_codes"] = list(row.get("reason_codes", [])) + [
        "P91_SCOPED_LOCAL_COMPLETE_DATA_READY",
        "FULL_FILTERING_LEADERBOARD_CELL_STILL_BLOCKED",
    ]
    row["nonclaims"] = list(row.get("nonclaims", [])) + sidecar["nonclaims"]
    return row


def _validate_analytical_score_contract(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        if row["comparison_status"] != "executed_value_score":
            continue
        score = row.get("score")
        provenance = str(row.get("score_derivative_provenance") or "")
        if not isinstance(score, list) or not score:
            raise ValueError(f"{row['row_id']} {row['algorithm_id']}: executed_value_score requires a score vector")
        if any(not math.isfinite(float(value)) for value in score):
            raise ValueError(f"{row['row_id']} {row['algorithm_id']}: score vector must be finite")
        value_route_id = row.get("value_route_id")
        score_route_id = row.get("score_route_id")
        if not isinstance(value_route_id, str) or not value_route_id:
            raise ValueError(f"{row['row_id']} {row['algorithm_id']}: admitted score row requires value_route_id")
        if value_route_id != score_route_id:
            raise ValueError(
                f"{row['row_id']} {row['algorithm_id']}: admitted score row mixes value route "
                f"{value_route_id!r} with score route {score_route_id!r}"
            )
        if row.get("value_score_route_status") != "same_route_value_score":
            raise ValueError(
                f"{row['row_id']} {row['algorithm_id']}: admitted score row must declare same_route_value_score"
            )
        if (
            row.get("algorithm_id") == "zhao_cui_scalar_or_multistate"
            and not _zhao_cui_hmc_route_is_admissible(row)
        ):
            raise ValueError(
                f"{row['row_id']} {row['algorithm_id']}: admitted Zhao-Cui score "
                "requires the fixed-variant HMC filtering route identity"
            )
        lower = provenance.lower()
        if "autodiff" in lower or "gradienttape" in lower or "gradient_tape" in lower:
            raise ValueError(f"{row['row_id']} {row['algorithm_id']}: score provenance must not use autodiff/tape fallback")
        if row["algorithm_id"] == "ukf" and ("svd" in lower or "eigenderivative" in lower):
            raise ValueError(
                f"{row['row_id']} {row['algorithm_id']}: UKF leaderboard score provenance "
                "must use a reviewed non-historical route, not the historical SVD route"
            )
        if (
            row["algorithm_id"] == "ukf"
            and row["row_id"] != "benchmark_lgssm_exact_oracle_m3_T50"
            and "principal_sqrt" not in lower
            and "principal-square-root" not in lower
            and "factor_propagating_srukf_manual_score" not in lower
        ):
            raise ValueError(
                f"{row['row_id']} {row['algorithm_id']}: UKF leaderboard score provenance "
                "must name a reviewed principal-square-root or factor-propagating SR-UKF route"
            )
        if row["algorithm_id"] == "fixed_sgqf" and "analytic" not in lower and "analytical" not in lower:
            raise ValueError(f"{row['row_id']} {row['algorithm_id']}: score provenance must name analytical route")


def _score_provenance_is_autodiff_or_historical_svd_ukf(row: dict[str, Any]) -> bool:
    provenance = str(row.get("score_derivative_provenance") or "").lower()
    if "autodiff" in provenance or "gradienttape" in provenance or "gradient_tape" in provenance:
        return True
    return row.get("algorithm_id") == "ukf" and ("svd" in provenance or "eigenderivative" in provenance)


def _enforce_analytical_score_admission(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    updated: list[dict[str, Any]] = []
    for row in rows:
        row = _normalize_legacy_zhao_cui_route_identity(dict(row))
        provenance = str(row.get("score_derivative_provenance") or "").lower()
        repaired_zhao_cui_tt = (
            row.get("algorithm_id") == "zhao_cui_scalar_or_multistate"
            and row.get("row_id") in {
                "zhao_cui_sv_actual_nongaussian_T1000",
                "zhao_cui_sv_ksc_gaussian_mixture_surrogate_T1000",
                "zhao_cui_predator_prey_T20",
                "zhao_cui_generalized_sv_synthetic_from_estimated_values",
            }
            and "manual_parameter_score_methods_only" in provenance
            and _zhao_cui_hmc_route_is_admissible(row)
        )
        repaired_parameterized_sir = (
            row.get("algorithm_id") == "zhao_cui_scalar_or_multistate"
            and row.get("row_id") == PARAMETERIZED_SIR_ROW
            and provenance == "zhao_cui_sir_d18_local_complete_data_manual_parameter_score_methods"
            and row.get("target_scope") == "local_complete_data_zhao_cui_sir_d18_component"
            and row.get("row_admission_status") == "scoped_component_row_admitted"
            and _zhao_cui_hmc_route_is_admissible(row)
        )
        if repaired_parameterized_sir and row.get("comparison_status") == "executed_value_score":
            updated.append(row)
            continue
        if (
            row.get("algorithm_id") == "zhao_cui_scalar_or_multistate"
            and row.get("row_id") == PARAMETERIZED_SIR_ROW
            and row.get("comparison_status") == "executed_value_score"
        ):
            row["comparison_status"] = "blocked"
            row["numeric_execution_status"] = "blocked_scoped_component_metadata_guard_failed"
            row["score_status"] = "blocked_scoped_component_metadata_guard_failed"
            row["score_status_reason"] = (
                "parameterized SIR scoped component row requires explicit "
                "row_admission_status, target_scope, and fixed-variant HMC "
                "route metadata before score admission"
            )
            row["reason"] = row["score_status_reason"]
            row["score"] = None
            row["score_l2_norm"] = None
            row["nonclaims"] = list(row.get("nonclaims", [])) + [
                "row id alone is insufficient for scoped component score admission",
            ]
            updated.append(row)
            continue
        if repaired_zhao_cui_tt and row.get("comparison_status") == "executed_value_score":
            updated.append(row)
            continue
        if (
            row.get("comparison_status") == "executed_value_score"
            and row.get("algorithm_id") == "zhao_cui_scalar_or_multistate"
            and _score_provenance_is_autodiff_or_historical_svd_ukf(row)
        ):
            row["comparison_status"] = "executed_value_only"
            row["numeric_execution_status"] = (
                "executed_numeric_value_only_autodiff_score_not_admitted"
            )
            row["score_status"] = "blocked_autodiff_not_admitted"
            row["score_status_reason"] = (
                "autodiff score provenance is diagnostic only; analytical "
                "gradient accuracy is required"
            )
            row["reason"] = row["score_status_reason"]
            row["score"] = None
            row["score_l2_norm"] = None
            row["score_coordinate_system"] = None
            row["nonclaims"] = list(row.get("nonclaims", [])) + [
                "autodiff or historical SVD score provenance is diagnostic only",
            ]
            updated.append(row)
            continue
        if (
            row.get("algorithm_id") == "zhao_cui_scalar_or_multistate"
            and row.get("comparison_status") == "executed_value_score"
        ):
            updated.append(_demote_zhao_cui_historical_score(row))
            continue
        if row.get("comparison_status") == "executed_value_score" and _score_provenance_is_autodiff_or_historical_svd_ukf(row):
            is_historical_svd_ukf = row.get("algorithm_id") == "ukf" and (
                "svd" in provenance or "eigenderivative" in provenance
            )
            row["comparison_status"] = "executed_value_only"
            row["numeric_execution_status"] = (
                "executed_numeric_value_only_historical_svd_ukf_score_not_admitted"
                if is_historical_svd_ukf
                else "executed_numeric_value_only_autodiff_score_not_admitted"
            )
            row["score_status"] = (
                "blocked_historical_svd_ukf_not_admitted"
                if is_historical_svd_ukf
                else "blocked_autodiff_not_admitted"
            )
            row["score_status_reason"] = (
                "historical SVD UKF score provenance is diagnostic only; principal-square-root "
                "UKF analytical gradient accuracy is required for leaderboard admission"
                if is_historical_svd_ukf
                else (
                    "actual-SV nonlinear UKF value uses the augmented-noise sigma-point diagnostic route; "
                    "its current score provenance is autodiff only, so analytical gradient accuracy is not admitted"
                    if row.get("algorithm_id") == "ukf"
                    and row.get("row_id") == "zhao_cui_sv_actual_nongaussian_T1000"
                    else (
                        "autodiff score provenance is diagnostic only; analytical gradient "
                        "accuracy is the leaderboard benchmark"
                    )
                )
            )
            row["reason"] = row["score_status_reason"]
            row["score"] = None
            row["score_l2_norm"] = None
            row["score_coordinate_system"] = None
            if is_historical_svd_ukf:
                row["nonclaims"] = list(row.get("nonclaims", [])) + [
                    "historical SVD UKF score provenance is diagnostic only and is not admitted as a leaderboard gradient benchmark",
                    "principal-square-root UKF analytical gradient accuracy is the benchmarked UKF score standard",
                ]
            elif row.get("algorithm_id") == "ukf" and row.get("row_id") == "zhao_cui_sv_actual_nongaussian_T1000":
                row["nonclaims"] = list(row.get("nonclaims", [])) + [
                    "augmented-noise UKF value is the default nonlinear UKF diagnostic for actual SV",
                    "autodiff score provenance is diagnostic only and is not admitted as a leaderboard gradient benchmark",
                    "analytical augmented-noise UKF score remains an open route before gradient admission",
                ]
            else:
                row["nonclaims"] = list(row.get("nonclaims", [])) + [
                    "autodiff score provenance is diagnostic only and is not admitted as a leaderboard gradient benchmark",
                    "analytical gradient accuracy is the benchmarked score standard",
                ]
        updated.append(row)
    return updated


def _phase7_batch_gpu_xla_status(row: dict[str, Any]) -> dict[str, Any]:
    if row.get("row_admission_status") == "scoped_component_row_admitted":
        return {
            "scope": "scoped_component_row",
            "batch_status": "p91_sidecar_batched_evidence_scoped_component_only",
            "cpu_timing_status": "p91_sidecar_cpu_timing_scoped_component_only",
            "gpu_xla_status": "p91_sidecar_gpu_xla_scoped_component_only",
            "timing_rank_status": "not_ranked_as_full_filtering_row",
            "evidence_paths": row.get("p91_scoped_evidence", {}).get("evidence_paths", []),
            "reason": (
                "Scoped local complete-data component row; P91 timing/GPU evidence "
                "is supporting sidecar evidence only and is not full observed-data/filtering timing."
            ),
            "nonclaims": [
                "not full observed-data/filtering SIR timing",
                "not ranked as a full three-algorithm observed-data/filtering row",
            ],
        }
    full_value_score = row["comparison_status"] == "executed_value_score"
    if full_value_score:
        cpu_status = (
            "source_runtime_reused_explanatory_only"
            if row.get("runtime_seconds") is not None
            else "not_measured_in_phase7"
        )
        return {
            "scope": "main_leaderboard_row",
            "batch_status": "not_claimed_no_reviewed_batched_main_row_evaluator",
            "cpu_timing_status": cpu_status,
            "gpu_xla_status": "not_claimed_no_trusted_row_specific_gpu_xla_manifest",
            "timing_rank_status": "not_ranked_by_phase7_timing",
            "evidence_paths": [],
            "reason": (
                "Phase 7 did not run or admit a row-specific batched/GPU/XLA "
                "main leaderboard benchmark for this value/score cell."
            ),
            "nonclaims": [
                "no row-specific batched main evaluator claim",
                "no row-specific GPU/XLA timing claim",
                "not ranked by Phase 7 timing",
            ],
        }
    return {
        "scope": "main_leaderboard_row",
        "batch_status": "not_applicable_until_value_score_row_exists",
        "cpu_timing_status": "not_applicable_until_value_score_row_exists",
        "gpu_xla_status": "not_applicable_until_value_score_row_exists",
        "timing_rank_status": "not_rankable_correctness_gate_open",
        "evidence_paths": [],
        "reason": "Value/score row is not admitted, so timing cannot rank or rescue this cell.",
        "nonclaims": [
            "no batch/GPU/XLA readiness claim while value/score gate remains open",
            "not ranked by timing",
        ],
    }


def _apply_phase7_status(row: dict[str, Any]) -> dict[str, Any]:
    row = dict(row)
    row["phase7_batch_gpu_xla_status"] = _phase7_batch_gpu_xla_status(row)
    return row


def build_artifact() -> dict[str, Any]:
    scope = _load(SOURCE_SCOPE)
    p8d_cells = _p8d_cells()
    rows: list[dict[str, Any]] = []

    for row_id in HIGHDIM_ROWS:
        rows.append(_apply_score_status(_cell_for_fixed_sgqf(row_id)))
        for algorithm_id in ("ukf", "zhao_cui_scalar_or_multistate"):
            cell = p8d_cells.get((algorithm_id, row_id))
            if row_id == PARAMETERIZED_SIR_ROW and algorithm_id == "ukf":
                row = _blocked_scoped_parameterized_sir_cell("ukf")
            elif row_id == PARAMETERIZED_SIR_ROW and algorithm_id == "zhao_cui_scalar_or_multistate":
                row = _zhao_cui_parameterized_sir_local_complete_data_cell()
            elif cell is None:
                raise KeyError((algorithm_id, row_id))
            elif algorithm_id == "ukf" and row_id == "zhao_cui_sv_actual_nongaussian_T1000":
                row = _actual_sv_augmented_noise_srukf_ukf_cell()
            elif algorithm_id == "ukf" and row_id == "zhao_cui_sv_ksc_gaussian_mixture_surrogate_T1000":
                row = _ksc_source_scope_principal_sqrt_ukf_cell()
            elif algorithm_id == "zhao_cui_scalar_or_multistate" and row_id == "zhao_cui_sv_actual_nongaussian_T1000":
                row = _zhao_cui_actual_sv_tt_cell()
            elif algorithm_id == "zhao_cui_scalar_or_multistate" and row_id == "zhao_cui_sv_ksc_gaussian_mixture_surrogate_T1000":
                row = _zhao_cui_ksc_tt_cell()
            elif algorithm_id == "zhao_cui_scalar_or_multistate" and row_id == "zhao_cui_predator_prey_T20":
                row = _zhao_cui_predator_prey_tt_cell()
            elif (
                algorithm_id == "zhao_cui_scalar_or_multistate"
                and row_id == "zhao_cui_generalized_sv_synthetic_from_estimated_values"
            ):
                row = _zhao_cui_generalized_sv_tt_cell()
            else:
                row = _cell_from_p8d(algorithm_id, row_id, cell)
            row = _apply_p91_zhao_cui_status(row)
            rows.append(row)
    rows = _enforce_analytical_score_admission(rows)
    rows = _apply_value_score_route_contract(rows)
    rows = [_apply_phase7_status(row) for row in rows]
    _validate_analytical_score_contract(rows)

    return _json_safe(
        {
            "benchmark": "bayesfilter_two_lane_highdim_leaderboard_reference",
            "metadata_date": DATE,
            "manifest": {
                "source_numeric_artifact": str(SOURCE_NUMERIC.relative_to(ROOT)),
                "source_scope_contract": str(SOURCE_SCOPE.relative_to(ROOT)),
                "comparison_program_master": MASTER_PROGRAM,
                "excluded_algorithms": ["cut4", *OMITTED_ALGORITHMS],
                "lane": "highdim_source_scope",
                "execution_mode": "non_ledh_rebuild_reviewed_p8d_numeric_artifact_plus_direct_sgqf_and_p91_zhaocui_sidecar",
                "phase7_batch_gpu_xla_status_policy": (
                    "main rows are not ranked by Phase 7 timing; P91 timings "
                    "are structurally isolated under p91_scoped_evidence"
                ),
                "scoped_component_row_policy": (
                    "parameterized SIR local complete-data component rows require "
                    "explicit target_scope metadata; row id or row presence alone "
                    "must not imply full observed-data/filtering admission"
                ),
            },
            "rows": rows,
            "row_summary": _row_summary_from_rows(rows),
            "nonclaims": [
                "This highdim packet combines the reviewed P8d numeric artifact with direct SGQF row routes where already supported in code/tests.",
                "CUT4 is excluded from the highdim lane by contract.",
                "LEDH/PFPF-OT and DPF transport rows are omitted from this non-LEDH rebuild.",
                "P91 Zhao-Cui SIR d18 evidence is included only as scoped local complete-data component evidence, not as full observed-data/filtering leaderboard execution.",
                "Actual transformed SV and KSC surrogate SV remain separate rows and must not be merged.",
                "Rows with blocked or missing algorithms are not full three-way leaderboard rows.",
                "Main leaderboard rows are not a production-GPU timing packet.",
                "P91 Zhao-Cui SIR d18 CPU/GPU/XLA timings are scoped local complete-data sidecar evidence and are not full observed-data/filtering leaderboard timings.",
                "The parameterized SIR local complete-data component row is metadata-scoped and not a full observed-data/filtering row.",
            ],
            "source_scope_row_ids": list(scope["source_scope_row_ids"]) + [
                PARAMETERIZED_SIR_ROW
            ],
            "comparison_algorithm_ids": HIGHDIM_ALGOS,
        }
    )


def build_artifact_from_cached_baseline(
    baseline: dict[str, Any],
    scoped_patch: dict[str, Any],
    *,
    baseline_path: Path = CURRENT_AUTH_JSON,
    scoped_patch_path: Path = ROOT
    / "docs/plans/bayesfilter-two-lane-highdim-leaderboard-results-2026-07-03-scoped-zhaocui-sir-row.json",
) -> dict[str, Any]:
    """Merge the validated scoped SIR row into a frozen full leaderboard.

    This mode is deliberately narrow: it is a reproducible fallback for Phase 4
    when unrelated full-runner rows exceed the visible runtime gate. It does not
    recompute, retune, or reinterpret any cached baseline row.
    """

    rows = list(baseline["rows"])
    patch_rows = [
        dict(row)
        for row in scoped_patch["rows"]
        if row.get("row_id") == PARAMETERIZED_SIR_ROW
    ]
    if {row.get("algorithm_id") for row in patch_rows} != set(HIGHDIM_ALGOS):
        raise ValueError("scoped patch must contain exactly the three comparison algorithms")

    rows = [row for row in rows if row.get("row_id") != PARAMETERIZED_SIR_ROW]
    rows.extend(patch_rows)
    rows = _enforce_analytical_score_admission(rows)
    rows = _apply_value_score_route_contract(rows)
    rows = [_apply_phase7_status(row) for row in rows]
    _validate_analytical_score_contract(rows)

    manifest = dict(baseline.get("manifest", {}))
    manifest.update(
        {
            "execution_mode": (
                "split_merge_cached_july1_full_leaderboard_plus_validated_"
                "scoped_zhaocui_sir_component_row"
            ),
            "cached_baseline_artifact": _display_path(baseline_path),
            "scoped_patch_artifact": _display_path(scoped_patch_path),
            "split_merge_reason": (
                "full runner exceeded the visible gate window; unaffected rows "
                "are preserved from the frozen July 1 full artifact"
            ),
            "scoped_component_row_policy": (
                "parameterized SIR local complete-data component rows require "
                "explicit target_scope metadata; row id or row presence alone "
                "must not imply full observed-data/filtering admission"
            ),
        }
    )

    nonclaims = list(baseline.get("nonclaims", []))
    for item in [
        "This July 3 artifact uses a split/merge regeneration: unaffected rows are preserved from the frozen July 1 full leaderboard artifact.",
        "The parameterized SIR local complete-data component row is metadata-scoped and not a full observed-data/filtering row.",
        "The split/merge artifact is not evidence that unrelated expensive rows were rerun on July 3.",
    ]:
        if item not in nonclaims:
            nonclaims.append(item)

    row_ids = list(baseline.get("source_scope_row_ids", []))
    if PARAMETERIZED_SIR_ROW not in row_ids:
        row_ids.append(PARAMETERIZED_SIR_ROW)

    return _json_safe(
        {
            "benchmark": baseline.get(
                "benchmark",
                "bayesfilter_two_lane_highdim_leaderboard_reference",
            ),
            "metadata_date": "2026-07-03",
            "manifest": manifest,
            "rows": rows,
            "row_summary": _row_summary_from_rows(rows),
            "nonclaims": nonclaims,
            "source_scope_row_ids": row_ids,
            "comparison_algorithm_ids": baseline.get("comparison_algorithm_ids", HIGHDIM_ALGOS),
        }
    )


def _fmt(x: float | None) -> str:
    if x is None:
        return "n/a"
    return f"{x:.6f}" if abs(x) < 1000 else f"{x:.3f}"


def _display_path(path: Path) -> str:
    resolved = path if path.is_absolute() else ROOT / path
    try:
        return str(resolved.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def _markdown(payload: dict[str, Any], json_path: Path) -> str:
    lines = [
        "# Two-Lane Highdim Leaderboard Result",
        "",
        f"Authoritative JSON artifact: `{_display_path(json_path)}`.",
        "",
        "## Executed / status cells",
        "",
        "| Row | Algorithm | Status | Score status | Batch status | GPU/XLA status | Timing rank status | Avg loglik | Runtime s | MC SE | Reason |",
        "| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in payload["rows"]:
        phase7_status = row.get("phase7_batch_gpu_xla_status", {})
        lines.append(
            f"| {row['row_id']} | {row['algorithm_id']} | {row['comparison_status']} | {row.get('score_status') or ''} | {phase7_status.get('batch_status') or ''} | {phase7_status.get('gpu_xla_status') or ''} | {phase7_status.get('timing_rank_status') or ''} | {_fmt(row['average_log_likelihood'])} | {_fmt(row['runtime_seconds'])} | {_fmt(row['mc_standard_error'])} | {row['reason'] or row.get('score_status_reason') or ''} |"
        )
    lines.extend([
        "",
        "## Row readiness summary",
        "",
        "| Row | Scope | Executed algorithms | Full three-way ready | Scoped component ready | Blocked / missing algorithms |",
        "| --- | --- | --- | --- | --- | --- |",
    ])
    for row in payload["row_summary"]:
        lines.append(
            f"| {row['row_id']} | {row.get('row_scope') or ''} | {', '.join(row['executed_algorithms']) or 'none'} | {row['full_three_way_ready']} | {row.get('scoped_component_ready')} | {', '.join(row['blocked_or_missing_algorithms']) or 'none'} |"
        )
    lines.extend([
        "",
        "## Nonclaims",
        "",
    ])
    lines.extend(f"- {item}" for item in payload["nonclaims"])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_MD)
    parser.add_argument(
        "--cached-baseline",
        type=Path,
        default=None,
        help="Optional frozen full leaderboard JSON for split/merge regeneration.",
    )
    parser.add_argument(
        "--scoped-patch",
        type=Path,
        default=None,
        help="Optional validated scoped-row JSON to merge into --cached-baseline.",
    )
    args = parser.parse_args()
    if (args.cached_baseline is None) != (args.scoped_patch is None):
        parser.error("--cached-baseline and --scoped-patch must be supplied together")
    if args.cached_baseline is not None and args.scoped_patch is not None:
        payload = build_artifact_from_cached_baseline(
            _load(args.cached_baseline),
            _load(args.scoped_patch),
            baseline_path=args.cached_baseline,
            scoped_patch_path=args.scoped_patch,
        )
    else:
        payload = build_artifact()
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.markdown_output.write_text(_markdown(payload, args.output) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
