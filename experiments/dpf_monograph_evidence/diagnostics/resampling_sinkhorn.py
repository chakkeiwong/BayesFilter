from __future__ import annotations

import math
from typing import Any

import numpy as np

from experiments.dpf_monograph_evidence.fixtures.resampling_sinkhorn import (
    SinkhornResidualFixture,
    SoftResamplingBiasFixture,
)

SOFT_RESAMPLING_NON_IMPLICATION = (
    "IE5 soft-resampling diagnostics validate only deterministic two-particle "
    "relaxed-target expectation arithmetic for selected affine and nonlinear test "
    "functions. They do not validate categorical resampling law preservation, "
    "unbiasedness for nonlinear observables, posterior equivalence, production "
    "bayesfilter code, banking use, model-risk use, or production readiness."
)

SINKHORN_NON_IMPLICATION = (
    "IE5 Sinkhorn diagnostics validate only finite-budget marginal residuals for a "
    "small deterministic regularized transport fixture. They do not validate exact "
    "unregularized OT equivalence, exact EOT equivalence at finite epsilon/iteration "
    "budget, posterior equivalence, production bayesfilter code, banking use, "
    "model-risk use, or production readiness."
)

SOFT_RESAMPLING_COMPARATOR_ID = "closed_form_two_particle_soft_resampling_reference"
SINKHORN_COMPARATOR_ID = "manual_balanced_sinkhorn_marginal_reference"


def _residual_entry(observed: float, threshold: float) -> dict[str, Any]:
    finite = bool(math.isfinite(observed))
    return {
        "threshold": float(threshold),
        "observed": float(observed),
        "finite": finite,
    }


def _sign_entry(observed_positive: bool) -> dict[str, Any]:
    return {
        "threshold": 0.0,
        "observed": 0.0 if observed_positive else 1.0,
        "finite": True,
    }


def _finite_descriptive_entry(observed: float) -> dict[str, Any]:
    return {
        "threshold": float("inf"),
        "observed": float(observed),
        "finite": bool(math.isfinite(observed)),
    }


def evaluate_soft_resampling_bias_fixture(fixture: SoftResamplingBiasFixture) -> dict[str, Any]:
    particles = np.asarray(fixture.particles, dtype=np.float64)
    base_weights = np.asarray(fixture.base_weights, dtype=np.float64)
    alpha = float(fixture.relaxed_mixture_parameter)
    categorical_reference = base_weights.copy()
    uniform_probabilities = np.full(base_weights.shape, 1.0 / float(base_weights.size), dtype=np.float64)
    relaxed_probabilities = (1.0 - alpha) * base_weights + alpha * uniform_probabilities
    closed_form_relaxed_probabilities = np.asarray(
        (
            (1.0 - alpha) * fixture.base_weights[0] + alpha / 2.0,
            (1.0 - alpha) * fixture.base_weights[1] + alpha / 2.0,
        ),
        dtype=np.float64,
    )

    affine_constant_values = np.ones_like(particles)
    affine_identity_values = particles
    affine_linear_summary_values = 2.0 * particles - 1.0
    nonlinear_values = np.asarray(fixture.nonlinear_test_values, dtype=np.float64)

    def expectation(values: np.ndarray, probs: np.ndarray) -> float:
        return float(np.sum(values * probs))

    constant_relaxed = expectation(affine_constant_values, relaxed_probabilities)
    constant_reference = expectation(affine_constant_values, categorical_reference)
    identity_relaxed = expectation(affine_identity_values, relaxed_probabilities)
    identity_reference = expectation(affine_identity_values, categorical_reference)
    linear_relaxed = expectation(affine_linear_summary_values, relaxed_probabilities)
    linear_reference = expectation(affine_linear_summary_values, categorical_reference)
    nonlinear_relaxed = expectation(nonlinear_values, relaxed_probabilities)
    nonlinear_reference = expectation(nonlinear_values, categorical_reference)
    nonlinear_delta = nonlinear_relaxed - nonlinear_reference
    nonlinear_delta_sign_expected = nonlinear_delta != 0.0 and math.isfinite(nonlinear_delta)
    probability_sum_abs = abs(float(np.sum(relaxed_probabilities)) - 1.0)
    relaxed_probability_formula_abs = float(np.max(np.abs(relaxed_probabilities - closed_form_relaxed_probabilities)))

    tolerance = {
        "relaxed_probability_formula_abs": _residual_entry(relaxed_probability_formula_abs, 1e-12),
        "relaxed_constant_expectation_abs": _residual_entry(abs(constant_relaxed - 1.0), 1e-12),
        "relaxed_identity_expectation_abs": _residual_entry(abs(identity_relaxed - expectation(affine_identity_values, closed_form_relaxed_probabilities)), 1e-12),
        "relaxed_linear_summary_abs": _residual_entry(abs(linear_relaxed - expectation(affine_linear_summary_values, closed_form_relaxed_probabilities)), 1e-12),
        "relaxed_nonlinear_expectation_abs": _residual_entry(abs(nonlinear_relaxed - expectation(nonlinear_values, closed_form_relaxed_probabilities)), 1e-12),
        "categorical_identity_delta_abs": _finite_descriptive_entry(abs(identity_relaxed - identity_reference)),
        "categorical_linear_summary_delta_abs": _finite_descriptive_entry(abs(linear_relaxed - linear_reference)),
        "categorical_nonlinear_delta_abs": _finite_descriptive_entry(abs(nonlinear_delta)),
        "categorical_nonlinear_delta_sign_expected": _sign_entry(nonlinear_delta_sign_expected),
        "probability_sum_abs": _residual_entry(probability_sum_abs, 1e-12),
    }

    finite_summary = all(value["finite"] for value in tolerance.values())
    finite_checks = {
        "normalized_base_weights": base_weights.tolist(),
        "relaxed_mixture_parameter": alpha,
        "relaxed_probabilities": relaxed_probabilities.tolist(),
        "closed_form_relaxed_probabilities": closed_form_relaxed_probabilities.tolist(),
        "categorical_reference_probabilities": categorical_reference.tolist(),
        "affine_test_function_values": {
            "constant": affine_constant_values.tolist(),
            "identity": affine_identity_values.tolist(),
            "linear_summary": affine_linear_summary_values.tolist(),
        },
        "affine_expectations": {
            "constant_relaxed": constant_relaxed,
            "constant_reference": constant_reference,
            "constant_closed_form_relaxed": 1.0,
            "identity_relaxed": identity_relaxed,
            "identity_reference": identity_reference,
            "identity_closed_form_relaxed": expectation(affine_identity_values, closed_form_relaxed_probabilities),
            "identity_categorical_delta": identity_relaxed - identity_reference,
            "linear_summary_relaxed": linear_relaxed,
            "linear_summary_reference": linear_reference,
            "linear_summary_closed_form_relaxed": expectation(affine_linear_summary_values, closed_form_relaxed_probabilities),
            "linear_summary_categorical_delta": linear_relaxed - linear_reference,
        },
        "nonlinear_test_function_values": nonlinear_values.tolist(),
        "nonlinear_expectations": {
            "relaxed_expectation": nonlinear_relaxed,
            "categorical_expectation": nonlinear_reference,
            "closed_form_relaxed_expectation": expectation(nonlinear_values, closed_form_relaxed_probabilities),
            "delta_value": nonlinear_delta,
            "delta_sign": "positive" if nonlinear_delta > 0.0 else "negative" if nonlinear_delta < 0.0 else "zero",
        },
        "finite_summary": finite_summary,
    }
    shape_checks = {
        "particle_count": int(particles.size),
        "particle_value_dimension": 1,
        "test_function_count": 4,
        "deterministic_fixture_id": fixture.fixture_id,
    }

    return {
        "fixture_id": fixture.fixture_id,
        "tolerance": tolerance,
        "finite_checks": finite_checks,
        "shape_checks": shape_checks,
    }


def _sinkhorn_plan(cost_matrix: np.ndarray, source: np.ndarray, target: np.ndarray, epsilon: float, iterations: int) -> np.ndarray:
    log_kernel = -cost_matrix / epsilon
    log_u = np.zeros_like(source)
    log_v = np.zeros_like(target)

    for _ in range(iterations):
        log_u = np.log(source) - np.logaddexp.reduce(log_kernel + log_v[None, :], axis=1)
        log_v = np.log(target) - np.logaddexp.reduce((log_kernel + log_u[:, None]), axis=0)

    return np.exp(log_kernel + log_u[:, None] + log_v[None, :])


def evaluate_sinkhorn_residual_fixture(fixture: SinkhornResidualFixture) -> dict[str, Any]:
    source = np.asarray(fixture.source_marginal, dtype=np.float64)
    target = np.asarray(fixture.target_marginal, dtype=np.float64)
    cost_matrix = np.asarray(fixture.cost_matrix, dtype=np.float64)
    budgets = list(fixture.budget_ladder)

    per_budget_row_residual_max: list[float] = []
    per_budget_column_residual_max: list[float] = []
    per_budget_total_residual_max: list[float] = []
    plans: dict[int, np.ndarray] = {}

    for budget in budgets:
        plan = _sinkhorn_plan(cost_matrix, source, target, fixture.epsilon, budget)
        plans[budget] = plan
        row_residual = float(np.max(np.abs(np.sum(plan, axis=1) - source)))
        column_residual = float(np.max(np.abs(np.sum(plan, axis=0) - target)))
        total_residual = max(row_residual, column_residual)
        per_budget_row_residual_max.append(row_residual)
        per_budget_column_residual_max.append(column_residual)
        per_budget_total_residual_max.append(total_residual)

    final_budget = budgets[-1]
    final_plan = plans[final_budget]
    row_marginal_abs_max = per_budget_row_residual_max[-1]
    column_marginal_abs_max = per_budget_column_residual_max[-1]
    total_mass_abs = abs(float(np.sum(final_plan)) - float(np.sum(source)))
    min_entry = float(np.min(final_plan))
    nonnegative_plan_violation_abs = abs(min_entry) if min_entry < 0.0 else 0.0
    finite_plan_violation_abs = 0.0 if np.isfinite(final_plan).all() else 1.0
    budget_violations = 0
    for previous, current in zip(per_budget_total_residual_max, per_budget_total_residual_max[1:]):
        if current > previous + 1e-15:
            budget_violations += 1

    tolerance = {
        "row_marginal_abs_max": _residual_entry(row_marginal_abs_max, 1e-9),
        "column_marginal_abs_max": _residual_entry(column_marginal_abs_max, 1e-9),
        "total_mass_abs": _residual_entry(total_mass_abs, 1e-12),
        "nonnegative_plan_violation_abs": _residual_entry(nonnegative_plan_violation_abs, 0.0),
        "finite_plan_violation_abs": _residual_entry(finite_plan_violation_abs, 0.0),
        "budget_residual_nonincrease_violations": _residual_entry(float(budget_violations), 0.0),
    }

    finite_summary = all(value["finite"] for value in tolerance.values())
    finite_checks = {
        "epsilon": fixture.epsilon,
        "stabilization_mode": fixture.stabilization_mode,
        "budget_ladder": budgets,
        "row_marginals": source.tolist(),
        "column_marginals": target.tolist(),
        "cost_matrix": cost_matrix.tolist(),
        "final_transport_plan": final_plan.tolist(),
        "per_budget_row_residual_max": per_budget_row_residual_max,
        "per_budget_column_residual_max": per_budget_column_residual_max,
        "per_budget_total_residual_max": per_budget_total_residual_max,
        "budget_residuals_nonincreasing": budget_violations == 0,
        "finite_summary": finite_summary,
    }
    shape_checks = {
        "source_support_size": int(source.size),
        "target_support_size": int(target.size),
        "cost_matrix_shape": list(cost_matrix.shape),
        "source_marginal_shape": list(source.shape),
        "target_marginal_shape": list(target.shape),
        "final_plan_shape": list(final_plan.shape),
        "budget_point_count": len(budgets),
    }

    return {
        "fixture_id": fixture.fixture_id,
        "tolerance": tolerance,
        "finite_checks": finite_checks,
        "shape_checks": shape_checks,
    }


def soft_resampling_status(metrics: dict[str, Any]) -> str:
    tolerance = metrics["tolerance"]
    promotion_keys = (
        "relaxed_probability_formula_abs",
        "relaxed_constant_expectation_abs",
        "relaxed_identity_expectation_abs",
        "relaxed_linear_summary_abs",
        "relaxed_nonlinear_expectation_abs",
        "probability_sum_abs",
    )
    for key in promotion_keys:
        value = tolerance[key]
        if (not value["finite"]) or value["observed"] > value["threshold"]:
            return "fail"
    for key in (
        "categorical_identity_delta_abs",
        "categorical_linear_summary_delta_abs",
        "categorical_nonlinear_delta_abs",
    ):
        if not tolerance[key]["finite"]:
            return "fail"
    sign_entry = tolerance["categorical_nonlinear_delta_sign_expected"]
    if (not sign_entry["finite"]) or sign_entry["observed"] > sign_entry["threshold"]:
        return "fail"
    return "pass"


def sinkhorn_status(metrics: dict[str, Any]) -> str:
    for value in metrics["tolerance"].values():
        if (not value["finite"]) or value["observed"] > value["threshold"]:
            return "fail"
    return "pass"


def repair_trigger_for_soft_resampling(metrics: dict[str, Any]) -> str:
    tolerance = metrics["tolerance"]
    if (not tolerance["relaxed_probability_formula_abs"]["finite"]) or tolerance["relaxed_probability_formula_abs"]["observed"] > tolerance["relaxed_probability_formula_abs"]["threshold"]:
        return "probability formula"
    if (not tolerance["probability_sum_abs"]["finite"]) or tolerance["probability_sum_abs"]["observed"] > tolerance["probability_sum_abs"]["threshold"]:
        return "probability normalization"
    for key in (
        "relaxed_constant_expectation_abs",
        "relaxed_identity_expectation_abs",
        "relaxed_linear_summary_abs",
        "relaxed_nonlinear_expectation_abs",
    ):
        if (not tolerance[key]["finite"]) or tolerance[key]["observed"] > tolerance[key]["threshold"]:
            return "relaxed expectation"
    for key in (
        "categorical_identity_delta_abs",
        "categorical_linear_summary_delta_abs",
        "categorical_nonlinear_delta_abs",
    ):
        if not tolerance[key]["finite"]:
            return "categorical delta finiteness"
    if (not tolerance["categorical_nonlinear_delta_sign_expected"]["finite"]) or tolerance["categorical_nonlinear_delta_sign_expected"]["observed"] > tolerance["categorical_nonlinear_delta_sign_expected"]["threshold"]:
        return "nonlinear delta sign"
    return "none"


def repair_trigger_for_sinkhorn(metrics: dict[str, Any]) -> str:
    tolerance = metrics["tolerance"]
    order = [
        ("row_marginal_abs_max", "row marginal"),
        ("column_marginal_abs_max", "column marginal"),
        ("total_mass_abs", "mass"),
        ("nonnegative_plan_violation_abs", "nonnegativity"),
        ("finite_plan_violation_abs", "finite-plan"),
        ("budget_residual_nonincrease_violations", "budget trend"),
    ]
    for key, label in order:
        value = tolerance[key]
        if (not value["finite"]) or value["observed"] > value["threshold"]:
            return label
    return "none"
