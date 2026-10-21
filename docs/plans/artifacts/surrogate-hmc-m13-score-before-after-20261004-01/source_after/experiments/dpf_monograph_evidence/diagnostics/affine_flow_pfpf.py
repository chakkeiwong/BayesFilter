from __future__ import annotations

import math
from typing import Any

import numpy as np

from experiments.dpf_monograph_evidence.fixtures.affine_flow import SyntheticAffineFlowFixture

ABS_TOL = 1e-12
SYNTHETIC_AFFINE_FLOW_NON_IMPLICATION = (
    "IE4 synthetic affine-flow checks validate only closed-form affine "
    "pushforward-density, inverse-map, and Jacobian-sign parity on deterministic "
    "clean-room fixtures. They do not validate nonlinear flow integration, solver "
    "stability, PF-PF filtering correctness, production bayesfilter code, real "
    "DPF-HMC targets, posterior quality, banking use, model-risk use, or production "
    "readiness."
)
PFPF_ALGEBRA_PARITY_NON_IMPLICATION = (
    "IE4 PF-PF algebra parity checks validate only closed-form proposal-density, "
    "Jacobian-sign, unnormalized corrected-log-weight, and normalized-weight parity "
    "on deterministic affine clean-room fixtures. They do not validate nonlinear "
    "flow integration, solver stability, filtering correctness, production "
    "bayesfilter code, real DPF-HMC targets, posterior quality, banking use, "
    "model-risk use, or production readiness."
)
RESIDUAL_KEYS = (
    "forward_reconstruction_abs_max",
    "inverse_reconstruction_abs_max",
    "log_det_abs_max",
    "pushforward_log_density_abs_max",
    "proposal_log_density_abs_max",
    "corrected_log_weight_abs_max",
    "normalized_weight_abs_max",
    "probability_sum_abs_max",
)


def _as_array(rows: tuple[tuple[float, ...], ...] | tuple[float, ...]) -> np.ndarray:
    return np.asarray(rows, dtype=np.float64)


def _logsumexp(values: np.ndarray) -> float:
    anchor = float(np.max(values))
    return anchor + math.log(float(np.sum(np.exp(values - anchor))))


def _multivariate_logpdf(points: np.ndarray, mean: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    centered = points - mean
    sign, log_det = np.linalg.slogdet(covariance)
    if sign <= 0.0:
        raise ValueError("covariance must be positive definite")
    precision = np.linalg.inv(covariance)
    quadratic = np.einsum("...i,ij,...j->...", centered, precision, centered)
    dimension = mean.shape[0]
    normalizer = dimension * math.log(2.0 * math.pi) + log_det
    return -0.5 * (normalizer + quadratic)


def _closed_form_2d_logpdf(points: np.ndarray, mean: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    if points.shape[-1] != 2:
        raise ValueError("closed-form analytic comparator is defined for 2D fixtures only")
    a = float(covariance[0, 0])
    b = float(covariance[0, 1])
    c = float(covariance[1, 1])
    determinant = a * c - b * b
    if determinant <= 0.0:
        raise ValueError("closed-form analytic comparator requires positive definite covariance")
    centered = points - mean
    x0 = centered[:, 0]
    x1 = centered[:, 1]
    quadratic = (c * x0 * x0 - 2.0 * b * x0 * x1 + a * x1 * x1) / determinant
    normalizer = 2.0 * math.log(2.0 * math.pi) + math.log(determinant)
    return -0.5 * (normalizer + quadratic)


def _closed_form_2d_log_abs_det(matrix: np.ndarray) -> float:
    if matrix.shape != (2, 2):
        raise ValueError("closed-form determinant comparator is defined for 2x2 fixtures only")
    determinant = float(matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0])
    if determinant == 0.0:
        raise ValueError("fixture matrix must be invertible")
    return math.log(abs(determinant))


def residual_entry(observed: float, threshold: float = ABS_TOL) -> dict[str, Any]:
    finite = bool(math.isfinite(observed))
    return {
        "threshold": threshold,
        "observed": float(observed),
        "finite": finite,
    }


def evaluate_affine_flow_fixture(fixture: SyntheticAffineFlowFixture) -> dict[str, Any]:
    matrix = _as_array(fixture.matrix)
    offset = _as_array(fixture.offset)
    base_mean = _as_array(fixture.base_mean)
    base_covariance = _as_array(fixture.base_covariance)
    base_particles = _as_array(fixture.base_particles)
    observation_particles = _as_array(fixture.observation_particles)
    proposal_noise_mean = _as_array(fixture.proposal_noise_mean)
    proposal_noise_covariance = _as_array(fixture.proposal_noise_covariance)

    determinant = float(np.linalg.det(matrix))
    if determinant == 0.0:
        raise ValueError("fixture matrix must be invertible")
    inverse_matrix = np.linalg.inv(matrix)
    determinant_sign = "positive" if determinant > 0.0 else "negative"
    implemented_log_det = float(math.log(abs(determinant)))
    analytic_log_det = _closed_form_2d_log_abs_det(matrix)

    forward_particles = base_particles @ matrix.T + offset
    reconstructed_base = (forward_particles - offset) @ inverse_matrix.T
    forward_reconstruction_abs_max = float(np.max(np.abs(forward_particles - (base_particles @ matrix.T + offset))))
    inverse_reconstruction_abs_max = float(np.max(np.abs(reconstructed_base - base_particles)))

    implemented_base_log_density = _multivariate_logpdf(base_particles, base_mean, base_covariance)
    analytic_base_log_density = _closed_form_2d_logpdf(base_particles, base_mean, base_covariance)
    implemented_pushforward_log_density = _multivariate_logpdf(
        (forward_particles - offset) @ inverse_matrix.T,
        base_mean,
        base_covariance,
    ) - implemented_log_det
    analytic_pushforward_log_density = analytic_base_log_density - analytic_log_det
    pushforward_log_density_abs_max = float(
        np.max(np.abs(implemented_pushforward_log_density - analytic_pushforward_log_density))
    )
    log_det_abs_max = abs(implemented_log_det - analytic_log_det)

    inverse_observation_particles = (observation_particles - offset) @ inverse_matrix.T
    implemented_proposal_log_density = _multivariate_logpdf(
        inverse_observation_particles,
        proposal_noise_mean,
        proposal_noise_covariance,
    ) - implemented_log_det
    analytic_proposal_log_density = _closed_form_2d_logpdf(
        inverse_observation_particles,
        proposal_noise_mean,
        proposal_noise_covariance,
    ) - analytic_log_det
    proposal_log_density_abs_max = float(
        np.max(np.abs(implemented_proposal_log_density - analytic_proposal_log_density))
    )

    prior_log_density = _as_array(fixture.prior_log_density)
    target_log_density = _as_array(fixture.target_log_density)
    corrected_log_weight = prior_log_density + target_log_density - implemented_proposal_log_density
    analytic_corrected_log_weight = prior_log_density + target_log_density - analytic_proposal_log_density
    corrected_log_weight_abs_max = float(
        np.max(np.abs(corrected_log_weight - analytic_corrected_log_weight))
    )

    implemented_log_normalizer = _logsumexp(corrected_log_weight)
    analytic_log_normalizer = _logsumexp(analytic_corrected_log_weight)
    implemented_normalized_weights = np.exp(corrected_log_weight - implemented_log_normalizer)
    analytic_normalized_weights = np.exp(analytic_corrected_log_weight - analytic_log_normalizer)
    normalized_weight_abs_max = float(
        np.max(np.abs(implemented_normalized_weights - analytic_normalized_weights))
    )
    probability_sum_abs_max = abs(float(np.sum(implemented_normalized_weights)) - 1.0)

    tolerance = {
        key: residual_entry(0.0) for key in RESIDUAL_KEYS
    }
    tolerance["forward_reconstruction_abs_max"] = residual_entry(forward_reconstruction_abs_max)
    tolerance["inverse_reconstruction_abs_max"] = residual_entry(inverse_reconstruction_abs_max)
    tolerance["log_det_abs_max"] = residual_entry(log_det_abs_max)
    tolerance["pushforward_log_density_abs_max"] = residual_entry(pushforward_log_density_abs_max)
    tolerance["proposal_log_density_abs_max"] = residual_entry(proposal_log_density_abs_max)
    tolerance["corrected_log_weight_abs_max"] = residual_entry(corrected_log_weight_abs_max)
    tolerance["normalized_weight_abs_max"] = residual_entry(normalized_weight_abs_max)
    tolerance["probability_sum_abs_max"] = residual_entry(probability_sum_abs_max)

    finite_checks = {
        "residuals": {key: dict(value) for key, value in tolerance.items()},
        "finite_summary": all(value["finite"] for value in tolerance.values()),
    }

    shape_checks = {
        "state_dimension": int(fixture.dimension),
        "particle_count": int(base_particles.shape[0]),
        "observation_particle_count": int(observation_particles.shape[0]),
        "matrix_shape": list(matrix.shape),
        "offset_shape": list(offset.shape),
    }

    return {
        "fixture_id": fixture.fixture_id,
        "dimension": fixture.dimension,
        "matrix": matrix.tolist(),
        "offset": offset.tolist(),
        "determinant": determinant,
        "determinant_sign": determinant_sign,
        "inverse_matrix": inverse_matrix.tolist(),
        "base_particles": base_particles.tolist(),
        "forward_particles": forward_particles.tolist(),
        "observation_particles": observation_particles.tolist(),
        "inverse_observation_particles": inverse_observation_particles.tolist(),
        "analytic_base_log_density": analytic_base_log_density.tolist(),
        "implemented_base_log_density": implemented_base_log_density.tolist(),
        "analytic_pushforward_log_density": analytic_pushforward_log_density.tolist(),
        "implemented_pushforward_log_density": implemented_pushforward_log_density.tolist(),
        "analytic_log_det": analytic_log_det,
        "implemented_log_det": implemented_log_det,
        "analytic_proposal_log_density": analytic_proposal_log_density.tolist(),
        "implemented_proposal_log_density": implemented_proposal_log_density.tolist(),
        "prior_log_density": prior_log_density.tolist(),
        "target_log_density": target_log_density.tolist(),
        "corrected_log_weight": corrected_log_weight.tolist(),
        "analytic_corrected_log_weight": analytic_corrected_log_weight.tolist(),
        "implemented_normalized_weights": implemented_normalized_weights.tolist(),
        "analytic_normalized_weights": analytic_normalized_weights.tolist(),
        "implemented_log_normalizer": implemented_log_normalizer,
        "analytic_log_normalizer": analytic_log_normalizer,
        "proposal_sign_convention": "log q(x_t | x_{t-1}, y_t) = log p_Z(A^{-1}(x_t - b)) - log|det A| using the affine inverse map.",
        "tolerance": tolerance,
        "finite_checks": finite_checks,
        "shape_checks": shape_checks,
    }


def repair_trigger_for_row(diagnostic_id: str, metrics: dict[str, Any]) -> str:
    exceeded = [
        key for key, value in metrics["tolerance"].items() if (not value["finite"]) or value["observed"] > value["threshold"]
    ]
    if not exceeded:
        return "none"
    if diagnostic_id == "synthetic_affine_flow":
        mapping = {
            "forward_reconstruction_abs_max": "map inversion",
            "inverse_reconstruction_abs_max": "map inversion",
            "log_det_abs_max": "determinant sign",
            "pushforward_log_density_abs_max": "pushforward-density mismatch",
            "proposal_log_density_abs_max": "base density",
        }
    else:
        mapping = {
            "proposal_log_density_abs_max": "proposal density",
            "log_det_abs_max": "log-det sign",
            "corrected_log_weight_abs_max": "target density",
            "normalized_weight_abs_max": "log-sum-exp normalization",
            "probability_sum_abs_max": "log-sum-exp normalization",
        }
    labels = []
    for key in exceeded:
        labels.append(mapping.get(key, "finite-value failure"))
    ordered = []
    for label in labels:
        if label not in ordered:
            ordered.append(label)
    return ", ".join(ordered)
