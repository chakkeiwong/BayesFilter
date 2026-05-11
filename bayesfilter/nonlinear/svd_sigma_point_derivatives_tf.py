"""Analytic first-order derivatives for TF SVD/eigen sigma-point filters."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import tensorflow as tf

from bayesfilter.diagnostics import TFFilterDiagnostics, TFRegularizationDiagnostics
from bayesfilter.linear.svd_factor_tf import (
    eigh_logdet,
    eigh_solve,
    floor_count,
    psd_eigh,
    symmetrize,
)
from bayesfilter.nonlinear.cut_tf import tf_cut4g_sigma_point_rule
from bayesfilter.nonlinear.sigma_points_tf import TFSigmaPointRule, tf_unit_sigma_point_rule
from bayesfilter.results_tf import TFFilterDerivativeResult
from bayesfilter.structural_tf import (
    TFStructuralStateSpace,
    structural_block_metadata,
    structural_filter_metadata,
)


TFTransitionStateJacobianFn = Callable[[tf.Tensor, tf.Tensor], tf.Tensor]
TFTransitionInnovationJacobianFn = Callable[[tf.Tensor, tf.Tensor], tf.Tensor]
TFTransitionParameterDerivativeFn = Callable[[tf.Tensor, tf.Tensor], tf.Tensor]
TFObservationStateJacobianFn = Callable[[tf.Tensor], tf.Tensor]
TFObservationParameterDerivativeFn = Callable[[tf.Tensor], tf.Tensor]


@dataclass(frozen=True)
class TFStructuralFirstDerivatives:
    """First-order derivatives required by analytic sigma-point scores.

    The map derivative callables return partial derivatives with sigma-point
    inputs held fixed.  Point-dependence is then propagated by the analytic
    sigma-point chain rule.
    """

    d_initial_mean: tf.Tensor
    d_initial_covariance: tf.Tensor
    d_innovation_covariance: tf.Tensor
    d_observation_covariance: tf.Tensor
    transition_state_jacobian_fn: TFTransitionStateJacobianFn
    transition_innovation_jacobian_fn: TFTransitionInnovationJacobianFn
    d_transition_fn: TFTransitionParameterDerivativeFn
    observation_state_jacobian_fn: TFObservationStateJacobianFn
    d_observation_fn: TFObservationParameterDerivativeFn
    name: str = "tf_structural_first_derivatives"

    def __post_init__(self) -> None:
        for name in (
            "d_initial_mean",
            "d_initial_covariance",
            "d_innovation_covariance",
            "d_observation_covariance",
        ):
            object.__setattr__(
                self,
                name,
                tf.convert_to_tensor(getattr(self, name), dtype=tf.float64),
            )
        self.validate_static_shapes()

    @property
    def parameter_dim(self) -> int | None:
        return self.d_initial_mean.shape[0]

    @property
    def state_dim(self) -> int | None:
        return self.d_initial_mean.shape[-1]

    @property
    def innovation_dim(self) -> int | None:
        return self.d_innovation_covariance.shape[-1]

    @property
    def observation_dim(self) -> int | None:
        return self.d_observation_covariance.shape[-1]

    def validate_static_shapes(self) -> None:
        p = self.parameter_dim
        n = self.state_dim
        q = self.innovation_dim
        m = self.observation_dim
        if p is None or n is None or q is None or m is None:
            return
        expected = {
            "d_initial_mean": (p, n),
            "d_initial_covariance": (p, n, n),
            "d_innovation_covariance": (p, q, q),
            "d_observation_covariance": (p, m, m),
        }
        for name, shape in expected.items():
            actual = tuple(getattr(self, name).shape.as_list())
            if actual != shape:
                raise ValueError(f"{name} has shape {actual}, expected {shape}")


@dataclass(frozen=True)
class TFSmoothEighFactorFirstDerivatives:
    """Smooth-branch eigensystem and first derivatives of a PSD factor."""

    eigenvalues: tf.Tensor
    floored_eigenvalues: tf.Tensor
    eigenvectors: tf.Tensor
    factor: tf.Tensor
    d_factor: tf.Tensor
    implemented_covariance: tf.Tensor
    floor_count: tf.Tensor
    min_eigen_gap: tf.Tensor
    psd_projection_residual: tf.Tensor
    derivative_reconstruction_residual: tf.Tensor


def _as_observation_matrix(observations: tf.Tensor) -> tf.Tensor:
    y = tf.convert_to_tensor(observations, dtype=tf.float64)
    if y.shape.rank == 1:
        y = y[:, tf.newaxis]
    if y.shape.rank != 2:
        raise ValueError("observations must be one- or two-dimensional")
    return y


def _static_dim(tensor: tf.Tensor, axis: int, name: str) -> int:
    dim = tensor.shape[axis]
    if dim is None:
        raise ValueError(f"{name} must be statically known")
    return int(dim)


def _static_num_timesteps(observations: tf.Tensor) -> int:
    return _static_dim(observations, 0, "observation length")


def _weighted_covariance(centered: tf.Tensor, weights: tf.Tensor) -> tf.Tensor:
    return symmetrize(tf.transpose(centered) @ (centered * weights[:, tf.newaxis]))


def _weighted_covariance_first_derivatives(
    centered: tf.Tensor,
    d_centered: tf.Tensor,
    weights: tf.Tensor,
) -> tf.Tensor:
    return symmetrize(
        tf.einsum("r,pri,rj->pij", weights, d_centered, centered)
        + tf.einsum("r,ri,prj->pij", weights, centered, d_centered)
    )


def _min_eigen_gap(eigenvalues: tf.Tensor) -> tf.Tensor:
    gaps = eigenvalues[1:] - eigenvalues[:-1]
    return tf.cond(
        tf.size(gaps) > 0,
        lambda: tf.reduce_min(tf.abs(gaps)),
        lambda: tf.constant(float("inf"), dtype=tf.float64),
    )


def _validate_shape(tensor: tf.Tensor, expected: tuple[int, ...], name: str) -> None:
    actual = tensor.shape.as_list()
    if any(dim is None for dim in actual):
        return
    if tuple(actual) != expected:
        raise ValueError(f"{name} has shape {tuple(actual)}, expected {expected}")


def _validate_model_derivative_shapes(
    model: TFStructuralStateSpace,
    derivatives: TFStructuralFirstDerivatives,
) -> tuple[int, int, int, int]:
    p = derivatives.parameter_dim
    n = model.partition.state_dim
    q = model.partition.innovation_dim
    m = model.observation_dim
    if p is None or n is None or q is None or m is None:
        raise ValueError("analytic SVD sigma-point scores require static dimensions")
    if derivatives.state_dim != n:
        raise ValueError("derivative state dimension does not match model")
    if derivatives.innovation_dim != q:
        raise ValueError("derivative innovation dimension does not match model")
    if derivatives.observation_dim != m:
        raise ValueError("derivative observation dimension does not match model")
    return int(p), int(n), int(q), int(m)


def _checked_smooth_eigh_factor_first_derivatives(
    covariance: tf.Tensor,
    d_covariance: tf.Tensor,
    *,
    singular_floor: tf.Tensor,
    spectral_gap_tolerance: tf.Tensor,
    label: str,
) -> TFSmoothEighFactorFirstDerivatives:
    covariance = symmetrize(tf.convert_to_tensor(covariance, dtype=tf.float64))
    d_covariance = symmetrize(tf.convert_to_tensor(d_covariance, dtype=tf.float64))
    (
        eigenvalues,
        floored_eigenvalues,
        eigenvectors,
        implemented_covariance,
        psd_projection_residual,
    ) = psd_eigh(covariance, singular_floor)
    min_gap = _min_eigen_gap(eigenvalues)
    active_floors = floor_count(eigenvalues, singular_floor)
    assertions = [
        tf.debugging.assert_equal(
            active_floors,
            tf.constant(0, dtype=tf.int32),
            message=f"blocked_active_floor: {label} floor is active",
        ),
        tf.debugging.assert_greater(
            min_gap,
            spectral_gap_tolerance,
            message=f"blocked_weak_spectral_gap: {label} spectrum is not separated",
        ),
        tf.debugging.assert_all_finite(
            eigenvalues,
            f"blocked_nonfinite_factor: {label} eigenvalues are nonfinite",
        ),
    ]
    with tf.control_dependencies(assertions):
        eigenvalues = tf.identity(eigenvalues)
        floored_eigenvalues = tf.identity(floored_eigenvalues)
        eigenvectors = tf.identity(eigenvectors)
        implemented_covariance = tf.identity(implemented_covariance)

    sqrt_eigenvalues = tf.sqrt(floored_eigenvalues)
    factor = eigenvectors @ tf.linalg.diag(sqrt_eigenvalues)
    projected = (
        tf.linalg.matrix_transpose(eigenvectors)[tf.newaxis, :, :]
        @ d_covariance
        @ eigenvectors[tf.newaxis, :, :]
    )
    dim = _static_dim(eigenvalues, 0, f"{label} eigen dimension")
    eye = tf.eye(dim, dtype=tf.float64)
    denominator = eigenvalues[tf.newaxis, :] - eigenvalues[:, tf.newaxis]
    safe_denominator = tf.where(tf.equal(eye, 1.0), tf.ones_like(denominator), denominator)
    coefficients = projected / safe_denominator[tf.newaxis, :, :] * (
        1.0 - eye[tf.newaxis, :, :]
    )
    d_eigenvectors = tf.einsum("db,pba->pda", eigenvectors, coefficients)
    d_eigenvalues = tf.linalg.diag_part(projected)
    d_sqrt_eigenvalues = 0.5 * d_eigenvalues / sqrt_eigenvalues[tf.newaxis, :]
    d_factor = (
        d_eigenvectors * sqrt_eigenvalues[tf.newaxis, tf.newaxis, :]
        + eigenvectors[tf.newaxis, :, :] * d_sqrt_eigenvalues[:, tf.newaxis, :]
    )
    reconstructed = tf.matmul(d_factor, factor[tf.newaxis, :, :], transpose_b=True) + tf.matmul(
        factor[tf.newaxis, :, :],
        d_factor,
        transpose_b=True,
    )
    reconstruction_residual = tf.reduce_max(tf.linalg.norm(reconstructed - d_covariance, axis=[-2, -1]))
    return TFSmoothEighFactorFirstDerivatives(
        eigenvalues=eigenvalues,
        floored_eigenvalues=floored_eigenvalues,
        eigenvectors=eigenvectors,
        factor=factor,
        d_factor=d_factor,
        implemented_covariance=implemented_covariance,
        floor_count=active_floors,
        min_eigen_gap=min_gap,
        psd_projection_residual=psd_projection_residual,
        derivative_reconstruction_residual=reconstruction_residual,
    )


def _matvec_points(jacobian: tf.Tensor, vectors: tf.Tensor) -> tf.Tensor:
    return tf.einsum("roi,pri->pro", jacobian, vectors)


def _smooth_sigma_point_score_with_rule(
    observations: tf.Tensor,
    model: TFStructuralStateSpace,
    derivatives: TFStructuralFirstDerivatives,
    *,
    sigma_rule: TFSigmaPointRule,
    backend_name: str,
    placement_floor: tf.Tensor | float,
    innovation_floor: tf.Tensor | float,
    rank_tolerance: tf.Tensor | float,
    spectral_gap_tolerance: tf.Tensor | float,
    jitter: tf.Tensor | float,
) -> TFFilterDerivativeResult:
    y = _as_observation_matrix(observations)
    n_timesteps = _static_num_timesteps(y)
    p, state_dim, innovation_dim, observation_dim = _validate_model_derivative_shapes(
        model,
        derivatives,
    )
    aug_dim = state_dim + innovation_dim
    if sigma_rule.dim != aug_dim:
        raise ValueError("sigma_rule dimension must equal state_dim + innovation_dim")

    mean = tf.convert_to_tensor(model.initial_mean, dtype=tf.float64)
    covariance = symmetrize(model.initial_covariance)
    d_mean = tf.convert_to_tensor(derivatives.d_initial_mean, dtype=tf.float64)
    d_covariance = symmetrize(derivatives.d_initial_covariance)
    innovation_covariance = symmetrize(model.innovation_covariance)
    d_innovation_covariance = symmetrize(derivatives.d_innovation_covariance)
    observation_covariance = symmetrize(model.observation_covariance)
    d_observation_covariance = symmetrize(derivatives.d_observation_covariance)
    placement_floor = tf.convert_to_tensor(placement_floor, dtype=tf.float64)
    innovation_floor = tf.convert_to_tensor(innovation_floor, dtype=tf.float64)
    rank_tolerance = tf.convert_to_tensor(rank_tolerance, dtype=tf.float64)
    spectral_gap_tolerance = tf.convert_to_tensor(
        spectral_gap_tolerance,
        dtype=tf.float64,
    )
    jitter = tf.convert_to_tensor(jitter, dtype=tf.float64)

    obs_identity = tf.eye(observation_dim, dtype=tf.float64)
    two_pi = tf.constant(2.0 * math.pi, dtype=tf.float64)
    log_likelihood = tf.constant(0.0, dtype=tf.float64)
    score = tf.zeros([p], dtype=tf.float64)
    max_placement_floor_count = tf.constant(0, dtype=tf.int32)
    max_innovation_floor_count = tf.constant(0, dtype=tf.int32)
    max_placement_residual = tf.constant(0.0, dtype=tf.float64)
    max_innovation_residual = tf.constant(0.0, dtype=tf.float64)
    max_support_residual = tf.constant(0.0, dtype=tf.float64)
    max_deterministic_residual = tf.constant(0.0, dtype=tf.float64)
    max_factor_derivative_residual = tf.constant(0.0, dtype=tf.float64)
    max_integration_rank = tf.constant(0, dtype=tf.int32)
    min_placement_eigen_gap = tf.constant(float("inf"), dtype=tf.float64)
    min_innovation_eigen_gap = tf.constant(float("inf"), dtype=tf.float64)
    last_implemented_innovation_covariance = tf.zeros(
        [observation_dim, observation_dim],
        dtype=tf.float64,
    )

    for t in range(n_timesteps):
        aug_mean = tf.concat(
            [mean, tf.zeros([innovation_dim], dtype=tf.float64)],
            axis=0,
        )
        d_aug_mean = tf.concat(
            [d_mean, tf.zeros([p, innovation_dim], dtype=tf.float64)],
            axis=1,
        )
        upper = tf.concat(
            [covariance, tf.zeros([state_dim, innovation_dim], dtype=tf.float64)],
            axis=1,
        )
        lower = tf.concat(
            [
                tf.zeros([innovation_dim, state_dim], dtype=tf.float64),
                innovation_covariance,
            ],
            axis=1,
        )
        aug_covariance = tf.concat([upper, lower], axis=0)
        d_aug_covariance = tf.concat(
            [
                tf.concat(
                    [
                        d_covariance,
                        tf.zeros([p, state_dim, innovation_dim], dtype=tf.float64),
                    ],
                    axis=2,
                ),
                tf.concat(
                    [
                        tf.zeros([p, innovation_dim, state_dim], dtype=tf.float64),
                        d_innovation_covariance,
                    ],
                    axis=2,
                ),
            ],
            axis=1,
        )
        placement = _checked_smooth_eigh_factor_first_derivatives(
            aug_covariance,
            d_aug_covariance,
            singular_floor=placement_floor,
            spectral_gap_tolerance=spectral_gap_tolerance,
            label="SVD sigma-point placement",
        )
        point_offsets = sigma_rule.offsets @ tf.transpose(placement.factor)
        aug_points = aug_mean[tf.newaxis, :] + point_offsets
        d_point_offsets = tf.einsum("rd,pad->pra", sigma_rule.offsets, placement.d_factor)
        d_aug_points = d_aug_mean[:, tf.newaxis, :] + d_point_offsets
        previous_points = aug_points[:, :state_dim]
        innovation_points = aug_points[:, state_dim:]
        d_previous_points = d_aug_points[:, :, :state_dim]
        d_innovation_points = d_aug_points[:, :, state_dim:]

        predicted_points = model.transition(previous_points, innovation_points)
        transition_state_jacobian = derivatives.transition_state_jacobian_fn(
            previous_points,
            innovation_points,
        )
        transition_innovation_jacobian = derivatives.transition_innovation_jacobian_fn(
            previous_points,
            innovation_points,
        )
        d_transition_param = derivatives.d_transition_fn(
            previous_points,
            innovation_points,
        )
        _validate_shape(
            transition_state_jacobian,
            (sigma_rule.point_count, state_dim, state_dim),
            "transition_state_jacobian",
        )
        _validate_shape(
            transition_innovation_jacobian,
            (sigma_rule.point_count, state_dim, innovation_dim),
            "transition_innovation_jacobian",
        )
        _validate_shape(
            d_transition_param,
            (p, sigma_rule.point_count, state_dim),
            "d_transition",
        )
        d_predicted_points = (
            _matvec_points(transition_state_jacobian, d_previous_points)
            + _matvec_points(transition_innovation_jacobian, d_innovation_points)
            + d_transition_param
        )
        residuals = model.deterministic_residual(
            previous_points,
            innovation_points,
            predicted_points,
        )
        deterministic_residual = (
            tf.constant(0.0, dtype=tf.float64)
            if residuals.shape[-1] == 0
            else tf.reduce_max(tf.abs(residuals))
        )

        predicted_mean = tf.linalg.matvec(
            tf.transpose(predicted_points),
            sigma_rule.mean_weights,
        )
        d_predicted_mean = tf.einsum(
            "r,prn->pn",
            sigma_rule.mean_weights,
            d_predicted_points,
        )
        centered_x = predicted_points - predicted_mean[tf.newaxis, :]
        d_centered_x = d_predicted_points - d_predicted_mean[:, tf.newaxis, :]
        predicted_covariance = _weighted_covariance(
            centered_x,
            sigma_rule.covariance_weights,
        )
        d_predicted_covariance = _weighted_covariance_first_derivatives(
            centered_x,
            d_centered_x,
            sigma_rule.covariance_weights,
        )

        observation_points = model.observe(predicted_points)
        observation_state_jacobian = derivatives.observation_state_jacobian_fn(
            predicted_points,
        )
        d_observation_param = derivatives.d_observation_fn(predicted_points)
        _validate_shape(
            observation_state_jacobian,
            (sigma_rule.point_count, observation_dim, state_dim),
            "observation_state_jacobian",
        )
        _validate_shape(
            d_observation_param,
            (p, sigma_rule.point_count, observation_dim),
            "d_observation",
        )
        d_observation_points = (
            _matvec_points(observation_state_jacobian, d_predicted_points)
            + d_observation_param
        )
        observation_mean = tf.linalg.matvec(
            tf.transpose(observation_points),
            sigma_rule.mean_weights,
        )
        d_observation_mean = tf.einsum(
            "r,prm->pm",
            sigma_rule.mean_weights,
            d_observation_points,
        )
        centered_y = observation_points - observation_mean[tf.newaxis, :]
        d_centered_y = d_observation_points - d_observation_mean[:, tf.newaxis, :]
        raw_innovation_covariance = symmetrize(
            _weighted_covariance(centered_y, sigma_rule.covariance_weights)
            + observation_covariance
            + jitter * obs_identity
        )
        d_raw_innovation_covariance = symmetrize(
            _weighted_covariance_first_derivatives(
                centered_y,
                d_centered_y,
                sigma_rule.covariance_weights,
            )
            + d_observation_covariance
        )
        innovation_factor = _checked_smooth_eigh_factor_first_derivatives(
            raw_innovation_covariance,
            d_raw_innovation_covariance,
            singular_floor=innovation_floor,
            spectral_gap_tolerance=spectral_gap_tolerance,
            label="SVD sigma-point innovation",
        )
        cross_covariance = tf.transpose(centered_x) @ (
            centered_y * sigma_rule.covariance_weights[:, tf.newaxis]
        )
        d_cross_covariance = (
            tf.einsum("r,prn,rm->pnm", sigma_rule.covariance_weights, d_centered_x, centered_y)
            + tf.einsum("r,rn,prm->pnm", sigma_rule.covariance_weights, centered_x, d_centered_y)
        )
        innovation = y[t] - observation_mean
        d_innovation = -d_observation_mean
        solve_innovation = eigh_solve(
            innovation_factor.eigenvectors,
            innovation_factor.floored_eigenvalues,
            innovation,
        )
        innovation_precision = eigh_solve(
            innovation_factor.eigenvectors,
            innovation_factor.floored_eigenvalues,
            obs_identity,
        )
        log_det = eigh_logdet(innovation_factor.floored_eigenvalues)
        mahalanobis = tf.reduce_sum(innovation * solve_innovation)
        contribution = -0.5 * (
            tf.cast(observation_dim, tf.float64) * tf.math.log(two_pi)
            + log_det
            + mahalanobis
        )
        trace_terms = tf.einsum(
            "ab,pba->p",
            innovation_precision,
            d_raw_innovation_covariance,
        )
        innovation_derivative_terms = tf.einsum("pm,m->p", d_innovation, solve_innovation)
        covariance_quadratic_terms = tf.einsum(
            "m,pmn,n->p",
            solve_innovation,
            d_raw_innovation_covariance,
            solve_innovation,
        )
        score = score - 0.5 * (
            trace_terms
            + 2.0 * innovation_derivative_terms
            - covariance_quadratic_terms
        )

        kalman_gain = cross_covariance @ innovation_precision
        d_kalman_gain = []
        kalman_gain_transpose = tf.transpose(kalman_gain)
        for i in range(p):
            rhs = (
                tf.transpose(d_cross_covariance[i])
                - d_raw_innovation_covariance[i] @ kalman_gain_transpose
            )
            solved = eigh_solve(
                innovation_factor.eigenvectors,
                innovation_factor.floored_eigenvalues,
                rhs,
            )
            d_kalman_gain.append(tf.transpose(solved))
        d_kalman_gain = tf.stack(d_kalman_gain, axis=0)

        mean = predicted_mean + tf.linalg.matvec(kalman_gain, innovation)
        d_mean = (
            d_predicted_mean
            + tf.einsum("pnm,m->pn", d_kalman_gain, innovation)
            + tf.einsum("nm,pm->pn", kalman_gain, d_innovation)
        )
        covariance = symmetrize(
            predicted_covariance
            - kalman_gain
            @ innovation_factor.implemented_covariance
            @ tf.transpose(kalman_gain)
        )
        d_covariance = symmetrize(
            d_predicted_covariance
            - tf.einsum(
                "pnm,ml,kl->pnk",
                d_kalman_gain,
                innovation_factor.implemented_covariance,
                kalman_gain,
            )
            - tf.einsum(
                "nm,pml,kl->pnk",
                kalman_gain,
                d_raw_innovation_covariance,
                kalman_gain,
            )
            - tf.einsum(
                "nm,ml,pkl->pnk",
                kalman_gain,
                innovation_factor.implemented_covariance,
                d_kalman_gain,
            )
        )
        log_likelihood = log_likelihood + contribution

        active = placement.eigenvalues > rank_tolerance
        rank = tf.reduce_sum(tf.cast(active, tf.int32))
        null_weights = tf.cast(tf.logical_not(active), tf.float64)
        null_projector = (
            placement.eigenvectors
            @ tf.linalg.diag(null_weights)
            @ tf.linalg.matrix_transpose(placement.eigenvectors)
        )
        support_residual = tf.reduce_max(tf.linalg.norm(point_offsets @ null_projector, axis=1))
        max_placement_floor_count = tf.maximum(
            max_placement_floor_count,
            placement.floor_count,
        )
        max_innovation_floor_count = tf.maximum(
            max_innovation_floor_count,
            innovation_factor.floor_count,
        )
        max_placement_residual = tf.maximum(
            max_placement_residual,
            placement.psd_projection_residual,
        )
        max_innovation_residual = tf.maximum(
            max_innovation_residual,
            innovation_factor.psd_projection_residual,
        )
        max_support_residual = tf.maximum(max_support_residual, support_residual)
        max_deterministic_residual = tf.maximum(
            max_deterministic_residual,
            deterministic_residual,
        )
        max_factor_derivative_residual = tf.maximum(
            max_factor_derivative_residual,
            tf.maximum(
                placement.derivative_reconstruction_residual,
                innovation_factor.derivative_reconstruction_residual,
            ),
        )
        max_integration_rank = tf.maximum(max_integration_rank, rank)
        min_placement_eigen_gap = tf.minimum(
            min_placement_eigen_gap,
            placement.min_eigen_gap,
        )
        min_innovation_eigen_gap = tf.minimum(
            min_innovation_eigen_gap,
            innovation_factor.min_eigen_gap,
        )
        last_implemented_innovation_covariance = (
            innovation_factor.implemented_covariance
        )

    checked_value = tf.debugging.check_numerics(
        log_likelihood,
        "blocked_nonfinite_value: SVD sigma-point score value is nonfinite",
    )
    checked_score = tf.debugging.check_numerics(
        score,
        "blocked_nonfinite_score: SVD sigma-point score is nonfinite",
    )
    block_metadata = dict(structural_block_metadata(model))
    extra = {
        **block_metadata,
        "rule": sigma_rule.name,
        "augmented_dim": tf.constant(aug_dim, dtype=tf.int32),
        "point_count": tf.constant(sigma_rule.point_count, dtype=tf.int32),
        "polynomial_degree": tf.constant(sigma_rule.polynomial_degree, dtype=tf.int32),
        "max_integration_rank": max_integration_rank,
        "support_residual": max_support_residual,
        "deterministic_residual": max_deterministic_residual,
        "min_placement_eigen_gap": min_placement_eigen_gap,
        "min_innovation_eigen_gap": min_innovation_eigen_gap,
        "factor_derivative_reconstruction_residual": max_factor_derivative_residual,
        "placement_psd_projection_residual": max_placement_residual,
        "innovation_psd_projection_residual": max_innovation_residual,
        "placement_floor_count": max_placement_floor_count,
        "innovation_floor_count": max_innovation_floor_count,
        "factorization": "tf.linalg.eigh",
        "derivative_branch": "smooth_simple_spectrum_no_active_floor",
        "derivative_method": "analytic_first_order_smooth_branch",
        "derivative_provider": derivatives.name,
        "hessian_status": "deferred",
    }
    diagnostics = TFFilterDiagnostics(
        backend=backend_name,
        mask_convention="none",
        regularization=TFRegularizationDiagnostics(
            jitter=jitter,
            singular_floor=innovation_floor,
            floor_count=max_innovation_floor_count,
            psd_projection_residual=max_innovation_residual,
            implemented_covariance=last_implemented_innovation_covariance,
            branch_label="svd_sigma_point_analytic_score_smooth_branch",
            derivative_target="implemented_regularized_law",
        ),
        extra=extra,
    )
    return TFFilterDerivativeResult(
        log_likelihood=checked_value,
        score=checked_score,
        hessian=None,
        metadata=structural_filter_metadata(
            model,
            filter_name=backend_name,
            differentiability_status="analytic_score_smooth_branch_hessian_deferred",
            compiled_status="eager_tf",
        ),
        diagnostics=diagnostics,
    )


def tf_svd_sigma_point_score_with_rule(
    observations: tf.Tensor,
    model: TFStructuralStateSpace,
    derivatives: TFStructuralFirstDerivatives,
    *,
    sigma_rule: TFSigmaPointRule,
    backend_name: str = "tf_svd_sigma_point_score",
    placement_floor: tf.Tensor | float = 0.0,
    innovation_floor: tf.Tensor | float = 1e-12,
    rank_tolerance: tf.Tensor | float = 1e-12,
    spectral_gap_tolerance: tf.Tensor | float = 1e-8,
    jitter: tf.Tensor | float = 0.0,
) -> TFFilterDerivativeResult:
    """Return an analytic score for a fixed SVD/eigen sigma-point rule."""

    return _smooth_sigma_point_score_with_rule(
        observations,
        model,
        derivatives,
        sigma_rule=sigma_rule,
        backend_name=backend_name,
        placement_floor=placement_floor,
        innovation_floor=innovation_floor,
        rank_tolerance=rank_tolerance,
        spectral_gap_tolerance=spectral_gap_tolerance,
        jitter=jitter,
    )


def tf_svd_cubature_score(
    observations: tf.Tensor,
    model: TFStructuralStateSpace,
    derivatives: TFStructuralFirstDerivatives,
    *,
    placement_floor: tf.Tensor | float = 0.0,
    innovation_floor: tf.Tensor | float = 1e-12,
    rank_tolerance: tf.Tensor | float = 1e-12,
    spectral_gap_tolerance: tf.Tensor | float = 1e-8,
    jitter: tf.Tensor | float = 0.0,
) -> TFFilterDerivativeResult:
    """Return the analytic smooth-branch SVD cubature score."""

    rule = tf_unit_sigma_point_rule(
        model.partition.state_dim + model.partition.innovation_dim,
        rule="cubature",
    )
    return tf_svd_sigma_point_score_with_rule(
        observations,
        model,
        derivatives,
        sigma_rule=rule,
        backend_name="tf_svd_cubature_score",
        placement_floor=placement_floor,
        innovation_floor=innovation_floor,
        rank_tolerance=rank_tolerance,
        spectral_gap_tolerance=spectral_gap_tolerance,
        jitter=jitter,
    )


def tf_svd_ukf_score(
    observations: tf.Tensor,
    model: TFStructuralStateSpace,
    derivatives: TFStructuralFirstDerivatives,
    *,
    alpha: float = 1.0,
    beta: float = 2.0,
    kappa: float = 0.0,
    placement_floor: tf.Tensor | float = 0.0,
    innovation_floor: tf.Tensor | float = 1e-12,
    rank_tolerance: tf.Tensor | float = 1e-12,
    spectral_gap_tolerance: tf.Tensor | float = 1e-8,
    jitter: tf.Tensor | float = 0.0,
) -> TFFilterDerivativeResult:
    """Return the analytic smooth-branch SVD-UKF score."""

    rule = tf_unit_sigma_point_rule(
        model.partition.state_dim + model.partition.innovation_dim,
        rule="unscented",
        alpha=alpha,
        beta=beta,
        kappa=kappa,
    )
    return tf_svd_sigma_point_score_with_rule(
        observations,
        model,
        derivatives,
        sigma_rule=rule,
        backend_name="tf_svd_ukf_score",
        placement_floor=placement_floor,
        innovation_floor=innovation_floor,
        rank_tolerance=rank_tolerance,
        spectral_gap_tolerance=spectral_gap_tolerance,
        jitter=jitter,
    )


def tf_svd_cut4_score(
    observations: tf.Tensor,
    model: TFStructuralStateSpace,
    derivatives: TFStructuralFirstDerivatives,
    *,
    placement_floor: tf.Tensor | float = 0.0,
    innovation_floor: tf.Tensor | float = 1e-12,
    rank_tolerance: tf.Tensor | float = 1e-12,
    spectral_gap_tolerance: tf.Tensor | float = 1e-8,
    jitter: tf.Tensor | float = 0.0,
) -> TFFilterDerivativeResult:
    """Return the analytic smooth-branch SVD-CUT4-G score."""

    rule = tf_cut4g_sigma_point_rule(
        model.partition.state_dim + model.partition.innovation_dim,
    )
    return tf_svd_sigma_point_score_with_rule(
        observations,
        model,
        derivatives,
        sigma_rule=rule,
        backend_name="tf_svd_cut4_score",
        placement_floor=placement_floor,
        innovation_floor=innovation_floor,
        rank_tolerance=rank_tolerance,
        spectral_gap_tolerance=spectral_gap_tolerance,
        jitter=jitter,
    )
