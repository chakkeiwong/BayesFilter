"""Candidate-only generic staged Cubature/GenUT value and total-JVP core."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import tensorflow as tf

from bayesfilter.highdim import ledh_contract_e_reset_tf as reset
from bayesfilter.highdim.higher_moment_contract_e import (
    affine_restore_cloud_jvp,
    higher_moment_shape_jvp,
)


Tensor = tf.Tensor
InitialValue = Callable[[Tensor, Tensor], Tensor]
InitialTangent = Callable[[Tensor, Tensor], Tensor]
TransitionValue = Callable[[Tensor, Tensor, Tensor, Tensor], Tensor]
TransitionTangent = Callable[[Tensor, Tensor, Tensor, Tensor, Tensor], Tensor]
ObservationValue = Callable[[Tensor, Tensor, Tensor, Tensor], Tensor]
ObservationTangent = Callable[[Tensor, Tensor, Tensor, Tensor, Tensor], Tensor]
TransitionResidual = Callable[[Tensor, Tensor, Tensor, Tensor], Tensor]
InitialLogDensity = Callable[[Tensor, Tensor], Tensor]
InitialLogDensityTangent = Callable[[Tensor, Tensor], Tensor]


@dataclass(frozen=True)
class BoundedFeatureShapeTeacher:
    """Time-indexed explicit targets in one bounded nonlinear state chart."""

    frame_mu: Tensor
    frame_matrix: Tensor
    skew: Tensor
    kurtosis: Tensor
    skew_tangent: Tensor
    kurtosis_tangent: Tensor
    pairwise_co_skew: Tensor
    pairwise_co_kurtosis: Tensor
    pairwise_co_skew_tangent: Tensor
    pairwise_co_kurtosis_tangent: Tensor
    pairwise_co_skew_mask: Tensor
    pairwise_co_kurtosis_mask: Tensor

    def __post_init__(self) -> None:
        mu = tf.convert_to_tensor(self.frame_mu)
        matrix = tf.convert_to_tensor(self.frame_matrix, mu.dtype)
        if mu.shape.rank != 2 or matrix.shape.rank != 3:
            raise ValueError("bounded teacher frames must be time-indexed")
        if matrix.shape[0] != mu.shape[0] or matrix.shape[1] != matrix.shape[2]:
            raise ValueError("bounded teacher frames have incompatible shapes")
        if matrix.shape[1] != mu.shape[1]:
            raise ValueError("bounded teacher frame dimension mismatch")
        time_count = int(mu.shape[0])
        state_dimension = int(mu.shape[1])
        for name, rank, tail in (
            ("skew", 2, (state_dimension,)),
            ("kurtosis", 2, (state_dimension,)),
            ("skew_tangent", 3, (state_dimension, None)),
            ("kurtosis_tangent", 3, (state_dimension, None)),
            ("pairwise_co_skew", 3, (state_dimension, state_dimension)),
            ("pairwise_co_kurtosis", 3, (state_dimension, state_dimension)),
            ("pairwise_co_skew_tangent", 4, (state_dimension, state_dimension, None)),
            ("pairwise_co_kurtosis_tangent", 4, (state_dimension, state_dimension, None)),
            ("pairwise_co_skew_mask", 3, (state_dimension, state_dimension)),
            ("pairwise_co_kurtosis_mask", 3, (state_dimension, state_dimension)),
        ):
            value = tf.convert_to_tensor(getattr(self, name), mu.dtype)
            if value.shape.rank != rank or int(value.shape[0]) != time_count:
                raise ValueError(f"{name} must share the teacher time axis")
            for observed, expected in zip(value.shape[1:], tail):
                if expected is not None and observed != expected:
                    raise ValueError(f"{name} has the wrong state shape")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "frame_mu", mu)
        object.__setattr__(self, "frame_matrix", matrix)


@dataclass(frozen=True)
class CandidateModelAdapter:
    """Value/tangent contract for one finite nonlinear state-space model."""

    state_dimension: int
    parameter_count: int
    initial_value: InitialValue
    initial_tangent: InitialTangent
    transition_value: TransitionValue
    transition_tangent: TransitionTangent
    observation_value: ObservationValue
    observation_tangent: ObservationTangent
    transition_residual: TransitionResidual | None = None
    initial_log_density: InitialLogDensity | None = None
    initial_log_density_tangent: InitialLogDensityTangent | None = None

    def __post_init__(self) -> None:
        if self.state_dimension < 1 or self.parameter_count < 1:
            raise ValueError("adapter dimensions must be positive")
        if (self.initial_log_density is None) != (
            self.initial_log_density_tangent is None
        ):
            raise ValueError(
                "initial log-density value and tangent must be supplied together"
            )


def _sinkhorn_barycentric_jvp(
    particles: Tensor,
    weights: Tensor,
    particle_tangent: Tensor,
    weight_tangent: Tensor,
    *,
    epsilon: float,
    sinkhorn_steps: int,
    balance_steps: int = 8,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    result = _sinkhorn_barycentric_jvp_core(
        particles,
        weights,
        particle_tangent,
        weight_tangent,
        epsilon=epsilon,
        sinkhorn_steps=sinkhorn_steps,
        balance_steps=balance_steps,
    )
    return (
        result["particles"],
        result["particles_tangent"],
        result["maximum_raw_row_residual"],
        result["maximum_post_quotient_column_residual"],
    )


def _sinkhorn_barycentric_jvp_core(
    particles: Tensor,
    weights: Tensor,
    particle_tangent: Tensor,
    weight_tangent: Tensor,
    *,
    epsilon: float,
    sinkhorn_steps: int,
    balance_steps: int,
) -> dict[str, Tensor]:
    """Return the finite Sinkhorn row quotient and its total JVP."""

    if epsilon <= 0.0 or sinkhorn_steps <= 0 or balance_steps < 0:
        raise ValueError("epsilon and Sinkhorn counts must be valid")
    deltas = particles[:, None, :] - particles[None, :, :]
    delta_tangent = particle_tangent[:, None, :, :] - particle_tangent[None, :, :, :]
    cost = tf.reduce_sum(tf.square(deltas), axis=-1)
    cost_tangent = 2.0 * tf.reduce_sum(
        deltas[:, :, :, None] * delta_tangent, axis=2
    )
    mean_cost = tf.reduce_mean(cost)
    floor = tf.constant(1.0e-3, particles.dtype)
    cost_scale = tf.maximum(mean_cost, floor)
    cost_scale_tangent = tf.where(
        mean_cost > floor, tf.reduce_mean(cost_tangent, axis=[0, 1]),
        tf.zeros([tf.shape(particle_tangent)[-1]], particles.dtype),
    )
    exponent = -cost / (cost_scale * tf.cast(epsilon, particles.dtype))
    exponent_tangent = -(
        cost_tangent / cost_scale
        - cost[:, :, None] * cost_scale_tangent[None, None, :]
        / tf.square(cost_scale)
    ) / tf.cast(epsilon, particles.dtype)
    kernel = tf.exp(exponent)
    kernel_tangent = kernel[:, :, None] * exponent_tangent
    n = tf.shape(particles)[0]
    uniform = tf.fill([n], tf.cast(1.0, particles.dtype) / tf.cast(n, particles.dtype))
    left = tf.ones_like(uniform)
    right = tf.ones_like(uniform)
    left_tangent = tf.zeros([n, tf.shape(particle_tangent)[-1]], particles.dtype)
    right_tangent = tf.zeros_like(left_tangent)
    tiny = tf.cast(1.0e-7, particles.dtype)
    def sinkhorn_body(
        index: Tensor,
        left_value: Tensor,
        right_value: Tensor,
        left_tangent_value: Tensor,
        right_tangent_value: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        left_denominator = tf.linalg.matvec(kernel, right_value) + tiny
        left_denominator_tangent = (
            tf.einsum("ijp,j->ip", kernel_tangent, right_value)
            + tf.einsum("ij,jp->ip", kernel, right_tangent_value)
        )
        left_new = uniform / left_denominator
        left_tangent_new = -uniform[:, None] * left_denominator_tangent / tf.square(
            left_denominator
        )[:, None]
        right_denominator = tf.linalg.matvec(tf.transpose(kernel), left_new) + tiny
        right_denominator_tangent = (
            tf.einsum("ijp,i->jp", kernel_tangent, left_new)
            + tf.einsum("ij,ip->jp", kernel, left_tangent_new)
        )
        right_new = weights / right_denominator
        right_tangent_new = (
            weight_tangent / right_denominator[:, None]
            - weights[:, None] * right_denominator_tangent
            / tf.square(right_denominator)[:, None]
        )
        return (
            index + 1,
            left_new,
            right_new,
            left_tangent_new,
            right_tangent_new,
        )

    _, left, right, left_tangent, right_tangent = tf.while_loop(
        lambda index, *_: index < tf.cast(sinkhorn_steps, tf.int32),
        sinkhorn_body,
        (
            tf.zeros([], tf.int32),
            left,
            right,
            left_tangent,
            right_tangent,
        ),
        parallel_iterations=1,
    )
    # Fixed terminal-epsilon IPFP refinement is a separate tuned control.  It
    # uses the same finite updates and therefore the same explicit JVP.
    _, left, right, left_tangent, right_tangent = tf.while_loop(
        lambda index, *_: index < tf.cast(balance_steps, tf.int32),
        sinkhorn_body,
        (
            tf.zeros([], tf.int32),
            left,
            right,
            left_tangent,
            right_tangent,
        ),
        parallel_iterations=1,
    )
    coupling = left[:, None] * kernel * right[None, :]
    coupling_tangent = (
        left_tangent[:, None, :] * kernel[:, :, None] * right[None, :, None]
        + left[:, None, None] * kernel_tangent * right[None, :, None]
        + left[:, None, None] * kernel[:, :, None] * right_tangent[None, :, :]
    )
    row_mass = tf.reduce_sum(coupling, axis=1)
    row_mass_tangent = tf.reduce_sum(coupling_tangent, axis=1)
    numerator = coupling @ particles
    numerator_tangent = (
        tf.einsum("ijp,jd->idp", coupling_tangent, particles)
        + tf.einsum("ij,jdp->idp", coupling, particle_tangent)
    )
    barycentric = numerator / row_mass[:, None]
    barycentric_tangent = (
        numerator_tangent
        - barycentric[:, :, None] * row_mass_tangent[:, None, :]
    ) / row_mass[:, None, None]
    quotient_coupling = uniform[:, None] * coupling / row_mass[:, None]
    quotient_column_mass = tf.reduce_sum(quotient_coupling, axis=0)
    quotient_column_residual = quotient_column_mass - weights
    row_mass_finite = tf.reduce_all(tf.math.is_finite(row_mass))
    row_mass_positive = tf.reduce_all(row_mass > tiny)
    quotient_finite = tf.reduce_all(tf.math.is_finite(barycentric)) & tf.reduce_all(
        tf.math.is_finite(barycentric_tangent)
    )
    post_quotient_column_tv_error = 0.5 * tf.reduce_sum(
        tf.abs(quotient_column_residual)
    )
    marginal_valid = (
        row_mass_finite
        & row_mass_positive
        & quotient_finite
        & (post_quotient_column_tv_error <= tf.cast(1.0e-4, particles.dtype))
    )
    return {
        "particles": barycentric,
        "particles_tangent": barycentric_tangent,
        "coupling": coupling,
        "coupling_tangent": coupling_tangent,
        "row_mass": row_mass,
        "row_mass_tangent": row_mass_tangent,
        "minimum_row_mass": tf.reduce_min(row_mass),
        "quotient_column_mass": quotient_column_mass,
        "maximum_raw_row_residual": tf.reduce_max(tf.abs(row_mass - uniform)),
        "maximum_raw_column_residual": tf.reduce_max(
            tf.abs(tf.reduce_sum(coupling, axis=0) - weights)
        ),
        "maximum_post_quotient_column_residual": tf.reduce_max(
            tf.abs(quotient_column_residual)
        ),
        "post_quotient_column_tv_error": post_quotient_column_tv_error,
        "marginal_valid": marginal_valid,
    }


def _restore_cloud_jvp(
    particles: Tensor,
    weights: Tensor,
    particle_tangent: Tensor,
    weight_tangent: Tensor,
    design: Tensor,
    *,
    epsilon: float,
    sinkhorn_steps: int,
    balance_steps: int = 8,
    ridge: float,
    parameter_count: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    result = _restore_cloud_jvp_core(
        particles,
        weights,
        particle_tangent,
        weight_tangent,
        design,
        epsilon=epsilon,
        sinkhorn_steps=sinkhorn_steps,
        balance_steps=balance_steps,
        ridge=ridge,
        parameter_count=parameter_count,
    )
    return (
        result["particles"],
        result["particles_tangent"],
        result["mean_residual"],
        tf.stack(
            [
                result["maximum_raw_row_residual"],
                result["maximum_post_quotient_column_residual"],
            ]
        ),
    )


def _restore_cloud_jvp_core(
    particles: Tensor,
    weights: Tensor,
    particle_tangent: Tensor,
    weight_tangent: Tensor,
    design: Tensor,
    *,
    epsilon: float,
    sinkhorn_steps: int,
    balance_steps: int,
    ridge: float,
    parameter_count: int,
) -> dict[str, Tensor]:
    transport = _sinkhorn_barycentric_jvp_core(
        particles,
        weights,
        particle_tangent,
        weight_tangent,
        epsilon=epsilon,
        sinkhorn_steps=sinkhorn_steps,
        balance_steps=balance_steps,
    )
    barycentric = transport["particles"]
    barycentric_tangent = transport["particles_tangent"]
    source = particles[None, :, :]
    source_weights = weights[None, :]
    transported = barycentric[None, :, :]
    residual_design = design[None, :, :]
    ridge_tensor = tf.constant([ridge], particles.dtype)
    # Contract E reset kernels are batch-native.  Treat each parameter tangent
    # as a batch member, avoiding a Python parameter loop in the XLA closure.
    parameter_count_tensor = tf.shape(particle_tangent)[-1]
    source_batch = tf.broadcast_to(
        source, tf.stack([parameter_count_tensor, tf.shape(source)[1], tf.shape(source)[2]])
    )
    source_weights_batch = tf.broadcast_to(
        source_weights, tf.stack([parameter_count_tensor, tf.shape(source_weights)[1]])
    )
    transported_batch = tf.broadcast_to(
        transported,
        tf.stack([parameter_count_tensor, tf.shape(transported)[1], tf.shape(transported)[2]]),
    )
    design_batch = tf.broadcast_to(
        residual_design,
        tf.stack([parameter_count_tensor, tf.shape(residual_design)[1], tf.shape(residual_design)[2]]),
    )
    ridge_batch = tf.broadcast_to(ridge_tensor, tf.reshape(parameter_count_tensor, [1]))
    target_mean = tf.reduce_sum(weights[:, None] * particles, axis=0)
    target_mean_tangent = tf.reduce_sum(
        weight_tangent[:, None, :] * particles[:, :, None]
        + weights[:, None, None] * particle_tangent,
        axis=0,
    )
    centered_source = particles - target_mean[None, :]
    target_cov = tf.einsum(
        "n,ni,nj->ij", weights, centered_source, centered_source
    )
    centered_transported = barycentric - tf.reduce_mean(barycentric, axis=0)[None, :]
    transported_cov = tf.einsum(
        "ni,nj->ij", centered_transported, centered_transported
    ) / tf.cast(tf.shape(particles)[0], particles.dtype)
    covariance_gap = 0.5 * (
        target_cov - transported_cov
        + tf.transpose(target_cov - transported_cov)
    )
    minimum_gap_eigenvalue = tf.reduce_min(tf.linalg.eigvalsh(covariance_gap))
    gap_valid = tf.math.is_finite(minimum_gap_eigenvalue) & (
        minimum_gap_eigenvalue + tf.cast(ridge, particles.dtype) > 0.0
    )
    pre_reset_valid = transport["marginal_valid"] & gap_valid
    safe_barycentric = tf.where(
        pre_reset_valid,
        barycentric,
        tf.broadcast_to(target_mean[None, :], tf.shape(barycentric)),
    )
    safe_barycentric_tangent = tf.where(
        pre_reset_valid,
        barycentric_tangent,
        tf.broadcast_to(
            target_mean_tangent[None, :, :], tf.shape(barycentric_tangent)
        ),
    )
    transported = safe_barycentric[None, :, :]
    transported_batch = tf.broadcast_to(
        transported,
        tf.stack([parameter_count_tensor, tf.shape(transported)[1], tf.shape(transported)[2]]),
    )
    forward_batch = reset._contract_e_chol_cloud_forward_core(  # noqa: SLF001
        source_batch,
        source_weights_batch,
        transported_batch,
        design_batch,
        ridge_batch,
    )
    tangent_batch = reset._contract_e_chol_cloud_jvp_from_forward_core(  # noqa: SLF001
        forward_batch,
        source_batch,
        source_weights_batch,
        transported_batch,
        design_batch,
        ridge_batch,
        tf.transpose(particle_tangent, [2, 0, 1]),
        tf.transpose(weight_tangent, [1, 0]),
        tf.transpose(safe_barycentric_tangent, [2, 0, 1]),
        tf.zeros_like(design_batch),
        tf.zeros_like(ridge_batch),
    )["particles"]
    restored_tangent = tf.transpose(tangent_batch, [1, 2, 0])
    forward = reset._contract_e_chol_cloud_forward_core(  # noqa: SLF001
        source, source_weights, transported, residual_design, ridge_tensor
    )
    reset_valid = (
        pre_reset_valid
        & forward["finite"][0]
        & forward["factor_diagonal_positive"][0]
        & tf.reduce_all(forward_batch["finite"])
        & tf.reduce_all(forward_batch["factor_diagonal_positive"])
        & tf.reduce_all(tf.math.is_finite(restored_tangent))
    )
    return {
        "particles": forward["particles"][0],
        "particles_tangent": restored_tangent,
        "mean_residual": tf.reduce_max(tf.abs(forward["mean_residual"])),
        "minimum_gap_eigenvalue": minimum_gap_eigenvalue,
        "gap_valid": gap_valid,
        "reset_valid": reset_valid,
        **{
            name: transport[name]
            for name in (
                "minimum_row_mass",
                "maximum_raw_row_residual",
                "maximum_raw_column_residual",
                "maximum_post_quotient_column_residual",
                "post_quotient_column_tv_error",
                "marginal_valid",
            )
        },
    }


def _projected_cumulant_mode_score(
    source: Tensor,
    weights: Tensor,
    current: Tensor,
    sketch_directions: Tensor,
) -> Tensor:
    """Return a diagnostic mode-one higher-moment residual score."""

    source_mean = tf.reduce_sum(weights[:, None] * source, axis=0)
    source_centered = source - source_mean[None, :]
    source_covariance = tf.einsum(
        "n,ni,nj->ij", weights, source_centered, source_centered
    )
    source_standardized = tf.transpose(
        tf.linalg.triangular_solve(
            tf.linalg.cholesky(source_covariance),
            tf.transpose(source_centered),
            lower=True,
        )
    )
    current_centered = current - tf.reduce_mean(current, axis=0, keepdims=True)
    current_covariance = tf.einsum(
        "ni,nj->ij", current_centered, current_centered
    ) / tf.cast(tf.shape(current)[0], current.dtype)
    current_standardized = tf.transpose(
        tf.linalg.triangular_solve(
            tf.linalg.cholesky(current_covariance),
            tf.transpose(current_centered),
            lower=True,
        )
    )
    source_projection = tf.linalg.matmul(source_standardized, sketch_directions)
    current_projection = tf.linalg.matmul(current_standardized, sketch_directions)
    source_sketch3 = tf.einsum(
        "n,nd,ns->ds",
        weights,
        source_standardized,
        tf.square(source_projection),
    )
    current_sketch3 = tf.reduce_mean(
        current_standardized[:, :, None] * tf.square(current_projection)[:, None, :],
        axis=0,
    )
    source_sketch4 = tf.einsum(
        "n,nd,ns->ds",
        weights,
        source_standardized,
        tf.pow(source_projection, 3.0),
    )
    current_sketch4 = tf.reduce_mean(
        current_standardized[:, :, None] * tf.pow(current_projection, 3.0)[:, None, :],
        axis=0,
    )
    # Conservative Gaussian component scales prevent fourth-order sampling
    # noise from dominating the calibration eigenspace by construction.
    sketch3 = (source_sketch3 - current_sketch3) / tf.sqrt(
        tf.cast(15.0, source.dtype)
    )
    sketch4 = (source_sketch4 - current_sketch4) / tf.sqrt(
        tf.cast(96.0, source.dtype)
    )
    return tf.linalg.matmul(sketch3, sketch3, transpose_b=True) + tf.linalg.matmul(
        sketch4, sketch4, transpose_b=True
    )


def finite_value_score(
    adapter: CandidateModelAdapter,
    theta: Tensor,
    observations: Tensor,
    initial_noise: Tensor,
    process_noise: Tensor,
    design: Tensor,
    *,
    epsilon: float = 2.0,
    sinkhorn_steps: int = 8,
    balance_steps: int = 8,
    ridge: float = 1.0e-5,
    transition_before_first_observation: bool = True,
    higher_moment_correction_steps: int = 0,
    higher_moment_strength: float = 0.0,
    higher_moment_floor: float = 1.0e-6,
    pairwise_moment_correction_steps: int = 0,
    pairwise_moment_strength: float = 0.0,
    pairwise_moment_floor: float = 1.0e-6,
    pairwise_particle_rms_cap: float = 0.0,
    coordinatewise_bounded_cap: float = 0.0,
    coordinatewise_bounded_cap_power: int = 8,
    coordinatewise_standardized_cap: float = 0.0,
    coordinatewise_standardized_cap_power: int = 8,
    projected_cumulant_basis: Tensor | None = None,
    projected_cumulant_correction_steps: int = 0,
    projected_cumulant_strength: float = 0.0,
    projected_cumulant_floor: float = 1.0e-6,
    projected_cumulant_sketch_directions: Tensor | None = None,
    explicit_target_skew: Tensor | None = None,
    explicit_target_kurtosis: Tensor | None = None,
    explicit_target_skew_tangent: Tensor | None = None,
    explicit_target_kurtosis_tangent: Tensor | None = None,
    explicit_target_pairwise_co_skew: Tensor | None = None,
    explicit_target_pairwise_co_kurtosis: Tensor | None = None,
    explicit_target_pairwise_co_skew_tangent: Tensor | None = None,
    explicit_target_pairwise_co_kurtosis_tangent: Tensor | None = None,
    pairwise_co_skew_target_mask: Tensor | None = None,
    pairwise_co_kurtosis_target_mask: Tensor | None = None,
    bounded_feature_teacher: BoundedFeatureShapeTeacher | None = None,
) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
    """Evaluate the finite candidate scalar and its complete forward JVP."""

    theta = tf.convert_to_tensor(theta, dtype=initial_noise.dtype)
    observations = tf.convert_to_tensor(observations, dtype=initial_noise.dtype)
    initial_noise = tf.convert_to_tensor(initial_noise, dtype=theta.dtype)
    process_noise = tf.convert_to_tensor(process_noise, dtype=theta.dtype)
    design = tf.convert_to_tensor(design, dtype=theta.dtype)
    parameter_count = adapter.parameter_count
    particles = adapter.initial_value(theta, initial_noise)
    particle_tangent = adapter.initial_tangent(theta, initial_noise)
    n_static = particles.shape[0]
    if n_static is None:
        raise ValueError("candidate XLA core requires a static particle dimension")
    n = tf.constant(n_static, tf.int32)
    weights = tf.fill([n_static], tf.cast(1.0, theta.dtype) / tf.cast(n, theta.dtype))
    weight_tangent = tf.zeros([n, parameter_count], theta.dtype)
    total = tf.zeros([], theta.dtype)
    score = tf.zeros([parameter_count], theta.dtype)
    max_mean = tf.zeros([], theta.dtype)
    max_marginal = tf.zeros([2], theta.dtype)
    max_transition_residual = tf.zeros([], theta.dtype)
    program_valid = tf.constant(True)
    minimum_row_mass = tf.constant(float("inf"), theta.dtype)
    maximum_column_tv_error = tf.zeros([], theta.dtype)
    minimum_gap_eigenvalue = tf.constant(float("inf"), theta.dtype)
    maximum_skew_residual = tf.zeros([], theta.dtype)
    maximum_kurtosis_residual = tf.zeros([], theta.dtype)
    maximum_pairwise_co_skew_residual = tf.zeros([], theta.dtype)
    maximum_pairwise_co_kurtosis_residual = tf.zeros([], theta.dtype)
    maximum_pairwise_pre_cap_particle_rms = tf.zeros([], theta.dtype)
    maximum_pairwise_post_cap_particle_rms = tf.zeros([], theta.dtype)
    minimum_pairwise_particle_cap_scale = tf.ones([], theta.dtype)
    maximum_physical_affine_mean_residual = tf.zeros([], theta.dtype)
    maximum_physical_affine_covariance_residual = tf.zeros([], theta.dtype)
    maximum_normalized_physical_affine_mean_residual = tf.zeros([], theta.dtype)
    maximum_normalized_physical_affine_covariance_residual = tf.zeros([], theta.dtype)
    maximum_absolute_bounded_coordinate = tf.zeros([], theta.dtype)
    maximum_coordinatewise_pre_cap_absolute = tf.zeros([], theta.dtype)
    maximum_coordinatewise_post_cap_absolute = tf.zeros([], theta.dtype)
    coordinatewise_cap_displacement_sum = tf.zeros([], theta.dtype)
    maximum_coordinatewise_cap_active_fraction = tf.zeros([], theta.dtype)
    minimum_coordinatewise_cap_derivative = tf.ones([], theta.dtype)
    maximum_coordinatewise_inverse_derivative = tf.zeros([], theta.dtype)
    maximum_shape_displacement = tf.zeros([], theta.dtype)
    maximum_normalized_shape_displacement = tf.zeros([], theta.dtype)
    shape_objective_sum = tf.zeros([], theta.dtype)
    pairwise_shape_objective_sum = tf.zeros([], theta.dtype)
    horizon = tf.shape(observations)[0]
    increments = tf.TensorArray(theta.dtype, size=horizon, element_shape=())
    score_increments = tf.TensorArray(
        theta.dtype, size=horizon, element_shape=(parameter_count,)
    )
    state_dimension = adapter.state_dimension
    if projected_cumulant_basis is None:
        projected_basis = None
    else:
        projected_basis = tf.convert_to_tensor(projected_cumulant_basis, dtype=theta.dtype)
        if projected_basis.shape.rank not in (2, 3):
            raise ValueError("projected_cumulant_basis must have rank 2 or 3")
        if projected_basis.shape.rank == 2:
            projected_basis = tf.ensure_shape(projected_basis, [state_dimension, None])
        else:
            projected_basis = tf.ensure_shape(projected_basis, [None, state_dimension, None])
    if projected_cumulant_sketch_directions is None:
        sketch_directions = tf.zeros([state_dimension, 1], theta.dtype)
        collect_mode_score = False
    else:
        sketch_directions = tf.ensure_shape(
            tf.convert_to_tensor(projected_cumulant_sketch_directions, dtype=theta.dtype),
            [state_dimension, None],
        )
        collect_mode_score = True
    mode_scores = tf.TensorArray(
        theta.dtype,
        size=horizon,
        element_shape=(state_dimension, state_dimension),
    )

    def time_body(
        time_index: Tensor,
        particles_value: Tensor,
        particle_tangent_value: Tensor,
        weights_value: Tensor,
        weight_tangent_value: Tensor,
        total_value: Tensor,
        score_value: Tensor,
        max_mean_value: Tensor,
        max_marginal_value: Tensor,
        max_transition_residual_value: Tensor,
        program_valid_value: Tensor,
        minimum_row_mass_value: Tensor,
        maximum_column_tv_error_value: Tensor,
        minimum_gap_eigenvalue_value: Tensor,
        maximum_skew_residual_value: Tensor,
        maximum_kurtosis_residual_value: Tensor,
        maximum_pairwise_co_skew_residual_value: Tensor,
        maximum_pairwise_co_kurtosis_residual_value: Tensor,
        maximum_pairwise_pre_cap_particle_rms_value: Tensor,
        maximum_pairwise_post_cap_particle_rms_value: Tensor,
        minimum_pairwise_particle_cap_scale_value: Tensor,
        maximum_physical_affine_mean_residual_value: Tensor,
        maximum_physical_affine_covariance_residual_value: Tensor,
        maximum_normalized_physical_affine_mean_residual_value: Tensor,
        maximum_normalized_physical_affine_covariance_residual_value: Tensor,
        maximum_absolute_bounded_coordinate_value: Tensor,
        maximum_coordinatewise_pre_cap_absolute_value: Tensor,
        maximum_coordinatewise_post_cap_absolute_value: Tensor,
        coordinatewise_cap_displacement_sum_value: Tensor,
        maximum_coordinatewise_cap_active_fraction_value: Tensor,
        minimum_coordinatewise_cap_derivative_value: Tensor,
        maximum_coordinatewise_inverse_derivative_value: Tensor,
        maximum_shape_displacement_value: Tensor,
        maximum_normalized_shape_displacement_value: Tensor,
        shape_objective_sum_value: Tensor,
        pairwise_shape_objective_sum_value: Tensor,
        maximum_projected_residual_value: Tensor,
        maximum_projected_third_residual_value: Tensor,
        maximum_projected_fourth_residual_value: Tensor,
        increments_value: tf.TensorArray,
        score_increments_value: tf.TensorArray,
        mode_scores_value: tf.TensorArray,
    ):
        noise = process_noise[time_index]
        time_tensor = time_index
        transition = tf.logical_or(
            tf.constant(transition_before_first_observation),
            tf.not_equal(time_index, tf.constant(0, tf.int32)),
        )
        particles_next = tf.cond(
            transition,
            lambda: adapter.transition_value(
                theta, particles_value, noise, time_tensor
            ),
            lambda: particles_value,
        )
        particle_tangent_next = tf.cond(
            transition,
            lambda: adapter.transition_tangent(
                theta, particles_value, noise, particle_tangent_value, time_tensor
            ),
            lambda: particle_tangent_value,
        )
        if adapter.transition_residual is None:
            transition_residual = tf.zeros([], theta.dtype)
        else:
            transition_residual = tf.cond(
                transition,
                lambda: tf.reduce_max(
                    tf.abs(
                        adapter.transition_residual(
                            theta, particles_value, particles_next, time_tensor
                        )
                    )
                ),
                lambda: tf.zeros([], theta.dtype),
            )
        observation = observations[time_index]
        log_likelihood = adapter.observation_value(
            theta, particles_next, observation, time_tensor
        )
        log_likelihood_tangent = adapter.observation_tangent(
            theta, particles_next, particle_tangent_next, observation, time_tensor
        )
        log_weights = tf.math.log(weights_value) + log_likelihood
        log_weight_tangent = (
            weight_tangent_value / weights_value[:, None] + log_likelihood_tangent
        )
        increment = tf.reduce_logsumexp(log_weights)
        normalized_weights = tf.exp(log_weights - increment)
        increment_tangent = tf.reduce_sum(
            normalized_weights[:, None] * log_weight_tangent, axis=0
        )
        normalized_weight_tangent = normalized_weights[:, None] * (
            log_weight_tangent - increment_tangent[None, :]
        )
        stage_valid = tf.reduce_all(
            tf.stack(
                [
                    tf.reduce_all(tf.math.is_finite(particles_next)),
                    tf.reduce_all(tf.math.is_finite(particle_tangent_next)),
                    tf.math.is_finite(transition_residual),
                    tf.reduce_all(tf.math.is_finite(log_likelihood)),
                    tf.reduce_all(tf.math.is_finite(log_likelihood_tangent)),
                    tf.math.is_finite(increment),
                    tf.reduce_all(tf.math.is_finite(normalized_weights)),
                    tf.reduce_all(tf.math.is_finite(normalized_weight_tangent)),
                    tf.reduce_all(normalized_weights >= 0.0),
                    tf.abs(tf.reduce_sum(normalized_weights) - 1.0)
                    <= tf.cast(1.0e-4, theta.dtype),
                ]
            )
        )
        current_design = design if design.shape.rank == 2 else design[time_index]
        restored = _restore_cloud_jvp_core(
                particles_next,
                normalized_weights,
                particle_tangent_next,
                normalized_weight_tangent,
                current_design,
                epsilon=epsilon,
                sinkhorn_steps=sinkhorn_steps,
                balance_steps=balance_steps,
                ridge=ridge,
                parameter_count=parameter_count,
        )
        teacher_active = bounded_feature_teacher is not None and (
            higher_moment_correction_steps > 0
            or pairwise_moment_correction_steps > 0
            or projected_cumulant_correction_steps > 0
        )
        if teacher_active:
            teacher_mu = tf.cast(
                bounded_feature_teacher.frame_mu[time_index], theta.dtype
            )
            teacher_matrix = tf.cast(
                bounded_feature_teacher.frame_matrix[time_index], theta.dtype
            )
            local_source = tf.transpose(
                tf.linalg.triangular_solve(
                    teacher_matrix,
                    tf.transpose(particles_next) - teacher_mu[:, None],
                    lower=True,
                )
            )
            local_points = tf.transpose(
                tf.linalg.triangular_solve(
                    teacher_matrix,
                    tf.transpose(restored["particles"]) - teacher_mu[:, None],
                    lower=True,
                )
            )
            parameter_dimension = tf.shape(particle_tangent_next)[-1]
            local_source_tangent = tf.transpose(
                tf.reshape(
                    tf.linalg.triangular_solve(
                        teacher_matrix,
                        tf.reshape(
                            tf.transpose(particle_tangent_next, [1, 0, 2]),
                            [state_dimension, -1],
                        ),
                        lower=True,
                    ),
                    [state_dimension, n_static, parameter_dimension],
                ),
                [1, 0, 2],
            )
            local_points_tangent = tf.transpose(
                tf.reshape(
                    tf.linalg.triangular_solve(
                        teacher_matrix,
                        tf.reshape(
                            tf.transpose(
                                restored["particles_tangent"], [1, 0, 2]
                            ),
                            [state_dimension, -1],
                        ),
                        lower=True,
                    ),
                    [state_dimension, n_static, parameter_dimension],
                ),
                [1, 0, 2],
            )
            source_denominator = tf.pow(1.0 + tf.square(local_source), 1.5)
            point_denominator = tf.pow(1.0 + tf.square(local_points), 1.5)
            shape_source = local_source / tf.sqrt(1.0 + tf.square(local_source))
            shape_points = local_points / tf.sqrt(1.0 + tf.square(local_points))
            shape_source_tangent = local_source_tangent / source_denominator[:, :, None]
            shape_points_tangent = local_points_tangent / point_denominator[:, :, None]
            target_skew_for_step = (
                tf.cast(bounded_feature_teacher.skew[time_index], theta.dtype)
                + tf.linalg.matvec(
                    tf.cast(
                        bounded_feature_teacher.skew_tangent[time_index],
                        theta.dtype,
                    ),
                    theta,
                )
            )
            target_kurtosis_for_step = (
                tf.cast(bounded_feature_teacher.kurtosis[time_index], theta.dtype)
                + tf.linalg.matvec(
                    tf.cast(
                        bounded_feature_teacher.kurtosis_tangent[time_index],
                        theta.dtype,
                    ),
                    theta,
                )
            )
            target_co_skew_for_step = (
                tf.cast(
                    bounded_feature_teacher.pairwise_co_skew[time_index],
                    theta.dtype,
                )
                + tf.einsum(
                    "ijp,p->ij",
                    tf.cast(
                        bounded_feature_teacher.pairwise_co_skew_tangent[
                            time_index
                        ],
                        theta.dtype,
                    ),
                    theta,
                )
            )
            target_co_kurtosis_for_step = (
                tf.cast(
                    bounded_feature_teacher.pairwise_co_kurtosis[time_index],
                    theta.dtype,
                )
                + tf.einsum(
                    "ijp,p->ij",
                    tf.cast(
                        bounded_feature_teacher.pairwise_co_kurtosis_tangent[
                            time_index
                        ],
                        theta.dtype,
                    ),
                    theta,
                )
            )
        else:
            shape_source = particles_next
            shape_points = restored["particles"]
            shape_source_tangent = particle_tangent_next
            shape_points_tangent = restored["particles_tangent"]
            target_skew_for_step = explicit_target_skew
            target_kurtosis_for_step = explicit_target_kurtosis
            target_co_skew_for_step = explicit_target_pairwise_co_skew
            target_co_kurtosis_for_step = explicit_target_pairwise_co_kurtosis
        if projected_basis is None:
            basis_for_step = None
        elif projected_basis.shape.rank == 2:
            basis_for_step = projected_basis
        else:
            basis_for_step = projected_basis[time_index]
        higher = higher_moment_shape_jvp(
            shape_source,
            normalized_weights,
            shape_source_tangent,
            normalized_weight_tangent,
            shape_points,
            shape_points_tangent,
            correction_steps=higher_moment_correction_steps,
            strength=higher_moment_strength,
            floor=higher_moment_floor,
            pairwise_correction_steps=pairwise_moment_correction_steps,
            pairwise_strength=pairwise_moment_strength,
            pairwise_floor=pairwise_moment_floor,
            pairwise_particle_rms_cap=pairwise_particle_rms_cap,
            coordinatewise_bounded_cap=coordinatewise_bounded_cap,
            coordinatewise_bounded_cap_power=coordinatewise_bounded_cap_power,
            coordinatewise_standardized_cap=coordinatewise_standardized_cap,
            coordinatewise_standardized_cap_power=coordinatewise_standardized_cap_power,
            projected_cumulant_basis=basis_for_step,
            projected_cumulant_correction_steps=projected_cumulant_correction_steps,
            projected_cumulant_strength=projected_cumulant_strength,
            projected_cumulant_floor=projected_cumulant_floor,
            explicit_target_skew=target_skew_for_step,
            explicit_target_kurtosis=target_kurtosis_for_step,
            explicit_target_skew_tangent=(
                tf.cast(
                    bounded_feature_teacher.skew_tangent[time_index], theta.dtype
                )
                if teacher_active
                else explicit_target_skew_tangent
            ),
            explicit_target_kurtosis_tangent=(
                tf.cast(
                    bounded_feature_teacher.kurtosis_tangent[time_index],
                    theta.dtype,
                )
                if teacher_active
                else explicit_target_kurtosis_tangent
            ),
            explicit_target_pairwise_co_skew=target_co_skew_for_step,
            explicit_target_pairwise_co_kurtosis=target_co_kurtosis_for_step,
            explicit_target_pairwise_co_skew_tangent=(
                tf.cast(
                    bounded_feature_teacher.pairwise_co_skew_tangent[time_index],
                    theta.dtype,
                )
                if teacher_active
                else explicit_target_pairwise_co_skew_tangent
            ),
            explicit_target_pairwise_co_kurtosis_tangent=(
                tf.cast(
                    bounded_feature_teacher.pairwise_co_kurtosis_tangent[
                        time_index
                    ],
                    theta.dtype,
                )
                if teacher_active
                else explicit_target_pairwise_co_kurtosis_tangent
            ),
            pairwise_co_skew_target_mask=(
                tf.cast(
                    bounded_feature_teacher.pairwise_co_skew_mask[time_index],
                    theta.dtype,
                )
                if teacher_active
                else pairwise_co_skew_target_mask
            ),
            pairwise_co_kurtosis_target_mask=(
                tf.cast(
                    bounded_feature_teacher.pairwise_co_kurtosis_mask[time_index],
                    theta.dtype,
                )
                if teacher_active
                else pairwise_co_kurtosis_target_mask
            ),
        )
        if teacher_active:
            absolute_bounded_coordinate = tf.reduce_max(tf.abs(higher["particles"]))
            bounded_valid = absolute_bounded_coordinate < 1.0
            safe_bounded = tf.clip_by_value(
                higher["particles"],
                tf.cast(-1.0 + 1.0e-6, theta.dtype),
                tf.cast(1.0 - 1.0e-6, theta.dtype),
            )
            inverse_derivative = tf.pow(1.0 - tf.square(safe_bounded), -1.5)
            local_corrected = safe_bounded / tf.sqrt(
                1.0 - tf.square(safe_bounded)
            )
            local_corrected_tangent = (
                higher["particles_tangent"] * inverse_derivative[:, :, None]
            )
            maximum_inverse_derivative = tf.reduce_max(inverse_derivative)
            physical_corrected = teacher_mu[None, :] + tf.linalg.matmul(
                local_corrected, teacher_matrix, transpose_b=True
            )
            physical_corrected_tangent = tf.einsum(
                "nip,ji->njp", local_corrected_tangent, teacher_matrix
            )
            physical_restored = affine_restore_cloud_jvp(
                particles_next,
                normalized_weights,
                particle_tangent_next,
                normalized_weight_tangent,
                physical_corrected,
                physical_corrected_tangent,
            )
            higher_particles = physical_restored["particles"]
            higher_particles_tangent = physical_restored["particles_tangent"]
            physical_affine_mean_residual = physical_restored[
                "maximum_mean_residual"
            ]
            physical_affine_covariance_residual = physical_restored[
                "maximum_covariance_residual"
            ]
            normalized_physical_affine_mean_residual = physical_restored[
                "maximum_normalized_mean_residual"
            ]
            normalized_physical_affine_covariance_residual = physical_restored[
                "maximum_normalized_covariance_residual"
            ]
            shape_valid = higher["valid"] & physical_restored["valid"] & bounded_valid
        else:
            higher_particles = higher["particles"]
            higher_particles_tangent = higher["particles_tangent"]
            physical_affine_mean_residual = tf.zeros([], theta.dtype)
            physical_affine_covariance_residual = tf.zeros([], theta.dtype)
            normalized_physical_affine_mean_residual = tf.zeros([], theta.dtype)
            normalized_physical_affine_covariance_residual = tf.zeros([], theta.dtype)
            absolute_bounded_coordinate = tf.zeros([], theta.dtype)
            maximum_inverse_derivative = tf.zeros([], theta.dtype)
            shape_valid = higher["valid"]
        step_valid = stage_valid & restored["reset_valid"] & shape_valid
        shape_displacement = tf.reduce_max(
            tf.abs(higher_particles - restored["particles"])
        )
        restored_centered = restored["particles"] - tf.reduce_mean(
            restored["particles"], axis=0, keepdims=True
        )
        normalized_shape_displacement = tf.sqrt(
            tf.reduce_mean(tf.square(higher_particles - restored["particles"]))
        ) / tf.maximum(
            tf.sqrt(tf.reduce_mean(tf.square(restored_centered))),
            tf.cast(1.0e-6, theta.dtype),
        )
        normalized_skew_residual = higher["skew_residual"] / tf.maximum(
            tf.ones_like(higher["target_skew"]), tf.abs(higher["target_skew"])
        )
        normalized_kurtosis_residual = higher["kurtosis_residual"] / tf.maximum(
            tf.ones_like(higher["target_kurtosis"]),
            tf.abs(higher["target_kurtosis"]),
        )
        shape_objective = tf.reduce_mean(tf.square(normalized_skew_residual))
        shape_objective += tf.reduce_mean(tf.square(normalized_kurtosis_residual))
        pair_mask = 1.0 - tf.eye(
            tf.shape(higher["target_pairwise_co_skew"])[0], dtype=theta.dtype
        )
        pair_count = tf.maximum(tf.reduce_sum(pair_mask), tf.cast(1.0, theta.dtype))
        normalized_pairwise_co_skew_residual = (
            higher["pairwise_co_skew_residual"]
            / tf.maximum(
                tf.ones_like(higher["target_pairwise_co_skew"]),
                tf.abs(higher["target_pairwise_co_skew"]),
            )
        )
        normalized_pairwise_co_kurtosis_residual = (
            higher["pairwise_co_kurtosis_residual"]
            / tf.maximum(
                tf.ones_like(higher["target_pairwise_co_kurtosis"]),
                tf.abs(higher["target_pairwise_co_kurtosis"]),
            )
        )
        pairwise_shape_objective = tf.reduce_sum(
            pair_mask * tf.square(normalized_pairwise_co_skew_residual)
        ) / pair_count
        pairwise_shape_objective += tf.reduce_sum(
            pair_mask * tf.square(normalized_pairwise_co_kurtosis_residual)
        ) / pair_count
        if collect_mode_score:
            mode_score = _projected_cumulant_mode_score(
                particles_next,
                normalized_weights,
                higher_particles,
                sketch_directions,
            )
        else:
            mode_score = tf.zeros(
                [state_dimension, state_dimension], theta.dtype
            )
        restored_particles = tf.where(step_valid, higher_particles, particles_value)
        restored_tangent = tf.where(
            step_valid,
            higher_particles_tangent,
            particle_tangent_value,
        )
        mean_residual = restored["mean_residual"]
        marginal = tf.stack(
            [
                restored["maximum_raw_row_residual"],
                restored["maximum_post_quotient_column_residual"],
            ]
        )
        uniform_weights = tf.fill(
            [n_static], tf.cast(1.0, theta.dtype) / tf.cast(n, theta.dtype)
        )
        return (
            time_index + 1,
            restored_particles,
            restored_tangent,
            uniform_weights,
            tf.zeros_like(weight_tangent_value),
            total_value + tf.where(stage_valid, increment, tf.zeros_like(increment)),
            score_value
            + tf.where(stage_valid, increment_tangent, tf.zeros_like(increment_tangent)),
            tf.maximum(max_mean_value, mean_residual),
            tf.maximum(max_marginal_value, marginal),
            tf.maximum(max_transition_residual_value, transition_residual),
            program_valid_value & step_valid,
            tf.minimum(minimum_row_mass_value, restored["minimum_row_mass"]),
            tf.maximum(
                maximum_column_tv_error_value,
                restored["post_quotient_column_tv_error"],
            ),
            tf.minimum(
                minimum_gap_eigenvalue_value,
                restored["minimum_gap_eigenvalue"],
            ),
            tf.maximum(
                maximum_skew_residual_value,
                tf.reduce_max(tf.abs(higher["skew_residual"])),
            ),
            tf.maximum(
                maximum_kurtosis_residual_value,
                tf.reduce_max(tf.abs(higher["kurtosis_residual"])),
            ),
            tf.maximum(
                maximum_pairwise_co_skew_residual_value,
                tf.reduce_max(tf.abs(higher["pairwise_co_skew_residual"])),
            ),
            tf.maximum(
                maximum_pairwise_co_kurtosis_residual_value,
                tf.reduce_max(tf.abs(higher["pairwise_co_kurtosis_residual"])),
            ),
            tf.maximum(
                maximum_pairwise_pre_cap_particle_rms_value,
                higher["maximum_pairwise_pre_cap_particle_rms"],
            ),
            tf.maximum(
                maximum_pairwise_post_cap_particle_rms_value,
                higher["maximum_pairwise_post_cap_particle_rms"],
            ),
            tf.minimum(
                minimum_pairwise_particle_cap_scale_value,
                higher["minimum_pairwise_particle_cap_scale"],
            ),
            tf.maximum(
                maximum_physical_affine_mean_residual_value,
                physical_affine_mean_residual,
            ),
            tf.maximum(
                maximum_physical_affine_covariance_residual_value,
                physical_affine_covariance_residual,
            ),
            tf.maximum(
                maximum_normalized_physical_affine_mean_residual_value,
                normalized_physical_affine_mean_residual,
            ),
            tf.maximum(
                maximum_normalized_physical_affine_covariance_residual_value,
                normalized_physical_affine_covariance_residual,
            ),
            tf.maximum(
                maximum_absolute_bounded_coordinate_value,
                absolute_bounded_coordinate,
            ),
            tf.maximum(
                maximum_coordinatewise_pre_cap_absolute_value,
                higher["maximum_coordinatewise_pre_cap_absolute"],
            ),
            tf.maximum(
                maximum_coordinatewise_post_cap_absolute_value,
                higher["maximum_coordinatewise_post_cap_absolute"],
            ),
            coordinatewise_cap_displacement_sum_value
            + higher["mean_coordinatewise_cap_displacement"],
            tf.maximum(
                maximum_coordinatewise_cap_active_fraction_value,
                higher["fraction_coordinatewise_cap_active"],
            ),
            tf.minimum(
                minimum_coordinatewise_cap_derivative_value,
                higher["minimum_coordinatewise_cap_derivative"],
            ),
            tf.maximum(
                maximum_coordinatewise_inverse_derivative_value,
                maximum_inverse_derivative,
            ),
            tf.maximum(maximum_shape_displacement_value, shape_displacement),
            tf.maximum(
                maximum_normalized_shape_displacement_value,
                normalized_shape_displacement,
            ),
            shape_objective_sum_value + shape_objective,
            pairwise_shape_objective_sum_value + pairwise_shape_objective,
            tf.maximum(
                maximum_projected_residual_value,
                higher["projected_cumulant_residual_norm"],
            ),
            tf.maximum(
                maximum_projected_third_residual_value,
                higher["projected_cumulant_third_residual_norm"],
            ),
            tf.maximum(
                maximum_projected_fourth_residual_value,
                higher["projected_cumulant_fourth_residual_norm"],
            ),
            increments_value.write(time_index, increment),
            score_increments_value.write(time_index, increment_tangent),
            mode_scores_value.write(time_index, mode_score),
        )

    (
        _,
        particles,
        particle_tangent,
        weights,
        weight_tangent,
        total,
        score,
        max_mean,
        max_marginal,
        max_transition_residual,
        program_valid,
        minimum_row_mass,
        maximum_column_tv_error,
        minimum_gap_eigenvalue,
        maximum_skew_residual,
        maximum_kurtosis_residual,
        maximum_pairwise_co_skew_residual,
        maximum_pairwise_co_kurtosis_residual,
        maximum_pairwise_pre_cap_particle_rms,
        maximum_pairwise_post_cap_particle_rms,
        minimum_pairwise_particle_cap_scale,
        maximum_physical_affine_mean_residual,
        maximum_physical_affine_covariance_residual,
        maximum_normalized_physical_affine_mean_residual,
        maximum_normalized_physical_affine_covariance_residual,
        maximum_absolute_bounded_coordinate,
        maximum_coordinatewise_pre_cap_absolute,
        maximum_coordinatewise_post_cap_absolute,
        coordinatewise_cap_displacement_sum,
        maximum_coordinatewise_cap_active_fraction,
        minimum_coordinatewise_cap_derivative,
        maximum_coordinatewise_inverse_derivative,
        maximum_shape_displacement,
        maximum_normalized_shape_displacement,
        shape_objective_sum,
        pairwise_shape_objective_sum,
        maximum_projected_residual,
        maximum_projected_third_residual,
        maximum_projected_fourth_residual,
        increments,
        score_increments,
        mode_scores,
    ) = tf.while_loop(
        lambda time_index, *_: time_index < horizon,
        time_body,
        (
            tf.zeros([], tf.int32),
            particles,
            particle_tangent,
            weights,
            weight_tangent,
            total,
            score,
            max_mean,
            max_marginal,
            max_transition_residual,
            program_valid,
            minimum_row_mass,
            maximum_column_tv_error,
            minimum_gap_eigenvalue,
            maximum_skew_residual,
            maximum_kurtosis_residual,
            maximum_pairwise_co_skew_residual,
            maximum_pairwise_co_kurtosis_residual,
            maximum_pairwise_pre_cap_particle_rms,
            maximum_pairwise_post_cap_particle_rms,
            minimum_pairwise_particle_cap_scale,
            maximum_physical_affine_mean_residual,
            maximum_physical_affine_covariance_residual,
            maximum_normalized_physical_affine_mean_residual,
            maximum_normalized_physical_affine_covariance_residual,
            maximum_absolute_bounded_coordinate,
            maximum_coordinatewise_pre_cap_absolute,
            maximum_coordinatewise_post_cap_absolute,
            coordinatewise_cap_displacement_sum,
            maximum_coordinatewise_cap_active_fraction,
            minimum_coordinatewise_cap_derivative,
            maximum_coordinatewise_inverse_derivative,
            maximum_shape_displacement,
            maximum_normalized_shape_displacement,
            shape_objective_sum,
            pairwise_shape_objective_sum,
            tf.zeros([], theta.dtype),
            tf.zeros([], theta.dtype),
            tf.zeros([], theta.dtype),
            increments,
            score_increments,
            mode_scores,
        ),
        parallel_iterations=1,
    )
    diagnostics = {
        "max_mean_residual": max_mean,
        "max_row_residual": max_marginal[0],
        "max_col_residual": max_marginal[1],
        "max_transition_residual": max_transition_residual,
        "program_valid": program_valid,
        "minimum_row_mass": minimum_row_mass,
        "maximum_post_quotient_column_tv_error": maximum_column_tv_error,
        "minimum_covariance_gap_eigenvalue": minimum_gap_eigenvalue,
        "maximum_skew_residual": maximum_skew_residual,
        "maximum_kurtosis_residual": maximum_kurtosis_residual,
        "maximum_pairwise_co_skew_residual": (
            maximum_pairwise_co_skew_residual
        ),
        "maximum_pairwise_co_kurtosis_residual": (
            maximum_pairwise_co_kurtosis_residual
        ),
        "maximum_pairwise_pre_cap_particle_rms": (
            maximum_pairwise_pre_cap_particle_rms
        ),
        "maximum_pairwise_post_cap_particle_rms": (
            maximum_pairwise_post_cap_particle_rms
        ),
        "minimum_pairwise_particle_cap_scale": (
            minimum_pairwise_particle_cap_scale
        ),
        "maximum_physical_affine_mean_residual": (
            maximum_physical_affine_mean_residual
        ),
        "maximum_physical_affine_covariance_residual": (
            maximum_physical_affine_covariance_residual
        ),
        "maximum_normalized_physical_affine_mean_residual": (
            maximum_normalized_physical_affine_mean_residual
        ),
        "maximum_normalized_physical_affine_covariance_residual": (
            maximum_normalized_physical_affine_covariance_residual
        ),
        "maximum_absolute_bounded_coordinate": maximum_absolute_bounded_coordinate,
        "maximum_coordinatewise_pre_cap_absolute": maximum_coordinatewise_pre_cap_absolute,
        "maximum_coordinatewise_post_cap_absolute": maximum_coordinatewise_post_cap_absolute,
        "mean_coordinatewise_cap_displacement": coordinatewise_cap_displacement_sum
        / tf.cast(horizon, theta.dtype),
        "fraction_coordinatewise_cap_active": maximum_coordinatewise_cap_active_fraction,
        "minimum_coordinatewise_cap_derivative": minimum_coordinatewise_cap_derivative,
        "maximum_coordinatewise_inverse_derivative": maximum_coordinatewise_inverse_derivative,
        "maximum_shape_displacement": maximum_shape_displacement,
        "maximum_normalized_shape_displacement": (
            maximum_normalized_shape_displacement
        ),
        "mean_normalized_shape_residual_objective": (
            shape_objective_sum / tf.cast(horizon, theta.dtype)
        ),
        "mean_normalized_pairwise_shape_residual_objective": (
            pairwise_shape_objective_sum / tf.cast(horizon, theta.dtype)
        ),
        "maximum_projected_cumulant_residual": maximum_projected_residual,
        "maximum_projected_cumulant_third_residual": (
            maximum_projected_third_residual
        ),
        "maximum_projected_cumulant_fourth_residual": (
            maximum_projected_fourth_residual
        ),
        "projected_cumulant_mode_score": mode_scores.stack(),
        "value_increments": increments.stack(),
        "score_increments": score_increments.stack(),
    }
    nan_value = tf.constant(float("nan"), theta.dtype)
    return (
        tf.where(program_valid, total, nan_value),
        tf.where(program_valid, score, tf.fill([parameter_count], nan_value)),
        diagnostics,
    )


__all__ = [
    "BoundedFeatureShapeTeacher",
    "CandidateModelAdapter",
    "finite_value_score",
]
