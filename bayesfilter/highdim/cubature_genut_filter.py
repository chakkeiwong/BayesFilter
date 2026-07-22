"""Candidate-only generic staged Cubature/GenUT value and total-JVP core."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import tensorflow as tf

from bayesfilter.highdim import ledh_contract_e_reset_tf as reset


Tensor = tf.Tensor
InitialValue = Callable[[Tensor, Tensor], Tensor]
InitialTangent = Callable[[Tensor, Tensor], Tensor]
TransitionValue = Callable[[Tensor, Tensor, Tensor, Tensor], Tensor]
TransitionTangent = Callable[[Tensor, Tensor, Tensor, Tensor, Tensor], Tensor]
ObservationValue = Callable[[Tensor, Tensor, Tensor, Tensor], Tensor]
ObservationTangent = Callable[[Tensor, Tensor, Tensor, Tensor, Tensor], Tensor]


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

    def __post_init__(self) -> None:
        if self.state_dimension < 1 or self.parameter_count < 1:
            raise ValueError("adapter dimensions must be positive")


def _sinkhorn_barycentric_jvp(
    particles: Tensor,
    weights: Tensor,
    particle_tangent: Tensor,
    weight_tangent: Tensor,
    *,
    epsilon: float,
    sinkhorn_steps: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
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
    coupling = left[:, None] * kernel * right[None, :]
    coupling_tangent = (
        left_tangent[:, None, :] * kernel[:, :, None] * right[None, :, None]
        + left[:, None, None] * kernel_tangent * right[None, :, None]
        + left[:, None, None] * kernel[:, :, None] * right_tangent[None, :, :]
    )
    gamma = tf.cast(n, particles.dtype) * coupling
    gamma_tangent = tf.cast(n, particles.dtype) * coupling_tangent
    barycentric = gamma @ particles
    barycentric_tangent = (
        tf.einsum("ijp,jd->idp", gamma_tangent, particles)
        + tf.einsum("ij,jdp->idp", gamma, particle_tangent)
    )
    row_residual = tf.reduce_max(tf.abs(tf.reduce_sum(coupling, axis=1) - uniform))
    col_residual = tf.reduce_max(tf.abs(tf.reduce_sum(coupling, axis=0) - weights))
    return barycentric, barycentric_tangent, row_residual, col_residual


def _restore_cloud_jvp(
    particles: Tensor,
    weights: Tensor,
    particle_tangent: Tensor,
    weight_tangent: Tensor,
    design: Tensor,
    *,
    epsilon: float,
    sinkhorn_steps: int,
    ridge: float,
    parameter_count: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    barycentric, barycentric_tangent, row_residual, col_residual = (
        _sinkhorn_barycentric_jvp(
            particles, weights, particle_tangent, weight_tangent,
            epsilon=epsilon, sinkhorn_steps=sinkhorn_steps,
        )
    )
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
        tf.transpose(barycentric_tangent, [2, 0, 1]),
        tf.zeros_like(design_batch),
        tf.zeros_like(ridge_batch),
    )["particles"]
    restored_tangent = tf.transpose(tangent_batch, [1, 2, 0])
    forward = reset._contract_e_chol_cloud_forward_core(  # noqa: SLF001
        source, source_weights, transported, residual_design, ridge_tensor
    )
    return (
        forward["particles"][0],
        restored_tangent,
        tf.reduce_max(tf.abs(forward["mean_residual"])),
        tf.stack([row_residual, col_residual]),
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
    ridge: float = 1.0e-5,
    transition_before_first_observation: bool = True,
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
    horizon = tf.shape(observations)[0]
    increments = tf.TensorArray(theta.dtype, size=horizon, element_shape=())
    score_increments = tf.TensorArray(
        theta.dtype, size=horizon, element_shape=(parameter_count,)
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
        increments_value: tf.TensorArray,
        score_increments_value: tf.TensorArray,
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
        current_design = design if design.shape.rank == 2 else design[time_index]
        restored_particles, restored_tangent, mean_residual, marginal = (
            _restore_cloud_jvp(
                particles_next,
                normalized_weights,
                particle_tangent_next,
                normalized_weight_tangent,
                current_design,
                epsilon=epsilon,
                sinkhorn_steps=sinkhorn_steps,
                ridge=ridge,
                parameter_count=parameter_count,
            )
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
            total_value + increment,
            score_value + increment_tangent,
            tf.maximum(max_mean_value, mean_residual),
            tf.maximum(max_marginal_value, marginal),
            increments_value.write(time_index, increment),
            score_increments_value.write(time_index, increment_tangent),
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
        increments,
        score_increments,
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
            increments,
            score_increments,
        ),
        parallel_iterations=1,
    )
    diagnostics = {
        "max_mean_residual": max_mean,
        "max_row_residual": max_marginal[0],
        "max_col_residual": max_marginal[1],
        "value_increments": increments.stack(),
        "score_increments": score_increments.stack(),
    }
    return total, score, diagnostics


__all__ = ["CandidateModelAdapter", "finite_value_score"]
