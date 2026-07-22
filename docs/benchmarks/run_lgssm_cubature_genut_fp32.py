#!/usr/bin/env python3
"""Float32/TF32 staged Contract E Cubature and GenUT LGSSM diagnostic.

This diagnostic keeps the staged order:
transition/current increment -> positive Sinkhorn OT -> barycentric cloud
-> residual injection -> Cholesky restoration.

For the Gaussian LGSSM, GenUT uses s=0 and k=3. Its central weight is zero,
so the positive equal-weight realization is the six-point cubature design.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tensorflow as tf


SCHEMA_VERSION = "bayesfilter.lgssm_cubature_genut_fp32.v3"
CAMPAIGN_ID = "lgssm-cubature-genut-recursive-score-fp32-20260721"
STATE_DIM = 3
NUM_PARTICLES = 1008
HORIZONS = (2, 10, 50)
DATASET_SEED = 81100
PARTICLE_SEEDS = tuple(range(82220, 82236))
EPSILON = 2.0
SINKHORN_STEPS = 8
RIDGE = 1.0e-5
FD_EPS = 2.0e-3
FD_MIN_STEP = 1.0e-4
THETA_VALUES = (0.72, 0.55, 0.35, 0.35, 0.45)
LOG_TWO_PI = math.log(2.0 * math.pi)
OBSERVATION_DIM = 3
OBSERVATION_MATRIX_VALUES = (
    (1.0, 0.25, -0.15),
    (0.2, 1.1, 0.3),
    (-0.1, 0.35, 0.9),
)
LABELS = ("value", "phi1", "phi2", "phi3", "q_scale", "r_scale")
CRITICAL_VALUE = 3.036283222821165
VALUE_MARGIN = 0.001
SCORE_MARGIN = 0.05
PARAMETER_DIM = STATE_DIM + 2
SCORE_ROUTE_ID = "compact_forward_sensitivity_no_autodiff_cubature_genut_v1"


def _observation_matrix() -> tf.Tensor:
    return tf.constant(OBSERVATION_MATRIX_VALUES, dtype=tf.float32)


def _contract_e_reset_module() -> Any:
    # Import after the runner has configured GPU memory growth. Importing the
    # highdim package at module load can create TensorFlow objects too early.
    from bayesfilter.highdim import ledh_contract_e_reset_tf

    return ledh_contract_e_reset_tf


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _tensor_hash(value: tf.Tensor) -> str:
    return hashlib.sha256(tf.io.serialize_tensor(value).numpy()).hexdigest()


def _configure_gpu(*, jit_compile: bool) -> dict[str, Any]:
    physical = tf.config.list_physical_devices("GPU")
    if not physical:
        raise RuntimeError("TF32 experiment requires a visible GPU")
    for device in physical:
        tf.config.experimental.set_memory_growth(device, True)
    tf.config.experimental.enable_tensor_float_32_execution(True)
    logical = tf.config.list_logical_devices("GPU")
    if not logical:
        raise RuntimeError("GPU initialization produced no logical GPU")
    return {
        "physical_devices": [device.name for device in physical],
        "logical_devices": [device.name for device in logical],
        "memory_growth": True,
        "tf32_execution_enabled": True,
        "tf32_mode": "enabled",
        "dtype": "float32",
        "jit_compile": jit_compile,
        "trust_basis": "owner_designated_managed_session_visible_gpu_trusted",
    }


def cubature_design(*, dim: int, num_particles: int) -> tf.Tensor:
    if dim <= 0 or num_particles <= 0 or num_particles % (2 * dim) != 0:
        raise ValueError("num_particles must be divisible by 2*dim")
    eye = tf.eye(dim, dtype=tf.float32)
    base = tf.sqrt(tf.cast(dim, tf.float32)) * tf.concat([eye, -eye], axis=0)
    return tf.repeat(base, repeats=num_particles // (2 * dim), axis=0)


def genut_gaussian_design(*, dim: int, num_particles: int) -> tuple[tf.Tensor, dict[str, Any]]:
    if dim <= 0 or num_particles <= 0 or num_particles % (2 * dim) != 0:
        raise ValueError("Gaussian GenUT realization requires divisibility by 2*dim")
    u = tf.sqrt(tf.constant(3.0, tf.float32))
    eye = tf.eye(dim, dtype=tf.float32)
    base = tf.concat([u * eye, -u * eye], axis=0)
    points = tf.repeat(base, repeats=num_particles // (2 * dim), axis=0)
    return points, {
        "rule": "genut",
        "standardized_skewness": [0.0] * dim,
        "standardized_kurtosis": [3.0] * dim,
        "central_weight": 0.0,
        "noncentral_weight": 1.0 / (2.0 * dim),
        "central_point_omitted": True,
        "effective_point_count": 2 * dim,
    }


def contract_e_gaussian_design(
    *, horizon: int, num_particles: int, particle_seed: int
) -> tuple[tf.Tensor, dict[str, Any]]:
    """Fixed centered Gaussian residual design used by original Contract E."""
    if horizon <= 0 or num_particles <= 1:
        raise ValueError("Gaussian residual design requires T > 0 and N > 1")
    raw = tf.random.stateless_normal(
        [horizon, num_particles, STATE_DIM],
        seed=[particle_seed, horizon + 2000],
        dtype=tf.float32,
    )
    centered = raw - tf.reduce_mean(raw, axis=1, keepdims=True)
    scale = tf.sqrt(
        tf.cast(num_particles, tf.float32)
        / tf.cast(num_particles - 1, tf.float32)
    )
    return scale * centered, {
        "rule": "contract_e_gaussian_residual",
        "residual_design_id": "contract_e_residual_centered_population_scaled_v1",
        "time_varying": True,
        "point_count": num_particles,
        "population_centered": True,
        "population_scale": "sqrt(N/(N-1))",
    }


def _design_at(design: tf.Tensor, time_index: tf.Tensor) -> tf.Tensor:
    if design.shape.rank == 2:
        return design
    if design.shape.rank == 3:
        return design[time_index]
    raise ValueError("residual design must have rank two or three")


def _sym(value: tf.Tensor) -> tf.Tensor:
    return 0.5 * (value + tf.transpose(value))


def _weighted_mean_covariance(
    particles: tf.Tensor, weights: tf.Tensor
) -> tuple[tf.Tensor, tf.Tensor]:
    mean = tf.reduce_sum(particles * weights[:, None], axis=0)
    centered = particles - mean[None, :]
    covariance = tf.transpose(centered) @ (centered * weights[:, None])
    return mean, _sym(covariance)


def _sinkhorn_barycentric(
    particles: tf.Tensor,
    weights: tf.Tensor,
    *,
    epsilon: float = EPSILON,
    sinkhorn_steps: int = SINKHORN_STEPS,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    if epsilon <= 0.0 or sinkhorn_steps <= 0:
        raise ValueError("epsilon and sinkhorn_steps must be positive")
    deltas = particles[:, None, :] - particles[None, :, :]
    cost = tf.reduce_sum(tf.square(deltas), axis=-1)
    cost_scale = tf.maximum(tf.reduce_mean(cost), tf.constant(1.0e-3, tf.float32))
    kernel = tf.exp(-cost / (cost_scale * tf.constant(epsilon, tf.float32)))
    n = tf.shape(particles)[0]
    uniform = tf.fill([n], tf.cast(1.0, tf.float32) / tf.cast(n, tf.float32))
    left = tf.ones_like(uniform)
    right = tf.ones_like(uniform)
    tiny = tf.constant(1.0e-7, tf.float32)

    def body(step: tf.Tensor, left_: tf.Tensor, right_: tf.Tensor):
        left_new = uniform / (tf.linalg.matvec(kernel, right_) + tiny)
        right_new = weights / (tf.linalg.matvec(tf.transpose(kernel), left_new) + tiny)
        return step + 1, left_new, right_new

    _, left, right = tf.while_loop(
        lambda step, _left, _right: step < sinkhorn_steps,
        body,
        (tf.constant(0, tf.int32), left, right),
        maximum_iterations=sinkhorn_steps,
    )
    coupling = left[:, None] * kernel * right[None, :]
    gamma = tf.cast(n, tf.float32) * coupling
    barycentric = gamma @ particles
    row_residual = tf.reduce_max(tf.abs(tf.reduce_sum(coupling, axis=1) - uniform))
    col_residual = tf.reduce_max(tf.abs(tf.reduce_sum(coupling, axis=0) - weights))
    return barycentric, row_residual, col_residual


def _sinkhorn_barycentric_jvp(
    particles: tf.Tensor,
    weights: tf.Tensor,
    particle_tangent: tf.Tensor,
    weight_tangent: tf.Tensor,
    *,
    epsilon: float = EPSILON,
    sinkhorn_steps: int = SINKHORN_STEPS,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Finite Sinkhorn primal and all-parameter forward sensitivities."""
    if epsilon <= 0.0 or sinkhorn_steps <= 0:
        raise ValueError("epsilon and sinkhorn_steps must be positive")
    deltas = particles[:, None, :] - particles[None, :, :]
    delta_tangent = particle_tangent[:, None, :, :] - particle_tangent[None, :, :, :]
    cost = tf.reduce_sum(tf.square(deltas), axis=-1)
    cost_tangent = 2.0 * tf.reduce_sum(deltas[:, :, :, None] * delta_tangent, axis=2)
    mean_cost = tf.reduce_mean(cost)
    mean_cost_tangent = tf.reduce_mean(cost_tangent, axis=[0, 1])
    floor = tf.constant(1.0e-3, tf.float32)
    cost_scale = tf.maximum(mean_cost, floor)
    cost_scale_tangent = tf.where(
        mean_cost > floor, mean_cost_tangent, tf.zeros_like(mean_cost_tangent)
    )
    epsilon_tensor = tf.constant(epsilon, tf.float32)
    exponent = -cost / (cost_scale * epsilon_tensor)
    exponent_tangent = -(
        cost_tangent / cost_scale
        - cost[:, :, None] * cost_scale_tangent[None, None, :] / tf.square(cost_scale)
    ) / epsilon_tensor
    kernel = tf.exp(exponent)
    kernel_tangent = kernel[:, :, None] * exponent_tangent
    n = tf.shape(particles)[0]
    uniform = tf.fill([n], tf.cast(1.0, tf.float32) / tf.cast(n, tf.float32))
    left = tf.ones_like(uniform)
    right = tf.ones_like(uniform)
    left_tangent = tf.zeros([n, PARAMETER_DIM], tf.float32)
    right_tangent = tf.zeros([n, PARAMETER_DIM], tf.float32)
    tiny = tf.constant(1.0e-7, tf.float32)

    def body(
        step: tf.Tensor,
        left_: tf.Tensor,
        right_: tf.Tensor,
        left_tangent_: tf.Tensor,
        right_tangent_: tf.Tensor,
    ):
        left_denominator = tf.linalg.matvec(kernel, right_) + tiny
        left_denominator_tangent = (
            tf.einsum("ijp,j->ip", kernel_tangent, right_)
            + tf.einsum("ij,jp->ip", kernel, right_tangent_)
        )
        left_new = uniform / left_denominator
        left_tangent_new = (
            -uniform[:, None]
            * left_denominator_tangent
            / tf.square(left_denominator)[:, None]
        )
        right_denominator = tf.linalg.matvec(tf.transpose(kernel), left_new) + tiny
        right_denominator_tangent = (
            tf.einsum("ijp,i->jp", kernel_tangent, left_new)
            + tf.einsum("ij,ip->jp", kernel, left_tangent_new)
        )
        right_new = weights / right_denominator
        right_tangent_new = (
            weight_tangent / right_denominator[:, None]
            - weights[:, None]
            * right_denominator_tangent
            / tf.square(right_denominator)[:, None]
        )
        return (
            step + 1,
            left_new,
            right_new,
            left_tangent_new,
            right_tangent_new,
        )

    _, left, right, left_tangent, right_tangent = tf.while_loop(
        lambda step, *_: step < sinkhorn_steps,
        body,
        (
            tf.constant(0, tf.int32),
            left,
            right,
            left_tangent,
            right_tangent,
        ),
        maximum_iterations=sinkhorn_steps,
    )
    coupling = left[:, None] * kernel * right[None, :]
    coupling_tangent = (
        left_tangent[:, None, :] * kernel[:, :, None] * right[None, :, None]
        + left[:, None, None] * kernel_tangent * right[None, :, None]
        + left[:, None, None] * kernel[:, :, None] * right_tangent[None, :, :]
    )
    gamma = tf.cast(n, tf.float32) * coupling
    gamma_tangent = tf.cast(n, tf.float32) * coupling_tangent
    barycentric = gamma @ particles
    barycentric_tangent = (
        tf.einsum("ijp,jd->idp", gamma_tangent, particles)
        + tf.einsum("ij,jdp->idp", gamma, particle_tangent)
    )
    row_residual = tf.reduce_max(tf.abs(tf.reduce_sum(coupling, axis=1) - uniform))
    col_residual = tf.reduce_max(tf.abs(tf.reduce_sum(coupling, axis=0) - weights))
    return barycentric, barycentric_tangent, row_residual, col_residual


def _restore_cloud(
    particles: tf.Tensor,
    weights: tf.Tensor,
    design: tf.Tensor,
    *,
    epsilon: float = EPSILON,
    sinkhorn_steps: int = SINKHORN_STEPS,
    ridge: float = RIDGE,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    barycentric, row_residual, col_residual = _sinkhorn_barycentric(
        particles,
        weights,
        epsilon=epsilon,
        sinkhorn_steps=sinkhorn_steps,
    )
    reset_module = _contract_e_reset_module()
    forward = reset_module._contract_e_chol_cloud_forward_core(  # noqa: SLF001
        particles[None, :, :],
        weights[None, :],
        barycentric[None, :, :],
        design[None, :, :],
        tf.constant([ridge], tf.float32),
    )
    mean_residual = tf.reduce_max(tf.abs(forward["mean_residual"]))
    covariance_residual = tf.reduce_max(tf.abs(forward["ridged_identity_residual"]))
    return forward["particles"][0], tf.stack(
        [mean_residual, covariance_residual]
    ), tf.stack(
        [row_residual, col_residual]
    )


def _restore_cloud_jvp(
    particles: tf.Tensor,
    weights: tf.Tensor,
    particle_tangent: tf.Tensor,
    weight_tangent: tf.Tensor,
    design: tf.Tensor,
    *,
    epsilon: float = EPSILON,
    sinkhorn_steps: int = SINKHORN_STEPS,
    ridge: float = RIDGE,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Apply the reset and its complete all-parameter manual JVP."""
    barycentric, barycentric_tangent, row_residual, col_residual = (
        _sinkhorn_barycentric_jvp(
            particles,
            weights,
            particle_tangent,
            weight_tangent,
            epsilon=epsilon,
            sinkhorn_steps=sinkhorn_steps,
        )
    )
    source = particles[None, :, :]
    normalized_weights = weights[None, :]
    transported = barycentric[None, :, :]
    residual_design = design[None, :, :]
    ridge_tensor = tf.constant([ridge], tf.float32)
    reset_module = _contract_e_reset_module()
    forward = reset_module._contract_e_chol_cloud_forward_core(  # noqa: SLF001
        source,
        normalized_weights,
        transported,
        residual_design,
        ridge_tensor,
    )
    tangent_columns = []
    for parameter_index in range(PARAMETER_DIM):
        reset_tangent = reset_module._contract_e_chol_cloud_jvp_from_forward_core(  # noqa: SLF001
            forward,
            source,
            normalized_weights,
            transported,
            residual_design,
            ridge_tensor,
            particle_tangent[None, :, :, parameter_index],
            weight_tangent[None, :, parameter_index],
            barycentric_tangent[None, :, :, parameter_index],
            tf.zeros_like(residual_design),
            tf.zeros_like(ridge_tensor),
        )["particles"][0]
        tangent_columns.append(reset_tangent)
    restored = forward["particles"][0]
    restored_tangent = tf.stack(tangent_columns, axis=-1)
    mean_residual = tf.reduce_max(tf.abs(forward["mean_residual"]))
    covariance_residual = tf.reduce_max(tf.abs(forward["ridged_identity_residual"]))
    return restored, restored_tangent, tf.stack(
        [mean_residual, covariance_residual]
    ), tf.stack([row_residual, col_residual])


def _lgssm_observations(theta: tf.Tensor, horizon: int) -> tf.Tensor:
    del theta
    # This is the exact fixed observation stream used by the prior Contract E
    # particle-bias runs; only the final arithmetic is cast to float32.
    from scripts.filtering_value_gradient_benchmark_generate_p8_datasets import (
        _lgssm_dataset,
    )

    return tf.cast(_lgssm_dataset(DATASET_SEED)["observations"][:horizon], tf.float32)


def _kalman_value(theta: tf.Tensor, observations: tf.Tensor) -> tf.Tensor:
    # Match the previous Contract E aggregation oracle exactly: observations
    # are first cast to float32, then the Kalman value/score is evaluated in
    # float64 by the repository linear-Gaussian backend.
    from bayesfilter.linear.kalman_tf import tf_kalman_log_likelihood

    theta64 = tf.cast(theta, tf.float64)
    observations64 = tf.cast(observations, tf.float64)
    phi = theta64[:STATE_DIM]
    q_scale = theta64[STATE_DIM]
    r_scale = theta64[STATE_DIM + 1]
    observation_matrix = tf.constant(OBSERVATION_MATRIX_VALUES, tf.float64)
    return tf_kalman_log_likelihood(
        observations=observations64,
        transition_offset=tf.zeros([STATE_DIM], tf.float64),
        transition_matrix=tf.linalg.diag(phi),
        transition_covariance=tf.square(q_scale) * tf.eye(STATE_DIM, dtype=tf.float64),
        observation_offset=tf.zeros([OBSERVATION_DIM], tf.float64),
        observation_matrix=observation_matrix,
        observation_covariance=tf.square(r_scale) * tf.eye(OBSERVATION_DIM, dtype=tf.float64),
        initial_state_mean=tf.zeros([STATE_DIM], tf.float64),
        initial_state_covariance=tf.linalg.diag(
            tf.square(q_scale) / (1.0 - tf.square(phi))
        ),
    )


def _kalman_value_score(
    theta: tf.Tensor, observations: tf.Tensor
) -> tuple[tf.Tensor, tf.Tensor]:
    """Exact recursive Kalman value and analytic physical-parameter score."""
    from bayesfilter.linear.experimental_batched_kalman_tf import (
        tf_batched_kalman_value_and_score,
    )

    theta64 = tf.cast(theta, tf.float64)
    observations64 = tf.cast(observations, tf.float64)
    phi = theta64[:STATE_DIM]
    q_scale = theta64[STATE_DIM]
    r_scale = theta64[STATE_DIM + 1]
    state_eye = tf.eye(STATE_DIM, dtype=tf.float64)
    observation_matrix = tf.constant(OBSERVATION_MATRIX_VALUES, tf.float64)
    transition_matrix = tf.linalg.diag(phi)
    transition_covariance = tf.square(q_scale) * state_eye
    observation_covariance = tf.square(r_scale) * tf.eye(
        OBSERVATION_DIM, dtype=tf.float64
    )
    initial_covariance = tf.linalg.diag(
        tf.square(q_scale) / (1.0 - tf.square(phi))
    )
    zero_state = tf.zeros([1, STATE_DIM], tf.float64)
    zero_observation = tf.zeros([1, OBSERVATION_DIM], tf.float64)
    zero_d_state = tf.zeros([1, PARAMETER_DIM, STATE_DIM], tf.float64)
    zero_d_state_matrix = tf.zeros(
        [1, PARAMETER_DIM, STATE_DIM, STATE_DIM], tf.float64
    )
    zero_d_observation = tf.zeros(
        [1, PARAMETER_DIM, OBSERVATION_DIM], tf.float64
    )
    zero_d_observation_matrix = tf.zeros(
        [1, PARAMETER_DIM, OBSERVATION_DIM, STATE_DIM], tf.float64
    )
    zero_d_observation_covariance = tf.zeros(
        [1, PARAMETER_DIM, OBSERVATION_DIM, OBSERVATION_DIM], tf.float64
    )
    d_transition_matrix = tf.stack(
        [
            tf.linalg.diag(tf.one_hot(index, STATE_DIM, dtype=tf.float64))
            for index in range(STATE_DIM)
        ]
        + [tf.zeros_like(state_eye), tf.zeros_like(state_eye)],
        axis=0,
    )[None, :, :, :]
    d_transition_covariance = tf.tensor_scatter_nd_update(
        zero_d_state_matrix,
        indices=tf.constant([[0, STATE_DIM]], tf.int32),
        updates=(2.0 * q_scale * state_eye)[None, :, :],
    )
    d_observation_covariance = tf.tensor_scatter_nd_update(
        zero_d_observation_covariance,
        indices=tf.constant([[0, STATE_DIM + 1]], tf.int32),
        updates=(
            2.0
            * r_scale
            * tf.eye(OBSERVATION_DIM, dtype=tf.float64)
        )[None, :, :],
    )
    initial_variance_derivatives = []
    for parameter_index in range(PARAMETER_DIM):
        if parameter_index < STATE_DIM:
            diagonal = tf.one_hot(
                parameter_index, STATE_DIM, dtype=tf.float64
            ) * (
                2.0
                * tf.square(q_scale)
                * phi
                / tf.square(1.0 - tf.square(phi))
            )
        elif parameter_index == STATE_DIM:
            diagonal = 2.0 * q_scale / (1.0 - tf.square(phi))
        else:
            diagonal = tf.zeros([STATE_DIM], tf.float64)
        initial_variance_derivatives.append(tf.linalg.diag(diagonal))
    d_initial_covariance = tf.stack(initial_variance_derivatives, axis=0)[
        None, :, :, :
    ]
    value, score = tf_batched_kalman_value_and_score(
        observations=observations64,
        transition_offset=zero_state,
        transition_matrix=transition_matrix[None, :, :],
        transition_covariance=transition_covariance[None, :, :],
        observation_offset=zero_observation,
        observation_matrix=observation_matrix[None, :, :],
        observation_covariance=observation_covariance[None, :, :],
        initial_state_mean=zero_state,
        initial_state_covariance=initial_covariance[None, :, :],
        d_initial_state_mean=zero_d_state,
        d_initial_state_covariance=d_initial_covariance,
        d_transition_offset=zero_d_state,
        d_transition_matrix=d_transition_matrix,
        d_transition_covariance=d_transition_covariance,
        d_observation_offset=zero_d_observation,
        d_observation_matrix=zero_d_observation_matrix,
        d_observation_covariance=d_observation_covariance,
    )
    return value[0], score[0]


def _particle_value(
    theta: tf.Tensor,
    observations: tf.Tensor,
    initial_noise: tf.Tensor,
    process_noise: tf.Tensor,
    design: tf.Tensor,
    *,
    epsilon: float = EPSILON,
    sinkhorn_steps: int = SINKHORN_STEPS,
    ridge: float = RIDGE,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    phi = theta[:STATE_DIM]
    q_scale = theta[STATE_DIM]
    r_scale = theta[STATE_DIM + 1]
    initial_std = q_scale / tf.sqrt(1.0 - tf.square(phi))
    observation_matrix = _observation_matrix()
    particles = initial_noise * initial_std[None, :]
    n = tf.shape(particles)[0]
    weights = tf.fill([n], 1.0 / tf.cast(n, tf.float32))
    total = tf.constant(0.0, tf.float32)
    max_row = tf.constant(0.0, tf.float32)
    max_col = tf.constant(0.0, tf.float32)
    max_mean_cov = tf.constant(0.0, tf.float32)
    def body(
        index: tf.Tensor,
        particles_: tf.Tensor,
        weights_: tf.Tensor,
        total_: tf.Tensor,
        max_mean_cov_: tf.Tensor,
        max_row_: tf.Tensor,
        max_col_: tf.Tensor,
    ):
        particles_ = particles_ * phi[None, :] + q_scale * process_noise[index]
        prediction = particles_ @ tf.transpose(observation_matrix)
        innovation = observations[index][None, :] - prediction
        log_likelihood = -0.5 * (
            tf.reduce_sum(tf.square(innovation / r_scale), axis=1)
            + tf.cast(OBSERVATION_DIM, tf.float32) * 2.0 * tf.math.log(r_scale)
            + tf.cast(OBSERVATION_DIM, tf.float32) * tf.constant(LOG_TWO_PI, tf.float32)
        )
        log_weights = tf.math.log(weights_) + log_likelihood
        increment = tf.reduce_logsumexp(log_weights)
        weights_ = tf.exp(log_weights - increment)
        particles_, mean_cov_residual, marginal_residual = _restore_cloud(
            particles_,
            weights_,
            _design_at(design, index),
            epsilon=epsilon,
            sinkhorn_steps=sinkhorn_steps,
            ridge=ridge,
        )
        # Contract E reset returns an equal-weight cloud for the next step.
        weights_ = tf.fill(
            [tf.shape(weights_)[0]],
            tf.constant(1.0, tf.float32) / tf.cast(tf.shape(weights_)[0], tf.float32),
        )
        return (
            index + 1,
            particles_,
            weights_,
            total_ + increment,
            tf.maximum(max_mean_cov_, tf.reduce_max(mean_cov_residual)),
            tf.maximum(max_row_, marginal_residual[0]),
            tf.maximum(max_col_, marginal_residual[1]),
        )

    _, particles, weights, total, max_mean_cov, max_row, max_col = tf.while_loop(
        lambda index, *_: index < tf.shape(observations)[0],
        body,
        (
            tf.constant(0, tf.int32),
            particles,
            weights,
            total,
            max_mean_cov,
            max_row,
            max_col,
        ),
        maximum_iterations=HORIZONS[-1],
    )
    del particles, weights
    return total, max_mean_cov, max_row, max_col


def _particle_value_score_recursive(
    theta: tf.Tensor,
    observations: tf.Tensor,
    initial_noise: tf.Tensor,
    process_noise: tf.Tensor,
    design: tf.Tensor,
    *,
    epsilon: float = EPSILON,
    sinkhorn_steps: int = SINKHORN_STEPS,
    ridge: float = RIDGE,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Finite filter value and compact all-parameter forward sensitivity."""
    phi = theta[:STATE_DIM]
    q_scale = theta[STATE_DIM]
    r_scale = theta[STATE_DIM + 1]
    parameter_eye = tf.eye(PARAMETER_DIM, dtype=tf.float32)
    phi_tangent = parameter_eye[:STATE_DIM, :]
    q_tangent = parameter_eye[STATE_DIM, :]
    r_tangent = parameter_eye[STATE_DIM + 1, :]
    stationary_root = tf.sqrt(1.0 - tf.square(phi))
    initial_std = q_scale / stationary_root
    initial_std_tangent = (
        q_scale
        * phi[:, None]
        * phi_tangent
        / tf.pow(1.0 - tf.square(phi), 1.5)[:, None]
        + q_tangent[None, :] / stationary_root[:, None]
    )
    particles = initial_noise * initial_std[None, :]
    particle_tangent = initial_noise[:, :, None] * initial_std_tangent[None, :, :]
    n = tf.shape(particles)[0]
    weights = tf.fill([n], 1.0 / tf.cast(n, tf.float32))
    weight_tangent = tf.zeros([n, PARAMETER_DIM], tf.float32)
    total = tf.constant(0.0, tf.float32)
    score = tf.zeros([PARAMETER_DIM], tf.float32)
    max_row = tf.constant(0.0, tf.float32)
    max_col = tf.constant(0.0, tf.float32)
    max_mean_cov = tf.constant(0.0, tf.float32)
    observation_matrix = _observation_matrix()

    def body(
        index: tf.Tensor,
        particles_: tf.Tensor,
        weights_: tf.Tensor,
        particle_tangent_: tf.Tensor,
        weight_tangent_: tf.Tensor,
        total_: tf.Tensor,
        score_: tf.Tensor,
        max_mean_cov_: tf.Tensor,
        max_row_: tf.Tensor,
        max_col_: tf.Tensor,
    ):
        ancestors = particles_
        ancestor_tangent = particle_tangent_
        noise = process_noise[index]
        particles_ = ancestors * phi[None, :] + q_scale * noise
        particle_tangent_ = (
            ancestor_tangent * phi[None, :, None]
            + ancestors[:, :, None] * phi_tangent[None, :, :]
            + noise[:, :, None] * q_tangent[None, None, :]
        )
        prediction = particles_ @ tf.transpose(observation_matrix)
        prediction_tangent = tf.einsum(
            "od,ndp->nop", observation_matrix, particle_tangent_
        )
        innovation = observations[index][None, :] - prediction
        innovation_tangent = -prediction_tangent
        squared_norm = tf.reduce_sum(tf.square(innovation), axis=1)
        log_likelihood = -0.5 * (
            squared_norm / tf.square(r_scale)
            + tf.cast(OBSERVATION_DIM, tf.float32) * 2.0 * tf.math.log(r_scale)
            + tf.cast(OBSERVATION_DIM, tf.float32)
            * tf.constant(LOG_TWO_PI, tf.float32)
        )
        log_likelihood_tangent = -tf.reduce_sum(
            innovation[:, :, None] * innovation_tangent, axis=1
        ) / tf.square(r_scale)
        log_likelihood_tangent += (
            squared_norm / tf.pow(r_scale, 3)
            - tf.cast(OBSERVATION_DIM, tf.float32) / r_scale
        )[:, None] * r_tangent[None, :]
        log_weights = tf.math.log(weights_) + log_likelihood
        log_weight_tangent = (
            weight_tangent_ / weights_[:, None] + log_likelihood_tangent
        )
        increment = tf.reduce_logsumexp(log_weights)
        weights_ = tf.exp(log_weights - increment)
        increment_tangent = tf.reduce_sum(
            weights_[:, None] * log_weight_tangent, axis=0
        )
        weight_tangent_ = weights_[:, None] * (
            log_weight_tangent - increment_tangent[None, :]
        )
        (
            particles_,
            particle_tangent_,
            mean_cov_residual,
            marginal_residual,
        ) = _restore_cloud_jvp(
            particles_,
            weights_,
            particle_tangent_,
            weight_tangent_,
            _design_at(design, index),
            epsilon=epsilon,
            sinkhorn_steps=sinkhorn_steps,
            ridge=ridge,
        )
        weights_ = tf.fill(
            [tf.shape(weights_)[0]],
            tf.constant(1.0, tf.float32) / tf.cast(tf.shape(weights_)[0], tf.float32),
        )
        weight_tangent_ = tf.zeros_like(weight_tangent_)
        return (
            index + 1,
            particles_,
            weights_,
            particle_tangent_,
            weight_tangent_,
            total_ + increment,
            score_ + increment_tangent,
            tf.maximum(max_mean_cov_, tf.reduce_max(mean_cov_residual)),
            tf.maximum(max_row_, marginal_residual[0]),
            tf.maximum(max_col_, marginal_residual[1]),
        )

    (
        _,
        particles,
        weights,
        particle_tangent,
        weight_tangent,
        total,
        score,
        max_mean_cov,
        max_row,
        max_col,
    ) = tf.while_loop(
        lambda index, *_: index < tf.shape(observations)[0],
        body,
        (
            tf.constant(0, tf.int32),
            particles,
            weights,
            particle_tangent,
            weight_tangent,
            total,
            score,
            max_mean_cov,
            max_row,
            max_col,
        ),
        maximum_iterations=HORIZONS[-1],
    )
    del particles, weights, particle_tangent, weight_tangent
    return total, score, max_mean_cov, max_row, max_col


def _central_difference_score(
    value_function: Any,
    theta: tf.Tensor,
    observations: tf.Tensor,
    *,
    relative_step: float = FD_EPS,
    minimum_step: float = FD_MIN_STEP,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Differentiate a scalar value program without tracing its derivatives.

    The same callable is evaluated at both sides of every coordinate.  This is
    intentionally a finite-difference score of the implemented value program,
    not an assertion that the program is the exact model likelihood.
    """
    steps = tf.maximum(
        tf.cast(minimum_step, theta.dtype),
        tf.cast(relative_step, theta.dtype) * tf.abs(theta),
    )
    columns = []
    for index in range(theta.shape[0]):
        direction = tf.one_hot(index, theta.shape[0], dtype=theta.dtype)
        plus = value_function(theta + steps[index] * direction, observations)["value"]
        minus = value_function(theta - steps[index] * direction, observations)["value"]
        columns.append((plus - minus) / (tf.cast(2.0, theta.dtype) * steps[index]))
    return tf.stack(columns), steps


def _make_value_score(
    *,
    horizon: int,
    design: tf.Tensor,
    particle_seed: int,
    epsilon: float = EPSILON,
    sinkhorn_steps: int = SINKHORN_STEPS,
    ridge: float = RIDGE,
    jit_compile: bool,
) -> tuple[Any, Any]:
    initial_noise = tf.random.stateless_normal(
        [NUM_PARTICLES, STATE_DIM], seed=[particle_seed, horizon], dtype=tf.float32
    )
    process_noise = tf.random.stateless_normal(
        [horizon, NUM_PARTICLES, STATE_DIM],
        seed=[particle_seed, horizon + 100],
        dtype=tf.float32,
    )

    @tf.function(jit_compile=jit_compile)
    def value_only(theta: tf.Tensor, observations: tf.Tensor) -> dict[str, tf.Tensor]:
        value, mean_cov, row_residual, col_residual = _particle_value(
            theta,
            observations,
            initial_noise,
            process_noise,
            design,
            epsilon=epsilon,
            sinkhorn_steps=sinkhorn_steps,
            ridge=ridge,
        )
        return {
            "value": value,
            "mean_cov_residual": mean_cov,
            "row_residual": row_residual,
            "col_residual": col_residual,
        }

    @tf.function(jit_compile=jit_compile)
    def value_score(theta: tf.Tensor, observations: tf.Tensor) -> dict[str, tf.Tensor]:
        value, score, mean_cov, row_residual, col_residual = (
            _particle_value_score_recursive(
                theta,
                observations,
                initial_noise,
                process_noise,
                design,
                epsilon=epsilon,
                sinkhorn_steps=sinkhorn_steps,
                ridge=ridge,
            )
        )
        return {
            "value": value,
            "score": score,
            "mean_cov_residual": mean_cov,
            "row_residual": row_residual,
            "col_residual": col_residual,
        }

    return value_only, value_score


def _evaluate_method(
    method: str,
    horizon: int,
    observations: tf.Tensor,
    *,
    particle_seed: int,
    epsilon: float = EPSILON,
    sinkhorn_steps: int = SINKHORN_STEPS,
    ridge: float = RIDGE,
    jit_compile: bool,
    diagnostics: bool = False,
    theta_values: tuple[float, ...] = THETA_VALUES,
) -> dict[str, Any]:
    if method == "cubature":
        design = cubature_design(dim=STATE_DIM, num_particles=NUM_PARTICLES)
        design_metadata = {
            "rule": "cubature",
            "point_count": 2 * STATE_DIM,
            "replication": NUM_PARTICLES // (2 * STATE_DIM),
        }
    elif method == "genut":
        design, design_metadata = genut_gaussian_design(
            dim=STATE_DIM, num_particles=NUM_PARTICLES
        )
        design_metadata["replication"] = NUM_PARTICLES // (2 * STATE_DIM)
    elif method == "contract_e_gaussian":
        design, design_metadata = contract_e_gaussian_design(
            horizon=horizon,
            num_particles=NUM_PARTICLES,
            particle_seed=particle_seed,
        )
    else:
        raise ValueError(f"unknown method: {method}")
    value_only, value_score = _make_value_score(
        horizon=horizon,
        design=design,
        particle_seed=particle_seed,
        epsilon=epsilon,
        sinkhorn_steps=sinkhorn_steps,
        ridge=ridge,
        jit_compile=jit_compile,
    )
    theta = tf.constant(theta_values, tf.float32)
    started = time.perf_counter()
    first = value_score(theta, observations)
    second = value_score(theta, observations)
    wall_time = time.perf_counter() - started
    replay = all(
        _tensor_hash(first[name]) == _tensor_hash(second[name])
        for name in (
            "value",
            "score",
            "mean_cov_residual",
            "row_residual",
            "col_residual",
        )
    )
    kalman_value, kalman_score = _kalman_value_score(theta, observations)
    if diagnostics:
        fd_score, fd_steps = _central_difference_score(value_only, theta, observations)
        fd_abs_error = tf.abs(first["score"] - fd_score)
        fd_relative_error = fd_abs_error / tf.maximum(
            tf.maximum(tf.abs(first["score"]), tf.abs(fd_score)),
            tf.constant(1.0e-3, tf.float32),
        )
    else:
        fd_score = None
        fd_steps = None
        fd_abs_error = None
        fd_relative_error = None
    finite = all(
        bool(tf.reduce_all(tf.math.is_finite(first[name])).numpy())
        for name in ("value", "score", "mean_cov_residual", "row_residual", "col_residual")
    )
    candidate_value64 = tf.cast(first["value"], tf.float64)
    candidate_score64 = tf.cast(first["score"], tf.float64)
    value_error = candidate_value64 - kalman_value
    score_error = candidate_score64 - tf.cast(kalman_score, tf.float64)
    hmc_chain = tf.constant(
        [
            1.0 - theta_values[0] ** 2,
            1.0 - theta_values[1] ** 2,
            1.0 - theta_values[2] ** 2,
            theta_values[3],
            theta_values[4],
        ],
        tf.float64,
    )
    particle_hmc_score = candidate_score64 * hmc_chain
    kalman_hmc_score = tf.cast(kalman_score, tf.float64) * hmc_chain
    coordinate_relative_error = tf.concat(
        [
            tf.reshape(value_error / tf.abs(kalman_value), [1]),
            (particle_hmc_score - kalman_hmc_score) / tf.abs(kalman_hmc_score),
        ],
        axis=0,
    )
    return {
        "method": method,
        "horizon": horizon,
        "particle_seed": particle_seed,
        "design": design_metadata,
        "particle_value": float(first["value"].numpy()),
        "kalman_value": float(kalman_value.numpy()),
        "value_error": float(value_error.numpy()),
        "value_relative_error": float((value_error / tf.abs(kalman_value)).numpy()),
        "particle_score": first["score"].numpy().tolist(),
        "kalman_score": kalman_score.numpy().tolist(),
        "particle_hmc_score": particle_hmc_score.numpy().tolist(),
        "kalman_hmc_score": kalman_hmc_score.numpy().tolist(),
        "coordinate_relative_error": coordinate_relative_error.numpy().tolist(),
        "relative_error": coordinate_relative_error.numpy().tolist(),
        "relative_error_compatibility_alias": "coordinate_relative_error",
        "score_error": score_error.numpy().tolist(),
        "score_l2_error": float(tf.linalg.norm(score_error).numpy()),
        "reset_mean_cov_residual": float(first["mean_cov_residual"].numpy()),
        "sinkhorn_row_residual": float(first["row_residual"].numpy()),
        "sinkhorn_col_residual": float(first["col_residual"].numpy()),
        "finite": finite,
        "bitwise_replay": replay,
        "finite_difference_score": fd_score.numpy().tolist() if diagnostics else None,
        "finite_difference_abs_error": (
            fd_abs_error.numpy().tolist() if diagnostics else None
        ),
        "finite_difference_relative_error": (
            fd_relative_error.numpy().tolist() if diagnostics else None
        ),
        "finite_difference_max_abs_error": (
            float(tf.reduce_max(fd_abs_error).numpy()) if diagnostics else None
        ),
        "finite_difference_max_relative_error": (
            float(tf.reduce_max(fd_relative_error).numpy()) if diagnostics else None
        ),
        "finite_difference_directional": None,
        "autodiff_directional": None,
        "directional_score_abs_error": None,
        "score_route": SCORE_ROUTE_ID,
        "no_autodiff_score_route": True,
        "fd_audit_executed": diagnostics,
        "fd_steps": fd_steps.numpy().tolist() if diagnostics else None,
        "kalman_score_route": "analytic_recursive_kalman_score",
        "wall_time_seconds": wall_time,
        "controls": {
            "epsilon": epsilon,
            "sinkhorn_steps": sinkhorn_steps,
            "ridge": ridge,
        },
    }


def _interval(values: list[float]) -> dict[str, float]:
    if len(values) != len(PARTICLE_SEEDS):
        raise ValueError("comparison intervals require exactly 16 particle seeds")
    mean = statistics.mean(values)
    stddev = statistics.stdev(values)
    standard_error = stddev / math.sqrt(len(values))
    radius = CRITICAL_VALUE * standard_error
    return {
        "mean": mean,
        "standard_deviation": stddev,
        "standard_error": standard_error,
        "critical_value": CRITICAL_VALUE,
        "lower": mean - radius,
        "upper": mean + radius,
    }


def _screen(intervals: dict[str, dict[str, float]], hard_valid: bool) -> str:
    margins = (VALUE_MARGIN,) + (SCORE_MARGIN,) * 5
    ordered = [intervals[label] for label in LABELS]
    if not hard_valid:
        return "screen_fail"
    if all(
        interval["lower"] >= -margin and interval["upper"] <= margin
        for interval, margin in zip(ordered, margins, strict=True)
    ):
        return "screen_pass"
    if any(
        interval["lower"] > margin or interval["upper"] < -margin
        for interval, margin in zip(ordered, margins, strict=True)
    ):
        return "screen_fail"
    return "inconclusive"


def run(output_root: Path, *, jit_compile: bool) -> dict[str, Any]:
    device = _configure_gpu(jit_compile=jit_compile)
    theta = tf.constant(THETA_VALUES, tf.float32)
    results: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for horizon in HORIZONS:
        observations = _lgssm_observations(theta, horizon)
        for method in ("cubature", "genut"):
            arm_results = [
                _evaluate_method(
                    method,
                    horizon,
                    observations,
                    particle_seed=particle_seed,
                    jit_compile=jit_compile,
                )
                for particle_seed in PARTICLE_SEEDS
            ]
            results.extend(arm_results)
            relative_rows = [item["relative_error"] for item in arm_results]
            relative_error_intervals = {
                label: _interval([row[index] for row in relative_rows])
                for index, label in enumerate(LABELS)
            }
            arm_hard_valid = all(
                item["finite"]
                and item["bitwise_replay"]
                and item["sinkhorn_row_residual"] < 5.0e-4
                and item["sinkhorn_col_residual"] < 5.0e-4
                and item["reset_mean_cov_residual"] < 5.0e-4
                for item in arm_results
            )
            summaries.append(
                {
                    "method": method,
                    "horizon": horizon,
                    "particle_seeds": list(PARTICLE_SEEDS),
                    "kalman_value": arm_results[0]["kalman_value"],
                    "kalman_score": arm_results[0]["kalman_score"],
                    "kalman_hmc_score": arm_results[0]["kalman_hmc_score"],
                    "relative_error_intervals": relative_error_intervals,
                    "screen": _screen(relative_error_intervals, arm_hard_valid),
                    "screen_margins": {
                        "value": VALUE_MARGIN,
                        "score": SCORE_MARGIN,
                    },
                    "hard_valid": arm_hard_valid,
                    "mean_particle_value": statistics.mean(
                        item["particle_value"] for item in arm_results
                    ),
                    "mean_particle_hmc_score": [
                        statistics.mean(
                            item["particle_hmc_score"][index]
                            for item in arm_results
                        )
                        for index in range(STATE_DIM + 2)
                    ],
                    "all_finite": all(item["finite"] for item in arm_results),
                    "all_bitwise_replay": all(
                        item["bitwise_replay"] for item in arm_results
                    ),
                }
            )
    memory = tf.config.experimental.get_memory_info("GPU:0")
    source_paths = (Path(__file__).relative_to(ROOT).as_posix(),)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "campaign_id": CAMPAIGN_ID,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": platform.node(),
        "git_commit": _git_commit(),
        "source_sha256": {path: _sha256(ROOT / path) for path in source_paths},
        "device": device,
        "tensorflow_version": tf.__version__,
        "configuration": {
            "state_dim": STATE_DIM,
            "num_particles": NUM_PARTICLES,
            "horizons": list(HORIZONS),
            "theta": list(THETA_VALUES),
            "dataset_seed": DATASET_SEED,
            "particle_seeds": list(PARTICLE_SEEDS),
            "observation_dim": OBSERVATION_DIM,
            "observation_matrix": [list(row) for row in OBSERVATION_MATRIX_VALUES],
            "comparison_metric": (
                "previous_lgssm_hmc_relative_error_simultaneous_ci"
            ),
            "comparison_labels": list(LABELS),
            "simultaneous_critical_value": CRITICAL_VALUE,
            "value_margin": VALUE_MARGIN,
            "hmc_score_margin": SCORE_MARGIN,
            "epsilon": EPSILON,
            "sinkhorn_steps": SINKHORN_STEPS,
            "ridge": RIDGE,
            "score_route": SCORE_ROUTE_ID,
            "no_autodiff_score_route": True,
            "finite_difference_runtime_score": False,
            "fd_audit_executed": False,
            "fd_audit_relative_step_when_requested": FD_EPS,
            "dtype": "float32",
            "tf32_mode": "enabled",
            "jit_compile": jit_compile,
        },
        "results": results,
        "summaries": summaries,
        "comparison_screens": {
            f"T{item['horizon']}_{item['method']}": item["screen"]
            for item in summaries
        },
        "gpu_allocator": {key: int(value) for key, value in memory.items()},
        "hard_valid": all(
            item["finite"]
            and item["bitwise_replay"]
            and item["score_route"] == SCORE_ROUTE_ID
            and item["no_autodiff_score_route"]
            and not item["fd_audit_executed"]
            and item["sinkhorn_row_residual"] < 5.0e-4
            and item["sinkhorn_col_residual"] < 5.0e-4
            and item["reset_mean_cov_residual"] < 5.0e-4
            for item in results
        ),
        "nonclaims": [
            "16-seed descriptive comparison only",
            "no exact filtering likelihood or score claim",
            "no statistical ranking or superiority claim",
            "Gaussian GenUT moments reduce to the cubature design here",
            "no nonlinear-model or NAWM claim",
        ],
    }
    output_root.mkdir(parents=True, exist_ok=False)
    (output_root / "result.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# Float32/TF32 LGSSM Cubature and GenUT Diagnostic",
        "",
        f"- hard_valid: {payload['hard_valid']}",
        f"- dtype: {payload['configuration']['dtype']}",
        f"- tf32_mode: {payload['configuration']['tf32_mode']}",
        f"- GPU allocator: {payload['gpu_allocator']}",
        "",
        "Comparison metric: previous Contract E six-coordinate HMC-relative-error metric.",
        "",
        "| T | Method | Value mean rel. error | Value simultaneous 95% CI | q_scale mean rel. error | q_scale simultaneous 95% CI |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for item in summaries:
        value_interval = item["relative_error_intervals"]["value"]
        q_interval = item["relative_error_intervals"]["q_scale"]
        lines.append(
            f"| {item['horizon']} | {item['method']} | "
            f"{value_interval['mean']:.3e} | "
            f"[{value_interval['lower']:.3e}, {value_interval['upper']:.3e}] | "
            f"{q_interval['mean']:.3e} | "
            f"[{q_interval['lower']:.3e}, {q_interval['upper']:.3e}] |"
        )
    lines += [
        "",
        "The JSON retains per-seed raw physical/HMC scores, raw score L2, and",
        "finite-difference diagnostics as secondary metrics.",
        "",
        "GenUT uses Gaussian moments s=0, k=3; its central weight is zero, so",
        "the positive equal-weight realization omits the zero-mass center and",
        "equals the six-point cubature design.",
        "",
        "This is descriptive feasibility evidence, not a correctness or",
        "superiority claim for exact filtering.",
    ]
    (output_root / "result.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--no-jit-compile",
        action="store_true",
        help="Use GPU TensorFlow graph mode without XLA; explicit diagnostic escape hatch.",
    )
    args = parser.parse_args()
    payload = run(args.output_root.resolve(), jit_compile=not args.no_jit_compile)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
