"""Contract E--Chol candidate for the latent pre-clipping SIR target.

This module exercises the actual model-agnostic Contract E--Chol streaming
reset.  It is candidate-only: no repository route identity or admission is
issued here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import tensorflow as tf

from bayesfilter.highdim.ledh_contract_e_streaming_tf import (
    _contract_e_streaming_forward_core,
    _contract_e_streaming_jvp_core,
)
from bayesfilter.highdim.sir_latent_preclip_tf import (
    LatentPreclipSIRSSM,
    latent_preclip_two_node_spatial_sir_model,
    latent_preclip_zhao_cui_sir_austria_model,
)
from bayesfilter.highdim.transport_chunk_policy import (
    select_transport_chunks,
    validate_transport_chunks,
)


DTYPE = tf.float64
PARAMETER_COUNT = 3
PARAMETER_NAMES = (
    "log_kappa_scale",
    "log_nu_scale",
    "log_obs_noise_scale",
)
CANDIDATE_ROUTE_ID = "contract_e_chol_latent_preclip_sir_candidate_v1"
CANDIDATE_STATUS = "candidate_not_canonical_not_admitted"
CANONICAL_ROUTE_SPECIFICATION_ID = "contract_e_chol_latent_preclip_sir_austria_v1"
TWO_NODE_ROUTE_SPECIFICATION_ID = "contract_e_chol_latent_preclip_sir_two_node_v1"
CANONICAL_ANNEALING_STEPS = 20
CANONICAL_BALANCE_STEPS = 100
# Compatibility alias for callers that report the annealing warm-start budget.
CANONICAL_STEPS = CANONICAL_ANNEALING_STEPS
_LOG_TWO_PI = tf.math.log(tf.constant(6.283185307179586476925286766559, DTYPE))
_EPSILON0_FLOOR = tf.constant(1.0e-6, DTYPE)


@dataclass(frozen=True)
class LatentSIRStaticSpec:
    state_dimension: int
    observation_dimension: int
    compartments: int
    initial_mean: tf.Tensor
    initial_covariance: tf.Tensor
    process_covariance: tf.Tensor
    base_observation_covariance: tf.Tensor
    base_kappa: tf.Tensor
    base_nu: tf.Tensor
    adjacency: tf.Tensor
    neighbor_degree: tf.Tensor
    step: tf.Tensor
    substeps: int
    zhao_cui_rk4_variant: bool


def static_spec_from_model(model: LatentPreclipSIRSSM) -> LatentSIRStaticSpec:
    base = model.physical_model.base_model
    return LatentSIRStaticSpec(
        state_dimension=model.state_dim(),
        observation_dimension=model.observation_dim(),
        compartments=model.observation_dim(),
        initial_mean=tf.convert_to_tensor(base.initial_mean, DTYPE),
        initial_covariance=tf.convert_to_tensor(base.initial_covariance, DTYPE),
        process_covariance=tf.convert_to_tensor(base.process_covariance, DTYPE),
        base_observation_covariance=tf.convert_to_tensor(
            base.observation_covariance, DTYPE
        ),
        base_kappa=tf.convert_to_tensor(base.kappa, DTYPE),
        base_nu=tf.convert_to_tensor(base.nu, DTYPE),
        adjacency=tf.convert_to_tensor(base._adjacency_matrix, DTYPE),
        neighbor_degree=tf.convert_to_tensor(base._neighbor_degree, DTYPE),
        step=tf.convert_to_tensor(
            base.delta / tf.cast(base._rk4_substeps, DTYPE), DTYPE
        ),
        substeps=int(base._rk4_substeps),
        zhao_cui_rk4_variant=base.rk4_variant == "zhao_cui_sir_step",
    )


def _as_prepared_tensors(
    prepared: Mapping[str, Any], spec: LatentSIRStaticSpec
) -> dict[str, tf.Tensor]:
    required = (
        "observations",
        "initial_noise",
        "transition_noise",
        "fixed_reset_mask",
        "residual_design",
        "prepared_ridge",
        "epsilon",
        "scaling",
    )
    missing = [name for name in required if name not in prepared]
    if missing:
        raise ValueError(f"missing prepared latent-SIR inputs: {missing}")
    tensors = {
        "observations": tf.convert_to_tensor(prepared["observations"], DTYPE),
        "initial_noise": tf.convert_to_tensor(prepared["initial_noise"], DTYPE),
        "transition_noise": tf.convert_to_tensor(prepared["transition_noise"], DTYPE),
        "fixed_reset_mask": tf.convert_to_tensor(prepared["fixed_reset_mask"], tf.bool),
        "residual_design": tf.convert_to_tensor(prepared["residual_design"], DTYPE),
        "prepared_ridge": tf.convert_to_tensor(prepared["prepared_ridge"], DTYPE),
        "epsilon": tf.convert_to_tensor(prepared["epsilon"], DTYPE),
        "scaling": tf.convert_to_tensor(prepared["scaling"], DTYPE),
    }
    observations = tensors["observations"]
    initial_noise = tensors["initial_noise"]
    if observations.shape.rank != 2 or observations.shape[1] != spec.observation_dimension:
        raise ValueError("observations have the wrong latent-SIR shape")
    if initial_noise.shape.rank != 3 or initial_noise.shape[2] != spec.state_dimension:
        raise ValueError("initial_noise has the wrong latent-SIR shape")
    batch_size = initial_noise.shape[0]
    particle_count = initial_noise.shape[1]
    observation_count = observations.shape[0]
    if batch_size is None or particle_count is None or observation_count is None:
        raise ValueError("latent-SIR candidate requires static batch, particle, and time")
    expected_transition = (
        batch_size,
        max(0, observation_count - 1),
        particle_count,
        spec.state_dimension,
    )
    if tensors["transition_noise"].shape != expected_transition:
        raise ValueError("transition_noise has the wrong latent-SIR shape")
    if tensors["fixed_reset_mask"].shape != (batch_size, observation_count):
        raise ValueError("fixed_reset_mask has the wrong latent-SIR shape")
    if tensors["residual_design"].shape != (
        batch_size,
        observation_count,
        particle_count,
        spec.state_dimension,
    ):
        raise ValueError("residual_design has the wrong latent-SIR shape")
    if tensors["prepared_ridge"].shape != (batch_size, observation_count):
        raise ValueError("prepared_ridge has the wrong latent-SIR shape")
    if tensors["epsilon"].shape.rank != 0 or tensors["scaling"].shape.rank != 0:
        raise ValueError("epsilon and scaling must be scalar transport settings")
    return tensors


def _components(theta: tf.Tensor, spec: LatentSIRStaticSpec, batch_size: int):
    theta = tf.reshape(tf.convert_to_tensor(theta, DTYPE), [PARAMETER_COUNT])
    kappa = spec.base_kappa * tf.exp(theta[0])
    nu = spec.base_nu * tf.exp(theta[1])
    observation_covariance_single = spec.base_observation_covariance * tf.exp(
        2.0 * theta[2]
    )
    zero_cov = tf.zeros_like(observation_covariance_single)
    observation_covariance_tangent_single = tf.stack(
        [zero_cov, zero_cov, 2.0 * observation_covariance_single], axis=-1
    )
    zero_vector = tf.zeros_like(kappa)
    return {
        "theta": theta,
        "kappa": kappa,
        "nu": nu,
        "d_kappa": tf.stack([kappa, zero_vector, zero_vector], axis=-1),
        "d_nu": tf.stack([zero_vector, nu, zero_vector], axis=-1),
        "initial_covariance": tf.tile(
            spec.initial_covariance[None, :, :], [batch_size, 1, 1]
        ),
        "process_covariance": tf.tile(
            spec.process_covariance[None, :, :], [batch_size, 1, 1]
        ),
        "observation_covariance": tf.tile(
            observation_covariance_single[None, :, :], [batch_size, 1, 1]
        ),
        "observation_covariance_tangent": tf.tile(
            observation_covariance_tangent_single[None, :, :, :],
            [batch_size, 1, 1, 1],
        ),
    }


def _physical_state_and_tangent(
    latent: tf.Tensor, latent_tangent: tf.Tensor, *, time_index: tf.Tensor | int
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    susceptible = latent[:, :, 0::2]
    infectious = latent[:, :, 1::2]
    active = susceptible > 0.0
    clipped_physical = tf.reshape(
        tf.stack([tf.maximum(susceptible, 0.0), infectious], axis=3),
        tf.shape(latent),
    )
    susceptible_tangent = latent_tangent[:, :, 0::2, :] * tf.cast(
        active[:, :, :, None], DTYPE
    )
    infectious_tangent = latent_tangent[:, :, 1::2, :]
    clipped_tangent = tf.reshape(
        tf.stack([susceptible_tangent, infectious_tangent], axis=3),
        tf.shape(latent_tangent),
    )
    initial = tf.equal(tf.cast(time_index, tf.int32), 0)
    away_from_boundary = initial | tf.reduce_all(susceptible != 0.0, axis=[1, 2])
    return (
        tf.where(initial, latent, clipped_physical),
        tf.where(initial, latent_tangent, clipped_tangent),
        away_from_boundary,
    )


def _rhs_and_tangent(
    state: tf.Tensor,
    state_tangent: tf.Tensor,
    components: Mapping[str, tf.Tensor],
    spec: LatentSIRStaticSpec,
) -> tuple[tf.Tensor, tf.Tensor]:
    susceptible = state[:, :, 0::2]
    infectious = state[:, :, 1::2]
    d_susceptible = state_tangent[:, :, 0::2, :]
    d_infectious = state_tangent[:, :, 1::2, :]
    susceptible_neighbor = (
        tf.einsum("bnj,kj->bnk", susceptible, spec.adjacency)
        - susceptible * spec.neighbor_degree[None, None, :]
    )
    infectious_neighbor = (
        tf.einsum("bnj,kj->bnk", infectious, spec.adjacency)
        - infectious * spec.neighbor_degree[None, None, :]
    )
    d_susceptible_neighbor = (
        tf.einsum("bnjp,kj->bnkp", d_susceptible, spec.adjacency)
        - d_susceptible * spec.neighbor_degree[None, None, :, None]
    )
    d_infectious_neighbor = (
        tf.einsum("bnjp,kj->bnkp", d_infectious, spec.adjacency)
        - d_infectious * spec.neighbor_degree[None, None, :, None]
    )
    kappa = components["kappa"]
    nu = components["nu"]
    infection = kappa[None, None, :] * susceptible * infectious
    infection_tangent = (
        components["d_kappa"][None, None, :, :]
        * susceptible[:, :, :, None]
        * infectious[:, :, :, None]
        + kappa[None, None, :, None]
        * (
            d_susceptible * infectious[:, :, :, None]
            + susceptible[:, :, :, None] * d_infectious
        )
    )
    rhs_s = -infection + 0.5 * susceptible_neighbor
    rhs_i = infection - nu[None, None, :] * infectious + 0.5 * infectious_neighbor
    d_rhs_s = -infection_tangent + 0.5 * d_susceptible_neighbor
    d_rhs_i = (
        infection_tangent
        - components["d_nu"][None, None, :, :] * infectious[:, :, :, None]
        - nu[None, None, :, None] * d_infectious
        + 0.5 * d_infectious_neighbor
    )
    return (
        tf.reshape(tf.stack([rhs_s, rhs_i], axis=3), tf.shape(state)),
        tf.reshape(
            tf.stack([d_rhs_s, d_rhs_i], axis=3), tf.shape(state_tangent)
        ),
    )


def _transition_mean_and_tangent(
    state: tf.Tensor,
    state_tangent: tf.Tensor,
    components: Mapping[str, tf.Tensor],
    spec: LatentSIRStaticSpec,
) -> tuple[tf.Tensor, tf.Tensor]:
    h = spec.step
    k4_factor = tf.constant(0.5 if spec.zhao_cui_rk4_variant else 1.0, DTYPE)

    def body(
        index: tf.Tensor, current: tf.Tensor, current_tangent: tf.Tensor
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        k1, d_k1 = _rhs_and_tangent(current, current_tangent, components, spec)
        state2 = current + 0.5 * h * k1
        tangent2 = current_tangent + 0.5 * h * d_k1
        k2, d_k2 = _rhs_and_tangent(state2, tangent2, components, spec)
        state3 = current + 0.5 * h * k2
        tangent3 = current_tangent + 0.5 * h * d_k2
        k3, d_k3 = _rhs_and_tangent(state3, tangent3, components, spec)
        state4 = current + k4_factor * h * k3
        tangent4 = current_tangent + k4_factor * h * d_k3
        k4, d_k4 = _rhs_and_tangent(state4, tangent4, components, spec)
        next_state = current + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        next_tangent = current_tangent + (h / 6.0) * (
            d_k1 + 2.0 * d_k2 + 2.0 * d_k3 + d_k4
        )
        return index + 1, next_state, next_tangent

    _, state, state_tangent = tf.while_loop(
        lambda index, _state, _tangent: index < spec.substeps,
        body,
        loop_vars=(tf.constant(0, tf.int32), state, state_tangent),
        maximum_iterations=spec.substeps,
    )
    return state, state_tangent


def _cholesky_jvp(chol: tf.Tensor, matrix_tangent: tf.Tensor) -> tf.Tensor:
    dimension = tf.shape(chol)[-1]
    batch_size = tf.shape(chol)[0]
    identity = tf.eye(dimension, batch_shape=[batch_size], dtype=chol.dtype)
    chol_inverse = tf.linalg.triangular_solve(chol, identity)
    columns = []
    for index in range(PARAMETER_COUNT):
        tangent = matrix_tangent[..., index]
        right_solved = tf.linalg.matmul(tangent, chol_inverse, transpose_b=True)
        inner = tf.linalg.matmul(chol_inverse, right_solved)
        lower = tf.linalg.band_part(inner, -1, 0)
        phi = tf.linalg.set_diag(lower, 0.5 * tf.linalg.diag_part(lower))
        columns.append(tf.linalg.matmul(chol, phi))
    return tf.stack(columns, axis=-1)


def _inverse_jvp(inverse: tf.Tensor, matrix_tangent: tf.Tensor) -> tf.Tensor:
    return tf.stack(
        [
            -inverse @ matrix_tangent[..., index] @ inverse
            for index in range(PARAMETER_COUNT)
        ],
        axis=-1,
    )


def _flow_forward_and_jvp(
    prior_mean: tf.Tensor,
    pre_flow: tf.Tensor,
    observation: tf.Tensor,
    prior_covariance: tf.Tensor,
    observation_covariance: tf.Tensor,
    prior_mean_tangent: tf.Tensor,
    pre_flow_tangent: tf.Tensor,
    observation_covariance_tangent: tf.Tensor,
    spec: LatentSIRStaticSpec,
) -> dict[str, tf.Tensor]:
    batch_size = tf.shape(prior_mean)[0]
    state_identity = tf.eye(spec.state_dimension, batch_shape=[batch_size], dtype=DTYPE)
    obs_identity = tf.eye(spec.observation_dimension, batch_shape=[batch_size], dtype=DTYPE)
    h = tf.one_hot(
        tf.range(1, spec.state_dimension, 2, dtype=tf.int32),
        depth=spec.state_dimension,
        dtype=DTYPE,
    )
    prior_chol = tf.linalg.cholesky(prior_covariance)
    observation_chol = tf.linalg.cholesky(observation_covariance)
    prior_precision = tf.linalg.cholesky_solve(prior_chol, state_identity)
    observation_precision = tf.linalg.cholesky_solve(observation_chol, obs_identity)
    post_precision = prior_precision + tf.einsum(
        "od,boq,qe->bde", h, observation_precision, h
    )
    post_covariance = tf.linalg.inv(post_precision)
    post_chol = tf.linalg.cholesky(post_covariance)
    observation_info = tf.einsum(
        "od,boq,q->bd", h, observation_precision, observation
    )
    info = tf.einsum("bij,bnj->bni", prior_precision, prior_mean) + observation_info[:, None, :]
    post_mean = tf.einsum("bij,bnj->bni", post_covariance, info)
    prior_chol_inverse = tf.linalg.triangular_solve(prior_chol, state_identity)
    affine = post_chol @ prior_chol_inverse
    particles = post_mean + tf.einsum("bnj,bij->bni", pre_flow - prior_mean, affine)
    forward_log_abs_det = (
        tf.reduce_sum(tf.math.log(tf.linalg.diag_part(post_chol)), axis=1)
        - tf.reduce_sum(tf.math.log(tf.linalg.diag_part(prior_chol)), axis=1)
    )

    zero_prior_tangent = tf.zeros(
        [batch_size, spec.state_dimension, spec.state_dimension, PARAMETER_COUNT], DTYPE
    )
    d_prior_precision = _inverse_jvp(prior_precision, zero_prior_tangent)
    d_observation_precision = _inverse_jvp(
        observation_precision, observation_covariance_tangent
    )
    d_post_precision = d_prior_precision + tf.einsum(
        "od,boqp,qe->bdep", h, d_observation_precision, h
    )
    d_post_covariance = _inverse_jvp(post_covariance, d_post_precision)
    d_post_chol = _cholesky_jvp(post_chol, d_post_covariance)
    d_observation_info = tf.einsum(
        "od,boqp,q->bdp", h, d_observation_precision, observation
    )
    d_info = (
        tf.einsum("bijp,bnj->bnip", d_prior_precision, prior_mean)
        + tf.einsum("bij,bnjp->bnip", prior_precision, prior_mean_tangent)
        + d_observation_info[:, None, :, :]
    )
    d_post_mean = (
        tf.einsum("bijp,bnj->bnip", d_post_covariance, info)
        + tf.einsum("bij,bnjp->bnip", post_covariance, d_info)
    )
    d_affine = tf.einsum("bijp,bjk->bikp", d_post_chol, prior_chol_inverse)
    delta = pre_flow - prior_mean
    d_delta = pre_flow_tangent - prior_mean_tangent
    particles_tangent = d_post_mean + (
        tf.einsum("bnj,bijp->bnip", delta, d_affine)
        + tf.einsum("bnjp,bij->bnip", d_delta, affine)
    )
    d_logdet_post = tf.reduce_sum(
        tf.linalg.diag_part(tf.transpose(d_post_chol, [0, 3, 1, 2]))
        / tf.linalg.diag_part(post_chol)[:, None, :],
        axis=2,
    )
    factor_diagonal = tf.concat(
        [tf.linalg.diag_part(prior_chol), tf.linalg.diag_part(observation_chol), tf.linalg.diag_part(post_chol)],
        axis=1,
    )
    valid = tf.reduce_all(tf.math.is_finite(factor_diagonal), axis=1) & tf.reduce_all(
        factor_diagonal > 0.0, axis=1
    )
    return {
        "particles": particles,
        "particles_tangent": particles_tangent,
        "forward_log_abs_det": forward_log_abs_det,
        "forward_log_abs_det_tangent": d_logdet_post,
        "valid_chart": valid & tf.reduce_all(tf.math.is_finite(particles), axis=[1, 2]),
    }


def _gaussian_density_and_jvp(
    residual: tf.Tensor,
    covariance: tf.Tensor,
    residual_tangent: tf.Tensor,
    covariance_tangent: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    inverse = tf.linalg.inv(covariance)
    solved = tf.einsum("bij,bnj->bni", inverse, residual)
    quadratic = tf.reduce_sum(residual * solved, axis=2)
    logdet = tf.linalg.logdet(covariance)
    dimension = tf.cast(tf.shape(residual)[2], DTYPE)
    value = -0.5 * (dimension * _LOG_TWO_PI + logdet[:, None] + quadratic)
    inverse_tangent = _inverse_jvp(inverse, covariance_tangent)
    quadratic_tangent = (
        2.0 * tf.reduce_sum(residual_tangent * solved[:, :, :, None], axis=2)
        + tf.einsum("bni,bijp,bnj->bnp", residual, inverse_tangent, residual)
    )
    logdet_tangent = tf.einsum("bij,bjip->bp", inverse, covariance_tangent)
    return value, -0.5 * (logdet_tangent[:, None, :] + quadratic_tangent)


def _normalize_and_jvp(logits: tf.Tensor, tangent: tf.Tensor):
    increment = tf.reduce_logsumexp(logits, axis=1)
    log_weights = logits - increment[:, None]
    weights = tf.exp(log_weights)
    increment_tangent = tf.reduce_sum(weights[:, :, None] * tangent, axis=1)
    log_weights_tangent = tangent - increment_tangent[:, None, :]
    return {
        "increment": increment,
        "increment_tangent": increment_tangent,
        "normalized_log_weights": log_weights,
        "normalized_log_weights_tangent": log_weights_tangent,
        "normalized_weights": weights,
        "normalized_weights_tangent": weights[:, :, None] * log_weights_tangent,
    }


def _geometry_and_jvp(particles: tf.Tensor, tangent: tf.Tensor, state_dimension: int):
    center = tf.reduce_mean(particles, axis=1, keepdims=True)
    center_tangent = tf.reduce_mean(tangent, axis=1, keepdims=True)
    centered = particles - center
    centered_tangent = tangent - center_tangent
    variance = tf.reduce_mean(tf.square(centered), axis=1)
    std = tf.sqrt(variance)
    variance_tangent = 2.0 * tf.reduce_mean(centered[:, :, :, None] * centered_tangent, axis=1)
    std_tangent = variance_tangent / (2.0 * std[:, :, None])
    diameter = tf.reduce_max(std, axis=1)
    mask = std == diameter[:, None]
    mask_weight = tf.cast(mask, DTYPE) / tf.reduce_sum(tf.cast(mask, DTYPE), axis=1)[:, None]
    diameter_tangent = tf.reduce_sum(mask_weight[:, :, None] * std_tangent, axis=1)
    scale = tf.sqrt(tf.cast(state_dimension, DTYPE)) * diameter
    scale_tangent = tf.sqrt(tf.cast(state_dimension, DTYPE)) * diameter_tangent
    scaled = centered / scale[:, None, None]
    scaled_tangent = (
        centered_tangent / scale[:, None, None, None]
        - centered[:, :, :, None]
        * scale_tangent[:, None, None, :]
        / tf.square(scale)[:, None, None, None]
    )
    maximum = tf.reduce_max(scaled, axis=[1, 2])
    minimum = tf.reduce_min(scaled, axis=[1, 2])
    max_mask = scaled == maximum[:, None, None]
    min_mask = scaled == minimum[:, None, None]
    max_weight = tf.cast(max_mask, DTYPE) / tf.reduce_sum(tf.cast(max_mask, DTYPE), axis=[1, 2])[:, None, None]
    min_weight = tf.cast(min_mask, DTYPE) / tf.reduce_sum(tf.cast(min_mask, DTYPE), axis=[1, 2])[:, None, None]
    max_tangent = tf.reduce_sum(max_weight[:, :, :, None] * scaled_tangent, axis=[1, 2])
    min_tangent = tf.reduce_sum(min_weight[:, :, :, None] * scaled_tangent, axis=[1, 2])
    coordinate_range = maximum - minimum
    epsilon0 = tf.maximum(tf.square(coordinate_range), _EPSILON0_FLOOR)
    epsilon0_tangent = tf.where(
        (tf.square(coordinate_range) >= _EPSILON0_FLOOR)[:, None],
        2.0 * coordinate_range[:, None] * (max_tangent - min_tangent),
        tf.zeros_like(max_tangent),
    )
    return {
        "scaled_geometry": scaled,
        "scaled_geometry_tangent": scaled_tangent,
        "epsilon0": epsilon0,
        "epsilon0_tangent": epsilon0_tangent,
        "valid_chart": tf.reduce_all(tf.math.is_finite(scaled), axis=[1, 2]) & (diameter > 0.0),
    }


def latent_sir_contract_e_value_and_score_core(
    theta: tf.Tensor,
    prepared: Mapping[str, tf.Tensor],
    spec: LatentSIRStaticSpec,
    *,
    steps: int,
    balance_steps: int = 0,
    row_chunk_size: int,
    col_chunk_size: int,
) -> dict[str, tf.Tensor | str]:
    initial_noise = prepared["initial_noise"]
    batch_size = int(initial_noise.shape[0])
    validate_transport_chunks(
        int(initial_noise.shape[1]),
        row_chunk_size=row_chunk_size,
        col_chunk_size=col_chunk_size,
    )
    particle_count = int(initial_noise.shape[1])
    observation_count = tf.shape(prepared["observations"], out_type=tf.int32)[0]
    components = _components(theta, spec, batch_size)
    initial_chol = tf.linalg.cholesky(spec.initial_covariance)
    particles = spec.initial_mean[None, None, :] + tf.einsum(
        "bnj,ij->bni", initial_noise, initial_chol
    )
    particles_tangent = tf.zeros(
        [batch_size, particle_count, spec.state_dimension, PARAMETER_COUNT], DTYPE
    )
    uniform_log_weight = -tf.math.log(tf.cast(particle_count, DTYPE))
    log_weights = tf.fill([batch_size, particle_count], uniform_log_weight)
    log_weights_tangent = tf.zeros(
        [batch_size, particle_count, PARAMETER_COUNT], DTYPE
    )
    total = tf.zeros([batch_size], DTYPE)
    total_score = tf.zeros([batch_size, PARAMETER_COUNT], DTYPE)
    valid = tf.ones([batch_size], tf.bool)
    increment_history = tf.TensorArray(
        DTYPE, size=observation_count, element_shape=[batch_size]
    )
    increment_score_history = tf.TensorArray(
        DTYPE, size=observation_count, element_shape=[batch_size, PARAMETER_COUNT]
    )
    clip_boundary_history = tf.TensorArray(
        tf.bool, size=observation_count, element_shape=[batch_size]
    )
    reset_valid_history = tf.TensorArray(
        tf.bool, size=observation_count, element_shape=[batch_size]
    )
    minimum_mass_history = tf.TensorArray(
        DTYPE, size=observation_count, element_shape=[batch_size]
    )
    flow_valid_history = tf.TensorArray(
        tf.bool, size=observation_count, element_shape=[batch_size]
    )
    geometry_valid_history = tf.TensorArray(
        tf.bool, size=observation_count, element_shape=[batch_size]
    )
    quotient_valid_history = tf.TensorArray(
        tf.bool, size=observation_count, element_shape=[batch_size]
    )
    reset_finite_history = tf.TensorArray(
        tf.bool, size=observation_count, element_shape=[batch_size]
    )
    reset_factor_positive_history = tf.TensorArray(
        tf.bool, size=observation_count, element_shape=[batch_size]
    )
    covariance_gap_eigenvalue_history = tf.TensorArray(
        DTYPE,
        size=observation_count,
        element_shape=[batch_size, spec.state_dimension],
    )
    quotient_row_residual_history = tf.TensorArray(
        DTYPE, size=observation_count, element_shape=[batch_size]
    )
    quotient_column_residual_history = tf.TensorArray(
        DTYPE, size=observation_count, element_shape=[batch_size]
    )
    quotient_column_residual_scale_history = tf.TensorArray(
        DTYPE, size=observation_count, element_shape=[batch_size]
    )
    quotient_post_column_residual_history = tf.TensorArray(
        DTYPE, size=observation_count, element_shape=[batch_size]
    )

    zero_covariance_tangent = tf.zeros(
        [batch_size, spec.state_dimension, spec.state_dimension, PARAMETER_COUNT], DTYPE
    )
    process_chol = tf.linalg.cholesky(spec.process_covariance)

    def body(
        time_index: tf.Tensor,
        particles: tf.Tensor,
        particles_tangent: tf.Tensor,
        log_weights: tf.Tensor,
        log_weights_tangent: tf.Tensor,
        total: tf.Tensor,
        total_score: tf.Tensor,
        valid: tf.Tensor,
        increment_history: tf.TensorArray,
        increment_score_history: tf.TensorArray,
        clip_boundary_history: tf.TensorArray,
        reset_valid_history: tf.TensorArray,
        minimum_mass_history: tf.TensorArray,
        flow_valid_history: tf.TensorArray,
        geometry_valid_history: tf.TensorArray,
        quotient_valid_history: tf.TensorArray,
        reset_finite_history: tf.TensorArray,
        reset_factor_positive_history: tf.TensorArray,
        covariance_gap_eigenvalue_history: tf.TensorArray,
        quotient_row_residual_history: tf.TensorArray,
        quotient_column_residual_history: tf.TensorArray,
        quotient_column_residual_scale_history: tf.TensorArray,
        quotient_post_column_residual_history: tf.TensorArray,
    ):
        def initial_prior():
            prior_mean = tf.broadcast_to(spec.initial_mean[None, None, :], tf.shape(particles))
            prior_mean_tangent = tf.zeros_like(particles_tangent)
            pre_flow = particles
            pre_flow_tangent = particles_tangent
            prior_covariance = components["initial_covariance"]
            return (
                prior_mean,
                prior_mean_tangent,
                pre_flow,
                pre_flow_tangent,
                prior_covariance,
                tf.ones([batch_size], tf.bool),
            )

        def transitioned_prior():
            physical, physical_tangent, away = _physical_state_and_tangent(
                particles, particles_tangent, time_index=time_index - 1
            )
            prior_mean, prior_mean_tangent = _transition_mean_and_tangent(
                physical, physical_tangent, components, spec
            )
            noise = prepared["transition_noise"][:, time_index - 1, :, :]
            pre_flow = prior_mean + tf.einsum("bnj,ij->bni", noise, process_chol)
            pre_flow_tangent = prior_mean_tangent
            prior_covariance = components["process_covariance"]
            return (
                prior_mean,
                prior_mean_tangent,
                pre_flow,
                pre_flow_tangent,
                prior_covariance,
                away,
            )

        (
            prior_mean,
            prior_mean_tangent,
            pre_flow,
            pre_flow_tangent,
            prior_covariance,
            away,
        ) = tf.cond(tf.equal(time_index, 0), initial_prior, transitioned_prior)
        observation = prepared["observations"][time_index]
        flow = _flow_forward_and_jvp(
            prior_mean,
            pre_flow,
            observation,
            prior_covariance,
            components["observation_covariance"],
            prior_mean_tangent,
            pre_flow_tangent,
            components["observation_covariance_tangent"],
            spec,
        )
        target_residual = flow["particles"] - prior_mean
        target_residual_tangent = flow["particles_tangent"] - prior_mean_tangent
        transition_value, transition_tangent = _gaussian_density_and_jvp(
            target_residual,
            prior_covariance,
            target_residual_tangent,
            zero_covariance_tangent,
        )
        proposal_value, proposal_tangent = _gaussian_density_and_jvp(
            pre_flow - prior_mean,
            prior_covariance,
            pre_flow_tangent - prior_mean_tangent,
            zero_covariance_tangent,
        )
        predicted_observation = flow["particles"][:, :, 1::2]
        predicted_observation_tangent = flow["particles_tangent"][:, :, 1::2, :]
        observation_value, observation_tangent = _gaussian_density_and_jvp(
            predicted_observation - observation[None, None, :],
            components["observation_covariance"],
            predicted_observation_tangent,
            components["observation_covariance_tangent"],
        )
        logits = (
            log_weights
            + transition_value
            + observation_value
            - proposal_value
            + flow["forward_log_abs_det"][:, None]
        )
        logits_tangent = (
            log_weights_tangent
            + transition_tangent
            + observation_tangent
            - proposal_tangent
            + flow["forward_log_abs_det_tangent"][:, None, :]
        )
        normalization = _normalize_and_jvp(logits, logits_tangent)
        total = total + normalization["increment"]
        total_score = total_score + normalization["increment_tangent"]
        geometry = _geometry_and_jvp(
            flow["particles"], flow["particles_tangent"], spec.state_dimension
        )
        reset = _contract_e_streaming_forward_core(
            geometry["scaled_geometry"],
            flow["particles"],
            normalization["normalized_log_weights"],
            normalization["normalized_weights"],
            prepared["residual_design"][:, time_index, :, :],
            prepared["prepared_ridge"][:, time_index],
            prepared["epsilon"],
            geometry["epsilon0"],
            prepared["scaling"],
            steps=steps,
            balance_steps=balance_steps,
            row_chunk_size=row_chunk_size,
            col_chunk_size=col_chunk_size,
        )
        reset_tangent = _contract_e_streaming_jvp_core(
            geometry["scaled_geometry"],
            flow["particles"],
            normalization["normalized_log_weights"],
            normalization["normalized_weights"],
            prepared["residual_design"][:, time_index, :, :],
            prepared["prepared_ridge"][:, time_index],
            geometry["scaled_geometry_tangent"],
            flow["particles_tangent"],
            normalization["normalized_log_weights_tangent"],
            normalization["normalized_weights_tangent"],
            tf.zeros(
                [batch_size, particle_count, spec.state_dimension, PARAMETER_COUNT], DTYPE
            ),
            tf.zeros([batch_size, PARAMETER_COUNT], DTYPE),
            geometry["epsilon0_tangent"],
            prepared["epsilon"],
            geometry["epsilon0"],
            prepared["scaling"],
            steps=steps,
            balance_steps=balance_steps,
            row_chunk_size=row_chunk_size,
            col_chunk_size=col_chunk_size,
        )
        marginal_valid = (
            reset["quotient"]["marginal_valid"]
            if balance_steps > 0
            else tf.ones([batch_size], tf.bool)
        )
        reset_valid = (
            reset["quotient"]["valid_chart"]
            & marginal_valid
            & reset["reset"]["finite"]
            & reset["reset"]["factor_diagonal_positive"]
        )
        active = prepared["fixed_reset_mask"][:, time_index]
        selected_particles = tf.where(
            active[:, None, None], reset["particles"], flow["particles"]
        )
        selected_particles_tangent = tf.where(
            active[:, None, None, None], reset_tangent["particles"], flow["particles_tangent"]
        )
        selected_log_weights = tf.where(
            active[:, None],
            tf.fill([batch_size, particle_count], uniform_log_weight),
            normalization["normalized_log_weights"],
        )
        selected_log_weights_tangent = tf.where(
            active[:, None, None],
            tf.zeros_like(normalization["normalized_log_weights_tangent"]),
            normalization["normalized_log_weights_tangent"],
        )
        time_valid = (
            away
            & flow["valid_chart"]
            & geometry["valid_chart"]
            & (~active | reset_valid)
        )
        valid = valid & time_valid
        nan = tf.constant(float("nan"), DTYPE)
        particles = tf.where(
            valid[:, None, None], selected_particles, tf.fill(tf.shape(selected_particles), nan)
        )
        particles_tangent = tf.where(
            valid[:, None, None, None],
            selected_particles_tangent,
            tf.fill(tf.shape(selected_particles_tangent), nan),
        )
        log_weights = tf.where(
            valid[:, None],
            selected_log_weights,
            tf.fill(tf.shape(selected_log_weights), nan),
        )
        log_weights_tangent = tf.where(
            valid[:, None, None],
            selected_log_weights_tangent,
            tf.fill(tf.shape(selected_log_weights_tangent), nan),
        )
        increment_history = increment_history.write(
            time_index, normalization["increment"]
        )
        increment_score_history = increment_score_history.write(
            time_index, normalization["increment_tangent"]
        )
        clip_boundary_history = clip_boundary_history.write(time_index, away)
        reset_valid_history = reset_valid_history.write(time_index, reset_valid)
        minimum_mass_history = minimum_mass_history.write(
            time_index, reset["quotient"]["minimum_mass"]
        )
        flow_valid_history = flow_valid_history.write(
            time_index, flow["valid_chart"]
        )
        geometry_valid_history = geometry_valid_history.write(
            time_index, geometry["valid_chart"]
        )
        quotient_valid_history = quotient_valid_history.write(
            time_index, reset["quotient"]["valid_chart"]
        )
        reset_finite_history = reset_finite_history.write(
            time_index, reset["reset"]["finite"]
        )
        reset_factor_positive_history = reset_factor_positive_history.write(
            time_index, reset["reset"]["factor_diagonal_positive"]
        )
        covariance_gap_eigenvalue_history = covariance_gap_eigenvalue_history.write(
            time_index, reset["reset"]["gap_eigenvalues"]
        )
        quotient_row_residual_history = quotient_row_residual_history.write(
            time_index, reset["quotient"]["row_residual_by_batch"]
        )
        quotient_column_residual_history = quotient_column_residual_history.write(
            time_index, reset["quotient"]["maximum_column_absolute_residual"]
        )
        quotient_column_residual_scale_history = (
            quotient_column_residual_scale_history.write(
                time_index, reset["quotient"]["column_residual_scale"]
            )
        )
        quotient_post_column_residual_history = (
            quotient_post_column_residual_history.write(
                time_index,
                reset["quotient"][
                    "maximum_post_quotient_column_absolute_residual"
                ],
            )
        )
        return (
            time_index + 1,
            particles,
            particles_tangent,
            log_weights,
            log_weights_tangent,
            total,
            total_score,
            valid,
            increment_history,
            increment_score_history,
            clip_boundary_history,
            reset_valid_history,
            minimum_mass_history,
            flow_valid_history,
            geometry_valid_history,
            quotient_valid_history,
            reset_finite_history,
            reset_factor_positive_history,
            covariance_gap_eigenvalue_history,
            quotient_row_residual_history,
            quotient_column_residual_history,
            quotient_column_residual_scale_history,
            quotient_post_column_residual_history,
        )

    (
        _,
        particles,
        particles_tangent,
        log_weights,
        log_weights_tangent,
        total,
        total_score,
        valid,
        increment_history,
        increment_score_history,
        clip_boundary_history,
        reset_valid_history,
        minimum_mass_history,
        flow_valid_history,
        geometry_valid_history,
        quotient_valid_history,
        reset_finite_history,
        reset_factor_positive_history,
        covariance_gap_eigenvalue_history,
        quotient_row_residual_history,
        quotient_column_residual_history,
        quotient_column_residual_scale_history,
        quotient_post_column_residual_history,
    ) = tf.while_loop(
        lambda time_index, *_state: time_index < observation_count,
        body,
        loop_vars=(
            tf.constant(0, tf.int32),
            particles,
            particles_tangent,
            log_weights,
            log_weights_tangent,
            total,
            total_score,
            valid,
            increment_history,
            increment_score_history,
            clip_boundary_history,
            reset_valid_history,
            minimum_mass_history,
            flow_valid_history,
            geometry_valid_history,
            quotient_valid_history,
            reset_finite_history,
            reset_factor_positive_history,
            covariance_gap_eigenvalue_history,
            quotient_row_residual_history,
            quotient_column_residual_history,
            quotient_column_residual_scale_history,
            quotient_post_column_residual_history,
        ),
        maximum_iterations=observation_count,
    )

    increment_history_tensor = tf.transpose(increment_history.stack(), [1, 0])
    increment_score_history_tensor = tf.transpose(
        increment_score_history.stack(), [1, 0, 2]
    )
    clip_boundary_history_tensor = tf.transpose(
        clip_boundary_history.stack(), [1, 0]
    )
    reset_valid_history_tensor = tf.transpose(reset_valid_history.stack(), [1, 0])
    minimum_mass_history_tensor = tf.transpose(
        minimum_mass_history.stack(), [1, 0]
    )
    flow_valid_history_tensor = tf.transpose(flow_valid_history.stack(), [1, 0])
    geometry_valid_history_tensor = tf.transpose(
        geometry_valid_history.stack(), [1, 0]
    )
    quotient_valid_history_tensor = tf.transpose(
        quotient_valid_history.stack(), [1, 0]
    )
    reset_finite_history_tensor = tf.transpose(
        reset_finite_history.stack(), [1, 0]
    )
    reset_factor_positive_history_tensor = tf.transpose(
        reset_factor_positive_history.stack(), [1, 0]
    )
    covariance_gap_eigenvalue_history_tensor = tf.transpose(
        covariance_gap_eigenvalue_history.stack(), [1, 0, 2]
    )
    quotient_row_residual_history_tensor = tf.transpose(
        quotient_row_residual_history.stack(), [1, 0]
    )
    quotient_column_residual_history_tensor = tf.transpose(
        quotient_column_residual_history.stack(), [1, 0]
    )
    quotient_column_residual_scale_history_tensor = tf.transpose(
        quotient_column_residual_scale_history.stack(), [1, 0]
    )
    quotient_post_column_residual_history_tensor = tf.transpose(
        quotient_post_column_residual_history.stack(), [1, 0]
    )
    nan = tf.constant(float("nan"), DTYPE)
    reported_total = tf.where(valid, total, tf.fill(tf.shape(total), nan))
    reported_score = tf.where(
        valid[:, None], total_score, tf.fill(tf.shape(total_score), nan)
    )

    return {
        "objective": tf.reduce_mean(reported_total),
        "score": tf.reduce_mean(reported_score, axis=0),
        "per_batch_log_likelihood": reported_total,
        "per_batch_score": reported_score,
        "increment_history": increment_history_tensor,
        "increment_score_history": increment_score_history_tensor,
        "valid_chart": valid,
        "reset_valid_history": reset_valid_history_tensor,
        "minimum_mass_history": minimum_mass_history_tensor,
        "flow_valid_history": flow_valid_history_tensor,
        "geometry_valid_history": geometry_valid_history_tensor,
        "quotient_valid_history": quotient_valid_history_tensor,
        "reset_finite_history": reset_finite_history_tensor,
        "reset_factor_positive_history": reset_factor_positive_history_tensor,
        "covariance_gap_eigenvalue_history": covariance_gap_eigenvalue_history_tensor,
        "quotient_row_residual_history": quotient_row_residual_history_tensor,
        "quotient_column_residual_history": quotient_column_residual_history_tensor,
        "quotient_column_residual_scale_history": quotient_column_residual_scale_history_tensor,
        "quotient_post_column_residual_history": quotient_post_column_residual_history_tensor,
        "clip_boundary_away_history": clip_boundary_history_tensor[:, 1:],
        "final_particles": particles,
        "final_particles_tangent": particles_tangent,
        "final_log_weights": log_weights,
        "final_log_weights_tangent": log_weights_tangent,
    }


_CANONICAL_AUSTRIA_STATIC_SPEC = static_spec_from_model(
    latent_preclip_zhao_cui_sir_austria_model()
)
_TWO_NODE_STATIC_SPEC = static_spec_from_model(
    latent_preclip_two_node_spatial_sir_model()
)


@tf.function(jit_compile=True, reduce_retracing=True)
def latent_sir_contract_e_canonical_value_and_score_tf(
    theta: tf.Tensor,
    observations: tf.Tensor,
    initial_noise: tf.Tensor,
    transition_noise: tf.Tensor,
    fixed_reset_mask: tf.Tensor,
    residual_design: tf.Tensor,
    prepared_ridge: tf.Tensor,
    epsilon: tf.Tensor,
    scaling: tf.Tensor,
) -> dict[str, tf.Tensor]:
    """Execute the registered Austria SIR Contract E value and total score."""

    particle_count = initial_noise.shape[1]
    if particle_count is None:
        raise ValueError("canonical latent-SIR route requires static particle count")
    chunks = select_transport_chunks(int(particle_count))

    return latent_sir_contract_e_value_and_score_core(
        theta,
        {
            "observations": observations,
            "initial_noise": initial_noise,
            "transition_noise": transition_noise,
            "fixed_reset_mask": fixed_reset_mask,
            "residual_design": residual_design,
            "prepared_ridge": prepared_ridge,
            "epsilon": epsilon,
            "scaling": scaling,
        },
        _CANONICAL_AUSTRIA_STATIC_SPEC,
        steps=CANONICAL_ANNEALING_STEPS,
        balance_steps=CANONICAL_BALANCE_STEPS,
        row_chunk_size=chunks.row_chunk_size,
        col_chunk_size=chunks.col_chunk_size,
    )


@tf.function(jit_compile=True, reduce_retracing=True)
def latent_sir_two_node_contract_e_value_and_score_tf(
    theta: tf.Tensor,
    observations: tf.Tensor,
    initial_noise: tf.Tensor,
    transition_noise: tf.Tensor,
    fixed_reset_mask: tf.Tensor,
    residual_design: tf.Tensor,
    prepared_ridge: tf.Tensor,
    epsilon: tf.Tensor,
    scaling: tf.Tensor,
) -> dict[str, tf.Tensor]:
    """Execute the registered coupled two-node Contract E route."""

    particle_count = initial_noise.shape[1]
    if particle_count is None:
        raise ValueError("two-node latent-SIR route requires static particle count")
    chunks = select_transport_chunks(int(particle_count))

    return latent_sir_contract_e_value_and_score_core(
        theta,
        {
            "observations": observations,
            "initial_noise": initial_noise,
            "transition_noise": transition_noise,
            "fixed_reset_mask": fixed_reset_mask,
            "residual_design": residual_design,
            "prepared_ridge": prepared_ridge,
            "epsilon": epsilon,
            "scaling": scaling,
        },
        _TWO_NODE_STATIC_SPEC,
        steps=CANONICAL_ANNEALING_STEPS,
        balance_steps=CANONICAL_BALANCE_STEPS,
        row_chunk_size=chunks.row_chunk_size,
        col_chunk_size=chunks.col_chunk_size,
    )


def make_latent_sir_contract_e_candidate(
    model: LatentPreclipSIRSSM,
    prepared: Mapping[str, Any],
    *,
    steps: int,
    balance_steps: int = 0,
    row_chunk_size: int,
    col_chunk_size: int,
    jit_compile: bool = True,
):
    """Bind the candidate graph with the repository-default XLA JIT policy."""

    spec = static_spec_from_model(model)
    tensors = _as_prepared_tensors(prepared, spec)
    validate_transport_chunks(
        int(tensors["initial_noise"].shape[1]),
        row_chunk_size=row_chunk_size,
        col_chunk_size=col_chunk_size,
    )

    @tf.function(
        input_signature=[tf.TensorSpec([PARAMETER_COUNT], DTYPE)],
        jit_compile=jit_compile,
        reduce_retracing=True,
    )
    def value_and_score(theta: tf.Tensor):
        return latent_sir_contract_e_value_and_score_core(
            theta,
            tensors,
            spec,
            steps=steps,
            balance_steps=balance_steps,
            row_chunk_size=row_chunk_size,
            col_chunk_size=col_chunk_size,
        )

    return value_and_score


__all__ = [
    "CANONICAL_ANNEALING_STEPS",
    "CANONICAL_BALANCE_STEPS",
    "CANONICAL_ROUTE_SPECIFICATION_ID",
    "CANONICAL_STEPS",
    "CANDIDATE_ROUTE_ID",
    "CANDIDATE_STATUS",
    "DTYPE",
    "LatentSIRStaticSpec",
    "PARAMETER_COUNT",
    "PARAMETER_NAMES",
    "TWO_NODE_ROUTE_SPECIFICATION_ID",
    "latent_sir_contract_e_value_and_score_core",
    "latent_sir_contract_e_canonical_value_and_score_tf",
    "latent_sir_two_node_contract_e_value_and_score_tf",
    "make_latent_sir_contract_e_candidate",
    "static_spec_from_model",
]
