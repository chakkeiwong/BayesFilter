"""Canonical fixed-noise LGSSM LEDH graph with the Contract E-Chol reset."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import tensorflow as tf

from bayesfilter.highdim.ledh_contract_e_streaming_tf import (
    _contract_e_streaming_forward_core,
    _contract_e_streaming_forward_jvp_core,
    _contract_e_streaming_jvp_core,
)
from bayesfilter.highdim.transport_chunk_policy import validate_transport_chunks


PARAMETER_NAMES = ("phi1", "phi2", "phi3", "q_scale", "r_scale")
PARAMETER_COUNT = len(PARAMETER_NAMES)
STATE_DIMENSION = 3
OBSERVATION_DIMENSION = 3
DTYPE = tf.float64
_LOG_TWO_PI = tf.math.log(tf.constant(6.283185307179586476925286766559, DTYPE))
_EPSILON0_FLOOR = tf.constant(1.0e-6, DTYPE)
_OBSERVATION_MATRIX = tf.constant(
    [
        [1.0, 0.25, -0.15],
        [0.2, 1.1, 0.3],
        [-0.1, 0.35, 0.9],
    ],
    dtype=DTYPE,
)


def _require_static_shape(value: tf.Tensor, shape: tuple[int | None, ...], name: str) -> None:
    actual = value.shape
    if actual.rank != len(shape):
        raise ValueError(f"{name} must have rank {len(shape)}, got {actual}")
    for index, expected in enumerate(shape):
        if expected is not None and actual[index] != expected:
            raise ValueError(f"{name} axis {index} must be {expected}, got {actual}")


def _canonical_dtype(dtype: tf.dtypes.DType) -> tf.dtypes.DType:
    dtype = tf.dtypes.as_dtype(dtype)
    if dtype not in (tf.float32, tf.float64):
        raise ValueError(f"canonical LGSSM dtype must be float32 or float64, got {dtype}")
    return dtype


def _observation_matrix(dtype: tf.dtypes.DType) -> tf.Tensor:
    return tf.cast(_OBSERVATION_MATRIX, _canonical_dtype(dtype))


def _as_prepared_tensors(
    prepared: Mapping[str, Any], *, dtype: tf.dtypes.DType = DTYPE
) -> dict[str, tf.Tensor]:
    dtype = _canonical_dtype(dtype)
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
        raise ValueError(f"missing prepared inputs: {missing}")
    tensors = {
        "observations": tf.convert_to_tensor(prepared["observations"], dtype),
        "initial_noise": tf.convert_to_tensor(prepared["initial_noise"], dtype),
        "transition_noise": tf.convert_to_tensor(prepared["transition_noise"], dtype),
        "fixed_reset_mask": tf.convert_to_tensor(prepared["fixed_reset_mask"], tf.bool),
        "residual_design": tf.convert_to_tensor(prepared["residual_design"], dtype),
        "prepared_ridge": tf.convert_to_tensor(prepared["prepared_ridge"], dtype),
        "epsilon": tf.convert_to_tensor(prepared["epsilon"], dtype),
        "scaling": tf.convert_to_tensor(prepared["scaling"], dtype),
    }
    _require_static_shape(tensors["observations"], (None, OBSERVATION_DIMENSION), "observations")
    _require_static_shape(tensors["initial_noise"], (None, None, STATE_DIMENSION), "initial_noise")
    _require_static_shape(
        tensors["transition_noise"],
        (tensors["initial_noise"].shape[0], tensors["observations"].shape[0], tensors["initial_noise"].shape[1], STATE_DIMENSION),
        "transition_noise",
    )
    _require_static_shape(
        tensors["fixed_reset_mask"],
        (tensors["initial_noise"].shape[0], tensors["observations"].shape[0]),
        "fixed_reset_mask",
    )
    _require_static_shape(
        tensors["residual_design"],
        (
            tensors["initial_noise"].shape[0],
            tensors["observations"].shape[0],
            tensors["initial_noise"].shape[1],
            STATE_DIMENSION,
        ),
        "residual_design",
    )
    _require_static_shape(
        tensors["prepared_ridge"],
        (tensors["initial_noise"].shape[0], tensors["observations"].shape[0]),
        "prepared_ridge",
    )
    if tensors["initial_noise"].shape[0] is None or tensors["initial_noise"].shape[1] is None:
        raise ValueError("batch size and particle count must be statically known")
    if tensors["observations"].shape[0] is None:
        raise ValueError("time length must be statically known")
    return tensors


def _lgssm_components(theta: tf.Tensor, batch_size: int) -> dict[str, tf.Tensor]:
    theta = tf.reshape(tf.convert_to_tensor(theta), [PARAMETER_COUNT])
    dtype = _canonical_dtype(theta.dtype)
    phi = theta[:STATE_DIMENSION]
    q_scale = theta[3]
    r_scale = theta[4]
    state_identity = tf.eye(STATE_DIMENSION, dtype=dtype)
    observation_identity = tf.eye(OBSERVATION_DIMENSION, dtype=dtype)
    transition_matrix_single = tf.linalg.diag(phi)
    transition_covariance_single = tf.square(q_scale) * state_identity
    observation_covariance_single = tf.square(r_scale) * observation_identity
    initial_std = q_scale / tf.sqrt(1.0 - tf.square(phi))
    return {
        "theta": theta,
        "phi": phi,
        "q_scale": q_scale,
        "r_scale": r_scale,
        "initial_std": initial_std,
        "transition_matrix": tf.tile(
            transition_matrix_single[None, :, :], [batch_size, 1, 1]
        ),
        "transition_covariance": tf.tile(
            transition_covariance_single[None, :, :], [batch_size, 1, 1]
        ),
        "observation_covariance": tf.tile(
            observation_covariance_single[None, :, :], [batch_size, 1, 1]
        ),
        "observation_matrix": _observation_matrix(dtype),
    }


def _lgssm_component_tangents(theta: tf.Tensor, batch_size: int) -> dict[str, tf.Tensor]:
    theta = tf.reshape(tf.convert_to_tensor(theta), [PARAMETER_COUNT])
    dtype = _canonical_dtype(theta.dtype)
    phi = theta[:STATE_DIMENSION]
    q_scale = theta[3]
    r_scale = theta[4]
    state_identity = tf.eye(STATE_DIMENSION, dtype=dtype)
    observation_identity = tf.eye(OBSERVATION_DIMENSION, dtype=dtype)
    zero_state = tf.zeros([STATE_DIMENSION, STATE_DIMENSION], dtype)
    zero_observation = tf.zeros([OBSERVATION_DIMENSION, OBSERVATION_DIMENSION], dtype)
    d_transition_matrix_single = tf.stack(
        [tf.linalg.diag(tf.one_hot(index, STATE_DIMENSION, dtype=dtype)) for index in range(STATE_DIMENSION)]
        + [zero_state, zero_state],
        axis=-1,
    )
    d_transition_covariance_single = tf.stack(
        [zero_state, zero_state, zero_state, 2.0 * q_scale * state_identity, zero_state],
        axis=-1,
    )
    d_observation_covariance_single = tf.stack(
        [zero_observation, zero_observation, zero_observation, zero_observation, 2.0 * r_scale * observation_identity],
        axis=-1,
    )
    one_minus_phi_squared = 1.0 - tf.square(phi)
    root = tf.sqrt(one_minus_phi_squared)
    phi_basis = tf.concat(
        [
            tf.eye(STATE_DIMENSION, dtype=dtype),
            tf.zeros([STATE_DIMENSION, 2], dtype),
        ],
        axis=1,
    )
    squared_phi_tangent = phi_basis * (2.0 * phi[:, None])
    root_tangent = 0.5 * (-squared_phi_tangent / root[:, None])
    quotient_denominator_factor = (-q_scale / root) / root
    q_basis = tf.constant([0.0, 0.0, 0.0, 1.0, 0.0], dtype)
    initial_std_tangent = (
        root_tangent * quotient_denominator_factor[:, None]
        + q_basis[None, :] / root[:, None]
    )
    d_initial_std = tf.concat(
        [
            initial_std_tangent[:, :3],
            initial_std_tangent[:, 3:4],
            initial_std_tangent[:, 4:5],
        ],
        axis=1,
    )
    return {
        "d_initial_std": d_initial_std,
        "d_transition_matrix": tf.tile(
            d_transition_matrix_single[None, :, :, :], [batch_size, 1, 1, 1]
        ),
        "d_transition_covariance": tf.tile(
            d_transition_covariance_single[None, :, :, :], [batch_size, 1, 1, 1]
        ),
        "d_observation_covariance": tf.tile(
            d_observation_covariance_single[None, :, :, :], [batch_size, 1, 1, 1]
        ),
        "d_transition_scale": tf.constant([0.0, 0.0, 0.0, 1.0, 0.0], dtype),
    }


def _cholesky_jvp(chol: tf.Tensor, matrix_tangent: tf.Tensor) -> tf.Tensor:
    dimension = tf.shape(chol)[-1]
    batch_size = tf.shape(chol)[0]
    chol_inverse = tf.linalg.triangular_solve(
        chol, tf.eye(dimension, batch_shape=[batch_size], dtype=chol.dtype)
    )
    columns = []
    for index in range(PARAMETER_COUNT):
        tangent = matrix_tangent[..., index]
        half_tangent = tf.constant(0.5, chol.dtype) * tangent
        symmetric_tangent = half_tangent + tf.linalg.matrix_transpose(half_tangent)
        right_solved = tf.linalg.matmul(
            symmetric_tangent, chol_inverse, transpose_b=True
        )
        inner = tf.linalg.matmul(chol_inverse, right_solved)
        lower = tf.linalg.band_part(inner, -1, 0)
        phi = tf.linalg.set_diag(
            lower, tf.constant(0.5, chol.dtype) * tf.linalg.diag_part(lower)
        )
        columns.append(tf.linalg.matmul(chol, phi))
    return tf.stack(columns, axis=-1)


def _cholesky_solve_jvp(
    chol: tf.Tensor,
    right_hand_side: tf.Tensor,
    solution: tf.Tensor,
    matrix_tangent: tf.Tensor,
    right_hand_side_tangent: tf.Tensor,
) -> tf.Tensor:
    chol_tangent = _cholesky_jvp(chol, matrix_tangent)
    first_solution = tf.linalg.triangular_solve(chol, right_hand_side)
    columns = []
    for index in range(PARAMETER_COUNT):
        first_tangent = tf.linalg.triangular_solve(
            chol,
            right_hand_side_tangent[..., index]
            - tf.linalg.matmul(chol_tangent[..., index], first_solution),
        )
        columns.append(
            tf.linalg.triangular_solve(
                chol,
                first_tangent
                - tf.linalg.matmul(
                    chol_tangent[..., index], solution, transpose_a=True
                ),
                adjoint=True,
            )
        )
    return tf.stack(columns, axis=-1)


def _lgssm_flow_forward_core(
    prior_mean: tf.Tensor,
    pre_flow: tf.Tensor,
    observation: tf.Tensor,
    transition_covariance: tf.Tensor,
    observation_covariance: tf.Tensor,
) -> dict[str, tf.Tensor]:
    batch_size = tf.shape(prior_mean)[0]
    dtype = _canonical_dtype(prior_mean.dtype)
    observation_matrix = _observation_matrix(dtype)
    state_identity = tf.eye(STATE_DIMENSION, batch_shape=[batch_size], dtype=dtype)
    observation_identity = tf.eye(
        OBSERVATION_DIMENSION, batch_shape=[batch_size], dtype=dtype
    )
    prior_chol = tf.linalg.cholesky(transition_covariance)
    observation_chol = tf.linalg.cholesky(observation_covariance)
    prior_precision = tf.linalg.cholesky_solve(prior_chol, state_identity)
    observation_precision = tf.linalg.cholesky_solve(
        observation_chol, observation_identity
    )
    post_precision = prior_precision + tf.einsum(
        "od,boq,qe->bde",
        observation_matrix,
        observation_precision,
        observation_matrix,
    )
    post_precision_chol = tf.linalg.cholesky(post_precision)
    post_covariance = tf.linalg.cholesky_solve(post_precision_chol, state_identity)
    post_chol = tf.linalg.cholesky(post_covariance)
    observation_info = tf.einsum(
        "od,boq,q->bd", observation_matrix, observation_precision, observation
    )
    info = tf.einsum("bij,bnj->bni", prior_precision, prior_mean) + observation_info[:, None, :]
    post_mean = tf.einsum("bij,bnj->bni", post_covariance, info)
    prior_chol_inverse = tf.linalg.triangular_solve(prior_chol, state_identity)
    affine = tf.linalg.matmul(post_chol, prior_chol_inverse)
    post_flow = post_mean + tf.linalg.matmul(
        pre_flow - prior_mean, affine, transpose_b=True
    )
    forward_log_abs_det = (
        tf.reduce_sum(tf.math.log(tf.linalg.diag_part(post_chol)), axis=1)
        - tf.reduce_sum(tf.math.log(tf.linalg.diag_part(prior_chol)), axis=1)
    )
    factor_diagonals = tf.concat(
        [
            tf.linalg.diag_part(prior_chol),
            tf.linalg.diag_part(observation_chol),
            tf.linalg.diag_part(post_precision_chol),
            tf.linalg.diag_part(post_chol),
        ],
        axis=1,
    )
    factor_finite = tf.reduce_all(tf.math.is_finite(factor_diagonals), axis=1)
    factor_positive = tf.reduce_all(factor_diagonals > 0.0, axis=1)
    output_finite = (
        tf.reduce_all(tf.math.is_finite(post_flow), axis=[1, 2])
        & tf.reduce_all(tf.math.is_finite(post_mean), axis=[1, 2])
        & tf.math.is_finite(forward_log_abs_det)
    )
    return {
        "particles": post_flow,
        "post_mean": post_mean,
        "affine": affine,
        "forward_log_abs_det": forward_log_abs_det,
        "prior_chol": prior_chol,
        "observation_chol": observation_chol,
        "prior_precision": prior_precision,
        "observation_precision": observation_precision,
        "post_precision": post_precision,
        "post_precision_chol": post_precision_chol,
        "post_covariance": post_covariance,
        "post_chol": post_chol,
        "prior_chol_inverse": prior_chol_inverse,
        "info": info,
        "factor_finite": factor_finite,
        "factor_positive": factor_positive,
        "valid_chart": factor_finite & factor_positive & output_finite,
    }


def _lgssm_flow_jvp_core(
    prior_mean: tf.Tensor,
    pre_flow: tf.Tensor,
    observation: tf.Tensor,
    transition_covariance: tf.Tensor,
    observation_covariance: tf.Tensor,
    prior_mean_tangent: tf.Tensor,
    pre_flow_tangent: tf.Tensor,
    transition_covariance_tangent: tf.Tensor,
    observation_covariance_tangent: tf.Tensor,
) -> dict[str, tf.Tensor]:
    forward = _lgssm_flow_forward_core(
        prior_mean,
        pre_flow,
        observation,
        transition_covariance,
        observation_covariance,
    )
    batch_size = tf.shape(prior_mean)[0]
    dtype = _canonical_dtype(prior_mean.dtype)
    observation_matrix = _observation_matrix(dtype)
    state_identity = tf.eye(STATE_DIMENSION, batch_shape=[batch_size], dtype=dtype)
    zero_state_rhs = tf.zeros(
        [batch_size, STATE_DIMENSION, STATE_DIMENSION, PARAMETER_COUNT], dtype
    )
    zero_observation_rhs = tf.zeros(
        [batch_size, OBSERVATION_DIMENSION, OBSERVATION_DIMENSION, PARAMETER_COUNT], dtype
    )
    d_prior_precision = _cholesky_solve_jvp(
        forward["prior_chol"],
        state_identity,
        forward["prior_precision"],
        transition_covariance_tangent,
        zero_state_rhs,
    )
    d_observation_precision = _cholesky_solve_jvp(
        forward["observation_chol"],
        tf.eye(
            OBSERVATION_DIMENSION, batch_shape=[batch_size], dtype=dtype
        ),
        forward["observation_precision"],
        observation_covariance_tangent,
        zero_observation_rhs,
    )
    d_post_precision = d_prior_precision + tf.einsum(
        "od,boqk,qe->bdek",
        observation_matrix,
        d_observation_precision,
        observation_matrix,
    )
    d_post_covariance = _cholesky_solve_jvp(
        forward["post_precision_chol"],
        state_identity,
        forward["post_covariance"],
        d_post_precision,
        zero_state_rhs,
    )
    d_post_chol = _cholesky_jvp(forward["post_chol"], d_post_covariance)
    d_prior_chol = _cholesky_jvp(
        forward["prior_chol"], transition_covariance_tangent
    )
    d_prior_chol_inverse = tf.stack(
        [
            tf.linalg.triangular_solve(
                forward["prior_chol"],
                -tf.linalg.matmul(
                    d_prior_chol[..., index], forward["prior_chol_inverse"]
                ),
            )
            for index in range(PARAMETER_COUNT)
        ],
        axis=-1,
    )
    d_affine = (
        tf.einsum("bijq,bjk->bikq", d_post_chol, forward["prior_chol_inverse"])
        + tf.einsum("bij,bjkq->bikq", forward["post_chol"], d_prior_chol_inverse)
    )
    d_observation_info = tf.einsum(
        "od,boqk,q->bdk", observation_matrix, d_observation_precision, observation
    )
    d_info = (
        tf.einsum("bijk,bnj->bnik", d_prior_precision, prior_mean)
        + tf.einsum("bij,bnjk->bnik", forward["prior_precision"], prior_mean_tangent)
        + d_observation_info[:, None, :, :]
    )
    d_post_mean = (
        tf.einsum("bijk,bnj->bnik", d_post_covariance, forward["info"])
        + tf.einsum("bij,bnjk->bnik", forward["post_covariance"], d_info)
    )
    delta = pre_flow - prior_mean
    d_delta = pre_flow_tangent - prior_mean_tangent
    particles_tangent = d_post_mean + (
        tf.einsum("bnj,bijk->bnik", delta, d_affine)
        + tf.einsum("bnjk,bij->bnik", d_delta, forward["affine"])
    )
    d_logdet_post = tf.reduce_sum(
        tf.linalg.diag_part(tf.transpose(d_post_chol, [0, 3, 1, 2]))
        / tf.linalg.diag_part(forward["post_chol"])[:, None, :],
        axis=2,
    )
    d_logdet_prior = tf.reduce_sum(
        tf.linalg.diag_part(tf.transpose(d_prior_chol, [0, 3, 1, 2]))
        / tf.linalg.diag_part(forward["prior_chol"])[:, None, :],
        axis=2,
    )
    return {
        "particles": particles_tangent,
        "forward_log_abs_det": d_logdet_post - d_logdet_prior,
    }


def _gaussian_log_density_forward_core(
    residual: tf.Tensor, covariance: tf.Tensor
) -> dict[str, tf.Tensor]:
    dtype = _canonical_dtype(residual.dtype)
    chol = tf.linalg.cholesky(covariance)
    solved = tf.linalg.cholesky_solve(
        chol, tf.linalg.matrix_transpose(residual)
    )
    solved_rows = tf.linalg.matrix_transpose(solved)
    quadratic = tf.reduce_sum(solved_rows * residual, axis=-1)
    logdet = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(chol)), axis=-1)
    dimension = tf.cast(tf.shape(residual)[-1], dtype)
    log_two_pi = tf.cast(_LOG_TWO_PI, dtype)
    return {
        "value": -0.5 * (dimension * log_two_pi + logdet[:, None] + quadratic),
        "chol": chol,
        "solved": solved_rows,
    }


def _gaussian_log_density_jvp_core(
    residual: tf.Tensor,
    covariance: tf.Tensor,
    residual_tangent: tf.Tensor,
    covariance_tangent: tf.Tensor,
) -> dict[str, tf.Tensor]:
    forward = _gaussian_log_density_forward_core(residual, covariance)
    solved_tangent = _cholesky_solve_jvp(
        forward["chol"],
        tf.linalg.matrix_transpose(residual),
        tf.linalg.matrix_transpose(forward["solved"]),
        covariance_tangent,
        tf.transpose(residual_tangent, [0, 2, 1, 3]),
    )
    solved_rows_tangent = tf.transpose(solved_tangent, [0, 2, 1, 3])
    quadratic_tangent = tf.reduce_sum(
        solved_rows_tangent * residual[:, :, :, None]
        + forward["solved"][:, :, :, None] * residual_tangent,
        axis=2,
    )
    chol_tangent = _cholesky_jvp(forward["chol"], covariance_tangent)
    logdet_tangent = 2.0 * tf.reduce_sum(
        tf.linalg.diag_part(tf.transpose(chol_tangent, [0, 3, 1, 2]))
        / tf.linalg.diag_part(forward["chol"])[:, None, :],
        axis=2,
    )
    tangent = -0.5 * (logdet_tangent[:, None, :] + quadratic_tangent)
    return {"value": forward["value"], "tangent": tangent}


def _normalize_log_weights_forward_core(logits: tf.Tensor) -> dict[str, tf.Tensor]:
    increment = tf.reduce_logsumexp(logits, axis=1)
    normalized_log_weights = logits - increment[:, None]
    normalized_weights = tf.exp(normalized_log_weights)
    return {
        "increment": increment,
        "normalized_log_weights": normalized_log_weights,
        "normalized_weights": normalized_weights,
    }


def _normalize_log_weights_jvp_core(
    logits: tf.Tensor, logits_tangent: tf.Tensor
) -> dict[str, tf.Tensor]:
    forward = _normalize_log_weights_forward_core(logits)
    raw_max = tf.reduce_max(logits, axis=1, keepdims=True)
    finite_max = tf.where(tf.math.is_finite(raw_max), raw_max, tf.zeros_like(raw_max))
    shifted_exponential = tf.exp(logits - finite_max)
    exponential_sum = tf.reduce_sum(shifted_exponential, axis=1)
    reciprocal_sum = tf.math.reciprocal(exponential_sum)
    increment_columns = []
    log_weight_columns = []
    weight_columns = []
    for index in range(PARAMETER_COUNT):
        direction = logits_tangent[..., index]
        shifted_tangent = tf.add_n(
            [direction, tf.broadcast_to(-tf.zeros_like(finite_max), tf.shape(direction))]
        )
        exponential_tangent = shifted_tangent * shifted_exponential
        increment_tangent = tf.reduce_sum(exponential_tangent, axis=1) * reciprocal_sum
        log_weight_tangent = tf.add_n(
            [
                direction,
                tf.broadcast_to(-increment_tangent[:, None], tf.shape(direction)),
            ]
        )
        increment_columns.append(increment_tangent)
        log_weight_columns.append(log_weight_tangent)
        weight_columns.append(log_weight_tangent * forward["normalized_weights"])
    increment_tangent = tf.stack(increment_columns, axis=-1)
    normalized_log_weights_tangent = tf.stack(log_weight_columns, axis=-1)
    normalized_weights_tangent = tf.stack(weight_columns, axis=-1)
    return {
        **forward,
        "increment_tangent": increment_tangent,
        "normalized_log_weights_tangent": normalized_log_weights_tangent,
        "normalized_weights_tangent": normalized_weights_tangent,
    }


def _normalize_log_weights_vjp_core(
    logits: tf.Tensor,
    increment_bar: tf.Tensor,
    normalized_log_weights_bar: tf.Tensor,
) -> dict[str, tf.Tensor]:
    forward = _normalize_log_weights_forward_core(logits)
    logits_bar = (
        increment_bar[:, None] * forward["normalized_weights"]
        + normalized_log_weights_bar
        - forward["normalized_weights"]
        * tf.reduce_sum(normalized_log_weights_bar, axis=1, keepdims=True)
    )
    return {**forward, "logits_bar": logits_bar}


def _geometry_forward_core(particles: tf.Tensor) -> dict[str, tf.Tensor]:
    dtype = _canonical_dtype(particles.dtype)
    center = tf.reduce_mean(particles, axis=1, keepdims=True)
    centered = particles - center
    variance = tf.reduce_mean(tf.square(centered), axis=1)
    standard_deviation = tf.sqrt(variance)
    diameter = tf.reduce_max(standard_deviation, axis=1)
    diameter_mask = standard_deviation == diameter[:, None]
    diameter_count = tf.reduce_sum(tf.cast(diameter_mask, tf.int32), axis=1)
    scale = tf.sqrt(tf.cast(STATE_DIMENSION, dtype)) * diameter
    scaled_geometry = centered / scale[:, None, None]
    maximum = tf.reduce_max(scaled_geometry, axis=[1, 2])
    minimum = tf.reduce_min(scaled_geometry, axis=[1, 2])
    maximum_mask = scaled_geometry == maximum[:, None, None]
    minimum_mask = scaled_geometry == minimum[:, None, None]
    maximum_count = tf.reduce_sum(tf.cast(maximum_mask, tf.int32), axis=[1, 2])
    minimum_count = tf.reduce_sum(tf.cast(minimum_mask, tf.int32), axis=[1, 2])
    coordinate_range = maximum - minimum
    range_squared = tf.square(coordinate_range)
    epsilon0_floor = tf.cast(_EPSILON0_FLOOR, dtype)
    epsilon0_floor_inactive = range_squared >= epsilon0_floor
    epsilon0 = tf.maximum(range_squared, epsilon0_floor)
    finite = (
        tf.reduce_all(tf.math.is_finite(particles), axis=[1, 2])
        & tf.reduce_all(tf.math.is_finite(scaled_geometry), axis=[1, 2])
        & tf.math.is_finite(diameter)
        & tf.math.is_finite(epsilon0)
    )
    return {
        "center": center,
        "centered": centered,
        "standard_deviation": standard_deviation,
        "diameter": diameter,
        "diameter_mask": diameter_mask,
        "diameter_count": diameter_count,
        "scale": scale,
        "scaled_geometry": scaled_geometry,
        "maximum_mask": maximum_mask,
        "minimum_mask": minimum_mask,
        "maximum_count": maximum_count,
        "minimum_count": minimum_count,
        "coordinate_range": coordinate_range,
        "epsilon0_floor_inactive": epsilon0_floor_inactive,
        "epsilon0": epsilon0,
        "valid_chart": finite & (diameter > 0.0),
    }


def _geometry_jvp_core(
    particles: tf.Tensor, particles_tangent: tf.Tensor
) -> dict[str, tf.Tensor]:
    dtype = _canonical_dtype(particles.dtype)
    forward = _geometry_forward_core(particles)
    center_tangent = tf.reduce_mean(particles_tangent, axis=1, keepdims=True)
    centered_tangent = particles_tangent - center_tangent
    variance_tangent = 2.0 * tf.reduce_mean(
        forward["centered"][:, :, :, None] * centered_tangent, axis=1
    )
    standard_deviation_tangent = (
        variance_tangent / (2.0 * forward["standard_deviation"][:, :, None])
    )
    diameter_weights = tf.cast(forward["diameter_mask"], dtype) / tf.cast(
        forward["diameter_count"][:, None], dtype
    )
    diameter_tangent = tf.reduce_sum(
        diameter_weights[:, :, None] * standard_deviation_tangent, axis=1
    )
    scale_tangent = tf.sqrt(tf.cast(STATE_DIMENSION, dtype)) * diameter_tangent
    scale = forward["scale"][:, None, None, None]
    scaled_geometry_tangent = (
        centered_tangent / scale
        + scale_tangent[:, None, None, :]
        * ((-forward["centered"][:, :, :, None] / scale) / scale)
    )
    maximum_weights = tf.cast(forward["maximum_mask"], dtype) / tf.cast(
        forward["maximum_count"][:, None, None], dtype
    )
    minimum_weights = tf.cast(forward["minimum_mask"], dtype) / tf.cast(
        forward["minimum_count"][:, None, None], dtype
    )
    maximum_tangent = tf.reduce_sum(
        maximum_weights[:, :, :, None] * scaled_geometry_tangent, axis=[1, 2]
    )
    minimum_tangent = tf.reduce_sum(
        minimum_weights[:, :, :, None] * scaled_geometry_tangent, axis=[1, 2]
    )
    epsilon0_tangent = tf.where(
        forward["epsilon0_floor_inactive"][:, None],
        2.0 * forward["coordinate_range"][:, None]
        * (maximum_tangent - minimum_tangent),
        tf.zeros_like(maximum_tangent),
    )
    return {
        **forward,
        "scaled_geometry_tangent": scaled_geometry_tangent,
        "epsilon0_tangent": epsilon0_tangent,
    }


def _physical_chart(theta: tf.Tensor) -> tf.Tensor:
    theta = tf.reshape(theta, [PARAMETER_COUNT])
    return (
        tf.reduce_all(tf.math.is_finite(theta))
        & tf.reduce_all(tf.abs(theta[:STATE_DIMENSION]) < 1.0)
        & (theta[3] > 0.0)
        & (theta[4] > 0.0)
    )


def _sinkhorn_schedule(epsilon0: tf.Tensor, epsilon: tf.Tensor, scaling: tf.Tensor, steps: int) -> tf.Tensor:
    running = epsilon0
    factor = tf.square(scaling)
    branches = tf.TensorArray(
        tf.bool, size=steps, element_shape=tf.TensorShape([None])
    )

    def cond(index: tf.Tensor, _running: tf.Tensor, _branches: tf.TensorArray) -> tf.Tensor:
        return index < steps

    def body(index: tf.Tensor, running_value: tf.Tensor, records: tf.TensorArray):
        proposed = running_value * factor
        records = records.write(index, proposed >= epsilon)
        return index + 1, tf.maximum(proposed, epsilon), records

    _, _, branches = tf.while_loop(
        cond,
        body,
        (tf.constant(0, tf.int32), running, branches),
        maximum_iterations=steps,
    )
    return tf.transpose(branches.stack(), [1, 0])


def _canonical_primal_core(
    theta: tf.Tensor,
    prepared: Mapping[str, tf.Tensor],
    *,
    steps: int,
    balance_steps: int,
    row_chunk_size: int,
    col_chunk_size: int,
) -> dict[str, tf.Tensor]:
    validate_transport_chunks(
        int(prepared["initial_noise"].shape[1]),
        row_chunk_size=row_chunk_size,
        col_chunk_size=col_chunk_size,
    )
    if balance_steps < 0:
        raise ValueError("balance_steps must be non-negative")
    initial_noise = prepared["initial_noise"]
    dtype = _canonical_dtype(initial_noise.dtype)
    theta = tf.reshape(tf.convert_to_tensor(theta, dtype), [PARAMETER_COUNT])
    batch_size = int(initial_noise.shape[0])
    particle_count = int(initial_noise.shape[1])
    time_steps = int(prepared["observations"].shape[0])
    components = _lgssm_components(theta, batch_size)
    particles = initial_noise * components["initial_std"][None, None, :]
    uniform_log_weight = -tf.math.log(tf.cast(particle_count, dtype))
    log_weights = tf.fill([batch_size, particle_count], uniform_log_weight)
    per_batch_log_likelihood = tf.zeros([batch_size], dtype)
    valid_chart = tf.fill([batch_size], _physical_chart(theta))
    minimum_mass = tf.fill([batch_size], tf.constant(float("inf"), dtype))
    flow_valid_history = []
    geometry_valid_history = []
    quotient_valid_history = []
    reset_valid_history = []
    mass_history = []
    diameter_masks = []
    maximum_masks = []
    minimum_masks = []
    epsilon0_branches = []
    sinkhorn_branches = []
    increment_history = []
    quotient_mass_history = []
    quotient_row_residual_history = []
    quotient_row_target_history = []
    quotient_row_signed_residual_history = []
    quotient_column_mass_history = []
    quotient_column_target_history = []
    quotient_column_signed_residual_history = []
    quotient_column_residual_history = []
    quotient_post_quotient_column_mass_history = []
    quotient_post_quotient_column_signed_residual_history = []
    quotient_post_quotient_column_residual_history = []
    quotient_marginal_roundoff_tolerance_history = []
    quotient_marginal_valid_history = []
    quotient_row_residual_scale_history = []
    quotient_column_residual_scale_history = []
    target_mean_history = []
    target_covariance_history = []
    plus_mean_history = []
    plus_covariance_history = []
    covariance_gap_history = []
    covariance_gap_eigenvalue_history = []
    output_mean_history = []
    output_covariance_history = []
    injected_covariance_history = []
    affine_history = []
    ridged_identity_residual_history = []
    ridged_identity_scale_history = []
    raw_covariance_residual_history = []
    predicted_raw_covariance_residual_history = []
    raw_covariance_prediction_error_history = []
    mean_residual_history = []
    residual_design_sum_history = []
    residual_design_absolute_scale_history = []
    gap_chol_diagonal_history = []
    target_chol_diagonal_history = []
    injected_chol_diagonal_history = []
    gap_condition_proxy_history = []
    target_condition_proxy_history = []
    injected_condition_proxy_history = []
    realized_ridge_history = []

    for time_index in range(time_steps):
        observation = prepared["observations"][time_index]
        prior_mean = tf.einsum("bnj,bdj->bnd", particles, components["transition_matrix"])
        pre_flow = (
            prior_mean
            + components["q_scale"] * prepared["transition_noise"][:, time_index, :, :]
        )
        flow = _lgssm_flow_forward_core(
            prior_mean,
            pre_flow,
            observation,
            components["transition_covariance"],
            components["observation_covariance"],
        )
        transition_density = _gaussian_log_density_forward_core(
            flow["particles"] - prior_mean, components["transition_covariance"]
        )["value"]
        proposal_density = _gaussian_log_density_forward_core(
            pre_flow - prior_mean, components["transition_covariance"]
        )["value"]
        predicted_observation = tf.einsum(
            "md,bnd->bnm", components["observation_matrix"], flow["particles"]
        )
        observation_density = _gaussian_log_density_forward_core(
            predicted_observation - observation[None, None, :],
            components["observation_covariance"],
        )["value"]
        logits = (
            log_weights
            + transition_density
            + observation_density
            - proposal_density
            + flow["forward_log_abs_det"][:, None]
        )
        normalization = _normalize_log_weights_forward_core(logits)
        per_batch_log_likelihood = (
            per_batch_log_likelihood + normalization["increment"]
        )
        increment_history.append(normalization["increment"])
        geometry = _geometry_forward_core(flow["particles"])
        contract_e = _contract_e_streaming_forward_core(
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
        reset_valid = (
            contract_e["quotient"]["valid_chart"]
            & contract_e["quotient"]["marginal_valid"]
            & contract_e["reset"]["finite"]
            & contract_e["reset"]["factor_diagonal_positive"]
        )
        active = prepared["fixed_reset_mask"][:, time_index]
        time_valid = flow["valid_chart"] & (
            ~active | (geometry["valid_chart"] & reset_valid)
        )
        valid_chart = valid_chart & time_valid
        active_mass = tf.where(
            active,
            contract_e["quotient"]["minimum_mass"],
            tf.fill([batch_size], tf.constant(float("inf"), dtype)),
        )
        minimum_mass = tf.minimum(minimum_mass, active_mass)
        particles = tf.where(
            active[:, None, None], contract_e["particles"], flow["particles"]
        )
        log_weights = tf.where(
            active[:, None],
            tf.fill([batch_size, particle_count], uniform_log_weight),
            normalization["normalized_log_weights"],
        )
        flow_valid_history.append(flow["valid_chart"])
        geometry_valid_history.append(geometry["valid_chart"])
        quotient_valid_history.append(contract_e["quotient"]["valid_chart"])
        reset_valid_history.append(reset_valid)
        mass_history.append(contract_e["quotient"]["minimum_mass"])
        diameter_masks.append(geometry["diameter_mask"])
        maximum_masks.append(geometry["maximum_mask"])
        minimum_masks.append(geometry["minimum_mask"])
        epsilon0_branches.append(geometry["epsilon0_floor_inactive"])
        sinkhorn_branches.append(
            _sinkhorn_schedule(
                geometry["epsilon0"], prepared["epsilon"], prepared["scaling"], steps
            )
        )
        quotient_mass_history.append(contract_e["quotient"]["mass"])
        quotient_row_residual_history.append(
            contract_e["quotient"]["row_residual_by_batch"]
        )
        quotient_row_target_history.append(contract_e["quotient"]["row_target"])
        quotient_row_signed_residual_history.append(
            contract_e["quotient"]["row_signed_residual"]
        )
        quotient_column_mass_history.append(
            contract_e["quotient"]["column_mass"]
        )
        quotient_column_target_history.append(
            contract_e["quotient"]["column_target"]
        )
        quotient_column_signed_residual_history.append(
            contract_e["quotient"]["column_signed_residual"]
        )
        quotient_column_residual_history.append(
            contract_e["quotient"]["maximum_column_absolute_residual"]
        )
        quotient_post_quotient_column_mass_history.append(
            contract_e["quotient"]["post_quotient_column_mass"]
        )
        quotient_post_quotient_column_signed_residual_history.append(
            contract_e["quotient"]["post_quotient_column_signed_residual"]
        )
        quotient_post_quotient_column_residual_history.append(
            contract_e["quotient"][
                "maximum_post_quotient_column_absolute_residual"
            ]
        )
        quotient_marginal_roundoff_tolerance_history.append(
            contract_e["quotient"]["marginal_roundoff_tolerance"]
        )
        quotient_marginal_valid_history.append(
            contract_e["quotient"]["marginal_valid"]
        )
        quotient_row_residual_scale_history.append(
            contract_e["quotient"]["row_residual_scale"]
        )
        quotient_column_residual_scale_history.append(
            contract_e["quotient"]["column_residual_scale"]
        )
        reset = contract_e["reset"]
        target_mean_history.append(reset["target_mean"])
        target_covariance_history.append(reset["target_cov"])
        plus_mean_history.append(reset["plus_mean"])
        plus_covariance_history.append(reset["plus_cov"])
        covariance_gap_history.append(reset["gap"])
        covariance_gap_eigenvalue_history.append(reset["gap_eigenvalues"])
        output_mean_history.append(reset["output_mean"])
        output_covariance_history.append(reset["output_cov"])
        injected_covariance_history.append(reset["injected_cov"])
        affine_history.append(reset["affine"])
        ridged_identity_residual_history.append(reset["ridged_identity_residual"])
        ridged_identity_scale_history.append(reset["ridged_identity_absolute_scale"])
        raw_covariance_residual_history.append(reset["raw_covariance_residual"])
        predicted_raw_covariance_residual_history.append(
            reset["predicted_raw_covariance_residual"]
        )
        raw_covariance_prediction_error_history.append(
            reset["raw_covariance_residual"]
            - reset["predicted_raw_covariance_residual"]
        )
        mean_residual_history.append(reset["mean_residual"])
        residual_design_sum_history.append(reset["residual_design_sum"])
        residual_design_absolute_scale_history.append(
            reset["residual_design_absolute_scale"]
        )
        gap_chol_diagonal_history.append(reset["gap_chol_diagonal"])
        target_chol_diagonal_history.append(reset["target_chol_diagonal"])
        injected_chol_diagonal_history.append(reset["injected_chol_diagonal"])
        gap_condition_proxy_history.append(reset["gap_condition_proxy"])
        target_condition_proxy_history.append(reset["target_condition_proxy"])
        injected_condition_proxy_history.append(reset["injected_condition_proxy"])
        realized_ridge_history.append(reset["ridge"])

    return {
        "objective": tf.reduce_mean(per_batch_log_likelihood),
        "per_batch_log_likelihood": per_batch_log_likelihood,
        "valid_chart": valid_chart,
        "minimum_mass": minimum_mass,
        "final_particles": particles,
        "final_log_weights": log_weights,
        "flow_valid_history": tf.stack(flow_valid_history, axis=1),
        "geometry_valid_history": tf.stack(geometry_valid_history, axis=1),
        "quotient_valid_history": tf.stack(quotient_valid_history, axis=1),
        "reset_valid_history": tf.stack(reset_valid_history, axis=1),
        "minimum_mass_history": tf.stack(mass_history, axis=1),
        "diameter_max_mask": tf.stack(diameter_masks, axis=1),
        "geometry_max_mask": tf.stack(maximum_masks, axis=1),
        "geometry_min_mask": tf.stack(minimum_masks, axis=1),
        "epsilon0_floor_inactive": tf.stack(epsilon0_branches, axis=1),
        "sinkhorn_running_branch": tf.stack(sinkhorn_branches, axis=1),
        "increment_history": tf.stack(increment_history, axis=1),
        "active_reset_history": prepared["fixed_reset_mask"],
        "quotient_mass_history": tf.stack(quotient_mass_history, axis=1),
        "quotient_row_residual_history": tf.stack(
            quotient_row_residual_history, axis=1
        ),
        "quotient_row_target_history": tf.stack(
            quotient_row_target_history, axis=1
        ),
        "quotient_row_signed_residual_history": tf.stack(
            quotient_row_signed_residual_history, axis=1
        ),
        "quotient_column_mass_history": tf.stack(
            quotient_column_mass_history, axis=1
        ),
        "quotient_column_target_history": tf.stack(
            quotient_column_target_history, axis=1
        ),
        "quotient_column_signed_residual_history": tf.stack(
            quotient_column_signed_residual_history, axis=1
        ),
        "quotient_column_residual_history": tf.stack(
            quotient_column_residual_history, axis=1
        ),
        "quotient_post_quotient_column_mass_history": tf.stack(
            quotient_post_quotient_column_mass_history, axis=1
        ),
        "quotient_post_quotient_column_signed_residual_history": tf.stack(
            quotient_post_quotient_column_signed_residual_history, axis=1
        ),
        "quotient_post_quotient_column_residual_history": tf.stack(
            quotient_post_quotient_column_residual_history, axis=1
        ),
        "quotient_marginal_roundoff_tolerance_history": tf.stack(
            quotient_marginal_roundoff_tolerance_history, axis=1
        ),
        "quotient_marginal_valid_history": tf.stack(
            quotient_marginal_valid_history, axis=1
        ),
        "quotient_row_residual_scale_history": tf.stack(
            quotient_row_residual_scale_history, axis=1
        ),
        "quotient_column_residual_scale_history": tf.stack(
            quotient_column_residual_scale_history, axis=1
        ),
        "target_mean_history": tf.stack(target_mean_history, axis=1),
        "target_covariance_history": tf.stack(target_covariance_history, axis=1),
        "plus_mean_history": tf.stack(plus_mean_history, axis=1),
        "plus_covariance_history": tf.stack(plus_covariance_history, axis=1),
        "covariance_gap_history": tf.stack(covariance_gap_history, axis=1),
        "covariance_gap_eigenvalue_history": tf.stack(
            covariance_gap_eigenvalue_history, axis=1
        ),
        "output_mean_history": tf.stack(output_mean_history, axis=1),
        "output_covariance_history": tf.stack(output_covariance_history, axis=1),
        "injected_covariance_history": tf.stack(
            injected_covariance_history, axis=1
        ),
        "reset_affine_history": tf.stack(affine_history, axis=1),
        "ridged_identity_residual_history": tf.stack(
            ridged_identity_residual_history, axis=1
        ),
        "ridged_identity_scale_history": tf.stack(
            ridged_identity_scale_history, axis=1
        ),
        "ridged_identity_residual_fro_history": tf.linalg.norm(
            tf.stack(ridged_identity_residual_history, axis=1), axis=[-2, -1]
        ),
        "raw_covariance_residual_history": tf.stack(
            raw_covariance_residual_history, axis=1
        ),
        "predicted_raw_covariance_residual_history": tf.stack(
            predicted_raw_covariance_residual_history, axis=1
        ),
        "raw_covariance_prediction_error_history": tf.stack(
            raw_covariance_prediction_error_history, axis=1
        ),
        "raw_covariance_residual_fro_history": tf.linalg.norm(
            tf.stack(raw_covariance_residual_history, axis=1), axis=[-2, -1]
        ),
        "raw_covariance_prediction_error_fro_history": tf.linalg.norm(
            tf.stack(raw_covariance_prediction_error_history, axis=1),
            axis=[-2, -1],
        ),
        "mean_residual_history": tf.stack(mean_residual_history, axis=1),
        "mean_residual_infinity_history": tf.reduce_max(
            tf.abs(tf.stack(mean_residual_history, axis=1)), axis=-1
        ),
        "residual_design_sum_history": tf.stack(
            residual_design_sum_history, axis=1
        ),
        "residual_design_absolute_scale_history": tf.stack(
            residual_design_absolute_scale_history, axis=1
        ),
        "gap_chol_diagonal_history": tf.stack(gap_chol_diagonal_history, axis=1),
        "target_chol_diagonal_history": tf.stack(
            target_chol_diagonal_history, axis=1
        ),
        "injected_chol_diagonal_history": tf.stack(
            injected_chol_diagonal_history, axis=1
        ),
        "gap_condition_proxy_history": tf.stack(
            gap_condition_proxy_history, axis=1
        ),
        "target_condition_proxy_history": tf.stack(
            target_condition_proxy_history, axis=1
        ),
        "injected_condition_proxy_history": tf.stack(
            injected_condition_proxy_history, axis=1
        ),
        "realized_ridge_history": tf.stack(realized_ridge_history, axis=1),
    }


def _canonical_manual_jvp_core(
    theta: tf.Tensor,
    prepared: Mapping[str, tf.Tensor],
    *,
    steps: int,
    balance_steps: int,
    row_chunk_size: int,
    col_chunk_size: int,
) -> dict[str, tf.Tensor]:
    validate_transport_chunks(
        int(prepared["initial_noise"].shape[1]),
        row_chunk_size=row_chunk_size,
        col_chunk_size=col_chunk_size,
    )
    if balance_steps < 0:
        raise ValueError("balance_steps must be non-negative")
    initial_noise = prepared["initial_noise"]
    dtype = _canonical_dtype(initial_noise.dtype)
    theta = tf.reshape(tf.convert_to_tensor(theta, dtype), [PARAMETER_COUNT])
    batch_size = int(initial_noise.shape[0])
    particle_count = int(initial_noise.shape[1])
    time_steps = int(prepared["observations"].shape[0])
    components = _lgssm_components(theta, batch_size)
    tangents = _lgssm_component_tangents(theta, batch_size)
    particles = initial_noise * components["initial_std"][None, None, :]
    particles_tangent = (
        initial_noise[:, :, :, None] * tangents["d_initial_std"][None, None, :, :]
    )
    uniform_log_weight = -tf.math.log(tf.cast(particle_count, dtype))
    log_weights = tf.fill([batch_size, particle_count], uniform_log_weight)
    log_weights_tangent = tf.zeros([batch_size, particle_count, PARAMETER_COUNT], dtype)
    per_batch_score = tf.zeros([batch_size, PARAMETER_COUNT], dtype)
    zero_residual_tangent = tf.zeros(
        [batch_size, particle_count, STATE_DIMENSION, PARAMETER_COUNT], dtype
    )
    zero_ridge_tangent = tf.zeros([batch_size, PARAMETER_COUNT], dtype)

    for time_index in range(time_steps):
        observation = prepared["observations"][time_index]
        prior_mean = tf.einsum("bnj,bdj->bnd", particles, components["transition_matrix"])
        prior_mean_tangent = (
            tf.einsum(
                "bnjk,bdj->bndk", particles_tangent, components["transition_matrix"]
            )
            + tf.einsum(
                "bnj,bdjk->bndk", particles, tangents["d_transition_matrix"]
            )
        )
        noise = prepared["transition_noise"][:, time_index, :, :]
        pre_flow = prior_mean + components["q_scale"] * noise
        pre_flow_tangent = prior_mean_tangent + (
            noise[:, :, :, None] * tangents["d_transition_scale"][None, None, None, :]
        )
        flow = _lgssm_flow_forward_core(
            prior_mean,
            pre_flow,
            observation,
            components["transition_covariance"],
            components["observation_covariance"],
        )
        flow_tangent = _lgssm_flow_jvp_core(
            prior_mean,
            pre_flow,
            observation,
            components["transition_covariance"],
            components["observation_covariance"],
            prior_mean_tangent,
            pre_flow_tangent,
            tangents["d_transition_covariance"],
            tangents["d_observation_covariance"],
        )
        transition_density = _gaussian_log_density_jvp_core(
            flow["particles"] - prior_mean,
            components["transition_covariance"],
            flow_tangent["particles"] - prior_mean_tangent,
            tangents["d_transition_covariance"],
        )
        proposal_density = _gaussian_log_density_jvp_core(
            pre_flow - prior_mean,
            components["transition_covariance"],
            pre_flow_tangent - prior_mean_tangent,
            tangents["d_transition_covariance"],
        )
        predicted_observation = tf.einsum(
            "md,bnd->bnm", components["observation_matrix"], flow["particles"]
        )
        predicted_observation_tangent = tf.einsum(
            "md,bndk->bnmk",
            components["observation_matrix"],
            flow_tangent["particles"],
        )
        observation_density = _gaussian_log_density_jvp_core(
            predicted_observation - observation[None, None, :],
            components["observation_covariance"],
            predicted_observation_tangent,
            tangents["d_observation_covariance"],
        )
        logits = (
            log_weights
            + transition_density["value"]
            + observation_density["value"]
            - proposal_density["value"]
            + flow["forward_log_abs_det"][:, None]
        )
        logits_tangent = (
            log_weights_tangent
            + transition_density["tangent"]
            + observation_density["tangent"]
            - proposal_density["tangent"]
            + flow_tangent["forward_log_abs_det"][:, None, :]
        )
        normalization = _normalize_log_weights_jvp_core(logits, logits_tangent)
        per_batch_score = per_batch_score + normalization["increment_tangent"]
        geometry = _geometry_jvp_core(flow["particles"], flow_tangent["particles"])
        contract_e = _contract_e_streaming_forward_core(
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
        contract_e_tangent = _contract_e_streaming_jvp_core(
            geometry["scaled_geometry"],
            flow["particles"],
            normalization["normalized_log_weights"],
            normalization["normalized_weights"],
            prepared["residual_design"][:, time_index, :, :],
            prepared["prepared_ridge"][:, time_index],
            geometry["scaled_geometry_tangent"],
            flow_tangent["particles"],
            normalization["normalized_log_weights_tangent"],
            normalization["normalized_weights_tangent"],
            zero_residual_tangent,
            zero_ridge_tangent,
            geometry["epsilon0_tangent"],
            prepared["epsilon"],
            geometry["epsilon0"],
            prepared["scaling"],
            steps=steps,
            balance_steps=balance_steps,
            row_chunk_size=row_chunk_size,
            col_chunk_size=col_chunk_size,
        )
        active = prepared["fixed_reset_mask"][:, time_index]
        particles = tf.where(
            active[:, None, None], contract_e["particles"], flow["particles"]
        )
        particles_tangent = tf.where(
            active[:, None, None, None],
            contract_e_tangent["particles"],
            flow_tangent["particles"],
        )
        log_weights = tf.where(
            active[:, None],
            tf.fill([batch_size, particle_count], uniform_log_weight),
            normalization["normalized_log_weights"],
        )
        log_weights_tangent = tf.where(
            active[:, None, None],
            tf.zeros_like(normalization["normalized_log_weights_tangent"]),
            normalization["normalized_log_weights_tangent"],
        )
    return {
        "per_batch_score": per_batch_score,
        "score": tf.reduce_mean(per_batch_score, axis=0),
        "final_particles_tangent": particles_tangent,
        "final_log_weights_tangent": log_weights_tangent,
    }


def _canonical_fused_step_core(
    time_index: tf.Tensor,
    particles: tf.Tensor,
    particles_tangent: tf.Tensor,
    log_weights: tf.Tensor,
    log_weights_tangent: tf.Tensor,
    components: Mapping[str, tf.Tensor],
    tangents: Mapping[str, tf.Tensor],
    prepared: Mapping[str, tf.Tensor],
    *,
    steps: int,
    balance_steps: int,
    row_chunk_size: int,
    col_chunk_size: int,
    execute_contract_e: bool,
    cache_same_cloud_geometry: bool = False,
) -> dict[str, tf.Tensor]:
    """Advance one finite LGSSM value/score step with shared primal state."""

    dtype = particles.dtype
    batch_size = tf.shape(particles)[0]
    particle_count = tf.shape(particles)[1]
    observation = prepared["observations"][time_index]
    prior_mean = tf.einsum("bnj,bdj->bnd", particles, components["transition_matrix"])
    prior_mean_tangent = (
        tf.einsum(
            "bnjk,bdj->bndk", particles_tangent, components["transition_matrix"]
        )
        + tf.einsum(
            "bnj,bdjk->bndk", particles, tangents["d_transition_matrix"]
        )
    )
    noise = prepared["transition_noise"][:, time_index, :, :]
    pre_flow = prior_mean + components["q_scale"] * noise
    pre_flow_tangent = prior_mean_tangent + (
        noise[:, :, :, None]
        * tangents["d_transition_scale"][None, None, None, :]
    )
    flow = _lgssm_flow_forward_core(
        prior_mean,
        pre_flow,
        observation,
        components["transition_covariance"],
        components["observation_covariance"],
    )
    flow_tangent = _lgssm_flow_jvp_core(
        prior_mean,
        pre_flow,
        observation,
        components["transition_covariance"],
        components["observation_covariance"],
        prior_mean_tangent,
        pre_flow_tangent,
        tangents["d_transition_covariance"],
        tangents["d_observation_covariance"],
    )
    transition_density = _gaussian_log_density_jvp_core(
        flow["particles"] - prior_mean,
        components["transition_covariance"],
        flow_tangent["particles"] - prior_mean_tangent,
        tangents["d_transition_covariance"],
    )
    proposal_density = _gaussian_log_density_jvp_core(
        pre_flow - prior_mean,
        components["transition_covariance"],
        pre_flow_tangent - prior_mean_tangent,
        tangents["d_transition_covariance"],
    )
    predicted_observation = tf.einsum(
        "md,bnd->bnm", components["observation_matrix"], flow["particles"]
    )
    predicted_observation_tangent = tf.einsum(
        "md,bndk->bnmk",
        components["observation_matrix"],
        flow_tangent["particles"],
    )
    observation_density = _gaussian_log_density_jvp_core(
        predicted_observation - observation[None, None, :],
        components["observation_covariance"],
        predicted_observation_tangent,
        tangents["d_observation_covariance"],
    )
    logits = (
        log_weights
        + transition_density["value"]
        + observation_density["value"]
        - proposal_density["value"]
        + flow["forward_log_abs_det"][:, None]
    )
    logits_tangent = (
        log_weights_tangent
        + transition_density["tangent"]
        + observation_density["tangent"]
        - proposal_density["tangent"]
        + flow_tangent["forward_log_abs_det"][:, None, :]
    )
    normalization = _normalize_log_weights_jvp_core(logits, logits_tangent)
    geometry = _geometry_jvp_core(flow["particles"], flow_tangent["particles"])
    active = prepared["fixed_reset_mask"][:, time_index]
    uniform_log_weight = -tf.math.log(tf.cast(particle_count, dtype))

    if execute_contract_e:
        zero_residual_tangent = tf.zeros(
            [batch_size, particle_count, STATE_DIMENSION, PARAMETER_COUNT], dtype
        )
        zero_ridge_tangent = tf.zeros([batch_size, PARAMETER_COUNT], dtype)
        contract_e = _contract_e_streaming_forward_jvp_core(
            geometry["scaled_geometry"],
            flow["particles"],
            normalization["normalized_log_weights"],
            normalization["normalized_weights"],
            prepared["residual_design"][:, time_index, :, :],
            prepared["prepared_ridge"][:, time_index],
            geometry["scaled_geometry_tangent"],
            flow_tangent["particles"],
            normalization["normalized_log_weights_tangent"],
            normalization["normalized_weights_tangent"],
            zero_residual_tangent,
            zero_ridge_tangent,
            geometry["epsilon0_tangent"],
            prepared["epsilon"],
            geometry["epsilon0"],
            prepared["scaling"],
            steps=steps,
            balance_steps=balance_steps,
            row_chunk_size=row_chunk_size,
            col_chunk_size=col_chunk_size,
            cache_same_cloud_geometry=cache_same_cloud_geometry,
        )
        quotient = contract_e["quotient"]
        reset = contract_e["reset"]
        reset_valid = (
            quotient["valid_chart"]
            & quotient["marginal_valid"]
            & reset["finite"]
            & reset["factor_diagonal_positive"]
        )
        next_particles = tf.where(
            active[:, None, None], contract_e["particles"], flow["particles"]
        )
        next_particles_tangent = tf.where(
            active[:, None, None, None],
            contract_e["particles_tangent"],
            flow_tangent["particles"],
        )
        quotient_valid = quotient["valid_chart"]
        minimum_mass = quotient["minimum_mass"]
        row_error = quotient["maximum_row_error"]
        post_column_error = quotient[
            "maximum_post_quotient_column_absolute_residual"
        ]
        roundoff_tolerance = quotient["marginal_roundoff_tolerance"]
        marginal_valid = quotient["marginal_valid"]
        tv_column_error = quotient["tv_column_error"]
        covariance_gap_eigenvalues = reset["gap_eigenvalues"]
        work = contract_e["work"]
    else:
        true_batch = tf.ones([batch_size], tf.bool)
        zero_batch = tf.zeros([batch_size], dtype)
        next_particles = flow["particles"]
        next_particles_tangent = flow_tangent["particles"]
        quotient_valid = true_batch
        reset_valid = true_batch
        minimum_mass = tf.fill([batch_size], tf.constant(float("inf"), dtype))
        row_error = zero_batch
        post_column_error = zero_batch
        roundoff_tolerance = zero_batch
        marginal_valid = true_batch
        tv_column_error = zero_batch
        covariance_gap_eigenvalues = tf.zeros(
            [batch_size, STATE_DIMENSION], dtype
        )
        work = {
            "sinkhorn_state_constructions": tf.zeros([], tf.int32),
            "terminal_balance_state_constructions": tf.zeros([], tf.int32),
            "transport_tile_sweeps": tf.zeros([], tf.int32),
            "marginal_tile_sweeps": tf.zeros([], tf.int32),
            "diagnostic_solver_reconstructions": tf.zeros([], tf.int32),
        }

    time_valid = flow["valid_chart"] & (
        ~active | (geometry["valid_chart"] & reset_valid)
    )
    next_log_weights = tf.where(
        active[:, None],
        tf.fill([batch_size, particle_count], uniform_log_weight),
        normalization["normalized_log_weights"],
    )
    next_log_weights_tangent = tf.where(
        active[:, None, None],
        tf.zeros_like(normalization["normalized_log_weights_tangent"]),
        normalization["normalized_log_weights_tangent"],
    )
    return {
        "particles": next_particles,
        "particles_tangent": next_particles_tangent,
        "weighted_source_particles": flow["particles"],
        "weighted_source_particles_tangent": flow_tangent["particles"],
        "normalized_weights": normalization["normalized_weights"],
        "normalized_weights_tangent": normalization["normalized_weights_tangent"],
        "log_weights": next_log_weights,
        "log_weights_tangent": next_log_weights_tangent,
        "increment": normalization["increment"],
        "increment_tangent": normalization["increment_tangent"],
        "time_valid": time_valid,
        "flow_valid": flow["valid_chart"],
        "geometry_valid": geometry["valid_chart"],
        "quotient_valid": quotient_valid,
        "reset_valid": reset_valid,
        "minimum_mass": minimum_mass,
        "row_error": row_error,
        "post_column_error": post_column_error,
        "roundoff_tolerance": roundoff_tolerance,
        "marginal_valid": marginal_valid,
        "tv_column_error": tv_column_error,
        "covariance_gap_eigenvalues": covariance_gap_eigenvalues,
        "diameter_mask": geometry["diameter_mask"],
        "maximum_mask": geometry["maximum_mask"],
        "minimum_mask": geometry["minimum_mask"],
        "epsilon0_floor_inactive": geometry["epsilon0_floor_inactive"],
        "sinkhorn_running_branch": _sinkhorn_schedule(
            geometry["epsilon0"], prepared["epsilon"], prepared["scaling"], steps
        ),
        "work": work,
    }


def _canonical_fused_loop_core(
    theta: tf.Tensor,
    prepared: Mapping[str, tf.Tensor],
    *,
    steps: int,
    balance_steps: int,
    row_chunk_size: int,
    col_chunk_size: int,
    execute_contract_e: bool,
    cache_same_cloud_geometry: bool = False,
) -> dict[str, tf.Tensor]:
    """Evaluate value and total score with one functional time recursion."""

    validate_transport_chunks(
        int(prepared["initial_noise"].shape[1]),
        row_chunk_size=row_chunk_size,
        col_chunk_size=col_chunk_size,
    )
    initial_noise = prepared["initial_noise"]
    dtype = _canonical_dtype(initial_noise.dtype)
    theta = tf.reshape(tf.convert_to_tensor(theta, dtype), [PARAMETER_COUNT])
    batch_size = int(initial_noise.shape[0])
    particle_count = int(initial_noise.shape[1])
    time_steps = int(prepared["observations"].shape[0])
    components = _lgssm_components(theta, batch_size)
    tangents = _lgssm_component_tangents(theta, batch_size)
    particles = initial_noise * components["initial_std"][None, None, :]
    particles_tangent = (
        initial_noise[:, :, :, None]
        * tangents["d_initial_std"][None, None, :, :]
    )
    uniform_log_weight = -tf.math.log(tf.cast(particle_count, dtype))
    log_weights = tf.fill([batch_size, particle_count], uniform_log_weight)
    log_weights_tangent = tf.zeros(
        [batch_size, particle_count, PARAMETER_COUNT], dtype
    )
    per_batch_log_likelihood = tf.zeros([batch_size], dtype)
    per_batch_score = tf.zeros([batch_size, PARAMETER_COUNT], dtype)
    valid_chart = tf.fill([batch_size], _physical_chart(theta))
    minimum_mass = tf.fill([batch_size], tf.constant(float("inf"), dtype))

    history_specs = (
        (tf.bool, [batch_size]),
        (tf.bool, [batch_size]),
        (tf.bool, [batch_size]),
        (tf.bool, [batch_size]),
        (dtype, [batch_size]),
        (dtype, [batch_size]),
        (dtype, [batch_size]),
        (dtype, [batch_size]),
        (tf.bool, [batch_size]),
        (dtype, [batch_size]),
        (dtype, [batch_size, STATE_DIMENSION]),
        (tf.bool, [batch_size, STATE_DIMENSION]),
        (tf.bool, [batch_size, particle_count, STATE_DIMENSION]),
        (tf.bool, [batch_size, particle_count, STATE_DIMENSION]),
        (tf.bool, [batch_size]),
        (tf.bool, [batch_size, steps]),
    )
    histories = tuple(
        tf.TensorArray(value_dtype, size=time_steps, element_shape=shape)
        for value_dtype, shape in history_specs
    )

    def cond(
        time_index: tf.Tensor,
        *_state: tf.Tensor | tf.TensorArray,
    ) -> tf.Tensor:
        return time_index < time_steps

    def body(
        time_index: tf.Tensor,
        current_particles: tf.Tensor,
        current_particles_tangent: tf.Tensor,
        current_log_weights: tf.Tensor,
        current_log_weights_tangent: tf.Tensor,
        current_likelihood: tf.Tensor,
        current_score: tf.Tensor,
        current_valid: tf.Tensor,
        current_minimum_mass: tf.Tensor,
        sinkhorn_count: tf.Tensor,
        balance_count: tf.Tensor,
        transport_count: tf.Tensor,
        marginal_count: tf.Tensor,
        diagnostic_reconstruction_count: tf.Tensor,
        *history_arrays: tf.TensorArray,
    ):
        step = _canonical_fused_step_core(
            time_index,
            current_particles,
            current_particles_tangent,
            current_log_weights,
            current_log_weights_tangent,
            components,
            tangents,
            prepared,
            steps=steps,
            balance_steps=balance_steps,
            row_chunk_size=row_chunk_size,
            col_chunk_size=col_chunk_size,
            execute_contract_e=execute_contract_e,
            cache_same_cloud_geometry=cache_same_cloud_geometry,
        )
        active = prepared["fixed_reset_mask"][:, time_index]
        active_mass = tf.where(
            active,
            step["minimum_mass"],
            tf.fill([batch_size], tf.constant(float("inf"), dtype)),
        )
        values = (
            step["flow_valid"],
            step["geometry_valid"],
            step["quotient_valid"],
            step["reset_valid"],
            step["increment"],
            step["row_error"],
            step["post_column_error"],
            step["roundoff_tolerance"],
            step["marginal_valid"],
            step["tv_column_error"],
            step["covariance_gap_eigenvalues"],
            step["diameter_mask"],
            step["maximum_mask"],
            step["minimum_mask"],
            step["epsilon0_floor_inactive"],
            step["sinkhorn_running_branch"],
        )
        next_histories = tuple(
            array.write(time_index, value)
            for array, value in zip(history_arrays, values, strict=True)
        )
        return (
            time_index + 1,
            step["particles"],
            step["particles_tangent"],
            step["log_weights"],
            step["log_weights_tangent"],
            current_likelihood + step["increment"],
            current_score + step["increment_tangent"],
            current_valid & step["time_valid"],
            tf.minimum(current_minimum_mass, active_mass),
            sinkhorn_count + step["work"]["sinkhorn_state_constructions"],
            balance_count + step["work"]["terminal_balance_state_constructions"],
            transport_count + step["work"]["transport_tile_sweeps"],
            marginal_count + step["work"]["marginal_tile_sweeps"],
            diagnostic_reconstruction_count
            + step["work"]["diagnostic_solver_reconstructions"],
            *next_histories,
        )

    result = tf.while_loop(
        cond,
        body,
        (
            tf.constant(0, tf.int32),
            particles,
            particles_tangent,
            log_weights,
            log_weights_tangent,
            per_batch_log_likelihood,
            per_batch_score,
            valid_chart,
            minimum_mass,
            tf.zeros([], tf.int32),
            tf.zeros([], tf.int32),
            tf.zeros([], tf.int32),
            tf.zeros([], tf.int32),
            tf.zeros([], tf.int32),
            *histories,
        ),
        maximum_iterations=time_steps,
    )
    (
        _,
        particles,
        particles_tangent,
        log_weights,
        log_weights_tangent,
        per_batch_log_likelihood,
        per_batch_score,
        valid_chart,
        minimum_mass,
        sinkhorn_count,
        balance_count,
        transport_count,
        marginal_count,
        diagnostic_reconstruction_count,
        *histories,
    ) = result
    stacked = tuple(tf.transpose(array.stack(), [1, 0, *range(2, array.stack().shape.rank)]) for array in histories)
    (
        flow_valid_history,
        geometry_valid_history,
        quotient_valid_history,
        reset_valid_history,
        increment_history,
        row_error_history,
        post_column_error_history,
        roundoff_tolerance_history,
        marginal_valid_history,
        tv_column_error_history,
        covariance_gap_eigenvalue_history,
        diameter_masks,
        maximum_masks,
        minimum_masks,
        epsilon0_branches,
        sinkhorn_branches,
    ) = stacked
    return {
        "objective": tf.reduce_mean(per_batch_log_likelihood),
        "per_batch_log_likelihood": per_batch_log_likelihood,
        "score": tf.reduce_mean(per_batch_score, axis=0),
        "per_batch_score": per_batch_score,
        "valid_chart": valid_chart,
        "minimum_mass": minimum_mass,
        "final_particles": particles,
        "final_particles_tangent": particles_tangent,
        "final_log_weights": log_weights,
        "final_log_weights_tangent": log_weights_tangent,
        "flow_valid_history": flow_valid_history,
        "geometry_valid_history": geometry_valid_history,
        "quotient_valid_history": quotient_valid_history,
        "reset_valid_history": reset_valid_history,
        "increment_history": increment_history,
        "active_reset_history": prepared["fixed_reset_mask"],
        "quotient_row_residual_history": row_error_history,
        "quotient_post_quotient_column_residual_history": post_column_error_history,
        "quotient_marginal_roundoff_tolerance_history": roundoff_tolerance_history,
        "quotient_marginal_valid_history": marginal_valid_history,
        "tv_column_error_history": tv_column_error_history,
        "maximum_row_error_history": row_error_history,
        "covariance_gap_eigenvalue_history": covariance_gap_eigenvalue_history,
        "diameter_max_mask": diameter_masks,
        "geometry_max_mask": maximum_masks,
        "geometry_min_mask": minimum_masks,
        "epsilon0_floor_inactive": epsilon0_branches,
        "sinkhorn_running_branch": sinkhorn_branches,
        "work_sinkhorn_state_constructions": sinkhorn_count,
        "work_terminal_balance_state_constructions": balance_count,
        "work_transport_tile_sweeps": transport_count,
        "work_marginal_tile_sweeps": marginal_count,
        "work_diagnostic_solver_reconstructions": diagnostic_reconstruction_count,
        "work_active_reset_rows": tf.reduce_sum(
            tf.cast(prepared["fixed_reset_mask"], tf.int32)
        ),
    }


def canonical_value_and_score_core(
    theta: tf.Tensor,
    prepared: Mapping[str, tf.Tensor],
    *,
    steps: int,
    balance_steps: int,
    row_chunk_size: int,
    col_chunk_size: int,
    cache_same_cloud_geometry: bool = False,
) -> dict[str, tf.Tensor]:
    if balance_steps <= 0:
        raise ValueError("canonical balance_steps must be positive")
    return _canonical_fused_loop_core(
        theta,
        prepared,
        steps=steps,
        balance_steps=balance_steps,
        row_chunk_size=row_chunk_size,
        col_chunk_size=col_chunk_size,
        execute_contract_e=True,
        cache_same_cloud_geometry=cache_same_cloud_geometry,
    )


def make_canonical_value_and_score_tf(
    prepared: Mapping[str, Any],
    *,
    steps: int,
    balance_steps: int,
    row_chunk_size: int,
    col_chunk_size: int,
    jit_compile: bool = True,
    dtype: tf.dtypes.DType = DTYPE,
    cache_same_cloud_geometry: bool = False,
):
    """Bind fixed prepared inputs into the one admissible value-and-score graph."""

    if min(steps, balance_steps, row_chunk_size, col_chunk_size) <= 0:
        raise ValueError(
            "steps, balance_steps, and chunk sizes must be positive"
        )
    dtype = _canonical_dtype(dtype)
    tensors = _as_prepared_tensors(prepared, dtype=dtype)
    validate_transport_chunks(
        int(tensors["initial_noise"].shape[1]),
        row_chunk_size=row_chunk_size,
        col_chunk_size=col_chunk_size,
    )
    static_reset_activity = tf.get_static_value(
        tf.reduce_any(tensors["fixed_reset_mask"])
    )
    if static_reset_activity is None:
        raise ValueError("fixed reset activity must be statically known")
    execute_contract_e = bool(static_reset_activity)

    @tf.function(
        input_signature=[tf.TensorSpec([PARAMETER_COUNT], dtype)],
        jit_compile=jit_compile,
        reduce_retracing=True,
    )
    def value_and_score(theta: tf.Tensor) -> dict[str, tf.Tensor]:
        return _canonical_fused_loop_core(
            theta,
            tensors,
            steps=steps,
            balance_steps=balance_steps,
            row_chunk_size=row_chunk_size,
            col_chunk_size=col_chunk_size,
            execute_contract_e=execute_contract_e,
            cache_same_cloud_geometry=cache_same_cloud_geometry,
        )

    return value_and_score


def make_canonical_prepared_value_and_score_tf(
    *,
    batch_size: int,
    time_steps: int,
    num_particles: int,
    steps: int,
    balance_steps: int,
    row_chunk_size: int,
    col_chunk_size: int,
    jit_compile: bool = True,
    dtype: tf.dtypes.DType = DTYPE,
    cache_same_cloud_geometry: bool = False,
):
    """Compile the canonical route once for fixed-shape prepared seed batches."""

    if min(
        batch_size,
        time_steps,
        num_particles,
        steps,
        balance_steps,
        row_chunk_size,
        col_chunk_size,
    ) <= 0:
        raise ValueError("canonical prepared-route dimensions and controls must be positive")
    dtype = _canonical_dtype(dtype)
    validate_transport_chunks(
        num_particles,
        row_chunk_size=row_chunk_size,
        col_chunk_size=col_chunk_size,
    )
    prepared_signature = {
        "observations": tf.TensorSpec([time_steps, OBSERVATION_DIMENSION], dtype),
        "initial_noise": tf.TensorSpec(
            [batch_size, num_particles, STATE_DIMENSION], dtype
        ),
        "transition_noise": tf.TensorSpec(
            [batch_size, time_steps, num_particles, STATE_DIMENSION], dtype
        ),
        "fixed_reset_mask": tf.TensorSpec([batch_size, time_steps], tf.bool),
        "residual_design": tf.TensorSpec(
            [batch_size, time_steps, num_particles, STATE_DIMENSION], dtype
        ),
        "prepared_ridge": tf.TensorSpec([batch_size, time_steps], dtype),
        "epsilon": tf.TensorSpec([], dtype),
        "scaling": tf.TensorSpec([], dtype),
    }

    @tf.function(
        input_signature=[tf.TensorSpec([PARAMETER_COUNT], dtype), prepared_signature],
        jit_compile=jit_compile,
        reduce_retracing=True,
    )
    def value_and_score(
        theta: tf.Tensor, prepared: Mapping[str, tf.Tensor]
    ) -> dict[str, tf.Tensor]:
        return _canonical_fused_loop_core(
            theta,
            prepared,
            steps=steps,
            balance_steps=balance_steps,
            row_chunk_size=row_chunk_size,
            col_chunk_size=col_chunk_size,
            execute_contract_e=True,
            cache_same_cloud_geometry=cache_same_cloud_geometry,
        )

    return value_and_score


__all__ = [
    "DTYPE",
    "OBSERVATION_DIMENSION",
    "PARAMETER_COUNT",
    "PARAMETER_NAMES",
    "STATE_DIMENSION",
    "canonical_value_and_score_core",
    "make_canonical_prepared_value_and_score_tf",
    "make_canonical_value_and_score_tf",
]
