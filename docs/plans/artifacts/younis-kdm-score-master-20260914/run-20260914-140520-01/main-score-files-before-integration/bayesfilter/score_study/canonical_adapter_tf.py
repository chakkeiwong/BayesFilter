"""Actual canonical LEDH endpoint for the Gaussian score-study fixture.

The route uses the shared canonical executor, its UKF covariance lifecycle,
Contract E reset and dual-cap GenUT correction. This module adds no flow,
transport, or correction reimplementation. It supplies analytical model and
initial-law tangents for one parameter direction.
"""
from functools import lru_cache

import tensorflow as tf

from bayesfilter.highdim.ledh_canonical_score_tf import (
    NonlinearScoreModel, canonical_value_and_analytical_score,
)
from bayesfilter.highdim.transport_chunk_policy import select_transport_chunk_size
from .gaussian_tf import chol_tangent, parameterized_model


@lru_cache(maxsize=16)
def make_canonical_kernel(dimension, observation_dimension, particles, horizon,
                          controls_tuple, dtype_name="float64", jit_compile=True):
    dtype = tf.as_dtype(dtype_name)
    d, o, N, T = dimension, observation_dimension, particles, horizon
    controls = dict(controls_tuple)
    if controls.get("reset_policy") != "contract_e":
        raise ValueError("score-study LEDH requires Contract E")
    for key in ("correction_steps", "pairwise_steps", "coordinate_cap"):
        if controls.get(key, 0) <= 0:
            raise ValueError(f"canonical dual-cap GenUT control is missing: {key}")
    # Validate the repository chunk policy before tracing. The small canonical
    # dense reset is eligible only on its policy-mandated one-block scope.
    chunk = select_transport_chunk_size(N)
    if chunk != N:
        raise ValueError("this shared dense canonical endpoint needs a streaming provider for N > 3000")

    @tf.function(input_signature=[tf.TensorSpec([6], dtype), tf.TensorSpec([6], dtype),
        tf.TensorSpec([T, o], dtype), tf.TensorSpec([N, d], dtype), tf.TensorSpec([T, N, d], dtype),
        tf.TensorSpec([N, d], dtype)], jit_compile=jit_compile)
    def kernel(theta, direction, observations, initial_noise, noise, reset_design):
        A, all_dA, H, all_dH, mean, all_dm, P, all_dP, Q, all_dQ, R, all_dR = parameterized_model(theta, d, o)
        dA, dH, dm, dP, dQ, dR = [tf.tensordot(direction, x, axes=1) for x in (all_dA, all_dH, all_dm, all_dP, all_dQ, all_dR)]
        L = tf.linalg.cholesky(P)
        dL = chol_tangent(L, dP)
        initial = mean + tf.einsum("ij,nj->ni", L, initial_noise)
        dinitial = dm + tf.einsum("ij,nj->ni", dL, initial_noise)
        model = NonlinearScoreModel(
            transition_mean_fn=lambda parameter, points: tf.einsum("ij,nj->ni", A, points),
            transition_mean_tangent_fn=lambda parameter, points, tangents: tf.einsum("ij,nj->ni", dA, points) + tf.einsum("ij,nj->ni", A, tangents),
            observation_fn=lambda points: tf.einsum("ij,nj->ni", H, points),
            observation_jacobian_fn=lambda points: tf.broadcast_to(H, [tf.shape(points)[0], o, d]),
            observation_tangent_fn=lambda points, tangents: tf.einsum("ij,nj->ni", dH, points) + tf.einsum("ij,nj->ni", H, tangents),
            process_covariance=Q, observation_covariance=R,
            observation_jacobian_tangent_fn=lambda points, tangents: tf.broadcast_to(dH, [tf.shape(points)[0], o, d]),
            process_covariance_tangent_fn=lambda parameter: dQ,
            observation_covariance_tangent_fn=lambda parameter: dR)
        value, directional_score = canonical_value_and_analytical_score(
            model, theta, initial, tf.broadcast_to(P, [N, d, d]), noise, observations,
            with_score=True, initial_state_tangent=dinitial,
            initial_covariance_tangent=tf.broadcast_to(dP, [N, d, d]),
            reset_design=reset_design, **controls)
        return value, tf.ensure_shape(directional_score, [1])[0]
    return kernel
