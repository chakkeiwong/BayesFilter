from __future__ import annotations

import inspect
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import tensorflow as tf

from experiments.dpf_implementation.tf_tfp.resampling import annealed_transport_tf


DTYPE = tf.float64
ATOL = 2.0e-8


def _fixture() -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    particles = tf.reshape(
        tf.linspace(tf.constant(-0.35, DTYPE), tf.constant(0.45, DTYPE), 6),
        [1, 3, 2],
    )
    scaled_x = particles + tf.constant(
        [[[0.015, -0.010], [0.000, 0.012], [-0.012, 0.006]]],
        DTYPE,
    )
    raw_logw = tf.constant([[-0.25, 0.05, 0.18]], dtype=DTYPE)
    logw = raw_logw - tf.reduce_logsumexp(raw_logw, axis=1, keepdims=True)
    epsilon = tf.constant(0.45, DTYPE)
    epsilon0 = tf.constant([0.9], DTYPE)
    scaling = tf.constant(0.8, DTYPE)
    return scaled_x, particles, logw, epsilon, epsilon0, scaling


def _particle_tangent(value: tf.Tensor, *, scale: float, param_dim: int) -> tf.Tensor:
    count = int(tf.size(value).numpy())
    columns = []
    for index in range(param_dim):
        column = tf.reshape(
            tf.linspace(
                tf.constant(-scale * (index + 1), DTYPE),
                tf.constant(scale * (index + 1), DTYPE),
                count,
            ),
            tf.shape(value),
        )
        columns.append(column)
    return tf.stack(columns, axis=-1)


def _weight_tangent(value: tf.Tensor, *, scale: float, param_dim: int) -> tf.Tensor:
    count = int(tf.size(value).numpy())
    columns = []
    for index in range(param_dim):
        column = tf.reshape(
            tf.linspace(
                tf.constant(scale * (index + 1), DTYPE),
                tf.constant(-scale * (index + 1), DTYPE),
                count,
            ),
            tf.shape(value),
        )
        columns.append(column)
    return tf.stack(columns, axis=-1)


def test_compact_softmin_jvp_matches_tape_directional_oracle() -> None:
    old_dtype = annealed_transport_tf.DTYPE
    annealed_transport_tf.DTYPE = DTYPE
    try:
        scaled_x, _particles, logw, _epsilon, epsilon0, _scaling = _fixture()
        param_dim = 2
        d_x = _particle_tangent(scaled_x, scale=0.013, param_dim=param_dim)
        d_logw = _weight_tangent(logw, scale=0.017, param_dim=param_dim)
        d_epsilon0 = tf.constant([[0.019, -0.011]], dtype=DTYPE)
        value, tangent = annealed_transport_tf._filterflow_streaming_softmin_jvp(  # noqa: SLF001
            epsilon0,
            scaled_x,
            scaled_x,
            logw,
            d_epsilon0,
            d_x,
            d_x,
            d_logw,
            row_chunk_size=2,
            col_chunk_size=2,
        )
        upstream = tf.reshape(
            tf.linspace(tf.constant(-0.04, DTYPE), tf.constant(0.03, DTYPE), 3),
            [1, 3],
        )
        manual_directional = tf.reduce_sum(tangent * upstream[:, :, None], axis=[0, 1])
        tape_directionals = []
        for index in range(param_dim):
            with tf.GradientTape() as tape:
                tape.watch([scaled_x, logw, epsilon0])
                softmin_value = annealed_transport_tf._filterflow_streaming_softmin(  # noqa: SLF001
                    epsilon0,
                    scaled_x,
                    scaled_x,
                    logw,
                    row_chunk_size=2,
                    col_chunk_size=2,
                )
                scalar = tf.reduce_sum(softmin_value * upstream)
            grads = tape.gradient(
                scalar,
                [scaled_x, logw, epsilon0],
                unconnected_gradients=tf.UnconnectedGradients.ZERO,
            )
            tape_directionals.append(
                tf.reduce_sum(grads[0] * d_x[..., index])
                + tf.reduce_sum(grads[1] * d_logw[..., index])
                + tf.reduce_sum(grads[2] * d_epsilon0[..., index])
            )
        tape_directional = tf.stack(tape_directionals)
    finally:
        annealed_transport_tf.DTYPE = old_dtype

    tf.debugging.assert_near(
        value,
        annealed_transport_tf._filterflow_streaming_softmin(  # noqa: SLF001
            epsilon0,
            scaled_x,
            scaled_x,
            logw,
            row_chunk_size=2,
            col_chunk_size=2,
        ),
        atol=ATOL,
    )
    tf.debugging.assert_near(manual_directional, tape_directional, atol=ATOL)


def test_half_pairwise_squared_cross_jvp_matches_broadcast_reference() -> None:
    old_dtype = annealed_transport_tf.DTYPE
    annealed_transport_tf.DTYPE = DTYPE
    try:
        scaled_x, _particles, _logw, _epsilon, _epsilon0, _scaling = _fixture()
        query = scaled_x[:, :2, :]
        key = scaled_x[:, 1:, :]
        d_query = _particle_tangent(query, scale=0.011, param_dim=3)
        d_key = _particle_tangent(key, scale=0.007, param_dim=3)
        actual = annealed_transport_tf._half_pairwise_squared_cross_jvp(  # noqa: SLF001
            query,
            key,
            d_query,
            d_key,
        )
        diff = query[:, :, None, :] - key[:, None, :, :]
        d_diff = d_query[:, :, None, :, :] - d_key[:, None, :, :, :]
        expected = tf.reduce_sum(diff[:, :, :, :, None] * d_diff, axis=3)
    finally:
        annealed_transport_tf.DTYPE = old_dtype

    tf.debugging.assert_near(actual, expected, atol=ATOL)


def test_compact_total_transport_jvp_matches_tape_directional_oracle() -> None:
    old_dtype = annealed_transport_tf.DTYPE
    annealed_transport_tf.DTYPE = DTYPE
    try:
        scaled_x, particles, logw, epsilon, epsilon0, scaling = _fixture()
        param_dim = 2
        d_scaled_x = _particle_tangent(scaled_x, scale=0.010, param_dim=param_dim)
        d_particles = _particle_tangent(particles, scale=0.008, param_dim=param_dim)
        d_logw = _weight_tangent(logw, scale=0.014, param_dim=param_dim)
        d_epsilon0 = tf.constant([[0.013, -0.009]], dtype=DTYPE)
        transported, d_transported, row_residual = (
            annealed_transport_tf._filterflow_manual_streaming_finite_transport_value_and_jvp_total(  # noqa: SLF001
                scaled_x,
                particles,
                logw,
                d_scaled_x,
                d_particles,
                d_logw,
                d_epsilon0,
                epsilon,
                epsilon0,
                scaling,
                steps=2,
                row_chunk_size=2,
                col_chunk_size=2,
            )
        )
        upstream = tf.reshape(
            tf.linspace(tf.constant(-0.03, DTYPE), tf.constant(0.025, DTYPE), 6),
            [1, 3, 2],
        )
        manual_directional = tf.reduce_sum(
            d_transported * upstream[:, :, :, None],
            axis=[0, 1, 2],
        )
        tape_directionals = []
        for index in range(param_dim):
            with tf.GradientTape() as tape:
                tape.watch([scaled_x, particles, logw, epsilon0])
                value, _ = (
                    annealed_transport_tf._filterflow_manual_streaming_finite_transport_value_total_vjp(  # noqa: SLF001
                        scaled_x,
                        particles,
                        logw,
                        epsilon,
                        epsilon0,
                        scaling,
                        steps=2,
                        row_chunk_size=2,
                        col_chunk_size=2,
                    )
                )
                scalar = tf.reduce_sum(value * upstream)
            grads = tape.gradient(
                scalar,
                [scaled_x, particles, logw, epsilon0],
                unconnected_gradients=tf.UnconnectedGradients.ZERO,
            )
            tape_directionals.append(
                tf.reduce_sum(grads[0] * d_scaled_x[..., index])
                + tf.reduce_sum(grads[1] * d_particles[..., index])
                + tf.reduce_sum(grads[2] * d_logw[..., index])
                + tf.reduce_sum(grads[3] * d_epsilon0[..., index])
            )
        tape_directional = tf.stack(tape_directionals)
    finally:
        annealed_transport_tf.DTYPE = old_dtype

    reference, reference_residual = (
        annealed_transport_tf._filterflow_manual_streaming_finite_transport_value_total_vjp(  # noqa: SLF001
            scaled_x,
            particles,
            logw,
            epsilon,
            epsilon0,
            scaling,
            steps=2,
            row_chunk_size=2,
            col_chunk_size=2,
        )
    )
    tf.debugging.assert_near(transported, reference, atol=ATOL)
    tf.debugging.assert_near(row_residual, reference_residual, atol=ATOL)
    tf.debugging.assert_near(manual_directional, tape_directional, atol=ATOL)


def test_compact_transport_jvp_production_helpers_have_no_tape() -> None:
    helper_names = (
        "_filterflow_streaming_softmin_jvp",
        "_filterflow_streaming_finite_sinkhorn_potentials_jvp_total",
        "_filterflow_streaming_transport_from_potentials_jvp",
        "_filterflow_manual_streaming_finite_transport_value_and_jvp_total",
    )
    for name in helper_names:
        source = inspect.getsource(getattr(annealed_transport_tf, name))
        assert "GradientTape" not in source
        assert "ForwardAccumulator" not in source
        assert ".gradient(" not in source


def test_compact_transport_jvp_avoids_known_large_broadcast_temporaries() -> None:
    source = inspect.getsource(
        annealed_transport_tf._filterflow_streaming_transport_from_potentials_jvp  # noqa: SLF001
    )
    assert "d_weighted" not in source
    assert "transport_block[:, :, :, None, None]" not in source
    assert "d_particle_block[:, None, :, :, :]" not in source

    helper_sources = "\n".join(
        inspect.getsource(getattr(annealed_transport_tf, name))
        for name in (
            "_filterflow_streaming_softmin_jvp",
            "_filterflow_streaming_column_log_normalizer_jvp",
            "_filterflow_streaming_transport_from_potentials_jvp",
        )
    )
    assert "d_diff = d_query_block[:, :, None, :, :] - d_key_block[:, None, :, :, :]" not in helper_sources
