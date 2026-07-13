from __future__ import annotations

import inspect
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import tensorflow as tf

from experiments.dpf_implementation.tf_tfp.resampling import annealed_transport_tf


DTYPE = tf.float64


def _fixture() -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    x = tf.reshape(
        tf.linspace(tf.constant(-0.35, DTYPE), tf.constant(0.45, DTYPE), 6),
        [1, 3, 2],
    )
    raw_logw = tf.constant([[-0.4, 0.1, 0.3]], dtype=DTYPE)
    log_alpha = raw_logw - tf.reduce_logsumexp(raw_logw, axis=1, keepdims=True)
    log_beta = -tf.math.log(tf.cast(tf.shape(x)[1], DTYPE)) * tf.ones_like(log_alpha)
    return log_alpha, log_beta, x


def test_total_streaming_transport_custom_gradient_has_no_local_tape() -> None:
    source = inspect.getsource(
        annealed_transport_tf._filterflow_manual_streaming_finite_transport_total_vjp  # noqa: SLF001
    )
    pullback_source = inspect.getsource(
        annealed_transport_tf._filterflow_manual_streaming_finite_transport_total_pullback  # noqa: SLF001
    )

    assert "GradientTape" not in source
    assert "ForwardAccumulator" not in source
    assert ".gradient(" not in source
    assert "_filterflow_manual_streaming_finite_transport_total_pullback" in source
    assert "GradientTape" not in pullback_source
    assert "ForwardAccumulator" not in pullback_source
    assert ".gradient(" not in pullback_source
    assert "_filterflow_streaming_finite_sinkhorn_potentials_vjp_total" in pullback_source


def test_total_transport_vjp_hot_helpers_avoid_scatter_update_path() -> None:
    helper_sources = {
        name: inspect.getsource(getattr(annealed_transport_tf, name))
        for name in (
            "_filterflow_streaming_softmin_vjp",
            "_filterflow_streaming_transport_from_potentials_vjp",
        )
    }

    for source in helper_sources.values():
        assert "GradientTape" not in source
        assert "ForwardAccumulator" not in source
        assert ".gradient(" not in source
        assert "_scatter_axis1_add_2d" not in source
        assert "_scatter_axis1_add_3d" not in source
        assert "tf.tensor_scatter_nd_add" not in source

    transport_source = helper_sources["_filterflow_streaming_transport_from_potentials_vjp"]
    assert "s_blocks = tf.TensorArray" in transport_source
    assert "particle_blocks = tf.TensorArray" in transport_source
    assert "query_blocks = tf.TensorArray" in transport_source
    assert "key_blocks = tf.TensorArray" in transport_source


def test_total_sinkhorn_potential_vjp_has_explicit_epsilon0_cotangent_path() -> None:
    source = inspect.getsource(
        annealed_transport_tf._filterflow_streaming_finite_sinkhorn_potentials_vjp_total  # noqa: SLF001
    )

    assert "return_epsilon=True" in source
    assert "d_epsilon0" in source
    assert "_match_epsilon_cotangent_shape" in source
    assert "_filterflow_streaming_same_points_softmin_vjp" in source
    assert "_filterflow_streaming_softmin_vjp(" not in source
    assert "GradientTape" not in source
    assert "ForwardAccumulator" not in source
    assert ".gradient(" not in source


def test_total_sinkhorn_potential_vjp_returns_finite_epsilon0_cotangent() -> None:
    old_dtype = annealed_transport_tf.DTYPE
    annealed_transport_tf.DTYPE = DTYPE
    try:
        log_alpha, log_beta, x = _fixture()
        upstream_alpha = tf.constant([[0.03, -0.02, 0.01]], dtype=DTYPE)
        upstream_beta = tf.constant([[-0.01, 0.025, -0.015]], dtype=DTYPE)
        epsilon = tf.constant(0.45, DTYPE)
        epsilon0 = tf.constant([0.9], DTYPE)
        scaling = tf.constant(0.8, DTYPE)

        cotangents = (
            annealed_transport_tf._filterflow_streaming_finite_sinkhorn_potentials_vjp_total(  # noqa: SLF001
                log_alpha,
                log_beta,
                x,
                upstream_alpha,
                upstream_beta,
                epsilon,
                epsilon0,
                scaling,
                steps=2,
                row_chunk_size=2,
                col_chunk_size=2,
            )
        )
    finally:
        annealed_transport_tf.DTYPE = old_dtype

    d_log_alpha, d_log_beta, d_x, d_epsilon0 = cotangents
    assert d_log_alpha.shape == log_alpha.shape
    assert d_log_beta.shape == log_beta.shape
    assert d_x.shape == x.shape
    assert d_epsilon0.shape == epsilon0.shape
    assert all(bool(tf.reduce_all(tf.math.is_finite(part)).numpy()) for part in cotangents)
    assert float(tf.reduce_max(tf.abs(d_epsilon0)).numpy()) > 0.0


def test_total_streaming_transport_custom_gradient_returns_finite_cotangents() -> None:
    old_dtype = annealed_transport_tf.DTYPE
    annealed_transport_tf.DTYPE = DTYPE
    try:
        logw, _uniform_logw, particles = _fixture()
        scaled_x = tf.identity(particles)
        epsilon = tf.constant(0.45, DTYPE)
        epsilon0 = tf.constant([0.9], DTYPE)
        scaling = tf.constant(0.8, DTYPE)
        upstream = tf.reshape(
            tf.linspace(tf.constant(-0.02, DTYPE), tf.constant(0.025, DTYPE), 6),
            [1, 3, 2],
        )

        with tf.GradientTape() as tape:
            tape.watch([scaled_x, particles, logw, epsilon0])
            transported, _ = (
                annealed_transport_tf._filterflow_manual_streaming_finite_transport_total_vjp(  # noqa: SLF001
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
            scalar = tf.reduce_sum(transported * upstream)
        cotangents = tape.gradient(
            scalar,
            [scaled_x, particles, logw, epsilon0],
            unconnected_gradients=tf.UnconnectedGradients.ZERO,
        )
    finally:
        annealed_transport_tf.DTYPE = old_dtype

    d_scaled_x, d_particles, d_logw, d_epsilon0 = cotangents
    assert d_scaled_x.shape == scaled_x.shape
    assert d_particles.shape == particles.shape
    assert d_logw.shape == logw.shape
    assert d_epsilon0.shape == epsilon0.shape
    assert all(bool(tf.reduce_all(tf.math.is_finite(part)).numpy()) for part in cotangents)
    assert float(tf.reduce_max(tf.abs(d_epsilon0)).numpy()) > 0.0
