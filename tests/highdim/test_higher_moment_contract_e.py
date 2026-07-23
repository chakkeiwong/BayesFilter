from __future__ import annotations

import inspect

import numpy as np
import tensorflow as tf

from bayesfilter.highdim.cubature_genut_adapters import exact_transformed_sv_candidate_adapter
from bayesfilter.highdim.cubature_genut_candidate import gaussian_genut_design, replicate_positive_genut
from bayesfilter.highdim.cubature_genut_filter import finite_value_score
from bayesfilter.highdim.higher_moment_contract_e import higher_moment_shape_jvp


def _fixture(n: int = 24, d: int = 2):
    source = tf.random.stateless_normal([n, d], [1, 2], dtype=tf.float64)
    weights = tf.nn.softmax(tf.random.stateless_normal([n], [3, 4], dtype=tf.float64))
    points = tf.random.stateless_normal([n, d], [5, 6], dtype=tf.float64)
    source_tangent = tf.random.stateless_normal([n, d], [7, 8], dtype=tf.float64) * 0.02
    weights_tangent = tf.random.stateless_normal([n], [9, 10], dtype=tf.float64) * 0.001
    weights_tangent -= tf.reduce_sum(weights_tangent) * weights
    points_tangent = tf.random.stateless_normal([n, d], [11, 12], dtype=tf.float64) * 0.02
    return source, weights, points, source_tangent, weights_tangent, points_tangent


def test_higher_moment_shape_restores_weighted_mean_covariance_and_zero_parity():
    source, weights, points, source_tangent, weights_tangent, points_tangent = _fixture()
    result = higher_moment_shape_jvp(
        source,
        weights,
        source_tangent[:, :, None],
        weights_tangent[:, None],
        points,
        points_tangent[:, :, None],
        correction_steps=2,
        strength=0.02,
        floor=1e-6,
    )
    mean = tf.reduce_mean(result["particles"], axis=0)
    centered = result["particles"] - mean[None, :]
    covariance = tf.einsum("ni,nj->ij", centered, centered) / tf.cast(tf.shape(source)[0], tf.float64)
    target_mean = tf.reduce_sum(weights[:, None] * source, axis=0)
    target_centered = source - target_mean[None, :]
    target_covariance = tf.einsum("n,ni,nj->ij", weights, target_centered, target_centered)
    tf.debugging.assert_near(mean, target_mean, atol=1e-10, rtol=1e-10)
    tf.debugging.assert_near(covariance, target_covariance, atol=1e-10, rtol=1e-10)
    assert bool(result["valid"].numpy())

    zero = higher_moment_shape_jvp(
        source,
        weights,
        source_tangent[:, :, None],
        weights_tangent[:, None],
        points,
        points_tangent[:, :, None],
        correction_steps=0,
        strength=0.02,
        floor=1e-6,
    )
    np.testing.assert_array_equal(zero["particles"].numpy(), points.numpy())
    np.testing.assert_array_equal(zero["particles_tangent"].numpy(), points_tangent[:, :, None].numpy())


def test_higher_moment_manual_jvp_matches_independent_forward_accumulator():
    source, weights, points, source_tangent, weights_tangent, points_tangent = _fixture()

    def forward(source_value, weights_value, points_value):
        zeros_source = tf.zeros([24, 2, 1], tf.float64)
        zeros_weights = tf.zeros([24, 1], tf.float64)
        return higher_moment_shape_jvp(
            source_value,
            weights_value,
            zeros_source,
            zeros_weights,
            points_value,
            zeros_source,
            correction_steps=2,
            strength=0.02,
            floor=1e-6,
        )["particles"]

    with tf.autodiff.ForwardAccumulator(
        (source, weights, points), (source_tangent, weights_tangent, points_tangent)
    ) as accumulator:
        output = forward(source, weights, points)
    automatic = accumulator.jvp(output)
    manual = higher_moment_shape_jvp(
        source,
        weights,
        source_tangent[:, :, None],
        weights_tangent[:, None],
        points,
        points_tangent[:, :, None],
        correction_steps=2,
        strength=0.02,
        floor=1e-6,
    )["particles_tangent"][:, :, 0]
    tf.debugging.assert_near(automatic, manual, atol=1e-10, rtol=1e-10)


def test_finite_value_score_default_is_unchanged_and_candidate_is_finite():
    n = 12
    adapter = exact_transformed_sv_candidate_adapter()
    theta = tf.constant([0.2, -0.1], tf.float32)
    observations = tf.constant([[0.1], [-0.2], [0.3]], tf.float32)
    initial = tf.random.stateless_normal([n, 1], [41, 42], dtype=tf.float32)
    process = tf.random.stateless_normal([3, n, 1], [43, 44], dtype=tf.float32)
    design = replicate_positive_genut(gaussian_genut_design(dim=1), num_particles=n)
    baseline = finite_value_score(
        adapter, theta, observations, initial, process, design,
        transition_before_first_observation=False,
    )
    explicit_zero = finite_value_score(
        adapter, theta, observations, initial, process, design,
        transition_before_first_observation=False,
        higher_moment_correction_steps=0,
        higher_moment_strength=0.02,
    )
    tf.debugging.assert_equal(baseline[0], explicit_zero[0])
    tf.debugging.assert_equal(baseline[1], explicit_zero[1])
    candidate = finite_value_score(
        adapter, theta, observations, initial, process, design,
        transition_before_first_observation=False,
        higher_moment_correction_steps=1,
        higher_moment_strength=0.02,
    )
    assert bool(candidate[2]["program_valid"].numpy())
    assert bool(tf.reduce_all(tf.math.is_finite(candidate[0])).numpy())
    assert bool(tf.reduce_all(tf.math.is_finite(candidate[1])).numpy())


def test_higher_moment_module_has_no_numpy_or_runtime_autodiff():
    source = inspect.getsource(higher_moment_shape_jvp)
    assert "numpy" not in source
    assert "GradientTape" not in source
    assert "ForwardAccumulator" not in source

