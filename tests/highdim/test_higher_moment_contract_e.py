from __future__ import annotations

import inspect
import math

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.highdim.cubature_genut_adapters import exact_transformed_sv_candidate_adapter
from bayesfilter.highdim.cubature_genut_candidate import gaussian_genut_design, replicate_positive_genut
from bayesfilter.highdim.cubature_genut_filter import (
    BoundedFeatureShapeTeacher,
    finite_value_score,
)
from bayesfilter.highdim.higher_moment_contract_e import (
    affine_restore_cloud_jvp,
    higher_moment_shape_jvp,
    weighted_shape_targets_jvp,
)
from docs.benchmarks.run_projected_cumulant_genut_austria import (
    _canonical_basis,
    _partition_summary,
    _validity_checks,
)


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


def test_zero_step_reports_actual_moment_mismatch():
    source, weights, points, source_tangent, weights_tangent, points_tangent = _fixture()
    result = higher_moment_shape_jvp(
        source,
        weights,
        source_tangent[:, :, None],
        weights_tangent[:, None],
        points,
        points_tangent[:, :, None],
        correction_steps=0,
        strength=0.0,
        floor=1e-6,
    )
    assert float(tf.reduce_sum(tf.abs(result["skew_residual"])).numpy()) > 0.0
    assert float(tf.reduce_sum(tf.abs(result["kurtosis_residual"])).numpy()) > 0.0
    tf.debugging.assert_near(result["target_skew"], result["target_skew"])
    tf.debugging.assert_near(result["target_kurtosis"], result["target_kurtosis"])


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


def test_pairwise_moment_shape_reduces_pair_residual_and_preserves_full_covariance():
    n = 96
    base = tf.random.stateless_normal([n, 3], [21, 22], dtype=tf.float64)
    source = tf.stack(
        [
            base[:, 0],
            0.7 * tf.square(base[:, 0]) + 0.4 * base[:, 1],
            0.6 * base[:, 0] * base[:, 1] + base[:, 2],
        ],
        axis=1,
    )
    weights = tf.nn.softmax(
        0.2 * tf.random.stateless_normal([n], [23, 24], dtype=tf.float64)
    )
    points = tf.random.stateless_normal([n, 3], [25, 26], dtype=tf.float64)
    zeros_points = tf.zeros([n, 3, 1], tf.float64)
    zeros_weights = tf.zeros([n, 1], tf.float64)
    baseline = higher_moment_shape_jvp(
        source,
        weights,
        zeros_points,
        zeros_weights,
        points,
        zeros_points,
        correction_steps=2,
        strength=0.02,
        floor=1e-6,
    )
    candidate = higher_moment_shape_jvp(
        source,
        weights,
        zeros_points,
        zeros_weights,
        points,
        zeros_points,
        correction_steps=2,
        strength=0.02,
        floor=1e-6,
        pairwise_correction_steps=4,
        pairwise_strength=0.02,
        pairwise_floor=1e-6,
    )
    baseline_pair_loss = tf.reduce_sum(
        tf.square(baseline["pairwise_co_skew_residual"])
    ) + tf.reduce_sum(tf.square(baseline["pairwise_co_kurtosis_residual"]))
    candidate_pair_loss = tf.reduce_sum(
        tf.square(candidate["pairwise_co_skew_residual"])
    ) + tf.reduce_sum(tf.square(candidate["pairwise_co_kurtosis_residual"]))
    assert float(candidate_pair_loss.numpy()) < float(baseline_pair_loss.numpy())

    output_mean = tf.reduce_mean(candidate["particles"], axis=0)
    output_centered = candidate["particles"] - output_mean[None, :]
    output_covariance = tf.einsum(
        "ni,nj->ij", output_centered, output_centered
    ) / tf.cast(n, tf.float64)
    target_mean = tf.reduce_sum(weights[:, None] * source, axis=0)
    target_centered = source - target_mean[None, :]
    target_covariance = tf.einsum(
        "n,ni,nj->ij", weights, target_centered, target_centered
    )
    tf.debugging.assert_near(output_mean, target_mean, atol=1e-10, rtol=1e-10)
    tf.debugging.assert_near(
        output_covariance, target_covariance, atol=1e-10, rtol=1e-10
    )


def test_pairwise_moment_manual_jvp_matches_independent_forward_accumulator():
    source, weights, points, source_tangent, weights_tangent, points_tangent = _fixture(
        n=24, d=2
    )

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
            pairwise_correction_steps=2,
            pairwise_strength=0.01,
            pairwise_floor=1e-6,
        )["particles"]

    with tf.autodiff.ForwardAccumulator(
        (source, weights, points),
        (source_tangent, weights_tangent, points_tangent),
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
        pairwise_correction_steps=2,
        pairwise_strength=0.01,
        pairwise_floor=1e-6,
    )["particles_tangent"][:, :, 0]
    tf.debugging.assert_near(automatic, manual, atol=2e-10, rtol=2e-10)


def test_pairwise_particle_rms_cap_is_bounded_and_restores_affine_moments():
    n = 96
    d = 3
    source, weights, points, source_tangent, weights_tangent, points_tangent = _fixture(
        n=n, d=d
    )
    cap = 0.75
    common = dict(
        correction_steps=2,
        strength=0.02,
        floor=1e-6,
        pairwise_correction_steps=3,
        pairwise_strength=0.02,
        pairwise_floor=1e-6,
    )
    uncapped = higher_moment_shape_jvp(
        source,
        weights,
        source_tangent[:, :, None],
        weights_tangent[:, None],
        points,
        points_tangent[:, :, None],
        **common,
    )
    explicit_disabled = higher_moment_shape_jvp(
        source,
        weights,
        source_tangent[:, :, None],
        weights_tangent[:, None],
        points,
        points_tangent[:, :, None],
        pairwise_particle_rms_cap=0.0,
        **common,
    )
    capped = higher_moment_shape_jvp(
        source,
        weights,
        source_tangent[:, :, None],
        weights_tangent[:, None],
        points,
        points_tangent[:, :, None],
        pairwise_particle_rms_cap=cap,
        **common,
    )
    tf.debugging.assert_equal(uncapped["particles"], explicit_disabled["particles"])
    tf.debugging.assert_equal(
        uncapped["particles_tangent"], explicit_disabled["particles_tangent"]
    )
    assert float(capped["maximum_pairwise_post_cap_particle_rms"].numpy()) < cap
    assert float(capped["maximum_pairwise_post_cap_particle_rms"].numpy()) < float(
        capped["maximum_pairwise_pre_cap_particle_rms"].numpy()
    )
    assert float(capped["minimum_pairwise_particle_cap_scale"].numpy()) < 1.0

    output_mean = tf.reduce_mean(capped["particles"], axis=0)
    output_centered = capped["particles"] - output_mean[None, :]
    output_covariance = tf.einsum(
        "ni,nj->ij", output_centered, output_centered
    ) / tf.cast(n, tf.float64)
    target_mean = tf.reduce_sum(weights[:, None] * source, axis=0)
    target_centered = source - target_mean[None, :]
    target_covariance = tf.einsum(
        "n,ni,nj->ij", weights, target_centered, target_centered
    )
    tf.debugging.assert_near(output_mean, target_mean, atol=2e-10, rtol=2e-10)
    tf.debugging.assert_near(
        output_covariance, target_covariance, atol=2e-10, rtol=2e-10
    )


def test_pairwise_particle_rms_cap_manual_jvp_matches_forward_accumulator():
    source, weights, points, source_tangent, weights_tangent, points_tangent = _fixture(
        n=24, d=3
    )

    def forward(source_value, weights_value, points_value):
        zeros_source = tf.zeros([24, 3, 1], tf.float64)
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
            pairwise_correction_steps=2,
            pairwise_strength=0.01,
            pairwise_floor=1e-6,
            pairwise_particle_rms_cap=0.8,
        )["particles"]

    with tf.autodiff.ForwardAccumulator(
        (source, weights, points),
        (source_tangent, weights_tangent, points_tangent),
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
        pairwise_correction_steps=2,
        pairwise_strength=0.01,
        pairwise_floor=1e-6,
        pairwise_particle_rms_cap=0.8,
    )["particles_tangent"][:, :, 0]
    tf.debugging.assert_near(automatic, manual, atol=3e-10, rtol=3e-10)


def test_coordinatewise_bounded_cap_is_strictly_bounded_and_tail_selective():
    source, weights, points, source_tangent, weights_tangent, points_tangent = _fixture(
        n=48, d=3
    )
    uncapped = higher_moment_shape_jvp(
        source,
        weights,
        source_tangent[:, :, None],
        weights_tangent[:, None],
        points,
        points_tangent[:, :, None],
        correction_steps=1,
        strength=0.02,
        floor=1e-6,
        coordinatewise_bounded_cap=0.0,
        coordinatewise_bounded_cap_power=8,
    )
    capped = higher_moment_shape_jvp(
        source,
        weights,
        source_tangent[:, :, None],
        weights_tangent[:, None],
        points,
        points_tangent[:, :, None],
        correction_steps=1,
        strength=0.02,
        floor=1e-6,
        coordinatewise_bounded_cap=0.9,
        coordinatewise_bounded_cap_power=8,
    )
    assert float(capped["maximum_coordinatewise_post_cap_absolute"].numpy()) < 0.9
    assert float(capped["mean_coordinatewise_cap_displacement"].numpy()) >= 0.0
    assert float(capped["fraction_coordinatewise_cap_active"].numpy()) >= 0.0
    assert float(capped["minimum_coordinatewise_cap_derivative"].numpy()) > 0.0
    assert bool(capped["valid"].numpy())
    assert float(capped["mean_coordinatewise_cap_displacement"].numpy()) > 0.0
    assert float(capped["maximum_coordinatewise_post_cap_absolute"].numpy()) < float(
        uncapped["maximum_coordinatewise_pre_cap_absolute"].numpy()
    )
    output = capped["particles"]
    output_mean = tf.reduce_mean(output, axis=0)
    output_centered = output - output_mean[None, :]
    output_covariance = tf.einsum("ni,nj->ij", output_centered, output_centered) / tf.cast(
        tf.shape(output)[0], output.dtype
    )
    output_chol = tf.linalg.cholesky(output_covariance)
    output_standardized = tf.transpose(
        tf.linalg.triangular_solve(
            output_chol, tf.transpose(output_centered), lower=True
        )
    )
    expected_skew = tf.reduce_mean(tf.pow(output_standardized, 3.0), axis=0)
    expected_kurtosis = tf.reduce_mean(tf.pow(output_standardized, 4.0), axis=0)
    tf.debugging.assert_near(
        capped["target_skew"] - capped["skew_residual"], expected_skew, atol=1e-10
    )
    tf.debugging.assert_near(
        capped["target_kurtosis"] - capped["kurtosis_residual"],
        expected_kurtosis,
        atol=1e-10,
    )


def test_coordinatewise_bounded_cap_manual_jvp_matches_forward_accumulator():
    source, weights, points, source_tangent, weights_tangent, points_tangent = _fixture(
        n=32, d=3
    )

    def forward(source_value, weights_value, points_value, cap):
        zeros_source = tf.zeros([32, 3, 1], tf.float64)
        zeros_weights = tf.zeros([32, 1], tf.float64)
        return higher_moment_shape_jvp(
            source_value,
            weights_value,
            zeros_source,
            zeros_weights,
            points_value,
            zeros_source,
            correction_steps=1,
            strength=0.02,
            floor=1e-6,
            coordinatewise_bounded_cap=cap,
            coordinatewise_bounded_cap_power=8,
        )["particles"]

    with tf.autodiff.ForwardAccumulator(
        (source, weights, points),
        (source_tangent, weights_tangent, points_tangent),
    ) as accumulator:
        output = forward(source, weights, points, 0.9)
    automatic = accumulator.jvp(output)
    manual = higher_moment_shape_jvp(
        source,
        weights,
        source_tangent[:, :, None],
        weights_tangent[:, None],
        points,
        points_tangent[:, :, None],
        correction_steps=1,
        strength=0.02,
        floor=1e-6,
        coordinatewise_bounded_cap=0.9,
        coordinatewise_bounded_cap_power=8,
    )["particles_tangent"][:, :, 0]
    tf.debugging.assert_near(automatic, manual, atol=4e-9, rtol=4e-9)

    with tf.autodiff.ForwardAccumulator(
        (source, weights, points),
        (source_tangent, weights_tangent, points_tangent),
    ) as accumulator:
        output = forward(source, weights, points, 0.0)
    automatic_disabled = accumulator.jvp(output)
    manual_disabled = higher_moment_shape_jvp(
        source,
        weights,
        source_tangent[:, :, None],
        weights_tangent[:, None],
        points,
        points_tangent[:, :, None],
        correction_steps=1,
        strength=0.02,
        floor=1e-6,
        coordinatewise_bounded_cap=0.0,
        coordinatewise_bounded_cap_power=8,
    )["particles_tangent"][:, :, 0]
    tf.debugging.assert_near(
        automatic_disabled, manual_disabled, atol=4e-9, rtol=4e-9
    )


def test_coordinatewise_standardized_cap_restores_affine_moments_and_is_opt_in():
    source, weights, points, source_tangent, weights_tangent, points_tangent = _fixture(
        n=48, d=3
    )
    common = dict(
        correction_steps=2,
        strength=0.02,
        floor=1e-6,
        pairwise_correction_steps=2,
        pairwise_strength=0.01,
    )
    baseline = higher_moment_shape_jvp(
        source,
        weights,
        source_tangent[:, :, None],
        weights_tangent[:, None],
        points,
        points_tangent[:, :, None],
        **common,
    )
    disabled = higher_moment_shape_jvp(
        source,
        weights,
        source_tangent[:, :, None],
        weights_tangent[:, None],
        points,
        points_tangent[:, :, None],
        coordinatewise_standardized_cap=0.0,
        **common,
    )
    tf.debugging.assert_equal(disabled["particles"], baseline["particles"])
    tf.debugging.assert_equal(
        disabled["particles_tangent"], baseline["particles_tangent"]
    )

    capped = higher_moment_shape_jvp(
        source,
        weights,
        source_tangent[:, :, None],
        weights_tangent[:, None],
        points,
        points_tangent[:, :, None],
        coordinatewise_standardized_cap=0.98,
        **common,
    )
    output_mean = tf.reduce_mean(capped["particles"], axis=0)
    output_centered = capped["particles"] - output_mean[None, :]
    output_covariance = tf.einsum(
        "ni,nj->ij", output_centered, output_centered
    ) / tf.cast(tf.shape(source)[0], tf.float64)
    target_mean = tf.reduce_sum(weights[:, None] * source, axis=0)
    target_centered = source - target_mean[None, :]
    target_covariance = tf.einsum(
        "n,ni,nj->ij", weights, target_centered, target_centered
    )
    tf.debugging.assert_near(output_mean, target_mean, atol=1e-10, rtol=1e-10)
    tf.debugging.assert_near(
        output_covariance, target_covariance, atol=1e-10, rtol=1e-10
    )
    assert bool(capped["valid"].numpy())


def test_coordinatewise_standardized_cap_manual_jvp_matches_forward_accumulator():
    source, weights, points, source_tangent, weights_tangent, points_tangent = _fixture(
        n=32, d=3
    )

    def forward(source_value, weights_value, points_value):
        zeros_source = tf.zeros([32, 3, 1], tf.float64)
        zeros_weights = tf.zeros([32, 1], tf.float64)
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
            pairwise_correction_steps=2,
            pairwise_strength=0.01,
            coordinatewise_standardized_cap=0.98,
        )["particles"]

    with tf.autodiff.ForwardAccumulator(
        (source, weights, points),
        (source_tangent, weights_tangent, points_tangent),
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
        pairwise_correction_steps=2,
        pairwise_strength=0.01,
        coordinatewise_standardized_cap=0.98,
    )["particles_tangent"][:, :, 0]
    tf.debugging.assert_near(automatic, manual, atol=8e-9, rtol=8e-9)


def test_projected_cumulant_reduces_complete_tensor_residual_and_restores_affine_moments():
    n = 96
    d = 4
    source_base = tf.random.stateless_normal([n, d], [71, 72], dtype=tf.float64)
    source = tf.stack(
        [
            source_base[:, 0],
            0.5 * tf.square(source_base[:, 0]) + source_base[:, 1],
            0.4 * source_base[:, 0] * source_base[:, 1] + source_base[:, 2],
            0.3 * source_base[:, 0] * source_base[:, 1] * source_base[:, 2]
            + source_base[:, 3],
        ],
        axis=1,
    )
    weights = tf.nn.softmax(
        tf.random.stateless_normal([n], [73, 74], dtype=tf.float64) * 0.15
    )
    points = tf.random.stateless_normal([n, d], [75, 76], dtype=tf.float64)
    zeros = tf.zeros([n, d, 1], tf.float64)
    zero_weights = tf.zeros([n, 1], tf.float64)
    basis, _ = tf.linalg.qr(
        tf.random.stateless_normal([d, 3], [77, 78], dtype=tf.float64)
    )
    baseline = higher_moment_shape_jvp(
        source,
        weights,
        zeros,
        zero_weights,
        points,
        zeros,
        correction_steps=2,
        strength=0.02,
        floor=1e-6,
        projected_cumulant_basis=basis,
    )
    candidate = higher_moment_shape_jvp(
        source,
        weights,
        zeros,
        zero_weights,
        points,
        zeros,
        correction_steps=2,
        strength=0.02,
        floor=1e-6,
        projected_cumulant_basis=basis,
        projected_cumulant_correction_steps=3,
        projected_cumulant_strength=0.005,
        projected_cumulant_floor=1e-6,
    )
    assert float(candidate["projected_cumulant_residual_norm"].numpy()) < float(
        baseline["projected_cumulant_residual_norm"].numpy()
    )
    output_mean = tf.reduce_mean(candidate["particles"], axis=0)
    output_centered = candidate["particles"] - output_mean[None, :]
    output_covariance = tf.einsum(
        "ni,nj->ij", output_centered, output_centered
    ) / tf.cast(n, tf.float64)
    target_mean = tf.reduce_sum(weights[:, None] * source, axis=0)
    target_centered = source - target_mean[None, :]
    target_covariance = tf.einsum(
        "n,ni,nj->ij", weights, target_centered, target_centered
    )
    tf.debugging.assert_near(output_mean, target_mean, atol=2e-10, rtol=2e-10)
    tf.debugging.assert_near(
        output_covariance, target_covariance, atol=2e-10, rtol=2e-10
    )


def test_projected_cumulant_manual_jvp_matches_independent_forward_accumulator():
    source, weights, points, source_tangent, weights_tangent, points_tangent = _fixture(
        n=24, d=3
    )
    basis, _ = tf.linalg.qr(
        tf.random.stateless_normal([3, 2], [79, 80], dtype=tf.float64)
    )

    def forward(source_value, weights_value, points_value):
        zeros_source = tf.zeros([24, 3, 1], tf.float64)
        zeros_weights = tf.zeros([24, 1], tf.float64)
        return higher_moment_shape_jvp(
            source_value,
            weights_value,
            zeros_source,
            zeros_weights,
            points_value,
            zeros_source,
            correction_steps=1,
            strength=0.01,
            floor=1e-6,
            projected_cumulant_basis=basis,
            projected_cumulant_correction_steps=2,
            projected_cumulant_strength=0.003,
            projected_cumulant_floor=1e-6,
        )["particles"]

    with tf.autodiff.ForwardAccumulator(
        (source, weights, points),
        (source_tangent, weights_tangent, points_tangent),
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
        correction_steps=1,
        strength=0.01,
        floor=1e-6,
        projected_cumulant_basis=basis,
        projected_cumulant_correction_steps=2,
        projected_cumulant_strength=0.003,
        projected_cumulant_floor=1e-6,
    )["particles_tangent"][:, :, 0]
    tf.debugging.assert_near(automatic, manual, atol=5e-10, rtol=5e-10)


def test_projected_campaign_basis_is_nested_and_sign_canonicalized():
    raw = tf.random.stateless_normal([3, 18, 18], [91, 92], dtype=tf.float32)
    score = tf.linalg.matmul(raw, raw, transpose_b=True)
    first, eigenvalues = _canonical_basis(score)
    second, repeated_eigenvalues = _canonical_basis(score)
    tf.debugging.assert_equal(first, second)
    tf.debugging.assert_equal(eigenvalues, repeated_eigenvalues)
    tf.debugging.assert_near(
        tf.linalg.matmul(first, first, transpose_a=True),
        tf.eye(8, batch_shape=[3]),
        atol=2e-5,
    )
    largest = tf.argmax(tf.abs(first), axis=1, output_type=tf.int32)
    time_index = tf.broadcast_to(tf.range(3)[:, None], tf.shape(largest))
    rank_index = tf.broadcast_to(tf.range(8)[None, :], tf.shape(largest))
    pivots = tf.gather_nd(first, tf.stack([time_index, largest, rank_index], axis=-1))
    assert bool(tf.reduce_all(pivots >= 0.0).numpy())
    tf.debugging.assert_equal(first[:, :, :4], first[:, :, :8][:, :, :4])
    tf.debugging.assert_equal(first[:, :, :6], first[:, :, :8][:, :, :6])


def test_projected_campaign_tuning_variance_is_within_dataset():
    rows = []
    for dataset_index, offset in enumerate((0.0, 1000.0)):
        for seed_index, delta in enumerate((-1.0, 1.0)):
            rows.append(
                {
                    "dataset_index": dataset_index,
                    "finite": True,
                    "device": "/device:GPU:0",
                    "value": offset + delta,
                    "score": [offset + 2.0 * delta, offset + 3.0 * delta, offset + 4.0 * delta],
                    "score_increment_sum_residual": 1.0e-6,
                    "max_mean_residual": 1.0e-6,
                    "max_row_residual": 1.0e-6,
                    "max_col_residual": 1.0e-6,
                    "wall_time_seconds": 1.0,
                    "maximum_normalized_shape_displacement": 0.5,
                    "maximum_projected_cumulant_residual": 1.0,
                }
            )
    summary = _partition_summary(rows)
    assert summary["variance_definition"].startswith("mean within-observation-dataset")
    assert summary["value"]["sample_sd"] == pytest.approx(math.sqrt(2.0))
    assert summary["score_0"]["sample_sd"] == pytest.approx(math.sqrt(8.0))


def test_projected_campaign_validity_uses_declared_austria_tolerance():
    row = {
        "finite": True,
        "device": "/device:GPU:0",
        "max_mean_residual": 1.0e-5,
        "max_row_residual": 1.0e-5,
        "max_col_residual": 1.0e-5,
        "score_increment_sum_residual": 4.88e-4,
        "maximum_normalized_shape_displacement": 1.0,
    }
    assert all(_validity_checks(row).values())
    row["score_increment_sum_residual"] = 5.01e-4
    assert not _validity_checks(row)["residual_tolerance"]


def test_projected_zero_steps_are_exact_structural_noop():
    source, weights, points, source_tangent, weights_tangent, points_tangent = _fixture(
        n=24, d=3
    )
    basis = tf.eye(3, dtype=tf.float64)[:, :2]
    common = dict(correction_steps=2, strength=0.02, floor=1e-6)
    baseline = higher_moment_shape_jvp(
        source,
        weights,
        source_tangent[:, :, None],
        weights_tangent[:, None],
        points,
        points_tangent[:, :, None],
        **common,
    )
    candidate = higher_moment_shape_jvp(
        source,
        weights,
        source_tangent[:, :, None],
        weights_tangent[:, None],
        points,
        points_tangent[:, :, None],
        projected_cumulant_basis=basis,
        projected_cumulant_correction_steps=0,
        projected_cumulant_strength=0.01,
        **common,
    )
    tf.debugging.assert_equal(candidate["particles"], baseline["particles"])
    tf.debugging.assert_equal(
        candidate["particles_tangent"], baseline["particles_tangent"]
    )


def test_pairwise_moment_controls_are_exact_structural_noop_for_scalar_state():
    source, weights, points, source_tangent, weights_tangent, points_tangent = _fixture(
        n=24, d=1
    )
    common = dict(
        correction_steps=2,
        strength=0.02,
        floor=1e-6,
    )
    baseline = higher_moment_shape_jvp(
        source,
        weights,
        source_tangent[:, :, None],
        weights_tangent[:, None],
        points,
        points_tangent[:, :, None],
        **common,
    )
    candidate = higher_moment_shape_jvp(
        source,
        weights,
        source_tangent[:, :, None],
        weights_tangent[:, None],
        points,
        points_tangent[:, :, None],
        pairwise_correction_steps=4,
        pairwise_strength=0.02,
        pairwise_floor=1e-6,
        **common,
    )
    for key in (
        "particles",
        "particles_tangent",
        "pairwise_co_skew_residual",
        "pairwise_co_kurtosis_residual",
    ):
        tf.debugging.assert_equal(candidate[key], baseline[key])


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


def test_zero_step_bounded_teacher_is_exact_noop():
    n = 12
    adapter = exact_transformed_sv_candidate_adapter()
    theta = tf.constant([0.2, -0.1], tf.float32)
    observations = tf.constant([[0.1], [-0.2]], tf.float32)
    initial = tf.random.stateless_normal([n, 1], [811, 812], dtype=tf.float32)
    process = tf.random.stateless_normal([2, n, 1], [813, 814], dtype=tf.float32)
    design = replicate_positive_genut(
        gaussian_genut_design(dim=1), num_particles=n
    )
    zeros_matrix = tf.zeros([2, 1, 1], tf.float32)
    zeros_tangent = tf.zeros([2, 1, 1, 2], tf.float32)
    teacher = BoundedFeatureShapeTeacher(
        frame_mu=tf.zeros([2, 1], tf.float32),
        frame_matrix=tf.ones([2, 1, 1], tf.float32),
        skew=tf.zeros([2, 1], tf.float32),
        kurtosis=tf.ones([2, 1], tf.float32) * 3.0,
        skew_tangent=tf.zeros([2, 1, 2], tf.float32),
        kurtosis_tangent=tf.zeros([2, 1, 2], tf.float32),
        pairwise_co_skew=zeros_matrix,
        pairwise_co_kurtosis=zeros_matrix,
        pairwise_co_skew_tangent=zeros_tangent,
        pairwise_co_kurtosis_tangent=zeros_tangent,
        pairwise_co_skew_mask=zeros_matrix,
        pairwise_co_kurtosis_mask=zeros_matrix,
    )
    baseline = finite_value_score(
        adapter,
        theta,
        observations,
        initial,
        process,
        design,
        transition_before_first_observation=False,
    )
    observed = finite_value_score(
        adapter,
        theta,
        observations,
        initial,
        process,
        design,
        transition_before_first_observation=False,
        bounded_feature_teacher=teacher,
    )
    tf.debugging.assert_equal(observed[0], baseline[0])
    tf.debugging.assert_equal(observed[1], baseline[1])
    tf.debugging.assert_equal(
        observed[2]["maximum_normalized_shape_displacement"],
        baseline[2]["maximum_normalized_shape_displacement"],
    )
    tf.debugging.assert_equal(
        observed[2]["maximum_physical_affine_mean_residual"], 0.0
    )
    tf.debugging.assert_equal(
        observed[2]["maximum_physical_affine_covariance_residual"], 0.0
    )


def test_finite_value_score_projected_cumulant_path_is_finite_and_reports_diagnostics():
    n = 12
    adapter = exact_transformed_sv_candidate_adapter()
    theta = tf.constant([0.2, -0.1], tf.float32)
    observations = tf.constant([[0.1], [-0.2], [0.3]], tf.float32)
    initial = tf.random.stateless_normal([n, 1], [81, 82], dtype=tf.float32)
    process = tf.random.stateless_normal([3, n, 1], [83, 84], dtype=tf.float32)
    design = replicate_positive_genut(gaussian_genut_design(dim=1), num_particles=n)
    result = finite_value_score(
        adapter,
        theta,
        observations,
        initial,
        process,
        design,
        transition_before_first_observation=False,
        higher_moment_correction_steps=1,
        higher_moment_strength=0.01,
        projected_cumulant_basis=tf.ones([3, 1, 1], tf.float32),
        projected_cumulant_correction_steps=1,
        projected_cumulant_strength=0.0025,
        projected_cumulant_sketch_directions=tf.ones([1, 1], tf.float32),
    )
    assert bool(result[2]["program_valid"].numpy())
    assert bool(tf.math.is_finite(result[0]).numpy())
    assert bool(tf.reduce_all(tf.math.is_finite(result[1])).numpy())
    tf.debugging.assert_equal(
        tf.shape(result[2]["projected_cumulant_mode_score"]), [3, 1, 1]
    )
    assert float(result[2]["maximum_projected_cumulant_residual"].numpy()) >= 0.0


def test_higher_moment_module_has_no_numpy_or_runtime_autodiff():
    source = inspect.getsource(higher_moment_shape_jvp)
    assert "numpy" not in source
    assert "GradientTape" not in source
    assert "ForwardAccumulator" not in source


def test_explicit_teacher_shape_targets_preserve_particle_mean_covariance():
    source, weights, points, source_tangent, weights_tangent, points_tangent = _fixture(
        n=48, d=2
    )
    parameter_count = 1
    explicit_skew = tf.constant([0.35, -0.20], tf.float64)
    explicit_kurtosis = tf.constant([3.5, 2.8], tf.float64)
    explicit_co_skew = tf.constant([[0.0, 0.15], [-0.10, 0.0]], tf.float64)
    explicit_co_kurtosis = tf.constant([[0.0, 1.25], [1.25, 0.0]], tf.float64)
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
        pairwise_correction_steps=2,
        pairwise_strength=0.01,
        pairwise_floor=1e-6,
        explicit_target_skew=explicit_skew,
        explicit_target_kurtosis=explicit_kurtosis,
        explicit_target_skew_tangent=tf.zeros([2, parameter_count], tf.float64),
        explicit_target_kurtosis_tangent=tf.zeros([2, parameter_count], tf.float64),
        explicit_target_pairwise_co_skew=explicit_co_skew,
        explicit_target_pairwise_co_kurtosis=explicit_co_kurtosis,
        explicit_target_pairwise_co_skew_tangent=tf.zeros(
            [2, 2, parameter_count], tf.float64
        ),
        explicit_target_pairwise_co_kurtosis_tangent=tf.zeros(
            [2, 2, parameter_count], tf.float64
        ),
    )
    tf.debugging.assert_equal(result["target_source_id"], tf.constant(1, tf.int32))
    tf.debugging.assert_equal(result["target_skew"], explicit_skew)
    tf.debugging.assert_equal(result["target_kurtosis"], explicit_kurtosis)
    tf.debugging.assert_equal(result["target_pairwise_co_skew"], explicit_co_skew)
    tf.debugging.assert_equal(
        result["target_pairwise_co_kurtosis"], explicit_co_kurtosis
    )

    output_mean = tf.reduce_mean(result["particles"], axis=0)
    output_centered = result["particles"] - output_mean[None, :]
    output_covariance = tf.einsum(
        "ni,nj->ij", output_centered, output_centered
    ) / tf.cast(tf.shape(source)[0], tf.float64)
    target_mean = tf.reduce_sum(weights[:, None] * source, axis=0)
    target_centered = source - target_mean[None, :]
    target_covariance = tf.einsum(
        "n,ni,nj->ij", weights, target_centered, target_centered
    )
    tf.debugging.assert_near(output_mean, target_mean, atol=1e-10, rtol=1e-10)
    tf.debugging.assert_near(
        output_covariance, target_covariance, atol=1e-10, rtol=1e-10
    )


def test_explicit_teacher_shape_targets_reject_partial_group():
    source, weights, points, source_tangent, weights_tangent, points_tangent = _fixture()
    with pytest.raises(ValueError, match="complete group"):
        higher_moment_shape_jvp(
            source,
            weights,
            source_tangent[:, :, None],
            weights_tangent[:, None],
            points,
            points_tangent[:, :, None],
            correction_steps=1,
            strength=0.02,
            floor=1e-6,
            explicit_target_skew=tf.zeros([2], tf.float64),
        )


def test_weighted_shape_target_score_jvp_matches_centered_difference():
    points = tf.random.stateless_uniform(
        [80, 3], [901, 902], minval=-0.9, maxval=0.9, dtype=tf.float64
    )
    score = tf.random.stateless_normal([80, 2], [903, 904], dtype=tf.float64)
    log_weight = tf.random.stateless_normal([80], [905, 906], dtype=tf.float64)
    weights = tf.nn.softmax(log_weight)
    centered_score = score - tf.reduce_sum(weights[:, None] * score, axis=0)
    weight_tangent = weights[:, None] * centered_score
    analytic = weighted_shape_targets_jvp(
        points,
        weights,
        tf.zeros([80, 3, 2], tf.float64),
        weight_tangent,
    )
    step = tf.constant(1.0e-5, tf.float64)
    for parameter in range(2):
        plus = tf.nn.softmax(log_weight + step * score[:, parameter])
        minus = tf.nn.softmax(log_weight - step * score[:, parameter])
        plus_target = weighted_shape_targets_jvp(
            points,
            plus,
            tf.zeros([80, 3, 1], tf.float64),
            tf.zeros([80, 1], tf.float64),
        )
        minus_target = weighted_shape_targets_jvp(
            points,
            minus,
            tf.zeros([80, 3, 1], tf.float64),
            tf.zeros([80, 1], tf.float64),
        )
        for name, tangent_name in (
            ("skew", "skew_tangent"),
            ("kurtosis", "kurtosis_tangent"),
            ("pairwise_co_skew", "pairwise_co_skew_tangent"),
            ("pairwise_co_kurtosis", "pairwise_co_kurtosis_tangent"),
        ):
            finite_difference = (
                plus_target[name] - minus_target[name]
            ) / (2.0 * step)
            tf.debugging.assert_near(
                analytic[tangent_name][..., parameter],
                finite_difference,
                atol=3e-7,
                rtol=3e-7,
            )


def test_affine_restore_cloud_jvp_matches_forward_autodiff():
    source, weights, points, source_dot, weights_dot, points_dot = _fixture(
        n=32, d=3
    )

    def forward(source_value, weights_value, points_value):
        return affine_restore_cloud_jvp(
            source_value,
            weights_value,
            tf.zeros([32, 3, 1], tf.float64),
            tf.zeros([32, 1], tf.float64),
            points_value,
            tf.zeros([32, 3, 1], tf.float64),
        )["particles"]

    with tf.autodiff.ForwardAccumulator(
        (source, weights, points), (source_dot, weights_dot, points_dot)
    ) as accumulator:
        output = forward(source, weights, points)
    automatic = accumulator.jvp(output)
    manual = affine_restore_cloud_jvp(
        source,
        weights,
        source_dot[:, :, None],
        weights_dot[:, None],
        points,
        points_dot[:, :, None],
    )
    tf.debugging.assert_near(
        manual["particles_tangent"][:, :, 0],
        automatic,
        atol=3e-10,
        rtol=3e-10,
    )
    assert float(manual["maximum_mean_residual"].numpy()) < 1e-12
    assert float(manual["maximum_covariance_residual"].numpy()) < 1e-12
    assert float(manual["maximum_normalized_mean_residual"].numpy()) < 1e-12
    assert float(manual["maximum_normalized_covariance_residual"].numpy()) < 1e-12
