"""Independent CPU mechanics and derivative checks for stable restoration."""

import pytest
import tensorflow as tf

from bayesfilter.highdim.higher_moment_contract_e import affine_restore_cloud_jvp, higher_moment_shape_jvp


def fixture():
    source = tf.random.stateless_normal([32, 3], [81, 12], dtype=tf.float64)
    weights = tf.nn.softmax(tf.linspace(tf.constant(-1., tf.float64), 1., 32))
    offsets = tf.random.stateless_normal([32, 3], [82, 12], dtype=tf.float64)
    source_dot = tf.random.stateless_normal([32, 3, 1], [83, 12], dtype=tf.float64) * .02
    weights_dot = tf.random.stateless_normal([32, 1], [84, 12], dtype=tf.float64) * .001
    weights_dot -= tf.reduce_sum(weights_dot) * weights[:, None]
    offsets_dot = tf.random.stateless_normal([32, 3, 1], [85, 12], dtype=tf.float64) * .03
    return source, weights, offsets, source_dot, weights_dot, offsets_dot


@pytest.mark.parametrize("translation", [0., 100., -1000.])
def test_offset_restoration_preserves_healthy_values_and_tangents(translation):
    source, weights, offsets, source_dot, weights_dot, offsets_dot = fixture()
    points = offsets + translation
    points_dot = offsets_dot + .7
    ordinary = affine_restore_cloud_jvp(source, weights, source_dot, weights_dot, points, points_dot)
    stable = affine_restore_cloud_jvp(source, weights, source_dot, weights_dot, points, points_dot,
        point_offsets=offsets, point_offsets_tangent=offsets_dot)
    for key in ("particles", "particles_tangent"):
        tf.debugging.assert_near(stable[key], ordinary[key], atol=1e-10, rtol=1e-10)
    assert bool(stable["valid"])


def test_offsets_rescue_translation_rounding_without_new_ridge():
    source, weights, offsets, source_dot, weights_dot, offsets_dot = fixture()
    points = offsets + tf.constant(2. ** 60, tf.float64)
    assert bool(tf.reduce_all(points == points[0]))
    ordinary = affine_restore_cloud_jvp(source, weights, source_dot, weights_dot, points, offsets_dot)
    stable = affine_restore_cloud_jvp(source, weights, source_dot, weights_dot, points, offsets_dot,
        point_offsets=offsets, point_offsets_tangent=offsets_dot)
    assert not bool(ordinary["valid"])
    assert bool(stable["valid"])
    assert float(stable["current_covariance_trace"]) > 0
    assert float(stable["maximum_normalized_covariance_residual"]) < 1e-10
    assert float(stable["maximum_normalized_mean_residual"]) < 1e-10


def test_offsets_total_tangent_matches_independent_derivatives():
    source, weights, offsets, source_dot, weights_dot, offsets_dot = fixture()

    def forward(source_value, weights_value, offsets_value):
        return affine_restore_cloud_jvp(source_value, weights_value, tf.zeros_like(source_dot),
            tf.zeros_like(weights_dot), offsets_value + 2. ** 60, tf.zeros_like(offsets_dot),
            point_offsets=offsets_value, point_offsets_tangent=tf.zeros_like(offsets_dot))["particles"]

    directions = (source_dot[..., 0], weights_dot[..., 0], offsets_dot[..., 0])
    with tf.autodiff.ForwardAccumulator((source, weights, offsets), directions) as accumulator:
        output = forward(source, weights, offsets)
    stable = affine_restore_cloud_jvp(source, weights, source_dot, weights_dot,
        offsets + 2. ** 60, offsets_dot, point_offsets=offsets, point_offsets_tangent=offsets_dot)
    tf.debugging.assert_near(stable["particles_tangent"][..., 0], accumulator.jvp(output),
        atol=3e-10, rtol=3e-10)
    for step in (1e-3, 1e-4):
        plus = forward(*(value + step * direction for value, direction in zip((source, weights, offsets), directions)))
        minus = forward(*(value - step * direction for value, direction in zip((source, weights, offsets), directions)))
        tf.debugging.assert_near(stable["particles_tangent"][..., 0], (plus - minus) / (2 * step),
            atol=3e-8, rtol=3e-8)


def test_offset_tangent_is_required():
    source, weights, offsets, source_dot, weights_dot, offsets_dot = fixture()
    with pytest.raises(ValueError, match="supplied together"):
        affine_restore_cloud_jvp(source, weights, source_dot, weights_dot, offsets, offsets_dot,
            point_offsets=offsets)


def test_relative_covariance_loss_exposes_sub_ulp_collapse():
    source = tf.constant([[8.], [9.]], tf.float64)
    weights = tf.constant([1., 1e-40], tf.float64)
    offsets = tf.constant([[-1.], [1.]], tf.float64)
    tangent = tf.zeros([2, 1, 1], tf.float64)
    result = affine_restore_cloud_jvp(source, weights, tangent,
        tf.zeros([2, 1], tf.float64), offsets, tangent,
        point_offsets=offsets, point_offsets_tangent=tangent)
    assert float(result["relative_covariance_residual"]) == 1.
    assert float(result["maximum_normalized_covariance_residual"]) < 1e-30


def test_collapsed_materialized_cloud_reports_unavailable_shape():
    source, weights, offsets, source_dot, weights_dot, offsets_dot = fixture()
    result = higher_moment_shape_jvp(source, weights, source_dot, weights_dot,
        tf.ones_like(offsets) * 8., tf.zeros_like(offsets_dot), correction_steps=0,
        strength=0., return_stages=True)
    assert not bool(result["valid"])
    assert not bool(result["value_valid"])
    assert not bool(result["output_standardization_valid"])
    assert not bool(result["materialized_shape_available"])
    tf.debugging.assert_equal(result["particles"], tf.ones_like(offsets) * 8.)


def test_shared_correction_preserves_offset_diagnostics():
    from bayesfilter.highdim.ledh_unified_correction_tf import batched_higher_moment_shape_jvp

    source, weights, offsets, source_dot, weights_dot, offsets_dot = fixture()
    controls = dict(correction_steps=1, strength=.02, diagonal_lm_damping=.01,
        diagonal_trust_radius=.5, coordinatewise_standardized_cap=.94, return_stages=True)
    single = higher_moment_shape_jvp(source, weights, source_dot, weights_dot, offsets, offsets_dot, **controls)
    batch = batched_higher_moment_shape_jvp(source[None], weights[None],
        tf.transpose(source_dot, [2, 0, 1])[:, None], tf.transpose(weights_dot)[..., None, :],
        offsets[None], tf.transpose(offsets_dot, [2, 0, 1])[:, None], **controls)
    for name in ("stage_post_cap_offsets", "stage_post_cap_offsets_tangent",
                 "affine_restoration_target_covariance_trace",
                 "affine_restoration_materialized_covariance_trace"):
        tf.debugging.assert_near(batch[name][0], single[name], atol=1e-10, rtol=1e-10)
    assert bool(batch["materialized_shape_available"][0])
