from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import tensorflow as tf

from bayesfilter.highdim.genut_shape_lm_tf import (
    necessary_marginal_feasibility,
    scaled_lm_coefficients_jvp,
    scaled_lm_coefficients_value,
    smooth_rms_cap_jvp,
    smooth_rms_cap_value,
)
from bayesfilter.highdim.higher_moment_contract_e import higher_moment_shape_jvp
from bayesfilter.highdim.cubature_genut_batch_tf import (
    _higher_moment_batch_jvp,
    _higher_moment_batch_value,
)


def test_scaled_lm_is_finite_and_condition_bounded_for_ill_scaled_jacobian() -> None:
    jacobian = tf.constant(
        [[[1.0e5, 1.0], [1.1e5, 1.1]]], dtype=tf.float64
    )
    residual = tf.constant([[1800.0, 2700.0]], dtype=tf.float64)
    damping = 1.0e-2
    result = scaled_lm_coefficients_value(
        jacobian,
        residual,
        strength=0.2,
        damping=damping,
        scale_floor=1.0e-6,
    )
    assert bool(tf.reduce_all(tf.math.is_finite(result["coefficient"])).numpy())
    assert float(tf.reduce_max(result["scaled_system_condition"]).numpy()) <= (
        (2.0 + damping) / damping + 1.0e-3
    )


def test_scaled_lm_manual_jvp_matches_forward_accumulator() -> None:
    jacobian = tf.constant(
        [[[5.0, -2.0], [3.0, 7.0]]], dtype=tf.float64
    )
    residual = tf.constant([[0.4, -1.2]], dtype=tf.float64)
    jacobian_tangent = tf.constant(
        [[[[0.2], [0.1]], [[-0.3], [0.4]]]], dtype=tf.float64
    )
    residual_tangent = tf.constant([[[0.05], [-0.02]]], dtype=tf.float64)

    with tf.autodiff.ForwardAccumulator(
        (jacobian, residual),
        (jacobian_tangent[..., 0], residual_tangent[..., 0]),
    ) as accumulator:
        value = scaled_lm_coefficients_value(
            jacobian,
            residual,
            strength=0.3,
            damping=0.05,
            scale_floor=1.0e-4,
        )["coefficient"]
    automatic = accumulator.jvp(value)
    manual = scaled_lm_coefficients_jvp(
        jacobian,
        residual,
        jacobian_tangent,
        residual_tangent,
        strength=0.3,
        damping=0.05,
        scale_floor=1.0e-4,
    )["coefficient_tangent"][..., 0]
    tf.debugging.assert_near(automatic, manual, atol=1.0e-11, rtol=1.0e-11)


def test_smooth_rms_cap_is_strictly_bounded_and_jvp_matches() -> None:
    displacement = tf.constant(
        [[3.0, 4.0], [0.1, -0.2]], dtype=tf.float64
    )
    tangent = tf.constant(
        [[[0.2], [-0.1]], [[0.05], [0.03]]], dtype=tf.float64
    )
    radius = 0.5
    value = smooth_rms_cap_value(displacement, radius=radius)
    assert float(tf.reduce_max(value["post_rms"]).numpy()) < radius

    with tf.autodiff.ForwardAccumulator(
        displacement, tangent[..., 0]
    ) as accumulator:
        capped = smooth_rms_cap_value(
            displacement, radius=radius
        )["displacement"]
    automatic = accumulator.jvp(capped)
    manual = smooth_rms_cap_jvp(
        displacement, tangent, radius=radius
    )["displacement_tangent"][..., 0]
    tf.debugging.assert_near(automatic, manual, atol=1.0e-12, rtol=1.0e-12)


def test_collapsed_weight_shape_repair_stays_finite_and_reports_infeasibility() -> None:
    particle_count = 128
    source = tf.linspace(
        tf.constant(-2.0, tf.float64),
        tf.constant(2.0, tf.float64),
        particle_count,
    )[:, None]
    dominant = tf.constant(0.998, tf.float64)
    weights = tf.concat(
        [
            dominant[None],
            tf.fill(
                [particle_count - 1],
                (1.0 - dominant) / tf.cast(particle_count - 1, tf.float64),
            ),
        ],
        axis=0,
    )
    points = tf.random.stateless_normal(
        [particle_count, 1], [2026, 815], dtype=tf.float64
    )
    zeros_source = tf.zeros([particle_count, 1, 1], tf.float64)
    zeros_weights = tf.zeros([particle_count, 1], tf.float64)
    result = higher_moment_shape_jvp(
        source,
        weights,
        zeros_source,
        zeros_weights,
        points,
        zeros_source,
        correction_steps=4,
        strength=0.2,
        floor=1.0e-5,
        diagonal_lm_damping=1.0e-2,
        diagonal_lm_scale_floor=1.0e-4,
        diagonal_trust_radius=0.5,
    )
    assert bool(result["valid"].numpy())
    assert bool(tf.reduce_all(tf.math.is_finite(result["particles"])).numpy())
    assert float(result["minimum_finite_particle_upper_margin"].numpy()) < 0.0
    assert (
        float(result["maximum_diagonal_post_cap_particle_rms"].numpy())
        < 0.5
    )
    assert float(result["maximum_diagonal_scaled_system_condition"].numpy()) <= 201.1


def test_necessary_feasibility_detects_preserved_failure_scale() -> None:
    result = necessary_marginal_feasibility(
        tf.constant([23.9423], tf.float32),
        tf.constant([3006.7969], tf.float32),
        tf.constant(1008),
    )
    assert not bool(result["valid"][0].numpy())
    assert float(result["finite_particle_upper_margin"][0].numpy()) < -1900.0


def test_batch_collapsed_weight_replay_stays_finite_at_n1008() -> None:
    particle_count = 1008
    axis = tf.linspace(
        tf.constant(-2.0, tf.float32),
        tf.constant(2.0, tf.float32),
        particle_count,
    )
    source = tf.stack(
        [axis, tf.sin(1.7 * axis) + 0.15 * axis],
        axis=-1,
    )[None, :, :]
    weights = tf.concat(
        [
            tf.constant([[0.998]], tf.float32),
            tf.fill([1, particle_count - 1], tf.constant(0.002 / 1007.0)),
        ],
        axis=1,
    )
    points = tf.random.stateless_normal(
        [1, particle_count, 2], [2026, 816], dtype=tf.float32
    )
    source_tangent = tf.zeros([1, particle_count, 2, 1], tf.float32)
    weights_tangent = tf.zeros([1, particle_count, 1], tf.float32)
    points_tangent = tf.zeros([1, particle_count, 2, 1], tf.float32)
    kwargs = {
        "correction_steps": 4,
        "strength": 0.2,
        "floor": 1.0e-5,
        "lm_damping": 1.0e-2,
        "lm_scale_floor": 1.0e-4,
        "trust_radius": 0.5,
    }
    value = _higher_moment_batch_value(source, weights, points, **kwargs)
    score = _higher_moment_batch_jvp(
        source,
        weights,
        source_tangent,
        weights_tangent,
        points,
        points_tangent,
        **kwargs,
    )
    assert bool(value["valid"][0].numpy())
    assert bool(score["valid"][0].numpy())
    assert bool(tf.reduce_all(tf.math.is_finite(value["particles"])).numpy())
    assert bool(
        tf.reduce_all(tf.math.is_finite(score["particles_tangent"])).numpy()
    )
    assert bool(
        tf.reduce_all(
            tf.math.is_finite(value["minimum_finite_particle_upper_margin"])
        ).numpy()
    )
