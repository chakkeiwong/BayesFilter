from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import tensorflow as tf

from bayesfilter.highdim import ledh_contract_e_tp_tf as tp


DTYPE = tf.float64
INDICES = tf.constant([0, 1, 2], tf.int32)
SCALE = tf.constant([1.0, 2.0], DTYPE)


def _inputs(theta: tf.Tensor) -> tuple[tf.Tensor, ...]:
    points = tf.stack(
        [-1.0 + 0.1 * theta, 0.2 * theta, 1.0 + 0.05 * theta]
    )[:, None]
    log_weights = tf.stack(
        [
            tf.math.log(tf.constant(0.2, DTYPE)) + 0.1 * theta,
            tf.math.log(tf.constant(0.5, DTYPE)) - 0.05 * theta,
            tf.math.log(tf.constant(0.3, DTYPE)) + 0.02 * theta,
        ]
    )
    coordinates = points[:, 0]
    features = tf.stack(
        [tf.ones([3], DTYPE), coordinates + 0.03 * theta]
    )
    reference = tf.constant([0.30, 0.33, 0.37], DTYPE)
    return points, log_weights, features, reference


def test_diagonal_kkt_matches_dense_precision_reference() -> None:
    points, log_weights, features, reference = _inputs(tf.constant(0.17, DTYPE))
    candidate = tp._contract_e_tp_diagonal_kkt_forward_core(
        points, log_weights, features, INDICES, SCALE, reference
    )
    dense = tp._contract_e_tp_dense_kkt_forward_core(
        points,
        log_weights,
        features,
        INDICES,
        SCALE,
        reference,
        tf.linalg.diag(1.0 / reference),
    )
    assert bool(candidate["valid_chart"].numpy())
    np.testing.assert_allclose(
        candidate["student_weights"], dense["student_weights"], rtol=2e-13, atol=2e-14
    )
    np.testing.assert_allclose(
        candidate["matched_target"], candidate["target"], rtol=0.0, atol=3e-15
    )


def test_diagonal_kkt_manual_jvp_and_vjp_match_tensorflow() -> None:
    theta = tf.constant(0.17, DTYPE)
    direction = tf.constant(-0.43, DTYPE)
    with tf.autodiff.ForwardAccumulator(theta, direction) as accumulator:
        points, log_weights, features, reference = _inputs(theta)
    point_dot, log_weight_dot, feature_dot, _ = tuple(
        accumulator.jvp(value) for value in (points, log_weights, features, reference)
    )
    manual = tp._contract_e_tp_diagonal_kkt_jvp_core(
        points,
        log_weights,
        features,
        INDICES,
        SCALE,
        reference,
        point_dot,
        log_weight_dot,
        feature_dot,
    )
    with tf.autodiff.ForwardAccumulator(theta, direction) as accumulator:
        ad_inputs = _inputs(theta)
        automatic = tp._contract_e_tp_diagonal_kkt_forward_core(
            ad_inputs[0], ad_inputs[1], ad_inputs[2], INDICES, SCALE, ad_inputs[3]
        )
    for name, tangent_name in (
        ("student_points", "student_points_tangent"),
        ("student_weights", "student_weights_tangent"),
        ("matched_target", "matched_target_tangent"),
        ("log_normalizer", "log_normalizer_tangent"),
    ):
        np.testing.assert_allclose(
            manual[tangent_name], accumulator.jvp(automatic[name]), rtol=3e-12, atol=3e-13
        )

    upstream_points = tf.constant([[0.3], [-0.2], [0.1]], DTYPE)
    upstream_weights = tf.constant([0.2, -0.1, 0.4], DTYPE)
    upstream_target = tf.constant([-0.3, 0.25], DTYPE)
    upstream_log_normalizer = tf.constant(0.11, DTYPE)
    manual_vjp = tp._contract_e_tp_diagonal_kkt_vjp_core(
        points,
        log_weights,
        features,
        INDICES,
        SCALE,
        reference,
        upstream_points,
        upstream_weights,
        upstream_target,
        upstream_log_normalizer,
    )
    with tf.GradientTape() as tape:
        tape.watch((points, log_weights, features))
        forward = tp._contract_e_tp_diagonal_kkt_forward_core(
            points, log_weights, features, INDICES, SCALE, reference
        )
        objective = (
            tf.reduce_sum(forward["student_points"] * upstream_points)
            + tf.tensordot(forward["student_weights"], upstream_weights, axes=1)
            + tf.tensordot(forward["matched_target"], upstream_target, axes=1)
            + forward["log_normalizer"] * upstream_log_normalizer
        )
    automatic_bars = tape.gradient(objective, (points, log_weights, features))
    for expected, actual in zip(
        (
            manual_vjp["teacher_points_bar"],
            manual_vjp["log_unnormalized_weights_bar"],
            manual_vjp["teacher_features_bar"],
        ),
        automatic_bars,
        strict=True,
    ):
        np.testing.assert_allclose(
            expected, tf.convert_to_tensor(actual), rtol=4e-12, atol=4e-13
        )


def test_diagonal_kkt_vectorized_jvp_matches_single_directions() -> None:
    theta = tf.constant(0.17, DTYPE)
    points, log_weights, features, reference = _inputs(theta)
    point_dots = tf.stack(
        [tf.ones_like(points) * 0.03, tf.ones_like(points) * -0.08], axis=-1
    )
    log_weight_dots = tf.stack(
        [
            tf.linspace(tf.constant(-0.1, DTYPE), tf.constant(0.2, DTYPE), 3),
            tf.linspace(tf.constant(0.07, DTYPE), tf.constant(-0.04, DTYPE), 3),
        ],
        axis=-1,
    )
    feature_dots = tf.stack(
        [tf.ones_like(features) * 0.02, tf.ones_like(features) * -0.05], axis=-1
    )
    vectorized = tp._contract_e_tp_diagonal_kkt_multi_jvp_core(
        points,
        log_weights,
        features,
        INDICES,
        SCALE,
        reference,
        point_dots,
        log_weight_dots,
        feature_dots,
    )
    for direction in range(2):
        single = tp._contract_e_tp_diagonal_kkt_jvp_core(
            points,
            log_weights,
            features,
            INDICES,
            SCALE,
            reference,
            point_dots[..., direction],
            log_weight_dots[..., direction],
            feature_dots[..., direction],
        )
        for name in (
            "student_points_tangent",
            "student_weights_tangent",
            "matched_target_tangent",
            "target_tangent",
            "log_normalizer_tangent",
        ):
            np.testing.assert_allclose(
                vectorized[name][..., direction],
                single[name],
                rtol=3e-12,
                atol=3e-13,
            )


def test_diagonal_kkt_square_limit_and_fail_closed_diagnostics() -> None:
    points = tf.constant([[-1.0], [1.0]], DTYPE)
    log_weights = tf.math.log(tf.constant([0.4, 0.6], DTYPE))
    features = tf.stack([tf.ones([2], DTYPE), points[:, 0]])
    reference = tf.constant([0.5, 0.5], DTYPE)
    candidate = tp._contract_e_tp_diagonal_kkt_forward_core(
        points, log_weights, features, tf.constant([0, 1]), SCALE, reference
    )
    square = tp._contract_e_tp_dense_square_forward_core(
        points, log_weights, features, tf.constant([0, 1]), SCALE
    )
    np.testing.assert_allclose(
        candidate["student_weights"], square["student_weights"], rtol=2e-13, atol=2e-14
    )

    duplicate = tp._contract_e_tp_diagonal_kkt_forward_core(
        points,
        log_weights,
        features,
        tf.constant([0, 0]),
        SCALE,
        reference,
    )
    assert not bool(duplicate["valid_chart"].numpy())
    assert bool(tf.reduce_all(tf.math.is_nan(duplicate["student_weights"])).numpy())

    nonpositive_reference = tp._contract_e_tp_diagonal_kkt_forward_core(
        points,
        log_weights,
        features,
        tf.constant([0, 1]),
        SCALE,
        tf.constant([1.0, 0.0], DTYPE),
    )
    assert not bool(nonpositive_reference["valid_chart"].numpy())
    assert bool(
        tf.reduce_all(tf.math.is_nan(nonpositive_reference["matched_target"])).numpy()
    )
