from __future__ import annotations

import importlib.util
import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import tensorflow as tf

from bayesfilter.highdim import ledh_contract_e_tp_tf as tp


ROOT = Path(__file__).resolve().parents[2]
WITNESS_PATH = ROOT / "docs/benchmarks/contract_e_score_aware_teacher_projection_2d_lgssm.py"
DTYPE = tf.float64
ACTIVE_INDICES = tf.constant([108, 221, 2317, 2402, 2474, 3942, 4001], tf.int32)
ROW_SCALE = tf.constant(
    [1.0, 5.1012693387019965, 4.522840653349041, 26.022948865981103,
     21.932472504120415, 20.456087575586782, 0.6326344960582179],
    DTYPE,
)


def _load_witness():
    specification = importlib.util.spec_from_file_location("contract_e_tp_witness_derivatives", WITNESS_PATH)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


WITNESS = _load_witness()


def _witness_inputs(theta: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    teacher = WITNESS._teacher(theta)
    log_weights = tf.math.log(teacher["weights"]) + teacher["current_increment"]
    return teacher["candidates"], log_weights, teacher["features"]


def _square_scalar(theta: tf.Tensor) -> tf.Tensor:
    points, log_weights, features = _witness_inputs(theta)
    result = tp._contract_e_tp_dense_square_forward_core(
        points, log_weights, features, ACTIVE_INDICES, ROW_SCALE
    )
    return result["log_normalizer"] + tf.math.log(result["matched_target"][-1])


def _teacher_scalar(theta: tf.Tensor) -> tf.Tensor:
    points, log_weights, features = _witness_inputs(theta)
    del points
    result = tp._dense_teacher_reduce_core(log_weights, features)
    return result["log_normalizer"] + tf.math.log(result["target"][-1])


def test_square_manual_jvp_matches_forward_ad_and_preserves_teacher_tangent() -> None:
    theta = tf.identity(WITNESS.THETA0)
    direction = tf.constant([0.31, -0.47, 0.19], DTYPE)
    with tf.autodiff.ForwardAccumulator(theta, direction) as accumulator:
        points, log_weights, features = _witness_inputs(theta)
    tangents = tuple(accumulator.jvp(value) for value in (points, log_weights, features))
    assert all(value is not None for value in tangents)
    manual = tp._contract_e_tp_dense_square_jvp_core(
        points, log_weights, features, ACTIVE_INDICES, ROW_SCALE, *tangents
    )
    with tf.autodiff.ForwardAccumulator(theta, direction) as accumulator:
        points_ad, log_weights_ad, features_ad = _witness_inputs(theta)
        automatic = tp._contract_e_tp_dense_square_forward_core(
            points_ad, log_weights_ad, features_ad, ACTIVE_INDICES, ROW_SCALE
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
    np.testing.assert_allclose(
        manual["matched_target_tangent"], manual["target_tangent"], rtol=2e-12, atol=5e-14
    )


def test_square_manual_vjp_matches_tensorflow_for_each_input_owner() -> None:
    points, log_weights, features = _witness_inputs(WITNESS.THETA0)
    upstream_points = tf.reshape(tf.linspace(tf.constant(-0.3, DTYPE), tf.constant(0.4, DTYPE), 14), [7, 2])
    upstream_weights = tf.linspace(tf.constant(-0.2, DTYPE), tf.constant(0.25, DTYPE), 7)
    upstream_target = tf.linspace(tf.constant(0.1, DTYPE), tf.constant(-0.15, DTYPE), 7)
    upstream_log_normalizer = tf.constant(0.37, DTYPE)
    manual = tp._contract_e_tp_dense_square_vjp_core(
        points, log_weights, features, ACTIVE_INDICES, ROW_SCALE,
        upstream_points, upstream_weights, upstream_target, upstream_log_normalizer,
    )
    with tf.GradientTape() as tape:
        tape.watch((points, log_weights, features))
        result = tp._contract_e_tp_dense_square_forward_core(
            points, log_weights, features, ACTIVE_INDICES, ROW_SCALE
        )
        objective = (
            tf.reduce_sum(result["student_points"] * upstream_points)
            + tf.tensordot(result["student_weights"], upstream_weights, axes=1)
            + tf.tensordot(result["matched_target"], upstream_target, axes=1)
            + result["log_normalizer"] * upstream_log_normalizer
        )
    automatic = tape.gradient(objective, (points, log_weights, features))
    for expected, actual in zip(
        (manual["teacher_points_bar"], manual["log_unnormalized_weights_bar"], manual["teacher_features_bar"]),
        automatic,
        strict=True,
    ):
        np.testing.assert_allclose(
            expected, tf.convert_to_tensor(actual), rtol=5e-12, atol=3e-13
        )


def test_square_student_value_and_score_match_teacher_and_same_scalar_fd() -> None:
    theta = tf.identity(WITNESS.THETA0)
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(theta)
        student_value = _square_scalar(theta)
        teacher_value = _teacher_scalar(theta)
    student_score = tape.gradient(student_value, theta)
    teacher_score = tape.gradient(teacher_value, theta)
    np.testing.assert_allclose(student_value, teacher_value, rtol=0.0, atol=3e-15)
    np.testing.assert_allclose(student_score, teacher_score, rtol=3e-13, atol=3e-14)

    step = 1.0e-5
    fd = []
    for index in range(3):
        basis = np.zeros(3)
        basis[index] = step
        plus = _square_scalar(tf.constant(theta.numpy() + basis, DTYPE))
        minus = _square_scalar(tf.constant(theta.numpy() - basis, DTYPE))
        fd.append(float((plus - minus).numpy() / (2.0 * step)))
    fd = np.asarray(fd)
    relative = np.linalg.norm(student_score.numpy() - fd) / max(np.linalg.norm(fd), 1.0)
    assert relative <= 0.05 * np.sqrt(3.0)


def _kkt_inputs(theta: tf.Tensor) -> tuple[tf.Tensor, ...]:
    points = tf.stack([-1.0 + 0.1 * theta, 0.2 * theta, 1.0 + 0.05 * theta])[:, None]
    log_weights = tf.stack([
        tf.math.log(tf.constant(0.2, DTYPE)) + 0.1 * theta,
        tf.math.log(tf.constant(0.5, DTYPE)) - 0.05 * theta,
        tf.math.log(tf.constant(0.3, DTYPE)) + 0.02 * theta,
    ])
    coordinates = points[:, 0]
    features = tf.stack([tf.ones([3], DTYPE), coordinates + 0.03 * theta])
    reference = tf.constant([0.30, 0.33, 0.37], DTYPE) + theta * tf.constant([0.01, -0.02, 0.01], DTYPE)
    lower = tf.stack([
        tf.stack([1.2 + 0.03 * theta, tf.constant(0.0, DTYPE), tf.constant(0.0, DTYPE)]),
        tf.stack([0.08 * theta, 1.1 - 0.02 * theta, tf.constant(0.0, DTYPE)]),
        tf.stack([tf.constant(0.04, DTYPE), -0.03 * theta, 0.9 + 0.01 * theta]),
    ])
    precision = lower @ tf.transpose(lower)
    return points, log_weights, features, reference, precision


def test_kkt_manual_jvp_and_vjp_match_tensorflow() -> None:
    theta = tf.constant(0.17, DTYPE)
    direction = tf.constant(-0.43, DTYPE)
    with tf.autodiff.ForwardAccumulator(theta, direction) as accumulator:
        inputs = _kkt_inputs(theta)
    tangents = tuple(accumulator.jvp(value) for value in inputs)
    points, log_weights, features, reference, precision = inputs
    point_tangent, log_weight_tangent, feature_tangent, reference_tangent, precision_tangent = tangents
    manual_jvp = tp._contract_e_tp_dense_kkt_jvp_core(
        points, log_weights, features, tf.constant([0, 1, 2]), tf.ones([2], DTYPE),
        reference, precision, point_tangent, log_weight_tangent, feature_tangent,
        reference_tangent, precision_tangent,
    )
    with tf.autodiff.ForwardAccumulator(theta, direction) as accumulator:
        ad_inputs = _kkt_inputs(theta)
        automatic = tp._contract_e_tp_dense_kkt_forward_core(
            ad_inputs[0], ad_inputs[1], ad_inputs[2], tf.constant([0, 1, 2]),
            tf.ones([2], DTYPE), ad_inputs[3], ad_inputs[4],
        )
    for name, tangent_name in (
        ("student_points", "student_points_tangent"),
        ("student_weights", "student_weights_tangent"),
        ("matched_target", "matched_target_tangent"),
        ("log_normalizer", "log_normalizer_tangent"),
    ):
        np.testing.assert_allclose(
            manual_jvp[tangent_name], accumulator.jvp(automatic[name]), rtol=2e-11, atol=2e-12
        )

    upstream_points = tf.constant([[0.3], [-0.2], [0.1]], DTYPE)
    upstream_weights = tf.constant([0.2, -0.1, 0.4], DTYPE)
    upstream_target = tf.constant([-0.3, 0.25], DTYPE)
    upstream_log_normalizer = tf.constant(0.11, DTYPE)
    manual_vjp = tp._contract_e_tp_dense_kkt_vjp_core(
        points, log_weights, features, tf.constant([0, 1, 2]), tf.ones([2], DTYPE),
        reference, precision, upstream_points, upstream_weights,
        upstream_target, upstream_log_normalizer,
    )
    with tf.GradientTape() as tape:
        tape.watch(inputs)
        result = tp._contract_e_tp_dense_kkt_forward_core(
            points, log_weights, features, tf.constant([0, 1, 2]), tf.ones([2], DTYPE),
            reference, precision,
        )
        objective = (
            tf.reduce_sum(result["student_points"] * upstream_points)
            + tf.tensordot(result["student_weights"], upstream_weights, axes=1)
            + tf.tensordot(result["matched_target"], upstream_target, axes=1)
            + result["log_normalizer"] * upstream_log_normalizer
        )
    automatic_bars = tape.gradient(objective, inputs)
    manual_bars = (
        manual_vjp["teacher_points_bar"],
        manual_vjp["log_unnormalized_weights_bar"],
        manual_vjp["teacher_features_bar"],
        manual_vjp["reference_weights_bar"],
        manual_vjp["precision_bar"],
    )
    for expected, actual in zip(manual_bars, automatic_bars, strict=True):
        np.testing.assert_allclose(
            expected, tf.convert_to_tensor(actual), rtol=3e-11, atol=3e-12
        )
