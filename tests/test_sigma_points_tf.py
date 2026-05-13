from pathlib import Path

import numpy as np
import tensorflow as tf

from bayesfilter.nonlinear.sigma_points_tf import (
    tf_svd_sigma_point_placement,
    tf_unit_sigma_point_rule,
)


ROOT = Path(__file__).resolve().parents[1]


def _weighted_covariance(points: tf.Tensor, weights: tf.Tensor) -> tf.Tensor:
    mean = tf.linalg.matvec(tf.transpose(points), weights)
    centered = points - mean[tf.newaxis, :]
    return tf.transpose(centered) @ (centered * weights[:, tf.newaxis])


def test_cubature_rule_reproduces_standard_normal_first_two_moments() -> None:
    rule = tf_unit_sigma_point_rule(3, rule="cubature")

    mean = tf.linalg.matvec(tf.transpose(rule.offsets), rule.mean_weights)
    covariance = _weighted_covariance(rule.offsets, rule.covariance_weights)

    np.testing.assert_allclose(tf.reduce_sum(rule.mean_weights).numpy(), 1.0, atol=1e-14)
    np.testing.assert_allclose(mean.numpy(), np.zeros(3), atol=1e-14)
    np.testing.assert_allclose(covariance.numpy(), np.eye(3), atol=1e-14)
    assert rule.point_count == 6
    assert rule.polynomial_degree == 3


def test_unscented_rule_reproduces_standard_normal_first_two_moments() -> None:
    rule = tf_unit_sigma_point_rule(3, rule="unscented", alpha=1.0, beta=2.0, kappa=0.0)

    mean = tf.linalg.matvec(tf.transpose(rule.offsets), rule.mean_weights)
    covariance = _weighted_covariance(rule.offsets, rule.covariance_weights)

    np.testing.assert_allclose(tf.reduce_sum(rule.mean_weights).numpy(), 1.0, atol=1e-14)
    np.testing.assert_allclose(mean.numpy(), np.zeros(3), atol=1e-14)
    np.testing.assert_allclose(covariance.numpy(), np.eye(3), atol=1e-14)
    np.testing.assert_allclose(rule.covariance_weights.numpy()[0], 2.0, atol=1e-14)
    assert rule.point_count == 7


def test_svd_placement_stays_on_rank_deficient_support() -> None:
    rule = tf_unit_sigma_point_rule(2, rule="cubature")
    mean = tf.constant([1.0, -2.0], dtype=tf.float64)
    covariance = tf.constant([[4.0, 0.0], [0.0, 0.0]], dtype=tf.float64)

    points, diagnostics = tf_svd_sigma_point_placement(
        mean,
        covariance,
        rule,
        singular_floor=tf.constant(0.0, dtype=tf.float64),
        rank_tolerance=tf.constant(1e-12, dtype=tf.float64),
    )
    reconstructed = _weighted_covariance(points, rule.covariance_weights)

    assert int(diagnostics.rank.numpy()) == 1
    assert int(diagnostics.floor_count.numpy()) == 1
    np.testing.assert_allclose(diagnostics.support_residual.numpy(), 0.0, atol=1e-14)
    np.testing.assert_allclose(points.numpy()[:, 1], -2.0, atol=1e-14)
    np.testing.assert_allclose(reconstructed.numpy(), covariance.numpy(), atol=1e-14)


def test_svd_placement_reports_implemented_floored_covariance() -> None:
    rule = tf_unit_sigma_point_rule(2, rule="cubature")
    covariance = tf.constant([[4.0, 0.0], [0.0, 0.0]], dtype=tf.float64)

    _points, diagnostics = tf_svd_sigma_point_placement(
        tf.zeros([2], dtype=tf.float64),
        covariance,
        rule,
        singular_floor=tf.constant(1e-6, dtype=tf.float64),
        rank_tolerance=tf.constant(1e-12, dtype=tf.float64),
    )

    np.testing.assert_allclose(
        diagnostics.implemented_covariance.numpy(),
        np.array([[4.0, 0.0], [0.0, 1e-6]]),
        atol=1e-14,
    )
    assert diagnostics.psd_projection_residual.numpy() > 0.0


def test_nonlinear_sigma_point_module_does_not_import_numpy_or_call_dot_numpy() -> None:
    text = (ROOT / "bayesfilter" / "nonlinear" / "sigma_points_tf.py").read_text(
        encoding="utf-8"
    )

    assert "import numpy" not in text
    assert "from numpy" not in text
    assert ".numpy(" not in text
