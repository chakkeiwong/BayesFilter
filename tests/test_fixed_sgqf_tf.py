from pathlib import Path

import numpy as np
import tensorflow as tf

from bayesfilter.nonlinear.fixed_sgqf_tf import (
    tf_fixed_sgqf_active_multi_indices,
    tf_fixed_sgqf_branch_identity,
    tf_fixed_sgqf_cloud,
    tf_fixed_sgqf_level2_axis_cloud,
    tf_fixed_sgqf_combination_coefficient,
    tf_standard_normal_ghq_level_rule,
)


ROOT = Path(__file__).resolve().parents[1]


def _weighted_covariance(points: tf.Tensor, weights: tf.Tensor) -> tf.Tensor:
    mean = tf.linalg.matvec(tf.transpose(points), weights)
    centered = points - mean[tf.newaxis, :]
    return tf.transpose(centered) @ (centered * weights[:, tf.newaxis])


def test_fixed_sgqf_level1_rule_is_the_center_point_rule() -> None:
    rule = tf_standard_normal_ghq_level_rule(1)

    np.testing.assert_allclose(rule.nodes, np.array([0.0]), atol=1e-14)
    np.testing.assert_allclose(rule.weights, np.array([1.0]), atol=1e-14)
    assert rule.point_count == 1
    assert rule.polynomial_degree == 1


def test_fixed_sgqf_level2_rule_matches_the_p47_three_point_rule() -> None:
    rule = tf_standard_normal_ghq_level_rule(2)
    mean = tf.reduce_sum(rule.nodes * rule.weights)
    variance = tf.reduce_sum(tf.square(rule.nodes) * rule.weights)

    np.testing.assert_allclose(
        rule.nodes,
        np.array([-np.sqrt(3.0), 0.0, np.sqrt(3.0)]),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        rule.weights,
        np.array([1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0]),
        atol=1e-12,
    )
    np.testing.assert_allclose(mean, 0.0, atol=1e-14)
    np.testing.assert_allclose(variance, 1.0, atol=1e-14)


def test_fixed_sgqf_level3_rule_reproduces_standard_normal_first_two_moments() -> None:
    rule = tf_standard_normal_ghq_level_rule(3)
    mean = tf.reduce_sum(rule.nodes * rule.weights)
    variance = tf.reduce_sum(tf.square(rule.nodes) * rule.weights)

    assert rule.point_count == 5
    np.testing.assert_allclose(tf.reduce_sum(rule.weights), 1.0, atol=1e-14)
    np.testing.assert_allclose(mean, 0.0, atol=1e-14)
    np.testing.assert_allclose(variance, 1.0, atol=1e-14)


def test_fixed_sgqf_active_indices_and_coefficients_match_p47_3d_level2_preview() -> None:
    active = tf_fixed_sgqf_active_multi_indices(3, 2)
    coefficients = tuple(
        tf_fixed_sgqf_combination_coefficient(3, 2, multi_index)
        for multi_index in active
    )

    assert active == ((1, 1, 1), (1, 1, 2), (1, 2, 1), (2, 1, 1))
    assert coefficients == (-2, 1, 1, 1)


def test_fixed_sgqf_level2_3d_cloud_merges_to_six_axis_points() -> None:
    cloud = tf_fixed_sgqf_cloud(3, 2)
    reconstructed = _weighted_covariance(cloud.points, cloud.weights)
    actual_points = sorted(tuple(point) for point in cloud.points.numpy())
    expected_points = sorted(
        [
            (-np.sqrt(3.0), 0.0, 0.0),
            (0.0, -np.sqrt(3.0), 0.0),
            (0.0, 0.0, -np.sqrt(3.0)),
            (0.0, 0.0, np.sqrt(3.0)),
            (0.0, np.sqrt(3.0), 0.0),
            (np.sqrt(3.0), 0.0, 0.0),
        ]
    )

    assert cloud.active_multi_indices == ((1, 1, 1), (1, 1, 2), (1, 2, 1), (2, 1, 1))
    assert cloud.combination_coefficients == (-2, 1, 1, 1)
    assert cloud.point_count == 6
    assert cloud.merge_tolerance_policy == "scaled_by_max_1_supnorm"
    np.testing.assert_allclose(cloud.weights, np.full(6, 1.0 / 6.0), atol=1e-12)
    np.testing.assert_allclose(np.array(actual_points), np.array(expected_points), atol=1e-12)
    np.testing.assert_allclose(tf.reduce_sum(cloud.weights), 1.0, atol=1e-14)
    np.testing.assert_allclose(cloud.weight_total, 1.0, atol=1e-14)
    assert cloud.negative_weight_count == 0
    np.testing.assert_allclose(reconstructed, np.eye(3), atol=1e-12)


def test_fixed_sgqf_level2_2d_cloud_reproduces_selected_standard_normal_moments() -> None:
    cloud = tf_fixed_sgqf_cloud(2, 2)
    mean = tf.linalg.matvec(tf.transpose(cloud.points), cloud.weights)
    reconstructed = _weighted_covariance(cloud.points, cloud.weights)
    x = cloud.points[:, 0]
    y = cloud.points[:, 1]

    assert cloud.point_count == 5
    np.testing.assert_allclose(cloud.weight_total, 1.0, atol=1e-14)
    np.testing.assert_allclose(mean, np.zeros(2), atol=1e-14)
    np.testing.assert_allclose(reconstructed, np.eye(2), atol=1e-12)
    np.testing.assert_allclose(tf.reduce_sum(cloud.weights * tf.pow(x, 4)), 3.0, atol=1e-12)
    np.testing.assert_allclose(tf.reduce_sum(cloud.weights * tf.square(x) * tf.square(y)), 0.0, atol=1e-12)


def test_fixed_sgqf_level2_4d_cloud_reproduces_standard_normal_covariance() -> None:
    cloud = tf_fixed_sgqf_cloud(4, 2)
    mean = tf.linalg.matvec(tf.transpose(cloud.points), cloud.weights)
    reconstructed = _weighted_covariance(cloud.points, cloud.weights)

    assert cloud.point_count == 9
    np.testing.assert_allclose(cloud.weight_total, 1.0, atol=1e-14)
    np.testing.assert_allclose(mean, np.zeros(4), atol=1e-14)
    np.testing.assert_allclose(reconstructed, np.eye(4), atol=1e-12)


def test_level2_axis_cloud_matches_generic_cloud_through_four_dimensions() -> None:
    for dim in range(1, 5):
        generic = tf_fixed_sgqf_cloud(dim, 2)
        exact = tf_fixed_sgqf_level2_axis_cloud(dim)

        np.testing.assert_allclose(exact.points, generic.points, atol=1e-12)
        np.testing.assert_allclose(exact.weights, generic.weights, atol=1e-12)
        assert exact.active_multi_indices == generic.active_multi_indices
        assert exact.combination_coefficients == generic.combination_coefficients


def test_level2_axis_cloud_is_feasible_and_moment_exact_in_18_dimensions() -> None:
    cloud = tf_fixed_sgqf_level2_axis_cloud(18)
    mean = tf.linalg.matvec(tf.transpose(cloud.points), cloud.weights)
    reconstructed = _weighted_covariance(cloud.points, cloud.weights)

    assert cloud.point_count == 37
    assert cloud.negative_weight_count == 1
    np.testing.assert_allclose(cloud.weight_total, 1.0, atol=1e-14)
    np.testing.assert_allclose(mean, np.zeros(18), atol=1e-14)
    np.testing.assert_allclose(reconstructed, np.eye(18), atol=1e-12)


def test_fixed_sgqf_level2_4d_cloud_reproduces_standard_normal_covariance() -> None:
    cloud = tf_fixed_sgqf_cloud(4, 2)
    mean = tf.linalg.matvec(tf.transpose(cloud.points), cloud.weights)
    reconstructed = _weighted_covariance(cloud.points, cloud.weights)

    assert cloud.point_count == 9
    np.testing.assert_allclose(cloud.weight_total, 1.0, atol=1e-14)
    np.testing.assert_allclose(mean, np.zeros(4), atol=1e-14)
    np.testing.assert_allclose(reconstructed, np.eye(4), atol=1e-12)



def test_fixed_sgqf_level3_1d_cloud_keeps_the_full_five_point_ghq_rule() -> None:
    rule = tf_standard_normal_ghq_level_rule(3)
    cloud = tf_fixed_sgqf_cloud(1, 3)

    assert rule.point_count == 5
    assert cloud.point_count == 5
    np.testing.assert_allclose(cloud.points[:, 0].numpy(), rule.nodes.numpy(), atol=1e-12)
    np.testing.assert_allclose(cloud.weights.numpy(), rule.weights.numpy(), atol=1e-12)



def test_fixed_sgqf_level3_2d_cloud_matches_jia_example_point_count_and_covariance() -> None:
    cloud = tf_fixed_sgqf_cloud(2, 3)
    mean = tf.linalg.matvec(tf.transpose(cloud.points), cloud.weights)
    reconstructed = _weighted_covariance(cloud.points, cloud.weights)

    assert cloud.point_count == 17
    np.testing.assert_allclose(cloud.weight_total, 1.0, atol=1e-14)
    np.testing.assert_allclose(mean, np.zeros(2), atol=1e-14)
    np.testing.assert_allclose(reconstructed, np.eye(2), atol=1e-12)



def test_fixed_sgqf_branch_identity_is_stable_and_sensitive_to_branch_fields() -> None:
    cloud = tf_fixed_sgqf_cloud(3, 2)
    base = tf_fixed_sgqf_branch_identity(cloud)
    same = tf_fixed_sgqf_branch_identity(cloud)
    changed = tf_fixed_sgqf_branch_identity(cloud, predictive_epsilon=1e-12)

    assert base.hash == same.hash
    assert base.hash != changed.hash


def test_fixed_sgqf_module_does_not_import_numpy_or_call_dot_numpy() -> None:
    text = (ROOT / "bayesfilter" / "nonlinear" / "fixed_sgqf_tf.py").read_text(
        encoding="utf-8"
    )

    assert "import numpy" not in text
    assert "from numpy" not in text
    assert ".numpy(" not in text
