"""Focused tests for the source-faithful GenUT fixture."""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import tensorflow as tf

from bayesfilter.testing.particle_authority_genut_tf import generalized_unscented_transform


def _fixture() -> tuple[tf.Tensor, tf.Tensor]:
    axis = tf.constant([-2.0, -1.0, 0.0, 1.0, 2.0], tf.float64)
    axis_weights = tf.constant([0.04, 0.22, 0.40, 0.25, 0.09], tf.float64)
    grid = tf.stack(tf.meshgrid(axis, axis, indexing="ij"), axis=-1)
    points = tf.reshape(grid, [25, 2])
    weights = tf.reshape(tf.tensordot(axis_weights, axis_weights, axes=0), [25])
    return points, weights


def test_genut_matches_selected_moments() -> None:
    points, weights = _fixture()
    _sigma_points, sigma_weights, diagnostics = generalized_unscented_transform(
        points, weights
    )
    assert bool(diagnostics["feasible"].numpy())
    assert float(tf.abs(tf.reduce_sum(sigma_weights) - 1.0).numpy()) <= 1.0e-12
    assert float(diagnostics["mean_residual"].numpy()) <= 1.0e-8
    assert float(diagnostics["covariance_residual"].numpy()) <= 1.0e-8
    assert float(diagnostics["third_moment_residual"].numpy()) <= 1.0e-8
    assert float(diagnostics["fourth_moment_residual"].numpy()) <= 1.0e-8


def test_genut_uses_asymmetric_offsets_when_skewed() -> None:
    points, weights = _fixture()
    _sigma_points, _sigma_weights, diagnostics = generalized_unscented_transform(
        points, weights
    )
    assert float(tf.reduce_max(tf.abs(diagnostics["standardized_skewness"])).numpy()) > 1.0e-6
    assert float(tf.reduce_max(tf.abs(diagnostics["u"] - diagnostics["v"])).numpy()) > 1.0e-6
