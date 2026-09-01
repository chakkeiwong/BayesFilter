"""Focused tests for the source-faithful ETPF fixture implementation."""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import tensorflow as tf

from bayesfilter.testing.particle_authority_etpf_tf import second_order_etpf_transform


def _fixture() -> tuple[tf.Tensor, tf.Tensor]:
    points = tf.constant(
        [[-2.0, -0.5], [-1.4, 0.2], [-0.7, 1.4], [0.0, -1.1],
         [0.5, 0.6], [1.1, 1.7], [1.8, -0.3], [2.4, 0.9]],
        tf.float64,
    )
    weights = tf.nn.softmax(
        tf.constant([1.1, 0.3, -0.4, 0.7, -0.8, 0.2, -0.2, 0.5], tf.float64)
    )
    return points, weights


def test_second_order_fixture_preserves_constraints() -> None:
    points, weights = _fixture()
    _analysis, diagnostics = second_order_etpf_transform(
        points, weights, sinkhorn_steps=400, riccati_max_steps=2000
    )
    assert bool(diagnostics["riccati_converged"].numpy())
    assert float(diagnostics["corrected_column_residual"].numpy()) <= 2.0e-6
    assert float(diagnostics["corrected_row_residual"].numpy()) <= 2.0e-6
    # The source route stops explicit Euler at a Riccati increment <= 1e-3.
    assert float(diagnostics["mean_residual"].numpy()) <= 1.0e-3
    assert float(diagnostics["covariance_residual"].numpy()) <= 1.0e-3


def test_second_order_fixture_records_possible_negative_correction() -> None:
    points, weights = _fixture()
    _analysis, diagnostics = second_order_etpf_transform(
        points, weights, sinkhorn_steps=400, riccati_max_steps=2000
    )
    assert "corrected_negative_fraction" in diagnostics
    assert float(diagnostics["corrected_negative_fraction"].numpy()) >= 0.0
