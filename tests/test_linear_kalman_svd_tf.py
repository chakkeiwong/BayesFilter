from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.linear.kalman_qr_tf import tf_qr_linear_gaussian_log_likelihood
from bayesfilter.linear.kalman_svd_tf import (
    tf_svd_kalman_log_likelihood,
    tf_svd_linear_gaussian_log_likelihood,
    tf_svd_masked_kalman_log_likelihood,
)
from bayesfilter.linear.svd_factor_tf import psd_eigh
from bayesfilter.linear.types_tf import TFLinearGaussianStateSpace


ROOT = Path(__file__).resolve().parents[1]


def _tiny_model() -> TFLinearGaussianStateSpace:
    return TFLinearGaussianStateSpace(
        initial_mean=tf.constant([0.0], dtype=tf.float64),
        initial_covariance=tf.constant([[0.4]], dtype=tf.float64),
        transition_offset=tf.constant([0.1], dtype=tf.float64),
        transition_matrix=tf.constant([[0.8]], dtype=tf.float64),
        transition_covariance=tf.constant([[0.05]], dtype=tf.float64),
        observation_offset=tf.constant([0.0, 0.2], dtype=tf.float64),
        observation_matrix=tf.constant([[1.0], [0.5]], dtype=tf.float64),
        observation_covariance=tf.constant(
            [[0.08, 0.01], [0.01, 0.10]],
            dtype=tf.float64,
        ),
    )


def test_svd_factor_reconstructs_implemented_covariance_not_raw_covariance() -> None:
    raw_covariance = tf.constant(
        [[2.0, 0.0], [0.0, 1e-14]],
        dtype=tf.float64,
    )
    floor = tf.constant(1e-6, dtype=tf.float64)

    eigenvalues, floored, _vectors, implemented, residual = psd_eigh(
        raw_covariance,
        floor,
    )

    np.testing.assert_allclose(eigenvalues.numpy(), np.array([1e-14, 2.0]), atol=1e-16)
    np.testing.assert_allclose(floored.numpy(), np.array([1e-6, 2.0]), atol=1e-16)
    np.testing.assert_allclose(
        implemented.numpy(),
        np.array([[2.0, 0.0], [0.0, 1e-6]]),
        atol=1e-16,
    )
    assert residual.numpy() > 0.0


def test_svd_dense_value_matches_qr_on_regular_case() -> None:
    model = _tiny_model()
    observations = tf.constant(
        [[0.3, -0.1], [0.2, 0.05], [0.1, 0.04]],
        dtype=tf.float64,
    )

    qr = tf_qr_linear_gaussian_log_likelihood(
        observations,
        model,
        backend="tf_qr",
        jitter=tf.constant(1e-9, dtype=tf.float64),
    )
    svd = tf_svd_linear_gaussian_log_likelihood(
        observations,
        model,
        backend="tf_svd",
        jitter=tf.constant(1e-9, dtype=tf.float64),
        singular_floor=tf.constant(1e-12, dtype=tf.float64),
    )

    np.testing.assert_allclose(
        svd.log_likelihood.numpy(),
        qr.log_likelihood.numpy(),
        atol=1e-9,
    )
    assert svd.metadata.filter_name == "tf_svd_kalman"
    assert svd.metadata.differentiability_status == "value_only"
    assert svd.diagnostics.regularization.derivative_target == "blocked"
    assert svd.diagnostics.extra["factorization"] == "tf.linalg.eigh"


def test_svd_masked_value_matches_masked_qr_on_regular_case() -> None:
    model = _tiny_model()
    observations = tf.constant(
        [[0.3, -0.1], [0.2, 0.05], [0.0, 0.0]],
        dtype=tf.float64,
    )
    mask = tf.constant([[True, False], [True, True], [False, False]], dtype=tf.bool)

    qr = tf_qr_linear_gaussian_log_likelihood(
        observations,
        model,
        backend="tf_masked_qr",
        observation_mask=mask,
        jitter=tf.constant(1e-9, dtype=tf.float64),
    )
    svd = tf_svd_linear_gaussian_log_likelihood(
        observations,
        model,
        backend="tf_masked_svd",
        observation_mask=mask,
        jitter=tf.constant(1e-9, dtype=tf.float64),
        singular_floor=tf.constant(1e-12, dtype=tf.float64),
    )

    np.testing.assert_allclose(
        svd.log_likelihood.numpy(),
        qr.log_likelihood.numpy(),
        atol=1e-9,
    )
    assert svd.metadata.filter_name == "tf_svd_masked_kalman"
    assert svd.diagnostics.mask_convention == "static_dummy_row"


def test_svd_value_reports_floor_diagnostics_for_active_floor() -> None:
    model = _tiny_model()
    observations = tf.constant([[0.0, 0.0]], dtype=tf.float64)

    result = tf_svd_linear_gaussian_log_likelihood(
        observations,
        model,
        backend="tf_svd",
        jitter=tf.constant(1e-9, dtype=tf.float64),
        singular_floor=tf.constant(10.0, dtype=tf.float64),
    )

    assert np.isfinite(result.log_likelihood.numpy())
    assert int(result.diagnostics.regularization.floor_count.numpy()) > 0
    assert result.diagnostics.regularization.psd_projection_residual.numpy() > 0.0
    assert result.diagnostics.regularization.implemented_covariance is not None


def test_svd_masked_all_missing_series_has_zero_likelihood() -> None:
    model = _tiny_model()
    observations = tf.constant([[0.0, 0.0], [0.0, 0.0]], dtype=tf.float64)
    mask = tf.zeros(tf.shape(observations), dtype=tf.bool)

    value, floor_count, residual, implemented_covariance = tf_svd_masked_kalman_log_likelihood(
        observations=observations,
        transition_offset=model.transition_offset,
        transition_matrix=model.transition_matrix,
        transition_covariance=model.transition_covariance,
        observation_offset=model.observation_offset,
        observation_matrix=model.observation_matrix,
        observation_covariance=model.observation_covariance,
        initial_state_mean=model.initial_mean,
        initial_state_covariance=model.initial_covariance,
        observation_mask=mask,
        jitter=tf.constant(1e-9, dtype=tf.float64),
        singular_floor=tf.constant(1e-12, dtype=tf.float64),
    )

    np.testing.assert_allclose(value.numpy(), 0.0, atol=1e-10)
    assert int(floor_count.numpy()) == 0
    np.testing.assert_allclose(residual.numpy(), 0.0, atol=1e-12)
    np.testing.assert_allclose(implemented_covariance.numpy(), np.eye(2), atol=1e-12)


def test_svd_wrapper_rejects_unknown_backend_and_missing_mask() -> None:
    model = _tiny_model()
    observations = tf.constant([[0.3, -0.1]], dtype=tf.float64)

    with pytest.raises(ValueError, match="requires an observation mask"):
        tf_svd_linear_gaussian_log_likelihood(
            observations,
            model,
            backend="tf_masked_svd",
        )
    with pytest.raises(ValueError, match="unknown TensorFlow SVD linear Gaussian backend"):
        tf_svd_linear_gaussian_log_likelihood(
            observations,
            model,
            backend="not_svd",
        )


def test_svd_low_level_dense_function_exposes_implemented_covariance() -> None:
    model = _tiny_model()
    observations = tf.constant([[0.3, -0.1]], dtype=tf.float64)

    value, floor_count, residual, implemented_covariance = tf_svd_kalman_log_likelihood(
        observations=observations,
        transition_offset=model.transition_offset,
        transition_matrix=model.transition_matrix,
        transition_covariance=model.transition_covariance,
        observation_offset=model.observation_offset,
        observation_matrix=model.observation_matrix,
        observation_covariance=model.observation_covariance,
        initial_state_mean=model.initial_mean,
        initial_state_covariance=model.initial_covariance,
        jitter=tf.constant(1e-9, dtype=tf.float64),
        singular_floor=tf.constant(1e-12, dtype=tf.float64),
    )

    assert np.isfinite(value.numpy())
    assert int(floor_count.numpy()) == 0
    assert residual.numpy() >= 0.0
    assert implemented_covariance.shape == (2, 2)


def test_svd_modules_do_not_import_numpy_or_call_dot_numpy() -> None:
    for path in (
        ROOT / "bayesfilter" / "linear" / "svd_factor_tf.py",
        ROOT / "bayesfilter" / "linear" / "kalman_svd_tf.py",
    ):
        text = path.read_text(encoding="utf-8")
        assert "import numpy" not in text
        assert "from numpy" not in text
        assert ".numpy(" not in text
