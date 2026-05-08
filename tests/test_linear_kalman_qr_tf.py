from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.linear.kalman_qr_tf import (
    tf_qr_linear_gaussian_log_likelihood,
    tf_qr_sqrt_kalman_log_likelihood,
    tf_qr_sqrt_masked_kalman_log_likelihood,
)
from bayesfilter.linear.kalman_tf import (
    tf_kalman_log_likelihood,
    tf_linear_gaussian_log_likelihood,
    tf_masked_kalman_log_likelihood,
)
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


def _call_dense_qr(
    observations: tf.Tensor,
    model: TFLinearGaussianStateSpace,
    *,
    jitter: float = 1e-9,
) -> tf.Tensor:
    return tf_qr_sqrt_kalman_log_likelihood(
        observations=observations,
        transition_offset=model.transition_offset,
        transition_matrix=model.transition_matrix,
        transition_covariance=model.transition_covariance,
        observation_offset=model.observation_offset,
        observation_matrix=model.observation_matrix,
        observation_covariance=model.observation_covariance,
        initial_state_mean=model.initial_mean,
        initial_state_covariance=model.initial_covariance,
        jitter=tf.constant(jitter, dtype=tf.float64),
    )


def _call_dense_cholesky(
    observations: tf.Tensor,
    model: TFLinearGaussianStateSpace,
    *,
    jitter: float = 1e-9,
) -> tf.Tensor:
    return tf_kalman_log_likelihood(
        observations=observations,
        transition_offset=model.transition_offset,
        transition_matrix=model.transition_matrix,
        transition_covariance=model.transition_covariance,
        observation_offset=model.observation_offset,
        observation_matrix=model.observation_matrix,
        observation_covariance=model.observation_covariance,
        initial_state_mean=model.initial_mean,
        initial_state_covariance=model.initial_covariance,
        jitter=tf.constant(jitter, dtype=tf.float64),
    )


def _call_masked_qr(
    observations: tf.Tensor,
    model: TFLinearGaussianStateSpace,
    mask: tf.Tensor,
    *,
    jitter: float = 1e-9,
) -> tf.Tensor:
    return tf_qr_sqrt_masked_kalman_log_likelihood(
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
        jitter=tf.constant(jitter, dtype=tf.float64),
    )


def _call_masked_cholesky(
    observations: tf.Tensor,
    model: TFLinearGaussianStateSpace,
    mask: tf.Tensor,
    *,
    jitter: float = 1e-9,
) -> tf.Tensor:
    return tf_masked_kalman_log_likelihood(
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
        jitter=tf.constant(jitter, dtype=tf.float64),
    )


def test_dense_qr_matches_dense_cholesky_value_backend() -> None:
    model = _tiny_model()
    observations = tf.constant(
        [[0.3, -0.1], [0.2, 0.05], [0.1, 0.04]],
        dtype=tf.float64,
    )

    qr_value = _call_dense_qr(observations, model)
    cholesky_value = _call_dense_cholesky(observations, model)

    np.testing.assert_allclose(qr_value.numpy(), cholesky_value.numpy(), atol=1e-10)


def test_masked_qr_all_true_matches_dense_qr_and_cholesky_masked() -> None:
    model = _tiny_model()
    observations = tf.constant(
        [[0.3, -0.1], [0.2, 0.05], [0.1, 0.04]],
        dtype=tf.float64,
    )
    mask = tf.ones(tf.shape(observations), dtype=tf.bool)

    dense_qr = _call_dense_qr(observations, model)
    masked_qr = _call_masked_qr(observations, model, mask)
    masked_cholesky = _call_masked_cholesky(observations, model, mask)

    np.testing.assert_allclose(masked_qr.numpy(), dense_qr.numpy(), atol=1e-10)
    np.testing.assert_allclose(masked_qr.numpy(), masked_cholesky.numpy(), atol=1e-10)


def test_masked_qr_sparse_rows_match_cholesky_masked() -> None:
    model = _tiny_model()
    observations = tf.constant(
        [[0.3, -0.1], [0.2, 0.05], [0.1, 0.04]],
        dtype=tf.float64,
    )
    mask = tf.constant([[True, False], [True, True], [False, True]], dtype=tf.bool)

    masked_qr = _call_masked_qr(observations, model, mask)
    masked_cholesky = _call_masked_cholesky(observations, model, mask)

    np.testing.assert_allclose(masked_qr.numpy(), masked_cholesky.numpy(), atol=1e-10)


def test_masked_qr_all_missing_row_contributes_zero_likelihood_and_predicts() -> None:
    model = _tiny_model()
    observations = tf.constant([[0.0, 0.0]], dtype=tf.float64)
    mask = tf.zeros(tf.shape(observations), dtype=tf.bool)

    result = tf_qr_linear_gaussian_log_likelihood(
        observations,
        model,
        observation_mask=mask,
        return_filtered=True,
        jitter=tf.constant(1e-9, dtype=tf.float64),
    )
    predicted_mean = model.transition_offset + tf.linalg.matvec(
        model.transition_matrix,
        model.initial_mean,
    )
    predicted_covariance = (
        model.transition_matrix
        @ model.initial_covariance
        @ tf.transpose(model.transition_matrix)
        + model.transition_covariance
    )

    np.testing.assert_allclose(result.log_likelihood.numpy(), 0.0, atol=1e-10)
    np.testing.assert_allclose(result.filtered_means.numpy()[0], predicted_mean.numpy(), atol=1e-12)
    np.testing.assert_allclose(
        result.filtered_covariances.numpy()[0],
        predicted_covariance.numpy(),
        atol=1e-12,
    )
    assert result.metadata.filter_name == "tf_qr_sqrt_masked_kalman"
    assert result.diagnostics.mask_convention == "static_dummy_row"


def test_qr_wrapper_rejects_unknown_backend_and_missing_mask() -> None:
    model = _tiny_model()
    observations = tf.constant([[0.3, -0.1]], dtype=tf.float64)

    with pytest.raises(ValueError, match="unknown TensorFlow QR linear Gaussian backend"):
        tf_qr_linear_gaussian_log_likelihood(observations, model, backend="not_a_backend")
    with pytest.raises(ValueError, match="requires an observation mask"):
        tf_qr_linear_gaussian_log_likelihood(observations, model, backend="tf_masked_qr")


def test_qr_mask_shape_mismatch_raises_clear_error() -> None:
    model = _tiny_model()
    observations = tf.constant([[0.3, -0.1]], dtype=tf.float64)
    bad_mask = tf.ones([1, 3], dtype=tf.bool)

    with pytest.raises(tf.errors.InvalidArgumentError, match="Observation mask shape"):
        _call_masked_qr(observations, model, bad_mask).numpy()


def test_qr_tf_function_reuses_concrete_function_for_same_shape_masks() -> None:
    model = _tiny_model()
    observations = tf.constant([[0.3, -0.1], [0.2, 0.05]], dtype=tf.float64)
    mask_a = tf.constant([[True, False], [True, True]], dtype=tf.bool)
    mask_b = tf.constant([[False, True], [True, True]], dtype=tf.bool)

    @tf.function(reduce_retracing=True)
    def compiled(mask: tf.Tensor) -> tf.Tensor:
        return _call_masked_qr(observations, model, mask)

    first = compiled(mask_a)
    second = compiled(mask_b)

    assert np.isfinite(first.numpy())
    assert np.isfinite(second.numpy())
    assert len(compiled._list_all_concrete_functions_for_serialization()) == 1


def test_qr_module_does_not_import_numpy_or_call_dot_numpy() -> None:
    text = (ROOT / "bayesfilter" / "linear" / "kalman_qr_tf.py").read_text(
        encoding="utf-8"
    )

    assert "import numpy" not in text
    assert "from numpy" not in text
    assert ".numpy(" not in text
