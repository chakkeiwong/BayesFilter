import inspect
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf
import tensorflow_probability as tfp

from bayesfilter.linear.kalman_tf import (
    _cholesky_validity,
    tf_kalman_log_likelihood,
    tf_linear_gaussian_log_likelihood,
    tf_masked_kalman_filter,
    tf_masked_kalman_filter_checked_batched_static_with_diagnostics,
    tf_masked_kalman_filter_checked_batched_static_value,
    tf_masked_kalman_filter_checked_value,
    tf_masked_kalman_filter_checked_with_diagnostics,
    tf_masked_kalman_filter_with_diagnostics,
    tf_masked_kalman_log_likelihood,
)
from bayesfilter.linear.types_tf import TFLinearGaussianStateSpace


ROOT = Path(__file__).resolve().parents[1]


def test_cholesky_validity_distinguishes_strict_spd_and_tolerated_psd() -> None:
    singular = tf.constant([[0.0, 0.0], [0.0, 1.0]], tf.float64)
    strict_valid, strict_factor = _cholesky_validity(singular)
    tolerant_valid, tolerant_factor = _cholesky_validity(
        singular, psd_tolerance=1.0e-10
    )
    indefinite = tf.constant([[1.0, 0.0], [0.0, -1.0]], tf.float64)
    indefinite_valid, indefinite_factor = _cholesky_validity(indefinite)

    assert not bool(strict_valid)
    assert bool(tolerant_valid)
    assert not bool(indefinite_valid)
    for factor in (strict_factor, tolerant_factor, indefinite_factor):
        assert bool(tf.reduce_all(tf.math.is_finite(factor)))


def _batched_checked_fixture(
    parameters: tf.Tensor,
) -> tuple[tf.Tensor, ...]:
    """Build independent one-state models without mapping over batch rows."""

    parameters = tf.convert_to_tensor(parameters, tf.float64)
    batch_size = int(parameters.shape[0])
    rho = 0.72 + 0.05 * tf.math.tanh(parameters[:, 0])
    observation_scale = 0.08 * tf.exp(0.1 * parameters[:, 1])
    transition_matrix = rho[:, tf.newaxis, tf.newaxis]
    observation_covariance = tf.linalg.diag(
        tf.stack((observation_scale, 1.25 * observation_scale), axis=-1)
    )
    return (
        tf.broadcast_to(tf.constant([[0.1]], tf.float64), [batch_size, 1]),
        transition_matrix,
        tf.broadcast_to(tf.constant([[[0.05]]], tf.float64), [batch_size, 1, 1]),
        tf.broadcast_to(tf.constant([[0.0, 0.2]], tf.float64), [batch_size, 2]),
        tf.broadcast_to(
            tf.constant([[[1.0], [0.5]]], tf.float64),
            [batch_size, 2, 1],
        ),
        observation_covariance,
        tf.broadcast_to(tf.constant([[0.0]], tf.float64), [batch_size, 1]),
        tf.broadcast_to(tf.constant([[[0.4]]], tf.float64), [batch_size, 1, 1]),
    )


def _call_batched_checked(parameters: tf.Tensor) -> tuple[tf.Tensor, ...]:
    observations = tf.constant(
        [[0.1, 0.0], [0.2, -0.1], [0.05, 0.3]], tf.float64
    )
    mask = tf.constant(
        [[True, True], [True, False], [False, True]], tf.bool
    )
    return tf_masked_kalman_filter_checked_batched_static_with_diagnostics(
        observations,
        *_batched_checked_fixture(parameters),
        mask,
        tf.constant(1.0e-9, tf.float64),
    )


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


def _call_dense(
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


def _call_masked(
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


def test_dense_one_step_matches_tfp_predictive_distribution() -> None:
    model = _tiny_model()
    observation = tf.constant([[0.3, -0.1]], dtype=tf.float64)
    jitter = tf.constant(1e-9, dtype=tf.float64)

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
    innovation_mean = model.observation_offset + tf.linalg.matvec(
        model.observation_matrix,
        predicted_mean,
    )
    innovation_covariance = (
        model.observation_matrix
        @ predicted_covariance
        @ tf.transpose(model.observation_matrix)
        + model.observation_covariance
        + jitter * tf.eye(model.observation_dim, dtype=tf.float64)
    )
    expected = tfp.distributions.MultivariateNormalTriL(
        loc=innovation_mean,
        scale_tril=tf.linalg.cholesky(innovation_covariance),
    ).log_prob(observation[0])

    actual = _call_dense(observation, model, jitter=1e-9)

    np.testing.assert_allclose(actual.numpy(), expected.numpy(), atol=1e-10)


def test_dense_accepts_singular_process_covariance_when_innovation_is_pd() -> None:
    model = TFLinearGaussianStateSpace(
        initial_mean=tf.constant([0.0, 0.0], dtype=tf.float64),
        initial_covariance=tf.eye(2, dtype=tf.float64),
        transition_offset=tf.zeros([2], dtype=tf.float64),
        transition_matrix=tf.eye(2, dtype=tf.float64),
        transition_covariance=tf.constant([[0.0, 0.0], [0.0, 0.1]], dtype=tf.float64),
        observation_offset=tf.zeros([1], dtype=tf.float64),
        observation_matrix=tf.constant([[1.0, 1.0]], dtype=tf.float64),
        observation_covariance=tf.constant([[0.2]], dtype=tf.float64),
    )
    observations = tf.constant([[0.1], [0.2]], dtype=tf.float64)

    value = _call_dense(observations, model)

    assert np.isfinite(value.numpy())


def test_masked_all_true_matches_dense_tensorflow_likelihood() -> None:
    model = _tiny_model()
    observations = tf.constant(
        [[0.3, -0.1], [0.2, 0.05], [0.1, 0.04]],
        dtype=tf.float64,
    )
    mask = tf.ones(tf.shape(observations), dtype=tf.bool)

    dense = _call_dense(observations, model)
    masked = _call_masked(observations, model, mask)

    np.testing.assert_allclose(masked.numpy(), dense.numpy(), atol=1e-10)


def test_masked_all_missing_row_contributes_zero_likelihood_and_predicts() -> None:
    model = _tiny_model()
    observations = tf.constant([[0.0, 0.0]], dtype=tf.float64)
    mask = tf.zeros(tf.shape(observations), dtype=tf.bool)

    result = tf_linear_gaussian_log_likelihood(
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
    assert result.metadata.filter_name == "tf_masked_cholesky_kalman"
    assert result.diagnostics.mask_convention == "static_dummy_row"


def test_masked_diagnostics_preserve_likelihood_and_filtered_state() -> None:
    model = _tiny_model()
    observations = tf.constant(
        [[0.3, -0.1], [0.2, 0.05], [0.1, 0.04]],
        dtype=tf.float64,
    )
    mask = tf.constant(
        [[True, True], [True, False], [False, False]],
        dtype=tf.bool,
    )
    arguments = dict(
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
    )

    value, means, covariances = tf_masked_kalman_filter(**arguments, return_filtered=True)
    diagnostic_value, minimums, conditions, final_mean, final_covariance = (
        tf_masked_kalman_filter_with_diagnostics(**arguments)
    )

    np.testing.assert_allclose(diagnostic_value.numpy(), value.numpy(), atol=0.0)
    np.testing.assert_allclose(final_mean.numpy(), means.numpy()[-1], atol=0.0)
    np.testing.assert_allclose(
        final_covariance.numpy(), covariances.numpy()[-1], atol=0.0
    )
    assert minimums.shape == [3]
    assert conditions.shape == [3]
    assert bool(tf.reduce_all(minimums > 0.0))
    assert bool(tf.reduce_all(conditions >= 1.0))


def test_masked_diagnostics_all_missing_row_reports_dummy_identity() -> None:
    model = _tiny_model()
    observations = tf.zeros([1, model.observation_dim], dtype=tf.float64)
    mask = tf.zeros(tf.shape(observations), dtype=tf.bool)

    value, minimums, conditions, _, _ = tf_masked_kalman_filter_with_diagnostics(
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
    )

    np.testing.assert_allclose(value.numpy(), 0.0, atol=1e-12)
    np.testing.assert_allclose(minimums.numpy(), [1.0 + 1e-9], atol=1e-12)
    np.testing.assert_allclose(conditions.numpy(), [1.0], atol=1e-12)


def test_checked_masked_filter_matches_exact_valid_recursion() -> None:
    model = _tiny_model()
    observations = tf.constant([[0.3, -0.1], [0.2, 0.05]], tf.float64)
    mask = tf.constant([[True, True], [True, False]], tf.bool)
    arguments = dict(
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
        jitter=tf.constant(1e-9, tf.float64),
    )
    expected = tf_masked_kalman_filter_with_diagnostics(**arguments)
    actual = tf_masked_kalman_filter_checked_with_diagnostics(**arguments)
    np.testing.assert_allclose(actual[0].numpy(), expected[0].numpy(), atol=0.0)
    np.testing.assert_allclose(actual[1].numpy(), expected[1].numpy(), atol=0.0)
    np.testing.assert_allclose(actual[2].numpy(), expected[2].numpy(), atol=0.0)
    np.testing.assert_allclose(actual[4].numpy(), expected[3].numpy(), atol=0.0)
    np.testing.assert_allclose(actual[5].numpy(), expected[4].numpy(), atol=0.0)
    assert bool(tf.reduce_all(actual[3]))


def test_checked_masked_filter_rejects_indefinite_innovation_before_cholesky() -> None:
    value, minimums, conditions, validity, final_mean, final_covariance = (
        tf_masked_kalman_filter_checked_with_diagnostics(
            observations=tf.zeros([1, 1], tf.float64),
            transition_offset=tf.zeros([1], tf.float64),
            transition_matrix=tf.eye(1, dtype=tf.float64),
            transition_covariance=tf.zeros([1, 1], tf.float64),
            observation_offset=tf.zeros([1], tf.float64),
            observation_matrix=tf.ones([1, 1], tf.float64),
            observation_covariance=tf.constant([[-2.0]], tf.float64),
            initial_state_mean=tf.zeros([1], tf.float64),
            initial_state_covariance=tf.eye(1, dtype=tf.float64),
            observation_mask=tf.ones([1, 1], tf.bool),
        )
    )
    assert bool(tf.math.is_finite(value))
    assert float(minimums[0]) < 0.0
    np.testing.assert_allclose(conditions.numpy(), [0.0], atol=0.0)
    assert not bool(validity[0])
    np.testing.assert_allclose(final_mean.numpy(), [0.0], atol=0.0)
    np.testing.assert_allclose(final_covariance.numpy(), [[1.0]], atol=0.0)


def test_checked_value_path_matches_valid_diagnostic_value_and_gradient() -> None:
    observations = tf.constant([[0.1], [0.0], [-0.05]], tf.float64)

    def evaluate(raw_transition: tf.Tensor, *, value_only: bool):
        with tf.GradientTape() as tape:
            tape.watch(raw_transition)
            transition = tf.reshape(
                0.8 * tf.math.sigmoid(raw_transition),
                [1, 1],
            )
            arguments = dict(
                observations=observations,
                transition_offset=tf.zeros([1], tf.float64),
                transition_matrix=transition,
                transition_covariance=tf.constant([[0.05]], tf.float64),
                observation_offset=tf.zeros([1], tf.float64),
                observation_matrix=tf.ones([1, 1], tf.float64),
                observation_covariance=tf.constant([[0.08]], tf.float64),
                initial_state_mean=tf.zeros([1], tf.float64),
                initial_state_covariance=tf.constant([[0.4]], tf.float64),
                observation_mask=tf.ones([3, 1], tf.bool),
            )
            if value_only:
                value, valid = tf_masked_kalman_filter_checked_value(**arguments)
            else:
                value, _, _, validity, _, _ = (
                    tf_masked_kalman_filter_checked_with_diagnostics(**arguments)
                )
                valid = tf.reduce_all(validity)
        return value, tape.gradient(value, raw_transition), valid

    raw = tf.constant(0.15, tf.float64)
    diagnostic = evaluate(raw, value_only=False)
    value_only = evaluate(raw, value_only=True)
    tf.debugging.assert_equal(value_only[0], diagnostic[0])
    tf.debugging.assert_equal(value_only[1], diagnostic[1])
    tf.debugging.assert_equal(value_only[2], diagnostic[2])


def test_checked_value_path_matches_invalid_diagnostic_rejection() -> None:
    arguments = dict(
        observations=tf.zeros([1, 1], tf.float64),
        transition_offset=tf.zeros([1], tf.float64),
        transition_matrix=tf.eye(1, dtype=tf.float64),
        transition_covariance=tf.zeros([1, 1], tf.float64),
        observation_offset=tf.zeros([1], tf.float64),
        observation_matrix=tf.ones([1, 1], tf.float64),
        observation_covariance=tf.constant([[-2.0]], tf.float64),
        initial_state_mean=tf.zeros([1], tf.float64),
        initial_state_covariance=tf.eye(1, dtype=tf.float64),
        observation_mask=tf.ones([1, 1], tf.bool),
    )
    diagnostic = tf_masked_kalman_filter_checked_with_diagnostics(**arguments)
    value, valid = tf_masked_kalman_filter_checked_value(**arguments)
    tf.debugging.assert_equal(value, diagnostic[0])
    tf.debugging.assert_equal(valid, tf.reduce_all(diagnostic[3]))
    assert bool(tf.math.is_finite(value))
    assert not bool(valid)


def test_checked_masked_filter_repeated_compiled_gradient_static_horizon() -> None:
    observations = tf.zeros([4, 1], tf.float64)

    def value_and_score(raw_transition: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        with tf.GradientTape() as tape:
            tape.watch(raw_transition)
            transition = tf.reshape(0.8 * tf.math.sigmoid(raw_transition), [1, 1])
            value, _, _, validity, _, _ = (
                tf_masked_kalman_filter_checked_with_diagnostics(
                    observations=observations,
                    transition_offset=tf.zeros([1], tf.float64),
                    transition_matrix=transition,
                    transition_covariance=tf.constant([[0.05]], tf.float64),
                    observation_offset=tf.zeros([1], tf.float64),
                    observation_matrix=tf.ones([1, 1], tf.float64),
                    observation_covariance=tf.constant([[0.08]], tf.float64),
                    initial_state_mean=tf.zeros([1], tf.float64),
                    initial_state_covariance=tf.constant([[0.4]], tf.float64),
                    observation_mask=tf.ones([4, 1], tf.bool),
                )
            )
        score = tape.gradient(value, raw_transition)
        return value, score, tf.reduce_all(validity)

    first = value_and_score(tf.constant(0.1, tf.float64))
    second = value_and_score(tf.constant(0.2, tf.float64))
    assert bool(first[2]) and bool(second[2])
    assert bool(tf.math.is_finite(first[0])) and bool(tf.math.is_finite(second[0]))
    assert bool(tf.math.is_finite(first[1])) and bool(tf.math.is_finite(second[1]))


def test_checked_masked_filter_xla_value_and_gradient_match_eager() -> None:
    observations = tf.constant([[0.1], [0.0]], tf.float64)

    def value_and_score(raw_transition: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        with tf.GradientTape() as tape:
            tape.watch(raw_transition)
            transition = tf.reshape(0.8 * tf.math.sigmoid(raw_transition), [1, 1])
            value, _, _, validity, _, _ = (
                tf_masked_kalman_filter_checked_with_diagnostics(
                    observations=observations,
                    transition_offset=tf.zeros([1], tf.float64),
                    transition_matrix=transition,
                    transition_covariance=tf.constant([[0.05]], tf.float64),
                    observation_offset=tf.zeros([1], tf.float64),
                    observation_matrix=tf.ones([1, 1], tf.float64),
                    observation_covariance=tf.constant([[0.08]], tf.float64),
                    initial_state_mean=tf.zeros([1], tf.float64),
                    initial_state_covariance=tf.constant([[0.4]], tf.float64),
                    observation_mask=tf.ones([2, 1], tf.bool),
                )
            )
            value = tf.where(
                tf.reduce_all(validity), value, tf.constant(-1.0e100, tf.float64)
            )
        return value, tape.gradient(value, raw_transition)

    compiled = tf.function(
        value_and_score,
        input_signature=[tf.TensorSpec([], tf.float64)],
        jit_compile=True,
    )
    raw = tf.constant(0.15, tf.float64)
    eager_value, eager_score = value_and_score(raw)
    xla_value, xla_score = compiled(raw)
    np.testing.assert_allclose(xla_value.numpy(), eager_value.numpy(), atol=1e-12)
    np.testing.assert_allclose(xla_score.numpy(), eager_score.numpy(), atol=1e-12)


def test_wrapper_uses_model_mask_and_preserves_tensor_result() -> None:
    mask = tf.constant([[True, False], [True, True]], dtype=tf.bool)
    model = TFLinearGaussianStateSpace(
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
        observation_mask=mask,
    )
    observations = tf.constant([[0.3, -0.1], [0.2, 0.05]], dtype=tf.float64)

    result = tf_linear_gaussian_log_likelihood(observations, model, return_filtered=True)
    direct = _call_masked(observations, model, mask, jitter=0.0)

    assert isinstance(result.log_likelihood, tf.Tensor)
    assert isinstance(result.filtered_means, tf.Tensor)
    np.testing.assert_allclose(result.log_likelihood.numpy(), direct.numpy(), atol=1e-10)


def test_mask_shape_mismatch_raises_clear_error() -> None:
    model = _tiny_model()
    observations = tf.constant([[0.3, -0.1]], dtype=tf.float64)
    bad_mask = tf.ones([1, 3], dtype=tf.bool)

    with pytest.raises(tf.errors.InvalidArgumentError, match="Observation mask shape"):
        _call_masked(observations, model, bad_mask).numpy()


def test_tf_function_reuses_concrete_function_for_same_shape_masks() -> None:
    model = _tiny_model()
    observations = tf.constant([[0.3, -0.1], [0.2, 0.05]], dtype=tf.float64)
    mask_a = tf.constant([[True, False], [True, True]], dtype=tf.bool)
    mask_b = tf.constant([[False, True], [True, True]], dtype=tf.bool)

    @tf.function(reduce_retracing=True)
    def compiled(mask: tf.Tensor) -> tf.Tensor:
        return _call_masked(observations, model, mask)

    first = compiled(mask_a)
    second = compiled(mask_b)

    assert np.isfinite(first.numpy())
    assert np.isfinite(second.numpy())
    assert len(compiled._list_all_concrete_functions_for_serialization()) == 1


def test_checked_batched_static_matches_scalar_rows_and_scores() -> None:
    parameters = tf.constant([[0.2, -0.4], [-0.35, 0.3]], tf.float64)
    observations = tf.constant(
        [[0.1, 0.0], [0.2, -0.1], [0.05, 0.3]], tf.float64
    )
    mask = tf.constant(
        [[True, True], [True, False], [False, True]], tf.bool
    )
    with tf.GradientTape() as tape:
        tape.watch(parameters)
        batch_value, batch_minimum, batch_condition, batch_valid = (
            _call_batched_checked(parameters)
        )
        batch_total = tf.reduce_sum(batch_value)
    batch_score = tape.gradient(batch_total, parameters)
    assert batch_score is not None

    scalar_values = []
    scalar_scores = []
    scalar_minimums = []
    scalar_conditions = []
    for row_index in range(2):
        row = parameters[row_index]
        with tf.GradientTape() as row_tape:
            row_tape.watch(row)
            tensors = _batched_checked_fixture(row[tf.newaxis, :])
            result = tf_masked_kalman_filter_checked_with_diagnostics(
                observations,
                *(tensor[0] for tensor in tensors),
                mask,
                tf.constant(1.0e-9, tf.float64),
            )
            row_value = result[0]
        row_score = row_tape.gradient(row_value, row)
        assert row_score is not None
        scalar_values.append(row_value)
        scalar_scores.append(row_score)
        scalar_minimums.append(tf.reduce_min(result[1]))
        scalar_conditions.append(tf.reduce_max(result[2]))
        assert bool(tf.reduce_all(result[3]).numpy())

    np.testing.assert_allclose(
        batch_value.numpy(), tf.stack(scalar_values).numpy(), atol=1.0e-12
    )
    np.testing.assert_allclose(
        batch_score.numpy(), tf.stack(scalar_scores).numpy(), atol=2.0e-12
    )
    np.testing.assert_allclose(
        batch_minimum.numpy(), tf.stack(scalar_minimums).numpy(), atol=1.0e-12
    )
    np.testing.assert_allclose(
        batch_condition.numpy(), tf.stack(scalar_conditions).numpy(), atol=1.0e-12
    )
    assert bool(tf.reduce_all(batch_valid).numpy())


def test_checked_batched_static_value_rejects_one_row_without_cross_talk() -> None:
    parameters = tf.constant([[0.2, -0.4], [-0.35, 0.3]], tf.float64)
    tensors = list(_batched_checked_fixture(parameters))
    tensors[5] = tf.concat(
        [tensors[5][:1], -100.0 * tf.eye(2, batch_shape=[1], dtype=tf.float64)],
        axis=0,
    )
    observations = tf.constant(
        [[0.1, 0.0], [0.2, -0.1], [0.05, 0.3]], tf.float64
    )
    mask = tf.constant(
        [[True, True], [True, False], [False, True]], tf.bool
    )
    value, valid = tf_masked_kalman_filter_checked_batched_static_value(
        observations,
        *tensors,
        mask,
        tf.constant(1.0e-9, tf.float64),
    )
    reference = _call_batched_checked(parameters[:1])[0]
    np.testing.assert_allclose(value[:1].numpy(), reference.numpy(), atol=1.0e-12)
    assert valid.numpy().tolist() == [True, False]
    assert bool(tf.reduce_all(tf.math.is_finite(value)).numpy())


def test_checked_batched_static_compiles_with_xla_and_has_no_row_mapping() -> None:
    parameters = tf.constant([[0.2, -0.4], [-0.35, 0.3]], tf.float64)

    @tf.function(
        input_signature=[tf.TensorSpec([2, 2], tf.float64)],
        jit_compile=True,
    )
    def compiled(values: tf.Tensor) -> tuple[tf.Tensor, ...]:
        return _call_batched_checked(values)

    expected = _call_batched_checked(parameters)
    actual = compiled(parameters)
    for expected_tensor, actual_tensor in zip(expected, actual, strict=True):
        if expected_tensor.dtype == tf.bool:
            np.testing.assert_array_equal(actual_tensor.numpy(), expected_tensor.numpy())
        else:
            np.testing.assert_allclose(
                actual_tensor.numpy(), expected_tensor.numpy(), rtol=1.0e-11, atol=1.0e-11
            )
    source = inspect.getsource(
        tf_masked_kalman_filter_checked_batched_static_with_diagnostics.python_function
    )
    assert "tf.map_fn" not in source
    assert "tf.vectorized_map" not in source
    assert "tf.unstack" not in source
    assert "for row" not in source


def test_wrapper_rejects_unknown_backend() -> None:
    model = _tiny_model()
    observations = tf.constant([[0.3, -0.1]], dtype=tf.float64)

    with pytest.raises(ValueError, match="unknown TensorFlow linear Gaussian backend"):
        tf_linear_gaussian_log_likelihood(observations, model, backend="not_a_backend")


def test_tf_linear_kalman_module_does_not_import_numpy_or_call_dot_numpy() -> None:
    text = (ROOT / "bayesfilter" / "linear" / "kalman_tf.py").read_text(
        encoding="utf-8"
    )

    assert "import numpy" not in text
    assert "from numpy" not in text
    assert ".numpy(" not in text
