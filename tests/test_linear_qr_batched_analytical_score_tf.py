from __future__ import annotations

import inspect
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import pytest
import tensorflow as tf

import bayesfilter.linear.kalman_qr_derivatives_tf as qr_derivatives_tf
from bayesfilter.linear.kalman_qr_derivatives_tf import (
    tf_qr_sqrt_kalman_score,
    tf_qr_sqrt_kalman_score_batched_static,
)
from bayesfilter.linear.kalman_qr_tf import (
    tf_qr_sqrt_kalman_log_likelihood_batched_static_while_loop,
)
from bayesfilter.linear.types_tf import (
    TFLinearGaussianStateSpace,
    TFLinearGaussianStateSpaceDerivatives,
)


def _observations(dtype: tf.DType = tf.float64) -> tf.Tensor:
    return tf.constant([[0.18], [0.05], [0.16], [0.11]], dtype=dtype)


def _multi_observations(dtype: tf.DType = tf.float64) -> tf.Tensor:
    return tf.constant(
        [[0.18, -0.04], [0.05, 0.03], [0.16, -0.02], [0.11, 0.01]],
        dtype=dtype,
    )


def _jitter(dtype: tf.DType) -> tf.Tensor:
    return tf.constant(1.0e-9, dtype=dtype)


def _model_and_derivatives(
    params: tf.Tensor,
    *,
    dtype: tf.DType,
) -> tuple[TFLinearGaussianStateSpace, TFLinearGaussianStateSpaceDerivatives]:
    dtype = tf.as_dtype(dtype)
    rho_param, log_measurement_noise = tf.unstack(tf.convert_to_tensor(params, dtype=dtype))
    rho = 0.75 * tf.math.tanh(rho_param)
    drho = 0.75 * (1.0 - tf.math.tanh(rho_param) ** 2)
    measurement_variance = tf.exp(2.0 * log_measurement_noise)
    d_measurement_variance = 2.0 * measurement_variance

    model = TFLinearGaussianStateSpace(
        initial_mean=tf.constant([0.1], dtype=dtype),
        initial_covariance=tf.constant([[0.35]], dtype=dtype),
        transition_offset=tf.constant([0.02], dtype=dtype),
        transition_matrix=tf.reshape(rho, [1, 1]),
        transition_covariance=tf.constant([[0.07]], dtype=dtype),
        observation_offset=tf.constant([0.01], dtype=dtype),
        observation_matrix=tf.constant([[1.2]], dtype=dtype),
        observation_covariance=tf.reshape(measurement_variance, [1, 1]),
    )
    derivatives = TFLinearGaussianStateSpaceDerivatives(
        d_initial_mean=tf.zeros([2, 1], dtype=dtype),
        d_initial_covariance=tf.zeros([2, 1, 1], dtype=dtype),
        d_transition_offset=tf.zeros([2, 1], dtype=dtype),
        d_transition_matrix=tf.reshape(
            tf.stack([drho, tf.constant(0.0, dtype=dtype)]),
            [2, 1, 1],
        ),
        d_transition_covariance=tf.zeros([2, 1, 1], dtype=dtype),
        d_observation_offset=tf.zeros([2, 1], dtype=dtype),
        d_observation_matrix=tf.zeros([2, 1, 1], dtype=dtype),
        d_observation_covariance=tf.reshape(
            tf.stack([tf.constant(0.0, dtype=dtype), d_measurement_variance]),
            [2, 1, 1],
        ),
        d2_initial_mean=tf.zeros([2, 2, 1], dtype=dtype),
        d2_initial_covariance=tf.zeros([2, 2, 1, 1], dtype=dtype),
        d2_transition_offset=tf.zeros([2, 2, 1], dtype=dtype),
        d2_transition_matrix=tf.zeros([2, 2, 1, 1], dtype=dtype),
        d2_transition_covariance=tf.zeros([2, 2, 1, 1], dtype=dtype),
        d2_observation_offset=tf.zeros([2, 2, 1], dtype=dtype),
        d2_observation_matrix=tf.zeros([2, 2, 1, 1], dtype=dtype),
        d2_observation_covariance=tf.zeros([2, 2, 1, 1], dtype=dtype),
    )
    return model, derivatives


def _multi_model_and_derivatives(
    params: tf.Tensor,
    *,
    dtype: tf.DType,
) -> tuple[TFLinearGaussianStateSpace, TFLinearGaussianStateSpaceDerivatives]:
    dtype = tf.as_dtype(dtype)
    raw_rho, raw_loading = tf.unstack(tf.convert_to_tensor(params, dtype=dtype))
    rho = 0.45 * tf.math.tanh(raw_rho)
    drho = 0.45 * (1.0 - tf.math.tanh(raw_rho) ** 2)
    loading = 0.2 * tf.math.tanh(raw_loading)
    dloading = 0.2 * (1.0 - tf.math.tanh(raw_loading) ** 2)
    zero = tf.constant(0.0, dtype=dtype)

    model = TFLinearGaussianStateSpace(
        initial_mean=tf.constant([0.08, -0.04], dtype=dtype),
        initial_covariance=tf.constant([[0.35, 0.03], [0.03, 0.28]], dtype=dtype),
        transition_offset=tf.constant([0.02, -0.01], dtype=dtype),
        transition_matrix=tf.stack(
            [
                tf.stack([rho, tf.constant(0.05, dtype=dtype)]),
                tf.stack([zero, tf.constant(0.35, dtype=dtype)]),
            ],
            axis=0,
        ),
        transition_covariance=tf.constant([[0.08, 0.01], [0.01, 0.06]], dtype=dtype),
        observation_offset=tf.constant([0.01, -0.02], dtype=dtype),
        observation_matrix=tf.stack(
            [
                tf.stack([tf.constant(1.1, dtype=dtype), loading]),
                tf.stack([tf.constant(0.35, dtype=dtype), tf.constant(0.9, dtype=dtype)]),
            ],
            axis=0,
        ),
        observation_covariance=tf.constant([[0.12, 0.015], [0.015, 0.10]], dtype=dtype),
    )
    derivatives = TFLinearGaussianStateSpaceDerivatives(
        d_initial_mean=tf.zeros([2, 2], dtype=dtype),
        d_initial_covariance=tf.zeros([2, 2, 2], dtype=dtype),
        d_transition_offset=tf.zeros([2, 2], dtype=dtype),
        d_transition_matrix=tf.stack(
            [
                tf.stack(
                    [
                        tf.stack([drho, zero]),
                        tf.stack([zero, zero]),
                    ],
                    axis=0,
                ),
                tf.zeros([2, 2], dtype=dtype),
            ],
            axis=0,
        ),
        d_transition_covariance=tf.zeros([2, 2, 2], dtype=dtype),
        d_observation_offset=tf.zeros([2, 2], dtype=dtype),
        d_observation_matrix=tf.stack(
            [
                tf.zeros([2, 2], dtype=dtype),
                tf.stack(
                    [
                        tf.stack([zero, dloading]),
                        tf.stack([zero, zero]),
                    ],
                    axis=0,
                ),
            ],
            axis=0,
        ),
        d_observation_covariance=tf.zeros([2, 2, 2], dtype=dtype),
        d2_initial_mean=tf.zeros([2, 2, 2], dtype=dtype),
        d2_initial_covariance=tf.zeros([2, 2, 2, 2], dtype=dtype),
        d2_transition_offset=tf.zeros([2, 2, 2], dtype=dtype),
        d2_transition_matrix=tf.zeros([2, 2, 2, 2], dtype=dtype),
        d2_transition_covariance=tf.zeros([2, 2, 2, 2], dtype=dtype),
        d2_observation_offset=tf.zeros([2, 2, 2], dtype=dtype),
        d2_observation_matrix=tf.zeros([2, 2, 2, 2], dtype=dtype),
        d2_observation_covariance=tf.zeros([2, 2, 2, 2], dtype=dtype),
    )
    return model, derivatives


def _batch_payload(
    params_batch: tf.Tensor,
    *,
    factory=_model_and_derivatives,
) -> tuple[dict[str, tf.Tensor], tuple[TFLinearGaussianStateSpace, ...], tuple[TFLinearGaussianStateSpaceDerivatives, ...]]:
    models = []
    derivatives = []
    for row in range(int(params_batch.shape[0])):
        model, derivs = factory(params_batch[row], dtype=params_batch.dtype)
        models.append(model)
        derivatives.append(derivs)
    payload = {
        "transition_offset": tf.stack([model.transition_offset for model in models], axis=0),
        "transition_matrix": tf.stack([model.transition_matrix for model in models], axis=0),
        "transition_covariance": tf.stack([model.transition_covariance for model in models], axis=0),
        "observation_offset": tf.stack([model.observation_offset for model in models], axis=0),
        "observation_matrix": tf.stack([model.observation_matrix for model in models], axis=0),
        "observation_covariance": tf.stack(
            [model.observation_covariance for model in models],
            axis=0,
        ),
        "initial_state_mean": tf.stack([model.initial_mean for model in models], axis=0),
        "initial_state_covariance": tf.stack(
            [model.initial_covariance for model in models],
            axis=0,
        ),
        "d_initial_state_mean": tf.stack(
            [derivs.d_initial_mean for derivs in derivatives],
            axis=0,
        ),
        "d_initial_state_covariance": tf.stack(
            [derivs.d_initial_covariance for derivs in derivatives],
            axis=0,
        ),
        "d_transition_offset": tf.stack(
            [derivs.d_transition_offset for derivs in derivatives],
            axis=0,
        ),
        "d_transition_matrix": tf.stack(
            [derivs.d_transition_matrix for derivs in derivatives],
            axis=0,
        ),
        "d_transition_covariance": tf.stack(
            [derivs.d_transition_covariance for derivs in derivatives],
            axis=0,
        ),
        "d_observation_offset": tf.stack(
            [derivs.d_observation_offset for derivs in derivatives],
            axis=0,
        ),
        "d_observation_matrix": tf.stack(
            [derivs.d_observation_matrix for derivs in derivatives],
            axis=0,
        ),
        "d_observation_covariance": tf.stack(
            [derivs.d_observation_covariance for derivs in derivatives],
            axis=0,
        ),
    }
    return payload, tuple(models), tuple(derivatives)


def _batched_score(
    observations: tf.Tensor,
    params_batch: tf.Tensor,
    *,
    factory=_model_and_derivatives,
    jitter_updates_filtered_covariance: bool = True,
) -> tuple[tf.Tensor, tf.Tensor]:
    payload, _models, _derivatives = _batch_payload(params_batch, factory=factory)
    return tf_qr_sqrt_kalman_score_batched_static(
        observations=observations,
        jitter=_jitter(params_batch.dtype),
        jitter_updates_filtered_covariance=jitter_updates_filtered_covariance,
        **payload,
    )


def _scalar_score_rows(
    observations: tf.Tensor,
    params_batch: tf.Tensor,
    *,
    factory=_model_and_derivatives,
    jitter_updates_filtered_covariance: bool = True,
) -> tuple[tf.Tensor, tf.Tensor]:
    values = []
    scores = []
    _payload, models, derivatives = _batch_payload(params_batch, factory=factory)
    for model, derivs in zip(models, derivatives, strict=True):
        value, score = tf_qr_sqrt_kalman_score(
            observations=observations,
            transition_offset=model.transition_offset,
            transition_matrix=model.transition_matrix,
            transition_covariance=model.transition_covariance,
            observation_offset=model.observation_offset,
            observation_matrix=model.observation_matrix,
            observation_covariance=model.observation_covariance,
            initial_state_mean=model.initial_mean,
            initial_state_covariance=model.initial_covariance,
            d_initial_state_mean=derivs.d_initial_mean,
            d_initial_state_covariance=derivs.d_initial_covariance,
            d_transition_offset=derivs.d_transition_offset,
            d_transition_matrix=derivs.d_transition_matrix,
            d_transition_covariance=derivs.d_transition_covariance,
            d_observation_offset=derivs.d_observation_offset,
            d_observation_matrix=derivs.d_observation_matrix,
            d_observation_covariance=derivs.d_observation_covariance,
            jitter=_jitter(params_batch.dtype),
            jitter_updates_filtered_covariance=jitter_updates_filtered_covariance,
        )
        values.append(value)
        scores.append(score)
    return tf.stack(values, axis=0), tf.stack(scores, axis=0)


def _autodiff_batched_reference(
    observations: tf.Tensor,
    params_batch: tf.Tensor,
    *,
    factory=_model_and_derivatives,
    jitter_updates_filtered_covariance: bool = True,
) -> tuple[tf.Tensor, tf.Tensor]:
    with tf.GradientTape() as tape:
        tape.watch(params_batch)
        payload, _models, _derivatives = _batch_payload(params_batch, factory=factory)
        values = tf_qr_sqrt_kalman_log_likelihood_batched_static_while_loop(
            observations=observations,
            transition_offset=payload["transition_offset"],
            transition_matrix=payload["transition_matrix"],
            transition_covariance=payload["transition_covariance"],
            observation_offset=payload["observation_offset"],
            observation_matrix=payload["observation_matrix"],
            observation_covariance=payload["observation_covariance"],
            initial_state_mean=payload["initial_state_mean"],
            initial_state_covariance=payload["initial_state_covariance"],
            jitter=_jitter(params_batch.dtype),
            jitter_updates_filtered_covariance=jitter_updates_filtered_covariance,
        )
    gradient = tape.gradient(values, params_batch)
    assert gradient is not None
    return values, gradient


def test_batched_qr_score_matches_scalar_analytical_rows_float64() -> None:
    params_batch = tf.constant([[0.25, -1.1], [-0.2, -0.9]], dtype=tf.float64)
    observations = _observations(tf.float64)

    batch_value, batch_score = _batched_score(observations, params_batch)
    scalar_value, scalar_score = _scalar_score_rows(observations, params_batch)

    assert batch_value.shape.as_list() == [2]
    assert batch_score.shape.as_list() == [2, 2]
    np.testing.assert_allclose(batch_value.numpy(), scalar_value.numpy(), atol=1.0e-10)
    np.testing.assert_allclose(batch_score.numpy(), scalar_score.numpy(), rtol=1.0e-8, atol=1.0e-9)


def test_batched_qr_score_preserves_float32_and_matches_autodiff_reference() -> None:
    params_batch = tf.constant([[0.25, -1.1], [-0.2, -0.9]], dtype=tf.float32)
    observations = _observations(tf.float32)

    batch_value, batch_score = _batched_score(observations, params_batch)
    autodiff_value, autodiff_score = _autodiff_batched_reference(observations, params_batch)

    assert batch_value.dtype == tf.float32
    assert batch_score.dtype == tf.float32
    np.testing.assert_allclose(batch_value.numpy(), autodiff_value.numpy(), rtol=2.0e-4, atol=2.0e-4)
    np.testing.assert_allclose(batch_score.numpy(), autodiff_score.numpy(), rtol=2.0e-4, atol=2.0e-4)


def test_batched_qr_score_handles_distinct_batch_parameter_and_state_axes() -> None:
    params_batch = tf.constant(
        [[0.25, -0.4], [-0.2, 0.1], [0.05, 0.3]],
        dtype=tf.float64,
    )
    observations = _multi_observations(tf.float64)

    batch_value, batch_score = _batched_score(
        observations,
        params_batch,
        factory=_multi_model_and_derivatives,
        jitter_updates_filtered_covariance=False,
    )
    scalar_value, scalar_score = _scalar_score_rows(
        observations,
        params_batch,
        factory=_multi_model_and_derivatives,
        jitter_updates_filtered_covariance=False,
    )

    assert batch_value.shape.as_list() == [3]
    assert batch_score.shape.as_list() == [3, 2]
    np.testing.assert_allclose(batch_value.numpy(), scalar_value.numpy(), atol=1.0e-10)
    np.testing.assert_allclose(batch_score.numpy(), scalar_score.numpy(), rtol=1.0e-8, atol=1.0e-9)


def test_batched_qr_score_default_jitter_preserves_input_dtype() -> None:
    params_batch = tf.constant([[0.25, -1.1], [-0.2, -0.9]], dtype=tf.float32)
    observations = _observations(tf.float32)
    payload, _models, _derivatives = _batch_payload(params_batch)

    value, score = tf_qr_sqrt_kalman_score_batched_static(
        observations=observations,
        **payload,
    )

    assert value.dtype == tf.float32
    assert score.dtype == tf.float32


def test_batched_qr_score_cpu_xla_preserves_dtype_and_signature() -> None:
    params_batch = tf.constant([[0.25, -1.1], [-0.2, -0.9]], dtype=tf.float32)
    observations = _observations(tf.float32)

    @tf.function(
        input_signature=[tf.TensorSpec(shape=(2, 2), dtype=tf.float32)],
        jit_compile=True,
        reduce_retracing=True,
    )
    def compiled(params: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        return _batched_score(observations, params)

    eager_value, eager_score = _batched_score(observations, params_batch)
    compiled_value, compiled_score = compiled(params_batch)
    second_value, second_score = compiled(tf.identity(params_batch))

    assert compiled_value.dtype == tf.float32
    assert compiled_score.dtype == tf.float32
    np.testing.assert_allclose(compiled_value.numpy(), eager_value.numpy(), rtol=2.0e-4, atol=2.0e-4)
    np.testing.assert_allclose(compiled_score.numpy(), eager_score.numpy(), rtol=2.0e-4, atol=2.0e-4)
    np.testing.assert_allclose(second_value.numpy(), eager_value.numpy(), rtol=2.0e-4, atol=2.0e-4)
    np.testing.assert_allclose(second_score.numpy(), eager_score.numpy(), rtol=2.0e-4, atol=2.0e-4)
    assert len(compiled._list_all_concrete_functions_for_serialization()) == 1


def test_batched_qr_score_source_contract_has_no_scalar_row_wrapper() -> None:
    source = inspect.getsource(tf_qr_sqrt_kalman_score_batched_static.python_function)
    helper_sources = "\n".join(
        inspect.getsource(item)
        for item in (
            tf_qr_sqrt_kalman_score_batched_static.python_function,
            qr_derivatives_tf._batched_stack_qr_lower_factor_first_derivatives,
            qr_derivatives_tf._batched_cholesky_factor_first_derivatives,
            qr_derivatives_tf._batched_factor_covariance_first_derivatives,
            qr_derivatives_tf._batched_factor_solve,
        )
    )

    assert "tf.vectorized_map" not in source
    assert "tf.map_fn" not in source
    assert "tf_qr_sqrt_kalman_score(" not in source
    assert "tf.vectorized_map" not in helper_sources
    assert "tf.map_fn" not in helper_sources
    assert "tf_qr_sqrt_kalman_score(" not in helper_sources


def test_batched_qr_score_rejects_derivative_tensor_without_parameter_axis() -> None:
    params_batch = tf.constant([[0.25, -1.1], [-0.2, -0.9]], dtype=tf.float64)
    observations = _observations(tf.float64)
    payload, _models, _derivatives = _batch_payload(params_batch)
    payload["d_transition_offset"] = payload["d_transition_offset"][:, 0, :]

    with pytest.raises(ValueError, match="d_transition_offset must have rank 3"):
        tf_qr_sqrt_kalman_score_batched_static(
            observations=observations,
            jitter=_jitter(tf.float64),
            **payload,
        )
