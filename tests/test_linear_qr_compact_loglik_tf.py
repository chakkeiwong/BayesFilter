from __future__ import annotations

import inspect
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import tensorflow as tf

from bayesfilter.diagnostics import TFFilterDiagnostics
from bayesfilter.linear.kalman_qr_tf import (
    tf_qr_linear_gaussian_log_likelihood,
    tf_qr_sqrt_kalman_filter,
    tf_qr_sqrt_kalman_log_likelihood_batched_static,
    tf_qr_sqrt_kalman_log_likelihood_batched_static_while_loop,
    tf_qr_sqrt_kalman_log_likelihood_compact,
    tf_qr_sqrt_kalman_log_likelihood_while_loop,
    tf_qr_sqrt_masked_kalman_filter,
    tf_qr_sqrt_masked_kalman_log_likelihood_batched_static,
    tf_qr_sqrt_masked_kalman_log_likelihood_compact,
)
from bayesfilter.linear.types_tf import TFLinearGaussianStateSpace


JITTER = tf.constant(1.0e-9, dtype=tf.float64)


def _jitter(dtype: tf.DType) -> tf.Tensor:
    return tf.constant(1.0e-9, dtype=dtype)


def _model(
    params: tf.Tensor | None = None,
    *,
    dtype: tf.DType = tf.float64,
) -> TFLinearGaussianStateSpace:
    dtype = tf.as_dtype(dtype)
    if params is None:
        rho = tf.constant(0.72, dtype=dtype)
        obs_scale = tf.constant(1.1, dtype=dtype)
        measurement_variance = tf.constant(0.12, dtype=dtype)
    else:
        rho_raw, obs_raw = tf.unstack(tf.convert_to_tensor(params, dtype=dtype))
        rho = 0.8 * tf.math.tanh(rho_raw)
        obs_scale = 1.0 + 0.2 * tf.math.tanh(obs_raw)
        measurement_variance = tf.exp(-2.0 + 0.2 * obs_raw)
    return TFLinearGaussianStateSpace(
        initial_mean=tf.constant([0.15], dtype=dtype),
        initial_covariance=tf.constant([[0.35]], dtype=dtype),
        transition_offset=tf.constant([0.03], dtype=dtype),
        transition_matrix=tf.reshape(rho, [1, 1]),
        transition_covariance=tf.constant([[0.08]], dtype=dtype),
        observation_offset=tf.constant([0.01, -0.02], dtype=dtype),
        observation_matrix=tf.reshape(tf.stack([obs_scale, 0.5 * obs_scale]), [2, 1]),
        observation_covariance=tf.linalg.diag(
            tf.stack([measurement_variance, measurement_variance + tf.constant(0.04, dtype=dtype)])
        ),
    )


def _observations(dtype: tf.DType = tf.float64) -> tf.Tensor:
    return tf.constant(
        [[0.12, 0.04], [0.18, 0.02], [0.08, -0.01], [0.20, 0.03]],
        dtype=dtype,
    )


def _mask() -> tf.Tensor:
    return tf.constant(
        [[True, True], [True, False], [False, False], [True, True]],
        dtype=tf.bool,
    )


def _compact_dense_value(observations: tf.Tensor, model: TFLinearGaussianStateSpace) -> tf.Tensor:
    return tf_qr_sqrt_kalman_log_likelihood_compact(
        observations=observations,
        transition_offset=model.transition_offset,
        transition_matrix=model.transition_matrix,
        transition_covariance=model.transition_covariance,
        observation_offset=model.observation_offset,
        observation_matrix=model.observation_matrix,
        observation_covariance=model.observation_covariance,
        initial_state_mean=model.initial_mean,
        initial_state_covariance=model.initial_covariance,
        jitter=_jitter(model.initial_mean.dtype),
    )


def _while_loop_dense_value(
    observations: tf.Tensor,
    model: TFLinearGaussianStateSpace,
) -> tf.Tensor:
    return tf_qr_sqrt_kalman_log_likelihood_while_loop(
        observations=observations,
        transition_offset=model.transition_offset,
        transition_matrix=model.transition_matrix,
        transition_covariance=model.transition_covariance,
        observation_offset=model.observation_offset,
        observation_matrix=model.observation_matrix,
        observation_covariance=model.observation_covariance,
        initial_state_mean=model.initial_mean,
        initial_state_covariance=model.initial_covariance,
        jitter=_jitter(model.initial_mean.dtype),
    )


def _batched_static_dense_value(
    observations: tf.Tensor,
    models: tuple[TFLinearGaussianStateSpace, ...],
) -> tf.Tensor:
    return tf_qr_sqrt_kalman_log_likelihood_batched_static(
        observations=observations,
        transition_offset=tf.stack([model.transition_offset for model in models], axis=0),
        transition_matrix=tf.stack([model.transition_matrix for model in models], axis=0),
        transition_covariance=tf.stack([model.transition_covariance for model in models], axis=0),
        observation_offset=tf.stack([model.observation_offset for model in models], axis=0),
        observation_matrix=tf.stack([model.observation_matrix for model in models], axis=0),
        observation_covariance=tf.stack([model.observation_covariance for model in models], axis=0),
        initial_state_mean=tf.stack([model.initial_mean for model in models], axis=0),
        initial_state_covariance=tf.stack([model.initial_covariance for model in models], axis=0),
        jitter=_jitter(models[0].initial_mean.dtype),
    )


def _batched_static_while_loop_dense_value(
    observations: tf.Tensor,
    models: tuple[TFLinearGaussianStateSpace, ...],
) -> tf.Tensor:
    return tf_qr_sqrt_kalman_log_likelihood_batched_static_while_loop(
        observations=observations,
        transition_offset=tf.stack([model.transition_offset for model in models], axis=0),
        transition_matrix=tf.stack([model.transition_matrix for model in models], axis=0),
        transition_covariance=tf.stack([model.transition_covariance for model in models], axis=0),
        observation_offset=tf.stack([model.observation_offset for model in models], axis=0),
        observation_matrix=tf.stack([model.observation_matrix for model in models], axis=0),
        observation_covariance=tf.stack([model.observation_covariance for model in models], axis=0),
        initial_state_mean=tf.stack([model.initial_mean for model in models], axis=0),
        initial_state_covariance=tf.stack([model.initial_covariance for model in models], axis=0),
        jitter=_jitter(models[0].initial_mean.dtype),
    )


def _batched_static_masked_value(
    observations: tf.Tensor,
    models: tuple[TFLinearGaussianStateSpace, ...],
) -> tf.Tensor:
    return tf_qr_sqrt_masked_kalman_log_likelihood_batched_static(
        observations=observations,
        transition_offset=tf.stack([model.transition_offset for model in models], axis=0),
        transition_matrix=tf.stack([model.transition_matrix for model in models], axis=0),
        transition_covariance=tf.stack([model.transition_covariance for model in models], axis=0),
        observation_offset=tf.stack([model.observation_offset for model in models], axis=0),
        observation_matrix=tf.stack([model.observation_matrix for model in models], axis=0),
        observation_covariance=tf.stack([model.observation_covariance for model in models], axis=0),
        initial_state_mean=tf.stack([model.initial_mean for model in models], axis=0),
        initial_state_covariance=tf.stack([model.initial_covariance for model in models], axis=0),
        observation_mask=_mask(),
        jitter=_jitter(models[0].initial_mean.dtype),
    )


def _filtered_dense_value(observations: tf.Tensor, model: TFLinearGaussianStateSpace) -> tf.Tensor:
    value, filtered_means, filtered_covariances = tf_qr_sqrt_kalman_filter(
        observations=observations,
        transition_offset=model.transition_offset,
        transition_matrix=model.transition_matrix,
        transition_covariance=model.transition_covariance,
        observation_offset=model.observation_offset,
        observation_matrix=model.observation_matrix,
        observation_covariance=model.observation_covariance,
        initial_state_mean=model.initial_mean,
        initial_state_covariance=model.initial_covariance,
        jitter=_jitter(model.initial_mean.dtype),
    )
    assert filtered_means is not None
    assert filtered_covariances is not None
    return value


def _compact_masked_value(observations: tf.Tensor, model: TFLinearGaussianStateSpace) -> tf.Tensor:
    return tf_qr_sqrt_masked_kalman_log_likelihood_compact(
        observations=observations,
        transition_offset=model.transition_offset,
        transition_matrix=model.transition_matrix,
        transition_covariance=model.transition_covariance,
        observation_offset=model.observation_offset,
        observation_matrix=model.observation_matrix,
        observation_covariance=model.observation_covariance,
        initial_state_mean=model.initial_mean,
        initial_state_covariance=model.initial_covariance,
        observation_mask=_mask(),
        jitter=_jitter(model.initial_mean.dtype),
    )


def _filtered_masked_value(observations: tf.Tensor, model: TFLinearGaussianStateSpace) -> tf.Tensor:
    value, filtered_means, filtered_covariances = tf_qr_sqrt_masked_kalman_filter(
        observations=observations,
        transition_offset=model.transition_offset,
        transition_matrix=model.transition_matrix,
        transition_covariance=model.transition_covariance,
        observation_offset=model.observation_offset,
        observation_matrix=model.observation_matrix,
        observation_covariance=model.observation_covariance,
        initial_state_mean=model.initial_mean,
        initial_state_covariance=model.initial_covariance,
        observation_mask=_mask(),
        jitter=_jitter(model.initial_mean.dtype),
    )
    assert filtered_means is not None
    assert filtered_covariances is not None
    return value


def _assert_float32_matches_float64(value32: tf.Tensor, value64: tf.Tensor) -> None:
    assert value32.dtype == tf.float32
    np.testing.assert_allclose(value32.numpy(), value64.numpy(), rtol=2.0e-5, atol=2.0e-5)


def test_compact_dense_qr_value_matches_existing_filtered_qr_value() -> None:
    observations = _observations()
    model = _model()

    np.testing.assert_allclose(
        _compact_dense_value(observations, model).numpy(),
        _filtered_dense_value(observations, model).numpy(),
        atol=1.0e-12,
    )


def test_while_loop_dense_qr_value_matches_compact_qr_value() -> None:
    observations = _observations()
    model = _model()

    np.testing.assert_allclose(
        _while_loop_dense_value(observations, model).numpy(),
        _compact_dense_value(observations, model).numpy(),
        atol=1.0e-12,
    )


def test_compact_masked_qr_value_matches_existing_filtered_masked_qr_value() -> None:
    observations = _observations()
    model = _model()

    np.testing.assert_allclose(
        _compact_masked_value(observations, model).numpy(),
        _filtered_masked_value(observations, model).numpy(),
        atol=1.0e-12,
    )


def test_qr_value_paths_preserve_float32_and_match_float64_reference() -> None:
    observations32 = _observations(tf.float32)
    observations64 = _observations(tf.float64)
    model32 = _model(dtype=tf.float32)
    model64 = _model(dtype=tf.float64)
    params32 = (
        tf.constant([0.25, -0.4], dtype=tf.float32),
        tf.constant([-0.15, 0.3], dtype=tf.float32),
    )
    params64 = (
        tf.constant([0.25, -0.4], dtype=tf.float64),
        tf.constant([-0.15, 0.3], dtype=tf.float64),
    )
    models32 = tuple(_model(params, dtype=tf.float32) for params in params32)
    models64 = tuple(_model(params, dtype=tf.float64) for params in params64)

    _assert_float32_matches_float64(
        _compact_dense_value(observations32, model32),
        _compact_dense_value(observations64, model64),
    )
    _assert_float32_matches_float64(
        _while_loop_dense_value(observations32, model32),
        _while_loop_dense_value(observations64, model64),
    )
    _assert_float32_matches_float64(
        _batched_static_dense_value(observations32, models32),
        _batched_static_dense_value(observations64, models64),
    )
    _assert_float32_matches_float64(
        _batched_static_while_loop_dense_value(observations32, models32),
        _batched_static_while_loop_dense_value(observations64, models64),
    )
    _assert_float32_matches_float64(
        _compact_masked_value(observations32, model32),
        _compact_masked_value(observations64, model64),
    )
    _assert_float32_matches_float64(
        _batched_static_masked_value(observations32, models32),
        _batched_static_masked_value(observations64, models64),
    )

    dense_value, dense_means, dense_covariances = tf_qr_sqrt_kalman_filter(
        observations=observations32,
        transition_offset=model32.transition_offset,
        transition_matrix=model32.transition_matrix,
        transition_covariance=model32.transition_covariance,
        observation_offset=model32.observation_offset,
        observation_matrix=model32.observation_matrix,
        observation_covariance=model32.observation_covariance,
        initial_state_mean=model32.initial_mean,
        initial_state_covariance=model32.initial_covariance,
        jitter=_jitter(tf.float32),
    )
    masked_value, masked_means, masked_covariances = tf_qr_sqrt_masked_kalman_filter(
        observations=observations32,
        transition_offset=model32.transition_offset,
        transition_matrix=model32.transition_matrix,
        transition_covariance=model32.transition_covariance,
        observation_offset=model32.observation_offset,
        observation_matrix=model32.observation_matrix,
        observation_covariance=model32.observation_covariance,
        initial_state_mean=model32.initial_mean,
        initial_state_covariance=model32.initial_covariance,
        observation_mask=_mask(),
        jitter=_jitter(tf.float32),
    )
    assert dense_value.dtype == tf.float32
    assert dense_means is not None and dense_means.dtype == tf.float32
    assert dense_covariances is not None and dense_covariances.dtype == tf.float32
    assert masked_value.dtype == tf.float32
    assert masked_means is not None and masked_means.dtype == tf.float32
    assert masked_covariances is not None and masked_covariances.dtype == tf.float32


def test_qr_dispatcher_preserves_float32_model_dtype() -> None:
    observations = _observations(tf.float32)
    model = _model(dtype=tf.float32)

    value_only = tf_qr_linear_gaussian_log_likelihood(
        observations,
        model,
        backend="tf_qr",
        jitter=_jitter(tf.float32),
        return_filtered=False,
    )
    filtered = tf_qr_linear_gaussian_log_likelihood(
        observations,
        model,
        backend="tf_qr",
        jitter=_jitter(tf.float32),
        return_filtered=True,
    )

    assert value_only.log_likelihood.dtype == tf.float32
    assert value_only.diagnostics.regularization.jitter.dtype == tf.float32
    assert filtered.log_likelihood.dtype == tf.float32
    assert filtered.filtered_means is not None
    assert filtered.filtered_means.dtype == tf.float32
    assert filtered.filtered_covariances is not None
    assert filtered.filtered_covariances.dtype == tf.float32


def test_dispatcher_preserves_return_filtered_true_and_false_contracts() -> None:
    observations = _observations()
    model = _model()

    value_only = tf_qr_linear_gaussian_log_likelihood(
        observations,
        model,
        backend="tf_qr",
        jitter=JITTER,
        return_filtered=False,
    )
    filtered = tf_qr_linear_gaussian_log_likelihood(
        observations,
        model,
        backend="tf_qr",
        jitter=JITTER,
        return_filtered=True,
    )

    assert value_only.filtered_means is None
    assert value_only.filtered_covariances is None
    assert isinstance(value_only.diagnostics, TFFilterDiagnostics)
    assert value_only.metadata.filter_name == "tf_qr_sqrt_kalman"
    assert value_only.metadata.differentiability_status == "value_only"
    assert value_only.diagnostics.backend == "tf_qr"
    assert value_only.diagnostics.mask_convention == "none"
    assert value_only.diagnostics.regularization.branch_label == "qr_square_root"
    assert filtered.filtered_means is not None
    assert filtered.filtered_covariances is not None
    assert tuple(filtered.filtered_means.shape.as_list()) == (4, 1)
    assert tuple(filtered.filtered_covariances.shape.as_list()) == (4, 1, 1)
    np.testing.assert_allclose(
        value_only.log_likelihood.numpy(),
        filtered.log_likelihood.numpy(),
        atol=1.0e-12,
    )


def test_compact_qr_gradient_matches_existing_filtered_qr_value_gradient() -> None:
    params = tf.constant([0.25, -0.4], dtype=tf.float64)
    observations = _observations()

    with tf.GradientTape() as compact_tape:
        compact_tape.watch(params)
        compact_value = _compact_dense_value(observations, _model(params))
    compact_gradient = compact_tape.gradient(compact_value, params)

    with tf.GradientTape() as filtered_tape:
        filtered_tape.watch(params)
        filtered_value = _filtered_dense_value(observations, _model(params))
    filtered_gradient = filtered_tape.gradient(filtered_value, params)

    np.testing.assert_allclose(compact_value.numpy(), filtered_value.numpy(), atol=1.0e-12)
    np.testing.assert_allclose(
        compact_gradient.numpy(),
        filtered_gradient.numpy(),
        atol=1.0e-10,
    )


def test_batched_static_qr_value_matches_scalar_compact_qr_rows() -> None:
    observations = _observations()
    params_batch = (
        tf.constant([0.25, -0.4], dtype=tf.float64),
        tf.constant([-0.15, 0.3], dtype=tf.float64),
    )
    models = tuple(_model(params) for params in params_batch)

    scalar_values = tf.stack(
        [_compact_dense_value(observations, model) for model in models],
        axis=0,
    )
    batch_values = _batched_static_dense_value(observations, models)

    assert batch_values.shape.as_list() == [2]
    np.testing.assert_allclose(batch_values.numpy(), scalar_values.numpy(), atol=1.0e-12)


def test_batched_static_while_loop_qr_value_matches_existing_batched_static_qr_value() -> None:
    observations = _observations()
    params_batch = (
        tf.constant([0.25, -0.4], dtype=tf.float64),
        tf.constant([-0.15, 0.3], dtype=tf.float64),
    )
    models = tuple(_model(params) for params in params_batch)

    existing_values = _batched_static_dense_value(observations, models)
    while_loop_values = _batched_static_while_loop_dense_value(observations, models)

    assert while_loop_values.shape.as_list() == [2]
    np.testing.assert_allclose(while_loop_values.numpy(), existing_values.numpy(), atol=1.0e-12)


def test_batched_static_qr_gradient_matches_scalar_compact_qr_rows() -> None:
    observations = _observations()
    params_batch = tf.constant([[0.25, -0.4], [-0.15, 0.3]], dtype=tf.float64)

    with tf.GradientTape() as batch_tape:
        batch_tape.watch(params_batch)
        batch_models = tuple(_model(params_batch[row]) for row in range(2))
        batch_value = _batched_static_dense_value(observations, batch_models)
    batch_gradient = batch_tape.gradient(batch_value, params_batch)

    scalar_values = []
    scalar_gradients = []
    for row in range(2):
        params = params_batch[row]
        with tf.GradientTape() as scalar_tape:
            scalar_tape.watch(params)
            scalar_value = _compact_dense_value(observations, _model(params))
        scalar_values.append(scalar_value)
        scalar_gradients.append(scalar_tape.gradient(scalar_value, params))

    assert batch_gradient is not None
    np.testing.assert_allclose(
        batch_value.numpy(),
        tf.stack(scalar_values, axis=0).numpy(),
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        batch_gradient.numpy(),
        tf.stack(scalar_gradients, axis=0).numpy(),
        atol=1.0e-10,
    )


def test_batched_static_masked_qr_value_matches_scalar_compact_qr_rows() -> None:
    observations = _observations()
    params_batch = (
        tf.constant([0.25, -0.4], dtype=tf.float64),
        tf.constant([-0.15, 0.3], dtype=tf.float64),
    )
    models = tuple(_model(params) for params in params_batch)

    scalar_values = tf.stack(
        [_compact_masked_value(observations, model) for model in models],
        axis=0,
    )
    batch_values = _batched_static_masked_value(observations, models)

    assert batch_values.shape.as_list() == [2]
    np.testing.assert_allclose(batch_values.numpy(), scalar_values.numpy(), atol=1.0e-12)


def test_batched_static_masked_qr_gradient_matches_scalar_compact_qr_rows() -> None:
    observations = _observations()
    params_batch = tf.constant([[0.25, -0.4], [-0.15, 0.3]], dtype=tf.float64)

    with tf.GradientTape() as batch_tape:
        batch_tape.watch(params_batch)
        batch_models = tuple(_model(params_batch[row]) for row in range(2))
        batch_value = _batched_static_masked_value(observations, batch_models)
    batch_gradient = batch_tape.gradient(batch_value, params_batch)

    scalar_values = []
    scalar_gradients = []
    for row in range(2):
        params = params_batch[row]
        with tf.GradientTape() as scalar_tape:
            scalar_tape.watch(params)
            scalar_value = _compact_masked_value(observations, _model(params))
        scalar_values.append(scalar_value)
        scalar_gradients.append(scalar_tape.gradient(scalar_value, params))

    assert batch_gradient is not None
    np.testing.assert_allclose(
        batch_value.numpy(),
        tf.stack(scalar_values, axis=0).numpy(),
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        batch_gradient.numpy(),
        tf.stack(scalar_gradients, axis=0).numpy(),
        atol=1.0e-10,
    )


def test_batched_static_qr_fixture_cpu_xla_value_gradient_parity_is_explanatory_only() -> None:
    observations = _observations()
    params_batch = tf.constant([[0.25, -0.4], [-0.15, 0.3]], dtype=tf.float64)

    def value_and_gradient(params: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        with tf.GradientTape() as tape:
            tape.watch(params)
            models = tuple(_model(params[row]) for row in range(2))
            value = _batched_static_dense_value(observations, models)
        gradient = tape.gradient(value, params)
        assert gradient is not None
        return value, gradient

    @tf.function(
        input_signature=[tf.TensorSpec(shape=(2, 2), dtype=tf.float64)],
        jit_compile=True,
        reduce_retracing=True,
    )
    def compiled(params: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        return value_and_gradient(params)

    eager_value, eager_gradient = value_and_gradient(params_batch)
    compiled_value, compiled_gradient = compiled(params_batch)
    compiled(tf.identity(params_batch))

    np.testing.assert_allclose(compiled_value.numpy(), eager_value.numpy(), atol=1.0e-12)
    np.testing.assert_allclose(compiled_gradient.numpy(), eager_gradient.numpy(), atol=1.0e-10)
    assert len(compiled._list_all_concrete_functions_for_serialization()) == 1


def test_compact_qr_fixture_cpu_xla_value_parity_is_explanatory_only() -> None:
    observations = _observations()
    model = _model()

    @tf.function(jit_compile=True, reduce_retracing=True)
    def compiled_dense(obs: tf.Tensor) -> tf.Tensor:
        return _compact_dense_value(obs, model)

    @tf.function(jit_compile=True, reduce_retracing=True)
    def compiled_masked(obs: tf.Tensor) -> tf.Tensor:
        return _compact_masked_value(obs, model)

    np.testing.assert_allclose(
        compiled_dense(observations).numpy(),
        _compact_dense_value(observations, model).numpy(),
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        compiled_masked(observations).numpy(),
        _compact_masked_value(observations, model).numpy(),
        atol=1.0e-12,
    )
    compiled_dense(tf.identity(observations))
    compiled_masked(tf.identity(observations))
    assert len(compiled_dense._list_all_concrete_functions_for_serialization()) == 1
    assert len(compiled_masked._list_all_concrete_functions_for_serialization()) == 1


def test_compact_qr_sources_do_not_accumulate_filtered_output_stacks() -> None:
    for fn in (
        tf_qr_sqrt_kalman_log_likelihood_compact.python_function,
        tf_qr_sqrt_masked_kalman_log_likelihood_compact.python_function,
    ):
        source = inspect.getsource(fn)
        assert "means =" not in source
        assert "covariances =" not in source
        assert ".append(" not in source
        assert "tf.stack" not in source
        assert "TensorArray" not in source


def test_batched_static_qr_source_has_explicit_contract_and_no_chain_map() -> None:
    for fn in (
        tf_qr_sqrt_kalman_log_likelihood_batched_static.python_function,
        tf_qr_sqrt_masked_kalman_log_likelihood_batched_static.python_function,
    ):
        source = inspect.getsource(fn)
        assert "batched-static" in source.lower()
        assert "tf.vectorized_map" not in source
        assert "tf.map_fn" not in source
        assert "TensorArray" not in source
        assert ".append(" not in source


def test_while_loop_qr_sources_use_dynamic_time_loop_without_history_stacks() -> None:
    for fn in (
        tf_qr_sqrt_kalman_log_likelihood_while_loop.python_function,
        tf_qr_sqrt_kalman_log_likelihood_batched_static_while_loop.python_function,
    ):
        source = inspect.getsource(fn)
        assert "tf.while_loop" in source
        assert "for t in range" not in source
        assert "TensorArray" not in source
        assert ".append(" not in source
        assert "tf.stack" not in source
