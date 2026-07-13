from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.linear.dtypes_tf import as_float_tensor, common_floating_dtype
from bayesfilter.results_tf import TFFilterDerivativeResult
from bayesfilter.structural import FilterRunMetadata
from bayesfilter.linear.types_tf import (
    TFLinearGaussianStateSpace,
    TFLinearGaussianStateSpaceDerivatives,
    TFLinearGaussianStateSpaceFirstDerivatives,
)


def test_common_floating_dtype_preserves_explicit_float32() -> None:
    values = tf.constant([1.0, 2.0], dtype=tf.float32)
    mask = tf.constant([True, False])

    dtype = common_floating_dtype(values, mask, [3.0, 4.0])

    assert dtype == tf.float32
    assert as_float_tensor([3.0, 4.0], dtype).dtype == tf.float32


def test_common_floating_dtype_preserves_explicit_float64() -> None:
    values = tf.constant([1.0, 2.0], dtype=tf.float64)

    dtype = common_floating_dtype(values)

    assert dtype == tf.float64
    assert as_float_tensor([3.0, 4.0], dtype).dtype == tf.float64


def test_common_floating_dtype_uses_historical_float64_default_for_literals() -> None:
    dtype = common_floating_dtype([1.0, 2.0], 3.0)

    assert dtype == tf.float64
    assert as_float_tensor([1.0, 2.0], dtype).dtype == tf.float64


def test_common_floating_dtype_accepts_numpy_explicit_dtype() -> None:
    values = np.asarray([1.0, 2.0], dtype=np.float32)

    dtype = common_floating_dtype(values)

    assert dtype == tf.float32


def test_common_floating_dtype_rejects_mixed_floating_dtypes() -> None:
    values32 = tf.constant([1.0], dtype=tf.float32)
    values64 = tf.constant([1.0], dtype=tf.float64)

    with pytest.raises(TypeError, match="must share one floating dtype"):
        common_floating_dtype(values32, values64)


def test_common_floating_dtype_rejects_unsupported_floating_dtype() -> None:
    values = tf.constant([1.0], dtype=tf.float16)

    with pytest.raises(TypeError, match="unsupported dtype float16"):
        common_floating_dtype(values)


def test_as_float_tensor_rejects_unsupported_requested_dtype() -> None:
    with pytest.raises(TypeError, match="unsupported dtype float16"):
        as_float_tensor([1.0], tf.float16)


def test_common_floating_dtype_works_in_cpu_xla_trace() -> None:
    @tf.function(
        input_signature=[tf.TensorSpec(shape=[2], dtype=tf.float32)],
        jit_compile=True,
        reduce_retracing=True,
    )
    def compiled(values: tf.Tensor) -> tf.Tensor:
        dtype = common_floating_dtype(values)
        offset = as_float_tensor([0.5, 1.5], dtype)
        return values + offset

    result = compiled(tf.constant([1.0, 2.0], dtype=tf.float32))

    assert result.dtype == tf.float32
    np.testing.assert_allclose(result.numpy(), np.asarray([1.5, 3.5], dtype=np.float32))


def test_linear_gaussian_state_space_preserves_explicit_float32_dtype() -> None:
    model = TFLinearGaussianStateSpace(
        initial_mean=tf.constant([0.0], dtype=tf.float32),
        initial_covariance=tf.constant([[1.0]], dtype=tf.float32),
        transition_offset=tf.constant([0.0], dtype=tf.float32),
        transition_matrix=tf.constant([[0.8]], dtype=tf.float32),
        transition_covariance=tf.constant([[0.1]], dtype=tf.float32),
        observation_offset=tf.constant([0.0], dtype=tf.float32),
        observation_matrix=tf.constant([[1.0]], dtype=tf.float32),
        observation_covariance=tf.constant([[0.2]], dtype=tf.float32),
    )

    for name in (
        "initial_mean",
        "initial_covariance",
        "transition_offset",
        "transition_matrix",
        "transition_covariance",
        "observation_offset",
        "observation_matrix",
        "observation_covariance",
    ):
        assert getattr(model, name).dtype == tf.float32


def test_linear_gaussian_derivative_containers_preserve_explicit_float32_dtype() -> None:
    first = TFLinearGaussianStateSpaceFirstDerivatives(
        d_initial_mean=tf.zeros([2, 1], dtype=tf.float32),
        d_initial_covariance=tf.zeros([2, 1, 1], dtype=tf.float32),
        d_transition_offset=tf.zeros([2, 1], dtype=tf.float32),
        d_transition_matrix=tf.zeros([2, 1, 1], dtype=tf.float32),
        d_transition_covariance=tf.zeros([2, 1, 1], dtype=tf.float32),
        d_observation_offset=tf.zeros([2, 1], dtype=tf.float32),
        d_observation_matrix=tf.zeros([2, 1, 1], dtype=tf.float32),
        d_observation_covariance=tf.zeros([2, 1, 1], dtype=tf.float32),
    )

    full = first.to_full_derivatives()

    for container in (first, full):
        for name in container.__dataclass_fields__:
            assert getattr(container, name).dtype == tf.float32


def test_linear_gaussian_full_derivatives_reject_mixed_float_dtypes() -> None:
    with pytest.raises(TypeError, match="must share one floating dtype"):
        TFLinearGaussianStateSpaceDerivatives(
            d_initial_mean=tf.zeros([1, 1], dtype=tf.float32),
            d_initial_covariance=tf.zeros([1, 1, 1], dtype=tf.float64),
            d_transition_offset=tf.zeros([1, 1], dtype=tf.float32),
            d_transition_matrix=tf.zeros([1, 1, 1], dtype=tf.float32),
            d_transition_covariance=tf.zeros([1, 1, 1], dtype=tf.float32),
            d_observation_offset=tf.zeros([1, 1], dtype=tf.float32),
            d_observation_matrix=tf.zeros([1, 1, 1], dtype=tf.float32),
            d_observation_covariance=tf.zeros([1, 1, 1], dtype=tf.float32),
            d2_initial_mean=tf.zeros([1, 1, 1], dtype=tf.float32),
            d2_initial_covariance=tf.zeros([1, 1, 1, 1], dtype=tf.float32),
            d2_transition_offset=tf.zeros([1, 1, 1], dtype=tf.float32),
            d2_transition_matrix=tf.zeros([1, 1, 1, 1], dtype=tf.float32),
            d2_transition_covariance=tf.zeros([1, 1, 1, 1], dtype=tf.float32),
            d2_observation_offset=tf.zeros([1, 1, 1], dtype=tf.float32),
            d2_observation_matrix=tf.zeros([1, 1, 1, 1], dtype=tf.float32),
            d2_observation_covariance=tf.zeros([1, 1, 1, 1], dtype=tf.float32),
        )


def test_filter_derivative_result_preserves_explicit_float32_dtype() -> None:
    metadata = FilterRunMetadata(
        filter_name="test_score",
        partition=None,
        integration_space="full_state",
        deterministic_completion="none",
        approximation_label=None,
        differentiability_status="analytic_score",
        compiled_status="tf_function",
    )
    result = TFFilterDerivativeResult(
        log_likelihood=tf.constant(1.0, dtype=tf.float32),
        score=tf.zeros([2], dtype=tf.float32),
        hessian=tf.zeros([2, 2], dtype=tf.float32),
        metadata=metadata,
    )

    assert result.log_likelihood.dtype == tf.float32
    assert result.score.dtype == tf.float32
    assert result.hessian is not None and result.hessian.dtype == tf.float32
