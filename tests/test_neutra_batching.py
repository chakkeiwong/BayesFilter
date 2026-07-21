from __future__ import annotations

import os
from dataclasses import replace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import pytest
import tensorflow as tf

from bayesfilter.inference.neutra_batching import (
    InvalidNeuTraBatchTarget,
    batch_native_value_status_target_fn,
    bind_batch_native_neutra_target,
    require_batch_native_neutra_target,
)
from bayesfilter.inference.posterior_adapter import ValueScoreCapability


TARGET_SIGNATURE = "b" * 64


class VectorizedAdapter:
    def __init__(self) -> None:
        self.call_count = 0

    def value_score_capability(self):
        return ValueScoreCapability(
            value_score_authority="graph_native",
            xla_hmc_ready=True,
            runtime_backend="test_vectorized_batch_native_gaussian",
            evidence_path="tests/test_neutra_batching.py",
            target_scope="test_vectorized_batch_native_gaussian",
        )

    def neutra_batch_log_prob_and_grad_status(self, theta):
        values = tf.convert_to_tensor(theta, tf.float64)
        self.call_count += 1
        leading = tf.shape(values)[:-1]
        return -0.5 * tf.reduce_sum(tf.square(values), axis=-1), -values, {
            "status_code": tf.zeros(leading, tf.int32),
            "valid_pre_regularized_score": tf.ones(leading, tf.bool),
            "floor_count_value": tf.zeros(leading, tf.int32),
            "min_innovation_eigenvalue": tf.ones(leading, tf.float64),
            "innovation_condition_estimate": tf.ones(leading, tf.float64),
        }


class MissingBindingAdapter(VectorizedAdapter):
    neutra_batch_log_prob_and_grad_status = None


class NoConditionEstimateAdapter(VectorizedAdapter):
    def neutra_batch_log_prob_and_grad_status(self, theta):
        values = tf.convert_to_tensor(theta, tf.float64)
        leading = tf.shape(values)[:-1]
        return -0.5 * tf.reduce_sum(tf.square(values), axis=-1), -values, {
            "status_code": tf.zeros(leading, tf.int32),
            "valid_pre_regularized_score": tf.ones(leading, tf.bool),
            "floor_count_value": tf.zeros(leading, tf.int32),
            "min_innovation_eigenvalue": tf.ones(leading, tf.float64),
        }


class RowMappedAdapter(VectorizedAdapter):
    def neutra_batch_log_prob_and_grad_status(self, theta):
        values = tf.convert_to_tensor(theta, tf.float64)
        rows = tf.map_fn(
            lambda row: -0.5 * tf.reduce_sum(tf.square(row)),
            values,
            fn_output_signature=tf.float64,
        )
        leading = tf.shape(values)[:-1]
        return rows, -values, {
            "status_code": tf.zeros(leading, tf.int32),
            "valid_pre_regularized_score": tf.ones(leading, tf.bool),
            "floor_count_value": tf.zeros(leading, tf.int32),
            "min_innovation_eigenvalue": tf.ones(leading, tf.float64),
            "innovation_condition_estimate": tf.ones(leading, tf.float64),
        }


class DelegatingAdapter(VectorizedAdapter):
    def scalar_target(self, theta):
        return -0.5 * tf.reduce_sum(tf.square(theta), axis=-1), -theta

    def neutra_batch_log_prob_and_grad_status(self, theta):
        values = tf.convert_to_tensor(theta, tf.float64)
        value, score = self.scalar_target(values)
        leading = tf.shape(values)[:-1]
        return value, score, {
            "status_code": tf.zeros(leading, tf.int32),
            "valid_pre_regularized_score": tf.ones(leading, tf.bool),
            "floor_count_value": tf.zeros(leading, tf.int32),
            "min_innovation_eigenvalue": tf.ones(leading, tf.float64),
            "innovation_condition_estimate": tf.ones(leading, tf.float64),
        }


class NonXLAAdapter(VectorizedAdapter):
    def value_score_capability(self):
        return ValueScoreCapability(
            value_score_authority="graph_native",
            xla_hmc_ready=False,
            runtime_backend="test_non_xla",
            target_scope="test_non_xla",
        )


def test_repository_binding_invokes_vectorized_batch_method() -> None:
    adapter = VectorizedAdapter()
    binding = require_batch_native_neutra_target(
        adapter,
        target_signature=TARGET_SIGNATURE,
        batch_size=8,
    )
    target = batch_native_value_status_target_fn(binding)
    values = tf.ones((8, 3), tf.float64)

    with tf.GradientTape() as tape:
        tape.watch(values)
        result, status = target(values)
        objective = tf.reduce_sum(result)
    score = tape.gradient(objective, values)

    assert adapter.call_count == 1
    assert result.shape == (8,)
    assert score is not None and score.shape == values.shape
    assert bool(tf.reduce_all(status["valid_pre_regularized_score"]).numpy())
    assert binding.payload()["scalar_fallback_used"] is False
    assert binding.payload()["row_mapped_scalar_target_used"] is False


def test_optional_condition_estimate_is_normalized_with_availability() -> None:
    binding = require_batch_native_neutra_target(
        NoConditionEstimateAdapter(),
        target_signature=TARGET_SIGNATURE,
        batch_size=8,
    )
    target = batch_native_value_status_target_fn(binding)

    _value, status = target(tf.ones((8, 3), tf.float64))

    tf.debugging.assert_equal(
        status["innovation_condition_estimate"], tf.ones([8], tf.float64)
    )
    assert not bool(
        tf.reduce_any(status["innovation_condition_estimate_available"]).numpy()
    )


@pytest.mark.parametrize(
    "adapter,match",
    (
        (MissingBindingAdapter(), "requires bound method"),
        (RowMappedAdapter(), "row-mapped or callback-backed"),
        (DelegatingAdapter(), "row-mapped or callback-backed"),
        (NonXLAAdapter(), "must be XLA ready"),
    ),
)
def test_binding_rejects_ineligible_adapter(adapter, match: str) -> None:
    with pytest.raises(InvalidNeuTraBatchTarget, match=match):
        bind_batch_native_neutra_target(
            adapter,
            target_signature=TARGET_SIGNATURE,
        )


def test_binding_rejects_singleton_training_batch() -> None:
    with pytest.raises(InvalidNeuTraBatchTarget, match="at least 2"):
        require_batch_native_neutra_target(
            VectorizedAdapter(),
            target_signature=TARGET_SIGNATURE,
            batch_size=1,
        )


def test_forged_binding_is_rejected() -> None:
    binding = bind_batch_native_neutra_target(
        VectorizedAdapter(),
        target_signature=TARGET_SIGNATURE,
    )
    forged = replace(binding, _issuer=object())
    with pytest.raises(InvalidNeuTraBatchTarget, match="repository-issued"):
        batch_native_value_status_target_fn(forged)


def test_forged_dependency_closure_is_rejected() -> None:
    binding = bind_batch_native_neutra_target(
        VectorizedAdapter(),
        target_signature=TARGET_SIGNATURE,
    )
    forged = replace(binding, dependency_closure_sha256="0" * 64)
    with pytest.raises(InvalidNeuTraBatchTarget, match="dependency closure"):
        batch_native_value_status_target_fn(forged)


def test_detached_callable_is_rejected() -> None:
    adapter = VectorizedAdapter()
    detached = adapter.neutra_batch_log_prob_and_grad_status.__func__
    adapter.neutra_batch_log_prob_and_grad_status = detached
    with pytest.raises(InvalidNeuTraBatchTarget, match="instance method"):
        bind_batch_native_neutra_target(
            adapter,
            target_signature=TARGET_SIGNATURE,
        )
