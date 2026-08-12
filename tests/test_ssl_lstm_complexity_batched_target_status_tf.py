from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import tensorflow as tf

from bayesfilter.inference.neutra_batching import (
    batch_native_value_status_target_fn,
    require_batch_native_neutra_target,
)
from bayesfilter.nonlinear import ssl_lstm_complexity_batched_target_tf as target_module
from bayesfilter.nonlinear.ssl_lstm_complexity_batched_target_tf import (
    batch_native_complexity_posterior_target,
)
from bayesfilter.nonlinear.ssl_lstm_complexity_target_tf import PRIOR_CENTER


def test_q20_target_issues_repository_batch_binding_without_fallback() -> None:
    target = batch_native_complexity_posterior_target(20, jit_compile=True)

    binding = require_batch_native_neutra_target(
        target,
        target_signature=target.target_signature(),
        batch_size=100,
    )

    payload = binding.payload()
    assert payload["minimum_batch_size"] == 2
    assert payload["jit_compile_required"] is True
    assert payload["status_telemetry_required"] is True
    assert payload["scalar_fallback_used"] is False
    assert payload["sample_axis_python_loop_used"] is False
    assert payload["row_mapped_scalar_target_used"] is False
    assert payload["adapter_signature"] == target.adapter_signature()


def test_q2_status_route_preserves_batch_value_score_and_custom_gradient() -> None:
    target = batch_native_complexity_posterior_target(
        2,
        jit_compile=False,
        principal_sqrt_backend="tensorflow_eigh",
    )
    theta = tf.stack((PRIOR_CENTER, PRIOR_CENTER + 0.01), axis=0)

    value, score, status = target.neutra_batch_log_prob_and_grad_status(theta)
    expected_value, expected_score = target.batch_value_and_score(theta)

    tf.debugging.assert_equal(value, expected_value)
    tf.debugging.assert_equal(score, expected_score)
    tf.debugging.assert_equal(status["status_code"], tf.zeros([2], tf.int32))
    tf.debugging.assert_equal(
        status["valid_pre_regularized_score"], tf.ones([2], tf.bool)
    )
    tf.debugging.assert_equal(status["floor_count_value"], tf.zeros([2], tf.int32))
    tf.debugging.assert_positive(status["min_innovation_eigenvalue"])

    binding_target = batch_native_complexity_posterior_target(
        2,
        jit_compile=True,
        principal_sqrt_backend="tensorflow_eigh",
    )
    binding = require_batch_native_neutra_target(
        binding_target,
        target_signature=binding_target.target_signature(),
        batch_size=2,
    )
    target_value_status = batch_native_value_status_target_fn(binding)
    with tf.GradientTape() as tape:
        tape.watch(theta)
        bound_value, bound_status = target_value_status(theta)
        objective = tf.reduce_sum(bound_value)
    bound_score = tape.gradient(objective, theta)

    tf.debugging.assert_near(bound_value, value, atol=1.0e-10, rtol=1.0e-10)
    tf.debugging.assert_near(bound_score, score, atol=1.0e-10, rtol=1.0e-10)
    tf.debugging.assert_equal(
        bound_status["valid_pre_regularized_score"], tf.ones([2], tf.bool)
    )
    tf.debugging.assert_equal(
        bound_status["innovation_condition_estimate_available"],
        tf.zeros([2], tf.bool),
    )


def test_training_value_score_fails_closed_when_kernel_status_is_invalid(
    monkeypatch,
) -> None:
    def invalid_kernel(theta, observations, model, derivatives, **kwargs):
        del observations, model, derivatives, kwargs
        batch = tf.shape(theta)[0]
        return (
            tf.zeros([batch], tf.float64),
            tf.zeros_like(theta),
            {
                "placement_floor_count": tf.ones([batch], tf.int32),
                "innovation_floor_count": tf.zeros([batch], tf.int32),
                "principal_sqrt_target_row_class_code": tf.ones([batch], tf.int32),
                "principal_sqrt_target_valid_count": tf.zeros([batch], tf.int32),
                "min_innovation_eigenvalue": tf.zeros([batch], tf.float64),
                "min_placement_eigenvalue": tf.zeros([batch], tf.float64),
                "min_placement_eigen_gap": tf.zeros([batch], tf.float64),
                "min_innovation_eigen_gap": tf.zeros([batch], tf.float64),
                "placement_classified_invalid_count": tf.ones([batch], tf.int32),
                "innovation_classified_invalid_count": tf.zeros([batch], tf.int32),
                "placement_derivative_rhs_nonfinite_count": tf.zeros(
                    [batch], tf.int32
                ),
                "innovation_derivative_rhs_nonfinite_count": tf.zeros(
                    [batch], tf.int32
                ),
                "principal_sqrt_target_classified_invalid_count": tf.ones(
                    [batch], tf.int32
                ),
                "principal_sqrt_target_derivative_rhs_nonfinite_count": tf.zeros(
                    [batch], tf.int32
                ),
                "placement_roundoff_repair_count": tf.zeros([batch], tf.int32),
                "innovation_roundoff_repair_count": tf.zeros([batch], tf.int32),
            },
        )

    monkeypatch.setattr(
        target_module,
        "tf_batched_svd_sigma_point_value_and_score_custom_gradient",
        invalid_kernel,
    )
    target = batch_native_complexity_posterior_target(
        1,
        jit_compile=False,
        principal_sqrt_backend="tensorflow_eigh",
    )
    theta = tf.stack((PRIOR_CENTER, PRIOR_CENTER + 0.01), axis=0)

    raw_value, raw_score, status = target.neutra_batch_log_prob_and_grad_status(theta)
    value, score = target.batch_value_and_score(theta)

    tf.debugging.assert_all_finite(raw_value, "raw classified values are retained")
    tf.debugging.assert_all_finite(raw_score, "raw classified scores are retained")
    tf.debugging.assert_equal(status["status_code"], tf.ones([2], tf.int32))
    assert not bool(tf.reduce_any(status["valid_pre_regularized_score"]).numpy())
    tf.debugging.assert_equal(status["placement_floor_count"], tf.ones([2], tf.int32))
    tf.debugging.assert_equal(status["innovation_floor_count"], tf.zeros([2], tf.int32))
    tf.debugging.assert_equal(
        status["principal_sqrt_target_row_class_code"], tf.ones([2], tf.int32)
    )
    tf.debugging.assert_equal(
        status["principal_sqrt_target_valid_count"], tf.zeros([2], tf.int32)
    )
    assert bool(tf.reduce_all(status["value_finite"]).numpy())
    assert bool(tf.reduce_all(status["score_finite"]).numpy())
    assert bool(tf.reduce_all(status["input_finite"]).numpy())
    tf.debugging.assert_equal(
        status["placement_classified_invalid_count"], tf.ones([2], tf.int32)
    )
    tf.debugging.assert_equal(
        status["innovation_classified_invalid_count"], tf.zeros([2], tf.int32)
    )
    tf.debugging.assert_equal(
        status["principal_sqrt_target_derivative_rhs_nonfinite_count"],
        tf.zeros([2], tf.int32),
    )
    assert not bool(tf.reduce_any(tf.math.is_finite(value)).numpy())
    assert not bool(tf.reduce_any(tf.math.is_finite(score)).numpy())
