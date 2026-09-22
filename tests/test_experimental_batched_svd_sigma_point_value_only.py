from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import tensorflow as tf

from bayesfilter.nonlinear.experimental_batched_svd_sigma_point_tf import (
    TFBatchedStructuralLinearizations,
    tf_batched_svd_sigma_point_value,
    tf_batched_svd_sigma_point_value_and_output_cotangents,
)
from tests.test_experimental_batched_svd_sigma_point_tf import (
    _lagged_linear_model_and_derivatives,
)


def _lagged_linearizations(derivatives):
    return TFBatchedStructuralLinearizations(
        transition_state_jacobian_fn=derivatives.transition_state_jacobian_fn,
        transition_innovation_jacobian_fn=(
            derivatives.transition_innovation_jacobian_fn
        ),
        observation_state_jacobian_fn=derivatives.observation_state_jacobian_fn,
        lagged_observation_previous_jacobian_fn=(
            derivatives.lagged_observation_previous_jacobian_fn
        ),
        lagged_observation_innovation_jacobian_fn=(
            derivatives.lagged_observation_innovation_jacobian_fn
        ),
        lagged_observation_next_jacobian_fn=(
            derivatives.lagged_observation_next_jacobian_fn
        ),
    )


def test_lagged_value_only_matches_existing_value_and_reports_nonclaims():
    theta = tf.constant([[0.04]], dtype=tf.float64)
    observations, model, derivatives = _lagged_linear_model_and_derivatives(theta)
    value_only = tf_batched_svd_sigma_point_value(
        observations,
        model,
        backend="tf_principal_sqrt_ukf",
    )
    output_cotangents = tf_batched_svd_sigma_point_value_and_output_cotangents(
        observations,
        model,
        _lagged_linearizations(derivatives),
        backend="tf_principal_sqrt_ukf",
    )

    np.testing.assert_array_equal(
        value_only.value.numpy(), output_cotangents.value.numpy()
    )
    assert value_only.diagnostics["observation_contract"].numpy() == (
        b"lagged_previous_innovation_predicted"
    )
    assert value_only.diagnostics["value_only_api"].numpy() == (
        b"tf_batched_svd_sigma_point_value"
    )
    assert bool(
        value_only.diagnostics["reverse_cotangent_pass_constructed"].numpy()
    ) is False
    assert bool(
        value_only.diagnostics["parameter_derivative_hooks_called"].numpy()
    ) is False


def test_lagged_value_only_fixed_graph_has_no_reverse_or_derivative_ops():
    theta = tf.constant([[0.04]], dtype=tf.float64)
    observations, model, _derivatives = _lagged_linear_model_and_derivatives(theta)

    @tf.function(
        input_signature=[tf.TensorSpec([2, 1], tf.float64)],
        autograph=False,
        jit_compile=False,
        reduce_retracing=True,
    )
    def compiled(observations_arg: tf.Tensor) -> tf.Tensor:
        return tf_batched_svd_sigma_point_value(
            observations_arg,
            model,
            backend="tf_principal_sqrt_ukf",
        ).value

    first = compiled(observations)
    second = compiled(observations)
    concrete = compiled.get_concrete_function()
    operations = concrete.graph.get_operations()
    forbidden = (
        "reverse",
        "cotangent",
        "gradienttape",
        "sylvester",
        "d_transition",
        "d_observation",
        "pfor",
        "pyfunc",
        "numpy_function",
        "random",
    )
    hits = [
        operation.name
        for operation in operations
        if any(token in operation.name.lower() for token in forbidden)
    ]

    assert not hits
    assert compiled.experimental_get_tracing_count() == 1
    np.testing.assert_array_equal(first.numpy(), second.numpy())
