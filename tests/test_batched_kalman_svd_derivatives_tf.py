from __future__ import annotations

import ast
import inspect
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import tensorflow as tf

from bayesfilter.linear import batched_kalman_svd_derivatives_tf as kernel
from bayesfilter.linear.kalman_svd_derivatives_tf import (
    tf_svd_linear_gaussian_score_first_order_graph_status,
)
from bayesfilter.linear.types_tf import (
    TFLinearGaussianStateSpace,
    TFLinearGaussianStateSpaceFirstDerivatives,
)
from bayesfilter.testing.multidim_triangular_lgssm_batched_tf import (
    materialize_lower_triangular_lgssm_batch,
)
from bayesfilter.testing.multidim_triangular_lgssm_tf import (
    load_lower_triangular_lgssm_contract,
    lower_triangular_lgssm_observations_from_fixture,
    materialize_lower_triangular_lgssm_with_first_derivatives,
    raw_truth_from_contract,
)


def _inputs():
    contract = load_lower_triangular_lgssm_contract()
    truth = raw_truth_from_contract(contract)
    raw = truth[tf.newaxis, :] + tf.constant(
        [
            [0.0] * 18,
            [0.01, -0.01, 0.02, -0.02, 0.01, 0.0, -0.01, 0.01, -0.02, 0.02, 0.01, -0.01, 0.02, -0.02, 0.01, -0.01, 0.02, -0.02],
            [-0.02, 0.01, -0.01, 0.02, 0.0, -0.01, 0.02, -0.02, 0.01, 0.01, -0.02, 0.02, -0.01, 0.01, -0.02, 0.02, -0.01, 0.01],
        ],
        tf.float64,
    )
    materialized = materialize_lower_triangular_lgssm_batch(raw, contract)
    observations = lower_triangular_lgssm_observations_from_fixture()
    kwargs = {
        "transition_offset": materialized.transition_offset,
        "transition_matrix": materialized.transition_matrix,
        "transition_covariance": materialized.transition_covariance,
        "observation_offset": materialized.observation_offset,
        "observation_matrix": materialized.observation_matrix,
        "observation_covariance": materialized.observation_covariance,
        "initial_state_mean": materialized.initial_mean,
        "initial_state_covariance": materialized.initial_covariance,
        "d_initial_state_mean": materialized.d_initial_mean,
        "d_initial_state_covariance": materialized.d_initial_covariance,
        "d_transition_offset": materialized.d_transition_offset,
        "d_transition_matrix": materialized.d_transition_matrix,
        "d_transition_covariance": materialized.d_transition_covariance,
        "d_observation_offset": materialized.d_observation_offset,
        "d_observation_matrix": materialized.d_observation_matrix,
        "d_observation_covariance": materialized.d_observation_covariance,
    }
    return contract, raw, observations, kwargs


def _scalar_result(raw, contract, observations, *, singular_floor=1.0e-12):
    materialized = materialize_lower_triangular_lgssm_with_first_derivatives(
        raw, contract
    )
    assert materialized.derivatives is not None
    return tf_svd_linear_gaussian_score_first_order_graph_status(
        observations,
        materialized.model,
        materialized.derivatives,
        jitter=tf.constant(1.0e-9, tf.float64),
        singular_floor=tf.constant(singular_floor, tf.float64),
    )


def _status(result):
    return result.diagnostics.extra


def test_regular_batch_matches_scalar_value_score_and_status() -> None:
    contract, raw, observations, kwargs = _inputs()
    eager_actual = (
        kernel.tf_batched_svd_linear_gaussian_score_first_order_graph_status.python_function(
            observations,
            **kwargs,
            jitter=tf.constant(1.0e-9, tf.float64),
            singular_floor=tf.constant(1.0e-12, tf.float64),
        )
    )
    expected = tuple(
        _scalar_result(raw[index], contract, observations) for index in range(3)
    )
    np.testing.assert_allclose(
        eager_actual.log_likelihood.numpy(),
        [item.log_likelihood.numpy() for item in expected],
        rtol=2e-13,
        atol=2e-13,
    )
    np.testing.assert_allclose(
        eager_actual.score.numpy(),
        tf.stack([item.score for item in expected]).numpy(),
        rtol=2e-12,
        atol=2e-12,
    )

    @tf.function(input_signature=[tf.TensorSpec((18,), tf.float64)], jit_compile=True)
    def scalar_compiled(theta):
        result = _scalar_result(theta, contract, observations)
        status = _status(result)
        return (
            result.log_likelihood,
            result.score,
            status["status_code"],
            status["valid_pre_regularized_score"],
            result.diagnostics.regularization.floor_count,
            status["min_innovation_eigenvalue"],
            status["innovation_condition_estimate"],
        )

    actual = kernel.tf_batched_svd_linear_gaussian_score_first_order_graph_status(
        observations,
        **kwargs,
        jitter=tf.constant(1.0e-9, tf.float64),
        singular_floor=tf.constant(1.0e-12, tf.float64),
    )
    expected_xla = tuple(scalar_compiled(raw[index]) for index in range(3))
    np.testing.assert_allclose(
        actual.log_likelihood.numpy(),
        [item[0].numpy() for item in expected_xla],
        rtol=2e-13,
        atol=2e-13,
    )
    np.testing.assert_allclose(
        actual.score.numpy(),
        tf.stack([item[1] for item in expected_xla]).numpy(),
        rtol=2e-12,
        atol=2e-12,
    )
    np.testing.assert_array_equal(
        actual.status_code.numpy(),
        [int(item[2].numpy()) for item in expected_xla],
    )
    np.testing.assert_array_equal(
        actual.valid_pre_regularized_score.numpy(),
        [bool(item[3].numpy()) for item in expected_xla],
    )
    np.testing.assert_array_equal(
        actual.floor_count_value.numpy(),
        [int(item[4].numpy()) for item in expected_xla],
    )
    np.testing.assert_allclose(
        actual.min_innovation_eigenvalue.numpy(),
        [float(item[5].numpy()) for item in expected_xla],
        rtol=2e-12,
        atol=2e-12,
    )
    np.testing.assert_allclose(
        actual.innovation_condition_estimate.numpy(),
        [float(item[6].numpy()) for item in expected_xla],
        rtol=2e-10,
        atol=2e-10,
    )


def test_active_floor_status_matches_scalar_rows() -> None:
    contract, raw, observations, kwargs = _inputs()
    floor = 1.0e3
    actual = kernel.tf_batched_svd_linear_gaussian_score_first_order_graph_status(
        observations,
        **kwargs,
        jitter=tf.constant(1.0e-9, tf.float64),
        singular_floor=tf.constant(floor, tf.float64),
    )
    expected = tuple(
        _scalar_result(raw[index], contract, observations, singular_floor=floor)
        for index in range(3)
    )
    np.testing.assert_array_equal(
        actual.status_code.numpy(),
        [int(_status(item)["status_code"].numpy()) for item in expected],
    )
    np.testing.assert_array_equal(actual.valid_pre_regularized_score.numpy(), [False] * 3)
    np.testing.assert_array_equal(
        actual.floor_count_value.numpy(),
        [int(item.diagnostics.regularization.floor_count.numpy()) for item in expected],
    )


def test_invalid_row_is_isolated_from_valid_rows() -> None:
    _contract, _raw, observations, kwargs = _inputs()
    invalid_covariance = tf.tensor_scatter_nd_update(
        kwargs["observation_covariance"],
        tf.constant([[1, 0, 0]], tf.int32),
        tf.constant([float("nan")], tf.float64),
    )
    mixed = kernel.tf_batched_svd_linear_gaussian_score_first_order_graph_status(
        observations,
        **{**kwargs, "observation_covariance": invalid_covariance},
        jitter=tf.constant(1.0e-9, tf.float64),
        singular_floor=tf.constant(1.0e-12, tf.float64),
    )
    regular = kernel.tf_batched_svd_linear_gaussian_score_first_order_graph_status(
        observations,
        **kwargs,
        jitter=tf.constant(1.0e-9, tf.float64),
        singular_floor=tf.constant(1.0e-12, tf.float64),
    )
    assert mixed.status_code.numpy().tolist() == [0, 2, 0]
    assert mixed.valid_pre_regularized_score.numpy().tolist() == [True, False, True]
    np.testing.assert_allclose(
        tf.gather(mixed.log_likelihood, [0, 2]).numpy(),
        tf.gather(regular.log_likelihood, [0, 2]).numpy(),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        tf.gather(mixed.score, [0, 2]).numpy(),
        tf.gather(regular.score, [0, 2]).numpy(),
        rtol=0.0,
        atol=0.0,
    )


def test_row_permutation_equivariance() -> None:
    _contract, _raw, observations, kwargs = _inputs()
    permutation = tf.constant([2, 0, 1], tf.int32)
    permuted_kwargs = {
        name: tf.gather(value, permutation) for name, value in kwargs.items()
    }
    direct = kernel.tf_batched_svd_linear_gaussian_score_first_order_graph_status(
        observations, **kwargs, jitter=1.0e-9, singular_floor=1.0e-12
    )
    permuted = kernel.tf_batched_svd_linear_gaussian_score_first_order_graph_status(
        observations, **permuted_kwargs, jitter=1.0e-9, singular_floor=1.0e-12
    )
    np.testing.assert_allclose(permuted.log_likelihood.numpy(), tf.gather(direct.log_likelihood, permutation).numpy(), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(permuted.score.numpy(), tf.gather(direct.score, permutation).numpy(), rtol=0.0, atol=0.0)
    np.testing.assert_array_equal(permuted.status_code.numpy(), tf.gather(direct.status_code, permutation).numpy())


def test_compiled_graph_has_one_time_loop_and_no_mapping_or_callback() -> None:
    _contract, _raw, observations, kwargs = _inputs()
    concrete = (
        kernel.tf_batched_svd_linear_gaussian_score_first_order_graph_status.get_concrete_function(
            observations,
            **kwargs,
            jitter=tf.constant(1.0e-9, tf.float64),
            singular_floor=tf.constant(1.0e-12, tf.float64),
        )
    )
    operations = tuple(operation.type for operation in concrete.graph.get_operations())
    assert sum("While" in operation for operation in operations) == 1
    assert not any("Map" in operation for operation in operations)
    assert not any("PyFunc" in operation for operation in operations)


def test_algorithmic_source_has_no_mapping_python_loop_numpy_or_callback() -> None:
    source = inspect.getsource(kernel)
    tree = ast.parse(source)
    assert "numpy" not in source.lower()
    assert "tf.map_fn" not in source
    assert "tf.vectorized_map" not in source
    assert "tf.py_function" not in source
    assert "tf.numpy_function" not in source
    assert not any(isinstance(node, (ast.For, ast.AsyncFor, ast.While)) for node in ast.walk(tree))
