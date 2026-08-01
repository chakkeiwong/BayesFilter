from __future__ import annotations

import ast
import inspect
import os
import textwrap

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import tensorflow as tf
import tensorflow_probability as tfp

from bayesfilter.inference.neutra_batching import bind_batch_native_neutra_target
from bayesfilter.ssm import stable_ssm_target_signature
from bayesfilter.testing.predator_prey_sgqf_neutra_target_tf import (
    make_predator_prey_sgqf_neutra_adapter,
    pp_sgqf_likelihood_value_only_status,
    pp_sgqf_posterior_value_only,
    pp_sgqf_likelihood_value_score_status,
)
from bayesfilter.testing.predator_prey_ukf_neutra_target_tf import (
    PP_PARAMETER_LOWER,
    PP_PARAMETER_UPPER,
    PP_TRUTH_PHYSICAL,
    generate_frozen_predator_prey_dataset_tf,
)


_NORMAL = tfp.distributions.Normal(
    loc=tf.constant(0.0, tf.float64), scale=tf.constant(1.0, tf.float64)
)


def _theta() -> tf.Tensor:
    probability = (PP_TRUTH_PHYSICAL - PP_PARAMETER_LOWER) / (
        PP_PARAMETER_UPPER - PP_PARAMETER_LOWER
    )
    truth = _NORMAL.quantile(probability)
    return tf.stack((truth, tf.zeros([6], tf.float64)), axis=0)


def test_level2_score_matches_centered_fd_and_status() -> None:
    _states, observations = generate_frozen_predator_prey_dataset_tf()
    adapter = make_predator_prey_sgqf_neutra_adapter(
        sparse_level=2, observations=observations
    )
    theta = _theta()
    value, score, status = adapter.neutra_batch_log_prob_and_grad_status(theta)
    epsilon = tf.constant(1.0e-5, tf.float64)
    columns = []
    for coordinate in range(6):
        basis = tf.one_hot(coordinate, 6, dtype=tf.float64)[None, :]
        columns.append(
            (adapter.log_prob(theta + epsilon * basis) - adapter.log_prob(theta - epsilon * basis))
            / (2.0 * epsilon)
        )
    finite_difference = tf.stack(columns, axis=1)
    assert bool(tf.reduce_all(tf.math.is_finite(value)).numpy()) is True
    tf.debugging.assert_near(score, finite_difference, atol=2.0e-5, rtol=2.0e-5)
    assert bool(tf.reduce_all(status["valid_pre_regularized_score"]).numpy()) is True


def test_level3_batch_permutation_binding_and_cpu_xla() -> None:
    _states, observations = generate_frozen_predator_prey_dataset_tf()
    adapter = make_predator_prey_sgqf_neutra_adapter(
        sparse_level=3, observations=observations
    )
    theta = _theta()

    @tf.function(
        input_signature=[tf.TensorSpec([2, 6], tf.float64)], jit_compile=True
    )
    def compiled(values):
        return adapter.neutra_batch_log_prob_and_grad_status(values)

    value, score, status = compiled(theta)
    reversed_value, reversed_score, _reversed_status = compiled(
        tf.reverse(theta, axis=(0,))
    )
    tf.debugging.assert_near(value, tf.reverse(reversed_value, axis=(0,)))
    tf.debugging.assert_near(score, tf.reverse(reversed_score, axis=(0,)))
    assert bool(tf.reduce_all(status["valid_pre_regularized_score"]).numpy()) is True
    signature = stable_ssm_target_signature(adapter.contract)
    binding = bind_batch_native_neutra_target(adapter, target_signature=signature)
    assert binding.target_signature == signature
    assert binding.sample_axis_python_loop_used is False


def test_level2_value_only_endpoint_matches_complete_posterior_under_cpu_xla() -> None:
    _states, observations = generate_frozen_predator_prey_dataset_tf()
    adapter = make_predator_prey_sgqf_neutra_adapter(
        sparse_level=2, observations=observations
    )

    @tf.function(input_signature=[tf.TensorSpec([2, 6], tf.float64)], jit_compile=True)
    def compiled(theta):
        value, status = pp_sgqf_likelihood_value_only_status(
            theta,
            observations=adapter.observations,
            nodes=adapter.nodes,
            weights=adapter.weights,
        )
        return value, status, pp_sgqf_posterior_value_only(
            theta,
            observations=adapter.observations,
            nodes=adapter.nodes,
            weights=adapter.weights,
        )

    likelihood, status, posterior = compiled(_theta())
    reference, _score, reference_status = (
        adapter.neutra_batch_log_prob_and_grad_status(_theta())
    )
    tf.debugging.assert_near(posterior, reference, atol=1.0e-9, rtol=1.0e-11)
    tf.debugging.assert_equal(status["status_code"], reference_status["status_code"])
    assert bool(tf.reduce_all(tf.math.is_finite(likelihood)).numpy()) is True


def test_active_recurrence_has_no_python_axis_loop_or_callback() -> None:
    for function in (
        pp_sgqf_likelihood_value_score_status,
        pp_sgqf_likelihood_value_only_status,
    ):
        source = textwrap.dedent(inspect.getsource(function))
        tree = ast.parse(source)
        assert not any(isinstance(node, (ast.For, ast.AsyncFor)) for node in ast.walk(tree))
        assert ".numpy(" not in source
        assert "numpy" not in source
        forbidden = {"map_fn", "vectorized_map", "numpy_function", "py_function"}
        called = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        assert forbidden.isdisjoint(called)
