from __future__ import annotations

import ast
import hashlib
import inspect
import os
import textwrap

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import tensorflow as tf
import tensorflow_probability as tfp

from bayesfilter.inference.neutra_batching import bind_batch_native_neutra_target
from bayesfilter.inference.neutra_campaign import (
    admit_independent_posterior_recomposition,
)
from bayesfilter.ssm import stable_ssm_target_signature
from bayesfilter.testing.predator_prey_ukf_neutra_target_tf import (
    PP_OBSERVATION_SHA256,
    PP_PARAMETER_LOWER,
    PP_PARAMETER_UPPER,
    PP_STATE_SHA256,
    PP_TRUTH_PHYSICAL,
    PredatorPreyUKFLikelihoodRecomposer,
    _initial_observation_update,
    pp_ukf_likelihood_value_score_status,
    pp_ukf_likelihood_value_only_status,
    pp_ukf_posterior_value_only,
    generate_frozen_predator_prey_dataset_tf,
    make_predator_prey_ukf_neutra_adapter,
    rk4_transition_value,
    rk4_transition_value_state_source_jacobians,
    source_chart_physical_parameters,
    source_six_probit_jacobian_value_score,
    source_uniform_prior_value_score,
)


_NORMAL = tfp.distributions.Normal(
    loc=tf.constant(0.0, tf.float64), scale=tf.constant(1.0, tf.float64)
)


def _truth_source() -> tf.Tensor:
    probabilities = (PP_TRUTH_PHYSICAL - PP_PARAMETER_LOWER) / (
        PP_PARAMETER_UPPER - PP_PARAMETER_LOWER
    )
    return _NORMAL.quantile(probabilities)


def _theta() -> tf.Tensor:
    truth = _truth_source()
    return tf.stack(
        (
            truth,
            tf.zeros([6], tf.float64),
            tf.constant([-0.4, -0.2, 0.0, 0.2, 0.4, -0.1], tf.float64),
        ),
        axis=0,
    )


def test_frozen_dataset_hashes_and_negative_state_domain() -> None:
    states, observations = generate_frozen_predator_prey_dataset_tf()
    assert states.shape == (20, 2)
    assert observations.shape == (20, 2)
    assert hashlib.sha256(tf.io.serialize_tensor(states).numpy()).hexdigest() == (
        PP_STATE_SHA256
    )
    assert hashlib.sha256(
        tf.io.serialize_tensor(observations).numpy()
    ).hexdigest() == PP_OBSERVATION_SHA256
    assert bool(tf.reduce_any(states < 0.0).numpy()) is True


def test_six_probit_prior_plus_jacobian_is_standard_normal() -> None:
    theta = _theta()
    prior_value, prior_score = source_uniform_prior_value_score(theta)
    jacobian_value, jacobian_score = source_six_probit_jacobian_value_score(theta)
    expected = tf.reduce_sum(_NORMAL.log_prob(theta), axis=1)
    tf.debugging.assert_near(prior_value + jacobian_value, expected)
    tf.debugging.assert_near(prior_score + jacobian_score, -theta)
    physical, _derivative = source_chart_physical_parameters(theta)
    assert bool(tf.reduce_all(physical > PP_PARAMETER_LOWER).numpy()) is True
    assert bool(tf.reduce_all(physical < PP_PARAMETER_UPPER).numpy()) is True


def test_initial_observation_is_assimilated_without_transition() -> None:
    _states, observations = generate_frozen_predator_prey_dataset_tf()
    mean, covariance, value = _initial_observation_update(observations, 3)
    expected_mean = tf.constant([50.0, 5.0], tf.float64) + 0.2 * (
        observations[0] - tf.constant([50.0, 5.0], tf.float64)
    )
    tf.debugging.assert_near(mean, tf.broadcast_to(expected_mean[None, :], [3, 2]))
    tf.debugging.assert_near(
        covariance, tf.broadcast_to((0.8 * tf.eye(2, dtype=tf.float64))[None, :, :], [3, 2, 2])
    )
    assert bool(tf.reduce_all(tf.math.is_finite(value)).numpy()) is True


def test_rk4_state_and_source_jacobians_match_centered_fd() -> None:
    theta = _theta()[:2]
    points = tf.constant(
        [[[50.0, 5.0], [40.0, 3.0]], [[48.0, 4.0], [35.0, 2.0]]],
        tf.float64,
    )
    value, state_jacobian, source_jacobian = (
        rk4_transition_value_state_source_jacobians(theta, points)
    )
    epsilon = tf.constant(1.0e-5, tf.float64)
    state_columns = []
    for coordinate in range(2):
        basis = tf.one_hot(coordinate, 2, dtype=tf.float64)[None, None, :]
        plus = rk4_transition_value_state_source_jacobians(
            theta, points + epsilon * basis
        )[0]
        minus = rk4_transition_value_state_source_jacobians(
            theta, points - epsilon * basis
        )[0]
        state_columns.append((plus - minus) / (2.0 * epsilon))
    state_fd = tf.stack(state_columns, axis=-1)
    source_columns = []
    for coordinate in range(6):
        basis = tf.one_hot(coordinate, 6, dtype=tf.float64)[None, :]
        plus = rk4_transition_value_state_source_jacobians(
            theta + epsilon * basis, points
        )[0]
        minus = rk4_transition_value_state_source_jacobians(
            theta - epsilon * basis, points
        )[0]
        source_columns.append((plus - minus) / (2.0 * epsilon))
    source_fd = tf.stack(source_columns, axis=1)
    assert bool(tf.reduce_all(tf.math.is_finite(value)).numpy()) is True
    tf.debugging.assert_near(state_jacobian, state_fd, atol=2.0e-7, rtol=2.0e-7)
    tf.debugging.assert_near(source_jacobian, source_fd, atol=2.0e-7, rtol=2.0e-7)
    tf.debugging.assert_near(rk4_transition_value(theta, points), value)


def test_posterior_score_batch_status_and_permutation() -> None:
    _states, observations = generate_frozen_predator_prey_dataset_tf()
    adapter = make_predator_prey_ukf_neutra_adapter(observations=observations)
    theta = _theta()
    value, score, status = adapter.neutra_batch_log_prob_and_grad_status(theta)
    epsilon = tf.constant(1.0e-5, tf.float64)
    fd_columns = []
    for coordinate in range(6):
        basis = tf.one_hot(coordinate, 6, dtype=tf.float64)[None, :]
        fd_columns.append(
            (adapter.log_prob(theta + epsilon * basis) - adapter.log_prob(theta - epsilon * basis))
            / (2.0 * epsilon)
        )
    finite_difference = tf.stack(fd_columns, axis=1)
    tf.debugging.assert_near(score, finite_difference, atol=2.0e-5, rtol=2.0e-5)
    assert bool(tf.reduce_all(status["valid_pre_regularized_score"]).numpy()) is True
    reversed_value, reversed_score, _reversed_status = (
        adapter.neutra_batch_log_prob_and_grad_status(tf.reverse(theta, axis=(0,)))
    )
    tf.debugging.assert_near(value, tf.reverse(reversed_value, axis=(0,)))
    tf.debugging.assert_near(score, tf.reverse(reversed_score, axis=(0,)))
    signature = stable_ssm_target_signature(adapter.contract)
    binding = bind_batch_native_neutra_target(adapter, target_signature=signature)
    assert binding.target_signature == signature
    assert binding.sample_axis_python_loop_used is False


def test_cpu_xla_value_score_status() -> None:
    _states, observations = generate_frozen_predator_prey_dataset_tf()
    adapter = make_predator_prey_ukf_neutra_adapter(observations=observations)

    @tf.function(
        input_signature=[tf.TensorSpec([3, 6], tf.float64)], jit_compile=True
    )
    def compiled(theta):
        return pp_ukf_likelihood_value_score_status(
            theta,
            observations=adapter.observations,
            principal_sqrt_backend="tensorflow_eigh",
        )

    value, score, status = compiled(_theta())
    assert value.shape == (3,)
    assert score.shape == (3, 6)
    assert bool(tf.reduce_all(status["valid_pre_regularized_score"]).numpy()) is True
    assert bool(
        tf.reduce_all(tf.equal(status["principal_sqrt_backend_code"], 1)).numpy()
    ) is True


def test_value_only_endpoint_matches_complete_posterior_under_cpu_xla() -> None:
    _states, observations = generate_frozen_predator_prey_dataset_tf()
    adapter = make_predator_prey_ukf_neutra_adapter(observations=observations)

    @tf.function(input_signature=[tf.TensorSpec([3, 6], tf.float64)], jit_compile=True)
    def compiled(theta):
        value, status = pp_ukf_likelihood_value_only_status(
            theta, observations=observations
        )
        return value, status, pp_ukf_posterior_value_only(
            theta, observations=observations
        )

    likelihood, status, posterior = compiled(_theta())
    reference, _score, reference_status = (
        adapter.neutra_batch_log_prob_and_grad_status(_theta())
    )
    tf.debugging.assert_near(posterior, reference, atol=1.0e-9, rtol=1.0e-11)
    tf.debugging.assert_equal(status["status_code"], reference_status["status_code"])
    assert bool(tf.reduce_all(tf.math.is_finite(likelihood)).numpy()) is True


def test_target_default_tensorflow_eigh_matches_compiled_custom_op() -> None:
    _states, observations = generate_frozen_predator_prey_dataset_tf()
    theta = _theta()
    native_value, native_score, native_status = pp_ukf_likelihood_value_score_status(
        theta, observations=observations
    )
    compiled_value, compiled_score, compiled_status = (
        pp_ukf_likelihood_value_score_status(
            theta,
            observations=observations,
            principal_sqrt_backend="compiled_custom_op",
        )
    )
    tf.debugging.assert_near(native_value, compiled_value, atol=1.0e-10, rtol=1.0e-10)
    tf.debugging.assert_near(native_score, compiled_score, atol=1.0e-9, rtol=1.0e-9)
    tf.debugging.assert_equal(native_status["status_code"], compiled_status["status_code"])


def test_independent_recomposition_accepts_bound_likelihood_method() -> None:
    _states, observations = generate_frozen_predator_prey_dataset_tf()
    adapter = make_predator_prey_ukf_neutra_adapter(observations=observations)
    recomposer = PredatorPreyUKFLikelihoodRecomposer(adapter)
    admitted = admit_independent_posterior_recomposition(
        adapter=adapter,
        points=_theta(),
        prior_value_score_fn=source_uniform_prior_value_score,
        likelihood_value_score_fn=recomposer.__call__,
        jacobian_value_score_fn=source_six_probit_jacobian_value_score,
        value_tolerance=1.0e-9,
        score_tolerance=1.0e-8,
    )
    assert admitted.passed is True


def test_active_adapter_has_no_python_axis_loop_or_callback() -> None:
    sources = (
        inspect.getsource(pp_ukf_likelihood_value_score_status),
        inspect.getsource(rk4_transition_value_state_source_jacobians),
        inspect.getsource(pp_ukf_likelihood_value_only_status),
    )
    for raw_source in sources:
        source = textwrap.dedent(raw_source)
        tree = ast.parse(source)
        assert not any(
            isinstance(node, (ast.For, ast.AsyncFor)) for node in ast.walk(tree)
        )
        assert ".numpy(" not in source
        assert "numpy" not in source
        forbidden = {"map_fn", "vectorized_map", "numpy_function", "py_function"}
        called = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        assert forbidden.isdisjoint(called)
