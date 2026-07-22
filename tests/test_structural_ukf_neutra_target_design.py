from __future__ import annotations

import ast
import hashlib
import inspect
import os
import textwrap

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import tensorflow as tf

from bayesfilter.testing.structural_ukf_neutra_target_design_tf import (
    STRUCTURAL_FINAL_OBSERVATION_SHA256,
    STRUCTURAL_FINAL_STATE_SHA256,
    STRUCTURAL_PARAMETER_LOWER,
    STRUCTURAL_PARAMETER_UPPER,
    STRUCTURAL_TRUTH_PHYSICAL,
    StructuralUKFLikelihoodRecomposer,
    generate_frozen_structural_dataset_tf,
    make_structural_ukf_neutra_adapter,
    simulate_structural_trajectories_tf,
    structural_likelihood_information_tf,
    structural_negative_control_one_step_tf,
    structural_prior_predictive_tf,
    structural_source_chart,
    structural_source_probit_jacobian_value_score,
    structural_source_uniform_prior_value_score,
    structural_transition_residual,
    structural_transition_value,
    structural_truth_source,
    structural_ukf_likelihood_value_score_status,
    structural_ukf_likelihood_value_only_status,
    structural_ukf_posterior_value_only,
)
from bayesfilter.inference.neutra_batching import bind_batch_native_neutra_target
from bayesfilter.inference.neutra_campaign import (
    admit_independent_posterior_recomposition,
)
from bayesfilter.ssm import stable_ssm_target_signature


def _fixture(horizon: int = 12) -> tuple[tf.Tensor, tf.Tensor]:
    physical = STRUCTURAL_TRUTH_PHYSICAL[None, :]
    states, observations, residuals = simulate_structural_trajectories_tf(
        physical,
        horizon=horizon,
        seed=tf.constant([20260716, 15001], tf.int32),
    )
    tf.debugging.assert_near(residuals, tf.zeros_like(residuals), atol=2e-15)
    return states[0], observations[0]


def test_chart_prior_and_jacobian_recompose_standard_normal() -> None:
    theta = tf.stack(
        [structural_truth_source(), tf.zeros([5], tf.float64)], axis=0
    )
    physical, _derivative = structural_source_chart(theta)
    prior_value, prior_score = structural_source_uniform_prior_value_score(theta)
    jacobian_value, jacobian_score = structural_source_probit_jacobian_value_score(theta)
    expected = tf.reduce_sum(
        -0.5 * tf.square(theta) - 0.5 * tf.math.log(tf.constant(2.0 * 3.141592653589793, tf.float64)),
        axis=1,
    )
    tf.debugging.assert_near(prior_value + jacobian_value, expected)
    tf.debugging.assert_near(prior_score + jacobian_score, -theta)
    tf.debugging.assert_greater(physical, STRUCTURAL_PARAMETER_LOWER)
    tf.debugging.assert_less(physical, STRUCTURAL_PARAMETER_UPPER)


def test_simulator_replays_and_preserves_structural_identity() -> None:
    physical = tf.stack(
        [STRUCTURAL_TRUTH_PHYSICAL, STRUCTURAL_TRUTH_PHYSICAL], axis=0
    )
    first = simulate_structural_trajectories_tf(
        physical, horizon=20, seed=tf.constant([20260716, 15001], tf.int32)
    )
    second = simulate_structural_trajectories_tf(
        physical, horizon=20, seed=tf.constant([20260716, 15001], tf.int32)
    )
    for left, right in zip(first, second, strict=True):
        tf.debugging.assert_equal(left, right)
    tf.debugging.assert_near(first[2], tf.zeros_like(first[2]), atol=2e-15)


def test_frozen_dataset_hashes_and_adapter_contract() -> None:
    states, observations = generate_frozen_structural_dataset_tf()
    assert hashlib.sha256(tf.io.serialize_tensor(states).numpy()).hexdigest() == (
        STRUCTURAL_FINAL_STATE_SHA256
    )
    assert hashlib.sha256(
        tf.io.serialize_tensor(observations).numpy()
    ).hexdigest() == STRUCTURAL_FINAL_OBSERVATION_SHA256
    adapter = make_structural_ukf_neutra_adapter(observations=observations)
    signature = stable_ssm_target_signature(adapter.contract)
    binding = bind_batch_native_neutra_target(adapter, target_signature=signature)
    assert binding.target_signature == signature
    assert binding.sample_axis_python_loop_used is False
    assert adapter.contract.problem.static_shape.innovation_dim == 1
    assert (
        adapter.contract.filter_program.filter_manifest["artificial_k_noise_allowed"]
        is False
    )


def test_transition_residual_and_negative_control_match_chapter() -> None:
    theta = structural_truth_source()[None, :]
    previous = tf.constant([[[0.1, -0.2], [0.0, 0.3]]], tf.float64)
    innovation = tf.constant([[[0.4], [-0.7]]], tf.float64)
    next_state = structural_transition_value(theta, previous, innovation)
    residual = structural_transition_residual(theta, previous, next_state)
    tf.debugging.assert_near(residual, tf.zeros_like(residual), atol=2e-16)
    result = structural_negative_control_one_step_tf()
    tf.debugging.assert_near(
        result["structural_innovation_variance"], tf.constant([0.6121674304], tf.float64), atol=5e-6
    )
    tf.debugging.assert_near(
        result["negative_control_innovation_variance"], tf.constant([0.6521674304], tf.float64), atol=5e-6
    )
    tf.debugging.assert_near(
        result["structural_log_likelihood"], tf.constant([-0.7029747608892933], tf.float64), atol=5e-6
    )
    tf.debugging.assert_near(
        result["negative_control_log_likelihood"], tf.constant([-0.7328186209822024], tf.float64), atol=5e-6
    )
    assert bool(tf.reduce_any(tf.not_equal(result["negative_control_pointwise_residuals"], 0.0)).numpy())


def test_manual_likelihood_score_matches_centered_fd_and_status() -> None:
    _states, observations = _fixture(10)
    truth = structural_truth_source()
    theta = tf.stack([truth, truth + 0.1], axis=0)
    value, score, status = structural_ukf_likelihood_value_score_status(
        theta, observations=observations
    )
    epsilon = tf.constant(1e-5, tf.float64)
    columns = []
    for coordinate in range(5):
        basis = tf.one_hot(coordinate, 5, dtype=tf.float64)[None, :]
        plus = structural_ukf_likelihood_value_score_status(
            theta + epsilon * basis, observations=observations
        )[0]
        minus = structural_ukf_likelihood_value_score_status(
            theta - epsilon * basis, observations=observations
        )[0]
        columns.append((plus - minus) / (2.0 * epsilon))
    finite_difference = tf.stack(columns, axis=1)
    tf.debugging.assert_near(score, finite_difference, atol=2e-5, rtol=2e-5)
    assert bool(tf.reduce_all(status["valid_pre_regularized_score"]).numpy())
    assert bool(tf.reduce_all(tf.math.is_finite(value)).numpy())
    tf.debugging.assert_near(status["deterministic_residual"], tf.zeros([2], tf.float64), atol=2e-14)


def test_independent_posterior_recomposition_and_status_fields() -> None:
    _states, observations = generate_frozen_structural_dataset_tf()
    adapter = make_structural_ukf_neutra_adapter(observations=observations)
    truth = structural_truth_source()
    points = tf.stack([truth, tf.zeros([5], tf.float64), truth + 0.1], axis=0)
    recomposer = StructuralUKFLikelihoodRecomposer(adapter)
    admission = admit_independent_posterior_recomposition(
        adapter=adapter,
        points=points,
        prior_value_score_fn=structural_source_uniform_prior_value_score,
        likelihood_value_score_fn=recomposer.__call__,
        jacobian_value_score_fn=structural_source_probit_jacobian_value_score,
        value_tolerance=1.0e-9,
        score_tolerance=1.0e-8,
    )
    assert admission.passed is True
    _value, _score, status = adapter.neutra_batch_log_prob_and_grad_status(points)
    required = {
        "status_code",
        "valid_pre_regularized_score",
        "floor_count_value",
        "min_innovation_eigenvalue",
        "innovation_condition_estimate",
    }
    assert required.issubset(status)
    assert bool(tf.reduce_all(status["valid_pre_regularized_score"]).numpy())


def test_likelihood_information_is_finite_psd_and_cumulative() -> None:
    _states, observations = _fixture(12)
    truth = structural_truth_source()
    theta = tf.stack([truth, truth + tf.constant([0.1, 0.0, 0.0, 0.0, 0.0], tf.float64)], axis=0)
    result = structural_likelihood_information_tf(theta, observations=observations)
    cumulative = result["cumulative_information"]
    eigenvalues = tf.linalg.eigvalsh(cumulative[:, -1])
    assert bool(tf.reduce_all(tf.math.is_finite(cumulative)).numpy())
    assert bool(tf.reduce_all(eigenvalues >= -1e-8).numpy())
    increments = cumulative[:, 1:] - cumulative[:, :-1]
    assert bool(tf.reduce_all(tf.linalg.eigvalsh(increments) >= -1e-8).numpy())


def test_cpu_xla_value_score_and_prior_predictive() -> None:
    _states, observations = _fixture(8)

    @tf.function(input_signature=[tf.TensorSpec([2, 5], tf.float64)], jit_compile=True)
    def compiled(theta: tf.Tensor):
        return structural_ukf_likelihood_value_score_status(
            theta, observations=observations, principal_sqrt_backend="tensorflow_eigh"
        )

    theta = tf.stack([structural_truth_source(), tf.zeros([5], tf.float64)], axis=0)
    value, score, status = compiled(theta)
    assert value.shape == (2,)
    assert score.shape == (2, 5)
    assert bool(tf.reduce_all(status["valid_pre_regularized_score"]).numpy())

    @tf.function(jit_compile=True)
    def prior_predictive():
        return structural_prior_predictive_tf(
            batch_size=32,
            horizon=20,
            seed=tf.constant([20260716, 15201], tf.int32),
        )

    result = prior_predictive()
    assert result["states"].shape == (32, 20, 2)
    assert result["observations"].shape == (32, 20, 1)


def test_value_only_endpoint_matches_complete_posterior_and_invariant_cpu_xla() -> None:
    _states, observations = generate_frozen_structural_dataset_tf()
    adapter = make_structural_ukf_neutra_adapter(observations=observations)
    theta = tf.stack([structural_truth_source(), tf.zeros([5], tf.float64)], axis=0)

    @tf.function(jit_compile=True)
    def compiled(values):
        likelihood, status = structural_ukf_likelihood_value_only_status(
            values, observations=observations
        )
        posterior = structural_ukf_posterior_value_only(
            values, observations=observations
        )
        return likelihood, status, posterior

    likelihood, status, posterior = compiled(theta)
    reference, _score, reference_status = (
        adapter.neutra_batch_log_prob_and_grad_status(theta)
    )
    tf.debugging.assert_near(posterior, reference, atol=1e-8, rtol=1e-11)
    tf.debugging.assert_equal(status["status_code"], reference_status["status_code"])
    tf.debugging.assert_near(
        status["deterministic_residual"], tf.zeros([2], tf.float64), atol=2e-14
    )
    tf.debugging.assert_equal(
        status["artificial_k_noise_allowed"], tf.zeros([2], tf.bool)
    )
    assert bool(tf.reduce_all(tf.math.is_finite(likelihood)).numpy())


def test_active_kernels_have_no_python_loop_numpy_or_callback() -> None:
    functions = (
        simulate_structural_trajectories_tf,
        structural_transition_value,
        structural_ukf_likelihood_value_score_status,
        structural_ukf_likelihood_value_only_status,
    )
    for function in functions:
        source = textwrap.dedent(inspect.getsource(function))
        tree = ast.parse(source)
        assert not any(isinstance(node, (ast.For, ast.AsyncFor)) for node in ast.walk(tree))
        assert "numpy" not in source.lower()
        assert ".numpy(" not in source
        forbidden = {"map_fn", "vectorized_map", "numpy_function", "py_function"}
        called = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        assert forbidden.isdisjoint(called)
    StructuralUKFLikelihoodRecomposer,
    generate_frozen_structural_dataset_tf,
    make_structural_ukf_neutra_adapter,
