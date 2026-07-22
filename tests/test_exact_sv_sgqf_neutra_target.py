from __future__ import annotations

import ast
import hashlib
import inspect
import os
import textwrap

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import tensorflow as tf
import tensorflow_probability as tfp

from docs.benchmarks.run_multimodel_neutra_p2_svx_sgqf_admission import (
    _build_audit_points,
)
from bayesfilter.highdim.sv_mixture_cut4 import (
    exact_transformed_sv_independent_panel_fixed_sgqf_filter,
)
from bayesfilter.inference.neutra_batching import bind_batch_native_neutra_target
from bayesfilter.ssm import stable_ssm_target_signature
from bayesfilter.testing.exact_sv_sgqf_neutra_target_tf import (
    SVX_SGQF_OBSERVATION_SHA256,
    SVX_SGQF_STATE_SHA256,
    _fixed_sgqf_value_score_status,
    _posterior_value_score,
    fixed_sgqf_likelihood_value_score,
    generate_frozen_exact_sv_dataset_tf,
    make_exact_sv_sgqf_neutra_adapter,
    source_chart_physical_parameters,
    source_two_probit_jacobian_value_score,
    source_uniform_prior_value_score,
)


def _theta() -> tf.Tensor:
    return tf.constant(
        [[-1.0, -1.0], [-1.0, 1.0], [0.0, 0.0], [1.0, -1.0], [1.0, 1.0]],
        tf.float64,
    )


def test_admission_audit_points_keep_truth_point_in_float64() -> None:
    audit_points = _build_audit_points(tf, tfp)
    assert audit_points.dtype == tf.float64
    assert audit_points.shape == (6, 2)
    truth_gamma, truth_beta = source_chart_physical_parameters(audit_points[-1:])
    tf.debugging.assert_near(truth_gamma, tf.constant([0.6], tf.float64))
    tf.debugging.assert_near(truth_beta, tf.constant([0.4], tf.float64))


def test_frozen_dataset_graph_replay_matches_preserved_hashes() -> None:
    states, observations = generate_frozen_exact_sv_dataset_tf()
    assert states.shape == (1000, 1)
    assert observations.shape == (1000, 1)
    assert hashlib.sha256(tf.io.serialize_tensor(states).numpy()).hexdigest() == (
        SVX_SGQF_STATE_SHA256
    )
    assert hashlib.sha256(
        tf.io.serialize_tensor(observations).numpy()
    ).hexdigest() == SVX_SGQF_OBSERVATION_SHA256
    assert bool(tf.reduce_any(tf.equal(observations, 0.0)).numpy()) is False


def test_source_chart_prior_plus_jacobian_is_standard_normal() -> None:
    theta = _theta()
    prior_value, prior_score = source_uniform_prior_value_score(theta)
    jacobian_value, jacobian_score = source_two_probit_jacobian_value_score(theta)
    expected_value = tf.reduce_sum(
        -0.5 * tf.square(theta)
        - 0.5 * tf.math.log(tf.constant(2.0 * 3.141592653589793, tf.float64)),
        axis=-1,
    )
    tf.debugging.assert_near(prior_value + jacobian_value, expected_value)
    tf.debugging.assert_near(prior_score + jacobian_score, -theta)
    gamma, beta = source_chart_physical_parameters(theta)
    assert bool(tf.reduce_all(gamma > 0.1).numpy()) is True
    assert bool(tf.reduce_all(gamma < 0.9).numpy()) is True
    assert bool(tf.reduce_all(beta > 0.1).numpy()) is True
    assert bool(tf.reduce_all(beta < 0.9).numpy()) is True


def test_graph_native_likelihood_matches_existing_value_route_on_prefix() -> None:
    _states, observations = generate_frozen_exact_sv_dataset_tf(horizon=8)
    adapter = make_exact_sv_sgqf_neutra_adapter(
        sparse_level=4, observations=observations
    )
    theta = _theta()
    value, _score = fixed_sgqf_likelihood_value_score(
        theta,
        observations=adapter.observations,
        nodes=adapter.nodes,
        weights=adapter.weights,
    )
    gamma, beta = source_chart_physical_parameters(theta)
    references = tuple(
        exact_transformed_sv_independent_panel_fixed_sgqf_filter(
            observations,
            gamma=gamma[index],
            beta=beta[index],
            sigma=1.0,
            sparse_level=4,
        ).log_likelihood
        for index in range(int(theta.shape[0]))
    )
    tf.debugging.assert_near(value, tf.stack(references), atol=1.0e-11, rtol=1.0e-11)


def test_manual_posterior_score_matches_centered_finite_difference() -> None:
    _states, observations = generate_frozen_exact_sv_dataset_tf(horizon=20)
    adapter = make_exact_sv_sgqf_neutra_adapter(
        sparse_level=4, observations=observations
    )
    theta = _theta()
    value, score = adapter.log_prob_and_grad(theta)
    epsilon = tf.constant(1.0e-5, tf.float64)
    columns = []
    for coordinate in range(2):
        basis = tf.one_hot(coordinate, 2, dtype=tf.float64)[None, :]
        plus = adapter.log_prob(theta + epsilon * basis)
        minus = adapter.log_prob(theta - epsilon * basis)
        columns.append((plus - minus) / (2.0 * epsilon))
    finite_difference = tf.stack(columns, axis=1)
    tf.debugging.assert_all_finite(value, "posterior value must be finite")
    tf.debugging.assert_near(score, finite_difference, atol=1.0e-6, rtol=1.0e-6)


def test_batch_permutation_status_and_binding() -> None:
    _states, observations = generate_frozen_exact_sv_dataset_tf(horizon=20)
    adapter = make_exact_sv_sgqf_neutra_adapter(
        sparse_level=4, observations=observations
    )
    theta = _theta()
    value, score, status = adapter.neutra_batch_log_prob_and_grad_status(theta)
    permutation = tf.constant([4, 2, 0, 3, 1], tf.int32)
    permuted = adapter.neutra_batch_log_prob_and_grad_status(
        tf.gather(theta, permutation)
    )
    tf.debugging.assert_near(permuted[0], tf.gather(value, permutation))
    tf.debugging.assert_near(permuted[1], tf.gather(score, permutation))
    assert bool(tf.reduce_all(status["valid_pre_regularized_score"]).numpy()) is True
    assert bool(tf.reduce_all(tf.equal(status["status_code"], 0)).numpy()) is True
    binding = bind_batch_native_neutra_target(
        adapter,
        target_signature=stable_ssm_target_signature(adapter.contract),
    )
    assert binding.sample_axis_python_loop_used is False
    assert binding.scalar_fallback_used is False
    assert binding.row_mapped_scalar_target_used is False


def test_cpu_xla_target_compile() -> None:
    _states, observations = generate_frozen_exact_sv_dataset_tf(horizon=20)
    adapter = make_exact_sv_sgqf_neutra_adapter(
        sparse_level=4, observations=observations
    )

    @tf.function(
        input_signature=[tf.TensorSpec([None, 2], tf.float64)],
        jit_compile=True,
    )
    def compiled(theta):
        return adapter.neutra_batch_log_prob_and_grad_status(theta)

    value, score, status = compiled(_theta())
    tf.debugging.assert_all_finite(value, "XLA value must be finite")
    tf.debugging.assert_all_finite(score, "XLA score must be finite")
    assert bool(tf.reduce_all(status["valid_pre_regularized_score"]).numpy()) is True


def test_active_tensor_functions_have_no_python_loops_or_callbacks() -> None:
    for function in (
        _fixed_sgqf_value_score_status,
        _posterior_value_score,
        fixed_sgqf_likelihood_value_score,
    ):
        source = textwrap.dedent(inspect.getsource(function))
        tree = ast.parse(source)
        for node in ast.walk(tree):
            assert not isinstance(node, (ast.For, ast.AsyncFor, ast.While))
        assert "tf.py_function" not in source
        assert "tf.numpy_function" not in source
        assert "tf.map_fn" not in source
        assert "tf.vectorized_map" not in source
