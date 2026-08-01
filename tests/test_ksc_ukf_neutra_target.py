from __future__ import annotations

import ast
import hashlib
import inspect
import os
import textwrap

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import tensorflow as tf
import tensorflow_probability as tfp

from docs.benchmarks.run_multimodel_neutra_p3_ksc_ukf_admission import (
    build_audit_points,
)
from bayesfilter.highdim.sv_mixture_cut4 import (
    independent_panel_sv_mixture_ukf_filter,
    independent_panel_sv_mixture_ukf_score,
)
from bayesfilter.inference.neutra_batching import bind_batch_native_neutra_target
from bayesfilter.ssm import stable_ssm_target_signature
from bayesfilter.testing.exact_sv_sgqf_neutra_target_tf import (
    generate_frozen_exact_sv_dataset_tf,
    source_chart_physical_parameters,
)
from bayesfilter.testing.ksc_ukf_neutra_target_tf import (
    KSC_UKF_RAW_OBSERVATION_SHA256,
    KSC_UKF_STATE_SHA256,
    KSC_UKF_TRANSFORM_OFFSET,
    _ksc_ukf_likelihood_value_score_status,
    ksc_ukf_likelihood_value_score,
    make_ksc_ukf_neutra_adapter,
    transformed_ksc_observations,
)


_NORMAL = tfp.distributions.Normal(
    loc=tf.constant(0.0, tf.float64), scale=tf.constant(1.0, tf.float64)
)


def _theta() -> tf.Tensor:
    return tf.constant(
        [[-1.0, -1.0], [-1.0, 1.0], [0.0, 0.0], [1.0, -1.0], [1.0, 1.0]],
        tf.float64,
    )


def test_p3_admission_audit_points_are_frozen_float64() -> None:
    audit_points = build_audit_points(tf, tfp)
    assert audit_points.dtype == tf.float64
    assert audit_points.shape == (6, 2)
    truth_gamma, truth_beta = source_chart_physical_parameters(audit_points[-1:])
    tf.debugging.assert_near(truth_gamma, tf.constant([0.6], tf.float64))
    tf.debugging.assert_near(truth_beta, tf.constant([0.4], tf.float64))


def _legacy_theta_and_score_scale(theta: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    gamma, beta = source_chart_physical_parameters(theta)
    legacy_theta = tf.stack((_NORMAL.quantile(gamma), tf.math.log(beta)), axis=1)
    legacy_gamma_density = _NORMAL.prob(legacy_theta[:, 0])
    source_density = _NORMAL.prob(theta)
    scale = tf.stack(
        (
            tf.constant(0.8, tf.float64)
            * source_density[:, 0]
            / legacy_gamma_density,
            tf.constant(0.8, tf.float64) * source_density[:, 1] / beta,
        ),
        axis=1,
    )
    return legacy_theta, scale


def test_frozen_raw_dataset_and_ksc_transform_are_distinct_and_stable() -> None:
    states, observations = generate_frozen_exact_sv_dataset_tf()
    assert hashlib.sha256(tf.io.serialize_tensor(states).numpy()).hexdigest() == (
        KSC_UKF_STATE_SHA256
    )
    assert hashlib.sha256(
        tf.io.serialize_tensor(observations).numpy()
    ).hexdigest() == KSC_UKF_RAW_OBSERVATION_SHA256
    transformed = transformed_ksc_observations(observations)
    exact = tf.math.log(tf.square(observations))
    assert bool(tf.reduce_all(tf.math.is_finite(transformed)).numpy()) is True
    assert bool(tf.reduce_any(tf.not_equal(transformed, exact)).numpy()) is True
    assert KSC_UKF_TRANSFORM_OFFSET == 1.0e-8


def test_graph_likelihood_matches_legacy_principal_sqrt_wrapper_t1_t2() -> None:
    _states, observations = generate_frozen_exact_sv_dataset_tf(horizon=2)
    theta = _theta()
    gamma, beta = source_chart_physical_parameters(theta)
    legacy_theta, score_scale = _legacy_theta_and_score_scale(theta)
    del legacy_theta
    for horizon in (1, 2):
        adapter = make_ksc_ukf_neutra_adapter(
            raw_observations=observations[:horizon]
        )
        graph_value, graph_score = ksc_ukf_likelihood_value_score(
            theta,
            transformed_observations=adapter.transformed_observations,
            mixture_weights=adapter.mixture_weights,
            mixture_means=adapter.mixture_means,
            mixture_variances=adapter.mixture_variances,
        )
        reference_values = []
        reference_scores = []
        for index in range(int(theta.shape[0])):
            reference = independent_panel_sv_mixture_ukf_score(
                observations[:horizon],
                gamma=gamma[index : index + 1],
                beta=beta[index : index + 1],
                sigma=tf.constant([1.0], tf.float64),
            )
            value_reference = independent_panel_sv_mixture_ukf_filter(
                observations[:horizon],
                gamma=gamma[index : index + 1],
                beta=beta[index : index + 1],
                sigma=tf.constant([1.0], tf.float64),
            )
            reference_values.append(value_reference.log_likelihood)
            reference_scores.append(reference.score * score_scale[index])
        tf.debugging.assert_near(
            graph_value, tf.stack(reference_values), atol=2.0e-10, rtol=2.0e-10
        )
        tf.debugging.assert_near(
            graph_score,
            tf.stack(reference_scores),
            atol=3.0e-8,
            rtol=3.0e-8,
        )


def test_manual_score_matches_centered_finite_difference() -> None:
    _states, observations = generate_frozen_exact_sv_dataset_tf(horizon=20)
    adapter = make_ksc_ukf_neutra_adapter(raw_observations=observations)
    theta = _theta()
    value, score = adapter.log_prob_and_grad(theta)
    epsilon = tf.constant(1.0e-5, tf.float64)
    columns = []
    for coordinate in range(2):
        basis = tf.one_hot(coordinate, 2, dtype=tf.float64)[None, :]
        columns.append(
            (adapter.log_prob(theta + epsilon * basis) - adapter.log_prob(theta - epsilon * basis))
            / (2.0 * epsilon)
        )
    finite_difference = tf.stack(columns, axis=1)
    assert bool(tf.reduce_all(tf.math.is_finite(value)).numpy()) is True
    tf.debugging.assert_near(score, finite_difference, atol=2.0e-7, rtol=2.0e-7)


def test_batch_permutation_status_and_typed_binding() -> None:
    _states, observations = generate_frozen_exact_sv_dataset_tf(horizon=20)
    adapter = make_ksc_ukf_neutra_adapter(raw_observations=observations)
    theta = _theta()
    value, score, status = adapter.neutra_batch_log_prob_and_grad_status(theta)
    reversed_value, reversed_score, reversed_status = (
        adapter.neutra_batch_log_prob_and_grad_status(tf.reverse(theta, axis=(0,)))
    )
    tf.debugging.assert_near(value, tf.reverse(reversed_value, axis=(0,)))
    tf.debugging.assert_near(score, tf.reverse(reversed_score, axis=(0,)))
    assert bool(tf.reduce_all(status["valid_pre_regularized_score"]).numpy()) is True
    assert bool(tf.reduce_all(tf.equal(status["status_code"], 0)).numpy()) is True
    tf.debugging.assert_near(
        status["maximum_mixture_weight_sum_error"],
        tf.reverse(reversed_status["maximum_mixture_weight_sum_error"], axis=(0,)),
    )
    signature = stable_ssm_target_signature(adapter.contract)
    binding = bind_batch_native_neutra_target(adapter, target_signature=signature)
    assert binding.target_signature == signature
    assert binding.sample_axis_python_loop_used is False
    assert binding.row_mapped_scalar_target_used is False


def test_cpu_xla_value_score_status() -> None:
    _states, observations = generate_frozen_exact_sv_dataset_tf(horizon=20)
    adapter = make_ksc_ukf_neutra_adapter(raw_observations=observations)

    @tf.function(
        input_signature=[tf.TensorSpec([None, 2], tf.float64)],
        jit_compile=True,
    )
    def compiled(theta):
        return adapter.neutra_batch_log_prob_and_grad_status(theta)

    value, score, status = compiled(_theta())
    assert value.shape == (5,)
    assert score.shape == (5, 2)
    assert bool(tf.reduce_all(status["valid_pre_regularized_score"]).numpy()) is True


def test_active_target_has_no_python_loop_or_callback() -> None:
    source = textwrap.dedent(
        inspect.getsource(_ksc_ukf_likelihood_value_score_status)
    )
    tree = ast.parse(source)
    assert not any(isinstance(node, (ast.For, ast.AsyncFor)) for node in ast.walk(tree))
    forbidden = {"map_fn", "vectorized_map", "numpy_function", "py_function"}
    called = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert forbidden.isdisjoint(called)
    assert ".numpy(" not in source
    assert "numpy" not in source
