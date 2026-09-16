"""Regression gates for numerical batching and the complete compiled filter."""

from __future__ import annotations

import importlib.util
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.linear.rectangular_factor_tf import _batched_qr_with_derivative
from bayesfilter.linear.stack_qr_tf import batched_stack_qr_lower
from bayesfilter.nonlinear.factor_srukf_tf import tf_factor_srukf_value_and_score
from bayesfilter.testing.compiled_filter_runtime_fixture_tf import (
    fixture_components,
    fixture_result,
    numerical_result,
)


def _runtime_audit_module():
    path = Path(__file__).resolve().parents[1] / "scripts/audit_kalman_ukf_runtime.py"
    spec = importlib.util.spec_from_file_location("runtime_audit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runtime_audit_has_no_unclassified_numerical_iteration():
    report = _runtime_audit_module().audit()
    assert report["guard_violations"] == []
    assert report["stale_exceptions"] == []


def test_runtime_audit_does_not_exempt_new_numerics_in_a_schema_function():
    module = _runtime_audit_module()
    path = "bayesfilter/linear/compiled_recurrence_tf.py"
    source = (Path(__file__).resolve().parents[1] / path).read_text()
    modified = source.replace(
        "    signature = tuple(",
        "    for row in state[0]:\n        tf.reduce_sum(row)\n    signature = tuple(",
    )
    assert modified != source
    iterations = module.inspect_source(path, modified, module.read_exceptions())[
        "iterations"
    ]
    violations = [
        row for row in iterations if row["classification"] == "runtime_python_iteration"
    ]
    assert len(violations) == 1
    assert violations[0]["kind"] == "For"


def graph_nodes(function):
    graph = function.graph.as_graph_def()
    return list(graph.node) + [
        node for body in graph.library.function for node in body.node_def
    ]


@pytest.mark.parametrize("rectangular", [False, True])
def test_qr_graph_does_not_grow_with_parameter_count(rectangular):
    counts = []
    for p in (2, 18):
        a_shape = [2, 5, 3] if rectangular else [2, 3, 5]
        da_shape = [2, p, *a_shape[1:]]

        def core(a, da):
            if rectangular:
                return _batched_qr_with_derivative(a, da)
            factor, derivative, _ = batched_stack_qr_lower(
                a, da, compute_covariance_diagnostics=False
            )
            return factor, derivative

        function = tf.function(
            core,
            input_signature=[
                tf.TensorSpec(a_shape, tf.float64),
                tf.TensorSpec(da_shape, tf.float64),
            ],
            jit_compile=True,
        )
        nodes = graph_nodes(function.get_concrete_function())
        assert not {"PyFunc", "PyFuncStateless", "EagerPyFunc"} & {n.op for n in nodes}
        counts.append(len(nodes))
    assert counts[1] <= counts[0] + 8


def test_rectangular_qr_dq_dr_independent_differences_and_chain_isolation():
    a = tf.random.stateless_normal([2, 5, 3], [641, 12], dtype=tf.float64)
    da = 0.2 * tf.random.stateless_normal([2, 4, 5, 3], [731, 18], dtype=tf.float64)
    core = tf.function(
        _batched_qr_with_derivative,
        input_signature=[
            tf.TensorSpec(a.shape, tf.float64),
            tf.TensorSpec(da.shape, tf.float64),
        ],
        jit_compile=True,
    )
    q, r, dq, dr = core(a, da)
    for i in range(4):
        qp, rp, _, _ = _batched_qr_with_derivative(a + 1e-5 * da[:, i])
        qm, rm, _, _ = _batched_qr_with_derivative(a - 1e-5 * da[:, i])
        np.testing.assert_allclose(dq[:, i], (qp - qm) / 2e-5, rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(dr[:, i], (rp - rm) / 2e-5, rtol=1e-6, atol=1e-8)
    changed = core(tf.concat((a[:1], 1.3 * a[1:]), 0), da)
    for before, after in zip((q, r, dq, dr), changed):
        np.testing.assert_array_equal(before[0], after[0])


@pytest.mark.parametrize("rectangular", [False, True])
def test_complete_nonlinear_filter_differences_and_joint_callback(rectangular):
    theta = tf.constant([[0.12, -0.17, 0.23], [-0.2, 0.19, -0.11]], tf.float64)

    @tf.function(input_signature=[tf.TensorSpec([2, 3], tf.float64)], jit_compile=True)
    def run(x):
        return numerical_result(fixture_result(x, rectangular=rectangular, joint=True))

    out = run(theta)
    reference = numerical_result(
        fixture_result(theta, rectangular=rectangular, joint=False, jit_compile=False)
    )
    for a, b in zip(out, reference):
        np.testing.assert_allclose(a, b, rtol=2e-9, atol=2e-10)
    for i in range(3):
        delta = tf.one_hot(i, 3, dtype=tf.float64)[None] * 1e-5
        plus, minus = run(theta + delta), run(theta - delta)
        np.testing.assert_allclose(
            out[1][:, i], (plus[0] - minus[0]) / 2e-5, rtol=2e-6, atol=2e-8
        )
        np.testing.assert_allclose(
            out[4][:, i], (plus[2] - minus[2]) / 2e-5, rtol=2e-6, atol=2e-8
        )
        np.testing.assert_allclose(
            out[5][:, i], (plus[3] - minus[3]) / 2e-5, rtol=2e-6, atol=2e-8
        )
    assert run.experimental_get_tracing_count() == 1


def test_full_rank_rectangular_specialization_preserves_all_outputs():
    theta = tf.constant([[0.12, -0.17, 0.23], [-0.2, 0.19, -0.11]], tf.float64)
    square = numerical_result(fixture_result(theta, rectangular=False))
    rectangular = numerical_result(fixture_result(theta, rectangular=True))
    for a, b in zip(square, rectangular):
        np.testing.assert_allclose(a, b, rtol=2e-9, atol=2e-10)


def test_compiled_direct_filter_rejects_invalid_input_factor_per_chain():
    theta = tf.constant([[0.12, -0.17, 0.23], [-0.2, 0.19, -0.11]], tf.float64)
    model, derivatives = fixture_components(theta)
    bad = tf.tensor_scatter_nd_update(model.initial_factor, [[1, 0, 0]], [-0.7])
    result = tf_factor_srukf_value_and_score(
        tf.zeros([2, 3, 2], tf.float64), replace(model, initial_factor=bad), derivatives
    )
    assert result.diagnostics["valid_pre_regularized_score"].numpy().tolist() == [
        True,
        False,
    ]
    assert bool(tf.reduce_all(tf.math.is_nan(result.score[1])))
    assert bool(tf.reduce_all(tf.math.is_finite(result.score[0])))


@pytest.mark.parametrize("rectangular", [False, True])
def test_filter_graph_size_is_bounded_in_observation_horizon(rectangular):
    def compile_horizon(horizon):
        @tf.function(
            input_signature=[tf.TensorSpec([2, 3], tf.float64)], jit_compile=True
        )
        def target(theta):
            return numerical_result(
                fixture_result(
                    theta, rectangular=rectangular, joint=True, horizon=horizon
                )
            )

        return graph_nodes(target.get_concrete_function())

    small, large = compile_horizon(2), compile_horizon(24)
    assert len(large) <= len(small) + 10
    assert any(node.op in {"While", "StatelessWhile"} for node in large)
    assert not {"PyFunc", "PyFuncStateless", "EagerPyFunc"} & {
        node.op for node in large
    }


def test_qr_derivative_preserves_an_ill_conditioned_valid_stack():
    scales = tf.constant([1.0, 0.03, 0.001], tf.float64)
    a = tf.random.stateless_normal([2, 6, 3], [902, 17], dtype=tf.float64) * scales
    da = (
        0.1
        * tf.random.stateless_normal([2, 2, 6, 3], [709, 23], dtype=tf.float64)
        * scales
    )
    q, r, dq, dr = tf.function(_batched_qr_with_derivative, jit_compile=True)(a, da)
    np.testing.assert_allclose(q @ r, a, rtol=2e-12, atol=2e-14)
    for i in range(2):
        plus = _batched_qr_with_derivative(a + 1e-5 * da[:, i])
        minus = _batched_qr_with_derivative(a - 1e-5 * da[:, i])
        np.testing.assert_allclose(
            dq[:, i], (plus[0] - minus[0]) / 2e-5, rtol=2e-5, atol=2e-8
        )
        np.testing.assert_allclose(
            dr[:, i], (plus[1] - minus[1]) / 2e-5, rtol=2e-5, atol=2e-8
        )


def test_sgqf_tensor_status_survives_outer_xla_and_stops_at_first_failure():
    from bayesfilter.nonlinear.fixed_sgqf_compiled_tf import fixed_sgqf_tensor_result
    from bayesfilter.nonlinear.fixed_sgqf_tf import (
        TFFixedSGQFBranchConfig,
        tf_fixed_sgqf_p47_one_step_oracle,
    )

    oracle, cloud = tf_fixed_sgqf_p47_one_step_oracle()
    model = oracle.model()
    observations = tf.repeat(oracle.observation, 3, axis=0)
    config = TFFixedSGQFBranchConfig()

    @tf.function(input_signature=[tf.TensorSpec([1, 1], tf.float64)], jit_compile=True)
    def target(covariance):
        return fixed_sgqf_tensor_result(
            observations, replace(model, initial_covariance=covariance), cloud, config
        )

    valid = target(model.initial_covariance)
    assert bool(valid["valid"])
    rejected = target(tf.constant([[-1.0]], tf.float64))
    assert int(rejected["status_code"]) == 1
    assert int(rejected["attempted_steps"]) == 1
    assert int(rejected["accepted_steps"]) == 0
    assert bool(tf.math.is_nan(rejected["log_likelihood"]))


def test_rectangular_invalid_chain_reports_numeric_status_under_outer_xla():
    from bayesfilter.nonlinear.rectangular_srukf_tf import (
        TFRectangularSRUKFFixedBranch,
        tf_rectangular_srukf_value_and_score,
    )

    theta = tf.constant([[0.12, -0.17, 0.23], [-0.2, 0.19, -0.11]], tf.float64)
    model, derivatives = fixture_components(theta, rectangular=True)
    branch = TFRectangularSRUKFFixedBranch(2, (0, 1), 2, (0, 1), 2, (0, 1))

    @tf.function(
        input_signature=[tf.TensorSpec([2, 2, 2], tf.float64)], jit_compile=True
    )
    def target(factor):
        result = tf_rectangular_srukf_value_and_score(
            tf.zeros([2, 3, 2], tf.float64),
            replace(model, initial_factor=factor),
            derivatives,
            branch=branch,
        )
        return (
            result.score,
            result.diagnostics["branch_status_code"],
            result.diagnostics["score_valid"],
        )

    bad = tf.tensor_scatter_nd_update(model.initial_factor, [[1, 0, 0]], [float("nan")])
    score, status, valid = target(bad)
    assert valid.numpy().tolist() == [True, False]
    assert status.numpy().tolist() == [0, 1]
    assert bool(tf.reduce_all(tf.math.is_finite(score[0])))
    assert bool(tf.reduce_all(tf.math.is_nan(score[1])))
