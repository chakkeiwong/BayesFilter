"""First-projection localization of the preserved raw TP residual mismatch.

This is a fresh engineering diagnostic of the existing noncanonical fixture.
"""

import json

import pytest
import tensorflow as tf

from bayesfilter.highdim import ledh_contract_e_tp_predator_prey_tf as candidate
from bayesfilter.highdim import ledh_contract_e_tp_tf as tp
from tests.test_filter_repair_predator_tp import _evaluate, _fixture
from tests.test_filter_repair_remaining_routes import _original

D = tf.float64


def test_gpu_finite_difference_step_localization(record_property):
    """Diagnostic only: keep the original gate and expose truncation/roundoff."""
    before = _original("ledh_contract_e_tp_predator_prey_tf")
    values = _fixture(4, 2)
    signature = tuple(tf.TensorSpec(value.shape, value.dtype) for value in values)
    reference = tf.function(lambda *args: _evaluate(before, args, 2),
        input_signature=signature, jit_compile=True, autograph=False)
    execute = tf.function(lambda *args: _evaluate(candidate, args, 2),
        input_signature=signature, jit_compile=True, autograph=False)
    direction = tf.constant([.1, -.04, .05, .03, -.06, .02], D)
    original, actual = reference(*values), execute(*values)
    directional = float(tf.tensordot(actual[1], direction, 1))
    rows = []
    for delta in (1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6, 3e-7):
        plus = execute(values[0] + delta * direction, *values[1:])[0]
        minus = execute(values[0] - delta * direction, *values[1:])[0]
        old_plus = reference(values[0] + delta * direction, *values[1:])[0]
        old_minus = reference(values[0] - delta * direction, *values[1:])[0]
        rows.append({"delta": delta, "central": float((plus["objective"] - minus["objective"]) / (2 * delta)),
            "baseline_central": float((old_plus["objective"] - old_minus["objective"]) / (2 * delta)),
            "directional": directional,
            "plus_difference": float(plus["objective"] - old_plus["objective"]),
            "minus_difference": float(minus["objective"] - old_minus["objective"]),
            "valid": bool(tf.reduce_all(plus["valid_history"]) & tf.reduce_all(minus["valid_history"]))})
    record_property("finite_difference_ladder", json.dumps(rows, sort_keys=True))
    record_property("baseline_directional", str(float(tf.tensordot(original[1], direction, 1))))
    record_property("diagnostic_only_original_derivative_gate_unchanged", "true")
    assert all(row["valid"] for row in rows)
    for row in rows:
        assert abs(row["plus_difference"]) < 1e-10
        assert abs(row["minus_difference"]) < 1e-10


@pytest.mark.parametrize("jit", [False, True])
def test_dynamic_continuation_windows_keep_values_and_total_gradients(jit):
    before = _original("ledh_contract_e_tp_predator_prey_tf")
    model = before.p30_predator_prey_fixture_model()
    theta, _observations, _nodes, _weights, _indices, _scales, grid, grid_weights = _fixture(4, 2)
    cloud = model.initial_mean[None, :] + tf.constant([[.1, .2], [-.2, .3], [.4, -.1]], D)
    observations = model.initial_mean[None, :] + tf.reshape(
        tf.linspace(tf.constant(-.15, D), tf.constant(.25, D), 8), [4, 2])
    specifications = (tf.TensorSpec(theta.shape, D), tf.TensorSpec([], tf.int32))

    @tf.function(input_signature=specifications, jit_compile=jit, autograph=False)
    def execute(theta, count):
        with tf.GradientTape() as tape:
            tape.watch(theta)
            values = candidate.gaussian_closure_continuation_log_likelihood(
                model, theta, cloud, observations, grid, grid_weights, future_count=count)
            total = tf.reduce_sum(values) + 0.0 * tf.reduce_sum(theta)
        return values, tape.gradient(total, theta)

    def reference(count):
        @tf.function(input_signature=[specifications[0]], jit_compile=jit, autograph=False)
        def run(theta):
            with tf.GradientTape() as tape:
                tape.watch(theta)
                values = before.gaussian_closure_continuation_log_likelihood(
                    model, theta, cloud, observations[:count], grid, grid_weights)
                total = tf.reduce_sum(values) + 0.0 * tf.reduce_sum(theta)
            return values, tape.gradient(total, theta)
        return run(theta)

    for count in range(5):
        expected = reference(count)
        actual = execute(theta, tf.constant(count))
        for left, right in zip(actual, expected, strict=True):
            tf.debugging.assert_near(left, right, atol=1e-10, rtol=1e-10)
    assert execute.experimental_get_tracing_count() == 1
    if jit:
        assert "HloModule" in execute.experimental_get_compiler_ir(theta, tf.constant(4))(stage="hlo")


@pytest.mark.parametrize("jit", [False, True])
def test_continuation_rounding_breakdown(jit, record_property):
    before = _original("ledh_contract_e_tp_predator_prey_tf")
    model = before.p30_predator_prey_fixture_model()
    theta, observations, nodes, weights, _indices, _scales, grid, grid_weights = _fixture(4, 2)
    parents, logs, points, standard_logs = before.initial_rule(model, nodes, weights)
    teacher = before._teacher_step(model, theta, parents, logs, points, standard_logs, observations[0], 0)
    inputs = (theta, teacher["particles"], observations[1:3], grid, grid_weights)
    specifications = tuple(tf.TensorSpec(value.shape, value.dtype) for value in inputs)

    def build(module):
        @tf.function(input_signature=specifications, jit_compile=jit, autograph=False)
        def execute(theta, cloud, future, grid, grid_weights):
            values = module.gaussian_closure_continuation_log_likelihood(
                model, theta, cloud, future, grid, grid_weights)
            reference = module.gaussian_closure_continuation_log_likelihood(
                model, theta, model.initial_mean[None, :], future, grid, grid_weights)[0]
            maximum = tf.maximum(reference, tf.reduce_max(values))
            return {"continuation": values, "reference": reference, "maximum": maximum,
                "normalized_log": values - maximum, "normalized_exp": tf.exp(values - maximum)}
        return execute

    expected, actual = build(before)(*inputs), build(candidate)(*inputs)
    errors = {name: float(tf.reduce_max(tf.abs(actual[name] - expected[name]))) for name in actual}
    record_property("maximum_absolute_errors", json.dumps(errors, sort_keys=True))
    record_property("intermediate_outputs_may_change_fusion", "true")
    print(json.dumps({"jit": jit, "continuation_breakdown": errors}, sort_keys=True))
    for name in expected:
        tf.debugging.assert_near(actual[name], expected[name], atol=1e-10, rtol=1e-10, message=name)


@pytest.mark.parametrize("jit", [False, True])
@pytest.mark.parametrize("freeze", ["none", "teacher", "features"])
def test_first_tp_projection_breakdown(jit, freeze, record_property):
    before = _original("ledh_contract_e_tp_predator_prey_tf")
    model = before.p30_predator_prey_fixture_model()
    theta, observations, nodes, weights, indices, scales, grid, grid_weights = _fixture(4, 2)
    parents, log_weights, points, standard_log_weights = before.initial_rule(model, nodes, weights)
    frozen = before._teacher_step(model, theta, parents, log_weights, points,
        standard_log_weights, observations[0], 0)
    frozen_features = before._features(model, theta, frozen["particles"], observations[1:3],
        grid, grid_weights, first_future_time_index=1)
    inputs = (theta, observations, nodes, weights, indices, scales, grid, grid_weights,
              frozen["particles"], frozen["log_unnormalized_weights"], frozen_features)
    specifications = tuple(tf.TensorSpec(value.shape, value.dtype) for value in inputs)

    def build(module):
        @tf.function(input_signature=specifications, jit_compile=jit, autograph=False)
        def execute(theta, observations, nodes, weights, indices, scales, grid, grid_weights,
                    frozen_points, frozen_log_weights, fixed_features):
            parents, log_weights, points, standard_log_weights = module.initial_rule(model, nodes, weights)
            teacher = module._teacher_step(model, theta, parents, log_weights, points,
                standard_log_weights, observations[0], 0)
            cloud = frozen_points if freeze != "none" else teacher["particles"]
            logs = frozen_log_weights if freeze != "none" else teacher["log_unnormalized_weights"]
            features = module._features(model, theta, cloud, observations[1:3], grid,
                grid_weights, first_future_time_index=1)
            if freeze == "features":
                features = fixed_features
            projected = tp._contract_e_tp_dense_square_forward_core(
                cloud, logs, features, indices[0], scales[0])
            return {"points": cloud, "log_weights": logs, "features": features,
                "scaled_matrix": projected["active_features"] / scales[0, :, None],
                "student_weights": projected["student_weights"],
                "matched_target": projected["matched_target"],
                "raw_residual": projected["feature_residual"]}
        return execute

    expected, actual = build(before)(*inputs), build(candidate)(*inputs)
    errors = {name: float(tf.reduce_max(tf.abs(actual[name] - expected[name]))) for name in actual}
    record_property("maximum_absolute_errors", json.dumps(errors, sort_keys=True))
    record_property("baseline_raw_residual", json.dumps(expected["raw_residual"].numpy().tolist()))
    record_property("candidate_raw_residual", json.dumps(actual["raw_residual"].numpy().tolist()))
    print(json.dumps({"jit": jit, "freeze": freeze, "errors": errors,
        "baseline_raw_residual": expected["raw_residual"].numpy().tolist(),
        "candidate_raw_residual": actual["raw_residual"].numpy().tolist()}, sort_keys=True))
    for name in expected:
        tf.debugging.assert_near(actual[name], expected[name], atol=1e-10, rtol=1e-10, message=name)
