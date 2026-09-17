"""Fresh frozen-chart TP refactor checks, with no canonical LEDH claim."""

from functools import lru_cache

import numpy as np
import pytest
import tensorflow as tf
from scipy.optimize import linprog

from bayesfilter.highdim import ledh_contract_e_tp_predator_prey_tf as candidate
from bayesfilter.highdim import ledh_contract_e_tp_tf as tp
from tests.test_filter_repair_remaining_routes import _graph, _original

D = tf.float64


@lru_cache(maxsize=8)
def _fixture(horizon, lookahead):
    before = _original("ledh_contract_e_tp_predator_prey_tf")
    model = before.p30_predator_prey_fixture_model()
    # The pinned public core constructs and host-validates its fixed model.
    # Lift only that constructor for a local numerical oracle, not a timing arm.
    before.p30_predator_prey_fixture_model = lambda: model
    theta = model.true_parameters()
    observations = model.initial_mean[None, :] + tf.reshape(
        tf.linspace(tf.constant(-.15, D), .2, horizon * 2), [horizon, 2])
    nodes, weights = np.polynomial.hermite.hermgauss(5)
    nodes, weights = tf.constant(nodes * np.sqrt(2.), D), tf.constant(weights / np.sqrt(np.pi), D)
    parents, log_weights, grid, grid_log_weights = before.initial_rule(model, nodes, weights)
    grid_weights = tf.exp(grid_log_weights)
    indices, scales = [], []
    for time in range(horizon - 1):
        teacher = before._teacher_step(model, theta, parents, log_weights, grid,
                                       grid_log_weights, observations[time], time)
        features = before._features(model, theta, teacher["particles"],
            observations[time+1:min(horizon, time+1+lookahead)], grid, grid_weights,
            first_future_time_index=time+1)
        scale = tf.reduce_max(tf.abs(features), axis=1)
        moment = tf.linalg.matvec(features, tf.nn.softmax(teacher["log_unnormalized_weights"]))
        matrix = (features / scale[:, None]).numpy()
        solution = linprog(np.sin(np.arange(matrix.shape[1], dtype=float)),
            A_eq=matrix, b_eq=(moment / scale).numpy(), bounds=(0., None), method="highs-ds")
        assert solution.success, solution.message
        active = np.flatnonzero(solution.x > 1e-9).astype(np.int32)
        assert active.size == candidate.FEATURE_COUNT
        projected = tp._contract_e_tp_dense_square_forward_core(teacher["particles"],
            teacher["log_unnormalized_weights"], features, tf.constant(active), scale)
        assert bool(projected["valid_chart"])
        indices.append(active)
        scales.append(scale)
        parents, log_weights = projected["student_points"], tf.math.log(projected["student_weights"])
    return (theta, observations, nodes, weights,
        tf.constant(np.asarray(indices, dtype=np.int32).reshape(horizon-1, candidate.FEATURE_COUNT)),
        tf.reshape(tf.convert_to_tensor(scales, D), [horizon-1, candidate.FEATURE_COUNT]), grid, grid_weights)


def _evaluate(module, values, lookahead):
    theta, *inputs = values
    with tf.GradientTape() as tape:
        tape.watch(theta)
        row = module.contract_e_tp_predator_prey_recursive_core(theta, *inputs, lookahead_steps=lookahead)
    return row, tape.gradient(row["objective"], theta)


@pytest.mark.parametrize("horizon,lookahead", [(1, 1), (3, 1), (4, 2)])
@pytest.mark.parametrize("jit", [False, True])
def test_complete_predator_tp_same_mode_value_gradient_and_history(horizon, lookahead, jit):
    values = _fixture(horizon, lookahead)
    before = _original("ledh_contract_e_tp_predator_prey_tf")
    specifications = tuple(tf.TensorSpec(x.shape, x.dtype) for x in values)
    reference = tf.function(lambda *args: _evaluate(before, args, lookahead),
                            input_signature=specifications, jit_compile=jit, autograph=False)
    call = tf.function(lambda *args: _evaluate(candidate, args, lookahead),
                       input_signature=specifications, jit_compile=jit, autograph=False)
    expected, actual = reference(*values), call(*values)
    print({name: float(tf.reduce_max(tf.abs(actual[0][name] - expected[0][name])))
        for name in actual[0] if actual[0][name].dtype != tf.bool and tf.size(actual[0][name])})
    print("gradient", actual[1].numpy(), expected[1].numpy(),
          "condition", actual[0]["condition_number_history"].numpy(),
          "scales", values[5].numpy())
    np.testing.assert_allclose(actual[0]["objective"], expected[0]["objective"],
                               rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(actual[1], expected[1], rtol=1e-10, atol=1e-10)
    assert bool(tf.reduce_all(actual[0]["valid_history"]))
    direction = tf.constant([.1, -.04, .05, .03, -.06, .02], D)
    delta = 1e-5
    plus = call(values[0] + delta * direction, *values[1:])[0]["objective"]
    minus = call(values[0] - delta * direction, *values[1:])[0]["objective"]
    np.testing.assert_allclose(tf.tensordot(actual[1], direction, 1), (plus-minus)/(2*delta),
                               atol=1e-8, rtol=1e-7)
    _graph(call)
    if jit:
        assert "HloModule" in call.experimental_get_compiler_ir(*values)(stage="hlo")
    print("objective, gradient, finite difference, validity, and graph checks passed")
    for result, authority in zip(tf.nest.flatten(actual[0]), tf.nest.flatten(expected[0]), strict=True):
        if result.dtype == tf.bool:
            np.testing.assert_array_equal(result, authority)
        else:
            assert bool(tf.reduce_all(tf.math.is_finite(result)))
            np.testing.assert_allclose(result, authority, rtol=1e-10, atol=1e-10)


def test_predator_tp_graph_size_is_independent_of_horizon():
    counts = []
    for horizon in (4, 6):
        values = _fixture(horizon, 2)
        specs = tuple(tf.TensorSpec(x.shape, x.dtype) for x in values)
        call = candidate._recursive_program(specs, 2)
        counts.append(len(_graph(call)))
    assert counts[0] == counts[1]
