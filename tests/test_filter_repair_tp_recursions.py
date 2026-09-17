"""Independent diagnostic charts and refactor parity; no LEDH admission claims."""

from functools import lru_cache

import numpy as np
import pytest
import tensorflow as tf
from scipy.optimize import linprog

from bayesfilter.highdim import ledh_contract_e_tp_lgssm_tf as model
from bayesfilter.highdim import ledh_contract_e_tp_tf as tp
from tests.test_filter_repair_remaining_routes import _graph, _original

D = tf.float64
MODES = ("next_predictive", "progressive_target_model_score", "exact_continuation", "finite_lookahead")


@lru_cache(maxsize=8)
def _fixture(mode, horizon):
    """Build fresh positive charts at one center, then freeze them for parity."""
    before = _original("ledh_contract_e_tp_lgssm_tf")
    theta = tf.constant([0.72, 0.55, 0.35, 0.35, 0.45], D)
    observations = tf.reshape(tf.linspace(tf.constant(-0.1, D), 0.2, horizon * 3), [horizon, 3])
    nodes, weights = np.polynomial.hermite.hermgauss(3)
    nodes, weights = tf.constant(nodes * np.sqrt(2.), D), tf.constant(weights / np.sqrt(np.pi), D)
    parents, log_parent_weights, innovations, log_innovation_weights = before.initial_parents(theta, nodes, weights)
    parent_marks = before._initial_target_model_score_marks(theta, parents)
    if mode == "exact_continuation":
        matrices, vectors = before._backward_information_parameters(theta, observations)
    elif mode == "finite_lookahead":
        matrices, vectors = before._finite_lookahead_information_parameters(theta, observations, 2)
    indices, scales = [], []
    feature_count = model.PROGRESSIVE_FEATURE_COUNT if mode == "progressive_target_model_score" else model.FEATURE_COUNT
    for time in range(horizon - 1):
        flow = before._flow_correction(theta, parents, innovations, observations[time])
        log_weights = before._combine_parent_innovation_log_weights(
            log_parent_weights, log_innovation_weights, flow["log_correction"]
        )
        if mode == "next_predictive":
            features = before._features(theta, flow["particles"], innovations, log_innovation_weights, observations[time + 1])
        elif mode == "progressive_target_model_score":
            marks = before._target_model_progressive_score_marks(
                theta, parents, log_parent_weights, parent_marks, flow["particles"], observations[time]
            )
            features, _, centered = before._progressive_features(
                theta, flow["particles"], log_weights, marks, innovations,
                log_innovation_weights, observations[time + 1],
            )
        else:
            features = before._continuation_features_from_information(flow["particles"], matrices[time], vectors[time])
        row_scale = tf.reduce_max(tf.abs(features), axis=1)
        normalized = tf.exp(log_weights - tf.reduce_logsumexp(log_weights))
        target = tf.linalg.matvec(features, normalized)
        matrix = (features / row_scale[:, None]).numpy()
        selected = linprog(
            np.sin(np.arange(matrix.shape[1], dtype=float)),
            A_eq=matrix, b_eq=(target / row_scale).numpy(), bounds=(0., None), method="highs-ds",
        )
        assert selected.success, selected.message
        active = np.flatnonzero(selected.x > 1e-9).astype(np.int32)
        assert active.size == feature_count
        projection = tp._contract_e_tp_dense_square_forward_core(
            flow["particles"], log_weights, features, tf.constant(active), row_scale
        )
        assert bool(projection["valid_chart"])
        indices.append(active)
        scales.append(row_scale)
        parents = projection["student_points"]
        log_parent_weights = tf.math.log(projection["student_weights"])
        if mode == "progressive_target_model_score":
            parent_marks = tf.gather(centered, active)
            parent_marks -= tf.einsum("n,np->p", projection["student_weights"], parent_marks)[None, :]
    return (
        theta, observations, nodes, weights,
        tf.constant(np.asarray(indices, dtype=np.int32).reshape(horizon - 1, feature_count)),
        tf.reshape(tf.convert_to_tensor(scales, D), [horizon - 1, feature_count]),
    )


def _evaluate(module, mode, theta, inputs):
    with tf.GradientTape() as tape:
        tape.watch(theta)
        if mode == "next_predictive":
            result = module.contract_e_tp_lgssm_recursive_core(theta, *inputs)
        else:
            result = module.contract_e_tp_lgssm_score_informed_recursive_core(
                theta, *inputs, feature_mode=mode, lookahead_steps=2
            )
    return result, tape.gradient(result["objective"], theta)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("horizon", [1, 4])
@pytest.mark.parametrize("jit", [False, True])
def test_tp_recursions_preserve_complete_values_scores_and_history(mode, horizon, jit):
    theta, *inputs = _fixture(mode, horizon)
    baseline = _original("ledh_contract_e_tp_lgssm_tf")
    eager_reference = _evaluate(baseline, mode, theta, inputs)
    reference = tf.function(lambda parameters: _evaluate(baseline, mode, parameters, inputs),
                            input_signature=[tf.TensorSpec([5], D)], jit_compile=jit, autograph=False)
    expected = reference(theta)
    call = tf.function(lambda parameters: _evaluate(model, mode, parameters, inputs),
                       input_signature=[tf.TensorSpec([5], D)], jit_compile=jit, autograph=False)
    actual = call(theta)
    tf.nest.assert_same_structure(actual, expected)
    for result, reference in zip(tf.nest.flatten(actual), tf.nest.flatten(expected), strict=True):
        if result.dtype == tf.bool:
            np.testing.assert_array_equal(result, reference)
        else:
            assert bool(tf.reduce_all(tf.math.is_finite(result)))
            np.testing.assert_allclose(result, reference, rtol=1e-10, atol=1e-10)
    assert bool(tf.reduce_all(actual[0]["valid_history"]))
    np.testing.assert_allclose(actual[0]["objective"], eager_reference[0]["objective"], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(actual[1], eager_reference[1], rtol=1e-10, atol=1e-10)
    direction = tf.constant([0.13, -0.08, 0.04, 0.12, -0.09], D)
    step = 1e-5
    fd = (call(theta + step * direction)[0]["objective"] - call(theta - step * direction)[0]["objective"]) / (2 * step)
    np.testing.assert_allclose(tf.tensordot(actual[1], direction, 1), fd, rtol=1e-7, atol=1e-8)
    _graph(call)
    if jit:
        assert "HloModule" in call.experimental_get_compiler_ir(theta)(stage="hlo")


def test_tp_epsilon_preserves_supported_dtype_thresholds():
    for dtype in (tf.float16, tf.float32, tf.float64):
        assert float(tp._dtype_epsilon(dtype)) == float(np.finfo(dtype.as_numpy_dtype).eps)
    with pytest.raises(TypeError, match="Unsupported"):
        tp._dtype_epsilon(tf.int32)


def test_tp_recursion_graph_does_not_unroll_horizon():
    counts = []
    for horizon in (4, 9):
        call = tf.function(lambda theta, horizon=horizon: model._contract_e_tp_lgssm_loop_core(
            theta, tf.zeros([horizon, 3], D), tf.constant([-1., 0., 1.], D),
            tf.constant([0.25, 0.5, 0.25], D),
            tf.zeros([horizon - 1, model.FEATURE_COUNT], tf.int32),
            tf.ones([horizon - 1, model.FEATURE_COUNT], D),
            feature_mode="next_predictive", lookahead_steps=None,
        )["objective"], input_signature=[tf.TensorSpec([5], D)], jit_compile=True, autograph=False)
        counts.append(len(_graph(call)))
    assert counts[0] == counts[1]
