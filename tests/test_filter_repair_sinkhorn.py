"""Independent September mechanics checks for finite Sinkhorn execution.

The reference deliberately uses NumPy outside runtime code. It is not LEDH
admission evidence and does not reuse any historical benchmark result.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import tensorflow as tf
from scipy.special import logsumexp

from experiments.dpf_implementation.tf_tfp.resampling.sinkhorn_tf import (
    build_sinkhorn_log_state_tf,
    make_sinkhorn_resample_tf,
    sinkhorn_resample_tf,
)
from experiments.dpf_implementation.tf_tfp.structural.particle_state_tf import (
    StructuralParticleStateTF,
)
from experiments.dpf_implementation.tf_tfp.structural.resampling_policies_tf import (
    SINKHORN_CURRENT_Z,
    SINKHORN_FULL_CONTEXT,
    apply_structural_resampling_policy_tf,
)


POINTS = np.array([[-0.8, 0.2], [-0.1, -0.3], [0.4, 0.1], [0.9, 0.5]])
WEIGHTS = np.array([0.1, 0.2, 0.3, 0.4])


def _reference(points, weights, *, cost=None, log_u=None, log_v=None,
               epsilon=0.8, iterations=80, tolerance=1e-10):
    count = len(weights)
    source = weights / weights.sum()
    target = np.full(count, 1.0 / count)
    if cost is None:
        cost = np.square(points[:, None] - points[None, :]).sum(axis=-1)
    kernel = -cost / epsilon
    log_u = np.zeros(count) if log_u is None else log_u.copy()
    log_v = np.zeros(count) if log_v is None else log_v.copy()
    for iteration in range(1, iterations + 1):
        log_u = np.log(np.maximum(source, 1e-300)) - logsumexp(kernel + log_v[None], axis=1)
        log_v = np.log(target) - logsumexp(kernel + log_u[:, None], axis=0)
        if iteration == iterations or iteration % 10 == 0:
            coupling = np.exp(log_u[:, None] + kernel + log_v[None])
            error = max(np.max(np.abs(coupling.sum(axis=1) - source)),
                        np.max(np.abs(coupling.sum(axis=0) - target)))
            if error <= tolerance:
                break
    coupling = np.exp(log_u[:, None] + kernel + log_v[None])
    particles = coupling.T @ points / np.maximum(coupling.sum(axis=0), 1e-300)[:, None]
    return particles, coupling, log_u, log_v, iteration


@pytest.mark.parametrize("jit", [False, True])
@pytest.mark.parametrize("supplied_cost", [False, True])
@pytest.mark.parametrize("warm_start", [False, True])
def test_native_sinkhorn_preserves_finite_reference(jit, supplied_cost, warm_start):
    cost = np.square(POINTS[:, None] - POINTS[None, :]).sum(axis=-1) * 0.7 if supplied_cost else None
    log_u = np.array([0.3, -0.2, 0.1, 0.4]) if warm_start else None
    log_v = np.array([-0.2, 0.4, -0.1, 0.2]) if warm_start else None
    state = None if not warm_start else build_sinkhorn_log_state_tf(log_u, log_v)
    expected = _reference(POINTS, WEIGHTS, cost=cost, log_u=log_u, log_v=log_v)
    actual = sinkhorn_resample_tf(POINTS, WEIGHTS, cost=cost, initial_state=state,
        epsilon=0.8, max_iterations=80, tolerance=1e-10, jit_compile=jit)
    for value, reference in zip((actual.particles, actual.coupling,
            actual.final_state.log_u, actual.final_state.log_v), expected[:4]):
        np.testing.assert_allclose(value, reference, rtol=1e-10, atol=1e-10)
    assert actual.diagnostics["iterations_used"] == expected[4]
    assert actual.diagnostics["jit_compile"] is jit
    assert actual.diagnostics["initialization_policy"] == ("provided_log_state" if warm_start else "zeros")
    assert abs(float(tf.reduce_mean(actual.canonicalized_final_state.log_u))) < 1e-14


@pytest.mark.parametrize("iterations,tolerance,used", [(7, 1.0, 7), (23, 1.0, 10), (23, 0.0, 23)])
def test_stopping_cadence_is_unchanged(iterations, tolerance, used):
    call = make_sinkhorn_resample_tf(tf.TensorSpec([4, 2], tf.float64),
        epsilon=0.8, max_iterations=iterations, tolerance=tolerance)
    actual = call(POINTS, WEIGHTS, tf.zeros([0, 0], tf.float64),
                  tf.zeros([4], tf.float64), tf.zeros([4], tf.float64))
    expected = _reference(POINTS, WEIGHTS, iterations=iterations, tolerance=tolerance)
    assert int(actual.iterations_used) == used == expected[4]
    np.testing.assert_allclose(actual.coupling, expected[1], rtol=1e-10, atol=1e-10)


def test_sinkhorn_complete_xla_gradient_matches_independent_finite_difference():
    call = make_sinkhorn_resample_tf(tf.TensorSpec([4, 2], tf.float64),
        epsilon=0.8, max_iterations=7, tolerance=1.0)

    @tf.function(input_signature=[tf.TensorSpec([4, 2], tf.float64)],
                 jit_compile=True, autograph=False)
    def score(points):
        with tf.GradientTape() as tape:
            tape.watch(points)
            result = call(points, WEIGHTS, tf.zeros([0, 0], tf.float64),
                          tf.zeros([4], tf.float64), tf.zeros([4], tf.float64))
            value = tf.reduce_sum(tf.square(result.particles))
        return value, tape.gradient(value, points)

    value, gradient = score(POINTS)
    expected = _reference(POINTS, WEIGHTS, iterations=7, tolerance=1.0)
    np.testing.assert_allclose(value, np.square(expected[0]).sum(), rtol=1e-10, atol=1e-10)
    direction = np.arange(8, dtype=float).reshape(4, 2) / 8.0 - 0.3
    epsilon = 1e-5
    plus = _reference(POINTS + epsilon * direction, WEIGHTS, iterations=7, tolerance=1.0)[0]
    minus = _reference(POINTS - epsilon * direction, WEIGHTS, iterations=7, tolerance=1.0)[0]
    finite_difference = (np.square(plus).sum() - np.square(minus).sum()) / (2 * epsilon)
    np.testing.assert_allclose(np.sum(gradient * direction), finite_difference, rtol=1e-8, atol=1e-8)


def test_factory_is_stable_and_iteration_budget_does_not_unroll_graph():
    spec = tf.TensorSpec([4, 2], tf.float64)
    first = make_sinkhorn_resample_tf(spec, max_iterations=7)
    assert first is make_sinkhorn_resample_tf(spec, max_iterations=7)

    def nodes(call):
        graph = call.get_concrete_function().graph.as_graph_def()
        nodes = [*graph.node, *(node for fn in graph.library.function for node in fn.node_def)]
        assert not ({node.op for node in nodes} & {"PyFunc", "EagerPyFunc", "PyFuncStateless"})
        return len(nodes)

    assert nodes(first) == nodes(make_sinkhorn_resample_tf(spec, max_iterations=83))


def test_eager_tape_over_xla_call_preserves_all_input_gradients():
    points = tf.constant(POINTS, tf.float64)
    weights = tf.constant(WEIGHTS, tf.float64)
    cost = tf.constant(np.square(POINTS[:, None] - POINTS[None, :]).sum(axis=-1), tf.float64)
    u = tf.constant([0.3, -0.2, 0.1, 0.4], tf.float64)
    v = tf.constant([-0.2, 0.4, -0.1, 0.2], tf.float64)
    inputs = (points, weights, cost, u, v)
    gradients = []
    for jit in (False, True):
        call = make_sinkhorn_resample_tf(tf.TensorSpec([4, 2], tf.float64),
            epsilon=0.8, max_iterations=7, tolerance=1.0, supplied_cost=True,
            jit_compile=jit)
        with tf.GradientTape() as tape:
            tape.watch(inputs)
            result = call(*inputs)
            objective = (tf.reduce_sum(tf.square(result.particles))
                         + tf.reduce_sum(result.log_u * 0.2 + result.log_v * 0.3)
                         + tf.reduce_sum(tf.square(result.coupling))
                         + tf.reduce_sum(tf.square(result.source)))
        gradients.append(tape.gradient(objective, inputs))
    for actual, expected in zip(gradients[1], gradients[0]):
        assert actual is not None and expected is not None
        np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)


def test_veto_and_warm_start_validation_are_preserved():
    with pytest.raises(FloatingPointError, match="row residual"):
        sinkhorn_resample_tf(POINTS, WEIGHTS, epsilon=0.1, max_iterations=1, tolerance=1e-12)
    with pytest.raises(ValueError, match="provided together"):
        sinkhorn_resample_tf(POINTS, WEIGHTS, initial_log_u=tf.zeros([4]))
    with pytest.raises(ValueError, match="match particle count"):
        sinkhorn_resample_tf(POINTS, WEIGHTS, initial_log_u=tf.zeros([3]), initial_log_v=tf.zeros([3]))


@pytest.mark.parametrize("policy", [SINKHORN_CURRENT_Z, SINKHORN_FULL_CONTEXT])
def test_structural_consumers_preserve_completion_and_context(policy):
    z = tf.constant(POINTS, tf.float64)
    previous_z = z * 0.4
    previous_s = tf.reduce_sum(z, axis=1, keepdims=True) * 0.2

    def complete(s, old_z, new_z):
        return s + tf.reduce_sum(new_z - old_z, axis=1, keepdims=True)

    model = SimpleNamespace(complete_s=complete,
        completion_residual=lambda s, old_z, new_z, new_s: new_s - complete(s, old_z, new_z))
    state = StructuralParticleStateTF(previous_z, previous_s, z, complete(previous_s, previous_z, z))
    actual = apply_structural_resampling_policy_tf(model=model, state=state, weights=WEIGHTS,
        policy_id=policy, seed=17, time_index=0, sinkhorn_epsilon=0.8,
        sinkhorn_iterations=80, sinkhorn_tolerance=1e-10)
    context = z if policy == SINKHORN_CURRENT_Z else tf.concat([previous_z, previous_s, z], axis=1)
    expected = _reference(context.numpy(), WEIGHTS)[0]
    if policy == SINKHORN_CURRENT_Z:
        expected_z = expected
        expected_s = complete(previous_s, previous_z, expected_z)
    else:
        expected_z = expected[:, 3:]
        expected_s = complete(expected[:, 2:3], expected[:, :2], expected_z)
    np.testing.assert_allclose(actual.next_z, expected_z, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(actual.next_s, expected_s, rtol=1e-10, atol=1e-10)
    assert actual.diagnostics["max_completion_residual_after_policy"] == 0.0
