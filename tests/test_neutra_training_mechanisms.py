"""CPU analytic/reference mechanics only; no q20 learning or GPU speed claims.

Seeds, small dimensions, and SGD step counts are convenience fixtures. NumPy
is used only for independent quadrature, finite differences and assertions.
"""

import math

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.inference.neutra_training_mechanisms import (
    ComposedMechanism, FreeDiagonalAffine, MechanismReverseKLTrainer,
    ScalarSigmoidMixture, mechanism_from_state,
)
from bayesfilter.inference.neutra_weighted_training import WeightedDenseIAFTransport, WeightedNeuTraConfig


def scalar(dimension=2, coordinate=1, **kwargs):
    config = dict(dimension=dimension, coordinate=coordinate,
                  log_slopes=[math.log(0.8), math.log(1.2)],
                  offsets=[-0.4, 0.7], weight_logits=[-0.1, 0.1],
                  inverse_atol=1e-11, inverse_rtol=1e-12, inverse_max_iterations=64)
    config.update(kwargs)
    return ScalarSigmoidMixture(**config)


def gaussian(x):
    # Stop target autodiff to verify that the trainer only needs the supplied score.
    return tf.stop_gradient(-0.5 * tf.reduce_sum(x*x, axis=1)), tf.stop_gradient(-x), tf.ones(tf.shape(x)[0], tf.bool)


def quartic(x):
    return (tf.stop_gradient(-tf.reduce_sum(x**4, axis=1)/12),
            tf.stop_gradient(-x**3/3), tf.ones(tf.shape(x)[0], tf.bool))


def make_trainer(transport, estimator="standard", target=gaussian, batch=4):
    return MechanismReverseKLTrainer(transport, target, batch_size=batch,
                                     estimator=estimator, learning_rate=0.01)


def flat_gradients(result):
    return np.concatenate([v.numpy().reshape(-1) for v in result["gradients"]])


def test_affine_identity_scale_derivative_and_inverse():
    affine = FreeDiagonalAffine([0., 0.], [0., 0.])
    x = tf.constant([[-2., 1.], [1., 3.]], tf.float64)
    y, ld = affine.forward_and_logdet(x)
    np.testing.assert_array_equal(y, x)
    np.testing.assert_array_equal(ld, [0., 0.])
    affine.log_scale.assign([-6., 2.])  # Deliberately beyond the inherited +/-2 cap.
    with tf.GradientTape() as tape:
        y, ld = affine.forward_and_logdet(x)
        summed = tf.reduce_sum(ld)
    np.testing.assert_allclose(tape.gradient(summed, affine.log_scale), [2., 2.], atol=0, rtol=0)
    z, inverse_ld = affine.inverse_and_forward_logdet(y)
    np.testing.assert_allclose(z, x, atol=2e-15, rtol=2e-15)
    np.testing.assert_allclose(inverse_ld, ld, atol=0, rtol=0)


def test_scalar_logdet_and_force_against_value_finite_differences():
    flow = scalar(dimension=1, coordinate=0)
    x = tf.constant([-5., -1.1, 0.2, 1.7, 5.], tf.float64)
    # Central-difference O(h^2) error, float64 cancellation; no research threshold.
    h = tf.constant(2e-5, tf.float64)
    value, ld, score = flow.scalar_value_logdet_score(x)
    plus = flow.scalar_value_logdet_score(x+h)
    minus = flow.scalar_value_logdet_score(x-h)
    np.testing.assert_allclose((plus[0]-minus[0])/(2*h), tf.exp(ld), atol=2e-9, rtol=2e-9)
    np.testing.assert_allclose((plus[1]-minus[1])/(2*h), score, atol=2e-9, rtol=2e-9)
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x)
        value, ld, score = flow.scalar_value_logdet_score(x)
    np.testing.assert_allclose(tape.gradient(value, x), tf.exp(ld), atol=2e-14, rtol=2e-14)
    np.testing.assert_allclose(tape.gradient(ld, x), score, atol=2e-14, rtol=2e-14)


def test_scalar_xla_full_range_tail_roundtrip_and_failure_at_cap():
    flow = scalar()
    rows = tf.constant([[2., -1000.], [3., -30.], [4., 0.], [5., 30.], [6., 1000.]], tf.float64)

    @tf.function(input_signature=[tf.TensorSpec([5, 2], tf.float64)], jit_compile=True)
    def run(x):
        y, ld = flow.forward_and_logdet(x)
        z, inverse_ld, valid, count = flow.inverse_with_status(y)
        return y, ld, z, inverse_ld, valid, count

    y, ld, z, inverse_ld, valid, count = run(rows)
    assert bool(tf.reduce_all(valid)) and int(count) <= 64
    np.testing.assert_array_equal(y[:, 0], rows[:, 0])
    assert float(y[-1, 1]) > 100 and float(y[0, 1]) < -100
    np.testing.assert_allclose(z, rows, atol=3e-9, rtol=3e-12)
    np.testing.assert_allclose(inverse_ld, ld, atol=2e-11, rtol=2e-11)
    failed = scalar(inverse_max_iterations=1, inverse_atol=1e-15, inverse_rtol=0.)
    latent, logdet, success, _ = failed.inverse_with_status(tf.constant([[0., 2.13]], tf.float64))
    assert not bool(success[0]) and bool(tf.reduce_all(tf.math.is_nan(latent)))
    assert bool(tf.math.is_nan(logdet[0]))


def test_single_component_affine_and_distinct_components_shape_direction():
    one = scalar(dimension=1, coordinate=0, log_slopes=[math.log(1.7)], offsets=[-.3], weight_logits=[0.])
    x = tf.constant([-2., .2, 1.3, 4.], tf.float64)
    y, ld, score = one.scalar_value_logdet_score(x)
    np.testing.assert_allclose(y, 1.7*x-.3, atol=2e-15, rtol=2e-15)
    np.testing.assert_allclose(ld, math.log(1.7), atol=2e-15, rtol=2e-15)
    np.testing.assert_allclose(score, 0., atol=2e-15, rtol=0)
    distinct = scalar(dimension=1, coordinate=0)
    y = distinct.scalar_value_logdet_score(tf.constant([0., 1., 2.], tf.float64))[0]
    assert abs(float(y[0]-2*y[1]+y[2])) > 1e-4  # Constructed non-affine example.


def test_exact_identity_components_have_only_affine_initial_directions():
    flow = scalar(dimension=1, coordinate=0, log_slopes=[0., 0.], offsets=[0., 0.])
    for value in (-1.2, .3, 2.1):
        with tf.GradientTape() as tape:
            y = flow.scalar_value_logdet_score(tf.constant(value, tf.float64))[0]
        slope, bias, logits = tape.gradient(y, flow.trainable_variables)
        weights = tf.nn.softmax(flow.weight_logits)
        np.testing.assert_allclose(slope, weights*value, atol=2e-15, rtol=2e-15)
        np.testing.assert_allclose(bias, weights, atol=2e-15, rtol=2e-15)
        np.testing.assert_allclose(logits, 0., atol=2e-15, rtol=0)


def test_path_gradient_zero_pointwise_at_exact_gaussian_and_real_loss_reported():
    flow = FreeDiagonalAffine([0., 0.], [0., 0.])
    rows = tf.constant([[-2., 1.], [.3, 2.], [1.5, -.5], [2., 3.]], tf.float64)
    path = make_trainer(flow, "path")
    standard = make_trainer(flow)
    p, s = path.evaluate(rows), standard.evaluate(rows)
    assert bool(p["valid"])
    np.testing.assert_allclose(p["proposal_score"], -rows, atol=0, rtol=0)
    np.testing.assert_allclose(flat_gradients(p), 0., atol=0, rtol=0)
    assert np.linalg.norm(flat_gradients(s)) > 0.1  # Finite-batch gradients need not match.
    expected = float(tf.reduce_mean(tf.reduce_sum(rows**2, axis=1)))/2
    np.testing.assert_allclose(p["loss"], expected, atol=1e-14, rtol=0)
    np.testing.assert_allclose(s["loss"], expected, atol=1e-14, rtol=0)
    nodes = path.evaluate.get_concrete_function().graph.as_graph_def()
    operations = [n.op for n in nodes.node]
    operations += [n.op for f in nodes.library.function for n in f.node_def]
    assert not {"PyFunc", "EagerPyFunc"}.intersection(operations)
    assert not any("pfor" in f.signature.name.lower() for f in nodes.library.function)
    assert path.evaluate.experimental_get_tracing_count() == 1


def test_nonlinear_path_expectation_agrees_with_standard_and_finite_difference():
    flow = scalar(dimension=1, coordinate=0)
    standard = make_trainer(flow, target=quartic, batch=2)
    path = make_trainer(flow, "path", target=quartic, batch=2)

    def integrate(order):
        nodes, weights = np.polynomial.hermite.hermgauss(order)
        nodes, weights = math.sqrt(2)*nodes, weights/math.sqrt(math.pi)
        gs, gp = [], []
        # Independent reference quadrature: sample loop is confined to this test.
        for x in nodes:
            rows = tf.constant([[x], [x]], tf.float64)
            gs.append(flat_gradients(standard.evaluate(rows)))
            gp.append(flat_gradients(path.evaluate(rows)))
        return weights @ np.array(gs), weights @ np.array(gp), nodes, weights

    s32, p32, _, _ = integrate(32)
    s64, p64, nodes, weights = integrate(64)
    np.testing.assert_allclose(s32, s64, atol=1e-9, rtol=1e-9)
    np.testing.assert_allclose(p32, p64, atol=1e-9, rtol=1e-9)
    np.testing.assert_allclose(s64, p64, atol=1e-9, rtol=1e-9)
    rows = tf.constant(nodes[:, None], tf.float64)

    def objective():
        y, ld = flow.forward_and_logdet(rows)
        return float(np.dot(weights, (tf.reduce_sum(y**4, axis=1)/12-ld).numpy()))

    finite = []
    h = 2e-5
    for variable in flow.trainable_variables:
        saved = variable.numpy().copy()
        for index in range(len(saved)):
            changed = saved.copy(); changed[index] += h; variable.assign(changed); plus = objective()
            changed[index] -= 2*h; variable.assign(changed); minus = objective()
            finite.append((plus-minus)/(2*h)); variable.assign(saved)
    np.testing.assert_allclose(s64, finite, atol=2e-8, rtol=2e-8)


def test_composed_existing_iaf_proposal_score_and_freezing():
    base = WeightedDenseIAFTransport(WeightedNeuTraConfig(
        dimension=2, stages=2, hidden_layers=(4,), initialization_seed=(20260923, 17),
        initialization_scale=.1, jit_compile=True))
    base.stages[0].weights[-1].assign(tf.ones_like(base.stages[0].weights[-1])*.2)
    affine = FreeDiagonalAffine([.1, -.2], [.2, -.1])
    flow = ComposedMechanism(base, affine, train_first=False, train_second=True)
    rows = tf.constant([[-2., 1.], [.3, 2.], [1.5, -.5], [2., 3.]], tf.float64)
    result = make_trainer(flow, "path").evaluate(rows)
    physical, _ = flow.forward_and_logdet(rows)
    # Independent proposal score obtained by differentiating the existing inverse density.
    with tf.GradientTape() as tape:
        tape.watch(physical)
        z, ld = flow.inverse_and_forward_logdet(physical)
        log_q = -.5*tf.reduce_sum(z*z, axis=1)-ld
    expected = tape.gradient(log_q, physical)
    np.testing.assert_allclose(result["proposal_score"], expected, atol=2e-12, rtol=2e-12)
    assert tuple(id(v) for v in flow.trainable_variables) == tuple(id(v) for v in affine.trainable_variables)


@pytest.mark.parametrize("estimator", ["standard", "path"])
def test_scalar_training_is_batched_forward_only_and_rejects_invalid_updates(estimator):
    flow = scalar()
    # Training must not introduce numerical inversion into the forward RKL hot path.
    def forbidden(*args):
        raise AssertionError("inverse entered forward RKL")
    flow.inverse_and_forward_logdet = forbidden
    rows = tf.constant([[-1., -.8], [.1, 0.], [.5, .9], [1.2, 1.4]], tf.float64)
    trainer = make_trainer(flow, estimator)
    before = [v.numpy().copy() for v in flow.trainable_variables]
    for _ in range(3):  # Tiny mechanics smoke, not convergence or quality training.
        assert bool(trainer.train_step(rows)["valid"])
    assert int(trainer.optimizer.iterations) == 3
    assert any(not np.array_equal(a, v.numpy()) for a, v in zip(before, flow.trainable_variables))
    before = [v.numpy().copy() for v in flow.trainable_variables]
    result = trainer.train_step(tf.fill([4, 2], tf.constant(float("nan"), tf.float64)))
    assert not bool(result["valid"]) and int(result["iteration"]) == 3
    for a, v in zip(before, flow.trainable_variables):
        np.testing.assert_array_equal(a, v)


@pytest.mark.parametrize("flow", [FreeDiagonalAffine([.3, -.1], [.1, -.3]), scalar()])
def test_mechanism_state_roundtrip_is_not_hmc_admission(flow):
    state = flow.parameter_state()
    rebuilt = mechanism_from_state(state)
    rows = tf.constant([[.1, 1.], [2., -.3]], tf.float64)
    for a, b in zip(flow.forward_and_logdet(rows), rebuilt.forward_and_logdet(rows)):
        np.testing.assert_array_equal(a, b)
    with pytest.raises(ValueError, match="non-admitted"):
        mechanism_from_state({**state, "hmc_admitted": True})


def test_invalid_configuration_and_degenerate_parameters_fail():
    with pytest.raises(ValueError, match="batch size"):
        make_trainer(scalar(), batch=1)
    with pytest.raises(ValueError, match="inverse controls"):
        scalar(inverse_atol=0.)
    with pytest.raises(ValueError, match="equal shape"):
        scalar(log_slopes=[0.])
    flow = scalar(log_slopes=[-1000., 0.])
    result = make_trainer(flow).evaluate(tf.zeros([4, 2], tf.float64))
    assert not bool(result["valid"])


def test_inverse_does_not_accept_a_small_residual_in_a_flat_region():
    flow = scalar(dimension=1, coordinate=0, log_slopes=[0., 0.],
                  offsets=[-20., 20.], weight_logits=[0., 0.],
                  inverse_max_iterations=1, inverse_atol=1e-5, inverse_rtol=0.)
    # At y=0 the initial midpoint has exactly zero residual, but a broad bracket.
    # The declared coordinate accuracy must not be inferred from that residual alone.
    _, _, valid, _ = flow.inverse_with_status(tf.constant([[0.]], tf.float64))
    assert not bool(valid[0])


def test_update_rejects_finite_parameters_that_would_destroy_positive_scale():
    flow = FreeDiagonalAffine([0.], [0.])
    trainer = MechanismReverseKLTrainer(flow, gaussian, batch_size=2,
                                        estimator="standard", learning_rate=1000.)
    result = trainer.train_step(tf.zeros([2, 1], tf.float64))
    assert not bool(result["valid"]) and int(result["iteration"]) == 0
    np.testing.assert_array_equal(flow.log_scale, [0.])
    flow.log_scale.assign([-1000.])
    assert not bool(trainer.evaluate(tf.zeros([2, 1], tf.float64))["valid"])


def test_finite_target_status_veto_does_not_update_parameters():
    def veto(x):
        value, score, _ = gaussian(x)
        return value, score, tf.zeros(tf.shape(x)[0], tf.bool)
    flow = FreeDiagonalAffine([1., -1.], [0., 0.])
    result = make_trainer(flow, target=veto).train_step(tf.ones([4, 2], tf.float64))
    assert not bool(result["valid"]) and int(result["iteration"]) == 0
    np.testing.assert_array_equal(flow.shift, [1., -1.])
