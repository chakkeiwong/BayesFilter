"""Analytic CPU/XLA reference checks for the bounded q20 correction optimizer."""
import math

import pytest
import tensorflow as tf

from bayesfilter.inference.neutra_training_mechanisms import (
    AdamMechanismCanary, FreeDiagonalAffine,
)


def target(x):
    return -.5 * tf.reduce_sum(x*x, axis=1), -x, tf.ones([x.shape[0]], tf.bool)


def trainer(flow, callback=target, rate=.01):
    return AdamMechanismCanary(flow, callback, batch_size=4, estimator="standard",
        learning_rate=rate, beta1=.9, beta2=.999, epsilon=1e-7,
        gradient_clip_norm=1e6, jit_compile=True)


ROWS = [[-1., .2], [.4, -.5], [1.2, .7], [-.2, 1.8]]


def test_first_adam_step_matches_bias_corrected_formula():
    flow = FreeDiagonalAffine([.2, -.1], [.1, -.2])
    fit = trainer(flow)
    z = tf.constant(ROWS, tf.float64)
    initial = [tf.identity(v) for v in flow.trainable_variables]
    gradients = fit.evaluate(z)["gradients"]
    result = fit.train_step(z)
    assert bool(result["valid"])
    assert int(result["iteration"]) == 1
    for old, grad, actual in zip(initial, gradients, flow.trainable_variables):
        # Installed Keras casts Python beta constants through float32 for powers,
        # while the moment increments use Python (1-beta) at the slot dtype.
        # Reproduce that checked backend arithmetic, including epsilon placement.
        alpha = tf.cast(fit.optimizer.learning_rate, tf.float64) * tf.sqrt(
            1-tf.cast(.999, tf.float64)) / (1-tf.cast(.9, tf.float64))
        expected = old - alpha*((1-.9)*grad)/(tf.sqrt((1-.999)*grad*grad)+1e-7)
        tf.debugging.assert_near(actual, expected, atol=1e-12, rtol=1e-12)


def test_finite_status_veto_preserves_parameters_and_all_slots():
    def veto(x):
        v, s, ok = target(x)
        return v, s, tf.zeros_like(ok)
    flow = FreeDiagonalAffine([.2, -.1], [.1, -.2])
    fit = trainer(flow, veto)
    state = flow.trainable_variables + tuple(fit.optimizer.variables)
    before = [tf.identity(v) for v in state]
    result = fit.train_step(tf.constant(ROWS, tf.float64))
    assert not bool(result["valid"])
    for value, saved in zip(state, before):
        tf.debugging.assert_equal(value, saved)


def test_post_update_constraint_failure_rolls_back_adam_slots_and_iteration():
    flow = FreeDiagonalAffine([.2, -.1], [.1, -.2])
    original = tf.identity(flow.shift)
    flow.parameter_constraints[id(flow.shift)] = lambda v: tf.reduce_all(v == original)
    fit = trainer(flow)
    state = flow.trainable_variables + tuple(fit.optimizer.variables)
    before = [tf.identity(v) for v in state]
    result = fit.train_step(tf.constant(ROWS, tf.float64))
    assert not bool(result["valid"])
    assert int(result["iteration"]) == 0
    for value, saved in zip(state, before):
        tf.debugging.assert_equal(value, saved)


def test_slot_and_parameter_restore_matches_uninterrupted_updates():
    first = trainer(FreeDiagonalAffine([.2, -.1], [.1, -.2]))
    second = trainer(FreeDiagonalAffine([.2, -.1], [.1, -.2]))
    z = tf.constant(ROWS, tf.float64)
    first.train_step(z)
    state_a = first.variables + tuple(first.optimizer.variables)
    state_b = second.variables + tuple(second.optimizer.variables)
    for a, b in zip(state_a, state_b):
        b.assign(a)
    first.train_step(z*.8)
    second.train_step(z*.8)
    for a, b in zip(state_a, state_b):
        tf.debugging.assert_equal(a, b)


@pytest.mark.parametrize("control,value", [("beta1", 1.), ("epsilon", 0.),
                                            ("gradient_clip_norm", 0.)])
def test_invalid_explicit_adam_controls_rejected(control, value):
    options = dict(batch_size=4, estimator="standard", learning_rate=.01,
                   beta1=.9, beta2=.999, epsilon=1e-7, gradient_clip_norm=10.)
    options[control] = value
    with pytest.raises(ValueError):
        AdamMechanismCanary(FreeDiagonalAffine([0., 0.], [0., 0.]), target, **options)
