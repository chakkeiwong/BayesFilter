"""Independent checks of log arithmetic and total finite-quadrature derivatives."""
import math

import pytest
import tensorflow as tf

from bayesfilter.testing.surrogate_hmc_nonlinear_log_reference import log_weight, make_log_reference
from bayesfilter.testing.surrogate_hmc_nonlinear_quadrature import make_reference

OBSERVATIONS = tf.constant([[.8160498738288879], [-.5775660872459412]], tf.float64)


def test_zero_weight_has_zero_mass_and_finite_derivative():
    weight = tf.constant([0., .25, .75], tf.float64)
    with tf.GradientTape() as tape:
        tape.watch(weight)
        result = tf.reduce_logsumexp(log_weight(weight))
    assert float(result) == pytest.approx(0., abs=1e-14)
    assert tape.gradient(result, weight).numpy().tolist() == pytest.approx([0., 1., 1.])


@pytest.mark.parametrize("point", [[0., 0.], [-.4, .3], [.7, -.5]])
def test_log_sum_matches_old_quadrature_and_total_derivative(point):
    z = tf.constant(point, tf.float64)
    old = make_reference(12, 64)(z, OBSERVATIONS)
    new = make_log_reference(12, 64)(z, OBSERVATIONS)
    assert bool(old['valid']) and bool(new['valid'])
    assert float(new['log_likelihood']) == pytest.approx(float(old['log_likelihood']), abs=1e-9)
    assert new['score'].numpy().tolist() == pytest.approx(old['score'].numpy().tolist(), rel=1e-9, abs=1e-9)
    old_values = make_reference(12, 64, value_only=True)(z, OBSERVATIONS)
    assert new['predictive'].numpy().tolist() == pytest.approx(
        (old_values['predictive_integrals']/old_values['likelihood']).numpy().tolist(), abs=1e-9)


def test_log_reference_recovers_finite_potential_after_density_underflow():
    z = tf.constant([-6., -6.], tf.float64)
    old = make_reference(12, 64, value_only=True)(z, OBSERVATIONS)
    new = make_log_reference(12, 64)(z, OBSERVATIONS)
    assert float(old['likelihood']) == 0.
    assert bool(new['valid']) and math.isfinite(float(new['log_likelihood']))
    assert float(new['log_likelihood']) < -745.


def test_log_reference_gradient_matches_finite_differences_at_two_steps():
    point = [.23, -.17]
    kernel = make_log_reference(12, 64)
    score = kernel(tf.constant(point, tf.float64), OBSERVATIONS)['score'].numpy().tolist()
    for step in (1e-4, 5e-5):
        for axis in (0, 1):
            plus, minus = list(point), list(point)
            plus[axis] += step
            minus[axis] -= step
            difference = (float(kernel(tf.constant(plus, tf.float64), OBSERVATIONS)['log_likelihood'])
                          -float(kernel(tf.constant(minus, tf.float64), OBSERVATIONS)['log_likelihood']))/(2*step)
            assert difference == pytest.approx(score[axis], rel=2e-5, abs=2e-5)
