"""Independent-reference checks for the M10 composite-quadrature repair."""
import tensorflow as tf

from bayesfilter.testing.surrogate_hmc_nonlinear_quadrature import make_reference, quadrature_rule
from bayesfilter.testing.surrogate_hmc_nonlinear_reference import grid_reference


def test_quadrature_rules_integrate_polynomial_moments():
    for normal, expected in ((True,3.),(False,2/5)):
        nodes,weights = quadrature_rule(12,normal=normal)
        assert abs(float(tf.reduce_sum(weights*nodes**4))-expected)<1e-12
        assert abs(float(tf.reduce_sum(weights*nodes**3)))<1e-12


def test_composite_integral_and_total_score_match_independent_state_grid():
    # These are the preserved calibration observations rounded to FP32 once.
    observations = tf.constant([[.5424982309341431],[2.675233840942383]],tf.float64)
    calculate = make_reference(20,128)
    grid = grid_reference(513,9.)
    for point in ((0.,0.),(.6,.2),(-.6,-.6)):
        z = tf.constant(point,tf.float64)
        result,reference = calculate(z,observations),grid(z,observations)
        assert bool(result['valid'])
        assert abs(float(result['log_likelihood']-reference['log_likelihood']))<1e-6
        assert float(tf.reduce_max(tf.abs(result['score']-reference['score'])))<1e-5
    z = tf.constant([.1,-.2],tf.float64)
    score = calculate(z,observations)['score']
    for index in range(2):
        offset = tf.one_hot(index,2,dtype=tf.float64)*1e-5
        difference = (calculate(z+offset,observations)['log_likelihood']-
                      calculate(z-offset,observations)['log_likelihood'])/(2e-5)
        assert abs(float(score[index]-difference))<1e-6
