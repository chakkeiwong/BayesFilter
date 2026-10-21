"""Independent integrals check finite-reference normalization and decisions."""
import tensorflow as tf
from bayesfilter.testing.surrogate_hmc_lg1_finite_reference import normalize_grid, compare_references


def test_uniform_density_integrals_and_cdf_with_large_log_offset():
    phi = tf.linspace(tf.constant(-1.,tf.float64),tf.constant(1.,tf.float64),1025)
    density, values, logz = normalize_grid(phi,tf.fill([1025],tf.constant(1000.,tf.float64)),
        .5+.1*phi,tf.constant([0.,1.,-.8,0.,.8],tf.float64))
    tf.debugging.assert_near(density,tf.fill([1025],tf.constant(.5,tf.float64)),atol=1e-12)
    tf.debugging.assert_near(values,tf.constant([0.,1/3,.1,.5,.9,.5],tf.float64),atol=1e-6)
    tf.debugging.assert_near(logz,1000+tf.math.log(tf.constant(2.,tf.float64)),atol=1e-12)


def test_reference_uncertainty_and_qualification_can_prevent_agreement():
    zero=tf.zeros([6],tf.float64); margin=tf.ones([6],tf.float64)*.1
    assert compare_references(zero,zero,zero,zero,margin,True)["status"]=="EQUIVALENT"
    assert compare_references(zero,margin,zero,zero,margin,True)["status"]=="INCONCLUSIVE"
    assert compare_references(zero,zero,zero,zero,margin,False)["status"]=="INCONCLUSIVE"
    assert compare_references(zero+1,zero,zero,zero,margin,True)["status"]=="DISCREPANCY"
