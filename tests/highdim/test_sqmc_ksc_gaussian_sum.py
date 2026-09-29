"""CPU-only diagnostics for the independent full-mixture Kalman reference."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import pytest
import tensorflow as tf
from bayesfilter.highdim.sqmc_ksc_tf import DTYPE, enumeration_reference, grid_reference
from bayesfilter.highdim.sqmc_ksc_gaussian_sum_reference_tf import gaussian_sum_reference


@pytest.mark.parametrize('theta_value', [[1.5, 0.], [.1, -.4], [-1., .3]])
@pytest.mark.parametrize('horizon', [1, 2])
def test_all_seven_components_agree_with_exact_enumeration(theta_value, horizon):
    theta = tf.constant(theta_value, DTYPE)
    obs = tf.constant([[-4.], [1.]][:horizon], DTYPE)
    actual = gaussian_sum_reference(401, 40., False)(theta, obs)
    exact = enumeration_reference(horizon)(theta, obs)
    assert bool(actual['valid'])
    assert int(actual['observation_components']) == 7
    assert int(actual['maximum_gaussian_branches']) == (7 if horizon == 1 else 2807)
    tf.debugging.assert_near(actual['value'], exact[0], atol=1e-9, rtol=1e-9)
    tf.debugging.assert_near(actual['score'], exact[1], atol=1e-9, rtol=1e-9)


@pytest.mark.parametrize('nodes,bound', [(401, 40.), (31, 4.)])
def test_total_score_includes_projection_normalization(nodes, bound):
    theta = tf.constant([.7, -.2], DTYPE)
    obs = tf.constant([[-4.], [1.], [-.6], [2.]], DTYPE)
    kernel = gaussian_sum_reference(nodes, bound, False)
    actual = kernel(theta, obs)
    assert bool(actual['valid'])
    for i in (0, 1):
        step = tf.one_hot(i, 2, dtype=DTYPE)*1e-5
        fd = (kernel(theta+step, obs)['value']-kernel(theta-step, obs)['value'])/2e-5
        tf.debugging.assert_near(actual['score'][i], fd, atol=2e-8, rtol=2e-8)
    if nodes == 31:
        assert float(actual['maximum_projection_mass_error']) > 1e-4
    else:
        reference = grid_reference(nodes, bound)(theta, obs)
        tf.debugging.assert_near(actual['value'], reference[0], atol=1e-9, rtol=1e-9)
        tf.debugging.assert_near(actual['score'], reference[1], atol=1e-9, rtol=1e-9)
