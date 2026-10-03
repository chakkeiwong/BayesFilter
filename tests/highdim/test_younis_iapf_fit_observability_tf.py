"""Independent identities for explanatory iAPF fit outputs; CPU reference tests."""
import pytest
import tensorflow as tf

from bayesfilter.score_study.iapf_fit_tf import bounded_density_fit


FIT = dict(mean_bound=4., sd_lower=.2, sd_upper=4., max_steps=0,
           max_backtracks=30, tolerance=1e-7, floor_ratio=.01)
POINTS = [[-2., .3], [-.7, 1.4], [.2, -.6], [.8, -.9], [1.1, .5], [2., -.2]]


def test_squared_target_concentration_and_amplitude_invariance():
    dtype = tf.float64
    x = tf.constant(POINTS, dtype)

    @tf.function(input_signature=[tf.TensorSpec([6], dtype)], jit_compile=True)
    def fit(logb):
        return bounded_density_fit(x, logb, **FIT)[-1]

    uniform = fit(tf.fill([6], tf.constant(-1.e6, dtype)))
    concentrated = fit(tf.constant([1.e6, 1.e6-1000., 1.e6-1000.,
                                   1.e6-1000., 1.e6-1000., 1.e6-1000.], dtype))
    tf.debugging.assert_near(uniform['target_squared_effective_count'], tf.constant(6., dtype))
    tf.debugging.assert_near(uniform['target_squared_max_weight'], tf.constant(1./6., dtype))
    tf.debugging.assert_equal(concentrated['target_squared_effective_count'], tf.constant(1., dtype))
    tf.debugging.assert_equal(concentrated['target_squared_max_weight'], tf.constant(1., dtype))
    logb = tf.constant([0., -.3, -.7, -1.1, -.1, -2.], dtype)
    ordinary, shifted = fit(logb), fit(logb+500.)
    for key in ('target_squared_effective_count', 'target_squared_max_weight'):
        tf.debugging.assert_near(ordinary[key], shifted[key], atol=1e-10, rtol=0.)
    assert fit.experimental_get_tracing_count() == 1


@pytest.mark.parametrize('objective,scale', [('density_l2','native'),
    ('density_l2','initial_peak'), ('relative_shape','native')])
def test_energy_and_initial_objective_are_distinct_and_consistent(objective, scale):
    dtype = tf.float64
    x = tf.constant(POINTS, dtype)
    logb = tf.constant([0., -.3, -.7, -1.1, -.1, -2.], dtype)
    z = (x-tf.reduce_mean(x, axis=0))/tf.math.reduce_std(x, axis=0)
    # Cloud-moment initialization has standardized mean zero and variance one.
    p = tf.exp(-.5*tf.reduce_sum(z*z, axis=1))
    b = tf.exp(logb)
    lam = tf.reduce_sum(p*b)/tf.reduce_sum(b*b)
    energy = tf.reduce_mean(p*p)
    residual = tf.reduce_mean((p-lam*b)**2)
    *_, info = bounded_density_fit(x, logb, **FIT, objective=objective, objective_scale=scale)
    expected = (residual/energy if objective == 'relative_shape' else
                residual*tf.exp(-2*info['objective_log_density_scale']))
    tf.debugging.assert_near(tf.exp(info['initial_log_density_energy']), energy, atol=1e-12)
    tf.debugging.assert_near(info['log_density_energy'], info['initial_log_density_energy'], atol=1e-12)
    tf.debugging.assert_near(info['initial_optimization_loss'], expected, atol=1e-12)
    tf.debugging.assert_near(info['optimization_loss'], expected, atol=1e-12)


def test_log_energy_remains_visible_when_native_energy_underflows():
    dtype = tf.float64
    x = tf.stack([-tf.ones([1024], dtype), tf.ones([1024], dtype)])
    *_, info = bounded_density_fit(x, tf.constant([0., -1.], dtype), **FIT)
    assert bool(info['objective_underflow']) and not bool(info['valid'])
    tf.debugging.assert_equal(info['scaled_loss'], tf.constant(0., dtype))
    tf.debugging.assert_near(info['initial_log_density_energy'], tf.constant(-1024., dtype))
    tf.debugging.assert_near(info['log_density_energy'], tf.constant(-1024., dtype))
