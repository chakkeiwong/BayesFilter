"""CPU reference checks for actual-consumer numerical observability."""
import pytest
import tensorflow as tf

from bayesfilter.score_study.fitted_twist_tf import make_fitted_twist_kernel, twisted_transition


@pytest.mark.parametrize("constant", [False, True])
@pytest.mark.parametrize("fisher,controls", [(False, False), (True, False), (True, True)])
def test_trace_preserves_values_scores_and_actual_resampling(constant, fisher, controls):
    dtype = tf.float64
    d, n, horizon = 2, 24, 3
    theta = tf.constant([.6, -.4, -.3, 1., .1, -.2], dtype)
    normal = lambda shape, key: tf.random.stateless_normal(shape, [93700000, key], dtype=dtype)
    # Values lie on the specified binary grid, also for resampling controls.
    uniform = lambda shape, key: tf.cast(tf.random.stateless_uniform(
        shape, [93700000, key], minval=0, maxval=2**23, dtype=tf.int32), dtype)/2**23
    arguments = (theta, normal([horizon, d], 1), normal([n, d], 2),
        normal([horizon, n, d], 3), uniform([horizon+1, n], 4),
        uniform([horizon, n], 5), normal([horizon, d], 6),
        tf.eye(d, batch_shape=[horizon], dtype=dtype), tf.fill([horizon], tf.constant(-3., dtype)))
    options = dict(dtype_name="float64", jit_compile=False, constant_twist=constant,
        include_fisher_score=fisher, include_resampling_controls=controls,
        resampling_uniform_bits=23 if controls else None)
    plain = make_fitted_twist_kernel(d, d, n, horizon, **options)(*arguments)
    observed = make_fitted_twist_kernel(d, d, n, horizon,
        include_numerical_trace=True, **options)(*arguments)
    assert len(observed) == len(plain)+1
    for expected, actual in zip(plain, observed[:-1]):
        tf.debugging.assert_equal(expected, actual)
    trace = observed[-1]
    reconstructed = tf.minimum(tf.searchsorted(trace["ancestor_cdf"], arguments[4], side="right"), n-1)
    tf.debugging.assert_equal(reconstructed, trace["ancestor_indices"])
    assert trace["ancestor_indices"].shape == (horizon+1, n)
    assert trace["gaussian_probability"].shape == (horizon, n)
    tf.debugging.assert_greater_equal(trace["gaussian_probability"], tf.constant(0., dtype))
    tf.debugging.assert_less_equal(trace["gaussian_probability"], tf.constant(1., dtype))


def test_compiled_transition_reads_each_time_center():
    """The GPU/TF32 version catches reuse of c[0] in a compiled loop."""
    n, horizon, d = 64, 5, 2
    centers = tf.constant([[0.,0.],[1.,-1.],[-2.,2.],[3.,-3.],[-4.,4.]])
    means = tf.random.stateless_normal([n,d], [93700001,1])*.3
    q = tf.constant([[.36,0.],[0.,.468]])
    v = tf.constant([[.42,.02],[.02,.82]])
    signature = [tf.TensorSpec([horizon,d],tf.float32), tf.TensorSpec([n,d],tf.float32),
                 tf.TensorSpec([d,d],tf.float32), tf.TensorSpec([d,d],tf.float32)]

    @tf.function(input_signature=signature, jit_compile=True)
    def kernel(cs, m, Q, V):
        cloud = tf.TensorArray(tf.float32, size=horizon, element_shape=[n,d])
        def step(t, cloud):
            x, _ = twisted_transition(m,tf.zeros([6,n,d]),Q,tf.zeros([6,d,d]),
                cs[t],V,tf.ones([n]),tf.zeros([n,d]),tf.zeros([n]))
            return t+1,cloud.write(t,x)
        result = tf.while_loop(lambda t,*_: t<horizon,step,(0,cloud),parallel_iterations=1)
        return result[1].stack()

    previous = tf.config.experimental.tensor_float_32_execution_enabled()
    try:
        tf.config.experimental.enable_tensor_float_32_execution(True)
        actual = kernel(centers,means,q,v)
    finally:
        tf.config.experimental.enable_tensor_float_32_execution(previous)
    # Independent precision-form Gaussian product in FP64.
    qi = tf.linalg.inv(tf.cast(q,tf.float64)); vi = tf.linalg.inv(tf.cast(v,tf.float64))
    covariance = tf.linalg.inv(qi+vi)
    rhs = tf.cast(means,tf.float64)@qi
    expected = (rhs[None,:,:]+(tf.cast(centers,tf.float64)@vi)[:,None,:])@covariance
    tf.debugging.assert_near(tf.cast(actual,tf.float64),expected,atol=.003,rtol=0.)
