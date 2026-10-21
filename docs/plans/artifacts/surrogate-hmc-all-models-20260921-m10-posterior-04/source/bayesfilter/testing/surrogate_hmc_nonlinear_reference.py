"""Independent M10 likelihood-reference preparation, not posterior admission.

The source scalar sine/quadratic law is retained, including its initialization
and TensorFlow constant conversions. This diagnostic wrapper tests log-noise
parameters; it neither implements nor substitutes for the full DCTR consumer.
"""
from functools import lru_cache
import math

import tensorflow as tf

from bayesfilter.score_study.gaussian_tf import parameterized_model
from bayesfilter.score_study.nonlinear_tf import make_data_kernel, make_grid_reference

DTYPE = tf.float64
BASE_THETA = (.62, -.8, -.6, .9, .25, -.3)
TRANSITION_CURVE, OBSERVATION_CURVE = .2, .12
PRIOR_SD = math.log(2.)
PROBES = tuple((q, r) for q in (-PRIOR_SD, 0., PRIOR_SD) for r in (-PRIOR_SD, 0., PRIOR_SD))
GRID_LADDER = ((129, 9.), (257, 9.), (513, 9.), (769, 13.5))
PARTITION_SEEDS = {'calibration': 14100, 'validation': 14200, 'claim': 14300}


def source_theta(log_scales):
    return tf.constant(BASE_THETA, DTYPE) + tf.stack(
        [tf.constant(0., DTYPE), log_scales[0], log_scales[1],
         tf.constant(0., DTYPE), tf.constant(0., DTYPE), tf.constant(0., DTYPE)])


@tf.function(input_signature=[tf.TensorSpec([2], DTYPE)], jit_compile=True, autograph=False)
def log_prior(log_scales):
    variance = tf.constant(PRIOR_SD**2, DTYPE)
    return (-tf.reduce_sum(log_scales**2)/(2*variance)-tf.math.log(2*math.pi*variance),
            -log_scales/variance)


def generate_dataset(seed, *, stress=False):
    scales = tf.constant([-PRIOR_SD, -PRIOR_SD] if stress else [0., 0.], DTYPE)
    return tf.cast(make_data_kernel(2, TRANSITION_CURVE, OBSERVATION_CURVE)(
        source_theta(scales), tf.constant(seed, tf.int32)), tf.float32)


@lru_cache(maxsize=8)
def grid_reference(points, radius, c=TRANSITION_CURVE, b=OBSERVATION_CURVE):
    source = make_grid_reference(2, points, radius, c, b, 'float64', True)

    @tf.function(input_signature=[tf.TensorSpec([2], DTYPE), tf.TensorSpec([2, 1], DTYPE)],
                 jit_compile=True, autograph=False)
    def calculate(log_scales, observations):
        value, score, mean, variance, domain_loss, edge = source(source_theta(log_scales), observations)
        return dict(log_likelihood=value, score=score[1:3], final_mean=mean,
                    final_variance=variance, domain_loss=domain_loss, posterior_edge=edge)
    return calculate


@tf.function(input_signature=[tf.TensorSpec([2], DTYPE), tf.TensorSpec([2, 1], DTYPE)],
             jit_compile=True, autograph=False)
def linear_kalman_log_likelihood(log_scales, observations):
    """Independent scalar recursion; only source parameter interpretation is shared."""
    a, _, h, _, m, _, p, _, q, _, r, _ = parameterized_model(source_theta(log_scales), 1, 1)
    a, h, m, p, q, r = a[0, 0], h[0, 0], m[0], p[0, 0], q[0, 0], r[0, 0]
    value = tf.constant(0., DTYPE)
    for index in range(2):
        m, p = a*m, a*a*p+q
        error, innovation = observations[index, 0]-h*m, h*h*p+r
        value -= .5*(tf.constant(math.log(2*math.pi), DTYPE)+tf.math.log(innovation)+error*error/innovation)
        gain = p*h/innovation
        m, p = m+gain*error, p-gain*h*p
    return value


@lru_cache(maxsize=4)
def direct_innovation_reference(order):
    """Separate Gaussian integration of x0, epsilon1 and epsilon2.

    The one-off CPU eigendecomposition constructs the normal quadrature rule;
    the repeated integration uses a fixed TensorFlow/XLA graph and no density
    derivative from the state-grid recursion.
    """
    if order not in (32, 64, 128):
        raise ValueError('unplanned independent quadrature order')
    with tf.device('/CPU:0'):
        off = tf.sqrt(tf.range(1, order, dtype=DTYPE))
        jacobi = tf.linalg.diag(off, k=1, num_rows=order, num_cols=order)
        nodes, vectors = tf.linalg.eigh(jacobi+tf.transpose(jacobi))
        weights = tf.square(vectors[0])
        tf.debugging.assert_near(tf.reduce_sum(weights), tf.constant(1., DTYPE), atol=1e-12)
        tf.debugging.assert_near(tf.reduce_sum(weights*nodes**2), tf.constant(1., DTYPE), atol=1e-12)
    log_weights = tf.math.log(weights)

    @tf.function(input_signature=[tf.TensorSpec([2], DTYPE), tf.TensorSpec([2, 1], DTYPE)],
                 jit_compile=True, autograph=False)
    def calculate(log_scales, observations):
        a, _, h, _, m, _, p, _, q, _, r, _ = parameterized_model(source_theta(log_scales), 1, 1)
        a, h, m, p, sq, r = a[0, 0], h[0, 0], m[0], p[0, 0], tf.sqrt(q[0, 0]), r[0, 0]
        c, b = tf.cast(TRANSITION_CURVE, DTYPE), tf.cast(OBSERVATION_CURVE, DTYPE)
        normalizer = tf.constant(math.log(2*math.pi), DTYPE)+tf.math.log(r)

        def step(index, total):
            x0 = m+tf.sqrt(p)*nodes[index]
            x1 = a*x0+c*tf.sin(x0)+sq*nodes
            x2 = a*x1[:, None]+c*tf.sin(x1[:, None])+sq*nodes[None, :]
            density1 = -.5*(normalizer+(observations[0, 0]-h*x1-b*x1*x1)**2/r)
            density2 = -.5*(normalizer+(observations[1, 0]-h*x2-b*x2*x2)**2/r)
            term = log_weights[index]+tf.reduce_logsumexp(density1[:, None]+density2
                + log_weights[:, None]+log_weights[None, :])
            return index+1, tf.reduce_logsumexp(tf.stack([total, term]))

        _, result = tf.while_loop(lambda index, *_: index < order, step,
            (tf.constant(0), tf.constant(float('-inf'), DTYPE)), parallel_iterations=1)
        return result
    return calculate
