"""KSC mixture adapter and independent diagnostic quadrature reference.

The candidate calls the shared canonical model/executor. The grid and
moment-matched Kalman routines are references, never candidate filtering lanes.
"""
from __future__ import annotations

from dataclasses import dataclass
import functools
import math
import tensorflow as tf

WEIGHTS = (.00730, .10556, .00002, .04395, .34001, .24566, .25750)
# Original binary64 centered constants; avoid a module-import numerical loop.
MEANS = (-11.40039, -5.2432099999999995, -9.83726, 1.50746, -0.65098, 0.52478, -2.35859)
VARIANCES = (5.79596, 2.61369, 5.17950, .16735, .64009, .34023, 1.26261)
DTYPE = tf.float64


class ReferenceConvergenceError(ValueError):
    """A scientific continuation veto, not an infrastructure retry."""

    def __init__(self, record):
        self.record = record
        super().__init__('KSC mixture reference did not meet its convergence contract')


def persistence(theta):
    return .5 * (1. + tf.math.erf(theta[0] / tf.sqrt(tf.constant(2., DTYPE))))


def observation_terms(theta, nodes, observation):
    residual = observation - 2. * theta[1] - nodes[:, None] - tf.constant(MEANS, DTYPE)
    variance = tf.constant(VARIANCES, DTYPE)
    terms = tf.math.log(tf.constant(WEIGHTS, DTYPE)) - .5 * (
        tf.square(residual) / variance + tf.math.log(variance) + math.log(2. * math.pi))
    return tf.reduce_logsumexp(terms, axis=1), 2. * tf.reduce_sum(tf.nn.softmax(terms, axis=1) * residual / variance, axis=1)


@dataclass(frozen=True)
class KSCSpec:
    dimension: int = 1
    parameter_count: int = 2
    family: str = 'ksc_mixture'
    target_id: str = 'sqmc_ksc_seven_mixture_q1_prior1_predict_first_v1'
    prepared_data_regime: str = 'ksc_seven_mixture_predict_first_independent_pairs_v1'

    def default_theta(self, dtype=DTYPE):
        return tf.constant([1.5, 0.], dtype)

    def model(self, theta, direction=None):
        from bayesfilter.highdim.ledh_canonical_models_tf import ksc_sv_canonical_model
        model, setter = ksc_sv_canonical_model(theta)
        setter(tf.zeros_like(theta) if direction is None else direction)
        return model, setter

    def initial_cloud(self, theta, normals, direction=None):
        n = tf.shape(normals)[0]
        return normals, tf.ones([n, 1, 1], theta.dtype), tf.zeros_like(normals), tf.zeros([n, 1, 1], theta.dtype)

    def simulate(self, theta, horizon, seed, *, jit_compile=True):
        return simulator(horizon, jit_compile)(theta, tf.constant(seed, tf.int32))

    def reference_value_and_score(self, theta, observations):
        # Host cache stores diagnostic outputs only; never feeds the particle kernel.
        key = (tuple(theta.numpy().tolist()), tuple(x[0] for x in observations.numpy().tolist()))
        record = checked_reference(*key)
        return tf.constant(record['value'], DTYPE), tf.constant(record['score'], DTYPE)


@functools.lru_cache(maxsize=12)
def simulator(horizon, jit_compile):
    @tf.function(input_signature=[tf.TensorSpec([2], DTYPE), tf.TensorSpec([], tf.int32)],
                 jit_compile=jit_compile, autograph=False)
    def generate(theta, seed):
        normal = tf.random.stateless_normal([2 * horizon + 1], [seed, 519], dtype=DTYPE)
        component = tf.random.stateless_categorical(tf.math.log(tf.constant([WEIGHTS], DTYPE)),
                                                   horizon, [seed, 521])[0]
        noise = tf.gather(tf.constant(MEANS, DTYPE), component) + tf.sqrt(
            tf.gather(tf.constant(VARIANCES, DTYPE), component)) * normal[1:horizon + 1]
        rows = tf.TensorArray(DTYPE, size=horizon)
        def body(t, state, rows):
            state = persistence(theta) * state + normal[horizon + 1 + t]
            return t + 1, state, rows.write(t, state + 2. * theta[1] + noise[t])
        return tf.while_loop(lambda t, *_: t < horizon, body, (0, normal[0], rows))[2].stack()[:, None]
    return generate


@functools.lru_cache(maxsize=16)
def grid_reference(nodes=1201, bound=48., jit_compile=False):
    """Independent analytic density/score recursion on a fixed trapezoidal grid."""
    @tf.function(input_signature=[tf.TensorSpec([2], DTYPE), tf.TensorSpec([None, 1], DTYPE)],
                 jit_compile=jit_compile, autograph=False)
    def compute(theta, observations):
        x = tf.linspace(tf.constant(-bound, DTYPE), tf.constant(bound, DTYPE), nodes)
        weights = tf.concat([tf.constant([.5], DTYPE), tf.ones([nodes-2], DTYPE), tf.constant([.5], DTYPE)], 0) * (2. * bound / (nodes - 1))
        gamma = persistence(theta)
        dgamma = tf.exp(-.5 * tf.square(theta[0])) / math.sqrt(2. * math.pi)
        residual = x[:, None] - gamma * x[None, :]
        kernel = tf.exp(-.5 * tf.square(residual)) / math.sqrt(2. * math.pi)
        dkernel = kernel * residual * x[None, :] * dgamma
        variance = 1. + gamma * gamma
        initial = tf.exp(-.5 * tf.square(x) / variance) / tf.sqrt(2. * math.pi * variance)
        dinitial = initial * .5 * (tf.square(x) / tf.square(variance) - 1. / variance) * 2. * gamma * dgamma
        def body(t, density, derivative, value, score):
            def predict():
                predicted = tf.linalg.matvec(kernel, density * weights)
                dpredicted = tf.matmul(kernel, derivative * weights[:, None])
                dpredicted += tf.stack([tf.linalg.matvec(dkernel, density * weights), tf.zeros([nodes], DTYPE)], 1)
                return predicted, dpredicted
            predicted, dpredicted = tf.cond(t == 0, lambda: (initial, tf.stack([dinitial, tf.zeros([nodes], DTYPE)], 1)), predict)
            logobs, dlogbeta = observation_terms(theta, x, observations[t, 0])
            shift = tf.reduce_max(logobs)
            obs = tf.exp(logobs - shift)
            unnormalized = predicted * obs
            dun = dpredicted * obs[:, None] + tf.stack([tf.zeros([nodes], DTYPE), unnormalized * dlogbeta], 1)
            normalizer = tf.reduce_sum(unnormalized * weights)
            increment = tf.reduce_sum(dun * weights[:, None], axis=0) / normalizer
            posterior = unnormalized / normalizer
            dposterior = dun / normalizer - posterior[:, None] * increment
            return t + 1, posterior, dposterior, value + tf.math.log(normalizer) + shift, score + increment
        result = tf.while_loop(lambda t, *_: t < tf.shape(observations)[0], body,
                              (0, initial, tf.zeros([nodes, 2], DTYPE), tf.constant(0., DTYPE), tf.zeros([2], DTYPE)))
        return result[3], result[4]
    return compute


@functools.lru_cache(maxsize=128)
def checked_reference(theta_values, observation_values):
    theta = tf.constant(theta_values, DTYPE)
    observations = tf.constant(observation_values, DTYPE)[:, None]
    records = []
    # Explicit CPU reference exception; no GPU default/readiness claim.
    with tf.device('/CPU:0'):
        for nodes, bound in ((401,40.), (801,40.), (1201,40.), (1201,48.)):
            value, score = grid_reference(nodes, bound)(theta, observations)
            records.append(dict(nodes=nodes, bound=bound, value=float(value.numpy()), score=score.numpy().tolist()))
    last = records[-1]
    delta_value = max(abs(last['value']-r['value']) for r in records[-3:-1])
    delta_score = max(abs(a-b) for r in records[-3:-1] for a,b in zip(last['score'], r['score']))
    if not all(math.isfinite(x) for r in records for x in [r['value'],*r['score']]) or delta_value > 1e-6 or delta_score > 1e-5:
        safe = lambda x: x if math.isfinite(x) else str(x)
        raise ReferenceConvergenceError(dict(
            convergence=[dict(r, value=safe(r['value']), score=[safe(x) for x in r['score']]) for r in records],
            maximum_value_difference=safe(delta_value), maximum_score_difference=safe(delta_score)))
    return dict(value=last['value'], score=last['score'], reference='converged_mixture_grid_approximation',
                convergence=records, maximum_value_difference=delta_value, maximum_score_difference=delta_score)


@functools.lru_cache(maxsize=8)
def enumeration_reference(horizon):
    """T<=2 exact Gaussian-mixture enumeration; autodiff is diagnostic only."""
    if horizon not in (1,2):
        raise ValueError('bounded exact enumeration supports T=1,2')
    @tf.function(input_signature=[tf.TensorSpec([2], DTYPE), tf.TensorSpec([horizon, 1], DTYPE)],
                 jit_compile=False, autograph=False)
    def compute(theta, observations):
        with tf.GradientTape() as tape:
            tape.watch(theta)
            mean, variance, logweight = tf.zeros([1], DTYPE), tf.ones([1], DTYPE), tf.zeros([1], DTYPE)
            for t in range(horizon):
                pm, pv = persistence(theta) * mean, tf.square(persistence(theta)) * variance + 1.
                innovation = observations[t,0] - 2. * theta[1] - pm[:,None] - tf.constant(MEANS, DTYPE)
                s = pv[:,None] + tf.constant(VARIANCES, DTYPE)
                logweight = tf.reshape(logweight[:,None] + tf.math.log(tf.constant(WEIGHTS, DTYPE)) - .5 * (
                    tf.square(innovation)/s + tf.math.log(s) + math.log(2.*math.pi)), [-1])
                mean = tf.reshape(pm[:,None] + pv[:,None]/s * innovation, [-1])
                variance = tf.reshape(pv[:,None] * tf.constant(VARIANCES, DTYPE)/s, [-1])
            value = tf.reduce_logsumexp(logweight)
        return value, tape.gradient(value, theta)
    return compute


@tf.function(input_signature=[tf.TensorSpec([2], DTYPE), tf.TensorSpec([None, 1], DTYPE)],
             jit_compile=False, autograph=False)
def gaussian_approximation_reference(theta, observations):
    """Moment-matched Kalman comparator, WRONG as an exact mixture oracle."""
    mean_e = tf.reduce_sum(tf.constant(WEIGHTS, DTYPE) * tf.constant(MEANS, DTYPE))
    variance_e = tf.reduce_sum(tf.constant(WEIGHTS, DTYPE) * (tf.constant(VARIANCES, DTYPE) + tf.square(tf.constant(MEANS, DTYPE)))) - mean_e**2
    with tf.GradientTape() as tape:
        tape.watch(theta)
        def body(t, mean, variance, value):
            pm, pv = persistence(theta)*mean, tf.square(persistence(theta))*variance+1.
            residual = observations[t,0]-pm-2.*theta[1]-mean_e
            s = pv+variance_e
            return t+1, pm+pv/s*residual, pv*variance_e/s, value-.5*(residual**2/s+tf.math.log(s)+math.log(2.*math.pi))
        value = tf.while_loop(lambda t,*_: t<tf.shape(observations)[0],body,
                              (0,tf.constant(0.,DTYPE),tf.constant(1.,DTYPE),tf.constant(0.,DTYPE)))[3]
    return value,tape.gradient(value,theta)
