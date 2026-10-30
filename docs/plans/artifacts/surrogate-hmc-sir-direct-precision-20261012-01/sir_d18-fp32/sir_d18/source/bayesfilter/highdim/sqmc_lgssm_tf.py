"""Shared SQMC LGSSM definitions and analytical parameter callbacks.

All families start at the raw initial law and transition before EVERY
observation, including the first. P44 q/r entries are variances; diagonal-AR
q/r parameters are standard deviations. No model is inferred from dimension.
"""
from __future__ import annotations

import functools
import math
from dataclasses import dataclass

import tensorflow as tf

from bayesfilter.highdim.ledh_canonical_score_tf import NonlinearScoreModel


@dataclass(frozen=True)
class LGSSMSpec:
    family: str
    dimension: int

    def __post_init__(self):
        if self.family not in {"p44", "diagonal_ar", "frozen_3d"}:
            raise ValueError("unknown LGSSM family")
        if self.dimension < 1 or (self.family == "frozen_3d" and self.dimension != 3):
            raise ValueError("invalid LGSSM dimension")

    @property
    def parameter_count(self):
        return 4 if self.family == "p44" else self.dimension + 2

    @property
    def target_id(self):
        return f"sqmc_lgssm_{self.family}_d{self.dimension}_predict_first_v2"

    def default_theta(self, dtype=tf.float64):
        if self.family == "p44":
            values = [0.25, math.log(0.18), math.log(0.12), 0.04]
        elif self.family == "frozen_3d":
            values = [0.9, 0.8, 0.7, 0.6, 0.8]
        else:
            return _diagonal_default(self.dimension, tf.as_dtype(dtype).name)()
        return tf.constant(values, dtype)

    def parts(self, theta, direction=None):
        """Physical diagonal parameters and their exact directional derivatives."""
        theta = tf.ensure_shape(tf.convert_to_tensor(theta), [self.parameter_count])
        dtype, d = theta.dtype, self.dimension
        v = tf.zeros_like(theta) if direction is None else tf.cast(direction, dtype)
        ones, zeros = tf.ones([d], dtype), tf.zeros([d], dtype)
        if self.family == "p44":
            def padded(first, rest):
                return tf.concat([tf.constant(first, dtype),
                                  tf.fill([max(0, d-3)], tf.constant(rest, dtype))], 0)[:d]
            a = padded([1., .85, .70], .55)
            q = tf.exp(theta[1]) * padded([.9, 1.1, 1.3], 1.)
            r = tf.exp(theta[2]) * padded([1., 1.2, .8], 1.)
            mscale = padded([1., -.5, .25], 0.)
            phi = .55 * tf.tanh(theta[0]) * a
            dphi = .55 * (1. - tf.square(tf.tanh(theta[0]))) * a * v[0]
            m, dm = theta[3] * mscale, v[3] * mscale
            p = padded([.6, .8, 1.], 1.)
            dq, dr = q * v[1], r * v[2]
        else:
            phi, dphi = theta[:d], v[:d]
            q, r = tf.square(theta[d]) * ones, tf.square(theta[d + 1]) * ones
            dq, dr = 2. * theta[d] * v[d] * ones, 2. * theta[d + 1] * v[d + 1] * ones
            m, dm, p = zeros, zeros, ones
        h = (tf.constant([[1., .25, -.15], [.2, 1.1, .3], [-.1, .35, .9]], dtype)
             if self.family == "frozen_3d" else tf.eye(d, dtype=dtype))
        return dict(phi=phi, q=q, r=r, m=m, p=p, h=h,
                    dphi=dphi, dq=dq, dr=dr, dm=dm, dp=zeros)

    def kalman_parameters(self, theta):
        p = self.parts(theta)
        return dict(transition_matrix=tf.linalg.diag(p['phi']),
                    process_covariance=tf.linalg.diag(p['q']),
                    observation_matrix=p['h'], observation_covariance=tf.linalg.diag(p['r']),
                    initial_mean=p['m'], initial_covariance=tf.linalg.diag(p['p']))

    def initial_cloud(self, theta, normals, direction=None):
        p = self.parts(theta, direction)
        states = p['m'] + normals * tf.sqrt(p['p'])
        tangents = p['dm'] + normals * p['dp'] / (2. * tf.sqrt(p['p']))
        shape = [tf.shape(normals)[0], self.dimension, self.dimension]
        return (states, tf.broadcast_to(tf.linalg.diag(p['p']), shape), tangents,
                tf.broadcast_to(tf.linalg.diag(p['dp']), shape))

    def model(self, theta, direction=None):
        """Analytical callbacks; setter retained for scalar diagnostic consumers."""
        theta = tf.convert_to_tensor(theta)
        active = [tf.zeros_like(theta) if direction is None else direction]
        fixed = self.parts(theta)

        def set_direction(value):
            active[0] = tf.cast(value, theta.dtype)

        def physical(value):
            return self.parts(value, active[0])

        def gaussian(residual, variance):
            return -.5 * tf.reduce_sum(tf.square(residual) / variance + tf.math.log(variance)
                                       + tf.constant(math.log(2. * math.pi), theta.dtype), axis=-1)

        def gaussian_tangent(residual, dr, variance, dv):
            return tf.reduce_sum(-residual * dr / variance
                                 + .5 * (tf.square(residual) / tf.square(variance) - 1. / variance) * dv,
                                 axis=-1)

        def transition(value, points):
            return points * physical(value)['phi']

        def transition_tangent(value, points, dpoints):
            p = physical(value)
            return dpoints * p['phi'] + points * p['dphi']

        def observation(points):
            return tf.linalg.matmul(points, fixed['h'], transpose_b=True)

        def transition_density(value, points, means):
            return gaussian(points - means, physical(value)['q'])

        def transition_density_tangent(value, points, means, dpoints, dmeans):
            p = physical(value)
            return gaussian_tangent(points - means, dpoints - dmeans, p['q'], p['dq'])

        def observation_density(value, points, obs):
            return gaussian(obs - observation(points), physical(value)['r'])

        def observation_density_tangent(value, points, obs, dpoints):
            p = physical(value)
            return gaussian_tangent(obs - observation(points), -observation(dpoints), p['r'], p['dr'])

        model = NonlinearScoreModel(
            transition_mean_fn=transition, transition_mean_tangent_fn=transition_tangent,
            observation_fn=observation,
            observation_jacobian_fn=lambda points: tf.broadcast_to(fixed['h'], [tf.shape(points)[0], self.dimension, self.dimension]),
            observation_tangent_fn=lambda points, dpoints: observation(dpoints),
            process_covariance=tf.linalg.diag(fixed['q']), observation_covariance=tf.linalg.diag(fixed['r']),
            process_covariance_tangent_fn=lambda value: tf.linalg.diag(physical(value)['dq']),
            observation_covariance_tangent_fn=lambda value: tf.linalg.diag(physical(value)['dr']),
            transition_log_density_fn=transition_density, transition_log_density_tangent_fn=transition_density_tangent,
            observation_log_density_fn=observation_density, observation_log_density_tangent_fn=observation_density_tangent)
        return model, set_direction

    def simulate(self, theta, horizon, seed, *, jit_compile=True):
        """Generate matched data using a cached graph with a stable signature."""
        theta=tf.convert_to_tensor(theta)
        return _simulator(self,horizon,theta.dtype.name,jit_compile)(theta,tf.convert_to_tensor(seed,tf.int32))

    def _simulate_tf(self, theta, horizon, seed):
        """TensorFlow generation from the same raw prior and predict-first law."""
        theta = tf.convert_to_tensor(theta)
        p = self.parts(theta)
        noise = tf.random.stateless_normal([2 * horizon + 1, self.dimension], [seed, 719], dtype=theta.dtype)
        state = p['m'] + tf.sqrt(p['p']) * noise[0]
        rows = tf.TensorArray(theta.dtype, size=horizon)
        def body(t, state, rows):
            state = p['phi'] * state + tf.sqrt(p['q']) * noise[2 * t + 1]
            obs = tf.linalg.matvec(p['h'], state) + tf.sqrt(p['r']) * noise[2 * t + 2]
            return t + 1, state, rows.write(t, obs)
        return tf.while_loop(lambda t, *_: t < horizon, body, (0, state, rows))[2].stack()

    def reference_value_and_score(self, theta, observations):
        """Independent Kalman diagnostic; autodiff is confined to this oracle."""
        theta = tf.convert_to_tensor(theta)
        observations = tf.convert_to_tensor(observations, dtype=theta.dtype)
        # Explicit CPU/non-XLA oracle exception: independent reference only.
        with tf.device('/CPU:0'):
            return _reference_kernel(self, int(observations.shape[0]), theta.dtype.name)(theta, observations)


@functools.lru_cache(maxsize=32)
def _diagonal_default(dimension, dtype_name):
    @tf.function(input_signature=[], jit_compile=True, autograph=False)
    def prepare():
        values = tf.maximum(tf.constant(.5, tf.float64),
                            .95-.05*tf.cast(tf.range(dimension), tf.float64))
        return tf.cast(tf.concat([values, tf.constant([.6, .8], tf.float64)], 0), dtype_name)
    return prepare


@functools.lru_cache(maxsize=32)
def _simulator(spec,horizon,dtype_name,jit_compile):
    dtype=tf.as_dtype(dtype_name)
    @tf.function(input_signature=[tf.TensorSpec([spec.parameter_count],dtype),tf.TensorSpec([],tf.int32)],
                 jit_compile=jit_compile,autograph=False)
    def generate(theta,seed):
        return spec._simulate_tf(theta,horizon,seed)
    return generate


@functools.lru_cache(maxsize=32)
def _reference_kernel(spec, horizon, dtype_name):
    from bayesfilter.highdim.ledh_kalman_oracle_tf import kalman_filter_marginal_loglik
    dtype = tf.as_dtype(dtype_name)

    @tf.function(input_signature=[tf.TensorSpec([spec.parameter_count], dtype),
                                  tf.TensorSpec([horizon, spec.dimension], dtype)],
                 jit_compile=False)
    def reference(theta, observations):
        with tf.GradientTape() as tape:
            tape.watch(theta)
            value = kalman_filter_marginal_loglik(observations=observations,
                                                **spec.kalman_parameters(theta))
        return value, tape.gradient(value, theta)
    return reference
