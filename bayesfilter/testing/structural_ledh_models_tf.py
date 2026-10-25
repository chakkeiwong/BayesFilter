"""Batch-native analytical laws for the structural Model A/B campaign.

The transition has one Gaussian innovation and two state coordinates. No
ambient transition density or artificial noise on the deterministic coordinate
is introduced. Model definitions match nonlinear_models_tf.py, but this module
also supplies every parameter sensitivity used by the shared LEDH recurrence.
"""
from dataclasses import dataclass
import math

import tensorflow as tf

from bayesfilter.highdim.analytical_jet_tf import Jet, exp, log, stack


def tanh(x):
    value = tf.math.tanh(x.v)
    return Jet(value, (1 - value * value) * x.t)


@dataclass(frozen=True)
class StructuralModelSpec:
    name: str
    dimension: int = 2
    observation_dimension: int = 1
    innovation_dimension: int = 1

    def __post_init__(self):
        if self.name not in ("model_a", "model_b"):
            raise ValueError("Unknown structural model: " + self.name)

    @property
    def parameter_count(self):
        return 2 if self.name == "model_a" else 3

    def structural_model(self, theta):
        return StructuralModelLaw(self, theta)

    def truth(self, dtype=tf.float32):
        q = ([math.log(.25), math.log(.15)] if self.name == "model_a" else
             [math.atanh(.70), math.log(.25), math.log(.80)])
        return tf.constant(q, dtype)

    def prior_parameters(self, dtype=tf.float32):
        # Declared synthetic regularizing priors, directly on the q chart.
        # They are not asserted to be literature priors for these toy models.
        mean, scale = (([math.log(.4), math.log(.3)], [.7, .7])
                       if self.name == "model_a" else
                       ([.5, math.log(.4), math.log(.7)], [.8, .7, .6]))
        return tf.constant(mean, dtype), tf.constant(scale, dtype)

    def physical(self, theta):
        return (tf.exp(theta) if self.name == "model_a" else
                tf.stack([tf.tanh(theta[..., 0]), tf.exp(theta[..., 1]),
                          tf.exp(theta[..., 2])], -1))


class StructuralModelLaw:
    """Leading B rows, with all P analytical tangents in a single recurrence."""
    def __init__(self, spec, theta):
        self.spec = spec
        self.batch, self.p = theta.shape
        self.q = Jet(theta, tf.broadcast_to(tf.eye(self.p, dtype=theta.dtype)[:, None],
                                           [self.p, self.batch, self.p]))
        self.sigma = exp(self.q[:, 0 if spec.name == "model_a" else 1])
        if spec.name == "model_a":
            self.R = exp(2 * self.q[:, 1, None, None])
        else:
            self.rho = tanh(self.q[:, 0])
            self.beta = exp(self.q[:, 2])
            self.R = self.q.constant(tf.fill([self.batch, 1, 1],
                                             tf.constant(.09, theta.dtype)))

    def _coefficient(self, coefficient, rank):
        return coefficient.reshape([self.batch] + [1] * (rank - 1))

    def complete(self, parent, innovation):
        old_m, old_k, e = parent[..., 0], parent[..., 1], innovation[..., 0]
        rank = max(old_m.v.shape.rank, e.v.shape.rank)
        sigma = self._coefficient(self.sigma, rank)
        if self.spec.name == "model_a":
            m = .35 * old_m - .10 * old_k + sigma * e
            k = old_m + 0 * e
        else:
            rho = self._coefficient(self.rho, rank)
            beta = self._coefficient(self.beta, rank)
            m = rho * old_m + sigma * e
            k = .55 * old_k + beta * tanh(m)
        return stack([m, k], -1)

    def observe(self, state):
        y = state[..., 0]
        if self.spec.name == "model_b":
            y = y + state[..., 1]
        return y[..., None]

    def innovation_observation_jacobian(self, parent, innovation):
        state = self.complete(parent, innovation)
        m = state[..., 0]
        sigma = self._coefficient(self.sigma, m.v.shape.rank)
        derivative = sigma + m * 0
        if self.spec.name == "model_b":
            beta = self._coefficient(self.beta, m.v.shape.rank)
            derivative = derivative * (1 + beta * (1 - tanh(m) ** 2))
        return derivative[..., None, None]

    def initial_cloud(self, normals):
        scale = [1., 1.] if self.spec.name == "model_a" else [.5, math.sqrt(.20)]
        initial = normals[None] * tf.constant(scale, normals.dtype)
        return self.q.constant(initial).broadcast([self.batch, *normals.shape])

    def observation_density(self, state, observation):
        residual = self.observe(state)[..., 0] - observation[0]
        variance = self.R[:, 0, 0, None]
        return -.5 * (math.log(2 * math.pi) + log(variance) + residual ** 2 / variance)


def make_structural_data(spec, horizon, seed, dtype=tf.float32):
    """CPU-friendly fixed synthetic data, predicting before each observation."""
    initial_scale = [1., 1.] if spec.name == "model_a" else [.5, math.sqrt(.20)]
    initial = tf.random.stateless_normal([2], [seed, 1], dtype=dtype) * tf.constant(initial_scale, dtype)
    noise = tf.random.stateless_normal([horizon, 2], [seed, 2], dtype=dtype)
    states = tf.TensorArray(dtype, horizon, element_shape=[2])
    observations = tf.TensorArray(dtype, horizon, element_shape=[1])

    def step(t, state, states, observations):
        if spec.name == "model_a":
            m = .35 * state[0] - .10 * state[1] + .25 * noise[t, 0]
            child = tf.stack([m, state[0]])
            y = m + .15 * noise[t, 1]
        else:
            m = .70 * state[0] + .25 * noise[t, 0]
            k = .55 * state[1] + .80 * tf.tanh(m)
            child = tf.stack([m, k])
            y = m + k + .30 * noise[t, 1]
        return t + 1, child, states.write(t, child), observations.write(t, y[None])

    _, _, states, observations = tf.while_loop(
        lambda t, *_: t < horizon, step, (0, initial, states, observations),
        parallel_iterations=1)
    return {"initial_state": initial, "states": states.stack(),
            "observations": observations.stack(), "truth": spec.truth(dtype)}


def make_structural_design(n, horizon, seed, dtype=tf.float32):
    return {"initial": tf.random.stateless_normal([n, 2], [seed, 11], dtype=dtype),
            "innovations": tf.random.stateless_normal([horizon, n, 1], [seed, 12], dtype=dtype),
            "uniforms": tf.random.stateless_uniform([horizon, n, 2], [seed, 13], dtype=dtype),
            "beta": tf.constant([.2, .4, .4], dtype)}


def make_model_a_kalman(batch_size, horizon, dtype=tf.float32, jit_compile=True):
    """Exact Model A value and analytical total score; singular Q is allowed."""
    from bayesfilter.highdim.analytical_jet_tf import mm, mv
    signature = [tf.TensorSpec([batch_size, 2], dtype),
                 tf.TensorSpec([horizon, 1], dtype)]

    @tf.function(input_signature=signature, jit_compile=jit_compile, autograph=False)
    def program(theta, observations):
        model = StructuralModelLaw(StructuralModelSpec("model_a"), theta)
        c = model.q.constant
        f = c(tf.broadcast_to(tf.constant([[.35, -.10], [1., 0.]], dtype),
                               [batch_size, 2, 2]))
        q = exp(2 * model.q[:, 0, None, None]) * c(tf.constant([[1., 0.], [0., 0.]], dtype))
        mu = c(tf.zeros([batch_size, 2], dtype))
        cov = c(tf.eye(2, batch_shape=[batch_size], dtype=dtype))
        ell = c(tf.zeros([batch_size], dtype))

        def step(t, muv, mut, pv, pt, lv, lt):
            forecast = mv(f, Jet(muv, mut))
            p = mm(mm(f, Jet(pv, pt)), f.transpose()) + q
            s = p[:, 0, 0] + model.R[:, 0, 0]
            residual = observations[t, 0] - forecast[:, 0]
            inc = -.5 * (math.log(2 * math.pi) + log(s) + residual ** 2 / s)
            gain = p[:, :, 0] / s[:, None]
            updated = forecast + gain * residual[:, None]
            covariance = p - gain[:, :, None] * p[:, None, 0, :]
            covariance = .5 * (covariance + covariance.transpose())
            total = Jet(lv, lt) + inc
            return t + 1, updated.v, updated.t, covariance.v, covariance.t, total.v, total.t

        out = tf.while_loop(lambda t, *_: t < horizon, step,
                            (0, mu.v, mu.t, cov.v, cov.t, ell.v, ell.t),
                            parallel_iterations=1)
        return {"value": out[5], "score": tf.transpose(out[6]),
                "mean": out[1], "covariance": out[3]}

    return program
