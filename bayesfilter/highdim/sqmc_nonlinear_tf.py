"""Nonlinear scopes for the shared analytical LEDH/SQMC campaign evaluator.

No filtering or moment-repair algorithm is reimplemented here. Synthetic data
use the same canonical transition and observation law, with a transition before
each observation. This explicit timing differs from some historical fixtures.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import tensorflow as tf

from bayesfilter.highdim.ledh_canonical_models_tf import (
    austria_sir_canonical_model,
    predator_prey_canonical_model,
)


@dataclass(frozen=True)
class NonlinearSQMCSpec:
    name: str

    def __post_init__(self):
        if self.name not in ("predator_prey", "sir_d18"):
            raise ValueError(f"unknown nonlinear scope: {self.name}")

    @property
    def dimension(self):
        return 2 if self.name == "predator_prey" else 18

    @property
    def observation_dimension(self):
        return 2 if self.name == "predator_prey" else 9

    @property
    def parameter_names(self):
        if self.name == "predator_prey":
            return ("r", "carrying_capacity", "half_saturation", "s", "u", "v")
        return ("log_kappa_scale", "log_nu_scale", "log_observation_noise_scale")

    @property
    def parameter_count(self):
        return len(self.parameter_names)

    @property
    def target_id(self):
        return f"canonical_{self.name}_x0_then_transition_observe_v1"

    def default_theta(self, dtype=tf.float32):
        return tf.constant([.6, 114., 25., .3, .5, .5] if self.dimension == 2
                           else [0., 0., 0.], dtype)

    def initial_mean(self, dtype):
        if self.dimension == 2:
            return tf.constant([50., 5.], dtype)
        from bayesfilter.highdim.models import zhao_cui_sir_austria_model
        with tf.init_scope():
            mean = zhao_cui_sir_austria_model().initial_mean
        return tf.cast(mean, dtype)

    def model(self, theta, direction):
        factory = predator_prey_canonical_model if self.dimension == 2 else austria_sir_canonical_model
        model, set_direction = factory(theta, dtype=theta.dtype)
        set_direction(direction)
        return model, set_direction

    def initial_cloud(self, theta, normals, direction=None):
        del direction  # The initial N(mean, I) law has no parameter dependence.
        states = self.initial_mean(theta.dtype) + normals
        covariance = tf.broadcast_to(tf.eye(self.dimension, dtype=theta.dtype),
                                     [tf.shape(normals)[0], self.dimension, self.dimension])
        return states, covariance, tf.zeros_like(states), tf.zeros_like(covariance)

    def observations(self, horizon, seed, *, dtype=tf.float32, jit_compile=True):
        """Fresh data at the declared physical truth; no historical result reuse."""
        if horizon < 1:
            raise ValueError("horizon must be positive")
        return _simulation(self, horizon, dtype.name, jit_compile)(
            self.default_theta(dtype), tf.constant(seed, tf.int32))


@lru_cache(maxsize=16)
def _simulation(spec, horizon, dtype_name, jit_compile):
    dtype = tf.as_dtype(dtype_name)
    mean = spec.initial_mean(dtype)

    @tf.function(input_signature=[tf.TensorSpec([spec.parameter_count], dtype),
                                  tf.TensorSpec([], tf.int32)],
                 jit_compile=jit_compile, autograph=False)
    def simulate(theta, seed):
        model, _ = spec.model(theta, tf.zeros_like(theta))
        q = tf.linalg.cholesky(model.process_covariance)
        r = tf.linalg.cholesky(model.observation_covariance)
        state = mean + tf.random.stateless_normal([spec.dimension], [seed, 0], dtype=dtype)
        output = tf.TensorArray(dtype, size=horizon)

        def step(t, state, output):
            process_noise = tf.random.stateless_normal([spec.dimension], [seed, 2*t+1], dtype=dtype)
            observation_noise = tf.random.stateless_normal([spec.observation_dimension], [seed, 2*t+2], dtype=dtype)
            state = model.transition_mean_fn(theta, state[None, :])[0] + tf.linalg.matvec(q, process_noise)
            observed = model.observation_fn(state[None, :])[0] + tf.linalg.matvec(r, observation_noise)
            return t+1, state, output.write(t, observed)

        _, _, output = tf.while_loop(lambda t, *_: t < horizon, step,
                                     (tf.constant(0), state, output))
        return output.stack()

    return simulate


def trace_kernel(spec, route, controls, n, horizon, dtype, *, jit_compile=True,
                 reset_design_kind="normal_quantiles"):
    """Retain diagnostics from the actual shared reset, using direction zero."""
    from bayesfilter.highdim import sqmc_campaign_tf as common
    from bayesfilter.highdim.ledh_canonical_score_tf import canonical_value_and_analytical_score
    design = common.reset_design(n, spec.dimension, dtype, reset_design_kind)
    settings = dict(common.numerical_settings(controls), **common.route_settings(route))
    signature = [tf.TensorSpec([spec.parameter_count], dtype),
                 tf.TensorSpec([n, spec.dimension], dtype),
                 tf.TensorSpec([horizon, n, spec.dimension], dtype),
                 tf.TensorSpec([horizon, n], dtype),
                 tf.TensorSpec([horizon, spec.observation_dimension], dtype)]

    @tf.function(input_signature=signature, jit_compile=jit_compile, autograph=False)
    def compute(theta, initial, noise, uniforms, observations):
        direction = tf.one_hot(0, spec.parameter_count, dtype=dtype)
        model, _ = spec.model(theta, direction)
        states, covs, ds, dc = spec.initial_cloud(theta, initial, direction)
        value, score, trace = canonical_value_and_analytical_score(
            model, theta, states, covs, noise, observations, with_score=True,
            return_trace=True, initial_state_tangent=ds, initial_covariance_tangent=dc,
            reset_design=design, process_ancestor_uniforms=uniforms, **settings)
        records = tuple({key: val for key, val in row.items()
                         if (tf.is_tensor(val) and val.shape.rank == 0)
                         or (key.startswith("higher_moment_") and "stage_" not in key)}
                        for row in trace)
        return value, score, records

    return compute
