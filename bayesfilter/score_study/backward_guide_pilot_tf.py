"""Diagnostic Gaussian transfer of a replicated controlled-SMC backward fit.

This reference scope issues no canonical tuning or route identity. It uses the
complete shared LEDH consumer and analytical parameter directions. The mixture
conversion is a one-component representation check, not trained KDM.
"""

import math

import tensorflow as tf

from bayesfilter.highdim.ledh_backward_guide_tf import (
    FrozenMixtureBackwardGuide,
    FrozenQuadraticBackwardGuide,
    build_backward_guided_ledh_kernel,
)
from bayesfilter.highdim.transport_chunk_policy import select_transport_chunk_size
from bayesfilter.replication.controlled_smc_tf import ScalarGaussianControlledSMC
from .canonical_adapter_tf import gaussian_direction_inputs
from .control_diagnostics_tf import compact_control_diagnostics
from .direction_assembly_tf import make_direction_kernel


REFERENCE_CONTROLS = {
    "flow_substeps": 24,
    "reset_policy": "contract_e",
    "reset_epsilon": 1.0,
    "reset_sinkhorn_steps": 5,
    "reset_balance_steps": 32,
    "reset_ridge": 1e-6,
    "correction_steps": 1,
    "correction_strength": 0.2,
    "correction_lm_damping": 0.01,
    "correction_lm_scale_floor": 1e-4,
    "correction_trust_radius": 0.5,
    "pairwise_steps": 1,
    "pairwise_strength": 0.02,
    "pairwise_rms_cap": 2.0,
    "coordinate_cap": 0.94,
    "coordinate_cap_power": 8.0,
    "coordinate_cap_identity_radius": 0.0,
    "moment_safety": False,
}


def reference_theta(alpha):
    return tf.constant([
        alpha, math.log(0.3), math.log(0.5), 1.0 / 1.15,
        0.0, -0.5 * math.log(1.15),
    ], tf.float64)


def make_scalar_fit(horizon, particles, alpha, *, jit_compile=True):
    def log_density(observation, state):
        return -0.5 * (math.log(2.0 * math.pi * 0.25)
                       + (observation - state)**2 / 0.25)

    engine = ScalarGaussianControlledSMC(
        observations=horizon, particles=particles, alpha=alpha,
        process_variance=0.09, initial_mean=0.0,
        initial_variance=alpha**2 + 0.09,
        observation_log_density=log_density,
        precision_policy="source_scalar_projection", jit_compile=jit_compile,
    )

    @tf.function(input_signature=[
        tf.TensorSpec([horizon], tf.float64),
        tf.TensorSpec([horizon, particles], tf.float64),
        tf.TensorSpec([horizon], tf.float64),
    ], jit_compile=jit_compile, autograph=False)
    def fit(observations, normals, uniforms):
        policy = tf.zeros([horizon, 3], tf.float64)
        forward = engine.forward(policy, observations, normals, uniforms)
        result = engine.fit(policy, forward["x"], forward["log_incremental_weight"])
        guide = FrozenQuadraticBackwardGuide.from_scalar_controlled_smc_future(
            result["policy"], alpha, 0.09)
        return {**result, "valid": result["valid"] & forward["valid"],
                "precision": guide.precision, "linear": guide.linear,
                "pilot_ess": forward["ess"]}

    return fit


def make_exact_future_information(horizon, alpha, *, jit_compile=True):
    @tf.function(input_signature=[tf.TensorSpec([horizon], tf.float64)],
                 jit_compile=jit_compile, autograph=False)
    def exact(observations):
        precisions = tf.TensorArray(tf.float64, size=horizon)
        vectors = tf.TensorArray(tf.float64, size=horizon)

        def step(time, precision, vector, precisions, vectors):
            precisions = precisions.write(time, precision)
            vectors = vectors.write(time, vector)
            posterior_precision = precision + 4.0
            posterior_vector = vector + 4.0 * observations[time]
            conditional_variance = 1.0 / (1.0 / 0.09 + posterior_precision)
            future_precision = alpha**2 * posterior_precision * conditional_variance / 0.09
            future_vector = alpha * conditional_variance * posterior_vector / 0.09
            return time - 1, future_precision, future_vector, precisions, vectors

        result = tf.while_loop(
            lambda time, *_: time >= 0, step,
            (horizon - 1, tf.constant(0.0, tf.float64), tf.constant(0.0, tf.float64),
             precisions, vectors), parallel_iterations=1)
        return result[3].stack()[:, None, None], result[4].stack()[:, None]

    return exact


def single_component_representation(guide):
    active = guide.precision[:, 0, 0] > 0
    safe_precision = tf.where(active, guide.precision[:, 0, 0], 1.0)
    return FrozenMixtureBackwardGuide(
        tf.zeros([guide.precision.shape[0], 1], guide.precision.dtype),
        (guide.linear[:, 0] / safe_precision)[:, None, None],
        (1.0 / safe_precision)[:, None, None, None], active)


def _score_error_diagnostics(trace):
    stacked = tf.nest.map_structure(lambda *values: tf.stack(values), *trace)
    weights = stacked["posterior_weights"]
    d_weights = stacked["d_posterior_weights"]
    result = {
        "value_increment": tf.reduce_logsumexp(stacked["posterior_logits"], axis=1),
        "score_increment": tf.reduce_sum(weights * stacked["d_posterior_logits"], axis=1),
        "prior_log_mass": stacked["prior_observation_log_normalizer"],
        "d_prior_log_mass": stacked["d_prior_observation_log_normalizer"],
    }

    def moments(name, points, tangents, probabilities, d_probabilities):
        mean = tf.reduce_sum(probabilities[..., None] * points, axis=1)
        d_mean = tf.reduce_sum(d_probabilities[..., None] * points
                               + probabilities[..., None] * tangents, axis=1)
        centered = points - mean[:, None, :]
        variance = tf.reduce_sum(probabilities[..., None] * centered ** 2, axis=1)
        d_variance = tf.reduce_sum(d_probabilities[..., None] * centered ** 2
            + 2 * probabilities[..., None] * centered * (tangents - d_mean[:, None, :]), axis=1)
        result.update({name + "_mean": mean, name + "_d_mean": d_mean,
                       name + "_variance": variance, name + "_d_variance": d_variance})

    moments("weighted_children", stacked["children"], stacked["d_children"], weights, d_weights)
    uniform = tf.ones_like(weights) / tf.cast(tf.shape(weights)[1], weights.dtype)
    zero = tf.zeros_like(weights)
    for stage in ("raw_reset", "pre_cap", "post_cap"):
        key = "higher_moment_stage_" + stage
        moments(stage, stacked[key], stacked[key + "_tangent"][..., 0], uniform, zero)
    moments("final_reset", stacked["states_after_reset"], stacked["d_states_after_reset"],
            uniform, zero)
    return result


def make_guided_reference_kernel(particles, horizon, controls=None, *,
                                jit_compile=True, all_directions=True,
                                error_diagnostics=False):
    if particles % 2 or select_transport_chunk_size(particles) != particles:
        raise ValueError("This scalar reference requires an even one-block particle count")
    settings = dict(REFERENCE_CONTROLS if controls is None else controls)
    reset_design = tf.tile(tf.constant([[-1.0], [1.0]], tf.float64), [particles // 2, 1])

    def model_factory(theta, direction):
        return gaussian_direction_inputs(
            theta, direction, tf.zeros([particles, 1], tf.float64), 1, 1)[0]

    shared = build_backward_guided_ledh_kernel(
        model_factory, None, horizon=horizon, particle_count=particles,
        state_dimension=1, observation_dimension=1, parameter_dimension=6,
        dtype=tf.float64, jit_compile=jit_compile, dynamic_quadratic=True,
        reset_design=reset_design, return_trace=True, **settings)

    @tf.function(input_signature=[
        tf.TensorSpec([6], tf.float64), tf.TensorSpec([6], tf.float64),
        tf.TensorSpec([horizon, 1], tf.float64),
        tf.TensorSpec([particles, 1], tf.float64),
        tf.TensorSpec([horizon, particles, 1], tf.float64),
        tf.TensorSpec([horizon, 1, 1], tf.float64),
        tf.TensorSpec([horizon, 1], tf.float64),
    ], jit_compile=jit_compile, autograph=False)
    def directional(theta, direction, observations, initial_noise, noises, precision, linear):
        _model, states, tangents, covariance, d_covariance, *_rest = gaussian_direction_inputs(
            theta, direction, initial_noise, 1, 1)
        value, score, trace = shared(
            theta, direction, states, tf.broadcast_to(covariance, [particles, 1, 1]),
            noises, observations, tangents,
            tf.broadcast_to(d_covariance, [particles, 1, 1]), precision, linear)
        diagnostics = compact_control_diagnostics(trace)
        if error_diagnostics:
            diagnostics.update(_score_error_diagnostics(trace))
        return value, score[0], diagnostics

    return make_direction_kernel(directional, jit_compile=jit_compile) if all_directions else directional
