"""Target-specific frozen fits for the two-observation KSC diagnostic.

The local kernel fit is a message representation, not the original KDM
algorithm or an exact twist sampler. These adapters do not issue canonical
Contract E admission, tuning, or scientific promotion evidence.
"""

import math

import tensorflow as tf

from bayesfilter.highdim.ledh_backward_guide_tf import FrozenQuadraticBackwardGuide
from bayesfilter.highdim.sqmc_ksc_tf import MEANS, VARIANCES, WEIGHTS, persistence
from bayesfilter.replication.controlled_smc_tf import ScalarGaussianControlledSMC


def ksc_observation_log_density(observation, states, log_beta):
    """Evaluate the actual seven-component observation mixture at any shape."""
    states = tf.convert_to_tensor(states, tf.float64)
    variances = tf.constant(VARIANCES, states.dtype)
    residual = (tf.cast(observation, states.dtype)[..., None] - states[..., None]
                - 2.0 * tf.convert_to_tensor(log_beta, states.dtype)
                - tf.constant(MEANS, states.dtype))
    terms = (tf.math.log(tf.constant(WEIGHTS, states.dtype))
             - 0.5 * (tf.square(residual) / variances + tf.math.log(variances)
                      + tf.math.log(tf.constant(2.0 * math.pi, states.dtype))))
    return tf.reduce_logsumexp(terms, axis=-1)


def _frozen_theta(theta):
    theta = tf.ensure_shape(tf.convert_to_tensor(theta, tf.float64), [2])
    values = tf.get_static_value(theta)
    if values is None or not all(math.isfinite(float(value)) for value in values):
        raise ValueError("fitting requires two finite, fixed base parameters")
    gamma = float(persistence(theta))
    if not 0.0 < gamma < 1.0:
        raise ValueError("the KSC diagnostic requires persistence in (0, 1)")
    return gamma, float(values[1])


def make_ksc_quadratic_fit(theta, particles, passes, *, jit_compile=True):
    """Fit replicated controlled-SMC policies and integrate the future once.

    Each pass has independent caller-supplied forward noise. The source
    projection remains the replicated engine's scalar projection with offset
    0.001. A valid controlled-SMC policy can still have negative guide precision;
    it is reported as ineligible for the shared LEDH consumer without clipping.
    """
    if not isinstance(particles, int) or particles < 3:
        raise ValueError("particles must be an integer at least three")
    if not isinstance(passes, int) or passes < 1:
        raise ValueError("passes must be a positive integer")
    gamma, log_beta = _frozen_theta(theta)
    engine = ScalarGaussianControlledSMC(
        observations=2, particles=particles, alpha=gamma,
        process_variance=1.0, initial_mean=0.0,
        initial_variance=gamma * gamma + 1.0,
        observation_log_density=lambda observation, states: ksc_observation_log_density(
            observation, states, log_beta),
        precision_policy="source_scalar_projection", jit_compile=jit_compile)

    @tf.function(input_signature=[
        tf.TensorSpec([2], tf.float64),
        tf.TensorSpec([passes, 2, particles], tf.float64),
        tf.TensorSpec([passes, 2], tf.float64),
    ], jit_compile=jit_compile)
    def fit(observations, normal_noise, resampling_uniforms):
        policy = tf.zeros([2, 3], tf.float64)
        healthy = (tf.reduce_all(tf.math.is_finite(observations))
                   & tf.reduce_all(tf.math.is_finite(normal_noise))
                   & tf.reduce_all(tf.math.is_finite(resampling_uniforms))
                   & tf.reduce_all(resampling_uniforms >= 0.0)
                   & tf.reduce_all(resampling_uniforms < 1.0))
        gram_ratios = tf.TensorArray(tf.float64, size=passes, element_shape=[2])
        residuals = tf.TensorArray(tf.float64, size=passes, element_shape=[2])
        projected = tf.TensorArray(tf.bool, size=passes, element_shape=[2])
        pilot_ess = tf.TensorArray(tf.float64, size=passes, element_shape=[2])
        pilot_values = tf.TensorArray(tf.float64, size=passes, element_shape=[2])

        def body(index, current_policy, valid, ratios, errors, projections, ess, values):
            forward = engine.forward(current_policy, observations,
                                     normal_noise[index], resampling_uniforms[index])
            update = engine.fit(current_policy, forward["x"],
                                forward["log_incremental_weight"])
            return (index + 1, update["policy"], valid & forward["valid"] & update["valid"],
                    ratios.write(index, update["gram_eigenvalue_ratio"]),
                    errors.write(index, update["max_residual"]),
                    projections.write(index, update["projected"]),
                    ess.write(index, forward["ess"]), values.write(index, forward["logZ"]))

        (_, policy, healthy, gram_ratios, residuals, projected,
         pilot_ess, pilot_values) = tf.while_loop(
             lambda index, *_: index < passes, body,
             (tf.constant(0), policy, healthy, gram_ratios, residuals,
              projected, pilot_ess, pilot_values))
        guide = FrozenQuadraticBackwardGuide.from_scalar_controlled_smc_future(
            policy, gamma, 1.0)
        precision = guide.precision
        linear = guide.linear
        eligible = (healthy & tf.reduce_all(tf.math.is_finite(precision))
                    & tf.reduce_all(tf.math.is_finite(linear))
                    & tf.reduce_all(precision >= 0.0))
        return {"policy": policy, "precision": precision, "linear": linear,
                "fit_valid": healthy, "guide_eligible": eligible,
                "density_eligible": eligible & (precision[0, 0, 0] > 0.0),
                "gram_eigenvalue_ratio": gram_ratios.stack(),
                "max_residual": residuals.stack(), "projected": projected.stack(),
                "pilot_ess": pilot_ess.stack(), "pilot_logZ": pilot_values.stack(),
                "source_projection_offset": tf.constant(0.001, tf.float64)}

    return fit


def make_ksc_kernel_fit(theta, centers, transition_replicates, bandwidth_factor,
                        *, jit_compile=True):
    """Fit a normalized kernel message using inverse-design-density weights.

    The future likelihood is estimated with independent transition replicates.
    Self-normalization and kernel smoothing are approximations. Nonpositive
    bandwidth or degenerate empirical variance rejects the fit; no ridge or
    exact-message substitution changes the candidate.
    """
    if not isinstance(centers, int) or centers < 2:
        raise ValueError("centers must be an integer at least two")
    if not isinstance(transition_replicates, int) or transition_replicates < 1:
        raise ValueError("transition_replicates must be a positive integer")
    if not math.isfinite(bandwidth_factor) or bandwidth_factor <= 0.0:
        raise ValueError("bandwidth_factor must be finite and positive")
    gamma, log_beta = _frozen_theta(theta)
    mixture_weights = tf.constant(WEIGHTS, tf.float64)
    mixture_means = tf.constant(MEANS, tf.float64)
    broad_variance = tf.constant(4.0 * max(1.0 + value for value in VARIANCES)
                                 / (gamma * gamma), tf.float64)
    predictive_variance = tf.constant(gamma * gamma + 1.0, tf.float64)

    @tf.function(input_signature=[
        tf.TensorSpec([2], tf.float64),
        tf.TensorSpec([centers], tf.float64),
        tf.TensorSpec([centers], tf.float64),
        tf.TensorSpec([centers, transition_replicates], tf.float64),
    ], jit_compile=jit_compile)
    def fit(observations, component_uniforms, design_normal, transition_normal):
        broad_mean = ((observations[1] - 2.0 * log_beta
                       - tf.reduce_sum(mixture_weights * mixture_means)) / gamma)
        nodes = tf.where(component_uniforms < 0.5,
                         tf.sqrt(predictive_variance) * design_normal,
                         broad_mean + tf.sqrt(broad_variance) * design_normal)
        design_means = tf.stack([tf.zeros([], tf.float64), broad_mean])
        design_variances = tf.stack([predictive_variance, broad_variance])
        design_terms = (-tf.math.log(tf.constant(2.0, tf.float64))
                        - 0.5 * (tf.square(nodes[:, None] - design_means) / design_variances
                                 + tf.math.log(design_variances)
                                 + tf.math.log(tf.constant(2.0 * math.pi, tf.float64))))
        log_design_density = tf.reduce_logsumexp(design_terms, axis=1)
        future_states = gamma * nodes[:, None] + transition_normal
        log_future = (tf.reduce_logsumexp(ksc_observation_log_density(
            observations[1], future_states, log_beta), axis=1)
                      - tf.math.log(tf.constant(transition_replicates, tf.float64)))
        log_unnormalized = log_future - log_design_density
        log_weights = tf.nn.log_softmax(log_unnormalized)
        weights = tf.exp(log_weights)
        mean = tf.reduce_sum(weights * nodes)
        variance = tf.reduce_sum(weights * tf.square(nodes - mean))
        ess = 1.0 / tf.reduce_sum(tf.square(weights))
        bandwidth = (tf.constant(bandwidth_factor, tf.float64) * tf.sqrt(variance)
                     * tf.pow(ess, tf.constant(-0.2, tf.float64)))
        kernel_variance = tf.square(bandwidth)
        log_normalization = (tf.reduce_logsumexp(log_unnormalized)
                             - tf.math.log(tf.constant(centers, tf.float64)))
        healthy = (tf.reduce_all(tf.math.is_finite(observations))
                   & tf.reduce_all(tf.math.is_finite(component_uniforms))
                   & tf.reduce_all(component_uniforms >= 0.0)
                   & tf.reduce_all(component_uniforms < 1.0)
                   & tf.reduce_all(tf.math.is_finite(design_normal))
                   & tf.reduce_all(tf.math.is_finite(transition_normal))
                   & tf.reduce_all(tf.math.is_finite(log_weights))
                   & tf.math.is_finite(log_normalization)
                   & tf.math.is_finite(variance) & (variance > 0.0)
                   & tf.math.is_finite(bandwidth) & (bandwidth > 0.0)
                   & tf.math.is_finite(kernel_variance) & (kernel_variance > 0.0))
        return {"logits": tf.stack([log_weights, tf.zeros_like(log_weights)]),
                "centers": tf.stack([nodes[:, None], tf.zeros([centers, 1], tf.float64)]),
                "covariances": tf.stack([
                    tf.fill([centers, 1, 1], kernel_variance),
                    tf.ones([centers, 1, 1], tf.float64)]),
                "active": tf.constant([True, False]),
                "nodes": nodes, "log_design_density": log_design_density,
                "log_future_estimate": log_future, "log_weights": log_weights,
                "log_normalization_estimate": log_normalization,
                "raw_variance": variance, "ess": ess,
                "max_weight": tf.reduce_max(weights), "bandwidth": bandwidth,
                "fit_valid": healthy, "guide_eligible": healthy,
                "density_eligible": healthy}

    return fit
