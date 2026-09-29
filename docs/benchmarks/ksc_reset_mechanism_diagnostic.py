"""Trace-only explanatory diagnostics; never imported by a filtering runtime."""
from __future__ import annotations
import math
import tensorflow as tf

from bayesfilter.highdim.sqmc_ksc_tf import MEANS, VARIANCES, WEIGHTS


def predictive_functional(theta, direction, states, d_states, logits, d_logits, observations):
    """Exact next-observation integral for a discrete current-state distribution.

    Arrays have shape [time, particles]. Logits may be unnormalized. The
    derivative includes theta, particle-location and normalized-weight terms.
    """
    dtype = theta.dtype
    phi = .5 * (1. + tf.math.erf(theta[0] / tf.sqrt(tf.constant(2., dtype))))
    d_phi = tf.exp(-.5 * theta[0] ** 2) / math.sqrt(2. * math.pi) * direction[0]
    mean = phi * states + 2. * theta[1]
    d_mean = d_phi * states + phi * d_states + 2. * direction[1]
    variance = 1. + tf.constant(VARIANCES, dtype)
    residual = observations[:, None, None] - mean[..., None] - tf.constant(MEANS, dtype)
    components = tf.math.log(tf.constant(WEIGHTS, dtype)) - .5 * (
        residual ** 2 / variance + tf.math.log(variance) + math.log(2. * math.pi))
    log_kernel = tf.reduce_logsumexp(components, axis=-1)
    d_log_kernel = tf.reduce_sum(tf.nn.softmax(components, axis=-1) * residual / variance,
                                 axis=-1) * d_mean
    weights = tf.nn.softmax(logits, axis=-1)
    log_weights = tf.nn.log_softmax(logits, axis=-1)
    d_log_weights = d_logits - tf.reduce_sum(weights * d_logits, axis=-1, keepdims=True)
    terms = log_weights + log_kernel
    value = tf.reduce_logsumexp(terms, axis=-1)
    tangent = tf.reduce_sum(tf.nn.softmax(terms, axis=-1) * (d_log_weights + d_log_kernel), axis=-1)
    return value, tangent


def make_kernel(spec, route, controls, n, horizon):
    """Use the canonical executor; keep only small diagnostic trace reductions."""
    from bayesfilter.highdim.sqmc_campaign_tf import numerical_settings, route_settings, reset_design
    from bayesfilter.highdim.ledh_canonical_score_tf import canonical_value_and_analytical_score
    settings = dict(numerical_settings(controls), **route_settings(route))
    design = reset_design(n, 1, tf.float64)
    signature = [tf.TensorSpec([2], tf.float64), tf.TensorSpec([2], tf.float64),
                 tf.TensorSpec([n, 1], tf.float64), tf.TensorSpec([horizon, n, 1], tf.float64),
                 tf.TensorSpec([horizon, n], tf.float64), tf.TensorSpec([horizon, 1], tf.float64)]

    @tf.function(input_signature=signature, jit_compile=True, autograph=False)
    def compute(theta, direction, initial, noise, uniforms, obs):
        model, _ = spec.model(theta, direction)
        states, covs, ds, dc = spec.initial_cloud(theta, initial, direction)
        value, score, trace = canonical_value_and_analytical_score(
            model, theta, states, covs, noise, obs, with_score=True, return_trace=True,
            initial_state_tangent=ds, initial_covariance_tangent=dc, reset_design=design,
            process_ancestor_uniforms=uniforms, **settings)
        def stack(key):
            return tf.stack([record[key] for record in trace])
        children = stack('children')[..., 0]
        d_children = stack('d_children')[..., 0]
        reset = stack('states_after_reset')[..., 0]
        d_reset = stack('d_states_after_reset')[..., 0]
        logits = stack('posterior_logits')
        d_logits = stack('d_posterior_logits')
        outgoing = stack('outgoing_log_weights')
        d_outgoing = stack('d_outgoing_log_weights')

        def moments(x, log_w):
            w = tf.nn.softmax(log_w, axis=-1)
            mean = tf.reduce_sum(w * x, axis=-1)
            centered = x - mean[:, None]
            variance = tf.reduce_sum(w * centered ** 2, axis=-1)
            skew = tf.reduce_sum(w * centered ** 3, axis=-1) / variance ** 1.5
            kurt = tf.reduce_sum(w * centered ** 4, axis=-1) / variance ** 2
            return tf.stack([mean, variance, skew, kurt], axis=-1)

        before = moments(children, logits)
        after = moments(reset, outgoing)
        # Outgoing weights are uniform for this canonical reset. Preserve the
        # discrepancy check to avoid applying this grouping silently elsewhere.
        outgoing_nonuniformity = tf.reduce_max(tf.abs(tf.nn.softmax(outgoing, axis=-1) - 1. / n))
        even_var = tf.math.reduce_variance(reset[:, ::2], axis=-1)
        odd_var = tf.math.reduce_variance(reset[:, 1::2], axis=-1)
        group_ratio = .5 * (even_var + odd_var) / after[:, 1]
        source_v, source_s = predictive_functional(theta, direction, children[:-1], d_children[:-1],
                                                   logits[:-1], d_logits[:-1], obs[1:, 0])
        reset_v, reset_s = predictive_functional(theta, direction, reset[:-1], d_reset[:-1],
                                                 outgoing[:-1], d_outgoing[:-1], obs[1:, 0])
        increments = tf.reduce_logsumexp(logits, axis=-1)
        d_increments = tf.reduce_sum(tf.nn.softmax(logits, axis=-1) * d_logits, axis=-1)
        return dict(value=value, score=score[0], valid=trace[-1]['program_valid'],
                    before_moments=before, after_moments=after,
                    outgoing_nonuniformity=outgoing_nonuniformity,
                    within_group_variance_ratio=group_ratio,
                    source_log_prediction=source_v, reset_log_prediction=reset_v,
                    source_predictive_score=source_s, reset_predictive_score=reset_s,
                    actual_log_increment=increments, actual_score_increment=d_increments)
    return compute
