"""Optional physical-output Gaussian mixture, with analytical directional tangents.

This candidate retains the shared LEDH flow and issues no canonical identity.
The current-observation Gaussian is exact only for linear Gaussian observations.
"""

import math

import tensorflow as tf

from bayesfilter.highdim.ledh_backward_guide_tf import (
    _checked_cholesky,
    _information_proposal_moments,
)


def gaussian_density_tangent(points, d_points, means, d_means, covariance, d_covariance):
    """Full Gaussian log density and total moving-point derivative."""
    factor, valid = _checked_cholesky(covariance)
    identity = tf.eye(points.shape[-1], dtype=points.dtype)
    precision = tf.linalg.cholesky_solve(
        factor, tf.broadcast_to(identity, tf.shape(covariance)))
    residual = points - means
    force = tf.linalg.matvec(precision, residual)
    log_density = -0.5 * (
        tf.reduce_sum(residual * force, axis=-1)
        + 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(factor)), axis=-1)
        + tf.constant(points.shape[-1] * math.log(2.0 * math.pi), points.dtype))
    tangent = (-tf.reduce_sum(force * (d_points - d_means), axis=-1)
               + 0.5 * tf.reduce_sum(force * tf.linalg.matvec(d_covariance, force), axis=-1)
               - 0.5 * tf.linalg.trace(precision @ d_covariance))
    valid &= tf.reduce_all(tf.math.is_finite(log_density))
    valid &= tf.reduce_all(tf.math.is_finite(tangent))
    return log_density, tangent, valid


def conditional_moments(model, anchors, d_anchors, observation, d_process, d_observation):
    """Local observation-conditioned Gaussian, including Jacobian derivatives."""
    covariance = model.process_covariance
    observation_covariance = model.observation_covariance
    jacobian = model.observation_jacobian_fn(anchors)
    d_jacobian = (tf.zeros_like(jacobian) if model.observation_jacobian_tangent_fn is None
                  else model.observation_jacobian_tangent_fn(anchors, d_anchors))
    transpose = tf.linalg.matrix_transpose(jacobian)
    d_transpose = tf.linalg.matrix_transpose(d_jacobian)
    cross = covariance @ transpose
    d_cross = d_process @ transpose + covariance @ d_transpose
    innovation_covariance = jacobian @ cross + observation_covariance
    d_innovation_covariance = d_jacobian @ cross + jacobian @ d_cross + d_observation
    factor, valid = _checked_cholesky(innovation_covariance)
    gain = tf.linalg.matrix_transpose(tf.linalg.cholesky_solve(
        factor, tf.linalg.matrix_transpose(cross)))
    d_gain = tf.linalg.matrix_transpose(tf.linalg.cholesky_solve(
        factor, tf.linalg.matrix_transpose(d_cross - gain @ d_innovation_covariance)))
    residual = observation - model.observation_fn(anchors)
    d_residual = -model.observation_tangent_fn(anchors, d_anchors)
    means = anchors + tf.linalg.matvec(gain, residual)
    d_means = d_anchors + tf.linalg.matvec(d_gain, residual) + tf.linalg.matvec(gain, d_residual)
    posterior_covariance = covariance - gain @ tf.linalg.matrix_transpose(cross)
    d_posterior_covariance = (d_process - d_gain @ tf.linalg.matrix_transpose(cross)
                             - gain @ tf.linalg.matrix_transpose(d_cross))
    posterior_covariance = 0.5 * (posterior_covariance + tf.linalg.matrix_transpose(posterior_covariance))
    d_posterior_covariance = 0.5 * (d_posterior_covariance + tf.linalg.matrix_transpose(d_posterior_covariance))
    return means, posterior_covariance, d_means, d_posterior_covariance, valid


def physical_mixture_draw(model, guide, time, anchors, d_anchors, observation,
                          d_process, d_observation, noise, uniforms, beta, flow_result,
                          *, return_diagnostics=False):
    """Draw one component and evaluate both full densities at that same point."""
    baseline, d_baseline, log_det, d_log_det, flow_mean, d_flow_mean, matrix, d_matrix = flow_result
    means, covariance, d_means, d_covariance, valid = conditional_moments(
        model, anchors, d_anchors, observation, d_process, d_observation)
    guide_valid = tf.constant(True)
    product_margin = tf.constant(0., anchors.dtype)
    margin_measured = tf.constant(False)
    neural_components = None
    if guide is not None:
        if getattr(guide, "conditioning", None) == "neural_conditional_kdm":
            neural_components = guide.offsets.shape[1]
            means, covariance, d_means, d_covariance, guide_valid = guide.component_moments(
                time, means, covariance, d_means, d_covariance)
        elif getattr(guide, "conditioning", None) == "mixture_product_moments":
            means, covariance, d_means, d_covariance, guide_valid, product_margin = guide.condition_gaussian(
                time, means, covariance, d_means, d_covariance)
            margin_measured = guide.active[time]
            product_margin = tf.where(margin_measured, product_margin, tf.zeros_like(product_margin))
        else:
            means, covariance, d_means, d_covariance, guide_valid = _information_proposal_moments(
                guide, time, means, d_means, covariance, d_covariance)
        valid &= guide_valid
    factor, factor_valid = _checked_cholesky(covariance)
    valid &= factor_valid
    left = tf.linalg.triangular_solve(factor, d_covariance)
    congruence = tf.linalg.matrix_transpose(tf.linalg.triangular_solve(
        factor, tf.linalg.matrix_transpose(left)))
    phi = tf.linalg.band_part(congruence, -1, 0) - 0.5 * tf.linalg.diag(tf.linalg.diag_part(congruence))
    valid &= tf.reduce_all(tf.math.is_finite(uniforms) & (uniforms >= 0.0) & (uniforms < 1.0))
    selected = uniforms < tf.cast(beta, uniforms.dtype)
    if neural_components is None:
        output = means + tf.linalg.matvec(factor, noise)
        d_output = d_means + tf.linalg.matvec(factor @ phi, noise)
    else:
        conditional_uniform = tf.where(selected, uniforms / tf.cast(beta, uniforms.dtype), 0.)
        component = tf.cast(tf.floor(conditional_uniform * neural_components), tf.int32)
        component = tf.minimum(component, neural_components - 1)
        chosen_means = tf.gather(means, component, axis=1, batch_dims=1)
        chosen_d_means = tf.gather(d_means, component, axis=1, batch_dims=1)
        chosen_factor = tf.gather(factor, component, axis=1, batch_dims=1)
        chosen_d_factor = tf.gather(factor @ phi, component, axis=1, batch_dims=1)
        output = chosen_means + tf.linalg.matvec(chosen_factor, noise)
        d_output = chosen_d_means + tf.linalg.matvec(chosen_d_factor, noise)
    points = tf.where(selected[:, None], output, baseline)
    d_points = tf.where(selected[:, None], d_output, d_baseline)
    if neural_components is None:
        output_log, d_output_log, output_valid = gaussian_density_tangent(
            points, d_points, means, d_means, covariance, d_covariance)
    else:
        component_log, component_tangent, output_valid = gaussian_density_tangent(
            points[:, None], d_points[:, None], means, d_means, covariance, d_covariance)
        output_log = tf.reduce_logsumexp(component_log, axis=-1) - tf.math.log(
            tf.cast(neural_components, points.dtype))
        d_output_log = tf.reduce_sum(tf.nn.softmax(component_log, axis=-1)
                                     * component_tangent, axis=-1)
    valid &= output_valid
    if beta == 1.0:
        proposal_log, d_proposal_log = output_log, d_output_log
    else:
        delta = tf.linalg.solve(matrix, (points - flow_mean)[..., None])[..., 0]
        d_delta = tf.linalg.solve(matrix, (d_points - d_flow_mean
            - tf.linalg.matvec(d_matrix, delta))[..., None])[..., 0]
        base_log, d_base_log, base_valid = gaussian_density_tangent(
            delta, d_delta, tf.zeros_like(delta), tf.zeros_like(delta),
            model.process_covariance, d_process)
        base_log -= log_det
        d_base_log -= d_log_det
        logits = tf.stack([base_log + tf.math.log(tf.cast(1.0 - beta, points.dtype)),
                           output_log + tf.math.log(tf.cast(beta, points.dtype))], axis=-1)
        proposal_log = tf.reduce_logsumexp(logits, axis=-1)
        responsibilities = tf.nn.softmax(logits, axis=-1)
        d_proposal_log = tf.reduce_sum(responsibilities * tf.stack(
            [d_base_log, d_output_log], axis=-1), axis=-1)
        valid &= base_valid
    valid &= tf.reduce_all(tf.math.is_finite(points)) & tf.reduce_all(tf.math.is_finite(d_points))
    valid &= tf.reduce_all(tf.math.is_finite(proposal_log)) & tf.reduce_all(tf.math.is_finite(d_proposal_log))
    result = (points, d_points, proposal_log, d_proposal_log, valid,
              tf.reduce_sum(tf.cast(selected, tf.int32)))
    if return_diagnostics:
        return result + (dict(backward_guide_valid=guide_valid,
            guide_conditioning_min_relative_spd_margin=product_margin,
            guide_conditioning_margin_measured=margin_measured),)
    return result
