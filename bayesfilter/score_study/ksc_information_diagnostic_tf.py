"""CPU/FP64 saved-fit diagnosis; not a proposal, score or promotion path."""

import math

import tensorflow as tf

from bayesfilter.highdim import covariance_proposal_tf as core
from bayesfilter.highdim.ledh_backward_guide_tf import FrozenMixtureBackwardGuide
from bayesfilter.highdim.sqmc_ksc_tf import KSCSpec


def mixture_information(anchors, logits, centers, variances):
    residual = centers[None] - anchors[:, None]
    scores = residual / variances[None]
    log_terms = logits[None] - 0.5 * (
        tf.square(residual) / variances[None] + tf.math.log(variances)[None]
        + tf.constant(math.log(2.0 * math.pi), anchors.dtype))
    responsibilities = tf.nn.softmax(log_terms, axis=1)
    score = tf.reduce_sum(responsibilities * scores, axis=1)
    precision = tf.reduce_sum(responsibilities / variances[None], axis=1)
    omitted = tf.reduce_sum(responsibilities * tf.square(scores - score[:, None]), axis=1)
    return {"score": score, "precision": precision, "omitted_score_variance": omitted,
            "negative_hessian": precision - omitted}


def gaussian_product(mean, variance, logits, centers, variances):
    component_variances = 1.0 / (1.0 / variance + 1.0 / variances)
    component_means = component_variances * (mean / variance + centers / variances)
    weights = tf.nn.softmax(logits - 0.5 * (
        tf.square(mean - centers) / (variance + variances) + tf.math.log(variance + variances)))
    product_mean = tf.reduce_sum(weights * component_means)
    within = tf.reduce_sum(weights * component_variances)
    between = tf.reduce_sum(weights * tf.square(component_means - product_mean))
    return {"mean": product_mean, "variance": within + between,
            "within": within, "between": between}


def map_moments(maps):
    component_means = maps[0][1, :, 0]
    mean = tf.reduce_mean(component_means)
    variance = tf.reduce_mean(maps[2][1, :, 0, 0] + tf.square(component_means - mean))
    return mean, variance


def make_information_probe(components, controls):
    dtype = tf.float64
    signature = [tf.TensorSpec([2], dtype), tf.TensorSpec([2, 1], dtype),
                 tf.TensorSpec([32, 1], dtype), tf.TensorSpec([2, components], dtype),
                 tf.TensorSpec([2, components, 1], dtype),
                 tf.TensorSpec([2, components, 1, 1], dtype), tf.TensorSpec([2], tf.bool)]

    @tf.function(input_signature=signature, jit_compile=False, autograph=False)
    def probe(theta, observations, ancestors, logits, centers, covariances, active):
        model, _ = KSCSpec().model(theta, tf.zeros_like(theta))
        zeros = tf.zeros_like(ancestors)
        plain = core.build_maps(model, theta, ancestors, zeros, observations[0], controls)
        mean, variance = map_moments(plain)
        guide = FrozenMixtureBackwardGuide(logits, centers, covariances, active)
        guided = core.build_maps(model, theta, ancestors, zeros, observations[0], controls,
                                 backward_guide=guide, time=tf.constant(0))
        guided_mean, guided_variance = map_moments(guided)
        anchors = tf.concat([mean[None], plain[8][:, 0]], axis=0)
        precision, linear, d_precision, d_linear, valid = guide.evaluate(
            tf.constant(0), anchors[:, None], tf.ones([33, 1], dtype))
        actual_score = linear[:, 0] - precision[:, 0, 0] * anchors
        actual_derivative = d_linear[:, 0] - d_precision[:, 0, 0] * anchors - precision[:, 0, 0]
        information = mixture_information(anchors, logits[0], centers[0, :, 0],
                                          covariances[0, :, 0, 0])
        product = gaussian_product(mean, variance, logits[0], centers[0, :, 0],
                                   covariances[0, :, 0, 0])
        consumer_variance = 1.0 / (1.0 / variance + information["precision"][0])
        consumer_mean = mean + consumer_variance * information["score"][0]
        weights = tf.nn.softmax(logits[0])
        message_mean = tf.reduce_sum(weights * centers[0, :, 0])
        message_variance = tf.reduce_sum(weights * (
            covariances[0, :, 0, 0] + tf.square(centers[0, :, 0] - message_mean)))
        errors = tf.stack([
            tf.reduce_max(tf.abs(precision[:, 0, 0] - information["precision"])),
            tf.reduce_max(tf.abs(actual_score - information["score"])),
            tf.reduce_max(tf.abs(actual_derivative + information["negative_hessian"])),
            tf.abs(guided_variance - consumer_variance), tf.abs(guided_mean - consumer_mean)])
        scale = tf.maximum(tf.constant(1.0, dtype), tf.reduce_max(tf.abs(tf.concat([
            information["precision"], information["score"], information["negative_hessian"],
            tf.stack([consumer_variance, consumer_mean])], axis=0))))
        return {"anchors": anchors, **information, "identity_error": tf.reduce_max(errors),
                "identity_relative_error": tf.reduce_max(errors) / scale,
                "current_variance": variance, "consumer_variance": guided_variance,
                "consumer_mean": guided_mean, "product_variance": product["variance"],
                "product_mean": product["mean"], "product_within": product["within"],
                "product_between": product["between"],
                "variance_ratio": product["variance"] / guided_variance,
                "moment_gaussian_variance": 1.0 / (1.0 / variance + 1.0 / message_variance),
                "kernel_variance_spread": tf.reduce_max(covariances[0]) - tf.reduce_min(covariances[0]),
                "valid": valid & plain[12] & guided[12]}

    return probe
