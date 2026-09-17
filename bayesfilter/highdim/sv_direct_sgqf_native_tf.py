"""Native exact-transformed SV quadrature and analytical moment sensitivities.

The coordinates are independent and share the fixed scalar cloud. Broadcasting
the coordinate axis preserves the existing direct likelihood reweighting rule;
the date recurrence is not a Gaussian observation closure.
"""

import math
from functools import lru_cache

import tensorflow as tf
import tensorflow_probability as tfp


@lru_cache(maxsize=32)
def direct_sgqf_program(dates, width, points, *, with_score=False, jit_compile=True):
    d = tf.float64

    @tf.function(
        input_signature=[
            tf.TensorSpec([dates, width], d),
            tf.TensorSpec([width], d),
            tf.TensorSpec([width], d),
            tf.TensorSpec([width], d),
            tf.TensorSpec([points], d),
            tf.TensorSpec([points], d),
        ],
        jit_compile=jit_compile,
        autograph=False,
    )
    def run(z, gamma, beta, sigma, nodes, weights):
        mean = tf.zeros([width], d)
        variance = sigma**2 / (1.0 - gamma**2)
        d_mean = tf.zeros([width, 2], d)
        if with_score:
            normal = tfp.distributions.Normal(tf.constant(0.0, d), tf.constant(1.0, d))
            d_gamma = normal.prob(normal.quantile(gamma))
            d_variance = tf.stack(
                [
                    2.0 * sigma**2 * gamma * d_gamma / (1.0 - gamma**2) ** 2,
                    tf.zeros_like(gamma),
                ],
                axis=1,
            )
        else:
            d_gamma = tf.zeros_like(gamma)
            d_variance = tf.zeros_like(d_mean)
        logs = tf.TensorArray(d, dates, element_shape=[width])
        means = tf.TensorArray(d, dates, element_shape=[width])
        variances = tf.TensorArray(d, dates, element_shape=[width])
        score = tf.zeros([width, 2], d)
        log_weights = tf.math.log(weights)[None]
        direct_beta = tf.constant([0.0, -2.0], d)[None, :, None]

        def step(t, mean, variance, d_mean, d_variance, score, logs, means, variances):
            predicted_mean = tf.where(t > 0, gamma * mean, mean)
            predicted_variance = tf.where(
                t > 0, gamma**2 * variance + sigma**2, variance
            )
            if with_score:
                predicted_d_mean = tf.where(
                    t > 0,
                    gamma[:, None] * d_mean
                    + tf.stack([d_gamma * mean, tf.zeros_like(mean)], axis=1),
                    d_mean,
                )
                predicted_d_variance = tf.where(
                    t > 0,
                    gamma[:, None] ** 2 * d_variance
                    + tf.stack(
                        [2.0 * gamma * d_gamma * variance, tf.zeros_like(variance)],
                        axis=1,
                    ),
                    d_variance,
                )
            scale = tf.sqrt(predicted_variance)
            cloud = predicted_mean[:, None] + scale[:, None] * nodes[None]
            residual = z[t, :, None] - 2.0 * tf.math.log(beta)[:, None] - cloud
            log_density = (
                -0.5 * tf.constant(math.log(2.0 * math.pi), d)
                + 0.5 * residual
                - 0.5 * tf.exp(residual)
            )
            log_normalizer = tf.reduce_logsumexp(log_weights + log_density, axis=1)
            normalized = tf.exp(log_weights + log_density - log_normalizer[:, None])
            mean = tf.reduce_sum(normalized * cloud, axis=1)
            second = tf.reduce_sum(normalized * cloud**2, axis=1)
            variance = second - mean**2
            if with_score:
                d_cloud = (
                    predicted_d_mean[:, :, None]
                    + (0.5 * predicted_d_variance / scale[:, None])[:, :, None]
                    * nodes[None, None]
                )
                d_log = (
                    0.5
                    * (1.0 - tf.exp(residual))[:, None, :]
                    * (-d_cloud + direct_beta)
                )
                increment = tf.reduce_sum(normalized[:, None, :] * d_log, axis=2)
                centered = d_log - increment[:, :, None]
                d_mean = tf.reduce_sum(
                    normalized[:, None, :] * (d_cloud + centered * cloud[:, None, :]),
                    axis=2,
                )
                d_second = tf.reduce_sum(
                    normalized[:, None, :]
                    * (
                        2.0 * cloud[:, None, :] * d_cloud
                        + centered * cloud[:, None, :] ** 2
                    ),
                    axis=2,
                )
                d_variance = d_second - 2.0 * mean[:, None] * d_mean
                score = score + increment
            return (
                t + 1,
                mean,
                variance,
                d_mean,
                d_variance,
                score,
                logs.write(t, log_normalizer),
                means.write(t, mean),
                variances.write(t, variance),
            )

        result = tf.while_loop(
            lambda t, *_: t < dates,
            step,
            (
                tf.constant(0),
                mean,
                variance,
                d_mean,
                d_variance,
                score,
                logs,
                means,
                variances,
            ),
            maximum_iterations=dates,
            parallel_iterations=1,
        )
        return (
            result[6].stack(),
            tf.reshape(result[5], [-1]),
            result[7].stack(),
            tf.linalg.diag(result[8].stack()),
        )

    return run
