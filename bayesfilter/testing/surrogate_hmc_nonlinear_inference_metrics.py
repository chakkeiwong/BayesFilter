"""Independent post-run M10 posterior functionals in the reference's units."""
import tensorflow as tf

from bayesfilter.testing.surrogate_hmc_nonlinear_log_reference import make_log_reference


def scalar_functionals(parameters, predictive, means, scales, cuts):
    standardized = (parameters-means)/scales
    a, b = standardized[:, 0], standardized[:, 1]
    moments = tf.stack([a, b, a*a, b*b, a*b], axis=1)
    cdf = tf.cast(parameters[:, :, None] <= cuts[None, :, :], tf.float64)
    return tf.concat([moments, tf.reshape(cdf, [-1, 6]), predictive], axis=1)


def make_functionals(claim, reference):
    observations = tf.constant(claim["raw_observations"], tf.float64)
    means, scales, cuts = [tf.constant(reference[k], tf.float64)
                          for k in ("posterior_mean", "posterior_sd", "marginal_quantiles")]
    kernel = make_log_reference(with_score=False)

    @tf.function(input_signature=[tf.TensorSpec([250, 4, 2], tf.float64)], jit_compile=True, autograph=False)
    def evaluate(samples):
        parameters = tf.reshape(samples, [1000, 2])
        def row(point):
            result = kernel(point, observations)
            return result["predictive"], result["valid"]
        # Independent reporting, not a training or canonical-score lane.
        predictive, valid = tf.map_fn(row, parameters, fn_output_signature=(
            tf.TensorSpec([2], tf.float64), tf.TensorSpec([], tf.bool)), parallel_iterations=1)
        values = scalar_functionals(parameters, predictive, means, scales, cuts)
        return samples, tf.reshape(values, [250, 4, 13]), valid
    return evaluate
