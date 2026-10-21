"""Independent LG2 retained diagnostics in the qualified reference's units.

This reporting path preserves the time and chain axes. The Kalman calculation
is batch native across all 1,000 positions in a fixed retained chunk.
"""
import tensorflow as tf

from bayesfilter.testing.surrogate_hmc_linear_reference import make_linear_reference
from bayesfilter.testing.surrogate_hmc_nonlinear_inference_metrics import scalar_functionals


def make_functionals(claim, reference):
    if claim.get("model_id") != "LG2" or claim.get("horizon") != 20:
        raise ValueError("LG2 T20 claim data required")
    kernel = make_linear_reference("LG2")
    observations = tf.cast(tf.constant(claim["observations"], tf.float32), tf.float64)
    lower = tf.constant(kernel.specification["lower"], tf.float64)
    upper = tf.constant(kernel.specification["upper"], tf.float64)
    means, scales, cuts = [tf.constant(reference[key], tf.float64) for key in
                          ("posterior_mean", "posterior_sd", "marginal_quantiles")]
    if not bool(tf.reduce_all(tf.math.is_finite(scales) & (scales > 0))):
        raise ValueError("finite positive reference scales required")

    @tf.function(input_signature=[tf.TensorSpec([250, 4, 2], tf.float64)],
                 jit_compile=True, autograph=False)
    def evaluate(samples):
        positions = tf.reshape(samples, [1000, 2])
        parameters = lower + (upper-lower)*tf.math.sigmoid(positions)
        logp, predictive = kernel.physical_statistics(parameters, observations)
        values = scalar_functionals(parameters, predictive, means, scales, cuts)
        valid = (tf.reduce_all(tf.math.is_finite(positions), axis=1)
                 & tf.math.is_finite(logp)
                 & tf.reduce_all(tf.math.is_finite(values), axis=1)
                 & tf.reduce_all((predictive >= 0) & (predictive <= 1), axis=1))
        return (tf.reshape(parameters, [250, 4, 2]),
                tf.reshape(values, [250, 4, 13]), valid)
    return evaluate
