"""Independent actual-SV retained functionals; batch-native reference diagnostics."""
import tensorflow as tf
from bayesfilter.testing.surrogate_hmc_sv_reference import chart_and_log_prior
from bayesfilter.testing.surrogate_hmc_sv_posterior_reference import make_statistics
from bayesfilter.testing.surrogate_hmc_nonlinear_inference_metrics import scalar_functionals


def make_functionals(claim, reference):
    observations = tf.cast(tf.constant(claim["raw_observations"], tf.float32), tf.float64)
    means, scales, cuts = [tf.constant(reference[k], tf.float64) for k in ("means", "scales", "cuts")]
    statistics = make_statistics(intervals=1024, half_width=16., batch_size=32)

    @tf.function(input_signature=[tf.TensorSpec([250, 4, 2], tf.float64)], jit_compile=True, autograph=False)
    def evaluate(samples):
        parameters = chart_and_log_prior(tf.reshape(samples, [1000, 2]))[0]
        batches = tf.reshape(tf.concat([parameters, tf.repeat(parameters[-1:], 24, axis=0)], axis=0), [32, 32, 2])
        predictions = tf.TensorArray(tf.float64, size=32, element_shape=[32, 2])
        valid = tf.TensorArray(tf.bool, size=32, element_shape=[32])
        def body(i, predictions, valid):
            # A gather preserves the fixed batch extent inside the XLA loop.
            # Dynamic slicing gives XLA a 1024-row upper bound instead of 32.
            logp, predictive, loss = statistics(tf.gather(batches, i), observations)
            good = (tf.math.is_finite(logp) & tf.reduce_all(tf.math.is_finite(predictive), axis=1)
                    & tf.reduce_all((predictive >= 0) & (predictive <= 1), axis=1)
                    & tf.math.is_finite(loss) & (loss <= 1e-9))
            return i+1, predictions.write(i, predictive), valid.write(i, good)
        _, predictions, valid = tf.while_loop(lambda i, *_: i < 32, body,
            (tf.constant(0), predictions, valid), parallel_iterations=1)
        values = scalar_functionals(parameters, predictions.concat()[:1000], means, scales, cuts)
        return tf.reshape(parameters, [250, 4, 2]), tf.reshape(values, [250, 4, 13]), valid.concat()[:1000]
    return evaluate
