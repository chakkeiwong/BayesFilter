"""Compiled update blocks for BayesFilter's existing weighted NeuTra trainer.

The optimizer and loss remain owned by WeightedForwardKLNeuTraTrainer. This
wrapper changes dispatch only: each loop iteration updates the entire batch.
It stops on invalid updates and returns to the host once per checkpoint block.
"""
import tensorflow as tf

from bayesfilter.inference.neutra_weighted_training import WeightedNeuTraTrainingError


class WeightedTrainingBlock:
    def __init__(self, trainer, batch_size, *, jit_compile=True):
        if batch_size < 2:
            raise ValueError('training requires more than one row')
        self.trainer = trainer
        self.program = tf.function(
            self._run, autograph=False, jit_compile=jit_compile,
            input_signature=[tf.TensorSpec([batch_size, trainer.config.dimension], tf.float64),
                             tf.TensorSpec([batch_size], tf.float64),
                             tf.TensorSpec([], tf.int32)])

    def _run(self, rows, weights, updates):
        trainer = self.trainer

        def body(index, valid, loss, gradient_norm):
            result = trainer._train_step_impl(rows, weights)
            # The trainer checks loss and gradients before Adam. Also check
            # the resulting variables so a finite-gradient overflow is visible.
            finite_state = tf.reduce_all(tf.stack([
                tf.reduce_all(tf.math.is_finite(tf.cast(v, tf.float64)))
                for v in (*trainer.variables, *trainer.optimizer.variables)]))
            return index + 1, result[-1] & finite_state, result[0], result[4]

        return tf.while_loop(
            lambda index, valid, *_: (index < updates) & valid, body,
            (tf.constant(0, tf.int32), tf.constant(True),
             tf.constant(0., tf.float64), tf.constant(0., tf.float64)),
            parallel_iterations=1)

    def __call__(self, rows, weights, updates):
        if updates <= 0:
            raise ValueError('updates must be positive')
        result = self.program(tf.convert_to_tensor(rows, tf.float64),
                              tf.convert_to_tensor(weights, tf.float64),
                              tf.convert_to_tensor(updates, tf.int32))
        if not bool(result[1].numpy()):
            raise WeightedNeuTraTrainingError('compiled training block stopped on nonfinite update')
        return {'completed_updates': int(result[0].numpy()), 'finite': True,
                'loss': float(result[2].numpy()), 'gradient_norm': float(result[3].numpy()),
                'trace_count': self.program.experimental_get_tracing_count()}
