"""Optional simultaneous weighted FKL/RKL on the canonical transport authority.

Noé et al. (2019), Methods Eq. 9, motivates the combined objective. The local
target remains exact; the author's energy clipping is not imported. This is
an optional training mechanism, not a new architecture or a calibrated default.
"""
from __future__ import annotations

import math
import tensorflow as tf
from bayesfilter.inference.neutra_transport import NeuTraTransportTrainer


class JointNeuTraTrainer(NeuTraTransportTrainer):
    """One weighted sum of gradients, one Adam update, full common rollback."""

    def __init__(self, transport, target_value_score, config, *, target_signature,
                 teacher_id, forward_weight=1., reverse_weight=1.):
        weights = (float(forward_weight), float(reverse_weight))
        if any(not math.isfinite(w) or w < 0 for w in weights) or not any(weights):
            raise ValueError('objective weights must be finite, nonnegative and nonzero')
        if not isinstance(teacher_id, str) or not teacher_id:
            raise ValueError('a teacher identity is required')
        self.forward_weight, self.reverse_weight = weights
        self.teacher_id = teacher_id
        super().__init__(transport, target_value_score, config, target_signature=target_signature)
        signature = [tf.TensorSpec([config.batch_size, transport.parameter_dim], tf.float64),
                     tf.TensorSpec([config.batch_size, transport.parameter_dim], tf.float64),
                     tf.TensorSpec([config.batch_size], tf.float64)]
        self.evaluate_joint = tf.function(self._evaluate_joint, input_signature=signature,
                                          jit_compile=config.jit_compile, autograph=False)
        self.train_joint_step = tf.function(self._joint_step, input_signature=signature,
                                            jit_compile=config.jit_compile, autograph=False)

    def _evaluate_joint(self, latent, physical, log_weights):
        zero = tf.constant(0., tf.float64)
        gradients = tuple(tf.zeros_like(v) for v in self.variables)
        forward, reverse, valid = zero, zero, tf.constant(True)
        if self.reverse_weight:
            reverse_result = super()._evaluate(latent)
            reverse = reverse_result['loss']
            gradients = tuple(self.reverse_weight*g for g in reverse_result['gradients'])
            valid &= reverse_result['valid']
        if self.forward_weight:
            with tf.GradientTape() as tape:
                weights = tf.stop_gradient(tf.nn.softmax(log_weights))
                forward = -tf.reduce_sum(weights*self.transport.log_prob(tf.stop_gradient(physical)))
            fg = tape.gradient(forward, self.variables)
            if any(g is None for g in fg):
                raise ValueError('missing forward objective gradient')
            gradients = tuple(g+self.forward_weight*f for g, f in zip(gradients, fg))
        loss = self.forward_weight*forward+self.reverse_weight*reverse
        valid &= tf.math.is_finite(loss) & tf.reduce_all(tf.stack([
            tf.reduce_all(tf.math.is_finite(g)) for g in gradients]))
        return {'loss': loss, 'forward_loss': forward, 'reverse_loss': reverse,
                'gradients': gradients, 'valid': valid}

    def _joint_step(self, latent, physical, log_weights):
        return self._apply_evaluation(self._evaluate_joint(latent, physical, log_weights), latent)

    def checkpoint(self):
        return {'schema': 'bayesfilter.neutra.joint_training.v1',
                'forward_weight': self.forward_weight, 'reverse_weight': self.reverse_weight,
                'teacher_id': self.teacher_id, 'base': super().checkpoint()}

    def restore(self, checkpoint):
        expected = ('bayesfilter.neutra.joint_training.v1', self.forward_weight,
                    self.reverse_weight, self.teacher_id)
        actual = tuple(checkpoint.get(k) for k in ('schema', 'forward_weight', 'reverse_weight', 'teacher_id'))
        if actual != expected:
            raise ValueError('joint objective or teacher mismatch')
        super().restore(checkpoint['base'])
