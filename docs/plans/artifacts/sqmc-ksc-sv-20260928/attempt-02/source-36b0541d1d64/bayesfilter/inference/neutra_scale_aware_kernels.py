"""Reusable TF/XLA numerical kernels for physical-width guarded IAF training.

The host trainer owns transactions, target dispatch, counters and artifacts.
These kernels own numerical work: batched maps/scores, Keras Adam, finite
checks, bounded damping and an independent frozen-representation replay.
They do not alter the reverse-KL objective or its positive log-Jacobian term.
All fixed Python loops below enumerate network layers/variables while tracing;
the dimension-sized inverse and variable-length damping use TensorFlow loops.
"""

from __future__ import annotations

import tensorflow as tf
from bayesfilter.inference import neutra_transport_core as _transport_core


METRIC_NAMES = ('zero', 'rms', 'p95', 'maximum', 'log_scale')


def finite_tensor(*values):
    """Return a tensor flag so XLA cannot discard a diagnostic Assert op."""
    flags = tuple(tf.reduce_all(tf.math.is_finite(tf.cast(x, tf.float64))) for x in values)
    return tf.reduce_all(tf.stack(flags))


def near_tensor(actual, expected, tolerance):
    return tf.reduce_all(tf.abs(actual-expected) < tolerance+tolerance*tf.abs(expected))


def width_metrics_tensor(before, after, lower, scales_before, scales_after):
    """Compute dimensionless dtheta L0^-T; row zero is separate from probes.

    RMS divides by probe count only, not parameter dimension. The p95 is the
    nearest-rank order statistic. Finiteness is returned explicitly, including
    intermediate displacements, so an invalid candidate cannot pass under XLA.
    """
    delta = tf.transpose(tf.linalg.triangular_solve(lower, tf.transpose(after-before)))
    lengths = tf.linalg.norm(delta, axis=-1)
    rows = lengths[1:]
    rank = tf.cast(tf.math.ceil(.95*tf.cast(tf.shape(rows)[0], tf.float64)), tf.int32)-1
    metrics = tf.stack((lengths[0], tf.sqrt(tf.reduce_mean(tf.square(rows))),
                        tf.gather(tf.sort(rows), rank), tf.reduce_max(rows),
                        tf.reduce_max(tf.abs(scales_after-scales_before))))
    return metrics, finite_tensor(before, after, lower, scales_before, scales_after, delta, metrics)


class ScaleAwareNumericalKernels:
    """One compiled kernel set per trainer; replay buffers are never learned.

    The canonical artifact reader builds independent frozen network components
    once. A compiled copy refreshes their parameters before each inverse check.
    Reusing these buffers avoids JSON serialization, model reconstruction and
    constant-captured graph recompilation on every accepted optimizer update.
    Actual saved artifacts still undergo fresh reader/replay checks at handoff.
    """

    def __init__(self, trainer):
        self.trainer = trainer
        self.jit_compile = bool(trainer._base.config.jit_compile)
        shape = tf.TensorSpec([None, trainer.dimension], tf.float64)
        self._frozen = trainer._frozen()
        replay_variables = []
        for component in self._frozen.components:
            if not hasattr(component, 'weights'):
                continue
            component.weights = tuple(tf.Variable(v, trainable=False) for v in component.weights)
            component.biases = tuple(tf.Variable(v, trainable=False) for v in component.biases)
            for weight, bias in zip(component.weights, component.biases, strict=True):
                replay_variables.extend((weight, bias))
            for name in ('anchor_shift_weight', 'anchor_shift_bias', 'anchor_scale_raw'):
                value = getattr(component, name)
                if value is not None:
                    variable = tf.Variable(value, trainable=False)
                    setattr(component, name, variable)
                    replay_variables.append(variable)
        self.replay_variables = tuple(replay_variables)
        if tuple(v.shape for v in self.replay_variables) != tuple(v.shape for v in trainer.variables):
            raise ValueError('independent replay parameter inventory differs')
        self.map = self.compile(self._map_impl, [shape])
        self.snapshot = self.compile(lambda: (tuple(tf.identity(v) for v in trainer.variables),
                                             tuple(tf.identity(v) for v in trainer.optimizer.variables)), [])
        self.restore = self.compile(self._restore_impl)
        self.gradient = self.compile(self._gradient_impl, [shape, tf.TensorSpec([None], tf.float64),
                                                          shape, tf.TensorSpec([], tf.int32)])
        self.aggregate = self.compile(self._aggregate_impl)
        self.proposals = tuple(self._proposal_kernel(stage) for stage in (1, 2, 3))
        self.damp = self.compile(self._damp_impl)
        self.inverse = self.compile(self._inverse_impl, [shape])
        self.frozen_map = self.compile(self._frozen._forward_and_logdet, [shape])
        self.replay = self.compile(self._replay_impl, [shape, shape, shape,
            tf.TensorSpec([None], tf.float64), tf.TensorSpec([None], tf.float64)])
        self.check_batch = self.compile(self._check_batch_impl, [shape, shape,
            tf.TensorSpec([None], tf.float64), shape, tf.TensorSpec([None], tf.bool)])
        self.width = self.compile(lambda a, b, c, d: width_metrics_tensor(a, b, trainer.lower, c, d))
        self._external_frozen = None
        self._external_inverse = None

    def compile(self, function, signature=None):
        return tf.function(function, input_signature=signature, autograph=False,
                           jit_compile=self.jit_compile, reduce_retracing=signature is None)

    def _map_impl(self, z):
        values = z
        logdet = tf.zeros(tf.shape(z)[:-1], z.dtype)
        stage_scales = []
        valid = finite_tensor(z, *self.trainer.variables)
        for component in self.trainer._base.transport.components:
            if hasattr(component, '_network_with_diagnostics'):
                scale, shift, logits, hidden = component._network_with_diagnostics(values)
                exp_scale = tf.exp(scale)
                values, increment = _transport_core.iaf_apply(values, scale, shift)
                logdet = logdet+increment
                stage_scales.append(scale)
                valid &= finite_tensor(values, logdet, scale, logits, hidden, exp_scale)
                valid &= tf.reduce_all(exp_scale > 0)
            else:
                values, increment = component.forward_and_logdet(values)
                logdet = logdet+increment
        scales = tf.concat(stage_scales, axis=-1)
        return values, logdet, scales, valid & finite_tensor(values, logdet, scales)

    def _gradient_impl(self, z, value, score, count):
        mask = tf.cast(tf.range(tf.shape(z)[0]) < count, tf.float64)
        return self.trainer._base._external_gradients_impl(z, value, score, mask)

    def _aggregate_impl(self, raw, total):
        total = tf.cast(total, tf.float64)
        gradients = tuple(tf.add_n([r[4+i] for r in raw])/total for i in range(len(self.trainer.variables)))
        loss = tf.add_n([r[0] for r in raw])/total
        return loss, gradients, finite_tensor(loss, *gradients)

    def _proposal_kernel(self, stage):
        # Stage is an architectural constant: exactly three compiled programs,
        # each using the unchanged Keras Adam implementation. Inactive slots
        # receive no optimizer update, including momentum decay.
        def apply(gradients):
            model = self.trainer
            model.optimizer.apply_gradients([
                (tf.clip_by_norm(g, model.policy.gradient_clip_norm), v)
                for g, v, s in zip(gradients, model.variables, model.variable_stages, strict=True) if s <= stage])
            proposal = tuple(tf.identity(v) for v in model.variables)
            return proposal, finite_tensor(*proposal, *model.optimizer.variables), tf.identity(model.optimizer.iterations)
        return self.compile(apply)

    def _restore_impl(self, variables, slots):
        for target, value in zip(self.trainer.variables, variables, strict=True):
            target.assign(value)
        for target, value in zip(self.trainer.optimizer.variables, slots, strict=True):
            target.assign(value)
        return tf.constant(True)

    def _damp_impl(self, previous, proposal, before, scales_before, active_stages):
        model = self.trainer
        maximum = model.policy.maximum_halvings+1
        limits = tf.constant((model.policy.zero_limit, model.policy.rms_limit,
            model.policy.quantile_limit, model.policy.maximum_limit, model.policy.log_scale_limit), tf.float64)
        history = tf.zeros([maximum, 5], tf.float64)

        def body(index, accepted, valid, history):
            fraction = tf.math.pow(tf.constant(2., tf.float64), -tf.cast(index, tf.float64))
            for variable, old, new, stage in zip(model.variables, previous, proposal, model.variable_stages, strict=True):
                # Preserve the exact full-step proposal (no subtract/add roundoff).
                value = tf.cond(index == 0, lambda new=new: new,
                                lambda old=old, new=new: old+fraction*(new-old))
                variable.assign(tf.where(stage <= active_stages, value, old))
            after, _ld, scales_after, map_valid = self._map_impl(model.guard)
            metrics, metric_valid = width_metrics_tensor(before, after, model.lower, scales_before, scales_after)
            valid = map_valid & metric_valid
            accepted = valid & tf.reduce_all(metrics <= limits)
            history = tf.tensor_scatter_nd_update(history, tf.reshape(index, [1, 1]), metrics[None])
            return index+1, accepted, valid, history

        return tf.while_loop(lambda i, accepted, valid, h: (i < maximum) & ~accepted & valid,
            body, (tf.constant(0), tf.constant(False), tf.constant(True), history),
            parallel_iterations=1, maximum_iterations=maximum)

    def _check_inverse(self, probes, frozen):
        model = self.trainer
        theta, ld, _scales, map_valid = self._map_impl(probes)
        reconstructed, replay_ld = frozen._forward_and_logdet(probes)
        inverse = frozen.inverse_theta_to_z_batch(theta)
        error = tf.transpose(tf.linalg.triangular_solve(model.lower, tf.transpose(reconstructed-theta)))
        tol = model.policy.replay_tolerance
        return (map_valid & finite_tensor(reconstructed, inverse, replay_ld, error)
                & near_tensor(error, tf.zeros_like(error), tol)
                & near_tensor(inverse, probes, tol) & near_tensor(replay_ld, ld, tol))

    def _inverse_impl(self, probes):
        for target, value in zip(self.replay_variables, self.trainer.variables, strict=True):
            target.assign(value)
        return self._check_inverse(probes, self._frozen)

    def check_external_inverse(self, probes, frozen):
        # Used only at explicit artifact handoff, never for the per-update check.
        if self._external_frozen is not frozen:
            self._external_frozen = frozen
            self._external_inverse = self.compile(lambda rows: self._check_inverse(rows, frozen),
                [tf.TensorSpec([None, self.trainer.dimension], tf.float64)])
        return self._external_inverse(probes)

    def _replay_impl(self, probes, score, replay_score, value, replay_value):
        model = self.trainer
        with tf.GradientTape() as tape:
            tape.watch(probes)
            theta, ld = model._base.transport.forward_and_logdet(probes)
            surrogate = tf.reduce_sum(theta*tf.stop_gradient(score), axis=-1)+ld
        actual_score = tape.gradient(surrogate, probes, output_gradients=tf.ones_like(surrogate))
        replay_total = self._frozen.pullback_score_batch(probes, replay_score)
        replay_total += self._frozen.log_abs_det_jacobian_score_batch(probes)
        return (finite_tensor(actual_score, replay_total, value+ld)
                & near_tensor(replay_value, value, model.policy.replay_tolerance)
                & near_tensor(tf.matmul(replay_score, model.lower), tf.matmul(score, model.lower), model.policy.score_tolerance)
                & near_tensor(replay_total, actual_score, model.policy.score_tolerance))

    @staticmethod
    def _check_batch_impl(theta, supplied, value, score, eligible):
        return finite_tensor(supplied, value, score) & tf.reduce_all(eligible) & tf.reduce_all(theta == supplied)

    @staticmethod
    def metrics_payload(values):
        return dict(zip(METRIC_NAMES, (float(v) for v in values), strict=True))
