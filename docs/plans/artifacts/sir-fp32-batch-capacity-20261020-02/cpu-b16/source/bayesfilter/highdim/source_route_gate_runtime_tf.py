"""XLA numerical work for the existing P72 support and admission gates.

The host assembles reasons and provenance from these results. No threshold,
cloud selection, or fitted-TT numerical rule is changed by this boundary.
"""

from collections import OrderedDict
from functools import lru_cache

import tensorflow as tf

from bayesfilter.ops.fixed_signature_tf import fixed_signature_function

D = tf.float64
_LINE_PROGRAMS = OrderedDict()


@fixed_signature_function(floating_dtype=D)
def finite_tensor(values):
    return tf.reduce_all(tf.math.is_finite(values))


@fixed_signature_function(floating_dtype=D)
def support_statistics(local, fit):
    """Preserve Euclidean coverage, upper median and saturation diagnostics."""
    point_count, fit_count = local.shape[1], fit.shape[1]
    local_finite = tf.reduce_all(tf.math.is_finite(local))
    fit_finite = tf.reduce_all(tf.math.is_finite(fit))
    available = local_finite & fit_finite & (point_count > 0) & (fit_count > 0)

    def distances():
        pairwise = tf.norm(local[:, :, None] - fit[:, None, :], axis=0)
        nearest = tf.reduce_min(pairwise, axis=1)
        fit_pairwise = tf.norm(fit[:, :, None] - fit[:, None, :], axis=0)
        leave_one_out = (tf.reduce_min(fit_pairwise + tf.eye(fit_count, dtype=D)
                                      * tf.constant(1e300, D), axis=1)
                         if fit_count > 1 else tf.zeros([fit_count], D))
        median = tf.where(tf.reduce_all(tf.math.is_finite(nearest)),
                          tf.sort(nearest)[point_count // 2], tf.constant(float("inf"), D))
        return (tf.reduce_min(nearest), median, tf.reduce_max(nearest),
                tf.reduce_max(leave_one_out))

    def unavailable():
        missing = tf.constant(float("nan"), D)
        return missing, missing, missing, missing

    # Empty shapes cannot trace the median's indexed branch.
    statistics = (tf.cond(available, distances, unavailable)
                  if point_count > 0 and fit_count > 0 else unavailable())
    saturated = (tf.reduce_mean(tf.cast(tf.reduce_any(tf.abs(local) >= 1.0, axis=0), D))
                 if point_count > 0 else tf.constant(float("nan"), D))
    return local_finite, fit_finite, available, *statistics, saturated


def _maximum(values):
    """Retain eager ReduceMax's NaN result across the XLA reduction boundary."""
    return tf.where(tf.reduce_any(tf.math.is_nan(values)), tf.constant(float("nan"), D),
                    tf.reduce_max(values))


def line_probe_program(fitted_tt, point_shape, start_shape, index_shape, endpoint_mode, *,
                       jit_compile=True):
    """Bind one complete prediction/reduction graph to the frozen fitted TT."""
    key = (id(fitted_tt), tuple(point_shape), tuple(start_shape), tuple(index_shape),
           endpoint_mode, bool(jit_compile))
    if key not in _LINE_PROGRAMS:
        point_count = point_shape[1]
        signature = (tf.TensorSpec(point_shape, D), tf.TensorSpec([point_count], D),
                     tf.TensorSpec(start_shape, D), tf.TensorSpec(index_shape, tf.int32),
                     tf.TensorSpec([], D))

        @tf.function(input_signature=signature, jit_compile=jit_compile, autograph=False)
        def evaluate(points, targets, starts, indices, scale):
            predictions = tf.convert_to_tensor(fitted_tt.evaluate(tf.transpose(points)), D)
            if predictions.shape != targets.shape:
                raise ValueError("line_probe_prediction: INVALID_SHAPE")
            residual = tf.abs(predictions - targets)
            max_abs = _maximum(tf.abs(predictions))
            max_residual = _maximum(residual)
            rms_residual = tf.sqrt(tf.reduce_mean(tf.square(residual)))
            growth_ratio = tf.constant(float("nan"), D)
            valid_indices = tf.constant(True)
            if endpoint_mode != "none":
                start_scale = tf.maximum(tf.abs(starts), scale)
                if endpoint_mode == "maximum":
                    denominators = tf.ones_like(predictions) * _maximum(start_scale)
                else:
                    flat_indices = tf.reshape(indices, [-1])
                    valid_indices = tf.reduce_all((flat_indices >= 0) & (flat_indices < starts.shape[0]))
                    denominators = tf.gather(start_scale, flat_indices)
                growth_ratio = _maximum(tf.abs(predictions) / denominators)
            return predictions, max_abs, max_residual, rms_residual, growth_ratio, valid_indices

        _LINE_PROGRAMS[key] = fitted_tt, evaluate
        if len(_LINE_PROGRAMS) > 32:
            _LINE_PROGRAMS.popitem(last=False)
    _LINE_PROGRAMS.move_to_end(key)
    return _LINE_PROGRAMS[key][1]


@lru_cache(maxsize=32)
def spectrum_rank_program(shapes, *, jit_compile=True):
    """Reduce heterogeneous recorded spectra in one native TensorFlow loop."""
    signature = (tuple(tf.TensorSpec(shape, D) for shape in shapes), tf.TensorSpec([], D))

    @tf.function(input_signature=signature, jit_compile=jit_compile, autograph=False)
    def evaluate(spectra, tolerance):
        def branch(index):
            def reduce_spectrum():
                values = tf.reshape(spectra[index], [-1])
                if values.shape[0] == 0:
                    return tf.constant(float("nan"), D), tf.constant(False)
                valid = tf.reduce_all(tf.math.is_finite(values))
                rank = tf.reduce_sum(tf.cast(values > tolerance * tf.reduce_max(values), D))
                return rank, valid
            return reduce_spectrum

        branches = tuple(branch(index) for index in range(len(shapes)))
        ranks = tf.zeros([len(shapes)], D)
        valid = tf.zeros([len(shapes)], tf.bool)

        def step(index, ranks, valid):
            rank, available = tf.switch_case(index, branches)
            return (index + 1, tf.tensor_scatter_nd_update(ranks, [[index]], [rank]),
                    tf.tensor_scatter_nd_update(valid, [[index]], [available]))

        if not shapes:
            return ranks, valid
        _, ranks, valid = tf.while_loop(lambda index, *_: index < len(shapes), step,
            (tf.constant(0), ranks, valid), maximum_iterations=len(shapes), parallel_iterations=1)
        return ranks, valid

    return evaluate
