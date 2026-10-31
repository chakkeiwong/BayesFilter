"""Native numerical preparation for the existing fixed source-route adapters.

Recentring preserves the repository's adaptation of author computeL.m:24--48.
Reference designs and seeded TT channels are repository extensions. Compiling
them does not establish Zhao-Cui source-faithfulness or scientific admission.
"""

from collections import OrderedDict
from functools import lru_cache

import tensorflow as tf

from bayesfilter.ops.fixed_signature_tf import fixed_signature_function

D = tf.float64
_COORDINATE_CACHE = OrderedDict()


def coordinate_frame_program(model):
    key = id(model)
    if key in _COORDINATE_CACHE:
        _COORDINATE_CACHE.move_to_end(key)
        return _COORDINATE_CACHE[key][1]
    dimension = model.state_dim()

    @tf.function(input_signature=[tf.TensorSpec([], tf.int32), tf.TensorSpec([dimension], D),
        tf.TensorSpec([dimension, dimension], D), tf.TensorSpec([dimension, dimension], D)],
        jit_compile=True, autograph=False)
    def evaluate(time_index, initial, process_covariance, initial_covariance):
        current = model.transition_mean(initial)[0]

        def step(index, current, previous):
            del previous
            return index + 1, model.transition_mean(current)[0], current

        _, current, previous = tf.while_loop(lambda index, *_: index < time_index, step,
            (tf.constant(1), current, initial), maximum_iterations=time_index,
            parallel_iterations=1)
        scale = tf.concat([tf.sqrt(tf.linalg.diag_part(process_covariance)),
                           tf.sqrt(tf.linalg.diag_part(initial_covariance))], axis=0)
        return tf.concat([current, previous], axis=0), tf.linalg.diag(scale)

    _COORDINATE_CACHE[key] = (model, evaluate)
    if len(_COORDINATE_CACHE) > 16:
        _COORDINATE_CACHE.popitem(last=False)
    return evaluate


@lru_cache(maxsize=16)
def reference_points_program(sample_count, dimension, unit):
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    @tf.function(input_signature=[], jit_compile=True, autograph=False)
    def evaluate():
        if unit:
            base = tf.linspace(tf.constant(.15, D), tf.constant(.85, D), dimension)
            shift = tf.cast(tf.range(sample_count), D) / tf.cast(max(sample_count, 1), D)
            return tf.math.floormod(base[:, None] + .13 * shift[None, :], .8) + .1
        base = tf.linspace(tf.constant(-.75, D), tf.constant(.75, D), dimension)
        shift = tf.cast(tf.range(sample_count), D) / tf.cast(max(sample_count - 1, 1), D)
        return tf.math.floormod(base[:, None] + .17 * shift[None, :] + 1., 2.) - 1.

    return evaluate


@lru_cache(maxsize=16)
def initial_cores_program(ranks, basis_dim, seeded):
    dimension, rank = len(ranks) - 1, max(ranks)

    @tf.function(input_signature=[tf.TensorSpec([], D), tf.TensorSpec([], D)],
                 jit_compile=True, autograph=False)
    def evaluate(constant, epsilon_per_channel):
        axis = tf.range(dimension)[:, None]
        channel = tf.range(rank)[None, :]
        zero = tf.one_hot(0, rank, dtype=D)
        coefficient = tf.one_hot(0, basis_dim, dtype=D)
        values = tf.where(axis[:, 0] == 0, constant, tf.constant(1., D))[:, None, None, None]
        values = values * zero[None, :, None, None] * coefficient[None, None, :, None] * zero[None, None, None, :]
        if seeded:
            left, right = tf.constant(ranks[:-1])[:, None], tf.constant(ranks[1:])[:, None]
            indices = tf.where(basis_dim > 1, 1 + tf.math.floormod(
                axis + channel - 1, max(basis_dim - 1, 1)), 0)
            basis = tf.one_hot(indices, basis_dim, dtype=D)
            diagonal = tf.cast((channel > 0) & (channel < tf.minimum(left, right)), D)
            values += basis[:, :, :, None] * tf.eye(rank, dtype=D)[None, :, None, :] * diagonal[:, :, None, None]
            first = tf.cast((axis == 0) & (channel > 0) & (channel < right), D)
            values += zero[None, :, None, None] * tf.transpose(basis * first[:, :, None], [0, 2, 1])[:, None, :, :] * (constant * epsilon_per_channel)
            last = tf.cast((axis == dimension - 1) & (channel > 0) & (channel < left), D)
            values += basis[:, :, :, None] * last[:, :, None, None] * zero[None, None, None, :]
        return values

    return evaluate


@fixed_signature_function(floating_dtype=D, tensor_dtypes={"channel_counts": tf.int32,
                                                         "fit_rank": tf.int32})
def channel_activity(packed, channel_counts, fit_rank, absolute_tolerance, relative_tolerance):
    dimension = packed.shape[0]
    scores = tf.sqrt(tf.reduce_sum(tf.square(packed[:-1]), axis=[1, 2])) * tf.sqrt(
        tf.reduce_sum(tf.square(packed[1:]), axis=[2, 3]))
    present = tf.range(packed.shape[-1])[None, :] < channel_counts[:, None]
    reference = tf.reduce_max(scores[:, 0]) if dimension > 1 else tf.constant(0., D)
    valid_reference = tf.math.is_finite(reference) & (reference > 0.)
    threshold = tf.where(valid_reference, tf.maximum(absolute_tolerance,
                        relative_tolerance * reference), tf.constant(float("inf"), D))
    counts = tf.reduce_sum(tf.cast(present & (scores >= threshold), tf.int32), axis=0)
    indices = tf.range(packed.shape[-1])
    inactive = (indices > 0) & (indices < fit_rank) & (counts < max(1, (dimension + 2) // 4))
    return scores, reference, threshold, counts, inactive, valid_reference & ~tf.reduce_any(inactive)


def _quantile_scale(samples, weights, q):
    order = tf.argsort(samples, axis=1, stable=True)
    ordered = tf.gather(samples, order, batch_dims=1)
    cumulative = tf.cumsum(tf.gather(weights, order), axis=1)
    left = tf.argmax(tf.cast(cumulative > q, tf.int32), axis=1, output_type=tf.int32)
    right = tf.argmax(tf.cast(cumulative > 1. - q, tf.int32), axis=1, output_type=tf.int32)
    width = tf.gather(ordered, right, batch_dims=1) - tf.gather(ordered, left, batch_dims=1)
    normal_q = tf.sqrt(tf.constant(2., D)) * tf.math.erfinv(2. * q - 1.)
    return tf.maximum(-width / normal_q / 2., tf.constant(1e-12, D))


@fixed_signature_function(floating_dtype=D)
def quantile_scale(samples, weights, q):
    return _quantile_scale(samples, weights, q)


@fixed_signature_function(floating_dtype=D, static_parameters=("use_quantile_scale",))
def recenter(samples, log_weights, expansion_factor, jitter, q, minimum_ess, *, use_quantile_scale):
    keep = tf.reduce_all(tf.math.is_finite(samples), axis=0) & tf.math.is_finite(log_weights)
    clean = tf.where(keep[None, :], samples, tf.zeros_like(samples))
    log_weights = tf.where(keep, log_weights, tf.constant(float("-inf"), D))
    normalized = log_weights - tf.reduce_logsumexp(log_weights)
    weights = tf.exp(normalized)
    mean = tf.reduce_sum(clean * weights[None, :], axis=1)
    centered = tf.where(keep[None, :], clean - mean[:, None], tf.zeros_like(clean))
    covariance = tf.einsum("n,in,jn->ij", weights, centered, centered)
    covariance = .5 * (covariance + tf.transpose(covariance))
    matrix = tf.linalg.cholesky(covariance + tf.eye(samples.shape[0], dtype=D) * jitter)
    if use_quantile_scale:
        ess = 1. / tf.reduce_sum(tf.square(weights))

        def stretch():
            standardized = tf.linalg.triangular_solve(matrix, centered, lower=True)
            standardized = tf.where(keep[None, :], standardized, tf.constant(float("inf"), D))
            return matrix @ tf.linalg.diag(_quantile_scale(standardized, weights, q))

        matrix = tf.cond(ess > minimum_ess, stretch, lambda: matrix)
    return mean, matrix * expansion_factor, tf.reduce_any(keep)
