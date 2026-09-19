"""Compiled numerical kernels for the reachable source-route preparation.

The surrounding source-route code still owns validation, provenance and the
fixed-HMC adaptation records.  These kernels own seeded Gaussian generation,
model push/likelihood evaluation and deterministic resampling.  Their tensor
schemas are fixed before tracing and their default execution target is XLA.
The TensorFlow Generator stream is represented by the repository Philox
counter implementation, preserving the existing ``Generator.from_seed`` draw
sequence exactly.
"""

from collections import OrderedDict
from functools import lru_cache

import tensorflow as tf

from bayesfilter.ops.fixed_signature_tf import fixed_signature_function
from bayesfilter.ops.generator_stream_tf import (
    generator_seed_state,
    normal_call_program,
)

D = tf.float64
_PUSH_PROGRAMS = OrderedDict()


@fixed_signature_function(floating_dtype=D)
def normalized_set_weights(weights):
    """P72 normalization with the existing host-rejection predicate."""
    total = tf.reduce_sum(weights)
    valid = (tf.reduce_all(tf.math.is_finite(weights))
             & ~tf.reduce_any(weights < 0.0) & ~(total <= 0.0))
    return weights / total, valid


@fixed_signature_function(floating_dtype=D)
def fit_guard_arrays(fit, guard, fit_targets, guard_targets, fit_weights, guard_weights, alpha):
    """Assemble the existing fit/guard objective with separate set masses."""
    return (tf.transpose(tf.concat([fit, guard], axis=1)),
            tf.concat([fit_targets, guard_targets], axis=0),
            tf.concat([fit_weights, alpha * guard_weights], axis=0))


@lru_cache(maxsize=32)
def guard_line_program(dimension, fit_count, guard_count, fraction_count, *, jit_compile=True):
    """Construct the P72 fixed-size line cloud before exact-key decisions."""
    raw_count = 3 * int(fraction_count)

    @tf.function(input_signature=[tf.TensorSpec([dimension, fit_count], D),
        tf.TensorSpec([dimension, guard_count], D), tf.TensorSpec([fraction_count], D)],
        jit_compile=jit_compile, autograph=False)
    def evaluate(fit, guard, fractions):
        center = tf.reduce_mean(fit, axis=1, keepdims=True)
        distances = tf.norm(guard - center, axis=0)
        selected = tf.gather(tf.argsort(distances), [0, guard_count // 2, guard_count - 1])
        endpoints = tf.gather(guard, selected, axis=1)
        starts = tf.repeat(center, repeats=3, axis=1)
        # The original concat orders fractions first, endpoints second.
        raw = tf.reshape((1.0 - fractions[None, :, None]) * starts[:, None, :]
                         + fractions[None, :, None] * endpoints[:, None, :],
                         [dimension, raw_count])
        return tf.stop_gradient(raw), selected

    return evaluate


@lru_cache(maxsize=32)
def guard_line_unique_program(dimension, raw_count, *, jit_compile=True):
    """Stable first-occurrence decisions on the realized binary64 columns.

    The old 17-digit key round-trips finite binary64 values. Equality therefore
    includes signed-zero equivalence, while NaN columns remain unequal. This
    separate compiled stage prevents the compiler from recomputing interpolated
    coordinates with different rounding inside different comparisons.
    """
    @tf.function(input_signature=[tf.TensorSpec([dimension, raw_count], D)],
                 jit_compile=jit_compile, autograph=False)
    def evaluate(raw):
        same = tf.reduce_all(raw[:, :, None] == raw[:, None, :], axis=0)
        indices = tf.range(raw_count)
        first = ~tf.reduce_any(same & (indices[None, :] < indices[:, None]), axis=1)
        order = tf.argsort(tf.where(first, indices, raw_count + indices))
        count = tf.reduce_sum(tf.cast(first, tf.int32))
        return order, count

    return evaluate


@lru_cache(maxsize=32)
def guard_line_selection_program(dimension, raw_count, selected_count, *, jit_compile=True):
    """Resolve a public output schema after the compiled design returns its count.

    The old public API reconstructed constants from a host array. Its line
    cloud was a frozen design, so preserve that existing derivative boundary.
    """
    @tf.function(input_signature=[tf.TensorSpec([dimension, raw_count], D),
        tf.TensorSpec([raw_count], tf.int32)], jit_compile=jit_compile, autograph=False)
    def evaluate(raw, order):
        indices = order[:selected_count]
        return tf.stop_gradient(tf.gather(raw, indices, axis=1)), tf.math.floormod(indices, 3)

    return evaluate


@lru_cache(maxsize=32)
def uniform_log_weights_program(sample_count, *, jit_compile=True):
    """Compile the existing equal-weight prior/retained-prefix rule."""

    count = int(sample_count)
    if count <= 0:
        raise ValueError("sample_count must be positive")

    @tf.function(input_signature=[], jit_compile=jit_compile, autograph=False)
    def evaluate():
        return tf.fill([count], -tf.math.log(tf.cast(count, D)))

    return evaluate


@lru_cache(maxsize=32)
def weighted_mean_target_program(sample_count, *, jit_compile=True):
    """Compile the existing fitting initializer and its input-validity flag.

    Keep the original positive-target/nonnegative-weight rule. The host wrapper
    retains shape checks and the rejection message. No output clipping, new
    finite-output veto or change to the weighted reduction is introduced.
    """

    count = int(sample_count)
    if count < 0:
        raise ValueError("sample_count must be nonnegative")

    @tf.function(
        input_signature=[tf.TensorSpec([count], D), tf.TensorSpec([count], D)],
        jit_compile=jit_compile,
        autograph=False,
    )
    def evaluate(targets, weights):
        total_weight = tf.reduce_sum(weights)
        valid = (tf.reduce_all(tf.math.is_finite(targets))
                 & tf.reduce_all(tf.math.is_finite(weights))
                 & ~tf.reduce_any(targets <= 0.0)
                 & ~tf.reduce_any(weights < 0.0)
                 & ~(total_weight <= 0.0))
        return tf.reduce_sum(weights * targets) / total_weight, valid

    return evaluate


@lru_cache(maxsize=32)
def normal_matrix_program(sample_count, dimension, *, jit_compile=True):
    """Return a fixed XLA program for one TensorFlow Generator matrix draw."""

    count = int(sample_count)
    width = int(dimension)
    if count <= 0 or width <= 0:
        raise ValueError("sample_count and dimension must be positive")
    calls = normal_call_program(1, count * width, 0, jit_compile=jit_compile)

    @tf.function(
        input_signature=[tf.TensorSpec([3], tf.uint64)],
        jit_compile=jit_compile,
        autograph=False,
    )
    def generate(seed_state):
        values, _unused = calls(seed_state)
        return tf.reshape(values[0], [count, width])

    return generate


@lru_cache(maxsize=32)
def prior_sample_program(parameter_dim, state_dim, sample_count, *, jit_compile=True):
    """Compile the exact prior-cloud construction for a fixed tensor schema."""

    parameters = int(parameter_dim)
    states = int(state_dim)
    count = int(sample_count)
    if parameters < 0 or states <= 0 or count <= 0:
        raise ValueError("invalid prior sample dimensions")
    noise_program = normal_matrix_program(count, states, jit_compile=jit_compile)

    @tf.function(
        input_signature=[
            tf.TensorSpec([states], D),
            tf.TensorSpec([states, states], D),
            tf.TensorSpec([3], tf.uint64),
        ],
        jit_compile=jit_compile,
        autograph=False,
    )
    def generate(initial_mean, initial_covariance, seed_state):
        noise = noise_program(seed_state)
        initial_chol = tf.linalg.cholesky(initial_covariance)
        states_value = initial_mean[tf.newaxis, :] + tf.linalg.matmul(
            noise, initial_chol, transpose_b=True
        )
        theta = tf.zeros([parameters, count], D)
        return tf.concat([theta, tf.transpose(states_value)], axis=0)

    return generate


def source_push_program(model, parameter_dim, state_dim, sample_count, time_index, *,
                        jit_compile=True):
    """Compile one source push and likelihood update, retaining model identity."""

    parameters = int(parameter_dim)
    states = int(state_dim)
    count = int(sample_count)
    time = int(time_index)
    key = (id(model), parameters, states, count, time, bool(jit_compile))
    observation_dimension = (
        model.observation_dimension()
        if hasattr(model, "observation_dimension")
        else model.observation_dim()
    )
    if key not in _PUSH_PROGRAMS:
        @tf.function(
            input_signature=[
                tf.TensorSpec([parameters + states, count], D),
                tf.TensorSpec([count], D),
                tf.TensorSpec([count, states], D),
                tf.TensorSpec([observation_dimension], D),
            ],
            jit_compile=jit_compile,
            autograph=False,
        )
        def evaluate(previous_samples, previous_log_weights, transition_noise, observation):
            theta = tf.transpose(previous_samples[:parameters, :])
            previous_state = tf.transpose(previous_samples[parameters:, :])
            pushed = model.transition_push_from_standard_normal(
                theta, previous_state, transition_noise, t=time
            )
            next_state = tf.convert_to_tensor(pushed, dtype=D)
            if next_state.shape != previous_state.shape:
                raise ValueError("transition output: INVALID_SHAPE")
            propagated = tf.concat([theta, next_state], axis=1)
            current_state = next_state
            log_likelihood = model.observation_log_density(
                theta, current_state, observation, t=time
            )
            log_likelihood = tf.convert_to_tensor(log_likelihood, dtype=D)
            if log_likelihood.shape != previous_log_weights.shape:
                raise ValueError("log_likelihood: INVALID_SHAPE")
            normalized = previous_log_weights + log_likelihood
            normalized = normalized - tf.reduce_logsumexp(normalized)
            propagated_columns = tf.transpose(propagated)
            augmented = tf.concat([theta, next_state, previous_state], axis=1)
            return (
                propagated_columns,
                normalized,
                tf.transpose(augmented),
                log_likelihood,
            )

        _PUSH_PROGRAMS[key] = (model, evaluate)
        if len(_PUSH_PROGRAMS) > 16:
            _PUSH_PROGRAMS.popitem(last=False)
    _PUSH_PROGRAMS.move_to_end(key)
    return _PUSH_PROGRAMS[key][1]


@lru_cache(maxsize=32)
def deterministic_weighted_resample_program(dimension, sample_count, *, jit_compile=True):
    """Compile the fixed-midpoint weighted resampling rule."""

    rows = int(dimension)
    count = int(sample_count)
    if rows <= 0 or count <= 0:
        raise ValueError("dimension and sample_count must be positive")

    @tf.function(
        input_signature=[tf.TensorSpec([rows, count], D), tf.TensorSpec([count], D)],
        jit_compile=jit_compile,
        autograph=False,
    )
    def resample(samples, log_weights):
        normalized = log_weights - tf.reduce_logsumexp(log_weights)
        weights = tf.exp(normalized)
        positions = (tf.cast(tf.range(count), D) + tf.constant(.5, D)) / tf.cast(count, D)
        cdf = tf.concat([tf.cumsum(weights)[:-1], tf.ones([1], D)], axis=0)
        indices = tf.searchsorted(cdf, positions, side="left", out_type=tf.int32)
        indices = tf.minimum(indices, tf.fill([count], count - 1))
        return tf.gather(samples, indices, axis=1), indices

    return resample


@lru_cache(maxsize=32)
def coordinate_transform_program(dimension, sample_count, *, jit_compile=True):
    """Compile affine solve, clipping and fit-cloud reconstruction."""

    width = int(dimension)
    count = int(sample_count)
    if width <= 0 or count <= 0:
        raise ValueError("dimension and sample_count must be positive")

    @tf.function(
        input_signature=[
            tf.TensorSpec([width, width], D),
            tf.TensorSpec([width], D),
            tf.TensorSpec([width, count], D),
        ],
        jit_compile=jit_compile,
        autograph=False,
    )
    def transform(matrix, center, samples):
        local_unclipped = tf.linalg.solve(matrix, samples - center[:, tf.newaxis])
        clipped_mask = tf.abs(local_unclipped) > tf.constant(1.0, D)
        clip_fraction = tf.reduce_mean(tf.cast(clipped_mask, D))
        local_points = tf.clip_by_value(local_unclipped, -1.0, 1.0)
        physical_points = tf.linalg.matmul(matrix, local_points) + center[:, tf.newaxis]
        return local_unclipped, clipped_mask, clip_fraction, local_points, physical_points

    return transform


@lru_cache(maxsize=32)
def target_values_program(sample_count, *, jit_compile=True):
    """Compile local negative-log conversion and source stability shifting."""

    count = int(sample_count)
    if count <= 0:
        raise ValueError("sample_count must be positive")

    @tf.function(
        input_signature=[tf.TensorSpec([count], D), tf.TensorSpec([], D)],
        jit_compile=jit_compile,
        autograph=False,
    )
    def evaluate(negative_log_physical, log_abs_det):
        local_negative_log = negative_log_physical - log_abs_det
        shift = tf.reduce_min(local_negative_log)
        target_values = tf.exp(-0.5 * (local_negative_log - shift))
        valid = (tf.reduce_all(tf.math.is_finite(local_negative_log))
                 & tf.math.is_finite(shift)
                 & tf.reduce_all(tf.math.is_finite(target_values)))
        return local_negative_log, shift, target_values, valid

    return evaluate


@lru_cache(maxsize=32)
def target_values_with_shift_program(sample_count, *, jit_compile=True):
    """Compile local negative-log conversion using a frozen diagnostic shift."""

    count = int(sample_count)
    if count <= 0:
        raise ValueError("sample_count must be positive")

    @tf.function(
        input_signature=[
            tf.TensorSpec([count], D),
            tf.TensorSpec([], D),
            tf.TensorSpec([], D),
        ],
        jit_compile=jit_compile,
        autograph=False,
    )
    def evaluate(negative_log_physical, log_abs_det, shift):
        local_negative_log = negative_log_physical - log_abs_det
        target_values = tf.exp(-0.5 * (local_negative_log - shift))
        valid = (tf.reduce_all(tf.math.is_finite(local_negative_log))
                 & tf.math.is_finite(shift)
                 & tf.reduce_all(tf.math.is_finite(target_values)))
        return local_negative_log, target_values, valid

    return evaluate


def seed_state(seed):
    """Expose the repository's exact Generator.from_seed state conversion."""

    return generator_seed_state(int(seed))
