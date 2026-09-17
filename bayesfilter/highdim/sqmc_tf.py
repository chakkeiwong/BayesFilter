"""TensorFlow primitives for bounded SQMC mechanics experiments.

The Hilbert key follows the transpose construction of Skilling's algorithm.
All candidate-runtime operations stay in TensorFlow and are XLA compatible.
"""

from __future__ import annotations

import tensorflow as tf
import tensorflow_probability as tfp

Tensor = tf.Tensor
HILBERT_IMPLEMENTATION_ID = "skilling_transpose_tf_lexicographic_int30_v2"
STATE_MAP_ID = "componentwise_logistic_empirical_mean_std_floor_v2"
ANCESTOR_CDF_ID = "empirical_inverse_cdf_right_open_v1"
POINT_SET_ID = "tfp_owen2017_randomized_halton_v1"
ENDPOINT_POLICY_ID = "dtype_nextafter_open_unit_interval_v1"


def _seed(seed: int, salt: int) -> Tensor:
    modulus = 2_147_483_647
    return tf.constant([int(seed) % modulus, int(salt) % modulus], tf.int32)


def open_unit_interval(values: Tensor) -> Tensor:
    """Clip uniforms to the largest dtype-representable open unit interval."""

    values = tf.convert_to_tensor(values)
    zero = tf.zeros([], values.dtype)
    one = tf.ones([], values.dtype)
    return tf.clip_by_value(
        values,
        tf.math.nextafter(zero, one),
        tf.math.nextafter(one, zero),
    )


def calibrated_logistic_map(
    points: Tensor, location: Tensor, scale: Tensor
) -> tuple[Tensor, Tensor]:
    """Map real-valued states into the Hilbert unit cube."""

    points = tf.convert_to_tensor(points)
    location = tf.ensure_shape(tf.cast(location, points.dtype), [points.shape[-1]])
    scale = tf.ensure_shape(tf.cast(scale, points.dtype), [points.shape[-1]])
    tf.debugging.assert_positive(scale)
    mapped = tf.math.sigmoid((points - location[None, :]) / scale[None, :])
    guarded = open_unit_interval(mapped)
    saturation = tf.reduce_mean(
        tf.cast((mapped <= tf.cast(1.0e-6, points.dtype)) | (mapped >= tf.cast(1.0 - 1.0e-6, points.dtype)), points.dtype)
    )
    return guarded, saturation


def _hilbert_transpose_axes(unit_points: Tensor, *, bits: int) -> Tensor:
    """Return Skilling-transformed integer axes for rows in ``[0, 1)^d``."""

    if bits < 1 or bits > 20:
        raise ValueError("Hilbert bit depth must be between 1 and 20")
    unit_points = tf.convert_to_tensor(unit_points)
    dimension = unit_points.shape[-1]
    if dimension is None or dimension < 1:
        raise ValueError("Hilbert ordering requires a static positive dimension")
    levels = 1 << bits
    coordinates = tf.cast(
        tf.floor(open_unit_interval(unit_points) * tf.cast(levels, unit_points.dtype)),
        tf.int32,
    )
    axes = tf.transpose(coordinates)

    # Inverse undo/exchange transform from Skilling's Hilbert transpose.
    def exchange_level(q, axes):
        p = q - 1
        has_bit = tf.not_equal(tf.bitwise.bitwise_and(axes, q), 0)
        # At this bit level only the lower p bits are exchanged. The first
        # coordinate carries the last clear axis's lower bits, XOR p once
        # for every intervening set axis. Compute all such prefixes together.
        index = tf.range(dimension)
        prefix_index = tf.range(dimension + 1)
        clear_before = (
            tf.logical_not(has_bit)[None, :, :]
            & (index[None, :, None] > 0)
            & (index[None, :, None] < prefix_index[:, None, None])
        )
        last_clear = tf.reduce_max(tf.where(clear_before, index[None, :, None], 0), axis=1)
        source = tf.transpose(tf.gather(tf.transpose(axes), tf.transpose(last_clear), batch_dims=1))
        set_count = tf.concat([tf.zeros_like(axes[:1]), tf.cumsum(tf.cast(has_bit, tf.int32), axis=0)], axis=0)
        count_at_source = tf.transpose(tf.gather(tf.transpose(set_count), tf.transpose(last_clear + 1), batch_dims=1))
        intervening = set_count - tf.where(last_clear > 0, count_at_source, 0)
        lower = tf.bitwise.bitwise_xor(tf.bitwise.bitwise_and(source, p), tf.where(intervening % 2 != 0, p, 0))
        high = tf.bitwise.bitwise_and(axes, tf.bitwise.invert(p))
        exchanged = tf.where(has_bit, axes, tf.bitwise.bitwise_or(high, lower[:-1]))
        axes = tf.concat([tf.bitwise.bitwise_or(high[:1], lower[-1:]), exchanged[1:]], axis=0)
        return tf.bitwise.right_shift(q, 1), axes

    _, axes = tf.while_loop(
        lambda q, _: q > 1, exchange_level,
        (tf.constant(1 << (bits - 1)), axes), maximum_iterations=bits - 1,
    )
    shifts = tf.range(bits)[None, None, :]
    binary = tf.bitwise.bitwise_and(tf.bitwise.right_shift(axes[:, :, None], shifts), 1)
    prefix_bits = tf.cumsum(binary, axis=0) % 2
    axes = tf.reduce_sum(tf.bitwise.left_shift(prefix_bits, shifts), axis=2)
    correction_bits = tf.cumsum(prefix_bits[-1], axis=1, exclusive=True, reverse=True) % 2
    correction = tf.reduce_sum(tf.bitwise.left_shift(correction_bits, shifts[0]), axis=1)
    return tf.bitwise.bitwise_xor(axes, correction[None, :])


def hilbert_transpose_words(unit_points: Tensor, *, bits: int) -> Tensor:
    """Return arbitrary-width Hilbert keys as lexicographic 30-bit words."""

    axes = _hilbert_transpose_axes(unit_points, bits=bits)
    dimension = int(axes.shape[0])
    total_bits = bits * dimension
    positions = tf.range(total_bits)
    axis_bits = tf.bitwise.bitwise_and(
        tf.bitwise.right_shift(
            tf.gather(axes, positions % dimension),
            (bits - 1 - positions // dimension)[:, None],
        ), 1,
    )
    widths = tf.minimum(30, total_bits - 30 * (positions // 30))
    weighted = tf.bitwise.left_shift(axis_bits, (widths - 1 - positions % 30)[:, None])
    word_count = (total_bits + 29) // 30
    padded = tf.pad(tf.transpose(weighted), [[0, 0], [0, 30 * word_count - total_bits]])
    return tf.reduce_sum(tf.reshape(padded, [-1, word_count, 30]), axis=2)


def hilbert_integer_keys(unit_points: Tensor, *, bits: int) -> Tensor:
    """Return legacy packed keys for 2D/3D parity and compatibility tests."""

    unit_points = tf.convert_to_tensor(unit_points)
    dimension = unit_points.shape[-1]
    if dimension not in (2, 3):
        raise ValueError("packed Hilbert keys support dimensions 2 and 3")
    axes = _hilbert_transpose_axes(unit_points, bits=bits)

    positions = tf.range(bits * dimension)
    axis_bits = tf.cast(tf.bitwise.bitwise_and(
        tf.bitwise.right_shift(
            tf.gather(axes, positions % dimension),
            (bits - 1 - positions // dimension)[:, None],
        ), 1,
    ), tf.int64)
    return tf.reduce_sum(tf.bitwise.left_shift(
        axis_bits, tf.cast(bits * dimension - 1 - positions, tf.int64)[:, None],
    ), axis=0)


def hilbert_permutation(
    points: Tensor,
    location: Tensor,
    scale: Tensor,
    *,
    bits: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return stable Hilbert order, adjacent tie count, and saturation rate."""

    mapped, saturation = calibrated_logistic_map(points, location, scale)
    words = hilbert_transpose_words(mapped, bits=bits)
    order = tf.range(tf.shape(words)[0], dtype=tf.int32)
    def sort_word(word_index, order):
        local_order = tf.argsort(
            tf.gather(words[:, word_index], order), axis=0, stable=True
        )
        return word_index - 1, tf.gather(order, local_order)
    _, order = tf.while_loop(
        lambda word_index, _: word_index >= 0, sort_word,
        (tf.shape(words)[1] - 1, order), maximum_iterations=int(words.shape[1]),
    )
    sorted_words = tf.gather(words, order)
    ties = tf.reduce_sum(
        tf.cast(
            tf.reduce_all(tf.equal(sorted_words[1:], sorted_words[:-1]), axis=1),
            tf.int32,
        )
    )
    return order, ties, saturation


def inverse_cdf_ancestor_indices(sorted_uniforms: Tensor, weights: Tensor) -> Tensor:
    """Select empirical-distribution indices with right-open CDF intervals."""

    uniforms = open_unit_interval(tf.convert_to_tensor(sorted_uniforms))
    weights = tf.cast(tf.convert_to_tensor(weights), uniforms.dtype)
    tf.debugging.assert_non_negative(weights)
    tf.debugging.assert_near(tf.reduce_sum(weights), tf.ones([], weights.dtype))
    cumulative = tf.cumsum(weights)
    indices = tf.reduce_sum(
        tf.cast(uniforms[:, None] >= cumulative[None, :], tf.int32), axis=1
    )
    return tf.minimum(indices, tf.shape(weights)[0] - 1)


def randomized_halton_joint(
    *,
    num_particles: int,
    state_dimension: int,
    seed: int,
    salt: int,
    dtype: tf.dtypes.DType = tf.float32,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return raw joint points and rows sorted by their ancestor coordinate."""

    joint = tfp.mcmc.sample_halton_sequence(
        state_dimension + 1,
        num_results=num_particles,
        dtype=dtype,
        randomized=True,
        seed=_seed(seed, salt),
    )
    joint = open_unit_interval(joint)
    row_order = tf.argsort(joint[:, 0], stable=True)
    ranked = tf.gather(joint, row_order)
    return joint, ranked[:, 0], ranked[:, 1:]


def randomized_halton_gaussian(
    *,
    num_particles: int,
    dimension: int,
    seed: int,
    salt: int,
    dtype: tf.dtypes.DType = tf.float32,
) -> Tensor:
    """Return a randomized Halton cloud transformed to standard Gaussian."""

    uniforms = tfp.mcmc.sample_halton_sequence(
        dimension,
        num_results=num_particles,
        dtype=dtype,
        randomized=True,
        seed=_seed(seed, salt),
    )
    return tf.math.ndtri(open_unit_interval(uniforms))


__all__ = [
    "ANCESTOR_CDF_ID",
    "ENDPOINT_POLICY_ID",
    "HILBERT_IMPLEMENTATION_ID",
    "POINT_SET_ID",
    "STATE_MAP_ID",
    "calibrated_logistic_map",
    "hilbert_integer_keys",
    "hilbert_permutation",
    "hilbert_transpose_words",
    "inverse_cdf_ancestor_indices",
    "open_unit_interval",
    "randomized_halton_gaussian",
    "randomized_halton_joint",
]
