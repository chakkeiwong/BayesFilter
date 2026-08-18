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


def _hilbert_transpose_axes(unit_points: Tensor, *, bits: int) -> list[Tensor]:
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
    axes = tf.unstack(coordinates, axis=1)

    # Inverse undo/exchange transform from Skilling's Hilbert transpose.
    q = 1 << (bits - 1)
    while q > 1:
        p = q - 1
        for axis in range(dimension):
            has_bit = tf.not_equal(tf.bitwise.bitwise_and(axes[axis], q), 0)
            exchange = tf.bitwise.bitwise_and(
                tf.bitwise.bitwise_xor(axes[0], axes[axis]), p
            )
            first_if_clear = tf.bitwise.bitwise_xor(axes[0], exchange)
            axis_if_clear = tf.bitwise.bitwise_xor(axes[axis], exchange)
            first_if_set = tf.bitwise.bitwise_xor(axes[0], p)
            axes[0] = tf.where(has_bit, first_if_set, first_if_clear)
            axes[axis] = tf.where(has_bit, axes[axis], axis_if_clear)
        q >>= 1

    for axis in range(1, dimension):
        axes[axis] = tf.bitwise.bitwise_xor(axes[axis], axes[axis - 1])

    correction = tf.zeros_like(axes[0])
    q = 1 << (bits - 1)
    while q > 1:
        correction = tf.where(
            tf.not_equal(tf.bitwise.bitwise_and(axes[dimension - 1], q), 0),
            tf.bitwise.bitwise_xor(correction, q - 1),
            correction,
        )
        q >>= 1
    axes = [tf.bitwise.bitwise_xor(axis, correction) for axis in axes]
    return axes


def hilbert_transpose_words(unit_points: Tensor, *, bits: int) -> Tensor:
    """Return arbitrary-width Hilbert keys as lexicographic 30-bit words."""

    axes = _hilbert_transpose_axes(unit_points, bits=bits)
    dimension = len(axes)
    words: list[Tensor] = []
    word = tf.zeros_like(axes[0])
    word_bits = 0
    for bit in range(bits - 1, -1, -1):
        for axis in range(dimension):
            word = tf.bitwise.left_shift(word, 1)
            word = tf.bitwise.bitwise_or(
                word,
                tf.bitwise.bitwise_and(
                    tf.bitwise.right_shift(axes[axis], bit), 1
                ),
            )
            word_bits += 1
            if word_bits == 30:
                words.append(word)
                word = tf.zeros_like(axes[0])
                word_bits = 0
    if word_bits:
        words.append(word)
    return tf.stack(words, axis=1)


def hilbert_integer_keys(unit_points: Tensor, *, bits: int) -> Tensor:
    """Return legacy packed keys for 2D/3D parity and compatibility tests."""

    unit_points = tf.convert_to_tensor(unit_points)
    dimension = unit_points.shape[-1]
    if dimension not in (2, 3):
        raise ValueError("packed Hilbert keys support dimensions 2 and 3")
    axes = _hilbert_transpose_axes(unit_points, bits=bits)

    key = tf.zeros_like(tf.cast(axes[0], tf.int64))
    for bit in range(bits - 1, -1, -1):
        for axis in range(dimension):
            key = tf.bitwise.left_shift(key, 1)
            key = tf.bitwise.bitwise_or(
                key,
                tf.cast(
                    tf.bitwise.bitwise_and(
                        tf.bitwise.right_shift(axes[axis], bit), 1
                    ),
                    tf.int64,
                ),
            )
    return key


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
    for word_index in range(int(words.shape[1]) - 1, -1, -1):
        local_order = tf.argsort(
            tf.gather(words[:, word_index], order), axis=0, stable=True
        )
        order = tf.gather(order, local_order)
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
