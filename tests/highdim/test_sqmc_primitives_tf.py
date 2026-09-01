from __future__ import annotations

import pytest
import tensorflow as tf

from bayesfilter.highdim.sqmc_tf import (
    calibrated_logistic_map,
    hilbert_integer_keys,
    hilbert_permutation,
    hilbert_transpose_words,
    inverse_cdf_ancestor_indices,
    randomized_halton_joint,
)


def test_hilbert_order_matches_order_one_2d_fixture() -> None:
    # Standard order-one traversal: lower-left, upper-left, upper-right, lower-right.
    points = tf.constant(
        ((0.25, 0.25), (0.25, 0.75), (0.75, 0.75), (0.75, 0.25)),
        tf.float32,
    )
    tf.debugging.assert_equal(
        hilbert_integer_keys(points, bits=1), tf.constant((0, 1, 2, 3), tf.int64)
    )


def test_hilbert_permutation_is_stable_and_reports_ties() -> None:
    points = tf.constant(((0.0, 0.0), (0.0, 0.0), (2.0, -1.0)), tf.float32)
    order, ties, saturation = hilbert_permutation(
        points, tf.zeros([2]), tf.ones([2]), bits=8
    )
    assert tuple(order.numpy()[:2]) == (0, 1)
    assert int(ties.numpy()) == 1
    assert float(saturation.numpy()) == 0.0


@pytest.mark.parametrize(("dimension", "bits"), ((2, 12), (3, 12)))
def test_lexicographic_words_match_legacy_packed_order(
    dimension: int, bits: int
) -> None:
    points = tf.random.stateless_uniform(
        [257, dimension], tf.constant((211, dimension), tf.int32)
    )
    legacy_order = tf.argsort(
        hilbert_integer_keys(points, bits=bits), stable=True
    )
    words = hilbert_transpose_words(points, bits=bits)
    word_order = tf.range(tf.shape(points)[0], dtype=tf.int32)
    for word_index in range(int(words.shape[1]) - 1, -1, -1):
        local = tf.argsort(
            tf.gather(words[:, word_index], word_order), stable=True
        )
        word_order = tf.gather(word_order, local)
    tf.debugging.assert_equal(word_order, legacy_order)


def test_scalar_hilbert_order_is_monotone() -> None:
    points = tf.constant(((0.75,), (0.125,), (0.5,), (0.25,)), tf.float32)
    order, ties, saturation = hilbert_permutation(
        points, tf.zeros([1]), tf.ones([1]), bits=12
    )
    tf.debugging.assert_equal(order, (1, 3, 2, 0))
    assert int(ties.numpy()) == 0
    assert float(saturation.numpy()) == 0.0


def test_eighteen_dimensional_hilbert_words_replay_without_overflow() -> None:
    points = tf.random.stateless_uniform([72, 18], tf.constant((317, 19)))
    first = hilbert_transpose_words(points, bits=12)
    second = hilbert_transpose_words(points, bits=12)
    assert first.shape == (72, 8)
    tf.debugging.assert_equal(first, second)
    tf.debugging.assert_greater_equal(first, tf.zeros_like(first))


def test_calibrated_logistic_map_rejects_nonpositive_scale() -> None:
    with pytest.raises(tf.errors.InvalidArgumentError):
        calibrated_logistic_map(
            tf.zeros([2, 2]), tf.zeros([2]), tf.constant((1.0, 0.0))
        )


def test_inverse_cdf_boundary_and_repeated_ancestor_fixture() -> None:
    indices = inverse_cdf_ancestor_indices(
        tf.constant((0.0, 0.24, 0.25, 0.74, 0.99), tf.float32),
        tf.constant((0.25, 0.0, 0.5, 0.25), tf.float32),
    )
    tf.debugging.assert_equal(indices, (0, 0, 2, 2, 3))


def test_randomized_halton_joint_replays_and_preserves_rows() -> None:
    first = randomized_halton_joint(
        num_particles=16, state_dimension=2, seed=7, salt=11
    )
    second = randomized_halton_joint(
        num_particles=16, state_dimension=2, seed=7, salt=11
    )
    for left, right in zip(first, second, strict=True):
        tf.debugging.assert_equal(left, right)
    raw, ancestor, innovations = first
    order = tf.argsort(raw[:, 0], stable=True)
    tf.debugging.assert_equal(ancestor, tf.gather(raw[:, 0], order))
    tf.debugging.assert_equal(innovations, tf.gather(raw[:, 1:], order))


def test_primitives_jit_compile_on_cpu() -> None:
    @tf.function(jit_compile=True, autograph=False)
    def compiled(points, uniforms, weights):
        order, ties, saturation = hilbert_permutation(
            points, tf.zeros([2]), tf.ones([2]), bits=8
        )
        ancestors = inverse_cdf_ancestor_indices(uniforms, weights)
        return order, ties, saturation, ancestors

    result = compiled(
        tf.constant(((0.0, 0.0), (1.0, 1.0), (-1.0, 0.5), (0.2, -0.3))),
        tf.constant((0.1, 0.3, 0.6, 0.9)),
        tf.fill([4], 0.25),
    )
    assert result[0].device.endswith("CPU:0")
    concrete = compiled.get_concrete_function(
        tf.TensorSpec([4, 2], tf.float32),
        tf.TensorSpec([4], tf.float32),
        tf.TensorSpec([4], tf.float32),
    )
    assert concrete.function_def.attr["_XlaMustCompile"].b


@pytest.mark.parametrize("dimension", (1, 18))
def test_arbitrary_dimension_hilbert_jit_compiles_on_cpu(dimension: int) -> None:
    @tf.function(jit_compile=True, autograph=False)
    def compiled(points):
        return hilbert_permutation(
            points, tf.zeros([dimension]), tf.ones([dimension]), bits=12
        )

    result = compiled(tf.zeros([72, dimension], tf.float32))
    assert result[0].device.endswith("CPU:0")
    assert int(result[1].numpy()) == 71
    concrete = compiled.get_concrete_function(
        tf.TensorSpec([72, dimension], tf.float32)
    )
    assert concrete.function_def.attr["_XlaMustCompile"].b
