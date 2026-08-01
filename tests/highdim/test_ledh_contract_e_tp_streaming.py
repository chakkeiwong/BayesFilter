from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.highdim import ledh_contract_e_tp_tf as tp


DTYPE = tf.float64


def _sources() -> tuple[tf.Tensor, ...]:
    coordinate = tf.linspace(tf.constant(-1.5, DTYPE), tf.constant(1.5, DTYPE), 11)
    points = tf.stack([coordinate, 0.2 * coordinate + 0.1 * tf.square(coordinate)], axis=1)
    log_weights = -0.35 * tf.square(coordinate) + 0.08 * coordinate
    features = tf.stack(
        [tf.ones_like(coordinate), coordinate, tf.square(coordinate)], axis=0
    )
    return points, log_weights, features


def _block_program(block_size: int):
    def evaluate(
        sources: tuple[tf.Tensor, ...], start: tf.Tensor
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        points, log_weights, features = sources
        indices = tf.minimum(
            start + tf.range(block_size, dtype=tf.int32), tf.shape(points)[0] - 1
        )
        return (
            tf.gather(points, indices),
            tf.gather(log_weights, indices),
            tf.gather(features, indices, axis=1),
        )

    return evaluate


ACTIVE = tf.constant([0, 5, 10], tf.int32)
ROW_SCALE = tf.constant([1.0, 1.5, 2.25], DTYPE)


@pytest.mark.parametrize("block_size", [2, 4, 7, 16])
def test_streaming_primal_matches_dense_across_chunk_sizes(block_size: int) -> None:
    points, log_weights, features = _sources()
    dense = tp._contract_e_tp_dense_square_forward_core(
        points, log_weights, features, ACTIVE, ROW_SCALE
    )
    streaming = tp._contract_e_tp_streaming_square_forward_core(
        (points, log_weights, features),
        tf.shape(points)[0],
        ACTIVE,
        ROW_SCALE,
        block_size=block_size,
        block_program=_block_program(block_size),
    )
    for name in (
        "log_normalizer",
        "target",
        "student_points",
        "student_weights",
        "matched_target",
    ):
        np.testing.assert_allclose(streaming[name], dense[name], rtol=2e-14, atol=3e-15)


def test_streaming_candidate_permutation_preserves_target_and_student_measure() -> None:
    sources = _sources()
    permutation = tf.constant([5, 1, 9, 0, 10, 3, 7, 2, 8, 4, 6], tf.int32)
    inverse = tf.math.invert_permutation(permutation)
    permuted = (
        tf.gather(sources[0], permutation),
        tf.gather(sources[1], permutation),
        tf.gather(sources[2], permutation, axis=1),
    )
    baseline = tp._contract_e_tp_streaming_square_forward_core(
        sources, 11, ACTIVE, ROW_SCALE, block_size=4, block_program=_block_program(4)
    )
    changed = tp._contract_e_tp_streaming_square_forward_core(
        permuted,
        11,
        tf.gather(inverse, ACTIVE),
        ROW_SCALE,
        block_size=4,
        block_program=_block_program(4),
    )
    for name in ("log_normalizer", "target", "student_points", "student_weights", "matched_target"):
        np.testing.assert_allclose(changed[name], baseline[name], rtol=2e-14, atol=4e-15)


def test_streaming_jvp_matches_dense_and_automatic_differentiation() -> None:
    sources = _sources()
    source_tangents = (
        tf.reshape(tf.linspace(tf.constant(-0.02, DTYPE), tf.constant(0.03, DTYPE), 22), [11, 2]),
        tf.linspace(tf.constant(0.04, DTYPE), tf.constant(-0.025, DTYPE), 11),
        tf.reshape(tf.linspace(tf.constant(-0.01, DTYPE), tf.constant(0.015, DTYPE), 33), [3, 11]),
    )
    manual = tp._contract_e_tp_streaming_square_jvp_core(
        sources,
        source_tangents,
        11,
        ACTIVE,
        ROW_SCALE,
        block_size=4,
        block_program=_block_program(4),
    )
    with tf.autodiff.ForwardAccumulator(sources, source_tangents) as accumulator:
        automatic = tp._contract_e_tp_streaming_square_forward_core(
            sources, 11, ACTIVE, ROW_SCALE, block_size=4, block_program=_block_program(4)
        )
        dense = tp._contract_e_tp_dense_square_forward_core(
            sources[0], sources[1], sources[2], ACTIVE, ROW_SCALE
        )
    for name, tangent_name in (
        ("log_normalizer", "log_normalizer_tangent"),
        ("target", "target_tangent"),
        ("student_points", "student_points_tangent"),
        ("student_weights", "student_weights_tangent"),
        ("matched_target", "matched_target_tangent"),
    ):
        np.testing.assert_allclose(
            manual[tangent_name], accumulator.jvp(automatic[name]), rtol=2e-12, atol=3e-14
        )
        np.testing.assert_allclose(
            manual[tangent_name], accumulator.jvp(dense[name]), rtol=2e-12, atol=3e-14
        )


def test_streaming_vjp_matches_dense_and_tensorflow_for_all_sources() -> None:
    sources = _sources()
    upstream_points = tf.constant(
        [[0.2, -0.1], [0.05, 0.3], [-0.2, 0.15]], DTYPE
    )
    upstream_weights = tf.constant([0.1, -0.4, 0.25], DTYPE)
    upstream_target = tf.constant([-0.2, 0.15, 0.05], DTYPE)
    upstream_log_normalizer = tf.constant(0.31, DTYPE)
    manual = tp._contract_e_tp_streaming_square_vjp_core(
        sources,
        11,
        ACTIVE,
        ROW_SCALE,
        upstream_points,
        upstream_weights,
        upstream_target,
        upstream_log_normalizer,
        block_size=4,
        block_program=_block_program(4),
    )
    with tf.GradientTape() as tape:
        tape.watch(sources)
        result = tp._contract_e_tp_streaming_square_forward_core(
            sources, 11, ACTIVE, ROW_SCALE, block_size=4, block_program=_block_program(4)
        )
        objective = (
            tf.reduce_sum(result["student_points"] * upstream_points)
            + tf.tensordot(result["student_weights"], upstream_weights, axes=1)
            + tf.tensordot(result["matched_target"], upstream_target, axes=1)
            + result["log_normalizer"] * upstream_log_normalizer
        )
    automatic = tape.gradient(objective, sources)
    for expected, actual in zip(manual["source_bars"], automatic, strict=True):
        np.testing.assert_allclose(
            expected, tf.convert_to_tensor(actual), rtol=3e-12, atol=5e-14
        )


def test_streaming_factory_binds_xla_default_and_block_program() -> None:
    evaluator = tp.make_contract_e_tp_streaming_square_forward_tf(
        block_size=4, block_program=_block_program(4)
    )
    assert evaluator.function_spec.jit_compile is True
    with pytest.raises(ValueError, match="positive"):
        tp.make_contract_e_tp_streaming_square_forward_tf(
            block_size=0, block_program=_block_program(1)
        )
