from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import pytest
import tensorflow as tf

from bayesfilter.testing.two_sample_energy_tf import (
    EnergyDiagnosticError,
    whole_path_energy_permutation_test,
)


def test_energy_statistic_matches_direct_v_statistic_and_replays() -> None:
    left = tf.constant([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], tf.float64)
    right = left + tf.constant([0.5, -0.25], tf.float64)
    first = whole_path_energy_permutation_test(
        left,
        right,
        permutation_count=99,
        seed=(20260809, 1001),
        permutation_batch_size=25,
        jit_compile=False,
    )
    second = whole_path_energy_permutation_test(
        left,
        right,
        permutation_count=99,
        seed=(20260809, 1001),
        permutation_batch_size=25,
        jit_compile=False,
    )

    def distances(a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
        return tf.norm(a[:, tf.newaxis, :] - b[tf.newaxis, :, :], axis=-1)

    direct = (
        2.0 * tf.reduce_mean(distances(left, right))
        - tf.reduce_mean(distances(left, left))
        - tf.reduce_mean(distances(right, right))
    )
    tf.debugging.assert_near(first.statistic, direct, atol=1.0e-12)
    tf.debugging.assert_equal(first.permutation_statistics, second.permutation_statistics)
    tf.debugging.assert_equal(first.p_value, second.p_value)
    assert float(first.p_value) == pytest.approx(
        (1 + int(first.exceedance_count)) / 100.0
    )


def test_energy_xla_matches_eager_and_detects_large_shift() -> None:
    left = tf.random.stateless_normal(
        [32, 5], seed=(20260809, 1101), dtype=tf.float64
    )
    right = tf.random.stateless_normal(
        [32, 5], seed=(20260809, 1102), dtype=tf.float64
    ) + 4.0
    eager = whole_path_energy_permutation_test(
        left,
        right,
        permutation_count=99,
        seed=(20260809, 1103),
        permutation_batch_size=25,
        jit_compile=False,
    )
    compiled = whole_path_energy_permutation_test(
        left,
        right,
        permutation_count=99,
        seed=(20260809, 1103),
        permutation_batch_size=25,
        jit_compile=True,
    )
    tf.debugging.assert_near(compiled.statistic, eager.statistic, atol=1.0e-10)
    tf.debugging.assert_near(
        compiled.permutation_statistics, eager.permutation_statistics, atol=1.0e-10
    )
    assert float(compiled.p_value) == pytest.approx(0.01)


def test_xla_distance_matrix_is_exactly_symmetric_for_large_1d_fixture() -> None:
    from bayesfilter.testing.two_sample_energy_tf import _distance_matrix_xla

    paths = tf.random.stateless_normal(
        [1024, 1], seed=(20260809, 1199), dtype=tf.float64
    )
    distances = _distance_matrix_xla(paths)
    tf.debugging.assert_equal(distances, tf.transpose(distances))
    tf.debugging.assert_equal(
        tf.linalg.diag_part(distances), tf.zeros([1024], tf.float64)
    )


@pytest.mark.parametrize(
    "left,right,permutations,batch,match",
    [
        (tf.zeros([2, 3], tf.float64), tf.zeros([3, 3], tf.float64), 9, 3, "equal shapes"),
        (tf.zeros([2, 3], tf.float64), tf.zeros([2, 4], tf.float64), 9, 3, "equal shapes"),
        (tf.zeros([2, 3], tf.float64), tf.zeros([2, 3], tf.float64), 0, 3, "positive"),
        (tf.zeros([2, 3], tf.float64), tf.zeros([2, 3], tf.float64), 9, 0, "positive"),
    ],
)
def test_energy_contract_rejects_invalid_geometry(
    left: tf.Tensor,
    right: tf.Tensor,
    permutations: int,
    batch: int,
    match: str,
) -> None:
    with pytest.raises(EnergyDiagnosticError, match=match):
        whole_path_energy_permutation_test(
            left,
            right,
            permutation_count=permutations,
            permutation_batch_size=batch,
            seed=(1, 2),
            jit_compile=False,
        )
