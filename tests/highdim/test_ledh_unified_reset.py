"""Gates for the shared batched Contract-E reset kernel."""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import tensorflow as tf

from bayesfilter.highdim.ledh_canonical_reset_score_tf import (
    sinkhorn_contract_e_reset_triple_with_tangent,
)
from bayesfilter.highdim.ledh_unified_reset_tf import (
    batched_sinkhorn_contract_e_reset_triple_with_tangent,
)
from test_ledh_canonical_covariance_carry import _reset_inputs

DTYPE = tf.float64
KWARGS = dict(epsilon=0.55, sinkhorn_steps=4, balance_steps=2, ridge=1.0e-6)


def _batched_inputs(batch_size: int = 3, direction_count: int = 2):
    base = _reset_inputs()
    rng = np.random.default_rng(26091101)
    children = np.stack([base[0] + 0.08 * row for row in range(batch_size)])
    covariances = np.stack([base[2] + 0.03 * row * np.eye(2) for row in range(batch_size)])
    weights = np.stack([np.roll(base[4], row) for row in range(batch_size)])
    d_children = 0.04 * rng.normal(size=(direction_count, *children.shape))
    raw_d_covariances = 0.01 * rng.normal(
        size=(direction_count, *covariances.shape)
    )
    d_covariances = 0.5 * (
        raw_d_covariances + raw_d_covariances.swapaxes(-1, -2)
    )
    raw_d_weights = 0.01 * rng.normal(
        size=(direction_count, batch_size, weights.shape[1])
    )
    d_weights = raw_d_weights - raw_d_weights.mean(axis=2, keepdims=True)
    return tuple(
        tf.constant(value, DTYPE)
        for value in (
            children,
            d_children,
            covariances,
            d_covariances,
            weights,
            d_weights,
            base[6],
        )
    )


def _run(values):
    return batched_sinkhorn_contract_e_reset_triple_with_tangent(
        *values, **KWARGS
    )


def test_b1_k1_matches_legacy_single_cloud_authority():
    base = _reset_inputs()
    old = sinkhorn_contract_e_reset_triple_with_tangent(
        *(tf.constant(value, DTYPE) for value in base), **KWARGS
    )
    values = tuple(
        tf.constant(value, DTYPE)
        for value in (
            base[0][None],
            base[1][None, None],
            base[2][None],
            base[3][None, None],
            base[4][None],
            base[5][None, None],
            base[6],
        )
    )
    new = _run(values)
    for index, (expected, actual) in enumerate(zip(old, new)):
        unbatched = actual[0, 0] if index in (1, 3, 5) else actual[0]
        np.testing.assert_allclose(
            unbatched.numpy(), expected.numpy(), rtol=2.0e-13, atol=2.0e-13
        )


def test_batched_rows_and_directions_equal_independent_calls():
    values = _batched_inputs()
    batched = _run(values)
    batch_size = int(values[0].shape[0])
    direction_count = int(values[1].shape[0])
    for row in range(batch_size):
        for direction in range(direction_count):
            one = (
                values[0][row : row + 1],
                values[1][direction : direction + 1, row : row + 1],
                values[2][row : row + 1],
                values[3][direction : direction + 1, row : row + 1],
                values[4][row : row + 1],
                values[5][direction : direction + 1, row : row + 1],
                values[6],
            )
            independent = _run(one)
            for index, (full, expected) in enumerate(zip(batched, independent)):
                if index in (1, 3, 5):
                    selected = full[direction, row]
                    target = expected[0, 0]
                else:
                    selected = full[row]
                    target = expected[0]
                np.testing.assert_allclose(
                    selected.numpy(), target.numpy(), rtol=3.0e-13, atol=3.0e-13
                )


def test_total_tangents_match_finite_difference():
    values = _batched_inputs(batch_size=2, direction_count=2)
    result = _run(values)
    epsilon = tf.constant(2.0e-6, DTYPE)
    for direction in range(2):
        primal_plus = list(values)
        primal_minus = list(values)
        for primal_index, tangent_index in ((0, 1), (2, 3), (4, 5)):
            tangent = values[tangent_index][direction]
            primal_plus[primal_index] = values[primal_index] + epsilon * tangent
            primal_minus[primal_index] = values[primal_index] - epsilon * tangent
        zeros = (
            tf.zeros_like(values[1][:1]),
            tf.zeros_like(values[3][:1]),
            tf.zeros_like(values[5][:1]),
        )
        for collection in (primal_plus, primal_minus):
            collection[1], collection[3], collection[5] = zeros
        plus = _run(tuple(primal_plus))
        minus = _run(tuple(primal_minus))
        for primal_output, tangent_output in ((0, 1), (2, 3), (4, 5)):
            finite_difference = (
                plus[primal_output] - minus[primal_output]
            ) / (2.0 * epsilon)
            np.testing.assert_allclose(
                result[tangent_output][direction].numpy(),
                finite_difference.numpy(),
                rtol=4.0e-6,
                atol=4.0e-8,
            )
