"""Gates for the shared batched higher-moment correction boundary."""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.highdim.higher_moment_contract_e import higher_moment_shape_jvp
from bayesfilter.highdim.ledh_unified_correction_tf import (
    batched_higher_moment_shape_jvp,
)

DTYPE = tf.float64
CONFIGS = {
    "off": dict(correction_steps=0, strength=0.0),
    "diagonal": dict(
        correction_steps=2,
        strength=0.2,
        diagonal_lm_damping=0.001,
        diagonal_trust_radius=0.0,
    ),
    "trust": dict(
        correction_steps=2,
        strength=0.2,
        diagonal_lm_damping=0.001,
        diagonal_trust_radius=0.1,
    ),
    "pairwise": dict(
        correction_steps=2,
        strength=0.2,
        diagonal_lm_damping=0.001,
        diagonal_trust_radius=0.1,
        pairwise_correction_steps=2,
        pairwise_strength=0.02,
        pairwise_particle_rms_cap=2.0,
    ),
    "full": dict(
        correction_steps=2,
        strength=0.2,
        diagonal_lm_damping=0.001,
        diagonal_trust_radius=0.1,
        pairwise_correction_steps=2,
        pairwise_strength=0.02,
        pairwise_particle_rms_cap=2.0,
        coordinatewise_standardized_cap=0.98,
    ),
}


def _inputs(batch_size: int = 3, direction_count: int = 2):
    rng = np.random.default_rng(26091102)
    particle_count, dimension = 16, 2
    source = rng.normal(size=(batch_size, particle_count, dimension))
    points = rng.normal(size=(batch_size, particle_count, dimension))
    logits = rng.normal(scale=0.2, size=(batch_size, particle_count))
    weights = np.exp(logits - logits.max(axis=1, keepdims=True))
    weights /= weights.sum(axis=1, keepdims=True)
    source_tangent = 0.02 * rng.normal(
        size=(direction_count, batch_size, particle_count, dimension)
    )
    points_tangent = 0.02 * rng.normal(
        size=(direction_count, batch_size, particle_count, dimension)
    )
    raw_weights_tangent = 0.002 * rng.normal(
        size=(direction_count, batch_size, particle_count)
    )
    weights_tangent = raw_weights_tangent - (
        raw_weights_tangent.sum(axis=2, keepdims=True) * weights[None, ...]
    )
    return tuple(
        tf.constant(value, DTYPE)
        for value in (
            source,
            weights,
            source_tangent,
            weights_tangent,
            points,
            points_tangent,
        )
    )


def _run(values, config):
    return batched_higher_moment_shape_jvp(*values, **config)


@pytest.mark.parametrize("config_name", tuple(CONFIGS))
def test_b1_k1_matches_single_cloud_authority(config_name: str):
    values = _inputs(batch_size=1, direction_count=1)
    config = CONFIGS[config_name]
    expected = higher_moment_shape_jvp(
        values[0][0],
        values[1][0],
        tf.transpose(values[2][:, 0], [1, 2, 0]),
        tf.transpose(values[3][:, 0], [1, 0]),
        values[4][0],
        tf.transpose(values[5][:, 0], [1, 2, 0]),
        **config,
    )
    actual = _run(values, config)
    for key, expected_value in expected.items():
        if key == "particles_tangent":
            actual_value = actual[key][0, 0]
            expected_value = expected_value[:, :, 0]
        else:
            actual_value = actual[key][0]
        if expected_value.dtype == tf.bool or expected_value.dtype.is_integer:
            np.testing.assert_array_equal(
                actual_value.numpy(), expected_value.numpy()
            )
        else:
            np.testing.assert_allclose(
                actual_value.numpy(), expected_value.numpy(),
                rtol=2.0e-13, atol=2.0e-13,
            )


@pytest.mark.parametrize("config_name", tuple(CONFIGS))
def test_batched_rows_and_directions_equal_independent_calls(config_name: str):
    values = _inputs()
    config = CONFIGS[config_name]
    batched = _run(values, config)
    for row in range(3):
        for direction in range(2):
            one = (
                values[0][row : row + 1],
                values[1][row : row + 1],
                values[2][direction : direction + 1, row : row + 1],
                values[3][direction : direction + 1, row : row + 1],
                values[4][row : row + 1],
                values[5][direction : direction + 1, row : row + 1],
            )
            independent = _run(one, config)
            np.testing.assert_allclose(
                batched["particles"][row].numpy(),
                independent["particles"][0].numpy(),
                rtol=3.0e-13,
                atol=3.0e-13,
            )
            np.testing.assert_allclose(
                batched["particles_tangent"][direction, row].numpy(),
                independent["particles_tangent"][0, 0].numpy(),
                rtol=3.0e-13,
                atol=3.0e-13,
            )


def test_shared_correction_traces_under_tf_function():
    values = _inputs(batch_size=2, direction_count=2)
    compiled = tf.function(
        lambda *args: batched_higher_moment_shape_jvp(
            *args, **CONFIGS["full"]
        ),
        autograph=False,
    )
    result = compiled(*values)
    assert result["particles"].shape == (2, 16, 2)
    assert result["particles_tangent"].shape == (2, 2, 16, 2)
    assert bool(tf.reduce_all(result["valid"]).numpy())


def test_analytical_tangents_match_autodiff_oracle():
    """Phase 2C oracle gate: batched higher-moment correction JVP vs autodiff."""
    values = _inputs(batch_size=2, direction_count=1)
    config = CONFIGS["full"]
    analytical = batched_higher_moment_shape_jvp(*values, **config)

    def primal_fn(source, weights, points):
        result = batched_higher_moment_shape_jvp(
            source,
            weights,
            tf.zeros_like(values[2][:1]),
            tf.zeros_like(values[3][:1]),
            points,
            tf.zeros_like(values[5][:1]),
            **config,
        )
        return tf.reshape(result["particles"], [-1])

    # Oracle: forward-mode autodiff on primal path
    source = tf.identity(values[0])
    weights = tf.identity(values[1])
    points = tf.identity(values[4])

    source_dir = values[2][0]
    weights_dir = values[3][0]
    points_dir = values[5][0]

    with tf.autodiff.ForwardAccumulator(
        primals=[source, weights, points],
        tangents=[source_dir, weights_dir, points_dir]
    ) as acc:
        primal_out = primal_fn(source, weights, points)
    oracle_tangent = acc.jvp(primal_out)

    # Unpack analytical tangents (particles only)
    analytical_tangent = tf.reshape(analytical["particles_tangent"][0], [-1])

    np.testing.assert_allclose(
        analytical_tangent.numpy(),
        oracle_tangent.numpy(),
        rtol=1.0e-6,
        atol=1.0e-8,
    )
