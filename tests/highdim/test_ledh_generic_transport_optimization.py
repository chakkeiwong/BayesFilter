from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import pytest
import tensorflow as tf

from bayesfilter.highdim import ledh_contract_e_streaming_tf as streaming


def _fixture(
    dtype: tf.dtypes.DType, parameter_count: int
) -> tuple[tf.Tensor, ...]:
    batch_size = 2
    particle_count = 4
    geometry_dimension = 3
    payload_dimension = 2
    geometry = tf.reshape(
        tf.linspace(tf.cast(-0.8, dtype), tf.cast(0.9, dtype),
                    batch_size * particle_count * geometry_dimension),
        [batch_size, particle_count, geometry_dimension],
    )
    payload = tf.reshape(
        tf.linspace(tf.cast(-0.5, dtype), tf.cast(0.7, dtype),
                    batch_size * particle_count * payload_dimension),
        [batch_size, particle_count, payload_dimension],
    )
    raw_log_weights = tf.reshape(
        tf.linspace(tf.cast(-0.4, dtype), tf.cast(0.6, dtype),
                    batch_size * particle_count),
        [batch_size, particle_count],
    )
    log_weights = raw_log_weights - tf.reduce_logsumexp(
        raw_log_weights, axis=1, keepdims=True
    )
    geometry_direction = tf.reshape(
        tf.linspace(
            tf.cast(-0.03, dtype),
            tf.cast(0.04, dtype),
            batch_size
            * particle_count
            * geometry_dimension
            * parameter_count,
        ),
        [batch_size, particle_count, geometry_dimension, parameter_count],
    )
    payload_direction = tf.reshape(
        tf.linspace(
            tf.cast(-0.02, dtype),
            tf.cast(0.025, dtype),
            batch_size
            * particle_count
            * payload_dimension
            * parameter_count,
        ),
        [batch_size, particle_count, payload_dimension, parameter_count],
    )
    log_weight_direction = tf.reshape(
        tf.linspace(
            tf.cast(-0.015, dtype),
            tf.cast(0.02, dtype),
            batch_size * particle_count * parameter_count,
        ),
        [batch_size, particle_count, parameter_count],
    )
    epsilon0_direction = tf.reshape(
        tf.linspace(
            tf.cast(-0.01, dtype),
            tf.cast(0.012, dtype),
            batch_size * parameter_count,
        ),
        [batch_size, parameter_count],
    )
    return (
        geometry,
        payload,
        log_weights,
        geometry_direction,
        payload_direction,
        log_weight_direction,
        epsilon0_direction,
        tf.cast(0.35, dtype),
        tf.fill([batch_size], tf.cast(0.8, dtype)),
        tf.cast(0.9, dtype),
    )


@pytest.mark.parametrize("parameter_count", [1, 3, 5, 18])
@pytest.mark.parametrize("dtype", [tf.float32, tf.float64])
def test_cached_same_cloud_state_matches_streamed_state(
    dtype: tf.dtypes.DType, parameter_count: int
) -> None:
    inputs = _fixture(dtype, parameter_count)
    kwargs = {
        "steps": 4,
        "balance_steps": 3,
        "row_chunk_size": 4,
        "col_chunk_size": 4,
    }
    baseline = streaming._balanced_transport_forward_jvp_state_core(  # noqa: SLF001
        *inputs, **kwargs, cache_same_cloud_geometry=False
    )
    cached = streaming._balanced_transport_forward_jvp_state_core(  # noqa: SLF001
        *inputs, **kwargs, cache_same_cloud_geometry=True
    )
    tolerances = (
        {"atol": 2.0e-5, "rtol": 2.0e-5}
        if dtype == tf.float32
        else {"atol": 2.0e-12, "rtol": 2.0e-12}
    )
    tensor_keys = (
        "augmented_numerator",
        "augmented_tangent",
        "row_potential",
        "column_potential",
        "row_potential_tangent",
        "column_potential_tangent",
        "column_mass",
        "post_quotient_column_mass",
    )
    for key in tensor_keys:
        assert baseline[key].shape == cached[key].shape
        assert bool(tf.reduce_all(tf.math.is_finite(cached[key])))
        tf.debugging.assert_near(baseline[key], cached[key], **tolerances)
    assert cached["augmented_tangent"].shape[-1] == parameter_count


def test_cache_is_opt_in_and_does_not_change_chunk_policy() -> None:
    inputs = _fixture(tf.float64, 3)
    with pytest.raises(ValueError, match="required K=4"):
        streaming._balanced_transport_forward_jvp_state_core(  # noqa: SLF001
            *inputs,
            steps=2,
            balance_steps=1,
            row_chunk_size=2,
            col_chunk_size=2,
            cache_same_cloud_geometry=True,
        )
