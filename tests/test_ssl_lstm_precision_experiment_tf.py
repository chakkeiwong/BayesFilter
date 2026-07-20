from __future__ import annotations

import tensorflow as tf

from bayesfilter.nonlinear.ssl_lstm_complexity_target_tf import PRIOR_CENTER
from bayesfilter.nonlinear.ssl_lstm_complexity_target_tf import complexity_posterior_target
from bayesfilter.nonlinear.ssl_lstm_precision_experiment_tf import policy_dtypes
from bayesfilter.nonlinear.ssl_lstm_precision_experiment_tf import (
    ssl_lstm_precision_value_and_score,
)


def _evaluate(q: int, policy: str):
    target = complexity_posterior_target(q, jit_compile=False)
    return target, ssl_lstm_precision_value_and_score(
        PRIOR_CENTER,
        target.config.fixture,
        target.config.observations,
        target.config.static_config,
        target.config.free_indices,
        policy=policy,
        prior_center=target.config.prior_center,
        prior_standard_deviation=target.config.prior_standard_deviation,
    )


def test_policy_dtypes_distinguish_storage_from_tf32_policy() -> None:
    assert policy_dtypes("all_float64") == (tf.float64, tf.float64)
    assert policy_dtypes("mixed_lstm32_filter64") == (tf.float32, tf.float64)
    assert policy_dtypes("all_float32_tf32") == (tf.float32, tf.float32)


def test_experimental_float64_reproduces_production_target() -> None:
    target, actual = _evaluate(2, "all_float64")
    expected_value, expected_score = target.eager_value_and_score(PRIOR_CENTER)
    # The isolated mirror changes graph grouping around the dense principal root;
    # the resulting roundoff envelope is below 5e-7 across the checked ladder.
    tf.debugging.assert_near(actual.value, expected_value, atol=1e-6, rtol=0.0)
    tf.debugging.assert_near(actual.score, expected_score, atol=1e-6, rtol=0.0)
    assert actual.value.dtype == actual.score.dtype == tf.float64
    assert int(actual.placement_floor_count) == 0
    assert int(actual.innovation_floor_count) == 0


def test_mixed_and_float32_are_finite_with_declared_storage_dtype() -> None:
    for policy, expected_dtype in (
        ("mixed_lstm32_filter64", tf.float64),
        ("all_float32_tf32", tf.float32),
    ):
        _, result = _evaluate(2, policy)
        assert result.value.dtype == result.score.dtype == expected_dtype
        assert bool(tf.math.is_finite(result.value))
        assert bool(tf.reduce_all(tf.math.is_finite(result.score)))
        if policy == "mixed_lstm32_filter64":
            assert int(result.placement_floor_count) == 0
            assert int(result.innovation_floor_count) == 0
        else:
            assert int(result.placement_floor_count) > 0


def test_precision_engine_compiles_with_cpu_xla() -> None:
    target = complexity_posterior_target(1, jit_compile=False)

    @tf.function(input_signature=[tf.TensorSpec([4], tf.float32)], jit_compile=True)
    def compiled(free):
        result = ssl_lstm_precision_value_and_score(
            free,
            target.config.fixture,
            target.config.observations,
            target.config.static_config,
            target.config.free_indices,
            policy="all_float32_tf32",
            prior_center=target.config.prior_center,
            prior_standard_deviation=target.config.prior_standard_deviation,
        )
        return result.value, result.score

    value, score = compiled(tf.cast(PRIOR_CENTER, tf.float32))
    assert value.dtype == score.dtype == tf.float32
    assert bool(tf.math.is_finite(value))
    assert bool(tf.reduce_all(tf.math.is_finite(score)))
