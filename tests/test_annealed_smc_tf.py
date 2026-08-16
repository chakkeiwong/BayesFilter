"""Known-law tests for adaptive annealed-SMC TensorFlow primitives."""

from __future__ import annotations

import tensorflow as tf

from bayesfilter.testing.annealed_smc_tf import (
    make_bridge_hmc_step,
    normalized_weight_diagnostics,
    select_next_beta,
    systematic_resample_indices,
)


DTYPE = tf.float64


def test_beta_selector_reaches_terminal_when_ess_is_admissible() -> None:
    result = select_next_beta(
        tf.constant(0.0, DTYPE),
        tf.zeros(100, DTYPE),
        tf.zeros(100, DTYPE),
        target_ess_fraction=0.8,
    )
    tf.debugging.assert_equal(result["next_beta"], tf.constant(1.0, DTYPE))
    tf.debugging.assert_near(
        result["effective_sample_size_fraction"], tf.constant(1.0, DTYPE)
    )


def test_beta_selector_hits_conditional_ess_boundary() -> None:
    ratio = tf.linspace(tf.constant(-8.0, DTYPE), tf.constant(8.0, DTYPE), 100)
    result = select_next_beta(
        tf.constant(0.0, DTYPE),
        ratio,
        tf.zeros(100, DTYPE),
        target_ess_fraction=0.8,
    )
    assert 0.0 < float(result["next_beta"].numpy()) < 1.0
    tf.debugging.assert_near(
        result["effective_sample_size_fraction"],
        tf.constant(0.8, DTYPE),
        atol=1.0e-6,
    )


def test_systematic_resampling_replays_and_is_global() -> None:
    diagnostics = normalized_weight_diagnostics(
        tf.math.log(tf.constant((0.05, 0.15, 0.3, 0.5), DTYPE))
    )
    first = systematic_resample_indices(
        diagnostics["normalized_log_weights"], seed=(20260810, 8101)
    )
    replay = systematic_resample_indices(
        diagnostics["normalized_log_weights"], seed=(20260810, 8101)
    )
    tf.debugging.assert_equal(first, replay)
    assert first.shape == (4,)
    assert int(tf.reduce_min(first).numpy()) >= 0
    assert int(tf.reduce_max(first).numpy()) < 4


def test_resampling_ancestry_tracks_both_regions_on_balanced_law() -> None:
    roots = tf.range(100, dtype=tf.int32)
    signs = roots >= 50
    log_weights = tf.zeros(100, DTYPE)
    diagnostics = normalized_weight_diagnostics(log_weights)
    parents = systematic_resample_indices(
        diagnostics["normalized_log_weights"], seed=(20260810, 8201)
    )
    resampled_signs = tf.gather(signs, parents)
    assert bool(tf.reduce_any(resampled_signs).numpy())
    assert bool(tf.reduce_any(tf.logical_not(resampled_signs)).numpy())
    tf.debugging.assert_equal(tf.gather(roots, parents), roots)


def test_reusable_bridge_hmc_replays_and_changes_seed() -> None:
    def proposal(state: tf.Tensor) -> tf.Tensor:
        return -0.5 * tf.reduce_sum(tf.square(state), axis=1)

    def target(state: tf.Tensor) -> tf.Tensor:
        return -0.5 * tf.reduce_sum(tf.square(state - 1.0), axis=1)

    step = make_bridge_hmc_step(
        proposal,
        target,
        path_count=8,
        dimension=1,
        step_size=0.2,
        num_leapfrog_steps=3,
        jit_compile=True,
    )
    state = tf.zeros((8, 1), DTYPE)
    first = step(state, tf.constant(0.5, DTYPE), tf.constant((20260810, 8301), tf.int32))
    replay = step(state, tf.constant(0.5, DTYPE), tf.constant((20260810, 8301), tf.int32))
    second = step(state, tf.constant(0.5, DTYPE), tf.constant((20260810, 8302), tf.int32))
    tf.debugging.assert_near(first["state"], replay["state"])
    assert bool(tf.reduce_any(first["state"] != second["state"]).numpy())
    assert bool(tf.reduce_all(tf.math.is_finite(first["log_accept_ratio"])).numpy())
