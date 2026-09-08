"""Row-independence tests for Phase 2B Step 1.

These tests verify that batch processing of multiple θ values does not mix
clouds across rows. This is the principal design risk during refactoring:
naive `[B,N] → [B*N]` flattening can create cross-row data dependencies.

Test Strategy:
    T1: Identical rows → identical outputs (bitwise)
    T2: Distinct rows → distinct outputs
    T3: Row isolation → Row i output unchanged by row j perturbation
"""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.highdim.ledh_canonical_score_tf import (
    NonlinearScoreModel,
    canonical_value_and_analytical_score,
)

# Tuned controls from 2026-09-03 LGSSM trust-region tuning
TUNED_CONTROLS = {
    "reset_policy": "contract_e",
    "reset_epsilon": 2.0,
    "reset_sinkhorn_steps": 8,
    "reset_balance_steps": 8,
    "reset_ridge": 1e-05,
    "correction_steps": 4,
    "correction_strength": 0.2,
    "correction_lm_damping": 0.001,
    "correction_lm_scale_floor": 1e-06,
    "correction_trust_radius": 0.1,
    "pairwise_steps": 4,
    "pairwise_strength": 0.02,
    "pairwise_rms_cap": 2.0,
    "coordinate_cap": 0.98,
    "coordinate_cap_power": 8,
    "flow_substeps": 24,
}


def _make_lgssm_model(state_dim: int, dtype: tf.DType) -> NonlinearScoreModel:
    """Create a simple LGSSM model for testing."""
    transition_matrix = 0.95 * tf.eye(state_dim, dtype=dtype)
    process_covariance = 0.1 * tf.eye(state_dim, dtype=dtype)
    observation_matrix = tf.eye(state_dim, dtype=dtype)
    observation_covariance = 0.2 * tf.eye(state_dim, dtype=dtype)

    def transition_mean_fn(theta: tf.Tensor, points: tf.Tensor) -> tf.Tensor:
        return tf.linalg.matvec(transition_matrix, points)

    def transition_mean_tangent_fn(
        theta: tf.Tensor, points: tf.Tensor, d_points: tf.Tensor
    ) -> tf.Tensor:
        return tf.linalg.matvec(transition_matrix, d_points)

    def observation_fn(points: tf.Tensor) -> tf.Tensor:
        return tf.linalg.matvec(observation_matrix, points)

    def observation_jacobian_fn(points: tf.Tensor) -> tf.Tensor:
        batch_size = tf.shape(points)[0]
        return tf.broadcast_to(
            observation_matrix, [batch_size, state_dim, state_dim]
        )

    def observation_tangent_fn(
        points: tf.Tensor, d_points: tf.Tensor
    ) -> tf.Tensor:
        return tf.linalg.matvec(observation_matrix, d_points)

    def observation_jacobian_tangent_fn(
        points: tf.Tensor, d_points: tf.Tensor
    ) -> tf.Tensor:
        batch_size = tf.shape(points)[0]
        return tf.zeros([batch_size, state_dim, state_dim], dtype=dtype)

    return NonlinearScoreModel(
        transition_mean_fn=transition_mean_fn,
        transition_mean_tangent_fn=transition_mean_tangent_fn,
        observation_fn=observation_fn,
        observation_jacobian_fn=observation_jacobian_fn,
        observation_tangent_fn=observation_tangent_fn,
        process_covariance=process_covariance,
        observation_covariance=observation_covariance,
        observation_jacobian_tangent_fn=observation_jacobian_tangent_fn,
    )


def _generate_inputs(
    horizon: int,
    particle_count: int,
    state_dim: int,
    direction_count: int,
    dtype: tf.DType,
    seed: int,
) -> dict[str, tf.Tensor]:
    """Generate deterministic inputs."""
    rng = tf.random.Generator.from_seed(seed)

    theta = rng.normal([direction_count, 2], dtype=dtype) * 0.1
    initial_states = rng.normal([particle_count, state_dim], dtype=dtype)
    initial_covariances = tf.broadcast_to(
        tf.eye(state_dim, dtype=dtype),
        [particle_count, state_dim, state_dim],
    )
    noises = rng.normal([horizon, particle_count, state_dim], dtype=dtype)
    observations = rng.normal([horizon, state_dim], dtype=dtype)

    # Reset design
    if particle_count % 2 == 0:
        base = tf.concat(
            [tf.eye(state_dim, dtype=dtype), -tf.eye(state_dim, dtype=dtype)],
            axis=0,
        )
        reset_design = tf.tile(base, [particle_count // (2 * state_dim), 1])
        if reset_design.shape[0] < particle_count:
            extra = particle_count - reset_design.shape[0]
            reset_design = tf.concat([reset_design, base[:extra]], axis=0)
    else:
        reset_design = tf.tile(
            tf.constant([[1.0], [-1.0]], dtype=dtype),
            [particle_count // 2 + 1, state_dim],
        )[:particle_count]

    return {
        "theta": theta,
        "initial_states": initial_states,
        "initial_covariances": initial_covariances,
        "noises": noises,
        "observations": observations,
        "reset_design": reset_design,
    }


@pytest.mark.parametrize("batch_size", [1, 2, 4])
@pytest.mark.parametrize("dtype", [tf.float64, tf.float32])
def test_t1_identical_rows_produce_identical_outputs(batch_size: int, dtype: tf.DType):
    """T1: Identical θ rows must produce bitwise identical outputs.

    This verifies that the implementation is deterministic and that row
    processing doesn't introduce spurious variation.
    """
    horizon = 3
    particle_count = 64
    state_dim = 2
    direction_count = 1

    model = _make_lgssm_model(state_dim, dtype)
    inputs = _generate_inputs(
        horizon, particle_count, state_dim, direction_count, dtype, seed=42
    )

    # Create batch with identical rows
    theta_single = inputs["theta"]
    theta_batch = tf.tile(theta_single, [batch_size, 1])

    value_batch, score_batch = canonical_value_and_analytical_score(
        model=model,
        theta=theta_batch,
        initial_states=inputs["initial_states"],
        initial_covariances=inputs["initial_covariances"],
        noises=inputs["noises"],
        observations=inputs["observations"],
        with_score=True,
        reset_design=inputs["reset_design"],
        **{k: v for k, v in TUNED_CONTROLS.items() if k != "reset_policy"},
        reset_policy=TUNED_CONTROLS["reset_policy"],
    )

    # All rows should be identical
    for i in range(batch_size):
        for j in range(i + 1, batch_size):
            np.testing.assert_array_equal(
                value_batch[i].numpy(),
                value_batch[j].numpy(),
                err_msg=f"T1 FAIL: Identical θ rows {i} and {j} produced different values",
            )
            np.testing.assert_array_equal(
                score_batch[i].numpy(),
                score_batch[j].numpy(),
                err_msg=f"T1 FAIL: Identical θ rows {i} and {j} produced different scores",
            )


@pytest.mark.parametrize("batch_size", [2, 4])
@pytest.mark.parametrize("dtype", [tf.float64, tf.float32])
def test_t2_distinct_rows_produce_distinct_outputs(batch_size: int, dtype: tf.DType):
    """T2: Distinct θ rows must produce distinct outputs.

    This verifies that the implementation actually uses the θ values and
    doesn't accidentally broadcast or share state across rows.
    """
    horizon = 3
    particle_count = 64
    state_dim = 2
    direction_count = 1

    model = _make_lgssm_model(state_dim, dtype)
    inputs = _generate_inputs(
        horizon, particle_count, state_dim, direction_count, dtype, seed=43
    )

    # Create batch with distinct rows
    rng = tf.random.Generator.from_seed(999)
    theta_batch = rng.normal([batch_size, direction_count, 2], dtype=dtype) * 0.1

    value_batch, score_batch = canonical_value_and_analytical_score(
        model=model,
        theta=theta_batch,
        initial_states=inputs["initial_states"],
        initial_covariances=inputs["initial_covariances"],
        noises=inputs["noises"],
        observations=inputs["observations"],
        with_score=True,
        reset_design=inputs["reset_design"],
        **{k: v for k, v in TUNED_CONTROLS.items() if k != "reset_policy"},
        reset_policy=TUNED_CONTROLS["reset_policy"],
    )

    # All rows should be different
    for i in range(batch_size):
        for j in range(i + 1, batch_size):
            # Values should differ
            value_diff = np.abs(value_batch[i].numpy() - value_batch[j].numpy())
            assert value_diff.max() > 1e-6, (
                f"T2 FAIL: Distinct θ rows {i} and {j} produced nearly identical values"
            )

            # Scores should differ
            score_diff = np.abs(score_batch[i].numpy() - score_batch[j].numpy())
            assert score_diff.max() > 1e-6, (
                f"T2 FAIL: Distinct θ rows {i} and {j} produced nearly identical scores"
            )


@pytest.mark.parametrize("batch_size", [2, 4])
@pytest.mark.parametrize("dtype", [tf.float64, tf.float32])
def test_t3_row_isolation(batch_size: int, dtype: tf.DType):
    """T3: Output for row i must be unchanged when row j is perturbed.

    This is the core row-independence test. It verifies that clouds/states
    don't leak across θ rows during processing.
    """
    horizon = 3
    particle_count = 64
    state_dim = 2
    direction_count = 1

    model = _make_lgssm_model(state_dim, dtype)
    inputs = _generate_inputs(
        horizon, particle_count, state_dim, direction_count, dtype, seed=44
    )

    # Create baseline batch
    rng = tf.random.Generator.from_seed(1000)
    theta_baseline = rng.normal([batch_size, direction_count, 2], dtype=dtype) * 0.1

    value_baseline, score_baseline = canonical_value_and_analytical_score(
        model=model,
        theta=theta_baseline,
        initial_states=inputs["initial_states"],
        initial_covariances=inputs["initial_covariances"],
        noises=inputs["noises"],
        observations=inputs["observations"],
        with_score=True,
        reset_design=inputs["reset_design"],
        **{k: v for k, v in TUNED_CONTROLS.items() if k != "reset_policy"},
        reset_policy=TUNED_CONTROLS["reset_policy"],
    )

    # Perturb each row one at a time and verify others are unchanged
    for perturb_row in range(batch_size):
        theta_perturbed = tf.identity(theta_baseline).numpy()
        theta_perturbed[perturb_row] += 0.5  # Large perturbation
        theta_perturbed = tf.constant(theta_perturbed, dtype=dtype)

        value_perturbed, score_perturbed = canonical_value_and_analytical_score(
            model=model,
            theta=theta_perturbed,
            initial_states=inputs["initial_states"],
            initial_covariances=inputs["initial_covariances"],
            noises=inputs["noises"],
            observations=inputs["observations"],
            with_score=True,
            reset_design=inputs["reset_design"],
            **{k: v for k, v in TUNED_CONTROLS.items() if k != "reset_policy"},
            reset_policy=TUNED_CONTROLS["reset_policy"],
        )

        # All non-perturbed rows should be unchanged (bitwise)
        for row in range(batch_size):
            if row == perturb_row:
                # Perturbed row should differ
                value_diff = np.abs(
                    value_perturbed[row].numpy() - value_baseline[row].numpy()
                )
                assert value_diff.max() > 1e-3, (
                    f"T3 FAIL: Perturbed row {row} didn't change"
                )
            else:
                # Unperturbed rows should be identical
                np.testing.assert_array_equal(
                    value_perturbed[row].numpy(),
                    value_baseline[row].numpy(),
                    err_msg=f"T3 FAIL: Row {row} changed when row {perturb_row} was perturbed",
                )
                np.testing.assert_array_equal(
                    score_perturbed[row].numpy(),
                    score_baseline[row].numpy(),
                    err_msg=f"T3 FAIL: Row {row} score changed when row {perturb_row} was perturbed",
                )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
