"""Golden-master characterisation tests for Phase 2B Step 1.

These tests lock down the exact numeric behavior of
`canonical_value_and_analytical_score` before refactoring.

Exit criterion: All tests pass at rtol=1e-12 (float64) or rtol=1e-6 (float32).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.highdim.ledh_canonical_score_tf import (
    NonlinearScoreModel,
    canonical_value_and_analytical_score,
)

# Repository root
REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DIR = REPO_ROOT / "tests" / "highdim" / "fixtures" / "ledh_golden_master_20260909"


def _make_lgssm_model(state_dim: int, dtype: tf.DType) -> NonlinearScoreModel:
    """Create a simple LGSSM model for testing (same as generator)."""
    # AR(1) transition: x_t = 0.95 * x_{t-1} + w_t
    transition_matrix = 0.95 * tf.eye(state_dim, dtype=dtype)
    process_covariance = 0.1 * tf.eye(state_dim, dtype=dtype)

    # Identity observation: y_t = x_t + v_t
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


def _discover_fixtures() -> list[Path]:
    """Discover all fixture files."""
    if not FIXTURE_DIR.exists():
        pytest.skip(f"Fixture directory not found: {FIXTURE_DIR}")

    fixtures = sorted(FIXTURE_DIR.glob("golden_*.json"))
    if not fixtures:
        pytest.skip(f"No fixtures found in {FIXTURE_DIR}")

    return fixtures


@pytest.mark.parametrize("fixture_path", _discover_fixtures())
def test_golden_master_regression(fixture_path: Path):
    """Test that current implementation matches golden-master fixture.

    This is the core Phase 2B Step 1 regression test. Any change to
    `canonical_value_and_analytical_score` that breaks this test must be
    intentional and requires regenerating fixtures.
    """
    # Load fixture
    with open(fixture_path) as f:
        fixture = json.load(f)

    assert fixture["schema"] == "ledh_golden_master_v1"

    config = fixture["config"]
    inputs = fixture["inputs"]
    expected_outputs = fixture["outputs"]

    # Extract config
    state_dim = config["state_dim"]
    dtype_str = config["dtype"]
    dtype = tf.float64 if dtype_str == "float64" else tf.float32
    annealed_stages = config["annealed_stages"]
    tuned_controls = config["tuned_controls"]

    # Create model
    model = _make_lgssm_model(state_dim, dtype)

    # Convert inputs to tensors
    theta = tf.constant(inputs["theta"], dtype=dtype)
    initial_states = tf.constant(inputs["initial_states"], dtype=dtype)
    initial_covariances = tf.constant(inputs["initial_covariances"], dtype=dtype)
    noises = tf.constant(inputs["noises"], dtype=dtype)
    observations = tf.constant(inputs["observations"], dtype=dtype)
    reset_design = tf.constant(inputs["reset_design"], dtype=dtype)

    # Run current implementation
    actual_value, actual_score = canonical_value_and_analytical_score(
        model=model,
        theta=theta,
        initial_states=initial_states,
        initial_covariances=initial_covariances,
        noises=noises,
        observations=observations,
        with_score=True,
        return_trace=False,
        reset_design=reset_design,
        annealed_stages=annealed_stages,
        **{k: v for k, v in tuned_controls.items() if k != "reset_policy"},
        reset_policy=tuned_controls["reset_policy"],
    )

    # Convert expected outputs
    expected_value = np.array(expected_outputs["value"])
    expected_score = (
        np.array(expected_outputs["score"])
        if expected_outputs["score"] is not None
        else None
    )

    # Compare with strict tolerance
    rtol = 1e-12 if dtype == tf.float64 else 1e-6
    atol = 0.0

    # Value comparison
    np.testing.assert_allclose(
        actual_value.numpy(),
        expected_value,
        rtol=rtol,
        atol=atol,
        err_msg=f"Value mismatch for {fixture_path.name}",
    )

    # Score comparison
    if expected_score is not None:
        assert actual_score is not None, f"Score is None for {fixture_path.name}"
        np.testing.assert_allclose(
            actual_score.numpy(),
            expected_score,
            rtol=rtol,
            atol=atol,
            err_msg=f"Score mismatch for {fixture_path.name}",
        )


def test_fixture_coverage():
    """Verify that we have adequate fixture coverage across the test matrix."""
    fixtures = _discover_fixtures()
    fixture_configs = []

    for fixture_path in fixtures:
        with open(fixture_path) as f:
            fixture = json.load(f)
            config = fixture["config"]
            fixture_configs.append(
                (
                    config["horizon"],
                    config["particle_count"],
                    config["state_dim"],
                    config["direction_count"],
                    config["dtype"],
                    config["annealed_stages"],
                )
            )

    # Expected matrix dimensions
    expected_horizons = {1, 2, 3, 10, 50}
    expected_particles = {6, 64, 1008}
    expected_dims = {2, 3}
    expected_directions = {1, 2, 5}
    expected_dtypes = {"float64", "float32"}
    expected_annealed = {1, 8}

    # Check coverage
    actual_horizons = {c[0] for c in fixture_configs}
    actual_particles = {c[1] for c in fixture_configs}
    actual_dims = {c[2] for c in fixture_configs}
    actual_directions = {c[3] for c in fixture_configs}
    actual_dtypes = {c[4] for c in fixture_configs}
    actual_annealed = {c[5] for c in fixture_configs}

    assert actual_horizons == expected_horizons, f"Missing horizons: {expected_horizons - actual_horizons}"
    assert actual_particles == expected_particles, f"Missing particle counts: {expected_particles - actual_particles}"
    assert actual_dims == expected_dims, f"Missing dimensions: {expected_dims - actual_dims}"
    assert actual_directions == expected_directions, f"Missing direction counts: {expected_directions - actual_directions}"
    assert actual_dtypes == expected_dtypes, f"Missing dtypes: {expected_dtypes - actual_dtypes}"
    assert actual_annealed == expected_annealed, f"Missing annealed stages: {expected_annealed - actual_annealed}"

    # Expected total: 5 × 3 × 2 × 3 × 2 × 2 = 360
    expected_total = 5 * 3 * 2 * 3 * 2 * 2
    assert len(fixtures) == expected_total, (
        f"Expected {expected_total} fixtures, found {len(fixtures)}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
