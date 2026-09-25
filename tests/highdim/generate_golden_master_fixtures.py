#!/usr/bin/env python3
"""Generate golden-master fixtures for Phase 2B Step 1 characterisation tests.

This script creates 360 reference fixtures capturing exact numeric behavior of
`canonical_value_and_analytical_score` with `reset_policy="contract_e"` and
2026-09-03 tuned controls.

Test Matrix:
    T ∈ {1, 2, 3, 10, 50}
    N ∈ {6, 64, 1008}
    d ∈ {2, 3}
    K (directions) ∈ {1, 2, 5}
    dtype ∈ {float64, float32}
    annealed_stages ∈ {1, 8}

Total: 5 × 3 × 2 × 3 × 2 × 2 = 360 fixtures

Usage:
    python tests/highdim/generate_golden_master_fixtures.py

Output:
    tests/highdim/fixtures/ledh_golden_master_20260909/*.json
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import tensorflow as tf

from bayesfilter.highdim.ledh_canonical_score_tf import (
    NonlinearScoreModel,
    canonical_value_and_analytical_score,
)

# Repository root
REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DIR = REPO_ROOT / "tests" / "highdim" / "fixtures" / "ledh_golden_master_20260909"

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

# Test matrix
TEST_MATRIX = {
    "horizons": [1, 2, 3, 10, 50],
    "particle_counts": [6, 64, 1008],
    "state_dims": [2, 3],
    "direction_counts": [1, 2, 5],
    "dtypes": ["float64", "float32"],
    "annealed_stages": [1, 8],
}


def _git_commit() -> str:
    """Get current git commit hash."""
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    ).stdout.strip()


def _make_lgssm_model(state_dim: int, dtype: tf.DType) -> NonlinearScoreModel:
    """Create a simple LGSSM model for testing."""
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


def _generate_inputs(
    horizon: int,
    particle_count: int,
    state_dim: int,
    direction_count: int,
    dtype: tf.DType,
    seed: int,
) -> dict[str, tf.Tensor]:
    """Generate deterministic inputs from seed."""
    rng = tf.random.Generator.from_seed(seed)

    # Theta: [K, P] where P is parameter dimension (for LGSSM, minimal)
    # Use K directions, even though LGSSM is parameter-free in this test
    theta = rng.normal([direction_count, 2], dtype=dtype) * 0.1

    # Initial states: [N, d]
    initial_states = rng.normal([particle_count, state_dim], dtype=dtype)

    # Initial covariances: [N, d, d]
    initial_covariances = tf.broadcast_to(
        tf.eye(state_dim, dtype=dtype),
        [particle_count, state_dim, state_dim],
    )

    # Noises: [T, N, d]
    noises = rng.normal([horizon, particle_count, state_dim], dtype=dtype)

    # Observations: [T, d]
    observations = rng.normal([horizon, state_dim], dtype=dtype)

    return {
        "theta": theta,
        "initial_states": initial_states,
        "initial_covariances": initial_covariances,
        "noises": noises,
        "observations": observations,
    }


def _tensor_to_nested_list(tensor: tf.Tensor) -> list:
    """Convert tensor to nested list for JSON serialization."""
    return tensor.numpy().tolist()


def _generate_fixture(
    horizon: int,
    particle_count: int,
    state_dim: int,
    direction_count: int,
    dtype_str: str,
    annealed_stages: int,
    seed: int,
    git_commit: str,
) -> dict[str, Any]:
    """Generate one golden-master fixture."""
    dtype = tf.float64 if dtype_str == "float64" else tf.float32

    # Create model
    model = _make_lgssm_model(state_dim, dtype)

    # Generate inputs
    inputs = _generate_inputs(
        horizon, particle_count, state_dim, direction_count, dtype, seed
    )

    # Create reset design
    if particle_count % 2 == 0:
        base = tf.concat(
            [tf.eye(state_dim, dtype=dtype), -tf.eye(state_dim, dtype=dtype)],
            axis=0,
        )
        reset_design = tf.tile(base, [particle_count // (2 * state_dim), 1])
        # Handle case where N is not divisible by 2*d
        if reset_design.shape[0] < particle_count:
            extra = particle_count - reset_design.shape[0]
            reset_design = tf.concat(
                [reset_design, base[:extra]], axis=0
            )
    else:
        # Odd particle count: use simple alternating pattern
        reset_design = tf.tile(
            tf.constant([[1.0], [-1.0]], dtype=dtype),
            [particle_count // 2 + 1, state_dim],
        )[:particle_count]

    # Run canonical score with trace (only for annealed_stages=1)
    # For annealed_stages>1, return_trace is not supported yet
    if annealed_stages == 1:
        value, score, trace = canonical_value_and_analytical_score(
            model=model,
            theta=inputs["theta"],
            initial_states=inputs["initial_states"],
            initial_covariances=inputs["initial_covariances"],
            noises=inputs["noises"],
            observations=inputs["observations"],
            with_score=True,
            return_trace=True,
            reset_design=reset_design,
            annealed_stages=annealed_stages,
            **{k: v for k, v in TUNED_CONTROLS.items() if k != "reset_policy"},
            reset_policy=TUNED_CONTROLS["reset_policy"],
        )
    else:
        value, score = canonical_value_and_analytical_score(
            model=model,
            theta=inputs["theta"],
            initial_states=inputs["initial_states"],
            initial_covariances=inputs["initial_covariances"],
            noises=inputs["noises"],
            observations=inputs["observations"],
            with_score=True,
            return_trace=False,
            reset_design=reset_design,
            annealed_stages=annealed_stages,
            **{k: v for k, v in TUNED_CONTROLS.items() if k != "reset_policy"},
            reset_policy=TUNED_CONTROLS["reset_policy"],
        )
        trace = None

    # Build fixture
    fixture = {
        "schema": "ledh_golden_master_v1",
        "config": {
            "horizon": horizon,
            "particle_count": particle_count,
            "state_dim": state_dim,
            "direction_count": direction_count,
            "dtype": dtype_str,
            "annealed_stages": annealed_stages,
            "tuned_controls": TUNED_CONTROLS,
        },
        "inputs": {
            "theta": _tensor_to_nested_list(inputs["theta"]),
            "initial_states": _tensor_to_nested_list(inputs["initial_states"]),
            "initial_covariances": _tensor_to_nested_list(inputs["initial_covariances"]),
            "noises": _tensor_to_nested_list(inputs["noises"]),
            "observations": _tensor_to_nested_list(inputs["observations"]),
            "reset_design": _tensor_to_nested_list(reset_design),
        },
        "outputs": {
            "value": _tensor_to_nested_list(value),
            "score": _tensor_to_nested_list(score) if score is not None else None,
        },
        "generation_metadata": {
            "git_commit": git_commit,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "python_version": sys.version,
            "tensorflow_version": tf.__version__,
            "seed": seed,
            "generator_script": __file__,
        },
    }

    # Add diagnostic trace (only keys, not full values for size)
    if trace:
        fixture["trace_keys"] = [list(step_dict.keys()) for step_dict in trace]

    return fixture


def main() -> None:
    """Generate all golden-master fixtures."""
    print(f"Generating golden-master fixtures...")
    print(f"Output directory: {FIXTURE_DIR}")

    # Create output directory
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)

    # Get git commit
    git_commit = _git_commit()
    print(f"Git commit: {git_commit}")

    # Generate fixtures
    total = (
        len(TEST_MATRIX["horizons"])
        * len(TEST_MATRIX["particle_counts"])
        * len(TEST_MATRIX["state_dims"])
        * len(TEST_MATRIX["direction_counts"])
        * len(TEST_MATRIX["dtypes"])
        * len(TEST_MATRIX["annealed_stages"])
    )

    print(f"Total fixtures to generate: {total}")

    generated = 0
    for horizon in TEST_MATRIX["horizons"]:
        for particle_count in TEST_MATRIX["particle_counts"]:
            for state_dim in TEST_MATRIX["state_dims"]:
                for direction_count in TEST_MATRIX["direction_counts"]:
                    for dtype_str in TEST_MATRIX["dtypes"]:
                        for annealed_stages in TEST_MATRIX["annealed_stages"]:
                            # Deterministic seed from config
                            config_str = f"{horizon}_{particle_count}_{state_dim}_{direction_count}_{dtype_str}_{annealed_stages}"
                            seed = int(hashlib.md5(config_str.encode()).hexdigest()[:8], 16) % (2**31)

                            # Generate fixture
                            try:
                                fixture = _generate_fixture(
                                    horizon,
                                    particle_count,
                                    state_dim,
                                    direction_count,
                                    dtype_str,
                                    annealed_stages,
                                    seed,
                                    git_commit,
                                )

                                # Save to file
                                filename = f"golden_T{horizon}_N{particle_count}_d{state_dim}_K{direction_count}_{dtype_str}_anneal{annealed_stages}.json"
                                filepath = FIXTURE_DIR / filename

                                with open(filepath, "w") as f:
                                    json.dump(fixture, f, indent=2)

                                generated += 1
                                if generated % 10 == 0:
                                    print(f"  {generated}/{total} fixtures generated...")

                            except Exception as e:
                                print(f"  FAILED: {config_str}")
                                print(f"    Error: {e}")
                                continue

    print(f"✓ Generated {generated}/{total} fixtures successfully")
    print(f"✓ Output: {FIXTURE_DIR}")

    # Create manifest
    manifest = {
        "schema": "ledh_golden_master_manifest_v1",
        "generation_date": datetime.utcnow().isoformat() + "Z",
        "git_commit": git_commit,
        "python_version": sys.version,
        "tensorflow_version": tf.__version__,
        "fixture_count": generated,
        "test_matrix": TEST_MATRIX,
        "tuned_controls": TUNED_CONTROLS,
        "generator_script": __file__,
    }

    manifest_path = FIXTURE_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"✓ Manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
