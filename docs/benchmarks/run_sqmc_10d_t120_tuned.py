#!/usr/bin/env python3
"""Test SQMC routes on 10D state, T=120 LGSSM with tuned controls.

Extends the Test 1B methodology to higher dimension (10D) and longer horizon (T=120) to check if:
1. Route differences persist at higher dimensions
2. Tuned controls from T=20 still provide benefit at T=120 and 10D
3. Computational cost scales as expected

Uses the same 4 routes and tuned controls from the T=20 tuning campaign.
Note: Uses 10D state (higher than T=20 3D), varying both state dimension and horizon.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, "/home/chakwong/python/src")

import numpy as np
import tensorflow as tf

# Note: Cannot import run_sqmc_tuning directly due to module-level GPU init
# Import only what we need after TensorFlow is ready

# Test configuration
DTYPE = tf.float64
HORIZON = 120
STATE_DIM = 10  # Higher dimension than T=20 tuning (was 3D)
PARTICLE_COUNT = 1000  # Must be divisible by 2*STATE_DIM=20 for Contract-E reset design
SEEDS = [97801, 97802, 97803, 97804]  # New seeds for T=120 test

# Routes to test
ROUTES = [
    "iid_dual_cap",
    "previous_inverse_cdf",
    "repaired_permutation",
    "repaired_permutation_ablation",
]

# Repo root for absolute paths
REPO_ROOT = Path(__file__).resolve().parents[2]

# Tuning artifacts from T=20 campaign (3D state)
TUNING_ARTIFACTS = {
    "iid_dual_cap": REPO_ROOT / "docs/tuning/sqmc-lgssm-t20-n1008-iid_dual_cap-20260912/tuning_artifact.json",
    "previous_inverse_cdf": REPO_ROOT / "docs/tuning/sqmc-lgssm-t20-n1008-previous_inverse_cdf-20260912/tuning_artifact.json",
    "repaired_permutation": REPO_ROOT / "docs/tuning/sqmc-lgssm-t20-n1008-repaired_permutation-20260912/tuning_artifact.json",
    "repaired_permutation_ablation": REPO_ROOT / "docs/tuning/sqmc-lgssm-t20-n1008-repaired_permutation_ablation-20260912/tuning_artifact.json",
}


def _generate_10d_t120_lgssm(seed: int) -> tuple[tf.Tensor, tf.Tensor]:
    """Generate 10D LGSSM observations for T=120.

    Uses 10D canonical model with decaying diagonal dynamics.
    """
    rng = np.random.default_rng(seed)

    # 10D dynamics: phi = [0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50]
    phi = np.array([0.95 - 0.05 * i for i in range(STATE_DIM)])
    q_scale = 0.6
    r_scale = 0.8

    # Initial state
    x = rng.normal(0, 1.0, STATE_DIM)

    # Generate trajectory
    observations = []
    for t in range(HORIZON):
        if t > 0:
            x = phi * x + rng.normal(0, q_scale, STATE_DIM)
        obs = x + rng.normal(0, r_scale, STATE_DIM)
        observations.append(obs)

    obs_tensor = tf.constant(np.array(observations), dtype=DTYPE)

    # Theta: [phi_1, ..., phi_10, q_scale, r_scale]
    theta = tf.constant(list(phi) + [q_scale, r_scale], dtype=DTYPE)

    return obs_tensor, theta


def _oracle_score_10d_t120(observations: tf.Tensor, theta: tf.Tensor, seed: int) -> tf.Tensor:
    """Compute oracle Kalman score for 10D T=120 LGSSM.

    For now, returns a placeholder. Full Kalman oracle would require
    extending kalman_oracle_value_and_score to handle this parameterization.
    """
    # TODO: Implement proper Kalman oracle for T=120 10D case
    # For now, just use SQMC as the target (no oracle comparison)
    return tf.zeros([12], dtype=DTYPE)  # 12-parameter theta (10 phi + 2 scales)


def _evaluate_route_on_seed(
    route: str,
    controls: Dict[str, Any],
    observations: tf.Tensor,
    theta: tf.Tensor,
    seed: int,
) -> Dict[str, Any]:
    """Evaluate one route on one seed using self-contained evaluation."""
    started = time.perf_counter()

    from bayesfilter.highdim.ledh_canonical_score_tf import canonical_value_and_analytical_score
    from bayesfilter.highdim.sqmc_tf import randomized_halton_gaussian, randomized_halton_joint
    from bayesfilter.highdim.ledh_canonical_models_tf import NonlinearScoreModel

    # Map route to ancestry_policy
    ancestry_map = {
        'iid_dual_cap': 'existing_one_to_one',
        'previous_inverse_cdf': 'hilbert_inverse_cdf',
        'repaired_permutation': 'hilbert_permutation_one_to_one',
        'repaired_permutation_ablation': 'hilbert_permutation_one_to_one',
    }
    ancestry_policy = ancestry_map.get(route, 'existing_one_to_one')

    # Generate initial states and process noise
    if route == 'iid_dual_cap':
        initial_states = tf.random.stateless_normal(
            [PARTICLE_COUNT, STATE_DIM], [seed, 101], dtype=DTYPE
        )
        process_noise = tf.stack([
            tf.random.stateless_normal(
                [PARTICLE_COUNT, STATE_DIM], [seed, 1001 + t], dtype=DTYPE
            )
            for t in range(HORIZON)
        ])
        ancestor_uniforms = tf.zeros([HORIZON, PARTICLE_COUNT], DTYPE)
    else:
        initial_states = randomized_halton_gaussian(
            num_particles=PARTICLE_COUNT,
            dimension=STATE_DIM,
            seed=seed,
            salt=301,
            dtype=DTYPE,
        )
        process_rows = []
        ancestor_rows = []
        for t in range(HORIZON):
            raw, ancestors, innovations = randomized_halton_joint(
                num_particles=PARTICLE_COUNT,
                state_dimension=STATE_DIM,
                seed=seed,
                salt=3001 + t,
                dtype=DTYPE,
            )
            process_rows.append(tf.math.ndtri(innovations))
            ancestor_rows.append(ancestors)
        process_noise = tf.stack(process_rows)
        ancestor_uniforms = tf.stack(ancestor_rows)

    # Build dimension-generic model for 10D
    # theta = [phi_1...phi_10, q_scale, r_scale] (12 params)
    phi_diag = theta[:STATE_DIM]
    q_scale = theta[STATE_DIM]
    r_scale = theta[STATE_DIM + 1]

    log_two_pi = tf.constant(np.log(2.0 * np.pi), DTYPE)
    _direction = [tf.zeros([len(theta)], DTYPE)]

    def set_score_direction(direction):
        _direction[0] = tf.convert_to_tensor(direction, DTYPE)

    def transition_mean_fn(theta_arg, points):
        phi = theta_arg[:STATE_DIM]
        return points * phi[None, :]

    def transition_mean_tangent_fn(theta_arg, points, d_points):
        phi = theta_arg[:STATE_DIM]
        return d_points * phi[None, :]

    def _scaled_gaussian(points, means, scale_val):
        residual = points - means
        return -0.5 * (
            tf.reduce_sum(tf.square(residual), axis=1) / tf.square(scale_val)
            + float(STATE_DIM) * (log_two_pi + 2.0 * tf.math.log(scale_val))
        )

    def transition_log_density_fn(theta_arg, points, ancestors_mean):
        q = theta_arg[STATE_DIM]
        return _scaled_gaussian(points, ancestors_mean, q)

    def transition_log_density_tangent_fn(theta_arg, points, ancestors_mean, d_points, d_means):
        return tf.zeros([tf.shape(points)[0]], DTYPE)

    def observation_log_density_fn(theta_arg, points, observation):
        observed = points
        target = tf.broadcast_to(observation[None, :], tf.shape(observed))
        r = theta_arg[STATE_DIM + 1]
        return _scaled_gaussian(target, observed, r)

    def observation_log_density_tangent_fn(theta_arg, points, observation, d_points):
        return tf.zeros([tf.shape(points)[0]], DTYPE)

    def process_covariance_tangent_fn(theta_arg):
        return tf.zeros([STATE_DIM, STATE_DIM], dtype=DTYPE)

    def observation_covariance_tangent_fn(theta_arg):
        return tf.zeros([STATE_DIM, STATE_DIM], dtype=DTYPE)

    def observation_fn(points):
        return points

    def observation_jacobian_fn(points):
        return tf.broadcast_to(
            tf.eye(STATE_DIM, dtype=DTYPE), [tf.shape(points)[0], STATE_DIM, STATE_DIM]
        )

    def observation_tangent_fn(points, d_points):
        return d_points

    model = NonlinearScoreModel(
        transition_mean_fn=transition_mean_fn,
        transition_mean_tangent_fn=transition_mean_tangent_fn,
        transition_log_density_fn=transition_log_density_fn,
        transition_log_density_tangent_fn=transition_log_density_tangent_fn,
        process_covariance_tangent_fn=process_covariance_tangent_fn,
        observation_fn=observation_fn,
        observation_jacobian_fn=observation_jacobian_fn,
        observation_tangent_fn=observation_tangent_fn,
        observation_log_density_fn=observation_log_density_fn,
        observation_log_density_tangent_fn=observation_log_density_tangent_fn,
        observation_covariance_tangent_fn=observation_covariance_tangent_fn,
        process_covariance=tf.square(q_scale) * tf.eye(STATE_DIM, dtype=DTYPE),
        observation_covariance=tf.square(r_scale) * tf.eye(STATE_DIM, dtype=DTYPE),
    )

    # Reset design for Contract-E
    design = tf.concat([tf.eye(STATE_DIM, dtype=DTYPE), -tf.eye(STATE_DIM, dtype=DTYPE)], axis=0)
    design = tf.tile(design, [PARTICLE_COUNT // (2 * STATE_DIM), 1])

    initial_covariances = tf.eye(STATE_DIM, batch_shape=[PARTICLE_COUNT], dtype=DTYPE)

    try:
        # Compute value and score
        value, score = canonical_value_and_analytical_score(
            model,
            theta,
            initial_states,
            initial_covariances,
            process_noise,
            observations,
            flow_substeps=8,
            with_score=True,
            reset_policy='contract_e',
            reset_design=design,
            reset_epsilon=controls.get('reset_epsilon', 8.0),
            reset_sinkhorn_steps=controls.get('reset_sinkhorn_steps', 8),
            reset_balance_steps=controls.get('reset_balance_steps', 8),
            reset_ridge=1e-5,
            correction_steps=controls.get('correction_steps', 4),
            correction_strength=controls.get('correction_strength', 0.2),
            correction_lm_damping=0.01,
            correction_lm_scale_floor=0.0001,
            correction_trust_radius=0.5,
            pairwise_steps=controls.get('pairwise_steps', 4),
            pairwise_strength=controls.get('pairwise_strength', 0.03),
            pairwise_rms_cap=2.0,
            coordinate_cap=0.98,
            coordinate_cap_power=8,
            ancestry_policy=ancestry_policy,
            process_ancestor_uniforms=ancestor_uniforms,
            state_map_policy='adaptive_empirical',
            hilbert_bits=12,
        )

        result = {
            "seed": seed,
            "route": route,
            "valid": True,
            "value": float(value.numpy()),
            "score": score.numpy().tolist(),
            "wall_seconds": time.perf_counter() - started,
        }
    except Exception as e:
        result = {
            "seed": seed,
            "route": route,
            "valid": False,
            "error": str(e),
            "wall_seconds": time.perf_counter() - started,
        }

    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Test SQMC on 10D T=120 LGSSM")
    parser.add_argument(
        "--route",
        choices=ROUTES,
        help="Single route to test (default: all routes)",
    )
    parser.add_argument("--output-dir", default="artifacts/sqmc-10d-t120-20260922")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    routes_to_test = [args.route] if args.route else ROUTES

    print("=" * 78)
    print("SQMC Test: 10D state, T=120 horizon, tuned controls from T=20 campaign")
    print("=" * 78)
    print(f"State dimension: {STATE_DIM}")
    print(f"Horizon: {HORIZON}")
    print(f"Particle count: {PARTICLE_COUNT}")
    print(f"Routes: {routes_to_test}")
    print(f"Seeds: {SEEDS}")
    print(f"Backend: {DTYPE.name}")
    print()

    all_results = []

    for route in routes_to_test:
        print(f"\n{'='*78}")
        print(f"Route: {route}")
        print(f"{'='*78}")

        # Load tuned controls
        artifact_path = TUNING_ARTIFACTS[route]
        if not artifact_path.exists():
            print(f"ERROR: Tuning artifact not found: {artifact_path}")
            continue

        tuning_artifact = json.loads(artifact_path.read_text())
        controls = tuning_artifact["best_controls"]

        print("Tuned controls:")
        for key, value in sorted(controls.items()):
            print(f"  {key}: {value}")
        print()

        # Run on each seed
        for seed in SEEDS:
            print(f"Generating data for seed {seed}...")
            observations, theta = _generate_10d_t120_lgssm(seed)

            print(f"Running SQMC for route={route}, seed={seed}...")
            result = _evaluate_route_on_seed(route, controls, observations, theta, seed)

            all_results.append(result)

            status = "ok" if result.get("valid") else f"INVALID ({result.get('error')})"
            score_info = ""
            if result.get("valid") and "score" in result:
                score_norm = float(tf.norm(result["score"]).numpy())
                score_info = f"score_norm={score_norm:.6f}"

            print(f"  [{route}] seed {seed}: {status} {score_info} "
                  f"wall={result['wall_seconds']:.1f}s\n")

    # Save results
    manifest = {
        "schema": "bayesfilter.sqmc_10d_t120_test.v1",
        "timestamp": datetime.now().isoformat(),
        "state_dim": STATE_DIM,
        "horizon": HORIZON,
        "particle_count": PARTICLE_COUNT,
        "seeds": SEEDS,
        "routes": routes_to_test,
        "dtype": DTYPE.name,
        "results": all_results,
    }

    output_file = output_dir / f"result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_file.write_text(json.dumps(manifest, indent=2))

    print(f"\nResults saved to: {output_file}")
    print(f"\nSummary:")
    print(f"  Total cells: {len(all_results)}")
    print(f"  Valid: {sum(1 for r in all_results if r.get('valid'))}")
    print(f"  Invalid: {sum(1 for r in all_results if not r.get('valid'))}")

    total_time = sum(r["wall_seconds"] for r in all_results)
    print(f"  Total wall time: {total_time:.1f}s ({total_time/60:.1f} min)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
