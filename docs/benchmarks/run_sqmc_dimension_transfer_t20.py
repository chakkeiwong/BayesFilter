#!/usr/bin/env python3
"""Phase 2: Dimension Transfer Test - 3D tuned controls on 10D T=20.

Tests whether controls tuned at 3D T=20 transfer to 10D T=20 without retuning.
Part of sqmc-control-generalization-master-program-2026-09-23.md Phase 2.
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import tensorflow as tf

# Test configuration
DTYPE = tf.float64
HORIZON = 20
SEEDS = [50001, 50002, 50003, 50004]  # Same seeds as tuning campaign

# Test dimensions
TEST_CONFIGS = [
    {"state_dim": 3, "particle_count": 1008, "name": "3D baseline"},
    {"state_dim": 10, "particle_count": 1000, "name": "10D transfer"},
]

# Routes to test
ROUTES = [
    "iid_dual_cap",
    "previous_inverse_cdf",
    "repaired_permutation",
    "repaired_permutation_ablation",
]

REPO_ROOT = Path(__file__).resolve().parents[2]

# Tuning artifacts from 3D T=20 campaign
TUNING_ARTIFACTS = {
    "iid_dual_cap": REPO_ROOT / "docs/tuning/sqmc-lgssm-t20-n1008-iid_dual_cap-20260912/tuning_artifact.json",
    "previous_inverse_cdf": REPO_ROOT / "docs/tuning/sqmc-lgssm-t20-n1008-previous_inverse_cdf-20260912/tuning_artifact.json",
    "repaired_permutation": REPO_ROOT / "docs/tuning/sqmc-lgssm-t20-n1008-repaired_permutation-20260912/tuning_artifact.json",
    "repaired_permutation_ablation": REPO_ROOT / "docs/tuning/sqmc-lgssm-t20-n1008-repaired_permutation_ablation-20260912/tuning_artifact.json",
}


def _generate_lgssm_data(seed: int, state_dim: int, horizon: int):
    """Generate LGSSM data using P44-style parameterization."""
    rng = np.random.default_rng(seed)

    # Dimension-dependent dynamics: phi decays from 0.95 for dim 1 to 0.50 for dim 10+
    phi = np.array([max(0.50, 0.95 - 0.05 * i) for i in range(state_dim)])
    q_scale = 0.6
    r_scale = 0.8

    # Initial state
    x = rng.normal(0, 1.0, state_dim)

    # Generate trajectory
    observations = []
    for t in range(horizon):
        if t > 0:
            x = phi * x + rng.normal(0, q_scale, state_dim)
        obs = x + rng.normal(0, r_scale, state_dim)
        observations.append(obs)

    obs_tensor = tf.constant(np.array(observations), dtype=DTYPE)
    theta = tf.constant(list(phi) + [q_scale, r_scale], dtype=DTYPE)

    return obs_tensor, theta


def _evaluate_sqmc(
    route: str,
    controls: dict,
    observations: tf.Tensor,
    theta: tf.Tensor,
    seed: int,
    state_dim: int,
    particle_count: int,
) -> dict:
    """Evaluate SQMC on given data with dimension-generic infrastructure."""
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

    # Generate particles
    if route == 'iid_dual_cap':
        initial_states = tf.random.stateless_normal(
            [particle_count, state_dim], [seed, 101], dtype=DTYPE
        )
        process_noise = tf.stack([
            tf.random.stateless_normal(
                [particle_count, state_dim], [seed, 1001 + t], dtype=DTYPE
            )
            for t in range(HORIZON)
        ])
        ancestor_uniforms = tf.zeros([HORIZON, particle_count], DTYPE)
    else:
        initial_states = randomized_halton_gaussian(
            num_particles=particle_count,
            dimension=state_dim,
            seed=seed,
            salt=301,
            dtype=DTYPE,
        )
        process_rows = []
        ancestor_rows = []
        for t in range(HORIZON):
            raw, ancestors, innovations = randomized_halton_joint(
                num_particles=particle_count,
                state_dimension=state_dim,
                seed=seed,
                salt=3001 + t,
                dtype=DTYPE,
            )
            process_rows.append(tf.math.ndtri(innovations))
            ancestor_rows.append(ancestors)
        process_noise = tf.stack(process_rows)
        ancestor_uniforms = tf.stack(ancestor_rows)

    # Build dimension-generic model
    phi_diag = theta[:state_dim]
    q_scale = theta[state_dim]
    r_scale = theta[state_dim + 1]

    log_two_pi = tf.constant(np.log(2.0 * np.pi), DTYPE)

    def transition_mean_fn(theta_arg, points):
        phi = theta_arg[:state_dim]
        return points * phi[None, :]

    def transition_mean_tangent_fn(theta_arg, points, d_points):
        phi = theta_arg[:state_dim]
        return d_points * phi[None, :]

    def _scaled_gaussian(points, means, scale_val):
        residual = points - means
        return -0.5 * (
            tf.reduce_sum(tf.square(residual), axis=1) / tf.square(scale_val)
            + float(state_dim) * (log_two_pi + 2.0 * tf.math.log(scale_val))
        )

    def transition_log_density_fn(theta_arg, points, ancestors_mean):
        q = theta_arg[state_dim]
        return _scaled_gaussian(points, ancestors_mean, q)

    def transition_log_density_tangent_fn(theta_arg, points, ancestors_mean, d_points, d_means):
        return tf.zeros([tf.shape(points)[0]], DTYPE)

    def observation_log_density_fn(theta_arg, points, observation):
        observed = points
        target = tf.broadcast_to(observation[None, :], tf.shape(observed))
        r = theta_arg[state_dim + 1]
        return _scaled_gaussian(target, observed, r)

    def observation_log_density_tangent_fn(theta_arg, points, observation, d_points):
        return tf.zeros([tf.shape(points)[0]], DTYPE)

    def process_covariance_tangent_fn(theta_arg):
        return tf.zeros([state_dim, state_dim], dtype=DTYPE)

    def observation_covariance_tangent_fn(theta_arg):
        return tf.zeros([state_dim, state_dim], dtype=DTYPE)

    def observation_fn(points):
        return points

    def observation_jacobian_fn(points):
        return tf.broadcast_to(
            tf.eye(state_dim, dtype=DTYPE), [tf.shape(points)[0], state_dim, state_dim]
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
        process_covariance=tf.square(q_scale) * tf.eye(state_dim, dtype=DTYPE),
        observation_covariance=tf.square(r_scale) * tf.eye(state_dim, dtype=DTYPE),
    )

    # Reset design for Contract-E
    design = tf.concat([tf.eye(state_dim, dtype=DTYPE), -tf.eye(state_dim, dtype=DTYPE)], axis=0)
    design = tf.tile(design, [particle_count // (2 * state_dim), 1])

    initial_covariances = tf.eye(state_dim, batch_shape=[particle_count], dtype=DTYPE)

    try:
        value, _ = canonical_value_and_analytical_score(
            model,
            theta,
            initial_states,
            initial_covariances,
            process_noise,
            observations,
            flow_substeps=8,
            with_score=False,
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
            "valid": True,
            "value": float(value.numpy()),
            "wall_seconds": time.perf_counter() - started,
        }
    except Exception as e:
        result = {
            "valid": False,
            "error": str(e),
            "wall_seconds": time.perf_counter() - started,
        }

    return result


def main() -> int:
    output_dir = Path("artifacts/sqmc-dimension-transfer-t20-20260924")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Phase 2: Dimension Transfer Test - 3D → 10D at T=20")
    print("=" * 80)
    print(f"Horizon: T={HORIZON}")
    print(f"Seeds: {SEEDS}")
    print(f"Configurations: {[c['name'] for c in TEST_CONFIGS]}")
    print(f"Routes: {ROUTES}")
    print()

    all_results = []

    for route in ROUTES:
        print(f"\n{'='*80}")
        print(f"Route: {route}")
        print(f"{'='*80}")

        # Load tuned controls from 3D campaign
        artifact_path = TUNING_ARTIFACTS[route]
        if not artifact_path.exists():
            print(f"ERROR: Tuning artifact not found: {artifact_path}")
            continue

        tuning_artifact = json.loads(artifact_path.read_text())
        controls = tuning_artifact["best_controls"]

        print("Controls (tuned from 3D T=20):")
        for key, value in sorted(controls.items()):
            print(f"  {key}: {value}")
        print()

        for config in TEST_CONFIGS:
            state_dim = config["state_dim"]
            particle_count = config["particle_count"]
            config_name = config["name"]

            print(f"\n{config_name} (D={state_dim}, N={particle_count}):")
            print("-" * 40)

            for seed in SEEDS:
                print(f"  Generating data for seed {seed}...")
                observations, theta = _generate_lgssm_data(seed, state_dim, HORIZON)

                print(f"  Running SQMC...")
                result = _evaluate_sqmc(
                    route, controls, observations, theta, seed, state_dim, particle_count
                )

                result.update({
                    "route": route,
                    "seed": seed,
                    "state_dim": state_dim,
                    "particle_count": particle_count,
                    "config_name": config_name,
                })

                all_results.append(result)

                status = "ok" if result["valid"] else f"INVALID ({result.get('error', 'unknown')})"
                value_str = f"value={result['value']:.2f}" if result["valid"] else ""
                print(f"    [{route}] seed {seed}: {status} {value_str} wall={result['wall_seconds']:.1f}s")

            print()

    # Save results
    manifest = {
        "schema": "bayesfilter.sqmc_dimension_transfer_t20.v1",
        "timestamp": datetime.now().isoformat(),
        "phase": "Phase 2: Dimension Transfer Test",
        "master_program": "sqmc-control-generalization-master-program-2026-09-23.md",
        "horizon": HORIZON,
        "seeds": SEEDS,
        "test_configs": TEST_CONFIGS,
        "routes": ROUTES,
        "dtype": DTYPE.name,
        "results": all_results,
    }

    output_file = output_dir / f"result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_file.write_text(json.dumps(manifest, indent=2))

    print("=" * 80)
    print(f"Results saved to: {output_file}")
    print()

    # Summary
    total_cells = len(all_results)
    valid_cells = sum(1 for r in all_results if r["valid"])
    print(f"Summary:")
    print(f"  Total cells: {total_cells}")
    print(f"  Valid: {valid_cells}")
    print(f"  Invalid: {total_cells - valid_cells}")
    print("=" * 80)

    return 0 if valid_cells == total_cells else 1


if __name__ == '__main__':
    sys.exit(main())
