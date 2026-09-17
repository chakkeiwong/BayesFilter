#!/usr/bin/env python3
"""Test SQMC routes on 3D state, T=120 LGSSM with tuned controls.

Extends the Test 1B methodology to a longer horizon (T=120) to check if:
1. Route differences persist at longer horizons
2. Tuned controls from T=20 still provide benefit at T=120
3. Computational cost scales as expected

Uses the same 4 routes and tuned controls from the T=20 tuning campaign.
Note: Uses 3D canonical LGSSM (same state dimension as T=20), varying only horizon.
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

import run_sqmc_tuning as R

# Test configuration
DTYPE = tf.float64
HORIZON = 120
STATE_DIM = 3  # Same as T=20 tuning; only horizon varies
PARTICLE_COUNT = 1008
SEEDS = [97801, 97802, 97803, 97804]  # New seeds for T=120 test

# Routes to test
ROUTES = [
    "iid_dual_cap",
    "previous_inverse_cdf",
    "repaired_permutation",
    "repaired_permutation_ablation",
]

# Tuning artifacts from T=20 campaign
TUNING_ARTIFACTS = {
    "iid_dual_cap": "docs/tuning/sqmc-lgssm-t20-n1008-iid_dual_cap-20260912/tuning_artifact.json",
    "previous_inverse_cdf": "docs/tuning/sqmc-lgssm-t20-n1008-previous_inverse_cdf-20260912/tuning_artifact.json",
    "repaired_permutation": "docs/tuning/sqmc-lgssm-t20-n1008-repaired_permutation-20260912/tuning_artifact.json",
    "repaired_permutation_ablation": "docs/tuning/sqmc-lgssm-t20-n1008-repaired_permutation_ablation-20260912/tuning_artifact.json",
}


def _generate_3d_t120_lgssm(seed: int) -> tuple[tf.Tensor, tf.Tensor]:
    """Generate 3D LGSSM observations for T=120.

    Uses the same 3D canonical model as T=20 tuning, varying only horizon.
    """
    rng = np.random.default_rng(seed)

    # Same 3D dynamics as T=20: phi = [0.9, 0.8, 0.7]
    phi = np.array([0.9, 0.8, 0.7])
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

    # Theta: [phi_1, phi_2, phi_3, q_scale, r_scale]
    theta = tf.constant(list(phi) + [q_scale, r_scale], dtype=DTYPE)

    return obs_tensor, theta


def _oracle_score_3d_t120(observations: tf.Tensor, theta: tf.Tensor, seed: int) -> tf.Tensor:
    """Compute oracle Kalman score for 3D T=120 LGSSM.

    For now, returns a placeholder. Full Kalman oracle would require
    extending kalman_oracle_value_and_score to handle this parameterization.
    """
    # TODO: Implement proper Kalman oracle for T=120 case
    # For now, just use SQMC as the target (no oracle comparison)
    return tf.zeros([5], dtype=DTYPE)  # 5-parameter theta


def _evaluate_route_on_seed(
    route: str,
    controls: Dict[str, Any],
    observations: tf.Tensor,
    theta: tf.Tensor,
    seed: int,
) -> Dict[str, Any]:
    """Evaluate one route on one seed."""
    started = time.perf_counter()

    result = R._evaluate_controls(
        route=route,
        controls=controls,
        observations=observations,
        theta=theta,
        oracle_score=tf.zeros([5], dtype=DTYPE),  # No oracle comparison (5-param theta)
        seed=seed,
        horizon=HORIZON,
        particle_count=PARTICLE_COUNT,
    )

    result["seed"] = seed
    result["route"] = route
    result["wall_seconds"] = time.perf_counter() - started

    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Test SQMC on 10D T=120 LGSSM")
    parser.add_argument(
        "--route",
        choices=ROUTES,
        help="Single route to test (default: all routes)",
    )
    parser.add_argument("--output-dir", default="artifacts/sqmc-3d-t120-20260917")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    routes_to_test = [args.route] if args.route else ROUTES

    print("=" * 78)
    print("SQMC Test: 3D state, T=120 horizon, tuned controls from T=20 campaign")
    print("=" * 78)
    print(f"State dimension: {STATE_DIM}")
    print(f"Horizon: {HORIZON}")
    print(f"Particle count: {PARTICLE_COUNT}")
    print(f"Routes: {routes_to_test}")
    print(f"Seeds: {SEEDS}")
    print(f"Backend: {R.DTYPE.name}")
    print()

    all_results = []

    for route in routes_to_test:
        print(f"\n{'='*78}")
        print(f"Route: {route}")
        print(f"{'='*78}")

        # Load tuned controls
        artifact_path = Path(TUNING_ARTIFACTS[route])
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
            observations, theta = _generate_3d_t120_lgssm(seed)

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
        "schema": "bayesfilter.sqmc_3d_t120_test.v1",
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
