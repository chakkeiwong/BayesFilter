#!/usr/bin/env python3
"""SQMC 10D LGSSM T=120 untuned route comparison.

Tests whether route differences persist at larger scale (10D state, T=120)
using warm-start controls only. This is a descriptive comparison before
deciding whether to invest in tuning artifacts for this configuration.

Model: 10D diagonal LGSSM with stable AR dynamics
Horizon: 120 timesteps
Particle count: 1008 (same as T=20 tests)
Backend: float64 GPU
Seeds: 4 independent replications per route
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
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

DTYPE = tf.float64

ARTIFACT_ROOT = Path("docs/benchmarks/artifacts/sqmc-10d-t120-untuned-20260917")

# Warm-start controls (from T=20 N=1008 baseline)
UNTUNED_CONTROLS = {
    "reset_epsilon": 8.0,
    "reset_sinkhorn_steps": 8,
    "reset_balance_steps": 8,
    "correction_strength": 0.2,
    "correction_steps": 4,
    "pairwise_strength": 0.02,
    "pairwise_steps": 4,
}

ROUTES = [
    "iid_dual_cap",
    "previous_inverse_cdf",
    "repaired_permutation",
    "repaired_permutation_ablation",
]


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unavailable"


def _lgssm_10d_model_and_data(horizon: int, seed: int):
    """Generate 10D LGSSM model and synthetic observations.

    Returns:
        observations: [T, 10] tensor
        theta: [22] parameter vector (10 phi + 10 q_diag + r_scale + obs_bias)
        oracle_score: [22] exact Kalman gradient
    """
    rng = np.random.default_rng(seed)

    # Stable diagonal AR dynamics
    phi = np.array([0.95, 0.90, 0.88, 0.85, 0.82,
                    0.80, 0.78, 0.75, 0.73, 0.70])

    # Process noise (diagonal covariance)
    q_diag = np.full(10, 0.09)  # 0.3^2

    # Observation noise scale
    r_scale = 0.25  # 0.5^2

    # Generate trajectory
    x = np.zeros((horizon, 10))
    y = np.zeros((horizon, 10))

    x[0] = rng.standard_normal(10) * 0.5
    y[0] = x[0] + rng.standard_normal(10) * r_scale

    for t in range(1, horizon):
        x[t] = phi * x[t-1] + rng.standard_normal(10) * np.sqrt(q_diag)
        y[t] = x[t] + rng.standard_normal(10) * r_scale

    observations = tf.constant(y, dtype=DTYPE)

    # Parameter vector: [phi(10), q_diag(10), r_scale(1), obs_bias(1)]
    # For now, simplified to just [phi(10), q_scale(1), r_scale(1)]
    theta = tf.constant(
        list(phi) + [0.3, 0.5],  # 12 parameters total
        dtype=DTYPE
    )

    # Placeholder oracle (would need actual Kalman filter implementation)
    # For descriptive comparison, we'll compare routes to each other
    oracle_score = None

    return observations, theta, oracle_score


def _evaluate_route(
    route: str,
    controls: Dict[str, Any],
    observations: tf.Tensor,
    theta: tf.Tensor,
    seed: int,
    particle_count: int,
) -> Dict[str, Any]:
    """Evaluate one SQMC route on the 10D LGSSM."""
    from bayesfilter.highdim.ledh_alg1_contract import sequential_qmc_gradient_lgssm_10d

    try:
        started = time.perf_counter()

        # Run SQMC gradient estimation
        estimated_score = sequential_qmc_gradient_lgssm_10d(
            route=route,
            observations=observations,
            theta=theta,
            particle_count=particle_count,
            seed=seed,
            **controls
        )

        wall_seconds = time.perf_counter() - started

        # Compute descriptive statistics
        score_norm = float(tf.norm(estimated_score).numpy())

        return {
            "valid": True,
            "route": route,
            "seed": seed,
            "score_norm": score_norm,
            "wall_seconds": wall_seconds,
            "estimated_score": estimated_score.numpy().tolist(),
        }

    except Exception as e:
        return {
            "valid": False,
            "route": route,
            "seed": seed,
            "error": str(e),
        }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="10D LGSSM T=120 untuned route comparison"
    )
    parser.add_argument("--horizon", type=int, default=120)
    parser.add_argument("--particles", type=int, default=1008)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[80001, 80002, 80003, 80004],
        help="Independent seeds for each route"
    )
    parser.add_argument("--output-tag", default="initial")
    args = parser.parse_args()

    output_dir = ARTIFACT_ROOT / args.output_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("SQMC 10D LGSSM T=120 UNTUNED ROUTE COMPARISON")
    print("=" * 78)
    print(f"Model: 10D diagonal LGSSM")
    print(f"Horizon: {args.horizon}")
    print(f"Particles: {args.particles}")
    print(f"Seeds: {args.seeds}")
    print(f"Routes: {', '.join(ROUTES)}")
    print(f"Controls (warm-start):")
    for key, value in sorted(UNTUNED_CONTROLS.items()):
        print(f"  {key}: {value}")
    print()

    # Generate model and data
    print("Generating 10D LGSSM model and observations...")
    observations, theta, oracle_score = _lgssm_10d_model_and_data(
        args.horizon,
        seed=20260917
    )
    print(f"Observations shape: {observations.shape}")
    print(f"Theta shape: {theta.shape}")
    print()

    # Evaluate each route
    results = []
    for route in ROUTES:
        print(f"Route: {route}")
        for seed in args.seeds:
            result = _evaluate_route(
                route=route,
                controls=UNTUNED_CONTROLS,
                observations=observations,
                theta=theta,
                seed=seed,
                particle_count=args.particles,
            )
            results.append(result)

            if result["valid"]:
                print(f"  seed {seed}: norm={result['score_norm']:.4f} "
                      f"time={result['wall_seconds']:.1f}s")
            else:
                print(f"  seed {seed}: FAILED - {result['error']}")
        print()

    # Save results
    manifest = {
        "schema": "bayesfilter.sqmc_10d_untuned_comparison.v1",
        "timestamp": datetime.now().isoformat(),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "config": {
            "model": "lgssm_10d_diagonal",
            "horizon": args.horizon,
            "particle_count": args.particles,
            "state_dim": 10,
            "obs_dim": 10,
            "seeds": args.seeds,
            "backend": "float64_gpu",
        },
        "controls": UNTUNED_CONTROLS,
        "results": results,
    }

    result_path = output_dir / "result.json"
    result_path.write_text(json.dumps(manifest, indent=2))
    print(f"Results saved to: {result_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
