#!/usr/bin/env python3
"""Historical-control transfer diagnostic, repaired 2026-09-25.

Reports the full analytical score and matched Kalman errors. Transferred
controls are diagnostic hypotheses, not scope-specific tuning authority.
Use run_sqmc_tuning.py for new-scope calibration, validation and untouched data.
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

import tensorflow as tf
from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
GPU_POLICY_RECORD = (dict(configure_tensorflow_gpu_memory_growth(tf, require_gpu=True))
                     if tf.config.list_physical_devices('GPU') else {'mode':'cpu_reference'})
from bayesfilter.highdim.sqmc_lgssm_tf import LGSSMSpec
from bayesfilter.highdim.sqmc_campaign_tf import evaluate_diagnostic

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


def _generate_10d_t120_lgssm(seed):
    spec = LGSSMSpec('diagonal_ar', STATE_DIM)
    theta = spec.default_theta(DTYPE)
    return spec.simulate(theta, HORIZON, seed), theta


def _oracle_score_10d_t120(observations, theta, seed):
    return LGSSMSpec('diagonal_ar', STATE_DIM).reference_value_and_score(theta, observations)[1]


def _evaluate_route_on_seed(route, controls, observations, theta, seed, *, jit_compile=True):
    return evaluate_diagnostic(LGSSMSpec('diagonal_ar', STATE_DIM), route, controls,
                               observations, theta, seed, PARTICLE_COUNT, jit_compile=jit_compile)


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
    output_dir = output_dir / datetime.now().strftime('%Y%m%dT%H%M%S%f')
    output_dir.mkdir(parents=True, exist_ok=False)

    routes_to_test = [args.route] if args.route else ROUTES

    print("=" * 78)
    print("SQMC Test: 10D state, T=120 horizon, historical T=20 controls (diagnostic only)")
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
            raise FileNotFoundError(f'historical control artifact missing: {artifact_path}')

        tuning_artifact = json.loads(artifact_path.read_text())
        controls = tuning_artifact["best_controls"]

        print("Historical controls (not tuned for this scope):")
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
        "diagnostic_only": True,
        "claim_eligible": False,
        "tuning_status": "historical_cross_scope_controls",
        "gpu_memory_policy": GPU_POLICY_RECORD,
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

    return 0 if all_results and all(r.get('valid', False) for r in all_results) else 1


if __name__ == "__main__":
    sys.exit(main())
