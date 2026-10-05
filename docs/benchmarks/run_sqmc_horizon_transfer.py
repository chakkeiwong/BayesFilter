#!/usr/bin/env python3
"""Historical-control transfer diagnostic, repaired 2026-09-25.

Reports the full analytical score and matched Kalman errors. Transferred
controls are diagnostic hypotheses, not scope-specific tuning authority.
Use run_sqmc_tuning.py for new-scope calibration, validation and untouched data.
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

import tensorflow as tf
from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
GPU_POLICY_RECORD = (dict(configure_tensorflow_gpu_memory_growth(tf, require_gpu=True))
                     if tf.config.list_physical_devices('GPU') else {'mode':'cpu_reference'})
from bayesfilter.highdim.sqmc_lgssm_tf import LGSSMSpec
from bayesfilter.highdim.sqmc_campaign_tf import evaluate_diagnostic

# Test configuration
DTYPE = tf.float64
SEEDS = [60001, 60002, 60003, 60004]  # Different from tuning and Phase 2

# Test configurations: horizon transfer at 3D and 10D
TEST_CONFIGS = [
    {"state_dim": 3, "particle_count": 1008, "horizon": 20, "name": "3D T=20 diagonal-AR diagnostic"},
    {"state_dim": 3, "particle_count": 1008, "horizon": 120, "name": "3D T=120 transfer"},
    {"state_dim": 10, "particle_count": 1000, "horizon": 20, "name": "10D T=20 diagonal-AR diagnostic"},
    {"state_dim": 10, "particle_count": 1000, "horizon": 120, "name": "10D T=120 transfer"},
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


def _generate_lgssm_data(seed, state_dim, horizon):
    spec = LGSSMSpec('diagonal_ar', state_dim)
    theta = spec.default_theta(DTYPE)
    return spec.simulate(theta, horizon, seed), theta


def _evaluate_sqmc(route, controls, observations, theta, seed, state_dim, particle_count, horizon, *, jit_compile=True):
    if observations.shape[0] != horizon:
        raise ValueError('observation horizon mismatch')
    return evaluate_diagnostic(LGSSMSpec('diagonal_ar', state_dim), route, controls,
                               observations, theta, seed, particle_count, jit_compile=jit_compile)


def main() -> int:
    output_dir = Path("artifacts/sqmc-horizon-transfer-20260924")
    output_dir = output_dir / datetime.now().strftime('%Y%m%dT%H%M%S%f')
    output_dir.mkdir(parents=True, exist_ok=False)

    print("=" * 80)
    print("Phase 3: Horizon Transfer Test - T=20 → T=120 at 3D and 10D")
    print("=" * 80)
    print(f"Seeds: {SEEDS}")
    print(f"Configurations: {[c['name'] for c in TEST_CONFIGS]}")
    print(f"Routes: {ROUTES}")
    print()

    all_results = []

    for route in ROUTES:
        print(f"\n{'='*80}")
        print(f"Route: {route}")
        print(f"{'='*80}")

        # Load tuned controls from 3D T=20 campaign
        artifact_path = TUNING_ARTIFACTS[route]
        if not artifact_path.exists():
            print(f"ERROR: Tuning artifact not found: {artifact_path}")
            raise FileNotFoundError(artifact_path)

        tuning_artifact = json.loads(artifact_path.read_text())
        controls = tuning_artifact["best_controls"]

        print("Historical 3D T=20 controls (UNTUNED for this target):")
        for key, value in sorted(controls.items()):
            print(f"  {key}: {value}")
        print()

        for config in TEST_CONFIGS:
            state_dim = config["state_dim"]
            particle_count = config["particle_count"]
            horizon = config["horizon"]
            config_name = config["name"]

            print(f"\n{config_name} (D={state_dim}, T={horizon}, N={particle_count}):")
            print("-" * 40)

            for seed in SEEDS:
                print(f"  Generating data for seed {seed}...")
                observations, theta = _generate_lgssm_data(seed, state_dim, horizon)

                print(f"  Running SQMC...")
                result = _evaluate_sqmc(
                    route, controls, observations, theta, seed, state_dim, particle_count, horizon
                )

                result.update({
                    "route": route,
                    "seed": seed,
                    "state_dim": state_dim,
                    "particle_count": particle_count,
                    "horizon": horizon,
                    "config_name": config_name,
                })

                all_results.append(result)

                status = "ok" if result["valid"] else f"INVALID ({result.get('error', 'unknown')})"
                value_str = f"value={result['value']:.2f}" if result["valid"] else ""
                print(f"    [{route}] seed {seed}: {status} {value_str} wall={result['wall_seconds']:.1f}s")

            print()

    # Save results
    manifest = {
        "diagnostic_only": True,
        "claim_eligible": False,
        "tuning_status": "historical_cross_scope_controls",
        "gpu_memory_policy": GPU_POLICY_RECORD,
        "schema": "bayesfilter.sqmc_horizon_transfer.v1",
        "timestamp": datetime.now().isoformat(),
        "phase": "Phase 3: Horizon Transfer Test",
        "master_program": "sqmc-control-generalization-master-program-2026-09-23.md",
        "seeds": SEEDS,
        "test_configs": TEST_CONFIGS,
        "routes": ROUTES,
        "dtype": DTYPE.name,
        "evidence_role": "diagnostic_only",
        "claim_eligible": False,
        "tuning_status": "cross_scope_historical_controls",
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

    return 0 if total_cells > 0 and valid_cells == total_cells else 1


if __name__ == '__main__':
    sys.exit(main())
