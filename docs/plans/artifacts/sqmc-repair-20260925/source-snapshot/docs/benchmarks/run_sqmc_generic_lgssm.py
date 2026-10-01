#!/usr/bin/env python3
"""SQMC Generic LGSSM: Dimension and horizon parameterized testing.

Uses P44 LGSSM infrastructure to test control transfer across dimensions and horizons.
Part of sqmc-control-generalization-master-program-2026-09-23.md.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple


# GPU memory policy: growth must be requested before TensorFlow initializes.
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import tensorflow as tf

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.insert(0, '/home/chakwong/python/src')

from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth

# Memory growth must be enabled and verified BEFORE anything initializes the GPU runtime
# For CPU smoke tests, GPU is optional
_GPU_AVAILABLE = len(tf.config.list_physical_devices('GPU')) > 0
if _GPU_AVAILABLE:
    _GPU_POLICY = dict(configure_tensorflow_gpu_memory_growth(tf, require_gpu=True))
else:
    _GPU_POLICY = {'gpu_policy': 'cpu_only', 'devices': []}

from bayesfilter.highdim.ledh_canonical_score_tf import canonical_value_and_analytical_score
from bayesfilter.highdim.sqmc_tf import randomized_halton_gaussian, randomized_halton_joint
from bayesfilter.linear.kalman_tf import tf_linear_gaussian_log_likelihood
from bayesfilter.structural_tf import affine_structural_to_linear_gaussian_tf

# Import P44 LGSSM infrastructure
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../tests/highdim')))
from bayesfilter.highdim.sqmc_lgssm_tf import LGSSMSpec
from bayesfilter.highdim.sqmc_campaign_tf import evaluate_diagnostic


DTYPE = tf.float64
GPU_POLICY_RECORD: Dict = _GPU_POLICY

# Repo root for absolute paths
REPO_ROOT = Path(__file__).resolve().parents[2]


def _p44_theta(dim):
    return LGSSMSpec('p44', dim).default_theta(DTYPE)


def _p44_observations(dim, horizon, seed):
    spec = LGSSMSpec('p44', dim)
    return spec.simulate(spec.default_theta(DTYPE), horizon, seed)


def _p44_oracle_score(theta, observations, dim):
    return LGSSMSpec('p44', dim).reference_value_and_score(theta, observations)


def _p44_nonlinear_model(theta, dim):
    return LGSSMSpec('p44', dim).model(theta)


def _reset_design(particle_count: int, dimension: int) -> tf.Tensor:
    """Contract-E reset design matrix."""
    if particle_count % (2 * dimension):
        raise ValueError(
            f"particle count {particle_count} must be divisible by 2 * state dimension {dimension} "
            f"(required: {particle_count} % {2*dimension} == 0)"
        )
    base = tf.concat(
        [tf.eye(dimension, dtype=DTYPE), -tf.eye(dimension, dtype=DTYPE)], axis=0
    )
    return tf.tile(base, [particle_count // (2 * dimension), 1])


def _evaluate_generic_sqmc(route, controls, observations, theta, oracle_score, seed, horizon, particle_count, state_dim, *, jit_compile=True):
    if observations.shape[0] != horizon:
        raise ValueError('observation horizon mismatch')
    # Recompute the oracle from the shared specification; reject stale comparators.
    spec = LGSSMSpec('p44', state_dim)
    _, exact_score = spec.reference_value_and_score(theta, observations)
    tf.debugging.assert_near(oracle_score, exact_score, atol=1e-8, rtol=1e-7)
    return evaluate_diagnostic(spec, route, controls, observations, theta, seed, particle_count, jit_compile=jit_compile)


def smoke_test(args):
    """Phase 0: CPU smoke test across dimensions."""
    print("=" * 80)
    print("Phase 0: SQMC Generic LGSSM Smoke Test")
    print("=" * 80)
    print(f"Dimensions: {args.dimensions}")
    print(f"Horizon: {args.horizon}")
    print(f"Particles: {args.particles}")
    print(f"Seeds: {args.seeds}")
    print()

    if not args.dimensions or not args.seeds or args.horizon < 1:
        raise ValueError('nonempty dimensions/seeds and a positive horizon are required')
    if any(d < 1 or args.particles < 2*d or args.particles % (2*d) for d in args.dimensions):
        raise ValueError('every requested dimension requires a positive N divisible by 2D')
    results = []

    for dim in args.dimensions:
        print(f"\nTesting dimension {dim}...")

        for seed in args.seeds:
            theta = _p44_theta(dim)
            observations = _p44_observations(dim, args.horizon, seed)
            oracle_value, oracle_score = _p44_oracle_score(theta, observations, dim)

            # Explicit mechanics fixture controls; no numerical-policy promotion
            controls = {
                'reset_epsilon': 8.0,
                'reset_sinkhorn_steps': 8,
                'reset_balance_steps': 8,
                'correction_steps': 4,
                'correction_strength': 0.2,
                'pairwise_steps': 4,
                'pairwise_strength': 0.03,
            }

            result = _evaluate_generic_sqmc(
                route='iid_dual_cap',
                controls=controls,
                observations=observations,
                theta=theta,
                oracle_score=oracle_score,
                seed=seed,
                horizon=args.horizon,
                particle_count=args.particles,
                state_dim=dim,
            )

            result['dimension'] = dim
            result['seed'] = seed
            results.append(result)

            status = "PASS" if result['valid'] else f"FAIL ({result.get('error', 'unknown')})"
            print(f"  dim={dim} seed={seed}: {status}")

    # Save results
    output_dir = Path(args.output_dir)
    output_dir = output_dir / datetime.now().strftime('%Y%m%dT%H%M%S%f')
    output_dir.mkdir(parents=True, exist_ok=False)

    manifest = {
        'schema': 'bayesfilter.sqmc_generic_smoke.v2',
        'gpu_memory_policy': GPU_POLICY_RECORD,
        'diagnostic_only': True,
        'claim_eligible': False,
        'timestamp': datetime.now().isoformat(),
        'dimensions': args.dimensions,
        'horizon': args.horizon,
        'particles': args.particles,
        'seeds': args.seeds,
        'results': results,
    }

    output_file = output_dir / 'smoke_results.json'
    output_file.write_text(json.dumps(manifest, indent=2))

    print(f"\nResults saved to: {output_file}")
    print(f"Valid: {sum(1 for r in results if r['valid'])}/{len(results)}")

    return 0 if len(results) == len(args.dimensions)*len(args.seeds) and all(r['valid'] for r in results) else 1


def main():
    parser = argparse.ArgumentParser(
        description="SQMC Generic LGSSM - Dimension and horizon transfer testing"
    )
    parser.add_argument('--mode', required=True,
                       choices=['smoke', 'p44-replication', 'dimension-transfer', 'horizon-transfer'],
                       help='Execution mode')
    parser.add_argument('--dimensions', type=int, nargs='+', help='State dimensions to test (smoke mode)')
    parser.add_argument('--dimension', type=int, help='State dimension (other modes)')
    parser.add_argument('--horizon', type=int, required=True, help='Time horizon T')
    parser.add_argument('--particles', type=int, required=True, help='Particle count N')
    parser.add_argument('--seeds', type=int, nargs='+', required=True, help='Random seeds')
    parser.add_argument('--routes', nargs='*', help='SQMC routes to test')
    parser.add_argument('--controls', help='Path to tuning artifact JSON')
    parser.add_argument('--output-dir', required=True, help='Output directory')

    args = parser.parse_args()

    if args.mode == 'smoke':
        if not args.dimensions:
            parser.error("--dimensions required for smoke mode")
        return smoke_test(args)
    else:
        print("Other modes not yet implemented - Phase 0 only")
        return 1


if __name__ == '__main__':
    sys.exit(main())
