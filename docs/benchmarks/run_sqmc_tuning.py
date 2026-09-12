#!/usr/bin/env python3
"""SQMC Tuning: Exact-scope grid search for 3D LGSSM T=20 N=1008.

This script produces repository-issued tuning artifacts for each SQMC route.
"""

import json
import os
import sys
import time
import hashlib
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from bayesfilter.highdim.ledh_canonical_models_tf import diagonal_lgssm_canonical_model
from bayesfilter.highdim.ledh_kalman_oracle_tf import kalman_oracle_value_and_score


def _frozen_observations(horizon: int) -> tf.Tensor:
    """Return frozen observation sequence for LGSSM canonical."""
    # Same sequence used in oracle comparison
    obs_seed = 88001
    rng = np.random.RandomState(obs_seed)
    # 3D observations, T steps
    return tf.constant(rng.randn(horizon, 3), dtype=tf.float64)


def _tuning_grid() -> List[Dict]:
    """Define tuning grid for SQMC controls.

    Coarse grid targeting ~50 configurations:
    - 3 Sinkhorn epsilon values
    - 3 Sinkhorn/balance step combinations (coupled)
    - 3 diagonal correction strengths
    - 2 pairwise correction strengths

    Total: 3 × 3 × 3 × 2 = 54 configurations
    """
    grid = []

    # Sinkhorn/balance parameters (coupled - same steps for both)
    epsilon_values = [4.0, 8.0, 16.0]
    step_pairs = [(4, 4), (8, 8), (16, 16)]  # (sinkhorn, balance)

    # Correction parameters
    diag_strengths = [0.1, 0.15, 0.2]
    pair_strengths = [0.02, 0.03]  # Narrower range, 0.02 is current default

    for epsilon in epsilon_values:
        for sinkhorn_steps, balance_steps in step_pairs:
            for diag_strength in diag_strengths:
                for pair_strength in pair_strengths:
                    grid.append({
                        'reset_epsilon': epsilon,
                        'reset_sinkhorn_steps': sinkhorn_steps,
                        'reset_balance_steps': balance_steps,
                        'correction_strength': diag_strength,
                        'correction_steps': 4,  # Fixed at current default
                        'pairwise_strength': pair_strength,
                        'pairwise_steps': 4,  # Fixed at current default
                    })

    return grid


def _evaluate_controls(
    route: str,
    controls: Dict,
    observations: tf.Tensor,
    theta: tf.Tensor,
    oracle_score: tf.Tensor,
    seed: int,
    horizon: int,
    particle_count: int,
    is_ablation: bool = False,
) -> Dict:
    """Evaluate one control configuration."""

    # Build full controls dict
    # For ablation variant, use conservative controls
    if is_ablation:
        full_controls = {
            **controls,
            'coordinate_cap': 0.97,  # Slightly more conservative
            'coordinate_cap_power': 8,
            'correction_lm_damping': 0.01,
            'correction_lm_scale_floor': 0.0001,
            'correction_trust_radius': 0.5,
            'pairwise_rms_cap': 2.0,
            'reset_policy': 'contract_e',
            'reset_ridge': 1e-5,
        }
    else:
        full_controls = {
            **controls,
            'coordinate_cap': 0.98,
            'coordinate_cap_power': 8,
            'correction_lm_damping': 0.01,
            'correction_lm_scale_floor': 0.0001,
            'correction_trust_radius': 0.5,
            'pairwise_rms_cap': 2.0,
            'reset_policy': 'contract_e',
            'reset_ridge': 1e-5,
        }

    # Import the executor function
    from bayesfilter.highdim.ledh_canonical_score_tf import canonical_value_and_analytical_score as score_fn

    # Call canonical score executor
    try:
        result = score_fn(
            model_factory=diagonal_lgssm_canonical_model,
            observations=observations,
            theta=theta,
            ancestry_route=route,
            controls=full_controls,
            seed=seed,
            horizon=horizon,
            particle_count=particle_count,
            dtype=tf.float32,  # Production target
        )

        # Check validity
        if not result.get('finite', False) or not result.get('valid', False):
            return {
                'valid': False,
                'score_l2_error': float('inf'),
                'cosine_similarity': 0.0,
            }

        # Compute metrics vs oracle
        sqmc_score = tf.constant(result['score'], dtype=tf.float64)

        # L2 error
        score_l2 = tf.norm(sqmc_score - oracle_score).numpy()

        # Cosine similarity
        dot = tf.reduce_sum(sqmc_score * oracle_score).numpy()
        norm_sqmc = tf.norm(sqmc_score).numpy()
        norm_oracle = tf.norm(oracle_score).numpy()
        cosine_sim = dot / (norm_sqmc * norm_oracle) if norm_oracle > 0 and norm_sqmc > 0 else 0.0

        # Relative norm error
        rel_norm_error = abs(norm_sqmc - norm_oracle) / norm_oracle if norm_oracle > 0 else 0.0

        # Fisher-scaled errors (Err_i / sqrt(|oracle_score_i|))
        fisher_scaled = []
        for i in range(len(oracle_score)):
            err = abs(sqmc_score[i] - oracle_score[i])
            oracle_mag = abs(oracle_score[i])
            if oracle_mag > 1e-10:
                fisher_scaled.append(float(err / tf.sqrt(oracle_mag).numpy()))
            else:
                fisher_scaled.append(0.0)

        # Induced HMC parameter error (with epsilon=0.01)
        epsilon_hmc = 0.01
        if norm_oracle > 0:
            induced_hmc_error = epsilon_hmc * (score_l2 / norm_oracle)
        else:
            induced_hmc_error = 0.0

        # Multi-objective tuning score
        # Hard vetoes
        if cosine_sim < 0.9995:
            tuning_score = float('inf')
        elif rel_norm_error > 0.05:
            tuning_score = float('inf')
        elif any(fs > 1.0 for fs in fisher_scaled):
            tuning_score = float('inf')
        else:
            # Primary: L2 error, Secondary: Fisher-scaled and HMC error
            fisher_penalty = 0.1 * np.mean(fisher_scaled)
            hmc_penalty = 0.1 * induced_hmc_error
            tuning_score = score_l2 + fisher_penalty + hmc_penalty

        return {
            'valid': True,
            'score_l2_error': float(score_l2),
            'cosine_similarity': float(cosine_sim),
            'relative_norm_error': float(rel_norm_error),
            'fisher_scaled_errors': fisher_scaled,
            'induced_hmc_error': float(induced_hmc_error),
            'tuning_score': float(tuning_score),
            'value': float(result['value']),
        }

    except Exception as e:
        return {
            'valid': False,
            'score_l2_error': float('inf'),
            'cosine_similarity': 0.0,
            'relative_norm_error': float('inf'),
            'fisher_scaled_errors': [],
            'induced_hmc_error': float('inf'),
            'tuning_score': float('inf'),
            'error': str(e),
        }


def tune_route(
    route: str,
    horizon: int,
    particle_count: int,
    tuning_seeds: List[int],
    output_dir: str,
    is_ablation: bool = False,
):
    """Run tuning campaign for one route."""

    print(f"\n{'='*80}")
    print(f"Tuning route: {route}" + (" (ablation)" if is_ablation else ""))
    print(f"{'='*80}\n")

    # Fixed LGSSM parameters
    theta = tf.constant([0.9, 0.8, 0.7, 0.6, 0.8], dtype=tf.float64)
    observations = _frozen_observations(horizon)

    # Compute oracle once (use float64 for oracle, convert SQMC scores to float64 for comparison)
    oracle = kalman_oracle_value_and_score(
        observations=observations,
        theta=theta,
        theta_to_lgssm_params=diagonal_lgssm_canonical_model,
        dtype=tf.float64,
    )
    oracle_score = oracle['score']

    # Get tuning grid
    grid = _tuning_grid()
    print(f"Grid size: {len(grid)} configurations")
    print(f"Seeds: {len(tuning_seeds)}")
    print(f"Total cells: {len(grid) * len(tuning_seeds)}")
    print()

    # Aggregate results across seeds
    grid_results = []

    for grid_idx, controls in enumerate(grid):
        print(f"Grid [{grid_idx+1}/{len(grid)}]: ", end='', flush=True)

        seed_results = []
        for seed in tuning_seeds:
            result = _evaluate_controls(
                route=route,
                controls=controls,
                observations=observations,
                theta=theta,
                oracle_score=oracle_score,
                seed=seed,
                horizon=horizon,
                particle_count=particle_count,
                is_ablation=is_ablation,
            )
            seed_results.append(result)

        # Aggregate
        valid_count = sum(1 for r in seed_results if r['valid'])
        if valid_count == 0:
            print("ALL INVALID")
            grid_results.append({
                'controls': controls,
                'valid_fraction': 0.0,
                'mean_score_l2': float('inf'),
                'mean_cosine_similarity': 0.0,
            })
            continue

        valid_results = [r for r in seed_results if r['valid']]
        mean_l2 = np.mean([r['score_l2_error'] for r in valid_results])
        mean_cos = np.mean([r['cosine_similarity'] for r in valid_results])
        mean_tuning_score = np.mean([r['tuning_score'] for r in valid_results])

        print(f"Valid={valid_count}/{len(tuning_seeds)}, L2={mean_l2:.4f}, Cos={mean_cos:.6f}, Score={mean_tuning_score:.4f}")

        grid_results.append({
            'controls': controls,
            'valid_fraction': valid_count / len(tuning_seeds),
            'mean_score_l2': float(mean_l2),
            'mean_cosine_similarity': float(mean_cos),
            'mean_tuning_score': float(mean_tuning_score),
            'seed_results': seed_results,
        })

    # Find best configuration (minimize multi-objective tuning score)
    valid_configs = [r for r in grid_results if r['valid_fraction'] > 0.5 and r['mean_tuning_score'] < float('inf')]

    if not valid_configs:
        print("\n⚠️  NO VALID CONFIGURATIONS FOUND")
        print("All configurations either failed validity or violated multi-objective constraints")
        print("(cosine < 0.9995, rel_norm > 0.05, or Fisher-scaled > 1.0)")
        return None

    # Sort by multi-objective tuning score
    valid_configs.sort(key=lambda x: x['mean_tuning_score'])
    best = valid_configs[0]

    print(f"\n{'='*80}")
    print(f"BEST CONFIGURATION")
    print(f"{'='*80}")
    print(f"Tuning score: {best['mean_tuning_score']:.4f}")
    print(f"Score L2 error: {best['mean_score_l2']:.4f}")
    print(f"Cosine similarity: {best['mean_cosine_similarity']:.6f}")
    print(f"Valid fraction: {best['valid_fraction']:.2%}")
    print()
    print("Controls:")
    for k, v in best['controls'].items():
        print(f"  {k}: {v}")
    print()

    # Write artifact
    os.makedirs(output_dir, exist_ok=True)

    artifact = {
        'schema': 'bayesfilter.sqmc_tuning_artifact.v1',
        'route': route,
        'is_ablation': is_ablation,
        'model': 'diagonal_lgssm_canonical',
        'horizon': horizon,
        'particle_count': particle_count,
        'dimensions': {'state': 3, 'observation': 3},
        'backend': 'float32_tf32_gpu',
        'chunk_policy': 'dpf_transport_exact_divisor_cap3000_v1',
        'reset_family': 'contract_e_genut_dual_cap',
        'tuning_seeds': tuning_seeds,
        'tuning_date': datetime.now().isoformat(),
        'git_commit': os.popen('git rev-parse HEAD').read().strip(),
        'tuning_metric': 'multi_objective',
        'tuning_objective': 'minimize L2 error while maintaining cosine >= 0.9995, rel_norm <= 0.05, Fisher-scaled <= 1.0',
        'veto_criteria': {
            'cosine_similarity': 0.9995,
            'relative_norm_error': 0.05,
            'fisher_scaled_max': 1.0,
        },
        'best_controls': best['controls'],
        'best_metrics': {
            'mean_tuning_score': best['mean_tuning_score'],
            'mean_score_l2_error': best['mean_score_l2'],
            'mean_cosine_similarity': best['mean_cosine_similarity'],
            'valid_fraction': best['valid_fraction'],
        },
        'grid_size': len(grid),
        'all_results': grid_results,
    }

    artifact_path = os.path.join(output_dir, 'tuning_artifact.json')
    with open(artifact_path, 'w') as f:
        json.dump(artifact, f, indent=2)

    print(f"✓ Artifact saved: {artifact_path}")

    return artifact


def main():
    """Run tuning for all routes."""

    import argparse
    parser = argparse.ArgumentParser(description='SQMC Tuning')
    parser.add_argument('--mode', choices=['pilot', 'full'], default='full',
                       help='Run pilot (1 route, 2 seeds) or full tuning')
    parser.add_argument('--routes', nargs='+',
                       choices=['iid_dual_cap', 'previous_inverse_cdf', 'repaired_permutation', 'repaired_permutation_ablation'],
                       help='Specific routes to tune (default: all)')
    args = parser.parse_args()

    # Configuration
    horizon = 20
    particle_count = 1008

    if args.mode == 'pilot':
        print("\n" + "="*80)
        print("PILOT MODE: Testing infrastructure")
        print("="*80)
        tuning_seeds = [50001, 50002]  # Just 2 seeds
        routes_to_run = ['iid_dual_cap']  # Just 1 route
    else:
        tuning_seeds = list(range(50001, 50017))  # 16 seeds
        if args.routes:
            routes_to_run = args.routes
        else:
            routes_to_run = ['iid_dual_cap', 'previous_inverse_cdf', 'repaired_permutation', 'repaired_permutation_ablation']

    # Enable GPU memory growth
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✓ GPU memory growth enabled for {len(gpus)} device(s)")
        except RuntimeError as e:
            print(f"⚠️  GPU memory growth setup failed: {e}")

    # Run tuning for each route
    for route in routes_to_run:
        is_ablation = (route == 'repaired_permutation_ablation')

        if is_ablation:
            actual_route = 'repaired_permutation'
            output_dir = f'docs/tuning/sqmc-lgssm-t20-n1008-repaired-permutation-ablation-20260912'
        else:
            actual_route = route
            output_dir = f'docs/tuning/sqmc-lgssm-t20-n1008-{route}-20260912'

        result = tune_route(
            route=actual_route,
            horizon=horizon,
            particle_count=particle_count,
            tuning_seeds=tuning_seeds,
            output_dir=output_dir,
            is_ablation=is_ablation,
        )

        if result is None:
            print(f"\n⚠️  TUNING FAILED FOR {route}")
            if args.mode == 'pilot':
                print("Pilot failed - stopping before full campaign")
                return 1

    print("\n" + "="*80)
    print("TUNING COMPLETE")
    print("="*80)
    return 0


if __name__ == '__main__':
    sys.exit(main())
