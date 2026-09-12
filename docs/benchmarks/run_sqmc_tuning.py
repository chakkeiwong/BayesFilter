#!/usr/bin/env python3
"""SQMC Tuning: Exact-scope grid search for 3D LGSSM T=20 N=1008.

This script produces repository-issued tuning artifacts for each SQMC route.
Uses Pareto-optimal multi-objective selection from established tools.
"""

import json
import os
import sys
import time
import hashlib
from datetime import datetime
from typing import Dict, List, Tuple
from dataclasses import dataclass

import numpy as np
import tensorflow as tf

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Add ~/python for multi-objective tools
sys.path.insert(0, '/home/chakwong/python/src')

from bayesfilter.highdim.ledh_canonical_models_tf import diagonal_lgssm_canonical_model
from bayesfilter.highdim.ledh_kalman_oracle_tf import kalman_oracle_value_and_score

# Import Pareto dominance tools
from common_utils.tf_multiobjective.population_switching import nondominated


@dataclass
class ConfigEvaluation:
    """Evaluation entry for Pareto analysis."""
    config_idx: int
    objectives: tuple  # (L2, 1-cosine, rel_norm, fisher, hmc) - all minimize
    controls: dict
    mean_cosine: float
    mean_l2: float
    mean_rel_norm: float
    mean_fisher: float
    mean_hmc: float
    valid_fraction: float

    @property
    def replica_id(self):
        return self.config_idx

    @property
    def candidate_id(self):
        return str(self.config_idx)

    @property
    def method(self):
        return 'grid_search'

    @property
    def parameters(self):
        return self.controls

    @property
    def valid(self):
        return self.valid_fraction > 0.5

    def checked(self):
        """Required for nondominated() API."""
        return self


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


def find_pareto_optimal(grid_results: List[Dict], tuning_seeds: List[int]) -> List[ConfigEvaluation]:
    """
    Find Pareto-optimal configurations using established dominance tools.

    Args:
        grid_results: List of dicts with per-config seed_results
        tuning_seeds: List of seed values

    Returns:
        List of Pareto-optimal ConfigEvaluation entries
    """

    entries = []

    for config_idx, config_result in enumerate(grid_results):
        # Get valid seeds
        valid_seeds = [r for r in config_result['seed_results'] if r['valid']]

        if len(valid_seeds) == 0:
            continue

        # Compute mean metrics
        mean_cosine = np.mean([r['cosine_similarity'] for r in valid_seeds])
        mean_l2 = np.mean([r['score_l2_error'] for r in valid_seeds])
        mean_rel_norm = np.mean([r['relative_norm_error'] for r in valid_seeds])

        fisher_errors = [r['fisher_scaled_errors'] for r in valid_seeds if r['fisher_scaled_errors']]
        if fisher_errors:
            mean_fisher = np.mean([np.mean(fs) for fs in fisher_errors])
            max_fisher = max(max(fs) for fs in fisher_errors)
        else:
            mean_fisher = 0.0
            max_fisher = 0.0

        mean_hmc = np.mean([r['induced_hmc_error'] for r in valid_seeds])

        # Apply hard constraints (vetoes)
        if mean_cosine < 0.9995:
            continue  # Direction quality veto
        if mean_rel_norm > 0.05:
            continue  # Magnitude quality veto
        if max_fisher > 1.0:
            continue  # Component quality veto

        # Compute objectives (all minimize)
        objectives = (
            mean_l2,  # L2 error
            1.0 - mean_cosine,  # Direction error (minimize means maximize cosine)
            mean_rel_norm,  # Magnitude error
            mean_fisher,  # Fisher-scaled balance
            mean_hmc,  # HMC parameter error
        )

        # Create evaluation entry
        entry = ConfigEvaluation(
            config_idx=config_idx,
            objectives=objectives,
            controls=config_result['controls'],
            mean_cosine=mean_cosine,
            mean_l2=mean_l2,
            mean_rel_norm=mean_rel_norm,
            mean_fisher=mean_fisher,
            mean_hmc=mean_hmc,
            valid_fraction=config_result['valid_fraction'],
        )
        entries.append(entry)

    # Find Pareto frontier using established function
    pareto_optimal = nondominated(entries, absolute_tolerance=1e-12)

    return pareto_optimal


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
            veto_reason = 'cosine_veto'
        elif rel_norm_error > 0.05:
            tuning_score = float('inf')
            veto_reason = 'rel_norm_veto'
        elif any(fs > 1.0 for fs in fisher_scaled):
            tuning_score = float('inf')
            veto_reason = 'fisher_veto'
        else:
            # Primary: L2 error, Secondary: Fisher-scaled and HMC error
            fisher_penalty = 0.1 * np.mean(fisher_scaled)
            hmc_penalty = 0.1 * induced_hmc_error
            tuning_score = score_l2 + fisher_penalty + hmc_penalty
            veto_reason = None

        return {
            'valid': True,
            'score_l2_error': float(score_l2),
            'cosine_similarity': float(cosine_sim),
            'relative_norm_error': float(rel_norm_error),
            'fisher_scaled_errors': fisher_scaled,
            'induced_hmc_error': float(induced_hmc_error),
            'tuning_score': float(tuning_score),
            'veto_reason': veto_reason,
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
            'veto_reason': 'exception',
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
            'mean_relative_norm_error': float(np.mean([r['relative_norm_error'] for r in valid_results])),
            'mean_fisher_scaled': float(np.mean([np.mean(r['fisher_scaled_errors']) for r in valid_results if r['fisher_scaled_errors']])),
            'mean_hmc_error': float(np.mean([r['induced_hmc_error'] for r in valid_results])),
            'mean_tuning_score': float(mean_tuning_score),
            'veto_counts': {
                'cosine': sum(1 for r in seed_results if r.get('veto_reason') == 'cosine_veto'),
                'rel_norm': sum(1 for r in seed_results if r.get('veto_reason') == 'rel_norm_veto'),
                'fisher': sum(1 for r in seed_results if r.get('veto_reason') == 'fisher_veto'),
                'exception': sum(1 for r in seed_results if r.get('veto_reason') == 'exception'),
            },
            'seed_results': seed_results,
        })

    # Pareto-optimal selection using established tools
    print(f"\n{'='*80}")
    print(f"PARETO ANALYSIS")
    print(f"{'='*80}\n")

    pareto_optimal = find_pareto_optimal(grid_results, tuning_seeds)

    if not pareto_optimal:
        print("⚠️  NO PARETO-OPTIMAL CONFIGURATIONS FOUND")
        print("All configurations either failed validity or violated hard constraints")
        print("\nHard constraints:")
        print("  - Cosine similarity >= 0.9995")
        print("  - Relative norm error <= 0.05")
        print("  - All Fisher-scaled errors <= 1.0")
        return None

    print(f"Pareto frontier: {len(pareto_optimal)} configurations\n")

    # Display Pareto frontier
    for i, entry in enumerate(pareto_optimal):
        print(f"Config {i+1}/{len(pareto_optimal)} (grid index {entry.config_idx}):")
        print(f"  L2 error: {entry.objectives[0]:.4f}")
        print(f"  Direction error (1-cosine): {entry.objectives[1]:.7f}")
        print(f"  Rel norm error: {entry.objectives[2]:.4f}")
        print(f"  Mean Fisher-scaled: {entry.objectives[3]:.4f}")
        print(f"  Induced HMC error: {entry.objectives[4]:.6f}")
        print()

    # Select best by lexicographic ordering (L2 primary)
    best_entry = min(pareto_optimal, key=lambda e: e.objectives)
    best = grid_results[best_entry.config_idx]

    print(f"{'='*80}")
    print(f"SELECTED (lexicographic: L2 primary, then direction, magnitude, fisher, hmc)")
    print(f"{'='*80}")
    print(f"Grid index: {best_entry.config_idx}")
    print(f"L2 error: {best_entry.mean_l2:.4f}")
    print(f"Cosine similarity: {best_entry.mean_cosine:.6f}")
    print(f"Relative norm error: {best_entry.mean_rel_norm:.4f}")
    print(f"Mean Fisher-scaled: {best_entry.mean_fisher:.4f}")
    print(f"Induced HMC error: {best_entry.mean_hmc:.6f}")
    print(f"Valid fraction: {best_entry.valid_fraction:.2%}")
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
        'tuning_method': 'pareto_optimal_grid_search',
        'tuning_objective': 'Find Pareto-optimal configs minimizing (L2, 1-cosine, rel_norm, fisher, hmc)',
        'selection_criterion': 'Lexicographic ordering from Pareto frontier (L2 primary)',
        'hard_constraints': {
            'cosine_similarity_min': 0.9995,
            'relative_norm_error_max': 0.05,
            'fisher_scaled_error_max': 1.0,
        },
        'pareto_tools': 'common_utils.tf_multiobjective.population_switching.nondominated',
        'best_controls': best['controls'],
        'best_metrics': {
            'mean_score_l2_error': best_entry.mean_l2,
            'mean_cosine_similarity': best_entry.mean_cosine,
            'mean_relative_norm_error': best_entry.mean_rel_norm,
            'mean_fisher_scaled': best_entry.mean_fisher,
            'mean_induced_hmc_error': best_entry.mean_hmc,
            'valid_fraction': best_entry.valid_fraction,
        },
        'pareto_frontier_size': len(pareto_optimal),
        'pareto_frontier': [
            {
                'config_idx': entry.config_idx,
                'controls': entry.controls,
                'objectives': {
                    'l2_error': entry.objectives[0],
                    'direction_error': entry.objectives[1],
                    'rel_norm_error': entry.objectives[2],
                    'fisher_scaled': entry.objectives[3],
                    'hmc_error': entry.objectives[4],
                },
            }
            for entry in pareto_optimal
        ],
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
