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

# GPU memory policy: growth must be requested before TensorFlow initializes.
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import tensorflow as tf

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Add ~/python for multi-objective tools
sys.path.insert(0, '/home/chakwong/python/src')

from bayesfilter.runtime.gpu_memory_policy import (
    configure_tensorflow_gpu_memory_growth,
)

# Memory growth must be enabled and verified BEFORE anything initializes the GPU
# runtime.  Importing the bayesfilter.highdim model modules below initializes it,
# so the policy is applied here at module scope rather than inside main().  This
# fails closed per the repository TensorFlow GPU Memory Rule.
_GPU_POLICY = dict(configure_tensorflow_gpu_memory_growth(tf, require_gpu=True))

from bayesfilter.highdim.ledh_canonical_models_tf import diagonal_lgssm_canonical_model
from bayesfilter.highdim.ledh_kalman_oracle_tf import kalman_oracle_value_and_score

# The canonical LGSSM model adapter is float64-internal
# (`ledh_canonical_models_tf.DTYPE`), and the UNTUNED baseline diagnostic ran
# float64.  Tuning therefore runs float64 so the TUNED-vs-UNTUNED comparison
# varies the controls only.  A float32/TF32 production-dtype arm is a separate
# tuning scope under the LEDH per-scope rule.
DTYPE = tf.float64

# Verified GPU memory policy, recorded in every tuning artifact.
GPU_POLICY_RECORD: Dict = _GPU_POLICY

# Hard constraints on the seed-AGGREGATED score quality, taken from the decision
# framework in docs/plans/sqmc-oracle-principled-score-metrics-2026-09-11.md
# ("Direction test: cosine > 0.999 (required)"; "Magnitude test: relative norm
# error < 5%"; "Fisher-scaled test: Err/sqrt|Oracle| < 0.5 acceptable, > 1.0 is
# error-accumulation risk").  These are correctness floors, not tuning targets:
# they reject a candidate whose gradients are unusable for HMC.  They are NOT
# derived from the UNTUNED baseline's observed values -- see the no-fire
# calibration in docs/benchmarks/calibrate_sqmc_tuning_vetoes.py.
COSINE_VETO = 0.999
REL_NORM_VETO = 0.05
FISHER_VETO = 1.0

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
    """Return the frozen canonical LGSSM observation sequence.

    This MUST be the same frozen target the UNTUNED baseline diagnostic used
    (`run_sqmc_oracle_characterization.py`), otherwise TUNED and UNTUNED cells
    would be measured on different data and the comparison would be invalid.
    """
    from bayesfilter.highdim.ledh_canonical_neutra_targets_tf import (
        _lgssm_frozen_observations,
    )

    return tf.cast(_lgssm_frozen_observations()[:horizon], DTYPE)


def _oracle_score(observations: tf.Tensor, theta: tf.Tensor) -> tf.Tensor:
    """Exact Kalman score for the canonical five-parameter diagonal LGSSM.

    Mirrors `_oracle` in `run_sqmc_oracle_characterization.py` so the tuning
    comparator is the same exact reference the UNTUNED baseline used.
    """

    observation_matrix = tf.constant(
        [[1.0, 0.25, -0.15], [0.2, 1.1, 0.3], [-0.1, 0.35, 0.9]], DTYPE
    )

    def parameters(value: tf.Tensor) -> Dict[str, tf.Tensor]:
        return {
            'transition_matrix': tf.linalg.diag(value[:3]),
            'process_covariance': tf.square(value[3]) * tf.eye(3, dtype=DTYPE),
            'observation_matrix': observation_matrix,
            'observation_covariance': tf.square(value[4]) * tf.eye(3, dtype=DTYPE),
            'initial_mean': tf.zeros([3], DTYPE),
            'initial_covariance': tf.eye(3, dtype=DTYPE),
        }

    oracle = kalman_oracle_value_and_score(
        observations, theta, parameters, dtype=DTYPE
    )
    return oracle['score']


def _reset_design(particle_count: int, dimension: int) -> tf.Tensor:
    """Contract-E reset design matrix (mirrors the baseline runner's `_design`)."""
    if particle_count % (2 * dimension):
        raise ValueError("particle count must be divisible by 2 * state dimension")
    base = tf.concat(
        [tf.eye(dimension, dtype=DTYPE), -tf.eye(dimension, dtype=DTYPE)], axis=0
    )
    return tf.tile(base, [particle_count // (2 * dimension), 1])


def _tuning_grid() -> List[Dict]:
    """Define tuning grid for SQMC controls.

    3 Sinkhorn epsilon x 3 diagonal strengths x 2 pairwise strengths
    = 18 configurations.

    The Sinkhorn/balance step count is NOT swept.  It was originally a third grid
    dimension with settings (4,4), (8,8), (16,16), which made the grid 54
    configurations.  Direct measurement of the reset core
    (`docs/benchmarks/check_sinkhorn_step_sensitivity.py`) shows that dimension is
    scientifically inert at every grid epsilon: the Sinkhorn marginals are already
    converged at the smallest setting (row error ~1e-16), so extra iterations
    refine the transport only at the ~1e-11 level -- roughly nine orders of
    magnitude below the control effects on score L2 (~1e-2) and ten orders below
    per-seed L2 noise (~1.7e-1).  The abandoned 54-cell pilot log confirmed this
    end to end at full scale: rows differing only in step count printed identical
    L2 and cosine.  Sweeping it triples cost and adds no decision-relevant
    information.

    The retained setting is (8,8), the UNTUNED baseline's own value, which keeps
    the baseline controls inside the grid and so preserves the baseline-in-grid
    self-check used by analyze_sqmc_pilot_frontier.py.
    """
    grid = []

    epsilon_values = [4.0, 8.0, 16.0]
    diag_strengths = [0.1, 0.15, 0.2]
    pair_strengths = [0.02, 0.03]  # 0.02 is the baseline value

    for epsilon in epsilon_values:
        for diag_strength in diag_strengths:
            for pair_strength in pair_strengths:
                grid.append({
                    'reset_epsilon': epsilon,
                    'reset_sinkhorn_steps': 8,  # inert dimension, baseline value
                    'reset_balance_steps': 8,
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

        # Apply hard constraints to the SEED-AGGREGATED estimate.
        #
        # Thresholds come from the decision framework in
        # `docs/plans/sqmc-oracle-principled-score-metrics-2026-09-11.md`
        # ("When are scores good enough for HMC?"), NOT from the UNTUNED
        # baseline's observed values.  An earlier revision used cosine >= 0.9995
        # -- the baseline's two-seed observed MEAN -- as a hard veto; a no-fire
        # calibration check (`calibrate_sqmc_tuning_vetoes.py`) showed that
        # threshold fires on 3 of 4 seeds of the known-good baseline, i.e. a
        # descriptive statistic had been promoted to a correctness criterion.
        if mean_cosine < COSINE_VETO:
            continue  # Direction quality veto (required for HMC)
        if mean_rel_norm > REL_NORM_VETO:
            continue  # Magnitude quality veto
        if max_fisher > FISHER_VETO:
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
    state_dim: int = 3,
) -> Dict:
    """Evaluate one control configuration."""

    from bayesfilter.highdim.ledh_canonical_score_tf import canonical_value_and_analytical_score
    from bayesfilter.highdim.sqmc_tf import randomized_halton_gaussian, randomized_halton_joint

    # Map route to ancestry_policy
    ancestry_map = {
        'iid_dual_cap': 'existing_one_to_one',
        'previous_inverse_cdf': 'hilbert_inverse_cdf',
        'repaired_fixed_previous_controls': 'hilbert_permutation_one_to_one',
        'repaired_permutation': 'hilbert_permutation_one_to_one',
    }
    ancestry_policy = ancestry_map.get(route, 'existing_one_to_one')

    # Generate initial states and process noise
    dtype = DTYPE
    if route == 'iid_dual_cap':
        initial_states = tf.random.stateless_normal(
            [particle_count, state_dim], [seed, 101], dtype=dtype
        )
        process_noise = tf.stack([
            tf.random.stateless_normal(
                [particle_count, state_dim], [seed, 1001 + t], dtype=dtype
            )
            for t in range(horizon)
        ])
        ancestor_uniforms = tf.zeros([horizon, particle_count], dtype)
    else:
        initial_states = randomized_halton_gaussian(
            num_particles=particle_count,
            dimension=state_dim,
            seed=seed,
            salt=301,
            dtype=dtype,
        )
        process_rows = []
        ancestor_rows = []
        for t in range(horizon):
            raw, ancestors, innovations = randomized_halton_joint(
                num_particles=particle_count,
                state_dimension=state_dim,
                seed=seed,
                salt=3001 + t,
                dtype=dtype,
            )
            process_rows.append(tf.math.ndtri(innovations))
            ancestor_rows.append(ancestors)
        process_noise = tf.stack(process_rows)
        ancestor_uniforms = tf.stack(ancestor_rows)

    # Build model - use dimension-generic P44 LGSSM for non-3D cases
    if state_dim == 3 and len(theta) == 5:
        # Use frozen 3D canonical model for 3D T=20 tuning baseline
        model, set_score_direction = diagonal_lgssm_canonical_model(theta)
    else:
        # Use dimension-generic P44 LGSSM infrastructure
        from bayesfilter.highdim.ledh_canonical_models_tf import NonlinearScoreModel

        # Extract P44-style parameters from theta
        # For 10D: theta = [phi_1...phi_10, q_scale, r_scale] (12 params)
        # For generic: theta = [rho_param, log_q_scale, log_r_scale, initial_mean_scale] (4 params)
        if len(theta) > 4:
            # Multi-parameter theta: assume [phi_1...phi_D, q_scale, r_scale]
            phi_diag = theta[:state_dim]
            q_scale = theta[state_dim]
            r_scale = theta[state_dim + 1]
        else:
            # 4-parameter P44 style
            scale = tf.constant([1.0, 0.85, 0.70] + [0.55] * max(0, state_dim - 3), dtype)[:state_dim]
            q_scale_base = tf.constant([0.90, 1.10, 1.30] + [1.0] * max(0, state_dim - 3), dtype)[:state_dim]
            r_scale_base = tf.constant([1.00, 1.20, 0.80] + [1.0] * max(0, state_dim - 3), dtype)[:state_dim]

            phi_diag = 0.55 * tf.tanh(theta[0]) * scale
            q_diag = tf.exp(theta[1]) * q_scale_base
            r_diag = tf.exp(theta[2]) * r_scale_base
            q_scale = tf.sqrt(tf.reduce_mean(tf.square(q_diag)))
            r_scale = tf.sqrt(tf.reduce_mean(tf.square(r_diag)))

        log_two_pi = tf.constant(np.log(2.0 * np.pi), dtype)

        # Direction for score computation
        _direction = [tf.zeros([len(theta)], dtype)]

        def set_score_direction(direction):
            _direction[0] = tf.convert_to_tensor(direction, dtype)

        def transition_mean_fn(theta_arg, points):
            if len(theta) > 4:
                phi = theta_arg[:state_dim]
            else:
                scale_fn = tf.constant([1.0, 0.85, 0.70] + [0.55] * max(0, state_dim - 3), dtype)[:state_dim]
                phi = 0.55 * tf.tanh(theta_arg[0]) * scale_fn
            return points * phi[None, :]

        def transition_mean_tangent_fn(theta_arg, points, d_points):
            # Simplified tangent - assume phi doesn't vary with theta for now
            if len(theta) > 4:
                phi = theta_arg[:state_dim]
            else:
                scale_fn = tf.constant([1.0, 0.85, 0.70] + [0.55] * max(0, state_dim - 3), dtype)[:state_dim]
                phi = 0.55 * tf.tanh(theta_arg[0]) * scale_fn
            return d_points * phi[None, :]

        def _scaled_gaussian(points, means, scale_val):
            residual = points - means
            return -0.5 * (
                tf.reduce_sum(tf.square(residual), axis=1) / tf.square(scale_val)
                + float(state_dim) * (log_two_pi + 2.0 * tf.math.log(scale_val))
            )

        def transition_log_density_fn(theta_arg, points, ancestors_mean):
            if len(theta) > 4:
                q = theta_arg[state_dim]
            else:
                q = tf.exp(theta_arg[1])
            return _scaled_gaussian(points, ancestors_mean, q)

        def transition_log_density_tangent_fn(theta_arg, points, ancestors_mean, d_points, d_means):
            return tf.zeros([tf.shape(points)[0]], dtype)

        def observation_log_density_fn(theta_arg, points, observation):
            observed = points  # Identity observation
            target = tf.broadcast_to(observation[None, :], tf.shape(observed))
            if len(theta) > 4:
                r = theta_arg[state_dim + 1]
            else:
                r = tf.exp(theta_arg[2])
            return _scaled_gaussian(target, observed, r)

        def observation_log_density_tangent_fn(theta_arg, points, observation, d_points):
            return tf.zeros([tf.shape(points)[0]], dtype)

        def process_covariance_tangent_fn(theta_arg):
            return tf.zeros([state_dim, state_dim], dtype=dtype)

        def observation_covariance_tangent_fn(theta_arg):
            return tf.zeros([state_dim, state_dim], dtype=dtype)

        def observation_fn(points):
            return points

        def observation_jacobian_fn(points):
            return tf.broadcast_to(
                tf.eye(state_dim, dtype=dtype), [tf.shape(points)[0], state_dim, state_dim]
            )

        def observation_tangent_fn(points, d_points):
            return d_points

        model = NonlinearScoreModel(
            transition_mean_fn=transition_mean_fn,
            transition_mean_tangent_fn=transition_mean_tangent_fn,
            observation_fn=observation_fn,
            observation_jacobian_fn=observation_jacobian_fn,
            observation_tangent_fn=observation_tangent_fn,
            process_covariance=tf.square(q_scale) * tf.eye(state_dim, dtype=dtype),
            observation_covariance=tf.square(r_scale) * tf.eye(state_dim, dtype=dtype),
            transition_log_density_fn=transition_log_density_fn,
            transition_log_density_tangent_fn=transition_log_density_tangent_fn,
            observation_log_density_fn=observation_log_density_fn,
            observation_log_density_tangent_fn=observation_log_density_tangent_fn,
            process_covariance_tangent_fn=process_covariance_tangent_fn,
            observation_covariance_tangent_fn=observation_covariance_tangent_fn,
        )

    initial_covariances = tf.eye(state_dim, batch_shape=[particle_count], dtype=dtype)
    design = _reset_design(particle_count, state_dim)

    # Evaluate score for each direction
    try:
        values = []
        scores = []

        for direction_idx in range(len(theta)):
            set_score_direction(tf.one_hot(direction_idx, len(theta), dtype=dtype))

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
                reset_epsilon=controls['reset_epsilon'],
                reset_sinkhorn_steps=controls['reset_sinkhorn_steps'],
                reset_balance_steps=controls['reset_balance_steps'],
                reset_ridge=1e-5,
                correction_steps=controls['correction_steps'],
                correction_strength=controls['correction_strength'],
                correction_lm_damping=0.01,
                correction_lm_scale_floor=0.0001,
                correction_trust_radius=0.5,
                pairwise_steps=controls['pairwise_steps'],
                pairwise_strength=controls['pairwise_strength'],
                pairwise_rms_cap=2.0,
                coordinate_cap=0.98 if not is_ablation else 0.97,
                coordinate_cap_power=8,
                ancestry_policy=ancestry_policy,
                process_ancestor_uniforms=ancestor_uniforms,
                state_map_policy='adaptive_empirical',
                hilbert_bits=12,
            )

            values.append(value)
            scores.append(score[0])  # directional derivative for this direction

        # One directional derivative per parameter -> full score vector (5,)
        all_scores = tf.stack(scores)

        # Check finiteness
        if not tf.reduce_all(tf.math.is_finite(all_scores)):
            return {
                'valid': False,
                'score_l2_error': float('inf'),
                'cosine_similarity': 0.0,
                'relative_norm_error': float('inf'),
                'fisher_scaled_errors': [],
                'induced_hmc_error': float('inf'),
                'tuning_score': float('inf'),
                'veto_reason': 'non_finite',
                'error': 'Non-finite scores',
            }

        # Compute metrics vs oracle (convert to float64 for precision)
        sqmc_score = tf.cast(all_scores, tf.float64)

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

        # NOTE: per-seed cells carry NO threshold veto.  Per-seed cosine is a
        # noisy single draw (the known-good baseline spans 0.99917-0.99973
        # across four seeds), so vetoing individual draws would reject the
        # baseline itself.  Threshold constraints are applied to the
        # seed-aggregated estimate in `find_pareto_optimal`.  A per-seed cell is
        # invalid only if it is non-finite or raised.
        tuning_score = score_l2
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
            'value': float(values[0].numpy()),
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
    theta = tf.constant([0.9, 0.8, 0.7, 0.6, 0.8], dtype=DTYPE)
    observations = _frozen_observations(horizon)

    oracle_score = _oracle_score(observations, theta)

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
            # Report first error for debugging
            first_error = next((r.get('error', 'unknown') for r in seed_results if not r['valid']), 'unknown')
            print(f"ALL INVALID (error: {first_error[:50]}...)")
            grid_results.append({
                'controls': controls,
                'valid_fraction': 0.0,
                'mean_score_l2': float('inf'),
                'mean_cosine_similarity': 0.0,
                'mean_relative_norm_error': float('inf'),
                'mean_fisher_scaled': float('inf'),
                'mean_hmc_error': float('inf'),
                'mean_tuning_score': float('inf'),
                'veto_counts': {'cosine': 0, 'rel_norm': 0, 'fisher': 0, 'exception': len(seed_results)},
                'seed_results': seed_results,
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
        print(f"  - Cosine similarity >= {COSINE_VETO}")
        print(f"  - Relative norm error <= {REL_NORM_VETO}")
        print(f"  - All Fisher-scaled errors <= {FISHER_VETO}")
        print("(applied to the seed-aggregated mean)")
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
        'backend': 'float64_gpu',
        'backend_note': (
            'float64 GPU, matching the UNTUNED baseline diagnostic dtype so the '
            'TUNED-vs-UNTUNED comparison varies controls only. The repository '
            'float32/TF32 production target is a separate tuning scope.'
        ),
        'observations_source': 'ledh_canonical_neutra_targets_tf._lgssm_frozen_observations',
        'chunk_policy': 'dpf_transport_exact_divisor_cap3000_v1',
        'reset_family': 'contract_e_genut_dual_cap',
        'tuning_seeds': tuning_seeds,
        'tuning_date': datetime.now().isoformat(),
        'git_commit': os.popen('git rev-parse HEAD').read().strip(),
        'gpu_memory_policy': GPU_POLICY_RECORD,
        'tuning_method': 'pareto_optimal_grid_search',
        'tuning_objective': 'Find Pareto-optimal configs minimizing (L2, 1-cosine, rel_norm, fisher, hmc)',
        'selection_criterion': 'Lexicographic ordering from Pareto frontier (L2 primary)',
        'hard_constraints': {
            'cosine_similarity_min': COSINE_VETO,
            'relative_norm_error_max': REL_NORM_VETO,
            'fisher_scaled_error_max': FISHER_VETO,
            'applied_to': 'seed_aggregated_mean',
            'threshold_source': (
                'docs/plans/sqmc-oracle-principled-score-metrics-2026-09-11.md '
                'decision framework; no-fire calibrated against the known-good '
                'warm-start baseline via docs/benchmarks/calibrate_sqmc_tuning_vetoes.py'
            ),
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

    # `--mode` selects the SEED BUDGET only.  It must never override an explicit
    # `--routes`: an earlier revision hardcoded routes_to_run=['iid_dual_cap'] in
    # pilot mode, so `--mode pilot --routes previous_inverse_cdf` silently retuned
    # iid_dual_cap and three "per-route" tuning runs produced bit-identical
    # artifacts labelled with different route names.
    if args.mode == 'pilot':
        print("\n" + "="*80)
        print("PILOT SEED BUDGET: 4 seeds")
        print("="*80)
        # 4 seeds, matching the baseline measured by
        # calibrate_sqmc_tuning_vetoes.py so the pilot's Pareto frontier is
        # directly comparable to the warm-start baseline on the same seeds.
        tuning_seeds = [50001, 50002, 50003, 50004]
        default_routes = ['iid_dual_cap']
    else:
        tuning_seeds = list(range(50001, 50017))  # 16 seeds
        default_routes = [
            'iid_dual_cap',
            'previous_inverse_cdf',
            'repaired_permutation',
            'repaired_permutation_ablation',
        ]

    routes_to_run = args.routes if args.routes else default_routes
    print(f"Routes to tune: {routes_to_run}")
    print(f"Tuning seeds: {tuning_seeds}")

    # GPU memory policy was applied and verified at module import, before the
    # model modules initialized the GPU runtime (fail-closed).
    print(
        "✓ GPU memory growth verified: "
        f"{[row['device'] for row in GPU_POLICY_RECORD.get('physical_devices', ())]}"
    )
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(f"✓ Logical GPUs: {[d.name for d in logical_gpus]}")

    # Run tuning for each route
    for route in routes_to_run:
        is_ablation = (route == 'repaired_permutation_ablation')

        # Anchor artifact paths to the repository root, not the caller's cwd.
        # Running from docs/benchmarks previously produced
        # docs/benchmarks/docs/tuning/... which is not the declared location.
        repo_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', '..')
        )
        tuning_root = os.path.join(repo_root, 'docs', 'tuning')

        if is_ablation:
            actual_route = 'repaired_permutation'
            output_dir = os.path.join(
                tuning_root,
                'sqmc-lgssm-t20-n1008-repaired_permutation_ablation-20260912',
            )
        else:
            actual_route = route
            output_dir = os.path.join(
                tuning_root, f'sqmc-lgssm-t20-n1008-{route}-20260912'
            )

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
