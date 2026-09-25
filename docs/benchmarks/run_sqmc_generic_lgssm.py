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

import numpy as np

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
import test_p44_lgssm_exact_baseline as P44_LGSSM

DTYPE = tf.float64
GPU_POLICY_RECORD: Dict = _GPU_POLICY

# Repo root for absolute paths
REPO_ROOT = Path(__file__).resolve().parents[2]


def _p44_theta(dim: int) -> tf.Tensor:
    """Return P44 LGSSM theta for dimension-generic testing.

    Uses same structure as P44 test: [rho_param, log_q_scale, log_r_scale, initial_mean_scale]
    """
    return tf.constant([0.25, np.log(0.18), np.log(0.12), 0.04], dtype=DTYPE)


def _p44_observations(dim: int, horizon: int, seed: int) -> tf.Tensor:
    """Generate P44 LGSSM observations for given dimension and horizon."""
    rng = np.random.default_rng(seed)
    theta = _p44_theta(dim)

    # Get P44 physical parameters
    theta_np = theta.numpy()
    scale = np.array([1.0, 0.85, 0.70] + [0.55] * max(0, dim - 3))[:dim]
    q_scale = np.array([0.90, 1.10, 1.30] + [1.0] * max(0, dim - 3))[:dim]
    r_scale = np.array([1.00, 1.20, 0.80] + [1.0] * max(0, dim - 3))[:dim]
    mean_scale = np.array([1.00, -0.50, 0.25] + [0.0] * max(0, dim - 3))[:dim]

    rho = 0.55 * np.tanh(theta_np[0]) * scale
    q_diag = np.exp(theta_np[1]) * q_scale
    r_diag = np.exp(theta_np[2]) * r_scale
    raw_initial_mean = theta_np[3] * mean_scale

    transition_matrix = np.diag(rho)

    # Generate trajectory
    x = raw_initial_mean + rng.normal(0, 1.0, dim)
    observations = []

    for t in range(horizon):
        if t > 0:
            x = transition_matrix @ x + rng.normal(0, 1.0, dim) * q_diag
        obs = x + rng.normal(0, 1.0, dim) * r_diag
        observations.append(obs)

    return tf.constant(np.array(observations), dtype=DTYPE)


def _p44_oracle_score(theta: tf.Tensor, observations: tf.Tensor, dim: int) -> Tuple[tf.Tensor, tf.Tensor]:
    """Compute exact Kalman oracle for P44 LGSSM.

    Uses P44's _lgssm_model to get LinearGaussianSSM, then Kalman oracle.
    """
    # Get P44 model
    lgssm_model = P44_LGSSM._lgssm_model(theta, dim)

    # Convert to structural form for Kalman filter
    structural = P44_LGSSM._structural_model(theta, dim)
    linear = affine_structural_to_linear_gaussian_tf(structural)

    # Compute exact Kalman log-likelihood and score via GradientTape
    theta_var = tf.Variable(theta, dtype=DTYPE)
    with tf.GradientTape() as tape:
        tape.watch(theta_var)
        # Recompute model with watched variable
        watched_structural = P44_LGSSM._structural_model(theta_var, dim)
        watched_linear = affine_structural_to_linear_gaussian_tf(watched_structural)
        value = tf_linear_gaussian_log_likelihood(
            observations,
            watched_linear,
            backend="tf_cholesky",
            jitter=tf.constant(0.0, dtype=DTYPE),
            return_filtered=False,
        ).log_likelihood

    score = tape.gradient(value, theta_var)
    return value, score


def _p44_nonlinear_model(theta: tf.Tensor, dim: int):
    """Build NonlinearScoreModel for P44 LGSSM for SQMC.

    This adapts the P44 LinearGaussianSSM to the NonlinearScoreModel interface
    that canonical_value_and_analytical_score expects.
    """
    from bayesfilter.highdim.ledh_canonical_models_tf import NonlinearScoreModel

    # Get P44 physical parameters
    theta_np = theta.numpy()
    scale = np.array([1.0, 0.85, 0.70] + [0.55] * max(0, dim - 3))[:dim]
    q_scale = np.array([0.90, 1.10, 1.30] + [1.0] * max(0, dim - 3))[:dim]
    r_scale = np.array([1.00, 1.20, 0.80] + [1.0] * max(0, dim - 3))[:dim]

    rho = 0.55 * np.tanh(theta_np[0]) * scale
    q_diag = np.exp(theta_np[1]) * q_scale
    r_diag = np.exp(theta_np[2]) * r_scale

    transition_matrix_diag = tf.constant(rho, dtype=DTYPE)
    process_covariance_diag = tf.constant(q_diag**2, dtype=DTYPE)
    observation_covariance_diag = tf.constant(r_diag**2, dtype=DTYPE)

    log_two_pi = tf.constant(np.log(2.0 * np.pi), DTYPE)

    # Direction for score computation
    _direction = [tf.zeros([4], DTYPE)]

    def set_score_direction(direction: tf.Tensor) -> None:
        _direction[0] = tf.convert_to_tensor(direction, DTYPE)

    def transition_mean_fn(theta, points):
        return points * transition_matrix_diag[None, :]

    def transition_mean_tangent_fn(theta, points, d_points):
        # For simplicity, assume rho doesn't vary with theta in this version
        return d_points * transition_matrix_diag[None, :]

    def _scaled_gaussian(points, means, scale_diag):
        residual = points - means
        return -0.5 * (
            tf.reduce_sum(tf.square(residual) / tf.square(scale_diag)[None, :], axis=1)
            + float(dim) * log_two_pi
            + 2.0 * tf.reduce_sum(tf.math.log(scale_diag))
        )

    def transition_log_density_fn(theta, points, ancestors_mean):
        return _scaled_gaussian(points, ancestors_mean, tf.sqrt(process_covariance_diag))

    def transition_log_density_tangent_fn(theta, points, ancestors_mean, d_points, d_means):
        # Simplified - full version would include d_scale terms
        return tf.zeros([tf.shape(points)[0]], DTYPE)

    def observation_log_density_fn(theta, points, observation):
        observed = points  # Identity observation matrix
        target = tf.broadcast_to(observation[None, :], tf.shape(observed))
        return _scaled_gaussian(target, observed, tf.sqrt(observation_covariance_diag))

    def observation_log_density_tangent_fn(theta, points, observation, d_points):
        # Simplified
        return tf.zeros([tf.shape(points)[0]], DTYPE)

    def process_covariance_tangent_fn(theta):
        return tf.zeros([dim, dim], dtype=DTYPE)

    def observation_covariance_tangent_fn(theta):
        return tf.zeros([dim, dim], dtype=DTYPE)

    # Identity observation function
    def observation_fn(points):
        return points

    def observation_jacobian_fn(points):
        return tf.broadcast_to(
            tf.eye(dim, dtype=DTYPE), [tf.shape(points)[0], dim, dim]
        )

    def observation_tangent_fn(points, d_points):
        return d_points

    model = NonlinearScoreModel(
        transition_mean_fn=transition_mean_fn,
        transition_mean_tangent_fn=transition_mean_tangent_fn,
        observation_fn=observation_fn,
        observation_jacobian_fn=observation_jacobian_fn,
        observation_tangent_fn=observation_tangent_fn,
        process_covariance=tf.linalg.diag(process_covariance_diag),
        observation_covariance=tf.linalg.diag(observation_covariance_diag),
        transition_log_density_fn=transition_log_density_fn,
        transition_log_density_tangent_fn=transition_log_density_tangent_fn,
        observation_log_density_fn=observation_log_density_fn,
        observation_log_density_tangent_fn=observation_log_density_tangent_fn,
        process_covariance_tangent_fn=process_covariance_tangent_fn,
        observation_covariance_tangent_fn=observation_covariance_tangent_fn,
    )

    return model, set_score_direction


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


def _evaluate_generic_sqmc(
    route: str,
    controls: Dict,
    observations: tf.Tensor,
    theta: tf.Tensor,
    oracle_score: tf.Tensor,
    seed: int,
    horizon: int,
    particle_count: int,
    state_dim: int,
) -> Dict:
    """Evaluate SQMC on dimension-generic P44 LGSSM."""

    # Map route to ancestry_policy
    ancestry_map = {
        'iid_dual_cap': 'existing_one_to_one',
        'previous_inverse_cdf': 'hilbert_inverse_cdf',
        'repaired_permutation': 'hilbert_permutation_one_to_one',
        'repaired_permutation_ablation': 'hilbert_permutation_one_to_one',
    }
    ancestry_policy = ancestry_map.get(route, 'existing_one_to_one')
    is_ablation = (route == 'repaired_permutation_ablation')

    # Generate initial states and process noise
    if route == 'iid_dual_cap':
        initial_states = tf.random.stateless_normal(
            [particle_count, state_dim], [seed, 101], dtype=DTYPE
        )
        process_noise = tf.stack([
            tf.random.stateless_normal(
                [particle_count, state_dim], [seed, 1001 + t], dtype=DTYPE
            )
            for t in range(horizon)
        ])
        ancestor_uniforms = tf.zeros([horizon, particle_count], DTYPE)
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
        for t in range(horizon):
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

    # Build model
    model, set_score_direction = _p44_nonlinear_model(theta, state_dim)

    initial_covariances = tf.eye(state_dim, batch_shape=[particle_count], dtype=DTYPE)
    design = _reset_design(particle_count, state_dim)

    # Evaluate score for each direction
    try:
        values = []
        scores = []

        for direction_idx in range(len(theta)):
            set_score_direction(tf.one_hot(direction_idx, len(theta), dtype=DTYPE))

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
            scores.append(score[0])

        full_value = tf.reduce_mean(values)
        full_score = tf.stack(scores)

        # Compute metrics
        score_l2 = float(tf.norm(full_score - oracle_score).numpy())
        cosine = float(
            (tf.reduce_sum(full_score * oracle_score) /
             (tf.norm(full_score) * tf.norm(oracle_score))).numpy()
        )

        oracle_norm = float(tf.norm(oracle_score).numpy())
        sqmc_norm = float(tf.norm(full_score).numpy())
        rel_norm_error = abs(sqmc_norm - oracle_norm) / oracle_norm if oracle_norm > 0 else 0.0

        return {
            'valid': True,
            'value': float(full_value.numpy()),
            'score': full_score.numpy().tolist(),
            'oracle_score': oracle_score.numpy().tolist(),
            'score_l2_error': score_l2,
            'cosine_similarity': cosine,
            'relative_norm_error': rel_norm_error,
            'sqmc_norm': sqmc_norm,
            'oracle_norm': oracle_norm,
        }

    except Exception as e:
        return {
            'valid': False,
            'error': str(e),
        }


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

    results = []

    for dim in args.dimensions:
        print(f"\nTesting dimension {dim}...")

        # Check particle count constraint
        if args.particles % (2 * dim) != 0:
            print(f"  ERROR: N={args.particles} not divisible by 2*D={2*dim}")
            continue

        for seed in args.seeds:
            theta = _p44_theta(dim)
            observations = _p44_observations(dim, args.horizon, seed)
            oracle_value, oracle_score = _p44_oracle_score(theta, observations, dim)

            # Default controls for smoke
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
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        'schema': 'bayesfilter.sqmc_generic_smoke.v1',
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

    return 0 if all(r['valid'] for r in results) else 1


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
