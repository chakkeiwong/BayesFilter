#!/usr/bin/env python3
"""SQMC Oracle Characterization: Minimal 8-cell pilot test.

Tests 4 SQMC routes against LGSSM Kalman oracle:
- LGSSM T=20, N=1008
- 4 routes × 2 seeds = 8 cells
- Warm-start controls + adaptive state map
- ~10-15 minutes GPU time

Purpose: Verify if route differences emerge with oracle before committing to full campaign.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tensorflow as tf
import numpy as np

from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth

MEMORY_POLICY = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)

from bayesfilter.highdim.ledh_kalman_oracle_tf import kalman_oracle_value_and_score
from bayesfilter.highdim.transport_chunk_policy import select_transport_chunks
from bayesfilter.highdim.ledh_pfpf_genut_initial_rqmc_tf import (
    finite_value_standard_score_initial_rqmc,
)
from bayesfilter.highdim.ledh_pfpf_genut_model_callbacks_tf import LEDHGenUTModelCallbacks

# Characterization scope
MODEL_ID = "lgssm_T20_characterization"
HORIZON = 20
STATE_DIM = 10
OBS_DIM = 10
PARTICLE_COUNT = 1008
SEEDS = (97701, 97702)  # Just 2 seeds for characterization
ROUTES = (
    "repaired_permutation",
    "iid_dual_cap",
    "previous_inverse_cdf",
    "repaired_fixed_previous_controls",
)

# Warm-start controls from Austria SIR + four-model evidence
# See: docs/plans/sqmc-parameters-tuning-analysis-2026-09-09.md
TRUST_CONTROLS = {
    "lm_damping": 1.0e-2,  # Conservative (Austria SIR current)
    "lm_scale_floor": 1.0e-4,
    "radius": 0.5,
}

ROUTE_CONTROLS = {
    "iid_dual_cap": {
        "epsilon": 8.0,
        "sinkhorn_steps": 8,
        "balance_steps": 8,
        "ridge": 1.0e-5,
        "hilbert_bits": 12,
        "diagonal_steps": 4,
        "diagonal_strength": 0.2,
        "pairwise_steps": 4,
        "pairwise_strength": 0.02,
        "radial_cap": 2.0,
        "coordinate_cap": 0.98,
        "coordinate_cap_power": 8,
    },
    "previous_inverse_cdf": {
        "epsilon": 8.0,
        "sinkhorn_steps": 8,
        "balance_steps": 8,
        "ridge": 1.0e-5,
        "hilbert_bits": 12,
        "diagonal_steps": 4,
        "diagonal_strength": 0.2,
        "pairwise_steps": 4,
        "pairwise_strength": 0.02,
        "radial_cap": 2.0,
        "coordinate_cap": 0.98,
        "coordinate_cap_power": 8,
    },
    "repaired_fixed_previous_controls": {
        "epsilon": 8.0,
        "sinkhorn_steps": 8,
        "balance_steps": 8,
        "ridge": 1.0e-5,
        "hilbert_bits": 12,
        "diagonal_steps": 4,
        "diagonal_strength": 0.2,
        "pairwise_steps": 4,
        "pairwise_strength": 0.02,
        "radial_cap": 2.0,
        "coordinate_cap": 0.98,
        "coordinate_cap_power": 8,
    },
    "repaired_permutation": {
        "epsilon": 8.0,
        "sinkhorn_steps": 8,
        "balance_steps": 8,
        "ridge": 1.0e-5,
        "hilbert_bits": 12,
        "diagonal_steps": 3,  # Slightly different from others
        "diagonal_strength": 0.15,
        "pairwise_steps": 3,
        "pairwise_strength": 0.01,
        "radial_cap": 1.5,
        "coordinate_cap": 0.97,
        "coordinate_cap_power": 6,
    },
}


def create_lgssm_model() -> tuple[LEDHGenUTModelCallbacks, tf.Tensor, dict]:
    """Create LGSSM T=20 model with 10D state/obs.

    Returns:
        callbacks: LEDH model callbacks
        observations: [T, obs_dim] synthetic observations
        true_params: Dict with true LGSSM matrices for oracle
    """
    # Simple diagonal LGSSM: x[t] = F @ x[t-1] + w[t], y[t] = C @ x[t] + v[t]
    # where F = diag(0.9), Q = diag(0.1), C = I, R = diag(0.2)

    F_diag = tf.constant([0.9] * STATE_DIM, dtype=tf.float32)
    Q_diag = tf.constant([0.1] * STATE_DIM, dtype=tf.float32)
    C = tf.eye(OBS_DIM, STATE_DIM, dtype=tf.float32)
    R_diag = tf.constant([0.2] * OBS_DIM, dtype=tf.float32)
    mu0 = tf.zeros(STATE_DIM, dtype=tf.float32)
    P0_diag = tf.constant([1.0] * STATE_DIM, dtype=tf.float32)

    # Generate synthetic observations (deterministic for characterization)
    np.random.seed(42)
    states = [np.zeros(STATE_DIM)]
    obs = []
    for t in range(HORIZON):
        if t > 0:
            states.append(0.9 * states[-1] + np.random.normal(0, np.sqrt(0.1), STATE_DIM))
        obs.append(states[-1] + np.random.normal(0, np.sqrt(0.2), OBS_DIM))

    observations = tf.constant(np.array(obs), dtype=tf.float32)

    # For oracle gradient tracking, we need a parameterization
    # Use simple: theta = [F_diag (10), log(Q_scale), log(R_scale)] = 12 params
    # But for characterization, we'll use fixed values and track just a scale parameter

    # Create simplified model class for 10D LGSSM
    class DiagonalLGSSM10D:
        def state_dim(self) -> int:
            return STATE_DIM

        def observation_dim(self) -> int:
            return OBS_DIM

        def parameter_dim(self) -> int:
            return 12  # 10 F_diag + 1 log(Q_scale) + 1 log(R_scale)

        def initial_log_density(self, theta: tf.Tensor, x0: tf.Tensor) -> tf.Tensor:
            # P0 = Q / (1 - F^2) for stationary initialization
            # Cast everything to float32 to match SQMC dtype
            theta32 = tf.cast(theta, tf.float32)
            x0_32 = tf.cast(x0, tf.float32)
            F_vals = theta32[:10]
            log_Q_scale = theta32[10]
            variance = tf.square(tf.exp(log_Q_scale)) / (1.0 - tf.square(F_vals))
            log_2pi = tf.constant(np.log(2.0 * np.pi), tf.float32)
            return -0.5 * tf.reduce_sum(tf.square(x0_32) / variance + tf.math.log(variance) + log_2pi)

        def transition_log_density(self, theta: tf.Tensor, x_prev: tf.Tensor, x_next: tf.Tensor, t: int) -> tf.Tensor:
            theta32 = tf.cast(theta, tf.float32)
            x_prev32 = tf.cast(x_prev, tf.float32)
            x_next32 = tf.cast(x_next, tf.float32)
            F_vals = theta32[:10]
            log_Q_scale = theta32[10]
            predicted = x_prev32 * F_vals
            residual = x_next32 - predicted
            variance = tf.square(tf.exp(log_Q_scale))
            log_2pi = tf.constant(np.log(2.0 * np.pi), tf.float32)
            return -0.5 * tf.reduce_sum(tf.square(residual) / variance + tf.math.log(variance) + log_2pi)

        def observation_log_density(self, theta: tf.Tensor, x_t: tf.Tensor, y_t: tf.Tensor, t: int) -> tf.Tensor:
            theta32 = tf.cast(theta, tf.float32)
            x_t32 = tf.cast(x_t, tf.float32)
            y_t32 = tf.cast(y_t, tf.float32)
            log_R_scale = theta32[11]
            residual = y_t32 - x_t32  # C = I
            variance = tf.square(tf.exp(log_R_scale))
            log_2pi = tf.constant(np.log(2.0 * np.pi), tf.float32)
            return -0.5 * tf.reduce_sum(tf.square(residual) / variance + tf.math.log(variance) + log_2pi)

        def initial_log_density_parameter_score(self, theta: tf.Tensor, x0: tf.Tensor) -> tf.Tensor:
            # Return per-particle score: [N, param_dim]
            # x0 is [N, state_dim], need to return [N, 12]
            theta32 = tf.cast(theta, tf.float32)
            x0_32 = tf.cast(x0, tf.float32)
            n_particles = tf.shape(x0_32)[0]
            return tf.zeros([n_particles, 12], dtype=theta32.dtype)

        def transition_log_density_parameter_score(self, theta: tf.Tensor, x_prev: tf.Tensor, x_next: tf.Tensor, t: int) -> tf.Tensor:
            # Return score for single transition: [param_dim]
            # This is called per time step, not batched over particles for score
            theta32 = tf.cast(theta, tf.float32)
            return tf.zeros([12], dtype=theta32.dtype)

        def observation_log_density_parameter_score(self, theta: tf.Tensor, x_t: tf.Tensor, y_t: tf.Tensor, t: int) -> tf.Tensor:
            # Return per-particle score: [N, param_dim]
            # x_t is [N, state_dim], need to return [N, 12]
            theta32 = tf.cast(theta, tf.float32)
            x_t32 = tf.cast(x_t, tf.float32)
            n_particles = tf.shape(x_t32)[0]
            return tf.zeros([n_particles, 12], dtype=theta32.dtype)

    model = DiagonalLGSSM10D()

    # Create theta: [F_diag, log(Q_scale), log(R_scale)]
    theta = tf.concat([
        F_diag,  # [10]
        [tf.math.log(tf.constant(np.sqrt(0.1), dtype=tf.float32))],  # log(Q_scale)
        [tf.math.log(tf.constant(np.sqrt(0.2), dtype=tf.float32))],  # log(R_scale)
    ], axis=0)

    # Create callbacks with a minimal push adapter
    # For oracle comparison, we don't need the full push adapter machinery
    # Create a simple identity adapter
    class IdentityPushAdapter:
        """Minimal push adapter for oracle comparison."""
        def __init__(self, state_dimension: int, parameter_count: int):
            self.state_dimension = state_dimension
            self.parameter_count = parameter_count

        def push(self, theta, particles):
            # Identity push - no transformation
            return particles

        def initial_value(self, theta, noise):
            # Return initial state particles (will be transformed by GenUT flow)
            # noise is [N, state_dim], we just return it (identity push)
            return noise

    push_adapter = IdentityPushAdapter(STATE_DIM, 12)  # 10 F_diag + 1 log_Q + 1 log_R

    # Create observation callbacks for LGSSM (C = I, identity observation)
    def _identity_residual(predicted, observed):
        return observed[None, None, :] - predicted

    def _linear_observation_callbacks(obs_matrix):
        """Create observation callbacks for linear observation model."""
        matrix = tf.convert_to_tensor(obs_matrix, tf.float32)

        def observation_fn(points):
            # points: [B, N, state_dim] -> [B, N, obs_dim]
            return tf.einsum("bnd,od->bno", points, matrix)

        def jacobian_fn(points):
            # Return [B, N, obs_dim, state_dim] Jacobian
            return tf.broadcast_to(
                matrix[None, None, :, :],
                [tf.shape(points)[0], tf.shape(points)[1], *matrix.shape],
            )

        return observation_fn, jacobian_fn, _identity_residual

    callbacks = LEDHGenUTModelCallbacks(
        model_id=MODEL_ID,
        model=model,
        push_adapter=push_adapter,
        transition_before_first_observation=False,
        target_time_offset=0,
        initial_covariance=lambda th: tf.linalg.diag(
            tf.square(tf.exp(th[10])) / (1.0 - tf.square(th[:10]))
        ),
        transition_mean=lambda th, points, _time: points * th[:10],
        transition_covariance=lambda th: tf.square(tf.exp(th[10])) * tf.eye(STATE_DIM, dtype=th.dtype),
        transition_matrix=lambda th: tf.linalg.diag(th[:10]),
        observation_covariance=lambda th: tf.square(tf.exp(th[11])) * tf.eye(OBS_DIM, dtype=th.dtype),
        proposal_observation=lambda _th, observation: observation,
        observation_callbacks=lambda _th, _time: _linear_observation_callbacks(C),
    )

    # True parameters for oracle
    true_params = {
        'transition_matrix': tf.linalg.diag(F_diag),
        'process_covariance': tf.linalg.diag(Q_diag),
        'observation_matrix': C,
        'observation_covariance': tf.linalg.diag(R_diag),
        'initial_mean': mu0,
        'initial_covariance': tf.linalg.diag(P0_diag),
    }

    return callbacks, observations, theta, true_params


def run_sqmc_cell(
    callbacks: LEDHGenUTModelCallbacks,
    observations: tf.Tensor,
    theta: tf.Tensor,
    route: str,
    route_controls: dict,
    trust_controls: dict,
    particle_count: int,
    seed: int,
) -> dict:
    """Run SQMC for one cell configuration.

    Args:
        callbacks: LEDH model callbacks
        observations: [T, obs_dim] observations
        theta: [param_dim] parameter vector
        route: Transport route name
        route_controls: Route-specific hyperparameters
        trust_controls: Trust-region hyperparameters
        particle_count: Number of particles
        seed: RNG seed

    Returns:
        Dict with 'value' (log-likelihood) and 'score' (gradient)
    """
    horizon = int(observations.shape[0])
    state_dim = callbacks.model.state_dim()

    # Expected process steps depends on transition_before_first_observation
    # Our model has transition_before_first_observation=False, so we need horizon-1 steps
    expected_process_steps = horizon - 1

    # Generate RQMC noise inputs
    # Initial noise: [N, state_dim]
    initial_noise = tf.random.stateless_normal(
        [particle_count, state_dim],
        seed=[seed, 0],
        dtype=tf.float32,
    )

    # Process noise: [expected_process_steps, N, state_dim]
    process_noise = tf.random.stateless_normal(
        [expected_process_steps, particle_count, state_dim],
        seed=[seed + 1, 0],
        dtype=tf.float32,
    )

    # Ancestor uniforms: [expected_process_steps, N] for resampling
    ancestor_uniforms = tf.random.stateless_uniform(
        [expected_process_steps, particle_count],
        seed=[seed + 2, 0],
        dtype=tf.float32,
    )

    # Design matrix for Contract-E (GenUT sigma points)
    # Must be [particle_count, state_dim] to match the particle dimension
    # Use simple random design (not optimal but sufficient for characterization)
    design = tf.random.stateless_normal(
        [particle_count, state_dim],
        seed=[seed + 3, 0],
        dtype=tf.float32,
    )

    # Determine ancestry policy
    ancestry_map = {
        'iid_dual_cap': 'existing_one_to_one',
        'previous_inverse_cdf': 'hilbert_inverse_cdf',
        'repaired_fixed_previous_controls': 'hilbert_permutation_one_to_one',
        'repaired_permutation': 'hilbert_permutation_one_to_one',
    }
    ancestry_policy = ancestry_map[route]

    # Transport chunks (streaming for K ≤ 3000)
    chunks = select_transport_chunks(particle_count)

    # Call production SQMC
    value, score, diagnostics = finite_value_standard_score_initial_rqmc(
        callbacks=callbacks,
        theta=theta,
        observations=observations,
        initial_noise=initial_noise,
        process_noise=process_noise,
        design=design,
        ancestry_policy=ancestry_policy,
        process_ancestor_uniforms=ancestor_uniforms,
        state_map_policy='adaptive_empirical',
        hilbert_bits=route_controls['hilbert_bits'],
        reset_policy='contract_e',
        dual_cap_enabled=True,
        dual_cap_diagonal_steps=route_controls['diagonal_steps'],
        dual_cap_diagonal_strength=route_controls['diagonal_strength'],
        dual_cap_pairwise_steps=route_controls['pairwise_steps'],
        dual_cap_pairwise_strength=route_controls['pairwise_strength'],
        dual_cap_pairwise_particle_rms_cap=route_controls['radial_cap'],
        dual_cap_coordinate_cap=route_controls['coordinate_cap'],
        dual_cap_coordinate_cap_power=route_controls['coordinate_cap_power'],
        trust_region_enabled=True,
        trust_region_lm_damping=trust_controls['lm_damping'],
        trust_region_lm_scale_floor=trust_controls['lm_scale_floor'],
        trust_region_radius=trust_controls['radius'],
        transport_plan_mode='streaming',
        transport_row_chunk_size=chunks.row_chunk_size,
        transport_col_chunk_size=chunks.col_chunk_size,
        epsilon=route_controls['epsilon'],
        sinkhorn_steps=route_controls['sinkhorn_steps'],
        balance_steps=route_controls['balance_steps'],
        ridge=route_controls['ridge'],
        functional_time_loop=True,
    )

    return {
        'value': float(value.numpy()),
        'score': score.numpy().tolist(),
        'program_valid': bool(diagnostics['program_valid'].numpy()),
    }


def compute_oracle(observations: tf.Tensor, theta: tf.Tensor, params: dict) -> dict:
    """Compute Kalman oracle value and score.

    Args:
        observations: [T, obs_dim] observations
        theta: [param_dim] parameter vector
        params: Dict with true LGSSM matrices

    Returns:
        Dict with 'value' and 'score'
    """
    # For LGSSM oracle, we need theta -> (F, Q, C, R, mu0, P0) mapping
    def theta_to_lgssm_params(th):
        F_diag = th[:10]
        log_Q_scale = th[10]
        log_R_scale = th[11]
        Q_scale_sq = tf.square(tf.exp(log_Q_scale))
        R_scale_sq = tf.square(tf.exp(log_R_scale))

        return {
            'transition_matrix': tf.linalg.diag(F_diag),
            'process_covariance': Q_scale_sq * tf.eye(STATE_DIM, dtype=th.dtype),
            'observation_matrix': tf.eye(OBS_DIM, STATE_DIM, dtype=th.dtype),
            'observation_covariance': R_scale_sq * tf.eye(OBS_DIM, dtype=th.dtype),
            'initial_mean': tf.zeros(STATE_DIM, dtype=th.dtype),
            'initial_covariance': tf.linalg.diag(Q_scale_sq / (1.0 - tf.square(F_diag))),
        }

    result = kalman_oracle_value_and_score(
        observations, theta, theta_to_lgssm_params, dtype=tf.float32
    )

    return {
        'value': float(result['value'].numpy()),
        'score': result['score'].numpy().tolist(),
    }


def main():
    print("=" * 80)
    print("SQMC Oracle Characterization - LGSSM T=20, N=1008")
    print("=" * 80)
    print()
    print("Configuration:")
    print(f"  Model: {MODEL_ID}")
    print(f"  Horizon: {HORIZON}")
    print(f"  State/Obs dim: {STATE_DIM}")
    print(f"  Particle count: {PARTICLE_COUNT}")
    print(f"  Routes: {len(ROUTES)}")
    print(f"  Seeds: {len(SEEDS)}")
    print(f"  Total cells: {len(ROUTES) * len(SEEDS)}")
    print()
    print("Parameters: Warm-start + adaptive state map")
    print("  (No per-model tuning, see sqmc-parameters-tuning-analysis-2026-09-09.md)")
    print()

    tf.config.set_soft_device_placement(False)
    tf.config.experimental.enable_tensor_float_32_execution(True)

    logical = tf.config.list_logical_devices("GPU")
    if len(logical) != 1:
        raise RuntimeError("Characterization requires exactly one visible logical GPU")

    print(f"GPU: {logical[0].name}")
    print()

    # Create model
    print("Creating LGSSM model...")
    callbacks, observations, theta, true_params = create_lgssm_model()
    print("✓ Model created")
    print(f"  State dim: {STATE_DIM}, Obs dim: {OBS_DIM}, Horizon: {HORIZON}")
    print(f"  Theta dim: {theta.shape[0]}")
    print()

    # Compute oracle
    print("Computing Kalman oracle...")
    oracle = compute_oracle(observations, theta, true_params)
    print(f"  Oracle value: {oracle['value']:.4f}")
    print(f"  Oracle score norm: {np.linalg.norm(oracle['score']):.4f}")
    print()

    # Run SQMC for each route × seed
    print(f"Running SQMC characterization: {len(ROUTES)} routes × {len(SEEDS)} seeds = {len(ROUTES) * len(SEEDS)} cells")
    print()

    results = []
    for route_idx, route in enumerate(ROUTES):
        print(f"Route {route_idx + 1}/{len(ROUTES)}: {route}")
        route_controls = ROUTE_CONTROLS[route]

        for seed_idx, seed in enumerate(SEEDS):
            cell_id = f"{route}_seed{seed}"
            print(f"  Cell {seed_idx + 1}/{len(SEEDS)}: seed={seed}", end=" ", flush=True)

            start_time = time.time()

            try:
                # Run SQMC (placeholder - need to implement)
                sqmc_result = run_sqmc_cell(
                    callbacks=callbacks,
                    observations=observations,
                    theta=theta,
                    route=route,
                    route_controls=route_controls,
                    trust_controls=TRUST_CONTROLS,
                    particle_count=PARTICLE_COUNT,
                    seed=seed,
                )

                elapsed = time.time() - start_time

                # Compute errors
                value_error = abs(sqmc_result['value'] - oracle['value'])
                score_error = np.linalg.norm(
                    np.array(sqmc_result['score']) - np.array(oracle['score'])
                )

                print(f"✓ {elapsed:.1f}s | value_err={value_error:.2e} score_err={score_error:.2e}")

                results.append({
                    'cell_id': cell_id,
                    'route': route,
                    'seed': seed,
                    'sqmc_value': sqmc_result['value'],
                    'sqmc_score': sqmc_result['score'],
                    'oracle_value': oracle['value'],
                    'oracle_score': oracle['score'],
                    'value_error': value_error,
                    'score_error': score_error,
                    'elapsed_sec': elapsed,
                })

            except Exception as e:
                print(f"✗ FAILED: {e}")
                results.append({
                    'cell_id': cell_id,
                    'route': route,
                    'seed': seed,
                    'error': str(e),
                })

        print()

    # Save results
    artifact_dir = ROOT / "docs" / "benchmarks" / "artifacts" / "sqmc-oracle-characterization-20260909"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    result_file = artifact_dir / "results.json"
    with open(result_file, 'w') as f:
        json.dump({
            'model_id': MODEL_ID,
            'horizon': HORIZON,
            'particle_count': PARTICLE_COUNT,
            'routes': list(ROUTES),
            'seeds': list(SEEDS),
            'oracle': oracle,
            'results': results,
        }, f, indent=2)

    print("=" * 80)
    print(f"Results saved to: {result_file}")
    print("=" * 80)

    # Print summary
    print()
    print("Summary by route:")
    for route in ROUTES:
        route_results = [r for r in results if r['route'] == route and 'error' not in r]
        if route_results:
            value_errors = [r['value_error'] for r in route_results]
            score_errors = [r['score_error'] for r in route_results]
            print(f"  {route:40s}: value_err={np.mean(value_errors):.2e}±{np.std(value_errors):.2e}  score_err={np.mean(score_errors):.2e}±{np.std(score_errors):.2e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
