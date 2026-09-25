#!/usr/bin/env python3
"""Unexecuted diagnostic draft: Phase 1 P44/frozen-model comparison.

This draft did not run in the closed campaign and supplies no replication
evidence. Its parameter mapping and score comparison have not been validated.
NumPy is used only for this independent diagnostic/reference calculation.
Do not import this module into runtime, tuning, or training routes.
Part of sqmc-control-generalization-master-program-2026-09-23.md.
"""

import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bayesfilter.highdim.ledh_canonical_models_tf import diagonal_lgssm_canonical_model

# Import P44 infrastructure
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests/highdim"))
import test_p44_lgssm_exact_baseline as P44_LGSSM

DTYPE = tf.float64
REPO_ROOT = Path(__file__).resolve().parents[2]

# Test configuration
THETA = tf.constant([0.25, np.log(0.18), np.log(0.12), 0.04], dtype=DTYPE)
SEEDS = [50001, 50002, 50003]
HORIZON = 20
STATE_DIM = 3


def _generate_3d_observations(seed: int) -> tf.Tensor:
    """Generate 3D LGSSM observations using P44 infrastructure."""
    rng = np.random.default_rng(seed)

    # Get P44 physical parameters for 3D
    theta_np = THETA.numpy()
    scale = np.array([1.0, 0.85, 0.70])
    q_scale = np.array([0.90, 1.10, 1.30])
    r_scale = np.array([1.00, 1.20, 0.80])
    mean_scale = np.array([1.00, -0.50, 0.25])

    rho = 0.55 * np.tanh(theta_np[0]) * scale
    q_diag = np.exp(theta_np[1]) * q_scale
    r_diag = np.exp(theta_np[2]) * r_scale
    raw_initial_mean = theta_np[3] * mean_scale

    transition_matrix = np.diag(rho)

    # Generate trajectory
    x = raw_initial_mean + rng.normal(0, 1.0, STATE_DIM)
    observations = []

    for t in range(HORIZON):
        if t > 0:
            x = transition_matrix @ x + rng.normal(0, 1.0, STATE_DIM) * q_diag
        obs = x + rng.normal(0, 1.0, STATE_DIM) * r_diag
        observations.append(obs)

    return tf.constant(np.array(observations), dtype=DTYPE)


def _frozen_3d_oracle(theta: tf.Tensor, observations: tf.Tensor) -> tuple:
    """Compute oracle using frozen 3D canonical model."""
    # Build 5-parameter theta for canonical model
    theta_np = theta.numpy()
    scale = np.array([1.0, 0.85, 0.70])
    q_scale_arr = np.array([0.90, 1.10, 1.30])

    phi = 0.55 * np.tanh(theta_np[0]) * scale
    q_diag = np.exp(theta_np[1]) * q_scale_arr

    # Canonical model expects [phi_1, phi_2, phi_3, q_scale, r_scale]
    q_scale_scalar = float(np.sqrt(np.mean(q_diag**2)))
    r_scale_scalar = float(np.exp(theta_np[2]))

    theta_5 = tf.constant(
        [phi[0], phi[1], phi[2], q_scale_scalar, r_scale_scalar],
        dtype=DTYPE
    )

    model, set_score_direction = diagonal_lgssm_canonical_model(theta_5)

    # Compute score for each direction
    from bayesfilter.highdim.ledh_canonical_score_tf import canonical_value_and_analytical_score
    from bayesfilter.highdim.sqmc_tf import randomized_halton_gaussian, randomized_halton_joint

    particle_count = 1008
    seed_int = 50000

    # Generate particles
    initial_states = randomized_halton_gaussian(
        num_particles=particle_count,
        dimension=STATE_DIM,
        seed=seed_int,
        salt=301,
        dtype=DTYPE,
    )

    process_rows = []
    ancestor_rows = []
    for t in range(HORIZON):
        raw, ancestors, innovations = randomized_halton_joint(
            num_particles=particle_count,
            state_dimension=STATE_DIM,
            seed=seed_int,
            salt=3001 + t,
            dtype=DTYPE,
        )
        process_rows.append(tf.math.ndtri(innovations))
        ancestor_rows.append(ancestors)

    process_noise = tf.stack(process_rows)
    ancestor_uniforms = tf.stack(ancestor_rows)

    initial_covariances = tf.eye(STATE_DIM, batch_shape=[particle_count], dtype=DTYPE)

    # Default controls
    controls = {
        'reset_epsilon': 8.0,
        'reset_sinkhorn_steps': 8,
        'reset_balance_steps': 8,
        'correction_steps': 4,
        'correction_strength': 0.2,
        'pairwise_steps': 4,
        'pairwise_strength': 0.03,
    }

    design = tf.concat([tf.eye(STATE_DIM, dtype=DTYPE), -tf.eye(STATE_DIM, dtype=DTYPE)], axis=0)
    design = tf.tile(design, [particle_count // 6, 1])

    values = []
    scores = []

    for direction_idx in range(5):
        set_score_direction(tf.one_hot(direction_idx, 5, dtype=DTYPE))

        value, score = canonical_value_and_analytical_score(
            model,
            theta_5,
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
            coordinate_cap=0.98,
            coordinate_cap_power=8,
            ancestry_policy='existing_one_to_one',
            process_ancestor_uniforms=ancestor_uniforms,
            state_map_policy='adaptive_empirical',
            hilbert_bits=12,
        )

        values.append(value)
        scores.append(score[0])

    full_value = tf.reduce_mean(values)
    full_score = tf.stack(scores)

    return full_value, full_score


def main():
    print("=" * 80)
    print("Phase 1: 3D P44 LGSSM Replication Check")
    print("=" * 80)
    print(f"Verifying dimension-generic P44 infrastructure matches frozen 3D baseline")
    print(f"State dimension: {STATE_DIM}")
    print(f"Horizon: {HORIZON}")
    print(f"Seeds: {SEEDS}")
    print()

    results = []

    for seed in SEEDS:
        print(f"Testing seed {seed}...")

        observations = _generate_3d_observations(seed)

        # Frozen 3D baseline
        print("  Computing frozen 3D canonical model baseline...")
        frozen_value, frozen_score = _frozen_3d_oracle(THETA, observations)

        # P44 Kalman oracle
        print("  Computing P44 Kalman oracle...")
        p44_value, p44_score = P44_LGSSM._p44_oracle_score(THETA, observations, STATE_DIM)

        # Compare
        value_diff = float(tf.abs(frozen_value - p44_value).numpy())
        score_l2 = float(tf.norm(frozen_score - p44_score).numpy())

        result = {
            'seed': seed,
            'frozen_value': float(frozen_value.numpy()),
            'p44_value': float(p44_value.numpy()),
            'value_diff': value_diff,
            'score_l2_diff': score_l2,
            'match': value_diff < 1e-6 and score_l2 < 1e-4,
        }

        results.append(result)

        status = "MATCH" if result['match'] else "MISMATCH"
        print(f"  {status}: value_diff={value_diff:.2e}, score_l2={score_l2:.2e}")
        print()

    # Save results
    output_dir = Path("artifacts/sqmc-3d-p44-replication-20260923")
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        'schema': 'bayesfilter.sqmc_3d_p44_replication.v1',
        'phase': 'Phase 1: 3D P44 LGSSM Replication Check',
        'master_program': 'sqmc-control-generalization-master-program-2026-09-23.md',
        'state_dim': STATE_DIM,
        'horizon': HORIZON,
        'seeds': SEEDS,
        'results': results,
        'all_match': all(r['match'] for r in results),
    }

    output_file = output_dir / 'replication_results.json'
    output_file.write_text(json.dumps(manifest, indent=2))

    print("=" * 80)
    print(f"Results saved to: {output_file}")
    print(f"All seeds match: {manifest['all_match']}")

    return 0 if manifest['all_match'] else 1


if __name__ == '__main__':
    sys.exit(main())
