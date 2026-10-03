#!/usr/bin/env python3
"""Debug single SQMC evaluation to see actual error."""

import os
import sys

# Ensure we import from the current worktree, not a stale installation
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import tensorflow as tf
import numpy as np

# Match tuning script settings
DTYPE = tf.float64

def diagonal_lgssm_canonical_model(theta):
    """Minimal LGSSM model for debugging."""
    from bayesfilter.highdim.ledh_canonical_models_tf import (
        diagonal_lgssm_canonical_model as build_model,
    )
    return build_model(theta)

def main():
    # Fixed parameters from tuning script
    theta = tf.constant([0.9, 0.8, 0.7, 0.6, 0.8], dtype=DTYPE)
    horizon = 20
    particle_count = 1008
    seed = 42

    # Generate minimal observations (frozen LGSSM sequence, 3-dimensional)
    # The actual tuning script uses _lgssm_frozen_observations() which is 3D
    from bayesfilter.highdim.ledh_canonical_neutra_targets_tf import (
        _lgssm_frozen_observations,
    )
    observations = tf.cast(_lgssm_frozen_observations()[:horizon], DTYPE)

    # Minimal controls
    controls = {
        'reset_epsilon': 0.001,
        'reset_sinkhorn_steps': 10,
        'reset_balance_steps': 10,
        'correction_steps': 1,
        'correction_strength': 0.1,
        'pairwise_steps': 0,
        'pairwise_strength': 0.0,
    }

    # Generate initial states (iid_dual_cap route)
    initial_states = tf.random.stateless_normal(
        [particle_count, 3], [seed, 101], dtype=DTYPE
    )
    process_noise = tf.stack([
        tf.random.stateless_normal(
            [particle_count, 3], [seed, 1001 + t], dtype=DTYPE
        )
        for t in range(horizon)
    ])
    ancestor_uniforms = tf.zeros([horizon, particle_count], DTYPE)

    # Build model
    model, set_score_direction = diagonal_lgssm_canonical_model(theta)

    initial_covariances = tf.eye(3, batch_shape=[particle_count], dtype=DTYPE)

    # Minimal reset design (same as tuning script)
    if particle_count % (2 * 3):
        raise ValueError("particle count must be divisible by 2 * state dimension")
    base = tf.concat(
        [tf.eye(3, dtype=DTYPE), -tf.eye(3, dtype=DTYPE)], axis=0
    )
    design = tf.tile(base, [particle_count // (2 * 3), 1])

    # Try to evaluate score for first direction
    print("Setting score direction...")
    set_score_direction(tf.one_hot(0, len(theta), dtype=DTYPE))

    print("Calling canonical_value_and_analytical_score...")
    from bayesfilter.highdim.ledh_canonical_score_tf import canonical_value_and_analytical_score

    try:
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
            coordinate_cap=0.98,
            coordinate_cap_power=8,
            ancestry_policy='existing_one_to_one',
            process_ancestor_uniforms=ancestor_uniforms,
            state_map_policy='adaptive_empirical',
            hilbert_bits=12,
        )

        print(f"SUCCESS! value={value.numpy()}, score shape={score.shape}")
        print(f"Score: {score.numpy()}")

    except Exception as e:
        print(f"\nFAILURE: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main())
