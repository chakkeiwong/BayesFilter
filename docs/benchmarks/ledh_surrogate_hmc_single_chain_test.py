"""Test single HMC chain with LEDH dual-parameter target."""

import tensorflow as tf
import tensorflow_probability as tfp

from bayesfilter.highdim.ledh_canonical_neutra_targets_tf import (
    _diagonal_lgssm_fused_model,
    _lgssm_frozen_observations,
)
from bayesfilter.inference.ledh_dual_parameter_target import (
    DualParameterLEDHTarget,
)

tfb = tfp.bijectors
tfd = tfp.distributions


def main():
    print("=" * 70)
    print("Single Chain Memory Test")
    print("=" * 70)

    # Load LGSSM T=50 fixture
    observations = _lgssm_frozen_observations()
    model = _diagonal_lgssm_fused_model()

    # Minimal parameters
    d = 3
    N = 500  # Even smaller
    T = observations.shape[0]
    dtype = observations.dtype

    print(f"\nParticles: {N}")
    print(f"Horizon: {T}")

    # Initialize
    generator = tf.random.Generator.from_seed(81100)
    initial_states = generator.normal([N, d], dtype=dtype) * 0.1
    initial_covariances = tf.tile(
        tf.eye(d, dtype=dtype)[None, :, :], [N, 1, 1]
    ) * 0.01
    noises = generator.normal([T, N, d], dtype=dtype) * 0.1

    # Shared parameters
    reset_basis = tf.concat(
        [tf.eye(d, dtype=dtype), -tf.eye(d, dtype=dtype)],
        axis=0,
    )
    reset_repeats = (N + 2 * d - 1) // (2 * d)
    reset_design = tf.tile(reset_basis, [reset_repeats, 1])[:N]

    shared_params = dict(
        substeps=8,
        reset_policy="contract_e",
        reset_design=reset_design,
        reset_epsilon=2.0,
        reset_sinkhorn_steps=8,
        reset_balance_steps=8,
        correction_steps=4,
        correction_strength=0.2,
        correction_lm_scale_floor=1e-4,
        correction_trust_radius=0.5,
        pairwise_steps=4,
        pairwise_strength=0.02,
        pairwise_rms_cap=2.0,
        coordinate_cap=0.0,
        annealed_stages=1,
        annealed_seed=0,
    )

    # Create target
    exact_params = {
        **shared_params,
        "reset_ridge": 1e-5,
        "correction_lm_damping": 1e-2,
    }
    biased_params = {
        **shared_params,
        "reset_ridge": 1e-3,
        "correction_lm_damping": 1.0,
    }

    target = DualParameterLEDHTarget(
        model=model,
        initial_states=initial_states,
        initial_covariances=initial_covariances,
        noises=noises,
        observations=observations,
        exact_params=exact_params,
        biased_params=biased_params,
    )

    print("\nRunning SHORT HMC chain (100 burn-in + 100 samples)...")

    # Very short chain without tf.function
    kernel = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=target,
        step_size=0.01,
        num_leapfrog_steps=5,
    )

    adaptive_kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
        inner_kernel=kernel,
        num_adaptation_steps=80,
        target_accept_prob=0.65,
    )

    true_theta = tf.constant([1.0, 1.0, 1.0, 0.5, 0.3], dtype=dtype)
    init_theta = true_theta + generator.normal([5], dtype=dtype) * 0.1

    # Run WITHOUT tf.function to avoid graph explosion
    samples, trace = tfp.mcmc.sample_chain(
        num_results=100,
        num_burnin_steps=100,
        current_state=init_theta,
        kernel=adaptive_kernel,
        trace_fn=lambda _, pkr: {
            "is_accepted": pkr.inner_results.is_accepted,
            "step_size": pkr.new_step_size,
        },
        seed=42,
    )

    acceptance_rate = float(tf.reduce_mean(tf.cast(trace["is_accepted"], tf.float32)))
    ess = tfp.mcmc.effective_sample_size(samples, filter_beyond_positive_pairs=True)
    ess_per_param = float(tf.reduce_mean(ess))

    print(f"\nResults:")
    print(f"  Acceptance rate: {acceptance_rate:.3f}")
    print(f"  ESS per param: {ess_per_param:.1f}")
    print(f"  Sample shape: {samples.shape}")

    print("\n" + "=" * 70)
    print("✓ Test completed successfully")


if __name__ == "__main__":
    main()
