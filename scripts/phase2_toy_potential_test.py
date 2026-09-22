"""Phase 2: Toy potential test for surrogate-force HMC mechanism.

Tests whether damped-force HMC samples correctly when the force is a
deterministic damping of an exact analytical gradient, isolating the
mechanism from LEDH score-estimation complexity.

Exact-force arm: F = -∇U, the analytical gradient of a 3D quadratic.
Damped-force arm: F_damped = -∇U / (1 + epsilon), a simple deterministic
                  transformation simulating OT gradient damping.

Corollary 5.2 states both must sample from the same distribution (the one
proportional to exp(-U)) because both use the exact U in the acceptance step
and both forces are deterministic functions of theta only.

If this test fails, the surrogate-force mechanism itself is broken, independent
of any LEDH-specific issues.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import tensorflow as tf
import tensorflow_probability as tfp

from bayesfilter.inference.tolerance_derivation import derive_parity_tolerance
from bayesfilter.inference.coverage import joint_mahalanobis_coverage


# 3D quadratic potential: U(θ) = 0.5 * θᵀ Q θ, Q = diag([1, 4, 9])
# True posterior: N(0, Q⁻¹) with covariance diag([1.0, 0.25, 0.111...])

Q_DIAGONAL = tf.constant([1.0, 4.0, 9.0], dtype=tf.float64)
TRUE_COVARIANCE_DIAGONAL = 1.0 / Q_DIAGONAL
TRUE_MEAN = tf.zeros([3], dtype=tf.float64)


def neg_log_prob(theta: tf.Tensor) -> tf.Tensor:
    """Negative log-probability (the potential U)."""
    return 0.5 * tf.reduce_sum(theta**2 * Q_DIAGONAL)


def exact_gradient(theta: tf.Tensor) -> tf.Tensor:
    """Exact gradient of neg_log_prob: ∇U = Q θ.

    Note: For HMC, the force is ∇(log_prob) = -∇U, so callers should negate this.
    """
    return theta * Q_DIAGONAL


def exact_force(theta: tf.Tensor) -> tf.Tensor:
    """Force for HMC leapfrog: ∇(log_prob) = -∇U."""
    return -exact_gradient(theta)


def damped_force(theta: tf.Tensor, epsilon: float = 0.01) -> tf.Tensor:
    """Damped force: exact force scaled by 1/(1+epsilon)."""
    return exact_force(theta) / (1.0 + epsilon)


def leapfrog_step(theta, momentum, force_fn, step_size, num_steps):
    """Manual leapfrog integrator with custom force.

    Args:
        theta: [num_chains, dim]
        momentum: [num_chains, dim]
        force_fn: function theta -> force (gradient of log prob)
        step_size: scalar
        num_steps: number of leapfrog steps

    Returns:
        theta_new, momentum_new after num_steps
    """
    theta = tf.identity(theta)
    momentum = tf.identity(momentum)

    # Half step for momentum
    force = force_fn(theta)
    momentum = momentum + 0.5 * step_size * force

    # Full steps
    for _ in range(num_steps - 1):
        theta = theta + step_size * momentum
        force = force_fn(theta)
        momentum = momentum + step_size * force

    # Final position full step
    theta = theta + step_size * momentum

    # Final momentum half step
    force = force_fn(theta)
    momentum = momentum + 0.5 * step_size * force

    return theta, momentum


def run_hmc_with_custom_force(
    force_fn,
    num_chains: int = 2,
    num_warmup: int = 500,
    num_samples: int = 1000,
    step_size: float = 0.1,
    num_leapfrog_steps: int = 10,
    seed: int = 42,
) -> tuple[tf.Tensor, dict]:
    """Run HMC with a custom force function via manual leapfrog integration.

    Returns:
        samples: [num_samples, num_chains, 3]
        diagnostics: dict with acceptance_rate, is_accepted trace
    """

    rng = tf.random.Generator.from_seed(seed)

    # Initial state: [num_chains, 3]
    current_theta = rng.normal([num_chains, 3], dtype=tf.float64)

    samples_list = []
    accepted_count = 0
    total_proposals = 0

    # Warmup + sampling
    for iteration in range(num_warmup + num_samples):
        # Sample momentum
        current_momentum = rng.normal(tf.shape(current_theta), dtype=tf.float64)

        # Compute current energy
        current_log_prob = -neg_log_prob(current_theta)
        current_kinetic = 0.5 * tf.reduce_sum(current_momentum**2, axis=-1)
        current_energy = -(current_log_prob - current_kinetic)  # Hamiltonian = -log_prob + kinetic

        # Propose via leapfrog
        proposed_theta, proposed_momentum = leapfrog_step(
            current_theta, current_momentum, force_fn, step_size, num_leapfrog_steps
        )

        # Compute proposed energy
        proposed_log_prob = -neg_log_prob(proposed_theta)
        proposed_kinetic = 0.5 * tf.reduce_sum(proposed_momentum**2, axis=-1)
        proposed_energy = -(proposed_log_prob - proposed_kinetic)

        # Metropolis acceptance
        log_accept_ratio = current_energy - proposed_energy
        log_uniform = tf.math.log(rng.uniform(tf.shape(log_accept_ratio), dtype=tf.float64))

        accept = log_uniform < log_accept_ratio

        # Update state (per-chain acceptance)
        current_theta = tf.where(
            accept[:, None],  # broadcast to [num_chains, 1]
            proposed_theta,
            current_theta
        )

        # Track acceptance after warmup
        if iteration >= num_warmup:
            accepted_count += tf.reduce_sum(tf.cast(accept, tf.int32)).numpy()
            total_proposals += num_chains
            samples_list.append(current_theta.numpy())

    # Stack samples: [num_samples, num_chains, 3]
    samples = tf.constant(samples_list, dtype=tf.float64)

    acceptance_rate = accepted_count / total_proposals if total_proposals > 0 else 0.0

    diagnostics = {
        "acceptance_rate": float(acceptance_rate),
        "num_chains": num_chains,
        "num_samples": num_samples,
        "num_warmup": num_warmup,
    }

    return samples, diagnostics


def compute_wasserstein_gaussian_approx(
    samples1: tf.Tensor, samples2: tf.Tensor
) -> float:
    """Gaussian Wasserstein-2 distance between two sample sets.

    W₂²(N(μ₁,Σ₁), N(μ₂,Σ₂)) = ||μ₁-μ₂||² + trace(Σ₁ + Σ₂ - 2(Σ₁^½ Σ₂ Σ₁^½)^½)

    Simplified to Frobenius-based upper bound for robustness.
    """
    mu1 = tf.reduce_mean(samples1, axis=0)
    mu2 = tf.reduce_mean(samples2, axis=0)

    cov1 = tfp.stats.covariance(samples1, sample_axis=0, event_axis=-1)
    cov2 = tfp.stats.covariance(samples2, sample_axis=0, event_axis=-1)

    mean_dist_sq = tf.reduce_sum((mu1 - mu2) ** 2)
    cov_dist_sq = tf.reduce_sum((cov1 - cov2) ** 2)

    w2_approx = tf.sqrt(mean_dist_sq + cov_dist_sq)

    return float(w2_approx.numpy())


def main():
    print("=" * 70)
    print("Phase 2: Toy Potential Surrogate-Force HMC Test")
    print("=" * 70)
    print()

    # MCMC-appropriate tolerance: agreement within Monte Carlo error.
    # With ~2000 effective samples (2 chains × 1000 × ~1.0 ESS ratio),
    # MCMC SE per coordinate is ~ posterior_std / sqrt(2000) ~ 1.0 / 45 ~ 0.02.
    # Pooled across 3 coordinates via W₂: expect ~ sqrt(3) * 0.02 ~ 0.035.
    # Tolerance = 3× MCMC SE to allow for finite-sample noise: 0.1.
    #
    # This is NOT the same as derive_parity_tolerance, which is for numerical
    # agreement (refactored code vs golden master). MCMC agreement operates
    # at the statistical scale, not the floating-point precision scale.

    tolerance_mcmc = 0.1

    print(f"MCMC agreement tolerance: {tolerance_mcmc:.4f}")
    print(f"(Based on ~sqrt(3) * posterior_std / sqrt(ESS) * 3-sigma margin)")
    print()

    # Arm 1: Exact-force HMC
    print("Running Arm 1: Exact-force HMC...")
    samples1, diag1 = run_hmc_with_custom_force(
        force_fn=exact_force,
        num_chains=2,
        num_warmup=500,
        num_samples=1000,
        seed=12345,
    )
    # Flatten chains: [num_samples * num_chains, 3]
    samples1_flat = tf.reshape(samples1, [-1, 3])

    print(f"  Acceptance rate: {diag1['acceptance_rate']:.3f}")
    print(f"  Mean: {tf.reduce_mean(samples1_flat, axis=0).numpy()}")
    print(f"  Std:  {tf.math.reduce_std(samples1_flat, axis=0).numpy()}")
    print()

    # Arm 2: Damped-force HMC
    print("Running Arm 2: Damped-force HMC (epsilon=0.01)...")
    samples2, diag2 = run_hmc_with_custom_force(
        force_fn=lambda theta: damped_force(theta, epsilon=0.01),
        num_chains=2,
        num_warmup=500,
        num_samples=1000,
        seed=12346,
    )
    samples2_flat = tf.reshape(samples2, [-1, 3])

    print(f"  Acceptance rate: {diag2['acceptance_rate']:.3f}")
    print(f"  Mean: {tf.reduce_mean(samples2_flat, axis=0).numpy()}")
    print(f"  Std:  {tf.math.reduce_std(samples2_flat, axis=0).numpy()}")
    print()

    # Primary criterion: W₂ distance
    print("Computing W₂ distance...")
    w2_dist = compute_wasserstein_gaussian_approx(samples1_flat, samples2_flat)
    print(f"  W₂ distance: {w2_dist:.6f}")
    print(f"  Tolerance:   {tolerance_mcmc:.6f}")
    print()

    passed = w2_dist < tolerance_mcmc
    print(f"Primary criterion: {'PASS' if passed else 'FAIL'}")
    print()

    # Explanatory diagnostics: coverage
    print("Explanatory diagnostics — True-θ coverage:")
    cov1 = joint_mahalanobis_coverage(samples1_flat, TRUE_MEAN, alpha=0.05)
    cov2 = joint_mahalanobis_coverage(samples2_flat, TRUE_MEAN, alpha=0.05)

    print(f"  Arm 1 covers: {cov1['covers']} (p={cov1['p_value']:.3f})")
    print(f"  Arm 2 covers: {cov2['covers']} (p={cov2['p_value']:.3f})")
    print()

    # Veto checks
    vetoes = []
    if diag1["acceptance_rate"] < 0.15:
        vetoes.append("Arm 1 acceptance < 0.15")
    if diag2["acceptance_rate"] < 0.15:
        vetoes.append("Arm 2 acceptance < 0.15")

    if vetoes:
        print(f"Veto conditions: {', '.join(vetoes)}")
        print()

    # Write result
    result = {
        "phase": "2",
        "status": "PASS" if (passed and not vetoes) else "FAIL",
        "date": "2026-09-07",
        "w2_distance": w2_dist,
        "tolerance": tolerance_mcmc,
        "tolerance_note": "MCMC agreement scale (~3× Monte Carlo SE), not numerical precision",
        "arm1": {
            "acceptance_rate": diag1["acceptance_rate"],
            "mean": samples1_flat.numpy().mean(axis=0).tolist(),
            "std": samples1_flat.numpy().std(axis=0).tolist(),
            "coverage": cov1,
        },
        "arm2": {
            "acceptance_rate": diag2["acceptance_rate"],
            "mean": samples2_flat.numpy().mean(axis=0).tolist(),
            "std": samples2_flat.numpy().std(axis=0).tolist(),
            "coverage": cov2,
        },
        "vetoes": vetoes,
        "interpretation": (
            "Damped-force HMC samples the same distribution as exact-force HMC "
            "on a 3D quadratic potential, confirming Corollary 5.2 mechanism."
            if passed and not vetoes
            else "Surrogate-force mechanism fails on toy potential."
        ),
    }

    output_path = Path("results/phase2-summary.json")
    output_path.parent.mkdir(exist_ok=True, parents=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Result written to: {output_path}")
    print()

    if passed and not vetoes:
        print("✓ Phase 2 complete: surrogate-force mechanism works on toy potential.")
        return 0
    else:
        print("✗ Phase 2 failed: see result summary for details.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
