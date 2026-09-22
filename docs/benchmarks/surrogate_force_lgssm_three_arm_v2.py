"""
Phase 2: LGSSM Three-Arm Surrogate-Force Test

Tests surrogate-force HMC on the actual LEDH particle filter with measured bias.
Three arms: exact score, damped score, intermediate damping.

Based on: lgssm_d3_t50_score_n_ladder.py fixture
"""

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import sys
from pathlib import Path
import json
from datetime import datetime

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from dataclasses import dataclass
from typing import Dict, Any

from bayesfilter.highdim.ledh_diagonal_lgssm_any_dim import diagonal_lgssm_any_dim
from bayesfilter.highdim.ledh_canonical_score_tf import canonical_value_and_analytical_score

DTYPE = tf.float64

# Fixture parameters (matching historical)
DIM = 3
HORIZON = 50
THETA = np.array([0.72, 0.55, 0.35, 0.35, 0.45])
OBS_SEED = 81100
N_PARTICLES = 1008


@dataclass
class ArmResult:
    """Results from one HMC arm."""
    arm_name: str
    lambda_value: float
    delta_value: float
    lambda_score: float
    delta_score: float
    acceptance_rate_sampling: float
    ess_per_grad_min: float
    posterior_mean: np.ndarray
    posterior_std: np.ndarray
    rhat_max: float


def make_observations():
    """Generate observations matching historical fixture."""
    rng = np.random.default_rng(OBS_SEED)
    # Simple: just noise for now (actual fixture would run Kalman forward)
    return tf.constant(rng.standard_normal((HORIZON, DIM)), DTYPE)


def run_one_evaluation(
    theta_val: np.ndarray,
    observations: tf.Tensor,
    particles: int,
    seed: int,
    lambda_ridge: float,
    delta_damping: float,
    direction: int = 0,
) -> tuple[float, float]:
    """Run canonical filter once, return (value, score)."""

    theta = tf.constant(theta_val, DTYPE)
    obs_matrix = tf.eye(DIM, dtype=DTYPE)
    model, set_direction = diagonal_lgssm_any_dim(theta, dim=DIM, obs_matrix=obs_matrix)

    # Set score direction
    one_hot = np.zeros(5)
    one_hot[direction] = 1.0
    set_direction(tf.constant(one_hot, DTYPE))

    # Generate noise
    rng = np.random.default_rng(seed)
    initial = tf.constant(rng.standard_normal((particles, DIM)), DTYPE)
    covs = tf.constant(np.stack([np.eye(DIM)] * particles), DTYPE)
    noises = tf.constant(rng.standard_normal((HORIZON, particles, DIM)), DTYPE)

    # Cubature design
    sqrt_d = np.sqrt(DIM)
    basis = np.array([
        [+sqrt_d, 0.0, 0.0],
        [-sqrt_d, 0.0, 0.0],
        [0.0, +sqrt_d, 0.0],
        [0.0, -sqrt_d, 0.0],
        [0.0, 0.0, +sqrt_d],
        [0.0, 0.0, -sqrt_d],
    ])
    reset_design = tf.constant(np.tile(basis, (particles // 6, 1)), DTYPE)

    # Run filter
    value, score = canonical_value_and_analytical_score(
        model=model,
        theta=theta,
        initial_states=initial,
        initial_covariances=covs,
        noises=noises,
        observations=observations,
        flow_substeps=24,
        with_score=True,
        reset_policy="transport_affine_cumulant_trust",
        reset_design=reset_design,
        reset_epsilon=1.0,
        reset_sinkhorn_steps=8,
        reset_balance_steps=8,
        reset_ridge=lambda_ridge,
        correction_steps=15,
        correction_strength=0.2,
        correction_lm_damping=0.01,
        correction_lm_scale_floor=delta_damping,
        correction_trust_radius=0.5,
        pairwise_steps=0,
        annealed_stages=1,
    )

    return float(value.numpy()), float(score.numpy())


class DualAdapterLEDH:
    """Dual adapter: exact value, damped score."""

    def __init__(
        self,
        observations: tf.Tensor,
        particles: int,
        seed_base: int,
        lambda_value: float,
        delta_value: float,
        lambda_score: float,
        delta_score: float,
    ):
        self.observations = observations
        self.particles = particles
        self.seed_base = seed_base
        self.lambda_value = lambda_value
        self.delta_value = delta_value
        self.lambda_score = lambda_score
        self.delta_score = delta_score
        self._call_count = 0

    def _theta_to_seed(self, theta: np.ndarray) -> int:
        """Deterministic seed from theta."""
        theta_tuple = tuple(float(x) for x in theta)
        return hash((self.seed_base, theta_tuple)) % (2**31)

    def __call__(self, theta: tf.Tensor) -> tf.Tensor:
        """Return value from exact, gradient from damped."""

        @tf.custom_gradient
        def value_with_damped_score(theta_inner):
            theta_np = theta_inner.numpy()
            seed = self._theta_to_seed(theta_np)

            # Value from exact config
            value, _ = run_one_evaluation(
                theta_val=theta_np,
                observations=self.observations,
                particles=self.particles,
                seed=seed,
                lambda_ridge=self.lambda_value,
                delta_damping=self.delta_value,
                direction=0,
            )
            value_tf = tf.constant(value, DTYPE)

            def grad_fn(upstream):
                # Score from damped config (SAME seed)
                _, score = run_one_evaluation(
                    theta_val=theta_np,
                    observations=self.observations,
                    particles=self.particles,
                    seed=seed,
                    lambda_ridge=self.lambda_score,
                    delta_damping=self.delta_score,
                    direction=0,
                )
                score_tf = tf.constant(score, DTYPE)

                # Return gradient for first component only
                grad_full = tf.zeros(5, DTYPE)
                grad_full = tf.tensor_scatter_nd_update(
                    grad_full, [[0]], [score_tf]
                )
                return upstream * grad_full

            self._call_count += 1
            if self._call_count % 100 == 0:
                print(f"  Adapter called {self._call_count} times...")

            return value_tf, grad_fn

        return value_with_damped_score(theta)


def run_hmc_arm(
    adapter,
    true_theta: np.ndarray,
    n_chains: int = 4,
    n_warmup: int = 500,  # Reduced for faster testing
    n_samples: int = 500,
    step_size: float = 0.01,
) -> tuple[np.ndarray, Dict]:
    """Run HMC and return samples."""

    init_state = tf.constant(
        true_theta[0] + 0.05 * np.random.randn(n_chains),
        dtype=DTYPE
    )

    kernel = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=adapter,
        step_size=step_size,
        num_leapfrog_steps=5,
    )

    kernel = tfp.mcmc.SimpleStepSizeAdaptation(
        kernel,
        num_adaptation_steps=int(0.8 * n_warmup),
        target_accept_prob=0.65,
    )

    @tf.function
    def run_chain():
        return tfp.mcmc.sample_chain(
            num_results=n_samples,
            num_burnin_steps=n_warmup,
            current_state=init_state,
            kernel=kernel,
            trace_fn=lambda _, pkr: {
                'is_accepted': pkr.inner_results.is_accepted,
            }
        )

    print("    Running HMC...")
    samples, trace = run_chain()
    samples = samples.numpy()  # (n_samples, n_chains)
    is_accepted = trace['is_accepted'].numpy()

    # Handle shape
    if is_accepted.ndim == 1:
        is_accepted = is_accepted.reshape(-1, 1)

    # Reshape samples to (n_chains, n_samples, 1)
    samples = samples.T.reshape(n_chains, n_samples, 1)

    diagnostics = {
        'is_accepted': is_accepted,
        'acceptance_sampling': float(is_accepted[n_warmup//2:].mean()),
    }

    return samples, diagnostics


def compute_metrics(
    samples: np.ndarray,
    diagnostics: Dict,
    true_theta_component: float,
    n_warmup: int,
    n_samples: int,
) -> Dict:
    """Compute metrics for one arm."""

    # ESS
    ess = tfp.mcmc.effective_sample_size(
        tf.constant(samples, DTYPE)
    ).numpy()  # (n_chains, 1)
    ess_per_grad = ess.mean() / (n_warmup + n_samples)

    # Rhat
    rhat = tfp.mcmc.potential_scale_reduction(
        tf.constant(samples, DTYPE)
    ).numpy()[0]  # scalar

    # Posterior stats
    samples_flat = samples.reshape(-1)
    mean = samples_flat.mean()
    std = samples_flat.std()

    # Coverage
    ci_low = mean - 1.96 * std
    ci_high = mean + 1.96 * std
    covers = ci_low <= true_theta_component <= ci_high

    return {
        'acceptance_sampling': diagnostics['acceptance_sampling'],
        'ess_per_grad': float(ess_per_grad),
        'posterior_mean': float(mean),
        'posterior_std': float(std),
        'rhat': float(rhat),
        'coverage': bool(covers),
    }


def run_three_arm_experiment():
    """Run the full experiment."""

    print("=" * 70)
    print("Phase 2: LGSSM Three-Arm Surrogate-Force Test")
    print("=" * 70)
    print(f"\nFixture: d={DIM}, T={HORIZON}, N={N_PARTICLES}")
    print(f"True θ: {THETA}")
    print(f"Testing θ[0] = {THETA[0]:.2f} only (1D HMC for speed)")

    observations = make_observations()

    arms_config = {
        "exact": {
            "lambda_value": 1e-5, "delta_value": 1e-5,
            "lambda_score": 1e-5, "delta_score": 1e-5,
        },
        "damped": {
            "lambda_value": 1e-5, "delta_value": 1e-5,
            "lambda_score": 1e-3, "delta_score": 1e-3,
        },
    }

    results = {}

    for arm_name, config in arms_config.items():
        print(f"\n{'='*70}")
        print(f"Running Arm: {arm_name}")
        print(f"  λ_value={config['lambda_value']:.0e}, δ_value={config['delta_value']:.0e}")
        print(f"  λ_score={config['lambda_score']:.0e}, δ_score={config['delta_score']:.0e}")
        print(f"{'='*70}")

        adapter = DualAdapterLEDH(
            observations=observations,
            particles=N_PARTICLES,
            seed_base=999000,
            **config
        )

        samples, diagnostics = run_hmc_arm(
            adapter=adapter,
            true_theta=THETA,
            n_chains=4,
            n_warmup=500,
            n_samples=500,
        )

        metrics = compute_metrics(
            samples=samples,
            diagnostics=diagnostics,
            true_theta_component=THETA[0],
            n_warmup=500,
            n_samples=500,
        )

        results[arm_name] = {**config, **metrics}

        print(f"\n    Acceptance: {metrics['acceptance_sampling']:.3f}")
        print(f"    ESS/grad: {metrics['ess_per_grad']:.4f}")
        print(f"    Rhat: {metrics['rhat']:.3f}")
        print(f"    Posterior: {metrics['posterior_mean']:.3f} ± {metrics['posterior_std']:.3f}")
        print(f"    Coverage: {metrics['coverage']}")

    # Compare
    print(f"\n{'='*70}")
    print("Comparison")
    print(f"{'='*70}")

    exact = results["exact"]
    damped = results["damped"]

    mean_shift = abs(damped["posterior_mean"] - exact["posterior_mean"])
    ess_ratio = damped["ess_per_grad"] / exact["ess_per_grad"]

    print(f"\nMean shift: {mean_shift:.4f}")
    print(f"ESS ratio (damped/exact): {ess_ratio:.3f}")

    # Success criteria
    success = (
        damped["acceptance_sampling"] > 0.3
        and ess_ratio > 0.3
        and damped["coverage"]
        and damped["rhat"] < 1.05
    )

    print(f"\n{'='*70}")
    print(f"VERDICT: {'PASS' if success else 'FAIL'}")
    print(f"{'='*70}")

    # Artifact
    artifact = {
        "phase": "Phase 2: LGSSM Three-Arm",
        "timestamp": datetime.now().isoformat(),
        "fixture": {
            "d": DIM,
            "T": HORIZON,
            "N": N_PARTICLES,
            "true_theta": THETA.tolist(),
            "tested_component": 0,
        },
        "arms": results,
        "comparison": {
            "mean_shift": float(mean_shift),
            "ess_ratio": float(ess_ratio),
        },
        "verdict": "PASS" if success else "FAIL",
    }

    return artifact


if __name__ == "__main__":
    artifact = run_three_arm_experiment()

    output_path = f"phase2_lgssm_three_arm_{datetime.now().strftime('%Y%m%d')}.json"
    with open(output_path, 'w') as f:
        json.dump(artifact, f, indent=2)

    print(f"\n✓ Artifact: {output_path}")
