"""Debug T=3 1D LGSSM score computation step-by-step.

T=3 shows 44% relative error (exact score -0.43, analytical -0.24, error +0.19).
This script instruments each step to identify where the tangent diverges.

Strategy:
1. Run the score lane normally to reproduce the error
2. Extract intermediate states at each step
3. Compute finite-difference tangents at each step for comparison
4. Report which component deviates

Components to check:
- UKF predict tangent
- Flow tangent
- UKF update tangent
- Resampling/reset tangent (Sinkhorn + trust-region)
"""

import numpy as np
import sys
import tensorflow as tf
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from bayesfilter.highdim.ledh_diagonal_lgssm_any_dim import diagonal_lgssm_any_dim

DTYPE = tf.float64
THETA = np.array([0.9, 0.6, 0.8])  # [phi, sw, sv]
SEED = 999000
PARTICLES = 1008
EPSILON = 1.0
SINKHORN_STEPS = 8
FLOW_SUBSTEPS = 12


def exact_kalman(obs, theta=THETA):
    """Exact Kalman filter for 1D AR(1)."""
    phi, sw, sv = theta
    mean, var, total = 0.0, 1.0, 0.0
    for y in obs:
        mean = phi * mean
        var = phi * var * phi + sw * sw
        s = var + sv * sv
        resid = y - mean
        total += -0.5 * (resid * resid / s + np.log(2.0 * np.pi * s))
        gain = var / s
        mean, var = mean + gain * resid, (1.0 - gain) * var
    return float(total)


def exact_score(obs, dir=0, h=1e-5):
    """Exact score by central differences."""
    e = np.zeros_like(THETA)
    e[dir] = h
    return (exact_kalman(obs, THETA + e) - exact_kalman(obs, THETA - e)) / (2 * h)


def generate_observations(T, seed=9000):
    """Generate T observations from the 1D LGSSM."""
    rng = np.random.default_rng(seed)
    phi, sw, sv = THETA
    x, obs = 0.0, []
    for _ in range(T):
        x = phi * x + sw * rng.standard_normal()
        obs.append(x + sv * rng.standard_normal())
    return np.array(obs)


def run_score_lane_instrumented(obs, seed):
    """Run score lane with step-by-step instrumentation."""
    from bayesfilter.highdim.ledh_canonical_score_tf import (
        canonical_value_and_analytical_score,
    )

    theta = tf.constant(THETA, DTYPE)
    model, set_direction = diagonal_lgssm_any_dim(
        theta, dim=1, obs_matrix=tf.constant([[1.0]], DTYPE)
    )

    one_hot = np.zeros(3)
    one_hot[0] = 1.0  # derivative w.r.t. phi
    set_direction(tf.constant(one_hot, DTYPE))

    horizon = int(obs.shape[0])
    rng = np.random.default_rng(9000 + seed)
    initial = tf.constant(
        np.zeros(1)[None, :] + rng.standard_normal((PARTICLES, 1)), DTYPE
    )
    covs = tf.constant(np.stack([np.eye(1)] * PARTICLES), DTYPE)
    noises = tf.constant(rng.standard_normal((horizon, PARTICLES, 1)), DTYPE)
    reset_design = tf.constant(
        np.tile(np.array([[1.0], [-1.0]]), (PARTICLES // 2, 1)), DTYPE
    )

    # Run the score lane
    value, score = canonical_value_and_analytical_score(
        model, theta, initial, covs, noises,
        tf.constant(obs[:, None], DTYPE),
        flow_substeps=FLOW_SUBSTEPS,
        with_score=True,
        reset_policy="contract_e",
        reset_design=reset_design,
        reset_epsilon=EPSILON,
        reset_sinkhorn_steps=SINKHORN_STEPS,
        reset_balance_steps=SINKHORN_STEPS,
        reset_ridge=1e-5,
        correction_steps=1,
        pairwise_steps=1,
        annealed_stages=1,
        annealed_seed=17,
    )

    return float(value.numpy()), float(score[0].numpy())


def finite_difference_score_per_step(obs, seed, h=1e-5):
    """Compute finite-difference score by perturbing theta and re-running.

    This gives us the 'true' tangent to compare against the analytical one.
    We can't easily instrument the internal steps, but we can verify the
    end-to-end result.
    """
    def run_with_theta(theta_val):
        from bayesfilter.highdim.ledh_canonical_score_tf import (
            canonical_value_and_analytical_score,
        )

        theta = tf.constant(theta_val, DTYPE)
        model, set_direction = diagonal_lgssm_any_dim(
            theta, dim=1, obs_matrix=tf.constant([[1.0]], DTYPE)
        )
        # Direction doesn't matter for value-only run
        set_direction(tf.constant([0.0, 0.0, 0.0], DTYPE))

        horizon = int(obs.shape[0])
        rng = np.random.default_rng(9000 + seed)
        initial = tf.constant(
            np.zeros(1)[None, :] + rng.standard_normal((PARTICLES, 1)), DTYPE
        )
        covs = tf.constant(np.stack([np.eye(1)] * PARTICLES), DTYPE)
        noises = tf.constant(rng.standard_normal((horizon, PARTICLES, 1)), DTYPE)
        reset_design = tf.constant(
            np.tile(np.array([[1.0], [-1.0]]), (PARTICLES // 2, 1)), DTYPE
        )

        value, _ = canonical_value_and_analytical_score(
            model, theta, initial, covs, noises,
            tf.constant(obs[:, None], DTYPE),
            flow_substeps=FLOW_SUBSTEPS,
            with_score=False,  # Don't need score
            reset_policy="contract_e",
            reset_design=reset_design,
            reset_epsilon=EPSILON,
            reset_sinkhorn_steps=SINKHORN_STEPS,
            reset_balance_steps=SINKHORN_STEPS,
            reset_ridge=1e-5,
            correction_steps=1,
            pairwise_steps=1,
            annealed_stages=1,
            annealed_seed=17,
        )
        return float(value.numpy())

    # Finite difference in phi direction
    e = np.zeros_like(THETA)
    e[0] = h
    plus = run_with_theta(THETA + e)
    minus = run_with_theta(THETA - e)
    return (plus - minus) / (2 * h)


def main():
    T = 3
    obs = generate_observations(T, seed=9000)

    print(f"Debugging T={T} 1D LGSSM score computation")
    print(f"Observations: {obs}")
    print(f"Parameters: phi={THETA[0]}, sw={THETA[1]}, sv={THETA[2]}")
    print(f"Controls: N={PARTICLES}, eps={EPSILON}, sk={SINKHORN_STEPS}, flow={FLOW_SUBSTEPS}")
    print()

    # Exact references
    ref_val = exact_kalman(obs)
    ref_score = exact_score(obs, dir=0)
    print(f"Exact Kalman:")
    print(f"  value:       {ref_val:.6f}")
    print(f"  score (φ):   {ref_score:+.6f}")
    print()

    # Score lane analytical
    val, sc = run_score_lane_instrumented(obs, SEED)
    print(f"Score lane (analytical JVP):")
    print(f"  value:       {val:.6f}  (error: {val - ref_val:+.6f})")
    print(f"  score (φ):   {sc:+.6f}  (error: {sc - ref_score:+.6f})")
    print(f"  relative error: {(sc - ref_score) / ref_score * 100:+.2f}%")
    print()

    # Finite difference on the score lane itself
    print("Finite-difference score (on the particle filter value, h=1e-5):")
    fd_score = finite_difference_score_per_step(obs, SEED, h=1e-5)
    print(f"  FD score:    {fd_score:+.6f}  (vs analytical {sc:+.6f})")
    print(f"  FD error vs exact Kalman: {fd_score - ref_score:+.6f}")
    print(f"  Analytical error vs exact Kalman: {sc - ref_score:+.6f}")
    print()

    print("Interpretation:")
    print("  If FD score ≈ analytical score: the JVP is correct, but the particle")
    print("    filter value is biased (unlikely at N=1008)")
    print("  If FD score ≈ exact Kalman: the JVP is wrong")
    print("  If both are wrong: something else is broken")


if __name__ == "__main__":
    main()
