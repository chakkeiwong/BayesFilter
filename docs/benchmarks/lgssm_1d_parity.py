"""1D LGSSM score parity check against exact Kalman.

Question: does the score error exist in d=1, or only emerge in higher
dimensions where particle ancestry and Sinkhorn convergence matter more?

Model: AR(1) with Gaussian observation
  x_{t+1} = φ x_t + w_t,  w_t ~ N(0, σ_w²)
  y_t = x_t + v_t,        v_t ~ N(0, σ_v²)

Parameters: θ = [φ, σ_w, σ_v] = [0.9, 0.6, 0.8]
Horizon: T=50
Direction: derivative w.r.t. φ

Controls: same as dlgssm selected config (eps=1.0, sk=8, flow=12, N=1008).
"""

import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

DTYPE = tf.float64
THETA = np.array([0.9, 0.6, 0.8])
HORIZON = 50
SEEDS = [999000, 999001, 999002]

# Controls from dlgssm pilot
EPSILON = 1.0
SINKHORN_STEPS = 8
FLOW_SUBSTEPS = 12
PARTICLES = 1008


def exact_kalman_loglik(observations, theta=THETA):
    """Exact Kalman filter for 1D AR(1) model."""
    phi, sw, sv = theta
    mean, var = 0.0, 1.0
    total = 0.0
    for y in observations:
        mean = phi * mean
        var = phi * var * phi + sw * sw
        s = var + sv * sv
        resid = y - mean
        total += -0.5 * (resid * resid / s + np.log(2.0 * np.pi * s))
        gain = var / s
        mean = mean + gain * resid
        var = (1.0 - gain) * var
    return float(total)


def exact_kalman_score(observations, direction=0, h=1.0e-5):
    """Exact score by central differences."""
    e = np.zeros_like(THETA)
    e[direction] = h
    plus = exact_kalman_loglik(observations, THETA + e)
    minus = exact_kalman_loglik(observations, THETA - e)
    return (plus - minus) / (2.0 * h)


def value_lane(observations, seed):
    """1D LGSSM value lane."""
    from bayesfilter.highdim.ledh_canonical_filter_tf import (
        canonical_value_and_diagnostics,
        CanonicalModelCallbacks,
    )
    from bayesfilter.highdim.ledh_diagonal_lgssm_any_dim import (
        diagonal_lgssm_any_dim,
    )

    theta = tf.constant(THETA, DTYPE)
    model, _ = diagonal_lgssm_any_dim(
        theta, dim=1, obs_matrix=tf.constant([[1.0]], DTYPE)
    )

    callbacks = CanonicalModelCallbacks(
        model_id="lgssm_1d_parity",
        state_dim=1,
        observation_dim=1,
        transition_mean_fn=lambda p, t: model.transition_mean_fn(theta, p),
        transition_log_density_fn=lambda p, m, t: model.transition_log_density_fn(
            theta, p, m
        ),
        process_noise_covariance=model.process_covariance,
        process_noise_covariance_provenance="model_exact",
        observation_fn=lambda p, t: model.observation_fn(p),
        observation_jacobian_fn=lambda p, t: model.observation_jacobian_fn(p),
        observation_covariance=model.observation_covariance,
        observation_log_density_fn=lambda p, obs, t: model.observation_log_density_fn(
            theta, p, obs
        ),
        initial_mean=tf.zeros([1], DTYPE),
        initial_covariance=tf.eye(1, dtype=DTYPE),
        initial_covariance_provenance="model_exact",
    )

    result = canonical_value_and_diagnostics(
        callbacks=callbacks,
        observations=tf.constant(observations[:, None], DTYPE),
        particle_count=PARTICLES,
        seed=seed,
        flow_substeps=FLOW_SUBSTEPS,
        temper_stages=1,
        annealed_resampling=False,
        flow_prior_cap=float("inf"),
        resample_seed=seed,
        epsilon=EPSILON,
        sinkhorn_steps=SINKHORN_STEPS,
        balance_steps=SINKHORN_STEPS,
        ridge=1.0e-5,
        dual_cap_enabled=True,
        trust_region_enabled=True,
        trust_region_lm_damping=1.0e-2,
        trust_region_lm_scale_floor=1.0e-4,
        trust_region_radius=0.5,
    )
    return {
        "value": float(result["value"].numpy()),
        "program_valid": bool(result["program_valid"].numpy()),
        "max_marginal_tv_error": float(
            result["max_marginal_tv_error"].numpy()
        ),
        "all_marginals_valid": bool(result["all_marginals_valid"].numpy()),
    }


def score_lane(observations, seed):
    """1D LGSSM score lane."""
    from bayesfilter.highdim.ledh_canonical_score_tf import (
        canonical_value_and_analytical_score,
    )
    from bayesfilter.highdim.ledh_diagonal_lgssm_any_dim import (
        diagonal_lgssm_any_dim,
    )

    theta = tf.constant(THETA, DTYPE)
    model, set_direction = diagonal_lgssm_any_dim(
        theta, dim=1, obs_matrix=tf.constant([[1.0]], DTYPE)
    )

    one_hot = np.zeros(3)  # [phi, sw, sv]
    one_hot[0] = 1.0  # derivative w.r.t. phi
    set_direction(tf.constant(one_hot, DTYPE))

    dim = 1
    horizon = int(observations.shape[0])
    rng = np.random.default_rng(9000 + seed)
    initial = tf.constant(
        np.zeros(dim)[None, :] + rng.standard_normal((PARTICLES, dim)), DTYPE
    )
    covs = tf.constant(np.stack([np.eye(dim)] * PARTICLES), DTYPE)
    noises = tf.constant(
        rng.standard_normal((horizon, PARTICLES, dim)), DTYPE
    )
    reset_design = tf.constant(
        np.tile(
            np.concatenate([np.eye(dim), -np.eye(dim)]),
            (PARTICLES // (2 * dim), 1),
        ),
        DTYPE,
    )

    value, score = canonical_value_and_analytical_score(
        model,
        theta,
        initial,
        covs,
        noises,
        tf.constant(observations[:, None], DTYPE),
        flow_substeps=FLOW_SUBSTEPS,
        with_score=True,
        reset_policy="contract_e",
        reset_design=reset_design,
        reset_epsilon=EPSILON,
        reset_sinkhorn_steps=SINKHORN_STEPS,
        reset_balance_steps=SINKHORN_STEPS,
        reset_ridge=1.0e-5,
        correction_steps=1,
        correction_lm_damping=1.0e-2,
        correction_lm_scale_floor=1.0e-4,
        correction_trust_radius=0.5,
        pairwise_steps=1,
        annealed_stages=1,
        annealed_seed=17,
    )
    return {
        "value": float(value.numpy()),
        "score": float(score[0].numpy()),
    }


def main():
    # Generate observations
    rng = np.random.default_rng(9000)
    phi, sw, sv = THETA
    obs_list = []
    for seed_offset in range(3):
        rng_rep = np.random.default_rng(9000 + seed_offset)
        x = 0.0
        observations = []
        for _ in range(HORIZON):
            x = phi * x + sw * rng_rep.standard_normal()
            y = x + sv * rng_rep.standard_normal()
            observations.append(y)
        obs_list.append(np.array(observations))

    rows = []
    for rep, obs in enumerate(obs_list):
        seed = SEEDS[rep]
        ref_value = exact_kalman_loglik(obs)
        ref_score = exact_kalman_score(obs, direction=0)

        v = value_lane(obs, seed)
        s = score_lane(obs, seed)

        row = {
            "replication": rep,
            "seed": seed,
            "exact_kalman_value": ref_value,
            "exact_kalman_score": ref_score,
            "value_lane_value": v["value"],
            "value_lane_error": v["value"] - ref_value,
            "value_lane_marginal_ok": v["all_marginals_valid"],
            "value_lane_max_tv": v["max_marginal_tv_error"],
            "score_lane_value": s["value"],
            "score_lane_error": s["value"] - ref_value,
            "score": s["score"],
            "score_error": s["score"] - ref_score,
        }
        rows.append(row)
        print(
            f"rep {rep}: "
            f"exact val={ref_value:10.4f} score={ref_score:+8.4f}  "
            f"value-lane {row['value_lane_value']:10.4f} (err {row['value_lane_error']:+.4f})  "
            f"score-lane {row['score_lane_value']:10.4f} (err {row['score_lane_error']:+.4f})  "
            f"score {row['score']:+8.4f} (err {row['score_error']:+.4f})"
        )

    output = {
        "study": "lgssm_1d_score_parity",
        "date": "2026-08-28",
        "model": "1D AR(1) LGSSM",
        "parameters": THETA.tolist(),
        "horizon": HORIZON,
        "controls": {
            "epsilon": EPSILON,
            "sinkhorn_steps": SINKHORN_STEPS,
            "flow_substeps": FLOW_SUBSTEPS,
            "particle_count": PARTICLES,
        },
        "rows": rows,
        "note": "1D AR(1) LGSSM with same controls as 3D dlgssm pilot.",
    }
    Path("docs/benchmarks/lgssm_1d_parity.json").write_text(
        json.dumps(output, indent=2)
    )
    print("\nsaved docs/benchmarks/lgssm_1d_parity.json")


if __name__ == "__main__":
    main()
