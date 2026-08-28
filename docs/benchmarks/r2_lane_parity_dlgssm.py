"""Value-lane vs score-lane parity against exact Kalman (dlgssm).

Question: under the ε=1.0 / sk=8 / flow=12 / N=1008 controls selected by
the R2-TUNE pilot, do the value lane and the score lane estimate the same
log-likelihood, and does the analytical score match the exact Kalman
score?

Why multi-seed. The value lane draws its particle cloud internally from
`seed` and exposes no injectable-cloud argument, so the two lanes cannot
be handed one common cloud. A single draw from each therefore cannot
separate a lane difference from Monte Carlo noise. Both lanes are run
over the same seed set and compared as sampling distributions.

Comparators, both exact for this model:
  - log-likelihood: the Kalman recursion.
  - score: central differences on the Kalman log-likelihood in the
    parameter direction. The model is linear Gaussian, so this is the
    exact derivative up to the difference truncation, not a proxy. The
    step is checked by Richardson comparison at h and h/2.

Not concluded: nothing here ranks the lanes or promotes a configuration.
This measures agreement of two estimators of one estimand, and the
accuracy of one analytical derivative, on one model at one control cell.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_q3_leaderboard_20260824 import (  # noqa: E402
    _density_obs_log,
    _gaussian_trans_log,
)

DTYPE = tf.float64

LGSSM_OBS_MATRIX = np.array(
    [[1.0, 0.25, -0.15], [0.2, 1.1, 0.3], [-0.1, 0.35, 0.9]]
)
LGSSM_THETA = np.array([0.9, 0.8, 0.7, 0.6, 0.8])

# Control cell selected by the R2-TUNE dlgssm pilot.
EPSILON = 1.0
SINKHORN_STEPS = 8
FLOW_SUBSTEPS = 12
PARTICLES = 1008
SCORE_DIRECTION = 0  # phi_1


def exact_kalman_loglik(observations, theta=LGSSM_THETA):
    """Exact log-likelihood of the diagonal LGSSM (NumPy reference)."""
    phi = np.diag(theta[:3])
    q = theta[3] ** 2 * np.eye(3)
    r = theta[4] ** 2 * np.eye(3)
    h = LGSSM_OBS_MATRIX
    mean, cov = np.zeros(3), np.eye(3)
    total = 0.0
    for t in range(int(observations.shape[0])):
        mean = phi @ mean
        cov = phi @ cov @ phi.T + q
        s = h @ cov @ h.T + r
        resid = observations[t] - h @ mean
        _sign, logdet = np.linalg.slogdet(2.0 * np.pi * s)
        total += -0.5 * (resid @ np.linalg.solve(s, resid) + logdet)
        gain = cov @ h.T @ np.linalg.inv(s)
        mean = mean + gain @ resid
        cov = (np.eye(3) - gain @ h) @ cov
    return float(total)


def exact_kalman_score(observations, direction=SCORE_DIRECTION, h=1.0e-5):
    """Exact Kalman score by central differences, with a step check.

    Returns the derivative at step h and at h/2. Agreement between them
    is the evidence that the difference truncation is small enough to
    use this as a score reference; a large gap would mean the step is
    in the roundoff or truncation regime and the reference is unusable.
    """
    def at(step):
        e = np.zeros_like(LGSSM_THETA)
        e[direction] = step
        plus = exact_kalman_loglik(observations, LGSSM_THETA + e)
        minus = exact_kalman_loglik(observations, LGSSM_THETA - e)
        return (plus - minus) / (2.0 * step)

    return at(h), at(h / 2.0)


def value_lane(observations, seed):
    from bayesfilter.highdim.ledh_canonical_models_tf import (
        diagonal_lgssm_canonical_model,
    )
    from bayesfilter.highdim.ledh_canonical_filter_tf import (
        canonical_value_and_diagnostics,
        CanonicalModelCallbacks,
    )

    theta0 = tf.constant(LGSSM_THETA, DTYPE)
    model, _ = diagonal_lgssm_canonical_model(theta0)
    dim = 3
    callbacks = CanonicalModelCallbacks(
        model_id="dlgssm_parity",
        state_dim=dim,
        observation_dim=dim,
        transition_mean_fn=lambda p, t: model.transition_mean_fn(theta0, p),
        transition_log_density_fn=_gaussian_trans_log(model, theta0),
        process_noise_covariance=model.process_covariance,
        process_noise_covariance_provenance="model_exact",
        observation_fn=lambda p, t: model.observation_fn(p),
        observation_jacobian_fn=lambda p, t: model.observation_jacobian_fn(p),
        observation_covariance=model.observation_covariance,
        observation_log_density_fn=_density_obs_log(model, theta0),
        initial_mean=tf.zeros([dim], DTYPE),
        initial_covariance=tf.eye(dim, dtype=DTYPE),
        initial_covariance_provenance="model_exact",
    )
    result = canonical_value_and_diagnostics(
        callbacks=callbacks,
        observations=tf.constant(observations, DTYPE),
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
        "marginal_steps_completed": int(
            result["marginal_steps_completed"].numpy()
        ),
    }


def score_lane(observations, seed):
    """Score lane at the same controls.

    Parameter-name map to the value lane (verified by reading both
    signatures; the two lanes spell the same Class-C family differently):
      dual_cap_enabled=True        -> correction_steps > 0
      trust_region_enabled=True    -> pairwise_steps > 0
      trust_region_lm_damping      -> correction_lm_damping
      trust_region_lm_scale_floor  -> correction_lm_scale_floor
      trust_region_radius          -> correction_trust_radius
    """
    from bayesfilter.highdim.ledh_canonical_models_tf import (
        diagonal_lgssm_canonical_model,
    )
    from bayesfilter.highdim.ledh_canonical_score_tf import (
        canonical_value_and_analytical_score,
    )

    theta = tf.constant(LGSSM_THETA, DTYPE)
    model, set_direction = diagonal_lgssm_canonical_model(theta)
    dim = 3
    horizon = int(observations.shape[0])

    one_hot = np.zeros(LGSSM_THETA.shape[0])
    one_hot[SCORE_DIRECTION] = 1.0
    set_direction(tf.constant(one_hot, DTYPE))

    # Same cloud construction the Q3 claim lane uses.
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
        tf.constant(observations, DTYPE),
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


def _summary(errors):
    arr = np.array([e for e in errors if e is not None and np.isfinite(e)])
    if arr.size == 0:
        return {"n": 0, "mean": None, "se": None}
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "se": (
            float(arr.std(ddof=1) / np.sqrt(arr.size))
            if arr.size > 1
            else None
        ),
    }


def main():
    seeds = [999000 + i for i in range(6)]
    replication = 0

    data = np.load(
        "docs/benchmarks/r2_tuning_20260827/dlgssm_tuning_data.npz"
    )
    obs = data["observations"][replication]

    ref_value = exact_kalman_loglik(obs)
    score_h, score_h2 = exact_kalman_score(obs)
    score_step_gap = abs(score_h - score_h2)

    print(f"exact Kalman log-likelihood: {ref_value:.6f}")
    print(f"exact Kalman score (h=1e-5):  {score_h:.6f}")
    print(f"exact Kalman score (h=5e-6):  {score_h2:.6f}")
    print(f"  step-refinement gap:        {score_step_gap:.3e}")
    print(f"\ncontrols: eps={EPSILON} sk={SINKHORN_STEPS} "
          f"flow={FLOW_SUBSTEPS} N={PARTICLES}, "
          f"score direction={SCORE_DIRECTION} (phi_1)\n")

    rows = []
    t0 = time.time()
    for seed in seeds:
        v = value_lane(obs, seed)
        s = score_lane(obs, seed)
        row = {
            "seed": seed,
            "value_lane_value": v["value"],
            "value_lane_error": v["value"] - ref_value,
            "value_lane_marginal_ok": v["all_marginals_valid"],
            "value_lane_max_tv": v["max_marginal_tv_error"],
            "score_lane_value": s["value"],
            "score_lane_error": s["value"] - ref_value,
            "score": s["score"],
            "score_error": s["score"] - score_h,
        }
        rows.append(row)
        print(
            f"seed {seed}: "
            f"value lane {row['value_lane_value']:10.4f} "
            f"(err {row['value_lane_error']:+.4f})  "
            f"score lane {row['score_lane_value']:10.4f} "
            f"(err {row['score_lane_error']:+.4f})  "
            f"score {row['score']:9.4f} "
            f"(err {row['score_error']:+.4f})"
        )
    wall = time.time() - t0

    value_lane_err = _summary([r["value_lane_error"] for r in rows])
    score_lane_err = _summary([r["score_lane_error"] for r in rows])
    lane_gap = _summary(
        [r["score_lane_value"] - r["value_lane_value"] for r in rows]
    )
    score_err = _summary([r["score_error"] for r in rows])

    print("\n--- log-likelihood, signed error vs exact Kalman ---")
    for name, s in (
        ("value lane", value_lane_err),
        ("score lane", score_lane_err),
    ):
        se = f"{s['se']:.4f}" if s["se"] is not None else "n/a"
        print(f"{name}: mean {s['mean']:+.4f}  SE {se}  n={s['n']}")
    se = f"{lane_gap['se']:.4f}" if lane_gap["se"] is not None else "n/a"
    print(
        f"paired gap (score - value): mean {lane_gap['mean']:+.4f}  "
        f"SE {se}"
    )

    print("\n--- score, signed error vs exact Kalman score ---")
    se = f"{score_err['se']:.4f}" if score_err["se"] is not None else "n/a"
    print(
        f"analytical score: mean {score_err['mean']:+.4f}  SE {se}  "
        f"n={score_err['n']}"
    )
    rel = (
        abs(score_err["mean"]) / abs(score_h) if score_h != 0.0 else None
    )
    if rel is not None:
        print(f"relative to |exact score| {abs(score_h):.4f}: {rel:.2%}")

    # A paired gap is separated from zero only if its magnitude exceeds
    # roughly two of its own standard errors. State that verdict rather
    # than letting the reader infer it from a mean alone.
    if lane_gap["se"]:
        separated = abs(lane_gap["mean"]) > 2.0 * lane_gap["se"]
    else:
        separated = None
    print(
        f"\nlane gap separated from zero at ~2 SE: {separated}"
    )

    out = {
        "study": "r2_lane_parity_dlgssm",
        "date": "2026-08-28",
        "question": (
            "do the value and score lanes estimate the same "
            "log-likelihood at the selected controls, and does the "
            "analytical score match the exact Kalman score"
        ),
        "controls": {
            "epsilon": EPSILON,
            "sinkhorn_steps": SINKHORN_STEPS,
            "balance_steps": SINKHORN_STEPS,
            "flow_substeps": FLOW_SUBSTEPS,
            "particle_count": PARTICLES,
            "score_direction": SCORE_DIRECTION,
            "provenance": "selected by R2-TUNE dlgssm pilot",
        },
        "tuning_data": {
            "path": "docs/benchmarks/r2_tuning_20260827/dlgssm_tuning_data.npz",
            "sha256": str(data["sha256"]),
            "replication": replication,
        },
        "exact_reference": {
            "log_likelihood": ref_value,
            "score_central_diff_h1e5": score_h,
            "score_central_diff_h5e6": score_h2,
            "score_step_refinement_gap": score_step_gap,
            "basis": (
                "model is linear Gaussian; Kalman recursion is the exact "
                "likelihood and central differences on it give the exact "
                "score up to difference truncation"
            ),
        },
        "seeds": seeds,
        "rows": rows,
        "summary": {
            "value_lane_error": value_lane_err,
            "score_lane_error": score_lane_err,
            "paired_lane_gap": lane_gap,
            "score_error": score_err,
            "lane_gap_separated_from_zero_2se": separated,
        },
        "manifest": {
            "git_commit": subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
            ).stdout.strip(),
            "conda_env": os.environ.get("CONDA_DEFAULT_ENV", "unknown"),
            "cuda_device_order": os.environ.get(
                "CUDA_DEVICE_ORDER", "unset"
            ),
            "cuda_visible_devices": os.environ.get(
                "CUDA_VISIBLE_DEVICES", "unset"
            ),
            "gpu": [
                tf.config.experimental.get_device_details(g).get(
                    "device_name", "unknown"
                )
                for g in tf.config.list_physical_devices("GPU")
            ],
            "wall_time_sec": wall,
        },
        "non_claims": [
            "Neither lane is claimed more accurate than the other; the "
            "two lanes estimate one estimand and are compared for "
            "agreement, not ranked.",
            "The lanes cannot be given a common particle cloud (the "
            "value lane samples internally from `seed`), so per-seed "
            "values are not paired realizations and only the "
            "distributions are comparable.",
            "One model, one control cell, one horizon. Nothing here "
            "transfers to another model, route, or horizon.",
        ],
    }
    Path("docs/benchmarks/r2_lane_parity_dlgssm.json").write_text(
        json.dumps(out, indent=2)
    )
    print("\nsaved docs/benchmarks/r2_lane_parity_dlgssm.json")


if __name__ == "__main__":
    main()
