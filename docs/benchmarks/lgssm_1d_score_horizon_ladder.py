"""Does the 1D LGSSM score error mean grow with horizon at fixed N?

Companion to lgssm_1d_t3_score_variance_ladder.py, which held T=3 and varied N.
That run found the score error mean not separated from zero at 16 seeds, with
per-seed SD comparable to the score itself.  The precision there (SE about 0.07
on a score of 0.43) cannot see a bias of the size the historical T=10 d=2
N=3000 artifact recorded (0.6% relative, z about -2.9 to -6.5).

This run holds N and the observation path family fixed and walks the horizon,
so the question is whether the error mean moves systematically with T.  Nested
prefixes of one long path are used so successive rungs share data.
"""

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import sys
from pathlib import Path
import json
import time

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import tensorflow as tf

from bayesfilter.highdim.ledh_canonical_score_tf import (
    canonical_value_and_analytical_score,
)
from bayesfilter.highdim.ledh_diagonal_lgssm_any_dim import diagonal_lgssm_any_dim

DTYPE = tf.float64
THETA = np.array([0.9, 0.6, 0.8])
PARTICLES = 1008
HORIZON_LADDER = [3, 10, 25, 50]
SEEDS = [999000 + i for i in range(16)]
EPSILON = 1.0
SINKHORN_STEPS = 8
FLOW_SUBSTEPS = 12
DATA_SEED = 4242


def kalman_value(obs, theta):
    phi, sw, sv = theta
    Q, R = sw**2, sv**2
    m, P, loglik = 0.0, 1.0, 0.0
    for y in obs:
        m_pred = phi * m
        P_pred = phi**2 * P + Q
        S = P_pred + R
        v = y - m_pred
        loglik += -0.5 * (np.log(2.0 * np.pi * S) + v * v / S)
        K = P_pred / S
        m = m_pred + K * v
        P = P_pred - K * P_pred
    return loglik


def kalman_score_phi(obs, h=1e-5):
    plus, minus = THETA.copy(), THETA.copy()
    plus[0] += h
    minus[0] -= h
    return (kalman_value(obs, plus) - kalman_value(obs, minus)) / (2.0 * h)


def run_one(obs, seed):
    theta = tf.constant(THETA, DTYPE)
    model, set_direction = diagonal_lgssm_any_dim(
        theta, dim=1, obs_matrix=tf.constant([[1.0]], DTYPE)
    )
    one_hot = np.zeros(3)
    one_hot[0] = 1.0
    set_direction(tf.constant(one_hot, DTYPE))

    horizon = int(obs.shape[0])
    rng = np.random.default_rng(seed)
    initial = tf.constant(rng.standard_normal((PARTICLES, 1)), DTYPE)
    covs = tf.constant(np.stack([np.eye(1)] * PARTICLES), DTYPE)
    noises = tf.constant(rng.standard_normal((horizon, PARTICLES, 1)), DTYPE)
    reset_design = tf.constant(
        np.tile(np.array([[1.0], [-1.0]]), (PARTICLES // 2, 1)), DTYPE
    )

    value, score = canonical_value_and_analytical_score(
        model, theta, initial, covs, noises,
        tf.constant(obs[:, None], DTYPE),
        flow_substeps=FLOW_SUBSTEPS, with_score=True,
        reset_policy="contract_e", reset_design=reset_design,
        reset_epsilon=EPSILON, reset_sinkhorn_steps=SINKHORN_STEPS,
        reset_balance_steps=SINKHORN_STEPS, reset_ridge=1e-5,
        correction_steps=1, pairwise_steps=1,
        annealed_stages=1, annealed_seed=17,
    )
    return float(value.numpy()), float(score[0].numpy())


def main():
    t0 = time.time()

    rng = np.random.default_rng(DATA_SEED)
    phi, sw, sv = THETA
    x, path = 0.0, []
    for _ in range(max(HORIZON_LADDER)):
        x = phi * x + sw * rng.standard_normal()
        path.append(x + sv * rng.standard_normal())
    path = np.asarray(path)

    print("1D LGSSM score error vs horizon at fixed N")
    print(f"theta = {THETA.tolist()}   N = {PARTICLES}   seeds = {len(SEEDS)}")
    print(f"observation path seed = {DATA_SEED} (nested prefixes)")
    print()
    header = (
        f"{'T':>3}  {'exact score':>12}  {'mean error':>12}  {'SE':>9}  "
        f"{'z':>7}  {'per-seed SD':>11}  {'rel err':>9}  {'sep?':>5}"
    )
    print(header)
    print("-" * len(header))

    rungs = []
    for horizon in HORIZON_LADDER:
        obs = path[:horizon]
        exact_value = kalman_value(obs, THETA)
        exact_score = kalman_score_phi(obs)

        values, scores = [], []
        for seed in SEEDS:
            value, score = run_one(obs, seed)
            values.append(value)
            scores.append(score)

        value_errors = np.asarray(values) - exact_value
        score_errors = np.asarray(scores) - exact_score
        n = len(SEEDS)
        score_sd = float(np.std(scores, ddof=1))
        error_mean = float(np.mean(score_errors))
        error_se = float(np.std(score_errors, ddof=1) / np.sqrt(n))
        z = error_mean / error_se if error_se > 0.0 else float("nan")
        relative = error_mean / abs(exact_score) if exact_score != 0.0 else float("nan")
        separated = abs(z) > 2.0

        print(
            f"{horizon:3d}  {exact_score:12.6f}  {error_mean:+12.6f}  "
            f"{error_se:9.6f}  {z:+7.2f}  {score_sd:11.6f}  "
            f"{relative:+8.2%}  {'yes' if separated else 'no':>5}"
        )

        rungs.append({
            "horizon": horizon,
            "exact_kalman_value": exact_value,
            "exact_kalman_score_phi": exact_score,
            "seeds": SEEDS,
            "values": values,
            "scores": scores,
            "summary": {
                "n": n,
                "value_error_mean": float(np.mean(value_errors)),
                "value_error_se": float(
                    np.std(value_errors, ddof=1) / np.sqrt(n)
                ),
                "score_error_mean": error_mean,
                "score_error_se": error_se,
                "score_sd": score_sd,
                "z_score": z,
                "relative_error_mean": relative,
                "separated_from_zero_2se": bool(separated),
            },
        })

    elapsed = time.time() - t0
    print()
    print(f"elapsed {elapsed:.1f}s")

    result = {
        "study": "1D LGSSM analytical score error versus horizon at fixed N",
        "date": "2026-08-29",
        "plan": (
            "docs/plans/"
            "bayesfilter-lgssm-1d-score-variance-vs-bias-plan-2026-08-29.md"
        ),
        "companion_run": "docs/benchmarks/lgssm_1d_t3_score_variance_ladder.json",
        "model": {
            "family": "1D AR(1) linear Gaussian",
            "dim": 1,
            "theta": THETA.tolist(),
            "theta_names": ["phi", "sigma_w", "sigma_v"],
            "score_direction": "phi",
            "observation_path_seed": DATA_SEED,
            "observation_path": path.tolist(),
            "nested_prefixes": True,
        },
        "config": {
            "particles": PARTICLES,
            "reset_policy": "contract_e",
            "epsilon": EPSILON,
            "sinkhorn_steps": SINKHORN_STEPS,
            "balance_steps": SINKHORN_STEPS,
            "ridge": 1e-5,
            "flow_substeps": FLOW_SUBSTEPS,
            "correction_steps": 1,
            "pairwise_steps": 1,
            "annealed_stages": 1,
            "dtype": "float64",
            "device": "CPU (CUDA_VISIBLE_DEVICES=-1, GPU intentionally hidden)",
            "chunk_policy": "dense, K=N since N<=3000",
        },
        "horizon_ladder": HORIZON_LADDER,
        "seed_count": len(SEEDS),
        "rungs": rungs,
        "elapsed_seconds": elapsed,
        "interpretation_rules": [
            "Score error mean and SE are descriptive for these 16 seeds and this "
            "single observation path; they are not a bias theorem.",
            "A not-separated rung means the bias is below this run's noise floor, "
            "never that the score is unbiased.",
            "Only the sign and trend of the error mean across T is being read; no "
            "ranking of configurations is supported.",
        ],
        "non_claims": [
            "No unbiasedness, convergence, Kalman-equivalence, HMC-readiness, or "
            "default-readiness claim.",
            "Nothing here transfers to d>1, to other directions, to other "
            "observation paths, or to the GPU/TF32 route.",
            "This scope has no score-lane tuning artifact, so no per-model claim "
            "is made.",
        ],
    }

    out_path = Path(__file__).with_suffix(".json")
    with open(out_path, "w") as handle:
        json.dump(result, handle, indent=2)
    print(f"result {out_path}")


if __name__ == "__main__":
    main()
