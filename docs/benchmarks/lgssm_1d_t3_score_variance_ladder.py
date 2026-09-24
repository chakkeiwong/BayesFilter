"""Paired N-ladder: is the 1D T=3 LGSSM score error bias or variance?

Plan: docs/plans/bayesfilter-lgssm-1d-score-variance-vs-bias-plan-2026-08-29.md
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
THETA = np.array([0.9, 0.6, 0.8])  # ablation fixture
HORIZON = 3
OBS = np.array([-0.9911206904960701, -2.7246862541942303, -0.30490733274540416])
EXACT_VALUE = -5.903444873434653
EPSILON = 1.0
SINKHORN_STEPS = 8
FLOW_SUBSTEPS = 12
PARTICLE_LADDER = [504, 1008, 2016, 3000]
SEEDS = [999000 + i for i in range(16)]


def exact_kalman_score(h=1e-5):
    """Central FD of the exact Kalman likelihood in phi direction."""
    def kalman_value(theta):
        phi, sw, sv = theta
        Q, R = sw**2, sv**2
        m, P, loglik = 0.0, 1.0, 0.0
        for y in OBS:
            m_pred = phi * m
            P_pred = phi**2 * P + Q
            S = P_pred + R
            K = P_pred / S
            v = y - m_pred
            loglik += -0.5 * (np.log(2 * np.pi * S) + v**2 / S)
            m = m_pred + K * v
            P = P_pred - K * P_pred
        return loglik

    theta_plus = THETA.copy()
    theta_plus[0] += h
    theta_minus = THETA.copy()
    theta_minus[0] -= h
    return (kalman_value(theta_plus) - kalman_value(theta_minus)) / (2 * h)


def run_one(particles, seed):
    """Return (value, score) for one configuration."""
    theta = tf.constant(THETA, DTYPE)
    model, set_direction = diagonal_lgssm_any_dim(
        theta, dim=1, obs_matrix=tf.constant([[1.0]], DTYPE)
    )
    one_hot = np.zeros(3)
    one_hot[0] = 1.0
    set_direction(tf.constant(one_hot, DTYPE))

    rng = np.random.default_rng(seed)
    initial = tf.constant(rng.standard_normal((particles, 1)), DTYPE)
    covs = tf.constant(np.stack([np.eye(1)] * particles), DTYPE)
    noises = tf.constant(rng.standard_normal((HORIZON, particles, 1)), DTYPE)
    reset_design = tf.constant(
        np.tile(np.array([[1.0], [-1.0]]), (particles // 2, 1)), DTYPE
    )

    value, score = canonical_value_and_analytical_score(
        model, theta, initial, covs, noises,
        tf.constant(OBS[:, None], DTYPE),
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
    exact_score = exact_kalman_score()

    print(f"1D T=3 LGSSM paired score variance vs bias ladder")
    print(f"theta = {THETA.tolist()}")
    print(f"exact value = {EXACT_VALUE:.12f}")
    print(f"exact score (phi) = {exact_score:.12f}")
    print(f"seeds: {len(SEEDS)} paired across all N")
    print()

    rungs = []
    for N in PARTICLE_LADDER:
        print(f"N={N}...")
        vals, scores = [], []
        for seed in SEEDS:
            val, sc = run_one(N, seed)
            vals.append(val)
            scores.append(sc)

        val_errs = np.array(vals) - EXACT_VALUE
        score_errs = np.array(scores) - exact_score

        rung = {
            "particles": N,
            "seeds": SEEDS,
            "values": vals,
            "scores": scores,
            "value_errors": val_errs.tolist(),
            "score_errors": score_errs.tolist(),
            "summary": {
                "n": len(SEEDS),
                "value_mean": float(np.mean(vals)),
                "value_error_mean": float(np.mean(val_errs)),
                "value_error_se": float(np.std(val_errs, ddof=1) / np.sqrt(len(val_errs))),
                "score_mean": float(np.mean(scores)),
                "score_error_mean": float(np.mean(score_errs)),
                "score_error_se": float(np.std(score_errs, ddof=1) / np.sqrt(len(score_errs))),
                "score_sd": float(np.std(scores, ddof=1)),
                "z_score": float(np.mean(score_errs) / (np.std(score_errs, ddof=1) / np.sqrt(len(score_errs)))),
                "score_sd_to_exact_ratio": float(np.std(scores, ddof=1) / abs(exact_score)),
            }
        }
        rungs.append(rung)

        s = rung["summary"]
        separated = abs(s["z_score"]) > 2.0
        print(f"  value error: {s['value_error_mean']:+.6f} ± {s['value_error_se']:.6f}")
        print(f"  score error: {s['score_error_mean']:+.6f} ± {s['score_error_se']:.6f}")
        print(f"  score SD: {s['score_sd']:.6f} (SD/|exact| = {s['score_sd_to_exact_ratio']:.3f})")
        print(f"  z = {s['z_score']:.3f} ({'separated' if separated else 'not separated'})")
        print()

    elapsed = time.time() - t0

    result = {
        "study": "1D T=3 LGSSM score variance vs bias ladder",
        "date": "2026-08-29",
        "plan": "docs/plans/bayesfilter-lgssm-1d-score-variance-vs-bias-plan-2026-08-29.md",
        "model": {
            "family": "1D AR(1) linear Gaussian",
            "theta": THETA.tolist(),
            "horizon": HORIZON,
            "observations": OBS.tolist(),
        },
        "exact_kalman_value": EXACT_VALUE,
        "exact_kalman_score_phi": exact_score,
        "config": {
            "epsilon": EPSILON,
            "sinkhorn_steps": SINKHORN_STEPS,
            "flow_substeps": FLOW_SUBSTEPS,
            "correction_steps": 1,
            "pairwise_steps": 1,
            "annealed_stages": 1,
            "dtype": "float64",
            "device": "CPU",
        },
        "particle_ladder": PARTICLE_LADDER,
        "seed_count": len(SEEDS),
        "rungs": rungs,
        "elapsed_seconds": elapsed,
    }

    out_path = Path(__file__).with_suffix(".json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Elapsed: {elapsed:.1f}s")
    print(f"Result: {out_path}")


if __name__ == "__main__":
    main()
