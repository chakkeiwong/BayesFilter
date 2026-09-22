"""D=3 T=50 LGSSM score bias check across N (matching historical benchmark fixture).

The historical Contract-E LGSSM score gate (2026-07-02) tested d=2, T=10, N=3000
and found all three scores separated from Kalman at |z|=2.9-6.5 (negative bias)
but passed on a 1% relative-error criterion.

This run tests d=3, T=50 at the particle ladder to see if the bias persists
and whether it shrinks with N. Uses the exact fixture from the historical
benchmark: benchmark_lgssm_exact_oracle_m3_T50.
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
# Historical fixture: benchmark_lgssm_exact_oracle_m3_T50
# theta = [phi_1, phi_2, phi_3, q_scale, r_scale]
THETA = np.array([0.72, 0.55, 0.35, 0.35, 0.45])
DIM = 3
HORIZON = 50
DATA_SEED = 81100
PARTICLE_LADDER = [1008, 2016, 3000, 4032]
SEEDS = [999000 + i for i in range(16)]
EPSILON = 1.0
SINKHORN_STEPS = 8
FLOW_SUBSTEPS = 12


def kalman_value_score(obs, theta, direction=0):
    """Exact Kalman likelihood and score via central FD."""
    def kalman_ll(th):
        phi = th[:DIM]
        Q = np.diag([th[DIM]**2] * DIM)
        R = np.diag([th[DIM+1]**2] * DIM)
        m = np.zeros(DIM)
        P = np.eye(DIM)
        loglik = 0.0
        for y in obs:
            m_pred = phi * m
            P_pred = np.diag(phi**2 * np.diag(P)) + Q
            S = P_pred + R
            K = P_pred @ np.linalg.inv(S)
            v = y - m_pred
            sign, logdet = np.linalg.slogdet(S)
            loglik += -0.5 * (DIM * np.log(2.0 * np.pi) + logdet + v @ np.linalg.inv(S) @ v)
            m = m_pred + K @ v
            P = P_pred - K @ P_pred
        return loglik

    h = 1e-5
    plus, minus = theta.copy(), theta.copy()
    plus[direction] += h
    minus[direction] -= h
    val = kalman_ll(theta)
    score = (kalman_ll(plus) - kalman_ll(minus)) / (2.0 * h)
    return val, score


def run_one(obs, particles, seed, direction=0):
    """Return (value, score) for one configuration."""
    theta = tf.constant(THETA, DTYPE)
    obs_matrix = tf.eye(DIM, dtype=DTYPE)
    model, set_direction = diagonal_lgssm_any_dim(theta, dim=DIM, obs_matrix=obs_matrix)

    one_hot = np.zeros(5)
    one_hot[direction] = 1.0
    set_direction(tf.constant(one_hot, DTYPE))

    rng = np.random.default_rng(seed)
    initial = tf.constant(rng.standard_normal((particles, DIM)), DTYPE)
    covs = tf.constant(np.stack([np.eye(DIM)] * particles), DTYPE)
    noises = tf.constant(rng.standard_normal((HORIZON, particles, DIM)), DTYPE)

    # Replicated cubature design for d=3: 2d=6 basis rows, repeated to fill N
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

    value, score = canonical_value_and_analytical_score(
        model, theta, initial, covs, noises,
        tf.constant(obs, DTYPE),
        flow_substeps=FLOW_SUBSTEPS, with_score=True,
        reset_policy="contract_e", reset_design=reset_design,
        reset_epsilon=EPSILON, reset_sinkhorn_steps=SINKHORN_STEPS,
        reset_balance_steps=SINKHORN_STEPS, reset_ridge=1e-5,
        correction_steps=1, pairwise_steps=1,
        annealed_stages=1, annealed_seed=17,
    )
    return float(value.numpy()), float(score[direction].numpy())


def main():
    t0 = time.time()

    # Generate observation path
    rng = np.random.default_rng(DATA_SEED)
    phi = THETA[:DIM]
    sw, sv = THETA[DIM], THETA[DIM+1]
    x = np.zeros(DIM)
    obs = []
    for _ in range(HORIZON):
        x = phi * x + sw * rng.standard_normal(DIM)
        obs.append(x + sv * rng.standard_normal(DIM))
    obs = np.array(obs)

    exact_value, exact_score = kalman_value_score(obs, THETA, direction=0)

    print(f"D=3 T=50 LGSSM score error across N")
    print(f"theta = {THETA.tolist()}")
    print(f"exact value = {exact_value:.12f}")
    print(f"exact score (phi_1) = {exact_score:.12f}")
    print(f"seeds: {len(SEEDS)} paired across all N")
    print()

    header = (
        f"{'N':>5}  {'value err':>11}  {'val SE':>9}  "
        f"{'score err':>11}  {'sc SE':>9}  {'z':>7}  {'score SD':>9}  "
        f"{'rel err':>9}  {'sep?':>5}"
    )
    print(header)
    print("-" * len(header))

    rungs = []
    for N in PARTICLE_LADDER:
        print(f"{N:5d}  ", end="", flush=True)
        vals, scores = [], []
        for seed in SEEDS:
            val, sc = run_one(obs, N, seed, direction=0)
            vals.append(val)
            scores.append(sc)

        val_errs = np.array(vals) - exact_value
        score_errs = np.array(scores) - exact_score

        n = len(SEEDS)
        val_err_mean = float(np.mean(val_errs))
        val_err_se = float(np.std(val_errs, ddof=1) / np.sqrt(n))
        sc_err_mean = float(np.mean(score_errs))
        sc_err_se = float(np.std(score_errs, ddof=1) / np.sqrt(n))
        score_sd = float(np.std(scores, ddof=1))
        z = sc_err_mean / sc_err_se if sc_err_se > 0.0 else float("nan")
        relative = sc_err_mean / abs(exact_score) if exact_score != 0.0 else float("nan")
        separated = abs(z) > 2.0

        print(
            f"{val_err_mean:+11.6f}  {val_err_se:9.6f}  "
            f"{sc_err_mean:+11.6f}  {sc_err_se:9.6f}  {z:+7.2f}  "
            f"{score_sd:9.6f}  {relative:+8.2%}  {'yes' if separated else 'no':>5}"
        )

        rungs.append({
            "particles": N,
            "seeds": SEEDS,
            "values": vals,
            "scores": scores,
            "summary": {
                "n": n,
                "value_error_mean": val_err_mean,
                "value_error_se": val_err_se,
                "score_error_mean": sc_err_mean,
                "score_error_se": sc_err_se,
                "score_sd": score_sd,
                "z_score": z,
                "relative_error": relative,
                "separated_from_zero_2se": bool(separated),
            },
        })

    elapsed = time.time() - t0

    result = {
        "study": "D=3 T=50 LGSSM analytical score error across N",
        "date": "2026-08-29",
        "fixture": "benchmark_lgssm_exact_oracle_m3_T50 (historical comparison)",
        "model": {
            "family": "diagonal AR(1) linear Gaussian",
            "dim": DIM,
            "horizon": HORIZON,
            "theta": THETA.tolist(),
            "theta_names": ["phi_1", "phi_2", "phi_3", "q_scale", "r_scale"],
            "score_direction": "phi_1",
            "observation_path_seed": DATA_SEED,
            "observations": obs.tolist(),
        },
        "exact_kalman_value": exact_value,
        "exact_kalman_score_phi1": exact_score,
        "config": {
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
            "device": "CPU",
        },
        "particle_ladder": PARTICLE_LADDER,
        "seed_count": len(SEEDS),
        "rungs": rungs,
        "elapsed_seconds": elapsed,
        "comparison_to_historical": (
            "The historical 2026-07-02 N=3000 gate tested d=2 T=10 and found "
            "|z|=2.9-6.5 on three parameters but passed on 1% relative error. "
            "This run tests d=3 T=50 to see if higher dimension and longer "
            "horizon show similar or worse bias."
        ),
    }

    out_path = Path(__file__).with_suffix(".json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print()
    print(f"Elapsed: {elapsed:.1f}s")
    print(f"Result: {out_path}")


if __name__ == "__main__":
    main()
