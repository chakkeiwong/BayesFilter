"""Test whether N=3000 reduces the 1D T=3 LGSSM score bias (historical N=3000 passed)."""

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import tensorflow as tf

from bayesfilter.highdim.ledh_canonical_score_tf import (
    canonical_value_and_analytical_score,
)
from bayesfilter.highdim.ledh_diagonal_lgssm_any_dim import diagonal_lgssm_any_dim

DTYPE = tf.float64
PARTICLES = 3000
HORIZON = 3
SEED = 9000
THETA = np.array([0.7, 0.3, 0.4])  # phi, sw, sv
EPSILON = 1.0
SINKHORN_STEPS = 8
FLOW_SUBSTEPS = 12

def exact_kalman(obs):
    phi, sw, sv = THETA
    Q, R = sw**2, sv**2
    m, P = 0.0, 1.0
    loglik = 0.0
    for y in obs:
        m_pred = phi * m
        P_pred = phi**2 * P + Q
        S = P_pred + R
        K = P_pred / S
        v = y - m_pred
        loglik += -0.5 * (np.log(2 * np.pi * S) + v**2 / S)
        m = m_pred + K * v
        P = P_pred - K * P_pred
    return loglik

def exact_score(obs, dir=0):
    eps = 1e-5
    theta_plus = THETA.copy()
    theta_plus[dir] += eps
    theta_minus = THETA.copy()
    theta_minus[dir] -= eps

    old_theta = THETA.copy()
    THETA[:] = theta_plus
    L_plus = exact_kalman(obs)
    THETA[:] = theta_minus
    L_minus = exact_kalman(obs)
    THETA[:] = old_theta

    return (L_plus - L_minus) / (2 * eps)

def score_lane(obs, seed):
    theta = tf.constant(THETA, DTYPE)
    model, set_direction = diagonal_lgssm_any_dim(
        theta, dim=1, obs_matrix=tf.constant([[1.0]], DTYPE)
    )
    one_hot = np.zeros(3)
    one_hot[0] = 1.0
    set_direction(tf.constant(one_hot, DTYPE))

    rng = np.random.default_rng(seed)
    initial = tf.constant(rng.standard_normal((PARTICLES, 1)), DTYPE)
    covs = tf.constant(np.stack([np.eye(1)] * PARTICLES), DTYPE)
    noises = tf.constant(rng.standard_normal((HORIZON, PARTICLES, 1)), DTYPE)
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


# Generate observation sequence
rng = np.random.default_rng(9000)
phi, sw, sv = THETA
x, obs = 0.0, []
for _ in range(HORIZON):
    x = phi * x + sw * rng.standard_normal()
    obs.append(x + sv * rng.standard_normal())
obs = np.array(obs)

ref_val = exact_kalman(obs)
ref_score = exact_score(obs, dir=0)

print(f"1D LGSSM T=3 at N={PARTICLES} (historical config)")
print(f"params: phi={THETA[0]}, sw={THETA[1]}, sv={THETA[2]}")
print(f"eps={EPSILON}, sinkhorn={SINKHORN_STEPS}, flow={FLOW_SUBSTEPS}\n")

print("Seed     Value Error    Exact Score  Analytical   Score Error  Rel Error")
print("─" * 78)

seeds = [9000 + i for i in range(10)]
val_errs, score_errs = [], []

for s in seeds:
    val, sc = score_lane(obs, s)
    val_err = val - ref_val
    sc_err = sc - ref_score
    rel = (sc_err / ref_score * 100) if ref_score != 0 else 0
    val_errs.append(val_err)
    score_errs.append(sc_err)
    print(f"{s:6d}  {val_err:+11.6f}  {ref_score:+11.6f}  {sc:+11.6f}  {sc_err:+11.6f}  {rel:+8.3f}%")

print("─" * 78)
mean_val_err = np.mean(val_errs)
mean_sc_err = np.mean(score_errs)
se_sc_err = np.std(score_errs, ddof=1) / np.sqrt(len(score_errs))
print(f"Mean   {mean_val_err:+11.6f}              {ref_score:+11.6f}             {mean_sc_err:+11.6f}  {mean_sc_err/ref_score*100:+8.3f}%")
print(f"SE                                                         {se_sc_err:11.6f}")
print(f"z-stat = {mean_sc_err / se_sc_err:.3f}")
