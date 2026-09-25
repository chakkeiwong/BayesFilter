"""1D LGSSM score error vs horizon length.

Question: does the score error exist at T=2,3,4, or only emerge at longer
horizons? This distinguishes:
  - Per-step error that accumulates with T
  - Single-component error present from step 1
"""

import numpy as np
import sys
import tensorflow as tf
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from bayesfilter.highdim.ledh_canonical_score_tf import (
    canonical_value_and_analytical_score,
)
from bayesfilter.highdim.ledh_diagonal_lgssm_any_dim import (
    diagonal_lgssm_any_dim,
)

DTYPE = tf.float64
THETA = np.array([0.9, 0.6, 0.8])  # [phi, sw, sv]
SEED = 999000
PARTICLES = 1008
EPSILON = 1.0
SINKHORN_STEPS = 8
FLOW_SUBSTEPS = 12


def exact_kalman(obs, theta=THETA):
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
    e = np.zeros_like(THETA)
    e[dir] = h
    return (exact_kalman(obs, THETA + e) - exact_kalman(obs, THETA - e)) / (2 * h)


def score_lane(obs, seed):
    theta = tf.constant(THETA, DTYPE)
    model, set_direction = diagonal_lgssm_any_dim(
        theta, dim=1, obs_matrix=tf.constant([[1.0]], DTYPE)
    )
    one_hot = np.zeros(3)
    one_hot[0] = 1.0
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


# Generate one long sequence
rng = np.random.default_rng(9000)
phi, sw, sv = THETA
x, obs_long = 0.0, []
for _ in range(10):
    x = phi * x + sw * rng.standard_normal()
    obs_long.append(x + sv * rng.standard_normal())
obs_long = np.array(obs_long)

print(f"1D LGSSM score error vs horizon (N={PARTICLES}, eps={EPSILON}, sk={SINKHORN_STEPS})")
print(f"params: phi={THETA[0]}, sw={THETA[1]}, sv={THETA[2]}\n")
print("T   Exact Value  Exact Score  Analytical Score  Score Error  Rel Error")
print("─" * 75)

for T in [2, 3, 4, 5, 10]:
    obs = obs_long[:T]
    ref_val = exact_kalman(obs)
    ref_score = exact_score(obs, dir=0)
    val, sc = score_lane(obs, SEED)
    err = sc - ref_score
    rel = (err / ref_score * 100) if ref_score != 0 else 0
    print(f"{T:2d}  {ref_val:11.4f}  {ref_score:+11.4f}  {sc:+16.4f}  {err:+11.4f}  {rel:+8.2f}%")
