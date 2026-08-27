"""Part 2: production-score-program bias + value entropic-epsilon ladder."""
import os, sys
sys.path.insert(0, "."); sys.path.insert(0, "tests/highdim"); sys.path.insert(0, "docs/benchmarks")
import numpy as np, tensorflow as tf
gpus = tf.config.list_physical_devices("GPU")
for g in gpus: tf.config.experimental.set_memory_growth(g, True)
DTYPE = tf.float64
from bayesfilter.highdim.ledh_canonical_models_tf import diagonal_lgssm_canonical_model
from bayesfilter.highdim.ledh_canonical_neutra_targets_tf import _lgssm_frozen_observations
from bayesfilter.highdim.ledh_canonical_score_tf import canonical_value_and_analytical_score
from bayesfilter.highdim.ledh_canonical_filter_tf import canonical_value_and_diagnostics
from run_q3_leaderboard_20260824 import make_callbacks

theta0_np = [0.9, 0.8, 0.7, 0.6, 0.8]
theta0 = tf.constant(theta0_np, DTYPE)
obs_all = tf.cast(_lgssm_frozen_observations(), DTYPE)
obs_matrix = np.array([[1.0,0.25,-0.15],[0.2,1.1,0.3],[-0.1,0.35,0.9]])
def exact_kalman(theta_np, T):
    phi = np.diag(theta_np[:3]); q = theta_np[3]**2*np.eye(3); r = theta_np[4]**2*np.eye(3)
    mean, cov = np.zeros(3), np.eye(3); total = 0.0
    onp = obs_all.numpy()
    for t in range(T):
        mean = phi @ mean; cov = phi @ cov @ phi.T + q
        s = obs_matrix @ cov @ obs_matrix.T + r
        resid = onp[t] - obs_matrix @ mean
        sign, logdet = np.linalg.slogdet(2*np.pi*s)
        total += -0.5*(resid @ np.linalg.solve(s, resid) + logdet)
        gain = cov @ obs_matrix.T @ np.linalg.inv(s)
        mean = mean + gain @ resid; cov = (np.eye(3) - gain @ obs_matrix) @ cov
    return float(total)
def exact_score(T, idx=0, h=1e-6):
    up = np.array(theta0_np); up[idx] += h
    dn = np.array(theta0_np); dn[idx] -= h
    return (exact_kalman(up, T) - exact_kalman(dn, T)) / (2*h)

model, set_direction = diagonal_lgssm_canonical_model(theta0)
set_direction(tf.constant([1.0,0,0,0,0], DTYPE))
n = 1008
base = np.concatenate([np.eye(3), -np.eye(3)], axis=0)
design = tf.constant(np.tile(base, (n // 6, 1)), DTYPE)

print("== score bias vs T: PRODUCTION program (contract_e reset, smooth transport) ==", flush=True)
for T in (10, 25, 50):
    obs = obs_all[:T]
    sc = []
    for seed in range(6):
        rng = np.random.default_rng(9000+seed)
        initial = tf.constant(rng.standard_normal((n,3)), DTYPE)
        covs = tf.constant(np.stack([np.eye(3)]*n), DTYPE)
        noises = tf.constant(rng.standard_normal((T,n,3)), DTYPE)
        _v, s = canonical_value_and_analytical_score(
            model, theta0, initial, covs, noises, obs, flow_substeps=8,
            with_score=True, reset_policy="contract_e", reset_design=design,
            reset_sinkhorn_steps=8, reset_balance_steps=8)
        sc.append(float(s[0].numpy()))
    sc = np.array(sc); ex = exact_score(T)
    print(f"[T={T}] mean={sc.mean():.3f} exact={ex:.3f} bias={sc.mean()-ex:+.3f} SE={sc.std(ddof=1)/np.sqrt(len(sc)):.3f}", flush=True)

print("== filter VALUE bias vs Sinkhorn epsilon (T=50, N=1008) ==", flush=True)
initial_mean = tf.zeros([3], DTYPE)
cb = make_callbacks(model, theta0, "diag_lgssm_epsdiag", 3, 3, initial_mean)
exact_v = exact_kalman(np.array(theta0_np), 50)
for eps in (2.0, 0.5, 0.1):
    vals = [float(canonical_value_and_diagnostics(cb, obs_all, particle_count=n, seed=s, flow_flow_substeps=16, resample_seed=s, epsilon=eps, sinkhorn_steps=24 if eps < 1.0 else 8)["value"].numpy()) for s in range(6)]
    print(f"[eps={eps}] mean={np.mean(vals):.3f} exact={exact_v:.3f} bias={np.mean(vals)-exact_v:+.3f} SE={np.std(vals,ddof=1)/np.sqrt(len(vals)):.3f}", flush=True)
