"""Structural-suspects review, instrument 1: per-step increment localization.

Note Section 5 decision (bayesfilter-p1b-n4-row-design-note-2026-08-18.md):
compare the engine's per-step log increments against the exact Kalman
per-step predictive increments, n=2 vs n=4 at matched configs, plus
per-step fit rms and retained-Gram conditioning. Localizes WHERE the
~2/step n=4 error enters (t=0 marginal? first transition? uniformly?)
before any further sweep compute. Single seed, descriptive.
"""
import os, sys, time

# --detach: self-daemonize before TF import; output -> LOG (classifier-
# outage mitigation: plain allowlisted foreground launch returns at once).
LOG = "/tmp/n4_step_localization.log"
if __name__ == "__main__" and "--detach" in sys.argv and os.fork() > 0:
    print(f"detached; output -> {LOG}")
    sys.exit(0)
if __name__ == "__main__" and "--detach" in sys.argv:
    os.setsid()
    fd = os.open(LOG, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
    os.dup2(fd, 1); os.dup2(fd, 2)

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "docs", "benchmarks"))

import numpy as np, tensorflow as tf
from run_p1b_lgssm_value_ladder_20260817 import HORIZON, _mvn_log_density
from bayesfilter.highdim.squared_tt_engine_v0_tf import DensityKernelAdapter, EngineConfig
from bayesfilter.highdim.squared_tt_engine_xla_tf import run_value_filter_branch_axis_xla

DTYPE = tf.float64


def case_with_steps(n, seed):
    """Ladder _case plus per-step Kalman increments."""
    rng = np.random.default_rng(seed)
    A = 0.7 * np.eye(n) + 0.1 * rng.standard_normal((n, n)) / max(1, n - 1)
    Q = 0.4 * np.eye(n); H = np.eye(n); R = 0.5 * np.eye(n)
    m0 = np.zeros(n); P0 = np.eye(n)
    x = rng.multivariate_normal(m0, P0)
    ys = []
    for t in range(HORIZON):
        if t > 0:
            x = A @ x + rng.multivariate_normal(np.zeros(n), Q)
        ys.append(H @ x + rng.multivariate_normal(np.zeros(n), R))
    ys = np.stack(ys)
    mean, cov = m0.copy(), P0.copy()
    steps = []
    for i, y in enumerate(ys):
        if i > 0:
            mean = A @ mean
            cov = A @ cov @ A.T + Q
        S = H @ cov @ H.T + R
        innov = y - H @ mean
        steps.append(float(-0.5 * (n * np.log(2 * np.pi) + np.linalg.slogdet(S)[1]
                                   + innov @ np.linalg.solve(S, innov))))
        K = cov @ H.T @ np.linalg.inv(S)
        mean = mean + K @ innov
        cov = cov - K @ S @ K.T
    adapter = DensityKernelAdapter(
        state_dim=n,
        transition_log_density=lambda xc, xp: _mvn_log_density(
            xc, tf.linalg.matvec(tf.constant(A, DTYPE), xp), Q),
        observation_log_density=lambda xc, y: _mvn_log_density(
            tf.linalg.matvec(tf.constant(H, DTYPE), xc), tf.convert_to_tensor(y, DTYPE), R),
        initial_log_density=lambda xc: _mvn_log_density(xc, tf.constant(m0, DTYPE), P0),
    )
    return adapter, tf.constant(ys, DTYPE), steps


if __name__ == "__main__":
    for n, rows in ((2, 8192), (4, 8192)):
        adapter, ys, kalman_steps = case_with_steps(n, 42 + n)
        config = EngineConfig(basis_degree=12, rank=6, row_count=rows, sweeps=3,
            ridge=1e-10, tau=1e-6, coordinate_half_width=3.0, seed=91000 + 10 * n + 6,
            row_design="sobol")
        t0 = time.time()
        value, diags = run_value_filter_branch_axis_xla(adapter, ys, config)
        print(f"\nn={n} rows={rows} total_gap={abs(float(value.numpy()) - sum(kalman_steps)):.3e} "
              f"wall={time.time()-t0:.0f}s", flush=True)
        print(f"{'t':>2} {'tt_incr':>12} {'kalman':>12} {'err':>10} {'fit_rms':>9} {'gram_cond':>10}")
        for d, k in zip(diags, kalman_steps):
            err = d["log_increment"] - k
            print(f"{d['time_index']:>2} {d['log_increment']:>12.6f} {k:>12.6f} {err:>+10.4f} "
                  f"{d['weighted_fit_rms']:>9.2e} {d.get('gram_condition', float('nan')):>10.2e}",
                  flush=True)
