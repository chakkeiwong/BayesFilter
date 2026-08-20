"""Adapted-maps engine validation: rungs 3-4 of the design-note ladder.

docs/plans/bayesfilter-adapted-coordinate-maps-design-note-2026-08-20.md
Section 5. Rung 3: n=2 — adapted engine must pass the ladder tolerance
at r=6 where the fixed engine already does (Sobol 32768 reference) AND
at the cheaper Sobol-8192 budget. Rung 4: n=4 smoke — per-step error
table vs exact Kalman under adapted maps (M2 hint = companion Kalman
predictive moments, the benchmark owns the model). Single seed,
descriptive; success bar per note Section 5 rung 4: per-step |err|<=0.1.
"""
import os, sys, time
LOG = "/tmp/adapted_engine_validation.log"
if __name__ == "__main__" and "--detach" in sys.argv and os.fork() > 0:
    print(f"detached; output -> {LOG}"); sys.exit(0)
if __name__ == "__main__" and "--detach" in sys.argv:
    os.setsid(); fd = os.open(LOG, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
    os.dup2(fd, 1); os.dup2(fd, 2)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "docs", "benchmarks"))

import numpy as np, tensorflow as tf
from run_n4_step_localization_20260819 import case_with_steps
from bayesfilter.highdim.squared_tt_engine_v0_tf import EngineConfig
from bayesfilter.highdim.squared_tt_engine_adapted_tf import (
    run_value_filter_branch_axis_adapted,
)

DTYPE = tf.float64
HORIZON = 8


def kalman_hint_factory(n, seed):
    """Companion Kalman filter: predictive-moment hints (m_t|y_1:t-ish).

    Uses the FILTERED moment at t (the paper's 5.2 estimates moments of
    q_t itself; the filtered moment is the tightest available already-
    computed proxy the benchmark owns). Model matrices duplicated from
    case_with_steps' fixture family.
    """

    rng = np.random.default_rng(seed)
    A = 0.7 * np.eye(n) + 0.1 * rng.standard_normal((n, n)) / max(1, n - 1)
    Q = 0.4 * np.eye(n); H = np.eye(n); R = 0.5 * np.eye(n)
    state = {"mean": np.zeros(n), "cov": np.eye(n), "t": 0}

    def hint(t, y_t):
        # advance the companion filter to time t (called once per step, in order)
        assert t == state["t"] + 1, "hints must be requested in step order"
        mean = A @ state["mean"]; cov = A @ state["cov"] @ A.T + Q
        S = H @ cov @ H.T + R
        K = cov @ H.T @ np.linalg.inv(S)
        y = np.asarray(y_t)
        f_mean = mean + K @ (y - H @ mean)
        f_cov = cov - K @ S @ K.T
        state["mean"], state["cov"], state["t"] = f_mean, f_cov, t
        return tf.constant(f_mean, DTYPE), tf.constant(f_cov, DTYPE)

    def observe_t0(y0):
        S = state["cov"] + R
        K = state["cov"] @ np.linalg.inv(S)
        state["mean"] = state["mean"] + K @ (np.asarray(y0) - state["mean"])
        state["cov"] = state["cov"] - K @ S @ K.T

    return hint, observe_t0


def run_case(n, rows, rank, label):
    adapter, ys, kalman_steps = case_with_steps(n, 42 + n)
    hint, observe_t0 = kalman_hint_factory(n, 42 + n)
    observe_t0(ys[0].numpy())
    config = EngineConfig(basis_degree=12, rank=rank, row_count=rows, sweeps=3,
        ridge=1e-10, tau=1e-6, coordinate_half_width=3.0, seed=91000 + 10 * n + rank,
        row_design="sobol")
    t0 = time.time()
    value, diags = run_value_filter_branch_axis_adapted(
        adapter, ys, config, predictive_moment_hint=hint)
    total_gap = abs(float(value.numpy()) - sum(kalman_steps))
    print(f"\n{label}: n={n} rows={rows} r={rank} total_gap={total_gap:.3e} "
          f"per_step_avg={total_gap/HORIZON:.3e} wall={time.time()-t0:.0f}s", flush=True)
    print(f"{'t':>2} {'err':>10} {'fit_rms':>9} {'shrink':>7} {'z_old_max':>9}")
    for d, k in zip(diags, kalman_steps):
        err = d["log_increment"] - k
        print(f"{d['time_index']:>2} {err:>+10.4f} {d['weighted_fit_rms']:>9.2e} "
              f"{d.get('map_shrink', float('nan')):>7.3f} {d.get('z_old_max', float('nan')):>9.3f}",
              flush=True)
    return total_gap


if __name__ == "__main__":
    # Rung 3: n=2 at the ladder budget and the cheap budget
    gap_a = run_case(2, 32768, 6, "rung3-n2-ladderbudget")
    print(f"rung3 ladder-tolerance check (2.5e-3/step): "
          f"{'PASS' if gap_a / HORIZON <= 2.5e-3 else 'FAIL'} ({gap_a/HORIZON:.3e})", flush=True)
    gap_b = run_case(2, 8192, 6, "rung3-n2-cheap")
    # Rung 4: n=4 smoke
    gap_c = run_case(4, 8192, 6, "rung4-n4-smoke")
    print(f"rung4 bar (<=0.1/step): "
          f"{'PASS' if gap_c / HORIZON <= 0.1 else 'FAIL'} ({gap_c/HORIZON:.3e})", flush=True)
