"""attempt04 pre-gate: adapted-XLA vs adapted-eager parity + wall check.

Plan: bayesfilter-p1b-attempt04-plan-2026-08-21.md (XLA parity gate at
1e-12 on the n=2 fixture BEFORE any ladder cell). Also measures the n=4
r=8 wall to confirm r in {8,10} cells fit the 45-min stop under XLA.
"""
import os, sys, time
LOG = "/tmp/attempt04_pregate.log"
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
from run_adapted_engine_validation_20260820 import kalman_hint_factory
from bayesfilter.highdim.squared_tt_engine_v0_tf import EngineConfig
from bayesfilter.highdim.squared_tt_engine_adapted_tf import (
    run_value_filter_branch_axis_adapted,
)
from bayesfilter.highdim.squared_tt_engine_adapted_xla_tf import (
    run_value_filter_branch_axis_adapted_xla,
)

if __name__ == "__main__":
    HORIZON = 8
    # parity gate: n=2, r=6
    n = 2
    adapter, ys, kalman_steps = case_with_steps(n, 42 + n)
    config = EngineConfig(basis_degree=12, rank=6, row_count=8192, sweeps=3,
        ridge=1e-10, tau=1e-6, coordinate_half_width=3.0, seed=91026,
        row_design="sobol")
    hint_e, obs0_e = kalman_hint_factory(n, 42 + n); obs0_e(ys[0].numpy())
    t0 = time.time()
    v_e, _de = run_value_filter_branch_axis_adapted(
        adapter, ys, config, predictive_moment_hint=hint_e)
    we = time.time() - t0
    hint_x, obs0_x = kalman_hint_factory(n, 42 + n); obs0_x(ys[0].numpy())
    t0 = time.time()
    v_x, _dx = run_value_filter_branch_axis_adapted_xla(
        adapter, ys, config, predictive_moment_hint=hint_x)
    wx1 = time.time() - t0
    rel = abs(float(v_x.numpy()) - float(v_e.numpy())) / max(1.0, abs(float(v_e.numpy())))
    print(f"parity n=2 r=6: eager {float(v_e.numpy()):.12f} xla {float(v_x.numpy()):.12f} "
          f"rel {rel:.3e} gate 1e-12: {'PASS' if rel <= 1e-12 else 'FAIL'} "
          f"walls eager {we:.0f}s xla(compile) {wx1:.0f}s", flush=True)

    # wall check: n=4, r=8 under XLA
    n = 4
    adapter, ys, kalman_steps = case_with_steps(n, 42 + n)
    config = EngineConfig(basis_degree=12, rank=8, row_count=8192, sweeps=3,
        ridge=1e-10, tau=1e-6, coordinate_half_width=3.0, seed=91048,
        row_design="sobol")
    hint, obs0 = kalman_hint_factory(n, 42 + n); obs0(ys[0].numpy())
    t0 = time.time()
    v4, d4 = run_value_filter_branch_axis_adapted_xla(
        adapter, ys, config, predictive_moment_hint=hint)
    w4 = time.time() - t0
    gap = abs(float(v4.numpy()) - sum(kalman_steps))
    print(f"wallcheck n=4 r=8 XLA: per_step={gap/HORIZON:.3e} wall={w4:.0f}s "
          f"(45-min stop: {'OK' if w4 < 2700 else 'OVER'}) "
          f"max_rms={max(d['weighted_fit_rms'] for d in d4[1:]):.2e}", flush=True)
