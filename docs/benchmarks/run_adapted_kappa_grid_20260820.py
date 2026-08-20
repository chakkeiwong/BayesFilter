"""Kappa-grid arm for the adapted engine (n=4 smoke only).

Round-2 diagnosis: kappa_c=4 relieved containment shrink but diluted
current-block coverage (err ~-0.8/step, matching the probe's kappa=4
arm); round-1 kappa_c=3 had coverage but shrink-truncation. This grid
measures the trade directly. Single seed, descriptive.
"""
import os, sys, time
LOG = "/tmp/adapted_kappa_grid.log"
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

HORIZON = 8
n = int(os.environ.get("GRID_N", "4"))
for kc, kp in ((5.0, 4.0), (6.0, 4.0), (6.0, 5.0)):
    adapter, ys, kalman_steps = case_with_steps(n, 42 + n)
    hint, observe_t0 = kalman_hint_factory(n, 42 + n)
    observe_t0(ys[0].numpy())
    config = EngineConfig(basis_degree=12, rank=6, row_count=8192, sweeps=3,
        ridge=1e-10, tau=1e-6, coordinate_half_width=3.0, seed=91046,
        row_design="sobol")
    t0 = time.time()
    try:
        value, diags = run_value_filter_branch_axis_adapted(
            adapter, ys, config, predictive_moment_hint=hint,
            map_kappa_prev=kp, map_kappa_current=kc)
        gap = abs(float(value.numpy()) - sum(kalman_steps))
        shr = min(d.get("map_shrink", 1.0) for d in diags[1:])
        print(f"kc={kc} kp={kp}: per_step={gap/HORIZON:.3e} min_shrink={shr:.3f} "
              f"wall={time.time()-t0:.0f}s", flush=True)
    except Exception as e:
        print(f"kc={kc} kp={kp}: VETO {str(e)[:120]}", flush=True)
