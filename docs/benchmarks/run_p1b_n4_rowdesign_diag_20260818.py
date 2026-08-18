"""n=4 Sobol row-scaling feasibility diagnostic (debugging/scoping run).

Section 3 instrument of
docs/plans/bayesfilter-p1b-n4-row-design-note-2026-08-18.md.
Single seed, descriptive only: does Sobol at affordable budgets move n=4
toward the declared tolerance, and what is the wall cost per cell?
r=6 is the n=2 sufficient rank used as a warm-start hypothesis; it is
NOT assumed sufficient at n=4.
"""
import os, sys, time
from pathlib import Path

# --detach: fork into the background before TF import (classifier-outage
# mitigation: the foreground launch returns immediately; output goes to
# the log file below).
LOG = "/tmp/p1b_n4_rowdesign_diag.log"
if "--detach" in sys.argv and os.fork() > 0:
    print(f"detached; output -> {LOG}")
    sys.exit(0)
if "--detach" in sys.argv:
    os.setsid()
    log_fd = os.open(LOG, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
    os.dup2(log_fd, 1)
    os.dup2(log_fd, 2)

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "docs" / "benchmarks")):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np, tensorflow as tf
from run_p1b_lgssm_value_ladder_20260817 import _case, HORIZON
from bayesfilter.highdim.squared_tt_engine_v0_tf import EngineConfig, run_value_filter_branch_axis

n, r, seed = 4, 6, 42
adapter, ys, exact = _case(n, seed + n, False)
for design, rows in (("sobol", 8192), ("sobol", 16384), ("sobol", 32768)):
    config = EngineConfig(basis_degree=12, rank=r, row_count=rows, sweeps=3,
        ridge=1e-10, tau=1e-6, coordinate_half_width=3.0, seed=91000 + 10*n + r,
        row_design=design)
    t0 = time.time()
    try:
        value, diags = run_value_filter_branch_axis(adapter, ys, config)
        gap = abs(float(value.numpy()) - exact)
        print(f"{design} rows={rows}: per_step={gap/HORIZON:.3e} max_rms={max(d['weighted_fit_rms'] for d in diags):.2e} wall={time.time()-t0:.0f}s", flush=True)
    except Exception as e:
        print(f"{design} rows={rows}: VETO {e} wall={time.time()-t0:.0f}s", flush=True)
