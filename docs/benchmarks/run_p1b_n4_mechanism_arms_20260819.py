"""n=4 mechanism arms (ARM-R / ARM-D / ARM-H) on the XLA value engine.

Declared: Section 5 of
docs/plans/bayesfilter-p1b-n4-row-design-note-2026-08-18.md.
Single seed, one variable at a time vs the baseline
(r=6, deg 12, hw 3.0, sobol 8192; per_step 2.209). Descriptive only.
"""
import os, sys, time
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "docs", "benchmarks"))

import numpy as np, tensorflow as tf
from run_p1b_lgssm_value_ladder_20260817 import _case, HORIZON
from bayesfilter.highdim.squared_tt_engine_v0_tf import EngineConfig
from bayesfilter.highdim.squared_tt_engine_xla_tf import run_value_filter_branch_axis_xla

n, seed = 4, 42
adapter, ys, exact = _case(n, seed + n, False)
BASE = dict(basis_degree=12, rank=6, row_count=8192, sweeps=3, ridge=1e-10,
            tau=1e-6, coordinate_half_width=3.0, row_design="sobol")
ARMS = [
    ("ARM-R rank8", dict(BASE, rank=8)),
    ("ARM-D deg16", dict(BASE, basis_degree=16)),
    ("ARM-H hw4.0", dict(BASE, coordinate_half_width=4.0)),
]
for name, kw in ARMS:
    config = EngineConfig(seed=91000 + 10 * n + kw["rank"], **kw)
    t0 = time.time()
    try:
        value, diags = run_value_filter_branch_axis_xla(adapter, ys, config)
        gap = abs(float(value.numpy()) - exact)
        rms = max(d["weighted_fit_rms"] for d in diags)
        print(f"{name}: per_step={gap/HORIZON:.3e} max_rms={rms:.2e} wall={time.time()-t0:.0f}s", flush=True)
    except Exception as e:
        print(f"{name}: VETO {str(e)[:160]} wall={time.time()-t0:.0f}s", flush=True)
print("baseline: per_step=2.209e+00 (Section 5)", flush=True)
