"""P3.3 gate: eager-vs-XLA value parity + wall comparison.

Scoping note: docs/plans/bayesfilter-p3-xla-port-scoping-note-2026-08-18.md.
Fixtures: the I-P2-verified n in {1,2} quadrature configs
(test_p2_adjoint_engine_fd._config) and one ladder-style scattered cell
(n=2, sobol 4096, r=4). Gate: |value_xla - value_eager| relative <= 1e-12.
Walls are descriptive (single runs; compile time reported separately).
"""
import os, sys, time
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "docs", "benchmarks"))

import numpy as np, tensorflow as tf
from bayesfilter.highdim.squared_tt_engine_v0_tf import run_value_filter_branch_axis
from bayesfilter.highdim.squared_tt_engine_xla_tf import run_value_filter_branch_axis_xla

sys.path.insert(0, os.path.join(ROOT, "tests"))
from tests.highdim.test_p2_adjoint_engine_fd import _family, _config
from run_p1b_lgssm_value_ladder_20260817 import _case, HORIZON
from bayesfilter.highdim.squared_tt_engine_v0_tf import EngineConfig

worst = 0.0
fixtures = []
for label, dim, seed in (("n1-quad", 1, 61), ("n2-quad", 2, 62)):
    adapter, _t, _o, _i, ys = _family(dim, np.zeros(dim), seed)
    fixtures.append((label, adapter, ys, _config(dim)))
for name, adapter, ys, config in fixtures:
    t0 = time.time(); v_e, d_e = run_value_filter_branch_axis(adapter, ys, config); we = time.time() - t0
    t0 = time.time(); v_x1, _ = run_value_filter_branch_axis_xla(adapter, ys, config); wx1 = time.time() - t0
    t0 = time.time(); v_x2, _ = run_value_filter_branch_axis_xla(adapter, ys, config); wx2 = time.time() - t0
    rel = abs(float(v_x1.numpy()) - float(v_e.numpy())) / max(1.0, abs(float(v_e.numpy())))
    worst = max(worst, rel)
    print(f"{name}: eager {float(v_e.numpy()):.12f} xla {float(v_x1.numpy()):.12f} rel {rel:.3e} "
          f"walls eager {we:.0f}s xla-compile {wx1:.0f}s xla-warm {wx2:.0f}s", flush=True)

adapter, ys, exact = _case(2, 44, False)
config = EngineConfig(basis_degree=12, rank=4, row_count=4096, sweeps=3,
    ridge=1e-10, tau=1e-6, coordinate_half_width=3.0, seed=91024, row_design="sobol")
t0 = time.time(); v_e, _ = run_value_filter_branch_axis(adapter, ys, config); we = time.time() - t0
t0 = time.time(); v_x1, _ = run_value_filter_branch_axis_xla(adapter, ys, config); wx1 = time.time() - t0
t0 = time.time(); v_x2, _ = run_value_filter_branch_axis_xla(adapter, ys, config); wx2 = time.time() - t0
rel = abs(float(v_x1.numpy()) - float(v_e.numpy())) / max(1.0, abs(float(v_e.numpy())))
worst = max(worst, rel)
print(f"n2-sobol-cell: eager {float(v_e.numpy()):.12f} xla {float(v_x1.numpy()):.12f} rel {rel:.3e} "
      f"walls eager {we:.0f}s xla-compile {wx1:.0f}s xla-warm {wx2:.0f}s", flush=True)
print(f"WORST rel: {worst:.3e}  gate 1e-12: {'PASS' if worst <= 1e-12 else 'FAIL'}", flush=True)
