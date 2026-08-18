"""Discriminating probe: per-update lstsq-vs-QR backend disagreement.

Runs the eager engine on the P3.3 sobol cell with the repo solver
monkeypatch-instrumented to ALSO solve each system by the XLA route's
QR kernel, recording per-update solution disagreement and condition.
Decides whether the P3.3 parity failure is backend divergence amplified
by the ALS feedback (expected signature: per-update diff ~ cond*eps,
growing through the filter) or an implementation defect (large diffs
uncorrelated with condition).
"""
import os, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "docs", "benchmarks"))

import numpy as np, tensorflow as tf
import bayesfilter.highdim.fitting as fitting
from bayesfilter.highdim.squared_tt_engine_xla_tf import _solve_scaled_qr
from bayesfilter.highdim.squared_tt_engine_v0_tf import EngineConfig, run_value_filter_branch_axis
from run_p1b_lgssm_value_ladder_20260817 import _case

DT = tf.float64
orig = fitting._solve_scaled_augmented_ridge
records = []

def instrumented(*, design, target_values, weights, ridge, **kw):
    result = orig(design=design, target_values=target_values, weights=weights, ridge=ridge, **kw)
    qr_sol, qr_cond = _solve_scaled_qr(
        tf.convert_to_tensor(design, DT), tf.convert_to_tensor(weights, DT),
        tf.convert_to_tensor(target_values, DT), tf.constant(float(ridge), DT))
    diff = float(tf.norm(qr_sol - result.solution) / tf.maximum(tf.norm(result.solution), 1.0))
    records.append((diff, float(result.scaled_augmented_condition_number), float(qr_cond)))
    return result

fitting._solve_scaled_augmented_ridge = instrumented
import bayesfilter.highdim.squared_tt_engine_v0_tf as eng
eng._solve_scaled_augmented_ridge = instrumented

adapter, ys, exact = _case(2, 44, False)
config = EngineConfig(basis_degree=12, rank=4, row_count=4096, sweeps=3,
    ridge=1e-10, tau=1e-6, coordinate_half_width=3.0, seed=91024, row_design="sobol")
value, _ = run_value_filter_branch_axis(adapter, ys, config)
print(f"updates instrumented: {len(records)}", flush=True)
diffs = np.array([r[0] for r in records]); conds = np.array([r[1] for r in records])
qconds = np.array([r[2] for r in records])
print(f"per-update solution diff: max {diffs.max():.3e} median {np.median(diffs):.3e}", flush=True)
print(f"repo cond: max {conds.max():.3e} median {np.median(conds):.3e}", flush=True)
print(f"qr cond estimate: max {qconds.max():.3e}", flush=True)
print(f"cond*eps bound at max cond: {conds.max()*2.2e-16:.3e}", flush=True)
for i in np.argsort(diffs)[-5:]:
    print(f"  update {i}: diff {diffs[i]:.3e} cond {conds[i]:.3e}", flush=True)
