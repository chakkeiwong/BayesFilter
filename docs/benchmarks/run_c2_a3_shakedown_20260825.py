"""A3: n=4-scale shakedown on the C2 lanes (campaign plan Phase A3).

Reinstates the reviewed ladder's item 5 in post-F-ENG-1 form. One LGSSM
rung at the stress config (degree 12, rank 6, 2n = 8 Hermite axes +
branch = 9 axes: the attempt04 compile-blowup regime), T = 4, sweeps 8.
Arms: eager@2048 (parity twin), xla@2048 (parity + compile probe),
xla@8192 (row-sizing arm). Records per arm: wall, compile probe, per-fit
row ESS (floor: 5 x 468 = 2340 — EXPECTED to fail at N=2048, forcing
the sizing decision), worst design condition, defensive-corrected
Kalman gap. Heartbeats 'A3 ...'; artifacts under phase_a3/.
CPU-only (parity is same-device; GPU policy is Gate A's).
"""
import json
import math
import os
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests", "highdim"))

import tensorflow as tf  # noqa: E402
import test_c2_gaussian_engine_oracle as T  # noqa: E402
from bayesfilter.highdim.squared_tt_engine_v0_tf import EngineConfig  # noqa: E402
from bayesfilter.highdim.squared_tt_engine_gaussian_tf import (  # noqa: E402
    run_value_filter_branch_axis_gaussian,
)
from bayesfilter.highdim.squared_tt_engine_gaussian_xla_tf import (  # noqa: E402
    run_value_filter_branch_axis_gaussian_xla,
)

OUT_DIR = os.path.join(
    ROOT, "docs", "benchmarks", "artifacts", "c2_completion_20260824", "phase_a3"
)
os.makedirs(OUT_DIR, exist_ok=True)
ESS_FLOOR = 5 * 468
N, HORIZON, DEGREE, RANK, SWEEPS, SEED = 4, 4, 12, 6, 8, 46


def run_arm(label, fn, rows, compile_probe=False):
    adapter, ys, steps, model = T._lgssm_fixture(N, HORIZON, SEED)
    ih, ph = T._exact_hint_factories(model)
    config = EngineConfig(
        basis_degree=DEGREE, rank=RANK, row_count=rows, sweeps=SWEEPS,
        ridge=1e-10, tau=1e-6, coordinate_half_width=3.0,
        seed=94000 + rows, row_design="sobol",
    )
    compile_s = None
    if compile_probe:
        a2, y2, _s2, m2 = T._lgssm_fixture(N, 2, SEED)
        ih2, ph2 = T._exact_hint_factories(m2)
        t0 = time.perf_counter()
        fn(a2, y2, config, predictive_moment_hint=ph2, initial_moment_hint=ih2)
        compile_s = time.perf_counter() - t0
        print(f"A3 {label} compile-probe(T=2) wall={compile_s:.0f}s", flush=True)
    t0 = time.perf_counter()
    value, diags = fn(adapter, ys, config,
                      predictive_moment_hint=ph, initial_moment_hint=ih)
    wall = time.perf_counter() - t0
    kalman_gap = abs(
        (float(value.numpy()) - sum(math.log1p(d["tau_t"]) for d in diags))
        - sum(steps)
    )
    ess_values = [d["row_ess"] for d in diags]
    record = {
        "arm": label, "rows": rows, "wall_s": wall, "compile_probe_s": compile_s,
        "kalman_gap": kalman_gap, "row_ess_min": min(ess_values),
        "row_ess_mean": sum(ess_values) / len(ess_values),
        "ess_floor": ESS_FLOOR,
        "ess_floor_met": min(ess_values) >= ESS_FLOOR,
        "worst_condition_max": max(d.get("worst_condition", 0.0) for d in diags),
        "increments": [d["log_increment"] for d in diags],
        "memory_growth_verified": "no_gpu_visible_intentional",
    }
    with open(os.path.join(OUT_DIR, f"arm_{label}_{rows}.json"), "w") as fh:
        json.dump(record, fh, indent=1)
    print(f"A3 {label} rows={rows} wall={wall:.0f}s kalman={kalman_gap:.2e} "
          f"ess_min={record['row_ess_min']:.0f} (floor {ESS_FLOOR}) "
          f"cond={record['worst_condition_max']:.1e}", flush=True)
    return record


eager = run_arm("eager", run_value_filter_branch_axis_gaussian, 2048)
xla = run_arm("xla", run_value_filter_branch_axis_gaussian_xla, 2048,
              compile_probe=True)
step_gap = max(
    abs(a - b) for a, b in zip(eager["increments"], xla["increments"])
)
print(f"A3 parity rows=2048 step={step_gap:.2e} (floor-ceiling 7e-5)", flush=True)
xla_big = run_arm("xla", run_value_filter_branch_axis_gaussian_xla, 8192)
decision = {
    "parity_step_gap_2048": step_gap,
    "parity_within_floor_ceiling": step_gap <= 7e-5,
    "ess_2048_met": xla["ess_floor_met"],
    "ess_8192_met": xla_big["ess_floor_met"],
    "sizing_decision": (
        "N=2048 sufficient" if xla["ess_floor_met"] else
        ("N=8192 adopted for 8-axis scopes" if xla_big["ess_floor_met"] else
         "ESCALATE: even N=8192 starved — row law or N ladder needed")
    ),
    "compile_probe_s": xla["compile_probe_s"],
}
with open(os.path.join(OUT_DIR, "a3_decision.json"), "w") as fh:
    json.dump(decision, fh, indent=1)
print(f"A3 DECISION {json.dumps(decision)}", flush=True)
print("A3 DONE", flush=True)
