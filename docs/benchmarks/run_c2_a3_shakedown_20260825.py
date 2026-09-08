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
# Sized per the beta calibration (2026-08-25): beta(d>4)=0.10 in the
# engine, N=8192 (measured product ESS 5347 >= floor 2340 at d=8).
N, HORIZON, DEGREE, RANK, SWEEPS, SEED = 4, 4, 12, 6, 8, 46


def run_arm(label, fn, rows, horizon=HORIZON, compile_probe=False):
    adapter, ys, steps, model = T._lgssm_fixture(N, horizon, SEED)
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
        "arm": label, "rows": rows, "horizon": horizon, "wall_s": wall, "compile_probe_s": compile_s,
        "kalman_gap": kalman_gap, "row_ess_min": min(ess_values),
        "row_ess_mean": sum(ess_values) / len(ess_values),
        "ess_floor": ESS_FLOOR,
        "ess_floor_met": min(ess_values) >= ESS_FLOOR,
        "worst_condition_max": max(d.get("worst_condition", 0.0) for d in diags),
        "increments": [d["log_increment"] for d in diags],
        "memory_growth_verified": "no_gpu_visible_intentional",
    }
    with open(os.path.join(OUT_DIR, f"arm_{label}_{rows}_T{horizon}.json"), "w") as fh:
        json.dump(record, fh, indent=1)
    print(f"A3 {label} rows={rows} wall={wall:.0f}s kalman={kalman_gap:.2e} "
          f"ess_min={record['row_ess_min']:.0f} (floor {ESS_FLOOR}) "
          f"cond={record['worst_condition_max']:.1e}", flush=True)
    return record


def run_arm_subprocess(label, rows, horizon, compile_probe):
    """Fresh process per arm: the LLVM section-memory crash after
    repeated large XLA compiles in one process is a known failure mode
    (reset memo 2026-08-19; reproduced here on the first A3 rerun)."""
    import subprocess
    args = [sys.executable, os.path.abspath(__file__), "--arm", label,
            str(rows), str(horizon), "1" if compile_probe else "0"]
    proc = subprocess.run(args, capture_output=True, text=True, timeout=3600)
    sys.stdout.write(proc.stdout)
    sys.stdout.flush()
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr[-2000:])
        raise RuntimeError(f"arm {label} rows={rows} T={horizon} failed rc={proc.returncode}")
    with open(os.path.join(OUT_DIR, f"arm_{label}_{rows}_T{horizon}.json")) as fh:
        return json.load(fh)


if len(sys.argv) > 1 and sys.argv[1] == "--arm":
    label, rows, horizon, probe_flag = (
        sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5] == "1")
    fn = (run_value_filter_branch_axis_gaussian if label == "eager"
          else run_value_filter_branch_axis_gaussian_xla)
    run_arm(label, fn, rows, horizon=horizon, compile_probe=probe_flag)
    sys.exit(0)

eager = run_arm_subprocess("eager", 8192, 2, False)
xla = run_arm_subprocess("xla", 8192, 2, True)
step_gap = max(
    abs(a - b) for a, b in zip(eager["increments"], xla["increments"])
)
print(f"A3 parity rows=8192 T=2 step={step_gap:.2e} (floor-ceiling 7e-5)", flush=True)
xla_big = run_arm_subprocess("xla", 8192, HORIZON, False)
decision = {
    "parity_step_gap_8192_T2": step_gap,
    "parity_within_floor_ceiling": step_gap <= 7e-5,
    "ess_8192_met": xla_big["ess_floor_met"],
    "row_law": "beta(d>4)=0.10 tempered Christoffel (engine); first A3 run at beta=0.5/N=2048 crashed fail-closed exactly as CF2 projected",
    "sizing_decision": (
        "beta=0.10 + N=8192 adopted for 8-axis scopes" if xla_big["ess_floor_met"] else
        "ESCALATE: beta=0.10 + N=8192 still starved — N ladder needed"
    ),
    "compile_probe_s": xla["compile_probe_s"],
}
with open(os.path.join(OUT_DIR, "a3_decision.json"), "w") as fh:
    json.dump(decision, fh, indent=1)
print(f"A3 DECISION {json.dumps(decision)}", flush=True)
print("A3 DONE", flush=True)
