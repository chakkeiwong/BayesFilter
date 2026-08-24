"""Gate A: eager-vs-XLA parity + device timing for the C2 Gaussian lane.

Campaign plan Phase A2 (REVIEWED-EXECUTABLE, 2026-08-24). Configs:
(i) degree-0 T=120 n=2 oracle; (ii) stress T=12 n=2 l=13 r=6 sweeps=8.
Parity is same-device (CPU) eager-vs-jitted, gate <= 1e-12 on the total
and on every per-step increment. Then one GPU timing run of the stress
config on the 4080 (CUDA_DEVICE_ORDER=PCI_BUS_ID, CUDA_VISIBLE_DEVICES=1
set by the launcher) with the memory-growth manifest field (carried
obligation CF11). Heartbeat lines: 'GATEA <stage> ...'; monitors key on
GATEA|Traceback|Error|FAILED|non-finite.

Artifacts: docs/benchmarks/artifacts/c2_completion_20260824/gate_a/
"""
import json
import math
import os
import sys
import time

MODE = sys.argv[1] if len(sys.argv) > 1 else "cpu"
if MODE == "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests", "highdim"))

import tensorflow as tf  # noqa: E402

GPUS = tf.config.list_physical_devices("GPU")
for g in GPUS:
    tf.config.experimental.set_memory_growth(g, True)

import test_c2_gaussian_engine_oracle as T  # noqa: E402
from bayesfilter.highdim.squared_tt_engine_v0_tf import EngineConfig  # noqa: E402
from bayesfilter.highdim.squared_tt_engine_gaussian_tf import (  # noqa: E402
    run_value_filter_branch_axis_gaussian,
)
from bayesfilter.highdim.squared_tt_engine_gaussian_xla_tf import (  # noqa: E402
    run_value_filter_branch_axis_gaussian_xla,
)

OUT_DIR = os.path.join(
    ROOT, "docs", "benchmarks", "artifacts", "c2_completion_20260824", "gate_a"
)
os.makedirs(OUT_DIR, exist_ok=True)
MANIFEST = {
    "mode": MODE,
    "visible_gpus": [
        tf.config.experimental.get_device_details(g).get("device_name") for g in GPUS
    ],
    "memory_growth_verified": all(
        tf.config.experimental.get_memory_growth(g) for g in GPUS
    ) if GPUS else "no_gpu_visible_intentional",
}
print("GATEA manifest", json.dumps(MANIFEST), flush=True)


def run_pair(name, n, horizon, degree, rank, rows, sweeps, seed):
    results = {}
    for label, fn in (("eager", run_value_filter_branch_axis_gaussian),
                      ("xla", run_value_filter_branch_axis_gaussian_xla)):
        adapter, ys, steps, model = T._lgssm_fixture(n, horizon, seed)
        ih, ph = T._exact_hint_factories(model)
        config = EngineConfig(
            basis_degree=degree, rank=rank, row_count=rows, sweeps=sweeps,
            ridge=1e-10, tau=1e-6, coordinate_half_width=3.0,
            seed=93000 + 10 * n + rank, row_design="sobol",
        )
        t0 = time.perf_counter()
        value, diags = fn(adapter, ys, config,
                          predictive_moment_hint=ph, initial_moment_hint=ih)
        wall = time.perf_counter() - t0
        results[label] = (float(value.numpy()), diags, wall)
        print(f"GATEA {name} {label} value={results[label][0]:+.12f} "
              f"wall={wall:.1f}s", flush=True)
    v_e, d_e, w_e = results["eager"]
    v_x, d_x, w_x = results["xla"]
    total_gap = abs(v_e - v_x)
    step_gap = max(
        abs(a["log_increment"] - b["log_increment"]) for a, b in zip(d_e, d_x)
    )
    kalman_gap = abs(
        (v_x - sum(math.log1p(d["tau_t"]) for d in d_x)) - sum(steps)
    )
    record = {
        "config": name, "total_parity_gap": total_gap,
        "max_step_parity_gap": step_gap, "xla_kalman_gap": kalman_gap,
        "wall_eager_s": w_e, "wall_xla_s": w_x, **MANIFEST,
    }
    with open(os.path.join(OUT_DIR, f"parity_{name}_{MODE}.json"), "w") as fh:
        json.dump(record, fh, indent=1)
    status = "PASS" if (total_gap <= 1e-12 and step_gap <= 1e-12) else "FAILED"
    print(f"GATEA {name} parity {status} total={total_gap:.2e} "
          f"step={step_gap:.2e} kalman={kalman_gap:.2e} "
          f"speedup={w_e / max(w_x, 1e-9):.2f}x", flush=True)
    return status == "PASS"


def run_gpu_timing(name, n, horizon, degree, rank, rows, sweeps, seed):
    adapter, ys, steps, model = T._lgssm_fixture(n, horizon, seed)
    ih, ph = T._exact_hint_factories(model)
    config = EngineConfig(
        basis_degree=degree, rank=rank, row_count=rows, sweeps=sweeps,
        ridge=1e-10, tau=1e-6, coordinate_half_width=3.0,
        seed=93000 + 10 * n + rank, row_design="sobol",
    )
    t0 = time.perf_counter()
    value, diags = run_value_filter_branch_axis_gaussian_xla(
        adapter, ys, config, predictive_moment_hint=ph, initial_moment_hint=ih)
    wall = time.perf_counter() - t0
    kalman_gap = abs(
        (float(value.numpy()) - sum(math.log1p(d["tau_t"]) for d in diags))
        - sum(steps)
    )
    record = {"config": name, "wall_gpu_s": wall, "kalman_gap": kalman_gap,
              **MANIFEST}
    with open(os.path.join(OUT_DIR, f"gpu_timing_{name}.json"), "w") as fh:
        json.dump(record, fh, indent=1)
    print(f"GATEA {name} gpu-timing wall={wall:.1f}s kalman={kalman_gap:.2e}",
          flush=True)


if MODE == "cpu":
    ok1 = run_pair("degree0_n2_t120", 2, 120, 0, 1, 512, 3, 44)
    ok2 = run_pair("stress_n2_t12", 2, 12, 12, 6, 2048, 8, 44)
    print(f"GATEA DONE cpu ok={ok1 and ok2}", flush=True)
else:
    run_gpu_timing("stress_n2_t12", 2, 12, 12, 6, 2048, 8, 44)
    print("GATEA DONE gpu", flush=True)
