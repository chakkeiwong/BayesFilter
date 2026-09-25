"""Does the graph-mode speedup hold at PLAN scale?

The x8.65 tf.function speedup was measured at T=5, substeps=2, N=24. Every
budget number for this campaign is currently extrapolated from that single
small-scale figure, which is not safe: at plan scale (T=50, substeps=8,
N=1008) the trace has ~10x more unrolled stages and the per-stage arithmetic is
~40x heavier, so the speedup could go either way.

  - larger, if the small config was dominated by dispatch latency that the
    graph removes;
  - smaller, if plan scale is arithmetic-bound, leaving little dispatch
    overhead to recover.

This probe measures eager and graph at plan scale directly, and reports trace
cost separately from steady-state cost. It also checks that graph and eager
return the same value, so a speedup cannot come from skipped work.

Diagnostic only. No scientific claim. Does not run HMC.
"""

import json
import os
import sys
import time
from pathlib import Path

import tensorflow as tf

_GPUS = tf.config.list_physical_devices("GPU")
for _g in _GPUS:
    tf.config.experimental.set_memory_growth(_g, True)
    if not tf.config.experimental.get_memory_growth(_g):
        raise RuntimeError(f"set_memory_growth failed on {_g.name}")

from bayesfilter.highdim.ledh_canonical_neutra_targets_tf import (
    _diagonal_lgssm_fused_model,
    _lgssm_frozen_observations,
)
from bayesfilter.inference.ledh_dual_parameter_target import (
    DualParameterLEDHTarget,
)

OBS_FULL = _lgssm_frozen_observations()
MODEL = _diagonal_lgssm_fused_model()
D = 3
DTYPE = OBS_FULL.dtype
THETA = tf.constant([1.0, 1.0, 1.0, 0.5, 0.3], dtype=DTYPE)
P = int(THETA.shape[0])


def build_target(N, T, substeps, sinkhorn):
    observations = OBS_FULL[:T]
    gen = tf.random.Generator.from_seed(81100)
    initial_states = gen.normal([N, D], dtype=DTYPE) * 0.1
    initial_covariances = (
        tf.tile(tf.eye(D, dtype=DTYPE)[None, :, :], [N, 1, 1]) * 0.01
    )
    noises = gen.normal([T, N, D], dtype=DTYPE) * 0.1
    basis = tf.concat([tf.eye(D, dtype=DTYPE), -tf.eye(D, dtype=DTYPE)], axis=0)
    reps = (N + 2 * D - 1) // (2 * D)
    reset_design = tf.tile(basis, [reps, 1])[:N]

    shared = dict(
        substeps=substeps,
        reset_policy="contract_e",
        reset_design=reset_design,
        reset_epsilon=2.0,
        reset_sinkhorn_steps=sinkhorn,
        reset_balance_steps=sinkhorn,
        correction_steps=4,
        correction_strength=0.2,
        correction_lm_scale_floor=1e-4,
        correction_trust_radius=0.5,
        pairwise_steps=4,
        pairwise_strength=0.02,
        pairwise_rms_cap=2.0,
        coordinate_cap=0.0,
        annealed_stages=1,
        annealed_seed=0,
    )
    return DualParameterLEDHTarget(
        model=MODEL,
        initial_states=initial_states,
        initial_covariances=initial_covariances,
        noises=noises,
        observations=observations,
        exact_params={**shared, "reset_ridge": 1e-5,
                      "correction_lm_damping": 1e-2},
        biased_params={**shared, "reset_ridge": 1e-3,
                       "correction_lm_damping": 1.0},
    )


def main():
    N = int(os.environ.get("PROBE_N", 1008))
    T = int(os.environ.get("PROBE_T", 50))
    substeps = int(os.environ.get("PROBE_SUBSTEPS", 8))
    sinkhorn = int(os.environ.get("PROBE_SINKHORN", 8))
    repeats = int(os.environ.get("PROBE_REPEATS", 2))

    print("Plan-scale eager-vs-graph probe (diagnostic only)")
    print(f"  gpus={[g.name for g in _GPUS]} "
          f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','unset')}")
    print(f"  N={N} T={T} substeps={substeps} sinkhorn={sinkhorn} "
          f"repeats={repeats}")
    print(f"  one value+grad = 1 (value, K=1) + {P} (score, K=P) "
          f"= {1+P} filter passes")
    print(f"  reference small-scale figures: eager 21.157s, graph steady "
          f"2.446s, x8.65\n", flush=True)

    target = build_target(N, T, substeps, sinkhorn)
    out = {"N": N, "T": T, "substeps": substeps, "sinkhorn": sinkhorn,
           "repeats": repeats}

    # --- eager ---
    print("  [eager] warmup ...", flush=True)
    t0 = time.time()
    with tf.GradientTape() as tape:
        tape.watch(THETA)
        v = target(THETA)
    g = tape.gradient(v, THETA)
    eager_value = float(v)
    eager_warm = time.time() - t0
    print(f"  [eager] warmup {eager_warm:.1f}s  value={eager_value!r}",
          flush=True)

    eager_times = []
    for i in range(repeats):
        t1 = time.time()
        with tf.GradientTape() as tape:
            tape.watch(THETA)
            v = target(THETA)
        g = tape.gradient(v, THETA)
        _ = float(v), float(tf.norm(g))
        dt = time.time() - t1
        eager_times.append(dt)
        print(f"  [eager] rep {i+1}: {dt:.1f}s", flush=True)
    out["eager_warm_s"] = eager_warm
    out["eager_s"] = eager_times
    out["eager_value"] = eager_value

    # --- graph ---
    print("\n  [graph] tracing (may be slow: ~T*substeps unrolled stages) ...",
          flush=True)

    @tf.function(input_signature=[tf.TensorSpec([P], DTYPE)])
    def graph_value_grad(theta):
        with tf.GradientTape() as tape:
            tape.watch(theta)
            val = target(theta)
        return val, tape.gradient(val, theta)

    try:
        t2 = time.time()
        v, g = graph_value_grad(THETA)
        graph_value = float(v)
        _ = float(tf.norm(g))
        trace_s = time.time() - t2
        print(f"  [graph] trace+first call {trace_s:.1f}s  "
              f"value={graph_value!r}", flush=True)

        steady = []
        for i in range(repeats):
            t3 = time.time()
            v, g = graph_value_grad(THETA)
            _ = float(v), float(tf.norm(g))
            dt = time.time() - t3
            steady.append(dt)
            print(f"  [graph] rep {i+1}: {dt:.1f}s", flush=True)

        out["graph_trace_s"] = trace_s
        out["graph_steady_s"] = steady
        out["graph_value"] = graph_value

        be, bs = min(eager_times), min(steady)
        speedup = be / bs if bs > 0 else float("nan")
        rel = abs(graph_value - eager_value) / max(abs(eager_value), 1e-300)
        out["speedup"] = speedup
        out["value_rel_diff"] = rel

        print(f"\n  eager best        : {be:8.1f}s")
        print(f"  graph steady best : {bs:8.1f}s")
        print(f"  SPEEDUP           : x{speedup:.2f}   "
              f"(small-scale reference: x8.65)")
        print(f"  value agreement   : rel diff {rel:.3e}  "
              f"{'OK' if rel <= 1e-12 else 'MISMATCH -- speedup not trustworthy'}")
        if bs < be:
            print(f"  trace amortizes after ~{trace_s/(be-bs):.1f} calls")

        # Campaign arithmetic from the MEASURED plan-scale number.
        print("\n  Campaign cost from this measurement:")
        for label, evals in (("pilot (4x2x400 steps x10 leapfrog)", 32000),
                             ("plan  (4x2x1000 steps x10 leapfrog)", 80000)):
            se, sg = be * evals, bs * evals
            print(f"    {label}")
            print(f"      eager: {se/86400:7.1f} days     "
                  f"graph: {sg/86400:7.1f} days")
        print(f"\n  Fits in 4 GPU-hours (graph): "
              f"{4*3600/bs:,.0f} value+grad calls "
              f"= {4*3600/bs/(4*2*10):,.1f} HMC steps per chain "
              f"(4 arms x 2 chains x 10 leapfrog)")
        out["calls_in_4h_graph"] = 4 * 3600 / bs
    except Exception as exc:  # noqa: BLE001
        print(f"  [graph] FAILED: {type(exc).__name__}: {exc}", flush=True)
        out["graph_error"] = f"{type(exc).__name__}: {exc}"
        print("\n  Graph mode failing at plan scale is itself the finding:")
        print("  it would remove the only no-approval lever and leave the")
        print("  campaign at the eager cost.")

    p = Path(os.environ.get("PROBE_OUT", "/tmp/ledh_plan_scale_graph.json"))
    p.write_text(json.dumps(out, indent=2))
    print(f"\n  saved: {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
