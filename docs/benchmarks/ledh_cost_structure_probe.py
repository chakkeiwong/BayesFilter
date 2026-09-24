"""What actually drives LEDH dual-target cost, and does tf.function fix it?

Three questions, cheapest-first, all at small scale:

Q1  Is per-call cost bound by particle count N, or by the sequential stage
    count (T x substeps)? Decides whether a reduced-N pilot is cheaper at all.
Q2  Does cost scale ~linearly in T x substeps? Confirms the loop-bound reading
    and lets us extrapolate to the plan scale (T=50, substeps=8).
Q3  Does tf.function with a stable input_signature reduce per-call cost, and by
    how much, separating one-time trace cost from steady-state cost?

Q3 is the lever that decides feasibility. The pilot script currently runs the
HMC target in eager mode; that was adopted to dodge an OOM that has since been
fixed by tf.custom_gradient, so the eager choice is now an unexamined default,
not a reviewed one. The repo TensorFlow Graph policy wants repeated numerical
kernels behind tf.function with a stable signature.

Diagnostic only. No scientific claim.
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
    generator = tf.random.Generator.from_seed(81100)
    initial_states = generator.normal([N, D], dtype=DTYPE) * 0.1
    initial_covariances = (
        tf.tile(tf.eye(D, dtype=DTYPE)[None, :, :], [N, 1, 1]) * 0.01
    )
    noises = generator.normal([T, N, D], dtype=DTYPE) * 0.1
    reset_basis = tf.concat(
        [tf.eye(D, dtype=DTYPE), -tf.eye(D, dtype=DTYPE)], axis=0
    )
    reset_repeats = (N + 2 * D - 1) // (2 * D)
    reset_design = tf.tile(reset_basis, [reset_repeats, 1])[:N]

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


def eager_value_grad(target, theta):
    with tf.GradientTape() as tape:
        tape.watch(theta)
        v = target(theta)
    g = tape.gradient(v, theta)
    return float(v), float(tf.norm(g))  # force sync


def time_eager(target, repeats):
    eager_value_grad(target, THETA)  # warm up, not counted
    out = []
    for _ in range(repeats):
        t0 = time.time()
        eager_value_grad(target, THETA)
        out.append(time.time() - t0)
    return out


def time_graph(target, repeats):
    """Separate one-time trace cost from steady-state cost."""
    @tf.function(input_signature=[tf.TensorSpec([P], DTYPE)])
    def graph_value_grad(theta):
        with tf.GradientTape() as tape:
            tape.watch(theta)
            v = target(theta)
        return v, tape.gradient(v, theta)

    t0 = time.time()
    v, g = graph_value_grad(THETA)          # includes tracing
    _ = float(v), float(tf.norm(g))
    trace_s = time.time() - t0

    steady = []
    for _ in range(repeats):
        t1 = time.time()
        v, g = graph_value_grad(THETA)
        _ = float(v), float(tf.norm(g))
        steady.append(time.time() - t1)
    return trace_s, steady, float(v)


def main():
    repeats = int(os.environ.get("COST_REPEATS", 2))
    do_graph = os.environ.get("COST_GRAPH", "1") == "1"

    print("LEDH dual-target cost-structure probe (diagnostic only)")
    print(f"  gpus={[g.name for g in _GPUS]} "
          f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','unset')}")
    print(f"  P={P}; one value+grad = 1 (value, K=1) + {P} (score, K=P) "
          f"= {1+P} filter passes")
    print(f"  repeats={repeats}\n")

    results = {"eager": [], "graph": []}

    # Q1: vary N at fixed small loop length.
    print("Q1  cost vs particle count N   (T=5, substeps=2, sinkhorn=2)")
    q1 = []
    for N in (24, 252):
        tgt = build_target(N, T=5, substeps=2, sinkhorn=2)
        ts = time_eager(tgt, repeats)
        q1.append({"N": N, "best_s": min(ts)})
        print(f"    N={N:5d}  best={min(ts):7.3f}s")
    if len(q1) == 2 and q1[0]["best_s"] > 0:
        nr = q1[1]["N"] / q1[0]["N"]
        tr = q1[1]["best_s"] / q1[0]["best_s"]
        print(f"    N x{nr:.1f} -> time x{tr:.2f}   "
              f"({'particle-bound' if tr > 0.5*nr else 'NOT particle-bound'})")
    results["q1"] = q1

    # Q2: vary sequential stage count at fixed N.
    print("\nQ2  cost vs sequential stages T*substeps   (N=24, sinkhorn=2)")
    q2 = []
    for T, ss in ((5, 2), (10, 2), (5, 4)):
        tgt = build_target(24, T=T, substeps=ss, sinkhorn=2)
        ts = time_eager(tgt, repeats)
        stages = T * ss
        q2.append({"T": T, "substeps": ss, "stages": stages, "best_s": min(ts)})
        print(f"    T={T:3d} substeps={ss}  stages={stages:4d}  "
              f"best={min(ts):7.3f}s  per_stage={min(ts)/stages:6.4f}s")
    results["q2"] = q2

    # Q3: eager vs tf.function at one config.
    if do_graph:
        print("\nQ3  eager vs tf.function   (N=24, T=5, substeps=2, sinkhorn=2)")
        tgt = build_target(24, T=5, substeps=2, sinkhorn=2)
        e = time_eager(tgt, repeats)
        print(f"    eager     best={min(e):7.3f}s")
        try:
            trace_s, steady, gval = time_graph(tgt, repeats)
            print(f"    graph     trace={trace_s:7.3f}s  "
                  f"steady_best={min(steady):7.3f}s")
            if min(steady) > 0:
                print(f"    speedup (steady state): x{min(e)/min(steady):.2f}")
                be = min(e)
                bs = min(steady)
                if bs < be:
                    breakeven = trace_s / (be - bs)
                    print(f"    trace cost amortizes after "
                          f"~{breakeven:.0f} calls")
            results["graph"] = {"trace_s": trace_s, "steady_s": steady,
                                "eager_s": e, "value": gval}
        except Exception as exc:  # noqa: BLE001
            print(f"    graph mode FAILED: {type(exc).__name__}: {exc}")
            results["graph"] = {"error": f"{type(exc).__name__}: {exc}"}

    # Extrapolation to plan scale, using the measured per-stage cost.
    if q2:
        per_stage = min(r["best_s"] / r["stages"] for r in q2)
        print("\nExtrapolation to plan scale (T=50, substeps=8 = 400 stages)")
        print(f"  measured per-stage (best case): {per_stage:.4f}s")
        est_call = per_stage * 400
        print(f"  => est. per value+grad call  : {est_call:.1f}s")
        for label, evals in (("pilot 4x2x400 steps x10 leapfrog", 32000),
                             ("plan  4x2x1000 steps x10 leapfrog", 80000)):
            h = est_call * evals / 3600
            print(f"  {label}: {h:,.0f} h = {h/24:,.1f} days")
        print("  NOTE: eager per-stage cost measured at small N; treat as an")
        print("  order-of-magnitude bound, not a calibrated forecast.")
        results["per_stage_s"] = per_stage

    out = Path(os.environ.get("COST_OUT", "/tmp/ledh_cost_structure.json"))
    out.write_text(json.dumps(results, indent=2))
    print(f"\n  saved: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
