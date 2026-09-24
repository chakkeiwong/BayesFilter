#!/usr/bin/env python3
"""Execution-mode matrix for one LEDH value+score evaluation.

Question: how much of the 39.3 s/call plan-scale cost is launch latency rather
than arithmetic, and which compilation lever removes it?

The measured cost law says time is independent of N (x0.99 over 24->252) and of
substeps (x1.01 over 2->4) but exactly linear in T (x2.00). Meanwhile
correction_steps 4->0 saves 37% and pairwise_steps 4->0 saves 32%. Fixed
iteration counts per timestep, insensitive to problem size, is the signature of
many tiny sequential kernels with the GPU mostly idle. If so, fusion (XLA) is
the lever, not vectorization.

Measures trace/compile time and steady-state time for each cell of
  {eager, graph, graph+XLA} x {sequential, pfor}

Reports steady state (min over repeats) separately from first-call cost, since
across 32,000 calls only steady state matters.

NOT concluded here: numerical equivalence under XLA. That is a separate gate
(ledh_k_batch_parity_and_timing.py covers sequential-vs-pfor parity; an XLA cell
that wins on time still owes a parity check before promotion).
"""
import os
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

import json
import time

import tensorflow as tf

_GPUS = tf.config.list_physical_devices("GPU")
for _g in _GPUS:
    tf.config.experimental.set_memory_growth(_g, True)
    if not tf.config.experimental.get_memory_growth(_g):
        raise RuntimeError(f"set_memory_growth failed on {_g.name}")
print(f"GPU_MEMORY_POLICY: growth verified on {len(_GPUS)} GPU(s)", flush=True)

from bayesfilter.highdim.ledh_canonical_batch_fused_tf import (
    canonical_batch_fused_value_score,
)
from bayesfilter.highdim.ledh_canonical_neutra_targets_tf import (
    _diagonal_lgssm_fused_model,
    _lgssm_frozen_observations,
)

OBS_FULL = _lgssm_frozen_observations()
MODEL = _diagonal_lgssm_fused_model()
D, P = 3, 5
DTYPE = OBS_FULL.dtype
THETA = tf.constant([[1.0, 1.0, 1.0, 0.5, 0.3]], dtype=DTYPE)
DIRECTIONS = tf.eye(P, dtype=DTYPE)[None, :, :]


def fixture(N, T, substeps, sinkhorn):
    basis = tf.concat([tf.eye(D, dtype=DTYPE), -tf.eye(D, dtype=DTYPE)], axis=0)
    reps = (N + 2 * D - 1) // (2 * D)
    params = dict(
        substeps=substeps, reset_policy="contract_e",
        reset_design=tf.tile(basis, [reps, 1])[:N], reset_epsilon=2.0,
        reset_sinkhorn_steps=sinkhorn, reset_balance_steps=sinkhorn,
        correction_steps=4, correction_strength=0.2,
        correction_lm_scale_floor=1e-4, correction_trust_radius=0.5,
        pairwise_steps=4, pairwise_strength=0.02, pairwise_rms_cap=2.0,
        coordinate_cap=0.0, annealed_stages=1, annealed_seed=0,
        reset_ridge=1e-5, correction_lm_damping=1e-2,
    )
    gen = tf.random.Generator.from_seed(81100)
    states = gen.normal([N, D], dtype=DTYPE) * 0.1
    covs = tf.tile(tf.eye(D, dtype=DTYPE)[None], [N, 1, 1]) * 0.01
    noises = gen.normal([T, N, D], dtype=DTYPE) * 0.1
    return states, covs, noises, OBS_FULL[:T], params


def build(N, T, substeps, sinkhorn, compile_mode, k_mode):
    """Closure over everything but theta/directions, so the signature is fixed
    and `self`-style free-variable AutoGraph conversion cannot trigger."""
    states, covs, noises, obs, params = fixture(N, T, substeps, sinkhorn)

    def raw(theta, directions):
        return canonical_batch_fused_value_score(
            MODEL, theta, directions, states, covs, noises, obs,
            k_batch_mode=k_mode, **params,
        )

    if compile_mode == "eager":
        return raw
    sig = [tf.TensorSpec([1, P], DTYPE), tf.TensorSpec([1, P, P], DTYPE)]
    return tf.function(
        raw, input_signature=sig, jit_compile=(compile_mode == "xla")
    )


def measure(fn, repeats):
    t0 = time.time()
    out = fn(THETA, DIRECTIONS)
    _ = float(tf.reduce_sum(out[1]))
    first = time.time() - t0
    best = float("inf")
    for _ in range(repeats):
        t1 = time.time()
        out = fn(THETA, DIRECTIONS)
        _ = float(tf.reduce_sum(out[1]))
        best = min(best, time.time() - t1)
    return first, best


def main():
    scale = os.environ.get("MATRIX_SCALE", "small")
    if scale == "plan":
        N, T, substeps, sinkhorn, repeats = 252, 50, 8, 8, 2
    else:
        N, T, substeps, sinkhorn, repeats = 24, 5, 2, 8, 3
    print(f"scale={scale}  N={N} T={T} substeps={substeps} "
          f"sinkhorn={sinkhorn} K={P} repeats={repeats}\n", flush=True)

    cells = [("eager", "sequential"), ("graph", "sequential"),
             ("xla", "sequential"), ("graph", "pfor"), ("xla", "pfor")]
    rows, baseline = [], None
    print(f"  {'compile':8s} {'K mode':11s} {'first':>10s} "
          f"{'steady':>10s} {'vs eager':>9s}", flush=True)
    for compile_mode, k_mode in cells:
        try:
            fn = build(N, T, substeps, sinkhorn, compile_mode, k_mode)
            first, steady = measure(fn, repeats)
        except Exception as exc:  # a failed cell is evidence, not a crash
            print(f"  {compile_mode:8s} {k_mode:11s}  FAILED: "
                  f"{type(exc).__name__}: {str(exc)[:80]}", flush=True)
            rows.append({"compile": compile_mode, "k_mode": k_mode,
                         "failed": f"{type(exc).__name__}: {exc}"})
            continue
        if baseline is None:
            baseline = steady
        print(f"  {compile_mode:8s} {k_mode:11s} {first:9.3f}s "
              f"{steady:9.3f}s {baseline / steady:8.2f}x", flush=True)
        rows.append({"compile": compile_mode, "k_mode": k_mode,
                     "first_s": first, "steady_s": steady,
                     "speedup_vs_eager": baseline / steady, "failed": None})

    out = os.environ.get("MATRIX_OUT", f"/tmp/ledh_exec_matrix_{scale}.json")
    with open(out, "w") as f:
        json.dump({"scale": scale, "N": N, "T": T, "substeps": substeps,
                   "sinkhorn": sinkhorn, "K": P, "rows": rows}, f, indent=2)
    print(f"\n  saved: {out}", flush=True)
    print("  Speedup is wall time only. A cell that wins on time still owes a")
    print("  parity check against the sequential non-XLA reference.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
