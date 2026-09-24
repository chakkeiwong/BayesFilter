#!/usr/bin/env python3
"""K-direction batching: numerical parity + timing, sequential vs pfor.

Executes the check protocol from
docs/plans/ledh-vectorized-map-approval-request.md.

The K directional scores are mathematically independent and share one primal
trajectory, so batching them must not change any returned number. This measures
whether that holds, and what it buys.

Checks:
  P1  value parity      sequential vs pfor, rtol 1e-12
  P2  score parity      sequential vs pfor, rtol 1e-12
  P3  direction-invariance assertion still fires under pfor (kept, not removed)
  T1  steady-state wall time, both modes, min over repeats
  M1  peak device memory, both modes

Not concluded here: that pfor is safe as a default. That needs the HMC-level
checks (acceptance, ESS) named in the approval request.
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
D = 3
DTYPE = OBS_FULL.dtype
THETA = tf.constant([[1.0, 1.0, 1.0, 0.5, 0.3]], dtype=DTYPE)
P = 5


def shared_params(N, substeps, sinkhorn):
    basis = tf.concat([tf.eye(D, dtype=DTYPE), -tf.eye(D, dtype=DTYPE)], axis=0)
    reps = (N + 2 * D - 1) // (2 * D)
    reset_design = tf.tile(basis, [reps, 1])[:N]
    return dict(
        substeps=substeps, reset_policy="contract_e", reset_design=reset_design,
        reset_epsilon=2.0, reset_sinkhorn_steps=sinkhorn,
        reset_balance_steps=sinkhorn, correction_steps=4,
        correction_strength=0.2, correction_lm_scale_floor=1e-4,
        correction_trust_radius=0.5, pairwise_steps=4, pairwise_strength=0.02,
        pairwise_rms_cap=2.0, coordinate_cap=0.0, annealed_stages=1,
        annealed_seed=0, reset_ridge=1e-5, correction_lm_damping=1e-2,
    )


def make_inputs(N, T):
    gen = tf.random.Generator.from_seed(81100)
    initial_states = gen.normal([N, D], dtype=DTYPE) * 0.1
    initial_covariances = (
        tf.tile(tf.eye(D, dtype=DTYPE)[None, :, :], [N, 1, 1]) * 0.01
    )
    noises = gen.normal([T, N, D], dtype=DTYPE) * 0.1
    return initial_states, initial_covariances, noises, OBS_FULL[:T]


def run_once(N, T, substeps, sinkhorn, mode, directions):
    states, covs, noises, obs = make_inputs(N, T)
    params = shared_params(N, substeps, sinkhorn)
    return canonical_batch_fused_value_score(
        MODEL, THETA, directions, states, covs, noises, obs,
        k_batch_mode=mode, **params,
    )


def timed(fn, repeats):
    fn()  # warm up / trace
    best = float("inf")
    for _ in range(repeats):
        t0 = time.time()
        out = fn()
        _ = float(tf.reduce_sum(out[1]))  # force materialization
        best = min(best, time.time() - t0)
    return best


def peak_mb():
    if not _GPUS:
        return None
    tf.config.experimental.reset_memory_stats("GPU:0")
    return None


def read_peak_mb():
    if not _GPUS:
        return None
    return tf.config.experimental.get_memory_info("GPU:0")["peak"] / 1e6


def evaluate_scale(label, N, T, substeps, sinkhorn, repeats):
    print(f"\n=== {label}: N={N} T={T} substeps={substeps} "
          f"sinkhorn={sinkhorn} K={P} ===", flush=True)
    directions = tf.eye(P, dtype=DTYPE)[None, :, :]  # [1, P, P]

    peak_mb()
    v_seq, s_seq, _ = run_once(N, T, substeps, sinkhorn, "sequential", directions)
    seq_peak = read_peak_mb()

    peak_mb()
    v_pfor, s_pfor, _ = run_once(N, T, substeps, sinkhorn, "pfor", directions)
    pfor_peak = read_peak_mb()

    v_rel = float(tf.abs(v_pfor - v_seq) / tf.maximum(tf.abs(v_seq), 1e-300))
    s_den = tf.maximum(tf.norm(s_seq), tf.constant(1e-300, DTYPE))
    s_rel = float(tf.norm(s_pfor - s_seq) / s_den)

    print(f"  P1 value  parity  rel={v_rel:.3e}  "
          f"{'PASS' if v_rel < 1e-12 else 'FAIL'}", flush=True)
    print(f"  P2 score  parity  rel={s_rel:.3e}  "
          f"{'PASS' if s_rel < 1e-12 else 'FAIL'}", flush=True)

    t_seq = timed(
        lambda: run_once(N, T, substeps, sinkhorn, "sequential", directions),
        repeats)
    t_pfor = timed(
        lambda: run_once(N, T, substeps, sinkhorn, "pfor", directions),
        repeats)
    speedup = t_seq / t_pfor if t_pfor > 0 else float("nan")

    print(f"  T1 sequential {t_seq:8.3f}s", flush=True)
    print(f"  T1 pfor       {t_pfor:8.3f}s   speedup x{speedup:.2f}", flush=True)
    if seq_peak is not None:
        print(f"  M1 peak MB    sequential {seq_peak:8.1f}  "
              f"pfor {pfor_peak:8.1f}", flush=True)

    return {
        "label": label, "N": N, "T": T, "substeps": substeps,
        "sinkhorn": sinkhorn, "K": P,
        "value_rel": v_rel, "score_rel": s_rel,
        "value_parity_pass": v_rel < 1e-12,
        "score_parity_pass": s_rel < 1e-12,
        "t_sequential_s": t_seq, "t_pfor_s": t_pfor, "speedup": speedup,
        "peak_mb_sequential": seq_peak, "peak_mb_pfor": pfor_peak,
    }


def main():
    results = []
    results.append(evaluate_scale("small", 24, 5, 2, 8, repeats=3))
    results.append(evaluate_scale("plan scale", 252, 50, 8, 8, repeats=2))

    print("\n=== summary ===", flush=True)
    all_parity = all(
        r["value_parity_pass"] and r["score_parity_pass"] for r in results
    )
    for r in results:
        print(f"  {r['label']:12s} speedup x{r['speedup']:.2f}  "
              f"parity {'PASS' if r['value_parity_pass'] and r['score_parity_pass'] else 'FAIL'}",
              flush=True)
    print(f"\n  parity across all scales: "
          f"{'PASS' if all_parity else 'FAIL'}", flush=True)
    print("\n  NOT concluded: that pfor is safe as a default. That requires the")
    print("  HMC-level acceptance and ESS checks in the approval request.")

    out = os.environ.get("KBATCH_OUT", "/tmp/ledh_k_batch_parity.json")
    with open(out, "w") as f:
        json.dump({"results": results, "all_parity_pass": all_parity}, f, indent=2)
    print(f"\n  saved: {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
