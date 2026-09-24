"""Timing probe: how does one dual-parameter LEDH target call scale with N?

Question: is the per-call cost dominated by particle count N, or by the
sequential eager loop (T x substeps), which is independent of N?

This decides whether reducing N for a pilot run actually saves wall time.
Diagnostic only - not a research-decision artifact.
"""

import json
import os
import sys
import time
from pathlib import Path

import tensorflow as tf

# Memory growth before any device init (CLAUDE.md TensorFlow GPU Memory Rule).
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


def build_target(model, observations, N, d, dtype, substeps, sinkhorn):
    generator = tf.random.Generator.from_seed(81100)
    initial_states = generator.normal([N, d], dtype=dtype) * 0.1
    initial_covariances = (
        tf.tile(tf.eye(d, dtype=dtype)[None, :, :], [N, 1, 1]) * 0.01
    )
    T = observations.shape[0]
    noises = generator.normal([T, N, d], dtype=dtype) * 0.1

    reset_basis = tf.concat(
        [tf.eye(d, dtype=dtype), -tf.eye(d, dtype=dtype)], axis=0
    )
    reset_repeats = (N + 2 * d - 1) // (2 * d)
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
    exact = {**shared, "reset_ridge": 1e-5, "correction_lm_damping": 1e-2}
    biased = {**shared, "reset_ridge": 1e-3, "correction_lm_damping": 1.0}

    return DualParameterLEDHTarget(
        model=model,
        initial_states=initial_states,
        initial_covariances=initial_covariances,
        noises=noises,
        observations=observations,
        exact_params=exact,
        biased_params=biased,
    )


def time_call(target, theta, repeats):
    """Time value+gradient (what one HMC leapfrog force eval costs)."""
    # Warm up once; do not count it.
    with tf.GradientTape() as tape:
        tape.watch(theta)
        v = target(theta)
    g = tape.gradient(v, theta)
    _ = float(v), float(tf.norm(g))

    times = []
    for _ in range(repeats):
        t0 = time.time()
        with tf.GradientTape() as tape:
            tape.watch(theta)
            v = target(theta)
        g = tape.gradient(v, theta)
        _ = float(v), float(tf.norm(g))  # force sync
        times.append(time.time() - t0)
    return times


def main():
    observations = _lgssm_frozen_observations()
    model = _diagonal_lgssm_fused_model()
    d = 3
    dtype = observations.dtype
    T = int(observations.shape[0])
    theta = tf.constant([1.0, 1.0, 1.0, 0.5, 0.3], dtype=dtype)

    substeps = int(os.environ.get("PROBE_SUBSTEPS", 8))
    sinkhorn = int(os.environ.get("PROBE_SINKHORN", 8))
    repeats = int(os.environ.get("PROBE_REPEATS", 3))
    n_list = [int(x) for x in os.environ.get("PROBE_N", "24,252,1008").split(",")]

    print("Dual-parameter LEDH target scaling probe (diagnostic only)")
    print(f"  gpus: {[g.name for g in _GPUS]}")
    print(f"  CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','unset')}"
          f" CUDA_DEVICE_ORDER={os.environ.get('CUDA_DEVICE_ORDER','unset')}")
    print(f"  T={T} d={d} substeps={substeps} sinkhorn={sinkhorn} repeats={repeats}")
    print(f"  one timed unit = value + gradient = 2 LEDH filter passes\n")

    rows = []
    for N in n_list:
        target = build_target(model, observations, N, d, dtype, substeps, sinkhorn)
        times = time_call(target, theta, repeats)
        best = min(times)
        mean = sum(times) / len(times)
        rows.append({"N": N, "best_s": best, "mean_s": mean, "times_s": times})
        print(f"  N={N:5d}  best={best:7.3f}s  mean={mean:7.3f}s  "
              f"per_LEDH_pass={best/2:7.3f}s")

    print("\n  Scaling relative to smallest N:")
    base = rows[0]
    for r in rows:
        n_ratio = r["N"] / base["N"]
        t_ratio = r["best_s"] / base["best_s"] if base["best_s"] > 0 else float("nan")
        print(f"    N x{n_ratio:6.1f}  ->  time x{t_ratio:5.2f}")

    print("\n  Interpretation guide:")
    print("    time ratio ~ N ratio      -> cost is particle-bound; small N helps")
    print("    time ratio ~ 1            -> cost is loop-bound; small N does NOT help")

    out = Path(os.environ.get("PROBE_OUT", "/tmp/ledh_scaling_probe.json"))
    out.write_text(json.dumps({
        "T": T, "d": d, "substeps": substeps, "sinkhorn": sinkhorn,
        "repeats": repeats, "rows": rows,
        "tensorflow": tf.__version__,
        "gpus": [g.name for g in _GPUS],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
    }, indent=2))
    print(f"\n  saved: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
