"""Q2 Curve 1: annealed-SMC k/c response surface, multi-seed.

Contract: `docs/plans/bayesfilter-q2-calibration-campaign-plan-2026-08-24.md`
(Curve 1). Frozen Austria observations, N=1008, through the claim-bearing
`canonical_value_and_diagnostics` filter (call-chain rule: NOT the probe's
inline loop). Lanes:
  --lane f32tf32  float32 + TF32 on GPU (production target); fails closed
                  if no GPU or memory growth cannot be verified.
  --lane f64cpu   float64 CPU anchor cell(s) (GPU intentionally hidden by
                  the launcher via CUDA_VISIBLE_DEVICES=-1).

Usage:
  python run_q2_annealed_kc_response_surface_20260824.py --lane f32tf32 \
      --stages 1,2,4,8 --caps 8,inf --seeds 0,1,2 [--tag full]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, _ROOT)

import numpy as np  # noqa: E402  (diagnostic/reporting only)
import tensorflow as tf  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lane", choices=["f32tf32", "f64cpu"], required=True)
    parser.add_argument("--stages", default="1,2,4,8")
    parser.add_argument("--caps", default="8,inf")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--tag", default="run")
    args = parser.parse_args()

    started = time.time()
    stages = [int(s) for s in args.stages.split(",")]
    caps = [float(c) for c in args.caps.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]

    gpus = tf.config.list_physical_devices("GPU")
    memory_growth_verified = False
    if args.lane == "f32tf32":
        if not gpus:
            raise SystemExit(
                "FAIL-CLOSED: f32tf32 lane requires a GPU; none visible"
            )
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        memory_growth_verified = all(
            tf.config.experimental.get_memory_growth(gpu) for gpu in gpus
        )
        if not memory_growth_verified:
            raise SystemExit("FAIL-CLOSED: memory growth not verified")
        tf.config.experimental.enable_tensor_float_32_execution(True)
        dtype = tf.float32
        device = "/GPU:0"
    else:
        if gpus:
            raise SystemExit(
                "FAIL-CLOSED: f64cpu anchor must be launched with GPUs "
                "hidden (CUDA_VISIBLE_DEVICES=-1) so the CPU claim is real"
            )
        dtype = tf.float64
        device = "/CPU:0"

    from bayesfilter.highdim.ledh_canonical_neutra_targets_tf import (
        make_canonical_neutra_target,
    )
    from bayesfilter.highdim.ledh_canonical_models_tf import (
        austria_sir_canonical_model,
    )
    from bayesfilter.highdim.ledh_canonical_filter_tf import (
        CanonicalModelCallbacks,
        canonical_value_and_diagnostics,
    )
    from bayesfilter.highdim.models import zhao_cui_sir_austria_model

    with tf.device("/CPU:0"):
        target = make_canonical_neutra_target(
            "austria_sir", particle_count=1008
        )
    theta0 = tf.constant([0.0, 0.0, 0.0], dtype)
    model, _sd = austria_sir_canonical_model(theta0, dtype=dtype)
    observations = tf.cast(target.observations, dtype)
    initial_mean = tf.cast(zhao_cui_sir_austria_model().initial_mean, dtype)
    variance = 100.0

    def transition_log_density_fn(points, ancestors, _t):
        mean = model.transition_mean_fn(theta0, ancestors)
        residual = points - mean
        return -0.5 * (
            tf.reduce_sum(tf.square(residual), axis=1)
            + 18.0 * tf.constant(np.log(2.0 * np.pi), dtype)
        )

    def observation_log_density_fn(points, observation, _t):
        observed = model.observation_fn(points)
        residual = observation[None, :] - observed
        return -0.5 * (
            tf.reduce_sum(tf.square(residual), axis=1) / variance
            + 9.0 * tf.constant(np.log(2.0 * np.pi * variance), dtype)
        )

    callbacks = CanonicalModelCallbacks(
        model_id=f"austria_sir_q2_kc_{args.lane}",
        state_dim=18,
        observation_dim=9,
        transition_mean_fn=lambda p, t: model.transition_mean_fn(theta0, p),
        transition_log_density_fn=transition_log_density_fn,
        process_noise_covariance=model.process_covariance,
        process_noise_covariance_provenance="model_exact",
        observation_fn=lambda p, t: model.observation_fn(p),
        observation_jacobian_fn=lambda p, t: model.observation_jacobian_fn(p),
        observation_covariance=model.observation_covariance,
        observation_log_density_fn=observation_log_density_fn,
        initial_mean=initial_mean,
        initial_covariance=tf.eye(18, dtype=dtype),
        initial_covariance_provenance="model_exact",
    )

    cells = {}
    veto_fired = False
    with tf.device(device):
        for k in stages:
            for cap in caps:
                for seed in seeds:
                    cell_start = time.time()
                    result = canonical_value_and_diagnostics(
                        callbacks,
                        observations,
                        particle_count=1008,
                        seed=seed,
                        flow_substeps=16,
                        temper_stages=k,
                        annealed_resampling=(k > 1),
                        flow_prior_cap=cap,
                        resample_seed=seed,
                    )
                    ess = result["per_step_ess"].numpy()
                    value = float(result["value"].numpy())
                    valid = bool(result["program_valid"].numpy())
                    finite = bool(
                        np.all(np.isfinite(ess)) and np.isfinite(value)
                    )
                    if not (valid and finite):
                        veto_fired = True
                    cells[f"k{k}_cap{cap:g}_seed{seed}"] = {
                        "k": k,
                        "cap": cap if np.isfinite(cap) else "inf",
                        "seed": seed,
                        "value": value,
                        "program_valid": valid,
                        "finite": finite,
                        "min_ess": float(ess.min()),
                        "min_ess_fraction": float(ess.min() / 1008.0),
                        "ess": [round(float(e), 1) for e in ess],
                        "wall_seconds": round(time.time() - cell_start, 1),
                    }
                    print(
                        f"[cell] k={k} cap={cap:g} seed={seed} "
                        f"valid={valid} min_ess_frac="
                        f"{ess.min() / 1008.0:.3f} "
                        f"wall={time.time() - cell_start:.0f}s",
                        flush=True,
                    )

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=_ROOT, capture_output=True,
        text=True,
    ).stdout.strip()
    payload = {
        "schema": "bayesfilter.q2_annealed_kc_response_surface.v1",
        "plan": "docs/plans/bayesfilter-q2-calibration-campaign-plan-2026-08-24.md",
        "lane": args.lane,
        "manifest": {
            "commit": commit,
            "command": " ".join(sys.argv),
            "conda_env": os.environ.get("CONDA_DEFAULT_ENV", "unknown"),
            "cuda_visible_devices": os.environ.get(
                "CUDA_VISIBLE_DEVICES", "unset"
            ),
            "gpus_visible": [g.name for g in gpus],
            "memory_growth_verified": memory_growth_verified,
            "tf32_enabled": bool(
                tf.config.experimental.tensor_float_32_execution_enabled()
            ),
            "dtype": dtype.name,
            "device": device,
            "particle_count": 1008,
            "flow_substeps": 16,
            "stages": stages,
            "caps": [c if np.isfinite(c) else "inf" for c in caps],
            "seeds": seeds,
            "wall_seconds": round(time.time() - started, 1),
        },
        "hard_veto_fired": veto_fired,
        "cells": cells,
    }
    out_dir = os.path.join(
        _ROOT, "docs", "benchmarks", "q2_calibration_20260824",
        f"kc_surface_{args.lane}_{args.tag}_{int(started)}",
    )
    os.makedirs(out_dir, exist_ok=False)
    out_path = os.path.join(out_dir, "result.json")
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1)
    print(f"[done] veto={veto_fired} artifact={out_path}", flush=True)


if __name__ == "__main__":
    main()
