#!/usr/bin/env python3
"""Isolate the process-noise confound in the RQMC initialization campaign.

The campaign runner draws initial noise and process noise from ONE tf.random
Generator. The mc arm consumes [N, d] from that generator for its initial cloud;
the sobol/halton/genut arms do not (their clouds come from scipy or a fixed
design). Process noise is therefore drawn from a DIFFERENT stream offset for mc
than for every RQMC arm, so the arms differ in process-noise trajectory as well
as in initialization.

This script holds the initial cloud fixed and varies only the process-noise
stream offset, which measures the confound directly.

  A: sobol_owen cloud + process noise at offset 0        (what the campaign ran)
  B: sobol_owen cloud + process noise at offset N*d      (mc's stream position)
  |A - B| is a pure process-noise effect with identical initialization.

Usage:
    python docs/benchmarks/diagnose_rqmc_process_noise_confound.py \
        --model ksc_sv_T10 \
        --tuning_artifact docs/benchmarks/artifacts/ledh_trust_region_ksc_sv_t10_20260903/result.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import tensorflow as tf

for _gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(_gpu, True)

from docs.benchmarks.run_rqmc_ledh_initialization import (  # noqa: E402
    N,
    _generate_initial_noise,
    _load_model_target,
    _load_tuning_artifact,
    _make_evaluator,
    replication_generator,
)

SEEDS = (98301, 98302, 98303)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tuning_artifact", required=True, type=Path)
    args = parser.parse_args()

    target = _load_model_target(args.model)
    controls = _load_tuning_artifact(args.tuning_artifact)
    evaluator = _make_evaluator(target, controls)

    state_dim = int(target["state_dim"])
    horizon = int(target["horizon"])
    observations = target["observations"]

    print(f"model={args.model}  state_dim={state_dim}  horizon={horizon}  N={N}")
    print(f"mc consumes {N * state_dim} normals for its initial cloud;")
    print("rqmc arms consume 0, so their process noise starts at stream offset 0.\n")

    def process_noise(seed: int, pre_advance: bool):
        rng = replication_generator(seed)
        if pre_advance:
            # Reproduce exactly what the mc arm's initial draw consumes.
            rng.normal([N, state_dim], dtype=tf.float32)
        return rng.normal([horizon, N, state_dim], dtype=tf.float32)

    def evaluate(initial, process):
        value, _score, _status = evaluator(
            target["theta"], observations, initial, process, target["design"]
        )
        return float(value.numpy())

    rows = []
    for seed in SEEDS:
        # Initial cloud held FIXED across A and B.
        sobol_cloud = _generate_initial_noise(
            "sobol_owen", state_dim, seed, replication_generator(seed)
        )
        a = evaluate(sobol_cloud, process_noise(seed, pre_advance=False))
        b = evaluate(sobol_cloud, process_noise(seed, pre_advance=True))

        # Same experiment for the mc cloud, for symmetry.
        mc_cloud = _generate_initial_noise(
            "mc", state_dim, seed, replication_generator(seed)
        )
        c = evaluate(mc_cloud, process_noise(seed, pre_advance=False))
        d = evaluate(mc_cloud, process_noise(seed, pre_advance=True))

        rows.append({
            "seed": seed,
            "sobol_offset0": a, "sobol_offsetNd": b, "sobol_delta": b - a,
            "mc_offset0": c, "mc_offsetNd": d, "mc_delta": d - c,
        })
        print(f"seed {seed}:")
        print(f"  sobol cloud: offset0={a:11.4f}  offsetNd={b:11.4f}  "
              f"process-noise effect={b - a:+.4f}")
        print(f"  mc    cloud: offset0={c:11.4f}  offsetNd={d:11.4f}  "
              f"process-noise effect={d - c:+.4f}")

    sob = np.array([r["sobol_delta"] for r in rows])
    mcd = np.array([r["mc_delta"] for r in rows])
    print("\n=== process-noise-only effect (initialization held fixed) ===")
    print(f"  sobol cloud: mean={sob.mean():+.4f}  |mean|={abs(sob.mean()):.4f}  "
          f"max|delta|={np.abs(sob).max():.4f}")
    print(f"  mc    cloud: mean={mcd.mean():+.4f}  |mean|={abs(mcd.mean()):.4f}  "
          f"max|delta|={np.abs(mcd).max():.4f}")

    # The campaign's as-run comparison: mc at offset N*d vs sobol at offset 0.
    as_run = np.array([r["sobol_offset0"] - r["mc_offsetNd"] for r in rows])
    # The clean comparison: both clouds at the SAME stream offset.
    clean = np.array([r["sobol_offset0"] - r["mc_offset0"] for r in rows])
    print("\n=== sobol_owen - mc ===")
    print(f"  as run by campaign (different offsets): mean={as_run.mean():+.4f}")
    print(f"  offset-matched (initialization only):   mean={clean.mean():+.4f}")
    print(f"  attributable to the confound:           {as_run.mean() - clean.mean():+.4f}")

    out = {
        "model": args.model,
        "state_dim": state_dim,
        "horizon": horizon,
        "particle_count": N,
        "rows": rows,
        "process_noise_only_effect": {
            "sobol_cloud_mean": float(sob.mean()),
            "mc_cloud_mean": float(mcd.mean()),
            "sobol_cloud_max_abs": float(np.abs(sob).max()),
            "mc_cloud_max_abs": float(np.abs(mcd).max()),
        },
        "sobol_minus_mc": {
            "as_run_different_offsets": float(as_run.mean()),
            "offset_matched_initialization_only": float(clean.mean()),
            "confound_contribution": float(as_run.mean() - clean.mean()),
        },
    }
    dest = ROOT / "docs/benchmarks/artifacts/rqmc_ledh_init_v1_20260904" / \
        f"process_noise_confound_{args.model}.json"
    dest.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {dest.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
