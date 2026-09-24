#!/usr/bin/env python3
"""Phase 4: route comparison across all tuned routes.

Evaluates each route at its own tuned controls on the same claim seeds, so the
comparison isolates the route choice while holding the tuning effort constant.

Unlike Phase 3 (which compares tuned vs untuned WITHIN one route), this compares
ACROSS routes, each at its own best-found controls.

Not concluded: production readiness, whether any route beats a bootstrap PF at
equal cost (Heuristic Dominance Gate), or generalization beyond this
model/horizon/N/dtype.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, "/home/chakwong/python/src")

import tensorflow as tf

import run_sqmc_tuning as R

SCHEMA = "bayesfilter.sqmc_route_comparison.v1"
ARTIFACT_ROOT = Path("docs/benchmarks/artifacts/sqmc-route-comparison-20260913")

# Disjoint from the tuning seeds (50001-50004).
CLAIM_SEEDS = tuple(range(97701, 97717))


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unavailable"


def _evaluate_route(
    *,
    route: str,
    controls: Dict[str, Any],
    observations: tf.Tensor,
    theta: tf.Tensor,
    oracle_score: tf.Tensor,
    seeds: List[int],
    horizon: int,
    particle_count: int,
) -> List[Dict[str, Any]]:
    rows = []
    for seed in seeds:
        started = time.perf_counter()
        result = R._evaluate_controls(
            route=route,
            controls=controls,
            observations=observations,
            theta=theta,
            oracle_score=oracle_score,
            seed=seed,
            horizon=horizon,
            particle_count=particle_count,
        )
        result["seed"] = seed
        result["route"] = route
        result["wall_seconds"] = time.perf_counter() - started
        rows.append(result)
        status = "ok" if result.get("valid") else f"INVALID ({result.get('error')})"
        print(
            f"  [{route:26s}] seed {seed}: "
            + (
                f"L2={result['score_l2_error']:.4f} "
                f"cos={result['cosine_similarity']:.7f}"
                if result.get("valid")
                else status
            ),
            flush=True,
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="SQMC route comparison")
    parser.add_argument(
        "--tuning-artifacts",
        nargs="+",
        required=True,
        help="Paths to tuning_artifact.json for each route",
    )
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--particles", type=int, default=1008)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(CLAIM_SEEDS),
        help="Claim seeds; must be disjoint from all tuning seeds",
    )
    parser.add_argument("--tag", default="final")
    args = parser.parse_args()

    # Load each route's tuned controls
    routes: List[Dict[str, Any]] = []
    all_tuning_seeds = set()
    for artifact_path in args.tuning_artifacts:
        artifact = json.loads(Path(artifact_path).read_text())
        routes.append(
            {
                "route": artifact["route"],
                "controls": artifact["best_controls"],
                "artifact_path": str(artifact_path),
                "tuning_seeds": set(artifact.get("tuning_seeds", ())),
            }
        )
        all_tuning_seeds.update(artifact.get("tuning_seeds", ()))

    overlap = all_tuning_seeds.intersection(args.seeds)
    if overlap:
        raise SystemExit(
            f"claim seeds overlap tuning seeds {sorted(overlap)}: selecting and "
            "evaluating on the same draw does not measure accuracy"
        )

    print("=" * 78)
    print("PHASE 4: ROUTE COMPARISON (each at its tuned controls)")
    print("=" * 78)
    print(f"Routes: {[r['route'] for r in routes]}")
    print(f"All tuning seeds: {sorted(all_tuning_seeds)}")
    print(f"Claim seeds: {args.seeds} (disjoint)")
    print(f"Backend: float64_gpu")
    print()

    theta = tf.constant([0.9, 0.8, 0.7, 0.6, 0.8], dtype=R.DTYPE)
    observations = R._frozen_observations(args.horizon)
    oracle_score = R._oracle_score(observations, theta)
    print(f"oracle score: {oracle_score.numpy()}")
    print(f"oracle norm: {float(tf.norm(oracle_score).numpy()):.6f}\n")

    started = time.perf_counter()
    all_rows = []
    for route_spec in routes:
        print(f"Evaluating route: {route_spec['route']}")
        for k, v in sorted(route_spec["controls"].items()):
            print(f"  {k}: {v}")
        print()

        rows = _evaluate_route(
            route=route_spec["route"],
            controls=route_spec["controls"],
            observations=observations,
            theta=theta,
            oracle_score=oracle_score,
            seeds=args.seeds,
            horizon=args.horizon,
            particle_count=args.particles,
        )
        all_rows.extend(rows)
        print()

    elapsed = time.perf_counter() - started

    # Group by route and compute aggregates
    by_route: Dict[str, List[Dict]] = {}
    for row in all_rows:
        by_route.setdefault(row["route"], []).append(row)

    valid_by_route = {
        route: [r for r in rows if r.get("valid")] for route, rows in by_route.items()
    }
    dropped_by_route = {
        route: [r["seed"] for r in rows if not r.get("valid")]
        for route, rows in by_route.items()
    }

    print("=" * 78)
    print("ROUTE COMPARISON SUMMARY")
    print("=" * 78)
    for route in sorted(by_route.keys()):
        valid = valid_by_route[route]
        if not valid:
            print(f"{route}: all seeds invalid")
            continue

        l2s = [r["score_l2_error"] for r in valid]
        cosines = [r["cosine_similarity"] for r in valid]
        rel_norms = [r["relative_norm_error"] for r in valid]
        hmcs = [r["induced_hmc_error"] for r in valid]

        print(f"\n{route}:")
        print(f"  valid: {len(valid)}/{len(by_route[route])}")
        print(f"  L2 error:          {statistics.fmean(l2s):.4f}  [{min(l2s):.4f}, {max(l2s):.4f}]")
        print(f"  cosine similarity: {statistics.fmean(cosines):.7f}  [{min(cosines):.7f}, {max(cosines):.7f}]")
        print(f"  rel norm error:    {statistics.fmean(rel_norms):.5f}  [{min(rel_norms):.5f}, {max(rel_norms):.5f}]")
        print(f"  HMC error:         {statistics.fmean(hmcs):.7f}  [{min(hmcs):.7f}, {max(hmcs):.7f}]")

    # Rank by L2 error (primary metric)
    route_l2_means = {
        route: statistics.fmean([r["score_l2_error"] for r in valid])
        for route, valid in valid_by_route.items()
        if valid
    }
    ranked = sorted(route_l2_means.items(), key=lambda x: x[1])

    print()
    print("=" * 78)
    print("RANKING (by mean L2 error, lower is better)")
    print("=" * 78)
    for rank, (route, l2) in enumerate(ranked, 1):
        print(f"{rank}. {route:30s}  L2 = {l2:.4f}")

    print()
    print("=" * 78)
    print("WHAT THIS ESTABLISHES")
    print("=" * 78)
    print("- Ranks routes at their own tuned controls on the same claim seeds.")
    print("- The ranking isolates the route choice while holding tuning effort constant.")
    print()
    print("=" * 78)
    print("WHAT THIS DOES NOT ESTABLISH")
    print("=" * 78)
    print("- Not a statistical test of route superiority (no uncertainty intervals).")
    print("- Not a cost comparison (routes may differ in wall-clock per seed).")
    print("- Not production readiness, HMC convergence benefit, or route preference")
    print("  claim without a paired test.")
    print("- Does not satisfy the Heuristic Dominance Gate: whether any route beats")
    print("  a bootstrap PF at equal computational cost remains unanswered.")

    output_dir = ARTIFACT_ROOT / args.tag
    output_dir.mkdir(parents=True, exist_ok=True)

    def _jsonable(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [
            {k: v for k, v in row.items() if k not in ("fisher_scaled_errors",)}
            | {"fisher_scaled_errors": list(row.get("fisher_scaled_errors", []))}
            for row in rows
        ]

    artifact = {
        "schema": SCHEMA,
        "program": "sqmc_route_comparison",
        "question": (
            "Which route gives the lowest L2 error at its own tuned controls?"
        ),
        "model": "diagonal_lgssm_canonical",
        "observations_source": (
            "ledh_canonical_neutra_targets_tf._lgssm_frozen_observations"
        ),
        "horizon": args.horizon,
        "particle_count": args.particles,
        "dtype": R.DTYPE.name,
        "backend": "float64_gpu",
        "routes": [
            {
                "route": r["route"],
                "controls": r["controls"],
                "tuning_artifact": r["artifact_path"],
                "tuning_seeds": sorted(r["tuning_seeds"]),
            }
            for r in routes
        ],
        "claim_seeds": list(args.seeds),
        "claim_seeds_disjoint_from_tuning": True,
        "all_tuning_seeds": sorted(all_tuning_seeds),
        "oracle_score": [float(v) for v in oracle_score.numpy()],
        "route_summaries": {
            route: {
                "valid_count": len(valid),
                "total_count": len(by_route[route]),
                "dropped_seeds": dropped_by_route[route],
                "mean_l2_error": statistics.fmean([r["score_l2_error"] for r in valid])
                if valid
                else float("inf"),
                "mean_cosine_similarity": statistics.fmean(
                    [r["cosine_similarity"] for r in valid]
                )
                if valid
                else 0.0,
                "mean_relative_norm_error": statistics.fmean(
                    [r["relative_norm_error"] for r in valid]
                )
                if valid
                else float("inf"),
                "mean_induced_hmc_error": statistics.fmean(
                    [r["induced_hmc_error"] for r in valid]
                )
                if valid
                else float("inf"),
            }
            for route, valid in valid_by_route.items()
        },
        "ranking": [
            {"rank": rank, "route": route, "mean_l2_error": l2}
            for rank, (route, l2) in enumerate(ranked, 1)
        ],
        "cells": _jsonable(all_rows),
        "manifest": {
            "git_commit": _git("rev-parse", "HEAD"),
            "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "git_dirty": bool(_git("status", "--porcelain")),
            "command": " ".join(sys.argv),
            "conda_env": os.environ.get("CONDA_DEFAULT_ENV", "unknown"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
            "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER", "unset"),
            "gpu_memory_policy": R.GPU_POLICY_RECORD,
            "logical_gpus": [d.name for d in tf.config.list_logical_devices("GPU")],
            "python": platform.python_version(),
            "tensorflow": tf.__version__,
            "platform": platform.platform(),
            "wall_seconds": elapsed,
            "timestamp": datetime.now().isoformat(),
            "plan": "docs/plans/sqmc-master-program-2026-09-12.md",
        },
        "non_claims": [
            "No statistical test of route superiority.",
            "No cost comparison (routes may differ in per-seed wall-clock).",
            "No production readiness or HMC convergence benefit claim.",
            "Heuristic Dominance Gate not satisfied.",
            "No generalisation beyond this model, horizon, particle count, dtype.",
        ],
    }

    result_path = output_dir / "result.json"
    result_path.write_text(json.dumps(artifact, indent=2, default=str))
    print(f"\nArtifact: {result_path}")
    print(f"Wall time: {elapsed / 60.0:.1f} min")

    return 0


if __name__ == "__main__":
    sys.exit(main())
