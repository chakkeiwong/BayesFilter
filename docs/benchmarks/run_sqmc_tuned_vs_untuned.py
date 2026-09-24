#!/usr/bin/env python3
"""Phase 3: paired TUNED vs UNTUNED comparison on the canonical LGSSM oracle.

Answers one question: does the exact-scope tuned control set reduce analytical
score error relative to the warm-start baseline, without degrading the score
quality metrics that were already good?

Design notes that make the comparison interpretable:

- PAIRED. Both arms run on the same seeds and the same frozen canonical target,
  so per-seed differences cancel the dominant source of variation. The reported
  interval is over paired differences, not over arm means.
- Claim seeds are DISJOINT from the tuning seeds (50001-50016). Selecting
  controls on a seed and then evaluating on that same seed measures fit to that
  draw, not accuracy.
- The comparator is the exact Kalman score, so "improvement" always means
  "closer to the exact reference", never "closer to some other estimator".

Bootstrap intervals over a modest number of paired seeds are descriptive-to-
suggestive. A non-overlapping interval nominates a real effect; it does not by
itself license a production default change.
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
from typing import Any, Dict, List, Sequence

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, "/home/chakwong/python/src")

import numpy as np
import tensorflow as tf

import run_sqmc_tuning as R

SCHEMA = "bayesfilter.sqmc_tuned_vs_untuned.v1"
ARTIFACT_ROOT = Path(
    "docs/benchmarks/artifacts/sqmc-tuned-vs-untuned-lgssm-20260913"
)

# Warm-start controls used by the UNTUNED baseline diagnostic.  This is the
# comparator arm, frozen.
UNTUNED_CONTROLS = {
    "reset_epsilon": 8.0,
    "reset_sinkhorn_steps": 8,
    "reset_balance_steps": 8,
    "correction_strength": 0.2,
    "correction_steps": 4,
    "pairwise_strength": 0.02,
    "pairwise_steps": 4,
}

# Disjoint from the tuning seeds (50001-50016).
CLAIM_SEEDS = tuple(range(97701, 97717))

METRICS = (
    "score_l2_error",
    "cosine_similarity",
    "relative_norm_error",
    "induced_hmc_error",
)

# For each metric: does a LOWER value mean better agreement with the oracle?
LOWER_IS_BETTER = {
    "score_l2_error": True,
    "cosine_similarity": False,
    "relative_norm_error": True,
    "induced_hmc_error": True,
}


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unavailable"


def _paired_bootstrap(
    differences: Sequence[float],
    *,
    resamples: int = 10000,
    seed: int = 20260913,
) -> Dict[str, float]:
    """Percentile bootstrap CI for the mean paired difference."""
    values = np.asarray(differences, dtype=np.float64)
    generator = np.random.default_rng(seed)
    count = values.size
    means = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        means[index] = float(
            np.mean(values[generator.integers(0, count, size=count)])
        )
    return {
        "mean_difference": float(np.mean(values)),
        "ci_low": float(np.percentile(means, 2.5)),
        "ci_high": float(np.percentile(means, 97.5)),
        "resamples": resamples,
    }


def _sign_test(differences: Sequence[float]) -> Dict[str, Any]:
    """Two-sided exact sign test on paired differences (ties dropped)."""
    from math import comb

    negative = sum(1 for value in differences if value < 0.0)
    positive = sum(1 for value in differences if value > 0.0)
    trials = negative + positive
    if trials == 0:
        return {"n_effective": 0, "p_value": 1.0, "n_negative": 0, "n_positive": 0}
    extreme = min(negative, positive)
    tail = sum(comb(trials, k) for k in range(extreme + 1)) / (2**trials)
    return {
        "n_effective": trials,
        "n_negative": negative,
        "n_positive": positive,
        "p_value": float(min(1.0, 2.0 * tail)),
    }


def _evaluate_arm(
    *,
    label: str,
    route: str,
    controls: Dict[str, Any],
    observations: tf.Tensor,
    theta: tf.Tensor,
    oracle_score: tf.Tensor,
    seeds: Sequence[int],
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
        result["arm"] = label
        result["wall_seconds"] = time.perf_counter() - started
        rows.append(result)
        status = "ok" if result.get("valid") else f"INVALID ({result.get('error')})"
        print(
            f"  [{label}] seed {seed}: "
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
    parser = argparse.ArgumentParser(description="TUNED vs UNTUNED comparison")
    parser.add_argument(
        "--tuning-artifact",
        required=True,
        help="Path to the tuning_artifact.json whose best_controls define the TUNED arm",
    )
    parser.add_argument("--route", default="iid_dual_cap")
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--particles", type=int, default=1008)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(CLAIM_SEEDS),
        help="Claim seeds; must be disjoint from the tuning seeds",
    )
    parser.add_argument("--tag", default="attempt01")
    args = parser.parse_args()

    artifact_path = Path(args.tuning_artifact)
    tuning_artifact = json.loads(artifact_path.read_text())
    tuned_controls = tuning_artifact["best_controls"]
    tuning_seeds = set(tuning_artifact.get("tuning_seeds", ()))

    overlap = tuning_seeds.intersection(args.seeds)
    if overlap:
        raise SystemExit(
            f"claim seeds overlap tuning seeds {sorted(overlap)}: selecting and "
            "evaluating on the same draw does not measure accuracy"
        )

    print("=" * 78)
    print("PHASE 3: TUNED vs UNTUNED (paired, exact Kalman oracle comparator)")
    print("=" * 78)
    print(f"route:            {args.route}")
    print(f"tuning artifact:  {artifact_path}")
    print(f"tuning seeds:     {sorted(tuning_seeds)}")
    print(f"claim seeds:      {args.seeds} (disjoint)")
    print(f"backend:          {tuning_artifact.get('backend')}")
    print(f"dtype:            {R.DTYPE.name}")
    print("\nUNTUNED (warm-start) controls:")
    for key, value in sorted(UNTUNED_CONTROLS.items()):
        print(f"  {key}: {value}")
    print("\nTUNED controls (from artifact best_controls):")
    for key, value in sorted(tuned_controls.items()):
        print(f"  {key}: {value}")
    print()

    theta = tf.constant([0.9, 0.8, 0.7, 0.6, 0.8], dtype=R.DTYPE)
    observations = R._frozen_observations(args.horizon)
    oracle_score = R._oracle_score(observations, theta)
    print(f"oracle score: {oracle_score.numpy()}")
    print(f"oracle score norm: {float(tf.norm(oracle_score).numpy()):.6f}\n")

    started = time.perf_counter()
    untuned_rows = _evaluate_arm(
        label="untuned",
        route=args.route,
        controls=UNTUNED_CONTROLS,
        observations=observations,
        theta=theta,
        oracle_score=oracle_score,
        seeds=args.seeds,
        horizon=args.horizon,
        particle_count=args.particles,
    )
    tuned_rows = _evaluate_arm(
        label="tuned",
        route=args.route,
        controls=tuned_controls,
        observations=observations,
        theta=theta,
        oracle_score=oracle_score,
        seeds=args.seeds,
        horizon=args.horizon,
        particle_count=args.particles,
    )
    elapsed = time.perf_counter() - started

    # Pair strictly by seed, keeping only seeds valid in BOTH arms.
    untuned_by_seed = {row["seed"]: row for row in untuned_rows}
    tuned_by_seed = {row["seed"]: row for row in tuned_rows}
    paired_seeds = [
        seed
        for seed in args.seeds
        if untuned_by_seed[seed].get("valid") and tuned_by_seed[seed].get("valid")
    ]
    dropped = [seed for seed in args.seeds if seed not in paired_seeds]

    print(f"\npaired seeds: {len(paired_seeds)}/{len(args.seeds)}")
    if dropped:
        print(f"dropped (invalid in at least one arm): {dropped}")

    comparison: Dict[str, Any] = {}
    for metric in METRICS:
        untuned_values = [untuned_by_seed[s][metric] for s in paired_seeds]
        tuned_values = [tuned_by_seed[s][metric] for s in paired_seeds]
        # difference = tuned - untuned
        differences = [t - u for t, u in zip(tuned_values, untuned_values)]
        boot = _paired_bootstrap(differences)
        sign = _sign_test(differences)
        lower_better = LOWER_IS_BETTER[metric]
        # "favours tuned" means the difference moves in the improving direction
        improving = (
            boot["ci_high"] < 0.0 if lower_better else boot["ci_low"] > 0.0
        )
        degrading = (
            boot["ci_low"] > 0.0 if lower_better else boot["ci_high"] < 0.0
        )
        comparison[metric] = {
            "untuned_mean": statistics.fmean(untuned_values),
            "tuned_mean": statistics.fmean(tuned_values),
            "untuned_min": min(untuned_values),
            "untuned_max": max(untuned_values),
            "tuned_min": min(tuned_values),
            "tuned_max": max(tuned_values),
            "paired_difference": boot,
            "sign_test": sign,
            "lower_is_better": lower_better,
            "interval_excludes_zero": bool(improving or degrading),
            "direction": (
                "favours_tuned"
                if improving
                else "favours_untuned"
                if degrading
                else "indistinguishable"
            ),
        }

    print("\n" + "=" * 78)
    print("PAIRED COMPARISON (difference = tuned - untuned)")
    print("=" * 78)
    for metric, row in comparison.items():
        boot = row["paired_difference"]
        print(f"\n{metric} ({'lower better' if row['lower_is_better'] else 'higher better'}):")
        print(f"  untuned mean: {row['untuned_mean']:.7f}  [{row['untuned_min']:.7f}, {row['untuned_max']:.7f}]")
        print(f"  tuned   mean: {row['tuned_mean']:.7f}  [{row['tuned_min']:.7f}, {row['tuned_max']:.7f}]")
        print(
            f"  paired diff:  {boot['mean_difference']:+.7f}  "
            f"95% CI [{boot['ci_low']:+.7f}, {boot['ci_high']:+.7f}]"
        )
        print(
            f"  sign test:    n={row['sign_test']['n_effective']} "
            f"neg={row['sign_test']['n_negative']} pos={row['sign_test']['n_positive']} "
            f"p={row['sign_test']['p_value']:.4f}"
        )
        print(f"  verdict:      {row['direction']}")

    # Constraint check on the tuned arm's aggregate, same thresholds as tuning.
    tuned_cosine = statistics.fmean(
        [tuned_by_seed[s]["cosine_similarity"] for s in paired_seeds]
    )
    tuned_rel_norm = statistics.fmean(
        [tuned_by_seed[s]["relative_norm_error"] for s in paired_seeds]
    )
    tuned_fisher_max = max(
        max(tuned_by_seed[s]["fisher_scaled_errors"]) for s in paired_seeds
    )
    constraint_failures = []
    if tuned_cosine < R.COSINE_VETO:
        constraint_failures.append(f"cosine {tuned_cosine:.7f} < {R.COSINE_VETO}")
    if tuned_rel_norm > R.REL_NORM_VETO:
        constraint_failures.append(f"rel_norm {tuned_rel_norm:.4f} > {R.REL_NORM_VETO}")
    if tuned_fisher_max > R.FISHER_VETO:
        constraint_failures.append(f"fisher {tuned_fisher_max:.4f} > {R.FISHER_VETO}")

    print("\n" + "=" * 78)
    print("TUNED ARM HARD-CONSTRAINT CHECK (aggregate, claim seeds)")
    print("=" * 78)
    if constraint_failures:
        print(f"VETO: {constraint_failures}")
    else:
        print(
            f"pass — cosine {tuned_cosine:.7f} >= {R.COSINE_VETO}, "
            f"rel_norm {tuned_rel_norm:.4f} <= {R.REL_NORM_VETO}, "
            f"fisher {tuned_fisher_max:.4f} <= {R.FISHER_VETO}"
        )

    l2 = comparison["score_l2_error"]
    cosine = comparison["cosine_similarity"]
    print("\n" + "=" * 78)
    print("READING")
    print("=" * 78)
    print(
        f"L2 (primary target): {l2['direction']} "
        f"({l2['untuned_mean']:.4f} -> {l2['tuned_mean']:.4f})"
    )
    print(
        f"cosine (must not degrade): {cosine['direction']} "
        f"({cosine['untuned_mean']:.7f} -> {cosine['tuned_mean']:.7f})"
    )
    print(
        "\nNot concluded: production readiness, HMC convergence benefit, route "
        "preference, or generalisation beyond LGSSM T="
        f"{args.horizon} N={args.particles} at {R.DTYPE.name}."
    )

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
        "program": "sqmc_tuned_vs_untuned_paired_lgssm",
        "question": (
            "Does the exact-scope tuned control set reduce analytical score error "
            "versus the warm-start baseline without degrading already-good score "
            "quality metrics?"
        ),
        "model": "diagonal_lgssm_canonical",
        "observations_source": (
            "ledh_canonical_neutra_targets_tf._lgssm_frozen_observations"
        ),
        "route": args.route,
        "horizon": args.horizon,
        "particle_count": args.particles,
        "dtype": R.DTYPE.name,
        "backend": tuning_artifact.get("backend"),
        "backend_note": tuning_artifact.get("backend_note"),
        "tuning_artifact": str(artifact_path),
        "tuning_artifact_git_commit": tuning_artifact.get("git_commit"),
        "tuning_seeds": sorted(tuning_seeds),
        "claim_seeds": list(args.seeds),
        "claim_seeds_disjoint_from_tuning": True,
        "paired_seeds": paired_seeds,
        "dropped_seeds": dropped,
        "untuned_controls": UNTUNED_CONTROLS,
        "tuned_controls": tuned_controls,
        "hard_constraints": {
            "cosine_similarity_min": R.COSINE_VETO,
            "relative_norm_error_max": R.REL_NORM_VETO,
            "fisher_scaled_error_max": R.FISHER_VETO,
            "applied_to": "seed_aggregated_mean",
        },
        "tuned_arm_constraint_failures": constraint_failures,
        "oracle_score": [float(v) for v in oracle_score.numpy()],
        "comparison": comparison,
        "cells": {
            "untuned": _jsonable(untuned_rows),
            "tuned": _jsonable(tuned_rows),
        },
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
            "No production readiness claim.",
            "No HMC convergence or efficiency benefit claim.",
            "No route preference claim (single route evaluated here).",
            "No generalisation beyond this model, horizon, particle count, dtype.",
            (
                "Bootstrap intervals over "
                f"{len(paired_seeds)} paired seeds are suggestive; they do not "
                "license a production default change."
            ),
        ],
    }

    result_path = output_dir / "result.json"
    result_path.write_text(json.dumps(artifact, indent=2, default=str))
    print(f"\nArtifact: {result_path}")
    print(f"Wall time: {elapsed / 60.0:.1f} min")

    return 2 if constraint_failures else 0


if __name__ == "__main__":
    sys.exit(main())
