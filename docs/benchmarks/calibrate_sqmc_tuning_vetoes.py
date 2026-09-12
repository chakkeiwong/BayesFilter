#!/usr/bin/env python3
"""Veto no-fire calibration for the SQMC tuning hard constraints.

A fail-closed guard must not fire on a known-good configuration.  This check
evaluates the UNTUNED baseline's warm-start controls on several seeds and reports
whether the tuning hard constraints would reject that baseline.

Thresholds are imported from `run_sqmc_tuning` rather than restated here, so this
check cannot silently drift from the constraints actually enforced.

History: the first revision of the tuning runner used cosine >= 0.9995 -- the
baseline's two-seed observed MEAN -- as a hard per-seed veto.  This check showed
it firing on 3 of 4 baseline seeds, i.e. a descriptive statistic promoted to a
correctness criterion.  The thresholds now come from the principled-metrics
decision framework and apply to the seed-aggregated mean.

Debug/calibration only: not a research-decision run, no tuning selection here.
"""

import os
import statistics
import sys

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, "/home/chakwong/python/src")

import tensorflow as tf

from bayesfilter.runtime.gpu_memory_policy import (
    configure_tensorflow_gpu_memory_growth,
)

configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)

import run_sqmc_tuning as R  # noqa: E402

# Warm-start controls used by the UNTUNED baseline diagnostic.
BASELINE_CONTROLS = {
    "reset_epsilon": 8.0,
    "reset_sinkhorn_steps": 8,
    "reset_balance_steps": 8,
    "correction_strength": 0.2,
    "correction_steps": 4,
    "pairwise_strength": 0.02,
    "pairwise_steps": 4,
}

SEEDS = [50001, 50002, 50003, 50004]


def main() -> int:
    horizon = 20
    particle_count = 1008

    theta = tf.constant([0.9, 0.8, 0.7, 0.6, 0.8], dtype=R.DTYPE)
    observations = R._frozen_observations(horizon)
    oracle_score = R._oracle_score(observations, theta)

    print("Baseline warm-start controls (UNTUNED baseline):")
    for key, value in BASELINE_CONTROLS.items():
        print(f"  {key}: {value}")
    print(f"\nSeeds: {SEEDS}\n")

    rows = []
    for seed in SEEDS:
        result = R._evaluate_controls(
            route="iid_dual_cap",
            controls=BASELINE_CONTROLS,
            observations=observations,
            theta=theta,
            oracle_score=oracle_score,
            seed=seed,
            horizon=horizon,
            particle_count=particle_count,
        )
        rows.append((seed, result))
        if not result.get("valid"):
            print(f"seed {seed}: INVALID ({result.get('error')})")
            continue
        print(
            f"seed {seed}: L2={result['score_l2_error']:.4f} "
            f"cos={result['cosine_similarity']:.7f} "
            f"rel_norm={result['relative_norm_error']:.4f} "
            f"fisher_max={max(result['fisher_scaled_errors']):.4f} "
            f"veto={result.get('veto_reason')}"
        )

    valid = [r for _, r in rows if r.get("valid")]
    if not valid:
        print("\nAll baseline cells invalid: cannot calibrate vetoes.")
        return 1

    cosines = [r["cosine_similarity"] for r in valid]
    rel_norms = [r["relative_norm_error"] for r in valid]
    fisher_maxes = [max(r["fisher_scaled_errors"]) for r in valid]
    l2s = [r["score_l2_error"] for r in valid]

    print("\n--- baseline summary ---")
    print(f"L2:        min={min(l2s):.4f} mean={statistics.fmean(l2s):.4f} max={max(l2s):.4f}")
    print(f"cosine:    min={min(cosines):.7f} mean={statistics.fmean(cosines):.7f} max={max(cosines):.7f}")
    print(f"rel_norm:  min={min(rel_norms):.4f} mean={statistics.fmean(rel_norms):.4f} max={max(rel_norms):.4f}")
    print(f"fisher_max:min={min(fisher_maxes):.4f} mean={statistics.fmean(fisher_maxes):.4f} max={max(fisher_maxes):.4f}")

    # The tuning constraints are enforced on the seed-aggregated mean, so the
    # no-fire check must be evaluated the same way.
    mean_cosine = statistics.fmean(cosines)
    mean_rel_norm = statistics.fmean(rel_norms)
    max_fisher = max(fisher_maxes)

    print(
        f"\n--- veto no-fire check (aggregate; cos>={R.COSINE_VETO}, "
        f"rel_norm<={R.REL_NORM_VETO}, fisher<={R.FISHER_VETO}) ---"
    )
    failures = []
    if mean_cosine < R.COSINE_VETO:
        failures.append(f"cosine {mean_cosine:.7f} < {R.COSINE_VETO}")
    if mean_rel_norm > R.REL_NORM_VETO:
        failures.append(f"rel_norm {mean_rel_norm:.4f} > {R.REL_NORM_VETO}")
    if max_fisher > R.FISHER_VETO:
        failures.append(f"fisher {max_fisher:.4f} > {R.FISHER_VETO}")

    if failures:
        print(f"VETO FIRES on known-good baseline: {failures}")
        print("=> Thresholds are MISCALIBRATED for this scope and must be revised")
        print("   before they can be used as hard constraints in tuning.")
        return 2

    print("No veto fires on the known-good baseline aggregate: thresholds are usable.")

    # Report per-seed margin so a threshold sitting just above the noise band is
    # visible rather than implicit.
    worst_seed_cosine = min(cosines)
    print(
        f"per-seed worst cosine = {worst_seed_cosine:.7f} "
        f"(margin to veto: {worst_seed_cosine - R.COSINE_VETO:+.7f})"
    )
    if worst_seed_cosine < R.COSINE_VETO:
        print(
            "NOTE: an individual baseline seed falls below the threshold; this is "
            "why the constraint is applied to the aggregate, not per-seed."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
