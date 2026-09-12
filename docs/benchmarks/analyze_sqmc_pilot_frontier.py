#!/usr/bin/env python3
"""Read a SQMC tuning artifact and report what its Pareto frontier does and does
not establish.

Two structural facts govern the reading, and both are checked here rather than
left to prose:

1. The warm-start baseline controls are themselves a grid point (index 28).  The
   frontier therefore contains something at least as good as the baseline on the
   tuning seeds by construction.  "Frontier L2 < baseline L2" is not evidence of
   a tuning benefit; it is guaranteed up to seed noise.

2. Selecting the argmin over 54 configurations scored on a handful of seeds
   biases the selected L2 downward (winner's curse).  The selected config's
   tuning-seed L2 is a biased-low estimate of its true L2, so it must not be
   quoted as the tuning improvement.  Phase 3 re-evaluates on disjoint claim
   seeds for exactly this reason.

This script only reads an existing artifact; it runs no model code.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Dict

# Grid index of the warm-start baseline controls, verified against _tuning_grid().
# The grid was reduced from 54 to 18 configurations after the Sinkhorn/balance
# step dimension was measured to be scientifically inert, which moved the
# baseline from index 28 to index 10.  The analyser falls back to a controls
# search if this index does not hold the baseline, so a future grid change
# degrades to a warning rather than a silent mislabel.
BASELINE_GRID_INDEX = 10

BASELINE_CONTROLS = {
    "reset_epsilon": 8.0,
    "reset_sinkhorn_steps": 8,
    "reset_balance_steps": 8,
    "correction_strength": 0.2,
    "correction_steps": 4,
    "pairwise_strength": 0.02,
    "pairwise_steps": 4,
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyse a SQMC tuning frontier")
    parser.add_argument("artifact", help="Path to tuning_artifact.json")
    args = parser.parse_args()

    artifact: Dict[str, Any] = json.loads(Path(args.artifact).read_text())
    results = artifact["all_results"]
    frontier = artifact.get("pareto_frontier", [])
    best = artifact["best_metrics"]

    print("=" * 78)
    print("SQMC TUNING FRONTIER ANALYSIS")
    print("=" * 78)
    print(f"artifact:      {args.artifact}")
    print(f"route:         {artifact['route']}")
    print(f"model:         {artifact['model']} T={artifact['horizon']} N={artifact['particle_count']}")
    print(f"backend:       {artifact.get('backend')}")
    print(f"tuning seeds:  {artifact.get('tuning_seeds')}")
    print(f"grid size:     {artifact.get('grid_size')}")
    print(f"constraints:   {artifact.get('hard_constraints')}")
    print()

    # Validity accounting over the whole grid.
    total = len(results)
    all_invalid = [r for r in results if r["valid_fraction"] == 0.0]
    partial = [r for r in results if 0.0 < r["valid_fraction"] < 1.0]
    print(f"grid cells:            {total}")
    print(f"  fully valid:         {sum(1 for r in results if r['valid_fraction'] == 1.0)}")
    print(f"  partially valid:     {len(partial)}")
    print(f"  all-seed invalid:    {len(all_invalid)}")
    print(f"  on Pareto frontier:  {len(frontier)}")
    rejected = total - len(all_invalid) - len(frontier)
    print(f"  valid but dominated or constraint-rejected: {rejected}")
    print()

    # The structural self-check: where does the baseline itself sit?
    baseline_row = None
    if BASELINE_GRID_INDEX < total:
        candidate = results[BASELINE_GRID_INDEX]
        if candidate["controls"] == BASELINE_CONTROLS:
            baseline_row = candidate
        else:
            print("WARNING: grid index 28 does not hold the baseline controls;")
            print(f"  found: {candidate['controls']}")
            for row in results:
                if row["controls"] == BASELINE_CONTROLS:
                    baseline_row = row
                    break

    print("=" * 78)
    print("BASELINE-IN-GRID SELF-CHECK")
    print("=" * 78)
    if baseline_row is None:
        print("Baseline controls NOT found in the grid.")
        print("=> The frontier is not guaranteed to contain a baseline-or-better")
        print("   point, and any comparison to the baseline is across designs.")
    else:
        print(f"Baseline is grid index {results.index(baseline_row)} (evaluated on the same seeds).")
        print(f"  baseline L2:     {baseline_row['mean_score_l2']:.4f}")
        print(f"  baseline cosine: {baseline_row['mean_cosine_similarity']:.7f}")
        print(f"  selected L2:     {best['mean_score_l2_error']:.4f}")
        print(f"  selected cosine: {best['mean_cosine_similarity']:.7f}")
        delta = best["mean_score_l2_error"] - baseline_row["mean_score_l2"]
        print(f"  selected - baseline L2: {delta:+.4f}")
        on_frontier = any(
            entry["config_idx"] == results.index(baseline_row) for entry in frontier
        )
        print(f"  baseline itself on frontier: {on_frontier}")
        if delta > 0.0:
            print()
            print("  NOTE: the selected config has HIGHER L2 than the baseline on the")
            print("  tuning seeds. Since selection is lexicographic with L2 first, this")
            print("  means the baseline was excluded by a hard constraint or dominated")
            print("  on another objective. Inspect before treating the selection as an")
            print("  improvement.")

    # Frontier spread: is there a real trade-off, or one dominant point?
    print()
    print("=" * 78)
    print("FRONTIER")
    print("=" * 78)
    if not frontier:
        print("Empty frontier.")
    else:
        l2s = [e["objectives"]["l2_error"] for e in frontier]
        coss = [1.0 - e["objectives"]["direction_error"] for e in frontier]
        print(f"frontier size: {len(frontier)}")
        print(f"  L2     range: {min(l2s):.4f} .. {max(l2s):.4f}  (spread {max(l2s) - min(l2s):.4f})")
        print(f"  cosine range: {min(coss):.7f} .. {max(coss):.7f}")
        print()
        for entry in sorted(frontier, key=lambda e: e["objectives"]["l2_error"]):
            obj = entry["objectives"]
            controls = entry["controls"]
            print(
                f"  idx {entry['config_idx']:>2}: L2={obj['l2_error']:.4f} "
                f"cos={1.0 - obj['direction_error']:.7f} "
                f"relnorm={obj['rel_norm_error']:.4f} "
                f"fisher={obj['fisher_scaled']:.4f} "
                f"| eps={controls['reset_epsilon']} "
                f"steps={controls['reset_sinkhorn_steps']} "
                f"diag={controls['correction_strength']} "
                f"pair={controls['pairwise_strength']}"
            )

    # Spread of L2 across the whole valid grid, to judge whether the controls
    # move the metric at all relative to seed noise.
    valid_l2 = [
        r["mean_score_l2"]
        for r in results
        if r["valid_fraction"] > 0.5 and r["mean_score_l2"] != float("inf")
    ]
    print()
    print("=" * 78)
    print("DOES THE CONTROL GRID MOVE L2 AT ALL?")
    print("=" * 78)
    if len(valid_l2) >= 2:
        print(f"valid configs:      {len(valid_l2)}")
        print(f"L2 across grid:     {min(valid_l2):.4f} .. {max(valid_l2):.4f}")
        print(f"L2 grid spread:     {max(valid_l2) - min(valid_l2):.4f}")
        print(f"L2 grid mean:       {statistics.fmean(valid_l2):.4f}")
        if len(valid_l2) >= 3:
            print(f"L2 grid stdev:      {statistics.stdev(valid_l2):.4f}")
        print()
        print("Compare the grid spread against the baseline's own seed-to-seed")
        print("spread (L2 1.3255..1.7043, i.e. ~0.38 on 4 seeds). If the grid")
        print("spread is not larger than the seed spread, the controls are not")
        print("moving L2 detectably at this seed count and no configuration")
        print("ranking is supportable.")
    else:
        print("Too few valid configs to assess grid spread.")

    print()
    print("=" * 78)
    print("WHAT THIS ARTIFACT DOES NOT ESTABLISH")
    print("=" * 78)
    print("- Not a tuning benefit. The baseline is a grid point, so a frontier at")
    print("  or below baseline L2 is guaranteed on these seeds by construction.")
    print("- Not an unbiased estimate of the selected config's L2. Selecting the")
    print("  argmin over the grid on these seeds biases it low (winner's curse);")
    print("  the unbiased read requires the disjoint claim seeds in Phase 3.")
    print("- Not a ranking among frontier configs. Pareto membership means")
    print("  non-dominated, not better.")
    print("- Not a route comparison. One route is tuned here.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
