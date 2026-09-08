#!/usr/bin/env python3
"""Quick progress check for SQMC 4-route campaign."""

import json
from pathlib import Path

base_dir = Path("docs/benchmarks/artifacts/sqmc-rerun-corrected-filter-20260906")

routes = [
    (1, "repaired_permutation", "Baseline (pre-existing)"),
    (3, "iid_dual_cap", "Run 1"),
    (4, "previous_inverse_cdf", "Run 2"),
    (5, "repaired_fixed_previous_controls", "Run 3"),
]

print("="*80)
print("SQMC 4-Route Campaign Progress")
print("="*80)

total_complete = 0
total_cells = 64

for attempt, route_name, label in routes:
    attempt_dir = base_dir / f"claim_attempt{attempt:02d}"
    result_file = attempt_dir / "result.json"
    checkpoint_file = attempt_dir / "checkpoint.json"

    if result_file.exists():
        result = json.load(open(result_file))
        n_seeds = len(result['rows'])
        status = f"✓ COMPLETE ({n_seeds}/16)"
        total_complete += n_seeds
    elif checkpoint_file.exists():
        checkpoint = json.load(open(checkpoint_file))
        n_seeds = len(checkpoint['rows'])
        status = f"⋯ IN PROGRESS ({n_seeds}/16)"
        total_complete += n_seeds
    elif attempt_dir.exists():
        status = "⋯ STARTING (0/16)"
    else:
        status = "○ NOT STARTED (0/16)"

    print(f"{label:30s} {route_name:40s} {status}")

print("="*80)
print(f"Overall progress: {total_complete}/{total_cells} cells ({100*total_complete/total_cells:.1f}%)")
print("="*80)
