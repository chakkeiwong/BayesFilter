#!/usr/bin/env python3
"""Analyze dense vs streaming systematic difference across claim-stage seeds."""
import json
import numpy as np
from pathlib import Path

# Load results
dense_path = Path("docs/benchmarks/artifacts/sqmc-rerun-corrected-filter-20260906/claim_attempt01/result.json")
streaming_path = Path("docs/benchmarks/artifacts/sqmc-rerun-corrected-filter-20260906/claim_attempt02/result.json")

if not dense_path.exists():
    print(f"✗ Dense results not found: {dense_path}")
    exit(1)
if not streaming_path.exists():
    print(f"✗ Streaming results not found: {streaming_path}")
    exit(1)

dense = json.load(open(dense_path))
streaming = json.load(open(streaming_path))

print(f"Dense rows: {len(dense['rows'])}")
print(f"Streaming rows: {len(streaming['rows'])}\n")

# Extract values by seed
dense_vals = {r['seed']: r['value'] for r in dense['rows']}
streaming_vals = {r['seed']: r['value'] for r in streaming['rows']}

common_seeds = sorted(set(dense_vals.keys()) & set(streaming_vals.keys()))
print(f"Common seeds: {len(common_seeds)}\n")

if not common_seeds:
    print("✗ No overlapping seeds!")
    exit(1)

# Compute differences
results = []
for seed in common_seeds:
    d = dense_vals[seed]
    s = streaming_vals[seed]
    diff = s - d
    results.append({'seed': seed, 'dense': d, 'streaming': s, 'diff': diff})
    print(f"Seed {seed}: dense={d:.6f}, streaming={s:.6f}, diff={diff:+.6f}")

diffs = [r['diff'] for r in results]

print("\n" + "="*70)
print("STATISTICAL ANALYSIS")
print("="*70)
print(f"\nMean difference (streaming - dense): {np.mean(diffs):+.6f}")
print(f"Std dev of differences: {np.std(diffs, ddof=1):.6f}")
print(f"Min difference: {min(diffs):+.6f}")
print(f"Max difference: {max(diffs):+.6f}")
print(f"Range: {max(diffs) - min(diffs):.6f}")

print("\n" + "="*70)
print("SYMMETRY TEST")
print("="*70)
positive = sum(1 for d in diffs if d > 0)
negative = sum(1 for d in diffs if d < 0)
zero = sum(1 for d in diffs if abs(d) < 1e-9)
print(f"\nPositive diffs (streaming > dense): {positive}")
print(f"Negative diffs (streaming < dense): {negative}")
print(f"Zero diffs (bitwise identical): {zero}")

mean_diff = np.mean(diffs)
std_diff = np.std(diffs, ddof=1)

print("\n" + "="*70)
print("CONCLUSION")
print("="*70)

if zero == len(diffs):
    print("\n✓ BITWISE IDENTICAL across all seeds")
    print("  No discrepancy exists.")
elif abs(mean_diff) < 0.01:
    print(f"\n✓ Error is SYMMETRIC (mean ≈ 0: {mean_diff:+.6f})")
    print("  → Pure floating-point roundoff, no systematic bias")
    print(f"  → Std dev: {std_diff:.6f}")
    if std_diff < 0.1:
        print("  → Low variance: consistent FP noise level")
    else:
        print("  → High variance: noise varies across seeds")
elif abs(mean_diff) < 0.1:
    print(f"\n⚠ Error is SMALL but NON-ZERO (mean = {mean_diff:+.6f})")
    print(f"  → Slight systematic bias, std dev: {std_diff:.6f}")
else:
    print(f"\n✗ Error is LARGE (mean = {mean_diff:+.6f})")
    print(f"  → Significant systematic difference, std dev: {std_diff:.6f}")
    print("  → Requires investigation or TF32 disable")

# Coefficient of variation
if mean_diff != 0:
    cv = std_diff / abs(mean_diff)
    print(f"\nCoefficient of variation (std/|mean|): {cv:.2f}")
    if cv < 0.5:
        print("  → Error is SYSTEMATIC (low relative variance)")
    else:
        print("  → Error is NOISY (high relative variance)")
