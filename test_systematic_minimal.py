#!/usr/bin/env python3
"""Minimal systematic test: dense vs streaming across 5 seeds at N=1008."""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import sys
sys.path.insert(0, '/home/chakwong/BayesFilter')

import subprocess
import json
from pathlib import Path

# Run the actual smoke runner 5 times with different seeds
seeds = [97701, 97702, 97703, 97704, 97705]
N = 1008
route = "repaired_permutation"

print(f"Testing {route} at N={N} with 5 seeds...")
print("Running actual smoke runner for ground truth\n")

results = []

for seed in seeds:
    print(f"Seed {seed}:")

    # Run dense
    out_dense = Path(f"/tmp/dense_{seed}")
    out_dense.mkdir(exist_ok=True)
    cmd_dense = [
        "conda", "run", "-n", "tftwogpu", "python",
        "docs/benchmarks/run_sqmc_rerun_corrected_filter_20260906.py",
        "smoke", str(out_dense),
        "--particle-counts", str(N),
        "--routes", route,
        "--seeds", str(seed),
        "--transport-plan", "dense"
    ]
    subprocess.run(cmd_dense, check=True, capture_output=True)
    dense_result = json.load(open(out_dense / "result.json"))
    dense_value = dense_result['rows'][0]['value']

    # Run streaming
    out_streaming = Path(f"/tmp/streaming_{seed}")
    out_streaming.mkdir(exist_ok=True)
    cmd_streaming = [
        "conda", "run", "-n", "tftwogpu", "python",
        "docs/benchmarks/run_sqmc_rerun_corrected_filter_20260906.py",
        "smoke", str(out_streaming),
        "--particle-counts", str(N),
        "--routes", route,
        "--seeds", str(seed),
        "--transport-plan", "streaming"
    ]
    subprocess.run(cmd_streaming, check=True, capture_output=True)
    streaming_result = json.load(open(out_streaming / "result.json"))
    streaming_value = streaming_result['rows'][0]['value']

    diff = streaming_value - dense_value
    results.append({'seed': seed, 'dense': dense_value,
                   'streaming': streaming_value, 'diff': diff})
    print(f"  dense={dense_value:.6f}, streaming={streaming_value:.6f}, diff={diff:+.6f}\n")

# Statistics
import numpy as np
diffs = [r['diff'] for r in results]
print("="*70)
print(f"Mean difference: {np.mean(diffs):+.6f}")
print(f"Std dev: {np.std(diffs, ddof=1):.6f}")
print(f"Min: {min(diffs):+.6f}, Max: {max(diffs):+.6f}")
print(f"Positive: {sum(1 for d in diffs if d>0)}, Negative: {sum(1 for d in diffs if d<0)}")
print("="*70)
