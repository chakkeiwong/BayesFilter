"""Preserved fixed-chain bandwidth diagnosis under the M21 follow-up design."""
from pathlib import Path
import hashlib
import importlib.util
import json
import math
import os
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[4]
assert os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
assert os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH") == "true"
started = time.monotonic()
out = ROOT / "estimator-diagnosis-r1"
out.mkdir(exist_ok=False)
sys.path.insert(0, str(ROOT / "source-r1"))
reference_path = REPO / "bayesfilter/testing/inference_validation/engines/ar1_batch_reference.py"
spec = importlib.util.spec_from_file_location("ar1_batch_reference", reference_path)
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)
from bayesfilter.inference.hmc_precision import mean_precision
from bayesfilter.testing.inference_validation.storage import read_tensor
from bayesfilter.testing.inference_validation.engines.statistics import binomial_interval

results = {}
input_hashes = {}
for case in ("slow_stationary", "dispersed"):
    path = ROOT / "confirmation-cpu-r1" / case / "summary.json"
    rows = json.loads(path.read_text())["rows"]
    assert len(rows) == 400 and all(r["status"] == "complete" for r in rows)
    arms = {str(b): [] for b in (100, 250, 500)}
    arms["autocorrelation"] = []
    for row in rows:
        assert time.monotonic() - started < 850, "stop at declared diagnosis ceiling"
        tensor_path = path.parent / f"rep-{row['rep']:04d}" / "fixed-retained.tensor"
        input_hashes[str(tensor_path)] = hashlib.sha256(tensor_path.read_bytes()).hexdigest()
        values = read_tensor(tensor_path)
        assert tuple(values.shape) == (10000, 4, 1)
        for arm, observations in arms.items():
            report = mean_precision(values, method="autocorrelation" if arm == "autocorrelation" else "lugsail",
                batch_size=None if arm == "autocorrelation" else int(arm), jit_compile=False)
            valid = bool(report["valid"][0])
            se = float(report["mcse"][0]) if valid else None
            estimate = float(report["estimate"][0])
            observations.append({"rep": row["rep"], "available": valid,
                "covered": valid and abs(estimate) <= 1.959963984540054 * se,
                "mcse_over_exact": se / row["fixed_exact_law"]["mcse"] if valid else None})
    summary = {}
    for arm, observations in arms.items():
        covered = sum(r["covered"] for r in observations)
        ratios = [r["mcse_over_exact"] for r in observations if r["available"]]
        expected = None if arm == "autocorrelation" else reference.expected_lugsail_mean_variance(
            .98 if case == "slow_stationary" else .995, (0.,)*4 if case == "slow_stationary" else (-8.,-4.,4.,8.),
            warmup=10000, draws=10000, batch=int(arm), stationary_start=case == "slow_stationary")
        summary[arm] = {"covered": covered, "planned": 400, "available": len(ratios),
            "coverage_interval": binomial_interval(covered, 400),
            "median_mcse_over_exact": statistics.median(ratios) if ratios else None,
            "expected_untruncated_variance_over_exact": expected / rows[0]["fixed_exact_law"]["variance"] if expected is not None else None,
            "rows": observations}
    results[case] = summary
record = {"cases": results, "input_hashes": input_hashes,
    "command": sys.argv, "python": sys.executable, "elapsed_seconds": time.monotonic() - started,
    "source_manifest": str(ROOT / "source-manifest-r1.json"),
    "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "reference_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
    "plan_file": "docs/plans/bayesfilter-hmc-repair-m21-estimator-diagnosis-2026-09-22.md",
    "device": "cpu_reference", "gpu_intentionally_hidden": True, "jit_compile": False,
    "environment": {k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "TF_FORCE_GPU_ALLOW_GROWTH", "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS")},
    "interpretation": "development diagnosis on previous confirmation draws; analytical expectation precedes per-chain negative-estimate rejection",
    "default_promoted": False, "ranking_supported": False}
(out / "result.json").write_text(json.dumps(record, indent=2) + "\n")
for case, arms in results.items():
    for arm, result in arms.items():
        print(case, arm, {k: v for k, v in result.items() if k != "rows"}, flush=True)
