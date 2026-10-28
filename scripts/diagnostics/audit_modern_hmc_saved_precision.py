"""Diagnostic-only scalar replay of saved count-planning precision outcomes."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from functools import lru_cache
import gzip
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def require(value, message):
    if not value:
        raise ValueError(message)

def same(actual, expected, message, *, mean=False):
    require(math.isfinite(actual) and math.isfinite(expected) and
            math.isclose(actual, expected, rel_tol=2e-10,
                         abs_tol=1e-12 if mean else 1e-20), message)

def replication_summary(chain_means):
    m = len(chain_means)
    require(m >= 2 and all(math.isfinite(x) for x in chain_means),
            "finite means from at least two chains required")
    center = math.fsum(chain_means) / m
    terms = [(x - center) ** 2 for x in chain_means]
    total = math.fsum(terms)
    return dict(estimate=center, variance=total / (m * (m - 1)),
                mcse=math.sqrt(total / (m * (m - 1))),
                largest_chain_variance_share=max(terms) / total if total else None)

@lru_cache(maxsize=64)
def stationary_variance(n, m, rho, power):
    """Finite stationary Gaussian reference scale, never selected-law variance."""
    require(n > 0 and m >= 2 and -1 < rho < 1 and power in (1, 2),
            "invalid reference scope")
    factor = 1.0 if power == 1 else 2.0
    r = rho ** power
    return factor * math.fsum([1.0] + [
        2 * (1 - lag / n) * r ** lag for lag in range(1, n)]) / (m * n)

def check_endpoint(endpoint, m):
    means = endpoint["chain_means"]
    quantities = endpoint["quantities"]
    targets = endpoint["report"]["targets"]
    require(len(means) == m and len(quantities) == len(targets) == 2 and
            all(len(x) == 2 for x in means), "endpoint dimension mismatch")
    rows = []
    for index, (quantity, target) in enumerate(zip(quantities, targets)):
        require(quantity["name"] == target["name"], "quantity ordering mismatch")
        observed = replication_summary([x[index] for x in means])
        for record in (quantity, target):
            same(record["estimate"], observed["estimate"], "stored mean mismatch", mean=True)
            same(record["mcse"], observed["mcse"], "stored MCSE mismatch")
        tolerance = quantity["absolute_tolerance"]
        same(target["mcse_absolute_max"], tolerance, "changed threshold")
        passed = observed["mcse"] <= tolerance
        require(passed == quantity["precision_passed"] == target["passed"],
                "stored precision decision mismatch")
        covered = abs(observed["estimate"] - quantity["truth"]) <= quantity["multiplier"] * observed["mcse"]
        require(covered == quantity["covered"], "stored interval decision mismatch")
        rows.append(observed | dict(name=quantity["name"], tolerance=tolerance,
                                   precision_passed=passed, covered=covered))
    return rows

def insufficient_quantities(assessment):
    if "precision" not in assessment:
        require(assessment.get("failed_checks") == ["warmup"] and
                assessment.get("passed") is False, "unexplained missing precision")
        return []
    return [q["name"] for q in assessment["precision"]["targets"] if not q["passed"]]

def audit(args):
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "-1", "CPU-only diagnostic required")
    require(os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH") == "true", "growth configuration required")
    inventory = json.loads((args.evidence / "inventory.json").read_text())
    require(all(sha(args.evidence / p) == h for p, h in inventory.items()),
            "compact evidence hash mismatch")
    manifest = json.loads((args.evidence / "manifest.json").read_text())
    with gzip.open(args.evidence / "records.jsonl.gz", "rt") as handle:
        records = [json.loads(line) for line in handle]
    design = manifest["design"]
    identity = lambda r: (r["stage"], r["case"], r["chains"], r["replication"])
    expected = {(stage, p["case"], p["chains"], rep)
                for p in design["profiles"]
                for stage, count in (("calibration", design["calibration_replicates"]),
                                     ("validation", design["validation_replicates"]))
                for rep in range(count)}
    require(len(records) == len(expected) and {identity(r) for r in records} == expected,
            "missing or duplicated planned records")
    source = Path(manifest["source"])
    relative = "bayesfilter/inference/hmc_precision_replication.py"
    numerical_source = source / relative
    require(sha(numerical_source) == manifest["source_sha256"][relative],
            "frozen variance source mismatch")
    import tensorflow as tf
    require(not tf.config.list_physical_devices("GPU"), "GPU was not hidden")
    spec = importlib.util.spec_from_file_location("frozen_precision_reference", numerical_source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    groups = defaultdict(lambda: dict(calibration=0, validation=0,
        outcomes=Counter(), calibration_by_count={}, endpoints=[], refused=0))
    replayed = 0
    for record in records:
        key = f"{record['case']}:m{record['chains']}"
        group = groups[key]
        stage = record["stage"]
        group[stage] += 1
        if stage == "calibration":
            for assessment in record.get("assessments") or []:
                count = str(assessment["count"])
                counts = group["calibration_by_count"].setdefault(count, dict(
                    assessed=0, failed_checks=Counter(), insufficient_quantities=Counter()))
                counts["assessed"] += 1
                counts["failed_checks"].update(assessment["failed_checks"])
                counts["insufficient_quantities"].update(insufficient_quantities(assessment))
            continue
        group["outcomes"][record["outcome"]] += 1
        endpoint = record.get("endpoint")
        if endpoint is None:
            group["refused"] += 1
            continue
        m = record["chains"]
        checked = check_endpoint(endpoint, m)
        tensor = tf.constant([endpoint["chain_means"]], tf.float64)
        evaluated = module.replication_program((1, m, 2), True)(tensor)
        actual = [float(x) for x in tf.unstack(evaluated["variance"])]
        for i, item in enumerate(checked):
            same(actual[i], item["variance"], "frozen TensorFlow variance mismatch")
            reference = endpoint["quantities"][i]["stationary_reference_variance"]
            item["stationary_reference_variance"] = reference
            item["estimated_to_stationary_variance_ratio"] = item["variance"] / reference
            item["stationary_scale_within_tolerance"] = reference <= item["tolerance"] ** 2
            if record["case"] in ("iid", "strong_dependent", "strong_antithetic"):
                rho = {"iid": 0.0, "strong_dependent": .99, "strong_antithetic": -.99}[record["case"]]
                exact = stationary_variance(record["calibration"]["selected_count"], m, rho, i + 1)
                same(reference, exact, "stationary fixed-count reference mismatch")
            values = [row[i] for row in endpoint["chain_means"]]
            item["leave_one_chain_mcse"] = [replication_summary(values[:j]+values[j+1:])["mcse"]
                                             for j in range(m)] if m > 2 else []
        group["endpoints"].append(dict(replication=record["replication"],
            outcome=record["outcome"], selected_count=record["calibration"]["selected_count"],
            quantities=checked))
        replayed += 1
    for group in groups.values():
        group["descriptive_variance_ratios"] = {}
        for i, name in enumerate(("coordinate", "square")):
            values = [r["quantities"][i]["estimated_to_stationary_variance_ratio"]
                      for r in group["endpoints"]]
            group["descriptive_variance_ratios"][name] = dict(count=len(values),
                median=statistics.median(values) if values else None,
                minimum=min(values) if values else None, maximum=max(values) if values else None)
    return dict(status="PASS", records_checked=len(records), validation_endpoints_recomputed=replayed,
                source=str(numerical_source), source_sha256=sha(numerical_source),
                evidence_inventory_sha256=sha(args.evidence/"inventory.json"),
                tensorflow_version=tf.__version__, groups=dict(groups),
                raw_trajectories_replayed=False, new_sampling=False,
                interpretation="scalar reduction audit; stationary reference scale is not selected-law precision",
                ranking_supported=False, coverage_established=False, default_promoted=False)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    manifest = dict(command=[sys.executable,*sys.argv], started_unix=time.time(),
        git_commit=subprocess.check_output(["git","rev-parse","HEAD"], text=True).strip(),
        driver_sha256=sha(__file__), cpu_only=True,
        environment={k:os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES","TF_FORCE_GPU_ALLOW_GROWTH",
                     "TF_NUM_INTRAOP_THREADS","TF_NUM_INTEROP_THREADS")},
        plan="docs/plans/bayesfilter-modern-hmc-c4-saved-precision-audit-2026-10-28.md")
    (args.output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    try:
        result = audit(args)
    except Exception:
        result = dict(status="ERROR",exception=traceback.format_exc())
        traceback.print_exc()
    result["inclusive_elapsed_seconds"] = time.monotonic()-start
    (args.output/"result.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps({k:result.get(k) for k in ("status","records_checked",
        "validation_endpoints_recomputed","inclusive_elapsed_seconds")}))
    return 0 if result["status"]=="PASS" else 1

if __name__ == "__main__":
    raise SystemExit(main())
