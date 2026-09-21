"""Offline M8 accounting and integrity audit; launches no numerical work."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[4]
sys.path.insert(0, str(REPO / "scripts"))
from audit_inference_validation_campaign import campaign_status, checked_seconds, file_hash

CAP = {"cpu_reference": 259200., "gpu": 172800.}
OPENING = {"cpu_reference": 85456.14800937325, "gpu": 50523.670210322656}
PLAN = "docs/plans/bayesfilter-hmc-repair-master-program-2026-09-16.md"


def read(path):
    return json.loads(Path(path).read_text())


def check_gpu(runtime):
    return (runtime.get("device_scope") == "gpu" and runtime.get("jit_compile") is True
            and runtime.get("memory_policy", {}).get("all_physical_devices_memory_growth") is True
            and runtime.get("memory_policy", {}).get("configured_before_logical_device_initialization") is True)


def reconcile(terminal):
    started = time.monotonic()
    audit = campaign_status(ROOT, {d: CAP[d] - OPENING[d] for d in CAP}, verify_artifacts=True)
    invalid = list(audit["invalid_artifacts"])
    costs = dict(audit["indexed_worker_seconds"])
    records, wrapped_xml, source_rows, runtime_rows = [], set(), [], []
    for path in sorted(ROOT.glob("*-run.json")) + sorted(ROOT.glob("*/gpu-diagnostic-run.json")):
        record = read(path)
        device = record["device"]
        costs[device] += checked_seconds(record["elapsed_seconds"])
        records.append({"path": str(path.relative_to(REPO)), "sha256": file_hash(path),
                        "seconds": record["elapsed_seconds"], "device": device,
                        "returncode": record["returncode"], "command": record["command"]})
        for arg in record["command"]:
            if "--junitxml=" in arg:
                wrapped_xml.add(Path(arg.split("=", 1)[1]).name)
        if device == "gpu":
            valid = check_gpu(record.get("runtime") or {})
            runtime_rows.append({"path": str(path.relative_to(REPO)), "valid": bool(valid)})
            if not valid:
                invalid.append({"path": str(path), "reason": "invalid GPU runtime provenance"})
    # These two test batches were launched directly. Their JUnit duration is
    # measured test time; process startup and other unmetered work are separate.
    for path in sorted(ROOT.glob("*.xml")):
        if path.name in wrapped_xml:
            continue
        suites = ET.parse(path).getroot().findall("testsuite")
        seconds = sum(float(s.attrib["time"]) for s in suites)
        costs["cpu_reference"] += checked_seconds(seconds)
        records.append({"path": str(path.relative_to(REPO)), "sha256": file_hash(path),
                        "seconds": seconds, "device": "cpu_reference",
                        "basis": "JUnit test duration; startup charged in overhead",
                        "tests": sum(int(s.attrib["tests"]) for s in suites),
                        "failures": sum(int(s.attrib["failures"]) + int(s.attrib["errors"]) for s in suites),
                        "skipped": sum(int(s.attrib["skipped"]) for s in suites)})
    for path in sorted(ROOT.glob("source*/source_snapshot.json")):
        snapshot = read(path)
        actual = {str(p.relative_to(path.parent)): file_hash(p)
                  for p in sorted((path.parent / "bayesfilter").rglob("*.py"))}
        identity = hashlib.sha256(json.dumps(actual, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        valid = actual == snapshot["source_files"] and identity == snapshot["source_identity"]
        if not valid:
            invalid.append({"path": str(path), "reason": "snapshot source mismatch"})
        changed = [name for name, value in actual.items()
                   if not (REPO / name).exists() or file_hash(REPO / name) != value]
        added = [str(p.relative_to(REPO)) for p in (REPO / "bayesfilter").rglob("*.py")
                 if str(p.relative_to(REPO)) not in actual]
        source_rows.append({"path": str(path.relative_to(REPO)), "sha256": file_hash(path),
                            "source_identity": identity, "valid": valid,
                            "different_from_current": changed, "current_only_files": sorted(added)})
    for path in sorted(ROOT.glob("*/run_index.json")):
        index = read(path)
        for name, job in index["jobs"].items():
            if job["status"] != "complete":
                if job["status"] == "failed":
                    attempt = job["attempts"][-1]["attempt"]
                    failure_path = path.parent / name / f"attempt-{attempt:03d}-failure.json"
                    failure = read(failure_path) if failure_path.exists() else {}
                    if failure.get("exception") != "HMCPreparationFailure":
                        invalid.append({"path": str(path.parent / name),
                                        "reason": "unresolved non-preparation failure", "failure": failure})
                elif terminal:
                    invalid.append({"path": str(path.parent / name),
                                    "reason": "unfinished indexed job", "status": job["status"]})
                continue
            result = read(job["result"])
            runtime = result["runtime"]
            valid = check_gpu(runtime) if runtime["device_scope"] == "gpu" else runtime["gpu_intentionally_hidden"]
            runtime_rows.append({"suite": path.parent.name, "job": name, "valid": bool(valid)})
            if not valid:
                invalid.append({"path": str(path.parent / name), "reason": "invalid runtime provenance"})
    sys.path.insert(0, str(REPO))
    from bayesfilter.testing.inference_validation.engines.pipeline import check_inventory
    inventories = []
    for path in sorted(ROOT.rglob("candidate_set_result.json")):
        result = check_inventory(read(path))
        inventories.append({"path": str(path.relative_to(REPO)), "sha256": file_hash(path), **result})
        if result["failures"]:
            invalid.append({"path": str(path), "reason": "invalid candidate inventory", **result})
    # Do not count coordinator wall time again: it contains the indexed workers.
    # This allowance covers inspections, source freezing, direct-test startup,
    # short unwrapped commands and supervision. It is an engineering allowance,
    # not a measurement of CPU utilization or agent elapsed time.
    overhead = {"cpu_reference": 600., "gpu": 0.}
    charged = {d: OPENING[d] + costs[d] + overhead[d] for d in CAP}
    remaining = {d: CAP[d] - charged[d] for d in CAP}
    expected_suites = ("funnel-fresh-gpu-r1", "power-16k-confirmation-gpu-r1",
                       "screen-fresh-gpu-r1", "stopping-fresh-gpu-r1")
    outstanding = []
    for name in expected_suites:
        path = ROOT / name / "run_index.json"
        if not path.exists():
            outstanding.append(name)
            continue
        index = read(path)
        for planned in index["plan"]["jobs"]:
            key = planned["design"]["design_id"]
            if index["jobs"].get(key, {}).get("status") not in {"complete", "failed"}:
                outstanding.append(name + "/" + key)
    for target in ("schools", "regression", "data-start-regression", "geometry-hint-regression"):
        for phase, count in (("pilot", 1), ("fresh", 3)):
            for i in range(count):
                path = ROOT / f"posteriordb-{target}-{phase}-{i}-gpu-r1"
                record_path = path / "gpu-diagnostic-run.json"
                if not record_path.exists():
                    outstanding.append(path.name)
                    continue
                record = read(record_path)
                if record["returncode"] == 0:
                    if not (path / "assessment.json").exists():
                        outstanding.append(path.name + "/missing-assessment")
                else:
                    failure_path = path / "failure.json"
                    failure = read(failure_path) if failure_path.exists() else {}
                    if failure.get("exception") != "HMCPreparationFailure":
                        invalid.append({"path": str(path),
                                        "reason": "unresolved matched-reference failure",
                                        "returncode": record["returncode"], "failure": failure})
    for i in range(3):
        path = ROOT / f"regression-interior-{i}-gpu-r2"
        if not (path / "gpu-diagnostic-run.json").exists():
            outstanding.append(path.name)
        elif read(path / "gpu-diagnostic-run.json")["returncode"] != 0:
            invalid.append({"path": str(path), "reason": "unresolved interval diagnostic failure"})
        elif not (path / "assessment.json").exists():
            outstanding.append(path.name + "/missing-assessment")
        elif read(path / "assessment.json")["completion"] != "complete":
            outstanding.append(path.name + "/incomplete-search")
    if terminal and outstanding:
        invalid.append({"reason": "unfinished planned work", "paths": outstanding})
    return {"schema": "bayesfilter.hmc_m8_reconciliation.v1", "terminal": terminal,
            "created_utc": datetime.now(timezone.utc).isoformat(), "plan_file": PLAN,
            "question": "Are saved evidence, runtime provenance and cumulative attempt costs intact?",
            "indexed_audit": audit, "other_attempts": records, "source_snapshots": source_rows,
            "candidate_inventories": inventories,
            "runtime_checks": runtime_rows, "invalid_artifacts": invalid,
            "outstanding": outstanding, "budget_seconds": {
                "authorized": CAP, "opening_charged": OPENING,
                "m8_measured_or_junit_seconds": costs, "m8_overhead_allowance": overhead,
                "cumulative_charged": charged, "remaining": remaining,
                "exceeded": any(value < 0 for value in remaining.values())},
            "manifest": {"command": sys.argv, "environment": sys.executable,
                "script_sha256": file_hash(__file__), "elapsed_seconds": time.monotonic() - started,
                "random_seeds": "N/A: deterministic saved-artifact audit",
                "gpu_used": False, "numerical_workers_launched": 0},
            "interpretation": "Integrity/accounting only; no scientific promotion or ranking."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--terminal", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = reconcile(args.terminal)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: result[key] for key in ("terminal", "budget_seconds", "invalid_artifacts", "outstanding")}, indent=2))
    raise SystemExit(bool(result["invalid_artifacts"]) or result["budget_seconds"]["exceeded"])


if __name__ == "__main__":
    main()
