"""Offline M10 integrity, attempt, test, and worker-wall reconciliation."""
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[4]
PLAN = "docs/plans/bayesfilter-hmc-repair-master-program-2026-09-16.md"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
    started = time.monotonic()
    sys.path.insert(0, str(ROOT / "source-gpu-r3"))
    from bayesfilter.inference.hmc_candidate_set_artifacts import load_candidate_set_result_payload
    from bayesfilter.inference.hmc_candidate_set_tuning import _sha256
    from bayesfilter.testing.inference_validation.engines.pipeline import check_inventory

    invalid, outstanding, attempts, inventories, snapshots = [], [], [], [], []
    costs = {"cpu_reference": 0., "gpu": 0.}
    opening_path = ROOT.parent / "m9-r1/reconciliation-terminal.json"
    opening = read(opening_path)["budget_seconds"]
    expected = ["no-hint-baseline-gpu-r1", "hinted-baseline-gpu-r1",
                "no-hint-serious-gpu-r1", "hinted-serious-gpu-r1",
                "no-hint-finite-window-gpu-r1", "no-hint-standard-replay-gpu-r1",
                *(f"regression-finite-window-fresh-{i}-gpu-r1" for i in range(3)),
                "fresh-0-boundary-replay-gpu-r1"]
    for name in expected:
        if not (ROOT / name / "gpu-diagnostic-run.json").exists():
            outstanding.append(name)
    source_paths = set(ROOT.glob("source*/source_snapshot.json"))
    harnesses = [ROOT / "run_regression.py", *ROOT.glob("diagnose-preparation-*.py.txt"),
                 *ROOT.glob("*/harness.py.txt")]
    harness_hashes = {sha(path): str(path.relative_to(REPO)) for path in harnesses}
    for path in sorted(ROOT.glob("*-run.json")) + sorted(ROOT.glob("*/gpu-diagnostic-run.json")):
        record = read(path)
        seconds = record["elapsed_seconds"]
        if not math.isfinite(seconds) or seconds < 0.:
            raise ValueError("invalid attempt cost")
        device = record["device"]
        costs[device] += seconds
        attempt = {"path": str(path.relative_to(REPO)), "sha256": sha(path),
                   "seconds": seconds, "device": device, "returncode": record["returncode"]}
        attempts.append(attempt)
        if device != "gpu":
            if record.get("gpu_intentionally_hidden") is not True:
                invalid.append(str(path) + ": CPU hiding provenance")
            continue
        runtime = record.get("runtime") or {}
        memory = runtime.get("memory_policy", {})
        if not (runtime.get("device_scope") == "gpu" and runtime.get("jit_compile") is True
                and memory.get("all_physical_devices_memory_growth") is True
                and memory.get("configured_before_logical_device_initialization") is True):
            invalid.append(str(path) + ": GPU launch provenance")
        source_arg = record["command"][record["command"].index("--source") + 1]
        source_path = (REPO / source_arg / "source_snapshot.json").resolve()
        source_paths.add(source_path)
        manifest = read(path.parent / "manifest.json")
        if manifest["source"]["identity"] != read(source_path)["source_identity"]:
            invalid.append(str(path) + ": runtime/source mismatch")
        if record["script_sha256"] not in harness_hashes:
            invalid.append(str(path) + ": executed harness copy unavailable")
        else:
            attempt["preserved_harness"] = harness_hashes[record["script_sha256"]]
        if record["returncode"]:
            failure_path = path.parent / "failure.json"
            progress_path = path.parent / "preparation_progress.json"
            if not progress_path.exists():
                progress_path = path.parent / "preparation/preparation_progress.json"
            if not failure_path.exists() or not progress_path.exists():
                invalid.append(str(path) + ": failure evidence unavailable")
                continue
            progress = read(progress_path)
            events = progress.get("events", [])
            numerical_failure = any(
                event.get("details", {}).get("diagnostics", {}).get("hmc_error_message")
                == "operational warmup produced a nonfinite trace" for event in events)
            if not numerical_failure:
                invalid.append(str(path) + ": unclassified execution failure")
            attempt["failure_classification"] = "preparation_numerical_veto" if numerical_failure else "unclassified"
            attempt["failure_record"] = str(failure_path.relative_to(REPO))
        elif not (path.parent / "assessment.json").exists():
            outstanding.append(path.parent.name + ": assessment")

    for path in sorted({p.resolve() for p in source_paths}):
        manifest = read(path)
        actual = {str(p.relative_to(path.parent)): sha(p)
                  for p in sorted((path.parent / "bayesfilter").rglob("*.py"))}
        if actual != manifest["source_files"] or _sha256(actual) != manifest["source_identity"]:
            invalid.append(str(path) + ": snapshot checksum mismatch")
        snapshots.append({"path": str(path.relative_to(REPO)), "identity": manifest["source_identity"],
            "files": len(actual), "different_from_current": [f for f, h in actual.items() if sha(REPO / f) != h]})

    receipt_count = 0
    for path in sorted(ROOT.glob("*/tuning/candidate_set_result.json")):
        payload = load_candidate_set_result_payload(path)
        inventory = check_inventory(payload)
        inventories.append({"path": str(path.relative_to(REPO)), "sha256": sha(path), **inventory})
        invalid.extend(inventory["failures"])
        if payload["resume_pending_work_item_ids"] or payload["reserved_budget_units"]:
            outstanding.append(str(path) + ": pending work/reservation")
        for receipt in payload["verification_receipts"]:
            digest = receipt["numerical_evidence_hash"]
            evidence_path = path.parent / "numerical_evidence" / (digest + ".json")
            evidence = read(evidence_path)
            if _sha256(evidence) != digest:
                invalid.append(str(evidence_path) + ": numerical evidence checksum")
            if evidence["candidate"]["candidate_record_hash"] != receipt["candidate_record_hash"]:
                invalid.append(str(evidence_path) + ": receipt identity mismatch")
            receipt_count += 1

    tensor_count = 0
    def check_tree(node, directory):
        nonlocal tensor_count
        if isinstance(node, dict):
            if "tensor" in node:
                tensor = node["tensor"]
                if sha(directory / tensor["file"]) != tensor["sha256"]:
                    invalid.append(str(directory / tensor["file"]) + ": tensor checksum")
                tensor_count += 1
            else:
                for value in node.values():
                    check_tree(value, directory)
        elif isinstance(node, list):
            for value in node:
                check_tree(value, directory)
    for path in ROOT.glob("*/posterior_chunks/committed/*/bundle.json"):
        check_tree(read(path)["tree"], path.parent)
    for path in ROOT.glob("*/*.tensor.json"):
        if sha(path.with_suffix("")) != read(path)["sha256"]:
            invalid.append(str(path) + ": serialized tensor checksum")
        tensor_count += 1

    latest_tests = {}
    for path in sorted(ROOT.glob("*.xml"), key=lambda p: p.stat().st_mtime):
        for case in ET.parse(path).getroot().iter("testcase"):
            key = case.attrib["classname"] + "::" + case.attrib["name"]
            status = "failed" if case.find("failure") is not None or case.find("error") is not None else (
                "skipped" if case.find("skipped") is not None else "passed")
            latest_tests[key] = {"status": status, "source": path.name}
    invalid.extend(key for key, value in latest_tests.items() if value["status"] == "failed")
    guide = read(ROOT / "guide-r1/build-manifest.json")
    if sha(REPO / "docs/main.pdf") != guide["pdf_sha256"]:
        invalid.append("installed book differs from inspected build")
    if any(sha(REPO / f) != h for f, h in guide["source_sha256"].items()):
        invalid.append("book source changed after build")
    # All builds are timed by run_cpu.py. Snapshot creation and this audit are
    # covered by the predeclared conservative inspection/bookkeeping allowance.
    overhead = {"cpu_reference": 600., "gpu": 0.}
    tranche = {device: costs[device] + overhead[device] for device in costs}
    charged = {device: opening["cumulative_charged"][device] + tranche[device] for device in costs}
    remaining = {device: opening["authorized"][device] - charged[device] for device in costs}
    gpu_attempts = sum(row["device"] == "gpu" for row in attempts)
    exceeded = (tranche["cpu_reference"] > 6000. or tranche["gpu"] > 9000.
                or gpu_attempts > 10 or any(v < 0 for v in remaining.values()))
    summary = {"schema": "bayesfilter.hmc_m10_reconciliation.v1", "terminal": True,
        "created_utc": datetime.now(timezone.utc).isoformat(), "plan_file": PLAN,
        "opening_ledger": str(opening_path.relative_to(REPO)), "opening_sha256": sha(opening_path),
        "attempts": attempts, "source_snapshots": snapshots, "candidate_inventories": inventories,
        "numerical_receipts_checked": receipt_count, "tensor_checksums_checked": tensor_count,
        "tests": {s: sum(row["status"] == s for row in latest_tests.values()) for s in ("passed", "failed", "skipped")},
        "test_details": latest_tests, "invalid_artifacts": invalid, "outstanding": outstanding,
        "budget_seconds": {"authorized": opening["authorized"], "opening_charged": opening["cumulative_charged"],
            "m10_measured": costs, "overhead": overhead, "m10_total": tranche,
            "cumulative_charged": charged, "remaining": remaining, "exceeded": exceeded},
        "manifest": {"command": sys.argv, "environment": sys.executable, "script_sha256": sha(Path(__file__)),
            "elapsed_seconds": time.monotonic() - started, "gpu_intentionally_hidden": True,
            "numerical_workers_launched": 0, "seeds": "N/A: offline deterministic audit"},
        "interpretation": "Integrity and accounting only. Recorded numerical and posterior failures remain failures."}
    with (ROOT / "reconciliation-terminal.json").open("x") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({k: summary[k] for k in ("tests", "budget_seconds", "invalid_artifacts", "outstanding")}, indent=2))
    if invalid or outstanding or exceeded:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
