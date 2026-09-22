"""Reconcile M7 tests, frozen designs, geometry parity and the existing budget.

Read-only inspection apart from the requested new report. Imports no numerical
framework and launches no sampler. Unmetered inspection is in the overhead.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[4]
sys.path.insert(0, str(REPO))
from bayesfilter.testing.inference_validation.execution import plan_suite


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def definitions(path):
    result = {}
    for node in ast.parse(path.read_text()).body:
        name = getattr(node, "name", None)
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            name = getattr(node.targets[0], "id", None)
        if name:
            result[name] = ast.dump(node, include_attributes=False)
    return result


def gpu_runs(snapshot_identity):
    """Check every planned job and attempt, including failures and missing work."""
    suites, charged, missing_reservations = [], 0., 0.
    for path in sorted(ROOT.glob("*-gpu-r*/run_index.json")):
        index = read(path)
        assert index["source"]["identity"] == snapshot_identity, path
        assert index["max_workers"] == 1, path
        jobs = []
        for planned in index["plan"]["jobs"]:
            name = planned["design"]["design_id"]
            job = index["jobs"].get(name, {"status": "not_run", "attempts": []})
            assert job["status"] != "running", (path, name)
            attempts = []
            for attempt in job["attempts"]:
                charged += attempt["elapsed_seconds"]
                row = dict(attempt)
                if "log" in row:
                    row["log_sha256"] = sha(Path(row["log"]))
                attempts.append(row)
            row = {"design_id": name, "design_identity": planned["identity"],
                   "design": planned["design"], "status": job["status"], "attempts": attempts}
            if job["status"] == "complete":
                result_path = Path(job["result"])
                assert sha(result_path) == job["result_sha256"], result_path
                result = read(result_path)
                assert result["execution_status"] == "complete", result_path
                assert result["design_identity"] == planned["identity"], result_path
                runtime = result["runtime"]
                assert runtime["device_scope"] == "gpu" and runtime["jit_compile"]
                assert not runtime["gpu_intentionally_hidden"]
                memory = runtime["memory_policy"]
                assert memory["all_physical_devices_memory_growth"]
                assert memory["configured_before_logical_device_initialization"]
                assert memory["tf_force_gpu_allow_growth"] == "true"
                row.update(result=str(result_path.relative_to(REPO)),
                           result_sha256=job["result_sha256"], runtime=runtime,
                           finding=result["assessment"].get("finding"))
                assessment = result["assessment"]
                if "rates" in assessment:
                    row["rates"] = assessment["rates"]
                if "metropolis_log_ratio_passed" in assessment:
                    row["energy_check"] = {key: assessment[key] for key in (
                        "metropolis_log_ratio_passed", "metropolis_log_ratio_max_error",
                        "metropolis_state_selection_passed")}
            else:
                row["reason"] = job.get("reason")
                missing_reservations += max(0., planned["design"]["budget_seconds"] - sum(
                    attempt["elapsed_seconds"] for attempt in attempts))
            jobs.append(row)
        artifacts = {str(item.relative_to(REPO)): sha(item)
                     for item in sorted(path.parent.rglob("*"))
                     if item.is_file() and item.suffix in {".json", ".tensor", ".prof"}}
        suites.append({"run_index": str(path.relative_to(REPO)), "jobs": jobs,
                       "artifact_sha256": artifacts})
    for directory in ROOT.glob("funnel-grid-r*"):
        assert (directory/"gpu-diagnostic-run.json").is_file(), "unfinished diagnostic: " + str(directory)
    for path in sorted(ROOT.glob("funnel-grid-r*/gpu-diagnostic-run.json")):
        record = read(path)
        charged += record["elapsed_seconds"]
        row = {"manifest": str(path.relative_to(REPO)), "manifest_sha256": sha(path),
               "attempt": record, "log_sha256": sha(Path(record["log"]))}
        if record["returncode"] == 0:
            manifest = read(path.parent/"manifest.json")
            assert manifest["source"]["identity"] == snapshot_identity
            assert manifest["runtime"]["jit_compile"]
            assert manifest["runtime"]["memory_policy"]["all_physical_devices_memory_growth"]
            assert manifest["runtime"]["memory_policy"]["configured_before_logical_device_initialization"]
            assert not manifest["historical_qualification_reused"]
            row["diagnostic"] = read(path.parent/"diagnostic.json")
            row["artifact_sha256"] = {str(item.relative_to(REPO)): sha(item)
                for item in sorted(path.parent.rglob("*.json"))}
        suites.append({"standalone_saved_geometry_diagnostic": row})
    return suites, charged, missing_reservations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    records = []
    for path in sorted(ROOT.glob("*-run.json")):
        record = read(path)
        assert record["gpu_intentionally_hidden"]
        records.append({"manifest": str(path.relative_to(REPO)),
            "manifest_sha256": sha(path), "log_sha256": sha(Path(record["log"])),
            "elapsed_seconds": record["elapsed_seconds"], "returncode": record["returncode"]})
    cases = {}
    final_xml = ("preparation-verified.xml", "validation-verified.xml", "guide-interface-tests.xml",
                 "coverage-verified.xml", "guide-interface-r2.xml", "checksum-verified.xml")
    for name in final_xml:
        for case in ET.parse(ROOT/name).iter("testcase"):
            assert case.find("failure") is None and case.find("error") is None
            cases[case.attrib["classname"], case.attrib["name"]] = (
                "passed" if case.find("skipped") is None else "skipped")
    before, after = read(ROOT/"geometry-before.json"), read(ROOT/"geometry-after.json")
    assert before == after
    checksum_parity = read(ROOT/"checksum-parity.json")
    assert checksum_parity["all_payload_hashes_identical"]
    assert checksum_parity["payload_count"] == len(checksum_parity["expected_hashes"]) == 208
    extraction = read(ROOT/"geometry-extraction.json")
    baseline = definitions(ROOT/"baseline-hmc_kernel_tuning.py")
    current = definitions(REPO/"bayesfilter/inference/hmc_geometry.py")
    for name in extraction["moved_symbols"]:
        assert baseline[name] == current[name], name
    snapshot = ROOT/"source-r1"
    frozen = read(snapshot/"source_snapshot.json")
    for name, value in frozen["source_files"].items():
        assert sha(snapshot/name) == value, name
    changed = [name for name, value in frozen["source_files"].items() if sha(REPO/name) != value]
    changed_definitions = {}
    for name in changed:
        old, new = definitions(snapshot/name), definitions(REPO/name)
        changed_definitions[name] = sorted(key for key in old.keys() | new.keys()
                                           if old.get(key) != new.get(key))
    suites = []
    for item in frozen["plans"]:
        path = Path(item["suite"])
        assert sha(path) == item["sha256"]
        plan = plan_suite(read(path))
        assert plan["budget_by_device"] == item["budget_by_device"]
        assert all(row["availability"] == "ready" for row in plan["jobs"])
        suites.append({"path": str(path.relative_to(REPO)), "designs": len(plan["jobs"]),
                       "budget_by_device": plan["budget_by_device"]})
    assert sum(row["budget_by_device"]["gpu"] for row in suites) == 9600.
    launched, measured_gpu, missing_reservations = gpu_runs(frozen["source_identity"])
    installed = read(ROOT/"official-guide-install-r3.json")
    build = read(ROOT/"guide-r3/build-manifest.json")
    assert sha(REPO/installed["official_pdf"]) == installed["pdf_sha256"] == build["pdf_sha256"]
    for name, value in build["source_sha256"].items():
        assert sha(REPO/name) == value, name
    measured_cpu = sum(row["elapsed_seconds"] for row in records)
    overhead_cpu = 300.
    cpu_charge = measured_cpu + overhead_cpu
    assert cpu_charge <= 2400.
    prior_cpu, prior_gpu = 83425.01779734538, 47585.583791223704
    result = {"created_utc": datetime.now(timezone.utc).isoformat(),
        "question": "M7 engineering consistency, GPU development execution and remaining budget; no calibration claim",
        "plan_file": "docs/plans/bayesfilter-hmc-repair-master-program-2026-09-16.md",
        "result_file": "docs/plans/bayesfilter-hmc-repair-m7-result-2026-09-17.md",
        "command": sys.argv, "numerical_framework_imported": False,
        "cpu_attempts": records,
        "final_distinct_tests": {outcome: sum(value == outcome for value in cases.values())
                                 for outcome in ("passed", "skipped")},
        "geometry_cases_identical": len(before), "moved_symbol_asts_identical": len(extraction["moved_symbols"]),
        "checksum_parity": {key: checksum_parity[key] for key in (
            "all_payload_hashes_identical", "payload_count", "median_seconds", "timing_role")},
        "snapshot_identity": frozen["source_identity"], "verified_snapshot_files": len(frozen["source_files"]),
        "live_source_changes_since_snapshot": changed, "suites": suites,
        "changed_definitions_since_snapshot": changed_definitions,
        "gpu_runs": launched,
        "official_guide_verified": installed,
        "budget_seconds": {"original_allowance_each_device": 86400.,
            "prior_cpu_charged": prior_cpu, "prior_gpu_charged": prior_gpu,
            "m7_cpu_measured": measured_cpu, "m7_cpu_overhead": overhead_cpu,
            "m7_cpu_charged": cpu_charge, "m7_gpu_charged": measured_gpu,
            "total_cpu_charged": prior_cpu + cpu_charge, "total_gpu_charged": prior_gpu + measured_gpu,
            "remaining_cpu": 86400. - prior_cpu - cpu_charge,
            "remaining_gpu": 86400. - prior_gpu - measured_gpu,
            "remaining_m7_cpu_allocation": 2400. - cpu_charge,
            "remaining_m7_gpu_allocation": 9600. - measured_gpu,
            "unfinished_launched_suite_reservations": missing_reservations},
        "scientific_interpretation": "engineering and GPU development evidence only; failures retain denominators; no candidate ranking or default promotion"}
    assert measured_gpu <= 9600., "M7 allocation exhausted"
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({key: result[key] for key in ("final_distinct_tests", "geometry_cases_identical",
        "moved_symbol_asts_identical", "verified_snapshot_files", "live_source_changes_since_snapshot",
        "budget_seconds")}, indent=2))


if __name__ == "__main__":
    main()
