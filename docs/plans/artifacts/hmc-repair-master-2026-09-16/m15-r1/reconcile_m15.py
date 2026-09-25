"""Terminal M15 provenance, integrity and worker-budget audit, without sampling."""
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[4]
SOURCE = ROOT.parent / "m14-r1/source-schedule-r1"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
    started = time.monotonic()
    sys.path.insert(0, str(SOURCE))
    from bayesfilter.inference.hmc_candidate_set_tuning import _sha256
    from bayesfilter.inference.hmc_candidate_set_artifacts import load_candidate_set_result_payload
    from bayesfilter.testing.inference_validation.engines.pipeline import check_inventory

    invalid, outstanding, attempts, inventories = [], [], [], []
    costs = {"cpu_reference": 0., "gpu": 0.}
    source = read(SOURCE / "source_snapshot.json")
    actual = {str(p.relative_to(SOURCE)): sha(p) for p in sorted((SOURCE / "bayesfilter").rglob("*.py"))}
    if actual != source["source_files"] or _sha256(actual) != source["source_identity"]:
        invalid.append("frozen numerical source changed")
    statuses, receipts, tensors = {}, 0, 0
    suite_roots = [ROOT / name for name in ("sbc-pilot-cpu-r1", "fresh-cpu-continuation-r3",
                                          "stopping-pilot-gpu-r1", "sbc-fresh-gpu-r2")]
    for directory in suite_roots:
        index = read(directory / "run_index.json")
        if index["source"]["identity"] != source["source_identity"]:
            invalid.append(str(directory) + ": source mismatch")
        planned = {job["design"]["design_id"]: job for job in index["plan"]["jobs"]}
        statuses[directory.name] = dict(Counter(job["status"] for job in index["jobs"].values()))
        job_roots = set()
        for name, job in index["jobs"].items():
            design = planned[name]["design"]
            device = design["device"]
            job_roots.add(Path(job["result"]).parent if job.get("result") else directory / name)
            if job["status"] not in {"complete", "failed", "timed_out", "unfunded"}:
                outstanding.append(name)
            for attempt in job.get("attempts", []):
                costs[device] += attempt["elapsed_seconds"]
                attempts.append({"design_id": name, "device": device, **attempt})
            if job.get("result"):
                result_path = Path(job["result"])
                if sha(result_path) != job["result_sha256"]:
                    invalid.append(name + ": result checksum")
                manifest_path = result_path.with_name(result_path.name.replace("-result.json", "-manifest.json"))
                manifest = read(manifest_path)
                if manifest["source"]["identity"] != source["source_identity"]:
                    invalid.append(name + ": manifest source")
                if _sha256(manifest["design"]) != planned[name]["identity"]:
                    invalid.append(name + ": manifest design")
                runtime = manifest["runtime"]
                if device == "gpu":
                    memory = runtime["memory_policy"]
                    if not (runtime["jit_compile"] and memory["all_physical_devices_memory_growth"]
                            and memory["configured_before_logical_device_initialization"]):
                        invalid.append(name + ": launch provenance")
                elif runtime.get("gpu_intentionally_hidden") is not True:
                    invalid.append(name + ": CPU provenance")
        for path in sorted(p for base in job_roots for p in base.rglob("candidate_set_result.json")):
            payload = load_candidate_set_result_payload(path)
            inventory = check_inventory(payload)
            invalid.extend(str(path) + ": " + item for item in inventory["failures"])
            inventories.append({"path": str(path.relative_to(REPO)), **inventory})
            for receipt in payload["verification_receipts"]:
                evidence = read(path.parent / "numerical_evidence" / (receipt["numerical_evidence_hash"] + ".json"))
                if (_sha256(evidence) != receipt["numerical_evidence_hash"] or
                        evidence["candidate"]["candidate_record_hash"] != receipt["candidate_record_hash"]):
                    invalid.append(str(path) + ": verification receipt")
                receipts += 1
        for path in sorted(p for base in job_roots for p in base.rglob("*.tensor.json")):
            if sha(path.with_suffix("")) != read(path)["sha256"]:
                invalid.append(str(path) + ": tensor checksum")
            tensors += 1
        def check_tree(node, base):
            nonlocal tensors
            if isinstance(node, dict):
                if "tensor" in node:
                    record = node["tensor"]
                    tensors += 1
                    if sha(base / record["file"]) != record["sha256"]:
                        invalid.append(str(base / record["file"]) + ": checkpoint tensor")
                else:
                    for value in node.values():
                        check_tree(value, base)
            elif isinstance(node, list):
                for value in node:
                    check_tree(value, base)
        for path in sorted(p for base in job_roots for p in base.rglob("bundle.json")):
            if "posterior_chunks" in path.parts:
                check_tree(read(path)["tree"], path.parent)
    repair_path = ROOT / "fresh-cpu-launch-repair.json"
    repair = read(repair_path)
    # The profile validation failed before any numerical worker was launched.
    costs["cpu_reference"] += repair["elapsed_seconds"]
    attempts.append({"path": str(repair_path), "device": "cpu_reference",
                     "status": "profile_validation_failure", "elapsed_seconds": repair["elapsed_seconds"]})
    opening = read(ROOT.parent / "m14-r1/reconciliation-terminal.json")["budget_seconds"]
    overhead = {"cpu_reference": 900., "gpu": 0.}
    total = {key: costs[key] + overhead[key] for key in costs}
    charged = {key: opening["cumulative_charged"][key] + total[key] for key in costs}
    remaining = {key: opening["authorized"][key] - charged[key] for key in costs}
    budget = {"authorized": opening["authorized"], "opening_charged": opening["cumulative_charged"],
        "measured": costs, "overhead": overhead, "phase_total": total,
        "cumulative_charged": charged, "remaining": remaining,
        "exceeded": total["cpu_reference"] > 84000 or total["gpu"] > 24000 or min(remaining.values()) < 0}
    result = {"schema": "bayesfilter.hmc_m15_reconciliation.v1", "terminal": not outstanding,
        "source_identity": source["source_identity"], "source_snapshot": str(SOURCE),
        "attempts": attempts, "job_statuses": statuses, "candidate_inventories": inventories,
        "numerical_receipts_checked": receipts, "tensor_checksums_checked": tensors,
        "invalid_artifacts": invalid, "outstanding_workers": outstanding,
        "budget_seconds": budget, "summary_sha256": sha(ROOT / "terminal-summary.json"),
        "manifest": {"command": sys.argv, "environment": sys.executable,
            "script_sha256": sha(Path(__file__)), "elapsed_seconds": time.monotonic() - started,
            "gpu_intentionally_hidden": True},
        "interpretation": "Integrity and accounting; posterior caps and missing ranks remain scientific outcomes."}
    if outstanding:
        raise ValueError("finish numerical workers before terminal reconciliation")
    with (ROOT / "reconciliation-terminal.json").open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({key: result[key] for key in ("job_statuses", "budget_seconds", "invalid_artifacts")}, indent=2))
    return 1 if invalid or budget["exceeded"] else 0


if __name__ == "__main__":
    sys.exit(main())
