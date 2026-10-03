"""Offline diagnostic audit of saved validation costs and incomplete SBC units.

This standard-library reader launches no numerical workers and never pools
retries into independent statistical evidence. Prior artifacts remain untouched.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time


REPO = Path(__file__).resolve().parents[1]
DEVICES = ("cpu_reference", "gpu")


def read_json(path):
    def reject(value):
        raise ValueError(f"nonfinite JSON value: {value}")
    return json.loads(Path(path).read_text(), parse_constant=reject)


def file_hash(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def canonical_hash(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def checked_seconds(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("worker seconds must be numeric")
    if not math.isfinite(value) or value < 0:
        raise ValueError("worker seconds must be finite and nonnegative")
    return value


def campaign_status(root, allowance, *, verify_artifacts=False, overhead_seconds=None):
    """Read ongoing or terminal suites; charge cancellation and queued work."""
    root=Path(root).resolve()
    if set(allowance)!=set(DEVICES):
        raise ValueError("explicit CPU and GPU campaign allowances required")
    for value in allowance.values(): checked_seconds(value)
    overhead = dict.fromkeys(DEVICES, 0.) if overhead_seconds is None else dict(overhead_seconds)
    if set(overhead) != set(DEVICES):
        raise ValueError("explicit CPU and GPU overhead required")
    for value in overhead.values(): checked_seconds(value)
    consumed=dict.fromkeys(DEVICES,0.)
    reserved=dict.fromkeys(DEVICES,0.)
    profiles=[]
    invalid=[]
    checked=0
    for path in sorted(root.glob("*/run_index.json")):
        index=read_json(path)
        designs={j["design"]["design_id"]:j for j in index["plan"]["jobs"]}
        active=not index.get("cancellation") and any(j["status"]=="running" for j in index["jobs"].values())
        costs=dict.fromkeys(DEVICES,0.)
        holds=dict.fromkeys(DEVICES,0.)
        for name,planned in designs.items():
            design=planned["design"]
            device=design["device"]
            job=index["jobs"].get(name,{"status":"not_run","attempts":[]})
            costs[device]+=sum(checked_seconds(a["elapsed_seconds"]) for a in job.get("attempts",[]))
            if job["status"]=="running": holds[device]+=checked_seconds(job["reserved_seconds"])
            elif job["status"]=="not_run" and active: holds[device]+=checked_seconds(design["budget_seconds"])
            if verify_artifacts and job["status"]=="complete":
                checked+=1
                try:
                    result_path=Path(job["result"])
                    result=read_json(result_path)
                    if (file_hash(result_path)!=job["result_sha256"] or result["design_identity"]!=planned["identity"]
                            or result["execution_status"]!="complete"):
                        raise ValueError("result checksum or identity mismatch")
                    manifest=result_path.with_name(result_path.name.replace("-result.json","-manifest.json"))
                    if manifest.exists() and read_json(manifest)["source"]["identity"]!=index["source"]["identity"]:
                        raise ValueError("worker source differs from coordinator source")
                except (OSError,ValueError,KeyError) as exc:
                    invalid.append({"path":str(path.parent/name),"reason":str(exc)})
        for device in DEVICES:
            consumed[device]+=costs[device]
            reserved[device]+=holds[device]
        profiles.append({"profile":path.parent.name,"index_sha256":file_hash(path),
                         "source_identity":index["source"]["identity"],"worker_seconds":costs,
                         "reserved_seconds":holds,"statuses":dict(Counter(
                             index["jobs"].get(name, {"status": "not_run"})["status"] for name in designs))})
    tensors=sorted(root.rglob("*.tensor.json")) if verify_artifacts else []
    for path in tensors:
        try:
            if file_hash(path.with_suffix(""))!=read_json(path)["sha256"]:
                raise ValueError("tensor checksum mismatch")
        except (OSError,ValueError,KeyError) as exc:
            invalid.append({"path":str(path),"reason":str(exc)})
    charged={d:consumed[d]+overhead[d] for d in DEVICES}
    remaining={d:allowance[d]-charged[d]-reserved[d] for d in DEVICES}
    return {"schema":"bayesfilter.inference_validation_campaign_status.v1", "artifact_root":str(root),
            "campaign_allowance_seconds":allowance,"indexed_worker_seconds":consumed,
            "overhead_charged_seconds":overhead,"total_charged_seconds":charged,
            "reserved_seconds":reserved,"uncommitted_seconds":remaining,"profiles":profiles,
            "budget_exceeded":any(v<0 for v in remaining.values()),"invalid_artifacts":invalid,
            "indexed_results_checked":checked,"tensor_checksums_checked":len(tensors),
            "numerical_workers_launched":0,"statistical_conclusions":"not assessed by accounting"}


def sbc_inventory(root, design):
    """A partial native fit never substitutes for a completed dataset record."""
    counts = Counter(complete=0, missing_fit=0, started_without_dataset_record=0, unstarted=0)
    reasons = Counter()
    rows = []
    for dataset in range(design["replications"]):
        path = root / f"dataset-{dataset:04d}.json"
        fit_root = root / f"dataset-{dataset:04d}"
        native_fits = sum((fit_root / f"fit-{fit:04d}" / "pipeline.json").is_file()
                          for fit in range(design["rank_draws"]))
        if path.is_file():
            record = read_json(path)
            complete = (record["status"] == "complete" and bool(record["ranks"])
                        and len(record["fits"]) == design["rank_draws"]
                        and len(record["fit_outputs_active"]) == design["rank_draws"]
                        and not any(fit.get("reason") for fit in record["fits"]))
            if record["dataset_id"] != dataset or (record["status"] == "complete" and not complete):
                raise ValueError(f"inconsistent SBC dataset: {path}")
            status = "complete" if complete else "missing_fit"
            reasons.update(fit["reason"] for fit in record["fits"] if fit.get("reason"))
        else:
            status = "started_without_dataset_record" if fit_root.is_dir() else "unstarted"
        counts[status] += 1
        rows.append({"dataset_id": dataset, "status": status,
                     "native_pipeline_records": native_fits})
    return {"planned_datasets": design["replications"],
            "fits_per_dataset": design["rank_draws"], "datasets": rows,
            "dataset_counts": dict(counts), "recorded_missing_fit_reasons": dict(reasons),
            "calibration_complete": counts["complete"] == design["replications"],
            "pooled_with_other_attempts": False}


def audit_campaign(root, *, repo=REPO):
    root, repo = Path(root).resolve(), Path(repo).resolve()
    terminal = read_json(root / "terminal_audit.json")
    allowance = terminal["campaign_allowance_seconds"]
    if set(allowance) != set(DEVICES):
        raise ValueError("explicit CPU and GPU campaign allowances required")
    for value in allowance.values():
        checked_seconds(value)
    overhead = terminal.get("overhead_charged_seconds", dict.fromkeys(DEVICES, 0.))
    if set(overhead) != set(DEVICES):
        raise ValueError("explicit CPU and GPU overhead required")
    for value in overhead.values():
        checked_seconds(value)
    sources = {str(p.relative_to(repo)): file_hash(p)
               for p in sorted((repo / "bayesfilter").rglob("*.py"))}
    current_identity = canonical_hash(sources)
    consumed = dict.fromkeys(DEVICES, 0.)
    invalid, profiles, sbc = [], [], []
    results_checked = 0
    paths = sorted(root.glob("*/run_index.json"))
    if not paths:
        raise ValueError("no saved campaign indexes")
    expected = {p["profile"] for p in terminal["profiles"]}
    actual = {p.parent.name for p in paths}
    if expected != actual:
        invalid.append({"path": str(root), "reason": "profile inventory differs from terminal audit",
                        "missing": sorted(expected - actual), "additional": sorted(actual - expected)})
    for path in paths:
        index = read_json(path)
        seconds = dict.fromkeys(DEVICES, 0.)
        designs = {j["design"]["design_id"]: j for j in index["plan"]["jobs"]}
        unexpected = set(index["jobs"]) - set(designs)
        if unexpected:
            invalid.append({"path": str(path), "reason": "unplanned jobs in campaign index",
                            "additional": sorted(unexpected)})
        for name, planned in designs.items():
            job = index["jobs"].get(name, {"status": "not_run", "attempts": []})
            design = planned["design"]
            device = design["device"]
            if device not in DEVICES:
                raise ValueError(f"unknown device in {path}: {device}")
            seconds[device] += sum(checked_seconds(a["elapsed_seconds"]) for a in job.get("attempts", []))
            if job["status"] == "running":
                seconds[device] += checked_seconds(job["reserved_seconds"])
                invalid.append({"path": str(path), "reason": f"unresolved running job: {name}"})
            if job["status"] == "complete":
                results_checked += 1
                result_path = Path(job["result"])
                if not result_path.is_absolute():
                    result_path = repo / result_path
                try:
                    result = read_json(result_path)
                    if (file_hash(result_path) != job["result_sha256"]
                            or result["design_identity"] != planned["identity"]
                            or result["execution_status"] != "complete"):
                        raise ValueError("result checksum, identity or completion mismatch")
                except (OSError, ValueError, KeyError) as exc:
                    invalid.append({"path": str(result_path), "reason": str(exc)})
            if design["engine"] == "sbc" and design["scenario"]["route"] in {
                    "ordinary", "prepared", "fixed_transport"}:
                try:
                    sbc.append({"profile": path.parent.name, "design_id": name, "device": device,
                                "process_status": job["status"],
                                "source_status": "current" if index["source"]["identity"] == current_identity else "historical",
                                **sbc_inventory(path.parent / name, design)})
                except (OSError, ValueError, KeyError) as exc:
                    invalid.append({"path": str(path.parent / name), "reason": str(exc)})
        for device in DEVICES:
            consumed[device] += seconds[device]
        profiles.append({"profile": path.parent.name, "index_sha256": file_hash(path),
                         "worker_seconds": seconds,
                         "statuses": dict(Counter(index["jobs"].get(name, {"status": "not_run"})["status"]
                                                  for name in designs)),
                         "source_status": "current" if index["source"]["identity"] == current_identity else "historical"})
    tensors = sorted(root.rglob("*.tensor.json"))
    for path in tensors:
        try:
            if file_hash(path.with_suffix("")) != read_json(path)["sha256"]:
                raise ValueError("tensor checksum mismatch")
        except (OSError, ValueError, KeyError) as exc:
            invalid.append({"path": str(path), "reason": str(exc)})
    for device in DEVICES:
        if not math.isclose(consumed[device], terminal["indexed_worker_seconds"][device], rel_tol=0., abs_tol=1e-6):
            invalid.append({"path": str(root / "terminal_audit.json"),
                            "reason": f"{device} accounting differs from terminal snapshot"})
    charged = {d: consumed[d] + overhead[d] for d in DEVICES}
    remaining = {d: allowance[d] - charged[d] for d in DEVICES}
    return {"schema": "bayesfilter.inference_validation_continuation_preflight.v1",
            "artifact_root": str(root), "current_source_identity": current_identity,
            "terminal_audit_sha256": file_hash(root / "terminal_audit.json"),
            "campaign_allowance_seconds": allowance, "indexed_worker_seconds": consumed,
            "overhead_charged_seconds": overhead, "total_charged_seconds": charged,
            "remaining_seconds": remaining, "budget_exceeded": any(v < 0 for v in remaining.values()),
            "profiles": profiles, "indexed_results_checked": results_checked,
            "tensor_checksums_checked": len(tensors), "invalid_artifacts": invalid,
            "full_procedure_sbc": sbc, "numerical_workers_launched": 0,
            "rate_precision_illustrations": [
                {"worst_case_standard_error": se, "independent_bernoulli_units": math.ceil(1 / (4 * se**2))}
                for se in (.05, .025)],
            "precision_interpretation": "Var(p_hat) <= 1/(4*n); illustrations only, not confidence guarantees or SBC power",
            "statistical_conclusions": "not assessed by accounting",
            "limitations": ["Complete processes can still have missing posterior outputs; inspect the dataset inventory.",
                            "Calibration and power conclusions require the predeclared statistical assessment.",
                            "Repeated or historical profiles are not pooled as independent evidence.",
                            "Checksums verify stored bytes, not numerical or statistical correctness."]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--live",action="store_true")
    parser.add_argument("--cpu-seconds",type=float)
    parser.add_argument("--gpu-seconds",type=float)
    parser.add_argument("--verify-artifacts",action="store_true")
    parser.add_argument("--overhead-cpu-seconds", type=float, default=0.)
    parser.add_argument("--overhead-gpu-seconds", type=float, default=0.)
    parser.add_argument("--plan-file", default="docs/plans/bayesfilter-inference-validation-phase2-plan-2026-09-16.md")
    args = parser.parse_args(argv)
    started = time.monotonic()
    if args.live:
        if args.cpu_seconds is None or args.gpu_seconds is None:
            parser.error("live accounting requires explicit CPU and GPU allowances")
        result=campaign_status(args.root,{"cpu_reference":args.cpu_seconds,"gpu":args.gpu_seconds},
                               verify_artifacts=args.verify_artifacts,
                               overhead_seconds={"cpu_reference": args.overhead_cpu_seconds,
                                                 "gpu": args.overhead_gpu_seconds})
    else:
        if args.overhead_cpu_seconds or args.overhead_gpu_seconds:
            parser.error("terminal audit reads overhead from terminal_audit.json")
        result = audit_campaign(args.root)
    result["manifest"] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv if argv is None else [str(Path(__file__)), *map(str, argv)],
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                                     capture_output=True, text=True, check=True).stdout.strip(),
        "script_sha256": file_hash(__file__), "environment": sys.executable,
        "python_version": platform.python_version(), "gpu_used": False,
        "numerical_framework_imported": False, "random_seeds": "N/A: deterministic saved-record audit",
        "plan_file": args.plan_file,
        "result_file": str(args.output), "elapsed_seconds": time.monotonic() - started}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps({k: result[k] for k in ("indexed_results_checked", "tensor_checksums_checked",
          "indexed_worker_seconds", "overhead_charged_seconds", "total_charged_seconds",
          "reserved_seconds", "uncommitted_seconds", "remaining_seconds",
          "invalid_artifacts", "statistical_conclusions") if k in result}, indent=2))
    return int(bool(result["invalid_artifacts"]) or result["budget_exceeded"])


if __name__ == "__main__":
    raise SystemExit(main())
