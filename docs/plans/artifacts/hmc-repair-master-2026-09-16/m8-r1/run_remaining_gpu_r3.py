"""Bounded continuation of unstarted M8 work after the reviewed scheduling repair."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from run_remaining_gpu import ROOT, PLAN, read, audit_suite

SOURCE = ROOT / "source-gpu-r6"
OUTPUT = ROOT / "stopping-fresh-gpu-r1"


def write_new(path, payload):
    with path.open("x") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")


def matched_fit(case, phase, replication, seconds):
    result = subprocess.run([sys.executable, str(ROOT / "run_posteriordb.py"),
        "--source", str(SOURCE), "--case", case, "--phase", phase,
        "--replication", str(replication), "--seconds", str(seconds)])
    tag = "schools" if case.startswith("eight_schools") else "regression"
    output = ROOT / f"posteriordb-{tag}-{phase}-{replication}-gpu-r1"
    record = read(output / "gpu-diagnostic-run.json")
    if result.returncode:
        failure = read(output / "failure.json") if (output / "failure.json").exists() else {}
        if failure.get("exception") != "HMCPreparationFailure":
            raise RuntimeError("unresolved matched-reference infrastructure failure: " + str(output))
    else:
        read(output / "assessment.json")
    return record["elapsed_seconds"]


def references():
    cases = ("eight_schools-eight_schools_noncentered", "sblrc-blr")
    pilots = [matched_fit(case, "pilot", 0, 1800) for case in cases]
    caps = [max(1800, math.ceil(seconds * 1.5 / 100) * 100) for seconds in pilots]
    record = {"plan_file": PLAN, "pilot_seconds": dict(zip(cases, pilots)),
        "fresh_caps": dict(zip(cases, caps)), "reservation_seconds": 16000,
        "worst_case_charged_seconds": sum(pilots) + 3 * sum(caps),
        "scientific_contract_changed": False}
    write_new(ROOT / "posteriordb-measured-fresh-caps.json", record)
    if record["worst_case_charged_seconds"] > 16000:
        raise RuntimeError("review additional matched-reference reservation before fresh fits")
    for case, cap in zip(cases, caps):
        for replication in range(3):
            matched_fit(case, "fresh", replication, cap)
    print("MATCHED REFERENCE QUEUE COMPLETE", flush=True)


def resource_sample():
    memory = subprocess.check_output(["nvidia-smi", "--id=1", "--query-gpu=memory.used",
        "--format=csv,noheader,nounits"], text=True).strip()
    host = {line.split(":")[0]: int(line.split()[1])
            for line in Path("/proc/meminfo").read_text().splitlines()
            if line.startswith(("MemTotal:", "MemAvailable:"))}
    return {"time": datetime.now(timezone.utc).isoformat(), "gpu_used_mib": float(memory),
            "host_used_mib": (host["MemTotal"] - host["MemAvailable"]) / 1024.}


def stopping(workers, max_jobs, label):
    suite = ROOT / "m8-stopping-fresh-interleaved-gpu.json"
    original = read(SOURCE / "suites/m8-stopping-fresh-gpu.json")
    rearranged = read(suite)
    assert sorted(json.dumps(d, sort_keys=True) for d in original["designs"]) == sorted(
        json.dumps(d, sort_keys=True) for d in rearranged["designs"])
    command = [sys.executable, "-m", "bayesfilter.testing.inference_validation", "run",
        str(suite), "--output", str(OUTPUT), "--max-workers", str(workers)]
    if OUTPUT.exists():
        command.append("--resume")
    if max_jobs is not None:
        command.extend(["--max-jobs", str(max_jobs)])
    started = time.monotonic()
    before = read(OUTPUT / "run_index.json")["jobs"] if OUTPUT.exists() else {}
    when = datetime.now(timezone.utc).isoformat()
    samples = [resource_sample()]
    print("START", label, when, flush=True)
    with (ROOT / (label + "-coordinator.log")).open("x") as log:
        process = subprocess.Popen(command, cwd=SOURCE, stdout=log, stderr=subprocess.STDOUT)
        while process.poll() is None:
            samples.append(resource_sample())
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                pass
    record = {"command": command, "cwd": str(SOURCE), "started_utc": when,
        "coordinator_wall_seconds": time.monotonic() - started, "returncode": process.returncode,
        "resource_samples": samples, "plan_file": PLAN,
        "max_gpu_used_mib": max(s["gpu_used_mib"] for s in samples),
        "max_host_used_mib": max(s["host_used_mib"] for s in samples),
        "accounting": "charge indexed worker attempts; coordinator overlaps those workers"}
    index = read(OUTPUT / "run_index.json")
    new_jobs = {k: v for k, v in index["jobs"].items()
                if v != before.get(k)}
    record["completed_jobs"] = {key: value["status"] for key, value in new_jobs.items()}
    record["worker_seconds"] = sum(a["elapsed_seconds"] for key, job in new_jobs.items()
        for a in job["attempts"][len(before.get(key, {}).get("attempts", [])):])
    write_new(ROOT / (label + "-coordinator.json"), record)
    if process.returncode or any(j["status"] != "complete" for j in new_jobs.values()):
        raise RuntimeError("stopping tranche has unresolved attempts; preserve and inspect")
    if record["max_gpu_used_mib"] >= 8192 or record["max_host_used_mib"] >= 102400:
        raise RuntimeError("resource-sharing bound reached; reduce subsequent concurrency")
    for key, job in new_jobs.items():
        path = Path(job["result"])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == job["result_sha256"]
        runtime = read(path)["runtime"]
        assert runtime["device_scope"] == "gpu" and runtime["jit_compile"]
        assert runtime["memory_policy"]["all_physical_devices_memory_growth"]
        assert runtime["memory_policy"]["configured_before_logical_device_initialization"]
    pilot_cost, _ = audit_suite(ROOT / "stopping-pilot-gpu-r1")
    spent = pilot_cost + sum(a["elapsed_seconds"] for j in index["jobs"].values() for a in j["attempts"])
    remaining = 70000 - spent
    maxima, counts = {}, {}
    for planned in index["plan"]["jobs"]:
        d = planned["design"]
        target = d["scenario"]["target"]
        job = index["jobs"].get(d["design_id"])
        if job and job["status"] == "complete":
            maxima[target] = max(maxima.get(target, 0), sum(a["elapsed_seconds"] for a in job["attempts"]))
        else:
            counts[target] = counts.get(target, 0) + 1
    projection = sum(count * maxima[target] * 1.5 for target, count in counts.items())
    write_new(ROOT / (label + "-review.json"), {"spent": spent, "remaining": remaining,
        "target_max_cost": maxima, "remaining_counts": counts,
        "remaining_projection_with_50_percent_margin": projection,
        "affordable": projection <= remaining, "resource_record": label + "-coordinator.json"})
    if projection > remaining:
        raise RuntimeError("measured remaining projection needs resource review")
    print("FINISHED", label, "worker_seconds", record["worker_seconds"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("references", "stopping"))
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-jobs", type=int)
    parser.add_argument("--label", default="stopping-concurrency-two")
    args = parser.parse_args()
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "1"
    assert os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH", "").lower() == "true"
    if args.phase == "references":
        references()
    else:
        stopping(args.workers, args.max_jobs, args.label)


if __name__ == "__main__":
    main()
