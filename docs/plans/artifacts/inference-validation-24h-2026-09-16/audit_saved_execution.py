"""Deterministic, diagnostic-only audit of this campaign's saved execution.

No framework is imported, sampler launched, or posterior result promoted.
Run after numerical workers finish; use a fresh output path for each audit.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time


def read(path):
    def reject(value):
        raise ValueError(f"nonfinite JSON: {value}")
    return json.loads(path.read_text(), parse_constant=reject)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    root = Path(__file__).resolve().parent
    issues, snapshots, profiles = [], [], []

    def check(condition, path, reason):
        if not condition:
            issues.append({"path": str(path.relative_to(root)), "reason": reason})

    for manifest_path in sorted(root.glob("source-*/source_snapshot.json")):
        manifest = read(manifest_path)
        actual = {str(p.relative_to(manifest_path.parent)): sha(p)
                  for p in sorted((manifest_path.parent / "bayesfilter").rglob("*.py"))}
        check(actual == manifest["source_files"], manifest_path, "frozen source bytes changed")
        check(digest(actual) == manifest["source_identity"], manifest_path, "source identity mismatch")
        for plan in manifest["plans"]:
            check(sha(Path(plan["suite"])) == plan["sha256"], manifest_path, "frozen suite changed")
        snapshots.append({"path": str(manifest_path.relative_to(root)),
                          "identity": manifest["source_identity"], "file_count": len(actual)})
    source_ids = {row["identity"] for row in snapshots}

    for index_path in sorted(root.glob("*/run_index.json")):
        index = read(index_path)
        profile_root = index_path.parent
        check(index["source"]["identity"] in source_ids, index_path, "index has no frozen source")
        planned = {j["design"]["design_id"]: j for j in index["plan"]["jobs"]}
        runtime_counts, counts, outcomes, pair_status = Counter(), Counter(), Counter(), Counter()
        zero_members, slowest = [], []
        manifests = []
        for path in sorted(profile_root.glob("*/attempt-*-manifest.json")):
            manifest = read(path)
            design = planned[path.parent.name]
            runtime = manifest["runtime"]
            device = design["design"]["device"]
            check(manifest["source"]["identity"] == index["source"]["identity"], path, "worker source mismatch")
            check(digest(manifest["design"]) == design["identity"], path, "worker design mismatch")
            check(runtime["device_scope"] == device, path, "worker device mismatch")
            policy = runtime["memory_policy"]
            if device == "gpu":
                check(runtime["jit_compile"] and not runtime["gpu_intentionally_hidden"], path, "GPU/XLA not active")
                check(policy["configured_before_logical_device_initialization"]
                      and policy["full_device_preallocation_disabled"]
                      and policy["all_physical_devices_memory_growth"]
                      and policy["tf_force_gpu_allow_growth"].lower() == "true"
                      and policy["physical_devices"]
                      and all(d["memory_growth"] for d in policy["physical_devices"]),
                      path, "GPU memory growth not established")
            else:
                check(runtime["gpu_intentionally_hidden"] and runtime["visible_devices"] == "-1",
                      path, "CPU reference did not hide GPUs")
            runtime_counts[device] += 1
            manifests.append({"path": str(path.relative_to(root)), "sha256": sha(path),
                              "device": device, "runtime": runtime})

        for path in sorted(profile_root.rglob("tuning_observation.json")):
            payload = read(path)
            candidates = {c["candidate_id"]: c for c in payload["candidates"]}
            verified = set(payload["verified_candidate_ids"])
            counts["tuning_records"] += 1
            counts["candidate_records"] += len(candidates)
            counts["verified_members"] += len(verified)
            check(len(candidates) == len(payload["candidates"]), path, "duplicate candidate ID")
            check(verified == {k for k, v in payload["candidate_states"].items() if v == "verified"},
                  path, "verified candidate inventory mismatch")
            check(payload["nominee_id"] is None, path, "unexpected nominee")
            observations = payload["observations"]
            streams = [r["observation"].get("stream_id") for r in observations]
            check(None not in streams and len(streams) == len(set(streams)), path, "missing/reused evidence stream")
            for row in observations:
                diagnostic = row["observation"].get("rhat_reporting_only", {})
                check(diagnostic.get("role") == "reporting_only", path, "R-hat diagnostic role changed")
            for cid in verified:
                counts["verified_with_nonpassing_rhat"] += int(any(
                    r["candidate_id"] == cid and r["stage"] == "verification"
                    and r["observation"]["decision"] == "passed"
                    and r["observation"]["promotion_eligible"]
                    and r["observation"].get("rhat_reporting_only", {}).get("passed") is False
                    for r in observations))
                receipts = [v for v in payload["verification_receipts"]
                            if v["candidate_id"] == cid and v["stage"] == "verification"
                            and v["decision"] == "passed" and v["promotion_eligible"]]
                check(bool(receipts), path, "verified member lacks fresh qualifying receipt")
                for receipt in receipts:
                    c = candidates[cid]
                    check(receipt["epsilon"] == c["epsilon"] and receipt["exact_l"] == c["leapfrog_steps"]
                          and receipt["mass_signature"] == c["mass_signature"], path, "receipt changed kernel")
            if not verified:
                zero_members.append({"path": str(path.relative_to(root)),
                    "completion": payload["completion_status"],
                    "epsilon_domain": payload["scope"]["epsilon_domain"],
                    "pairs": [[c["leapfrog_steps"], c["epsilon"]] for c in candidates.values()],
                    "decisions": dict(Counter(r["observation"]["decision"] for r in observations)),
                    "repair_limits": [r for r in payload["accounting_events"]
                                      if r["event"] == "repair_limit_exhausted"]})
            pipeline_path = path.with_name("pipeline.json")
            if not pipeline_path.exists():
                counts["tuning_records_without_finished_pipeline"] += 1
                continue
            pipeline = read(pipeline_path)
            members = {m["candidate_id"]: m for m in pipeline["members"]}
            check(set(members) == verified == set(pipeline["verified_candidate_ids"]),
                  pipeline_path, "pipeline dropped or added a verified member")
            check(len(members) == len(pipeline["members"]), pipeline_path, "duplicate pipeline member")
            check(pipeline["candidate_count"] == len(candidates), pipeline_path, "candidate count mismatch")
            selected = set(pipeline["selection"]["candidate_ids"])
            selection = read(path.with_name("posterior_selection.json"))
            check(selected == set(selection["selected_candidate_ids"]), pipeline_path, "selection changed after sampling")
            if pipeline["selection"]["rule"] == "first_verified":
                check(selected == set(sorted(verified)[:1]), pipeline_path, "first_verified rule mismatch")
            for cid, member in members.items():
                counts["member_status:" + member["status"]] += 1
                if member["status"] == "assessed":
                    posterior = member["posterior"]
                    check(member["warmup_exclusion_matches"], pipeline_path, "warmup entered retained draws")
                    check(member["recorded_retained_count"] == posterior["retained_results_per_chain"],
                          pipeline_path, "retained count mismatch")
                    outcomes[posterior["decision"]] += 1
                    pair_status[member.get("fixed_comparator", {}).get("status", "not_requested")] += 1
            slowest.append({"pipeline": str(pipeline_path.relative_to(root)), **pipeline["timing"]})
        profiles.append({"profile": profile_root.name, "source_identity": index["source"]["identity"],
            "runtime_manifest_counts": dict(runtime_counts), "runtime_manifests": manifests,
            "counts": dict(counts), "posterior_outcomes": dict(outcomes),
            "fixed_comparator_statuses": dict(pair_status), "zero_member_fits": zero_members,
            "slowest_pipelines": sorted(slowest, key=lambda r: r["invocation_seconds"], reverse=True)[:5]})
    result = {"schema": "bayesfilter.campaign_saved_execution_audit.v1", "snapshots": snapshots,
        "profiles": profiles, "issues": issues, "numerical_workers_launched": 0,
        "interpretation": "Saved-record consistency and provenance only; no posterior calibration conclusion.",
        "manifest": {"created_utc": datetime.now(timezone.utc).isoformat(), "command": sys.argv,
            "python": platform.python_version(), "environment": sys.executable,
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "script_sha256": sha(Path(__file__)), "random_seeds": "N/A: deterministic audit",
            "plan_file": "docs/plans/bayesfilter-inference-validation-24h-campaign-2026-09-16.md",
            "result_file": str(args.output), "gpu_used": False,
            "elapsed_seconds": time.monotonic() - started}}
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"sources_checked": len(snapshots),
        "tuning_records_checked": sum(p["counts"].get("tuning_records", 0) for p in profiles),
        "issues": issues}, indent=2))
    return int(bool(issues))


if __name__ == "__main__":
    raise SystemExit(main())
