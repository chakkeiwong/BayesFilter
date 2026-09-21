"""Offline summary of this bounded repair; never launches or reassesses a sampler."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[3]


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    started = time.monotonic()
    indexes, fits, commands = [], [], []
    indexed_seconds = {"cpu_reference": 0.0, "gpu": 0.0}
    for path in sorted(ROOT.glob("*/run_index.json")):
        index = read(path)
        jobs = {j["design"]["design_id"]: j for j in index["plan"]["jobs"]}
        if any(j["status"] == "running" for j in index["jobs"].values()):
            raise ValueError("Cannot finalize while a numerical worker is running")
        profiles = []
        for name, planned in jobs.items():
            job = index["jobs"][name]
            design = planned["design"]
            seconds = sum(a["elapsed_seconds"] for a in job["attempts"])
            indexed_seconds[design["device"]] += seconds
            profiles.append({"design_id": name, "status": job["status"],
                             "seconds": seconds, "attempt_count": len(job["attempts"])})
            commands.append({"profile": path.parent.name, "design_id": name,
                             "design": design, "attempts": job["attempts"],
                             "device_provenance": "See each attempt's original manifest"})
        indexes.append({"path": str(path.relative_to(REPO)), "sha256": sha(path),
                        "source_identity": index["source"]["identity"], "jobs": profiles})
        for pipeline in sorted(path.parent.glob("*/dataset-*/*/pipeline.json")):
            value = read(pipeline)
            selected = set(value["selection"]["candidate_ids"])
            members = value["members"]
            if set(m["candidate_id"] for m in members) != set(value["verified_candidate_ids"]):
                raise ValueError("Pipeline members do not preserve the verified set")
            fits.append({
                "path": str(pipeline.relative_to(REPO)), "sha256": sha(pipeline),
                "profile": path.parent.name, "completion": value["completion"],
                "candidate_count": value["candidate_count"],
                "verified_count": len(value["verified_candidate_ids"]),
                "selection": value["selection"],
                "unassessed_count": sum(m["status"] == "unassessed_by_design" for m in members),
                "selected_posteriors": [{
                    "candidate_id": m["candidate_id"], "status": m["status"],
                    "decision": m.get("posterior", {}).get("decision"),
                    "passed": m.get("posterior", {}).get("passed"),
                    "warmup_results_per_chain": m.get("posterior", {}).get("warmup_results_per_chain"),
                    "retained_results_per_chain": m.get("recorded_retained_count"),
                } for m in members if m["candidate_id"] in selected],
            })
    tests = []
    for path in sorted(ROOT.glob("tests-*.xml")):
        root = ET.parse(path).getroot()
        suites = [root] if root.tag == "testsuite" else list(root)
        tests.append({"path": str(path.relative_to(REPO)), "sha256": sha(path),
                      **{key: sum(float(s.attrib.get(key, 0)) for s in suites)
                         for key in ("tests", "failures", "errors", "skipped", "time")}})
    diagnostics = []
    for relative in ("saved-search-r1/comparison.json", "saved-mode-r1.json",
                     "saved-mode-denominators-r2.json"):
        path = ROOT / relative
        value = read(path)
        diagnostics.append({"path": str(path.relative_to(REPO)), "sha256": sha(path),
                            **{key: value[key] for key in ("elapsed_seconds", "command", "device")}})
    build_path = ROOT / "guide-r1/build-manifest.json"
    build = read(build_path)
    # These are deliberately conservative accounting charges, not measured
    # performance. They include earlier builds, imports, coordinator work,
    # read-only inspections, and final reporting that lack separate timers.
    unmeasured_cpu_charge = 1000.0
    unmeasured_gpu_charge = 120.0
    nonindexed = {
        "cpu_reference": sum(t["time"] for t in tests)
            + sum(d["elapsed_seconds"] for d in diagnostics)
            + build["elapsed_seconds"] + unmeasured_cpu_charge,
        "gpu": unmeasured_gpu_charge,
    }
    previous = {"cpu_reference": 78475.8264957075, "gpu": 40224.73805749172}
    charged = {d: indexed_seconds[d] + nonindexed[d] for d in indexed_seconds}
    total = {d: previous[d] + charged[d] for d in charged}
    files = {str(p.relative_to(REPO)): sha(p)
             for p in sorted((REPO / "bayesfilter").rglob("*.py"))}
    source_identity = hashlib.sha256(json.dumps(
        files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Terminal inventory and accounting; no new statistical assessment",
        "command": [sys.executable, *sys.argv], "environment": sys.executable,
        "python_version": platform.python_version(), "numerical_workers_launched": 0,
        "gpu_used": False, "seeds": "N/A for this deterministic reader; worker designs below record seeds",
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                                      check=True, capture_output=True, text=True).stdout.strip(),
        "checkout_source_identity_at_summary": source_identity,
        "source_classification": "Numerical jobs belong to their frozen source; test XML alone does not bind source",
        "plan_file": "docs/plans/bayesfilter-hmc-repair-master-program-2026-09-16.md",
        "result_file": "docs/plans/bayesfilter-hmc-repair-master-result-2026-09-17.md",
        "script_sha256": sha(Path(__file__)), "indexes": indexes,
        "worker_commands_and_designs": commands, "ordinary_fits": fits,
        "fit_counts_by_profile": dict(Counter(f["profile"] for f in fits)),
        "tests": tests,
        "test_interpretation": "Batches overlap; failures and zero-test invocations remain charged. No summed unique-test claim.",
        "diagnostics": diagnostics,
        "guide_build": {"path": str(build_path.relative_to(REPO)), "sha256": sha(build_path)},
        "indexed_worker_seconds": indexed_seconds, "nonindexed_charged_seconds": nonindexed,
        "unmeasured_conservative_charge": {"cpu_reference": unmeasured_cpu_charge, "gpu": unmeasured_gpu_charge},
        "overhead_provenance": "Convenience-chosen conservative charge for untimed work; not measured device utilization",
        "master_charged_seconds": charged, "prior_campaign_charged_seconds": previous,
        "cumulative_charged_seconds": total,
        "remaining_original_24h_allowance_seconds": {d: 86400.0 - v for d, v in total.items()},
        "elapsed_seconds": time.monotonic() - started,
    }
    output = ROOT / "execution-evidence-r1.json"
    with output.open("x") as handle:
        json.dump(result, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({key: result[key] for key in (
        "fit_counts_by_profile", "nonindexed_charged_seconds", "master_charged_seconds",
        "cumulative_charged_seconds", "remaining_original_24h_allowance_seconds")}, indent=2))


if __name__ == "__main__":
    main()
