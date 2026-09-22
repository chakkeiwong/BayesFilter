"""Diagnostic-only summary of declared campaign groups; never pool retries.

Read saved results with ordinary JSON and use SciPy only for descriptive
binomial intervals across the predeclared independent stopping replications.
"""
from collections import Counter
from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

from scipy.stats import beta


def interval(count, total):
    if not total:
        return None
    return [0. if not count else float(beta.ppf(.025, count, total-count+1)),
            1. if count == total else float(beta.ppf(.975, count+1, total-count))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    root = Path(__file__).resolve().parent
    result = {"schema": "bayesfilter.campaign_saved_results_summary.v1",
              "profiles": [], "stopping_groups": [], "sbc_groups": [],
              "no_ranking_or_calibration_promotion": True}
    for index_path in sorted(root.glob("*/run_index.json")):
        index = json.loads(index_path.read_text())
        rows = []
        for job in index["plan"]["jobs"]:
            design = job["design"]
            name = design["design_id"]
            execution = index["jobs"].get(name, {"status": "not_run"})
            row = {"design": name, "target": design["scenario"]["target"],
                   "route": design["scenario"]["route"], "engine": design["engine"],
                   "status": execution["status"], "seconds": sum(a["elapsed_seconds"]
                   for a in execution.get("attempts", []))}
            if execution["status"] == "complete":
                path = Path(execution["result"])
                if hashlib.sha256(path.read_bytes()).hexdigest() != execution["result_sha256"]:
                    raise ValueError(f"result checksum changed: {path}")
                assessment = json.loads(path.read_text())["assessment"]
                for key in ("finding", "planned", "completed", "assessment_complete", "comparison_complete",
                            "verified_members", "assessed_members", "posterior_output_members",
                            "rates", "tests"):
                    if key in assessment:
                        row[key] = assessment[key]
                row["result_path"] = str(path.relative_to(root))
            rows.append(row)
        result["profiles"].append({"profile": index_path.parent.name,
                                  "source_identity": index["source"]["identity"], "jobs": rows})
        for path in sorted(index_path.parent.glob("*-aggregate.json")):
            data = json.loads(path.read_text())
            result["sbc_groups"].append({"path": str(path.relative_to(root)),
                **{k: v for k, v in data.items() if k != "datasets"}})

    # These two target groups and eight replications per device were declared
    # together before launch. Device arms and changed procedures remain separate.
    for profile in ("main-cpu-r1", "repair-gpu-r1"):
        for target in ("gaussian", "mixture"):
            planned = 8
            records, posterior, decisions, coverage = [], [], Counter(), {}
            for path in sorted((root/profile).glob(f"*-{target}-stopping-*/attempt-001-result.json")):
                assessment = json.loads(path.read_text())["assessment"]
                for row in assessment["replications"]:
                    pipeline = json.loads(Path(row["pipeline"]).read_text())
                    selected = set(pipeline["selection"]["candidate_ids"])
                    members = [m for m in row["members"] if m["candidate_id"] in selected]
                    for member in members:
                        if "member_record" not in member:
                            continue
                        m = member["member_record"]
                        p = m["posterior"]
                        record = {"result_path": str(path.relative_to(root)),
                            "replication": row["replication"], "candidate_id": m["candidate_id"],
                            "passed": p["passed"], "warmup": p["warmup_results_per_chain"],
                            "retained": p["retained_results_per_chain"],
                            "warmup_cap_hit": p["warmup_cap_hit"], "retained_cap_hit": p["retained_cap_hit"],
                            "finding": member["assessment"]["finding"],
                            "false_favorable": member["false_favorable_screen_observed"],
                            "stopping_pair": member.get("stopping_pair")}
                        records.append(record)
                        decisions[p["decision"]] += 1
            names = sorted({name for row in records if row["stopping_pair"]
                            for arm in ("fixed", "stopped")
                            for name in row["stopping_pair"].get(arm, {})})
            for name in names:
                coverage[name] = {}
                for arm in ("fixed", "stopped"):
                    quantities = [(r["stopping_pair"] or {}).get(arm, {}).get(name, {}) for r in records]
                    available = sum(bool(q.get("available")) for q in quantities)
                    covered = sum(bool(q.get("covered")) for q in quantities)
                    coverage[name][arm] = {"planned": planned, "available": available,
                        "covered": covered, "coverage_interval": interval(covered, planned),
                        "unavailable": planned-available}
            false_count = sum(r["false_favorable"] for r in records)
            result["stopping_groups"].append({"profile": profile, "target": target,
                "planned": planned, "records": records, "completed_records": len(records),
                "posterior_decisions": dict(decisions), "coverage": coverage,
                "false_favorable": false_count, "false_favorable_interval": interval(false_count, planned),
                "interpretation": "small predeclared development group; no sequential-coverage or method-ranking claim"})
    result["manifest"] = {"created_utc": datetime.now(timezone.utc).isoformat(), "command": sys.argv,
        "environment": sys.executable, "python": platform.python_version(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "random_seeds": "N/A: deterministic saved-result summary", "gpu_used": False,
        "plan_file": "docs/plans/bayesfilter-inference-validation-24h-campaign-2026-09-16.md",
        "result_file": str(args.output), "elapsed_seconds": time.monotonic()-started}
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"profiles": len(result["profiles"]), "stopping_groups": len(result["stopping_groups"]),
                      "sbc_groups": len(result["sbc_groups"])}))


if __name__ == "__main__":
    main()
