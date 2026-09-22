"""Read M7 saved GPU evidence and host profiles without importing a framework."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pstats
import sys

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def profile_summary(path):
    stats = pstats.Stats(str(path))
    rows = []
    for (file, line, name), (primitive, calls, own, cumulative, callers) in stats.stats.items():
        rows.append({"file": file, "line": line, "name": name,
                     "calls": calls, "primitive_calls": primitive,
                     "self_seconds": own, "cumulative_seconds": cumulative})
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "total_seconds": stats.total_tt, "total_calls": stats.total_calls,
            "top_self": sorted(rows, key=lambda x: x["self_seconds"], reverse=True)[:20],
            "top_cumulative": sorted(rows, key=lambda x: x["cumulative_seconds"], reverse=True)[:30],
            "interpretation": "Host elapsed profiles; cumulative rows overlap. Framework-call time includes compilation/execution unless separately measured."}


def summarize():
    rows = []
    for index_path in sorted(ROOT.glob("*-gpu-r*/run_index.json")):
        index = read(index_path)
        for planned in index["plan"]["jobs"]:
            d = planned["design"]
            name = d["design_id"]
            job = index["jobs"].get(name, {})
            row = {"design_id": name, "target": d["scenario"]["target"],
                   "engine": d["engine"], "execution_status": job.get("status", "not_run"),
                   "worker_seconds": sum(a["elapsed_seconds"] for a in job.get("attempts", [])),
                   "source_identity": index["source"]["identity"]}
            directory = index_path.parent / name
            row["failures"] = [read(p) for p in sorted(directory.glob("attempt-*-failure.json"))]
            if job.get("result"):
                result = read(Path(job["result"]))
                assessment = result["assessment"]
                row["finding"] = assessment["finding"]
                row["runtime"] = result["runtime"]
                if d["engine"] == "mechanics":
                    row["control"] = d["scenario"]["control"]
                    row["epsilon"] = d["step_size"]
                    row["oracle"] = {k: assessment[k] for k in (
                        "metropolis_log_ratio_max_error", "metropolis_log_ratio_passed",
                        "metropolis_state_selection_passed", "density_passed", "score_passed", "leapfrog_passed")}
                elif d["engine"] == "power":
                    base = d["options"]["calibration_design"]
                    row.update(epsilon=base["step_size"], power=base["options"]["kernel_power"],
                               rates=assessment["rates"])
                    experiments = [read(p) for p in sorted(directory.glob("trial-*/*/invariance.json"))]
                    row["experiment_count"] = len(experiments)
                    row["every_substep_finite"] = bool(experiments) and all(
                        p["all_substep_states_and_log_ratios_finite"] for p in experiments)
                else:
                    row["assessment"] = {k: v for k, v in assessment.items() if k != "replications"}
            pipelines = []
            for path in sorted(directory.glob("replication-*/pipeline.json")):
                p = read(path)
                posterior = []
                for member in p["members"]:
                    info = {k: member.get(k) for k in ("candidate_id", "status", "L", "epsilon", "timing")}
                    if "posterior" in member:
                        info["posterior"] = {k: member["posterior"].get(k) for k in (
                            "passed", "warmup_cap_hit", "retained_cap_hit", "warmup_results_per_chain",
                            "retained_results_per_chain", "status", "reason")}
                    posterior.append(info)
                pipelines.append({"path": str(path), "timing": p["timing"], "completion": p["completion"],
                                  "candidate_count": p["candidate_count"],
                                  "verified_count": len(p["verified_candidate_ids"]),
                                  "selection": p["selection"], "members": posterior})
            row["pipelines"] = pipelines
            row["profiles"] = [profile_summary(path) for path in sorted(directory.glob("attempt-*-host.prof"))]
            rows.append(row)
    return {"created_utc": datetime.now(timezone.utc).isoformat(), "command": sys.argv,
            "plan_file": "docs/plans/bayesfilter-hmc-repair-master-program-2026-09-16.md",
            "numerical_framework_imported": False, "rows": rows,
            "inference_status": "Development sensitivity and availability only. No ranking, calibrated stopping, global convergence or default promotion."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize()
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    for row in result["rows"]:
        print(row["design_id"], row["execution_status"], row.get("finding"), round(row["worker_seconds"], 3))


if __name__ == "__main__":
    main()
