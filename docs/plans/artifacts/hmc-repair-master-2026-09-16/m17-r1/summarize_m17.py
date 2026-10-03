"""Preserve every M17 model/route outcome, including unavailable posteriors."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def main():
    rows = []
    for name in ("matrix-cpu-r1", "matrix-gpu-r1"):
        index = read(ROOT / name / "run_index.json")
        for planned in index["plan"]["jobs"]:
            design = planned["design"]
            job = index["jobs"].get(design["design_id"], {"status": "not_run", "attempts": []})
            row = {"design_id": design["design_id"], "device": design["device"],
                   "target": design["scenario"]["target"], "route": design["scenario"]["route"],
                   "status": job["status"], "result": job.get("result"),
                   "elapsed_seconds": sum(a["elapsed_seconds"] for a in job["attempts"])}
            if job.get("result"):
                result = read(Path(job["result"]))
                assessment = result.get("assessment", {})
                row.update(finding=assessment.get("finding"),
                           search_complete=assessment.get("search_complete"),
                           assessment_complete=assessment.get("assessment_complete"),
                           verified_members=assessment.get("verified_members"),
                           assessed_members=assessment.get("assessed_members"),
                           unassessed_members=assessment.get("unassessed_by_design_members"),
                           failure=result.get("error"))
                row["members"] = [{key: member.get(key) for key in
                    ("candidate_id", "L", "epsilon", "runtime_checks_passed", "warmup_count", "retained_count",
                     "warmup_exclusion_matches", "duplicate_chains", "false_favorable_screen_observed", "stopped_intervals")}
                    | {"reference_finding": member["assessment"]["finding"]}
                    for replication in assessment.get("replications", [])
                    for member in replication["members"] if "assessment" in member]
            rows.append(row)
    special = []
    for path in sorted(ROOT.glob("*/diagnostic-run.json")):
        record = read(path)
        assessment = path.parent / "assessment.json"
        failure = path.parent / "failure.json"
        observed = read(assessment) if assessment.exists() else None
        if observed and "posterior" in observed:
            posterior = observed["posterior"]
            observed = {key: value for key, value in observed.items()
                        if key not in {"posterior", "reference_metadata", "reference_rhat"}}
            observed["posterior"] = {key: posterior.get(key) for key in
                ("passed", "warmup_passed", "warmup_results_per_chain", "retained_results_per_chain",
                 "retained_passed", "warmup_excluded_from_posterior")}
        special.append({"path": str(path.parent), "device": record["device"],
                        "returncode": record["returncode"], "elapsed_seconds": record["elapsed_seconds"],
                        "assessment_path": str(assessment) if assessment.exists() else None,
                        "assessment": observed,
                        "failure": read(failure) if failure.exists() else None})
    result = {"matrix": rows, "position_field_and_matched_references": special,
              "macrofinance_original_reference": "unavailable: exact target/data/coordinates/reference not bound",
              "ranking_supported": False, "default_promoted": False,
              "interpretation": "Single-fit development cells; candidate, posterior and reference outcomes are separate."}
    with (ROOT / "terminal-summary.json").open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"matrix_cells": len(rows), "special_attempts": len(special),
                      "summary": str(ROOT / "terminal-summary.json")}))


if __name__ == "__main__":
    main()
