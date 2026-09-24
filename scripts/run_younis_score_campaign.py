#!/usr/bin/env python
"""Bounded multi-scope calibration then untouched evaluation of the score master."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

os.environ["BAYESFILTER_PRELOAD_CUSTOM_OP"] = "0"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bayesfilter.score_study.contracts import validate_study
from bayesfilter.score_study.coordinator import execute, write_json
from bayesfilter.score_study.registry import default_registry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--action", required=True, choices=("validate", "run", "compare"))
    args = parser.parse_args()
    campaign = json.loads(args.campaign.read_text())
    registry = default_registry()
    stages = campaign["calibration"] + campaign["evaluation"]
    studies = [json.loads(Path(stage["study"]).read_text()) for stage in stages]
    for study in studies:
        validate_study(study, registry)
    rows = sum(len(s["rows"]) for s in studies)
    if rows > campaign["max_numerical_rows"]:
        raise ValueError("campaign row budget exceeded")
    if args.action == "validate":
        print(json.dumps({"valid": True, "studies": len(studies), "rows": rows}))
        return 0
    if args.action == "compare":
        from bayesfilter.score_study.heldout_reporting import assemble_heldout
        result = assemble_heldout([x["output"] for x in campaign["evaluation"]],
                                 campaign["comparison_output"], campaign["comparison_contract"])
        print(json.dumps({"rows": len(result["rows"]), "heuristic_dominance": result["heuristic_dominance_verdict"]}))
        return 0
    progress_path = Path(campaign["progress_output"])
    if progress_path.exists() or any(Path(s["output"]).exists() for s in stages):
        raise ValueError("use a fresh versioned campaign; prior attempts are preserved")
    started = time.monotonic()
    progress = {"schema": "younis_score_campaign_progress_v1", "command": sys.argv,
                "campaign": str(args.campaign), "completed": [], "status": "calibration",
                "wall_seconds": 0., "evaluation_opened": False}
    write_json(progress_path, progress)
    try:
        for stage, study in zip(stages, studies):
            if time.monotonic() - started >= campaign["wall_seconds"]:
                raise RuntimeError("campaign wall budget exhausted")
            if stage in campaign["evaluation"] and not progress["evaluation_opened"]:
                if len(progress["completed"]) != len(campaign["calibration"]):
                    raise ValueError("incomplete calibration before evaluation")
                progress.update(status="evaluation", evaluation_opened=True)
                write_json(progress_path, progress)
            result = execute(study, registry, Path(stage["output"]), resume=False)
            if result["execution_status"] != "complete":
                raise RuntimeError("incomplete study: " + stage["output"])
            if "selection" in stage:
                from bayesfilter.score_study.tuning import issue_selection
                issue_selection(stage["output"], stage["selection"])
            progress["completed"].append({"output": stage["output"], "rows": len(study["rows"]),
                                           "wall_seconds": result["wall_seconds"]})
            progress["wall_seconds"] = time.monotonic() - started
            write_json(progress_path, progress)
            print(json.dumps(progress["completed"][-1]), flush=True)
        progress["status"] = "complete"
    except Exception as error:
        progress.update(status="failed", error=type(error).__name__ + ": " + str(error))
        raise
    finally:
        progress["wall_seconds"] = time.monotonic() - started
        write_json(progress_path, progress)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
