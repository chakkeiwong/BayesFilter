#!/usr/bin/env python
"""CLI for the executable Younis score master; validation imports no TF."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

# Validation and reporting must not trigger the package's opt-in TF preload.
os.environ["BAYESFILTER_PRELOAD_CUSTOM_OP"] = "0"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bayesfilter.score_study.contracts import validate_study
from bayesfilter.score_study.coordinator import execute, report
from bayesfilter.score_study.registry import default_registry


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--action", choices=("validate", "dry-run", "run", "resume", "report", "select", "compare", "combine", "fd-report", "fd-select", "normalization", "consistency"), required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--related-runs", nargs=3, type=Path)
    args = parser.parse_args()
    study = json.loads(args.study.read_text())
    registry = default_registry()
    decisions = validate_study(study, registry)
    if args.action in ("validate", "dry-run"):
        print(json.dumps({"valid": True, "rows": decisions}, indent=2))
        return 0
    if args.output is None:
        parser.error("--output is required for execution or reporting")
    if args.action == "normalization":
        from bayesfilter.score_study.normalization_reporting import assemble_normalization
        result = assemble_normalization(args.output)
        print(json.dumps({"normalization": str(args.output / "normalization.json"), "groups": len(result["groups"])}))
        return 0
    if args.action == "consistency":
        if args.related_runs is None:
            parser.error("--related-runs requires N, 2N, 4N run directories")
        from bayesfilter.score_study.normalization_reporting import assemble_consistency
        result = assemble_consistency(args.related_runs, args.output / "consistency.json")
        print(json.dumps({"consistency": str(args.output / "consistency.json"), "groups": len(result["groups"])}))
        return 0
    if args.action == "fd-select":
        if args.selection is None: parser.error("--selection is required")
        from bayesfilter.score_study.fd_selection import issue_fd_selection
        issue_fd_selection(args.output,args.selection)
        print(json.dumps({"selection":str(args.selection),"default_ready":False}))
        return 0
    if args.action == "fd-report":
        from bayesfilter.score_study.fd_reporting import assemble_fd_diagnostics
        result = assemble_fd_diagnostics(args.output)
        print(json.dumps({"finite_differences": str(args.output / "finite-differences.json"),
                          "statistically_supported_ranking": result["statistically_supported_ranking"]}))
        return 0
    if args.action == "combine":
        from bayesfilter.score_study.combinations import assemble_combinations
        result = assemble_combinations(args.output)
        print(json.dumps({"combinations": str(args.output / "combinations.json"),
                          "inference_status": result["inference_status"]}))
        return 0
    if args.action == "select":
        if args.selection is None:
            parser.error("--selection is required for offline selection")
        from bayesfilter.score_study.tuning import issue_selection
        issue_selection(args.output, args.selection)
        print(json.dumps({"selection": str(args.selection), "default_ready": False}))
        return 0
    if args.action == "compare":
        from bayesfilter.score_study.reporting import assemble_diagnostics
        result = assemble_diagnostics(args.output)
        print(json.dumps({"comparison": str(args.output / "comparison.json"), "failed_rows": len(result["failed_or_missing_rows"])}))
        return 0 if not result["failed_or_missing_rows"] else 2
    if args.action == "report":
        result = report(args.output, registry)
    else:
        result = execute(study, registry, args.output, resume=args.action == "resume")
    print(json.dumps({"execution_status": result["execution_status"], "output": str(args.output)}))
    return 0 if result["execution_status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
