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
    parser.add_argument("--action", choices=("validate", "dry-run", "run", "resume", "report", "select", "compare"), required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--selection", type=Path)
    args = parser.parse_args()
    study = json.loads(args.study.read_text())
    registry = default_registry()
    decisions = validate_study(study, registry)
    if args.action in ("validate", "dry-run"):
        print(json.dumps({"valid": True, "rows": decisions}, indent=2))
        return 0
    if args.output is None:
        parser.error("--output is required for execution or reporting")
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
