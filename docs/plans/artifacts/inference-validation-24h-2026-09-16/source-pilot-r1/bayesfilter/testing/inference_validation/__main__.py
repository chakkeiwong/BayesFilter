"""Explicit profile CLI; list/plan/report do not import numerical frameworks."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

from .catalog import TARGETS
from .storage import read_json,write_json


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    sub.add_parser("list")
    plan=sub.add_parser("plan"); plan.add_argument("suite",type=Path);plan.add_argument("--output",type=Path)
    run=sub.add_parser("run");run.add_argument("suite",type=Path);run.add_argument("--output",type=Path,required=True)
    run.add_argument("--resume",action="store_true");run.add_argument("--max-jobs",type=int)
    report=sub.add_parser("report");report.add_argument("root",type=Path)
    assess=sub.add_parser("assess");assess.add_argument("root",type=Path);assess.add_argument("--output",type=Path,required=True)
    worker=sub.add_parser("_worker");worker.add_argument("design",type=Path);worker.add_argument("root",type=Path);worker.add_argument("budget",type=float)
    worker.add_argument("attempt",type=int)
    args=parser.parse_args(argv)
    if args.command=="list":
        print(json.dumps({k:v.payload() for k,v in TARGETS.items()},indent=2)); return 0
    if args.command=="plan":
        from .execution import plan_suite
        result=plan_suite(read_json(args.suite))
        if args.output: write_json(args.output,result)
        print(json.dumps(result,indent=2)); return 0
    if args.command=="run":
        from .execution import run_suite
        index=run_suite(read_json(args.suite),args.output,resume=args.resume,max_jobs=args.max_jobs)
        return 1 if any(j["status"] in {"failed","timed_out"} for j in index["jobs"].values()) else 0
    if args.command=="report":
        from .reporting import report
        report(args.root); return 0
    if args.command=="assess":
        from .assessment import assess
        assess(args.root,args.output); return 0
    from .execution import worker
    return worker(args.design,args.root,args.budget,args.attempt)


if __name__=="__main__": raise SystemExit(main())
