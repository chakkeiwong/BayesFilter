"""Freeze a local source version and resolve suites without numerical execution."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from bayesfilter.testing.inference_validation.execution import plan_suite, source_state
from bayesfilter.testing.inference_validation.storage import read_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("suites",nargs="+",type=Path)
    args=parser.parse_args()
    root=args.output.resolve()
    if root.exists(): raise FileExistsError(root)
    before=source_state()
    root.mkdir(parents=True)
    shutil.copytree(REPO/"bayesfilter",root/"bayesfilter",
                    ignore=shutil.ignore_patterns("__pycache__","*.pyc"))
    copied={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((root/"bayesfilter").rglob("*.py"))}
    if before["files"]!=copied or before!=source_state():
        raise RuntimeError("source changed while copying; preserve failed snapshot and start fresh")
    plans=[]
    for path in args.suites:
        plan=plan_suite(read_json(path))
        dest=root/"suites"/path.name
        dest.parent.mkdir(exist_ok=True)
        shutil.copyfile(path,dest)
        with dest.with_suffix(".plan.json").open("x") as f: json.dump(plan,f,indent=2)
        plans.append({"suite":str(dest),"sha256":hashlib.sha256(dest.read_bytes()).hexdigest(),
                      "budget_by_device":plan["budget_by_device"]})
    manifest={"created_utc":datetime.now(timezone.utc).isoformat(),"git_commit":before["commit"],
              "source_identity":before["identity"],"source_files":copied,"command":sys.argv,
              "origin":str(REPO),"plans":plans,"numerical_execution":False}
    with (root/"source_snapshot.json").open("x") as f: json.dump(manifest,f,indent=2)
    print(json.dumps({"source":str(root),"source_identity":before["identity"],"plans":plans},indent=2))


if __name__=="__main__": main()
