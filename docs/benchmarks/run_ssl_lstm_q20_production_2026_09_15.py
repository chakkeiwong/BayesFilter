#!/usr/bin/env python3
"""Supervised q20 master: validate, price, train, campaign, resume and status.

Numerical workers are children of the standard-library coordinator. Its
external deadlines include imports, initialization, native compilation and
all numerical stages. A failed scientific screen is reported, never promoted.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bayesfilter.inference.q20_production_config import (
    digest, load_protocol, protocol_template, validate_protocol, training_cohort,
)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=("validate","price","train","campaign","status","worker","write-config"))
    parser.add_argument("--config",type=Path)
    parser.add_argument("--output-dir",type=Path)
    parser.add_argument("--budget-record",type=Path,
                        help="existing remaining allowance and unsettled cost holds; required for a new campaign")
    parser.add_argument("--request",type=Path,help=argparse.SUPPRESS)
    parser.add_argument("--cpu-reference",action="store_true",help="explicit smoke-only CPU exception")
    args=parser.parse_args(argv)
    if args.mode=="worker":
        if args.request is None or args.output_dir is None:
            parser.error("worker requires --request and --output-dir")
        from bayesfilter.inference.q20_master_stages import run_worker
        result=run_worker(json.loads(args.request.read_text()),args.output_dir)
    elif args.mode=="status":
        if args.output_dir is None:
            parser.error("status requires --output-dir")
        result=json.loads((args.output_dir/"campaign.json").read_text())
        result={key:result[key] for key in ("status","spent_seconds","diagnostic_spent_seconds","campaign_limit","diagnostic_limit","stages")}
    else:
        config=load_protocol(args.config) if args.config else validate_protocol(protocol_template())
        if args.cpu_reference:
            config.update(role="smoke",cpu_reference=True,jit_compile=False)
            validate_protocol(config)
        if args.mode=="write-config":
            if args.output_dir is None:
                parser.error("write-config requires --output-dir (fresh directory)")
            args.output_dir.mkdir(parents=True,exist_ok=False)
            (args.output_dir/"protocol.json").write_text(json.dumps(config,indent=2)+"\n")
            result={"config_path":str(args.output_dir/"protocol.json"),"config_hash":digest(config)}
        elif args.mode=="validate":
            result={"schema":config["schema"],"config_hash":digest(config),"role":config["role"],
                "promotion_eligible":False,"cohort_size":len(training_cohort(config)),
                "methods":config["comparison"]["methods"],
                "stages":["qualify","price","reference","train","tune","sample","replica_exchange","ensemble","compare","reverify","confirmation"],
                "runtime_limit":"external_process_supervisor","budget_required_before_numerical_work":True}
        else:
            if args.output_dir is None:
                parser.error("execution requires --output-dir")
            from bayesfilter.inference.q20_master_program import execute_master
            allowance=None if args.budget_record is None else json.loads(args.budget_record.read_text())
            result=execute_master(config,args.output_dir,repo=ROOT,allowance=allowance,
                                  stop_after=args.mode if args.mode in {"price","train"} else None)
    print(json.dumps(result,indent=2,allow_nan=False))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
