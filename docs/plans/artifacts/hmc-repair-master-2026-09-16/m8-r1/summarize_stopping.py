"""Post-run diagnostic aggregation of independent M8 fits; no sampler execution."""
import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parents[4]))
from bayesfilter.testing.inference_validation.designs import ValidationDesign
from bayesfilter.testing.inference_validation.engines.pipeline import summarize_replications
from bayesfilter.testing.inference_validation.storage import write_json


def read(path):
    return json.loads(path.read_text())


def summarize(index_path):
    index = read(index_path)
    groups = {}
    hashes = {str(index_path):hashlib.sha256(index_path.read_bytes()).hexdigest()}
    for job in index["plan"]["jobs"]:
        design = ValidationDesign.from_payload(job["design"])
        groups.setdefault(design.scenario.target,[]).append(design)
    summaries = {}
    for target,designs in groups.items():
        records,seeds,rows = [],set(),[]
        first = designs[0].payload()
        for design in designs:
            payload = design.payload()
            for key in ("design_id","seed"):
                payload.pop(key)
            comparator = {k:v for k,v in first.items() if k not in ("design_id","seed")}
            if payload != comparator or design.seed in seeds or design.replications != 1:
                raise ValueError("incompatible or duplicated complete-fit units")
            seeds.add(design.seed)
            job = index["jobs"].get(design.design_id,{})
            row = {"design_id":design.design_id,"seed":design.seed,"status":job.get("status","unstarted")}
            if job.get("result"):
                path = Path(job["result"])
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                if digest != job["result_sha256"]:
                    raise ValueError("changed result " + str(path))
                hashes[str(path)] = digest
                result = read(path)
                if result["design_identity"] != design.identity:
                    raise ValueError("design mismatch")
                assessment = result["assessment"]
                for record in assessment["replications"]:
                    if record["inventory"]["failures"]:
                        raise ValueError("invalid tuning inventory")
                    record = dict(record,replication=len(records),design_id=design.design_id)
                    records.append(record)
                row["finding"] = assessment["finding"]
                row["verified_members"] = assessment["verified_members"]
            rows.append(row)
        aggregate = summarize_replications(replace(designs[0],replications=len(designs)),records)
        members = [m for r in records for m in r["members"] if "assessment" in m]
        summaries[target] = {"aggregation":aggregate,
            "runtime_passed":sum(bool(m["runtime_checks_passed"]) for m in members),
            "warmup_counts":dict(Counter(m["warmup_count"] for m in members)),
            "retained_counts":dict(Counter(m["retained_count"] for m in members)),
            "fits":rows,"source_identity":index["source"]["identity"],
            "fixed_dataset":designs[0].options["data"],
            "interpretation":"conditional repeated fits of one dataset, not prior-predictive SBC; pointwise uncertainty only"}
    return {"summaries":summaries,"inputs":hashes,"command":sys.argv,
            "plan_file":"docs/plans/bayesfilter-hmc-repair-master-program-2026-09-16.md"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("index",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = summarize(args.index)
    write_json(args.output,result)
    for target,row in result["summaries"].items():
        aggregate = row["aggregation"]
        print(target,"planned",aggregate["planned"],"runtime_passed",row["runtime_passed"])
        print(json.dumps(aggregate["stopped_versus_fixed"],indent=2))


if __name__ == "__main__":
    main()
