"""Bounded resource retry on unchanged source, data, method and random streams.

The supplied seconds are an additional resource allocation from the enclosing
campaign. All prior attempts remain charged. This does not change the design's
scientific settings or relabel a retry as an independent replication.
"""
from __future__ import annotations

import argparse
import fcntl
import math
from pathlib import Path
import sys
import time


def resume_job(root, job_id, *, seconds, reason):
    from bayesfilter.testing.inference_validation import execution
    from bayesfilter.testing.inference_validation.designs import ValidationDesign, digest
    from bayesfilter.testing.inference_validation.storage import read_json,write_json,file_hash

    if not math.isfinite(seconds) or seconds <= 0 or not reason.strip():
        raise ValueError("a positive finite allocation and repair reason are required")
    root=Path(root).resolve()
    with (root/".coordinator.lock").open("a+b") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        index=read_json(root/"run_index.json")
        if execution.source_state()["identity"] != index["source"]["identity"]:
            raise ValueError("resource retry requires unchanged frozen source")
        prior=index["jobs"][job_id]
        if prior["status"] not in {"failed","timed_out","unfunded"}:
            raise ValueError("resource retry requires an ended unsuccessful attempt")
        planned=next(j for j in index["plan"]["jobs"] if j["design"]["design_id"]==job_id)
        job_root=root/job_id
        design_file=job_root/"design.json"
        raw=read_json(design_file)
        if digest(raw)!=planned["identity"]:
            raise ValueError("resource retry cannot alter scientific design")
        design=ValidationDesign.from_payload(raw)
        attempt=len(prior["attempts"])+1
        log=job_root/f"attempt-{attempt:03d}.log"
        if log.exists():
            raise FileExistsError(log)
        command=[sys.executable,"-m","bayesfilter.testing.inference_validation","_worker",
                 str(design_file),str(job_root),str(seconds),str(attempt)]
        extension={"job_id":job_id,"attempt":attempt,"additional_seconds":seconds,
                   "reason":reason,"source_identity":index["source"]["identity"],
                   "design_identity":planned["identity"],"command":command}
        index.setdefault("resource_extensions",[]).append(extension)
        index["jobs"][job_id]={**prior,"status":"running","started_at":time.time(),
            "log":str(log),"reserved_seconds":seconds}
        write_json(root/"run_index.json",index)
        record=execution._execute_job(design,command,log,seconds,attempt)
        record["resource_extension"]=extension
        result_path=job_root/f"attempt-{attempt:03d}-result.json"
        index["jobs"][job_id]={"status":record["status"],"attempts":[*prior["attempts"],record],
                              "result":str(result_path) if record["exit_code"]==0 else None}
        if record["exit_code"]==0:
            index["jobs"][job_id]["result_sha256"]=file_hash(result_path)
        write_json(root/"run_index.json",index)
        from bayesfilter.testing.inference_validation.reporting import report
        from bayesfilter.testing.inference_validation.aggregation import aggregate_groups
        report(root)
        aggregate_groups(root)
        return record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root",type=Path)
    parser.add_argument("job_id")
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--seconds",type=float,required=True)
    parser.add_argument("--reason",required=True)
    args=parser.parse_args()
    sys.path.insert(0,str(args.source.resolve()))
    record=resume_job(args.root,args.job_id,seconds=args.seconds,reason=args.reason)
    print({key:record[key] for key in ("attempt","status","elapsed_seconds")})
    return 0 if record["status"]=="complete" else 1


if __name__=="__main__":
    raise SystemExit(main())
