"""Bounded local queue for explicit q20 calibration/training requests.

Run with trusted GPU permissions. This supervisor never imports a framework;
workers enforce memory growth, target identity, deadline and numerical checks.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def save(path, value):
    temporary=path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")
    temporary.replace(path)


def available_devices(allowed):
    devices={}
    output=subprocess.check_output(["nvidia-smi","--query-gpu=index,uuid","--format=csv,noheader,nounits"],text=True)
    for line in output.splitlines():
        index,uuid=[s.strip() for s in line.split(",")]
        devices[uuid]=int(index)
    blocked=set()
    output=subprocess.check_output(["nvidia-smi","--query-compute-apps=gpu_uuid,pid,process_name","--format=csv,noheader,nounits"],text=True)
    for line in output.splitlines():
        uuid,pid,name=[s.strip() for s in line.split(",",2)]
        if name.endswith(("nxnode.bin","gnome-remote-desktop-daemon")):
            continue
        if uuid in devices:
            blocked.add(devices[uuid])
    return [d for d in allowed if d not in blocked]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    spec=json.loads(args.queue.read_text())
    args.output.mkdir(parents=True,exist_ok=False)
    if sum(j["worker_seconds"] for j in spec["jobs"])>spec["allocation_seconds"]:
        raise ValueError("queue reservations exceed its allocation")
    deadline=datetime.fromisoformat(spec["deadline_utc"])
    state={"spec":spec,"status":"running","jobs":[],"spent_worker_seconds":0.,
        "started_at":datetime.now(timezone.utc).isoformat(),"command":sys.argv}
    active={}
    pending=list(spec["jobs"])
    stopped=False
    def terminate(signum,frame):
        nonlocal stopped
        stopped=True
    signal.signal(signal.SIGTERM,terminate)
    signal.signal(signal.SIGINT,terminate)
    save(args.output/"queue.json",state)
    while pending or active:
        if stopped or datetime.now(timezone.utc)>=deadline:
            state["status"]="deadline_or_interrupt"
            pending.clear()
            for entry in active.values():
                entry["process"].terminate()
        for name,entry in list(active.items()):
            proc=entry["process"]
            row=entry["row"]
            elapsed=time.monotonic()-entry["began"]
            if elapsed>row["worker_seconds"] and proc.poll() is None:
                proc.terminate()
                row["timed_out"]=True
            if elapsed>row["worker_seconds"]+10. and proc.poll() is None:
                proc.kill()
            code=proc.poll()
            if code is None:
                continue
            entry["log"].close()
            row.update(exit_code=code,supervised_worker_seconds=elapsed,status="completed" if code==0 else "failed")
            state["spent_worker_seconds"]+=elapsed
            del active[name]
            if code!=0 and not row.get("timed_out"):
                failure_path=Path(row["output"])/"failure.json"
                failure=json.loads(failure_path.read_text()) if failure_path.exists() else {}
                candidate_messages=("invalid training candidate", "gradient guard clips a majority",
                    "post-training verification is invalid", "final exported-map inverse failed")
                if failure.get("type")=="ValueError" and failure.get("message","").startswith(candidate_messages):
                    row["status"]="candidate_rejected"
            print(json.dumps({"job":name,**{k:row[k] for k in ("status","supervised_worker_seconds","exit_code")}}),flush=True)
            if code!=0 and row["status"]!="candidate_rejected":
                state["status"]="repair_required"
                pending.clear()
        if state["status"]=="running" and pending:
            free=available_devices(spec["gpus"])
            used={e["row"]["gpu"] for e in active.values()}
            for gpu in (d for d in free if d not in used):
                if not pending:
                    break
                job=pending.pop(0)
                request=json.loads(Path(job["request"]).read_text())
                if request["worker_seconds"]!=job["worker_seconds"] or request["deadline_utc"]!=spec["deadline_utc"]:
                    raise ValueError("request/queue budget or deadline mismatch")
                output=args.output/job["name"]
                log_path=args.output/(job["name"]+".log")
                command=[sys.executable,str(Path(__file__).resolve().parents[1]/"docs/benchmarks/run_q20_configured_training_2026_09_24.py"),
                    "--request",job["request"],"--output",str(output),"--gpu",str(gpu)]
                log=log_path.open("x")
                env={**os.environ,"TF_FORCE_GPU_ALLOW_GROWTH":"true","CUDA_VISIBLE_DEVICES":str(gpu)}
                proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env)
                row={**job,"gpu":gpu,"pid":proc.pid,"output":str(output),"log":str(log_path),"status":"running","command":command}
                state["jobs"].append(row)
                active[job["name"]]={"process":proc,"log":log,"row":row,"began":time.monotonic()}
                print(json.dumps({"launched":job["name"],"gpu":gpu,"pid":proc.pid}),flush=True)
        state["pending"]=[j["name"] for j in pending]
        state["active_worker_seconds"]={name:time.monotonic()-e["began"] for name,e in active.items()}
        save(args.output/"queue.json",state)
        if pending or active:
            time.sleep(5.)
    if state["status"]=="running":
        state["status"]=("complete_with_rejected_candidates" if any(
            r["status"]=="candidate_rejected" for r in state["jobs"]) else "complete")
    state["finished_at"]=datetime.now(timezone.utc).isoformat()
    save(args.output/"queue.json",state)
    if state["status"] not in ("complete","complete_with_rejected_candidates"):
        raise SystemExit(1)


if __name__=="__main__":
    main()
