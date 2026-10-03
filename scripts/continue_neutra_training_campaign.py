"""Continue the explicit q20 training phases and preserve phase decisions.

Waits for the already running initial queue, runs the funded continuation and
control, then evaluates untouched banks and produces the paired result. This
controller cannot grant posterior authority or extend the campaign budget.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def save(path,value):
    path=Path(path)
    temporary=path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")
    temporary.replace(path)


def save_request(path, value):
    """Preserve the exact inputs when reusing a completed phase."""
    if path.exists():
        if read(path) != value:
            raise ValueError("existing phase request differs: " + str(path))
    else:
        save(path, value)


def continuation_requests(root, initial, selected):
    """Complete the seeded protocol without discarding failed candidates."""
    expected = {f"{family}-seed{s}-u4096" for family in ("iaf16", "naf16") for s in range(3)}
    by_name = {row["name"]: row for row in initial["jobs"]}
    if len(by_name) != len(initial["jobs"]) or set(by_name) != expected:
        raise ValueError("initial cohort does not account for all six seeded fits")
    repair_path = root/"naf-log-weight-repair.json"
    repaired = read(repair_path) if repair_path.exists() else None
    if repaired:
        replay = read(repaired["verification_result"])
        if {row["seed"] for row in replay["rows"]} != {0, 1} or not all(
                row["exact_replay"]["valid"] for row in replay["rows"]):
            raise ValueError("log-weight repair lacks the exact failed-update verification")
        for row in by_name.values():
            if row["status"] == "failed":
                failure = read(Path(row["output"])/"failure.json")
                if (row["name"] != "naf16-seed2-u4096" or failure["type"] != "TypeError"
                        or "tf__inverse() takes 2 positional arguments but 3 were given" not in failure["message"]):
                    raise ValueError("initial cohort has an unexamined infrastructure failure")
    elif any(row["status"] not in ("completed", "candidate_rejected") for row in by_name.values()):
        raise ValueError("initial cohort has an unresolved infrastructure failure")
    naf_ok = all(by_name[f"naf16-seed{s}-u4096"]["status"] == "completed" for s in range(3))
    family = "naf16" if naf_ok or repaired else "naf32"
    decision = {"family": family, "reason":
        "complete the funded 8192-update NAF rung; 4096 is intermediate" if naf_ok else
        "NAF16 candidate rejection triggers the predeclared NAF32 capacity repair",
        "candidate_failure_rejects_research_direction": False}
    if repaired:
        decision.update(reason="resume the same three NAF16 seeds after the measured log-domain guard repair",
            repair_evidence=str(repair_path), initial_failed_attempts_preserved=True)
    requests = []
    for seed_index in range(3):
        template = read(root/f"training-request-naf16-seed{seed_index}-u4096-01.json")
        request = {**template, "family": family, "updates": 8192 if naf_ok or repaired else 4096,
            "validation_rungs": [6144, 8192] if naf_ok else [1024, 2048, 4096, 6144, 8192] if repaired else [1024, 2048, 4096],
            "learning_rate": selected[family]["learning_rate"],
            "gradient_clip_norm": selected[family]["gradient_clip_norm"],
            "calibration_receipt": str(root/"calibration-queue-01"/f"calibration-{family}-01"/"result.json"),
            "worker_seconds": 11500.}
        request.pop("resume_checkpoint", None)
        if repaired:
            output = Path(by_name[f"naf16-seed{seed_index}-u4096"]["output"])
            checkpoints = list(output.glob("checkpoint-*.json"))
            if (output/"last-valid-checkpoint.json").exists():
                checkpoints.append(output/"last-valid-checkpoint.json")
            if not checkpoints:
                raise ValueError("missing optimizer state for repaired NAF seed")
            path = max(checkpoints, key=lambda p: read(p)["optimizer"][0]["value"])
            accepted = read(path)["optimizer"][0]["value"]
            request.update(resume_checkpoint=str(path), worker_seconds=float(math.ceil((8192-accepted)*3.+400.)),
                repair_evidence=str(repair_path))
        elif naf_ok:
            request["resume_checkpoint"] = str(Path(by_name[f"naf16-seed{seed_index}-u4096"]["output"])/"checkpoint-004096.json")
        requests.append((f"{family}-seed{seed_index}-u{request['updates']}", request))
    template = read(root/"training-request-iaf16-seed0-u4096-01.json")
    control = {**template, "family": "legacy_control", "root_seed": 0, "updates": 4096,
        "learning_rate": selected["legacy_control"]["learning_rate"],
        "gradient_clip_norm": selected["legacy_control"]["gradient_clip_norm"],
        "calibration_receipt": str(root/"calibration-queue-01/calibration-legacy_control-01/result.json"),
        "worker_seconds": 11500.}
    control.pop("resume_checkpoint", None)
    requests.insert(0, ("legacy-control-seed0-u4096", control))
    retained = [row for row in initial["jobs"] if row["status"] == "completed"
        and (row["name"].startswith("iaf16") or (not naf_ok and not repaired))]
    return decision, requests, retained


def check_allocation(root, phases, allocation, reserve):
    accounting = read(root/"accounting-after-calibration.json")
    extra_path = root/"supplemental-charges.json"
    extra = read(extra_path) if extra_path.exists() else {"charges": []}
    spent = sum(p.get("spent_worker_seconds", 0.) for p in phases)
    spent += sum(c["worker_seconds"] for c in extra["charges"])
    remaining = accounting["remaining_before_sustained_training_seconds"] - spent
    if allocation + reserve > remaining:
        raise ValueError(f"phase allocation exceeds remaining campaign budget: {remaining}")
    return remaining


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    args=parser.parse_args()
    root=args.root.resolve()
    interrupted = False
    def interrupt(signum, frame):
        nonlocal interrupted
        interrupted = True
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    state={"status":"waiting_for_initial_fits","command":sys.argv,"started_at":datetime.now(timezone.utc).isoformat(),
        "plan":"docs/plans/bayesfilter-neutra-precision-training-plan-2026-09-24.md","phases":[]}
    state_path=root/"master-progress.json"
    save(state_path,state)
    base=read(root/"pricing-request-01.json")
    common={k:base[k] for k in ("baseline_checkpoint","baseline_sha256","deadline_utc","plan_file")}
    deadline=datetime.fromisoformat(base["deadline_utc"])
    def check_deadline():
        if interrupted:
            raise InterruptedError("campaign controller interrupted")
        if datetime.now(timezone.utc)>=deadline:
            raise TimeoutError("campaign deadline reached")
    def queue(name,jobs,allocation,reserve):
        check_deadline()
        state["remaining_before_phase_seconds"] = check_allocation(root,state["phases"],allocation,reserve)
        request={"jobs":jobs,"allocation_seconds":allocation,"gpus":[1,2,0],
            "deadline_utc":base["deadline_utc"],"plan_file":base["plan_file"]}
        request_path=root/(name+"-request.json")
        save_request(request_path,request)
        command=[sys.executable,str(ROOT/"scripts/run_neutra_training_queue.py"),
            "--queue",str(request_path),"--output",str(root/name)]
        existing=root/name/"queue.json"
        proc=None
        log=None
        if not existing.exists():
            log=(root/(name+".log")).open("x")
            proc=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                env={**os.environ,"TF_FORCE_GPU_ALLOW_GROWTH":"true"})
        phase={"name":name,"command":command,"pid":proc.pid if proc else None,"status":"running"}
        state["phases"].append(phase);save(state_path,state)
        try:
            while (proc is not None and proc.poll() is None) or (
                    proc is None and read(existing)["status"]=="running"):
                check_deadline()
                time.sleep(10.)
        except BaseException:
            if proc is not None and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=20.)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
            raise
        finally:
            if log is not None:
                log.close()
        result=read(root/name/"queue.json")
        phase.update(status=result["status"],spent_worker_seconds=result["spent_worker_seconds"])
        save(state_path,state)
        if result["status"] not in ("complete","complete_with_rejected_candidates") or (
                proc is not None and proc.returncode!=0):
            raise RuntimeError("localized queue repair required: "+name)
        return result
    def job(name,request):
        path=root/(name+"-request.json");save_request(path,request)
        return {"name":name,"request":str(path),"worker_seconds":request["worker_seconds"]}
    try:
        while True:
            check_deadline()
            initial=read(root/"training-queue-01/queue.json")
            if initial["status"]!="running":
                break
            time.sleep(10.)
        if initial["status"] not in ("complete","complete_with_rejected_candidates") and not (
                initial["status"]=="repair_required" and (root/"naf-log-weight-repair.json").exists()):
            raise RuntimeError("initial training queue needs an infrastructure repair")
        state["phases"].append({"name":"training-queue-01","status":initial["status"],
            "spent_worker_seconds":initial["spent_worker_seconds"]})
        selected=read(root/"calibration-selection.json")
        decision,requests,candidates=continuation_requests(root,initial,selected)
        state["continuation_decision"]=decision
        jobs=[job(name,request) for name,request in requests]
        state["status"]="running_continuation_and_control";save(state_path,state)
        continuation=queue("continuation-queue-01",jobs,sum(j["worker_seconds"] for j in jobs),16000.)
        candidates += [r for r in continuation["jobs"] if r["status"]=="completed"]
        if not candidates:
            raise RuntimeError("all planned candidate maps were rejected; no final training claim")
        state["status"]="untouched_final_evaluation";save(state_path,state)
        jobs=[job("baseline-final",{**common,"stage":"final_evaluation","worker_seconds":300.,
            "rows":2048,"evaluation_batch_size":32})]
        records=[]
        for row in candidates:
            result=read(Path(row["output"])/"result.json")
            name=row["name"]+"-final"
            jobs.append(job(name,{**common,"stage":"final_evaluation","worker_seconds":300.,
                "rows":2048,"evaluation_batch_size":32,"checkpoint":result["checkpoint"]}))
            records.append({"family":result["family"],"root_seed":result["seed"],
                "updates":result["updates"],"checkpoint":result["checkpoint"],"finalized":result["finalized"],
                "measurement":str(root/"final-evaluation-queue-01"/name/"final-bank.json")})
        if len(jobs)>10:
            raise ValueError("final evaluation exceeds its 3000-second reservation")
        final=queue("final-evaluation-queue-01",jobs,3000.,13000.)
        request={"baseline":str(root/"final-evaluation-queue-01/baseline-final/final-bank.json"),
            "candidates":records,"plan_file":base["plan_file"],"initial_queue":str(root/"training-queue-01/queue.json"),
            "continuation_queue":str(root/"continuation-queue-01/queue.json")}
        save_request(root/"paired-final-request.json",request)
        state["status"]="paired_final_analysis";save(state_path,state)
        began=time.monotonic()
        report_path=root/"paired-final-analysis/result.json"
        if not report_path.exists():
            check_allocation(root,state["phases"],300.,13000.)
            subprocess.run([sys.executable,str(ROOT/"docs/benchmarks/summarize_q20_configured_training_2026_09_24.py"),
                "--request",str(root/"paired-final-request.json"),"--output",str(root/"paired-final-analysis")],
                cwd=ROOT,env={**os.environ,"CUDA_VISIBLE_DEVICES":"-1"},check=True,timeout=300.)
        elif read(report_path)["request"] != request:
            raise ValueError("completed report has a different input request")
        state["analysis_worker_seconds"]=time.monotonic()-began
        state["phases"].append({"name":"paired_final_analysis","status":"complete",
            "spent_worker_seconds":read(report_path)["wall_seconds"]})
        state["status"]="running_downstream_assessment"
        state["result"]=str(root/"paired-final-analysis/result.json")
        save(state_path,state)
        manifest=read(Path(initial["jobs"][0]["output"])/"manifest.json")
        downstream_request={"stage":"hmc_assessment","worker_seconds":8000.,
            "deadline_utc":base["deadline_utc"],"plan_file":base["plan_file"],
            "training_report":str(report_path),
            "training_report_sha256":hashlib.sha256(report_path.read_bytes()).hexdigest(),
            "target_signature":manifest["target_signature"],"bridge_signature":manifest["bridge_signature"]}
        downstream=queue("downstream-queue-01",[job("hmc-assessment",downstream_request)],8000.,5000.)
        state["downstream_result"]=str(root/"downstream-queue-01/hmc-assessment/result.json")
        state["status"]="training_and_bounded_downstream_evaluation_complete"
        state["finished_at"]=datetime.now(timezone.utc).isoformat()
        save(state_path,state)
    except BaseException as error:
        state.update(status="repair_required",failure={"type":type(error).__name__,"message":str(error)},
            stopped_at=datetime.now(timezone.utc).isoformat())
        save(state_path,state)
        raise


if __name__=="__main__":
    main()
