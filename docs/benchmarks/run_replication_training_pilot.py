"""Cost-readiness supervisor for the repaired KDM replication training pilot."""
from __future__ import annotations
import datetime, hashlib, json, os, shutil, signal, subprocess, sys, time, traceback
from pathlib import Path

ROOT=Path("/home/chakwong/BayesFilter")
BASE=ROOT/"docs/plans/artifacts/controlled-smc-kdm-replication-20260928"
PHASE=BASE/"repaired-training-20261011"
PLAN="docs/plans/controlled-smc-kdm-repaired-training-pilot-20261011.md"
RESULT="docs/plans/controlled-smc-kdm-mechanism-result-20261011.md"
TF="/home/chakwong/anaconda3/envs/tftwogpu/bin/python"
GPU="GPU-d54fdcfc-c6ed-dbe7-25c7-93f737e0f93a"
COST_CAP=900.
PILOT_CAP=28800.

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text())
def write(path,value):
    path=Path(path);temp=path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n");temp.replace(path)

def main(out):
    from run_replication_gap_localization import campaign
    if out.parent!=PHASE or not out.name.startswith("training-pilot-"):
        raise ValueError("Unreviewed output path")
    prior=list(PHASE.glob("training-pilot-*/manifest.json"))
    if len(prior)>=2: raise RuntimeError("Two-attempt pilot cap exhausted")
    for path in prior:
        if read(path)["status"]=="RUNNING": raise RuntimeError("Pilot phase already running")
    spent=sum(read(p)["resource_wall_seconds"]["gpu"] for p in prior)
    if spent>=PILOT_CAP: raise RuntimeError("Pilot GPU cap exhausted")
    balance=campaign()
    remaining_campaign=min(balance["remaining_seconds"]["gpu"],balance["tranche_remaining_seconds"]["gpu"])
    if remaining_campaign<min(COST_CAP,PILOT_CAP-spent):
        raise RuntimeError("Insufficient existing campaign GPU allocation")
    stage=BASE/"stage-refresh-20261011/stage-refresh-01"
    stage_manifest=read(stage/"manifest.json")
    proof=stage/"verification/diagnostic/result.json"
    oldsmoke=stage/"smoke/result.json"
    assert stage_manifest["status"]=="COMPLETED" and read(proof)["status"]=="PASS"
    assert read(oldsmoke)["status"]=="PASS" and len(read(oldsmoke)["completed_stages"])==3
    assert sha(proof)==read(stage/"verification/manifest.json")["result_sha256"]
    for path,digest in stage_manifest["source_hashes"].items():
        if path.startswith("bayesfilter/replication/") or Path(path).name in (
            "diagnose_kdm_source_epoch_tf.py","diagnose_kdm_fresh_training_cost_tf.py"):
            assert sha(ROOT/path)==digest, "Runtime changed since proof: "+path
    reference=BASE/"kdm-fresh-training-cost-20261007-01/diagnostic/result.json"
    source=BASE/"kdm-complete-training-20261007-03/diagnostic/result.json"
    ref,trained=read(reference),read(source)
    assert ref["status"]=="PASS" and trained["status"]=="PASS"
    assert trained["source_training_protocol_complete"]
    assert ref["source_pin"]==trained["source_pin"]=="b0e2fd54db7b6c36d70e8e701ddc6a3f3d5dee18"
    assert ref["source_training_configs"]==trained["source_training_configs"]
    counts={x["stage"]:x["completed_epochs"] for x in trained["stages"]}
    assert list(counts)==["weights","mdpf","am_dpf"] and counts=={"weights":107,"mdpf":454,"am_dpf":230}
    inputs=[proof,stage/"verification/manifest.json",stage/"manifest.json",oldsmoke,reference,source]
    for section in ("datasets","fixtures"):
        assert set(ref[section])==set(trained[section])
        for name,info in ref[section].items():
            assert info["sha256"]==trained[section][name]["sha256"]
            for item in (info,trained[section][name]):
                assert sha(item["path"])==item["sha256"];inputs.append(Path(item["path"]))
    paths=[ROOT/PLAN,ROOT/"docs/benchmarks/run_replication_gap_localization.py",
        ROOT/"docs/benchmarks/run_replication_training_pilot.py",
        ROOT/"docs/benchmarks/diagnose_kdm_fresh_training_cost_tf.py",
        ROOT/"docs/benchmarks/diagnose_kdm_source_epoch_tf.py",
        ROOT/"bayesfilter/runtime/gpu_memory_policy.py"]
    paths+=list((ROOT/"bayesfilter/replication").glob("*.py"))
    out.mkdir(parents=True,exist_ok=False)
    for path in paths:
        dest=out/"sources"/path.relative_to(ROOT)
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPU,TF_FORCE_GPU_ALLOW_GROWTH="true",
        TF_CPP_MIN_LOG_LEVEL="2",OMP_NUM_THREADS="1",TF_NUM_INTEROP_THREADS="1",
        TF_NUM_INTRAOP_THREADS="1")
    manifest=dict(schema="replication_training_cost_readiness_v1",status="RUNNING",
        command=[sys.executable,*sys.argv],cwd=str(ROOT),plan_file=PLAN,result_file=RESULT,
        git_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        supervisor_pid=os.getpid(),resource_wall_seconds=dict(cpu=0.,gpu=0.),
        source_hashes={str(p.relative_to(ROOT)):sha(p) for p in paths},
        input_hashes={str(p.relative_to(ROOT)):sha(p) for p in inputs},
        source_training_completed_epochs=counts,
        environment={k:env[k] for k in ("CUDA_VISIBLE_DEVICES","TF_FORCE_GPU_ALLOW_GROWTH",
            "TF_NUM_INTEROP_THREADS","TF_NUM_INTRAOP_THREADS","OMP_NUM_THREADS")},
        tensorflow_interpreter=TF,data_version="pinned source constructor train5000/validation1000",
        random_seeds=[202610081,202610082],gpu_trust_basis="escalated_local_gpu_access",
        replication="INCOMPLETE",steps=[])
    def save():write(out/"manifest.json",manifest)
    def run(command,label,progress=None):
        remaining=min(COST_CAP-manifest["resource_wall_seconds"]["gpu"],PILOT_CAP-spent-manifest["resource_wall_seconds"]["gpu"])
        if remaining<=0:raise RuntimeError("Readiness budget exhausted")
        step=dict(command=command,label=label,log=label+".log",status="RUNNING")
        manifest["steps"].append(step);save()
        start=time.monotonic();child=None
        try:
            with (out/step["log"]).open("w") as log:
                child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                step["pid"]=child.pid;save()
                while True:
                    try:
                        code=child.wait(timeout=max(.1,min(30,remaining-(time.monotonic()-start))))
                        break
                    except subprocess.TimeoutExpired:
                        elapsed=time.monotonic()-start
                        age=time.time()-progress.stat().st_mtime if progress is not None and progress.exists() else elapsed
                        step.update(elapsed_seconds=elapsed,progress_age_seconds=age);save()
                        if elapsed>=remaining or (progress is not None and age>1800):
                            raise TimeoutError("READINESS_TIME_LIMIT" if elapsed>=remaining else "HEARTBEAT_STALL")
            step.update(status="COMPLETED" if code==0 else "FAILED",exit_code=code)
            if code:raise RuntimeError(label+" failed; inspect "+step["log"])
        except BaseException:
            step["status"]="FAILED"
            if child is not None and child.poll() is None:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
            raise
        finally:
            step["wall_seconds"]=time.monotonic()-start
            manifest["resource_wall_seconds"]["gpu"]+=step["wall_seconds"];save()
    save()
    try:
        run(["nvidia-smi","--query-gpu=uuid,name,memory.used,memory.total","--format=csv"],"nvidia")
        script=ROOT/"docs/benchmarks/diagnose_kdm_fresh_training_cost_tf.py"
        run([TF,"-I","-B",str(script),"--epochs","smoke","--output",str(out/"diagnostic"),
            "--reference",str(reference),"--verification",str(proof)],
            "thread-parity-smoke",out/"diagnostic/progress.json")
        smoke=read(out/"diagnostic/result.json")
        assert smoke["status"]=="PASS" and smoke["checks_failed"]==0
        assert [s["stage"] for s in smoke["completed_stages"]]==["weights","mdpf","am_dpf"]
        epochs=[json.loads(line) for line in (out/"diagnostic/epochs.jsonl").read_text().splitlines()]
        seconds={x["stage"]:x["wall_seconds"] for x in epochs if x["epoch"]==1}
        assert set(seconds)==set(counts)
        projection=sum(seconds[name]*count for name,count in counts.items())
        remaining=PILOT_CAP-spent-manifest["resource_wall_seconds"]["gpu"]
        ready=projection<=.8*remaining
        decision=dict(status="READY_FOR_FULL" if ready else "COST_NOT_READY",
            scope="EXECUTION_COST_ONLY",replication="INCOMPLETE",
            second_epoch_seconds=seconds,source_completed_epochs=counts,
            projected_seconds=projection,remaining_pilot_gpu_seconds=remaining,
            threshold_seconds=.8*remaining,training_complete=False)
        for group in ("source_hashes","input_hashes"):
            for name,digest in manifest[group].items():
                assert sha(ROOT/name)==digest,"Input/source changed during readiness: "+name
        manifest["status"]="COMPLETED"
        manifest["smoke_result_sha256"]=sha(out/"diagnostic/result.json")
    except BaseException:
        manifest["status"]="FAILED";(out/"error.txt").write_text(traceback.format_exc())
        decision=dict(status="ERROR",scope="READINESS_HARNESS_OR_RUNTIME",replication="INCOMPLETE")
    finally:
        write(out/"decision.json",decision);manifest["decision_sha256"]=sha(out/"decision.json");save()
        ledger_path=BASE/"reference-diagnostics.json";ledger=read(ledger_path)
        record=dict(id=out.name+"_gpu",resource="gpu",charged_seconds=manifest["resource_wall_seconds"]["gpu"],
            summary=str((out/"decision.json").relative_to(ROOT)),summary_sha256=sha(out/"decision.json"),
            manifest=str((out/"manifest.json").relative_to(ROOT)),plan_file=PLAN,result_file=RESULT,
            stage="REPAIRED_TRAINING_COST_READINESS",next_action="Full training only on READY_FOR_FULL",
            after_attempt=balance["attempts"],executions=[dict(path=str(out.relative_to(ROOT)),resource="gpu",
                wall_seconds=manifest["resource_wall_seconds"]["gpu"])])
        assert not any(x["id"]==record["id"] for x in ledger)
        ledger.append(record);write(ledger_path,ledger)
    print(json.dumps(dict(status=manifest["status"],decision=decision,
        resource_wall_seconds=manifest["resource_wall_seconds"],output=str(out))))
    return 0 if manifest["status"]=="COMPLETED" else 1
