"""Bounded supervisor: source stage audit, exact control, isolated training, native evaluation."""
import datetime, hashlib, json, os, shutil, signal, subprocess, sys, time, traceback
from pathlib import Path
ROOT=Path("/home/chakwong/BayesFilter")
BASE=ROOT/"docs/plans/artifacts/controlled-smc-kdm-replication-20260928"
PHASE=BASE/"stage-isolation-20261013"
PLAN="docs/plans/controlled-smc-kdm-stage-isolation-plan-20261013.md"
RESULT="docs/plans/controlled-smc-kdm-stage-isolation-result-20261013.md"
TFPY="/home/chakwong/anaconda3/envs/tftwogpu/bin/python"
SOURCEPY="/home/chakwong/anaconda3/envs/tf-gpu2/bin/python"
STAGE_REFERENCE=BASE/"kdm-source-stages-20261007-04/diagnostic/result.json"
GPU="GPU-d54fdcfc-c6ed-dbe7-25c7-93f737e0f93a"
CAPS=dict(cpu=16200.,gpu=14400.)
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):
    tmp=p.with_suffix(p.suffix+".tmp");tmp.write_text(json.dumps(x,indent=2,allow_nan=False)+"\n");tmp.replace(p)
def main():
    sys.path.insert(0,str(ROOT/"docs/benchmarks"))
    from run_replication_gap_localization import campaign
    balance=campaign();assert not balance["running"]
    prior=list(PHASE.glob("attempt*/manifest.json"))
    assert len(prior)<3 and all(read(p)["status"]!="RUNNING" for p in prior)
    remaining={r:min(CAPS[r]-sum(read(p)["resource_wall_seconds"][r] for p in prior),
        balance["remaining_seconds"][r],balance["tranche_remaining_seconds"][r]) for r in CAPS}
    assert min(remaining.values())>300
    out=PHASE/("attempt%02d"%(len(prior)+1));out.mkdir(parents=True,exist_ok=False)
    files=[Path(__file__).resolve(),ROOT/PLAN,
        *[ROOT/"docs/benchmarks"/name for name in ("reference_kdm_stage_isolation.py","diagnose_kdm_stage_isolation_tf.py","diagnose_kdm_source_stages_tf.py",
        "diagnose_kdm_source_epoch_tf.py","reference_kdm_source_stages.py","reference_kdm_neural.py","reference_kdm_neural_exact_replay.py")],
        *[ROOT/"bayesfilter/replication"/name for name in ("kdm_source_stages_tf.py","kdm_neural_replay_tf.py","kdm_sampling_tf.py",
        "kdm_normal_von_mises_tf.py","kdm_training_controller.py","kdm_stage_transfer_tf.py")],
        ROOT/"bayesfilter/runtime/gpu_memory_policy.py"]
    files=sorted(set(files+list((ROOT/"bayesfilter/replication").glob("kdm*.py"))))
    for p in files:
        dest=out/"sources"/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
    manifest=dict(status="RUNNING",supervisor_pid=os.getpid(),plan_file=PLAN,result_file=RESULT,
        command=[sys.executable,"-I","-B",str(Path(__file__).resolve())],
        start_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        git_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        git_dirty=subprocess.check_output(["git","status","--short","--untracked-files=no"],cwd=ROOT,text=True),
        source_hashes={str(p.relative_to(ROOT)):sha(p) for p in files},
        resource_wall_seconds=dict(cpu=0.,gpu=0.),campaign_remaining_before=balance["remaining_seconds"],
        phase_remaining_before=remaining,attempt_cap=3,phase_caps=CAPS,stages=[],replication="INCOMPLETE",
        data_version="unit01/02 fixed source datasets and repaired saved checkpoints; input SHA256 in each stage",
        seed_policy="original TF base1100000000; CPU draw seeds in stage artifacts",gpu_trust_basis="escalated_local_gpu_access")
    write(out/"manifest.json",manifest)
    proc=None;active_start=None;active_gpu=False
    decision=dict(status="FAILED",replication="INCOMPLETE")
    def execute(name,command,gpu,cap):
        nonlocal proc,active_start,active_gpu
        directory=out/name;directory.mkdir()
        env=dict(os.environ,OMP_NUM_THREADS="1")
        if gpu:
            env.update(CUDA_VISIBLE_DEVICES=GPU,TF_FORCE_GPU_ALLOW_GROWTH="true",TF_NUM_INTEROP_THREADS="1",TF_NUM_INTRAOP_THREADS="1",
                BAYESFILTER_PRELOAD_CUSTOM_OP="0",LD_LIBRARY_PATH="/home/chakwong/anaconda3/envs/tftwogpu/lib")
        else:env.update(CUDA_VISIBLE_DEVICES="-1",LD_LIBRARY_PATH="/home/chakwong/anaconda3/envs/tf-gpu2/lib")
        available=remaining["cpu"]-manifest["resource_wall_seconds"]["cpu"]
        if gpu:available=min(available,remaining["gpu"]-manifest["resource_wall_seconds"]["gpu"])
        timeout=min(cap,available)
        if timeout<30:raise RuntimeError("Phase budget exhausted before "+name)
        stage=dict(name=name,status="RUNNING",command=command,gpu=gpu,cpu_only=not gpu,
            gpu_intentionally_hidden=not gpu,timeout_seconds=timeout,interpreter=command[0],
            environment={k:env.get(k) for k in ("CUDA_VISIBLE_DEVICES","TF_FORCE_GPU_ALLOW_GROWTH",
                "TF_NUM_INTEROP_THREADS","TF_NUM_INTRAOP_THREADS","OMP_NUM_THREADS","LD_LIBRARY_PATH")})
        manifest["stages"].append(stage);manifest["current_stage"]=name;write(out/"manifest.json",manifest)
        active_start=time.monotonic();active_gpu=gpu
        try:
            with (directory/"execution.log").open("w") as log:
                proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                stage["pid"]=proc.pid;write(out/"manifest.json",manifest)
                while proc.poll() is None:
                    elapsed=time.monotonic()-active_start
                    progressfile=directory/"diagnostic/progress.json"
                    progress=read(progressfile) if progressfile.exists() else {}
                    write(out/"progress.json",dict(stage=name,pid=proc.pid,wall_seconds=elapsed,worker=progress,
                        cap_seconds=timeout,terminal=False))
                    if elapsed>timeout:raise TimeoutError(name+" exceeded bounded wall time")
                    time.sleep(10)
                stage["exit_code"]=proc.returncode
                if proc.returncode:raise RuntimeError(name+" exited "+str(proc.returncode))
            result=read(directory/"diagnostic/result.json")
            if result["status"]!="PASS":raise RuntimeError(name+" failed diagnostic")
            stage.update(status="COMPLETED",result_sha256=sha(directory/"diagnostic/result.json"))
            return result
        except BaseException:
            stage["status"]="FAILED";raise
        finally:
            if proc is not None and proc.poll() is None:
                os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
            elapsed=time.monotonic()-active_start
            stage["wall_seconds"]=elapsed;manifest["resource_wall_seconds"]["cpu"]+=elapsed
            if gpu:manifest["resource_wall_seconds"]["gpu"]+=elapsed
            write(out/"manifest.json",manifest);proc=None;active_start=None
    try:
        with (out/"nvidia-smi.log").open("w") as log:
            subprocess.run(["nvidia-smi","--query-gpu=name,uuid,memory.used,memory.total","--format=csv"],
                stdout=log,stderr=subprocess.STDOUT,check=True,timeout=30)
        audit=execute("source-audit",[SOURCEPY,"-I","-B",str(ROOT/"docs/benchmarks/reference_kdm_stage_isolation.py"),
            "--mode","audit","--output",str(out/"source-audit/diagnostic")],False,1800)
        verification=execute("verification",[TFPY,"-I","-B",str(ROOT/"docs/benchmarks/diagnose_kdm_source_stages_tf.py"),
            "--mode","verify","--reference",str(STAGE_REFERENCE),"--output",str(out/"verification/diagnostic")],True,900)
        proof=out/"verification/diagnostic/result.json"
        write(out/"verification/manifest.json",dict(status="COMPLETED",hashes=manifest["source_hashes"],
            reference=str(STAGE_REFERENCE),reference_sha256=sha(STAGE_REFERENCE),result_sha256=sha(proof)))
        control=execute("control",[TFPY,"-I","-B",str(ROOT/"docs/benchmarks/diagnose_kdm_stage_isolation_tf.py"),
            "--mode","control","--verification",str(proof),"--output",str(out/"control/diagnostic")],True,300)
        trained=execute("training",[TFPY,"-I","-B",str(ROOT/"docs/benchmarks/diagnose_kdm_stage_isolation_tf.py"),
            "--mode","train","--verification",str(proof),"--author-start",str(out/"source-audit/diagnostic/unit01-author-start.json"),
            "--output",str(out/"training/diagnostic")],True,14400)
        evaluation=execute("evaluation",[SOURCEPY,"-I","-B",str(ROOT/"docs/benchmarks/reference_kdm_stage_isolation.py"),
            "--mode","evaluate","--trained",str(out/"training/diagnostic"),"--output",str(out/"evaluation/diagnostic")],False,1800)
        decision=dict(status="PASS",replication="INCOMPLETE",scope="single controlled pretraining-state intervention",
            pretraining_statistics=audit["statistics"],verification_checks=verification["checks_total"],control_max_loss_difference=control["control_max_batch_loss_difference"],
            training_stages=trained["completed_stages"],native_statistics=evaluation["statistics"],
            inference="descriptive checkpoint-conditional; no population noninferiority or superiority",
            next_action="Review stage attribution before any next training campaign")
        manifest["status"]="COMPLETED"
    except BaseException:
        decision.update(error=traceback.format_exc(),failed_stage=manifest.get("current_stage"),
            next_action="Inspect failed stage; repair harness only if scientific contract and budget unchanged")
        manifest["status"]="FAILED"
    finally:
        manifest["end_utc"]=datetime.datetime.now(datetime.timezone.utc).isoformat()
        write(out/"decision.json",decision);write(out/"manifest.json",manifest)
        write(out/"progress.json",dict(terminal=True,status=manifest["status"],decision=str(out/"decision.json")))
        ledger=BASE/"reference-diagnostics.json";records=read(ledger)
        for resource in CAPS:
            identity="kdm-stage-isolation-"+out.name+"_"+resource
            assert not any(x["id"]==identity for x in records)
            records.append(dict(id=identity,resource=resource,charged_seconds=manifest["resource_wall_seconds"][resource],
                summary=str((out/"decision.json").relative_to(ROOT)),summary_sha256=sha(out/"decision.json"),
                manifest=str((out/"manifest.json").relative_to(ROOT)),plan_file=PLAN,result_file=RESULT,
                stage="KDM_STAGE_ISOLATION_DIAGNOSTIC",after_attempt=balance["attempts"],
                next_action=decision["next_action"],executions=[dict(path=str(out.relative_to(ROOT)),
                    resource=resource,wall_seconds=manifest["resource_wall_seconds"][resource],status=manifest["status"])]))
        write(ledger,records)
    print(json.dumps(dict(status=manifest["status"],output=str(out),resources=manifest["resource_wall_seconds"])),flush=True)
    return 0 if manifest["status"]=="COMPLETED" else 1
if __name__=="__main__":raise SystemExit(main())
