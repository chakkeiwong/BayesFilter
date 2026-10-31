"""Bounded supervisor for common-base-variate replication diagnostics."""
import argparse,datetime,hashlib,json,os,signal,shutil,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path("/home/chakwong/BayesFilter")
BASE=ROOT/"docs/plans/artifacts/controlled-smc-kdm-replication-20260928"
PHASE=BASE/"gradient-followup-20261013"
PLAN="docs/plans/controlled-smc-kdm-gradient-followup-plan-20261013.md"
RESULT="docs/plans/controlled-smc-kdm-gradient-followup-result-20261013.md"
PY="/home/chakwong/anaconda3/envs/tftwogpu/bin/python"
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):
    tmp=p.with_suffix(p.suffix+".tmp");tmp.write_text(json.dumps(x,indent=2,allow_nan=False)+"\n");tmp.replace(p)
def main():
    parser=argparse.ArgumentParser();parser.add_argument("--stage",choices=("pretraining","trained","angle"),required=True);parser.add_argument("--angle-intervention",action="store_true");args=parser.parse_args()
    assert not args.angle_intervention or args.stage=="angle"
    driver={"pretraining":"diagnose_kdm_fresh_pretraining.py","trained":"diagnose_kdm_trained_gradient_replay.py","angle":"diagnose_kdm_angle_adjoint.py"}[args.stage]
    if args.angle_intervention:driver="diagnose_kdm_angle_intervention.py"
    sys.path.insert(0,str(ROOT/"docs/benchmarks"))
    from run_replication_gap_localization import campaign
    balance=campaign();assert not balance["running"]
    prior=list(PHASE.glob(args.stage+"*/manifest.json"));assert len(prior)<2
    assert all(read(p)["status"]!="RUNNING" for p in prior)
    initial=[p for p in PHASE.glob("*/manifest.json") if p not in prior]
    assert all(read(p)["status"]!="RUNNING" for p in initial)
    remaining={r:min(3600-sum(read(p)["resource_wall_seconds"][r] for p in prior+initial),balance["remaining_seconds"][r],balance["tranche_remaining_seconds"][r]) for r in ("cpu","gpu")}
    assert min(remaining.values())>180
    out=PHASE/(args.stage+"%02d"%(len(prior)+1));out.mkdir(parents=True,exist_ok=False)
    files=[Path(__file__).resolve(),ROOT/PLAN,ROOT/("docs/benchmarks/"+driver),
        ROOT/"docs/benchmarks/diagnose_kdm_crossed_starts_tf.py",
        ROOT/"docs/benchmarks/reference_kdm_coupled_trajectory.py",ROOT/"docs/benchmarks/reference_kdm_neural_exact_replay.py",
        *[ROOT/("bayesfilter/replication/"+n) for n in ("kdm_neural_replay_tf.py","kdm_sampling_tf.py","kdm_source_stages_tf.py","kdm_normal_von_mises_tf.py")],
        ROOT/"bayesfilter/runtime/gpu_memory_policy.py"]
    files.extend([p for p in (ROOT/"docs/benchmarks/reference_kdm_angle_adjoint.py",ROOT/"docs/benchmarks/reference_kdm_trained_gradient_replay.py",ROOT/"docs/benchmarks/diagnose_kdm_validation_control.py",ROOT/"docs/benchmarks/reference_kdm_crossed_starts.py",ROOT/"docs/benchmarks/reference_kdm_source_stages.py",ROOT/"docs/benchmarks/reference_kdm_neural.py",ROOT/"bayesfilter/replication/kdm_training_controller.py") if p.exists()])
    for p in files:
        dest=out/"sources"/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
    command=[PY,"-I","-B",str(ROOT/("docs/benchmarks/"+driver)),"--output",str(out/"diagnostic")]
    if args.stage=="pretraining":command.append("--gradient-replay")
    env=dict(os.environ,TF_FORCE_GPU_ALLOW_GROWTH="true",OMP_NUM_THREADS="1",TF_NUM_INTEROP_THREADS="1",TF_NUM_INTRAOP_THREADS="1",
        LD_LIBRARY_PATH="/home/chakwong/anaconda3/envs/tftwogpu/lib",BAYESFILTER_PRELOAD_CUSTOM_OP="0")
    manifest=dict(status="RUNNING",supervisor_pid=os.getpid(),command=command,
        git_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        plan_file=PLAN,result_file=RESULT,start_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        environment=PY,source_hashes={str(p.relative_to(ROOT)):sha(p) for p in files},
        stage=args.stage,seeds=[202610123 if args.stage=="pretraining" else 202610122],phase_cap_seconds=3600,attempt_cap=2,per_launch_cap_seconds=1200,campaign_remaining_before=remaining,
        resource_wall_seconds=dict(cpu=0.,gpu=0.),data_version="unit01/unit02 source training and validation; hashes in diagnostic provenance",
        gpu_trust_basis="escalated_local_gpu_access",replication="INCOMPLETE")
    manifest["git_dirty"]=subprocess.check_output(["git","status","--short","--untracked-files=no"],cwd=ROOT,text=True)
    write(out/"manifest.json",manifest)
    start=time.monotonic();proc=None;decision=dict(status="FAILED")
    try:
        with (out/"nvidia-smi.log").open("w") as log:
            subprocess.run(["nvidia-smi","--query-gpu=name,uuid,memory.used,memory.total","--format=csv"],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=30)
        with (out/"execution.log").open("w") as log:
            proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            manifest["worker_pid"]=proc.pid;write(out/"manifest.json",manifest)
            code=proc.wait(timeout=min(1200,min(remaining.values()))-(time.monotonic()-start))
            if code:raise RuntimeError("Trajectory diagnostic exit "+str(code))
        decision=read(out/"diagnostic/result.json");manifest["status"]="COMPLETED"
    except BaseException:
        manifest["status"]="FAILED";decision["error"]=traceback.format_exc()
    finally:
        if proc is not None and proc.poll() is None:
            os.killpg(proc.pid,signal.SIGTERM)
            try:proc.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
        elapsed=time.monotonic()-start
        manifest["resource_wall_seconds"]={r:elapsed for r in ("cpu","gpu")}
        manifest["end_utc"]=datetime.datetime.now(datetime.timezone.utc).isoformat()
        write(out/"decision.json",decision);write(out/"manifest.json",manifest)
        ledger=BASE/"reference-diagnostics.json";records=read(ledger)
        for r in ("cpu","gpu"):
            identity="kdm-gradient-followup-"+out.name+"_"+r
            assert not any(x["id"]==identity for x in records)
            records.append(dict(id=identity,resource=r,charged_seconds=elapsed,
                summary=str((out/"decision.json").relative_to(ROOT)),summary_sha256=sha(out/"decision.json"),
                manifest=str((out/"manifest.json").relative_to(ROOT)),plan_file=PLAN,result_file=RESULT,
                stage="KDM_GRADIENT_FOLLOWUP_DIAGNOSTIC",after_attempt=balance["attempts"],
                next_action="Review identical-source-state gradient replay; no automatic qualification",
                executions=[dict(path=str(out.relative_to(ROOT)),resource=r,wall_seconds=elapsed,status=manifest["status"])]))
        write(ledger,records)
    print(json.dumps(dict(status=manifest["status"],output=str(out),charged_seconds_each_resource=elapsed)),flush=True)
    return 0 if manifest["status"]=="COMPLETED" else 1
if __name__=="__main__":raise SystemExit(main())
