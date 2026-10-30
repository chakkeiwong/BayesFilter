"""Bounded replication-only diagnostic supervisor; standard-library orchestration."""
from __future__ import annotations
import argparse,datetime,hashlib,importlib.util,json,os,signal,shutil,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path("/home/chakwong/BayesFilter")
BASE=ROOT/"docs/plans/artifacts/controlled-smc-kdm-replication-20260928"
PHASE=BASE/"gap-closure-20261010"
PLAN="docs/plans/controlled-smc-kdm-gap-closure-execution-20261010.md"
RESULT="docs/plans/controlled-smc-kdm-gap-closure-result-20261010.md"
TF="/home/chakwong/anaconda3/envs/tftwogpu/bin/python"
SOURCE="/home/chakwong/anaconda3/envs/tf-gpu2/bin/python"
GPU="GPU-d54fdcfc-c6ed-dbe7-25c7-93f737e0f93a"
CAPS={"cpu":1200.,"gpu":3600.}
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text())
def write(p,x):
    p=Path(p);tmp=p.with_suffix(p.suffix+".tmp")
    tmp.write_text(json.dumps(x,indent=2,allow_nan=False)+"\n");tmp.replace(p)
def campaign():
    sys.path.insert(0,str(ROOT/"scripts"))
    spec=importlib.util.spec_from_file_location("gap_master",ROOT/"scripts/run_controlled_smc_kdm.py")
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.status()
def main(out):
    if out.parent!=PHASE or not out.name.startswith("localization-"):
        raise ValueError("Use the reviewed versioned localization output root")
    prior=list(PHASE.glob("localization-*/manifest.json"))
    if len(prior)>=2: raise RuntimeError("Attempt cap exhausted")
    used={"cpu":25.767,"gpu":0.}
    for p in prior:
        m=read(p)
        if m["status"]=="RUNNING": raise RuntimeError("A localization is active")
        for r in used: used[r]+=m["resource_wall_seconds"][r]
    current=campaign()
    remaining={r:min(current["remaining_seconds"][r],current["tranche_remaining_seconds"][r]) for r in CAPS}
    for r in CAPS:
        if remaining[r]<CAPS[r]-used[r]: raise RuntimeError("Insufficient existing "+r+" campaign budget")
    out.mkdir(exist_ok=False)
    sources=out/"sources";sources.mkdir()
    paths=[Path(__file__).resolve(),ROOT/PLAN,ROOT/"docs/benchmarks/reference_kdm_coupled_updates.py",
        ROOT/"docs/benchmarks/diagnose_kdm_neural_precision_tf.py",ROOT/"docs/benchmarks/compare_lorenz_fitting_forward_tf.py",
        ROOT/"docs/benchmarks/reference_kdm_neural_exact_replay.py",ROOT/"bayesfilter/replication/kdm_neural_replay_tf.py",
        ROOT/"bayesfilter/replication/kdm_normal_von_mises_tf.py",ROOT/"bayesfilter/replication/multivariate_controlled_smc_tf.py",
        ROOT/"bayesfilter/replication/controlled_smc_tf.py",ROOT/"bayesfilter/runtime/gpu_memory_policy.py"]
    for p in paths:
        dest=sources/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    manifest=dict(schema="replication_gap_localization_v1",status="RUNNING",command=[sys.executable,*sys.argv],
        cwd=str(ROOT),git_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        plan_file=PLAN,result_file=RESULT,started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        supervisor_pid=os.getpid(),source_hashes={str(p.relative_to(ROOT)):sha(p) for p in paths},
        phase_cap_seconds=CAPS,prior_phase_seconds=used,campaign_remaining_before=remaining,
        resource_wall_seconds={"cpu":0.,"gpu":0.},executions=[],seeds=[202610104],
        data_version="Preserved development training rows384:640; frozen six-case cSMC source fixture",
        gpu_trust_basis="escalated_local_gpu_access",replication="INCOMPLETE")
    def save(): write(out/"manifest.json",manifest)
    save()
    def run(command,resource,label,env):
        budget=CAPS[resource]-used[resource]-manifest["resource_wall_seconds"][resource]
        if budget<=0: raise RuntimeError("Phase resource cap exhausted")
        e=dict(command=command,resource=resource,status="RUNNING",timeout_seconds=budget,log=label+".log",
            environment={k:env.get(k) for k in ("CUDA_VISIBLE_DEVICES","LD_LIBRARY_PATH","TF_FORCE_GPU_ALLOW_GROWTH",
                "OMP_NUM_THREADS","TF_NUM_INTEROP_THREADS","TF_NUM_INTRAOP_THREADS")})
        manifest["executions"].append(e);save()
        tick=time.monotonic();process=None
        try:
            with (out/(label+".log")).open("x") as log:
                process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                e["pid"]=process.pid;save()
                code=process.wait(timeout=budget)
            e.update(returncode=code,status="COMPLETED" if code==0 else "FAILED")
            if code: raise RuntimeError(label+" exit "+str(code))
        except BaseException:
            e["status"]="FAILED"
            if process is not None and process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
                try: process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGKILL);process.wait()
            raise
        finally:
            elapsed=time.monotonic()-tick;e["wall_seconds"]=elapsed
            manifest["resource_wall_seconds"][resource]+=elapsed;save()
    cpu=dict(os.environ,CUDA_VISIBLE_DEVICES="-1",LD_LIBRARY_PATH="/home/chakwong/anaconda3/envs/tf-gpu2/lib",
        OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1")
    gpu=dict(os.environ,CUDA_VISIBLE_DEVICES=GPU,TF_FORCE_GPU_ALLOW_GROWTH="true",
        TF_NUM_INTEROP_THREADS="1",TF_NUM_INTRAOP_THREADS="1",OMP_NUM_THREADS="1")
    gpu.pop("LD_LIBRARY_PATH",None)
    try:
        run(["nvidia-smi","--query-gpu=uuid,name,memory.used,memory.total,utilization.gpu","--format=csv"],"gpu","nvidia",gpu)
        probe="import sys,json;sys.path.insert(0,"+repr(str(ROOT))+");import tensorflow as tf;from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth;m=configure_tensorflow_gpu_memory_growth(tf,require_gpu=True);tf.config.experimental.enable_tensor_float_32_execution(False);print(json.dumps(dict(memory_policy=m,tensorflow=tf.__version__,gpus=[d.name for d in tf.config.list_logical_devices('GPU')])) )"
        run([TF,"-c",probe],"gpu","framework",gpu)
        csmc=out/"csmc";csmc.mkdir();shutil.copytree(sources/"bayesfilter",csmc/"sources/bayesfilter")
        fixture=PHASE/"c0-forward-source-20261010"
        manifest["csmc_fixture_sha256"]=sha(fixture/"fixture.json");save()
        run([TF,str(sources/"docs/benchmarks/compare_lorenz_fitting_forward_tf.py"),"--root",str(ROOT),
            "--output",str(csmc),"--reference",str(fixture)],"gpu","csmc",gpu)
        cs=read(csmc/"summary.json")
        if cs["status"]!="PASS" or not all(x["forward_screen_passed"] for x in cs["source_controls"]):
            raise RuntimeError("cSMC source-policy controls invalid")
        run([SOURCE,"-I","-B",str(sources/"docs/benchmarks/reference_kdm_coupled_updates.py"),
            "--output",str(out/"author.json")],"cpu","kdm-source",cpu)
        author=read(out/"author.json")
        if len(author["cases"])!=4 or not all(x["pass_check"] for x in author["replay_controls"]):
            raise RuntimeError("Coupled source controls invalid")
        run([TF,"-B",str(sources/"docs/benchmarks/diagnose_kdm_neural_precision_tf.py"),"--output",str(out/"kdm"),
            "--reference",str(out/"author.json"),"--plan",PLAN,"--result-note",RESULT],"gpu","kdm-tf",gpu)
        kd=read(out/"kdm/decision.json")
        manifest["status"]="COMPLETED"
        decision=dict(status="PASS",scope="BOUNDED_LOCALIZATION_EXECUTION_ONLY",replication="INCOMPLETE",
            csmc=cs["solver_verdicts"],kdm=kd,comparison_verdict="LOCALIZATION_ACCOUNTED_NOT_QUALIFICATION")
    except BaseException:
        manifest["status"]="FAILED";(out/"error.txt").write_text(traceback.format_exc())
        decision=dict(status="ERROR",scope="HARNESS_OR_EXECUTION_FAILURE",replication="INCOMPLETE")
    finally:
        write(out/"decision.json",decision);manifest["decision_sha256"]=sha(out/"decision.json")
        save()
        ledger_path=BASE/"reference-diagnostics.json";ledger=read(ledger_path)
        for r in CAPS:
            record=dict(id=out.name+"_"+r,resource=r,charged_seconds=manifest["resource_wall_seconds"][r],
                summary=str((out/"decision.json").relative_to(ROOT)),summary_sha256=sha(out/"decision.json"),
                manifest=str((out/"manifest.json").relative_to(ROOT)),plan_file=PLAN,result_file=RESULT,
                stage="REPLICATION_GAP_LOCALIZATION",next_action="Inspect first-divergence; candidate mismatch triggers planned repair.",
                after_attempt=current["attempts"],executions=[dict(path=str(out.relative_to(ROOT)),resource=r,
                    wall_seconds=manifest["resource_wall_seconds"][r])])
            if not any(x["id"]==record["id"] for x in ledger): ledger.append(record)
        write(ledger_path,ledger)
    print(json.dumps(dict(status=manifest["status"],decision=decision,resource_wall_seconds=manifest["resource_wall_seconds"])))
    return 0 if manifest["status"]=="COMPLETED" else 1
if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--phase",choices=("localization","root-causes","mechanism","mechanism-adjoints","native-smoke","stage-refresh","training-pilot","training-full"),default="localization");args=parser.parse_args()
    if args.phase=="training-full":
        from run_replication_training_pilot_full import main as selected_main
    elif args.phase=="training-pilot":
        from run_replication_training_pilot import main as selected_main
    elif args.phase=="stage-refresh":
        from run_replication_stage_refresh import main as selected_main
    elif args.phase=="native-smoke":
        from run_replication_native_smoke import main as selected_main
    elif args.phase=="root-causes":
        from run_replication_root_causes import main as selected_main
    elif args.phase=="mechanism":
        from run_replication_mechanism import main as selected_main
    elif args.phase=="mechanism-adjoints":
        from run_replication_mechanism import main as mechanism_main
        selected_main=lambda out:mechanism_main(out,adjoints=True)
    else:selected_main=main
    raise SystemExit(selected_main(args.output.resolve()))
