"""Bounded supervisor for the reviewed source-stage refresh and complete-data smoke."""
from __future__ import annotations
import datetime, hashlib, json, os, shutil, signal, subprocess, sys, time, traceback
from pathlib import Path
ROOT=Path("/home/chakwong/BayesFilter")
BASE=ROOT/"docs/plans/artifacts/controlled-smc-kdm-replication-20260928"
PHASE=BASE/"stage-refresh-20261011"
PLAN="docs/plans/controlled-smc-kdm-mechanism-repair-20261011.md"
RESULT="docs/plans/controlled-smc-kdm-mechanism-result-20261011.md"
TF="/home/chakwong/anaconda3/envs/tftwogpu/bin/python"
SOURCE="/home/chakwong/anaconda3/envs/tf-gpu2/bin/python"
CAPS={"cpu":0.,"gpu":900.}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def write(p,value):
    tmp=p.with_suffix(p.suffix+".tmp");tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n");tmp.replace(p)
def main(out):
    from run_replication_gap_localization import campaign
    if out.parent!=PHASE or not out.name.startswith("stage-refresh-"):raise ValueError("Unreviewed output path")
    prior=list(PHASE.glob("stage-refresh-*/manifest.json"))
    if len(prior)>=3:raise RuntimeError("Root-cause attempt cap exhausted")
    used=dict(cpu=0.,gpu=0.)
    for p in prior:
        m=read(p)
        if m["status"]=="RUNNING":raise RuntimeError("A root-cause attempt is active")
        for r in used:used[r]+=m["resource_wall_seconds"][r]
    current=campaign()
    for r,cap in CAPS.items():
        if min(current["remaining_seconds"][r],current["tranche_remaining_seconds"][r])<cap-used[r]:
            raise RuntimeError("Insufficient "+r+" budget")
    out.mkdir(parents=True,exist_ok=False);sources=out/"sources";sources.mkdir()
    scripts=["run_replication_gap_localization.py","run_replication_stage_refresh.py",
        "diagnose_kdm_dense_layout_tf.py","diagnose_csmc_source_feasibility.py",
        "reference_kdm_proposal_adjoints.py","diagnose_kdm_proposal_adjoints_tf.py","reference_kdm_neural_exact_replay.py",
        "reference_kdm_native_training_smoke.py","diagnose_kdm_native_training_smoke_tf.py",
        "diagnose_kdm_source_stages_tf.py","diagnose_kdm_source_epoch_tf.py",
        "diagnose_kdm_fresh_training_cost_tf.py"]
    paths=[ROOT/"docs/benchmarks"/s for s in scripts]+[ROOT/PLAN,ROOT/"bayesfilter/runtime/gpu_memory_policy.py"]
    paths+=list((ROOT/"bayesfilter/replication").glob("*.py"))
    for p in paths:
        dest=sources/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    previous=BASE/"gap-closure-20261010"
    stage_reference=BASE/"kdm-source-stages-20261007-04/diagnostic/result.json"
    fresh_reference=BASE/"kdm-fresh-training-cost-20261007-01/diagnostic/result.json"
    inputs=[stage_reference,fresh_reference]
    for ref in (read(stage_reference),read(fresh_reference)):
        inputs += [Path(x["path"]) for x in ref.get("datasets",{}).values()]
        inputs += [Path(x["path"]) for x in ref.get("fixtures",{}).values()]
    baseline=read(stage_reference)
    inputs.append(Path(baseline["baseline"]))
    assert sha(Path(baseline["baseline"]))==baseline["baseline_sha256"]
    manifest=dict(schema="replication_stage_refresh_v1",status="RUNNING",command=[sys.executable,*sys.argv],
        git_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        cwd=str(ROOT),plan_file=PLAN,result_file=RESULT,source_hashes={str(p.relative_to(ROOT)):sha(p) for p in paths},
        input_hashes={str(p.relative_to(ROOT)):sha(p) for p in inputs},started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        supervisor_pid=os.getpid(),resource_wall_seconds=dict(cpu=0.,gpu=0.),steps=[],environments=dict(cpu=SOURCE,gpu=TF),
        data_version="pinned source stage fixtures and fresh-constructor development data",
        random_seeds=[811,731,202610081,202610082],replication="INCOMPLETE",gpu_trust_basis="escalated_local_gpu_access")
    def save():write(out/"manifest.json",manifest)
    save()
    cpu=dict(os.environ,CUDA_VISIBLE_DEVICES="-1",OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",
        LD_LIBRARY_PATH="/home/chakwong/anaconda3/envs/tf-gpu2/lib")
    gpu=dict(os.environ,CUDA_VISIBLE_DEVICES="GPU-d54fdcfc-c6ed-dbe7-25c7-93f737e0f93a",
        TF_FORCE_GPU_ALLOW_GROWTH="true",TF_CPP_MIN_LOG_LEVEL="2",OMP_NUM_THREADS="1")
    def run(command,resource,label,env):
        remaining=CAPS[resource]-used[resource]-manifest["resource_wall_seconds"][resource]
        if remaining<=0:raise RuntimeError("Stage budget exhausted")
        start=time.monotonic();step=dict(command=command,resource=resource,log=label+".log",status="RUNNING")
        manifest["steps"].append(step);save()
        try:
            with (out/step["log"]).open("w") as log:
                child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                try:code=child.wait(timeout=remaining)
                except BaseException:
                    os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=10)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                    raise
            step.update(exit_code=code,status="COMPLETED" if code==0 else "FAILED")
            if code:raise RuntimeError(label+" failed; inspect "+step["log"])
        finally:
            elapsed=time.monotonic()-start;step["wall_seconds"]=elapsed
            manifest["resource_wall_seconds"][resource]+=elapsed;save()
    try:
        run(["nvidia-smi","--query-gpu=uuid,name,memory.used,memory.total","--format=csv"],"gpu","nvidia",gpu)
        probe="import sys,json;sys.path.insert(0,"+repr(str(ROOT))+");import tensorflow as tf;from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth;m=configure_tensorflow_gpu_memory_growth(tf,require_gpu=True);print(json.dumps(dict(memory_policy=m,tensorflow=tf.__version__,gpus=[d.name for d in tf.config.list_logical_devices('GPU')])) )"
        run([TF,"-c",probe],"gpu","framework",gpu)
        verification=out/"verification";verification.mkdir()
        proof_manifest=dict(status="RUNNING",hashes=manifest["source_hashes"],reference=str(stage_reference))
        write(verification/"manifest.json",proof_manifest)
        run([TF,"-B",str(ROOT/"docs/benchmarks/diagnose_kdm_source_stages_tf.py"),
            "--mode","verify","--output",str(verification/"diagnostic"),
            "--reference",str(stage_reference)],"gpu","verification",gpu)
        proof=verification/"diagnostic/result.json"
        proof_manifest.update(status="COMPLETED",result_sha256=sha(proof))
        write(verification/"manifest.json",proof_manifest)
        if read(proof)["status"]!="PASS":raise RuntimeError("Source-stage verification failed")
        run([TF,"-B",str(ROOT/"docs/benchmarks/diagnose_kdm_fresh_training_cost_tf.py"),
            "--epochs","smoke","--output",str(out/"smoke"),
            "--reference",str(fresh_reference),"--verification",str(proof)],
            "gpu","complete-data-smoke",gpu)
        smoke=read(out/"smoke/result.json")
        if smoke["status"]!="PASS" or len(smoke["completed_stages"])!=3:
            raise RuntimeError("Complete-data stage smoke failed")
        manifest["result_paths"]=[str(proof),str(out/"smoke/result.json")]
        assert all(sha(ROOT/p)==value for p,value in manifest["source_hashes"].items()),"Source changed during run"
        assert all(sha(ROOT/p)==value for p,value in manifest["input_hashes"].items()),"Input changed during run"
        manifest["status"]="COMPLETED"
        decision=dict(status="COMPLETED",scope="DIAGNOSTICS_ONLY",replication="INCOMPLETE")
    except BaseException:
        manifest["status"]="FAILED";(out/"error.txt").write_text(traceback.format_exc())
        decision=dict(status="ERROR",scope="HARNESS_OR_EXECUTION_FAILURE",replication="INCOMPLETE")
    finally:
        write(out/"decision.json",decision);manifest["decision_sha256"]=sha(out/"decision.json");save()
        ledger_path=BASE/"reference-diagnostics.json";ledger=read(ledger_path)
        for r in CAPS:
            record=dict(id=out.name+"_"+r,resource=r,charged_seconds=manifest["resource_wall_seconds"][r],
                summary=str((out/"decision.json").relative_to(ROOT)),summary_sha256=sha(out/"decision.json"),
                manifest=str((out/"manifest.json").relative_to(ROOT)),plan_file=PLAN,result_file=RESULT,
                stage="REPLICATION_STAGE_REFRESH",next_action="Review full-stage results before complete training.",
                after_attempt=current["attempts"],executions=[dict(path=str(out.relative_to(ROOT)),resource=r,
                    wall_seconds=manifest["resource_wall_seconds"][r])])
            assert not any(x["id"]==record["id"] for x in ledger),"Duplicate budget charge"
            ledger.append(record)
        write(ledger_path,ledger)
    print(json.dumps(dict(status=manifest["status"],resource_wall_seconds=manifest["resource_wall_seconds"],output=str(out))))
    return 0 if manifest["status"]=="COMPLETED" else 1
