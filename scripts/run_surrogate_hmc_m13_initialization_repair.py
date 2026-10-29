#!/usr/bin/env python3
"""One bounded initialization repair through the public surrogate-HMC tuner."""
from pathlib import Path
import argparse,json,os,resource,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
PLAN="docs/plans/surrogate-hmc-m13-initialization-repair-2026-09-27.md"
PRIOR=ROOT/"docs/plans/artifacts/surrogate-hmc-closure-20260922-m13-public-tuning-03"
CAP=18000.
REPAIR_CAP=22000.

def worker(output):
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]="true"
    os.environ["TF_NUM_INTRAOP_THREADS"]="2"
    os.environ["TF_NUM_INTEROP_THREADS"]="1"
    os.environ["OMP_NUM_THREADS"]="2"
    import tensorflow as tf
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
    memory=configure_tensorflow_gpu_memory_growth(tf)
    if not tf.config.list_physical_devices("GPU"): raise RuntimeError("GPU required")
    from bayesfilter.testing.surrogate_hmc_m13_public_tuning import public_tuning_stage
    from bayesfilter.inference.surrogate_hmc_program import write_json
    result=public_tuning_stage(output/"report.json",initialization_repair=True)
    result.update(memory_policy=memory,device="gpu",
        gpu_trust_basis="owner_designated_managed_session_visible_gpu_trusted")
    write_json(output/"report.json",result)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--worker",action="store_true")
    args=parser.parse_args()
    if args.worker:
        worker(args.output); return 0
    from bayesfilter.inference.surrogate_hmc_program import run_job,write_json,file_hash
    out=args.output.resolve()
    from bayesfilter.inference.surrogate_hmc_closure_program import preceding_charges
    previous_global,_=preceding_charges(out)
    followup=dict(cpu_worker_seconds=56000.,gpu_seconds=48000.)
    for key,limit in dict(cpu_worker_seconds=198000.,gpu_seconds=226800.).items():
        if previous_global[key]+CAP+followup[key]>limit:
            raise ValueError("repair plus retained reservation exceeds campaign budget")
    receipts=[ROOT/"docs/plans/artifacts/surrogate-hmc-closure-20260922-m13-materiality-repair-01/accounting.json"]
    artifacts=ROOT/"docs/plans/artifacts"
    for pattern in ("surrogate-hmc-m13-force-localization-*/accounting.json",
                    "surrogate-hmc-m13-resume-regression-*/accounting.json",
                    "surrogate-hmc-m13-initialization-tests-*/accounting.json",
                    "surrogate-hmc-m13-initialization-repair-*/accounting.json"):
        receipts.extend(artifacts.glob(pattern))
    receipts.extend(artifacts/f"surrogate-hmc-m13-assessment-rwm-20260926-{i:02}/accounting.json" for i in range(2,7))
    charges={k:sum(float(json.loads(p.read_text())["charged_this_run"][k]) for p in receipts)
             for k in ("cpu_worker_seconds","gpu_seconds")}
    if any(v+CAP>REPAIR_CAP for v in charges.values()):
        raise ValueError("single repair would exceed revised allocation")
    if any(p.parent.name.startswith("surrogate-hmc-m13-initialization-repair-") for p in receipts):
        raise ValueError("the one initialization retry was already attempted")
    out.mkdir(parents=True,exist_ok=False)
    prior=json.loads((PRIOR/"run-manifest.json").read_text())
    names=set(prior["source_hashes"])
    names.update((PLAN,str(Path(__file__).relative_to(ROOT)),
        "bayesfilter/testing/surrogate_hmc_m13_initialization_repair.py",
        "bayesfilter/testing/surrogate_hmc_m13_public_tuning.py",
        "tests/test_surrogate_hmc_m13_initialization_repair.py"))
    hashes={name:file_hash(ROOT/name) for name in names}
    command=[sys.executable,str(Path(__file__).resolve()),"--output",str(out/"m13-public-tuning"),"--worker"]
    manifest=dict(plan=PLAN,command=sys.argv,worker_command=command,source_hashes=hashes,
        commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        python=sys.executable,device="gpu",data_seed=1704,seeds=dict(nomination=[20260926,1101],fresh=[20260927,1102]),
        cpu_cap=CAP,gpu_cap=CAP,previous_repair_charges=charges,
        previous_campaign_charges=previous_global,retained_reservation=followup,
        prior_receipts={str(p):file_hash(p) for p in receipts},result_file=str(out/"report.json"))
    write_json(out/"run-manifest.json",manifest)
    ledger=dict(limits=dict(cpu_worker_seconds=CAP,gpu_seconds=CAP),
                charged=dict(cpu_worker_seconds=0.,gpu_seconds=0.),jobs=[])
    before=resource.getrusage(resource.RUSAGE_SELF); start=time.monotonic()
    job=run_job(command,out/"m13-public-tuning",ledger,device="gpu",timeout=CAP-10,
                cpu_allowance=CAP-10,max_timeout_seconds=CAP)
    after=resource.getrusage(resource.RUSAGE_SELF)
    ledger["charged"]["cpu_worker_seconds"]+=after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime
    changed=[name for name,digest in hashes.items() if file_hash(ROOT/name)!=digest]
    result_path=out/"m13-public-tuning/report.json"
    result=json.loads(result_path.read_text()) if result_path.exists() else {}
    result_status=job["status"] if not changed else "SOURCE_CHANGED"
    report=dict(status=result_status,source_unchanged=not changed,changed_sources=changed,
        stages={"m13-public-tuning":dict(execution=result_status,result=result)},
        master_complete=False,plan=PLAN)
    write_json(out/"report.json",report)
    manifest.update(wall_seconds=time.monotonic()-start,charged=ledger["charged"])
    write_json(out/"run-manifest.json",manifest)
    write_json(out/"accounting.json",dict(charged_this_run=ledger["charged"],jobs=ledger["jobs"]))
    print(json.dumps(dict(status=result_status,qualified=result.get("qualified"),charged=ledger["charged"])),flush=True)
    return 0 if result_status=="COMPLETED" else 1

if __name__=="__main__": raise SystemExit(main())
