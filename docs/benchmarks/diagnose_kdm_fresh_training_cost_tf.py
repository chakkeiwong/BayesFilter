"""GPU/XLA cost probe from fresh author constructor weights and fresh data."""
import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]="true"
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/"docs/plans/artifacts/controlled-smc-kdm-replication-20260928"
sys.path[:0]=[str(ROOT),str(Path(__file__).resolve().parent)]

def write(path,value):
    temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value,allow_nan=False)+"\n")
    temporary.replace(path)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--reference",type=Path,required=True)
    p.add_argument("--verification",type=Path,required=True)
    p.add_argument("--epochs",choices=("cost","smoke","full"),default="cost")
    p.add_argument("--seed-base",type=int,default=202610081)
    args=p.parse_args()
    if not 0 <= args.seed_base <= 2**31-1000000:
        p.error("--seed-base must leave room for the source epoch offsets")
    out=args.output.resolve(); out.mkdir(parents=True,exist_ok=False)
    checks=[]
    result=dict(schema="kdm_fresh_tf_cost_v1",replication="INCOMPLETE",checks=checks,timings=[],
        reference=str(args.reference),reference_sha256=hashlib.sha256(args.reference.read_bytes()).hexdigest(),
        initialization="fresh source constructor values; TF preceding best checkpoint transfers",
        existing_float32_failures="PRESERVED",source_training_protocol_complete=False,
        scope="FRESH_INITIALIZATION_COMPLETE_EPOCH_COST_NOT_QUALIFICATION")
    result["epoch_mode"]=args.epochs
    result["seed_base"]=args.seed_base
    if args.epochs!="cost":
        result["schema"]="kdm_complete_tf_training_v1"
        result["scope"]="DEVELOPMENT_TRAINING_NOT_QUALIFICATION"
    started=time.monotonic()
    try:
        source=json.loads(args.reference.read_text())
        manifest=json.loads((args.reference.parent.parent/"manifest.json").read_text())
        if (source["status"]!="PASS" or len(source["stages"])!=3 or manifest["status"]!="COMPLETED"
                or manifest["result_sha256"]!=result["reference_sha256"]):
            raise ValueError("Incomplete source cost reference")
        fixtures={}
        for name,info in source["fixtures"].items():
            path=Path(info["path"])
            if hashlib.sha256(path.read_bytes()).hexdigest()!=info["sha256"]: raise ValueError("Fixture checksum mismatch")
            fixtures[name]=json.loads(path.read_text())
        for info in source["datasets"].values():
            if hashlib.sha256(Path(info["path"]).read_bytes()).hexdigest()!=info["sha256"]: raise ValueError("Data checksum mismatch")
        cfg=source["source_training_configs"]
        reference=dict(datasets=source["datasets"],pretraining_cases=[fixtures["weights"]],
            pretraining_config=cfg["weights"],full_training_config=cfg["mdpf"],full_training_configs=cfg)
        baseline=dict(cases=[fixtures["mdpf"],fixtures["am_dpf"]])
        import tensorflow as tf
        from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
        memory=configure_tensorflow_gpu_memory_growth(tf,require_gpu=True)
        tf.config.experimental.enable_tensor_float_32_execution(False)
        tf.config.set_soft_device_placement(False)
        result.update(tensorflow_version=tf.__version__,memory_policy=memory,
            visible_device=os.environ.get("CUDA_VISIBLE_DEVICES"),jit_compile=True,tf32=False,
            datasets=source["datasets"],constructor_fixtures=source["fixtures"],
            random_backend="TF stateless Philox; same checked laws, independent Torch streams")
        def boolean(name,value):
            checks.append(dict(name=name,passed=bool(value)))
            if not value: raise RuntimeError("Cost screen failed: "+name)
        def run(label,function,*inputs):
            write(out/"progress.json",dict(stage=label,completed_checks=len(checks),terminal=False))
            tick=time.monotonic(); value=function(*inputs); bool(value["valid"].numpy())
            if args.epochs=="cost": result["timings"].append(dict(stage=label,wall_seconds=time.monotonic()-tick))
            return value
        from diagnose_kdm_source_epoch_tf import run_epochs
        run_epochs(tf,reference,baseline,args,out,result,boolean,run,write,
            require_same_reference=False,seed_base=args.seed_base,
            epoch_limit={"cost":1,"smoke":2,"full":None}[args.epochs])
        result["status"]="PASS"
    except Exception as error:
        result.update(status="FAIL",error=repr(error),traceback=traceback.format_exc())
    result.update(wall_seconds=time.monotonic()-started,checks_total=len(checks),
                  checks_failed=sum(not c["passed"] for c in checks))
    write(out/"result.json",result)
    write(out/"progress.json",dict(terminal=True,status=result["status"]))
    return 0 if result["status"]=="PASS" else 1
if __name__=="__main__": raise SystemExit(main())
