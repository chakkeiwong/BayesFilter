#!/usr/bin/env python3
"""Actual consumer integration check for the selectively recomputed covariance."""
from __future__ import annotations
import argparse,hashlib,json,os,pathlib,subprocess,sys,time
ROOT=pathlib.Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser();p.add_argument("--output",type=pathlib.Path,required=True)
    p.add_argument("--snapshot",choices=("batch","both"),required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    os.environ.update(CUDA_VISIBLE_DEVICES="GPU-68251639-fe82-8f81-3ccc-2953c32e805b",
        TF_FORCE_GPU_ALLOW_GROWTH="true",TF_CPP_MIN_LOG_LEVEL="2",
        TF_NUM_INTRAOP_THREADS="4",TF_NUM_INTEROP_THREADS="2")
    sys.path.insert(0,str(ROOT))
    import tensorflow as tf
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
    memory=configure_tensorflow_gpu_memory_growth(tf)
    if not tf.config.list_physical_devices("GPU"):raise RuntimeError("trusted GPU required")
    tf.config.experimental.enable_tensor_float_32_execution(True)
    import importlib.util,importlib
    archive=ROOT/"docs/plans/artifacts/surrogate-hmc-marginal-expanded-20261006-03/source-before-continuation"
    names=(["ledh_canonical_score_tf"] if args.snapshot=="both" else [])+["ledh_canonical_batch_fused_tf"]
    source_overrides={}
    for name in names:
        fullname="bayesfilter.highdim."+name
        path=archive/"bayesfilter/highdim"/(name+".py")
        spec=importlib.util.spec_from_file_location(fullname,path)
        module=importlib.util.module_from_spec(spec);sys.modules[fullname]=module
        spec.loader.exec_module(module)
        setattr(importlib.import_module("bayesfilter.highdim"),name,module)
        source_overrides[fullname]=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    from bayesfilter.testing import marginal_weight_campaign as base,marginal_expanded as ex
    prior=ROOT/"docs/plans/artifacts/surrogate-hmc-marginal-covariance-selective-repair-20261006-01/result.json"
    old=json.loads(prior.read_text())
    start=time.monotonic();rows=[]
    for name in ("m13-t10",):
        spec=base.specification(name)
        fn=ex.compile_batch(spec,1024,base.controls(0),"marginal_mixture")
        for saved in (r for r in old["rows"] if r["scope"]==name):
            partition=saved["partition"];design_id=saved["design"];point=saved["point"]
            obs=tf.cast(base.dataset(spec,partition,dataset_id=1),tf.float32)
            design=base.designs(spec,1024,partition,design_id,dataset_id=1)
            tick=time.monotonic()
            raw=fn(tf.constant([spec["points"][point]],tf.float32),*design,obs)
            for key in ("value","score","valid"):
                raw[key].numpy()
            output=ex.one(dict(base.serial(raw),seconds=time.monotonic()-tick))
            before=saved["marginal_mixture"]
            delta=None
            if before["valid"] and output["valid"]:
                delta=dict(value=abs(output["value"]-before["value"]),
                    score=.1*(sum((a-b)**2 for a,b in zip(output["score"],before["score"]))/len(before["score"]))**.5)
            count=output["diagnostics"].get("pushed_covariance_precision_repair_count",{}).get("maximum",0)
            healthy_preserved=(not before["valid"] or
                output["valid"] and delta["value"]<.005 and delta["score"]<.005)
            required_recovery=(spec["d"]<10 or output["valid"])
            observed_recovery=(before["valid"] or spec["d"]<10 or count>0)
            rows.append(dict(scope=name,partition=partition,design=design_id,point=point,
                baseline_valid=before["valid"],candidate=output,delta=delta,
                healthy_preserved=healthy_preserved,required_recovery=required_recovery,
                observed_recovery=observed_recovery,
                passed=healthy_preserved and required_recovery and observed_recovery))
            base.save(args.output/"progress.json",dict(rows=rows))
        print(name,"checked",flush=True)
    paths=[pathlib.Path(__file__)]+[ROOT/p for p in
        ("bayesfilter/highdim/ledh_marginal_weights_tf.py",
         "bayesfilter/highdim/ledh_canonical_score_tf.py",
         "bayesfilter/highdim/ledh_canonical_batch_fused_tf.py",
         "bayesfilter/testing/marginal_expanded.py")]
    result=dict(status="COMPLETE",passed=all(r["passed"] for r in rows),rows=rows,
        source_overrides=source_overrides,baseline=str(prior),baseline_sha256=hashlib.sha256(prior.read_bytes()).hexdigest(),
        command=sys.argv,wall_seconds=time.monotonic()-start,
        environment=dict(python=sys.executable,tensorflow=tf.__version__,gpu=os.environ["CUDA_VISIBLE_DEVICES"],
            memory_policy=memory,jit_compile=True,tf32=True,cpu_only=False),
        git_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        plan="docs/plans/surrogate-hmc-marginal-expanded-execution-2026-10-06.md",
        data_version="calibration/validation dataset1, designs0/1, all declared parameter points",
        seed_policy="unchanged paired repository designs",
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        nonclaim="integration and numerical non-harm only; no accuracy ranking or HMC admission")
    base.save(args.output/"result.json",result)
    print("COMPLETE",result["passed"],flush=True)
if __name__=="__main__":main()
