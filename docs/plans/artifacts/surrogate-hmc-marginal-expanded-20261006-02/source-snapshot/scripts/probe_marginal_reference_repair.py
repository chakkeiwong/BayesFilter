#!/usr/bin/env python3
"""Bounded GPU/XLA diagnostic for dynamic reference factories; no model claim."""
import argparse, hashlib, json, os, subprocess, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    os.environ.update(CUDA_VISIBLE_DEVICES="GPU-68251639-fe82-8f81-3ccc-2953c32e805b",
        TF_FORCE_GPU_ALLOW_GROWTH="true",TF_CPP_MIN_LOG_LEVEL="2",
        TF_NUM_INTRAOP_THREADS="6",TF_NUM_INTEROP_THREADS="2")
    sys.path.insert(0,str(ROOT))
    import tensorflow as tf
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
    memory=configure_tensorflow_gpu_memory_growth(tf)
    if not tf.config.list_physical_devices("GPU"): raise RuntimeError("GPU required")
    tf.config.experimental.enable_tensor_float_32_execution(True)
    from bayesfilter.testing import marginal_expanded as ex
    m=ex.base;rows=[];start=time.monotonic()
    for name in ("lg3-t2","ksc-t2","m13-t2"):
        spec=m.specification(name);precise=m.specification(name,tf.float64)
        kernels=ex.Kernels(spec);helper=ex.comparator_module()
        theta=tf.constant(spec["points"][0],tf.float64);seed=tf.constant([731,91],tf.int32)
        kinds=["bootstrap","prior","gaussian",
            "kalman" if name.startswith("lg") else "ksc" if name.startswith("ksc") else "fisher"]
        for kind in kinds:
            function=kernels.get(kind,32,18.,4)
            for idx in (0,1):
                observed=m.dataset(spec,"engineering",dataset_id=idx)
                if kind=="kalman": old=m.kalman_reference(precise,observed)(theta)
                elif kind=="ksc": old=m.ksc_grid_reference(precise,observed,32,18.)(theta)
                elif kind=="fisher": old=m.m13_fisher_reference(precise,observed,32,4)(theta,seed)
                elif kind=="bootstrap": old=dict(value=m.bootstrap_reference(precise,observed,32)(theta,seed))
                elif kind=="prior": old=dict(value=helper.prior_trajectory_reference(tf,m,precise,observed,32)(theta,seed))
                else: old=helper.gaussian_reference(tf,m,precise,observed)(theta)
                current=function(theta,observed,seed)
                deltas={}
                for key in ("value","score"):
                    if key in old:
                        tf.debugging.assert_near(current[key],old[key],atol=1e-7,rtol=1e-7)
                        deltas[key]=float(tf.reduce_max(tf.abs(current[key]-old[key])))
                if not bool(tf.reduce_all(tf.math.is_finite(current["value"]))): raise RuntimeError("nonfinite")
                rows.append(dict(model=name,kind=kind,dataset=idx,legacy_delta=deltas))
            if function.experimental_get_tracing_count()!=1: raise RuntimeError("unexpected retracing")
            if kind in ("bootstrap","prior","gaussian"):
                single=kernels.get(kind,32,dtype=tf.float32)
                result=single(tf.cast(theta,tf.float32),tf.cast(observed,tf.float32),seed)
                if result["value"].dtype!=tf.float32 or not bool(tf.math.is_finite(result["value"])):
                    raise RuntimeError("invalid FP32 comparator")
                rows[-1]["fp32_value"]=float(result["value"])
                rows[-1]["fp32_finite"]=True
            print(name,kind,"pass",flush=True)
    payload=dict(status="COMPLETE",passed=True,checks=rows,wall_seconds=time.monotonic()-start,
        command=sys.argv,git_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        environment=dict(python=sys.executable,tensorflow=tf.__version__,gpu=os.environ["CUDA_VISIBLE_DEVICES"],
            memory_policy=memory,jit_compile=True,tf32=True,cpu_only=False),
        seed=[731,91],datasets=[0,1],plan="docs/plans/surrogate-hmc-marginal-expanded-execution-2026-10-06.md",
        nonclaim="Graph compatibility and legacy parity only; no full reference accuracy or HMC claim",
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [Path(__file__),ROOT/"bayesfilter/testing/marginal_expanded.py",
             ROOT/"bayesfilter/testing/marginal_weight_campaign.py",
             ROOT/"scripts/diagnose_marginal_weight_campaign.py"]})
    (args.output/"result.json").write_text(json.dumps(payload,indent=2))
    print("COMPLETE",round(payload["wall_seconds"],2))
if __name__=="__main__":main()
