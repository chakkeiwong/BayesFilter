#!/usr/bin/env python3
"""Diagnostic GPU timing: synchronized kernel versus report serialization."""
import argparse, hashlib, json, os, statistics, subprocess, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    os.environ.update(CUDA_VISIBLE_DEVICES="GPU-68251639-fe82-8f81-3ccc-2953c32e805b",
        TF_FORCE_GPU_ALLOW_GROWTH="true",TF_CPP_MIN_LOG_LEVEL="2",
        TF_NUM_INTRAOP_THREADS="4",TF_NUM_INTEROP_THREADS="2")
    sys.path.insert(0,str(ROOT))
    import tensorflow as tf
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
    memory=configure_tensorflow_gpu_memory_growth(tf)
    if not tf.config.list_physical_devices("GPU"): raise RuntimeError("GPU required")
    tf.config.experimental.enable_tensor_float_32_execution(True)
    from bayesfilter.testing import marginal_weight_campaign as base,marginal_expanded as ex
    start=time.monotonic(); rows=[]
    campaign=ROOT/"docs/plans/artifacts/surrogate-hmc-marginal-expanded-20261006-03"
    for name in ("ksc-t10","m13-t10"):
        spec=base.specification(name)
        controls=json.loads((campaign/"scopes"/(name+"-n1024")/"controls.json").read_text())["controls"]
        observed=tf.cast(base.dataset(spec,"validation",dataset_id=1),tf.float32)
        design=base.designs(spec,1024,"validation",0,dataset_id=1)
        theta=tf.constant(spec["points"][0],tf.float32)
        ledh=ex.compile_batch(spec,1024,controls,"marginal_mixture")
        bpf=ex.Kernels(spec).get("bootstrap",1024,dtype=tf.float32)
        ledh_args=(theta[None],*design,observed)
        base.serial(ledh(*ledh_args))
        base.serial(bpf(theta,observed,tf.constant([20261006,940000],tf.int32)))
        timings={"marginal_mixture":[],"bootstrap":[]}
        for replicate in range(12):
            seed=tf.constant([20261006,940001+replicate],tf.int32)
            for method,function,inputs in (
                ("marginal_mixture",ledh,ledh_args),
                ("bootstrap",bpf,(theta,observed,seed))):
                tick=time.perf_counter()
                value=function(*inputs)
                value["value"].numpy()
                synced=time.perf_counter()
                ready=base.serial(value)
                end=time.perf_counter()
                timings[method].append(dict(synchronized_call_seconds=synced-tick,
                    report_ready_call_seconds=end-tick,serialization_seconds=end-synced,evaluation=ready))
        summary={method:{key:statistics.median(r[key] for r in values)
                         for key in values[0] if key != 'evaluation'} for method,values in timings.items()}
        rows.append(dict(scope=name,n=1024,timings=timings,medians=summary))
        base.save(args.output/"progress.json",dict(rows=rows))
        print(name,summary,flush=True)
    base.save(args.output/"result.json",dict(status="COMPLETE",rows=rows,
        command=sys.argv,wall_seconds=time.monotonic()-start,
        git_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        environment=dict(python=sys.executable,tensorflow=tf.__version__,gpu=os.environ["CUDA_VISIBLE_DEVICES"],
            memory_policy=memory,jit_compile=True,tf32=True,cpu_only=False),
        data_version="validation dataset 1, design 0, scope-frozen controls",
        seed_policy="20261006, 940000..940012 for timing only; no accuracy selection",
        plan="docs/plans/surrogate-hmc-marginal-expanded-execution-2026-10-06.md",
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            (Path(__file__),ROOT/"bayesfilter/testing/marginal_expanded.py")},
        nonclaim="Timing localization only. No accuracy, statistical speed ranking, or HMC-force comparison."))

if __name__=="__main__": main()
