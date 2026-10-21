#!/usr/bin/env python3
"""Diagnostic-only stationary initialization check; no likelihood/HMC claim."""
import argparse, hashlib, json, os, subprocess, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--tf32", choices=("on", "off"), default="on")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ.update(CUDA_VISIBLE_DEVICES=("-1" if args.device=="cpu" else
        "GPU-68251639-fe82-8f81-3ccc-2953c32e805b"), TF_FORCE_GPU_ALLOW_GROWTH="true",
        TF_CPP_MIN_LOG_LEVEL="2", TF_NUM_INTRAOP_THREADS="2", TF_NUM_INTEROP_THREADS="1")
    sys.path.insert(0, str(ROOT))
    import tensorflow as tf
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
    memory = configure_tensorflow_gpu_memory_growth(tf) if args.device=="gpu" else "GPU intentionally hidden"
    if args.device=="gpu" and not tf.config.list_physical_devices("GPU"):
        raise RuntimeError("GPU required")
    tf.config.experimental.enable_tensor_float_32_execution(args.tf32=="on")
    from bayesfilter.highdim.ledh_linear_parameter_adapter_tf import stationary_covariance_and_tangent
    from bayesfilter.testing.marginal_weight_campaign import specification, serial, save
    rows=[]; start=time.monotonic()
    for d in (1,3,10,20):
        for dtype in (tf.float32,tf.float64):
            spec=specification(f"lg{d}-t10",dtype); a=spec["a"]; p=spec["p"]
            def calculation(theta):
                f=theta[0]*a
                q=tf.eye(d,dtype=dtype)*(tf.exp(2*theta[1]) if p==2 else tf.constant(1.,dtype))
                cov,dcov,status=stationary_covariance_and_tangent(f,q,a,tf.zeros_like(q))
                # Independent spectral solution, with exactly the input F and Q
                # supplied to the candidate solve; no float64 decimal reconstruction.
                af=tf.cast(a,tf.float64); th=tf.cast(theta,tf.float64)
                # Closed-form sine basis of this symmetric tridiagonal Toeplitz
                # matrix avoids XLA eigh's approximate iterative stopping.
                if d==1:
                    eig=tf.ones([1],tf.float64); vec=tf.ones([1,1],tf.float64)
                else:
                    import math
                    k=tf.cast(tf.range(1,d+1),tf.float64)
                    angle=tf.constant(math.pi/(d+1),tf.float64)*k
                    eig=af[0,0]+2*af[0,1]*tf.cos(angle)
                    vec=tf.sqrt(tf.constant(2./(d+1),tf.float64))*tf.sin(k[:,None]*angle[None,:])
                scale=tf.cast(q[0,0],tf.float64)
                den=1-(th[0]*eig)**2
                oracle=(vec*(scale/den)[None,:])@tf.transpose(vec)
                tangent=(vec*(2*th[0]*eig**2*scale/den**2)[None,:])@tf.transpose(vec)
                return dict(status=status,
                    covariance_relative_error=tf.reduce_max(tf.abs(tf.cast(cov,tf.float64)-oracle))/tf.reduce_max(tf.abs(oracle)),
                    tangent_relative_error=tf.reduce_max(tf.abs(tf.cast(dcov,tf.float64)-tangent))/tf.reduce_max(tf.abs(tangent)))
            fn=tf.function(calculation,input_signature=[tf.TensorSpec([p],dtype)],jit_compile=True,autograph=False)
            for theta in spec["points"]:
                out=serial(fn(tf.constant(theta,dtype)))
                row=dict(d=d,dtype=dtype.name,theta=theta,**out);rows.append(row)
                print(json.dumps(row,allow_nan=True),flush=True)
    result=dict(status="COMPLETE",rows=rows,wall_seconds=time.monotonic()-start,
        command=sys.argv,git_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        environment=dict(python=sys.executable,tensorflow=tf.__version__,cpu_only=args.device=="cpu",
            gpu=os.environ["CUDA_VISIBLE_DEVICES"],memory_policy=memory,jit_compile=True,tf32=args.tf32=="on"),
        seed="N/A deterministic matrices",data_version="declared campaign matrices and parameter points",
        plan="docs/plans/surrogate-hmc-marginal-expanded-execution-2026-10-06.md",
        nonclaim="Initialization localization only; no likelihood, sampler, or default-readiness evidence",
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [Path(__file__),ROOT/"bayesfilter/highdim/ledh_linear_parameter_adapter_tf.py"]})
    save(args.output/"result.json",result)
    print("COMPLETE",round(result["wall_seconds"],2))
if __name__=="__main__": main()
