#!/usr/bin/env python3
"""Bounded GPU diagnostic for the stationary-adapter precision repair."""
import argparse, hashlib, importlib.util, json, os, subprocess, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--old-adapter",type=Path,required=True)
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
    from bayesfilter.testing import marginal_weight_campaign as base, marginal_expanded as ex
    from bayesfilter.highdim.ledh_linear_parameter_adapter_tf import stationary_covariance_and_tangent
    module_spec=importlib.util.spec_from_file_location("protected_stationary_adapter",args.old_adapter)
    old=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(old)
    current_factory=base.linear_gaussian_parameter_model
    start=time.monotonic();rows=[]
    for name,n in (("lg1-t2",32),("lg3-t2",32),("ksc-t2",32),("m13-t2",32),("lg10-t2",1024),("lg20-t2",1024)):
        spec=base.specification(name)
        observed=tf.cast(base.dataset(spec,"engineering",dataset_id=3),tf.float32)
        design=base.designs(spec,n,"engineering",0,dataset_id=3)
        theta=tf.constant(spec["points"],tf.float32)
        base.linear_gaussian_parameter_model=old.linear_gaussian_parameter_model
        old_spec=base.specification(name)
        base.linear_gaussian_parameter_model=current_factory
        for policy in ex.POLICIES:
            current=ex.compile_batch(spec,n,base.controls(1 if spec["kind"]=="m13" else 0),policy,rows=len(spec["points"]))
            result=base.serial(current(theta,*design,observed))
            row=dict(scope=name,n=n,policy=policy,current=result)
            if n==32:
                previous=ex.compile_batch(old_spec,n,base.controls(1 if spec["kind"]=="m13" else 0),policy,rows=len(spec["points"]))
                reference=base.serial(previous(theta,*design,observed))
                errors=[abs(a-b)/(1+abs(b)) for a,b in zip(result["value"],reference["value"])]
                errors.extend(abs(a-b)/(1+abs(b)) for ar,br in zip(result["score"],reference["score"]) for a,b in zip(ar,br))
                row.update(previous=reference,maximum_scaled_difference=max(errors),
                    passed=all(result["valid"]) and all(reference["valid"]) and max(errors)<5e-5)
                if spec["kind"] in ("ksc","m13"):
                    row["passed"]=row["passed"] and max(errors)==0.
            else:
                row["initialization_rejected"]=any(result["diagnostics"][key]["maximum"]==0 and
                    result["diagnostics"].get("first_failure_code",{}).get("maximum",0)!=0
                    for key in ("first_failure_stage",) if key in result["diagnostics"])
                row["passed"]=all(result["valid"])
            rows.append(row)
            base.save(args.output/"progress.json",dict(rows=rows))
            print(name,policy,"valid",result["valid"],"difference",row.get("maximum_scaled_difference"),flush=True)
    @tf.function(input_signature=[tf.TensorSpec([1,1],tf.float32)],jit_compile=True,autograph=False)
    def guard(f):
        return stationary_covariance_and_tangent(f,tf.ones([1,1]),tf.ones([1,1]),tf.zeros([1,1]))[2]
    invalid=[base.serial(guard(tf.constant([[value]],tf.float32))) for value in (1.,1.1)]
    checks=all(not x["value_valid"] and not x["force_valid"] for x in invalid)
    result=dict(status="COMPLETE",passed=all(row["passed"] for row in rows[:8]) and checks,
        high_dimension_full_filter_valid=all(row["passed"] for row in rows[8:]),rows=rows,
        unstable_checks=invalid,wall_seconds=time.monotonic()-start,command=sys.argv,
        environment=dict(python=sys.executable,tensorflow=tf.__version__,gpu=os.environ["CUDA_VISIBLE_DEVICES"],
            memory_policy=memory,jit_compile=True,tf32=True),
        git_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        data_version="engineering dataset 3; 20261006 seed law",seed="stateless design 0",
        plan="docs/plans/surrogate-hmc-marginal-expanded-execution-2026-10-06.md",
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [args.old_adapter,ROOT/"bayesfilter/highdim/ledh_linear_parameter_adapter_tf.py",Path(__file__)]},
        nonclaim="Full-consumer repair diagnostic only; no accuracy, HMC, or default promotion")
    base.save(args.output/"result.json",result)
    print("COMPLETE",result["passed"],result["high_dimension_full_filter_valid"],result["wall_seconds"],flush=True)
if __name__=="__main__":main()
