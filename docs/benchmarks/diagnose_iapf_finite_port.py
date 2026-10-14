"""Independent-reference diagnostic of the finite iAPF TensorFlow consumer."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

ROOT = Path("/home/chakwong/BayesFilter")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--mode", choices=["cpu", "gpu-graph", "gpu-xla"], required=True)
    args = parser.parse_args()
    gpu = args.mode != "cpu"
    jit = args.mode == "gpu-xla"
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    os.environ["BAYESFILTER_PRELOAD_CUSTOM_OP"] = "0"
    if not gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    if gpu:
        probe = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=True)
        (args.output/"nvidia-smi.log").write_text(probe.stdout)
    sys.path.insert(0,str(ROOT))
    import tensorflow as tf
    memory = None
    if gpu:
        from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
        memory = configure_tensorflow_gpu_memory_growth(tf)
        if not tf.config.list_logical_devices("GPU"):
            raise RuntimeError("Trusted TensorFlow probe found no GPU")
    tf.config.experimental.enable_tensor_float_32_execution(False)
    from bayesfilter.score_study.iapf_finite_tf import (
        make_finite_fit_kernel, make_finite_objective_kernel, make_finite_backward_kernel,
        make_controller_decision_kernel, run_finite_iapf)
    from bayesfilter.score_study.iapf_paper_tf import make_paper_apf_value_kernel
    def plain(x):
        if tf.is_tensor(x):
            return plain(x.numpy().tolist())
        if isinstance(x,dict):
            return {str(k):plain(v) for k,v in x.items()}
        if isinstance(x,(tuple,list)):
            return [plain(v) for v in x]
        if isinstance(x,float) and not math.isfinite(x):
            return str(x)
        return x
    def save(name,payload):
        (args.output/name).write_text(json.dumps(plain(payload),indent=2,allow_nan=False)+"\n")
    def rows(path):
        with path.open() as stream:
            return list(csv.DictReader(stream))
    def read(folder,name,shape,dtype=tf.float64):
        with (folder/(name+".csv")).open() as stream:
            values=[float(value) for row in csv.reader(stream) for value in row]
        return tf.reshape(tf.constant(values,dtype),shape)
    checks=[]
    def check(name,got,want,tolerance=1e-5,exact=False):
        got=tf.convert_to_tensor(got)
        want=tf.cast(tf.convert_to_tensor(want),got.dtype)
        if exact or got.dtype in (tf.bool,tf.int32,tf.int64):
            passed=bool(tf.reduce_all(got==want).numpy())
            error=None
        else:
            difference=tf.abs(got-want)
            passed=bool(tf.reduce_all(tf.math.is_finite(got)&
                                      (difference<=tolerance*(1+tf.abs(want)))).numpy())
            error=float(tf.reduce_max(difference).numpy())
        checks.append(dict(name=name,passed=passed,maximum_absolute_error=error,
                           tolerance=None if exact else tolerance))
        save("checks.json",checks)
        return passed
    source_hashes={}
    for relative in ["bayesfilter/score_study/iapf_finite_tf.py",
                     "bayesfilter/score_study/iapf_paper_tf.py",
                     "bayesfilter/runtime/gpu_memory_policy.py"]:
        source=ROOT/relative
        digest=hashlib.sha256(source.read_bytes()).hexdigest()
        snapshot=args.output/"sources"/source.name
        if snapshot.exists() and hashlib.sha256(snapshot.read_bytes()).hexdigest()!=digest:
            raise RuntimeError("Runtime source changed after snapshot: "+relative)
        source_hashes[relative]=digest
    save("preflight-runtime.json",dict(tensorflow=tf.__version__,mode=args.mode,
         jit_compile=jit,dtype="float64",tf32=False,gpu_memory_policy=memory,
         cpu_only=not gpu,gpu_intentionally_hidden=not gpu,source_sha256=source_hashes,
         device_trust="escalated_execution" if gpu else "deliberate_cpu_reference_exception",
         sample_wise_python_loop=False,time_wise_python_loop=False,pfor=False,
         host_adaptive_pass_orchestration=True))
    kernels=[]
    start=time.monotonic()
    if gpu:
        tf.config.experimental.reset_memory_stats("GPU:0")
    with tf.device("/GPU:0" if gpu else "/CPU:0"):
        for case in rows(args.fixture/"fit-index.csv"):
            name=case["name"]; folder=args.fixture/name
            d,n=int(case["d"]),int(case["N"])
            points=read(folder,"points",[n,d]);targets=read(folder,"targets",[n])
            initial=read(folder,"initial",[2*d]);probe=read(folder,"probe",[2*d+1])
            scaled=tf.constant(case["scaled"]=="TRUE")
            objective=make_finite_objective_kernel(d,n,"float64",jit)
            kernels.append(objective)
            ev=objective(points,targets,initial,probe,scaled)
            check(name+"/objective",ev[0],tf.constant(float(case["probe_loss"]),tf.float64),1e-8)
            check(name+"/analytic_gradient",ev[1],read(folder,"probe_gradient",[2*d+1]),1e-8)
            kernel=make_finite_fit_kernel(d,n,"float64",jit);kernels.append(kernel)
            fit=kernel(points,targets,scaled)
            save(name+"-fit.json",fit)
            check(name+"/initializer",fit["initial"],initial,1e-8)
            check(name+"/initializer_method",fit["initial_method"],
                  1 if case["initializer"]=="diagonal_log_quadratic" else 0,exact=True)
            check(name+"/endpoint",fit["theta"],read(folder,"theta",[2*d]))
            check(name+"/amplitude",fit["amplitude"],tf.constant(float(case["amplitude"]),tf.float64))
            check(name+"/code",fit["code"],int(case["code"]),exact=True)
            check(name+"/valid",fit["valid"],case["valid"]=="TRUE",exact=True)
        fit_pass=all(c["passed"] for c in checks)
        controller=None
        if fit_pass:
            folder=args.fixture/"controller"
            scope_path=folder/"scope.csv"
            scope=rows(scope_path)[0] if scope_path.exists() else dict(d=2,T=8,N0=64,max_particles=512,max_iterations=12)
            d,horizon=int(scope["d"]),int(scope["T"])
            n0,max_n,max_it=int(scope["N0"]),int(scope["max_particles"]),int(scope["max_iterations"])
            A=read(folder,"A",[d,d]);C=read(folder,"C",[d,d]);R=read(folder,"R",[d,d])
            obs=read(folder,"observations",[horizon,d]);eye=tf.eye(d,dtype=tf.float64)
            call_rows=rows(folder/"calls.csv");ref=rows(folder/"decision.csv")[0]
            # First isolate the whole backward recursion on exactly the R clouds.
            for index in range(1,int(ref["backward_calls"])+1):
                n=int(call_rows[index-1]["N"])
                cloud=read(folder/("call"+str(index)),"clouds",[horizon,n,d])
                kernel=make_finite_backward_kernel(d,d,n,horizon,"float64",jit);kernels.append(kernel)
                backward=kernel(A,C,eye,R,obs,cloud,tf.constant(False))
                save("fixed-cloud-backward"+str(index)+".json",backward)
                reference=folder/("backward"+str(index))
                check("backward"+str(index)+"/valid",backward["valid"],True,exact=True)
                for key,shape in [("centers",[horizon,d]),("covariances",[horizon,d,d]),("floors",[horizon])]:
                    check("backward"+str(index)+"/"+key,backward[key],read(reference,key,shape))
                check("backward"+str(index)+"/codes",backward["codes"],
                      [int(r["code"]) for r in rows(reference/"fits.csv")],exact=True)
            if all(c["passed"] for c in checks):
                def entropy(index,n):
                    if index>len(call_rows) or n!=int(call_rows[index-1]["N"]):
                        raise RuntimeError("Adaptive controller diverged from reference calls/particle counts")
                    call=folder/("call"+str(index))
                    return (read(call,"noise",[horizon,n,d]),read(call,"mixture",[horizon,n]),
                            read(call,"ancestor_uniforms",[horizon,n]))
                def observer(kind,index,payload):
                    if kind=="filter" and horizon>8:
                        v=payload["value"]
                        save("adaptive-"+kind+str(index)+".json",
                             dict(particles=payload["particles"],log_likelihood=v[0],
                                  prefix=v[-1],ess=v[4],resampled=v[5],
                                  clouds_sha256=hashlib.sha256(tf.io.serialize_tensor(v[1]).numpy()).hexdigest()))
                    else:
                        save("adaptive-"+kind+str(index)+".json",payload)
                    if kind=="filter":
                        n=payload["particles"];reference=folder/("call"+str(index))
                        check("adaptive/filter"+str(index)+"/clouds",payload["value"][1],
                              read(reference,"clouds",[horizon,n,d]))
                        check("adaptive/filter"+str(index)+"/prefix",payload["value"][-1],
                              read(reference,"prefix",[horizon]))
                    else:
                        reference=folder/("backward"+str(index))
                        check("adaptive/backward"+str(index)+"/valid",payload["valid"],True,exact=True)
                        check("adaptive/backward"+str(index)+"/centers",payload["centers"],
                              read(reference,"centers",[horizon,d]))
                controller=run_finite_iapf(A,C,tf.zeros([d],tf.float64),eye,eye,R,obs,entropy,
                                         jit_compile=jit,observer=observer,
                                         n0=n0,max_particles=max_n,max_iterations=max_it)
                saved_controller=dict(controller)
                if horizon>8 and saved_controller["final"] is not None:
                    saved_controller["final"]={"log_likelihood":controller["final"][0],"prefix":controller["final"][-1]}
                save("adaptive-controller.json",saved_controller)
                check("adaptive/status",controller["status"]==ref["status"],True,exact=True)
                for key in ["filter_calls","backward_calls","stop_iteration","final_particles"]:
                    check("adaptive/"+key,controller[key],int(ref[key]),exact=True)
                if controller["final"] is not None:
                    check("adaptive/fresh_final",controller["final"][0],
                          tf.constant(float(ref["final_log_likelihood"]),tf.float64))
                for n,kind in controller["signatures"]:
                    kernels.append(make_paper_apf_value_kernel(d,d,n,horizon,"float64",jit) if kind=="filter"
                                   else make_finite_backward_kernel(d,d,n,horizon,"float64",jit))
                kernels.append(make_controller_decision_kernel(max_it,5,"float64",jit))
        trace_counts=[kernel.experimental_get_tracing_count() for kernel in set(kernels)]
        check("stable_signatures",all(x==1 for x in trace_counts),True,exact=True)
    decision=dict(status="PASS" if all(c["passed"] for c in checks) and controller is not None else "FAIL",
                  scope="FINITE_RECONSTRUCTION_PARITY_NOT_FULL_REPLICATION",checks=len(checks),
                  failures=[c["name"] for c in checks if not c["passed"]],mode=args.mode,
                  controller_executed=controller is not None,trace_counts=trace_counts,
                  seconds=time.monotonic()-start,host_peak_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  gpu_allocator=tf.config.experimental.get_memory_info("GPU:0") if gpu else None)
    save("decision.json",decision)
    with (args.output/"decision.csv").open("w") as stream:
        writer=csv.DictWriter(stream,fieldnames=["status","scope","checks","mode"])
        writer.writeheader()
        writer.writerow({key:decision[key] for key in writer.fieldnames})
    print(json.dumps(decision),flush=True)
    return 0 if decision["status"]=="PASS" else 2


if __name__=="__main__":
    raise SystemExit(main())
