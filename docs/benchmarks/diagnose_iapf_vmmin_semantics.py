"""Diagnostic-only differential tests of the actual shared finite optimizer."""
import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["BAYESFILTER_PRELOAD_CUSTOM_OP"] = "0"
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
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
    start = time.monotonic()
    if args.mode == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    else:
        (args.output/"nvidia-smi.log").write_text(subprocess.check_output(["nvidia-smi"], text=True))
    sys.path.insert(0, str(ROOT))
    import tensorflow as tf
    from bayesfilter.score_study.iapf_finite_tf import (
        _vmmin, make_finite_fit_kernel, make_diagnostic_fixed_initial_fit_kernel)
    gpu = args.mode != "cpu"
    memory = None
    if gpu:
        from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
        memory = configure_tensorflow_gpu_memory_growth(tf)
        if not tf.config.list_logical_devices("GPU"):
            raise RuntimeError("Trusted GPU unavailable")
    tf.config.experimental.enable_tensor_float_32_execution(False)
    jit = args.mode == "gpu-xla"
    device = "/GPU:0" if gpu else "/CPU:0"

    def plain(x):
        if tf.is_tensor(x):
            return plain(x.numpy().tolist())
        if isinstance(x, dict):
            return {str(k): plain(v) for k,v in x.items()}
        if isinstance(x, (list,tuple)):
            return [plain(v) for v in x]
        if isinstance(x, float) and not math.isfinite(x):
            return str(x)
        return x

    def save(name, x):
        (args.output/name).write_text(json.dumps(plain(x), indent=2)+"\n")

    def read(folder, name, shape=None):
        with (folder/(name+".csv")).open() as f:
            x=tf.constant([[float(a) for a in r] for r in csv.reader(f)],tf.float64)
        return tf.reshape(x,shape) if shape is not None else x

    sources={}
    for name in ["bayesfilter/score_study/iapf_finite_tf.py",
                 "bayesfilter/score_study/iapf_paper_tf.py",
                 "bayesfilter/runtime/gpu_memory_policy.py"]:
        p=ROOT/name
        if p.read_bytes() != (args.output/"sources"/p.name).read_bytes():
            raise RuntimeError("Runtime source drift: "+name)
        sources[name]=hashlib.sha256(p.read_bytes()).hexdigest()
    save("runtime.json", dict(tensorflow=tf.__version__,mode=args.mode,dtype="float64",
        jit_compile=jit,device=device,tf32=False,gpu_memory_policy=memory,
        gpu_intentionally_hidden=not gpu,trust="escalated_execution" if gpu else "CPU reference",
        nonjit_exception="semantic diagnostic; no default promotion" if not jit else None,
        source_sha256=sources,
        gradient_counts="R logical accepted-gradient counts; TF jointly computes trial f/g",
        trace_columns="z...,f,max_abs_gradient,step,iteration,healthy",
        numerical_trace_columns=["rejected_trials","reset","curvature","hessian_asymmetry"]))

    def kernel_for(name, dim, maxit):
        def ev(z):
            if name == "quadratic":
                delta=z-tf.constant([-.2,.4],tf.float64)
                c=tf.constant([1.,3.],tf.float64);f=tf.reduce_sum(c*delta**2);g=2*c*delta
            elif name == "overflow_trial":
                e=tf.exp(z[0]);f=e-1000*z[0];g=tf.reshape(e-1000,[1])
            elif name == "rosenbrock":
                r=z[1]-z[0]**2;f=100*r**2+(1-z[0])**2
                g=tf.stack([-400*z[0]*r-2*(1-z[0]),200*r])
            elif name == "stationary":
                f=tf.zeros([],tf.float64);g=tf.zeros_like(z)
            elif name == "nochange":
                f=z[0];g=tf.ones_like(z)
            elif name == "negative_curvature":
                f=-.5*tf.reduce_sum(z*z);g=-z
            else:
                f=tf.exp(z[0]);g=tf.reshape(f,[1])
            valid=tf.logical_and(tf.math.is_finite(f),tf.reduce_all(tf.math.is_finite(g)))
            return f,g,z,tf.ones([],tf.float64),valid
        @tf.function(input_signature=[tf.TensorSpec([dim],tf.float64)],jit_compile=jit)
        def kernel(z):
            return _vmmin(ev,z,maxit=maxit)
        return kernel

    checks=[];kernels=[]
    def check(name, ok, detail):
        checks.append(dict(name=name,passed=bool(ok),detail=plain(detail)))
    with (args.fixture/"index.csv").open() as f:
        cases=list(csv.DictReader(f))
    with tf.device(device):
        for row in cases:
            name=row["name"];dim=int(row["dim"]);folder=args.fixture/name
            kernel=kernel_for(name,dim,int(row["maxit"]));kernels.append(kernel)
            fit=plain(kernel(read(folder,"initial",[dim])));save(name+".json",fit)
            valid=fit["valid"];code=fit["code"]
            check(name+"_code",code==int(row["code"]),dict(actual=code,expected=row["code"]))
            if row["rejected_initial"] == "TRUE":
                check(name+"_invalid",not valid,"invalid initial value must fail closed")
                continue
            check(name+"_valid",valid,"finite accepted result")
            expected=plain(read(folder,"endpoint",[dim]))
            errors=[abs(a-b) for a,b in zip(fit["z"],expected)]
            check(name+"_endpoint",all(e <= 1e-8*(1+abs(b)) for e,b in zip(errors,expected)),
                  dict(error=max(errors),expected=expected,actual=fit["z"]))
            check(name+"_counts",fit["function_calls"]==int(row["function_calls"]) and
                fit["gradient_calls"]==int(row["gradient_calls"]),
                dict(actual=[fit["function_calls"],fit["gradient_calls"]],
                     expected=[int(row["function_calls"]),int(row["gradient_calls"])]))
            trace=fit["numerical_trace"][:fit["trace_length"]]
            check(name+"_symmetric",all(r[3]==0 for r in trace),"stored lower triangle mirrored")
            check(name+"_rejection_count",all(r[0]>=0 for r in trace),"rejected trials are nonnegative")
            if name == "overflow_trial":
                check(name+"_backtracked",int(row["nonfinite_function_calls"])>0 and
                      sum(r[0] for r in trace)>0,"reference overflow and actual rejected trial")
            if name == "negative_curvature":
                check(name+"_resets",sum(r[1] for r in trace)>=2,"nonpositive curvature restarts")
        # Exact saved inputs: diagnose initializer sensitivity without relaxing old gates.
        paper=ROOT/"docs/plans/artifacts/iapf-mdpf-whole-repair-20260928-01/attempt26_iapf_paper_adaptive_fixture_v2/controller"
        ref=paper/"backward2";x=read(paper/"call2","clouds",[100,1000,5])[53]
        ly=read(ref,"fit54_targets",[1000]);initial=read(ref,"fit54_initial",[10])
        expected=plain(read(ref,"fit54_theta",[10]));rtrace=plain(read(ref,"fit54_accepted"))
        replay=[]
        for fixed in (False,True):
            k=(make_diagnostic_fixed_initial_fit_kernel(5,1000,"float64",jit) if fixed
               else make_finite_fit_kernel(5,1000,"float64",jit));kernels.append(k)
            out=k(x,ly,initial,tf.constant(False)) if fixed else k(x,ly,tf.constant(False))
            fit=plain(out);save("paper-fixed-initial-"+str(fixed)+".json",fit)
            actual={1:[0.]*11}
            for tr in fit["trace"][:fit["trace_length"]]:
                actual.setdefault(int(tr[-2]),tr[:11])
            trajectory=[dict(iteration=i+1,
                z_error=max(abs(a-b) for a,b in zip(actual[i+1],rr[:11])))
                for i,rr in enumerate(rtrace) if i+1 in actual]
            save("paper-trajectory-fixed-"+str(fixed)+".json",trajectory)
            replay.append(dict(fixed_reference_initial=fixed,valid=fit["valid"],code=fit["code"],
                parameter_error=max(abs(a-b) for a,b in zip(fit["theta"],expected)),
                old_parameter_criterion=all(abs(a-b)<=1e-5*(1+abs(b)) for a,b in zip(fit["theta"],expected)),
                initializer_error=max(abs(a-b) for a,b in zip(fit["initial"],plain(initial))),
                first_iteration_error_above_1e_minus8=next((r["iteration"] for r in trajectory if r["z_error"]>1e-8),None)))
            check("paper_replay_valid_"+str(fixed),fit["valid"],"validity only; parity separately retained")
    counts=[k.experimental_get_tracing_count() for k in set(kernels)]
    check("one_trace_per_signature",all(n==1 for n in counts),counts)
    save("paper-replay-summary.json",replay);save("checks.json",checks)
    passed=all(c["passed"] for c in checks)
    decision=dict(status="PASS" if passed else "FAIL",verdict="FINITE_SEMANTIC_CONTROLS_ONLY",
                  checks=len(checks),failures=sum(not c["passed"] for c in checks),
                  paper_parameter_parity="PASS" if all(r["old_parameter_criterion"] for r in replay) else "FAIL",
                  scientific_replications="INCOMPLETE",wall_seconds=time.monotonic()-start)
    if gpu:
        decision["allocator"]=tf.config.experimental.get_memory_info("GPU:0")
    save("decision.json",decision)
    with (args.output/"decision.csv").open("w") as f:
        w=csv.DictWriter(f,fieldnames=["status","verdict","checks","failures","paper_parameter_parity"])
        w.writeheader();w.writerow({k:decision[k] for k in w.fieldnames})
    print(json.dumps(decision))
    return 0 if passed else 2


if __name__ == "__main__":
    sys.exit(main())
