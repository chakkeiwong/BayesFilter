"""Independent R/TF adaptive-consumer comparison; diagnostic only.

Records actual factory calls without changing their outputs. R receives realized
draws but constructs its own clouds and recursively fitted guides. No NumPy
computation, alternate runtime filter, or production admission is introduced.
"""
import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def r_literal(value):
    if isinstance(value, dict):
        return "list(" + ",".join(json.dumps(k) + "=" + r_literal(v) for k, v in value.items()) + ")"
    if isinstance(value, (list, tuple)):
        return "list(" + ",".join(r_literal(v) for v in value) + ")"
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, (int, float)) and math.isfinite(value):
        return repr(value)
    raise ValueError(f"nonfinite or unsupported R input: {type(value)}")


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def r_run(output, mode, payload):
    source = output / "input.R"
    source.write_text("input <- " + r_literal(payload) + "\n")
    command = ["Rscript", "--vanilla", "docs/benchmarks/check_iapf_tf_adaptive.R", mode,
               str(source), str(output)]
    started = time.monotonic()
    with (output / "R.log").open("w") as log:
        run = subprocess.run(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, timeout=600)
    return dict(command=command, exit_code=run.returncode, wall_seconds=time.monotonic()-started)


def setup(device):
    import tensorflow as tf
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    if device == "gpu":
        from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
        policy = configure_tensorflow_gpu_memory_growth(tf)
        if not tf.config.list_physical_devices("GPU"):
            raise RuntimeError("GPU requested but unavailable")
    else:
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
            raise RuntimeError("CPU reference requires explicitly hidden GPU")
        policy = {"mode": "CPU_reference_GPU_intentionally_hidden"}
    tf.config.experimental.enable_tensor_float_32_execution(False)
    return tf, dict(tensorflow=tf.__version__, build=tf.sysconfig.get_build_info(),
                    memory_policy=policy, device=device, dtype="float64", jit_compile=True,
                    tf32=False, visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"))


def controls():
    return dict(k=2, tau=.5, max_iterations=12, max_particles=512, mean_bound=4.,
                sd_lower=.2, sd_upper=4., max_fit_steps=2000, max_backtracks=30,
                fit_tolerance=1e-7, floor_ratio=.01,
                fit_theta=[.55, math.log(.6), math.log(.7), 1., .2, math.log(.8)])


def preflight(tf, output):
    from bayesfilter.score_study.iapf_fit_tf import bounded_density_fit, _density_profile, make_density_recursive_fit_kernel
    fixtures = []
    for d in (2, 5):
        points = tf.random.stateless_normal([128, d], [93011, d], dtype=tf.float64)
        center = tf.linspace(tf.constant(-.2, tf.float64), tf.constant(.3, tf.float64), d)
        variance = tf.linspace(tf.constant(.7, tf.float64), tf.constant(1.1, tf.float64), d)
        targets = -.5*tf.reduce_sum((points-center)**2/variance, axis=1)
        config = controls()
        @tf.function(input_signature=[tf.TensorSpec([128, d], tf.float64), tf.TensorSpec([128], tf.float64)],
                     jit_compile=True)
        def fit(x, y):
            return bounded_density_fit(x, y, mean_bound=config["mean_bound"], sd_lower=config["sd_lower"],
                sd_upper=config["sd_upper"], max_steps=config["max_fit_steps"],
                max_backtracks=config["max_backtracks"], tolerance=config["fit_tolerance"],
                floor_ratio=config["floor_ratio"])
        c, v, floor, info = fit(points, targets)
        par = tf.linspace(tf.constant(-.15, tf.float64), tf.constant(.2, tf.float64), 2*d)
        standardized = (points-tf.reduce_mean(points, 0))/tf.math.reduce_std(points, 0)
        profile = _density_profile(standardized, targets-tf.reduce_max(targets), par)
        fixtures.append(dict(d=d, points=points.numpy().tolist(), targets=targets.numpy().tolist(),
            center=c.numpy().tolist(), covariance=v.numpy().tolist(), floor=float(floor.numpy()),
            expected_center=center.numpy().tolist(), expected_variance=variance.numpy().tolist(),
            valid=bool(info["valid"].numpy()), converged=bool(info["converged"].numpy()),
            projected_gradient=float(info["projected_gradient"].numpy()),
            known_center_error=float(tf.reduce_max(tf.abs(c-center)).numpy()),
            known_variance_error=float(tf.reduce_max(tf.abs(tf.linalg.diag_part(v)-variance)).numpy()),
            parameters=par.numpy().tolist(), profile_loss=float(profile[0].numpy()),
            profile_gradient=profile[1].numpy().tolist(), trace_count=fit.experimental_get_tracing_count()))
        clouds=tf.stack([points,points*.8+.1])
        observations=tf.zeros([2,d],tf.float64)
        recursive=make_density_recursive_fit_kernel(d,d,128,2,4.,.2,4.,2000,30,1e-7,.01)
        recursive_result=recursive(tf.constant(config["fit_theta"],tf.float64),observations,clouds)
        fixtures[-1]["recursive"]=dict(clouds=clouds.numpy().tolist(),observations=observations.numpy().tolist(),
            centers=recursive_result[0].numpy().tolist(),covariances=recursive_result[1].numpy().tolist(),
            floors=recursive_result[2].numpy().tolist(),valid=bool(recursive_result[3].numpy()),
            converged=bool(recursive_result[4].numpy()),diagnostics=recursive_result[5].numpy().tolist())
    payload=dict(config=controls(), fixtures=fixtures)
    save(output / "tf.json", payload)
    return dict(r=r_run(output,"preflight",payload),
                tf_pass=all(x["valid"] and x["converged"] and x["known_center_error"]<2e-5
                            and x["known_variance_error"]<2e-5 and x["recursive"]["valid"]
                            and x["recursive"]["converged"] for x in fixtures))


def consumer(tf, output, cases):
    from bayesfilter.score_study import fitted_twist_tf, iapf_fit_tf
    from bayesfilter.score_study.iapf_adapter import execute_iapf
    from bayesfilter.score_study.gaussian_tf import make_data_kernel
    from bayesfilter.score_study.contracts import DiagnosticFailure
    factory = fitted_twist_tf.make_fitted_twist_kernel
    fit_factory = iapf_fit_tf.make_density_recursive_fit_kernel
    summaries = []
    for d, o, case_seed in cases:
        directory = output / f"d{d}-o{o}-s{case_seed}"
        directory.mkdir()
        N, T = 64, 4
        config=controls(); theta=tf.constant(config["fit_theta"],tf.float64)
        generated=make_data_kernel(d,o,T)(theta,tf.constant([937000+case_seed,d*10+o],tf.int32))
        observations=generated
        # Separate namespaces for data, fitting, and final streams.
        def seed(name, replicate=case_seed, group="final"):
            encoded=f"{case_seed}:{d}:{o}:{group}:{replicate}:{name}".encode()
            raw=hashlib.sha256(encoded).digest()
            return [int.from_bytes(raw[:4],"little")%(2**31-1),int.from_bytes(raw[4:8],"little")%(2**31-1)]
        runs=[]; fits=[]
        def capture_factory(*args,**kwargs):
            kernel=factory(*args,**kwargs)
            trace_kernel=factory(*args,**kwargs,include_numerical_trace=True)
            class Captured:
                def __call__(self,*values):
                    result=kernel(*values)
                    trace=trace_kernel(*values)
                    equal=all(bool(tf.reduce_all(a==b).numpy()) for a,b in zip(result,trace[:3]))
                    names=("theta","observations","initial","process","ancestors","mixture",
                           "centers","covariances","floors")
                    runs.append(dict(constant=bool(kwargs.get("constant_twist",False)),
                        particles=int(values[2].shape[0]),input={k:v.numpy().tolist() for k,v in zip(names,values)},
                        value=float(result[0].numpy()),score=result[1].numpy().tolist(),clouds=result[2].numpy().tolist(),
                        trace={k:v.numpy().tolist() for k,v in trace[3].items()},trace_bitwise_equal=equal))
                    return result
                def experimental_get_tracing_count(self):
                    return kernel.experimental_get_tracing_count()
            return Captured()
        def capture_fit_factory(*args,**kwargs):
            kernel=fit_factory(*args,**kwargs)
            class Captured:
                def __call__(self,*values):
                    result=kernel(*values)
                    fits.append(dict(centers=result[0].numpy().tolist(),covariances=result[1].numpy().tolist(),
                        floors=result[2].numpy().tolist(),valid=bool(result[3].numpy()),
                        converged=bool(result[4].numpy()),diagnostics=result[5].numpy().tolist()))
                    return result
                def experimental_get_tracing_count(self):
                    return kernel.experimental_get_tracing_count()
            return Captured()
        settings=dict(dimension=d,observation_dimension=o,particles=N,horizon=T,dtype="float64",jit_compile=True)
        row=dict(model="linear_gaussian",method="iapf",role="diagnostic",iapf=config)
        started=time.monotonic(); status="complete"; error=None; diagnostics={}
        with patch.object(fitted_twist_tf,"make_fitted_twist_kernel",capture_factory), \
             patch.object(iapf_fit_tf,"make_density_recursive_fit_kernel",capture_fit_factory):
            try:
                _,_,diagnostics,calls=execute_iapf(row,settings,theta,observations,seed)
            except DiagnosticFailure as exc:
                status="diagnostic_failure"; error=str(exc); diagnostics=exc.diagnostics["details"]; calls=len(runs)+len(fits)
        elapsed=time.monotonic()-started
        payload=dict(d=d,o=o,seed=case_seed,config=config,settings=settings,status=status,error=error,
            runs=runs,fits=fits,diagnostics=diagnostics,calls=calls,wall_seconds=elapsed,
            actual_consumer="bayesfilter.score_study.iapf_adapter.execute_iapf",
            actual_filter="bayesfilter.score_study.fitted_twist_tf.make_fitted_twist_kernel",
            actual_fitter="bayesfilter.score_study.iapf_fit_tf.make_density_recursive_fit_kernel")
        save(directory/"tf.json",payload)
        r=r_run(directory,"consumer",payload)
        summary=dict(d=d,o=o,seed=case_seed,status=status,error=error,calls=calls,wall_seconds=elapsed,
                     r=r,actual_particle_count=diagnostics.get("actual_particle_count"),
                     offline_iterations=len(diagnostics.get("fit_iterations",[])))
        summaries.append(summary); save(output/"partial.json",summaries)
        print(json.dumps(summary),flush=True)
    return dict(cases=summaries)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--mode",choices=("preflight","consumer","r_replay"),required=True)
    parser.add_argument("--device",choices=("cpu","gpu"),default="cpu")
    parser.add_argument("--source-attempt",type=Path)
    parser.add_argument("--cases",default="1:1:41,1:1:42,2:1:41,2:1:42,2:2:41,2:2:42,5:3:41,5:3:42,5:5:41,5:5:42")
    args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    if args.mode=="r_replay":
        if args.source_attempt is None or args.device!="cpu":
            raise ValueError("R replay requires saved source and CPU classification")
        results=[]
        for source in sorted(args.source_attempt.glob("*/tf.json")):
            payload=json.loads(source.read_text()); destination=args.output/source.parent.name; destination.mkdir()
            record=dict(source=str(source),sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                        classification="R_only_recheck_of_preserved_TF_execution")
            save(destination/"input-source.json",record)
            results.append(dict(d=payload["d"],o=payload["o"],seed=payload["seed"],status=payload["status"],
                                r=r_run(destination,"consumer",payload)))
        save(args.output/"summary.json",dict(cases=results,wall_seconds=time.monotonic()-started))
        return
    tf,environment=setup(args.device)
    save(args.output/"manifest.json",dict(command=sys.argv,git_commit=subprocess.check_output(
        ["git","rev-parse","HEAD"],cwd=REPO,text=True).strip(),environment=environment,
        utc=dt.datetime.now(dt.timezone.utc).isoformat(),plan="docs/plans/iapf-adaptive-consumer-parity-2026-09-22.md"))
    with tf.device("/GPU:0" if args.device=="gpu" else "/CPU:0"):
        result=preflight(tf,args.output) if args.mode=="preflight" else consumer(tf,args.output,
            [tuple(map(int,item.split(":"))) for item in args.cases.split(",")])
    result["wall_seconds"]=time.monotonic()-started
    save(args.output/"summary.json",result)


if __name__=="__main__":
    main()
