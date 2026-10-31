"""Explanatory finite-fit sensitivity and actual-consumer diagnostic, not admission."""
import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["BAYESFILTER_PRELOAD_CUSTOM_OP"] = "0"
import argparse
import csv
import hashlib
import json
import math
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
    args = parser.parse_args()
    start = time.monotonic()
    (args.output/"nvidia-smi.log").write_text(
        subprocess.check_output(["nvidia-smi"], text=True))
    sys.path.insert(0, str(ROOT))
    import tensorflow as tf
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
    memory = configure_tensorflow_gpu_memory_growth(tf)
    if not tf.config.list_logical_devices("GPU"):
        raise RuntimeError("Trusted GPU unavailable")
    tf.config.experimental.enable_tensor_float_32_execution(False)
    from bayesfilter.score_study.iapf_finite_tf import (
        make_finite_fit_kernel, make_finite_backward_kernel,
        make_controller_decision_kernel, run_finite_iapf)
    from bayesfilter.score_study.iapf_paper_tf import make_paper_apf_value_kernel

    def plain(value):
        if tf.is_tensor(value):
            return plain(value.numpy().tolist())
        if isinstance(value, dict):
            return {str(k): plain(v) for k, v in value.items()}
        if isinstance(value, (tuple, list)):
            return [plain(v) for v in value]
        if isinstance(value, float) and not math.isfinite(value):
            return str(value)
        return value

    def save(name, value):
        (args.output/name).write_text(json.dumps(plain(value), indent=2)+"\n")

    def rows(path):
        with path.open() as stream:
            return list(csv.DictReader(stream))

    def read(folder, name, shape=None):
        with (folder/(name+".csv")).open() as stream:
            value = tf.constant([[float(x) for x in row] for row in csv.reader(stream)],
                                tf.float64)
        return tf.reshape(value, shape) if shape is not None else value

    def number(value):
        return float(value.numpy())

    def maximum(a, b):
        return number(tf.reduce_max(tf.abs(a-b)))

    sources = {}
    for name in ["bayesfilter/score_study/iapf_finite_tf.py",
                 "bayesfilter/score_study/iapf_paper_tf.py",
                 "bayesfilter/runtime/gpu_memory_policy.py"]:
        path = ROOT/name
        if path.read_bytes() != (args.output/"sources"/path.name).read_bytes():
            raise RuntimeError("Runtime source drift: "+name)
        sources[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    save("runtime.json", dict(tensorflow=tf.__version__, dtype="float64", tf32=False,
        jit_compile=True, gpu_memory_policy=memory, device="/GPU:0",
        gpu_intentionally_hidden=False, trust="escalated_execution",
        seeds="Exact R fixture data9286300; entropy9286400+call; no new random draws",
        source_sha256=sources, role="DIAGNOSTIC_ONLY_NO_PARITY_OR_SCIENTIFIC_PROMOTION"))

    folder = args.fixture/"controller"
    scope = rows(folder/"scope.csv")[0]
    d, horizon, n0 = int(scope["d"]), int(scope["T"]), int(scope["N0"])
    max_n, max_it = int(scope["max_particles"]), int(scope["max_iterations"])
    if (d,horizon,n0)!=(5,100,1000):
        raise ValueError("This diagnostic is frozen to d5 T100 N1000")
    reference = rows(folder/"decision.csv")[0]
    call_rows = rows(folder/"calls.csv")
    summaries = []
    kernels = []
    with tf.device("/GPU:0"):
        A = read(folder, "A", [d, d])
        C = read(folder, "C", [d, d])
        R = read(folder, "R", [d, d])
        obs = read(folder, "observations", [horizon, d])
        eye = tf.eye(d, dtype=tf.float64)
        x = read(folder/"call2", "clouds", [horizon, n0, d])[53]
        ly = read(folder/"backward2", "fit54_targets", [n0])
        rtheta = read(folder/"backward2", "fit54_theta", [2*d])
        kernel = make_finite_fit_kernel(d, n0, "float64", True)
        kernels.append(kernel)

        @tf.function(input_signature=[tf.TensorSpec([n0, d], tf.float64),
            tf.TensorSpec([2*d], tf.float64)], jit_compile=True)
        def density(points, theta):
            return tf.exp(-.5*(tf.cast(d, tf.float64)*
                tf.math.log(tf.constant(6.283185307179586, tf.float64))+
                tf.reduce_sum(theta[d:])+tf.reduce_sum(
                    tf.square(points-theta[:d])/tf.exp(theta[d:]), axis=1)))

        kernels.append(density)
        pattern = tf.cast(2*(tf.range(n0) % 2)-1, tf.float64)
        pattern -= tf.reduce_mean(pattern)
        sensitivities = []
        baseline = None
        baseline_density = None
        for scale in [0., 1e-14, -1e-14, 1e-12, -1e-12]:
            target = ly + tf.constant(scale, tf.float64)*pattern
            fit = kernel(x, target, tf.constant(False))
            save("sensitivity-"+str(scale)+".json", fit)
            p = density(x, fit["theta"])
            if baseline is None:
                baseline, baseline_density = fit, p
            base_residual = p-fit["amplitude"]*tf.exp(ly-tf.reduce_max(ly))
            sensitivities.append(dict(perturbation=scale,
                realized_target_error=maximum(target, ly),
                initializer_change=maximum(fit["initial"], baseline["initial"]),
                theta_change=maximum(fit["theta"], baseline["theta"]),
                theta_error_from_R=maximum(fit["theta"], rtheta),
                density_relative_max_change=number(tf.reduce_max(tf.abs(p-baseline_density))/
                                                   tf.reduce_max(baseline_density)),
                original_target_loss=number(tf.reduce_sum(tf.square(base_residual))),
                loss=number(fit["loss"]), gradient_max=number(tf.reduce_max(tf.abs(fit["gradient"]))),
                code=plain(fit["code"]), iterations=plain(fit["iterations"]),
                valid=plain(fit["valid"])))
            save("sensitivity-summary.json", sensitivities)

        class EntropyDivergence(RuntimeError):
            pass

        def entropy(index, n):
            if index > len(call_rows) or n != int(call_rows[index-1]["N"]):
                raise EntropyDivergence("Saved entropy mismatch: call=%s N=%s" % (index, n))
            call = folder/("call"+str(index))
            return (read(call, "noise", [horizon, n, d]),
                    read(call, "mixture", [horizon, n]),
                    read(call, "ancestor_uniforms", [horizon, n]))

        def observer(kind, index, payload):
            if kind == "filter":
                v, n = payload["value"], payload["particles"]
                ref = folder/("call"+str(index))
                cloud_ref = read(ref, "clouds", [horizon, n, d])
                prefix_ref = read(ref, "prefix", [horizon])
                summary = dict(kind=kind, index=index, particles=n,
                    log_likelihood=number(v[0]),
                    prefix_max_error=maximum(v[-1], prefix_ref),
                    final_prefix_error=number(v[-1][-1]-prefix_ref[-1]),
                    cloud_max_error=maximum(v[1], cloud_ref),
                    finite=bool(tf.reduce_all(tf.math.is_finite(v[1])).numpy()) and
                           bool(tf.reduce_all(tf.math.is_finite(v[2])).numpy()),
                    minimum_ess=number(tf.reduce_min(v[4])),
                    resampling_count=int(tf.reduce_sum(tf.cast(v[5],tf.int32)).numpy()))
                save("actual-filter"+str(index)+".json", dict(**summary, prefix=v[-1],
                    ess=v[4],resampled=v[5],
                    clouds_sha256=hashlib.sha256(tf.io.serialize_tensor(v[1]).numpy()).hexdigest()))
                kernels.append(make_paper_apf_value_kernel(d,d,n,horizon,"float64",True))
            else:
                ref = folder/("backward"+str(index))
                if not ref.is_dir():
                    raise EntropyDivergence("Adaptive backward count exceeds reference")
                summary = dict(kind=kind,index=index,valid=plain(payload["valid"]),
                    centers_max_error=maximum(payload["centers"],read(ref,"centers",[horizon,d])),
                    covariance_max_error=maximum(payload["covariances"],read(ref,"covariances",[horizon,d,d])),
                    floor_max_error=maximum(payload["floors"],read(ref,"floors",[horizon])),
                    cap100_count=int(tf.reduce_sum(tf.cast(payload["codes"]==1,tf.int32)).numpy()))
                save("actual-backward"+str(index)+".json",dict(**summary,
                    centers=payload["centers"],covariances=payload["covariances"],
                    floors=payload["floors"],codes=payload["codes"],summary=payload["summary"]))
            if kind=="filter" and not summary["finite"]:
                save("invalid-consumer.json", summary)
                raise RuntimeError("Non-finite filter cloud or weights")
            summaries.append(summary)
            save("actual-consumer-summary.json",summaries)
            print(json.dumps(summary),flush=True)

        try:
            result = run_finite_iapf(A,C,tf.zeros([d],tf.float64),eye,eye,R,obs,entropy,
                n0=n0,max_particles=max_n,max_iterations=max_it,
                jit_compile=True,observer=observer)
            for n,kind in result["signatures"]:
                kernels.append(make_paper_apf_value_kernel(d,d,n,horizon,"float64",True)
                    if kind=="filter" else make_finite_backward_kernel(d,d,n,horizon,"float64",True))
            kernels.append(make_controller_decision_kernel(max_it,5,"float64",True))
            final = result["final"]
            final_value = number(final[0]) if final is not None else None
            result["final"] = dict(log_likelihood=final[0],prefix=final[-1]) if final is not None else None
            save("actual-controller.json",result)
            status = result["status"]
            impact = dict(actual_status=status,
                reference_status=reference["status"],
                final_log_likelihood=final_value,
                reference_final_log_likelihood=float(reference["final_log_likelihood"]),
                reference_kalman=float(reference["kalman"]),
                final_minus_reference=final_value-float(reference["final_log_likelihood"]) if final_value is not None else None,
                final_minus_kalman=final_value-float(reference["kalman"]) if final_value is not None else None,
                controller_matches=all(result[k]==int(reference[k]) for k in
                    ["filter_calls","backward_calls","stop_iteration","final_particles"]),
                filter_calls=result["filter_calls"],backward_calls=result["backward_calls"],
                stop_iteration=result["stop_iteration"],final_particles=result["final_particles"])
        except EntropyDivergence as error:
            impact = dict(actual_status="entropy_shape_divergence",error=str(error))
    trace_counts = [k.experimental_get_tracing_count() for k in set(kernels)]
    save("impact.json",impact)
    decision = dict(status="PASS" if all(n==1 for n in trace_counts) else "FAIL",
        meaning="DIAGNOSTIC_COMPLETED_NOT_PARITY_PROMOTION",
        parity_status="FAIL_PRESERVED_FROM_ATTEMPT27",
        consumer=impact,sensitivity=sensitivities,trace_counts=trace_counts,
        seconds=time.monotonic()-start,
        host_peak_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        gpu_allocator=tf.config.experimental.get_memory_info("GPU:0"),
        nonclaims=["no stochastic accuracy or superiority","no complete replication","no default or HMC promotion"])
    save("decision.json",decision)
    with (args.output/"decision.csv").open("w") as stream:
        writer=csv.DictWriter(stream,fieldnames=["status","scope"])
        writer.writeheader()
        writer.writerow(dict(status=decision["status"],scope="SENSITIVITY_AND_IMPACT_DIAGNOSTIC_ONLY"))
    print(json.dumps(dict(impact=impact,trace_counts=trace_counts)),flush=True)
    return 0 if decision["status"]=="PASS" else 2


if __name__=="__main__":
    raise SystemExit(main())
