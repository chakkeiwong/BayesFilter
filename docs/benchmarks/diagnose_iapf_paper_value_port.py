"""Reference/parity diagnostic; CPU and graph modes are explicit exceptions."""
import argparse
import csv
import hashlib
import json
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
    if not gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    os.environ["BAYESFILTER_PRELOAD_CUSTOM_OP"] = "0"
    if gpu:
        probe = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=True)
        (args.output / "nvidia-smi.log").write_text(probe.stdout)
    sys.path.insert(0, str(ROOT))
    import tensorflow as tf
    memory_policy = None
    if gpu:
        from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
        memory_policy = configure_tensorflow_gpu_memory_growth(tf)
        if not tf.config.list_logical_devices("GPU"):
            raise RuntimeError("Trusted TensorFlow probe found no GPU")
    tf.config.experimental.enable_tensor_float_32_execution(False)
    from bayesfilter.score_study.iapf_paper_tf import make_paper_apf_value_kernel
    source = ROOT / "bayesfilter/score_study/iapf_paper_tf.py"
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    cases = list(csv.DictReader((args.fixture / "index.csv").open()))
    reports = []
    (args.output/"preflight-runtime.json").write_text(json.dumps(dict(
        tensorflow=tf.__version__, mode=args.mode, jit_compile=args.mode=="gpu-xla",
        dtype="float64", tf32=False, gpu_memory_policy=memory_policy,
        cpu_only=not gpu, gpu_intentionally_hidden=not gpu,
        source_sha256=source_hash, device_trust="escalated_execution" if gpu else
        "deliberate_cpu_reference_exception"),indent=2)+"\n")
    device = "/GPU:0" if gpu else "/CPU:0"
    if gpu:
        tf.config.experimental.reset_memory_stats("GPU:0")
    for case in cases:
        folder = args.fixture / case["name"]
        d, o, n, horizon = (int(case[k]) for k in ["d", "o", "N", "T"])
        def read(name, shape, dtype=tf.float64):
            with (folder / (name + ".csv")).open() as stream:
                flat = [float(x) for row in csv.reader(stream) for x in row]
            value = tf.constant(flat, tf.float64)
            return tf.cast(tf.reshape(value, shape), dtype)
        with tf.device(device):
            inputs = [
                read("A", [d, d]), read("C", [o, d]), read("initial_mean", [d]),
                read("P", [d, d]), read("Q", [d, d]), read("R", [o, o]),
                read("observations", [horizon, o]), read("centers", [horizon, d]),
                read("covariances", [horizon, d, d]), read("floors", [horizon]),
                read("constants", [horizon], tf.bool), read("noise", [horizon, n, d]),
                read("mixture", [horizon, n]), read("ancestor_uniforms", [horizon, n]),
                tf.constant(float(case["kappa"]), tf.float64)]
            kernel = make_paper_apf_value_kernel(d, o, n, horizon, "float64", args.mode == "gpu-xla")
            start = time.monotonic()
            actual = kernel(*inputs)
            value = float(actual[0].numpy())
            first_seconds = time.monotonic() - start
            start = time.monotonic()
            repeated = kernel(*inputs)
            float(repeated[0].numpy())
            repeat_seconds = time.monotonic() - start
            checks = []
            expected = [tf.constant(float(case["log_likelihood"]), tf.float64),
                read("clouds", [horizon, n, d]), read("weights", [horizon, n]),
                read("ancestors", [horizon, n], tf.int32), read("ess", [horizon]),
                read("resampled", [horizon], tf.bool), read("prefix", [horizon])]
            for name, got, want in zip(
                    ["log_likelihood", "clouds", "weights", "ancestors", "ess", "resampled", "prefix"],
                    actual, expected):
                if got.dtype in (tf.bool, tf.int32):
                    passed = bool(tf.reduce_all(got == want).numpy())
                    error = None
                else:
                    diff = tf.abs(got - want)
                    passed = bool(tf.reduce_all(tf.math.is_finite(got) &
                        (diff <= 1e-8 * (1 + tf.abs(want)))).numpy())
                    error = float(tf.reduce_max(diff).numpy())
                checks.append(dict(name=name, passed=passed, maximum_absolute_error=error))
            if case["exact"] == "TRUE":
                checks.append(dict(name="exact_twist_kalman",
                    passed=abs(value-float(case["kalman"])) <= 1e-8*(1+abs(float(case["kalman"])))))
            checks.append(dict(name="one_trace", passed=kernel.experimental_get_tracing_count() == 1))
            if gpu:
                checks.append(dict(name="gpu_output", passed="GPU:0" in actual[0].device))
            report = dict(case=case["name"], d=d, N=n, T=horizon, checks=checks,
                first_call_including_trace_seconds=first_seconds,
                repeat_seconds=repeat_seconds, trace_count=kernel.experimental_get_tracing_count(),
                output_device=actual[0].device)
            reports.append(report)
            (args.output/"comparisons.json").write_text(json.dumps(reports, indent=2)+"\n")
            if not all(c["passed"] for c in checks):
                raise RuntimeError("Paired value parity failed: "+case["name"])

    legacy = None
    if not gpu:
        from unittest.mock import patch
        from bayesfilter.score_study import fitted_twist_tf as old, iapf_adapter
        from bayesfilter.score_study.gaussian_tf import parameterized_model
        captured = {}
        class ReachedFactory(Exception):
            pass
        def capture(*factory_args, **kwargs):
            captured.update(args=list(factory_args), kwargs=kwargs)
            raise ReachedFactory()
        controls = dict(k=5, tau=.5, max_iterations=8, max_particles=4,
            mean_bound=10., sd_lower=.01, sd_upper=10., max_fit_steps=10,
            max_backtracks=10, fit_tolerance=1e-6, floor_ratio=.001,
            fit_theta=[.42, 0., 0., 0., 1., 0.])
        settings = dict(dimension=1, observation_dimension=1, particles=4, horizon=2,
                        dtype="float64", jit_compile=False)
        theta = tf.constant(controls["fit_theta"], tf.float64)
        with patch.object(old, "make_fitted_twist_kernel", side_effect=capture):
            try:
                iapf_adapter.execute_iapf(dict(role="diagnostic", model="linear_gaussian",
                    iapf=controls), settings, theta, tf.zeros([2,1],tf.float64), None)
            except ReachedFactory:
                pass
        if not captured:
            raise RuntimeError("Legacy endpoint wiring was not established")
        model = parameterized_model(theta, 1, 1)
        A, m, P = model[0], model[4], model[6]
        initial_noise = tf.constant([[-1.],[-.25],[.25],[1.]], tf.float64)
        u = tf.constant([[.125,.375,.625,.875],[.875,.625,.375,.125],
                         [.125,.375,.625,.875]], tf.float64)
        old_kernel = old.make_fitted_twist_kernel(1,1,4,2,"float64",False,constant_twist=True)
        legacy_inputs = (theta,tf.zeros([2,1],tf.float64),initial_noise,
            tf.zeros([2,4,1],tf.float64),u,tf.zeros([2,4],tf.float64),
            tf.zeros([2,1],tf.float64),tf.ones([2,1,1],tf.float64),tf.zeros([2],tf.float64))
        graph_result = old_kernel(*legacy_inputs)
        legacy_graph_shapes = [x.shape.as_list() for x in graph_result]
        legacy_graph_valid = legacy_graph_shapes == [[], [6], [2,4,1]]
        # Same callable body, explicit eager diagnostic; never a runtime fallback.
        result = old_kernel.python_function(*legacy_inputs)
        if legacy_graph_valid:
            for graph_value, body_value in zip(graph_result,result):
                if not bool(tf.reduce_all(tf.abs(graph_value-body_value)<=1e-10).numpy()):
                    legacy_graph_valid = False
        direct = m + tf.linalg.matmul(initial_noise, tf.linalg.cholesky(P), transpose_b=True)
        first = tf.linalg.matmul(direct,A,transpose_b=True)
        second = tf.linalg.matmul(tf.reverse(first,[0]),A,transpose_b=True)
        legacy_error = max(float(tf.reduce_max(tf.abs(result[2][0]-first)).numpy()),
                           float(tf.reduce_max(tf.abs(result[2][1]-second)).numpy()))
        initial_mismatch = float(tf.reduce_max(tf.abs(result[2][0]-direct)).numpy())
        resampling_mismatch = float(tf.reduce_max(tf.abs(result[2][1]-
            tf.linalg.matmul(first,A,transpose_b=True))).numpy())
        if legacy_error > 1e-10 or min(initial_mismatch,resampling_mismatch) <= .01:
            raise RuntimeError("Legacy scope mismatch did not reproduce as predicted")
        legacy = dict(endpoint="iapf_adapter.execute_iapf", captured_factory=captured,
            graph_shapes=legacy_graph_shapes, graph_valid=legacy_graph_valid,
            graph_status="VALID_FOR_FIXTURE" if legacy_graph_valid else "INVALID_OUTPUT_REQUIRES_LOCALIZATION",
            numerical_equations_lane="same callable body; eager diagnostic only",
            runtime_fallback=False,
            matches_legacy_equations_max_error=legacy_error,
            initial_law_mismatch=initial_mismatch, compulsory_resampling_mismatch=resampling_mismatch,
            paper_target_match=False,
            conclusion="Wrong for this paper-target claim; distinct score-study model and semantics")
        (args.output/"legacy-consumer-audit.json").write_text(json.dumps(legacy,indent=2)+"\n")
    if hashlib.sha256(source.read_bytes()).hexdigest() != source_hash:
        raise RuntimeError("Source changed during parity run")
    runtime = dict(tensorflow=tf.__version__, python=sys.version, mode=args.mode,
        jit_compile=args.mode=="gpu-xla", dtype="float64", tf32=False,
        cpu_only=not gpu, gpu_intentionally_hidden=not gpu,
        gpu_memory_policy=memory_policy, source_sha256=source_hash,
        fixture_index_sha256=hashlib.sha256((args.fixture/"index.csv").read_bytes()).hexdigest(),
        host_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        gpu_allocator=tf.config.experimental.get_memory_info("GPU:0") if gpu else None,
        device_trust="escalated_execution" if gpu else "deliberate_cpu_reference_exception")
    (args.output/"runtime.json").write_text(json.dumps(runtime,indent=2)+"\n")
    with (args.output/"decision.csv").open("w") as stream:
        writer=csv.DictWriter(stream,fieldnames=["status","cases","scope","replication"])
        writer.writeheader()
        writer.writerow(dict(status="PASS",cases=len(cases),scope="PAPER_TARGET_FROZEN_TWIST_VALUE_PARITY",
                             replication="INCOMPLETE"))
    print(json.dumps(dict(status="PASS",mode=args.mode,cases=len(cases),legacy_paper_match=legacy["paper_target_match"] if legacy else None)))

if __name__=="__main__":
    main()
