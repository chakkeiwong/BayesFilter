"""GPU/XLA parity of the actual author circular KDM density/loss consumers."""
import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["BAYESFILTER_PRELOAD_CUSTOM_OP"] = "0"
import argparse
import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path("/home/chakwong/BayesFilter")
PLAN = "docs/plans/controlled-smc-kdm-circular-consumer-20261006.md"
RESULT = "docs/plans/controlled-smc-kdm-circular-consumer-result-20261006.md"
sys.path.insert(0, str(ROOT))


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    sources = output / "sources"
    sources.mkdir()
    started = time.monotonic()
    paths = [Path(__file__), ROOT / "docs/benchmarks/reference_kdm_circular.py",
             ROOT / "bayesfilter/replication/kdm_normal_von_mises_tf.py", ROOT / PLAN]
    hashes = {}
    for path in paths:
        target = sources / path.name
        shutil.copyfile(path, target)
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = dict(schema="kdm_circular_consumer_manifest_v1",
                    command=[sys.executable] + sys.argv,
                    git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    branch=subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
                    environment=sys.executable, start_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    plan_file=PLAN, result_file=RESULT, source_hashes=hashes,
                    data_version="constructed deterministic mechanics fixture, eight cases",
                    seed_policy="20261006+regime index; independent reset per dtype",
                    cpu_gpu_status="CPU-only author reference; TF GPU/XLA comparator",
                    gpu_trust_basis="escalated_local_gpu_access",
                    attempt_timeout_seconds=180, replication="INCOMPLETE")
    write(output / "manifest.json", manifest)
    exit_code = 1
    try:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "-1"
        reference_lib = "/home/chakwong/anaconda3/envs/tf-gpu2/lib"
        env["LD_LIBRARY_PATH"] = reference_lib + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
        command = [str(ROOT / ".localresources/envs/mdpf-reference-20260924/bin/python"),
                   "-B", str(sources / "reference_kdm_circular.py"), "--output", str(output / "author.json")]
        with (output / "author.log").open("w") as log:
            subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=120, check=True)
        manifest["reference_command"] = command
        manifest["reference_environment"] = {key: env[key] for key in ("CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH")}
        reference = json.loads((output / "author.json").read_text())
        import tensorflow as tf
        from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
        memory = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
        tf.config.experimental.enable_tensor_float_32_execution(False)
        tf.config.set_soft_device_placement(False)
        spec = importlib.util.spec_from_file_location("circular_snapshot", sources / "kdm_normal_von_mises_tf.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        checks, outputs, traces = [], [], {}
        kernels = {}
        def boolean(name, value):
            checks.append(dict(name=name, pass_check=bool(value)))
        def compare(name, actual, expected, dtype):
            if isinstance(actual, (tuple, list)):
                for index, (left, right) in enumerate(zip(actual, expected)):
                    compare(name + "/" + str(index), left, right, dtype)
                boolean(name + "/group_count", len(actual) == len(expected))
                return
            actual = tf.convert_to_tensor(actual, dtype)
            expected = tf.convert_to_tensor(expected, dtype)
            tolerance = 128 * 2**-23 if dtype == tf.float32 else 1e-9
            bound = tf.constant(tolerance, dtype) * (1 + tf.abs(expected))
            finite = tf.reduce_all(tf.math.is_finite(actual)) & tf.reduce_all(tf.math.is_finite(expected))
            difference = tf.abs(actual - expected)
            checks.append(dict(name=name, pass_check=bool((finite & tf.reduce_all(difference <= bound)).numpy()),
                               max_absolute_error=float(tf.reduce_max(difference).numpy()),
                               max_scaled_error=float(tf.reduce_max(difference / bound).numpy()),
                               tolerance=tolerance))
        def host(value):
            if isinstance(value, dict):
                return {key: host(v) for key, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [host(v) for v in value]
            return value.numpy().tolist()
        with tf.device("/GPU:0"):
            for case in reference["cases"]:
                dtype = getattr(tf, case["dtype"])
                if case["dtype"] not in kernels:
                    kernels[case["dtype"]] = module.NormalVonMisesKDM(
                        batch_size=2, particles=25, circular_dimensions=[False, False, True], dtype=dtype)
                kernel = kernels[case["dtype"]]
                values = {key: tf.constant(value, dtype) for key, value in case["inputs"].items()}
                l, w, b, s = [values[k] for k in ("locations", "weights", "bandwidth", "samples")]
                resampling = kernel.resampling_score(l, w, b, s)
                posterior = kernel.posterior_score(l, w, b, values["states"])
                consumer = kernel.consumer_score(l, w, b, s, values["posterior_locations"],
                                                 values["model_weights"], values["posterior_bandwidth"],
                                                 values["states"])
                for category, result in (("resampling", resampling), ("posterior", posterior), ("consumer", consumer)):
                    for key, expected in case[category].items():
                        if key == "sample_gradient_stopped":
                            boolean(case["id"] + "/" + category + "/stopped", bool(result[key].numpy()) and expected)
                        else:
                            compare(case["id"] + "/" + category + "/" + key, result[key], expected, dtype)
                    boolean(case["id"] + "/" + category + "/valid", result["valid"].numpy())
                    device = result["loss"].device if "loss" in result else result["log_density"].device
                    boolean(case["id"] + "/" + category + "/gpu", "GPU:0" in device)
                boolean(case["id"] + "/consumer/stopped", consumer["sample_gradient_stopped"].numpy())
                shifted = l + tf.constant([0., 0., 2 * 3.141592653589793], dtype)
                periodic = kernel.posterior_score(shifted, w, b, values["states"])
                compare(case["id"] + "/angular_periodicity", periodic["nll"], posterior["nll"], dtype)
                if case["regime"] == "ordinary":
                    for name, bad_weight, bad_band in (
                        ("zero_mass", tf.zeros_like(w), b),
                        ("negative_weight", -w, b),
                        ("negative_bandwidth", w, -b)):
                        invalid = kernel.posterior_score(l, bad_weight, bad_band, values["states"])
                        boolean(case["id"] + "/reject/" + name, not bool(invalid["valid"].numpy()))
                    repeated = kernel.consumer_score(l, w, b, s, values["posterior_locations"],
                                                     values["model_weights"], values["posterior_bandwidth"], values["states"])
                    compare(case["id"] + "/repeat", repeated["loss"], consumer["loss"], dtype)
                outputs.append(dict(id=case["id"], resampling=host(resampling),
                                    posterior=host(posterior), consumer=host(consumer)))
        for dtype_name, kernel in kernels.items():
            counts = {name: getattr(kernel, name).experimental_get_tracing_count()
                      for name in ("resampling_score", "posterior_score", "consumer_score")}
            traces[dtype_name] = counts
            boolean(dtype_name + "/stable_traces", all(value == 1 for value in counts.values()))
        substitution = {case["id"]: case["wrong_density_substitution_max_error"] for case in reference["cases"]}
        boolean("posterior_density_substitution_detected",
                all(value > 1. for key, value in substitution.items() if key.endswith("tiny_weights")))
        summary = dict(
            status="PASS" if all(check["pass_check"] for check in checks) else "FAIL",
            scope="MATCHED_AUTHOR_CIRCULAR_KDM_RESAMPLING_WEIGHT_NLL_CONSUMERS",
            replication="INCOMPLETE", ranking="NOT_TESTED", checks=checks, case_count=len(outputs),
            scientific_verdict="PENDING", trace_counts=traces, tensorflow=tf.__version__,
            gpu_memory_policy=memory, gpu_allocator=tf.config.experimental.get_memory_info("GPU:0"),
            dtype=["float32", "float64"], jit_compile=True, tf32=False, batch_size=2, particles=25,
            numpy_runtime=False, pfor=False, sample_wise_loop=False,
            fixed_horner_loop_max_iterations=8, wrong_density_substitution_errors=substitution,
            limitations=["fixed author samples; TF circular random sampling not tested",
                         "constructed positive weight model and downstream clouds; neural models not ported",
                         "not an optimizer or training comparison",
                         "no autonomous filtering, no-worse quality or full replication claim"])
        summary["scientific_verdict"] = ("CIRCULAR_DENSITY_AND_NLL_CONSUMERS_MATCH_SOURCE"
                                        if summary["status"] == "PASS" else "CIRCULAR_CONSUMER_MISMATCH")
        write(output / "outputs.json", outputs)
        write(output / "summary.json", summary)
        manifest["gpu_memory_policy"] = memory
        manifest["gpu_allocator"] = summary["gpu_allocator"]
        manifest["tensorflow"] = tf.__version__
        exit_code = 0 if summary["status"] == "PASS" else 1
        print(json.dumps(dict(status=summary["status"], checks=len(checks),
                              failed=[c["name"] for c in checks if not c["pass_check"]])))
    except Exception as exc:
        traceback.print_exc()
        write(output / "summary.json", dict(status="ERROR", replication="INCOMPLETE",
                                            scope="CIRCULAR_CONSUMER_DIAGNOSTIC",
                                            error=repr(exc)))
    finally:
        manifest["wall_seconds"] = time.monotonic() - started
        manifest["returncode"] = exit_code
        manifest["output_artifacts"] = {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in output.iterdir() if p.is_file() and p.name != "manifest.json"}
        write(output / "manifest.json", manifest)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
