"""Author-code parity for a batched KDM sequential sampling-to-SGD consumer."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["BAYESFILTER_PRELOAD_CUSTOM_OP"] = "0"
ROOT = Path("/home/chakwong/BayesFilter")
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    output = Path(parser.parse_args().output)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "-1"
    reference_lib = "/home/chakwong/anaconda3/envs/tf-gpu2/lib"
    env["LD_LIBRARY_PATH"] = reference_lib + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    command = [str(ROOT / ".localresources/envs/mdpf-reference-20260924/bin/python"),
               str(output / "sources/reference_kdm_sequential.py"),
               "--output", str(output / "author-fixture.json")]
    subprocess.run(command, env=env, check=True)
    reference = json.loads((output / "author-fixture.json").read_text())
    import tensorflow as tf
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
    memory = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    tf.config.experimental.enable_tensor_float_32_execution(False)
    for name, filename in [
        ("bayesfilter.replication.kdm_iwsg_tf", "kdm_iwsg_tf.py"),
        ("kdm_sequential_snapshot", "kdm_sequential_tf.py")]:
        spec = importlib.util.spec_from_file_location(name, output / "sources" / filename)
        loaded = importlib.util.module_from_spec(spec)
        sys.modules[name] = loaded
        spec.loader.exec_module(loaded)
    module = sys.modules["kdm_sequential_snapshot"]
    checks, modes, failure, device = [], {}, None, None

    def check(name, actual, expected, atol=1e-8, rtol=1e-8):
        actual = tf.convert_to_tensor(actual, dtype=tf.float64)
        expected = tf.convert_to_tensor(expected, dtype=tf.float64)
        error = tf.abs(actual - expected)
        scaled = error / (atol + rtol * tf.abs(expected))
        passed = bool(tf.reduce_all(tf.math.is_finite(error) & (scaled <= 1)).numpy())
        checks.append(dict(name=name, passed=passed,
                           max_abs=float(tf.reduce_max(error).numpy()),
                           max_tolerance_units=float(tf.reduce_max(scaled).numpy())))
        if not passed:
            raise RuntimeError("Parity check failed: " + name)

    def flag(name, value):
        passed = bool(value)
        checks.append(dict(name=name, passed=passed))
        if not passed:
            raise RuntimeError("Invariant failed: " + name)

    settings = reference["settings"]
    configuration = {key: settings[key] for key in (
        "batch", "horizon", "particles", "sigma", "gamma",
        "posterior_bandwidth", "resampling_bandwidth", "density_epsilon", "injection_epsilon")}
    try:
        check("host_fp64_roundtrip", [1.0000000001],
              tf.constant([1.0000000001], tf.float64), 1e-12, 1e-12)
        for mode in ("iwsg", "truncated"):
            fixture = reference["results"][mode]
            kernel = module.BimodalLinearKDM(**configuration, gradient_mode=mode)
            inputs = [tf.constant(fixture[key], tf.float64) for key in (
                "theta", "initial_particles", "actions", "observations", "terminal_states",
                "uniforms", "kernel_noise", "dynamics_noise")]
            result = kernel.evaluate(*inputs)
            device = result["loss"].device
            flag(mode + ".gpu", "GPU:0" in device)
            flag(mode + ".valid", bool(result["valid"].numpy()))
            for key in ("resampled", "particles", "weights", "terminal_logdensity", "loss"):
                check(mode + "." + key, result[key], fixture[key])
            flag(mode + ".ancestry", bool(tf.reduce_all(
                result["ancestors"] == tf.constant(fixture["ancestors"], tf.int32)).numpy()))
            check(mode + ".weight_normalization", tf.reduce_sum(result["weights"], axis=2),
                  tf.ones([5, 4], tf.float64), 1e-12, 1e-12)
            gradient = kernel.value_and_gradient(*inputs)
            flag(mode + ".gradient_valid", bool(gradient["valid"].numpy()))
            check(mode + ".gradient", gradient["gradient"], fixture["gradient"])
            step = kernel.step(*inputs, tf.constant(settings["learning_rate"], tf.float64))
            flag(mode + ".update_valid", bool(step["valid"].numpy()))
            check(mode + ".update", step["next_theta"], fixture["next_theta"])
            check(mode + ".update_gradient", step["gradient"], fixture["gradient"])
            check(mode + ".update_loss", step["loss"], fixture["loss"])
            kernel.evaluate(*inputs)
            kernel.value_and_gradient(*inputs)
            kernel.step(*inputs, tf.constant(settings["learning_rate"], tf.float64))
            traces = {name: getattr(kernel, name).experimental_get_tracing_count()
                      for name in ("evaluate", "value_and_gradient", "step")}
            flag(mode + ".one_trace_per_signature", all(value == 1 for value in traces.values()))
            invalid = kernel.step(*inputs, tf.constant(-.001, tf.float64))
            flag(mode + ".negative_learning_rate_rejected", not bool(invalid["valid"].numpy()))
            modes[mode] = dict(loss=float(result["loss"].numpy()),
                               gradient=gradient["gradient"].numpy().tolist(),
                               next_theta=step["next_theta"].numpy().tolist(),
                               terminal_logdensity=result["terminal_logdensity"].numpy().tolist(),
                               traces=traces,
                               minimum_raw_weight_sum=float(result["minimum_raw_weight_sum"].numpy()))
        check("iwsg_truncated_forward_equal", modes["iwsg"]["terminal_logdensity"],
              modes["truncated"]["terminal_logdensity"])
        gradient_difference = tf.linalg.norm(tf.constant(modes["iwsg"]["gradient"], tf.float64) -
                                              tf.constant(modes["truncated"]["gradient"], tf.float64))
        flag("iwsg_changes_sequential_gradient", float(gradient_difference.numpy()) > 1e-7)
        try:
            module.BimodalLinearKDM(**dict(configuration, gamma=0), gradient_mode="iwsg")
            flag("zero_observation_scale_rejected", False)
        except ValueError:
            flag("zero_observation_scale_rejected", True)
    except Exception as exc:
        failure = repr(exc)
    summary = dict(
        status="PASS" if failure is None and all(row["passed"] for row in checks) else "FAIL",
        scope="CONSTRUCTED_ANALYTIC_KDM_SEQUENTIAL_SOURCE_CONSUMER_PARITY",
        replication="INCOMPLETE", ranking="NOT_TESTED", checks=checks, failure=failure,
        settings=settings, modes=modes, dtype="float64", jit_compile=True, tf32=False,
        batch_size=4, scalar_sample_loop=False, pfor=False, numpy_runtime=False,
        tensorflow=tf.__version__, device=device, gpu_memory_policy=memory,
        gpu_allocator=tf.config.experimental.get_memory_info("GPU:0"),
        peak_host_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        source_command=command,
        source_environment={key: env[key] for key in ("CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH")},
        source_sha256=reference["source_sha256"],
        call_chain="step -> shared value_and_gradient -> shared evaluate -> GaussianMixtureIWSG",
        limitations=["constructed prior/timing/coefficients, not recovered author training setup",
                     "one diagnostic SGD update, not convergence or training-quality evidence",
                     "author epsilon conventions, no unbiased-estimator claim",
                     "neural bearings training, normalized comparators and independent repeats remain open"])
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: summary[key] for key in ("status", "scope", "failure")}))
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
