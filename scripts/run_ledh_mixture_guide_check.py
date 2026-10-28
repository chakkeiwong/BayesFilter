"""Bounded FP64 GPU/XLA integration diagnostic for the shared mixture filter.

This synthetic guide fixture tests finite-program derivatives and compilation.
It does not fit a KDM model or establish model-score accuracy or admission.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget-seconds", type=float, default=370.)
    args = parser.parse_args()
    started = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "4")
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    manifest = {
        "schema": "ledh_mixture_guide_integration_diagnostic_v1",
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=root, text=True).strip(),
        "command": [sys.executable, *sys.argv],
        "python": sys.executable, "platform": platform.platform(),
        "plan": "docs/plans/ledh-mixture-guide-integration-20261016.md",
        "result": str(args.output / "result.json"),
        "execution": "escalated GPU FP64 XLA; CPU graph parity reference",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "all"),
        "claim": "finite-program implementation diagnostic only",
        "data_version": "synthetic_fixed_gaussian_mixture_guide_v1",
        "seeds": [[861, 1], [861, 2], [861, 3]],
        "dtype": "float64", "tf32": False, "jit_compile": True,
        "epsilon": 1., "sinkhorn_iterations": 5,
        "particles": 24, "state_dimension": 2, "horizon": 3,
        "flow_steps": 16, "correction_steps": 2,
        "beta": [.2, .3, .5], "status": "started",
    }
    sources = [
        "bayesfilter/highdim/covariance_proposal_tf.py",
        "bayesfilter/highdim/covariance_proposal_moments_tf.py",
        "bayesfilter/highdim/ledh_backward_guide_tf.py",
        "bayesfilter/highdim/sqmc_lgssm_tf.py",
        "scripts/run_ledh_mixture_guide_check.py",
    ]
    manifest["source_sha256"] = {path: hashlib.sha256((root / path).read_bytes()).hexdigest() for path in sources}
    (args.output / "working-tree.diff").write_text(subprocess.check_output(["git", "diff", "--", *sources], cwd=root, text=True))
    write_json(args.output / "manifest.json", manifest)
    checks = []
    try:
        import tensorflow as tf
        from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth

        manifest["tensorflow"] = tf.__version__
        manifest["gpu_memory_policy"] = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
        tf.config.experimental.enable_tensor_float_32_execution(False)
        manifest["physical_gpus"] = [device.name for device in tf.config.list_physical_devices("GPU")]
        with tf.device("/GPU:0"):
            probe = tf.reduce_sum(tf.constant([1., 2.], tf.float64))
        if float(probe) != 3. or "GPU:0" not in probe.device:
            raise RuntimeError("GPU device probe failed")
        manifest["probe_device"] = probe.device

        from bayesfilter.highdim.covariance_proposal_tf import ProposalControls, TRACE_FIELDS, make_filter
        from bayesfilter.highdim.ledh_backward_guide_tf import FrozenMixtureBackwardGuide
        from bayesfilter.highdim.sqmc_lgssm_tf import LGSSMSpec

        dtype = tf.float64
        spec = LGSSMSpec("diagonal_ar", 2)
        theta = tf.constant([.8, -.3, .6, .8], dtype)
        observations = tf.constant([[.5, -.2], [.2, .8], [-.4, .3]], dtype)
        initial = tf.random.stateless_normal([24, 2], [861, 1], dtype=dtype)
        innovations = tf.random.stateless_normal([3, 24, 2], [861, 2], dtype=dtype)
        uniforms = tf.random.stateless_uniform([3, 24, 2], [861, 3], dtype=dtype)
        beta = tf.constant(manifest["beta"], dtype)
        guide = FrozenMixtureBackwardGuide(
            tf.constant([[.2, -.3], [-.4, .1], [0., 0.]], dtype),
            tf.constant([[[-.8, .4], [.7, -.6]], [[-.4, -.7], [.9, .5]], [[0., 0.], [0., 0.]]], dtype),
            tf.constant([[[[.8, .15], [.15, 1.1]], [[1.2, -.2], [-.2, .7]]]] * 3, dtype),
            tf.constant([True, True, False]),
        )
        controls = ProposalControls(proposal_policy="original_three_branch", reset_epsilon=1., reset_steps=5, correction_steps=2)
        compiled = make_filter(spec, 24, 3, controls, dtype, jit_compile=True, backward_guide=guide)
        reference = make_filter(spec, 24, 3, controls, dtype, jit_compile=False, backward_guide=guide)
        arguments = (observations, initial, innovations, uniforms, beta)
        manifest["theta"] = theta.numpy().tolist()
        manifest["observations"] = observations.numpy().tolist()
        manifest["guide"] = {name: getattr(guide, name).numpy().tolist() for name in ("logits", "centers", "covariances", "active")}

        def check_output(output):
            if not bool(output["valid"]) or int(output["steps_completed"]) != 3:
                raise RuntimeError("Shared mixture consumer rejected the fixture")
            if not bool(tf.reduce_all(tf.math.is_finite(output["trace"]))):
                raise RuntimeError("Nonfinite trace in an accepted fixture")

        for parameter in range(spec.parameter_count):
            if time.monotonic() - started >= args.budget_seconds:
                raise RuntimeError("Numerical execution budget exhausted")
            direction = tf.one_hot(parameter, spec.parameter_count, dtype=dtype)
            delta = tf.constant(2e-5, dtype)
            with tf.device("/GPU:0"):
                output = compiled(theta, direction, *arguments)
                plus = compiled(theta + delta * direction, tf.zeros_like(theta), *arguments)
                minus = compiled(theta - delta * direction, tf.zeros_like(theta), *arguments)
            for item in (output, plus, minus):
                check_output(item)
            if "GPU:0" not in output["value"].device:
                raise RuntimeError("Compiled consumer did not return a GPU value")
            with tf.device("/CPU:0"):
                comparison = reference(theta, direction, *arguments)
            check_output(comparison)
            finite_difference = (plus["value"] - minus["value"]) / (2 * delta)
            particle_difference = (plus["particles"] - minus["particles"]) / (2 * delta)
            score_error = float(tf.abs(output["score"] - finite_difference))
            particle_error = float(tf.reduce_max(tf.abs(output["particle_tangent"] - particle_difference)))
            parity = {key: float(tf.reduce_max(tf.abs(output[key] - comparison[key])))
                      for key in ("value", "score", "particles", "particle_tangent")}
            score_pass = score_error <= 2e-7 + 2e-6 * abs(float(finite_difference))
            particle_pass = bool(tf.reduce_all(tf.abs(output["particle_tangent"] - particle_difference) <= 1e-6 + 1e-5 * tf.abs(particle_difference)))
            parity_pass = all(error <= 2e-8 for error in parity.values())
            passed = score_pass and particle_pass and parity_pass
            checks.append({"parameter": parameter, "score": float(output["score"]),
                           "score_absolute_fd_error": score_error,
                           "particle_maximum_fd_error": particle_error,
                           "cpu_graph_maximum_absolute_errors": parity, "passed": passed,
                           "trace": output["trace"].numpy().tolist()})
            write_json(args.output / "result.json", {"checks": checks, "trace_fields": TRACE_FIELDS,
                       "complete": len(checks) == spec.parameter_count, "passed": all(row["passed"] for row in checks)})
            if not passed or not math.isfinite(score_error):
                raise RuntimeError("Derivative or CPU/GPU parity check failed")
        manifest["tracing_count"] = compiled.experimental_get_tracing_count()
        manifest["allocator"] = tf.config.experimental.get_memory_info("GPU:0")
        manifest["status"] = "passed"
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        manifest["wall_seconds"] = time.monotonic() - started
        write_json(args.output / "manifest.json", manifest)
    print(json.dumps({"status": manifest["status"], "wall_seconds": manifest["wall_seconds"], "checks": len(checks)}))


if __name__ == "__main__":
    main()
