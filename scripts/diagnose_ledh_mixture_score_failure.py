"""Exact-seed validity localization; outputs are not admitted score evidence."""

import argparse
import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time


def serializable(value):
    if isinstance(value, list):
        return [serializable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--inputs", type=Path)
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    root = Path(__file__).resolve().parents[1]
    gpu = arguments.device == "gpu"
    if not gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    manifest = {
        "status": "started", "command": [sys.executable, *sys.argv],
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "plan": "docs/plans/ledh-guided-mixture-score-comparison-20261016.md",
        "device": arguments.device, "gpu_intentionally_hidden": not gpu,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "dtype": "float64", "tf32": False, "jit_compile": gpu,
        "seed": 100204, "data_version": "exact_failed_case_attempt01",
        "result": str((arguments.output / "result.json").resolve()),
        "claim": "failure localization only", "environment": sys.executable,
    }
    def save(name, value):
        (arguments.output / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    try:
        import tensorflow as tf
        if gpu:
            from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
            manifest["gpu_memory_policy"] = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
            manifest["devices"] = [tf.config.experimental.get_device_details(device)
                                      for device in tf.config.list_physical_devices("GPU")]
        tf.config.experimental.enable_tensor_float_32_execution(False)
        manifest["tensorflow"] = tf.__version__
        from bayesfilter.highdim.covariance_proposal_tf import ProposalControls, TRACE_FIELDS, MOMENT_VALIDITY_POLICY_ID
        import bayesfilter.highdim.covariance_proposal_moments_tf
        from bayesfilter.highdim.sqmc_lgssm_tf import LGSSMSpec
        from bayesfilter.score_study.guided_mixture_diagnostic_tf import make_diagnostic_programs
        controls = ProposalControls(reset_steps=5, correction_steps=2)
        manifest["moment_validity_policy"] = MOMENT_VALIDITY_POLICY_ID
        manifest["controls"] = dataclasses.asdict(controls)
        source_paths = {Path(__file__).resolve()}
        source_paths.update(Path(module.__file__).resolve() for name, module in list(sys.modules.items())
                            if name.startswith("bayesfilter") and getattr(module, "__file__", None))
        manifest["source_sha256"] = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                                      for path in sorted(source_paths) if path.is_relative_to(root)}
        save("manifest.json", manifest)
        dtype = tf.float64
        theta = tf.constant([.9, .9, .3, .5], dtype)
        if arguments.inputs:
            saved = json.loads(arguments.inputs.read_text())
            data = [tf.constant(saved[name], dtype) for name in ("observations", "initial", "normals", "uniforms")]
            manifest["input_sha256"] = hashlib.sha256(arguments.inputs.read_bytes()).hexdigest()
        else:
            observations = LGSSMSpec("diagonal_ar", 2).simulate(theta, 5, 100204, jit_compile=gpu)
            variance = 1.
            for _ in range(3):
                variance = .9 ** 2 * variance + .09
            observations += tf.scatter_nd([[2, 0]], tf.constant([3. * math.sqrt(variance + .25)], dtype), [5, 2])
            data = [observations,
                    tf.random.stateless_normal([32, 2], [100204, 2], dtype=dtype),
                    tf.random.stateless_normal([5, 32, 2], [100204, 3], dtype=dtype),
                    tf.random.stateless_uniform([5, 32, 2], [100204, 4], dtype=dtype)]
            save("inputs.json", dict(zip(("observations", "initial", "normals", "uniforms"),
                                         [value.numpy().tolist() for value in data])))
            manifest["input_sha256"] = hashlib.sha256((arguments.output / "inputs.json").read_bytes()).hexdigest()
        evaluate, _ = make_diagnostic_programs(32, 5, controls, jit_compile=gpu)
        result = evaluate(theta, *data, tf.constant([1., 0., 0.], dtype),
                          tf.zeros([5, 2, 2], dtype), tf.zeros([5, 2], dtype))
        trace = result["direction_trace"].numpy().tolist()
        save("result.json", {"valid": bool(result["valid"]),
                             "direction_valid": result["direction_valid"].numpy().tolist(),
                             "score_finite": tf.math.is_finite(result["score"]).numpy().tolist(),
                             "trace_fields": TRACE_FIELDS, "direction_trace": serializable(trace),
                             "covariance_relative_error_limit": math.sqrt(2.220446049250313e-16),
                             "cap_limit": controls.correction_radius * (1 + 100 * 2.220446049250313e-16 * 32)})
        manifest["status"] = "diagnostic_complete"
        print(json.dumps({"valid": bool(result["valid"]), "wall_seconds": time.monotonic() - started}))
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        manifest["wall_seconds"] = time.monotonic() - started
        save("manifest.json", manifest)


if __name__ == "__main__":
    main()
