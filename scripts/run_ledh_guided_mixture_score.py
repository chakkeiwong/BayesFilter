"""Bounded diagnostic comparison; no canonical LEDH or tuned-default claim."""

import argparse
import dataclasses
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
    parser.add_argument("--budget-seconds", type=float, default=840.)
    parser.add_argument("--guidance-factorial", action="store_true")
    parser.add_argument("--beta-max-steps", type=int, default=2000)
    arguments = parser.parse_args()
    if arguments.beta_max_steps < 1:
        parser.error("beta-max-steps must be positive")
    seed_base = 217000 if arguments.guidance_factorial else 97000
    arguments.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    root = Path(__file__).resolve().parents[1]
    manifest = {
        "status": "started", "command": [sys.executable, *sys.argv],
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=root, text=True).strip(),
        "python": sys.executable, "platform": platform.platform(),
        "plan": ("docs/plans/ledh-guidance-beta-factorial-20261016.md" if arguments.guidance_factorial
                 else "docs/plans/ledh-guided-mixture-score-comparison-20261016.md"),
        "output": str(arguments.output.resolve()),
        "result": str((arguments.output / "result.json").resolve()),
        "execution": "escalated FP64 GPU/XLA diagnostic; CPU Kalman reference exception",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "jit_compile": True, "tf32": False, "dtype": "float64",
        "data_version": "diagonal_ar_predict_first_qsd03_rsd05_T5_v1",
        "partitions": {"calibration": 2, "validation": 2, "final": 12},
        "seed_rule": f"{seed_base}+1000*condition+100*partition+index; streams719:data,2:eval initial,3:eval noise,4:eval uniform,10:pilot normal,11:pilot uniform,20+:fit normal,30+:fit uniform",
        "guidance_factorial": arguments.guidance_factorial,
        "beta_max_steps": arguments.beta_max_steps,
        "guide_source": "exact_only_no_fit" if arguments.guidance_factorial else "fitted_and_exact",
        "budget_seconds": arguments.budget_seconds, "particles": 32,
        "horizon": 5, "fit_particles": None if arguments.guidance_factorial else 64,
        "claim": "same-consumer diagnostic; not canonical admission or trained KDM",
    }
    write_json(arguments.output / "manifest.json", manifest)
    rows = []
    calibrations = []
    timings = {"guide_fit": 0., "beta_calibration": 0., "evaluation": 0., "reference": 0.}
    try:
        os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
        import tensorflow as tf
        from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
        manifest["gpu_memory_policy"] = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
        manifest["tensorflow"] = tf.__version__
        tf.config.experimental.enable_tensor_float_32_execution(False)
        manifest["devices"] = [tf.config.experimental.get_device_details(device)
                                  for device in tf.config.list_physical_devices("GPU")]
        with tf.device("/GPU:0"):
            probe = tf.reduce_sum(tf.constant([1., 2.], tf.float64))
        if float(probe) != 3. or "GPU:0" not in probe.device:
            raise RuntimeError("GPU placement check failed")
        manifest["probe_device"] = probe.device
        from bayesfilter.highdim.covariance_proposal_tf import ProposalControls, TRACE_FIELDS, MOMENT_VALIDITY_POLICY_ID
        import bayesfilter.highdim.covariance_proposal_moments_tf
        from bayesfilter.highdim.covariance_proposal_beta_tf import make_beta_solver
        from bayesfilter.highdim.sqmc_lgssm_tf import LGSSMSpec
        from bayesfilter.score_study.backward_guide_pilot_tf import make_scalar_fit, make_exact_future_information
        from bayesfilter.score_study.guided_mixture_diagnostic_tf import make_diagnostic_programs
        from bayesfilter.score_study.guidance_factorial_design import make_factorial_arms
        spec = LGSSMSpec("diagonal_ar", 2)
        controls = ProposalControls(proposal_policy="original_three_branch", reset_steps=5, correction_steps=2)
        manifest["controls"] = dataclasses.asdict(controls)
        manifest["moment_validity_policy"] = MOMENT_VALIDITY_POLICY_ID
        manifest["moment_relative_tolerance"] = math.sqrt(2.220446049250313e-16)
        manifest["evidence_status"] = ("prespecified fresh mechanism diagnostic" if arguments.guidance_factorial
                                       else "exploratory completion after numerical-validity policy amendment")
        paths = {Path(__file__).resolve()}
        paths.update(Path(module.__file__).resolve() for name, module in list(sys.modules.items())
                     if name.startswith("bayesfilter") and getattr(module, "__file__", None))
        manifest["source_sha256"] = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                                      for path in sorted(paths) if path.is_relative_to(root)}
        write_json(arguments.output / "manifest.json", manifest)
        dtype = tf.float64
        evaluate, pilot = make_diagnostic_programs(32, 5, controls)
        solver = make_beta_solver(10, 32, dtype, max_steps=arguments.beta_max_steps)
        equal = tf.constant([1. / 3.] * 3, dtype)
        zero = (tf.zeros([5, 2, 2], dtype), tf.zeros([5, 2], dtype))
        fixed = {"identity": [1., 0., 0.], "global": [.1, .9, 0.],
                 "local": [.1, 0., .9], "equal": [1. / 3.] * 3}

        def budget():
            if time.monotonic() - started > arguments.budget_seconds:
                raise RuntimeError("campaign wall budget exhausted")

        def design(seed, theta, stress):
            observations = spec.simulate(theta, 5, seed)
            if stress:
                alpha = float(theta[0])
                variance = 1.
                for _ in range(3):
                    variance = alpha * alpha * variance + .09
                observations += tf.scatter_nd([[2, 0]], tf.constant([3. * math.sqrt(variance + .25)], dtype), [5, 2])
            initial = tf.random.stateless_normal([32, 2], [seed, 2], dtype=dtype)
            normals = tf.random.stateless_normal([5, 32, 2], [seed, 3], dtype=dtype)
            uniforms = tf.random.stateless_uniform([5, 32, 2], [seed, 4], dtype=dtype)
            return observations, initial, normals, uniforms

        def guides(observations, seed, fitter, exact):
            clock = time.monotonic()
            fitted_precisions, fitted_vectors, exact_precisions, exact_vectors = [], [], [], []
            for coordinate in range(2):
                if fitter is not None:
                    result = fitter(observations[:, coordinate],
                                    tf.random.stateless_normal([5, 64], [seed, 20 + coordinate], dtype=dtype),
                                    tf.random.stateless_uniform([5], [seed, 30 + coordinate], dtype=dtype))
                    if not bool(result["valid"]):
                        raise RuntimeError("controlled-SMC guide fit invalid")
                    fitted_precisions.append(result["precision"][:, 0, 0])
                    fitted_vectors.append(result["linear"][:, 0])
                exact_precision, exact_vector = exact(observations[:, coordinate])
                exact_precisions.append(exact_precision[:, 0, 0])
                exact_vectors.append(exact_vector[:, 0])
            output = {"current": zero,
                      "exact": (tf.linalg.diag(tf.stack(exact_precisions, -1)), tf.stack(exact_vectors, -1))}
            if fitter is not None:
                output["fitted"] = (tf.linalg.diag(tf.stack(fitted_precisions, -1)), tf.stack(fitted_vectors, -1))
            elapsed = time.monotonic() - clock
            timings["guide_fit"] += elapsed
            return output, elapsed

        for condition, (alpha, stress) in enumerate(((.2, False), (.2, True), (.9, False), (.9, True))):
            budget()
            theta = tf.constant([alpha, alpha, .3, .5], dtype)
            scale = tf.constant([1., 1., .3, .5], dtype)
            fitter = None if arguments.guidance_factorial else make_scalar_fit(5, 64, alpha)
            exact = make_exact_future_information(5, alpha)
            collected = {name: [] for name in (("current", "exact") if arguments.guidance_factorial
                                             else ("current", "fitted", "exact"))}
            for index in range(2):
                seed = seed_base + condition * 1000 + index
                data = design(seed, theta, stress)
                messages, _ = guides(data[0], seed, fitter, exact)
                for name, message in messages.items():
                    budget()
                    clock = time.monotonic()
                    result = pilot(theta, *data, equal, *message,
                                   tf.random.stateless_normal([5, 32, 2], [seed, 10], dtype=dtype),
                                   tf.random.stateless_uniform([5, 32, 2], [seed, 11], dtype=dtype))
                    if not bool(result["valid"]):
                        raise RuntimeError(f"invalid calibration context: condition={condition}, guide={name}")
                    collected[name].append(result)
                    timings["beta_calibration"] += time.monotonic() - clock
            betas = {}
            for name, contexts in collected.items():
                clock = time.monotonic()
                density = tf.concat([context["log_components"] for context in contexts], 0)
                target = tf.concat([context["log_target"] for context in contexts], 0)
                result = solver(density, target, tf.constant(.1, dtype), tf.constant(1e-7, dtype))
                if not bool(result["converged"]):
                    write_json(arguments.output / "calibration-failure.json", {
                        "condition": condition, "guide": name, "beta": result["beta"].numpy().tolist(),
                        "objective": float(result["objective"]), "gap": float(result["gap"]),
                        "iterations": int(result["iterations"]),
                        "log_components": density.numpy().tolist(), "log_target": target.numpy().tolist()})
                    raise RuntimeError(f"beta solver did not converge: condition={condition}, guide={name}")
                betas[name] = result["beta"]
                calibrations.append({"condition": condition, "guide": name,
                                     "beta": result["beta"].numpy().tolist(),
                                     "objective": float(result["objective"]),
                                     "gap": float(result["gap"]), "iterations": int(result["iterations"])})
                timings["beta_calibration"] += time.monotonic() - clock
            write_json(arguments.output / "calibration.json", calibrations)
            for partition, count in ((1, 2), (2, 12)):
                for index in range(count):
                    budget()
                    seed = seed_base + condition * 1000 + 100 * partition + index
                    data = design(seed, theta, stress)
                    messages, fit_seconds = guides(data[0], seed, fitter, exact)
                    clock = time.monotonic()
                    reference_value, reference_score = spec.reference_value_and_score(theta, data[0])
                    oracle = reference_score * scale
                    if not bool(tf.reduce_all(tf.math.is_finite(oracle))):
                        raise RuntimeError("nonfinite Kalman reference")
                    timings["reference"] += time.monotonic() - clock
                    arms = [(name, tf.constant(beta, dtype), zero) for name, beta in fixed.items()]
                    arms += [(name, betas[name], messages[name]) for name in betas]
                    if arguments.guidance_factorial:
                        arms = make_factorial_arms(
                            {name: tf.constant(beta, dtype) for name, beta in fixed.items()}, betas, messages)
                    paired_input_hash = hashlib.sha256(json.dumps(
                        [tensor.numpy().tolist() for tensor in data], allow_nan=False).encode()).hexdigest()
                    for name, beta, message in arms:
                        budget()
                        clock = time.monotonic()
                        result = evaluate(theta, *data, beta, *message)
                        valid = bool(result["valid"])
                        if not valid or not bool(tf.reduce_all(tf.math.is_finite(result["score"]))):
                            write_json(arguments.output / "failure.json", {"condition": condition,
                                       "partition": partition, "seed": seed, "arm": name, "valid": valid,
                                       "direction_valid": result["direction_valid"].numpy().tolist(),
                                       "trace_fields": TRACE_FIELDS,
                                       "direction_trace": [[[value if math.isfinite(value) else None for value in row]
                                                            for row in trace] for trace in result["direction_trace"].numpy().tolist()]})
                            raise RuntimeError("invalid evaluation; no incomplete row admitted")
                        elapsed = time.monotonic() - clock
                        timings["evaluation"] += elapsed
                        score = result["score"] * scale
                        trace = result["trace"]
                        row = {"condition": condition, "alpha": alpha, "impulse": stress,
                               "partition": "validation" if partition == 1 else "final",
                               "seed": seed, "arm": name, "score": score.numpy().tolist(),
                               "paired_input_sha256": paired_input_hash,
                               "reference_score": oracle.numpy().tolist(),
                               "squared_error": tf.square(score - oracle).numpy().tolist(),
                               "value_error": float(result["value"] - reference_value),
                               "evaluation_seconds": elapsed, "paired_guide_seconds": fit_seconds,
                               "beta": beta.numpy().tolist(),
                               "trace_min": {field: float(tf.reduce_min(trace[:, column])) for column, field in enumerate(TRACE_FIELDS)},
                               "trace_max": {field: float(tf.reduce_max(trace[:, column])) for column, field in enumerate(TRACE_FIELDS)}}
                        rows.append(row)
                        with (arguments.output / "rows.jsonl").open("a") as stream:
                            stream.write(json.dumps(row, allow_nan=False) + "\n")
                    print(json.dumps({"condition": condition, "partition": partition,
                                      "index": index, "elapsed": time.monotonic() - started}), flush=True)
            del fitter, exact
        manifest["status"] = "complete"
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        if "tf" in locals() and "probe_device" in manifest:
            manifest["gpu_allocator_bytes"] = tf.config.experimental.get_memory_info("GPU:0")
        manifest["wall_seconds"] = time.monotonic() - started
        manifest["timings"] = timings
        manifest["row_count"] = len(rows)
        write_json(arguments.output / "result.json", {"status": manifest["status"],
                   "rows": len(rows), "calibrations": calibrations,
                   "error": manifest.get("error"), "timings": timings})
        write_json(arguments.output / "manifest.json", manifest)


if __name__ == "__main__":
    main()
