"""Conditional cloud diagnostic using the shared mixture; no canonical claim."""

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

from bayesfilter.score_study.conditional_cloud_design import (
    ARMS, FIXED_BETAS, PARTICLE_COUNTS, REPLICATES, cloud_seed, data_seed, make_clouds)
from bayesfilter.score_study.guidance_factorial_design import make_factorial_arms


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def tensor_hash(tensors):
    values = [tensor.numpy().tolist() for tensor in tensors]
    return hashlib.sha256(json.dumps(values, allow_nan=False).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--particles", type=int, choices=PARTICLE_COUNTS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget-seconds", type=float, required=True)
    parser.add_argument("--beta-max-steps", type=int, default=20000)
    arguments = parser.parse_args()
    if arguments.beta_max_steps < 1 or arguments.budget_seconds <= 0:
        parser.error("positive solver and wall budgets required")
    arguments.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    started = time.monotonic()
    particles = arguments.particles
    manifest = {
        "status": "started", "command": [sys.executable, *sys.argv],
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=root, text=True).strip(),
        "python": sys.executable, "conda_environment": "tftwogpu", "platform": platform.platform(),
        "plan": "docs/plans/ledh-conditional-cloud-localization-20261016.md",
        "result": str(arguments.output.resolve() / "result.json"),
        "output": str(arguments.output.resolve()), "particles": particles, "horizon": 5,
        "data_version": "diagonal_ar_predict_first_qsd03_rsd05_T5_v1",
        "theta": [.9, .9, .3, .5], "partitions": {"calibration": 2, "validation": 2, "final": 2},
        "cloud_replicates": REPLICATES, "arms": ARMS,
        "data_seed_rule": "317000+1000*regime+100*partition+dataset",
        "cloud_seed_rule": "417000+10000*regime+1000*partition+100*dataset+replicate; calibration adds 100000*N",
        "cloud_streams": {"initial": 2, "innovations": 3, "uniforms": 4,
                          "pilot_normal": 10, "pilot_uniform": 11},
        "budget_seconds": arguments.budget_seconds, "beta_max_steps": arguments.beta_max_steps,
        "execution": "escalated FP64 GPU/XLA diagnostic; CPU Kalman reference exception",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "dtype": "float64", "tf32": False, "jit_compile": True,
        "transport_extent": {"rows": particles, "columns": particles, "grid": [1, 1]},
        "score_engine": "shared covariance_proposal_tf analytical recursion",
        "claim": "conditional mechanism diagnostic; optional mixture, no canonical admission or learned KDM",
    }
    write_json(arguments.output / "manifest.json", manifest)
    row_count = 0
    calibrations = []
    timings = {"calibration": 0., "evaluation": 0., "reference": 0.}
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
            raise RuntimeError("GPU placement failed")
        manifest["probe_device"] = probe.device
        from bayesfilter.highdim.covariance_proposal_tf import (
            MOMENT_VALIDITY_POLICY_ID, ProposalControls, TRACE_FIELDS)
        from bayesfilter.highdim.covariance_proposal_beta_tf import make_beta_solver
        import bayesfilter.highdim.covariance_proposal_moments_tf
        from bayesfilter.highdim.sqmc_lgssm_tf import LGSSMSpec
        from bayesfilter.score_study.backward_guide_pilot_tf import make_exact_future_information
        from bayesfilter.score_study.guided_mixture_diagnostic_tf import make_diagnostic_programs
        controls = ProposalControls(reset_steps=5, correction_steps=2)
        manifest["controls"] = dataclasses.asdict(controls)
        manifest["moment_validity_policy"] = MOMENT_VALIDITY_POLICY_ID
        paths = {Path(__file__).resolve()}
        paths.update(Path(module.__file__).resolve() for name, module in list(sys.modules.items())
                     if name.startswith("bayesfilter") and getattr(module, "__file__", None))
        manifest["source_sha256"] = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                                      for path in sorted(paths) if path.is_relative_to(root)}
        write_json(arguments.output / "manifest.json", manifest)
        dtype = tf.float64
        theta = tf.constant(manifest["theta"], dtype)
        scale = tf.constant([1., 1., .3, .5], dtype)
        spec = LGSSMSpec("diagonal_ar", 2)
        evaluate, pilot = make_diagnostic_programs(particles, 5, controls)
        solver = make_beta_solver(10, particles, dtype, max_steps=arguments.beta_max_steps)
        exact = make_exact_future_information(5, .9)
        zero = (tf.zeros([5, 2, 2], dtype), tf.zeros([5, 2], dtype))
        fixed = {name: tf.constant(beta, dtype) for name, beta in FIXED_BETAS.items()}

        def budget():
            if time.monotonic() - started >= arguments.budget_seconds:
                raise RuntimeError("scope wall budget exhausted")

        def dataset_inputs(regime, partition, dataset):
            observations = spec.simulate(theta, 5, data_seed(regime, partition, dataset))
            if regime:
                variance = 1.
                for _ in range(3):
                    variance = .81 * variance + .09
                observations += tf.scatter_nd([[2, 0]], tf.constant([3. * math.sqrt(variance + .25)], dtype), [5, 2])
            components = [exact(observations[:, coordinate]) for coordinate in range(2)]
            message = (tf.linalg.diag(tf.stack([item[0][:, 0, 0] for item in components], -1)),
                       tf.stack([item[1][:, 0] for item in components], -1))
            return observations, {"current": zero, "exact": message}

        for regime in range(2):
            budget()
            collected = {"current": [], "exact": []}
            clock = time.monotonic()
            for dataset in range(2):
                observations, messages = dataset_inputs(regime, 0, dataset)
                seed = cloud_seed(regime, 0, dataset, 0, particles)
                data = (observations, *make_clouds(tf, particles, seed))
                for guide, message in messages.items():
                    budget()
                    result = pilot(theta, *data, fixed["equal"], *message,
                                   tf.random.stateless_normal([5, particles, 2], [seed, 10], dtype=dtype),
                                   tf.random.stateless_uniform([5, particles, 2], [seed, 11], dtype=dtype))
                    if not bool(result["valid"]):
                        write_json(arguments.output / "failure.json", {
                            "stage": "calibration_context", "regime": regime, "guide": guide,
                            "dataset": dataset, "cloud_seed": seed})
                        raise RuntimeError("invalid calibration context")
                    collected[guide].append(result)
            betas = {}
            for guide, contexts in collected.items():
                budget()
                density = tf.concat([context["log_components"] for context in contexts], 0)
                target = tf.concat([context["log_target"] for context in contexts], 0)
                result = solver(density, target, tf.constant(.1, dtype), tf.constant(1e-7, dtype))
                record = {"regime": regime, "guide": guide, "particles": particles,
                          "beta": result["beta"].numpy().tolist(), "converged": bool(result["converged"]),
                          "objective": float(result["objective"]), "gap": float(result["gap"]),
                          "iterations": int(result["iterations"]),
                          "log_components": density.numpy().tolist(), "log_target": target.numpy().tolist()}
                calibrations.append(record)
                write_json(arguments.output / "calibration.json", calibrations)
                if not record["converged"]:
                    raise RuntimeError("calibration certificate failed")
                betas[guide] = result["beta"]
            timings["calibration"] += time.monotonic() - clock
            for partition, count in ((1, 1), (2, REPLICATES)):
                for dataset in range(2):
                    observations, messages = dataset_inputs(regime, partition, dataset)
                    clock = time.monotonic()
                    reference_value, reference_score = spec.reference_value_and_score(theta, observations)
                    oracle = reference_score * scale
                    timings["reference"] += time.monotonic() - clock
                    data_hash = tensor_hash([observations])
                    with (arguments.output / "datasets.jsonl").open("a") as stream:
                        stream.write(json.dumps({"regime": regime, "partition": partition,
                            "dataset": dataset, "data_seed": data_seed(regime, partition, dataset),
                            "data_sha256": data_hash, "observations": observations.numpy().tolist(),
                            "reference_value": float(reference_value), "reference_score": oracle.numpy().tolist()},
                            allow_nan=False) + "\n")
                    arms = [arm for arm in make_factorial_arms(fixed, betas, messages) if arm[0] in ARMS]
                    for replicate in range(count):
                        seed = cloud_seed(regime, partition, dataset, replicate, particles)
                        initial, normals, uniforms = make_clouds(tf, particles, seed)
                        data = observations, initial, normals, uniforms
                        input_hash = tensor_hash(data)
                        prefix_hash = tensor_hash((observations, initial[:32], normals[:, :32], uniforms[:, :32]))
                        for arm, beta, message in arms:
                            budget()
                            clock = time.monotonic()
                            result = evaluate(theta, *data, beta, *message)
                            if not bool(result["valid"]) or not bool(tf.reduce_all(tf.math.is_finite(result["score"]))):
                                write_json(arguments.output / "failure.json", {
                                    "regime": regime, "partition": partition, "dataset": dataset,
                                    "replicate": replicate, "arm": arm, "valid": bool(result["valid"]),
                                    "trace_fields": TRACE_FIELDS,
                                    "direction_valid": result["direction_valid"].numpy().tolist(),
                                    "direction_trace": [[[value if math.isfinite(value) else None for value in row]
                                                         for row in trace] for trace in result["direction_trace"].numpy().tolist()]})
                                raise RuntimeError("invalid evaluation; incomplete row not admitted")
                            elapsed = time.monotonic() - clock
                            timings["evaluation"] += elapsed
                            score = result["score"] * scale
                            trace = result["trace"]
                            row = {"particles": particles, "regime": regime, "partition": partition,
                                   "dataset": dataset, "replicate": replicate, "arm": arm,
                                   "data_seed": data_seed(regime, partition, dataset), "cloud_seed": seed,
                                   "data_sha256": data_hash, "paired_input_sha256": input_hash,
                                   "prefix32_sha256": prefix_hash, "beta": beta.numpy().tolist(),
                                   "score": score.numpy().tolist(), "reference_score": oracle.numpy().tolist(),
                                   "squared_error": tf.square(score - oracle).numpy().tolist(),
                                   "value_error": float(result["value"] - reference_value),
                                   "evaluation_seconds": elapsed,
                                   "trace_min": {field: float(tf.reduce_min(trace[:, column])) for column, field in enumerate(TRACE_FIELDS)},
                                   "trace_max": {field: float(tf.reduce_max(trace[:, column])) for column, field in enumerate(TRACE_FIELDS)}}
                            with (arguments.output / "rows.jsonl").open("a") as stream:
                                stream.write(json.dumps(row, allow_nan=False) + "\n")
                            row_count += 1
                    print(json.dumps({"particles": particles, "regime": regime,
                                      "partition": partition, "dataset": dataset,
                                      "rows": row_count, "wall_seconds": time.monotonic() - started}), flush=True)
        manifest["status"] = "complete"
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        if "tf" in locals() and "probe_device" in manifest:
            manifest["gpu_allocator_bytes"] = tf.config.experimental.get_memory_info("GPU:0")
        manifest["wall_seconds"] = time.monotonic() - started
        manifest["row_count"] = row_count
        manifest["timings"] = timings
        write_json(arguments.output / "result.json", {"status": manifest["status"], "rows": row_count,
                   "error": manifest.get("error"), "timings": timings})
        write_json(arguments.output / "manifest.json", manifest)


if __name__ == "__main__":
    main()
