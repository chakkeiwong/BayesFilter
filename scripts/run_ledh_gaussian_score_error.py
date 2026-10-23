"""Bounded GPU/XLA reference diagnosis of the full LEDH Gaussian score."""

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


def diagnostic_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, list):
        return [diagnostic_json(item) for item in value]
    if isinstance(value, dict):
        return {key: diagnostic_json(item) for key, item in value.items()}
    return value


def summarize(rows):
    groups = {}
    for row in rows:
        key = (row["condition"], row["particles"], row["arm"], row["flow_substeps"])
        groups.setdefault(key, []).append(row)
    summary = []
    for (condition, particles, arm, substeps), group in groups.items():
        summary.append({
            "condition": condition, "particles": particles, "arm": arm,
            "flow_substeps": substeps, "rows": len(group),
            "datasets": len({row["dataset"] for row in group}),
            "score_mse": [sum((row["score"][coordinate] - row["reference_score"][coordinate]) ** 2
                               for row in group) / len(group) for coordinate in range(6)],
            "value_mse": sum((row["value"] - row["reference_value"]) ** 2 for row in group) / len(group),
            "primary_score_errors": [row["score"][1] - row["reference_score"][1] for row in group],
        })
    for item in summary:
        if item["arm"] == "ledh":
            comparators = [other for other in summary if other["condition"] == item["condition"]
                           and other["particles"] == item["particles"] and other["arm"] != "ledh"]
            losses = [other["arm"] for other in comparators if item["score_mse"][1] > other["score_mse"][1]]
            item["heuristic_losses"] = losses
            item["heuristic_dominance"] = ("PROMOTION_VETO_DESCRIPTIVE_HEURISTIC_LOSS" if losses
                                           else "NO_DESCRIPTIVE_LOSS_NOT_PROMOTION")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget-seconds", type=float, default=1200.0)
    parser.add_argument("--resume-from", type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    plan = "docs/plans/ledh-gaussian-score-error-20261016.md"
    manifest = {
        "schema": "ledh_gaussian_score_error_diagnostic_v1",
        "plan": plan, "result": str(args.output / "result.json"),
        "command": [sys.executable, *sys.argv],
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=root, text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--short"], cwd=root, text=True),
        "environment": {name: os.environ.get(name) for name in (
            "CUDA_VISIBLE_DEVICES", "TF_FORCE_GPU_ALLOW_GROWTH", "TF_NUM_INTRAOP_THREADS",
            "TF_NUM_INTEROP_THREADS")},
        "python": platform.python_version(), "executable": sys.executable, "host": platform.node(),
        "budget_seconds": args.budget_seconds, "jit_compile": True, "dtype": "float64",
        "execution_role": "GPU/XLA FP64 reference exception; no canonical or production admission",
        "data_version": "scalar_gaussian_T6_fresh_seed710000_diagnostic_v1",
        "seed_base": 20261016, "primary_coordinate": "log_process_noise_standard_deviation",
        "source_sha256": {}, "status": "initializing", "nonexclusive_gpu": True,
    }
    for directory in ("bayesfilter/highdim", "bayesfilter/score_study", "bayesfilter/runtime"):
        for source in sorted((root / directory).glob("*.py")):
            manifest["source_sha256"][str(source.relative_to(root))] = hashlib.sha256(source.read_bytes()).hexdigest()
    for relative in (plan, "scripts/run_ledh_gaussian_score_error.py"):
        manifest["source_sha256"][relative] = hashlib.sha256((root / relative).read_bytes()).hexdigest()
    write_json(args.output / "manifest.json", manifest)

    def check_budget():
        if time.monotonic() - started > args.budget_seconds:
            raise TimeoutError("Diagnostic wall-time allocation exhausted")

    rows = []
    kernels = {}
    try:
        if args.resume_from is not None:
            previous = json.loads((args.resume_from / "manifest.json").read_text())
            if not previous["source_hashes_unchanged"]:
                raise ValueError("Previous numerical source provenance failed")
            for name, digest in previous["source_sha256"].items():
                if name.startswith("bayesfilter/") and manifest["source_sha256"].get(name) != digest:
                    raise ValueError("Resume cannot change numerical dependencies")
            rows = [json.loads(line) for line in (args.resume_from / "rows.jsonl").read_text().splitlines()]
            manifest["resume"] = {"path": str(args.resume_from), "retained_rows": len(rows),
                                  "previous_wall_seconds": previous["wall_seconds"],
                                  "rows_sha256": hashlib.sha256((args.resume_from / "rows.jsonl").read_bytes()).hexdigest()}
        import tensorflow as tf
        from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
        manifest["gpu_memory_policy"] = configure_tensorflow_gpu_memory_growth(tf)
        tf.config.experimental.enable_tensor_float_32_execution(False)
        manifest["tf32"] = False
        manifest["tensorflow"] = tf.__version__
        manifest["gpu_before"] = subprocess.check_output([
            "nvidia-smi", "--query-gpu=name,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader"], text=True).strip()
        manifest["devices"] = [device.name for device in tf.config.list_logical_devices("GPU")]
        if not manifest["devices"]:
            raise RuntimeError("Trusted GPU device is required")

        from bayesfilter.score_study.backward_guide_pilot_tf import (
            REFERENCE_CONTROLS, make_guided_reference_kernel, reference_theta)
        from bayesfilter.score_study.control_diagnostics_tf import materialize_control_diagnostics
        from bayesfilter.score_study.gaussian_tf import (
            make_data_kernel, make_gaussian_kernel, make_particle_kernel, parameterized_model)

        horizon = 6
        directions = tf.eye(6, dtype=tf.float64)
        zero_precision = tf.zeros([horizon, 1, 1], tf.float64)
        zero_linear = tf.zeros([horizon, 1], tf.float64)
        data = make_data_kernel(1, 1, horizon)
        kalman = make_gaussian_kernel(1, 1, 6)
        manifest["controls"] = {**REFERENCE_CONTROLS, "flow_substeps": [48, 192]}
        manifest["grid"] = {"particles": [32, 128], "flow_substeps": [48, 192], "horizon": horizon,
                            "datasets_per_alpha": 6, "stress_datasets": 2, "streams": 2}
        write_json(args.output / "manifest.json", manifest)

        @tf.function(input_signature=[tf.TensorSpec([2], tf.int32)], jit_compile=True)
        def random_inputs(seed):
            return (
                tf.random.stateless_normal([128, 1], tf.random.experimental.stateless_fold_in(seed, 1), dtype=tf.float64),
                tf.random.stateless_normal([horizon, 128, 1], tf.random.experimental.stateless_fold_in(seed, 2), dtype=tf.float64),
                tf.random.stateless_uniform([horizon, 128], tf.random.experimental.stateless_fold_in(seed, 3), dtype=tf.float64),
            )

        def get_kernel(particles, steps):
            key = (particles, steps)
            if key not in kernels:
                check_budget()
                print(f"Compiling full shared LEDH N={particles}, flow={steps}", flush=True)
                kernels[key] = make_guided_reference_kernel(
                    particles, horizon, {**REFERENCE_CONTROLS, "flow_substeps": steps}, error_diagnostics=True)
            return kernels[key]

        def evaluate(theta, observations, inputs, particles, steps):
            check_budget()
            tick = time.monotonic()
            output = get_kernel(particles, steps)(theta, directions, observations, inputs[0][:particles],
                                                  inputs[1][:, :particles], zero_precision, zero_linear)
            if not bool(output["valid"]) or not bool(output["value_invariant"]):
                write_json(args.output / "rejected-evaluation.json", diagnostic_json({
                    "particles": particles, "flow_substeps": steps,
                    "theta": theta.numpy().tolist(), "observations": observations.numpy().tolist(),
                    "initial_noise": inputs[0][:particles].numpy().tolist(),
                    "process_noise": inputs[1][:, :particles].numpy().tolist(),
                    "valid": bool(output["valid"]), "value_invariant": bool(output["value_invariant"]),
                    "values": output["outputs"][0].numpy().tolist(), "scores": output["score"].numpy().tolist(),
                    "diagnostics": {name: tensor.numpy().tolist() for name, tensor in output["outputs"][2].items()},
                }))
                raise ValueError("Shared LEDH validity or direction-invariance veto")
            tensors = output["outputs"][2]
            diagnostics = materialize_control_diagnostics(tensors)
            value_error = tf.reduce_max(tf.abs(tf.reduce_sum(tensors["value_increment"], axis=1) - output["value"]))
            score_error = tf.reduce_max(tf.abs(tf.reduce_sum(tensors["score_increment"], axis=1) - output["score"]))
            if float(value_error / (1 + tf.abs(output["value"]))) > 1e-9 or float(
                    score_error / (1 + tf.reduce_max(tf.abs(output["score"])))) > 1e-9:
                raise ValueError("Trace increment reconstruction veto")
            return {"value": float(output["value"]), "score": output["score"].numpy().tolist(),
                    "diagnostics": diagnostics, "wall_seconds": time.monotonic() - tick,
                    "increment_value_error": float(value_error), "increment_score_error": float(score_error)}

        cases = []
        for alpha_index, alpha in enumerate((0.0, 0.9)):
            theta = reference_theta(alpha)
            model = parameterized_model(theta, 1, 1)
            for dataset in range(6):
                seed = [20261016, 710000 + alpha_index * 100 + dataset]
                observations = data(theta, tf.constant(seed, tf.int32))
                condition = "independent" if alpha == 0 else "persistent"
                cases.append({"condition": condition, "dataset": dataset, "theta": theta,
                              "observations": observations, "observation_seed": seed,
                              "stream_id": alpha_index * 100 + dataset, "stress_delta": 0.0})
                if alpha > 0 and dataset < 2:
                    prefix = kalman(observations[:2], *model)
                    predictive = model[0] @ prefix[3] @ tf.transpose(model[0]) + model[8]
                    variance = model[2] @ predictive @ tf.transpose(model[2]) + model[10]
                    delta = 3 * tf.sqrt(variance[0, 0])
                    shifted = tf.tensor_scatter_nd_add(observations, [[2, 0]], [delta])
                    cases.append({**cases[-1], "condition": "persistent_stress",
                                  "observations": shifted, "stress_delta": float(delta)})

        for case in cases:
            model = parameterized_model(case["theta"], 1, 1)
            prefix = [kalman(case["observations"][:length], *model) for length in range(1, horizon + 1)]
            values = tf.stack([item[0] for item in prefix])
            scores = tf.stack([item[1] for item in prefix])
            case["reference"] = {
                "reference_value": float(values[-1]), "reference_score": scores[-1].numpy().tolist(),
                "reference_value_increments": (values - tf.concat([tf.zeros([1], tf.float64), values[:-1]], 0)).numpy().tolist(),
                "reference_score_increments": (scores - tf.concat([tf.zeros([1, 6], tf.float64), scores[:-1]], 0)).numpy().tolist(),
                "reference_means": [float(item[2][0]) for item in prefix],
                "reference_variances": [float(item[3][0, 0]) for item in prefix],
            }

        finite_differences = []
        if args.resume_from is not None:
            finite_differences = json.loads((args.resume_from / "finite-differences.json").read_text())
            if len(finite_differences) != 2 or not all(item["passed"] for item in finite_differences):
                raise ValueError("Resume requires completed derivative checks")
            parity = []
            for condition in ("independent", "persistent"):
                retained = next(row for row in rows if row["arm"] == "ledh" and row["condition"] == condition)
                case = next(case for case in cases if case["condition"] == condition and case["dataset"] == retained["dataset"])
                observation_error = float(tf.reduce_max(tf.abs(case["observations"] - tf.constant(retained["observations"], tf.float64))))
                inputs = random_inputs(tf.constant(retained["stream_seed"], tf.int32))
                actual = evaluate(case["theta"], case["observations"], inputs, retained["particles"], retained["flow_substeps"])
                error = max(abs(actual["value"] - retained["value"]) / (1 + abs(retained["value"])),
                            *[abs(after - before) / (1 + abs(before)) for after, before in zip(actual["score"], retained["score"], strict=True)])
                parity.append({"condition": condition, "observation_error": observation_error, "value_score_scaled_error": error})
                if observation_error > 1e-12 or error > 1e-9:
                    raise ValueError("Cross-device replay parity veto")
            write_json(args.output / "resume-parity.json", parity)
            write_json(args.output / "finite-differences.json", finite_differences)
            print(f"Cross-device replay passes; retaining {len(rows)} completed rows", flush=True)
        for condition in (() if args.resume_from is not None else ("persistent", "persistent_stress")):
            case = next(case for case in cases if case["condition"] == condition and case["dataset"] == 0)
            theta, observations = case["theta"], case["observations"]
            stream_seed = [20261016, 720000 + case["stream_id"] * 2]
            inputs = random_inputs(tf.constant(stream_seed, tf.int32))
            base = evaluate(theta, observations, inputs, 32, 48)
            check = {"condition": condition, "observation_seed": case["observation_seed"],
                     "stream_seed": stream_seed, "score": base["score"],
                     "reference_score": case["reference"]["reference_score"], "steps": []}
            for step in (1e-4, 3e-5):
                differences, oracle_differences = [], []
                for coordinate in range(6):
                    displacement = step * directions[coordinate]
                    plus = evaluate(theta + displacement, observations, inputs, 32, 48)
                    minus = evaluate(theta - displacement, observations, inputs, 32, 48)
                    differences.append((plus["value"] - minus["value"]) / (2 * step))
                    oracle_plus = kalman(observations, *parameterized_model(theta + displacement, 1, 1))[0]
                    oracle_minus = kalman(observations, *parameterized_model(theta - displacement, 1, 1))[0]
                    oracle_differences.append(float((oracle_plus - oracle_minus) / (2 * step)))
                check["steps"].append({"step": step, "finite_difference": differences,
                                       "oracle_finite_difference": oracle_differences})
            check["scaled_error"] = [min(abs(row["finite_difference"][coordinate] - base["score"][coordinate])
                / (1 + abs(base["score"][coordinate])) for row in check["steps"]) for coordinate in range(6)]
            check["oracle_scaled_error"] = [min(abs(row["oracle_finite_difference"][coordinate]
                - check["reference_score"][coordinate]) / (1 + abs(check["reference_score"][coordinate]))
                for row in check["steps"]) for coordinate in range(6)]
            check["passed"] = max(check["scaled_error"] + check["oracle_scaled_error"]) <= 2e-5
            finite_differences.append(check)
            write_json(args.output / "finite-differences.json", finite_differences)
            if not check["passed"]:
                raise ValueError("Finite-program derivative veto; stop model-score comparison")
            print(f"Finite differences pass: {condition}", flush=True)

        manifest["status"] = "running_grid"
        write_json(args.output / "manifest.json", manifest)
        completed = {(row["condition"], row["dataset"], row["stream"], row["particles"], row["arm"], row["flow_substeps"]) for row in rows}
        with (args.output / "rows.jsonl").open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row, allow_nan=False) + "\n")
            handle.flush()
            for particles in (32, 128):
                comparators = {
                    "prior_sis": make_particle_kernel(1, 1, particles, horizon),
                    "bootstrap_pf": make_particle_kernel(1, 1, particles, horizon, resampling=True),
                    "adapted_sis": make_particle_kernel(1, 1, particles, horizon, adapted=True),
                }
                for case in cases:
                    print(f"N={particles} {case['condition']} dataset={case['dataset']}", flush=True)
                    for stream in range(2):
                        stream_seed = [20261016, 720000 + case["stream_id"] * 2 + stream]
                        inputs = random_inputs(tf.constant(stream_seed, tf.int32))
                        common = {key: value for key, value in case.items() if key not in ("theta", "observations", "reference")}
                        common.update(case["reference"], particles=particles, stream=stream, stream_seed=stream_seed,
                                      observations=case["observations"].numpy().tolist(), theta=case["theta"].numpy().tolist())
                        for arm, steps in [("ledh", 48), ("ledh", 192), *[(name, None) for name in comparators]]:
                            if (case["condition"], case["dataset"], stream, particles, arm, steps) in completed:
                                continue
                            check_budget()
                            if arm == "ledh":
                                result = evaluate(case["theta"], case["observations"], inputs, particles, steps)
                            else:
                                tick = time.monotonic()
                                operands = (case["theta"], case["observations"], inputs[0][:particles], inputs[1][:, :particles])
                                actual = (comparators[arm](*operands, inputs[2][:, :particles]) if arm == "bootstrap_pf"
                                          else comparators[arm](*operands))
                                if not all(bool(tf.reduce_all(tf.math.is_finite(tensor))) for tensor in actual):
                                    raise ValueError("Nonfinite heuristic comparator")
                                result = {"value": float(actual[0]), "score": actual[1].numpy().tolist(),
                                          "minimum_ess": float(actual[2]), "wall_seconds": time.monotonic() - tick}
                            row = {**common, "arm": arm, "flow_substeps": steps, **result}
                            handle.write(json.dumps(row, allow_nan=False) + "\n")
                            handle.flush()
                            rows.append(row)

        summary = summarize(rows)
        write_json(args.output / "result.json", {
            "status": "DIAGNOSTIC_COMPLETE_NO_PROMOTION", "rows": len(rows), "summary": summary,
            "hard_veto": False, "statistically_supported_ranking": False,
            "default_readiness": False, "interpretation": "Descriptive Gaussian error attribution only",
        })
        manifest["status"] = "complete"
    except (Exception, KeyboardInterrupt) as error:
        manifest["status"] = "stopped"
        manifest["failure"] = {"type": type(error).__name__, "message": str(error)}
        write_json(args.output / "partial-result.json", {"rows": len(rows), "summary": summarize(rows)})
        raise
    finally:
        manifest["wall_seconds"] = time.monotonic() - started
        manifest["remaining_budget_seconds"] = max(0.0, args.budget_seconds - manifest["wall_seconds"])
        manifest["trace_counts"] = {f"N{particles}_flow{steps}": kernel.experimental_get_tracing_count()
                                    for (particles, steps), kernel in kernels.items()}
        if manifest.get("devices"):
            manifest["gpu_memory_final"] = tf.config.experimental.get_memory_info("GPU:0")
        manifest["source_hashes_unchanged"] = all(hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
                                                 for name, digest in manifest["source_sha256"].items())
        manifest["artifact_paths"] = sorted(str(path) for path in args.output.iterdir())
        write_json(args.output / "manifest.json", manifest)


if __name__ == "__main__":
    main()
