"""Bounded diagnostic Gaussian experiment; no canonical admission or tuning."""

import argparse
import hashlib
import json
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
    parser.add_argument("--budget-seconds", type=float, default=1800.0)
    args = parser.parse_args()
    started = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    plan = "docs/plans/ledh-fitted-guide-pilot-20261016.md"
    manifest = {
        "schema": "ledh_fitted_guide_reference_pilot_v1",
        "plan": plan, "result": str(args.output / "result.json"),
        "command": [sys.executable, *sys.argv],
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=root, text=True).strip(),
        "environment": {name: os.environ.get(name) for name in (
            "CUDA_VISIBLE_DEVICES", "TF_FORCE_GPU_ALLOW_GROWTH",
            "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS")},
        "python": platform.python_version(), "host": platform.node(),
        "budget_seconds": args.budget_seconds,
        "status": "initializing", "jit_compile": True, "dtype": "float64",
        "execution_role": "GPU/XLA Gaussian reference exception; no default or canonical claim",
        "seed_base": 20261016, "data_version": "scalar_gaussian_adapter_T6_N32_v1",
        "source_sha256": {},
    }
    for name in (
        "scripts/run_ledh_fitted_guide_pilot.py",
        "bayesfilter/score_study/backward_guide_pilot_tf.py",
        "bayesfilter/score_study/gaussian_tf.py",
        "bayesfilter/score_study/canonical_adapter_tf.py",
        "bayesfilter/score_study/direction_assembly_tf.py",
        "bayesfilter/score_study/control_diagnostics_tf.py",
        "bayesfilter/highdim/ledh_backward_guide_tf.py",
        "bayesfilter/highdim/ledh_canonical_score_tf.py",
        "bayesfilter/highdim/ledh_canonical_reset_score_tf.py",
        "bayesfilter/highdim/ledh_unified_correction_tf.py",
        "bayesfilter/replication/controlled_smc_tf.py",
    ):
        manifest["source_sha256"][name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
    write_json(args.output / "manifest.json", manifest)

    def check_budget():
        if time.monotonic() - started > args.budget_seconds:
            raise TimeoutError("The declared GPU campaign budget is exhausted")

    try:
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
        manifest["nonexclusive_gpu"] = True
        write_json(args.output / "manifest.json", manifest)

        from bayesfilter.score_study.backward_guide_pilot_tf import (
            REFERENCE_CONTROLS, make_exact_future_information,
            make_guided_reference_kernel, make_scalar_fit, reference_theta)
        from bayesfilter.score_study.control_diagnostics_tf import materialize_control_diagnostics
        from bayesfilter.score_study.gaussian_tf import (
            make_data_kernel, make_gaussian_kernel, make_particle_kernel, parameterized_model)

        horizon, particles = 6, 32
        manifest["controls"] = REFERENCE_CONTROLS
        manifest["scope"] = {
            "horizon": horizon, "particles": particles, "pilot_particles": 64,
            "state_dimension": 1, "parameter_dimension": 6, "chunk": particles,
            "calibration_datasets": 4, "validation_datasets": 4,
            "untouched_datasets": 12, "particle_streams": 2,
            "settings_status": "untuned development comparator; exploratory guide nomination only",
        }
        data = make_data_kernel(1, 1, horizon)
        kalman = make_gaussian_kernel(1, 1, 6)
        bootstrap = make_particle_kernel(1, 1, particles, horizon, resampling=True)
        adapted = make_particle_kernel(1, 1, particles, horizon, adapted=True)
        directions = tf.eye(6, dtype=tf.float64)
        zero_precision = tf.zeros([horizon, 1, 1], tf.float64)
        zero_linear = tf.zeros([horizon, 1], tf.float64)

        @tf.function(input_signature=[tf.TensorSpec([2], tf.int32)], jit_compile=True)
        def random_inputs(seed):
            def folded(offset):
                return tf.random.experimental.stateless_fold_in(seed, offset)
            return (
                tf.random.stateless_normal([particles, 1], folded(1), dtype=tf.float64),
                tf.random.stateless_normal([horizon, particles, 1], folded(2), dtype=tf.float64),
                tf.random.stateless_uniform([horizon, particles], folded(3), dtype=tf.float64),
                tf.random.stateless_normal([horizon, 64], folded(4), dtype=tf.float64),
                tf.random.stateless_uniform([horizon], folded(5), dtype=tf.float64),
            )

        @tf.function(input_signature=[tf.TensorSpec([12], tf.float64)], jit_compile=True)
        def paired_bootstrap(differences):
            indices = tf.random.stateless_uniform([2000, 12], [20261016, 99001],
                                                 minval=0, maxval=12, dtype=tf.int32)
            means = tf.sort(tf.reduce_mean(tf.gather(differences, indices), axis=1))
            return tf.reduce_mean(differences), tf.gather(means, [49, 1949])

        print("GPU memory policy verified; compiling the shared six-direction consumer", flush=True)
        guided = make_guided_reference_kernel(particles, horizon)

        def evaluate_guided(theta, observations, inputs, precision, linear, kernel=None):
            check_budget()
            if kernel is None:
                kernel = guided
            tick = time.monotonic()
            output = kernel(theta, directions, observations, inputs[0], inputs[1], precision, linear)
            if not bool(output["valid"]) or not bool(output["value_invariant"]):
                raise ValueError("Shared LEDH finite-value or direction-invariance veto")
            diagnostics = materialize_control_diagnostics(output["outputs"][2])
            return {
                "value": float(output["value"]), "score": output["score"].numpy().tolist(),
                "wall_seconds": time.monotonic() - tick, "diagnostics": diagnostics,
            }

        theta = reference_theta(0.9)
        observations = data(theta, tf.constant([20261016, 1000], tf.int32))
        inputs = random_inputs(tf.constant([20261016, 10000], tf.int32))
        exact = make_exact_future_information(horizon, 0.9)
        exact_precision, exact_linear = exact(observations[:, 0])
        reference_value, reference_score, *_ = kalman(observations, *parameterized_model(theta, 1, 1))
        base_safety = evaluate_guided(theta, observations, inputs, exact_precision, exact_linear)
        safety = {"base": base_safety, "reference_score": reference_score.numpy().tolist(),
                  "reference_value": float(reference_value), "variants": []}
        write_json(args.output / "safety.json", safety)
        previous = base_safety
        previous_change = None
        flow_tolerance = 0.05 * (1 + abs(float(reference_score[1])))
        for setting in (48, 96, 192):
            check_budget()
            print(f"Numerical sensitivity: flow_substeps={setting}", flush=True)
            controls = {**REFERENCE_CONTROLS, "flow_substeps": setting}
            kernel = make_guided_reference_kernel(particles, horizon, controls)
            row = evaluate_guided(theta, observations, inputs, exact_precision, exact_linear, kernel)
            change = abs(row["score"][1] - previous["score"][1])
            row.update(control="flow_substeps", setting=setting,
                       adjacent_log_q_score_change=change, tolerance=flow_tolerance)
            safety["variants"].append(row)
            write_json(args.output / "safety.json", safety)
            if change <= flow_tolerance and (previous_change is None or change <= previous_change):
                guided, base_safety = kernel, row
                break
            previous, previous_change = row, change
        else:
            raise ValueError("Numerical sensitivity veto: flow refinement is unresolved at 192 steps")
        manifest["controls"] = controls
        safety["selected_flow_substeps"] = controls["flow_substeps"]
        print("Numerical sensitivity: reset_ridge=1e-7 at the frozen flow resolution", flush=True)
        ridge_kernel = make_guided_reference_kernel(
            particles, horizon, {**controls, "reset_ridge": 1e-7})
        ridge_row = evaluate_guided(theta, observations, inputs, exact_precision, exact_linear, ridge_kernel)
        ridge_change = abs(ridge_row["score"][1] - base_safety["score"][1])
        ridge_tolerance = 1e-3 * (1 + abs(float(reference_score[1])))
        ridge_row.update(control="reset_ridge", setting=1e-7,
                         log_q_score_change=ridge_change, tolerance=ridge_tolerance)
        safety["variants"].append(ridge_row)
        write_json(args.output / "safety.json", safety)
        if ridge_change > ridge_tolerance:
            raise ValueError("Numerical sensitivity veto: ridge dependence is unresolved")
        safety["status"] = "passed_bounded_reference_sensitivity"
        write_json(args.output / "safety.json", safety)
        manifest["status"] = "running_partitions"
        write_json(args.output / "manifest.json", manifest)

        conditions = [("independent", 0.0, 0.0), ("informative", 0.9, 0.9),
                      ("mismatched_pilot", 0.9, 0.2)]
        all_rows, selections = [], {}
        with (args.output / "rows.jsonl").open("w") as output_file:
            for condition_index, (condition, alpha, pilot_alpha) in enumerate(conditions):
                theta = reference_theta(alpha)
                fitter = make_scalar_fit(horizon, 64, pilot_alpha)
                oracle = make_exact_future_information(horizon, pilot_alpha)
                selected = None
                for partition, count, offset in (("calibration", 4, 0),
                                                  ("validation", 4, 100),
                                                  ("untouched", 12, 200)):
                    if partition == "validation":
                        calibration = [row for row in all_rows
                                       if row["condition"] == condition and row["partition"] == "calibration"]
                        losses = {
                            arm: sum(row["squared_score_error"][1] for row in calibration if row["arm"] == arm) / 8
                            for arm in ("future_0.25", "future_1")}
                        selected = min(losses, key=losses.get)
                        selections[condition] = {
                            "arm": selected, "calibration_losses": losses,
                            "selection_role": "exploratory_nomination_not_canonical_tuning",
                            "frozen_before_validation_and_untouched": True,
                        }
                        write_json(args.output / "selection.json", selections)
                    for dataset in range(count):
                        check_budget()
                        identifier = (condition_index + 1) * 1000 + offset + dataset
                        observation_seed = [20261016, identifier]
                        observations = data(theta, tf.constant(observation_seed, tf.int32))
                        value, score, *_ = kalman(observations, *parameterized_model(theta, 1, 1))
                        for stream in range(2):
                            stream_seed = [20261016, identifier * 10 + stream]
                            inputs = random_inputs(tf.constant(stream_seed, tf.int32))
                            tick = time.monotonic()
                            fitted = fitter(observations[:, 0], inputs[3], inputs[4])
                            expected = oracle(observations[:, 0])
                            fit_error = max(float(tf.reduce_max(tf.abs(fitted["precision"] - expected[0]))),
                                            float(tf.reduce_max(tf.abs(fitted["linear"] - expected[1]))))
                            if not bool(fitted["valid"]) or bool(tf.reduce_any(fitted["projected"])) or fit_error > 1e-8:
                                raise ValueError("Controlled-SMC fit validity/oracle veto")
                            fit_seconds = time.monotonic() - tick
                            guides = {
                                "plain_ledh": (zero_precision, zero_linear),
                                "current_guide": (tf.ones_like(zero_precision) * 4.0, observations * 4.0),
                                "future_0.25": (fitted["precision"] * 0.25, fitted["linear"] * 0.25),
                                "future_1": (fitted["precision"], fitted["linear"]),
                            }
                            if partition != "calibration":
                                guides.pop("future_1" if selected == "future_0.25" else "future_0.25")
                            for arm in [*guides, "bootstrap_pf", "adapted_sis"]:
                                if arm in guides:
                                    result = evaluate_guided(theta, observations, inputs, *guides[arm])
                                else:
                                    check_budget()
                                    tick = time.monotonic()
                                    arguments = (theta, observations, inputs[0], inputs[1])
                                    actual = bootstrap(*arguments, inputs[2]) if arm == "bootstrap_pf" else adapted(*arguments)
                                    if not all(bool(tf.reduce_all(tf.math.is_finite(tensor))) for tensor in actual):
                                        raise ValueError("Particle comparator nonfinite veto")
                                    result = {"value": float(actual[0]), "score": actual[1].numpy().tolist(),
                                              "minimum_ess": float(actual[2]),
                                              "wall_seconds": time.monotonic() - tick}
                                row = {
                                    "condition": condition, "partition": partition,
                                    "dataset": dataset, "stream": stream, "arm": arm,
                                    "observation_seed": observation_seed, "stream_seed": stream_seed,
                                    "observations": observations.numpy().tolist(),
                                    "theta": theta.numpy().tolist(), "pilot_alpha": pilot_alpha,
                                    "reference_value": float(value), "reference_score": score.numpy().tolist(),
                                    "value_error": result["value"] - float(value),
                                    "squared_score_error": ((tf.constant(result["score"], tf.float64) - score)**2).numpy().tolist(),
                                    "fit_seconds": fit_seconds if arm.startswith("future_") else 0.0,
                                    "fit": {key: tensor.numpy().tolist() for key, tensor in fitted.items()} if arm.startswith("future_") else None,
                                    "fit_oracle_max_error": fit_error,
                                    **result,
                                }
                                all_rows.append(row)
                                output_file.write(json.dumps(row, allow_nan=False) + "\n")
                                output_file.flush()
                        print(f"{condition} {partition} dataset {dataset + 1}/{count}; {time.monotonic() - started:.1f}s", flush=True)

        summaries = {}
        for condition, *_ in conditions:
            rows = [row for row in all_rows if row["condition"] == condition and row["partition"] == "untouched"]
            selected = selections[condition]["arm"]
            arms = sorted({row["arm"] for row in rows})
            per_dataset = {
                arm: [sum(row["squared_score_error"][1] for row in rows
                          if row["arm"] == arm and row["dataset"] == dataset) / 2
                      for dataset in range(12)]
                for arm in arms
            }
            comparisons = {}
            for arm in arms:
                if arm == selected:
                    continue
                differences = tf.constant([candidate - comparator for candidate, comparator
                                           in zip(per_dataset[selected], per_dataset[arm])], tf.float64)
                average, interval = paired_bootstrap(differences)
                comparisons[arm] = {
                    "paired_mse_difference": float(average),
                    "exploratory_bootstrap_95_interval": interval.numpy().tolist(),
                    "descriptive_heuristic_loss": float(average) > 0,
                }
            any_loss = any(row["descriptive_heuristic_loss"] for row in comparisons.values())
            summaries[condition] = {
                "selected": selected,
                "mean_squared_log_q_score_error": {arm: sum(values) / 12 for arm, values in per_dataset.items()},
                "comparisons": comparisons,
                "heuristic_dominance_verdict": "PROMOTION_VETO_DESCRIPTIVE_HEURISTIC_LOSS" if any_loss else "NO_DESCRIPTIVE_LOSS_NOT_PROMOTION_EVIDENCE",
                "ranking_status": "exploratory_intervals_only_no_confirmatory_ranking",
            }
        write_json(args.output / "result.json", {
            "status": "complete", "conditions": summaries, "selections": selections,
            "rows": len(all_rows), "hard_vetoes": [],
            "canonical_admission": False, "default_ready": False,
            "trained_kdm": False, "nonlinear_model_evidence": False,
        })
        manifest["status"] = "complete"
        manifest["trace_count"] = guided.experimental_get_tracing_count()
        manifest["gpu_memory_final"] = tf.config.experimental.get_memory_info("GPU:0")
    except Exception as error:
        manifest["status"] = "stopped"
        manifest["failure"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        manifest["wall_seconds"] = time.monotonic() - started
        manifest["remaining_budget_seconds"] = max(0.0, args.budget_seconds - manifest["wall_seconds"])
        manifest["artifact_paths"] = sorted(str(path) for path in args.output.iterdir())
        write_json(args.output / "manifest.json", manifest)


if __name__ == "__main__":
    main()
