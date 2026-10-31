#!/usr/bin/env python3
"""Separately tune and test four Gaussian surrogate-HMC mechanism controls."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("BAYESFILTER_PRELOAD_CUSTOM_OP", "0")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qualification-root", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    # Enforce the predeclared two-look attempt even for direct worker restarts.
    # Historical reports remain evidence, never fresh tuning authority.
    qualification_path = args.qualification_root.resolve()/"report.json"
    qualification = json.loads(qualification_path.read_text())
    if qualification.get("status") != "QUALIFIED_AFFORDABLE":
        raise ValueError("C01 mechanics and affordable forecast required before inference")
    import tensorflow as tf
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
    memory = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    tf.config.experimental.enable_tensor_float_32_execution(False)
    from bayesfilter.inference.surrogate_hmc_program import gaussian_attempt_history
    if gaussian_attempt_history(ROOT):
        raise ValueError("C01 attempt already has retained evidence or a terminal report; report it without new draws")
    from bayesfilter.testing.surrogate_hmc_gaussian_control import (
        ARMS, SEEDS, SCOPE, PRECISION, source_hashes, run_provenance, write_report, cpu_seconds,
        bind_control, tuning_config, reference_contract, functions_for_chunk, cost_forecast)
    from bayesfilter.inference import (
        tune_hmc_kernel, load_verified_tensorflow_hmc_result,
        build_retained_bound_hmc_archive_runner_from_tuning_result, BoundRetainedHMCArchiveConfig)
    from bayesfilter.inference.surrogate_hmc_equivalence import assess_equivalence
    if (qualification["provenance"]["scope"] != SCOPE
            or qualification["provenance"]["source_hashes"] != source_hashes()):
        raise ValueError("C01 qualified source/scope drift")
    forecast = cost_forecast({arm: qualification["arms"][arm]["timings"] for arm in ARMS})
    if not qualification["qualified"] or not forecast["affordable"] or any(
            not all(qualification["arms"][arm]["checks"].values()) for arm in ARMS):
        raise ValueError("C01 qualified mechanics or forecast invalid")
    output.mkdir(parents=True, exist_ok=False)
    started, started_cpu = time.monotonic(), cpu_seconds()
    reference = reference_contract()
    report = dict(provenance=run_provenance(output, memory, "four-arm retained Gaussian mechanism inference"),
        status="RUNNING", inference_attempt=1, arms={}, reference=reference, forecast=forecast,
        qualification=dict(path=str(qualification_path), sha256=hashlib.sha256(qualification_path.read_bytes()).hexdigest()),
        heuristic_dominance="NOT_RUN", statistically_supported_efficiency_ranking=False,
        default_readiness=False, dctr_inference_evidence=False)
    save = lambda: write_report(output/"report.json", report)
    save()
    try:
        @tf.function(input_signature=[], jit_compile=True, autograph=False)
        def independent_reference():
            return tf.random.stateless_normal([8192, 3], reference["independent_seed"], dtype=tf.float64)/tf.sqrt(
                tf.constant(PRECISION, tf.float64))
        independent = independent_reference()
        tf.debugging.assert_all_finite(independent, "nonfinite analytic reference samples")
        path = output/"independent-reference.tensor"
        tf.io.write_file(str(path), tf.io.serialize_tensor(independent))
        reference["independent_draws"] = dict(path=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            count=8192, mean=tf.reduce_mean(independent, axis=0).numpy().tolist(),
            variance=tf.math.reduce_variance(independent, axis=0).numpy().tolist(),
            role="sanity archive only; analytic expectations determine equivalence")
        save()
        for arm in ARMS:
            arm_start = time.monotonic()
            _, _, binding, adapter = bind_control(arm)
            row = report["arms"][arm] = dict(status="TUNING", chunks=[], looks=[], retained_draws=0)
            save()
            result = tune_hmc_kernel(adapter=adapter, initial_position=tf.zeros([3], tf.float64),
                parameter_scales=tf.math.rsqrt(tf.constant(PRECISION, tf.float64)),
                config=tuning_config(arm), output_dir=output/arm/"tuning", runner_binding=binding)
            row["tuning"] = dict(passed=result.passed, checks=result.checks,
                manifest=result.artifact_manifest_path, wall_seconds=time.monotonic()-arm_start)
            if not result.passed:
                row["status"] = "TUNING_REJECTED"
                save()
                print(arm, "TUNING_REJECTED", flush=True)
                continue
            verified = load_verified_tensorflow_hmc_result(result.artifact_manifest_path,
                                                           adapter=adapter, runner_binding=binding)
            runner = build_retained_bound_hmc_archive_runner_from_tuning_result(
                tuning_result=verified, runner_binding=binding)
            previous, parameters, functions = None, [], []
            row["status"] = "SAMPLING"
            save()
            for index, seed in enumerate(SEEDS[arm]["retained"]):
                chunk_root = output/arm/f"chunk-{index+1:02d}"
                tick = time.monotonic()
                archive = runner.run(BoundRetainedHMCArchiveConfig(250, seed, chunk_root,
                    "C01 fixed-kernel retained inference; every warmup and verification draw excluded", previous))
                previous = archive["archive_manifest_path"]
                row["chunks"].append(dict(manifest=previous, health_passed=archive["health_passed"],
                    wall_seconds=time.monotonic()-tick, seed=seed))
                row["retained_draws"] = (index+1)*250
                if not archive["health_passed"]:
                    row["status"] = "HEALTH_VETO"
                    save()
                    break
                manifest = json.loads(Path(previous).read_text())
                sample_path = chunk_root/manifest["raw_samples"]["path"]
                if hashlib.sha256(sample_path.read_bytes()).hexdigest() != manifest["raw_samples"]["sha256"]:
                    raise ValueError("retained sample hash mismatch")
                samples = tf.io.parse_tensor(tf.io.read_file(str(sample_path)), tf.float64)
                w, values = functions_for_chunk(samples)
                parameters.append(w)
                functions.append(values)
                if index+1 in (4, 8):
                    look = assess_equivalence(tf.transpose(tf.concat(parameters, axis=0), [1, 0, 2]),
                        tf.transpose(tf.concat(functions, axis=0), [1, 0, 2]),
                        tf.constant(reference["expectations"], tf.float64),
                        tf.constant(reference["allowance"], tf.float64),
                        tf.constant(reference["margins"], tf.float64), comparison_families=4,
                        parameter_dimension=3, function_count=15, analytic_reference=True)
                    row["looks"].append(look)
                    row["status"] = look["status"]
                    print(arm, "LOOK", row["retained_draws"], look["status"],
                          "diagnostics", look["diagnostic_valid"], flush=True)
                row["wall_seconds"] = time.monotonic()-arm_start
                save()
                if row["status"] in ("EQUIVALENT", "DISCREPANCY"):
                    break
            row["closed"] = True
            save()
        report["status"] = "COMPLETED"
        report["closed"] = True
        report["comparisons"] = {arm+"/analytic": report["arms"][arm]["status"] for arm in ARMS}
    except Exception as exc:
        report.update(status="INFRASTRUCTURE_FAILURE", error=repr(exc))
        raise
    finally:
        report.update(wall_seconds=time.monotonic()-started, cpu_worker_seconds=cpu_seconds()-started_cpu,
                      allocator=tf.config.experimental.get_memory_info("GPU:0"))
        save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
