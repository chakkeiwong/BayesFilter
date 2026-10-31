#!/usr/bin/env python3
"""Qualify a full-prior actual-SV20 posterior reference within fixed budgets."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ["BAYESFILTER_PRELOAD_CUSTOM_OP"] = "0"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PLAN = "docs/plans/surrogate-hmc-sv20-posterior-2026-09-20.md"


def save(path, value):
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cpu_seconds():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime+usage.ru_stime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--wall-budget", type=float, default=600.)
    parser.add_argument("--cpu-budget", type=float, default=900.)
    args = parser.parse_args()
    if not (math.isfinite(args.wall_budget) and 0 < args.wall_budget <= 600
            and math.isfinite(args.cpu_budget) and 0 < args.cpu_budget <= 900):
        parser.error("budget exceeds the approved stage or is invalid")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    begun = time.monotonic()
    report = dict(status="RUNNING", qualified=False, scope="SV20-attempt1", plan=PLAN,
        surrogate_hmc_tested=False, grids=[], checks={}, heuristic_dominance="NOT_RUN")
    exit_code = 1
    try:
        parent = args.reference_root.resolve()
        preparation = json.loads((parent/"report.json").read_text())
        provenance = json.loads((parent/"run-manifest.json").read_text())
        if preparation.get("status") != "QUALIFIED_LIKELIHOOD_PROBES" or not preparation.get("qualified"):
            raise ValueError("qualified likelihood reference required")
        for name, expected in provenance["source_hashes"].items():
            if digest(ROOT/name) != expected:
                raise ValueError("qualified likelihood source drift: "+name)
        data_path = parent/"claim/data.json"
        if digest(data_path) != preparation["partitions"]["claim"]["data_sha256"]:
            raise ValueError("claim data hash mismatch")
        dataset = json.loads(data_path.read_text())
        if (dataset["model_id"], dataset["attempt"], dataset["partition"], dataset["horizon"]) != ("SV20", 1, "claim", 20):
            raise ValueError("wrong SV20 claim dataset")
        shutil.copy2(data_path, output/"data.json")
        import tensorflow as tf
        from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
        memory = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
        tf.config.experimental.enable_tensor_float_32_execution(False)
        from bayesfilter.testing.surrogate_hmc_sv_posterior_reference import (
            make_statistics, normalize_likelihood, assess_refinement, BATCH_SIZE,
            PARAMETER_INTERVALS, LATENT_GRIDS, FUNCTION_NAMES, MARGINS)
        observations = tf.constant(dataset["raw_observations"], tf.float64)
        tf.debugging.assert_all_finite(observations, "claim observations")
        names = [PLAN, "scripts/prepare_surrogate_hmc_sv_posterior.py",
            "bayesfilter/testing/surrogate_hmc_sv_posterior_reference.py",
            "bayesfilter/testing/surrogate_hmc_sv_reference.py",
            "bayesfilter/testing/surrogate_hmc_linear_posterior_reference.py",
            "bayesfilter/testing/surrogate_hmc_linear_reference.py",
            "tests/test_surrogate_hmc_sv_posterior_reference.py"]
        hashes = {}
        for name in names:
            hashes[name] = digest(ROOT/name)
            target = output/"source"/name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT/name, target)
        manifest = dict(git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            command=[sys.executable, *sys.argv], environment=sys.prefix, source_hashes=hashes,
            plan=PLAN, result_file=str(output/"report.json"), output_artifact_paths=[str(output)],
            data_sha256=digest(data_path), data_version="SV20-attempt1 claim, raw FP32 values promoted to FP64",
            seeds=provenance["seeds"]["claim"], memory_policy=memory,
            gpu=True, dtype="float64", tf32=False, jit_compile=True, batch_size=BATCH_SIZE,
            scalar_fallback=False, cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
            device=tf.config.list_logical_devices("GPU")[0].name, tensorflow=tf.__version__,
            parent_reference=str(parent), parent_manifest_sha256=digest(parent/"run-manifest.json"))
        save(output/"run-manifest.json", manifest)

        def check_resources():
            if time.monotonic()-begun >= args.wall_budget or cpu_seconds() >= args.cpu_budget:
                raise RuntimeError("reference stage budget exhausted")
            if tf.config.experimental.get_memory_info("GPU:0")["peak"] >= 4*1024**3:
                raise RuntimeError("reference allocator exceeds 4 GiB")

        kernels, timing = [], []
        with tf.device("/GPU:0"):
            probe = tf.tile(tf.constant([[.1, .1], [.1, .9], [.9, .1], [.9, .9]], tf.float64), [BATCH_SIZE//4, 1])
            for intervals, half_width in LATENT_GRIDS:
                kernel = make_statistics(intervals=intervals, half_width=half_width)
                kernels.append(kernel)
                # Materialization synchronizes GPU before each timing measurement.
                for value in kernel(probe, observations):
                    tf.debugging.assert_all_finite(value, "forecast output")
                    value.numpy()
                measurements = []
                for _ in range(3):
                    start, cpu_start = time.monotonic(), cpu_seconds()
                    for value in kernel(probe, observations):
                        value.numpy()
                    measurements.append((time.monotonic()-start, cpu_seconds()-cpu_start))
                timing.append(dict(wall_seconds=max(x[0] for x in measurements),
                                   cpu_worker_seconds=max(x[1] for x in measurements)))
                check_resources()
        batches = [sum(math.ceil((n+1)**2/BATCH_SIZE) for n in PARAMETER_INTERVALS)]
        batches += [math.ceil((PARAMETER_INTERVALS[-1]+1)**2/BATCH_SIZE)]*2
        forecast = {key: 60+3*sum(count*row[key] for count, row in zip(batches, timing))
                    for key in ("wall_seconds", "cpu_worker_seconds")}
        report["forecast"] = dict(remaining=forecast, batch_counts=batches, measurements=timing, factor=3.)
        save(output/"report.json", report)
        if (forecast["wall_seconds"] > args.wall_budget-(time.monotonic()-begun)
                or forecast["cpu_worker_seconds"] > args.cpu_budget-cpu_seconds()):
            report["status"] = "UNDER_BUDGETED_REFERENCE"
            return 2

        grids, likelihoods = [], []
        settings = [(n, 0) for n in PARAMETER_INTERVALS]+[(PARAMETER_INTERVALS[-1], 1), (PARAMETER_INTERVALS[-1], 2)]
        for grid_index, (parameter_intervals, latent_index) in enumerate(settings):
            start = time.monotonic()
            with tf.device("/GPU:0"):
                knots = tf.linspace(tf.constant(.1, tf.float64), tf.constant(.9, tf.float64), parameter_intervals+1)
                points = tf.reshape(tf.stack(tf.meshgrid(knots, knots, indexing="ij"), -1), [-1, 2])
                values, predictions, losses = [], [], []
                for offset in range(0, int(points.shape[0]), BATCH_SIZE):
                    count = min(BATCH_SIZE, int(points.shape[0])-offset)
                    batch = points[offset:offset+count]
                    batch = tf.concat([batch, tf.repeat(batch[:1], BATCH_SIZE-count, axis=0)], 0)
                    value, prediction, loss = kernels[latent_index](batch, observations)
                    for tensor in (value, prediction, loss):
                        tf.debugging.assert_all_finite(tensor, "nonfinite posterior integrand")
                    tf.debugging.assert_greater_equal(prediction, tf.constant(-1e-12, tf.float64))
                    tf.debugging.assert_less_equal(prediction, tf.constant(1+1e-12, tf.float64))
                    values.append(value[:count]); predictions.append(prediction[:count]); losses.append(loss[:count])
                    if offset//BATCH_SIZE % 32 == 0:
                        check_resources()
                shape = [parameter_intervals+1]*2
                likelihood = tf.reshape(tf.concat(values, 0), shape)
                prediction = tf.reshape(tf.concat(predictions, 0), shape+[2])
                density, evidence = normalize_likelihood(knots, knots, likelihood)
                tf.debugging.assert_all_finite(density, "posterior density")
                tf.debugging.assert_all_finite(evidence, "model evidence")
                grids.append((knots, knots, density, prediction))
                likelihoods.append(likelihood)
                maximum_loss = float(tf.reduce_max(tf.concat(losses, 0)))
            files = {}
            for name, tensor in dict(knots=knots, likelihood=likelihood, density=density, predictive=prediction).items():
                path = output/f"grid-{grid_index}-{name}.tfbin"
                path.write_bytes(tf.io.serialize_tensor(tensor).numpy())
                files[name] = dict(path=path.name, sha256=digest(path))
            report["grids"].append(dict(parameter_intervals=parameter_intervals,
                latent_grid=LATENT_GRIDS[latent_index], maximum_predicted_domain_loss=maximum_loss,
                log_evidence=float(evidence), wall_seconds=time.monotonic()-start, arrays=files))
            save(output/"report.json", report)
            print("measured-posterior-grid", grid_index, report["grids"][-1]["wall_seconds"], flush=True)
        with tf.device("/GPU:0"):
            assessment = assess_refinement(grids)
        report["numerical_refinement"] = {key:value.numpy().tolist() for key, value in assessment.items()}
        latent_changes = [float(tf.reduce_max(tf.abs(likelihoods[i]-likelihoods[i-1]))) for i in (4, 5)]
        report["maximum_full_prior_likelihood_changes"] = latent_changes
        report["checks"] = dict(function_refinement=bool(assessment["qualified"]),
            full_prior_latent_refinement=all(value <= 1e-6 for value in latent_changes),
            domain_mass=all(row["maximum_predicted_domain_loss"] <= 1e-9 for row in report["grids"][-2:]))
        report["function_names"] = list(FUNCTION_NAMES)
        report["margins"] = list(MARGINS)
        report["allocator"] = tf.config.experimental.get_memory_info("GPU:0")
        check_resources()
        for name, expected in hashes.items():
            if digest(ROOT/name) != expected:
                raise RuntimeError("source changed during posterior preparation")
        report["qualified"] = all(report["checks"].values())
        report["status"] = "QUALIFIED_MODEL_POSTERIOR_REFERENCE" if report["qualified"] else "REFERENCE_NOT_QUALIFIED"
        exit_code = 0 if report["qualified"] else 2
    except Exception as exc:
        report.update(status="FAILED", error=type(exc).__name__+": "+str(exc), qualified=False)
    finally:
        report.update(wall_seconds=time.monotonic()-begun, cpu_worker_seconds=cpu_seconds())
        save(output/"report.json", report)
        path = output/"run-manifest.json"
        if path.exists():
            manifest = json.loads(path.read_text())
            manifest.update(wall_seconds=report["wall_seconds"], cpu_worker_seconds=report["cpu_worker_seconds"], status=report["status"])
            save(path, manifest)
    print(json.dumps({key:report.get(key) for key in ("status", "qualified", "wall_seconds", "cpu_worker_seconds", "error")}))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
