#!/usr/bin/env python3
"""Freeze SV20 data and qualify an independent actual-SV likelihood reference."""
import argparse
import hashlib
import json
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
PLAN = "docs/plans/surrogate-hmc-sv20-reference-2026-09-20.md"


def save(path, value):
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    begun = time.monotonic()
    report = dict(status="RUNNING", checks={}, grids=[], partitions={}, posterior_reference_qualified=False,
        retained_inference=False, heuristic_dominance="NOT_RUN", scope="SV20-attempt1",
        classification="independent likelihood reference preparation", plan=PLAN)
    exit_code = 0
    try:
        import tensorflow as tf
        from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
        memory = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
        tf.config.experimental.enable_tensor_float_32_execution(False)
        from bayesfilter.testing.surrogate_hmc_sv_reference import (
            GRID_LADDER, PARTITION_SEEDS, PROBES, generate_dataset, make_sv_reference,
            direct_short_likelihood, chart_and_log_prior)
        names = [PLAN, "scripts/prepare_surrogate_hmc_sv_reference.py",
                 "bayesfilter/testing/surrogate_hmc_sv_reference.py",
                 "tests/test_surrogate_hmc_sv_reference.py",
                 "bayesfilter/testing/exact_sv_sgqf_neutra_target_tf.py"]
        hashes = {}
        for name in names:
            hashes[name] = hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
            destination = output/"source"/name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT/name, destination)
        manifest = dict(git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            command=[sys.executable, *sys.argv], environment=sys.prefix, source_hashes=hashes,
            plan=PLAN, result_file=str(output/"report.json"), output_artifact_paths=[str(output)],
            data_version="SV20-attempt1; independent calibration/validation/claim; FP32 raw observations",
            seeds={name: [[20260920, base+j] for j in range(3)] for name, base in PARTITION_SEEDS.items()},
            gpu=True, cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
            memory_policy=memory, dtype="float64", tf32=False, jit_compile=True,
            exceptions=["FP64 independent reference", "one-off CPU/eager independent Hermite check"],
            tensorflow=tf.__version__, device=tf.config.list_logical_devices("GPU")[0].name)
        save(output/"run-manifest.json", manifest)
        report["memory_policy"] = memory
        datasets = {}
        for partition, base in PARTITION_SEEDS.items():
            with tf.device("/CPU:0"):
                initial, states, raw = generate_dataset(tf.constant([20260920, base], tf.int32))
            tf.debugging.assert_all_finite(raw, "raw observations")
            tf.debugging.assert_positive(tf.abs(raw), "zero raw observation")
            data = dict(model_id="SV20", attempt=1, replicate=0, horizon=20, partition=partition,
                raw_observations=raw.numpy().tolist(), initial_state=float(initial), states=states.numpy().tolist(),
                data_seed=[20260920, base], truth=[.6, .4], sigma=1., observation_dtype="float32",
                law="stationary x0; transition then Normal(0,beta^2*exp(x)) observation",
                prior="independent Uniform(.1,.9)", chart="gamma,beta=.1+.8*Phi(z)")
            path = output/partition
            path.mkdir()
            save(path/"data.json", data)
            report["partitions"][partition] = dict(data_file=str(path/"data.json"),
                data_sha256=hashlib.sha256((path/"data.json").read_bytes()).hexdigest())
            datasets[partition] = tf.cast(raw, tf.float64)
        observations = datasets["calibration"]
        probes = tf.constant(PROBES, tf.float64)
        results = []
        for intervals, half_width in GRID_LADDER:
            start = time.monotonic()
            with tf.device("/GPU:0"):
                reference = make_sv_reference(intervals=intervals, half_width=half_width)
                batches = [reference.physical_value_score(probes[i:i+4], observations) for i in range(0, 12, 4)]
            for batch in batches:
                for value in batch.values():
                    tf.debugging.assert_all_finite(value, "nonfinite reference output")
                tf.debugging.assert_near(tf.reduce_sum(batch["final_mass"], axis=1), tf.ones([4], tf.float64), atol=1e-12)
                tf.debugging.assert_greater_equal(batch["final_mass"], tf.constant(0., tf.float64))
                tf.debugging.assert_near(tf.reduce_sum(batch["final_mass_derivative"], axis=1), tf.zeros([4, 2], tf.float64), atol=1e-10)
            result = {name: tf.concat([v[name] for v in batches], 0) for name in (
                "log_likelihood", "score", "maximum_predicted_domain_loss")}
            files = {}
            for name, tensor in result.items():
                filename = f"grid-{intervals}-{int(half_width)}-{name}.tfbin"
                payload = tf.io.serialize_tensor(tensor).numpy()
                (output/filename).write_bytes(payload)
                files[name] = dict(path=filename, sha256=hashlib.sha256(payload).hexdigest())
            row = dict(intervals=intervals, half_width=half_width, wall_seconds=time.monotonic()-start,
                maximum_predicted_domain_loss=float(tf.reduce_max(result["maximum_predicted_domain_loss"])), arrays=files)
            report["grids"].append(row)
            results.append(result)
            save(output/"report.json", report)
            print("measured-grid", intervals, half_width, row["wall_seconds"], flush=True)
        for name, coarse, fine in (("mesh_refinement", results[-3], results[-2]),
                                   ("domain_refinement", results[-2], results[-1])):
            value_error = float(tf.reduce_max(tf.abs(coarse["log_likelihood"]-fine["log_likelihood"])))
            score_error = float(tf.reduce_max(tf.abs(coarse["score"]-fine["score"])/(1+tf.abs(fine["score"]))))
            report[name] = dict(maximum_log_likelihood_change=value_error, maximum_relative_score_change=score_error)
            report["checks"][name] = value_error <= 1e-6 and score_error <= 1e-5
        report["checks"]["domain_mass"] = report["grids"][-1]["maximum_predicted_domain_loss"] <= 1e-9

        # Independent short Gaussian integration uses one-off CPU/eager reference
        # construction. The repeated filtering kernels above execute on GPU/XLA.
        selected = tf.constant([[.2, .8], [.6, .4], [.9, .1], [.9, .9]], tf.float64)
        report["short_reference"] = {}
        for horizon in (1, 2):
            with tf.device("/CPU:0"):
                lower = direct_short_likelihood(selected, observations[:horizon], order=256)
                higher = direct_short_likelihood(selected, observations[:horizon], order=512)
            with tf.device("/GPU:0"):
                short = make_sv_reference(intervals=1024, half_width=16., horizon=horizon)
                actual = short.physical_value_score(selected, observations[:horizon])["log_likelihood"]
            for value in (lower, higher, actual):
                tf.debugging.assert_all_finite(value, "nonfinite independent short reference")
            refinement = float(tf.reduce_max(tf.abs(lower-higher)))
            error = float(tf.reduce_max(tf.abs(actual-higher)))
            report["short_reference"][str(horizon)] = dict(hermite_refinement=refinement, grid_difference=error)
            report["checks"][f"independent_T{horizon}"] = refinement <= 1e-6 and error <= 1e-6

        with tf.device("/GPU:0"):
            final = make_sv_reference(intervals=1024, half_width=16.)
            baseline = final.physical_value_score(selected, observations)
            replay = final.physical_value_score(selected, observations)
            reordered = final.physical_value_score(tf.reverse(selected, [0]), observations)
            maximum_gradient_error = 0.
            for coordinate in (0, 1):
                step = tf.one_hot(coordinate, 2, dtype=tf.float64)*1e-6
                plus = final.physical_value_score(selected+step, observations)["log_likelihood"]
                minus = final.physical_value_score(selected-step, observations)["log_likelihood"]
                error = tf.reduce_max(tf.abs((plus-minus)/2e-6-baseline["score"][:, coordinate])/(1+tf.abs(baseline["score"][:, coordinate])))
                tf.debugging.assert_all_finite(error, "nonfinite finite-difference diagnostic")
                maximum_gradient_error = max(maximum_gradient_error, float(error))
            report["maximum_finite_difference_relative_error"] = maximum_gradient_error
            report["checks"]["total_score_finite_difference"] = maximum_gradient_error <= 1e-5
            report["checks"]["replay"] = all(bool(tf.reduce_all(baseline[k] == replay[k])) for k in baseline)
            report["checks"]["row_independence"] = all(bool(tf.reduce_all(baseline[k] == tf.reverse(reordered[k], [0]))) for k in baseline)
            theta = tf.constant([[-1., .2], [.3, -.5], [1., -1.], [0., .7]], tf.float64)
            physical, derivative, prior = chart_and_log_prior(theta)
            physical_result = final.physical_value_score(physical, observations)
            value, score = final.unconstrained_value_score(theta, observations)
            report["checks"]["chart_prior_chain_rule"] = (
                bool(tf.reduce_all(tf.abs(value-physical_result["log_likelihood"]-prior) < 1e-10)) and
                bool(tf.reduce_all(tf.abs(score-(physical_result["score"]*derivative-theta)) < 1e-9)))
        report["allocator"] = tf.config.experimental.get_memory_info("GPU:0")
        report["probes"] = [list(p) for p in PROBES]
        report["qualified"] = all(report["checks"].values())
        report["status"] = "QUALIFIED_LIKELIHOOD_PROBES" if report["qualified"] else "REFERENCE_NOT_QUALIFIED"
        report["next_action"] = "Full-prior posterior refinement and separate actual-SV DCTR consumer qualification"
        for name, digest in hashes.items():
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != digest:
                raise RuntimeError("source changed during reference preparation")
        if not report["qualified"]:
            exit_code = 2
    except Exception as exc:
        report.update(status="FAILED", error=type(exc).__name__+": "+str(exc), qualified=False)
        exit_code = 1
    finally:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        report.update(wall_seconds=time.monotonic()-begun, cpu_worker_seconds=usage.ru_utime+usage.ru_stime)
        save(output/"report.json", report)
        path = output/"run-manifest.json"
        if path.exists():
            manifest = json.loads(path.read_text())
            manifest.update(wall_seconds=report["wall_seconds"], cpu_worker_seconds=report["cpu_worker_seconds"],
                            status=report["status"], partitions=report["partitions"])
            save(path, manifest)
    print(json.dumps({k:report.get(k) for k in ("status", "qualified", "wall_seconds", "cpu_worker_seconds", "error")}))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
