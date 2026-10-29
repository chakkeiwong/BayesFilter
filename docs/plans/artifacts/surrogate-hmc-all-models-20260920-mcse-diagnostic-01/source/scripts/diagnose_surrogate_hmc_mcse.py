#!/usr/bin/env python3
"""CPU-only known-reference diagnostic of the existing surrogate-HMC MCSE veto."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import shutil
import signal
import statistics
import subprocess
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("BAYESFILTER_PRELOAD_CUSTOM_OP", "0")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PLAN = "docs/plans/surrogate-hmc-mcse-diagnostic-2026-09-20.md"


def cpu_seconds():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    return usage.ru_utime + usage.ru_stime + children.ru_utime + children.ru_stime


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def wilson(successes, count):
    z = 1.959963984540054
    p = successes / count
    denominator = 1 + z*z/count
    center = (p + z*z/(2*count))/denominator
    radius = z*math.sqrt(p*(1-p)/count + z*z/(4*count*count))/denominator
    return [max(0., center-radius), min(1., center+radius)]


def independent_batch_check(values, actual_mcse, actual_lag):
    """Independent scalar diagnostic for one mean; no runtime NumPy path."""
    for index, size in enumerate((50, 100, 200)):
        batches = 2000 // size
        means = [[statistics.fmean(chain[i:i+size]) for i in range(0, 2000, size)]
                 for chain in values]
        variance = statistics.fmean(statistics.variance(chain) for chain in means)
        centered = [[v-statistics.fmean(chain) for v in chain] for chain in means]
        numerator = sum(sum(a*b for a, b in zip(chain[1:], chain[:-1])) for chain in centered)
        denominator = sum(sum(v*v for v in chain) for chain in centered)
        expected_mcse = math.sqrt(variance/(4*batches))
        if not math.isclose(actual_mcse[index][0], expected_mcse, rel_tol=1e-11, abs_tol=1e-13):
            raise ValueError("independent MCSE formula mismatch")
        if not math.isclose(actual_lag[index][0], numerator/denominator, rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError("independent batch-lag formula mismatch")


def run(output, report):
    import tensorflow as tf
    import tensorflow_probability as tfp
    from bayesfilter.inference.surrogate_hmc_equivalence import _batch_statistics
    from bayesfilter.testing.surrogate_hmc_gaussian_control import (
        PRECISION, functions_for_chunk, reference_contract)

    @tf.function(input_signature=[tf.TensorSpec([2], tf.int32), tf.TensorSpec([], tf.float64)],
                 jit_compile=True, autograph=False)
    def generate(seed, rho):
        innovations = tf.random.stateless_normal([2000, 4, 3], seed, dtype=tf.float64)
        values = tf.cond(rho == 0., lambda: innovations,
            lambda: tf.concat([innovations[:1], tf.scan(
                lambda prior, noise: rho*prior + tf.sqrt(1-rho*rho)*noise,
                innovations[1:], initializer=innovations[0])], axis=0))
        parts = [functions_for_chunk(chunk/tf.sqrt(tf.constant(PRECISION, tf.float64)))[1]
                 for chunk in tf.split(values, 8, axis=0)]
        return tf.transpose(tf.concat(parts, axis=0), [1, 0, 2])

    contract = reference_contract()
    report["tensorflow"] = tf.__version__
    report["tensorflow_probability"] = tfp.__version__
    report["function_names"] = contract["names"]
    report["independent_formula_parity"] = False
    groups = [("iid", 0., 512, 90000), ("ar1_099", .99, 128, 100000)]
    for name, rho, count, seed_base in groups:
        rows = []
        report["groups"][name] = {"rho": rho, "requested_replications": count,
                                  "completed_replications": 0}
        with (output/(name+"-replications.jsonl")).open("x") as stream:
            for replication in range(count):
                functions = generate(tf.constant([20260920, seed_base+replication], tf.int32),
                                     tf.constant(rho, tf.float64))
                tf.debugging.assert_all_finite(functions, "nonfinite Gaussian functions")
                row = {"replication": replication, "seed": [20260920, seed_base+replication],
                       "function_tensor_sha256": hashlib.sha256(tf.io.serialize_tensor(functions).numpy()).hexdigest(),
                       "looks": {}}
                for draws in (1000, 2000):
                    values = functions[:, :draws, :]
                    mcse, lag = _batch_statistics(draws, 15)(values)
                    tf.debugging.assert_all_finite(mcse, "nonfinite MCSE")
                    tf.debugging.assert_all_finite(lag, "nonfinite batch lag")
                    tf.debugging.assert_positive(mcse)
                    if name == "iid" and replication == 0 and draws == 2000:
                        independent_batch_check(values[:, :, 0].numpy().tolist(), mcse.numpy().tolist(), lag.numpy().tolist())
                        report["independent_formula_parity"] = True
                    conservative = tf.reduce_max(mcse, axis=0)
                    sensitivity = conservative/tf.reduce_min(mcse, axis=0)
                    actual_lag = lag[-1].numpy().tolist()
                    negative = [v < -.3 for v in actual_lag]
                    positive = [v > .3 for v in actual_lag]
                    truth = math.sqrt((1+2*sum((1-k/draws)*rho**k for k in range(1, draws)))/(4*draws))
                    naive = tf.math.reduce_std(values[:, :, :3], axis=[0, 1])/math.sqrt(4*draws)
                    row["looks"][str(draws)] = dict(
                        lag=actual_lag, negative=negative, positive=positive,
                        any_lag_veto=any(negative) or any(positive),
                        mcse_sensitivity=sensitivity.numpy().tolist(),
                        any_sensitivity_veto=bool(tf.reduce_any(sensitivity > 2.)),
                        mean_mcse_over_truth=(conservative[:3]/truth).numpy().tolist(),
                        naive_mean_mcse_over_truth=(naive/truth).numpy().tolist(),
                        exact_finite_mean_mcse=truth, iid_theoretical_mean_mcse=1/math.sqrt(4*draws))
                row["any_look_lag_veto"] = any(v["any_lag_veto"] for v in row["looks"].values())
                stream.write(json.dumps(row, allow_nan=False)+"\n")
                stream.flush()
                rows.append(row)
                report["groups"][name]["completed_replications"] = len(rows)
                if len(rows) % 64 == 0:
                    save(output/"report.json", report)
                    print(name, len(rows), "replications", flush=True)
        successes = sum(row["any_look_lag_veto"] for row in rows)
        group = report["groups"][name]
        group.update(any_look_lag_veto_count=successes, any_look_lag_veto_rate=successes/count,
                     wilson95=wilson(successes, count), looks={})
        for draws in (1000, 2000):
            looks = [row["looks"][str(draws)] for row in rows]
            group["looks"][str(draws)] = dict(
                lag_veto_rate=statistics.fmean(float(v["any_lag_veto"]) for v in looks),
                negative_veto_rate=statistics.fmean(float(any(v["negative"])) for v in looks),
                positive_veto_rate=statistics.fmean(float(any(v["positive"])) for v in looks),
                sensitivity_veto_rate=statistics.fmean(float(v["any_sensitivity_veto"]) for v in looks),
                per_function_lag_veto_rate=[statistics.fmean(float(v["negative"][j] or v["positive"][j]) for v in looks)
                                           for j in range(15)],
                average_lag=[statistics.fmean(v["lag"][j] for v in looks) for j in range(15)],
                mean_mcse_over_truth=[statistics.fmean(v["mean_mcse_over_truth"][j] for v in looks) for j in range(3)],
                naive_mean_mcse_over_truth=[statistics.fmean(v["naive_mean_mcse_over_truth"][j] for v in looks) for j in range(3)])
    lower, upper = report["groups"]["iid"]["wilson95"]
    report["calibration_verdict"] = ("EXCESSIVE_IID_FALSE_VETO" if lower > .05 else
        "PASSES_NARROW_IID_SCREEN" if upper < .05 else "UNRESOLVED")
    report["status"] = "COMPLETED"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    # Charge all process CPU, including Python startup, imports and git provenance.
    prior = 0.
    for accounting in (ROOT/"docs/plans/artifacts").glob("surrogate-hmc-all-models-20260920-mcse-diagnostic-*/accounting.json"):
        item = json.loads(accounting.read_text())
        prior += item.get("charged_this_run", item.get("charged", {})).get("cpu_worker_seconds", 0.)
    remaining = 300.-prior
    if remaining < 15.:
        raise RuntimeError("aggregate diagnostic CPU allocation exhausted")
    def stop(signum, frame):
        raise TimeoutError("bounded diagnostic CPU/wall allocation exhausted")
    signal.signal(signal.SIGXCPU, stop)
    signal.signal(signal.SIGALRM, stop)
    resource.setrlimit(resource.RLIMIT_CPU, (max(1, int(remaining)-5), max(2, int(remaining))))
    signal.alarm(330)
    report = dict(status="RUNNING", groups={}, cpu_only="CUDA_VISIBLE_DEVICES=-1; GPUs intentionally hidden",
        plan=PLAN, jit_compile=True, dtype="float64", no_hmc_sampling=True,
        full_equivalence_calibration=False, completed_hmc_criteria_changed=False,
        heuristic_dominance="DIAGNOSTIC_REFERENCES_ONLY; no sampler comparison",
        prior_diagnostic_cpu_seconds=prior, remaining_diagnostic_cpu_seconds=remaining)
    exit_code = 0
    try:
        names = ["scripts/diagnose_surrogate_hmc_mcse.py", PLAN,
                 "bayesfilter/testing/surrogate_hmc_gaussian_control.py",
                 "bayesfilter/inference/surrogate_hmc_equivalence.py",
                 "bayesfilter/inference/hmc_posterior_diagnostics.py"]
        hashes = {}
        for name in names:
            path = ROOT/name
            hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
            destination = output/"source"/name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
        manifest = dict(git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            command=[sys.executable, *sys.argv], environment=sys.prefix, cpu_only=report["cpu_only"],
            gpu=False, plan=PLAN, result_file=str(output/"report.json"), source_hashes=hashes,
            seeds={"iid": [20260920, [90000, 90511]], "ar1_099": [20260920, [100000, 100127]]},
            data_version="stateless synthetic stationary Gaussian; no model or HMC data",
            output_artifact_paths=[str(output/"report.json"), str(output/"iid-replications.jsonl"),
                                   str(output/"ar1_099-replications.jsonl")])
        save(output/"run-manifest.json", manifest)
        run(output, report)
        for name, digest in hashes.items():
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != digest:
                raise RuntimeError("source changed during diagnostic")
    except Exception as exc:
        report.update(status="FAILED", error=type(exc).__name__+": "+str(exc))
        exit_code = 1
    finally:
        signal.alarm(0)
        elapsed, charged = time.monotonic()-started, cpu_seconds()
        report.update(wall_seconds=elapsed, cpu_worker_seconds=charged)
        save(output/"report.json", report)
        accounting = dict(command=[sys.executable, *sys.argv], exit_code=exit_code, plan=PLAN,
            cpu_only=report["cpu_only"], wall_seconds=elapsed,
            charged_this_run=dict(gpu_seconds=0., cpu_worker_seconds=charged))
        save(output/"accounting.json", accounting)
        manifest_path = output/"run-manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            manifest.update(wall_seconds=elapsed, cpu_worker_seconds=charged, status=report["status"])
            save(manifest_path, manifest)
    print(json.dumps({k:report.get(k) for k in ("status", "calibration_verdict", "wall_seconds", "cpu_worker_seconds", "error")}))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
