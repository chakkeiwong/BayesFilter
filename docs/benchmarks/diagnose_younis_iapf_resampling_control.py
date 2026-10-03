"""Diagnostic only: independently calibrated, exactly centered iAPF controls.

Particle kernels are analytical TF/XLA. FP64 controls and regression are an
explicit diagnostic precision exception. This driver issues no tuning or
admission artifact and never interprets the control as an HMC gradient.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import diagnose_younis_iapf_curved as base

PLAN = "docs/plans/younis-iapf-resampling-control-2026-09-18.md"
DATASETS = {1490: "affine", 1500: "weak", 1501: "weak", 1510: "curved", 1511: "curved"}
CALIBRATION, FINAL, UNIFORM_BITS = 192, 128, 23
HEURISTICS = ("bootstrap_fisher", "no_resampling", "ekf", "ukf")


@lru_cache(maxsize=2)
def stream_kernel(n):
    @tf.function(input_signature=[tf.TensorSpec([4, 2], tf.int32)], jit_compile=True)
    def kernel(seeds):
        ancestors = tf.random.stateless_uniform([3, n], seeds[2],
            minval=0, maxval=2**UNIFORM_BITS, dtype=tf.int32)
        return (tf.random.stateless_normal([n, 1], seeds[0]),
                tf.random.stateless_normal([2, n, 1], seeds[1]),
                tf.cast(ancestors, tf.float32) * tf.constant(2.**-UNIFORM_BITS),
                tf.random.stateless_uniform([2, n], seeds[3]))
    return kernel


def streams(dataset, n, replicate, partition):
    group = f"resampling_control_20260918_{partition}_N{n}"
    seeds = [base.seed(dataset, label, replicate, group)
             for label in ("initial", "process", "ancestors", "mixture")]
    return stream_kernel(n)(tf.constant(seeds, tf.int32))


@lru_cache(maxsize=1)
def interval_kernel():
    @tf.function(input_signature=[tf.TensorSpec([FINAL], tf.float64)], jit_compile=True)
    def kernel(differences):
        indices = tf.random.stateless_uniform([20000, FINAL], [923, 2026],
            minval=0, maxval=FINAL, dtype=tf.int32)
        means = tf.sort(tf.reduce_mean(tf.gather(differences, indices), 1))
        return tf.stack([means[24], means[19974], means[99], means[19899]])
    return kernel


def finite(*outputs):
    for tensor in outputs:
        tf.debugging.assert_all_finite(tensor, "invalid diagnostic output")


def squared_error(score, oracle):
    return sum((a-b)**2 for a, b in zip(score, oracle["score"]))


def summarize(rows, oracle, critical):
    summary = base.summarize(rows, oracle)
    summary["total_score_variance"] = sum(statistics.variance(column)
        for column in zip(*(row["score"] for row in rows)))
    intervals = [[error-critical*se-5e-6*(1+abs(target)), error+critical*se+5e-6*(1+abs(target))]
        for error, se, target in zip(summary["mean_score_error"], summary["score_standard_error"], oracle["score"])]
    summary["mean_error_intervals"] = intervals
    summary["bias_screen_pass"] = all(lo <= 0 <= hi for lo, hi in intervals)
    summary["bias_screen_scope"] = "approximate Student-t Bonferroni 99%, 24 components per method/rung; separate FP32 allowance"
    return summary


def calibrate(dataset, n, kernel, arguments, budget, calibration, save):
    from bayesfilter.score_study.combinations_tf import make_combination_kernels
    fit, apply, _, _ = make_combination_kernels(6, 12, "float64")
    rows = calibration.setdefault("rows", [])
    if "coefficient" in calibration:
        if len(rows) != CALIBRATION:
            raise RuntimeError("incomplete frozen calibration")
        return tf.constant(calibration["coefficient"], tf.float64), apply
    theta, obs, coefficients = arguments
    for rep in range(len(rows), CALIBRATION):
        budget.charge("filter_calls")
        out = kernel(theta, obs, *streams(dataset, n, rep, "calibration"), *coefficients)
        finite(*out)
        rows.append(dict(replicate=rep, score=base.materialize(out[3]),
            control=base.materialize(tf.reshape(out[4], [12]))))
        if (rep+1) % 32 == 0:
            save()
    scores = tf.constant([row["score"] for row in rows], tf.float64)
    controls = tf.constant([row["control"] for row in rows], tf.float64)
    coefficient, rank, valid = fit(scores, controls)
    if not bool(valid):
        raise RuntimeError("invalid control coefficient computation")
    corrected = apply(scores, controls, coefficient)
    finite(coefficient, corrected)
    singular = tf.linalg.svd(controls-tf.reduce_mean(controls, 0), compute_uv=False)
    retained = int(rank)
    calibration.update(coefficient=base.materialize(coefficient), rank=retained,
        singular_values=base.materialize(singular),
        retained_condition_number=float(singular[0]/singular[retained-1]) if retained else None,
        raw_variance=float(tf.reduce_sum(tf.math.reduce_variance(scores, 0))),
        corrected_variance=float(tf.reduce_sum(tf.math.reduce_variance(corrected, 0))),
        max_coefficient=float(tf.reduce_max(tf.abs(coefficient))),
        known_control_mean=[0.]*12, oracle_used=False,
        frozen_before_final=True, fit_trace_count=fit.experimental_get_tracing_count())
    save()
    return coefficient, apply


def evaluate(dataset, regime, observations, oracle, fitted, budget, critical, result, save):
    from bayesfilter.score_study.fitted_twist_tf import make_fitted_twist_kernel
    from bayesfilter.score_study.nonlinear_tf import make_particle_filter, make_moment_filter
    c, b = base.REGIMES[regime]
    theta, obs = tf.constant(base.THETA, tf.float32), tf.cast(observations, tf.float32)
    for name in ("ekf", "ukf"):
        if name in result["deterministic"]:
            continue
        kernel = make_moment_filter(2, c, b, name, "float32")
        outputs = []
        for direction in tf.unstack(tf.eye(6)):
            budget.charge("filter_calls")
            output = kernel(theta, direction, obs)
            finite(*output)
            outputs.append(output)
        score = [base.materialize(out[1]) for out in outputs]
        result["deterministic"][name] = dict(value=base.materialize(outputs[0][0]),
            score=score, score_squared_error=squared_error(score, oracle))
        save()
    usable_fit = fitted["status"] == "valid"
    coefficients = ([tf.constant(fitted["fit"][key], tf.float32)
        for key in ("centers", "covariances", "log_floors")] if usable_fit else
        [tf.zeros([2, 1]), tf.ones([2, 1, 1]), tf.zeros([2])])
    for n in ((4096,) if regime == "affine" else (1024, 4096)):
        size = result["sizes"].setdefault(str(n), {"calibration": {}, "rows": {}, "controls": []})
        if size.get("complete"):
            continue
        bootstrap = make_fitted_twist_kernel(1, 1, n, 2, "float32", constant_twist=True,
            transition_curve=c, observation_curve=b, include_fisher_score=True)
        no_resampling = make_particle_filter(n, 2, c, b, resampling=False, dtype_name="float32")
        if usable_fit:
            iapf = make_fitted_twist_kernel(1, 1, n, 2, "float32", transition_curve=c,
                observation_curve=b, include_fisher_score=True, include_resampling_controls=True,
                resampling_uniform_bits=UNIFORM_BITS)
            coefficient, apply = calibrate(dataset, n, iapf, (theta, obs, coefficients),
                budget, size["calibration"], save)
        rows = size["rows"]
        names = ["bootstrap_fixed", "bootstrap_fisher", "no_resampling"]
        if usable_fit:
            names += ["iapf_fixed", "iapf_fisher"]
        for name in names:
            rows.setdefault(name, [])
        for rep in range(FINAL):
            draws = streams(dataset, n, rep, "final")
            if usable_fit and rep >= len(rows["iapf_fisher"]):
                budget.charge("filter_calls")
                out = iapf(theta, obs, *draws, *coefficients)
                finite(*out)
                for name, index in (("iapf_fixed", 1), ("iapf_fisher", 3)):
                    rows[name].append(dict(replicate=rep, value=base.materialize(out[0]), score=base.materialize(out[index])))
                size["controls"].append(base.materialize(tf.reshape(out[4], [12])))
            if rep >= len(rows["bootstrap_fisher"]):
                budget.charge("filter_calls")
                out = bootstrap(theta, obs, *draws, *coefficients)
                finite(*out)
                for name, index in (("bootstrap_fixed", 1), ("bootstrap_fisher", 3)):
                    rows[name].append(dict(replicate=rep, value=base.materialize(out[0]), score=base.materialize(out[index])))
            if rep >= len(rows["no_resampling"]):
                budget.charge("filter_calls")
                out = no_resampling(theta, obs, draws[0], draws[1], draws[2][1:])
                finite(*out)
                rows["no_resampling"].append(dict(replicate=rep, value=base.materialize(out[0]),
                    score=base.materialize(out[1]), minimum_ess=base.materialize(out[2])))
            if (rep+1) % 32 == 0:
                save()
        if usable_fit:
            corrected = apply(tf.constant([row["score"] for row in rows["iapf_fisher"]], tf.float64),
                              tf.constant(size["controls"], tf.float64), coefficient)
            finite(corrected)
            rows["iapf_control"] = [dict(replicate=row["replicate"], value=row["value"], score=score)
                for row, score in zip(rows["iapf_fisher"], base.materialize(corrected))]
            size["control_mean"] = [statistics.mean(column) for column in zip(*size["controls"])]
            size["control_mcse"] = [statistics.stdev(column)/math.sqrt(FINAL) for column in zip(*size["controls"])]
        size["summary"] = {name: summarize(data, oracle, critical) for name, data in rows.items()}
        comparisons = {}
        for name in ("iapf_control", "iapf_fisher", "bootstrap_fisher"):
            if name not in rows:
                continue
            for other in dict.fromkeys(("iapf_fisher", "iapf_fixed", *HEURISTICS)):
                if other == name or (other not in rows and other not in result["deterministic"]):
                    continue
                candidate = [squared_error(row["score"], oracle) for row in rows[name]]
                comparator = ([squared_error(row["score"], oracle) for row in rows[other]]
                    if other in rows else [result["deterministic"][other]["score_squared_error"]]*FINAL)
                differences = [a-b for a, b in zip(candidate, comparator)]
                ci = base.materialize(interval_kernel()(tf.constant(differences, tf.float64)))
                comparisons[name+"_minus_"+other] = dict(mean_difference=statistics.mean(differences),
                    paired_bootstrap_9975_interval=ci[:2], paired_bootstrap_99_interval=ci[2:],
                    observed_heuristic_veto=other in HEURISTICS and statistics.mean(differences) > 0,
                    inference="conditional approximate intervals; 99.75% primary family has four N4096 nonlinear comparisons")
        size.update(comparisons=comparisons, complete=True,
            score_pairing="one invocation for raw Fisher, fixed-label derivative and control",
            tracing_counts=dict(bootstrap=bootstrap.experimental_get_tracing_count(),
                no_resampling=no_resampling.experimental_get_tracing_count(),
                streams=stream_kernel(n).experimental_get_tracing_count(),
                **({"iapf_control": iapf.experimental_get_tracing_count()} if usable_fit else {})))
        save()
        print(f"dataset={dataset} regime={regime} N={n} complete", flush=True)
    result["heuristic_dominance_verdict"] = {
        name: "promotion_veto" if any(item["observed_heuristic_veto"]
            for size in result["sizes"].values() for key, item in size["comparisons"].items()
            if key.startswith(name+"_minus_")) else "screen_pass_only"
        for name in ("iapf_control", "iapf_fisher", "bootstrap_fisher")
        if all(name in size["rows"] for size in result["sizes"].values())}
    result["complete"] = True


def run(args):
    global tf
    started = time.monotonic()
    output = Path(args.output).resolve()
    if len(list(output.parent.glob("attempt*/manifest.json"))) >= 4:
        raise RuntimeError("campaign attempt budget exhausted")
    output.mkdir(parents=True, exist_ok=False)
    from bayesfilter.score_study.runtime import configure_runtime, memory_usage
    runtime = configure_runtime(device="GPU", tf32=True, jit_compile=True)
    import tensorflow as tf
    import tensorflow_probability as tfp
    base.tf = tf
    base.REGIMES["affine"] = (0., 0.)
    base.THETA = base.materialize(tf.cast(tf.constant(base.THETA, tf.float32), tf.float64))
    budget = base.Budget(1800.)
    budget.started = started
    budget.limits = {"adaptive_fits": 8, "fixed_cloud_fits": 0, "filter_calls": 8000}
    files = sorted((REPO/"bayesfilter/score_study").glob("*.py")) + [Path(__file__), Path(base.__file__),
        REPO/PLAN, REPO/"bayesfilter/highdim/ledh_canonical_score_stages_tf.py",
        REPO/"bayesfilter/highdim/ledh_canonical_score_tf.py",
        REPO/"bayesfilter/highdim/ledh_ukf_lifecycle_tf.py", REPO/"bayesfilter/runtime/gpu_memory_policy.py",
        REPO/"bayesfilter/nonlinear/sgqf_covariance_provider_tf.py"]
    manifest = dict(schema="iapf_resampling_control_diagnostic_v1", plan=PLAN, output=str(output),
        result_file=str(output/"results.json"), command=sys.argv, cwd=str(REPO),
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        source_sha256={str(path.relative_to(REPO)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
        python=sys.executable, python_version=sys.version, runtime=runtime,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        data_version="scalar_sine_quadratic_resampling_control_fresh_20260918", master_seed=9182026,
        datasets=DATASETS, theta=base.THETA, regimes=base.REGIMES,
        calibration_replicates=CALIBRATION, final_replicates=FINAL, uniform_bits=UNIFORM_BITS,
        seed_groups="resampling_control_20260918_{calibration|final}_N{N}; labels initial/process/ancestors/mixture",
        jit_compile=True, filter_dtype="float32", control_dtype="float64", fit_dtype="float64",
        reference_exception="CPU FP64 independent quadrature/Kalman; FP64 offline fitting, control centering and regression/application",
        control_target="same expectation as terminal Fisher statistic, including finite-N bias",
        finite_N_unbiasedness=False, hmc_eligibility=False, status="running", prior_driver_seconds=0.)
    results = {"references": {}, "fits": {}, "downstream": {}}
    base.write(output/"manifest.json", manifest)

    def save(stage="progress"):
        results.update(stage=stage, budget=budget.record())
        base.write(output/"results.json", results)

    try:
        if args.resume_from:
            prior_root = Path(args.resume_from)
            prior_manifest = json.loads((prior_root/"manifest.json").read_text())
            for key in ("theta", "datasets", "regimes", "data_version", "runtime", "cuda_visible_devices",
                        "calibration_replicates", "final_replicates", "uniform_bits", "seed_groups"):
                if prior_manifest[key] != json.loads(json.dumps(manifest[key])):
                    raise RuntimeError("resume contract mismatch: " + key)
            for path, digest in manifest["source_sha256"].items():
                if path not in (PLAN, str(Path(__file__).relative_to(REPO))) and prior_manifest["source_sha256"].get(path) != digest:
                    raise RuntimeError("resume numerical source mismatch: " + path)
            if prior_manifest["status"] != "stopped":
                raise RuntimeError("resume requires a recorded stopped attempt")
            manifest["prior_driver_seconds"] = prior_manifest["prior_driver_seconds"] + prior_manifest["wall_seconds"]
            budget.seconds -= manifest["prior_driver_seconds"]
            budget.counts.update(prior_manifest["budget"]["counts"])
            results = json.loads((prior_root/"results.json").read_text())
            manifest["resume_from"] = str(prior_root)
            save("resumed")
        with tf.device("/CPU:0"):
            critical = float(tfp.distributions.StudentT(tf.constant(FINAL-1., tf.float64), 0., 1.).quantile(
                tf.constant(1.-.01/48, tf.float64)))
        manifest["bias_interval_critical"] = critical
        for dataset, regime in DATASETS.items():
            key = str(dataset)
            if results["downstream"].get(key, {}).get("complete"):
                continue
            budget.charge("filter_calls", 0)
            observations, back, oracle = base.reference(dataset, regime)
            if regime == "affine":
                from bayesfilter.score_study.gaussian_tf import make_gaussian_kernel, parameterized_model
                with tf.device("/CPU:0"):
                    exact = make_gaussian_kernel(1, 1, 6)(observations,
                        *parameterized_model(tf.constant(base.THETA, tf.float64), 1, 1))
                    value, score = exact[:2]
                oracle.update(kalman_value=base.materialize(value), kalman_score=base.materialize(score))
                oracle["kalman_grid_max_error"] = max(abs(oracle["value"]-oracle["kalman_value"]),
                    max(abs(x-y) for x, y in zip(oracle["score"], oracle["kalman_score"])))
                if oracle["kalman_grid_max_error"] > 1e-7:
                    raise RuntimeError("affine Kalman/reference veto")
            results["references"][key] = oracle
            with tf.device("/GPU:0"):
                if key not in results["fits"]:
                    fitted = base.consumer_fit(dataset, regime, observations, back, "floor_001", budget)
                    if fitted.get("actual_calls", 0) > 8:
                        budget.charge("filter_calls", fitted["actual_calls"]-8)
                    results["fits"][key] = fitted
                    save(f"fitted_{dataset}")
                partial = results["downstream"].setdefault(key, {"regime": regime, "sizes": {}, "deterministic": {}})
                evaluate(dataset, regime, observations, oracle, results["fits"][key], budget,
                    critical, partial, lambda: save(f"dataset_{dataset}_progress"))
            save(f"completed_{dataset}")
        primary = {}
        for dataset, result in results["downstream"].items():
            if result["regime"] != "affine":
                comparison = result["sizes"]["4096"]["comparisons"].get("iapf_control_minus_iapf_fisher")
                primary[dataset] = comparison["paired_bootstrap_9975_interval"][1] < 0 if comparison else None
        decision = dict(primary_conditional_pass=primary, primary_all_pass=all(x is True for x in primary.values()),
            heuristic_dominance={key: value["heuristic_dominance_verdict"] for key, value in results["downstream"].items()},
            nonlinear_bias_screen={key: value["sizes"]["4096"]["summary"].get("iapf_control", {}).get("bias_screen_pass")
                for key, value in results["downstream"].items() if value["regime"] != "affine"},
            fitting_boundary_active={key: value.get("boundary_active") for key, value in results["fits"].items()},
            default_readiness=False, hmc_readiness=False,
            continuation="candidate failure does not invalidate the centering identity or research direction")
        results["decision"] = decision
        save("complete")
        manifest.update(status="complete", memory_usage=memory_usage("GPU"),
            wall_seconds=time.monotonic()-started, budget=budget.record())
        base.write(output/"manifest.json", manifest)
        base.write(output/"decision.json", decision)
    except Exception as error:
        save("stopped")
        manifest.update(status="stopped", exception=repr(error), wall_seconds=time.monotonic()-started,
                        budget=budget.record())
        base.write(output/"manifest.json", manifest)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume-from", help="Reuse compatible saved results and cumulative budget")
    run(parser.parse_args())
