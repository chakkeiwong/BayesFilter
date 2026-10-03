"""Diagnostic only: same-particle Fisher versus fixed-label iAPF scores.

The reference and report are diagnostic. No tuning or admission authority is
issued. All numerical particle kernels use analytical TensorFlow/XLA code.
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

PLAN = "docs/plans/younis-iapf-fisher-score-2026-09-18.md"
DATASETS = {1390: "affine", 1400: "weak", 1401: "weak", 1410: "curved", 1411: "curved"}
REPLICATES = 64


@lru_cache(maxsize=1)
def interval_kernel():
    @tf.function(input_signature=[tf.TensorSpec([REPLICATES], tf.float64)], jit_compile=True)
    def kernel(differences):
        indices = tf.random.stateless_uniform([4000, REPLICATES], [919, 2026],
            minval=0, maxval=REPLICATES, dtype=tf.int32)
        means = tf.sort(tf.reduce_mean(tf.gather(differences, indices), 1))
        return tf.stack([means[19], means[3979]])
    return kernel


def squared_error(score, oracle):
    return sum((x-y)**2 for x, y in zip(score, oracle["score"]))


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
            for tensor in output:
                tf.debugging.assert_all_finite(tensor, "invalid moment comparator")
            outputs.append(output)
        value, score = base.materialize(outputs[0][0]), [base.materialize(out[1]) for out in outputs]
        result["deterministic"][name] = dict(value=value, score=score,
            score_squared_error=squared_error(score, oracle))
        save()

    coefficients = ([tf.constant(fitted["fit"][key], tf.float32)
        for key in ("centers", "covariances", "log_floors")]
        if fitted["status"] == "valid" else None)
    for n in ((4096,) if regime == "affine" else (256, 1024, 4096)):
        if str(n) in result["sizes"]:
            continue
        kwargs = dict(transition_curve=c, observation_curve=b)
        kernels = {"bootstrap": make_fitted_twist_kernel(1, 1, n, 2, "float32",
            constant_twist=True, include_fisher_score=True, **kwargs)}
        if coefficients is not None:
            kernels["iapf"] = make_fitted_twist_kernel(1, 1, n, 2, "float32",
                include_fisher_score=True, **kwargs)
        no_resampling = make_particle_filter(n, 2, c, b, resampling=False, dtype_name="float32")
        rows = {prefix+"_"+score: [] for prefix in kernels for score in ("fixed", "fisher")}
        rows["no_resampling"] = []
        for rep in range(REPLICATES):
            draws = base.streams(dataset, n, rep, tf.float32)
            for name, kernel in kernels.items():
                fit = coefficients if name == "iapf" else (tf.zeros([2, 1]), tf.ones([2, 1, 1]), tf.zeros([2]))
                budget.charge("filter_calls")
                output = kernel(theta, obs, *draws, *fit)
                for tensor in output:
                    tf.debugging.assert_all_finite(tensor, "invalid scored particle output")
                for suffix, index in (("fixed", 1), ("fisher", 3)):
                    rows[name+"_"+suffix].append(dict(replicate=rep,
                        value=base.materialize(output[0]), score=base.materialize(output[index])))
            budget.charge("filter_calls")
            output = no_resampling(theta, obs, draws[0], draws[1], draws[2][1:])
            for tensor in output:
                tf.debugging.assert_all_finite(tensor, "invalid no-resampling output")
            rows["no_resampling"].append(dict(replicate=rep,
                value=base.materialize(output[0]), score=base.materialize(output[1]),
                minimum_ess=base.materialize(output[2])))
        summaries = {name: base.summarize(values, oracle) for name, values in rows.items()}
        for summary in summaries.values():
            radius = [critical*s+5e-6*(1+abs(target))
                      for s, target in zip(summary["score_standard_error"], oracle["score"])]
            intervals = [[error-r, error+r] for error, r in zip(summary["mean_score_error"], radius)]
            summary["mean_error_intervals"] = intervals
            summary["bias_screen_pass"] = all(lo <= 0 <= hi for lo, hi in intervals)
            summary["interval_scope"] = ("24 nonlinear components, approximate Student-t Bonferroni 99%; "
                "affine six components use same conservative critical value; numerical allowance separate")
        comparisons = {}
        for name in ("iapf_fisher", "bootstrap_fisher"):
            if name not in rows:
                continue
            for other in (name.replace("fisher", "fixed"), "bootstrap_fisher", "no_resampling", "ekf", "ukf"):
                if other == name:
                    continue
                candidate = [squared_error(row["score"], oracle) for row in rows[name]]
                comparator = ([squared_error(row["score"], oracle) for row in rows[other]]
                    if other in rows else [result["deterministic"][other]["score_squared_error"]]*REPLICATES)
                differences = [x-y for x, y in zip(candidate, comparator)]
                interval = base.materialize(interval_kernel()(tf.constant(differences, tf.float64)))
                comparisons[name+"_minus_"+other] = dict(mean_difference=statistics.mean(differences),
                    paired_bootstrap_99_interval=interval,
                    observed_heuristic_veto=not other.endswith("fixed") and statistics.mean(differences) > 0,
                    inference="conditional exploratory interval, not a simultaneous cross-dataset ranking")
        result["sizes"][str(n)] = dict(rows=rows, summary=summaries, comparisons=comparisons,
            score_pairing="same kernel invocation, particles, ancestors, fit and likelihood value",
            tracing_counts={name: kernel.experimental_get_tracing_count() for name, kernel in kernels.items()})
        save()
        print(f"dataset={dataset} regime={regime} N={n} complete", flush=True)
    result["heuristic_dominance_verdict"] = {
        name: "promotion_veto" if any(item["observed_heuristic_veto"]
            for size in result["sizes"].values() for key, item in size["comparisons"].items()
            if key.startswith(name+"_minus_")) else "screen_pass_only"
        for name in ("iapf_fisher", "bootstrap_fisher")
        if all(name in size["rows"] for size in result["sizes"].values())}
    result["complete"] = True
    return result


def run(args):
    global tf
    started = time.monotonic()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    from bayesfilter.score_study.runtime import configure_runtime, memory_usage
    runtime = configure_runtime(device="GPU", tf32=True, jit_compile=True)
    import tensorflow as tf
    import tensorflow_probability as tfp
    base.tf = tf
    base.REGIMES["affine"] = (0., 0.)
    base.THETA = base.materialize(tf.cast(tf.constant(base.THETA, tf.float32), tf.float64))
    budget = base.Budget(args.wall_seconds)
    budget.started = started
    budget.limits = {"adaptive_fits": 8, "fixed_cloud_fits": 0, "filter_calls": 3000}
    budget.counts.update(json.loads(args.consumed_counts))
    files = sorted((REPO/"bayesfilter/score_study").glob("*.py")) + [Path(__file__), Path(base.__file__),
        REPO/PLAN, REPO/"bayesfilter/highdim/ledh_canonical_score_stages_tf.py",
        REPO/"bayesfilter/highdim/ledh_canonical_score_tf.py",
        REPO/"bayesfilter/highdim/ledh_ukf_lifecycle_tf.py",
        REPO/"bayesfilter/runtime/gpu_memory_policy.py",
        REPO/"bayesfilter/nonlinear/sgqf_covariance_provider_tf.py"]
    manifest = dict(schema="iapf_fisher_diagnostic_v1", plan=PLAN, output=str(output),
        result_file=str(output/"results.json"), command=sys.argv, cwd=str(REPO),
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        source_sha256={str(path.relative_to(REPO)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
        python=sys.executable, python_version=sys.version, runtime=runtime,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        data_version="scalar_sine_quadratic_fisher_fresh_20260918", master_seed=9182026,
        datasets=DATASETS, theta=base.THETA, regimes=base.REGIMES, replicates=REPLICATES,
        jit_compile=True, reference_dtype="float64", filter_dtype="float32", fit_dtype="float64",
        reference_exception="CPU FP64 independent quadrature/Kalman; FP64 offline fitting, GPU FP32 final kernels",
        fisher_target="physical model score, terminal weighted complete-data genealogy",
        fixed_target="fixed ancestor and mixture labels, derivative of finite likelihood program",
        finite_N_unbiasedness=False, hmc_eligibility=False, status="running",
        prior_driver_seconds=args.prior_seconds, consumed_counts=json.loads(args.consumed_counts))
    base.write(output/"manifest.json", manifest)
    results = {"references": {}, "fits": {}, "downstream": {}}

    def save(stage):
        results.update(stage=stage, budget=budget.record())
        base.write(output/"results.json", results)

    try:
        if args.resume_from:
            prior_root = Path(args.resume_from)
            prior_manifest_path, prior_results_path = prior_root/"manifest.json", prior_root/"results.json"
            prior_manifest = json.loads(prior_manifest_path.read_text())
            prior_results = json.loads(prior_results_path.read_text())
            for key in ("theta", "datasets", "regimes", "replicates", "data_version",
                        "filter_dtype", "fit_dtype", "reference_dtype", "cuda_visible_devices", "runtime"):
                # JSON object keys (dataset integers) normalize to strings on disk.
                if prior_manifest[key] != json.loads(json.dumps(manifest[key])):
                    raise RuntimeError("resume contract mismatch: " + key)
            for path, digest in manifest["source_sha256"].items():
                if path not in (PLAN, str(Path(__file__).relative_to(REPO))):
                    if prior_manifest["source_sha256"].get(path) != digest:
                        raise RuntimeError("resume numerical source mismatch: " + path)
            prior_seconds = prior_manifest["prior_driver_seconds"] + prior_manifest["wall_seconds"]
            prior_counts = prior_manifest["budget"]["counts"]
            if args.prior_seconds and not math.isclose(args.prior_seconds, prior_seconds):
                raise RuntimeError("resume prior wall time mismatch")
            if args.consumed_counts != "{}" and json.loads(args.consumed_counts) != prior_counts:
                raise RuntimeError("resume prior counts mismatch")
            budget.counts.update(prior_counts)
            budget.seconds = min(args.wall_seconds, 1800.-prior_seconds)
            manifest.update(prior_driver_seconds=prior_seconds, consumed_counts=dict(prior_counts),
                resume={"path": str(prior_root),
                    "manifest_sha256": hashlib.sha256(prior_manifest_path.read_bytes()).hexdigest(),
                    "results_sha256": hashlib.sha256(prior_results_path.read_bytes()).hexdigest(),
                    "numerical_source_hashes_verified": True})
            results.update({key: prior_results[key] for key in ("references", "fits", "downstream")})
            for dataset, value in results["downstream"].items():
                expected = {"4096"} if value["regime"] == "affine" else {"256", "1024", "4096"}
                if set(value["sizes"]) == expected and "heuristic_dominance_verdict" in value:
                    value["complete"] = True
            base.write(output/"manifest.json", manifest)
            save("resumed")
        with tf.device("/CPU:0"):
            critical = float(tfp.distributions.StudentT(tf.constant(63., tf.float64), 0., 1.).quantile(
                tf.constant(1.-.01/(2*24), tf.float64)).numpy())
        manifest["bias_interval_critical"] = critical
        manifest["numerical_allowance"] = "5e-6*(1+abs(reference component)), FP32 diagnostic tolerance"
        for dataset, regime in DATASETS.items():
            key = str(dataset)
            if results["downstream"].get(key, {}).get("complete"):
                continue
            observations, back, oracle = base.reference(dataset, regime)
            if key in results["references"]:
                if results["references"][key]["executed_observations"] != oracle["executed_observations"]:
                    raise RuntimeError("resume observations mismatch")
            if regime == "affine":
                from bayesfilter.score_study.gaussian_tf import make_gaussian_kernel, parameterized_model
                with tf.device("/CPU:0"):
                    exact = make_gaussian_kernel(1, 1, 6)(observations,
                        *parameterized_model(tf.constant(base.THETA, tf.float64), 1, 1))
                oracle["kalman_value"] = base.materialize(exact[0])
                oracle["kalman_score"] = base.materialize(exact[1])
                oracle["kalman_grid_max_error"] = max(abs(oracle["value"]-oracle["kalman_value"]),
                    max(abs(x-y) for x, y in zip(oracle["score"], oracle["kalman_score"])))
                if oracle["kalman_grid_max_error"] > 1e-7:
                    raise RuntimeError("affine Kalman/reference veto")
            results["references"][key] = oracle
            with tf.device("/GPU:0"):
                fitted = results["fits"].get(key)
                if fitted is None:
                    fitted = base.consumer_fit(dataset, regime, observations, back, "floor_001", budget)
                    if fitted.get("actual_calls", 0) > 8:
                        budget.charge("filter_calls", fitted["actual_calls"]-8)
                    results["fits"][key] = fitted
                save(f"fitted_{dataset}")
                partial = results["downstream"].setdefault(key,
                    {"regime": regime, "sizes": {}, "deterministic": {}})
                evaluate(dataset, regime, observations, oracle, fitted, budget, critical,
                    partial, lambda: save(f"dataset_{dataset}_progress"))
            save(f"completed_{dataset}")
        decision = dict(implementation="analytical Fisher estimator checked separately from fixed-label derivative",
            nonlinear_bias_screen={dataset: value["sizes"]["4096"]["summary"].get("iapf_fisher", {}).get("bias_screen_pass")
                for dataset, value in results["downstream"].items() if value["regime"] != "affine"},
            heuristic_dominance={dataset: value["heuristic_dominance_verdict"] for dataset, value in results["downstream"].items()},
            fitting_boundary_active={dataset: value.get("boundary_active") for dataset, value in results["fits"].items()},
            default_readiness=False, hmc_readiness=False, ranking="only reported conditional paired intervals",
            continuation="candidate losses do not invalidate the physical identity; inspect uncertainty and fit bounds")
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
    parser.add_argument("--wall-seconds", type=float, default=1800.)
    parser.add_argument("--prior-seconds", type=float, default=0.)
    parser.add_argument("--consumed-counts", default="{}")
    parser.add_argument("--resume-from", help="Reuse compatible saved results and cumulative budget")
    arguments = parser.parse_args()
    if arguments.wall_seconds+arguments.prior_seconds > 1800.:
        parser.error("combined campaign wall budget exceeds 1800 seconds")
    run(arguments)
