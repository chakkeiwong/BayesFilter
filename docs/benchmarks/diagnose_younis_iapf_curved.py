"""Diagnostic only: localize curved iAPF fitting and fixed-label score errors.

Numerical kernels use TensorFlow, including the independent backward reference.
No runtime algorithm, tuning authority, or default is supplied by this script.
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
PLAN = "docs/plans/younis-iapf-curved-diagnosis-2026-09-18.md"
THETA = [.62, -.8, -.6, .9, .25, -.3]
REGIMES = {"weak": (.12, .04), "curved": (.35, .12)}
PARTITIONS = {"calibration": (1300, 1301), "validation": (1310, 1311),
              "confirmation": (1320, 1321)}
BASE = dict(k=1, tau=100., max_iterations=4, max_particles=128,
            mean_bound=4., sd_lower=.2, sd_upper=4., max_fit_steps=2000,
            max_backtracks=30, fit_tolerance=1e-7, floor_ratio=.01,
            fit_theta=THETA, fit_dtype="float64")
ARMS = {
    "density_2000": {**BASE, "fit_objective": "density_l2"},
    "shape_2000": {**BASE, "fit_objective": "relative_shape"},
    "shape_10000": {**BASE, "fit_objective": "relative_shape", "max_fit_steps": 10000},
    "wide_10000": {**BASE, "fit_objective": "relative_shape", "max_fit_steps": 10000,
                   "mean_bound": 8., "sd_lower": .1, "sd_upper": 8.},
    "floor_001": {**BASE, "fit_objective": "relative_shape", "max_fit_steps": 10000,
                  "floor_ratio": .001},
    "floor_1": {**BASE, "fit_objective": "relative_shape", "max_fit_steps": 10000,
                "floor_ratio": .1},
}


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def materialize(value):
    if hasattr(value, "numpy"):
        return value.numpy().tolist()
    if isinstance(value, dict):
        return {key: materialize(val) for key, val in value.items()}
    if isinstance(value, (tuple, list)):
        return [materialize(val) for val in value]
    return value


class Budget:
    def __init__(self, seconds):
        self.started = time.monotonic()
        self.seconds = seconds
        self.counts = {"adaptive_fits": 0, "fixed_cloud_fits": 0, "filter_calls": 0}
        self.limits = {"adaptive_fits": 40, "fixed_cloud_fits": 320, "filter_calls": 3500}

    def charge(self, kind, count=1):
        if time.monotonic() - self.started >= self.seconds:
            raise RuntimeError("continuation veto: wall budget exhausted")
        self.counts[kind] += count
        if self.counts[kind] > self.limits[kind]:
            raise RuntimeError("continuation veto: " + kind + " budget exhausted")

    def record(self):
        return {"wall_seconds": time.monotonic() - self.started,
                "wall_limit": self.seconds, "counts": self.counts, "limits": self.limits}


def seed(dataset, stream, replicate=0, group="paired_final"):
    from bayesfilter.score_study.contracts import seed_pair
    return seed_pair(master_seed=9182026, model="nonlinear_scalar", dataset=dataset,
                     replicate=replicate, stream=stream, coupling_group=group)


def streams(dataset, n, replicate, dtype):
    normal = lambda label, shape: tf.random.stateless_normal(
        shape, seed(dataset, label, replicate), dtype=dtype)
    uniform = lambda label, shape: tf.random.stateless_uniform(
        shape, seed(dataset, label, replicate), dtype=dtype)
    return (normal("initial", [n, 1]), normal("process", [2, n, 1]),
            uniform("ancestors", [3, n]), uniform("mixture", [2, n]))


@lru_cache(maxsize=8)
def backward_kernel(points, radius, c, b):
    from bayesfilter.score_study.gaussian_tf import parameterized_model
    dtype = tf.float64

    @tf.function(input_signature=[tf.TensorSpec([6], dtype), tf.TensorSpec([2, 1], dtype)],
                 jit_compile=True)
    def kernel(theta, observations):
        A, _, H, _, m, _, P, _, Q, _, R, _ = parameterized_model(theta, 1, 1)
        x = tf.linspace(tf.cast(-radius, dtype), tf.cast(radius, dtype), points)
        measure = tf.concat([tf.constant([.5], dtype), tf.ones([points-2], dtype),
                             tf.constant([.5], dtype)], 0) * (2*radius/(points-1))
        lognormal = lambda residual, variance: -.5*(
            tf.math.log(2*tf.acos(tf.cast(-1., dtype))*variance)+residual**2/variance)
        transition_mean = A[0, 0]*x + tf.cast(c, dtype)*tf.sin(x)
        log_matrix = lognormal(x[None, :]-transition_mean[:, None], Q[0, 0])
        log_matrix += tf.math.log(measure)[None, :]
        matrix = tf.exp(log_matrix)
        means = H[0, 0]*x + tf.cast(b, dtype)*x*x
        log_g = lognormal(observations-means[None, :], R[0, 0])
        log_psi_last = log_g[1]
        log_psi_first = log_g[0] + tf.reduce_logsumexp(log_matrix+log_psi_last[None, :], 1)
        initial = tf.exp(lognormal(x-m[0], P[0, 0]))*measure
        first_prediction = tf.einsum("i,ij->j", initial, matrix)
        first_raw = first_prediction*tf.exp(log_g[0]-tf.reduce_max(log_g[0]))
        first_filter = first_raw/tf.reduce_sum(first_raw)
        second_prediction = tf.einsum("i,ij->j", first_filter, matrix)
        predictive = tf.stack([first_prediction, second_prediction])
        predictive /= tf.reduce_sum(predictive, 1)[:, None]
        backward_value = tf.reduce_logsumexp(tf.math.log(first_prediction)+log_psi_first)
        return x, tf.stack([log_psi_first, log_psi_last]), predictive, backward_value
    return kernel


@lru_cache(maxsize=8)
def shape_kernel(points):
    @tf.function(input_signature=[tf.TensorSpec([points], tf.float64),
        tf.TensorSpec([2, points], tf.float64), tf.TensorSpec([2, points], tf.float64),
        tf.TensorSpec([2, 1], tf.float64), tf.TensorSpec([2, 1, 1], tf.float64),
        tf.TensorSpec([2], tf.float64)], jit_compile=True)
    def kernel(x, targets, weights, centers, covariances, floors):
        variance = covariances[:, 0, 0]
        log_p = -.5*(tf.math.log(2*tf.acos(tf.constant(-1., tf.float64))*variance)[:, None]
                       +(x[None, :]-centers[:, 0, None])**2/variance[:, None])
        log_total = tf.reduce_logsumexp(tf.stack([log_p, tf.broadcast_to(floors[:, None], tf.shape(log_p))]), 0)

        def residual(log_values):
            p = tf.exp(log_values-tf.reduce_max(log_values, 1)[:, None])
            y = tf.exp(targets-tf.reduce_max(targets, 1)[:, None])
            a = tf.reduce_sum(weights*p*p, 1)
            cross = tf.reduce_sum(weights*p*y, 1)
            norm_y = tf.reduce_sum(weights*y*y, 1)
            return tf.maximum(tf.constant(0., tf.float64), 1-cross*cross/(a*norm_y))
        floor_fraction = tf.reduce_sum(weights*tf.exp(floors[:, None]-log_total), 1)
        return residual(log_p), residual(log_total), floor_fraction
    return kernel


@lru_cache(maxsize=16)
def point_fit_kernel(n, mean_bound, lower, upper, steps, objective):
    from bayesfilter.score_study.iapf_fit_tf import bounded_density_fit

    @tf.function(input_signature=[tf.TensorSpec([n, 1], tf.float64),
                                  tf.TensorSpec([n], tf.float64)], jit_compile=True)
    def kernel(points, targets):
        return bounded_density_fit(points, targets, mean_bound=mean_bound,
            sd_lower=lower, sd_upper=upper, max_steps=steps, max_backtracks=30,
            tolerance=1e-7, floor_ratio=.01, objective=objective)
    return kernel


@lru_cache(maxsize=4)
def box_kernel(n, mean_bound, sd_lower, sd_upper):
    @tf.function(input_signature=[tf.TensorSpec([n, 1], tf.float64),
                                  tf.TensorSpec([n], tf.float64)], jit_compile=True)
    def kernel(points, targets):
        mean = tf.reduce_mean(points[:, 0])
        sd = tf.math.reduce_std(points[:, 0])
        z = (points[:, 0]-mean)/sd
        y = tf.exp(targets-tf.reduce_max(targets))
        lower = tf.constant([-mean_bound, math.log(sd_lower)], tf.float64)
        upper = tf.constant([mean_bound, math.log(sd_upper)], tf.float64)
        lo, hi = lower, upper
        best = tf.zeros([2], tf.float64)
        loss = tf.constant(1., tf.float64)
        for _ in range(3):
            m, l = tf.meshgrid(tf.linspace(lo[0], hi[0], 81), tf.linspace(lo[1], hi[1], 81))
            m, l = tf.reshape(m, [-1]), tf.reshape(l, [-1])
            logp = -.5*((z[None, :]-m[:, None])*tf.exp(-l[:, None]))**2
            p = tf.exp(logp-tf.reduce_max(logp, 1)[:, None])
            cross = tf.reduce_sum(p*y[None, :], 1)
            losses = tf.maximum(tf.constant(0., tf.float64),
                1-cross**2/(tf.reduce_sum(p*p, 1)*tf.reduce_sum(y*y)))
            index = tf.argmin(losses)
            best = tf.stack([m[index], l[index]])
            loss = losses[index]
            width = (hi-lo)/80
            lo, hi = tf.maximum(lower, best-width), tf.minimum(upper, best+width)
        center, fitted_sd = mean+sd*best[0], sd*tf.exp(best[1])
        floor = tf.math.log(tf.constant(.01, tf.float64))-.5*tf.math.log(
            2*tf.acos(tf.constant(-1., tf.float64)))-tf.math.log(fitted_sd)
        return tf.reshape(center, [1]), tf.reshape(fitted_sd**2, [1, 1]), floor, loss, best
    return kernel


def reference(dataset, regime):
    from bayesfilter.score_study.nonlinear_tf import make_data_kernel, make_grid_reference
    c, b = REGIMES[regime]
    with tf.device("/CPU:0"):
        theta = tf.constant(THETA, tf.float64)
        physical = make_data_kernel(2, c, b)(theta, tf.constant(seed(dataset, "observations", group="common_data")))
        observations = tf.cast(tf.cast(physical, tf.float32), tf.float64)
        results = [make_grid_reference(2, p, r, c, b)(theta, observations)
                   for p, r in ((201, 9.), (401, 9.), (601, 13.5))]
        base = results[1]
        errors = [max(float(tf.reduce_max(tf.abs(a-z)/(1+tf.abs(a))).numpy())
                      for a, z in zip(base[:4], result[:4])) for result in (results[0], results[2])]
        tail = max(float(result[k].numpy()) for result in results for k in (4, 5))
        back = backward_kernel(401, 9., c, b)(theta, observations)
        tie = float(tf.abs(back[3]-base[0]).numpy())/(1+abs(float(base[0].numpy())))
        for value in (*base, *back):
            tf.debugging.assert_all_finite(value, "reference nonfinite")
        if max(errors) > 1e-7 or tail > 1e-9 or tie > 1e-7:
            raise RuntimeError(f"reference veto: {dataset}: {errors}, {tail}, {tie}")
        record = dict(dataset=dataset, regime=regime, data_seed=seed(dataset, "observations", group="common_data"),
            physical_observations=materialize(physical), executed_observations=materialize(observations),
            value=materialize(base[0]), score=materialize(base[1]), mesh_error=errors[0],
            domain_error=errors[1], tail_mass=tail, forward_backward_error=tie,
            reference_device=base[0].device, backward_value=materialize(back[3]))
    return observations, back, record


def shape_for_fit(back, fit):
    result = shape_kernel(401)(*back[:3], *[tf.constant(fit[key], tf.float64)
        for key in ("centers", "covariances", "log_floors")])
    return dict(zip(("gaussian_shape", "with_floor_shape", "predictive_floor_fraction"), materialize(result)))


def localization(dataset, regime, observations, back, budget):
    from bayesfilter.score_study.fitted_twist_tf import make_fitted_twist_kernel
    from bayesfilter.score_study.iapf_fit_tf import make_density_recursive_fit_kernel
    c, b = REGIMES[regime]
    kernel = make_fitted_twist_kernel(1, 1, 16, 2, constant_twist=True,
        transition_curve=c, observation_curve=b)
    budget.charge("filter_calls")
    cloud = kernel(tf.constant(THETA, tf.float64), observations,
        *streams(dataset, 16, 991, tf.float64), tf.zeros([2, 1], tf.float64),
        tf.ones([2, 1, 1], tf.float64), tf.zeros([2], tf.float64))[2]
    output = {"same_cloud": materialize(cloud), "solver_ladder": {}, "backward_fits": []}
    controls = [("density_2000", 2000, 4., .2, 4., "density_l2"),
                ("shape_2000", 2000, 4., .2, 4., "relative_shape"),
                ("shape_5000", 5000, 4., .2, 4., "relative_shape"),
                ("shape_10000", 10000, 4., .2, 4., "relative_shape"),
                ("wide_10000", 10000, 8., .1, 8., "relative_shape")]
    for name, steps, mean_bound, low, high, objective in controls:
        budget.charge("fixed_cloud_fits", 2)
        fit = make_density_recursive_fit_kernel(1, 1, 16, 2, mean_bound, low, high,
            steps, 30, 1e-7, .01, transition_curve=c, observation_curve=b, objective=objective)
        values = fit(tf.constant(THETA, tf.float64), observations, cloud)
        record = {"centers": materialize(values[0]), "covariances": materialize(values[1]),
            "log_floors": materialize(values[2]), "valid": materialize(values[3]),
            "converged": materialize(values[4]), "diagnostics": materialize(values[5])}
        record.update(shape_for_fit(back, record))
        output["solver_ladder"][name] = record
    for t in range(2):
        cdf = tf.cumsum(back[2][t])
        quantiles = (tf.range(257, dtype=tf.float64)+.5)/257
        indices = tf.minimum(tf.searchsorted(cdf, quantiles), 400)
        points, targets = tf.gather(back[0], indices)[:, None], tf.gather(back[1][t], indices)
        for label, mean_bound, lower, upper in (("original", 4., .2, 4.), ("wide", 8., .1, 8.)):
            budget.charge("fixed_cloud_fits", 2)
            fitted = point_fit_kernel(257, mean_bound, lower, upper, 10000, "relative_shape")(points, targets)
            competitor = box_kernel(257, mean_bound, lower, upper)(points, targets)
            output["backward_fits"].append(dict(time=t, bounds=label,
                projected_fit=materialize(fitted), box_competitor=materialize(competitor)))
    return output


def consumer_fit(dataset, regime, observations, back, name, budget):
    from bayesfilter.score_study.iapf_adapter import execute_iapf
    from bayesfilter.score_study.contracts import DiagnosticFailure
    c, b = REGIMES[regime]
    settings = dict(dimension=1, observation_dimension=1, horizon=2, particles=16,
                    dtype="float32", jit_compile=True, transition_curve=c, observation_curve=b)

    def draw(label, replicate=None, group=None):
        return seed(dataset, label, 0 if replicate is None else replicate, group or "consumer_final")
    budget.charge("adaptive_fits")
    budget.charge("filter_calls", 8)
    started = time.monotonic()
    try:
        _, final, details, calls = execute_iapf({"iapf": ARMS[name], "role": "mechanics",
            "model": "nonlinear_scalar"},
            settings, tf.constant(THETA, tf.float32), tf.cast(observations, tf.float32), draw)
        steps = [step for record in details["fit_iterations"]
                 for step in record.get("density_fit_diagnostics", [])]
        record = dict(status="valid", arm=name, fit=details["fit"], details=details,
            actual_calls=calls, boundary_active=any(step[3] != 0 for step in steps),
            max_projected_gradient=max(step[5] for step in steps),
            max_fit_steps=max(step[4] for step in steps), final_value=materialize(final[0]))
        record.update(shape_for_fit(back, record["fit"]))
    except DiagnosticFailure as error:
        record = dict(status="rejected", arm=name, reason=str(error),
                      diagnostics=error.diagnostics)
    record["wall_seconds"] = time.monotonic()-started
    return record


def summarize(rows, oracle):
    scores = [row["score"] for row in rows]
    values = [row["value"] for row in rows]
    means = [statistics.mean(column) for column in zip(*scores)]
    se = [statistics.stdev(column)/math.sqrt(len(rows)) for column in zip(*scores)]
    errors = [sum((x-y)**2 for x, y in zip(score, oracle["score"])) for score in scores]
    value_errors = [(value-oracle["value"])**2 for value in values]
    z = [abs(a-b)/s if s > 0 else (0. if a == b else None)
         for a, b, s in zip(means, oracle["score"], se)]
    return dict(replicates=len(rows), mean_value=statistics.mean(values),
        value_standard_error=statistics.stdev(values)/math.sqrt(len(rows)),
        value_squared_error=statistics.mean(value_errors), mean_score=means,
        score_standard_error=se, mean_score_error=[a-b for a,b in zip(means, oracle["score"])],
        mean_score_error_over_mcse=z, score_squared_error=statistics.mean(errors),
        likelihood_ratio_mean=statistics.mean(math.exp(value-oracle["value"]) for value in values))


@lru_cache(maxsize=2)
def paired_interval_kernel():
    @tf.function(input_signature=[tf.TensorSpec([32], tf.float64)], jit_compile=True)
    def kernel(differences):
        indices = tf.random.stateless_uniform([2000, 32], [918, 2026], minval=0, maxval=32, dtype=tf.int32)
        means = tf.sort(tf.reduce_mean(tf.gather(differences, indices), 1))
        return tf.stack([means[9], means[1989]])
    return kernel


def downstream(dataset, regime, observations, fitted, sizes, oracle, budget):
    from bayesfilter.score_study.fitted_twist_tf import make_fitted_twist_kernel
    from bayesfilter.score_study.nonlinear_tf import make_particle_filter, make_moment_filter
    c, b = REGIMES[regime]
    theta, obs = tf.constant(THETA, tf.float32), tf.cast(observations, tf.float32)
    result = {"sizes": {}, "deterministic": {}}
    for name in ("ekf", "ukf"):
        kernel = make_moment_filter(2, c, b, name, "float32")
        outputs = [kernel(theta, direction, obs) for direction in tf.unstack(tf.eye(6))]
        budget.charge("filter_calls", 6)
        value, score = materialize(outputs[0][0]), [materialize(out[1]) for out in outputs]
        result["deterministic"][name] = dict(value=value, score=score,
            score_squared_error=sum((x-y)**2 for x,y in zip(score, oracle["score"])),
            value_squared_error=(value-oracle["value"])**2)
    for n in sizes:
        arms = {}
        coefficients = {name: [tf.constant(record["fit"][key], tf.float32)
            for key in ("centers", "covariances", "log_floors")]
            for name, record in fitted.items() if record["status"] == "valid"}
        twisted = make_fitted_twist_kernel(1, 1, n, 2, "float32", transition_curve=c, observation_curve=b)
        particle = {name: make_particle_filter(n, 2, c, b, adapted=(name == "local_linear"),
            resampling=(name != "no_resampling"), dtype_name="float32")
            for name in ("bootstrap", "local_linear", "no_resampling")}
        for name in (*coefficients, *particle):
            arms[name] = []
        for rep in range(32):
            draws = streams(dataset, n, rep, tf.float32)
            for name, fit in coefficients.items():
                budget.charge("filter_calls")
                output = twisted(theta, obs, *draws, *fit)
                for part in output[:2]:
                    tf.debugging.assert_all_finite(part, "nonfinite downstream twist")
                arms[name].append(dict(replicate=rep, value=materialize(output[0]), score=materialize(output[1])))
            for name, kernel in particle.items():
                budget.charge("filter_calls")
                output = kernel(theta, obs, draws[0], draws[1], draws[2][1:])
                for part in output[:2]:
                    tf.debugging.assert_all_finite(part, "nonfinite heuristic")
                arms[name].append(dict(replicate=rep, value=materialize(output[0]),
                    score=materialize(output[1]), minimum_ess=materialize(output[2])))
        summaries = {name: summarize(rows, oracle) for name, rows in arms.items()}
        comparisons = {}
        for name in coefficients:
            candidate_errors = [sum((x-y)**2 for x,y in zip(row["score"], oracle["score"])) for row in arms[name]]
            for other in (*particle, "ekf", "ukf"):
                comparator_errors = ([sum((x-y)**2 for x,y in zip(row["score"], oracle["score"]))
                    for row in arms[other]] if other in particle else
                    [result["deterministic"][other]["score_squared_error"]]*32)
                differences = [x-y for x,y in zip(candidate_errors, comparator_errors)]
                comparisons[name+"_minus_"+other] = dict(mean_difference=statistics.mean(differences),
                    paired_bootstrap_99_interval=materialize(paired_interval_kernel()(tf.constant(differences, tf.float64))),
                    observed_heuristic_veto=statistics.mean(differences) > 0)
        result["sizes"][str(n)] = dict(rows=arms, summary=summaries, comparisons=comparisons)
        print(f"downstream dataset={dataset} N={n} arms={list(arms)}", flush=True)
    result["heuristic_dominance_verdict"] = {
        name: "veto" if any(item["observed_heuristic_veto"] for size in result["sizes"].values()
            for key, item in size["comparisons"].items() if key.startswith(name+"_minus_")) else "screen_pass_only"
        for name, record in fitted.items() if record["status"] == "valid"}
    return result


def finite_difference(dataset, regime, observations, record, budget):
    from bayesfilter.score_study.fitted_twist_tf import make_fitted_twist_kernel
    c, b = REGIMES[regime]
    n = 16
    kernel = make_fitted_twist_kernel(1, 1, n, 2, "float64", transition_curve=c, observation_curve=b)
    theta = tf.constant(THETA, tf.float64)
    draws = streams(dataset, n, 88, tf.float64)
    coeff = [tf.constant(record["fit"][key], tf.float64) for key in ("centers", "covariances", "log_floors")]
    base = kernel(theta, observations, *draws, *coeff)
    budget.charge("filter_calls")
    checks = []
    for h in (1e-4, 1e-5, 1e-6):
        estimates = []
        for direction in tf.unstack(tf.eye(6, dtype=tf.float64)):
            budget.charge("filter_calls", 2)
            plus = kernel(theta+h*direction, observations, *draws, *coeff)[0]
            minus = kernel(theta-h*direction, observations, *draws, *coeff)[0]
            estimates.append((plus-minus)/(2*h))
        fd = tf.stack(estimates)
        checks.append(dict(step=h, finite_difference=materialize(fd),
            max_absolute_error=materialize(tf.reduce_max(tf.abs(fd-base[1])))))
    return dict(target="same frozen coefficients and uniforms; locally fixed labels only",
                analytical_score=materialize(base[1]), checks=checks)


def run(args):
    global tf
    started = time.monotonic()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    from bayesfilter.score_study.runtime import configure_runtime, memory_usage
    runtime = configure_runtime(device=args.device, tf32=args.device == "GPU", jit_compile=True)
    import tensorflow as tf
    budget = Budget(args.wall_seconds)
    files = sorted((REPO/"bayesfilter/score_study").glob("*.py")) + [Path(__file__), REPO/PLAN,
        REPO/"bayesfilter/highdim/ledh_canonical_score_stages_tf.py",
        REPO/"bayesfilter/highdim/ledh_ukf_lifecycle_tf.py",
        REPO/"bayesfilter/runtime/gpu_memory_policy.py"]
    manifest = dict(schema="iapf_curved_diagnostic_v1", plan=PLAN, output=str(output),
        result_file=str(output/"results.json"), command=sys.argv, cwd=str(REPO),
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        source_sha256={str(path.relative_to(REPO)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
        python=sys.executable, python_version=sys.version, runtime=runtime,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        data_version="scalar_sine_quadratic_fresh_20260918", master_seed=9182026,
        partitions=PARTITIONS, theta=THETA, regimes=REGIMES, jit_compile=True,
        reference_dtype="float64", filter_dtype="float32", fit_dtype="float64",
        reference_exception="CPU FP64 quadrature; FP64 fitting/FD diagnostic; GPU FP32 final kernels",
        status="running")
    write(output/"manifest.json", manifest)
    results = {"references": {}, "localization": {}, "fits": {}, "downstream": {}, "fd": {}}

    def save(stage):
        results["stage"] = stage
        results["budget"] = budget.record()
        write(output/"results.json", results)
        print(stage, budget.record(), flush=True)

    device = "/GPU:0" if args.device == "GPU" else "/CPU:0"
    try:
        prepared = {}
        for dataset, regime in zip(PARTITIONS["calibration"], REGIMES):
            observations, back, oracle = reference(dataset, regime)
            prepared[dataset] = observations, back, oracle
            results["references"][str(dataset)] = oracle
            with tf.device(device):
                results["localization"][str(dataset)] = localization(dataset, regime, observations, back, budget)
            save(f"localized {dataset}")
            records = {}
            for name in ARMS:
                with tf.device(device):
                    records[name] = consumer_fit(dataset, regime, observations, back, name, budget)
                results["fits"][str(dataset)] = records
                save(f"fit {dataset} {name}: {records[name]['status']}")
        nominations = {}
        for name in ("shape_10000", "wide_10000", "floor_001", "floor_1"):
            records = [results["fits"][str(dataset)][name] for dataset in PARTITIONS["calibration"]]
            valid = all(record["status"] == "valid" and not record["boundary_active"] for record in records)
            nominations[name] = dict(eligible=valid, worst_shape=max(max(record["with_floor_shape"])
                for record in records) if valid else None)
        eligible = [name for name, record in nominations.items() if record["eligible"]]
        chosen = min(eligible, key=lambda name: nominations[name]["worst_shape"]) if eligible else "shape_10000"
        selection = dict(chosen=chosen, status="nominated" if eligible else "diagnostic_representative_only",
            rule="minimum worst calibration predictive-grid residual among converged interior relative fits",
            candidates=nominations, frozen_controls=ARMS[chosen],
            tuned_scope_authority=False, confirmation_inspected=False)
        write(output/"selection.json", selection)
        results["selection"] = selection
        names = list(dict.fromkeys(["density_2000", "shape_10000", chosen]))
        save("controls frozen before validation/confirmation")
        for partition, datasets in PARTITIONS.items():
            for dataset, regime in zip(datasets, REGIMES):
                if partition == "calibration":
                    observations, back, oracle = prepared[dataset]
                else:
                    observations, back, oracle = reference(dataset, regime)
                    results["references"][str(dataset)] = oracle
                    records = {}
                    for name in names:
                        with tf.device(device):
                            records[name] = consumer_fit(dataset, regime, observations, back, name, budget)
                        results["fits"][str(dataset)] = records
                        save(f"{partition} fit {dataset} {name}: {records[name]['status']}")
                fitted = {name: results["fits"][str(dataset)][name] for name in names}
                with tf.device(device):
                    sizes = [16, 64, 256, 1024, 4096] if partition == "calibration" else [4096]
                    results["downstream"][str(dataset)] = downstream(dataset, regime, observations,
                        fitted, sizes, oracle, budget)
                    if partition == "calibration" and fitted["shape_10000"]["status"] == "valid":
                        results["fd"][str(dataset)] = finite_difference(dataset, regime,
                            observations, fitted["shape_10000"], budget)
                save(f"{partition} downstream {dataset} complete")
        results["status"] = "complete"
        save("complete")
        manifest.update(status="complete", memory_usage=memory_usage(args.device),
                        wall_seconds=time.monotonic()-started, budget=budget.record())
        write(output/"manifest.json", manifest)
    except BaseException as error:
        results["status"] = "stopped"
        results["exception"] = repr(error)
        save("stopped with preserved partial evidence")
        manifest.update(status="stopped", exception=repr(error), wall_seconds=time.monotonic()-started,
                        budget=budget.record())
        write(output/"manifest.json", manifest)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=("CPU", "GPU"), default="GPU")
    parser.add_argument("--wall-seconds", type=float, default=1800.)
    run(parser.parse_args())
