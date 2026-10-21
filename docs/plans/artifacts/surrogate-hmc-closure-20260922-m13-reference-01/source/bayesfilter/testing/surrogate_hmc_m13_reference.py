"""Independent FP64 range/bearing likelihood references, not an HMC endpoint.

The fixed grid and finite alias bound are defined in the M13 reference plan.
No DCTR implementation is imported. Replicates preserve likelihood estimates,
rather than incorrectly averaging their logarithms.
"""
from __future__ import annotations

import math
import resource
import time
from pathlib import Path

import tensorflow as tf
import tensorflow_probability as tfp

F64 = tf.float64
SEED = 20260923
PARTICLES = (4096, 16384, 65536)
REPLICATES = 16
PLAN = "docs/plans/surrogate-hmc-m13-reference-2026-09-23.md"


def parameter_grid():
    step = math.log(2.)
    return [(a, b) for a in (-step, 0., step) for b in (-step, 0., step)]


def observation_log_density(position, observed, sigma):
    """Independent diagonal wrapped Gaussian; callers bind sigma <= .08."""
    radial = tf.sqrt(tf.reduce_sum(position * position, axis=-1) + 1e-12)
    angle = tf.atan2(position[..., 1], position[..., 0])
    pi = tf.constant(math.pi, F64)
    delta = tf.math.floormod(observed[1] - angle + pi, 2*pi) - pi
    aliases = tf.cast(tf.range(-2, 3), F64) * (2*pi)
    angular = (delta[..., None] + aliases) / sigma[1]
    angular_log = tf.reduce_logsumexp(-.5*angular*angular, axis=-1)
    radial_standard = (observed[0] - radial) / sigma[0]
    return (-.5*radial_standard*radial_standard + angular_log
            - tf.math.log(2*pi) - tf.reduce_sum(tf.math.log(sigma)))


def systematic_indices(weights, shifts):
    """One shift in [0,1) per replicate; no clipping of accepted weights."""
    cumulative = tf.cumsum(weights, axis=-1)
    cumulative = cumulative / cumulative[:, -1:]
    n = tf.shape(weights)[1]
    positions = (shifts[:, None] + tf.cast(tf.range(n), F64)) / tf.cast(n, F64)
    return tf.searchsorted(cumulative, positions, side="right", out_type=tf.int32)


def particle_filter(fixture, particles, replicates=REPLICATES, *, resample=True):
    """Batch-native independent reference replicates with one stable graph."""
    n, b = int(particles), int(replicates)
    horizon = int(fixture.observations.shape[0])
    a = tf.convert_to_tensor(fixture.A, F64)
    q_chol = tf.linalg.cholesky(tf.convert_to_tensor(fixture.Q, F64))
    p_chol = tf.linalg.cholesky(tf.convert_to_tensor(fixture.P0, F64))
    m0 = tf.convert_to_tensor(fixture.m0, F64)
    observations = tf.convert_to_tensor(fixture.observations, F64)
    base_sigma = tf.sqrt(tf.linalg.diag_part(tf.convert_to_tensor(fixture.R, F64)))
    log_n = tf.math.log(tf.cast(n, F64))

    @tf.function(input_signature=[tf.TensorSpec([2], F64), tf.TensorSpec([2], tf.int32)],
                 jit_compile=True, autograph=False)
    def evaluate(theta, seed):
        x = m0 + tf.linalg.matmul(
            tf.random.stateless_normal([b, n, 4], seed, dtype=F64),
            p_chol, transpose_b=True)
        log_weights = tf.fill([b, n], -log_n)
        ll = tf.zeros([b], F64)
        first = tf.zeros([b], F64)
        minimum_ess = tf.fill([b], tf.cast(n, F64))
        valid = tf.ones([b], tf.bool)
        sigma = base_sigma * tf.exp(theta[1])
        scale_valid = tf.reduce_all(tf.math.is_finite(sigma)) & tf.reduce_all(sigma > 0.) & (sigma[1] <= .080000000000001)

        def body(t, cloud, logw, total, first_value, minimum, ok):
            noise_seed = tf.stack([seed[0], seed[1] + 3*t + 1])
            noise = tf.random.stateless_normal([b, n, 4], noise_seed, dtype=F64)
            cloud = (tf.linalg.matmul(cloud, a, transpose_b=True)
                     + tf.exp(theta[0])*tf.linalg.matmul(noise, q_chol, transpose_b=True))
            obs_ll = observation_log_density(cloud[..., :2], observations[t], sigma)
            unnormalized = logw + obs_ll
            increment = tf.reduce_logsumexp(unnormalized, axis=1)
            weights = tf.nn.softmax(unnormalized, axis=1)
            ok = (ok & scale_valid & tf.math.is_finite(increment)
                  & tf.reduce_all(tf.math.is_finite(cloud), axis=[1, 2])
                  & tf.reduce_all(tf.math.is_finite(weights), axis=1))
            ess = 1. / tf.reduce_sum(weights*weights, axis=1)
            total = total + increment
            first_value = tf.where(t == 0, total, first_value)
            if resample:
                u = tf.random.stateless_uniform([b], tf.stack([seed[0], seed[1] + 3*t + 2]), dtype=F64)
                # Invalid rows are marked and cannot qualify a reference.
                safe_weights = tf.where(ok[:, None], weights, tf.ones_like(weights)/tf.cast(n, F64))
                indices = systematic_indices(safe_weights, u)
                index_valid = tf.reduce_all((indices >= 0) & (indices < n), axis=1)
                ok = ok & index_valid
                indices = tf.where(index_valid[:, None], indices, tf.zeros_like(indices))
                cloud = tf.gather(cloud, indices, batch_dims=1)
                logw = tf.fill([b, n], -log_n)
            else:
                logw = unnormalized - increment[:, None]
            return t+1, cloud, logw, total, first_value, tf.minimum(minimum, ess), ok

        _, _, _, ll, first, minimum_ess, valid = tf.while_loop(
            lambda t, *_: t < horizon, body,
            (tf.constant(0), x, log_weights, ll, first, minimum_ess, valid),
            parallel_iterations=1)
        return dict(first=first, log_likelihood=ll, minimum_ess=minimum_ess, valid=valid)
    return evaluate


def quadrature(fixture, intervals):
    n = int(intervals)
    mean = tf.linalg.matvec(fixture.A, fixture.m0)[:2]
    propagated = tf.linalg.matmul(tf.linalg.matmul(fixture.A, fixture.P0), fixture.A, transpose_b=True)
    obs = fixture.observations[0]
    base_sigma = tf.sqrt(tf.linalg.diag_part(fixture.R))
    z = tf.linspace(tf.constant(-8., F64), tf.constant(8., F64), n+1)
    z1, z2 = tf.meshgrid(z, z, indexing="ij")
    standard = tf.stack([tf.reshape(z1, [-1]), tf.reshape(z2, [-1])], axis=1)
    edge = tf.concat([tf.constant([.5], F64), tf.ones([n-1], F64), tf.constant([.5], F64)], axis=0)
    log_weights = tf.math.log(tf.reshape(edge[:, None]*edge[None, :], [-1]))

    @tf.function(input_signature=[tf.TensorSpec([2], F64)], jit_compile=True, autograph=False)
    def evaluate(theta):
        covariance = (propagated + tf.exp(2*theta[0])*fixture.Q)[:2, :2]
        sd = tf.sqrt(tf.linalg.diag_part(covariance))
        points = mean + standard * sd
        chol = tf.linalg.cholesky(covariance)
        solved = tf.linalg.triangular_solve(chol, tf.transpose(points - mean))
        log_prior = (-tf.math.log(tf.constant(2*math.pi, F64))
                     - tf.reduce_sum(tf.math.log(tf.linalg.diag_part(chol)))
                     - .5*tf.reduce_sum(solved*solved, axis=0))
        sigma = base_sigma * tf.exp(theta[1])
        log_like = observation_log_density(points, obs, sigma)
        value = tf.reduce_logsumexp(log_prior + log_like + log_weights) + tf.reduce_sum(tf.math.log(sd*(16/n)))
        # Five aliases each bounded by the peak of the same angular Gaussian.
        tail_bound = tf.constant(2*math.erfc(8/math.sqrt(2))*5/(2*math.pi), F64) / tf.reduce_prod(sigma)
        tail_relative = tail_bound*tf.exp(-value)
        valid = (tf.reduce_all(tf.math.is_finite(covariance))
                 & tf.reduce_all(tf.linalg.diag_part(covariance) > 0.)
                 & (covariance[0, 1] == 0.) & (covariance[1, 0] == 0.)
                 & tf.math.is_finite(value) & (sigma[1] <= .080000000000001))
        return dict(value=value, relative_tail_bound=tail_relative, valid=valid)
    return evaluate


def likelihood_interval(log_estimates, critical):
    values = tf.convert_to_tensor(log_estimates, F64)
    shift = tf.reduce_max(values)
    scaled = tf.exp(values-shift)
    mean = tf.reduce_mean(scaled)
    count = tf.cast(tf.size(values), F64)
    variance = tf.reduce_sum(tf.square(scaled-mean))/(count-1)
    se = tf.sqrt(variance/count)
    lower, upper = mean-critical*se, mean+critical*se
    finite = bool(tf.reduce_all(tf.math.is_finite(values)).numpy())
    positive = finite and float(lower.numpy()) > 0.
    center = float((shift+tf.math.log(mean)).numpy()) if finite else None
    lo = float((shift+tf.math.log(lower)).numpy()) if positive else None
    hi = float((shift+tf.math.log(upper)).numpy()) if positive else None
    return dict(log_mean=center, log_lower=lo, log_upper=hi,
                log_halfwidth=max(center-lo, hi-center) if positive else None,
                interval_positive=positive, finite=finite,
                coefficient_of_variation=float((tf.sqrt(variance)/mean).numpy()) if finite else None,
                replicate_likelihood_ess=float((tf.reduce_sum(scaled)**2/tf.reduce_sum(scaled*scaled)).numpy()) if finite else None,
                log_estimates=[float(v) if math.isfinite(float(v)) else None for v in values.numpy().tolist()])


def finite_list(value):
    return [float(v) if math.isfinite(float(v)) else None for v in value.numpy().tolist()]


def intervals_overlap(a, b):
    return (a["interval_positive"] and b["interval_positive"]
            and max(a["log_lower"], b["log_lower"]) <= min(a["log_upper"], b["log_upper"]))


def one_step_baselines(fixture, theta):
    mean = tf.linalg.matvec(fixture.A, fixture.m0)
    covariance = tf.linalg.matmul(tf.linalg.matmul(fixture.A, fixture.P0), fixture.A, transpose_b=True) + tf.exp(2*theta[0])*fixture.Q
    sigma = tf.sqrt(tf.linalg.diag_part(fixture.R))*tf.exp(theta[1])
    plugin = observation_log_density(mean[None, :2], fixture.observations[0], sigma)[0]
    px, py = mean[0], mean[1]
    r2 = px*px+py*py
    radial = tf.sqrt(r2+1e-12)
    jacobian = tf.stack([tf.stack([px/radial, py/radial, tf.constant(0., F64), tf.constant(0., F64)]),
                         tf.stack([-py/r2, px/r2, tf.constant(0., F64), tf.constant(0., F64)])])
    predictive = tf.linalg.matmul(tf.linalg.matmul(jacobian, covariance), jacobian, transpose_b=True) + tf.linalg.diag(sigma*sigma)
    residual = fixture.observations[0]-tf.stack([radial, tf.atan2(py, px)])
    shifts = tf.cast(tf.range(-2, 3), F64)*tf.constant(2*math.pi, F64)
    rows = residual[None, :] + tf.stack([tf.zeros_like(shifts), shifts], axis=1)
    gaussian = tfp.distributions.MultivariateNormalTriL(tf.zeros([2], F64), tf.linalg.cholesky(predictive))
    ekf = tf.reduce_logsumexp(gaussian.log_prob(rows))
    return dict(predicted_mean=float(plugin.numpy()), linearized_gaussian=float(ekf.numpy()))


def reference_stage(output: Path):
    from dataclasses import replace
    from experiments.dpf_implementation.tf_tfp.fixtures.range_bearing_tf import (
        build_range_bearing_fixture_tf, observation_log_density_tf)
    from bayesfilter.inference.surrogate_hmc_program import write_json
    start, cpu_start = time.monotonic(), resource.getrusage(resource.RUSAGE_SELF)
    fixture = build_range_bearing_fixture_tf()
    report = dict(stage="m13-reference", status="RUNNING", qualified=False,
                  likelihood_reference_qualified=False, posterior_reference_qualified=False,
                  inference_launched=False, all_models_validated=False, plan=PLAN,
                  scope=dict(model="M13", state_dimension=4, observation_dimension=2, horizon=20,
                             fixture=fixture.model_definition(), observations=fixture.observations.numpy().tolist(), parameter_grid=parameter_grid(),
                             scaling="Q,R multiplied by exp(2*lambda_Q),exp(2*lambda_R)", prior=None),
                  replicates=REPLICATES, particle_ladder=PARTICLES, seed=SEED,
                  alias_log_omitted_probability_upper_bound=math.log(2)-.5*(5*math.pi/.08)**2,
                  uncertainty="Student-t intervals for independent likelihood estimates; approximate coverage",
                  interval_alpha=.01/45, jit_compile=True, dtype="float64_reference",
                  numerical_rows=[], seed_schedule=[], source_independent_of_dctr=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, report)
    critical = float(tfp.distributions.StudentT(tf.constant(REPLICATES-1., F64),
                     tf.constant(0., F64), tf.constant(1., F64)).quantile(1-.01/(2*45)).numpy())
    report["student_critical_value"] = critical
    quad = {n: quadrature(fixture, n) for n in (128, 256, 512)}
    for index, pair in enumerate(parameter_grid()):
        theta = tf.constant(pair, F64)
        values = {}
        for n, evaluate in quad.items():
            r = evaluate(theta)
            values[str(n)] = {k: (bool(v.numpy()) if v.dtype == tf.bool else float(v.numpy())) for k, v in r.items()}
        finest = values["512"]
        comparison_points = tf.constant([[1.2, .7, .18, -.06], [-1., 1e-4, 0., 0.], [-1., -1e-4, 0., 0.]], F64)
        modified = replace(fixture, R=fixture.R*tf.exp(2*theta[1]))
        callback = observation_log_density_tf(modified, comparison_points, fixture.observations[0])
        independent = observation_log_density(comparison_points[:, :2], fixture.observations[0],
                                             tf.sqrt(tf.linalg.diag_part(modified.R)))
        callback_error = float(tf.reduce_max(tf.abs(callback-independent)/(1+tf.abs(callback))).numpy())
        qpass = (all(v["valid"] for v in values.values())
                 and abs(values["256"]["value"]-finest["value"]) <= 1e-6
                 and finest["relative_tail_bound"] <= 1e-8 and callback_error <= 1e-12)
        baselines = one_step_baselines(fixture, theta)
        report["numerical_rows"].append(dict(index=index, theta=pair, quadrature=values,
            quadrature_qualified=qpass, callback_scaled_error=callback_error,
            one_step_baselines=baselines, particle_levels={},
            baseline_absolute_log_errors={k:abs(v-finest["value"]) for k,v in baselines.items()}))
    write_json(output, report)
    if not all(r["quadrature_qualified"] for r in report["numerical_rows"]):
        report.update(status="M13_ONE_STEP_REFERENCE_NOT_QUALIFIED", next_action="Existing master: repair the independent one-step integral before the full-sequence ladder")
        return report

    pilot = particle_filter(fixture, 1024)
    for iteration in range(2):
        seed = [SEED, 11+iteration]
        tick, used = time.monotonic(), resource.getrusage(resource.RUSAGE_SELF)
        result = pilot(tf.zeros([2], F64), tf.constant(seed, tf.int32))
        healthy = bool(tf.reduce_all(result["valid"]).numpy())
        report["kernel_device"] = result["log_likelihood"].device
        healthy = healthy and "GPU" in report["kernel_device"]
        elapsed = time.monotonic()-tick
        now = resource.getrusage(resource.RUSAGE_SELF)
        cpu = now.ru_utime+now.ru_stime-used.ru_utime-used.ru_stime
        report["seed_schedule"].append(dict(purpose="timing_only", seed=seed))
        if not healthy:
            report.update(status="M13_REFERENCE_PILOT_INVALID")
            return report
    factor = len(parameter_grid())*sum(PARTICLES)/1024
    wall_forecast, cpu_forecast = elapsed*factor+180, cpu*factor+180
    current = resource.getrusage(resource.RUSAGE_SELF)
    used_cpu = current.ru_utime+current.ru_stime-cpu_start.ru_utime-cpu_start.ru_stime
    report["cost_forecast"] = dict(pilot_wall=elapsed, pilot_cpu=cpu,
        ladder_wall=wall_forecast, ladder_cpu=cpu_forecast, used_wall=time.monotonic()-start, used_cpu=used_cpu,
        admitted=wall_forecast < 2200-(time.monotonic()-start) and cpu_forecast < 1600-used_cpu)
    write_json(output, report)
    if not report["cost_forecast"]["admitted"]:
        report.update(status="M13_REFERENCE_COST_NOT_ADMITTED")
        return report

    # One unresampled path-importance comparator at the pilot particle count.
    path_reference = particle_filter(fixture, 1024, resample=False)
    filters = {n:particle_filter(fixture, n) for n in PARTICLES}
    for row in report["numerical_rows"]:
        index, theta = row["index"], tf.constant(row["theta"], F64)
        seed = [SEED, 70+index]
        paths = path_reference(theta, tf.constant(seed, tf.int32))
        row["prior_path_comparator"] = dict(first=likelihood_interval(paths["first"], critical),
            full=likelihood_interval(paths["log_likelihood"], critical),
            valid=bool(tf.reduce_all(paths["valid"]).numpy()),
            minimum_ess=finite_list(paths["minimum_ess"]))
        report["seed_schedule"].append(dict(purpose="prior_path", point=index, seed=seed))
        for tier, (n, evaluate) in enumerate(filters.items()):
            seed = [SEED, 701+index*10000+tier*1000]
            tick = time.monotonic()
            result = evaluate(theta, tf.constant(seed, tf.int32))
            row["particle_levels"][str(n)] = dict(first=likelihood_interval(result["first"], critical),
                full=likelihood_interval(result["log_likelihood"], critical),
                valid=bool(tf.reduce_all(result["valid"]).numpy()),
                minimum_ess=finite_list(result["minimum_ess"]), wall_seconds=time.monotonic()-tick)
            report["seed_schedule"].append(dict(purpose="reference", point=index, particles=n, seed=seed))
            write_json(output, report)
        levels = list(row["particle_levels"].values())
        final = levels[-1]
        oracle = row["quadrature"]["512"]["value"]
        first_interval = final["first"]
        oracle_inside = first_interval["interval_positive"] and first_interval["log_lower"] <= oracle <= first_interval["log_upper"]
        all_agree = all(intervals_overlap(levels[i]["full"], levels[j]["full"])
                        for i in range(len(levels)) for j in range(i+1, len(levels)))
        # A heuristic must be closer even than the interval's nearest point
        # to constitute supported reference underperformance at this oracle.
        nearest_error = (0. if oracle_inside else min(abs(first_interval[k]-oracle) for k in ("log_lower","log_upper"))
                         if first_interval["interval_positive"] else math.inf)
        heuristic_veto = any(error < nearest_error for error in row["baseline_absolute_log_errors"].values())
        path_first = row["prior_path_comparator"]["first"]
        path_upper_error = (max(abs(path_first[k]-oracle) for k in ("log_lower","log_upper"))
                            if path_first["interval_positive"] else math.inf)
        heuristic_veto = heuristic_veto or path_upper_error < nearest_error
        row["checks"] = dict(all_estimates_valid=all(r["valid"] for r in levels),
            one_step_oracle_inside=oracle_inside, full_sequence_level_agreement=all_agree,
            full_sequence_precision=final["full"]["interval_positive"] and final["full"]["log_halfwidth"] <= .02,
            no_supported_heuristic_underperformance=not heuristic_veto)
        row["qualified"] = all(row["checks"].values())
        row["heuristic_dominance"] = dict(verdict="NO_SUPPORTED_UNDERPERFORMANCE" if not heuristic_veto else "VETO",
            ranking_supported=False, scope="T1 oracle accuracy only; T20 comparator spread is descriptive")
        write_json(output, report)
    qualified = all(row["qualified"] for row in report["numerical_rows"])
    report.update(qualified=qualified, likelihood_reference_qualified=qualified,
        status="M13_FIXED_POINT_LIKELIHOOD_REFERENCE_QUALIFIED" if qualified else "M13_LIKELIHOOD_REFERENCE_NOT_QUALIFIED",
        next_action="Existing master: canonical M13 consumer and scope-specific controls; parameter-prior/domain and posterior reference remain prerequisites for HMC",
        wall_seconds=time.monotonic()-start,
        nonclaims=["parameter posterior reference", "HMC correctness", "DCTR accuracy", "full parameter-domain coverage", "method ranking", "all-model completion"])
    report["allocator"] = tf.config.experimental.get_memory_info("GPU:0")
    return report
