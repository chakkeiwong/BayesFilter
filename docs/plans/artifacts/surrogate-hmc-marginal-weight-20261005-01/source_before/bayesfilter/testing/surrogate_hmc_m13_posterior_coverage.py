"""Diagnostic coverage checks for the frozen finite-grid M13 reference law."""
from __future__ import annotations

import resource
from pathlib import Path

import tensorflow as tf

from bayesfilter.inference.surrogate_hmc_program import write_json
from bayesfilter.testing.surrogate_hmc_m13_staged_assessment import (
    F64, LOOKS, METHOD, critical_value, draw_bilinear, estimates_graph, assessment_scope,
)
from bayesfilter.testing.surrogate_hmc_m13_posterior_reference import load_reference

ENSEMBLES = 256
BATCH = 16
REFRESH = (1., .5, .2, .05)
SEED = (20260926, 2710)


def coverage_lower(successes, trials, alpha=.05):
    """Exact one-sided binomial lower bound by inversion of the beta CDF."""
    if not 0 <= successes <= trials or trials <= 0:
        raise ValueError("invalid binomial counts")
    if successes == 0:
        return 0.
    low, high = tf.constant(0., F64), tf.constant(1., F64)
    aa, bb = tf.constant(float(successes), F64), tf.constant(float(trials-successes+1), F64)
    # A scalar diagnostic inversion, with a fixed number of iterations.
    for _ in range(60):
        middle = (low+high)/2
        below = tf.math.betainc(aa, bb, middle) < alpha
        low, high = tf.where(below, middle, low), tf.where(below, high, middle)
    return float((low+high)/2)


def make_draw_graph(axis, density, *, batch=BATCH, count=2000):
    @tf.function(input_signature=[tf.TensorSpec([], F64), tf.TensorSpec([2], tf.int32)],
                 jit_compile=True, autograph=False)
    def draw(probability, seed):
        seeds = tf.random.experimental.stateless_split(seed, 2)
        independent = draw_bilinear(axis, density, tf.constant([batch, 4, count]), seeds[0])
        refresh = tf.random.stateless_uniform([batch, 4, count], seeds[1], dtype=F64) < probability
        index = tf.where(refresh, tf.range(count)[None, None, :], 0)
        latest = tf.transpose(tf.scan(tf.maximum, tf.transpose(index, [2, 0, 1])), [1, 2, 0])
        # Index zero initializes every chain from the exact law, whether its
        # first refresh flag is true or false. Holding preserves this law.
        return tf.gather(independent, latest, axis=2, batch_dims=2)
    return draw


def run_coverage(reference_path, output, *, looks=LOOKS, seed=SEED):
    looks = tuple(looks)
    method = assessment_scope(looks)
    initial_seed = tuple(seed)
    reference, axis, density = load_reference(reference_path)
    mean, sd, cuts = [tf.constant(reference[k], F64) for k in ("mean", "sd", "cuts")]
    expected = tf.constant([0., 0., 1., 1.]+[.25, .5, .75]*2, F64)
    critical = critical_value(looks=looks)
    generate = make_draw_graph(axis, density, count=max(looks))
    conditions = []
    for condition, probability in enumerate(REFRESH):
        joint_success = 0
        component_success = tf.zeros([len(looks), 10], tf.int32)
        worst_standardized = tf.zeros([len(looks), 10], F64)
        finite = True
        for batch in range(ENSEMBLES//BATCH):
            seed = tf.random.experimental.stateless_fold_in(tf.constant(initial_seed), condition*100+batch)
            points = generate(tf.constant(probability, F64), seed)
            peak_host_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
            write_json(Path(output).with_name("coverage-progress.json"), dict(
                condition=condition, refresh_probability=probability, batch=batch,
                peak_host_bytes=peak_host_bytes, completed_conditions=conditions))
            if peak_host_bytes > 8*1024**3:
                raise MemoryError("reference coverage exceeded its 8 GiB host-memory guard")
            all_covered = tf.ones([BATCH], tf.bool)
            covered_components, standardized = [], []
            for count in looks:
                stats = estimates_graph(count)(points[:, :, :count], mean, sd, cuts)
                finite = finite and bool(tf.reduce_all(stats["finite"]))
                error = tf.abs(stats["estimates"]-expected)
                covered = (error <= critical*stats["mcse"]) & stats["finite"][:, None]
                all_covered &= tf.reduce_all(covered, -1)
                covered_components.append(tf.reduce_sum(tf.cast(covered, tf.int32), 0))
                standardized.append(tf.reduce_max(error/stats["mcse"], 0))
            joint_success += int(tf.reduce_sum(tf.cast(all_covered, tf.int32)))
            component_success += tf.stack(covered_components)
            worst_standardized = tf.maximum(worst_standardized, tf.stack(standardized))
        lower = coverage_lower(joint_success, ENSEMBLES)
        row = dict(refresh_probability=probability, joint_successes=joint_success,
            ensembles=ENSEMBLES, observed_joint_coverage=joint_success/ENSEMBLES,
            one_sided_95_lower=lower, finite=finite,
            component_successes=component_success.numpy().tolist(),
            maximum_standardized_errors=worst_standardized.numpy().tolist(),
            passed=finite and lower >= .95)
        conditions.append(row)
        print("coverage", probability, joint_success, "/", ENSEMBLES,
              "lower", round(lower, 6), flush=True)
    report = dict(method=method, qualified=all(row["passed"] for row in conditions),
        reference_sha256=reference["reference_sha256"], looks=list(looks),
        conditions=conditions, ensembles=ENSEMBLES, batch=BATCH, seed=list(initial_seed),
        calibration_scope="stationary IID and exact-law sticky-refresh M13 controls",
        extrapolation_to_arbitrary_chains=False, numerical_backend="tensorflow_only",
        jit_compile=True, draw_trace_count=generate.experimental_get_tracing_count(),
        peak_host_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        peak_host_limit_bytes=8*1024**3,
        decision="qualify scoped intervals" if all(row["passed"] for row in conditions)
                 else "repair interval evidence; no posterior verdict")
    write_json(output, report)
    return report
