"""Diagnostic same-data endpoint precision and MH materiality; no admission."""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import tensorflow as tf

from bayesfilter.inference.surrogate_hmc_program import ROOT, file_hash, write_json
from bayesfilter.testing.surrogate_hmc_m13_controls import partition_fixture, control_points
from bayesfilter.testing.surrogate_hmc_m13_endpoint_mechanics import prior_log_density, serial
from bayesfilter.testing.surrogate_hmc_m13_trajectory_reference import trajectory_importance_reference

PLAN = "docs/plans/surrogate-hmc-m13-endpoint-materiality-2026-09-25.md"
BASE = ROOT / "docs/plans/artifacts/surrogate-hmc-closure-20260922-m13-endpoint-mechanics-04"
OLD_SEED = (2026092402, 9101)
FRESH_SEEDS = ((2026092501, 9101), (2026092502, 9101), (2026092503, 9101))
MARGIN = 0.01
F64 = tf.float64


@tf.function(input_signature=[tf.TensorSpec([None, None], F64)],
             jit_compile=True, autograph=False)
def likelihood_statistics(log_estimates):
    """Delta-method log-mean MCSE and normalized replicate influences."""
    n = tf.cast(tf.shape(log_estimates)[1], F64)
    log_mean = tf.reduce_logsumexp(log_estimates, axis=1)-tf.math.log(n)
    influence = tf.exp(log_estimates-log_mean[:, None])-1.0
    centered = influence-tf.reduce_mean(influence, axis=1, keepdims=True)
    mcse = tf.sqrt(tf.reduce_sum(tf.square(centered), axis=1)/(n*(n-1.0)))
    return log_mean, mcse, influence


@tf.function(input_signature=[tf.TensorSpec([None, None], F64),
                             tf.TensorSpec([None, 2], tf.int32)],
             jit_compile=True, autograph=False)
def pair_statistics(log_estimates, pairs):
    log_mean, _, influence = likelihood_statistics(log_estimates)
    old, new = pairs[:, 0], pairs[:, 1]
    difference = tf.gather(log_mean, new)-tf.gather(log_mean, old)
    paired = tf.gather(influence, new)-tf.gather(influence, old)
    paired -= tf.reduce_mean(paired, axis=1, keepdims=True)
    n = tf.cast(tf.shape(log_estimates)[1], F64)
    mcse = tf.sqrt(tf.reduce_sum(paired*paired, axis=1)/(n*(n-1.0)))
    return difference, mcse


def accepted_path(report):
    """Only accepted states identify proposal endpoints in the saved schema."""
    hmc = report["hmc"]
    if not (report["status"] == "M13_SURROGATE_HMC_MECHANICS_PASSED"
            and hmc.get("loop_jit_compile") is True
            and len(hmc["positions"]) == 24
            and len(hmc["accepted"]) == 24
            and all(row == [True] for row in hmc["accepted"])):
        raise ValueError("requires all 24 accepted compiled mechanics transitions")
    return [[0.0, 0.0]]+[row[0] for row in hmc["positions"]]


def _load():
    path = BASE / "m13-endpoint-mechanics/report.json"
    d = json.loads(path.read_text())
    parent = json.loads((BASE / "report.json").read_text())
    snapshot = json.loads((BASE / "run-manifest.json").read_text())
    if not parent.get("source_unchanged"):
        raise ValueError("baseline sources changed while running")
    for name in ("bayesfilter/testing/surrogate_hmc_m13_trajectory_reference.py",
                 "bayesfilter/testing/surrogate_hmc_m13_wrapped_reference.py",
                 "bayesfilter/testing/surrogate_hmc_m13_reference.py",
                 "bayesfilter/testing/surrogate_hmc_m13_controls.py",
                 "experiments/dpf_implementation/tf_tfp/fixtures/range_bearing_tf.py"):
        if snapshot["source_hashes"].get(name) != file_hash(ROOT/name):
            raise ValueError("baseline reference dependency changed: "+name)
    fixture = partition_fixture("calibration", attempt=4)
    if d["scope"]["observation_checksum"] != fixture.observation_checksum:
        raise ValueError("same-data endpoint check observation checksum mismatch")
    return d, fixture, path


def materiality_stage(output):
    started = time.monotonic()
    out = Path(output).parent
    out.mkdir(parents=True, exist_ok=True)
    baseline, fixture, baseline_path = _load()
    positions = accepted_path(baseline)
    rows = [dict(label=f"transition-state-{i}", situation="trajectory", theta=p)
            for i, p in enumerate(positions)]
    for regime, points in control_points().items():
        rows += [dict(label=f"{regime}-{i}", situation=regime, theta=p)
                 for i, p in enumerate(points.numpy().tolist())]
    values = tf.constant([row["theta"] for row in rows], F64)
    evaluations = {}
    all_valid = True
    for particles in (4096, 16384):
        evaluate = trajectory_importance_reference(fixture, particles, 8, full_support=True)
        seeds = (OLD_SEED, *FRESH_SEEDS) if particles == 4096 else FRESH_SEEDS
        for seed in seeds:
            key = f"n{particles}-s{seed[0]}"
            records = []
            for i, row in enumerate(rows):
                result = evaluate(values[i], tf.constant(seed, tf.int32))
                logs = result["log_likelihood"]
                valid = bool(tf.reduce_all(result["valid"] & tf.math.is_finite(logs)))
                all_valid &= valid
                records.append(row | dict(log_likelihood=serial(logs), valid=valid,
                    minimum_ess=serial(result["minimum_ess"]),
                    maximum_weight=serial(result["proposal_maximum_normalized_weight"])))
                write_json(out/(key+".json"), dict(particles=particles, replicates=8,
                    seed=list(seed), data_checksum=fixture.observation_checksum,
                    complete=len(records)==len(rows), rows=records))
                if not valid:
                    return dict(status="M13_ENDPOINT_MATERIALITY_INVALID", qualified=False,
                        plan=PLAN, reason="nonfinite_or_invalid_importance_reference",
                        invalid_point=row, particles=particles, seed=list(seed),
                        posterior_qualified=False, endpoint_qualified=False,
                        inference_launched=False)
            evaluations[key] = tf.constant([row["log_likelihood"] for row in records], F64)
            print(key, "complete", len(records), flush=True)
    reference = evaluations[f"n4096-s{OLD_SEED[0]}"]
    baseline_mean, baseline_mcse, _ = likelihood_statistics(reference)
    pairs = tf.constant([[i, i+1] for i in range(24)], tf.int32)
    kinetic_change = tf.squeeze(tf.constant(baseline["hmc"]["final_kinetic"], F64)
                                -tf.constant(baseline["hmc"]["initial_kinetic"], F64), -1)
    saved_log_ratio = -tf.squeeze(tf.constant(baseline["hmc"]["delta_h"], F64), -1)
    saved_alpha = tf.exp(tf.minimum(saved_log_ratio, 0.0))
    prior = prior_log_density(values)
    prior_difference = prior[1:25]-prior[:24]
    base_difference, base_pair_mcse = pair_statistics(reference, pairs)
    summaries, checks = {}, []
    for particles in (4096, 16384):
        logs = tf.concat([evaluations[f"n{particles}-s{seed[0]}"] for seed in FRESH_SEEDS], axis=1)
        means, mcse, _ = likelihood_statistics(logs)
        differences, paired_mcse = pair_statistics(logs, pairs)
        log_ratio = differences+prior_difference-kinetic_change
        alpha = tf.exp(tf.minimum(log_ratio, 0.0))
        low = tf.exp(tf.minimum(log_ratio-3.0*paired_mcse, 0.0))
        high = tf.exp(tf.minimum(log_ratio+3.0*paired_mcse, 0.0))
        envelope_change = tf.maximum(tf.abs(low-saved_alpha), tf.abs(high-saved_alpha))
        passes = envelope_change <= MARGIN
        checks.append(bool(tf.reduce_all(passes)))
        summaries[str(particles)] = serial(dict(
            log_likelihood=means, mcse=mcse,
            difference_from_fixed_seed=means-baseline_mean,
            difference_combined_mcse=tf.sqrt(mcse**2+baseline_mcse**2),
            pair_log_likelihood_difference=differences, pair_mcse=paired_mcse,
            acceptance_probability=alpha, acceptance_probability_change=alpha-saved_alpha,
            acceptance_change_3mcse_envelope=envelope_change,
            local_screen_pass=passes, maximum_acceptance_envelope=tf.reduce_max(envelope_change)))
    potential64 = -(baseline_mean+prior)
    rounded = tf.cast(tf.cast(potential64, tf.float32), F64)
    log_ratio64 = potential64[:24]-potential64[1:25]-kinetic_change
    log_ratio32 = rounded[:24]-rounded[1:25]-kinetic_change
    alpha64, alpha32 = (tf.exp(tf.minimum(x, 0.0)) for x in (log_ratio64, log_ratio32))
    sources = {str(Path(__file__).relative_to(ROOT)): file_hash(Path(__file__)), PLAN:file_hash(ROOT/PLAN)}
    result = dict(status="M13_ENDPOINT_LOCAL_MATERIALITY_PASSED" if all(checks) else
                  "M13_ENDPOINT_LOCAL_MATERIALITY_REPAIR_REQUIRED", qualified=all(checks),
        qualification_scope="local_acceptance_precision_screen_only", plan=PLAN,
        inference_launched=False, posterior_qualified=False, endpoint_qualified=False,
        data_seed=1704, observation_checksum=fixture.observation_checksum,
        baseline_report=str(baseline_path), baseline_sha256=file_hash(baseline_path),
        points=rows, same_data_verified=True, all_reference_valid=all_valid,
        baseline_log_likelihood=serial(baseline_mean), baseline_log_likelihood_mcse=serial(baseline_mcse),
        baseline_pair_mcse=serial(base_pair_mcse), saved_acceptance_probability=serial(saved_alpha),
        fresh_seed_groups=[list(x) for x in FRESH_SEEDS], pooled_replicates=24,
        comparator=summaries, criterion=dict(acceptance_probability_margin=MARGIN,
            uncertainty="pointwise approximate three paired MCSE envelope; not certified coverage"),
        casting=serial(dict(maximum_potential_rounding=tf.reduce_max(tf.abs(rounded-potential64)),
            maximum_log_acceptance_change=tf.reduce_max(tf.abs(log_ratio32-log_ratio64)),
            maximum_acceptance_probability_change=tf.reduce_max(tf.abs(alpha32-alpha64)),
            same_seed_replay_log_ratio_change=tf.reduce_max(tf.abs(log_ratio32-saved_log_ratio)))),
        explanatory_only=["nominal/stress likelihood uncertainty", "importance ESS", "maximum importance weights"],
        source_hashes=sources, wall_seconds=time.monotonic()-started,
        nonclaims=["no posterior correctness or convergence", "no control selection", "no endpoint admission", "no sampler ranking"],
        next_action="Existing master: repair reference if material; otherwise prepare same-data posterior precision and scoped controls.")
    write_json(output, result)
    return result
