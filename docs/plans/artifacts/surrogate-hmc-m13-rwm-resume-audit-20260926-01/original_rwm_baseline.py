"""Independent CPU RWM comparator for the identical frozen M13 endpoint.

This is a classical diagnostic baseline. It never supplies surrogate forces,
HMC tuning authority, or a different inference target.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import time

import tensorflow as tf

from bayesfilter.inference.surrogate_hmc_program import ROOT, file_hash, write_json
from bayesfilter.testing import surrogate_hmc_m13_endpoint_mechanics as mechanics
from bayesfilter.testing.surrogate_hmc_m13_endpoint_materiality import _load
from bayesfilter.testing.surrogate_hmc_m13_four_chain_cost import batched_endpoint
from bayesfilter.testing.surrogate_hmc_m13_posterior_assessment import F64, LOOKS, assess
from bayesfilter.testing.surrogate_hmc_m13_posterior_reference import load_reference, tensor_record

PRIOR_SD = math.log(2.)
STARTS = ((-1., -1.), (-.33, .33), (.33, -.33), (1., 1.))
SEED = (20260926, 2810)
PLAN = "docs/plans/surrogate-hmc-m13-rwm-baseline-2026-09-26.md"


def random_walk_graph(endpoint, count, *, adapt):
    @tf.function(input_signature=[tf.TensorSpec([4, 2], F64), tf.TensorSpec([], F64),
        tf.TensorSpec([2, 2], F64), tf.TensorSpec([2], tf.int32)],
        jit_compile=True, autograph=False)
    def run(initial, log_scale, factor, seed):
        first = endpoint(initial)
        initial_valid = first.status["value_valid"] & tf.math.is_finite(first.potential)
        states = tf.TensorArray(F64, size=count, element_shape=[4, 2])
        proposals = tf.TensorArray(F64, size=count, element_shape=[4, 2])
        log_accepts = tf.TensorArray(F64, size=count, element_shape=[4])
        accepts = tf.TensorArray(tf.bool, size=count, element_shape=[4])
        validities = tf.TensorArray(tf.bool, size=count, element_shape=[4])
        def body(i, state, potential, scale, states, proposals, log_accepts, accepts, validities):
            step_seed = tf.random.experimental.stateless_fold_in(seed, i)
            seeds = tf.random.experimental.stateless_split(step_seed, 2)
            noise = tf.random.stateless_normal([4, 2], seeds[0], dtype=F64)
            proposal = state+tf.exp(scale)*tf.linalg.matmul(noise, factor, transpose_b=True)
            result = endpoint(proposal)
            valid = (result.status["value_valid"] & tf.math.is_finite(result.potential) &
                     tf.reduce_all(tf.math.is_finite(proposal), -1))
            log_accept = tf.minimum(tf.constant(0., F64), potential-result.potential)
            log_accept = tf.where(valid, log_accept, tf.constant(-float("inf"), F64))
            accepted = tf.math.log(tf.random.stateless_uniform([4], seeds[1], dtype=F64)) < log_accept
            state = tf.where(accepted[:, None], proposal, state)
            potential = tf.where(accepted, result.potential, potential)
            if adapt:
                rate = tf.pow(tf.cast(i+10, F64), tf.constant(-.6, F64))
                scale += rate*(tf.reduce_mean(tf.cast(accepted, F64))-.25)
            return (i+1, state, potential, scale, states.write(i, state),
                proposals.write(i, proposal), log_accepts.write(i, log_accept),
                accepts.write(i, accepted), validities.write(i, valid))
        result = tf.while_loop(lambda i, *_: i < count, body,
            (tf.constant(0), initial, first.potential, log_scale, states,
             proposals, log_accepts, accepts, validities), parallel_iterations=1)
        return dict(final_state=result[1], final_potential=result[2], log_scale=result[3],
            states=result[4].stack(), proposals=result[5].stack(), log_accept=result[6].stack(),
            accepted=result[7].stack(), valid=result[8].stack(), initial_valid=initial_valid,
            initial_state=initial)
    return run


def health(chunk):
    moved = tf.reduce_any(chunk["states"] != chunk["initial_state"][None], [0, 2])
    return bool(tf.reduce_all(chunk["initial_valid"]) & tf.reduce_all(chunk["valid"]) &
        tf.reduce_all(tf.math.is_finite(chunk["states"])) &
        tf.reduce_all(tf.math.is_finite(chunk["log_accept"])) &
        tf.math.is_finite(chunk["log_scale"]) & tf.reduce_all(moved))


def archive(root, name, chunk):
    directory = root/name
    directory.mkdir()
    return {key: tensor_record(directory/(key+".tfbin"), value) for key, value in chunk.items()}


def conditional_table(chunks):
    state = tf.concat([tf.concat([c["initial_state"][None], c["states"][:-1]], 0)
                       for c in chunks], 0)
    accepted = tf.concat([c["accepted"] for c in chunks], 0)
    log_accept = tf.concat([c["log_accept"] for c in chunks], 0)
    extent = tf.reduce_max(tf.abs(state/PRIOR_SD), -1)
    masks = (extent <= 1., (extent > 1.) & (extent < 3.), extent >= 3.)
    rows = []
    for name, mask in zip(("nominal", "intermediate", "outer"), masks):
        count = int(tf.reduce_sum(tf.cast(mask, tf.int32)))
        rows.append(dict(region=name, proposals=count,
            acceptance_rate=float(tf.reduce_mean(tf.cast(tf.boolean_mask(accepted, mask), F64))) if count else None,
            mean_acceptance_probability=float(tf.reduce_mean(tf.exp(tf.boolean_mask(log_accept, mask)))) if count else None,
            evidence="observed" if count else "not visited; no conditional sampler claim"))
    return rows


def run_baseline(output_dir, reference_path, coverage_path):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    reference, _, _ = load_reference(reference_path)
    calibration = json.loads(Path(coverage_path).read_text())
    if calibration.get("qualified") is not True:
        raise ValueError("RWM comparison needs qualified diagnostic intervals")
    _, fixture, _ = _load()
    if fixture.observation_checksum != reference["observation_checksum"]:
        raise ValueError("RWM/reference data mismatch")
    _, evaluator = mechanics.endpoint_components(fixture)
    endpoint = batched_endpoint(evaluator)
    initial = tf.constant(STARTS, F64)*PRIOR_SD
    factor = tf.eye(2, dtype=F64)*PRIOR_SD
    scale = tf.constant(math.log(.5), F64)
    seed = tf.constant(SEED)
    graph = random_walk_graph(endpoint, 512, adapt=True)
    warm1 = graph(initial, scale, factor, tf.random.experimental.stateless_fold_in(seed, 1))
    records = {"metric_warmup": archive(out, "metric-warmup", warm1)}
    if not health(warm1):
        raise ValueError("RWM initial adaptive pilot health failed; archive preserved")
    cloud = tf.reshape(warm1["states"][-256:], [-1, 2])
    centered = cloud-tf.reduce_mean(cloud, 0)
    empirical = tf.linalg.matmul(centered, centered, transpose_a=True)/tf.cast(tf.shape(cloud)[0]-1, F64)
    covariance = .9*empirical+.1*PRIOR_SD**2*tf.eye(2, dtype=F64)
    factor = tf.linalg.cholesky(covariance)
    warm2 = graph(warm1["final_state"], tf.constant(0., F64), factor,
                  tf.random.experimental.stateless_fold_in(seed, 2))
    records["step_warmup"] = archive(out, "step-warmup", warm2)
    if not health(warm2):
        raise ValueError("RWM step adaptation health failed; archive preserved")
    scale = warm2["log_scale"]
    fresh = random_walk_graph(endpoint, 500, adapt=False)(
        initial, scale, factor, tf.random.experimental.stateless_fold_in(seed, 3))
    records["fresh_warmup"] = archive(out, "fresh-warmup", fresh)
    if not health(fresh):
        raise ValueError("RWM fresh warmup health failed; archive preserved")
    state, chunks, assessments = fresh["final_state"], [], []
    retained = random_walk_graph(endpoint, 250, adapt=False)
    for index in range(8):
        chunk = retained(state, scale, factor, tf.random.experimental.stateless_fold_in(seed, 10+index))
        records[f"retained_{index}"] = archive(out, f"retained-{index:02}", chunk)
        chunks.append(chunk)
        if not health(chunk):
            raise ValueError("RWM retained chunk health failed; archive preserved")
        state = chunk["final_state"]
        count = 250*(index+1)
        if count in LOOKS:
            points = tf.transpose(tf.concat([c["states"] for c in chunks], 0), [1, 0, 2])
            comparison = assess(points, reference, calibration=calibration)
            assessments.append(comparison)
            print("RWM", count, comparison["status"], "Rhat", comparison["rhat"], flush=True)
    acceptance = tf.reduce_mean(tf.cast(tf.concat([c["accepted"] for c in chunks], 0), F64), 0)
    report = dict(plan=PLAN, status="M13_RWM_BASELINE_COMPLETE", health_passed=True,
        qualified=assessments[-1]["status"] == "EQUIVALENT", assessments=assessments,
        acceptance_per_chain=acceptance.numpy().tolist(), conditional_behavior=conditional_table(chunks),
        heuristic_role="same frozen endpoint classical sampler comparator",
        reference_sha256=reference["reference_sha256"], coverage_sha256=file_hash(Path(coverage_path)),
        target=dict(data_seed=1704, observation_checksum=fixture.observation_checksum,
            endpoint_particles=mechanics.REFERENCE_PARTICLES, endpoint_replicates=mechanics.REFERENCE_REPLICATES,
            endpoint_seed=mechanics.REFERENCE_SEED.numpy().tolist(), prior_sd=PRIOR_SD,
            endpoint_dtype="float64", prior_support="R^2"),
        covariance=covariance.numpy().tolist(), scale=float(tf.exp(scale)),
        warmup_excluded=True, warmup_counts=[512, 512, 500], draws_per_chain=2000,
        seed=list(SEED), archives=records, wall_seconds=time.monotonic()-started,
        ranking_supported=False, numerical_backend="tensorflow_only", jit_compile=True,
        cpu_reference_exception=True, retained_trace_count=retained.experimental_get_tracing_count())
    write_json(out/"report.json", report)
    return report
