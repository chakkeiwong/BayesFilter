"""Diagnostic selected-filter arithmetic at the actual four-chain HMC starts.

The candidate force remains the existing runtime binding. FP64 runs the same
finite filter on exactly the physical inputs obtained by the candidate chart.
This module issues no tuning or posterior authority.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
from pathlib import Path
import time

import tensorflow as tf

from bayesfilter.highdim.covariance_proposal_score_batch_tf import make_batched_score_filter
from bayesfilter.testing import surrogate_hmc_five_model as runtime


TOLERANCE = .01
STARTS = (-.15, -.05, .05, .15)


def scaled_errors(candidate, reference):
    """Return per-entry owner-materiality errors in FP64."""
    candidate, reference = tf.cast(candidate, tf.float64), tf.cast(reference, tf.float64)
    return tf.abs(candidate-reference)/tf.maximum(tf.ones_like(reference), tf.abs(reference))


def compare(candidate, reference, *, tolerance=TOLERANCE):
    finite = bool(tf.reduce_all(tf.math.is_finite(candidate)).numpy() and
                  tf.reduce_all(tf.math.is_finite(reference)).numpy())
    errors = scaled_errors(candidate, reference)
    return dict(finite=finite, candidate=runtime.json_ready(candidate),
                reference=runtime.json_ready(reference),
                scaled_errors=runtime.json_ready(errors),
                maximum_scaled_error=float(tf.reduce_max(errors).numpy()) if finite else None,
                tolerance=tolerance,
                passed=finite and bool(tf.reduce_all(errors <= tolerance).numpy()))


def run(case, selection, output):
    """One selected full-scope diagnostic; save intermediate and terminal states."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    selected = runtime.load_selection(case, selection)
    positions = tf.constant([[v]*case.p for v in STARTS], tf.float64)
    source_paths = [Path(__file__), Path(runtime.__file__)]
    from bayesfilter.highdim import covariance_proposal_score_batch_tf as batch
    source_paths.append(Path(batch.__file__))
    source_hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    result = dict(status="RUNNING", model=case.name, scope=case.scope(),
        selection=str(Path(selection).resolve()), selection_hash=case.selection_hash,
        controls=asdict(case.controls), beta=runtime.json_ready(case.beta),
        positions=runtime.json_ready(positions), precision_policy=case.precision_policy,
        tolerance=TOLERANCE, n=runtime.N, horizon=case.horizon,
        source_sha256=source_hashes,
        data_sha256=hashlib.sha256(case.data_path.read_bytes()).hexdigest(),
        input_sha256=[hashlib.sha256(tf.io.serialize_tensor(x).numpy()).hexdigest()
                      for x in (positions, case.observations, *case.fixed, case.beta)],
        source_selection_status=selected["status"], candidate_rows=[],
        reference_rows=[], binding=None, comparisons=None,
        posterior_sampling_complete=False, canonical_admission=False,
        reference_role="same_finite_program_FP64_not_model_oracle",
        starts_role="initial_region_only_not_posterior_coverage")
    destination = output / "selected-consumer-precision.json"
    def save():
        runtime.save(destination, result)
    save()
    dtype = case.force_dtype
    spec64 = case.spec
    if case.name == "range_bearing":
        from scripts.run_range_bearing_covariance_comparison import specs
        _, spec64 = specs(tf, tf.float64, case.horizon)
    candidate = make_batched_score_filter(case.spec, runtime.N, case.horizon,
        case.controls, dtype, jit_compile=True, batch_size=2,
        precision_policy=case.precision_policy)
    reference = make_batched_score_filter(spec64, runtime.N, case.horizon,
        case.controls, tf.float64, jit_compile=True, batch_size=2,
        precision_policy="uniform")
    start = time.monotonic()
    actual = runtime.bindings(case)["force"].function(positions)
    tf.nest.map_structure(lambda x: x.numpy(), actual)
    result["actual_binding_seconds_cold"] = time.monotonic()-start
    result["binding"] = runtime.json_ready(actual._asdict())
    save()
    values_c, scores_c, forces_c, values_r, scores_r, forces_r = [], [], [], [], [], []
    health_c, health_r = [], []
    for group in range(2):
        q = positions[group*2:group*2+2]
        physical, jacobian = case.chart(tf.cast(q, dtype))
        c = candidate(physical, case.observations, *case.fixed, case.beta)
        r = reference(tf.cast(physical, tf.float64),
            tf.cast(case.observations, tf.float64),
            *[tf.cast(x, tf.float64) for x in case.fixed],
            tf.cast(case.beta, tf.float64))
        tf.nest.map_structure(lambda x: x.numpy(), (c, r))
        jac64 = tf.cast(jacobian, tf.float64)
        # Match the consumer multiplication order for the wiring check.
        fc = q-tf.cast(jacobian*c["score"], tf.float64)
        fr = q-jac64*r["score"]
        values_c.append(c["value"]); scores_c.append(c["score"]); forces_c.append(fc)
        values_r.append(r["value"]); scores_r.append(r["score"]); forces_r.append(fr)
        hc = tf.logical_and(c["valid"], c["steps_completed"] == case.horizon)
        hr = tf.logical_and(r["valid"], r["steps_completed"] == case.horizon)
        health_c.append(hc); health_r.append(hr)
        result["candidate_rows"].append(runtime.json_ready(dict(value=c["value"],
            score=c["score"], force=fc, valid=hc, physical_parameters=physical)))
        result["reference_rows"].append(runtime.json_ready(dict(value=r["value"],
            score=r["score"], force=fr, valid=hr)))
        result["completed_groups"] = group+1
        save()
    values_c, scores_c, forces_c = [tf.concat(rows, 0) for rows in (values_c,scores_c,forces_c)]
    values_r, scores_r, forces_r = [tf.concat(rows, 0) for rows in (values_r,scores_r,forces_r)]
    complete_c = tf.concat(health_c,0)
    complete_r = tf.concat(health_r,0)
    result["health"] = dict(candidate=runtime.json_ready(complete_c),
        reference=runtime.json_ready(complete_r),
        actual_binding=runtime.json_ready(actual.force_valid))
    result["comparisons"] = dict(
        likelihood=compare(values_c, values_r),
        score=compare(scores_c, scores_r),
        force=compare(forces_c, forces_r),
        binding_wiring=compare(actual.force, forces_c, tolerance=1e-6))
    passed = bool(tf.reduce_all(complete_c).numpy() and
                  tf.reduce_all(complete_r).numpy() and
                  tf.reduce_all(actual.force_valid).numpy())
    passed &= all(v["passed"] for v in result["comparisons"].values())
    result["status"] = ("SELECTED_CONSUMER_PRECISION_PASS" if passed else
                        "SELECTED_CONSUMER_PRECISION_OR_HEALTH_VETO")
    result["wall_seconds"] = time.monotonic()-start
    save()
    return result
