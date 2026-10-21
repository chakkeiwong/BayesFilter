"""Bounded CPU reference campaign for the experimental acceptance diagnostic.

Synthetic Markov traces calibrate the diagnostic; actual fixed HMC traces test
integration only. Neither can issue a canonical tuning or posterior artifact.
Run with GPU intentionally hidden. Estimator and generator kernels use TF/XLA;
legacy comparators and host report arithmetic retain their diagnostic execution.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

import tensorflow as tf

from bayesfilter.inference.hmc_acceptance_uncertainty import (
    AcceptanceUncertaintyPolicy, DECISIONS, acceptance_uncertainty_tensors,
    evaluate_acceptance_uncertainty,
)
from bayesfilter.inference.hmc_verification import temporal_block_conflicts


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False)+"\n")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def experimental_policy(n, *, method="lugsail", looks=1, candidates=1):
    return AcceptanceUncertaintyPolicy(batch_size=max(3, math.isqrt(n//4)),
        min_batches=8, family_alpha=.10, temporal_tolerance=.10, chain_tolerance=.10,
        planned_looks=looks, planned_candidates=candidates, method=method)


def wilson(count, total):
    if not total:
        return {"count": count, "total": total, "estimate": None, "interval": [None, None]}
    z = 1.959963984540054  # Derived standard normal .975 quantile.
    p = count/total
    denominator = 1+z*z/total
    center = (p+z*z/(2*total))/denominator
    half = z*math.sqrt(p*(1-p)/total+z*z/(4*total**2))/denominator
    return {"count": count, "total": total, "estimate": p,
            "interval": [max(0., center-half), min(1., center+half)]}


def rate(mask):
    return wilson(int(tf.reduce_sum(tf.cast(mask, tf.int32))), int(tf.size(mask)))


def exact_mean_variance(n, rho, amplitude):
    """Independent scalar covariance-of-a-sum reference, not estimated ESS."""
    return amplitude**2 * (n + 2*math.fsum((n-k)*rho**k for k in range(1, n))) / n**2


@lru_cache(maxsize=8)
def _markov_program(replications, n):
    @tf.function(input_signature=[tf.TensorSpec([2], tf.int32),
        tf.TensorSpec([], tf.float64)], autograph=False, jit_compile=True)
    def generate(seed, rho):
        uniforms = tf.random.stateless_uniform([replications, n, 4], seed, dtype=tf.float64)
        flips = tf.where(uniforms < (1.-rho)/2., -1., 1.)
        start = tf.where(tf.random.stateless_uniform([replications, 1, 4],
            tf.random.experimental.stateless_fold_in(seed, 1), dtype=tf.float64) < .5, -1., 1.)
        return tf.cast(start*tf.math.cumprod(flips, axis=1), tf.float64)
    return generate


def synthetic_cases():
    cases = []
    for n in (512, 8192):
        for rho in (0., .8, .95, .995, -.8):
            cases.append(dict(name=f"stationary-rho{rho}-n{n}", n=n, rho=rho,
                              mean=.7, amplitude=.25, mutation=None))
    for mutation in ("common_drift", "opposing_drift", "single_drift", "offset",
                     "boundary_shift", "oscillation", "initial_transient"):
        cases.append(dict(name=mutation, n=8192, rho=0., mean=.7,
                          amplitude=.05, mutation=mutation))
    for mean, amplitude, name in ((.02, .015, "near_zero"), (.98, .015, "near_one"),
                                 (.65, .25, "at_practical_boundary")):
        cases.append(dict(name=name, n=8192, rho=0., mean=mean,
                          amplitude=amplitude, mutation=None))
    return cases


def simulate(spec, *, replications, seed):
    n = spec["n"]
    signs = _markov_program(replications, n)(tf.constant(seed, tf.int32),
                                            tf.constant(spec["rho"], tf.float64))
    values = spec["mean"] + spec["amplitude"]*signs
    mutation = spec["mutation"]
    tick = tf.range(n)
    offset = tf.zeros([n, 4], tf.float64)
    if mutation in {"common_drift", "opposing_drift", "single_drift", "boundary_shift", "initial_transient"}:
        cutoff = n//2 + (n//8 if mutation == "boundary_shift" else 0)
        if mutation == "initial_transient":
            cutoff = n//16
        trend = tf.where(tick < cutoff, tf.constant(-.15, tf.float64), tf.constant(.15, tf.float64))
        weights = {"opposing_drift": [1., 1., -1., -1.],
                   "single_drift": [1., 0., 0., 0.]}.get(mutation, [1.]*4)
        offset = trend[:, None]*tf.constant(weights, tf.float64)[None]
    elif mutation == "offset":
        offset = tf.broadcast_to(tf.constant([[-.2, 0., 0., 0.]], tf.float64), [n, 4])
    elif mutation == "oscillation":
        offset = tf.broadcast_to(tf.where((tick//(n//4)) % 2 == 0,
            tf.constant(-.15, tf.float64), tf.constant(.15, tf.float64))[:, None], [n, 4])
    return values+offset[None]


def baseline_screens(values, *, low=.65, high=.75):
    r, n, m = map(int, values.shape)
    blocks = tf.transpose(tf.reduce_mean(tf.reshape(values[:, :n//4*4],
        [r, 4, n//4, m]), axis=2), [0, 2, 1])
    return {
        "raw_temporal_conflict": tf.reduce_any((tf.reduce_min(blocks, axis=2) < low)
                                               & (tf.reduce_max(blocks, axis=2) > high), axis=1),
        "paired_v6_temporal_conflict": temporal_block_conflicts(blocks, practical_width=high-low),
    }


def calibration_cell(spec, values, *, policy):
    report = acceptance_uncertainty_tensors(values, policy=policy)
    codes = report["decision_code"]
    result = {"case": spec, "policy": policy.payload(),
              "decisions": {name: rate(codes == i) for i, name in enumerate(DECISIONS)},
              "variance_estimate_available": rate(report["variance_estimate_available"]),
              "conflict": rate(report["temporal_conflict"] | report["chain_conflict"]),
              "baselines": {k: rate(v) for k, v in baseline_screens(values).items()}}
    if spec["mutation"] is None:
        mu = spec["mean"]
        low, high = tf.unstack(report["pooled_interval"], axis=-1)
        covered = report["variance_estimate_available"] & (low <= mu) & (high >= mu)
        joint = covered & tf.reduce_all(tf.abs(report["window_means"]-mu) <= report["window_half_width"], axis=(1, 2))
        joint &= tf.reduce_all(tf.abs(report["chain_means"]-mu) <= report["chain_half_width"], axis=1)
        actual = report["pooled_mean"]
        oracle_var = exact_mean_variance(spec["n"], spec["rho"], spec["amplitude"])/4
        iid_var = spec["amplitude"]**2/(4*spec["n"])
        result.update(pooled_coverage_all_planned=rate(covered), joint_coverage_all_planned=rate(joint),
            coverage_among_available=wilson(int(tf.reduce_sum(tf.cast(covered, tf.int32))),
                int(tf.reduce_sum(tf.cast(report["variance_estimate_available"], tf.int32)))),
            oracle_mean_variance=oracle_var,
            empirical_mean_squared_error=float(tf.reduce_mean((actual-mu)**2)),
            oracle_normal_90_coverage=rate(tf.abs(actual-mu) <= 1.6448536269514722*math.sqrt(oracle_var)),
            iid_normal_90_coverage=rate(tf.abs(actual-mu) <= 1.6448536269514722*math.sqrt(iid_var)))
    return result


def promotion_screens(rows):
    failures = []
    for row in rows:
        spec = row["case"]
        label = spec["name"]+":"+row["policy"]["method"]
        if spec["mutation"] is None:
            if row["joint_coverage_all_planned"]["interval"][0] < .90:
                failures.append(label+":joint_coverage_delivery")
            if row["conflict"]["interval"][1] > .05:
                failures.append(label+":false_conflict")
        elif spec["mutation"] in {"common_drift", "opposing_drift", "single_drift", "offset", "oscillation"}:
            if row["conflict"]["interval"][0] < .80:
                failures.append(label+":material_difference_detection")
        if spec["name"] == "stationary-rho0.0-n8192":
            if row["decisions"]["compatible_for_fresh_verification"]["interval"][0] < .80:
                failures.append(label+":compatibility_delivery")
    return {"default_promotion_supported": not failures, "failed_screens": failures}


def search_stress(seed, *, searches=128, candidates=8, mean=.855, rho=.995, seed_offset=710):
    """Actual repeated looks/selection with disjoint verification simulations."""
    n = 8192
    spec = dict(name="search-stress", n=n, rho=rho,
                mean=mean, amplitude=.14, mutation=None)
    measurements = simulate(spec, replications=searches*candidates, seed=(seed, seed_offset))
    fresh = simulate(spec, replications=searches*candidates, seed=(seed, seed_offset+1))
    rows = []
    for adjusted in (False, True):
        ever = tf.zeros([searches, candidates], tf.bool)
        naive = tf.zeros_like(ever)
        for length in (512, 2048, 8192):
            policy = experimental_policy(length, looks=4 if adjusted else 1,
                                         candidates=candidates if adjusted else 1)
            result = acceptance_uncertainty_tensors(measurements[:, :length], policy=policy)
            ever |= tf.reshape(result["decision_code"] == 5, [searches, candidates])
            naive |= tf.reshape((result["pooled_mean"] >= .65) & (result["pooled_mean"] <= .75),
                                [searches, candidates])
        # The fourth allocated look covers the fresh bank; select first passing
        # ordinal, never the most favorable descriptive acceptance score.
        verified = acceptance_uncertainty_tensors(fresh, policy=experimental_policy(n,
            looks=4 if adjusted else 1, candidates=candidates if adjusted else 1))["decision_code"] == 5
        chosen = tf.argmax(tf.cast(ever, tf.int32), axis=1, output_type=tf.int32)
        delivered = tf.reduce_any(ever, axis=1) & tf.gather_nd(tf.reshape(verified, [searches, candidates]),
            tf.stack([tf.range(searches), chosen], axis=1))
        rows.append({"adjusted": adjusted, "ever_compatible": rate(tf.reduce_any(ever, axis=1)),
            "fresh_delivery": rate(delivered), "naive_point_band_lottery": rate(tf.reduce_any(naive, axis=1))})
    return {"searches": searches, "candidates_per_search": candidates,
            "measurement_looks": [512, 2048, 8192], "verification_draws": n,
            "true_mean": mean, "rho": rho, "seed_measurement": [seed, seed_offset],
            "seed_verification": [seed, seed_offset+1], "rows": rows,
            "interpretation": "descriptive stress; marginal validity and declared allocation assumed"}


def sensitivity_diagnosis(seed, replications):
    spec = dict(name="bandwidth-rho0.995", n=8192, rho=.995,
                mean=.7, amplitude=.25, mutation=None)
    values = simulate(spec, replications=replications, seed=(seed, 810))
    rows = []
    for method in ("batch_means", "lugsail"):
        for batch in (16, 64, 128):
            p = replace(experimental_policy(8192, method=method), batch_size=batch)
            row = calibration_cell(spec, values, policy=p)
            row["seed"] = [seed, 810]
            rows.append(row)
    longer = simulate({**spec, "n": 9216}, replications=replications, seed=(seed, 811))
    p = replace(experimental_policy(8192, method="batch_means"), batch_size=128)
    first = acceptance_uncertainty_tensors(longer[:, :8192], policy=p)
    shifted = acceptance_uncertainty_tensors(longer[:, 1024:], policy=p)
    return {"rows": rows, "boundary_shift": {
        "offset_draws": 1024, "seed": [seed, 811], "policy": p.payload(),
        "decision_changed": rate(first["decision_code"] != shifted["decision_code"]),
        "conflict_changed": rate(first["temporal_conflict"] != shifted["temporal_conflict"]),
        "interpretation": "overlapping paired windows, not additional independent replications"},
        "promotion_role": "diagnosis_only; no post-hoc best batch selection"}


def replay_bgs(summary_path):
    from bayesfilter.inference.hmc_candidate_set_execution import _tensor_from_payload
    from bayesfilter.inference.hmc_verification import _acceptance_policy_from_payload
    summary = json.loads(Path(summary_path).read_text())
    checkpoint_paths = [Path(summary["source_checkpoint"]),
        Path(summary["source_checkpoint"]).parents[2]/"bgs-deadline-resume-09"/"tuning"/"controller_checkpoint.json"]
    observations = {}
    for path in checkpoint_paths:
        checkpoint = json.loads(path.read_text())
        for row in checkpoint["result"]["observations"]:
            observations[row["work_item_id"]] = row["observation"]
    results = []
    for row in summary["rows"]:
        saved = observations[row["work_item_id"]]
        tensors = {k: _tensor_from_payload(v) for k, v in saved["numerical_mechanics"].items()}
        start, end = saved["draw_range"]
        old = saved["acceptance_evidence"]
        ratio = tensors["log_accept_ratio"][start:end]
        kwargs = dict(samples=tensors["states"][start:end], log_accept_ratio=ratio,
            is_accepted=tensors["is_accepted"][start:end],
            health_policy=_acceptance_policy_from_payload(old["policy"]),
            native_divergence_status=old["native_divergence_status"],
            native_divergence_count=old["native_divergence_count"])
        if "support_rejection" in tensors:
            kwargs["support_rejection"] = tensors["support_rejection"][start:end]
        report = evaluate_acceptance_uncertainty(uncertainty_policy=experimental_policy(end-start), fixed_kernel=True, **kwargs)
        legacy = report["legacy_evidence"]
        if legacy["decision"] != old["decision"] or abs(legacy["pooled_mean"]-row["pooled_mean"]) > 1e-12:
            raise AssertionError("BGS legacy decision/mean replay mismatch")
        results.append({"candidate_id": row["candidate_id"], "epsilon": row["epsilon"],
            "L": row["leapfrog_steps"], "historical_decision": old["decision"],
            "log_accept_ratio_sha256": saved["numerical_mechanics"]["log_accept_ratio"]["sha256"],
            "legacy_replay_matches": True, "report": report})
    return {"summary": str(summary_path), "summary_sha256": digest(summary_path),
            "checkpoints": [{"path": str(p), "sha256": digest(p)} for p in checkpoint_paths], "rows": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replications", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--wall-seconds", type=float, default=3600.)
    parser.add_argument("--bgs-summary", type=Path)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise RuntimeError("CPU diagnostic campaign requires CUDA_VISIBLE_DEVICES=-1 before import")
    if args.replications < 32 or not 0 < args.wall_seconds <= 7200:
        raise ValueError("bounded replications and wall allocation required")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    root = Path(__file__).resolve().parents[1]
    sources = [Path(__file__), root/"inference/mcmc_uncertainty.py", root/"inference/hmc_acceptance_uncertainty.py",
               root/"inference/hmc_verification.py"]
    snapshot = args.output/"source"
    snapshot.mkdir()
    for path in sources:
        shutil.copyfile(path, snapshot/path.name)
    manifest = {"schema": "bayesfilter.acceptance_uncertainty_validation.v1",
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_checksums": {str(p): digest(p) for p in sources},
        "command": [sys.executable, "-m", "bayesfilter.testing.acceptance_uncertainty_validation", *sys.argv[1:]],
        "python": sys.version, "executable": sys.executable, "tensorflow": tf.__version__,
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
        "started_utc": datetime.now(timezone.utc).isoformat(), "source_snapshot": str(snapshot),
        "platform": platform.platform(), "device": "CPU reference; GPU intentionally hidden",
        "environment": {k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "TF_FORCE_GPU_ALLOW_GROWTH",
            "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS", "OPENBLAS_NUM_THREADS")},
        "seed": args.seed, "replications": args.replications, "wall_budget_seconds": args.wall_seconds,
        "plan": "docs/plans/bayesfilter-acceptance-uncertainty-validation-plan-2026-10-01.md",
        "result": str(args.output/"result.json"), "data_version": "bounded_symmetric_two_state_markov.v1",
        "jit_compile": True, "default_promoted": False, "status": "running"}
    write_json(args.output/"manifest.json", manifest)
    rows = []
    try:
        for index, spec in enumerate(synthetic_cases()):
            if time.monotonic()-started > args.wall_seconds:
                raise TimeoutError("calibration wall allocation exhausted between cells")
            values = simulate(spec, replications=args.replications, seed=(args.seed, index))
            for method in ("batch_means", "lugsail"):
                row = calibration_cell(spec, values, policy=experimental_policy(spec["n"], method=method))
                row["seed"] = [args.seed, index]
                rows.append(row)
                write_json(args.output/f"cell-{index:02d}-{method}.json", row)
            print(spec["name"], "complete", flush=True)
        result = {"cells": rows, "screens": promotion_screens(rows),
            "inference_status": "pointwise_descriptive_calibration; no universal/sequential coverage claim"}
        result["search_stress"] = [search_stress(args.seed), search_stress(args.seed,
            mean=.7, rho=0., seed_offset=720)]
        result["sensitivity_diagnosis"] = sensitivity_diagnosis(args.seed, args.replications)
        if args.bgs_summary:
            result["bgs_replay"] = replay_bgs(args.bgs_summary)
        write_json(args.output/"result.json", result)
        manifest["status"] = "complete"
    except BaseException as error:
        manifest.update(status="failed", error=type(error).__name__+": "+str(error))
        raise
    finally:
        manifest["wall_seconds"] = time.monotonic()-started
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(args.output/"manifest.json", manifest)


if __name__ == "__main__":
    main()
