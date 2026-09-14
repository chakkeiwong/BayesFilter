"""Standalone non-MacroFinance synthetic initializer verification; CPU diagnostic."""

from __future__ import annotations

import argparse
import json
import os
import resource
import time
from datetime import datetime, timezone
from itertools import pairwise
from pathlib import Path

if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1" or os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH") != "true":
    raise RuntimeError("this diagnostic requires hidden GPUs and memory growth before import")

import tensorflow as tf

from bayesfilter.inference.batched_local_center import BatchedLocalCenterConfig
from bayesfilter.inference.batched_quadratic_center import (
    BatchedQuadraticCenterConfig,
    initialize_batched_posterior_local_location_scale,
    refine_batched_quadratic_center,
)
from bayesfilter.inference.posterior_curvature_refinement import (
    PosteriorCurvatureRefinementConfig,
    refine_posterior_local_curvature,
)


def gaussian_target(target_mean, precision):
    @tf.function(input_signature=[tf.TensorSpec([4, 3], tf.float64)], autograph=False)
    def target(points):
        difference = points - target_mean
        scores = -difference @ precision
        return 0.5 * tf.reduce_sum(difference * scores, axis=1), scores, tf.ones([4], tf.bool)
    return target


def nonlinear_target(name):
    @tf.function(input_signature=[tf.TensorSpec([4, 3], tf.float64)], autograph=False)
    def target(points):
        if name == "quartic":
            values = -tf.reduce_sum(0.5*points**2 + 0.02*points**4, axis=1)
            return values, -points-0.08*points**3, tf.ones([4], tf.bool)
        if name == "banana":
            first, second, third = tf.unstack(points, axis=1)
            residual = second-0.3*first**2
            values = -0.5*(first**2 + residual**2 + third**2)
            scores = tf.stack((-first+0.6*first*residual, -residual, -third), axis=1)
            return values, scores, tf.ones([4], tf.bool)
        valid = tf.reduce_all(tf.abs(points) < 0.03, axis=1)
        return tf.where(valid, -0.5*tf.reduce_sum(points**2, axis=1), -1e200), -points, valid
    return target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    started = time.monotonic()
    started_utc = datetime.now(timezone.utc).isoformat()
    cases = []
    dimension = 3
    mean = tf.constant([0.2, -0.3, 0.1], tf.float64)
    base_factor = tf.constant([[0.4, 0., 0.], [0.1, 0.7, 0.], [-0.2, 0.1, 0.5]], tf.float64)

    for name, units in (("correlated_gaussian", [1., 1., 1.]),
                        ("extreme_coordinate_units", [1e-7, 1e3, 1.])):
        units = tf.constant(units, tf.float64)
        factor = units[:, None] * base_factor
        target_mean = units * mean
        precision = tf.linalg.cholesky_solve(factor, tf.eye(dimension, dtype=tf.float64))

        target = gaussian_target(target_mean, precision)
        result = refine_batched_quadratic_center(target, target_mean + units, units,
                                                 config=BatchedQuadraticCenterConfig(centeredness_cap=1e-9))
        center_error = tf.reduce_max(tf.abs((result.center - target_mean) / units))
        covariance_error = tf.constant(float("inf"), tf.float64)
        if result.accepted:
            actual = result.pilot_factor @ tf.transpose(result.pilot_factor) / units[:, None] / units[None, :]
            expected = base_factor @ tf.transpose(base_factor)
            covariance_error = tf.linalg.norm(actual-expected) / tf.linalg.norm(expected)
        checks = {"accepted": result.accepted, "center": bool(center_error <= 1e-7),
                  "factor": bool(covariance_error <= 1e-7),
                  "bounded": result.diagnostics["physical_rows"] <= result.diagnostics["planned_physical_rows"]}
        cases.append({"case": name, "passed": all(checks.values()), "checks": checks, "result": result.payload()})

    @tf.function(input_signature=[tf.TensorSpec([4, 3], tf.float64)], autograph=False)
    def normal(points):
        return -0.5*tf.reduce_sum(points**2, axis=1), -points, tf.ones([4], tf.bool)

    starts = tf.constant([[1., 1., 1.], [-1., -1., -1.], [2., -1., 0.], [-2., 0., 1.]], tf.float64)
    composed = initialize_batched_posterior_local_location_scale(normal, starts, tf.ones(3, tf.float64),
                                                                 locator_config=BatchedLocalCenterConfig(
                                                                     trust_refinement_rounds=1, jit_compile=False))
    cases.append({"case": "independent_starts", "passed": composed.accepted, "result": composed.payload()})
    capped = initialize_batched_posterior_local_location_scale(normal, starts, tf.ones(3, tf.float64),
                                                               locator_config=BatchedLocalCenterConfig(
                                                                   trust_refinement_rounds=1, jit_compile=False,
                                                                   max_optimizer_callback_batches_per_round=1))
    cases.append({"case": "cap_veto", "passed": not capped.accepted and capped.pilot_factor is None,
                  "result": capped.payload()})

    for name in ("quartic", "banana", "restricted_support"):
        nonlinear = nonlinear_target(name)
        center = tf.zeros(3, tf.float64) if name == "restricted_support" else tf.ones(3, tf.float64)
        result = refine_batched_quadratic_center(nonlinear, center, tf.ones(3, tf.float64))
        values = [float(entry["anchor_value"]) for entry in result.diagnostics["rounds"]]
        checks = {"monotone_exact_anchors": all(second >= first for first, second in pairwise(values)),
                  "bounded": result.diagnostics["physical_rows"] <= result.diagnostics["planned_physical_rows"],
                  "no_false_factor": result.accepted or result.pilot_factor is None}
        checks["expected_disposition"] = not result.accepted if name == "restricted_support" else result.accepted
        cases.append({"case": name, "passed": all(checks.values()), "checks": checks, "result": result.payload()})

    curvature = None
    if composed.accepted:
        curvature = refine_posterior_local_curvature(
            lambda points: normal(points)[:2], composed.center, composed.pilot_factor,
            batched_eligibility_fn=lambda points: tf.ones([4], tf.bool),
            config=PosteriorCurvatureRefinementConfig(batch_size=4, rows_per_partition=32))
    cases.append({"case": "independent_fixed_center_handoff",
                  "passed": curvature is not None and curvature.accepted and bool(tf.reduce_all(curvature.center == composed.center)),
                  "result": None if curvature is None else curvature.payload()})
    payload = {"passed": all(case["passed"] for case in cases), "cases": cases,
               "started_utc": started_utc, "ended_utc": datetime.now(timezone.utc).isoformat(),
               "elapsed_seconds": time.monotonic()-started, "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
               "gpu_intentionally_hidden": True, "nonclaims": ["no MacroFinance model", "no GPU/full-initializer XLA or posterior claim"]}
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with arguments.output.open("x") as output:
        json.dump(payload, output, indent=2, allow_nan=False)
    print(json.dumps({"passed": payload["passed"], "cases": [{"case": case["case"], "passed": case["passed"]} for case in cases],
                      "elapsed_seconds": payload["elapsed_seconds"], "peak_rss_kib": payload["peak_rss_kib"]}))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
