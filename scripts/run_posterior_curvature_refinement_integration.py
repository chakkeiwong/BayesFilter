"""Standalone CPU synthetic verification, independent of MacroFinance.

All evaluated targets use analytical TensorFlow values/scores. NumPy is confined
to deterministic fixtures and independent covariance reference checks. This
driver does not implement a production numerical backend or a sampler. Host-XLA
coverage applies only to its Gaussian target, not the refiner or an HMC chain.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
    raise RuntimeError("this synthetic verification requires CUDA_VISIBLE_DEVICES=-1 before import")
if os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH") != "true":
    raise RuntimeError("set TF_FORCE_GPU_ALLOW_GROWTH=true before import")

import numpy as np
import tensorflow as tf

from bayesfilter.inference import (
    PosteriorCurvatureRefinementConfig,
    refine_posterior_local_curvature,
)


def gaussian(mean, precision):
    """Analytical Gaussian target; precision is raw negative-log curvature."""
    mean_tf = tf.constant(mean, tf.float64)
    precision_tf = tf.constant(precision, tf.float64)

    def callback(theta):
        delta = theta - mean_tf
        score = -delta @ precision_tf
        return 0.5 * tf.reduce_sum(delta * score, axis=1), score

    return callback


def all_eligible(theta):
    return tf.ones(tf.shape(theta)[0], tf.bool)


def banana(theta):
    horizontal, vertical = theta[:, 0], theta[:, 1]
    residual = vertical - 2.0 * (horizontal**2 - 1.0)
    value = -0.5 * horizontal**2 - 0.5 * residual**2
    return value, tf.stack((-horizontal + 4.0 * horizontal * residual, -residual), axis=1)


def quartic(theta):
    value = -0.5 * tf.reduce_sum(theta**2, axis=1) - 0.02 * tf.reduce_sum(theta**4, axis=1)
    return value, -theta - 0.08 * theta**3


def radial_quartic(theta):
    """A local quadratic fits while a full Gaussian explores stronger curvature."""
    squared_radius = tf.reduce_sum(theta**2, axis=1)
    return (-0.5 * squared_radius - 0.025 * squared_radius**2,
            -(1.0 + 0.1 * squared_radius[:, None]) * theta)


def mixture_saddle(theta):
    right = -0.5 * (theta[:, 0] - 3.0)**2
    left = -0.5 * (theta[:, 0] + 3.0)**2
    components = tf.stack((right, left), axis=1)
    value = tf.reduce_logsumexp(components, axis=1) - tf.math.log(tf.constant(2.0, tf.float64)) - 0.5 * theta[:, 1]**2
    weight_right = tf.nn.softmax(components, axis=1)[:, 0]
    score = tf.stack((6.0 * weight_right - 3.0 - theta[:, 0], -theta[:, 1]), axis=1)
    return value, score


def bounded_eligibility(theta):
    return theta[:, 0] <= 0.75


def bounded_sentinel(theta):
    """Finite sentinel cannot be detected by a finiteness-only caller."""
    valid = bounded_eligibility(theta)
    values = -0.5 * tf.reduce_sum(theta**2, axis=1)
    return tf.where(valid, values, tf.constant(-1e100, tf.float64)), tf.where(valid[:, None], -theta, tf.zeros_like(theta))


def score_check(callback, dimension):
    """Independent autodiff of the supplied value catches inconsistent fixtures."""
    points = tf.reshape(tf.linspace(tf.constant(-0.4, tf.float64), tf.constant(0.6, tf.float64), 16 * dimension), [16, dimension])
    with tf.GradientTape() as tape:
        tape.watch(points)
        values, scores = callback(points)
        total = tf.reduce_sum(values)
    automatic = tape.gradient(total, points)
    difference = float(tf.reduce_max(tf.abs(automatic - scores)).numpy())
    scale = max(1.0, float(tf.reduce_max(tf.abs(automatic)).numpy()))
    return difference / scale


def source_provenance(root):
    paths = [
        "bayesfilter/__init__.py", "bayesfilter/inference/__init__.py",
        "bayesfilter/inference/score_curvature_tf.py",
        "bayesfilter/inference/posterior_curvature_refinement.py",
        "scripts/run_posterior_curvature_refinement_integration.py",
    ]
    return {
        "root": str(root),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "source_sha256": {path: hashlib.sha256((root / path).read_bytes()).hexdigest() for path in paths},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"will not overwrite prior evidence: {args.output}")
    started = time.monotonic()
    payload = {
        "schema": "bayesfilter.posterior_curvature_refinement.integration.v1",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "CPU-only synthetic verification; GPU deliberately hidden",
        "numpy_role": "independent fixtures and reference checks only",
        "tensorflow_version": tf.__version__, "passed": False,
        "environment": {name: os.environ.get(name) for name in (
            "CUDA_VISIBLE_DEVICES", "TF_FORCE_GPU_ALLOW_GROWTH", "BAYESFILTER_PRELOAD_CUSTOM_OP",
            "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS", "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS", "PYTHONPATH",
        )},
        "provenance": source_provenance(Path(__file__).resolve().parents[1]),
        "nonclaims": ["no MacroFinance target", "no training or HMC", "no whitening or posterior convergence",
                      "no GPU qualification", "no full-refiner or full-chain XLA claim"],
        "cases": [],
    }
    results = {}

    def run_case(name, callback, center, pilot, expected_status="eligible_for_local_position_factor",
                 reference=None, eligibility=all_eligible, seed=20260908,
                 fit_design="uniform_box"):
        case_start = time.monotonic()
        dimension = len(center)
        config = PosteriorCurvatureRefinementConfig(
            rows_per_partition=max(48, 4 * dimension), batch_size=16,
            max_physical_rows=10000, seed=seed, fit_design=fit_design,
        )
        record = {"name": name, "dimension": dimension, "config": asdict(config),
                  "expected_status": expected_status, "passed": False}
        payload["cases"].append(record)
        try:
            fixture_error = score_check(callback, dimension)
            result = refine_posterior_local_curvature(
                callback, center, pilot, batched_eligibility_fn=eligibility, config=config,
            )
            results[name] = result
            data = result.payload()
            diagnostics = data["diagnostics"]
            for replicate in diagnostics.get("replicates", []):
                replicate.pop("precision_z", None)
            record.update(status=result.status, accepted=result.accepted, diagnostics=diagnostics,
                          fixture_score_relative_error=fixture_error,
                          center_unchanged=np.array_equal(result.center.numpy(), center))
            geometry_ok = True
            if reference is not None:
                error = None if not result.accepted else float(
                    np.linalg.norm(result.refined_covariance.numpy() - reference) / np.linalg.norm(reference)
                )
                record["covariance_relative_error"] = error
                record["covariance_relative_tolerance"] = 2e-7
                geometry_ok = error is not None and error <= 2e-7
            if not result.accepted:
                geometry_ok = geometry_ok and result.refined_factor is None and result.refined_covariance is None and result.precision_z is None
            record["passed"] = (result.status == expected_status and geometry_ok and
                                fixture_error <= 1e-12 and record["center_unchanged"])
        except Exception as error:
            record["exception"] = f"{type(error).__name__}: {error}"
        finally:
            record["elapsed_seconds"] = time.monotonic() - case_start
        print(json.dumps({key: record.get(key) for key in ("name", "passed", "status", "exception", "elapsed_seconds")}), flush=True)

    try:
        precision = np.array([[3.0, 0.6], [0.6, 1.5]])
        reference = np.linalg.inv(precision)
        run_case("shifted_1d_gaussian", gaussian([0.7], [[4.0]]), np.array([-0.3]), np.array([[0.1]]), reference=np.array([[0.25]]))
        run_case("correlated_gaussian", gaussian([0.7, -0.4], precision), np.zeros(2), np.array([[1.3, 0.0], [0.2, 0.9]]), reference=reference)
        run_case(
            "correlated_gaussian_uniform_ball",
            gaussian([0.7, -0.4], precision), np.zeros(2),
            np.array([[1.3, 0.0], [0.2, 0.9]]), reference=reference,
            fit_design="uniform_ball", seed=20260918,
        )
        run_case("small_pilot_gaussian", gaussian(np.zeros(2), precision), np.zeros(2), np.diag([0.03, 0.04]), reference=reference)
        run_case("large_pilot_gaussian", gaussian(np.zeros(2), precision), np.zeros(2), np.diag([4.0, 0.5]), reference=reference)
        rotation, _ = np.linalg.qr(np.random.default_rng(819).normal(size=(6, 6)))
        spectrum = np.geomspace(1.0, 1e6, 6)
        hard_precision = (rotation * spectrum) @ rotation.T
        hard_covariance = (rotation / spectrum) @ rotation.T
        run_case("six_order_correlated_gaussian", gaussian(np.zeros(6), hard_precision), np.zeros(6), np.eye(6), reference=hard_covariance)
        indices = np.arange(142)
        large_covariance = 0.7 ** np.abs(indices[:, None] - indices[None, :])
        run_case("142d_correlated_gaussian", gaussian(np.zeros(142), np.linalg.inv(large_covariance)),
                 np.zeros(142), np.diag(np.linspace(0.5, 2.0, 142)), reference=large_covariance)
        run_case("142d_correlated_gaussian_uniform_ball", gaussian(np.zeros(142), np.linalg.inv(large_covariance)),
                 np.zeros(142), np.diag(np.linspace(0.5, 2.0, 142)), reference=large_covariance,
                 fit_design="uniform_ball", seed=20260912)
        run_case("ball_local_fit_gaussian_proposal_rejected", radial_quartic,
                 np.zeros(24), np.eye(24), "refined_proposal_rejected",
                 fit_design="uniform_ball", seed=20260912)
        run_case("ball_mixture_saddle", mixture_saddle, np.zeros(2), np.eye(2),
                 "curvature_fit_rejected", fit_design="uniform_ball", seed=20260912)
        run_case("ball_support_expansion", bounded_sentinel, np.zeros(2), np.eye(2) * 0.01,
                 "ineligible_target_row", eligibility=bounded_eligibility,
                 fit_design="uniform_ball", seed=20260912)
        run_case("banana_saddle", banana, np.zeros(2), np.eye(2), "curvature_fit_rejected")
        run_case("mild_quartic", quartic, np.zeros(2), np.eye(2))
        run_case("mixture_saddle", mixture_saddle, np.zeros(2), np.eye(2), "curvature_fit_rejected")
        run_case("flat_direction", gaussian(np.zeros(2), np.diag([1.0, 0.0])), np.zeros(2), np.eye(2), "curvature_fit_rejected")
        run_case("bounded_support_sentinel", bounded_sentinel, np.zeros(2), np.eye(2), "ineligible_target_row", eligibility=bounded_eligibility)
        eager_target = gaussian(np.zeros(2), precision)
        xla_target = tf.function(eager_target, input_signature=[tf.TensorSpec([16, 2], tf.float64)], jit_compile=True)
        probe = tf.reshape(tf.linspace(tf.constant(-0.6, tf.float64), tf.constant(0.8, tf.float64), 32), [16, 2])
        eager_values, eager_scores = eager_target(probe)
        xla_values, xla_scores = xla_target(probe)
        run_case("host_xla_gaussian", xla_target, np.zeros(2), np.eye(2), reference=reference)
        run_case("host_eager_gaussian", eager_target, np.zeros(2), np.eye(2), reference=reference)
        xla_result, eager_result = results["host_xla_gaussian"], results["host_eager_gaussian"]
        geometry_error = float(tf.reduce_max(tf.abs(xla_result.refined_covariance - eager_result.refined_covariance)).numpy())
        value_error = float(tf.reduce_max(tf.abs(eager_values - xla_values)).numpy())
        score_error = float(tf.reduce_max(tf.abs(eager_scores - xla_scores)).numpy())
        traces = xla_target.experimental_get_tracing_count()
        payload["target_only_host_xla"] = {
            "value_max_abs_error": value_error, "score_max_abs_error": score_error,
            "refined_covariance_max_abs_error": geometry_error, "trace_count": traces,
            "fixed_batch_shape": [16, 2],
            "passed": max(value_error, score_error, geometry_error) <= 1e-10 and traces == 1,
        }
        payload["passed"] = payload["target_only_host_xla"]["passed"] and all(case["passed"] for case in payload["cases"])
    except Exception as error:
        payload["driver_exception"] = f"{type(error).__name__}: {error}"
    finally:
        payload["elapsed_seconds"] = time.monotonic() - started
        payload["finished_utc"] = datetime.now(timezone.utc).isoformat()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            stream.write(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"passed": payload["passed"], "cases": len(payload["cases"]), "output": str(args.output), "elapsed_seconds": payload["elapsed_seconds"]}), flush=True)
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
