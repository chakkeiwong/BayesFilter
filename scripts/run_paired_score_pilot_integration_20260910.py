"""Independent small CPU integration fixtures, with no MacroFinance dependency.

These fixtures test local curvature and rejection, not posterior geometry or
NeuTra/HMC quality. Closed-form Hessians are independent verification oracles.
The canonical initializer, including exact incumbent selection, is exercised.
"""

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
    raise RuntimeError("this integration is an intentionally CPU-only diagnostic")

import tensorflow as tf

from bayesfilter.inference.batched_quadratic_center import (
    BatchedQuadraticCenterConfig,
    refine_batched_quadratic_center,
)


def fixtures():
    rotation, _ = tf.linalg.qr(tf.constant([[1., 2., 4.], [-2., 1., 3.], [3., -1., 1.]], tf.float64))
    precision = rotation @ tf.linalg.diag(tf.constant([.03, 4., 3e4], tf.float64)) @ tf.transpose(rotation)

    def gaussian(points):
        score = -points @ precision
        return .5 * tf.reduce_sum(points * score, axis=1), score, tf.ones([4], tf.bool)

    yield "rotated_narrow_gaussian", gaussian, [0., 0., 0.], True, precision

    def coupled(coefficient):
        def callback(points):
            first, second = points[:, 0], points[:, 1]
            potential = .5 * first**2 + .25 * second**2 + coefficient * first * second**2
            potential += .25 * (first**4 + second**4)
            score = -tf.stack((first + coefficient * second**2 + first**3,
                               .5 * second + 2 * coefficient * first * second + second**3), axis=1)
            return -potential, score, tf.ones([4], tf.bool)
        return callback

    yield "cubic_quartic_mode", coupled(1.), [0., 0.], True, tf.linalg.diag(tf.constant([1., .5], tf.float64))
    yield "strong_cubic_coupling", coupled(1000.), [0., 0.], False, None

    def rosenbrock(points):
        first, second = points[:, 0], points[:, 1]
        residual = second - first**2
        potential = (1 - first)**2 + 100 * residual**2
        gradient = tf.stack((2 * (first - 1) - 400 * first * residual, 200 * residual), axis=1)
        return -potential, -gradient, tf.ones([4], tf.bool)

    yield "rosenbrock_mode", rosenbrock, [1., 1.], True, tf.constant([[802., -400.], [-400., 200.]], tf.float64)
    yield "rosenbrock_indefinite", rosenbrock, [0., 1.], False, None

    def funnel(points):
        log_scale, location = points[:, 0], points[:, 1]
        conditional_precision = tf.exp(-log_scale)
        potential = log_scale**2 / 18 + .5 * log_scale + .5 * conditional_precision * location**2
        score = tf.stack((-log_scale / 9 - .5 + .5 * conditional_precision * location**2,
                          -conditional_precision * location), axis=1)
        return -potential, score, tf.ones([4], tf.bool)

    yield "funnel_mode", funnel, [-4.5, 0.], True, tf.linalg.diag(tf.constant([1 / 9, 90.01713130052181], tf.float64))
    yield "funnel_extreme_scale", funnel, [-30., 0.], False, None

    def boundary(points):
        eligible = points[:, 0] > 0
        return -.5 * tf.reduce_sum(points**2, axis=1), -points, eligible

    yield "support_boundary", boundary, [.0005, 0.], False, None

    def cancellation(points):
        values = tf.reduce_sum(1e20 * points - .5 * points**2, axis=1)
        return values, 1e20 - points, tf.ones([4], tf.bool)

    yield "cancellation_sensitive_score", cancellation, [0., 0.], False, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    started = time.monotonic()
    payload = {"started_utc": datetime.now(timezone.utc).isoformat(), "gpu_intentionally_hidden": True,
               "runtime": "canonical BayesFilter TF; CPU diagnostic; ordinary-graph localization exception",
               "seed": 20260910, "cases": [], "hmc_transitions": 0, "training_updates": 0}
    for name, callback, center, expected, oracle in fixtures():
        compiled = tf.function(callback, input_signature=[tf.TensorSpec([4, len(center)], tf.float64)], autograph=False)
        case_started = time.monotonic()
        result = refine_batched_quadratic_center(compiled, center, [1.] * len(center),
                  config=BatchedQuadraticCenterConfig(pilot_method="paired_local", jit_compile_trust=False, max_fit_rounds=2))
        checks = {"expected_decision": result.accepted == expected,
                  "bounded_rows": result.diagnostics["physical_rows"] <= result.diagnostics["planned_physical_rows"],
                  "no_rejected_factor": result.accepted or result.pilot_factor is None,
                  "reusable_target_graph": compiled.experimental_get_tracing_count() == 1}
        if expected and result.accepted:
            relative = tf.linalg.norm(result.precision_z - oracle) / tf.linalg.norm(oracle)
            reference_covariance = tf.linalg.solve(oracle, tf.eye(len(center), dtype=tf.float64))
            covariance = result.pilot_factor @ tf.transpose(result.pilot_factor)
            checks["closed_form_precision"] = bool(relative < 1e-5)
            checks["closed_form_covariance"] = bool(tf.linalg.norm(covariance-reference_covariance) / tf.linalg.norm(reference_covariance) < 1e-4)
            checks["fixed_mode"] = bool(tf.reduce_all(result.center == tf.constant(center, tf.float64)))
        case = {"case": name, "expected_accepted": expected, "passed": all(checks.values()),
                "checks": checks, "result": result.payload(), "elapsed_seconds": time.monotonic()-case_started}
        payload["cases"].append(case)
        print(json.dumps({key: case[key] for key in ("case", "passed", "checks", "elapsed_seconds")}), flush=True)
    payload.update(passed=all(case["passed"] for case in payload["cases"]), elapsed_seconds=time.monotonic()-started,
                   nonclaims=["not CCMA admission, default/GPU readiness, NeuTra training, or posterior accuracy"])
    with arguments.output.open("x") as output:
        json.dump(payload, output, indent=2, allow_nan=False)
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
