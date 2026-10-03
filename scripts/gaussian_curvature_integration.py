"""Standalone synthetic curvature checks; no MacroFinance imports or data."""

import argparse
import json
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import tensorflow as tf
from bayesfilter.inference.gaussian_curvature_audit import (
    GaussianCurvatureAuditConfig,
    audit_gaussian_curvature,
)


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    started = time.monotonic()
    started_utc = datetime.now(timezone.utc).isoformat()
    config = GaussianCurvatureAuditConfig()
    cases = []
    for name, curvature, quartic, restricted, expected in (
        ("matched_gaussian", 1.0, 0.0, False, True),
        ("benign_quartic", 1.0, 0.03, False, True),
        ("stiff_gaussian", 9.0, 0.0, False, False),
        ("nearly_flat", 1e-10, 0.0, False, False),
        ("bounded_support", 1.0, 0.0, True, False),
        ("ill_conditioned_raw_gaussian", 1.0, 0.0, False, True),
        ("142d_correlated_gaussian", 1.0, 0.0, False, True),
    ):
        factor = tf.constant([[100.0, 0.0], [20.0, 0.01]], tf.float64) if name.startswith("ill_conditioned") else tf.eye(2, dtype=tf.float64)
        if name.startswith("142d"):
            factor = tf.eye(142, dtype=tf.float64) + 0.2 * tf.linalg.diag(tf.ones([141], tf.float64), k=-1)
        inverse = tf.linalg.inv(factor)

        def target(points, inverse=inverse, curvature=curvature, quartic=quartic, restricted=restricted):
            latent = tf.matmul(points, inverse, transpose_b=True)
            values = -tf.reduce_sum(curvature * latent**2 / 2 + quartic * latent**4 / 4, axis=1)
            scores = tf.matmul(-curvature * latent - quartic * latent**3, inverse)
            eligible = tf.reduce_all(tf.abs(latent) < 0.5, axis=1) if restricted else tf.ones(tf.shape(points)[0], tf.bool)
            return values, scores, eligible

        result = audit_gaussian_curvature(target, tf.zeros([factor.shape[0]], tf.float64), factor,
                                          config=replace(config, seed=(20260912, 4102)))
        cases.append({"name": name, "passed": result["passed"] is expected,
                      "candidate_passed": result["passed"], "expected": expected,
                      "status": result["status"], "rows": result["physical_rows"],
                      "nonpass_probability_upper": result["nonpass_probability_upper"]})
    payload = {"passed": all(case["passed"] for case in cases), "cases": cases,
               "started_utc": started_utc, "elapsed_seconds": time.monotonic() - started,
               "execution": "CPU-hidden independent synthetic diagnostic; no posterior or GPU claim"}
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with arguments.output.open("x") as destination:
        json.dump(payload, destination, indent=2, allow_nan=False)
    print(f"integration passed={payload['passed']} elapsed={payload['elapsed_seconds']:.3f}s")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(run())
