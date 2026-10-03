"""Standalone CPU-hidden reference integration, with no MacroFinance imports.

Known Gaussian modes verify recovery, not just optimizer return flags. A smooth
mixture tests local stationarity without a mode-coverage claim. Invalid-domain
and cap cases test rejection. This is an explicit non-XLA diagnostic exception;
Host-XLA parity is covered separately in the unit suite, not inferred here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
    raise RuntimeError("integration requires CUDA_VISIBLE_DEVICES=-1 before import")
if os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH") != "true":
    raise RuntimeError(
        "integration requires TF_FORCE_GPU_ALLOW_GROWTH=true before import"
    )

import tensorflow as tf
import tensorflow_probability as tfp

from bayesfilter.inference.batched_local_center import (
    BatchedLocalCenterConfig,
    locate_batched_local_center,
)


def _gaussian(mode, precision):
    mode = tf.constant(mode, tf.float64)
    precision = tf.constant(precision, tf.float64)

    def target(positions):
        displacement = positions - mode
        score = -tf.linalg.matmul(displacement, precision, transpose_b=True)
        return (
            0.5 * tf.reduce_sum(displacement * score, axis=1),
            score,
            tf.ones([4], tf.bool),
        )

    return target


def _support(positions):
    values, scores, _ = _gaussian([0.3, -0.2], [[2.0, 0.0], [0.0, 2.0]])(positions)
    eligible = tf.reduce_all(tf.abs(positions) <= 1.5, axis=1)
    return tf.where(eligible, values, tf.constant(1e100, tf.float64)), scores, eligible


def _nonfinite(positions):
    values, scores, _ = _gaussian([0.4, -0.1], [[2.0, 0.0], [0.0, 2.0]])(positions)
    valid = positions[:, 0] > -0.5
    return (
        tf.where(valid, values, tf.constant(float("nan"), tf.float64)),
        scores,
        tf.ones([4], tf.bool),
    )


def _flat(positions):
    return tf.zeros([4], tf.float64), tf.zeros_like(positions), tf.ones([4], tf.bool)


def _invalid(positions):
    values, scores, eligible = _flat(positions)
    return values, scores, ~eligible


def _mixture(positions):
    """Two equal-covariance Gaussian components and exact responsibility scores."""
    modes = tf.constant([[-1.0, 0.2], [1.0, -0.3]], tf.float64)
    displacement = positions[:, None, :] - modes[None, :, :]
    component_values = -2.0 * tf.reduce_sum(tf.square(displacement), axis=2)
    component_values += tf.math.log(tf.constant([0.35, 0.65], tf.float64))
    weights = tf.nn.softmax(component_values, axis=1)
    return (
        tf.reduce_logsumexp(component_values, axis=1),
        tf.reduce_sum(weights[:, :, None] * (-4.0 * displacement), axis=1),
        tf.ones([4], tf.bool),
    )


def _run_case(case):
    name, target, starts, scale, mode, tolerance, expected_status, cfg = case
    started = time.monotonic()
    calls = tf.Variable(0, dtype=tf.int32)

    def counted(positions):
        if positions.shape != (4, 2):
            raise ValueError(f"changed callback shape: {positions.shape}")
        calls.assign_add(1)
        return target(positions)

    result = locate_batched_local_center(counted, starts, scale, config=cfg)
    batch_count = int(result.target_callback_batches)
    expected_replays = int(result.rounds_completed) + int(
        tf.reduce_any(result.valid_rows)
    )
    gates = {
        "exact_callback_count": int(calls) == batch_count,
        "physical_rows": int(result.physical_target_rows) == 4 * batch_count,
        "call_decomposition": batch_count
        == 1 + int(result.optimizer_target_batches) + int(result.replay_batches),
        "replay_count": int(result.replay_batches) == expected_replays,
        "physical_cap": batch_count <= cfg.maximum_physical_rows_multiplier,
        "one_trace": result.trace_count == 1,
    }
    metrics = {}
    if expected_status is not None:
        gates["expected_rejection"] = (
            not bool(result.accepted) and result.status == expected_status
        )
    else:
        gates["accepted"] = bool(result.accepted)
        gates["replay_consistent"] = bool(result.replay_consistent)
        gates["finite_center_and_score"] = bool(
            tf.reduce_all(tf.math.is_finite(result.center))
        ) and (bool(tf.reduce_all(tf.math.is_finite(result.center_score))))
        if mode is not None:
            error = float(
                tf.reduce_max(tf.abs(result.center - tf.constant(mode, tf.float64)))
            )
            metrics["mode_max_absolute_error"] = error
            metrics["frozen_mode_tolerance"] = tolerance
            gates["mode_recovery"] = error <= tolerance
        elif name == "flat_exact_tie":
            gates["earliest_exact_tie"] = bool(
                tf.reduce_all(result.center == starts[0])
            ) and (int(result.selected_evaluation_index) == 0)
            gates["no_movement_stop"] = int(result.rounds_completed) == 1
        else:
            initial_values, _, initial_valid = target(starts)
            best_initial = tf.reduce_max(
                tf.where(initial_valid, initial_values, -float("inf"))
            )
            metrics["independent_reference_batches"] = 1
            metrics["independent_reference_physical_rows"] = 4
            metrics["raw_score_max_absolute"] = float(
                tf.reduce_max(tf.abs(result.center_score))
            )
            gates["local_stationarity"] = metrics["raw_score_max_absolute"] <= 1e-5
            gates["nondecreasing_best_value"] = bool(
                result.center_value >= best_initial
            )
    return {
        "case": name,
        "passed": all(gates.values()),
        "gates": gates,
        "metrics": metrics,
        "config": cfg.payload(),
        "result": result.payload(),
        "target_batch_shape": [4, 2],
        "initial_positions": starts.numpy().tolist(),
        "scale": scale.numpy().tolist(),
        "elapsed_seconds": time.monotonic() - started,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        started = time.monotonic()
        cfg = BatchedLocalCenterConfig(jit_compile=False)
        starts = tf.constant(
            [[-1.5, 1.2], [1.3, 1.0], [-1.0, -1.2], [0.4, -0.8]], tf.float64
        )
        scale = tf.constant([0.5, 2.0], tf.float64)
        ill_scale = tf.constant([0.01, 10.0], tf.float64)
        ill_starts = tf.constant([0.1, -0.2], tf.float64) + starts * ill_scale
        cases = (
            (
                "shifted_correlated_gaussian",
                _gaussian([0.2, -0.15], [[2.0, 0.4], [0.4, 1.3]]),
                starts,
                scale,
                [0.2, -0.15],
                1e-5,
                None,
                cfg,
            ),
            (
                "ill_conditioned_gaussian",
                _gaussian([0.1, -0.2], [[1e4, 3.0], [3.0, 0.01]]),
                ill_starts,
                ill_scale,
                [0.1, -0.2],
                1e-4,
                None,
                cfg,
            ),
            (
                "bounded_support_invalid_rows",
                _support,
                starts * 1.2,
                scale,
                [0.3, -0.2],
                1e-5,
                None,
                cfg,
            ),
            ("flat_exact_tie", _flat, starts, scale, None, None, None, cfg),
            (
                "smooth_mixture_local_stationarity",
                _mixture,
                starts,
                scale,
                None,
                None,
                None,
                cfg,
            ),
            ("nonfinite_rows", _nonfinite, starts, scale, [0.4, -0.1], 1e-5, None, cfg),
            (
                "all_invalid",
                _invalid,
                starts,
                scale,
                None,
                None,
                "initial_target_invalid",
                cfg,
            ),
            (
                "forced_callback_cap",
                _gaussian([0.2, -0.15], [[2.0, 0.4], [0.4, 1.3]]),
                starts,
                scale,
                None,
                None,
                "callback_cap_exhausted",
                BatchedLocalCenterConfig(
                    jit_compile=False, max_optimizer_callback_batches_per_round=1
                ),
            ),
        )
        root = Path(__file__).resolve().parents[1]
        sources = (
            root / "bayesfilter/inference/batched_local_center.py",
            root / "bayesfilter/inference/__init__.py",
            root / "tests/test_batched_local_center.py",
            Path(__file__).resolve(),
        )
        report = {
            "schema": "bayesfilter.batched_local_center.integration.v1",
            "scope": "CPU-hidden non-XLA standalone diagnostic; no MacroFinance target",
            "runtime_classification": "accepted TF/TFP direction; reference execution exception",
            "gpu_intentionally_hidden": True,
            "jit_compile": False,
            "tensorflow_version": tf.__version__,
            "tfp_version": tfp.__version__,
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "source_sha256": {
                str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sources
            },
            "cases": [],
        }
        for case in cases:
            case_start = time.monotonic()
            try:
                result = _run_case(case)
            except Exception as error:
                logging.getLogger(__name__).exception("Standalone localizer case failed")
                result = {
                    "case": case[0],
                    "passed": False,
                    "error": repr(error),
                    "elapsed_seconds": time.monotonic() - case_start,
                }
            report["cases"].append(result)
            print(
                json.dumps(
                    {key: result[key] for key in ("case", "passed", "elapsed_seconds")}
                ),
                flush=True,
            )
        forbidden = (
            "filters.",
            "inference.hmc",
            "inference.mass_matrix",
            "inference.posterior_adapter",
        )
        report["forbidden_runtime_imports"] = sorted(
            name for name in sys.modules if name.startswith(forbidden)
        )
        report["passed"] = (
            all(case["passed"] for case in report["cases"])
            and not report["forbidden_runtime_imports"]
        )
        report["ended_utc"] = datetime.now(timezone.utc).isoformat()
        report["elapsed_seconds"] = time.monotonic() - started
        report["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        json.dump(report, output, indent=2, sort_keys=True, allow_nan=False)
        output.write("\n")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
