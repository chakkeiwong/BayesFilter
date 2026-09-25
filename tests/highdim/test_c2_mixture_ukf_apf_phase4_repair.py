"""Regression checks for the Phase 4 scale-aware APF diagnostic repair."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import tensorflow as tf


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "docs/benchmarks/run_c2_mixture_ukf_apf_20260902.py"
SPEC = importlib.util.spec_from_file_location("c2_mixture_ukf_apf_runner", RUNNER)
assert SPEC is not None and SPEC.loader is not None
RUNNER_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER_MODULE)


def test_large_log_terms_use_scale_aware_backward_error() -> None:
    residual = tf.constant([9.313225746154785e-10], tf.float64)
    absolute, scale, relative = RUNNER_MODULE._scaled_backward_error(
        residual,
        (
            tf.constant([-4.0e6], tf.float64),
            tf.constant([-3.0e6], tf.float64),
            tf.constant([7.0e6], tf.float64),
            tf.constant([-1.0e6], tf.float64),
        ),
    )
    assert float(absolute.numpy()[0]) > RUNNER_MODULE.ABS_TOL
    assert float(scale.numpy()[0]) == 15.0e6
    assert float(relative.numpy()[0]) <= RUNNER_MODULE.APF_IDENTITY_BACKWARD_BOUND


def test_small_scale_same_residual_is_rejected() -> None:
    residual = tf.constant([9.313225746154785e-10], tf.float64)
    _, _, relative = RUNNER_MODULE._scaled_backward_error(
        residual,
        (
            tf.constant([1.0], tf.float64),
            tf.constant([2.0], tf.float64),
            tf.constant([3.0], tf.float64),
            tf.constant([4.0], tf.float64),
        ),
    )
    assert float(relative.numpy()[0]) > RUNNER_MODULE.APF_IDENTITY_BACKWARD_BOUND


def test_candidate_summary_distinguishes_evaluated_and_valid() -> None:
    records = [
        {
            "label": "arm",
            "minimum_ess": 10.0,
            "ess_by_time": [10.0, 11.0],
            "branch_wall_seconds": 1.0,
            "all_checks_pass": False,
        },
        {
            "label": "arm",
            "minimum_ess": 12.0,
            "ess_by_time": [12.0, 13.0],
            "branch_wall_seconds": 1.0,
            "all_checks_pass": True,
        },
        {"label": "arm", "error": "retained"},
    ]
    summary = RUNNER_MODULE._phase3_candidate_summary(records)["arm"]
    assert summary["record_count"] == 3
    assert summary["evaluated_count"] == 2
    assert summary["valid_count"] == 1
