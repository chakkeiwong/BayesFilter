"""CPU-only schema tests for diagnostic component profiling."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("phase9b_component_tests", ROOT / "docs/benchmarks/profile_ssl_lstm_q20_phase9b_components_2026_09_09.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def fixture():
    return {"status": "completed", "banks": ["initial", "terminal"],
            "memory_policy": {"all_physical_devices_memory_growth": True},
            "components": {name: {"first_seconds": 1.0, "tracing_count": 1, "hlo_sha256": "cpu_test_hash",
                          "repeat_outputs_equal": True,
                          "warmed_seconds": {"initial": [0.5] * 4, "terminal": [0.6] * 4}}
                           for name in runner.COMPONENTS}}


def test_profile_receipt_requires_complete_evidence():
    runner.validate_receipt(fixture())


@pytest.mark.parametrize("case", ("memory", "missing_component", "missing_bank", "retrace", "xla", "nonfinite", "negative", "repeat", "different_output"))
def test_profile_receipt_rejects_incomplete_evidence(case):
    value = fixture()
    component = value["components"]["target_status"]
    if case == "memory":
        value["memory_policy"]["all_physical_devices_memory_growth"] = False
    elif case == "missing_component":
        del value["components"]["transport"]
    elif case == "missing_bank":
        del component["warmed_seconds"]["terminal"]
    elif case == "retrace":
        component["tracing_count"] = 2
    elif case == "xla":
        component["hlo_sha256"] = None
    elif case == "nonfinite":
        component["warmed_seconds"]["initial"][0] = float("nan")
    elif case == "negative":
        component["first_seconds"] = -1
    elif case == "repeat":
        component["warmed_seconds"]["initial"].pop()
    else:
        component["repeat_outputs_equal"] = False
    with pytest.raises(runner.CheckpointError):
        runner.validate_receipt(value)


@pytest.mark.parametrize("case", ("scalar_gap", "multidimensional_gap", "negative_infinite_gap", "nan_gap", "infinite_eigenvalue", "nonfinite_score"))
def test_only_scalar_positive_infinite_gap_sentinel_is_allowed(case):
    import tensorflow as tf

    result = {"status_code": tf.zeros([4], tf.int32), "valid_pre_regularized_score": tf.ones([4], tf.bool),
              "min_innovation_eigen_gap": tf.fill([4], tf.constant(float("inf"), tf.float64))}
    dimension = 2 if case == "multidimensional_gap" else 1
    name = "physical_value_score" if case == "nonfinite_score" else "target_status"
    if case == "negative_infinite_gap":
        result["min_innovation_eigen_gap"] = -result["min_innovation_eigen_gap"]
    elif case == "nan_gap":
        result["min_innovation_eigen_gap"] = tf.fill([4], tf.constant(float("nan"), tf.float64))
    elif case == "infinite_eigenvalue":
        result["min_innovation_eigenvalue"] = result["min_innovation_eigen_gap"]
    if case == "scalar_gap":
        runner.validate_values(tf, name, result, dimension)
    else:
        with pytest.raises(runner.CheckpointError, match="nonfinite"):
            runner.validate_values(tf, name, result, dimension)
