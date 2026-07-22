from __future__ import annotations

import json
from pathlib import Path

import pytest
import tensorflow as tf

from docs.benchmarks import (
    emit_contract_e_canonical_lgssm_phase8_fd_reclassification as fd,
)


def _records(*, valid: bool = True, eligible: bool = True, passed: bool = True):
    return [
        {
            "endpoint_valid": valid,
            "relative_denominator_eligible": eligible,
            "relative_error_pass": passed,
        }
    ]


def test_outcome_classification_is_complete_and_precedence_is_frozen() -> None:
    assert fd._classify_screen(_records()) == (
        "SEVEN_STEP_FD_HEURISTIC_SCREEN_PASSED"
    )
    assert fd._classify_screen(_records(passed=False)) == (
        "SEVEN_STEP_FD_HEURISTIC_SCREEN_FAILED"
    )
    assert fd._classify_screen(_records(eligible=False)) == (
        "FD_HEURISTIC_INCONCLUSIVE_NEAR_ZERO"
    )
    assert fd._classify_screen(_records(valid=False, eligible=False, passed=False)) == (
        "FD_HEURISTIC_INCONCLUSIVE_INVALID_ENDPOINT"
    )


def test_frozen_fixture_baseline_and_threshold_identity() -> None:
    assert fd._sha256(fd.FIXTURE) == fd.EXPECTED_FIXTURE_SHA256
    assert fd._sha256(fd.BASELINE_CERTIFICATE) == fd.EXPECTED_BASELINE_SHA256
    assert fd.EXPECTED_OBJECTIVE_HEX == "-0x1.55564a66d9848p+2"
    assert len(fd.EXPECTED_SCORE_HEX) == 5
    assert len(fd.EXPECTED_PREPARED_HASHES) == 8
    assert fd.LADDER_MULTIPLIERS == (8.0, 4.0, 2.0, 1.0, 0.5, 0.25, 0.125)
    assert fd.FD_THRESHOLD == pytest.approx(0.05 * 5**0.5)


def test_exclusive_writer_preserves_first_attempt(tmp_path: Path) -> None:
    output = tmp_path / "result.json"
    fd._write_json_exclusive(output, {"attempt": 1})
    assert json.loads(output.read_text(encoding="utf-8")) == {"attempt": 1}
    with pytest.raises(FileExistsError):
        fd._write_json_exclusive(output, {"attempt": 2})


def test_harness_scope_is_same_program_fd_only() -> None:
    source = Path(fd.__file__).read_text(encoding="utf-8")
    assert "make_canonical_value_and_score_tf" in source
    assert "jit_compile=True" in source
    assert "tf_kalman" not in source
    assert "historical_raw" not in source
    assert "raw_barycentric" not in source
    assert "0.05 * math.sqrt(5.0)" in source


def test_one_hot_parameter_direction_uses_float64_dtype() -> None:
    direction = tf.one_hot(2, 5, dtype=tf.float64)
    assert direction.dtype == tf.float64
    tf.debugging.assert_equal(
        direction, tf.constant([0.0, 0.0, 1.0, 0.0, 0.0], tf.float64)
    )


def test_nearest_dyadic_step_mode_is_frozen_and_exact_for_all_pairs() -> None:
    assert fd.NEAREST_DYADIC_EXPONENT == -17
    assert fd.NEAREST_DYADIC_BASE.hex() == "0x1.0000000000000p-17"
    theta = tf.constant([0.5, 0.25, -0.25, 0.5, 0.75], tf.float64)
    checks = fd._representability_preflight(theta, "nearest_dyadic_cuberoot")
    assert len(checks) == 35
    assert all(check["plus_equals_nominal"] for check in checks)
    assert all(check["minus_equals_nominal"] for check in checks)
    assert all(check["actual_steps_equal"] for check in checks)
    assert [value.hex() for value in fd._nominal_steps("nearest_dyadic_cuberoot", 0.5)] == [
        "0x1.0000000000000p-14",
        "0x1.0000000000000p-15",
        "0x1.0000000000000p-16",
        "0x1.0000000000000p-17",
        "0x1.0000000000000p-18",
        "0x1.0000000000000p-19",
        "0x1.0000000000000p-20",
    ]
