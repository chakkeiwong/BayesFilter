from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import pytest

from bayesfilter.ledh_fd_policy import (
    LEDH_FD_BASE_RELATIVE_TOLERANCE,
    LEDH_FD_FLOAT32_EPSILON,
    LEDH_FD_STEP_COEFFICIENT,
    coordinate_central_difference_step,
    coordinate_relative_error,
    evaluate_ledh_fd_policy,
    ledh_fd_step_policy_metadata,
    validate_declared_ledh_fd_policy,
)
from docs.benchmarks import reclassify_ledh_phase9_fd_policy as reclassifier


INPUT_MANIFEST = (
    reclassifier.ROOT
    / "docs/plans/ledh-score-wiring-repair-phase9-fd-reclassification-inputs-2026-07-11.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_fd_policy_preserves_phase9_denominator_floor() -> None:
    assert coordinate_relative_error(0.0, 0.0) == (0.0, 1.0e-12, 0.0)
    assert coordinate_relative_error(0.0, 1.0e-13) == (1.0e-13, 1.0e-12, 0.1)


def test_fd_policy_uses_observed_scale_above_denominator_floor() -> None:
    result = evaluate_ledh_fd_policy(
        [1.0e-6],
        [0.99e-6],
        ["tiny"],
    )

    assert result["parameters"][0]["relative_error_scale"] == 1.0e-6
    assert result["parameters"][0]["relative_error"] == pytest.approx(0.01)
    assert result["status"] == "pass"


def test_fd_policy_uses_five_percent_sqrt_p_maximum_direction() -> None:
    at_boundary = evaluate_ledh_fd_policy(
        [10.0, 10.0, 10.0, 10.0],
        [9.0, 9.1, 9.1, 9.1],
        ["a", "b", "c", "d"],
    )
    above_boundary = evaluate_ledh_fd_policy(
        [10.0, 10.0, 10.0, 10.0],
        [8.99, 10.0, 10.0, 10.0],
        ["a", "b", "c", "d"],
    )

    assert at_boundary["max_coordinate_relative_error"] == pytest.approx(0.1)
    assert at_boundary["max_coordinate_relative_error_threshold"] == pytest.approx(0.1)
    assert at_boundary["status"] == "pass"
    assert "relative_error_rss" not in at_boundary
    assert "relative_error_rms" not in at_boundary
    assert above_boundary["status"] == "fail"


def test_fd_policy_rejects_nonfinite_and_forged_declared_results() -> None:
    with pytest.raises(ValueError, match="nonfinite"):
        evaluate_ledh_fd_policy([math.inf], [1.0], ["a"])

    declared = evaluate_ledh_fd_policy([1.0], [0.99], ["a"])
    forged = copy.deepcopy(declared)
    forged["max_coordinate_relative_error"] = 0.0
    with pytest.raises(ValueError, match="does not match recomputed"):
        validate_declared_ledh_fd_policy(forged, [1.0], [0.99], ["a"])


def test_float32_central_fd_step_is_derived_and_coordinate_scaled() -> None:
    assert LEDH_FD_STEP_COEFFICIENT == LEDH_FD_FLOAT32_EPSILON ** (1.0 / 3.0)
    assert coordinate_central_difference_step(0.0) == LEDH_FD_STEP_COEFFICIENT
    assert coordinate_central_difference_step(-2.0) == 2.0 * LEDH_FD_STEP_COEFFICIENT
    assert coordinate_central_difference_step(114.0) == 114.0 * LEDH_FD_STEP_COEFFICIENT
    metadata = ledh_fd_step_policy_metadata()
    assert metadata["nominal_step_formula"] == (
        "cbrt(float32_epsilon) * max(1, abs(theta_j))"
    )
    assert metadata["effective_denominator"] == "plus_theta_j - minus_theta_j"

    with pytest.raises(ValueError, match="theta must be finite"):
        coordinate_central_difference_step(math.inf)


def test_actual_phase9_manifest_reclassifies_all_live_completed_rungs() -> None:
    result = reclassifier.reclassify_manifest(INPUT_MANIFEST)
    terminal = {
        entry["row"]: entry["corrected_policy"]
        for entry in result["entries"]
        if entry["historical_terminal_decision"]
    }
    corrected_stops = {
        entry["row"]: entry
        for entry in result["entries"]
        if entry["corrected_ladder_stop"]
    }

    assert result["summary"] == {
        "num_reclassified_rungs": 11,
        "num_historical_terminal_decisions": 5,
        "num_corrected_ladder_stops": 2,
        "all_rungs": {"pass": 9, "fail": 2},
        "historical_terminal_reclassification": {"pass": 3, "fail": 2},
        "corrected_ladder_stops": {"pass": 0, "fail": 2},
        "rows_with_stored_fd_failure": ["generalized-sv", "predator-prey"],
        "rows_without_stored_fd_failure": ["actual-sv", "fixed-sir", "ksc-sv"],
        "ranking_statistically_supported": False,
        "hmc_readiness_established": False,
    }
    assert set(terminal) == {
        "fixed-sir",
        "predator-prey",
        "actual-sv",
        "generalized-sv",
        "ksc-sv",
    }
    assert {row: policy["status"] for row, policy in terminal.items()} == {
        "fixed-sir": "pass",
        "predator-prey": "fail",
        "actual-sv": "pass",
        "generalized-sv": "fail",
        "ksc-sv": "pass",
    }
    assert corrected_stops["generalized-sv"]["id"] == "gate-c-generalized-sv-t4-n10000-seed81120"
    assert corrected_stops["predator-prey"]["id"] == "gate-b-predator-prey-t1-n2-seed81120"
    assert terminal["fixed-sir"]["max_coordinate_relative_error"] == pytest.approx(0.05667000855870846)
    assert terminal["predator-prey"]["max_coordinate_relative_error"] == pytest.approx(1.0)
    assert terminal["actual-sv"]["max_coordinate_relative_error"] == pytest.approx(0.06029246881250902)
    assert terminal["generalized-sv"]["max_coordinate_relative_error"] == pytest.approx(0.4427539621608173)
    assert terminal["ksc-sv"]["max_coordinate_relative_error"] == pytest.approx(0.036935149298161114)
    assert terminal["actual-sv"]["num_parameters"] == 2
    assert terminal["actual-sv"]["max_coordinate_relative_error_threshold"] == pytest.approx(
        0.05 * math.sqrt(2)
    )
    assert result["policy"]["base_relative_tolerance"] == LEDH_FD_BASE_RELATIVE_TOLERANCE


def _write_fake_legacy_pair(root: Path) -> tuple[dict, Path, Path]:
    score_path = root / "score.json"
    fd_path = root / "fd.json"
    common_manifest = {
        "row": "fixed-sir",
        "row_id": "fixture-row",
        "time_steps": 1,
        "num_particles": 4,
        "batch_seeds": [81120],
    }
    score = [1.0, 2.0]
    finite_difference = [1.0, 2.0]
    score_payload = {
        "schema_version": reclassifier.LEGACY_SHARD_SCHEMA_VERSION,
        "artifact_status": "completed",
        "terminal_artifact": True,
        "row_id": "fixture-row",
        "run_manifest": {**common_manifest, "stage": "score-only", "output": "score.json"},
        "prepared_input_fingerprint": {"fixture": "same"},
        "precision": {"dtype": "float32", "tf32_mode": "enabled"},
        "score_parameter_names": ["a", "b"],
        "score": score,
    }
    score_path.write_text(json.dumps(score_payload), encoding="utf-8")
    score_hash = _sha256(score_path)
    fd_payload = {
        "schema_version": reclassifier.LEGACY_SHARD_SCHEMA_VERSION,
        "artifact_status": "completed",
        "terminal_artifact": True,
        "row_id": "fixture-row",
        "run_manifest": {**common_manifest, "stage": "fd-only", "output": "fd.json"},
        "prepared_input_fingerprint": {"fixture": "same"},
        "precision": {"dtype": "float32", "tf32_mode": "enabled"},
        "score_parameter_names": ["a", "b"],
        "score": score,
        "score_reference_sha256": score_hash,
        "score_correctness": {
            "kind": "same_scalar_finite_difference",
            "status": "pass",
            "step": 1.0e-4,
            "atol": 5.0e-3,
            "rtol": 5.0e-3,
            "pass_rule": "max_abs_error <= atol OR max_relative_error <= rtol",
            "max_abs_error": 0.0,
            "max_relative_error": 0.0,
            "uses_value_only_scalar_route": True,
            "parameters": [
                {
                    "parameter": name,
                    "score": score[index],
                    "finite_difference": finite_difference[index],
                    "abs_error": 0.0,
                    "relative_error": 0.0,
                }
                for index, name in enumerate(("a", "b"))
            ],
        },
    }
    fd_path.write_text(json.dumps(fd_payload), encoding="utf-8")
    entry = {
        "id": "gate-b-fixture",
        "row": "fixed-sir",
        "historical_terminal_decision": False,
        "score_path": "score.json",
        "score_sha256": score_hash,
        "fd_path": "fd.json",
        "fd_sha256": _sha256(fd_path),
    }
    return entry, score_path, fd_path


def test_reclassifier_rejects_forged_manifest_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(reclassifier, "ROOT", tmp_path)
    entry, _score_path, _fd_path = _write_fake_legacy_pair(tmp_path)
    entry["score_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="source SHA-256 mismatch"):
        reclassifier._reclassify_entry(entry)  # noqa: SLF001


def test_reclassifier_rejects_broken_embedded_score_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(reclassifier, "ROOT", tmp_path)
    entry, _score_path, fd_path = _write_fake_legacy_pair(tmp_path)
    payload = json.loads(fd_path.read_text(encoding="utf-8"))
    payload["score_reference_sha256"] = "0" * 64
    fd_path.write_text(json.dumps(payload), encoding="utf-8")
    entry["fd_sha256"] = _sha256(fd_path)

    with pytest.raises(ValueError, match="FD-to-score SHA-256 binding mismatch"):
        reclassifier._reclassify_entry(entry)  # noqa: SLF001


def test_reclassifier_rejects_wrong_parameter_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(reclassifier, "ROOT", tmp_path)
    entry, _score_path, fd_path = _write_fake_legacy_pair(tmp_path)
    payload = json.loads(fd_path.read_text(encoding="utf-8"))
    payload["score_parameter_names"] = ["b", "a"]
    fd_path.write_text(json.dumps(payload), encoding="utf-8")
    entry["fd_sha256"] = _sha256(fd_path)

    with pytest.raises(ValueError, match="score/FD parameter order mismatch"):
        reclassifier._reclassify_entry(entry)  # noqa: SLF001


def test_reclassifier_module_does_not_import_tensorflow() -> None:
    source = Path(reclassifier.__file__).read_text(encoding="utf-8")
    assert "import tensorflow" not in source
    assert "bayesfilter.highdim" not in source
