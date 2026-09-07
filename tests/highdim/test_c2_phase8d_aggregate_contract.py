"""Checks that Phase 8D aggregation treats the initial cloud separately."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "docs/benchmarks/aggregate_c2_phase8d_replication_20260905.py"
SPEC = importlib.util.spec_from_file_location("c2_phase8d_aggregate", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _write_result(path: Path, candidate: list[float], baseline: list[float]) -> None:
    path.mkdir()
    payload = {
        "proposal_active_start_time": 1,
        "records": [
            {"label": "exact_likelihood_laplace_k1", "all_checks_pass": True, "minimum_ess": min(candidate), "ess_by_time": candidate},
            {"label": "bootstrap_conditional", "all_checks_pass": True, "ess_by_time": baseline},
        ],
        "sources": {"fixture": {"sha256": "fixture"}},
        "status": "PASS_PHASE8D_VALIDITY_DIAGNOSTIC",
        "checks": {"all_branch_validity": True},
        "heuristic_dominance": {
            "exact_likelihood_laplace_k1": {"verdict": "PASSES_WEAK_HEURISTIC_SCREEN_ONLY"}
        },
    }
    (path / "result.json").write_text(json.dumps(payload), encoding="utf-8")


def test_aggregate_excludes_t0_from_active_contrasts(tmp_path: Path) -> None:
    run = tmp_path / "run"
    candidate = [1.0, 10.0, 11.0] + [10.0] * 12
    baseline = [2.0, 9.0, 12.0] + [9.0] * 12
    _write_result(run, candidate, baseline)
    result = MODULE.aggregate([run])
    contrast = result["rows"][0]["contrasts"]["bootstrap_conditional"]
    assert contrast["initial_difference"] == -1.0
    assert contrast["minimum_active_difference"] == -1.0
    assert contrast["active_comparison_count"] == 14
    assert contrast["positive_active_count"] == 13


def test_aggregate_rejects_a_run_without_active_time(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _write_result(run, [1.0], [2.0])
    with pytest.raises(ValueError, match="no active transition"):
        MODULE.aggregate([run])


def test_aggregate_preserves_a_candidate_failure_without_inventing_ess(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    payload = {
        "proposal_active_start_time": 1,
        "records": [
            {
                "label": "exact_likelihood_laplace_k1",
                "all_checks_pass": False,
                "failure_class": "candidate_or_comparator_failure",
                "error": "fixed Laplace result has invalid rows",
            },
            *[
                {"label": label, "all_checks_pass": True}
                for label in (
                    "ukf_apf_k1",
                    "bootstrap_conditional",
                    "transformed_student_nu8",
                    "stationary_independence",
                )
            ],
        ],
        "sources": {"fixture": {"sha256": "fixture"}},
        "status": "VETO_PHASE8D_CANDIDATE_VALIDITY",
        "checks": {"all_branch_validity": False},
        "heuristic_dominance": {
            "exact_likelihood_laplace_k1": {
                "verdict": "PASSES_WEAK_HEURISTIC_SCREEN_ONLY"
            }
        },
    }
    (run / "result.json").write_text(json.dumps(payload), encoding="utf-8")
    result = MODULE.aggregate([run])
    row = result["rows"][0]
    assert row["candidate_branch_validity"] is False
    assert row["candidate_minimum_ess"] is None
    assert row["contrasts"] == {}
    assert result["validity_pass_count"] == 0
