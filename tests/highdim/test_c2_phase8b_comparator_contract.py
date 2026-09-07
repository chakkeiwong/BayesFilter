"""Regression checks for the Phase 8B active-time comparator contract."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DRIVER_PATH = ROOT / "docs/benchmarks/run_c2_exact_likelihood_laplace_phase8b_20260904.py"
SPEC = importlib.util.spec_from_file_location("c2_phase8b_driver", DRIVER_PATH)
assert SPEC is not None and SPEC.loader is not None
DRIVER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DRIVER)


def _records(candidate: list[float], heuristic: list[float]):
    return [
        {"label": "exact_likelihood_laplace_k1", "branch_index": 0, "ess_by_time": candidate},
        {"label": "bootstrap_conditional", "branch_index": 0, "ess_by_time": heuristic},
    ]


def test_initial_cloud_loss_is_explanatory_only() -> None:
    result = DRIVER._heuristic_table(_records([1.0, 10.0], [2.0, 9.0]))[
        "exact_likelihood_laplace_k1"
    ]
    assert result["verdict"] == "PASSES_WEAK_HEURISTIC_SCREEN_ONLY"
    initial = result["comparisons"][0]
    assert initial["time_index"] == 0
    assert initial["candidate_loses"] is True
    assert initial["screen_candidate_loses"] is False
    assert initial["decision_role"] == "explanatory_only"


def test_active_transition_loss_remains_a_promotion_veto() -> None:
    result = DRIVER._heuristic_table(_records([10.0, 8.0], [9.0, 9.0]))[
        "exact_likelihood_laplace_k1"
    ]
    assert result["verdict"] == "PROMOTION_VETO_HEURISTIC_LOSS"
    active = result["comparisons"][1]
    assert active["proposal_active"] is True
    assert active["screen_candidate_loses"] is True


def test_candidate_validity_failure_is_not_implicitly_a_continuation_veto() -> None:
    records = [
        {
            "label": "exact_likelihood_laplace_k1",
            "branch_index": 0,
            "all_checks_pass": False,
            "failure_class": "candidate_or_comparator_failure",
            "error": "ValueError: fixed Laplace result has 2 invalid row(s)",
        },
        *[
            {"label": label, "branch_index": 0, "all_checks_pass": True}
            for label in (
                "ukf_apf_k1",
                "bootstrap_conditional",
                "transformed_student_nu8",
                "stationary_independence",
            )
        ],
    ]
    summary = DRIVER._record_validity_summary(
        records, branch_count=1, family_count=5
    )
    assert summary["candidate_failure"] is True
    assert summary["comparator_branch_validity"] is True
    assert summary["continuation_veto"] is False


def test_nonfinite_accepted_program_is_a_continuation_veto() -> None:
    records = [
        {
            "label": "exact_likelihood_laplace_k1",
            "branch_index": 0,
            "all_checks_pass": False,
            "checks": {"finite_exact_value_and_score": False},
            "error": "non-finite accepted program",
        },
        *[
            {"label": label, "branch_index": 0, "all_checks_pass": True}
            for label in (
                "ukf_apf_k1",
                "bootstrap_conditional",
                "transformed_student_nu8",
                "stationary_independence",
            )
        ],
    ]
    summary = DRIVER._record_validity_summary(
        records, branch_count=1, family_count=5
    )
    assert summary["continuation_veto"] is True


def test_result_markdown_renders_a_failed_candidate_row() -> None:
    payload = {
        "phase": "c2_exact_likelihood_laplace_phase8d_recovery_v1",
        "status": "VETO_PHASE8D_CANDIDATE_VALIDITY",
        "continuation": "CONTINUE_PHASE8D_REPAIR_AFTER_CANDIDATE_FAILURE",
        "failure_class": "candidate_failure",
        "continuation_veto": False,
        "time14_repair_nomination": False,
        "checks": {
            "schedule_calibration_valid": True,
            "all_branch_validity": False,
            "candidate_branch_validity": False,
            "comparator_branch_validity": True,
        },
        "records": [
            {
                "label": "exact_likelihood_laplace_k1",
                "all_checks_pass": False,
                "error": "ValueError: fixed Laplace result has invalid rows",
                "branch_wall_seconds": 1.25,
            },
            {
                "label": "ukf_apf_k1",
                "all_checks_pass": True,
                "minimum_ess": 10.0,
                "ess_by_time": [10.0] * 20,
                "branch_wall_seconds": 2.5,
            },
        ],
        "calibration": {
            "selected": {"config_id": "quarter_long"},
            "records": [
                {
                    "config_id": "quarter_long",
                    "selection_valid": True,
                    "worst_selection_residual": 1e-12,
                    "wall_seconds": 0.5,
                }
            ],
        },
    }
    rendered = DRIVER._result_markdown(payload)
    assert "fixed Laplace result has invalid rows" in rendered
    assert "| exact_likelihood_laplace_k1 | False | n/a | n/a | 1.25 |" in rendered


def test_repair_schedules_are_opt_in_and_do_not_change_baseline_ladder() -> None:
    baseline_ids = {str(row["config_id"]) for row in DRIVER.SCHEDULE_LADDER}
    repair_ids = {str(row["config_id"]) for row in DRIVER.REPAIR_SCHEDULE_LADDER}
    assert "quarter_long_12" not in baseline_ids
    assert repair_ids == {"quarter_long_12", "quarter_long_16", "fine_tempering_12"}
    assert baseline_ids.isdisjoint(repair_ids)


def test_phase8e_has_its_own_status_and_terminal_review_tokens() -> None:
    assert (
        DRIVER._phase_token(
            "c2_exact_likelihood_laplace_phase8e_statistical_replication_v1"
        )
        == "PHASE8E"
    )
    assert (
        DRIVER._continuation_token("PHASE8E", True)
        == "CONTINUE_PHASE8E_TERMINAL_REVIEW"
    )
    assert (
        DRIVER._continuation_token("PHASE8E", False)
        == "CONTINUATION_VETO_REPAIR_PHASE8E"
    )
