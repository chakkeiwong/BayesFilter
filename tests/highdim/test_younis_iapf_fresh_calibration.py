"""Backend-free checks of campaign coverage and fail-closed execution accounting."""
import importlib.util
import json
from pathlib import Path

import pytest

from bayesfilter.score_study.contracts import DiagnosticFailure


spec = importlib.util.spec_from_file_location(
    "iapf_fresh_driver", Path(__file__).resolve().parents[2] / "scripts/run_younis_iapf_fresh_calibration.py")
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


def test_complete_matrix_is_disjoint_and_within_budget(tmp_path):
    driver.preflight(tmp_path / "preflight")
    record = json.loads((tmp_path / "preflight/preflight.json").read_text())
    assert record["rows"] == 80
    assert record["upper_charges"] + record["reserved_charges"] == record["limit"] == 280
    assert len(record["datasets"]) == len(set(record["datasets"])) == 14
    assert record["numerical_work"] is False


def test_failed_fit_is_preserved_and_charged_without_stopping_independent_work():
    budget = driver.Budget(lambda: None)
    row = driver.studies("weak")[0]["rows"][0]
    failure = DiagnosticFailure("underflow", {"objective_underflow": True})
    def fail(*_):
        raise failure
    with pytest.raises(DiagnosticFailure):
        budget.wrap(fail, "selected-source")(row, {})
    assert budget.used == 4
    assert budget.failures[0]["diagnostics"] == failure.diagnostics
    heuristic = driver.studies("weak")[2]["rows"][0]
    assert budget.wrap(lambda *_: {}, "heuristic")(heuristic, {}) == {}
    assert budget.used == 5


def test_success_uses_observed_fit_count_and_budget_prevents_next_call():
    budget = driver.Budget(lambda: None, limit=5)
    row = driver.studies("weak")[0]["rows"][0]
    def success(*_):
        return {"diagnostics": {"work_accounting": {"offline_fit_calls": 1}}}
    budget.wrap(success, "source")(row, {})
    assert budget.used == 2
    with pytest.raises(driver.CampaignStop, match="budget exhausted"):
        budget.wrap(lambda *_: pytest.fail("over-budget endpoint ran"), "source")(row, {})
    assert budget.used == 2


def test_infrastructure_failure_stops_campaign():
    budget = driver.Budget(lambda: None)
    row = driver.studies("weak")[0]["rows"][0]
    def fail(*_):
        raise ValueError("reference invalid")
    with pytest.raises(driver.CampaignStop, match="reference invalid"):
        budget.wrap(fail, "source")(row, {})
    assert budget.used == 4
    assert not budget.failures


def test_incomplete_source_cannot_drop_failed_candidate_to_issue_selection(tmp_path):
    state = {"execution_status":"incomplete", "rows":{
        "bad": {"execution_status":"failed"}, "good": {"execution_status":"complete"}}}
    result = driver.selection_or_blocked(state, tmp_path, tmp_path / "selection.json",
        lambda *_: pytest.fail("selection issued from an incomplete source"))
    assert result["status"] == "blocked" and result["rows"] == {"bad":"failed"}


def test_complete_baseline_study_can_issue_its_own_selection(tmp_path):
    result = driver.selection_or_blocked({"execution_status":"complete"}, tmp_path,
        tmp_path / "selection.json", lambda *_: {"selected_controls":{"iapf":{"k":1}}})
    assert result["status"] == "issued"


def test_missing_claims_do_not_become_heuristic_passes():
    tables = driver.conditional_tables([], {})
    assert len(tables) == 2
    for table in tables:
        assert table["statistically_supported_ranking"] is False
        for entry in table["datasets"]:
            for arm in ("selected", "baseline"):
                assert entry[arm]["mse"] is None
                assert entry[arm]["heuristic_dominance_verdict"] == "not_evaluable"
