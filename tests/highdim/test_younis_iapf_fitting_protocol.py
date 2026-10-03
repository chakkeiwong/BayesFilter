"""Diagnostic selection isolation and the informative stopping alternative."""
import json

import pytest

from bayesfilter.score_study.iapf_adapter import iteration_decision
from docs.benchmarks import diagnose_younis_iapf_fitting_protocol as driver


def test_informative_screen_rejects_unstable_history_that_old_rule_accepts():
    logs = [0., -1., 0., -1., 0.]
    assert iteration_decision(logs, [256]*5, k=1, tau=100., max_particles=2048)["action"] == "final"
    decision = iteration_decision(logs, [256]*5, k=3, tau=.05, max_particles=2048)
    assert decision["action"] == "fit"
    assert decision["next_particles"] == 512
    assert iteration_decision([0., .001, 0., .001, 0.], [256]*5,
                              k=3, tau=.05, max_particles=2048)["action"] == "final"


def test_selection_cannot_consume_validation_or_promote_invalid_fit():
    records = {str(key): {"arms": {
        "large_early": {"fit": {"status": "valid"}, "summary": {"innovation": {"score_squared_error": 2.}}},
        "large_stable": {"fit": {"status": "rejected"}, "summary": {"innovation": {"score_squared_error": 0.}}},
    }} for key in driver.DATASETS["calibration"]}
    assert driver.choose_protocol(records)["selected"] == "large_early"
    records["2200"] = records.pop("2000")
    with pytest.raises(ValueError, match="calibration observations"):
        driver.choose_protocol(records)


def test_failed_attempt_charges_remain_and_completed_stage_cannot_repeat(tmp_path, monkeypatch):
    monkeypatch.setattr(driver, "ROOT", tmp_path)
    path = tmp_path/"failed01"
    path.mkdir()
    record = dict(stage="calibration", status="stopped", wall_seconds=20.,
        prior_counts=dict(adaptive_fits=0, fixed_cloud_fits=0, filter_calls=0),
        budget=dict(counts=dict(adaptive_fits=4, fixed_cloud_fits=0, filter_calls=200)))
    (path/"manifest.json").write_text(json.dumps(record))
    budget, seconds, launch = driver.campaign_budget(tmp_path/"retry02", "calibration")
    assert (budget.counts["adaptive_fits"], budget.counts["filter_calls"], seconds, launch) == (4, 200, 20., 2)
    record["status"] = "complete"
    (path/"manifest.json").write_text(json.dumps(record))
    with pytest.raises(RuntimeError, match="cannot be rerun"):
        driver.campaign_budget(tmp_path/"retry02", "calibration")
