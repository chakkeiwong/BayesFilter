from dataclasses import replace

import pytest

from bayesfilter.testing.inference_validation.execution import run_suite
from bayesfilter.testing.inference_validation.reporting import report
from bayesfilter.testing.inference_validation.assessment import assess
from bayesfilter.testing.inference_validation.storage import read_json


def test_cli_workers_resume_without_duplicate_trials_and_detect_corruption(design, tmp_path):
    d=design("invariance", control="identity", replications=8)
    second=replace(d,design_id="second-job")
    suite=dict(schema="bayesfilter.inference_validation_suite.v1",suite_id="integration",
        profile="numerical",profiles={"numerical":["invariance"]},designs=[d.payload(),second.payload()],
        required_coverage=[d.coverage_key(),{"engine":"sbc","route":"ordinary","target":"normal_conjugate"}])
    root=tmp_path/"run"
    first=run_suite(suite,root,max_jobs=1)
    assert first["jobs"][d.design_id]["status"]=="complete"
    assert second.design_id not in first["jobs"]
    resumed=run_suite(suite,root,resume=True)
    assert len(resumed["jobs"][d.design_id]["attempts"])==1
    assert len(resumed["jobs"][second.design_id]["attempts"])==1
    summary=report(root)
    assert not summary["all_required_executed"]
    original=read_json(resumed["jobs"][d.design_id]["result"])
    fresh=assess(root,tmp_path/"reassessment.json")
    assert fresh["rows"][0]["rank_tests"]==original["assessment"]["rank_tests"]
    assert not fresh["sampler_executed"]
    with pytest.raises(ValueError,match="identical"):
        run_suite({**suite,"suite_id":"changed"},root,resume=True)
    from pathlib import Path
    Path(resumed["jobs"][d.design_id]["result"]).write_text("{}")
    assert report(root)["rows"][0]["execution_status"]=="invalid_artifact"


def test_interrupted_coordinator_reservation_is_not_forgotten(design, tmp_path):
    from bayesfilter.testing.inference_validation import execution
    from bayesfilter.testing.inference_validation.storage import write_json
    d=design()
    suite=dict(schema="bayesfilter.inference_validation_suite.v1",suite_id="interrupted",
        profile="fast",profiles={"fast":["mechanics"]},designs=[d.payload()])
    root=tmp_path/"interrupted"
    plan=execution.plan_suite(suite)
    write_json(root/"run_index.json",{"suite_identity":plan["suite_identity"],"source":execution.source_state(),
        "plan":plan,"jobs":{d.design_id:{"status":"running","reserved_seconds":d.budget_seconds,
                                       "attempts":[],"started_at":0}}})
    index=run_suite(suite,root,resume=True)
    assert index["jobs"][d.design_id]["status"]=="unfunded"
    assert index["jobs"][d.design_id]["attempts"][0]["elapsed_seconds"]==d.budget_seconds


def test_old_saved_empty_posteriors_do_not_count_as_complete_assessment():
    from bayesfilter.testing.inference_validation.reporting import assessment_complete
    assert not assessment_complete({"finding":"pipeline_assessed", "planned":1,
        "replications":[{"members":[{"assessment":{"finding":"unavailable"}}]}]})
