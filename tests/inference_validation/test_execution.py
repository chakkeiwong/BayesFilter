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


@pytest.mark.parametrize("explicit_category_requirement", [False, True])
def test_planned_cells_do_not_borrow_completion_from_the_same_category(
        design, tmp_path, monkeypatch, explicit_category_requirement):
    from bayesfilter.testing.inference_validation import execution
    from bayesfilter.testing.inference_validation.storage import write_json, file_hash
    first = design(step_size=.3)
    second = replace(first, design_id="second-epsilon", step_size=.6)
    suite = dict(schema="bayesfilter.inference_validation_suite.v1", suite_id="coverage",
                 profile="fast", profiles={"fast": ["mechanics"]},
                 designs=[first.payload(), second.payload()])
    if explicit_category_requirement:
        suite["required_coverage"] = [first.coverage_key()]
    monkeypatch.setattr(execution, "source_state", lambda: {"identity": "fixture"})
    index = {"source": execution.source_state(), "plan": execution.plan_suite(suite), "jobs": {}}

    def complete(d):
        result = write_json(tmp_path/d.design_id/"result.json", {
            "design_identity": d.identity, "assessment": {"finding": "mechanics_passed"}})
        index["jobs"][d.design_id] = {"status": "complete", "result": str(result),
                                      "result_sha256": file_hash(result), "attempts": []}
        write_json(tmp_path/"run_index.json", index)
        return result

    complete(first)
    partial = report(tmp_path)
    assert partial["rows"][1]["execution_status"] == "not_run"
    assert partial["all_required_executed"] is explicit_category_requirement
    assert partial["all_required_assessed"] is explicit_category_requirement
    second_result = complete(second)
    assert report(tmp_path)["all_required_assessed"]
    second_result.write_text("{}")
    corrupt = report(tmp_path)
    assert corrupt["rows"][1]["execution_status"] == "invalid_artifact"
    assert corrupt["all_required_executed"] is explicit_category_requirement
    assert corrupt["all_required_assessed"] is explicit_category_requirement


@pytest.mark.parametrize("fail", [False, True])
def test_opt_in_host_profile_survives_success_and_failure(design, tmp_path, monkeypatch, fail):
    import pstats
    from bayesfilter.testing.inference_validation import execution
    from bayesfilter.testing.inference_validation.engines import mechanics
    from bayesfilter.testing.inference_validation.storage import write_json
    d = design(options={"profile_execution": True})
    path = write_json(tmp_path/"design.json", d.payload())
    monkeypatch.setattr(execution, "configure_worker", lambda _: {"device_scope": "cpu_reference"})
    monkeypatch.setattr(execution, "source_state", lambda: {"identity": "fixture"})
    def run(*args):
        if fail:
            raise RuntimeError("deliberate engine failure")
        return {"finding": "mechanics_passed"}
    monkeypatch.setattr(mechanics, "run", run)
    assert execution.worker(path, tmp_path, 10.) == int(fail)
    profile = pstats.Stats(str(tmp_path/"attempt-001-host.prof"))
    assert profile.total_calls > 0
    if fail:
        assert read_json(tmp_path/"attempt-001-failure.json")["reason"] == "deliberate engine failure"


def test_profile_write_failure_preserves_the_original_engine_failure(design, tmp_path, monkeypatch):
    import cProfile
    from bayesfilter.testing.inference_validation import execution
    from bayesfilter.testing.inference_validation.engines import mechanics
    from bayesfilter.testing.inference_validation.storage import write_json
    class UnwritableProfile:
        def enable(self, **kwargs): pass
        def disable(self): pass
        def dump_stats(self, path): raise OSError("profile volume full")
    monkeypatch.setattr(cProfile, "Profile", UnwritableProfile)
    monkeypatch.setattr(execution, "configure_worker", lambda _: {"device_scope": "cpu_reference"})
    monkeypatch.setattr(execution, "source_state", lambda: {"identity": "fixture"})
    def fail(*args): raise RuntimeError("original engine failure")
    monkeypatch.setattr(mechanics, "run", fail)
    path = write_json(tmp_path/"design.json", design(options={"profile_execution": True}).payload())
    with pytest.warns(RuntimeWarning, match="host profile unavailable"):
        assert execution.worker(path, tmp_path, 10.) == 1
    assert read_json(tmp_path/"attempt-001-failure.json")["reason"] == "original engine failure"
