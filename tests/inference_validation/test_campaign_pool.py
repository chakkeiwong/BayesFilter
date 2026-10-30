"""Common-pool recovery exercises the real dispatcher and frozen child boundary."""
from dataclasses import replace
from pathlib import Path
import time

import pytest

from bayesfilter.testing.inference_validation.campaign_pool import run_pool
from bayesfilter.testing.inference_validation import campaign_recovery as recovery
from bayesfilter.testing.inference_validation.designs import ValidationDesign, digest
from bayesfilter.testing.inference_validation.storage import read_json, write_json, file_hash
from scripts import run_hmc_ssm_pooled_campaign as campaign
from tests.inference_validation.test_campaign_recovery import saved_fit, stopped, frozen_source


def test_round_robin_recovery_uses_time_released_by_terminal_peer():
    now, calls = [0.], []
    def execute(job, allowance, deadline, attempt):
        calls.append((job["job_id"], attempt, allowance, deadline))
        now[0] += 2. if job["job_id"] == "B" else allowance
        return ({"status": "complete", "posterior_passed": False} if job["job_id"] == "B"
                or attempt == 2 else stopped())
    result = run_pool([{"job_id": x, "case": "K0"} for x in "ABC"], execute,
        deadline=23., quantum_seconds=5., max_attempts=2, clock=lambda: now[0])
    assert [r[0] for r in calls] == ["A", "B", "C", "A", "C"]
    assert result["all_workloads_complete"] and now[0] == 22.
    assert result["rows"][1]["attempts"][0]["posterior_passed"] is False
    assert all(t <= 23. for _, _, _, t in calls)


@pytest.mark.parametrize("receipt,status,count", [
    (stopped(), "incomplete_attempt_limit", 2),
    (stopped(progress={}), "incomplete_no_progress", 1),
    ({"status": "failed"}, "failed", 1),
    ({"status": "no_resumable_numerical_checkpoint"}, "no_resumable_numerical_checkpoint", 1),
])
def test_no_infinite_retry_or_numerical_failure_reset(receipt, status, count):
    result = run_pool([{"job_id": "fit", "case": "K6"}], lambda *a: receipt,
        deadline=100., quantum_seconds=5., max_attempts=2, clock=lambda: 0.)
    assert result["rows"][0]["status"] == status
    assert len(result["rows"][0]["attempts"]) == count
    assert not result["all_workloads_complete"]


def test_oversized_job_does_not_hide_smaller_peer():
    calls = []
    result = run_pool([{"job_id": "large", "case": "K7", "first_quantum_seconds": 20.},
                       {"job_id": "small", "case": "K4"}],
        lambda j, *args: calls.append(j["job_id"]) or {"status": "complete"},
        deadline=10., quantum_seconds=5., max_attempts=1, clock=lambda: 0.)
    assert calls == ["small"]
    assert result["rows"][0]["status"] == "unstarted_pool_deadline"


def test_duplicate_fit_and_unexpected_result_stop_pool():
    job = {"job_id": "fit", "case": "K0"}
    options = dict(deadline=10., quantum_seconds=5., max_attempts=1, clock=lambda: 0.)
    with pytest.raises(ValueError, match="duplicate"):
        run_pool([job, job], lambda *a: pytest.fail("launched"), **options)
    with pytest.raises(ValueError, match="unexpected"):
        run_pool([job], lambda *a: {"status": "supervision_failed"}, **options)


def test_exhausted_original_cap_dispatches_child_and_preserves_negative_result(tmp_path, design, monkeypatch):
    source, cell, d, identity = saved_fit(tmp_path, design)
    calls = []
    def supervise(command, log, seconds, device, **kwargs):
        calls.append(command)
        assert kwargs["cwd"] == source
        assert 0 < seconds <= 4.
        prefix = str(log).removesuffix(".log")
        write_json(prefix + "-manifest.json", {"design_identity": d.identity,
            "source": {"identity": identity["source_identity"]}, "runtime": {}})
        write_json(cell / "replication-0000/independent_assessment.json", {"posterior_passed": False})
        return {"status": "complete", "exit_code": 0, "elapsed_seconds": 1.}
    monkeypatch.setattr(recovery, "supervise_fit", supervise)
    job = campaign.make_job("fit", "K0", cell, first_quantum=4.)
    result = run_pool([job], campaign.executor(source, tmp_path / "attempts"),
        deadline=time.monotonic()+20., quantum_seconds=4., max_attempts=2)
    assert result["all_workloads_complete"] and len(calls) == 1
    assert "_pipeline_fit" in calls[0]
    receipt = result["rows"][0]["attempts"][0]
    assert receipt["campaign_continuation"] and receipt["cumulative_fit_seconds"] == 11.
    assert d.options["fit_process_timeout_seconds"] == 10.
    again = campaign.executor(source, tmp_path / "again")(job, 4., time.monotonic()+10., 1)
    assert again["status"] == "reused_final_assessment" and len(calls) == 1


def test_fresh_dispatch_uses_frozen_source_and_native_identity(tmp_path, design, monkeypatch):
    source = tmp_path / "source"
    frozen = frozen_source(source)
    d = design("accuracy", "gaussian", "prepared", replications=1)
    cell = tmp_path / "fit"
    def supervise(command, log, seconds, device, **kwargs):
        assert kwargs["cwd"] == source
        assert "--reuse-leapfrog-graphs" in command
        prefix = str(log).removesuffix(".log")
        write_json(prefix + "-manifest.json", {"design_identity": d.identity,
            "source": frozen, "runtime": {}})
        write_json(cell / "replication-0000/independent_assessment.json", {"posterior_passed": False})
        return {"status": "complete", "exit_code": 0, "elapsed_seconds": 1.}
    monkeypatch.setattr(recovery, "supervise_fit", supervise)
    job = campaign.make_job("fit", "K0", cell, design=d.payload(), first_quantum=4.)
    result = run_pool([job], campaign.executor(source, tmp_path / "attempts"),
        deadline=time.monotonic()+20., quantum_seconds=4., max_attempts=2)
    assert result["all_workloads_complete"]
    assert read_json(cell / "isolation_identity.json")["source_identity"] == frozen["identity"]
    exit_path = cell / "replication-0000/process-attempt-001-exit.json"
    assert read_json(exit_path)["assessment_sha256"] == file_hash(exit_path.parent / "independent_assessment.json")


@pytest.mark.parametrize("full_workload", [True, False])
def test_completed_negative_assessment_is_terminal_but_incomplete_work_cannot_price(tmp_path, design, full_workload):
    source, cell, original, identity = saved_fit(tmp_path, design)
    d = replace(original, options={**original.options, "member_rule": "shortest_verified_l",
        "posterior_members": "selected", "posterior_member_count": 1,
        "posterior_settings": {"retained_min_results": 4}})
    write_json(cell / "isolated_design.json", d.payload())
    identity["design_identity"] = d.identity
    write_json(cell / "isolation_identity.json", identity)
    path = cell / "replication-0000"
    fit = read_json(path / "fit_identity.json"); fit["design"] = d.identity
    write_json(path / "fit_identity.json", fit)
    prefix = path / "process-attempt-001"
    assessment = write_json(path / "independent_assessment.json", {"posterior_passed": False})
    write_json(str(prefix)+"-manifest.json", {"design_identity": d.identity,
        "source": {"identity": identity["source_identity"]}, "runtime": {}})
    write_json(str(prefix)+"-exit.json", {"status": "complete", "elapsed_seconds": 12.,
        "assessment_sha256": file_hash(assessment)})
    write_json(path / "pipeline.json", {"completion": "complete", "members": [
        {"status": "assessed", "recorded_retained_count": 4 if full_workload else 0}]})
    price = campaign.fit_price(cell, identity["source_identity"])
    assert price["final_assessment"] and not price["posterior_outcome_used_for_pricing"]
    assert (price["status"] == "complete_workload") == full_workload
    with pytest.raises(ValueError, match="source changed"):
        campaign.fit_price(cell, "wrong")
    write_json(assessment, {"posterior_passed": True})
    with pytest.raises(ValueError, match="assessment changed"):
        campaign.fit_price(cell, identity["source_identity"])

