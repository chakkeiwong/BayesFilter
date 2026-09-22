"""Process-boundary failures must not become successful numerical replications."""
from dataclasses import replace
from pathlib import Path
import os
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from bayesfilter.testing.inference_validation import fit_process
from bayesfilter.testing.inference_validation.storage import read_json, write_json


def isolated(design, **options):
    return design("stopping", "gaussian", "ordinary", replications=2,
                  options={"isolate_fits": True, "fit_process_timeout_seconds": 10, **options})


@pytest.mark.parametrize("options", [
    {"isolate_fits": 1}, {"isolate_fits": True},
    {"isolate_fits": True, "fit_process_timeout_seconds": True},
    {"isolate_fits": True, "fit_process_timeout_seconds": float("nan")},
    {"isolate_fits": True, "fit_process_timeout_seconds": 121},
    {"fit_process_timeout_seconds": 2},
    {"isolate_fits": True, "fit_process_timeout_seconds": 10, "profile_execution": True},
])
def test_isolation_requires_supported_explicit_allocation(design, options):
    with pytest.raises(ValueError):
        design("stopping", "gaussian", "ordinary", options=options)
    with pytest.raises(ValueError, match="pipeline engine"):
        design("mechanics", "gaussian", "frozen", options={"isolate_fits": True,
                                                          "fit_process_timeout_seconds": 10})


def fake_parent(monkeypatch):
    from bayesfilter.testing.inference_validation import execution
    monkeypatch.setattr(fit_process, "sys", SimpleNamespace(modules={}, executable=sys.executable))
    monkeypatch.setattr(execution, "source_state", lambda: {"identity": "source-a"})


def saved_assessment(command):
    root, rep = Path(command[5]), int(command[6])
    row = {"replication": rep, "inventory": {"failures": []}, "members": [],
           "tuning_completion": "complete", "pipeline": str(root / "pipeline.json")}
    write_json(root / f"replication-{rep:04d}" / "independent_assessment.json", row)


def test_fresh_children_keep_replication_ids_and_resume_without_replay(design, tmp_path, monkeypatch):
    fake_parent(monkeypatch)
    commands = []
    def complete(command, log, seconds, device):
        commands.append(command)
        saved_assessment(command)
        return {"status": "complete", "exit_code": 0, "elapsed_seconds": .01}
    monkeypatch.setattr(fit_process, "_supervise", complete)
    d = isolated(design)
    first = fit_process.run_isolated_replications(d, tmp_path, deadline=time.monotonic()+30)
    assert first["completed"] == first["planned"] == 2
    assert [int(command[6]) for command in commands] == [0, 1]
    again = fit_process.run_isolated_replications(d, tmp_path, deadline=time.monotonic()+30)
    assert len(commands) == 2 and again == first
    with pytest.raises(ValueError, match="identical design and source"):
        fit_process.run_isolated_replications(replace(d, seed=d.seed+1), tmp_path,
                                              deadline=time.monotonic()+30)
    from bayesfilter.testing.inference_validation import execution
    monkeypatch.setattr(execution, "source_state", lambda: {"identity": "source-b"})
    with pytest.raises(ValueError, match="identical design and source"):
        fit_process.run_isolated_replications(d, tmp_path, deadline=time.monotonic()+30)


@pytest.mark.parametrize("status,code,save", [("failed", 7, False), ("timed_out", -15, False),
                                               ("complete", 0, False), ("failed", -15, True)])
def test_failed_child_and_missing_output_keep_full_denominator(design, tmp_path, monkeypatch,
                                                              status, code, save):
    fake_parent(monkeypatch)
    def fail(command, log, seconds, device):
        if save:
            saved_assessment(command)
        return {"status": status, "exit_code": code, "elapsed_seconds": 10.}
    monkeypatch.setattr(fit_process, "_supervise", fail)
    result = fit_process.run_isolated_replications(isolated(design), tmp_path,
                                                  deadline=time.monotonic()+30)
    assert result["completed"] == 0 and result["execution_failures"] == 1
    assert result["planned"] == 2 and not result["assessment_complete"]
    assert all(row["planned"] == row["unavailable"] == 2
               for row in result["interval_coverage_at_stop"].values())
    if save:
        monkeypatch.setattr(fit_process, "_supervise", lambda *a: pytest.fail("completed evidence rerun"))
        # A completed but abnormal process remains failed evidence on restart.
        result = fit_process.run_isolated_replications(replace(isolated(design), replications=2), tmp_path,
                                                       deadline=time.monotonic()-1)
        assert result["execution_failures"] == 1
        path = tmp_path / "replication-0000" / "independent_assessment.json"
        write_json(path, {"changed": True})
        with pytest.raises(ValueError, match="changed after process exit"):
            fit_process.run_isolated_replications(isolated(design), tmp_path,
                                                  deadline=time.monotonic()+30)


def test_unreconciled_launch_does_not_start_duplicate_worker(design, tmp_path, monkeypatch):
    fake_parent(monkeypatch)
    write_json(tmp_path / "replication-0000" / "process-attempt-001-launch.json", {})
    with pytest.raises(ValueError, match="unreconciled child"):
        fit_process.run_isolated_replications(isolated(design), tmp_path, deadline=time.monotonic()+30)


def test_partial_failure_resumes_under_remaining_cap(design, tmp_path, monkeypatch):
    fake_parent(monkeypatch)
    cap = []
    def execute(command, log, seconds, device):
        cap.append(seconds)
        if len(cap) == 1:
            write_json(Path(command[5]) / "replication-0000" / "partial.json", {"preserved": True})
            return {"status": "failed", "exit_code": 2, "elapsed_seconds": 3.}
        saved_assessment(command)
        return {"status": "complete", "exit_code": 0, "elapsed_seconds": .1}
    monkeypatch.setattr(fit_process, "_supervise", execute)
    d = isolated(design)
    fit_process.run_isolated_replications(d, tmp_path, deadline=time.monotonic()+30)
    result = fit_process.run_isolated_replications(d, tmp_path, deadline=time.monotonic()+30)
    assert cap == [10., 7., 10.]
    assert result["completed"] == 2
    assert read_json(tmp_path / "replication-0000" / "partial.json")["preserved"]
    assert read_json(tmp_path / "replication-0000" / "process-attempt-001-exit.json")["exit_code"] == 2


def test_real_supervisor_records_exit_and_bounds_hung_process(tmp_path):
    failure = fit_process._supervise([sys.executable, "-c", "raise SystemExit(7)"],
                                     tmp_path / "failure.log", 5., "cpu_reference")
    assert failure["status"] == "failed" and failure["exit_code"] == 7
    timeout = fit_process._supervise([sys.executable, "-c", "import time; time.sleep(30)"],
                                     tmp_path / "timeout.log", .1, "cpu_reference")
    assert timeout["status"] == "timed_out" and timeout["elapsed_seconds"] < 5.
    with pytest.raises(ProcessLookupError):
        os.kill(timeout["pid"], 0)


def test_coordinator_import_stays_framework_free():
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", BAYESFILTER_PRELOAD_CUSTOM_OP="0")
    code = ("import sys; from bayesfilter.testing.inference_validation.fit_process import run_isolated_replications; "
            "from bayesfilter.testing.inference_validation.engines.pipeline import summarize_replications; "
            "assert 'tensorflow' not in sys.modules")
    subprocess.run([sys.executable, "-c", code], env=env, check=True, timeout=30)


def test_resource_inspection_accepts_extension_type_module_descriptors(monkeypatch):
    # Some extension metaclasses expose __module__ as a descriptor, not a str.
    class Unusual:
        __module__ = property(lambda self: "extension")
    monkeypatch.setattr(fit_process.gc, "get_objects", lambda: [Unusual()])
    snapshot = fit_process.resource_snapshot()
    assert snapshot["live_objects"] == {}
    assert snapshot["max_rss_kib"] > 0


@pytest.mark.parametrize("target", ["gaussian", "beta_binomial"])
def test_public_isolated_pipeline_preserves_outputs_and_restart(design, tmp_path, target):
    """Actual tuning, export/reload, posterior, independent check and normal exit."""
    from bayesfilter.testing.inference_validation.execution import run_suite
    d = design("stopping", target, "prepared", replications=1, budget_seconds=120,
        l_grid=(3,), step_size=.8, posterior_cap=128,
        options={"isolate_fits": True, "fit_process_timeout_seconds": 100,
                 "posterior_members": "selected", "member_rule": "first_verified",
                 "acceptance_policy": {"practical_region": (.41,.99), "repair_region": (.405,.995)},
                 "search": {"pilot_enabled": False, "refinement_rounds": 0,
                            "total_budget_units": 24, "repair_reserve_units": 4, "evidence_rungs": (1,)},
                 "fixed_comparator": {"warmup_results": 64, "retained_results": 128}})
    suite = {"schema": "bayesfilter.inference_validation_suite.v1",
             "suite_id": "isolated-integration", "profile": "test", "profiles": {"test": ["stopping"]},
             "designs": [d.payload()]}
    # Use the public resolver's explicit-design form.
    from bayesfilter.testing.inference_validation.designs import resolve_suite
    assert resolve_suite(suite) == [d]
    index = run_suite(suite, tmp_path / "run")
    assert index["jobs"][d.design_id]["status"] == "complete"
    root = tmp_path / "run" / d.design_id
    assessment = read_json(root / "assessment.json")
    assert not assessment["framework_initialized_in_coordinator"]
    assert assessment["completed"] == 1 and assessment["verified_members"] > 0
    native = read_json(root / "replication-0000" / "pipeline.json")
    assert native["verified_candidate_ids"]
    member = next(m for m in native["members"] if m["status"] == "assessed")
    assert member["warmup_exclusion_matches"]
    assert member["fixed_comparator"]["status"] == "assessed"
    receipt = read_json(root / "replication-0000" / "process-attempt-001-exit.json")
    assert receipt["exit_code"] == 0
    resumed = run_suite(suite, tmp_path / "run", resume=True)
    assert resumed["jobs"] == index["jobs"]
    assert len(list(root.glob("replication-*/process-attempt-*-exit.json"))) == 1
