"""Profiling observes the numerical child without changing or replaying a fit."""
from hashlib import sha256
import pstats
import time

import pytest

from bayesfilter.testing.inference_validation import execution, fit_process
from bayesfilter.testing.inference_validation.profiling import HostProfile, profile_report
from bayesfilter.testing.inference_validation.storage import read_json, write_json


def test_profile_checksum_change_is_reported_without_changing_numerical_receipt(tmp_path):
    prefix = tmp_path / "replication-0000/process-attempt-001"
    prefix.parent.mkdir()
    profile = HostProfile(prefix, requested=True, scope="isolated_numerical_child")
    profile.start()
    profile.finish()
    receipt = write_json(str(prefix) + "-exit.json", {"status": "complete", "exit_code": 0})
    original = receipt.read_bytes()
    assert profile_report(tmp_path, isolated=True, requested=True)["status"] == "available"
    path = prefix.with_name(prefix.name + "-host.prof")
    path.write_bytes(path.read_bytes() + b"corrupted")
    report = profile_report(tmp_path, isolated=True, requested=True)
    assert report["status"] == "unavailable"
    assert report["profiles"][0]["reason"] == "saved profile checksum changed"
    assert receipt.read_bytes() == original


@pytest.mark.parametrize("fail", [False, True])
def test_child_profiles_actual_execution_even_when_it_fails(design, tmp_path, monkeypatch, fail):
    from bayesfilter.testing.inference_validation.engines import pipeline
    monkeypatch.setattr(execution, "configure_worker", lambda _: {})
    monkeypatch.setattr(execution, "source_state", lambda: {"identity": "fixture"})
    monkeypatch.setattr(fit_process, "resource_snapshot", lambda: {})
    def numerical_call(*args):
        if fail:
            raise ValueError("original numerical failure")
    monkeypatch.setattr(pipeline, "run_replication", numerical_call)
    path = write_json(tmp_path / "design.json", design().payload())
    assert fit_process.fit_worker(path, tmp_path, 0, 10., 1, profile_execution=True) == int(fail)
    prefix = tmp_path / "replication-0000/process-attempt-001"
    stats = pstats.Stats(str(prefix) + "-host.prof")
    assert any(key[2] == "numerical_call" for key in stats.stats)
    assert read_json(str(prefix) + "-profile.json")["scope"] == "isolated_numerical_child"
    if fail:
        assert read_json(str(prefix) + "-failure.json")["reason"] == "original numerical failure"


@pytest.mark.parametrize("broken_stage", ["enable", "disable", "dump_stats"])
@pytest.mark.parametrize("numerical_failure", [False, True])
def test_broken_profiling_preserves_child_outcome(design, tmp_path, monkeypatch,
                                                broken_stage, numerical_failure):
    import cProfile
    from bayesfilter.testing.inference_validation.engines import pipeline
    class BrokenProfile:
        def enable(self, **kwargs): pass
        def disable(self): pass
        def dump_stats(self, path): pass
    def broken(*args):
        raise RuntimeError("instrumentation unavailable")
    monkeypatch.setattr(BrokenProfile, broken_stage, broken)
    monkeypatch.setattr(cProfile, "Profile", BrokenProfile)
    monkeypatch.setattr(execution, "configure_worker", lambda _: {})
    monkeypatch.setattr(execution, "source_state", lambda: {"identity": "fixture"})
    monkeypatch.setattr(fit_process, "resource_snapshot", lambda: {})
    def numerical_call(*args):
        if numerical_failure:
            raise ValueError("original numerical failure")
    monkeypatch.setattr(pipeline, "run_replication", numerical_call)
    path = write_json(tmp_path / "design.json", design().payload())
    # Even warning-as-error policy must leave the original outcome intact.
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert fit_process.fit_worker(path, tmp_path, 0, 10., 1,
                                      profile_execution=True) == int(numerical_failure)
    prefix = tmp_path / "replication-0000/process-attempt-001"
    assert read_json(str(prefix) + "-profile.json")["status"] == "unavailable"
    if numerical_failure:
        assert read_json(str(prefix) + "-failure.json")["reason"] == "original numerical failure"


def test_completed_isolated_resume_reports_missing_profile_without_a_child(design, tmp_path, monkeypatch):
    from types import SimpleNamespace
    import sys
    monkeypatch.setattr(fit_process, "sys", SimpleNamespace(modules={}, executable=sys.executable))
    monkeypatch.setattr(execution, "source_state", lambda: {"identity": "fixture"})
    d = design("stopping", "gaussian", "ordinary", replications=1,
               options={"isolate_fits": True, "fit_process_timeout_seconds": 10})
    def complete(command, *args):
        write_json(tmp_path / "replication-0000/independent_assessment.json", {
            "replication": 0, "inventory": {"failures": []}, "members": [],
            "tuning_completion": "complete", "pipeline": str(tmp_path / "pipeline.json")})
        return {"status": "complete", "exit_code": 0, "elapsed_seconds": .01}
    monkeypatch.setattr(fit_process, "_supervise", complete)
    fit_process.run_isolated_replications(d, tmp_path, deadline=time.monotonic() + 10)
    receipt = (tmp_path / "replication-0000/process-attempt-001-exit.json").read_bytes()
    monkeypatch.setattr(fit_process, "_supervise", lambda *a: pytest.fail("completed fit was replayed"))
    result = fit_process.run_isolated_replications(d, tmp_path, deadline=time.monotonic() + 10,
                                                   profile_execution=True)
    assert result["profiling"]["status"] == "unavailable"
    assert (tmp_path / "replication-0000/process-attempt-001-exit.json").read_bytes() == receipt


@pytest.mark.parametrize("target", ["gaussian", "beta_binomial"])
def test_public_ordinary_pipeline_profile_preserves_numerics_and_resume(design, tmp_path, target):
    """Small CPU reference pair; no posterior adequacy or default claim."""
    d = design("stopping", target, "ordinary", replications=1, budget_seconds=180,
        l_grid=(3,), step_size=.4, posterior_cap=256,
        options={"isolate_fits": True, "fit_process_timeout_seconds": 160,
                 "posterior_members": "selected", "member_rule": "first_verified",
                 "acceptance_policy": {"practical_region": (.41, .99), "repair_region": (.405, .995)},
                 "search": {"pilot_enabled": False, "refinement_rounds": 0,
                            "total_budget_units": 24, "repair_reserve_units": 4, "evidence_rungs": (1,)},
                 "fixed_comparator": {"warmup_results": 64, "retained_results": 128}})
    suite = {"schema": "bayesfilter.inference_validation_suite.v1", "suite_id": "profile-pair",
             "profile": "test", "profiles": {"test": ["stopping"]}, "designs": [d.payload()]}
    outputs = []
    for flag in (False, True):
        root = tmp_path / ("on" if flag else "off")
        index = execution.run_suite(suite, root, profile_execution=flag)
        job = index["jobs"][d.design_id]
        assert job["status"] == "complete"
        fit = root / d.design_id / "replication-0000"
        native = read_json(fit / "pipeline.json")
        assert native["verified_candidate_ids"]
        assert any(m["status"] == "assessed" for m in native["members"])
        tuning = read_json(fit / "tuning/candidate_set_result.json")
        tensors = {str(p.relative_to(fit)): sha256(p.read_bytes()).hexdigest()
                   for pattern in ("*.tensor", "*.bin") for p in fit.rglob(pattern)}
        assert tensors
        outputs.append(({k: tuning[k] for k in ("scope", "config", "candidates", "candidate_states",
                                               "verified_candidate_ids", "completion_status")}, tensors))
        assert not (root / d.design_id / "attempt-001-host.prof").exists()
        if flag:
            assert job["profiling"]["status"] == "available"
            stats = pstats.Stats(str(fit / "process-attempt-001-host.prof"))
            names = {key[2] for key in stats.stats}
            assert {"run_replication", "run_hmc_posterior", "run_fixed_comparator"} <= names
            assert "tune_hmc_kernel" in names
    assert outputs[0] == outputs[1]
    # Request instrumentation for already completed, unprofiled evidence.
    before = read_json(tmp_path / "off/run_index.json")["jobs"][d.design_id]
    resumed = execution.run_suite(suite, tmp_path / "off", resume=True, profile_execution=True)
    after = resumed["jobs"][d.design_id]
    assert after["profiling"]["status"] == "unavailable"
    assert after["attempts"] == before["attempts"]
    assert after["result_sha256"] == before["result_sha256"]
    assert len(list((tmp_path / "off").rglob("process-attempt-*-exit.json"))) == 1
    # A once-present profile that is missing on resume is also unavailable.
    fit = tmp_path / "on" / d.design_id / "replication-0000"
    (fit / "process-attempt-001-host.prof").unlink()
    resumed = execution.run_suite(suite, tmp_path / "on", resume=True, profile_execution=True)
    assert resumed["jobs"][d.design_id]["profiling"]["status"] == "unavailable"
    assert len(list((tmp_path / "on").rglob("process-attempt-*-exit.json"))) == 1
