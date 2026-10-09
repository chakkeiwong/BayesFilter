"""Additional campaign allocation preserves numerical identity and evidence."""
from dataclasses import replace
from pathlib import Path
import shutil
import sys
import time

import pytest

from bayesfilter.testing.inference_validation import campaign_recovery as recovery
from bayesfilter.testing.inference_validation.designs import digest
from bayesfilter.testing.inference_validation.storage import file_hash, read_json, write_json


def stopped(seconds=10., **extra):
    return {"status": "timed_out", "elapsed_seconds": seconds, "exit_code": -15,
        "progress": {"initial": {"evidence_count": 0}, "last": {"evidence_count": 1}}, **extra}


@pytest.mark.parametrize("status,progress,expected", [
    ("timed_out", True, 4.), ("budget_exhausted", True, 4.),
    ("failed", True, 0.), ("complete", True, 0.), ("timed_out", False, 0.),
])
def test_additional_allocation_is_distinct_from_exhausted_fit(status, progress, expected):
    row = stopped(status=status)
    if not progress:
        row["progress"]["last"]["evidence_count"] = 0
    assert recovery.continuation_allowance([row], cumulative_cap_seconds=18., quantum_seconds=4.,
        outer_remaining_seconds=20., max_additional_attempts=2) == expected


def test_cumulative_cap_outer_deadline_and_attempt_limit_survive_restart():
    prior = [stopped(), stopped(4., campaign_continuation=True)]
    options = dict(cumulative_cap_seconds=18., quantum_seconds=4., max_additional_attempts=2)
    assert recovery.continuation_allowance(prior, outer_remaining_seconds=1.5, **options) == 1.5
    prior.append(stopped(1.5, campaign_continuation=True))
    assert recovery.continuation_allowance(prior, outer_remaining_seconds=100., **options) == 0.
    assert recovery.continuation_allowance([stopped(18.)], outer_remaining_seconds=100., **options) == 0.


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1., True])
def test_invalid_budget_cannot_launch(bad):
    with pytest.raises(ValueError):
        recovery.continuation_allowance([stopped()], cumulative_cap_seconds=bad, quantum_seconds=4.,
            outer_remaining_seconds=20., max_additional_attempts=2)


def frozen_source(root):
    root.mkdir(parents=True)
    file = root / "bayesfilter/example.py"
    file.parent.mkdir()
    file.write_text("original = True\n")
    files = {"bayesfilter/example.py": file_hash(file)}
    record = {"commit": "fixture", "files": files, "identity": digest(files)}
    write_json(root / "source_snapshot.json", {"git_commit": "fixture", "source": record})
    return record


def saved_fit(tmp_path, design):
    source = tmp_path / "source"
    record = frozen_source(source)
    cell = tmp_path / "cell"
    d = design("accuracy", "gaussian", "prepared", replications=1, device="cpu_reference", budget_seconds=20.,
        options={"data": None, "isolate_fits": True, "fit_process_timeout_seconds": 10.})
    write_json(cell / "isolated_design.json", d.payload())
    identity = {"source_identity": record["identity"], "design_identity": d.identity,
                "reuse_leapfrog_graphs": True}
    write_json(cell / "isolation_identity.json", identity)
    fit = cell / "replication-0000"
    write_json(fit / "fit_identity.json", {"source_identity": record["identity"], "design": d.identity,
        "fit_id": 0, "data": digest(None), "reuse_leapfrog_graphs": True})
    write_json(fit / "process-attempt-001-exit.json", stopped())
    write_json(fit / "process-attempt-001-launch.json", {"original": True})
    write_json(fit / "tuning/tuning_checkpoint.json", {"checkpoint": "preserved"})
    write_json(fit / "members/unfavorable/result.json", {"posterior_passed": False})
    return source, cell, d, identity


def test_frozen_child_resumes_under_new_allocation_and_reuses_negative_assessment(tmp_path, design, monkeypatch):
    source, cell, d, identity = saved_fit(tmp_path, design)
    calls = []
    def supervise(command, log, seconds, device, **kwargs):
        calls.append(command)
        assert kwargs["cwd"] == source
        assert command[4] == str(cell / "isolated_design.json")
        assert digest(read_json(cell / "isolated_design.json")) == digest(d.payload())
        assert seconds == 4.
        assert kwargs["timeout_policy"].max_extension_seconds == 0.
        assert read_json(cell / "replication-0000/members/unfavorable/result.json")["posterior_passed"] is False
        prefix = str(log).removesuffix(".log")
        write_json(prefix + "-manifest.json", {"design_identity": d.identity,
            "source": {"identity": identity["source_identity"]}, "runtime": {}})
        write_json(cell / "replication-0000/independent_assessment.json", {"posterior_passed": False})
        return {"status": "complete", "elapsed_seconds": 2., "exit_code": 0}
    monkeypatch.setattr(recovery, "supervise_fit", supervise)
    opts = dict(cell=cell, source=source, replication=0, cumulative_cap_seconds=18., quantum_seconds=4.,
                deadline=time.monotonic() + 20., max_additional_attempts=2)
    result = recovery.continue_frozen_fit(output=tmp_path / "attempt-1", **opts)
    assert result["cumulative_fit_seconds"] == 12.
    assert result["preserved_evidence_verified"] is True
    repeated = recovery.continue_frozen_fit(output=tmp_path / "attempt-2", **opts)
    assert repeated["status"] == "reused_final_assessment" and len(calls) == 1
    assert not (tmp_path / "attempt-2").exists()


@pytest.mark.parametrize("mutation,match", [
    ("source", "source changed"), ("design", "identical original"),
    ("fit", "fit identity"), ("launch", "unreconciled"), ("assessment", "assessment changed"),
])
def test_changed_evidence_never_reaches_worker(tmp_path, design, monkeypatch, mutation, match):
    source, cell, d, _ = saved_fit(tmp_path, design)
    fit = cell / "replication-0000"
    if mutation == "source": (source / "bayesfilter/example.py").write_text("changed = True\n")
    elif mutation == "design": write_json(cell / "isolated_design.json", replace(d, seed=d.seed + 1).payload())
    elif mutation == "fit":
        x = read_json(fit / "fit_identity.json"); x["fit_id"] = 1; write_json(fit / "fit_identity.json", x)
    elif mutation == "launch": write_json(fit / "process-attempt-002-launch.json", {})
    else: write_json(fit / "independent_assessment.json", {"changed": True})
    monkeypatch.setattr(recovery, "supervise_fit", lambda *a, **kw: pytest.fail("worker must not launch"))
    with pytest.raises(ValueError, match=match):
        recovery.continue_frozen_fit(cell=cell, source=source, replication=0, output=tmp_path / "attempt",
            cumulative_cap_seconds=18., quantum_seconds=4., deadline=time.monotonic() + 20., max_additional_attempts=2)


def test_numerical_failure_is_not_a_campaign_resource_retry(tmp_path, design, monkeypatch):
    source, cell, _, _ = saved_fit(tmp_path, design)
    write_json(cell / "replication-0000/process-attempt-001-exit.json", stopped(status="failed"))
    monkeypatch.setattr(recovery, "supervise_fit", lambda *a, **kw: pytest.fail("numerical retry forbidden"))
    result = recovery.continue_frozen_fit(cell=cell, source=source, replication=0, output=tmp_path / "attempt",
        cumulative_cap_seconds=18., quantum_seconds=4., deadline=time.monotonic() + 20., max_additional_attempts=2)
    assert result["status"] == "ineligible_or_allocation_exhausted"


def test_post_run_evidence_failure_keeps_the_terminal_process_receipt(tmp_path, design, monkeypatch):
    source, cell, d, identity = saved_fit(tmp_path, design)
    def supervise(command, log, seconds, device, **kwargs):
        prefix = str(log).removesuffix(".log")
        write_json(prefix + "-manifest.json", {"design_identity": d.identity,
            "source": {"identity": identity["source_identity"]}, "runtime": {}})
        write_json(cell / "replication-0000/members/unfavorable/result.json", {"posterior_passed": True})
        return {"status": "complete", "exit_code": 0, "elapsed_seconds": 1.}
    monkeypatch.setattr(recovery, "supervise_fit", supervise)
    with pytest.raises(ValueError, match="evidence changed"):
        recovery.continue_frozen_fit(cell=cell, source=source, replication=0, output=tmp_path / "attempt",
            cumulative_cap_seconds=18., quantum_seconds=4., deadline=time.monotonic() + 20., max_additional_attempts=2)
    receipt = read_json(cell / "replication-0000/process-attempt-002-exit.json")
    assert receipt["status"] == "continuation_validation_failed"
    assert receipt["elapsed_seconds"] == 1.
    assert (tmp_path / "attempt/result.json").exists()


def test_new_source_dependency_file_is_checked(tmp_path):
    source = tmp_path / "frozen"; frozen_source(source)
    dep = source / "data.json"; dep.write_text("[1]\n")
    write_json(tmp_path / "dependencies.json", {"data.json": file_hash(dep)})
    assert recovery.checked_frozen_source(source)["commit"] == "fixture"
    dep.write_text("[2]\n")
    with pytest.raises(ValueError, match="dependencies changed"):
        recovery.checked_frozen_source(source)


@pytest.mark.parametrize("target", ["ssm_campaign_location", "ssm_campaign_nonlinear"])
def test_real_frozen_ssm_child_resumes_checkpoint_with_external_allocation(tmp_path, target):
    from tests.inference_validation.test_ssm_public_pipeline import _design
    from tests.inference_validation.test_shared_gpu_recovery import STOP_AFTER_CHECKPOINT
    from bayesfilter.testing.inference_validation.execution import REPO
    from bayesfilter.testing.inference_validation.fit_supervision import supervise_fit
    source = tmp_path / "frozen"
    shutil.copytree(REPO / "bayesfilter", source / "bayesfilter", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    files = {str(p.relative_to(source)): file_hash(p) for p in sorted((source / "bayesfilter").rglob("*.py"))}
    record = {"commit": "CPU-regression", "files": files, "identity": digest(files)}
    write_json(source / "source_snapshot.json", {"git_commit": "CPU-regression", "source": record})
    cell = tmp_path / "fit"
    d = _design(target, "prepared")
    write_json(cell / "isolated_design.json", d.payload())
    write_json(cell / "isolation_identity.json", {"design_identity": d.identity,
        "source_identity": record["identity"], "reuse_leapfrog_graphs": True})
    path = cell / "replication-0000"; path.mkdir()
    prefix = path / "process-attempt-001"
    command = [sys.executable, "-c", STOP_AFTER_CHECKPOINT, "_pipeline_fit",
        str(cell / "isolated_design.json"), str(cell), "0", "120", "1", "--reuse-leapfrog-graphs"]
    write_json(str(prefix) + "-launch.json", {"command": command})
    receipt = supervise_fit(command, str(prefix) + ".log", 120., "cpu_reference", cwd=source,
        progress_root=path, allowance_path=str(prefix) + "-allowance.json",
        budget_receipt_path=str(prefix) + "-budget.json")
    assert receipt["status"] == "budget_exhausted", receipt
    # This is an injected cooperative stop after durable numerical work. The
    # pure scheduling tests above cover exhaustion of the original hard cap.
    write_json(str(prefix) + "-exit.json", receipt)
    before = recovery.immutable_fit_evidence(path)
    assert any("numerical_" in p for p in before)
    # Exercise campaign dispatch all the way into the actual frozen SSM child.
    from bayesfilter.testing.inference_validation.campaign_pool import run_pool
    from scripts.run_hmc_ssm_pooled_campaign import executor, make_job
    job = make_job("resume-ssm", "K0" if target.endswith("location") else "K7",
                   cell, first_quantum=120.)
    pool = run_pool([job], executor(source, tmp_path / "continuation-1"),
        quantum_seconds=120., deadline=time.monotonic() + 125., max_attempts=1)
    result = pool["rows"][0]["attempts"][0]
    assert result["status"] == "complete", result
    assert all(file_hash(p) == sha for p, sha in before.items())
    pipeline = read_json(path / "pipeline.json")
    assert pipeline["verified_candidate_ids"]
    assert any(m.get("status") == "assessed" and m["warmup_exclusion_matches"] for m in pipeline["members"])
    assert read_json(path / "process-attempt-002-manifest.json")["runtime"]["gpu_intentionally_hidden"]
