"""CPU-only continuation mechanics; short chains establish no scientific claims."""
from dataclasses import replace
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest
import tensorflow as tf

from scripts import resume_modern_hmc_frozen_full_model as driver
from bayesfilter.runtime.durable_tensor_checkpoint import DurableTensorCheckpoint, CheckpointError
from bayesfilter.runtime.execution_budget import execution_budget, ExecutionBudgetExceeded


@pytest.fixture
def cohort(tmp_path, request):
    from tests.test_modern_hmc_pipeline import setup
    from bayesfilter.inference import (tune_hmc_kernel, run_hmc_posterior,
        build_retained_bound_hmc_archive_runner_from_candidate_set_result, SequentialNeuTraHMCConfig)
    binding, search = setup()
    if getattr(request, "param", "uniform_duration") == "fixed":
        from tests.test_hmc_candidate_set_execution import make_binding
        from bayesfilter.inference.hmc_transition_spec import HMCTransitionProfile
        binding = make_binding(config=replace(binding.config,
            transition_profile=HMCTransitionProfile("hmc", "fixed")))
    run = tune_hmc_kernel(adapter=binding._base_adapter, initial_position=binding.initial_active_state,
        config=search, candidate_set_adapter=binding.typed_adapter, output_dir=tmp_path / "tuning")
    members = {cid: build_retained_bound_hmc_archive_runner_from_candidate_set_result(
        candidate_set_result=run.result, candidate_id=cid, retained_binding=binding)
        for cid in run.result.verified_candidate_ids}
    candidates = [c.payload() for c in run.result.candidates]

    def factory(member, index):
        return SequentialNeuTraHMCConfig(step_size=member.step_size,
            num_leapfrog_steps=member.num_leapfrog_steps, warmup_seed=(90031, 2 * index),
            retained_seed=(90031, 2 * index + 1), warmup_chunk_results=128,
            warmup_min_results=256, warmup_check_window_results=128, warmup_max_results=512,
            retained_chunk_results=128, retained_min_results=128, retained_max_results=512,
            jit_compile=False)

    first = next(iter(members.values()))
    config = factory(first, 0)
    source = tmp_path / "previous/member-000"
    source.mkdir(parents=True)
    identity = dict(member_hash=first.member_hash, policy=driver.native(config.payload(chain_count=4)),
                    quantities=["x", "y"])
    driver.write(source / "policy.json", identity["policy"])
    checks = []
    def stop(_):
        checks.append(1)
        if len(checks) == 2:
            raise ExecutionBudgetExceeded("intentional stop after committed chunk")
    with DurableTensorCheckpoint(source / "chunks", identity) as store:
        with pytest.raises(ExecutionBudgetExceeded):
            run_hmc_posterior(member=first, config=config, parameter_names=("x", "y"),
                              checkpoint_store=store, budget_check=stop)
    rows = [dict(candidate_id=c["candidate_id"], tuning_state="verified",
        status="posterior_interrupted_resource_stopped" if i == 0 else "posterior_not_started",
        result_file=str(source / "result.json") if i == 0 else None) for i, c in enumerate(candidates)]
    return dict(candidates=candidates, members=members, previous_report=dict(rows=rows,
        verified_candidate_ids=list(members)), factory=factory, names=("x", "y"),
        transform=None, reference_check=lambda values, config: dict(passed=True),
        check_inputs=lambda: None)


def check_equal(direct, destination):
    for key in ("private_warmup_raw", "private_retained_raw"):
        if direct.get(key) is not None:
            tensor = tf.io.parse_tensor(tf.io.read_file(str(destination / (key.removeprefix("private_") + ".tensor"))), tf.float64)
            tf.debugging.assert_equal(tensor, direct[key])


@pytest.mark.parametrize("cohort", ["uniform_duration", "fixed"], indirect=True)
def test_native_partial_and_unstarted_members_equal_uninterrupted(cohort, tmp_path):
    from bayesfilter.inference import run_hmc_posterior
    direct = [run_hmc_posterior(member=m, config=cohort["factory"](m, i), parameter_names=("x", "y"))
              for i, m in enumerate(cohort["members"].values())]
    before = driver.tree_hashes(tmp_path / "previous")
    output = tmp_path / "resume/posterior"
    result = driver.run_cohort(**cohort, output=output)
    assert result["status"] == "completed" and len(result["rows"]) == 2
    for i, baseline in enumerate(direct):
        check_equal(baseline, output / f"member-{i:03d}")
        assert driver.read(output / f"member-{i:03d}/result.json")["warmup_excluded_from_posterior"]
    assert driver.read(output / "member-000/checkpoint-replay.json")["replayed_count"] == 1
    assert driver.read(output / "member-001/checkpoint-replay.json")["replayed_count"] == 0
    assert driver.tree_hashes(tmp_path / "previous") == before


@pytest.mark.parametrize("cohort", ["uniform_duration", "fixed"], indirect=True)
def test_repeated_interruptions_preserve_prefix_and_original_index_seeds(cohort, tmp_path, monkeypatch):
    from bayesfilter.inference import run_hmc_posterior
    direct = [run_hmc_posterior(member=m, config=cohort["factory"](m, i), parameter_names=("x", "y"))
              for i, m in enumerate(cohort["members"].values())]
    binding = next(iter(cohort["members"].values()))._binding
    original = binding._run
    calls = []
    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(binding, "_run", counted)
    with execution_budget(check=lambda: len(calls) < 1):
        first = driver.run_cohort(**cohort, output=tmp_path / "again/posterior")
    assert first["status"] == "resource_stopped"
    replay = driver.read(tmp_path / "again/posterior/member-000/checkpoint-replay.json")
    assert replay["replayed_count"] == replay["new_count"] == 1
    second = driver.run_cohort(**dict(cohort, previous_report=first), output=tmp_path / "last/posterior")
    assert second["status"] == "completed"
    assert driver.read(tmp_path / "last/posterior/member-000/checkpoint-replay.json")["replayed_count"] == 2
    for i, baseline in enumerate(direct):
        check_equal(baseline, tmp_path / f"last/posterior/member-{i:03d}")


def test_complete_success_and_failure_are_carried_without_sampling(cohort, tmp_path, monkeypatch):
    first = driver.run_cohort(**cohort, output=tmp_path / "first/posterior")
    failed_path = tmp_path / "first/posterior/member-001/result.json"
    failure = driver.read(failed_path)
    failure.update(passed=False, hard_vetoes=["fixture_terminal_veto"])
    driver.write(failed_path, failure)
    first["rows"][1].update(status="posterior_incomplete", delivered=False,
        posterior_hard_vetoes=["fixture_terminal_veto"])
    import bayesfilter.inference as inference
    monkeypatch.setattr(inference, "run_hmc_posterior", lambda **kwargs: pytest.fail("completed member repeated"))
    second = driver.run_cohort(**dict(cohort, previous_report=first), output=tmp_path / "second/posterior")
    assert second["status"] == "completed"
    assert all(r["continuation_action"] == "carried_terminal_result" for r in second["rows"])
    assert not second["rows"][1]["delivered"]
    assert second["rows"][1]["posterior_hard_vetoes"] == ["fixture_terminal_veto"]


def test_saved_result_without_reference_is_assessed_without_new_sampling(cohort, tmp_path, monkeypatch):
    first = driver.run_cohort(**cohort, output=tmp_path / "first/posterior")
    first["rows"][0].pop("reference")
    first["rows"][0]["delivered"] = False  # Provisional decision before reference serialization.
    seen = []
    import bayesfilter.inference as inference
    monkeypatch.setattr(inference, "run_hmc_posterior", lambda **kwargs: pytest.fail("saved result repeated"))
    def assess(values, config):
        seen.append(values)
        return dict(passed=True)
    second = driver.run_cohort(**dict(cohort, previous_report=first, reference_check=assess),
                               output=tmp_path / "second/posterior")
    assert second["status"] == "completed" and len(seen) == 1
    saved = driver.read(first["rows"][0]["result_file"])
    assert second["rows"][0]["delivered"] == bool(saved["passed"] and not saved["hard_vetoes"])


@pytest.mark.parametrize("missing", ["middle", "last"])
def test_missing_committed_chunk_is_rejected_before_numerical_work(cohort, tmp_path, monkeypatch, missing):
    first = driver.run_cohort(**cohort, output=tmp_path / "first/posterior")
    source = driver.member_source(first["rows"][0])
    chunks = sorted((source / "chunks/committed").iterdir())
    assert len(chunks) >= 2
    shutil.rmtree(chunks[0] if missing == "middle" else chunks[-1])
    for member in cohort["members"].values():
        monkeypatch.setattr(member._binding, "_run", lambda *a, **k: pytest.fail("missing chunk recomputed"))
    with pytest.raises(ValueError, match="gap|missing"):
        driver.run_cohort(**dict(cohort, previous_report=first), output=tmp_path / "rejected/posterior")


@pytest.mark.parametrize("change", ["seed", "policy", "member", "quantities", "missing_directory"])
def test_changed_saved_identity_rejected_before_output(cohort, tmp_path, change):
    member = next(iter(cohort["members"].values()))
    config = cohort["factory"](member, 0)
    source = driver.member_source(cohort["previous_report"]["rows"][0])
    names = ("x", "y")
    if change == "seed":
        config = replace(config, retained_seed=(111, 222))
    elif change == "policy":
        config = replace(config, retained_max_results=1024)
    elif change == "member":
        member = SimpleNamespace(member_hash="wrong", initial_active_state=member.initial_active_state)
    elif change == "quantities":
        names = ("x", "wrong")
    else:
        source = tmp_path / "missing"
    with pytest.raises(ValueError, match="changed|missing"):
        driver.validate_member_source(member, config, names, source)


def test_corrupt_committed_prefix_never_recomputes_numerical_work(cohort, tmp_path, monkeypatch):
    tensor = next((tmp_path / "previous/member-000/chunks/committed/chunk-000000").glob("tensor-*.bin"))
    tensor.write_bytes(b"broken")
    for member in cohort["members"].values():
        monkeypatch.setattr(member._binding, "_run", lambda *a, **k: pytest.fail("corrupt prefix recomputed"))
    with pytest.raises(CheckpointError, match="checksum"):
        driver.run_cohort(**cohort, output=tmp_path / "rejected/posterior")


@pytest.mark.parametrize("rows", [[], [{"candidate_id": "a"}, {"candidate_id": "a"}], [{"candidate_id": "b"}]])
def test_missing_duplicate_or_foreign_inventory_rejected(rows):
    with pytest.raises(ValueError, match="identity|inventory"):
        driver.validate_inventory([{"candidate_id": "a"}], {"a": object()}, rows)


@pytest.mark.parametrize("uuid", ["original-device", "changed-device"])
def test_physical_gpu_identity_checked_without_requiring_idle_device(monkeypatch, uuid):
    snapshot = f"1, {uuid}, Test device, 32000, 4000, 100\n"
    monkeypatch.setattr(driver.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=snapshot))
    expected = dict(index=1, uuid="original-device")
    if uuid != expected["uuid"]:
        with pytest.raises(ValueError, match="GPU identity changed"):
            driver.checked_gpu_identity(expected)
    else:
        assert driver.checked_gpu_identity(expected)["utilization_percent"] == 100


def test_hard_timeout_preserves_completed_and_unstarted_inventory(tmp_path, monkeypatch):
    prepared = dict(tuning_run=str(tmp_path / "original"), report=dict(rows=[
        dict(candidate_id="done", status="posterior_checks_passed", delivered=True),
        dict(candidate_id="waiting", status="posterior_not_started")]))
    monkeypatch.setattr(driver, "preparation", lambda *a: prepared)
    class Process:
        pid = 123
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def wait(self, **kwargs): return -9
    monkeypatch.setattr(driver.subprocess, "Popen", lambda *a, **kw: Process())
    killed = []
    monkeypatch.setattr(driver.os, "killpg", lambda *args: killed.append(args))
    clock = iter([0., 61., 62.])
    monkeypatch.setattr(driver.time, "monotonic", lambda: next(clock))
    args = SimpleNamespace(output=tmp_path / "output", previous=tmp_path / "original",
        source_root=tmp_path, plan="test", max_wall_seconds=60., stop_margin_seconds=1., preflight_only=False)
    assert driver.supervise(args) == 1
    result = driver.read(args.output / "run/result.json")
    assert killed and result["status"] == "supervisor_budget_exhausted"
    assert result["delivered_count"] == 1 and len(result["rows"]) == 2
