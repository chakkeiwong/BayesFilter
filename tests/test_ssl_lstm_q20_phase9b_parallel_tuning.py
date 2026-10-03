"""CPU-only integration checks using synthetic devices and subprocess workers."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from bayesfilter.runtime.campaign_budget_ledger import CampaignBudgetLedger
from bayesfilter.runtime import display_gpu_policy, parallel_tuning


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "docs/benchmarks/run_ssl_lstm_q20_phase9b_parallel_tuning_2026_09_08.py"


@pytest.fixture
def runner():
    spec = importlib.util.spec_from_file_location("phase9b_parallel_tuning_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _ledger(tmp_path, runner, budget=20.0):
    return CampaignBudgetLedger.create(
        tmp_path / "ledger.json", campaign_id="synthetic-parallel-test", total_budget_seconds=budget,
        source_hash=runner._json_hash(runner._source_hashes()), plan_hash=runner._sha256(runner.PLAN),
        claim_boundary=runner.CLAIM_BOUNDARY,
    )


def _inventory():
    return {"trust_basis": "trusted_escalated_gpu_execution", "gpus": [
        {"index": index, "uuid": f"GPU-synthetic-{index}", "is_display": index == 0,
         "utilization_gpu_pct": 0, "memory_free_mib": 20000,
         "memory_total_mib": 20000, "memory_used_mib": 0}
        for index in range(3)
    ]}


def test_plan_only_imports_no_framework_and_detects_no_devices():
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": "-1"}
    code = (
        "import runpy, sys; namespace = runpy.run_path(sys.argv[1]); "
        "status = namespace['main'](['--plan-only']); "
        "assert not any(name in sys.modules for name in ('tensorflow', 'jax', 'torch')); "
        "raise SystemExit(status)"
    )
    result = subprocess.run([sys.executable, "-c", code, str(SCRIPT)], env=environment,
                            text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["status"] == "PLAN_ONLY_NO_GPU_WORK"


def test_new_profiles_keep_grid_and_use_fresh_disjoint_arm_seeds(runner, tmp_path):
    source = runner._load_source()
    profiles = [runner._profile(source, tmp_path, arm, 10) for arm in runner.ARMS]
    for profile in profiles:
        baseline = source._FACTOR_TUNING_R2_PROFILE
        assert profile.step_size_candidates == baseline.step_size_candidates
        assert profile.leapfrog_grid == baseline.leapfrog_grid
        assert profile.initial_state_bank == baseline.initial_state_bank
        assert profile.selection_replications == baseline.selection_replications
        assert profile.verification_num_results == baseline.verification_num_results
        assert profile.tuning_roots != baseline.tuning_roots
        assert (profile.scope_start, profile.scope_limit) == (2, 1)
    assert profiles[0].tuning_roots != profiles[1].tuning_roots
    assert profiles[0].training_roots != profiles[1].training_roots
    assert runner._profile(source, tmp_path / "retry", "factor", 10).tuning_roots != profiles[0].tuning_roots


def test_budget_is_checked_before_gpu_detection_or_output_creation(runner, tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "_source_hashes", lambda: {"synthetic": "source"})
    ledger = _ledger(tmp_path, runner, budget=1)
    monkeypatch.setattr(display_gpu_policy, "probe_inventory", lambda: pytest.fail("GPU detection before budget"))
    output = tmp_path / "attempt"
    with pytest.raises(runner.ParallelDiagnosticError, match="complete worker allocations"):
        runner._coordinate(output, ledger, 10, 0.1, 2)
    assert not output.exists()
    assert ledger.read()["attempts"] == []


def test_coordinator_refuses_other_campaign_without_resetting_ledger(runner, tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "_source_hashes", lambda: {"synthetic": "source"})
    ledger = _ledger(tmp_path, runner)
    monkeypatch.setattr(runner, "_source_hashes", lambda: {"synthetic": "changed"})
    before = ledger.checksum()
    with pytest.raises(runner.ParallelDiagnosticError, match="mismatch"):
        runner._coordinate(tmp_path / "attempt", ledger, 5, 0.1, 2)
    assert ledger.checksum() == before


@pytest.mark.parametrize("fail_worker", (False, True))
def test_coordinator_launches_real_processes_and_settles_each_once(runner, tmp_path, monkeypatch, fail_worker):
    monkeypatch.setattr(runner, "_source_hashes", lambda: {"synthetic": "source"})
    monkeypatch.setattr(display_gpu_policy, "probe_inventory", _inventory)
    monkeypatch.setattr(runner, "_validate_result", lambda *args: None)
    monkeypatch.setattr(runner, "_headroom_monitor", lambda *args: lambda: None)
    ledger = _ledger(tmp_path, runner)
    actual_wave = parallel_tuning.run_parallel_tuning_wave
    commands = []

    def synthetic_wave(tasks, **kwargs):
        commands.extend(task.command for task in tasks)
        code = (
            "from pathlib import Path; import json, os, time; "
            "root = Path(os.environ['BAYESFILTER_WORKER_OUTPUT_DIR']); root.mkdir(); "
            "time.sleep(0.1); "
            "root.joinpath('run_manifest.json').write_text(json.dumps({'status': 'completed'}))"
        )
        synthetic = tuple(replace(task, command=(sys.executable, "-c",
                          "raise SystemExit(2)" if fail_worker and task.task_id == "strict" else code)) for task in tasks)
        return actual_wave(synthetic, **kwargs)

    monkeypatch.setattr(parallel_tuning, "run_parallel_tuning_wave", synthetic_wave)
    result = runner._coordinate(tmp_path / "attempt", ledger, 5, 0.1, 2)
    assert result["status"].startswith("FAIL" if fail_worker else "PASS")
    assert result["p1_closeout_issued"] is False
    assert len(commands) == 2
    assert all(command[1:3] == (str(SCRIPT), "--worker-job") for command in commands)
    rows = result["waves"][0]["results"]
    assert {row["task"]["gpu_uuid"] for row in rows} == {"GPU-synthetic-1", "GPU-synthetic-2"}
    assert len({row["pid"] for row in rows}) == 2
    payload = ledger.read()
    assert payload["reserved_seconds"] == 0
    assert payload["consumed_seconds"] == pytest.approx(sum(row["elapsed_seconds"] for row in rows))
    settlements = [event for event in payload["events"] if event["event"] == "arm_settled"]
    assert len(settlements) == 2
    assert {event["arm"] for event in settlements} == {"factor", "strict"}


def test_no_device_releases_reservations_without_worker_charge(runner, tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "_source_hashes", lambda: {"synthetic": "source"})
    monkeypatch.setattr(display_gpu_policy, "probe_inventory", lambda: {"trust_basis": "trusted_escalated_gpu_execution", "gpus": []})
    ledger = _ledger(tmp_path, runner)
    result = runner._coordinate(tmp_path / "attempt", ledger, 5, 0.1, 2)
    assert result["status"].startswith("FAIL")
    assert ledger.read()["reserved_seconds"] == 0
    assert ledger.read()["consumed_seconds"] == 0


def _valid_result(runner, tmp_path):
    source = runner._load_source()
    job = runner._job(source, tmp_path, "factor", 10, {"synthetic": "source"})
    output = Path(job["output_dir"])
    profile = job["profile"]
    tuning = {
        "tuning_policy": "measured_joint_grid_v1",
        "candidates": [{"selected_step_size": step, "num_leapfrog_steps": leapfrog}
                       for step in profile["step_size_candidates"] for leapfrog in profile["leapfrog_grid"]],
        "candidate_selection": {"all_candidate_pairs_measured": True, "final_status": "passed",
                                "seed_ledger": {"all_seeds_unique": True},
                                "heldout_verification": {"final_status": "passed"}},
    }
    memory = {"all_physical_devices_memory_growth": True, "configured_before_logical_device_initialization": True}
    receipt = {"tuning_result": tuning, "handoff_hash": "synthetic-handoff", "scope_index": 2}
    runner._write_json(output / "tuning_receipt.json", receipt)
    runner._write_json(output / "memory_policy.json", memory)
    runner._write_json(output / "tuning/fixed_transport_hmc_tuning_result.json", tuning)
    hashes = {str(path.relative_to(output)): runner._sha256(path) for path in output.rglob("*.json")}
    result = {"status": "PASS_PHASE0_PARALLEL_TUNING_WORKER", "job": job,
              "gpu_uuid": "GPU-synthetic-1", "claim_boundary": runner.CLAIM_BOUNDARY,
              "memory_policy": memory, "jit_compile": True, "tf32_enabled": True,
              "headroom": {"status": "pass"}, "artifact_hashes": hashes}
    task = SimpleNamespace(gpu_uuid="GPU-synthetic-1", output_dir=output)
    return task, result, job


def test_complete_worker_reconciliation_is_not_just_manifest_existence(runner, tmp_path):
    task, payload, job = _valid_result(runner, tmp_path)
    runner._validate_result(task, payload, job)
    payload["gpu_uuid"] = "GPU-wrong"
    with pytest.raises(runner.ParallelDiagnosticError, match="device"):
        runner._validate_result(task, payload, job)


def test_missing_grid_pair_or_heldout_cannot_be_promoted(runner, tmp_path):
    task, payload, job = _valid_result(runner, tmp_path)
    receipt_path = task.output_dir / "tuning_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["tuning_result"]["candidates"][-1] = receipt["tuning_result"]["candidates"][0]
    receipt_path.write_text(json.dumps(receipt))
    tuning_path = task.output_dir / "tuning/fixed_transport_hmc_tuning_result.json"
    tuning_path.write_text(json.dumps(receipt["tuning_result"]))
    for path in (receipt_path, tuning_path):
        payload["artifact_hashes"][str(path.relative_to(task.output_dir))] = runner._sha256(path)
    with pytest.raises(runner.ParallelDiagnosticError, match="grid"):
        runner._validate_result(task, payload, job)


def test_runtime_memory_policy_is_required(runner, tmp_path):
    task, payload, job = _valid_result(runner, tmp_path)
    payload["memory_policy"]["all_physical_devices_memory_growth"] = False
    with pytest.raises(runner.ParallelDiagnosticError, match="runtime policy"):
        runner._validate_result(task, payload, job)


def test_smoke_job_and_tuning_job_are_distinct(runner, tmp_path):
    source = runner._load_source()
    smoke = runner._job(source, tmp_path, "factor", 300, {}, smoke_only=True)
    tuning = runner._job(source, tmp_path, "factor", 300, {})
    assert smoke["mode"] == "gpu_smoke"
    assert tuning["mode"] == "tuning"
    task, payload, job = _valid_result(runner, tmp_path / "worker")
    payload["status"] = "PASS_PHASE0_PARALLEL_GPU_SMOKE"
    with pytest.raises(runner.ParallelDiagnosticError, match="incomplete"):
        runner._validate_result(task, payload, job)


def test_smoke_reconciliation_requires_xla_and_reference_error(runner, tmp_path):
    task, payload, job = _valid_result(runner, tmp_path)
    job["mode"] = "gpu_smoke"
    payload["status"] = "PASS_PHASE0_PARALLEL_GPU_SMOKE"
    smoke = {"status": "PASS_GPU_XLA_STARTUP_SMOKE", "tracing_count": 1,
             "hlo_sha256": "synthetic-hlo", "call_seconds": [1.0, 0.1],
             "max_absolute_errors": [0.0, 0.0]}
    path = task.output_dir / "smoke_receipt.json"
    runner._write_json(path, smoke)
    payload["artifact_hashes"]["smoke_receipt.json"] = runner._sha256(path)
    runner._validate_result(task, payload, job)
    smoke["max_absolute_errors"] = [0.0, 1e-4]
    path.write_text(json.dumps(smoke))
    payload["artifact_hashes"]["smoke_receipt.json"] = runner._sha256(path)
    with pytest.raises(runner.ParallelDiagnosticError, match="smoke evidence"):
        runner._validate_result(task, payload, job)


def test_runtime_headroom_loss_is_a_resource_stop_not_a_load_veto(runner, tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "MONITOR_INTERVAL_SECONDS", 0)
    commands = []

    def telemetry(command, **kwargs):
        commands.append(command)
        return "GPU-synthetic-1, 5119\n"

    monkeypatch.setattr(runner.subprocess, "check_output", telemetry)
    monitor = runner._headroom_monitor((SimpleNamespace(gpu_uuid="GPU-synthetic-1"),), tmp_path, 0)
    with pytest.raises(runner.ParallelDiagnosticError, match="headroom lost"):
        monitor()
    assert commands[0][1] == "--query-gpu=uuid,memory.free"
    assert (tmp_path / "wave-0-headroom-0.json").is_file()


def test_parent_record_failure_still_charges_completed_workers(runner, tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "_source_hashes", lambda: {"synthetic": "source"})
    monkeypatch.setattr(display_gpu_policy, "probe_inventory", _inventory)
    ledger = _ledger(tmp_path, runner)
    write = runner._write_json

    def cannot_record_wave(path, payload):
        if path.name == "wave-0-result.json":
            raise OSError("synthetic full disk after completed processes")
        write(path, payload)

    def measured_wave(tasks, **kwargs):
        return {"status": "completed", "task_count": len(tasks), "worker_seconds": 1.5,
                "results": [{"status": "completed", "task": task.payload(), "elapsed_seconds": 0.75} for task in tasks]}

    monkeypatch.setattr(runner, "_write_json", cannot_record_wave)
    monkeypatch.setattr(parallel_tuning, "run_parallel_tuning_wave", measured_wave)
    result = runner._coordinate(tmp_path / "attempt", ledger, 5, 0.1, 2)
    assert result["status"].startswith("FAIL")
    assert ledger.read()["consumed_seconds"] == 1.5
    assert ledger.read()["reserved_seconds"] == 0.0


def test_single_non_display_gpu_queues_both_arms_and_records_actual_width(runner, tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "_source_hashes", lambda: {"synthetic": "source"})
    inventory = _inventory()
    inventory["gpus"][2]["utilization_gpu_pct"] = 41
    monkeypatch.setattr(display_gpu_policy, "probe_inventory", lambda: inventory)
    ledger = _ledger(tmp_path, runner)
    seen = []

    def queued_wave(tasks, **kwargs):
        seen.extend(task.gpu_uuid for task in tasks)
        assert len(tasks) == 1
        return {"status": "completed", "task_count": 1, "worker_seconds": 0.5,
                "results": [{"status": "completed", "task": tasks[0].payload(), "elapsed_seconds": 0.5}]}

    monkeypatch.setattr(parallel_tuning, "run_parallel_tuning_wave", queued_wave)
    result = runner._coordinate(tmp_path / "attempt", ledger, 5, 0.1, 2)
    assert result["status"].startswith("PASS")
    assert result["maximum_concurrent_workers"] == 1
    assert seen == ["GPU-synthetic-1", "GPU-synthetic-1"]
    assert ledger.read()["consumed_seconds"] == 1.0


def test_worker_calls_existing_chart_builder_and_public_scope_tuner(runner, tmp_path, monkeypatch):
    source = runner._load_source()
    monkeypatch.setattr(runner, "_load_source", lambda: source)
    monkeypatch.setattr(runner, "_source_hashes", lambda: {"synthetic": "source"})
    monkeypatch.setattr(runner, "_require_framework_free", lambda: None)
    job = runner._job(source, tmp_path, "factor", 10, runner._source_hashes())
    job_path = tmp_path / "factor-job.json"
    runner._write_json(job_path, job)
    monkeypatch.setenv("BAYESFILTER_WORKER_OUTPUT_DIR", job["output_dir"])
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-synthetic-1")
    monkeypatch.setenv("BAYESFILTER_SELECTED_GPU_UUID", "GPU-synthetic-1")
    monkeypatch.setenv("BAYESFILTER_GPU_SELECTION_POLICY_ID", display_gpu_policy.POLICY_ID)
    monkeypatch.setenv("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    monkeypatch.setattr(display_gpu_policy, "probe_inventory", _inventory)
    monkeypatch.setattr(display_gpu_policy, "check_runtime_headroom", lambda *args: {"status": "pass"})
    import bayesfilter.runtime.gpu_memory_policy as memory_module
    from contextlib import nullcontext

    memory = {"all_physical_devices_memory_growth": True, "configured_before_logical_device_initialization": True}
    monkeypatch.setattr(memory_module, "configure_tensorflow_gpu_memory_growth", lambda *args, **kwargs: memory)
    device = SimpleNamespace(name="/device:GPU:0")
    fake_tf = SimpleNamespace(
        __version__="synthetic-no-gpu", device=lambda *args: nullcontext(),
        config=SimpleNamespace(
            list_logical_devices=lambda *args: [device], list_physical_devices=lambda *args: [device],
            set_soft_device_placement=lambda *args: None,
            experimental=SimpleNamespace(enable_tensor_float_32_execution=lambda *args: None,
                                         tensor_float_32_execution_enabled=lambda: True,
                                         get_memory_info=lambda *args: {"current": 0, "peak": 0}),
        ),
    )
    monkeypatch.setitem(sys.modules, "tensorflow", fake_tf)
    monkeypatch.setitem(sys.modules, "tensorflow_probability", SimpleNamespace(__version__="synthetic-no-gpu"))
    monkeypatch.setitem(sys.modules, "bayesfilter.inference.tempered_target_tf", SimpleNamespace(
        make_q20_tempered_bridge=lambda *args, **kwargs: SimpleNamespace(target_signature=source.EXPECTED_TARGET_SIGNATURE)))
    calls = []

    def charts(*args):
        assert (Path(job["output_dir"]) / "memory_policy.json").is_file()
        calls.append("fresh_chart")
        return {1.0: "synthetic-chart"}, [], []

    def tune(*args, **kwargs):
        assert args[2] == "synthetic-chart"
        assert kwargs["scope_index"] == 2 and kwargs["beta"] == 1.0
        calls.append("existing_scope_tuner")
        return {"synthetic_mechanics_receipt": True}

    monkeypatch.setattr(source, "_build_fresh_chart", charts)
    monkeypatch.setattr(source, "_tune_scope", tune)
    assert runner._worker(job_path) == 0
    assert calls == ["fresh_chart", "existing_scope_tuner"]
    manifest = json.loads(Path(job["output_dir"], "run_manifest.json").read_text())
    assert manifest["p1_closeout_issued"] is False
    assert manifest["tensorflow"] == "synthetic-no-gpu"
