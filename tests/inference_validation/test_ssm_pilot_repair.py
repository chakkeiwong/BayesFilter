"""Repair allocation and complete cumulative pricing, without numerical work."""
from pathlib import Path
from types import SimpleNamespace
import sys
import time

import pytest

from scripts import run_hmc_ssm_pilot_repair as repair
from scripts import continue_hmc_ssm_after_diagnosis as sequence
from bayesfilter.testing.inference_validation.storage import file_hash, read_json, write_json


def campaign(tmp_path):
    pool = tmp_path / "repair"
    ledger = tmp_path / "ledger.json"
    write_json(ledger, {"grant_gpu_seconds": 100000., "remaining_gpu_seconds": 99400.,
        "allocations": {"canonical_neutra_pricing_reserved_only": 1200.},
        "records": [{"receipt": str(pool / "earlier/execution.json"), "elapsed_seconds": 500.},
                    {"receipt": repair.ssm_campaign.C1_RECEIPT, "elapsed_seconds": 100.}]})
    config = {"ledger": str(ledger), "campaign_started_epoch": time.time() - 40 * 3600,
        "ssm_accounting_roots": [], "prior_ssm_receipts": [], "ssm_gpu_cap_seconds": 129600.,
        "root": str(tmp_path / "old"), "ssm_root": str(tmp_path / "prepared")}
    return config, pool


def test_repair_pool_accounts_prior_charges_and_preserves_wall_clock(tmp_path):
    config, pool = campaign(tmp_path)
    assert repair.allocation(config, pool, "recover") <= repair.POOL_SECONDS - 500. - 30.
    assert repair.allocation(config, pool, "diagnose") == 700.
    assert repair.allocation(config, pool, "diagnose-k6-recovery") == 1775.
    config["campaign_started_epoch"] = time.time() - 43 * 3600
    with pytest.raises(ValueError, match="42/46-hour"):
        repair.allocation(config, pool, "recover")


def test_active_reservation_cannot_be_spent_again(tmp_path):
    config, pool = campaign(tmp_path)
    ledger = read_json(config["ledger"]); ledger["active_reservation"] = {"service": "other"}
    write_json(config["ledger"], ledger)
    with pytest.raises(ValueError, match="reservation"):
        repair.allocation(config, pool, "diagnose")


def test_final_nonlinear_allocation_is_one_attempt_and_keeps_all_prior_costs():
    assert repair.recovery_schedule("recover-k7-final") == ((3, 7),)
    assert len(repair.recovery_schedule("recover")) == 12
    prior = [{"elapsed_seconds": 1775.},
             {"elapsed_seconds": 1776., "campaign_continuation": True},
             {"elapsed_seconds": 1774., "campaign_continuation": True}]
    assert repair.recovery_limits(prior, final_k7=True, remaining_seconds=3200.) == (8525., 3200., 3)
    with pytest.raises(ValueError, match="exactly two"):
        repair.recovery_limits(prior[:-1], final_k7=True, remaining_seconds=3200.)


@pytest.mark.parametrize("incomplete", [None, "search", "members", "retained", "process"])
def test_price_counts_original_and_resume_work_without_using_posterior_outcome(tmp_path, incomplete):
    cell = tmp_path / "price-K0"; path = cell / "replication-0000"
    write_json(cell / "isolated_design.json", {"options": {
        "posterior_member_count": 2, "posterior_settings": {"retained_min_results": 1000}}})
    assessment = write_json(path / "independent_assessment.json", {"posterior_passed": False})
    write_json(path / "process-attempt-001-exit.json", {"status": "timed_out", "elapsed_seconds": 1775.})
    write_json(path / "process-attempt-002-exit.json", {"status": "complete", "elapsed_seconds": 300.,
        "continuation_invocation_seconds": 302., "campaign_continuation": True,
        "assessment_sha256": file_hash(assessment)})
    payload = {"completion": "complete", "candidate_count": 16,
        "members": [{"status": "assessed", "recorded_retained_count": 1000, "posterior_passed": False}] * 2}
    if incomplete == "search": payload["completion"] = "partial_budget"
    elif incomplete == "members": payload["members"] = payload["members"][:1]
    elif incomplete == "retained": payload["members"][0]["recorded_retained_count"] = 500
    elif incomplete == "process":
        receipt = read_json(path / "process-attempt-002-exit.json"); receipt["status"] = "timed_out"
        write_json(path / "process-attempt-002-exit.json", receipt)
    write_json(path / "pipeline.json", payload)
    result = repair.complete_price(cell, {"attempts": [{"elapsed_seconds": 1776.}]})
    if incomplete is None:
        assert result["price_seconds"] == 2078.
        assert result["posterior_outcome_used_for_pricing"] is False
    else:
        assert result["price_seconds"] is None


@pytest.mark.parametrize("diagnosis_exit,launch_expected", [(0, True), (1, False)])
def test_sequence_waits_for_settlement_then_continues_independent_resource_fits(tmp_path, monkeypatch, diagnosis_exit, launch_expected):
    from types import SimpleNamespace
    config = write_json(tmp_path / "config.json", {"campaign_started_epoch": time.time()})
    diagnosis = tmp_path / "diagnosis"
    write_json(diagnosis / "manifest.json", {"command": ["python", "run.py", "--service", "bounded-diagnosis"]})
    write_json(diagnosis / "result.json", {"exit_code": diagnosis_exit, "target_result": "failed_numerically"})
    monkeypatch.setattr(sequence.sys, "argv", ["sequence.py", str(config), str(diagnosis), str(tmp_path / "recovery"),
        "--gpu", "GPU-test", "--service", "bounded-recovery"])
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        if command[0] == "systemctl":
            return SimpleNamespace(stdout="active\n" if len(calls) == 1 else "inactive\n")
        assert "--mode" in command and command[command.index("--mode") + 1] == "recover"
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(sequence.subprocess, "run", run)
    monkeypatch.setattr(sequence.time, "sleep", lambda seconds: None)
    if launch_expected:
        with pytest.raises(SystemExit) as exc: sequence.main()
        assert exc.value.code == 0
        assert len(calls) == 3
    else:
        with pytest.raises(RuntimeError, match="coordinator failed"): sequence.main()
        assert len(calls) == 2


def test_launcher_leaves_closeout_time_inside_service_and_reservation(tmp_path, monkeypatch):
    config, pool = campaign(tmp_path)
    config_path = write_json(tmp_path / "config.json", config)
    monkeypatch.setattr(repair.sys, "argv", ["repair.py", str(config_path), str(pool / "recovery"),
        "--mode", "recover", "--service", "test-recovery", "--gpu", "GPU-test"])
    monkeypatch.setattr(repair, "allocation", lambda *args: 60.7)
    monkeypatch.setattr(repair.time, "monotonic", lambda: 1000.)
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(repair.subprocess, "run", run)
    with pytest.raises(SystemExit) as exc:
        repair.main()
    assert exc.value.code == 0
    command, kwargs = calls[0]
    assert "--property=RuntimeMaxSec=60" in command
    assert "--property=TimeoutStopSec=30" in command
    deadline = float(command[command.index("--work-deadline-monotonic") + 1])
    assert deadline == pytest.approx(1030.7)
    assert kwargs["timeout"] == pytest.approx(90.7)
    assert 60 + 30 <= kwargs["timeout"]


def _worker_fixture(tmp_path, monkeypatch, deadline):
    config, pool = campaign(tmp_path)
    config["source_identity"] = "test-source"
    config_path = write_json(tmp_path / "config.json", config)
    monkeypatch.setattr(repair, "checked_frozen_source", lambda path: {"identity": "test-source"})
    monkeypatch.setattr(repair.ssm_campaign, "check_preflight", lambda path: None)
    monkeypatch.setenv("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-test")
    return config, SimpleNamespace(config=config_path, output=pool / "diagnosis",
        mode="diagnose", seconds=60., service="test-diagnosis",
        work_deadline_monotonic=deadline)


@pytest.mark.parametrize("deadline", [None, float("nan"), -1.])
def test_worker_rejects_missing_or_expired_deadline_before_reservation(tmp_path, monkeypatch, deadline):
    config, args = _worker_fixture(tmp_path, monkeypatch, deadline)
    with pytest.raises(ValueError, match="deadline|no numerical time"):
        repair.worker(args)
    assert not args.output.exists()
    assert not read_json(config["ledger"]).get("active_reservation")


def test_real_diagnostic_timeout_settles_before_outer_guard(tmp_path, monkeypatch):
    # The numerical stand-in is a real CPU subprocess. No GPU is detected or
    # initialized: this checks coordinator closeout after a killed child.
    config, args = _worker_fixture(tmp_path, monkeypatch, time.monotonic() + 1.)
    real_supervise = repair.supervise_fit

    def cpu_child(command, log, seconds, device, **kwargs):
        assert 0 < seconds <= 1.
        kwargs["cwd"] = tmp_path
        kwargs["workload_probe"] = lambda *args: {"trusted_for_extension": False, "contended": False}
        return real_supervise([sys.executable, "-c", "import time; time.sleep(60)"],
                              log, seconds, "cpu_reference", **kwargs)

    monkeypatch.setattr(repair, "supervise_fit", cpu_child)
    repair.worker(args)
    result = read_json(args.output / "result.json")
    assert result["exit_code"] == 0
    assert len(result["outcomes"]) == 1
    assert result["outcomes"][0]["status"] == "timed_out"
    assert result["outcomes"][0]["pid"] is not None
    assert result["elapsed_seconds"] < args.seconds - repair.CLOSEOUT_RESERVE_SECONDS
    ledger = read_json(config["ledger"])
    assert not ledger.get("active_reservation")
    assert ledger["records"][-1] == read_json(args.output / "execution.json")
    manifest = read_json(args.output / "manifest.json")
    assert manifest["initial_numerical_allowance_seconds"] <= 1.
    assert manifest["closeout_reserve_seconds"] == 30.
