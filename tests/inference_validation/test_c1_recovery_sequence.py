"""C1 recovery must preserve its population, original source and grant."""
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from scripts import run_hmc_c1_recovery_then_ssm as recovery
from bayesfilter.testing.inference_validation.storage import file_hash, read_json, write_json


def test_original_selection_is_exact_and_requires_all_receipts(tmp_path):
    for index in range(256):
        write_json(tmp_path/f"replication-{index:04d}/process-attempt-001-exit.json",
            {"status":"timed_out" if index in recovery.INDICES else "complete"})
    assert recovery.selected_failures(tmp_path) == recovery.INDICES
    write_json(tmp_path/"replication-0000/process-attempt-001-exit.json", {"status":"timed_out"})
    with pytest.raises(ValueError, match="selection changed"):
        recovery.selected_failures(tmp_path)
    (tmp_path/"replication-0000/process-attempt-001-exit.json").unlink()
    with pytest.raises(ValueError, match="inventory is incomplete"):
        recovery.selected_failures(tmp_path)


def test_historical_snapshot_layout_and_unrecorded_source_are_checked(tmp_path):
    p = tmp_path/"bayesfilter/example.py"
    p.parent.mkdir()
    p.write_text("original = 1\n")
    files = {"bayesfilter/example.py":file_hash(p)}
    write_json(tmp_path/"source_snapshot.json", {"git_commit":"original", "files":files})
    assert recovery.checked_original_source(tmp_path)["identity"] == recovery.digest(files)
    (p.parent/"extra.py").write_text("unexpected = 1\n")
    with pytest.raises(ValueError, match="numerical source changed"):
        recovery.checked_original_source(tmp_path)


def test_charge_is_once_and_cannot_clear_another_reservation(tmp_path):
    ledger = tmp_path/"ledger.json"
    receipt = {"receipt":"recovery", "elapsed_seconds":12.}
    write_json(ledger, {"records":[], "grant_gpu_seconds":100.,
        "active_reservation":{"receipt":"recovery"}})
    recovery.settle(ledger,receipt)
    recovery.settle(ledger,receipt)
    assert read_json(ledger)["remaining_gpu_seconds"] == 88.
    value = read_json(ledger)
    value["active_reservation"] = {"receipt":"ssm"}
    write_json(ledger,value)
    with pytest.raises(ValueError, match="reservation changed"):
        recovery.settle(ledger,receipt)
    assert read_json(ledger)["active_reservation"]["receipt"] == "ssm"


def test_external_supervisor_runs_in_original_child_directory(tmp_path):
    old = tmp_path/"original"
    old.mkdir()
    marker = old/"unchanged-source"
    marker.write_text("original numerical package")
    record = recovery.supervise_fit([sys.executable,"-c",
        "from pathlib import Path; assert Path('unchanged-source').read_text() == 'original numerical package'"],
        tmp_path/"child.log",5.,"cpu_reference",cwd=old,
        workload_probe=lambda *a:{"contended":False,"trusted_for_extension":True})
    assert record["status"] == "complete"


def test_recovery_cgroup_cap_and_abnormal_settlement(tmp_path, monkeypatch):
    ledger = tmp_path/"ledger.json"
    output = tmp_path/"run"
    write_json(ledger,{"records":[],"grant_gpu_seconds":10000.})
    args = SimpleNamespace(service="parent.service",repo=tmp_path,output=output,
        ssm_root=tmp_path/"ssm",ledger=ledger)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES","GPU-test")
    commands = []
    def run(command, **kwargs):
        commands.append(command)
        assert "--property=RuntimeMaxSec=4800.0" in command
        assert "--property=KillMode=control-group" in command
        assert "--setenv=TF_FORCE_GPU_ALLOW_GROWTH=true" in command
        value = read_json(ledger)
        value["active_reservation"] = {"service":"parent-recovery", "receipt":str(output/"receipt.json")}
        write_json(ledger,value)
        return SimpleNamespace(returncode=137)
    monkeypatch.setattr(recovery.subprocess,"run",run)
    with pytest.raises(RuntimeError,match="recovery stopped"):
        recovery.run_recovery(args)
    assert len(commands) == 1
    value = read_json(ledger)
    assert "active_reservation" not in value
    assert value["charged_gpu_seconds"] == 4830.


def prior_completion(tmp_path):
    original, previous = tmp_path/"original", tmp_path/"previous"
    for root in (original,previous):
        write_json(root/"isolated_design.json",{"seed":123})
        write_json(root/"replication-0078/fit_identity.json",{"source_identity":"original","fit_id":78})
    path = previous/"replication-0078"
    assessment = write_json(path/"independent_assessment.json",{"posterior_passed":False})
    write_json(path/"process-attempt-001-exit.json",{"status":"complete","exit_code":0,
        "assessment_sha256":file_hash(assessment)})
    write_json(path/"process-attempt-001-manifest.json",{"runtime":{"jit_compile":True,
        "gpu_tensor_device":"GPU:0","memory_policy":{"configured_before_logical_device_initialization":True,
        "all_physical_devices_memory_growth":True,"physical_devices":["GPU:0"]}}})
    write_json(previous/"replication-0079/process-attempt-001-exit.json",{"status":"timed_out","exit_code":-15})
    return original, previous


def test_completed_recovery_is_reused_even_if_posterior_failed(tmp_path):
    original, previous = prior_completion(tmp_path)
    reused = recovery.completed_recoveries(previous,original)
    assert [r["replication"] for r in reused] == [78]
    assert reused[0]["new_elapsed_seconds"] == 0.
    assert tuple(i for i in recovery.INDICES if i not in {r["replication"] for r in reused}) == recovery.INDICES[1:]


@pytest.mark.parametrize("mutation",["identity","assessment","design","provenance","worker_failure"])
def test_invalid_previous_recovery_cannot_be_reused(tmp_path,mutation):
    original, previous = prior_completion(tmp_path)
    path=previous/"replication-0078"
    if mutation=="identity": write_json(path/"fit_identity.json",{"source_identity":"different"})
    elif mutation=="assessment": write_json(path/"independent_assessment.json",{"posterior_passed":True})
    elif mutation=="design": write_json(previous/"isolated_design.json",{"seed":124})
    elif mutation=="provenance": write_json(path/"process-attempt-001-manifest.json",{
        "runtime":{"jit_compile":False,"memory_policy":{}}})
    else: write_json(previous/"replication-0079/process-attempt-001-exit.json",{"status":"failed","exit_code":1})
    with pytest.raises(ValueError): recovery.completed_recoveries(previous,original)
