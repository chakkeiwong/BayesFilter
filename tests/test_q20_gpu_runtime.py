"""CPU-only resource/lifecycle regression tests; no GPU is probed."""
import builtins
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from bayesfilter.inference import q20_gpu_runtime as gpu
from bayesfilter.inference.q20_campaign_runtime import Campaign
from bayesfilter.inference.q20_master_program import execute_master
from bayesfilter.inference.q20_production_config import digest, protocol_template
from tests.test_q20_campaign_runtime import campaign


def inventory(monkeypatch, processes=""):
    devices = "0, GPU-zero, same GPU, 300, 32760, 4\n1, GPU-one, same GPU, 900, 32760, 2\n2, GPU-two, same GPU, 250, 32760, 36\n"
    def output(command, **kwargs):
        return devices if command[1].startswith("--query-gpu=") else processes
    monkeypatch.setattr(gpu.subprocess, "check_output", output)


def test_capacity_receipt_prevents_framework_import(monkeypatch):
    monkeypatch.delitem(sys.modules, "tensorflow", raising=False)
    monkeypatch.setenv("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    inventory(monkeypatch, "GPU-zero, 11, /env/bin/python\nGPU-one, 12, /env/bin/python\nGPU-two, 13, /env/bin/python\n")
    with pytest.raises(gpu.GPUResourceUnavailable) as caught:
        gpu.select_worker_gpu()
    assert caught.value.receipt["error"] == "no_gpu_without_numerical_workload"
    assert len(caught.value.receipt["gpu_inventory"]) == 3
    assert "tensorflow" not in sys.modules


def test_actual_launch_selection_and_contention(monkeypatch):
    monkeypatch.delitem(sys.modules, "tensorflow", raising=False)
    monkeypatch.setenv("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    inventory(monkeypatch, "GPU-zero, 111, /env/bin/python\nGPU-one, 112, /env/bin/python\n")
    receipt = gpu.select_worker_gpu()
    assert receipt["selected_host_gpu"] == 2
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "2"
    inventory(monkeypatch, f"GPU-zero, 111, /env/bin/python\nGPU-two, {os.getpid()}, /env/bin/python\nGPU-two, 222, /env/bin/python\n")
    with pytest.raises(gpu.GPUResourceUnavailable) as caught:
        gpu.check_gpu_contention(receipt)
    assert caught.value.receipt["other_gpu_pids"] == [222]


def test_desktop_services_are_recorded_but_do_not_block_research(monkeypatch):
    inventory(monkeypatch, "GPU-zero, 111, /env/bin/python\nGPU-one, 112, /env/bin/python\nGPU-two, 6247, /usr/libexec/gnome-remote-desktop-daemon\n")
    receipt = gpu.probe_gpu_inventory()
    assert receipt["selected_host_gpu"] == 2
    assert receipt["desktop_coexistence"]
    assert gpu.check_gpu_contention(receipt)["desktop_processes"][0]["pid"] == 6247
    inventory(monkeypatch, "GPU-zero, 111, /env/bin/python\nGPU-one, 112, /env/bin/python\nGPU-two, 123, /tmp/nxnode.bin\n")
    with pytest.raises(gpu.GPUResourceUnavailable):
        gpu.probe_gpu_inventory()


def test_clean_device_preferred_and_explicit_device_respected(monkeypatch):
    inventory(monkeypatch, "GPU-zero, 6866, /usr/NX/bin/nxnode.bin\nGPU-two, 6247, /usr/libexec/gnome-remote-desktop-daemon\n")
    assert gpu.probe_gpu_inventory()["selected_host_gpu"] == 1
    assert gpu.probe_gpu_inventory(requested="2")["selected_host_gpu"] == 2
    with pytest.raises(RuntimeError, match="not present"):
        gpu.probe_gpu_inventory(requested="3")


def test_unmapped_gpu_process_is_not_silently_ignored(monkeypatch):
    inventory(monkeypatch, "MIG-unknown, 111, /env/bin/python\n")
    with pytest.raises(RuntimeError, match="unmapped"):
        gpu.probe_gpu_inventory()


def test_framework_and_growth_guards_precede_probe(monkeypatch):
    monkeypatch.setitem(sys.modules, "tensorflow", SimpleNamespace())
    with pytest.raises(RuntimeError, match="precede"):
        gpu.select_worker_gpu()
    monkeypatch.delitem(sys.modules, "tensorflow")
    monkeypatch.delenv("TF_FORCE_GPU_ALLOW_GROWTH", raising=False)
    with pytest.raises(ValueError, match="before import"):
        gpu.select_worker_gpu()


def test_busy_training_worker_preserves_receipt_without_import(tmp_path, monkeypatch):
    from bayesfilter.inference import q20_master_stages as stages
    receipt = {"status": "FAILED", "error": "no_gpu_without_numerical_workload"}
    def busy(**kwargs):
        raise gpu.GPUResourceUnavailable(receipt)
    monkeypatch.setattr(stages, "select_worker_gpu", busy)
    monkeypatch.setenv("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name == "tensorflow":
            pytest.fail("resource wait reached TensorFlow import")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    output = tmp_path / "worker"
    result = stages.run_worker({"stage": "train", "config": protocol_template()}, output)
    assert result["status"] == "waiting_for_gpu"
    assert not result["completed"]
    assert not (output / "data").exists()
    assert json.loads((output / "manifest.json").read_text())["failure"]["resource_receipt"] == receipt


def fake_worker(monkeypatch, payloads, *, code=0, diagnostic=False):
    """Keep the real supervisor, replacing only its numerical child command."""
    original = Campaign.execute
    def execute(self, stage, command, **kwargs):
        payload = payloads.pop(0)
        filename = "result.json" if diagnostic else "worker-result.json"
        source = ("import json,pathlib,sys; p=pathlib.Path(sys.argv[1]); p.mkdir(); "
                  f"(p/{filename!r}).write_text({json.dumps(payload)!r}); sys.exit({code})")
        return original(self, stage, [sys.executable, "-c", source, "{attempt}/worker"], **kwargs)
    monkeypatch.setattr(Campaign, "execute", execute)


def test_nonzero_worker_failure_is_not_a_training_checkpoint(tmp_path, monkeypatch):
    c = campaign(tmp_path)
    fake_worker(monkeypatch, [{"completed": False, "status": "worker_failure",
                              "message": "original initialization error"}], code=1)
    with c.locked():
        result = c.numerical_stage("train", {"stage": "train"}, cap_seconds=2., diagnostic=False, gpu=1)
        assert result["status"] == "worker_failure"
        assert result["message"] == "original initialization error"
        assert not result["completed"] and not c.state["stages"]
        assert c.state["spent_seconds"] > 0


def test_checkpoint_survives_an_intervening_capacity_wait(tmp_path, monkeypatch):
    c = campaign(tmp_path)
    request = {"stage": "train"}
    data = tmp_path / "earlier/worker/data"
    data.mkdir(parents=True)
    checkpoint = data / "cohort-00001.json"
    checkpoint.write_text('{}')
    c.state["attempts"] = [{"stage": "train", "status": "completed", "elapsed_seconds": 0.,
        "directory": str(tmp_path / name), "request_hash": digest(request)} for name in ("earlier", "wait")]
    fake_worker(monkeypatch, [{"completed": False, "status": "waiting_for_gpu"}])
    with c.locked():
        result = c.numerical_stage("train", request, cap_seconds=2., diagnostic=False, gpu=1)
        assert result["status"] == "waiting_for_gpu"
        attempt = c.state["attempts"][-1]
        saved = json.loads((Path(attempt["directory"]) / "request.json").read_text())
        assert saved["resume_checkpoint"] == str(checkpoint)
        assert attempt["failure_classification"] == "resource_unavailable"
        assert not c.state["stages"]


def test_master_diagnostic_wait_resume_accounting_and_cached_completion(tmp_path, monkeypatch):
    c = campaign(tmp_path)
    c.config["execution"].update(stage_attempts=1, diagnostic_attempts=1, diagnostic_attempt_seconds=2.)
    kwargs = dict(repo=c.repo, allowance={"campaign_remaining_seconds":30.,
                  "diagnostic_remaining_seconds":20.}, stop_after="diagnose")
    payloads = [{"status": "waiting_for_gpu", "wall_seconds": .01},
                {"status": "waiting_for_gpu", "wall_seconds": .01},
                {"status": "completed", "wall_seconds": .01,
                 "paired_health_and_transition_checks": "passed"}]
    fake_worker(monkeypatch, payloads, diagnostic=True)
    for _ in range(2):
        result = execute_master(c.config, tmp_path, **kwargs)
        assert result["status"] == "WAITING_FOR_GPU"
    result = execute_master(c.config, tmp_path, **kwargs)
    assert result["status"] == "STATUS_REUSE_DIAGNOSTIC_COMPLETE"
    state = json.loads((tmp_path / "campaign.json").read_text())
    assert len(state["attempts"]) == 3 and set(state["stages"]) == {"status-reuse"}
    assert state["spent_seconds"] == state["diagnostic_spent_seconds"] > 0
    assert execute_master(c.config, tmp_path, **kwargs) == result
    assert json.loads((tmp_path / "campaign.json").read_text())["attempts"] == state["attempts"]
