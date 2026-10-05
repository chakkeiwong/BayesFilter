"""Execution mechanics only: no timing threshold is a sampler criterion."""
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from bayesfilter.testing.inference_validation import fit_supervision as supervision
from bayesfilter.testing.inference_validation.storage import write_json, read_json
from bayesfilter.testing.inference_validation.timeout_policy import (
    FitAllowance, ProgressObserver, ProgressEvidenceError, TimeoutPolicy,
    machine_workload_snapshot, progress_changed,
)
from bayesfilter.runtime.execution_budget import execution_budget, execution_budget_available


@pytest.mark.parametrize("overrides", [
    {"poll_interval_seconds": False}, {"stall_timeout_seconds": -1},
    {"max_extension_seconds": float("inf")}, {"progress_grace_seconds": 0},
    {"normalized_load_threshold": float("nan")}, {"shutdown_grace_seconds": -1},
    {"telemetry_interval_seconds": True}, {"workload_probe_timeout_seconds": 0},
    {"extension_mode": "unknown"}, {"gpu_admission_wait_seconds": float("nan")},
    {"gpu_admission_wait_seconds": -1}, {"gpu_admission_wait_seconds": True},
    {"gpu_admission_mode":"untrusted"}, {"max_contention_retries":True},
    {"max_contention_retries":-1},
])
def test_policy_rejects_invalid_bounds(overrides):
    with pytest.raises(ValueError):
        TimeoutPolicy(**overrides)


def test_policy_requires_isolation_and_total_allocation_fits_design(design):
    with pytest.raises(ValueError, match="requires isolated"):
        design(options={"timeout_policy": {}})
    with pytest.raises(ValueError, match="extension"):
        design("search", "gaussian", "prepared", budget_seconds=10,
               options={"isolate_fits": True, "fit_process_timeout_seconds": 9,
                        "timeout_policy": {"max_extension_seconds": 2}})
    with pytest.raises(ValueError, match="unknown timeout_policy"):
        TimeoutPolicy.from_options({"timeout_policy": {"made_up": 1}})


def checkpoint(root, hashes, elapsed=0):
    payload = {"schema": "bayesfilter.hmc_numerical_tuning_checkpoint.v1",
        "numerical_evidence_hashes": hashes, "partial_chunks": {},
        "result": {"search_state": {"elapsed_seconds": elapsed}}}
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    write_json(root / "tuning/tuning_checkpoint.json", {**payload, "content_hash": hashlib.sha256(raw).hexdigest()})


def test_progress_ignores_heartbeat_mtime_and_clock_only_checkpoint(tmp_path):
    observer = ProgressObserver(tmp_path)
    first = observer.snapshot()
    write_json(tmp_path / "heartbeat.json", {"count": 1})
    write_json(tmp_path / "tuning/preparation_progress.json", {
        "schema": "bayesfilter.hmc_preparation_progress.v1", "events": [{"phase": "probe_started"}]})
    assert not progress_changed(first, observer.snapshot())
    checkpoint(tmp_path, ["evidence-1"])
    second = observer.snapshot()
    assert progress_changed(first, second)
    checkpoint(tmp_path, ["evidence-1"], elapsed=10)
    assert not progress_changed(second, observer.snapshot())
    path = tmp_path / "tuning/tuning_checkpoint.json"
    os.utime(path, None)
    assert not progress_changed(second, observer.snapshot())
    checkpoint(tmp_path, ["evidence-1", "evidence-2"])
    assert progress_changed(second, observer.snapshot())
    path.write_text('{"schema":"broken"}')
    with pytest.raises(ProgressEvidenceError):
        observer.snapshot()


def test_posterior_chunk_commit_is_progress(tmp_path):
    observer = ProgressObserver(tmp_path)
    first = observer.snapshot()
    path = tmp_path / "members/candidate/posterior_chunks/committed/warmup-0001/bundle.sha256"
    path.parent.mkdir(parents=True)
    content = json.dumps({"schema": "bayesfilter.durable_tensor_checkpoint.v1"}).encode()
    path.with_name("bundle.json").write_bytes(content)
    path.write_text(hashlib.sha256(content).hexdigest())
    assert progress_changed(first, observer.snapshot())
    path.with_name("bundle.json").write_text('{}')
    with pytest.raises(ProgressEvidenceError, match="posterior"):
        observer.snapshot()


def test_repeated_completion_callback_is_not_new_progress(tmp_path):
    path = tmp_path / "tuning/preparation_progress.json"
    event = {"phase": "bootstrap_chunk_complete", "details": {"offset": 0, "count": 4}}
    write_json(path, {"schema": "bayesfilter.hmc_preparation_progress.v1", "events": [event]})
    observer = ProgressObserver(tmp_path)
    first = observer.snapshot()
    write_json(path, {"schema": "bayesfilter.hmc_preparation_progress.v1", "events": [event, event]})
    assert not progress_changed(first, observer.snapshot())


def test_execution_scopes_intersect_and_reset():
    assert execution_budget_available()
    with execution_budget(check=lambda: False):
        with execution_budget(deadline=time.monotonic()+100):
            assert not execution_budget_available()
    assert execution_budget_available()
    with pytest.raises(ValueError):
        with execution_budget(deadline=float("nan")):
            pass


def test_child_observes_extended_allowance_and_malformed_file_fails(tmp_path):
    allowance = FitAllowance(tmp_path / "allowance.json")
    now = time.monotonic()
    allowance.publish(deadline=now-1, hard_deadline=now+1, reason="base")
    with execution_budget(check=allowance.available):
        assert not execution_budget_available()
        allowance.publish(deadline=now+1, hard_deadline=now+2, reason="extension")
        assert execution_budget_available()
        allowance.publish(deadline=now+3, hard_deadline=now+2, reason="bad")
        with pytest.raises(ValueError):
            execution_budget_available()


def fake_supervisor(monkeypatch, *, progress=True, workload=None, finish=100, probe_delay=0):
    clock = SimpleNamespace(now=0.)
    monkeypatch.setattr(supervision, "time", SimpleNamespace(monotonic=lambda: clock.now))
    class Process:
        pid = 8123
        returncode = None
        def poll(self):
            if clock.now >= finish and self.returncode is None:
                self.returncode = 0
            return self.returncode
        def wait(self, timeout=None):
            if self.poll() is not None:
                return self.returncode
            clock.now += min(timeout, max(0., finish-clock.now))
            if self.poll() is None:
                raise subprocess.TimeoutExpired("test", timeout)
            return self.returncode
        def terminate(self): self.returncode = -15
        def kill(self): self.returncode = -9
    monkeypatch.setattr(supervision.subprocess, "Popen", lambda *a, **k: Process())
    class Observer:
        def __init__(self, root): pass
        def snapshot(self):
            return {"signature": (int(clock.now) if progress else 0,)}
    monkeypatch.setattr(supervision, "ProgressObserver", Observer)
    def probe(*args):
        clock.now += probe_delay
        return workload or {"contended": True, "trusted_for_extension": True}
    return clock, probe


@pytest.mark.parametrize("progress,load,expected", [
    (True, {"contended": True, "trusted_for_extension": True}, 10.),
    (False, {"contended": True, "trusted_for_extension": True}, 0.),
    (True, {"contended": False, "trusted_for_extension": True}, 0.),
    (True, {"contended": True, "trusted_for_extension": False}, 0.),
])
def test_extension_needs_progress_and_trusted_contention(tmp_path, monkeypatch, progress, load, expected):
    clock, probe = fake_supervisor(monkeypatch, progress=progress, workload=load)
    policy = TimeoutPolicy(max_extension_seconds=10, shutdown_grace_seconds=1, telemetry_interval_seconds=1)
    record = supervision.supervise_fit(["fake"], tmp_path/"worker.log", 10, "cpu_reference",
        progress_root=tmp_path, cell_deadline=50, timeout_policy=policy, workload_probe=probe,
        allowance_path=tmp_path/"allowance.json")
    assert record["extension_granted_seconds"] == expected
    assert record["elapsed_seconds"] == 10+expected
    assert record["status"] == "timed_out"
    assert read_json(tmp_path/"allowance.json")["hard_deadline"] == 10+expected


def test_extension_cannot_cross_outer_deadline(tmp_path, monkeypatch):
    clock, probe = fake_supervisor(monkeypatch)
    record = supervision.supervise_fit(["fake"], tmp_path/"worker.log", 10, "cpu_reference",
        progress_root=tmp_path, cell_deadline=12,
        timeout_policy=TimeoutPolicy(max_extension_seconds=100, shutdown_grace_seconds=1), workload_probe=probe)
    assert record["extension_granted_seconds"] == 2
    assert record["elapsed_seconds"] == 12
    assert record["timeout_reason"] == "outer_cell_deadline"
    assert record["deadline_overrun_seconds"] == 0


def test_stall_stops_even_with_high_load(tmp_path, monkeypatch):
    clock, probe = fake_supervisor(monkeypatch, progress=False)
    record = supervision.supervise_fit(["fake"], tmp_path/"worker.log", 100, "cpu_reference",
        progress_root=tmp_path, cell_deadline=200,
        timeout_policy=TimeoutPolicy(stall_timeout_seconds=5, max_extension_seconds=100), workload_probe=probe)
    assert record["status"] == "stalled"
    assert record["elapsed_seconds"] == 5
    assert record["extension_granted_seconds"] == 0


def test_gpu_load_filters_device_and_own_process(tmp_path, monkeypatch):
    from bayesfilter.testing.inference_validation import timeout_policy as module
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-selected")
    def command(args, **kwargs):
        output = ("GPU-selected, 99, 100, 1000\n" if "--query-gpu=" in args[1] else
                  "123, GPU-selected\n456, GPU-other\n789, GPU-selected\n")
        return SimpleNamespace(stdout=output)
    monkeypatch.setattr(module.subprocess, "run", command)
    monkeypatch.setattr(module.os, "getpgid", lambda pid: 10 if pid==123 else 20)
    record = machine_workload_snapshot(device="gpu", child_pid=123, policy=TimeoutPolicy())
    assert record["gpu"]["foreign_compute_pids"] == [789]
    assert record["contended"] and record["trusted_for_extension"]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    record = machine_workload_snapshot(device="gpu", child_pid=123, policy=TimeoutPolicy())
    assert not record["trusted_for_extension"] and not record["contended"]


def test_real_child_sees_grace_before_cooperative_deadline(tmp_path):
    """Real subprocess/control-file coordination; synthetic progress only."""
    code = '''import json, pathlib, sys, time
root = pathlib.Path(sys.argv[1]); started = time.monotonic(); events=[]
(root/'tuning').mkdir(exist_ok=True)
while time.monotonic()-started < .65:
    events.append({'phase':'probe_completed', 'details':{'offset':len(events)}})
    path=root/'tuning/preparation_progress.json'; temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps({'schema':'bayesfilter.hmc_preparation_progress.v1','events':events}))
    temporary.replace(path)
    allowance=json.loads((root/'allowance.json').read_text())
    if time.monotonic()>=allowance['deadline']: raise SystemExit(9)
    time.sleep(.02)
'''
    record = supervision.supervise_fit([sys.executable, "-c", code, str(tmp_path)], tmp_path/"worker.log",
        .45, "cpu_reference", progress_root=tmp_path, cell_deadline=time.monotonic()+3,
        timeout_policy=TimeoutPolicy(poll_interval_seconds=.02, max_extension_seconds=1.,
            shutdown_grace_seconds=.03, workload_probe_timeout_seconds=.01, telemetry_interval_seconds=.1),
        workload_probe=lambda *args: {"contended": True, "trusted_for_extension": True},
        allowance_path=tmp_path/"allowance.json")
    assert record["status"] == "complete", record
    assert record["extension_granted_seconds"] > 0 and record["progress"]["events"] > 0


def test_corrupt_checkpoint_stops_before_launch(tmp_path, monkeypatch):
    write_json(tmp_path / "tuning/tuning_checkpoint.json", {"schema": "broken"})
    monkeypatch.setattr(supervision.subprocess, "Popen", lambda *a, **k: pytest.fail("invalid progress launched"))
    record = supervision.supervise_fit(["fake"], tmp_path / "worker.log", 10, "cpu_reference",
                                      progress_root=tmp_path)
    assert record["status"] == "supervision_failed"
    assert record["pid"] is None
    assert "ProgressEvidenceError" in record["timeout_reason"]
    assert (tmp_path / "worker-supervision.json").is_file()


def test_cooperative_exit_requires_explicit_receipt(tmp_path):
    code = "import json,sys; open(sys.argv[1],'w').write(json.dumps({'status':'budget_exhausted'})); sys.exit(75)"
    receipt = tmp_path / "budget.json"
    record = supervision.supervise_fit([sys.executable, "-c", code, str(receipt)],
        tmp_path / "worker.log", 10, "cpu_reference", budget_receipt_path=receipt)
    assert record["status"] == "budget_exhausted"
    without = supervision.supervise_fit([sys.executable, "-c", "raise SystemExit(75)"],
        tmp_path / "missing.log", 10, "cpu_reference", budget_receipt_path=tmp_path / "missing.json")
    assert without["status"] == "failed"


def test_slow_probe_cannot_grant_extension_after_child_deadline(tmp_path, monkeypatch):
    clock, probe = fake_supervisor(monkeypatch)
    def delayed(*args):
        clock.now += 10 if clock.now >= 6 else 0
        return {"contended": True, "trusted_for_extension": True}
    record = supervision.supervise_fit(["fake"], tmp_path / "worker.log", 10, "cpu_reference",
        progress_root=tmp_path, timeout_policy=TimeoutPolicy(max_extension_seconds=10,
            telemetry_interval_seconds=1, shutdown_grace_seconds=1), workload_probe=delayed)
    assert record["extension_granted_seconds"] == 0


def test_cell_deadline_uses_parent_clock_without_reset(monkeypatch):
    from bayesfilter.testing.inference_validation.execution import cell_execution_deadline
    monkeypatch.setenv("BAYESFILTER_VALIDATION_CELL_DEADLINE", "100")
    assert cell_execution_deadline(90, 30) == 100
    assert cell_execution_deadline(50, 30) == 80
    monkeypatch.setenv("BAYESFILTER_VALIDATION_CELL_DEADLINE", "nan")
    with pytest.raises(ValueError):
        cell_execution_deadline(50, 30)


def test_earlier_contention_still_earns_time_after_other_job_leaves(tmp_path, monkeypatch):
    clock, _ = fake_supervisor(monkeypatch, finish=12)
    def probe(*args):
        return {"contended": clock.now < 4, "trusted_for_extension": True}
    record = supervision.supervise_fit(["fake"], tmp_path/"worker.log", 10, "cpu_reference",
        progress_root=tmp_path, cell_deadline=30,
        timeout_policy=TimeoutPolicy(extension_mode="observed_intervals", max_extension_seconds=10,
            telemetry_interval_seconds=1, shutdown_grace_seconds=1),
        workload_probe=probe, allowance_path=tmp_path/"allowance.json")
    assert record["status"] == "complete"
    assert record["extension_granted_seconds"] == record["observed_contention_seconds"] == 3
    assert record["extension_decisions"][0]["elapsed_seconds"] < 4
    assert not record["workload_samples"][-1]["contended"]


@pytest.mark.parametrize("progress,trusted", [(False, True), (True, False)])
def test_accumulated_extension_needs_trusted_load_and_progress(tmp_path, monkeypatch, progress, trusted):
    clock, probe = fake_supervisor(monkeypatch, progress=progress,
        workload={"contended": True, "trusted_for_extension": trusted})
    record = supervision.supervise_fit(["fake"], tmp_path/"worker.log", 10, "cpu_reference",
        progress_root=tmp_path, timeout_policy=TimeoutPolicy(extension_mode="observed_intervals",
            max_extension_seconds=10, telemetry_interval_seconds=1), workload_probe=probe)
    assert record["extension_granted_seconds"] == 0
    assert record["elapsed_seconds"] == 10


def test_observed_intervals_cannot_credit_unobserved_gaps_or_double_count():
    from bayesfilter.testing.inference_validation.timeout_policy import ObservedContention
    observed = ObservedContention(TimeoutPolicy(telemetry_interval_seconds=1, workload_probe_timeout_seconds=.1))
    busy = {"trusted_for_extension": True, "contended": True}
    assert observed.observe(0, busy) == 0
    assert observed.observe(100, busy) == pytest.approx(1.2)
    assert observed.observe(100, busy) == pytest.approx(1.2)
    assert observed.observe(101, {"trusted_for_extension": False, "contended": True}) == pytest.approx(1.2)
    assert observed.observe(102, busy) == pytest.approx(1.2)
    with pytest.raises(ValueError, match="backwards"):
        observed.observe(99, busy)


def test_observed_intervals_remain_bounded_by_outer_deadline(tmp_path, monkeypatch):
    clock, probe = fake_supervisor(monkeypatch)
    record = supervision.supervise_fit(["fake"], tmp_path/"worker.log", 10, "cpu_reference",
        progress_root=tmp_path, cell_deadline=12,
        timeout_policy=TimeoutPolicy(extension_mode="observed_intervals", max_extension_seconds=100,
            shutdown_grace_seconds=1, telemetry_interval_seconds=1), workload_probe=probe)
    assert record["extension_granted_seconds"] == 2
    assert record["elapsed_seconds"] == 12
    assert record["timeout_reason"] == "outer_cell_deadline"


@pytest.mark.parametrize("trusted,clears,admitted", [(True, True, True), (True, False, False), (False, True, False)])
def test_gpu_admission_waits_without_starting_a_worker(monkeypatch, trusted, clears, admitted):
    from bayesfilter.testing.inference_validation import timeout_policy as module
    clock = SimpleNamespace(now=0.)
    monkeypatch.setattr(module.time, "monotonic", lambda: clock.now)
    monkeypatch.setattr(module.time, "sleep", lambda seconds: setattr(clock, "now", clock.now+seconds))
    monkeypatch.setattr(module.subprocess, "Popen", lambda *a, **k: pytest.fail("admission starts no process"))
    result = module.wait_for_gpu_admission(device="gpu", deadline=10,
        policy=TimeoutPolicy(gpu_admission_wait_seconds=5, telemetry_interval_seconds=1),
        workload_probe=lambda *a: {"trusted_for_extension": trusted, "contended": not clears or clock.now < 2})
    assert result["admitted"] is admitted
    assert result["wait_seconds"] == (2 if admitted else 5)


def test_gpu_admission_cannot_cross_outer_deadline(monkeypatch):
    from bayesfilter.testing.inference_validation import timeout_policy as module
    clock = SimpleNamespace(now=0.)
    monkeypatch.setattr(module.time, "monotonic", lambda: clock.now)
    monkeypatch.setattr(module.time, "sleep", lambda seconds: setattr(clock, "now", clock.now+seconds))
    result = module.wait_for_gpu_admission(device="gpu", deadline=2,
        policy=TimeoutPolicy(gpu_admission_wait_seconds=10, telemetry_interval_seconds=1),
        workload_probe=lambda *a: {"trusted_for_extension": True, "contended": True})
    assert not result["admitted"] and result["wait_seconds"] == 2
