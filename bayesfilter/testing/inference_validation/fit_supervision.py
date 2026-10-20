"""Bounded, framework-free supervisor for one complete numerical fit."""
from dataclasses import replace
import math
import os
from pathlib import Path
import subprocess
import time

from .storage import read_json, write_json
from .timeout_policy import (
    FitAllowance, ObservedContention, ProgressObserver, TimeoutPolicy, default_workload_probe, progress_changed,
)


def supervise_fit(command, log, seconds, device, *, progress_root=None, cell_deadline=None,
                  timeout_policy=None, workload_probe=None, allowance_path=None,
                  budget_receipt_path=None, cwd=None):
    policy = timeout_policy or TimeoutPolicy()
    if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("fit allowance must be finite and positive")
    if cell_deadline is not None and not math.isfinite(cell_deadline):
        raise ValueError("cell deadline must be finite")
    started = time.monotonic()
    base_end = started + seconds
    absolute_end = base_end + policy.max_extension_seconds
    if cell_deadline is not None:
        absolute_end = min(absolute_end, cell_deadline)
    current_end = min(base_end, absolute_end)
    # Reserve shutdown/serialization time inside the cap. Tiny process tests
    # use the same proportional reserve; no scientific work count changes.
    reserve = min(policy.shutdown_grace_seconds, max(0., seconds / 10))
    observer = ProgressObserver(progress_root) if progress_root is not None else None
    previous = {"signature": ()}
    initial = dict(previous)
    last_progress = None
    progress_events = 0
    extension = 0.
    workload_samples = []
    decisions = []
    contention = ObservedContention(policy)
    probe = workload_probe or default_workload_probe
    allowance = FitAllowance(allowance_path) if allowance_path is not None else None
    def publish(reason):
        if allowance is not None:
            allowance.publish(deadline=current_end - reserve, hard_deadline=current_end, reason=reason)
    env = dict(os.environ, TF_FORCE_GPU_ALLOW_GROWTH="true", BAYESFILTER_PRELOAD_CUSTOM_OP="0")
    if device == "cpu_reference":
        env["CUDA_VISIBLE_DEVICES"] = "-1"
    next_probe = started
    extension_decided = False
    status, reason, code = "failed", None, None
    process = None
    with Path(log).open("w") as handle:
        try:
            if observer is not None:
                previous = observer.snapshot()
                initial = dict(previous)
            if time.monotonic() >= current_end - reserve:
                status, reason = "budget_exhausted", "no_remaining_launch_allowance"
            else:
                publish("base_allocation")
                process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, env=env, cwd=cwd)
            while process is not None:
                now = time.monotonic()
                code = process.poll()
                if code is not None:
                    if observer is not None:
                        observed = observer.snapshot()
                        if progress_changed(previous, observed):
                            last_progress, progress_events = now, progress_events + 1
                            previous = observed
                    status = "complete" if code == 0 else "failed"
                    if code == 75 and budget_receipt_path is not None and Path(budget_receipt_path).is_file():
                        receipt = read_json(budget_receipt_path)
                        if receipt.get("status") == "budget_exhausted":
                            status, reason = "budget_exhausted", "cooperative_budget_stop"
                    break
                if now >= current_end:
                    status, reason = "timed_out", ("outer_cell_deadline" if now >= absolute_end and
                        cell_deadline is not None and absolute_end == cell_deadline else "fit_budget_exhausted")
                    break
                if observer is not None:
                    observed = observer.snapshot()
                    if progress_changed(previous, observed):
                        last_progress, progress_events = now, progress_events + 1
                        previous = observed
                # Capture telemetry throughout execution, including before a
                # slowdown. Probe latency is bounded and charged to wall time.
                margin = reserve + 2*policy.workload_probe_timeout_seconds + policy.poll_interval_seconds
                decision_due = (policy.extension_mode == "latest_sample" and
                                not extension_decided and now >= base_end - margin)
                recent_sample = workload_samples and now-started-workload_samples[-1]["elapsed_seconds"] <= (
                    2*policy.workload_probe_timeout_seconds + policy.poll_interval_seconds)
                if now >= next_probe or decision_due and not recent_sample:
                    try:
                        probe_policy = replace(policy, workload_probe_timeout_seconds=min(
                            policy.workload_probe_timeout_seconds, max(.001, (current_end-now)/2)))
                        workload = dict(probe(device, process.pid, probe_policy))
                    except Exception as exc:
                        workload = {"trusted_for_extension": False, "contended": False,
                                    "reason": type(exc).__name__ + ": " + str(exc)}
                    workload_samples.append({"elapsed_seconds": now-started, **workload})
                    contention.observe(now-started, workload)
                    next_probe = time.monotonic() + policy.telemetry_interval_seconds
                now = time.monotonic()
                if policy.extension_mode == "observed_intervals":
                    recent = last_progress is not None and now-last_progress <= policy.progress_grace_seconds
                    earned = min(contention.seconds, policy.max_extension_seconds, absolute_end-base_end)
                    if recent and earned > extension and now < current_end-reserve:
                        extension = earned
                        current_end = min(absolute_end, base_end+extension)
                        publish("observed_contention_intervals_with_progress")
                        decisions.append({"elapsed_seconds": now-started, "recent_progress": True,
                            "observed_contention_seconds": contention.seconds,
                            "delivered_before_cooperative_stop": True,
                            "extension_granted_seconds": extension})
                # Decide before the child reaches its cooperative boundary,
                # leaving the configured probe time and one poll for delivery.
                if policy.extension_mode == "latest_sample" and not extension_decided and now >= base_end - margin:
                    fresh = workload_samples and now - started - workload_samples[-1]["elapsed_seconds"] <= (
                        policy.telemetry_interval_seconds + 2*policy.workload_probe_timeout_seconds)
                    recent = last_progress is not None and now-last_progress <= policy.progress_grace_seconds
                    eligible = (recent and fresh and workload_samples[-1].get("trusted_for_extension") is True
                                and workload_samples[-1].get("contended") is True)
                    in_time = now < current_end - reserve
                    if eligible and in_time and policy.max_extension_seconds > 0 and absolute_end > current_end:
                        extension = absolute_end-current_end
                        current_end = absolute_end
                        publish("progress_and_contention_extension")
                    if extension or not in_time:
                        extension_decided = True
                        decisions.append({"elapsed_seconds": now-started, "recent_progress": recent,
                            "fresh_telemetry": bool(fresh), "delivered_before_cooperative_stop": in_time,
                            "extension_granted_seconds": extension})
                age = now - (started if last_progress is None else last_progress)
                if policy.stall_timeout_seconds is not None and age >= policy.stall_timeout_seconds:
                    status, reason = "stalled", "no_durable_progress"
                    break
                try:
                    process.wait(timeout=max(0., min(policy.poll_interval_seconds, current_end-now)))
                except subprocess.TimeoutExpired:
                    pass
        except Exception as exc:
            status, reason = "supervision_failed", type(exc).__name__ + ": " + str(exc)
        finally:
            if process is not None and process.poll() is None:
                process.terminate()
                # Cleanup consumes remaining wall allowance. At the hard cap
                # there is no extra five-second permission to keep computing.
                grace = max(0., min(policy.shutdown_grace_seconds, current_end-time.monotonic()))
                try:
                    process.wait(timeout=grace)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            if process is not None:
                code = process.returncode
    elapsed = time.monotonic()-started
    record = {"status": status, "exit_code": code, "pid": None if process is None else process.pid,
        "elapsed_seconds": elapsed, "command": command, "log": str(log),
        "base_budget_seconds": seconds, "extension_granted_seconds": extension,
        "extension_consumed_seconds": max(0., elapsed-seconds),
        "maximum_allocation_seconds": absolute_end-started,
        "final_hard_allocation_seconds": current_end-started,
        "deadline_overrun_seconds": max(0., elapsed-(current_end-started)),
        "cooperative_reserve_seconds": reserve, "timeout_policy": policy.payload(),
        "progress": {"events": progress_events, "initial": {k:v for k,v in initial.items() if k!="signature"},
                     "last": {k:v for k,v in previous.items() if k!="signature"},
                     "last_progress_age_seconds": None if last_progress is None else time.monotonic()-last_progress},
        "workload_samples": workload_samples, "extension_decisions": decisions}
    record["observed_contention_seconds"] = contention.seconds
    record["contention_allowance_interpretation"] = "bounded scheduling allowance; not measured lost compute"
    if reason is not None:
        record["timeout_reason"] = reason
    # Persist independently of the coordinator so a later reporting exception
    # cannot lose the terminal resource/progress evidence.
    write_json(str(log).removesuffix(".log") + "-supervision.json", record)
    return record
