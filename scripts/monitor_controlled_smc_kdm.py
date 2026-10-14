#!/usr/bin/env python3
"""Host-only watchdog for the authorized replication batch; no numerical retries.

CPU-only inspection deliberately hides CUDA. This imports no numerical framework.
Notifications are local Windows desktop submissions, not chat messages.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import argparse
import base64
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from xml.sax.saxutils import escape

import run_controlled_smc_kdm as runner

ROOT = runner.ROOT
CAMPAIGN = runner.CAMPAIGN
PLAN = "docs/plans/controlled-smc-kdm-monitoring-20261005.md"
QUEUES = ("c1_source_failure_replay", "p0_forward_capture", "p0_forward_compare")
PHASES = ("c1_source_full",) + QUEUES
POWERSHELL = "/mnt/c/WINDOWS/System32/WindowsPowerShell/v1.0/powershell.exe"
INTERVAL = 60
STALL_SECONDS = 1200
MAX_REPAIRS = 2
KNOWN_FAILURES = {
    ("lorenz_csmc", "SOURCE_OR_RUNTIME_FAILURE", 3537),
    ("lorenz_apf", "BLOCKED_CSMC_ITERATIONS", 0),
}


def live(pid, identity):
    if not runner.shared.alive(pid, identity):
        return False
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] not in ("Z", "X")
    except OSError:
        return False


def notify(title, body):
    """Submit a bounded local toast without shell interpolation or network access."""
    xml = ("<toast><visual><binding template=\"ToastGeneric\"><text>" + escape(title)
           + "</text><text>" + escape(body[:700]) + "</text></binding></visual></toast>")
    payload = base64.b64encode(xml.encode()).decode()
    script = "\n".join([
        "$ErrorActionPreference = 'Stop'",
        "[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType=WindowsRuntime] | Out-Null",
        "[Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom.XmlDocument, ContentType=WindowsRuntime] | Out-Null",
        "$xml = New-Object Windows.Data.Xml.Dom.XmlDocument",
        "$xml.LoadXml([Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('" + payload + "')))",
        "$toast = [Windows.UI.Notifications.ToastNotification]::new($xml)",
        "$toast.ExpirationTime = [DateTimeOffset]::Now.AddHours(8)",
        "$notifier = [Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier('Microsoft.Windows.PowerShell')",
        "$notifier.Show($toast)",
        "Write-Output ('SUBMITTED; setting=' + $notifier.Setting)",
    ])
    try:
        result = subprocess.run([POWERSHELL, "-NoLogo", "-NoProfile", "-NonInteractive",
                                 "-EncodedCommand", base64.b64encode(script.encode("utf-16le")).decode()],
                                capture_output=True, timeout=20, check=False)
        output = result.stdout.decode(errors="replace").strip()
        error = result.stderr.decode(errors="replace").strip()
        return dict(submitted=result.returncode == 0 and "SUBMITTED" in output,
                    notifications_enabled="setting=Enabled" in output,
                    returncode=result.returncode, output=output[:500], error=error[-500:],
                    delivery_or_read_receipt=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return dict(submitted=False, error=str(exc)[:500], delivery_or_read_receipt=False)


def sample():
    progress = runner.progress()
    attempts = {}
    for path, m in runner.campaign_attempts():
        if m["phase"] not in PHASES or int(path.parent.name.split("_")[0][7:]) < 145:
            continue
        attempts[m["phase"]] = {key: m.get(key) for key in (
            "phase", "status", "decision", "returncode", "started_epoch", "timeout_seconds",
            "output_root", "resource")}
        attempts[m["phase"]]["decision"] = runner.reported_decision(m)
        attempts[m["phase"]].update(manifest=str(path),
            worker_alive=live(m.get("pid"), m.get("process_identity")),
            supervisor_alive=live(m.get("supervisor_pid"), m.get("supervisor_identity")))
    queues = {}
    for phase in QUEUES:
        path = CAMPAIGN / f"queue-{phase}.json"
        q = json.loads(path.read_text()) if path.exists() else {"status": "NOT_QUEUED"}
        queues[phase] = {key: q.get(key) for key in ("status", "deadline_epoch", "error", "pid")}
        queues[phase]["supervisor_alive"] = live(q.get("pid"), q.get("process_identity"))
    memory = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith(("MemTotal:", "MemAvailable:")):
            key, value, _ = line.split()
            memory[key.rstrip(":")] = int(value) * 1024
    # Include both ledgers: the original GPU grant is not the active tranche.
    current = runner.status()
    return dict(at=time.time(), utc=runner.shared.stamp(), progress=progress,
                attempts=attempts, queues=queues, memory_bytes=memory,
                current_allocation_remaining_seconds=current.get("tranche_remaining_seconds"))


def truth(value):
    return value is True or str(value).lower() == "true"


def source_partial_expected(snapshot):
    actual = {(x["driver"], x["status"], x.get("completed_cases"))
              for x in snapshot["progress"].get("source_driver_failures", [])}
    drivers = {x["driver"]: x for x in snapshot["progress"].get("drivers", [])}
    return actual == KNOWN_FAILURES and all(
        key in drivers and drivers[key]["completed_cases"] == 2000
        and drivers[key]["nonfinite_cases"] == 0 and drivers[key]["partial_or_invalid_rows"] == 0
        for key in ("neuroscience_csmc", "neuroscience_bpf"))


def evaluate(snapshot, state, config):
    """Pure decision step: notifications and narrowly eligible queue repairs."""
    now = snapshot["at"]
    events, repairs, conditions = [], [], {}
    counters = state.setdefault("counters", {})
    attempts, queues = snapshot["attempts"], snapshot["queues"]
    for phase, m in attempts.items():
        if m["status"] == "RUNNING":
            grace_over = now > m["started_epoch"] + 120
            conditions[f"worker_missing:{phase}"] = grace_over and not m["worker_alive"]
            conditions[f"supervisor_missing:{phase}"] = grace_over and not m["supervisor_alive"]
            conditions[f"deadline_exceeded:{phase}"] = now > m["started_epoch"] + m["timeout_seconds"] + 30
        else:
            decision = m.get("decision") or {}
            severity = "INFO" if m["status"] == "COMPLETE" and decision.get("status") == "PASS" else "ERROR"
            detail = f"{phase}: {m['status']}; decision={decision}"
            if phase == "c1_source_full" and m.get("returncode") == 1 and source_partial_expected(snapshot):
                severity = "INFO"
                detail = "Source batch finished: neuroscience complete; known Lorenz failure and blocked APF retained. Replication incomplete."
            if phase == "p0_forward_compare" and not truth(decision.get("forward_screen_passed")):
                severity = "ERROR"
                detail += "; forward comparison requires investigation; diagnostic completion is not acceptance"
            events.append((f"terminal:{phase}:{m['manifest']}", severity, detail))
    for fail in snapshot["progress"].get("source_driver_failures", []):
        key = (fail["driver"], fail["status"], fail.get("completed_cases"))
        if key not in KNOWN_FAILURES:
            events.append(("source_failure:" + repr(key), "ERROR", "New source failure: " + repr(key)))
    drivers = snapshot["progress"].get("drivers", [])
    count = sum(d["completed_cases"] for d in drivers)
    if state.get("completed_cases") != count:
        state.update(completed_cases=count, last_progress_epoch=now)
    source = attempts.get("c1_source_full", {})
    conditions["source_progress_stalled"] = (source.get("status") == "RUNNING"
        and now - state.get("last_progress_epoch", now) > STALL_SECONDS)
    for driver in drivers:
        conditions["invalid_rows:" + driver["driver"]] = (
            driver["nonfinite_cases"] > 0 or driver["partial_or_invalid_rows"] > 0)
    memory = snapshot["memory_bytes"]
    conditions["low_host_memory"] = memory["MemAvailable"] < max(512 * 1024**2, .05 * memory["MemTotal"])
    for phase, q in queues.items():
        waiting = q["status"] == "WAITING"
        dead = waiting and not q["supervisor_alive"]
        conditions["queue_missing:" + phase] = dead or (q["status"] == "EXECUTING" and not q["supervisor_alive"])
        conditions["queue_deadline:" + phase] = waiting and now > config["queue_deadlines"][phase]
        if q["status"] not in ("WAITING", "EXECUTING", "COMPLETE"):
            events.append((f"queue_terminal:{phase}:{q['status']}", "ERROR", f"{phase} queue {q['status']}: {q.get('error')}"))
        if dead and phase not in attempts and now < config["queue_deadlines"][phase] and (
                state.get("repair_counts", {}).get(phase, 0) < MAX_REPAIRS) and counters.get("queue_missing:" + phase, 0) >= 1:
            repairs.append(phase)
    for key, triggered in conditions.items():
        counters[key] = counters.get(key, 0) + 1 if triggered else 0
        if counters[key] == 2:
            events.append((f"health:{key}:{now}", "ERROR", key.replace("_", " ") + "; inspect live-status.json and command logs"))
        if not triggered and key in state.get("active_problems", []):
            events.append((f"resolved:{key}:{now}", "INFO", "Resolved: " + key))
    state["active_problems"] = [key for key, value in counters.items() if value >= 2]
    running = any(m["status"] == "RUNNING" for m in attempts.values())
    pending = any(q["status"] in ("WAITING", "EXECUTING") and (
        q["supervisor_alive"] or phase in repairs or
        (q["status"] == "WAITING" and phase not in attempts and now < config["queue_deadlines"][phase]
         and state.get("repair_counts", {}).get(phase, 0) < MAX_REPAIRS))
        for phase, q in queues.items())
    done = not running and not pending
    return events, repairs, done


def repair_queue(phase, state, config):
    """Recheck immediately before the dispatcher call; never retry numerical work."""
    snapshot = sample()
    q = snapshot["queues"][phase]
    if (q["status"] != "WAITING" or q["supervisor_alive"] or phase in snapshot["attempts"]
            or snapshot["at"] >= config["queue_deadlines"][phase]
            or state.setdefault("repair_counts", {}).get(phase, 0) >= MAX_REPAIRS):
        return dict(status="SKIPPED_STATE_CHANGED")
    state["repair_counts"][phase] = state["repair_counts"].get(phase, 0) + 1
    try:
        _, proposed_deadline = runner.queue_wait_state(phase, runner.status())
        if proposed_deadline > config["queue_deadlines"][phase]:
            return dict(status="REPAIR_FAILED", error="Restart would extend the original queue deadline")
        result = runner.start_detached(phase)
        return dict(status="RESTART_REQUESTED", launch=result.get("launch"), launch_file=result.get("launch_file"))
    except Exception as exc:
        return dict(status="REPAIR_FAILED", error=str(exc))


def record_event(output, state, key, severity, message):
    if key in state.setdefault("seen_events", []):
        return
    state["seen_events"].append(key)
    event = dict(utc=runner.shared.stamp(), key=key, severity=severity, message=message)
    # Preserve the event even if notification hangs, fails, or the daemon exits.
    with (output / "events.jsonl").open("a") as stream:
        stream.write(json.dumps(event) + "\n")
    delivery = notify("BayesFilter: " + severity.lower(), message)
    with (output / "notifications.jsonl").open("a") as stream:
        stream.write(json.dumps(dict(utc=runner.shared.stamp(), key=key, **delivery)) + "\n")
    state["last_notification"] = delivery


def write_status(output, state, snapshot, config):
    payload = dict(monitor_status=state["status"], heartbeat_utc=runner.shared.stamp(),
        poll_seconds=INTERVAL, active_problems=state.get("active_problems", []),
        repair_counts=state.get("repair_counts", {}), full_replication_complete=False,
        last_notification=state.get("last_notification"), deadline_epoch=config["deadline_epoch"],
        **snapshot)
    runner.shared.atomic_json(output / "live-status.json", payload)
    lines = ["# Replication run monitor", "", f"Heartbeat UTC: {payload['heartbeat_utc']}",
        f"Monitor: {state['status']}; interval {INTERVAL}s. Full replication remains incomplete.",
        "", "## Source progress", ""]
    for d in snapshot["progress"].get("drivers", []):
        lines.append(f"- {d['driver']}: {d['completed_cases']}/{d['expected_cases']}; nonfinite={d['nonfinite_cases']}; incomplete/invalid rows={d['partial_or_invalid_rows']}")
    lines += ["", "## Scheduled continuation", ""]
    for phase, q in snapshot["queues"].items():
        lines.append(f"- {phase}: {q['status']}; supervisor alive={q['supervisor_alive']}")
    lines += ["", "Active health alerts: " + repr(payload["active_problems"]),
              "", "Terminal decisions, errors and repairs: [events](events.jsonl).",
              "Notification submission records: [notifications](notifications.jsonl).",
              "Desktop submission does not confirm display/read receipt. This monitor does not post chat messages.", ""]
    temporary = output / "live-status.md.tmp"
    temporary.write_text("\n".join(lines))
    temporary.replace(output / "live-status.md")
    runner.shared.atomic_json(output / "state.json", state)


def run(output):
    config = json.loads((output / "manifest.json").read_text())
    state = dict(status="RUNNING", repair_counts={}, seen_events=[])
    with (CAMPAIGN / "run-monitor.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another monitor already holds the singleton lock")
        config.update(pid=os.getpid(), process_identity=runner.shared.proc_identity(os.getpid()), status="RUNNING")
        runner.shared.atomic_json(output / "manifest.json", config)
        runner.shared.atomic_json(CAMPAIGN / "monitor-latest.json", dict(output=str(output), pid=config["pid"],
                                 process_identity=config["process_identity"], plan=PLAN))
        snapshot = None
        try:
            record_event(output, state, "started", "INFO", "Replication monitoring enabled: checks every minute. Known Lorenz failure retained; neuroscience is running. Completion/error notifications will follow.")
            while True:
                try:
                    snapshot = sample()
                    state["consecutive_read_errors"] = 0
                except (OSError, ValueError) as exc:
                    state["consecutive_read_errors"] = state.get("consecutive_read_errors", 0) + 1
                    with (output / "read-errors.jsonl").open("a") as stream:
                        stream.write(json.dumps(dict(utc=runner.shared.stamp(), error=repr(exc))) + "\n")
                    if state["consecutive_read_errors"] == 3:
                        record_event(output, state, "repeated_read_error:" + str(time.time()), "ERROR",
                                     "Monitor cannot read current run evidence: " + repr(exc))
                    runner.shared.atomic_json(output / "state.json", state)
                    if snapshot is not None:
                        snapshot["stale_observation"] = True
                        write_status(output, state, snapshot, config)
                    if time.time() >= config["deadline_epoch"]:
                        raise RuntimeError("Monitoring deadline reached while evidence reads fail") from exc
                    time.sleep(INTERVAL)
                    continue
                events, repairs, done = evaluate(snapshot, state, config)
                for event in events:
                    if event[1] == "ERROR" and event[0].startswith(("terminal:", "queue_terminal:", "source_failure:")):
                        state.setdefault("terminal_errors", {})[event[0]] = event[2]
                    record_event(output, state, *event)
                for phase in repairs:
                    repair = repair_queue(phase, state, config)
                    runner.shared.atomic_json(output / "state.json", state)
                    record_event(output, state, f"repair:{phase}:{state['repair_counts'].get(phase, 0)}",
                        "ERROR" if repair["status"] == "REPAIR_FAILED" else "INFO",
                        f"Queue recovery {phase}: " + json.dumps(repair))
                if done or snapshot["at"] >= config["deadline_epoch"]:
                    needs_review = bool(state.get("terminal_errors") or state.get("active_problems"))
                    state["status"] = ("BATCH_TERMINAL_REVIEW_REQUIRED" if needs_review else "BATCH_TERMINAL") if done else "MONITOR_DEADLINE_REACHED"
                    record_event(output, state, "monitor_finished", "INFO" if done and not needs_review else "ERROR",
                        state["status"] + ". See terminal phase decisions; full CSMC/KDM replication remains incomplete.")
                write_status(output, state, snapshot, config)
                if state["status"] != "RUNNING":
                    break
                time.sleep(INTERVAL)
        except BaseException as exc:
            state["status"] = "MONITOR_ERROR"
            record_event(output, state, "monitor_error", "ERROR", "Replication monitor stopped: " + repr(exc))
            runner.shared.atomic_json(output / "state.json", state)
            raise
        finally:
            if snapshot is not None:
                write_status(output, state, snapshot, config)
            config.update(status=state["status"], ended_utc=runner.shared.stamp(),
                          wall_seconds=time.time() - config["started_epoch"])
            runner.shared.atomic_json(output / "manifest.json", config)


def monitoring_deadline(deadlines):
    # Queue deadlines bound waiting, not the final numerical execution itself.
    return max(deadline + runner.PHASES[phase]["limit"] + 120
               for phase, deadline in deadlines.items())


def start():
    with (CAMPAIGN / "monitor-launch.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pointer = CAMPAIGN / "monitor-latest.json"
        if pointer.exists():
            existing = json.loads(pointer.read_text())
            if live(existing.get("pid"), existing.get("process_identity")):
                return dict(status="ALREADY_RUNNING", **existing)
        snapshot = sample()
        deadlines = {key: value["deadline_epoch"] for key, value in snapshot["queues"].items()}
        if any(value is None for value in deadlines.values()):
            raise RuntimeError("Monitor requires the already authorized scheduled queues")
        output = CAMPAIGN / ("monitor-" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + str(time.time_ns()))
        output.mkdir()
        command = [runner.PYTHON, "-B", str(Path(__file__).resolve()), "run", "--output", str(output)]
        source = Path(__file__).read_bytes()
        (output / "monitor-source.py").write_bytes(source)
        config = dict(status="STARTING", started_epoch=time.time(), started_utc=runner.shared.stamp(),
            command=command, plan=PLAN, output_root=str(output), queue_deadlines=deadlines,
            deadline_epoch=monitoring_deadline(deadlines), cpu_only=True, gpu_intentionally_hidden=True,
            python=sys.executable, python_version=sys.version,
            git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            source_sha256=hashlib.sha256(source).hexdigest(),
            dispatcher_sha256=runner.shared.sha(Path(runner.__file__)),
            seeds="N/A: no random or numerical work", data_version="N/A: observing existing run artifacts",
            poll_seconds=INTERVAL, stall_warning_seconds=STALL_SECONDS, max_queue_repairs=MAX_REPAIRS)
        runner.shared.atomic_json(output / "manifest.json", config)
        with (output / "monitor.log").open("x") as log:
            child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        return dict(status="STARTED", output=str(output), pid=child.pid)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("start", "run", "status", "notify-test"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.action == "start":
        print(json.dumps(start(), indent=2))
    elif args.action == "run":
        if args.output is None:
            parser.error("run requires --output")
        run(args.output.resolve())
    elif args.action == "notify-test":
        print(json.dumps(notify("BayesFilter monitoring test", "Testing local completion and error notifications for the replication run."), indent=2))
    else:
        pointer = json.loads((CAMPAIGN / "monitor-latest.json").read_text())
        print(json.dumps(dict(**pointer, alive=live(pointer["pid"], pointer["process_identity"]),
            status=json.loads((Path(pointer["output"]) / "live-status.json").read_text())), indent=2))


if __name__ == "__main__":
    main()
