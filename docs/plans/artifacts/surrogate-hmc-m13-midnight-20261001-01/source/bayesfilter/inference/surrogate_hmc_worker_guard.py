"""Independent local-process watchdog and durable resource receipt (no TF).
The launcher may disappear; this guard kills its worker and preserves accounting.
Linux parent identity includes process start time, so PID reuse is not survival.
"""
from __future__ import annotations
import argparse
import ctypes
import json
import math
import os
from pathlib import Path
import resource
import signal
import subprocess
import time

def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n")
    temporary.replace(path)

def process_identity(pid):
    try:
        stat = Path(f"/proc/{pid}/stat").read_text().rsplit(")",1)[1].split()
        if stat[0] == "Z":
            return None
        return dict(pid=pid, start=stat[19])
    except (FileNotFoundError, ProcessLookupError):
        return None

def group_cpu(group):
    total = 0
    for entry in Path("/proc").iterdir():
        if not entry.name.isdecimal():
            continue
        try:
            stat = (entry/"stat").read_text().rsplit(")",1)[1].split()
            if int(stat[2]) == group:
                total += sum(int(stat[i]) for i in (11,12,13,14))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return total / os.sysconf("SC_CLK_TCK")

def run_guard(spec_path):
    spec = json.loads(Path(spec_path).read_text())
    directory = Path(spec["directory"])
    record = dict(spec["record"], guard_pid=os.getpid(), owner=spec["owner"],
                  accounting_complete=False)
    start = spec["started_monotonic"]
    parent_signal = []
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, lambda received, frame: parent_signal.append(received))
    # Orphan descendants become our children, allowing exact wait4 receipts.
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
        raise OSError(ctypes.get_errno(), "cannot establish worker subreaper")
    process, stopped = None, None
    cpu_lower_bound = 0.
    environment = dict(os.environ, **spec["environment"])
    environment["BAYESFILTER_WORKER_WALL_DEADLINE"] = str(start + record["timeout_seconds"] - 2.)
    environment["BAYESFILTER_WORKER_CPU_ALLOWANCE"] = str(record["cpu_allowance_seconds"])
    before = resource.getrusage(resource.RUSAGE_SELF)
    before_cpu = before.ru_utime + before.ru_stime
    def measured_cpu():
        self_usage = resource.getrusage(resource.RUSAGE_SELF)
        child_usage = resource.getrusage(resource.RUSAGE_CHILDREN)
        return self_usage.ru_utime+self_usage.ru_stime-before_cpu + child_usage.ru_utime+child_usage.ru_stime
    expected_guard = os.getpid()
    def child_limits():
        # Guard and worker have independent sessions. A lost guard still kills
        # the worker; ordinary signals leave the guard time to write its receipt.
        expected = expected_guard
        child_libc = ctypes.CDLL(None, use_errno=True)
        if child_libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), "cannot set parent-death signal")
        if os.getppid() != expected:
            os._exit(127)
        hard = max(1, math.ceil(record["cpu_allowance_seconds"]))
        resource.setrlimit(resource.RLIMIT_CPU, (hard, hard))
    try:
        if process_identity(spec["owner"]["pid"]) != spec["owner"]:
            stopped = "OWNER_LOST"
        else:
            with (directory/"command.log").open("w") as log:
                process = subprocess.Popen(record["command"], cwd=record["cwd"], env=environment,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True, preexec_fn=child_limits)
                record["worker_pid"] = process.pid
                while process.poll() is None:
                    cpu_lower_bound = max(cpu_lower_bound, group_cpu(process.pid)+measured_cpu())
                    elapsed = time.monotonic()-start
                    if parent_signal:
                        stopped = "INTERRUPTED"
                    elif process_identity(spec["owner"]["pid"]) != spec["owner"]:
                        stopped = "OWNER_LOST"
                    elif elapsed >= record["timeout_seconds"]:
                        stopped = "WALL_TIME_LIMIT"
                    elif cpu_lower_bound >= record["cpu_allowance_seconds"]:
                        stopped = "JOB_CPU_BUDGET_LIMIT"
                    atomic_json(directory/"heartbeat.json", dict(record,
                        wall_seconds=elapsed, cpu_worker_seconds_lower_bound=cpu_lower_bound,
                        gpu_seconds=elapsed if record["device"]=="gpu" else 0.,
                        status=stopped or "RUNNING"))
                    if stopped:
                        break
                    time.sleep(.1)
    except BaseException as exc:
        stopped = "LAUNCH_ERROR" if process is None else "GUARD_ERROR"
        record.update(error=f"{type(exc).__name__}: {exc}", error_type=type(exc).__name__)
    finally:
        if process is not None:
            # Kill the complete group even when the main worker has exited.
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=2.)
            except subprocess.TimeoutExpired:
                pass
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            # Reap adopted grandchildren too, including multiprocessing workers.
            cleanup_deadline = time.monotonic()+3.
            accounting_complete = False
            while time.monotonic()<cleanup_deadline:
                try:
                    pid, _ = os.waitpid(-1, os.WNOHANG)
                except ChildProcessError:
                    accounting_complete = True
                    break
                if pid == 0:
                    children = Path(f"/proc/{os.getpid()}/task/{os.getpid()}/children").read_text().split()
                    for child in children:
                        try:
                            os.kill(int(child), signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                    time.sleep(.05)
        else:
            accounting_complete = True
        elapsed = time.monotonic()-start
        cpu = measured_cpu()
        code = process.returncode if process is not None else None
        if code == -signal.SIGKILL and stopped is None and cpu >= record["cpu_allowance_seconds"]:
            stopped = "JOB_CPU_BUDGET_LIMIT"
        record.update(status=stopped or ("COMPLETED" if code==0 else "FAILED"),
            exit_code=code, wall_seconds=elapsed, cpu_worker_seconds=cpu,
            cpu_worker_seconds_observed_lower_bound=cpu_lower_bound,
            gpu_seconds=elapsed if record["device"]=="gpu" else 0.,
            accounting_complete=accounting_complete)
        atomic_json(directory/"guard-receipt.json", record)
        atomic_json(directory/"run-manifest.json", record)
    return record

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", type=Path)
    run_guard(parser.parse_args().spec)
