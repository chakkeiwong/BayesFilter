"""Read-only Linux process diagnostics for the filter repair campaign.

This host-side observer is not a benchmark kernel or a runtime decision path.
The parent owns the output so native child failures preserve the samples.
"""

import json
import subprocess
import time
from collections import Counter
from pathlib import Path


def memory_snapshot(pid):
    process = Path(f"/proc/{pid}")
    mappings = (process / "maps").read_text().splitlines()
    status = dict(line.split(":", 1) for line in (process / "status").read_text().splitlines())
    memory = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    permissions = Counter()
    anonymous = Counter()
    for line in mappings:
        fields = line.split(maxsplit=5)
        permissions[fields[1]] += 1
        if len(fields) == 5:
            anonymous[fields[1]] += 1
    return {
        "map_count": len(mappings),
        "mapping_permissions": dict(permissions),
        "anonymous_mapping_permissions": dict(anonymous),
        "process_status": {key: status[key].strip() for key in
                           ("VmPeak", "VmSize", "VmHWM", "VmRSS", "VmData", "Threads")
                           if key in status},
        "system_memory": {key: memory[key].strip() for key in
                          ("MemAvailable", "SwapFree", "CommitLimit", "Committed_AS")},
    }


def process_memory_limits(pid):
    process = Path(f"/proc/{pid}")
    cgroup = (process / "cgroup").read_text()
    limits = {
        "pid": pid,
        "vm_max_map_count": int(Path("/proc/sys/vm/max_map_count").read_text()),
        "vm_overcommit_memory": int(Path("/proc/sys/vm/overcommit_memory").read_text()),
        "process_limits": (process / "limits").read_text(),
        "cgroup": cgroup,
        "cgroup_memory": {},
    }
    for line in cgroup.splitlines():
        if line.startswith("0::"):
            root = Path("/sys/fs/cgroup")
            directory = root / line[3:].lstrip("/")
            while directory.is_relative_to(root):
                limits["cgroup_memory"][str(directory)] = {
                    name: (directory / name).read_text().strip()
                    for name in ("memory.max", "memory.high", "memory.current", "memory.events")
                    if (directory / name).is_file()
                }
                directory = directory.parent
    return limits


def wait_observing_memory(process, timeout, directory):
    """Wait with the ordinary deadline and durable once-per-second snapshots."""
    started = time.monotonic()
    (directory / "process-memory-limits.json").write_text(
        json.dumps(process_memory_limits(process.pid), indent=2) + "\n")
    with (directory / "process-memory.jsonl").open("x") as output:
        while True:
            try:
                snapshot = memory_snapshot(process.pid)
            except (FileNotFoundError, ProcessLookupError):
                return process.wait(timeout=max(0, timeout - (time.monotonic() - started)))
            snapshot["elapsed_seconds"] = time.monotonic() - started
            output.write(json.dumps(snapshot) + "\n")
            output.flush()
            remaining = timeout - (time.monotonic() - started)
            if remaining <= 0:
                raise subprocess.TimeoutExpired(process.args, timeout)
            try:
                return process.wait(timeout=min(1.0, remaining))
            except subprocess.TimeoutExpired:
                pass
