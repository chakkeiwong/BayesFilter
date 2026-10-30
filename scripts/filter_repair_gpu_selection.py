"""Host-only GPU scheduling for the bounded filter-repair campaign."""

import csv
import subprocess
import time
from datetime import datetime, timezone

POLICY = "filter_repair_available_gpu_desktop_fallback_v1"
MINIMUM_FREE_MIB = 8 * 1024
MAX_UTILIZATION = 50
DESKTOP_PROCESSES = ("gnome-remote-desktop", "xrdp", "krfb", "x11vnc", "wayvnc")


def gpu_snapshot():
    devices = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,name,memory.free,memory.used,utilization.gpu,display_active",
        "--format=csv,noheader,nounits",
    ], text=True, timeout=5)
    processes = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,process_name", "--format=csv,noheader,nounits",
    ], text=True, timeout=5)
    desktop_processes, compute_processes = {}, {}
    for uuid, name in csv.reader(processes.splitlines(), skipinitialspace=True):
        compute_processes.setdefault(uuid, []).append(name)
        if any(marker in name.lower() for marker in DESKTOP_PROCESSES):
            desktop_processes.setdefault(uuid, []).append(name)
    result = []
    for index, uuid, name, free, used, utilization, display in csv.reader(devices.splitlines(), skipinitialspace=True):
        if display not in ("Enabled", "Disabled"):
            raise ValueError(f"Unknown display status for GPU{index}: {display}")
        reasons = (["active_display"] if display == "Enabled" else []) + desktop_processes.get(uuid, [])
        result.append({"index": int(index), "uuid": uuid, "name": name, "free_mib": int(free),
            "memory_mib": int(used), "utilization_percent": int(utilization),
            "desktop": bool(reasons), "desktop_reasons": reasons,
            "compute_processes": compute_processes.get(uuid, [])})
    if len(result) != 4 or {row["index"] for row in result} != {0, 1, 2, 3}:
        raise ValueError("GPU inventory changed; expected the owner's four-device machine")
    if len({row["uuid"] for row in result}) != len(result):
        raise ValueError("Duplicate physical GPU identity")
    return result


def eligible_devices(snapshot):
    non_desktop = [row for row in snapshot if not row["desktop"]]
    fallback = bool(non_desktop) and all(row["utilization_percent"] > MAX_UTILIZATION
        and row["free_mib"] < MINIMUM_FREE_MIB for row in non_desktop)
    return [row for row in snapshot if (not row["desktop"] or fallback)
        and row["utilization_percent"] <= MAX_UTILIZATION and row["free_mib"] >= MINIMUM_FREE_MIB]


def check_gpu_available(requested=None, *, snapshot_fn=gpu_snapshot):
    if requested is not None and requested not in (0, 1, 2, 3):
        raise ValueError("Campaign GPU index must be 0, 1, 2 or 3")
    samples, previous_uuid, consecutive = [], None, 0
    for attempt in range(6):
        snapshot = snapshot_fn()
        candidates = [row for row in eligible_devices(snapshot) if requested is None or row["index"] == requested]
        candidates.sort(key=lambda row: (row["utilization_percent"], -row["free_mib"], row["index"]))
        # Retain an admissible previous candidate for the second confirmation.
        selected = next((row for row in candidates if row["uuid"] == previous_uuid), candidates[0] if candidates else None)
        sample = {"policy": POLICY, "sampled_utc": datetime.now(timezone.utc).isoformat(),
            "minimum_free_mib": MINIMUM_FREE_MIB, "maximum_utilization_percent": MAX_UTILIZATION,
            "devices": snapshot, "selected_gpu_index": None if selected is None else selected["index"]}
        if selected is None:
            previous_uuid, consecutive = None, 0
        else:
            consecutive = consecutive + 1 if selected["uuid"] == previous_uuid else 1
            previous_uuid = selected["uuid"]
            sample.update(selected_uuid=selected["uuid"], desktop_fallback=selected["desktop"],
                memory_mib=selected["memory_mib"], free_mib=selected["free_mib"],
                utilization_percent=selected["utilization_percent"],
                performance_preflight_uncontended=not selected["desktop"]
                    and not selected.get("compute_processes") and selected["utilization_percent"] <= 5)
        samples.append(sample)
        if consecutive == 2:
            return samples
        if attempt < 5:
            time.sleep(2)
    raise RuntimeError(f"GPU contention veto after bounded recheck: {samples}")
