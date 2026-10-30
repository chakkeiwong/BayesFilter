"""Fresh process-local GPU readiness; no accelerator imports in this module."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import io
import json
import os
import subprocess
import sys

# These observed desktop services may use CUDA for video encoding. They are
# recorded as co-resident load, not mistaken for another research worker.
DESKTOP_EXECUTABLES = frozenset({
    "/usr/NX/bin/nxnode.bin", "/usr/libexec/gnome-remote-desktop-daemon",
})


class GPUResourceUnavailable(RuntimeError):
    """Capacity wait, carrying evidence without rejecting a numerical candidate."""

    def __init__(self, receipt):
        self.receipt = receipt
        super().__init__(json.dumps(receipt, sort_keys=True))


def _query(option):
    raw = subprocess.check_output(["nvidia-smi", option, "--format=csv,noheader,nounits"],
                                  text=True, timeout=60)
    return [row for row in csv.reader(io.StringIO(raw), skipinitialspace=True) if row]


def _process_inventory():
    rows = _query("--query-compute-apps=gpu_uuid,pid,process_name")
    result = []
    for row in rows:
        if len(row) != 3:
            raise RuntimeError("unexpected GPU process inventory: " + repr(row))
        uuid, pid, executable = (value.strip() for value in row)
        result.append({"gpu_uuid": uuid, "pid": int(pid), "executable": executable,
                       "desktop_service": executable in DESKTOP_EXECUTABLES})
    return result


def probe_gpu_inventory(*, requested="auto"):
    """All-device capacity observation; TensorFlow readiness is checked in worker."""
    inventory = []
    for row in _query("--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu"):
        if len(row) != 6:
            raise RuntimeError("unexpected GPU device inventory: " + repr(row))
        index, uuid, name, used, total, utilization = (value.strip() for value in row)
        inventory.append({"index": int(index), "uuid": uuid, "name": name,
            "memory_used_mib": int(used), "memory_total_mib": int(total),
            "utilization_percent": int(utilization)})
    processes = _process_inventory()
    if {p["gpu_uuid"] for p in processes} - {g["uuid"] for g in inventory}:
        raise RuntimeError("unmapped compute GPU; cannot establish resource ownership")
    for gpu in inventory:
        rows = [p for p in processes if p["gpu_uuid"] == gpu["uuid"] and p["pid"] != os.getpid()]
        gpu["desktop_processes"] = [p for p in rows if p["desktop_service"]]
        gpu["other_compute_processes"] = [p for p in rows if not p["desktop_service"]]
        gpu["compute_busy"] = bool(gpu["other_compute_processes"])
    receipt = {"schema": "bayesfilter.q20.gpu_readiness.v2",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(), "status": "PASSED",
        "requested_gpu": str(requested), "gpu_inventory": inventory,
        "selection_policy": "all_devices_no_other_numerical_workload_v1",
        "trusted_execution_required": True, "framework_readiness": "worker_verifies_before_numerical_work"}
    candidates = inventory if str(requested) == "auto" else [g for g in inventory if g["index"] == int(requested)]
    if not candidates:
        raise RuntimeError("requested GPU not present: " + str(requested))
    available = [g for g in candidates if not g["compute_busy"]]
    if not available:
        raise GPUResourceUnavailable({**receipt, "status": "FAILED", "resource_unavailable": True,
                                      "error": "no_gpu_without_numerical_workload"})
    selected = min(available, key=lambda g: (bool(g["desktop_processes"]),
        g["utilization_percent"], g["memory_used_mib"], g["index"]))
    return {**receipt, "selected_host_gpu": selected["index"],
            "desktop_coexistence": bool(selected["desktop_processes"])}


def select_worker_gpu(*, requested="auto"):
    """Select fresh capacity after scheduling delay and before TF import."""
    if "tensorflow" in sys.modules:
        raise RuntimeError("GPU selection must precede TensorFlow import")
    if os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH", "").lower() != "true":
        raise ValueError("worker requires TF_FORCE_GPU_ALLOW_GROWTH=true before import")
    receipt = probe_gpu_inventory(requested=requested)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(receipt["selected_host_gpu"])
    check_gpu_contention(receipt)
    return receipt


def check_gpu_contention(receipt):
    """Check an already selected device; ordinary observations are not a lock."""
    selected = receipt["selected_host_gpu"]
    uuid = next(row["uuid"] for row in receipt["gpu_inventory"] if row["index"] == selected)
    rows = [p for p in _process_inventory() if p["gpu_uuid"] == uuid and p["pid"] != os.getpid()]
    others = [p["pid"] for p in rows if not p["desktop_service"]]
    if others:
        raise GPUResourceUnavailable({"status": "FAILED", "error": "gpu_resource_contention",
            "selected_host_gpu": selected, "other_gpu_pids": others, "launch_readiness": receipt})
    return {"selected_host_gpu": selected, "other_gpu_pids": others,
            "desktop_processes": [p for p in rows if p["desktop_service"]]}


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default="auto")
    args = parser.parse_args()
    try:
        receipt = probe_gpu_inventory(requested=args.gpu)
    except GPUResourceUnavailable as error:
        print(json.dumps(error.receipt, indent=2))
        return 3
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
