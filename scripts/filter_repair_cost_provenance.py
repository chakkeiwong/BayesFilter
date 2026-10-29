"""Host-only provenance and device-sharing checks for diagnostic cost reports."""

import csv
import os
import subprocess
import threading
import time

SCHEMA = "filter_repair_cost_provenance.v1"


class GPUProcessMonitor:
    """Observe sharing around and during costs; samples cannot prove exclusivity."""

    def __init__(self, enabled):
        self.enabled = enabled
        self.uuid = os.environ.get("CUDA_VISIBLE_DEVICES") if enabled else None
        if enabled and (not self.uuid or not self.uuid.startswith("GPU-")):
            raise ValueError("GPU cost monitoring requires a physical UUID")
        self.samples = []
        self.errors = []
        self.pid = os.getpid()
        self._stop = threading.Event()
        self._thread = None

    def _sample(self):
        try:
            result = subprocess.run([
                "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name",
                "--format=csv,noheader,nounits",
            ], check=True, capture_output=True, text=True, timeout=5)
            processes = [{"uuid": uuid, "pid": int(pid), "name": name}
                for uuid, pid, name in csv.reader(result.stdout.splitlines(), skipinitialspace=True)
                if uuid == self.uuid]
            self.samples.append({"monotonic_seconds": time.monotonic(), "processes": processes})
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            self.errors.append(f"{type(error).__name__}: {error}")

    def _observe(self):
        while not self._stop.wait(1.0):
            self._sample()

    def __enter__(self):
        if self.enabled:
            self._sample()
            self._thread = threading.Thread(target=self._observe, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *_error):
        if self.enabled:
            self._stop.set()
            self._thread.join(timeout=6)
            if self._thread.is_alive():
                self.errors.append("monitor_thread_did_not_stop")
            self._sample()

    def payload(self):
        return {"schema": SCHEMA, "enabled": self.enabled, "uuid": self.uuid,
            "pid": self.pid, "samples": self.samples, "errors": self.errors,
            "limitation": "Sampled sharing evidence; brief sharing between samples can be missed."}


def validate_cost_device(run, provenance, observation):
    """Reject timing without matching visibility, growth and clean observations."""
    if run["state"] != "passed":
        raise ValueError("Cost worker did not pass")
    if observation["schema"] != SCHEMA or observation["errors"]:
        raise ValueError("Missing or failed cost-device observations")
    visible = run["environment"]["CUDA_VISIBLE_DEVICES"]
    if visible != provenance["cuda_visible_devices"]:
        raise ValueError("Worker and manifest device visibility differ")
    memory = provenance["gpu_memory_policy"]
    if (memory["mode"] != "memory_growth" or not memory["all_physical_devices_memory_growth"]
            or not memory["configured_before_logical_device_initialization"]):
        raise ValueError("Unverified pre-initialization GPU growth policy")
    if run["environment"]["TF_FORCE_GPU_ALLOW_GROWTH"] != "true":
        raise ValueError("Missing growth environment policy")
    if run["device"] == "CPU":
        if (visible != "-1" or observation["enabled"] or observation["uuid"] is not None
                or provenance["trust_basis"] != "explicit_cpu_reference"
                or memory["physical_devices"]):
            raise ValueError("CPU reference did not explicitly hide GPUs")
        return None
    if run["device"] != "GPU":
        raise ValueError("Unknown cost device")
    if (run["environment"].get("BAYESFILTER_TEST_DEVICE_SCOPE") != "visible"
            or provenance.get("bayesfilter_test_device_scope") != "visible"):
        raise ValueError("GPU costs require explicit visible pytest device scope")
    uuid = run["gpu_uuid"]
    if (not uuid.startswith("GPU-") or visible != uuid or observation["uuid"] != uuid
            or not observation["enabled"]):
        raise ValueError("Unmatched physical GPU UUID")
    if provenance["trust_basis"] != "owner_designated_managed_session_visible_gpu_trusted":
        raise ValueError("Untrusted GPU evidence")
    if (len(memory["physical_devices"]) != 1
            or memory["physical_devices"][0]["memory_growth"] is not True):
        raise ValueError("Unexpected visible GPU or growth state")
    preflight = run["gpu_preflight"]
    if len(preflight) < 2 or run["gpu_performance_preflight_uncontended"] is not True:
        raise ValueError("Shared-device timing or missing preflight")
    for sample in preflight[-2:]:
        if (sample["selected_uuid"] != uuid or sample["performance_preflight_uncontended"] is not True
                or sample["desktop_fallback"] or sample["utilization_percent"] > 5):
            raise ValueError("Shared or mismatched GPU preflight")
        selected = [row for row in sample["devices"] if row["uuid"] == uuid]
        if len(selected) != 1 or selected[0]["compute_processes"] or selected[0]["desktop"]:
            raise ValueError("Shared or desktop GPU in preflight inventory")
    if len(observation["samples"]) < 2:
        raise ValueError("Missing in-run GPU observations")
    previous = float("-inf")
    for sample in observation["samples"]:
        if sample["monotonic_seconds"] < previous:
            raise ValueError("Unordered GPU observations")
        previous = sample["monotonic_seconds"]
        for process in sample["processes"]:
            if process["uuid"] != uuid or process["pid"] != observation["pid"]:
                raise ValueError("Shared or mismatched GPU during costs")
    return uuid


def require_same_physical_gpu(uuids):
    devices = {uuid for uuid in uuids if uuid is not None}
    if len(devices) > 1:
        raise ValueError("Mixed physical GPU UUIDs cannot support paired costs")
