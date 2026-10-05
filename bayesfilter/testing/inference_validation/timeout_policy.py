"""Framework-free progress, workload, and timeout policy for isolated fits.

The numerical worker is supervised by a parent that must remain free of
TensorFlow.  This module deliberately treats workload telemetry as advisory:
missing or untrusted telemetry can explain a timeout, but cannot grant extra
time.  Numerical progress is inferred from durable tuning evidence, never from
stdout or a heartbeat alone.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Mapping


TIMEOUT_POLICY_SCHEMA = "bayesfilter.inference_validation_timeout_policy.v1"


@dataclass(frozen=True)
class TimeoutPolicy:
    """Explicit supervision controls; all durations are wall-clock seconds."""

    poll_interval_seconds: float = 1.0
    stall_timeout_seconds: float | None = None
    max_extension_seconds: float = 0.0
    progress_grace_seconds: float = 60.0
    normalized_load_threshold: float = 1.0
    workload_probe_timeout_seconds: float = 1.0
    shutdown_grace_seconds: float = 5.0
    telemetry_interval_seconds: float = 30.0
    extension_mode: str = "latest_sample"
    gpu_admission_wait_seconds: float = 0.0
    gpu_admission_mode: str = "idle"
    max_contention_retries: int = 0

    def __post_init__(self) -> None:
        for name in (
            "poll_interval_seconds",
            "progress_grace_seconds",
            "normalized_load_threshold",
            "workload_probe_timeout_seconds",
            "shutdown_grace_seconds",
            "telemetry_interval_seconds",
        ):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(float(value)) or float(value) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.stall_timeout_seconds is not None and (
            type(self.stall_timeout_seconds) not in (int, float)
            or not math.isfinite(float(self.stall_timeout_seconds))
            or float(self.stall_timeout_seconds) <= 0
        ):
            raise ValueError("stall_timeout_seconds must be positive or null")
        if type(self.max_extension_seconds) not in (int, float) or not math.isfinite(float(self.max_extension_seconds)) or float(self.max_extension_seconds) < 0:
            raise ValueError("max_extension_seconds must be finite and nonnegative")
        if self.max_extension_seconds and self.progress_grace_seconds <= 0:
            raise ValueError("contention grace requires a positive progress window")
        if self.extension_mode not in {"latest_sample", "observed_intervals"}:
            raise ValueError("unknown contention extension_mode")
        value = self.gpu_admission_wait_seconds
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError("gpu_admission_wait_seconds must be finite and nonnegative")
        if self.gpu_admission_mode not in {"idle", "shared"}:
            raise ValueError("gpu_admission_mode must be idle or shared")
        if type(self.max_contention_retries) is not int or self.max_contention_retries < 0:
            raise ValueError("max_contention_retries must be a nonnegative integer")

    @classmethod
    def from_options(cls, options: Mapping[str, Any] | None) -> "TimeoutPolicy":
        raw = {} if options is None else options.get("timeout_policy", {})
        if not isinstance(raw, Mapping):
            raise ValueError("timeout_policy must be a mapping")
        allowed = {
            "poll_interval_seconds",
            "stall_timeout_seconds",
            "max_extension_seconds",
            "progress_grace_seconds",
            "normalized_load_threshold",
            "workload_probe_timeout_seconds",
            "shutdown_grace_seconds",
            "telemetry_interval_seconds",
            "extension_mode",
            "gpu_admission_wait_seconds",
            "gpu_admission_mode",
            "max_contention_retries",
        }
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError("unknown timeout_policy fields: " + ", ".join(sorted(unknown)))
        return cls(**dict(raw))

    def payload(self) -> dict[str, Any]:
        return {
            "schema": TIMEOUT_POLICY_SCHEMA,
            "poll_interval_seconds": self.poll_interval_seconds,
            "stall_timeout_seconds": self.stall_timeout_seconds,
            "max_extension_seconds": self.max_extension_seconds,
            "progress_grace_seconds": self.progress_grace_seconds,
            "normalized_load_threshold": self.normalized_load_threshold,
            "workload_probe_timeout_seconds": self.workload_probe_timeout_seconds,
            "shutdown_grace_seconds": self.shutdown_grace_seconds,
            "telemetry_interval_seconds": self.telemetry_interval_seconds,
            "extension_mode": self.extension_mode,
            "gpu_admission_wait_seconds": self.gpu_admission_wait_seconds,
            "gpu_admission_mode": self.gpu_admission_mode,
            "max_contention_retries": self.max_contention_retries,
        }


class ObservedContention:
    """Bounded observed busy intervals; never an estimate of lost GPU share."""

    def __init__(self, policy):
        self.maximum_interval = policy.telemetry_interval_seconds + 2 * policy.workload_probe_timeout_seconds
        self.previous = None
        self.seconds = 0.0

    def observe(self, elapsed, sample):
        busy = sample.get("trusted_for_extension") is True and sample.get("contended") is True
        if self.previous is not None:
            prior_time, prior_busy = self.previous
            interval = elapsed - prior_time
            if interval < 0:
                raise ValueError("workload sample clock moved backwards")
            if busy and prior_busy:
                self.seconds += min(interval, self.maximum_interval)
        self.previous = (elapsed, busy)
        return self.seconds


def gpu_admission_supported(sample, policy):
    """Trusted sharing is eligible; utilization is not a numerical validity gate."""
    if sample.get("trusted_for_extension") is not True:
        return False
    if policy.gpu_admission_mode == "idle":
        return sample.get("contended") is False
    gpu = sample.get("gpu", {})
    used, total = gpu.get("memory_used_mib"), gpu.get("memory_total_mib")
    return (gpu.get("available") is True and gpu.get("trusted") is True
            and all(type(x) in (int, float) and math.isfinite(x) for x in (used, total))
            and 0 <= used < total)


def wait_for_gpu_admission(*, device, policy, deadline, workload_probe=None):
    """Wait inside the enclosing deadline; do not initialize a GPU framework.

    Unknown telemetry is not an idle device. The numerical child is not yet
    launched, and this waiting does not spend its base work allowance.
    """
    if device != "gpu" or policy.gpu_admission_wait_seconds == 0:
        return {"admitted": True, "wait_seconds": 0., "samples": [], "policy": "disabled"}
    if not math.isfinite(deadline):
        raise ValueError("admission requires a finite enclosing deadline")
    started = time.monotonic()
    end = min(deadline, started + policy.gpu_admission_wait_seconds)
    samples = []
    probe = workload_probe or default_workload_probe
    admitted = False
    while time.monotonic() < end:
        from dataclasses import replace
        remaining = end - time.monotonic()
        if remaining <= 0:
            break
        probe_policy = replace(policy, workload_probe_timeout_seconds=min(
            policy.workload_probe_timeout_seconds, max(.001, remaining / 2)))
        try:
            sample = dict(probe(device, None, probe_policy))
        except Exception as exc:
            sample = {"trusted_for_extension": False, "reason": type(exc).__name__ + ": " + str(exc)}
        samples.append({"elapsed_seconds": time.monotonic()-started, **sample})
        if gpu_admission_supported(sample, policy) and time.monotonic() < end:
            admitted = True
            break
        delay = min(policy.telemetry_interval_seconds, end-time.monotonic())
        if delay > 0:
            time.sleep(delay)
    return {"admitted": admitted, "wait_seconds": time.monotonic()-started, "samples": samples,
            "policy": ("trusted_shared_gpu" if policy.gpu_admission_mode == "shared"
                       else "trusted_selected_gpu_without_foreign_process"),
            "enclosing_deadline": deadline}


class ProgressEvidenceError(ValueError):
    """A committed progress record is malformed; do not treat it as a stall."""


class ProgressObserver:
    """Read small control records; never rescan numerical tensor payloads."""

    def __init__(self, root):
        self.root = Path(root)
        self.cache = {}
        self.bundle_cache = {}

    def _read(self, path):
        if not path.exists():
            return None
        stat = path.stat()
        key = (stat.st_mtime_ns, stat.st_size)
        cached = self.cache.get(path)
        if cached is not None and cached[0] == key:
            return cached[1]
        try:
            value = json.loads(path.read_text())
            if not isinstance(value, dict):
                raise ValueError("expected JSON object")
        except (OSError, ValueError) as exc:
            raise ProgressEvidenceError(f"invalid committed progress: {path}: {exc}") from exc
        self.cache[path] = (key, value)
        return value

    def snapshot(self):
        tuning = self.root / "tuning"
        preparation = self._read(tuning / "preparation_progress.json")
        phases = set()
        if preparation is not None:
            if preparation.get("schema") != "bayesfilter.hmc_preparation_progress.v1":
                raise ProgressEvidenceError("unknown preparation progress schema")
            try:
                for event in preparation["events"]:
                    if event["phase"].endswith(("_complete", "_completed", ".completed")):
                        # Stable work identifiers distinguish completed chunks;
                        # duplicated callbacks and timing changes are not work.
                        details = event.get("details", {})
                        identity = {key: details[key] for key in (
                            "offset", "count", "seed", "round_seed", "window_index",
                            "segment_index", "attempt_index", "preparation_attempt_index",
                            "completed_transition_count", "checkpoint") if key in details}
                        operational = details.get("operational_progress", {})
                        identity["operational"] = {key: operational[key] for key in (
                            "completed_transition_count", "completed_window_count",
                            "completed_segment_count", "segment_index") if key in operational}
                        phases.add((event["phase"], json.dumps(identity, sort_keys=True, allow_nan=False)))
            except (KeyError, TypeError, ValueError, AttributeError) as exc:
                raise ProgressEvidenceError("malformed preparation completion") from exc
        checkpoint = self._read(tuning / "tuning_checkpoint.json")
        evidence = ()
        partial = ()
        if checkpoint is not None:
            if checkpoint.get("schema") != "bayesfilter.hmc_numerical_tuning_checkpoint.v1":
                raise ProgressEvidenceError("unknown tuning checkpoint schema")
            body = {k: v for k, v in checkpoint.items() if k != "content_hash"}
            encoded = json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
            if hashlib.sha256(encoded).hexdigest() != checkpoint.get("content_hash"):
                raise ProgressEvidenceError("tuning checkpoint checksum mismatch")
            evidence = tuple(checkpoint["numerical_evidence_hashes"])
            partial = tuple(sorted(h for hs in checkpoint["partial_chunks"].values() for h in hs))
        # The directory is committed atomically. Check its small control bundle;
        # tensor payload verification belongs to the native loader on resume.
        posterior = []
        for directory in sorted(self.root.glob("members/*/posterior_chunks/committed/*")):
            if not directory.is_dir():
                raise ProgressEvidenceError("non-directory posterior commit")
            try:
                bundle_path, hash_path = directory / "bundle.json", directory / "bundle.sha256"
                key = tuple((p.stat().st_mtime_ns, p.stat().st_size) for p in (bundle_path, hash_path))
                cached = self.bundle_cache.get(directory)
                if cached is None or cached[0] != key:
                    content, digest = bundle_path.read_bytes(), hash_path.read_text()
                    bundle = json.loads(content)
                    if (hashlib.sha256(content).hexdigest() != digest or
                            bundle["schema"] != "bayesfilter.durable_tensor_checkpoint.v1"):
                        raise ValueError("posterior bundle checksum or schema mismatch")
                    self.bundle_cache[directory] = (key, digest)
                posterior.append((str(directory.relative_to(self.root)), self.bundle_cache[directory][1]))
            except (OSError, ValueError, KeyError, TypeError) as exc:
                raise ProgressEvidenceError(f"invalid committed posterior progress: {directory}: {exc}") from exc
        signature = (tuple(("preparation", *phase) for phase in sorted(phases))
                     + tuple(("evidence", h) for h in evidence)
                     + tuple(("partial", h) for h in partial)
                     + tuple(("posterior", *p) for p in posterior))
        return {"signature": signature,
                "preparation_completions": len(phases), "evidence_count": len(evidence),
                "partial_chunks": len(partial), "posterior_chunks": len(posterior)}


def numerical_progress_snapshot(root):
    return ProgressObserver(root).snapshot()


def progress_changed(previous, current):
    return bool(set(current["signature"]) - set(previous["signature"]))


def _gpu_workload(timeout_seconds, child_pid):
    # CUDA ordinal is not nvidia-smi index. Extensions require an unambiguous
    # UUID selection; numeric CUDA selections remain diagnostic-only.
    selected = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not selected.startswith("GPU-") or "," in selected:
        return {"available": False, "trusted": False, "reason": "selected_gpu_uuid_required"}
    try:
        devices = subprocess.run(
            ["nvidia-smi", "--query-gpu=uuid,utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"], check=True, capture_output=True,
            text=True, timeout=timeout_seconds)
        apps = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=timeout_seconds)
        device_rows = [row.strip().split(",") for row in devices.stdout.splitlines() if row.strip()]
        matched = [row for row in device_rows if row[0].strip() == selected]
        if len(matched) != 1 or len(matched[0]) != 4:
            raise ValueError("selected device missing or malformed")
        values = [float(value) for value in matched[0][1:]]
        if not all(math.isfinite(value) and value >= 0 for value in values):
            raise ValueError("invalid utilization/memory telemetry")
        foreign = []
        own_group = os.getpgid(child_pid) if child_pid is not None else None
        for line in apps.stdout.splitlines():
            if not line.strip():
                continue
            fields = [field.strip() for field in line.split(",")]
            if len(fields) != 2:
                raise ValueError("malformed GPU process telemetry")
            pid = int(fields[0])
            if fields[1] != selected or pid == child_pid:
                continue
            try:
                if own_group is not None and os.getpgid(pid) == own_group:
                    continue
            except ProcessLookupError:
                continue
            foreign.append(pid)
        return {"available": True, "trusted": True, "uuid": selected,
                "utilization_gpu_pct": values[0], "memory_used_mib": values[1],
                "memory_total_mib": values[2], "foreign_compute_pids": sorted(set(foreign))}
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        return {"available": False, "trusted": False, "reason": type(exc).__name__ + ": " + str(exc)}


def machine_workload_snapshot(*, device, child_pid, policy):
    cpu_count = max(1, os.cpu_count() or 1)
    try:
        load = float(os.getloadavg()[0])
        cpu = {"available": True, "load_1m": load, "cpu_count": cpu_count,
               "normalized_load": load / cpu_count}
    except (AttributeError, OSError):
        cpu = {"available": False, "reason": "load_unavailable"}
    gpu = (_gpu_workload(policy.workload_probe_timeout_seconds, child_pid)
           if device == "gpu" else {"available": False, "reason": "gpu_intentionally_hidden"})
    if device == "gpu":
        contended = bool(gpu.get("available") and gpu.get("foreign_compute_pids"))
        trusted = gpu.get("trusted", False)
        basis = "foreign_compute_process_on_selected_gpu" if trusted else "unknown"
    else:
        contended = bool(cpu.get("available") and
                         cpu["normalized_load"] >= policy.normalized_load_threshold)
        trusted = cpu.get("available", False)
        basis = "host_load_heuristic_not_proof_of_foreign_contention"
    return {"device": device, "child_pid": child_pid, "cpu": cpu, "gpu": gpu,
            "contended": contended, "contention_basis": basis, "trusted_for_extension": trusted}


def default_workload_probe(device, child_pid, policy):
    return machine_workload_snapshot(device=device, child_pid=child_pid, policy=policy)


class FitAllowance:
    """The supervisor owns an atomic cooperative deadline read by its child."""

    def __init__(self, path):
        self.path = Path(path)

    def publish(self, *, deadline, hard_deadline, reason):
        from .storage import write_json
        write_json(self.path, {"schema": "bayesfilter.fit_allowance.v1", "deadline": deadline,
                              "hard_deadline": hard_deadline, "reason": reason})

    def available(self):
        import time
        from .storage import read_json
        record = read_json(self.path)
        if record.get("schema") != "bayesfilter.fit_allowance.v1":
            raise ValueError("invalid fit allowance schema")
        deadline, hard = record["deadline"], record["hard_deadline"]
        if not all(type(x) in (int, float) and math.isfinite(x) for x in (deadline, hard)) or deadline > hard:
            raise ValueError("invalid fit allowance deadlines")
        return time.monotonic() < deadline
