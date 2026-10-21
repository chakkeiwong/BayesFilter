"""Robust-runner schemas and JSON helpers for postmortem-safe artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RunManifest:
    run_id: str
    command: tuple[str, ...]
    git_commit: str
    config_hash: str
    artifact_root: str
    cpu_gpu_status: str
    program_signature: str = "unspecified"
    device_policy: Mapping[str, Any] = field(default_factory=dict)
    thread_caps: Mapping[str, Any] = field(default_factory=dict)
    worker_config_hash: str = "unspecified"
    nonclaims: tuple[str, ...] = (
        "runner manifest is engineering metadata, not scientific evidence",
    )


@dataclass(frozen=True)
class WorkerRecord:
    worker_id: str
    command: tuple[str, ...]
    config_hash: str
    return_code: int | None
    runtime_s: float | None
    timed_out: bool = False
    pid: int | None = None
    status: str = "unknown"
    device_policy: Mapping[str, Any] = field(default_factory=dict)
    thread_caps: Mapping[str, Any] = field(default_factory=dict)
    worker_config_hash: str = "unspecified"


@dataclass(frozen=True)
class TimeoutRecord:
    worker_id: str
    timeout_s: float
    elapsed_s: float
    action: str


@dataclass(frozen=True)
class StageEvent:
    stage: str
    status: str
    timestamp: str
    payload: Mapping[str, Any]


@dataclass(frozen=True)
class WorkerManifest:
    worker_id: str
    command: tuple[str, ...]
    pid: int | None
    git_commit: str
    artifact_root: str
    normalized_config: Mapping[str, Any]
    program_signature: str
    device_policy: Mapping[str, Any]
    thread_caps: Mapping[str, Any]
    worker_config: Mapping[str, Any]
    environment: Mapping[str, Any] = field(default_factory=dict)
    nonclaims: tuple[str, ...] = (
        "worker manifest is engineering metadata, not scientific evidence",
    )

    @property
    def config_hash(self) -> str:
        return stable_config_hash(self.normalized_config)

    @property
    def worker_config_hash(self) -> str:
        return stable_config_hash(self.worker_config)

    @property
    def stale_match_payload(self) -> Mapping[str, Any]:
        return stale_match_payload(
            self.normalized_config,
            program_signature=self.program_signature,
            device_policy=self.device_policy,
            thread_caps=self.thread_caps,
            worker_config_hash=self.worker_config_hash,
        )


@dataclass(frozen=True)
class PartialResultSnapshot:
    worker_id: str
    stage: str
    status: str
    payload: Mapping[str, Any]
    nonfinite_count: int = 0
    first_failure: Mapping[str, Any] | None = None
    nonclaims: tuple[str, ...] = (
        "partial result snapshot is diagnostic evidence only",
    )


@dataclass(frozen=True)
class ReducerRowStatus:
    worker_id: str
    status: str
    reason: str
    artifact_path: str | None = None
    return_code: int | None = None
    timed_out: bool = False
    stale: bool = False
    nonclaims: tuple[str, ...] = (
        "reducer status is artifact-validity metadata, not sampler validity evidence",
    )


@dataclass(frozen=True)
class TimingBucket:
    name: str
    elapsed_s: float
    role: str = "explanatory_only"
    nonclaims: tuple[str, ...] = (
        "timing is not a validity or promotion criterion",
    )


VALID_REDUCER_STATUSES = (
    "complete",
    "partial",
    "stale",
    "invalid",
    "failed",
    "timed_out",
    "missing",
)

TIMING_BUCKET_NAMES = (
    "target",
    "value_score",
    "filter",
    "map_mass",
    "compile",
    "warm_call",
    "hmc_kernel",
    "trace",
    "artifact_overhead",
)

_PROCESS_LOCAL_FIELD_PATTERNS = (
    re.compile(r"\b0x[0-9a-fA-F]+\b"),
    re.compile(r"\bobject at\b"),
    re.compile(r"\bid\s*\("),
)

EVIDENCE_MANIFEST_SCOPES = ("hmc", "target_only", "no_hmc_parity")


@dataclass(frozen=True)
class EvidenceManifest:
    """JSON-stable evidence wrapper around runtime manifest authority.

    This schema is deliberately target-agnostic.  It records enough provenance
    for HMC, target-only, and no-HMC parity artifacts while leaving client
    projects free to attach domain-specific details under separate result
    payloads.
    """

    run_scope: str
    git_state: Mapping[str, Any]
    command: tuple[str, ...]
    environment: Mapping[str, Any]
    cpu_gpu_status: str
    data_hash: str
    target_scope: str
    backend: str
    transform_signature: Mapping[str, Any]
    map_covariance_source: str
    tuning_policy: str
    diagnostic_policy: str
    result_paths: Mapping[str, str]
    nonclaims: tuple[str, ...]
    run_manifest: RunManifest | None = None
    worker_manifest: WorkerManifest | None = None

    def __post_init__(self) -> None:
        run_scope = str(self.run_scope)
        if run_scope not in EVIDENCE_MANIFEST_SCOPES:
            raise ValueError(
                "run_scope must be one of " + ", ".join(EVIDENCE_MANIFEST_SCOPES)
            )
        required_text_fields = {
            "cpu_gpu_status": self.cpu_gpu_status,
            "data_hash": self.data_hash,
            "target_scope": self.target_scope,
            "backend": self.backend,
            "map_covariance_source": self.map_covariance_source,
            "tuning_policy": self.tuning_policy,
            "diagnostic_policy": self.diagnostic_policy,
        }
        for name, value in required_text_fields.items():
            text = str(value)
            if not text:
                raise ValueError(f"{name} must be non-empty")
            _reject_process_local_text(text, name)
            object.__setattr__(self, name, text)
        command = tuple(str(part) for part in self.command)
        if not command:
            raise ValueError("command must be non-empty")
        for index, part in enumerate(command):
            _reject_process_local_text(part, f"command[{index}]")
        result_paths = {str(key): str(value) for key, value in self.result_paths.items()}
        if not result_paths:
            raise ValueError("result_paths must be non-empty")
        for key, value in result_paths.items():
            if not value:
                raise ValueError("result path values must be non-empty")
            _reject_process_local_text(key, "result_paths key")
            _reject_process_local_text(value, f"result_paths[{key}]")
        nonclaims = tuple(str(item) for item in self.nonclaims)
        if not nonclaims:
            raise ValueError("nonclaims must be non-empty")
        for index, item in enumerate(nonclaims):
            _reject_process_local_text(item, f"nonclaims[{index}]")
        normalized = {
            "git_state": _normalize_for_json(self.git_state),
            "environment": _normalize_for_json(self.environment),
            "transform_signature": _normalize_for_json(self.transform_signature),
        }
        _reject_process_local_values(normalized)
        object.__setattr__(self, "run_scope", run_scope)
        object.__setattr__(self, "command", command)
        object.__setattr__(self, "git_state", normalized["git_state"])
        object.__setattr__(self, "environment", normalized["environment"])
        object.__setattr__(self, "transform_signature", normalized["transform_signature"])
        object.__setattr__(self, "result_paths", result_paths)
        object.__setattr__(self, "nonclaims", nonclaims)

    @property
    def manifest_hash(self) -> str:
        return stable_config_hash(self.payload())

    def payload(self) -> Mapping[str, Any]:
        payload = {
            "artifact_type": "bayesfilter_evidence_manifest",
            "run_scope": self.run_scope,
            "git_state": self.git_state,
            "command": self.command,
            "environment": self.environment,
            "cpu_gpu_status": self.cpu_gpu_status,
            "data_hash": self.data_hash,
            "target_scope": self.target_scope,
            "backend": self.backend,
            "transform_signature": self.transform_signature,
            "map_covariance_source": self.map_covariance_source,
            "tuning_policy": self.tuning_policy,
            "diagnostic_policy": self.diagnostic_policy,
            "result_paths": self.result_paths,
            "nonclaims": self.nonclaims,
            "run_manifest": _as_payload(self.run_manifest) if self.run_manifest is not None else None,
            "worker_manifest": (
                _as_payload(self.worker_manifest) if self.worker_manifest is not None else None
            ),
        }
        normalized = _normalize_for_json(payload)
        _reject_process_local_values(normalized)
        return normalized

    def markdown_note(self) -> str:
        lines = [
            "# BayesFilter Evidence Manifest",
            "",
            f"- run scope: `{self.run_scope}`",
            f"- target scope: `{self.target_scope}`",
            f"- backend: `{self.backend}`",
            f"- diagnostic policy: `{self.diagnostic_policy}`",
            f"- manifest hash: `{self.manifest_hash}`",
            "",
            "## Nonclaims",
            "",
        ]
        lines.extend(f"- {item}" for item in self.nonclaims)
        lines.append("")
        return "\n".join(lines)


def stable_config_hash(config: Mapping[str, Any] | Any) -> str:
    normalized = _normalize_for_json(config)
    blob = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def configs_match_exact(expected: Mapping[str, Any] | Any, observed: Mapping[str, Any] | Any) -> bool:
    return stable_config_hash(expected) == stable_config_hash(observed)


def stale_match_payload(
    normalized_config: Mapping[str, Any] | Any,
    *,
    program_signature: str,
    device_policy: Mapping[str, Any] | Any,
    thread_caps: Mapping[str, Any] | Any,
    worker_config_hash: str,
) -> Mapping[str, Any]:
    return {
        "normalized_config": _normalize_for_json(normalized_config),
        "program_signature": str(program_signature),
        "device_policy": _normalize_for_json(device_policy),
        "thread_caps": _normalize_for_json(thread_caps),
        "worker_config_hash": str(worker_config_hash),
    }


def stale_artifacts_match_exact(
    expected: Mapping[str, Any] | Any,
    observed: Mapping[str, Any] | Any,
) -> bool:
    return configs_match_exact(expected, observed)


def build_worker_manifest(
    *,
    worker_id: str,
    command: tuple[str, ...] | list[str],
    git_commit: str,
    artifact_root: str | Path,
    normalized_config: Mapping[str, Any] | Any,
    program_signature: str,
    device_policy: Mapping[str, Any] | Any,
    thread_caps: Mapping[str, Any] | Any,
    worker_config: Mapping[str, Any] | Any,
    pid: int | None = None,
    environment: Mapping[str, Any] | None = None,
) -> WorkerManifest:
    return WorkerManifest(
        worker_id=str(worker_id),
        command=tuple(str(part) for part in command),
        pid=None if pid is None else int(pid),
        git_commit=str(git_commit),
        artifact_root=str(artifact_root),
        normalized_config=_normalize_for_json(normalized_config),
        program_signature=str(program_signature),
        device_policy=_normalize_for_json(device_policy),
        thread_caps=_normalize_for_json(thread_caps),
        worker_config=_normalize_for_json(worker_config),
        environment=_normalize_for_json(environment or {}),
    )


def atomic_write_json(path: str | Path, payload: Mapping[str, Any] | Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    data = _normalize_for_json(_as_payload(payload))
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=str(destination.parent),
        delete=False,
    ) as handle:
        json.dump(data, handle, sort_keys=True, indent=2)
        handle.write("\n")
        temp_name = handle.name
    os.replace(temp_name, destination)


def append_jsonl(path: str | Path, payload: Mapping[str, Any] | Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    data = _normalize_for_json(_as_payload(payload))
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(data, sort_keys=True, separators=(",", ":")))
        handle.write("\n")


def write_worker_manifest(path: str | Path, manifest: WorkerManifest) -> None:
    payload = _as_payload(manifest)
    payload["config_hash"] = manifest.config_hash
    payload["worker_config_hash"] = manifest.worker_config_hash
    payload["stale_match_payload"] = manifest.stale_match_payload
    atomic_write_json(path, payload)


def write_evidence_manifest(path: str | Path, manifest: EvidenceManifest) -> None:
    payload = dict(manifest.payload())
    payload["manifest_hash"] = manifest.manifest_hash
    atomic_write_json(path, payload)


def append_stage_event(
    path: str | Path,
    *,
    stage: str,
    status: str,
    timestamp: str,
    payload: Mapping[str, Any] | None = None,
) -> StageEvent:
    event = StageEvent(
        stage=str(stage),
        status=str(status),
        timestamp=str(timestamp),
        payload=_normalize_for_json(payload or {}),
    )
    append_jsonl(path, event)
    return event


def append_heartbeat(
    path: str | Path,
    *,
    worker_id: str,
    timestamp: str,
    stage: str,
    payload: Mapping[str, Any] | None = None,
) -> StageEvent:
    heartbeat_payload = {"worker_id": str(worker_id), **dict(payload or {})}
    return append_stage_event(
        path,
        stage=str(stage),
        status="heartbeat",
        timestamp=str(timestamp),
        payload=heartbeat_payload,
    )


def write_partial_result_snapshot(
    path: str | Path,
    snapshot: PartialResultSnapshot,
) -> None:
    atomic_write_json(path, snapshot)


def record_timeout(
    path: str | Path,
    *,
    worker_id: str,
    timeout_s: float,
    elapsed_s: float,
    action: str,
) -> TimeoutRecord:
    record = TimeoutRecord(
        worker_id=str(worker_id),
        timeout_s=float(timeout_s),
        elapsed_s=float(elapsed_s),
        action=str(action),
    )
    atomic_write_json(path, record)
    return record


def record_worker_result(path: str | Path, record: WorkerRecord) -> None:
    atomic_write_json(path, record)


def make_timing_bucket(name: str, elapsed_s: float) -> TimingBucket:
    if name not in TIMING_BUCKET_NAMES:
        raise ValueError(f"unknown timing bucket {name!r}")
    return TimingBucket(name=str(name), elapsed_s=float(elapsed_s))


def reduce_worker_artifacts(
    *,
    worker_id: str,
    expected_stale_payload: Mapping[str, Any] | Any,
    manifest_path: str | Path | None = None,
    result_path: str | Path | None = None,
    partial_path: str | Path | None = None,
    timeout_path: str | Path | None = None,
) -> ReducerRowStatus:
    if timeout_path is not None and Path(timeout_path).exists():
        timeout_payload = _read_json(timeout_path)
        return ReducerRowStatus(
            worker_id=str(worker_id),
            status="timed_out",
            reason="timeout_record_present",
            artifact_path=str(timeout_path),
            timed_out=True,
            return_code=timeout_payload.get("return_code"),
        )

    if manifest_path is None or not Path(manifest_path).exists():
        if partial_path is not None and Path(partial_path).exists():
            return ReducerRowStatus(
                worker_id=str(worker_id),
                status="partial",
                reason="partial_present_manifest_missing",
                artifact_path=str(partial_path),
            )
        return ReducerRowStatus(
            worker_id=str(worker_id),
            status="missing",
            reason="manifest_missing",
            artifact_path=None,
        )

    manifest = _read_json(manifest_path)
    observed_stale_payload = manifest.get("stale_match_payload")
    if not stale_artifacts_match_exact(expected_stale_payload, observed_stale_payload):
        return ReducerRowStatus(
            worker_id=str(worker_id),
            status="stale",
            reason="stale_match_payload_mismatch",
            artifact_path=str(manifest_path),
            stale=True,
        )

    if result_path is not None and Path(result_path).exists():
        result = _read_json(result_path)
        return_code = result.get("return_code")
        timed_out = bool(result.get("timed_out", False))
        status = result.get("status")
        if timed_out:
            reducer_status = "timed_out"
            reason = "worker_record_timed_out"
        elif return_code not in (None, 0):
            reducer_status = "failed"
            reason = "worker_return_code_nonzero"
        elif status in (None, "ok", "complete", "success"):
            reducer_status = "complete"
            reason = "complete_result_present"
        else:
            reducer_status = "invalid"
            reason = "worker_record_status_invalid"
        return ReducerRowStatus(
            worker_id=str(worker_id),
            status=reducer_status,
            reason=reason,
            artifact_path=str(result_path),
            return_code=return_code,
            timed_out=timed_out,
            stale=False,
        )

    if partial_path is not None and Path(partial_path).exists():
        return ReducerRowStatus(
            worker_id=str(worker_id),
            status="partial",
            reason="partial_present_result_missing",
            artifact_path=str(partial_path),
        )

    return ReducerRowStatus(
        worker_id=str(worker_id),
        status="invalid",
        reason="manifest_present_result_missing",
        artifact_path=str(manifest_path),
    )


def _as_payload(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    return value


def _read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _normalize_for_json(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return _normalize_for_json(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _normalize_for_json(val) for key, val in value.items()}
    if isinstance(value, (tuple, list)):
        return [_normalize_for_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return _normalize_for_json(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def _reject_process_local_text(value: str, field_name: str) -> None:
    if any(pattern.search(value) for pattern in _PROCESS_LOCAL_FIELD_PATTERNS):
        raise ValueError(f"{field_name} must not contain process-local object identity")


def _reject_process_local_values(value: Any, field_name: str = "payload") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _reject_process_local_text(str(key), f"{field_name} key")
            _reject_process_local_values(item, f"{field_name}.{key}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _reject_process_local_values(item, f"{field_name}[{index}]")
        return
    if isinstance(value, str):
        _reject_process_local_text(value, field_name)
