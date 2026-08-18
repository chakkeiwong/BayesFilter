"""Device-policy helpers that avoid hidden framework initialization."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any, Iterable


_FRAMEWORK_MODULES = ("tensorflow", "tensorflow_probability", "jax", "torch")
GPU_MEMORY_GROWTH_ENV = "TF_FORCE_GPU_ALLOW_GROWTH"


@dataclass(frozen=True)
class GPUSelection:
    selected_gpu: int | None
    reason: str
    visible_devices: tuple[int, ...]
    busy_devices: tuple[int, ...]
    trusted_or_escalated: bool = False
    fallback_used: bool = False
    preferred_gpu_busy: bool | None = None
    veto_reasons: tuple[str, ...] = ()


def ensure_cpu_only_env(env: dict[str, str] | None = None) -> dict[str, str]:
    """Set CUDA_VISIBLE_DEVICES=-1 before TensorFlow/JAX/PyTorch import."""

    target_env = os.environ if env is None else env
    current_process_env = env is None or target_env is os.environ
    if (
        current_process_env
        and _framework_already_imported()
        and target_env.get("CUDA_VISIBLE_DEVICES") != "-1"
    ):
        raise RuntimeError(
            "CPU-only mode must set CUDA_VISIBLE_DEVICES=-1 before framework import"
        )
    target_env["CUDA_VISIBLE_DEVICES"] = "-1"
    return target_env


def assert_cpu_only_env(env: dict[str, str] | None = None) -> None:
    target_env = os.environ if env is None else env
    if target_env.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise RuntimeError("CUDA_VISIBLE_DEVICES=-1 is required for CPU-only workers")


def ensure_gpu_memory_growth_env(env: dict[str, str] | None = None) -> dict[str, str]:
    """Require TensorFlow GPU memory growth before framework import."""

    target_env = os.environ if env is None else env
    current_process_env = env is None or target_env is os.environ
    if current_process_env and _framework_already_imported():
        if target_env.get(GPU_MEMORY_GROWTH_ENV, "").lower() != "true":
            raise RuntimeError(
                "GPU memory growth must be enabled before framework import"
            )
    target_env[GPU_MEMORY_GROWTH_ENV] = "true"
    return target_env


def assert_gpu_memory_growth_env(env: dict[str, str] | None = None) -> None:
    target_env = os.environ if env is None else env
    if target_env.get(GPU_MEMORY_GROWTH_ENV, "").lower() != "true":
        raise RuntimeError("TF_FORCE_GPU_ALLOW_GROWTH=true is required for GPU workers")


def select_preferred_gpu(
    available_gpu_ids: Iterable[int],
    *,
    busy_gpu_ids: Iterable[int] = (),
    preferred_gpu: int = 1,
    fallback_gpu: int = 0,
    trusted_or_escalated: bool = False,
    gpu_snapshot: dict[str, Any] | None = None,
    busy_memory_fraction: float = 0.50,
    busy_utilization_pct: float = 10.0,
    minimum_free_memory_mb: float | None = None,
) -> GPUSelection:
    """Pure metadata selector: prefer GPU1 and fallback only on trusted evidence."""

    trusted = bool(trusted_or_escalated) or _snapshot_trusted(gpu_snapshot)
    available = _normalize_ids(available_gpu_ids)
    supplied_busy = _normalize_ids(busy_gpu_ids)
    snapshot_available, snapshot_busy = _snapshot_devices(
        gpu_snapshot,
        busy_memory_fraction=busy_memory_fraction,
        busy_utilization_pct=busy_utilization_pct,
        minimum_free_memory_mb=minimum_free_memory_mb,
    )
    veto_reasons: list[str] = []
    if gpu_snapshot is not None and not _snapshot_trusted(gpu_snapshot):
        veto_reasons.append("gpu_snapshot_not_trusted")
    if supplied_busy and not trusted:
        veto_reasons.append("gpu_busy_evidence_not_trusted")
    if snapshot_available and trusted:
        available = snapshot_available
    busy = tuple(sorted(set(supplied_busy) | set(snapshot_busy))) if trusted else ()
    preferred = int(preferred_gpu)
    fallback = int(fallback_gpu)
    preferred_busy = preferred in set(busy) if trusted else None
    visible = set(available)

    if available and preferred not in visible:
        if trusted and fallback in visible and fallback not in set(busy):
            return GPUSelection(
                fallback,
                "preferred_gpu_not_visible_fallback_selected",
                available,
                busy,
                trusted,
                True,
                None,
                tuple(veto_reasons),
            )
        veto_reasons.append("preferred_gpu_not_visible")
        return GPUSelection(
            None,
            "preferred_gpu_not_visible",
            available,
            busy if trusted else supplied_busy,
            trusted,
            False,
            preferred_busy,
            tuple(veto_reasons),
        )
    if trusted and preferred_busy is True:
        if fallback in visible and fallback not in set(busy):
            return GPUSelection(
                fallback,
                "preferred_gpu_busy_fallback_selected",
                available,
                busy,
                trusted,
                True,
                preferred_busy,
                tuple(veto_reasons),
            )
        veto_reasons.append("fallback_gpu_not_available")
        return GPUSelection(
            None,
            "preferred_gpu_busy_no_fallback",
            available,
            busy,
            trusted,
            False,
            preferred_busy,
            tuple(veto_reasons),
        )
    return GPUSelection(
        preferred,
        "preferred_gpu_selected_trusted"
        if trusted
        else "preferred_gpu_selected_no_trusted_busy_evidence",
        available,
        busy if trusted else supplied_busy,
        trusted,
        False,
        preferred_busy,
        tuple(veto_reasons),
    )


def build_trusted_gpu_snapshot(
    gpus: Iterable[dict[str, Any]],
    *,
    trusted_or_escalated: bool,
    source: str,
) -> dict[str, Any]:
    """Build a selector snapshot without probing hardware or importing frameworks."""

    return {
        "artifact_type": "bayesfilter_trusted_gpu_snapshot",
        "trusted_or_escalated": bool(trusted_or_escalated),
        "source": str(source),
        "gpus": [dict(row) for row in gpus],
    }


def _framework_already_imported() -> bool:
    return any(name in sys.modules for name in _FRAMEWORK_MODULES)


def _normalize_ids(values: Iterable[int]) -> tuple[int, ...]:
    return tuple(sorted({int(value) for value in values}))


def _snapshot_trusted(snapshot: dict[str, Any] | None) -> bool:
    return isinstance(snapshot, dict) and snapshot.get("trusted_or_escalated") is True


def _snapshot_devices(
    snapshot: dict[str, Any] | None,
    *,
    busy_memory_fraction: float,
    busy_utilization_pct: float,
    minimum_free_memory_mb: float | None,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if not isinstance(snapshot, dict):
        return (), ()
    rows = snapshot.get("gpus", snapshot.get("devices", ()))
    if not isinstance(rows, list):
        return (), ()
    available: list[int] = []
    busy: list[int] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        device = _row_device_id(row)
        if device is None:
            continue
        available.append(device)
        explicit_busy = row.get("busy")
        if explicit_busy is not None:
            if bool(explicit_busy):
                busy.append(device)
            continue
        memory_fraction = _memory_fraction(row)
        utilization = _utilization_pct(row)
        free_memory = _free_memory_mb(row)
        if (
            memory_fraction is not None
            and memory_fraction >= float(busy_memory_fraction)
        ) or (
            utilization is not None
            and utilization >= float(busy_utilization_pct)
        ) or (
            minimum_free_memory_mb is not None
            and (
                free_memory is None
                or free_memory <= float(minimum_free_memory_mb)
            )
        ):
            busy.append(device)
    return _normalize_ids(available), _normalize_ids(busy)


def _row_device_id(row: dict[str, Any]) -> int | None:
    for key in ("index", "id", "gpu", "minor_number"):
        value = row.get(key)
        if value is None:
            continue
        text = str(value)
        if text.startswith("GPU:"):
            text = text.split(":", 1)[1]
        try:
            return int(text)
        except ValueError:
            continue
    return None


def _memory_fraction(row: dict[str, Any]) -> float | None:
    used = row.get("memory_used_mb", row.get("memory.used"))
    total = row.get("memory_total_mb", row.get("memory.total"))
    try:
        used_f = float(used)
        total_f = float(total)
    except (TypeError, ValueError):
        return None
    if total_f <= 0:
        return None
    return max(0.0, used_f / total_f)


def _utilization_pct(row: dict[str, Any]) -> float | None:
    value = row.get("utilization_gpu_pct", row.get("utilization.gpu"))
    if value is None:
        value = row.get("gpu_utilization_pct")
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _free_memory_mb(row: dict[str, Any]) -> float | None:
    value = row.get("memory_free_mb", row.get("memory.free"))
    if value is None:
        value = row.get("free_memory_mb")
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
