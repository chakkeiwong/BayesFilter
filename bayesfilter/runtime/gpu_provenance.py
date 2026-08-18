"""Stable NVIDIA UUID provenance for GPU launch and diagnostic artifacts."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Iterable, Mapping
from typing import Any


class GPUProvenanceError(RuntimeError):
    """Raised when a stable selected physical GPU cannot be established."""


def parse_nvidia_smi_rows(output: str) -> tuple[dict[str, object], ...]:
    """Parse the fixed CSV query used by :func:`query_nvidia_smi_gpus`."""

    rows = []
    for line in output.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 8:
            raise GPUProvenanceError(f"unexpected nvidia-smi row: {line!r}")
        index, uuid, name, pci_bus_id, utilization, used, free, total = fields
        rows.append(
            {
                "nvidia_smi_index": int(index),
                "uuid": uuid,
                "name": name,
                "pci_bus_id": pci_bus_id,
                "utilization_gpu_pct": float(utilization),
                "memory_used_mib": float(used),
                "memory_free_mib": float(free),
                "memory_total_mib": float(total),
            }
        )
    return tuple(rows)


def query_nvidia_smi_gpus() -> tuple[dict[str, object], ...]:
    """Return a live physical-GPU snapshot without importing a framework."""

    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,pci.bus_id,utilization.gpu,memory.used,memory.free,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return parse_nvidia_smi_rows(completed.stdout)


def selected_nvidia_gpu(
    rows: Iterable[Mapping[str, Any]],
    *,
    cuda_visible_devices: str | None = None,
) -> dict[str, object]:
    """Resolve one UUID-pinned CUDA selection against a physical snapshot."""

    selected = (
        os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if cuda_visible_devices is None
        else str(cuda_visible_devices)
    )
    if not selected.startswith("GPU-") or "," in selected:
        raise GPUProvenanceError(
            "CUDA_VISIBLE_DEVICES must contain exactly one stable NVIDIA GPU UUID"
        )
    for row in rows:
        if str(row.get("uuid")) == selected:
            return dict(row)
    raise GPUProvenanceError(
        f"selected CUDA GPU UUID is absent from the NVIDIA snapshot: {selected}"
    )
