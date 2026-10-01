"""Framework-free, display-aware GPU placement for the Phase 9B campaign."""

from __future__ import annotations

import csv
import io
import math
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Any, Mapping


POLICY_ID = "bayesfilter_non_display_first_load40_headroom5g_v1"
MAX_UTILIZATION_PERCENT = 40.0
HEADROOM_MIB = 5120.0


class GPUPlacementError(RuntimeError):
    """The live device inventory cannot satisfy the placement policy."""


def _number(value: Any) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise ValueError("device telemetry must be finite and nonnegative")
    return parsed


def parse_inventory(csv_text: str, xml_text: str) -> list[dict[str, Any]]:
    """Join NVIDIA inventory indices to XML processes by UUID, not minor number."""
    xml_gpus = {node.findtext("uuid"): node for node in ET.fromstring(xml_text).findall("gpu")}
    rows = []
    for fields in csv.reader(io.StringIO(csv_text), skipinitialspace=True):
        if not fields:
            continue
        if len(fields) != 9:
            raise GPUPlacementError("unexpected NVIDIA inventory columns")
        index, uuid, name, bus, display, utilization, total, used, free = (
            field.strip() for field in fields
        )
        node = xml_gpus.get(uuid)
        if node is None:
            raise GPUPlacementError("NVIDIA CSV/XML identities disagree")
        attached = node.findtext("display_attached", "Unknown")
        rows.append({
            "index": int(index), "uuid": uuid, "name": name, "pci_bus_id": bus,
            "display_active": display, "display_attached": attached,
            "is_display": True if display == "Enabled" or attached == "Yes" else (
                False if display == "Disabled" and attached == "No" else None
            ),
            "utilization_gpu_pct": utilization,
            "memory_total_mib": total, "memory_used_mib": used, "memory_free_mib": free,
            "processes": [
                {key: process.findtext(key) for key in ("pid", "type", "process_name", "used_memory")}
                for process in node.findall("processes/process_info")
            ],
        })
    return rows


def probe_inventory() -> dict[str, Any]:
    """Caller must run with trusted/elevated device access."""
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,pci.bus_id,display_active,utilization.gpu,memory.total,memory.used,memory.free",
        "--format=csv,noheader,nounits",
    ]
    csv_text = subprocess.check_output(command, text=True, timeout=10)
    xml_text = subprocess.check_output(["nvidia-smi", "-q", "-x"], text=True, timeout=10)
    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "trust_basis": "trusted_escalated_gpu_execution",
        "commands": [command, ["nvidia-smi", "-q", "-x"]],
        "raw_csv": csv_text, "raw_xml": xml_text,
        "gpus": parse_inventory(csv_text, xml_text),
    }


def _validated_candidates(snapshot: Mapping[str, Any]) -> list[dict[str, Any]]:
    if snapshot.get("trust_basis") != "trusted_escalated_gpu_execution":
        raise GPUPlacementError("trusted GPU inventory is required")
    candidates = []
    identities: set[str] = set()
    indices: set[int] = set()
    for raw in snapshot["gpus"]:
        row = dict(raw)
        reasons = []
        uuid = row.get("uuid", "")
        index = row.get("index")
        if not isinstance(uuid, str) or not uuid.startswith("GPU-") or not isinstance(index, int):
            raise GPUPlacementError("missing physical GPU identity")
        if uuid in identities or index in indices:
            raise GPUPlacementError("duplicate physical GPU identity")
        identities.add(uuid)
        indices.add(index)
        if not isinstance(row.get("is_display"), bool):
            reasons.append("unknown_display_status")
        try:
            for field in ("utilization_gpu_pct", "memory_total_mib", "memory_used_mib", "memory_free_mib"):
                row[field] = _number(row[field])
            if row["utilization_gpu_pct"] > 100.0 or row["memory_total_mib"] <= 0.0:
                raise ValueError("invalid utilization or capacity")
            if row["memory_free_mib"] + row["memory_used_mib"] > row["memory_total_mib"]:
                raise ValueError("inconsistent memory telemetry")
            if row["utilization_gpu_pct"] > MAX_UTILIZATION_PERCENT:
                reasons.append("utilization_above_40_percent")
            if row["memory_free_mib"] < HEADROOM_MIB:
                reasons.append("insufficient_5g_free_headroom")
        except (KeyError, TypeError, ValueError):
            reasons.append("invalid_device_telemetry")
        row["rejection_reasons"] = reasons
        candidates.append(row)
    return candidates


def _selection_order(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    eligible = [row for row in candidates if not row["rejection_reasons"]]
    eligible.sort(key=lambda row: (
        row["is_display"], row["utilization_gpu_pct"], -row["memory_free_mib"], row["index"],
    ))
    return eligible


def select_gpus(
    snapshot: Mapping[str, Any], count: int, *, estimated_peak_mib: float = 0.0
) -> dict[str, Any]:
    """Select independent worker devices, preferring non-display GPUs.

    Count is a concurrency ceiling, not permission to fill idle slots using
    the display GPU. Excess work queues behind eligible non-display devices.
    """

    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise GPUPlacementError("worker GPU count must be a positive integer")
    try:
        peak = _number(estimated_peak_mib)
    except (TypeError, ValueError) as exc:
        raise GPUPlacementError("estimated worker peak must be finite and nonnegative") from exc
    candidates = _validated_candidates(snapshot)
    for row in candidates:
        if not row["rejection_reasons"] and row["memory_free_mib"] < HEADROOM_MIB + peak:
            row["rejection_reasons"].append("insufficient_worker_peak_plus_5g_headroom")
    eligible = _selection_order(candidates)
    non_display = [row for row in eligible if not row["is_display"]]
    selected = (non_display or eligible)[:count]
    if not selected:
        reason = "no_eligible_gpu"
    elif non_display:
        reason = "eligible_non_display_preferred"
    else:
        reason = "display_fallback_no_eligible_non_display"
    return {
        "schema": "bayesfilter.parallel_gpu_selection.v1", "policy_id": POLICY_ID,
        "snapshot": dict(snapshot), "candidates": candidates,
        "selected": selected,
        "selected_uuids": [row["uuid"] for row in selected],
        "requested_count": count,
        "reason": reason,
        "maximum_utilization_percent": MAX_UTILIZATION_PERCENT,
        "minimum_headroom_mib": HEADROOM_MIB,
        "estimated_worker_peak_mib": peak,
    }


def select_gpu(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    selection = select_gpus(snapshot, 1)
    selection["schema"] = "bayesfilter.display_gpu_selection.v1"
    selected = selection["selected"]
    if isinstance(selected, list):
        selected = selected[0] if selected else None
    selection["selected"] = selected
    selection["selected_uuids"] = [] if selected is None else [selected["uuid"]]
    if selected is None:
        selection["reason"] = "no_eligible_gpu"
    elif selected["is_display"]:
        selection["reason"] = "display_fallback_no_eligible_non_display"
    else:
        selection["reason"] = "eligible_non_display_preferred"
    return selection


def select_and_pin_gpu() -> dict[str, Any]:
    if any(name in sys.modules for name in ("tensorflow", "tensorflow_probability", "jax", "torch")):
        raise GPUPlacementError("GPU selection must precede accelerator framework import")
    if os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH", "").lower() != "true":
        raise GPUPlacementError("TF_FORCE_GPU_ALLOW_GROWTH=true must precede GPU selection")
    selection = select_gpu(probe_inventory())
    if selection["selected"] is None:
        raise GPUPlacementError(f"no eligible GPU: {selection['candidates']}")
    os.environ["CUDA_VISIBLE_DEVICES"] = selection["selected"]["uuid"]
    os.environ["BAYESFILTER_SELECTED_GPU_UUID"] = selection["selected"]["uuid"]
    os.environ["BAYESFILTER_GPU_SELECTION_POLICY_ID"] = POLICY_ID
    return selection


def check_runtime_headroom(selection: Mapping[str, Any], allocator_peak_bytes: int) -> dict[str, Any]:
    if not isinstance(allocator_peak_bytes, int) or allocator_peak_bytes < 0:
        raise GPUPlacementError("allocator peak must be a nonnegative integer")
    snapshot = probe_inventory()
    uuid = selection["selected"]["uuid"]
    matching = [row for row in snapshot["gpus"] if row["uuid"] == uuid]
    if len(matching) != 1 or _number(matching[0]["memory_free_mib"]) < HEADROOM_MIB:
        raise GPUPlacementError("selected GPU no longer has 5 GiB free headroom")
    return {"captured_at_utc": snapshot["captured_at_utc"], "selected": matching[0],
            "allocator_peak_bytes": allocator_peak_bytes, "status": "pass"}
