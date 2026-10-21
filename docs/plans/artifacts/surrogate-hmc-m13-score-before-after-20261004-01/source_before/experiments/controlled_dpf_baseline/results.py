"""Result schema, summaries, validation, and report writers."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from experiments.controlled_dpf_baseline.metrics import REQUIRED_METRICS


IMPLEMENTATION_NAME = "clean_room_regularized_particle_flow_pf"
IMPLEMENTATION_TYPE = "clean_room_controlled_baseline"

ALLOWED_STATUSES = {"ok", "blocked", "failed"}
ALLOWED_FAILURE_REASONS = {
    "blocked_missing_dependency",
    "blocked_environment_drift",
    "blocked_missing_assumption",
    "blocked_runtime_limit",
    "blocked_schema_validation",
    "failed_algorithmic",
    "failed_nonfinite_output",
    "failed_metric_contract",
    "failed_unexpected_exception",
}


def make_target(
    *,
    grid: str,
    fixture_name: str,
    num_particles: int,
    flow_steps: int,
) -> dict[str, Any]:
    return {
        "grid": grid,
        "fixture_name": fixture_name,
        "num_particles": int(num_particles),
        "flow_steps": int(flow_steps),
    }


def make_ok_record(
    *,
    fixture_name: str,
    seed: int,
    num_particles: int,
    flow_steps: int,
    horizon: int,
    target: dict[str, Any],
    runtime_seconds: float,
    metrics: dict[str, Any],
    diagnostics: dict[str, Any],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    return {
        "implementation_name": IMPLEMENTATION_NAME,
        "implementation_type": IMPLEMENTATION_TYPE,
        "fixture_name": fixture_name,
        "target": target,
        "status": "ok",
        "failure_reason": None,
        "seed": int(seed),
        "num_particles": int(num_particles),
        "flow_steps": int(flow_steps),
        "horizon": int(horizon),
        "runtime_seconds": float(runtime_seconds),
        "metrics": to_jsonable(metrics),
        "diagnostics": to_jsonable(diagnostics),
        "provenance": provenance,
    }


def make_failure_record(
    *,
    status: str,
    failure_reason: str,
    fixture_name: str,
    seed: int,
    num_particles: int,
    flow_steps: int,
    horizon: int | None,
    target: dict[str, Any],
    runtime_seconds: float,
    diagnostics: dict[str, Any],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    if status not in {"blocked", "failed"}:
        raise ValueError(f"failure status must be blocked or failed, got {status!r}")
    if failure_reason not in ALLOWED_FAILURE_REASONS:
        raise ValueError(f"unknown failure_reason {failure_reason!r}")
    return {
        "implementation_name": IMPLEMENTATION_NAME,
        "implementation_type": IMPLEMENTATION_TYPE,
        "fixture_name": fixture_name,
        "target": target,
        "status": status,
        "failure_reason": failure_reason,
        "seed": int(seed),
        "num_particles": int(num_particles),
        "flow_steps": int(flow_steps),
        "horizon": None if horizon is None else int(horizon),
        "runtime_seconds": float(runtime_seconds),
        "metrics": {},
        "diagnostics": to_jsonable(diagnostics),
        "provenance": provenance,
    }


def summarize_records(
    records: list[dict[str, Any]],
    *,
    runtime_warning_seconds: float,
    expected_records: int | None = None,
) -> dict[str, Any]:
    status_counts = Counter(record.get("status") for record in records)
    blockers = [
        {
            "fixture_name": record.get("fixture_name"),
            "seed": record.get("seed"),
            "num_particles": record.get("num_particles"),
            "flow_steps": record.get("flow_steps"),
            "status": record.get("status"),
            "failure_reason": record.get("failure_reason"),
            "runtime_seconds": record.get("runtime_seconds"),
        }
        for record in records
        if record.get("status") != "ok"
    ]
    grouped = _group_ok_records(records)
    return {
        "implementation_name": IMPLEMENTATION_NAME,
        "implementation_type": IMPLEMENTATION_TYPE,
        "planned_records": len(records),
        "expected_records": expected_records,
        "ok_records": int(status_counts.get("ok", 0)),
        "blocked_records": int(status_counts.get("blocked", 0)),
        "failed_records": int(status_counts.get("failed", 0)),
        "fixed_grid_records": int(
            sum(1 for record in records if record.get("target", {}).get("grid") == "first-target")
        ),
        "smoke_records": int(
            sum(1 for record in records if record.get("target", {}).get("grid") == "smoke")
        ),
        "runtime_warning_seconds": float(runtime_warning_seconds),
        "runtime_warning_count": int(
            sum(
                1
                for record in records
                if float(record.get("runtime_seconds") or 0.0)
                > float(runtime_warning_seconds)
            )
        ),
        "status_counts": dict(status_counts),
        "metric_medians": grouped,
        "finite_success_metric_count": int(
            sum(_record_required_metrics_are_finite(record) for record in records)
        ),
        "structured_blockers": blockers,
    }


def validate_records(
    records: list[dict[str, Any]],
    *,
    expected_records: int,
    require_finite_success_metrics: bool,
    require_smoke_only: bool = False,
    require_fixed_grid: bool = False,
) -> list[str]:
    errors: list[str] = []
    if len(records) != expected_records:
        errors.append(f"expected {expected_records} records, found {len(records)}")
    for i, record in enumerate(records):
        errors.extend(_validate_record_schema(record, index=i))
        if require_finite_success_metrics and record.get("status") == "ok":
            if not _record_required_metrics_are_finite(record):
                errors.append(f"record {i} has missing or nonfinite required metrics")
    if require_smoke_only:
        grids = {record.get("target", {}).get("grid") for record in records}
        if grids != {"smoke"}:
            errors.append(f"smoke validation expected only smoke grid, found {sorted(grids)}")
    if require_fixed_grid:
        errors.extend(_validate_fixed_grid(records))
    return errors


def read_records(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("records"), list):
        return payload["records"]
    raise ValueError(f"records file {path} must contain a list or object with records")


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_markdown_report(
    path: str | Path,
    *,
    title: str,
    decision: str,
    summary: dict[str, Any],
    records: list[dict[str, Any]],
    notes: list[str] | None = None,
) -> None:
    lines = [
        f"# {title}",
        "",
        f"Decision: `{decision}`.",
        "",
        "## Summary",
        "",
        f"- planned records: `{summary.get('planned_records')}`",
        f"- ok records: `{summary.get('ok_records')}`",
        f"- blocked records: `{summary.get('blocked_records')}`",
        f"- failed records: `{summary.get('failed_records')}`",
        f"- runtime warnings: `{summary.get('runtime_warning_count')}`",
        "",
        "## Records",
        "",
        "| Fixture | Seed | N | Flow steps | Status | Position RMSE | Observation proxy RMSE | Runtime seconds |",
        "| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    for record in records:
        metrics = record.get("metrics", {})
        lines.append(
            "| {fixture} | {seed} | {n} | {steps} | `{status}` | {pos} | {obs} | {runtime} |".format(
                fixture=record.get("fixture_name"),
                seed=record.get("seed"),
                n=record.get("num_particles"),
                steps=record.get("flow_steps"),
                status=record.get("status"),
                pos=_format_float(metrics.get("position_rmse")),
                obs=_format_float(metrics.get("observation_proxy_rmse")),
                runtime=_format_float(record.get("runtime_seconds")),
            )
        )
    if notes:
        lines.extend(["", "## Notes", ""])
        lines.extend(f"- {note}" for note in notes)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(val) for key, val in value.items()}
    if isinstance(value, list | tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _group_ok_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record.get("status") != "ok":
            continue
        key = "{fixture}/N{n}/steps{steps}".format(
            fixture=record.get("fixture_name"),
            n=record.get("num_particles"),
            steps=record.get("flow_steps"),
        )
        grouped[key].append(record)
    summaries: dict[str, Any] = {}
    for key, group in grouped.items():
        summaries[key] = {
            metric: _median([record.get("metrics", {}).get(metric) for record in group])
            for metric in REQUIRED_METRICS
        }
        summaries[key]["runs"] = len(group)
        summaries[key]["median_average_ess"] = _median(
            [record.get("diagnostics", {}).get("average_ess") for record in group]
        )
        summaries[key]["median_min_ess"] = _median(
            [record.get("diagnostics", {}).get("min_ess") for record in group]
        )
        summaries[key]["median_resampling_count"] = _median(
            [record.get("diagnostics", {}).get("resampling_count") for record in group]
        )
    return summaries


def _validate_record_schema(record: dict[str, Any], *, index: int) -> list[str]:
    errors: list[str] = []
    required = [
        "implementation_name",
        "implementation_type",
        "fixture_name",
        "target",
        "status",
        "failure_reason",
        "seed",
        "num_particles",
        "flow_steps",
        "horizon",
        "runtime_seconds",
        "metrics",
        "diagnostics",
        "provenance",
    ]
    for key in required:
        if key not in record:
            errors.append(f"record {index} missing {key}")
    status = record.get("status")
    if status not in ALLOWED_STATUSES:
        errors.append(f"record {index} has invalid status {status!r}")
    failure_reason = record.get("failure_reason")
    if status == "ok" and failure_reason is not None:
        errors.append(f"record {index} status ok must have null failure_reason")
    if status != "ok" and failure_reason not in ALLOWED_FAILURE_REASONS:
        errors.append(f"record {index} has invalid failure_reason {failure_reason!r}")
    return errors


def _validate_fixed_grid(records: list[dict[str, Any]]) -> list[str]:
    expected = {
        ("range_bearing_gaussian_low_noise", 128, 20, seed)
        for seed in (31, 43, 59, 71, 83)
    }
    expected |= {
        ("range_bearing_gaussian_moderate", 128, steps, seed)
        for steps in (10, 20)
        for seed in (31, 43, 59, 71, 83)
    }
    observed = [
        (
            record.get("fixture_name"),
            record.get("num_particles"),
            record.get("flow_steps"),
            record.get("seed"),
        )
        for record in records
    ]
    errors: list[str] = []
    observed_set = set(observed)
    if len(observed) != len(observed_set):
        errors.append("fixed-grid records contain duplicate cells")
    missing = sorted(expected - observed_set)
    extra = sorted(observed_set - expected)
    if missing:
        errors.append(f"fixed-grid records missing cells: {missing}")
    if extra:
        errors.append(f"fixed-grid records contain extra cells: {extra}")
    for i, record in enumerate(records):
        if record.get("target", {}).get("grid") != "first-target":
            errors.append(f"record {i} is not labeled first-target")
    return errors


def _record_required_metrics_are_finite(record: dict[str, Any]) -> bool:
    if record.get("status") != "ok":
        return False
    metrics = record.get("metrics", {})
    for key in REQUIRED_METRICS:
        value = metrics.get(key)
        if not isinstance(value, int | float) or not np.isfinite(value):
            return False
    finite_outputs = metrics.get("finite_outputs", {})
    if finite_outputs.get("means_finite") is not True:
        return False
    if finite_outputs.get("scalar_metrics_finite") is not True:
        return False
    if finite_outputs.get("particles_finite") is False:
        return False
    if finite_outputs.get("covariances_finite") is False:
        return False
    return True


def _median(values: list[Any]) -> float | None:
    finite = [float(value) for value in values if isinstance(value, int | float) and np.isfinite(value)]
    if not finite:
        return None
    return float(np.median(finite))


def _format_float(value: Any) -> str:
    if not isinstance(value, int | float) or not np.isfinite(value):
        return ""
    return f"{float(value):.6g}"
