#!/usr/bin/env python3
"""Audit one factor-bound Phase 9A canary or full-run manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


FACTOR_BACKEND = "tensorflow_eigh_strict_factor_cached"
PLAN_PATH = (
    "docs/plans/"
    "bayesfilter-ssl-lstm-q20-factor-route-fresh-tuning-admission-plan-2026-09-04.md"
)
RESULT_PATH = (
    "docs/plans/"
    "bayesfilter-ssl-lstm-q20-factor-route-fresh-tuning-admission-result-2026-09-04.md"
)
ALLOCATOR_CAP_BYTES = 4 * 1024**3


class AuditError(RuntimeError):
    """Raised when a factor-admission artifact violates the plan."""


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _require(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def _roots(profile: Mapping[str, Any]) -> tuple[tuple[int, int], ...]:
    namespace = profile.get("seed_namespace", {})
    if not isinstance(namespace, Mapping):
        return ()
    rows: list[tuple[int, int]] = []
    for key in (
        "initialization_roots",
        "preflight_roots",
        "training_roots",
        "tuning_roots",
    ):
        value = namespace.get(key, ())
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            for row in value:
                if isinstance(row, Sequence) and len(row) == 2:
                    rows.append((int(row[0]), int(row[1])))
    for key in ("transition_root", "reliability_root"):
        row = namespace.get(key, ())
        if isinstance(row, Sequence) and len(row) == 2:
            rows.append((int(row[0]), int(row[1])))
    return tuple(rows)


def _audit_scope(
    row: Mapping[str, Any],
    *,
    expected_index: int,
    attempt_dir: Path,
    bridge_signature: str,
    failures: list[str],
) -> Mapping[str, Any]:
    prefix = f"scope[{expected_index}]"
    _require(row.get("scope_index") == expected_index, f"{prefix}: wrong index", failures)
    scope = row.get("scope", {})
    _require(isinstance(scope, Mapping), f"{prefix}: scope identity missing", failures)
    if isinstance(scope, Mapping):
        _require(
            scope.get("principal_sqrt_backend") == FACTOR_BACKEND,
            f"{prefix}: scope backend mismatch",
            failures,
        )
        _require(
            scope.get("bridge_backend") == FACTOR_BACKEND,
            f"{prefix}: bridge backend mismatch",
            failures,
        )
    result = row.get("tuning_result", {})
    _require(isinstance(result, Mapping), f"{prefix}: tuning result missing", failures)
    if not isinstance(result, Mapping):
        return {}
    _require(result.get("passed") is True, f"{prefix}: tuning did not pass", failures)
    _require(result.get("final_status") == "passed", f"{prefix}: final status not passed", failures)
    grid = result.get("fixed_grid_scale_selection", {})
    _require(isinstance(grid, Mapping), f"{prefix}: grid record missing", failures)
    if isinstance(grid, Mapping):
        _require(
            grid.get("tuning_policy") == "measured_joint_grid_v1",
            f"{prefix}: wrong tuning policy",
            failures,
        )
        _require(
            grid.get("all_declared_pairs_attempted") is True,
            f"{prefix}: incomplete attempted grid",
            failures,
        )
        _require(
            grid.get("all_declared_pairs_measured") is True,
            f"{prefix}: incomplete measured grid",
            failures,
        )
        _require(grid.get("candidate_count") == 8, f"{prefix}: expected eight pairs", failures)
        _require(
            len(grid.get("attempts", ())) == 8,
            f"{prefix}: grid attempt table is incomplete",
            failures,
        )
    selection = result.get("candidate_selection", {})
    _require(isinstance(selection, Mapping), f"{prefix}: selection record missing", failures)
    selected_index = None
    if isinstance(selection, Mapping):
        selected_index = selection.get("selected_candidate_index")
        _require(selection.get("final_status") == "passed", f"{prefix}: selection failed", failures)
        _require(
            selection.get("all_candidate_pairs_measured") is True,
            f"{prefix}: selection did not bind complete grid",
            failures,
        )
        seed_ledger = selection.get("seed_ledger", {})
        _require(
            isinstance(seed_ledger, Mapping)
            and seed_ledger.get("all_seeds_unique") is True,
            f"{prefix}: tuner seeds are not certified unique",
            failures,
        )
        selected_rows = [
            item
            for item in selection.get("candidate_rows", ())
            if isinstance(item, Mapping)
            and item.get("candidate_index") == selected_index
        ]
        _require(len(selected_rows) == 1, f"{prefix}: selected evidence row missing", failures)
        if len(selected_rows) == 1:
            _require(selected_rows[0].get("eligible") is True, f"{prefix}: selected pair ineligible", failures)
            _require(not selected_rows[0].get("hard_vetoes"), f"{prefix}: selected pair has hard veto", failures)
        heldout = selection.get("heldout_verification", {})
        _require(
            isinstance(heldout, Mapping) and heldout.get("final_status") == "passed",
            f"{prefix}: held-out verification failed",
            failures,
        )
        if isinstance(heldout, Mapping):
            _require(not heldout.get("hard_vetoes"), f"{prefix}: held-out hard veto", failures)
            diagnostics = heldout.get("diagnostics", {})
            _require(
                isinstance(diagnostics, Mapping)
                and diagnostics.get("all_chains_moved") is True,
                f"{prefix}: held-out chain movement failed",
                failures,
            )
            _require(
                isinstance(diagnostics, Mapping)
                and diagnostics.get("target_status_telemetry", {}).get("all_status_valid") is True,
                f"{prefix}: held-out target status failed",
                failures,
            )
    runner = result.get("full_chain_runner_evidence", {})
    _require(
        isinstance(runner, Mapping)
        and runner.get("all_runners_traced_exactly_once") is True,
        f"{prefix}: reusable runner tracing failed",
        failures,
    )
    handoff = row.get("handoff", {})
    _require(isinstance(handoff, Mapping), f"{prefix}: handoff missing", failures)
    if isinstance(handoff, Mapping):
        _require(
            handoff.get("base_adapter_signature") == result.get("base_adapter_signature"),
            f"{prefix}: handoff/base identity mismatch",
            failures,
        )
        _require(
            handoff.get("transformed_adapter_signature")
            == result.get("transformed_adapter_signature"),
            f"{prefix}: handoff/transformed identity mismatch",
            failures,
        )
        _require(
            handoff.get("handoff_hash") == row.get("handoff_hash"),
            f"{prefix}: handoff hash mismatch",
            failures,
        )
    artifact = Path(str(row.get("tuning_artifact", ""))).resolve()
    _require(artifact.is_file(), f"{prefix}: tuning artifact missing", failures)
    _require(
        artifact.is_relative_to(attempt_dir.resolve()),
        f"{prefix}: tuning artifact escapes attempt directory",
        failures,
    )
    return {
        "scope_index": expected_index,
        "chart_index": row.get("chart_index"),
        "beta": row.get("beta"),
        "selected_candidate_index": selected_index,
        "step_size": row.get("step_size"),
        "num_leapfrog_steps": row.get("num_leapfrog_steps"),
        "tuning_artifact": str(artifact),
        "tuning_artifact_sha256": _sha256(artifact) if artifact.is_file() else None,
        "bridge_signature": bridge_signature,
    }


def audit(attempt_dir: Path, role: str) -> Mapping[str, Any]:
    manifest_path = attempt_dir / "run_manifest.json"
    if not manifest_path.is_file():
        raise AuditError(f"missing run manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping):
        raise AuditError("run manifest must be a JSON object")
    failures: list[str] = []
    expected_profile = {
        "canary": "phase9a_factor_tuning_canary_v1",
        "full": "phase9a_factor_tuning_full_v1",
        "r2": "phase9a_factor_tuning_full_r2_v1",
        "source-sync": "phase9a_factor_tuning_full_source_sync_v1",
    }[role]
    expected_indices = [3] if role == "canary" else list(range(6))
    expected_status = (
        "PASS_PHASE9A_SCOPE_PREFLIGHT_PARTIAL"
        if role == "canary"
        else "PASS_PHASE9A_SCOPE_PREFLIGHT"
    )
    _require(manifest.get("status") == expected_status, "manifest status mismatch", failures)
    _require(manifest.get("profile_id") == expected_profile, "profile identity mismatch", failures)
    _require(manifest.get("principal_sqrt_backend") == FACTOR_BACKEND, "manifest backend mismatch", failures)
    profile = manifest.get("profile", {})
    _require(isinstance(profile, Mapping), "profile payload missing", failures)
    if isinstance(profile, Mapping):
        _require(profile.get("principal_sqrt_backend") == FACTOR_BACKEND, "profile backend mismatch", failures)
        _require(profile.get("plan_path") == PLAN_PATH, "plan path mismatch", failures)
        roots = _roots(profile)
        _require(bool(roots), "profile seed roots missing", failures)
        _require(len(roots) == len(set(roots)), "profile seed roots overlap", failures)
    _require(manifest.get("result_note_path") == RESULT_PATH, "result note path mismatch", failures)
    _require(manifest.get("selected_scope_indices") == expected_indices, "scope coverage mismatch", failures)
    _require(manifest.get("route_scan", {}).get("passed") is True, "forbidden-route scan failed", failures)
    _require(manifest.get("tf32_execution_enabled") is True, "TF32 provenance missing", failures)
    _require(manifest.get("logical_gpus") == ["/device:GPU:0"], "GPU0 visibility mismatch", failures)
    memory = manifest.get("memory_policy", {})
    _require(
        isinstance(memory, Mapping)
        and memory.get("all_physical_devices_memory_growth") is True
        and memory.get("configured_before_logical_device_initialization") is True,
        "memory-growth policy failed",
        failures,
    )
    peak = manifest.get("allocator", {}).get("peak")
    _require(isinstance(peak, int), "allocator peak missing", failures)
    if isinstance(peak, int):
        _require(peak <= ALLOCATOR_CAP_BYTES, "allocator peak exceeds 4 GiB", failures)
    component_signature = manifest.get("component_target_adapter_signature")
    _require(isinstance(component_signature, str) and bool(component_signature), "component adapter signature missing", failures)
    bridge_signature = str(manifest.get("bridge_signature", ""))
    _require(bool(bridge_signature), "bridge signature missing", failures)
    chart_records = manifest.get("charts", ())
    _require(len(chart_records) == 2, "expected two chart records", failures)
    for chart in chart_records if isinstance(chart_records, Sequence) else ():
        if not isinstance(chart, Mapping):
            failures.append("malformed chart record")
            continue
        checkpoints = chart.get("checkpoints", ())
        _require(len(checkpoints) == 3, "chart does not contain three beta checkpoints", failures)
        for entry in checkpoints if isinstance(checkpoints, Sequence) else ():
            checkpoint = entry.get("checkpoint", {}) if isinstance(entry, Mapping) else {}
            _require(checkpoint.get("bridge_signature") == bridge_signature, "checkpoint bridge mismatch", failures)
            _require(
                checkpoint.get("checkpoint_scope", {}).get("principal_sqrt_backend")
                == FACTOR_BACKEND,
                "checkpoint backend mismatch",
                failures,
            )
            _require(entry.get("compiled_training_trace_count") == 1, "training graph retraced", failures)
    scope_rows = manifest.get("scope_records", ())
    _require(len(scope_rows) == len(expected_indices), "scope record count mismatch", failures)
    summaries = []
    if isinstance(scope_rows, Sequence):
        by_index = {
            int(row.get("scope_index", -1)): row
            for row in scope_rows
            if isinstance(row, Mapping)
        }
        for index in expected_indices:
            row = by_index.get(index, {})
            summaries.append(
                _audit_scope(
                    row,
                    expected_index=index,
                    attempt_dir=attempt_dir,
                    bridge_signature=bridge_signature,
                    failures=failures,
                )
            )
    transition = manifest.get("transition")
    if role == "canary":
        _require(transition is None, "canary unexpectedly ran full transition", failures)
    else:
        _require(isinstance(transition, Mapping), "full transition missing", failures)
        if isinstance(transition, Mapping):
            _require(transition.get("controller_passed") is True, "transition controller failed", failures)
            _require(not transition.get("hard_vetoes"), "transition controller has hard veto", failures)
            _require(transition.get("device") == "/device:GPU:0", "transition device mismatch", failures)
            _require(transition.get("warmup_results_per_chain") == 4, "mechanics warmup count changed", failures)
            _require(transition.get("retained_results_per_chain") == 4, "mechanics retained count changed", failures)
    stored_manifest_hash = manifest.get("manifest_hash")
    unhashed = dict(manifest)
    unhashed.pop("manifest_hash", None)
    _require(stored_manifest_hash == _stable_hash(unhashed), "manifest self-hash mismatch", failures)
    return {
        "schema": "bayesfilter.ssl_lstm_q20.factor_tuning_admission_audit.v1",
        "status": (
            f"PASS_FACTOR_{role.upper()}_AUDIT" if not failures else f"FAIL_FACTOR_{role.upper()}_AUDIT"
        ),
        "role": role,
        "attempt_dir": str(attempt_dir.resolve()),
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": _sha256(manifest_path),
        "profile_id": manifest.get("profile_id"),
        "principal_sqrt_backend": manifest.get("principal_sqrt_backend"),
        "bridge_signature": bridge_signature,
        "component_target_adapter_signature": component_signature,
        "selected_scope_indices": manifest.get("selected_scope_indices"),
        "scope_summaries": summaries,
        "allocator_peak_bytes": peak,
        "wall_time_seconds": manifest.get("wall_time_seconds"),
        "failures": failures,
        "claim_boundary": "phase9a_numerical_backend_admission_not_posterior_evidence",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt-dir", required=True, type=Path)
    parser.add_argument(
        "--role", required=True, choices=("canary", "full", "r2", "source-sync")
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise AuditError(f"refusing to overwrite audit output: {args.output}")
    payload = audit(args.attempt_dir.resolve(), args.role)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "output": str(args.output), "failures": payload["failures"]}, sort_keys=True))
    return 0 if not payload["failures"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
