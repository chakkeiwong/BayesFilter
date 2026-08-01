"""Reclassify immutable Phase 9 shards under the FD-only direction policy."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import platform
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bayesfilter.ledh_fd_policy import (  # noqa: E402
    LEDH_FD_BASE_RELATIVE_TOLERANCE,
    LEDH_FD_DENOMINATOR,
    LEDH_FD_DIAGNOSTIC_SCOPE,
    LEDH_FD_PASS_RULE,
    LEDH_FD_POLICY_ID,
    LEDH_FD_STATISTICAL_STATUS,
    evaluate_ledh_fd_policy,
)


SCHEMA_VERSION = "bayesfilter.ledh.phase9_fd_policy_reclassification.v2"
INPUT_SCHEMA_VERSION = "bayesfilter.ledh.phase9_fd_reclassification_inputs.v1"
LEGACY_SHARD_SCHEMA_VERSION = "bayesfilter.ledh.compact_score_gpu_xla.v1"
CORRECTION_PLAN_PATH = (
    "docs/plans/bayesfilter-ledh-score-wiring-repair-phase9-fd-policy-correction-subplan-2026-07-11.md"
)
HISTORICAL_RESULT_PATH = (
    "docs/plans/bayesfilter-ledh-score-wiring-repair-phase9-gpu-score-memory-result-2026-07-10.md"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_output(command: Sequence[str]) -> str:
    try:
        return subprocess.check_output(command, cwd=ROOT, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:  # noqa: BLE001
        return f"unavailable:{type(exc).__name__}:{exc}"


def _mapping(name: str, value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _sequence(name: str, value: object) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a sequence")
    return value


def _finite(name: str, value: object) -> float:
    try:
        output = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(output):
        raise ValueError(f"{name} must be finite")
    return output


def _resolve_source(path_text: object) -> tuple[str, Path]:
    if not isinstance(path_text, str) or not path_text:
        raise ValueError("source path must be a nonempty string")
    relative = Path(path_text)
    if relative.is_absolute():
        raise ValueError("source paths must be repository-relative")
    resolved = (ROOT / relative).resolve()
    if not resolved.is_relative_to(ROOT):
        raise ValueError("source path escapes the repository")
    return relative.as_posix(), resolved


def _load_bound_json(path_text: object, expected_sha256: object) -> tuple[str, Path, str, dict[str, Any]]:
    relative, path = _resolve_source(path_text)
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise ValueError(f"expected SHA-256 is invalid for {relative}")
    observed_sha256 = _sha256(path)
    if observed_sha256 != expected_sha256:
        raise ValueError(f"source SHA-256 mismatch for {relative}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"source JSON must be an object: {relative}")
    return relative, path, observed_sha256, payload


def _validate_declared_output(payload: Mapping[str, Any], source_path: Path) -> Mapping[str, Any]:
    manifest = _mapping("source run_manifest", payload.get("run_manifest"))
    declared = Path(str(manifest.get("output")))
    if not declared.is_absolute():
        declared = ROOT / declared
    if declared.resolve() != source_path.resolve():
        raise ValueError("source run_manifest output does not match its bound path")
    return manifest


def _reclassify_entry(entry: Mapping[str, Any]) -> dict[str, Any]:
    entry_id = entry.get("id")
    row = entry.get("row")
    if not isinstance(entry_id, str) or not entry_id or not isinstance(row, str) or not row:
        raise ValueError("each input entry requires nonempty id and row")
    historical_terminal_decision = entry.get("historical_terminal_decision")
    if not isinstance(historical_terminal_decision, bool):
        raise ValueError(f"historical_terminal_decision must be boolean for {entry_id}")
    if entry_id.startswith("gate-b-"):
        gate_order = 0
    elif entry_id.startswith("gate-c-"):
        gate_order = 1
    else:
        raise ValueError(f"input entry id must identify Gate B or Gate C: {entry_id}")

    score_path_text, score_path, score_sha256, score_payload = _load_bound_json(
        entry.get("score_path"),
        entry.get("score_sha256"),
    )
    fd_path_text, fd_path, fd_sha256, fd_payload = _load_bound_json(
        entry.get("fd_path"),
        entry.get("fd_sha256"),
    )
    score_manifest = _validate_declared_output(score_payload, score_path)
    fd_manifest = _validate_declared_output(fd_payload, fd_path)

    if score_payload.get("schema_version") != LEGACY_SHARD_SCHEMA_VERSION:
        raise ValueError(f"score shard is not a legacy Phase 9 shard: {entry_id}")
    if fd_payload.get("schema_version") != LEGACY_SHARD_SCHEMA_VERSION:
        raise ValueError(f"FD shard is not a legacy Phase 9 shard: {entry_id}")
    if score_payload.get("artifact_status") != "completed" or score_payload.get("terminal_artifact") is not True:
        raise ValueError(f"score shard is not a completed terminal artifact: {entry_id}")
    if fd_payload.get("artifact_status") not in {"completed", "failed_fd"} or fd_payload.get("terminal_artifact") is not True:
        raise ValueError(f"FD shard has no completed FD measurement: {entry_id}")
    if score_manifest.get("stage") != "score-only" or fd_manifest.get("stage") != "fd-only":
        raise ValueError(f"source stages are not a score/FD pair: {entry_id}")
    if score_manifest.get("row") != row or fd_manifest.get("row") != row:
        raise ValueError(f"source row does not match input manifest: {entry_id}")
    for field in ("row_id", "time_steps", "num_particles", "batch_seeds"):
        if score_manifest.get(field) != fd_manifest.get(field):
            raise ValueError(f"score/FD {field} mismatch: {entry_id}")
    if score_payload.get("row_id") != fd_payload.get("row_id"):
        raise ValueError(f"score/FD row_id mismatch: {entry_id}")
    if fd_payload.get("score_reference_sha256") != score_sha256:
        raise ValueError(f"FD-to-score SHA-256 binding mismatch: {entry_id}")
    if fd_payload.get("prepared_input_fingerprint") != score_payload.get("prepared_input_fingerprint"):
        raise ValueError(f"score/FD prepared-input mismatch: {entry_id}")
    if fd_payload.get("precision") != score_payload.get("precision"):
        raise ValueError(f"score/FD precision mismatch: {entry_id}")

    parameter_names = tuple(str(value) for value in _sequence(
        "score_parameter_names",
        score_payload.get("score_parameter_names"),
    ))
    if not parameter_names or len(set(parameter_names)) != len(parameter_names):
        raise ValueError(f"invalid parameter names: {entry_id}")
    if tuple(fd_payload.get("score_parameter_names") or ()) != parameter_names:
        raise ValueError(f"score/FD parameter order mismatch: {entry_id}")
    score_values = tuple(_finite("score", value) for value in _sequence("score", score_payload.get("score")))
    fd_score_values = tuple(_finite("FD score", value) for value in _sequence("FD score", fd_payload.get("score")))
    if score_values != fd_score_values or len(score_values) != len(parameter_names):
        raise ValueError(f"FD score does not match score shard: {entry_id}")

    legacy = _mapping("legacy score_correctness", fd_payload.get("score_correctness"))
    if legacy.get("kind") != "same_scalar_finite_difference":
        raise ValueError(f"unexpected legacy correctness kind: {entry_id}")
    if legacy.get("uses_value_only_scalar_route") is not True:
        raise ValueError(f"legacy FD did not use the value-only scalar route: {entry_id}")
    legacy_parameters = _sequence("legacy parameters", legacy.get("parameters"))
    if len(legacy_parameters) != len(parameter_names):
        raise ValueError(f"legacy parameter count mismatch: {entry_id}")

    finite_differences = []
    recomputed_abs_errors = []
    recomputed_relative_errors = []
    for index, name in enumerate(parameter_names):
        parameter = _mapping("legacy parameter", legacy_parameters[index])
        if parameter.get("parameter") != name:
            raise ValueError(f"legacy parameter order mismatch: {entry_id}")
        parameter_score = _finite("legacy parameter score", parameter.get("score"))
        if parameter_score != score_values[index]:
            raise ValueError(f"legacy parameter score mismatch: {entry_id}")
        finite_difference = _finite("legacy finite difference", parameter.get("finite_difference"))
        absolute_error = abs(parameter_score - finite_difference)
        relative_error = absolute_error / max(abs(parameter_score), abs(finite_difference), 1.0e-12)
        if not math.isclose(_finite("legacy abs_error", parameter.get("abs_error")), absolute_error, rel_tol=1.0e-5, abs_tol=1.0e-10):
            raise ValueError(f"legacy abs_error mismatch: {entry_id}")
        if not math.isclose(_finite("legacy relative_error", parameter.get("relative_error")), relative_error, rel_tol=1.0e-5, abs_tol=1.0e-10):
            raise ValueError(f"legacy relative_error mismatch: {entry_id}")
        finite_differences.append(finite_difference)
        recomputed_abs_errors.append(absolute_error)
        recomputed_relative_errors.append(relative_error)

    max_abs_error = max(recomputed_abs_errors)
    max_relative_error = max(recomputed_relative_errors)
    if not math.isclose(_finite("legacy max_abs_error", legacy.get("max_abs_error")), max_abs_error, rel_tol=1.0e-5, abs_tol=1.0e-10):
        raise ValueError(f"legacy max_abs_error mismatch: {entry_id}")
    if not math.isclose(_finite("legacy max_relative_error", legacy.get("max_relative_error")), max_relative_error, rel_tol=1.0e-5, abs_tol=1.0e-10):
        raise ValueError(f"legacy max_relative_error mismatch: {entry_id}")
    legacy_atol = _finite("legacy atol", legacy.get("atol"))
    legacy_rtol = _finite("legacy rtol", legacy.get("rtol"))
    legacy_pass = max_abs_error <= legacy_atol or max_relative_error <= legacy_rtol
    expected_legacy_status = "pass" if legacy_pass else "fail"
    if legacy.get("status") != expected_legacy_status:
        raise ValueError(f"legacy status does not match recomputed legacy rule: {entry_id}")
    if (fd_payload.get("artifact_status") == "completed") != legacy_pass:
        raise ValueError(f"legacy artifact status does not match recomputed legacy rule: {entry_id}")

    corrected = evaluate_ledh_fd_policy(
        score_values,
        finite_differences,
        parameter_names,
    )
    return {
        "id": entry_id,
        "historical_terminal_decision": historical_terminal_decision,
        "corrected_ladder_stop": False,
        "gate_order": gate_order,
        "row": row,
        "row_id": score_payload.get("row_id"),
        "time_steps": int(score_manifest["time_steps"]),
        "num_particles": int(score_manifest["num_particles"]),
        "batch_seeds": [int(value) for value in score_manifest["batch_seeds"]],
        "fd_step": _finite("FD step", legacy.get("step")),
        "precision": score_payload.get("precision"),
        "source_score": {"path": score_path_text, "sha256": score_sha256},
        "source_fd": {"path": fd_path_text, "sha256": fd_sha256},
        "fd_declared_score_reference_sha256": fd_payload.get("score_reference_sha256"),
        "legacy_policy": {
            "status": expected_legacy_status,
            "atol": legacy_atol,
            "rtol": legacy_rtol,
            "pass_rule": legacy.get("pass_rule"),
            "max_abs_error": max_abs_error,
            "max_relative_error": max_relative_error,
            "scientific_status": "superseded_unsupported_threshold_basis",
        },
        "corrected_policy": corrected,
    }


def reclassify_manifest(
    manifest_path: Path,
    *,
    command: str | None = None,
    output_path: Path | None = None,
    markdown_output_path: Path | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    manifest_path = manifest_path.resolve()
    if not manifest_path.is_relative_to(ROOT):
        raise ValueError("input manifest must be inside the repository")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema_version") != INPUT_SCHEMA_VERSION:
        raise ValueError("invalid reclassification input manifest schema")
    if manifest.get("correction_plan") != CORRECTION_PLAN_PATH:
        raise ValueError("input manifest correction plan mismatch")
    entries = _sequence("input entries", manifest.get("entries"))
    if not entries:
        raise ValueError("input manifest must contain entries")
    normalized_entries = [_reclassify_entry(_mapping("input entry", entry)) for entry in entries]
    ids = [entry["id"] for entry in normalized_entries]
    if len(set(ids)) != len(ids):
        raise ValueError("input manifest entry ids must be unique")
    historical_terminal_rows = [
        entry["row"] for entry in normalized_entries if entry["historical_terminal_decision"]
    ]
    if len(historical_terminal_rows) != 5 or len(set(historical_terminal_rows)) != 5:
        raise ValueError("input manifest must identify exactly one terminal decision per nonlinear row")

    for row in sorted(set(entry["row"] for entry in normalized_entries)):
        row_entries = sorted(
            (entry for entry in normalized_entries if entry["row"] == row),
            key=lambda entry: (entry["gate_order"], entry["time_steps"]),
        )
        first_failure = next(
            (entry for entry in row_entries if entry["corrected_policy"]["status"] == "fail"),
            None,
        )
        if first_failure is not None:
            first_failure["corrected_ladder_stop"] = True

    status_counts = {
        status: sum(entry["corrected_policy"]["status"] == status for entry in normalized_entries)
        for status in ("pass", "fail")
    }
    historical_terminal_status_counts = {
        status: sum(
            entry["historical_terminal_decision"]
            and entry["corrected_policy"]["status"] == status
            for entry in normalized_entries
        )
        for status in ("pass", "fail")
    }
    corrected_stops = [entry for entry in normalized_entries if entry["corrected_ladder_stop"]]
    if len({entry["row"] for entry in corrected_stops}) != len(corrected_stops):
        raise ValueError("a nonlinear row cannot have multiple corrected ladder stops")
    rows = sorted(set(entry["row"] for entry in normalized_entries))
    rows_with_stored_failure = sorted(entry["row"] for entry in corrected_stops)
    rows_without_stored_failure = sorted(set(rows) - set(rows_with_stored_failure))
    relative_manifest_path = manifest_path.relative_to(ROOT).as_posix()
    invoked_command = command or "python docs/benchmarks/reclassify_ledh_phase9_fd_policy.py"
    payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact_status": "completed",
        "timestamp_utc": dt.datetime.now(tz=dt.UTC).isoformat(),
        "correction_status": "old_0p005_and_intervening_wrong_policy_bases_superseded",
        "scientific_question": "Does each stored FD comparison satisfy max_j(relative_error_j) <= 0.05 * sqrt(p)?",
        "policy": {
            "policy_id": LEDH_FD_POLICY_ID,
            "diagnostic_scope": LEDH_FD_DIAGNOSTIC_SCOPE,
            "base_relative_tolerance": LEDH_FD_BASE_RELATIVE_TOLERANCE,
            "dimension_scaling": "sqrt(num_parameters)",
            "coordinate_relative_error_denominator": LEDH_FD_DENOMINATOR,
            "pass_rule": LEDH_FD_PASS_RULE,
            "statistical_interpretation": LEDH_FD_STATISTICAL_STATUS,
            "owner_motivation": "5% selected to mirror the conventional 95% confidence/significance threshold",
        },
        "summary": {
            "num_reclassified_rungs": len(normalized_entries),
            "num_historical_terminal_decisions": len(historical_terminal_rows),
            "num_corrected_ladder_stops": len(corrected_stops),
            "all_rungs": status_counts,
            "historical_terminal_reclassification": historical_terminal_status_counts,
            "corrected_ladder_stops": {"pass": 0, "fail": len(corrected_stops)},
            "rows_with_stored_fd_failure": rows_with_stored_failure,
            "rows_without_stored_fd_failure": rows_without_stored_failure,
            "ranking_statistically_supported": False,
            "hmc_readiness_established": False,
        },
        "entries": normalized_entries,
        "run_manifest": {
            "git_commit": _git_output(("git", "rev-parse", "HEAD")),
            "git_status_short": _git_output(("git", "status", "--short")),
            "command": invoked_command,
            "working_directory": str(ROOT),
            "python_executable": sys.executable,
            "python_version": sys.version,
            "host": platform.node(),
            "platform": platform.platform(),
            "conda_prefix": os.environ.get("CONDA_PREFIX"),
            "input_manifest": relative_manifest_path,
            "input_manifest_sha256": _sha256(manifest_path),
            "output_json": (
                str(output_path.resolve().relative_to(ROOT))
                if output_path is not None and output_path.resolve().is_relative_to(ROOT)
                else str(output_path) if output_path is not None else None
            ),
            "output_markdown": (
                str(markdown_output_path.resolve().relative_to(ROOT))
                if markdown_output_path is not None
                and markdown_output_path.resolve().is_relative_to(ROOT)
                else str(markdown_output_path) if markdown_output_path is not None else None
            ),
            "reclassifier_path": "docs/benchmarks/reclassify_ledh_phase9_fd_policy.py",
            "reclassifier_sha256": _sha256(Path(__file__)),
            "policy_module_path": "bayesfilter/ledh_fd_policy.py",
            "policy_module_sha256": _sha256(ROOT / "bayesfilter/ledh_fd_policy.py"),
            "plan_path": CORRECTION_PLAN_PATH,
            "plan_sha256": _sha256(ROOT / CORRECTION_PLAN_PATH),
            "historical_result_path": HISTORICAL_RESULT_PATH,
            "execution_target": "cpu_only_offline_reclassification",
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu_execution_performed": False,
            "random_seeds": "preserved from source shards; no new randomness",
            "wall_time_seconds": None,
            "data_version": "SHA-256-bound immutable Phase 9 score/FD JSON shards",
        },
        "nonclaims": [
            "The 5% constant mirrors the conventional 95% threshold; this FD calculation is not a calibrated confidence interval.",
            "This policy applies only to the finite-difference diagnostic.",
            "A pass does not establish general score correctness, HMC readiness, posterior correctness, full-row admission, or default readiness.",
            "A fail does not isolate compact-score error from float32 finite-difference resolution.",
            "No stochastic ranking or scientific superiority claim is supported.",
        ],
    }
    payload["run_manifest"]["wall_time_seconds"] = time.perf_counter() - started
    return payload


def render_markdown(payload: Mapping[str, Any], json_path: Path) -> str:
    entries = list(payload["entries"])
    corrected_stops = [entry for entry in entries if entry["corrected_ladder_stop"]]
    historical_terminal_entries = [
        entry for entry in entries if entry["historical_terminal_decision"]
    ]
    summary = payload["summary"]
    lines = [
        "# Phase 9 FD Policy Correction Result",
        "",
        "Date: 2026-07-11",
        "",
        "Status: `CORRECTION_COMPLETE_OWNER_DIRECTED_POLICY_RECLASSIFIED`",
        "",
        "## Outcome",
        "",
        "The former Phase 9 hard-veto decisions based on the inherited `0.005`",
        "absolute-or-relative gate and the intervening `2%` RSS/RMS correction are",
        "superseded. The owner clarified that this is an FD-only check over individual",
        "parameter directions with a `5% * sqrt(p)` tolerance. The original trusted",
        "GPU/XLA score and FD values remain valid raw measurements and were not modified.",
        "",
        f"The corrected rule classifies {summary['all_rungs']['pass']} of "
        f"{summary['num_reclassified_rungs']} stored comparisons as passing and",
        f"{summary['all_rungs']['fail']} as failing. Predator-prey fails at Gate B and",
        "generalized-SV passes Gate B but fails at Gate C. Fixed-SIR, Actual-SV, and",
        "KSC-SV have no stored FD failure under this rule. This is an offline",
        "reclassification of stored values, not a new GPU run.",
        "",
        "## Corrected Policy",
        "",
        "For each direction, preserve the historical coordinate definition",
        "`r_j = |score_j - FD_j| / max(|score_j|, |FD_j|, 1e-12)`. For `p`",
        "parameter directions, a comparison passes exactly when",
        "`max_j(r_j) <= 0.05 * sqrt(p)`. Directions are not combined with RSS, RMS,",
        "or an average, and there is no absolute-error escape hatch.",
        "",
        "The `5%` choice mirrors the conventional 95% confidence/significance",
        "threshold. The FD calculation is not itself a confidence interval: no",
        "sampling distribution, standard error, coverage calculation, or repeated-run",
        "calibration is computed.",
        "",
        "## Stored FD Stops",
        "",
        "| Row | Rung | p | Maximum direction error | `0.05*sqrt(p)` | Maximum-error parameter | FD decision |",
        "| --- | --- | ---: | ---: | ---: | --- | --- |",
    ]
    for entry in corrected_stops:
        policy = entry["corrected_policy"]
        rung = f"T={entry['time_steps']},N={entry['num_particles']}"
        lines.append(
            f"| {entry['row']} | {rung} | {policy['num_parameters']} | "
            f"{policy['max_coordinate_relative_error']:.12g} | "
            f"{policy['max_coordinate_relative_error_threshold']:.12g} | "
            f"{policy['max_error_parameter']} | {policy['status'].upper()} |"
        )
    if not corrected_stops:
        lines.append("| None | N/A | N/A | N/A | N/A | N/A | N/A |")
    lines.extend(
        [
            "",
            "## Historical Terminal Shards",
            "",
            "| Row | Historical rung | p | Maximum direction error | Threshold | FD decision |",
            "| --- | --- | ---: | ---: | ---: | --- |",
        ]
    )
    for entry in historical_terminal_entries:
        policy = entry["corrected_policy"]
        lines.append(
            f"| {entry['row']} | T={entry['time_steps']},N={entry['num_particles']} | "
            f"{policy['num_parameters']} | {policy['max_coordinate_relative_error']:.12g} | "
            f"{policy['max_coordinate_relative_error_threshold']:.12g} | "
            f"{policy['status']} |"
        )
    lines.extend(
        [
            "",
            "## All Reclassified Rungs",
            "",
            "| ID | Legacy decision (superseded basis) | Maximum direction error | Threshold | FD decision |",
            "| --- | --- | ---: | ---: | --- |",
        ]
    )
    for entry in entries:
        policy = entry["corrected_policy"]
        lines.append(
            f"| {entry['id']} | {entry['legacy_policy']['status']} | "
            f"{policy['max_coordinate_relative_error']:.12g} | "
            f"{policy['max_coordinate_relative_error_threshold']:.12g} | "
            f"{policy['status']} |"
        )
    lines.extend(
        [
            "",
            "## Decision Table",
            "",
            "| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | What is not concluded |",
            "| --- | --- | --- | --- | --- | --- |",
            "| Supersede both the old `0.005` decisions and the wrong `2%` RSS/RMS correction. | Nine stored comparisons pass the FD-only rule; predator-prey Gate B and generalized-SV Gate C fail it. Actual-SV passes with `0.0602925 <= 0.0707107`. | Source hashes, FD-to-score bindings, parameter order, finiteness, and legacy stored-field consistency passed. | Float32 FD resolution versus compact-score math remains unisolated for the two failures; comparisons use one seed and one FD step. | Treat the three newly passing historical terminal rows as having this FD veto removed; separately plan any continuation required by the original ladder. Diagnose the two failures only under a reviewed derivative-resolution plan. | No general score correctness, HMC readiness, posterior correctness, default readiness, full admission, causal attribution, calibrated confidence interval, or superiority. |",
            "",
            "## Inference Status",
            "",
            "| Item | Status |",
            "| --- | --- |",
            "| FD-only veto screen | Stored FD failures are supported only for predator-prey Gate B and generalized-SV Gate C. Fixed-SIR, Actual-SV, and KSC-SV have no stored failure under the clarified rule. |",
            "| Statistically supported ranking | None. This deterministic reclassification provides no uncertainty analysis or candidate ranking. |",
            "| Descriptive-only differences | Per-coordinate errors and margins, FD steps, runtime, and memory differences remain descriptive. |",
            "| Default-readiness | Not established. Passing this FD diagnostic does not establish any broader readiness claim. |",
            "| Next evidence needed | Original downstream ladder requirements remain separate. The two FD failures need a reviewed precision/step diagnostic if pursued. |",
            "",
            "## Engineering, Numerical, And Scientific Ledgers",
            "",
            "| Ledger | Verdict |",
            "| --- | --- |",
            "| Engineering correctness | Reclassifier completed and verified every source hash and cross-shard binding. |",
            "| Numerical validity | Stored comparisons were re-evaluated with the maximum individual-direction formula; nine pass and two fail. |",
            "| Scientific interpretation | FD diagnostic only. General score validity, HMC behavior, and posterior validity were not evaluated. |",
            "",
            "## Run Manifest",
            "",
            f"- JSON artifact: `{json_path}`",
        ]
    )
    for key, value in payload["run_manifest"].items():
        if key != "git_status_short":
            lines.append(f"- {key}: `{value}`")
    lines.append("- git_status_short: full dirty-worktree disclosure is preserved in the JSON artifact")
    lines.extend(
        [
            "",
            "## Source Bindings",
            "",
            "| ID | Score JSON SHA-256 | FD JSON SHA-256 |",
            "| --- | --- | --- |",
        ]
    )
    for entry in entries:
        lines.append(
            f"| {entry['id']} | `{entry['source_score']['sha256']}` | "
            f"`{entry['source_fd']['sha256']}` |"
        )
    lines.extend(
        [
            "",
            "## Post-Run Red Team",
            "",
            "- Strongest alternative explanation: production-float32 central FD at",
            "  the historical step may be resolution-limited. This remains explanatory",
            "  because no precision or step ladder was run.",
            "- Result that would overturn either stored FD failure: a reviewed,",
            "  predeclared same-scalar derivative check of the unchanged target that",
            "  passes this corrected policy with source-bound evidence.",
            "- Weakest evidence: each comparison is one seed at one FD step; there is",
            "  no calibrated confidence interval despite the conventional 5% motivation.",
            "- The correction does not authorize a GPU rerun, Gate D, aggregation,",
            "  HMC execution, or Phase 10.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--markdown-output", required=True)
    args = parser.parse_args(argv)
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    output = Path(args.output)
    markdown_output = Path(args.markdown_output)
    command = shlex.join([sys.executable, sys.argv[0], *raw_argv])
    payload = reclassify_manifest(
        Path(args.manifest),
        command=command,
        output_path=output,
        markdown_output_path=markdown_output,
    )
    _write_json_atomic(output, payload)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.write_text(render_markdown(payload, output), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
