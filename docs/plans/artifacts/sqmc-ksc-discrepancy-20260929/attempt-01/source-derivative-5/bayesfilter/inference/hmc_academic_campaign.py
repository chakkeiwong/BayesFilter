"""Versioned artifact and budget controls for the Phase 7 academic campaign."""

from __future__ import annotations

import json
import fcntl
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bayesfilter.inference.hmc_identity import (
    artifact_file_sha256,
    canonical_artifact_payload_hash,
)
from bayesfilter.runtime import atomic_write_json


ACADEMIC_CAMPAIGN_ID = "phase7-typed-identity-serious-hmc-2026-07-13"
ACADEMIC_CONTEXT_KIND = "phase7_academic_campaign"
ACADEMIC_CAMPAIGN_SCHEMA = "bayesfilter.hmc_phase7_academic_campaign.v1"
ACADEMIC_MANIFEST_SCHEMA = "bayesfilter.hmc_phase7_academic_manifest.v1"
ACADEMIC_ATTEMPT_SUMMARY_SCHEMA = (
    "bayesfilter.hmc_phase7_academic_attempt_summary.v1"
)
ACADEMIC_CHECKSUM_SCHEMA = "bayesfilter.hmc_phase7_academic_checksums.v1"
ACADEMIC_PROGRESS_SCHEMA = "bayesfilter.hmc_phase7_academic_progress.v1"
ACADEMIC_RESULT_SCHEMA = "bayesfilter.hmc_phase7_academic_result.v1"
ACADEMIC_FAILURE_SCHEMA = "bayesfilter.hmc_phase7_academic_failure.v1"
CAMPAIGN_MAX_ATTEMPTS = 3
CAMPAIGN_WALL_TIME_CAP_SECONDS = 8 * 60 * 60
CAMPAIGN_TERMINALIZATION_RESERVE_SECONDS = 15.0
EXPECTED_TRANSITION_IDENTITY = (
    "sha256:10d9a9d2d71562d0c278b5bbc0ba0bb3eed3fc2ae77510a6d09e5c16a6f16d6a"
)
EXPECTED_SERIOUS_EXECUTION_IDENTITY = (
    "sha256:ceefd154f97510f2b432c45287a0f309792a3def3855dec3ffd2061f2b4587e4"
)
ACADEMIC_PASS_DECISION = "PASS_PHASE7_ACADEMIC_CAMPAIGN_TO_PHASE8_CLOSEOUT"
ACADEMIC_BLOCK_DECISION = "BLOCK_PHASE7_ACADEMIC_CAMPAIGN"
ACADEMIC_NONCLAIMS = (
    "no posterior recovery or calibrated uncertainty claim",
    "no sampler superiority claim",
    "no production, default, or GPU readiness claim",
    "no Phase 8 or NeuTra execution claim",
    "strict Phase 7 diagnostics only evaluate this fixed HMC campaign",
)
ACADEMIC_FAILURE_NONCLAIMS = (
    "candidate or attempt failure is not research-direction rejection",
    *ACADEMIC_NONCLAIMS,
)
ACADEMIC_DIAGNOSTIC_DEFINITIONS = {
    "rank_transform": "Blom average-rank normal score",
    "rhat": "max(rank-normalized split R-hat, folded rank-normalized split R-hat)",
    "bulk_ess": "split-chain cross-chain ESS of rank-normalized draws",
    "tail_ess": "minimum split-chain cross-chain ESS of pooled q05/q95 indicators",
    "autocorrelation_truncation": "TFP initial positive pairs",
    "quantile_interpolation": "linear",
}
ACADEMIC_DIAGNOSTIC_NONCLAIMS = (
    "all-parameter HMC convergence screen only",
    "no posterior recovery claim",
    "no sampler superiority claim",
    "no production or default readiness claim",
)


class AcademicCampaignError(RuntimeError):
    """Raised when campaign admission, budget, or terminal evidence is invalid."""


@dataclass(frozen=True)
class AcademicLaunchContext:
    """Validated ordinary context admitting one serious Phase 7 attempt."""

    context_kind: str
    campaign_id: str
    attempt_number: int
    campaign_root: Path
    run_directory: Path
    paths: Mapping[str, Path]
    config: Any
    preflight: Mapping[str, Any]
    command: tuple[str, ...]
    remaining_wall_time_seconds: float
    controller_wall_time_cap_seconds: float
    controller_deadline_monotonic: float
    prior_cumulative_wall_time_seconds: float
    implementation_references: Mapping[str, Mapping[str, Any]]
    manifest: Mapping[str, Any]
    lock_handle: Any


def default_campaign_root(config: Any) -> Path:
    return config.artifact_root / "phase7_academic_campaign"


def prepare_academic_launch(
    *,
    config: Any,
    preflight: Mapping[str, Any],
    command: Sequence[str],
    campaign_root: str | Path | None = None,
    now: datetime | None = None,
    invocation_elapsed_seconds: float = 0.0,
    invocation_started_monotonic: float | None = None,
) -> AcademicLaunchContext:
    """Exclusively create and bind the next versioned campaign attempt."""

    _validate_config_and_preflight(config, preflight)
    root = (
        default_campaign_root(config)
        if campaign_root is None
        else Path(campaign_root)
    ).resolve()
    expected_parent = config.artifact_root.resolve()
    if root.parent != expected_parent:
        raise AcademicCampaignError("campaign root must be inside the config artifact root")
    root.mkdir(parents=False, exist_ok=True)
    lock_handle = (root / ".campaign.lock").open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        lock_handle.close()
        raise AcademicCampaignError("another academic campaign launch is active") from error
    staging_directory: Path | None = None
    published_directory: Path | None = None
    try:
        for stale_staging in root.glob(".staging-*"):
            if stale_staging.is_dir():
                shutil.rmtree(stale_staging)
        history = load_attempt_history(root)
        if history and history[-1]["classification"] != "infrastructure_failure":
            raise AcademicCampaignError(
                "academic campaign is terminal after a non-infrastructure result"
            )
        attempt_number = len(history) + 1
        if attempt_number > CAMPAIGN_MAX_ATTEMPTS:
            raise AcademicCampaignError("academic campaign attempt cap reached")
        cumulative = sum(float(item["elapsed_seconds"]) for item in history)
        remaining = float(CAMPAIGN_WALL_TIME_CAP_SECONDS) - cumulative
        if invocation_started_monotonic is not None and invocation_elapsed_seconds != 0.0:
            raise AcademicCampaignError(
                "academic invocation timing inputs are mutually exclusive"
            )
        invocation_elapsed = (
            time.monotonic() - float(invocation_started_monotonic)
            if invocation_started_monotonic is not None
            else float(invocation_elapsed_seconds)
        )
        if invocation_elapsed < 0.0:
            raise AcademicCampaignError("academic invocation elapsed time is negative")
        references = build_implementation_references()
        invocation_elapsed = (
            time.monotonic() - float(invocation_started_monotonic)
            if invocation_started_monotonic is not None
            else float(invocation_elapsed_seconds)
        )
        controller_cap = min(
            float(config.payload["execution"]["wall_time_cap_seconds"]),
            remaining
            - invocation_elapsed
            - CAMPAIGN_TERMINALIZATION_RESERVE_SECONDS,
        )
        if remaining <= 0.0 or controller_cap <= 0.0:
            raise AcademicCampaignError("academic campaign wall-time budget exhausted")
        controller_deadline = time.monotonic() + controller_cap
        timestamp = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        run_name = (
            f"attempt-{attempt_number:03d}-"
            f"{timestamp.strftime('%Y%m%dT%H%M%SZ')}"
        )
        run_directory = root / run_name
        if run_directory.exists():
            raise AcademicCampaignError("academic campaign run directory collision")
        staging_directory = root / f".staging-{run_name}-{os.getpid()}"
        staging_directory.mkdir(parents=False, exist_ok=False)
        (staging_directory / "private").mkdir(mode=0o700)
        paths = _attempt_paths(run_directory)
        manifest_body = _build_manifest_body(
            config=config,
            preflight=preflight,
            command=command,
            references=references,
            timestamp=timestamp,
            attempt_number=attempt_number,
            cumulative=cumulative,
            remaining=remaining,
            invocation_elapsed=invocation_elapsed,
            controller_cap=controller_cap,
            run_directory=run_directory,
            paths=paths,
        )
        manifest = _with_hash(manifest_body)
        atomic_write_json(staging_directory / "run_manifest.json", manifest)
        os.rename(staging_directory, run_directory)
        staging_directory = None
        published_directory = run_directory
        context = AcademicLaunchContext(
            context_kind=ACADEMIC_CONTEXT_KIND,
            campaign_id=ACADEMIC_CAMPAIGN_ID,
            attempt_number=attempt_number,
            campaign_root=root,
            run_directory=run_directory,
            paths=paths,
            config=config,
            preflight=dict(preflight),
            command=tuple(str(item) for item in command),
            remaining_wall_time_seconds=remaining,
            controller_wall_time_cap_seconds=controller_cap,
            controller_deadline_monotonic=controller_deadline,
            prior_cumulative_wall_time_seconds=cumulative,
            implementation_references=references,
            manifest=manifest,
            lock_handle=lock_handle,
        )
    except BaseException:
        if staging_directory is not None:
            shutil.rmtree(staging_directory, ignore_errors=True)
        if published_directory is not None:
            shutil.rmtree(published_directory, ignore_errors=True)
        _release_lock_handle(lock_handle)
        raise
    return context


def _build_manifest_body(
    *,
    config: Any,
    preflight: Mapping[str, Any],
    command: Sequence[str],
    references: Mapping[str, Mapping[str, Any]],
    timestamp: datetime,
    attempt_number: int,
    cumulative: float,
    remaining: float,
    invocation_elapsed: float,
    controller_cap: float,
    run_directory: Path,
    paths: Mapping[str, Path],
) -> Mapping[str, Any]:
    return {
        "schema": ACADEMIC_MANIFEST_SCHEMA,
        "campaign_id": ACADEMIC_CAMPAIGN_ID,
        "attempt_number": attempt_number,
        "created_at_utc": timestamp.isoformat(),
        "git_commit": _git_commit(),
        "command": [str(item) for item in command],
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": platform.python_version(),
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV", "N/A"),
        "conda_prefix": os.environ.get("CONDA_PREFIX", "N/A"),
        "data_version": config.payload["governed_source_references"]["fixture"][
            "file_sha256"
        ],
        "config_path": str(config.path.resolve()),
        "config_file_sha256": artifact_file_sha256(config.path),
        "config_hash": config.hash,
        "transition_identity_hash": EXPECTED_TRANSITION_IDENTITY,
        "serious_execution_contract_hash": EXPECTED_SERIOUS_EXECUTION_IDENTITY,
        "preflight_artifact_hash": preflight["artifact_hash"],
        "target_scope": preflight["target_scope"],
        "parameter_count": len(preflight["parameter_names"]),
        "device": "cpu_only_cuda_visible_devices_minus_1",
        "dtype": "float64",
        "jit_compile": True,
        "use_xla": True,
        "worker_count": config.worker_count,
        "chains_per_worker": config.chains_per_worker,
        "root_seed": list(config.payload["execution"]["root_seed"]),
        "thread_environment": dict(config.payload["execution"]["thread_environment"]),
        "campaign_max_attempts": CAMPAIGN_MAX_ATTEMPTS,
        "campaign_wall_time_cap_seconds": CAMPAIGN_WALL_TIME_CAP_SECONDS,
        "terminalization_reserve_seconds": (
            CAMPAIGN_TERMINALIZATION_RESERVE_SECONDS
        ),
        "prior_cumulative_wall_time_seconds": cumulative,
        "remaining_wall_time_seconds": remaining,
        "invocation_elapsed_before_attempt_seconds": invocation_elapsed,
        "controller_wall_time_cap_seconds": controller_cap,
        "run_directory": str(run_directory),
        "artifact_paths": {name: str(path) for name, path in paths.items()},
        "implementation_references": references,
        "implementation_inventory_hash": canonical_artifact_payload_hash(
            references
        ),
        "plan_path": (
            "docs/plans/bayesfilter-hmc-semantic-identity-migration-"
            "phase7-academic-campaign-subplan-2026-07-13.md"
        ),
        "nonclaims": list(ACADEMIC_NONCLAIMS),
    }


def _attempt_paths(run_directory: Path) -> Mapping[str, Path]:
    return {
        "run_manifest_path": run_directory / "run_manifest.json",
        "public_progress_path": run_directory / "progress.json",
        "public_result_path": run_directory / "result.json",
        "failure_path": run_directory / "failure.json",
        "private_samples_path": run_directory / "private" / "retained_samples.npz",
        "log_path": run_directory / "run.log",
        "checksums_path": run_directory / "checksums.json",
        "attempt_summary_path": run_directory / "attempt_summary.json",
    }


def validate_academic_launch_context(
    context: AcademicLaunchContext,
    *,
    config: Any,
) -> None:
    if not isinstance(context, AcademicLaunchContext):
        raise AcademicCampaignError("academic launch context type mismatch")
    if context.context_kind != ACADEMIC_CONTEXT_KIND or (
        context.campaign_id != ACADEMIC_CAMPAIGN_ID
    ):
        raise AcademicCampaignError("academic launch context identity mismatch")
    if context.config.hash != config.hash or (
        context.config.path.resolve() != config.path.resolve()
    ):
        raise AcademicCampaignError("academic launch config mismatch")
    _validate_config_and_preflight(config, context.preflight)
    if context.run_directory.parent != context.campaign_root:
        raise AcademicCampaignError("academic run-directory containment mismatch")
    expected_paths = {
        "run_manifest_path": context.run_directory / "run_manifest.json",
        "public_progress_path": context.run_directory / "progress.json",
        "public_result_path": context.run_directory / "result.json",
        "failure_path": context.run_directory / "failure.json",
        "private_samples_path": context.run_directory / "private" / "retained_samples.npz",
        "log_path": context.run_directory / "run.log",
        "checksums_path": context.run_directory / "checksums.json",
        "attempt_summary_path": context.run_directory / "attempt_summary.json",
    }
    if dict(context.paths) != expected_paths:
        raise AcademicCampaignError("academic output path contract mismatch")
    if not context.run_directory.is_dir() or not (
        context.run_directory / "private"
    ).is_dir():
        raise AcademicCampaignError("academic run directory is missing")
    if context.paths["public_result_path"].exists() or context.paths[
        "failure_path"
    ].exists():
        raise AcademicCampaignError("academic attempt already has terminal output")
    manifest = _read_json(context.paths["run_manifest_path"])
    if manifest != context.manifest:
        raise AcademicCampaignError("academic run manifest drift")
    _verify_hash(manifest, label="academic run manifest")
    if (
        manifest.get("campaign_id") != context.campaign_id
        or manifest.get("attempt_number") != context.attempt_number
        or manifest.get("config_hash") != config.hash
        or manifest.get("transition_identity_hash")
        != EXPECTED_TRANSITION_IDENTITY
        or manifest.get("serious_execution_contract_hash")
        != EXPECTED_SERIOUS_EXECUTION_IDENTITY
        or manifest.get("preflight_artifact_hash")
        != context.preflight["artifact_hash"]
        or manifest.get("run_directory") != str(context.run_directory)
        or manifest.get("remaining_wall_time_seconds")
        != context.remaining_wall_time_seconds
        or manifest.get("controller_wall_time_cap_seconds")
        != context.controller_wall_time_cap_seconds
        or manifest.get("prior_cumulative_wall_time_seconds")
        != context.prior_cumulative_wall_time_seconds
    ):
        raise AcademicCampaignError("academic context/manifest mismatch")
    verify_implementation_references(context.implementation_references)
    if context.remaining_wall_time_seconds <= 0.0 or (
        context.remaining_wall_time_seconds > CAMPAIGN_WALL_TIME_CAP_SECONDS
    ):
        raise AcademicCampaignError("academic remaining wall time mismatch")
    deadline_remaining = context.controller_deadline_monotonic - time.monotonic()
    if deadline_remaining > context.controller_wall_time_cap_seconds + 1e-6:
        raise AcademicCampaignError("academic controller deadline mismatch")
    if context.lock_handle.closed:
        raise AcademicCampaignError("academic campaign lock is not held")


def load_attempt_history(campaign_root: str | Path) -> tuple[Mapping[str, Any], ...]:
    root = Path(campaign_root)
    if not root.exists():
        return ()
    directories = sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and path.name.startswith("attempt-")
    )
    unexpected = tuple(
        path.name
        for path in root.iterdir()
        if path.is_dir()
        and not path.name.startswith("attempt-")
        and not path.name.startswith(".staging-")
    )
    if unexpected:
        raise AcademicCampaignError(
            f"academic campaign contains unexpected directories: {unexpected}"
        )
    if len(directories) > CAMPAIGN_MAX_ATTEMPTS:
        raise AcademicCampaignError("academic campaign contains too many attempts")
    summaries: list[Mapping[str, Any]] = []
    cumulative = 0.0
    for expected_number, directory in enumerate(directories, start=1):
        summary_path = directory / "attempt_summary.json"
        if not summary_path.is_file():
            raise AcademicCampaignError("prior academic attempt is not terminal")
        summary = _read_json(summary_path)
        _validate_attempt_summary(summary, directory=directory)
        if int(summary["attempt_number"]) != expected_number or not (
            directory.name.startswith(f"attempt-{expected_number:03d}-")
        ):
            raise AcademicCampaignError(
                "academic attempt numbering is not contiguous"
            )
        terminal = Path(str(summary["terminal_path"])).resolve()
        if terminal.parent != directory.resolve() or not terminal.is_file() or (
            artifact_file_sha256(terminal) != summary["terminal_file_sha256"]
        ):
            raise AcademicCampaignError("academic terminal artifact drift")
        checksums = verify_checksum_manifest(
            directory / "checksums.json",
            directory,
        )
        if checksums["artifact_hash"] != summary["checksums_artifact_hash"]:
            raise AcademicCampaignError("academic checksum summary link mismatch")
        cumulative += float(summary["elapsed_seconds"])
        expected_remaining = max(
            0.0,
            CAMPAIGN_WALL_TIME_CAP_SECONDS - cumulative,
        )
        if abs(float(summary["cumulative_wall_time_seconds"]) - cumulative) > 1e-6 or (
            abs(float(summary["remaining_wall_time_seconds"]) - expected_remaining)
            > 1e-6
        ):
            raise AcademicCampaignError("academic cumulative wall time mismatch")
        expected_overrun = max(
            0.0,
            cumulative - CAMPAIGN_WALL_TIME_CAP_SECONDS,
        )
        if abs(float(summary["budget_overrun_seconds"]) - expected_overrun) > 1e-6:
            raise AcademicCampaignError("academic budget overrun mismatch")
        summaries.append(summary)
    return tuple(summaries)


def finalize_academic_attempt(
    context: AcademicLaunchContext,
    *,
    elapsed_seconds: float,
    terminal_path: str | Path,
) -> Mapping[str, Any]:
    elapsed = float(elapsed_seconds)
    if elapsed < 0.0:
        raise AcademicCampaignError("academic attempt elapsed time is negative")
    terminal = Path(terminal_path).resolve()
    if terminal.parent != context.run_directory or not terminal.is_file():
        raise AcademicCampaignError("academic terminal artifact path mismatch")
    classification, exit_code = validate_academic_terminal(
        context,
        terminal_path=terminal,
    )
    cumulative = context.prior_cumulative_wall_time_seconds + elapsed
    budget_overrun = max(0.0, cumulative - CAMPAIGN_WALL_TIME_CAP_SECONDS)
    checksums = write_checksum_manifest(context, terminal_path=terminal)
    summary = _with_hash(
        {
            "schema": ACADEMIC_ATTEMPT_SUMMARY_SCHEMA,
            "campaign_id": context.campaign_id,
            "attempt_number": context.attempt_number,
            "run_directory": str(context.run_directory),
            "classification": classification,
            "exit_code": int(exit_code),
            "elapsed_seconds": elapsed,
            "cumulative_wall_time_seconds": cumulative,
            "remaining_wall_time_seconds": max(
                0.0, CAMPAIGN_WALL_TIME_CAP_SECONDS - cumulative
            ),
            "budget_overrun_seconds": budget_overrun,
            "terminal_path": str(terminal),
            "terminal_file_sha256": artifact_file_sha256(terminal),
            "checksums_artifact_hash": checksums["artifact_hash"],
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    atomic_write_json(context.paths["attempt_summary_path"], summary)
    _release_lock_handle(context.lock_handle)
    return summary


def release_academic_launch(context: AcademicLaunchContext) -> None:
    """Release the ordinary single-launch process lock without changing evidence."""

    _release_lock_handle(context.lock_handle)


def validate_academic_terminal(
    context: AcademicLaunchContext,
    *,
    terminal_path: str | Path,
) -> tuple[str, int]:
    """Validate terminal semantics before checksums or campaign advancement."""

    path = Path(terminal_path).resolve()
    if path.parent != context.run_directory or not path.is_file():
        raise AcademicCampaignError("academic terminal artifact path mismatch")
    payload = _read_json(path)
    _verify_hash(payload, label="academic terminal artifact")
    _validate_terminal_links(payload, context=context)
    schema = payload.get("schema")
    if schema == ACADEMIC_RESULT_SCHEMA:
        _validate_strict_pass_terminal(payload, context=context)
        _validate_terminal_progress(payload, context=context)
        return "strict_pass", 0
    if schema != ACADEMIC_FAILURE_SCHEMA or payload.get("passed") is not False or (
        payload.get("decision") != ACADEMIC_BLOCK_DECISION
    ):
        raise AcademicCampaignError("academic terminal status/schema mismatch")
    classification = payload.get("failure_classification")
    if classification not in {
        "diagnostic_cap_failure",
        "infrastructure_failure",
        "continuation_veto",
    }:
        raise AcademicCampaignError("academic terminal failure classification mismatch")
    if classification == "diagnostic_cap_failure":
        if payload.get("workers_started") is not True or payload.get(
            "hmc_transition_executed"
        ) is not True:
            raise AcademicCampaignError("academic diagnostic-cap execution mismatch")
        if payload.get("reason") not in {
            "burnin_diagnostics_failed_at_cap",
            "retained_diagnostics_failed_at_cap",
        }:
            raise AcademicCampaignError("academic diagnostic-cap reason mismatch")
        _validate_diagnostics(
            payload.get("final_diagnostics"),
            context=context,
            require_pass=False,
        )
    return str(classification), 2 if classification == "infrastructure_failure" else 1


def _validate_terminal_links(
    payload: Mapping[str, Any],
    *,
    context: AcademicLaunchContext,
) -> None:
    if (
        payload.get("campaign_id") != context.campaign_id
        or payload.get("attempt_number") != context.attempt_number
        or payload.get("config_hash") != context.config.hash
        or payload.get("run_manifest_artifact_hash")
        != context.manifest["artifact_hash"]
    ):
        raise AcademicCampaignError("academic terminal campaign link mismatch")
    preflight_hash = payload.get("preflight_before_runtime_artifact_hash")
    if preflight_hash != context.preflight["artifact_hash"]:
        raise AcademicCampaignError("academic terminal preflight link mismatch")
    if payload.get("phase8_executed") is not False or payload.get(
        "neutra_executed"
    ) is not False:
        raise AcademicCampaignError("academic terminal boundary mismatch")
    facts = tuple(
        payload.get(name)
        for name in (
            "controller_entered",
            "workers_started",
            "hmc_transition_executed",
            "serious_runtime_executed",
        )
    )
    if type(facts[0]) is not bool or any(
        value is not None and type(value) is not bool for value in facts[1:]
    ) or facts[3] is not facts[2]:
        raise AcademicCampaignError("academic terminal execution facts mismatch")


def _validate_strict_pass_terminal(
    payload: Mapping[str, Any],
    *,
    context: AcademicLaunchContext,
) -> None:
    if payload.get("passed") is not True or payload.get("decision") != (
        ACADEMIC_PASS_DECISION
    ) or payload.get("smoke") is not False:
        raise AcademicCampaignError("academic strict-pass status mismatch")
    embedded_preflight = payload.get("preflight_before_runtime")
    normalized_preflight = json.loads(
        json.dumps(context.preflight, sort_keys=True, separators=(",", ":"))
    )
    if embedded_preflight != normalized_preflight or tuple(
        payload.get("nonclaims", ())
    ) != ACADEMIC_NONCLAIMS:
        raise AcademicCampaignError("academic strict-pass evidence mismatch")
    burnin_count = payload.get("burnin_results_per_chain")
    retained_count = payload.get("retained_results_per_chain")
    if not _valid_schedule_count(burnin_count, initial=2000, step=1000, cap=16000) or (
        not _valid_schedule_count(retained_count, initial=4000, step=2000, cap=40000)
    ):
        raise AcademicCampaignError("academic strict-pass schedule mismatch")
    if any(
        payload.get(name) is not True
        for name in (
            "controller_entered",
            "workers_started",
            "hmc_transition_executed",
            "serious_runtime_executed",
        )
    ):
        raise AcademicCampaignError("academic strict-pass execution mismatch")
    if payload.get("jit_compile") is not True or payload.get("cuda_visible_devices") != (
        "-1"
    ) or payload.get("jit_compile_false_runtime_executed") is not False:
        raise AcademicCampaignError("academic strict-pass XLA/device mismatch")
    if (
        payload.get("worker_count") != 2
        or payload.get("chains_per_worker") != 2
        or payload.get("chain_count") != 4
    ):
        raise AcademicCampaignError("academic strict-pass topology mismatch")
    pids = payload.get("worker_pids")
    if not isinstance(pids, Sequence) or isinstance(pids, (str, bytes)) or (
        len(pids) != 2
    ) or any(isinstance(pid, bool) or not isinstance(pid, int) or pid < 1 for pid in pids) or (
        len(set(pids)) != 2
    ):
        raise AcademicCampaignError("academic strict-pass worker PID mismatch")
    metadata = payload.get("worker_metadata")
    if not isinstance(metadata, Sequence) or isinstance(metadata, (str, bytes)) or (
        len(metadata) != 2
    ):
        raise AcademicCampaignError("academic strict-pass worker metadata mismatch")
    for index, item in enumerate(metadata):
        if not isinstance(item, Mapping) or item.get("worker_index") != index or (
            item.get("pid") != pids[index]
        ) or item.get("jit_compile") is not True or item.get("use_xla") is not True or (
            item.get("compile_trace_count") != 1
        ) or item.get("cuda_visible_devices") != "-1" or (
            item.get("child_transition_identity_hash")
            != EXPECTED_TRANSITION_IDENTITY
        ) or item.get("child_transition_identity_verified") is not True or (
            item.get("child_implementation_verification_status")
            != "launch_inventory_only_not_child_byte_verified"
        ) or item.get("launch_implementation_inventory_hash") != (
            context.manifest["implementation_inventory_hash"]
        ):
            raise AcademicCampaignError(
                "academic strict-pass worker evidence mismatch"
            )
    _validate_diagnostics(
        payload.get("final_diagnostics"),
        context=context,
        require_pass=True,
        expected_draw_count=retained_count,
    )
    teardown = payload.get("worker_teardown")
    if not isinstance(teardown, Mapping) or teardown.get("all_exited") is not True or (
        tuple(teardown.get("worker_pids", ())) != tuple(pids)
    ):
        raise AcademicCampaignError("academic strict-pass worker teardown mismatch")
    reference = payload.get("private_retained_sample_reference")
    private_path = context.paths["private_samples_path"]
    if not isinstance(reference, Mapping) or not private_path.is_file() or (
        reference.get("file_sha256") != artifact_file_sha256(private_path)
    ) or reference.get("byte_count") != private_path.stat().st_size or (
        reference.get("shape_verified") is not True
        or reference.get("finite_verified") is not True
        or reference.get("provenance_verified") is not True
        or reference.get("path_publicized") is not False
        or reference.get("raw_samples_publicized") is not False
    ):
        raise AcademicCampaignError("academic strict-pass private sample mismatch")
    _validate_private_sample_archive(payload, context=context)


def _validate_diagnostics(
    value: Any,
    *,
    context: AcademicLaunchContext,
    require_pass: bool,
    expected_draw_count: int | None = None,
) -> None:
    fields = {
        "schema", "passed", "input_all_finite", "diagnostics_all_finite",
        "draw_count_per_chain", "chain_count", "parameter_count",
        "split_draw_count_per_chain", "split_chain_count", "thresholds",
        "definitions", "max_rhat", "min_bulk_ess", "min_tail_ess",
        "parameter_diagnostics", "hard_vetoes", "nonclaims",
    }
    if not isinstance(value, Mapping) or set(value) != fields or value.get(
        "schema"
    ) != "bayesfilter.rank_normalized_hmc_diagnostics.v1" or value.get(
        "input_all_finite"
    ) is not True or (
        value.get("diagnostics_all_finite") is not True
    ) or tuple(value.get("hard_vetoes", ())) != () or value.get(
        "parameter_count"
    ) != len(context.preflight["parameter_names"]) or value.get("chain_count") != 4:
        raise AcademicCampaignError("academic convergence diagnostic validity mismatch")
    draw_count = value.get("draw_count_per_chain")
    if isinstance(draw_count, bool) or not isinstance(draw_count, int) or draw_count < 4 or (
        expected_draw_count is not None and draw_count != expected_draw_count
    ):
        raise AcademicCampaignError("academic convergence draw count mismatch")
    if value.get("split_draw_count_per_chain") != draw_count // 2 or value.get(
        "split_chain_count"
    ) != 8:
        raise AcademicCampaignError("academic convergence split topology mismatch")
    thresholds = value.get("thresholds")
    if not isinstance(thresholds, Mapping) or (
        float(thresholds.get("rhat_max", float("nan"))) != 1.01
        or float(thresholds.get("bulk_ess_min", float("nan"))) != 1000.0
        or float(thresholds.get("tail_ess_min", float("nan"))) != 400.0
    ):
        raise AcademicCampaignError("academic convergence thresholds mismatch")
    if value.get("definitions") != ACADEMIC_DIAGNOSTIC_DEFINITIONS or tuple(
        value.get("nonclaims", ())
    ) != ACADEMIC_DIAGNOSTIC_NONCLAIMS:
        raise AcademicCampaignError("academic convergence definition mismatch")
    rows = value.get("parameter_diagnostics")
    names = tuple(context.preflight["parameter_names"])
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)) or (
        tuple(item.get("parameter") for item in rows if isinstance(item, Mapping))
        != names
    ):
        raise AcademicCampaignError("academic parameter diagnostic inventory mismatch")
    computed_passes: list[bool] = []
    parsed_metrics: list[tuple[float, float, float]] = []
    row_fields = {
        "parameter", "rank_normalized_split_rhat",
        "folded_rank_normalized_split_rhat", "rhat", "bulk_ess", "tail_ess",
        "lower_tail_ess", "upper_tail_ess", "passed",
    }
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != row_fields:
            raise AcademicCampaignError("academic parameter diagnostic row mismatch")
        metrics = tuple(
            row.get(name)
            for name in (
                "rank_normalized_split_rhat",
                "folded_rank_normalized_split_rhat",
                "rhat",
                "bulk_ess",
                "tail_ess",
                "lower_tail_ess",
                "upper_tail_ess",
            )
        )
        if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in metrics) or (
            not all(math.isfinite(float(item)) for item in metrics)
        ):
            raise AcademicCampaignError("academic parameter diagnostic is nonfinite")
        rank_rhat, folded_rhat, rhat, bulk, tail, lower, upper = (
            float(item) for item in metrics
        )
        if rhat != max(rank_rhat, folded_rhat) or tail != min(lower, upper):
            raise AcademicCampaignError("academic parameter derived metric mismatch")
        row_pass = bool(
            rhat <= 1.01 and bulk >= 1000.0 and tail >= 400.0
        )
        if row.get("passed") is not row_pass:
            raise AcademicCampaignError("academic parameter diagnostic pass mismatch")
        computed_passes.append(row_pass)
        parsed_metrics.append((rhat, bulk, tail))
    aggregates = tuple(
        value.get(name) for name in ("max_rhat", "min_bulk_ess", "min_tail_ess")
    )
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in aggregates) or (
        not all(math.isfinite(float(item)) for item in aggregates)
    ) or tuple(float(item) for item in aggregates) != (
        max(item[0] for item in parsed_metrics),
        min(item[1] for item in parsed_metrics),
        min(item[2] for item in parsed_metrics),
    ):
        raise AcademicCampaignError("academic convergence aggregate mismatch")
    overall = all(computed_passes)
    if value.get("passed") is not overall or (require_pass and not overall) or (
        not require_pass and overall
    ):
        raise AcademicCampaignError("academic convergence decision mismatch")


def _validate_terminal_progress(
    result: Mapping[str, Any],
    *,
    context: AcademicLaunchContext,
) -> None:
    path = context.paths["public_progress_path"]
    if not path.is_file():
        raise AcademicCampaignError("academic strict-pass progress is missing")
    progress = _read_json(path)
    _verify_hash(progress, label="academic terminal progress")
    fields = {
        "schema", "status", "config_hash", "smoke", "campaign_id",
        "attempt_number", "run_manifest_artifact_hash",
        "preflight_before_runtime_artifact_hash", "burnin_checks",
        "retained_checks", "completed", "passed", "result_artifact_hash",
        "artifact_hash",
    }
    if set(progress) != fields or progress.get("schema") != ACADEMIC_PROGRESS_SCHEMA or (
        progress.get("status") != "result_written"
        or progress.get("completed") is not True
        or progress.get("passed") is not True
        or progress.get("smoke") is not False
    ):
        raise AcademicCampaignError("academic strict-pass progress state mismatch")
    for name, expected in (
        ("campaign_id", context.campaign_id),
        ("attempt_number", context.attempt_number),
        ("config_hash", context.config.hash),
        ("run_manifest_artifact_hash", context.manifest["artifact_hash"]),
        ("preflight_before_runtime_artifact_hash", context.preflight["artifact_hash"]),
        ("result_artifact_hash", result["artifact_hash"]),
    ):
        if progress.get(name) != expected:
            raise AcademicCampaignError("academic progress/result link mismatch")
    burnin_checks = progress.get("burnin_checks")
    retained_checks = progress.get("retained_checks")
    if not isinstance(burnin_checks, Sequence) or isinstance(
        burnin_checks, (str, bytes)
    ) or not isinstance(retained_checks, Sequence) or isinstance(
        retained_checks, (str, bytes)
    ) or not burnin_checks or not retained_checks:
        raise AcademicCampaignError("academic strict-pass progress checks mismatch")
    _validate_progress_schedule(burnin_checks, stage="burnin")
    _validate_progress_schedule(retained_checks, stage="retained")
    if burnin_checks[-1]["completed_results_per_chain"] != result[
        "burnin_results_per_chain"
    ] or retained_checks[-1]["completed_results_per_chain"] != result[
        "retained_results_per_chain"
    ] or burnin_checks[-1]["passed"] is not True or retained_checks[-1][
        "passed"
    ] is not True:
        raise AcademicCampaignError("academic progress/result schedule mismatch")
    final_summary = retained_checks[-1]
    diagnostic = result["final_diagnostics"]
    for summary_name in ("max_rhat", "min_bulk_ess", "min_tail_ess"):
        if final_summary[summary_name] != diagnostic[summary_name]:
            raise AcademicCampaignError("academic progress/diagnostic mismatch")


def _validate_progress_schedule(
    checks: Sequence[Mapping[str, Any]],
    *,
    stage: str,
) -> None:
    initial, step, cap = (2000, 1000, 16000) if stage == "burnin" else (
        4000, 2000, 40000
    )
    for check in checks:
        _validate_progress_check(check, stage=stage)
    counts = tuple(check["completed_results_per_chain"] for check in checks)
    expected = tuple(initial + step * index for index in range(len(checks)))
    if counts != expected or counts[-1] > cap or any(
        check["passed"] is True for check in checks[:-1]
    ):
        raise AcademicCampaignError(f"academic {stage} progress schedule mismatch")


def _validate_progress_check(check: Mapping[str, Any], *, stage: str) -> None:
    fields = {
        "stage", "completed_results_per_chain", "passed", "max_rhat",
        "min_bulk_ess", "min_tail_ess", "input_all_finite",
        "diagnostics_all_finite", "hard_vetoes",
    }
    if not isinstance(check, Mapping) or set(check) != fields or check.get(
        "stage"
    ) != stage or check.get("input_all_finite") is not True or check.get(
        "diagnostics_all_finite"
    ) is not True or tuple(check.get("hard_vetoes", ())) != ():
        raise AcademicCampaignError(f"academic {stage} progress check mismatch")
    metrics = tuple(
        check.get(name) for name in ("max_rhat", "min_bulk_ess", "min_tail_ess")
    )
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in metrics) or (
        not all(math.isfinite(float(item)) for item in metrics)
    ):
        raise AcademicCampaignError(f"academic {stage} progress metric mismatch")
    expected_pass = bool(
        float(metrics[0]) <= 1.01
        and float(metrics[1]) >= 1000.0
        and float(metrics[2]) >= 400.0
    )
    if check.get("passed") is not expected_pass:
        raise AcademicCampaignError(f"academic {stage} progress pass mismatch")


def _valid_schedule_count(value: Any, *, initial: int, step: int, cap: int) -> bool:
    return bool(
        not isinstance(value, bool)
        and isinstance(value, int)
        and initial <= value <= cap
        and (value - initial) % step == 0
    )


def _validate_private_sample_archive(
    payload: Mapping[str, Any],
    *,
    context: AcademicLaunchContext,
) -> None:
    import numpy as np

    with np.load(context.paths["private_samples_path"], allow_pickle=False) as archive:
        retained = np.asarray(archive["retained_raw_samples"], dtype=np.float64)
        states = np.asarray(archive["final_worker_states"], dtype=np.float64)
        config_hash = str(archive["config_hash"].item())
        replay_hash = str(archive["private_replay_hash"].item())
    expected_draws = int(payload["retained_results_per_chain"])
    if retained.shape != (expected_draws, 4, 18) or states.shape != (2, 2, 18) or (
        not np.all(np.isfinite(retained))
        or not np.all(np.isfinite(states))
        or config_hash != context.config.hash
        or replay_hash != context.preflight["private_replay_artifact_hash"]
    ):
        raise AcademicCampaignError("academic private sample archive mismatch")


def write_infrastructure_failure(
    context: AcademicLaunchContext,
    *,
    stage: str,
    error: BaseException,
    elapsed_seconds: float,
    classification: str = "infrastructure_failure",
    workers_started: bool | None = None,
    hmc_transition_executed: bool | None = None,
) -> Mapping[str, Any]:
    if classification not in {"infrastructure_failure", "continuation_veto"}:
        raise AcademicCampaignError("unsupported launcher failure classification")
    payload = _with_hash(
        {
            "schema": ACADEMIC_FAILURE_SCHEMA,
            "passed": False,
            "decision": ACADEMIC_BLOCK_DECISION,
            "campaign_id": context.campaign_id,
            "attempt_number": context.attempt_number,
            "failure_classification": classification,
            "stage": str(stage),
            "reason": f"{type(error).__name__}: {error}",
            "config_hash": context.config.hash,
            "run_manifest_artifact_hash": context.manifest["artifact_hash"],
            "preflight_before_runtime_artifact_hash": context.preflight[
                "artifact_hash"
            ],
            "transition_identity_hash": EXPECTED_TRANSITION_IDENTITY,
            "serious_execution_contract_hash": EXPECTED_SERIOUS_EXECUTION_IDENTITY,
            "elapsed_seconds": float(elapsed_seconds),
            "controller_entered": True,
            "workers_started": workers_started,
            "hmc_transition_executed": hmc_transition_executed,
            "serious_runtime_executed": hmc_transition_executed,
            "phase8_executed": False,
            "neutra_executed": False,
            "nonclaims": ACADEMIC_FAILURE_NONCLAIMS,
        }
    )
    atomic_write_json(context.paths["failure_path"], payload)
    return payload


def write_checksum_manifest(
    context: AcademicLaunchContext,
    *,
    terminal_path: Path,
) -> Mapping[str, Any]:
    included = [context.paths["run_manifest_path"], terminal_path]
    for name in ("public_result_path", "failure_path"):
        path = context.paths[name]
        if path.is_file() and path not in included:
            included.append(path)
    for name in ("public_progress_path", "private_samples_path", "log_path"):
        path = context.paths[name]
        if path.is_file():
            included.append(path)
    records = tuple(
        {
            "relative_path": str(path.relative_to(context.run_directory)),
            "file_sha256": artifact_file_sha256(path),
            "byte_count": path.stat().st_size,
        }
        for path in included
    )
    payload = _with_hash(
        {
            "schema": ACADEMIC_CHECKSUM_SCHEMA,
            "campaign_id": context.campaign_id,
            "attempt_number": context.attempt_number,
            "files": list(records),
        }
    )
    atomic_write_json(context.paths["checksums_path"], payload)
    verify_checksum_manifest(context.paths["checksums_path"], context.run_directory)
    return payload


def verify_checksum_manifest(
    path: str | Path,
    run_directory: str | Path,
) -> Mapping[str, Any]:
    payload = _read_json(Path(path))
    if payload.get("schema") != ACADEMIC_CHECKSUM_SCHEMA:
        raise AcademicCampaignError("academic checksum schema mismatch")
    _verify_hash(payload, label="academic checksums")
    root = Path(run_directory).resolve()
    files = payload.get("files")
    if (
        not isinstance(files, Sequence)
        or isinstance(files, (str, bytes))
        or not files
    ):
        raise AcademicCampaignError("academic checksum inventory is empty")
    for record in files:
        if not isinstance(record, Mapping) or set(record) != {
            "relative_path", "file_sha256", "byte_count"
        }:
            raise AcademicCampaignError("academic checksum record mismatch")
        candidate = (root / str(record["relative_path"])).resolve()
        if candidate.parent != root and root not in candidate.parents:
            raise AcademicCampaignError("academic checksum path escapes run directory")
        if artifact_file_sha256(candidate) != record["file_sha256"] or (
            candidate.stat().st_size != record["byte_count"]
        ):
            raise AcademicCampaignError("academic terminal checksum mismatch")
    return payload


def build_implementation_references() -> Mapping[str, Mapping[str, Any]]:
    root = Path(__file__).resolve().parents[2]
    from bayesfilter.inference.hmc_smoke_authority import (
        default_implementation_paths,
    )

    paths = dict(default_implementation_paths(sys.executable))
    paths.update(
        {
            "academic_campaign_module": Path(__file__).resolve(),
            "academic_campaign_launcher": root
            / "scripts/run_hmc_phase7_academic_campaign.py",
            "academic_campaign_tests": root / "tests/test_hmc_academic_campaign.py",
        }
    )
    return {
        name: {
            "path": str(path),
            "file_sha256": artifact_file_sha256(path),
            "byte_count": path.stat().st_size,
        }
        for name, path in paths.items()
    }


def verify_implementation_references(
    references: Mapping[str, Mapping[str, Any]],
) -> None:
    expected = build_implementation_references()
    if dict(references) != dict(expected):
        raise AcademicCampaignError("academic implementation source drift")


def _validate_config_and_preflight(config: Any, preflight: Mapping[str, Any]) -> None:
    if config.payload.get("schema") != (
        "bayesfilter.deterministic_lgssm_hmc_phase7_config.v2"
    ):
        raise AcademicCampaignError("academic campaign requires the V2 config")
    adopted = config.payload.get("adopted_identities", {})
    if adopted.get("transition_identity_hash") != EXPECTED_TRANSITION_IDENTITY or (
        adopted.get("serious_execution_contract_hash")
        != EXPECTED_SERIOUS_EXECUTION_IDENTITY
    ):
        raise AcademicCampaignError("academic campaign typed identity drift")
    if preflight.get("passed") is not True or (
        preflight.get("runtime_executed") is not False
    ):
        raise AcademicCampaignError("academic campaign preflight did not pass")
    identities = preflight.get("identity_hashes")
    if not isinstance(identities, Mapping) or (
        identities.get("transition_identity_hash") != EXPECTED_TRANSITION_IDENTITY
        or identities.get("serious_execution_contract_hash")
        != EXPECTED_SERIOUS_EXECUTION_IDENTITY
    ):
        raise AcademicCampaignError("academic campaign preflight identity drift")
    _verify_hash(preflight, label="academic preflight")


def _validate_attempt_summary(payload: Mapping[str, Any], *, directory: Path) -> None:
    if payload.get("schema") != ACADEMIC_ATTEMPT_SUMMARY_SCHEMA or (
        payload.get("campaign_id") != ACADEMIC_CAMPAIGN_ID
    ):
        raise AcademicCampaignError("academic attempt summary identity mismatch")
    _verify_hash(payload, label="academic attempt summary")
    if Path(str(payload.get("run_directory"))).resolve() != directory.resolve():
        raise AcademicCampaignError("academic attempt summary directory mismatch")
    elapsed = payload.get("elapsed_seconds")
    if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)) or elapsed < 0:
        raise AcademicCampaignError("academic attempt elapsed time mismatch")
    if payload.get("classification") not in {
        "strict_pass",
        "diagnostic_cap_failure",
        "infrastructure_failure",
        "continuation_veto",
    }:
        raise AcademicCampaignError("academic attempt classification mismatch")
    if isinstance(payload.get("exit_code"), bool) or not isinstance(
        payload.get("exit_code"), int
    ):
        raise AcademicCampaignError("academic attempt exit code mismatch")
    for name in (
        "cumulative_wall_time_seconds",
        "remaining_wall_time_seconds",
        "budget_overrun_seconds",
    ):
        value = payload.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            raise AcademicCampaignError(f"academic attempt {name} mismatch")


def _release_lock_handle(handle: Any) -> None:
    if handle is None or getattr(handle, "closed", True):
        return
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()


def _with_hash(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    result = dict(payload)
    result["artifact_hash"] = canonical_artifact_payload_hash(result)
    return result


def _verify_hash(payload: Mapping[str, Any], *, label: str) -> None:
    observed = payload.get("artifact_hash")
    expected = canonical_artifact_payload_hash(
        {name: value for name, value in payload.items() if name != "artifact_hash"}
    )
    if observed != expected:
        raise AcademicCampaignError(f"{label} embedded artifact hash mismatch")


def _read_json(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise AcademicCampaignError(f"JSON artifact must be an object: {path.name}")
    return payload


def _git_commit() -> str:
    try:
        return subprocess.run(
            ("git", "rev-parse", "HEAD"),
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"
