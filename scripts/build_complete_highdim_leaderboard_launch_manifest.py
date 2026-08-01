#!/usr/bin/env python3
"""Build and verify the fail-closed detached launch manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from scripts.launch_complete_highdim_leaderboard import (
        CODEX_SOFT_DEADLINE_SECONDS,
        CONSTRAINED_PATH,
        NAMESPACE_OUTER_DEADLINE_SECONDS,
        OUTER_FINALIZER_DEADLINE_SECONDS,
        OUTER_KILL_DEADLINE_SECONDS,
        OUTER_TERM_DEADLINE_SECONDS,
        PRIMARY_EXPORT_TIMEOUT_SECONDS,
        PROCESS_TERMINATION_DEADLINE_SECONDS,
        PRODUCER_CLOSE_DEADLINE_SECONDS,
        SUPERVISOR_AND_WATCHDOG_HARD_DEADLINE_SECONDS,
        WATCHDOG_VERIFICATION_DEADLINE_SECONDS,
        exact_supervisor_command,
        exact_wrapper_argv,
    )
except ModuleNotFoundError:
    from launch_complete_highdim_leaderboard import (  # type: ignore[no-redef]
        CODEX_SOFT_DEADLINE_SECONDS,
        CONSTRAINED_PATH,
        NAMESPACE_OUTER_DEADLINE_SECONDS,
        OUTER_FINALIZER_DEADLINE_SECONDS,
        OUTER_KILL_DEADLINE_SECONDS,
        OUTER_TERM_DEADLINE_SECONDS,
        PRIMARY_EXPORT_TIMEOUT_SECONDS,
        PROCESS_TERMINATION_DEADLINE_SECONDS,
        PRODUCER_CLOSE_DEADLINE_SECONDS,
        SUPERVISOR_AND_WATCHDOG_HARD_DEADLINE_SECONDS,
        WATCHDOG_VERIFICATION_DEADLINE_SECONDS,
        exact_supervisor_command,
        exact_wrapper_argv,
    )


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / (
    "docs/plans/complete-highdim-leaderboard-exact-command-manifest-2026-07-11.json"
)
DEFAULT_LAUNCH_REVIEW = ROOT / (
    "docs/reviews/bayesfilter-complete-highdim-leaderboard-launch-readiness-"
    "review-receipt-2026-07-11.json"
)
LAUNCHER = Path("/home/chakwong/python/claudecodex/scripts/overnight_gated_launch.sh")
WORKER_SETTINGS = Path("/home/chakwong/.claude/settings.codex-worker.json")
CLAUDE_REVIEW_GATE = Path(
    "/home/chakwong/python/claudecodex/scripts/claude_review_gate.sh"
)
CLAUDE_REVIEW_GUIDE = Path(
    "/home/chakwong/python/claudecodex/docs/claude-review-gate-agent-guide.md"
)
RUNBOOK_TEMPLATE = Path(
    "/home/chakwong/python/claudecodex/docs/templates/"
    "visible-gated-execution-runbook-template.md"
)
LOG_DIR = "docs/plans/logs"
PHASE1_PATH = (
    "docs/plans/bayesfilter-complete-highdim-leaderboard-"
    "phase1-ledh-harness-subplan-2026-07-11.md"
)
PHASE1_REVIEW_RECEIPT = (
    "docs/reviews/bayesfilter-complete-highdim-leaderboard-"
    "phase1-subplan-review-receipt-2026-07-11.json"
)
GPU_PREFLIGHT = (
    "docs/plans/artifacts/complete-highdim-leaderboard/"
    "launch-gpu-preflight-2026-07-11.json"
)
CODEX_PREFLIGHT = (
    "docs/plans/artifacts/complete-highdim-leaderboard/"
    "launch-codex-preflight-2026-07-11.json"
)
ISOLATION_PREFLIGHT = (
    "docs/plans/artifacts/complete-highdim-leaderboard/"
    "launch-isolation-preflight-2026-07-11.json"
)
RUN_ID = "complete-highdim-leaderboard-20260711-221500"
LAUNCH_ROOT = Path(f"/tmp/{RUN_ID}-workspace")
HANDOFF_DIR = ROOT / "docs/plans/logs" / RUN_ID
COPY_SENTINEL_NONCE = "53a8d896f02a35096f2bb8ff28bfb3fa"
SOURCE_SNAPSHOT_ROOT = ROOT / f".complete-highdim-source-snapshot-{RUN_ID}"
SOURCE_INVENTORY = ROOT / (
    "docs/plans/artifacts/complete-highdim-leaderboard/"
    "source-snapshot-inventory-2026-07-11.json"
)
RUNTIME_FINGERPRINT = ROOT / (
    "docs/plans/artifacts/complete-highdim-leaderboard/"
    "runtime-fingerprint-2026-07-11.json"
)
APPROVAL_INSTANCE_ID = "d34c0b4076dd7b1e9cdf225a785fa58c"
APPROVAL_NOT_AFTER_EPOCH = 1783882800
RISK_ACCEPTANCE_AMENDMENT = (
    "docs/plans/bayesfilter-complete-highdim-leaderboard-"
    "run-risk-acceptance-amendment-2026-07-12.md"
)
POST_RUN_INTEGRITY_AUDIT_PLAN = (
    "docs/plans/bayesfilter-complete-highdim-leaderboard-"
    "post-run-integrity-audit-plan-2026-07-12.md"
)
POST_RUN_INTEGRITY_AUDITOR = (
    "scripts/audit_complete_highdim_leaderboard_post_run_integrity.py"
)
CLAUDE_AUDIT_WORKER = (
    "scripts/complete_highdim_leaderboard_claude_audit_worker.sh"
)
POST_LOCK_RECEIPT_WRITER = (
    "scripts/write_complete_highdim_leaderboard_post_lock_receipt.py"
)
OWNER_ACCEPTED_RUN_SCOPED_LIMITATION_IDS = (
    "CLAUDE_TOOL_CAPABILITY",
    "CLAUDE_CODEX_CREDENTIAL_ACCESS",
    "PRIMARY_EXPORT_COMPLETENESS",
    "SEAL_LOCK_TOCTOU",
    "TRUSTED_PREFLIGHT_OUTER_COVERAGE",
)
LAUNCH_REVIEW_REL = (
    "docs/reviews/bayesfilter-complete-highdim-leaderboard-launch-readiness-"
    "review-receipt-2026-07-11.json"
)

BOUND_PATHS = (
    "AGENTS.md",
    "docs/plans/bayesfilter-complete-highdim-leaderboard-master-program-2026-07-11.md",
    "docs/plans/bayesfilter-complete-highdim-leaderboard-phase0-boundary-freeze-subplan-2026-07-11.md",
    "docs/plans/bayesfilter-complete-highdim-leaderboard-phase0-boundary-freeze-result-2026-07-11.md",
    PHASE1_PATH,
    PHASE1_REVIEW_RECEIPT,
    "docs/plans/bayesfilter-complete-highdim-leaderboard-visible-gated-execution-runbook-2026-07-11.md",
    "docs/plans/bayesfilter-complete-highdim-leaderboard-detached-overnight-supervisor-plan-2026-07-11.md",
    "docs/plans/bayesfilter-complete-highdim-leaderboard-overnight-supervisor-prompt-2026-07-11.md",
    RISK_ACCEPTANCE_AMENDMENT,
    POST_RUN_INTEGRITY_AUDIT_PLAN,
    "docs/plans/artifacts/complete-highdim-leaderboard/phase0-boundary-freeze-2026-07-11.json",
    GPU_PREFLIGHT,
    CODEX_PREFLIGHT,
    "docs/plans/artifacts/complete-highdim-leaderboard/launch-codex-preflight-events-2026-07-11.jsonl",
    "docs/plans/artifacts/complete-highdim-leaderboard/launch-codex-preflight-final-2026-07-11.txt",
    "docs/plans/artifacts/complete-highdim-leaderboard/launch-codex-preflight-stderr-2026-07-11.log",
    ISOLATION_PREFLIGHT,
    "docs/reviews/bayesfilter-complete-highdim-leaderboard-phase0-review-receipts-2026-07-11.json",
    "docs/reviews/bayesfilter-complete-highdim-leaderboard-claude-availability-2026-07-11.md",
    "docs/reviews/bayesfilter-complete-highdim-leaderboard-phase1-codex-substitute-review-iter1-2026-07-11.md",
    "docs/reviews/bayesfilter-complete-highdim-leaderboard-phase1-codex-substitute-review-iter2-2026-07-11.md",
    "docs/reviews/bayesfilter-complete-highdim-leaderboard-phase1-codex-substitute-review-iter3-2026-07-11.md",
    "docs/reviews/bayesfilter-complete-highdim-leaderboard-phase1-codex-substitute-review-iter4-2026-07-11.md",
    "docs/reviews/bayesfilter-complete-highdim-leaderboard-phase1-codex-substitute-review-iter5-2026-07-11.md",
    "scripts/build_complete_highdim_leaderboard_launch_manifest.py",
    "scripts/build_complete_highdim_leaderboard_phase0_freeze.py",
    "scripts/audit_complete_highdim_leaderboard_phase0_freeze.py",
    "scripts/export_complete_highdim_leaderboard_isolated_changes.py",
    "scripts/write_complete_highdim_leaderboard_terminal_status.py",
    "scripts/run_complete_highdim_leaderboard_gpu_preflight.py",
    "scripts/run_complete_highdim_leaderboard_codex_preflight.sh",
    "scripts/run_complete_highdim_leaderboard_isolation_preflight.py",
    "scripts/write_complete_highdim_leaderboard_codex_preflight.py",
    "scripts/complete_highdim_leaderboard_overnight_supervisor.sh",
    "scripts/complete_highdim_leaderboard_overnight_supervisor.py",
    "scripts/complete_highdim_leaderboard_watchdog.py",
    "scripts/complete_highdim_leaderboard_namespace_entrypoint.sh",
    "scripts/complete_highdim_leaderboard_codex_sandbox_entrypoint.sh",
    "scripts/complete_highdim_leaderboard_exec_codex_after_boundary_check.py",
    "scripts/prepare_complete_highdim_leaderboard_launch.py",
    "scripts/launch_complete_highdim_leaderboard.py",
    "scripts/verify_complete_highdim_leaderboard_review_receipt.py",
    "scripts/freeze_complete_highdim_leaderboard_source_snapshot.py",
    "scripts/build_complete_highdim_leaderboard_runtime_fingerprint.py",
    "scripts/finalize_complete_highdim_leaderboard_handoff.py",
    CLAUDE_AUDIT_WORKER,
    POST_LOCK_RECEIPT_WRITER,
    POST_RUN_INTEGRITY_AUDITOR,
    "scripts/complete_highdim_leaderboard_outer_launch_boundary.sh",
    "scripts/complete_highdim_leaderboard_exact_wrapper.sh",
    "tests/test_complete_highdim_leaderboard_launch_manifest.py",
    "tests/test_complete_highdim_leaderboard_phase0_freeze.py",
    "tests/test_complete_highdim_leaderboard_isolated_export.py",
    "tests/test_complete_highdim_leaderboard_overnight_supervisor.py",
    "tests/test_complete_highdim_leaderboard_watchdog.py",
    "tests/test_complete_highdim_leaderboard_source_snapshot.py",
    "tests/test_complete_highdim_leaderboard_runtime_fingerprint.py",
    "tests/test_complete_highdim_leaderboard_handoff_finalizer.py",
    "tests/test_complete_highdim_leaderboard_review_receipt.py",
    "tests/test_complete_highdim_leaderboard_post_run_integrity.py",
)

OVERLAY_PATHS = (
    "docs/plans/bayesfilter-complete-highdim-leaderboard-visible-execution-ledger-2026-07-11.md",
    "docs/plans/bayesfilter-complete-highdim-leaderboard-visible-stop-handoff-2026-07-11.md",
)

CONTROL_PATHS = (
    "docs/reviews/bayesfilter-complete-highdim-leaderboard-launch-plan-codex-substitute-review-iter1-2026-07-11.md",
    "docs/reviews/bayesfilter-complete-highdim-leaderboard-launch-implementation-codex-substitute-review-iter1-2026-07-11.md",
    "docs/reviews/bayesfilter-complete-highdim-leaderboard-launch-manifest-codex-substitute-review-iter1-2026-07-11.md",
)

EXTERNAL_PATHS = (
    LAUNCHER,
    WORKER_SETTINGS,
    CLAUDE_REVIEW_GATE,
    CLAUDE_REVIEW_GUIDE,
    RUNBOOK_TEMPLATE,
)

EXECUTABLE_NAMES = (
    "bash",
    "node",
    "codex",
    "claude",
    "timeout",
    "unshare",
    "setsid",
    "nvidia-smi",
    "python",
    "git",
    "cp",
    "mount",
    "env",
    "sort",
    "mkdir",
    "rm",
    "sleep",
    "tr",
    "date",
    "grep",
    "tail",
    "sed",
    "cat",
    "findmnt",
    "awk",
    "cut",
    "stat",
    "touch",
    "setpriv",
    "setsid",
    "basename",
    "chmod",
    "umount",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_binding(path: Path, *, label: str | None = None) -> dict[str, Any]:
    resolved = path.resolve() if path.exists() else None
    return {
        "path": label or str(path),
        "exists": path.is_file(),
        "resolved_path": str(resolved) if resolved is not None else None,
        "sha256": _sha256(path) if path.is_file() else None,
    }


def _phase_status(path: Path) -> str | None:
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("Status: `") and line.endswith("`"):
            return line[len("Status: `") : -1]
    return None


def _read_json(path: Path) -> Mapping[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return value if isinstance(value, Mapping) else None


def _phase1_review_state() -> dict[str, Any]:
    subplan = ROOT / PHASE1_PATH
    receipt_path = ROOT / PHASE1_REVIEW_RECEIPT
    receipt = _read_json(receipt_path)
    observed_sha = _sha256(subplan) if subplan.is_file() else None
    valid = bool(
        receipt
        and receipt.get("schema_version")
        == "bayesfilter.complete_highdim_leaderboard.review_receipt.v1"
        and receipt.get("verdict") == "AGREE"
        and receipt.get("reviewed_path") == PHASE1_PATH
        and receipt.get("reviewed_sha256") == observed_sha
        and receipt.get("reviewer_type") == "fresh_codex_readonly_substitute"
    )
    return {
        "path": PHASE1_REVIEW_RECEIPT,
        "exists": receipt is not None,
        "reviewed_sha256": receipt.get("reviewed_sha256") if receipt else None,
        "observed_subplan_sha256": observed_sha,
        "verdict": receipt.get("verdict") if receipt else None,
        "valid": valid,
    }


def _gpu_preflight_state() -> dict[str, Any]:
    path = ROOT / GPU_PREFLIGHT
    script = ROOT / "scripts/run_complete_highdim_leaderboard_gpu_preflight.py"
    payload = _read_json(path)
    valid = bool(
        payload
        and payload.get("preflight_pass") is True
        and payload.get("preflight_script_sha256") == _sha256(script)
        and payload.get("gpu_trust_basis")
        == "owner_designated_managed_session_visible_gpu_trusted"
        and payload.get("nvidia_smi_pass") is True
        and payload.get("jit_compile") is True
        and payload.get("tf32_execution_enabled") is True
        and payload.get("physical_gpus")
        and payload.get("logical_gpus")
        and "GPU" in str(payload.get("output_device", "")).upper()
    )
    return {"path": GPU_PREFLIGHT, "exists": payload is not None, "valid": valid}


def _codex_preflight_state() -> dict[str, Any]:
    path = ROOT / CODEX_PREFLIGHT
    runner = ROOT / "scripts/run_complete_highdim_leaderboard_codex_preflight.sh"
    writer = ROOT / "scripts/write_complete_highdim_leaderboard_codex_preflight.py"
    payload = _read_json(path)
    valid = bool(
        payload
        and payload.get("preflight_pass") is True
        and payload.get("runner_script_sha256") == _sha256(runner)
        and payload.get("writer_script_sha256") == _sha256(writer)
        and payload.get("exit_code") == 0
        and payload.get("probe_token") == "CODEX_PROBE_OK"
        and payload.get("noninteractive") is True
        and payload.get("trusted_execution") is True
    )
    return {
        "path": CODEX_PREFLIGHT,
        "exists": payload is not None,
        "preliminary_health_valid": valid,
        "sufficient_for_conditional_launch_authorization": False,
    }


def _isolation_preflight_state() -> dict[str, Any]:
    path = ROOT / ISOLATION_PREFLIGHT
    script = ROOT / "scripts/run_complete_highdim_leaderboard_isolation_preflight.py"
    payload = _read_json(path)
    boundary = payload.get("boundary") if payload else None
    sandbox_receipt = payload.get("sandbox_receipt") if payload else None
    valid = bool(
        payload
        and payload.get("schema_version")
        == "bayesfilter.complete_highdim_leaderboard.isolation_preflight.v3"
        and payload.get("mode") == "fake-gpu-boundary"
        and payload.get("preflight_pass") is True
        and payload.get("preflight_script_sha256") == _sha256(script)
        and payload.get("launcher_sha256") == _sha256(LAUNCHER)
        and payload.get("launcher_exit_code") == 0
        and payload.get("handoff_ready") is True
        and payload.get("source_workspace_unchanged") is True
        and payload.get("isolated_workspace_changed") is True
        and payload.get("fake_supervisor_exit_code") == 0
        and payload.get("host_sibling_unchanged") is True
        and isinstance(sandbox_receipt, Mapping)
        and sandbox_receipt.get("cap_eff") == 0
        and sandbox_receipt.get("cap_bnd") == 0
        and sandbox_receipt.get("no_new_privs") == 1
        and sandbox_receipt.get("pid_in_private_namespace") == 1
        and isinstance(sandbox_receipt.get("pid_namespace_inode"), int)
        and sandbox_receipt.get("pid_namespace_inode", 0) > 0
        and sandbox_receipt.get("handoff_read_only") is True
        and sandbox_receipt.get("handoff_contents_hidden") is True
        and sandbox_receipt.get("support_read_only") is True
        and sandbox_receipt.get("support_hashes_verified") is True
        and len(
            sandbox_receipt.get("staging_aliases_hidden_and_read_only", [])
        )
        == 2
        and sandbox_receipt.get("sibling_home_hidden") is True
        and sandbox_receipt.get("selected_runtime_mounts_only") is True
        and isinstance(boundary, Mapping)
        and boundary.get("jit_compile") is True
        and boundary.get("tf32_execution_enabled") is True
        and boundary.get("physical_gpus")
        and boundary.get("logical_gpus")
        and "GPU" in str(boundary.get("output_device", "")).upper()
        and boundary.get("output_finite") is True
    )
    return {"path": ISOLATION_PREFLIGHT, "exists": payload is not None, "valid": valid}


def _preflight_state() -> dict[str, Any]:
    phase0 = ROOT / (
        "docs/plans/bayesfilter-complete-highdim-leaderboard-"
        "phase0-boundary-freeze-result-2026-07-11.md"
    )
    phase1_review = _phase1_review_state()
    gpu = _gpu_preflight_state()
    codex = _codex_preflight_state()
    isolation = _isolation_preflight_state()
    phase0_status = _phase_status(phase0)
    ready_except_human = bool(
        phase0_status == "PASS_PHASE0_BOUNDARY_FREEZE"
        and phase1_review["valid"]
        and gpu["valid"]
        and codex["preliminary_health_valid"]
        and isolation["valid"]
    )
    return {
        "phase0_status": phase0_status,
        "phase1_subplan_review": phase1_review,
        "gpu": gpu,
        "codex": codex,
        "isolation": isolation,
        "human_launch_approval_recorded": False,
        "static_evidence_ready_for_launch_review": ready_except_human,
        "ready_except_human_approval": False,
        "ready_to_request_human_approval": False,
        "launch_authorized": False,
    }


def _executable_bindings() -> list[dict[str, Any]]:
    bindings = []
    for name in EXECUTABLE_NAMES:
        found = shutil.which(name, path=CONSTRAINED_PATH)
        path = Path(found) if found else None
        bindings.append(
            {
                "name": name,
                **(
                    _file_binding(path, label=found)
                    if path is not None
                    else {
                        "path": None,
                        "exists": False,
                        "resolved_path": None,
                        "sha256": None,
                    }
                ),
            }
        )
    return bindings


def build_manifest() -> dict[str, Any]:
    bindings = [_file_binding(ROOT / rel, label=rel) for rel in BOUND_PATHS]
    external = [_file_binding(path) for path in EXTERNAL_PATHS]
    supervisor_cmd = exact_supervisor_command(
        nonce=COPY_SENTINEL_NONCE, source_snapshot_root=SOURCE_SNAPSHOT_ROOT
    )
    launch_argv = exact_wrapper_argv(
        run_id=RUN_ID,
        launch_root=LAUNCH_ROOT,
        nonce=COPY_SENTINEL_NONCE,
        manifest=DEFAULT_OUTPUT,
        launch_review_receipt=ROOT / LAUNCH_REVIEW_REL,
        source_snapshot_root=SOURCE_SNAPSHOT_ROOT,
        source_inventory=SOURCE_INVENTORY,
        runtime_fingerprint=RUNTIME_FINGERPRINT,
        approval_instance_id=APPROVAL_INSTANCE_ID,
        approval_not_after_epoch=APPROVAL_NOT_AFTER_EPOCH,
    )
    overlay = [_file_binding(ROOT / rel, label=rel) for rel in OVERLAY_PATHS]
    control = [_file_binding(ROOT / rel, label=rel) for rel in CONTROL_PATHS]
    source_inventory = _file_binding(SOURCE_INVENTORY)
    runtime_fingerprint = _file_binding(RUNTIME_FINGERPRINT)
    fresh_preflight_base = [
        "/usr/bin/timeout",
        "--signal=TERM",
        "--kill-after=20s",
        "240s",
        "/home/chakwong/anaconda3/envs/tf-gpu/bin/python",
        str(ROOT / "scripts/run_complete_highdim_leaderboard_isolation_preflight.py"),
    ]
    return {
        "schema_version": "bayesfilter.complete_highdim_leaderboard.launch_manifest.v7",
        "program_root": str(ROOT),
        "environment": {
            "path": CONSTRAINED_PATH,
            "python_executable": shutil.which("python", path=CONSTRAINED_PATH),
            "conda_prefix": "/home/chakwong/anaconda3/envs/tf-gpu",
        },
        "repository_bound_files": bindings,
        "snapshot_overlay_files": overlay,
        "codex_readable_initial_disclosure": {
            "frozen_source_inventory_sha256": source_inventory["sha256"],
            "overlay_files": overlay,
            "supervisor_prompt_inside_frozen_snapshot": _file_binding(
                ROOT
                / "docs/plans/bayesfilter-complete-highdim-leaderboard-"
                "overnight-supervisor-prompt-2026-07-11.md"
            ),
            "support_files": [
                {
                    "destination_name": "trusted-review-verifier.py",
                    "source": _file_binding(
                        ROOT
                        / "scripts/verify_complete_highdim_leaderboard_review_receipt.py"
                    ),
                },
                {
                    "destination_name": "trusted-claude-review-gate.sh",
                    "source": _file_binding(CLAUDE_REVIEW_GATE),
                },
                {
                    "destination_name": "trusted-claude-worker.sh",
                    "source": _file_binding(
                        ROOT / CLAUDE_AUDIT_WORKER,
                        label=CLAUDE_AUDIT_WORKER,
                    ),
                },
                {
                    "destination_name": "trusted-claude-worker-settings.json",
                    "source": _file_binding(WORKER_SETTINGS),
                },
            ],
            "dynamic_handoff_contents_hidden": True,
            "dynamic_handoff_staging_aliases_hidden": True,
            "runtime_trees": {
                "tensorflow_conda_tree": "/home/chakwong/anaconda3/envs/tf-gpu",
                "node_codex_claude_tree": (
                    "/home/chakwong/.nvm/versions/node/v22.23.1"
                ),
                "read_only": True,
                "selected_payloads_bound_by_runtime_fingerprint_sha256": (
                    runtime_fingerprint["sha256"]
                ),
                "entire_mounted_tree_byte_complete_fingerprint": False,
            },
            "system_execution_surface": {
                "linux_system_libraries_and_executables": True,
                "private_proc_and_temporary_filesystems": True,
                "approved_nvidia_device_and_driver_interfaces": True,
                "network_transport_for_approved_model_api_calls": True,
                "unrelated_sibling_home_hidden": True,
                "mounted_host_drives_hidden": True,
            },
            "credential_channels": {
                "ephemeral_private_codex_auth_copy": True,
                "anthropic_environment_channel_for_bounded_reviews": True,
                "claude_can_read_ephemeral_private_codex_auth_copy": True,
                "secret_values_logged_or_hashed": False,
                "unrelated_inherited_environment_stripped": True,
            },
            "model_session_state_private_tmpfs": True,
            "claude_technical_access": {
                "inherits_isolated_codex_filesystem_and_runtime_surface": True,
                "copied_snapshot_readable": True,
                "copied_snapshot_os_writable": True,
                "private_temporary_storage_os_writable": True,
                "support_files_readable": True,
                "runtime_trees_readable": True,
                "system_and_network_interfaces_readable_as_required": True,
                "anthropic_credential_channel_available": True,
                "ephemeral_private_codex_auth_copy_readable": True,
                "dynamic_handoff_hidden": True,
                "unrelated_sibling_home_and_host_drives_hidden": True,
                "single_path_is_filesystem_enforced": False,
                "read_only_role_is_filesystem_enforced": False,
                "edit_and_command_tools_technically_available": True,
                "read_only_role_enforced_by_bound_settings": False,
                "read_only_role_is_instruction_and_prompt_contract": True,
            },
            "claude_review_prompt_policy": {
                "single_exact_path_first": True,
                "read_only_role": True,
                "not_a_filesystem_isolation_boundary": True,
            },
        },
        "live_control_files_not_disclosed_to_detached_codex": control,
        "frozen_source_snapshot": {
            "root": str(SOURCE_SNAPSHOT_ROOT),
            "inventory": source_inventory,
            "must_verify_before_materialization": True,
            "materialization_must_use_inventory_only": True,
            "live_worktree_is_not_copy_source": True,
        },
        "runtime_fingerprint": runtime_fingerprint,
        "owner_accepted_run_scoped_limitations": {
            "run_id": RUN_ID,
            "repository_default": False,
            "reusable_for_other_runs": False,
            "technical_guarantees_repaired": False,
            "accepted_limitation_ids": list(
                OWNER_ACCEPTED_RUN_SCOPED_LIMITATION_IDS
            ),
            "authorization_amendment": _file_binding(
                ROOT / RISK_ACCEPTANCE_AMENDMENT,
                label=RISK_ACCEPTANCE_AMENDMENT,
            ),
            "sixth_review_round": {
                "authorized": True,
                "iteration": 6,
                "scope": (
                    "bind_this_waiver_post_run_audit_and_exact_launch_package_only"
                ),
                "ordinary_phase_review_limit_unchanged": 5,
            },
            "accepted_limitations": [
                {
                    "id": "CLAUDE_TOOL_CAPABILITY",
                    "accepted_not_repaired": True,
                    "meaning": (
                        "bound Claude settings and worker technically permit edit "
                        "and command tools; read-only is an instruction contract"
                    ),
                },
                {
                    "id": "CLAUDE_CODEX_CREDENTIAL_ACCESS",
                    "accepted_not_repaired": True,
                    "meaning": (
                        "Claude may read the ephemeral private Codex auth copy "
                        "through inherited CODEX_HOME"
                    ),
                },
                {
                    "id": "PRIMARY_EXPORT_COMPLETENESS",
                    "accepted_not_repaired": True,
                    "meaning": (
                        "launch-time validators do not independently require the "
                        "complete primary export payload set"
                    ),
                },
                {
                    "id": "SEAL_LOCK_TOCTOU",
                    "accepted_not_repaired": True,
                    "meaning": (
                        "the final seal is written before all handoff aliases are "
                        "locked read-only; a post-lock receipt detects final mismatch "
                        "but cannot prove no transient pre-lock write occurred"
                    ),
                },
                {
                    "id": "TRUSTED_PREFLIGHT_OUTER_COVERAGE",
                    "accepted_not_repaired": True,
                    "meaning": (
                        "trusted preflight covers the synthetic inner boundary, "
                        "not the exact production outer wrapper and seal route"
                    ),
                },
            ],
        },
        "mandatory_post_run_integrity_audit": {
            "required_before_completion_or_release_claim": True,
            "plan": _file_binding(
                ROOT / POST_RUN_INTEGRITY_AUDIT_PLAN,
                label=POST_RUN_INTEGRITY_AUDIT_PLAN,
            ),
            "auditor": _file_binding(
                ROOT / POST_RUN_INTEGRITY_AUDITOR,
                label=POST_RUN_INTEGRITY_AUDITOR,
            ),
            "auditor_scope": "structural_stage_only",
            "structural_stage_pass_is_not_full_audit_pass": True,
            "post_lock_receipt": {
                "path": f"/tmp/{RUN_ID}-post-lock-integrity.json",
                "writer": _file_binding(
                    ROOT / POST_LOCK_RECEIPT_WRITER,
                    label=POST_LOCK_RECEIPT_WRITER,
                ),
                "required_for_structural_pass": True,
                "outside_self_excluding_seal": True,
            },
            "credential_leak_scan": {
                "current_anthropic_values_scanned_in_memory": True,
                "codex_auth_string_values_scanned_in_memory": True,
                "secret_values_or_hashes_persisted": False,
                "handoff_and_safe_archive_members_required_zero_matches": True,
            },
            "claude_tool_audit": {
                "worker": _file_binding(
                    ROOT / CLAUDE_AUDIT_WORKER,
                    label=CLAUDE_AUDIT_WORKER,
                ),
                "raw_stream_and_tool_metadata_preserved": True,
                "observed_non_read_only_tool_use_vetoes_full_audit": True,
                "technical_tool_capability_removed": False,
            },
            "semantic_inspection_receipt_schema": (
                "bayesfilter.complete_highdim_leaderboard."
                "post_run_semantic_inspection.v1"
            ),
            "semantic_inspection_receipt_required": True,
            "exported_phase8_phase9_completion_checks_required_when_claimed": True,
            "phase8_phase9_validator_exit_zero_and_all_checks_required": True,
            "separate_post_run_result_required": True,
            "pass_verdict_required": "PASS_POST_RUN_INTEGRITY_AUDIT",
            "launch_itself_grants_completion_or_release_authority": False,
        },
        "external_bound_files": external,
        "runtime_executables": _executable_bindings(),
        "supervisor_command": supervisor_cmd,
        "constrained_runtime_path": CONSTRAINED_PATH,
        "runtime_deadlines_seconds": {
            "common_clock_origin": "exact_wrapper_LAUNCH_STARTED_MONOTONIC",
            "codex_soft_cutoff": CODEX_SOFT_DEADLINE_SECONDS,
            "codex_termination_deadline": PROCESS_TERMINATION_DEADLINE_SECONDS,
            "primary_export_timeout": PRIMARY_EXPORT_TIMEOUT_SECONDS,
            "watchdog_primary_verification_deadline": (
                WATCHDOG_VERIFICATION_DEADLINE_SECONDS
            ),
            "namespace_outer_timeout": NAMESPACE_OUTER_DEADLINE_SECONDS,
            "supervisor_and_watchdog_hard_deadline": (
                SUPERVISOR_AND_WATCHDOG_HARD_DEADLINE_SECONDS
            ),
            "producer_close_deadline": PRODUCER_CLOSE_DEADLINE_SECONDS,
            "outer_finalizer_deadline": OUTER_FINALIZER_DEADLINE_SECONDS,
            "outer_term_deadline": OUTER_TERM_DEADLINE_SECONDS,
            "outer_kill_deadline": OUTER_KILL_DEADLINE_SECONDS,
        },
        "concrete_run": {
            "run_id": RUN_ID,
            "launch_root": str(LAUNCH_ROOT),
            "handoff_dir": str(HANDOFF_DIR),
            "copy_sentinel_nonce": COPY_SENTINEL_NONCE,
            "source_snapshot_root": str(SOURCE_SNAPSHOT_ROOT),
            "source_inventory": str(SOURCE_INVENTORY),
            "runtime_fingerprint": str(RUNTIME_FINGERPRINT),
            "approval_instance_id": APPROVAL_INSTANCE_ID,
            "approval_not_after_epoch": APPROVAL_NOT_AFTER_EPOCH,
            "launch_review_receipt": LAUNCH_REVIEW_REL,
            "external_codex_workspace_disclosure_requires_human_approval": True,
            "full_claude_technical_surface_disclosure_requires_human_approval": True,
            "single_path_claude_prompt_policy_requires_human_approval": True,
            "trusted_gpu_and_model_api_execution_requires_human_approval": True,
            "final_exact_launch_command_requires_fresh_human_approval": True,
            "final_exact_launch_command_approved": False,
            "completion_or_release_claim_held_for_post_run_integrity_audit": True,
        },
        "exact_wrapper_argv": launch_argv,
        "exact_wrapper_shell": shlex.join(launch_argv),
        "preflight": _preflight_state(),
        "authorization_state_machine": {
            "static_technical_evidence_ready_to_request_human_approval": (
                _preflight_state()["static_evidence_ready_for_launch_review"]
            ),
            "launch_review_receipt_required_before_approval_request": True,
            "run_scoped_sixth_review_receipt_required": True,
            "human_approval_recorded": False,
            "post_approval_fresh_probes_passed": False,
            "conditional_copy_and_launch_authorized": False,
            "launch_or_scientific_authority_granted_by_manifest": False,
            "completion_or_release_authority_granted_by_manifest": False,
        },
        "post_approval_fresh_preflights": {
            "required_before_copy": True,
            "trusted_or_escalated_execution_required": True,
            "parent_identity": {
                "run_id": RUN_ID,
                "nonce": COPY_SENTINEL_NONCE,
                "approval_instance_id": APPROVAL_INSTANCE_ID,
            },
            "fake_gpu_boundary": {
                "argv_prefix": fresh_preflight_base,
                "mode": "fake-gpu-boundary",
                "output": str(HANDOFF_DIR / f"{RUN_ID}-fresh-boundary-gpu-preflight.json"),
                "expected_schema": (
                    "bayesfilter.complete_highdim_leaderboard.isolation_preflight.v3"
                ),
            },
            "real_restricted_codex": {
                "argv_prefix": fresh_preflight_base,
                "mode": "real-codex",
                "output": str(HANDOFF_DIR / f"{RUN_ID}-fresh-restricted-codex-preflight.json"),
                "expected_schema": (
                    "bayesfilter.complete_highdim_leaderboard.isolation_preflight.v3"
                ),
            },
            "acceptance": {
                "preflight_pass": True,
                "mode_must_match": True,
                "parent_identity_must_match": True,
                "artifact_mtime_not_before_launch_clock": True,
                "any_failure_vetoes_copy": True,
                "approval_expiry_rechecked_before_each_probe": True,
                "approval_expiry_rechecked_before_copy_and_supervisor_launch": True,
            },
            "preliminary_outside_boundary_codex_probe_is_not_sufficient": True,
        },
        "approval_contract": {
            "human_approval_required_after_static_reviewed_readiness": True,
            "human_approval_scope": [
                "this exact wrapper argv and one-time approval instance",
                "restricted Codex access to the exact frozen source snapshot",
                "restricted Codex access to the two exact hash-bound overlays",
                "restricted Codex access to four exact hash-bound support files, including the audited Claude stream wrapper in place of the generic worker",
                "restricted Codex access to the exact frozen supervisor prompt",
                "read-only access to the runtime-fingerprint-bound TensorFlow/TFP/CUDA conda tree",
                "read-only access to the runtime-fingerprint-bound Node/Codex/Claude tree",
                "ordinary Linux system libraries, executables, private proc/tmp, network transport, and approved NVIDIA device interfaces required by the reviewed run",
                "an ephemeral private Codex authentication copy and configured Anthropic credential channel, with secret values neither logged nor hashed",
                "dynamic handoff contents hidden from Codex",
                "all temporary staging aliases hidden from Codex",
                "Claude inherits the isolated Codex process surface, including OS-level read/write access to the copied repository and private temporary storage plus read access to exposed runtime/support files, required system/network interfaces, the configured Anthropic credential channel, and the ephemeral private Codex authentication copy",
                "Claude edit and command tools are technically available under the bound worker/settings; read-only and single-path review are instruction/prompt contracts, not tool or filesystem isolation boundaries",
                "the five owner-accepted limitations apply only to complete-highdim-leaderboard-20260711-221500 and do not become repository defaults",
                "a separate post-run integrity audit must require structural exit zero and PASS_STRUCTURAL_POST_RUN_INTEGRITY, the external post-lock receipt, zero current credential-value matches, no observed non-read-only Claude tool use, a semantic-inspection pass, and Phase 8/9 validator exit zero with every check passing before any completion or release claim",
                "trusted GPU/XLA work and Codex/Claude model API calls",
            ],
            "claude_or_codex_review_cannot_authorize_launch": True,
            "claude_or_codex_review_cannot_authorize_scientific_claims": True,
            "automatic_merge_back": False,
            "commit_push_merge_allowed": False,
            "nested_detached_launches_allowed": False,
            "primary_export_only": True,
            "watchdog_may_read_or_write_launch_copy": False,
            "fallback_export_allowed": False,
            "writable_source_surface": str(HANDOFF_DIR),
            "writable_source_surface_must_be_fresh_and_empty": True,
            "launch_root_must_not_exist": True,
            "source_snapshot_root_must_exist_and_match_inventory": True,
            "one_time_approval_consumed_by_fresh_handoff": True,
            "review_receipt_verified_before_live_handoff_creation": True,
            "approval_expiry_checked_at_each_disclosure_copy_and_launch_boundary": True,
            "primary_only_export_must_be_observed_by_artifact_enumeration": True,
            "whole_outer_pid_namespace_quiescence_required_before_outcome_and_seal": True,
            "owner_accepted_limitations_are_not_repaired_guarantees": True,
            "final_exact_launch_command_requires_separate_human_approval": True,
            "completion_and_release_held_for_post_run_integrity_audit": True,
            "post_lock_receipt_is_detection_not_absence_of_race_proof": True,
            "approval_expires_at_epoch": APPROVAL_NOT_AFTER_EPOCH,
        },
    }


def _verify_launch_review(path: Path, manifest_path: Path) -> None:
    receipt = _read_json(path)
    if not receipt:
        raise ValueError(f"launch review receipt missing or invalid: {path}")
    try:
        expected_manifest_path = str(manifest_path.relative_to(ROOT))
    except ValueError:
        expected_manifest_path = str(manifest_path)
    if receipt.get("verdict") != "AGREE":
        raise ValueError("launch review verdict is not AGREE")
    if receipt.get("reviewed_manifest_path") != expected_manifest_path:
        raise ValueError("launch review receipt names the wrong manifest")
    if receipt.get("reviewed_manifest_sha256") != _sha256(manifest_path):
        raise ValueError("launch review receipt does not bind the current manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    limitations = manifest["owner_accepted_run_scoped_limitations"]
    audit = manifest["mandatory_post_run_integrity_audit"]
    if not (
        receipt.get("schema_version")
        == "bayesfilter.complete_highdim_leaderboard.launch_readiness_review_receipt.v2"
        and receipt.get("iteration") == 6
        and receipt.get("review_scope")
        == "run_scoped_waiver_post_run_audit_and_exact_launch_package"
        and receipt.get("accepted_limitation_ids")
        == list(OWNER_ACCEPTED_RUN_SCOPED_LIMITATION_IDS)
        and receipt.get("risk_acceptance_amendment_sha256")
        == limitations["authorization_amendment"]["sha256"]
        and receipt.get("post_run_integrity_audit_plan_sha256")
        == audit["plan"]["sha256"]
        and receipt.get("post_run_integrity_auditor_sha256")
        == audit["auditor"]["sha256"]
        and receipt.get("final_exact_launch_command_authorized") is False
        and receipt.get("completion_or_release_authority_granted") is False
    ):
        raise ValueError("launch review receipt does not bind the run-scoped waiver")
    concrete = manifest["concrete_run"]
    wrapper = manifest["exact_wrapper_argv"]
    for key, expected in (
        ("run_id", concrete["run_id"]),
        ("launch_root", concrete["launch_root"]),
        ("nonce", concrete["copy_sentinel_nonce"]),
        ("source_snapshot_root", concrete["source_snapshot_root"]),
        ("approval_instance_id", concrete["approval_instance_id"]),
        ("approval_not_after_epoch", concrete["approval_not_after_epoch"]),
        ("exact_wrapper_argv", wrapper),
    ):
        if receipt.get(key) != expected:
            raise ValueError(f"launch review receipt does not bind {key}")
    if receipt.get("source_inventory_sha256") != manifest["frozen_source_snapshot"][
        "inventory"
    ]["sha256"]:
        raise ValueError("launch review receipt does not bind source inventory")
    if receipt.get("runtime_fingerprint_sha256") != manifest["runtime_fingerprint"][
        "sha256"
    ]:
        raise ValueError("launch review receipt does not bind runtime fingerprint")


def _write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--require-ready-except-human", action="store_true")
    parser.add_argument("--require-reviewed-static-readiness", action="store_true")
    parser.add_argument("--launch-review-receipt", type=Path, default=DEFAULT_LAUNCH_REVIEW)
    args = parser.parse_args(argv)
    output = args.output if args.output.is_absolute() else ROOT / args.output
    review_path = (
        args.launch_review_receipt
        if args.launch_review_receipt.is_absolute()
        else ROOT / args.launch_review_receipt
    )
    expected = build_manifest()
    if args.check:
        observed = json.loads(output.read_text(encoding="utf-8"))
        if observed != expected:
            raise ValueError("launch manifest is stale or a bound dependency drifted")
        if args.require_ready_except_human:
            raise ValueError(
                "ready-except-human is not a valid v5 state; use reviewed static readiness"
            )
        if args.require_reviewed_static_readiness:
            if (
                expected["preflight"]["static_evidence_ready_for_launch_review"]
                is not True
            ):
                raise ValueError("static technical evidence is not ready for review")
            _verify_launch_review(review_path, output)
        print(f"LAUNCH_MANIFEST_CHECK_PASS {output}")
        return 0
    _write(output, expected)
    print(f"LAUNCH_MANIFEST_WRITTEN {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
