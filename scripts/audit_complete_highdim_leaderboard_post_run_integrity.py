#!/usr/bin/env python3
"""Read-only structural audit for the one waived leaderboard handoff."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any, Sequence


RUN_ID = "complete-highdim-leaderboard-20260711-221500"
MANIFEST_SCHEMA = "bayesfilter.complete_highdim_leaderboard.launch_manifest.v7"
SEAL_SCHEMA = "bayesfilter.complete_highdim_leaderboard.final_handoff_seal.v1"
EXPORT_HASH_SCHEMA = "bayesfilter.complete_highdim_leaderboard.export_hashes.v1"
EXPORT_MANIFEST_SCHEMA = (
    "bayesfilter.complete_highdim_leaderboard.isolated_export.v1"
)
PRIMARY_FILES = {
    f"{RUN_ID}-primary-isolated-change-manifest.json",
    f"{RUN_ID}-primary-isolated-changed-files.tar.gz",
    f"{RUN_ID}-primary-isolated-tracked.diff",
    f"{RUN_ID}-primary-isolated-git-status.txt",
    f"{RUN_ID}-primary-export-sha256.json",
}
REQUIRED_CONTROL_FILES = {
    f"{RUN_ID}-baseline-snapshot.json",
    f"{RUN_ID}-codex-events.jsonl",
    f"{RUN_ID}-codex-final-message.txt",
    f"{RUN_ID}-codex-namespace-closed.json",
    f"{RUN_ID}-codex-stderr.log",
    f"{RUN_ID}-conditional-launch-authorization.json",
    f"{RUN_ID}-foreground-outcome.json",
    f"{RUN_ID}-fresh-boundary-gpu-preflight.json",
    f"{RUN_ID}-fresh-restricted-codex-preflight.json",
    f"{RUN_ID}-human-launch-approval.json",
    f"{RUN_ID}-launch-preparation.json",
    f"{RUN_ID}-launcher-handoff.json",
    f"{RUN_ID}-namespace-boundary.json",
    f"{RUN_ID}-post-export-verification.json",
    f"{RUN_ID}-supervisor-producer.json",
    f"{RUN_ID}-terminal-status.json",
    f"{RUN_ID}-watchdog-producer.json",
    f"{RUN_ID}-watchdog-status.json",
    f"{RUN_ID}-final-handoff-seal.json",
}
OWNER_ACCEPTED_RUN_SCOPED_LIMITATION_IDS = [
    "CLAUDE_TOOL_CAPABILITY",
    "CLAUDE_CODEX_CREDENTIAL_ACCESS",
    "PRIMARY_EXPORT_COMPLETENESS",
    "SEAL_LOCK_TOCTOU",
    "TRUSTED_PREFLIGHT_OUTER_COVERAGE",
]
EXPORT_SCHEMAS = {EXPORT_HASH_SCHEMA, EXPORT_MANIFEST_SCHEMA}
POST_LOCK_SCHEMA = "bayesfilter.complete_highdim_leaderboard.post_lock_integrity.v1"
CLAUDE_AUDIT_SCHEMA = (
    "bayesfilter.complete_highdim_leaderboard.claude_tool_audit.v1"
)
SEMANTIC_INSPECTION_SCHEMA = (
    "bayesfilter.complete_highdim_leaderboard.post_run_semantic_inspection.v1"
)
POST_LOCK_RECEIPT = Path(f"/tmp/{RUN_ID}-post-lock-integrity.json")
SNAPSHOT_HANDOFF = (
    Path("/home/chakwong/BayesFilter")
    / f".complete-highdim-source-snapshot-{RUN_ID}"
    / "docs/plans/logs"
    / RUN_ID
)
SEMANTIC_INSPECTION_RECEIPT = Path(
    "/home/chakwong/BayesFilter/docs/plans/artifacts/complete-highdim-leaderboard/"
    "post-run-semantic-inspection-2026-07-12.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact is not an object: {path}")
    return value


def _safe_regular(path: Path, parent: Path) -> os.stat_result:
    if path.parent != parent or path.is_symlink():
        raise ValueError(f"artifact path is outside the exact handoff: {path}")
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise ValueError(f"artifact is not a regular single-link file: {path}")
    if info.st_mode & 0o222:
        raise ValueError(f"artifact remains writable at audit time: {path}")
    return info


def _canonical_recorded_file(recorded: Path, handoff: Path) -> tuple[Path, os.stat_result]:
    if (
        not recorded.is_absolute()
        or recorded.parent != SNAPSHOT_HANDOFF
        or not recorded.name.startswith(RUN_ID)
    ):
        raise ValueError(f"recorded artifact path has the wrong alias: {recorded}")
    canonical = handoff / recorded.name
    return canonical, _safe_regular(canonical, handoff)


def _recorded_path_matches(value: Any, canonical: Path) -> bool:
    recorded = Path(str(value))
    return recorded.parent == SNAPSHOT_HANDOFF and recorded.name == canonical.name


def _safe_archive_names(path: Path) -> list[str]:
    names: list[str] = []
    with tarfile.open(path, "r:gz") as archive:
        for member in archive.getmembers():
            pure = PurePosixPath(member.name)
            if (
                pure.is_absolute()
                or not pure.parts
                or ".." in pure.parts
                or member.name in names
                or not member.isfile()
            ):
                raise ValueError(f"unsafe or duplicate archive member: {member.name}")
            names.append(member.name)
    return sorted(names)


def _safe_change_paths(changes: dict[str, Any]) -> tuple[list[str], list[str], list[str]]:
    groups: list[list[str]] = []
    for key in ("added", "modified", "deleted"):
        values = changes.get(key)
        if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
            raise ValueError(f"primary change manifest {key} paths are invalid")
        if len(values) != len(set(values)):
            raise ValueError(f"primary change manifest {key} paths contain duplicates")
        for value in values:
            pure = PurePosixPath(value)
            if pure.is_absolute() or not pure.parts or ".." in pure.parts:
                raise ValueError(f"primary change manifest contains an unsafe path: {value}")
        groups.append(values)
    added, modified, deleted = groups
    if (set(added) & set(modified)) or (set(added) & set(deleted)) or (
        set(modified) & set(deleted)
    ):
        raise ValueError("primary change manifest path classes overlap")
    return added, modified, deleted


def _credential_values(codex_auth_source: Path) -> tuple[bytes, ...]:
    values: list[str] = []
    for name in ("ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_API_KEY"):
        value = os.environ.get(name)
        if value:
            values.append(value)
    if codex_auth_source.is_symlink() or not codex_auth_source.is_file():
        raise ValueError("Codex authentication source is missing or unsafe")
    auth = json.loads(codex_auth_source.read_text(encoding="utf-8"))

    def walk(value: Any, *, sensitive_parent: bool = False) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                normalized = str(key).upper()
                sensitive = sensitive_parent or any(
                    marker in normalized
                    for marker in ("KEY", "TOKEN", "SECRET", "PASSWORD", "AUTH")
                )
                walk(child, sensitive_parent=sensitive)
        elif isinstance(value, list):
            for child in value:
                walk(child, sensitive_parent=sensitive_parent)
        elif isinstance(value, str) and value and sensitive_parent:
            values.append(value)

    walk(auth)
    encoded = tuple(dict.fromkeys(value.encode("utf-8") for value in values))
    if not os.environ.get("ANTHROPIC_AUTH_TOKEN") and not os.environ.get(
        "ANTHROPIC_API_KEY"
    ):
        raise ValueError("current Anthropic credential value is unavailable for scan")
    if not encoded:
        raise ValueError("no current credential values are available for scan")
    return encoded


def _stream_contains_secret(stream: Any, secrets: Sequence[bytes]) -> bool:
    overlap = max(len(secret) for secret in secrets) - 1
    tail = b""
    while True:
        chunk = stream.read(1024 * 1024)
        if not chunk:
            return False
        value = tail + chunk
        if any(secret in value for secret in secrets):
            return True
        tail = value[-overlap:] if overlap > 0 else b""


def _scan_credentials(
    *, handoff: Path, archive_path: Path, archive_names: Sequence[str], secrets: Sequence[bytes]
) -> dict[str, Any]:
    scanned_handoff = 0
    for path in sorted(handoff.iterdir()):
        with path.open("rb") as stream:
            if _stream_contains_secret(stream, secrets):
                raise ValueError(f"credential value leaked into handoff file: {path.name}")
        scanned_handoff += 1
    scanned_members = 0
    with tarfile.open(archive_path, "r:gz") as archive:
        members = {member.name: member for member in archive.getmembers()}
        if sorted(members) != sorted(archive_names):
            raise ValueError("archive changed during credential scan")
        for name in sorted(archive_names):
            stream = archive.extractfile(members[name])
            if stream is None:
                raise ValueError(f"safe archive member cannot be read: {name}")
            with stream:
                if _stream_contains_secret(stream, secrets):
                    raise ValueError(
                        f"credential value leaked into archive member: {name}"
                    )
            scanned_members += 1
    return {
        "current_credential_value_count_scanned_in_memory": len(secrets),
        "handoff_files_scanned": scanned_handoff,
        "safe_archive_members_scanned": scanned_members,
        "credential_value_matches": 0,
        "credential_values_or_hashes_persisted_by_auditor": False,
    }


def _archive_member_payloads(
    archive_path: Path, archive_names: Sequence[str]
) -> dict[str, bytes]:
    selected = {
        name
        for name in archive_names
        if name.startswith(".complete_highdim_claude_audit/")
    }
    payloads: dict[str, bytes] = {}
    with tarfile.open(archive_path, "r:gz") as archive:
        members = {member.name: member for member in archive.getmembers()}
        for name in sorted(selected):
            stream = archive.extractfile(members[name])
            if stream is None:
                raise ValueError(f"Claude audit archive member cannot be read: {name}")
            with stream:
                payloads[name] = stream.read()
    return payloads


def _validate_claude_audit_records(payloads: dict[str, bytes]) -> dict[str, Any]:
    metadata_names = {name for name in payloads if name.endswith("-metadata.json")}
    raw_names = {name for name in payloads if name.endswith("-stream.jsonl")}
    stderr_names = {name for name in payloads if name.endswith("-stderr.log")}
    if not metadata_names:
        raise ValueError("no audited Claude invocation evidence was exported")
    other = set(payloads) - metadata_names - raw_names - stderr_names
    if other:
        raise ValueError(f"unexpected Claude audit artifact: {sorted(other)}")
    expected_raw = {name.removesuffix("-metadata.json") + "-stream.jsonl" for name in metadata_names}
    expected_stderr = {name.removesuffix("-metadata.json") + "-stderr.log" for name in metadata_names}
    if raw_names != expected_raw or stderr_names != expected_stderr:
        raise ValueError("Claude raw stream, stderr, and metadata sets are incomplete")
    records = []
    for name in sorted(metadata_names):
        try:
            payload = json.loads(payloads[name].decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"Claude audit metadata is invalid: {name}") from error
        prefix = name.removesuffix("-metadata.json")
        raw_name = prefix + "-stream.jsonl"
        stderr_name = prefix + "-stderr.log"
        raw_events: list[dict[str, Any]] = []
        raw_invalid_lines = 0
        for line in payloads[raw_name].decode("utf-8", errors="replace").splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                raw_invalid_lines += 1
                continue
            if not isinstance(event, dict):
                raw_invalid_lines += 1
                continue
            raw_events.append(event)
        raw_tool_uses: list[dict[str, str]] = []
        seen_tools: set[tuple[str, str]] = set()

        def walk(value: Any) -> None:
            if isinstance(value, dict):
                if value.get("type") == "tool_use" and isinstance(
                    value.get("name"), str
                ):
                    key = (str(value.get("id", "")), value["name"])
                    if key not in seen_tools:
                        seen_tools.add(key)
                        raw_tool_uses.append({"id": key[0], "name": key[1]})
                for child in value.values():
                    walk(child)
            elif isinstance(value, list):
                for child in value:
                    walk(child)

        for event in raw_events:
            walk(event)
        raw_disallowed = [
            record
            for record in raw_tool_uses
            if record["name"] not in {"Read", "Glob", "Grep", "LS"}
        ]
        if not (
            isinstance(payload, dict)
            and payload.get("schema_version") == CLAUDE_AUDIT_SCHEMA
            and payload.get("permission_mode") == "plan"
            and payload.get("metadata_generated_after_worker_exit_or_signal") is True
            and payload.get("stream_parse_complete") is True
            and payload.get("invalid_stream_line_count") == 0
            and isinstance(payload.get("observed_tool_uses"), list)
            and payload.get("parsed_event_count") == len(raw_events)
            and raw_invalid_lines == 0
            and payload.get("observed_tool_uses") == raw_tool_uses
            and payload.get("disallowed_tool_uses") == []
            and raw_disallowed == []
            and payload.get(
                "read_only_instruction_contract_satisfied_by_observed_tools"
            )
            is True
            and payload.get("technical_tool_capability_absent") is False
            and payload.get("prompt_or_credential_value_recorded_by_wrapper") is False
            and payload.get("raw_stream_path") == raw_name
            and payload.get("stderr_path") == stderr_name
        ):
            raise ValueError(f"Claude read-only tool audit failed: {name}")
        records.append(
            {
                "metadata_path": name,
                "metadata_sha256": hashlib.sha256(payloads[name]).hexdigest(),
                "raw_stream_path": raw_name,
                "raw_stream_sha256": hashlib.sha256(payloads[raw_name]).hexdigest(),
                "stderr_path": stderr_name,
                "stderr_sha256": hashlib.sha256(payloads[stderr_name]).hexdigest(),
                "observed_tool_uses": payload["observed_tool_uses"],
            }
        )
    return {
        "audited_invocation_count": len(records),
        "records": records,
        "all_streams_parse_complete": True,
        "observed_non_read_only_tool_use": False,
        "technical_tool_capability_absent": False,
    }


def _recorded_process_is_live(record: dict[str, Any]) -> bool:
    pid = record.get("pid")
    if not isinstance(pid, int) or pid <= 0:
        raise ValueError("producer descriptor PID is invalid")
    path = Path(f"/proc/{pid}/stat")
    try:
        value = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return False
    closing = value.rfind(")")
    fields = value[closing + 2 :].split() if closing >= 0 else []
    if len(fields) < 20 or fields[0] == "Z":
        return False
    return bool(
        int(fields[2]) == record.get("pgid")
        and int(fields[3]) == record.get("sid")
        and int(fields[19]) == record.get("start_time_ticks")
    )


def _decode_mount_path(value: str) -> str:
    for escaped, literal in (
        ("\\040", " "),
        ("\\011", "\t"),
        ("\\012", "\n"),
        ("\\134", "\\"),
    ):
        value = value.replace(escaped, literal)
    return value


def _mount_aliases_are_closed(handoff: Path) -> dict[str, Any]:
    aliases = {
        str(handoff),
        str(SNAPSHOT_HANDOFF),
        f"/tmp/{RUN_ID}-live-handoff",
    }
    observed: list[dict[str, Any]] = []
    with Path("/proc/self/mountinfo").open(encoding="utf-8") as stream:
        for line in stream:
            fields = line.split()
            if len(fields) < 10 or "-" not in fields:
                raise ValueError("malformed /proc/self/mountinfo record")
            target = _decode_mount_path(fields[4])
            if target not in aliases:
                continue
            options = set(fields[5].split(","))
            observed.append(
                {
                    "target": target,
                    "read_only": "ro" in options,
                    "mount_id": int(fields[0]),
                }
            )
            if "rw" in options or "ro" not in options:
                raise ValueError(f"handoff alias is mounted writable: {target}")
    return {
        "checked_aliases": sorted(aliases),
        "observed_mounts": observed,
        "all_observed_mounts_read_only": True,
    }


def _validate_post_lock_receipt(
    *, handoff: Path, seal_path: Path, receipt_path: Path
) -> dict[str, Any]:
    if receipt_path != POST_LOCK_RECEIPT or receipt_path.is_symlink():
        raise ValueError("post-lock receipt path is not the exact external path")
    info = receipt_path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_mode & 0o222:
        raise ValueError("post-lock receipt is writable or unsafe")
    payload = _object(receipt_path)
    aliases = payload.get("aliases")
    expected_alias_paths = {
        str(handoff),
        str(
            SNAPSHOT_HANDOFF
        ),
        f"/tmp/{RUN_ID}-live-handoff",
    }
    if not (
        payload.get("schema_version") == POST_LOCK_SCHEMA
        and payload.get("run_id") == RUN_ID
        and payload.get("recorded_after_all_aliases_locked_read_only") is True
        and payload.get("aliases_identify_same_directory") is True
        and isinstance(aliases, list)
        and len(aliases) == 3
        and {record.get("path") for record in aliases if isinstance(record, dict)}
        == expected_alias_paths
        and len(
            {
                (record.get("device"), record.get("inode"))
                for record in aliases
                if isinstance(record, dict)
            }
        )
        == 1
        and all(
            isinstance(record, dict)
            and record.get("effective_mount", {}).get("mount_options")
            and "ro" in record["effective_mount"]["mount_options"]
            and "rw" not in record["effective_mount"]["mount_options"]
            for record in aliases
        )
        and payload.get("seal_path_at_lock")
        == str(SNAPSHOT_HANDOFF / seal_path.name)
        and payload.get("seal_sha256_after_lock") == _sha256(seal_path)
        and payload.get("seal_size_after_lock") == seal_path.stat().st_size
        and payload.get("sealed_files_rehashed_after_lock") is True
        and payload.get(
            "post_lock_match_does_not_prove_absence_of_transient_pre_lock_write"
        )
        is True
    ):
        raise ValueError("post-lock alias or seal receipt is invalid")
    seal = _object(seal_path)
    expected_records = sorted(
        (
            Path(str(record["path"])).name,
            record["size"],
            record["sha256"],
        )
        for record in seal["files"]
    )
    observed_records = sorted(
        (record.get("name"), record.get("size"), record.get("sha256"))
        for record in payload.get("sealed_files", [])
        if isinstance(record, dict)
    )
    if observed_records != expected_records:
        raise ValueError("post-lock sealed-file receipt differs from the seal")
    return {
        "receipt_path": str(receipt_path),
        "receipt_sha256": _sha256(receipt_path),
        "all_three_aliases_same_identity_at_lock": True,
        "all_three_aliases_read_only_at_lock": True,
        "seal_and_sealed_files_rehashed_after_lock": True,
        "absence_of_transient_pre_lock_write_proved": False,
    }


def _validate_semantic_inspection_receipt(
    *,
    receipt_path: Path,
    change_path: Path,
    diff_path: Path,
    status_path: Path,
    claude_audit: dict[str, Any],
) -> dict[str, Any]:
    if receipt_path != SEMANTIC_INSPECTION_RECEIPT or receipt_path.is_symlink():
        raise ValueError("semantic inspection receipt path is not exact")
    info = receipt_path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_mode & 0o222:
        raise ValueError("semantic inspection receipt is writable or unsafe")
    payload = _object(receipt_path)
    expected_claude_evidence: dict[str, str] = {}
    for record in claude_audit["records"]:
        expected_claude_evidence[record["metadata_path"]] = record[
            "metadata_sha256"
        ]
        expected_claude_evidence[record["raw_stream_path"]] = record[
            "raw_stream_sha256"
        ]
        expected_claude_evidence[record["stderr_path"]] = record[
            "stderr_sha256"
        ]
    if not (
        payload.get("schema_version") == SEMANTIC_INSPECTION_SCHEMA
        and payload.get("run_id") == RUN_ID
        and payload.get("verdict") == "PASS_SEMANTIC_POST_RUN_INSPECTION"
        and payload.get("inspector_role")
        == "independent_read_only_post_run_inspector"
        and payload.get("change_manifest_sha256") == _sha256(change_path)
        and payload.get("tracked_diff_sha256") == _sha256(diff_path)
        and payload.get("git_status_sha256") == _sha256(status_path)
        and payload.get("diff_status_semantically_consistent_with_change_manifest")
        is True
        and payload.get("diff_contains_only_expected_tracked_changes") is True
        and payload.get("status_accounts_for_added_modified_deleted_paths") is True
        and payload.get("claude_event_and_tool_audit_evidence_sha256")
        == expected_claude_evidence
        and payload.get("claude_event_evidence_semantically_inspected") is True
        and payload.get("full_tracked_diff_read") is True
        and payload.get("full_git_status_read") is True
        and payload.get("all_claude_raw_events_and_stderr_read") is True
        and payload.get("observed_non_read_only_claude_tool_use") is False
        and payload.get("credential_values_or_hashes_recorded") is False
        and payload.get("completion_or_release_authority_granted") is False
    ):
        raise ValueError("semantic post-run inspection receipt is invalid")
    return {
        "receipt_path": str(receipt_path),
        "receipt_sha256": _sha256(receipt_path),
        "verdict": "PASS_SEMANTIC_POST_RUN_INSPECTION",
    }


def _recorded_pid_file_is_live(path: Path) -> bool:
    value = path.read_text(encoding="utf-8").strip()
    if not value.isdigit():
        raise ValueError(f"recorded PID file is invalid: {path.name}")
    try:
        stat_text = Path(f"/proc/{int(value)}/stat").read_text(encoding="utf-8")
    except FileNotFoundError:
        return False
    closing = stat_text.rfind(")")
    fields = stat_text[closing + 2 :].split() if closing >= 0 else []
    return bool(fields and fields[0] != "Z")


def _validate_completion_chain(
    *,
    handoff: Path,
    manifest: dict[str, Any],
    manifest_sha256: str,
) -> dict[str, bool]:
    run_id = RUN_ID
    hashes_path = handoff / f"{run_id}-primary-export-sha256.json"
    namespace_path = handoff / f"{run_id}-codex-namespace-closed.json"
    namespace = _object(namespace_path)
    if not (
        namespace.get("schema_version")
        == "bayesfilter.complete_highdim_leaderboard.codex_namespace_closed.v1"
        and namespace.get("run_id") == run_id
        and namespace.get("unshare_process_returned") is True
        and namespace.get("unshare_process_group_absent") is True
        and namespace.get("private_pid_namespace_init_required_pid_one") is True
        and namespace.get("whole_private_pid_namespace_quiescent") is True
        and namespace.get("untrusted_code_started_only_after_capability_drop")
        is True
        and namespace.get("namespace_escape_available_to_untrusted_code") is False
        and namespace.get("pytest_direct_codex_bypass") is False
    ):
        raise ValueError("Codex namespace-close receipt is invalid")

    verification = _object(
        handoff / f"{run_id}-post-export-verification.json"
    )
    if not (
        verification.get("schema_version")
        == "bayesfilter.complete_highdim_leaderboard.post_export_verification.v1"
        and verification.get("run_id") == run_id
        and verification.get("export_label") == "primary"
        and verification.get("all_bound_files_recomputed") is True
        and verification.get("verified_after_codex_namespace_closed") is True
        and _recorded_path_matches(verification.get("export_hashes"), hashes_path)
        and verification.get("export_hashes_sha256") == _sha256(hashes_path)
        and _recorded_path_matches(
            verification.get("codex_namespace_closed_receipt"), namespace_path
        )
        and verification.get("codex_namespace_closed_receipt_sha256")
        == _sha256(namespace_path)
    ):
        raise ValueError("post-export verification chain is invalid")

    watchdog = _object(handoff / f"{run_id}-watchdog-status.json")
    if not (
        watchdog.get("schema_version")
        == "bayesfilter.complete_highdim_leaderboard.watchdog.v3"
        and watchdog.get("run_id") == run_id
        and watchdog.get("primary_export_verified") is True
        and watchdog.get("primary_export_only") is True
        and watchdog.get("primary_export_only_observed") is True
        and watchdog.get("unapproved_export_artifacts") == []
        and watchdog.get("fallback_export_allowed") is False
        and watchdog.get("writable_copy_read_by_watchdog") is False
        and watchdog.get("writable_copy_written_by_watchdog") is False
        and watchdog.get("verification_exit_code") == 0
    ):
        raise ValueError("watchdog completion chain is invalid")

    outcome = _object(handoff / f"{run_id}-foreground-outcome.json")
    if not (
        outcome.get("schema_version")
        == "bayesfilter.complete_highdim_leaderboard.foreground_outcome.v2"
        and outcome.get("run_id") == run_id
        and outcome.get("terminal_present") is True
        and outcome.get("primary_export_present") is True
        and outcome.get("post_export_verification_present") is True
        and outcome.get("watchdog_primary_verification_passed") is True
        and outcome.get("producer_descriptors_valid") is True
        and outcome.get("producers_closed_before_outcome") is True
        and outcome.get("whole_outer_pid_namespace_quiescent_before_outcome")
        is True
        and outcome.get("fallback_export_allowed") is False
        and outcome.get("fallback_export_present") is False
        and outcome.get("primary_export_only_observed") is True
        and outcome.get("unapproved_export_artifacts") == []
    ):
        raise ValueError("foreground completion chain is invalid")

    approval = _object(handoff / f"{run_id}-human-launch-approval.json")
    concrete = manifest["concrete_run"]
    if not (
        approval.get("schema_version")
        == "bayesfilter.complete_highdim_leaderboard.human_launch_approval.v1"
        and approval.get("approval_instance_id")
        == concrete.get("approval_instance_id")
        and approval.get("run_id") == run_id
        and approval.get("manifest_sha256") == manifest_sha256
        and approval.get("nonce") == concrete.get("copy_sentinel_nonce")
        and approval.get("owner_accepted_run_scoped_limitations")
        == manifest.get("owner_accepted_run_scoped_limitations")
        and approval.get("mandatory_post_run_integrity_audit")
        == manifest.get("mandatory_post_run_integrity_audit")
        and approval.get("final_exact_launch_command_separately_approved_by_human")
        is True
        and approval.get("completion_or_release_authority_granted") is False
    ):
        raise ValueError("human launch approval record is inconsistent")

    conditional = _object(
        handoff / f"{run_id}-conditional-launch-authorization.json"
    )
    if not (
        conditional.get("schema_version")
        == (
            "bayesfilter.complete_highdim_leaderboard."
            "conditional_launch_authorization.v1"
        )
        and conditional.get("run_id") == run_id
        and conditional.get("nonce") == concrete.get("copy_sentinel_nonce")
        and conditional.get("approval_instance_id")
        == concrete.get("approval_instance_id")
        and conditional.get("fresh_fake_gpu_boundary_probe_passed") is True
        and conditional.get("fresh_real_restricted_codex_probe_passed") is True
        and conditional.get("copy_and_supervisor_launch_condition_satisfied")
        is True
        and conditional.get("scientific_or_release_authority_granted") is False
        and conditional.get(
            "post_run_integrity_audit_required_before_completion_or_release"
        )
        is True
        and conditional.get("owner_accepted_run_scoped_limitations")
        == OWNER_ACCEPTED_RUN_SCOPED_LIMITATION_IDS
    ):
        raise ValueError("conditional launch authorization is inconsistent")

    return {
        "namespace_close_valid": True,
        "post_export_verification_valid": True,
        "watchdog_status_valid": True,
        "foreground_outcome_valid": True,
        "human_approval_record_valid": True,
        "conditional_authorization_valid": True,
    }


def audit(
    *,
    handoff: Path,
    manifest_path: Path,
    receipt_path: Path,
    post_lock_receipt_path: Path,
    semantic_inspection_receipt_path: Path,
    codex_auth_source: Path,
) -> dict[str, Any]:
    if handoff != handoff.resolve(strict=True) or handoff.is_symlink():
        raise ValueError("handoff must be the canonical real directory")
    expected = Path("/home/chakwong/BayesFilter/docs/plans/logs") / RUN_ID
    if handoff != expected:
        raise ValueError("handoff is not the exact waived run path")
    handoff_info = handoff.stat()
    if not stat.S_ISDIR(handoff_info.st_mode) or handoff_info.st_mode & 0o222:
        raise ValueError("handoff directory remains writable at audit time")
    observed_names: set[str] = set()
    observed_hashes: dict[str, str] = {}
    for path in handoff.iterdir():
        _safe_regular(path, handoff)
        if not path.name.startswith(RUN_ID):
            raise ValueError(f"handoff contains an unscoped file: {path.name}")
        observed_names.add(path.name)
        observed_hashes[path.name] = _sha256(path)
    missing = sorted((PRIMARY_FILES | REQUIRED_CONTROL_FILES) - observed_names)
    if missing:
        raise ValueError(f"required post-run artifacts are missing: {missing}")

    manifest = _object(manifest_path)
    if (
        manifest.get("schema_version") != MANIFEST_SCHEMA
        or manifest.get("concrete_run", {}).get("run_id") != RUN_ID
    ):
        raise ValueError("schema-v7 manifest does not bind the waived run")
    limitations = manifest.get("owner_accepted_run_scoped_limitations")
    post_run_audit = manifest.get("mandatory_post_run_integrity_audit")
    if not (
        isinstance(limitations, dict)
        and limitations.get("run_id") == RUN_ID
        and limitations.get("repository_default") is False
        and limitations.get("reusable_for_other_runs") is False
        and limitations.get("technical_guarantees_repaired") is False
        and limitations.get("accepted_limitation_ids")
        == OWNER_ACCEPTED_RUN_SCOPED_LIMITATION_IDS
        and isinstance(post_run_audit, dict)
        and post_run_audit.get("required_before_completion_or_release_claim")
        is True
        and post_run_audit.get("structural_stage_pass_is_not_full_audit_pass")
        is True
        and post_run_audit.get("semantic_inspection_receipt_required") is True
        and post_run_audit.get("post_lock_receipt", {}).get(
            "required_for_structural_pass"
        )
        is True
        and post_run_audit.get("phase8_phase9_validator_exit_zero_and_all_checks_required")
        is True
    ):
        raise ValueError("manifest waiver or post-run audit contract is invalid")
    receipt = _object(receipt_path)
    if (
        receipt.get("verdict") != "AGREE"
        or receipt.get("iteration") != 6
        or receipt.get("run_id") != RUN_ID
        or receipt.get("reviewed_manifest_sha256") != _sha256(manifest_path)
        or receipt.get("accepted_limitation_ids")
        != OWNER_ACCEPTED_RUN_SCOPED_LIMITATION_IDS
        or receipt.get("risk_acceptance_amendment_sha256")
        != limitations.get("authorization_amendment", {}).get("sha256")
        or receipt.get("post_run_integrity_audit_plan_sha256")
        != post_run_audit.get("plan", {}).get("sha256")
        or receipt.get("post_run_integrity_auditor_sha256")
        != post_run_audit.get("auditor", {}).get("sha256")
        or receipt.get("final_exact_launch_command_authorized") is not False
        or receipt.get("completion_or_release_authority_granted") is not False
    ):
        raise ValueError("iteration-6 launch-readiness receipt is invalid")

    unapproved_exports: list[str] = []
    for name in observed_names:
        path = handoff / name
        if "-fallback-" in name:
            unapproved_exports.append(name)
            continue
        if path.suffix != ".json":
            continue
        try:
            payload = _object(path)
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
            continue
        if (
            payload.get("schema_version") in EXPORT_SCHEMAS
            and name not in PRIMARY_FILES
        ):
            unapproved_exports.append(name)
    if unapproved_exports:
        raise ValueError(
            f"handoff contains unapproved export artifacts: {sorted(unapproved_exports)}"
        )

    hashes_path = handoff / f"{RUN_ID}-primary-export-sha256.json"
    hashes = _object(hashes_path)
    if not (
        hashes.get("schema_version") == EXPORT_HASH_SCHEMA
        and hashes.get("run_id") == RUN_ID
        and hashes.get("export_label") == "primary"
        and isinstance(hashes.get("files"), list)
    ):
        raise ValueError("primary export hash ledger is invalid")
    ledger_names: set[str] = set()
    for record in hashes["files"]:
        if not isinstance(record, dict):
            raise ValueError("primary export hash record is invalid")
        path = Path(str(record.get("path", "")))
        path, info = _canonical_recorded_file(path, handoff)
        if path.name in ledger_names:
            raise ValueError(f"duplicate primary export hash record: {path.name}")
        if info.st_size != record.get("size") or _sha256(path) != record.get("sha256"):
            raise ValueError(f"primary export hash mismatch: {path.name}")
        ledger_names.add(path.name)
    required_ledger_members = PRIMARY_FILES - {hashes_path.name}
    if not required_ledger_members.issubset(ledger_names):
        missing_ledger = sorted(required_ledger_members - ledger_names)
        raise ValueError(f"primary hash ledger is incomplete: {missing_ledger}")

    change_path = handoff / f"{RUN_ID}-primary-isolated-change-manifest.json"
    changes = _object(change_path)
    if not (
        changes.get("schema_version") == EXPORT_MANIFEST_SCHEMA
        and changes.get("run_id") == RUN_ID
        and changes.get("export_label") == "primary"
        and isinstance(changes.get("added"), list)
        and isinstance(changes.get("modified"), list)
        and isinstance(changes.get("deleted"), list)
    ):
        raise ValueError("primary change manifest is invalid")
    added, modified, deleted = _safe_change_paths(changes)
    expected_archive = sorted([*added, *modified])
    archive_path = handoff / f"{RUN_ID}-primary-isolated-changed-files.tar.gz"
    archive_names = _safe_archive_names(archive_path)
    if archive_names != expected_archive:
        raise ValueError("archive membership disagrees with the change manifest")
    if set(deleted) & set(archive_names):
        raise ValueError("deleted paths appear in the primary archive")

    seal_path = handoff / f"{RUN_ID}-final-handoff-seal.json"
    seal = _object(seal_path)
    if not (
        seal.get("schema_version") == SEAL_SCHEMA
        and seal.get("run_id") == RUN_ID
        and seal.get("approval_instance_id")
        == manifest.get("concrete_run", {}).get("approval_instance_id")
        and seal.get("seal_file_excluded_from_its_own_hash_ledger") is True
        and isinstance(seal.get("files"), list)
    ):
        raise ValueError("final handoff seal is invalid")
    sealed_names: set[str] = set()
    for record in seal["files"]:
        if not isinstance(record, dict):
            raise ValueError("final seal file record is invalid")
        path = Path(str(record.get("path", "")))
        path, info = _canonical_recorded_file(path, handoff)
        if path.name == seal_path.name or path.name in sealed_names:
            raise ValueError("final seal contains itself or a duplicate record")
        if info.st_size != record.get("size") or _sha256(path) != record.get("sha256"):
            raise ValueError(f"final seal rehash failed: {path.name}")
        sealed_names.add(path.name)
    if sealed_names != observed_names - {seal_path.name}:
        raise ValueError("final seal inventory differs from current handoff inventory")

    post_lock_state = _validate_post_lock_receipt(
        handoff=handoff,
        seal_path=seal_path,
        receipt_path=post_lock_receipt_path,
    )
    secrets = _credential_values(codex_auth_source)
    credential_scan = _scan_credentials(
        handoff=handoff,
        archive_path=archive_path,
        archive_names=archive_names,
        secrets=secrets,
    )
    claude_audit = _validate_claude_audit_records(
        _archive_member_payloads(archive_path, archive_names)
    )
    semantic_inspection = _validate_semantic_inspection_receipt(
        receipt_path=semantic_inspection_receipt_path,
        change_path=change_path,
        diff_path=handoff / f"{RUN_ID}-primary-isolated-tracked.diff",
        status_path=handoff / f"{RUN_ID}-primary-isolated-git-status.txt",
        claude_audit=claude_audit,
    )

    producer_names = (
        f"{RUN_ID}-supervisor-producer.json",
        f"{RUN_ID}-watchdog-producer.json",
    )
    for name in producer_names:
        if _recorded_process_is_live(_object(handoff / name)):
            raise ValueError(f"recorded producer is still live: {name}")
    for name in (
        f"{RUN_ID}.pid",
        f"{RUN_ID}.supervisor.pid",
        f"{RUN_ID}.wrapper.pid",
        f"{RUN_ID}-codex-process-group.pid",
        f"{RUN_ID}-watchdog.pid",
    ):
        path = handoff / name
        if path.is_file() and _recorded_pid_file_is_live(path):
            raise ValueError(f"recorded run process is still live: {name}")
    mount_state = _mount_aliases_are_closed(handoff)
    completion_chain = _validate_completion_chain(
        handoff=handoff,
        manifest=manifest,
        manifest_sha256=_sha256(manifest_path),
    )

    return {
        "schema_version": (
            "bayesfilter.complete_highdim_leaderboard.post_run_integrity_audit.v1"
        ),
        "run_id": RUN_ID,
        "verdict": "PASS_STRUCTURAL_POST_RUN_INTEGRITY",
        "manifest_sha256": _sha256(manifest_path),
        "launch_readiness_receipt_sha256": _sha256(receipt_path),
        "seal_sha256": _sha256(seal_path),
        "primary_export_hash_ledger_sha256": _sha256(hashes_path),
        "observed_handoff_file_count": len(observed_names),
        "observed_handoff_files": sorted(observed_names),
        "primary_export_files_complete": True,
        "archive_matches_change_manifest": True,
        "sealed_files_rehashed": True,
        "post_lock_integrity": post_lock_state,
        "credential_leak_scan": credential_scan,
        "claude_tool_audit": claude_audit,
        "semantic_inspection": semantic_inspection,
        "recorded_producers_closed": True,
        "mount_alias_state": mount_state,
        "completion_chain": completion_chain,
        "numeric_completion_claim_checked": False,
        "numeric_completion_check_required_before_release": True,
        "accepted_seal_race_absence_proved": False,
        "release_authority_granted": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--launch-readiness-receipt", type=Path, required=True)
    parser.add_argument(
        "--post-lock-receipt", type=Path, default=POST_LOCK_RECEIPT
    )
    parser.add_argument(
        "--semantic-inspection-receipt",
        type=Path,
        default=SEMANTIC_INSPECTION_RECEIPT,
    )
    parser.add_argument(
        "--codex-auth-source",
        type=Path,
        default=Path("/home/chakwong/.codex/auth.json"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = audit(
        handoff=args.handoff,
        manifest_path=args.manifest.resolve(strict=True),
        receipt_path=args.launch_readiness_receipt.resolve(strict=True),
        post_lock_receipt_path=args.post_lock_receipt,
        semantic_inspection_receipt_path=args.semantic_inspection_receipt,
        codex_auth_source=args.codex_auth_source,
    )
    serialized = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(serialized, end="")
    else:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
