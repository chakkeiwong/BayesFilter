#!/usr/bin/env python3
"""Independently verify the primary export without reading the writable copy."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import time
from pathlib import Path
from typing import Any, Sequence


DESCRIPTOR_SCHEMA = "bayesfilter.complete_highdim_leaderboard.producer_descriptor.v1"
STATUS_SCHEMA = "bayesfilter.complete_highdim_leaderboard.watchdog.v3"
EXPORT_SCHEMAS = {
    "bayesfilter.complete_highdim_leaderboard.isolated_export.v1",
    "bayesfilter.complete_highdim_leaderboard.export_hashes.v1",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_exclusive(path: Path, payload: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _proc_start_time_ticks(pid: int) -> int:
    fields = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()
    if len(fields) < 22:
        raise RuntimeError(f"process identity is incomplete for PID {pid}")
    return int(fields[21])


def _export_hashes_are_valid(path: Path, run_id: str, handoff_dir: Path) -> bool:
    if path.is_symlink() or not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    if not (
        payload.get("schema_version")
        == "bayesfilter.complete_highdim_leaderboard.export_hashes.v1"
        and payload.get("run_id") == run_id
        and payload.get("export_label") == "primary"
        and isinstance(payload.get("files"), list)
    ):
        return False
    expected_parent = handoff_dir.resolve()
    for record in payload["files"]:
        if not isinstance(record, dict):
            return False
        try:
            bound = Path(record["path"])
            resolved = bound.resolve(strict=True)
            info = bound.lstat()
        except (KeyError, OSError, RuntimeError):
            return False
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
            or resolved.parent != expected_parent
            or info.st_size != record.get("size")
            or _sha256(bound) != record.get("sha256")
        ):
            return False
    return True


def _verification_is_valid(
    path: Path, *, run_id: str, handoff_dir: Path, primary_hashes: Path
) -> bool:
    if path.is_symlink() or not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    namespace_closed = Path(str(payload.get("codex_namespace_closed_receipt", "")))
    try:
        namespace_resolved = namespace_closed.resolve(strict=True)
    except (OSError, RuntimeError):
        return False
    if not (
        payload.get("schema_version")
        == "bayesfilter.complete_highdim_leaderboard.post_export_verification.v1"
        and payload.get("run_id") == run_id
        and payload.get("export_label") == "primary"
        and payload.get("all_bound_files_recomputed") is True
        and payload.get("verified_after_codex_namespace_closed") is True
        and Path(str(payload.get("export_hashes", ""))) == primary_hashes
        and payload.get("export_hashes_sha256") == _sha256(primary_hashes)
        and namespace_resolved.parent == handoff_dir.resolve()
        and namespace_closed.name == f"{run_id}-codex-namespace-closed.json"
        and payload.get("codex_namespace_closed_receipt_sha256")
        == _sha256(namespace_closed)
    ):
        return False
    try:
        closed = json.loads(namespace_closed.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    return bool(
        closed.get("schema_version")
        == "bayesfilter.complete_highdim_leaderboard.codex_namespace_closed.v1"
        and closed.get("run_id") == run_id
        and closed.get("unshare_process_returned") is True
        and closed.get("unshare_process_group_absent") is True
        and closed.get("private_pid_namespace_init_required_pid_one") is True
        and closed.get("whole_private_pid_namespace_quiescent") is True
        and closed.get("quiescence_proof")
        == "linux_pid_namespace_init_exit_kills_all_members_and_unshare_returned"
        and closed.get("untrusted_code_started_only_after_capability_drop") is True
        and closed.get("namespace_escape_available_to_untrusted_code") is False
    )


def _unapproved_export_artifacts(handoff_dir: Path, run_id: str) -> list[str]:
    allowed = {
        f"{run_id}-primary-isolated-change-manifest.json",
        f"{run_id}-primary-isolated-changed-files.tar.gz",
        f"{run_id}-primary-isolated-tracked.diff",
        f"{run_id}-primary-isolated-git-status.txt",
        f"{run_id}-primary-export-sha256.json",
    }
    export_suffixes = (
        "-isolated-change-manifest.json",
        "-isolated-changed-files.tar.gz",
        "-isolated-tracked.diff",
        "-isolated-git-status.txt",
        "-export-sha256.json",
    )
    rejected: set[str] = set()
    for path in handoff_dir.iterdir():
        if not path.name.startswith(f"{run_id}-") or path.name in allowed:
            continue
        if "-fallback-" in path.name or path.name.endswith(export_suffixes):
            rejected.add(path.name)
            continue
        if path.suffix != ".json" or path.is_symlink() or not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(payload, dict) and payload.get("schema_version") in EXPORT_SCHEMAS:
            rejected.add(path.name)
    return sorted(rejected)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--handoff-dir", type=Path, required=True)
    parser.add_argument("--started-epoch", type=float, required=True)
    parser.add_argument("--started-monotonic", type=float, required=True)
    parser.add_argument("--verification-deadline", type=int, default=27360)
    args = parser.parse_args(argv)

    if args.handoff_dir.is_symlink() or not args.handoff_dir.is_dir():
        raise RuntimeError("handoff directory is missing or unsafe")
    descriptor = args.handoff_dir / f"{args.run_id}-watchdog-producer.json"
    pid = os.getpid()
    _write_exclusive(
        descriptor,
        {
            "schema_version": DESCRIPTOR_SCHEMA,
            "run_id": args.run_id,
            "role": "primary_export_verification_watchdog",
            "pid": pid,
            "pgid": os.getpgid(0),
            "sid": os.getsid(0),
            "start_time_ticks": _proc_start_time_ticks(pid),
            "pid_namespace_inode": os.stat("/proc/self/ns/pid").st_ino,
            "command_path": str(Path(__file__).resolve()),
            "command_sha256": _sha256(Path(__file__).resolve()),
            "created_at_epoch": time.time(),
        },
    )

    deadline = args.started_monotonic + args.verification_deadline
    primary_hashes = args.handoff_dir / f"{args.run_id}-primary-export-sha256.json"
    verification = args.handoff_dir / f"{args.run_id}-post-export-verification.json"
    valid = False
    while time.monotonic() < deadline:
        if _export_hashes_are_valid(primary_hashes, args.run_id, args.handoff_dir):
            valid = _verification_is_valid(
                verification,
                run_id=args.run_id,
                handoff_dir=args.handoff_dir,
                primary_hashes=primary_hashes,
            )
            if valid and not _unapproved_export_artifacts(
                args.handoff_dir, args.run_id
            ):
                break
            valid = False
        time.sleep(max(0.05, min(2.0, deadline - time.monotonic())))

    status = args.handoff_dir / f"{args.run_id}-watchdog-status.json"
    unapproved_exports = _unapproved_export_artifacts(args.handoff_dir, args.run_id)
    primary_export_only_observed = not unapproved_exports
    valid = valid and primary_export_only_observed
    _write_exclusive(
        status,
        {
            "schema_version": STATUS_SCHEMA,
            "run_id": args.run_id,
            "timestamp_epoch": time.time(),
            "primary_export_verified": valid,
            "primary_export_only": True,
            "primary_export_only_observed": primary_export_only_observed,
            "unapproved_export_artifacts": unapproved_exports,
            "fallback_export_allowed": False,
            "writable_copy_read_by_watchdog": False,
            "writable_copy_written_by_watchdog": False,
            "verification_exit_code": 0 if valid else 95,
            "reason": (
                "primary_export_and_namespace_close_verified"
                if valid
                else "verified_primary_export_missing_or_invalid_at_deadline"
            ),
        },
    )
    return 0 if valid else 95


if __name__ == "__main__":
    raise SystemExit(main())
