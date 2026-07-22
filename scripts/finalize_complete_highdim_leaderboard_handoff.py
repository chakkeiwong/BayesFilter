#!/usr/bin/env python3
"""Seal the per-run handoff after every detached producer has closed."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import time
from pathlib import Path
from typing import Any, Sequence


SCHEMA = "bayesfilter.complete_highdim_leaderboard.final_handoff_seal.v1"
PRODUCER_SCHEMA = "bayesfilter.complete_highdim_leaderboard.producer_descriptor.v1"
SCRIPT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_PRODUCER_COMMANDS = {
    "detached_supervisor": {
        "execution_path": (
            SCRIPT_ROOT
            / "scripts/complete_highdim_leaderboard_overnight_supervisor.py"
        ),
        "hash_source": (
            SCRIPT_ROOT
            / "scripts/complete_highdim_leaderboard_overnight_supervisor.py"
        ),
    },
    "primary_export_verification_watchdog": {
        "execution_path": (
            SCRIPT_ROOT / "scripts/complete_highdim_leaderboard_watchdog.py"
        ),
        "hash_source": (
            SCRIPT_ROOT / "scripts/complete_highdim_leaderboard_watchdog.py"
        ),
    },
}
EXPORT_SCHEMAS = {
    "bayesfilter.complete_highdim_leaderboard.isolated_export.v1",
    "bayesfilter.complete_highdim_leaderboard.export_hashes.v1",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _process_identity(pid: int) -> dict[str, int] | None:
    status = Path(f"/proc/{pid}/stat")
    try:
        value = status.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    closing = value.rfind(")")
    if closing < 0:
        raise ValueError(f"malformed process stat for PID {pid}")
    tail = value[closing + 2 :].split()
    if len(tail) < 20:
        raise ValueError(f"incomplete process stat for PID {pid}")
    return {
        "state_zombie": int(tail[0] == "Z"),
        "ppid": int(tail[1]),
        "pgid": int(tail[2]),
        "sid": int(tail[3]),
        "start_time_ticks": int(tail[19]),
    }


def _process_group_alive(pgid: int) -> bool:
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            value = (entry / "stat").read_text(encoding="utf-8")
        except FileNotFoundError:
            continue
        except PermissionError:
            return True
        closing = value.rfind(")")
        if closing < 0:
            return True
        tail = value[closing + 2 :].split()
        if len(tail) < 4:
            return True
        if int(tail[2]) == pgid and tail[0] != "Z":
            return True
    return False


def _load_producer_descriptor(path: Path, *, handoff_dir: Path, run_id: str) -> dict:
    if path.parent.resolve(strict=True) != handoff_dir.resolve(strict=True):
        raise ValueError(f"producer descriptor is outside the handoff: {path}")
    if path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1:
        raise ValueError(f"producer descriptor is missing or unsafe: {path}")
    payload = _load_object(path)
    required_ints = ("pid", "pgid", "sid", "start_time_ticks", "pid_namespace_inode")
    if not (
        payload.get("schema_version") == PRODUCER_SCHEMA
        and payload.get("run_id") == run_id
        and payload.get("role")
        in {"detached_supervisor", "primary_export_verification_watchdog"}
        and all(
            isinstance(payload.get(name), int) and payload[name] > 0
            for name in required_ints
        )
        and isinstance(payload.get("command_path"), str)
        and isinstance(payload.get("command_sha256"), str)
        and len(payload["command_sha256"]) == 64
    ):
        raise ValueError(f"producer descriptor contract is invalid: {path}")
    expected_command = EXPECTED_PRODUCER_COMMANDS[payload["role"]]
    if (
        payload["command_path"] != str(expected_command["execution_path"])
        or payload["command_sha256"] != _sha256(expected_command["hash_source"])
        or payload["pid_namespace_inode"] != os.stat("/proc/self/ns/pid").st_ino
    ):
        raise ValueError(f"producer descriptor identity is invalid: {path}")
    return payload


def _producer_closed(descriptor: dict) -> bool:
    identity = _process_identity(descriptor["pid"])
    same_identity_alive = bool(
        identity
        and not identity["state_zombie"]
        and identity["pgid"] == descriptor["pgid"]
        and identity["sid"] == descriptor["sid"]
        and identity["start_time_ticks"] == descriptor["start_time_ticks"]
    )
    return not same_identity_alive and not _process_group_alive(descriptor["pgid"])


def _extra_live_namespace_pids() -> list[int]:
    table: dict[int, dict[str, int]] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        identity = _process_identity(int(entry.name))
        if identity is not None and not identity["state_zombie"]:
            table[int(entry.name)] = identity
    ancestors: set[int] = set()
    pid = os.getpid()
    while pid > 0 and pid not in ancestors:
        ancestors.add(pid)
        identity = table.get(pid)
        if identity is None:
            break
        pid = identity["ppid"]
    return sorted(set(table) - ancestors)


def _whole_private_namespace_quiescent() -> bool:
    if _extra_live_namespace_pids():
        return False
    time.sleep(0.05)
    return not _extra_live_namespace_pids()


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
            payload = _load_object(path)
        except ValueError:
            continue
        if payload.get("schema_version") in EXPORT_SCHEMAS:
            rejected.add(path.name)
    return sorted(rejected)


def wait_for_producers(
    descriptor_files: Sequence[Path],
    *,
    handoff_dir: Path,
    run_id: str,
    deadline_monotonic: float,
) -> list[dict]:
    if len(descriptor_files) != 2:
        raise ValueError("exactly two producer descriptors are required")
    descriptors = [
        _load_producer_descriptor(path, handoff_dir=handoff_dir, run_id=run_id)
        for path in descriptor_files
    ]
    if {descriptor["role"] for descriptor in descriptors} != {
        "detached_supervisor",
        "primary_export_verification_watchdog",
    }:
        raise ValueError("producer descriptor roles are incomplete")
    while time.monotonic() < deadline_monotonic:
        if all(
            _producer_closed(descriptor) for descriptor in descriptors
        ) and _whole_private_namespace_quiescent():
            return descriptors
        remaining = deadline_monotonic - time.monotonic()
        if remaining > 0.0:
            time.sleep(min(1.0, remaining))
    raise TimeoutError("handoff producers did not close before the seal deadline")


def _load_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        raise ValueError(f"invalid required completion JSON: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"required completion JSON is not an object: {path}")
    return payload


def _validate_post_export_verification(
    path: Path, *, handoff_dir: Path, run_id: str
) -> None:
    payload = _load_object(path)
    if not (
        payload.get("schema_version")
        == "bayesfilter.complete_highdim_leaderboard.post_export_verification.v1"
        and payload.get("run_id") == run_id
        and payload.get("export_label") == "primary"
        and payload.get("all_bound_files_recomputed") is True
        and payload.get("verified_after_codex_namespace_closed") is True
    ):
        raise ValueError("post-export verification contract is invalid")
    export_hashes = Path(str(payload.get("export_hashes", "")))
    try:
        resolved = export_hashes.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise ValueError("post-export hash ledger is missing") from error
    if (
        export_hashes.is_symlink()
        or not export_hashes.is_file()
        or resolved.parent != handoff_dir
        or _sha256(export_hashes) != payload.get("export_hashes_sha256")
    ):
        raise ValueError("post-export hash ledger binding is invalid")
    hashes = _load_object(export_hashes)
    if not (
        hashes.get("schema_version")
        == "bayesfilter.complete_highdim_leaderboard.export_hashes.v1"
        and hashes.get("run_id") == run_id
        and hashes.get("export_label") == "primary"
        and isinstance(hashes.get("files"), list)
    ):
        raise ValueError("post-export hash ledger contract is invalid")
    for record in hashes["files"]:
        if not isinstance(record, dict):
            raise ValueError("post-export hash record is invalid")
        bound = Path(str(record.get("path", "")))
        try:
            bound_resolved = bound.resolve(strict=True)
        except (OSError, RuntimeError) as error:
            raise ValueError("post-export bound file is missing") from error
        if (
            bound.is_symlink()
            or not bound.is_file()
            or bound_resolved.parent != handoff_dir
            or bound.stat().st_nlink != 1
            or bound.stat().st_size != record.get("size")
            or _sha256(bound) != record.get("sha256")
        ):
            raise ValueError("post-export bound file verification failed")
    namespace_closed = Path(str(payload.get("codex_namespace_closed_receipt", "")))
    try:
        namespace_resolved = namespace_closed.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise ValueError("Codex namespace-close receipt is missing") from error
    if (
        namespace_closed.is_symlink()
        or not namespace_closed.is_file()
        or namespace_resolved.parent != handoff_dir
        or namespace_closed.name != f"{run_id}-codex-namespace-closed.json"
        or _sha256(namespace_closed)
        != payload.get("codex_namespace_closed_receipt_sha256")
    ):
        raise ValueError("Codex namespace-close receipt binding is invalid")
    closed = _load_object(namespace_closed)
    if not (
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
    ):
        raise ValueError("Codex namespace-close receipt contract is invalid")


def _validate_foreground_outcome(path: Path, *, run_id: str) -> None:
    payload = _load_object(path)
    if not (
        payload.get("schema_version")
        == "bayesfilter.complete_highdim_leaderboard.foreground_outcome.v2"
        and payload.get("run_id") == run_id
        and payload.get("producers_closed_before_outcome") is True
        and payload.get("producer_descriptors_valid") is True
        and payload.get("watchdog_primary_verification_passed") is True
        and payload.get("post_export_verification_present") is True
        and payload.get("primary_export_present") is True
        and payload.get("fallback_export_allowed") is False
        and payload.get("fallback_export_present") is False
        and payload.get("primary_export_only_observed") is True
        and payload.get("unapproved_export_artifacts") == []
        and payload.get("whole_outer_pid_namespace_quiescent_before_outcome") is True
    ):
        raise ValueError("foreground outcome completion contract is invalid")


def _validate_watchdog_status(path: Path, *, run_id: str) -> None:
    payload = _load_object(path)
    if not (
        payload.get("schema_version")
        == "bayesfilter.complete_highdim_leaderboard.watchdog.v3"
        and payload.get("run_id") == run_id
        and payload.get("primary_export_verified") is True
        and payload.get("primary_export_only") is True
        and payload.get("primary_export_only_observed") is True
        and payload.get("unapproved_export_artifacts") == []
        and payload.get("fallback_export_allowed") is False
        and payload.get("writable_copy_read_by_watchdog") is False
        and payload.get("writable_copy_written_by_watchdog") is False
        and payload.get("verification_exit_code") == 0
    ):
        raise ValueError("watchdog primary-verification contract is invalid")


def seal(
    handoff_dir: Path,
    run_id: str,
    *,
    launcher_exit_code: int,
    approval_instance_id: str,
    required_files: Sequence[Path] = (),
    producer_descriptors: Sequence[dict] = (),
) -> dict[str, Any]:
    if not handoff_dir.is_absolute() or handoff_dir.is_symlink():
        raise ValueError("handoff must be an absolute real directory")
    handoff_dir = handoff_dir.resolve(strict=True)
    output = handoff_dir / f"{run_id}-final-handoff-seal.json"
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"final handoff seal already exists: {output}")
    if {descriptor.get("role") for descriptor in producer_descriptors} != set(
        EXPECTED_PRODUCER_COMMANDS
    ):
        raise ValueError("verified producer descriptors are required for sealing")
    extra_pids = _extra_live_namespace_pids()
    if extra_pids or not _whole_private_namespace_quiescent():
        raise ValueError(
            f"outer PID namespace is not quiescent before sealing: {extra_pids}"
        )
    unapproved_exports = _unapproved_export_artifacts(handoff_dir, run_id)
    if unapproved_exports:
        raise ValueError(
            f"handoff contains unapproved export artifacts: {unapproved_exports}"
        )
    required_records = []
    for required in required_files:
        if required.parent.resolve(strict=True) != handoff_dir:
            raise ValueError(f"required seal input is outside the handoff: {required}")
        if required.is_symlink() or not required.is_file():
            raise ValueError(f"required seal input is missing or unsafe: {required}")
        required_records.append(
            {"path": str(required), "sha256": _sha256(required)}
        )
    verification = handoff_dir / f"{run_id}-post-export-verification.json"
    outcome = handoff_dir / f"{run_id}-foreground-outcome.json"
    watchdog_status = handoff_dir / f"{run_id}-watchdog-status.json"
    if verification in required_files:
        _validate_post_export_verification(
            verification, handoff_dir=handoff_dir, run_id=run_id
        )
    if outcome in required_files:
        _validate_foreground_outcome(outcome, run_id=run_id)
    if watchdog_status in required_files:
        _validate_watchdog_status(watchdog_status, run_id=run_id)
    records = []
    for path in sorted(handoff_dir.iterdir()):
        if path == output:
            continue
        if not path.name.startswith(run_id):
            raise ValueError(f"handoff contains an unscoped entry: {path}")
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ValueError(f"handoff contains an unsafe entry: {path}")
        records.append(
            {"path": str(path), "size": info.st_size, "sha256": _sha256(path)}
        )
    payload = {
        "schema_version": SCHEMA,
        "run_id": run_id,
        "approval_instance_id": approval_instance_id,
        "launcher_exit_code": launcher_exit_code,
        "sealed_after_all_producers_closed": True,
        "whole_outer_pid_namespace_quiescent_before_seal": True,
        "primary_export_only_observed_by_finalizer": True,
        "unapproved_export_artifacts": [],
        "producer_descriptors_verified": [
            {
                "role": descriptor["role"],
                "pid": descriptor["pid"],
                "pgid": descriptor["pgid"],
                "sid": descriptor["sid"],
                "start_time_ticks": descriptor["start_time_ticks"],
                "pid_namespace_inode": descriptor["pid_namespace_inode"],
            }
            for descriptor in producer_descriptors
        ],
        "seal_file_excluded_from_its_own_hash_ledger": True,
        "post_seal_writes_forbidden": True,
        "required_completion_files": required_records,
        "automatic_merge_performed": False,
        "sealed_at_epoch": time.time(),
        "files": records,
    }
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--launcher-exit-code", type=int, required=True)
    parser.add_argument("--approval-instance-id", required=True)
    parser.add_argument("--deadline-monotonic", type=float, required=True)
    parser.add_argument(
        "--producer-descriptor", type=Path, action="append", default=[]
    )
    parser.add_argument("--required-file", type=Path, action="append", default=[])
    args = parser.parse_args(argv)
    producers = wait_for_producers(
        args.producer_descriptor,
        handoff_dir=args.handoff_dir,
        run_id=args.run_id,
        deadline_monotonic=args.deadline_monotonic,
    )
    seal(
        args.handoff_dir,
        args.run_id,
        launcher_exit_code=args.launcher_exit_code,
        approval_instance_id=args.approval_instance_id,
        required_files=args.required_file,
        producer_descriptors=producers,
    )
    print(f"FINAL_HANDOFF_SEAL_PASS {args.handoff_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
