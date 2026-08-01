#!/usr/bin/env python3
"""Record alias identity and sealed hashes after every handoff alias is locked."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import time
from pathlib import Path
from typing import Any, Sequence


SCHEMA = "bayesfilter.complete_highdim_leaderboard.post_lock_integrity.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _decode(value: str) -> str:
    for escaped, literal in (
        ("\\040", " "),
        ("\\011", "\t"),
        ("\\012", "\n"),
        ("\\134", "\\"),
    ):
        value = value.replace(escaped, literal)
    return value


def _mount_records() -> list[dict[str, Any]]:
    records = []
    with Path("/proc/self/mountinfo").open(encoding="utf-8") as stream:
        for line in stream:
            fields = line.split()
            if len(fields) < 10 or "-" not in fields:
                raise ValueError("malformed mountinfo record")
            separator = fields.index("-")
            records.append(
                {
                    "mount_id": int(fields[0]),
                    "target": _decode(fields[4]),
                    "mount_options": fields[5].split(","),
                    "super_options": fields[separator + 3].split(","),
                }
            )
    return records


def _effective_mount(path: Path, records: list[dict[str, Any]]) -> dict[str, Any]:
    text = str(path)
    matches = [
        record
        for record in records
        if text == record["target"] or text.startswith(record["target"].rstrip("/") + "/")
    ]
    if not matches:
        raise ValueError(f"no effective mount found for alias: {path}")
    record = max(matches, key=lambda value: len(value["target"]))
    if "ro" not in record["mount_options"] or "rw" in record["mount_options"]:
        raise ValueError(f"handoff alias is not locked read-only: {path}")
    return record


def build_receipt(
    *, run_id: str, aliases: Sequence[Path], seal_path: Path
) -> dict[str, Any]:
    if len(aliases) != 3:
        raise ValueError("canonical, snapshot, and staging aliases are required")
    identities = []
    mounts = _mount_records()
    for alias in aliases:
        if not alias.is_absolute() or alias.is_symlink() or not alias.is_dir():
            raise ValueError(f"handoff alias is missing or unsafe: {alias}")
        info = alias.stat()
        if info.st_mode & 0o222:
            raise ValueError(f"handoff alias mode remains writable: {alias}")
        identities.append(
            {
                "path": str(alias),
                "device": info.st_dev,
                "inode": info.st_ino,
                "effective_mount": _effective_mount(alias, mounts),
            }
        )
    if len({(record["device"], record["inode"]) for record in identities}) != 1:
        raise ValueError("handoff aliases do not identify the same directory")
    expected_seal = aliases[1] / f"{run_id}-final-handoff-seal.json"
    if seal_path != expected_seal:
        raise ValueError("final handoff seal path is not the exact snapshot alias")
    if seal_path.is_symlink() or not seal_path.is_file():
        raise ValueError("final handoff seal is missing or unsafe")
    seal_info = seal_path.lstat()
    if (
        not stat.S_ISREG(seal_info.st_mode)
        or seal_info.st_nlink != 1
        or seal_info.st_mode & 0o222
    ):
        raise ValueError("final handoff seal mode or link count is unsafe")
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if not (
        isinstance(seal, dict)
        and seal.get("schema_version")
        == "bayesfilter.complete_highdim_leaderboard.final_handoff_seal.v1"
        and seal.get("run_id") == run_id
        and isinstance(seal.get("files"), list)
    ):
        raise ValueError("final handoff seal contract is invalid")
    canonical = aliases[0]
    rehashed = []
    seen_names: set[str] = set()
    for record in seal["files"]:
        if not isinstance(record, dict):
            raise ValueError("final handoff seal file record is invalid")
        recorded_path = Path(str(record.get("path", "")))
        if recorded_path.parent != aliases[1] or recorded_path.name in seen_names:
            raise ValueError("final handoff seal contains an unsafe or duplicate path")
        name = recorded_path.name
        if not name.startswith(run_id):
            raise ValueError("final handoff seal contains an unscoped path")
        path = canonical / name
        info = path.lstat()
        if (
            path.is_symlink()
            or not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
            or info.st_size != record.get("size")
            or _sha256(path) != record.get("sha256")
        ):
            raise ValueError(f"post-lock sealed-file rehash failed: {name}")
        seen_names.add(name)
        rehashed.append(
            {"name": name, "size": info.st_size, "sha256": record["sha256"]}
        )
    observed_names = {
        path.name
        for path in canonical.iterdir()
        if path.name != seal_path.name
    }
    if seen_names != observed_names:
        raise ValueError("post-lock sealed inventory differs from the handoff")
    return {
        "schema_version": SCHEMA,
        "run_id": run_id,
        "recorded_after_all_aliases_locked_read_only": True,
        "aliases_identify_same_directory": True,
        "aliases": identities,
        "seal_path_at_lock": str(seal_path),
        "seal_sha256_after_lock": _sha256(seal_path),
        "seal_size_after_lock": seal_info.st_size,
        "sealed_files_rehashed_after_lock": True,
        "sealed_files": rehashed,
        "post_lock_match_does_not_prove_absence_of_transient_pre_lock_write": True,
        "recorded_at_epoch": time.time(),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--canonical-alias", type=Path, required=True)
    parser.add_argument("--snapshot-alias", type=Path, required=True)
    parser.add_argument("--staging-alias", type=Path, required=True)
    parser.add_argument("--seal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    expected = Path(f"/tmp/{args.run_id}-post-lock-integrity.json")
    if args.output != expected or args.output.exists() or args.output.is_symlink():
        raise ValueError("post-lock receipt output is not fresh or exact")
    payload = build_receipt(
        run_id=args.run_id,
        aliases=(args.canonical_alias, args.snapshot_alias, args.staging_alias),
        seal_path=args.seal,
    )
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(args.output, 0o444)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
