#!/usr/bin/env python3
"""Freeze and verify the exact source tree disclosed to detached Codex."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import subprocess
from pathlib import Path
from typing import Any, Sequence


SCHEMA = "bayesfilter.complete_highdim_leaderboard.source_snapshot.v1"
EXCLUDED_ROOT_NAMES = {
    ".agents",
    ".cache",
    ".claude",
    ".claude_reviews",
    ".codex",
    ".complete-highdim-preflight-tmp",
    ".localenv",
    ".pytest_cache",
    ".research",
}
EXCLUDED_ANYWHERE_NAMES = {"__pycache__", ".mypy_cache", ".ruff_cache"}
EXCLUDED_PREFIXES = (Path("docs/plans/logs"),)
DYNAMIC_EXCLUDED_PATHS = {
    Path("docs/plans/complete-highdim-leaderboard-exact-command-manifest-2026-07-11.json"),
    Path("docs/plans/artifacts/complete-highdim-leaderboard/source-snapshot-inventory-2026-07-11.json"),
    Path("docs/plans/artifacts/complete-highdim-leaderboard/runtime-fingerprint-2026-07-11.json"),
    Path("docs/plans/bayesfilter-complete-highdim-leaderboard-visible-execution-ledger-2026-07-11.md"),
    Path("docs/plans/bayesfilter-complete-highdim-leaderboard-visible-stop-handoff-2026-07-11.md"),
    Path("docs/reviews/bayesfilter-complete-highdim-leaderboard-launch-readiness-review-receipt-2026-07-11.json"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _excluded(rel: Path) -> bool:
    if rel in DYNAMIC_EXCLUDED_PATHS:
        return True
    if rel.parts and rel.parts[0].startswith(".complete-highdim-source-snapshot-"):
        return True
    if (
        len(rel.parts) == 3
        and rel.parts[:2] == ("docs", "reviews")
        and rel.name.startswith(
            "bayesfilter-complete-highdim-leaderboard-launch-"
        )
    ):
        return True
    if rel.parts and rel.parts[0] in EXCLUDED_ROOT_NAMES:
        return True
    if any(part in EXCLUDED_ANYWHERE_NAMES for part in rel.parts):
        return True
    return any(rel == prefix or prefix in rel.parents for prefix in EXCLUDED_PREFIXES)


def _tracked_log_paths(root: Path) -> set[str]:
    result = subprocess.run(
        ["/usr/bin/git", "ls-files", "-z", "--", "docs/plans/logs"],
        cwd=root,
        check=True,
        stdout=subprocess.PIPE,
        timeout=60,
    )
    return {
        value.decode("utf-8")
        for value in result.stdout.split(b"\0")
        if value
    }


def inventory(root: Path) -> dict[str, Any]:
    root = root.resolve(strict=True)
    tracked_logs = _tracked_log_paths(root) if (root / ".git").is_dir() else set()
    entries: list[dict[str, Any]] = []
    total_bytes = 0
    for current, dirs, files in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        retained_dirs = []
        for name in sorted(dirs):
            path = current_path / name
            rel = path.relative_to(root)
            rel_text = rel.as_posix()
            if path.is_symlink():
                target = os.readlink(path)
                resolved = (path.parent / target).resolve()
                if not resolved.is_relative_to(root) and not target.startswith(
                    "/home/chakwong/anaconda3/envs/tf-gpu/"
                ):
                    raise ValueError(
                        f"snapshot symlink escapes approved roots: {rel_text}"
                    )
                entries.append(
                    {
                        "path": rel_text,
                        "kind": "symlink",
                        "mode": stat.S_IMODE(path.lstat().st_mode),
                        "target": target,
                    }
                )
                continue
            tracked_descendant = any(
                value == rel_text or value.startswith(f"{rel_text}/")
                for value in tracked_logs
            )
            if _excluded(rel) and not tracked_descendant:
                continue
            retained_dirs.append(name)
        dirs[:] = retained_dirs
        for name in sorted(files):
            path = current_path / name
            rel = path.relative_to(root)
            rel_text = rel.as_posix()
            if _excluded(rel) and rel_text not in tracked_logs:
                continue
            info = path.lstat()
            mode = stat.S_IMODE(info.st_mode)
            if stat.S_ISLNK(info.st_mode):
                target = os.readlink(path)
                resolved = (path.parent / target).resolve()
                if not resolved.is_relative_to(root) and not target.startswith(
                    "/home/chakwong/anaconda3/envs/tf-gpu/"
                ):
                    raise ValueError(
                        f"snapshot symlink escapes approved roots: {rel_text}"
                    )
                entries.append(
                    {
                        "path": rel_text,
                        "kind": "symlink",
                        "mode": mode,
                        "target": target,
                    }
                )
                continue
            if not stat.S_ISREG(info.st_mode):
                raise ValueError(f"snapshot contains a special file: {rel_text}")
            digest = _sha256(path)
            entries.append(
                {
                    "path": rel_text,
                    "kind": "file",
                    "mode": mode,
                    "size": info.st_size,
                    "sha256": digest,
                }
            )
            total_bytes += info.st_size
    entries.sort(key=lambda record: record["path"])
    canonical_entries = json.dumps(
        entries, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return {
        "schema_version": SCHEMA,
        "root": str(root),
        "entry_count": len(entries),
        "total_file_bytes": total_bytes,
        "entries_sha256": hashlib.sha256(canonical_entries).hexdigest(),
        "entries": entries,
        "exclusion_policy": {
            "excluded_root_names": sorted(EXCLUDED_ROOT_NAMES),
            "excluded_anywhere_names": sorted(EXCLUDED_ANYWHERE_NAMES),
            "excluded_prefixes": [path.as_posix() for path in EXCLUDED_PREFIXES],
            "dynamic_excluded_paths": sorted(
                path.as_posix() for path in DYNAMIC_EXCLUDED_PATHS
            ),
            "source_snapshot_root_prefix_excluded": True,
            "tracked_docs_plan_logs_reincluded": True,
        },
    }


def _write_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _copy_from_inventory(
    source: Path,
    destination: Path,
    payload: dict[str, Any],
    *,
    allow_existing_empty: bool = False,
) -> None:
    if destination.exists() or destination.is_symlink():
        if (
            not allow_existing_empty
            or destination.is_symlink()
            or not destination.is_dir()
            or any(destination.iterdir())
        ):
            raise FileExistsError(f"snapshot destination already exists: {destination}")
    else:
        destination.mkdir(mode=0o700, parents=True)
    for record in payload["entries"]:
        rel = Path(record["path"])
        source_path = source / rel
        destination_path = destination / rel
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        if record["kind"] == "symlink":
            destination_path.symlink_to(record["target"])
        else:
            with source_path.open("rb") as input_stream, destination_path.open(
                "xb"
            ) as output_stream:
                shutil.copyfileobj(input_stream, output_stream, 1024 * 1024)
            os.chmod(destination_path, record["mode"])
    for value in payload.get("required_empty_directories", []):
        directory = destination / value
        directory.mkdir(parents=True, exist_ok=False)


def _verify_required_empty_directories(root: Path, payload: dict[str, Any]) -> None:
    for value in payload.get("required_empty_directories", []):
        directory = root / value
        if directory.is_symlink() or not directory.is_dir() or any(directory.iterdir()):
            raise ValueError(f"required empty snapshot directory is invalid: {value}")


def _load_inventory(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA:
        raise ValueError("source snapshot inventory schema is invalid")
    return payload


def verify(
    root: Path,
    expected: dict[str, Any],
    *,
    check_required_empty_directories: bool = True,
) -> None:
    observed = inventory(root)
    for key in (
        "schema_version",
        "entry_count",
        "total_file_bytes",
        "entries_sha256",
        "entries",
        "exclusion_policy",
    ):
        if observed.get(key) != expected.get(key):
            raise ValueError(f"source snapshot verification failed: {key}")
    if check_required_empty_directories:
        _verify_required_empty_directories(root, expected)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--source", type=Path, required=True)
    freeze.add_argument("--snapshot-root", type=Path, required=True)
    freeze.add_argument("--inventory", type=Path, required=True)
    freeze.add_argument("--required-empty-directory", action="append", default=[])
    check = subparsers.add_parser("verify")
    check.add_argument("--root", type=Path, required=True)
    check.add_argument("--inventory", type=Path, required=True)
    check.add_argument(
        "--allow-populated-required-empty-directories", action="store_true"
    )
    materialize = subparsers.add_parser("materialize")
    materialize.add_argument("--source", type=Path, required=True)
    materialize.add_argument("--destination", type=Path, required=True)
    materialize.add_argument("--inventory", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.command == "freeze":
        payload = inventory(args.source)
        payload["required_empty_directories"] = sorted(
            set(args.required_empty_directory)
        )
        _copy_from_inventory(args.source.resolve(strict=True), args.snapshot_root, payload)
        frozen = inventory(args.snapshot_root)
        payload["frozen_root"] = str(args.snapshot_root.resolve(strict=True))
        if frozen["entries"] != payload["entries"]:
            raise ValueError("frozen snapshot differs from its source inventory")
        _verify_required_empty_directories(args.snapshot_root, payload)
        _write_exclusive(args.inventory, payload)
        print(f"SOURCE_SNAPSHOT_FROZEN {args.snapshot_root}")
        return 0

    payload = _load_inventory(args.inventory)
    if args.command == "verify":
        verify(
            args.root,
            payload,
            check_required_empty_directories=(
                not args.allow_populated_required_empty_directories
            ),
        )
        print(f"SOURCE_SNAPSHOT_VERIFY_PASS {args.root}")
        return 0

    verify(args.source, payload, check_required_empty_directories=False)
    _copy_from_inventory(
        args.source.resolve(strict=True),
        args.destination,
        payload,
        allow_existing_empty=True,
    )
    verify(args.destination, payload)
    print(f"SOURCE_SNAPSHOT_MATERIALIZE_PASS {args.destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
