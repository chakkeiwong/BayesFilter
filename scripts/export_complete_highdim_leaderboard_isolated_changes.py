#!/usr/bin/env python3
"""Snapshot and export changed files from an isolated leaderboard workspace."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
import tarfile
from pathlib import Path
from typing import Any, Sequence


EXCLUDED_DIR_NAMES = {
    ".git",
    ".claude",
    ".codex",
    ".claude_reviews",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "__pycache__",
}
EXCLUDED_REL_PREFIXES = ("docs/plans/logs/",)
SNAPSHOT_SCHEMA = "bayesfilter.complete_highdim_leaderboard.workspace_snapshot.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _excluded(rel: Path) -> bool:
    if any(part in EXCLUDED_DIR_NAMES for part in rel.parts):
        return True
    value = rel.as_posix()
    return any(value.startswith(prefix) for prefix in EXCLUDED_REL_PREFIXES)


def snapshot(root: Path, *, root_identity: Path | None = None) -> dict[str, Any]:
    files: dict[str, Any] = {}
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if _excluded(rel) or path.is_dir():
            continue
        value = rel.as_posix()
        if path.is_symlink():
            files[value] = {
                "kind": "symlink",
                "target": os.readlink(path),
            }
        elif path.is_file():
            mode = stat.S_IMODE(path.stat().st_mode)
            files[value] = {
                "kind": "file",
                "sha256": _sha256(path),
                "size": path.stat().st_size,
                "mode": mode,
            }
    return {
        "schema_version": SNAPSHOT_SCHEMA,
        "root": str((root_identity or root).resolve()),
        "files": files,
    }


def _write_json(path: Path, payload: Any, *, exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if exclusive:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(serialized)
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(serialized, encoding="utf-8")
    temporary.replace(path)


def _git_output(root: Path, *args: str) -> str:
    result = subprocess.run(
        args,
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed with exit {result.returncode}: {' '.join(args)}\n"
            f"{result.stdout}"
        )
    return result.stdout


def _validate_output_dir(path: Path) -> None:
    if not path.is_absolute():
        raise ValueError("output directory must be absolute")
    if path.is_symlink() or not path.is_dir():
        raise ValueError("output directory must be a real directory")
    if path.stat().st_nlink < 2:
        raise ValueError("output directory link metadata is invalid")
    for child in path.iterdir():
        if child.is_symlink() or not child.is_file():
            raise ValueError(f"handoff directory contains unsafe entry: {child}")
        if child.stat().st_nlink != 1:
            raise ValueError(f"handoff file is hard-linked: {child}")


def _validate_baseline(baseline: Any, root: Path) -> dict[str, Any]:
    if not isinstance(baseline, dict):
        raise ValueError("baseline snapshot must be a JSON object")
    if baseline.get("schema_version") != SNAPSHOT_SCHEMA:
        raise ValueError("baseline snapshot schema is invalid")
    if baseline.get("root") != str(root.resolve()):
        raise ValueError("baseline snapshot belongs to a different workspace")
    files = baseline.get("files")
    if not isinstance(files, dict):
        raise ValueError("baseline snapshot files must be an object")
    for rel, record in files.items():
        path = Path(rel)
        if (
            not isinstance(rel, str)
            or not rel
            or path.is_absolute()
            or ".." in path.parts
            or not isinstance(record, dict)
            or record.get("kind") not in {"file", "symlink"}
        ):
            raise ValueError(f"invalid baseline entry: {rel!r}")
    return baseline


def _assert_available(paths: Sequence[Path]) -> None:
    collisions = [str(path) for path in paths if path.exists() or path.is_symlink()]
    if collisions:
        raise FileExistsError(f"refusing to overwrite export files: {collisions}")


def export_changes(
    root: Path,
    baseline_path: Path,
    output_dir: Path,
    run_id: str,
    *,
    baseline_root_identity: Path | None = None,
    export_label: str | None = None,
    hash_excluded_names: Sequence[str] = (),
) -> dict[str, Any]:
    root = root.resolve()
    if not output_dir.is_absolute() or output_dir.is_symlink():
        raise ValueError("output directory must be an absolute real path")
    output_dir = output_dir.resolve(strict=True)
    _validate_output_dir(output_dir)
    baseline = _validate_baseline(
        json.loads(baseline_path.read_text(encoding="utf-8")),
        (baseline_root_identity or root).resolve(),
    )
    current = snapshot(root)
    before = baseline["files"]
    after = current["files"]
    added = sorted(set(after) - set(before))
    deleted = sorted(set(before) - set(after))
    modified = sorted(path for path in set(before) & set(after) if before[path] != after[path])

    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{run_id}-{export_label}" if export_label else run_id
    manifest_path = output_dir / f"{prefix}-isolated-change-manifest.json"
    archive_path = output_dir / f"{prefix}-isolated-changed-files.tar.gz"
    diff_path = output_dir / f"{prefix}-isolated-tracked.diff"
    status_path = output_dir / f"{prefix}-isolated-git-status.txt"
    hashes_path = output_dir / f"{prefix}-export-sha256.json"

    _assert_available(
        (manifest_path, archive_path, diff_path, status_path, hashes_path)
    )

    with tarfile.open(archive_path, "w:gz") as archive:
        for rel in [*added, *modified]:
            path = root / rel
            if path.exists() or path.is_symlink():
                archive.add(path, arcname=rel, recursive=False)

    diff_path.write_text(
        _git_output(root, "git", "diff", "--binary", "--no-ext-diff"),
        encoding="utf-8",
    )
    status_path.write_text(
        _git_output(root, "git", "status", "--short"), encoding="utf-8"
    )
    manifest = {
        "schema_version": "bayesfilter.complete_highdim_leaderboard.isolated_export.v1",
        "run_id": run_id,
        "export_label": export_label,
        "root_inside_isolated_namespace": str(root.resolve()),
        "baseline_snapshot": str(baseline_path),
        "added": added,
        "modified": modified,
        "deleted": deleted,
        "archive": str(archive_path),
        "tracked_diff": str(diff_path),
        "git_status": str(status_path),
        "automatic_merge_performed": False,
    }
    _write_json(manifest_path, manifest, exclusive=True)

    excluded_hash_names = set(hash_excluded_names)
    export_paths = sorted(
        path
        for path in output_dir.glob(f"{run_id}-*")
        if path.is_file()
        and not path.name.endswith("-export-sha256.json")
        and path.name not in excluded_hash_names
    )
    hashes = {
        "schema_version": "bayesfilter.complete_highdim_leaderboard.export_hashes.v1",
        "run_id": run_id,
        "export_label": export_label,
        "hash_excluded_live_or_control_files": sorted(excluded_hash_names),
        "files": [
            {"path": str(path), "sha256": _sha256(path), "size": path.stat().st_size}
            for path in export_paths
        ],
    }
    _write_json(hashes_path, hashes, exclusive=True)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    snapshot_parser = subparsers.add_parser("snapshot")
    snapshot_parser.add_argument("--root", type=Path, required=True)
    snapshot_parser.add_argument("--output", type=Path, required=True)
    snapshot_parser.add_argument("--root-identity", type=Path)
    export_parser = subparsers.add_parser("export")
    export_parser.add_argument("--root", type=Path, required=True)
    export_parser.add_argument("--baseline", type=Path, required=True)
    export_parser.add_argument("--output-dir", type=Path, required=True)
    export_parser.add_argument("--run-id", required=True)
    export_parser.add_argument("--baseline-root-identity", type=Path)
    export_parser.add_argument("--export-label")
    export_parser.add_argument("--hash-exclude-name", action="append", default=[])
    args = parser.parse_args(argv)

    if args.command == "snapshot":
        _write_json(
            args.output,
            snapshot(
                args.root.resolve(),
                root_identity=(
                    args.root_identity.resolve() if args.root_identity else None
                ),
            ),
            exclusive=True,
        )
        print(f"ISOLATED_SNAPSHOT_WRITTEN {args.output}")
    else:
        manifest = export_changes(
            args.root.resolve(),
            args.baseline.resolve(),
            args.output_dir.resolve(),
            args.run_id,
            baseline_root_identity=(
                args.baseline_root_identity.resolve()
                if args.baseline_root_identity
                else None
            ),
            export_label=args.export_label,
            hash_excluded_names=args.hash_exclude_name,
        )
        print(
            "ISOLATED_EXPORT_WRITTEN "
            f"added={len(manifest['added'])} modified={len(manifest['modified'])} "
            f"deleted={len(manifest['deleted'])}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
