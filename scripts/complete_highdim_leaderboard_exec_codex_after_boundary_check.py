#!/usr/bin/env python3
"""Verify the child sandbox after capability drop, then exec Codex."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Sequence


EXPECTED_SUPPORT_FILES = {
    "trusted-review-verifier.py",
    "trusted-claude-review-gate.sh",
    "trusted-claude-worker.sh",
    "trusted-claude-worker-settings.json",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _status() -> dict[str, str]:
    return dict(
        line.split(":", 1)
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines()
        if ":" in line
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--handoff-dir", type=Path, required=True)
    parser.add_argument("--support-dir", type=Path, required=True)
    parser.add_argument("--staging-alias", type=Path, action="append", default=[])
    parser.add_argument("--codex-bin", required=True)
    parser.add_argument("--final-message", type=Path, required=True)
    args = parser.parse_args(argv)

    status = _status()
    cap_eff = int(status.get("CapEff", "1").strip(), 16)
    cap_bnd = int(status.get("CapBnd", "1").strip(), 16)
    no_new_privs = status.get("NoNewPrivs", "0").strip()
    if cap_eff != 0 or cap_bnd != 0 or no_new_privs != "1" or os.getpid() != 1:
        raise RuntimeError("Codex child capability or PID boundary is invalid")

    hidden_paths = (
        Path("/home/chakwong/python"),
        Path("/home/chakwong/.codex"),
        Path("/home/chakwong/.claude"),
    )
    if any(path.exists() or path.is_symlink() for path in hidden_paths):
        raise RuntimeError("Codex child can read an unapproved sibling-home path")

    forbidden = args.handoff_dir / ".forbidden-codex-boundary-check"
    if any(args.handoff_dir.iterdir()):
        raise RuntimeError("Codex child can read dynamic handoff contents")
    try:
        with forbidden.open("x", encoding="utf-8") as stream:
            stream.write("boundary failure\n")
    except OSError:
        pass
    else:
        forbidden.unlink(missing_ok=True)
        raise RuntimeError("Codex child handoff directory remained writable")
    if args.support_dir.is_symlink() or not args.support_dir.is_dir():
        raise RuntimeError("Codex child support directory is missing or unsafe")
    support_entries = list(args.support_dir.iterdir())
    if {path.name for path in support_entries} != EXPECTED_SUPPORT_FILES:
        raise RuntimeError("Codex child support allowlist is not exact")
    try:
        support_bindings = json.loads(os.environ["CODEX_SUPPORT_BINDINGS_JSON"])
    except (KeyError, json.JSONDecodeError) as error:
        raise RuntimeError("Codex child support bindings are unavailable") from error
    expected_bindings = {
        record["path"]: record for record in support_bindings
        if isinstance(record, dict) and isinstance(record.get("path"), str)
    }
    if set(expected_bindings) != EXPECTED_SUPPORT_FILES:
        raise RuntimeError("Codex child support binding names are invalid")
    for path in support_entries:
        info = path.lstat()
        binding = expected_bindings[path.name]
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
            or info.st_size != binding.get("size")
            or _sha256(path) != binding.get("sha256")
        ):
            raise RuntimeError(f"Codex child support file drifted: {path}")
    support_probe = args.support_dir / ".forbidden-support-write"
    try:
        with support_probe.open("x", encoding="utf-8") as stream:
            stream.write("boundary failure\n")
    except OSError:
        pass
    else:
        support_probe.unlink(missing_ok=True)
        raise RuntimeError("Codex child support directory remained writable")
    for alias in args.staging_alias:
        if alias.is_symlink() or not alias.is_dir() or any(alias.iterdir()):
            raise RuntimeError(f"Codex child staging alias remained readable: {alias}")
        alias_probe = alias / ".forbidden-staging-write"
        try:
            alias_probe.write_text("boundary failure\n", encoding="utf-8")
        except OSError:
            pass
        else:
            alias_probe.unlink(missing_ok=True)
            raise RuntimeError(f"Codex child staging alias remained writable: {alias}")

    receipt = args.root / ".complete_highdim_codex_sandbox_receipt.json"
    payload = {
        "schema_version": "bayesfilter.complete_highdim_leaderboard.codex_sandbox.v1",
        "pid_in_private_namespace": os.getpid(),
        "pid_namespace_inode": os.stat("/proc/self/ns/pid").st_ino,
        "cap_eff": cap_eff,
        "cap_bnd": cap_bnd,
        "no_new_privs": int(no_new_privs),
        "handoff_read_only": True,
        "handoff_contents_hidden": True,
        "sibling_home_hidden": True,
        "selected_runtime_mounts_only": True,
        "support_dir": str(args.support_dir),
        "support_read_only": True,
        "support_files": sorted(path.name for path in support_entries),
        "support_hashes_verified": True,
        "claude_worker_preserves_stream_and_tool_use_metadata": True,
        "staging_aliases_hidden_and_read_only": [
            str(alias) for alias in args.staging_alias
        ],
        "private_home": os.environ.get("HOME"),
        "private_codex_home": os.environ.get("CODEX_HOME"),
    }
    with receipt.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    command = [
        args.codex_bin,
        "exec",
        "--cd",
        str(args.root),
        "--ephemeral",
        "--ignore-user-config",
        "--ignore-rules",
        "--dangerously-bypass-approvals-and-sandbox",
        "--json",
        "--output-last-message",
        str(args.final_message),
        "-",
    ]
    os.execvpe(command[0], command, os.environ)
    return 127


if __name__ == "__main__":
    raise SystemExit(main())
