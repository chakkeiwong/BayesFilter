#!/usr/bin/env python3
"""Materialize the frozen workspace, freeze its baseline, and start verification."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence


CODEX_SUPPORT_DIR_NAME = ".complete_highdim_codex_support"

def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_exclusive(path: Path, payload: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _require_unexpired(approval_not_after_epoch: int, *, action: str) -> None:
    if time.time() >= approval_not_after_epoch:
        raise RuntimeError(f"one-time launch approval expired before {action}")


def _remaining_timeout(
    started_monotonic: float, deadline_seconds: int, maximum: float
) -> float:
    remaining = started_monotonic + deadline_seconds - time.monotonic()
    if remaining <= 0.0:
        raise TimeoutError("common monotonic preparation budget is exhausted")
    return min(remaining, maximum)


def _safe_handoff(path: Path, run_id: str, source_root: Path) -> None:
    expected = source_root / "docs/plans/logs" / run_id
    if path != expected or path.is_symlink() or not path.is_dir():
        raise RuntimeError("handoff directory is not the exact fresh per-run path")
    for child in path.iterdir():
        if child.is_symlink() or not child.is_file() or child.stat().st_nlink != 1:
            raise RuntimeError(f"unsafe handoff entry: {child}")


def _install_snapshot_overlays(manifest: dict, launch_root: Path) -> None:
    program_root = Path(manifest["program_root"])
    if program_root.is_symlink() or not program_root.is_dir():
        raise RuntimeError("manifest program root is missing or unsafe")
    for binding in manifest["snapshot_overlay_files"]:
        rel = Path(binding["path"])
        if rel.is_absolute() or not rel.parts or ".." in rel.parts:
            raise RuntimeError(f"unsafe snapshot overlay path: {rel}")
        source = program_root / rel
        destination = launch_root / rel
        if (
            binding.get("exists") is not True
            or source.is_symlink()
            or not source.is_file()
            or str(source.resolve(strict=True)) != binding.get("resolved_path")
            or _sha256(source) != binding.get("sha256")
        ):
            raise RuntimeError(f"snapshot overlay drifted: {rel}")
        if destination.exists() or destination.is_symlink():
            raise RuntimeError(f"snapshot overlay destination collided: {rel}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with source.open("rb") as input_stream, destination.open("xb") as output_stream:
            shutil.copyfileobj(input_stream, output_stream, 1024 * 1024)
        os.chmod(destination, source.stat().st_mode & 0o777)
        if _sha256(destination) != binding["sha256"]:
            raise RuntimeError(f"copied snapshot overlay drifted: {rel}")


def _copy_exclusive(source: Path, destination: Path) -> None:
    with source.open("rb") as source_stream, destination.open("xb") as output_stream:
        shutil.copyfileobj(source_stream, output_stream, 1024 * 1024)
    os.chmod(destination, source.stat().st_mode & 0o777)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--launch-root", type=Path, required=True)
    parser.add_argument("--handoff-dir", type=Path, required=True)
    parser.add_argument("--nonce", required=True)
    parser.add_argument("--started-epoch", type=float, required=True)
    parser.add_argument("--started-monotonic", type=float, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--source-inventory", type=Path, required=True)
    parser.add_argument("--expected-source-inventory-sha256", required=True)
    parser.add_argument("--runtime-fingerprint", type=Path, required=True)
    parser.add_argument("--expected-runtime-fingerprint-sha256", required=True)
    parser.add_argument("--watchdog-verification-deadline", type=int, default=27360)
    parser.add_argument("--approval-not-after-epoch", type=int, required=True)
    args = parser.parse_args(argv)

    source = args.source_root.resolve(strict=True)
    _require_unexpired(args.approval_not_after_epoch, action="copy preparation")
    if args.launch_root.is_symlink() or not args.launch_root.is_dir():
        raise RuntimeError("launcher did not create a real launch root")
    if any(args.launch_root.iterdir()):
        raise RuntimeError("launcher-created launch root is not empty")
    _safe_handoff(args.handoff_dir, args.run_id, source)
    if time.monotonic() - args.started_monotonic > 1800:
        raise RuntimeError("copy preparation started too late")
    if _sha256(args.manifest) != args.expected_manifest_sha256:
        raise RuntimeError("launch manifest drifted before copy")
    if _sha256(args.source_inventory) != args.expected_source_inventory_sha256:
        raise RuntimeError("source inventory drifted before copy")
    if _sha256(args.runtime_fingerprint) != args.expected_runtime_fingerprint_sha256:
        raise RuntimeError("runtime fingerprint artifact drifted before copy")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    concrete = manifest["concrete_run"]
    limitations = manifest.get("owner_accepted_run_scoped_limitations")
    post_run_audit = manifest.get("mandatory_post_run_integrity_audit")
    if (
        concrete.get("source_snapshot_root") != str(source)
        or concrete.get("source_inventory") != str(args.source_inventory)
        or concrete.get("runtime_fingerprint") != str(args.runtime_fingerprint)
    ):
        raise RuntimeError("manifest source or runtime identity is inconsistent")
    expected_limitation_ids = [
        "CLAUDE_TOOL_CAPABILITY",
        "CLAUDE_CODEX_CREDENTIAL_ACCESS",
        "PRIMARY_EXPORT_COMPLETENESS",
        "SEAL_LOCK_TOCTOU",
        "TRUSTED_PREFLIGHT_OUTER_COVERAGE",
    ]
    if not (
        manifest.get("schema_version")
        == "bayesfilter.complete_highdim_leaderboard.launch_manifest.v7"
        and isinstance(limitations, dict)
        and limitations.get("run_id") == args.run_id
        and limitations.get("accepted_limitation_ids") == expected_limitation_ids
        and limitations.get("repository_default") is False
        and limitations.get("reusable_for_other_runs") is False
        and limitations.get("technical_guarantees_repaired") is False
        and isinstance(post_run_audit, dict)
        and post_run_audit.get("required_before_completion_or_release_claim")
        is True
        and post_run_audit.get("semantic_inspection_receipt_required") is True
        and post_run_audit.get("post_lock_receipt", {}).get(
            "required_for_structural_pass"
        )
        is True
        and post_run_audit.get("phase8_phase9_validator_exit_zero_and_all_checks_required")
        is True
        and post_run_audit.get(
            "launch_itself_grants_completion_or_release_authority"
        )
        is False
    ):
        raise RuntimeError("manifest waiver or post-run audit contract is invalid")
    for binding in manifest["external_bound_files"]:
        external = Path(binding["path"])
        if (
            not external.is_file()
            or str(external.resolve()) != binding["resolved_path"]
            or _sha256(external) != binding["sha256"]
        ):
            raise RuntimeError(f"external bound file drifted: {binding['path']}")
    for binding in manifest["runtime_executables"]:
        executable = Path(binding["path"])
        if (
            not executable.is_file()
            or str(executable.resolve()) != binding["resolved_path"]
            or _sha256(executable) != binding["sha256"]
        ):
            raise RuntimeError(f"runtime executable drifted: {binding['name']}")

    snapshot_helper = source / (
        "scripts/freeze_complete_highdim_leaderboard_source_snapshot.py"
    )
    runtime_helper = source / (
        "scripts/build_complete_highdim_leaderboard_runtime_fingerprint.py"
    )
    subprocess.run(
        [
            "/home/chakwong/anaconda3/envs/tf-gpu/bin/python",
            str(snapshot_helper),
            "verify",
            "--root",
            str(source),
            "--inventory",
            str(args.source_inventory),
            "--allow-populated-required-empty-directories",
        ],
        check=True,
        timeout=_remaining_timeout(args.started_monotonic, 1800, 600.0),
    )
    subprocess.run(
        [
            "/home/chakwong/anaconda3/envs/tf-gpu/bin/python",
            str(runtime_helper),
            "--output",
            str(args.runtime_fingerprint),
            "--check",
        ],
        check=True,
        timeout=_remaining_timeout(args.started_monotonic, 1800, 900.0),
    )
    _require_unexpired(
        args.approval_not_after_epoch, action="frozen snapshot materialization"
    )
    subprocess.run(
        [
            "/home/chakwong/anaconda3/envs/tf-gpu/bin/python",
            str(snapshot_helper),
            "materialize",
            "--source",
            str(source),
            "--destination",
            str(args.launch_root),
            "--inventory",
            str(args.source_inventory),
        ],
        check=True,
        timeout=_remaining_timeout(args.started_monotonic, 2100, 900.0),
    )
    if time.monotonic() - args.started_monotonic > 2100:
        raise RuntimeError("workspace copy exceeded the preparation budget")

    _install_snapshot_overlays(manifest, args.launch_root)

    for binding in manifest["repository_bound_files"]:
        copied = args.launch_root / binding["path"]
        if not copied.is_file() or _sha256(copied) != binding["sha256"]:
            raise RuntimeError(f"copied bound file drifted: {binding['path']}")

    sentinel = args.launch_root / ".complete_highdim_leaderboard_copy_sentinel.json"
    _write_exclusive(
        sentinel,
        {"run_id": args.run_id, "nonce": args.nonce, "launch_root": str(args.launch_root)},
    )
    baseline = args.handoff_dir / f"{args.run_id}-baseline-snapshot.json"
    exporter = args.launch_root / "scripts/export_complete_highdim_leaderboard_isolated_changes.py"
    status_writer = args.launch_root / "scripts/write_complete_highdim_leaderboard_terminal_status.py"
    trusted_exporter = args.handoff_dir / f"{args.run_id}-trusted-exporter.py"
    trusted_status_writer = args.handoff_dir / f"{args.run_id}-trusted-status-writer.py"
    review_verifier = args.launch_root / (
        "scripts/verify_complete_highdim_leaderboard_review_receipt.py"
    )
    trusted_review_verifier = args.handoff_dir / (
        f"{args.run_id}-trusted-review-verifier.py"
    )
    trusted_claude_gate = args.handoff_dir / (
        f"{args.run_id}-trusted-claude-review-gate.sh"
    )
    trusted_claude_worker = args.handoff_dir / (
        f"{args.run_id}-trusted-claude-worker.sh"
    )
    trusted_claude_settings = args.handoff_dir / (
        f"{args.run_id}-trusted-claude-worker-settings.json"
    )
    with exporter.open("rb") as source_stream, trusted_exporter.open("xb") as output_stream:
        output_stream.write(source_stream.read())
    with status_writer.open("rb") as source_stream, trusted_status_writer.open("xb") as output_stream:
        output_stream.write(source_stream.read())
    with review_verifier.open("rb") as source_stream, trusted_review_verifier.open(
        "xb"
    ) as output_stream:
        output_stream.write(source_stream.read())
    for source_path, destination in (
        (
            Path("/home/chakwong/python/claudecodex/scripts/claude_review_gate.sh"),
            trusted_claude_gate,
        ),
        (
            args.launch_root
            / "scripts/complete_highdim_leaderboard_claude_audit_worker.sh",
            trusted_claude_worker,
        ),
        (
            Path("/home/chakwong/.claude/settings.codex-worker.json"),
            trusted_claude_settings,
        ),
    ):
        with source_path.open("rb") as source_stream, destination.open("xb") as output_stream:
            output_stream.write(source_stream.read())
    support_dir = args.launch_root / CODEX_SUPPORT_DIR_NAME
    support_dir.mkdir(mode=0o700)
    support_sources = (
        (review_verifier, "trusted-review-verifier.py"),
        (
            Path("/home/chakwong/python/claudecodex/scripts/claude_review_gate.sh"),
            "trusted-claude-review-gate.sh",
        ),
        (
            args.launch_root
            / "scripts/complete_highdim_leaderboard_claude_audit_worker.sh",
            "trusted-claude-worker.sh",
        ),
        (
            Path("/home/chakwong/.claude/settings.codex-worker.json"),
            "trusted-claude-worker-settings.json",
        ),
    )
    support_bindings = []
    for source_path, name in support_sources:
        destination = support_dir / name
        _copy_exclusive(source_path, destination)
        support_bindings.append(
            {
                "path": name,
                "sha256": _sha256(destination),
                "size": destination.stat().st_size,
            }
        )
    expected_support = {
        record["destination_name"]: record["source"]["sha256"]
        for record in manifest["codex_readable_initial_disclosure"]["support_files"]
    }
    if set(expected_support) != {record["path"] for record in support_bindings}:
        raise RuntimeError("Codex support disclosure names are inconsistent")
    for record in support_bindings:
        if record["sha256"] != expected_support[record["path"]]:
            raise RuntimeError(f"Codex support disclosure drifted: {record['path']}")
    trusted_exporter_sha256 = _sha256(trusted_exporter)
    trusted_status_writer_sha256 = _sha256(trusted_status_writer)
    trusted_review_verifier_sha256 = _sha256(trusted_review_verifier)
    trusted_claude_gate_sha256 = _sha256(trusted_claude_gate)
    trusted_claude_worker_sha256 = _sha256(trusted_claude_worker)
    trusted_claude_settings_sha256 = _sha256(trusted_claude_settings)
    subprocess.run(
        [
            "/home/chakwong/anaconda3/envs/tf-gpu/bin/python",
            str(exporter),
            "snapshot",
            "--root",
            str(args.launch_root),
            "--root-identity",
            str(args.launch_root),
            "--output",
            str(baseline),
        ],
        check=True,
        timeout=_remaining_timeout(args.started_monotonic, 2100, 300.0),
    )
    preparation = args.handoff_dir / f"{args.run_id}-launch-preparation.json"
    _write_exclusive(
        preparation,
        {
            "schema_version": (
                "bayesfilter.complete_highdim_leaderboard.launch_preparation.v1"
            ),
            "run_id": args.run_id,
            "nonce": args.nonce,
            "source_root": str(source),
            "launch_root": str(args.launch_root),
            "handoff_dir": str(args.handoff_dir),
            "started_epoch": args.started_epoch,
            "started_monotonic": args.started_monotonic,
            "approval_not_after_epoch": args.approval_not_after_epoch,
            "baseline": str(baseline),
            "manifest": str(args.manifest),
            "manifest_sha256": args.expected_manifest_sha256,
            "snapshot_overlay_files": manifest["snapshot_overlay_files"],
            "owner_accepted_run_scoped_limitations": limitations,
            "mandatory_post_run_integrity_audit": post_run_audit,
            "completion_or_release_authority_granted": False,
            "codex_support_dir_name": CODEX_SUPPORT_DIR_NAME,
            "codex_support_files": support_bindings,
            "dynamic_handoff_hidden_from_codex": True,
            "trusted_exporter": str(trusted_exporter),
            "trusted_exporter_sha256": trusted_exporter_sha256,
            "trusted_status_writer": str(trusted_status_writer),
            "trusted_status_writer_sha256": trusted_status_writer_sha256,
            "trusted_review_verifier": str(trusted_review_verifier),
            "trusted_review_verifier_sha256": trusted_review_verifier_sha256,
            "trusted_claude_gate": str(trusted_claude_gate),
            "trusted_claude_gate_sha256": trusted_claude_gate_sha256,
            "trusted_claude_worker": str(trusted_claude_worker),
            "trusted_claude_worker_sha256": trusted_claude_worker_sha256,
            "trusted_claude_settings": str(trusted_claude_settings),
            "trusted_claude_settings_sha256": trusted_claude_settings_sha256,
        },
    )
    watchdog_log = args.handoff_dir / f"{args.run_id}-watchdog.log"
    watchdog_pid = args.handoff_dir / f"{args.run_id}-watchdog.pid"
    watchdog = source / "scripts/complete_highdim_leaderboard_watchdog.py"
    watchdog_timeout = (
        args.started_monotonic
        + args.watchdog_verification_deadline
        + 30
        - time.monotonic()
    )
    if watchdog_timeout <= 0.0:
        raise TimeoutError("watchdog launch budget is exhausted")
    _require_unexpired(
        args.approval_not_after_epoch, action="detached supervisor handoff"
    )
    with watchdog_log.open("xb") as log_stream:
        process = subprocess.Popen(
            [
                "/usr/bin/timeout",
                "--signal=TERM",
                "--kill-after=10s",
                f"{watchdog_timeout:.6f}s",
                "/home/chakwong/anaconda3/envs/tf-gpu/bin/python",
                str(watchdog),
                "--run-id",
                args.run_id,
                "--handoff-dir",
                str(args.handoff_dir),
                "--started-epoch",
                str(args.started_epoch),
                "--started-monotonic",
                str(args.started_monotonic),
                "--verification-deadline",
                str(args.watchdog_verification_deadline),
            ],
            stdin=subprocess.DEVNULL,
            stdout=log_stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    with watchdog_pid.open("x", encoding="utf-8") as stream:
        stream.write(f"{process.pid}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
