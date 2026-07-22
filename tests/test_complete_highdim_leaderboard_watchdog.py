from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

from scripts import complete_highdim_leaderboard_watchdog as watchdog


RUN_ID = "test-watchdog-run"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_primary(handoff: Path, *, tamper: bool = False) -> None:
    closed = handoff / f"{RUN_ID}-codex-namespace-closed.json"
    closed.write_text(
        json.dumps(
            {
                "schema_version": (
                    "bayesfilter.complete_highdim_leaderboard."
                    "codex_namespace_closed.v1"
                ),
                "run_id": RUN_ID,
                "unshare_process_returned": True,
                "unshare_process_group_absent": True,
                "private_pid_namespace_init_required_pid_one": True,
                "whole_private_pid_namespace_quiescent": True,
                "quiescence_proof": (
                    "linux_pid_namespace_init_exit_kills_all_members_and_"
                    "unshare_returned"
                ),
                "untrusted_code_started_only_after_capability_drop": True,
                "namespace_escape_available_to_untrusted_code": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    exported = handoff / f"{RUN_ID}-primary-isolated-change-manifest.json"
    exported.write_text("{}\n", encoding="utf-8")
    hashes = handoff / f"{RUN_ID}-primary-export-sha256.json"
    hashes.write_text(
        json.dumps(
            {
                "schema_version": (
                    "bayesfilter.complete_highdim_leaderboard.export_hashes.v1"
                ),
                "run_id": RUN_ID,
                "export_label": "primary",
                "files": [
                    {
                        "path": str(exported),
                        "size": exported.stat().st_size,
                        "sha256": _sha256(exported),
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    verification = handoff / f"{RUN_ID}-post-export-verification.json"
    verification.write_text(
        json.dumps(
            {
                "schema_version": (
                    "bayesfilter.complete_highdim_leaderboard."
                    "post_export_verification.v1"
                ),
                "run_id": RUN_ID,
                "export_label": "primary",
                "export_hashes": str(hashes),
                "export_hashes_sha256": "0" * 64 if tamper else _sha256(hashes),
                "all_bound_files_recomputed": True,
                "codex_namespace_closed_receipt": str(closed),
                "codex_namespace_closed_receipt_sha256": _sha256(closed),
                "verified_after_codex_namespace_closed": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _args(handoff: Path, *, elapsed: float = 0.0) -> list[str]:
    return [
        "--run-id",
        RUN_ID,
        "--handoff-dir",
        str(handoff),
        "--started-epoch",
        str(time.time() - elapsed),
        "--started-monotonic",
        str(time.monotonic() - elapsed),
        "--verification-deadline",
        "1",
    ]


def test_watchdog_validates_primary_export_without_copy_access(tmp_path: Path) -> None:
    handoff = tmp_path / "handoff"
    handoff.mkdir()
    _write_primary(handoff)

    result = watchdog.main(_args(handoff))

    assert result == 0
    status = json.loads(
        (handoff / f"{RUN_ID}-watchdog-status.json").read_text(encoding="utf-8")
    )
    assert status["primary_export_verified"] is True
    assert status["primary_export_only"] is True
    assert status["primary_export_only_observed"] is True
    assert status["unapproved_export_artifacts"] == []
    assert status["fallback_export_allowed"] is False
    assert status["writable_copy_read_by_watchdog"] is False
    assert status["writable_copy_written_by_watchdog"] is False
    assert not any("fallback" in path.name for path in handoff.iterdir())


def test_watchdog_fails_closed_without_verified_primary_export(tmp_path: Path) -> None:
    handoff = tmp_path / "handoff"
    handoff.mkdir()

    result = watchdog.main(_args(handoff, elapsed=2.0))

    assert result == 95
    status = json.loads(
        (handoff / f"{RUN_ID}-watchdog-status.json").read_text(encoding="utf-8")
    )
    assert status["primary_export_verified"] is False
    assert status["fallback_export_allowed"] is False


def test_watchdog_rejects_tampered_primary_verification(tmp_path: Path) -> None:
    handoff = tmp_path / "handoff"
    handoff.mkdir()
    _write_primary(handoff, tamper=True)

    result = watchdog.main(_args(handoff, elapsed=2.0))

    assert result == 95


def test_watchdog_rejects_unapproved_fallback_export(tmp_path: Path) -> None:
    handoff = tmp_path / "handoff"
    handoff.mkdir()
    _write_primary(handoff)
    fallback = handoff / f"{RUN_ID}-fallback-isolated-change-manifest.json"
    fallback.write_text(
        json.dumps(
            {
                "schema_version": (
                    "bayesfilter.complete_highdim_leaderboard.isolated_export.v1"
                ),
                "run_id": RUN_ID,
                "export_label": "fallback",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = watchdog.main(_args(handoff, elapsed=2.0))

    assert result == 95
    status = json.loads(
        (handoff / f"{RUN_ID}-watchdog-status.json").read_text(encoding="utf-8")
    )
    assert status["primary_export_only_observed"] is False
    assert status["unapproved_export_artifacts"] == [fallback.name]


def test_watchdog_descriptor_binds_live_process_identity(tmp_path: Path) -> None:
    handoff = tmp_path / "handoff"
    handoff.mkdir()

    watchdog.main(_args(handoff, elapsed=2.0))

    descriptor = json.loads(
        (handoff / f"{RUN_ID}-watchdog-producer.json").read_text(encoding="utf-8")
    )
    assert descriptor["pid"] == os.getpid()
    assert descriptor["pgid"] == os.getpgid(0)
    assert descriptor["sid"] == os.getsid(0)
    assert descriptor["start_time_ticks"] > 0
    assert descriptor["role"] == "primary_export_verification_watchdog"
