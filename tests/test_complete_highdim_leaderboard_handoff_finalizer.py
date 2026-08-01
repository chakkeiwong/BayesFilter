from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

from scripts import finalize_complete_highdim_leaderboard_handoff as finalizer


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _process_identity(pid: int) -> tuple[int, int, int]:
    value = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    tail = value[value.rfind(")") + 2 :].split()
    return int(tail[2]), int(tail[3]), int(tail[19])


def _descriptor(path: Path, run_id: str, role: str, process: subprocess.Popen) -> None:
    pgid, sid, start = _process_identity(process.pid)
    command = finalizer.EXPECTED_PRODUCER_COMMANDS[role]
    path.write_text(
        json.dumps(
            {
                "schema_version": (
                    "bayesfilter.complete_highdim_leaderboard.producer_descriptor.v1"
                ),
                "run_id": run_id,
                "role": role,
                "pid": process.pid,
                "pgid": pgid,
                "sid": sid,
                "start_time_ticks": start,
                "pid_namespace_inode": os.stat("/proc/self/ns/pid").st_ino,
                "command_path": str(command["execution_path"]),
                "command_sha256": _sha256(command["hash_source"]),
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _closed_descriptors() -> list[dict]:
    return [
        {
            "role": role,
            "pid": index + 1,
            "pgid": index + 1,
            "sid": index + 1,
            "start_time_ticks": index + 1,
            "pid_namespace_inode": os.stat("/proc/self/ns/pid").st_ino,
        }
        for index, role in enumerate(finalizer.EXPECTED_PRODUCER_COMMANDS)
    ]


def test_final_seal_hashes_closed_run_files(tmp_path: Path) -> None:
    run_id = "test-run"
    first = tmp_path / f"{run_id}.log"
    second = tmp_path / f"{run_id}-result.json"
    first.write_text("closed log\n", encoding="utf-8")
    second.write_text("{}\n", encoding="utf-8")

    payload = finalizer.seal(
        tmp_path,
        run_id,
        launcher_exit_code=0,
        approval_instance_id="approval",
        producer_descriptors=_closed_descriptors(),
    )

    assert payload["sealed_after_all_producers_closed"] is True
    assert {entry["sha256"] for entry in payload["files"]} == {
        _sha256(first),
        _sha256(second),
    }


def test_final_seal_rejects_unscoped_entry(tmp_path: Path) -> None:
    (tmp_path / "other.txt").write_text("not scoped\n", encoding="utf-8")

    try:
        finalizer.seal(
            tmp_path,
            "test-run",
            launcher_exit_code=0,
            approval_instance_id="approval",
            producer_descriptors=_closed_descriptors(),
        )
    except ValueError as error:
        assert "unscoped" in str(error)
    else:
        raise AssertionError("unscoped handoff file was accepted")


def test_final_seal_requires_completion_inputs_and_excludes_itself(tmp_path: Path) -> None:
    run_id = "test-run"
    bound = tmp_path / f"{run_id}-result.json"
    bound.write_text("{}\n", encoding="utf-8")
    hashes = tmp_path / f"{run_id}-primary-export-sha256.json"
    hashes.write_text(
        json.dumps(
            {
                "schema_version": (
                    "bayesfilter.complete_highdim_leaderboard.export_hashes.v1"
                ),
                "run_id": run_id,
                "export_label": "primary",
                "files": [
                    {
                        "path": str(bound),
                        "sha256": _sha256(bound),
                        "size": bound.stat().st_size,
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    closed = tmp_path / f"{run_id}-codex-namespace-closed.json"
    closed.write_text(
        json.dumps(
            {
                "schema_version": (
                    "bayesfilter.complete_highdim_leaderboard."
                    "codex_namespace_closed.v1"
                ),
                "run_id": run_id,
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
    verification = tmp_path / f"{run_id}-post-export-verification.json"
    outcome = tmp_path / f"{run_id}-foreground-outcome.json"
    verification.write_text(
        json.dumps(
            {
                "schema_version": (
                    "bayesfilter.complete_highdim_leaderboard."
                    "post_export_verification.v1"
                ),
                "run_id": run_id,
                "export_label": "primary",
                "export_hashes": str(hashes),
                "export_hashes_sha256": _sha256(hashes),
                "all_bound_files_recomputed": True,
                "codex_namespace_closed_receipt": str(closed),
                "codex_namespace_closed_receipt_sha256": _sha256(closed),
                "verified_after_codex_namespace_closed": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    outcome.write_text(
        json.dumps(
            {
                "schema_version": (
                    "bayesfilter.complete_highdim_leaderboard.foreground_outcome.v2"
                ),
                "run_id": run_id,
                "producers_closed_before_outcome": True,
                "producer_descriptors_valid": True,
                "watchdog_primary_verification_passed": True,
                "post_export_verification_present": True,
                "primary_export_present": True,
                "fallback_export_allowed": False,
                "fallback_export_present": False,
                "primary_export_only_observed": True,
                "unapproved_export_artifacts": [],
                "whole_outer_pid_namespace_quiescent_before_outcome": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    payload = finalizer.seal(
        tmp_path,
        run_id,
        launcher_exit_code=0,
        approval_instance_id="approval",
        required_files=[verification, outcome],
        producer_descriptors=_closed_descriptors(),
    )

    seal_path = tmp_path / f"{run_id}-final-handoff-seal.json"
    assert payload["seal_file_excluded_from_its_own_hash_ledger"] is True
    assert str(seal_path) not in {entry["path"] for entry in payload["files"]}
    assert json.loads(seal_path.read_text())["post_seal_writes_forbidden"] is True


def test_final_seal_rejects_missing_completion_input(tmp_path: Path) -> None:
    try:
        finalizer.seal(
            tmp_path,
            "test-run",
            launcher_exit_code=0,
            approval_instance_id="approval",
            required_files=[tmp_path / "test-run-post-export-verification.json"],
            producer_descriptors=_closed_descriptors(),
        )
    except ValueError as error:
        assert "required seal input" in str(error)
    else:
        raise AssertionError("missing completion verification was accepted")


def test_final_seal_rejects_forged_post_export_verification(tmp_path: Path) -> None:
    run_id = "test-run"
    verification = tmp_path / f"{run_id}-post-export-verification.json"
    verification.write_text(
        json.dumps(
            {
                "schema_version": (
                    "bayesfilter.complete_highdim_leaderboard."
                    "post_export_verification.v1"
                ),
                "run_id": run_id,
                "export_label": "primary",
                "export_hashes": str(tmp_path / "missing.json"),
                "export_hashes_sha256": "0" * 64,
                "all_bound_files_recomputed": True,
                "verified_after_codex_namespace_closed": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    try:
        finalizer.seal(
            tmp_path,
            run_id,
            launcher_exit_code=0,
            approval_instance_id="approval",
            required_files=[verification],
            producer_descriptors=_closed_descriptors(),
        )
    except ValueError as error:
        assert "post-export hash ledger" in str(error)
    else:
        raise AssertionError("forged post-export verification was accepted")


def test_wait_for_producer_observes_process_exit(tmp_path: Path) -> None:
    run_id = "test-run"
    processes = [
        subprocess.Popen(["/usr/bin/sleep", "0.1"], start_new_session=True),
        subprocess.Popen(["/usr/bin/sleep", "0.1"], start_new_session=True),
    ]
    descriptors = [
        tmp_path / f"{run_id}-supervisor-producer.json",
        tmp_path / f"{run_id}-watchdog-producer.json",
    ]
    _descriptor(descriptors[0], run_id, "detached_supervisor", processes[0])
    _descriptor(
        descriptors[1],
        run_id,
        "primary_export_verification_watchdog",
        processes[1],
    )

    observed = finalizer.wait_for_producers(
        descriptors,
        handoff_dir=tmp_path,
        run_id=run_id,
        deadline_monotonic=time.monotonic() + 5,
    )

    for process in processes:
        process.wait(timeout=1)
        assert process.returncode == 0
    assert {item["role"] for item in observed} == {
        "detached_supervisor",
        "primary_export_verification_watchdog",
    }


def test_wait_for_producers_rejects_missing_descriptor(tmp_path: Path) -> None:
    try:
        finalizer.wait_for_producers(
            [tmp_path / "missing.json"],
            handoff_dir=tmp_path,
            run_id="test-run",
            deadline_monotonic=time.monotonic() + 1,
        )
    except ValueError as error:
        assert "exactly two" in str(error)
    else:
        raise AssertionError("missing producer descriptor was accepted")


def test_whole_namespace_quiescence_rejects_setsid_descendant() -> None:
    process = subprocess.Popen(
        ["/usr/bin/sleep", "5"], start_new_session=True
    )
    try:
        assert process.pid in finalizer._extra_live_namespace_pids()  # noqa: SLF001
        assert finalizer._whole_private_namespace_quiescent() is False  # noqa: SLF001
    finally:
        process.terminate()
        process.wait(timeout=1)


def test_final_seal_rejects_unapproved_fallback_export(tmp_path: Path) -> None:
    run_id = "test-run"
    fallback = tmp_path / f"{run_id}-fallback-isolated-change-manifest.json"
    fallback.write_text(
        json.dumps(
            {
                "schema_version": (
                    "bayesfilter.complete_highdim_leaderboard.isolated_export.v1"
                ),
                "run_id": run_id,
                "export_label": "fallback",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    try:
        finalizer.seal(
            tmp_path,
            run_id,
            launcher_exit_code=0,
            approval_instance_id="approval",
            producer_descriptors=_closed_descriptors(),
        )
    except ValueError as error:
        assert "unapproved export artifacts" in str(error)
    else:
        raise AssertionError("fallback export artifact was accepted")
