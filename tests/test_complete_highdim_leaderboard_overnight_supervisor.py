from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SUPERVISOR = REPO / "scripts/complete_highdim_leaderboard_overnight_supervisor.py"
EXPORTER = REPO / "scripts/export_complete_highdim_leaderboard_isolated_changes.py"
STATUS_WRITER = REPO / "scripts/write_complete_highdim_leaderboard_terminal_status.py"
REVIEW_VERIFIER = REPO / "scripts/verify_complete_highdim_leaderboard_review_receipt.py"
CLAUDE_GATE = Path("/home/chakwong/python/claudecodex/scripts/claude_review_gate.sh")
CLAUDE_WORKER = REPO / "scripts/complete_highdim_leaderboard_claude_audit_worker.sh"
CLAUDE_SETTINGS = Path("/home/chakwong/.claude/settings.codex-worker.json")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _workspace(
    tmp_path: Path,
    fake_body: str,
    *,
    soft_deadline: int = 10,
    termination_deadline: int = 12,
    hard_deadline: int = 20,
) -> tuple[Path, Path, dict[str, str]]:
    launch_root = tmp_path / "launch-copy"
    root_alias = tmp_path / "mounted-source-path"
    run_id = "test-run"
    handoff = root_alias / "docs/plans/logs" / run_id
    launch_root.mkdir()
    root_alias.symlink_to(launch_root, target_is_directory=True)
    (launch_root / "docs/plans").mkdir(parents=True)
    handoff.mkdir(parents=True)
    (launch_root / "scripts").mkdir()
    shutil.copy2(EXPORTER, launch_root / "scripts" / EXPORTER.name)
    shutil.copy2(STATUS_WRITER, launch_root / "scripts" / STATUS_WRITER.name)
    prompt = launch_root / (
        "docs/plans/bayesfilter-complete-highdim-leaderboard-"
        "overnight-supervisor-prompt-2026-07-11.md"
    )
    prompt.write_text("bounded test prompt\n", encoding="utf-8")
    phase1 = launch_root / (
        "docs/plans/bayesfilter-complete-highdim-leaderboard-"
        "phase1-ledh-harness-subplan-2026-07-11.md"
    )
    phase1.write_text("reviewed phase 1\n", encoding="utf-8")
    phase1_receipt = launch_root / (
        "docs/reviews/bayesfilter-complete-highdim-leaderboard-"
        "phase1-subplan-review-receipt-2026-07-11.json"
    )
    phase1_receipt.parent.mkdir(parents=True)
    phase1_receipt.write_text(
        json.dumps(
            {
                "schema_version": (
                    "bayesfilter.complete_highdim_leaderboard.review_receipt.v1"
                ),
                "reviewed_path": phase1.relative_to(launch_root).as_posix(),
                "reviewed_sha256": _sha256(phase1),
                "reviewer_type": "fresh_codex_readonly_substitute",
                "iteration": 1,
                "verdict": "AGREE",
            }
        ),
        encoding="utf-8",
    )
    (launch_root / "before.txt").write_text("before\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(launch_root)], check=True)

    nonce = "test-nonce"
    sentinel = launch_root / ".complete_highdim_leaderboard_copy_sentinel.json"
    sentinel.write_text(
        json.dumps(
            {"run_id": run_id, "nonce": nonce, "launch_root": str(launch_root)}
        )
        + "\n",
        encoding="utf-8",
    )
    baseline = handoff / f"{run_id}-baseline-snapshot.json"
    subprocess.run(
        [
            os.environ.get("PYTHON", "python"),
            str(EXPORTER),
            "snapshot",
            "--root",
            str(launch_root),
            "--root-identity",
            str(launch_root),
            "--output",
            str(baseline),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    preparation = handoff / f"{run_id}-launch-preparation.json"
    trusted_exporter = handoff / f"{run_id}-trusted-exporter.py"
    trusted_status_writer = handoff / f"{run_id}-trusted-status-writer.py"
    trusted_review_verifier = handoff / f"{run_id}-trusted-review-verifier.py"
    trusted_claude_gate = handoff / f"{run_id}-trusted-claude-review-gate.sh"
    trusted_claude_worker = handoff / f"{run_id}-trusted-claude-worker.sh"
    trusted_claude_settings = handoff / f"{run_id}-trusted-claude-worker-settings.json"
    shutil.copy2(EXPORTER, trusted_exporter)
    shutil.copy2(STATUS_WRITER, trusted_status_writer)
    shutil.copy2(REVIEW_VERIFIER, trusted_review_verifier)
    shutil.copy2(CLAUDE_GATE, trusted_claude_gate)
    shutil.copy2(CLAUDE_WORKER, trusted_claude_worker)
    shutil.copy2(CLAUDE_SETTINGS, trusted_claude_settings)
    preparation.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "nonce": nonce,
                "launch_root": str(launch_root),
                "started_epoch": time.time(),
                "started_monotonic": time.monotonic(),
                "codex_support_dir_name": ".complete_highdim_codex_support",
                "codex_support_files": [],
                "dynamic_handoff_hidden_from_codex": True,
                "trusted_exporter": str(trusted_exporter),
                "trusted_exporter_sha256": _sha256(trusted_exporter),
                "trusted_status_writer": str(trusted_status_writer),
                "trusted_status_writer_sha256": _sha256(trusted_status_writer),
                "trusted_review_verifier": str(trusted_review_verifier),
                "trusted_review_verifier_sha256": _sha256(trusted_review_verifier),
                "trusted_claude_gate": str(trusted_claude_gate),
                "trusted_claude_gate_sha256": _sha256(trusted_claude_gate),
                "trusted_claude_worker": str(trusted_claude_worker),
                "trusted_claude_worker_sha256": _sha256(trusted_claude_worker),
                "trusted_claude_settings": str(trusted_claude_settings),
                "trusted_claude_settings_sha256": _sha256(trusted_claude_settings),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    boundary = handoff / f"{run_id}-namespace-boundary.json"
    boundary.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "nonce": nonce,
                "root_mount_matches_launch_copy": True,
                "home_tree_read_only": True,
                "mounted_host_drives_hidden_by_private_read_only_tmpfs": True,
                "private_tmpfs": True,
                "codex_child_handoff_read_only": True,
                "codex_child_private_pid_namespace": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    fake = tmp_path / "fake-codex"
    fake.write_text("#!/usr/bin/env bash\nset -eu\n" + fake_body, encoding="utf-8")
    fake.chmod(0o755)
    support_dir = launch_root / ".complete_highdim_codex_support"
    support_dir.mkdir()
    support_sources = (
        (trusted_review_verifier, "trusted-review-verifier.py"),
        (trusted_claude_gate, "trusted-claude-review-gate.sh"),
        (trusted_claude_worker, "trusted-claude-worker.sh"),
        (trusted_claude_settings, "trusted-claude-worker-settings.json"),
    )
    support_bindings = []
    for source, name in support_sources:
        destination = support_dir / name
        shutil.copy2(source, destination)
        support_bindings.append(
            {"path": name, "sha256": _sha256(destination), "size": destination.stat().st_size}
        )
    preparation_payload = json.loads(preparation.read_text(encoding="utf-8"))
    preparation_payload["codex_support_files"] = support_bindings
    preparation.write_text(json.dumps(preparation_payload) + "\n", encoding="utf-8")
    baseline.unlink()
    subprocess.run(
        [
            os.environ.get("PYTHON", "python"),
            str(EXPORTER),
            "snapshot",
            "--root",
            str(launch_root),
            "--root-identity",
            str(launch_root),
            "--output",
            str(baseline),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    started_epoch = time.time()
    started_monotonic = time.monotonic()
    env = {
        **os.environ,
        "ROOT": str(root_alias),
        "SOURCE_ROOT": str(root_alias),
        "LAUNCH_ROOT": str(launch_root),
        "RUN_ID": run_id,
        "OUTER_HANDOFF_DIR": str(handoff),
        "COPY_SENTINEL_NONCE": nonce,
        "LAUNCH_STARTED_EPOCH": str(started_epoch),
        "LAUNCH_STARTED_MONOTONIC": str(started_monotonic),
        "CODEX_SOFT_DEADLINE_SECONDS": str(soft_deadline),
        "PROCESS_TERMINATION_DEADLINE_SECONDS": str(termination_deadline),
        "HARD_DEADLINE_SECONDS": str(hard_deadline),
        "EXPORT_TIMEOUT_SECONDS": "10",
        "CODEX_BIN": str(fake),
        "SUPERVISOR_TEST_ALLOW_DIRECT_CODEX": "1",
    }
    return launch_root, handoff, env


def _run(env: dict[str, str], *, timeout: int = 30) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [os.environ.get("PYTHON", "python"), str(SUPERVISOR)],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=timeout,
    )


def test_supervisor_exports_successful_isolated_changes(tmp_path: Path) -> None:
    root, handoff, env = _workspace(
        tmp_path,
        'printf \'new\\n\' > "$ROOT/added.txt"\nexit 0\n',
    )

    result = _run(env)

    assert result.returncode == 0, result.stderr
    terminal = json.loads(
        (handoff / "test-run-terminal-status.json").read_text(encoding="utf-8")
    )
    assert terminal["supervisor_process_status"] == "codex_exit_zero"
    assert terminal["automatic_merge_performed"] is False
    manifest = json.loads(
        (handoff / "test-run-primary-isolated-change-manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["added"] == ["added.txt"]
    assert (root / "added.txt").read_text(encoding="utf-8") == "new\n"
    hashes_path = handoff / "test-run-primary-export-sha256.json"
    hashes = json.loads(hashes_path.read_text(encoding="utf-8"))
    for entry in hashes["files"]:
        exported = Path(entry["path"])
        assert exported.is_file()
        assert exported.stat().st_size == entry["size"]
        assert _sha256(exported) == entry["sha256"]


def test_supervisor_exports_after_codex_failure(tmp_path: Path) -> None:
    _root, handoff, env = _workspace(
        tmp_path,
        'printf \'partial\\n\' > "$ROOT/partial.txt"\nexit 23\n',
    )

    result = _run(env)

    assert result.returncode == 23
    terminal = json.loads(
        (handoff / "test-run-terminal-status.json").read_text(encoding="utf-8")
    )
    assert terminal["supervisor_process_status"] == "codex_failed"
    assert terminal["codex_exit_code"] == 23
    manifest = json.loads(
        (handoff / "test-run-primary-isolated-change-manifest.json").read_text()
    )
    assert manifest["added"] == ["partial.txt"]


def test_supervisor_exports_after_soft_timeout(tmp_path: Path) -> None:
    _root, handoff, env = _workspace(
        tmp_path,
        'printf \'partial\\n\' > "$ROOT/partial-timeout.txt"\nsleep 20\n',
        soft_deadline=1,
        termination_deadline=3,
        hard_deadline=10,
    )

    result = _run(env, timeout=15)

    assert result.returncode == 124
    terminal = json.loads(
        (handoff / "test-run-terminal-status.json").read_text(encoding="utf-8")
    )
    assert terminal["supervisor_process_status"] == "timed_out"
    assert terminal["timed_out"] is True
    manifest = json.loads(
        (handoff / "test-run-primary-isolated-change-manifest.json").read_text()
    )
    assert manifest["added"] == ["partial-timeout.txt"]


def test_supervisor_rejects_source_root_without_mount_alias(tmp_path: Path) -> None:
    root, _handoff, env = _workspace(tmp_path, "exit 0\n")
    env["ROOT"] = str(root)
    env["SOURCE_ROOT"] = str(root)

    result = _run(env)

    assert result.returncode == 70
    assert "mount namespace path contract is invalid" in result.stderr


def test_supervisor_rejects_wrong_copy_nonce(tmp_path: Path) -> None:
    _root, _handoff, env = _workspace(tmp_path, "exit 0\n")
    env["COPY_SENTINEL_NONCE"] = "forged"

    result = _run(env)

    assert result.returncode == 70
    assert "copy sentinel" in result.stderr


def test_supervisor_rejects_direct_codex_bypass_outside_pytest_identity(
    tmp_path: Path,
) -> None:
    _root, _handoff, env = _workspace(tmp_path, "exit 0\n")
    env.pop("PYTEST_CURRENT_TEST", None)

    result = _run(env)

    assert result.returncode == 70
    assert "direct Codex bypass is restricted" in result.stderr
