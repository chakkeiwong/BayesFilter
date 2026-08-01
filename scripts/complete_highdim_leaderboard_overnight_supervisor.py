#!/usr/bin/env python3
"""Run the isolated Codex supervisor and export before the hard deadline."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import signal
import stat
import subprocess
import sys
import time
from pathlib import Path


def _required(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is required")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _safe_handoff_dir(path: Path) -> None:
    if path.is_symlink() or not path.is_dir():
        raise RuntimeError("handoff directory must be a real directory")
    for child in path.iterdir():
        if child.is_symlink() or not child.is_file() or child.stat().st_nlink != 1:
            raise RuntimeError(f"unsafe handoff entry: {child}")


def _write_exclusive(path: Path, value: str) -> None:
    with path.open("x", encoding="utf-8") as stream:
        stream.write(value)


def _write_json_exclusive(path: Path, payload: dict) -> None:
    _write_exclusive(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _proc_start_time_ticks(pid: int) -> int:
    fields = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()
    if len(fields) < 22:
        raise RuntimeError(f"process identity is incomplete for PID {pid}")
    return int(fields[21])


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


def _wait_process_group_absent(pgid: int, *, deadline_monotonic: float) -> bool:
    while _process_group_alive(pgid) and time.monotonic() < deadline_monotonic:
        time.sleep(min(0.1, max(0.01, deadline_monotonic - time.monotonic())))
    return not _process_group_alive(pgid)


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


def _child_environment(*, support_dir: Path, support_bindings: list[dict]) -> dict[str, str]:
    required = {
        "ROOT",
        "SOURCE_ROOT",
        "LAUNCH_ROOT",
        "RUN_ID",
        "OUTER_HANDOFF_DIR",
        "OUTER_LOG_DIR",
        "COPY_SENTINEL_NONCE",
        "PATH",
        "CONDA_PREFIX",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "TERM",
        "CUDA_VISIBLE_DEVICES",
        "TF_CPP_MIN_LOG_LEVEL",
        "TF_ENABLE_ONEDNN_OPTS",
        "XLA_FLAGS",
        "LD_LIBRARY_PATH",
        "SSL_CERT_FILE",
        "REQUESTS_CA_BUNDLE",
        "HTTPS_PROXY",
        "HTTP_PROXY",
        "NO_PROXY",
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_BASE_URL",
        "CODEX_SANDBOX_NETWORK_DISABLED",
    }
    environment = {key: value for key, value in os.environ.items() if key in required}
    environment["CODEX_SUPPORT_DIR"] = str(support_dir)
    environment["CODEX_SUPPORT_BINDINGS_JSON"] = json.dumps(
        support_bindings, sort_keys=True, separators=(",", ":")
    )
    return environment


def _terminate_group(process: subprocess.Popen[bytes], grace_seconds: int) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=max(0, grace_seconds))
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=10)


def main() -> int:
    root = Path(_required("ROOT"))
    source_root = Path(_required("SOURCE_ROOT"))
    launch_root = Path(_required("LAUNCH_ROOT"))
    run_id = _required("RUN_ID")
    handoff_dir = Path(os.environ.get("OUTER_HANDOFF_DIR") or _required("OUTER_LOG_DIR"))
    nonce = _required("COPY_SENTINEL_NONCE")
    started_epoch_value = os.environ.get("LAUNCH_STARTED_EPOCH")
    started_monotonic_value = os.environ.get("LAUNCH_STARTED_MONOTONIC")
    soft_deadline = int(os.environ.get("CODEX_SOFT_DEADLINE_SECONDS", "26400"))
    termination_deadline = int(
        os.environ.get("PROCESS_TERMINATION_DEADLINE_SECONDS", "26700")
    )
    hard_deadline = int(os.environ.get("HARD_DEADLINE_SECONDS", "28200"))
    codex_bin = os.environ.get(
        "CODEX_BIN", "/home/chakwong/.nvm/versions/node/v22.23.1/bin/codex"
    )
    export_timeout = int(os.environ.get("EXPORT_TIMEOUT_SECONDS", "600"))
    if not (0 < soft_deadline < termination_deadline < hard_deadline):
        raise RuntimeError("deadline ordering is invalid")
    if root == launch_root or source_root != root:
        raise RuntimeError("mount namespace path contract is invalid")
    expected_handoff = source_root / "docs/plans/logs" / run_id
    if handoff_dir != expected_handoff:
        raise RuntimeError("handoff path is not the approved per-run directory")
    _safe_handoff_dir(handoff_dir)

    sentinel = root / ".complete_highdim_leaderboard_copy_sentinel.json"
    preparation = handoff_dir / f"{run_id}-launch-preparation.json"
    boundary = handoff_dir / f"{run_id}-namespace-boundary.json"
    baseline = handoff_dir / f"{run_id}-baseline-snapshot.json"
    sentinel_payload = json.loads(sentinel.read_text(encoding="utf-8"))
    preparation_payload = json.loads(preparation.read_text(encoding="utf-8"))
    boundary_payload = json.loads(boundary.read_text(encoding="utf-8"))
    if (
        sentinel_payload.get("run_id") != run_id
        or sentinel_payload.get("nonce") != nonce
        or preparation_payload.get("run_id") != run_id
        or preparation_payload.get("nonce") != nonce
        or preparation_payload.get("launch_root") != str(launch_root)
        or boundary_payload.get("run_id") != run_id
        or boundary_payload.get("nonce") != nonce
        or boundary_payload.get("root_mount_matches_launch_copy") is not True
        or boundary_payload.get("home_tree_read_only") is not True
        or boundary_payload.get(
            "mounted_host_drives_hidden_by_private_read_only_tmpfs"
        )
        is not True
        or boundary_payload.get("private_tmpfs") is not True
        or boundary_payload.get("codex_child_handoff_read_only") is not True
        or boundary_payload.get("codex_child_private_pid_namespace") is not True
        or not baseline.is_file()
    ):
        raise RuntimeError("copy sentinel or launch preparation is invalid")
    started_epoch = float(
        started_epoch_value
        if started_epoch_value is not None
        else preparation_payload["started_epoch"]
    )
    started_monotonic = float(
        started_monotonic_value
        if started_monotonic_value is not None
        else preparation_payload["started_monotonic"]
    )

    prompt = root / (
        "docs/plans/bayesfilter-complete-highdim-leaderboard-"
        "overnight-supervisor-prompt-2026-07-11.md"
    )
    support_dir = root / preparation_payload["codex_support_dir_name"]
    if support_dir.is_symlink() or not support_dir.is_dir():
        raise RuntimeError("Codex support directory is missing or unsafe")
    support_paths = {}
    support_bindings = preparation_payload["codex_support_files"]
    if not isinstance(support_bindings, list):
        raise RuntimeError("Codex support bindings are invalid")
    for binding in support_bindings:
        path = support_dir / binding["path"]
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_nlink != 1
            or path.stat().st_size != binding["size"]
            or _sha256(path) != binding["sha256"]
        ):
            raise RuntimeError(f"Codex support file drifted: {path}")
        support_paths[binding["path"]] = path
    expected_support_names = {
        "trusted-review-verifier.py",
        "trusted-claude-review-gate.sh",
        "trusted-claude-worker.sh",
        "trusted-claude-worker-settings.json",
    }
    if set(support_paths) != expected_support_names or {
        path.name for path in support_dir.iterdir()
    } != expected_support_names:
        raise RuntimeError("Codex support allowlist is not exact")
    exporter = Path(preparation_payload["trusted_exporter"])
    status_writer = Path(preparation_payload["trusted_status_writer"])
    review_verifier = support_paths["trusted-review-verifier.py"]
    claude_gate = support_paths["trusted-claude-review-gate.sh"]
    claude_worker = support_paths["trusted-claude-worker.sh"]
    claude_settings = support_paths["trusted-claude-worker-settings.json"]
    for path in (
        prompt,
        exporter,
        status_writer,
        review_verifier,
        claude_gate,
        claude_worker,
        claude_settings,
    ):
        if not path.is_file():
            raise RuntimeError(f"required supervisor artifact is missing: {path}")
    if (
        _sha256(exporter) != preparation_payload["trusted_exporter_sha256"]
        or _sha256(status_writer)
        != preparation_payload["trusted_status_writer_sha256"]
        or _sha256(claude_worker)
        != preparation_payload["trusted_claude_worker_sha256"]
        or preparation_payload.get("dynamic_handoff_hidden_from_codex") is not True
    ):
        raise RuntimeError("trusted post-Codex helper hash mismatch")

    producer_descriptor = handoff_dir / f"{run_id}-supervisor-producer.json"
    producer_pid = os.getpid()
    producer_pgid = os.getpgid(0)
    _write_json_exclusive(
        producer_descriptor,
        {
            "schema_version": (
                "bayesfilter.complete_highdim_leaderboard.producer_descriptor.v1"
            ),
            "run_id": run_id,
            "role": "detached_supervisor",
            "pid": producer_pid,
            "pgid": producer_pgid,
            "sid": os.getsid(0),
            "start_time_ticks": _proc_start_time_ticks(producer_pid),
            "pid_namespace_inode": os.stat("/proc/self/ns/pid").st_ino,
            "command_path": str(Path(__file__).resolve()),
            "command_sha256": _sha256(Path(__file__).resolve()),
            "created_at_epoch": time.time(),
        },
    )
    phase1_subplan = root / (
        "docs/plans/bayesfilter-complete-highdim-leaderboard-"
        "phase1-ledh-harness-subplan-2026-07-11.md"
    )
    phase1_receipt = root / (
        "docs/reviews/bayesfilter-complete-highdim-leaderboard-"
        "phase1-subplan-review-receipt-2026-07-11.json"
    )
    review_check = subprocess.run(
        [
            sys.executable,
            str(review_verifier),
            "--root",
            str(root),
            "--artifact",
            str(phase1_subplan),
            "--receipt",
            str(phase1_receipt),
        ],
        check=False,
        timeout=30,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if review_check.returncode != 0:
        raise RuntimeError("trusted Phase 1 review receipt verification failed")

    events = handoff_dir / f"{run_id}-codex-events.jsonl"
    stderr = handoff_dir / f"{run_id}-codex-stderr.log"
    final_message = handoff_dir / f"{run_id}-codex-final-message.txt"
    final_message_in_copy = root / ".complete_highdim_codex_final_message.txt"
    codex_pid_file = handoff_dir / f"{run_id}-codex-process-group.pid"
    terminal = handoff_dir / f"{run_id}-terminal-status.json"
    for path in (events, stderr):
        _write_exclusive(path, "")

    process: subprocess.Popen[bytes] | None = None
    interrupted_signal: int | None = None

    def forward(signum: int, _frame: object) -> None:
        nonlocal interrupted_signal
        interrupted_signal = signum
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signum)
            except ProcessLookupError:
                pass

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, forward)

    codex_exit = 70
    started_utc = dt.datetime.now(dt.timezone.utc).isoformat()
    with (
        prompt.open("rb") as prompt_stream,
        events.open("wb") as events_stream,
        stderr.open("wb") as stderr_stream,
    ):
        direct_test_requested = (
            os.environ.get("SUPERVISOR_TEST_ALLOW_DIRECT_CODEX") == "1"
        )
        pytest_tmp = os.environ.get("PYTEST_CURRENT_TEST") and str(root).startswith(
            "/tmp/pytest-of-"
        )
        if direct_test_requested and not (
            run_id == "test-run" and pytest_tmp and str(launch_root).startswith("/tmp/")
        ):
            raise RuntimeError("direct Codex bypass is restricted to pytest test-run")
        direct_test = direct_test_requested
        codex_command = (
            [codex_bin, "exec"]
            if direct_test
            else [
                "/usr/bin/unshare",
                "--mount",
                "--pid",
                "--fork",
                "--mount-proc",
                "--kill-child=TERM",
                "/usr/bin/env",
                f"CODEX_BIN={codex_bin}",
                f"CODEX_FINAL_MESSAGE_IN_COPY={final_message_in_copy}",
                "/usr/bin/bash",
                str(root / "scripts/complete_highdim_leaderboard_codex_sandbox_entrypoint.sh"),
            ]
        )
        process = subprocess.Popen(
            codex_command,
            stdin=prompt_stream,
            stdout=events_stream,
            stderr=stderr_stream,
            start_new_session=True,
            env=_child_environment(
                support_dir=support_dir, support_bindings=support_bindings
            ),
        )
        _write_exclusive(codex_pid_file, f"{process.pid}\n")
        soft_monotonic = started_monotonic + soft_deadline
        while process.poll() is None:
            if interrupted_signal is not None:
                _terminate_group(process, 30)
                codex_exit = 128 + interrupted_signal
                break
            if time.monotonic() >= soft_monotonic:
                termination_remaining = int(
                    max(
                        0,
                        started_monotonic
                        + termination_deadline
                        - time.monotonic(),
                    )
                )
                _terminate_group(process, termination_remaining)
                codex_exit = 124
                break
            time.sleep(min(1.0, max(0.05, soft_monotonic - time.monotonic())))
        else:
            codex_exit = int(process.returncode or 0)

    if process is None or process.poll() is None:
        raise RuntimeError("Codex namespace process did not return")
    namespace_close_deadline = min(
        started_monotonic + termination_deadline,
        time.monotonic() + 10.0,
    )
    if not _wait_process_group_absent(
        process.pid, deadline_monotonic=namespace_close_deadline
    ):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        if not _wait_process_group_absent(
            process.pid,
            deadline_monotonic=min(
                started_monotonic + termination_deadline,
                time.monotonic() + 5.0,
            ),
        ):
            raise RuntimeError("Codex private PID namespace did not quiesce")

    sandbox_receipt = root / ".complete_highdim_codex_sandbox_receipt.json"
    if direct_test:
        private_namespace_inode = os.stat("/proc/self/ns/pid").st_ino
        sandbox_receipt_sha256 = None
        whole_namespace_quiescent = False
    else:
        if sandbox_receipt.is_symlink() or not sandbox_receipt.is_file():
            raise RuntimeError("Codex post-drop sandbox receipt is missing or unsafe")
        sandbox_payload = json.loads(sandbox_receipt.read_text(encoding="utf-8"))
        private_namespace_inode = sandbox_payload.get("pid_namespace_inode")
        if not (
            sandbox_payload.get("pid_in_private_namespace") == 1
            and isinstance(private_namespace_inode, int)
            and private_namespace_inode > 0
            and sandbox_payload.get("support_hashes_verified") is True
            and len(sandbox_payload.get("staging_aliases_hidden_and_read_only", []))
            == 2
        ):
            raise RuntimeError("Codex post-drop sandbox receipt is invalid")
        sandbox_receipt_sha256 = _sha256(sandbox_receipt)
        # Linux terminates all remaining members when PID 1 of a PID namespace
        # exits. The untrusted process was verified as that PID 1 after dropping
        # every capability, and unshare has now returned.
        whole_namespace_quiescent = True

    namespace_closed = handoff_dir / f"{run_id}-codex-namespace-closed.json"
    _write_json_exclusive(
        namespace_closed,
        {
            "schema_version": (
                "bayesfilter.complete_highdim_leaderboard."
                "codex_namespace_closed.v1"
            ),
            "run_id": run_id,
            "unshare_process_pid": process.pid,
            "unshare_process_returned": True,
            "unshare_process_group_absent": True,
            "private_pid_namespace_init_required_pid_one": True,
            "private_pid_namespace_inode": private_namespace_inode,
            "whole_private_pid_namespace_quiescent": whole_namespace_quiescent,
            "quiescence_proof": (
                "linux_pid_namespace_init_exit_kills_all_members_and_unshare_returned"
                if not direct_test
                else "pytest_direct_bypass_no_private_namespace_claim"
            ),
            "untrusted_code_started_only_after_capability_drop": not direct_test,
            "namespace_escape_available_to_untrusted_code": False,
            "pytest_direct_codex_bypass": direct_test,
            "sandbox_receipt_path_in_copy": str(sandbox_receipt),
            "sandbox_receipt_sha256": sandbox_receipt_sha256,
            "codex_exit_code": codex_exit,
            "closed_at_epoch": time.time(),
        },
    )

    if final_message_in_copy.is_file():
        with final_message.open("x", encoding="utf-8") as destination:
            destination.write(final_message_in_copy.read_text(encoding="utf-8"))
        final_message_in_copy.unlink()
    else:
        _write_exclusive(final_message, "")

    status_command = [
        sys.executable,
        str(status_writer),
        "--output",
        str(terminal),
        "--run-id",
        run_id,
        "--root",
        str(root),
        "--source-root",
        str(source_root),
        "--launch-root",
        str(launch_root),
        "--started-utc",
        started_utc,
        "--exit-code",
        str(codex_exit),
        "--timeout-seconds",
        str(soft_deadline),
        "--events",
        str(events),
        "--stderr",
        str(stderr),
        "--final-message",
        str(final_message),
        "--baseline",
        str(baseline),
    ]
    status_remaining = int(
        max(0, started_monotonic + hard_deadline - time.monotonic())
    )
    if status_remaining < 1:
        return 72
    status = subprocess.run(
        status_command, check=False, timeout=min(60, status_remaining)
    )
    if status.returncode != 0:
        return 72

    export_command = [
        sys.executable,
        str(exporter),
        "export",
        "--root",
        str(root),
        "--baseline",
        str(baseline),
        "--baseline-root-identity",
        str(launch_root),
        "--output-dir",
        str(handoff_dir),
        "--run-id",
        run_id,
        "--export-label",
        "primary",
    ]
    for name in (
        f"{run_id}.log",
        f"{run_id}.pid",
        f"{run_id}.wrapper.pid",
        f"{run_id}.supervisor.pid",
        f"{run_id}-codex-process-group.pid",
        f"{run_id}.handoff.ready",
    ):
        export_command.extend(("--hash-exclude-name", name))
    export_deadline = min(
        started_monotonic + termination_deadline + export_timeout,
        started_monotonic + hard_deadline,
    )
    export_remaining = int(max(0, export_deadline - time.monotonic()))
    if export_remaining < 1:
        return 71
    try:
        exported = subprocess.run(
            export_command,
            check=False,
            timeout=export_remaining,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except subprocess.TimeoutExpired:
        return 71
    if exported.returncode != 0:
        print(exported.stderr.decode(errors="replace"), file=sys.stderr)
        return 71
    export_hashes = handoff_dir / f"{run_id}-primary-export-sha256.json"
    if not _export_hashes_are_valid(export_hashes, run_id, handoff_dir):
        return 71
    verification = handoff_dir / f"{run_id}-post-export-verification.json"
    _write_json_exclusive(
        verification,
        {
            "schema_version": (
                "bayesfilter.complete_highdim_leaderboard."
                "post_export_verification.v1"
            ),
            "run_id": run_id,
            "export_label": "primary",
            "export_hashes": str(export_hashes),
            "export_hashes_sha256": _sha256(export_hashes),
            "verified_at_epoch": time.time(),
            "all_bound_files_recomputed": True,
            "codex_namespace_closed_receipt": str(namespace_closed),
            "codex_namespace_closed_receipt_sha256": _sha256(namespace_closed),
            "verified_after_codex_namespace_closed": True,
        },
    )
    return codex_exit


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:  # noqa: BLE001
        print(f"SUPERVISOR_FATAL: {error}", file=sys.stderr)
        raise SystemExit(70) from error
