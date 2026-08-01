#!/usr/bin/env python3
"""Exercise the scoped inner detached Codex namespace and GPU boundary."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import signal
import subprocess
import tempfile
import textwrap
import time
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = Path("/home/chakwong/python/claudecodex/scripts/overnight_gated_launch.sh")
PYTHON = "/home/chakwong/anaconda3/envs/tf-gpu/bin/python"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _snapshot(root: Path, excluded: Path) -> dict[str, str]:
    values = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path == excluded or excluded in path.parents:
            continue
        values[path.relative_to(root).as_posix()] = _sha256(path)
    return values


def _write_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _fake_codex_source() -> str:
    return r'''#!/home/chakwong/anaconda3/envs/tf-gpu/bin/python
import json
import os
import subprocess
import sys
from pathlib import Path

root = Path(os.environ["ROOT"])
handoff = Path(os.environ["OUTER_HANDOFF_DIR"])
support = Path(os.environ["CODEX_SUPPORT_DIR"])
host_sibling = Path("/home/chakwong/.complete_highdim_isolation_preflight_forbidden")
status = dict(
    line.split(":", 1)
    for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines()
    if ":" in line
)
assert int(status["CapEff"].strip(), 16) == 0
assert int(status["CapBnd"].strip(), 16) == 0
assert status["NoNewPrivs"].strip() == "1"
assert os.getpid() == 1
assert Path(os.environ["HOME"]).is_relative_to(Path("/tmp"))
assert Path(os.environ["CODEX_HOME"]).is_relative_to(Path("/tmp"))
assert not Path("/home/chakwong/python").exists()
assert not Path("/home/chakwong/.codex").exists()
assert not Path("/home/chakwong/.claude").exists()
for forbidden in (handoff / "codex-forbidden", host_sibling, Path("/mnt/forbidden")):
    try:
        forbidden.write_text("forbidden\n", encoding="utf-8")
    except OSError:
        pass
    else:
        forbidden.unlink(missing_ok=True)
        raise AssertionError(f"forbidden path was writable: {forbidden}")
assert not Path("/mnt/c").exists()
assert not any(handoff.iterdir())
assert sorted(path.name for path in support.iterdir()) == [
    "trusted-claude-review-gate.sh",
    "trusted-claude-worker-settings.json",
    "trusted-claude-worker.sh",
    "trusted-review-verifier.py",
]
try:
    (support / "forbidden").write_text("forbidden\n", encoding="utf-8")
except OSError:
    pass
else:
    raise AssertionError("support directory remained writable")

import tensorflow as tf
tf.config.experimental.enable_tensor_float_32_execution(True)
@tf.function(jit_compile=True)
def compiled(x):
    return tf.linalg.matmul(x, x)
y = compiled(tf.ones([8, 8], dtype=tf.float32))
record = {
    "cap_eff": 0,
    "cap_bnd": 0,
    "no_new_privs": 1,
    "private_pid": os.getpid(),
    "pid_namespace_inode": os.stat("/proc/self/ns/pid").st_ino,
    "handoff_read_only": True,
    "handoff_contents_hidden": True,
    "support_read_only": True,
    "support_hashes_verified": True,
    "claude_worker_preserves_stream_and_tool_use_metadata": True,
    "staging_aliases_hidden_and_read_only": [
        "/tmp/complete-highdim-model-visible",
        "/tmp/complete-highdim-support-staging",
    ],
    "host_sibling_hidden": True,
    "selected_runtime_mounts_only": True,
    "mnt_hidden_and_read_only": True,
    "home_private_tmpfs": True,
    "codex_home_private_tmpfs": True,
    "physical_gpus": [str(d) for d in tf.config.list_physical_devices("GPU")],
    "logical_gpus": [str(d) for d in tf.config.list_logical_devices("GPU")],
    "jit_compile": True,
    "tf32_execution_enabled": bool(tf.config.experimental.tensor_float_32_execution_enabled()),
    "output_device": str(y.device),
    "output_finite": bool(tf.reduce_all(tf.math.is_finite(y)).numpy()),
}
(root / "scoped-inner-isolation-marker.json").write_text(
    json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
args = sys.argv[1:]
if "--output-last-message" in args:
    path = Path(args[args.index("--output-last-message") + 1])
    path.write_text("FAKE_CODEX_BOUNDARY_OK\n", encoding="utf-8")
print(json.dumps({"type": "fake_codex_boundary", "status": "ok"}))
'''


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("fake-gpu-boundary", "real-codex"),
        default="fake-gpu-boundary",
    )
    parser.add_argument("--workspace-root", type=Path)
    parser.add_argument("--parent-run-id")
    parser.add_argument("--parent-nonce")
    parser.add_argument("--approval-instance-id")
    args = parser.parse_args(argv)
    run_id = f"complete-leaderboard-isolation-preflight-v3-{args.mode}"
    nonce = f"isolation-preflight-v3-{args.mode}-nonce"

    workspace_root = args.workspace_root or ROOT
    workspace_root = workspace_root.resolve(strict=True)
    with tempfile.TemporaryDirectory(
        prefix="complete-leaderboard-isolation-", dir=workspace_root
    ) as tmp:
        base = Path(tmp)
        source = base / "source"
        source.mkdir()
        logs_parent = source / "docs/plans/logs"
        handoff = logs_parent / run_id
        logs_parent.mkdir(parents=True)
        handoff.mkdir(mode=0o700)
        launch_root = Path(f"/tmp/{run_id}-{os.getpid()}-workspace")
        fake_codex = source / "fake-codex.py"
        if args.mode == "fake-gpu-boundary":
            fake_codex.write_text(_fake_codex_source(), encoding="utf-8")
            fake_codex.chmod(0o755)

        for relative in (
            "docs/plans/bayesfilter-complete-highdim-leaderboard-overnight-supervisor-prompt-2026-07-11.md",
            "scripts/export_complete_highdim_leaderboard_isolated_changes.py",
            "scripts/write_complete_highdim_leaderboard_terminal_status.py",
            "scripts/complete_highdim_leaderboard_overnight_supervisor.sh",
            "scripts/complete_highdim_leaderboard_overnight_supervisor.py",
            "scripts/complete_highdim_leaderboard_namespace_entrypoint.sh",
            "scripts/complete_highdim_leaderboard_codex_sandbox_entrypoint.sh",
            "scripts/complete_highdim_leaderboard_exec_codex_after_boundary_check.py",
            "scripts/complete_highdim_leaderboard_claude_audit_worker.sh",
            "scripts/verify_complete_highdim_leaderboard_review_receipt.py",
            "docs/plans/bayesfilter-complete-highdim-leaderboard-phase1-ledh-harness-subplan-2026-07-11.md",
            "docs/reviews/bayesfilter-complete-highdim-leaderboard-phase1-subplan-review-receipt-2026-07-11.json",
        ):
            destination = source / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / relative, destination)
        subprocess.run(["git", "init", "-q", str(source)], check=True)
        source_sentinel = source / "source-sentinel.txt"
        source_sentinel.write_text("source-only\n", encoding="utf-8")
        prompt = source / (
            "docs/plans/bayesfilter-complete-highdim-leaderboard-"
            "overnight-supervisor-prompt-2026-07-11.md"
        )
        if args.mode == "real-codex":
            prompt.write_text(
                "Return exactly RESTRICTED_CODEX_PROBE_OK. Do not run tools.\n",
                encoding="utf-8",
            )
        forbidden_sibling = Path(
            "/home/chakwong/.complete_highdim_isolation_preflight_forbidden"
        )
        if forbidden_sibling.exists() or forbidden_sibling.is_symlink():
            raise RuntimeError("forbidden sibling probe path already exists")
        before = _snapshot(source, handoff)

        started_epoch = time.time()
        started_monotonic = time.monotonic()
        # The external launcher copies source before namespace handoff. Its copy
        # command below creates the sentinel and baseline in that copy.
        preparation_script = base / "prepare-preflight.py"
        preparation_script.write_text(
            textwrap.dedent(
                '''\
                import hashlib
                import json
                import shutil
                import subprocess
                import sys
                from pathlib import Path

                source = Path(sys.argv[1])
                launch = Path(sys.argv[2])
                handoff = Path(sys.argv[3])
                run_id = sys.argv[4]
                nonce = sys.argv[5]
                started_epoch = float(sys.argv[6])
                started_monotonic = float(sys.argv[7])
                subprocess.run(
                    ["/usr/bin/cp", "-a", f"{source}/.", f"{launch}/"], check=True
                )
                (launch / ".complete_highdim_leaderboard_copy_sentinel.json").write_text(
                    json.dumps(
                        {"run_id": run_id, "nonce": nonce, "launch_root": str(launch)}
                    ) + "\\n",
                    encoding="utf-8",
                )
                exporter = launch / "scripts/export_complete_highdim_leaderboard_isolated_changes.py"
                trusted_exporter = handoff / f"{run_id}-trusted-exporter.py"
                trusted_status = handoff / f"{run_id}-trusted-status-writer.py"
                trusted_verifier = handoff / f"{run_id}-trusted-review-verifier.py"
                trusted_claude_gate = handoff / f"{run_id}-trusted-claude-review-gate.sh"
                trusted_claude_worker = handoff / f"{run_id}-trusted-claude-worker.sh"
                trusted_claude_settings = handoff / f"{run_id}-trusted-claude-worker-settings.json"
                shutil.copy2(exporter, trusted_exporter)
                shutil.copy2(
                    launch / "scripts/write_complete_highdim_leaderboard_terminal_status.py",
                    trusted_status,
                )
                shutil.copy2(
                    launch / "scripts/verify_complete_highdim_leaderboard_review_receipt.py",
                    trusted_verifier,
                )
                shutil.copy2(
                    "/home/chakwong/python/claudecodex/scripts/claude_review_gate.sh",
                    trusted_claude_gate,
                )
                shutil.copy2(
                    launch / "scripts/complete_highdim_leaderboard_claude_audit_worker.sh",
                    trusted_claude_worker,
                )
                shutil.copy2(
                    "/home/chakwong/.claude/settings.codex-worker.json",
                    trusted_claude_settings,
                )
                sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
                support = launch / ".complete_highdim_codex_support"
                support.mkdir()
                support_sources = (
                    (launch / "scripts/verify_complete_highdim_leaderboard_review_receipt.py", "trusted-review-verifier.py"),
                    (Path("/home/chakwong/python/claudecodex/scripts/claude_review_gate.sh"), "trusted-claude-review-gate.sh"),
                    (launch / "scripts/complete_highdim_leaderboard_claude_audit_worker.sh", "trusted-claude-worker.sh"),
                    (Path("/home/chakwong/.claude/settings.codex-worker.json"), "trusted-claude-worker-settings.json"),
                )
                support_bindings = []
                for support_source, name in support_sources:
                    destination = support / name
                    shutil.copy2(support_source, destination)
                    support_bindings.append(
                        {"path": name, "sha256": sha(destination), "size": destination.stat().st_size}
                    )
                baseline = handoff / f"{run_id}-baseline-snapshot.json"
                subprocess.run(
                    [
                        "/home/chakwong/anaconda3/envs/tf-gpu/bin/python",
                        str(exporter),
                        "snapshot",
                        "--root",
                        str(launch),
                        "--root-identity",
                        str(launch),
                        "--output",
                        str(baseline),
                    ],
                    check=True,
                )
                preparation = {
                    "run_id": run_id,
                    "nonce": nonce,
                    "launch_root": str(launch),
                    "started_epoch": started_epoch,
                    "started_monotonic": started_monotonic,
                    "approval_not_after_epoch": started_epoch + 3600,
                    "codex_support_dir_name": ".complete_highdim_codex_support",
                    "codex_support_files": support_bindings,
                    "dynamic_handoff_hidden_from_codex": True,
                    "trusted_exporter": str(trusted_exporter),
                    "trusted_exporter_sha256": sha(trusted_exporter),
                    "trusted_status_writer": str(trusted_status),
                    "trusted_status_writer_sha256": sha(trusted_status),
                    "trusted_review_verifier": str(trusted_verifier),
                    "trusted_review_verifier_sha256": sha(trusted_verifier),
                    "trusted_claude_gate": str(trusted_claude_gate),
                    "trusted_claude_gate_sha256": sha(trusted_claude_gate),
                    "trusted_claude_worker": str(trusted_claude_worker),
                    "trusted_claude_worker_sha256": sha(trusted_claude_worker),
                    "trusted_claude_settings": str(trusted_claude_settings),
                    "trusted_claude_settings_sha256": sha(trusted_claude_settings),
                }
                (handoff / f"{run_id}-launch-preparation.json").write_text(
                    json.dumps(preparation) + "\\n", encoding="utf-8"
                )
                '''
            ),
            encoding="utf-8",
        )
        copy_cmd = subprocess.list2cmdline(
            [
                PYTHON,
                str(preparation_script),
                str(source),
                str(launch_root),
                str(handoff),
                run_id,
                nonce,
                str(started_epoch),
                str(started_monotonic),
            ]
        )
        codex_bin = (
            "$ROOT/fake-codex.py"
            if args.mode == "fake-gpu-boundary"
            else "/home/chakwong/.nvm/versions/node/v22.23.1/bin/codex"
        )
        supervisor_cmd = " ".join(
            (
                "/usr/bin/env",
                f"COPY_SENTINEL_NONCE={nonce}",
                f"LAUNCH_STARTED_EPOCH={started_epoch}",
                f"LAUNCH_STARTED_MONOTONIC={started_monotonic}",
                "CODEX_SOFT_DEADLINE_SECONDS=60",
                "PROCESS_TERMINATION_DEADLINE_SECONDS=90",
                "HARD_DEADLINE_SECONDS=180",
                f"CODEX_BIN={codex_bin}",
                "/usr/bin/bash",
                "scripts/complete_highdim_leaderboard_namespace_entrypoint.sh",
            )
        )
        command = [
            "/usr/bin/bash",
            str(LAUNCHER),
            "--root",
            str(source),
            "--run-id",
            run_id,
            "--log-dir",
            str(handoff),
            "--launch-root",
            str(launch_root),
            "--supervisor-cmd",
            supervisor_cmd,
            "--copy-cmd",
            copy_cmd,
            "--wait-attempts",
            "200",
            "--wait-seconds",
            "0.05",
        ]
        launched = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=180,
        )
        launcher_log = handoff / f"{run_id}.log"
        terminal = handoff / f"{run_id}-terminal-status.json"
        verification_path = handoff / f"{run_id}-post-export-verification.json"
        namespace_closed_path = handoff / f"{run_id}-codex-namespace-closed.json"
        deadline = time.monotonic() + (60.0 if launched.returncode == 0 else 5.0)
        while (
            not (terminal.is_file() and verification_path.is_file())
            and time.monotonic() < deadline
        ):
            time.sleep(0.1)
        launcher_log_text = (
            launcher_log.read_text(encoding="utf-8", errors="replace")
            if launcher_log.is_file()
            else ""
        )
        process_diagnostics: dict[str, Any] = {}
        for label, pid_path in (
            ("supervisor", handoff / f"{run_id}.supervisor.pid"),
            ("codex_group", handoff / f"{run_id}-codex-process-group.pid"),
        ):
            if not pid_path.is_file():
                process_diagnostics[label] = {"pid_file_present": False}
                continue
            value = pid_path.read_text(encoding="utf-8").strip()
            record: dict[str, Any] = {
                "pid_file_present": True,
                "pid_file_value": value,
            }
            if value.isdigit():
                pid = int(value)
                proc = Path(f"/proc/{pid}")
                record["proc_present"] = proc.is_dir()
                if proc.is_dir():
                    try:
                        record["stat"] = (proc / "stat").read_text(
                            encoding="utf-8"
                        )
                        record["status"] = (proc / "status").read_text(
                            encoding="utf-8"
                        )
                        record["pid_namespace_inode"] = (proc / "ns/pid").stat().st_ino
                    except (FileNotFoundError, PermissionError, OSError) as error:
                        record["inspection_error"] = repr(error)
            process_diagnostics[label] = record
        marker = launch_root / "scoped-inner-isolation-marker.json"
        sandbox_receipt = launch_root / ".complete_highdim_codex_sandbox_receipt.json"
        marker_payload = (
            json.loads(marker.read_text(encoding="utf-8")) if marker.is_file() else {}
        )
        sandbox_payload = (
            json.loads(sandbox_receipt.read_text(encoding="utf-8"))
            if sandbox_receipt.is_file()
            else {}
        )
        terminal_payload = (
            json.loads(terminal.read_text(encoding="utf-8"))
            if terminal.is_file()
            else {}
        )
        verification_payload = (
            json.loads(verification_path.read_text(encoding="utf-8"))
            if verification_path.is_file()
            else {}
        )
        namespace_closed_payload = (
            json.loads(namespace_closed_path.read_text(encoding="utf-8"))
            if namespace_closed_path.is_file()
            else {}
        )
        namespace_receipt = handoff / f"{run_id}-namespace-boundary.json"
        namespace_payload = (
            json.loads(namespace_receipt.read_text(encoding="utf-8"))
            if namespace_receipt.is_file()
            else {}
        )
        codex_stderr_path = handoff / f"{run_id}-codex-stderr.log"
        codex_final_path = handoff / f"{run_id}-codex-final-message.txt"
        codex_final_message = (
            codex_final_path.read_text(encoding="utf-8", errors="replace")
            if codex_final_path.is_file()
            else ""
        )
        after = _snapshot(source, handoff)
        source_unchanged = before == after and source_sentinel.read_text() == "source-only\n"
        sibling_unchanged = not forbidden_sibling.exists()
        common_pass = bool(
            launched.returncode == 0
            and terminal.is_file()
            and terminal_payload.get("codex_exit_code") == 0
            and verification_payload.get("export_label") == "primary"
            and verification_payload.get("all_bound_files_recomputed") is True
            and verification_payload.get("verified_after_codex_namespace_closed")
            is True
            and namespace_closed_payload.get("whole_private_pid_namespace_quiescent")
            is True
            and namespace_closed_payload.get("quiescence_proof")
            == "linux_pid_namespace_init_exit_kills_all_members_and_unshare_returned"
            and namespace_closed_payload.get(
                "untrusted_code_started_only_after_capability_drop"
            )
            is True
            and namespace_closed_payload.get(
                "namespace_escape_available_to_untrusted_code"
            )
            is False
            and namespace_closed_payload.get("pytest_direct_codex_bypass") is False
            and sandbox_payload.get("cap_eff") == 0
            and sandbox_payload.get("cap_bnd") == 0
            and sandbox_payload.get("no_new_privs") == 1
            and sandbox_payload.get("pid_in_private_namespace") == 1
            and isinstance(sandbox_payload.get("pid_namespace_inode"), int)
            and sandbox_payload.get("pid_namespace_inode", 0) > 0
            and sandbox_payload.get("handoff_read_only") is True
            and sandbox_payload.get("handoff_contents_hidden") is True
            and sandbox_payload.get("support_read_only") is True
            and sandbox_payload.get("support_hashes_verified") is True
            and sandbox_payload.get(
                "claude_worker_preserves_stream_and_tool_use_metadata"
            )
            is True
            and len(
                sandbox_payload.get("staging_aliases_hidden_and_read_only", [])
            )
            == 2
            and sandbox_payload.get("sibling_home_hidden") is True
            and sandbox_payload.get("selected_runtime_mounts_only") is True
            and source_unchanged
            and sibling_unchanged
        )
        mode_pass = (
            bool(
                marker_payload.get("cap_eff") == 0
                and marker_payload.get("cap_bnd") == 0
                and marker_payload.get("no_new_privs") == 1
                and marker_payload.get("private_pid") == 1
                and marker_payload.get("handoff_read_only") is True
                and marker_payload.get("handoff_contents_hidden") is True
                and marker_payload.get("support_read_only") is True
                and marker_payload.get("host_sibling_hidden") is True
                and marker_payload.get("selected_runtime_mounts_only") is True
                and marker_payload.get("mnt_hidden_and_read_only") is True
                and marker_payload.get("jit_compile") is True
                and marker_payload.get("tf32_execution_enabled") is True
                and marker_payload.get("physical_gpus")
                and marker_payload.get("logical_gpus")
                and "GPU" in str(marker_payload.get("output_device", "")).upper()
                and marker_payload.get("output_finite") is True
            )
            if args.mode == "fake-gpu-boundary"
            else codex_final_message.strip() == "RESTRICTED_CODEX_PROBE_OK"
        )
        passed = common_pass and mode_pass
        record = {
            "schema_version": (
                "bayesfilter.complete_highdim_leaderboard.isolation_preflight.v3"
            ),
            "mode": args.mode,
            "parent_run_id": args.parent_run_id,
            "parent_nonce": args.parent_nonce,
            "approval_instance_id": args.approval_instance_id,
            "is_post_approval_fresh_probe": bool(
                args.parent_run_id
                and args.parent_nonce
                and args.approval_instance_id
            ),
            "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "launcher": str(LAUNCHER),
            "launcher_sha256": _sha256(LAUNCHER),
            "preflight_script_path": str(Path(__file__).resolve()),
            "preflight_script_sha256": _sha256(Path(__file__).resolve()),
            "command": command,
            "launcher_exit_code": launched.returncode,
            "launcher_stdout": launched.stdout,
            "launcher_stderr": launched.stderr,
            "launcher_log": launcher_log_text,
            "process_diagnostics_at_collection": process_diagnostics,
            "handoff_files": sorted(path.name for path in handoff.iterdir()),
            "handoff_ready": (handoff / f"{run_id}.handoff.ready").is_file(),
            "terminal_present": terminal.is_file(),
            "source_workspace_unchanged": source_unchanged,
            "host_sibling_unchanged": sibling_unchanged,
            "isolated_workspace_changed": marker.is_file(),
            "fake_supervisor_exit_code": (
                terminal_payload.get("codex_exit_code")
            ),
            "boundary": marker_payload,
            "sandbox_receipt": sandbox_payload,
            "namespace_receipt": namespace_payload,
            "terminal": terminal_payload,
            "post_export_verification": verification_payload,
            "codex_namespace_closed": namespace_closed_payload,
            "codex_stderr": (
                codex_stderr_path.read_text(encoding="utf-8", errors="replace")
                if codex_stderr_path.is_file()
                else ""
            ),
            "codex_final_message": codex_final_message,
            "preflight_pass": passed,
            "nonclaims": [
                "scoped inner fake-Codex handoff does not authorize the real launch",
                "preflight does not cover the exact outer wrapper, production preparer, watchdog, finalizer, alias lock, or post-lock seal validation",
                "namespace and GPU preflight is not leaderboard or scientific evidence"
            ],
        }
        _write_atomic(args.output, record)
        if launch_root.exists():
            shutil.rmtree(launch_root)
        return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
