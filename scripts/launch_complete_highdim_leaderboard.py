#!/usr/bin/env python3
"""Execute the one concrete human-approved detached leaderboard launch."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence


SCRIPT_ROOT = Path(__file__).resolve().parents[1]
CONTROL_ROOT = Path("/home/chakwong/BayesFilter")
LAUNCHER = Path("/home/chakwong/python/claudecodex/scripts/overnight_gated_launch.sh")
PYTHON = "/home/chakwong/anaconda3/envs/tf-gpu/bin/python"
BASH = "/usr/bin/bash"
EXACT_WRAPPER_NAME = "complete_highdim_leaderboard_exact_wrapper.sh"
CONSTRAINED_PATH = ":".join(
    (
        "/home/chakwong/.nvm/versions/node/v22.23.1/bin",
        "/home/chakwong/.local/bin",
        "/home/chakwong/anaconda3/envs/tf-gpu/bin",
        "/usr/bin",
        "/bin",
        "/usr/lib/wsl/lib",
    )
)
CODEX_SOFT_DEADLINE_SECONDS = 26400
PROCESS_TERMINATION_DEADLINE_SECONDS = 26700
PRIMARY_EXPORT_TIMEOUT_SECONDS = 600
WATCHDOG_VERIFICATION_DEADLINE_SECONDS = 27360
NAMESPACE_OUTER_DEADLINE_SECONDS = 27600
SUPERVISOR_AND_WATCHDOG_HARD_DEADLINE_SECONDS = 28200
PRODUCER_CLOSE_DEADLINE_SECONDS = 28230
OUTER_FINALIZER_DEADLINE_SECONDS = 28720
OUTER_TERM_DEADLINE_SECONDS = 28740
OUTER_KILL_DEADLINE_SECONDS = 28800
MANIFEST_SCHEMA = "bayesfilter.complete_highdim_leaderboard.launch_manifest.v7"
OWNER_ACCEPTED_RUN_SCOPED_LIMITATION_IDS = [
    "CLAUDE_TOOL_CAPABILITY",
    "CLAUDE_CODEX_CREDENTIAL_ACCESS",
    "PRIMARY_EXPORT_COMPLETENESS",
    "SEAL_LOCK_TOCTOU",
    "TRUSTED_PREFLIGHT_OUTER_COVERAGE",
]
EXPORT_SCHEMAS = {
    "bayesfilter.complete_highdim_leaderboard.isolated_export.v1",
    "bayesfilter.complete_highdim_leaderboard.export_hashes.v1",
}


def _required_environment(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is required")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_exclusive(path: Path, payload: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _load_review_receipt(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("launch review receipt is not an object")
    return value


def _load_manifest(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != MANIFEST_SCHEMA:
        raise RuntimeError("launch manifest schema is invalid")
    return value


def _require_unexpired(approval_not_after_epoch: int, *, action: str) -> None:
    if time.time() >= approval_not_after_epoch:
        raise RuntimeError(f"one-time launch approval expired before {action}")


def _remaining_timeout(
    started_monotonic: float,
    deadline_seconds: int,
    *,
    maximum: float | None = None,
    reserve: float = 0.0,
) -> float:
    remaining = started_monotonic + deadline_seconds - time.monotonic() - reserve
    if remaining <= 0.0:
        raise TimeoutError("common monotonic launch budget is exhausted")
    return min(remaining, maximum) if maximum is not None else remaining


def exact_wrapper_argv(
    *,
    run_id: str,
    launch_root: Path,
    nonce: str,
    manifest: Path,
    launch_review_receipt: Path,
    source_snapshot_root: Path,
    source_inventory: Path,
    runtime_fingerprint: Path,
    approval_instance_id: str,
    approval_not_after_epoch: int,
) -> list[str]:
    return [
        BASH,
        str(
            source_snapshot_root
            / "scripts"
            / EXACT_WRAPPER_NAME
        ),
        "--run-id",
        run_id,
        "--launch-root",
        str(launch_root),
        "--nonce",
        nonce,
        "--manifest",
        str(manifest),
        "--launch-review-receipt",
        str(launch_review_receipt),
        "--source-snapshot-root",
        str(source_snapshot_root),
        "--source-inventory",
        str(source_inventory),
        "--runtime-fingerprint",
        str(runtime_fingerprint),
        "--approval-instance-id",
        approval_instance_id,
        "--approval-not-after-epoch",
        str(approval_not_after_epoch),
    ]


def exact_supervisor_command(*, nonce: str, source_snapshot_root: Path) -> str:
    return " ".join(
        (
            "/usr/bin/env",
            f"PATH={CONSTRAINED_PATH}",
            f"COPY_SENTINEL_NONCE={nonce}",
            f"CODEX_SOFT_DEADLINE_SECONDS={CODEX_SOFT_DEADLINE_SECONDS}",
            (
                "PROCESS_TERMINATION_DEADLINE_SECONDS="
                f"{PROCESS_TERMINATION_DEADLINE_SECONDS}"
            ),
            (
                "HARD_DEADLINE_SECONDS="
                f"{SUPERVISOR_AND_WATCHDOG_HARD_DEADLINE_SECONDS}"
            ),
            f"EXPORT_TIMEOUT_SECONDS={PRIMARY_EXPORT_TIMEOUT_SECONDS}",
            "SUPERVISOR_TEST_ALLOW_DIRECT_CODEX=0",
            "CODEX_BIN=/home/chakwong/.nvm/versions/node/v22.23.1/bin/codex",
            "/usr/bin/timeout",
            "--signal=TERM",
            "--kill-after=60s",
            f"{NAMESPACE_OUTER_DEADLINE_SECONDS}s",
            "/usr/bin/bash",
            str(
                source_snapshot_root
                / "scripts/complete_highdim_leaderboard_namespace_entrypoint.sh"
            ),
        )
    )


def _kill_group(
    process: subprocess.Popen[str], *, deadline_monotonic: float
) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    remaining = deadline_monotonic - time.monotonic()
    try:
        if remaining > 0.0:
            process.wait(timeout=min(30.0, remaining))
        else:
            raise subprocess.TimeoutExpired(process.args, 0.0)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _process_identity(pid: int) -> dict[str, int] | None:
    try:
        value = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    closing = value.rfind(")")
    if closing < 0:
        return None
    tail = value[closing + 2 :].split()
    if len(tail) < 20:
        return None
    return {
        "state_zombie": int(tail[0] == "Z"),
        "ppid": int(tail[1]),
        "pgid": int(tail[2]),
        "sid": int(tail[3]),
        "start_time_ticks": int(tail[19]),
    }


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


def _load_producer(
    path: Path,
    *,
    run_id: str,
    role: str,
    expected_command_path: Path,
    expected_command_hash_source: Path,
) -> dict | None:
    if path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1:
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not (
        payload.get("schema_version")
        == "bayesfilter.complete_highdim_leaderboard.producer_descriptor.v1"
        and payload.get("run_id") == run_id
        and payload.get("role") == role
        and all(
            isinstance(payload.get(name), int) and payload[name] > 0
            for name in ("pid", "pgid", "sid", "start_time_ticks", "pid_namespace_inode")
        )
        and payload.get("pid_namespace_inode") == os.stat("/proc/self/ns/pid").st_ino
        and payload.get("command_path") == str(expected_command_path)
        and payload.get("command_sha256") == _sha256(expected_command_hash_source)
    ):
        return None
    return payload


def _producer_closed(descriptor: dict) -> bool:
    identity = _process_identity(descriptor["pid"])
    same_identity_alive = bool(
        identity
        and not identity["state_zombie"]
        and identity["pgid"] == descriptor["pgid"]
        and identity["sid"] == descriptor["sid"]
        and identity["start_time_ticks"] == descriptor["start_time_ticks"]
    )
    return not same_identity_alive and not _process_group_alive(descriptor["pgid"])


def _extra_live_namespace_pids() -> list[int]:
    table: dict[int, dict[str, int]] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        identity = _process_identity(int(entry.name))
        if identity is not None and not identity["state_zombie"]:
            table[int(entry.name)] = identity
    ancestors: set[int] = set()
    pid = os.getpid()
    while pid > 0 and pid not in ancestors:
        ancestors.add(pid)
        identity = table.get(pid)
        if identity is None:
            break
        pid = identity["ppid"]
    return sorted(set(table) - ancestors)


def _whole_private_namespace_quiescent() -> bool:
    if _extra_live_namespace_pids():
        return False
    # Two observations close the ordinary fork/exit enumeration race.
    time.sleep(0.05)
    return not _extra_live_namespace_pids()


def _unapproved_export_artifacts(handoff_dir: Path, run_id: str) -> list[str]:
    allowed = {
        f"{run_id}-primary-isolated-change-manifest.json",
        f"{run_id}-primary-isolated-changed-files.tar.gz",
        f"{run_id}-primary-isolated-tracked.diff",
        f"{run_id}-primary-isolated-git-status.txt",
        f"{run_id}-primary-export-sha256.json",
    }
    export_suffixes = (
        "-isolated-change-manifest.json",
        "-isolated-changed-files.tar.gz",
        "-isolated-tracked.diff",
        "-isolated-git-status.txt",
        "-export-sha256.json",
    )
    rejected: set[str] = set()
    for path in handoff_dir.iterdir():
        if not path.name.startswith(f"{run_id}-") or path.name in allowed:
            continue
        if "-fallback-" in path.name or path.name.endswith(export_suffixes):
            rejected.add(path.name)
            continue
        if path.suffix != ".json" or path.is_symlink() or not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(payload, dict) and payload.get("schema_version") in EXPORT_SCHEMAS:
            rejected.add(path.name)
    return sorted(rejected)


def _verify_launch_authorization(args: argparse.Namespace) -> tuple[dict, str, list[str]]:
    _require_unexpired(
        args.approval_not_after_epoch, action="pre-handoff authorization verification"
    )
    manifest_sha256 = _sha256(args.manifest)
    manifest = _load_manifest(args.manifest)
    receipt = _load_review_receipt(args.launch_review_receipt)
    expected_manifest_rel = str(args.manifest.relative_to(CONTROL_ROOT))
    expected_wrapper_argv = exact_wrapper_argv(
        run_id=args.run_id,
        launch_root=args.launch_root,
        nonce=args.nonce,
        manifest=args.manifest,
        launch_review_receipt=args.launch_review_receipt,
        source_snapshot_root=args.source_snapshot_root,
        source_inventory=args.source_inventory,
        runtime_fingerprint=args.runtime_fingerprint,
        approval_instance_id=args.approval_instance_id,
        approval_not_after_epoch=args.approval_not_after_epoch,
    )
    concrete = manifest.get("concrete_run")
    limitations = manifest.get("owner_accepted_run_scoped_limitations")
    audit = manifest.get("mandatory_post_run_integrity_audit")
    if not isinstance(concrete, dict) or any(
        (
            manifest.get("program_root") != str(CONTROL_ROOT),
            concrete.get("run_id") != args.run_id,
            concrete.get("launch_root") != str(args.launch_root),
            concrete.get("copy_sentinel_nonce") != args.nonce,
            concrete.get("source_snapshot_root") != str(args.source_snapshot_root),
            concrete.get("source_inventory") != str(args.source_inventory),
            concrete.get("runtime_fingerprint") != str(args.runtime_fingerprint),
            concrete.get("approval_instance_id") != args.approval_instance_id,
            concrete.get("approval_not_after_epoch")
            != args.approval_not_after_epoch,
            manifest.get("exact_wrapper_argv") != expected_wrapper_argv,
            manifest.get("exact_wrapper_shell") != shlex.join(expected_wrapper_argv),
        )
    ):
        raise RuntimeError("launch manifest does not bind this exact launch")
    if not (
        isinstance(limitations, dict)
        and limitations.get("run_id") == args.run_id
        and limitations.get("repository_default") is False
        and limitations.get("reusable_for_other_runs") is False
        and limitations.get("technical_guarantees_repaired") is False
        and limitations.get("accepted_limitation_ids")
        == OWNER_ACCEPTED_RUN_SCOPED_LIMITATION_IDS
        and limitations.get("sixth_review_round", {}).get("authorized") is True
        and limitations.get("sixth_review_round", {}).get("iteration") == 6
        and limitations.get("sixth_review_round", {}).get(
            "ordinary_phase_review_limit_unchanged"
        )
        == 5
        and isinstance(audit, dict)
        and audit.get("required_before_completion_or_release_claim") is True
        and audit.get("semantic_inspection_receipt_required") is True
        and audit.get("post_lock_receipt", {}).get(
            "required_for_structural_pass"
        )
        is True
        and audit.get("phase8_phase9_validator_exit_zero_and_all_checks_required")
        is True
        and audit.get("launch_itself_grants_completion_or_release_authority")
        is False
    ):
        raise RuntimeError("run-scoped risk acceptance or post-run audit is invalid")
    if (
        receipt.get("schema_version")
        != "bayesfilter.complete_highdim_leaderboard.launch_readiness_review_receipt.v2"
        or receipt.get("verdict") != "AGREE"
        or receipt.get("iteration") != 6
        or receipt.get("review_scope")
        != "run_scoped_waiver_post_run_audit_and_exact_launch_package"
        or receipt.get("reviewed_manifest_path") != expected_manifest_rel
        or receipt.get("reviewed_manifest_sha256") != manifest_sha256
        or receipt.get("run_id") != args.run_id
        or receipt.get("launch_root") != str(args.launch_root)
        or receipt.get("nonce") != args.nonce
        or receipt.get("source_snapshot_root") != str(args.source_snapshot_root)
        or receipt.get("source_inventory_sha256") != _sha256(args.source_inventory)
        or receipt.get("runtime_fingerprint_sha256")
        != _sha256(args.runtime_fingerprint)
        or receipt.get("approval_instance_id") != args.approval_instance_id
        or receipt.get("approval_not_after_epoch") != args.approval_not_after_epoch
        or receipt.get("exact_wrapper_argv") != expected_wrapper_argv
        or receipt.get("accepted_limitation_ids")
        != OWNER_ACCEPTED_RUN_SCOPED_LIMITATION_IDS
        or receipt.get("risk_acceptance_amendment_sha256")
        != limitations["authorization_amendment"]["sha256"]
        or receipt.get("post_run_integrity_audit_plan_sha256")
        != audit["plan"]["sha256"]
        or receipt.get("post_run_integrity_auditor_sha256")
        != audit["auditor"]["sha256"]
        or receipt.get("final_exact_launch_command_authorized") is not False
        or receipt.get("completion_or_release_authority_granted") is not False
    ):
        raise RuntimeError("launch review receipt does not bind this exact launch")
    return manifest, manifest_sha256, expected_wrapper_argv


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--launch-root", type=Path, required=True)
    parser.add_argument("--nonce", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--launch-review-receipt", type=Path, required=True)
    parser.add_argument("--source-snapshot-root", type=Path, required=True)
    parser.add_argument("--source-inventory", type=Path, required=True)
    parser.add_argument("--runtime-fingerprint", type=Path, required=True)
    parser.add_argument("--approval-instance-id", required=True)
    parser.add_argument("--approval-not-after-epoch", type=int, required=True)
    parser.add_argument("--verify-only-before-handoff", action="store_true")
    args = parser.parse_args(argv)

    manifest, manifest_sha256, expected_wrapper_argv = _verify_launch_authorization(args)
    if args.verify_only_before_handoff:
        print("PRE_HANDOFF_LAUNCH_AUTHORIZATION_PASS")
        return 0

    expected_handoff = args.source_snapshot_root / "docs/plans/logs" / args.run_id
    live_handoff = CONTROL_ROOT / "docs/plans/logs" / args.run_id
    if os.environ.get("COMPLETE_HIGHDIM_OUTER_BOUNDARY_ACTIVE") != "1":
        raise RuntimeError("outer read-only PID/mount boundary is not active")
    started_epoch = float(_required_environment("LAUNCH_STARTED_EPOCH"))
    started_monotonic = float(_required_environment("LAUNCH_STARTED_MONOTONIC"))
    _require_unexpired(args.approval_not_after_epoch, action="outer-boundary launch")
    if args.launch_root != Path(f"/tmp/{args.run_id}-workspace"):
        raise RuntimeError("launch root does not match the concrete run id")
    if args.launch_root.exists() or args.launch_root.is_symlink():
        raise RuntimeError("launch root already exists")
    if expected_handoff.is_symlink() or not expected_handoff.is_dir():
        raise RuntimeError("outer boundary did not create the exact handoff")
    if any(expected_handoff.iterdir()):
        raise RuntimeError("outer boundary handoff is not fresh and empty")
    parent = expected_handoff.parent
    if parent.is_symlink() or parent.resolve(strict=True) != parent:
        raise RuntimeError("snapshot handoff parent is not a canonical directory")
    if live_handoff.is_symlink() or not live_handoff.is_dir():
        raise RuntimeError("live handoff bind source is missing or unsafe")
    source_write_probe = CONTROL_ROOT / f".{args.run_id}-forbidden-outer-source-write"
    try:
        source_write_probe.write_text("boundary failure\n", encoding="utf-8")
    except OSError:
        pass
    else:
        source_write_probe.unlink(missing_ok=True)
        raise RuntimeError("live source root is writable inside the outer boundary")
    handoff_probe = expected_handoff / f"{args.run_id}-outer-handoff-probe.tmp"
    handoff_probe.write_text("probe\n", encoding="utf-8")
    handoff_probe.unlink()
    approval = expected_handoff / f"{args.run_id}-human-launch-approval.json"
    preparation_cmd = [
        PYTHON,
        str(
            args.source_snapshot_root
            / "scripts/prepare_complete_highdim_leaderboard_launch.py"
        ),
        "--run-id",
        args.run_id,
        "--source-root",
        str(args.source_snapshot_root),
        "--launch-root",
        str(args.launch_root),
        "--handoff-dir",
        str(expected_handoff),
        "--nonce",
        args.nonce,
        "--started-epoch",
        str(started_epoch),
        "--started-monotonic",
        str(started_monotonic),
        "--manifest",
        str(args.manifest),
        "--expected-manifest-sha256",
        manifest_sha256,
        "--source-inventory",
        str(args.source_inventory),
        "--expected-source-inventory-sha256",
        _sha256(args.source_inventory),
        "--runtime-fingerprint",
        str(args.runtime_fingerprint),
        "--expected-runtime-fingerprint-sha256",
        _sha256(args.runtime_fingerprint),
        "--watchdog-verification-deadline",
        str(WATCHDOG_VERIFICATION_DEADLINE_SECONDS),
        "--approval-not-after-epoch",
        str(args.approval_not_after_epoch),
    ]
    supervisor_cmd = exact_supervisor_command(
        nonce=args.nonce, source_snapshot_root=args.source_snapshot_root
    )
    preflight_cmd = " ".join(
        (
            PYTHON,
            str(CONTROL_ROOT / "scripts/build_complete_highdim_leaderboard_launch_manifest.py"),
            "--output",
            str(args.manifest),
            "--check",
            "--require-reviewed-static-readiness",
            "--launch-review-receipt",
            str(args.launch_review_receipt),
        )
    )
    launcher_argv = [
        BASH,
        str(LAUNCHER),
        "--root",
        str(args.source_snapshot_root),
        "--run-id",
        args.run_id,
        "--log-dir",
        str(expected_handoff),
        "--launch-root",
        str(args.launch_root),
        "--worker-settings",
        "/home/chakwong/.claude/settings.codex-worker.json",
        "--supervisor-cmd",
        supervisor_cmd,
        "--preflight-cmd",
        preflight_cmd,
        "--copy-cmd",
        shlex.join(preparation_cmd),
    ]
    approval_payload = {
        "schema_version": (
            "bayesfilter.complete_highdim_leaderboard.human_launch_approval.v1"
        ),
        "approval_basis": "human_approved_exact_escalated_launch_command",
        "approval_instance_id": args.approval_instance_id,
        "approval_not_after_epoch": args.approval_not_after_epoch,
        "approved_external_disclosures": [
            "restricted Codex access to the exact isolated source snapshot",
            "restricted Codex access to the two manifest-bound overlays",
            "restricted Codex access to four manifest-bound support files, with the generic Claude worker replaced by the run-specific stream-auditing wrapper",
            "restricted Codex access to the exact supervisor prompt already inside the snapshot",
            "read-only access to the pinned TensorFlow/TFP/CUDA conda runtime tree",
            "read-only access to the pinned Node/Codex/Claude runtime tree",
            "ordinary Linux system libraries, executables, proc/dev interfaces, and approved NVIDIA GPU devices required to execute the reviewed program",
            "an ephemeral private copy of existing Codex authentication state, without logging or hashing secret contents",
            "the configured Anthropic credential channel for bounded Claude reviews, without logging or hashing secret values",
            "dynamic handoff contents hidden from Codex",
            "unrelated sibling-home paths and mounted host drives hidden from Codex",
            "Claude technically inherits the isolated Codex process surface, including OS-level read/write access to the copied repository and private temporary storage plus read access to exposed read-only runtime trees, support files, system interfaces, network transport, the configured Anthropic credential channel, and the ephemeral private Codex authentication copy through inherited CODEX_HOME",
            "the bound Claude worker/settings technically permit edit and command capabilities; Claude's read-only role and single-path review are instruction and prompt contracts, not deny-by-construction tool or filesystem boundaries",
            "launch-time export validators do not independently require the complete five-file primary export set",
            "the final seal is written before all handoff aliases are locked read-only; the outer boundary writes a post-lock rehash receipt, which detects final mismatch but cannot prove no transient pre-lock write occurred",
            "the trusted GPU/XLA preflight covers the synthetic inner boundary and not the exact production outer wrapper, preparer, watchdog, finalizer, alias lock, or post-lock seal route",
            "all five limitations are accepted only for complete-highdim-leaderboard-20260711-221500 and are not repository defaults or repaired guarantees",
            "all results remain provisional and no completion or release claim is permitted until the separately bound post-run integrity audit passes",
            "the post-run audit must find zero current credential-value matches in handoff/archive bytes, no observed non-read-only Claude tool use, a passing semantic inspection receipt, structural pass, and Phase 8/9 validator exit zero with all checks passing",
        ],
        "owner_accepted_run_scoped_limitations": limitations,
        "mandatory_post_run_integrity_audit": audit,
        "codex_readable_initial_disclosure": manifest[
            "codex_readable_initial_disclosure"
        ],
        "trusted_execution_authority": [
            "GPU/XLA preflight and production GPU work",
            "Codex and Claude model/API calls",
        ],
        "run_id": args.run_id,
        "launch_root": str(args.launch_root),
        "handoff_dir_inside_outer_namespace": str(expected_handoff),
        "live_handoff_dir": str(live_handoff),
        "nonce": args.nonce,
        "manifest_path": str(args.manifest),
        "manifest_sha256": manifest_sha256,
        "launch_review_receipt": str(args.launch_review_receipt),
        "exact_wrapper_argv": expected_wrapper_argv,
        "external_handoff_launcher_argv": launcher_argv,
        "source_snapshot_root": str(args.source_snapshot_root),
        "source_inventory": str(args.source_inventory),
        "source_inventory_sha256": _sha256(args.source_inventory),
        "runtime_fingerprint": str(args.runtime_fingerprint),
        "runtime_fingerprint_sha256": _sha256(args.runtime_fingerprint),
        "final_exact_launch_command_separately_approved_by_human": True,
        "completion_or_release_authority_granted": False,
        "started_epoch": started_epoch,
        "started_monotonic": started_monotonic,
    }
    _write_exclusive(approval, approval_payload)

    fresh_preflight = args.source_snapshot_root / (
        "scripts/run_complete_highdim_leaderboard_isolation_preflight.py"
    )
    for mode, suffix in (
        ("fake-gpu-boundary", "fresh-boundary-gpu-preflight.json"),
        ("real-codex", "fresh-restricted-codex-preflight.json"),
    ):
        _require_unexpired(
            args.approval_not_after_epoch,
            action=f"post-approval {mode} model or GPU probe",
        )
        output = expected_handoff / f"{args.run_id}-{suffix}"
        remaining = _remaining_timeout(
            started_monotonic, 1800, maximum=270.0
        )
        if remaining <= 20.0:
            raise TimeoutError("insufficient common budget for restricted preflight")
        inner_timeout = min(240.0, remaining - 10.0)
        result = subprocess.run(
            [
                "/usr/bin/timeout",
                "--signal=TERM",
                "--kill-after=20s",
                f"{inner_timeout:.6f}s",
                PYTHON,
                str(fresh_preflight),
                "--mode",
                mode,
                "--workspace-root",
                str(expected_handoff),
                "--parent-run-id",
                args.run_id,
                "--parent-nonce",
                args.nonce,
                "--approval-instance-id",
                args.approval_instance_id,
                "--output",
                str(output),
            ],
            cwd=args.source_snapshot_root,
            env={**os.environ, "PATH": CONSTRAINED_PATH},
            check=False,
            timeout=remaining,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if result.returncode != 0:
            raise RuntimeError(f"fresh restricted preflight failed: {mode}")
        payload = json.loads(output.read_text(encoding="utf-8"))
        if (
            payload.get("schema_version")
            != "bayesfilter.complete_highdim_leaderboard.isolation_preflight.v3"
            or payload.get("mode") != mode
            or payload.get("preflight_pass") is not True
            or payload.get("parent_run_id") != args.run_id
            or payload.get("parent_nonce") != args.nonce
            or payload.get("approval_instance_id") != args.approval_instance_id
            or output.stat().st_mtime < started_epoch
        ):
            raise RuntimeError(f"fresh restricted preflight artifact is invalid: {mode}")

    conditional_authorization = expected_handoff / (
        f"{args.run_id}-conditional-launch-authorization.json"
    )
    _write_exclusive(
        conditional_authorization,
        {
            "schema_version": (
                "bayesfilter.complete_highdim_leaderboard."
                "conditional_launch_authorization.v1"
            ),
            "run_id": args.run_id,
            "nonce": args.nonce,
            "approval_instance_id": args.approval_instance_id,
            "human_approval_record": str(approval),
            "fresh_fake_gpu_boundary_probe_passed": True,
            "fresh_real_restricted_codex_probe_passed": True,
            "copy_and_supervisor_launch_condition_satisfied": True,
            "scientific_or_release_authority_granted": False,
            "post_run_integrity_audit_required_before_completion_or_release": True,
            "owner_accepted_run_scoped_limitations": (
                OWNER_ACCEPTED_RUN_SCOPED_LIMITATION_IDS
            ),
        },
    )

    environment = {
        **os.environ,
        "PATH": CONSTRAINED_PATH,
        "CONDA_PREFIX": "/home/chakwong/anaconda3/envs/tf-gpu",
    }
    _require_unexpired(
        args.approval_not_after_epoch, action="real copy and detached supervisor launch"
    )
    process = subprocess.Popen(
        launcher_argv,
        cwd=args.source_snapshot_root,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    handoff_exit_code: int
    launcher_wait_deadline = started_monotonic + NAMESPACE_OUTER_DEADLINE_SECONDS
    try:
        stdout, stderr = process.communicate(
            timeout=_remaining_timeout(
                started_monotonic,
                NAMESPACE_OUTER_DEADLINE_SECONDS,
                maximum=900.0,
            )
        )
        handoff_exit_code = int(process.returncode or 0)
    except subprocess.TimeoutExpired:
        _kill_group(process, deadline_monotonic=launcher_wait_deadline)
        remaining = launcher_wait_deadline - time.monotonic()
        if remaining > 0.0:
            try:
                stdout, stderr = process.communicate(timeout=remaining)
            except subprocess.TimeoutExpired:
                stdout, stderr = "", "launcher did not close before common deadline"
        else:
            stdout, stderr = "", "launcher exhausted the common deadline"
        handoff_exit_code = 124
    handoff_result = expected_handoff / f"{args.run_id}-launcher-handoff.json"
    _write_exclusive(
        handoff_result,
        {
            "schema_version": (
                "bayesfilter.complete_highdim_leaderboard.launch_handoff.v1"
            ),
            "run_id": args.run_id,
            "exit_code": handoff_exit_code,
            "stdout": stdout,
            "stderr": stderr,
            "handoff_ready": (expected_handoff / f"{args.run_id}.handoff.ready").is_file(),
        },
    )
    if handoff_exit_code != 0:
        launch_failure = expected_handoff / f"{args.run_id}-launcher-failure.json"
        _write_exclusive(
            launch_failure,
            {
                "schema_version": (
                    "bayesfilter.complete_highdim_leaderboard.launcher_failure.v1"
                ),
                "run_id": args.run_id,
                "launcher_exit_code": handoff_exit_code,
                "timestamp_epoch": time.time(),
            },
        )
    producer_specs = (
        (
            expected_handoff / f"{args.run_id}-supervisor-producer.json",
            "detached_supervisor",
            args.source_snapshot_root
            / "scripts/complete_highdim_leaderboard_overnight_supervisor.py",
            args.source_snapshot_root
            / "scripts/complete_highdim_leaderboard_overnight_supervisor.py",
        ),
        (
            expected_handoff / f"{args.run_id}-watchdog-producer.json",
            "primary_export_verification_watchdog",
            args.source_snapshot_root
            / "scripts/complete_highdim_leaderboard_watchdog.py",
            args.source_snapshot_root
            / "scripts/complete_highdim_leaderboard_watchdog.py",
        ),
    )
    seal_deadline = started_monotonic + PRODUCER_CLOSE_DEADLINE_SECONDS
    producers: list[dict | None] = []
    while time.monotonic() < seal_deadline:
        producers = [
            _load_producer(
                path,
                run_id=args.run_id,
                role=role,
                expected_command_path=expected_command_path,
                expected_command_hash_source=expected_command_hash_source,
            )
            for path, role, expected_command_path, expected_command_hash_source in producer_specs
        ]
        if all(producer is not None for producer in producers):
            break
        remaining = seal_deadline - time.monotonic()
        if remaining > 0.0:
            time.sleep(min(0.25, remaining))
    descriptors_valid = all(producer is not None for producer in producers)
    producers_closed = False
    if descriptors_valid:
        while time.monotonic() < seal_deadline:
            if all(
                _producer_closed(producer) for producer in producers if producer
            ) and _whole_private_namespace_quiescent():
                producers_closed = True
                break
            remaining = seal_deadline - time.monotonic()
            if remaining > 0.0:
                time.sleep(min(2.0, remaining))
    if not producers_closed:
        for producer in producers:
            if producer is None:
                continue
            try:
                os.killpg(producer["pgid"], signal.SIGKILL)
            except ProcessLookupError:
                pass
        for pid in _extra_live_namespace_pids():
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        handoff_exit_code = handoff_exit_code or 124

    program_exit_code = handoff_exit_code
    terminal = expected_handoff / f"{args.run_id}-terminal-status.json"
    verification = expected_handoff / f"{args.run_id}-post-export-verification.json"
    watchdog_status = expected_handoff / f"{args.run_id}-watchdog-status.json"
    try:
        watchdog_payload = json.loads(watchdog_status.read_text(encoding="utf-8"))
        watchdog_verified = bool(
            watchdog_payload.get("schema_version")
            == "bayesfilter.complete_highdim_leaderboard.watchdog.v3"
            and watchdog_payload.get("run_id") == args.run_id
            and watchdog_payload.get("primary_export_verified") is True
            and watchdog_payload.get("fallback_export_allowed") is False
            and watchdog_payload.get("verification_exit_code") == 0
        )
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        watchdog_verified = False
    if handoff_exit_code == 0 and terminal.is_file():
        try:
            terminal_payload = json.loads(terminal.read_text(encoding="utf-8"))
            program_exit_code = int(terminal_payload["codex_exit_code"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            program_exit_code = 96
    elif handoff_exit_code == 0:
        program_exit_code = 96
    unapproved_exports = _unapproved_export_artifacts(expected_handoff, args.run_id)
    primary_export_only_observed = not unapproved_exports
    if (
        not verification.is_file()
        or not watchdog_verified
        or not producers_closed
        or not primary_export_only_observed
    ):
        program_exit_code = 98

    outcome = expected_handoff / f"{args.run_id}-foreground-outcome.json"
    _write_exclusive(
        outcome,
        {
            "schema_version": (
                "bayesfilter.complete_highdim_leaderboard.foreground_outcome.v2"
            ),
            "run_id": args.run_id,
            "handoff_exit_code": handoff_exit_code,
            "program_exit_code": program_exit_code,
            "terminal_present": terminal.is_file(),
            "primary_export_present": (
                expected_handoff / f"{args.run_id}-primary-export-sha256.json"
            ).is_file(),
            "fallback_export_allowed": False,
            "fallback_export_present": any(
                "fallback" in name for name in unapproved_exports
            ),
            "primary_export_only_observed": primary_export_only_observed,
            "unapproved_export_artifacts": unapproved_exports,
            "post_export_verification_present": verification.is_file(),
            "watchdog_primary_verification_passed": watchdog_verified,
            "producer_descriptors_valid": descriptors_valid,
            "producers_closed_before_outcome": producers_closed,
            "whole_outer_pid_namespace_quiescent_before_outcome": producers_closed,
        },
    )
    if program_exit_code == 0:
        print(stdout, end="")
    return program_exit_code


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:  # noqa: BLE001
        print(f"LAUNCH_FATAL: {error}", file=sys.stderr)
        raise SystemExit(2) from error
