#!/usr/bin/env python3
"""Generate Phase A3 scalar-LGSSM oracle and predictive-statistics artifacts."""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import re
import struct
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PHASE_DIR = Path("docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a3")
PLAN_PATH = Path(
    "docs/plans/bayesfilter-ssl-lstm-completion-phase-a3-forecast-oracle-statistics-subplan-2026-07-11.md"
)
RESULT_PATH = Path(
    "docs/plans/bayesfilter-ssl-lstm-completion-phase-a3-forecast-oracle-statistics-result-2026-07-11.md"
)
BOUNDARY_PATH = PHASE_DIR / "pre-run-boundary.json"
FIXTURE_PATH = PHASE_DIR / "fixture-contract.json"
HARNESS_ANCHOR_PATH = PHASE_DIR / "harness-review-anchor.json"
HARNESS_REVIEW_PATH = Path(
    "docs/reviews/bayesfilter-ssl-lstm-completion-phase-a3-harness-codex-substitute-review-2026-07-13.md"
)
CPU_REFERENCE_PATH = PHASE_DIR / "oracle-cpu-reference.json"
CPU_VERIFY_RECEIPT_PATH = PHASE_DIR / "oracle-cpu-reference-verify.log"
CPU_GENERATION_TRACE_PATH = PHASE_DIR / "oracle-cpu-generation-write-trace.log"
CPU_VERIFICATION_TRACE_PATH = PHASE_DIR / "oracle-cpu-verification-write-trace.log"
PREDICTIVE_SOURCE = Path("bayesfilter/inference/predictive_equivalence.py")
ORACLE_SOURCE = Path("bayesfilter/testing/scalar_lgssm_forecast_oracle.py")
PREDICTIVE_TEST = Path("tests/test_predictive_equivalence.py")
ORACLE_TEST = Path("tests/test_scalar_lgssm_forecast_oracle.py")
GENERATOR_PATH = Path(
    "docs/benchmarks/benchmark_ssl_lstm_completion_phase_a3_forecast_oracle_2026_07_13.py"
)
VERIFIER_PATH = Path(
    "docs/benchmarks/verify_ssl_lstm_completion_phase_a3_forecast_oracle_2026_07_13.py"
)

HEAD_SHA256 = "a644d29c5c2fd09a0deb3a7b5212799ff1fcb163"
PLAN_SHA256 = "67ee503a15f5e7a81ca2a37e52cc6b60264c1cff89ff5cff1a9fddd3187161c4"
EXPECTED_SOURCE_HASHES = {
    PREDICTIVE_SOURCE: "99ddaa1dcb15e9f3ec7a5a18f96ebd0f656848c40ea76c896b387cace294bc16",
    ORACLE_SOURCE: "74889d699e3575ee163c64d9a67325f0376e161106e9b36fb6b61453c3a5eb43",
    PREDICTIVE_TEST: "5e6a137c12b3131c8ff7471d74abd4a877777ef6432a2c51f5c62cceedf9290d",
    ORACLE_TEST: "977134cbc92b63ca6d8dab7a1e6ca25eb58137cb27430518a1aacc120cecfab8",
}
CPU_GPU_TOLERANCE_MULTIPLIER = 8192
FAMILY_CODES = {
    "terminal_standard_normal": 3101,
    "process_standard_normal": 3102,
    "observation_standard_normal": 3103,
}

CPU_SCHEMA = "bayesfilter.ssl_lstm_completion.phase_a3_cpu_oracle.v1"
GPU_SCHEMA = "bayesfilter.ssl_lstm_completion.phase_a3_gpu_xla_oracle.v1"
CPU_STATUS = "A3_CPU_ORACLE_PASSED"
GPU_STATUS = "A3_GPU_XLA_ORACLE_PASSED"

NONCLAIMS = [
    "A3 scalar-LGSSM oracle and predictive-statistics engineering evidence only",
    "not SSL-LSTM predictive equivalence or calibration evidence",
    "not posterior correctness or parameter agreement evidence",
    "not HMC or NeuTra validity, readiness, training, or comparison evidence",
    "not calibrated A4 margins, bandwidths, blocks, bootstrap counts, or seeds",
    "not performance, product, public API, default, or release evidence",
    "not a sampler ranking, model-adequacy result, or scientific claim",
]

CHECK_NAMES = (
    "analytic_formula_exact",
    "analytic_covariance_valid",
    "direct_simulation_replay",
    "monte_carlo_oracle_agreement",
    "summary_statistics",
    "standardization",
    "quadratic_mmd_roles",
    "signed_u_form_preserved",
    "common_random_numbers_excluded",
    "cross_chain_schedule",
    "cross_chain_inference_admission",
    "cross_chain_null_coverage",
    "hierarchical_indices",
    "joint_alpha_allocation",
    "simultaneous_intervals",
    "controlled_alternatives",
    "decision_fail_closed",
    "fixture_binding",
    "source_binding",
    "compiler_hlo",
    "device_placement",
)


class ContractError(RuntimeError):
    """Raised when an A3 artifact cannot satisfy the reviewed contract."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256(path: Path) -> str:
    return _sha256_bytes((ROOT / path).read_bytes())


def _evidence_signature(payload: dict[str, Any]) -> str:
    projection = copy.deepcopy(payload)
    projection.pop("evidence_signature", None)
    projection.pop("created_at_utc", None)
    manifest = projection.get("run_manifest")
    if isinstance(manifest, dict):
        for field in ("started_at_utc", "completed_at_utc", "wall_time_seconds"):
            manifest.pop(field, None)
    return _sha256_bytes(_canonical_bytes(projection))


def _strict_load(path: Path) -> dict[str, Any]:
    def pairs_hook(rows: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in rows:
            if key in result:
                raise ContractError(f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ContractError(f"nonfinite JSON constant {value!r} in {path}")

    value = json.loads(
        (ROOT / path).read_text(encoding="utf-8"),
        object_pairs_hook=pairs_hook,
        parse_constant=reject_constant,
    )
    if not isinstance(value, dict):
        raise ContractError(f"{path} must contain a JSON object")
    return value


_TRACE_INTERPRETER = "/home/ubuntu/anaconda3/envs/tfgpu/bin/python"
_TRACE_CALL = re.compile(
    r"^(?P<pid>\d+)\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\((?P<args>.*)\)\s+=\s+(?P<result>.*)$"
)
_TRACE_UNFINISHED = re.compile(
    r"^(?P<pid>\d+)\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\((?P<prefix>.*)<unfinished \.\.\.>$"
)
_TRACE_RESUMED = re.compile(
    r"^(?P<pid>\d+)\s+<\.\.\. (?P<name>[A-Za-z_][A-Za-z0-9_]*) resumed>(?P<suffix>.*)$"
)
_TRACE_SIGNAL = re.compile(r"^\d+\s+--- SIG[A-Z0-9]+ \{.*\} ---$")
_TRACE_LIFECYCLE = re.compile(
    r"^\d+\s+\+\+\+ (?:exited with \d+|killed by SIG[A-Z0-9]+(?: \(core dumped\))?) \+\+\+$"
)
_WRITE_FLAGS = frozenset(
    {"O_WRONLY", "O_RDWR", "O_CREAT", "O_TRUNC", "O_APPEND", "O_TMPFILE"}
)
_READ_ONLY_FILE_SYSCALLS = frozenset(
    {
        "access", "chdir", "faccessat", "faccessat2", "fchdir", "getcwd",
        "getxattr", "lgetxattr", "listxattr", "llistxattr", "lstat",
        "name_to_handle_at", "newfstatat", "readlink", "readlinkat", "stat",
        "statfs", "statfs64", "statx",
    }
)
_PATH_MUTATION_ARGUMENTS = {
    "chmod": (0,), "chown": (0,), "fchmodat": (1,), "fchmodat2": (1,),
    "fchownat": (1,), "futimesat": (1,), "lchown": (0,), "link": (0, 1),
    "linkat": (1, 3), "lremovexattr": (0,), "lsetxattr": (0,),
    "mkdir": (0,), "mkdirat": (1,), "mknod": (0,), "mknodat": (1,),
    "removexattr": (0,), "rename": (0, 1), "renameat": (1, 3),
    "renameat2": (1, 3), "rmdir": (0,), "setxattr": (0,),
    "symlink": (1,), "symlinkat": (2,), "truncate": (0,), "unlink": (0,),
    "unlinkat": (1,), "utime": (0,), "utimensat": (1,), "utimes": (0,),
}
_DESCRIPTOR_MUTATION_DESTINATION = {
    "copy_file_range": 2,
    "fallocate": 0,
    "fchmod": 0,
    "fchown": 0,
    "fremovexattr": 0,
    "fsetxattr": 0,
    "ftruncate": 0,
    "pwrite64": 0,
    "pwritev": 0,
    "pwritev2": 0,
    "sendfile": 0,
    "sendfile64": 0,
    "splice": 2,
    "tee": 1,
    "vmsplice": 0,
    "write": 0,
    "writev": 0,
}
_WRITABLE_MMAP_SYSCALLS = frozenset({"mmap", "mmap2"})
_MAPPED_WRITE_FLUSH_SYSCALLS = frozenset({"msync"})
_PROCESS_TERMINATION_SYSCALLS = frozenset({"exit", "exit_group"})
_FORBIDDEN_NAMESPACE_MUTATIONS = frozenset(
    {
        "chroot", "fspick", "fsopen", "fsmount", "move_mount", "mount",
        "open_tree", "pivot_root", "swapoff", "swapon", "umount", "umount2",
    }
)


def _trace_contract(path: Path) -> dict[str, Any]:
    contracts = {
        CPU_GENERATION_TRACE_PATH.name: {
            "argv": (
                _TRACE_INTERPRETER,
                GENERATOR_PATH.as_posix(),
                "--mode", "cpu-reference",
                "--fixture", FIXTURE_PATH.as_posix(),
                "--output", CPU_REFERENCE_PATH.as_posix(),
                "--log-path", (PHASE_DIR / "oracle-cpu-reference.log").as_posix(),
            ),
            "writes": (CPU_REFERENCE_PATH, PHASE_DIR / "oracle-cpu-reference.log"),
        },
        CPU_VERIFICATION_TRACE_PATH.name: {
            "argv": (
                _TRACE_INTERPRETER,
                VERIFIER_PATH.as_posix(),
                "--artifact", CPU_REFERENCE_PATH.as_posix(),
                "--log-path", CPU_VERIFY_RECEIPT_PATH.as_posix(),
            ),
            "writes": (CPU_VERIFY_RECEIPT_PATH,),
        },
    }
    try:
        return contracts[path.name]
    except KeyError as exc:
        raise ContractError(f"unmapped CPU replay-authority trace: {path}") from exc


def _split_trace_arguments(arguments: str) -> list[str]:
    result: list[str] = []
    start = 0
    quote = False
    escaped = False
    depth = 0
    for index, character in enumerate(arguments):
        if quote:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                quote = False
            continue
        if character == '"':
            quote = True
        elif character in "([{":
            depth += 1
        elif character in ")]}":
            depth = max(0, depth - 1)
        elif character == "," and depth == 0:
            result.append(arguments[start:index].strip())
            start = index + 1
    result.append(arguments[start:].strip())
    return result


def _trace_records(lines: list[str]) -> list[dict[str, str]]:
    pending: dict[str, tuple[str, str]] = {}
    records: list[dict[str, str]] = []
    for raw_line in lines:
        if _TRACE_SIGNAL.fullmatch(raw_line) or _TRACE_LIFECYCLE.fullmatch(raw_line):
            continue
        unfinished = _TRACE_UNFINISHED.fullmatch(raw_line)
        if unfinished is not None:
            pid = unfinished.group("pid")
            if pid in pending:
                raise ContractError(f"nested unfinished trace record for PID {pid}")
            pending[pid] = (unfinished.group("name"), unfinished.group("prefix"))
            continue
        resumed = _TRACE_RESUMED.fullmatch(raw_line)
        if resumed is not None:
            pid = resumed.group("pid")
            if pid not in pending:
                raise ContractError(f"resumed trace record lacks pending call: {raw_line}")
            name, prefix = pending.pop(pid)
            if resumed.group("name") != name:
                raise ContractError(f"resumed trace syscall mismatch for PID {pid}")
            raw_line = f"{pid} {name}({prefix}{resumed.group('suffix')}"
        elif "<unfinished ...>" in raw_line or " resumed>" in raw_line:
            raise ContractError(f"malformed split trace record: {raw_line}")
        match = _TRACE_CALL.fullmatch(raw_line)
        if match is None:
            raise ContractError(f"unparsed trace record: {raw_line}")
        if (
            match.group("name") not in _DESCRIPTOR_MUTATION_DESTINATION
            and re.search(r'"(?:[^"\\]|\\.)*"\.\.\.', raw_line)
        ) or re.search(r"\d+</[^>]*\.\.\.[^>]*>", raw_line):
            raise ContractError(f"truncated trace value: {raw_line}")
        records.append(
            {
                "pid": match.group("pid"),
                "name": match.group("name"),
                "arguments": match.group("args"),
                "result": match.group("result").strip(),
                "line": raw_line,
            }
        )
    if pending:
        raise ContractError(f"trace ended with pending calls for PIDs {sorted(pending)}")
    return records


def _successful_trace_call(result: str) -> bool:
    if result.startswith("-1 "):
        return False
    if result == "?" or result.startswith("? "):
        raise ContractError(f"trace call has indeterminate result {result!r}")
    return True


def _trace_string(token: str) -> str:
    try:
        value = ast.literal_eval(token.strip())
    except (SyntaxError, ValueError) as exc:
        raise ContractError(f"invalid strace string token {token!r}") from exc
    if not isinstance(value, str):
        raise ContractError(f"strace token is not a string: {token!r}")
    return value


def _trace_argv(token: str) -> tuple[str, ...]:
    token = token.strip()
    if not token.startswith("[") or not token.endswith("]"):
        raise ContractError(f"strace argv is not a complete array: {token!r}")
    body = token[1:-1].strip()
    if not body:
        return ()
    return tuple(_trace_string(item) for item in _split_trace_arguments(body))


def _fd_annotation(token: str) -> tuple[int, str] | None:
    match = re.match(r"^(?P<fd>-?\d+)<(?P<target>[^>]*)>", token.strip())
    if match is None:
        return None
    return int(match.group("fd")), match.group("target")


def _resolved_exec(record: dict[str, str]) -> tuple[str, tuple[str, ...]]:
    arguments = _split_trace_arguments(record["arguments"])
    if record["name"] == "execve" and len(arguments) >= 2:
        executable = _trace_string(arguments[0])
        argv_token = arguments[1]
    elif record["name"] == "execveat" and len(arguments) >= 3:
        executable = _trace_string(arguments[1])
        if not executable:
            annotation = _fd_annotation(arguments[0])
            if annotation is None or not annotation[1].startswith("/"):
                raise ContractError("execveat AT_EMPTY_PATH lacks a resolved executable")
            executable = annotation[1]
        elif not executable.startswith("/"):
            annotation = _fd_annotation(arguments[0])
            if annotation is None or not annotation[1].startswith("/"):
                raise ContractError("relative execveat executable lacks resolved dirfd")
            executable = str(Path(annotation[1]) / executable)
        argv_token = arguments[2]
    else:
        raise ContractError(f"malformed execution trace record: {record['line']}")
    return str(Path(executable).resolve(strict=False)), _trace_argv(argv_token)


def _authenticate_trace_execution(
    records: list[dict[str, str]], expected_argv: tuple[str, ...]
) -> tuple[str, int]:
    if not records:
        raise ContractError("trace contains no syscall records")
    first = records[0]
    root_pid = first["pid"]
    if (
        first["name"] not in {"execve", "execveat"}
        or not _successful_trace_call(first["result"])
    ):
        raise ContractError("trace does not begin with the successful root execution")
    root_executions = [
        row
        for row in records
        if row["pid"] == root_pid
        and row["name"] in {"execve", "execveat"}
        and _successful_trace_call(row["result"])
    ]
    if len(root_executions) != 1:
        raise ContractError("trace must contain exactly one successful root execution")
    executable, argv = _resolved_exec(root_executions[0])
    expected_executable = str(Path(expected_argv[0]).resolve(strict=False))
    if executable != expected_executable or argv != expected_argv:
        raise ContractError("trace root execution does not match the reviewed role argv")
    child_executions = sum(
        row["pid"] != root_pid
        and row["name"] in {"execve", "execveat"}
        and _successful_trace_call(row["result"])
        for row in records
    )
    root_terminations = [
        row
        for row in records
        if row["pid"] == root_pid and row["name"] in _PROCESS_TERMINATION_SYSCALLS
    ]
    if len(root_terminations) != 1 or _trace_exit_code(root_terminations[0]) != 0:
        raise ContractError("trace root process did not terminate exactly once with exit code zero")
    return root_pid, child_executions


def _inside(path: Path, roots: tuple[Path, ...]) -> bool:
    resolved = path.resolve(strict=False)
    return any(resolved == root or root in resolved.parents for root in roots)


def _resolved_path_argument(arguments: list[str], index: int) -> Path:
    if index >= len(arguments):
        raise ContractError("mutation trace lacks a required path argument")
    value = _trace_string(arguments[index])
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate.resolve(strict=False)
    if index > 0:
        previous = arguments[index - 1].strip()
        match = re.match(r"^(?:AT_FDCWD|-?\d+)<(?P<target>[^>]*)>", previous)
        if match is not None:
            target = match.group("target")
            if not target.startswith("/"):
                raise ContractError("relative mutation dirfd lacks an absolute -yy path")
            return (Path(target) / candidate).resolve(strict=False)
        if previous == "AT_FDCWD" or re.fullmatch(r"-?\d+", previous):
            raise ContractError("relative mutation dirfd lacks a -yy annotation")
    raise ContractError(f"relative mutation path lacks a resolved dirfd: {value!r}")


def _write_open_destination(
    name: str, arguments: list[str], result: str
) -> tuple[Path, str] | None:
    if name == "creat":
        flags = "O_WRONLY|O_CREAT|O_TRUNC"
    elif name == "open" and len(arguments) >= 2:
        flags = arguments[1]
    elif name in {"openat", "openat2"} and len(arguments) >= 3:
        flags = arguments[2]
    else:
        return None
    if name != "creat" and not any(flag in flags for flag in _WRITE_FLAGS):
        return None
    annotation = _fd_annotation(result)
    if annotation is None or not annotation[1].startswith("/"):
        raise ContractError("write open lacks a resolved -yy destination")
    return Path(annotation[1]).resolve(strict=False), flags


def _is_null_device_result(result: str) -> bool:
    return re.fullmatch(r"-?\d+</dev/null<char 1:3>>", result.strip()) is not None


def _is_authenticated_thread_name_path(path: Path, root_pid: str) -> bool:
    parts = path.resolve(strict=False).parts
    return (
        len(parts) == 6
        and parts[:2] == ("/", "proc")
        and parts[2] == root_pid
        and parts[3] == "task"
        and parts[4].isdigit()
        and parts[5] == "comm"
    )


def _descriptor_destination(arguments: list[str], index: int) -> tuple[int, str]:
    if index >= len(arguments):
        raise ContractError("descriptor mutation lacks its destination argument")
    annotation = _fd_annotation(arguments[index])
    if annotation is None:
        raise ContractError("descriptor mutation destination lacks -yy annotation")
    return annotation


def _trace_exit_code(record: dict[str, str]) -> int:
    arguments = _split_trace_arguments(record["arguments"])
    if (
        record["name"] not in _PROCESS_TERMINATION_SYSCALLS
        or len(arguments) != 1
        or not re.fullmatch(r"-?\d+", arguments[0])
        or record["result"] != "?"
    ):
        raise ContractError(f"malformed process-termination trace: {record['line']}")
    return int(arguments[0])


def _audit_trace(path: Path) -> str:
    absolute = ROOT / path
    if not absolute.is_file():
        raise ContractError(f"required CPU replay-authority trace missing: {path}")
    lines = absolute.read_text(encoding="utf-8", errors="strict").splitlines()
    if not lines:
        raise ContractError(f"empty CPU replay-authority trace: {path}")
    contract = _trace_contract(path)
    records = _trace_records(lines)
    root_pid, child_execution_count = _authenticate_trace_execution(
        records, contract["argv"]
    )
    if child_execution_count < 0 or not root_pid:
        raise ContractError("invalid child-execution accounting")
    temporary_roots = tuple(
        item.resolve(strict=False)
        for item in (Path("/tmp/bayesfilter-a3-pycache"), Path("/tmp/bayesfilter-a3-tmp"))
    )
    expected = {
        (ROOT / item).resolve(strict=False) for item in contract["writes"]
    }
    observed_repository_writes: set[Path] = set()
    for record in records:
        name = record["name"]
        arguments = _split_trace_arguments(record["arguments"])
        result = record["result"]
        if name in _PROCESS_TERMINATION_SYSCALLS:
            _trace_exit_code(record)
            continue
        if not _successful_trace_call(result):
            continue
        if name in {"execve", "execveat"} or name in _READ_ONLY_FILE_SYSCALLS:
            continue
        if name in {"open", "openat", "openat2", "creat"}:
            destination = _write_open_destination(name, arguments, result)
            if destination is None:
                continue
            target, flags = destination
            if _is_null_device_result(result) or target == Path("/dev/null") or (
                target.parts[:2] == ("/", "dev")
                and all(
                    flag not in flags
                    for flag in ("O_CREAT", "O_TRUNC", "O_APPEND", "O_TMPFILE")
                )
            ):
                continue
            if _is_authenticated_thread_name_path(target, root_pid):
                continue
            if _inside(target, temporary_roots):
                continue
            if target not in expected:
                raise ContractError(f"unexpected CPU replay-authority write: {record['line']}")
            observed_repository_writes.add(target)
            continue
        if name in _PATH_MUTATION_ARGUMENTS:
            targets = tuple(
                _resolved_path_argument(arguments, index)
                for index in _PATH_MUTATION_ARGUMENTS[name]
            )
            if not targets or not all(_inside(target, temporary_roots) for target in targets):
                raise ContractError(f"path mutation escaped reviewed temp roots: {record['line']}")
            continue
        if name in _DESCRIPTOR_MUTATION_DESTINATION:
            fd, annotation = _descriptor_destination(
                arguments, _DESCRIPTOR_MUTATION_DESTINATION[name]
            )
            if annotation.startswith("/"):
                target = Path(annotation).resolve(strict=False)
                if _is_authenticated_thread_name_path(target, root_pid):
                    continue
                if _inside(target, temporary_roots):
                    continue
                if target in expected:
                    observed_repository_writes.add(target)
                    continue
                if target.parts[:2] == ("/", "dev"):
                    continue
            elif annotation.startswith(("pipe:[", "socket:[", "anon_inode:")):
                continue
            raise ContractError(f"descriptor mutation escaped reviewed outputs: {record['line']}")
        if name in _WRITABLE_MMAP_SYSCALLS:
            if len(arguments) < 5:
                raise ContractError(f"malformed mmap trace record: {record['line']}")
            writable_shared = (
                "PROT_WRITE" in arguments[2] and "MAP_SHARED" in arguments[3]
            )
            anonymous = "MAP_ANONYMOUS" in arguments[3] or "MAP_ANON" in arguments[3]
            if not writable_shared or anonymous:
                continue
            fd, annotation = _descriptor_destination(arguments, 4)
            if annotation.startswith("/"):
                target = Path(annotation).resolve(strict=False)
                if _inside(target, temporary_roots) or target in expected:
                    if target in expected:
                        observed_repository_writes.add(target)
                    continue
                if target.parts[:2] == ("/", "dev"):
                    continue
            raise ContractError(f"writable shared mmap escaped reviewed outputs: {record['line']}")
        if name in _MAPPED_WRITE_FLUSH_SYSCALLS:
            continue
        if name in _FORBIDDEN_NAMESPACE_MUTATIONS:
            raise ContractError(f"forbidden namespace mutation: {record['line']}")
        raise ContractError(f"unclassified successful file syscall: {record['line']}")
    missing = sorted(str(item) for item in expected - observed_repository_writes)
    if missing:
        raise ContractError(f"trace lacks expected successful write opens: {missing}")
    return _sha256(path)


def _verified_cpu_replay_authority(path: Path) -> tuple[dict[str, Any], str]:
    if path != CPU_REFERENCE_PATH:
        raise ContractError("GPU mode requires the canonical CPU replay-authority path")
    for required in (
        CPU_REFERENCE_PATH,
        CPU_VERIFY_RECEIPT_PATH,
        CPU_GENERATION_TRACE_PATH,
        CPU_VERIFICATION_TRACE_PATH,
    ):
        if not (ROOT / required).is_file():
            raise ContractError(f"required CPU replay-authority chain member missing: {required}")
    payload = _strict_load(CPU_REFERENCE_PATH)
    if (ROOT / CPU_REFERENCE_PATH).read_bytes() != _canonical_bytes(payload) + b"\n":
        raise ContractError("CPU replay-authority artifact is not canonical JSON")
    if (
        payload.get("schema_version") != CPU_SCHEMA
        or payload.get("status") != CPU_STATUS
        or payload.get("evidence_signature") != _evidence_signature(payload)
    ):
        raise ContractError("CPU replay authority identity/signature mismatch")
    artifact_sha256 = _sha256(CPU_REFERENCE_PATH)
    generation_trace_sha256 = _audit_trace(CPU_GENERATION_TRACE_PATH)
    receipt = _strict_load(CPU_VERIFY_RECEIPT_PATH)
    if (ROOT / CPU_VERIFY_RECEIPT_PATH).read_bytes() != _canonical_bytes(receipt) + b"\n":
        raise ContractError("CPU verification receipt is not canonical JSON")
    expected_receipt = {
        "status": "A3_RUNTIME_ARTIFACT_VERIFIED",
        "artifact_sha256": artifact_sha256,
        "evidence_signature": payload["evidence_signature"],
        "generation_trace_sha256": generation_trace_sha256,
    }
    if receipt != expected_receipt:
        raise ContractError("CPU verification receipt does not bind the current replay authority")
    _audit_trace(CPU_VERIFICATION_TRACE_PATH)
    return payload, artifact_sha256


def _strict_write(path: Path, payload: dict[str, Any]) -> None:
    absolute = ROOT / path
    absolute.parent.mkdir(parents=True, exist_ok=True)
    absolute.write_bytes(_canonical_bytes(payload) + b"\n")


def _require_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        missing = sorted(expected - set(value))
        extra = sorted(set(value) - expected)
        raise ContractError(f"{label} keys differ; missing={missing}, extra={extra}")


def _verify_signed_input(path: Path, *, schema: str, status: str) -> dict[str, Any]:
    payload = _strict_load(path)
    if (ROOT / path).read_bytes() != _canonical_bytes(payload) + b"\n":
        raise ContractError(f"fixed input is not canonical JSON: {path}")
    if payload.get("schema_version") != schema or payload.get("status") != status:
        raise ContractError(f"fixed input identity/status mismatch: {path}")
    if payload.get("evidence_signature") != _evidence_signature(payload):
        raise ContractError(f"fixed input evidence signature mismatch: {path}")
    return payload


def _semantic_contract_sha256(path: Path, excluded_fields: tuple[str, ...]) -> str:
    projection = copy.deepcopy(_strict_load(path))
    for field in excluded_fields:
        projection.pop(field, None)
    return _sha256_bytes(_canonical_bytes(projection))


def _verify_harness_anchor() -> str:
    anchor = _verify_signed_input(
        HARNESS_ANCHOR_PATH,
        schema="bayesfilter.ssl_lstm_completion.phase_a3_harness_review_anchor.v1",
        status="A3_HARNESS_REVIEW_ANCHOR_FROZEN",
    )
    expected_keys = {
        "schema_version", "status", "created_at_utc", "review_class", "verdict",
        "reviewed_files", "boundary_semantic_sha256", "fixture_semantic_sha256",
        "review_binding", "evidence_signature", "nonclaims",
    }
    if set(anchor) != expected_keys:
        raise ContractError("harness review anchor fields differ")
    expected_files = [
        {"path": GENERATOR_PATH.as_posix(), "sha256": _sha256(GENERATOR_PATH)},
        {"path": VERIFIER_PATH.as_posix(), "sha256": _sha256(VERIFIER_PATH)},
    ]
    if (
        anchor["review_class"] != "CODEX_SUBSTITUTE_REVIEW_WEAKER_THAN_CLAUDE"
        or anchor["verdict"] != "AGREE"
        or anchor["reviewed_files"] != expected_files
        or anchor["nonclaims"] != NONCLAIMS
    ):
        raise ContractError("harness review anchor verdict or file hashes differ")
    review_binding = anchor["review_binding"]
    expected_review_binding = {
        "path": HARNESS_REVIEW_PATH.as_posix(),
        "sha256": _sha256(HARNESS_REVIEW_PATH),
    }
    if review_binding != expected_review_binding:
        raise ContractError("harness review-record binding differs")
    boundary_semantic_sha256 = _semantic_contract_sha256(
        BOUNDARY_PATH,
        ("created_at_utc", "evidence_signature", "harness_review_anchor_sha256"),
    )
    fixture_semantic_sha256 = _semantic_contract_sha256(
        FIXTURE_PATH,
        ("created_at_utc", "evidence_signature", "boundary_sha256"),
    )
    if (
        anchor["boundary_semantic_sha256"] != boundary_semantic_sha256
        or anchor["fixture_semantic_sha256"] != fixture_semantic_sha256
    ):
        raise ContractError("harness review anchor contract semantics differ")
    review_text = (ROOT / HARNESS_REVIEW_PATH).read_text(encoding="utf-8")
    required_review_text = (
        "CODEX_SUBSTITUTE_REVIEW",
        "explicitly weaker than Claude",
        expected_files[0]["sha256"],
        expected_files[1]["sha256"],
        boundary_semantic_sha256,
        fixture_semantic_sha256,
        "VERDICT: AGREE",
    )
    if not all(fragment in review_text for fragment in required_review_text):
        raise ContractError("harness review record does not bind the agreed exact hashes")
    return _sha256(HARNESS_ANCHOR_PATH)


def _git(*arguments: str) -> str:
    environment = dict(os.environ)
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    return subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    ).stdout


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not_installed"


def _float_values(value: Any) -> list[float]:
    import tensorflow as tf

    tensor = tf.convert_to_tensor(value, dtype=tf.float64)
    values = [float(item) for item in tf.reshape(tensor, [-1])]
    if not all(math.isfinite(item) for item in values):
        raise ContractError("nonfinite tensor cannot enter an artifact")
    return values


def _tensor_row(name: str, value: Any) -> dict[str, Any]:
    import tensorflow as tf

    tensor = tf.convert_to_tensor(value, dtype=tf.float64)
    values = _float_values(tensor)
    raw = b"".join(struct.pack("<d", item) for item in values)
    return {
        "name": name,
        "dtype": "float64",
        "shape": [int(size) for size in tensor.shape],
        "values_hex": [item.hex() for item in values],
        "raw_little_endian_sha256": _sha256_bytes(raw),
    }


def _bool_tensor_row(name: str, value: Any) -> dict[str, Any]:
    import tensorflow as tf

    tensor = tf.convert_to_tensor(value, dtype=tf.bool)
    values = [bool(item) for item in tf.reshape(tensor, [-1])]
    raw = bytes(int(item) for item in values)
    return {
        "name": name,
        "dtype": "bool",
        "shape": [int(size) for size in tensor.shape],
        "values": values,
        "raw_little_endian_sha256": _sha256_bytes(raw),
    }


def _int_tensor_row(name: str, value: Any) -> dict[str, Any]:
    import tensorflow as tf

    tensor = tf.convert_to_tensor(value, dtype=tf.int32)
    values = [int(item) for item in tf.reshape(tensor, [-1])]
    raw = b"".join(struct.pack("<i", item) for item in values)
    return {
        "name": name,
        "dtype": "int32",
        "shape": [int(size) for size in tensor.shape],
        "values": values,
        "raw_little_endian_sha256": _sha256_bytes(raw),
    }


def _decode_tensor_row(row: dict[str, Any], tf: Any) -> Any:
    expected = {
        "name",
        "dtype",
        "shape",
        "values_hex" if row.get("dtype") == "float64" else "values",
        "raw_little_endian_sha256",
    }
    _require_keys(row, expected, f"tensor row {row.get('name')!r}")
    shape = row["shape"]
    if not isinstance(shape, list) or not all(
        type(item) is int and item >= 0 for item in shape
    ):
        raise ContractError("tensor row has an invalid shape")
    if row["dtype"] == "float64":
        values = [float.fromhex(item) for item in row["values_hex"]]
        if not all(math.isfinite(item) for item in values):
            raise ContractError("persisted float tensor contains a nonfinite value")
        raw = b"".join(struct.pack("<d", item) for item in values)
        dtype = tf.float64
    elif row["dtype"] == "int32":
        values = row["values"]
        if not all(type(item) is int and -(2**31) <= item < 2**31 for item in values):
            raise ContractError("persisted int32 tensor contains an invalid value")
        raw = b"".join(struct.pack("<i", item) for item in values)
        dtype = tf.int32
    elif row["dtype"] == "bool":
        values = row["values"]
        if not all(type(item) is bool for item in values):
            raise ContractError("persisted bool tensor contains an invalid value")
        raw = bytes(int(item) for item in values)
        dtype = tf.bool
    else:
        raise ContractError(f"unsupported persisted tensor dtype {row['dtype']!r}")
    if math.prod(shape) != len(values):
        raise ContractError("persisted tensor shape/value count mismatch")
    if _sha256_bytes(raw) != row["raw_little_endian_sha256"]:
        raise ContractError("persisted tensor raw hash mismatch")
    return tf.reshape(tf.constant(values, dtype=dtype), shape)


def _section_rows(payload: dict[str, Any], section: str) -> dict[str, dict[str, Any]]:
    rows = payload.get("tensor_sections", {}).get(section)
    if not isinstance(rows, list):
        raise ContractError(f"CPU reference lacks tensor section {section!r}")
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("name"), str):
            raise ContractError(f"invalid tensor row in section {section!r}")
        if row["name"] in result:
            raise ContractError(f"duplicate tensor row {section}/{row['name']}")
        result[row["name"]] = row
    return result


def _file_row(path: Path, role: str) -> dict[str, Any]:
    absolute = ROOT / path
    return {
        "path": path.as_posix(),
        "sha256": _sha256(path),
        "role": role,
        "exists": absolute.is_file(),
    }


def _check_row(
    name: str,
    *,
    passed: bool,
    role: str,
    residual: float | None = None,
    threshold: float | None = None,
) -> dict[str, Any]:
    if residual is not None and not math.isfinite(residual):
        raise ContractError(f"nonfinite residual for {name}")
    if threshold is not None and not math.isfinite(threshold):
        raise ContractError(f"nonfinite threshold for {name}")
    return {
        "name": name,
        "role": role,
        "passed": bool(passed),
        "residual": residual,
        "threshold": threshold,
    }


def _hex_float(value: str) -> float:
    result = float.fromhex(value)
    if not math.isfinite(result):
        raise ContractError(f"nonfinite fixture value {value!r}")
    return result


def _load_contracts(fixture_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    if fixture_path != FIXTURE_PATH:
        raise ContractError(f"A3 evidence requires exact fixture path {FIXTURE_PATH}")
    if _git("rev-parse", "HEAD").strip() != HEAD_SHA256:
        raise ContractError("HEAD drift from frozen A3 boundary")
    if _sha256(PLAN_PATH) != PLAN_SHA256:
        raise ContractError("reviewed A3 subplan hash drift")
    anchor_sha256 = _verify_harness_anchor()
    boundary = _verify_signed_input(
        BOUNDARY_PATH,
        schema="bayesfilter.ssl_lstm_completion.phase_a3_scoped_boundary.v1",
        status="A3_SCOPED_BOUNDARY_FROZEN",
    )
    fixture = _verify_signed_input(
        FIXTURE_PATH,
        schema="bayesfilter.ssl_lstm_completion.phase_a3_fixture_contract.v1",
        status="A3_FIXTURE_CONTRACT_FROZEN",
    )
    if boundary.get("harness_review_anchor_sha256") != anchor_sha256:
        raise ContractError("boundary does not bind the reviewed harness anchor")
    if fixture.get("boundary_sha256") != _sha256(BOUNDARY_PATH):
        raise ContractError("fixture does not bind frozen boundary")
    for row in boundary.get("a2_entry_bindings", []):
        path = Path(row["path"])
        if _sha256(path) != row["sha256"]:
            raise ContractError(f"A1/A2 entry binding drift: {path}")
    return boundary, fixture


def _load_runtime_modules() -> tuple[Any, Any, Any]:
    try:
        import tensorflow as tf

        oracle = importlib.import_module(
            "bayesfilter.testing.scalar_lgssm_forecast_oracle"
        )
        statistics = importlib.import_module(
            "bayesfilter.inference.predictive_equivalence"
        )
    except (ImportError, ModuleNotFoundError) as exc:
        raise ContractError(
            "reviewed A3 production modules are unavailable; no harness fallback is allowed"
        ) from exc
    return tf, oracle, statistics


def _public_call(module: Any, names: Iterable[str], *args: Any, **kwargs: Any) -> Any:
    for name in names:
        candidate = getattr(module, name, None)
        if callable(candidate):
            return candidate(*args, **kwargs)
    raise ContractError(
        f"required reviewed API missing from {module.__name__}: {tuple(names)}"
    )


def _dataclass_dict(value: Any) -> dict[str, Any]:
    import dataclasses

    if dataclasses.is_dataclass(value):
        return {field.name: getattr(value, field.name) for field in dataclasses.fields(value)}
    if isinstance(value, dict):
        return value
    raise ContractError(f"expected dataclass/dict result, got {type(value).__name__}")


def _compiler_row(name: str, program: Any, inputs: tuple[Any, ...]) -> dict[str, Any]:
    try:
        hlo = str(program.experimental_get_compiler_ir(*inputs)(stage="hlo"))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ContractError(f"unable to obtain compiler HLO for {name}") from exc
    if not hlo or "ENTRY" not in hlo:
        raise ContractError(f"missing concrete HLO ENTRY for {name}")
    outputs = program(*inputs)
    try:
        concrete_trace_count = len(
            program._list_all_concrete_functions_for_serialization()
        )
    except AttributeError as exc:
        raise ContractError(f"unable to enumerate concrete traces for {name}") from exc
    if concrete_trace_count != 1:
        raise ContractError(
            f"compiled program {name} has {concrete_trace_count} concrete traces; expected 1"
        )
    output_devices = sorted(
        {
            str(tensor.device)
            for tensor in _flatten_tensors(outputs)
            if getattr(tensor, "device", "")
        }
    )
    if not output_devices:
        raise ContractError(f"no output device provenance for {name}")
    encoded = hlo.encode("utf-8")
    return {
        "callable_name": name,
        "hlo_text": hlo,
        "hlo_sha256": _sha256_bytes(encoded),
        "hlo_byte_count": len(encoded),
        "hlo_entry_present": True,
        "concrete_trace_count": concrete_trace_count,
        "output_devices": output_devices,
    }


def _flatten_tensors(value: Any) -> list[Any]:
    import dataclasses

    if hasattr(value, "dtype") and hasattr(value, "shape"):
        return [value]
    if dataclasses.is_dataclass(value):
        result: list[Any] = []
        for field in dataclasses.fields(value):
            result.extend(_flatten_tensors(getattr(value, field.name)))
        return result
    if isinstance(value, dict):
        result = []
        for item in value.values():
            result.extend(_flatten_tensors(item))
        return result
    if isinstance(value, (list, tuple)):
        result = []
        for item in value:
            result.extend(_flatten_tensors(item))
        return result
    return []


def _manifest(
    *,
    mode: str,
    fixture: dict[str, Any],
    output: Path,
    started: str,
    completed: str,
    wall_time: float,
    tf: Any,
    reviewed_command_key: str,
    reviewed_command: str,
) -> dict[str, Any]:
    physical = tf.config.list_physical_devices()
    logical = tf.config.list_logical_devices()
    environment_names = (
        "CUDA_VISIBLE_DEVICES",
        "PYTHONDONTWRITEBYTECODE",
        "PYTHONPYCACHEPREFIX",
        "TMPDIR",
        "CUDA_CACHE_PATH",
        "XLA_FLAGS",
    )
    return {
        "git_commit": _git("rev-parse", "HEAD").strip(),
        "git_dirty": bool(_git("status", "--porcelain=v1", "--untracked-files=all")),
        "command": " ".join(sys.argv),
        "reviewed_command_key": reviewed_command_key,
        "reviewed_command": reviewed_command,
        "cwd": str(ROOT),
        "interpreter": sys.executable,
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV", "unknown"),
        "python_version": sys.version.split()[0],
        "packages": {
            "tensorflow": str(tf.__version__),
            "tensorflow_probability": _package_version("tensorflow-probability"),
        },
        "environment": {name: os.environ.get(name) for name in environment_names},
        "physical_devices": [
            {"name": str(item.name), "device_type": str(item.device_type)}
            for item in physical
        ],
        "logical_devices": [
            {"name": str(item.name), "device_type": str(item.device_type)}
            for item in logical
        ],
        "tf32_enabled": bool(tf.config.experimental.tensor_float_32_execution_enabled()),
        "jit_compile": True,
        "dtype": "float64",
        "random_seeds": fixture["fixture_constants"]["root_seed"],
        "started_at_utc": started,
        "completed_at_utc": completed,
        "wall_time_seconds": wall_time,
        "output_paths": [output.as_posix()],
        "plan_path": PLAN_PATH.as_posix(),
        "result_path": RESULT_PATH.as_posix(),
        "fixture_path": FIXTURE_PATH.as_posix(),
        "fixture_evidence_signature": fixture["evidence_signature"],
        "execution_role": (
            "cpu_hidden_xla_reference"
            if mode == "cpu-reference"
            else "trusted_gpu_xla_oracle"
        ),
        "trust_basis": (
            "cpu_hidden_reference_exception_not_gpu_evidence"
            if mode == "cpu-reference"
            else "owner_designated_managed_session_visible_gpu_trusted"
        ),
    }


def _source_rows() -> list[dict[str, Any]]:
    paths = (
        PREDICTIVE_SOURCE,
        ORACLE_SOURCE,
        PREDICTIVE_TEST,
        ORACLE_TEST,
        GENERATOR_PATH,
        VERIFIER_PATH,
    )
    missing = [path.as_posix() for path in paths if not (ROOT / path).is_file()]
    if missing:
        raise ContractError(f"required A3 source/test paths missing: {missing}")
    for path, expected in EXPECTED_SOURCE_HASHES.items():
        if _sha256(path) != expected:
            raise ContractError(f"frozen A3 source/test hash drift: {path}")
    return [_file_row(path, "a3_source_or_test") for path in paths]


def _fixture_parameters(fixture: dict[str, Any], tf: Any) -> Any:
    values = fixture["lgssm"]
    return {
        "transition_coefficient": tf.constant(
            _hex_float(values["a_hex"]), tf.float64
        ),
        "transition_offset": tf.constant(_hex_float(values["b_hex"]), tf.float64),
        "observation_coefficient": tf.constant(
            _hex_float(values["c_hex"]), tf.float64
        ),
        "observation_offset": tf.constant(_hex_float(values["d_hex"]), tf.float64),
        "process_variance": tf.constant(
            _hex_float(values["process_variance_q_hex"]), tf.float64
        ),
        "observation_variance": tf.constant(
            _hex_float(values["observation_variance_r_hex"]), tf.float64
        ),
        "terminal_mean": tf.constant(
            _hex_float(values["terminal_mean_hex"]), tf.float64
        ),
        "terminal_variance": tf.constant(
            _hex_float(values["p_terminal_hex"]), tf.float64
        ),
    }


def _status_text(value: Any) -> str:
    raw = value.numpy() if hasattr(value, "numpy") else value
    if isinstance(raw, bytes):
        return raw.decode("ascii")
    return str(raw)


def _finite_dataclass(value: Any, *, allow_negative_infinity: set[str] | None = None) -> None:
    allow_negative_infinity = allow_negative_infinity or set()
    for name, item in _dataclass_dict(value).items():
        if not hasattr(item, "dtype") or not getattr(item.dtype, "is_floating", False):
            continue
        values = _float_values_allowing(item, negative_infinity=name in allow_negative_infinity)
        if not values:
            raise ContractError(f"empty tensor in {name}")


def _float_values_allowing(value: Any, *, negative_infinity: bool) -> list[float]:
    import tensorflow as tf

    tensor = tf.convert_to_tensor(value, dtype=tf.float64)
    values = [float(item) for item in tf.reshape(tensor, [-1])]
    for item in values:
        if math.isfinite(item):
            continue
        if negative_infinity and item == float("-inf"):
            continue
        raise ContractError("nonfinite tensor cannot enter an artifact")
    return values


def _analytic_rows(analytic: Any) -> list[dict[str, Any]]:
    rows = []
    for name, value in _dataclass_dict(analytic).items():
        if not hasattr(value, "dtype"):
            continue
        if getattr(value.dtype, "is_floating", False):
            rows.append(_tensor_row(name, value))
        elif value.dtype.name == "bool":
            rows.append(_bool_tensor_row(name, value))
    return rows


def _bank_from_cpu_reference(
    cpu_payload: dict[str, Any], section: str, *, arm_id: int, oracle: Any, tf: Any
) -> Any:
    rows = _section_rows(cpu_payload, section)
    expected = {
        "terminal_standard_normal",
        "process_standard_normal",
        "observation_standard_normal",
    }
    if set(rows) != expected:
        raise ContractError(
            f"persisted bank section {section!r} differs; "
            f"missing={sorted(expected - set(rows))}, extra={sorted(set(rows) - expected)}"
        )
    return oracle.ScalarLGSSMInnovationBank(
        terminal_standard_normal=_decode_tensor_row(rows["terminal_standard_normal"], tf),
        process_standard_normal=_decode_tensor_row(rows["process_standard_normal"], tf),
        observation_standard_normal=_decode_tensor_row(
            rows["observation_standard_normal"], tf
        ),
        # Metadata is retained for the public bank type; it is not replay authority.
        root_seed=tf.constant([0, 0], tf.int32),
        arm_id=arm_id,
    )


def _stacked_bank_from_cpu_reference(
    cpu_payload: dict[str, Any], section: str, *, tf: Any
) -> dict[str, Any]:
    rows = _section_rows(cpu_payload, section)
    expected = {
        "terminal_standard_normal",
        "process_standard_normal",
        "observation_standard_normal",
    }
    if set(rows) != expected:
        raise ContractError(f"persisted stacked bank section {section!r} differs")
    return {name: _decode_tensor_row(rows[name], tf) for name in sorted(expected)}


def _bank_tensor_hashes(bank: Any) -> dict[str, str]:
    return {
        name: _tensor_row(name, value)["raw_little_endian_sha256"]
        for name, value in _dataclass_dict(bank).items()
        if hasattr(value, "dtype") and getattr(value.dtype, "is_floating", False)
    }


def _stacked_bank_tensor_hashes(tensors: dict[str, Any]) -> dict[str, str]:
    return {
        name: _tensor_row(name, tensors[name])["raw_little_endian_sha256"]
        for name in sorted(tensors)
    }


def _pairwise_domain_nonreuse(left: dict[str, str], right: dict[str, str]) -> bool:
    return set(left) == set(right) and bool(left) and all(
        left[name] != right[name] for name in left
    )


def _bank_domain_ledger(
    primary: tuple[dict[str, str], dict[str, str]],
    coverage_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    ledger = []
    for arm_id, hashes in enumerate(primary, start=1):
        for family in sorted(hashes):
            ledger.append(
                {
                    "purpose": "primary",
                    "replication": None,
                    "arm_id": arm_id,
                    "family": family,
                    "raw_little_endian_sha256": hashes[family],
                }
            )
    for row in coverage_rows:
        for arm_id, key in ((1, "left_tensor_hashes"), (2, "right_tensor_hashes")):
            hashes = row[key]
            for family in sorted(hashes):
                ledger.append(
                    {
                        "purpose": "coverage",
                        "replication": row["replication"],
                        "arm_id": arm_id,
                        "family": family,
                        "raw_little_endian_sha256": hashes[family],
                    }
                )
    return ledger


def _seed_pair(seed: Any) -> list[int]:
    return [int(item) for item in seed]


def _seed_domain_ledger(
    tf: Any, root_seed: list[int], domain_ledger: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    root = tf.constant(root_seed, tf.int32)
    result = []
    for domain in domain_ledger:
        parent = root
        if domain["purpose"] == "coverage":
            parent = tf.random.experimental.stateless_fold_in(
                root,
                tf.constant(10000 + int(domain["replication"]), tf.int32),
                alg="philox",
            )
        arm_seed = tf.random.experimental.stateless_fold_in(
            parent, tf.constant(domain["arm_id"], tf.int32), alg="philox"
        )
        family_code = FAMILY_CODES[domain["family"]]
        family_seed = tf.random.experimental.stateless_fold_in(
            arm_seed, tf.constant(family_code, tf.int32), alg="philox"
        )
        result.append(
            {
                **domain,
                "parent_seed": _seed_pair(parent),
                "arm_seed": _seed_pair(arm_seed),
                "family_code": family_code,
                "family_seed": _seed_pair(family_seed),
            }
        )
    return result


def _indices_from_cpu_reference(
    cpu_payload: dict[str, Any], section: str, *, constants: dict[str, Any], statistics: Any, tf: Any
) -> Any:
    rows = _section_rows(cpu_payload, section)
    expected = {"chain_indices", "draw_indices", "forecast_replication_indices"}
    if set(rows) != expected:
        raise ContractError(f"persisted index section {section!r} differs")
    return statistics.HierarchicalBootstrapIndices(
        chain_indices=_decode_tensor_row(rows["chain_indices"], tf),
        draw_indices=_decode_tensor_row(rows["draw_indices"], tf),
        forecast_replication_indices=_decode_tensor_row(
            rows["forecast_replication_indices"], tf
        ),
        block_length=int(constants["block_length"]),
        block_mode="moving",
        chain_mode="stratified_fixed_chains",
        seed=tf.constant([0, 0], tf.int32),
        status=tf.constant("VALID"),
    )


def _indices_valid(indices: Any, *, constants: dict[str, Any], tf: Any) -> bool:
    rows = _dataclass_dict(indices)
    bootstrap_count = int(constants["bootstrap_count"])
    chain_count = int(constants["chain_count_per_arm"])
    draw_count = int(constants["draw_count_per_chain"])
    replication_count = int(constants["forecast_replication_count"])
    block_length = int(constants["block_length"])
    expected_chains = tf.broadcast_to(
        tf.range(chain_count, dtype=tf.int32)[None, :],
        [bootstrap_count, chain_count],
    )
    block_count = draw_count // block_length
    blocks = tf.reshape(
        rows["draw_indices"],
        [bootstrap_count, chain_count, block_count, block_length],
    )
    expected_offsets = tf.range(block_length, dtype=tf.int32)
    starts = blocks[..., 0]
    return bool(
        _status_text(indices.status) == "VALID"
        and indices.block_length == block_length
        and indices.block_mode == "moving"
        and indices.chain_mode == "stratified_fixed_chains"
        and draw_count % block_length == 0
        and tuple(rows["chain_indices"].shape) == (bootstrap_count, chain_count)
        and tuple(rows["draw_indices"].shape)
        == (bootstrap_count, chain_count, draw_count)
        and tuple(rows["forecast_replication_indices"].shape)
        == (bootstrap_count, chain_count, draw_count, replication_count)
        and tf.reduce_all(rows["chain_indices"] == expected_chains)
        and tf.reduce_all(rows["draw_indices"] >= 0)
        and tf.reduce_all(rows["draw_indices"] < draw_count)
        and tf.reduce_all(blocks - starts[..., None] == expected_offsets)
        and tf.reduce_all(starts >= 0)
        and tf.reduce_all(starts <= draw_count - block_length)
        and tf.reduce_all(rows["forecast_replication_indices"] >= 0)
        and tf.reduce_all(rows["forecast_replication_indices"] < replication_count)
    )


def _max_abs(tf: Any, left: Any, right: Any) -> float:
    return float(tf.reduce_max(tf.abs(tf.cast(left, tf.float64) - tf.cast(right, tf.float64))))


def _scale_tolerance(tf: Any, left: Any, right: Any, multiplier: int) -> float:
    scale = tf.maximum(
        tf.constant(1.0, tf.float64),
        tf.maximum(
            tf.reduce_max(tf.abs(tf.cast(left, tf.float64))),
            tf.reduce_max(tf.abs(tf.cast(right, tf.float64))),
        ),
    )
    return float(tf.constant(multiplier * 2.0**-52, tf.float64) * scale)


def _manual_analytic_reference(
    fixture: dict[str, Any], tf: Any, quantile_probabilities: Any
) -> dict[str, Any]:
    values = fixture["lgssm"]
    a = _hex_float(values["a_hex"])
    b = _hex_float(values["b_hex"])
    c = _hex_float(values["c_hex"])
    d = _hex_float(values["d_hex"])
    terminal_mean = _hex_float(values["terminal_mean_hex"])
    terminal_variance = _hex_float(values["p_terminal_hex"])
    process_variance = _hex_float(values["process_variance_q_hex"])
    observation_variance = _hex_float(values["observation_variance_r_hex"])
    horizon = int(fixture["fixture_constants"]["horizon"])
    state_mean = []
    for step in range(1, horizon + 1):
        state_mean.append(a**step * terminal_mean + b * sum(a**power for power in range(step)))
    state_covariance = []
    for left in range(1, horizon + 1):
        row = []
        for right in range(1, horizon + 1):
            row.append(
                a ** (left + right) * terminal_variance
                + process_variance
                * sum(a ** (left - index) * a ** (right - index) for index in range(1, min(left, right) + 1))
            )
        state_covariance.append(row)
    state_mean_tensor = tf.constant(state_mean, tf.float64)
    state_covariance_tensor = tf.constant(state_covariance, tf.float64)
    observation_mean_tensor = c * state_mean_tensor + d
    observation_covariance_tensor = (
        c * c * state_covariance_tensor
        + observation_variance * tf.eye(horizon, dtype=tf.float64)
    )
    variance = tf.linalg.diag_part(observation_covariance_tensor)
    tfp = importlib.import_module("tensorflow_probability")
    quantiles = tfp.distributions.Normal(
        loc=observation_mean_tensor[:, None],
        scale=tf.sqrt(variance)[:, None],
    ).quantile(quantile_probabilities[None, :])
    return {
        "state_mean": state_mean_tensor,
        "state_covariance": state_covariance_tensor,
        "observation_mean": observation_mean_tensor,
        "observation_covariance": observation_covariance_tensor,
        "observation_variance": variance,
        "observation_log_variance": tf.math.log(variance),
        "observation_third_central_moment": tf.zeros([horizon], tf.float64),
        "observation_fourth_central_moment": 3.0 * tf.square(variance),
        "observation_quantiles": quantiles,
    }


def _resample_paths(tf: Any, paths: Any, indices: Any) -> Any:
    rows = _dataclass_dict(indices)
    chain_indices = rows["chain_indices"]
    draw_indices = rows["draw_indices"]
    replication_indices = rows["forecast_replication_indices"]
    bootstrap_count = int(draw_indices.shape[0])
    chain_count = int(paths.shape[0])
    draw_count = int(paths.shape[1])
    samples = []
    for bootstrap in range(bootstrap_count):
        chains = []
        for chain in range(chain_count):
            source_chain = int(chain_indices[bootstrap, chain])
            selected_draws = tf.gather(paths[source_chain], draw_indices[bootstrap, chain])
            draw_rows = []
            for draw in range(draw_count):
                draw_rows.append(
                    tf.gather(
                        selected_draws[draw], replication_indices[bootstrap, chain, draw]
                    )
                )
            chains.append(tf.stack(draw_rows))
        samples.append(tf.stack(chains))
    return tf.stack(samples)


def _decision_row(label: str, decision: Any) -> dict[str, Any]:
    return {
        "label": label,
        "status": str(decision.status),
        "primary_interval_status": str(decision.primary_interval_status),
        "mmd_upper_bound_status": str(decision.mmd_upper_bound_status),
        "hard_veto_codes": list(decision.hard_veto_codes),
    }


def _tensor_row_values(row: dict[str, Any]) -> list[float]:
    return [float.fromhex(value) for value in row["values_hex"]]


def _tensor_section_parity(
    current: dict[str, list[dict[str, Any]]],
    reference: dict[str, list[dict[str, Any]]],
) -> tuple[float, float]:
    if set(current) != set(reference):
        raise ContractError("CPU/GPU tensor-section names differ")
    maximum_residual = 0.0
    maximum_threshold = 0.0
    for section in sorted(current):
        current_rows = current[section]
        reference_rows = reference[section]
        if [row["name"] for row in current_rows] != [row["name"] for row in reference_rows]:
            raise ContractError(f"CPU/GPU tensor row order differs in {section}")
        for current_row, reference_row in zip(current_rows, reference_rows):
            if current_row["shape"] != reference_row["shape"] or current_row["dtype"] != reference_row["dtype"]:
                raise ContractError(f"CPU/GPU tensor metadata differs for {section}/{current_row['name']}")
            if current_row["dtype"] != "float64":
                if current_row != reference_row:
                    raise ContractError(
                        f"CPU/GPU exact tensor parity failed for {section}/{current_row['name']}"
                    )
                continue
            current_values = _tensor_row_values(current_row)
            reference_values = _tensor_row_values(reference_row)
            if len(current_values) != len(reference_values):
                raise ContractError("CPU/GPU tensor lengths differ")
            scale = max(
                1.0,
                max((abs(value) for value in current_values), default=0.0),
                max((abs(value) for value in reference_values), default=0.0),
            )
            threshold = CPU_GPU_TOLERANCE_MULTIPLIER * 2.0**-52 * scale
            residual = max(
                (abs(left - right) for left, right in zip(current_values, reference_values)),
                default=0.0,
            )
            maximum_residual = max(maximum_residual, residual)
            maximum_threshold = max(maximum_threshold, threshold)
            if residual > threshold:
                raise ContractError(
                    f"CPU/GPU tensor parity failed for {section}/{current_row['name']}: {residual} > {threshold}"
                )
    return maximum_residual, maximum_threshold


def _manual_summary(tf: Any, paths: Any, probabilities: Any) -> dict[str, Any]:
    horizon = int(paths.shape[-1])
    flat = tf.reshape(paths, [-1, horizon])
    count = int(flat.shape[0])
    means = tf.reduce_mean(flat, axis=0)
    centered = flat - means
    variances = tf.reduce_sum(tf.square(centered), axis=0) / tf.constant(
        float(count - 1), tf.float64
    )
    covariance = tf.matmul(centered, centered, transpose_a=True) / tf.constant(
        float(count - 1), tf.float64
    )
    moments = tf.stack(
        [tf.reduce_mean(tf.pow(centered, order), axis=0) for order in (3, 4)]
    )
    sorted_values = tf.sort(flat, axis=0)
    positions = probabilities * tf.constant(float(count - 1), tf.float64)
    lower = tf.cast(tf.floor(positions), tf.int32)
    upper = tf.cast(tf.math.ceil(positions), tf.int32)
    fraction = positions - tf.floor(positions)
    quantiles = tf.gather(sorted_values, lower) + fraction[:, None] * (
        tf.gather(sorted_values, upper) - tf.gather(sorted_values, lower)
    )
    return {
        "means": means,
        "variances": variances,
        "log_variances": tf.math.log(variances),
        "central_moments": moments,
        "quantiles": quantiles,
        "cross_horizon_covariance": covariance,
    }


def _bootstrap_feature_matrix(tf: Any, left: Any, right: Any) -> Any:
    if left.shape != right.shape or left.shape.rank != 5:
        raise ContractError("bootstrap arms must have equal [bootstrap,chain,draw,rep,horizon] shape")
    count = math.prod(int(item) for item in left.shape[1:4])

    def features(paths: Any) -> tuple[Any, Any]:
        means = tf.reduce_mean(paths, axis=[1, 2, 3])
        centered = paths - means[:, None, None, None, :]
        variances = tf.reduce_sum(tf.square(centered), axis=[1, 2, 3]) / tf.constant(
            float(count - 1), tf.float64
        )
        return means, tf.math.log(variances)

    left_mean, left_log_variance = features(left)
    right_mean, right_log_variance = features(right)
    result = tf.concat(
        [left_mean - right_mean, left_log_variance - right_log_variance], axis=1
    )
    if not bool(tf.reduce_all(tf.math.is_finite(result))):
        raise ContractError("bootstrap feature matrix is nonfinite")
    return result


def _interval_rows(value: Any) -> list[dict[str, Any]]:
    rows = []
    for name, item in _dataclass_dict(value).items():
        if not hasattr(item, "dtype"):
            continue
        if getattr(item.dtype, "is_floating", False):
            rows.append(_tensor_row(name, item))
        elif getattr(item.dtype, "is_integer", False):
            rows.append(_int_tensor_row(name, item))
    return rows


def _interval_metadata(value: Any) -> dict[str, Any]:
    rows = _dataclass_dict(value)
    return {
        name: (
            _status_text(item)
            if hasattr(item, "dtype") and item.dtype.name == "string"
            else item
        )
        for name, item in rows.items()
        if not hasattr(item, "dtype") or item.dtype.name == "string"
    }


def _alternative_evidence(
    *,
    name: str,
    alternative_paths: Any,
    left_paths: Any,
    bootstrap_left_paths: Any,
    indices_right: Any,
    left_summary: Any,
    config: Any,
    analytic: Any,
    bandwidths: Any,
    weights: Any,
    schedule: Any,
    margins: Any,
    mmd_tolerance: Any,
    total_alpha: Any,
    feature_alpha: Any,
    mmd_alpha: Any,
    independent_arm_banks_verified: bool,
    constants: dict[str, Any],
    statistics: Any,
    tf: Any,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    summary = statistics.summarize_forecast_paths(alternative_paths, config)
    estimate = tf.concat(
        [
            left_summary.means - summary.means,
            left_summary.log_variances - summary.log_variances,
        ],
        axis=0,
    )
    bootstrap_alternative = _resample_paths(tf, alternative_paths, indices_right)
    bootstrap_estimates = _bootstrap_feature_matrix(
        tf, bootstrap_left_paths, bootstrap_alternative
    )
    feature_interval = statistics.simultaneous_feature_intervals(
        estimate,
        feature_alpha=feature_alpha,
        method="bootstrap_max_statistic",
        bootstrap_estimates=bootstrap_estimates,
        minimum_bootstrap_count=20,
        jit_compile=True,
    )
    standardized_alternative = statistics.standardize_forecast_paths(
        alternative_paths,
        analytic.observation_mean,
        tf.sqrt(analytic.observation_variance),
        scale_floor=tf.constant(2.0**-40, tf.float64),
        jit_compile=True,
        allow_floor_use=False,
    )
    cross_statistic = statistics.cross_chain_linear_mmd(
        statistics.standardize_forecast_paths(
            left_paths,
            analytic.observation_mean,
            tf.sqrt(analytic.observation_variance),
            scale_floor=tf.constant(2.0**-40, tf.float64),
            jit_compile=True,
            allow_floor_use=False,
        ),
        standardized_alternative,
        bandwidths=bandwidths,
        mixture_weights=weights,
        chain_pair_schedule=schedule,
        independent_arm_banks_verified=independent_arm_banks_verified,
        stationarity_verified=True,
        mixing_verified=True,
        jit_compile=True,
    )
    mmd_interval = statistics.cross_chain_mmd_upper_interval(
        cross_statistic,
        mmd_alpha=mmd_alpha,
        block_length=int(constants["block_length"]),
        jit_compile=True,
    )
    decision = statistics.classify_predictive_evidence(
        feature_interval,
        mmd_interval,
        margins=margins,
        mmd_tolerance=mmd_tolerance,
        total_alpha=total_alpha,
        feature_alpha=feature_alpha,
        mmd_alpha=mmd_alpha,
    )
    valid = (
        _status_text(summary.status) == "VALID"
        and feature_interval.inference_admissible
        and _status_text(feature_interval.status) == "VALID"
        and mmd_interval.inference_admissible
        and _status_text(mmd_interval.status) == "VALID"
        and decision.status != "INVALID_HARD_VETO"
    )
    record = {
        "name": name,
        "valid": bool(valid),
        "decision": _decision_row(name, decision),
        "repair_trigger": bool(
            valid and decision.status != "MATERIAL_DIFFERENCE"
        ),
        "feature_max_abs": float(tf.reduce_max(tf.abs(estimate))),
        "mmd_estimate": float(cross_statistic.squared_mmd_linear),
        "mmd_lower": float(mmd_interval.lower),
        "mmd_upper": float(mmd_interval.upper),
    }
    tensors = {
        f"alternative_{name}_inputs": [
            _tensor_row("paths", alternative_paths),
            _tensor_row("feature_estimate", estimate),
            _tensor_row("bootstrap_feature_estimates", bootstrap_estimates),
        ],
        f"alternative_{name}_feature_interval": _interval_rows(feature_interval),
        f"alternative_{name}_cross_chain": [
            _tensor_row("squared_mmd_linear", cross_statistic.squared_mmd_linear),
            _tensor_row("kernel_contrast_sequence", cross_statistic.kernel_contrast_sequence),
            _int_tensor_row("chain_pair_schedule", cross_statistic.chain_pair_schedule),
        ],
        f"alternative_{name}_mmd_interval": _interval_rows(mmd_interval),
    }
    return record, tensors


def _artifact_payload(
    *,
    mode: str,
    fixture_path: Path,
    cpu_reference: Path | None,
    output: Path,
) -> dict[str, Any]:
    boundary, fixture = _load_contracts(fixture_path)
    cpu_payload = None
    cpu_reference_sha256 = None
    if mode == "cpu-reference":
        if cpu_reference is not None:
            raise ContractError("CPU mode must not accept --cpu-reference")
    else:
        if cpu_reference is None:
            raise ContractError("GPU mode requires --cpu-reference")
        cpu_payload, cpu_reference_sha256 = _verified_cpu_replay_authority(
            cpu_reference
        )
    tf, oracle, statistics = _load_runtime_modules()
    started_at = _utc_now()
    started = time.monotonic()
    reviewed_command_key = "cpu_artifact" if mode == "cpu-reference" else "gpu_artifact"
    reviewed_command = boundary.get("exact_commands", {}).get(reviewed_command_key)
    if not isinstance(reviewed_command, str) or not reviewed_command:
        raise ContractError(f"frozen boundary lacks command {reviewed_command_key!r}")

    constants = fixture["fixture_constants"]
    horizon = int(constants["horizon"])
    quantile_probabilities = tf.constant(
        [_hex_float(item) for item in fixture["quantile_contract"]["probabilities_hex"]],
        tf.float64,
    )
    parameter_type = getattr(oracle, "ScalarLGSSMParameters", None)
    if parameter_type is None:
        raise ContractError("missing ScalarLGSSMParameters")
    parameters = parameter_type(**_fixture_parameters(fixture, tf))
    analytic = oracle.analytic_scalar_lgssm_forecast(
        parameters,
        horizon=horizon,
        quantile_probabilities=quantile_probabilities,
        jit_compile=True,
    )
    _finite_dataclass(analytic)
    manual = _manual_analytic_reference(fixture, tf, quantile_probabilities)
    analytic_dict = _dataclass_dict(analytic)
    deterministic_residuals: dict[str, dict[str, float]] = {}
    analytic_passed = True
    for name, expected in manual.items():
        observed = analytic_dict[name]
        residual = _max_abs(tf, observed, expected)
        threshold = _scale_tolerance(tf, observed, expected, 512)
        deterministic_residuals[name] = {
            "residual": residual,
            "threshold": threshold,
        }
        analytic_passed = analytic_passed and residual <= threshold
    state_symmetry = float(analytic.state_symmetry_residual)
    observation_symmetry = float(analytic.observation_symmetry_residual)
    minimum_state_eigenvalue = float(analytic.minimum_state_covariance_eigenvalue)
    minimum_observation_eigenvalue = float(
        analytic.minimum_observation_covariance_eigenvalue
    )
    state_psd_tolerance = float(analytic.state_psd_tolerance)
    observation_psd_tolerance = float(analytic.observation_psd_tolerance)
    covariance_passed = (
        all(
            math.isfinite(item)
            for item in (
                state_symmetry,
                observation_symmetry,
                minimum_state_eigenvalue,
                minimum_observation_eigenvalue,
                state_psd_tolerance,
                observation_psd_tolerance,
            )
        )
        and state_symmetry <= state_psd_tolerance
        and observation_symmetry <= observation_psd_tolerance
        and minimum_state_eigenvalue >= -state_psd_tolerance
        and minimum_observation_eigenvalue >= -observation_psd_tolerance
        and not bool(tf.reduce_any(analytic.degenerate_variance_mask))
        and bool(analytic.log_variance_valid)
        and _status_text(analytic.status) == "VALID"
    )

    if mode == "cpu-reference":
        bank_left = oracle.make_scalar_lgssm_innovation_bank(
            chain_count=int(constants["chain_count_per_arm"]),
            draw_count=int(constants["draw_count_per_chain"]),
            forecast_replication_count=int(constants["forecast_replication_count"]),
            horizon=horizon,
            seed=tf.constant(constants["root_seed"], tf.int32),
            arm_id=1,
        )
        bank_right = oracle.make_scalar_lgssm_innovation_bank(
            chain_count=int(constants["chain_count_per_arm"]),
            draw_count=int(constants["draw_count_per_chain"]),
            forecast_replication_count=int(constants["forecast_replication_count"]),
            horizon=horizon,
            seed=tf.constant(constants["root_seed"], tf.int32),
            arm_id=2,
        )
    else:
        if cpu_payload is None:
            raise ContractError("GPU mode lacks a verified CPU replay authority")
        bank_left = _bank_from_cpu_reference(
            cpu_payload, "innovation_bank_left", arm_id=1, oracle=oracle, tf=tf
        )
        bank_right = _bank_from_cpu_reference(
            cpu_payload, "innovation_bank_right", arm_id=2, oracle=oracle, tf=tf
        )
    left_simulation = oracle.simulate_scalar_lgssm_forecast(
        parameters, bank_left, horizon=horizon, jit_compile=True
    )
    right_simulation = oracle.simulate_scalar_lgssm_forecast(
        parameters, bank_right, horizon=horizon, jit_compile=True
    )
    left_replay = oracle.simulate_scalar_lgssm_forecast(
        parameters, bank_left, horizon=horizon, jit_compile=True
    )
    _finite_dataclass(left_simulation)
    _finite_dataclass(right_simulation)
    left_paths = left_simulation.observations
    right_paths = right_simulation.observations
    replay_residual = _max_abs(tf, left_paths, left_replay.observations)
    replay_passed = replay_residual == 0.0
    bank_hashes_left = _bank_tensor_hashes(bank_left)
    bank_hashes_right = _bank_tensor_hashes(bank_right)
    primary_domain_nonreuse = _pairwise_domain_nonreuse(
        bank_hashes_left, bank_hashes_right
    )
    if mode == "gpu-xla-canary":
        primary_domain_nonreuse = primary_domain_nonreuse and all(
            _tensor_row(name, _dataclass_dict(bank)[name])
            == _section_rows(cpu_payload, section)[name]
            for bank, section in (
                (bank_left, "innovation_bank_left"),
                (bank_right, "innovation_bank_right"),
            )
            for name in (
                "terminal_standard_normal",
                "process_standard_normal",
                "observation_standard_normal",
            )
        )

    config_type = getattr(statistics, "PredictiveStatisticsConfig", None)
    if config_type is None:
        raise ContractError("missing PredictiveStatisticsConfig")
    config = config_type(
        horizon=horizon,
        quantile_probabilities=tuple(float(item) for item in quantile_probabilities),
        jit_compile=True,
    )
    left_summary = statistics.summarize_forecast_paths(left_paths, config)
    right_summary = statistics.summarize_forecast_paths(right_paths, config)
    for summary in (left_summary, right_summary):
        if _status_text(summary.status) != "VALID":
            raise ContractError("summary status is not VALID")
        _finite_dataclass(summary)
    manual_summary = _manual_summary(tf, left_paths, quantile_probabilities)
    summary_residuals = {
        name: _max_abs(tf, getattr(left_summary, name), expected)
        for name, expected in manual_summary.items()
    }
    summary_thresholds = {
        name: _scale_tolerance(tf, getattr(left_summary, name), expected, 1024)
        for name, expected in manual_summary.items()
    }
    summary_passed = (
        _status_text(left_summary.status) == "VALID"
        and int(left_summary.path_count) == int(math.prod(left_paths.shape[:-1]))
        and all(
            summary_residuals[name] <= summary_thresholds[name]
            for name in manual_summary
        )
    )

    pooled_count = int(left_paths.shape[0] * left_paths.shape[1] * left_paths.shape[2])
    feature_alpha = tf.constant(_hex_float(constants["feature_alpha_hex"]), tf.float64)
    tfp = importlib.import_module("tensorflow_probability")
    simultaneous_tail = feature_alpha / tf.cast(4 * horizon, tf.float64)
    mean_critical = tfp.distributions.Normal(
        loc=tf.constant(0.0, tf.float64), scale=tf.constant(1.0, tf.float64)
    ).quantile(1.0 - simultaneous_tail)
    mean_se = tf.sqrt(analytic.observation_variance / tf.cast(pooled_count, tf.float64))
    mean_z = tf.abs(left_summary.means - analytic.observation_mean) / mean_se
    degrees_of_freedom = tf.constant(float(pooled_count - 1), tf.float64)
    chi_square = tfp.distributions.Chi2(df=degrees_of_freedom)
    variance_ratio = degrees_of_freedom * left_summary.variances / analytic.observation_variance
    variance_lower = chi_square.quantile(simultaneous_tail)
    variance_upper = chi_square.quantile(1.0 - simultaneous_tail)
    monte_carlo_passed = bool(tf.reduce_max(mean_z) <= mean_critical) and bool(
        tf.reduce_all(
            tf.logical_and(variance_ratio >= variance_lower, variance_ratio <= variance_upper)
        )
    )

    standardized_left = statistics.standardize_forecast_paths(
        left_paths,
        analytic.observation_mean,
        tf.sqrt(analytic.observation_variance),
        scale_floor=tf.constant(2.0**-40, tf.float64),
        jit_compile=True,
        allow_floor_use=False,
    )
    standardized_right = statistics.standardize_forecast_paths(
        right_paths,
        analytic.observation_mean,
        tf.sqrt(analytic.observation_variance),
        scale_floor=tf.constant(2.0**-40, tf.float64),
        jit_compile=True,
        allow_floor_use=False,
    )
    standardization_passed = bool(tf.reduce_all(tf.math.is_finite(standardized_left)))
    bandwidths = tf.constant(
        [_hex_float(item) for item in constants["bandwidths_hex"]], tf.float64
    )
    weights = tf.constant(
        [_hex_float(item) for item in constants["mixture_weights_hex"]], tf.float64
    )
    flattened_left = tf.reshape(standardized_left, [-1, horizon])
    flattened_right = tf.reshape(standardized_right, [-1, horizon])
    mmd = statistics.fixed_rbf_mmd(
        flattened_left,
        flattened_right,
        bandwidths=bandwidths,
        mixture_weights=weights,
        sampling_contract="iid_oracle_fixture",
        iid_samples_verified=True,
        independent_arm_banks_verified=primary_domain_nonreuse,
        jit_compile=True,
    )
    paired_diagnostic = statistics.fixed_rbf_mmd(
        flattened_left,
        flattened_left,
        bandwidths=bandwidths,
        mixture_weights=weights,
        sampling_contract="paired_diagnostic_shared",
        jit_compile=True,
    )
    signed_u_fixture = statistics.fixed_rbf_mmd(
        tf.constant([[0.0], [2.0]], tf.float64),
        tf.constant([[0.0], [2.0]], tf.float64),
        bandwidths=tf.constant([0.5, 1.0], tf.float64),
        mixture_weights=tf.constant([0.5, 0.5], tf.float64),
        sampling_contract="paired_diagnostic_shared",
        jit_compile=True,
    )
    signed_u_passed = (
        float(signed_u_fixture.squared_mmd_u) < 0.0
        and float(signed_u_fixture.squared_mmd_v_biased)
        >= float(signed_u_fixture.squared_mmd_u)
        and not signed_u_fixture.inference_admissible
        and _status_text(signed_u_fixture.status) == "VALID"
    )
    mmd_roles_passed = (
        not mmd.inference_admissible
        and not paired_diagnostic.inference_admissible
        and mmd.iid_samples_verified
        and mmd.independent_arm_banks_verified
        and _status_text(mmd.status) == "VALID"
        and _status_text(paired_diagnostic.status) == "VALID"
        and math.isfinite(float(mmd.squared_mmd_u))
        and math.isfinite(float(mmd.squared_mmd_v_biased))
    )
    schedule = tf.constant(constants["chain_pair_schedule"], tf.int32)
    cross_chain = statistics.cross_chain_linear_mmd(
        standardized_left,
        standardized_right,
        bandwidths=bandwidths,
        mixture_weights=weights,
        chain_pair_schedule=schedule,
        independent_arm_banks_verified=primary_domain_nonreuse,
        stationarity_verified=True,
        mixing_verified=True,
        jit_compile=True,
    )
    cross_chain_passed = (
        cross_chain.inference_admissible
        and _status_text(cross_chain.status) == "VALID"
        and list(tf.reshape(cross_chain.chain_pair_schedule, [-1]).numpy()) == [0, 1, 2, 3]
    )
    mmd_interval = statistics.cross_chain_mmd_upper_interval(
        cross_chain,
        mmd_alpha=tf.constant(_hex_float(constants["mmd_alpha_hex"]), tf.float64),
        block_length=int(constants["block_length"]),
        jit_compile=True,
    )
    mmd_interval_passed = (
        mmd_interval.inference_admissible and _status_text(mmd_interval.status) == "VALID"
    )

    if mode == "cpu-reference":
        bootstrap_seeds = tf.random.experimental.stateless_split(
            tf.constant(constants["root_seed"], tf.int32), 2, alg="philox"
        )
        indices_left, indices_right = [
            statistics.hierarchical_resample_indices(
                chain_count=int(constants["chain_count_per_arm"]),
                draw_count=int(constants["draw_count_per_chain"]),
                forecast_replication_count=int(constants["forecast_replication_count"]),
                block_length=int(constants["block_length"]),
                bootstrap_count=int(constants["bootstrap_count"]),
                seed=bootstrap_seeds[arm],
                chain_mode="stratified_fixed_chains",
                block_mode="moving",
                jit_compile=True,
            )
            for arm in range(2)
        ]
    else:
        indices_left = _indices_from_cpu_reference(
            cpu_payload,
            "resampling_indices_left",
            constants=constants,
            statistics=statistics,
            tf=tf,
        )
        indices_right = _indices_from_cpu_reference(
            cpu_payload,
            "resampling_indices_right",
            constants=constants,
            statistics=statistics,
            tf=tf,
        )
    index_dict_left = _dataclass_dict(indices_left)
    index_dict_right = _dataclass_dict(indices_right)
    resampling_passed = _indices_valid(
        indices_left, constants=constants, tf=tf
    ) and _indices_valid(indices_right, constants=constants, tf=tf)
    resampling_passed = resampling_passed and any(
        not bool(tf.reduce_all(index_dict_left[name] == index_dict_right[name]))
        for name in ("draw_indices", "forecast_replication_indices")
    )
    if mode == "cpu-reference":
        for arm, source in enumerate((indices_left, indices_right)):
            repeated = statistics.hierarchical_resample_indices(
                chain_count=int(constants["chain_count_per_arm"]),
                draw_count=int(constants["draw_count_per_chain"]),
                forecast_replication_count=int(constants["forecast_replication_count"]),
                block_length=int(constants["block_length"]),
                bootstrap_count=int(constants["bootstrap_count"]),
                seed=bootstrap_seeds[arm],
                chain_mode="stratified_fixed_chains",
                block_mode="moving",
                jit_compile=True,
            )
            resampling_passed = resampling_passed and all(
                bool(
                    tf.reduce_all(
                        _dataclass_dict(source)[name] == _dataclass_dict(repeated)[name]
                    )
                )
                for name in (
                    "chain_indices",
                    "draw_indices",
                    "forecast_replication_indices",
                )
            )

    bootstrap_left_paths = _resample_paths(tf, left_paths, indices_left)
    bootstrap_right_paths = _resample_paths(tf, right_paths, indices_right)
    base_features = tf.concat(
        [left_summary.means - right_summary.means, left_summary.log_variances - right_summary.log_variances],
        axis=0,
    )
    bootstrap_features = _bootstrap_feature_matrix(
        tf, bootstrap_left_paths, bootstrap_right_paths
    )
    feature_intervals = statistics.simultaneous_feature_intervals(
        base_features,
        feature_alpha=feature_alpha,
        method="bootstrap_max_statistic",
        bootstrap_estimates=bootstrap_features,
        minimum_bootstrap_count=20,
        jit_compile=True,
    )
    intervals_passed = (
        feature_intervals.inference_admissible
        and _status_text(feature_intervals.status) == "VALID"
    )
    total_alpha = tf.constant(_hex_float(constants["total_alpha_hex"]), tf.float64)
    mmd_alpha = tf.constant(_hex_float(constants["mmd_alpha_hex"]), tf.float64)
    joint_alpha_passed = bool(feature_alpha + mmd_alpha <= total_alpha)
    margins = tf.concat(
        [
            tf.fill([horizon], tf.constant(_hex_float(constants["mean_margin_hex"]), tf.float64)),
            tf.fill([horizon], tf.constant(_hex_float(constants["log_variance_margin_hex"]), tf.float64)),
        ],
        axis=0,
    )
    mmd_tolerance = tf.constant(_hex_float(constants["mmd_tolerance_hex"]), tf.float64)
    observed_decision = statistics.classify_predictive_evidence(
        feature_intervals,
        mmd_interval,
        margins=margins,
        mmd_tolerance=mmd_tolerance,
        total_alpha=total_alpha,
        feature_alpha=feature_alpha,
        mmd_alpha=mmd_alpha,
    )
    mechanics_decision = statistics.classify_predictive_evidence(
        feature_intervals,
        mmd_interval,
        margins=margins,
        mmd_tolerance=mmd_tolerance,
        total_alpha=total_alpha,
        feature_alpha=feature_alpha,
        mmd_alpha=mmd_alpha,
        mechanics_only=True,
    )
    invalid_total_alpha = total_alpha / tf.constant(2.0, tf.float64)
    invalid_alpha_decision = statistics.classify_predictive_evidence(
        feature_intervals,
        mmd_interval,
        margins=margins,
        mmd_tolerance=mmd_tolerance,
        total_alpha=invalid_total_alpha,
        feature_alpha=feature_alpha,
        mmd_alpha=mmd_alpha,
    )
    decision_passed = (
        mechanics_decision.status == "INVALID_HARD_VETO"
        and invalid_alpha_decision.status == "INVALID_HARD_VETO"
        and observed_decision.status
        in {"PASS", "MATERIAL_DIFFERENCE", "INCONCLUSIVE_UNDERPOWERED"}
    )

    # Exercise every decision branch through authenticated public constructors.
    # These values are branch fixtures only, never law-comparison evidence.
    zero_estimate = tf.zeros([2 * horizon], tf.float64)
    narrow_se = margins / tf.cast(horizon**2, tf.float64)
    pass_features = statistics.simultaneous_feature_intervals(
        zero_estimate,
        feature_alpha=feature_alpha,
        method="bonferroni_studentized",
        standard_error=narrow_se,
        jit_compile=True,
    )
    material_features = statistics.simultaneous_feature_intervals(
        2.0 * margins,
        feature_alpha=feature_alpha,
        method="bonferroni_studentized",
        standard_error=narrow_se,
        jit_compile=True,
    )
    inconclusive_features = statistics.simultaneous_feature_intervals(
        zero_estimate,
        feature_alpha=feature_alpha,
        method="bonferroni_studentized",
        standard_error=margins,
        jit_compile=True,
    )
    branch_mmd_tolerance = mmd_tolerance + tf.abs(mmd_interval.upper)
    branch_pass = statistics.classify_predictive_evidence(
        pass_features,
        mmd_interval,
        margins=margins,
        mmd_tolerance=branch_mmd_tolerance,
        total_alpha=total_alpha,
        feature_alpha=feature_alpha,
        mmd_alpha=mmd_alpha,
    )
    branch_material = statistics.classify_predictive_evidence(
        material_features,
        mmd_interval,
        margins=margins,
        mmd_tolerance=branch_mmd_tolerance,
        total_alpha=total_alpha,
        feature_alpha=feature_alpha,
        mmd_alpha=mmd_alpha,
    )
    branch_inconclusive = statistics.classify_predictive_evidence(
        inconclusive_features,
        mmd_interval,
        margins=margins,
        mmd_tolerance=branch_mmd_tolerance,
        total_alpha=total_alpha,
        feature_alpha=feature_alpha,
        mmd_alpha=mmd_alpha,
    )
    decision_passed = decision_passed and (
        branch_pass.status == "PASS"
        and branch_material.status == "MATERIAL_DIFFERENCE"
        and branch_inconclusive.status == "INCONCLUSIVE_UNDERPOWERED"
    )

    coverage_count = int(constants["coverage_replication_count"])
    coverage_successes = 0
    if mode == "cpu-reference":
        coverage_left_rows = []
        coverage_right_rows = []
        coverage_bank_rows = {
            1: {
                "terminal_standard_normal": [],
                "process_standard_normal": [],
                "observation_standard_normal": [],
            },
            2: {
                "terminal_standard_normal": [],
                "process_standard_normal": [],
                "observation_standard_normal": [],
            },
        }
        root_seed = tf.constant(constants["root_seed"], tf.int32)
        for replication in range(coverage_count):
            replication_seed = tf.random.experimental.stateless_fold_in(
                root_seed, tf.constant(10000 + replication, tf.int32), alg="philox"
            )
            coverage_banks = [
                oracle.make_scalar_lgssm_innovation_bank(
                    chain_count=int(constants["chain_count_per_arm"]),
                    draw_count=int(constants["draw_count_per_chain"]),
                    forecast_replication_count=int(
                        constants["forecast_replication_count"]
                    ),
                    seed=replication_seed,
                    arm_id=arm_id,
                    horizon=horizon,
                )
                for arm_id in (1, 2)
            ]
            for arm_id, bank in zip((1, 2), coverage_banks):
                for name in coverage_bank_rows[arm_id]:
                    coverage_bank_rows[arm_id][name].append(
                        getattr(bank, name)
                    )
            coverage_left_rows.append(
                oracle.simulate_scalar_lgssm_forecast(
                    parameters, coverage_banks[0], horizon=horizon, jit_compile=True
                ).observations
            )
            coverage_right_rows.append(
                oracle.simulate_scalar_lgssm_forecast(
                    parameters, coverage_banks[1], horizon=horizon, jit_compile=True
                ).observations
            )
        coverage_left_all = tf.stack(coverage_left_rows)
        coverage_right_all = tf.stack(coverage_right_rows)
        coverage_bank_stacked = {
            name: tf.stack(
                [
                    tf.stack(coverage_bank_rows[1][name]),
                    tf.stack(coverage_bank_rows[2][name]),
                ],
                axis=1,
            )
            for name in coverage_bank_rows[1]
        }
    else:
        coverage_bank_stacked = _stacked_bank_from_cpu_reference(
            cpu_payload, "coverage_innovation_banks", tf=tf
        )
        coverage_left_rows = []
        coverage_right_rows = []
        for replication in range(coverage_count):
            coverage_banks = [
                oracle.ScalarLGSSMInnovationBank(
                    terminal_standard_normal=coverage_bank_stacked[
                        "terminal_standard_normal"
                    ][replication, arm_index],
                    process_standard_normal=coverage_bank_stacked[
                        "process_standard_normal"
                    ][replication, arm_index],
                    observation_standard_normal=coverage_bank_stacked[
                        "observation_standard_normal"
                    ][replication, arm_index],
                    root_seed=tf.constant([0, 0], tf.int32),
                    arm_id=arm_id,
                )
                for arm_index, arm_id in enumerate((1, 2))
            ]
            coverage_left_rows.append(
                oracle.simulate_scalar_lgssm_forecast(
                    parameters, coverage_banks[0], horizon=horizon, jit_compile=True
                ).observations
            )
            coverage_right_rows.append(
                oracle.simulate_scalar_lgssm_forecast(
                    parameters, coverage_banks[1], horizon=horizon, jit_compile=True
                ).observations
            )
        coverage_left_all = tf.stack(coverage_left_rows)
        coverage_right_all = tf.stack(coverage_right_rows)
    expected_coverage_shape = (
        coverage_count,
        int(constants["chain_count_per_arm"]),
        int(constants["draw_count_per_chain"]),
        int(constants["forecast_replication_count"]),
        horizon,
    )
    if (
        tuple(coverage_left_all.shape) != expected_coverage_shape
        or tuple(coverage_right_all.shape) != expected_coverage_shape
    ):
        raise ContractError("persisted coverage observations have the wrong hierarchy")
    expected_terminal_bank_shape = (
        coverage_count,
        2,
        int(constants["chain_count_per_arm"]),
        int(constants["draw_count_per_chain"]),
        int(constants["forecast_replication_count"]),
    )
    expected_extended_bank_shape = (*expected_terminal_bank_shape, horizon)
    if (
        tuple(coverage_bank_stacked["terminal_standard_normal"].shape)
        != expected_terminal_bank_shape
        or tuple(coverage_bank_stacked["process_standard_normal"].shape)
        != expected_extended_bank_shape
        or tuple(coverage_bank_stacked["observation_standard_normal"].shape)
        != expected_extended_bank_shape
    ):
        raise ContractError("coverage_innovation_banks has the wrong hierarchy")
    coverage_provenance_rows = []
    for replication in range(coverage_count):
        coverage_replication_banks = [
            oracle.ScalarLGSSMInnovationBank(
                terminal_standard_normal=coverage_bank_stacked[
                    "terminal_standard_normal"
                ][replication, arm_index],
                process_standard_normal=coverage_bank_stacked[
                    "process_standard_normal"
                ][replication, arm_index],
                observation_standard_normal=coverage_bank_stacked[
                    "observation_standard_normal"
                ][replication, arm_index],
                root_seed=tf.constant([0, 0], tf.int32),
                arm_id=arm_id,
            )
            for arm_index, arm_id in enumerate((1, 2))
        ]
        coverage_hashes_left = _bank_tensor_hashes(coverage_replication_banks[0])
        coverage_hashes_right = _bank_tensor_hashes(coverage_replication_banks[1])
        coverage_domain_nonreuse = _pairwise_domain_nonreuse(
            coverage_hashes_left, coverage_hashes_right
        )
        coverage_provenance_rows.append(
            {
                "replication": replication,
                "left_tensor_hashes": coverage_hashes_left,
                "right_tensor_hashes": coverage_hashes_right,
                "domain_separation_nonreuse_verified": coverage_domain_nonreuse,
            }
        )
        coverage_left = statistics.standardize_forecast_paths(
            coverage_left_all[replication],
            analytic.observation_mean,
            tf.sqrt(analytic.observation_variance),
            scale_floor=tf.constant(2.0**-40, tf.float64),
            jit_compile=True,
            allow_floor_use=False,
        )
        coverage_right = statistics.standardize_forecast_paths(
            coverage_right_all[replication],
            analytic.observation_mean,
            tf.sqrt(analytic.observation_variance),
            scale_floor=tf.constant(2.0**-40, tf.float64),
            jit_compile=True,
            allow_floor_use=False,
        )
        coverage_statistic = statistics.cross_chain_linear_mmd(
            coverage_left,
            coverage_right,
            bandwidths=bandwidths,
            mixture_weights=weights,
            chain_pair_schedule=schedule,
            independent_arm_banks_verified=coverage_domain_nonreuse,
            stationarity_verified=True,
            mixing_verified=True,
            jit_compile=True,
        )
        coverage_interval = statistics.cross_chain_mmd_upper_interval(
            coverage_statistic,
            mmd_alpha=mmd_alpha,
            block_length=int(constants["block_length"]),
            jit_compile=True,
        )
        if not coverage_interval.inference_admissible:
            raise ContractError("null-coverage replicate produced an inadmissible MMD interval")
        if float(coverage_interval.lower) <= 0.0 <= float(coverage_interval.upper):
            coverage_successes += 1
    domain_ledger = _bank_domain_ledger(
        (bank_hashes_left, bank_hashes_right), coverage_provenance_rows
    )
    domain_hashes = [row["raw_little_endian_sha256"] for row in domain_ledger]
    global_domain_nonreuse = len(domain_hashes) == len(set(domain_hashes))
    if not global_domain_nonreuse:
        raise ContractError("innovation tensor reused across purpose/domain/family")
    seed_domain_ledger = _seed_domain_ledger(tf, constants["root_seed"], domain_ledger)
    family_seeds = [tuple(row["family_seed"]) for row in seed_domain_ledger]
    global_seed_domain_nonreuse = len(family_seeds) == len(set(family_seeds))
    if not global_seed_domain_nonreuse:
        raise ContractError("Philox family seed reused across purpose/domain/family")
    coverage_confidence_alpha = float(mmd_alpha)
    if coverage_successes == 0:
        coverage_lower = 0.0
    else:
        coverage_lower = float(
            tfp.distributions.Beta(
                tf.constant(float(coverage_successes), tf.float64),
                tf.constant(float(coverage_count - coverage_successes + 1), tf.float64),
            ).quantile(tf.constant(coverage_confidence_alpha, tf.float64))
        )
    nominal_coverage = 1.0 - float(mmd_alpha)
    required_coverage_lower = nominal_coverage - _hex_float(constants["coverage_slack_hex"])
    coverage_passed = coverage_lower >= required_coverage_lower

    mean_shift = tf.constant(
        _hex_float(fixture["controlled_alternatives"]["mean_shift_hex"]), tf.float64
    )
    variance_increment = tf.constant(
        _hex_float(fixture["controlled_alternatives"]["variance_increment_hex"]), tf.float64
    )
    skew_coefficient = tf.constant(
        _hex_float(fixture["controlled_alternatives"]["skew_coefficient_hex"]), tf.float64
    )
    dependence_correlation = tf.constant(
        _hex_float(fixture["controlled_alternatives"]["dependence_correlation_hex"]), tf.float64
    )
    mean_alternative = right_paths + mean_shift
    variance_alternative = analytic.observation_mean + (
        right_paths - analytic.observation_mean
    ) * tf.sqrt(
        (analytic.observation_variance + variance_increment)
        / analytic.observation_variance
    )
    centered_right = right_paths - analytic.observation_mean
    skew_alternative = right_paths + skew_coefficient * (
        tf.square(centered_right) - analytic.observation_variance
    )
    standardized_base = (right_paths - analytic.observation_mean) / tf.sqrt(
        analytic.observation_variance
    )
    common = standardized_base[..., :1]
    correlations = analytic.observation_covariance[:, 0] / tf.sqrt(
        analytic.observation_variance * analytic.observation_variance[0]
    )
    independent_weight = tf.sqrt(1.0 - tf.square(dependence_correlation))
    normalization = tf.sqrt(
        1.0
        + 2.0
        * independent_weight
        * dependence_correlation
        * correlations
    )
    dependence_standardized = (
        independent_weight * standardized_base + dependence_correlation * common
    ) / normalization
    dependence_alternative = analytic.observation_mean + tf.sqrt(
        analytic.observation_variance
    ) * dependence_standardized
    alternative_paths = {
        "mean": mean_alternative,
        "variance": variance_alternative,
        "skew": skew_alternative,
        "dependence": dependence_alternative,
    }
    alternative_records = []
    alternative_tensor_sections: dict[str, list[dict[str, Any]]] = {}
    for alternative_name, paths in alternative_paths.items():
        record, sections = _alternative_evidence(
            name=alternative_name,
            alternative_paths=paths,
            left_paths=left_paths,
            bootstrap_left_paths=bootstrap_left_paths,
            indices_right=indices_right,
            left_summary=left_summary,
            config=config,
            analytic=analytic,
            bandwidths=bandwidths,
            weights=weights,
            schedule=schedule,
            margins=margins,
            mmd_tolerance=mmd_tolerance,
            total_alpha=total_alpha,
            feature_alpha=feature_alpha,
            mmd_alpha=mmd_alpha,
            independent_arm_banks_verified=global_domain_nonreuse,
            constants=constants,
            statistics=statistics,
            tf=tf,
        )
        alternative_records.append(record)
        alternative_tensor_sections.update(sections)
    alternative_diagnostics = {
        "mechanics": {
            "mean_shift_mean_residual": _max_abs(
                tf,
                tf.reduce_mean(mean_alternative - right_paths, axis=[0, 1, 2]),
                tf.fill([horizon], mean_shift),
            ),
            "variance_log_variance_direction": float(
                tf.reduce_max(
                    statistics.summarize_forecast_paths(
                        variance_alternative, config
                    ).log_variances
                    - right_summary.log_variances
                )
            ),
            "skew_third_moment_change": float(
                tf.reduce_max(
                    tf.abs(
                        statistics.summarize_forecast_paths(
                            skew_alternative, config
                        ).central_moments[0]
                        - right_summary.central_moments[0]
                    )
                )
            ),
            "dependence_covariance_change": float(
                tf.reduce_max(
                    tf.abs(
                        statistics.summarize_forecast_paths(
                            dependence_alternative, config
                        ).cross_horizon_covariance
                        - right_summary.cross_horizon_covariance
                    )
                )
            ),
        },
        "records": alternative_records,
        "policy": "valid_underpowered_is_repair_trigger_not_hard_veto",
    }
    alternatives_passed = (
        alternative_diagnostics["mechanics"]["mean_shift_mean_residual"]
        <= 512.0 * 2.0**-52
        and alternative_diagnostics["mechanics"]["variance_log_variance_direction"] > 0.0
        and alternative_diagnostics["mechanics"]["skew_third_moment_change"] > 0.0
        and alternative_diagnostics["mechanics"]["dependence_covariance_change"] > 0.0
        and all(record["valid"] for record in alternative_records)
        and next(record for record in alternative_records if record["name"] == "mean")
        ["decision"]["status"]
        == "MATERIAL_DIFFERENCE"
        and next(record for record in alternative_records if record["name"] == "variance")
        ["decision"]["status"]
        == "MATERIAL_DIFFERENCE"
    )

    analytic_program = oracle.scalar_lgssm_analytic_compiled_program(
        quantile_count=int(quantile_probabilities.shape[0])
    )
    simulation_program = oracle.scalar_lgssm_simulation_compiled_program(
        int(constants["chain_count_per_arm"]),
        int(constants["draw_count_per_chain"]),
        int(constants["forecast_replication_count"]),
    )
    compiler_evidence = [
        _compiler_row(
            "scalar_lgssm_analytic",
            analytic_program,
            (parameters.as_tensor(), quantile_probabilities),
        ),
        _compiler_row(
            "scalar_lgssm_simulation",
            simulation_program,
            (
                parameters.as_tensor(),
                bank_left.terminal_standard_normal,
                bank_left.process_standard_normal,
                bank_left.observation_standard_normal,
            ),
        ),
    ]
    expected_device_fragment = "CPU:" if mode == "cpu-reference" else "GPU:"
    device_passed = all(
        all(expected_device_fragment in device for device in row["output_devices"])
        for row in compiler_evidence
    ) and all(
        expected_device_fragment in str(tensor.device)
        for tensor in (left_summary.means, mmd.squared_mmd_u, cross_chain.squared_mmd_linear)
    )

    summary_dict = _dataclass_dict(left_summary)
    mmd_dict = _dataclass_dict(mmd)
    cross_dict = _dataclass_dict(cross_chain)
    left_bank_dict = _dataclass_dict(bank_left)
    right_bank_dict = _dataclass_dict(bank_right)

    tensor_sections = {
        "analytic": _analytic_rows(analytic),
        "manual_analytic": [_tensor_row(name, value) for name, value in manual.items()],
        "simulation_left": [_tensor_row("observations", left_paths)],
        "simulation_right": [_tensor_row("observations", right_paths)],
        "summary": [
            _tensor_row(name, value)
            for name, value in summary_dict.items()
            if hasattr(value, "dtype") and getattr(value.dtype, "is_floating", False)
        ],
        "quadratic_mmd": [
            _tensor_row(name, value)
            for name, value in mmd_dict.items()
            if hasattr(value, "dtype") and getattr(value.dtype, "is_floating", False)
        ],
        "cross_chain_mmd": [
            _tensor_row(name, value)
            for name, value in cross_dict.items()
            if hasattr(value, "dtype") and getattr(value.dtype, "is_floating", False)
        ],
        "innovation_bank_left": [
            _tensor_row(name, value)
            for name, value in left_bank_dict.items()
            if hasattr(value, "dtype") and getattr(value.dtype, "is_floating", False)
        ],
        "innovation_bank_right": [
            _tensor_row(name, value)
            for name, value in right_bank_dict.items()
            if hasattr(value, "dtype") and getattr(value.dtype, "is_floating", False)
        ],
        "coverage_innovation_banks": [
            _tensor_row(name, coverage_bank_stacked[name])
            for name in sorted(coverage_bank_stacked)
        ],
        "resampling_indices_left": [
            _int_tensor_row(name, value)
            for name, value in index_dict_left.items()
            if hasattr(value, "dtype") and getattr(value.dtype, "is_integer", False)
        ],
        "resampling_indices_right": [
            _int_tensor_row(name, value)
            for name, value in index_dict_right.items()
            if hasattr(value, "dtype") and getattr(value.dtype, "is_integer", False)
        ],
        "coverage_observations": [
            _tensor_row("left", coverage_left_all),
            _tensor_row("right", coverage_right_all),
        ],
        "feature_inputs": [
            _tensor_row("base_features", base_features),
            _tensor_row("bootstrap_features", bootstrap_features),
            _tensor_row("margins", margins),
            _tensor_row("mmd_tolerance", mmd_tolerance),
        ],
        "feature_interval": _interval_rows(feature_intervals),
        "mmd_interval": _interval_rows(mmd_interval),
        "decision_branch_inputs": [
            _tensor_row("zero_estimate", zero_estimate),
            _tensor_row("narrow_standard_error", narrow_se),
            _tensor_row("inconclusive_standard_error", margins),
            _tensor_row("branch_mmd_tolerance", branch_mmd_tolerance),
        ],
    }
    tensor_sections.update(alternative_tensor_sections)

    checks_by_name = {
        "analytic_formula_exact": _check_row(
            "analytic_formula_exact",
            passed=analytic_passed,
            role="promotion_criterion_and_hard_veto",
            residual=max(row["residual"] for row in deterministic_residuals.values()),
            threshold=max(row["threshold"] for row in deterministic_residuals.values()),
        ),
        "analytic_covariance_valid": _check_row(
            "analytic_covariance_valid",
            passed=covariance_passed,
            role="hard_veto",
            residual=max(
                state_symmetry,
                observation_symmetry,
                max(0.0, -minimum_state_eigenvalue),
                max(0.0, -minimum_observation_eigenvalue),
            ),
            threshold=max(state_psd_tolerance, observation_psd_tolerance),
        ),
        "direct_simulation_replay": _check_row(
            "direct_simulation_replay", passed=replay_passed, role="promotion_criterion", residual=replay_residual, threshold=0.0
        ),
        "monte_carlo_oracle_agreement": _check_row(
            "monte_carlo_oracle_agreement",
            passed=monte_carlo_passed,
            role="promotion_criterion_uncertainty_aware",
            residual=float(tf.reduce_max(mean_z)),
            threshold=float(mean_critical),
        ),
        "summary_statistics": _check_row(
            "summary_statistics",
            passed=summary_passed,
            role="promotion_criterion",
            residual=max(summary_residuals.values()),
            threshold=max(summary_thresholds.values()),
        ),
        "standardization": _check_row("standardization", passed=standardization_passed, role="hard_veto"),
        "quadratic_mmd_roles": _check_row("quadratic_mmd_roles", passed=mmd_roles_passed, role="promotion_and_role_veto"),
        "signed_u_form_preserved": _check_row(
            "signed_u_form_preserved", passed=signed_u_passed, role="role_veto"
        ),
        "common_random_numbers_excluded": _check_row("common_random_numbers_excluded", passed=not paired_diagnostic.inference_admissible, role="hard_veto"),
        "cross_chain_schedule": _check_row("cross_chain_schedule", passed=cross_chain_passed, role="hard_veto"),
        "cross_chain_inference_admission": _check_row("cross_chain_inference_admission", passed=mmd_interval_passed, role="promotion_and_hard_veto"),
        "cross_chain_null_coverage": _check_row(
            "cross_chain_null_coverage",
            passed=coverage_passed,
            role="promotion_fixture_exact_binomial_lower_bound",
            residual=coverage_lower,
            threshold=required_coverage_lower,
        ),
        "hierarchical_indices": _check_row("hierarchical_indices", passed=resampling_passed, role="promotion_and_hard_veto"),
        "joint_alpha_allocation": _check_row("joint_alpha_allocation", passed=joint_alpha_passed, role="hard_veto"),
        "simultaneous_intervals": _check_row("simultaneous_intervals", passed=intervals_passed, role="promotion_and_hard_veto"),
        "controlled_alternatives": _check_row("controlled_alternatives", passed=alternatives_passed, role="promotion_fixture_sensitivity"),
        "decision_fail_closed": _check_row("decision_fail_closed", passed=decision_passed, role="hard_veto"),
        "fixture_binding": _check_row(
            "fixture_binding",
            passed=(
                fixture.get("evidence_signature") == _evidence_signature(fixture)
                and fixture.get("boundary_sha256") == _sha256(BOUNDARY_PATH)
                and boundary.get("harness_review_anchor_sha256")
                == _sha256(HARNESS_ANCHOR_PATH)
            ),
            role="hard_veto",
        ),
        "source_binding": _check_row(
            "source_binding",
            passed=all(
                _sha256(path) == expected
                for path, expected in EXPECTED_SOURCE_HASHES.items()
            ),
            role="hard_veto",
        ),
        "compiler_hlo": _check_row("compiler_hlo", passed=len(compiler_evidence) == 2, role="hard_veto_oracle_programs"),
        "device_placement": _check_row("device_placement", passed=device_passed, role="hard_veto"),
    }
    checks = [checks_by_name[name] for name in CHECK_NAMES]
    failed = [row["name"] for row in checks if not row["passed"]]
    if failed:
        raise ContractError(f"A3 runtime hard checks failed: {failed}")
    cpu_binding = None
    cpu_gpu_parity = None
    if mode == "gpu-xla-canary":
        if (
            cpu_reference is None
            or cpu_payload is None
            or cpu_reference_sha256 is None
            or _sha256(cpu_reference) != cpu_reference_sha256
        ):
            raise ContractError("GPU mode lacks a verified CPU replay authority")
        cpu_binding = {
            "path": cpu_reference.as_posix(),
            "file_sha256": cpu_reference_sha256,
            "evidence_signature": cpu_payload["evidence_signature"],
        }
        parity_residual, parity_threshold = _tensor_section_parity(
            tensor_sections, cpu_payload["tensor_sections"]
        )
        cpu_gpu_parity = {
            "maximum_absolute_residual": parity_residual,
            "maximum_scale_aware_threshold": parity_threshold,
            "tolerance_multiplier": CPU_GPU_TOLERANCE_MULTIPLIER,
            "passed": parity_residual <= parity_threshold,
        }

    completed_at = _utc_now()
    wall_time = time.monotonic() - started
    payload = {
        "schema_version": CPU_SCHEMA if mode == "cpu-reference" else GPU_SCHEMA,
        "artifact_role": (
            "phase_a3_cpu_hidden_oracle_reference"
            if mode == "cpu-reference"
            else "phase_a3_trusted_gpu_xla_oracle_canary"
        ),
        "status": CPU_STATUS if mode == "cpu-reference" else GPU_STATUS,
        "created_at_utc": completed_at,
        "run_manifest": _manifest(
            mode=mode,
            fixture=fixture,
            output=output,
            started=started_at,
            completed=completed_at,
            wall_time=wall_time,
            tf=tf,
            reviewed_command_key=reviewed_command_key,
            reviewed_command=reviewed_command,
        ),
        "boundary_binding": {
            "path": BOUNDARY_PATH.as_posix(),
            "file_sha256": _sha256(BOUNDARY_PATH),
            "evidence_signature": boundary["evidence_signature"],
        },
        "fixture_binding": {
            "path": FIXTURE_PATH.as_posix(),
            "file_sha256": _sha256(FIXTURE_PATH),
            "evidence_signature": fixture["evidence_signature"],
            "classification": "A3_TEST_FIXTURE_ONLY_NOT_A4_FROZEN",
        },
        "source_files": _source_rows(),
        "cpu_reference_binding": cpu_binding,
        "cpu_gpu_parity": cpu_gpu_parity,
        "tensor_sections": tensor_sections,
        "bank_provenance": {
            "authority": "materialized_float64_tensor_rows_and_raw_hashes",
            "seed_metadata_is_replay_authority": False,
            "generation_mode": (
                "cpu_stateless_philox_materialization"
                if mode == "cpu-reference"
                else "gpu_reconstruction_from_cpu_artifact_values"
            ),
            "cpu_generation_root_seed": constants["root_seed"],
            "arm_rows": [
                {
                    "arm_id": arm_id,
                    "section": section,
                    "role": "domain_separated_oracle_arm_not_probabilistic_independence",
                    "tensor_hashes": hashes,
                }
                for arm_id, section, hashes in (
                    (1, "innovation_bank_left", bank_hashes_left),
                    (2, "innovation_bank_right", bank_hashes_right),
                )
            ],
            "coverage_section": "coverage_innovation_banks",
            "coverage_arm_axis": {"axis": 1, "arm_ids": [1, 2]},
            "coverage_stacked_tensor_hashes": _stacked_bank_tensor_hashes(
                coverage_bank_stacked
            ),
            "coverage_replication_rows": coverage_provenance_rows,
            "domain_separation_ledger": domain_ledger,
            "global_raw_tensor_hash_nonreuse": global_domain_nonreuse,
            "seed_domain_ledger": seed_domain_ledger,
            "global_family_seed_nonreuse": global_seed_domain_nonreuse,
            "seed_domain_attestation_role": (
                "cpu_only_seed_domain_generation_attestation_not_cross_backend_replay_authority"
            ),
            "domain_separation_claim": (
                "raw_tensor_hash_global_nonreuse_only_not_probabilistic_independence"
            ),
        },
        "resampling_provenance": {
            "authority": "materialized_indices_replay_authority",
            "seed_metadata_is_replay_authority": False,
            "cpu_seed_derivation": "tf.random.experimental.stateless_split(root_seed,2,alg=philox)",
            "cpu_root_seed": constants["root_seed"],
            "sections": ["resampling_indices_left", "resampling_indices_right"],
            "arm_tensor_hashes": {
                "left": {
                    row["name"]: row["raw_little_endian_sha256"]
                    for row in tensor_sections["resampling_indices_left"]
                },
                "right": {
                    row["name"]: row["raw_little_endian_sha256"]
                    for row in tensor_sections["resampling_indices_right"]
                },
            },
        },
        "statistical_metadata": {
            "analytic_status": _status_text(analytic.status),
            "analytic_log_variance_valid": bool(analytic.log_variance_valid),
            "summary_statuses": [
                _status_text(left_summary.status),
                _status_text(right_summary.status),
            ],
            "quadratic_mmd": {
                "status": _status_text(mmd.status),
                "sampling_contract": mmd.sampling_contract,
                "iid_samples_verified": mmd.iid_samples_verified,
                "independent_arm_banks_verified": mmd.independent_arm_banks_verified,
                "inference_admissible": mmd.inference_admissible,
            },
            "cross_chain": {
                "status": _status_text(cross_chain.status),
                "stationarity_verified": cross_chain.stationarity_verified,
                "mixing_verified": cross_chain.mixing_verified,
                "mechanics_only": cross_chain.mechanics_only,
                "inference_admissible": cross_chain.inference_admissible,
            },
            "feature_interval": {
                **_interval_metadata(feature_intervals),
                "inference_admissible": feature_intervals.inference_admissible,
            },
            "mmd_interval": {
                **_interval_metadata(mmd_interval),
                "inference_admissible": mmd_interval.inference_admissible,
            },
            "decision_input_derivations": {
                "margins": "concat(fill(horizon,mean_margin_hex),fill(horizon,log_variance_margin_hex))",
                "mmd_tolerance": "float.fromhex(fixture_constants.mmd_tolerance_hex)",
                "zero_estimate": "zeros([2*horizon],float64)",
                "narrow_standard_error": "margins/horizon**2",
                "inconclusive_standard_error": "margins",
                "branch_mmd_tolerance": "mmd_tolerance+abs(main_mmd_interval.upper)",
                "invalid_total_alpha": "total_alpha/2",
            },
        },
        "compiler_evidence": compiler_evidence,
        "deterministic_residuals": deterministic_residuals,
        "monte_carlo_diagnostics": {
            "pooled_path_count": pooled_count,
            "maximum_mean_standard_error_units": float(tf.reduce_max(mean_z)),
            "mean_simultaneous_critical_value": float(mean_critical),
            "minimum_variance_chi_square_ratio": float(tf.reduce_min(variance_ratio)),
            "maximum_variance_chi_square_ratio": float(tf.reduce_max(variance_ratio)),
            "variance_chi_square_lower": float(variance_lower),
            "variance_chi_square_upper": float(variance_upper),
            "acceptance_rule": "bonferroni_normal_means_and_chi_square_variances_at_frozen_feature_alpha",
        },
        "alternative_diagnostics": alternative_diagnostics,
        "role_ledger": {
            "quadratic_mmd_u": "descriptive_only_even_iid_fixture",
            "quadratic_mmd_v": "explanatory_only",
            "cross_chain_linear_mmd": "inference_route_fixture_only",
            "high_moments_quantiles_covariance": "explanatory_only",
            "continuous_residuals_runtime": "descriptive_only_after_hard_veto",
        },
        "decision_rows": [
            _decision_row("independent_identical_law_fixture", observed_decision),
            _decision_row("mechanics_only_hard_veto", mechanics_decision),
            _decision_row("invalid_joint_alpha_hard_veto", invalid_alpha_decision),
            _decision_row("synthetic_pass_branch", branch_pass),
            _decision_row("synthetic_material_difference_branch", branch_material),
            _decision_row("synthetic_inconclusive_branch", branch_inconclusive),
        ],
        "uncertainty": {
            "coverage_replication_count": int(constants["coverage_replication_count"]),
            "coverage_interval": "exact_binomial_clopper_pearson",
            "coverage_successes": coverage_successes,
            "coverage_lower_bound": coverage_lower,
            "coverage_required_lower_bound": required_coverage_lower,
            "coverage_confidence_alpha": coverage_confidence_alpha,
            "coverage_slack_hex": constants["coverage_slack_hex"],
            "bootstrap_count": int(constants["bootstrap_count"]),
            "block_length": int(constants["block_length"]),
            "classification": "A3_TEST_FIXTURE_ONLY_NOT_A4_FROZEN",
        },
        "contract_checks": checks,
        "evidence_signature": "",
        "nonclaims": NONCLAIMS,
    }
    payload["evidence_signature"] = _evidence_signature(payload)
    return payload


def _write_log(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"status={payload['status']}",
        f"schema_version={payload['schema_version']}",
        f"evidence_signature={payload['evidence_signature']}",
    ]
    lines.extend(
        f"check.{row['name']}={'PASS' if row['passed'] else 'FAIL'}"
        for row in payload["contract_checks"]
    )
    absolute = ROOT / path
    absolute.parent.mkdir(parents=True, exist_ok=True)
    absolute.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("cpu-reference", "gpu-xla-canary"), required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--cpu-reference", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--log-path", type=Path, required=True)
    args = parser.parse_args()
    os.chdir(ROOT)
    payload = _artifact_payload(
        mode=args.mode,
        fixture_path=args.fixture,
        cpu_reference=args.cpu_reference,
        output=args.output,
    )
    _strict_write(args.output, payload)
    _write_log(args.log_path, payload)
    print(_canonical_bytes({"status": payload["status"], "evidence_signature": payload["evidence_signature"]}).decode("utf-8"))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ContractError as exc:
        print(f"A3_CONTRACT_ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
