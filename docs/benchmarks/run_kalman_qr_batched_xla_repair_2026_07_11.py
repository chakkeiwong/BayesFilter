#!/usr/bin/env python
"""Run method-isolated Kalman QR repair schedules under the v4 contract."""

from __future__ import annotations

import argparse
import base64
import contextlib
import copy
import fcntl
import hashlib
import math
import os
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import kalman_qr_benchmark_contract as contract


PYTHON = Path("/home/ubuntu/anaconda3/envs/tfgpu/bin/python")
BENCHMARK = REPO_ROOT / "scripts/benchmark_kalman_qr_parameter_count_scaling.py"
PLAN_PATH = (
    "docs/plans/"
    "bayesfilter-kalman-qr-batched-xla-repair-phase5-measurement-subplan-2026-07-11.md"
)
RESULT_PATH = (
    "docs/plans/"
    "bayesfilter-kalman-qr-batched-xla-repair-phase5-measurement-result-2026-07-11.md"
)
DEFAULT_OUTPUT_DIR = Path("/tmp/kalman_qr_phase5_measurement")
FIXTURE_CONTRACT_VERSION = contract.FIXTURE_CONTRACT_VERSION
PARAMETER_BATCH_VERSION = contract.PARAMETER_BATCH_VERSION
OBSERVATION_GENERATION_VERSION = contract.OBSERVATION_GENERATION_VERSION
TIMING_BOUNDARY_VERSION = contract.TIMING_BOUNDARY_VERSION
PHASE5_SMOKE_SCHEMA = "bayesfilter.kalman_qr_batched_xla_repair.phase5.measurement.v1"
PHASE5_SMOKE_OUTPUT = (
    "docs/benchmarks/"
    "kalman_qr_batched_xla_repair_phase5_measurement_smoke_2026-07-11.json"
)
PHASE5_NONCLAIMS = (
    "no pure compilation-time estimate",
    "no method speed ranking",
    "no CPU or GPU scalability claim",
    "no GPU readiness claim",
    "no HMC or posterior correctness claim",
    "no default, production, or scientific validity claim",
)
PHASE6_PLAN_PATH = contract.PHASE6_PLAN_RELATIVE
PHASE6_RESULT_PATH = (
    "docs/plans/"
    "bayesfilter-kalman-qr-batched-xla-repair-phase6-cpu-xla-gates-result-2026-07-11.md"
)
PHASE6_WORK_DIR = Path(contract.PHASE6_WORK_ROOT)
PHASE6_REQUIRED_DISCOVERY_PATHS = contract.PHASE6_REQUIRED_SOURCE_PATHS
PHASE6_BUDGET_STATE_SCHEMA = (
    "bayesfilter.kalman_qr_batched_xla_repair.phase6.budget_state.v2"
)
PHASE6_BUDGET_LEASE_SCHEMA = (
    "bayesfilter.kalman_qr_batched_xla_repair.phase6.budget_lease.v1"
)
PHASE6_BUDGET_COMMAND_ORDER = {
    "gate_b": ("trace_census_and_pilot",),
    "gate_c": ("scalar_references", "remaining_lattice"),
}
PHASE4_EVIDENCE_PATH = REPO_ROOT / (
    "docs/benchmarks/kalman_qr_batched_xla_repair_phase4_autodiff_2026-07-11.json"
)
PHASE5_EVIDENCE_PATH = REPO_ROOT / PHASE5_SMOKE_OUTPUT


class Phase6OuterTermination(BaseException):
    """Raised after a scoped outer SIGTERM callback has run."""


class Phase6TerminationController:
    """Defer SIGTERM only across process-ownership and durable-write windows."""

    def __init__(self, on_terminate: Callable[[int], None]):
        self._on_terminate = on_terminate
        self._defer_depth = 0
        self._pending_signum: int | None = None
        self._callback_called = False

    def receive(self, signum: int) -> None:
        if self._defer_depth:
            if self._pending_signum is None:
                self._pending_signum = signum
            return
        self._raise(signum)

    def _raise(self, signum: int) -> None:
        if not self._callback_called:
            self._callback_called = True
            self._on_terminate(signum)
        raise Phase6OuterTermination(f"received signal {signum}")

    def deliver_pending(self) -> None:
        if self._defer_depth or self._pending_signum is None:
            return
        signum = self._pending_signum
        self._pending_signum = None
        self._raise(signum)

    @contextlib.contextmanager
    def defer(self):
        self._defer_depth += 1
        failed = False
        try:
            yield
        except BaseException:
            failed = True
            raise
        finally:
            self._defer_depth -= 1
            if not failed and self._defer_depth == 0:
                self.deliver_pending()


@contextlib.contextmanager
def phase6_outer_sigterm_guard(
    on_terminate: Callable[[int], None],
):
    prior = signal.getsignal(signal.SIGTERM)
    controller = Phase6TerminationController(on_terminate)

    def handle(signum: int, frame: Any) -> None:
        del frame
        controller.receive(signum)

    signal.signal(signal.SIGTERM, handle)
    try:
        yield controller
        controller.deliver_pending()
    finally:
        signal.signal(signal.SIGTERM, prior)


def _phase6_budget_state_valid(state: Any) -> bool:
    if not isinstance(state, Mapping) or set(state) != {
        "schema",
        "authority_id",
        "gate",
        "hard_ceiling_seconds",
        "boot_id",
        "started_ns",
        "deadline_ns",
        "last_observed_ns",
        "elapsed_seconds",
        "state",
        "update_index",
        "commands",
    }:
        return False
    gate = state.get("gate")
    order = PHASE6_BUDGET_COMMAND_ORDER.get(gate)
    commands = state.get("commands")
    if (
        state.get("schema") != PHASE6_BUDGET_STATE_SCHEMA
        or not isinstance(state.get("authority_id"), str)
        or len(state["authority_id"]) != 64
        or any(character not in "0123456789abcdef" for character in state["authority_id"])
        or order is None
        or not isinstance(state.get("hard_ceiling_seconds"), (int, float))
        or isinstance(state.get("hard_ceiling_seconds"), bool)
        or not math.isfinite(float(state["hard_ceiling_seconds"]))
        or state["hard_ceiling_seconds"] <= 0
        or not isinstance(state.get("boot_id"), str)
        or not state["boot_id"]
        or type(state.get("started_ns")) is not int
        or state["started_ns"] < 0
        or type(state.get("deadline_ns")) is not int
        or state["deadline_ns"]
        != state["started_ns"] + int(float(state["hard_ceiling_seconds"]) * 1.0e9)
        or type(state.get("last_observed_ns")) is not int
        or not state["started_ns"] <= state["last_observed_ns"] <= state["deadline_ns"]
        or not isinstance(state.get("elapsed_seconds"), (int, float))
        or isinstance(state.get("elapsed_seconds"), bool)
        or not math.isfinite(float(state["elapsed_seconds"]))
        or state["elapsed_seconds"] < 0
        or state.get("state") not in {"running", "closed"}
        or type(state.get("update_index")) is not int
        or state["update_index"] < 0
        or not isinstance(commands, list)
        or not commands
        or len(commands) > len(order)
        or [command.get("name") for command in commands if isinstance(command, Mapping)]
        != list(order[: len(commands)])
    ):
        return False
    running_count = 0
    previous_finished = state["started_ns"]
    for index, command in enumerate(commands):
        if not isinstance(command, Mapping) or set(command) != {
            "name",
            "started_ns",
            "finished_ns",
            "elapsed_seconds",
            "state",
        }:
            return False
        started = command.get("started_ns")
        finished = command.get("finished_ns")
        elapsed = command.get("elapsed_seconds")
        command_state = command.get("state")
        if type(started) is not int or started < previous_finished or started > state["last_observed_ns"]:
            return False
        if command_state == "running":
            running_count += 1
            if index != len(commands) - 1 or finished is not None or elapsed is not None:
                return False
        elif command_state == "closed":
            if (
                type(finished) is not int
                or finished < started
                or not isinstance(elapsed, (int, float))
                or isinstance(elapsed, bool)
                or not math.isfinite(float(elapsed))
                or abs(float(elapsed) - (finished - started) / 1.0e9) > 1.0e-9
                or finished > state["last_observed_ns"]
            ):
                return False
            previous_finished = finished
        else:
            return False
    expected_update_index = 2 * len(commands) - (1 if commands[-1]["state"] == "running" else 0) - 1
    return (
        running_count <= 1
        and state["update_index"] == expected_update_index
        and state["elapsed_seconds"]
        == (state["last_observed_ns"] - state["started_ns"]) / 1.0e9
        and (state["state"] == "closed")
        == (len(commands) == len(order) and commands[-1]["state"] == "closed")
    )


def _phase6_budget_elapsed_at_last_update(
    started_ns: int, commands: Sequence[Mapping[str, Any]]
) -> float:
    last = commands[-1]
    endpoint = (
        last["finished_ns"] if last["state"] == "closed" else last["started_ns"]
    )
    return (endpoint - started_ns) / 1.0e9


def _phase6_boot_id() -> str:
    try:
        value = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()
    except OSError as exc:
        raise contract.ContractError("cannot establish Phase 6 monotonic clock epoch") from exc
    if not value:
        raise contract.ContractError("empty Phase 6 boot identity")
    return value


def _phase6_budget_lease_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.lease")


def _phase6_lease_owner_alive(record: Mapping[str, Any]) -> bool:
    pid = record.get("pid")
    if type(pid) is not int or pid <= 0 or not Path(f"/proc/{pid}").exists():
        return False
    try:
        _, start_ticks = _phase6_process_identity(pid)
    except contract.ContractError:
        return True
    return start_ticks == record.get("process_start_ticks")


class Phase6BudgetLease:
    """Linux flock plus durable owner provenance for one gate command."""

    def __init__(self, budget_path: Path, command_name: str):
        self.budget_path = budget_path
        self.path = _phase6_budget_lease_path(budget_path)
        self.command_name = command_name
        self.fd: int | None = None
        self.record: dict[str, Any] | None = None

    def __enter__(self) -> "Phase6BudgetLease":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_CREAT | os.O_RDWR | os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            self.fd = os.open(self.path, flags, 0o600)
            fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException as exc:
            if self.fd is not None:
                os.close(self.fd)
                self.fd = None
            if isinstance(exc, OSError):
                raise contract.ContractError(
                    "Phase 6 budget authority already has a live supervisor"
                ) from exc
            raise
        try:
            prior: Any = None
            raw = os.pread(self.fd, 1_000_000, 0)
            if raw:
                try:
                    prior = contract.strict_json_loads(raw.decode("utf-8"))
                except (UnicodeDecodeError, contract.ContractError) as exc:
                    raise contract.ContractError(
                        "Phase 6 budget lease record is malformed"
                    ) from exc
            if prior is not None and (
                not isinstance(prior, Mapping)
                or prior.get("schema") != PHASE6_BUDGET_LEASE_SCHEMA
                or prior.get("budget_path") != str(self.budget_path.resolve())
                or prior.get("state") not in {"active", "released"}
                or type(prior.get("generation")) is not int
            ):
                raise contract.ContractError("Phase 6 budget lease identity is invalid")
            if (
                isinstance(prior, Mapping)
                and prior.get("state") == "active"
                and _phase6_lease_owner_alive(prior)
            ):
                raise contract.ContractError(
                    "Phase 6 budget lease has an ambiguous live prior owner"
                )
            _, start_ticks = _phase6_process_identity(os.getpid())
            now = time.monotonic_ns()
            self.record = {
                "schema": PHASE6_BUDGET_LEASE_SCHEMA,
                "budget_path": str(self.budget_path.resolve()),
                "command_name": self.command_name,
                "generation": 1 if prior is None else int(prior["generation"]) + 1,
                "boot_id": _phase6_boot_id(),
                "pid": os.getpid(),
                "process_start_ticks": start_ticks,
                "state": "active",
                "acquired_ns": now,
                "released_ns": None,
            }
            self._write(self.record)
            return self
        except BaseException:
            self.record = None
            self._unlock()
            raise

    def _write(self, record: Mapping[str, Any]) -> None:
        if self.fd is None:
            raise contract.ContractError("Phase 6 budget lease is not held")
        raw = (contract.strict_json_dumps(record, indent=2) + "\n").encode("utf-8")
        os.ftruncate(self.fd, 0)
        os.pwrite(self.fd, raw, 0)
        os.fsync(self.fd)

    def assert_current(self, command_name: str) -> None:
        if self.fd is None or self.record is None or command_name != self.command_name:
            raise contract.ContractError("Phase 6 budget lease command mismatch")
        raw = os.pread(self.fd, 1_000_000, 0)
        current = contract.strict_json_loads(raw.decode("utf-8"))
        if current != self.record or current.get("state") != "active":
            raise contract.ContractError("Phase 6 budget lease changed while held")

    def _unlock(self) -> None:
        if self.fd is not None:
            try:
                fcntl.flock(self.fd, fcntl.LOCK_UN)
            finally:
                os.close(self.fd)
                self.fd = None

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        del exc_type, exc, traceback
        try:
            if self.fd is not None and self.record is not None:
                released = dict(self.record)
                released["state"] = "released"
                released["released_ns"] = time.monotonic_ns()
                self._write(released)
                self.record = released
        finally:
            self._unlock()


def phase6_budget_lease(path: Path, command_name: str) -> Phase6BudgetLease:
    return Phase6BudgetLease(path, command_name)


def _phase6_persist_budget_state(path: Path, state: Mapping[str, Any]) -> dict[str, Any]:
    if not _phase6_budget_state_valid(state):
        raise contract.ContractError("invalid Phase 6 budget state")
    contract.durable_atomic_write_json(path, state)
    reparsed = contract.read_strict_json(path)
    if reparsed != state or not _phase6_budget_state_valid(reparsed):
        raise contract.ContractError("persisted Phase 6 budget state failed reparse")
    return reparsed


def phase6_budget_state_open(
    path: Path,
    authority_id: str,
    gate: str,
    hard_ceiling_seconds: float,
    command_name: str,
    *,
    lease: Phase6BudgetLease,
    now_ns: int | None = None,
) -> dict[str, Any]:
    lease.assert_current(command_name)
    now = time.monotonic_ns() if now_ns is None else now_ns
    boot_id = _phase6_boot_id()
    order = PHASE6_BUDGET_COMMAND_ORDER.get(gate)
    if order is None or command_name not in order or type(now) is not int or now < 0:
        raise contract.ContractError("invalid Phase 6 budget command")
    if path.exists():
        state = contract.read_strict_json(path)
        if (
            not _phase6_budget_state_valid(state)
            or state["authority_id"] != authority_id
            or state["gate"] != gate
            or state["hard_ceiling_seconds"] != hard_ceiling_seconds
            or state["boot_id"] != boot_id
            or now < state["last_observed_ns"]
            or now > state["deadline_ns"]
        ):
            raise contract.ContractError("Phase 6 budget state authority or clock drift")
        names = [command["name"] for command in state["commands"]]
        if command_name in names:
            if names[-1] != command_name or state["commands"][-1]["state"] != "running":
                raise contract.ContractError("Phase 6 budget command cannot be reopened")
            updated = copy.deepcopy(state)
            updated["last_observed_ns"] = now
            updated["elapsed_seconds"] = (now - updated["started_ns"]) / 1.0e9
            return _phase6_persist_budget_state(path, updated)
        if state["state"] != "running" or command_name != order[len(names)]:
            raise contract.ContractError("Phase 6 budget command order mismatch")
        updated = copy.deepcopy(state)
        updated["commands"].append(
            {
                "name": command_name,
                "started_ns": now,
                "finished_ns": None,
                "elapsed_seconds": None,
                "state": "running",
            }
        )
        updated["elapsed_seconds"] = (now - updated["started_ns"]) / 1.0e9
        updated["last_observed_ns"] = now
        updated["update_index"] += 1
        return _phase6_persist_budget_state(path, updated)
    if command_name != order[0]:
        raise contract.ContractError("Phase 6 budget must start with the first command")
    state = {
        "schema": PHASE6_BUDGET_STATE_SCHEMA,
        "authority_id": authority_id,
        "gate": gate,
        "hard_ceiling_seconds": float(hard_ceiling_seconds),
        "boot_id": boot_id,
        "started_ns": now,
        "deadline_ns": now + int(float(hard_ceiling_seconds) * 1.0e9),
        "last_observed_ns": now,
        "elapsed_seconds": 0.0,
        "state": "running",
        "update_index": 0,
        "commands": [
            {
                "name": command_name,
                "started_ns": now,
                "finished_ns": None,
                "elapsed_seconds": None,
                "state": "running",
            }
        ],
    }
    return _phase6_persist_budget_state(path, state)


def phase6_budget_state_remaining(state: Mapping[str, Any], now_ns: int) -> float:
    if not _phase6_budget_state_valid(state) or type(now_ns) is not int:
        raise contract.ContractError("invalid Phase 6 budget remaining request")
    if now_ns < state["last_observed_ns"]:
        raise contract.ContractError("Phase 6 monotonic clock moved backwards")
    return (state["deadline_ns"] - now_ns) / 1.0e9


def phase6_budget_state_checkpoint(
    path: Path,
    lease: Phase6BudgetLease,
    *,
    authority_id: str,
    gate: str,
    hard_ceiling_seconds: float,
    command_name: str,
    now_ns: int | None = None,
) -> dict[str, Any]:
    lease.assert_current(command_name)
    now = time.monotonic_ns() if now_ns is None else now_ns
    current = contract.read_strict_json(path)
    if (
        not _phase6_budget_state_valid(current)
        or current["authority_id"] != authority_id
        or current["gate"] != gate
        or current["hard_ceiling_seconds"] != hard_ceiling_seconds
        or current["boot_id"] != _phase6_boot_id()
        or current["state"] != "running"
        or current["commands"][-1]["name"] != command_name
        or current["commands"][-1]["state"] != "running"
        or type(now) is not int
        or now < current["last_observed_ns"]
    ):
        raise contract.ContractError("Phase 6 budget checkpoint does not match active authority")
    updated = copy.deepcopy(current)
    observed = min(now, updated["deadline_ns"])
    updated["last_observed_ns"] = observed
    updated["elapsed_seconds"] = (observed - updated["started_ns"]) / 1.0e9
    return _phase6_persist_budget_state(path, updated)


def phase6_budget_state_close_command(
    path: Path,
    state: Mapping[str, Any],
    command_name: str,
    *,
    lease: Phase6BudgetLease,
    now_ns: int | None = None,
) -> dict[str, Any]:
    del state
    lease.assert_current(command_name)
    now = time.monotonic_ns() if now_ns is None else now_ns
    current = contract.read_strict_json(path)
    if not _phase6_budget_state_valid(current):
        raise contract.ContractError("Phase 6 budget state changed before close")
    command = current["commands"][-1]
    if (
        command["name"] != command_name
        or command["state"] != "running"
        or now < current["last_observed_ns"]
        or now > current["deadline_ns"]
    ):
        raise contract.ContractError("Phase 6 budget close does not match running command")
    updated = copy.deepcopy(current)
    closing = updated["commands"][-1]
    closing["finished_ns"] = now
    closing["elapsed_seconds"] = (now - closing["started_ns"]) / 1.0e9
    closing["state"] = "closed"
    updated["elapsed_seconds"] = (now - updated["started_ns"]) / 1.0e9
    updated["last_observed_ns"] = now
    updated["update_index"] += 1
    order = PHASE6_BUDGET_COMMAND_ORDER[updated["gate"]]
    updated["state"] = "closed" if len(updated["commands"]) == len(order) else "running"
    return _phase6_persist_budget_state(path, updated)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimensions", nargs="+", type=int, default=[2])
    parser.add_argument("--parameter-counts", nargs="+", type=int, default=[2])
    parser.add_argument("--timesteps", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=contract.METHOD_IDS,
        default=list(contract.PRIMARY_METHOD_IDS),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--plan-path", default=PLAN_PATH)
    parser.add_argument("--result-path", default=RESULT_PATH)
    parser.add_argument("--harness-contract-test-only", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--jit-compile", dest="jit_compile", action="store_true")
    parser.add_argument("--no-jit-compile", dest="jit_compile", action="store_false")
    parser.set_defaults(jit_compile=True)
    parser.add_argument("--tf32-enabled", dest="tf32_enabled", action="store_true")
    parser.add_argument("--no-tf32", dest="tf32_enabled", action="store_false")
    parser.set_defaults(tf32_enabled=True)
    parser.add_argument("--evaluate-phase5-smoke", action="store_true")
    parser.add_argument("--phase5-input", type=Path)
    parser.add_argument("--phase5-log", type=Path)
    parser.add_argument("--phase5-output", type=Path)
    phase6_modes = parser.add_mutually_exclusive_group()
    phase6_modes.add_argument("--phase6-pilot", action="store_true")
    phase6_modes.add_argument("--phase6-scalar-references", action="store_true")
    phase6_modes.add_argument("--phase6-remaining", action="store_true")
    phase6_modes.add_argument("--phase6-evaluate", action="store_true")
    phase6_modes.add_argument("--phase6-archive-r1", action="store_true")
    phase6_modes.add_argument("--phase6-archive-r2", action="store_true")
    phase6_modes.add_argument(
        "--phase6-prepare-proposal",
        choices=("gate_b", "gate_c"),
    )
    phase6_modes.add_argument(
        "--phase6-create-attestation",
        choices=("gate_b",),
    )
    phase6_modes.add_argument(
        "--phase6-validate-authority",
        choices=("gate_b",),
    )
    parser.add_argument("--batch-sizes", nargs="+", type=int)
    parser.add_argument("--trace-child-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--xla-child-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--xla-cell-timeout-seconds", type=float, default=160.0)
    parser.add_argument("--child-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--cell-timeout-seconds", type=float, default=160.0)
    parser.add_argument("--budget-contract", type=Path)
    parser.add_argument("--budget-attestation", type=Path)
    parser.add_argument("--review-path", type=Path)
    parser.add_argument("--trace-output-json", type=Path)
    parser.add_argument("--trace-input", type=Path)
    parser.add_argument("--pilot-input", type=Path)
    parser.add_argument("--scalar-reference-input", type=Path)
    parser.add_argument("--routing-output-json", type=Path)
    parser.add_argument("--routing-input", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--phase6-input", type=Path)
    return parser.parse_args()


def _git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )
    if completed.returncode != 0 or not completed.stdout.strip():
        raise contract.ContractError("cannot resolve git commit")
    return completed.stdout.strip()


def _phase6_live_target_pids() -> list[int]:
    target_modes = {
        b"--phase6-pilot",
        b"--phase6-scalar-references",
        b"--phase6-remaining",
        b"--phase6-trace-only",
    }
    supervisor = str(Path(__file__).resolve()).encode("utf-8")
    benchmark = str(BENCHMARK.resolve()).encode("utf-8")
    matches = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            argv = (entry / "cmdline").read_bytes().split(b"\x00")
        except OSError:
            continue
        if (supervisor in argv or benchmark in argv) and target_modes.intersection(argv):
            matches.append(int(entry.name))
    return sorted(matches)


def build_phase6_r1_archive() -> dict[str, Any]:
    trace_path = REPO_ROOT / contract.PHASE6_R1_GATE_B_ARTIFACTS["trace_output_json"]
    trace = contract.read_bounded_phase6_trace_json(trace_path)
    if not (
        trace.get("state") == "running"
        and trace.get("update_index") == 2
        and trace.get("bindings", {}).get("authority_id") == contract.PHASE6_R1_AUTHORITY_ID
    ):
        raise contract.ContractError("r1 trace ledger identity changed")
    record = trace["records"][0]
    child = record.get("evidence", {}).get("child_artifact", {}).get("strict_json")
    schedule_row = trace["bindings"]["schedule"]["payload"]["records"][0]
    schedule_argv = schedule_row.get("child_command_argv")
    child_argv = child.get("command_argv") if isinstance(child, Mapping) else None
    if not (
        isinstance(schedule_argv, list)
        and isinstance(child_argv, list)
        and len(schedule_argv) == len(child_argv)
    ):
        raise contract.ContractError("r1 argv evidence is malformed")
    differences = [
        {"index": index, "schedule": expected, "child": observed}
        for index, (expected, observed) in enumerate(
            zip(schedule_argv, child_argv, strict=True)
        )
        if expected != observed
    ]
    process = record.get("process", {})
    matching_pids = _phase6_live_target_pids()
    archive = {
        "schema": contract.PHASE6_R1_ARCHIVE_SCHEMA,
        "authority_id": contract.PHASE6_R1_AUTHORITY_ID,
        "disposition": "invalid_harness_authority_exhausted_never_resume_or_import",
        "files": contract.phase6_r1_archive_file_records(),
        "diagnosis": {
            "trace_ledger_state": trace["state"],
            "trace_ledger_update_index": trace["update_index"],
            "first_record_state": record.get("state"),
            "first_record_reason": record.get("reason"),
            "child_state": child.get("state"),
            "child_stage": child.get("stage"),
            "child_returncode": (
                0 if child.get("state") == "passed" and child.get("error") is None else None
            ),
            "argv_differences": differences,
            "other_argv_elements_equal": schedule_argv[1:] == child_argv[1:],
            "reviewed_gate_b_subplan_sha256_history": [
                "4403929e2f58d9027b88c21f8840e265a14666a3a7311eb7a0a833723e137bb3",
                "bd449a78fb19c06e90da00892e814eecfa62623c6bcf6f08f2befca29813c332",
                "165e4870155a999661a7502c79347e4b618dee17006749d702b787b7c2b75565",
                "d1c46aacdc6e15c234a3c3d739837d0fcb6fd8c57dbdc6c78f8c532ef0cc1214",
            ],
            "full_child_validity_recomputed": False,
            "classification": "common_invalidity_not_method_evidence",
        },
        "no_live_process": {
            "scan": "proc_cmdline_exact_supervisor_target_modes",
            "matching_pids": matching_pids,
            "passed": not matching_pids,
        },
        "pre_edit_lane_hashes": dict(contract.PHASE6_R1_PRE_EDIT_LANE_HASHES),
        "protected_hashes": dict(contract.PHASE6_PROTECTED_HASHES),
        "nonclaims": list(contract.PHASE6_R1_ARCHIVE_NONCLAIMS),
    }
    if process.get("returncode") is not None:
        raise contract.ContractError("r1 recovered process unexpectedly has a return code")
    contract.validate_phase6_r1_archive(archive)
    return archive


def run_phase6_archive_r1(args: argparse.Namespace) -> int:
    expected_relative = contract.PHASE6_R1_ARCHIVE_RELATIVE
    expected_argv = ["--phase6-archive-r1", "--output-json", expected_relative]
    if sys.argv[1:] != expected_argv or args.output_json != Path(expected_relative):
        raise contract.ContractError("Phase 6 r1 archive requires its exact closed invocation")
    output_path = _phase6_repo_path(args.output_json)
    if output_path.is_symlink() or output_path.exists() and not output_path.is_file():
        raise contract.ContractError("Phase 6 r1 archive output path is unsafe")
    archive = build_phase6_r1_archive()
    if output_path.exists():
        existing = contract.read_strict_json(output_path)
        contract.validate_phase6_r1_archive(existing)
        if existing != archive:
            raise contract.ContractError("refusing to overwrite a nonidentical r1 archive")
        return 0
    contract.durable_atomic_write_json(output_path, archive)
    reparsed = contract.read_strict_json(output_path)
    contract.validate_phase6_r1_archive(reparsed)
    if reparsed != archive:
        raise contract.ContractError("Phase 6 r1 archive failed durable reparse")
    return 0


def build_phase6_r2_archive() -> dict[str, Any]:
    matching_pids = _phase6_live_target_pids()
    root = Path(contract.PHASE6_R2_WORK_ROOT)
    budget_dir = root / "budget_state"
    archive = {
        "schema": contract.PHASE6_R2_ARCHIVE_SCHEMA,
        "authority_id": contract.PHASE6_R2_AUTHORITY_ID,
        "disposition": "invalid_harness_authority_exhausted_never_resume_or_import",
        "files": contract.phase6_r2_archive_file_records(),
        "absent_paths": contract.phase6_r2_archive_absent_paths(),
        "work_root_entries": sorted(path.name for path in root.iterdir()),
        "budget_state_entries": sorted(path.name for path in budget_dir.iterdir()),
        "diagnosis": {
            "classification": "common_invalidity_not_method_evidence",
            "failure_stage": "pre_trace_ledger_binding_validation",
            "target_fixture_constructed": False,
            "target_trace_requested": False,
            "target_xla_requested": False,
            "trace_output_present": False,
            "pilot_output_present": False,
            "budget_state": "running",
            "budget_update_index": 0,
            "lease_state": "released",
            "mixed_format_inputs": ["json", "markdown", "markdown", "markdown"],
        },
        "no_live_process": {
            "scan": "proc_cmdline_exact_supervisor_target_modes",
            "matching_pids": matching_pids,
            "passed": not matching_pids,
        },
        "protected_hashes": dict(contract.PHASE6_PROTECTED_HASHES),
        "nonclaims": list(contract.PHASE6_R2_ARCHIVE_NONCLAIMS),
    }
    contract.validate_phase6_r2_archive(archive)
    return archive


def run_phase6_archive_r2(args: argparse.Namespace) -> int:
    expected_relative = contract.PHASE6_R2_ARCHIVE_RELATIVE
    expected_argv = ["--phase6-archive-r2", "--output-json", expected_relative]
    if sys.argv[1:] != expected_argv or args.output_json != Path(expected_relative):
        raise contract.ContractError("Phase 6 r2 archive requires its exact closed invocation")
    output_path = _phase6_repo_path(args.output_json)
    if output_path.is_symlink() or output_path.exists() and not output_path.is_file():
        raise contract.ContractError("Phase 6 r2 archive output path is unsafe")
    archive = build_phase6_r2_archive()
    if output_path.exists():
        existing = contract.read_strict_json(output_path)
        contract.validate_phase6_r2_archive(existing)
        if existing != archive:
            raise contract.ContractError("refusing to overwrite a nonidentical r2 archive")
        return 0
    contract.durable_atomic_write_json(output_path, archive)
    reparsed = contract.read_strict_json(output_path)
    contract.validate_phase6_r2_archive(reparsed)
    if reparsed != archive:
        raise contract.ContractError("Phase 6 r2 archive failed durable reparse")
    return 0


def _phase6_repo_path(path: Path) -> Path:
    candidate = path if path.is_absolute() else REPO_ROOT / path
    resolved_parent = candidate.parent.resolve(strict=True)
    resolved = resolved_parent / candidate.name
    try:
        resolved.relative_to(REPO_ROOT.resolve())
    except ValueError as exc:
        raise contract.ContractError(f"Phase 6 artifact path escapes repository: {path}") from exc
    return resolved


def _phase6_environment() -> dict[str, str]:
    return {
        "CUDA_VISIBLE_DEVICES": "-1",
        "OMP_NUM_THREADS": "1",
        "TF_NUM_INTRAOP_THREADS": "1",
        "TF_NUM_INTEROP_THREADS": "1",
    }


def _phase6_process_group_exists(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _phase6_signal_group(pgid: int, signum: int) -> bool:
    try:
        os.killpg(pgid, signum)
        return True
    except ProcessLookupError:
        return False


def _phase6_process_identity(pid: int) -> tuple[int, int]:
    try:
        stat_text = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
        command_end = stat_text.rfind(")")
        if command_end < 0 or command_end + 2 >= len(stat_text):
            raise ValueError("malformed /proc stat record")
        # Field 2 is parenthesized and may contain spaces or closing parentheses.
        fields_after_command = stat_text[command_end + 2 :].split()
        return os.getpgid(pid), int(fields_after_command[19])
    except (OSError, ValueError, IndexError) as exc:
        raise contract.ContractError(f"cannot establish process identity for PID {pid}") from exc


def _phase6_recover_running_process(process: Mapping[str, Any]) -> dict[str, Any]:
    pid = process["pid"]
    pgid = process["pgid"]
    term_sent = False
    kill_sent = False
    pid_exists = Path(f"/proc/{pid}").exists()
    group_exists = _phase6_process_group_exists(pgid)
    if pid_exists:
        actual_pgid, actual_start_ticks = _phase6_process_identity(pid)
        if (
            actual_pgid != pgid
            or actual_start_ticks != process["process_start_ticks"]
        ):
            raise contract.ContractError("stale running process identity is ambiguous")
        term_sent = _phase6_signal_group(pgid, signal.SIGTERM)
        deadline = time.monotonic() + 5.0
        while _phase6_process_group_exists(pgid) and time.monotonic() < deadline:
            time.sleep(0.01)
        if _phase6_process_group_exists(pgid):
            kill_sent = _phase6_signal_group(pgid, signal.SIGKILL)
            deadline = time.monotonic() + 5.0
            while _phase6_process_group_exists(pgid) and time.monotonic() < deadline:
                time.sleep(0.01)
    elif group_exists:
        raise contract.ContractError("stale running process group exists without its recorded PID")
    if _phase6_process_group_exists(pgid):
        raise contract.ContractError("stale running process group could not be terminated")
    finished_ns = time.monotonic_ns()
    empty = b""
    return {
        **dict(process),
        "finished_ns": finished_ns,
        "elapsed_seconds": max(0.0, (finished_ns - process["started_ns"]) / 1.0e9),
        "term_sent": term_sent,
        "kill_sent": kill_sent,
        "reaped": False,
        "reap_status": "already_gone_not_waitable_after_recovery",
        "process_group_gone": True,
        "returncode": None,
        "timed_out": False,
        "stdout_bytes": 0,
        "stdout_total_bytes": None,
        "stdout_capture_status": "unavailable_after_recovery",
        "stdout_sha256": hashlib.sha256(empty).hexdigest(),
        "stdout_base64": "",
        "stdout_tail": "",
        "stderr_bytes": 0,
        "stderr_total_bytes": None,
        "stderr_capture_status": "unavailable_after_recovery",
        "stderr_sha256": hashlib.sha256(empty).hexdigest(),
        "stderr_base64": "",
        "stderr_tail": "",
    }


def _phase6_cleanup_process_group(
    process: subprocess.Popen[bytes],
    *,
    pgid: int,
    term_grace_seconds: float,
    kill_reap_grace_seconds: float,
) -> tuple[bool, bool]:
    term_sent = False
    kill_sent = False
    if process.poll() is None or _phase6_process_group_exists(pgid):
        term_sent = _phase6_signal_group(pgid, signal.SIGTERM)
        try:
            process.communicate(timeout=term_grace_seconds)
        except subprocess.TimeoutExpired:
            kill_sent = _phase6_signal_group(pgid, signal.SIGKILL)
            try:
                process.communicate(timeout=kill_reap_grace_seconds)
            except subprocess.TimeoutExpired as exc:
                raise contract.ContractError(
                    f"failed to reap managed child {process.pid} after SIGKILL"
                ) from exc
    else:
        process.communicate(timeout=kill_reap_grace_seconds)
    if process.poll() is None:
        try:
            process.wait(timeout=kill_reap_grace_seconds)
        except subprocess.TimeoutExpired as exc:
            raise contract.ContractError("managed process could not be reaped") from exc
    if _phase6_process_group_exists(pgid):
        kill_sent = _phase6_signal_group(pgid, signal.SIGKILL) or kill_sent
        deadline = time.monotonic() + kill_reap_grace_seconds
        while _phase6_process_group_exists(pgid) and time.monotonic() < deadline:
            time.sleep(0.01)
    if _phase6_process_group_exists(pgid):
        raise contract.ContractError("managed process group still exists after cleanup")
    return term_sent, kill_sent


def _phase6_read_stream_capture(handle: Any) -> tuple[bytes, int, str]:
    handle.flush()
    total_bytes = os.fstat(handle.fileno()).st_size
    if total_bytes > contract.PHASE6_STREAM_MAX_BYTES:
        handle.seek(total_bytes - contract.PHASE6_STREAM_MAX_BYTES)
        status = "truncated_at_cap"
    else:
        handle.seek(0)
        status = "complete"
    raw = handle.read(contract.PHASE6_STREAM_MAX_BYTES)
    if not isinstance(raw, bytes):
        raise contract.ContractError("managed process stream was not captured as bytes")
    return raw, total_bytes, status


def run_managed_process_group(
    command: Sequence[str],
    *,
    environment: Mapping[str, str],
    deadline_seconds: float,
    term_grace_seconds: float = 5.0,
    kill_reap_grace_seconds: float = 5.0,
    cwd: Path = REPO_ROOT,
    on_started: Callable[[Mapping[str, Any]], None] | None = None,
    on_completed: Callable[[Mapping[str, Any]], None] | None = None,
    termination: Phase6TerminationController | None = None,
    absolute_deadline_ns: int | None = None,
) -> dict[str, Any]:
    if (
        not command
        or deadline_seconds <= 0
        or term_grace_seconds < 0
        or kill_reap_grace_seconds < 0
    ):
        raise contract.ContractError("invalid managed process-group request")
    env = os.environ.copy()
    env.update({str(key): str(value) for key, value in environment.items()})
    started_ns = time.monotonic_ns()
    deadline_ns = (
        started_ns + int(deadline_seconds * 1.0e9)
        if absolute_deadline_ns is None
        else absolute_deadline_ns
    )
    if type(deadline_ns) is not int or deadline_ns < started_ns:
        raise contract.ContractError("managed process deadline is already exhausted")
    process: subprocess.Popen[bytes] | None = None
    pgid: int | None = None
    with tempfile.TemporaryFile(mode="w+b") as stdout_file, tempfile.TemporaryFile(
        mode="w+b"
    ) as stderr_file:
        timed_out = False
        term_sent = False
        kill_sent = False
        try:
            defer = termination.defer() if termination is not None else contextlib.nullcontext()
            with defer:
                process = subprocess.Popen(
                    list(command),
                    cwd=cwd,
                    env=env,
                    stdout=stdout_file,
                    stderr=stderr_file,
                    start_new_session=True,
                )
                # start_new_session=True makes the child the leader of its new group.
                pgid = process.pid
                actual_pgid, process_start_ticks = _phase6_process_identity(process.pid)
                if actual_pgid != pgid:
                    raise contract.ContractError("managed child did not enter its owned process group")
                running = {
                    "command_argv": list(command),
                    "cwd": str(cwd.resolve()),
                    "environment": dict(environment),
                    "pid": process.pid,
                    "pgid": pgid,
                    "process_start_ticks": process_start_ticks,
                    "started_ns": started_ns,
                    "deadline_seconds": (deadline_ns - started_ns) / 1.0e9,
                }
                if on_started is not None:
                    on_started(running)
            remaining_seconds = max(0.0, (deadline_ns - time.monotonic_ns()) / 1.0e9)
            process.communicate(timeout=remaining_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            assert process is not None and pgid is not None
            term_sent, kill_sent = _phase6_cleanup_process_group(
                process,
                pgid=pgid,
                term_grace_seconds=term_grace_seconds,
                kill_reap_grace_seconds=kill_reap_grace_seconds,
            )
        except BaseException:
            if process is not None:
                _phase6_cleanup_process_group(
                    process,
                    pgid=process.pid if pgid is None else pgid,
                    term_grace_seconds=term_grace_seconds,
                    kill_reap_grace_seconds=kill_reap_grace_seconds,
                )
            raise
        assert process is not None and pgid is not None
        finished_ns = time.monotonic_ns()
        if process.poll() is None or _phase6_process_group_exists(pgid):
            cleanup_term, cleanup_kill = _phase6_cleanup_process_group(
                process,
                pgid=pgid,
                term_grace_seconds=term_grace_seconds,
                kill_reap_grace_seconds=kill_reap_grace_seconds,
            )
            term_sent = term_sent or cleanup_term
            kill_sent = kill_sent or cleanup_kill
        stdout, stdout_total_bytes, stdout_capture_status = _phase6_read_stream_capture(
            stdout_file
        )
        stderr, stderr_total_bytes, stderr_capture_status = _phase6_read_stream_capture(
            stderr_file
        )
        result = {
        **running,
        "finished_ns": finished_ns,
        "elapsed_seconds": (finished_ns - started_ns) / 1.0e9,
        "term_sent": term_sent,
        "kill_sent": kill_sent,
        "reaped": True,
        "reap_status": "reaped_direct_child",
        "process_group_gone": True,
        "returncode": process.returncode,
        "timed_out": timed_out,
        "stdout_bytes": len(stdout),
        "stdout_total_bytes": stdout_total_bytes,
        "stdout_capture_status": stdout_capture_status,
        "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
        "stdout_base64": base64.b64encode(stdout).decode("ascii"),
        "stdout_tail": stdout.decode("utf-8", errors="replace")[-4000:],
        "stderr_bytes": len(stderr),
        "stderr_total_bytes": stderr_total_bytes,
        "stderr_capture_status": stderr_capture_status,
        "stderr_sha256": hashlib.sha256(stderr).hexdigest(),
        "stderr_base64": base64.b64encode(stderr).decode("ascii"),
        "stderr_tail": stderr.decode("utf-8", errors="replace")[-4000:],
        }
        if on_completed is not None:
            defer = termination.defer() if termination is not None else contextlib.nullcontext()
            try:
                with defer:
                    on_completed(result)
            except BaseException:
                if process.poll() is None or _phase6_process_group_exists(pgid):
                    _phase6_cleanup_process_group(
                        process,
                        pgid=pgid,
                        term_grace_seconds=term_grace_seconds,
                        kill_reap_grace_seconds=kill_reap_grace_seconds,
                    )
                raise
    return result


def phase6_persist_and_validate(
    path: Path,
    payload: Mapping[str, Any],
    *,
    final: bool,
) -> dict[str, Any]:
    resolved = _phase6_repo_path(path)
    checks = contract.phase6_ledger_checks(payload, final=final)
    if not all(checks.values()):
        raise contract.ContractError(f"Phase 6 ledger failed prewrite validation: {checks}")
    contract.durable_atomic_write_json(resolved, payload)
    reparsed = contract.read_strict_json(resolved)
    checks = contract.phase6_ledger_checks(reparsed, final=final)
    if not all(checks.values()):
        raise contract.ContractError(f"persisted Phase 6 ledger failed reparse: {checks}")
    return reparsed


def _phase6_schedule_config(
    identity: Mapping[str, Any],
    *,
    child_timeout_seconds: float,
) -> dict[str, Any]:
    return {
        "method_id": identity["method_id"],
        "method_contract_version": contract.METHOD_CONTRACT_VERSION,
        "dimension": identity["dimension"],
        "parameter_count": identity["parameter_count"],
        "timesteps": 120,
        "batch_size": identity["batch_size"],
        "dtype": identity["dtype"],
        "device": "cpu",
        "jit_compile": identity["operation"] == "xla",
        "cpu_threads": 1,
        "repeats": 2 if identity["operation"] == "xla" else 1,
        "subprocess_timeout_seconds": child_timeout_seconds,
        "xla_flags": os.environ.get("XLA_FLAGS", "UNSET"),
        "tf32_enabled": True,
        "jitter": 1.0e-9,
        "jitter_updates_filtered_covariance": True,
        "fixture_contract_version": contract.FIXTURE_CONTRACT_VERSION,
        "timing_boundary_version": contract.TIMING_BOUNDARY_VERSION,
        "method_options": {},
    }


def _phase6_fixture_config(identity: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "fixture_contract_version": contract.FIXTURE_CONTRACT_VERSION,
        "randomness": "deterministic",
        "seed": None,
        "dimension": identity["dimension"],
        "parameter_count": identity["parameter_count"],
        "timesteps": 120,
        "batch_size": identity["batch_size"],
        "dtype": identity["dtype"],
        "parameter_batch_version": contract.PARAMETER_BATCH_VERSION,
        "observation_generation_version": contract.OBSERVATION_GENERATION_VERSION,
        "external_input_hashes": {},
    }


def _phase6_child_paths(identity: Mapping[str, Any]) -> dict[str, Path]:
    digest = contract.canonical_sha256(identity)[:24]
    root = PHASE6_WORK_DIR / identity["operation"] / digest
    return {
        "artifact": root.with_suffix(".json"),
        "markdown": root.with_suffix(".md"),
        "sidecar": root.with_suffix(".payload.json"),
        "journal": root.with_suffix(".jsonl"),
        "dependency_before": root.with_suffix(".dependency-before.json"),
        "dependency_after": root.with_suffix(".dependency-after.json"),
        "authority_snapshot": root.with_suffix(".authority.json"),
    }


def _phase6_child_command(
    identity: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    fingerprints: Mapping[str, str],
    resume_key_value: str,
    attempt_id: str,
) -> list[str]:
    paths = _phase6_child_paths(identity)
    if identity["operation"] == "trace":
        return [
            str(PYTHON),
            str(BENCHMARK),
            "--phase6-trace-only",
            "--dimensions", str(identity["dimension"]),
            "--parameter-counts", str(identity["parameter_count"]),
            "--timesteps", "120",
            "--batch-size", str(identity["batch_size"]),
            "--dtype", identity["dtype"],
            "--device", "cpu",
            "--cpu-threads", "1",
            "--method", identity["method_id"],
            "--case-id", contract.case_id(config),
            "--attempt-id", attempt_id,
            "--progress-journal", str(paths["journal"]),
            "--source-fingerprint", fingerprints["source_fingerprint"],
            "--config-fingerprint", fingerprints["config_fingerprint"],
            "--runtime-fingerprint", fingerprints["runtime_fingerprint"],
            "--fixture-fingerprint", fingerprints["fixture_fingerprint"],
            "--schedule-fingerprint", fingerprints["schedule_fingerprint"],
            "--resume-key", resume_key_value,
            "--phase6-authority-snapshot", str(paths["authority_snapshot"]),
            "--output-json", str(paths["artifact"]),
            "--output-md", str(paths["markdown"]),
            "--no-jit-compile",
        ]
    return [
        str(PYTHON),
        str(BENCHMARK),
        "--dimensions", str(identity["dimension"]),
        "--parameter-counts", str(identity["parameter_count"]),
        "--timesteps", "120",
        "--batch-size", str(identity["batch_size"]),
        "--dtype", identity["dtype"],
        "--device", "cpu",
        "--cpu-threads", "1",
        "--repeats", str(config["repeats"]),
        "--method", identity["method_id"],
        "--case-id", contract.case_id(config),
        "--attempt-id", attempt_id,
        "--progress-journal", str(paths["journal"]),
        "--source-fingerprint", fingerprints["source_fingerprint"],
        "--config-fingerprint", fingerprints["config_fingerprint"],
        "--runtime-fingerprint", fingerprints["runtime_fingerprint"],
        "--fixture-fingerprint", fingerprints["fixture_fingerprint"],
        "--schedule-fingerprint", fingerprints["schedule_fingerprint"],
        "--resume-key", resume_key_value,
        "--phase6-authority-snapshot", str(paths["authority_snapshot"]),
        "--plan-path", PHASE6_PLAN_PATH,
        "--output-json", str(paths["artifact"]),
        "--output-md", str(paths["markdown"]),
        "--phase6-dependency-before", str(paths["dependency_before"]),
        "--phase6-dependency-after", str(paths["dependency_after"]),
        "--jit-compile" if identity["operation"] == "xla" else "--no-jit-compile",
        "--tf32-enabled",
    ]


def phase6_build_schedule(
    schema: str,
    *,
    gate: str,
    child_timeout_seconds: float,
) -> dict[str, Any]:
    identities = contract.phase6_expected_roster(schema)
    source = contract.source_manifest(REPO_ROOT, include_supervisor=True)
    runtime = contract.runtime_manifest()
    provisional: list[dict[str, Any]] = []
    for index, identity in enumerate(identities):
        config_manifest = contract.config_manifest(
            _phase6_schedule_config(identity, child_timeout_seconds=child_timeout_seconds)
        )
        fixture_manifest = contract.fixture_manifest(_phase6_fixture_config(identity))
        provisional.append(
            {
                "identity": identity,
                "case_id": contract.case_id(config_manifest["config"]),
                "config": config_manifest["config"],
                "fingerprints": {
                    "source_fingerprint": source["source_fingerprint"],
                    "config_fingerprint": config_manifest["config_fingerprint"],
                    "runtime_fingerprint": runtime["runtime_fingerprint"],
                    "fixture_fingerprint": fixture_manifest["fixture_fingerprint"],
                    "schedule_fingerprint": "pending",
                },
                "resume_key": "pending",
                "child_command_argv": [],
                "attempt_id": f"phase6-{index:02d}-{contract.canonical_sha256(identity)[:16]}",
            }
        )
    schedule_identity = {
        "schema": contract.PHASE6_SCHEDULE_SCHEMA,
        "ledger_schema": schema,
        "gate": gate,
        "records": [
            {
                "identity": row["identity"],
                "case_id": row["case_id"],
                "config": row["config"],
                "source_fingerprint": row["fingerprints"]["source_fingerprint"],
                "runtime_fingerprint": row["fingerprints"]["runtime_fingerprint"],
                "fixture_fingerprint": row["fingerprints"]["fixture_fingerprint"],
            }
            for row in provisional
        ],
    }
    schedule_fingerprint = contract.canonical_sha256(schedule_identity)
    records = []
    for row in provisional:
        row["fingerprints"]["schedule_fingerprint"] = schedule_fingerprint
        key = contract.resume_key(
            case_identity=row["case_id"],
            method_id=row["identity"]["method_id"],
            fingerprints=row["fingerprints"],
        )
        row["resume_key"] = key
        row["child_command_argv"] = _phase6_child_command(
            row["identity"],
            config=row["config"],
            fingerprints=row["fingerprints"],
            resume_key_value=key,
            attempt_id=row["attempt_id"],
        )
        records.append(dict(row))
    payload = {
        "schema": contract.PHASE6_SCHEDULE_SCHEMA,
        "ledger_schema": schema,
        "gate": gate,
        "records": records,
    }
    payload["schedule_sha256"] = contract.canonical_sha256(payload)
    return payload


def phase6_build_bindings(
    *,
    proposal_path: Path,
    attestation_path: Path,
    schedule: Mapping[str, Any],
    phase45_paths: Sequence[Path],
    authority_input_paths: Sequence[Path] = (),
    runtime_predecessor_paths: Sequence[Path] = (),
) -> dict[str, Any]:
    proposal_blob = contract.phase6_blob_record(proposal_path)
    attestation_blob = contract.phase6_blob_record(attestation_path)
    proposal = proposal_blob["strict_json"]
    if not isinstance(proposal, Mapping):
        raise contract.ContractError("Phase 6 proposal is not strict JSON")
    return {
        "authority_id": proposal["authority_id"],
        "proposal": proposal_blob,
        "attestation": attestation_blob,
        "schedule": {"payload": dict(schedule), "sha256": contract.canonical_sha256(schedule)},
        "phase45_evidence": [contract.phase6_blob_record(path) for path in phase45_paths],
        "authority_inputs": [
            contract.phase6_blob_record(path) for path in authority_input_paths
        ],
        "runtime_predecessors": [
            contract.phase6_runtime_predecessor_record(path)
            for path in runtime_predecessor_paths
        ],
    }


def phase6_revalidate_launch_authority(bindings: Mapping[str, Any]) -> None:
    if not contract._phase6_bindings_valid(bindings):
        raise contract.ContractError("Phase 6 bindings no longer satisfy authority")
    proposal_blob = contract.phase6_blob_record(Path(bindings["proposal"]["path"]))
    attestation_blob = contract.phase6_blob_record(Path(bindings["attestation"]["path"]))
    if proposal_blob != bindings["proposal"] or attestation_blob != bindings["attestation"]:
        raise contract.ContractError("Phase 6 proposal or attestation changed before launch")
    proposal, attestation = contract.validate_phase6_runtime_authority(
        Path(proposal_blob["path"]),
        Path(attestation_blob["path"]),
        expected_gate=bindings["schedule"]["payload"]["gate"],
    )
    if proposal != proposal_blob["strict_json"] or attestation != attestation_blob["strict_json"]:
        raise contract.ContractError("Phase 6 runtime authority bytes changed before launch")
    for field in ("phase45_evidence", "authority_inputs"):
        for blob in bindings[field]:
            if contract.phase6_blob_record(Path(blob["path"])) != blob:
                raise contract.ContractError(f"Phase 6 {field} changed before launch")
    for predecessor in bindings["runtime_predecessors"]:
        artifact = predecessor["artifact"]
        if contract.phase6_blob_record(Path(artifact["path"])) != artifact:
            raise contract.ContractError("Phase 6 runtime predecessor changed before launch")
    schedule = bindings["schedule"]["payload"]
    current_source = contract.source_manifest(REPO_ROOT, include_supervisor=True)[
        "source_fingerprint"
    ]
    current_runtime = contract.runtime_manifest()["runtime_fingerprint"]
    if (
        not all(contract.phase6_schedule_checks(schedule).values())
        or any(
            row["fingerprints"]["source_fingerprint"] != current_source
            or row["fingerprints"]["runtime_fingerprint"] != current_runtime
            for row in schedule["records"]
        )
    ):
        raise contract.ContractError("Phase 6 source/runtime/schedule drift before launch")


def _phase6_prepare_child_artifacts(identity: Mapping[str, Any]) -> None:
    for path in _phase6_child_paths(identity).values():
        if path.is_symlink():
            raise contract.ContractError(f"stale Phase 6 child path is a symlink: {path}")
        if path.exists():
            if not path.is_file():
                raise contract.ContractError(
                    f"stale Phase 6 child path is not a regular file: {path}"
                )
            raise contract.ContractError(
                f"stale Phase 6 child evidence requires reviewed recovery: {path}"
            )
        path.parent.mkdir(parents=True, exist_ok=True)


def _phase6_empty_manifest() -> dict[str, Any]:
    entries: list[dict[str, str]] = []
    return {
        "schema": contract.PHASE6_DEPENDENCY_SCHEMA,
        "repository_root": str(REPO_ROOT.resolve()),
        "entries": entries,
        "manifest_sha256": contract.canonical_sha256(entries),
    }


def _phase6_record_evidence(
    identity: Mapping[str, Any],
    *,
    classification: str,
    discovery: Mapping[str, Any],
) -> dict[str, Any]:
    paths = _phase6_child_paths(identity)
    child_blob = contract.phase6_blob_record(paths["artifact"])
    sidecar_blob = contract.phase6_blob_record(paths["sidecar"])
    journal_blob = contract.phase6_blob_record(paths["journal"])
    child_payload = child_blob.get("strict_json")
    if identity["operation"] == "trace" and isinstance(child_payload, Mapping):
        before = child_payload.get("dependency_manifest_before_builder")
        after = child_payload.get("dependency_manifest_after_terminal")
    else:
        try:
            before = contract.read_strict_json(paths["dependency_before"])
        except contract.ContractError:
            before = None
        try:
            after = contract.read_strict_json(paths["dependency_after"])
        except contract.ContractError:
            after = None
    return {
        "classification": classification,
        "child_artifact": child_blob,
        "payload_sidecar": sidecar_blob,
        "progress_journal": journal_blob,
        "dependency_manifest_before_builder": before,
        "dependency_manifest_after_terminal": after,
        "dependency_coverage_before": contract.dependency_manifest_covers(
            discovery, before, required_paths=PHASE6_REQUIRED_DISCOVERY_PATHS
        ),
        "dependency_coverage_after": contract.dependency_manifest_covers(
            discovery, after, required_paths=PHASE6_REQUIRED_DISCOVERY_PATHS
        ),
    }


def _phase6_terminal_classification(
    identity: Mapping[str, Any], process: Mapping[str, Any]
) -> tuple[str, str, str]:
    if process["timed_out"]:
        return (
            "timed_out",
            "child_execution_deadline_exceeded",
            "trace_timeout"
            if identity["operation"] == "trace"
            else "scalar_reference_timeout"
            if identity["operation"] == "scalar_reference"
            else "cpu_backend_or_cell_timeout",
        )
    returncode = process["returncode"]
    if returncode == 0:
        return (
            "passed",
            "child_passed",
            "trace_pass"
            if identity["operation"] == "trace"
            else "scalar_reference_pass"
            if identity["operation"] == "scalar_reference"
            else "method_pass",
        )
    if returncode is not None and returncode < 0:
        return (
            "crashed",
            "child_signal_exit",
            "trace_crash"
            if identity["operation"] == "trace"
            else "scalar_reference_crash"
            if identity["operation"] == "scalar_reference"
            else "cpu_backend_or_method_failure",
        )
    return (
        "failed",
        "child_nonzero_exit",
        "trace_structural_failure"
        if identity["operation"] == "trace"
        else "method_local_failure",
    )


def _phase6_should_prune(
    payload: Mapping[str, Any], identity: Mapping[str, Any]
) -> str | None:
    if payload.get("schema") != contract.PHASE6_FINAL_SCHEMA:
        return (
            "common_invalidity"
            if any(
                record.get("evidence", {}).get("classification") == "common_invalidity"
                for record in payload.get("records", [])
                if record.get("state") in contract.PHASE6_TERMINAL_STATES
            )
            else None
        )
    by_key = {
        (
            record["identity"]["dimension"],
            record["identity"]["parameter_count"],
            record["identity"]["batch_size"],
            record["identity"]["method_id"],
        ): record
        for record in payload["records"]
    }
    d = identity["dimension"]
    p = identity["parameter_count"]
    b = identity["batch_size"]
    method = identity["method_id"]
    if any(
        record.get("evidence", {}).get("classification") == "common_invalidity"
        for record in payload["records"]
        if record.get("state") in contract.PHASE6_TERMINAL_STATES
    ):
        return "common_invalidity"
    smaller_b = {4: 1, 16: 4}.get(b)
    if smaller_b is not None:
        predecessor = by_key[(d, p, smaller_b, method)]
        if predecessor["state"] != "passed":
            return (
                "after_smaller_p150_batch_failure"
                if p == 150
                else "after_smaller_batch_failure"
            )
    if p == 150:
        p50 = by_key[(d, 50, b, method)]
        if str(p50["state"]).startswith("not_launched:"):
            return "p50_dependency_not_launched"
        if p50["state"] != "passed":
            return "p50_dependency_failed"
    return None


def _phase6_final_identity_for_route(identity: Mapping[str, Any]) -> dict[str, Any]:
    return contract.phase6_identity(
        dimension=identity["dimension"],
        parameter_count=150,
        batch_size=identity["batch_size"],
        dtype=identity["dtype"],
        method_id=identity["method_id"],
        operation="xla",
    )


def _phase6_prelaunch_snapshot(
    final_payload: Mapping[str, Any], identity: Mapping[str, Any]
) -> dict[str, Any]:
    """Bind a route to the exact append-only ledger prefix before its transition."""
    records = final_payload.get("records")
    roster = final_payload.get("roster")
    events = final_payload.get("events")
    update_index = final_payload.get("update_index")
    bindings = final_payload.get("bindings")
    final_identity = _phase6_final_identity_for_route(identity)
    final_identity_id = final_identity["identity_id"]
    if (
        final_payload.get("schema") != contract.PHASE6_FINAL_SCHEMA
        or not isinstance(records, list)
        or not isinstance(roster, list)
        or len(records) != len(roster)
        or not isinstance(events, list)
        or type(update_index) is not int
        or update_index != len(events)
        or not isinstance(bindings, Mapping)
        or not isinstance(bindings.get("authority_id"), str)
    ):
        raise contract.ContractError("routing snapshot requires a closed final-ledger shape")
    roster_ids = [
        row.get("identity_id") if isinstance(row, Mapping) else None for row in roster
    ]
    if roster_ids.count(final_identity_id) != 1:
        raise contract.ContractError("routing snapshot target is absent or duplicated")
    target_index = roster_ids.index(final_identity_id)
    if records[target_index].get("identity") != final_identity:
        raise contract.ContractError("routing snapshot target record identity differs")
    prior_records = records[:target_index]
    if any(
        not isinstance(record, Mapping)
        or record.get("identity") != roster[index]
        or record.get("state") in {"pending", "running"}
        for index, record in enumerate(prior_records)
    ):
        raise contract.ContractError("routing snapshot does not follow a closed record prefix")
    target_event_indexes = [
        index
        for index, event in enumerate(events)
        if isinstance(event, Mapping)
        and event.get("identity_id") == final_identity_id
    ]
    target_state = records[target_index].get("state")
    if target_state == "pending":
        if target_event_indexes:
            raise contract.ContractError("pending routing target already has ledger events")
        prefix_length = len(events)
    else:
        if not target_event_indexes:
            raise contract.ContractError("closed routing target lacks its first ledger event")
        prefix_length = target_event_indexes[0]
    prefix_events = events[:prefix_length]
    prior_ids = set(roster_ids[:target_index])
    observed_prior_ids: set[str] = set()
    for expected_index, event in enumerate(prefix_events, 1):
        if (
            not isinstance(event, Mapping)
            or event.get("update_index") != expected_index
            or event.get("identity_id") not in prior_ids
        ):
            raise contract.ContractError("routing snapshot event prefix is not canonical")
        observed_prior_ids.add(event["identity_id"])
    if observed_prior_ids != prior_ids:
        raise contract.ContractError("routing snapshot event prefix is incomplete")
    core = {
        "final_ledger_schema": contract.PHASE6_FINAL_SCHEMA,
        "authority_id": bindings["authority_id"],
        "ledger_update_index": prefix_length,
        "event_prefix_sha256": contract.canonical_sha256(prefix_events),
        "closed_record_count": target_index,
        "closed_records_sha256": contract.canonical_sha256(prior_records),
        "next_identity_id": final_identity_id,
        "common_invalidity_present": any(
            record.get("state") in contract.PHASE6_TERMINAL_STATES
            and record.get("evidence", {}).get("classification")
            == "common_invalidity"
            for record in prior_records
        ),
    }
    return {**core, "ledger_prefix_sha256": contract.canonical_sha256(core)}


def _phase6_prelaunch_snapshot_valid(snapshot: Any) -> bool:
    fields = {
        "final_ledger_schema",
        "authority_id",
        "ledger_update_index",
        "event_prefix_sha256",
        "closed_record_count",
        "closed_records_sha256",
        "next_identity_id",
        "common_invalidity_present",
        "ledger_prefix_sha256",
    }
    if not isinstance(snapshot, Mapping) or set(snapshot) != fields:
        return False
    core = {key: snapshot[key] for key in fields - {"ledger_prefix_sha256"}}
    digest_fields = (
        "authority_id",
        "event_prefix_sha256",
        "closed_records_sha256",
        "ledger_prefix_sha256",
    )
    return (
        snapshot.get("final_ledger_schema") == contract.PHASE6_FINAL_SCHEMA
        and all(
            isinstance(snapshot.get(field), str)
            and len(snapshot[field]) == 64
            and all(character in "0123456789abcdef" for character in snapshot[field])
            for field in digest_fields
        )
        and type(snapshot.get("ledger_update_index")) is int
        and snapshot["ledger_update_index"] >= 0
        and type(snapshot.get("closed_record_count")) is int
        and snapshot["closed_record_count"] >= 0
        and isinstance(snapshot.get("next_identity_id"), str)
        and bool(snapshot["next_identity_id"])
        and type(snapshot.get("common_invalidity_present")) is bool
        and snapshot["ledger_prefix_sha256"] == contract.canonical_sha256(core)
    )


def _phase6_terminal_overlay_structurally_valid(
    overlay: Any, routing_roster: Sequence[Mapping[str, Any]]
) -> bool:
    if not isinstance(overlay, Mapping) or set(overlay) != {
        "final_ledger_sha256",
        "mode",
        "common_invalidity_sources",
        "dispositions",
    }:
        return False
    final_digest = overlay.get("final_ledger_sha256")
    mode = overlay.get("mode")
    sources = overlay.get("common_invalidity_sources")
    dispositions = overlay.get("dispositions")
    if (
        not isinstance(final_digest, str)
        or len(final_digest) != 64
        or any(character not in "0123456789abcdef" for character in final_digest)
        or mode not in {"none", "common_invalidity"}
        or not isinstance(sources, list)
        or not isinstance(dispositions, list)
        or len(dispositions) != len(routing_roster)
    ):
        return False
    final_roster = contract.phase6_expected_roster(contract.PHASE6_FINAL_SCHEMA)
    final_ids = {identity["identity_id"] for identity in final_roster}
    source_ids: list[str] = []
    for source in sources:
        if (
            not isinstance(source, Mapping)
            or set(source) != {"identity_id", "record_sha256"}
            or source.get("identity_id") not in final_ids
            or not isinstance(source.get("record_sha256"), str)
            or len(source["record_sha256"]) != 64
            or any(
                character not in "0123456789abcdef"
                for character in source["record_sha256"]
            )
        ):
            return False
        source_ids.append(source["identity_id"])
    if len(source_ids) != len(set(source_ids)) or (mode == "common_invalidity") != bool(sources):
        return False
    effective_actions = {
        "globally_invalidated_by_common_invalidity",
        "launched_under_prelaunch_eligibility",
        "not_launched_after_eligibility",
        "not_launched_by_prelaunch_route",
    }
    for route_identity, disposition in zip(
        routing_roster, dispositions, strict=True
    ):
        final_identity = _phase6_final_identity_for_route(route_identity)
        if (
            not isinstance(disposition, Mapping)
            or set(disposition)
            != {
                "identity_id",
                "final_identity_id",
                "final_state",
                "final_reason",
                "final_record_sha256",
                "effective_action",
            }
            or disposition.get("identity_id") != route_identity["identity_id"]
            or disposition.get("final_identity_id") != final_identity["identity_id"]
            or not isinstance(disposition.get("final_state"), str)
            or not (
                disposition["final_state"] in contract.PHASE6_TERMINAL_STATES
                or disposition["final_state"].startswith("not_launched:")
            )
            or not isinstance(disposition.get("final_reason"), str)
            or not disposition["final_reason"]
            or disposition["final_state"].startswith("not_launched:")
            and disposition["final_reason"]
            != disposition["final_state"].split(":", 1)[1]
            or not isinstance(disposition.get("final_record_sha256"), str)
            or len(disposition["final_record_sha256"]) != 64
            or any(
                character not in "0123456789abcdef"
                for character in disposition["final_record_sha256"]
            )
            or disposition.get("effective_action") not in effective_actions
            or mode == "common_invalidity"
            and disposition["effective_action"]
            != "globally_invalidated_by_common_invalidity"
            or mode == "none"
            and disposition["effective_action"]
            == "globally_invalidated_by_common_invalidity"
        ):
            return False
    return True


def phase6_new_routing_ledger(bindings: Mapping[str, Any]) -> dict[str, Any]:
    if not contract._phase6_bindings_valid(bindings):
        raise contract.ContractError("routing ledger requires valid Phase 6 bindings")
    if bindings["schedule"]["payload"]["ledger_schema"] != contract.PHASE6_FINAL_SCHEMA:
        raise contract.ContractError("routing ledger requires the Gate C final schedule")
    roster = contract.phase6_expected_roster(contract.PHASE6_ROUTING_SCHEMA)
    return {
        "schema": contract.PHASE6_ROUTING_SCHEMA,
        "authority_id": bindings["authority_id"],
        "state": "running",
        "update_index": 0,
        "terminal_overlay": None,
        "records": [
            {
                "identity": identity,
                "state": "pending_dependency",
                "reason": None,
                "dependencies": None,
                "prelaunch_snapshot": None,
                "fingerprints": None,
                "rule_id": None,
                "action": None,
            }
            for identity in roster
        ],
    }


def _phase6_routing_checks(payload: Any, *, final: bool) -> dict[str, bool]:
    roster = contract.phase6_expected_roster(contract.PHASE6_ROUTING_SCHEMA)
    records = payload.get("records") if isinstance(payload, Mapping) else None
    records_valid = isinstance(records, list) and len(records) == len(roster)
    decided = 0
    if records_valid:
        for identity, record in zip(roster, records, strict=True):
            if (
                not isinstance(record, Mapping)
                or set(record) != {
                    "identity",
                    "state",
                    "reason",
                    "dependencies",
                    "prelaunch_snapshot",
                    "fingerprints",
                    "rule_id",
                    "action",
                }
                or record.get("identity") != identity
                or record.get("state") not in {"pending_dependency", "decided"}
            ):
                records_valid = False
                break
            if record["state"] == "pending_dependency":
                if any(
                    record.get(field) is not None
                    for field in (
                        "reason",
                        "dependencies",
                        "prelaunch_snapshot",
                        "fingerprints",
                        "rule_id",
                        "action",
                    )
                ):
                    records_valid = False
                    break
                continue
            decided += 1
            dependencies = record.get("dependencies")
            prelaunch_snapshot = record.get("prelaunch_snapshot")
            fingerprints = record.get("fingerprints")
            rule_id = record.get("rule_id")
            action = record.get("action")
            reason = record.get("reason")
            if (
                not isinstance(dependencies, Mapping)
                or set(dependencies)
                != {"p50", "preceding_p150"}
                or not isinstance(dependencies.get("p50"), Mapping)
                or set(dependencies["p50"])
                != {"identity_id", "state", "record_sha256"}
                or not isinstance(dependencies["p50"].get("identity_id"), str)
                or not isinstance(dependencies["p50"].get("state"), str)
                or not isinstance(dependencies["p50"].get("record_sha256"), str)
                or len(dependencies["p50"]["record_sha256"]) != 64
                or dependencies.get("preceding_p150") is not None
                and (
                    not isinstance(dependencies["preceding_p150"], Mapping)
                    or set(dependencies["preceding_p150"])
                    != {"identity_id", "state", "record_sha256"}
                    or not isinstance(
                        dependencies["preceding_p150"].get("identity_id"), str
                    )
                    or not isinstance(
                        dependencies["preceding_p150"].get("state"), str
                    )
                    or not isinstance(
                        dependencies["preceding_p150"].get("record_sha256"), str
                    )
                    or len(dependencies["preceding_p150"]["record_sha256"])
                    != 64
                )
                or not _phase6_prelaunch_snapshot_valid(prelaunch_snapshot)
                or prelaunch_snapshot.get("authority_id")
                != payload.get("authority_id")
                or prelaunch_snapshot.get("next_identity_id")
                != _phase6_final_identity_for_route(identity)["identity_id"]
                or prelaunch_snapshot.get("closed_record_count")
                != contract.phase6_expected_roster(
                    contract.PHASE6_FINAL_SCHEMA
                ).index(_phase6_final_identity_for_route(identity))
                or not isinstance(fingerprints, Mapping)
                or set(fingerprints) != set(contract.FINGERPRINT_FIELDS)
                or any(
                    not isinstance(value, str)
                    or len(value) != 64
                    or any(character not in "0123456789abcdef" for character in value)
                    for value in fingerprints.values()
                )
                or not isinstance(rule_id, str)
                or not rule_id
                or not isinstance(action, str)
                or not action
            ):
                records_valid = False
                break
            p50_state = dependencies["p50"]["state"]
            preceding = dependencies["preceding_p150"]
            route_semantics = (
                rule_id == "common_invalidity"
                and action == "not_launched_common_invalidity"
                and reason == "common_invalidity"
                and prelaunch_snapshot["common_invalidity_present"]
                or rule_id == "invalid_dependency_evidence"
                and action == "not_launched_invalid_dependency_evidence"
                and reason == "invalid_dependency_evidence"
                and not prelaunch_snapshot["common_invalidity_present"]
                or rule_id == "p50_dependency_not_launched"
                and action == "not_launched_p50_dependency_not_launched"
                and reason == "p50_dependency_not_launched"
                and not prelaunch_snapshot["common_invalidity_present"]
                and str(p50_state).startswith("not_launched:")
                or rule_id == "p50_dependency_failed"
                and action == f"not_launched_p50_dependency_failed:{p50_state}"
                and reason == "p50_dependency_failed"
                and not prelaunch_snapshot["common_invalidity_present"]
                and p50_state in contract.PHASE6_TERMINAL_STATES
                and p50_state != "passed"
                or rule_id == "preceding_p150_batch_failed"
                and preceding is not None
                and action
                == f"not_launched_after_smaller_p150_batch_failure:{preceding['state']}"
                and reason == "after_smaller_p150_batch_failure"
                and not prelaunch_snapshot["common_invalidity_present"]
                and preceding["state"] != "passed"
                or rule_id
                == "p50_passed_and_preceding_p150_passed_or_not_applicable"
                and action == "eligible_under_gate_c_budget"
                and reason is None
                and not prelaunch_snapshot["common_invalidity_present"]
                and p50_state == "passed"
                and (preceding is None or preceding["state"] == "passed")
            )
            if not route_semantics:
                records_valid = False
                break
    return {
        "closed_schema": isinstance(payload, Mapping)
        and set(payload)
        == {
            "schema",
            "authority_id",
            "state",
            "update_index",
            "records",
            "terminal_overlay",
        },
        "schema_identity": isinstance(payload, Mapping)
        and payload.get("schema") == contract.PHASE6_ROUTING_SCHEMA,
        "authority_identity": isinstance(payload, Mapping)
        and isinstance(payload.get("authority_id"), str)
        and len(payload["authority_id"]) == 64
        and all(
            character in "0123456789abcdef"
            for character in payload["authority_id"]
        ),
        "records_identity": records_valid,
        "update_index": isinstance(payload, Mapping)
        and type(payload.get("update_index")) is int
        and payload["update_index"] == decided,
        "top_state": isinstance(payload, Mapping)
        and (
            payload.get("state") == "running"
            and decided < len(roster)
            and payload.get("terminal_overlay") is None
            or payload.get("state") == "decisions_complete"
            and decided == len(roster)
            and payload.get("terminal_overlay") is None
            or payload.get("state") == "closed"
            and decided == len(roster)
            and _phase6_terminal_overlay_structurally_valid(
                payload.get("terminal_overlay"), roster
            )
        ),
        "final": not final
        or (
            decided == len(roster)
            and payload.get("state") == "closed"
            and _phase6_terminal_overlay_structurally_valid(
                payload.get("terminal_overlay"), roster
            )
        ),
    }


def phase6_persist_routing(
    path: Path, payload: Mapping[str, Any], *, final: bool
) -> dict[str, Any]:
    checks = _phase6_routing_checks(payload, final=final)
    if not all(checks.values()):
        raise contract.ContractError(f"invalid Phase 6 routing ledger: {checks}")
    if path.exists():
        prior = contract.read_strict_json(path)
        prior_checks = _phase6_routing_checks(prior, final=False)
        if not all(prior_checks.values()):
            raise contract.ContractError("existing Phase 6 routing ledger is invalid")
        if payload["authority_id"] != prior["authority_id"]:
            raise contract.ContractError("Phase 6 routing authority changed")
        overlay_transition = (
            payload["update_index"] == prior["update_index"]
            and prior["state"] == "decisions_complete"
            and prior["terminal_overlay"] is None
            and payload["state"] == "closed"
            and isinstance(payload["terminal_overlay"], Mapping)
            and payload["records"] == prior["records"]
        )
        if payload["update_index"] < prior["update_index"] or (
            payload["update_index"] == prior["update_index"] and not overlay_transition
        ):
            raise contract.ContractError("Phase 6 routing update index did not advance")
        for old, new in zip(prior["records"], payload["records"], strict=True):
            if old["state"] == "decided" and old != new:
                raise contract.ContractError("Phase 6 routing decision is immutable")
    contract.durable_atomic_write_json(path, payload)
    reparsed = contract.read_strict_json(path)
    if reparsed != payload or not all(
        _phase6_routing_checks(reparsed, final=final).values()
    ):
        raise contract.ContractError("persisted Phase 6 routing ledger failed reparse")
    return reparsed


def _phase6_dependency_digest(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "identity_id": record["identity"]["identity_id"],
        "state": record["state"],
        "record_sha256": contract.canonical_sha256(record),
    }


def phase6_routing_decision(
    payload: Mapping[str, Any],
    final_payload: Mapping[str, Any],
    identity: Mapping[str, Any],
) -> dict[str, Any]:
    if not all(_phase6_routing_checks(payload, final=False).values()):
        raise contract.ContractError("cannot decide an invalid routing ledger")
    if identity not in contract.phase6_expected_roster(contract.PHASE6_ROUTING_SCHEMA):
        raise contract.ContractError("unknown Phase 6 routing identity")
    target = next(record for record in payload["records"] if record["identity"] == identity)
    by_key = {
        (
            record["identity"]["dimension"],
            record["identity"]["parameter_count"],
            record["identity"]["batch_size"],
            record["identity"]["method_id"],
        ): record
        for record in final_payload["records"]
    }
    d, b, method = identity["dimension"], identity["batch_size"], identity["method_id"]
    p50 = by_key[(d, 50, b, method)]
    smaller_b = {4: 1, 16: 4}.get(b)
    preceding = by_key[(d, 150, smaller_b, method)] if smaller_b is not None else None
    schedule_row = next(
        row
        for row in final_payload["bindings"]["schedule"]["payload"]["records"]
        if (
            row["identity"]["dimension"],
            row["identity"]["parameter_count"],
            row["identity"]["batch_size"],
            row["identity"]["method_id"],
        )
        == (
            identity["dimension"],
            150,
            identity["batch_size"],
            identity["method_id"],
        )
    )
    expected_dependencies = {
        "p50": _phase6_dependency_digest(p50),
        "preceding_p150": (
            _phase6_dependency_digest(preceding) if preceding is not None else None
        ),
    }
    expected_snapshot = _phase6_prelaunch_snapshot(final_payload, identity)
    if target["state"] != "pending_dependency":
        if (
            target["dependencies"] != expected_dependencies
            or target["prelaunch_snapshot"] != expected_snapshot
            or target["fingerprints"] != schedule_row["fingerprints"]
            or (target["rule_id"] == "common_invalidity")
            is not expected_snapshot["common_invalidity_present"]
        ):
            raise contract.ContractError(
                "persisted Phase 6 routing decision no longer matches dependencies"
            )
        return copy.deepcopy(dict(payload))
    common_invalidity = expected_snapshot["common_invalidity_present"]
    if common_invalidity:
        rule_id = "common_invalidity"
        action = "not_launched_common_invalidity"
        reason = "common_invalidity"
    elif p50["state"] in {"pending", "running"} or (
        preceding is not None and preceding["state"] in {"pending", "running"}
    ):
        rule_id = "invalid_dependency_evidence"
        action = "not_launched_invalid_dependency_evidence"
        reason = "invalid_dependency_evidence"
    elif str(p50["state"]).startswith("not_launched:"):
        rule_id = "p50_dependency_not_launched"
        action = "not_launched_p50_dependency_not_launched"
        reason = "p50_dependency_not_launched"
    elif p50["state"] != "passed":
        rule_id = "p50_dependency_failed"
        action = f"not_launched_p50_dependency_failed:{p50['state']}"
        reason = "p50_dependency_failed"
    elif preceding is not None and preceding["state"] != "passed":
        rule_id = "preceding_p150_batch_failed"
        action = f"not_launched_after_smaller_p150_batch_failure:{preceding['state']}"
        reason = "after_smaller_p150_batch_failure"
    else:
        rule_id = "p50_passed_and_preceding_p150_passed_or_not_applicable"
        action = "eligible_under_gate_c_budget"
        reason = None
    updated = copy.deepcopy(dict(payload))
    record = next(row for row in updated["records"] if row["identity"] == identity)
    record.update(
        {
            "state": "decided",
            "reason": reason,
            "dependencies": expected_dependencies,
            "prelaunch_snapshot": expected_snapshot,
            "fingerprints": copy.deepcopy(schedule_row["fingerprints"]),
            "rule_id": rule_id,
            "action": action,
        }
    )
    updated["update_index"] += 1
    if updated["update_index"] == len(updated["records"]):
        updated["state"] = "decisions_complete"
    if not all(_phase6_routing_checks(updated, final=False).values()):
        raise contract.ContractError("Phase 6 routing decision failed validation")
    return updated


def _phase6_routing_terminal_overlay(
    routing: Mapping[str, Any], final_payload: Mapping[str, Any]
) -> dict[str, Any]:
    if routing.get("state") != "decisions_complete":
        raise contract.ContractError("routing decisions are not complete")
    common_sources = [
        {
            "identity_id": record["identity"]["identity_id"],
            "record_sha256": contract.canonical_sha256(record),
        }
        for record in final_payload["records"]
        if record.get("state") in contract.PHASE6_TERMINAL_STATES
        and record.get("evidence", {}).get("classification") == "common_invalidity"
    ]
    common = bool(common_sources)
    final_by_id = {
        record["identity"]["identity_id"]: record for record in final_payload["records"]
    }
    dispositions = []
    for route in routing["records"]:
        route_id = route["identity"]["identity_id"]
        final_identity = _phase6_final_identity_for_route(route["identity"])
        record = final_by_id[final_identity["identity_id"]]
        if common:
            effective = "globally_invalidated_by_common_invalidity"
        elif record["state"] in contract.PHASE6_TERMINAL_STATES:
            if route["action"] != "eligible_under_gate_c_budget":
                raise contract.ContractError("launched P=150 record had an ineligible route")
            effective = "launched_under_prelaunch_eligibility"
        elif route["action"] == "eligible_under_gate_c_budget":
            if record["state"] != "not_launched:global_budget_exhausted":
                raise contract.ContractError("eligible P=150 route has an invalid final disposition")
            effective = "not_launched_after_eligibility"
        else:
            if record["state"] != f"not_launched:{route['reason']}":
                raise contract.ContractError("ineligible P=150 route does not match final record")
            effective = "not_launched_by_prelaunch_route"
        dispositions.append(
            {
                "identity_id": route_id,
                "final_identity_id": final_identity["identity_id"],
                "final_state": record["state"],
                "final_reason": record["reason"],
                "final_record_sha256": contract.canonical_sha256(record),
                "effective_action": effective,
            }
        )
    return {
        "final_ledger_sha256": contract.canonical_sha256(final_payload),
        "mode": "common_invalidity" if common else "none",
        "common_invalidity_sources": common_sources,
        "dispositions": dispositions,
    }


def phase6_close_and_validate_routing(
    routing: Mapping[str, Any], final_payload: Mapping[str, Any]
) -> dict[str, Any]:
    if not all(contract.phase6_ledger_checks(final_payload, final=True).values()):
        raise contract.ContractError("routing closure requires a valid final ledger")
    if routing.get("authority_id") != final_payload.get("bindings", {}).get("authority_id"):
        raise contract.ContractError("routing and final ledger authority differ")
    if not all(_phase6_routing_checks(routing, final=False).values()):
        raise contract.ContractError("routing decisions are invalid before closure")
    for identity in contract.phase6_expected_roster(contract.PHASE6_ROUTING_SCHEMA):
        checked = phase6_routing_decision(routing, final_payload, identity)
        if checked != routing:
            raise contract.ContractError("routing decision changed during terminal validation")
    updated = copy.deepcopy(dict(routing))
    updated["terminal_overlay"] = _phase6_routing_terminal_overlay(
        routing, final_payload
    )
    updated["state"] = "closed"
    if not all(_phase6_routing_checks(updated, final=True).values()):
        raise contract.ContractError("closed routing artifact is structurally invalid")
    return updated


def phase6_final_routing_checks(
    final_payload: Mapping[str, Any], routing: Mapping[str, Any]
) -> dict[str, bool]:
    checks = {
        "final_ledger": all(contract.phase6_ledger_checks(final_payload, final=True).values()),
        "routing_closed": all(_phase6_routing_checks(routing, final=True).values()),
        "authority_identity": routing.get("authority_id")
        == final_payload.get("bindings", {}).get("authority_id"),
    }
    expected_overlay: Mapping[str, Any] | None = None
    if checks["routing_closed"] and checks["final_ledger"]:
        decisions = copy.deepcopy(dict(routing))
        decisions["state"] = "decisions_complete"
        decisions["terminal_overlay"] = None
        try:
            for identity in contract.phase6_expected_roster(
                contract.PHASE6_ROUTING_SCHEMA
            ):
                if phase6_routing_decision(decisions, final_payload, identity) != decisions:
                    raise contract.ContractError("routing decision drift")
            expected_overlay = _phase6_routing_terminal_overlay(decisions, final_payload)
        except (KeyError, contract.ContractError):
            expected_overlay = None
    checks["decision_correspondence"] = expected_overlay is not None
    checks["terminal_overlay"] = (
        expected_overlay is not None
        and routing.get("terminal_overlay") == expected_overlay
    )
    return checks


def _phase6_routing_identity(identity: Mapping[str, Any]) -> dict[str, Any]:
    return contract.phase6_identity(
        dimension=identity["dimension"],
        parameter_count=150,
        batch_size=identity["batch_size"],
        dtype=identity["dtype"],
        method_id=identity["method_id"],
        operation="p150_routing",
    )


def phase6_execute_ledger(
    *,
    schema: str,
    output_path: Path,
    bindings: Mapping[str, Any],
    child_timeout_seconds: float,
    eligible_identity_ids: set[str] | None = None,
    imported_records: Mapping[str, Mapping[str, Any]] | None = None,
    authority_validator: Callable[[Mapping[str, Any]], None] | None = None,
    budget_path: Path | None = None,
    budget_lease: Phase6BudgetLease | None = None,
    expected_budget_command_name: str | None = None,
    gate_hard_ceiling_seconds: float | None = None,
    cell_cap_seconds: float | None = None,
    routing_path: Path | None = None,
    initial_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    gate, artifact_kind, _ = contract.PHASE6_SCHEMA_CONTRACTS[schema]
    roster = contract.phase6_expected_roster(schema)
    output_path = _phase6_repo_path(output_path)
    resolved_routing_path: Path | None = None
    routing: dict[str, Any] | None = None
    if schema == contract.PHASE6_FINAL_SCHEMA:
        if routing_path is None:
            raise contract.ContractError("Gate C final execution requires routing output")
        resolved_routing_path = _phase6_repo_path(routing_path)
        if resolved_routing_path.exists():
            routing = contract.read_strict_json(resolved_routing_path)
            if (
                routing.get("authority_id") != bindings["authority_id"]
                or not all(_phase6_routing_checks(routing, final=False).values())
            ):
                raise contract.ContractError("existing Phase 6 routing ledger is invalid")
        else:
            routing = phase6_persist_routing(
                resolved_routing_path,
                phase6_new_routing_ledger(bindings),
                final=False,
            )
    elif routing_path is not None:
        raise contract.ContractError("routing output is valid only for Gate C final")
    if output_path.exists():
        payload = contract.read_strict_json(output_path)
        if payload.get("schema") != schema or payload.get("bindings") != bindings:
            raise contract.ContractError("existing Phase 6 ledger does not match current authority")
        final_checks = contract.phase6_ledger_checks(payload, final=True)
        if all(final_checks.values()):
            if authority_validator is not None:
                authority_validator(bindings)
            if routing is not None:
                assert resolved_routing_path is not None
                if routing.get("state") == "decisions_complete":
                    routing = phase6_persist_routing(
                        resolved_routing_path,
                        phase6_close_and_validate_routing(routing, payload),
                        final=True,
                    )
                elif routing.get("state") != "closed":
                    raise contract.ContractError(
                        "terminal final ledger has incomplete routing decisions"
                    )
                correspondence = phase6_final_routing_checks(payload, routing)
                if not all(correspondence.values()):
                    raise contract.ContractError(
                        f"terminal Phase 6 routing does not match final ledger: {correspondence}"
                    )
            return payload
        checks = contract.phase6_ledger_checks(payload, final=False)
        if not all(checks.values()):
            raise contract.ContractError(f"existing Phase 6 ledger is invalid: {checks}")
    else:
        payload = (
            contract.new_phase6_ledger(
                schema=schema,
                gate=gate,
                artifact_kind=artifact_kind,
                identities=roster,
                bindings=bindings,
            )
            if initial_payload is None
            else copy.deepcopy(dict(initial_payload))
        )
        expected_initial = contract.new_phase6_ledger(
            schema=schema,
            gate=gate,
            artifact_kind=artifact_kind,
            identities=roster,
            bindings=bindings,
        )
        if payload != expected_initial:
            raise contract.ContractError(
                "preconstructed Phase 6 initial ledger differs from current authority"
            )
        payload = phase6_persist_and_validate(output_path, payload, final=False)
    schedule_by_id = {
        row["identity"]["identity_id"]: row
        for row in bindings["schedule"]["payload"]["records"]
    }
    discovery = bindings["proposal"]["strict_json"]["dependency_discovery"]["manifest"]
    imported_common_invalidity = False
    if imported_records is not None:
        if schema != contract.PHASE6_FINAL_SCHEMA:
            raise contract.ContractError("imported Phase 6 records are valid only for Gate C final")
        pilot_blobs = [
            blob
            for blob in bindings["authority_inputs"]
            if blob.get("strict_json", {}).get("schema")
            == contract.PHASE6_PILOT_SCHEMA
        ]
        if len(pilot_blobs) != 1:
            raise contract.ContractError("Gate C import requires one immutable pilot blob")
        pilot_payload = pilot_blobs[0]["strict_json"]
        expected_records = {
            record["identity"]["identity_id"]: record
            for record in pilot_payload.get("records", [])
        }
        if (
            set(imported_records) != set(expected_records)
            or any(imported_records[key] != expected_records[key] for key in expected_records)
            or not all(
                contract.phase6_ledger_checks(pilot_payload, final=True).values()
            )
        ):
            raise contract.ContractError("Gate C imported pilot roster or bytes are invalid")
        imported_common_invalidity = any(
            record.get("state") in contract.PHASE6_TERMINAL_STATES
            and record.get("evidence", {}).get("classification")
            == "common_invalidity"
            for record in imported_records.values()
        )
    running_records = [record for record in payload["records"] if record["state"] == "running"]
    if running_records:
        running_record = running_records[0]
        recovered_process = _phase6_recover_running_process(running_record["process"])
        recovery_evidence = _phase6_record_evidence(
            running_record["identity"],
            classification="supervisor_interruption",
            discovery=discovery,
        )
        payload = contract.transition_phase6_record(
            payload,
            identity_id=running_record["identity"]["identity_id"],
            new_state="interrupted",
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            reason="supervisor_recovery",
            process=recovered_process,
            evidence=recovery_evidence,
        )
        payload = phase6_persist_and_validate(output_path, payload, final=False)
    if imported_common_invalidity and any(
        record.get("imported_from") is None
        and record.get("state") not in {"pending", "not_launched:common_invalidity"}
        for record in payload["records"]
    ):
        raise contract.ContractError(
            "Gate C already launched work after imported common invalidity"
        )
    cell_started_ns: dict[tuple[int, int, int], int] = {}

    def decide_route(identity: Mapping[str, Any]) -> Mapping[str, Any] | None:
        nonlocal routing
        if identity["parameter_count"] != 150 or routing is None:
            return None
        if authority_validator is not None:
            authority_validator(bindings)
        route_identity = _phase6_routing_identity(identity)
        prior_update_index = routing["update_index"]
        routing = phase6_routing_decision(routing, payload, route_identity)
        if routing["update_index"] != prior_update_index:
            assert resolved_routing_path is not None
            routing = phase6_persist_routing(
                resolved_routing_path,
                routing,
                final=routing["state"] == "closed",
            )
        return next(
            record
            for record in routing["records"]
            if record["identity"] == route_identity
        )

    for identity in roster:
        identity_id = identity["identity_id"]
        current_record = next(
            record for record in payload["records"] if record["identity"] == identity
        )
        if current_record["state"] != "pending":
            if identity["parameter_count"] == 150 and routing is not None:
                route_identity = _phase6_routing_identity(identity)
                route_record = next(
                    record
                    for record in routing["records"]
                    if record["identity"] == route_identity
                )
                if route_record["state"] != "decided":
                    raise contract.ContractError(
                        "terminal P=150 record lacks a prelaunch routing decision"
                    )
                decide_route(identity)
            continue
        imported = imported_records.get(identity_id) if imported_records is not None else None
        if imported is not None:
            pilot_blob = pilot_blobs[0]
            payload = contract.transition_phase6_record(
                payload,
                identity_id=identity_id,
                new_state=imported["state"],
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
                reason=imported["reason"],
                process=imported["process"],
                evidence=imported["evidence"],
                imported_from={
                    "kind": "gate_b_pilot",
                    "pilot_artifact_sha256": pilot_blob["sha256"],
                    "pilot_record_sha256": contract.canonical_sha256(imported),
                },
            )
            payload = phase6_persist_and_validate(output_path, payload, final=False)
            continue
        if imported_common_invalidity:
            route_record = decide_route(identity)
            if route_record is not None and route_record["reason"] != "common_invalidity":
                raise contract.ContractError(
                    "imported common invalidity did not close the P=150 route"
                )
            payload = contract.transition_phase6_record(
                payload,
                identity_id=identity_id,
                new_state="not_launched:common_invalidity",
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
                reason="common_invalidity",
            )
            payload = phase6_persist_and_validate(output_path, payload, final=False)
            continue
        route_record = decide_route(identity)
        if (
            route_record is not None
            and route_record["action"] != "eligible_under_gate_c_budget"
        ):
            route_reason = route_record["reason"]
            if not isinstance(route_reason, str):
                raise contract.ContractError("ineligible Phase 6 route lacks reason")
            payload = contract.transition_phase6_record(
                payload,
                identity_id=identity_id,
                new_state=f"not_launched:{route_reason}",
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
                reason=route_reason,
            )
            payload = phase6_persist_and_validate(output_path, payload, final=False)
            continue
        if eligible_identity_ids is not None and identity_id not in eligible_identity_ids:
            reason = (
                "not_in_gate_b_pilot"
                if schema == contract.PHASE6_FINAL_SCHEMA
                else "trace_gate_not_passed"
            )
            payload = contract.transition_phase6_record(
                payload,
                identity_id=identity_id,
                new_state=f"not_launched:{reason}",
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
                reason=reason,
            )
            payload = phase6_persist_and_validate(output_path, payload, final=False)
            continue
        if route_record is not None:
            prune_reason = None
        else:
            prune_reason = _phase6_should_prune(payload, identity)
        if prune_reason is not None:
            payload = contract.transition_phase6_record(
                payload,
                identity_id=identity_id,
                new_state=f"not_launched:{prune_reason}",
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
                reason=prune_reason,
            )
            payload = phase6_persist_and_validate(output_path, payload, final=False)
            continue
        now_ns: int | None = None
        budget_exhausted = False
        if budget_path is not None:
            if (
                gate_hard_ceiling_seconds is None
                or budget_lease is None
                or expected_budget_command_name is None
            ):
                raise contract.ContractError(
                    "Phase 6 budget path requires its lease, command, and hard ceiling"
                )
            now_ns = time.monotonic_ns()
            budget_state = phase6_budget_state_checkpoint(
                budget_path,
                budget_lease,
                authority_id=bindings["authority_id"],
                gate=gate,
                hard_ceiling_seconds=gate_hard_ceiling_seconds,
                command_name=expected_budget_command_name,
                now_ns=now_ns,
            )
            budget_exhausted = (
                phase6_budget_state_remaining(budget_state, now_ns)
                < child_timeout_seconds + 10.0
            )
        launch_deadline = float(child_timeout_seconds)
        launch_clock_ns = time.monotonic_ns() if now_ns is None else now_ns
        if cell_cap_seconds is not None:
            if cell_cap_seconds <= 0:
                raise contract.ContractError("Phase 6 cell cap must be positive")
            cell_key = (
                identity["dimension"],
                identity["parameter_count"],
                identity["batch_size"],
            )
            cell_now_ns = launch_clock_ns
            if cell_key not in cell_started_ns:
                prior_starts = [
                    record.get("process", {}).get("started_ns")
                    for record in payload["records"]
                    if (
                        record["identity"]["dimension"],
                        record["identity"]["parameter_count"],
                        record["identity"]["batch_size"],
                    )
                    == cell_key
                    and record.get("imported_from") is None
                    and isinstance(record.get("process"), Mapping)
                    and type(record["process"].get("started_ns")) is int
                ]
                cell_started_ns[cell_key] = (
                    min(prior_starts) if prior_starts else cell_now_ns
                )
            cell_remaining = cell_cap_seconds - (
                cell_now_ns - cell_started_ns[cell_key]
            ) / 1.0e9
            if cell_remaining <= 10.0:
                budget_exhausted = True
            else:
                launch_deadline = min(launch_deadline, cell_remaining - 10.0)
        if budget_exhausted:
            payload = contract.transition_phase6_record(
                payload,
                identity_id=identity_id,
                new_state="not_launched:global_budget_exhausted",
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
                reason="global_budget_exhausted",
            )
            payload = phase6_persist_and_validate(output_path, payload, final=False)
            continue
        row = schedule_by_id.get(identity_id)
        if row is None:
            raise contract.ContractError(f"missing Phase 6 schedule row {identity_id}")
        def mark_running(process_identity: Mapping[str, Any]) -> None:
            nonlocal payload
            payload = contract.transition_phase6_record(
                payload,
                identity_id=identity_id,
                new_state="running",
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
                process=process_identity,
            )
            payload = phase6_persist_and_validate(output_path, payload, final=False)

        terminal_committed = False

        def commit_completed(process: Mapping[str, Any]) -> None:
            nonlocal payload, terminal_committed
            state, reason, classification = _phase6_terminal_classification(identity, process)
            evidence = _phase6_record_evidence(
                identity,
                classification=classification,
                discovery=discovery,
            )
            authority_invalid = False
            if authority_validator is not None:
                try:
                    authority_validator(bindings)
                except contract.ContractError:
                    authority_invalid = True
            if authority_invalid:
                state = (
                    "timed_out"
                    if process["timed_out"]
                    else "crashed"
                    if process["returncode"] is not None
                    and process["returncode"] < 0
                    else "failed"
                )
                reason = "authority_revalidation_failed"
                evidence["classification"] = "common_invalidity"
            candidate_record = {
                "identity": identity,
                "state": state,
                "reason": reason,
                "process": process,
                "evidence": evidence,
                "imported_from": None,
            }
            if not authority_invalid and not contract.phase6_terminal_record_semantics_valid(
                candidate_record, bindings=bindings
            ):
                state, reason, evidence["classification"] = (
                    "failed",
                    "invalid_child_evidence",
                    "common_invalidity",
                )
            payload = contract.transition_phase6_record(
                payload,
                identity_id=identity_id,
                new_state=state,
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
                reason=reason,
                process=process,
                evidence=evidence,
            )
            payload = phase6_persist_and_validate(output_path, payload, final=False)
            terminal_committed = True

        absolute_deadline_ns = launch_clock_ns + int(launch_deadline * 1.0e9)
        try:
            with phase6_outer_sigterm_guard(lambda signum: None) as termination:
                if budget_path is not None:
                    assert budget_lease is not None
                    assert expected_budget_command_name is not None
                    assert gate_hard_ceiling_seconds is not None
                    prelaunch_now = time.monotonic_ns()
                    budget_state = phase6_budget_state_checkpoint(
                        budget_path,
                        budget_lease,
                        authority_id=bindings["authority_id"],
                        gate=gate,
                        hard_ceiling_seconds=gate_hard_ceiling_seconds,
                        command_name=expected_budget_command_name,
                        now_ns=prelaunch_now,
                    )
                    gate_deadline = budget_state["deadline_ns"] - int(10.0e9)
                    absolute_deadline_ns = min(absolute_deadline_ns, gate_deadline)
                    if absolute_deadline_ns <= prelaunch_now:
                        payload = contract.transition_phase6_record(
                            payload,
                            identity_id=identity_id,
                            new_state="not_launched:global_budget_exhausted",
                            timestamp_utc=datetime.now(timezone.utc).isoformat(),
                            reason="global_budget_exhausted",
                        )
                        payload = phase6_persist_and_validate(
                            output_path, payload, final=False
                        )
                        continue
                _phase6_prepare_child_artifacts(identity)
                if authority_validator is not None:
                    try:
                        authority_validator(bindings)
                    except contract.ContractError:
                        for pending in [
                            record
                            for record in payload["records"]
                            if record["state"] == "pending"
                        ]:
                            payload = contract.transition_phase6_record(
                                payload,
                                identity_id=pending["identity"]["identity_id"],
                                new_state="not_launched:common_invalidity",
                                timestamp_utc=datetime.now(timezone.utc).isoformat(),
                                reason="common_invalidity",
                            )
                            payload = phase6_persist_and_validate(
                                output_path, payload, final=False
                            )
                        payload = contract.finalize_phase6_ledger(payload)
                        return phase6_persist_and_validate(
                            output_path, payload, final=True
                        )
                if (
                    authority_validator is not None
                    and time.monotonic_ns() >= absolute_deadline_ns
                ):
                    payload = contract.transition_phase6_record(
                        payload,
                        identity_id=identity_id,
                        new_state="not_launched:global_budget_exhausted",
                        timestamp_utc=datetime.now(timezone.utc).isoformat(),
                        reason="global_budget_exhausted",
                    )
                    payload = phase6_persist_and_validate(
                        output_path, payload, final=False
                    )
                    continue
                child_environment = _phase6_environment()
                if authority_validator is not None:
                    snapshot = contract.phase6_child_authority_snapshot(bindings, row)
                    snapshot_path = _phase6_child_paths(identity)["authority_snapshot"]
                    contract.durable_atomic_write_json(snapshot_path, snapshot)
                    snapshot_sha256 = contract.file_sha256(snapshot_path)
                    if snapshot_sha256 != contract.durable_json_sha256(snapshot):
                        raise contract.ContractError(
                            "Phase 6 child authority snapshot failed durable digest validation"
                        )
                    child_environment[
                        contract.PHASE6_CHILD_AUTHORITY_SHA256_ENV
                    ] = snapshot_sha256
                process = run_managed_process_group(
                    row["child_command_argv"],
                    environment=child_environment,
                    deadline_seconds=launch_deadline,
                    on_started=mark_running,
                    on_completed=commit_completed,
                    termination=termination,
                    absolute_deadline_ns=absolute_deadline_ns,
                )
                if not terminal_committed:
                    with termination.defer():
                        commit_completed(process)
        except BaseException as exc:
            if output_path.exists():
                try:
                    durable_payload = contract.read_strict_json(output_path)
                    durable_checks = contract.phase6_ledger_checks(
                        durable_payload, final=False
                    )
                except (OSError, contract.ContractError):
                    durable_payload = None
                    durable_checks = {}
                if (
                    isinstance(durable_payload, Mapping)
                    and durable_payload.get("schema") == schema
                    and durable_payload.get("bindings") == bindings
                    and all(durable_checks.values())
                ):
                    payload = dict(durable_payload)
            running_record = next(
                record
                for record in payload["records"]
                if record["identity"] == identity
            )
            if running_record["state"] == "running":
                interrupted_process = _phase6_recover_running_process(
                    running_record["process"]
                )
                interrupted_evidence = _phase6_record_evidence(
                    identity,
                    classification="supervisor_interruption",
                    discovery=discovery,
                )
                payload = contract.transition_phase6_record(
                    payload,
                    identity_id=identity_id,
                    new_state="interrupted",
                    timestamp_utc=datetime.now(timezone.utc).isoformat(),
                    reason=(
                        "outer_termination"
                        if isinstance(exc, Phase6OuterTermination)
                        else "keyboard_interrupt"
                        if isinstance(exc, KeyboardInterrupt)
                        else "supervisor_recovery"
                    ),
                    process=interrupted_process,
                    evidence=interrupted_evidence,
                )
                payload = phase6_persist_and_validate(
                    output_path, payload, final=False
                )
            raise
    payload = contract.finalize_phase6_ledger(payload)
    payload = phase6_persist_and_validate(output_path, payload, final=True)
    if routing is not None:
        assert resolved_routing_path is not None
        if routing.get("state") != "decisions_complete":
            raise contract.ContractError("Phase 6 routing decisions did not complete")
        routing = phase6_persist_routing(
            resolved_routing_path,
            phase6_close_and_validate_routing(routing, payload),
            final=True,
        )
        reparsed_payload = contract.read_strict_json(output_path)
        reparsed_routing = contract.read_strict_json(resolved_routing_path)
        correspondence = phase6_final_routing_checks(
            reparsed_payload, reparsed_routing
        )
        if not all(correspondence.values()):
            raise contract.ContractError(
                f"Phase 6 routing/final closure failed correspondence: {correspondence}"
            )
    return payload


def _phase6_resolve_authority_path(path: Path | None, label: str) -> Path:
    if path is None:
        raise contract.ContractError(f"Phase 6 {label} path is required")
    resolved = _phase6_repo_path(path)
    if not resolved.is_file() or path.is_symlink():
        raise contract.ContractError(f"Phase 6 {label} is not a regular file")
    return resolved


def _phase6_validate_cli_common(args: argparse.Namespace) -> None:
    if (
        args.timesteps != 120
        or args.dtype != "float32"
        or args.device != "cpu"
        or args.cpu_threads != 1
        or os.environ.get("CUDA_VISIBLE_DEVICES") != "-1"
    ):
        raise contract.ContractError("Phase 6 CLI common target identity mismatch")


def _phase6_budget_state_path(gate: str, authority_id: str) -> Path:
    if gate not in PHASE6_BUDGET_COMMAND_ORDER or len(authority_id) != 64:
        raise contract.ContractError("invalid Phase 6 budget-state identity")
    return (
        PHASE6_WORK_DIR
        / "budget_state"
        / f"{gate}-{authority_id}.json"
    )


def _phase6_gate_b_command() -> dict[str, Any]:
    return {
        "name": "trace_census_and_pilot",
        "argv": [
            contract.PHASE6_PYTHON,
            contract.PHASE6_SUPERVISOR_RELATIVE,
            "--phase6-pilot",
            "--dimensions", "10", "20", "30",
            "--parameter-counts", "50", "150",
            "--batch-sizes", "1", "4", "16",
            "--timesteps", "120",
            "--dtype", "float32",
            "--device", "cpu",
            "--cpu-threads", "1",
            "--jit-compile",
            "--trace-child-timeout-seconds", "60",
            "--xla-child-timeout-seconds", "60",
            "--xla-cell-timeout-seconds", "160",
            "--budget-contract", contract.PHASE6_GATE_B_BUDGET_RELATIVE,
            "--budget-attestation", contract.PHASE6_GATE_B_ATTESTATION_RELATIVE,
            "--trace-output-json", contract.PHASE6_GATE_B_ARTIFACTS["trace_output_json"],
            "--output-json", contract.PHASE6_GATE_B_ARTIFACTS["pilot_output_json"],
        ],
        "environment": dict(contract.PHASE6_ENVIRONMENT),
        "term_deadline_seconds": 3000,
        "kill_grace_seconds": 45,
    }


def _phase6_exact_supervisor_argv() -> list[str]:
    raw = getattr(sys, "orig_argv", None)
    if (
        not isinstance(raw, list)
        or len(raw) < 2
        or any(not isinstance(value, str) for value in raw)
        or raw[0] != contract.PHASE6_PYTHON
        or raw[2:] != sys.argv[1:]
    ):
        raise contract.ContractError(
            "Phase 6 supervisor cannot establish its exact process argv"
        )
    invoked_script = Path(raw[1])
    if not invoked_script.is_absolute():
        invoked_script = Path.cwd() / invoked_script
    try:
        script_matches = invoked_script.resolve(strict=True) == Path(__file__).resolve(
            strict=True
        )
    except OSError:
        script_matches = False
    if not script_matches:
        raise contract.ContractError(
            "Phase 6 supervisor process argv names a different script"
        )
    return list(raw)


def _phase6_gate_c_commands() -> list[dict[str, Any]]:
    scalar = {
        "name": "scalar_references",
        "argv": [
            contract.PHASE6_PYTHON,
            contract.PHASE6_SUPERVISOR_RELATIVE,
            "--phase6-scalar-references",
            "--dimensions", "10",
            "--parameter-counts", "50",
            "--batch-sizes", "1", "4",
            "--timesteps", "120",
            "--dtype", "float32",
            "--device", "cpu",
            "--cpu-threads", "1",
            "--no-jit-compile",
            "--child-timeout-seconds", "60",
            "--budget-contract", contract.PHASE6_GATE_C_BUDGET_RELATIVE,
            "--budget-attestation", contract.PHASE6_GATE_C_ATTESTATION_RELATIVE,
            "--output-json", contract.PHASE6_GATE_C_ARTIFACTS["scalar_output_json"],
        ],
        "environment": dict(contract.PHASE6_ENVIRONMENT),
        "term_deadline_seconds": 330,
        "kill_grace_seconds": 45,
    }
    remaining = {
        "name": "remaining_lattice",
        "argv": [
            contract.PHASE6_PYTHON,
            contract.PHASE6_SUPERVISOR_RELATIVE,
            "--phase6-remaining",
            "--dimensions", "10", "20", "30",
            "--parameter-counts", "50", "150",
            "--batch-sizes", "1", "4", "16",
            "--timesteps", "120",
            "--dtype", "float32",
            "--device", "cpu",
            "--cpu-threads", "1",
            "--jit-compile",
            "--child-timeout-seconds", "60",
            "--cell-timeout-seconds", "160",
            "--trace-input", contract.PHASE6_GATE_B_ARTIFACTS["trace_output_json"],
            "--pilot-input", contract.PHASE6_GATE_B_ARTIFACTS["pilot_output_json"],
            "--scalar-reference-input", contract.PHASE6_GATE_C_ARTIFACTS["scalar_output_json"],
            "--budget-contract", contract.PHASE6_GATE_C_BUDGET_RELATIVE,
            "--budget-attestation", contract.PHASE6_GATE_C_ATTESTATION_RELATIVE,
            "--routing-output-json", contract.PHASE6_GATE_C_ARTIFACTS["routing_output_json"],
            "--output-json", contract.PHASE6_GATE_C_ARTIFACTS["final_output_json"],
        ],
        "environment": dict(contract.PHASE6_ENVIRONMENT),
        "term_deadline_seconds": 2700,
        "kill_grace_seconds": 45,
    }
    return [scalar, remaining]


def _phase6_run_import_discovery() -> dict[str, Any]:
    output_path = Path(contract.PHASE6_IMPORT_DISCOVERY_OUTPUT)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() or output_path.is_symlink():
        raise contract.ContractError(
            "Phase 6 r3 import-discovery output must be strictly absent"
        )
    environment = os.environ.copy()
    environment.update(contract.PHASE6_ENVIRONMENT)
    completed = subprocess.run(
        list(contract.PHASE6_IMPORT_DISCOVERY_ARGV),
        cwd=REPO_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if completed.returncode != 0:
        raise contract.ContractError(
            "Phase 6 import discovery failed: " + completed.stderr[-4000:]
        )
    payload = contract.read_strict_json(output_path)
    if payload.get("command_argv") != contract.PHASE6_IMPORT_DISCOVERY_ARGV:
        raise contract.ContractError("Phase 6 import discovery command drifted")
    return payload


def phase6_build_budget_proposal(
    gate: str,
    dependency_discovery: Mapping[str, Any],
) -> dict[str, Any]:
    plan_path = REPO_ROOT / contract.PHASE6_PLAN_RELATIVE
    source_hashes = [
        contract.path_digest_record(REPO_ROOT / relative)
        for relative in contract.PHASE6_REQUIRED_SOURCE_PATHS
    ]
    if gate == "gate_b":
        commands = [_phase6_gate_b_command()]
        schedules = {
            schema: phase6_build_schedule(
                schema,
                gate="gate_b",
                child_timeout_seconds=60,
            )
            for schema in (
                contract.PHASE6_TRACE_SCHEMA,
                contract.PHASE6_PILOT_SCHEMA,
            )
        }
        artifacts = dict(contract.PHASE6_GATE_B_ARTIFACTS)
        budget = {
            "child_execution_deadline_seconds": 60,
            "child_term_grace_seconds": 5,
            "child_kill_reap_grace_seconds": 5,
            "child_lifecycle_cap_seconds": 70,
            "cell_cap_seconds": 160,
            "outer_term_deadline_seconds": 3000,
            "outer_kill_grace_seconds": 45,
            "hard_ceiling_seconds": 3045,
        }
        inputs = contract.phase6_gate_b_input_records(repo_root=REPO_ROOT)
    elif gate == "gate_c":
        trace_path = REPO_ROOT / contract.PHASE6_GATE_B_ARTIFACTS[
            "trace_output_json"
        ]
        pilot_path = REPO_ROOT / contract.PHASE6_GATE_B_ARTIFACTS[
            "pilot_output_json"
        ]
        trace = contract.read_bounded_phase6_trace_json(trace_path)
        pilot = contract.read_strict_json(pilot_path)
        if (
            contract.evaluate_phase6_trace_census(trace)["trace_common_valid"]
            is not True
            or not all(contract.phase6_ledger_checks(pilot, final=True).values())
        ):
            raise contract.ContractError(
                "Gate C proposal requires closed valid Gate B trace and pilot"
            )
        trace_blob = contract.phase6_blob_record(trace_path)
        predecessors = pilot.get("bindings", {}).get("runtime_predecessors", [])
        if len(predecessors) != 1 or predecessors[0].get("artifact") != trace_blob:
            raise contract.ContractError(
                "Gate C proposal requires the pilot's exact trace predecessor"
            )
        commands = _phase6_gate_c_commands()
        schedules = {
            schema: phase6_build_schedule(
                schema,
                gate="gate_c",
                child_timeout_seconds=60,
            )
            for schema in (
                contract.PHASE6_SCALAR_SCHEMA,
                contract.PHASE6_FINAL_SCHEMA,
            )
        }
        artifacts = dict(contract.PHASE6_GATE_C_ARTIFACTS)
        budget = {
            "child_execution_deadline_seconds": 60,
            "child_term_grace_seconds": 5,
            "child_kill_reap_grace_seconds": 5,
            "child_lifecycle_cap_seconds": 70,
            "cell_cap_seconds": 160,
            "outer_term_deadline_seconds": 2700,
            "outer_kill_grace_seconds": 45,
            "hard_ceiling_seconds": 3120,
        }
        inputs = [
            contract.path_digest_record(trace_path),
            contract.path_digest_record(pilot_path),
        ]
    else:
        raise contract.ContractError("unknown Phase 6 proposal gate")
    proposal = {
        "schema": contract.PHASE6_BUDGET_SCHEMA,
        "authority_id": "0" * 64,
        "gate": gate,
        "plan": contract.path_digest_record(plan_path),
        "opening_hash_ledger": contract.phase6_opening_hash_ledger_record(
            Path(contract.PHASE6_OPENING_HASH_LEDGER)
        ),
        "dependency_discovery": copy.deepcopy(dict(dependency_discovery)),
        "source_hashes": source_hashes,
        "commands": commands,
        "schedules": schedules,
        "artifacts": artifacts,
        "budget": budget,
        "inputs": inputs,
        "nonclaims": list(contract.PHASE6_NONCLAIMS),
    }
    identity = {
        key: proposal[key]
        for key in contract.PHASE6_BUDGET_FIELDS
        if key != "authority_id"
    }
    proposal["authority_id"] = contract.canonical_sha256(identity)
    contract.validate_phase6_budget_proposal(proposal, expected_gate=gate)
    return proposal


def _phase6_assert_gate_b_work_root_absent() -> None:
    if PHASE6_WORK_DIR.exists() or PHASE6_WORK_DIR.is_symlink():
        raise contract.ContractError(
            "Phase 6 r3 work root must be strictly absent before proposal construction"
        )


def run_phase6_prepare_proposal(args: argparse.Namespace) -> int:
    gate = args.phase6_prepare_proposal
    expected_relative = (
        contract.PHASE6_GATE_B_BUDGET_RELATIVE
        if gate == "gate_b"
        else contract.PHASE6_GATE_C_BUDGET_RELATIVE
    )
    expected_argv = [
        "--phase6-prepare-proposal",
        gate,
        "--output-json",
        expected_relative,
    ]
    if sys.argv[1:] != expected_argv or args.output_json != Path(expected_relative):
        raise contract.ContractError(
            "Phase 6 proposal construction requires its exact closed invocation"
        )
    if gate == "gate_b":
        _phase6_assert_gate_b_work_root_absent()
    output_path = _phase6_repo_path(args.output_json)
    if output_path.exists() or output_path.is_symlink():
        raise contract.ContractError("Phase 6 r3 proposal output must be strictly absent")
    discovery = _phase6_run_import_discovery()
    proposal = phase6_build_budget_proposal(gate, discovery)
    contract.durable_atomic_write_json(output_path, proposal)
    reparsed = contract.read_strict_json(output_path)
    if reparsed != proposal:
        raise contract.ContractError("Phase 6 proposal failed durable reparse")
    contract.validate_phase6_budget_proposal(reparsed, expected_gate=gate)
    return 0


def _phase6_review_strength(path: Path) -> str:
    _, strength, agree = contract._review_declarations(path)
    if not agree or strength not in {"claude_opus_max", "codex_substitute_weaker"}:
        raise contract.ContractError("Phase 6 review lacks a closed agreeing verdict")
    return strength


def run_phase6_create_attestation(args: argparse.Namespace) -> int:
    gate = args.phase6_create_attestation
    expected_argv = [
        "--phase6-create-attestation",
        gate,
        "--budget-contract",
        contract.PHASE6_GATE_B_BUDGET_RELATIVE,
        "--review-path",
        contract.PHASE6_GATE_B_REVIEW_RELATIVE,
        "--output-json",
        contract.PHASE6_GATE_B_ATTESTATION_RELATIVE,
    ]
    if (
        sys.argv[1:] != expected_argv
        or args.budget_contract != Path(contract.PHASE6_GATE_B_BUDGET_RELATIVE)
        or args.review_path != Path(contract.PHASE6_GATE_B_REVIEW_RELATIVE)
        or args.output_json != Path(contract.PHASE6_GATE_B_ATTESTATION_RELATIVE)
    ):
        raise contract.ContractError(
            "Phase 6 attestation creation requires its exact closed invocation"
        )
    proposal_path = _phase6_resolve_authority_path(
        args.budget_contract, "Gate B proposal"
    )
    review_path = _phase6_resolve_authority_path(args.review_path, "Gate B review")
    output_path = _phase6_repo_path(args.output_json)
    if output_path.exists() or output_path.is_symlink():
        raise contract.ContractError("Phase 6 r3 attestation output must be strictly absent")
    proposal = contract.read_strict_json(proposal_path)
    contract.validate_phase6_budget_proposal(proposal, expected_gate="gate_b")
    strength = _phase6_review_strength(review_path)
    attestation = {
        "schema": contract.PHASE6_ATTESTATION_SCHEMA,
        "authority_id": proposal["authority_id"],
        "gate": "gate_b",
        "proposal": contract.path_digest_record(proposal_path),
        "plan": proposal["plan"],
        "review": contract.path_digest_record(review_path),
        "verdict": "AGREE",
        "review_strength": strength,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    checks = contract.phase6_attestation_checks(
        attestation, proposal_path=proposal_path, expected_gate="gate_b"
    )
    if not all(checks.values()):
        raise contract.ContractError(f"invalid Phase 6 r3 attestation: {checks}")
    contract.durable_atomic_write_json(output_path, attestation)
    contract.validate_phase6_runtime_authority(
        proposal_path, output_path, expected_gate="gate_b"
    )
    return 0


def run_phase6_validate_authority(args: argparse.Namespace) -> int:
    gate = args.phase6_validate_authority
    expected_argv = [
        "--phase6-validate-authority",
        gate,
        "--budget-contract",
        contract.PHASE6_GATE_B_BUDGET_RELATIVE,
        "--budget-attestation",
        contract.PHASE6_GATE_B_ATTESTATION_RELATIVE,
    ]
    if (
        sys.argv[1:] != expected_argv
        or args.budget_contract != Path(contract.PHASE6_GATE_B_BUDGET_RELATIVE)
        or args.budget_attestation != Path(contract.PHASE6_GATE_B_ATTESTATION_RELATIVE)
    ):
        raise contract.ContractError(
            "Phase 6 authority validation requires its exact closed invocation"
        )
    proposal_path = _phase6_resolve_authority_path(
        args.budget_contract, "Gate B proposal"
    )
    attestation_path = _phase6_resolve_authority_path(
        args.budget_attestation, "Gate B attestation"
    )
    proposal, _ = contract.validate_phase6_runtime_authority(
        proposal_path, attestation_path, expected_gate="gate_b"
    )
    _phase6_assert_fresh_gate_b_namespace(
        proposal,
        trace_output=REPO_ROOT / contract.PHASE6_GATE_B_ARTIFACTS["trace_output_json"],
        pilot_output=REPO_ROOT / contract.PHASE6_GATE_B_ARTIFACTS["pilot_output_json"],
    )
    return 0


def _phase6_bindings_for_gate(
    *,
    gate: str,
    schema: str,
    proposal_path: Path,
    attestation_path: Path,
    child_timeout_seconds: float,
    authority_input_paths: Sequence[Path] = (),
    runtime_predecessor_paths: Sequence[Path] = (),
) -> dict[str, Any]:
    proposal, _ = contract.validate_phase6_runtime_authority(
        proposal_path, attestation_path, expected_gate=gate
    )
    schedule = proposal["schedules"].get(schema)
    if not isinstance(schedule, Mapping) or not all(
        contract.phase6_schedule_checks(schedule).values()
    ):
        raise contract.ContractError("reviewed Phase 6 schedule is absent or invalid")
    current_source = contract.source_manifest(REPO_ROOT, include_supervisor=True)[
        "source_fingerprint"
    ]
    current_runtime = contract.runtime_manifest()["runtime_fingerprint"]
    if any(
        row["fingerprints"]["source_fingerprint"] != current_source
        or row["fingerprints"]["runtime_fingerprint"] != current_runtime
        or row["config"]["subprocess_timeout_seconds"] != child_timeout_seconds
        for row in schedule["records"]
    ):
        raise contract.ContractError("reviewed Phase 6 schedule drifted from runtime")
    if gate == "gate_b":
        if authority_input_paths:
            raise contract.ContractError(
                "Gate B authority inputs are derived only from the reviewed proposal"
            )
        authority_input_paths = tuple(
            Path(record["path"]) for record in proposal["inputs"]
        )
    return phase6_build_bindings(
        proposal_path=proposal_path,
        attestation_path=attestation_path,
        schedule=schedule,
        phase45_paths=(PHASE4_EVIDENCE_PATH, PHASE5_EVIDENCE_PATH),
        authority_input_paths=authority_input_paths,
        runtime_predecessor_paths=runtime_predecessor_paths,
    )


def _phase6_write_pruned_pilot(
    *,
    output_path: Path,
    bindings: Mapping[str, Any],
) -> dict[str, Any]:
    roster = contract.phase6_expected_roster(contract.PHASE6_PILOT_SCHEMA)
    payload = contract.new_phase6_ledger(
        schema=contract.PHASE6_PILOT_SCHEMA,
        gate="gate_b",
        artifact_kind="cpu_xla_pilot",
        identities=roster,
        bindings=bindings,
    )
    payload = phase6_persist_and_validate(output_path, payload, final=False)
    for identity in roster:
        payload = contract.transition_phase6_record(
            payload,
            identity_id=identity["identity_id"],
            new_state="not_launched:trace_gate_not_passed",
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            reason="trace_gate_not_passed",
        )
        payload = phase6_persist_and_validate(output_path, payload, final=False)
    payload = contract.finalize_phase6_ledger(payload)
    return phase6_persist_and_validate(output_path, payload, final=True)


def _phase6_assert_fresh_gate_b_namespace(
    proposal: Mapping[str, Any],
    *,
    trace_output: Path,
    pilot_output: Path,
) -> None:
    if _phase6_live_target_pids():
        raise contract.ContractError("another Phase 6 target worker is live")
    for path in (trace_output, pilot_output):
        if path.exists() or path.is_symlink():
            raise contract.ContractError(
                f"Phase 6 r3 runtime output must be strictly absent: {path}"
            )
    root = PHASE6_WORK_DIR
    discovery_path = Path(contract.PHASE6_IMPORT_DISCOVERY_OUTPUT)
    if (
        root.is_symlink()
        or not root.is_dir()
        or discovery_path.is_symlink()
        or not discovery_path.is_file()
        or contract.read_strict_json(discovery_path)
        != proposal.get("dependency_discovery")
    ):
        raise contract.ContractError(
            "Phase 6 r3 import-discovery namespace does not match authority"
        )
    entries = list(root.iterdir())
    if entries != [discovery_path]:
        raise contract.ContractError(
            "Phase 6 r3 work root contains unexpected prelaunch state"
        )


def run_phase6_pilot(args: argparse.Namespace) -> int:
    _phase6_validate_cli_common(args)
    exact_argv = _phase6_exact_supervisor_argv()
    reviewed_argv = _phase6_gate_b_command()["argv"]
    if (
        exact_argv != reviewed_argv
        or args.dimensions != [10, 20, 30]
        or args.parameter_counts != [50, 150]
        or args.batch_sizes != [1, 4, 16]
        or not args.jit_compile
        or args.trace_child_timeout_seconds != 60
        or args.xla_child_timeout_seconds != 60
        or args.xla_cell_timeout_seconds != 160
    ):
        raise contract.ContractError("Phase 6 Gate B CLI identity mismatch")
    proposal_path = _phase6_resolve_authority_path(args.budget_contract, "Gate B proposal")
    attestation_path = _phase6_resolve_authority_path(
        args.budget_attestation, "Gate B attestation"
    )
    trace_output = _phase6_repo_path(args.trace_output_json)
    pilot_output = _phase6_repo_path(args.output_json)
    proposal, _ = contract.validate_phase6_runtime_authority(
        proposal_path,
        attestation_path,
        expected_gate="gate_b",
    )
    if proposal["commands"][0]["argv"] != exact_argv:
        raise contract.ContractError("Phase 6 Gate B runtime argv is not proposal-bound")
    _phase6_assert_fresh_gate_b_namespace(
        proposal,
        trace_output=trace_output,
        pilot_output=pilot_output,
    )
    trace_bindings = _phase6_bindings_for_gate(
        gate="gate_b",
        schema=contract.PHASE6_TRACE_SCHEMA,
        proposal_path=proposal_path,
        attestation_path=attestation_path,
        child_timeout_seconds=args.trace_child_timeout_seconds,
    )
    phase6_revalidate_launch_authority(trace_bindings)
    trace_initial_payload = contract.new_phase6_ledger(
        schema=contract.PHASE6_TRACE_SCHEMA,
        gate="gate_b",
        artifact_kind="trace_census",
        identities=contract.phase6_expected_roster(contract.PHASE6_TRACE_SCHEMA),
        bindings=trace_bindings,
    )
    if not all(
        contract.phase6_ledger_checks(trace_initial_payload, final=False).values()
    ):
        raise contract.ContractError("Phase 6 trace initial ledger failed pre-budget validation")
    pilot_schedule = proposal["schedules"].get(contract.PHASE6_PILOT_SCHEMA)
    if not isinstance(pilot_schedule, Mapping) or not all(
        contract.phase6_schedule_checks(pilot_schedule).values()
    ):
        raise contract.ContractError("Phase 6 pilot schedule failed pre-budget validation")
    budget = proposal["budget"]
    budget_path = _phase6_budget_state_path(
        "gate_b", proposal["authority_id"]
    )
    with phase6_budget_lease(budget_path, "trace_census_and_pilot") as budget_lease:
        budget_state = phase6_budget_state_open(
            budget_path,
            proposal["authority_id"],
            "gate_b",
            budget["hard_ceiling_seconds"],
            "trace_census_and_pilot",
            lease=budget_lease,
        )
        command_completed = False
        try:
            trace_payload = phase6_execute_ledger(
                schema=contract.PHASE6_TRACE_SCHEMA,
                output_path=trace_output,
                bindings=trace_bindings,
                child_timeout_seconds=args.trace_child_timeout_seconds,
                authority_validator=phase6_revalidate_launch_authority,
                budget_path=budget_path,
                budget_lease=budget_lease,
                expected_budget_command_name="trace_census_and_pilot",
                gate_hard_ceiling_seconds=budget["hard_ceiling_seconds"],
                initial_payload=trace_initial_payload,
            )
            trace_evaluation = contract.evaluate_phase6_trace_census(trace_payload)
            trace_summary = contract.phase6_terminal_summary(trace_payload)
            if trace_summary["has_common_invalidity"]:
                command_completed = True
                return 1
            pilot_bindings = _phase6_bindings_for_gate(
                gate="gate_b",
                schema=contract.PHASE6_PILOT_SCHEMA,
                proposal_path=proposal_path,
                attestation_path=attestation_path,
                child_timeout_seconds=args.xla_child_timeout_seconds,
                runtime_predecessor_paths=(trace_output,),
            )
            if trace_evaluation["trace_common_valid"] is not True:
                phase6_revalidate_launch_authority(pilot_bindings)
                _phase6_write_pruned_pilot(
                    output_path=pilot_output, bindings=pilot_bindings
                )
                command_completed = True
                return 1
            phase6_revalidate_launch_authority(pilot_bindings)
            pilot_payload = phase6_execute_ledger(
                schema=contract.PHASE6_PILOT_SCHEMA,
                output_path=pilot_output,
                bindings=pilot_bindings,
                child_timeout_seconds=args.xla_child_timeout_seconds,
                authority_validator=phase6_revalidate_launch_authority,
                budget_path=budget_path,
                budget_lease=budget_lease,
                expected_budget_command_name="trace_census_and_pilot",
                gate_hard_ceiling_seconds=budget["hard_ceiling_seconds"],
                cell_cap_seconds=budget["cell_cap_seconds"],
            )
            command_completed = True
            return 0 if pilot_payload["state"] == "passed" else 1
        finally:
            if command_completed:
                phase6_budget_state_close_command(
                    budget_path,
                    budget_state,
                    "trace_census_and_pilot",
                    lease=budget_lease,
                )


def run_phase6_scalar_references(args: argparse.Namespace) -> int:
    _phase6_validate_cli_common(args)
    if (
        args.dimensions != [10]
        or args.parameter_counts != [50]
        or args.batch_sizes != [1, 4]
        or args.jit_compile
        or args.child_timeout_seconds != 60
    ):
        raise contract.ContractError("Phase 6 Gate C scalar CLI identity mismatch")
    proposal_path = _phase6_resolve_authority_path(args.budget_contract, "Gate C proposal")
    attestation_path = _phase6_resolve_authority_path(
        args.budget_attestation, "Gate C attestation"
    )
    proposal, _ = contract.validate_phase6_runtime_authority(
        proposal_path,
        attestation_path,
        expected_gate="gate_c",
    )
    budget = proposal["budget"]
    budget_path = _phase6_budget_state_path(
        "gate_c", proposal["authority_id"]
    )
    with phase6_budget_lease(budget_path, "scalar_references") as budget_lease:
        budget_state = phase6_budget_state_open(
            budget_path,
            proposal["authority_id"],
            "gate_c",
            budget["hard_ceiling_seconds"],
            "scalar_references",
            lease=budget_lease,
        )
        command_completed = False
        try:
            bindings = _phase6_bindings_for_gate(
                gate="gate_c",
                schema=contract.PHASE6_SCALAR_SCHEMA,
                proposal_path=proposal_path,
                attestation_path=attestation_path,
                child_timeout_seconds=args.child_timeout_seconds,
                authority_input_paths=(
                    REPO_ROOT / contract.PHASE6_GATE_B_ARTIFACTS["trace_output_json"],
                    REPO_ROOT / contract.PHASE6_GATE_B_ARTIFACTS["pilot_output_json"],
                ),
            )
            phase6_execute_ledger(
                schema=contract.PHASE6_SCALAR_SCHEMA,
                output_path=_phase6_repo_path(args.output_json),
                bindings=bindings,
                child_timeout_seconds=args.child_timeout_seconds,
                authority_validator=phase6_revalidate_launch_authority,
                budget_path=budget_path,
                budget_lease=budget_lease,
                expected_budget_command_name="scalar_references",
                gate_hard_ceiling_seconds=budget["hard_ceiling_seconds"],
                cell_cap_seconds=budget["cell_cap_seconds"],
            )
            phase6_revalidate_launch_authority(bindings)
            # A valid nonpass scalar artifact is evidence for the already-reviewed
            # remaining command, not an operational veto on creating that artifact.
            command_completed = True
            return 0
        finally:
            if command_completed:
                phase6_budget_state_close_command(
                    budget_path,
                    budget_state,
                    "scalar_references",
                    lease=budget_lease,
                )


def run_phase6_remaining(args: argparse.Namespace) -> int:
    _phase6_validate_cli_common(args)
    if (
        args.dimensions != [10, 20, 30]
        or args.parameter_counts != [50, 150]
        or args.batch_sizes != [1, 4, 16]
        or not args.jit_compile
        or args.child_timeout_seconds != 60
        or args.cell_timeout_seconds != 160
    ):
        raise contract.ContractError("Phase 6 Gate C remaining CLI identity mismatch")
    proposal_path = _phase6_resolve_authority_path(args.budget_contract, "Gate C proposal")
    attestation_path = _phase6_resolve_authority_path(
        args.budget_attestation, "Gate C attestation"
    )
    proposal, _ = contract.validate_phase6_runtime_authority(
        proposal_path,
        attestation_path,
        expected_gate="gate_c",
    )
    budget = proposal["budget"]
    budget_path = _phase6_budget_state_path(
        "gate_c", proposal["authority_id"]
    )
    with phase6_budget_lease(budget_path, "remaining_lattice") as budget_lease:
        budget_state = phase6_budget_state_open(
            budget_path,
            proposal["authority_id"],
            "gate_c",
            budget["hard_ceiling_seconds"],
            "remaining_lattice",
            lease=budget_lease,
        )
        command_completed = False
        try:
            trace_input = _phase6_resolve_authority_path(args.trace_input, "trace input")
            pilot_input = _phase6_resolve_authority_path(args.pilot_input, "pilot input")
            scalar_input = _phase6_resolve_authority_path(
                args.scalar_reference_input, "scalar input"
            )
            trace_payload = contract.read_bounded_phase6_trace_json(trace_input)
            if (
                contract.evaluate_phase6_trace_census(trace_payload)[
                    "trace_common_valid"
                ]
                is not True
            ):
                raise contract.ContractError(
                    "Phase 6 Gate C requires a valid trace census"
                )
            pilot_payload = contract.read_strict_json(pilot_input)
            scalar_payload = contract.read_strict_json(scalar_input)
            if not all(contract.phase6_ledger_checks(pilot_payload, final=True).values()):
                raise contract.ContractError("Phase 6 Gate C pilot input is invalid")
            if not all(contract.phase6_ledger_checks(scalar_payload, final=True).values()):
                raise contract.ContractError("Phase 6 Gate C scalar input is invalid")
            bindings = _phase6_bindings_for_gate(
                gate="gate_c",
                schema=contract.PHASE6_FINAL_SCHEMA,
                proposal_path=proposal_path,
                attestation_path=attestation_path,
                child_timeout_seconds=args.child_timeout_seconds,
                authority_input_paths=(trace_input, pilot_input),
                runtime_predecessor_paths=(scalar_input,),
            )
            imported = {
                record["identity"]["identity_id"]: record
                for record in pilot_payload["records"]
            }
            final_payload = phase6_execute_ledger(
                schema=contract.PHASE6_FINAL_SCHEMA,
                output_path=_phase6_repo_path(args.output_json),
                bindings=bindings,
                child_timeout_seconds=args.child_timeout_seconds,
                imported_records=imported,
                authority_validator=phase6_revalidate_launch_authority,
                budget_path=budget_path,
                budget_lease=budget_lease,
                expected_budget_command_name="remaining_lattice",
                gate_hard_ceiling_seconds=budget["hard_ceiling_seconds"],
                cell_cap_seconds=budget["cell_cap_seconds"],
                routing_path=_phase6_repo_path(args.routing_output_json),
            )
            phase6_revalidate_launch_authority(bindings)
            handoff = contract.evaluate_phase6_handoff(final_payload)
            command_completed = True
            return 0 if handoff["phase7_scope"] != "blocked" else 1
        finally:
            if command_completed:
                phase6_budget_state_close_command(
                    budget_path,
                    budget_state,
                    "remaining_lattice",
                    lease=budget_lease,
                )


def run_phase6_evaluate(args: argparse.Namespace) -> int:
    if args.phase6_input is None:
        raise contract.ContractError("--phase6-evaluate requires --phase6-input")
    path = _phase6_repo_path(args.phase6_input)
    payload = (
        contract.read_bounded_phase6_trace_json(path)
        if "trace_census" in path.name
        else contract.read_strict_json(path)
    )
    if payload.get("schema") == contract.PHASE6_TRACE_SCHEMA:
        result = contract.evaluate_phase6_trace_census(payload)
        passed = result["trace_common_valid"] is True
    elif payload.get("schema") == contract.PHASE6_FINAL_SCHEMA:
        if args.routing_input is None:
            result = {"phase7_scope": "blocked", "reason": "routing_input_required"}
            passed = False
        else:
            routing = contract.read_strict_json(_phase6_repo_path(args.routing_input))
            routing_checks = phase6_final_routing_checks(payload, routing)
            result = {
                "routing_checks": routing_checks,
                "handoff": contract.evaluate_phase6_handoff(payload),
            }
            passed = all(routing_checks.values()) and result["handoff"][
                "phase7_scope"
            ] != "blocked"
    else:
        checks = contract.phase6_ledger_checks(payload, final=True)
        result = {"checks": checks, "state": payload.get("state")}
        passed = all(checks.values())
    print(contract.strict_json_dumps(result, indent=2))
    return 0 if passed else 1


def _execution_contract(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.resolve()
    return {
        "command_argv": [
            str(Path(sys.executable).resolve()),
            str(Path(sys.argv[0]).resolve()),
            *sys.argv[1:],
        ],
        "cwd": str(REPO_ROOT),
        "git_commit": _git_commit(),
        "output_dir": str(output_dir),
        "schedule_path": str(output_dir / "schedule.json"),
        "status_path": str(output_dir / "status.json"),
        "external_log_path": str(output_dir / "smoke.log"),
        "plan_path": getattr(args, "plan_path", PLAN_PATH),
        "result_path": getattr(args, "result_path", RESULT_PATH),
        "dimensions": list(args.dimensions),
        "parameter_counts": list(args.parameter_counts),
        "timesteps": args.timesteps,
        "batch_size": args.batch_size,
        "dtype": args.dtype,
        "device": args.device,
        "cpu_threads": args.cpu_threads,
        "repeats": args.repeats,
        "timeout_seconds": args.timeout_seconds,
        "methods": list(args.methods),
        "no_resume": args.no_resume,
        "jit_compile": args.jit_compile,
        "xla_execution": "xla_jit_enabled" if args.jit_compile else "xla_disabled_debug",
        "xla_flags": os.environ.get("XLA_FLAGS", "UNSET"),
        "tf32_enabled": args.tf32_enabled,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "UNSET"),
        "thread_environment": {
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "UNSET"),
            "TF_NUM_INTRAOP_THREADS": os.environ.get("TF_NUM_INTRAOP_THREADS", "UNSET"),
            "TF_NUM_INTEROP_THREADS": os.environ.get("TF_NUM_INTEROP_THREADS", "UNSET"),
        },
        "timing_boundary_version": TIMING_BOUNDARY_VERSION,
        "trust_basis": "gpu_hidden_cpu_debug_reference",
        "outer_timeout_seconds": 210.0,
        "outer_timeout_scope": "declared_shell_emergency_cap_not_observed_by_process",
        "nonclaims": list(PHASE5_NONCLAIMS),
    }


def _config_payload(args: argparse.Namespace, *, dimension: int, parameter_count: int, method_id: str) -> dict[str, Any]:
    return {
        "method_id": method_id,
        "method_contract_version": contract.METHOD_CONTRACT_VERSION,
        "dimension": dimension,
        "parameter_count": parameter_count,
        "timesteps": args.timesteps,
        "batch_size": args.batch_size,
        "dtype": args.dtype,
        "device": args.device,
        "jit_compile": args.jit_compile,
        "cpu_threads": args.cpu_threads if args.device == "cpu" else None,
        "repeats": args.repeats,
        "subprocess_timeout_seconds": args.timeout_seconds,
        "xla_flags": os.environ.get("XLA_FLAGS", "UNSET"),
        "tf32_enabled": args.tf32_enabled,
        "jitter": 1.0e-9,
        "jitter_updates_filtered_covariance": True,
        "fixture_contract_version": FIXTURE_CONTRACT_VERSION,
        "timing_boundary_version": TIMING_BOUNDARY_VERSION,
        "method_options": {},
    }


def _fixture_payload(args: argparse.Namespace, *, dimension: int, parameter_count: int) -> dict[str, Any]:
    return {
        "fixture_contract_version": FIXTURE_CONTRACT_VERSION,
        "randomness": "deterministic",
        "seed": None,
        "dimension": dimension,
        "parameter_count": parameter_count,
        "timesteps": args.timesteps,
        "batch_size": args.batch_size,
        "dtype": args.dtype,
        "parameter_batch_version": PARAMETER_BATCH_VERSION,
        "observation_generation_version": OBSERVATION_GENERATION_VERSION,
        "external_input_hashes": {},
    }


def _base_identity_rows(
    args: argparse.Namespace,
    *,
    source_fingerprint: str,
    runtime_fingerprint: str,
) -> list[dict[str, str]]:
    identities = []
    for dimension in args.dimensions:
        for parameter_count in args.parameter_counts:
            if parameter_count <= 0 or dimension <= 0:
                raise contract.ContractError("dimensions and parameter counts must be positive")
            for method_id in args.methods:
                config = contract.config_manifest(
                    _config_payload(
                        args,
                        dimension=dimension,
                        parameter_count=parameter_count,
                        method_id=method_id,
                    )
                )
                fixture = contract.fixture_manifest(
                    _fixture_payload(args, dimension=dimension, parameter_count=parameter_count)
                )
                identity = {
                    "case_id": contract.case_id(config["config"]),
                    "method_id": method_id,
                    "source_fingerprint": source_fingerprint,
                    "config_fingerprint": config["config_fingerprint"],
                    "runtime_fingerprint": runtime_fingerprint,
                    "fixture_fingerprint": fixture["fixture_fingerprint"],
                }
                identities.append(identity)
    return identities


def build_schedule(args: argparse.Namespace) -> dict[str, Any]:
    source = contract.source_manifest(REPO_ROOT, include_supervisor=True)
    runtime = contract.runtime_manifest()
    identities = _base_identity_rows(
        args,
        source_fingerprint=source["source_fingerprint"],
        runtime_fingerprint=runtime["runtime_fingerprint"],
    )
    if args.harness_contract_test_only:
        checks = contract.HARNESS_ONLY_CHECKS
    elif set(contract.PRIMARY_METHOD_IDS).issubset(args.methods):
        checks = contract.PRIMARY_PAIR_CHECKS
    else:
        checks = contract.METHOD_LOCAL_CHECKS
    schedule = contract.build_schedule_manifest(
        identities,
        checks,
        harness_contract_test_only=args.harness_contract_test_only,
    )
    schedule["source_manifest"] = source
    schedule["runtime_manifest"] = runtime
    schedule["timing_boundary_version"] = TIMING_BOUNDARY_VERSION
    schedule["created_utc"] = datetime.now(timezone.utc).isoformat()
    schedule["plan_path"] = getattr(args, "plan_path", PLAN_PATH)
    schedule["nonclaims"] = [
        "method-local viability is not comparison completeness",
        "Phase 4 comparator correctness is not timing evidence",
        "no XLA viability or runtime ranking claim",
        "no GPU readiness claim",
        "no HMC, posterior, default, production, or scientific claim",
    ]
    return schedule


def _fingerprints(identity: Mapping[str, str], schedule_fingerprint: str) -> dict[str, str]:
    return {
        **{field: identity[field] for field in contract.FINGERPRINT_FIELDS[:-1]},
        "schedule_fingerprint": schedule_fingerprint,
    }


def method_artifact_path(output_dir: Path, identity: Mapping[str, str]) -> Path:
    digest = contract.canonical_sha256(
        {"case_id": identity["case_id"], "method_id": identity["method_id"]}
    )[:20]
    return output_dir / "methods" / f"{digest}.json"


def method_markdown_path(output_dir: Path, identity: Mapping[str, str]) -> Path:
    return method_artifact_path(output_dir, identity).with_suffix(".md")


def _child_command(
    args: argparse.Namespace,
    *,
    identity: Mapping[str, str],
    attempt_id: str,
    progress_path: Path,
    output_json: Path,
    output_md: Path,
    schedule_fingerprint: str,
) -> list[str]:
    config = _config_payload(
        args,
        dimension=int(identity["case_id"].split("dimension=", 1)[1].split("-", 1)[0]),
        parameter_count=int(identity["case_id"].split("parameter_count=", 1)[1].split("-", 1)[0]),
        method_id=identity["method_id"],
    )
    fingerprints = _fingerprints(identity, schedule_fingerprint)
    key = contract.resume_key(
        case_identity=identity["case_id"],
        method_id=identity["method_id"],
        fingerprints=fingerprints,
    )
    command = [
        str(PYTHON),
        str(BENCHMARK),
        "--dimensions",
        str(config["dimension"]),
        "--parameter-counts",
        str(config["parameter_count"]),
        "--timesteps",
        str(config["timesteps"]),
        "--batch-size",
        str(config["batch_size"]),
        "--dtype",
        str(config["dtype"]),
        "--device",
        str(config["device"]),
        "--repeats",
        str(config["repeats"]),
        "--method",
        identity["method_id"],
        "--case-id",
        identity["case_id"],
        "--attempt-id",
        attempt_id,
        "--progress-journal",
        str(progress_path),
        "--source-fingerprint",
        fingerprints["source_fingerprint"],
        "--config-fingerprint",
        fingerprints["config_fingerprint"],
        "--runtime-fingerprint",
        fingerprints["runtime_fingerprint"],
        "--fixture-fingerprint",
        fingerprints["fixture_fingerprint"],
        "--schedule-fingerprint",
        fingerprints["schedule_fingerprint"],
        "--resume-key",
        key,
        "--plan-path",
        getattr(args, "plan_path", PLAN_PATH),
        "--output-json",
        str(output_json),
        "--output-md",
        str(output_md),
        "--jit-compile" if args.jit_compile else "--no-jit-compile",
        "--tf32-enabled" if args.tf32_enabled else "--no-tf32",
    ]
    if args.device == "cpu":
        command.extend(["--cpu-threads", str(args.cpu_threads)])
    return command


def expected_progress_identity(
    identity: Mapping[str, str], attempt_id: str, schedule_fingerprint: str
) -> dict[str, str]:
    fingerprints = _fingerprints(identity, schedule_fingerprint)
    return {
        "attempt_id": attempt_id,
        "case_id": identity["case_id"],
        "method_id": identity["method_id"],
        **fingerprints,
        "resume_key": contract.resume_key(
            case_identity=identity["case_id"],
            method_id=identity["method_id"],
            fingerprints=fingerprints,
        ),
    }


def _read_reusable_record(
    path: Path,
    *,
    identity: Mapping[str, str],
    schedule_fingerprint: str,
) -> tuple[dict[str, Any] | None, str]:
    if not path.exists():
        return None, "artifact_missing"
    try:
        record = contract.read_strict_json(path)
    except contract.ContractError as exc:
        return None, f"strict_json_rejected:{exc}"
    fingerprints = _fingerprints(identity, schedule_fingerprint)
    reusable, reason = contract.method_record_reuse_decision(
        record,
        expected_case_id=identity["case_id"],
        expected_method_id=identity["method_id"],
        expected_fingerprints=fingerprints,
        expected_resume_key=contract.resume_key(
            case_identity=identity["case_id"],
            method_id=identity["method_id"],
            fingerprints=fingerprints,
        ),
    )
    if reusable and not contract.payload_sidecar_matches_record(
        record, expected_path=path.with_suffix(".payload.json")
    ):
        return None, "payload_sidecar_invalid"
    return (record if reusable else None), reason


def run_identity(
    args: argparse.Namespace,
    *,
    identity: Mapping[str, str],
    schedule_fingerprint: str,
    progress_dir: Path,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> tuple[dict[str, Any], str]:
    output_json = method_artifact_path(args.output_dir, identity)
    output_md = method_markdown_path(args.output_dir, identity)
    resume_reason = "resume_disabled"
    if not args.no_resume:
        reusable, reason = _read_reusable_record(
            output_json,
            identity=identity,
            schedule_fingerprint=schedule_fingerprint,
        )
        if reusable is not None:
            return reusable, reason
        resume_reason = reason

    attempt_id, progress_path = contract.new_attempt(
        progress_dir, identity["case_id"], identity["method_id"]
    )
    progress_identity = expected_progress_identity(identity, attempt_id, schedule_fingerprint)
    command = _child_command(
        args,
        identity=identity,
        attempt_id=attempt_id,
        progress_path=progress_path,
        output_json=output_json,
        output_md=output_md,
        schedule_fingerprint=schedule_fingerprint,
    )
    env = os.environ.copy()
    if args.device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = "-1"
        env["OMP_NUM_THREADS"] = str(args.cpu_threads)
        env["TF_NUM_INTRAOP_THREADS"] = str(args.cpu_threads)
        env["TF_NUM_INTEROP_THREADS"] = str(args.cpu_threads)
    else:
        env["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    started = time.perf_counter()
    try:
        completed = runner(
            command,
            cwd=REPO_ROOT,
            env=env,
            check=False,
            capture_output=True,
            text=True,
            timeout=args.timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        record = contract.synthesize_process_record(
            identity=progress_identity,
            progress_path=progress_path,
            timed_out=True,
            returncode=None,
            error_tail=(exc.stderr or "") if isinstance(exc.stderr, str) else "",
        )
        record["elapsed_seconds"] = time.perf_counter() - started
        return record, f"resume_rejected:{resume_reason};executed_timeout"

    if output_json.exists():
        try:
            record = contract.read_strict_json(output_json)
        except contract.ContractError:
            record = None
        child_state_matches_returncode = (
            isinstance(record, dict)
            and (
                (record.get("state") == "passed" and completed.returncode == 0)
                or (record.get("state") != "passed" and completed.returncode != 0)
            )
        )
        if (
            isinstance(record, dict)
            and record.get("attempt_id") == attempt_id
            and all(record.get(key) == value for key, value in progress_identity.items())
            and record.get("state") in contract.METHOD_TERMINAL_STATES
            and child_state_matches_returncode
        ):
            record["child_returncode"] = completed.returncode
            record["elapsed_seconds"] = time.perf_counter() - started
            record["stdout_tail"] = completed.stdout[-4000:]
            record["stderr_tail"] = completed.stderr[-4000:]
            return record, f"resume_rejected:{resume_reason};executed_child_record"

    record = contract.synthesize_process_record(
        identity=progress_identity,
        progress_path=progress_path,
        timed_out=False,
        returncode=completed.returncode,
        error_tail=(completed.stdout + "\n" + completed.stderr)[-4000:],
    )
    record["elapsed_seconds"] = time.perf_counter() - started
    return record, f"resume_rejected:{resume_reason};executed_synthesized_record"


def _aggregate_checks(schedule: Mapping[str, Any], records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    expected = schedule["expected_identities"]
    observed = [(row.get("case_id"), row.get("method_id")) for row in records]
    expected_pairs = [(row["case_id"], row["method_id"]) for row in expected]
    expected_by_pair = {
        (row["case_id"], row["method_id"]): row
        for row in expected
    }

    def record_integrity(record: Mapping[str, Any]) -> bool:
        pair = (record.get("case_id"), record.get("method_id"))
        identity = expected_by_pair.get(pair)
        if identity is None:
            return False
        fingerprints = _fingerprints(identity, schedule["schedule_fingerprint"])
        expected_resume_key = contract.resume_key(
            case_identity=identity["case_id"],
            method_id=identity["method_id"],
            fingerprints=fingerprints,
        )
        invocation_valid = (
            record.get("invoked_method_ids") == [record.get("method_id")]
            if record.get("state") == "passed"
            else record.get("invoked_method_ids") in ([record.get("method_id")], [])
        )
        return (
            record.get("schema") == contract.SCHEMA
            and record.get("method_contract_version") == contract.METHOD_CONTRACT_VERSION
            and record.get("state") in contract.METHOD_TERMINAL_STATES
            and bool(record.get("attempt_id"))
            and all(record.get(field) == value for field, value in fingerprints.items())
            and record.get("resume_key") == expected_resume_key
            and invocation_valid
            and (
                record.get("state") != "passed"
                or contract.measurement_record_is_valid(record)
            )
        )

    checks = {
        "identity_integrity": len(observed) == len(set(observed)) and sorted(observed) == sorted(expected_pairs),
        "record_integrity": all(record_integrity(row) for row in records),
    }
    schedule_is_gpu = all(
        _case_field(str(row.get("case_id", "")), "device") == "gpu"
        for row in expected
    )
    if schedule_is_gpu:
        checks["gpu_memory_growth"] = all(
            isinstance(row.get("gpu_memory_growth_policy"), Mapping)
            and row["gpu_memory_growth_policy"].get("policy")
            == "required_no_full_device_preallocation"
            and row["gpu_memory_growth_policy"].get("environment_variable")
            == "TF_FORCE_GPU_ALLOW_GROWTH"
            and row["gpu_memory_growth_policy"].get("environment_value") == "true"
            and isinstance(row.get("gpu_allocator_memory"), Mapping)
            and row["gpu_allocator_memory"].get("device") == "/GPU:0"
            and type(row["gpu_allocator_memory"].get("current_bytes")) is int
            and row["gpu_allocator_memory"]["current_bytes"] >= 0
            and type(row["gpu_allocator_memory"].get("peak_bytes")) is int
            and row["gpu_allocator_memory"]["peak_bytes"]
            >= row["gpu_allocator_memory"]["current_bytes"]
            for row in records
            if row.get("state") == "passed"
        )
    if schedule["harness_contract_test_only"]:
        return checks
    checks["finite_output_metadata"] = all(
        isinstance(row.get("output_metadata"), Mapping)
        and row["output_metadata"].get("all_finite") is True
        for row in records
    )
    checks["expected_dtype_shape"] = all(_expected_dtype_shape(row) for row in records)
    expected_methods: dict[str, set[str]] = {}
    observed_methods: dict[str, set[str]] = {}
    for case_id, method_id in expected_pairs:
        expected_methods.setdefault(case_id, set()).add(method_id)
    for case_id, method_id in observed:
        if isinstance(case_id, str) and isinstance(method_id, str):
            observed_methods.setdefault(case_id, set()).add(method_id)
    primary_set = set(contract.PRIMARY_METHOD_IDS)
    checks["primary_pair_complete"] = (
        schedule["primary_pair_complete"] is True
        and checks["identity_integrity"]
        and all(primary_set.issubset(methods) for methods in expected_methods.values())
        and expected_methods == observed_methods
    )
    checks["comparator_parity"] = (
        _comparator_parity(records)
        if schedule["comparator_parity_applicable"]
        else None
    )
    return checks


_TOLERANCES = {
    "float32": {
        "value": {"rtol": 2.0e-4, "atol": 2.0e-4},
        "score": {"rtol": 2.0e-4, "atol": 2.0e-4},
    },
    "float64": {
        "value": {"rtol": 1.0e-10, "atol": 1.0e-10},
        "score": {"rtol": 1.0e-8, "atol": 1.0e-9},
    },
}

_COMPARISON_REFERENCES = {
    "batch_native_autodiff_qr_score": "batch_native_analytical_qr_score",
    "scalar_analytical_row_loop": "batch_native_analytical_qr_score",
    "autodiff_row_loop_qr_score": "batch_native_autodiff_qr_score",
}


def _case_field(case_id: str, field: str) -> str | None:
    prefix = f"{field}="
    for token in case_id.split("-"):
        if token.startswith(prefix):
            return token[len(prefix) :]
    return None


def _expected_dtype_shape(record: Mapping[str, Any]) -> bool:
    metadata = record.get("output_metadata")
    case_id = record.get("case_id")
    if not isinstance(metadata, Mapping) or not isinstance(case_id, str):
        return False
    try:
        batch_size = int(_case_field(case_id, "batch_size") or "")
        parameter_count = int(_case_field(case_id, "parameter_count") or "")
    except ValueError:
        return False
    dtype = _case_field(case_id, "dtype")
    return (
        dtype in _TOLERANCES
        and metadata.get("value_dtype") == dtype
        and metadata.get("score_dtype") == dtype
        and metadata.get("value_shape") == [batch_size]
        and metadata.get("score_shape") == [batch_size, parameter_count]
    )


def _same_numeric_shape(candidate: Any, reference: Any) -> bool:
    if isinstance(reference, list):
        return (
            isinstance(candidate, list)
            and len(candidate) == len(reference)
            and all(
                _same_numeric_shape(candidate_item, reference_item)
                for candidate_item, reference_item in zip(candidate, reference, strict=True)
            )
        )
    return (
        not isinstance(candidate, (list, bool))
        and not isinstance(reference, bool)
        and isinstance(candidate, (int, float))
        and isinstance(reference, (int, float))
    )


def _directed_allclose(candidate: Any, reference: Any, *, rtol: float, atol: float) -> bool:
    if not _same_numeric_shape(candidate, reference):
        return False
    if isinstance(reference, list):
        return all(
            _directed_allclose(candidate_item, reference_item, rtol=rtol, atol=atol)
            for candidate_item, reference_item in zip(candidate, reference, strict=True)
        )
    candidate_value = float(candidate)
    reference_value = float(reference)
    return (
        math.isfinite(candidate_value)
        and math.isfinite(reference_value)
        and abs(candidate_value - reference_value)
        <= atol + rtol * abs(reference_value)
    )


def _record_pair_matches(
    candidate: Mapping[str, Any], reference: Mapping[str, Any]
) -> bool:
    candidate_metadata = candidate.get("output_metadata")
    reference_metadata = reference.get("output_metadata")
    candidate_output = candidate.get("outputs")
    reference_output = reference.get("outputs")
    if not all(
        isinstance(value, Mapping)
        for value in (
            candidate_metadata,
            reference_metadata,
            candidate_output,
            reference_output,
        )
    ):
        return False
    dtype = reference_metadata.get("value_dtype")
    if (
        dtype not in _TOLERANCES
        or candidate_metadata.get("value_dtype") != dtype
        or candidate_metadata.get("score_dtype") != dtype
        or reference_metadata.get("score_dtype") != dtype
        or candidate_metadata.get("value_shape") != reference_metadata.get("value_shape")
        or candidate_metadata.get("score_shape") != reference_metadata.get("score_shape")
    ):
        return False
    tolerances = _TOLERANCES[dtype]
    return _directed_allclose(
        candidate_output.get("value"),
        reference_output.get("value"),
        **tolerances["value"],
    ) and _directed_allclose(
        candidate_output.get("score"),
        reference_output.get("score"),
        **tolerances["score"],
    )


def _comparator_parity(records: Sequence[Mapping[str, Any]]) -> bool | None:
    by_case: dict[str, dict[str, Mapping[str, Any]]] = {}
    for record in records:
        case_id = str(record.get("case_id"))
        method_id = str(record.get("method_id"))
        if method_id in by_case.setdefault(case_id, {}):
            return False
        by_case[case_id][method_id] = record
    if not by_case or any(
        not set(contract.PRIMARY_METHOD_IDS).issubset(siblings)
        for siblings in by_case.values()
    ):
        return None
    for siblings in by_case.values():
        for candidate_id, reference_id in _COMPARISON_REFERENCES.items():
            if candidate_id not in siblings:
                continue
            if reference_id not in siblings or not _record_pair_matches(
                siblings[candidate_id], siblings[reference_id]
            ):
                return False
    return True


def _schedule_identity_stable(args: argparse.Namespace, schedule: Mapping[str, Any]) -> bool:
    current = build_schedule(args)
    return (
        current["schedule_fingerprint"] == schedule["schedule_fingerprint"]
        and current["expected_identities"] == schedule["expected_identities"]
        and current["mandatory_aggregate_checks"] == schedule["mandatory_aggregate_checks"]
    )


def execute_schedule(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    schedule = build_schedule(args)
    execution_contract = _execution_contract(args)
    schedule["execution_contract"] = execution_contract
    schedule_path = args.output_dir / "schedule.json"
    status_path = args.output_dir / "status.json"
    contract.atomic_write_json(schedule_path, schedule)

    records = []
    decisions = []
    progress_dir = args.output_dir / "progress"
    interrupted = False
    try:
        for identity in schedule["expected_identities"]:
            if not _schedule_identity_stable(args, schedule):
                raise contract.ContractError("source/config/runtime/fixture/schedule drift before child launch")
            record, decision = run_identity(
                args,
                identity=identity,
                schedule_fingerprint=schedule["schedule_fingerprint"],
                progress_dir=progress_dir,
            )
            records.append(record)
            decisions.append(
                {
                    "case_id": identity["case_id"],
                    "method_id": identity["method_id"],
                    "decision": decision,
                }
            )
    except KeyboardInterrupt:
        interrupted = True
    except Exception as exc:
        payload = {
            "schema": contract.SCHEMA,
            "status": "failed",
            "schedule": schedule,
            "records": records,
            "decisions": decisions,
            "execution_contract": execution_contract,
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }
        contract.atomic_write_json(status_path, payload)
        return payload, 1

    stable_at_close = _schedule_identity_stable(args, schedule)
    checks = _aggregate_checks(schedule, records)
    mandatory_results = {
        name: bool(checks.get(name))
        for name in schedule["mandatory_aggregate_checks"]
    }
    status = contract.classify_top_level_status(
        schedule["expected_identities"],
        records,
        mandatory_results,
        schedule["mandatory_aggregate_checks"],
        interrupted=interrupted,
        structural_failure=(
            not stable_at_close
            or not checks.get("identity_integrity", False)
            or not checks.get("record_integrity", False)
        ),
    )
    payload = {
        "schema": contract.SCHEMA,
        "status": status,
        "schedule": schedule,
        "records": records,
        "execution_contract": execution_contract,
        "resume_decisions": decisions,
        "aggregate_checks": checks,
        "comparison_summary": {
            "mode": schedule["comparison_mode"],
            "comparison_complete": bool(
                schedule["primary_pair_complete"]
                and checks.get("comparator_parity") is True
            ),
            "primary_pair_complete": schedule["primary_pair_complete"],
            "comparator_parity_applicable": schedule["comparator_parity_applicable"],
            "comparator_parity": checks.get("comparator_parity"),
            "reason": schedule["comparator_parity_reason"],
        },
        "schedule_identity_stable_at_close": stable_at_close,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
    }
    contract.atomic_write_json(status_path, payload)
    return payload, contract.exit_code_for_status(status)


def _phase5_expected_execution(input_path: Path, log_path: Path) -> dict[str, Any]:
    output_dir = input_path.resolve().parent
    return {
        "cwd": str(REPO_ROOT),
        "output_dir": str(output_dir),
        "schedule_path": str(output_dir / "schedule.json"),
        "status_path": str(input_path.resolve()),
        "external_log_path": str(log_path.resolve()),
        "plan_path": PLAN_PATH,
        "result_path": RESULT_PATH,
        "dimensions": [2],
        "parameter_counts": [3],
        "timesteps": 4,
        "batch_size": 4,
        "dtype": "float32",
        "device": "cpu",
        "cpu_threads": 1,
        "repeats": 2,
        "timeout_seconds": 90.0,
        "methods": list(contract.PRIMARY_METHOD_IDS),
        "no_resume": True,
        "jit_compile": True,
        "xla_execution": "xla_jit_enabled",
        "xla_flags": "UNSET",
        "tf32_enabled": True,
        "cuda_visible_devices": "-1",
        "thread_environment": {
            "OMP_NUM_THREADS": "1",
            "TF_NUM_INTRAOP_THREADS": "1",
            "TF_NUM_INTEROP_THREADS": "1",
        },
        "timing_boundary_version": TIMING_BOUNDARY_VERSION,
        "trust_basis": "gpu_hidden_cpu_debug_reference",
        "outer_timeout_seconds": 210.0,
        "outer_timeout_scope": "declared_shell_emergency_cap_not_observed_by_process",
        "nonclaims": list(PHASE5_NONCLAIMS),
    }


def evaluate_phase5_smoke_raw(
    raw: Mapping[str, Any],
    *,
    input_path: Path,
    log_path: Path,
    sidecar_reader: Callable[[Path], str] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    reader = sidecar_reader or (lambda path: path.read_text(encoding="utf-8"))
    execution = raw.get("execution_contract")
    expected_execution = _phase5_expected_execution(input_path, log_path)
    execution_common = (
        isinstance(execution, Mapping)
        and set(execution) == {*expected_execution, "command_argv", "git_commit"}
        and all(execution.get(key) == value for key, value in expected_execution.items())
    )
    command_valid = (
        isinstance(execution, Mapping)
        and isinstance(execution.get("command_argv"), list)
        and execution["command_argv"][0] == str(PYTHON.resolve())
        and execution["command_argv"][1:]
        == [
            str(Path(__file__).resolve()),
            "--dimensions", "2",
            "--parameter-counts", "3",
            "--timesteps", "4",
            "--batch-size", "4",
            "--dtype", "float32",
            "--device", "cpu",
            "--cpu-threads", "1",
            "--repeats", "2",
            "--timeout-seconds", "90",
            "--methods", *contract.PRIMARY_METHOD_IDS,
            "--output-dir", str(input_path.resolve().parent),
            "--no-resume",
            "--jit-compile",
            "--tf32-enabled",
        ]
    )
    records = raw.get("records")
    records_valid = isinstance(records, list) and len(records) == 2
    embedded_sidecars: list[dict[str, Any]] = []
    method_ids: list[Any] = []
    measurement_checks: dict[str, bool] = {}
    sidecars_valid = records_valid
    child_runtime_valid = records_valid
    record_terminal_valid = records_valid
    resume_contract_valid = records_valid
    if records_valid:
        expected_by_method = {
            identity["method_id"]: identity
            for identity in raw.get("schedule", {}).get("expected_identities", [])
            if isinstance(identity, Mapping)
        }
        for record in records:
            method_id = record.get("method_id") if isinstance(record, Mapping) else None
            method_ids.append(method_id)
            prefix = str(method_id)
            for name, passed in contract.measurement_record_checks(record).items():
                measurement_checks[f"{prefix}:{name}"] = passed
            sidecar = record.get("measurement", {}).get("payload_sidecar", {})
            try:
                sidecar_path = Path(sidecar["path"])
                identity = expected_by_method[method_id]
                expected_sidecar_path = method_artifact_path(
                    input_path.resolve().parent, identity
                ).with_suffix(".payload.json")
                content = reader(sidecar_path)
                decoded = contract.strict_json_loads(content)
                digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
                expected_payload = {
                    "case_id": record.get("case_id"),
                    "method_id": record.get("method_id"),
                    "output_metadata": record.get("output_metadata"),
                    "outputs": record.get("outputs"),
                    "graphdef": record.get("measurement", {}).get("graphdef"),
                    "direct_output_parity": record.get("measurement", {}).get(
                        "direct_output_parity"
                    ),
                }
                sidecars_valid = (
                    sidecars_valid
                    and sidecar_path.resolve() == expected_sidecar_path.resolve()
                    and digest == sidecar.get("sha256")
                    and decoded == expected_payload
                )
                embedded_sidecars.append(
                    {
                        "method_id": method_id,
                        "path": str(sidecar_path),
                        "sha256": digest,
                        "payload": decoded,
                    }
                )
            except (KeyError, OSError, contract.ContractError, TypeError, ValueError):
                sidecars_valid = False
            device = record.get("device_manifest")
            threads = record.get("cpu_thread_manifest")
            child_runtime_valid = child_runtime_valid and (
                isinstance(device, Mapping)
                and device.get("requested_device") == "cpu"
                and device.get("selected_device") == "/CPU:0"
                and device.get("physical_gpus") == []
                and device.get("logical_gpus") == []
                and device.get("cpu_only_exception") is True
                and device.get("trust_basis") == "cpu_debug_or_reference_exception"
                and isinstance(threads, Mapping)
                and threads.get("requested_cpu_threads") == 1
                and threads.get("tf_intra_op_parallelism_threads") == 1
                and threads.get("tf_inter_op_parallelism_threads") == 1
                and threads.get("omp_num_threads") == "1"
                and threads.get("tf_num_intraop_threads_env") == "1"
                and threads.get("tf_num_interop_threads_env") == "1"
            )
            record_terminal_valid = record_terminal_valid and (
                record.get("state") == "passed"
                and record.get("returncode") == 0
                and record.get("timed_out") is False
                and record.get("last_entered_stage") == "envelope_write"
                and record.get("terminal_stage") == "envelope_write"
                and record.get("failure_stage") is None
                and record.get("error") is None
            )
            try:
                fingerprints = _fingerprints(identity, raw["schedule"]["schedule_fingerprint"])
                reusable, _ = contract.method_record_reuse_decision(
                    record,
                    expected_case_id=identity["case_id"],
                    expected_method_id=method_id,
                    expected_fingerprints=fingerprints,
                    expected_resume_key=contract.resume_key(
                        case_identity=identity["case_id"],
                        method_id=method_id,
                        fingerprints=fingerprints,
                    ),
                )
                resume_contract_valid = resume_contract_valid and reusable
            except (KeyError, TypeError, contract.ContractError):
                resume_contract_valid = False
    source_current = contract.source_manifest(REPO_ROOT, include_supervisor=True)
    schedule = raw.get("schedule")
    expected_args = argparse.Namespace(
        dimensions=[2],
        parameter_counts=[3],
        timesteps=4,
        batch_size=4,
        dtype="float32",
        device="cpu",
        cpu_threads=1,
        repeats=2,
        timeout_seconds=90.0,
        methods=list(contract.PRIMARY_METHOD_IDS),
        output_dir=input_path.resolve().parent,
        harness_contract_test_only=False,
        no_resume=True,
        jit_compile=True,
        tf32_enabled=True,
    )
    expected_schedule = build_schedule(expected_args)
    schedule_identity_fields = {
        "schema",
        "method_contract_version",
        "comparison_mode",
        "primary_pair_complete",
        "comparator_parity_applicable",
        "comparator_parity_reason",
        "expected_identities",
        "mandatory_aggregate_checks",
        "harness_contract_test_only",
        "schedule_fingerprint",
        "source_manifest",
        "runtime_manifest",
        "plan_path",
        "timing_boundary_version",
        "nonclaims",
    }
    expected_schedule_fields = schedule_identity_fields | {
        "created_utc",
        "execution_contract",
    }
    schedule_valid = (
        isinstance(schedule, Mapping)
        and set(schedule) == expected_schedule_fields
        and all(
            schedule.get(field) == expected_schedule.get(field)
            for field in schedule_identity_fields
        )
        and schedule.get("source_manifest") == source_current
        and schedule.get("execution_contract") == execution
        and isinstance(schedule.get("created_utc"), str)
    )
    aggregate = raw.get("aggregate_checks")
    recomputed_aggregate = (
        _aggregate_checks(schedule, records)
        if isinstance(schedule, Mapping) and records_valid
        else None
    )
    aggregate_valid = (
        isinstance(aggregate, Mapping)
        and aggregate == recomputed_aggregate
        and set(aggregate) == set(contract.PRIMARY_PAIR_CHECKS)
        and all(aggregate.get(name) is True for name in contract.PRIMARY_PAIR_CHECKS)
    )
    checks = {
        "schema_identity": raw.get("schema") == contract.SCHEMA,
        "top_level_passed": raw.get("status") == "complete",
        "execution_identity": execution_common,
        "argv_identity": command_valid,
        "git_commit_recorded": isinstance(execution, Mapping)
        and isinstance(execution.get("git_commit"), str)
        and len(execution["git_commit"]) == 40
        and all(character in "0123456789abcdef" for character in execution["git_commit"]),
        "schedule_source_runtime_identity": schedule_valid,
        "primary_method_identity": method_ids == list(contract.PRIMARY_METHOD_IDS),
        "method_records_passed": records_valid
        and all(record.get("state") == "passed" for record in records),
        "method_record_terminal_identity": record_terminal_valid,
        "method_record_resume_contract": resume_contract_valid,
        "child_cpu_thread_device_identity": child_runtime_valid,
        "sidecar_content_hash_identity": sidecars_valid,
        "aggregate_checks_recomputed_identity": aggregate_valid,
        "comparison_complete": raw.get("comparison_summary", {}).get("comparison_complete")
        is True,
        **measurement_checks,
    }
    return checks, embedded_sidecars


def build_phase5_smoke_export(
    raw: Mapping[str, Any],
    *,
    input_path: Path,
    log_path: Path,
    output_path: Path,
    expected_input: Path | None = None,
    expected_log: Path | None = None,
    expected_output: Path | None = None,
    sidecar_reader: Callable[[Path], str] | None = None,
) -> tuple[dict[str, Any], int]:
    checks, sidecars = evaluate_phase5_smoke_raw(
        raw,
        input_path=input_path,
        log_path=log_path,
        sidecar_reader=sidecar_reader,
    )
    expected_input = expected_input or Path(
        "/tmp/kalman_qr_phase5_measurement/status.json"
    )
    expected_log = expected_log or Path("/tmp/kalman_qr_phase5_measurement/smoke.log")
    expected_output = expected_output or REPO_ROOT / PHASE5_SMOKE_OUTPUT
    checks.update(
        {
            "phase5_input_path_identity": input_path.resolve()
            == expected_input.resolve(),
            "phase5_log_path_identity": log_path.resolve() == expected_log.resolve(),
            "phase5_output_path_identity": output_path.resolve()
            == expected_output.resolve(),
            "external_log_exists": log_path.is_file(),
        }
    )
    passed = bool(checks) and all(checks.values())
    payload = {
        "schema": PHASE5_SMOKE_SCHEMA,
        "state": "passed" if passed else "failed",
        "source_status_path": str(input_path.resolve()),
        "external_log_path": str(log_path.resolve()),
        "plan_path": PLAN_PATH,
        "result_path": RESULT_PATH,
        "evaluator_command_argv": [
            str(Path(sys.executable).resolve()),
            str(Path(sys.argv[0]).resolve()),
            *sys.argv[1:],
        ],
        "raw_status": raw,
        "embedded_payload_sidecars": sidecars,
        "checks": checks,
        "nonclaims": list(PHASE5_NONCLAIMS),
    }
    return payload, 0 if passed else 1


def export_phase5_smoke(input_path: Path, log_path: Path, output_path: Path) -> int:
    raw = contract.read_strict_json(input_path)
    payload, returncode = build_phase5_smoke_export(
        raw,
        input_path=input_path,
        log_path=log_path,
        output_path=output_path,
    )
    contract.atomic_write_json(output_path, payload)
    return returncode


def main() -> int:
    args = parse_args()
    if args.phase6_archive_r1:
        return run_phase6_archive_r1(args)
    if args.phase6_archive_r2:
        return run_phase6_archive_r2(args)
    if args.phase6_prepare_proposal is not None:
        return run_phase6_prepare_proposal(args)
    if args.phase6_create_attestation is not None:
        return run_phase6_create_attestation(args)
    if args.phase6_validate_authority is not None:
        return run_phase6_validate_authority(args)
    if args.phase6_pilot:
        return run_phase6_pilot(args)
    if args.phase6_scalar_references:
        return run_phase6_scalar_references(args)
    if args.phase6_remaining:
        return run_phase6_remaining(args)
    if args.phase6_evaluate:
        return run_phase6_evaluate(args)
    if args.evaluate_phase5_smoke:
        if not all((args.phase5_input, args.phase5_log, args.phase5_output)):
            raise ValueError("Phase 5 evaluator requires input, log, and output paths")
        return export_phase5_smoke(
            args.phase5_input.resolve(),
            args.phase5_log.resolve(),
            (REPO_ROOT / args.phase5_output).resolve()
            if not args.phase5_output.is_absolute()
            else args.phase5_output.resolve(),
        )
    if args.timesteps <= 0 or args.batch_size <= 0 or args.repeats <= 0:
        raise ValueError("timesteps, batch size, and repeats must be positive")
    if args.timeout_seconds <= 0 or args.cpu_threads <= 0:
        raise ValueError("timeout and CPU threads must be positive")
    payload, returncode = execute_schedule(args)
    print(
        contract.strict_json_dumps(
            {
                "status": payload["status"],
                "status_path": str(args.output_dir.resolve() / "status.json"),
                "record_count": len(payload.get("records", [])),
            },
            indent=2,
        )
    )
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
