#!/usr/bin/env python3
"""Supervise seed-B fixed-HMC tuning followed by sequential validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = Path(__file__).resolve()
PLAN = Path(
    "docs/plans/"
    "bayesfilter-ssl-lstm-q20-seed-b-terminal-neutra-validation-plan-2026-08-07.md"
)
TUNER = ROOT / (
    "docs/benchmarks/"
    "run_ssl_lstm_q20_seed_b_terminal_six_l_tuning_2026_08_07.py"
)
SEQUENTIAL = ROOT / (
    "docs/benchmarks/"
    "run_ssl_lstm_q20_seed_b_terminal_sequential_hmc_2026_08_07.py"
)
DEFAULT_OUTPUT = Path(
    "docs/plans/artifacts/"
    "ssl-lstm-q20-seed-b-terminal-neutra-validation-2026-08-07/r1"
)
SCHEMA = "bayesfilter.ssl_lstm.q20_seed_b_terminal_neutra_validation.v1"
TUNING_CAP_SECONDS = 43_200.0
SEQUENTIAL_CAP_SECONDS = 86_400.0
CAMPAIGN_CAP_SECONDS = TUNING_CAP_SECONDS + SEQUENTIAL_CAP_SECONDS


class SupervisorError(RuntimeError):
    """Raised when the phase chain violates the campaign contract."""


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _canonical(payload: Any) -> bytes:
    return (
        json.dumps(
            _json_ready(payload),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def _write_json(path: Path, payload: Mapping[str, Any], *, replace: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        raise SupervisorError(f"refusing to overwrite artifact: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(_canonical(payload))
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(payload, dict):
        raise SupervisorError(f"expected JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*args: str) -> str:
    return subprocess.run(
        ("git", *args),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _phase_command(
    *, cpu: int, script: Path, mode: str, output: Path, cap_seconds: float | None
) -> list[str]:
    command = [
        "taskset",
        "-c",
        str(int(cpu)),
        sys.executable,
        str(script),
        "--mode",
        str(mode),
        "--output-root",
        output.relative_to(ROOT).as_posix(),
    ]
    if cap_seconds is not None:
        command.extend(("--cap-seconds", f"{float(cap_seconds):.1f}"))
    return command


def _run_phase(
    *, label: str, command: Sequence[str], log_path: Path, progress_path: Path,
    campaign_started: float, phase_rows: list[Mapping[str, Any]],
) -> Mapping[str, Any]:
    if log_path.exists():
        raise SupervisorError(f"phase log already exists: {log_path}")
    started_utc = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    _write_json(
        progress_path,
        {
            "schema": SCHEMA,
            "status": "RUNNING",
            "active_phase": label,
            "phase_started_utc": started_utc,
            "campaign_elapsed_seconds": time.perf_counter() - campaign_started,
            "completed_phases": phase_rows,
        },
        replace=progress_path.exists(),
    )
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "-1",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    with log_path.open("x", encoding="utf-8") as log:
        completed = subprocess.run(
            tuple(command),
            cwd=ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    row = {
        "phase": label,
        "started_utc": started_utc,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": time.perf_counter() - started,
        "exit_code": int(completed.returncode),
        "command": list(command),
        "log_path": log_path.relative_to(ROOT).as_posix(),
        "log_sha256": _sha256(log_path),
    }
    phase_rows.append(row)
    return row


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    output = (ROOT / args.output_root).resolve()
    if not output.is_relative_to(ROOT):
        raise SupervisorError("output root must stay inside the repository")
    if output.exists() and any(output.iterdir()):
        raise SupervisorError("campaign output root must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    campaign_started = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat()
    progress_path = output / "campaign-progress.json"
    phase_rows: list[Mapping[str, Any]] = []
    status = "CAMPAIGN_EXECUTION_ERROR"
    stop_reason: str | None = None

    tuning_output = output / "tuning"
    tuning = _run_phase(
        label="seed_b_fixed_hmc_tuning",
        command=_phase_command(
            cpu=127,
            script=TUNER,
            mode="supervisor",
            output=tuning_output,
            cap_seconds=TUNING_CAP_SECONDS,
        ),
        log_path=output / "tuning-supervisor.log",
        progress_path=progress_path,
        campaign_started=campaign_started,
        phase_rows=phase_rows,
    )
    if tuning["exit_code"] != 0:
        stop_reason = "tuning_process_failed_or_timed_out"
    else:
        merged_path = tuning_output / "merged-tuning-result.json"
        if not merged_path.exists():
            stop_reason = "tuning_emitted_no_merged_artifact"
        else:
            merged = _read_json(merged_path)
            if merged.get("passed") is not True:
                status = "NO_SEED_B_FIXED_HMC_KERNEL_NOMINATED"
                stop_reason = "public_tuner_found_no_viable_seed_b_kernel"
            else:
                sequential_preflight = _run_phase(
                    label="seed_b_sequential_preflight",
                    command=_phase_command(
                        cpu=32,
                        script=SEQUENTIAL,
                        mode="preflight",
                        output=output / "sequential-preflight",
                        cap_seconds=None,
                    ),
                    log_path=output / "sequential-preflight.log",
                    progress_path=progress_path,
                    campaign_started=campaign_started,
                    phase_rows=phase_rows,
                )
                if sequential_preflight["exit_code"] != 0:
                    stop_reason = "sequential_preflight_failed"
                else:
                    sequential = _run_phase(
                        label="seed_b_sequential_hmc",
                        command=_phase_command(
                            cpu=32,
                            script=SEQUENTIAL,
                            mode="run",
                            output=output / "sequential",
                            cap_seconds=SEQUENTIAL_CAP_SECONDS,
                        ),
                        log_path=output / "sequential-supervisor.log",
                        progress_path=progress_path,
                        campaign_started=campaign_started,
                        phase_rows=phase_rows,
                    )
                    sequential_summary_path = output / "sequential" / "summary.json"
                    if sequential_summary_path.exists():
                        sequential_summary = _read_json(sequential_summary_path)
                        status = str(sequential_summary.get("status", status))
                        stop_reason = str(
                            sequential_summary.get("stop_reason", "unknown")
                        )
                    elif sequential["exit_code"] != 0:
                        stop_reason = "sequential_process_failed_without_summary"
                    else:
                        stop_reason = "sequential_process_emitted_no_summary"

    manifest = {
        "schema": "bayesfilter.serious_run_manifest.v1",
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "command": [sys.executable, *sys.argv],
        "environment": os.environ.get("CONDA_DEFAULT_ENV", "N/A"),
        "python": sys.version.split()[0],
        "started_at_utc": started_utc,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": time.perf_counter() - campaign_started,
        "cap_seconds": CAMPAIGN_CAP_SECONDS,
        "cpu_gpu_status": (
            "CPU/XLA HMC validation lane; CUDA_VISIBLE_DEVICES=-1 in supervisor "
            "and all children"
        ),
        "random_seed_roots": {
            "tuning": [20260807, 10100, 20100, 30100],
            "sequential": [20260807, 41001],
        },
        "output_root": output.relative_to(ROOT).as_posix(),
        "plan_file": PLAN.as_posix(),
        "result_file": (output / "campaign-summary.json").relative_to(ROOT).as_posix(),
        "source_sha256": {
            "supervisor": _sha256(SCRIPT),
            "tuner": _sha256(TUNER),
            "sequential": _sha256(SEQUENTIAL),
            "plan": _sha256(ROOT / PLAN),
        },
    }
    summary = {
        "schema": SCHEMA,
        "status": status,
        "stop_reason": stop_reason,
        "phase_rows": phase_rows,
        "manifest": manifest,
        "decision": {
            "primary_criterion_status": (
                "passed" if status == "SEQUENTIAL_SCREEN_PASSED" else "not_passed"
            ),
            "veto_diagnostic_status": stop_reason,
            "next_justified_action": (
                "untouched posterior/reference validation"
                if status == "SEQUENTIAL_SCREEN_PASSED"
                else "diagnose the recorded tuning, kernel, numerical, or resource stop"
            ),
            "not_concluded": [
                "posterior correctness",
                "model adequacy",
                "NeuTra superiority",
                "default readiness",
            ],
        },
        "inference_status": {
            "hard_veto_screen": stop_reason,
            "statistically_supported_ranking": "none",
            "descriptive_only_differences": [
                "tuning acceptance",
                "runtime",
                "continuous R-hat and ESS before threshold passage",
            ],
            "default_readiness": "not_evaluated",
            "next_evidence_needed": (
                "posterior/reference validation after a sequential pass"
            ),
        },
        "nonclaims": [
            "a failed seed-B candidate does not reject NeuTra generally",
            "a sequential pass is not posterior correctness",
            "no stochastic candidate ranking is supported",
        ],
    }
    _write_json(output / "run-manifest.json", manifest)
    _write_json(output / "campaign-summary.json", summary)
    _write_json(
        progress_path,
        {
            "schema": SCHEMA,
            "status": status,
            "active_phase": None,
            "stop_reason": stop_reason,
            "campaign_elapsed_seconds": time.perf_counter() - campaign_started,
            "completed_phases": phase_rows,
        },
        replace=True,
    )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    if args.output_root != DEFAULT_OUTPUT:
        parser.error("material supervisor output is fixed to the reviewed r1 root")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    result = run(parse_args(argv))
    print(json.dumps(_json_ready(result), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

