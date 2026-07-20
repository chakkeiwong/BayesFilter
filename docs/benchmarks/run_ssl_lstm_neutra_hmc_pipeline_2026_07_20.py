#!/usr/bin/env python3
"""One-command orchestration for the q-general SSL-LSTM NeuTra/HMC pipeline.

Numerical work remains in the existing stage scripts. This wrapper only binds
their commands, artifacts, and fail-closed handoffs into one repeatable run.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PYTHON = Path(sys.executable)
TRAINING = Path("docs/benchmarks/run_ssl_lstm_neutra_complexity_training_2026_07_19.py")
TUNING = Path("docs/benchmarks/run_ssl_lstm_neutra_complexity_hmc_tuning_2026_07_19.py")
RETAINED = Path("docs/benchmarks/run_ssl_lstm_neutra_complexity_retained_hmc_2026_07_19.py")
Q_VALUES = (1, 2, 5, 10, 20)
SCHEMA = "bayesfilter.ssl_lstm.neutra_hmc_pipeline.v1"


class PipelineError(RuntimeError):
    pass


def canonical(payload: Any) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(canonical(payload))
    temporary.replace(path)


def repo_path(path: Path, label: str) -> Path:
    resolved = (ROOT / path).resolve()
    if not resolved.is_relative_to(ROOT):
        raise PipelineError(f"{label} must remain inside the repository")
    return resolved


def read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PipelineError(f"cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise PipelineError(f"{label} must be a JSON object: {path}")
    return value


def command_record(command: list[str], output: Path, *, resume: bool) -> dict[str, Any]:
    return {
        "command": command,
        "output_root": output.relative_to(ROOT).as_posix(),
        "resume": bool(resume),
    }


def run_child(
    *,
    name: str,
    command: list[str],
    output: Path,
    resume: bool,
) -> dict[str, Any]:
    log_path = output / "logs" / f"{name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    env = os.environ.copy()
    env.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    elapsed = time.perf_counter() - started
    record = {
        **command_record(command, output, resume=resume),
        "name": name,
        "returncode": int(completed.returncode),
        "elapsed_seconds": elapsed,
        "log_path": log_path.relative_to(ROOT).as_posix(),
    }
    if completed.returncode != 0:
        raise PipelineError(f"{name} exited with status {completed.returncode}")
    return record


def require_status(path: Path, *, label: str, statuses: set[str]) -> dict[str, Any]:
    payload = read_json(path, label)
    if str(payload.get("status")) not in statuses:
        raise PipelineError(
            f"{label} status {payload.get('status')!r} is not one of {sorted(statuses)}"
        )
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("contract-smoke", "run"), default="run")
    parser.add_argument("--q", type=int, choices=Q_VALUES, required=True)
    parser.add_argument("--batch-size", type=int, default=480)
    parser.add_argument("--params-json", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--training-cap-seconds", type=float)
    parser.add_argument("--hmc-tuning-cap-seconds", type=float)
    parser.add_argument("--retained-hmc-cap-seconds", type=float)
    parser.add_argument("--authorize-material-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.mode == "contract-smoke":
        return args
    if not args.authorize_material_run:
        parser.error("run requires --authorize-material-run")
    for name in (
        "training_cap_seconds",
        "hmc_tuning_cap_seconds",
        "retained_hmc_cap_seconds",
    ):
        value = getattr(args, name)
        if value is None or value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.params_json is None:
        parser.error("run requires --params-json")
    if args.output_root is None:
        parser.error("run requires --output-root")
    repo_path(args.params_json, label="parameter file")
    output = repo_path(args.output_root, label="output root")
    if output.exists() and not args.resume and any(output.iterdir()):
        parser.error("fresh runs require an empty or nonexistent output root")
    return args


def contract_payload(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "mode": "contract-smoke",
        "status": "PASSED",
        "q": args.q,
        "batch_size": args.batch_size,
        "stages": ["training", "hmc_tuning", "retained_hmc"],
        "handoffs": [
            "two ADMITTED training results",
            "KERNELS_FROZEN HMC tuning summary",
            "retained HMC sequential checkpoints",
        ],
        "material_execution_authorized": False,
    }


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    output = repo_path(args.output_root, label="output root")
    output.mkdir(parents=True, exist_ok=args.resume)
    training_output = output / "training"
    tuning_output = output / "hmc-tuning"
    retained_output = output / "retained-hmc"
    params = repo_path(args.params_json, label="parameter file")
    records: list[dict[str, Any]] = []
    started = time.perf_counter()
    training_command = [
        str(PYTHON), str(ROOT / TRAINING), "--mode", "final", "--q", str(args.q),
        "--batch-size", str(args.batch_size), "--authorize-material-run",
        "--gpu-cap-seconds", str(args.training_cap_seconds), "--params-json", str(params),
        "--output-root", str(training_output),
    ]
    if args.resume:
        training_command.append("--resume")
    training_summary_path = training_output / "final-summary.json"
    tuning_summary_path = tuning_output / "summary.json"
    retained_summary_path = retained_output / "summary.json"
    try:
        training_record = run_child(
            name="training", command=training_command, output=output, resume=args.resume
        )
        records.append(training_record)
        training_summary = require_status(
            training_summary_path, label="training summary", statuses={"COMPLETED"}
        )
        results = training_summary.get("results", [])
        if len(results) != 2 or any(row.get("status") != "ADMITTED" for row in results):
            raise PipelineError("training did not produce two ADMITTED streams")
        result_paths = [repo_path(Path(row["path"]), label="training result") for row in results]
        tuning_command = [
            str(PYTHON), str(ROOT / TUNING), "--mode", "tune", "--q", str(args.q),
            "--phase3-result-a", str(result_paths[0]), "--phase3-result-b", str(result_paths[1]),
            "--output-root", str(tuning_output), "--cap-seconds", str(args.hmc_tuning_cap_seconds),
            "--authorize-material-run",
        ]
        if args.resume:
            tuning_command.append("--resume")
        records.append(run_child(name="hmc-tuning", command=tuning_command, output=output, resume=args.resume))
        require_status(tuning_summary_path, label="HMC tuning summary", statuses={"KERNELS_FROZEN"})
        retained_command = [
            str(PYTHON), str(ROOT / RETAINED), "--mode", "acquire", "--q", str(args.q),
            "--phase4-summary", str(tuning_summary_path), "--output-root", str(retained_output),
            "--cap-seconds", str(args.retained_hmc_cap_seconds), "--authorize-material-run",
        ]
        if args.resume:
            retained_command.append("--resume")
        records.append(run_child(name="retained-hmc", command=retained_command, output=output, resume=args.resume))
        retained_summary = read_json(retained_summary_path, "retained-HMC summary")
        status = str(retained_summary.get("status", "UNKNOWN"))
        failure = None
    except PipelineError as exc:
        status = "PIPELINE_STOPPED"
        failure = str(exc)
    payload = {
        "schema": SCHEMA,
        "mode": "run",
        "status": status,
        "q": args.q,
        "batch_size": args.batch_size,
        "elapsed_seconds": time.perf_counter() - started,
        "stage_records": records,
        "training_summary": training_summary_path.relative_to(ROOT).as_posix(),
        "hmc_tuning_summary": tuning_summary_path.relative_to(ROOT).as_posix(),
        "retained_hmc_summary": retained_summary_path.relative_to(ROOT).as_posix(),
        "failure": failure,
        "nonclaims": [
            "wrapper execution is not posterior correctness",
            "retained finite-sample admission is not predictive equivalence",
        ],
    }
    write_json(output / "pipeline-summary.json", payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = contract_payload(args) if args.mode == "contract-smoke" else run_pipeline(args)
    print("JSON_SUMMARY " + json.dumps({"mode": payload["mode"], "status": payload["status"], "q": payload["q"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
