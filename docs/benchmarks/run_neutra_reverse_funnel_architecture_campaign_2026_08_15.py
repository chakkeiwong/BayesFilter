#!/usr/bin/env python3
"""Run the reviewed reverse-funnel architecture and LR-schedule campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "docs/plans/bayesfilter-neutra-reverse-funnel-architecture-tuning-plan-2026-08-15.md"
RUNNER = ROOT / "docs/benchmarks/run_neutra_reverse_funnel_capacity_2026_08_14.py"
DEFAULT_OUTPUT = ROOT / "docs/plans/artifacts/neutra-reverse-funnel-architecture-tuning-2026-08-15"
DEFAULT_REPLAY = ROOT / "docs/plans/artifacts/weighted-forward-kl-positive-controls-2026-08-12/paper-d100/funnel-replay-r1"

ARCHITECTURES = {
    "one_stage_exact": {
        "stages": 1,
        "hidden_width": 100,
        "stage_s_max": "4",
        "permutation_policy": "full_reverse",
    },
    "three_root_preserving": {
        "stages": 3,
        "hidden_width": 100,
        "stage_s_max": "4,0.5,0.5",
        "permutation_policy": "root_preserving_reverse",
    },
    "three_full_reverse": {
        "stages": 3,
        "hidden_width": 100,
        "stage_s_max": "4,0.5,0.5",
        "permutation_policy": "full_reverse",
    },
    "three_root_wide": {
        "stages": 3,
        "hidden_width": 200,
        "stage_s_max": "4,0.5,0.5",
        "permutation_policy": "root_preserving_reverse",
    },
}
PEAK_RATES = (2.0e-4, 5.0e-4, 1.0e-3, 2.0e-3)
SCHEDULES = ("constant", "piecewise_60_85")


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--replay-root", type=Path, default=DEFAULT_REPLAY)
    parser.add_argument("--device", default="1")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--calibration-updates", type=int, default=1000)
    parser.add_argument("--confirmation-updates", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _load(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _rate_id(value: float) -> str:
    return f"{value:.0e}".replace("-", "m").replace("+", "p")


def _command(
    args: argparse.Namespace,
    architecture_id: str,
    architecture: Mapping[str, Any],
    output: Path,
    *,
    run_mode: str,
    updates: int,
    learning_rate: float,
    schedule: str,
    seed_index: int,
) -> list[str]:
    return [
        str(args.python),
        str(RUNNER),
        "--output-root",
        str(output),
        "--replay-root",
        str(args.replay_root.resolve()),
        "--plan",
        str(PLAN),
        "--architecture-id",
        architecture_id,
        "--run-mode",
        run_mode,
        "--device",
        str(args.device),
        "--stages",
        str(architecture["stages"]),
        "--hidden-width",
        str(architecture["hidden_width"]),
        "--permutation-policy",
        str(architecture["permutation_policy"]),
        "--stage-s-max",
        str(architecture["stage_s_max"]),
        "--first-stage-unbounded-scale-linear",
        "--updates",
        str(updates),
        "--batch-size",
        str(args.batch_size),
        "--checkpoint-every",
        "250",
        "--learning-rate",
        str(learning_rate),
        "--learning-rate-schedule",
        schedule,
        "--seed-index",
        str(seed_index),
        "--selection-count",
        "65536",
        "--proposal-audit-count",
        "32768" if run_mode == "calibration" else "131072",
    ]


def _run(command: list[str], output: Path, resume: bool) -> Mapping[str, Any]:
    result_path = output / "result.json"
    if resume and result_path.is_file():
        return _load(result_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env={**os.environ, "TF_FORCE_GPU_ALLOW_GROWTH": "true"},
        text=True,
        capture_output=True,
        check=False,
    )
    log_path = output.parent / f"{output.name}.process.log"
    log_path.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    if completed.returncode != 0 or not result_path.is_file():
        raise RuntimeError(
            f"campaign cell failed with exit {completed.returncode}: {' '.join(command)}"
        )
    return _load(result_path)


def main() -> int:
    args = _args()
    if not PLAN.is_file() or not RUNNER.is_file():
        raise FileNotFoundError("reviewed plan or cell runner is missing")
    output_root = args.output_root.resolve()
    if output_root.exists() and not args.resume:
        raise FileExistsError(f"output root must be fresh unless --resume is used: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    calibration_rows = []
    selected: dict[str, Mapping[str, Any]] = {}

    for architecture_id, architecture in ARCHITECTURES.items():
        candidates = []
        for peak_rate in PEAK_RATES:
            for schedule in SCHEDULES:
                cell_id = f"lr-{_rate_id(peak_rate)}-{schedule}"
                cell_root = output_root / "calibration" / architecture_id / cell_id
                command = _command(
                    args,
                    architecture_id,
                    architecture,
                    cell_root,
                    run_mode="calibration",
                    updates=int(args.calibration_updates),
                    learning_rate=peak_rate,
                    schedule=schedule,
                    seed_index=0,
                )
                result = _run(command, cell_root, bool(args.resume))
                row = {
                    "architecture_id": architecture_id,
                    "peak_learning_rate": peak_rate,
                    "learning_rate_schedule": schedule,
                    "selected_loss": float(result["selected_loss"]),
                    "selected_update": int(result["selected_update"]),
                    "proposal_screen_passed": bool(
                        result["proposal_audit"]["all_individual_intervals_passed"]
                    ),
                    "output_root": str(cell_root),
                    "result_sha256": _sha256(cell_root / "result.json"),
                    "command": command,
                }
                calibration_rows.append(row)
                candidates.append(row)
        selected[architecture_id] = min(candidates, key=lambda row: row["selected_loss"])
        _write(
            output_root / "progress.json",
            {
                "schema": "bayesfilter.neutra.reverse_funnel_architecture_progress.v1",
                "completed_calibration_cells": len(calibration_rows),
                "selected": selected,
            },
        )

    confirmation_rows = []
    for architecture_id, architecture in ARCHITECTURES.items():
        choice = selected[architecture_id]
        for seed_index in (1, 2):
            cell_root = output_root / "confirmation" / architecture_id / f"seed-{seed_index}"
            command = _command(
                args,
                architecture_id,
                architecture,
                cell_root,
                run_mode="confirmation",
                updates=int(args.confirmation_updates),
                learning_rate=float(choice["peak_learning_rate"]),
                schedule=str(choice["learning_rate_schedule"]),
                seed_index=seed_index,
            )
            result = _run(command, cell_root, bool(args.resume))
            confirmation_rows.append(
                {
                    "architecture_id": architecture_id,
                    "seed_index": seed_index,
                    "peak_learning_rate": float(choice["peak_learning_rate"]),
                    "learning_rate_schedule": str(choice["learning_rate_schedule"]),
                    "selected_loss": float(result["selected_loss"]),
                    "selected_update": int(result["selected_update"]),
                    "proposal_gate_passed": bool(
                        result["decision"]["proposal_gate_passed"]
                    ),
                    "output_root": str(cell_root),
                    "result_sha256": _sha256(cell_root / "result.json"),
                    "command": command,
                }
            )
            _write(
                output_root / "progress.json",
                {
                    "schema": "bayesfilter.neutra.reverse_funnel_architecture_progress.v1",
                    "completed_calibration_cells": len(calibration_rows),
                    "completed_confirmation_cells": len(confirmation_rows),
                    "selected": selected,
                    "confirmation": confirmation_rows,
                },
            )

    summary = {
        "schema": "bayesfilter.neutra.reverse_funnel_architecture_campaign.v1",
        "plan": str(PLAN),
        "git_commit": subprocess.run(
            ("git", "rev-parse", "HEAD"),
            cwd=ROOT,
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip(),
        "calibration": calibration_rows,
        "selected": selected,
        "confirmation": confirmation_rows,
        "wall_seconds": time.perf_counter() - started,
        "inference_status": {
            "hard_veto_screen": "see per-cell results",
            "statistically_supported_ranking": False,
            "descriptive_only_differences": True,
            "default_readiness": False,
            "next_evidence_needed": "HMC only after repeat proposal-law passage",
        },
    }
    _write(output_root / "campaign_result.json", summary)
    _write(
        output_root / "artifact_hashes.json",
        {
            "schema": "bayesfilter.neutra.reverse_funnel_architecture_campaign_hashes.v1",
            "artifacts": {
                path.relative_to(output_root).as_posix(): _sha256(path)
                for path in sorted(output_root.rglob("*"))
                if path.is_file() and path.name != "artifact_hashes.json"
            },
        },
    )
    print(json.dumps({"output_root": str(output_root), "wall_seconds": summary["wall_seconds"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
