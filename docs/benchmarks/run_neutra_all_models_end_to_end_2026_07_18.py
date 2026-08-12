#!/usr/bin/env python3
"""Run the reusable NeuTra pipeline for every executable BayesFilter cell."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PLAN_PATH = Path(
    "docs/plans/bayesfilter-neutra-all-executable-models-end-to-end-python-plan-"
    "2026-07-18.md"
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--action",
        choices=(
            "registry",
            "preflight",
            "training-throughput",
            "validate-frozen",
            "broad-grid-frozen",
            "sample-broad-grid-frozen",
            "cell",
            "campaign",
        ),
        required=True,
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--cell")
    parser.add_argument("--screen-steps", type=int, default=500)
    parser.add_argument("--final-steps", type=int, default=5000)
    parser.add_argument("--final-segment-steps", type=int, default=1000)
    parser.add_argument("--screen-only", action="store_true")
    parser.add_argument("--screen-result", type=Path)
    parser.add_argument("--screen-result-sha256")
    parser.add_argument("--recipe")
    parser.add_argument("--throughput-steps", type=int, default=25)
    parser.add_argument("--cells", nargs="+")
    parser.add_argument("--frozen-transport", type=Path)
    parser.add_argument("--frozen-transport-sha256")
    parser.add_argument("--admitted-kernel-replay", type=Path)
    parser.add_argument(
        "--tuning-only",
        action="store_true",
        help="run frozen-transport public tuning without sequential HMC",
    )
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--broad-grid-root-seed", nargs=2, type=int)
    parser.add_argument("--initial-step-size", type=float)
    parser.add_argument("--broad-grid-screen-results", type=int, default=128)
    parser.add_argument("--broad-grid-result", type=Path)
    parser.add_argument("--broad-grid-result-sha256")
    parser.add_argument("--hmc-chunk-results", type=int, default=65)
    return parser.parse_args()


def _registry() -> Mapping[str, Any]:
    from bayesfilter.testing.neutra_model_registry_tf import validate_registry

    return validate_registry()


def _spec(cell_id: str) -> Any:
    from bayesfilter.testing.neutra_model_registry_tf import EXECUTABLE_CELLS

    for spec in EXECUTABLE_CELLS:
        if spec.cell_id == cell_id:
            return spec
    raise ValueError(f"cell is not executable: {cell_id}")


def _run_cell(args: argparse.Namespace) -> Mapping[str, Any]:
    if args.output_root is None or not args.cell:
        raise ValueError("cell action requires --output-root and --cell")
    spec = _spec(args.cell)
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    from bayesfilter.inference.neutra_end_to_end import (
        EndToEndConfig,
        run_neutra_end_to_end_cell,
    )

    return run_neutra_end_to_end_cell(
        spec=spec,
        config=EndToEndConfig(
            output_root=args.output_root,
            screen_steps=args.screen_steps,
            final_steps=args.final_steps,
            final_segment_steps=args.final_segment_steps,
            screen_only=args.screen_only,
            screen_result_path=args.screen_result,
            expected_screen_result_sha256=args.screen_result_sha256,
        ),
    )


def _run_preflight(args: argparse.Namespace) -> Mapping[str, Any]:
    if args.output_root is None or not args.cell or not args.recipe:
        raise ValueError("preflight requires --output-root, --cell, and --recipe")
    spec = _spec(args.cell)
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    from bayesfilter.inference.neutra_end_to_end import run_neutra_preflight_cell

    return run_neutra_preflight_cell(
        spec=spec,
        recipe_id=args.recipe,
        output_root=args.output_root,
        steps=args.screen_steps,
    )


def _run_training_throughput(args: argparse.Namespace) -> Mapping[str, Any]:
    if args.output_root is None or not args.cell or not args.recipe:
        raise ValueError(
            "training-throughput requires --output-root, --cell, and --recipe"
        )
    spec = _spec(args.cell)
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    from bayesfilter.inference.neutra_end_to_end import (
        run_neutra_training_throughput_cell,
    )

    return run_neutra_training_throughput_cell(
        spec=spec,
        recipe_id=args.recipe,
        output_root=args.output_root,
        repeated_steps=args.throughput_steps,
    )


def _run_frozen_validation(args: argparse.Namespace) -> Mapping[str, Any]:
    if (
        args.output_root is None
        or not args.cell
        or args.frozen_transport is None
        or not args.frozen_transport_sha256
    ):
        raise ValueError(
            "validate-frozen requires --output-root, --cell, "
            "--frozen-transport, and --frozen-transport-sha256"
        )
    if args.tuning_only and args.admitted_kernel_replay is not None:
        raise ValueError("--tuning-only cannot use --admitted-kernel-replay")
    spec = _spec(args.cell)
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    from bayesfilter.inference.neutra_end_to_end import (
        FrozenTransportValidationConfig,
        run_neutra_frozen_transport_validation_cell,
    )

    return run_neutra_frozen_transport_validation_cell(
        spec=spec,
        config=FrozenTransportValidationConfig(
            output_root=args.output_root,
            frozen_transport_path=args.frozen_transport,
            expected_frozen_transport_sha256=args.frozen_transport_sha256,
            admitted_kernel_replay_path=args.admitted_kernel_replay,
            tuning_only=args.tuning_only,
            seed_offset=args.seed_offset,
        ),
    )


def _run_frozen_broad_grid(args: argparse.Namespace) -> Mapping[str, Any]:
    if (
        args.output_root is None
        or not args.cell
        or args.frozen_transport is None
        or not args.frozen_transport_sha256
        or args.broad_grid_root_seed is None
    ):
        raise ValueError(
            "broad-grid-frozen requires --output-root, --cell, "
            "--frozen-transport, --frozen-transport-sha256, and "
            "--broad-grid-root-seed"
        )
    spec = _spec(args.cell)
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    from bayesfilter.inference.neutra_end_to_end import (
        FrozenTransportBroadGridConfig,
        run_neutra_frozen_transport_broad_grid_cell,
    )
    return run_neutra_frozen_transport_broad_grid_cell(
        spec=spec,
        config=FrozenTransportBroadGridConfig(
            output_root=args.output_root,
            frozen_transport_path=args.frozen_transport,
            expected_frozen_transport_sha256=args.frozen_transport_sha256,
            root_seed=tuple(args.broad_grid_root_seed),
            initial_step_size=args.initial_step_size,
            screen_results=args.broad_grid_screen_results,
        ),
    )


def _run_broad_grid_sequential(args: argparse.Namespace) -> Mapping[str, Any]:
    if (
        args.output_root is None
        or not args.cell
        or args.frozen_transport is None
        or not args.frozen_transport_sha256
        or args.broad_grid_result is None
        or not args.broad_grid_result_sha256
    ):
        raise ValueError(
            "sample-broad-grid-frozen requires --output-root, --cell, "
            "--frozen-transport, --frozen-transport-sha256, "
            "--broad-grid-result, and --broad-grid-result-sha256"
        )
    spec = _spec(args.cell)
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    from bayesfilter.inference.neutra_end_to_end import (
        BroadGridSequentialConfig,
        run_neutra_broad_grid_sequential_cell,
    )

    return run_neutra_broad_grid_sequential_cell(
        spec=spec,
        config=BroadGridSequentialConfig(
            output_root=args.output_root,
            frozen_transport_path=args.frozen_transport,
            expected_frozen_transport_sha256=args.frozen_transport_sha256,
            broad_grid_result_path=args.broad_grid_result,
            expected_broad_grid_result_sha256=args.broad_grid_result_sha256,
            chunk_results=args.hmc_chunk_results,
        ),
    )

def _run_campaign(args: argparse.Namespace) -> Mapping[str, Any]:
    if args.output_root is None:
        raise ValueError("campaign action requires --output-root")
    root = args.output_root
    if root.exists():
        raise FileExistsError(f"campaign root must be fresh: {root}")
    root.mkdir(parents=True)
    registry = _registry()
    _write_new_json(root / "registry.json", registry)
    launch_log_root = root / "launch-logs"
    launch_log_root.mkdir()
    _write_new_json(
        root / "campaign_state.json",
        {
            "schema": "bayesfilter.neutra.all_models.campaign_state.v1",
            "role": "diagnostic_execution_state_not_scientific_evidence",
            "status": "started",
            "pid": os.getpid(),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "terminal_result_authority": "aggregate_result.json",
            "cell_rows": (),
        },
    )
    executable_rows = list(registry["executable"])
    if args.cells:
        requested = tuple(dict.fromkeys(str(cell) for cell in args.cells))
        available = {str(row["cell_id"]) for row in executable_rows}
        unknown = tuple(cell for cell in requested if cell not in available)
        if unknown:
            raise ValueError(f"requested cells are not executable: {unknown}")
        requested_set = set(requested)
        executable_rows = [
            row for row in executable_rows if str(row["cell_id"]) in requested_set
        ]
        executable_rows.sort(key=lambda row: requested.index(str(row["cell_id"])))
    selected_cell_ids = tuple(str(row["cell_id"]) for row in executable_rows)
    started = time.monotonic()
    rows = []
    for registry_row in executable_rows:
        cell = str(registry_row["cell_id"])
        command = (
            sys.executable,
            str(Path(__file__).resolve()),
            "--action", "cell",
            "--cell", cell,
            "--output-root", str(root),
            "--screen-steps", str(args.screen_steps),
            "--final-steps", str(args.final_steps),
            "--final-segment-steps", str(args.final_segment_steps),
            *(("--screen-only",) if args.screen_only else ()),
        )
        log_path = launch_log_root / f"{cell}.log"
        with log_path.open("w", encoding="utf-8") as log_stream:
            child = subprocess.Popen(
                command,
                cwd=ROOT,
                text=True,
                stdout=log_stream,
                stderr=subprocess.STDOUT,
            )
            _write_new_json(
                root / f"{cell}-launch-state.json",
                {
                    "schema": "bayesfilter.neutra.all_models.launch_state.v1",
                    "role": "diagnostic_execution_state_not_scientific_evidence",
                    "cell_id": cell,
                    "pid": child.pid,
                    "command": command,
                    "log_path": str(log_path),
                    "status": "running",
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                },
            )
            returncode = child.wait()
        result_path = root / cell / "result.json"
        launch_state = {
            "schema": "bayesfilter.neutra.all_models.launch_state.v1",
            "role": "diagnostic_execution_state_not_scientific_evidence",
            "cell_id": cell,
            "pid": child.pid,
            "command": command,
            "log_path": str(log_path),
            "returncode": returncode,
            "status": "completed" if returncode == 0 else "failed",
            "result_exists": result_path.is_file(),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        _write_new_json(root / f"{cell}-launch-state-terminal.json", launch_state)
        if returncode != 0 or not result_path.is_file():
            log_text = log_path.read_text(encoding="utf-8", errors="replace")
            failure = {
                "cell_id": cell,
                "process_returncode": returncode,
                "launch_log": str(log_path),
                "output_tail": log_text[-4000:],
                "classification": "SHARED_OR_CELL_HARNESS_FAILURE",
                "continuation_veto": True,
            }
            _write_new_json(root / f"{cell}-launch-failure.json", failure)
            rows.append(failure)
            break
        result = _read_mapping(result_path)
        rows.append(
            {
                "cell_id": cell,
                "passed": result.get("passed") is True,
                "decision": result.get("decision"),
                "result_path": str(result_path),
                "continuation_veto": False,
            }
        )
    aggregate = {
        "schema": "bayesfilter.neutra.all_models.aggregate_result.v1",
        "campaign_id": "bayesfilter-neutra-all-executable-models-e2e-20260718",
        "campaign_scope_cell_ids": selected_cell_ids,
        "completed_executable_count": len(rows),
        "declared_executable_count": len(executable_rows),
        "passed_cell_count": sum(row.get("passed") is True for row in rows),
        "cell_rows": rows,
        "blocked_inventory": registry["blocked"],
        "blocked_inventory_count": len(registry["blocked"]),
        "all_executable_completed": len(rows) == len(executable_rows)
        and not any(row.get("continuation_veto") is True for row in rows),
        "statistically_supported_ranking": False,
        "descriptive_only": "runtime, acceptance, posterior moments, and tail magnitudes",
        "default_readiness": False,
        "elapsed_seconds": time.monotonic() - started,
        "plan_path": str(PLAN_PATH),
        "nonclaims": (
            "blocked inventory cells were not launched and are not NeuTra failures",
            "one seed per executable target only",
            "no cross-model ranking, universal validity, or default-readiness claim",
        ),
    }
    _write_new_json(root / "aggregate_result.json", aggregate)
    _write_new_json(
        root / "campaign_state_terminal.json",
        {
            "schema": "bayesfilter.neutra.all_models.campaign_state.v1",
            "role": "diagnostic_execution_state_not_scientific_evidence",
            "status": "completed" if aggregate["all_executable_completed"] else "stopped",
            "pid": os.getpid(),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "terminal_result_authority": "aggregate_result.json",
            "completed_executable_count": len(rows),
            "declared_executable_count": len(executable_rows),
        },
    )
    return aggregate


def _read_mapping(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"expected JSON mapping: {path}")
    return value


def _write_new_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")


def main() -> None:
    args = _args()
    if args.action == "registry":
        print(json.dumps(_registry(), indent=2, sort_keys=True))
        return
    if args.action == "preflight":
        payload = _run_preflight(args)
    elif args.action == "training-throughput":
        payload = _run_training_throughput(args)
    elif args.action == "validate-frozen":
        payload = _run_frozen_validation(args)
    elif args.action == "broad-grid-frozen":
        payload = _run_frozen_broad_grid(args)
    elif args.action == "sample-broad-grid-frozen":
        payload = _run_broad_grid_sequential(args)
    elif args.action == "cell":
        payload = _run_cell(args)
    else:
        payload = _run_campaign(args)
    if args.action in {
        "preflight",
        "training-throughput",
        "validate-frozen",
        "broad-grid-frozen",
        "sample-broad-grid-frozen",
        "cell",
    }:
        payload = {
            "cell_id": payload.get("cell_id"),
            "passed": payload.get("passed"),
            "decision": payload.get("decision"),
            "output_root": payload.get("output_root"),
        }
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
