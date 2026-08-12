#!/usr/bin/env python3
"""Run the shared common NeuTra broad-grid procedure for one cell.

This is the single supported entry point for frozen-transport NeuTra
broad-grid tuning. The common/default tuning procedure is the repaired
state-continuing epsilon-repair route; the legacy operational broad-grid route
remains available only as an explicit reference mode via `--variant`.
Registry cells resolve from ``bayesfilter.testing.neutra_model_registry_tf``;
the KSC gaussian-sum lane resolves from its dedicated runner spec.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bayesfilter.testing.neutra_model_registry_tf import EXECUTABLE_CELLS
from bayesfilter.inference.neutra_shared_procedure import (
    DEFAULT_COMMON_VARIANT,
    OPERATIONAL_BROAD_GRID_V1,
    STATE_CONTINUING_EPSILON_REPAIR_V1,
)


KSC_RUNNER = ROOT / "docs/benchmarks/run_neutra_ksc_gaussian_sum_ukf_end_to_end_20260802.py"

STATUS_KEY_PRESETS = {
    "ukf": (
        "status_code",
        "valid_pre_regularized_score",
        "floor_count_value",
        "min_innovation_eigenvalue",
        "innovation_condition_estimate",
    ),
    "basic": ("status_code", "valid_pre_regularized_score"),
}


def _resolve_spec(cell_id: str):
    if cell_id == "KSC-UKF-GAUSSIAN-SUM-T20":
        spec_loader = importlib.util.spec_from_file_location(
            "ksc_gaussian_sum_end_to_end_runner", KSC_RUNNER
        )
        if spec_loader is None or spec_loader.loader is None:
            raise RuntimeError("cannot load the KSC gaussian-sum runner spec")
        module = importlib.util.module_from_spec(spec_loader)
        spec_loader.loader.exec_module(module)
        return module.build_spec()

    for spec in EXECUTABLE_CELLS:
        if spec.cell_id == cell_id:
            return spec
    known = ", ".join(item.cell_id for item in EXECUTABLE_CELLS)
    raise ValueError(
        f"unknown cell {cell_id!r}; executable registry cells: {known}, "
        "plus KSC-UKF-GAUSSIAN-SUM-T20"
    )


def _status_keys(raw: str) -> tuple[str, ...]:
    if raw in STATUS_KEY_PRESETS:
        return STATUS_KEY_PRESETS[raw]
    keys = tuple(item.strip() for item in raw.split(",") if item.strip())
    if not keys:
        raise ValueError("required-status-keys must be 'ukf', 'basic', or a CSV list")
    return keys


def _variant(raw: str | None) -> str:
    if raw is None:
        return DEFAULT_COMMON_VARIANT
    text = str(raw).strip()
    if text not in (OPERATIONAL_BROAD_GRID_V1, STATE_CONTINUING_EPSILON_REPAIR_V1):
        raise ValueError(
            "variant must be one of operational_broad_grid_v1 or "
            "state_continuing_epsilon_repair_v1"
        )
    return text


def _spec_status_keys(spec: object) -> tuple[str, ...]:
    keys = getattr(spec, "common_tuning_status_keys", None)
    if keys:
        return tuple(str(item) for item in keys)
    return STATUS_KEY_PRESETS["ukf"]


def _spec_initial_epsilon_by_l(spec: object) -> dict[int, float] | None:
    warm = getattr(spec, "common_tuning_initial_epsilon_by_l", None)
    if warm is None:
        return None
    return {int(key): float(value) for key, value in dict(warm).items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", required=True)
    parser.add_argument(
        "--variant",
        choices=(OPERATIONAL_BROAD_GRID_V1, STATE_CONTINUING_EPSILON_REPAIR_V1),
        help=(
            "Reviewed tuning procedure variant. Omit to use the common repaired "
            "default; pass operational_broad_grid_v1 only for legacy/reference "
            "testing."
        ),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--frozen-transport", type=Path, required=True)
    parser.add_argument("--frozen-transport-sha256", required=True)
    parser.add_argument("--root-seed", nargs=2, type=int, required=True)
    parser.add_argument("--initial-step-size", type=float)
    parser.add_argument("--screen-results", type=int, default=128)
    parser.add_argument("--final-screen-results", type=int, default=96)
    parser.add_argument("--final-screen-burnin", type=int, default=8)
    parser.add_argument("--max-epsilon-repairs", type=int, default=3)
    parser.add_argument(
        "--initial-epsilon-by-l",
        help="JSON object of warm-start epsilons per leapfrog count, e.g. "
        '\'{"3": 0.87, "5": 0.84}\' (warm-start hypothesis only)',
    )
    parser.add_argument(
        "--required-status-keys",
        default="auto",
        help="Use auto to read the shared target metadata, or pass ukf/basic/CSV",
    )
    parser.add_argument("--launch-sequential", action="store_true")
    args = parser.parse_args()

    spec = _resolve_spec(args.cell)
    variant = _variant(args.variant)
    warm = _spec_initial_epsilon_by_l(spec)
    if args.initial_epsilon_by_l:
        warm = {
            int(key): float(value)
            for key, value in json.loads(args.initial_epsilon_by_l).items()
        }

    from bayesfilter.inference.neutra_shared_procedure import (
        SharedNeuTraProcedureConfig,
        run_shared_neutra_procedure,
    )

    result = run_shared_neutra_procedure(
        spec=spec,
        config=SharedNeuTraProcedureConfig(
            output_root=args.output_root,
            frozen_transport_path=args.frozen_transport,
            expected_frozen_transport_sha256=args.frozen_transport_sha256,
            root_seed=tuple(args.root_seed),
            variant=variant,
            launch_sequential=args.launch_sequential,
            initial_step_size=args.initial_step_size,
            screen_results=args.screen_results,
            final_screen_results=args.final_screen_results,
            final_screen_burnin=args.final_screen_burnin,
            max_epsilon_repairs=args.max_epsilon_repairs,
            initial_epsilon_by_l=warm,
            required_status_keys=(
                _status_keys(args.required_status_keys)
                if args.required_status_keys != "auto"
                else _spec_status_keys(spec)
            ),
        ),
    )
    print(
        {
            "cell": args.cell,
            "procedure_variant": result.get("procedure_variant"),
            "decision": result.get("decision"),
            "passed": result.get("passed"),
            "target_signature": spec.target_signature,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
