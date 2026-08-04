#!/usr/bin/env python3
"""Select one viable T1 centered-density arm from validation artifacts only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping


SCHEMA = "bayesfilter.zhao_cui_austria_sir_parameter_density_selection.v1"
PILOT_SCHEMA = "bayesfilter.zhao_cui_austria_sir_parameter_density_pilot.v1"


def select(pilot_root: Path) -> Mapping[str, object]:
    rows = []
    for path in sorted(Path(pilot_root).glob("*/result.json")):
        payload = json.loads(path.read_text())
        if payload.get("schema_version") != PILOT_SCHEMA:
            continue
        validation = payload.get("validation")
        metrics = (
            validation["selector_metrics"]
            if isinstance(validation, Mapping)
            else {
                "maximum_standardized_score_residual": float("inf"),
                "mean_paired_shape_ratio": float("inf"),
                "maximum_mass_standardized_residual": float("inf"),
            }
        )
        rows.append(
            {
                "arm_id": payload["arm_id"],
                "result_path": path.as_posix(),
                "artifact_directory": payload["artifact_directory"],
                "child_identity": payload["child_identity"],
                "rank": int(payload["arm"]["rank"]),
                "viable": payload.get("status") == "VIABLE_T1_PARAMETER_DENSITY_ARM"
                and bool(payload["gates"]["passed"]),
                "maximum_standardized_score_residual": float(
                    metrics["maximum_standardized_score_residual"]
                ),
                "mean_paired_shape_ratio": float(metrics["mean_paired_shape_ratio"]),
                "maximum_mass_standardized_residual": float(
                    metrics["maximum_mass_standardized_residual"]
                ),
            }
        )
    viable = [row for row in rows if row["viable"]]
    selected = None
    if viable:
        selected = min(
            viable,
            key=lambda row: (
                row["maximum_standardized_score_residual"],
                row["mean_paired_shape_ratio"],
                row["maximum_mass_standardized_residual"],
                row["rank"],
                row["arm_id"],
            ),
        )
    return {
        "schema_version": SCHEMA,
        "status": "SELECTED_T1_PARAMETER_DENSITY_ARM" if selected else "NO_VIABLE_T1_PARAMETER_DENSITY_ARM",
        "pilot_root": Path(pilot_root).as_posix(),
        "rows": rows,
        "selected": selected,
        "selection_rule": [
            "minimum maximum standardized validation likelihood/prefix score residual",
            "minimum mean paired non-origin shape ratio",
            "minimum maximum mass standardized residual",
            "minimum rank",
            "lexicographic arm id",
        ],
        "nonclaims": [
            "deterministic validation selection does not establish statistical superiority",
            "no score admission before untouched claim",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = select(args.pilot_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if payload["selected"] is None:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
