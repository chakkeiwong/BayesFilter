#!/usr/bin/env python3
"""Validate SVX-ZC values with HMC-compatible center-frozen UKF cores."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("MPLCONFIGDIR", "/tmp")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tensorflow as tf

import docs.benchmarks.run_contract_e_tp_phase6_zhao_cui_comparator as comparator
import docs.benchmarks.run_svx_zc_capacity_self_convergence_tuning_20260801 as tuning
from bayesfilter.highdim.capacity_tuning import compare_likelihood_values
from bayesfilter.highdim.zhao_cui_fixed_adjacent_tt_tf import (
    scalar_adjacent_state_fixed_tt_value,
)


CENTER = tf.constant([0.2533471031357997, -0.916290731874155], tf.float64)
POINTS = (("validation1", (-0.05, 0.0)), ("validation2", (0.05, 0.0)),
          ("validation3", (0.0, -0.05)), ("validation4", (0.0, 0.05)))
CAPACITIES = ((10, 2, 25), (12, 2, 25), (10, 4, 25), (10, 2, 33))


def _json_value(value: Any) -> Any:
    if isinstance(value, tf.Tensor):
        if value.shape.rank == 0:
            raw = value.numpy()
            return bool(raw) if value.dtype == tf.bool else float(raw)
        return [_json_value(item) for item in tf.unstack(value)]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return value


def run(output_root: Path) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite existing root: {output_root}")
    output_root.mkdir(parents=True)
    started = time.perf_counter()
    model, _theta, observations = comparator._row_inputs("actual_sv", 10)
    raw = tf.convert_to_tensor(comparator._sv_dataset(81101)["observations"], tf.float64)[:10]
    frozen = {}
    for degree, rank, order in CAPACITIES:
        initial, adjacent, manifest = comparator._ukf_initial_cores(
            model=model, theta=CENTER, raw_observations=raw, degree=degree,
            order=order, rank=rank, coordinate_half_width=8.0,
        )
        seed = f"svx-zc-hmc-frozen-init-20260802:d{degree}:r{rank}:o{order}"
        frozen[(degree, rank, order)] = (
            comparator._comparator_config(
                degree=degree, order=order, rank=rank, seed=seed,
                transition_before_first_observation=False,
                coordinate_half_width=8.0, density_tau=0.0,
                initial_cores=initial, adjacent_initial_cores=adjacent,
                initialization_rule=str(manifest["initializer_rule"]),
            ),
            manifest,
            seed,
        )
    records = {}
    validation = {}
    all_pass = True
    for point_id, offset in POINTS:
        theta = CENTER + tf.constant(offset, tf.float64)
        cells = {}
        for capacity, (config, initializer, seed) in frozen.items():
            degree, rank, order = capacity
            result = scalar_adjacent_state_fixed_tt_value(
                model, theta, observations, config,
                fixture_id=f"svx-zc-hmc-frozen-init.{point_id}.d{degree}.r{rank}.o{order}",
                branch_seed_prefix=seed,
            )
            diagnostics = tuning._step_diagnostics(result)
            record = {
                "point_id": point_id,
                "theta": _json_value(theta),
                "capacity": {"degree": degree, "rank": rank, "order": order},
                "value": float(result.log_likelihood.numpy()),
                "increments": _json_value(result.log_increments),
                "invariant_pass": bool(diagnostics["invariant_pass"]),
                "diagnostics": _json_value(diagnostics),
                "initializer_parameter_role": "center_frozen_not_runtime_retuned",
                "initializer_center": _json_value(CENTER),
                "initial_core_hash": initializer["initial_core_hash"],
                "adjacent_core_hash": initializer["adjacent_core_hash"],
            }
            cells[capacity] = record
            records[f"{point_id}:d{degree}:r{rank}:o{order}"] = record
        base = cells[(10, 2, 25)]
        comparisons = {
            "degree": compare_likelihood_values(
                low_value=base["value"], high_value=cells[(12, 2, 25)]["value"],
                low_increments=base["increments"], high_increments=cells[(12, 2, 25)]["increments"],
                policy=tuning.POLICY,
            ),
            "rank": compare_likelihood_values(
                low_value=base["value"], high_value=cells[(10, 4, 25)]["value"],
                low_increments=base["increments"], high_increments=cells[(10, 4, 25)]["increments"],
                policy=tuning.POLICY,
            ),
            "order": compare_likelihood_values(
                low_value=base["value"], high_value=cells[(10, 2, 33)]["value"],
                low_increments=base["increments"], high_increments=cells[(10, 2, 33)]["increments"],
                policy=tuning.POLICY,
            ),
        }
        point_pass = all(row["invariant_pass"] for row in cells.values()) and all(
            bool(item["stable"]) for item in comparisons.values()
        )
        all_pass = all_pass and point_pass
        validation[point_id] = {
            "theta": _json_value(theta),
            "value_stability_pass": point_pass,
            "comparisons": comparisons,
        }
    result = {
        "schema": "bayesfilter.svx_zc.frozen_initializer_value_validation.v1",
        "status": "SELF_CONVERGED_FIXED_INITIALIZER_VALUE" if all_pass else "FROZEN_INITIALIZER_VALUE_BLOCKED",
        "value_only_decision": True,
        "score_role": "not_computed_diagnostic_only",
        "validation": validation,
        "records": records,
        "run_manifest": {
            "schema": "bayesfilter.svx_zc.frozen_initializer_value_validation_manifest.v1",
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "command": " ".join(sys.argv),
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "tensorflow_version": tf.__version__,
            "device": "CPU-only; CUDA_VISIBLE_DEVICES=-1",
            "dtype": "float64",
            "horizon": 10,
            "data_seed": 81101,
            "initializer_policy": "UKF cores built once at center per capacity and frozen",
            "executed_cell_count": len(records),
            "completed_wall_time_seconds": time.perf_counter() - started,
            "output_root": str(output_root.relative_to(ROOT)),
            "plan": "docs/plans/bayesfilter-svx-zc-value-validation-neutra-hmc-continuation-plan-2026-08-02.md",
        },
        "nonclaims": ["not exact likelihood", "not score validation", "not HMC convergence", "not cross-scope transfer"],
    }
    (output_root / "result.json").write_text(json.dumps(_json_value(result), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_root / "run_manifest.json").write_text(json.dumps(_json_value(result["run_manifest"]), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_root if args.output_root.is_absolute() else ROOT / args.output_root
    result = run(root)
    print(json.dumps({"status": result["status"], "output_root": str(root)}))


if __name__ == "__main__":
    main()
