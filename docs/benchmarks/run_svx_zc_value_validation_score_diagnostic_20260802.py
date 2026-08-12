#!/usr/bin/env python3
"""Record SVX-ZC score direction diagnostics at frozen value validation points."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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
from bayesfilter.highdim.zhao_cui_fixed_adjacent_tt_tf import (
    scalar_adjacent_state_fixed_tt_score,
)


CELLS = (("center", (0.0, 0.0)), ("validation1", (-0.05, 0.0)),
         ("validation2", (0.05, 0.0)), ("validation3", (0.0, -0.05)),
         ("validation4", (0.0, 0.05)))
FD_STEPS = (1.0e-2, 3.0e-3, 1.0e-3, 3.0e-4)
DEGREE = 10
RANK = 2
ORDER = 25
HORIZON = 10
ROW = "actual_sv"
DATA_SEED = 81101
CENTER = tf.constant([0.2533471031357997, -0.916290731874155], tf.float64)


def _json_value(value: Any) -> Any:
    if isinstance(value, tf.Tensor):
        if value.shape.rank == 0:
            value = value.numpy()
            if isinstance(value, (float, int)):
                return float(value) if isinstance(value, float) else int(value)
        else:
            return [_json_value(item) for item in tf.unstack(value)]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    return value


def _git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def _fd_summary(result: Any) -> dict[str, Any]:
    rows = []
    valid = True
    hashes: set[str] = set()
    for row in result.finite_difference_table.rows:
        hashes.update((row.branch_hash_base, row.branch_hash_plus, row.branch_hash_minus))
        status = row.row_status.value
        valid = valid and status == "VALID"
        rows.append({
            "parameter_index": int(row.parameter_index),
            "h": float(row.h.numpy()),
            "score": float(row.analytic_gradient.numpy()),
            "finite_difference": float(row.centered_difference.numpy()),
            "relative_error": float(row.rel_error.numpy()),
            "row_status": status,
            "branch_hash_base": row.branch_hash_base,
            "branch_hash_plus": row.branch_hash_plus,
            "branch_hash_minus": row.branch_hash_minus,
        })
    return {"rows": rows, "all_rows_valid": valid, "branch_hash_count": len(hashes)}


def run(output_root: Path) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite existing root: {output_root}")
    output_root.mkdir(parents=True)
    started = time.perf_counter()
    model, _theta, observations = comparator._row_inputs(ROW, HORIZON)
    raw = tf.convert_to_tensor(comparator._sv_dataset(DATA_SEED)["observations"], tf.float64)[:HORIZON]
    rows = []
    for point_id, offset in CELLS:
        theta = CENTER + tf.constant(offset, tf.float64)
        initial, adjacent, ukf = comparator._ukf_initial_cores(
            model=model, theta=theta, raw_observations=raw, degree=DEGREE,
            order=ORDER, rank=RANK, coordinate_half_width=8.0,
        )
        seed = f"svx-zc-value-score-diagnostic-20260802:{point_id}"
        config = comparator._comparator_config(
            degree=DEGREE, order=ORDER, rank=RANK, seed=seed,
            transition_before_first_observation=False, coordinate_half_width=8.0,
            density_tau=0.0, initial_cores=initial, adjacent_initial_cores=adjacent,
            initialization_rule=str(ukf["initializer_rule"]),
        )
        result = scalar_adjacent_state_fixed_tt_score(
            model, theta, observations, config, finite_difference_h=FD_STEPS,
            fixture_id=f"svx-zc-value-score-diagnostic.{point_id}",
            branch_seed_prefix=seed,
        )
        score = [float(item) for item in result.score.numpy()]
        rows.append({
            "point_id": point_id,
            "theta": _json_value(theta),
            "value": float(result.log_likelihood.numpy()),
            "score": score,
            "score_sign": [0 if item == 0.0 else (1 if item > 0.0 else -1) for item in score],
            "compatibility_hash": result.diagnostics["compatibility_hash"],
            "finite_difference": _fd_summary(result),
            "ukf_initializer_rule": ukf["initializer_rule"],
        })
    center_score = tf.constant(rows[0]["score"], tf.float64)
    for row in rows:
        score = tf.constant(row["score"], tf.float64)
        denom = tf.linalg.norm(center_score) * tf.linalg.norm(score)
        cosine = tf.where(denom > 0.0, tf.reduce_sum(center_score * score) / denom, tf.constant(float("nan"), tf.float64))
        row["direction_vs_center"] = {
            "component_sign_agreement": [a == b for a, b in zip(row["score_sign"], rows[0]["score_sign"])],
            "cosine_similarity": float(cosine.numpy()),
            "score_direction_role": "diagnostic_only",
        }
    result = {
        "schema": "bayesfilter.svx_zc.value_validation.score_diagnostic.v1",
        "status": "DIAGNOSTIC_COMPLETE",
        "value_promotion_affected": False,
        "cells": rows,
        "nonclaims": ["not score accuracy", "not score convergence", "not HMC readiness", "not value veto evidence"],
        "run_manifest": {
            "schema": "bayesfilter.svx_zc.value_validation.score_diagnostic_manifest.v1",
            "git_commit": _git_commit(),
            "command": " ".join(sys.argv),
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "tensorflow_version": tf.__version__,
            "device": "CPU-only; CUDA_VISIBLE_DEVICES=-1",
            "dtype": "float64",
            "data_seed": DATA_SEED,
            "horizon": HORIZON,
            "degree": DEGREE,
            "rank": RANK,
            "order": ORDER,
            "finite_difference_h": FD_STEPS,
            "completed_wall_time_seconds": time.perf_counter() - started,
            "output_root": str(output_root.relative_to(ROOT)),
            "plan": "docs/plans/bayesfilter-svx-zc-value-validation-neutra-hmc-continuation-plan-2026-08-02.md",
        },
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
