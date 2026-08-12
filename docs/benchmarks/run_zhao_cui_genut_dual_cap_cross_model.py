#!/usr/bin/env python3
"""Test the opt-in dual-cap GenUT map on the non-Austria model lanes."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
from typing import Any

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tensorflow as tf

from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
from docs.benchmarks import run_moment_retuned_genut_whole_leaderboard as base
from docs.benchmarks import run_pairwise_moment_genut_cross_model_trial as refs


PLAN = ROOT / "docs/plans/bayesfilter-zhao-cui-genut-dual-cap-cross-model-plan-2026-08-07.md"
SCHEMA = "bayesfilter.zhao_cui_genut_dual_cap_cross_model.v1"
OUTPUT_ROOT = ROOT / "docs/benchmarks/artifacts/zhao_cui_genut_dual_cap_cross_model_20260807"
N = base.N
TUNING_SEEDS = (98401, 98402)
CLAIM_SEEDS = tuple(range(98201, 98207))
FD_STEP = 1.0e-3
FD_ABSOLUTE_LIMIT = 0.08
FD_NORMALIZED_LIMIT = 0.03
RESIDUAL_TOLERANCE = 5.0e-4
CAP = 0.98
CAP_POWER = 8


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _safe(value: Any) -> Any:
    if hasattr(value, "numpy"):
        value = value.numpy()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _baseline_controls(row_id: str, target: dict[str, Any]) -> dict[str, Any]:
    row = refs._prior_row(row_id, "genut")
    if row["scope"]["source_observation_sha256"] != target["source_observation_sha256"]:
        raise RuntimeError(f"{row_id} prior target hash mismatch")
    if row["scope"]["event_order"] != target["event_order"]:
        raise RuntimeError(f"{row_id} prior event-order mismatch")
    if row["scope"]["particle_count"] != N:
        raise RuntimeError(f"{row_id} prior particle-count mismatch")
    controls = dict(row["tuning"]["selected_controls"])
    controls.update(
        {
            "pairwise_moment_correction_steps": 0,
            "pairwise_moment_strength": 0.0,
            "pairwise_moment_floor": 1.0e-5,
            "pairwise_particle_rms_cap": 0.0,
            "coordinatewise_standardized_cap": 0.0,
            "coordinatewise_standardized_cap_power": CAP_POWER,
        }
    )
    return controls


def _arms(row_id: str, baseline: dict[str, Any]) -> list[dict[str, Any]]:
    pair_strength = 0.02 if row_id != "predator_prey_T20" else 0.05

    def arm(name: str, *, radial: float = 0.0, coordinate: float = 0.0) -> dict[str, Any]:
        controls = dict(baseline)
        controls.update(
            {
                "pairwise_moment_correction_steps": 4,
                "pairwise_moment_strength": pair_strength,
                "pairwise_particle_rms_cap": radial,
                "coordinatewise_standardized_cap": coordinate,
                "coordinatewise_standardized_cap_power": CAP_POWER,
            }
        )
        return {"arm_id": name, "controls": controls}

    return [
        {"arm_id": "baseline_diagonal", "controls": baseline},
        arm("pairwise_uncapped"),
        arm("pairwise_radial_cap", radial=2.0),
        arm("pairwise_coordinate_cap", coordinate=CAP),
        arm("dual_cap", radial=2.0, coordinate=CAP),
    ]


def _make_evaluator(target: dict[str, Any], controls: dict[str, Any]):
    return base._make_evaluator(
        adapter=target["adapter"],
        horizon=int(target["observations"].shape[0]),
        observation_dim=target["observation_dim"],
        state_dim=target["state_dim"],
        parameter_dim=target["parameter_dim"],
        transition_before_first_observation=target["transition_before"],
        controls=controls,
    )


def _evaluate(evaluator: Any, target: dict[str, Any], observations: tf.Tensor, seed: int) -> dict[str, Any]:
    return base._evaluate(evaluator, target["theta"], observations, seed, target["design"])
    return base._evaluate(evaluator, target["theta"], observations, seed, target["design"])


def _valid(row: dict[str, Any]) -> bool:
    return bool(
        base._valid(row)
        and max(
            row["max_mean_residual"],
            row["max_row_residual"],
            row["max_col_residual"],
            row["score_increment_sum_residual"],
        )
        <= RESIDUAL_TOLERANCE
    )


def _calibration(target: dict[str, Any], arm: dict[str, Any], evaluator: Any) -> dict[str, Any]:
    rows = []
    for dataset in target["calibration"]:
        for seed in TUNING_SEEDS:
            rows.append(_evaluate(evaluator, target, dataset, seed))
    valid = all(_valid(row) for row in rows)
    return {
        "valid": valid,
        "rows": rows,
        "mean_displacement": (
            statistics.mean(row["maximum_normalized_shape_displacement"] for row in rows)
            if rows else None
        ),
        "mean_cap_displacement": (
            statistics.mean(row["mean_coordinatewise_cap_displacement"] for row in rows)
            if rows else None
        ),
        "min_cap_derivative": (
            min(row["minimum_coordinatewise_cap_derivative"] for row in rows)
            if rows else None
        ),
    }


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = ("value", *[f"score_{i}" for i in range(len(rows[0]["score"]))])
    vectors = [[row["value"], *row["score"]] for row in rows]
    result: dict[str, Any] = {"count": len(rows), "all_valid": all(_valid(row) for row in rows)}
    for index, label in enumerate(labels):
        values = [vector[index] for vector in vectors]
        sd = statistics.stdev(values) if len(values) > 1 else 0.0
        result[label] = {
            "mean": statistics.mean(values),
            "sample_sd": sd,
            "mcse": sd / math.sqrt(len(values)) if values else None,
        }
    result["mean_cap_displacement"] = statistics.mean(
        row["mean_coordinatewise_cap_displacement"] for row in rows
    )
    result["max_cap_active_fraction"] = max(
        row["fraction_coordinatewise_cap_active"] for row in rows
    )
    result["min_cap_derivative"] = min(
        row["minimum_coordinatewise_cap_derivative"] for row in rows
    )
    return result


def _paired(baseline: list[dict[str, Any]], candidate: list[dict[str, Any]]) -> dict[str, Any]:
    labels = ("value", *[f"score_{i}" for i in range(len(baseline[0]["score"]))])
    output = {}
    for index, label in enumerate(labels):
        differences = []
        for left, right in zip(baseline, candidate):
            differences.append(
                ([right["value"], *right["score"]][index]
                 - [left["value"], *left["score"]][index])
            )
        sd = statistics.stdev(differences) if len(differences) > 1 else 0.0
        output[label] = {
            "mean_candidate_minus_baseline": statistics.mean(differences),
            "sample_sd": sd,
            "mcse": sd / math.sqrt(len(differences)) if differences else None,
            "negative_count": sum(v < 0.0 for v in differences),
            "positive_count": sum(v > 0.0 for v in differences),
        }
    return output


def _fd(target: dict[str, Any], evaluator: Any, seed: int) -> list[dict[str, Any]]:
    initial, process = base._noise(seed, int(target["observations"].shape[0]), target["state_dim"])
    theta = target["theta"]
    origin = evaluator(theta, target["observations"], initial, process, target["design"])
    rows = []
    for index in range(target["parameter_dim"]):
        direction = tf.one_hot(index, target["parameter_dim"], dtype=tf.float32)
        plus = evaluator(theta + FD_STEP * direction, target["observations"], initial, process, target["design"])[0]
        minus = evaluator(theta - FD_STEP * direction, target["observations"], initial, process, target["design"])[0]
        observed = (plus - minus) / (2.0 * FD_STEP)
        expected = origin[1][index]
        absolute = tf.abs(observed - expected)
        normalized = absolute / tf.maximum(tf.abs(expected), 1.0)
        row = {
            "parameter": index,
            "manual_score": float(expected.numpy()),
            "finite_difference": float(observed.numpy()),
            "absolute_residual": float(absolute.numpy()),
            "normalized_residual": float(normalized.numpy()),
        }
        row["gate_pass"] = bool(
            math.isfinite(row["absolute_residual"])
            and math.isfinite(row["normalized_residual"])
            and row["absolute_residual"] <= FD_ABSOLUTE_LIMIT
            and row["normalized_residual"] <= FD_NORMALIZED_LIMIT
        )
        rows.append(row)
    return rows


def _reference(row_id: str, target: dict[str, Any]) -> dict[str, Any] | None:
    if row_id == "lgssm_T50":
        return refs._lgssm_reference(target)
    if row_id == "ksc_sv_T10":
        return refs._ksc_dense_reference(target, smoke_only=False)
    return None


def _render(payload: dict[str, Any]) -> str:
    lines = [
        "# Zhao-Cui/GenUT Dual-Cap Cross-Model Result",
        "",
        f"Status: `{payload['status']}`",
        "",
        "| Model | Selected arm | Valid rows | Value mean (SD) | Score SDs | FD gate |",
        "|---|---|---:|---:|---|---|",
    ]
    for model in payload["models"]:
        summary = model["selected"]["summary"]
        score_sds = [summary[f"score_{i}"]["sample_sd"] for i in range(model["parameter_dimension"])]
        lines.append(
            f"| {model['row_id']} | {model['selected']['arm_id']} | "
            f"{summary['count']} | {summary['value']['mean']:.7g} ({summary['value']['sample_sd']:.4g}) | "
            f"`{score_sds}` | {model['fd']['all_pass']} |"
        )
    lines += [
        "",
        "The coordinatewise standardized cap is an extension/invention outside the Austria bounded-teacher chart.",
        "Rows are numerical-feasibility evidence; few-seed differences are descriptive only.",
    ]
    return "\n".join(lines) + "\n"


def run(output_root: Path) -> dict[str, Any]:
    started = time.perf_counter()
    output_root.mkdir(parents=True, exist_ok=False)
    memory_policy = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    tf.config.set_soft_device_placement(False)
    tf.config.experimental.enable_tensor_float_32_execution(True)
    logical = tf.config.list_logical_devices("GPU")
    if not logical:
        raise RuntimeError("dual-cap cross-model run requires a logical GPU")
    targets = base._build_targets()
    models = []
    for row_id in ("lgssm_T50", "ksc_sv_T10", "predator_prey_T20"):
        target = targets[row_id]
        baseline = _baseline_controls(row_id, target)
        arms = _arms(row_id, baseline)
        calibration = []
        for arm in arms:
            evaluator = _make_evaluator(target, arm["controls"])
            calibration.append({**arm, "calibration": _calibration(target, arm, evaluator), "evaluator": evaluator})
        valid_cap_arms = [
            item for item in calibration
            if item["arm_id"] in {"pairwise_coordinate_cap", "dual_cap"}
            and item["calibration"]["valid"]
        ]
        if not valid_cap_arms:
            raise RuntimeError(f"{row_id}: no coordinate-cap arm passed calibration")
        selected = min(
            valid_cap_arms,
            key=lambda item: (
                item["calibration"]["mean_cap_displacement"],
                item["calibration"]["mean_displacement"],
                item["arm_id"],
            ),
        )
        selected_evaluator = selected["evaluator"]
        baseline_evaluator = _make_evaluator(target, baseline)
        claim_rows = [
            _evaluate(selected_evaluator, target, target["observations"], seed)
            for seed in CLAIM_SEEDS
        ]
        baseline_rows = [
            _evaluate(baseline_evaluator, target, target["observations"], seed)
            for seed in CLAIM_SEEDS
        ]
        if not all(_valid(row) for row in baseline_rows + claim_rows):
            raise RuntimeError(f"{row_id}: claim finite-program veto")
        fd = _fd(target, selected_evaluator, CLAIM_SEEDS[0])
        reference = _reference(row_id, target)
        selected_payload = {
            "arm_id": selected["arm_id"],
            "controls": selected["controls"],
            "calibration": selected["calibration"],
            "rows": claim_rows,
            "summary": _summary(claim_rows),
            "paired_vs_baseline": _paired(baseline_rows, claim_rows),
        }
        models.append({
            "row_id": row_id,
            "model_id": target["model_id"],
            "horizon": int(target["observations"].shape[0]),
            "state_dimension": target["state_dim"],
            "parameter_dimension": target["parameter_dim"],
            "event_order": target["event_order"],
            "source_observation_sha256": target["source_observation_sha256"],
            "calibration_arms": [
                {key: value for key, value in item.items() if key != "evaluator"}
                for item in calibration
            ],
            "selected": selected_payload,
            "baseline": {"controls": baseline, "rows": baseline_rows, "summary": _summary(baseline_rows)},
            "fd": {"rows": fd, "all_pass": all(row["gate_pass"] for row in fd)},
            "reference": reference,
            "pairwise_constraint_count": target["state_dim"] * (target["state_dim"] - 1),
            "classification": "extension_or_invention_outside_austria_bounded_teacher",
        })
    payload = {
        "schema": SCHEMA,
        "status": "PASS_NUMERICAL_FEASIBILITY" if all(model["fd"]["all_pass"] for model in models) else "FD_VETO",
        "plan": str(PLAN.relative_to(ROOT)),
        "models": models,
        "configuration": {
            "particle_count": N,
            "tuning_seeds": TUNING_SEEDS,
            "claim_seeds": CLAIM_SEEDS,
            "dtype": "float32",
            "tf32": True,
            "jit_compile": True,
            "coordinatewise_standardized_cap": CAP,
            "coordinatewise_standardized_cap_power": CAP_POWER,
            "fd_step": FD_STEP,
            "fd_absolute_limit": FD_ABSOLUTE_LIMIT,
            "fd_normalized_limit": FD_NORMALIZED_LIMIT,
        },
        "device": {
            "logical_devices": [device.name for device in logical],
            "trust_basis": "owner_designated_managed_session_visible_gpu_trusted",
        },
        "memory_policy": dict(memory_policy),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "wall_time_seconds": time.perf_counter() - started,
        "run_manifest": {
            "command": [sys.executable, *sys.argv],
            "environment": sys.prefix,
            "host": platform.node(),
            "python": platform.python_version(),
            "tensorflow": tf.__version__,
            "source_sha256": {
                str(PLAN.relative_to(ROOT)): _sha256(PLAN),
                str(Path(__file__).relative_to(ROOT)): _sha256(Path(__file__)),
                "bayesfilter/highdim/higher_moment_contract_e.py": _sha256(ROOT / "bayesfilter/highdim/higher_moment_contract_e.py"),
                "bayesfilter/highdim/cubature_genut_filter.py": _sha256(ROOT / "bayesfilter/highdim/cubature_genut_filter.py"),
            },
        },
        "nonclaims": [
            "coordinatewise standardized cap is extension_or_invention outside Austria bounded-teacher chart",
            "no exact nonlinear score claim for predator-prey",
            "KSC dense reference is diagnostic only",
            "six seeds give descriptive uncertainty, not statistically supported ranking",
            "no default, HMC, NeuTra, posterior, or broad superiority claim",
        ],
    }
    (output_root / "result.json").write_text(json.dumps(_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_root / "result.md").write_text(_render(payload), encoding="utf-8")
    return payload


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT / "attempt01")
    args = parser.parse_args()
    payload = run(args.output_root.resolve())
    print(json.dumps({"status": payload["status"], "output": str(args.output_root.resolve()), "wall_time_seconds": payload["wall_time_seconds"]}))


if __name__ == "__main__":
    main()
