#!/usr/bin/env python3
"""Tune and test pairwise-moment GenUT on the active Austria SIR target."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tensorflow as tf

from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
from docs.benchmarks import run_moment_retuned_genut_whole_leaderboard as base


PLAN = Path(
    "docs/plans/bayesfilter-austria-sir-pairwise-moment-genut-score-trial-plan-2026-07-30.md"
)
SCHEMA = "bayesfilter.austria_sir_pairwise_moment_genut_score_trial.v1"
N = 1008
CLAIM_SEEDS = tuple(range(98201, 98217))
TUNING_SEEDS = (98301, 98302)
BASE_CONTROLS = {
    "epsilon": 8.0,
    "sinkhorn_steps": 16,
    "balance_steps": 16,
    "ridge": 1.0e-5,
    "higher_moment_correction_steps": 4,
    "higher_moment_strength": 0.2,
    "higher_moment_floor": 1.0e-5,
    "pairwise_moment_correction_steps": 0,
    "pairwise_moment_strength": 0.0,
    "pairwise_moment_floor": 1.0e-5,
}
PAIRWISE_GRID = (BASE_CONTROLS,) + tuple(
    {
        **BASE_CONTROLS,
        "pairwise_moment_correction_steps": steps,
        "pairwise_moment_strength": strength,
    }
    for steps in (1, 2, 4)
    for strength in (0.005, 0.01, 0.02, 0.05)
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _safe(value: Any) -> Any:
    if hasattr(value, "numpy"):
        value = value.numpy()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _target() -> dict[str, Any]:
    return base._build_targets()["austria_sir_T20"]


def _evaluator(target: dict[str, Any], controls: dict[str, Any]):
    return base._make_evaluator(
        adapter=target["adapter"],
        horizon=20,
        observation_dim=9,
        state_dim=18,
        parameter_dim=3,
        transition_before_first_observation=True,
        controls=controls,
    )


def _row(
    evaluator: Any,
    target: dict[str, Any],
    observations: tf.Tensor,
    seed: int,
) -> dict[str, Any]:
    return base._evaluate(
        evaluator,
        target["theta"],
        tf.cast(observations, tf.float32),
        seed,
        target["design"],
    )


def _coordinate_variances(rows: list[dict[str, Any]]) -> list[float]:
    vectors = [[row["value"], *row["score"]] for row in rows]
    return [
        statistics.variance(vector[index] for vector in vectors)
        for index in range(len(vectors[0]))
    ]


def _partition(
    evaluator: Any,
    target: dict[str, Any],
    datasets: list[tf.Tensor],
) -> dict[str, Any]:
    all_rows: list[dict[str, Any]] = []
    per_dataset_variances: list[list[float]] = []
    for dataset in datasets:
        rows = [
            _row(evaluator, target, dataset, seed) for seed in TUNING_SEEDS
        ]
        all_rows.extend(rows)
        if all(base._valid(row) for row in rows):
            per_dataset_variances.append(_coordinate_variances(rows))
    valid = bool(all_rows) and all(base._valid(row) for row in all_rows)
    coordinate_variances = None
    if valid and len(per_dataset_variances) == len(datasets):
        coordinate_variances = [
            statistics.mean(row[index] for row in per_dataset_variances)
            for index in range(4)
        ]
    return {
        "valid": valid,
        "rows": all_rows,
        "mean_pairwise_objective": (
            statistics.mean(
                row["mean_normalized_pairwise_shape_residual_objective"]
                for row in all_rows
            )
            if valid
            else None
        ),
        "mean_diagonal_objective": (
            statistics.mean(
                row["mean_normalized_shape_residual_objective"] for row in all_rows
            )
            if valid
            else None
        ),
        "coordinate_variances": coordinate_variances,
        "maximum_displacement": (
            max(row["maximum_normalized_shape_displacement"] for row in all_rows)
            if valid
            else None
        ),
    }


def _tune(target: dict[str, Any], *, smoke_only: bool) -> dict[str, Any]:
    grid = (
        (BASE_CONTROLS, PAIRWISE_GRID[1]) if smoke_only else PAIRWISE_GRID
    )
    candidates: list[dict[str, Any]] = []
    for controls in grid:
        started = time.perf_counter()
        evaluator = _evaluator(target, controls)
        calibration = _partition(evaluator, target, target["calibration"])
        validation = _partition(evaluator, target, target["validation"])
        candidates.append(
            {
                "controls": controls,
                "calibration": calibration,
                "validation": validation,
                "wall_time_seconds": time.perf_counter() - started,
            }
        )

    baseline = candidates[0]
    baseline_validation = baseline["validation"]
    if not baseline_validation["valid"]:
        raise RuntimeError("diagonal-only validation baseline is invalid")
    baseline_variances = baseline_validation["coordinate_variances"]
    assert baseline_variances is not None
    baseline_value_sd = math.sqrt(baseline_variances[0])
    baseline_pairwise = baseline_validation["mean_pairwise_objective"]
    assert baseline_pairwise is not None

    for candidate in candidates:
        validation = candidate["validation"]
        variances = validation["coordinate_variances"]
        residual_improved = bool(
            validation["valid"]
            and validation["mean_pairwise_objective"] is not None
            and validation["mean_pairwise_objective"] < baseline_pairwise
        )
        score_variance_veto = bool(
            variances is None
            or any(
                variances[index] > baseline_variances[index]
                for index in range(1, 4)
            )
        )
        value_variance_veto = bool(
            variances is None
            or math.sqrt(variances[0]) > 1.25 * baseline_value_sd
        )
        candidate["selection_diagnostics"] = {
            "pairwise_residual_improved": residual_improved,
            "score_variance_veto": score_variance_veto,
            "value_variance_veto": value_variance_veto,
            "eligible": bool(
                validation["valid"]
                and residual_improved
                and not score_variance_veto
                and not value_variance_veto
            ),
        }

    eligible = [
        candidate
        for candidate in candidates[1:]
        if candidate["selection_diagnostics"]["eligible"]
    ]
    if smoke_only:
        selected = candidates[1]
        selection_status = "SMOKE_FORCED_NONZERO_PAIRWISE_ARM"
    elif eligible:
        selected = min(
            eligible,
            key=lambda candidate: (
                max(candidate["validation"]["coordinate_variances"][1:]),
                sum(candidate["validation"]["coordinate_variances"][1:]),
                candidate["validation"]["mean_pairwise_objective"],
                candidate["validation"]["maximum_displacement"],
                candidate["controls"]["pairwise_moment_strength"],
                candidate["controls"]["pairwise_moment_correction_steps"],
            ),
        )
        selection_status = "PAIRWISE_CANDIDATE_SELECTED"
    else:
        selected = baseline
        selection_status = "NO_PAIRWISE_CANDIDATE_PASSED_TUNING_VETOES"
    return {
        "selection_status": selection_status,
        "selected_controls": selected["controls"],
        "baseline_controls": BASE_CONTROLS,
        "candidates": candidates,
        "claim_data_read_during_selection": False,
    }


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = ("value", "log_kappa_scale", "log_nu_scale", "log_observation_noise_scale")
    vectors = [[row["value"], *row["score"]] for row in rows]
    result: dict[str, Any] = {"count": len(rows), "all_valid": all(base._valid(row) for row in rows)}
    for index, label in enumerate(labels):
        sample = [vector[index] for vector in vectors]
        mean = statistics.mean(sample)
        sd = statistics.stdev(sample)
        half = 2.131449545559323 * sd / math.sqrt(len(sample))
        result[label] = {
            "mean": mean,
            "sample_sd": sd,
            "ci95_lower": mean - half,
            "ci95_upper": mean + half,
        }
    result["mean_pairwise_objective"] = statistics.mean(
        row["mean_normalized_pairwise_shape_residual_objective"] for row in rows
    )
    result["mean_diagonal_objective"] = statistics.mean(
        row["mean_normalized_shape_residual_objective"] for row in rows
    )
    return result


def _bootstrap_variance_ratio(
    baseline: list[dict[str, Any]], candidate: list[dict[str, Any]], *, draws: int = 10000
) -> dict[str, Any]:
    rng = random.Random(20260730)
    baseline_vectors = [[*row["score"]] for row in baseline]
    candidate_vectors = [[*row["score"]] for row in candidate]

    def aggregate_ratio(indices: list[int]) -> float:
        baseline_sds = [
            statistics.stdev(baseline_vectors[index][coordinate] for index in indices)
            for coordinate in range(3)
        ]
        candidate_sds = [
            statistics.stdev(candidate_vectors[index][coordinate] for index in indices)
            for coordinate in range(3)
        ]
        return statistics.geometric_mean(
            (candidate_sds[index] / baseline_sds[index]) ** 2
            for index in range(3)
        )

    ratios = []
    while len(ratios) < draws:
        indices = [rng.randrange(len(baseline)) for _ in baseline]
        try:
            ratio = aggregate_ratio(indices)
        except statistics.StatisticsError:
            continue
        if math.isfinite(ratio):
            ratios.append(ratio)
    ratios.sort()
    point = aggregate_ratio(list(range(len(baseline))))
    return {
        "aggregate_geometric_variance_ratio": point,
        "bootstrap_ci95_lower": ratios[int(0.025 * draws)],
        "bootstrap_ci95_upper": ratios[int(0.975 * draws) - 1],
        "bootstrap_draws": draws,
        "role": "limited paired-seed statistical diagnostic",
    }


def _render(payload: dict[str, Any]) -> str:
    lines = [
        "# Austria SIR Pairwise-Moment GenUT Score Trial",
        "",
        f"Status: `{payload['status']}`",
        "",
        "| Arm | Value mean (SD) | log kappa score mean (SD) | log nu score mean (SD) | log obs-noise score mean (SD) |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for arm in ("baseline", "candidate"):
        summary = payload["claim"][arm]["summary"]
        cells = []
        for label in ("value", "log_kappa_scale", "log_nu_scale", "log_observation_noise_scale"):
            cells.append(f"{summary[label]['mean']:.6g} ({summary[label]['sample_sd']:.6g})")
        lines.append(f"| {arm} | " + " | ".join(cells) + " |")
    lines += [
        "",
        f"Aggregate score variance ratio: `{payload['claim']['variance_ratio']['aggregate_geometric_variance_ratio']:.6g}`",
        "",
        "Cross-method and variance differences remain limited to this fixed target and seed set.",
    ]
    return "\n".join(lines) + "\n"


def run(output_root: Path, *, smoke_only: bool) -> dict[str, Any]:
    started = time.perf_counter()
    output_root.mkdir(parents=True, exist_ok=False)
    memory_policy = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    tf.config.set_soft_device_placement(False)
    tf.config.experimental.enable_tensor_float_32_execution(True)
    logical = tf.config.list_logical_devices("GPU")
    if not logical:
        raise RuntimeError("pairwise trial requires a logical GPU")
    target = _target()
    tuning = _tune(target, smoke_only=smoke_only)
    baseline_evaluator = _evaluator(target, BASE_CONTROLS)
    candidate_evaluator = _evaluator(target, tuning["selected_controls"])
    seeds = CLAIM_SEEDS[:1] if smoke_only else CLAIM_SEEDS
    baseline_rows = [
        _row(baseline_evaluator, target, target["observations"], seed) for seed in seeds
    ]
    candidate_rows = [
        _row(candidate_evaluator, target, target["observations"], seed) for seed in seeds
    ]
    if not all(base._valid(row) for row in baseline_rows):
        raise RuntimeError("claim baseline is invalid")
    if not all(base._valid(row) for row in candidate_rows):
        raise RuntimeError("selected candidate claim is invalid")
    if smoke_only:
        variance_ratio = {
            "aggregate_geometric_variance_ratio": None,
            "bootstrap_ci95_lower": None,
            "bootstrap_ci95_upper": None,
            "bootstrap_draws": 0,
            "role": "not computed for one-seed smoke",
        }
        status = "PASS_GPU_XLA_SMOKE"
        claim = {
            "baseline": {"rows": baseline_rows},
            "candidate": {"rows": candidate_rows},
            "variance_ratio": variance_ratio,
        }
    else:
        baseline_summary = _summary(baseline_rows)
        candidate_summary = _summary(candidate_rows)
        variance_ratio = _bootstrap_variance_ratio(baseline_rows, candidate_rows)
        all_score_sd_lower = all(
            candidate_summary[label]["sample_sd"] < baseline_summary[label]["sample_sd"]
            for label in ("log_kappa_scale", "log_nu_scale", "log_observation_noise_scale")
        )
        value_sd_ok = candidate_summary["value"]["sample_sd"] <= 1.25 * baseline_summary["value"]["sample_sd"]
        value_shift_ok = abs(candidate_summary["value"]["mean"] - baseline_summary["value"]["mean"]) <= baseline_summary["value"]["sample_sd"] / math.sqrt(len(seeds))
        supported_ratio = variance_ratio["bootstrap_ci95_upper"] < 1.0
        promoted = bool(
            tuning["selection_status"] == "PAIRWISE_CANDIDATE_SELECTED"
            and all_score_sd_lower
            and value_sd_ok
            and value_shift_ok
            and supported_ratio
        )
        status = "PAIRWISE_SCORE_VARIANCE_PROMOTION_PASS" if promoted else "PAIRWISE_SCORE_VARIANCE_PROMOTION_FAIL"
        claim = {
            "baseline": {"controls": BASE_CONTROLS, "rows": baseline_rows, "summary": baseline_summary},
            "candidate": {"controls": tuning["selected_controls"], "rows": candidate_rows, "summary": candidate_summary},
            "variance_ratio": variance_ratio,
            "promotion_gates": {
                "all_score_coordinate_sds_lower": all_score_sd_lower,
                "value_sd_within_25_percent": value_sd_ok,
                "value_mean_shift_within_baseline_se": value_shift_ok,
                "aggregate_variance_ratio_ci_below_one": supported_ratio,
                "passed": promoted,
            },
        }
    payload = {
        "schema_version": SCHEMA,
        "status": status,
        "hard_valid": True,
        "smoke_only": smoke_only,
        "plan": PLAN.as_posix(),
        "target": {
            "row_id": "austria_sir_T20",
            "source_observation_sha256": target["source_observation_sha256"],
            "runtime_fp32_observation_sha256": base._tensor_hash(target["observations"], tf.float32),
            "event_order": target["event_order"],
            "theta": [float(item) for item in target["theta"].numpy()],
            "state_dimension": 18,
            "observation_dimension": 9,
            "parameter_dimension": 3,
            "horizon": 20,
            "particle_count": N,
        },
        "tuning": tuning,
        "claim": claim,
        "configuration": {
            "claim_seeds": seeds,
            "tuning_seeds": TUNING_SEEDS,
            "dtype": "float32",
            "tf32": True,
            "jit_compile": True,
            "score_policy": "manual recursive forward sensitivity",
        },
        "device": {
            "logical_devices": [device.name for device in logical],
            "trust_basis": "owner_designated_managed_session_visible_gpu_trusted",
        },
        "memory_policy": dict(memory_policy),
        "gpu_allocator": {
            key: int(value)
            for key, value in tf.config.experimental.get_memory_info("GPU:0").items()
        },
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "wall_time_seconds": time.perf_counter() - started,
        "run_manifest": {
            "command": [sys.executable, *sys.argv],
            "environment": sys.prefix,
            "host": platform.node(),
            "python": platform.python_version(),
            "tensorflow": tf.__version__,
            "source_sha256": {
                PLAN.as_posix(): _sha256(ROOT / PLAN),
                Path(__file__).relative_to(ROOT).as_posix(): _sha256(Path(__file__)),
                "bayesfilter/highdim/higher_moment_contract_e.py": _sha256(ROOT / "bayesfilter/highdim/higher_moment_contract_e.py"),
                "bayesfilter/highdim/cubature_genut_filter.py": _sha256(ROOT / "bayesfilter/highdim/cubature_genut_filter.py"),
            },
        },
        "nonclaims": [
            "lower score variance does not prove lower score bias",
            "SGQF is a comparator, not an exact Austria SIR oracle",
            "no HMC, default, superiority, Zhao-Cui, or NAWM claim",
        ],
    }
    (output_root / "result.json").write_text(
        json.dumps(_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if not smoke_only:
        (output_root / "result.md").write_text(_render(payload), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    payload = run(args.output_root.resolve(), smoke_only=args.smoke_only)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "output": str(args.output_root.resolve()),
                "wall_time_seconds": payload["wall_time_seconds"],
            }
        )
    )


if __name__ == "__main__":
    main()
