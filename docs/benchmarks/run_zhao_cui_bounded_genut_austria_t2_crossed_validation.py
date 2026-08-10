#!/usr/bin/env python3
"""Calibrate and validate the bounded Zhao-Cui/GenUT Austria T2 repair."""

from __future__ import annotations

import argparse
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
import traceback
from typing import Any, Callable, Mapping


os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tensorflow as tf  # noqa: E402

from bayesfilter.runtime.gpu_memory_policy import (  # noqa: E402
    configure_tensorflow_gpu_memory_growth,
)


MEMORY_POLICY = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
tf.config.set_soft_device_placement(False)
tf.config.experimental.enable_tensor_float_32_execution(False)

from bayesfilter.highdim.cubature_genut_adapters import (  # noqa: E402
    parameterized_austria_sir_candidate_adapter,
)
from bayesfilter.highdim.cubature_genut_candidate import cubature_design  # noqa: E402
from bayesfilter.highdim.cubature_genut_filter import finite_value_score  # noqa: E402
from bayesfilter.highdim.zhao_cui_austria_sir_bounded_teacher_tf import (  # noqa: E402
    load_austria_t1_t2_bounded_teacher,
)
from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_target_tf import (  # noqa: E402
    SIR_OBSERVATION_SHA256,
    generate_sealed_lane_b_dataset,
)


PLAN = Path(
    "docs/plans/"
    "bayesfilter-zhao-cui-genut-austria-t2-dual-cap-plan-2026-08-06.md"
)
SCHEMA = "bayesfilter.zhao_cui_genut_austria_t2_dual_cap.v1"
N = 1008
CALIBRATION_PARTICLE_SEEDS = (98701, 98702)
VALIDATION_PARTICLE_SEEDS = (98801, 98802, 98803, 98804, 98805, 98806)
FILTER_CONTROLS = {
    "epsilon": 8.0,
    "sinkhorn_steps": 16,
    "balance_steps": 16,
    "ridge": 1.0e-5,
    "higher_moment_correction_steps": 4,
    "higher_moment_floor": 1.0e-5,
    "pairwise_moment_correction_steps": 4,
    "pairwise_moment_floor": 1.0e-5,
}
DUAL_CAP_CANDIDATES = tuple(
    {
        "candidate_id": (
            f"radial_{'off' if radial_cap == 0 else format(radial_cap, 'g')}_"
            f"coord_{'off' if coordinate_cap == 0 else format(coordinate_cap, 'g')}"
        ),
        "diagonal_strength": 0.0,
        "pairwise_strength": 0.02,
        "pairwise_cap": float(radial_cap),
        "coordinatewise_cap": float(coordinate_cap),
        "coordinatewise_power": 8,
    }
    for radial_cap in (0.0, 2.0)
    for coordinate_cap in (0.0, 0.90, 0.95, 0.98)
)
FD_STEP = 1.0e-3
FD_ABSOLUTE_LIMIT = 0.08
FD_NORMALIZED_LIMIT = 0.03
AFFINE_NORMALIZED_LIMIT = 2.0e-4
TEACHER_TO_PARTICLE_SD_LIMIT = 0.5


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _safe(value: Any) -> Any:
    if isinstance(value, tf.Tensor):
        materialized = value.numpy()
        return materialized.item() if materialized.ndim == 0 else materialized.tolist()
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _mean(values: list[float]) -> float:
    return float(statistics.fmean(values))


def _sample_sd(values: list[float]) -> float:
    return float(statistics.stdev(values)) if len(values) > 1 else 0.0


def _noise(seed: int) -> tuple[tf.Tensor, tf.Tensor]:
    return (
        tf.random.stateless_normal([N, 18], [seed, 101], dtype=tf.float32),
        tf.random.stateless_normal([2, N, 18], [seed, 102], dtype=tf.float32),
    )


def _make_evaluator(
    teacher,
    *,
    diagonal_steps: int,
    diagonal_strength: float,
    pairwise_steps: int,
    pairwise_strength: float,
    pairwise_cap: float,
    coordinatewise_cap: float = 0.0,
    coordinatewise_power: int = 8,
) -> Callable[..., tuple[tf.Tensor, tf.Tensor, Mapping[str, tf.Tensor]]]:
    adapter = parameterized_austria_sir_candidate_adapter(latent_preclip=True)

    @tf.function(jit_compile=True, reduce_retracing=True)
    def evaluate(theta, observations, initial_noise, process_noise, design):
        with tf.device("/GPU:0"):
            return finite_value_score(
                adapter,
                tf.ensure_shape(theta, [3]),
                tf.ensure_shape(observations, [2, 9]),
                tf.ensure_shape(initial_noise, [N, 18]),
                tf.ensure_shape(process_noise, [2, N, 18]),
                tf.ensure_shape(design, [N, 18]),
                epsilon=FILTER_CONTROLS["epsilon"],
                sinkhorn_steps=FILTER_CONTROLS["sinkhorn_steps"],
                balance_steps=FILTER_CONTROLS["balance_steps"],
                ridge=FILTER_CONTROLS["ridge"],
                transition_before_first_observation=True,
                higher_moment_correction_steps=diagonal_steps,
                higher_moment_strength=diagonal_strength,
                higher_moment_floor=FILTER_CONTROLS["higher_moment_floor"],
                pairwise_moment_correction_steps=pairwise_steps,
                pairwise_moment_strength=pairwise_strength,
                pairwise_moment_floor=FILTER_CONTROLS["pairwise_moment_floor"],
                pairwise_particle_rms_cap=pairwise_cap,
                coordinatewise_bounded_cap=coordinatewise_cap,
                coordinatewise_bounded_cap_power=coordinatewise_power,
                bounded_feature_teacher=teacher,
            )

    return evaluate


def _row(evaluate, theta, observations, design, seed: int) -> dict[str, Any]:
    initial, process = _noise(seed)
    started = time.perf_counter()
    value, score, diagnostics = evaluate(
        theta, observations, initial, process, design
    )
    program_valid = bool(diagnostics["program_valid"].numpy())
    finite = (
        program_valid
        and bool(tf.math.is_finite(value).numpy())
        and bool(tf.reduce_all(tf.math.is_finite(score)).numpy())
    )
    row = {
        "particle_seed": seed,
        "finite": finite,
        "program_valid": program_valid,
        "device": value.device,
        "value": float(value.numpy()) if finite else None,
        "score": [float(item) for item in score.numpy()] if finite else None,
        "wall_time_seconds": time.perf_counter() - started,
        "maximum_absolute_bounded_coordinate": float(
            diagnostics["maximum_absolute_bounded_coordinate"].numpy()
        ),
        "maximum_physical_affine_mean_residual": float(
            diagnostics["maximum_physical_affine_mean_residual"].numpy()
        ),
        "maximum_physical_affine_covariance_residual": float(
            diagnostics["maximum_physical_affine_covariance_residual"].numpy()
        ),
        "maximum_normalized_physical_affine_mean_residual": float(
            diagnostics[
                "maximum_normalized_physical_affine_mean_residual"
            ].numpy()
        ),
        "maximum_normalized_physical_affine_covariance_residual": float(
            diagnostics[
                "maximum_normalized_physical_affine_covariance_residual"
            ].numpy()
        ),
        "diagonal_shape_objective": float(
            diagnostics["mean_normalized_shape_residual_objective"].numpy()
        ),
        "pairwise_shape_objective": float(
            diagnostics["mean_normalized_pairwise_shape_residual_objective"].numpy()
        ),
        "maximum_pairwise_pre_cap_particle_rms": float(
            diagnostics["maximum_pairwise_pre_cap_particle_rms"].numpy()
        ),
        "maximum_pairwise_post_cap_particle_rms": float(
            diagnostics["maximum_pairwise_post_cap_particle_rms"].numpy()
        ),
        "minimum_pairwise_particle_cap_scale": float(
            diagnostics["minimum_pairwise_particle_cap_scale"].numpy()
        ),
        "maximum_coordinatewise_pre_cap_absolute": float(
            diagnostics["maximum_coordinatewise_pre_cap_absolute"].numpy()
        ),
        "maximum_coordinatewise_post_cap_absolute": float(
            diagnostics["maximum_coordinatewise_post_cap_absolute"].numpy()
        ),
        "mean_coordinatewise_cap_displacement": float(
            diagnostics["mean_coordinatewise_cap_displacement"].numpy()
        ),
        "fraction_coordinatewise_cap_active": float(
            diagnostics["fraction_coordinatewise_cap_active"].numpy()
        ),
        "minimum_coordinatewise_cap_derivative": float(
            diagnostics["minimum_coordinatewise_cap_derivative"].numpy()
        ),
        "maximum_coordinatewise_inverse_derivative": float(
            diagnostics["maximum_coordinatewise_inverse_derivative"].numpy()
        ),
        "maximum_normalized_shape_displacement": float(
            diagnostics["maximum_normalized_shape_displacement"].numpy()
        ),
        "value_increments": _safe(diagnostics["value_increments"]),
        "score_increments": _safe(diagnostics["score_increments"]),
    }
    row["numerical_gate_pass"] = bool(
        finite
        and "GPU" in row["device"].upper()
        and row["maximum_absolute_bounded_coordinate"] < 1.0
        and row["maximum_normalized_physical_affine_mean_residual"]
        <= AFFINE_NORMALIZED_LIMIT
        and row["maximum_normalized_physical_affine_covariance_residual"]
        <= AFFINE_NORMALIZED_LIMIT
    )
    return row


def _finite_difference(
    evaluate, theta, observations, design, seed: int
) -> list[dict[str, Any]]:
    initial, process = _noise(seed)
    step = tf.constant(FD_STEP, tf.float32)
    origin = evaluate(theta, observations, initial, process, design)
    rows = []
    for parameter in range(3):
        direction = tf.one_hot(parameter, 3, dtype=tf.float32)
        plus = evaluate(
            theta + step * direction, observations, initial, process, design
        )[0]
        minus = evaluate(
            theta - step * direction, observations, initial, process, design
        )[0]
        observed = (plus - minus) / (2.0 * step)
        expected = origin[1][parameter]
        absolute = tf.abs(observed - expected)
        normalized = absolute / tf.maximum(tf.abs(expected), 1.0)
        finite = bool(
            tf.reduce_all(
                tf.math.is_finite(tf.stack([observed, expected, absolute, normalized]))
            ).numpy()
        )
        row = {
            "parameter": parameter,
            "step": FD_STEP,
            "manual_score": float(expected.numpy()) if finite else None,
            "finite_difference": float(observed.numpy()) if finite else None,
            "absolute_residual": float(absolute.numpy()) if finite else None,
            "normalized_residual": float(normalized.numpy()) if finite else None,
        }
        row["gate_pass"] = bool(
            finite
            and row["absolute_residual"] <= FD_ABSOLUTE_LIMIT
            and row["normalized_residual"] <= FD_NORMALIZED_LIMIT
        )
        rows.append(row)
    return rows


def _metric(row: Mapping[str, Any], index: int) -> float:
    if index == -1:
        return float(row["value"])
    return float(row["score"][index])


def _teacher_sensitivity(
    validation: list[dict[str, Any]], metric_index: int, metric_id: str
) -> dict[str, Any]:
    if not all(
        row["finite"] for teacher in validation for row in teacher["rows"]
    ):
        return {
            "metric_id": metric_id,
            "teacher_marginal_means": None,
            "between_teacher_sd": None,
            "pooled_within_teacher_particle_sd": None,
            "teacher_to_particle_sd_ratio": None,
            "limit": TEACHER_TO_PARTICLE_SD_LIMIT,
            "gate_pass": False,
            "not_computable_reason": "at least one validation row is non-finite",
        }
    teacher_means = [
        _mean([_metric(row, metric_index) for row in teacher["rows"]])
        for teacher in validation
    ]
    within_variances = [
        _sample_sd([_metric(row, metric_index) for row in teacher["rows"]]) ** 2
        for teacher in validation
    ]
    teacher_sd = _sample_sd(teacher_means)
    pooled_particle_sd = math.sqrt(_mean(within_variances))
    ratio = teacher_sd / pooled_particle_sd if pooled_particle_sd > 0.0 else math.inf
    return {
        "metric_id": metric_id,
        "teacher_marginal_means": teacher_means,
        "between_teacher_sd": teacher_sd,
        "pooled_within_teacher_particle_sd": pooled_particle_sd,
        "teacher_to_particle_sd_ratio": ratio,
        "limit": TEACHER_TO_PARTICLE_SD_LIMIT,
        "gate_pass": bool(math.isfinite(ratio) and ratio <= TEACHER_TO_PARTICLE_SD_LIMIT),
    }


def _paired_differences(
    baseline_rows: list[dict[str, Any]],
    validation: list[dict[str, Any]],
    metric_index: int,
    metric_id: str,
) -> dict[str, Any]:
    if not all(row["finite"] for row in baseline_rows) or not all(
        row["finite"] for teacher in validation for row in teacher["rows"]
    ):
        return {
            "metric_id": metric_id,
            "mean_candidate_minus_baseline": None,
            "sd_candidate_minus_baseline": None,
            "negative_count": None,
            "zero_count": None,
            "positive_count": None,
            "role": "explanatory_only_no_accuracy_authority",
            "not_computable_reason": "at least one baseline or validation row is non-finite",
        }
    baseline_by_seed = {row["particle_seed"]: row for row in baseline_rows}
    differences = []
    for teacher in validation:
        for row in teacher["rows"]:
            differences.append(
                _metric(row, metric_index)
                - _metric(baseline_by_seed[row["particle_seed"]], metric_index)
            )
    return {
        "metric_id": metric_id,
        "mean_candidate_minus_baseline": _mean(differences),
        "sd_candidate_minus_baseline": _sample_sd(differences),
        "negative_count": sum(value < 0.0 for value in differences),
        "zero_count": sum(value == 0.0 for value in differences),
        "positive_count": sum(value > 0.0 for value in differences),
        "role": "explanatory_only_no_accuracy_authority",
    }


def run(
    calibration_teacher_dir: Path,
    validation_teacher_dirs: list[Path],
    output_root: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    output_root.mkdir(parents=True, exist_ok=False)
    if len(validation_teacher_dirs) != 3:
        raise ValueError("exactly three validation teacher directories are required")
    logical = tf.config.list_logical_devices("GPU")
    if not logical:
        raise RuntimeError("crossed validation requires a logical GPU")
    calibration_teacher, calibration_manifest = (
        load_austria_t1_t2_bounded_teacher(calibration_teacher_dir)
    )
    _states, observations64, _all = generate_sealed_lane_b_dataset()
    observations = tf.cast(observations64[:2], tf.float32)
    theta = tf.zeros([3], tf.float32)
    design = cubature_design(dim=18, num_particles=N)

    calibration = []
    for candidate in DUAL_CAP_CANDIDATES:
        evaluate = _make_evaluator(
            calibration_teacher,
            diagonal_steps=FILTER_CONTROLS["higher_moment_correction_steps"],
            diagonal_strength=0.0,
            pairwise_steps=FILTER_CONTROLS["pairwise_moment_correction_steps"],
            pairwise_strength=candidate["pairwise_strength"],
            pairwise_cap=candidate["pairwise_cap"],
            coordinatewise_cap=candidate["coordinatewise_cap"],
            coordinatewise_power=candidate["coordinatewise_power"],
        )
        rows = [
            _row(evaluate, theta, observations, design, seed)
            for seed in CALIBRATION_PARTICLE_SEEDS
        ]
        fd = _finite_difference(
            evaluate, theta, observations, design, CALIBRATION_PARTICLE_SEEDS[0]
        )
        candidate_pass = all(row["numerical_gate_pass"] for row in rows) and all(
            row["gate_pass"] for row in fd
        )
        calibration.append(
            {
                **candidate,
                "rows": rows,
                "finite_difference": fd,
                "mean_coordinatewise_cap_displacement": (
                    _mean(
                        [row["mean_coordinatewise_cap_displacement"] for row in rows]
                    )
                    if candidate_pass
                    else None
                ),
                "mean_inverse_derivative": (
                    _mean(
                        [
                            row["maximum_coordinatewise_inverse_derivative"]
                            for row in rows
                        ]
                    )
                    if candidate_pass
                    else None
                ),
                "calibration_gate_pass": candidate_pass,
            }
        )
    calibration_ledger = {
        "schema_version": SCHEMA,
        "phase": "calibration",
        "teacher_directory": calibration_teacher_dir.as_posix(),
        "particle_seeds": list(CALIBRATION_PARTICLE_SEEDS),
        "thresholds": {
            "finite_difference_step": FD_STEP,
            "finite_difference_absolute_limit": FD_ABSOLUTE_LIMIT,
            "finite_difference_normalized_limit": FD_NORMALIZED_LIMIT,
            "normalized_physical_affine_residual_limit": AFFINE_NORMALIZED_LIMIT,
        },
        "candidates": calibration,
    }
    (output_root / "calibration.json").write_text(
        json.dumps(_safe(calibration_ledger), indent=2, sort_keys=True) + "\n"
    )
    eligible = [row for row in calibration if row["calibration_gate_pass"]]
    if not eligible:
        raise RuntimeError("no calibration candidate passed the numerical gates")
    selected = min(
        eligible,
        key=lambda row: (
            row["mean_coordinatewise_cap_displacement"],
            row["mean_inverse_derivative"],
            row["pairwise_cap"],
            row["coordinatewise_cap"],
        ),
    )
    selection = {
        "candidate_id": selected["candidate_id"],
        "diagonal_strength": selected["diagonal_strength"],
        "pairwise_strength": selected["pairwise_strength"],
        "pairwise_cap": selected["pairwise_cap"],
        "coordinatewise_cap": selected["coordinatewise_cap"],
        "coordinatewise_power": selected["coordinatewise_power"],
        "mean_coordinatewise_cap_displacement": selected[
            "mean_coordinatewise_cap_displacement"
        ],
        "mean_inverse_derivative": selected["mean_inverse_derivative"],
        "selection_rule": (
            "lowest calibration mean coordinatewise cap displacement among "
            "numerical-gate passes; ties by lower mean inverse derivative, "
            "lower radial cap, then lower coordinate cap"
        ),
    }
    (output_root / "selection.json").write_text(
        json.dumps(_safe(selection), indent=2, sort_keys=True) + "\n"
    )

    baseline_evaluate = _make_evaluator(
        None,
        diagonal_steps=0,
        diagonal_strength=0.0,
        pairwise_steps=0,
        pairwise_strength=0.0,
        pairwise_cap=0.0,
        coordinatewise_cap=0.0,
        coordinatewise_power=8,
    )
    baseline_rows = [
        _row(baseline_evaluate, theta, observations, design, seed)
        for seed in VALIDATION_PARTICLE_SEEDS
    ]
    validation = []
    validation_manifests = []
    for teacher_index, teacher_dir in enumerate(validation_teacher_dirs):
        teacher, teacher_manifest = load_austria_t1_t2_bounded_teacher(teacher_dir)
        validation_manifests.append(teacher_manifest)
        evaluate = _make_evaluator(
            teacher,
            diagonal_steps=FILTER_CONTROLS["higher_moment_correction_steps"],
            diagonal_strength=selection["diagonal_strength"],
            pairwise_steps=FILTER_CONTROLS["pairwise_moment_correction_steps"],
            pairwise_strength=selection["pairwise_strength"],
            pairwise_cap=selection["pairwise_cap"],
            coordinatewise_cap=selection["coordinatewise_cap"],
            coordinatewise_power=selection["coordinatewise_power"],
        )
        rows = [
            _row(evaluate, theta, observations, design, seed)
            for seed in VALIDATION_PARTICLE_SEEDS
        ]
        fd = _finite_difference(
            evaluate, theta, observations, design, VALIDATION_PARTICLE_SEEDS[0]
        )
        validation.append(
            {
                "teacher_index": teacher_index,
                "teacher_directory": teacher_dir.as_posix(),
                "rows": rows,
                "finite_difference": fd,
                "numerical_gate_pass": all(
                    row["numerical_gate_pass"] for row in rows
                )
                and all(row["gate_pass"] for row in fd),
            }
        )

    metric_specs = [(-1, "value"), (0, "score_0"), (1, "score_1"), (2, "score_2")]
    teacher_sensitivity = [
        _teacher_sensitivity(validation, metric_index, metric_id)
        for metric_index, metric_id in metric_specs
    ]
    paired_differences = [
        _paired_differences(
            baseline_rows, validation, metric_index, metric_id
        )
        for metric_index, metric_id in metric_specs
    ]
    numerical_pass = all(row["numerical_gate_pass"] for row in baseline_rows) and all(
        teacher["numerical_gate_pass"] for teacher in validation
    )
    sensitivity_pass = all(row["gate_pass"] for row in teacher_sensitivity)
    campaign_pass = numerical_pass and sensitivity_pass
    result = {
        "schema_version": SCHEMA,
        "status": (
            "PASS_T2_CANDIDATE_FOR_T3_PLUS_TEACHER_EXTENSION"
            if campaign_pass
            else "FAIL_T2_CANDIDATE_NOT_ELIGIBLE_FOR_EXTENSION"
        ),
        "campaign_pass": campaign_pass,
        "numerical_gate_pass": numerical_pass,
        "teacher_sensitivity_gate_pass": sensitivity_pass,
        "plan": PLAN.as_posix(),
        "research_question": (
            "is the calibrated T2 bounded-teacher candidate numerically valid and "
            "teacher-seed robust enough to justify T3+ teacher construction?"
        ),
        "target": {
            "model_id": "austria_sir_lane_b_latent_preclip_T2",
            "horizon": 2,
            "particle_count": N,
            "source_observation_sha256": SIR_OBSERVATION_SHA256,
            "teacher_owner": "independent_sampled_zhao_cui_tt_marginal",
            "teacher_route_classification": "extension_or_invention",
            "empirical_genut_particle_target_used": False,
        },
        "thresholds": {
            "finite_difference_step": FD_STEP,
            "finite_difference_absolute_limit": FD_ABSOLUTE_LIMIT,
            "finite_difference_normalized_limit": FD_NORMALIZED_LIMIT,
            "normalized_physical_affine_residual_limit": AFFINE_NORMALIZED_LIMIT,
            "teacher_to_particle_sd_ratio_limit": TEACHER_TO_PARTICLE_SD_LIMIT,
        },
        "filter_controls": FILTER_CONTROLS,
        "calibration_particle_seeds": list(CALIBRATION_PARTICLE_SEEDS),
        "validation_particle_seeds": list(VALIDATION_PARTICLE_SEEDS),
        "calibration_teacher_directory": calibration_teacher_dir.as_posix(),
        "validation_teacher_directories": [
            path.as_posix() for path in validation_teacher_dirs
        ],
        "calibration_teacher_manifest": calibration_manifest,
        "validation_teacher_manifests": validation_manifests,
        "calibration": calibration,
        "selection": selection,
        "baseline_rows": baseline_rows,
        "validation": validation,
        "teacher_sensitivity": teacher_sensitivity,
        "paired_candidate_minus_baseline": paired_differences,
        "configuration": {
            "dtype": "float32",
            "tf32": False,
            "jit_compile": True,
            "deterministic_ops": True,
            "gpu_memory_growth": True,
        },
        "device": {
            "logical_devices": [device.name for device in logical],
            "trust_basis": "owner_designated_managed_session_visible_gpu_trusted",
        },
        "memory_policy": dict(MEMORY_POLICY),
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
            "teacher_manifest_sha256": {
                calibration_teacher_dir.as_posix(): _sha256(
                    calibration_teacher_dir / "manifest.json"
                ),
                **{
                    path.as_posix(): _sha256(path / "manifest.json")
                    for path in validation_teacher_dirs
                },
            },
            "source_sha256": {
                path.as_posix(): _sha256(ROOT / path)
                for path in (
                    PLAN,
                    Path("bayesfilter/highdim/cubature_genut_filter.py"),
                    Path("bayesfilter/highdim/higher_moment_contract_e.py"),
                    Path(
                        "bayesfilter/highdim/"
                        "zhao_cui_austria_sir_bounded_teacher_tf.py"
                    ),
                    Path(__file__).relative_to(ROOT),
                )
            },
        },
        "inference_status": {
            "hard_veto_screen": "pass" if numerical_pass else "fail",
            "statistically_supported_ranking": False,
            "descriptive_only_differences": True,
            "default_readiness": False,
            "next_evidence_needed": (
                "construct and validate T3 through T20 teachers"
                if campaign_pass
                else "repair teacher sampling or bounded correction at T2"
            ),
        },
        "nonclaims": [
            "same-program finite differences do not establish exact score accuracy",
            "bounded-feature targets are not physical third/fourth moments",
            "paired candidate-baseline differences are explanatory only",
            "T2 cannot establish T20, HMC, NeuTra, posterior, or default readiness",
            "three teacher seeds and six particle seeds do not support arm ranking",
        ],
    }
    (output_root / "result.json").write_text(
        json.dumps(_safe(result), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-teacher-dir", type=Path, required=True)
    parser.add_argument(
        "--validation-teacher-dir", type=Path, action="append", required=True
    )
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    try:
        result = run(
            args.calibration_teacher_dir.resolve(),
            [path.resolve() for path in args.validation_teacher_dir],
            output,
        )
    except Exception as exc:
        output.mkdir(parents=True, exist_ok=True)
        (output / "failure.json").write_text(
            json.dumps(
                {
                    "status": "FAILED_CAMPAIGN_ATTEMPT",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "command": [sys.executable, *sys.argv],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        raise
    print(
        json.dumps(
            {
                "status": result["status"],
                "campaign_pass": result["campaign_pass"],
                "selected_candidate": result["selection"]["candidate_id"],
                "output": output.as_posix(),
                "wall_time_seconds": result["wall_time_seconds"],
            }
        )
    )


if __name__ == "__main__":
    main()
