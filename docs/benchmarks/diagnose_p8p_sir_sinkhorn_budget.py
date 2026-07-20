"""Diagnose whether P8p SIR gradient mismatch tracks Sinkhorn budget.

This is a focused diagnostic.  It reuses the P8p parameterized SIR d18 target,
varies only the finite Sinkhorn budget, records the streaming row residual, and
compares the manual reverse score with 13-point trimmed regression finite
differences on the raw theta coordinates.
"""

from __future__ import annotations

import argparse
import copy
import datetime as _dt
import json
import math
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any


_PRE_PARSER = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
_PRE_PARSER.add_argument("--device-scope", choices=("cpu", "visible"), default="visible")
_PRE_PARSER.add_argument("--cuda-visible-devices", default=None)
_PRE_ARGS, _UNKNOWN = _PRE_PARSER.parse_known_args()
if _PRE_ARGS.device_scope == "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
elif _PRE_ARGS.cuda_visible_devices is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = _PRE_ARGS.cuda_visible_devices
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tensorflow as tf

from docs.benchmarks import benchmark_p8p_parameterized_sir_gradient as p8p
from docs.benchmarks import benchmark_p8p_regression_fd_reparameterization as p8p_reg


def _parse_int_csv(value: str) -> list[int]:
    parsed = [int(item.strip()) for item in str(value).split(",") if item.strip()]
    if not parsed:
        raise ValueError("expected at least one integer")
    return parsed


def _parse_float_csv(value: str, *, expected: int | None = None) -> list[float]:
    parsed = [float(item.strip()) for item in str(value).split(",") if item.strip()]
    if expected is not None and len(parsed) != expected:
        raise ValueError(f"expected {expected} comma-separated floats")
    if not parsed or not all(math.isfinite(item) for item in parsed):
        raise ValueError("expected finite floats")
    return parsed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--batch-seeds", default="81120,81121,81122,81123,81124")
    parser.add_argument("--time-steps", type=int, default=3)
    parser.add_argument("--num-particles", type=int, default=64)
    parser.add_argument("--theta", default="0.02,-0.01,0.01")
    parser.add_argument("--candidate-steps", default="10,100,200,400")
    parser.add_argument("--base-step", type=float, default=0.001)
    parser.add_argument("--regression-offsets", default="-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6")
    parser.add_argument("--trim-extreme-values", type=int, default=1)
    parser.add_argument("--row-residual-threshold", type=float, default=1.0e-3)
    parser.add_argument("--sinkhorn-epsilon", type=float, default=1.0)
    parser.add_argument("--annealed-scaling", type=float, default=0.9)
    parser.add_argument("--annealed-convergence-threshold", type=float, default=1.0e-3)
    parser.add_argument(
        "--transport-policy",
        choices=("active-all", "active-odd", "no-resampling"),
        default="active-all",
    )
    parser.add_argument(
        "--transport-gradient-mode",
        choices=(
            p8p.core_tf.MANUAL_STREAMING_FINITE_TRANSPORT_GRADIENT_MODE,
            p8p.core_tf.MANUAL_STREAMING_BLOCKWISE_VJP_FINITE_TRANSPORT_GRADIENT_MODE,
        ),
        default=p8p.core_tf.MANUAL_STREAMING_FINITE_TRANSPORT_GRADIENT_MODE,
    )
    parser.add_argument("--row-chunk-size", type=int, default=64)
    parser.add_argument("--col-chunk-size", type=int, default=64)
    parser.add_argument("--particle-chunk-size", type=int, default=64)
    parser.add_argument("--seed-microbatch-size", type=int, default=0)
    parser.add_argument(
        "--theta-offset-batch-size",
        type=int,
        default=13,
        help="maximum FD theta rows per batched value call; 0 evaluates all offsets at once",
    )
    parser.add_argument("--dtype", choices=("float64", "float32"), default="float32")
    parser.add_argument("--tf32-mode", choices=("default", "enabled", "disabled"), default="enabled")
    parser.add_argument("--device", default="/GPU:0")
    parser.add_argument("--device-scope", choices=("cpu", "visible"), default=_PRE_ARGS.device_scope)
    parser.add_argument("--cuda-visible-devices", default=_PRE_ARGS.cuda_visible_devices)
    parser.add_argument("--expect-device-kind", choices=("any", "cpu", "gpu"), default="gpu")
    parser.add_argument(
        "--manual-reverse-compiler",
        choices=("eager", "tf-function", "xla"),
        default="xla",
    )
    parser.add_argument("--manual-reverse-warmups", type=int, default=0)
    parser.add_argument("--manual-reverse-repeats", type=int, default=0)
    parser.add_argument("--progress-output", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--markdown-output", default="")
    args = parser.parse_args()

    args.batch_seeds = _parse_int_csv(args.batch_seeds)
    args.theta_values = _parse_float_csv(args.theta, expected=3)
    args.candidate_steps_values = [int(item) for item in _parse_float_csv(args.candidate_steps)]
    args.regression_offsets_values = _parse_float_csv(args.regression_offsets)
    args.base_step_ladder_values = [float(args.base_step)]
    args.adaptive_step_factor_values = [1.0]
    args.direction_filter_values = list(p8p.PARAMETER_NAMES)
    args.trim_extreme_offsets = int(args.trim_extreme_values)
    args.trim_extreme_mode = "value"
    args.fd_evaluation_mode = "batched-theta"
    args.basis_set = "raw"
    args.transport_plan_mode = "streaming"
    args.transport_ad_mode = "stabilized"
    args.ad_evaluation_mode = "manual-reverse"
    args.fd_mode = "enabled"
    args.sinkhorn_iterations = "varies_by_candidate"
    args.manual_score_decomposition = False
    args.memory_sample_output = ""
    args.memory_sample_interval_seconds = 30.0
    args.phase_label = "P8p SIR Sinkhorn budget hypothesis diagnostic"

    if args.time_steps <= 0:
        raise ValueError("time-steps must be positive")
    if args.num_particles <= 1:
        raise ValueError("num-particles must be greater than one")
    if any(step <= 0 for step in args.candidate_steps_values):
        raise ValueError("candidate steps must be positive")
    if args.base_step <= 0.0:
        raise ValueError("base-step must be positive")
    if args.trim_extreme_values < 0:
        raise ValueError("trim-extreme-values must be nonnegative")
    if 2 * args.trim_extreme_values >= len(args.regression_offsets_values) - 2:
        raise ValueError("trim-extreme-values leaves too few regression points")
    if args.row_residual_threshold <= 0.0:
        raise ValueError("row-residual-threshold must be positive")
    if args.row_chunk_size <= 0 or args.col_chunk_size <= 0 or args.particle_chunk_size <= 0:
        raise ValueError("chunk sizes must be positive")
    if args.seed_microbatch_size < 0 or args.theta_offset_batch_size < 0:
        raise ValueError("microbatch sizes must be nonnegative")
    return args


def _write_progress(args: argparse.Namespace, payload: dict[str, Any]) -> None:
    if not args.progress_output:
        return
    progress_path = Path(args.progress_output)
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _args_for_steps(args: argparse.Namespace, steps: int) -> argparse.Namespace:
    cloned = copy.copy(args)
    cloned.sinkhorn_iterations = int(steps)
    return cloned


def _value_residual_diagnostic(
    contexts: list[dict[str, Any]],
    theta_values: list[float],
) -> dict[str, Any]:
    theta_components = p8p._theta_components(theta_values)
    weighted_objectives = []
    log_likelihoods = []
    row_residuals = []
    column_residuals = []
    total_seeds = 0
    for context in contexts:
        value = p8p._value_core(
            tensors=context["tensors"],
            args=context["args"],
            theta_components=theta_components,
        )
        weight = len(context["seeds"])
        weighted_objectives.append(tf.reduce_mean(value.log_likelihood) * tf.cast(weight, p8p.DTYPE))
        log_likelihoods.append(value.log_likelihood)
        row_residuals.append(value.max_row_residual)
        column_residuals.append(value.max_column_residual)
        total_seeds += weight
    objective = tf.add_n(weighted_objectives) / tf.cast(total_seeds, p8p.DTYPE)
    all_values = tf.concat(log_likelihoods, axis=0)
    max_row = tf.reduce_max(tf.stack(row_residuals))
    max_column = tf.reduce_max(tf.stack(column_residuals))
    return {
        "objective": float(objective.numpy()),
        "log_likelihood": p8p._to_float_list(all_values),
        "max_row_residual": float(max_row.numpy()),
        "max_column_residual_reported": float(max_column.numpy()),
        "column_residual_note": (
            "manual streaming route reports column residual as 0.0 because the "
            "full transport matrix is not materialized; row residual is the "
            "active convergence diagnostic in this run"
        ),
    }


def _regression_for_steps(
    contexts: list[dict[str, Any]],
    args: argparse.Namespace,
    theta0: tf.Tensor,
    gradient: tf.Tensor,
) -> list[dict[str, Any]]:
    direction_results = []
    raw_directions = tf.eye(len(p8p.PARAMETER_NAMES), dtype=p8p.DTYPE)
    for index, name in enumerate(p8p.PARAMETER_NAMES):
        direction_results.append(
            p8p_reg._regression_diagnostic_for_direction(
                contexts,
                args,
                theta0,
                gradient,
                raw_directions[:, index],
                basis_name="raw_theta",
                direction_name=name,
                base_step=float(args.base_step),
            )
        )
    return direction_results


def _route_prerequisite_gate(
    args: argparse.Namespace,
    *,
    precision: dict[str, Any],
    output_devices: list[str],
    gradient_diag: dict[str, Any],
) -> dict[str, Any]:
    compiler = gradient_diag.get("compiler", {})
    score_route = gradient_diag.get("score_route", "")
    gradient_tensor = tf.convert_to_tensor(gradient_diag["gradient_tensor"], dtype=p8p.DTYPE)
    checks = {
        "device_scope_visible": args.device_scope == "visible",
        "expect_device_kind_gpu": args.expect_device_kind == "gpu",
        "outputs_on_gpu": all("GPU" in str(device).upper() for device in output_devices),
        "dtype_float32": args.dtype == "float32",
        "tf32_enabled": bool(precision.get("tf32_execution_enabled", False)),
        "manual_reverse_compiler_xla": args.manual_reverse_compiler == "xla",
        "compiler_jit_compile": bool(compiler.get("jit_compile", False)),
        "manual_score_route": score_route == "manual_reverse_scan_no_autodiff",
        "streaming_transport_plan": args.transport_plan_mode == "streaming",
        "stabilized_transport_ad": args.transport_ad_mode == "stabilized",
        "manual_streaming_transport_gradient": args.transport_gradient_mode
        in {
            p8p.core_tf.MANUAL_STREAMING_FINITE_TRANSPORT_GRADIENT_MODE,
            p8p.core_tf.MANUAL_STREAMING_BLOCKWISE_VJP_FINITE_TRANSPORT_GRADIENT_MODE,
        },
        "finite_objective": math.isfinite(float(gradient_diag["objective"])),
        "finite_gradient": bool(tf.reduce_all(tf.math.is_finite(gradient_tensor)).numpy()),
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "route_prerequisite_pass": not failed,
        "checks": checks,
        "failed_checks": failed,
        "score_route": score_route,
        "compiler": compiler,
    }


def _sir_hmc_direction_gate(
    *,
    manual_gradient: float,
    regression_fd_slope: float,
    regression_slope_standard_error: float,
    seed_gradient_standard_error: float,
    row_residual_pass: bool,
    route_prerequisite_pass: bool = True,
    route_prerequisite_failed_checks: list[str] | None = None,
    ladder_certificate: bool = False,
) -> dict[str, Any]:
    manual = float(manual_gradient)
    fd_slope = float(regression_fd_slope)
    slope_se = float(regression_slope_standard_error)
    seed_se = float(seed_gradient_standard_error)
    finite_values = math.isfinite(manual) and math.isfinite(fd_slope)
    finite_uncertainty = (
        math.isfinite(slope_se)
        and math.isfinite(seed_se)
        and slope_se >= 0.0
        and seed_se >= 0.0
    )
    residual = manual - fd_slope if finite_values else float("nan")
    combined_se = (
        math.sqrt(slope_se * slope_se + seed_se * seed_se)
        if finite_uncertainty
        else float("nan")
    )
    direction_scale = (
        max(abs(fd_slope), abs(manual), 1.0) if finite_values else float("nan")
    )
    precision_half_width = 2.0 * combined_se if finite_uncertainty else float("nan")
    precision_limit = (
        0.25 * direction_scale
        if finite_values and math.isfinite(direction_scale)
        else float("nan")
    )
    precision_pass = bool(
        finite_values
        and finite_uncertainty
        and precision_half_width <= precision_limit
    )
    if finite_values and finite_uncertainty:
        if combined_se > 0.0:
            combined_z = residual / combined_se
            within_2 = abs(residual) <= 2.0 * combined_se
            within_4 = abs(residual) <= 4.0 * combined_se
            non_negligible = (
                abs(manual) > 2.0 * combined_se
                or abs(fd_slope) > 2.0 * combined_se
            )
        else:
            combined_z = 0.0 if residual == 0.0 else math.copysign(float("inf"), residual)
            within_2 = residual == 0.0
            within_4 = residual == 0.0
            non_negligible = abs(manual) > 0.0 or abs(fd_slope) > 0.0
    else:
        combined_z = None
        within_2 = False
        within_4 = False
        non_negligible = False
    if finite_values:
        relative_error = abs(residual) / max(abs(fd_slope), 1.0)
        near_equal_supportive = relative_error < 0.01
    else:
        relative_error = None
        near_equal_supportive = False
    same_sign = (
        True
        if not non_negligible
        else bool((manual > 0.0 and fd_slope > 0.0) or (manual < 0.0 and fd_slope < 0.0))
    )
    near_zero_direction = bool(
        finite_values
        and finite_uncertainty
        and abs(manual) <= 2.0 * combined_se
        and abs(fd_slope) <= 2.0 * combined_se
    )

    numeric_direction_pass = False
    numeric_reason = "failed_hmc_direction_gate"
    if not finite_values:
        numeric_reason = "nonfinite_value_veto"
    elif not finite_uncertainty:
        numeric_reason = "nonfinite_uncertainty_veto"
    elif not row_residual_pass:
        numeric_reason = "row_residual_veto"
    elif not precision_pass:
        numeric_reason = "inconclusive_precision_veto"
    elif not same_sign:
        numeric_reason = "sign_disagreement_veto"
    elif within_2:
        numeric_direction_pass = True
        numeric_reason = "within_2_combined_se"
    elif within_4 and ladder_certificate:
        numeric_direction_pass = True
        numeric_reason = "within_4_combined_se_with_ladder_certificate"
    elif within_4:
        numeric_reason = "within_4_combined_se_requires_ladder_certificate"

    route_failed = list(route_prerequisite_failed_checks or [])
    direction_pass = bool(route_prerequisite_pass and numeric_direction_pass)
    reason = numeric_reason if route_prerequisite_pass else "route_prerequisite_veto"

    return {
        "direction_pass": direction_pass,
        "direction_gate_reason": reason,
        "numeric_direction_pass": numeric_direction_pass,
        "numeric_direction_gate_reason": numeric_reason,
        "route_prerequisite_pass": bool(route_prerequisite_pass),
        "route_prerequisite_failed_checks": route_failed,
        "manual_gradient": manual,
        "regression_fd_slope": fd_slope,
        "manual_minus_regression_fd": residual,
        "regression_slope_standard_error": slope_se,
        "seed_gradient_standard_error": seed_se,
        "combined_se": combined_se,
        "manual_minus_fd_over_combined_se": combined_z,
        "within_2_combined_se": within_2,
        "within_4_combined_se": within_4,
        "ladder_certificate": bool(ladder_certificate),
        "direction_scale": direction_scale,
        "precision_half_width": precision_half_width,
        "precision_limit": precision_limit,
        "precision_pass": precision_pass,
        "row_residual_pass": bool(row_residual_pass),
        "relative_error_to_regression_fd": relative_error,
        "near_equal_supportive": near_equal_supportive,
        "non_negligible_direction": bool(non_negligible),
        "same_sign_or_near_zero": bool(same_sign),
        "near_zero_direction": near_zero_direction,
    }


def _summarize_direction(
    record: dict[str, Any],
    mc_noise: dict[str, dict[str, Any]],
    *,
    row_residual_pass: bool,
    route_prerequisite_gate: dict[str, Any],
    ladder_certificate: bool = False,
) -> dict[str, Any]:
    parameter = record["direction_name"]
    se = record["regression_slope_standard_error"]
    residual = record["ad_minus_regression_slope"]
    z = None if se <= 0.0 else residual / se
    seed_se = mc_noise[parameter]["standard_error_of_batch_mean"]
    gate = _sir_hmc_direction_gate(
        manual_gradient=record["ad_directional_derivative"],
        regression_fd_slope=record["regression_slope"],
        regression_slope_standard_error=se,
        seed_gradient_standard_error=seed_se,
        row_residual_pass=row_residual_pass,
        route_prerequisite_pass=bool(route_prerequisite_gate["route_prerequisite_pass"]),
        route_prerequisite_failed_checks=list(route_prerequisite_gate["failed_checks"]),
        ladder_certificate=ladder_certificate,
    )
    return {
        "parameter": parameter,
        "manual_gradient": record["ad_directional_derivative"],
        "regression_fd_slope": record["regression_slope"],
        "manual_minus_regression_fd": residual,
        "regression_slope_standard_error": se,
        "seed_gradient_standard_error": seed_se,
        "manual_minus_fd_over_slope_se": z,
        "within_2_slope_se": None if z is None else abs(z) <= 2.0,
        "manual_minus_fd_over_combined_se": gate["manual_minus_fd_over_combined_se"],
        "combined_se": gate["combined_se"],
        "within_2_combined_se": gate["within_2_combined_se"],
        "within_4_combined_se": gate["within_4_combined_se"],
        "direction_scale": gate["direction_scale"],
        "precision_pass": gate["precision_pass"],
        "direction_pass": gate["direction_pass"],
        "direction_gate_reason": gate["direction_gate_reason"],
        "numeric_direction_pass": gate["numeric_direction_pass"],
        "numeric_direction_gate_reason": gate["numeric_direction_gate_reason"],
        "route_prerequisite_pass": gate["route_prerequisite_pass"],
        "route_prerequisite_failed_checks": gate["route_prerequisite_failed_checks"],
        "near_equal_supportive": gate["near_equal_supportive"],
        "near_zero_direction": gate["near_zero_direction"],
        "same_sign_or_near_zero": gate["same_sign_or_near_zero"],
        "hmc_direction_gate": gate,
        "regression_r_squared": record["regression_r_squared"],
        "regression_max_abs_residual": record["regression_max_abs_residual"],
        "trim_extreme_points": record["trim_extreme_points"],
    }


def _interpret(records: list[dict[str, Any]], row_threshold: float) -> dict[str, Any]:
    if not records:
        return {"status": "blocked", "reason": "no records"}
    row10 = next((item for item in records if item["sinkhorn_steps"] == 10), records[0])
    best = min(records, key=lambda item: item["max_abs_fd_z_finite_or_inf"])
    lowest_row = min(records, key=lambda item: item["transport"]["max_row_residual"])
    residuals_drop = lowest_row["transport"]["max_row_residual"] < row10["transport"]["max_row_residual"]
    row_pass_exists = any(item["transport"]["row_residual_pass"] for item in records)
    hmc_pass_exists = any(item.get("all_raw_directions_hmc_direction_pass", False) for item in records)
    fd_z_drop = best["max_abs_fd_z_finite_or_inf"] < row10["max_abs_fd_z_finite_or_inf"]
    if row10["transport"]["max_row_residual"] > row_threshold and residuals_drop and fd_z_drop:
        status = "supports_budget_as_possible_contributor"
    elif row10["transport"]["max_row_residual"] > row_threshold and residuals_drop and not fd_z_drop:
        status = "residual_budget_improves_but_gradient_gap_persists"
    elif row10["transport"]["max_row_residual"] <= row_threshold:
        status = "budget10_row_residual_already_passes_this_screen"
    else:
        status = "inconclusive"
    return {
        "status": status,
        "row10_residual": row10["transport"]["max_row_residual"],
        "lowest_row_steps": lowest_row["sinkhorn_steps"],
        "lowest_row_residual": lowest_row["transport"]["max_row_residual"],
        "best_fd_z_steps": best["sinkhorn_steps"],
        "best_max_abs_fd_z": best["max_abs_fd_z_finite_or_inf"],
        "row_pass_exists": row_pass_exists,
        "hmc_direction_pass_exists": hmc_pass_exists,
        "residuals_drop": residuals_drop,
        "fd_z_drop_from_steps10": fd_z_drop,
        "caution": (
            "This diagnostic can implicate finite Sinkhorn budget, but it cannot "
            "certify SIR gradient correctness or rule out reset/covariance or "
            "objective-semantics errors."
        ),
    }


def _render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# P8p SIR Sinkhorn Budget Hypothesis Diagnostic",
        "",
        f"Date: {result['timestamp_utc']}",
        "",
        f"Status: `{result['interpretation']['status']}`",
        "",
        "## Question",
        "",
        (
            "Does the P8p SIR d18 gradient mismatch track finite Sinkhorn "
            "under-convergence when only the transport budget is varied?"
        ),
        "",
        "## Evidence Contract",
        "",
        "- Same fixed-randomness SIR target, theta, seeds, route, chunks, dtype, and TF32 policy across candidate budgets.",
        "- Primary transport veto: streaming row residual must be below the predeclared threshold.",
        "- FD comparator: 13-point raw-coordinate regression FD, dropping the lowest and highest objective values before fitting.",
        "- Report slope standard error, seed-gradient MCSE, combined SE, precision vetoes, and direction pass reasons explicitly.",
        "- No SIR gradient correctness, HMC readiness, production readiness, or posterior claim is made.",
        "",
        "## Run Summary",
        "",
        f"- Shape: `T={result['shape']['time_steps']}`, `N={result['shape']['num_particles']}`, seeds `{result['batch_seeds']}`",
        f"- Candidate steps: `{result['candidate_steps']}`",
        f"- Row residual threshold: `{result['row_residual_threshold']}`",
        f"- Device expectation: `{result['expect_device_kind']}`, outputs `{result['output_devices']}`",
        "",
        "## Budget Table",
        "",
        "| Steps | route pass | row residual | row pass | max slope z | max combined z | all HMC direction pass | objective |",
        "| ---: | --- | ---: | --- | ---: | ---: | --- | ---: |",
    ]
    for record in result["records"]:
        lines.append(
            "| "
            f"{record['sinkhorn_steps']} | "
            f"{record['route_prerequisites']['route_prerequisite_pass']} | "
            f"{record['transport']['max_row_residual']:.6e} | "
            f"{record['transport']['row_residual_pass']} | "
            f"{record['max_abs_fd_z_finite_or_inf']:.3f} | "
            f"{record['max_abs_combined_z_finite_or_inf']:.3f} | "
            f"{record['all_raw_directions_hmc_direction_pass']} | "
            f"{record['transport']['objective']:.6f} |"
        )
    lines.extend(["", "## Direction Details", ""])
    for record in result["records"]:
        lines.extend(
            [
                f"### Steps {record['sinkhorn_steps']}",
                "",
                "| Parameter | manual grad | FD slope | manual-FD | slope SE | seed MCSE | combined SE | combined z | precision pass | direction pass | reason | supportive |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |",
            ]
        )
        for item in record["raw_direction_summaries"]:
            z = item["manual_minus_fd_over_combined_se"]
            z_text = "NA" if z is None else f"{z:.3f}"
            lines.append(
                "| "
                f"`{item['parameter']}` | "
                f"{item['manual_gradient']:.6f} | "
                f"{item['regression_fd_slope']:.6f} | "
                f"{item['manual_minus_regression_fd']:.6f} | "
                f"{item['regression_slope_standard_error']:.6f} | "
                f"{item['seed_gradient_standard_error']:.6f} | "
                f"{item['combined_se']:.6f} | "
                f"{z_text} | "
                f"{item['precision_pass']} | "
                f"{item['direction_pass']} | "
                f"`{item['direction_gate_reason']}` | "
                f"{item['near_equal_supportive']} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Interpretation",
            "",
            json.dumps(result["interpretation"], indent=2, sort_keys=True),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    raise RuntimeError("ARCHIVAL_WRONG_TRANSPORT_CHUNK_POLICY: this route is preserved only as provenance and cannot emit new evidence")
    args = _parse_args()
    start = time.perf_counter()
    precision = p8p._configure_precision(args)
    physical_gpus, logical_gpus = p8p._configure_gpus()
    theta0 = tf.constant(args.theta_values, dtype=p8p.DTYPE)
    memory_before = p8p._gpu_memory_info()
    records = []
    completed: list[dict[str, Any]] = []
    with tf.device(args.device):
        output_devices: list[str] | None = None
        for steps in args.candidate_steps_values:
            step_args = _args_for_steps(args, steps)
            contexts, sir_semantics = p8p_reg._build_microbatch_contexts(step_args)
            _write_progress(
                args,
                {
                    "stage": "budget_started",
                    "steps": int(steps),
                    "completed": completed,
                    "elapsed_seconds": time.perf_counter() - start,
                },
            )
            value_diag = _value_residual_diagnostic(contexts, step_args.theta_values)
            gradient_diag = p8p_reg._manual_gradient_diagnostic_for_contexts(
                contexts,
                step_args.theta_values,
                compiler=step_args.manual_reverse_compiler,
                warmups=step_args.manual_reverse_warmups,
                repeats=step_args.manual_reverse_repeats,
            )
            if output_devices is None:
                output_devices = p8p_reg._validate_device(
                    (
                        tf.convert_to_tensor(gradient_diag["objective"], dtype=p8p.DTYPE),
                        tf.convert_to_tensor(gradient_diag["gradient_tensor"], dtype=p8p.DTYPE),
                    ),
                    args.expect_device_kind,
                )
            route_gate = _route_prerequisite_gate(
                step_args,
                precision=precision,
                output_devices=output_devices,
                gradient_diag=gradient_diag,
            )
            direction_details = _regression_for_steps(
                contexts,
                step_args,
                theta0,
                tf.convert_to_tensor(gradient_diag["gradient_tensor"], dtype=p8p.DTYPE),
            )
            mc_noise = p8p._mc_noise_summary(
                tf.convert_to_tensor(gradient_diag["per_seed_gradient"], dtype=p8p.DTYPE)
            )
            row_residual_pass = bool(value_diag["max_row_residual"] <= args.row_residual_threshold)
            summaries = [
                _summarize_direction(
                    item,
                    mc_noise,
                    row_residual_pass=row_residual_pass,
                    route_prerequisite_gate=route_gate,
                    ladder_certificate=False,
                )
                for item in direction_details
            ]
            z_values = [
                abs(float(item["manual_minus_fd_over_slope_se"]))
                for item in summaries
                if item["manual_minus_fd_over_slope_se"] is not None
                and math.isfinite(float(item["manual_minus_fd_over_slope_se"]))
            ]
            max_abs_z = max(z_values) if z_values else float("inf")
            combined_z_values = [
                abs(float(item["manual_minus_fd_over_combined_se"]))
                for item in summaries
                if item["manual_minus_fd_over_combined_se"] is not None
                and math.isfinite(float(item["manual_minus_fd_over_combined_se"]))
            ]
            max_abs_combined_z = max(combined_z_values) if combined_z_values else float("inf")
            record = {
                "sinkhorn_steps": int(steps),
                "transport": {
                    **value_diag,
                    "row_residual_threshold": float(args.row_residual_threshold),
                    "row_residual_pass": row_residual_pass,
                },
                "gradient": {
                    "manual_gradient_values": p8p._to_float_list(
                        tf.convert_to_tensor(gradient_diag["gradient_tensor"], dtype=p8p.DTYPE)
                    ),
                    "per_seed_gradient": p8p._to_float_matrix(
                        tf.convert_to_tensor(gradient_diag["per_seed_gradient"], dtype=p8p.DTYPE)
                    ),
                    "monte_carlo_gradient_noise": mc_noise,
                    "score_route": gradient_diag.get("score_route", "manual_reverse_scan_no_autodiff"),
                },
                "route_prerequisites": route_gate,
                "raw_direction_summaries": summaries,
                "raw_direction_details": direction_details,
                "max_abs_fd_z_finite_or_inf": float(max_abs_z),
                "max_abs_combined_z_finite_or_inf": float(max_abs_combined_z),
                "all_raw_directions_within_2se": bool(
                    summaries
                    and all(item["within_2_slope_se"] is True for item in summaries)
                ),
                "all_raw_directions_hmc_direction_pass": bool(
                    summaries
                    and all(item["direction_pass"] is True for item in summaries)
                ),
            }
            records.append(record)
            completed.append(
                {
                    "steps": int(steps),
                    "max_row_residual": value_diag["max_row_residual"],
                    "max_abs_fd_z": float(max_abs_z),
                }
            )
            _write_progress(
                args,
                {
                    "stage": "budget_complete",
                    "steps": int(steps),
                    "completed": completed,
                    "elapsed_seconds": time.perf_counter() - start,
                },
            )
    memory_after = p8p._gpu_memory_info()
    if output_devices is None:
        raise RuntimeError("no candidate records were evaluated")
    result = {
        "schema_version": "filter_bench.p8p_sir_sinkhorn_budget_hypothesis.v1",
        "timestamp_utc": _dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
        "host": platform.node(),
        "python_version": platform.python_version(),
        "tensorflow_version": tf.__version__,
        "git_commit": p8p._git_commit(),
        "elapsed_seconds": time.perf_counter() - start,
        "precision": precision,
        "physical_gpus": physical_gpus,
        "logical_gpus": logical_gpus,
        "device": args.device,
        "device_scope": args.device_scope,
        "expect_device_kind": args.expect_device_kind,
        "output_devices": output_devices,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "shape": {
            "batch_size": len(args.batch_seeds),
            "time_steps": int(args.time_steps),
            "num_particles": int(args.num_particles),
            "state_dim": 18,
            "obs_dim": 9,
            "parameter_dim": 3,
        },
        "batch_seeds": [int(seed) for seed in args.batch_seeds],
        "theta": dict(zip(p8p.PARAMETER_NAMES, [float(x) for x in args.theta_values], strict=True)),
        "candidate_steps": [int(step) for step in args.candidate_steps_values],
        "row_residual_threshold": float(args.row_residual_threshold),
        "regression_fd_contract": {
            "basis": "raw_theta",
            "offsets": [float(item) for item in args.regression_offsets_values],
            "base_step": float(args.base_step),
            "trim": "drop lowest and highest objective values before regression",
            "trim_extreme_values": int(args.trim_extreme_values),
            "fd_evaluation_mode": args.fd_evaluation_mode,
            "theta_offset_batch_size": int(args.theta_offset_batch_size),
        },
        "transport": p8p_reg._transport_metadata(args),
        "sir_semantics": sir_semantics,
        "records": records,
        "interpretation": _interpret(records, float(args.row_residual_threshold)),
        "gpu_memory_info_before": memory_before,
        "gpu_memory_info_after": memory_after,
        "nonclaims": [
            "not SIR gradient correctness",
            "not HMC readiness",
            "not posterior correctness",
            "not production readiness",
            "not evidence that Sinkhorn budget is the only remaining issue",
        ],
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.markdown_output:
        markdown_path = Path(args.markdown_output)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(_render_markdown(result), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
