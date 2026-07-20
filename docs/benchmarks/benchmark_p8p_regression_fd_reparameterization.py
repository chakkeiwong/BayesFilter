"""P8p regression finite-difference diagnostics in reparameterized directions.

This companion diagnostic keeps the P8p SIR d18 target and TF32 route intact,
but replaces single-pair finite differences with symmetric local linear
regression along raw, physics-informed, and locally whitened directions.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os
import platform
import sys
import threading
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


def _parse_int_csv(value: str) -> list[int]:
    parsed = [int(item.strip()) for item in str(value).split(",") if item.strip()]
    if not parsed:
        raise ValueError("expected at least one integer")
    return parsed


def _parse_float_csv(value: str, *, expected: int) -> list[float]:
    parsed = [float(item.strip()) for item in str(value).split(",") if item.strip()]
    if len(parsed) != expected:
        raise ValueError(f"expected {expected} comma-separated floats")
    if not all(math.isfinite(item) for item in parsed):
        raise ValueError("expected finite floats")
    return parsed


def _parse_offsets(value: str) -> list[float]:
    parsed = [float(item.strip()) for item in str(value).split(",") if item.strip()]
    if len(parsed) not in (7, 9, 13, 15, 17):
        raise ValueError("regression offsets must contain 7, 9, 13, 15, or 17 values")
    if not all(math.isfinite(item) for item in parsed):
        raise ValueError("regression offsets must be finite")
    if not any(item < 0.0 for item in parsed) or not any(item > 0.0 for item in parsed):
        raise ValueError("regression offsets must include negative and positive values")
    return parsed


def _parse_positive_float_csv(value: str) -> list[float]:
    parsed = [float(item.strip()) for item in str(value).split(",") if item.strip()]
    if not parsed:
        raise ValueError("expected at least one positive float")
    if not all(math.isfinite(item) and item > 0.0 for item in parsed):
        raise ValueError("base steps must be positive finite values")
    return parsed


def _parse_string_csv(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--batch-seeds", default="81120,81121,81122,81123,81124")
    parser.add_argument("--time-steps", type=int, default=3)
    parser.add_argument("--num-particles", type=int, default=16)
    parser.add_argument("--theta", default="0.02,-0.01,0.01")
    parser.add_argument("--phase-label", default="P8p regression FD reparameterization")
    parser.add_argument("--base-step", type=float, default=0.001)
    parser.add_argument("--base-step-ladder", default="")
    parser.add_argument(
        "--base-step-mode",
        choices=("fixed", "ad-signal"),
        default="fixed",
        help=(
            "fixed uses --base-step-ladder as absolute steps; ad-signal chooses "
            "a direction-specific step from the AD directional derivative and "
            "uses --adaptive-step-factors for nested windows"
        ),
    )
    parser.add_argument("--target-objective-delta", type=float, default=0.15)
    parser.add_argument("--adaptive-step-factors", default="1.0,0.5,0.25")
    parser.add_argument("--min-adaptive-base-step", type=float, default=2.5e-4)
    parser.add_argument("--max-adaptive-base-step", type=float, default=5.0e-2)
    parser.add_argument("--regression-offsets", default="-4,-3,-2,-1,0,1,2,3,4")
    parser.add_argument(
        "--trim-extreme-offsets",
        type=int,
        default=0,
        help=(
            "number of largest and smallest FD offsets to evaluate but exclude "
            "from the regression fit"
        ),
    )
    parser.add_argument(
        "--trim-extreme-mode",
        choices=("offset", "value"),
        default="offset",
        help=(
            "offset drops the smallest/largest FD offsets; value drops the "
            "lowest/highest mean objective values after seed aggregation"
        ),
    )
    parser.add_argument(
        "--fd-evaluation-mode",
        choices=("serial", "batched-theta"),
        default="serial",
        help="batched-theta folds offset points into the filter batch axis",
    )
    parser.add_argument(
        "--theta-offset-batch-size",
        type=int,
        default=0,
        help=(
            "maximum number of FD theta offsets per batched-theta value call; "
            "0 evaluates all offsets in one call"
        ),
    )
    parser.add_argument(
        "--basis-set",
        choices=(
            "raw-physics-whitened",
            "raw",
            "physics",
            "whitened",
            "semantic-orthogonal",
        ),
        default="raw-physics-whitened",
    )
    parser.add_argument("--transport-policy", choices=("active-all", "active-odd", "no-resampling"), default="active-all")
    parser.add_argument("--sinkhorn-iterations", type=int, default=10)
    parser.add_argument("--sinkhorn-epsilon", type=float, default=1.0)
    parser.add_argument("--annealed-scaling", type=float, default=0.9)
    parser.add_argument("--annealed-convergence-threshold", type=float, default=1.0e-3)
    parser.add_argument(
        "--transport-plan-mode",
        choices=("streaming", "dense"),
        default="streaming",
    )
    parser.add_argument(
        "--transport-gradient-mode",
        choices=(
            "raw",
            p8p.core_tf.MANUAL_STREAMING_FINITE_TRANSPORT_GRADIENT_MODE,
            p8p.core_tf.MANUAL_STREAMING_BLOCKWISE_VJP_FINITE_TRANSPORT_GRADIENT_MODE,
        ),
        default="raw",
    )
    parser.add_argument(
        "--transport-ad-mode",
        choices=(
            "stabilized",
            "diff-scale",
            "diff-keys",
            "diff-potentials",
            "full",
        ),
        default="stabilized",
    )
    parser.add_argument("--row-chunk-size", type=int, default=16)
    parser.add_argument("--col-chunk-size", type=int, default=16)
    parser.add_argument("--particle-chunk-size", type=int, default=16)
    parser.add_argument("--dtype", choices=("float64", "float32"), default="float32")
    parser.add_argument("--tf32-mode", choices=("default", "enabled", "disabled"), default="enabled")
    parser.add_argument("--device", default="/GPU:0")
    parser.add_argument("--device-scope", choices=("cpu", "visible"), default=_PRE_ARGS.device_scope)
    parser.add_argument("--cuda-visible-devices", default=_PRE_ARGS.cuda_visible_devices)
    parser.add_argument("--expect-device-kind", choices=("any", "cpu", "gpu"), default="gpu")
    parser.add_argument(
        "--seed-microbatch-size",
        type=int,
        default=0,
        help=(
            "evaluate seed groups sequentially and combine them exactly as the "
            "batch mean; 0 means evaluate all seeds in one batch"
        ),
    )
    parser.add_argument(
        "--ad-evaluation-mode",
        choices=("reverse-gradient", "forward-jvp", "manual-reverse"),
        default="reverse-gradient",
        help=(
            "manual-reverse computes the opt-in no-autodiff manual score; "
            "reverse-gradient and forward-jvp are diagnostic-only modes"
        ),
    )
    parser.add_argument(
        "--manual-reverse-compiler",
        choices=("eager", "tf-function", "xla"),
        default="xla",
        help=(
            "compiler for manual-reverse seed microbatch value/score; "
            "defaults to xla because non-XLA manual-reverse is a known "
            "misleadingly slow diagnostic route"
        ),
    )
    parser.add_argument(
        "--manual-reverse-warmups",
        type=int,
        default=0,
        help="warm calls after compile for manual-reverse compiler timing",
    )
    parser.add_argument(
        "--manual-reverse-repeats",
        type=int,
        default=0,
        help="timed warm calls after compile and warmups for manual-reverse compiler timing",
    )
    parser.add_argument(
        "--manual-score-decomposition",
        action="store_true",
        help=(
            "for manual-reverse only, return named manual score components; "
            "diagnostic-only because it replays selected SIR VJPs"
        ),
    )
    parser.add_argument(
        "--direction-filter",
        default="",
        help=(
            "optional comma-separated direction names to evaluate within the "
            "selected basis set; empty evaluates all directions"
        ),
    )
    parser.add_argument(
        "--fd-mode",
        choices=("enabled", "ad-only"),
        default="enabled",
        help="ad-only records the objective, gradient, and seed MCSE without FD lines",
    )
    parser.add_argument(
        "--progress-output",
        default="",
        help="optional JSON file updated after AD and after each FD window",
    )
    parser.add_argument(
        "--memory-sample-output",
        default="",
        help=(
            "optional JSON file periodically updated with in-process TensorFlow "
            "GPU allocator current/peak memory"
        ),
    )
    parser.add_argument(
        "--memory-sample-interval-seconds",
        type=float,
        default=30.0,
        help="positive sampling interval for --memory-sample-output",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    args.batch_seeds = _parse_int_csv(args.batch_seeds)
    args.theta_values = _parse_float_csv(args.theta, expected=3)
    args.regression_offsets_values = _parse_offsets(args.regression_offsets)
    args.base_step_ladder_values = (
        _parse_positive_float_csv(args.base_step_ladder)
        if args.base_step_ladder.strip()
        else [float(args.base_step)]
    )
    args.adaptive_step_factor_values = _parse_positive_float_csv(args.adaptive_step_factors)
    args.direction_filter_values = _parse_string_csv(args.direction_filter)
    if args.time_steps <= 0:
        raise ValueError("time_steps must be positive")
    if args.num_particles <= 1:
        raise ValueError("num_particles must be greater than one")
    if args.base_step <= 0.0:
        raise ValueError("base_step must be positive")
    if args.target_objective_delta <= 0.0:
        raise ValueError("target-objective-delta must be positive")
    if args.min_adaptive_base_step <= 0.0 or args.max_adaptive_base_step <= 0.0:
        raise ValueError("adaptive base-step bounds must be positive")
    if args.min_adaptive_base_step > args.max_adaptive_base_step:
        raise ValueError("min-adaptive-base-step must be <= max-adaptive-base-step")
    if args.trim_extreme_offsets < 0:
        raise ValueError("trim-extreme-offsets must be nonnegative")
    if 2 * args.trim_extreme_offsets >= len(args.regression_offsets_values) - 2:
        raise ValueError("trim-extreme-offsets leaves too few regression points")
    if args.theta_offset_batch_size < 0:
        raise ValueError("theta-offset-batch-size must be nonnegative")
    if args.sinkhorn_iterations <= 0:
        raise ValueError("sinkhorn_iterations must be positive")
    if args.row_chunk_size <= 0 or args.col_chunk_size <= 0 or args.particle_chunk_size <= 0:
        raise ValueError("chunk sizes must be positive")
    if args.seed_microbatch_size < 0:
        raise ValueError("seed-microbatch-size must be nonnegative")
    if args.manual_reverse_warmups < 0 or args.manual_reverse_repeats < 0:
        raise ValueError("manual reverse warmups/repeats must be nonnegative")
    if args.manual_score_decomposition and args.ad_evaluation_mode != "manual-reverse":
        raise ValueError("manual-score-decomposition requires ad-evaluation-mode=manual-reverse")
    if args.memory_sample_interval_seconds <= 0.0:
        raise ValueError("memory-sample-interval-seconds must be positive")
    return args


def _as_matrix(rows: list[list[float]]) -> tf.Tensor:
    return tf.constant(rows, dtype=p8p.DTYPE)


def _normalise_columns(matrix: tf.Tensor) -> tf.Tensor:
    norms = tf.linalg.norm(matrix, axis=0)
    return matrix / norms[tf.newaxis, :]


def _metric_inner(metric: tf.Tensor, lhs: tf.Tensor, rhs: tf.Tensor) -> tf.Tensor:
    return tf.tensordot(lhs, tf.linalg.matvec(metric, rhs), axes=1)


def _metric_gram_schmidt_columns(
    columns: tf.Tensor,
    metric: tf.Tensor,
) -> tf.Tensor:
    metric = tf.convert_to_tensor(metric, dtype=p8p.DTYPE)
    output = []
    for column in tf.unstack(columns, axis=1):
        vector = tf.identity(column)
        for previous in output:
            denominator = _metric_inner(metric, previous, previous)
            projection = _metric_inner(metric, previous, vector) / denominator
            vector = vector - projection * previous
        norm = tf.sqrt(tf.maximum(_metric_inner(metric, vector, vector), tf.constant(1.0e-30, p8p.DTYPE)))
        output.append(vector / norm)
    return tf.stack(output, axis=1)


def _basis_matrices(seed_gradient_covariance: tf.Tensor, basis_set: str) -> list[dict[str, Any]]:
    sqrt2 = math.sqrt(2.0)
    raw = _as_matrix(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    physics = _as_matrix(
        [
            [1.0 / sqrt2, 1.0 / sqrt2, 0.0],
            [-1.0 / sqrt2, 1.0 / sqrt2, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    outputs: list[dict[str, Any]] = []
    if basis_set in ("raw-physics-whitened", "raw"):
        outputs.append(
            {
                "basis_name": "raw_theta",
                "direction_names": list(p8p.PARAMETER_NAMES),
                "matrix_columns": raw,
                "construction": "identity in original theta coordinates",
            }
        )
    if basis_set in ("raw-physics-whitened", "physics"):
        outputs.append(
            {
                "basis_name": "physics_rho_tau_omega",
                "direction_names": ["rho_log_kappa_minus_log_nu", "tau_common_rate", "omega_obs_noise"],
                "matrix_columns": physics,
                "construction": "rho=(a-b)/sqrt2, tau=(a+b)/sqrt2, omega=c; columns map coordinate increments to original theta",
            }
        )
    if basis_set in ("raw-physics-whitened", "whitened"):
        jitter = tf.constant(1.0e-6, dtype=p8p.DTYPE)
        cov = tf.convert_to_tensor(seed_gradient_covariance, dtype=p8p.DTYPE)
        eigvals, eigvecs = tf.linalg.eigh(cov + jitter * tf.eye(3, dtype=p8p.DTYPE))
        whitening_columns = eigvecs / tf.sqrt(tf.maximum(eigvals, jitter))[tf.newaxis, :]
        whitening_columns = _normalise_columns(whitening_columns)
        outputs.append(
            {
                "basis_name": "local_seed_gradient_whitened",
                "direction_names": ["whitened_0", "whitened_1", "whitened_2"],
                "matrix_columns": whitening_columns,
                "construction": "eigenvectors of seed-gradient covariance scaled by inverse sqrt eigenvalue and normalized in theta coordinates",
                "seed_gradient_covariance_eigenvalues": [
                    float(item) for item in eigvals.numpy().tolist()
                ],
            }
        )
    if basis_set == "semantic-orthogonal":
        jitter = tf.constant(1.0e-6, dtype=p8p.DTYPE)
        metric = tf.convert_to_tensor(seed_gradient_covariance, dtype=p8p.DTYPE) + jitter * tf.eye(3, dtype=p8p.DTYPE)
        semantic_orthogonal = _metric_gram_schmidt_columns(physics, metric)
        outputs.append(
            {
                "basis_name": "semantic_metric_orthogonal",
                "direction_names": [
                    "rho",
                    "tau_perp_given_rho",
                    "omega_perp_given_rho_tau",
                ],
                "matrix_columns": semantic_orthogonal,
                "construction": (
                    "Gram-Schmidt of rho=(a-b)/sqrt2, tau=(a+b)/sqrt2, omega=c "
                    "under the local seed-gradient covariance metric; columns map "
                    "coordinate increments to original theta"
                ),
                "seed_gradient_metric": p8p._to_float_matrix(metric),
            }
        )
    return outputs


def _linear_regression(xs: tf.Tensor, ys: tf.Tensor) -> dict[str, float]:
    x = tf.convert_to_tensor(xs, dtype=p8p.DTYPE)
    y = tf.convert_to_tensor(ys, dtype=p8p.DTYPE)
    x_mean = tf.reduce_mean(x)
    y_mean = tf.reduce_mean(y)
    x_centered = x - x_mean
    y_centered = y - y_mean
    sxx = tf.reduce_sum(tf.square(x_centered))
    slope = tf.reduce_sum(x_centered * y_centered) / sxx
    intercept = y_mean - slope * x_mean
    fitted = intercept + slope * x
    residuals = y - fitted
    sse = tf.reduce_sum(tf.square(residuals))
    sst = tf.reduce_sum(tf.square(y_centered))
    dof = tf.cast(tf.shape(x)[0] - 2, p8p.DTYPE)
    residual_variance = sse / tf.maximum(dof, tf.constant(1.0, dtype=p8p.DTYPE))
    slope_se = tf.sqrt(residual_variance / sxx)
    r_squared = tf.constant(1.0, dtype=p8p.DTYPE) - sse / tf.maximum(
        sst,
        tf.constant(1.0e-30, dtype=p8p.DTYPE),
    )
    return {
        "slope": float(slope.numpy()),
        "intercept": float(intercept.numpy()),
        "slope_standard_error": float(slope_se.numpy()),
        "residual_sse": float(sse.numpy()),
        "r_squared": float(r_squared.numpy()),
        "max_abs_residual": float(tf.reduce_max(tf.abs(residuals)).numpy()),
    }


def _seed_groups(args: argparse.Namespace) -> list[list[int]]:
    microbatch_size = int(args.seed_microbatch_size)
    if microbatch_size <= 0 or microbatch_size >= len(args.batch_seeds):
        return [[int(seed) for seed in args.batch_seeds]]
    return [
        [int(seed) for seed in args.batch_seeds[start : start + microbatch_size]]
        for start in range(0, len(args.batch_seeds), microbatch_size)
    ]


def _args_for_seed_group(args: argparse.Namespace, seeds: list[int]) -> argparse.Namespace:
    grouped = argparse.Namespace(**vars(args))
    grouped.batch_seeds = [int(seed) for seed in seeds]
    return grouped


def _build_microbatch_contexts(
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    contexts = []
    first_semantics: dict[str, Any] | None = None
    groups = _seed_groups(args)
    for seeds in groups:
        grouped_args = _args_for_seed_group(args, seeds)
        tensors, semantics = p8p._build_base_tensors(grouped_args)
        if first_semantics is None:
            first_semantics = dict(semantics)
        contexts.append(
            {
                "args": grouped_args,
                "seeds": [int(seed) for seed in seeds],
                "tensors": tensors,
            }
        )
    if first_semantics is None:
        raise ValueError("expected at least one seed microbatch")
    first_semantics["seed_microbatching"] = {
        "enabled": len(groups) > 1,
        "seed_microbatch_size": int(args.seed_microbatch_size),
        "seed_groups": [[int(seed) for seed in group] for group in groups],
        "combination_rule": "exact seed-weighted batch mean over independent fixed-seed evaluations",
    }
    return contexts, first_semantics


def _objective_from_contexts(
    contexts: list[dict[str, Any]],
    theta_components: tuple[tf.Tensor, tf.Tensor, tf.Tensor],
) -> tuple[tf.Tensor, tf.Tensor]:
    weighted_objectives = []
    values = []
    total = 0
    for context in contexts:
        objective, value = p8p._objective_from_components(
            context["tensors"],
            context["args"],
            theta_components,
        )
        weight = len(context["seeds"])
        weighted_objectives.append(objective * tf.cast(weight, p8p.DTYPE))
        values.append(value)
        total += weight
    combined_objective = tf.add_n(weighted_objectives) / tf.cast(total, p8p.DTYPE)
    return combined_objective, tf.concat(values, axis=0)


def _gradient_diagnostic_for_contexts(
    contexts: list[dict[str, Any]],
    theta_values: list[float],
) -> dict[str, Any]:
    per_seed_gradients = []
    values = []
    connected = []
    for context in contexts:
        diag = p8p._gradient_diagnostic(
            context["tensors"],
            context["args"],
            theta_values,
        )
        per_seed_gradients.append(tf.convert_to_tensor(diag["per_seed_gradient"], dtype=p8p.DTYPE))
        values.append(tf.convert_to_tensor(diag["log_likelihood"], dtype=p8p.DTYPE))
        connected.append(bool(diag["gradients_connected"]))
    per_seed_gradient = tf.concat(per_seed_gradients, axis=0)
    value = tf.concat(values, axis=0)
    return {
        "objective": tf.reduce_mean(value),
        "log_likelihood": value,
        "gradient_tensor": tf.reduce_mean(per_seed_gradient, axis=0),
        "per_seed_gradient": per_seed_gradient,
        "gradients_connected": bool(all(connected)),
    }


def _manual_gradient_diagnostic_for_contexts(
    contexts: list[dict[str, Any]],
    theta_values: list[float],
    *,
    compiler: str = "xla",
    warmups: int = 0,
    repeats: int = 0,
    return_score_decomposition: bool = False,
) -> dict[str, Any]:
    if compiler not in {"eager", "tf-function", "xla"}:
        raise ValueError("compiler must be eager, tf-function, or xla")
    per_seed_gradients = []
    values = []
    connected = []
    compile_and_first_call_timings = []
    warm_call_timings_by_context = []
    score_components = []
    for context in contexts:
        theta_tensor = tf.constant(theta_values, dtype=p8p.DTYPE)

        def materialize(*items: tf.Tensor) -> None:
            for item in items:
                _ = item.numpy()

        def run_manual(theta_row: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
            theta_components = tuple(tf.unstack(theta_row, axis=0))
            manual = p8p._manual_value_and_score_from_components(
                context["tensors"],
                context["args"],
                theta_components,  # type: ignore[arg-type]
                return_score_decomposition=return_score_decomposition,
            )
            component_tensor = (
                manual["manual_score_components"]
                if return_score_decomposition
                else tf.zeros(
                    [0, 0, len(p8p.PARAMETER_NAMES)],
                    dtype=p8p.DTYPE,
                )
            )
            return manual["log_likelihood"], manual["per_seed_gradient"], component_tensor

        if compiler == "eager":
            value, per_seed_gradient, component_tensor = run_manual(theta_tensor)
            compile_and_first_call_timings.append(None)
            warm_call_timings_by_context.append([])
        else:
            compiled = tf.function(
                run_manual,
                jit_compile=compiler == "xla",
                reduce_retracing=True,
            )
            compile_start = time.perf_counter()
            value, per_seed_gradient, component_tensor = compiled(theta_tensor)
            materialize(value, per_seed_gradient, component_tensor)
            compile_and_first_call_timings.append(time.perf_counter() - compile_start)
            for _ in range(warmups):
                warm_value, warm_gradient, warm_components = compiled(theta_tensor)
                materialize(warm_value, warm_gradient, warm_components)
            warm_timings = []
            for _ in range(repeats):
                warm_start = time.perf_counter()
                value, per_seed_gradient, component_tensor = compiled(theta_tensor)
                materialize(value, per_seed_gradient, component_tensor)
                warm_timings.append(time.perf_counter() - warm_start)
            warm_call_timings_by_context.append(warm_timings)
        per_seed_gradients.append(tf.convert_to_tensor(per_seed_gradient, dtype=p8p.DTYPE))
        values.append(tf.convert_to_tensor(value, dtype=p8p.DTYPE))
        if return_score_decomposition:
            score_components.append(tf.convert_to_tensor(component_tensor, dtype=p8p.DTYPE))
        connected.append(True)
    per_seed_gradient = tf.concat(per_seed_gradients, axis=0)
    value = tf.concat(values, axis=0)
    result = {
        "objective": tf.reduce_mean(value),
        "log_likelihood": value,
        "gradient_tensor": tf.reduce_mean(per_seed_gradient, axis=0),
        "per_seed_gradient": per_seed_gradient,
        "gradients_connected": bool(all(connected)),
        "score_route": "manual_reverse_scan_no_autodiff",
        "compiler": {
            "mode": compiler,
            "tf_function": compiler in {"tf-function", "xla"},
            "jit_compile": compiler == "xla",
            "compiled_unit": "manual_reverse_seed_microbatch_value_score",
            "compile_and_first_call_timings_seconds": compile_and_first_call_timings,
            "warmups": int(warmups),
            "repeats": int(repeats),
            "warm_call_timings_seconds_by_context": warm_call_timings_by_context,
        },
    }
    if return_score_decomposition:
        component_tensor = tf.concat(score_components, axis=1)
        component_sum = tf.reduce_sum(component_tensor, axis=0)
        result["manual_score_decomposition"] = {
            "component_names": list(p8p.MANUAL_SCORE_COMPONENT_NAMES),
            "components": component_tensor,
            "component_sum": component_sum,
            "component_mean": tf.reduce_mean(component_tensor, axis=1),
            "component_sum_reconstruction_max_abs_delta": tf.reduce_max(
                tf.abs(component_sum - per_seed_gradient)
            ),
        }
    return result


def _forward_jvp_diagnostic_for_contexts(
    contexts: list[dict[str, Any]],
    theta_values: list[float],
) -> dict[str, Any]:
    values: list[tf.Tensor] = []
    per_seed_columns: list[tf.Tensor] = []
    connected = []
    raw_directions = tf.eye(len(p8p.PARAMETER_NAMES), dtype=p8p.DTYPE)
    for direction_index, raw_direction in enumerate(tf.unstack(raw_directions, axis=0)):
        direction_jvps = []
        direction_values = []
        for context in contexts:
            theta_components = p8p._theta_components(theta_values)
            tangent_components = tuple(
                tf.convert_to_tensor(item, dtype=p8p.DTYPE)
                for item in tf.unstack(raw_direction)
            )
            with tf.autodiff.ForwardAccumulator(theta_components, tangent_components) as accumulator:
                _objective, value = p8p._objective_from_components(
                    context["tensors"],
                    context["args"],
                    theta_components,
                )
            value_jvp = accumulator.jvp(value)
            connected.append(value_jvp is not None)
            if value_jvp is None:
                value_jvp = tf.fill(tf.shape(value), tf.constant(float("nan"), dtype=p8p.DTYPE))
            direction_jvps.append(tf.reshape(tf.convert_to_tensor(value_jvp, dtype=p8p.DTYPE), [-1]))
            if direction_index == 0:
                direction_values.append(tf.reshape(tf.convert_to_tensor(value, dtype=p8p.DTYPE), [-1]))
        per_seed_columns.append(tf.concat(direction_jvps, axis=0))
        if direction_index == 0:
            values = direction_values
    value = tf.concat(values, axis=0)
    per_seed_gradient = tf.stack(per_seed_columns, axis=1)
    return {
        "objective": tf.reduce_mean(value),
        "log_likelihood": value,
        "gradient_tensor": tf.reduce_mean(per_seed_gradient, axis=0),
        "per_seed_gradient": per_seed_gradient,
        "gradients_connected": bool(all(connected)),
    }


def _value_at_theta(
    contexts: list[dict[str, Any]],
    theta: tf.Tensor,
) -> tf.Tensor:
    objective, _value = _objective_from_contexts(
        contexts,
        tuple(tf.unstack(tf.convert_to_tensor(theta, dtype=p8p.DTYPE))),  # type: ignore[arg-type]
    )
    return objective


def _batched_theta_contexts(
    contexts: list[dict[str, Any]],
    theta_rows: tf.Tensor,
) -> tuple[list[dict[str, Any]], int]:
    rows = tf.convert_to_tensor(theta_rows, dtype=p8p.DTYPE)
    if rows.shape.rank != 2 or rows.shape[1] != len(p8p.PARAMETER_NAMES):
        raise ValueError("theta_rows must have shape [num_offsets, parameter_dim]")
    num_offsets = int(rows.shape[0])
    if num_offsets <= 0:
        raise ValueError("expected at least one theta row")
    batched_contexts = []
    for context in contexts:
        seeds = [int(seed) for seed in context["seeds"]]
        tensors = context["tensors"]
        tiled_tensors = dict(tensors)
        tiled_tensors["initial_particles"] = tf.repeat(
            tf.convert_to_tensor(tensors["initial_particles"], dtype=p8p.DTYPE),
            repeats=num_offsets,
            axis=0,
        )
        tiled_tensors["fixed_resampling_mask"] = tf.repeat(
            tf.convert_to_tensor(tensors["fixed_resampling_mask"]),
            repeats=num_offsets,
            axis=0,
        )
        tiled_tensors["transition_matrix"] = tf.repeat(
            tf.convert_to_tensor(tensors["transition_matrix"], dtype=p8p.DTYPE),
            repeats=num_offsets,
            axis=0,
        )
        tiled_tensors["transition_covariance"] = tf.repeat(
            tf.convert_to_tensor(tensors["transition_covariance"], dtype=p8p.DTYPE),
            repeats=num_offsets,
            axis=0,
        )
        grouped_args = _args_for_seed_group(
            context["args"],
            [seed for seed in seeds for _ in range(num_offsets)],
        )
        batched_contexts.append(
            {
                "args": grouped_args,
                "seeds": [seed for seed in seeds for _ in range(num_offsets)],
                "source_seed_count": len(seeds),
                "num_offsets": num_offsets,
                "theta_rows": rows,
                "tensors": tiled_tensors,
            }
        )
    return batched_contexts, num_offsets


def _value_at_theta_rows(
    contexts: list[dict[str, Any]],
    theta_rows: tf.Tensor,
    *,
    theta_offset_batch_size: int = 0,
) -> tf.Tensor:
    rows = tf.convert_to_tensor(theta_rows, dtype=p8p.DTYPE)
    if theta_offset_batch_size > 0 and int(rows.shape[0]) > theta_offset_batch_size:
        chunks = []
        for start in range(0, int(rows.shape[0]), int(theta_offset_batch_size)):
            chunks.append(
                _value_at_theta_rows(
                    contexts,
                    rows[start : start + int(theta_offset_batch_size)],
                    theta_offset_batch_size=0,
                )
            )
        return tf.concat(chunks, axis=0)
    batched_contexts, num_offsets = _batched_theta_contexts(contexts, rows)
    weighted_values = []
    total_seeds = 0
    for context in batched_contexts:
        repeated_theta_rows = tf.tile(
            rows,
            [int(context["source_seed_count"]), 1],
        )
        theta_components = tuple(tf.unstack(repeated_theta_rows, axis=1))
        _objective, value = p8p._objective_from_components(
            context["tensors"],
            context["args"],
            theta_components,  # type: ignore[arg-type]
        )
        value_by_seed_offset = tf.reshape(
            value,
            [int(context["source_seed_count"]), num_offsets],
        )
        weighted_values.append(tf.reduce_sum(value_by_seed_offset, axis=0))
        total_seeds += int(context["source_seed_count"])
    return tf.add_n(weighted_values) / tf.cast(total_seeds, p8p.DTYPE)


def _trim_extreme_points(
    xs: tf.Tensor,
    y: tf.Tensor,
    theta_rows: tf.Tensor,
    trim_count: int,
    *,
    trim_mode: str = "offset",
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, dict[str, Any]]:
    x_tensor = tf.convert_to_tensor(xs, dtype=p8p.DTYPE)
    y_tensor = tf.convert_to_tensor(y, dtype=p8p.DTYPE)
    theta_tensor = tf.convert_to_tensor(theta_rows, dtype=p8p.DTYPE)
    point_count = int(x_tensor.shape[0])
    if trim_mode not in ("offset", "value"):
        raise ValueError("trim_mode must be 'offset' or 'value'")
    if trim_count <= 0:
        return (
            x_tensor,
            y_tensor,
            theta_tensor,
            {
                "trimmed": False,
                "trim_mode": trim_mode,
                "trim_extreme_offsets": 0,
                "evaluated_point_count": point_count,
                "fit_point_count": point_count,
                "fit_point_indices": list(range(point_count)),
            },
        )
    if trim_mode == "offset":
        order = tf.argsort(x_tensor, stable=True)
        drop_order = tf.concat(
            [
                order[:trim_count],
                order[point_count - trim_count :],
            ],
            axis=0,
        )
        keep_order = order[trim_count : point_count - trim_count]
        fit_xs = tf.gather(x_tensor, keep_order)
        fit_y = tf.gather(y_tensor, keep_order)
        fit_theta_rows = tf.gather(theta_tensor, keep_order)
        dropped_indices = [int(item) for item in drop_order.numpy().tolist()]
        fit_indices = [int(item) for item in keep_order.numpy().tolist()]
        dropped_x_values = [float(item) for item in tf.gather(x_tensor, drop_order).numpy().tolist()]
        dropped_y_values = [float(item) for item in tf.gather(y_tensor, drop_order).numpy().tolist()]
        return (
            fit_xs,
            fit_y,
            fit_theta_rows,
            {
                "trimmed": True,
                "trim_mode": "offset",
                "trim_extreme_offsets": int(trim_count),
                "evaluated_point_count": point_count,
                "fit_point_count": int(fit_xs.shape[0]),
                "fit_point_indices": fit_indices,
                "dropped_point_indices": dropped_indices,
                "dropped_x_values": dropped_x_values,
                "dropped_objective_values": dropped_y_values,
                "tie_break_rule": "sort by x ascending with stable original-index tie break",
            },
        )

    x_values = [float(item) for item in x_tensor.numpy().tolist()]
    y_values = [float(item) for item in y_tensor.numpy().tolist()]
    indices = list(range(point_count))
    low_order = sorted(indices, key=lambda index: (y_values[index], x_values[index], index))
    low_indices = low_order[:trim_count]
    low_set = set(low_indices)
    high_order = sorted(
        [index for index in indices if index not in low_set],
        key=lambda index: (-y_values[index], x_values[index], index),
    )
    high_indices = high_order[:trim_count]
    dropped_index_set = set(low_indices + high_indices)
    keep_indices = [index for index in indices if index not in dropped_index_set]
    keep_order = tf.constant(keep_indices, dtype=tf.int32)
    fit_xs = tf.gather(x_tensor, keep_order)
    fit_y = tf.gather(y_tensor, keep_order)
    fit_theta_rows = tf.gather(theta_tensor, keep_order)

    def _point_records(point_indices: list[int]) -> list[dict[str, float | int]]:
        return [
            {
                "index": int(index),
                "x": float(x_values[index]),
                "objective_value": float(y_values[index]),
            }
            for index in point_indices
        ]

    dropped_indices = low_indices + high_indices
    return (
        fit_xs,
        fit_y,
        fit_theta_rows,
        {
            "trimmed": True,
            "trim_mode": "value",
            "trim_extreme_offsets": int(trim_count),
            "trim_extreme_values": int(trim_count),
            "evaluated_point_count": point_count,
            "fit_point_count": int(fit_xs.shape[0]),
            "fit_point_indices": keep_indices,
            "fit_x_values": [float(x_values[index]) for index in keep_indices],
            "fit_objective_values": [float(y_values[index]) for index in keep_indices],
            "dropped_point_indices": dropped_indices,
            "dropped_low_value_points": _point_records(low_indices),
            "dropped_high_value_points": _point_records(high_indices),
            "dropped_x_values": [float(x_values[index]) for index in dropped_indices],
            "dropped_objective_values": [float(y_values[index]) for index in dropped_indices],
            "tie_break_rule": (
                "low values: objective ascending, x ascending, original index ascending; "
                "high values: objective descending, x ascending, original index ascending "
                "after excluding low-value drops"
            ),
        },
    )


def _regression_diagnostic_for_direction(
    contexts: list[dict[str, Any]],
    args: argparse.Namespace,
    theta0: tf.Tensor,
    gradient: tf.Tensor,
    direction: tf.Tensor,
    *,
    basis_name: str,
    direction_name: str,
    base_step: float,
) -> dict[str, Any]:
    offsets = tf.constant(args.regression_offsets_values, dtype=p8p.DTYPE)
    xs = offsets * tf.constant(float(base_step), dtype=p8p.DTYPE)
    theta_rows_tensor = theta0[tf.newaxis, :] + xs[:, tf.newaxis] * direction[tf.newaxis, :]
    if args.fd_evaluation_mode == "batched-theta":
        y = _value_at_theta_rows(
            contexts,
            theta_rows_tensor,
            theta_offset_batch_size=int(args.theta_offset_batch_size),
        )
    else:
        values = []
        for theta in tf.unstack(theta_rows_tensor, axis=0):
            values.append(_value_at_theta(contexts, theta))
        y = tf.stack(values)
    fit_xs, fit_y, fit_theta_rows, trim_record = _trim_extreme_points(
        xs,
        y,
        theta_rows_tensor,
        int(args.trim_extreme_offsets),
        trim_mode=args.trim_extreme_mode,
    )
    fit = _linear_regression(fit_xs, fit_y)
    ad_directional = tf.reduce_sum(gradient * direction)
    residual = float(ad_directional.numpy()) - fit["slope"]
    combined_se = fit["slope_standard_error"]
    return {
        "basis_name": basis_name,
        "direction_name": direction_name,
        "direction_original_theta": [float(item) for item in direction.numpy().tolist()],
        "base_step": float(base_step),
        "offsets": [float(item) for item in args.regression_offsets_values],
        "x_values": [float(item) for item in xs.numpy().tolist()],
        "theta_values": [
            [float(item) for item in row]
            for row in theta_rows_tensor.numpy().tolist()
        ],
        "objective_values": [float(item) for item in y.numpy().tolist()],
        "fd_evaluation_mode": args.fd_evaluation_mode,
        "fit_x_values": [float(item) for item in fit_xs.numpy().tolist()],
        "fit_theta_values": [
            [float(item) for item in row]
            for row in fit_theta_rows.numpy().tolist()
        ],
        "fit_objective_values": [float(item) for item in fit_y.numpy().tolist()],
        "trim_extreme_points": trim_record,
        "ad_directional_derivative": float(ad_directional.numpy()),
        "regression_slope": fit["slope"],
        "ad_minus_regression_slope": residual,
        "regression_slope_standard_error": fit["slope_standard_error"],
        "ad_minus_slope_over_slope_se": (
            residual / combined_se if combined_se > 0.0 else None
        ),
        "regression_intercept": fit["intercept"],
        "regression_residual_sse": fit["residual_sse"],
        "regression_max_abs_residual": fit["max_abs_residual"],
        "regression_r_squared": fit["r_squared"],
    }


def _base_steps_for_direction(
    args: argparse.Namespace,
    gradient: tf.Tensor,
    direction: tf.Tensor,
) -> tuple[list[float], dict[str, Any]]:
    if args.base_step_mode == "fixed":
        return (
            [float(item) for item in args.base_step_ladder_values],
            {
                "mode": "fixed",
                "absolute_base_steps": [float(item) for item in args.base_step_ladder_values],
            },
        )

    ad_directional = tf.reduce_sum(gradient * direction)
    ad_abs = float(tf.abs(ad_directional).numpy())
    max_offset = max(abs(float(item)) for item in args.regression_offsets_values)
    unconstrained = args.target_objective_delta / max(
        ad_abs * max_offset,
        1.0e-12,
    )
    selected = min(
        max(unconstrained, float(args.min_adaptive_base_step)),
        float(args.max_adaptive_base_step),
    )
    steps = [selected * float(factor) for factor in args.adaptive_step_factor_values]
    steps = [
        min(
            max(float(step), float(args.min_adaptive_base_step)),
            float(args.max_adaptive_base_step),
        )
        for step in steps
    ]
    deduped_steps = []
    for step in steps:
        if not any(math.isclose(step, previous, rel_tol=1.0e-12, abs_tol=0.0) for previous in deduped_steps):
            deduped_steps.append(step)
    return (
        deduped_steps,
        {
            "mode": "ad-signal",
            "ad_directional_derivative": float(ad_directional.numpy()),
            "target_objective_delta_at_max_offset": float(args.target_objective_delta),
            "max_abs_offset": float(max_offset),
            "unconstrained_base_step": float(unconstrained),
            "selected_base_step": float(selected),
            "adaptive_step_factors": [float(item) for item in args.adaptive_step_factor_values],
            "min_adaptive_base_step": float(args.min_adaptive_base_step),
            "max_adaptive_base_step": float(args.max_adaptive_base_step),
            "absolute_base_steps": [float(item) for item in deduped_steps],
        },
    )


def _plateau_summary(direction_results: list[dict[str, Any]]) -> dict[str, Any]:
    if len(direction_results) < 2:
        return {
            "checked": False,
            "reason": "at least two base steps are required",
        }
    ordered = sorted(direction_results, key=lambda item: item["base_step"], reverse=True)
    slopes = [float(item["regression_slope"]) for item in ordered]
    base_steps = [float(item["base_step"]) for item in ordered]
    adjacent = []
    for left, right in zip(ordered, ordered[1:], strict=False):
        delta = float(right["regression_slope"]) - float(left["regression_slope"])
        scale = max(1.0, abs(float(right["regression_slope"])), abs(float(left["regression_slope"])))
        adjacent.append(
            {
                "from_base_step": float(left["base_step"]),
                "to_base_step": float(right["base_step"]),
                "slope_delta": delta,
                "relative_delta": abs(delta) / scale,
            }
        )
    return {
        "checked": True,
        "base_steps": base_steps,
        "slopes": slopes,
        "adjacent_slope_deltas": adjacent,
        "smallest_window_slope": slopes[-1],
        "largest_window_slope": slopes[0],
        "range": max(slopes) - min(slopes),
    }


def _validate_device(tensors: tuple[tf.Tensor, ...], expect_device_kind: str) -> list[str]:
    devices = [tensor.device for tensor in tensors]
    if expect_device_kind == "gpu":
        if not all("GPU" in device.upper() for device in devices):
            raise RuntimeError(f"expected GPU tensors, got {devices}")
    elif expect_device_kind == "cpu":
        if not all("CPU" in device.upper() for device in devices):
            raise RuntimeError(f"expected CPU tensors, got {devices}")
    return devices


def _write_progress(
    args: argparse.Namespace,
    *,
    start: float,
    stage: str,
    completed: list[dict[str, Any]],
    current: dict[str, Any] | None = None,
) -> None:
    if not args.progress_output:
        return
    progress_path = Path(args.progress_output)
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp_utc": _dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
        "phase": args.phase_label,
        "stage": stage,
        "elapsed_seconds": time.perf_counter() - start,
        "output": args.output,
        "direction_filter": list(args.direction_filter_values),
        "completed": completed,
        "current": current,
    }
    progress_path.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _append_memory_sample(
    args: argparse.Namespace,
    *,
    start: float,
    samples: list[dict[str, Any]],
    lock: threading.Lock,
    stage: str,
) -> None:
    if not args.memory_sample_output:
        return
    sample = {
        "timestamp_utc": _dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - start,
        "stage": stage,
        "gpu_memory_info": p8p._gpu_memory_info(),
    }
    with lock:
        samples.append(sample)
        record = {
            "schema_version": "filter_bench.gpu_memory_sampler.v1",
            "phase": args.phase_label,
            "output": args.output,
            "sample_count": len(samples),
            "latest": sample,
            "samples": samples,
        }
        memory_path = Path(args.memory_sample_output)
        memory_path.parent.mkdir(parents=True, exist_ok=True)
        memory_path.write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def _memory_sampler_worker(
    args: argparse.Namespace,
    *,
    start: float,
    stop_event: threading.Event,
    samples: list[dict[str, Any]],
    lock: threading.Lock,
) -> None:
    while not stop_event.wait(float(args.memory_sample_interval_seconds)):
        _append_memory_sample(
            args,
            start=start,
            samples=samples,
            lock=lock,
            stage="running",
        )


def _result_contract_metadata(
    args: argparse.Namespace,
    *,
    objective: tf.Tensor,
    gradient: tf.Tensor,
    per_seed_gradient: tf.Tensor,
    gradients_connected: bool,
) -> dict[str, Any]:
    gradient_values = p8p._to_float_list(gradient)
    mc_noise = p8p._mc_noise_summary(per_seed_gradient)
    objective_finite = math.isfinite(float(objective.numpy()))
    gradient_finite = all(math.isfinite(item) for item in gradient_values)
    mcse_finite = all(
        math.isfinite(float(record["standard_error_of_batch_mean"]))
        for record in mc_noise.values()
    )
    primary_pass = bool(
        objective_finite
        and gradient_finite
        and mcse_finite
        and bool(gradients_connected)
    )
    ad_evaluation_mode = getattr(args, "ad_evaluation_mode", "reverse-gradient")
    gradient_path_label = (
        "connected manual score route"
        if ad_evaluation_mode == "manual-reverse"
        else "connected diagnostic autodiff path"
    )
    return {
        "status": "pass" if primary_pass else "blocked_or_failed",
        "batch_seeds": [int(seed) for seed in args.batch_seeds],
        "objective_finite": objective_finite,
        "gradient_values": gradient_values,
        "gradient_finite": gradient_finite,
        "monte_carlo_gradient_noise": mc_noise,
        "monte_carlo_gradient_noise_mcse_finite": mcse_finite,
        "gradients_connected": bool(gradients_connected),
        "primary_pass": primary_pass,
        "primary_pass_criterion": (
            "finite objective, finite gradient components, finite seed-gradient "
            f"MCSE values, and {gradient_path_label}; not FD agreement or "
            "scientific validity"
        ),
    }


def _transport_metadata(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "value_core_mode": "streaming",
        "transport_plan_mode": args.transport_plan_mode,
        "transport_ad_mode": args.transport_ad_mode,
        "gradient_mode": args.transport_gradient_mode,
        "row_chunk_size": args.row_chunk_size,
        "col_chunk_size": args.col_chunk_size,
        "particle_chunk_size": args.particle_chunk_size,
        "sinkhorn_iterations": args.sinkhorn_iterations,
        "sinkhorn_epsilon": args.sinkhorn_epsilon,
        "annealed_scaling": args.annealed_scaling,
        "annealed_convergence_threshold": args.annealed_convergence_threshold,
        "dense_transport_matrix_materialized": False,
    }


def main() -> None:
    raise RuntimeError("ARCHIVAL_WRONG_TRANSPORT_CHUNK_POLICY: this route is preserved only as provenance and cannot emit new evidence")
    args = _parse_args()
    precision = p8p._configure_precision(args)
    physical_gpus, logical_gpus = p8p._configure_gpus()
    contexts, sir_semantics = _build_microbatch_contexts(args)
    theta0 = tf.constant(args.theta_values, dtype=p8p.DTYPE)
    start = time.perf_counter()
    memory_samples: list[dict[str, Any]] = []
    memory_lock = threading.Lock()
    memory_stop = threading.Event()
    memory_thread = None
    if args.memory_sample_output:
        _append_memory_sample(
            args,
            start=start,
            samples=memory_samples,
            lock=memory_lock,
            stage="start",
        )
        memory_thread = threading.Thread(
            target=_memory_sampler_worker,
            kwargs={
                "args": args,
                "start": start,
                "stop_event": memory_stop,
                "samples": memory_samples,
                "lock": memory_lock,
            },
            daemon=True,
        )
        memory_thread.start()
    memory_before = p8p._gpu_memory_info()
    try:
        with tf.device(args.device):
            if args.ad_evaluation_mode == "forward-jvp":
                gradient_diag = _forward_jvp_diagnostic_for_contexts(contexts, args.theta_values)
            elif args.ad_evaluation_mode == "manual-reverse":
                gradient_diag = _manual_gradient_diagnostic_for_contexts(
                    contexts,
                    args.theta_values,
                    compiler=args.manual_reverse_compiler,
                    warmups=args.manual_reverse_warmups,
                    repeats=args.manual_reverse_repeats,
                    return_score_decomposition=args.manual_score_decomposition,
                )
            else:
                gradient_diag = _gradient_diagnostic_for_contexts(contexts, args.theta_values)
            objective = gradient_diag["objective"]
            gradient = gradient_diag["gradient_tensor"]
            per_seed_gradient = gradient_diag["per_seed_gradient"]
            geometry = p8p._gradient_geometry_summary(per_seed_gradient)
            seed_covariance = tf.constant(geometry["seed_gradient_covariance"], dtype=p8p.DTYPE)
            completed_progress: list[dict[str, Any]] = []
            _write_progress(
                args,
                start=start,
                stage="ad_complete",
                completed=completed_progress,
                current=None,
            )
            basis_results = []
            if args.fd_mode == "enabled":
                for basis in _basis_matrices(seed_covariance, args.basis_set):
                    matrix = tf.convert_to_tensor(basis["matrix_columns"], dtype=p8p.DTYPE)
                    direction_results = []
                    for index, direction_name in enumerate(basis["direction_names"]):
                        if (
                            args.direction_filter_values
                            and direction_name not in args.direction_filter_values
                        ):
                            continue
                        window_results = []
                        direction = matrix[:, index]
                        base_steps, step_selection = _base_steps_for_direction(
                            args,
                            gradient,
                            direction,
                        )
                        for base_step in base_steps:
                            _write_progress(
                                args,
                                start=start,
                                stage="fd_window_started",
                                completed=completed_progress,
                                current={
                                    "basis_name": basis["basis_name"],
                                    "direction_name": direction_name,
                                    "base_step": float(base_step),
                                },
                            )
                            window_results.append(
                                _regression_diagnostic_for_direction(
                                    contexts,
                                    args,
                                    theta0,
                                    gradient,
                                    direction,
                                    basis_name=basis["basis_name"],
                                    direction_name=direction_name,
                                    base_step=base_step,
                                )
                            )
                            completed_progress.append(
                                {
                                    "basis_name": basis["basis_name"],
                                    "direction_name": direction_name,
                                    "base_step": float(base_step),
                                    "regression_slope": float(
                                        window_results[-1]["regression_slope"]
                                    ),
                                    "ad_minus_regression_slope": float(
                                        window_results[-1]["ad_minus_regression_slope"]
                                    ),
                                }
                            )
                            _write_progress(
                                args,
                                start=start,
                                stage="fd_window_complete",
                                completed=completed_progress,
                                current=None,
                            )
                        direction_results.append(
                            {
                                "direction_name": direction_name,
                                "direction_original_theta": [
                                    float(item) for item in matrix[:, index].numpy().tolist()
                                ],
                                "base_step_selection": step_selection,
                                "ad_directional_derivative": window_results[0][
                                    "ad_directional_derivative"
                                ],
                                "window_results": window_results,
                                "plateau_summary": _plateau_summary(window_results),
                            }
                        )
                    basis_record = dict(basis)
                    basis_record["matrix_columns"] = p8p._to_float_matrix(matrix)
                    basis_record["direction_results"] = direction_results
                    basis_results.append(basis_record)
                if args.direction_filter_values and not any(
                    basis["direction_results"] for basis in basis_results
                ):
                    raise ValueError(
                        "direction-filter did not match any direction in the selected basis set: "
                        + ",".join(args.direction_filter_values)
                    )
    finally:
        memory_stop.set()
        if memory_thread is not None:
            memory_thread.join(timeout=5.0)
        if args.memory_sample_output:
            _append_memory_sample(
                args,
                start=start,
                samples=memory_samples,
                lock=memory_lock,
                stage="final",
            )
    elapsed = time.perf_counter() - start
    memory_after = p8p._gpu_memory_info()
    output_devices = _validate_device((objective, gradient), args.expect_device_kind)
    contract_metadata = _result_contract_metadata(
        args,
        objective=objective,
        gradient=gradient,
        per_seed_gradient=per_seed_gradient,
        gradients_connected=bool(gradient_diag["gradients_connected"]),
    )
    manual_score_decomposition = gradient_diag.get("manual_score_decomposition")
    if manual_score_decomposition is None:
        manual_score_decomposition_record = {
            "checked": False,
            "reason": "manual-score-decomposition flag was not enabled",
        }
    else:
        component_names = list(manual_score_decomposition["component_names"])
        component_tensor = tf.convert_to_tensor(
            manual_score_decomposition["components"],
            dtype=p8p.DTYPE,
        )
        component_mean = tf.convert_to_tensor(
            manual_score_decomposition["component_mean"],
            dtype=p8p.DTYPE,
        )
        component_sum = tf.convert_to_tensor(
            manual_score_decomposition["component_sum"],
            dtype=p8p.DTYPE,
        )
        reconstruction_delta = tf.convert_to_tensor(
            manual_score_decomposition["component_sum_reconstruction_max_abs_delta"],
            dtype=p8p.DTYPE,
        )
        manual_score_decomposition_record = {
            "checked": True,
            "component_names": component_names,
            "component_means": {
                name: dict(
                    zip(
                        p8p.PARAMETER_NAMES,
                        [float(item) for item in component_mean[index].numpy().tolist()],
                        strict=True,
                    )
                )
                for index, name in enumerate(component_names)
            },
            "component_sums_per_seed": p8p._to_float_matrix(component_sum),
            "per_seed_components": {
                name: p8p._to_float_matrix(component_tensor[index])
                for index, name in enumerate(component_names)
            },
            "component_sum_reconstruction_max_abs_delta": float(
                reconstruction_delta.numpy()
            ),
            "component_sum_reconstructs_per_seed_gradient": bool(
                reconstruction_delta.numpy() <= 1.0e-4
            ),
            "parameter_order": list(p8p.PARAMETER_NAMES),
        }
    result = {
        "schema_version": "filter_bench.p8p_regression_fd_reparameterization.v1",
        "timestamp_utc": _dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
        "host": platform.node(),
        "python_version": platform.python_version(),
        "tensorflow_version": tf.__version__,
        "git_commit": p8p._git_commit(),
        "phase": args.phase_label,
        "status": contract_metadata["status"],
        "elapsed_seconds": elapsed,
        "physical_gpus": physical_gpus,
        "logical_gpus": logical_gpus,
        "gpu_memory_info_before": memory_before,
        "gpu_memory_info_after": memory_after,
        "gpu_memory_samples": list(memory_samples),
        "memory_sample_output": args.memory_sample_output or None,
        "device": args.device,
        "device_scope": args.device_scope,
        "expect_device_kind": args.expect_device_kind,
        "output_devices": output_devices,
        "precision": precision,
        "shape": {
            "batch_size": len(args.batch_seeds),
            "time_steps": args.time_steps,
            "num_particles": args.num_particles,
            "state_dim": 18,
            "obs_dim": 9,
            "parameter_dim": 3,
            "seed_microbatch_size": int(args.seed_microbatch_size),
            "seed_microbatch_count": len(contexts),
            "ad_evaluation_mode": args.ad_evaluation_mode,
        },
        "compiler": gradient_diag.get(
            "compiler",
            {
                "mode": "diagnostic-autodiff-default",
                "tf_function": False,
                "jit_compile": False,
                "compiled_unit": None,
            },
        ),
        "sir_semantics": sir_semantics,
        "batch_seeds": contract_metadata["batch_seeds"],
        "theta": dict(zip(p8p.PARAMETER_NAMES, [float(x) for x in args.theta_values], strict=True)),
        "parameter_order": list(p8p.PARAMETER_NAMES),
        "objective": float(objective.numpy()),
        "objective_finite": contract_metadata["objective_finite"],
        "gradient": dict(zip(p8p.PARAMETER_NAMES, contract_metadata["gradient_values"], strict=True)),
        "gradient_values": contract_metadata["gradient_values"],
        "gradient_finite": contract_metadata["gradient_finite"],
        "per_seed_gradient_contributions": p8p._to_float_matrix(per_seed_gradient),
        "manual_score_decomposition": manual_score_decomposition_record,
        "monte_carlo_gradient_noise": contract_metadata["monte_carlo_gradient_noise"],
        "monte_carlo_gradient_noise_mcse_finite": contract_metadata[
            "monte_carlo_gradient_noise_mcse_finite"
        ],
        "gradients_connected": contract_metadata["gradients_connected"],
        "seed_gradient_geometry": geometry,
        "regression_fd": {
            "method": "ordinary least squares fit f(theta0 + x direction) = intercept + slope * x",
            "fd_mode": args.fd_mode,
            "ad_evaluation_mode": args.ad_evaluation_mode,
            "manual_score_decomposition": bool(args.manual_score_decomposition),
            "base_step": float(args.base_step),
            "base_step_ladder": [float(item) for item in args.base_step_ladder_values],
            "base_step_mode": args.base_step_mode,
            "target_objective_delta": float(args.target_objective_delta),
            "adaptive_step_factors": [float(item) for item in args.adaptive_step_factor_values],
            "min_adaptive_base_step": float(args.min_adaptive_base_step),
            "max_adaptive_base_step": float(args.max_adaptive_base_step),
            "offsets": [float(item) for item in args.regression_offsets_values],
            "trim_extreme_offsets": int(args.trim_extreme_offsets),
            "trim_extreme_mode": args.trim_extreme_mode,
            "basis_set": args.basis_set,
            "basis_results": basis_results,
        },
        "transport_policy": args.transport_policy,
        "transport": _transport_metadata(args),
        "primary_pass": contract_metadata["primary_pass"],
        "primary_pass_criterion": contract_metadata["primary_pass_criterion"],
        "nonclaims": list(p8p.NONCLAIMS)
        + [
            "regression FD diagnostic only",
            "not a global reparameterization proof",
            "not HMC geometry adequacy evidence",
        ],
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
