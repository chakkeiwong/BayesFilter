"""Compact staged low-rank LEDH gradient-path diagnostic.

This P02B-R2 diagnostic reuses the P02B route-internal LGSSM path but replaces
the blocked all-checkpoint Jacobian readout with compact staged whole-sum
gradients.  It is a localization artifact only: no repair, posterior
correctness, HMC readiness, threshold calibration, P03 handoff, or
default-readiness claim is made here.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_PRE = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
_PRE.add_argument("--device-scope", choices=("cpu", "visible"), default="visible")
_PRE.add_argument("--cuda-visible-devices", default=None)
_PRE_ARGS, _ = _PRE.parse_known_args()
if _PRE_ARGS.device_scope == "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
elif _PRE_ARGS.cuda_visible_devices is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = _PRE_ARGS.cuda_visible_devices
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tensorflow as tf  # noqa: E402
import tensorflow_probability as tfp  # noqa: E402

from docs.benchmarks import benchmark_low_rank_ledh_lgssm_kalman_gate as lgssm_gate  # noqa: E402
from docs.benchmarks import benchmark_low_rank_ledh_posterior_gradient_calibration as calibration  # noqa: E402
from docs.benchmarks import benchmark_low_rank_ledh_route_internal_gradient_connectivity as p02b  # noqa: E402


PLAN_PATH = (
    "docs/plans/bayesfilter-dpf-ledh-pfpf-ot-low-rank-residual-posterior-gradient-"
    "calibration-master-program-2026-06-24.md"
)
SUBPLAN_PATH = (
    "docs/plans/bayesfilter-dpf-ledh-pfpf-ot-low-rank-residual-posterior-gradient-"
    "calibration-p02b-r3-xla-graph-reduction-plan-2026-06-26.md"
)
P02A_RESULT_PATH = (
    "docs/plans/bayesfilter-dpf-ledh-pfpf-ot-low-rank-residual-posterior-gradient-"
    "calibration-p02a-gradient-repair-diagnostic-result-2026-06-25.md"
)
P02B_RESULT_PATH = (
    "docs/plans/bayesfilter-dpf-ledh-pfpf-ot-low-rank-residual-posterior-gradient-"
    "calibration-p02b-route-internal-gradient-connectivity-result-2026-06-25.md"
)
P02B_R2_RESULT_PATH = (
    "docs/plans/bayesfilter-dpf-ledh-pfpf-ot-low-rank-residual-posterior-gradient-"
    "calibration-p02b-r2-staged-gradient-path-harness-reduction-result-2026-06-26.md"
)
GPU_TRUST_BASIS = "owner_designated_managed_session_visible_gpu_trusted"
DEFAULT_SEED_PROBES = ((91003, "center"), (91002, "qr_plus"))
NONCLAIMS = (
    "P02B-R2 compact staged gradient-path localization diagnostic only",
    "no low-rank solver repair claim",
    "no residual-threshold calibration claim",
    "no P03 handoff claim",
    "no posterior correctness claim",
    "no HMC readiness claim",
    "no default/package/public API readiness claim",
    "no statistical superiority claim",
    "no scientific validity claim",
)


@dataclass(frozen=True)
class StageSpec:
    name: str
    stage: str
    expected_connected: bool
    expected_note: str
    checkpoint_kind: str = "route_captured"


STAGE_SPECS: tuple[StageSpec, ...] = (
    StageSpec("scaled_Q", "C_pre_low_rank", True, "direct theta -> exp(q_scale) * Q path"),
    StageSpec("scaled_R", "C_pre_low_rank", True, "direct theta -> exp(r_scale) * R path"),
    StageSpec("pre_flow_t0", "C_pre_low_rank", False, "fixed initial particles and transition matrix at t0"),
    StageSpec("post_flow_t0", "C_pre_low_rank", True, "LEDH flow should depend on scaled Q/R"),
    StageSpec("pre_flow_log_density_t0", "C_pre_low_rank", True, "LEDH proposal density should depend on scaled Q/R"),
    StageSpec("forward_log_det_t0", "C_pre_low_rank", True, "LEDH flow Jacobian should depend on scaled Q/R"),
    StageSpec("transition_log_density_t0", "C_pre_low_rank", True, "transition density uses scaled Q"),
    StageSpec("observation_log_density_t0", "C_pre_low_rank", True, "observation density uses scaled R"),
    StageSpec("corrected_log_weights_t0", "C_pre_low_rank", True, "corrected weights combine connected densities"),
    StageSpec("normalized_weights_t0", "C_pre_low_rank", True, "normalized weights inherit corrected weights"),
    StageSpec("incremental_log_likelihood_t0", "C_pre_low_rank", True, "incremental likelihood inherits corrected weights"),
    StageSpec("normalized_log_weights_t0", "C_pre_low_rank", True, "normalized log weights inherit corrected weights"),
    StageSpec("scaled_x_t0", "D_low_rank_t0", True, "diagnostic reconstruction from post_flow", "diagnostic_reconstruction"),
    StageSpec("eps_q_t0", "D_low_rank_t0", True, "diagnostic reconstruction from scaled_x", "diagnostic_reconstruction"),
    StageSpec("eps_r_t0", "D_low_rank_t0", True, "diagnostic reconstruction from reflected scaled_x", "diagnostic_reconstruction"),
    StageSpec("eps_g_t0", "D_low_rank_t0", False, "fixed uniform rank weights independent of theta", "diagnostic_reconstruction"),
    StageSpec("q_factor_t0", "D_low_rank_t0", True, "Dykstra q factor should inherit eps_q/log-weight path"),
    StageSpec("r_factor_t0", "D_low_rank_t0", True, "Dykstra r factor should inherit eps_r/log-weight path"),
    StageSpec("g_weights_t0", "D_low_rank_t0", True, "Dykstra shared rank weights should inherit factor path"),
    StageSpec("transported_particles_t0", "D_low_rank_t0", True, "transported particles should inherit factor and post_flow path"),
    StageSpec("resampled_log_weights_t0", "D_low_rank_t0", False, "solver intentionally resets log weights to uniform"),
    StageSpec("next_pre_flow_t1", "E_cross_time", True, "next pre-flow should inherit transported particles"),
    StageSpec("next_corrected_log_weights_t1", "E_cross_time", True, "next corrected weights should inherit carried particles and scaled Q/R"),
    StageSpec("next_incremental_log_likelihood_t1", "E_cross_time", True, "next incremental likelihood should inherit corrected weights"),
    StageSpec("final_particles", "E_cross_time", True, "final particles should inherit repeated transport path"),
    StageSpec("final_log_likelihood", "E_cross_time", True, "final likelihood should inherit incremental likelihoods"),
)
STAGE_SPEC_BY_NAME = {spec.name: spec for spec in STAGE_SPECS}
STAGE_GROUPS = ("C_pre_low_rank", "D_low_rank_t0", "E_cross_time")
STAGE_NAMES_BY_GROUP = {
    group: tuple(spec.name for spec in STAGE_SPECS if spec.stage == group)
    for group in STAGE_GROUPS
}
STAGE_NAMES = tuple(spec.name for spec in STAGE_SPECS)
COMPACT_VALUE_PREVIEW_COUNT = 6
COMPACT_SHAPE_RANK = 3
COMPACT_STAGE_OUTPUT_COUNT = 11
STAGE_ROUTE_SUMMARY_OUTPUT_COUNT = 27


def _parse_args() -> argparse.Namespace:
    return _parse_args_impl(None)


def _parse_args_from_list_for_test(argv: list[str]) -> argparse.Namespace:
    return _parse_args_impl(argv)


def _parse_args_impl(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--case-id", default="lgssm_small_exact_ref")
    parser.add_argument("--seed-probes", default=None)
    parser.add_argument("--num-particles", type=int, default=1024)
    parser.add_argument("--time-steps", type=int, default=12)
    parser.add_argument("--state-dim", type=int, default=None)
    parser.add_argument("--obs-dim", type=int, default=None)
    parser.add_argument("--low-rank-rank", type=int, default=16)
    parser.add_argument("--low-rank-assignment-epsilon", type=float, default=0.25)
    parser.add_argument("--low-rank-alpha", type=float, default=1.0e-8)
    parser.add_argument("--low-rank-max-projection-iterations", type=int, default=120)
    parser.add_argument("--low-rank-convergence-threshold", type=float, default=1.0e-6)
    parser.add_argument("--low-rank-denominator-floor", type=float, default=1.0e-30)
    parser.add_argument("--sinkhorn-iterations", type=int, default=10)
    parser.add_argument("--sinkhorn-epsilon", type=float, default=1.0)
    parser.add_argument("--annealed-scaling", type=float, default=0.9)
    parser.add_argument("--annealed-convergence-threshold", type=float, default=1.0e-3)
    parser.add_argument("--row-chunk-size", type=int, default=128)
    parser.add_argument("--col-chunk-size", type=int, default=128)
    parser.add_argument("--particle-chunk-size", type=int, default=64)
    parser.add_argument("--theta-probe-radius", type=float, default=0.05)
    parser.add_argument("--theta-prior-scale", type=float, default=0.50)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--jit-compile", dest="jit_compile", action="store_true", default=True)
    parser.add_argument("--no-jit-compile", dest="jit_compile", action="store_false")
    parser.add_argument("--tf32-mode", choices=("default", "enabled", "disabled"), default="enabled")
    parser.add_argument("--device", default="/GPU:0")
    parser.add_argument("--device-scope", choices=("cpu", "visible"), default=_PRE_ARGS.device_scope)
    parser.add_argument("--cuda-visible-devices", default=_PRE_ARGS.cuda_visible_devices)
    parser.add_argument("--expect-device-kind", choices=("any", "cpu", "gpu"), default="gpu")
    parser.add_argument(
        "--readout-mode",
        choices=("staged-only", "primary-and-staged"),
        default="staged-only",
        help="staged-only compiles one route readout per seed; primary-and-staged also runs the heavier A/B tape readout",
    )
    parser.add_argument("--phase-id", default="LOW_RANK_STAGED_GRADIENT_PATH_P02B_R3_XLA_GRAPH_REDUCTION")
    parser.add_argument("--output", required=True)
    parser.add_argument("--markdown-output", default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    _validate_args(args)
    return args


def _validate_args(args: argparse.Namespace) -> None:
    if args.case_id not in lgssm_gate.PINNED_CASES:
        raise ValueError(f"unknown case id: {args.case_id}")
    if args.num_particles <= 1:
        raise ValueError("num_particles must be greater than one")
    for name in ("time_steps", "low_rank_rank", "low_rank_max_projection_iterations"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if args.time_steps < 2:
        raise ValueError("time_steps must be at least 2 for cross-time H5 checkpoints")
    for name in ("state_dim", "obs_dim"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            raise ValueError(f"{name} must be positive")
    if args.low_rank_rank > args.num_particles:
        raise ValueError("low_rank_rank must be <= num_particles")
    if args.low_rank_assignment_epsilon <= 0.0:
        raise ValueError("low_rank_assignment_epsilon must be positive")
    if args.low_rank_alpha <= 0.0:
        raise ValueError("low_rank_alpha must be positive")
    if args.low_rank_alpha * args.low_rank_rank >= 1.0:
        raise ValueError("low_rank_alpha must be smaller than 1/low_rank_rank")
    if args.low_rank_convergence_threshold <= 0.0:
        raise ValueError("low_rank_convergence_threshold must be positive")
    if args.low_rank_denominator_floor <= 0.0:
        raise ValueError("low_rank_denominator_floor must be positive")
    for name in ("row_chunk_size", "col_chunk_size", "particle_chunk_size"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if args.theta_probe_radius <= 0.0:
        raise ValueError("theta_probe_radius must be positive")
    if args.theta_prior_scale <= 0.0:
        raise ValueError("theta_prior_scale must be positive")
    if args.device_scope == "cpu" and args.expect_device_kind == "gpu":
        raise ValueError("cpu device scope cannot expect gpu outputs")


def _seed_probe_pairs(args: argparse.Namespace) -> list[tuple[int, str]]:
    if args.seed_probes is None:
        return list(DEFAULT_SEED_PROBES)
    pairs = []
    for item in args.seed_probes.split(","):
        item = item.strip()
        if not item:
            continue
        seed_text, label = item.split(":", 1)
        pairs.append((int(seed_text), label.strip()))
    if not pairs:
        raise ValueError("expected at least one seed:probe entry")
    return pairs


def _configure(args: argparse.Namespace) -> tuple[tf.DType, dict[str, Any]]:
    precision = lgssm_gate._configure_precision(args)
    gpu_config = lgssm_gate._configure_gpus()
    return calibration._dtype(args), {**precision, **gpu_config}


def _probe_spec_by_label(args: argparse.Namespace, dtype: tf.DType) -> dict[str, dict[str, Any]]:
    return {spec["label"]: spec for spec in calibration._probe_specs(args, dtype)}


def _git_output(command: list[str]) -> str:
    try:
        return subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _to_json_scalar(value: tf.Tensor) -> float:
    return float(tf.reshape(tf.cast(value, tf.float64), [-1])[0].numpy())


def _to_json_int(value: tf.Tensor) -> int:
    return int(tf.reshape(tf.cast(value, tf.int64), [-1])[0].numpy())


def _to_json_bool(value: tf.Tensor | bool) -> bool:
    if isinstance(value, bool):
        return value
    return bool(tf.reshape(tf.cast(value, tf.bool), [-1])[0].numpy())


def _to_json_vector(value: tf.Tensor) -> list[float]:
    flat = tf.reshape(tf.cast(value, tf.float64), [-1])
    return [float(item) for item in flat.numpy().tolist()]


def _tensor_value_summary(tensor: tf.Tensor) -> dict[str, Any]:
    flat = tf.reshape(tf.cast(tensor, tf.float64), [-1])
    finite = tf.math.is_finite(flat)
    finite_count = int(tf.reduce_sum(tf.cast(finite, tf.int32)).numpy())
    count = int(tf.size(flat).numpy())
    if finite_count:
        values = tf.boolean_mask(flat, finite)
        min_value = float(tf.reduce_min(values).numpy())
        max_value = float(tf.reduce_max(values).numpy())
        mean_value = float(tf.reduce_mean(values).numpy())
    else:
        min_value = None
        max_value = None
        mean_value = None
    return {
        "shape": tensor.shape.as_list(),
        "count": count,
        "finite_count": finite_count,
        "nonfinite_count": count - finite_count,
        "min": min_value,
        "max": max_value,
        "mean": mean_value,
        "preview": [float(item) for item in flat[: min(6, count)].numpy().tolist()],
    }


def _compact_value_parts(tensor: tf.Tensor, dtype: tf.DType) -> tuple[tf.Tensor, ...]:
    tensor = tf.cast(tensor, dtype)
    flat = tf.reshape(tensor, [-1])
    count = tf.size(flat)
    finite = tf.math.is_finite(flat)
    finite_count = tf.reduce_sum(tf.cast(finite, tf.int32))
    finite_for_min = tf.where(finite, flat, tf.fill(tf.shape(flat), tf.constant(float("inf"), dtype=dtype)))
    finite_for_max = tf.where(finite, flat, tf.fill(tf.shape(flat), tf.constant(float("-inf"), dtype=dtype)))
    finite_or_zero = tf.where(finite, flat, tf.zeros_like(flat))
    value_min = tf.cond(
        finite_count > 0,
        lambda: tf.reduce_min(finite_for_min),
        lambda: tf.constant(float("nan"), dtype=dtype),
    )
    value_max = tf.cond(
        finite_count > 0,
        lambda: tf.reduce_max(finite_for_max),
        lambda: tf.constant(float("nan"), dtype=dtype),
    )
    value_mean = tf.cond(
        finite_count > 0,
        lambda: tf.reduce_sum(finite_or_zero) / tf.cast(finite_count, dtype),
        lambda: tf.constant(float("nan"), dtype=dtype),
    )

    preview_indices = tf.range(COMPACT_VALUE_PREVIEW_COUNT, dtype=tf.int32)
    preview = tf.cond(
        count > 0,
        lambda: tf.gather(flat, tf.minimum(preview_indices, count - 1)),
        lambda: tf.fill([COMPACT_VALUE_PREVIEW_COUNT], tf.constant(float("nan"), dtype=dtype)),
    )
    preview = tf.where(
        preview_indices < count,
        preview,
        tf.fill([COMPACT_VALUE_PREVIEW_COUNT], tf.constant(float("nan"), dtype=dtype)),
    )

    rank = tf.rank(tensor)
    shape = tf.shape(tensor, out_type=tf.int32)
    pad_count = tf.maximum(tf.constant(0, dtype=tf.int32), tf.constant(COMPACT_SHAPE_RANK, dtype=tf.int32) - rank)
    shape_padded = tf.concat([shape, tf.zeros([pad_count], dtype=tf.int32)], axis=0)[:COMPACT_SHAPE_RANK]
    return shape_padded, rank, count, finite_count, value_min, value_max, value_mean, preview


def _compact_value_summary_from_parts(parts: tuple[tf.Tensor, ...]) -> dict[str, Any]:
    shape_padded, rank, count, finite_count, value_min, value_max, value_mean, preview = parts
    rank_int = _to_json_int(rank)
    count_int = _to_json_int(count)
    finite_count_int = _to_json_int(finite_count)
    shape_values = [int(item) for item in tf.reshape(shape_padded, [-1])[:rank_int].numpy().tolist()]
    preview_count = min(COMPACT_VALUE_PREVIEW_COUNT, count_int)
    preview_values = tf.reshape(tf.cast(preview, tf.float64), [-1])[:preview_count]
    if finite_count_int:
        min_value = _to_json_scalar(value_min)
        max_value = _to_json_scalar(value_max)
        mean_value = _to_json_scalar(value_mean)
    else:
        min_value = None
        max_value = None
        mean_value = None
    return {
        "shape": shape_values,
        "count": count_int,
        "finite_count": finite_count_int,
        "nonfinite_count": count_int - finite_count_int,
        "min": min_value,
        "max": max_value,
        "mean": mean_value,
        "preview": [float(item) for item in preview_values.numpy().tolist()],
        "readout_mode": "compiled_compact_value_summary",
    }


def _gradient_summary_from_parts(connected: tf.Tensor, finite: tf.Tensor, gradient: tf.Tensor) -> dict[str, Any]:
    finite_components = tf.math.is_finite(gradient)
    return {
        "connected": _to_json_bool(connected),
        "finite": _to_json_bool(finite),
        "finite_components": [bool(item) for item in tf.reshape(finite_components, [-1]).numpy().tolist()],
        "gradient": _to_json_vector(gradient),
        "norm": float(tf.linalg.norm(tf.cast(gradient, tf.float64)).numpy()),
    }


def _gradient_parts(
    tape: tf.GradientTape,
    target: tf.Tensor,
    theta: tf.Tensor,
    dtype: tf.DType,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    gradient = tape.gradient(tf.cast(target, dtype), theta)
    connected = tf.constant(gradient is not None)
    if gradient is None:
        gradient = tf.fill(tf.shape(theta), tf.constant(float("nan"), dtype=dtype))
    finite = tf.reduce_all(tf.math.is_finite(gradient))
    return connected, finite, tf.cast(gradient, dtype)


def _connected_jvp_parts(jvp: tf.Tensor | None, value: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    connected = tf.constant(jvp is not None)
    if jvp is None:
        jvp = tf.fill(tf.shape(value), tf.constant(float("nan"), dtype=value.dtype))
    return connected, tf.cast(jvp, value.dtype)


def _jvp_gradient_parts(
    acc0: tf.autodiff.ForwardAccumulator,
    target0: tf.Tensor,
    acc1: tf.autodiff.ForwardAccumulator,
    target1: tf.Tensor,
    dtype: tf.DType,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    conn0, jvp0 = _connected_jvp_parts(acc0.jvp(target0), tf.cast(target0, dtype))
    conn1, jvp1 = _connected_jvp_parts(acc1.jvp(target1), tf.cast(target1, dtype))
    gradient = tf.stack(
        [
            tf.reshape(tf.cast(jvp0, dtype), [-1])[0],
            tf.reshape(tf.cast(jvp1, dtype), [-1])[0],
        ]
    )
    connected = tf.logical_and(conn0, conn1)
    finite = tf.reduce_all(tf.math.is_finite(gradient))
    return connected, finite, gradient


def _tensor_parts(tensor: tf.Tensor, connected: tf.Tensor, finite: tf.Tensor, gradient: tf.Tensor) -> dict[str, Any]:
    return {
        "value_summary": _tensor_value_summary(tensor),
        "whole_sum_gradient": _gradient_summary_from_parts(connected, finite, gradient),
    }


def _compact_tensor_parts(value_parts: tuple[tf.Tensor, ...], connected: tf.Tensor, finite: tf.Tensor, gradient: tf.Tensor) -> dict[str, Any]:
    return {
        "value_summary": _compact_value_summary_from_parts(value_parts),
        "whole_sum_gradient": _gradient_summary_from_parts(connected, finite, gradient),
    }


def _checkpoint_tuple_to_row(
    checkpoint_name: str,
    values: tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor],
) -> dict[str, Any]:
    spec = STAGE_SPEC_BY_NAME[checkpoint_name]
    tensor, connected, finite, gradient = values
    row = _tensor_parts(tensor, connected, finite, gradient)
    row.update(
        {
            "stage": spec.stage,
            "checkpoint_kind": spec.checkpoint_kind,
            "expected_connected": spec.expected_connected,
            "expected_note": spec.expected_note,
            "unexpected_break": bool(spec.expected_connected)
            and (not row["whole_sum_gradient"]["connected"] or not row["whole_sum_gradient"]["finite"]),
        }
    )
    return row


def _compact_checkpoint_tuple_to_row(
    checkpoint_name: str,
    values: tuple[tf.Tensor, ...],
) -> dict[str, Any]:
    spec = STAGE_SPEC_BY_NAME[checkpoint_name]
    row = _compact_tensor_parts(values[:8], values[8], values[9], values[10])
    row.update(
        {
            "stage": spec.stage,
            "checkpoint_kind": spec.checkpoint_kind,
            "expected_connected": spec.expected_connected,
            "expected_note": spec.expected_note,
            "unexpected_break": bool(spec.expected_connected)
            and (not row["whole_sum_gradient"]["connected"] or not row["whole_sum_gradient"]["finite"]),
        }
    )
    return row


def _compiled_primary_function(
    fixture: lgssm_gate.LGSSMGateFixture,
    base_tensors: dict[str, Any],
    args: argparse.Namespace,
    dtype: tf.DType,
):
    @tf.function(jit_compile=args.jit_compile, reduce_retracing=True)
    def compiled(theta: tf.Tensor) -> tuple[tf.Tensor, ...]:
        theta = tf.cast(theta, dtype)
        with tf.GradientTape(persistent=True) as same_tape:
            same_tape.watch(theta)
            same_outputs = p02b._route_internal_outputs(fixture, base_tensors, theta, args, dtype)
            same_final_particles_sum = tf.reduce_sum(tf.cast(same_outputs.final_particles, dtype))
        value_connected, value_finite, value_gradient = _gradient_parts(same_tape, same_outputs.value, theta, dtype)
        loglik_connected, loglik_finite, loglik_gradient = _gradient_parts(
            same_tape,
            same_outputs.log_likelihood,
            theta,
            dtype,
        )
        prior_connected, prior_finite, prior_gradient = _gradient_parts(same_tape, same_outputs.prior, theta, dtype)
        particles_connected, particles_finite, particles_gradient = _gradient_parts(
            same_tape,
            same_final_particles_sum,
            theta,
            dtype,
        )
        same_arm = (
            same_outputs.value,
            same_outputs.log_likelihood,
            same_outputs.prior,
            value_connected,
            value_finite,
            value_gradient,
            loglik_connected,
            loglik_finite,
            loglik_gradient,
            prior_connected,
            prior_finite,
            prior_gradient,
            particles_connected,
            particles_finite,
            particles_gradient,
            same_outputs.route_invocations,
            same_outputs.active_resampling_mask_count,
            same_outputs.active_resampling_batch_entries,
            same_outputs.max_factor_marginal_residual,
            same_outputs.max_induced_row_residual,
            same_outputs.max_induced_column_residual,
            same_outputs.projection_iterations_used_max,
            same_outputs.finite_factors,
            same_outputs.finite_particles,
            same_outputs.nonnegative_factors,
            same_outputs.positive_g,
            same_outputs.route_outputs_finite,
        )

        with tf.GradientTape() as value_tape:
            value_tape.watch(theta)
            value_outputs = p02b._route_internal_outputs(fixture, base_tensors, theta, args, dtype)
        sep_value_connected, sep_value_finite, sep_value_gradient = _gradient_parts(
            value_tape,
            value_outputs.value,
            theta,
            dtype,
        )

        with tf.GradientTape() as loglik_tape:
            loglik_tape.watch(theta)
            loglik_outputs = p02b._route_internal_outputs(fixture, base_tensors, theta, args, dtype)
        sep_loglik_connected, sep_loglik_finite, sep_loglik_gradient = _gradient_parts(
            loglik_tape,
            loglik_outputs.log_likelihood,
            theta,
            dtype,
        )

        with tf.GradientTape() as prior_tape:
            prior_tape.watch(theta)
            prior = calibration._theta_prior_log_density(theta, args.theta_prior_scale, dtype)
        sep_prior_connected, sep_prior_finite, sep_prior_gradient = _gradient_parts(prior_tape, prior, theta, dtype)

        with tf.GradientTape() as particles_tape:
            particles_tape.watch(theta)
            particle_outputs = p02b._route_internal_outputs(fixture, base_tensors, theta, args, dtype)
            final_particles_sum = tf.reduce_sum(tf.cast(particle_outputs.final_particles, dtype))
        sep_particles_connected, sep_particles_finite, sep_particles_gradient = _gradient_parts(
            particles_tape,
            final_particles_sum,
            theta,
            dtype,
        )
        separated_arm = (
            value_outputs.value,
            loglik_outputs.log_likelihood,
            prior,
            sep_value_connected,
            sep_value_finite,
            sep_value_gradient,
            sep_loglik_connected,
            sep_loglik_finite,
            sep_loglik_gradient,
            sep_prior_connected,
            sep_prior_finite,
            sep_prior_gradient,
            sep_particles_connected,
            sep_particles_finite,
            sep_particles_gradient,
            value_outputs.route_invocations,
            value_outputs.active_resampling_mask_count,
            value_outputs.active_resampling_batch_entries,
            value_outputs.route_outputs_finite,
        )
        return same_arm + separated_arm

    return compiled


def _compiled_stage_function(
    fixture: lgssm_gate.LGSSMGateFixture,
    base_tensors: dict[str, Any],
    args: argparse.Namespace,
    dtype: tf.DType,
    checkpoint_names: tuple[str, ...],
):
    @tf.function(jit_compile=args.jit_compile, reduce_retracing=True)
    def compiled(theta: tf.Tensor) -> tuple[tf.Tensor, ...]:
        theta = tf.cast(theta, dtype)
        tangent0 = tf.stack([tf.constant(1.0, dtype=dtype), tf.constant(0.0, dtype=dtype)])
        tangent1 = tf.stack([tf.constant(0.0, dtype=dtype), tf.constant(1.0, dtype=dtype)])
        with tf.autodiff.ForwardAccumulator(theta, tangent0) as acc0:
            outputs0 = p02b._route_internal_outputs(fixture, base_tensors, theta, args, dtype)
            value_parts = tuple(_compact_value_parts(outputs0.checkpoints[name], dtype) for name in checkpoint_names)
            scalars0 = tuple(
                tf.reduce_sum(tf.reshape(tf.cast(outputs0.checkpoints[name], dtype), [-1]))
                for name in checkpoint_names
            )
            final_particles_sum0 = tf.reduce_sum(tf.cast(outputs0.final_particles, dtype))
        with tf.autodiff.ForwardAccumulator(theta, tangent1) as acc1:
            outputs = p02b._route_internal_outputs(fixture, base_tensors, theta, args, dtype)
            scalars = tuple(
                tf.reduce_sum(tf.reshape(tf.cast(outputs.checkpoints[name], dtype), [-1]))
                for name in checkpoint_names
            )
            final_particles_sum = tf.reduce_sum(tf.cast(outputs.final_particles, dtype))
        value_connected, value_finite, value_gradient = _jvp_gradient_parts(
            acc0, outputs0.value, acc1, outputs.value, dtype
        )
        loglik_connected, loglik_finite, loglik_gradient = _jvp_gradient_parts(
            acc0, outputs0.log_likelihood, acc1, outputs.log_likelihood, dtype
        )
        prior_connected, prior_finite, prior_gradient = _jvp_gradient_parts(
            acc0, outputs0.prior, acc1, outputs.prior, dtype
        )
        particles_connected, particles_finite, particles_gradient = _jvp_gradient_parts(
            acc0, final_particles_sum0, acc1, final_particles_sum, dtype
        )
        parts: tuple[tf.Tensor, ...] = ()
        parts += (
            outputs0.value,
            outputs0.log_likelihood,
            outputs0.prior,
            value_connected,
            value_finite,
            value_gradient,
            loglik_connected,
            loglik_finite,
            loglik_gradient,
            prior_connected,
            prior_finite,
            prior_gradient,
            particles_connected,
            particles_finite,
            particles_gradient,
            outputs0.route_invocations,
            outputs0.active_resampling_mask_count,
            outputs0.active_resampling_batch_entries,
            outputs0.max_factor_marginal_residual,
            outputs0.max_induced_row_residual,
            outputs0.max_induced_column_residual,
            outputs0.projection_iterations_used_max,
            outputs0.finite_factors,
            outputs0.finite_particles,
            outputs0.nonnegative_factors,
            outputs0.positive_g,
            outputs0.route_outputs_finite,
        )
        for compact_parts, scalar0, scalar in zip(value_parts, scalars0, scalars, strict=True):
            connected, finite, gradient = _jvp_gradient_parts(acc0, scalar0, acc1, scalar, dtype)
            parts += compact_parts + (connected, finite, gradient)
        return parts

    return compiled


def _stage_route_summary_from_compiled(values: tuple[tf.Tensor, ...]) -> dict[str, Any]:
    return {
        "mode": "staged_only_route_summary",
        "output_device": values[0].device,
        "value": _to_json_scalar(values[0]),
        "log_likelihood": _to_json_scalar(values[1]),
        "prior": _to_json_scalar(values[2]),
        "value_gradient": _gradient_summary_from_parts(values[3], values[4], values[5]),
        "log_likelihood_gradient": _gradient_summary_from_parts(values[6], values[7], values[8]),
        "prior_gradient": _gradient_summary_from_parts(values[9], values[10], values[11]),
        "final_particles_sum_gradient": _gradient_summary_from_parts(values[12], values[13], values[14]),
        "route_invocations": _to_json_int(values[15]),
        "active_resampling_mask_count": _to_json_int(values[16]),
        "active_resampling_batch_entries": _to_json_int(values[17]),
        "max_factor_marginal_residual": _to_json_scalar(values[18]),
        "max_induced_row_residual": _to_json_scalar(values[19]),
        "max_induced_column_residual": _to_json_scalar(values[20]),
        "projection_iterations_used_max": _to_json_int(values[21]),
        "finite_factors": _to_json_bool(values[22]),
        "finite_particles": _to_json_bool(values[23]),
        "nonnegative_factors": _to_json_bool(values[24]),
        "positive_g": _to_json_bool(values[25]),
        "route_outputs_finite": _to_json_bool(values[26]),
    }


def _primary_rows_from_compiled(values: tuple[tf.Tensor, ...]) -> tuple[dict[str, Any], dict[str, Any]]:
    same = {
        "mode": "same_tape_internal",
        "output_device": values[0].device,
        "value": _to_json_scalar(values[0]),
        "log_likelihood": _to_json_scalar(values[1]),
        "prior": _to_json_scalar(values[2]),
        "value_gradient": _gradient_summary_from_parts(values[3], values[4], values[5]),
        "log_likelihood_gradient": _gradient_summary_from_parts(values[6], values[7], values[8]),
        "prior_gradient": _gradient_summary_from_parts(values[9], values[10], values[11]),
        "final_particles_sum_gradient": _gradient_summary_from_parts(values[12], values[13], values[14]),
        "route_invocations": _to_json_int(values[15]),
        "active_resampling_mask_count": _to_json_int(values[16]),
        "active_resampling_batch_entries": _to_json_int(values[17]),
        "max_factor_marginal_residual": _to_json_scalar(values[18]),
        "max_induced_row_residual": _to_json_scalar(values[19]),
        "max_induced_column_residual": _to_json_scalar(values[20]),
        "projection_iterations_used_max": _to_json_int(values[21]),
        "finite_factors": _to_json_bool(values[22]),
        "finite_particles": _to_json_bool(values[23]),
        "nonnegative_factors": _to_json_bool(values[24]),
        "positive_g": _to_json_bool(values[25]),
        "route_outputs_finite": _to_json_bool(values[26]),
    }
    parts = values[27:]
    separated = {
        "mode": "p02a_style_separated_tape",
        "output_device": parts[0].device,
        "readout_pattern": "separate route execution per value/loglik/final-particle readout plus separate prior tape",
        "value": _to_json_scalar(parts[0]),
        "log_likelihood": _to_json_scalar(parts[1]),
        "prior": _to_json_scalar(parts[2]),
        "value_gradient": _gradient_summary_from_parts(parts[3], parts[4], parts[5]),
        "log_likelihood_gradient": _gradient_summary_from_parts(parts[6], parts[7], parts[8]),
        "prior_gradient": _gradient_summary_from_parts(parts[9], parts[10], parts[11]),
        "final_particles_sum_gradient": _gradient_summary_from_parts(parts[12], parts[13], parts[14]),
        "route_invocations": _to_json_int(parts[15]),
        "active_resampling_mask_count": _to_json_int(parts[16]),
        "active_resampling_batch_entries": _to_json_int(parts[17]),
        "route_outputs_finite": _to_json_bool(parts[18]),
    }
    return same, separated


def _stage_rows_from_compiled(checkpoint_names: tuple[str, ...], values: tuple[tf.Tensor, ...]) -> dict[str, Any]:
    rows = {}
    values = values[STAGE_ROUTE_SUMMARY_OUTPUT_COUNT:]
    for index, name in enumerate(checkpoint_names):
        start = index * COMPACT_STAGE_OUTPUT_COUNT
        rows[name] = _compact_checkpoint_tuple_to_row(name, values[start : start + COMPACT_STAGE_OUTPUT_COUNT])
    return rows


def _ab_comparison(same: dict[str, Any], separated: dict[str, Any]) -> dict[str, Any]:
    keys = ("value_gradient", "log_likelihood_gradient", "final_particles_sum_gradient")
    fields = {}
    same_connected_all = True
    separated_disconnected_any = False
    for key in keys:
        same_grad = same[key]
        sep_grad = separated[key]
        fields[key] = {
            "same_tape_connected": bool(same_grad["connected"]),
            "same_tape_finite": bool(same_grad["finite"]),
            "separated_tape_connected": bool(sep_grad["connected"]),
            "separated_tape_finite": bool(sep_grad["finite"]),
            "connection_differs": bool(same_grad["connected"]) != bool(sep_grad["connected"]),
        }
        same_connected_all = same_connected_all and bool(same_grad["connected"]) and bool(same_grad["finite"])
        separated_disconnected_any = separated_disconnected_any or not bool(sep_grad["connected"])
    return {
        "fields": fields,
        "supports_h1_tape_artifact": bool(same_connected_all and separated_disconnected_any),
    }


def _first_observed_expected_break(checkpoints: dict[str, Any]) -> dict[str, Any]:
    for spec in STAGE_SPECS:
        row = checkpoints.get(spec.name)
        if row is None:
            return {
                "checkpoint": spec.name,
                "stage": spec.stage,
                "reason": "missing_required_checkpoint",
                "expected_connected": spec.expected_connected,
            }
        grad = row["whole_sum_gradient"]
        if spec.expected_connected and not grad["connected"]:
            return {
                "checkpoint": spec.name,
                "stage": spec.stage,
                "reason": "expected_connected_whole_sum_gradient_disconnected",
                "expected_connected": True,
            }
        if spec.expected_connected and not grad["finite"]:
            return {
                "checkpoint": spec.name,
                "stage": spec.stage,
                "reason": "expected_connected_whole_sum_gradient_nonfinite",
                "expected_connected": True,
            }
    return {
        "checkpoint": None,
        "stage": None,
        "reason": "no_observed_expected_connected_break",
        "expected_connected": None,
    }


def _artifact_vetoes(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[str]:
    vetoes = []
    required = {spec.name for spec in STAGE_SPECS}
    for row in rows:
        label = f"{row['seed']}:{row['probe_label']}"
        route_summary = row["route_summary"]
        if args.expect_device_kind == "gpu":
            if "GPU" not in route_summary["output_device"].upper():
                vetoes.append(f"{label}:route_summary_not_gpu_output")
        if not route_summary["route_outputs_finite"]:
            vetoes.append(f"{label}:route_summary_route_outputs_nonfinite")
        missing = sorted(required - set(row["staged_checkpoints"]))
        if missing:
            vetoes.append(f"{label}:missing_required_stages:{missing}")
        if route_summary["route_invocations"] != route_summary["active_resampling_mask_count"]:
            vetoes.append(f"{label}:unexpected_route_invocation_count")
        same = row.get("same_tape")
        separated = row.get("p02a_style_separated_tape")
        if same is not None and separated is not None:
            if args.expect_device_kind == "gpu":
                if "GPU" not in same["output_device"].upper():
                    vetoes.append(f"{label}:same_tape_not_gpu_output")
                if "GPU" not in separated["output_device"].upper():
                    vetoes.append(f"{label}:separated_tape_not_gpu_output")
            if not same["route_outputs_finite"]:
                vetoes.append(f"{label}:same_tape_route_outputs_nonfinite")
            if not separated["route_outputs_finite"]:
                vetoes.append(f"{label}:separated_tape_route_outputs_nonfinite")
        for name in ("scaled_Q", "scaled_R"):
            checkpoint = row["staged_checkpoints"].get(name)
            if checkpoint is not None and not checkpoint["whole_sum_gradient"]["connected"]:
                vetoes.append(f"{label}:{name}:direct_scaled_covariance_gradient_disconnected")
    return vetoes


def _diagnostic_findings(row: dict[str, Any]) -> list[str]:
    findings = []
    label = f"{row['seed']}:{row['probe_label']}"
    for mode_key in ("route_summary", "same_tape", "p02a_style_separated_tape"):
        mode = row.get(mode_key)
        if mode is None:
            continue
        prefix = {
            "route_summary": "staged_only_route_summary",
            "same_tape": "same_tape",
            "p02a_style_separated_tape": "separated_tape",
        }[mode_key]
        for gradient_key in ("value_gradient", "log_likelihood_gradient", "final_particles_sum_gradient"):
            gradient = mode[gradient_key]
            if not gradient["connected"]:
                findings.append(f"{label}:{prefix}:{gradient_key}:disconnected")
            elif not gradient["finite"]:
                findings.append(f"{label}:{prefix}:{gradient_key}:nonfinite")
    first_break = row["first_observed_expected_connected_break"]
    if first_break["checkpoint"] is not None:
        findings.append(
            f"{label}:first_observed_expected_break:{first_break['checkpoint']}:{first_break['reason']}"
        )
    return findings


def _h6_sir_inventory() -> dict[str, Any]:
    return {
        "question": "Do existing SIR tests contradict the P02A LGSSM posterior-parameter gradient failure?",
        "answer": "No. They exercise a forward route or a different gradient target/path.",
        "items": [
            {
                "path": "tests/test_actual_sir_low_rank_route_validation.py",
                "line_anchor": "131-159",
                "observed_coverage": "asserts route execution, finite factors, residuals, and low-rank metadata",
                "gradient_target": None,
                "contradicts_p02a": False,
            },
            {
                "path": "docs/benchmarks/benchmark_actual_sir_low_rank_route_validation.py",
                "line_anchor": "508-680",
                "observed_coverage": "forward low-rank route loop and resampling",
                "gradient_target": None,
                "contradicts_p02a": False,
            },
            {
                "path": "docs/benchmarks/run_actual_sir_nystrom_gradient_mechanics_smoke.py",
                "line_anchor": "133-200",
                "observed_coverage": "Nystrom mechanics gradient smoke",
                "gradient_target": "initial_particles",
                "contradicts_p02a": False,
            },
        ],
    }


def _row(
    seed: int,
    label: str,
    theta: tf.Tensor,
    route_summary: dict[str, Any],
    checkpoints: dict[str, Any],
    timing: dict[str, float],
    same: dict[str, Any] | None = None,
    separated: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row = {
        "seed": int(seed),
        "probe_label": label,
        "theta": _to_json_vector(theta),
        "readout_mode": "primary-and-staged" if same is not None and separated is not None else "staged-only",
        "timing_seconds": timing,
        "route_summary": route_summary,
        "staged_checkpoints": checkpoints,
    }
    if same is not None and separated is not None:
        row["same_tape"] = same
        row["p02a_style_separated_tape"] = separated
        row["ab_comparison"] = _ab_comparison(same, separated)
    else:
        row["ab_comparison"] = None
    row["first_observed_expected_connected_break"] = _first_observed_expected_break(checkpoints)
    row["diagnostic_findings"] = _diagnostic_findings(row)
    return row


def build_result(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    started_at = _dt.datetime.now(tz=_dt.UTC).isoformat()
    dtype, config = _configure(args)
    seed_probe_pairs = _seed_probe_pairs(args)
    probe_specs = _probe_spec_by_label(args, dtype)
    rows = []
    fixtures: dict[int, lgssm_gate.LGSSMGateFixture] = {}
    base_tensors_by_seed: dict[int, dict[str, Any]] = {}
    primary_by_seed = {}
    stage_by_seed = {}
    with tf.device(args.device):
        for seed, label in seed_probe_pairs:
            if label not in probe_specs:
                raise ValueError(f"unknown probe label: {label}")
            if seed not in fixtures:
                fixture = lgssm_gate.build_lgssm_gate_fixture(args.case_id, seed, args)
                fixtures[seed] = fixture
                base_tensors_by_seed[seed] = lgssm_gate._fixture_tensors(fixture, args.num_particles, seed, dtype)
                if args.readout_mode == "primary-and-staged":
                    primary_by_seed[seed] = _compiled_primary_function(fixture, base_tensors_by_seed[seed], args, dtype)
                stage_by_seed[seed] = _compiled_stage_function(
                    fixture,
                    base_tensors_by_seed[seed],
                    args,
                    dtype,
                    STAGE_NAMES,
                )
            theta = tf.cast(probe_specs[label]["theta"], dtype)
            primary_call_seconds = None
            same = None
            separated = None
            if args.readout_mode == "primary-and-staged":
                primary_started = time.perf_counter()
                same, separated = _primary_rows_from_compiled(primary_by_seed[seed](theta))
                primary_call_seconds = time.perf_counter() - primary_started
            stage_started = time.perf_counter()
            stage_values = stage_by_seed[seed](theta)
            staged_call_seconds = time.perf_counter() - stage_started
            route_summary = _stage_route_summary_from_compiled(stage_values)
            checkpoints = _stage_rows_from_compiled(STAGE_NAMES, stage_values)
            timing = {
                "staged_compiled_call_seconds": staged_call_seconds,
                "primary_compiled_call_seconds": primary_call_seconds,
            }
            rows.append(_row(seed, label, theta, route_summary, checkpoints, timing, same, separated))
    artifact_vetoes = _artifact_vetoes(rows, args)
    ended_at = _dt.datetime.now(tz=_dt.UTC).isoformat()
    evidence_class = "cpu_hidden_debug_only" if args.device_scope == "cpu" else GPU_TRUST_BASIS
    status = "PASS" if not artifact_vetoes else "FAIL"
    diagnostic_findings = [finding for row in rows for finding in row["diagnostic_findings"]]
    return {
        "schema_version": "low_rank_ledh_staged_gradient_path.v1",
        "phase": args.phase_id,
        "status": status,
        "evidence_class": evidence_class,
        "gpu_trust_basis": None if args.device_scope == "cpu" else GPU_TRUST_BASIS,
        "question": "localize the first expected-connected low-rank route gradient-path break",
        "interpretation_scope": "staged whole-sum gradient localization only",
        "candidate": calibration._candidate_settings(args),
        "stage_specs": [
            {
                "name": spec.name,
                "stage": spec.stage,
                "expected_connected": spec.expected_connected,
                "expected_note": spec.expected_note,
                "checkpoint_kind": spec.checkpoint_kind,
            }
            for spec in STAGE_SPECS
        ],
        "artifact_vetoes": artifact_vetoes,
        "diagnostic_findings": diagnostic_findings,
        "rows": rows,
        "h6_sir_inventory": _h6_sir_inventory(),
        "baseline_artifacts": {
            "p02a_result_note": P02A_RESULT_PATH,
            "p02b_blocker_result_note": P02B_RESULT_PATH,
        },
        "run_manifest": {
            "started_at": started_at,
            "ended_at": ended_at,
            "wall_time_seconds": time.perf_counter() - started,
            "working_directory": str(ROOT),
            "command": " ".join(sys.argv),
            "git_commit": _git_output(["git", "rev-parse", "HEAD"]),
            "git_status_short": _git_output(["git", "status", "--short"]),
            "python_executable": sys.executable,
            "python_version": sys.version,
            "platform": platform.platform(),
            "tensorflow_version": tf.__version__,
            "tensorflow_probability_version": tfp.__version__,
            "plan_path": PLAN_PATH,
            "subplan_path": SUBPLAN_PATH,
            "device_scope": args.device_scope,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "device": args.device,
            "expect_device_kind": args.expect_device_kind,
            "jit_compile": bool(args.jit_compile),
            "readout_mode": args.readout_mode,
            "staged_readout_mode": "single_compiled_compact_whole_sum_per_seed_with_route_summary",
            "case_id": args.case_id,
            "seed_probe_pairs": [{"seed": seed, "probe_label": label} for seed, label in seed_probe_pairs],
            **config,
        },
        "nonclaims": list(NONCLAIMS),
    }


def write_markdown(result: dict[str, Any], path: Path, json_path: Path | None = None) -> None:
    lines = [
        "# P02B-R2 Compact Staged Gradient Path Diagnostic",
        "",
        f"- Status: `{result['status']}`",
        f"- Phase: `{result['phase']}`",
        f"- Evidence class: `{result['evidence_class']}`",
        f"- Artifact vetoes: `{result['artifact_vetoes']}`",
        f"- Diagnostic findings: `{result['diagnostic_findings']}`",
        f"- Interpretation scope: `{result['interpretation_scope']}`",
    ]
    if json_path is not None:
        lines.append(f"- JSON artifact: `{json_path}`")
    lines.extend(
        [
            "",
            "## Rows",
            "",
            "| Seed | Probe | Readout | Route loglik grad | Route final-particle grad | First expected break | H1 tape artifact | Staged call seconds |",
            "| ---: | --- | --- | --- | --- | --- | --- | ---: |",
        ]
    )
    for row in result["rows"]:
        first_break = row["first_observed_expected_connected_break"]
        route_summary = row["route_summary"]
        h1 = None if row["ab_comparison"] is None else row["ab_comparison"]["supports_h1_tape_artifact"]
        lines.append(
            "| {seed} | `{probe}` | `{readout}` | `{route_loglik}` | `{route_particles}` | `{breakpoint}:{reason}` | `{h1}` | `{staged_seconds:.6f}` |".format(
                seed=row["seed"],
                probe=row["probe_label"],
                readout=row["readout_mode"],
                route_loglik=route_summary["log_likelihood_gradient"],
                route_particles=route_summary["final_particles_sum_gradient"],
                breakpoint=first_break["checkpoint"],
                reason=first_break["reason"],
                h1=h1,
                staged_seconds=row["timing_seconds"]["staged_compiled_call_seconds"],
            )
        )
    lines.extend(
        [
            "",
            "## H6 SIR Inventory",
            "",
            f"- Answer: {result['h6_sir_inventory']['answer']}",
            "",
            "## Run Manifest",
            "",
            f"- Git commit: `{result['run_manifest']['git_commit']}`",
            f"- Device scope: `{result['run_manifest']['device_scope']}`",
            f"- CUDA_VISIBLE_DEVICES: `{result['run_manifest']['cuda_visible_devices']}`",
            f"- TF32 recorded: `{result['run_manifest']['tf32_execution_recorded']}`",
            f"- JIT compile: `{result['run_manifest']['jit_compile']}`",
            f"- Readout mode: `{result['run_manifest']['readout_mode']}`",
            f"- GPU trust basis: `{result.get('gpu_trust_basis')}`",
            "",
            "## Non-Claims",
            "",
        ]
    )
    lines.extend(f"- {claim}" for claim in result["nonclaims"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    raise RuntimeError("ARCHIVAL_WRONG_TRANSPORT_CHUNK_POLICY: this route is preserved only as provenance and cannot emit new evidence")
    args = _parse_args()
    output = Path(args.output)
    markdown = Path(args.markdown_output) if args.markdown_output else output.with_suffix(".md")
    output.parent.mkdir(parents=True, exist_ok=True)
    markdown.parent.mkdir(parents=True, exist_ok=True)
    result = build_result(args)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(result, markdown, output)
    if not args.quiet:
        print(json.dumps({"status": result["status"], "artifact_vetoes": result["artifact_vetoes"]}, indent=2))


if __name__ == "__main__":
    main()
