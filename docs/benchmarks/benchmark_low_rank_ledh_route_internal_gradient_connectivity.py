"""Route-internal gradient connectivity diagnostic for low-rank LEDH-PFPF-OT.

This P02B diagnostic reuses the P02/P02A LGSSM fixture and probe definitions.
It compares a same-tape internal route readout with a P02A-style separated-tape
readout, then records component/block-sensitive gradients for selected internal
low-rank route tensors.  It is a localization artifact only, not a repair,
calibration, posterior-correctness, HMC-readiness, default-readiness, or
scientific-validity artifact.
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
from pathlib import Path
from typing import Any, NamedTuple


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
from experiments.dpf_implementation.tf_tfp.filters import (  # noqa: E402
    experimental_batched_ledh_pfpf_ot_streaming_tf as streaming_tf,
)
from experiments.dpf_implementation.tf_tfp.filters import (  # noqa: E402
    experimental_batched_ledh_pfpf_ot_tf as core_tf,
)
from experiments.dpf_implementation.tf_tfp.resampling.low_rank_coupling_solver_tf import (  # noqa: E402
    low_rank_coupling_solver_resample_tensors_tf,
)


PLAN_PATH = (
    "docs/plans/bayesfilter-dpf-ledh-pfpf-ot-low-rank-residual-posterior-gradient-"
    "calibration-master-program-2026-06-24.md"
)
P02B_SUBPLAN_PATH = (
    "docs/plans/bayesfilter-dpf-ledh-pfpf-ot-low-rank-residual-posterior-gradient-"
    "calibration-p02b-route-internal-gradient-connectivity-subplan-2026-06-25.md"
)
P02_RESULT_PATH = (
    "docs/plans/bayesfilter-dpf-ledh-pfpf-ot-low-rank-residual-posterior-gradient-"
    "calibration-p02-reproduction-determinism-result-2026-06-24.md"
)
P02A_RESULT_PATH = (
    "docs/plans/bayesfilter-dpf-ledh-pfpf-ot-low-rank-residual-posterior-gradient-"
    "calibration-p02a-gradient-repair-diagnostic-result-2026-06-25.md"
)
P02_JSON_PATH = "docs/benchmarks/low-rank-residual-posterior-gradient-calibration-p02-reproduction-2026-06-24.json"
P02A_JSON_PATH = (
    "docs/benchmarks/low-rank-residual-posterior-gradient-calibration-p02a-gradient-repair-diagnostic-2026-06-25.json"
)
GPU_TRUST_BASIS = "owner_designated_managed_session_visible_gpu_trusted"
DEFAULT_SEED_PROBES = ((91003, "center"), (91002, "qr_plus"))
REQUIRED_TENSORS = (
    "scaled_Q",
    "scaled_R",
    "pre_flow_t0",
    "post_flow_t0",
    "pre_flow_log_density_t0",
    "forward_log_det_t0",
    "transition_log_density_t0",
    "observation_log_density_t0",
    "corrected_log_weights_t0",
    "normalized_weights_t0",
    "incremental_log_likelihood_t0",
    "normalized_log_weights_t0",
    "scaled_x_t0",
    "eps_q_t0",
    "eps_r_t0",
    "eps_g_t0",
    "q_factor_t0",
    "r_factor_t0",
    "g_weights_t0",
    "transported_particles_t0",
    "resampled_log_weights_t0",
    "next_pre_flow_t1",
    "next_corrected_log_weights_t1",
    "next_incremental_log_likelihood_t1",
    "final_particles",
    "final_log_likelihood",
)
NONCLAIMS = (
    "P02B route-internal gradient-connectivity diagnostic only",
    "no low-rank solver repair claim",
    "no calibrated residual threshold claim",
    "no P03 handoff claim",
    "no posterior correctness claim",
    "no HMC readiness claim",
    "no default/package/public API readiness claim",
    "no statistical superiority claim",
    "no scientific validity claim",
)


class RouteInternalOutputs(NamedTuple):
    log_likelihood: tf.Tensor
    value: tf.Tensor
    prior: tf.Tensor
    final_particles: tf.Tensor
    route_invocations: tf.Tensor
    active_resampling_mask_count: tf.Tensor
    active_resampling_batch_entries: tf.Tensor
    max_factor_marginal_residual: tf.Tensor
    max_induced_row_residual: tf.Tensor
    max_induced_column_residual: tf.Tensor
    projection_iterations_used_max: tf.Tensor
    finite_factors: tf.Tensor
    finite_particles: tf.Tensor
    nonnegative_factors: tf.Tensor
    positive_g: tf.Tensor
    route_outputs_finite: tf.Tensor
    checkpoints: dict[str, tf.Tensor]


def _compiled_internal_function(
    fixture: lgssm_gate.LGSSMGateFixture,
    base_tensors: dict[str, Any],
    args: argparse.Namespace,
    dtype: tf.DType,
):
    @tf.function(jit_compile=args.jit_compile, reduce_retracing=True)
    def compiled(theta: tf.Tensor) -> tuple[tf.Tensor, ...]:
        outputs = _route_internal_outputs(fixture, base_tensors, theta, args, dtype)
        return (
            outputs.log_likelihood,
            outputs.value,
            outputs.prior,
            outputs.final_particles,
            outputs.route_invocations,
            outputs.active_resampling_mask_count,
            outputs.active_resampling_batch_entries,
            outputs.max_factor_marginal_residual,
            outputs.max_induced_row_residual,
            outputs.max_induced_column_residual,
            outputs.projection_iterations_used_max,
            outputs.finite_factors,
            outputs.finite_particles,
            outputs.nonnegative_factors,
            outputs.positive_g,
            outputs.route_outputs_finite,
            *(outputs.checkpoints[name] for name in REQUIRED_TENSORS),
        )

    return compiled


def _unpack_internal_tuple(values: tuple[tf.Tensor, ...]) -> RouteInternalOutputs:
    checkpoint_values = values[16:]
    return RouteInternalOutputs(
        log_likelihood=values[0],
        value=values[1],
        prior=values[2],
        final_particles=values[3],
        route_invocations=values[4],
        active_resampling_mask_count=values[5],
        active_resampling_batch_entries=values[6],
        max_factor_marginal_residual=values[7],
        max_induced_row_residual=values[8],
        max_induced_column_residual=values[9],
        projection_iterations_used_max=values[10],
        finite_factors=values[11],
        finite_particles=values[12],
        nonnegative_factors=values[13],
        positive_g=values[14],
        route_outputs_finite=values[15],
        checkpoints=dict(zip(REQUIRED_TENSORS, checkpoint_values, strict=True)),
    )


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
    parser.add_argument("--phase-id", default="LOW_RANK_ROUTE_INTERNAL_GRADIENT_CONNECTIVITY")
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
        raise ValueError("time_steps must be at least 2 for required H5 next-step checkpoints")
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


def _fill_disconnected_gradient(theta: tf.Tensor, dtype: tf.DType) -> tf.Tensor:
    return tf.fill(tf.shape(theta), tf.constant(float("nan"), dtype=dtype))


def _gradient_parts_for_target(
    tape: tf.GradientTape,
    target: tf.Tensor,
    theta: tf.Tensor,
    dtype: tf.DType,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    gradient = tape.gradient(target, theta)
    connected = tf.constant(gradient is not None)
    if gradient is None:
        gradient = _fill_disconnected_gradient(theta, dtype)
    finite = tf.reduce_all(tf.math.is_finite(gradient))
    return connected, finite, tf.cast(gradient, dtype)


def _checkpoint_gradient_outputs(
    tape: tf.GradientTape,
    tensor: tf.Tensor,
    theta: tf.Tensor,
    dtype: tf.DType,
) -> tuple[tf.Tensor, ...]:
    flat = tf.reshape(tf.cast(tensor, dtype), [-1])
    size = tf.size(flat)
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

    last = tf.maximum(size - 1, 0)
    selected_indices = tf.stack([tf.constant(0, dtype=tf.int32), size // 2, last])
    selected_values = tf.gather(flat, selected_indices)
    whole_connected, whole_finite, whole_gradient = _gradient_parts_for_target(
        tape,
        tf.reduce_sum(flat),
        theta,
        dtype,
    )
    selected_parts = [
        _gradient_parts_for_target(tape, tf.gather(flat, selected_indices[index]), theta, dtype)
        for index in range(3)
    ]
    selected_connected = tf.stack([part[0] for part in selected_parts])
    selected_finite = tf.stack([part[1] for part in selected_parts])
    selected_gradients = tf.stack([part[2] for part in selected_parts])

    block = tf.maximum(tf.constant(1, dtype=tf.int32), tf.minimum(size, tf.maximum(tf.constant(1, dtype=tf.int32), size // 4)))
    block_starts = tf.stack(
        [
            tf.constant(0, dtype=tf.int32),
            tf.maximum(tf.constant(0, dtype=tf.int32), size // 2 - block // 2),
            tf.maximum(tf.constant(0, dtype=tf.int32), size - block),
        ]
    )
    block_ends = tf.minimum(size, block_starts + block)
    block_parts = [
        _gradient_parts_for_target(tape, tf.reduce_sum(flat[block_starts[index] : block_ends[index]]), theta, dtype)
        for index in range(3)
    ]
    block_connected = tf.stack([part[0] for part in block_parts])
    block_finite = tf.stack([part[1] for part in block_parts])
    block_gradients = tf.stack([part[2] for part in block_parts])
    return (
        size,
        finite_count,
        value_min,
        value_max,
        value_mean,
        selected_indices,
        selected_values,
        whole_connected,
        whole_finite,
        whole_gradient,
        selected_connected,
        selected_finite,
        selected_gradients,
        block_starts,
        block_ends,
        block_connected,
        block_finite,
        block_gradients,
    )


def _arm_gradient_outputs(
    tape: tf.GradientTape,
    outputs: RouteInternalOutputs,
    theta: tf.Tensor,
    dtype: tf.DType,
) -> tuple[tf.Tensor, ...]:
    value_connected, value_finite, value_gradient = _gradient_parts_for_target(tape, outputs.value, theta, dtype)
    loglik_connected, loglik_finite, loglik_gradient = _gradient_parts_for_target(
        tape,
        outputs.log_likelihood,
        theta,
        dtype,
    )
    prior_connected, prior_finite, prior_gradient = _gradient_parts_for_target(tape, outputs.prior, theta, dtype)
    particles_connected, particles_finite, particles_gradient = _gradient_parts_for_target(
        tape,
        tf.reduce_sum(outputs.final_particles),
        theta,
        dtype,
    )
    return (
        outputs.value,
        outputs.log_likelihood,
        outputs.prior,
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
    )


SAME_BASE_COUNT = 27
CHECKPOINT_OUTPUT_COUNT = 18
SEPARATED_BASE_COUNT = 19


def _compiled_probe_function(
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
            same_outputs = _route_internal_outputs(fixture, base_tensors, theta, args, dtype)
        same_arm = _arm_gradient_outputs(same_tape, same_outputs, theta, dtype)
        checkpoint_parts: tuple[tf.Tensor, ...] = ()
        for name in REQUIRED_TENSORS:
            checkpoint_parts += _checkpoint_gradient_outputs(same_tape, same_outputs.checkpoints[name], theta, dtype)

        with tf.GradientTape() as value_tape:
            value_tape.watch(theta)
            value_outputs = _route_internal_outputs(fixture, base_tensors, theta, args, dtype)
        value_connected, value_finite, value_gradient = _gradient_parts_for_target(
            value_tape,
            value_outputs.value,
            theta,
            dtype,
        )

        with tf.GradientTape() as loglik_tape:
            loglik_tape.watch(theta)
            loglik_outputs = _route_internal_outputs(fixture, base_tensors, theta, args, dtype)
        loglik_connected, loglik_finite, loglik_gradient = _gradient_parts_for_target(
            loglik_tape,
            loglik_outputs.log_likelihood,
            theta,
            dtype,
        )

        with tf.GradientTape() as prior_tape:
            prior_tape.watch(theta)
            prior = calibration._theta_prior_log_density(theta, args.theta_prior_scale, dtype)
        prior_connected, prior_finite, prior_gradient = _gradient_parts_for_target(prior_tape, prior, theta, dtype)

        with tf.GradientTape() as particles_tape:
            particles_tape.watch(theta)
            particle_outputs = _route_internal_outputs(fixture, base_tensors, theta, args, dtype)
            final_particles_sum = tf.reduce_sum(particle_outputs.final_particles)
        particles_connected, particles_finite, particles_gradient = _gradient_parts_for_target(
            particles_tape,
            final_particles_sum,
            theta,
            dtype,
        )

        same_metrics = (
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
        )
        separated_arm = (
            value_outputs.value,
            loglik_outputs.log_likelihood,
            prior,
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
            value_outputs.route_invocations,
            value_outputs.active_resampling_mask_count,
            value_outputs.active_resampling_batch_entries,
            value_outputs.route_outputs_finite,
        )
        return same_arm + same_metrics + (same_outputs.route_outputs_finite,) + checkpoint_parts + separated_arm

    return compiled


def _compiled_primary_ab_function(
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
            same_outputs = _route_internal_outputs(fixture, base_tensors, theta, args, dtype)
        same_arm = _arm_gradient_outputs(same_tape, same_outputs, theta, dtype)
        same_metrics = (
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
            value_outputs = _route_internal_outputs(fixture, base_tensors, theta, args, dtype)
        value_connected, value_finite, value_gradient = _gradient_parts_for_target(
            value_tape,
            value_outputs.value,
            theta,
            dtype,
        )

        with tf.GradientTape() as loglik_tape:
            loglik_tape.watch(theta)
            loglik_outputs = _route_internal_outputs(fixture, base_tensors, theta, args, dtype)
        loglik_connected, loglik_finite, loglik_gradient = _gradient_parts_for_target(
            loglik_tape,
            loglik_outputs.log_likelihood,
            theta,
            dtype,
        )

        with tf.GradientTape() as prior_tape:
            prior_tape.watch(theta)
            prior = calibration._theta_prior_log_density(theta, args.theta_prior_scale, dtype)
        prior_connected, prior_finite, prior_gradient = _gradient_parts_for_target(prior_tape, prior, theta, dtype)

        with tf.GradientTape() as particles_tape:
            particles_tape.watch(theta)
            particle_outputs = _route_internal_outputs(fixture, base_tensors, theta, args, dtype)
            final_particles_sum = tf.reduce_sum(particle_outputs.final_particles)
        particles_connected, particles_finite, particles_gradient = _gradient_parts_for_target(
            particles_tape,
            final_particles_sum,
            theta,
            dtype,
        )

        separated_arm = (
            value_outputs.value,
            loglik_outputs.log_likelihood,
            prior,
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
            value_outputs.route_invocations,
            value_outputs.active_resampling_mask_count,
            value_outputs.active_resampling_batch_entries,
            value_outputs.route_outputs_finite,
        )
        return same_arm + same_metrics + separated_arm

    return compiled


def _connected_jvp_parts(jvp: tf.Tensor | None, value: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    connected = tf.constant(jvp is not None)
    if jvp is None:
        jvp = tf.fill(tf.shape(value), tf.constant(float("nan"), dtype=value.dtype))
    return connected, tf.cast(jvp, value.dtype)


def _compiled_checkpoint_jvp_function(
    fixture: lgssm_gate.LGSSMGateFixture,
    base_tensors: dict[str, Any],
    args: argparse.Namespace,
    dtype: tf.DType,
):
    @tf.function(jit_compile=args.jit_compile, reduce_retracing=True)
    def compiled(theta: tf.Tensor) -> tuple[tf.Tensor, ...]:
        theta = tf.cast(theta, dtype)
        tangent0 = tf.stack([tf.constant(1.0, dtype=dtype), tf.constant(0.0, dtype=dtype)])
        tangent1 = tf.stack([tf.constant(0.0, dtype=dtype), tf.constant(1.0, dtype=dtype)])
        with tf.autodiff.ForwardAccumulator(theta, tangent0) as acc0:
            outputs0 = _route_internal_outputs(fixture, base_tensors, theta, args, dtype)
        with tf.autodiff.ForwardAccumulator(theta, tangent1) as acc1:
            outputs1 = _route_internal_outputs(fixture, base_tensors, theta, args, dtype)
        parts: tuple[tf.Tensor, ...] = ()
        for name in REQUIRED_TENSORS:
            value = tf.cast(outputs0.checkpoints[name], dtype)
            conn0, jvp0 = _connected_jvp_parts(acc0.jvp(outputs0.checkpoints[name]), value)
            conn1, jvp1 = _connected_jvp_parts(acc1.jvp(outputs1.checkpoints[name]), value)
            parts += (value, conn0, jvp0, conn1, jvp1)
        return parts

    return compiled


def _compiled_checkpoint_reverse_function(
    fixture: lgssm_gate.LGSSMGateFixture,
    base_tensors: dict[str, Any],
    args: argparse.Namespace,
    dtype: tf.DType,
    checkpoint_name: str,
):
    @tf.function(jit_compile=args.jit_compile, reduce_retracing=True)
    def compiled(theta: tf.Tensor) -> tuple[tf.Tensor, ...]:
        theta = tf.cast(theta, dtype)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(theta)
            outputs = _route_internal_outputs(fixture, base_tensors, theta, args, dtype)
            tensor = outputs.checkpoints[checkpoint_name]
        return _checkpoint_gradient_outputs(tape, tensor, theta, dtype)

    return compiled


def _checkpoint_scalar_metadata(tensor: tf.Tensor, dtype: tf.DType) -> tuple[tf.Tensor, ...]:
    flat = tf.reshape(tf.cast(tensor, dtype), [-1])
    size = tf.size(flat)
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
    last = tf.maximum(size - 1, 0)
    selected_indices = tf.stack([tf.constant(0, dtype=tf.int32), size // 2, last])
    block = tf.maximum(tf.constant(1, dtype=tf.int32), tf.minimum(size, tf.maximum(tf.constant(1, dtype=tf.int32), size // 4)))
    block_starts = tf.stack(
        [
            tf.constant(0, dtype=tf.int32),
            tf.maximum(tf.constant(0, dtype=tf.int32), size // 2 - block // 2),
            tf.maximum(tf.constant(0, dtype=tf.int32), size - block),
        ]
    )
    block_ends = tf.minimum(size, block_starts + block)
    scalars = tf.stack(
        [
            tf.reduce_sum(flat),
            tf.gather(flat, selected_indices[0]),
            tf.gather(flat, selected_indices[1]),
            tf.gather(flat, selected_indices[2]),
            tf.reduce_sum(flat[block_starts[0] : block_ends[0]]),
            tf.reduce_sum(flat[block_starts[1] : block_ends[1]]),
            tf.reduce_sum(flat[block_starts[2] : block_ends[2]]),
        ]
    )
    return size, finite_count, value_min, value_max, value_mean, selected_indices, block_starts, block_ends, scalars


def _compiled_checkpoint_jacobian_function(
    fixture: lgssm_gate.LGSSMGateFixture,
    base_tensors: dict[str, Any],
    args: argparse.Namespace,
    dtype: tf.DType,
):
    @tf.function(jit_compile=args.jit_compile, reduce_retracing=True)
    def compiled(theta: tf.Tensor) -> tuple[tf.Tensor, ...]:
        theta = tf.cast(theta, dtype)
        with tf.GradientTape() as tape:
            tape.watch(theta)
            outputs = _route_internal_outputs(fixture, base_tensors, theta, args, dtype)
            metadata = [_checkpoint_scalar_metadata(outputs.checkpoints[name], dtype) for name in REQUIRED_TENSORS]
            scalar_vector = tf.concat([item[8] for item in metadata], axis=0)
        jacobian = tape.jacobian(
            scalar_vector,
            theta,
            unconnected_gradients=tf.UnconnectedGradients.ZERO,
            experimental_use_pfor=False,
        )
        connected = tf.constant(jacobian is not None)
        if jacobian is None:
            jacobian = tf.fill([tf.shape(scalar_vector)[0], tf.shape(theta)[0]], tf.constant(float("nan"), dtype=dtype))
        return (
            scalar_vector,
            tf.cast(jacobian, dtype),
            connected,
            tf.stack([item[0] for item in metadata]),
            tf.stack([item[1] for item in metadata]),
            tf.stack([item[2] for item in metadata]),
            tf.stack([item[3] for item in metadata]),
            tf.stack([item[4] for item in metadata]),
            tf.stack([item[5] for item in metadata]),
            tf.stack([item[6] for item in metadata]),
            tf.stack([item[7] for item in metadata]),
        )

    return compiled


def _tensor_value_summary(tensor: tf.Tensor) -> dict[str, Any]:
    flat = tf.reshape(tf.cast(tensor, tf.float64), [-1])
    finite = tf.math.is_finite(flat)
    finite_values = tf.boolean_mask(flat, finite)
    count = int(tf.size(flat).numpy())
    finite_count = int(tf.reduce_sum(tf.cast(finite, tf.int32)).numpy())
    if finite_count:
        min_value = float(tf.reduce_min(finite_values).numpy())
        max_value = float(tf.reduce_max(finite_values).numpy())
        mean_value = float(tf.reduce_mean(finite_values).numpy())
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


def _selected_indices(tensor: tf.Tensor) -> list[int]:
    size = int(tf.size(tensor).numpy())
    if size <= 0:
        return []
    raw = [0, size // 2, size - 1]
    selected = []
    for index in raw:
        if index not in selected:
            selected.append(index)
    return selected


def _gradient_summary(gradient: tf.Tensor | None, theta: tf.Tensor) -> dict[str, Any]:
    if gradient is None:
        return {
            "connected": False,
            "finite": False,
            "finite_components": [False for _ in range(int(tf.size(theta).numpy()))],
            "gradient": None,
            "norm": None,
        }
    finite = tf.math.is_finite(gradient)
    return {
        "connected": True,
        "finite": bool(tf.reduce_all(finite).numpy()),
        "finite_components": [bool(item) for item in tf.reshape(finite, [-1]).numpy().tolist()],
        "gradient": _to_json_vector(gradient),
        "norm": float(tf.linalg.norm(tf.cast(gradient, tf.float64)).numpy()),
    }


def _block_slices(size: int) -> list[tuple[str, int, int]]:
    if size <= 0:
        return []
    block = max(1, min(size, size // 4 or 1))
    starts = [0, max(0, size // 2 - block // 2), max(0, size - block)]
    slices = []
    used = set()
    for label, start in zip(("leading", "middle", "trailing"), starts, strict=True):
        end = min(size, start + block)
        key = (start, end)
        if key in used:
            continue
        used.add(key)
        slices.append((label, start, end))
    return slices


def _tensor_gradient_diagnostics(tape: tf.GradientTape, tensor: tf.Tensor, theta: tf.Tensor) -> dict[str, Any]:
    tensor = tf.cast(tensor, theta.dtype)
    flat = tf.reshape(tensor, [-1])
    whole_grad = tape.gradient(tf.reduce_sum(flat), theta)
    selected = []
    for index in _selected_indices(tensor):
        grad = tape.gradient(flat[index], theta)
        selected.append(
            {
                "index": int(index),
                "value": _to_json_scalar(flat[index]),
                "gradient": _gradient_summary(grad, theta),
            }
        )
    blocks = []
    for label, start, end in _block_slices(int(tf.size(flat).numpy())):
        grad = tape.gradient(tf.reduce_sum(flat[start:end]), theta)
        blocks.append(
            {
                "label": label,
                "start": int(start),
                "end": int(end),
                "gradient": _gradient_summary(grad, theta),
            }
        )
    return {
        "value_summary": _tensor_value_summary(tensor),
        "whole_sum_gradient": _gradient_summary(whole_grad, theta),
        "selected_scalar_gradients": selected,
        "block_sum_gradients": blocks,
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


def _checkpoint_summary_from_parts(parts: tuple[tf.Tensor, ...]) -> dict[str, Any]:
    (
        size,
        finite_count,
        value_min,
        value_max,
        value_mean,
        selected_indices,
        selected_values,
        whole_connected,
        whole_finite,
        whole_gradient,
        selected_connected,
        selected_finite,
        selected_gradients,
        block_starts,
        block_ends,
        block_connected,
        block_finite,
        block_gradients,
    ) = parts
    count = _to_json_int(size)
    finite_count_int = _to_json_int(finite_count)
    selected = []
    for index in range(3):
        selected.append(
            {
                "index": int(tf.reshape(selected_indices, [-1])[index].numpy()),
                "value": float(tf.reshape(tf.cast(selected_values, tf.float64), [-1])[index].numpy()),
                "gradient": _gradient_summary_from_parts(
                    tf.reshape(selected_connected, [-1])[index],
                    tf.reshape(selected_finite, [-1])[index],
                    tf.reshape(selected_gradients[index], [-1]),
                ),
            }
        )
    blocks = []
    for index, label in enumerate(("leading", "middle", "trailing")):
        blocks.append(
            {
                "label": label,
                "start": int(tf.reshape(block_starts, [-1])[index].numpy()),
                "end": int(tf.reshape(block_ends, [-1])[index].numpy()),
                "gradient": _gradient_summary_from_parts(
                    tf.reshape(block_connected, [-1])[index],
                    tf.reshape(block_finite, [-1])[index],
                    tf.reshape(block_gradients[index], [-1]),
                ),
            }
        )
    return {
        "value_summary": {
            "shape": None,
            "count": count,
            "finite_count": finite_count_int,
            "nonfinite_count": count - finite_count_int,
            "min": _to_json_scalar(value_min),
            "max": _to_json_scalar(value_max),
            "mean": _to_json_scalar(value_mean),
            "preview": [float(item) for item in tf.reshape(tf.cast(selected_values, tf.float64), [-1]).numpy().tolist()],
        },
        "whole_sum_gradient": _gradient_summary_from_parts(whole_connected, whole_finite, whole_gradient),
        "selected_scalar_gradients": selected,
        "block_sum_gradients": blocks,
    }


def _same_tape_row_from_compiled(values: tuple[tf.Tensor, ...]) -> dict[str, Any]:
    checkpoints_start = SAME_BASE_COUNT
    checkpoint_diagnostics = {}
    for index, name in enumerate(REQUIRED_TENSORS):
        start = checkpoints_start + index * CHECKPOINT_OUTPUT_COUNT
        end = start + CHECKPOINT_OUTPUT_COUNT
        checkpoint_diagnostics[name] = _checkpoint_summary_from_parts(values[start:end])
    return {
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
        "required_checkpoint_names": list(REQUIRED_TENSORS),
        "missing_required_checkpoints": [],
        "checkpoints": checkpoint_diagnostics,
    }


def _separated_tape_row_from_compiled(values: tuple[tf.Tensor, ...]) -> dict[str, Any]:
    separated_start = SAME_BASE_COUNT + len(REQUIRED_TENSORS) * CHECKPOINT_OUTPUT_COUNT
    parts = values[separated_start : separated_start + SEPARATED_BASE_COUNT]
    return {
        "mode": "p02a_style_separated_tape",
        "output_device": parts[0].device,
        "readout_pattern": "separate route execution per value/loglik/final-particle readout plus separate prior tape",
        "behavioral_equivalence_note": (
            "Matches P02A separated-tape structure for value, likelihood, prior, and final-particle "
            "sum gradients while using the P02B instrumented route loop."
        ),
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


def _same_tape_primary_from_compiled(values: tuple[tf.Tensor, ...], checkpoints: dict[str, Any]) -> dict[str, Any]:
    return {
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
        "required_checkpoint_names": list(REQUIRED_TENSORS),
        "missing_required_checkpoints": [],
        "checkpoints": checkpoints,
        "checkpoint_gradient_mode": "compiled_reverse_mode_per_checkpoint",
    }


def _separated_tape_primary_from_compiled(values: tuple[tf.Tensor, ...]) -> dict[str, Any]:
    parts = values[SAME_BASE_COUNT : SAME_BASE_COUNT + SEPARATED_BASE_COUNT]
    return {
        "mode": "p02a_style_separated_tape",
        "output_device": parts[0].device,
        "readout_pattern": "separate route execution per value/loglik/final-particle readout plus separate prior tape",
        "behavioral_equivalence_note": (
            "Matches P02A separated-tape structure for value, likelihood, prior, and final-particle "
            "sum gradients while using the P02B instrumented route loop."
        ),
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


def _jvp_gradient_summary(conn0: tf.Tensor, jvp0: tf.Tensor, conn1: tf.Tensor, jvp1: tf.Tensor) -> dict[str, Any]:
    gradient = tf.stack([tf.reshape(tf.cast(jvp0, tf.float64), [])[0], tf.reshape(tf.cast(jvp1, tf.float64), [])[0]])
    finite_components = tf.math.is_finite(gradient)
    connected = _to_json_bool(conn0) and _to_json_bool(conn1)
    finite = bool(tf.reduce_all(finite_components).numpy()) if connected else False
    return {
        "connected": connected,
        "finite": finite,
        "finite_components": [bool(item) for item in finite_components.numpy().tolist()],
        "gradient": [float(item) for item in gradient.numpy().tolist()],
        "norm": float(tf.linalg.norm(gradient).numpy()) if connected else None,
        "mode": "forward_jvp",
    }


def _checkpoint_diagnostic_from_jvp(value: tf.Tensor, conn0: tf.Tensor, jvp0: tf.Tensor, conn1: tf.Tensor, jvp1: tf.Tensor) -> dict[str, Any]:
    flat_value = tf.reshape(tf.cast(value, tf.float64), [-1])
    flat_jvp0 = tf.reshape(tf.cast(jvp0, tf.float64), [-1])
    flat_jvp1 = tf.reshape(tf.cast(jvp1, tf.float64), [-1])
    count = int(tf.size(flat_value).numpy())
    finite = tf.math.is_finite(flat_value)
    finite_count = int(tf.reduce_sum(tf.cast(finite, tf.int32)).numpy())
    finite_values = tf.boolean_mask(flat_value, finite)
    if finite_count:
        value_min = float(tf.reduce_min(finite_values).numpy())
        value_max = float(tf.reduce_max(finite_values).numpy())
        value_mean = float(tf.reduce_mean(finite_values).numpy())
    else:
        value_min = None
        value_max = None
        value_mean = None
    whole = _jvp_gradient_summary(conn0, tf.reduce_sum(flat_jvp0), conn1, tf.reduce_sum(flat_jvp1))
    selected = []
    for index in _selected_indices(value):
        selected.append(
            {
                "index": int(index),
                "value": float(flat_value[index].numpy()),
                "gradient": _jvp_gradient_summary(conn0, flat_jvp0[index], conn1, flat_jvp1[index]),
            }
        )
    blocks = []
    for label, start, end in _block_slices(count):
        blocks.append(
            {
                "label": label,
                "start": int(start),
                "end": int(end),
                "gradient": _jvp_gradient_summary(
                    conn0,
                    tf.reduce_sum(flat_jvp0[start:end]),
                    conn1,
                    tf.reduce_sum(flat_jvp1[start:end]),
                ),
            }
        )
    return {
        "value_summary": {
            "shape": value.shape.as_list(),
            "count": count,
            "finite_count": finite_count,
            "nonfinite_count": count - finite_count,
            "min": value_min,
            "max": value_max,
            "mean": value_mean,
            "preview": [float(item) for item in flat_value[: min(6, count)].numpy().tolist()],
        },
        "whole_sum_gradient": whole,
        "selected_scalar_gradients": selected,
        "block_sum_gradients": blocks,
    }


def _checkpoint_rows_from_jvp(values: tuple[tf.Tensor, ...]) -> dict[str, Any]:
    checkpoints = {}
    width = 5
    for index, name in enumerate(REQUIRED_TENSORS):
        start = index * width
        value, conn0, jvp0, conn1, jvp1 = values[start : start + width]
        checkpoints[name] = _checkpoint_diagnostic_from_jvp(value, conn0, jvp0, conn1, jvp1)
    return checkpoints


def _jacobian_gradient_summary(row: tf.Tensor, connected: tf.Tensor) -> dict[str, Any]:
    gradient = tf.reshape(tf.cast(row, tf.float64), [-1])
    finite_components = tf.math.is_finite(gradient)
    connected_bool = _to_json_bool(connected)
    finite = bool(tf.reduce_all(finite_components).numpy()) if connected_bool else False
    return {
        "connected": connected_bool,
        "finite": finite,
        "finite_components": [bool(item) for item in finite_components.numpy().tolist()],
        "gradient": [float(item) for item in gradient.numpy().tolist()],
        "norm": float(tf.linalg.norm(gradient).numpy()) if connected_bool else None,
        "mode": "compiled_reverse_jacobian",
        "unconnected_gradients_policy": "ZERO",
    }


def _checkpoint_rows_from_jacobian(values: tuple[tf.Tensor, ...]) -> dict[str, Any]:
    (
        scalar_vector,
        jacobian,
        connected,
        sizes,
        finite_counts,
        value_mins,
        value_maxes,
        value_means,
        selected_indices,
        block_starts,
        block_ends,
    ) = values
    scalars = tf.reshape(tf.cast(scalar_vector, tf.float64), [-1])
    jac = tf.cast(jacobian, tf.float64)
    checkpoints = {}
    for checkpoint_index, name in enumerate(REQUIRED_TENSORS):
        base = checkpoint_index * 7
        selected = []
        for offset in range(3):
            selected.append(
                {
                    "index": int(selected_indices[checkpoint_index, offset].numpy()),
                    "value": float(scalars[base + 1 + offset].numpy()),
                    "gradient": _jacobian_gradient_summary(jac[base + 1 + offset], connected),
                }
            )
        blocks = []
        for offset, label in enumerate(("leading", "middle", "trailing")):
            blocks.append(
                {
                    "label": label,
                    "start": int(block_starts[checkpoint_index, offset].numpy()),
                    "end": int(block_ends[checkpoint_index, offset].numpy()),
                    "gradient": _jacobian_gradient_summary(jac[base + 4 + offset], connected),
                }
            )
        count = int(sizes[checkpoint_index].numpy())
        finite_count = int(finite_counts[checkpoint_index].numpy())
        checkpoints[name] = {
            "value_summary": {
                "shape": None,
                "count": count,
                "finite_count": finite_count,
                "nonfinite_count": count - finite_count,
                "min": float(value_mins[checkpoint_index].numpy()),
                "max": float(value_maxes[checkpoint_index].numpy()),
                "mean": float(value_means[checkpoint_index].numpy()),
                "preview": [float(item["value"]) for item in selected],
            },
            "whole_sum_gradient": _jacobian_gradient_summary(jac[base], connected),
            "selected_scalar_gradients": selected,
            "block_sum_gradients": blocks,
        }
    return checkpoints


def _scaled_centered_particles_for_diagnostic(x: tf.Tensor) -> tf.Tensor:
    dtype = x.dtype
    centered = x - tf.stop_gradient(tf.reduce_mean(x, axis=1, keepdims=True))
    std = tf.math.reduce_std(x, axis=1)
    scale = tf.reduce_max(std, axis=1)
    scale = tf.where(scale == 0.0, tf.ones_like(scale), scale)
    dim = tf.cast(tf.shape(x)[2], dtype)
    return centered / tf.stop_gradient(scale[:, None, None] * tf.sqrt(dim))


def _deterministic_landmark_indices(num_particles: tf.Tensor, rank: int, dtype: tf.DType) -> tf.Tensor:
    if rank == 1:
        return tf.zeros([1], dtype=tf.int32)
    end = tf.cast(num_particles - 1, dtype)
    return tf.cast(tf.round(tf.linspace(tf.constant(0.0, dtype), end, rank)), tf.int32)


def _assignment_kernel(x: tf.Tensor, latent: tf.Tensor, epsilon: tf.Tensor, denominator_floor: float) -> tf.Tensor:
    dtype = x.dtype
    xx = tf.reduce_sum(x * x, axis=2, keepdims=True)
    xy = tf.matmul(x, latent, transpose_b=True)
    yy = tf.expand_dims(tf.reduce_sum(latent * latent, axis=-1), axis=1)
    cost = 0.5 * tf.clip_by_value(xx - 2.0 * xy + yy, 0.0, float("inf"))
    logits = -cost / tf.cast(epsilon, dtype)
    logits = logits - tf.reduce_max(logits, axis=2, keepdims=True)
    return tf.maximum(tf.exp(logits), tf.constant(denominator_floor, dtype=dtype))


def _low_rank_inputs_for_diagnostic(
    post_flow: tf.Tensor,
    normalized_log_weights: tf.Tensor,
    args: argparse.Namespace,
) -> dict[str, tf.Tensor]:
    dtype = post_flow.dtype
    num_particles = tf.shape(post_flow)[1]
    rank = int(args.low_rank_rank)
    source_weights = tf.exp(normalized_log_weights)
    source_weights = source_weights / tf.reduce_sum(source_weights, axis=1, keepdims=True)
    rank_tensor = tf.cast(rank, dtype)
    scaled_x = _scaled_centered_particles_for_diagnostic(post_flow)
    latent = tf.gather(scaled_x, _deterministic_landmark_indices(num_particles, rank, dtype), axis=1)
    eps_q = _assignment_kernel(
        scaled_x,
        latent,
        tf.constant(float(args.low_rank_assignment_epsilon), dtype=dtype),
        args.low_rank_denominator_floor,
    )
    eps_r = _assignment_kernel(
        -scaled_x,
        latent,
        tf.constant(float(args.low_rank_assignment_epsilon), dtype=dtype),
        args.low_rank_denominator_floor,
    )
    eps_g = tf.maximum(
        tf.ones([tf.shape(post_flow)[0], rank], dtype=dtype) / rank_tensor,
        tf.constant(float(args.low_rank_alpha), dtype=dtype),
    )
    return {
        "source_weights": source_weights,
        "scaled_x": scaled_x,
        "eps_q": eps_q,
        "eps_r": eps_r,
        "eps_g": eps_g,
    }


def _route_internal_outputs(
    fixture: lgssm_gate.LGSSMGateFixture,
    base_tensors: dict[str, Any],
    theta: tf.Tensor,
    args: argparse.Namespace,
    dtype: tf.DType,
) -> RouteInternalOutputs:
    theta = tf.cast(theta, dtype)
    scaled_fixture = calibration._scaled_fixture(fixture, theta, dtype)
    tensors = dict(base_tensors)
    tensors["transition_covariance"] = tf.cast(scaled_fixture.Q[None, :, :], dtype)
    tensors["observation_covariance"] = tf.cast(scaled_fixture.R[None, :, :], dtype)
    callbacks = lgssm_gate._callbacks(tensors, dtype)
    observations = tensors["observations"]
    particles = tensors["initial_particles"]
    fixed_mask = tensors["fixed_resampling_mask"]
    transition_matrix = tensors["transition_matrix"]
    transition_covariance = tensors["transition_covariance"]
    observation_covariance = tensors["observation_covariance"]
    batch_size = int(particles.shape[0])
    num_particles = int(particles.shape[1])
    state_dim = int(particles.shape[2])
    time_steps = int(observations.shape[0])
    log_weights = core_tf.uniform_log_weights(batch_size, num_particles)
    log_likelihood = tf.zeros([batch_size], dtype=dtype)
    active_steps, active_entries = lgssm_gate._active_step_counts_tensor(fixed_mask)
    route_invocations = tf.constant(0, dtype=tf.int32)
    max_factor = tf.constant(0.0, dtype=dtype)
    max_row = tf.constant(0.0, dtype=dtype)
    max_col = tf.constant(0.0, dtype=dtype)
    max_iter = tf.constant(0, dtype=tf.int32)
    finite_factors = tf.constant(True)
    finite_particles = tf.constant(True)
    nonnegative_factors = tf.constant(True)
    positive_g = tf.constant(True)
    outputs_finite = tf.constant(True)
    checkpoints: dict[str, tf.Tensor] = {
        "scaled_Q": scaled_fixture.Q,
        "scaled_R": scaled_fixture.R,
    }

    time_steps_tensor = tf.constant(time_steps, dtype=tf.int32)
    zero_particles = tf.zeros_like(particles)
    zero_log_weights = tf.zeros_like(log_weights)
    zero_batch = tf.zeros_like(log_likelihood)
    zero_factors = tf.zeros([batch_size, num_particles, int(args.low_rank_rank)], dtype=dtype)
    zero_rank_weights = tf.zeros([batch_size, int(args.low_rank_rank)], dtype=dtype)

    def route_cond(loop_state: tuple[tf.Tensor, ...], _checkpoint_state: tuple[tf.Tensor, ...]) -> tf.Tensor:
        return loop_state[0] < time_steps_tensor

    def route_body(
        loop_state: tuple[tf.Tensor, ...],
        checkpoint_state: tuple[tf.Tensor, ...],
    ) -> tuple[tuple[tf.Tensor, ...], tuple[tf.Tensor, ...]]:
        (
            time_index,
            particles,
            log_weights,
            log_likelihood,
            route_invocations,
            max_factor,
            max_row,
            max_col,
            max_iter,
            finite_factors,
            finite_particles,
            nonnegative_factors,
            positive_g,
            outputs_finite,
        ) = loop_state

        def capture_at(index: int, new_value: tf.Tensor, old_value: tf.Tensor) -> tf.Tensor:
            return tf.cond(tf.equal(time_index, index), lambda: new_value, lambda: old_value)

        (
            pre_flow_t0,
            post_flow_t0,
            pre_flow_log_density_t0,
            forward_log_det_t0,
            transition_log_density_t0,
            observation_log_density_t0,
            corrected_log_weights_t0,
            normalized_weights_t0,
            incremental_log_likelihood_t0,
            normalized_log_weights_t0,
            scaled_x_t0,
            eps_q_t0,
            eps_r_t0,
            eps_g_t0,
            q_factor_t0,
            r_factor_t0,
            g_weights_t0,
            transported_particles_t0,
            resampled_log_weights_t0,
            next_pre_flow_t1,
            next_corrected_log_weights_t1,
            next_incremental_log_likelihood_t1,
        ) = checkpoint_state
        observation = observations[time_index]
        ancestors = particles
        pre_flow = callbacks["pre_flow_step_fn"](particles, time_index)
        step_prior_mean_fn = lambda points: callbacks["prior_mean_fn"](points, time_index)
        flow = streaming_tf.batched_ledh_flow_streaming_particles_tf(
            pre_flow_particles=pre_flow,
            ancestors=ancestors,
            observation=observation,
            transition_matrix=transition_matrix,
            transition_covariance=transition_covariance,
            observation_covariance=observation_covariance,
            observation_fn=callbacks["observation_fn"],
            observation_jacobian_fn=callbacks["observation_jacobian_fn"],
            observation_residual_fn=callbacks["observation_residual_fn"],
            prior_mean_fn=step_prior_mean_fn,
            particle_chunk_size=args.particle_chunk_size,
        )
        post_flow = flow.post_flow_particles
        transition_log_density = callbacks["transition_log_density_fn"](post_flow, ancestors, time_index)
        observation_log_density = callbacks["observation_log_density_fn"](post_flow, observation, time_index)
        corrected_log_weights = (
            log_weights
            + transition_log_density
            + observation_log_density
            - flow.pre_flow_log_density
            + flow.forward_log_det
        )
        weights, incremental = lgssm_gate._normalize_log_weights(corrected_log_weights)
        log_likelihood += incremental
        normalized_log_weights = tf.math.log(tf.maximum(weights, lgssm_gate._log_weight_floor()))
        lr_inputs = _low_rank_inputs_for_diagnostic(post_flow, normalized_log_weights, args)
        resampled = low_rank_coupling_solver_resample_tensors_tf(
            post_flow,
            normalized_log_weights,
            rank=args.low_rank_rank,
            assignment_epsilon=args.low_rank_assignment_epsilon,
            alpha=args.low_rank_alpha,
            max_projection_iterations=args.low_rank_max_projection_iterations,
            convergence_threshold=args.low_rank_convergence_threshold,
            denominator_floor=args.low_rank_denominator_floor,
        )
        transported = tf.cast(resampled.particles, dtype)
        resampled_log_weights = tf.cast(resampled.log_weights, dtype)
        checkpoint_state = (
            capture_at(0, pre_flow, pre_flow_t0),
            capture_at(0, post_flow, post_flow_t0),
            capture_at(0, flow.pre_flow_log_density, pre_flow_log_density_t0),
            capture_at(0, flow.forward_log_det, forward_log_det_t0),
            capture_at(0, transition_log_density, transition_log_density_t0),
            capture_at(0, observation_log_density, observation_log_density_t0),
            capture_at(0, corrected_log_weights, corrected_log_weights_t0),
            capture_at(0, weights, normalized_weights_t0),
            capture_at(0, incremental, incremental_log_likelihood_t0),
            capture_at(0, normalized_log_weights, normalized_log_weights_t0),
            capture_at(0, lr_inputs["scaled_x"], scaled_x_t0),
            capture_at(0, lr_inputs["eps_q"], eps_q_t0),
            capture_at(0, lr_inputs["eps_r"], eps_r_t0),
            capture_at(0, lr_inputs["eps_g"], eps_g_t0),
            capture_at(0, tf.cast(resampled.q_factor, dtype), q_factor_t0),
            capture_at(0, tf.cast(resampled.r_factor, dtype), r_factor_t0),
            capture_at(0, tf.cast(resampled.g_weights, dtype), g_weights_t0),
            capture_at(0, transported, transported_particles_t0),
            capture_at(0, resampled_log_weights, resampled_log_weights_t0),
            capture_at(1, pre_flow, next_pre_flow_t1),
            capture_at(1, corrected_log_weights, next_corrected_log_weights_t1),
            capture_at(1, incremental, next_incremental_log_likelihood_t1),
        )
        max_factor = tf.maximum(max_factor, tf.cast(resampled.max_factor_marginal_residual, dtype))
        max_row = tf.maximum(max_row, tf.cast(resampled.max_induced_row_residual, dtype))
        max_col = tf.maximum(max_col, tf.cast(resampled.max_induced_column_residual, dtype))
        max_iter = tf.maximum(max_iter, resampled.projection_iterations_used)
        finite_factors = tf.logical_and(finite_factors, resampled.finite_factors)
        finite_particles = tf.logical_and(finite_particles, resampled.finite_particles)
        nonnegative_factors = tf.logical_and(nonnegative_factors, resampled.nonnegative_factors)
        positive_g = tf.logical_and(positive_g, resampled.positive_g)
        route_invocations += tf.constant(1, dtype=tf.int32)
        outputs_finite = (
            outputs_finite
            & tf.reduce_all(tf.math.is_finite(post_flow))
            & tf.reduce_all(tf.math.is_finite(corrected_log_weights))
            & tf.reduce_all(tf.math.is_finite(transported))
            & tf.reduce_all(tf.math.is_finite(resampled_log_weights))
        )
        loop_state = (
            time_index + tf.constant(1, dtype=tf.int32),
            transported,
            resampled_log_weights,
            log_likelihood,
            route_invocations,
            max_factor,
            max_row,
            max_col,
            max_iter,
            finite_factors,
            finite_particles,
            nonnegative_factors,
            positive_g,
            outputs_finite,
        )
        return loop_state, checkpoint_state

    initial_loop_state = (
        tf.constant(0, dtype=tf.int32),
        particles,
        log_weights,
        log_likelihood,
        route_invocations,
        max_factor,
        max_row,
        max_col,
        max_iter,
        finite_factors,
        finite_particles,
        nonnegative_factors,
        positive_g,
        outputs_finite,
    )
    initial_checkpoint_state = (
        zero_particles,
        zero_particles,
        zero_log_weights,
        zero_log_weights,
        zero_log_weights,
        zero_log_weights,
        zero_log_weights,
        zero_log_weights,
        zero_batch,
        zero_log_weights,
        zero_particles,
        zero_factors,
        zero_factors,
        zero_rank_weights,
        zero_factors,
        zero_factors,
        zero_rank_weights,
        zero_particles,
        zero_log_weights,
        zero_particles,
        zero_log_weights,
        zero_batch,
    )
    (
        (
            _,
            particles,
            log_weights,
            log_likelihood,
            route_invocations,
            max_factor,
            max_row,
            max_col,
            max_iter,
            finite_factors,
            finite_particles,
            nonnegative_factors,
            positive_g,
            outputs_finite,
        ),
        (
            pre_flow_t0,
            post_flow_t0,
            pre_flow_log_density_t0,
            forward_log_det_t0,
            transition_log_density_t0,
            observation_log_density_t0,
            corrected_log_weights_t0,
            normalized_weights_t0,
            incremental_log_likelihood_t0,
            normalized_log_weights_t0,
            scaled_x_t0,
            eps_q_t0,
            eps_r_t0,
            eps_g_t0,
            q_factor_t0,
            r_factor_t0,
            g_weights_t0,
            transported_particles_t0,
            resampled_log_weights_t0,
            next_pre_flow_t1,
            next_corrected_log_weights_t1,
            next_incremental_log_likelihood_t1,
        ),
    ) = tf.while_loop(
        route_cond,
        route_body,
        loop_vars=(initial_loop_state, initial_checkpoint_state),
        parallel_iterations=1,
        maximum_iterations=time_steps,
    )

    final_log_likelihood = tf.reshape(tf.cast(log_likelihood, dtype), [-1])[0]
    prior = calibration._theta_prior_log_density(theta, args.theta_prior_scale, dtype)
    value = final_log_likelihood + prior
    checkpoints.update(
        {
            "pre_flow_t0": pre_flow_t0,
            "post_flow_t0": post_flow_t0,
            "pre_flow_log_density_t0": pre_flow_log_density_t0,
            "forward_log_det_t0": forward_log_det_t0,
            "transition_log_density_t0": transition_log_density_t0,
            "observation_log_density_t0": observation_log_density_t0,
            "corrected_log_weights_t0": corrected_log_weights_t0,
            "normalized_weights_t0": normalized_weights_t0,
            "incremental_log_likelihood_t0": incremental_log_likelihood_t0,
            "normalized_log_weights_t0": normalized_log_weights_t0,
            "scaled_x_t0": scaled_x_t0,
            "eps_q_t0": eps_q_t0,
            "eps_r_t0": eps_r_t0,
            "eps_g_t0": eps_g_t0,
            "q_factor_t0": q_factor_t0,
            "r_factor_t0": r_factor_t0,
            "g_weights_t0": g_weights_t0,
            "transported_particles_t0": transported_particles_t0,
            "resampled_log_weights_t0": resampled_log_weights_t0,
            "next_pre_flow_t1": next_pre_flow_t1,
            "next_corrected_log_weights_t1": next_corrected_log_weights_t1,
            "next_incremental_log_likelihood_t1": next_incremental_log_likelihood_t1,
        }
    )
    checkpoints["final_particles"] = particles
    checkpoints["final_log_likelihood"] = final_log_likelihood
    outputs_finite = (
        outputs_finite
        & tf.reduce_all(tf.math.is_finite(log_likelihood))
        & tf.reduce_all(tf.math.is_finite(particles))
        & tf.reduce_all(tf.math.is_finite(log_weights))
    )
    return RouteInternalOutputs(
        log_likelihood=final_log_likelihood,
        value=value,
        prior=prior,
        final_particles=particles,
        route_invocations=route_invocations,
        active_resampling_mask_count=active_steps,
        active_resampling_batch_entries=active_entries,
        max_factor_marginal_residual=max_factor,
        max_induced_row_residual=max_row,
        max_induced_column_residual=max_col,
        projection_iterations_used_max=max_iter,
        finite_factors=finite_factors,
        finite_particles=finite_particles,
        nonnegative_factors=nonnegative_factors,
        positive_g=positive_g,
        route_outputs_finite=outputs_finite,
        checkpoints=checkpoints,
    )


def _same_tape_row(
    compiled_route: Any,
    theta: tf.Tensor,
) -> dict[str, Any]:
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(theta)
        outputs = _unpack_internal_tuple(compiled_route(theta))
        value = outputs.value
        log_likelihood = outputs.log_likelihood
        prior = outputs.prior
        final_particles_sum = tf.reduce_sum(outputs.final_particles)
    value_gradient = tape.gradient(value, theta)
    loglik_gradient = tape.gradient(log_likelihood, theta)
    prior_gradient = tape.gradient(prior, theta)
    final_particles_gradient = tape.gradient(final_particles_sum, theta)
    checkpoint_diagnostics = {
        name: _tensor_gradient_diagnostics(tape, outputs.checkpoints[name], theta)
        for name in REQUIRED_TENSORS
        if name in outputs.checkpoints
    }
    missing = [name for name in REQUIRED_TENSORS if name not in checkpoint_diagnostics]
    del tape
    return {
        "mode": "same_tape_internal",
        "output_device": value.device,
        "value": _to_json_scalar(value),
        "log_likelihood": _to_json_scalar(log_likelihood),
        "prior": _to_json_scalar(prior),
        "value_gradient": _gradient_summary(value_gradient, theta),
        "log_likelihood_gradient": _gradient_summary(loglik_gradient, theta),
        "prior_gradient": _gradient_summary(prior_gradient, theta),
        "final_particles_sum_gradient": _gradient_summary(final_particles_gradient, theta),
        "route_invocations": _to_json_int(outputs.route_invocations),
        "active_resampling_mask_count": _to_json_int(outputs.active_resampling_mask_count),
        "active_resampling_batch_entries": _to_json_int(outputs.active_resampling_batch_entries),
        "max_factor_marginal_residual": _to_json_scalar(outputs.max_factor_marginal_residual),
        "max_induced_row_residual": _to_json_scalar(outputs.max_induced_row_residual),
        "max_induced_column_residual": _to_json_scalar(outputs.max_induced_column_residual),
        "projection_iterations_used_max": _to_json_int(outputs.projection_iterations_used_max),
        "finite_factors": _to_json_bool(outputs.finite_factors),
        "finite_particles": _to_json_bool(outputs.finite_particles),
        "nonnegative_factors": _to_json_bool(outputs.nonnegative_factors),
        "positive_g": _to_json_bool(outputs.positive_g),
        "route_outputs_finite": _to_json_bool(outputs.route_outputs_finite),
        "required_checkpoint_names": list(REQUIRED_TENSORS),
        "missing_required_checkpoints": missing,
        "checkpoints": checkpoint_diagnostics,
    }


def _separated_tape_row(
    compiled_route: Any,
    theta: tf.Tensor,
    args: argparse.Namespace,
    dtype: tf.DType,
) -> dict[str, Any]:
    def route_terms() -> RouteInternalOutputs:
        return _unpack_internal_tuple(compiled_route(theta))

    with tf.GradientTape() as tape:
        tape.watch(theta)
        value_outputs = route_terms()
        value = value_outputs.value
    value_gradient = tape.gradient(value, theta)

    with tf.GradientTape() as tape:
        tape.watch(theta)
        loglik_outputs = route_terms()
        log_likelihood = loglik_outputs.log_likelihood
    loglik_gradient = tape.gradient(log_likelihood, theta)

    with tf.GradientTape() as tape:
        tape.watch(theta)
        prior = calibration._theta_prior_log_density(theta, args.theta_prior_scale, dtype)
    prior_gradient = tape.gradient(prior, theta)

    with tf.GradientTape() as tape:
        tape.watch(theta)
        particle_outputs = route_terms()
        final_particles_sum = tf.reduce_sum(particle_outputs.final_particles)
    final_particles_gradient = tape.gradient(final_particles_sum, theta)

    return {
        "mode": "p02a_style_separated_tape",
        "output_device": value.device,
        "readout_pattern": "separate route execution per value/loglik/final-particle readout plus separate prior tape",
        "behavioral_equivalence_note": (
            "Matches P02A separated-tape structure for value, likelihood, prior, and final-particle "
            "sum gradients while using the P02B instrumented route loop."
        ),
        "value": _to_json_scalar(value),
        "log_likelihood": _to_json_scalar(log_likelihood),
        "prior": _to_json_scalar(prior),
        "value_gradient": _gradient_summary(value_gradient, theta),
        "log_likelihood_gradient": _gradient_summary(loglik_gradient, theta),
        "prior_gradient": _gradient_summary(prior_gradient, theta),
        "final_particles_sum_gradient": _gradient_summary(final_particles_gradient, theta),
        "route_invocations": _to_json_int(value_outputs.route_invocations),
        "active_resampling_mask_count": _to_json_int(value_outputs.active_resampling_mask_count),
        "active_resampling_batch_entries": _to_json_int(value_outputs.active_resampling_batch_entries),
        "route_outputs_finite": _to_json_bool(value_outputs.route_outputs_finite),
    }


def _ab_comparison(same: dict[str, Any], separated: dict[str, Any]) -> dict[str, Any]:
    keys = ("value_gradient", "log_likelihood_gradient", "final_particles_sum_gradient")
    comparison = {}
    same_connected_all = True
    separated_disconnected_any = False
    for key in keys:
        same_grad = same[key]
        sep_grad = separated[key]
        comparison[key] = {
            "same_tape_connected": bool(same_grad["connected"]),
            "same_tape_finite": bool(same_grad["finite"]),
            "separated_tape_connected": bool(sep_grad["connected"]),
            "separated_tape_finite": bool(sep_grad["finite"]),
            "connection_differs": bool(same_grad["connected"]) != bool(sep_grad["connected"]),
        }
        same_connected_all = same_connected_all and bool(same_grad["connected"]) and bool(same_grad["finite"])
        separated_disconnected_any = separated_disconnected_any or not bool(sep_grad["connected"])
    return {
        "fields": comparison,
        "supports_h1_tape_artifact": bool(same_connected_all and separated_disconnected_any),
        "interpretation_guard": (
            "A/B differences isolate the readout pattern only to the extent that the instrumented "
            "route loop remains route-equivalent to P02/P02A."
        ),
    }


def _first_observed_break(same: dict[str, Any]) -> dict[str, Any]:
    for name in REQUIRED_TENSORS:
        diagnostic = same["checkpoints"].get(name)
        if diagnostic is None:
            return {"checkpoint": name, "reason": "missing_required_checkpoint"}
        whole = diagnostic["whole_sum_gradient"]
        if not whole["connected"]:
            return {"checkpoint": name, "reason": "whole_sum_gradient_disconnected"}
        if not whole["finite"]:
            return {"checkpoint": name, "reason": "whole_sum_gradient_nonfinite"}
        for scalar in diagnostic["selected_scalar_gradients"]:
            gradient = scalar["gradient"]
            if not gradient["connected"]:
                return {"checkpoint": name, "reason": f"selected_scalar_{scalar['index']}_disconnected"}
            if not gradient["finite"]:
                return {"checkpoint": name, "reason": f"selected_scalar_{scalar['index']}_nonfinite"}
        for block in diagnostic["block_sum_gradients"]:
            gradient = block["gradient"]
            if not gradient["connected"]:
                return {"checkpoint": name, "reason": f"{block['label']}_block_disconnected"}
            if not gradient["finite"]:
                return {"checkpoint": name, "reason": f"{block['label']}_block_nonfinite"}
    return {"checkpoint": None, "reason": "no_observed_checkpoint_break"}


def _row(
    seed: int,
    label: str,
    theta: tf.Tensor,
    same: dict[str, Any],
    separated: dict[str, Any],
) -> dict[str, Any]:
    return {
        "seed": int(seed),
        "probe_label": label,
        "theta": _to_json_vector(theta),
        "same_tape": same,
        "p02a_style_separated_tape": separated,
        "ab_comparison": _ab_comparison(same, separated),
        "first_observed_checkpoint_break": _first_observed_break(same),
    }


def _artifact_vetoes(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[str]:
    vetoes = []
    for row in rows:
        label = f"{row['seed']}:{row['probe_label']}"
        same = row["same_tape"]
        separated = row["p02a_style_separated_tape"]
        if args.expect_device_kind == "gpu":
            if "GPU" not in same["output_device"].upper():
                vetoes.append(f"{label}:same_tape_not_gpu_output")
            if "GPU" not in separated["output_device"].upper():
                vetoes.append(f"{label}:separated_tape_not_gpu_output")
        if not same["route_outputs_finite"]:
            vetoes.append(f"{label}:same_tape_route_outputs_nonfinite")
        if not separated["route_outputs_finite"]:
            vetoes.append(f"{label}:separated_tape_route_outputs_nonfinite")
        if same["missing_required_checkpoints"]:
            vetoes.append(f"{label}:missing_required_checkpoints:{same['missing_required_checkpoints']}")
        if same["route_invocations"] != same["active_resampling_mask_count"]:
            vetoes.append(f"{label}:unexpected_route_invocation_count")
    return vetoes


def _diagnostic_findings(rows: list[dict[str, Any]]) -> list[str]:
    findings = []
    for row in rows:
        label = f"{row['seed']}:{row['probe_label']}"
        for mode_key in ("same_tape", "p02a_style_separated_tape"):
            mode = row[mode_key]
            prefix = "same_tape" if mode_key == "same_tape" else "separated_tape"
            for gradient_key in ("value_gradient", "log_likelihood_gradient", "final_particles_sum_gradient"):
                gradient = mode[gradient_key]
                if not gradient["connected"]:
                    findings.append(f"{label}:{prefix}:{gradient_key}:disconnected")
                elif not gradient["finite"]:
                    findings.append(f"{label}:{prefix}:{gradient_key}:nonfinite")
        first_break = row["first_observed_checkpoint_break"]
        if first_break["checkpoint"] is not None:
            findings.append(f"{label}:first_observed_checkpoint_break:{first_break['checkpoint']}:{first_break['reason']}")
    return findings


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
    checkpoints_by_seed = {}
    with tf.device(args.device):
        for seed, label in seed_probe_pairs:
            if label not in probe_specs:
                raise ValueError(f"unknown probe label: {label}")
            if seed not in fixtures:
                fixture = lgssm_gate.build_lgssm_gate_fixture(args.case_id, seed, args)
                fixtures[seed] = fixture
                base_tensors_by_seed[seed] = lgssm_gate._fixture_tensors(fixture, args.num_particles, seed, dtype)
                primary_by_seed[seed] = _compiled_primary_ab_function(fixture, base_tensors_by_seed[seed], args, dtype)
                checkpoints_by_seed[seed] = _compiled_checkpoint_jacobian_function(
                    fixture,
                    base_tensors_by_seed[seed],
                    args,
                    dtype,
                )
            theta = tf.cast(probe_specs[label]["theta"], dtype)
            primary_values = primary_by_seed[seed](theta)
            checkpoints = _checkpoint_rows_from_jacobian(checkpoints_by_seed[seed](theta))
            same = _same_tape_primary_from_compiled(primary_values, checkpoints)
            separated = _separated_tape_primary_from_compiled(primary_values)
            rows.append(_row(seed, label, theta, same, separated))
    artifact_vetoes = _artifact_vetoes(rows, args)
    diagnostic_findings = _diagnostic_findings(rows)
    ended_at = _dt.datetime.now(tz=_dt.UTC).isoformat()
    evidence_class = "cpu_hidden_debug_only" if args.device_scope == "cpu" else GPU_TRUST_BASIS
    h5_decidable = all(not row["same_tape"]["missing_required_checkpoints"] for row in rows)
    status = "PASS" if not artifact_vetoes and h5_decidable else "FAIL"
    return {
        "schema_version": "low_rank_ledh_route_internal_gradient_connectivity.v1",
        "phase": args.phase_id,
        "status": status,
        "evidence_class": evidence_class,
        "gpu_trust_basis": None if args.device_scope == "cpu" else GPU_TRUST_BASIS,
        "question": "localize first observed low-rank route-internal gradient-connectivity break on P02 failing probes",
        "interpretation_scope": "first observed checkpoint break, not exhaustive primitive-op proof",
        "candidate": calibration._candidate_settings(args),
        "required_tensor_checkpoints": list(REQUIRED_TENSORS),
        "artifact_vetoes": artifact_vetoes,
        "diagnostic_findings": diagnostic_findings,
        "h5_decidable": h5_decidable,
        "rows": rows,
        "baseline_artifacts": {
            "p02_result_note": P02_RESULT_PATH,
            "p02_raw_json": P02_JSON_PATH,
            "p02a_result_note": P02A_RESULT_PATH,
            "p02a_raw_json": P02A_JSON_PATH,
            "p02_raw_metadata_quarantine": "P02 result note quarantines stale raw phase/title metadata",
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
            "subplan_path": P02B_SUBPLAN_PATH,
            "p02_result_path": P02_RESULT_PATH,
            "p02a_result_path": P02A_RESULT_PATH,
            "device_scope": args.device_scope,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "device": args.device,
            "expect_device_kind": args.expect_device_kind,
            "jit_compile": bool(args.jit_compile),
            "case_id": args.case_id,
            "seed_probe_pairs": [{"seed": seed, "probe_label": label} for seed, label in seed_probe_pairs],
            **config,
        },
        "nonclaims": list(NONCLAIMS),
    }


def write_markdown(result: dict[str, Any], path: Path, json_path: Path | None = None) -> None:
    lines = [
        "# P02B Route-Internal Gradient Connectivity Diagnostic",
        "",
        f"- Status: `{result['status']}`",
        f"- Phase: `{result['phase']}`",
        f"- Evidence class: `{result['evidence_class']}`",
        f"- Artifact vetoes: `{result['artifact_vetoes']}`",
        f"- Diagnostic findings: `{result['diagnostic_findings']}`",
        f"- H5 decidable: `{result['h5_decidable']}`",
        f"- Interpretation scope: `{result['interpretation_scope']}`",
    ]
    if json_path is not None:
        lines.append(f"- JSON artifact: `{json_path}`")
    lines.extend(
        [
            "",
            "## Rows",
            "",
            "| Seed | Probe | Same loglik grad | Same final-particle grad | Separated loglik grad | First observed checkpoint break | H1 tape artifact |",
            "| ---: | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in result["rows"]:
        same = row["same_tape"]
        separated = row["p02a_style_separated_tape"]
        first_break = row["first_observed_checkpoint_break"]
        lines.append(
            "| {seed} | `{probe}` | `{same_loglik}` | `{same_particles}` | `{sep_loglik}` | `{breakpoint}:{reason}` | `{h1}` |".format(
                seed=row["seed"],
                probe=row["probe_label"],
                same_loglik=same["log_likelihood_gradient"],
                same_particles=same["final_particles_sum_gradient"],
                sep_loglik=separated["log_likelihood_gradient"],
                breakpoint=first_break["checkpoint"],
                reason=first_break["reason"],
                h1=row["ab_comparison"]["supports_h1_tape_artifact"],
            )
        )
    lines.extend(
        [
            "",
            "## Run Manifest",
            "",
            f"- Git commit: `{result['run_manifest']['git_commit']}`",
            f"- Device scope: `{result['run_manifest']['device_scope']}`",
            f"- CUDA_VISIBLE_DEVICES: `{result['run_manifest']['cuda_visible_devices']}`",
            f"- TF32 recorded: `{result['run_manifest']['tf32_execution_recorded']}`",
            f"- JIT compile: `{result['run_manifest']['jit_compile']}`",
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
