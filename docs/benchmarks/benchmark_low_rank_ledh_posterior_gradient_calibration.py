"""LGSSM posterior value/gradient instrumentation for low-rank LEDH-PFPF-OT.

This P01 harness measures the inherited low-rank residual diagnostics next to
exact LGSSM Kalman value/gradient diagnostics at fixed probe parameters.  Local
CPU-hidden runs are command-shape checks only.  Calibration, default readiness,
posterior correctness, HMC readiness, and statistical superiority require later
gated phases.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import platform
import statistics
import subprocess
import sys
import time
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


PLAN_PATH = (
    "docs/plans/bayesfilter-dpf-ledh-pfpf-ot-low-rank-residual-posterior-gradient-"
    "calibration-master-program-2026-06-24.md"
)
GPU_TRUST_BASIS = "owner_designated_managed_session_visible_gpu_trusted"
P01_SUBPLAN_PATH = (
    "docs/plans/bayesfilter-dpf-ledh-pfpf-ot-low-rank-residual-posterior-gradient-"
    "calibration-p01-instrumentation-subplan-2026-06-24.md"
)
P01_RESULT_PATH = (
    "docs/plans/bayesfilter-dpf-ledh-pfpf-ot-low-rank-residual-posterior-gradient-"
    "calibration-p01-instrumentation-result-2026-06-24.md"
)

NONCLAIMS = (
    "P01 instrumentation and command-shape artifact only",
    "residual remains a proxy until calibrated against value/gradient/peak diagnostics",
    "fixed probe-neighborhood peak is not a global MAP claim",
    "no calibrated threshold claim",
    "no posterior correctness claim",
    "no HMC readiness claim",
    "no package/public default readiness claim",
    "no statistical superiority claim",
    "no scientific validity claim",
)


def _parse_args() -> argparse.Namespace:
    return _parse_args_impl(None)


def _parse_args_from_list_for_test(argv: list[str]) -> argparse.Namespace:
    return _parse_args_impl(argv)


def _parse_args_impl(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--case-ids", nargs="+", default=["lgssm_small_exact_ref"])
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--route", choices=("low_rank", "streaming", "both"), default="both")
    parser.add_argument("--num-particles", type=int, default=128)
    parser.add_argument("--time-steps", type=int, default=4)
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
    parser.add_argument("--gradient-norm-floor", type=float, default=1.0e-8)
    parser.add_argument("--warmups", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--jit-compile", dest="jit_compile", action="store_true", default=True)
    parser.add_argument("--no-jit-compile", dest="jit_compile", action="store_false")
    parser.add_argument("--tf32-mode", choices=("default", "enabled", "disabled"), default="enabled")
    parser.add_argument("--device", default="/GPU:0")
    parser.add_argument("--device-scope", choices=("cpu", "visible"), default=_PRE_ARGS.device_scope)
    parser.add_argument("--cuda-visible-devices", default=_PRE_ARGS.cuda_visible_devices)
    parser.add_argument("--expect-device-kind", choices=("any", "cpu", "gpu"), default="gpu")
    parser.add_argument("--phase-id", default="LOW_RANK_RESIDUAL_POSTERIOR_GRADIENT_CALIBRATION")
    parser.add_argument("--output", required=True)
    parser.add_argument("--markdown-output", default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    if args.seeds is not None:
        args.seeds = lgssm_gate._parse_int_csv(args.seeds)
    _validate_args(args)
    return args


def _validate_args(args: argparse.Namespace) -> None:
    unknown = [case_id for case_id in args.case_ids if case_id not in lgssm_gate.PINNED_CASES]
    if unknown:
        raise ValueError(f"unknown case ids: {unknown}")
    if args.num_particles <= 1:
        raise ValueError("num_particles must be greater than one")
    for name in ("time_steps", "low_rank_rank", "low_rank_max_projection_iterations"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
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
    if args.sinkhorn_iterations <= 0:
        raise ValueError("sinkhorn_iterations must be positive")
    for name in ("row_chunk_size", "col_chunk_size", "particle_chunk_size"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if args.theta_probe_radius <= 0.0:
        raise ValueError("theta_probe_radius must be positive")
    if args.theta_prior_scale <= 0.0:
        raise ValueError("theta_prior_scale must be positive")
    if args.gradient_norm_floor <= 0.0:
        raise ValueError("gradient_norm_floor must be positive")
    if args.warmups < 0 or args.repeats <= 0:
        raise ValueError("warmups must be nonnegative and repeats must be positive")
    if args.device_scope == "cpu" and args.expect_device_kind == "gpu":
        raise ValueError("cpu device scope cannot expect gpu outputs")


def _dtype(args: argparse.Namespace) -> tf.DType:
    return tf.float64 if args.dtype == "float64" else tf.float32


def _route_names(route: str) -> list[str]:
    return ["streaming", "low_rank"] if route == "both" else [route]


def _case_seeds(case_id: str, args: argparse.Namespace) -> list[int]:
    if args.seeds is not None:
        return [int(seed) for seed in args.seeds]
    return [int(lgssm_gate.PINNED_CASES[case_id]["seeds"][0])]


def _probe_specs(args: argparse.Namespace, dtype: tf.DType) -> list[dict[str, Any]]:
    radius = tf.constant(float(args.theta_probe_radius), dtype=dtype)
    zero = tf.constant(0.0, dtype=dtype)
    raw = [
        ("center", [zero, zero]),
        ("q_plus", [radius, zero]),
        ("q_minus", [-radius, zero]),
        ("r_plus", [zero, radius]),
        ("r_minus", [zero, -radius]),
        ("qr_plus", [radius, radius]),
        ("qr_minus", [-radius, -radius]),
    ]
    return [{"label": label, "theta": tf.stack(values)} for label, values in raw]


def _scaled_fixture(
    fixture: lgssm_gate.LGSSMGateFixture,
    theta: tf.Tensor,
    dtype: tf.DType,
) -> lgssm_gate.LGSSMGateFixture:
    theta = tf.cast(theta, dtype)
    q_scale = tf.exp(theta[0])
    r_scale = tf.exp(theta[1])
    return lgssm_gate.LGSSMGateFixture(
        case_id=fixture.case_id,
        role=fixture.role,
        A=tf.cast(fixture.A, dtype),
        C=tf.cast(fixture.C, dtype),
        Q=tf.cast(fixture.Q, dtype) * q_scale,
        R=tf.cast(fixture.R, dtype) * r_scale,
        m0=tf.cast(fixture.m0, dtype),
        P0=tf.cast(fixture.P0, dtype),
        states=tf.cast(fixture.states, dtype),
        observations=tf.cast(fixture.observations, dtype),
        seed=fixture.seed,
    )


def _theta_prior_log_density(theta: tf.Tensor, prior_scale: float, dtype: tf.DType) -> tf.Tensor:
    scaled = tf.cast(theta, dtype) / tf.constant(float(prior_scale), dtype=dtype)
    return -0.5 * tf.reduce_sum(tf.square(scaled))


def _compiled_exact_value_gradient(
    fixture: lgssm_gate.LGSSMGateFixture,
    args: argparse.Namespace,
    dtype: tf.DType,
):
    @tf.function(jit_compile=args.jit_compile, reduce_retracing=True)
    def compiled(theta: tf.Tensor) -> tuple[tf.Tensor, ...]:
        theta = tf.cast(theta, dtype)
        with tf.GradientTape() as tape:
            tape.watch(theta)
            scaled_fixture = _scaled_fixture(fixture, theta, dtype)
            kalman = lgssm_gate.run_kalman_reference(scaled_fixture, dtype)
            log_likelihood = tf.reshape(tf.cast(kalman["log_likelihood"], dtype), [])
            prior = _theta_prior_log_density(theta, args.theta_prior_scale, dtype)
            value = log_likelihood + prior
        gradient = tape.gradient(value, theta)
        if gradient is None:
            gradient = tf.zeros_like(theta)
        finite = (
            kalman["finite"]
            & tf.math.is_finite(value)
            & tf.reduce_all(tf.math.is_finite(gradient))
        )
        return value, gradient, log_likelihood, prior, finite

    return compiled


def _compiled_route_value_gradient(
    route: str,
    fixture: lgssm_gate.LGSSMGateFixture,
    args: argparse.Namespace,
    dtype: tf.DType,
):
    base_tensors = lgssm_gate._fixture_tensors(fixture, args.num_particles, fixture.seed, dtype)

    @tf.function(jit_compile=args.jit_compile, reduce_retracing=True)
    def compiled(theta: tf.Tensor) -> tuple[tf.Tensor, ...]:
        theta = tf.cast(theta, dtype)
        with tf.GradientTape() as tape:
            tape.watch(theta)
            scaled_fixture = _scaled_fixture(fixture, theta, dtype)
            tensors = dict(base_tensors)
            tensors["transition_covariance"] = tf.cast(scaled_fixture.Q[None, :, :], dtype)
            tensors["observation_covariance"] = tf.cast(scaled_fixture.R[None, :, :], dtype)
            callbacks = lgssm_gate._callbacks(tensors, dtype)
            route_outputs = lgssm_gate._route_value_core(route, tensors, callbacks, args, dtype)
            log_likelihood = tf.reshape(tf.cast(route_outputs.log_likelihood, dtype), [-1])[0]
            prior = _theta_prior_log_density(theta, args.theta_prior_scale, dtype)
            value = log_likelihood + prior
        gradient = tape.gradient(value, theta)
        if gradient is None:
            gradient = tf.zeros_like(theta)
        finite_output = (
            tf.reduce_all(tf.math.is_finite(route_outputs.log_likelihood))
            & tf.reduce_all(tf.math.is_finite(route_outputs.filtered_means))
            & tf.reduce_all(tf.math.is_finite(route_outputs.filtered_variances))
            & tf.reduce_all(tf.math.is_finite(route_outputs.ess_by_time))
            & tf.math.is_finite(value)
            & tf.reduce_all(tf.math.is_finite(gradient))
        )
        return (
            value,
            gradient,
            log_likelihood,
            prior,
            tf.reduce_min(route_outputs.ess_by_time),
            route_outputs.final_logsumexp_residual,
            route_outputs.route_invocations,
            route_outputs.active_resampling_mask_count,
            route_outputs.active_resampling_batch_entries,
            route_outputs.max_factor_marginal_residual,
            route_outputs.max_induced_row_residual,
            route_outputs.max_induced_column_residual,
            route_outputs.projection_iterations_used_max,
            route_outputs.finite_factors,
            route_outputs.finite_particles,
            route_outputs.nonnegative_factors,
            route_outputs.positive_g,
            finite_output,
        )

    return compiled


def _run_route_case(
    route: str,
    fixture: lgssm_gate.LGSSMGateFixture,
    args: argparse.Namespace,
) -> dict[str, Any]:
    dtype = _dtype(args)
    exact_compiled = _compiled_exact_value_gradient(fixture, args, dtype)
    route_compiled = _compiled_route_value_gradient(route, fixture, args, dtype)
    probe_specs = _probe_specs(args, dtype)
    probe_rows: list[dict[str, Any]] = []
    output_devices: list[str] = []

    start = time.perf_counter()
    with tf.device(args.device):
        first_exact = exact_compiled(probe_specs[0]["theta"])
        first_route = route_compiled(probe_specs[0]["theta"])
        probe_rows.append(_probe_row(0, probe_specs[0], first_exact, first_route, args))
        output_devices.extend([first_exact[0].device, first_exact[1].device, first_route[0].device, first_route[1].device])
        first_call_seconds = time.perf_counter() - start
        for _ in range(args.warmups):
            for spec in probe_specs:
                exact_compiled(spec["theta"])
                route_compiled(spec["theta"])
        timings = []
        repeat_summaries = []
        for _ in range(args.repeats):
            repeat_start = time.perf_counter()
            rows = []
            for index, spec in enumerate(probe_specs):
                exact_outputs = exact_compiled(spec["theta"])
                route_outputs = route_compiled(spec["theta"])
                rows.append(_probe_row(index, spec, exact_outputs, route_outputs, args))
                output_devices.extend(
                    [
                        exact_outputs[0].device,
                        exact_outputs[1].device,
                        route_outputs[0].device,
                        route_outputs[1].device,
                    ]
                )
            timings.append(time.perf_counter() - repeat_start)
            probe_rows = rows
            repeat_summaries.append(_repeat_summary(len(repeat_summaries), rows))

    peak = _peak_neighborhood_summary(probe_rows)
    gradient_summary = _gradient_summary(probe_rows)
    hard_vetoes = _hard_vetoes(route, probe_rows, output_devices, args)
    return {
        "case_id": fixture.case_id,
        "case_role": fixture.role,
        "seed": fixture.seed,
        "route": route,
        "status": "PASS" if not hard_vetoes else "FAIL",
        "hard_vetoes": hard_vetoes,
        "shape": {
            "state_dim": fixture.state_dim,
            "obs_dim": fixture.obs_dim,
            "time_steps": fixture.horizon,
            "num_particles": int(args.num_particles),
        },
        "theta_parameterization": {
            "theta_names": ["log_transition_covariance_scale", "log_observation_covariance_scale"],
            "theta_prior": "zero_mean_diagonal_normal_up_to_constant",
            "theta_prior_scale": float(args.theta_prior_scale),
            "probe_radius": float(args.theta_probe_radius),
            "peak_scope": "fixed_predeclared_probe_neighborhood_not_global_map",
        },
        "candidate_settings": _candidate_settings(args),
        "first_call_seconds": first_call_seconds,
        "warm_call_timings_seconds": timings,
        "warm_call_timing_summary_seconds": _summary(timings),
        "diagnostic_repeat_summaries": repeat_summaries,
        "output_devices": sorted(set(output_devices)),
        "probe_rows": probe_rows,
        "peak_neighborhood": peak,
        "gradient_summary": gradient_summary,
    }


def _probe_row(
    index: int,
    spec: dict[str, Any],
    exact_outputs: tuple[tf.Tensor, ...],
    route_outputs: tuple[tf.Tensor, ...],
    args: argparse.Namespace,
) -> dict[str, Any]:
    exact_value, exact_gradient, exact_loglik, exact_prior, exact_finite = exact_outputs
    route_value, route_gradient, route_loglik, route_prior = route_outputs[:4]
    gradient_delta = tf.cast(route_gradient, tf.float64) - tf.cast(exact_gradient, tf.float64)
    exact_norm = tf.linalg.norm(tf.cast(exact_gradient, tf.float64))
    route_norm = tf.linalg.norm(tf.cast(route_gradient, tf.float64))
    delta_norm = tf.linalg.norm(gradient_delta)
    floor = tf.constant(float(args.gradient_norm_floor), dtype=tf.float64)
    cosine = tf.reduce_sum(tf.cast(route_gradient, tf.float64) * tf.cast(exact_gradient, tf.float64))
    cosine = cosine / tf.maximum(route_norm * exact_norm, floor)
    return {
        "probe_index": int(index),
        "probe_label": str(spec["label"]),
        "theta": _vector(spec["theta"]),
        "exact": {
            "posterior_value": _scalar(exact_value),
            "log_likelihood": _scalar(exact_loglik),
            "prior_log_density_without_constant": _scalar(exact_prior),
            "gradient": _vector(exact_gradient),
            "gradient_norm": _scalar(exact_norm),
            "finite_value_gradient": _bool(exact_finite),
        },
        "route": {
            "posterior_value": _scalar(route_value),
            "log_likelihood": _scalar(route_loglik),
            "prior_log_density_without_constant": _scalar(route_prior),
            "gradient": _vector(route_gradient),
            "gradient_norm": _scalar(route_norm),
            "finite_value_gradient": _bool(route_outputs[17]),
        },
        "value_abs_error": _scalar(tf.abs(tf.cast(route_value, tf.float64) - tf.cast(exact_value, tf.float64))),
        "gradient_relative_norm_error": _scalar(delta_norm / tf.maximum(exact_norm, floor)),
        "gradient_max_coordinate_error": _scalar(tf.reduce_max(tf.abs(gradient_delta))),
        "gradient_cosine_similarity": _scalar(cosine),
        "route_diagnostics": {
            "ess_min": _scalar(route_outputs[4]),
            "final_logsumexp_residual": _scalar(route_outputs[5]),
            "route_invocations": _int(route_outputs[6]),
            "active_resampling_mask_count": _int(route_outputs[7]),
            "active_resampling_batch_entries": _int(route_outputs[8]),
            "max_factor_marginal_residual": _scalar(route_outputs[9]),
            "max_induced_row_residual": _scalar(route_outputs[10]),
            "max_induced_column_residual": _scalar(route_outputs[11]),
            "projection_iterations_used_max": _int(route_outputs[12]),
            "finite_factors": _bool(route_outputs[13]),
            "finite_particles": _bool(route_outputs[14]),
            "nonnegative_factors": _bool(route_outputs[15]),
            "positive_g": _bool(route_outputs[16]),
            "transport_object_kind": "low_rank_coupling_factors"
            if _int(route_outputs[12]) > 0
            else "streaming_transport_or_no_projection",
            "transport_matrix_materialized": False,
        },
    }


def _peak_neighborhood_summary(probe_rows: list[dict[str, Any]]) -> dict[str, Any]:
    exact_values = [float(row["exact"]["posterior_value"]) for row in probe_rows]
    route_values = [float(row["route"]["posterior_value"]) for row in probe_rows]
    exact_peak_index = max(range(len(exact_values)), key=lambda item: exact_values[item])
    route_peak_index = max(range(len(route_values)), key=lambda item: route_values[item])
    exact_peak_row = probe_rows[exact_peak_index]
    route_peak_row = probe_rows[route_peak_index]
    return {
        "scope": "fixed_predeclared_probe_neighborhood_not_global_map",
        "exact_peak_probe_index": int(exact_peak_index),
        "exact_peak_probe_label": exact_peak_row["probe_label"],
        "exact_peak_theta": exact_peak_row["theta"],
        "exact_peak_value": exact_peak_row["exact"]["posterior_value"],
        "route_peak_probe_index": int(route_peak_index),
        "route_peak_probe_label": route_peak_row["probe_label"],
        "route_peak_theta": route_peak_row["theta"],
        "route_peak_value": route_peak_row["route"]["posterior_value"],
        "peak_probe_match": bool(exact_peak_index == route_peak_index),
        "value_error_at_exact_peak": exact_peak_row["value_abs_error"],
        "route_gradient_norm_at_exact_peak": exact_peak_row["route"]["gradient_norm"],
        "exact_gradient_norm_at_exact_peak": exact_peak_row["exact"]["gradient_norm"],
        "max_value_abs_error_over_probes": max(float(row["value_abs_error"]) for row in probe_rows),
        "max_gradient_relative_norm_error_over_probes": max(
            float(row["gradient_relative_norm_error"]) for row in probe_rows
        ),
    }


def _gradient_summary(probe_rows: list[dict[str, Any]]) -> dict[str, float]:
    rel = [float(row["gradient_relative_norm_error"]) for row in probe_rows]
    coord = [float(row["gradient_max_coordinate_error"]) for row in probe_rows]
    cos = [float(row["gradient_cosine_similarity"]) for row in probe_rows]
    return {
        "gradient_relative_norm_error_max": max(rel),
        "gradient_relative_norm_error_mean": statistics.fmean(rel),
        "gradient_max_coordinate_error_max": max(coord),
        "gradient_cosine_similarity_min": min(cos),
    }


def _repeat_summary(repeat_index: int, probe_rows: list[dict[str, Any]]) -> dict[str, Any]:
    peak = _peak_neighborhood_summary(probe_rows)
    gradient = _gradient_summary(probe_rows)
    center = probe_rows[0]
    return {
        "repeat_index": int(repeat_index),
        "peak_probe_match": peak["peak_probe_match"],
        "exact_peak_probe_label": peak["exact_peak_probe_label"],
        "route_peak_probe_label": peak["route_peak_probe_label"],
        "max_value_abs_error_over_probes": peak["max_value_abs_error_over_probes"],
        "max_gradient_relative_norm_error_over_probes": peak["max_gradient_relative_norm_error_over_probes"],
        "gradient_relative_norm_error_max": gradient["gradient_relative_norm_error_max"],
        "gradient_cosine_similarity_min": gradient["gradient_cosine_similarity_min"],
        "center_factor_marginal_residual": center["route_diagnostics"]["max_factor_marginal_residual"],
        "center_induced_row_residual": center["route_diagnostics"]["max_induced_row_residual"],
        "center_induced_column_residual": center["route_diagnostics"]["max_induced_column_residual"],
        "center_projection_iterations_used_max": center["route_diagnostics"]["projection_iterations_used_max"],
    }


def _hard_vetoes(
    route: str,
    probe_rows: list[dict[str, Any]],
    output_devices: list[str],
    args: argparse.Namespace,
) -> list[str]:
    vetoes: list[str] = []
    if args.expect_device_kind == "gpu" and not all("GPU" in device.upper() for device in output_devices):
        vetoes.append("expected_gpu_outputs_missing")
    if args.expect_device_kind == "cpu" and not all("CPU" in device.upper() for device in output_devices):
        vetoes.append("expected_cpu_outputs_missing")
    for row in probe_rows:
        label = row["probe_label"]
        if not row["exact"]["finite_value_gradient"]:
            vetoes.append(f"{label}:exact_value_gradient_nonfinite")
        if not row["route"]["finite_value_gradient"]:
            vetoes.append(f"{label}:route_value_gradient_nonfinite")
        diagnostics = row["route_diagnostics"]
        if diagnostics["route_invocations"] != diagnostics["active_resampling_mask_count"]:
            vetoes.append(f"{label}:route_invocation_count_mismatch")
        if diagnostics["transport_matrix_materialized"]:
            vetoes.append(f"{label}:transport_matrix_materialized")
        if route == "low_rank":
            if not diagnostics["finite_factors"]:
                vetoes.append(f"{label}:nonfinite_factors")
            if not diagnostics["finite_particles"]:
                vetoes.append(f"{label}:low_rank_nonfinite_particles")
            if not diagnostics["nonnegative_factors"]:
                vetoes.append(f"{label}:negative_factors")
            if not diagnostics["positive_g"]:
                vetoes.append(f"{label}:nonpositive_g")
    return vetoes


def build_result(args: argparse.Namespace) -> dict[str, Any]:
    tf_metadata = lgssm_gate._configure_precision(args)
    device_metadata = lgssm_gate._configure_gpus()
    started_at = _dt.datetime.now(tz=_dt.UTC).isoformat()
    start = time.perf_counter()
    rows = []
    for case_id in args.case_ids:
        for seed in _case_seeds(case_id, args):
            fixture = lgssm_gate.build_lgssm_gate_fixture(case_id, int(seed), args)
            for route in _route_names(args.route):
                rows.append(_run_route_case(route, fixture, args))
    ended_at = _dt.datetime.now(tz=_dt.UTC).isoformat()
    hard_vetoes = [
        f"{row['case_id']}:{row['seed']}:{row['route']}:{veto}"
        for row in rows
        for veto in row.get("hard_vetoes", [])
    ]
    if args.device_scope == "cpu":
        evidence_class = "cpu_hidden_command_shape_debug_only"
    elif args.expect_device_kind == "gpu":
        evidence_class = "owner_designated_managed_session_visible_gpu_trusted"
    else:
        evidence_class = "local_debug_only"
    return {
        "schema_version": "low_rank_ledh_posterior_gradient_calibration.v1",
        "status": "PASS" if not hard_vetoes else "FAIL",
        "phase": args.phase_id,
        "evidence_class": evidence_class,
        "algorithm_family": "low_rank_ledh_pfpf_ot_residual_posterior_gradient_calibration",
        "candidate": _candidate_settings(args),
        "hard_vetoes": hard_vetoes,
        "rows": rows,
        "run_manifest": _run_manifest(
            args,
            started_at=started_at,
            ended_at=ended_at,
            wall_time_seconds=time.perf_counter() - start,
            tf_metadata=tf_metadata,
            device_metadata=device_metadata,
        ),
        "gpu_trust_basis": GPU_TRUST_BASIS if args.expect_device_kind == "gpu" else "not_gpu_trusted_evidence",
        "nonclaims": list(NONCLAIMS),
    }


def _candidate_settings(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "route": "low_rank_ledh_pfpf_ot",
        "candidate_id": _candidate_id(args),
        "rank": int(args.low_rank_rank),
        "assignment_epsilon": float(args.low_rank_assignment_epsilon),
        "alpha": float(args.low_rank_alpha),
        "max_projection_iterations": int(args.low_rank_max_projection_iterations),
        "convergence_threshold": float(args.low_rank_convergence_threshold),
        "denominator_floor": float(args.low_rank_denominator_floor),
        "sinkhorn_iterations": int(args.sinkhorn_iterations),
        "sinkhorn_epsilon": float(args.sinkhorn_epsilon),
        "annealed_scaling": float(args.annealed_scaling),
    }


def _candidate_id(args: argparse.Namespace) -> str:
    eps = str(args.low_rank_assignment_epsilon).replace(".", "p").replace("-", "m")
    alpha = f"{args.low_rank_alpha:.0e}".replace("-", "m")
    return f"r{args.low_rank_rank}_eps{eps}_alpha{alpha}_it{args.low_rank_max_projection_iterations}"


def _run_manifest(
    args: argparse.Namespace,
    *,
    started_at: str,
    ended_at: str,
    wall_time_seconds: float,
    tf_metadata: dict[str, Any],
    device_metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "git_commit": _git_output(["git", "rev-parse", "HEAD"]),
        "git_status_short": _git_output(["git", "status", "--short"]),
        "command": " ".join(sys.argv),
        "working_directory": str(ROOT),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "tensorflow_version": tf.__version__,
        "tensorflow_probability_version": tfp.__version__,
        "device_scope": args.device_scope,
        "device": args.device,
        "expect_device_kind": args.expect_device_kind,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "jit_compile": bool(args.jit_compile),
        "gpu_trust_basis": GPU_TRUST_BASIS if args.expect_device_kind == "gpu" else "not_gpu_trusted_evidence",
        "warmups": int(args.warmups),
        "repeats": int(args.repeats),
        "case_ids": list(args.case_ids),
        "seeds_override": list(args.seeds) if args.seeds is not None else None,
        "plan_path": PLAN_PATH,
        "subplan_path": P01_SUBPLAN_PATH,
        "p01_result_path": P01_RESULT_PATH,
        **tf_metadata,
        **device_metadata,
        "started_at": started_at,
        "ended_at": ended_at,
        "wall_time_seconds": wall_time_seconds,
    }


def _git_output(command: list[str]) -> str:
    try:
        return subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "median": None, "mean": None, "max": None}
    return {
        "min": min(values),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "max": max(values),
    }


def _scalar(value: Any) -> float:
    return float(tf.reshape(tf.cast(value, tf.float64), [-1])[0])


def _int(value: Any) -> int:
    return int(tf.reshape(tf.cast(value, tf.int64), [-1])[0])


def _bool(value: Any) -> bool:
    return bool(tf.reshape(tf.cast(value, tf.bool), [-1])[0])


def _vector(value: tf.Tensor) -> list[float]:
    flat = tf.reshape(tf.cast(value, tf.float64), [-1])
    return [float(item) for item in flat]


def write_markdown(result: dict[str, Any], path: Path, json_path: Path | None = None) -> None:
    lines = [
        "# Low-Rank Residual Posterior-Gradient Calibration P01",
        "",
        f"- Status: `{result['status']}`",
        f"- Phase: `{result['phase']}`",
        f"- Evidence class: `{result['evidence_class']}`",
        f"- Hard vetoes: `{result['hard_vetoes']}`",
    ]
    if json_path is not None:
        lines.append(f"- JSON artifact: `{json_path}`")
    lines.extend(
        [
            "",
            "## Rows",
            "",
            "| Case | Seed | Route | Status | Peak Match | Max Value Error | Max Grad Rel Error | Min Grad Cos | Vetoes |",
            "| --- | ---: | --- | --- | --- | ---: | ---: | ---: | --- |",
        ]
    )
    for row in result["rows"]:
        lines.append(
            "| {case} | {seed} | `{route}` | `{status}` | `{peak}` | {value_err} | {grad_err} | {cosine} | `{vetoes}` |".format(
                case=row["case_id"],
                seed=row["seed"],
                route=row["route"],
                status=row["status"],
                peak=row["peak_neighborhood"]["peak_probe_match"],
                value_err=row["peak_neighborhood"]["max_value_abs_error_over_probes"],
                grad_err=row["gradient_summary"]["gradient_relative_norm_error_max"],
                cosine=row["gradient_summary"]["gradient_cosine_similarity_min"],
                vetoes=row["hard_vetoes"],
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
    result = build_result(args)
    output = Path(args.output)
    markdown = Path(args.markdown_output) if args.markdown_output else output.with_suffix(".md")
    output.parent.mkdir(parents=True, exist_ok=True)
    markdown.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(result, markdown, output)
    if not args.quiet:
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
