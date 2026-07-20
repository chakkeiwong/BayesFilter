"""Focused low-rank LEDH-PFPF-OT gradient nonfinite diagnostic.

This repair diagnostic reuses the posterior-gradient calibration fixture and
probe definitions, but only evaluates predeclared failing low-rank probes from
P02.  It localizes whether nonfinite gradients arise in the route likelihood,
the prior term, or the comparison layer.  It is not a calibration, holdout,
posterior-correctness, HMC-readiness, default-readiness, or statistical-ranking
artifact.
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


PLAN_PATH = (
    "docs/plans/bayesfilter-dpf-ledh-pfpf-ot-low-rank-residual-posterior-gradient-"
    "calibration-master-program-2026-06-24.md"
)
P02_RESULT_PATH = (
    "docs/plans/bayesfilter-dpf-ledh-pfpf-ot-low-rank-residual-posterior-gradient-"
    "calibration-p02-reproduction-determinism-result-2026-06-24.md"
)
GPU_TRUST_BASIS = "owner_designated_managed_session_visible_gpu_trusted"

DEFAULT_PROBES_BY_SEED = {
    91002: ("qr_plus",),
    91003: ("center", "q_plus", "q_minus", "r_plus", "r_minus", "qr_plus"),
}
NONCLAIMS = (
    "focused repair diagnostic only",
    "no calibrated residual threshold claim",
    "no holdout validation claim",
    "no posterior correctness claim",
    "no HMC readiness claim",
    "no default/package/public API readiness claim",
    "no statistical superiority claim",
    "no scientific validity claim",
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
    parser.add_argument("--phase-id", default="LOW_RANK_GRADIENT_REPAIR_DIAGNOSTIC")
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
    if args.device_scope == "cpu" and args.expect_device_kind == "gpu":
        raise ValueError("cpu device scope cannot expect gpu outputs")
    if args.particle_chunk_size <= 0:
        raise ValueError("particle_chunk_size must be positive")


def _seed_probe_pairs(args: argparse.Namespace) -> list[tuple[int, str]]:
    if args.seed_probes is None:
        return [
            (seed, label)
            for seed, labels in DEFAULT_PROBES_BY_SEED.items()
            for label in labels
        ]
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
    dtype = calibration._dtype(args)
    return dtype, {**precision, **gpu_config}


def _git_output(command: list[str]) -> str:
    try:
        return subprocess.check_output(command, cwd=ROOT, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:  # noqa: BLE001
        return f"unavailable:{type(exc).__name__}:{exc}"


def _scalar(value: tf.Tensor) -> float:
    return float(tf.reshape(value, []).numpy())


def _vector(value: tf.Tensor) -> list[float]:
    return [float(item) for item in tf.reshape(value, [-1]).numpy().tolist()]


def _bool(value: tf.Tensor) -> bool:
    return bool(tf.reshape(tf.cast(value, tf.bool), []).numpy())


def _int(value: tf.Tensor) -> int:
    return int(tf.reshape(value, []).numpy())


def _probe_spec_by_label(args: argparse.Namespace, dtype: tf.DType) -> dict[str, dict[str, Any]]:
    return {spec["label"]: spec for spec in calibration._probe_specs(args, dtype)}


def _diagnostic_function(fixture: lgssm_gate.LGSSMGateFixture, args: argparse.Namespace, dtype: tf.DType):
    base_tensors = lgssm_gate._fixture_tensors(fixture, args.num_particles, fixture.seed, dtype)

    @tf.function(jit_compile=args.jit_compile, reduce_retracing=True)
    def compiled(theta: tf.Tensor) -> tuple[tf.Tensor, ...]:
        theta = tf.cast(theta, dtype)
        def route_terms() -> tuple[lgssm_gate.RouteValueTensors, tf.Tensor, tf.Tensor, tf.Tensor]:
            tape.watch(theta)
            scaled_fixture = calibration._scaled_fixture(fixture, theta, dtype)
            tensors = dict(base_tensors)
            tensors["transition_covariance"] = tf.cast(scaled_fixture.Q[None, :, :], dtype)
            tensors["observation_covariance"] = tf.cast(scaled_fixture.R[None, :, :], dtype)
            callbacks = lgssm_gate._callbacks(tensors, dtype)
            route_outputs = lgssm_gate._route_value_core("low_rank", tensors, callbacks, args, dtype)
            log_likelihood = tf.reshape(tf.cast(route_outputs.log_likelihood, dtype), [-1])[0]
            prior = calibration._theta_prior_log_density(theta, args.theta_prior_scale, dtype)
            value = log_likelihood + prior
            return route_outputs, log_likelihood, prior, value

        with tf.GradientTape() as tape:
            route_outputs, log_likelihood, prior, value = route_terms()
        value_gradient = tape.gradient(value, theta)
        value_gradient_connected = tf.constant(value_gradient is not None)

        with tf.GradientTape() as tape:
            _loglik_route_outputs, log_likelihood_for_gradient, _prior_for_loglik, _value_for_loglik = route_terms()
        loglik_gradient = tape.gradient(log_likelihood_for_gradient, theta)
        loglik_gradient_connected = tf.constant(loglik_gradient is not None)

        with tf.GradientTape() as tape:
            tape.watch(theta)
            prior_for_gradient = calibration._theta_prior_log_density(theta, args.theta_prior_scale, dtype)
        prior_gradient = tape.gradient(prior_for_gradient, theta)
        prior_gradient_connected = tf.constant(prior_gradient is not None)

        with tf.GradientTape() as tape:
            final_particles_route_outputs, _loglik_for_particles, _prior_for_particles, _value_for_particles = route_terms()
            final_particles_sum = tf.reduce_sum(final_particles_route_outputs.final_particles)
        final_particles_grad = tape.gradient(final_particles_sum, theta)
        final_particles_grad_connected = tf.constant(final_particles_grad is not None)

        if value_gradient is None:
            value_gradient = tf.fill(tf.shape(theta), tf.constant(float("nan"), dtype=dtype))
        if loglik_gradient is None:
            loglik_gradient = tf.fill(tf.shape(theta), tf.constant(float("nan"), dtype=dtype))
        if prior_gradient is None:
            prior_gradient = tf.fill(tf.shape(theta), tf.constant(float("nan"), dtype=dtype))
        if final_particles_grad is None:
            final_particles_grad = tf.fill(tf.shape(theta), tf.constant(float("nan"), dtype=dtype))

        value_finite = tf.math.is_finite(value) & tf.reduce_all(tf.math.is_finite(value_gradient))
        loglik_finite = tf.math.is_finite(log_likelihood) & tf.reduce_all(tf.math.is_finite(loglik_gradient))
        prior_finite = tf.math.is_finite(prior) & tf.reduce_all(tf.math.is_finite(prior_gradient))
        final_particles_grad_finite = tf.reduce_all(tf.math.is_finite(final_particles_grad))
        outputs_finite = (
            tf.reduce_all(tf.math.is_finite(route_outputs.log_likelihood))
            & tf.reduce_all(tf.math.is_finite(route_outputs.filtered_means))
            & tf.reduce_all(tf.math.is_finite(route_outputs.filtered_variances))
            & tf.reduce_all(tf.math.is_finite(route_outputs.ess_by_time))
            & tf.reduce_all(tf.math.is_finite(route_outputs.final_particles))
            & tf.reduce_all(tf.math.is_finite(route_outputs.final_log_weights))
        )
        return (
            value,
            value_gradient,
            log_likelihood,
            loglik_gradient,
            prior,
            prior_gradient,
            value_gradient_connected,
            loglik_gradient_connected,
            prior_gradient_connected,
            final_particles_grad_connected,
            final_particles_grad,
            route_outputs.route_invocations,
            route_outputs.active_resampling_mask_count,
            route_outputs.max_factor_marginal_residual,
            route_outputs.max_induced_row_residual,
            route_outputs.max_induced_column_residual,
            route_outputs.projection_iterations_used_max,
            route_outputs.finite_factors,
            route_outputs.finite_particles,
            route_outputs.nonnegative_factors,
            route_outputs.positive_g,
            outputs_finite,
            value_finite,
            loglik_finite,
            prior_finite,
            final_particles_grad_finite,
        )

    return compiled


def _row(seed: int, label: str, theta: tf.Tensor, outputs: tuple[tf.Tensor, ...]) -> dict[str, Any]:
    output_device = outputs[0].device
    value_grad = outputs[1]
    loglik_grad = outputs[3]
    prior_grad = outputs[5]
    final_particles_grad = outputs[10]
    return {
        "seed": int(seed),
        "probe_label": label,
        "theta": _vector(theta),
        "output_device": output_device,
        "value": _scalar(outputs[0]),
        "value_gradient": _vector(value_grad),
        "value_gradient_finite_components": [bool(x) for x in tf.math.is_finite(value_grad).numpy().tolist()],
        "log_likelihood": _scalar(outputs[2]),
        "log_likelihood_gradient": _vector(loglik_grad),
        "log_likelihood_gradient_finite_components": [bool(x) for x in tf.math.is_finite(loglik_grad).numpy().tolist()],
        "prior": _scalar(outputs[4]),
        "prior_gradient": _vector(prior_grad),
        "prior_gradient_finite_components": [bool(x) for x in tf.math.is_finite(prior_grad).numpy().tolist()],
        "value_gradient_connected": _bool(outputs[6]),
        "log_likelihood_gradient_connected": _bool(outputs[7]),
        "prior_gradient_connected": _bool(outputs[8]),
        "final_particles_sum_gradient_connected": _bool(outputs[9]),
        "final_particles_sum_gradient": _vector(final_particles_grad),
        "final_particles_sum_gradient_finite_components": [
            bool(x) for x in tf.math.is_finite(final_particles_grad).numpy().tolist()
        ],
        "route_invocations": _int(outputs[11]),
        "active_resampling_mask_count": _int(outputs[12]),
        "max_factor_marginal_residual": _scalar(outputs[13]),
        "max_induced_row_residual": _scalar(outputs[14]),
        "max_induced_column_residual": _scalar(outputs[15]),
        "projection_iterations_used_max": _int(outputs[16]),
        "finite_factors": _bool(outputs[17]),
        "finite_particles": _bool(outputs[18]),
        "nonnegative_factors": _bool(outputs[19]),
        "positive_g": _bool(outputs[20]),
        "route_outputs_finite": _bool(outputs[21]),
        "value_and_gradient_finite": _bool(outputs[22]),
        "loglik_and_gradient_finite": _bool(outputs[23]),
        "prior_and_gradient_finite": _bool(outputs[24]),
        "final_particles_sum_gradient_finite": _bool(outputs[25]),
    }


def _hard_vetoes(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[str]:
    vetoes = []
    for row in rows:
        label = f"{row['seed']}:{row['probe_label']}"
        if args.expect_device_kind == "gpu" and "GPU" not in row["output_device"].upper():
            vetoes.append(f"{label}:not_gpu_output")
        if not row["route_outputs_finite"]:
            vetoes.append(f"{label}:route_outputs_nonfinite")
        if not row["value_and_gradient_finite"]:
            vetoes.append(f"{label}:value_gradient_nonfinite")
        if not row["loglik_and_gradient_finite"]:
            vetoes.append(f"{label}:loglik_gradient_nonfinite")
        if not row["log_likelihood_gradient_connected"]:
            vetoes.append(f"{label}:loglik_gradient_disconnected")
        if not row["prior_and_gradient_finite"]:
            vetoes.append(f"{label}:prior_gradient_nonfinite")
        if not row["final_particles_sum_gradient_connected"]:
            vetoes.append(f"{label}:final_particles_gradient_disconnected")
        if not row["finite_factors"]:
            vetoes.append(f"{label}:nonfinite_factors")
        if not row["finite_particles"]:
            vetoes.append(f"{label}:nonfinite_particles")
        if not row["nonnegative_factors"]:
            vetoes.append(f"{label}:negative_factors")
        if not row["positive_g"]:
            vetoes.append(f"{label}:nonpositive_g")
    return vetoes


def build_result(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    started_at = _dt.datetime.now(tz=_dt.UTC).isoformat()
    dtype, config = _configure(args)
    seed_probe_pairs = _seed_probe_pairs(args)
    rows = []
    fixtures: dict[int, lgssm_gate.LGSSMGateFixture] = {}
    compiled_by_seed = {}
    probe_specs = _probe_spec_by_label(args, dtype)
    with tf.device(args.device):
        for seed, label in seed_probe_pairs:
            if label not in probe_specs:
                raise ValueError(f"unknown probe label: {label}")
            if seed not in fixtures:
                fixtures[seed] = lgssm_gate.build_lgssm_gate_fixture(args.case_id, seed, args)
                compiled_by_seed[seed] = _diagnostic_function(fixtures[seed], args, dtype)
            theta = probe_specs[label]["theta"]
            outputs = compiled_by_seed[seed](theta)
            rows.append(_row(seed, label, theta, outputs))
    hard_vetoes = _hard_vetoes(rows, args)
    ended_at = _dt.datetime.now(tz=_dt.UTC).isoformat()
    run_manifest = {
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
        "p02_result_path": P02_RESULT_PATH,
        "gpu_trust_basis": GPU_TRUST_BASIS,
        "device_scope": args.device_scope,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "device": args.device,
        "expect_device_kind": args.expect_device_kind,
        "jit_compile": bool(args.jit_compile),
        "case_id": args.case_id,
        "seed_probe_pairs": [{"seed": seed, "probe_label": label} for seed, label in seed_probe_pairs],
        **config,
    }
    return {
        "schema_version": "low_rank_ledh_gradient_nonfinite_diagnostic.v1",
        "phase": args.phase_id,
        "status": "PASS" if not hard_vetoes else "FAIL",
        "evidence_class": "cpu_hidden_repair_debug_only"
        if args.device_scope == "cpu"
        else GPU_TRUST_BASIS,
        "gpu_trust_basis": GPU_TRUST_BASIS if args.device_scope != "cpu" else None,
        "question": "localize low-rank route nonfinite posterior gradients on P02 failing probes",
        "candidate": calibration._candidate_settings(args),
        "hard_vetoes": hard_vetoes,
        "rows": rows,
        "run_manifest": run_manifest,
        "nonclaims": list(NONCLAIMS),
    }


def write_markdown(result: dict[str, Any], path: Path, json_path: Path | None = None) -> None:
    lines = [
        "# Low-Rank Gradient Nonfinite Diagnostic",
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
            "| Seed | Probe | Value finite | Loglik grad finite | Prior grad finite | Final-particle grad finite | Route outputs finite | Factor residual | Row residual | Col residual | Iterations |",
            "| ---: | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in result["rows"]:
        lines.append(
            "| {seed} | `{probe}` | `{value}` | `{loglik}` | `{prior}` | `{particles}` | `{outputs}` | {factor} | {row_resid} | {col_resid} | {iters} |".format(
                seed=row["seed"],
                probe=row["probe_label"],
                value=row["value_and_gradient_finite"],
                loglik=row["loglik_and_gradient_finite"],
                prior=row["prior_and_gradient_finite"],
                particles=row["final_particles_sum_gradient_finite"],
                outputs=row["route_outputs_finite"],
                factor=row["max_factor_marginal_residual"],
                row_resid=row["max_induced_row_residual"],
                col_resid=row["max_induced_column_residual"],
                iters=row["projection_iterations_used_max"],
            )
        )
    lines.extend(["", "## Non-Claims", ""])
    lines.extend(f"- {item}" for item in result["nonclaims"])
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
        print(json.dumps({"status": result["status"], "hard_vetoes": result["hard_vetoes"]}, indent=2))


if __name__ == "__main__":
    main()
