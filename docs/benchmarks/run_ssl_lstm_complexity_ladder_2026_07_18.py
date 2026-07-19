#!/usr/bin/env python3
"""Bounded SSL-LSTM structural-adapter complexity ladder."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import resource
import shlex
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import tensorflow as tf


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bayesfilter.nonlinear.sigma_points_tf import tf_svd_sigma_point_log_likelihood  # noqa: E402
from bayesfilter.nonlinear.ssl_lstm_protocol import SSLLSTMStaticConfig  # noqa: E402
from bayesfilter.nonlinear.ssl_lstm_sgqf_ukf_adapters import (  # noqa: E402
    make_ssl_lstm_svd_ukf_components,
    ssl_lstm_observation,
    ssl_lstm_observation_parameter_derivative,
    ssl_lstm_observation_state_jacobian,
    ssl_lstm_parameter_slices,
    ssl_lstm_transition,
    ssl_lstm_transition_parameter_derivative,
    ssl_lstm_transition_state_jacobian,
    tf_ssl_lstm_svd_ukf_score,
    unpack_ssl_lstm_parameters,
)


PLAN_PATH = Path("docs/plans/bayesfilter-ssl-lstm-complexity-ladder-plan-2026-07-18.md")
RESULT_PATH = Path("docs/plans/bayesfilter-ssl-lstm-complexity-ladder-result-2026-07-18.md")
SCRIPT_PATH = Path(__file__).resolve().relative_to(ROOT)
RUNGS = (1, 2, 5, 10, 20)
HORIZON = 3
TRANSITION_RTOL = 2.0e-5
TRANSITION_ATOL = 2.0e-6
OBSERVATION_RTOL = 1.0e-7
OBSERVATION_ATOL = 1.0e-8
SCORE_RTOL = 5.0e-3
SCORE_ATOL = 8.0e-4
FD_STEP = 1.0e-5
MEMORY_SOFT_BYTES = 2 * 1024**3
EVIDENCE_PATH = PLAN_PATH.as_posix()


class ComplexityLadderError(RuntimeError):
    """Raised when a rung violates the engineering evidence contract."""


def _absolute(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _sha256(path: Path) -> str:
    return hashlib.sha256(_absolute(path).read_bytes()).hexdigest()


def _report_path(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _git(*args: str) -> str:
    return subprocess.run(
        ("git", *args), cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout


def closed_form_parameter_dim(q: int) -> int:
    if isinstance(q, bool) or int(q) != q or q <= 0:
        raise ValueError("q must be a positive integer")
    return 9 * int(q) ** 2 + 13 * int(q) + 2


def make_config(q: int) -> SSLLSTMStaticConfig:
    return SSLLSTMStaticConfig(
        horizon=HORIZON,
        latent_dim=q,
        hidden_dim=q,
        observation_dim=1,
    )


def make_fixture_theta(config: SSLLSTMStaticConfig) -> tf.Tensor:
    """Construct a deterministic, nondegenerate TensorFlow fixture."""

    p = config.parameter_dim
    base = 0.025 * tf.sin(tf.cast(tf.range(p), tf.float64) * 0.371)
    slices = ssl_lstm_parameter_slices(config)
    n = config.augmented_state_dim
    k = config.latent_dim
    updates: list[tuple[int, tf.Tensor]] = []
    for index in range(n):
        updates.append(
            (
                slices.initial_std_start + index,
                tf.constant(-0.85 + 0.011 * index, tf.float64),
            )
        )
    for index in range(k):
        updates.append(
            (
                slices.process_std_start + index,
                tf.constant(0.55 + 0.017 * index, tf.float64),
            )
        )
    updates.append((slices.observation_std_start, tf.constant(-0.2, tf.float64)))
    indices = tf.constant([[index] for index, _ in updates], tf.int32)
    values = tf.stack([value for _, value in updates])
    theta = tf.tensor_scatter_nd_update(base, indices, values)
    return tf.ensure_shape(theta, [p])


def make_points(config: SSLLSTMStaticConfig) -> tf.Tensor:
    n = config.augmented_state_dim
    return tf.reshape(tf.linspace(tf.constant(-0.2, tf.float64), tf.constant(0.3, tf.float64), 2 * n), [2, n])


def observations() -> tf.Tensor:
    return tf.constant([[0.12], [-0.03], [0.08]], tf.float64)


def _parameter_indices(config: SSLLSTMStaticConfig) -> tuple[int, ...]:
    slices = ssl_lstm_parameter_slices(config)
    candidates = (
        0,
        slices.lstm_recurrent_start,
        slices.lstm_bias_start,
        slices.latent_weight_start,
        slices.latent_bias_start,
        slices.observation_weight_start,
        slices.observation_bias_start,
        slices.initial_mean_start,
        slices.initial_std_start,
        slices.process_std_start,
        slices.observation_std_start,
    )
    return tuple(dict.fromkeys(int(value) for value in candidates))


def _transition_parameter_indices(config: SSLLSTMStaticConfig) -> tuple[int, ...]:
    slices = ssl_lstm_parameter_slices(config)
    return tuple(
        dict.fromkeys(
            (
                0,
                slices.lstm_recurrent_start,
                slices.lstm_bias_start,
                slices.latent_weight_start,
                slices.latent_bias_start,
            )
        )
    )


def _central_difference_vector(
    theta: tf.Tensor,
    indices: tuple[int, ...],
    value_fn: Any,
    *,
    step: float,
) -> tf.Tensor:
    rows = []
    for index in indices:
        direction = tf.one_hot(index, int(theta.shape[0]), dtype=tf.float64)
        rows.append((value_fn(theta + step * direction) - value_fn(theta - step * direction)) / (2.0 * step))
    return tf.stack(rows)


def _central_difference_state(
    points: tf.Tensor,
    value_fn: Any,
    *,
    step: float,
) -> tf.Tensor:
    n = int(points.shape[1])
    columns = []
    for index in range(n):
        direction = tf.one_hot(index, n, dtype=tf.float64)[tf.newaxis, :]
        columns.append((value_fn(points + step * direction) - value_fn(points - step * direction)) / (2.0 * step))
    return tf.stack(columns, axis=-1)


def _max_abs(left: tf.Tensor, right: tf.Tensor) -> float:
    return float(tf.reduce_max(tf.abs(left - right)).numpy())


def _assert_close(
    left: tf.Tensor,
    right: tf.Tensor,
    *,
    rtol: float,
    atol: float,
    label: str,
) -> float:
    if not bool(tf.reduce_all(tf.math.is_finite(left)).numpy()) or not bool(
        tf.reduce_all(tf.math.is_finite(right)).numpy()
    ):
        raise ComplexityLadderError(f"{label} contains non-finite values")
    tolerance = atol + rtol * tf.abs(right)
    residual = tf.abs(left - right)
    if not bool(tf.reduce_all(residual <= tolerance).numpy()):
        raise ComplexityLadderError(
            f"{label} mismatch: max_abs={float(tf.reduce_max(residual).numpy())}"
        )
    return float(tf.reduce_max(residual).numpy())


def _memory_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value * 1024


def run_rung(q: int) -> dict[str, Any]:
    started = time.perf_counter()
    config = make_config(q)
    if config.parameter_dim != closed_form_parameter_dim(q):
        raise ComplexityLadderError("source-derived parameter count disagrees with formula")
    if config.augmented_state_dim != 3 * q:
        raise ComplexityLadderError("augmented-state formula mismatch")
    theta = make_fixture_theta(config)
    params = unpack_ssl_lstm_parameters(theta, config)
    points = make_points(config)
    slices = ssl_lstm_parameter_slices(config)

    shapes = {
        "lstm_input": list(params.lstm_input.shape),
        "lstm_recurrent": list(params.lstm_recurrent.shape),
        "lstm_bias": list(params.lstm_bias.shape),
        "latent_weight": list(params.latent_weight.shape),
        "observation_weight": list(params.observation_weight.shape),
        "initial_covariance": list(params.initial_covariance.shape),
        "innovation_covariance": list(params.ukf_innovation_covariance.shape),
    }
    expected_shapes = {
        "lstm_input": [4, q, q],
        "lstm_recurrent": [4, q, q],
        "lstm_bias": [4, q],
        "latent_weight": [q, q],
        "observation_weight": [1, q],
        "initial_covariance": [3 * q, 3 * q],
        "innovation_covariance": [q, q],
    }
    if shapes != expected_shapes:
        raise ComplexityLadderError(f"parameter block shape mismatch: {shapes}")
    covariance_diagonals = tf.concat(
        (
            tf.linalg.diag_part(params.initial_covariance),
            tf.linalg.diag_part(params.ukf_innovation_covariance),
            tf.linalg.diag_part(params.observation_covariance),
        ),
        axis=0,
    )
    if not bool(tf.reduce_all(covariance_diagonals > 0.0).numpy()):
        raise ComplexityLadderError("fixture covariance diagonal is not positive")

    transition = ssl_lstm_transition(params, points)
    observation = ssl_lstm_observation(params, points)
    state_jacobian = ssl_lstm_transition_state_jacobian(params, points)
    observation_state_jacobian = ssl_lstm_observation_state_jacobian(params, points)
    parameter_derivative = ssl_lstm_transition_parameter_derivative(params, points)
    observation_parameter_derivative = ssl_lstm_observation_parameter_derivative(params, points)

    fd_state = _central_difference_state(
        points, lambda values: ssl_lstm_transition(params, values), step=1.0e-6
    )
    transition_state_error = _assert_close(
        state_jacobian,
        fd_state,
        rtol=TRANSITION_RTOL,
        atol=TRANSITION_ATOL,
        label="transition state Jacobian",
    )
    transition_indices = _transition_parameter_indices(config)
    fd_transition_parameter = _central_difference_vector(
        theta,
        transition_indices,
        lambda values: ssl_lstm_transition(unpack_ssl_lstm_parameters(values, config), points),
        step=1.0e-6,
    )
    transition_parameter_error = _assert_close(
        tf.gather(parameter_derivative, transition_indices),
        fd_transition_parameter,
        rtol=TRANSITION_RTOL,
        atol=TRANSITION_ATOL,
        label="transition parameter derivative",
    )
    fd_observation_state = _central_difference_state(
        points, lambda values: ssl_lstm_observation(params, values), step=1.0e-6
    )
    observation_state_error = _assert_close(
        observation_state_jacobian,
        fd_observation_state,
        rtol=OBSERVATION_RTOL,
        atol=OBSERVATION_ATOL,
        label="observation state Jacobian",
    )
    observation_indices = (
        slices.observation_weight_start,
        slices.observation_bias_start,
    )
    fd_observation_parameter = _central_difference_vector(
        theta,
        observation_indices,
        lambda values: ssl_lstm_observation(unpack_ssl_lstm_parameters(values, config), points),
        step=1.0e-6,
    )
    observation_parameter_error = _assert_close(
        tf.gather(observation_parameter_derivative, observation_indices),
        fd_observation_parameter,
        rtol=OBSERVATION_RTOL,
        atol=OBSERVATION_ATOL,
        label="observation parameter derivative",
    )

    @tf.function(jit_compile=True, reduce_retracing=True)
    def score_program(theta_value: tf.Tensor, y: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        result, _ = tf_ssl_lstm_svd_ukf_score(
            y,
            theta_value,
            config,
            evidence_path=EVIDENCE_PATH,
            spectral_gap_tolerance=tf.constant(1.0e-12, tf.float64),
        )
        return result.log_likelihood, result.score, result.trace[0]["filtered_covariance"]

    @tf.function(jit_compile=True, reduce_retracing=True)
    def value_program(theta_value: tf.Tensor, y: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        components = make_ssl_lstm_svd_ukf_components(
            theta_value, config, evidence_path=EVIDENCE_PATH
        )
        value, means, covariances, diagnostics = tf_svd_sigma_point_log_likelihood(
            y,
            components.model,
            rule="unscented",
            innovation_floor=tf.constant(1.0e-12, tf.float64),
            return_filtered=True,
        )
        assert means is not None and covariances is not None
        return value, means, covariances, diagnostics["point_count"]

    y = observations()
    log_likelihood, score, analytic_covariance = score_program(theta, y)
    repeated_log_likelihood, repeated_score, _ = score_program(theta, y)
    value, filtered_means, filtered_covariances, point_count = value_program(theta, y)
    if not all(
        bool(tf.reduce_all(tf.math.is_finite(item)).numpy())
        for item in (
            transition,
            observation,
            log_likelihood,
            score,
            value,
            filtered_means,
            filtered_covariances,
        )
    ):
        raise ComplexityLadderError("non-finite transition/filter/score output")
    repeat_value_error = _assert_close(
        log_likelihood,
        repeated_log_likelihood,
        rtol=0.0,
        atol=1.0e-12,
        label="repeated score value",
    )
    repeat_score_error = _assert_close(
        score,
        repeated_score,
        rtol=0.0,
        atol=1.0e-12,
        label="repeated analytic score",
    )
    value_score_parity_error = _assert_close(
        log_likelihood,
        value,
        rtol=1.0e-8,
        atol=1.0e-8,
        label="value/score likelihood parity",
    )
    covariance_trace_error = _max_abs(analytic_covariance, filtered_covariances[-1])
    eigenvalues = tf.linalg.eigvalsh(filtered_covariances)
    minimum_eigenvalue = float(tf.reduce_min(eigenvalues).numpy())
    if minimum_eigenvalue < -1.0e-10:
        raise ComplexityLadderError("filtered covariance is not PSD")

    score_indices = _parameter_indices(config)
    fd_score = _central_difference_vector(
        theta,
        score_indices,
        lambda values: value_program(values, y)[0],
        step=FD_STEP,
    )
    score_error = _assert_close(
        tf.gather(score, score_indices),
        fd_score,
        rtol=SCORE_RTOL,
        atol=SCORE_ATOL,
        label="analytic score finite difference",
    )

    memory = _memory_bytes()
    gpu_memory = None
    try:
        gpu_memory = {
            key: int(value)
            for key, value in tf.config.experimental.get_memory_info("GPU:0").items()
        }
    except (ValueError, tf.errors.OpError):
        pass
    wall = time.perf_counter() - started
    return {
        "q": q,
        "status": "PASSED",
        "latent_dim": q,
        "hidden_dim": q,
        "observation_dim": 1,
        "augmented_state_dim": config.augmented_state_dim,
        "parameter_dim": config.parameter_dim,
        "parameter_formula_dim": closed_form_parameter_dim(q),
        "sigma_point_count": int(point_count.numpy()),
        "shapes": shapes,
        "log_likelihood": float(log_likelihood.numpy()),
        "score_l2_norm": float(tf.linalg.norm(score).numpy()),
        "minimum_filtered_covariance_eigenvalue": minimum_eigenvalue,
        "maximum_errors": {
            "transition_state_jacobian": transition_state_error,
            "transition_parameter_derivative": transition_parameter_error,
            "observation_state_jacobian": observation_state_error,
            "observation_parameter_derivative": observation_parameter_error,
            "analytic_score_finite_difference": score_error,
            "repeated_value": repeat_value_error,
            "repeated_score": repeat_score_error,
            "value_score_parity": value_score_parity_error,
            "terminal_covariance_path_difference_explanatory": covariance_trace_error,
        },
        "finite_difference_parameter_indices": list(score_indices),
        "compiled_trace_counts": {
            "score": score_program.experimental_get_tracing_count(),
            "value": value_program.experimental_get_tracing_count(),
        },
        "wall_time_seconds": wall,
        "process_peak_rss_bytes": memory,
        "memory_soft_warning": memory > MEMORY_SOFT_BYTES,
        "gpu_memory_bytes": gpu_memory,
        "hard_vetoes": [],
        "nonclaims": [
            "engineering and numerical adapter evidence only",
            "no posterior, HMC, NeuTra training, or model-adequacy claim",
        ],
    }


def _plain(value: Any) -> Any:
    if isinstance(value, tf.Tensor):
        return _plain(value.numpy())
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _write(path: Path, payload: dict[str, Any]) -> None:
    target = _absolute(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(_plain(payload), sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _run_rung_subprocess(
    q: int,
    *,
    output_dir: Path,
    timeout_seconds: float,
) -> dict[str, Any]:
    """Run one rung in a killable process so the ladder cap is enforceable."""

    child_path = _absolute(output_dir / f"rung-q{q}-child.json")
    log_path = _absolute(output_dir / f"rung-q{q}.log")
    child_path.parent.mkdir(parents=True, exist_ok=True)
    command = (
        sys.executable,
        str(_absolute(SCRIPT_PATH)),
        "--single-rung",
        str(q),
        "--single-output",
        str(child_path),
    )
    try:
        with log_path.open("w", encoding="utf-8") as log_handle:
            completed = subprocess.run(
                command,
                cwd=ROOT,
                check=False,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=max(float(timeout_seconds), 1.0),
            )
    except subprocess.TimeoutExpired:
        return {
            "q": q,
            "status": "FAILED",
            "wall_time_seconds": float(timeout_seconds),
            "hard_vetoes": ["per_rung_subprocess_timeout_at_remaining_wall_cap"],
            "failure_classification": "resource_continuation_veto_not_numerical_or_research_direction_rejection",
            "log_path": _report_path(log_path),
        }
    if not child_path.exists():
        return {
            "q": q,
            "status": "FAILED",
            "hard_vetoes": [f"rung_subprocess_exit_{completed.returncode}_without_receipt"],
            "failure_classification": "engineering_rung_failure_not_research_direction_rejection",
            "log_path": _report_path(log_path),
        }
    child = json.loads(child_path.read_text(encoding="utf-8"))
    row = child["rung"]
    row["log_path"] = _report_path(log_path)
    row["child_receipt_path"] = _report_path(child_path)
    return row


def run(output_dir: Path, *, wall_cap_seconds: float) -> dict[str, Any]:
    started_at = datetime.now(UTC).isoformat()
    started = time.perf_counter()
    results = []
    decision = "COMPLEXITY_LADDER_PASSED_ALL_RUNGS"
    continuation_veto = None
    for q in RUNGS:
        if time.perf_counter() - started >= wall_cap_seconds:
            decision = "COMPLEXITY_LADDER_STOPPED_RESOURCE_CAP"
            continuation_veto = "wall_cap_reached_before_next_rung"
            break
        remaining = wall_cap_seconds - (time.perf_counter() - started)
        row = _run_rung_subprocess(
            q,
            output_dir=output_dir,
            timeout_seconds=remaining,
        )
        if row["status"] != "PASSED":
            if "timeout" in " ".join(row.get("hard_vetoes", ())):
                decision = "COMPLEXITY_LADDER_STOPPED_RESOURCE_CAP"
            else:
                decision = "COMPLEXITY_LADDER_STOPPED_HARD_VETO"
            continuation_veto = row["hard_vetoes"][0]
        results.append(row)
        partial = {
            "schema": "bayesfilter.ssl_lstm.complexity_ladder.rung.v1",
            "decision": "RUNG_PASSED" if row["status"] == "PASSED" else "RUNG_FAILED",
            "rung": row,
        }
        _write(output_dir / f"rung-q{q}.json", partial)
        if row["status"] != "PASSED":
            break

    wall = time.perf_counter() - started
    payload = {
        "schema": "bayesfilter.ssl_lstm.complexity_ladder.v1",
        "status": "PASSED" if decision == "COMPLEXITY_LADDER_PASSED_ALL_RUNGS" else "STOPPED",
        "decision": decision,
        "rung_semantics": {
            "q_values": list(RUNGS),
            "latent_dim": "q",
            "hidden_dim": "q",
            "observation_dim": 1,
            "augmented_state_dim": "3q",
            "parameter_dim": "9q^2+13q+2",
        },
        "rungs": results,
        "continuation_veto": continuation_veto,
        "source_bindings": {
            "plan": {"path": PLAN_PATH.as_posix(), "sha256": _sha256(PLAN_PATH)},
            "runner": {"path": SCRIPT_PATH.as_posix(), "sha256": _sha256(SCRIPT_PATH)},
            "protocol": {"path": "bayesfilter/nonlinear/ssl_lstm_protocol.py", "sha256": _sha256(Path("bayesfilter/nonlinear/ssl_lstm_protocol.py"))},
            "adapter": {"path": "bayesfilter/nonlinear/ssl_lstm_sgqf_ukf_adapters.py", "sha256": _sha256(Path("bayesfilter/nonlinear/ssl_lstm_sgqf_ukf_adapters.py"))},
        },
        "run_manifest": {
            "command": " ".join(shlex.quote(item) for item in (sys.executable, *sys.argv)),
            "cwd": str(ROOT),
            "git_commit": _git("rev-parse", "HEAD").strip(),
            "git_dirty": bool(_git("status", "--porcelain").strip()),
            "python": platform.python_version(),
            "tensorflow": tf.__version__,
            "tensorflow_probability": __import__("tensorflow_probability").__version__,
            "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "physical_gpus": [device.name for device in tf.config.list_physical_devices("GPU")],
            "logical_gpus": [device.name for device in tf.config.list_logical_devices("GPU")],
            "jit_compile": True,
            "tf32_enabled": bool(tf.config.experimental.tensor_float_32_execution_enabled()),
            "trust_basis": "owner_designated_managed_session_visible_gpu_trusted",
            "random_seeds": "none; deterministic trigonometric fixture",
            "started_at_utc": started_at,
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "wall_time_seconds": wall,
            "wall_cap_seconds": wall_cap_seconds,
        },
        "inference_status": {
            "hard_veto_screen": "see each rung",
            "statistically_supported_ranking": "none",
            "descriptive_only_differences": "runtime, memory, score norms, and scaling ratios",
            "default_readiness": "not established",
            "next_evidence_needed": "separate q>1 training, sampler, and predictive-validation programs",
        },
        "nonclaims": [
            "no q>1 posterior samples, NeuTra transport, or HMC run",
            "no high-dimensional posterior correctness or model-adequacy claim",
            "single deterministic fixture does not rank dimensions or algorithms",
        ],
    }
    _write(output_dir / "complexity-ladder-result.json", payload)
    return payload


def _require_gpu() -> None:
    physical = tf.config.list_physical_devices("GPU")
    if not physical:
        raise ComplexityLadderError("trusted GPU is required")
    for gpu in physical:
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.experimental.enable_tensor_float_32_execution(True)
    if not tf.config.list_logical_devices("GPU"):
        raise ComplexityLadderError("logical GPU is unavailable")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--wall-cap-seconds", type=float, default=600.0)
    parser.add_argument("--single-rung", type=int, choices=RUNGS)
    parser.add_argument("--single-output", type=Path)
    args = parser.parse_args(argv)
    if not math.isfinite(args.wall_cap_seconds) or args.wall_cap_seconds <= 0.0:
        raise ComplexityLadderError("wall cap must be positive and finite")
    if args.single_rung is not None:
        if args.single_output is None:
            raise ComplexityLadderError("--single-output is required with --single-rung")
        _require_gpu()
        try:
            with tf.device("/GPU:0"):
                row = run_rung(args.single_rung)
        except Exception as exc:
            row = {
                "q": args.single_rung,
                "status": "FAILED",
                "hard_vetoes": [f"{type(exc).__name__}: {exc}"],
                "failure_classification": "engineering_or_numerical_rung_failure_not_research_direction_rejection",
            }
        _write(
            args.single_output,
            {
                "schema": "bayesfilter.ssl_lstm.complexity_ladder.child.v1",
                "decision": "RUNG_PASSED" if row["status"] == "PASSED" else "RUNG_FAILED",
                "rung": row,
            },
        )
        print(json.dumps({"q": args.single_rung, "status": row["status"]}, sort_keys=True))
        return 0 if row["status"] == "PASSED" else 2
    if args.output_dir is None:
        raise ComplexityLadderError("--output-dir is required for the ladder supervisor")
    payload = run(args.output_dir, wall_cap_seconds=float(args.wall_cap_seconds))
    print(json.dumps({"decision": payload["decision"], "rungs": [(row["q"], row["status"]) for row in payload["rungs"]], "wall_time_seconds": payload["run_manifest"]["wall_time_seconds"]}, sort_keys=True))
    return 0 if payload["status"] == "PASSED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
