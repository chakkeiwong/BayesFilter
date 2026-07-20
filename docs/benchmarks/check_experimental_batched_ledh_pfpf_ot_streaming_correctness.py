"""Correctness gate for streaming batched LEDH-PFPF-OT.

This gate compares the new streaming value path against the existing
fixed-branch baseline on deterministic tiny fixtures.  It is a correctness
artifact, not a performance benchmark.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import inspect
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


_PRE_PARSER = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
_PRE_PARSER.add_argument("--device-scope", choices=("cpu", "visible"), default="cpu")
_PRE_PARSER.add_argument("--cuda-visible-devices", default=None)
_PRE_ARGS, _UNKNOWN = _PRE_PARSER.parse_known_args()
if _PRE_ARGS.cuda_visible_devices is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = _PRE_ARGS.cuda_visible_devices
elif _PRE_ARGS.device_scope == "cpu":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import tensorflow as tf

from experiments.dpf_implementation.tf_tfp.filters import (
    experimental_batched_ledh_pfpf_ot_streaming_tf,
)
from experiments.dpf_implementation.tf_tfp.filters import experimental_batched_ledh_pfpf_ot_tf
from experiments.dpf_implementation.tf_tfp.filters.experimental_batched_ledh_pfpf_ot_streaming_tf import (
    streaming_batched_ledh_pfpf_ot_value_and_score_tf,
    streaming_batched_ledh_pfpf_ot_value_core_tf,
)
from experiments.dpf_implementation.tf_tfp.resampling import annealed_transport_tf
from experiments.dpf_implementation.tf_tfp.filters.experimental_batched_ledh_pfpf_ot_tf import (
    SCALAR_PARITY_ATOL,
    SCALAR_PARITY_RTOL,
    batched_ledh_pfpf_ot_value_core_tf,
)


DTYPE = tf.float64
experimental_batched_ledh_pfpf_ot_tf.DTYPE = DTYPE
experimental_batched_ledh_pfpf_ot_streaming_tf.DTYPE = DTYPE
annealed_transport_tf.DTYPE = DTYPE

NONCLAIMS = (
    "streaming DPF correctness gate only",
    "production/default target by owner directive",
    "no GPU performance claim",
    "no posterior validity claim",
    "no active-transport finite-difference score equivalence claim",
    "no HMC/NeuTra readiness claim",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--time-steps", type=int, default=3)
    parser.add_argument("--num-particles", type=int, default=4)
    parser.add_argument(
        "--transport-policy",
        choices=("active", "no-resampling"),
        default="active",
    )
    parser.add_argument("--sinkhorn-iterations", type=int, default=8)
    parser.add_argument("--row-chunk-size", type=int, default=2)
    parser.add_argument("--col-chunk-size", type=int, default=2)
    parser.add_argument("--particle-chunk-size", type=int, default=2)
    parser.add_argument("--device", default="/CPU:0")
    parser.add_argument("--device-scope", choices=("cpu", "visible"), default=_PRE_ARGS.device_scope)
    parser.add_argument("--cuda-visible-devices", default=_PRE_ARGS.cuda_visible_devices)
    parser.add_argument("--expect-device-kind", choices=("any", "cpu", "gpu"), default="cpu")
    parser.add_argument("--value-atol", type=float, default=SCALAR_PARITY_ATOL)
    parser.add_argument("--value-rtol", type=float, default=SCALAR_PARITY_RTOL)
    parser.add_argument("--score-fd-step", type=float, default=1.0e-5)
    parser.add_argument("--score-fd-atol", type=float, default=2.0e-4)
    parser.add_argument("--score-fd-rtol", type=float, default=2.0e-4)
    parser.add_argument("--skip-jit-smoke", action="store_true")
    parser.add_argument("--skip-score-fd", action="store_true")
    parser.add_argument("--output", required=True)
    parser.add_argument("--markdown-output", default=None)
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if args.time_steps <= 0:
        raise ValueError("time_steps must be positive")
    if args.num_particles <= 1:
        raise ValueError("num_particles must be greater than one")
    if args.sinkhorn_iterations <= 0:
        raise ValueError("sinkhorn_iterations must be positive")
    if args.row_chunk_size <= 0 or args.col_chunk_size <= 0 or args.particle_chunk_size <= 0:
        raise ValueError("chunk sizes must be positive")
    return args


def _configure_gpus() -> tuple[list[str], list[str]]:
    physical_gpus = tf.config.list_physical_devices("GPU")
    for gpu in physical_gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
    logical_gpus = tf.config.list_logical_devices("GPU")
    return ([str(device) for device in physical_gpus], [str(device) for device in logical_gpus])


def _git_commit() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"
    return completed.stdout.strip()


def _fixture(args: argparse.Namespace, *, transport_policy: str | None = None) -> dict[str, tf.Tensor]:
    policy = transport_policy or args.transport_policy
    batch = np.arange(args.batch_size, dtype=np.float64)
    particles = np.linspace(-0.30, 0.45, args.num_particles, dtype=np.float64)
    initial_particles = particles[None, :, None] + 0.002 * batch[:, None, None]
    transition_matrix = (0.82 + 0.0002 * batch)[:, None, None]
    time_grid = np.arange(args.time_steps, dtype=np.float64)
    offsets = np.linspace(-0.04, 0.05, args.num_particles, dtype=np.float64)
    pre_flow_particles = (
        transition_matrix[:, None, :, :] * initial_particles[:, None, :, :]
        + offsets[None, None, :, None]
        + 0.002 * time_grid[None, :, None, None]
    )
    observations = (0.03 + 0.02 * np.sin(0.7 * time_grid))[:, None]
    if policy == "active":
        mask = (np.arange(args.time_steps)[None, :] % 2) == 1
        fixed_resampling_mask = np.broadcast_to(mask, (args.batch_size, args.time_steps)).copy()
    else:
        fixed_resampling_mask = np.zeros((args.batch_size, args.time_steps), dtype=bool)
    return {
        "observations": tf.constant(observations, dtype=DTYPE),
        "initial_particles": tf.constant(initial_particles, dtype=DTYPE),
        "pre_flow_particles": tf.constant(pre_flow_particles, dtype=DTYPE),
        "fixed_resampling_mask": tf.constant(fixed_resampling_mask, dtype=tf.bool),
        "transition_matrix": tf.constant(transition_matrix, dtype=DTYPE),
        "transition_covariance": tf.constant(
            np.tile([[[0.16]]], (args.batch_size, 1, 1)),
            dtype=DTYPE,
        ),
        "observation_covariance": tf.constant(
            np.tile([[[0.25]]], (args.batch_size, 1, 1)),
            dtype=DTYPE,
        ),
    }


def _observation(points: tf.Tensor) -> tf.Tensor:
    return points


def _observation_jacobian(points: tf.Tensor) -> tf.Tensor:
    batch_size = points.shape[0]
    num_particles = points.shape[1]
    if batch_size is None or num_particles is None:
        raise ValueError("correctness fixture requires static batch and particle dimensions")
    return tf.ones([batch_size, num_particles, 1, 1], dtype=DTYPE)


def _observation_residual(h_ref: tf.Tensor, observation: tf.Tensor) -> tf.Tensor:
    return observation[None, None, :] - h_ref


def _make_transition_log_density(
    transition_matrix: tf.Tensor,
    transition_covariance: tf.Tensor,
):
    def _transition_log_density(
        x_next: tf.Tensor,
        x_prev: tf.Tensor,
        _time_index: tf.Tensor,
    ) -> tf.Tensor:
        del _time_index
        mean = tf.einsum("bnj,bdj->bnd", x_prev, transition_matrix)
        residual = x_next - mean
        variance = transition_covariance[:, 0, 0]
        quad = residual[:, :, 0] * residual[:, :, 0] / variance[:, None]
        return -0.5 * (
            tf.math.log(tf.constant(2.0 * np.pi, dtype=DTYPE))
            + tf.math.log(variance)[:, None]
            + quad
        )

    return _transition_log_density


def _make_observation_log_density(observation_covariance: tf.Tensor):
    def _observation_log_density(
        x: tf.Tensor,
        observation: tf.Tensor,
        _time_index: tf.Tensor,
    ) -> tf.Tensor:
        del _time_index
        variance = observation_covariance[:, 0, 0]
        residual = x[:, :, 0] - observation[0]
        return -0.5 * (
            tf.math.log(tf.constant(2.0 * np.pi, dtype=DTYPE))
            + tf.math.log(variance)[:, None]
            + residual * residual / variance[:, None]
        )

    return _observation_log_density


def _baseline_value(fixture: dict[str, tf.Tensor], args: argparse.Namespace):
    return batched_ledh_pfpf_ot_value_core_tf(
        observations=fixture["observations"],
        initial_particles=fixture["initial_particles"],
        pre_flow_particles=fixture["pre_flow_particles"],
        fixed_resampling_mask=fixture["fixed_resampling_mask"],
        transition_matrix=fixture["transition_matrix"],
        transition_covariance=fixture["transition_covariance"],
        observation_covariance=fixture["observation_covariance"],
        observation_fn=_observation,
        observation_jacobian_fn=_observation_jacobian,
        observation_residual_fn=_observation_residual,
        transition_log_density_fn=_make_transition_log_density(
            fixture["transition_matrix"],
            fixture["transition_covariance"],
        ),
        observation_log_density_fn=_make_observation_log_density(
            fixture["observation_covariance"],
        ),
        sinkhorn_iterations=args.sinkhorn_iterations,
        transport_gradient_mode="raw",
        transport_plan_mode="streaming",
        row_chunk_size=args.row_chunk_size,
        col_chunk_size=args.col_chunk_size,
    )


def _streaming_value(
    fixture: dict[str, tf.Tensor],
    args: argparse.Namespace,
    *,
    return_history: bool,
    pre_flow_step_fn=None,
):
    kwargs = dict(
        observations=fixture["observations"],
        initial_particles=fixture["initial_particles"],
        fixed_resampling_mask=fixture["fixed_resampling_mask"],
        transition_matrix=fixture["transition_matrix"],
        transition_covariance=fixture["transition_covariance"],
        observation_covariance=fixture["observation_covariance"],
        observation_fn=_observation,
        observation_jacobian_fn=_observation_jacobian,
        observation_residual_fn=_observation_residual,
        transition_log_density_fn=_make_transition_log_density(
            fixture["transition_matrix"],
            fixture["transition_covariance"],
        ),
        observation_log_density_fn=_make_observation_log_density(
            fixture["observation_covariance"],
        ),
        sinkhorn_iterations=args.sinkhorn_iterations,
        row_chunk_size=args.row_chunk_size,
        col_chunk_size=args.col_chunk_size,
        particle_chunk_size=args.particle_chunk_size,
        return_history=return_history,
    )
    if pre_flow_step_fn is None:
        kwargs["pre_flow_particles"] = fixture["pre_flow_particles"]
    else:
        kwargs["pre_flow_step_fn"] = pre_flow_step_fn
    return streaming_batched_ledh_pfpf_ot_value_core_tf(**kwargs)


def _check(checks: list[dict[str, Any]], name: str, passed: bool, role: str, details: dict[str, Any]) -> None:
    checks.append({"name": name, "passed": bool(passed), "role": role, "details": details})


def _device_check(outputs: tuple[tf.Tensor, ...], expect: str, physical_gpus: list[str]) -> dict[str, Any]:
    devices = [tensor.device for tensor in outputs]
    if expect == "any":
        return {"passed": True, "devices": devices}
    if expect == "gpu" and not physical_gpus:
        return {"passed": False, "devices": devices, "reason": "no physical GPU visible"}
    passed = all(expect.upper() in device.upper() for device in devices)
    return {"passed": passed, "devices": devices, "expected": expect}


def _source_diagnostics() -> dict[str, Any]:
    source = inspect.getsource(streaming_batched_ledh_pfpf_ot_value_core_tf)
    return {
        "module": inspect.getsourcefile(streaming_batched_ledh_pfpf_ot_value_core_tf),
        "uses_python_time_loop": "for t in range" in source,
        "uses_tf_while_loop": "tf.while_loop" in source,
        "default_transport_plan_mode_streaming": 'transport_plan_mode: str = "streaming"' in source,
        "default_return_history_false": "return_history: bool = False" in source,
        "stores_python_history_lists": "means = []" in source,
        "calls_numpy": ".numpy(" in source,
    }


def _score_fd_check(args: argparse.Namespace) -> dict[str, Any]:
    fd_args = argparse.Namespace(**vars(args))
    fd_args.transport_policy = "no-resampling"
    fixture = _fixture(fd_args, transport_policy="no-resampling")
    theta = tf.stack(
        [
            fixture["transition_matrix"][:, 0, 0],
            tf.math.log(fixture["transition_covariance"][:, 0, 0]),
            tf.math.log(fixture["observation_covariance"][:, 0, 0]),
        ],
        axis=1,
    )

    def value_from_theta(values: tf.Tensor):
        local = dict(fixture)
        local["transition_matrix"] = values[:, 0:1, None]
        local["transition_covariance"] = tf.exp(values[:, 1:2])[:, :, None]
        local["observation_covariance"] = tf.exp(values[:, 2:3])[:, :, None]
        return _streaming_value(local, fd_args, return_history=False)

    result = streaming_batched_ledh_pfpf_ot_value_and_score_tf(theta, value_from_theta)
    theta_np = theta.numpy()
    finite_difference = np.zeros_like(theta_np)
    step = fd_args.score_fd_step
    for row in range(theta_np.shape[0]):
        for param in range(theta_np.shape[1]):
            direction = np.zeros_like(theta_np)
            direction[row, param] = step
            plus = value_from_theta(tf.constant(theta_np + direction, dtype=DTYPE))
            minus = value_from_theta(tf.constant(theta_np - direction, dtype=DTYPE))
            finite_difference[row, param] = (
                plus.log_likelihood.numpy()[row] - minus.log_likelihood.numpy()[row]
            ) / (2.0 * step)
    delta = np.abs(result.score.numpy() - finite_difference)
    rel = delta / np.maximum(1.0, np.abs(finite_difference))
    abs_max = float(np.max(delta))
    rel_max = float(np.max(rel))
    return {
        "passed": bool(abs_max <= fd_args.score_fd_atol or rel_max <= fd_args.score_fd_rtol),
        "abs_max": abs_max,
        "rel_max": rel_max,
        "atol": fd_args.score_fd_atol,
        "rtol": fd_args.score_fd_rtol,
        "policy": "no-resampling",
    }


def _write_markdown(path: Path, result: dict[str, Any], json_path: Path) -> None:
    lines = [
        "# Streaming LEDH-PFPF-OT Correctness Gate",
        "",
        f"- JSON artifact: `{json_path}`",
        f"- Overall passed: `{result['overall_passed']}`",
        f"- Shape: `{result['shape']}`",
        f"- Source diagnostics: `{result['source_diagnostics']}`",
        "",
        "## Checks",
        "",
    ]
    lines.extend(
        f"- {check['name']}: `{check['passed']}` ({check['role']})"
        for check in result["checks"]
    )
    lines.extend(["", "## Nonclaims", ""])
    lines.extend(f"- {claim}" for claim in result["nonclaims"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    raise RuntimeError("ARCHIVAL_WRONG_TRANSPORT_CHUNK_POLICY: this route is preserved only as provenance and cannot emit new evidence")
    args = _parse_args()
    physical_gpus, logical_gpus = _configure_gpus()
    fixture = _fixture(args)
    checks: list[dict[str, Any]] = []

    with tf.device(args.device):
        baseline = _baseline_value(fixture, args)
        streaming = _streaming_value(fixture, args, return_history=True)
        likelihood_only = _streaming_value(fixture, args, return_history=False)

        value_delta = np.max(
            np.abs(streaming.log_likelihood.numpy() - baseline.log_likelihood.numpy())
        )
        history_delta = np.max(
            np.abs(streaming.filtered_means.numpy() - baseline.filtered_means.numpy())
        )
        finite_outputs = bool(
            np.isfinite(streaming.log_likelihood.numpy()).all()
            and np.isfinite(streaming.filtered_means.numpy()).all()
            and np.isfinite(likelihood_only.log_likelihood.numpy()).all()
        )
        _check(
            checks,
            "finite_outputs",
            finite_outputs,
            "promotion_veto",
            {},
        )
        parity_passed = bool(value_delta <= args.value_atol and history_delta <= args.value_atol)
        _check(
            checks,
            "streaming_vs_baseline_parity",
            parity_passed,
            "promotion_veto",
            {
                "value_max_abs_delta": float(value_delta),
                "history_mean_max_abs_delta": float(history_delta),
                "value_atol": args.value_atol,
                "value_rtol": args.value_rtol,
            },
        )
        _check(
            checks,
            "likelihood_only_omits_history",
            bool(
                likelihood_only.filtered_means.shape[0] == 0
                and likelihood_only.filtered_variances.shape[0] == 0
                and likelihood_only.ess_by_time.shape[0] == 0
            ),
            "promotion_veto",
            {
                "filtered_means_shape": list(likelihood_only.filtered_means.shape),
                "ess_shape": list(likelihood_only.ess_by_time.shape),
            },
        )
        device = _device_check((streaming.log_likelihood,), args.expect_device_kind, physical_gpus)
        _check(checks, "device_placement", bool(device["passed"]), "promotion_veto", device)

        if not args.skip_jit_smoke:
            @tf.function(jit_compile=True, reduce_retracing=True)
            def compiled_value() -> tf.Tensor:
                return _streaming_value(fixture, args, return_history=False).log_likelihood

            compiled = compiled_value()
            compiled_delta = np.max(
                np.abs(compiled.numpy() - likelihood_only.log_likelihood.numpy())
            )
            _check(
                checks,
                "jit_compile_smoke",
                bool(compiled_delta <= args.value_atol),
                "promotion_veto",
                {"compiled_vs_eager_max_abs_delta": float(compiled_delta)},
            )

    if not args.skip_score_fd:
        score_fd = _score_fd_check(args)
        _check(
            checks,
            "no_resampling_score_finite_difference",
            bool(score_fd["passed"]),
            "promotion_veto",
            score_fd,
        )

    source = _source_diagnostics()
    _check(
        checks,
        "source_uses_tf_while_loop_not_python_time_loop",
        bool(
            source["uses_tf_while_loop"]
            and not source["uses_python_time_loop"]
            and source["default_transport_plan_mode_streaming"]
            and source["default_return_history_false"]
            and not source["stores_python_history_lists"]
            and not source["calls_numpy"]
        ),
        "promotion_veto",
        source,
    )

    overall = all(check["passed"] for check in checks)
    result: dict[str, Any] = {
        "timestamp_utc": _dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
        "host": platform.node(),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "tensorflow": tf.__version__,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "physical_gpus": physical_gpus,
        "logical_gpus": logical_gpus,
        "device": args.device,
        "device_scope": args.device_scope,
        "expect_device_kind": args.expect_device_kind,
        "shape": {
            "batch_size": args.batch_size,
            "time_steps": args.time_steps,
            "num_particles": args.num_particles,
            "state_dim": 1,
            "obs_dim": 1,
            "parameter_dim": 3,
        },
        "transport": {
            "policy": args.transport_policy,
            "plan_mode": "streaming",
            "gradient_mode": "raw",
            "sinkhorn_iterations": args.sinkhorn_iterations,
            "row_chunk_size": args.row_chunk_size,
            "col_chunk_size": args.col_chunk_size,
            "particle_chunk_size": args.particle_chunk_size,
        },
        "checks": checks,
        "source_diagnostics": source,
        "overall_passed": overall,
        "nonclaims": list(NONCLAIMS),
    }
    if not overall:
        failed = [check["name"] for check in checks if not check["passed"]]
        result["failed_checks"] = failed

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.markdown_output is not None:
        markdown_output = Path(args.markdown_output)
        markdown_output.parent.mkdir(parents=True, exist_ok=True)
        _write_markdown(markdown_output, result, output)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not overall:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
