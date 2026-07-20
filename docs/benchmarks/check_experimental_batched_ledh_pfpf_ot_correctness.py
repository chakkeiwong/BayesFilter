"""Correctness gate for experimental batched LEDH-PFPF-OT DPF.

The gate is intentionally separate from timing benchmarks.  It exercises a
deterministic fixed-input fixture, compares the batched value path against a
scalar-row stack, and writes a JSON artifact with pass/fail diagnostics.
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

from experiments.dpf_implementation.tf_tfp.filters import experimental_batched_ledh_pfpf_ot_tf
from experiments.dpf_implementation.tf_tfp.resampling import annealed_transport_tf
from experiments.dpf_implementation.tf_tfp.filters.experimental_batched_ledh_pfpf_ot_tf import (
    SCALAR_PARITY_ATOL,
    SCALAR_PARITY_RTOL,
    BatchedLEDHPFPFOTValueTensors,
    batched_ledh_pfpf_ot_value_and_score_tf,
    batched_ledh_pfpf_ot_value_core_tf,
)


DTYPE = tf.float64
experimental_batched_ledh_pfpf_ot_tf.DTYPE = DTYPE
annealed_transport_tf.DTYPE = DTYPE

NONCLAIMS = (
    "experimental opt-in DPF correctness gate only",
    "no production default readiness claim",
    "no GPU performance claim",
    "no posterior validity claim",
    "no active-transport score finite-difference claim",
    "no HMC/NeuTra readiness claim",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--time-steps", type=int, default=3)
    parser.add_argument("--num-particles", type=int, default=4)
    parser.add_argument(
        "--transport-policy",
        choices=("active", "no-resampling"),
        default="active",
    )
    parser.add_argument(
        "--transport-gradient-mode",
        choices=("filterflow_clipped", "filterflow_custom_op", "raw"),
        default="raw",
    )
    parser.add_argument(
        "--transport-plan-mode",
        choices=("dense", "streaming"),
        default="dense",
    )
    parser.add_argument("--row-chunk-size", type=int, default=1024)
    parser.add_argument("--col-chunk-size", type=int, default=1024)
    parser.add_argument("--sinkhorn-iterations", type=int, default=10)
    parser.add_argument("--device", default="/CPU:0")
    parser.add_argument(
        "--device-scope",
        choices=("cpu", "visible"),
        default=_PRE_ARGS.device_scope,
    )
    parser.add_argument("--cuda-visible-devices", default=_PRE_ARGS.cuda_visible_devices)
    parser.add_argument(
        "--expect-device-kind",
        choices=("any", "cpu", "gpu"),
        default="cpu",
    )
    parser.add_argument("--value-atol", type=float, default=SCALAR_PARITY_ATOL)
    parser.add_argument("--value-rtol", type=float, default=SCALAR_PARITY_RTOL)
    parser.add_argument("--score-fd-step", type=float, default=1.0e-5)
    parser.add_argument("--score-fd-atol", type=float, default=5.0e-6)
    parser.add_argument("--score-fd-rtol", type=float, default=5.0e-6)
    parser.add_argument("--skip-jit-smoke", action="store_true")
    parser.add_argument("--skip-score-fd", action="store_true")
    parser.add_argument(
        "--expect-no-python-time-loop",
        action="store_true",
        help="Fail if the value core still contains a Python time loop.",
    )
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
    if args.transport_plan_mode == "streaming" and args.transport_gradient_mode != "raw":
        raise ValueError("streaming transport requires --transport-gradient-mode raw")
    if args.row_chunk_size <= 0 or args.col_chunk_size <= 0:
        raise ValueError("row_chunk_size and col_chunk_size must be positive")
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


def _stable_fixture(args: argparse.Namespace, *, transport_policy: str | None = None) -> dict[str, np.ndarray]:
    policy = transport_policy or args.transport_policy
    batch = np.arange(args.batch_size, dtype=np.float64)
    particles = np.linspace(-0.30, 0.45, args.num_particles, dtype=np.float64)
    initial_particles = particles[None, :, None] + 0.002 * batch[:, None, None]
    theta = np.stack(
        [
            0.82 + 0.0002 * batch,
            np.log(0.16 + 0.00005 * batch),
            np.log(0.25 + 0.00005 * batch),
        ],
        axis=1,
    )
    transition_matrix = theta[:, 0:1, None]
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
        "theta": theta,
        "observations": observations,
        "initial_particles": initial_particles,
        "pre_flow_particles": pre_flow_particles,
        "fixed_resampling_mask": fixed_resampling_mask,
    }


def _to_tensors(fixture: dict[str, np.ndarray]) -> dict[str, tf.Tensor]:
    tensors: dict[str, tf.Tensor] = {}
    for name, value in fixture.items():
        dtype = tf.bool if value.dtype == np.bool_ else DTYPE
        tensors[name] = tf.constant(value, dtype=dtype)
    return tensors


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


def _value_from_theta(
    theta: tf.Tensor,
    tensors: dict[str, tf.Tensor],
    args: argparse.Namespace,
) -> BatchedLEDHPFPFOTValueTensors:
    transition_matrix = theta[:, 0:1, None]
    transition_covariance = tf.exp(theta[:, 1:2])[:, :, None]
    observation_covariance = tf.exp(theta[:, 2:3])[:, :, None]
    return batched_ledh_pfpf_ot_value_core_tf(
        observations=tensors["observations"],
        initial_particles=tensors["initial_particles"],
        pre_flow_particles=tensors["pre_flow_particles"],
        fixed_resampling_mask=tensors["fixed_resampling_mask"],
        transition_matrix=transition_matrix,
        transition_covariance=transition_covariance,
        observation_covariance=observation_covariance,
        observation_fn=_observation,
        observation_jacobian_fn=_observation_jacobian,
        observation_residual_fn=_observation_residual,
        transition_log_density_fn=_make_transition_log_density(
            transition_matrix,
            transition_covariance,
        ),
        observation_log_density_fn=_make_observation_log_density(observation_covariance),
        sinkhorn_iterations=args.sinkhorn_iterations,
        transport_gradient_mode=args.transport_gradient_mode,
        transport_plan_mode=args.transport_plan_mode,
        row_chunk_size=args.row_chunk_size,
        col_chunk_size=args.col_chunk_size,
    )


def _select_row(tensors: dict[str, tf.Tensor], row: int) -> dict[str, tf.Tensor]:
    selected: dict[str, tf.Tensor] = {}
    for key, value in tensors.items():
        if key == "observations":
            selected[key] = value
        else:
            selected[key] = value[row : row + 1]
    return selected


def _scalar_stack_value(
    tensors: dict[str, tf.Tensor],
    args: argparse.Namespace,
) -> tf.Tensor:
    values = []
    batch_size = int(tensors["theta"].shape[0])
    for row in range(batch_size):
        selected = _select_row(tensors, row)
        values.append(_value_from_theta(selected["theta"], selected, args).log_likelihood[0])
    return tf.stack(values, axis=0)


def _device_check(
    *,
    tensors: tuple[tf.Tensor, ...],
    expect_device_kind: str,
    physical_gpus: list[str],
) -> tuple[bool, dict[str, Any]]:
    devices = [tensor.device for tensor in tensors]
    if expect_device_kind == "any":
        return True, {"devices": devices}
    if expect_device_kind == "gpu" and not physical_gpus:
        return False, {"devices": devices, "reason": "TensorFlow sees no physical GPUs"}
    expected = expect_device_kind.upper()
    ok = all(expected in device.upper() for device in devices)
    return ok, {"devices": devices, "expected": expect_device_kind}


def _check(
    checks: list[dict[str, Any]],
    *,
    name: str,
    passed: bool,
    role: str,
    details: dict[str, Any] | None = None,
) -> None:
    checks.append(
        {
            "name": name,
            "passed": bool(passed),
            "role": role,
            "details": details or {},
        }
    )


def _source_diagnostics() -> dict[str, Any]:
    source = inspect.getsource(batched_ledh_pfpf_ot_value_core_tf)
    return {
        "module": experimental_batched_ledh_pfpf_ot_tf.__file__,
        "uses_python_time_loop": "for t in range(time_steps)" in source,
        "uses_tf_while_loop": "tf.while_loop" in source,
        "stores_python_history_lists": "means = []" in source and "tf.stack(means" in source,
        "requires_full_pre_flow_tensor": "pre_flow_particles[:, t" in source,
    }


def _run_score_fd_check(
    args: argparse.Namespace,
) -> dict[str, Any]:
    fd_args = argparse.Namespace(**vars(args))
    fd_args.transport_policy = "no-resampling"
    tensors = _to_tensors(_stable_fixture(fd_args, transport_policy="no-resampling"))
    theta = tensors["theta"]
    value_score = batched_ledh_pfpf_ot_value_and_score_tf(
        theta,
        lambda values: _value_from_theta(values, tensors, fd_args),
    )
    row = 0
    step = tf.constant(fd_args.score_fd_step, dtype=DTYPE)
    fd_values = []
    for param in range(int(theta.shape[1])):
        index = tf.constant([[row, param]], dtype=tf.int32)
        plus = tf.tensor_scatter_nd_add(theta, index, tf.reshape(step, [1]))
        minus = tf.tensor_scatter_nd_sub(theta, index, tf.reshape(step, [1]))
        plus_value = _value_from_theta(plus, tensors, fd_args).log_likelihood[row]
        minus_value = _value_from_theta(minus, tensors, fd_args).log_likelihood[row]
        fd_values.append((plus_value - minus_value) / (2.0 * step))
    finite_difference = tf.stack(fd_values, axis=0)
    autodiff = value_score.score[row]
    delta = tf.abs(autodiff - finite_difference)
    abs_max = float(tf.reduce_max(delta).numpy())
    rel = delta / tf.maximum(tf.constant(1.0, dtype=DTYPE), tf.abs(finite_difference))
    rel_max = float(tf.reduce_max(rel).numpy())
    passed = bool(abs_max <= fd_args.score_fd_atol or rel_max <= fd_args.score_fd_rtol)
    return {
        "passed": passed,
        "abs_max": abs_max,
        "rel_max": rel_max,
        "autodiff": [float(value) for value in autodiff.numpy().reshape(-1)],
        "finite_difference": [float(value) for value in finite_difference.numpy().reshape(-1)],
        "atol": fd_args.score_fd_atol,
        "rtol": fd_args.score_fd_rtol,
        "step": fd_args.score_fd_step,
    }


def _write_markdown(path: Path, result: dict[str, Any], json_path: Path) -> None:
    lines = [
        "# Experimental Batched LEDH-PFPF-OT Correctness Gate",
        "",
        f"- JSON artifact: `{json_path}`",
        f"- Overall passed: `{result['overall_passed']}`",
        f"- Shape: `{result['shape']}`",
        f"- Transport: `{result['transport']}`",
        f"- Source diagnostics: `{result['source_diagnostics']}`",
        "",
        "## Checks",
        "",
        "| Check | Role | Passed | Details |",
        "| --- | --- | --- | --- |",
    ]
    for check in result["checks"]:
        lines.append(
            f"| `{check['name']}` | `{check['role']}` | `{check['passed']}` | "
            f"`{check['details']}` |"
        )
    lines.extend(["", "## Nonclaims", ""])
    lines.extend(f"- {claim}" for claim in result["nonclaims"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    raise RuntimeError("ARCHIVAL_WRONG_TRANSPORT_CHUNK_POLICY: this route is preserved only as provenance and cannot emit new evidence")
    args = _parse_args()
    physical_gpus, logical_gpus = _configure_gpus()
    tensors = _to_tensors(_stable_fixture(args))
    checks: list[dict[str, Any]] = []

    with tf.device(args.device):
        batched = _value_from_theta(tensors["theta"], tensors, args)
        scalar_value = _scalar_stack_value(tensors, args)
        batched_value = batched.log_likelihood

        finite = bool(
            tf.reduce_all(tf.math.is_finite(batched_value)).numpy()
            and tf.reduce_all(tf.math.is_finite(batched.filtered_means)).numpy()
            and tf.reduce_all(tf.math.is_finite(batched.filtered_variances)).numpy()
            and tf.reduce_all(tf.math.is_finite(batched.ess_by_time)).numpy()
        )
        _check(
            checks,
            name="finite_outputs",
            passed=finite,
            role="promotion_veto",
            details={
                "value_shape": list(batched_value.shape),
                "means_shape": list(batched.filtered_means.shape),
                "ess_shape": list(batched.ess_by_time.shape),
            },
        )

        value_delta = tf.abs(batched_value - scalar_value)
        max_abs_delta = float(tf.reduce_max(value_delta).numpy())
        rel_delta = value_delta / tf.maximum(tf.constant(1.0, dtype=DTYPE), tf.abs(scalar_value))
        max_rel_delta = float(tf.reduce_max(rel_delta).numpy())
        parity_passed = bool(max_abs_delta <= args.value_atol or max_rel_delta <= args.value_rtol)
        _check(
            checks,
            name="batched_vs_scalar_value_parity",
            passed=parity_passed,
            role="primary_promotion_criterion",
            details={
                "max_abs_delta": max_abs_delta,
                "max_rel_delta": max_rel_delta,
                "atol": args.value_atol,
                "rtol": args.value_rtol,
            },
        )

        permutation = tf.reverse(tf.range(args.batch_size, dtype=tf.int32), axis=[0])
        permuted_tensors = {
            key: value if key == "observations" else tf.gather(value, permutation, axis=0)
            for key, value in tensors.items()
        }
        permuted = _value_from_theta(permuted_tensors["theta"], permuted_tensors, args)
        permutation_delta = tf.abs(permuted.log_likelihood - tf.gather(batched_value, permutation))
        permutation_max = float(tf.reduce_max(permutation_delta).numpy())
        _check(
            checks,
            name="row_permutation_equivariance",
            passed=bool(permutation_max <= args.value_atol),
            role="promotion_veto",
            details={"max_abs_delta": permutation_max},
        )

        first = _select_row(tensors, 0)
        repeated_tensors = {
            key: value if key == "observations" else tf.repeat(value, repeats=args.batch_size, axis=0)
            for key, value in first.items()
        }
        repeated = _value_from_theta(repeated_tensors["theta"], repeated_tensors, args)
        identical_delta = tf.abs(repeated.log_likelihood - repeated.log_likelihood[:1])
        identical_max = float(tf.reduce_max(identical_delta).numpy())
        _check(
            checks,
            name="identical_rows_identical_values",
            passed=bool(identical_max <= args.value_atol),
            role="promotion_veto",
            details={"max_abs_delta": identical_max},
        )

        device_ok, device_details = _device_check(
            tensors=(batched_value,),
            expect_device_kind=args.expect_device_kind,
            physical_gpus=physical_gpus,
        )
        _check(
            checks,
            name="device_placement",
            passed=device_ok,
            role="promotion_veto",
            details=device_details,
        )

        if not args.skip_jit_smoke:
            @tf.function(jit_compile=True, reduce_retracing=True)
            def compiled(theta: tf.Tensor) -> tf.Tensor:
                return _value_from_theta(theta, tensors, args).log_likelihood

            jit_value = compiled(tensors["theta"])
            jit_delta = tf.abs(jit_value - batched_value)
            jit_max = float(tf.reduce_max(jit_delta).numpy())
            jit_finite = bool(tf.reduce_all(tf.math.is_finite(jit_value)).numpy())
            _check(
                checks,
                name="xla_jit_value_smoke",
                passed=bool(jit_finite and jit_max <= args.value_atol),
                role="promotion_veto",
                details={"max_abs_delta_vs_eager": jit_max, "finite": jit_finite},
            )

    if not args.skip_score_fd:
        fd_result = _run_score_fd_check(args)
        _check(
            checks,
            name="no_resampling_score_finite_difference",
            passed=bool(fd_result["passed"]),
            role="promotion_veto",
            details=fd_result,
        )

    source_diagnostics = _source_diagnostics()
    if args.expect_no_python_time_loop:
        _check(
            checks,
            name="source_has_no_python_time_loop",
            passed=not source_diagnostics["uses_python_time_loop"],
            role="promotion_veto",
            details=source_diagnostics,
        )

    overall_passed = all(check["passed"] for check in checks)
    result = {
        "timestamp_utc": _dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
        "host": platform.node(),
        "python": platform.python_version(),
        "tensorflow": tf.__version__,
        "git_commit": _git_commit(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "physical_gpus": physical_gpus,
        "logical_gpus": logical_gpus,
        "device": args.device,
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
            "gradient_mode": args.transport_gradient_mode,
            "plan_mode": args.transport_plan_mode,
            "row_chunk_size": args.row_chunk_size,
            "col_chunk_size": args.col_chunk_size,
            "sinkhorn_iterations": args.sinkhorn_iterations,
        },
        "checks": checks,
        "overall_passed": overall_passed,
        "source_diagnostics": source_diagnostics,
        "evidence_contract": {
            "question": "Does the fixed deterministic batched DPF path preserve scalar-row correctness before timing?",
            "baseline": "scalar-row stack over the same fixed inputs",
            "primary_criterion": "batched/scalar log-likelihood parity",
            "vetoes": [
                "non-finite outputs",
                "failed parity",
                "failed row locality/equivariance",
                "failed JIT smoke",
                "failed no-resampling finite-difference score check",
                "wrong requested device placement",
            ],
            "explanatory_only": [
                "source loop diagnostics",
                "active transport score behavior",
                "timing or memory from separate benchmark scripts",
            ],
            "must_not_conclude": list(NONCLAIMS),
        },
        "nonclaims": list(NONCLAIMS),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.markdown_output is not None:
        markdown_output = Path(args.markdown_output)
        markdown_output.parent.mkdir(parents=True, exist_ok=True)
        _write_markdown(markdown_output, result, output)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not overall_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
