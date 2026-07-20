"""Actual-SIR Nystrom gradient mechanics smoke.

This is a tiny hard-veto diagnostic for the actual-SIR Nystrom route.  It
checks that a scalar built from the Nystrom route log likelihood has a finite
TensorFlow gradient with respect to the initial particles.  It is not an HMC
readiness, posterior correctness, or default-readiness claim.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any


_PRE = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
_PRE.add_argument("--device-scope", choices=("cpu", "visible"), default="cpu")
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

from docs.benchmarks import benchmark_actual_sir_nystrom_compiled_redo as redo  # noqa: E402


NONCLAIMS = (
    "actual-SIR Nystrom tiny gradient mechanics smoke only",
    "no HMC readiness claim",
    "no posterior convergence claim",
    "no posterior correctness claim",
    "no default readiness claim",
    "no target-shape HMC viability claim",
    "no statistical ranking claim",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--batch-seeds", default="84920")
    parser.add_argument("--time-steps", type=int, default=2)
    parser.add_argument("--num-particles", type=int, default=32)
    parser.add_argument("--transport-policy", choices=("active-all", "active-odd", "no-resampling"), default="active-all")
    parser.add_argument("--sinkhorn-iterations", type=int, default=10)
    parser.add_argument("--sinkhorn-epsilon", type=float, default=1.0)
    parser.add_argument("--annealed-scaling", type=float, default=0.9)
    parser.add_argument("--annealed-convergence-threshold", type=float, default=1.0e-3)
    parser.add_argument("--row-chunk-size", type=int, default=32)
    parser.add_argument("--col-chunk-size", type=int, default=32)
    parser.add_argument("--particle-chunk-size", type=int, default=32)
    parser.add_argument("--nystrom-rank", type=int, default=32)
    parser.add_argument("--nystrom-epsilon", type=float, default=0.5)
    parser.add_argument("--nystrom-max-iterations", type=int, default=80)
    parser.add_argument("--nystrom-convergence-threshold", type=float, default=1.0e-4)
    parser.add_argument("--nystrom-cholesky-jitter", type=float, default=1.0e-8)
    parser.add_argument("--nystrom-denominator-floor", type=float, default=1.0e-30)
    parser.add_argument("--nystrom-diagnostics", action="store_true")
    parser.add_argument("--nystrom-core-solver", choices=("cholesky", "eigh_truncated", "svd_truncated"), default="cholesky")
    parser.add_argument("--nystrom-core-rcond", type=float, default=1.0e-6)
    parser.add_argument("--nystrom-kernel-mode", choices=("raw", "positive_projected"), default="raw")
    parser.add_argument("--nystrom-scaling-normalization", choices=("none", "balanced"), default="none")
    parser.add_argument("--history-mode", choices=("full", "value-only"), default="value-only")
    parser.add_argument("--dtype", choices=("float64", "float32"), default="float64")
    parser.add_argument("--tf32-mode", choices=("default", "enabled", "disabled"), default="disabled")
    parser.add_argument("--jit-compile", dest="jit_compile", action="store_true", default=True)
    parser.add_argument("--no-jit-compile", dest="jit_compile", action="store_false")
    parser.add_argument("--device", default="/CPU:0")
    parser.add_argument("--device-scope", choices=("cpu", "visible"), default=_PRE_ARGS.device_scope)
    parser.add_argument("--cuda-visible-devices", default=_PRE_ARGS.cuda_visible_devices)
    parser.add_argument("--expect-device-kind", choices=("any", "cpu", "gpu"), default="cpu")
    parser.add_argument("--output", required=True)
    parser.add_argument("--markdown-output", default=None)
    args = parser.parse_args()
    args.batch_seeds = redo._parse_int_csv(args.batch_seeds)
    if args.time_steps <= 0:
        raise ValueError("time_steps must be positive")
    if args.num_particles <= 1:
        raise ValueError("num_particles must be greater than one")
    if args.nystrom_rank <= 0 or args.nystrom_rank > args.num_particles:
        raise ValueError("nystrom_rank must be positive and <= num_particles")
    if args.history_mode != "value-only":
        raise ValueError("gradient mechanics smoke only supports value-only history")
    if args.expect_device_kind == "gpu" and args.device_scope == "cpu":
        raise ValueError("cpu device scope cannot expect gpu outputs")
    return args


def _finite(value: tf.Tensor) -> bool:
    return bool(tf.reduce_all(tf.math.is_finite(value)).numpy())


def _device_ok(tensor: tf.Tensor, expect_device_kind: str) -> bool:
    device = tensor.device.upper()
    if expect_device_kind == "cpu":
        return "CPU" in device
    if expect_device_kind == "gpu":
        return "GPU" in device
    return True


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, tf.Tensor):
        return _json_ready(value.numpy().tolist())
    return value


def build_result(args: argparse.Namespace) -> dict[str, Any]:
    started = _dt.datetime.now(tz=_dt.timezone.utc).isoformat()
    start = time.perf_counter()
    precision = redo._configure_precision(args)
    physical_gpus, logical_gpus = redo._configure_gpus()
    callbacks = redo.actual_sir._dpf_sir_callbacks()
    tensors, sir_semantics = redo.actual_sir._build_actual_sir_tensors(args)

    initial_particles = tf.Variable(tensors["initial_particles"], trainable=True)
    tensors = dict(tensors)
    tensors["initial_particles"] = initial_particles
    adapter_callbacks = redo.actual_sir._make_actual_sir_callbacks(callbacks, tensors, args.batch_seeds, args)

    with tf.device(args.device):
        with tf.GradientTape() as tape:
            tape.watch(initial_particles)
            value = redo._nystrom_value_core(tensors, adapter_callbacks, args)
            scalar = tf.reduce_sum(value.log_likelihood)
        gradient = tape.gradient(
            scalar,
            initial_particles,
            unconnected_gradients=tf.UnconnectedGradients.ZERO,
        )

    if gradient is None:
        gradient_finite = False
        gradient_norm = float("nan")
        gradient_shape = None
    else:
        gradient_finite = _finite(gradient)
        gradient_norm = float(tf.linalg.global_norm([gradient]).numpy())
        gradient_shape = gradient.shape.as_list()

    hard_vetoes: list[str] = []
    if not _device_ok(scalar, args.expect_device_kind):
        hard_vetoes.append("device_kind_mismatch")
    if not _finite(scalar):
        hard_vetoes.append("nonfinite_scalar")
    if gradient is None:
        hard_vetoes.append("gradient_none")
    elif not gradient_finite:
        hard_vetoes.append("nonfinite_gradient")
    if gradient_norm != gradient_norm or gradient_norm == float("inf"):
        hard_vetoes.append("nonfinite_gradient_norm")
    if int(value.route_invocations.numpy()) <= 0:
        hard_vetoes.append("nystrom_route_invocation_count_zero")
    if not bool(value.finite_factors.numpy()):
        hard_vetoes.append("nonfinite_nystrom_factors")
    if not bool(value.finite_particles.numpy()):
        hard_vetoes.append("nonfinite_nystrom_particles")

    ended = _dt.datetime.now(tz=_dt.timezone.utc).isoformat()
    wall = time.perf_counter() - start
    return {
        "schema_version": "actual_sir_nystrom_gradient_mechanics_smoke.v1",
        "status": "PASS" if not hard_vetoes else "FAIL",
        "overall_passed": not hard_vetoes,
        "hard_veto_status": "passed" if not hard_vetoes else "failed",
        "hard_vetoes": hard_vetoes,
        "route_under_test": "actual_sir_nystrom",
        "question": "Does the actual-SIR Nystrom route expose a finite scalar and finite initial-particle gradient on a tiny mechanics smoke?",
        "shape": {
            "batch_size": len(args.batch_seeds),
            "time_steps": args.time_steps,
            "num_particles": args.num_particles,
            "state_dim": 18,
            "obs_dim": 9,
        },
        "batch_seeds": [int(seed) for seed in args.batch_seeds],
        "scalar_value": float(scalar.numpy()),
        "scalar_finite": _finite(scalar),
        "gradient_is_none": gradient is None,
        "gradient_finite": gradient_finite,
        "gradient_norm": gradient_norm,
        "gradient_shape": gradient_shape,
        "gradient_target": "initial_particles",
        "nystrom": {
            "route_invocations": int(value.route_invocations.numpy()),
            "finite_factors": bool(value.finite_factors.numpy()),
            "finite_particles": bool(value.finite_particles.numpy()),
            "max_row_residual": float(value.max_row_residual.numpy()),
            "max_column_residual": float(value.max_column_residual.numpy()),
            "rank": args.nystrom_rank,
            "epsilon": args.nystrom_epsilon,
            "kernel_mode": args.nystrom_kernel_mode,
            "scaling_normalization": args.nystrom_scaling_normalization,
            "core_solver": args.nystrom_core_solver,
            "diagnostics_enabled": bool(args.nystrom_diagnostics),
        },
        "precision": precision,
        "device": {
            "device": args.device,
            "device_scope": args.device_scope,
            "expect_device_kind": args.expect_device_kind,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "physical_gpus": physical_gpus,
            "logical_gpus": logical_gpus,
            "scalar_device": scalar.device,
            "gradient_device": None if gradient is None else gradient.device,
        },
        "sir_semantics": sir_semantics,
        "run_manifest": {
            "git_commit": redo._run_text(["git", "rev-parse", "HEAD"]),
            "git_status_short": redo._run_text(["git", "status", "--short"]),
            "command": " ".join(sys.argv),
            "working_directory": str(ROOT),
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "tensorflow_version": tf.__version__,
            "started_at": started,
            "ended_at": ended,
            "wall_time_seconds": wall,
            "output": args.output,
            "markdown_output": args.markdown_output,
        },
        "inference_status": {
            "hard_veto_screen": "PASS" if not hard_vetoes else "FAIL",
            "hmc_readiness": "NO",
            "posterior_correctness": "NO",
            "default_readiness": "NO",
            "statistically_supported_ranking": "NO",
            "next_evidence_needed": "reviewed G4 closeout; this smoke cannot establish HMC readiness",
        },
        "nonclaims": list(NONCLAIMS),
    }


def write_markdown(result: dict[str, Any], path: Path, json_path: Path) -> None:
    lines = [
        "# Actual-SIR Nystrom Gradient Mechanics Smoke",
        "",
        f"- JSON artifact: `{json_path}`",
        f"- Status: `{result['status']}`",
        f"- Hard vetoes: `{result['hard_vetoes']}`",
        f"- Route under test: `{result['route_under_test']}`",
        f"- Shape: `{result['shape']}`",
        f"- Scalar finite: `{result['scalar_finite']}`",
        f"- Gradient is None: `{result['gradient_is_none']}`",
        f"- Gradient finite: `{result['gradient_finite']}`",
        f"- Gradient norm: `{result['gradient_norm']}`",
        f"- Nystrom route invocations: `{result['nystrom']['route_invocations']}`",
        "",
        "## Nonclaims",
        "",
    ]
    lines.extend(f"- {claim}" for claim in result["nonclaims"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    raise RuntimeError("ARCHIVAL_WRONG_TRANSPORT_CHUNK_POLICY: this route is preserved only as provenance and cannot emit new evidence")
    args = _parse_args()
    result = build_result(args)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(_json_ready(result), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.markdown_output is not None:
        write_markdown(result, Path(args.markdown_output), output_path)
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
