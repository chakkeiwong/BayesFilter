"""Trusted-GPU production-shape preflight for Phase 4 Contract E streaming.

This harness measures one deterministic float32/XLA forward chart. It does not
set or infer a scientific acceptance threshold and cannot admit a route.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import sys
import time
import traceback
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tensorflow as tf
import tensorflow_probability as tfp

from bayesfilter.highdim import ledh_contract_e_streaming_tf as streaming
from experiments.dpf_implementation.tf_tfp.resampling import annealed_transport_tf


PROGRAM_ID = "contract-e-canonical-gradient-migration-20260713"
BATCH_SIZE = 1
NUM_PARTICLES = 10_000
STATE_DIMENSION = 3
DTYPE = tf.float32
STEPS = 2
ROW_CHUNK_SIZE = 1024
COL_CHUNK_SIZE = 1024
EPSILON = 0.5
EPSILON0 = 1.0
SCALING = 0.75
RIDGE = 2.0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("forward", "vjp"), default="forward")
    return parser.parse_args()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _memory_info() -> dict[str, Any]:
    try:
        return {
            key: int(value)
            for key, value in tf.config.experimental.get_memory_info("GPU:0").items()
        }
    except (ValueError, RuntimeError) as exc:
        return {"status": "unavailable", "reason": f"{type(exc).__name__}: {exc}"}


def _deterministic_fixture() -> tuple[tf.Tensor, ...]:
    grid = tf.linspace(tf.constant(-1.0, DTYPE), tf.constant(1.0, DTYPE), NUM_PARTICLES)
    pi = tf.constant(3.141592653589793, DTYPE)
    scaled_geometry = tf.stack(
        [grid, tf.sin(pi * grid), tf.cos(0.5 * pi * grid)], axis=1
    )[None, :, :]
    source_particles = tf.stack(
        [
            0.80 * grid + 0.05 * tf.sin(2.0 * pi * grid),
            0.65 * tf.sin(pi * grid) - 0.04 * grid,
            0.55 * tf.cos(0.5 * pi * grid) + 0.03 * tf.sin(3.0 * pi * grid),
        ],
        axis=1,
    )[None, :, :]
    logits = 0.20 * tf.sin(1.5 * pi * grid) + 0.07 * tf.cos(2.5 * pi * grid)
    normalized_log_weights = tf.nn.log_softmax(logits[None, :], axis=1)
    normalized_weights = tf.exp(normalized_log_weights)
    residual_design = tf.stack(
        [
            tf.sin(2.0 * pi * grid),
            tf.cos(3.0 * pi * grid),
            tf.sin(5.0 * pi * grid) + 0.25 * tf.cos(pi * grid),
        ],
        axis=1,
    )
    residual_design -= tf.reduce_mean(residual_design, axis=0, keepdims=True)
    residual_design = residual_design[None, :, :]
    return (
        scaled_geometry,
        source_particles,
        normalized_log_weights,
        normalized_weights,
        residual_design,
        tf.constant([RIDGE], DTYPE),
        tf.constant(EPSILON, DTYPE),
        tf.constant([EPSILON0], DTYPE),
        tf.constant(SCALING, DTYPE),
    )


def _deterministic_upstream() -> tf.Tensor:
    grid = tf.linspace(tf.constant(-1.0, DTYPE), tf.constant(1.0, DTYPE), NUM_PARTICLES)
    pi = tf.constant(3.141592653589793, DTYPE)
    upstream = tf.stack(
        [
            0.5 * tf.cos(2.0 * pi * grid),
            -0.4 * tf.sin(3.0 * pi * grid),
            0.3 * grid + 0.1 * tf.cos(pi * grid),
        ],
        axis=1,
    )
    return upstream[None, :, :]


@tf.function(jit_compile=True, reduce_retracing=True)
def _compiled_forward_diagnostics(*inputs: tf.Tensor) -> dict[str, tf.Tensor]:
    result = streaming._contract_e_streaming_forward_core(  # noqa: SLF001
        *inputs,
        steps=STEPS,
        row_chunk_size=ROW_CHUNK_SIZE,
        col_chunk_size=COL_CHUNK_SIZE,
    )
    quotient = result["quotient"]
    reset = result["reset"]
    particles = result["particles"]
    chart_valid = (
        quotient["valid_chart"]
        & reset["finite"]
        & reset["factor_diagonal_positive"]
        & tf.reduce_all(tf.math.is_finite(particles), axis=[1, 2])
    )
    return {
        "chart_valid": chart_valid,
        "quotient_valid_chart": quotient["valid_chart"],
        "reset_finite": reset["finite"],
        "reset_factor_diagonal_positive": reset["factor_diagonal_positive"],
        "mass_minimum": tf.reduce_min(quotient["mass"], axis=1),
        "mass_maximum": tf.reduce_max(quotient["mass"], axis=1),
        "row_residual_maximum": quotient["row_residual_by_batch"],
        "output_checksum": tf.reduce_sum(particles, axis=[1, 2]),
        "output_maximum_absolute": tf.reduce_max(tf.abs(particles), axis=[1, 2]),
        "mean_residual_maximum_absolute": tf.reduce_max(
            tf.abs(reset["mean_residual"]), axis=1
        ),
        "raw_covariance_residual_frobenius": tf.linalg.norm(
            reset["raw_covariance_residual"], ord="fro", axis=[1, 2]
        ),
        "gap_cholesky_diagonal_minimum": tf.reduce_min(
            reset["gap_chol_diagonal"], axis=1
        ),
        "target_cholesky_diagonal_minimum": tf.reduce_min(
            reset["target_chol_diagonal"], axis=1
        ),
        "injected_cholesky_diagonal_minimum": tf.reduce_min(
            reset["injected_chol_diagonal"], axis=1
        ),
    }


@tf.function(jit_compile=True, reduce_retracing=True)
def _compiled_vjp_diagnostics(
    inputs: tuple[tf.Tensor, ...], upstream: tf.Tensor
) -> dict[str, tf.Tensor]:
    result = streaming._contract_e_streaming_vjp_core(  # noqa: SLF001
        *inputs,
        upstream,
        steps=STEPS,
        row_chunk_size=ROW_CHUNK_SIZE,
        col_chunk_size=COL_CHUNK_SIZE,
    )
    quotient = result["quotient"]
    cotangents = {
        name: result[name]
        for name in (
            "source_particles",
            "source_particles_direct",
            "source_particles_transport",
            "normalized_weights_probability",
            "normalized_log_weights",
            "normalized_log_weights_moment",
            "normalized_log_weights_transport",
            "scaled_geometry",
            "residual_design",
            "ridge",
            "epsilon0",
            "constant_payload",
        )
    }
    all_cotangents_finite = tf.reduce_all(
        tf.stack(
            [tf.reduce_all(tf.math.is_finite(value)) for value in cotangents.values()]
        )
    )
    diagnostics = {
        "chart_valid": tf.reduce_all(quotient["valid_chart"]),
        "all_cotangents_finite": all_cotangents_finite,
        "mass_minimum": tf.reduce_min(quotient["mass"]),
        "mass_maximum": tf.reduce_max(quotient["mass"]),
        "mass_cotangent_maximum_absolute": tf.reduce_max(
            tf.abs(quotient["mass_bar"])
        ),
        "direct_source_path_nonzero": tf.reduce_any(
            tf.not_equal(result["source_particles_direct"], 0.0)
        ),
        "transport_source_path_nonzero": tf.reduce_any(
            tf.not_equal(result["source_particles_transport"], 0.0)
        ),
        "direct_weight_path_nonzero": tf.reduce_any(
            tf.not_equal(result["normalized_log_weights_moment"], 0.0)
        ),
        "transport_weight_path_nonzero": tf.reduce_any(
            tf.not_equal(result["normalized_log_weights_transport"], 0.0)
        ),
    }
    for name, value in cotangents.items():
        diagnostics[f"{name}_maximum_absolute"] = tf.reduce_max(tf.abs(value))
    return diagnostics


def _to_python(result: dict[str, tf.Tensor]) -> dict[str, Any]:
    converted: dict[str, Any] = {}
    for name, tensor in result.items():
        values = tensor.numpy().tolist()
        converted[name] = values
    return converted


def _write_payload(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> int:
    raise RuntimeError("ARCHIVAL_WRONG_TRANSPORT_CHUNK_POLICY: this route is preserved only as provenance and cannot emit new evidence")
    args = _parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started_at = time.time()
    payload: dict[str, Any] = {
        "schema_version": "bayesfilter.contract_e.phase4_gpu_preflight.v1",
        "program_id": PROGRAM_ID,
        "phase": 4,
        "run_kind": (
            "trusted_gpu_xla_tf32_production_shape_forward_preflight"
            if args.mode == "forward"
            else "trusted_gpu_xla_tf32_production_shape_analytic_vjp_preflight"
        ),
        "status": "RUNNING",
        "configuration": {
            "batch_size": BATCH_SIZE,
            "num_particles": NUM_PARTICLES,
            "state_dimension": STATE_DIMENSION,
            "dtype": DTYPE.name,
            "finite_sinkhorn_steps": STEPS,
            "row_chunk_size": ROW_CHUNK_SIZE,
            "col_chunk_size": COL_CHUNK_SIZE,
            "epsilon": EPSILON,
            "epsilon0": EPSILON0,
            "scaling": SCALING,
            "prepared_fixed_ridge": RIDGE,
            "jit_compile": True,
            "tf32_requested": True,
            "fixture": "deterministic_smooth_inline_feasibility_fixture_v1",
            "mode": args.mode,
        },
        "evidence_contract": {
            "question": (
                "Can the repaired quotient-plus-Contract-E forward execute at B=1,N=10000,d=3 on trusted GPU/XLA/TF32 without invalid chart or observed memory failure?"
                if args.mode == "forward"
                else "Can one analytic quotient-plus-Contract-E VJP execute at B=1,N=10000,d=3 on trusted GPU/XLA/TF32 with a valid forward chart and finite separated cotangents?"
            ),
            "hard_vetoes": [
                "missing logical GPU",
                "XLA execution failure",
                "nonfinite or nonpositive row mass",
                "nonfinite Contract E factors or output",
                "nonpositive Cholesky diagonal",
                "observed out-of-memory failure",
                "nonfinite analytic cotangent in VJP mode",
            ],
            "explanatory_only": [
                "row residual",
                "mass range",
                "reset residuals",
                "compile/first-call time",
                "warm execution time",
                "peak memory",
            ],
            "not_concluded": [
                "row-mass or finite-Sinkhorn adequacy",
                "chunk-accumulation adequacy",
                "derivative feasibility or correctness",
                "covariance-restoration or reset adequacy",
                "production admission",
                "Kalman, HMC, nonlinear, leaderboard, or release readiness",
            ],
        },
        "trust_basis": "owner_designated_managed_session_visible_gpu_trusted",
    }
    exit_code = 1
    try:
        annealed_transport_tf.DTYPE = DTYPE
        tf.config.experimental.enable_tensor_float_32_execution(True)
        physical_gpus = tf.config.list_physical_devices("GPU")
        for gpu in physical_gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                pass
        logical_gpus = tf.config.list_logical_devices("GPU")
        payload["environment"] = {
            "python": sys.version.split()[0],
            "tensorflow": tf.__version__,
            "tensorflow_probability": tfp.__version__,
            "platform": platform.platform(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "physical_gpus": [str(device) for device in physical_gpus],
            "logical_gpus": [str(device) for device in logical_gpus],
            "gpu_details": [
                _json_safe(tf.config.experimental.get_device_details(device))
                for device in physical_gpus
            ],
            "tensorflow_build_info": _json_safe(tf.sysconfig.get_build_info()),
            "tf32_execution_enabled": bool(
                tf.config.experimental.tensor_float_32_execution_enabled()
            ),
        }
        if not physical_gpus or not logical_gpus:
            raise RuntimeError("trusted preflight requires a visible logical GPU")
        try:
            tf.config.experimental.reset_memory_stats("GPU:0")
            payload["memory_stats_reset"] = True
        except (AttributeError, ValueError, RuntimeError) as exc:
            payload["memory_stats_reset"] = False
            payload["memory_stats_reset_error"] = f"{type(exc).__name__}: {exc}"

        payload["status"] = "GPU_XLA_EXECUTION_STARTED_CHECKPOINT"
        payload["execution_checkpoint_epoch_seconds"] = time.time()
        _write_payload(args.output, payload)
        with tf.device("/GPU:0"):
            inputs = _deterministic_fixture()
            first_start = time.perf_counter()
            if args.mode == "forward":
                first_result = _to_python(_compiled_forward_diagnostics(*inputs))
            else:
                first_result = _to_python(
                    _compiled_vjp_diagnostics(inputs, _deterministic_upstream())
                )
            first_seconds = time.perf_counter() - first_start

        payload["timing_seconds"] = {"first_compile_and_execute": first_seconds}
        payload["diagnostics"] = first_result
        payload["memory"] = {
            "gpu_allocator": _memory_info(),
            "process_max_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        }
        chart_value = first_result["chart_valid"]
        chart_valid = (
            all(bool(value) for value in chart_value)
            if isinstance(chart_value, list)
            else bool(chart_value)
        )
        if not chart_valid:
            payload["status"] = "HARD_VETO_INVALID_EXECUTED_CHART"
        elif args.mode == "vjp" and not bool(first_result["all_cotangents_finite"]):
            payload["status"] = "HARD_VETO_NONFINITE_ANALYTIC_COTANGENT"
        elif not bool(payload["environment"]["tf32_execution_enabled"]):
            payload["status"] = "HARD_VETO_TF32_NOT_ENABLED"
        else:
            payload["status"] = (
                "GPU_FORWARD_EXECUTED_VALID_CHART_FEASIBILITY_DESCRIPTIVE_ONLY"
                if args.mode == "forward"
                else "GPU_ANALYTIC_VJP_EXECUTED_VALID_CHART_FINITE_COTANGENTS_FEASIBILITY_DESCRIPTIVE_ONLY"
            )
            exit_code = 0
    except Exception as exc:  # Preserve a structured failed-attempt artifact.
        payload["status"] = "GPU_FORWARD_PREFLIGHT_EXECUTION_FAILED"
        payload["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        payload.setdefault("memory", {})["gpu_allocator_after_failure"] = _memory_info()
    finally:
        payload["wall_time_seconds"] = time.time() - started_at
        payload["exit_code"] = exit_code
        _write_payload(args.output, payload)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
