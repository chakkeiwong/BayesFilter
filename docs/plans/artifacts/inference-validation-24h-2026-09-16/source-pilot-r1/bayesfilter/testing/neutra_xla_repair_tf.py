"""Manual-score XLA compile diagnostic for the LGSSM NeuTra route.

This helper is a compile-only gate. It uses the current no-GradientTape LGSSM
target adapter and an affine NeuTra objective with explicit score pullbacks.
It records compile timing and compilation-size proxies for `jit_compile=True`.

It does not train NeuTra, run optimizer updates, run HMC sampling or tuning,
generate external samples, use DSGE/c603, or establish HMC/posterior/product/
scientific readiness.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tensorflow as tf

from bayesfilter.ssm import (
    stable_ssm_posterior_adapter_signature,
    stable_ssm_target_signature,
)
from bayesfilter.testing.lgssm_generic_target_adapter_tf import (
    make_lgssm_generic_target_fixture,
)
from bayesfilter.testing.neutra_gpu_bounded_training_tf import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_INITIAL_RAW_SCALE,
    DEFAULT_SEED,
    LGSSM_QR_ROUTE_ID,
)


EXPECTED_TARGET_SIGNATURE = (
    "275bdd37a82d8c09d2c1b1935b6481f18224644ac659830a921c7958b6ed9038"
)
EXPECTED_ADAPTER_SIGNATURE = (
    "d89bdedcf759566f490ce5222ef753cc8c0c8ea39805d68c064c12d2bec07900"
)
OLD_TAPED_TARGET_SIGNATURE = (
    "290a91d2a8f90d5b29243965b258b1ec6fd965aa46ffca69dcb78f7fa1ecabcb"
)
DEFAULT_PHASE15_OUTPUT = Path(
    "docs/plans/"
    "bayesfilter-lgssm-first-neutra-hmc-phase15-manual-score-xla-compile-"
    "diagnostic-2026-07-08.json"
)
XLA_REPAIR_ROUTE_LGSSM_WHILE_LOOP = "manual_score_lgssm_affine_neutra_xla"
XLA_REPAIR_ROUTE_LEGACY_AUTODIFF = "legacy_autodiff_rejected"

NEUTRA_XLA_REPAIR_NONCLAIMS = (
    "Phase 15 manual-score XLA compile diagnostic only",
    "no jit_compile_false runtime run",
    "no NeuTra training claim",
    "no optimizer update claim",
    "no HMC tuning or sampling claim",
    "no external sample generation claim",
    "no posterior convergence claim",
    "no broad XLA readiness claim",
    "no route ranking claim",
    "no production readiness claim",
    "no default execution readiness claim",
    "no scientific validity claim",
)


class NeuTraXLARepairError(RuntimeError):
    """Raised when the XLA compile diagnostic contract is violated."""


@dataclass(frozen=True)
class NeuTraXLARepairConfig:
    """Configuration for the trusted GPU/XLA compile diagnostic."""

    route: str = XLA_REPAIR_ROUTE_LGSSM_WHILE_LOOP
    seed: int = DEFAULT_SEED
    batch_size: int = DEFAULT_BATCH_SIZE
    initial_raw_scale: float = DEFAULT_INITIAL_RAW_SCALE
    device: str = "/GPU:0"
    require_gpu: bool = True
    disallow_soft_placement: bool = True
    output_path: Path = DEFAULT_PHASE15_OUTPUT

    def normalized(self) -> Mapping[str, Any]:
        return {
            "schema": "bayesfilter.neutra.manual_score_xla_compile_config.v1",
            "phase": "phase15_manual_score_xla_compile_gate",
            "route": str(self.route),
            "route_id": LGSSM_QR_ROUTE_ID,
            "seed": int(self.seed),
            "batch_size": int(self.batch_size),
            "initial_raw_scale": float(self.initial_raw_scale),
            "device": str(self.device),
            "require_gpu": bool(self.require_gpu),
            "disallow_soft_placement": bool(self.disallow_soft_placement),
            "output_path": str(self.output_path),
            "jit_compile": True,
            "jit_compile_false_runtime_allowed": False,
            "training_execution_target": "not_training_xla_compile_diagnostic_only",
            "optimizer_update_executed": False,
            "cpu_training_fallback_policy": "forbidden",
            "external_sample_generation_policy": "not_run_phase12_separate_boundary",
            "hmc_policy": "not_run_not_authorized_for_phase15",
            "nonclaims": NEUTRA_XLA_REPAIR_NONCLAIMS,
        }


def run_neutra_xla_repair_diagnostic(
    config: NeuTraXLARepairConfig | None = None,
) -> Mapping[str, Any]:
    """Run the trusted GPU/XLA compile diagnostic and write JSON."""

    cfg = NeuTraXLARepairConfig() if config is None else config
    start = time.monotonic()
    _validate_config(cfg)
    previous_soft_placement = tf.config.get_soft_device_placement()
    if cfg.disallow_soft_placement:
        tf.config.set_soft_device_placement(False)
    try:
        gpu_manifest = _gpu_manifest(require_gpu=bool(cfg.require_gpu))
        route_payload = _run_xla_route(cfg)
    finally:
        tf.config.set_soft_device_placement(previous_soft_placement)

    passed = bool(route_payload["passed"])
    payload = {
        "schema": "bayesfilter.neutra.manual_score_xla_compile_result.v1",
        "phase": "phase15_manual_score_xla_compile_gate",
        "passed": passed,
        "decision": (
            "PASS_PHASE15_MANUAL_SCORE_XLA_COMPILE_GATE"
            if passed
            else "BLOCK_PHASE15_MANUAL_SCORE_XLA_COMPILE_GATE"
        ),
        "config": cfg.normalized(),
        "gpu_manifest": gpu_manifest,
        "route_result": route_payload,
        "elapsed_seconds": time.monotonic() - start,
        "hmc_executed": False,
        "training_executed": False,
        "optimizer_update_executed": False,
        "external_sample_generation_executed": False,
        "jit_compile": True,
        "jit_compile_false_runtime_executed": False,
        "evidence_contract": {
            "question": (
                "Can the no-tape LGSSM affine NeuTra objective compile and "
                "execute under trusted GPU XLA with jit_compile=True, and what "
                "are the compile-time/size diagnostics?"
            ),
            "primary_criterion": (
                "trusted GPU diagnostic compiles with jit_compile=True, executes "
                "two calls with finite value/gradient diagnostics, and records "
                "compile-time and size proxies; or preserves a parseable blocker"
            ),
            "vetoes": (
                "jit_compile=false runtime run",
                "CPU runtime evidence",
                "hidden training or optimizer update",
                "hidden HMC sampling or tuning",
                "hidden external sample generation",
                "stale target or adapter signature",
                "nonfinite diagnostics",
                "unsupported readiness or scientific claim",
            ),
            "not_concluded": NEUTRA_XLA_REPAIR_NONCLAIMS,
        },
        "nonclaims": NEUTRA_XLA_REPAIR_NONCLAIMS,
    }
    payload = {
        **payload,
        "artifact_hash": f"sha256:{_stable_payload_sha256(payload)}",
        "artifact_hash_semantics": "stable_json_sha256_excluding_artifact_hash_fields",
    }
    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(cfg.output_path, payload)
    return payload


def phase13_error_payload(
    error: BaseException,
    *,
    config: NeuTraXLARepairConfig,
) -> Mapping[str, Any]:
    """Build a blocker payload without converting failure into readiness."""

    return {
        "schema": "bayesfilter.neutra.manual_score_xla_compile_result.v1",
        "phase": "phase15_manual_score_xla_compile_gate",
        "passed": False,
        "decision": "BLOCK_PHASE15_MANUAL_SCORE_XLA_COMPILE_GATE",
        "config": config.normalized(),
        "error_type": type(error).__name__,
        "error_message": str(error),
        "hmc_executed": False,
        "training_executed": False,
        "optimizer_update_executed": False,
        "external_sample_generation_executed": False,
        "jit_compile": True,
        "jit_compile_false_runtime_executed": False,
        "nonclaims": NEUTRA_XLA_REPAIR_NONCLAIMS,
    }


def _run_xla_route(config: NeuTraXLARepairConfig) -> Mapping[str, Any]:
    fixture = make_lgssm_generic_target_fixture()
    target_signature = stable_ssm_target_signature(fixture.contract)
    adapter_signature = stable_ssm_posterior_adapter_signature(fixture.adapter)
    if target_signature != EXPECTED_TARGET_SIGNATURE:
        raise NeuTraXLARepairError("target signature changed unexpectedly")
    if adapter_signature != EXPECTED_ADAPTER_SIGNATURE:
        raise NeuTraXLARepairError("adapter signature changed unexpectedly")
    initial_shift = tf.cast(fixture.source_target.initial_parameters, tf.float64)
    parameter_dim = int(initial_shift.shape[0])
    if parameter_dim != 2:
        raise NeuTraXLARepairError("LGSSM parameter dimension must be 2")

    with tf.device(config.device):
        tf.random.set_seed(int(config.seed))
        shift = tf.Variable(initial_shift, name="phase15_lgssm_shift")
        raw_scale = tf.Variable(
            tf.fill(tf.shape(initial_shift), tf.cast(config.initial_raw_scale, tf.float64)),
            name="phase15_lgssm_raw_scale",
        )
        z = tf.random.stateless_normal(
            shape=(int(config.batch_size), parameter_dim),
            seed=tf.constant([int(config.seed), 15], dtype=tf.int32),
            dtype=tf.float64,
        )
        objective = tf.function(
            lambda shift_arg, raw_scale_arg, z_arg: _manual_affine_neutra_objective(
                fixture.adapter,
                z_arg,
                shift=shift_arg,
                raw_scale=raw_scale_arg,
            ),
            jit_compile=True,
        )
        concrete_size = _concrete_graph_size(objective, shift, raw_scale, z)
        hlo_size = _compiler_ir_size(objective, shift, raw_scale, z)
        first_start = time.perf_counter()
        first = objective(shift, raw_scale, z)
        first_wall_seconds = time.perf_counter() - first_start
        second_start = time.perf_counter()
        second = objective(shift, raw_scale, z)
        second_wall_seconds = time.perf_counter() - second_start

    first_loss, first_shift_grad, first_raw_scale_grad = first
    second_loss, second_shift_grad, second_raw_scale_grad = second
    compile_time_proxy = max(0.0, float(first_wall_seconds - second_wall_seconds))
    finite_checks = {
        "first_loss_finite": _tensor_is_finite(first_loss),
        "first_shift_gradient_finite": _tensor_is_finite(first_shift_grad),
        "first_raw_scale_gradient_finite": _tensor_is_finite(first_raw_scale_grad),
        "second_loss_finite": _tensor_is_finite(second_loss),
        "second_shift_gradient_finite": _tensor_is_finite(second_shift_grad),
        "second_raw_scale_gradient_finite": _tensor_is_finite(second_raw_scale_grad),
    }
    output_devices = tuple(
        str(device)
        for device in (
            first_loss.device,
            first_shift_grad.device,
            first_raw_scale_grad.device,
            second_loss.device,
            second_shift_grad.device,
            second_raw_scale_grad.device,
            z.device,
            shift.device,
            raw_scale.device,
        )
    )
    gpu_outputs = all("GPU" in device.upper() for device in output_devices)
    passed = bool(all(finite_checks.values()) and gpu_outputs)
    return {
        "route": str(config.route),
        "route_id": LGSSM_QR_ROUTE_ID,
        "passed": passed,
        "target_signature": target_signature,
        "adapter_signature": adapter_signature,
        "old_taped_target_signature": OLD_TAPED_TARGET_SIGNATURE,
        "parameter_dim": parameter_dim,
        "batch_size": int(config.batch_size),
        "jit_compile": True,
        "jit_compile_false_runtime_executed": False,
        "first_call_wall_seconds": float(first_wall_seconds),
        "second_call_wall_seconds": float(second_wall_seconds),
        "compile_time_proxy_seconds": compile_time_proxy,
        "concrete_graph_serialized_bytes": concrete_size,
        "compiler_ir_hlo_text_bytes": hlo_size,
        "initial_loss": float(first_loss.numpy()),
        "second_loss": float(second_loss.numpy()),
        "gradient_norms": {
            "first_shift": float(tf.linalg.global_norm([first_shift_grad]).numpy()),
            "first_raw_scale": float(
                tf.linalg.global_norm([first_raw_scale_grad]).numpy()
            ),
            "second_shift": float(tf.linalg.global_norm([second_shift_grad]).numpy()),
            "second_raw_scale": float(
                tf.linalg.global_norm([second_raw_scale_grad]).numpy()
            ),
        },
        "finite_checks": finite_checks,
        "device_checks": {
            "expected_device": str(config.device),
            "output_devices": list(output_devices),
            "all_outputs_on_gpu": bool(gpu_outputs),
            "soft_device_placement": bool(tf.config.get_soft_device_placement()),
        },
        "xla_boundary": "manual_score_affine_neutra_objective",
        "hmc_executed": False,
        "training_executed": False,
        "optimizer_update_executed": False,
        "external_sample_generation_executed": False,
        "nonclaims": NEUTRA_XLA_REPAIR_NONCLAIMS,
    }


def _manual_affine_neutra_objective(
    adapter: Any,
    z: tf.Tensor,
    *,
    shift: tf.Tensor,
    raw_scale: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    scale = tf.exp(raw_scale)
    theta = shift[tf.newaxis, :] + scale[tf.newaxis, :] * z
    value, theta_score = adapter.log_prob_and_grad(theta)
    logdet = tf.reduce_sum(raw_scale)
    loss = -tf.reduce_mean(value + logdet)
    batch_size = tf.cast(tf.shape(z)[0], theta_score.dtype)
    shift_gradient = -tf.reduce_mean(theta_score, axis=0)
    raw_scale_gradient = -tf.reduce_sum(theta_score * scale[tf.newaxis, :] * z, axis=0)
    raw_scale_gradient = raw_scale_gradient / batch_size
    raw_scale_gradient = raw_scale_gradient - tf.ones_like(raw_scale)
    tf.debugging.assert_all_finite(loss, "Phase 15 XLA diagnostic loss must be finite")
    tf.debugging.assert_all_finite(
        shift_gradient,
        "Phase 15 XLA diagnostic shift gradient must be finite",
    )
    tf.debugging.assert_all_finite(
        raw_scale_gradient,
        "Phase 15 XLA diagnostic raw-scale gradient must be finite",
    )
    return loss, shift_gradient, raw_scale_gradient


def _concrete_graph_size(
    objective: Any,
    shift: tf.Tensor,
    raw_scale: tf.Tensor,
    z: tf.Tensor,
) -> int | None:
    try:
        concrete = objective.get_concrete_function(shift, raw_scale, z)
        graph_def = concrete.graph.as_graph_def()
        return int(graph_def.ByteSize())
    except Exception:
        return None


def _compiler_ir_size(
    objective: Any,
    shift: tf.Tensor,
    raw_scale: tf.Tensor,
    z: tf.Tensor,
) -> int | None:
    try:
        compiler_ir = objective.experimental_get_compiler_ir(shift, raw_scale, z)
        hlo_text = compiler_ir(stage="hlo")
        if isinstance(hlo_text, bytes):
            return len(hlo_text)
        return len(str(hlo_text).encode("utf-8"))
    except Exception:
        return None


def _gpu_manifest(*, require_gpu: bool) -> Mapping[str, Any]:
    physical_gpus = tf.config.list_physical_devices("GPU")
    for gpu in physical_gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
    logical_gpus = tf.config.list_logical_devices("GPU")
    manifest = {
        "physical_gpus": [str(device) for device in physical_gpus],
        "physical_gpu_details": [_device_details(device) for device in physical_gpus],
        "logical_gpus": [str(device) for device in logical_gpus],
        "tensorflow_version": tf.__version__,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
        "tf_force_gpu_allow_growth": os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH", "unset"),
        "trusted_gpu_context_required": True,
        "xla_jit_diagnostic": True,
    }
    if require_gpu and not physical_gpus:
        raise NeuTraXLARepairError("trusted TensorFlow GPU visibility is required")
    return manifest


def _validate_config(config: NeuTraXLARepairConfig) -> None:
    if str(config.route) != XLA_REPAIR_ROUTE_LGSSM_WHILE_LOOP:
        raise NeuTraXLARepairError(f"unknown Phase 15 route: {config.route}")
    if int(config.seed) < 0:
        raise NeuTraXLARepairError("seed must be nonnegative")
    if int(config.batch_size) <= 0 or int(config.batch_size) > 512:
        raise NeuTraXLARepairError("batch_size must be in 1..512")
    if str(config.device).upper().find("GPU") < 0:
        raise NeuTraXLARepairError("Phase 15 trusted diagnostic requires GPU device")
    if bool(config.require_gpu) is not True:
        raise NeuTraXLARepairError("Phase 15 forbids CPU fallback for GPU/XLA evidence")
    if os.environ.get("CUDA_VISIBLE_DEVICES") == "-1":
        raise NeuTraXLARepairError(
            "CUDA_VISIBLE_DEVICES=-1 is forbidden for Phase 15 GPU/XLA diagnostic"
        )


def _tensor_is_finite(tensor: tf.Tensor) -> bool:
    return bool(tf.reduce_all(tf.math.is_finite(tensor)).numpy())


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _stable_payload_sha256(payload: Mapping[str, Any]) -> str:
    normalized = dict(_json_safe(payload))
    normalized.pop("artifact_hash", None)
    normalized.pop("artifact_hash_semantics", None)
    blob = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _device_details(device: Any) -> Mapping[str, Any]:
    try:
        details = tf.config.experimental.get_device_details(device)
    except (RuntimeError, ValueError):
        details = {}
    return {
        "device": str(device),
        "details": _json_safe(details),
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if hasattr(value, "numpy"):
        return _json_safe(value.numpy())
    if hasattr(value, "tolist") and hasattr(value, "shape"):
        return _json_safe(value.tolist())
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for the trusted GPU/XLA compile diagnostic."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_PHASE15_OUTPUT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--initial-raw-scale", type=float, default=DEFAULT_INITIAL_RAW_SCALE)
    parser.add_argument("--device", default="/GPU:0")
    args = parser.parse_args(argv)
    config = NeuTraXLARepairConfig(
        seed=int(args.seed),
        batch_size=int(args.batch_size),
        initial_raw_scale=float(args.initial_raw_scale),
        device=str(args.device),
        require_gpu=True,
        output_path=args.output,
    )
    try:
        payload = run_neutra_xla_repair_diagnostic(config)
    except BaseException as exc:
        payload = phase13_error_payload(exc, config=config)
        config.output_path.parent.mkdir(parents=True, exist_ok=True)
        _write_json(config.output_path, payload)
        print(json.dumps(_json_safe(payload), sort_keys=True))
        return 1
    print(
        json.dumps(
            {
                "passed": bool(payload["passed"]),
                "decision": payload["decision"],
                "output_path": str(config.output_path),
                "route": str(config.route),
                "jit_compile": True,
                "nonclaims": NEUTRA_XLA_REPAIR_NONCLAIMS,
            },
            sort_keys=True,
        )
    )
    return 0 if bool(payload["passed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
