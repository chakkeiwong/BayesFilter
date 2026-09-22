"""GPU-only NeuTra training-objective preflight for generic SSM targets.

This Phase 9 fixture checks that admitted non-DSGE BayesFilter SSM targets can
bind to a tiny affine NeuTra-style training objective on GPU and emit finite
initial loss/gradient diagnostics.  It does not run optimizer steps, train a
transport, generate external samples, run HMC, tune a sampler, or establish
posterior correctness.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import tensorflow as tf

from bayesfilter.ssm import stable_ssm_posterior_adapter_signature
from bayesfilter.testing.lgssm_generic_target_adapter_tf import (
    make_lgssm_generic_target_fixture,
)
from bayesfilter.testing.simple_nonlinear_generic_target_adapter_tf import (
    SIMPLE_NONLINEAR_PRINCIPAL_SQRT_UKF_FILTER_ID,
    SIMPLE_NONLINEAR_SVD_CUBATURE_FILTER_ID,
    SIMPLE_NONLINEAR_SVD_CUT4_FILTER_ID,
    SIMPLE_NONLINEAR_SVD_UKF_FILTER_ID,
    make_simple_nonlinear_generic_target_fixture,
)


NEUTRA_GPU_PREFLIGHT_NONCLAIMS = (
    "Phase 9 GPU NeuTra training-objective preflight only",
    "no optimizer step or learned transport claim",
    "no full NeuTra training claim",
    "no external sample generation claim",
    "no HMC tuning or sampling claim",
    "no posterior convergence claim",
    "no route ranking claim",
    "no production readiness claim",
    "no scientific validity claim",
)

LGSSM_QR_ROUTE_ID = "lgssm-static-qr-exact-kalman"
MODEL_B_SVD_UKF_ROUTE_ID = SIMPLE_NONLINEAR_SVD_UKF_FILTER_ID
MODEL_B_SVD_CUBATURE_ROUTE_ID = SIMPLE_NONLINEAR_SVD_CUBATURE_FILTER_ID
ADMITTED_NEUTRA_GPU_PREFLIGHT_ROUTE_IDS = (
    LGSSM_QR_ROUTE_ID,
    MODEL_B_SVD_UKF_ROUTE_ID,
    MODEL_B_SVD_CUBATURE_ROUTE_ID,
)
DEFERRED_NEUTRA_GPU_PREFLIGHT_ROUTE_IDS = (
    SIMPLE_NONLINEAR_SVD_CUT4_FILTER_ID,
    SIMPLE_NONLINEAR_PRINCIPAL_SQRT_UKF_FILTER_ID,
)

DEFAULT_PHASE9_PREFLIGHT_OUTPUT = Path(
    "docs/plans/"
    "bayesfilter-lgssm-first-neutra-hmc-phase9-gpu-neutra-training-preflight-"
    "2026-07-07.json"
)


class NeuTraGPUPreflightError(RuntimeError):
    """Raised when the Phase 9 GPU preflight contract is violated."""


@dataclass(frozen=True)
class NeuTraGPUPreflightRouteSpec:
    """A fail-closed route admitted to the Phase 9 preflight."""

    route_id: str
    target_family: str
    filter_id: str
    filter_backend: str
    status: str
    reason: str

    def manifest_payload(self) -> Mapping[str, str]:
        return {
            "route_id": self.route_id,
            "target_family": self.target_family,
            "filter_id": self.filter_id,
            "filter_backend": self.filter_backend,
            "status": self.status,
            "reason": self.reason,
        }


_ROUTE_SPECS = {
    LGSSM_QR_ROUTE_ID: NeuTraGPUPreflightRouteSpec(
        route_id=LGSSM_QR_ROUTE_ID,
        target_family="lgssm_static_qr",
        filter_id="tf-qr-exact-lgssm-loglikelihood",
        filter_backend="tf_qr_exact_kalman",
        status="admitted",
        reason="LGSSM exact QR/Kalman target admitted by the LGSSM-first program.",
    ),
    MODEL_B_SVD_UKF_ROUTE_ID: NeuTraGPUPreflightRouteSpec(
        route_id=MODEL_B_SVD_UKF_ROUTE_ID,
        target_family="model_b_simple_nonlinear",
        filter_id=MODEL_B_SVD_UKF_ROUTE_ID,
        filter_backend="tf_svd_ukf",
        status="admitted",
        reason="Phase 7 admitted the deterministic SVD-UKF generic adapter route.",
    ),
    MODEL_B_SVD_CUBATURE_ROUTE_ID: NeuTraGPUPreflightRouteSpec(
        route_id=MODEL_B_SVD_CUBATURE_ROUTE_ID,
        target_family="model_b_simple_nonlinear",
        filter_id=MODEL_B_SVD_CUBATURE_ROUTE_ID,
        filter_backend="tf_svd_cubature",
        status="admitted",
        reason="Phase 8 admitted the deterministic SVD cubature adapter route.",
    ),
    SIMPLE_NONLINEAR_SVD_CUT4_FILTER_ID: NeuTraGPUPreflightRouteSpec(
        route_id=SIMPLE_NONLINEAR_SVD_CUT4_FILTER_ID,
        target_family="model_b_simple_nonlinear",
        filter_id=SIMPLE_NONLINEAR_SVD_CUT4_FILTER_ID,
        filter_backend="tf_svd_cut4",
        status="deferred",
        reason="CUT4 is deferred until a dedicated branch-diagnostic gate.",
    ),
    SIMPLE_NONLINEAR_PRINCIPAL_SQRT_UKF_FILTER_ID: NeuTraGPUPreflightRouteSpec(
        route_id=SIMPLE_NONLINEAR_PRINCIPAL_SQRT_UKF_FILTER_ID,
        target_family="model_b_simple_nonlinear",
        filter_id=SIMPLE_NONLINEAR_PRINCIPAL_SQRT_UKF_FILTER_ID,
        filter_backend="tf_principal_sqrt_ukf",
        status="deferred",
        reason="Principal-square-root UKF is deferred until its own route gate.",
    ),
}


@dataclass(frozen=True)
class NeuTraGPUTrainingPreflightConfig:
    """Configuration for the Phase 9 GPU-only training objective preflight."""

    route_ids: tuple[str, ...] = field(
        default_factory=lambda: ADMITTED_NEUTRA_GPU_PREFLIGHT_ROUTE_IDS
    )
    seed: int = 20260707
    batch_size: int = 16
    initial_raw_scale: float = -1.3862943611198906
    device: str = "/GPU:0"
    jit_compile: bool = False
    require_gpu: bool = True
    disallow_soft_placement: bool = True
    output_path: Path = DEFAULT_PHASE9_PREFLIGHT_OUTPUT

    def normalized(self) -> Mapping[str, Any]:
        return {
            "schema": "bayesfilter.neutra.gpu_training_preflight_config.v1",
            "route_ids": list(self.route_ids),
            "seed": int(self.seed),
            "batch_size": int(self.batch_size),
            "initial_raw_scale": float(self.initial_raw_scale),
            "device": str(self.device),
            "jit_compile": bool(self.jit_compile),
            "xla_readiness_policy": (
                "deferred_to_explicit_gate_not_required_for_phase9_preflight"
            ),
            "require_gpu": bool(self.require_gpu),
            "disallow_soft_placement": bool(self.disallow_soft_placement),
            "output_path": str(self.output_path),
            "training_execution_target": "gpu_required",
            "cpu_training_fallback_policy": "forbidden",
            "external_sample_generation_policy": (
                "multicore_cpu_separate_phase_not_run_here"
            ),
            "full_training_executed": False,
            "optimizer_step_executed": False,
            "hmc_executed": False,
            "nonclaims": NEUTRA_GPU_PREFLIGHT_NONCLAIMS,
        }


def neutra_gpu_preflight_route_inventory() -> tuple[Mapping[str, str], ...]:
    """Return admitted and deferred Phase 9 route metadata."""

    return tuple(spec.manifest_payload() for spec in _ROUTE_SPECS.values())


def admitted_neutra_gpu_preflight_route_ids() -> tuple[str, ...]:
    """Return route ids allowed for the GPU preflight."""

    return ADMITTED_NEUTRA_GPU_PREFLIGHT_ROUTE_IDS


def run_neutra_gpu_training_preflight(
    config: NeuTraGPUTrainingPreflightConfig | None = None,
) -> Mapping[str, Any]:
    """Run the bounded GPU training-objective preflight and write its manifest."""

    cfg = NeuTraGPUTrainingPreflightConfig() if config is None else config
    start = time.monotonic()
    _validate_config(cfg)
    previous_soft_placement = tf.config.get_soft_device_placement()
    if cfg.disallow_soft_placement:
        tf.config.set_soft_device_placement(False)
    try:
        gpu_manifest = _gpu_manifest(require_gpu=bool(cfg.require_gpu))
        route_results = [
            _run_route_preflight(cfg, route_id=route_id, route_index=index)
            for index, route_id in enumerate(cfg.route_ids)
        ]
    finally:
        tf.config.set_soft_device_placement(previous_soft_placement)

    passed = all(bool(result["passed"]) for result in route_results)
    payload = {
        "schema": "bayesfilter.neutra.gpu_training_preflight_result.v1",
        "phase": "phase9_gpu_neutra_training_preflight",
        "passed": bool(passed),
        "decision": (
            "PASS_PHASE9_GPU_NEUTRA_TRAINING_PREFLIGHT"
            if passed
            else "BLOCK_PHASE9_GPU_NEUTRA_TRAINING_PREFLIGHT"
        ),
        "config": cfg.normalized(),
        "gpu_manifest": gpu_manifest,
        "route_inventory": neutra_gpu_preflight_route_inventory(),
        "route_results": route_results,
        "elapsed_seconds": time.monotonic() - start,
        "evidence_contract": {
            "question": (
                "Can admitted non-DSGE SSM targets bind a GPU NeuTra "
                "training objective and emit finite initial loss/gradient "
                "diagnostics without full training or HMC?"
            ),
            "primary_criterion": (
                "all admitted route losses and trainable affine-gradient "
                "diagnostics are finite and placed on GPU output devices"
            ),
            "vetoes": (
                "missing trusted TensorFlow GPU",
                "CPU training fallback",
                "deferred route use",
                "nonfinite loss or gradient",
                "non-GPU output device",
                "hidden HMC",
                "hidden optimizer step",
                "hidden external sample generation",
            ),
            "not_concluded": NEUTRA_GPU_PREFLIGHT_NONCLAIMS,
        },
        "nonclaims": NEUTRA_GPU_PREFLIGHT_NONCLAIMS,
    }
    payload = {
        **payload,
        "artifact_hash": f"sha256:{_stable_payload_sha256(payload)}",
        "artifact_hash_semantics": "stable_json_sha256_excluding_artifact_hash_fields",
    }
    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(cfg.output_path, payload)
    return payload


def phase9_error_payload(
    error: BaseException,
    *,
    config: NeuTraGPUTrainingPreflightConfig,
) -> Mapping[str, Any]:
    """Build a blocker payload without converting failure into evidence."""

    return {
        "schema": "bayesfilter.neutra.gpu_training_preflight_result.v1",
        "phase": "phase9_gpu_neutra_training_preflight",
        "passed": False,
        "decision": "BLOCK_PHASE9_GPU_NEUTRA_TRAINING_PREFLIGHT",
        "config": config.normalized(),
        "error_type": type(error).__name__,
        "error_message": str(error),
        "full_training_executed": False,
        "optimizer_step_executed": False,
        "hmc_executed": False,
        "external_sample_generation_executed": False,
        "nonclaims": NEUTRA_GPU_PREFLIGHT_NONCLAIMS,
    }


def _run_route_preflight(
    config: NeuTraGPUTrainingPreflightConfig,
    *,
    route_id: str,
    route_index: int,
) -> Mapping[str, Any]:
    spec = _require_admitted_route(route_id)
    fixture = _make_fixture(spec)
    adapter = fixture.adapter
    initial_shift = tf.cast(fixture.initial_batch[0], tf.float64)
    parameter_dim = int(initial_shift.shape[0])
    if parameter_dim <= 0:
        raise NeuTraGPUPreflightError("parameter dimension must be positive")

    with tf.device(config.device):
        tf.random.set_seed(int(config.seed))
        shift = tf.Variable(initial_shift, name=f"phase9_shift_{route_index}")
        raw_scale = tf.Variable(
            tf.fill(tf.shape(initial_shift), tf.cast(config.initial_raw_scale, tf.float64)),
            name=f"phase9_raw_scale_{route_index}",
        )
        z = tf.random.stateless_normal(
            shape=(int(config.batch_size), parameter_dim),
            seed=tf.constant([int(config.seed), int(route_index)], dtype=tf.int32),
            dtype=tf.float64,
        )
        objective = tf.function(
            lambda shift_arg, raw_scale_arg, z_arg: _reverse_kl_style_loss(
                adapter,
                z_arg,
                shift=shift_arg,
                raw_scale=raw_scale_arg,
            ),
            jit_compile=bool(config.jit_compile),
        )
        with tf.GradientTape() as tape:
            loss = objective(shift, raw_scale, z)
        gradients = tape.gradient(loss, [shift, raw_scale])

    if any(gradient is None for gradient in gradients):
        raise NeuTraGPUPreflightError(f"missing trainable gradient for route {route_id}")
    loss_value = float(loss.numpy())
    gradient_norms = [float(tf.linalg.global_norm([gradient]).numpy()) for gradient in gradients]
    finite_loss = _is_finite_number(loss_value)
    finite_gradients = all(_tensor_is_finite(gradient) for gradient in gradients)
    output_devices = tuple(
        str(device)
        for device in (loss.device, z.device, shift.device, raw_scale.device)
        + tuple(gradient.device for gradient in gradients)
    )
    gpu_outputs = all("GPU" in device.upper() for device in output_devices)
    passed = bool(finite_loss and finite_gradients and gpu_outputs)
    return {
        "route": spec.manifest_payload(),
        "passed": passed,
        "target_signature": adapter.target_signature,
        "adapter_signature": stable_ssm_posterior_adapter_signature(adapter),
        "parameter_dim": parameter_dim,
        "parameter_names": list(adapter.parameter_names),
        "batch_size": int(config.batch_size),
        "initial_loss": loss_value,
        "gradient_norms": gradient_norms,
        "finite_checks": {
            "loss_finite": bool(finite_loss),
            "shift_gradient_finite": bool(_tensor_is_finite(gradients[0])),
            "raw_scale_gradient_finite": bool(_tensor_is_finite(gradients[1])),
        },
        "device_checks": {
            "expected_device": str(config.device),
            "output_devices": list(output_devices),
            "all_outputs_on_gpu": bool(gpu_outputs),
            "soft_device_placement": bool(tf.config.get_soft_device_placement()),
        },
        "jit_compile": bool(config.jit_compile),
        "full_training_executed": False,
        "optimizer_step_executed": False,
        "hmc_executed": False,
        "external_sample_generation_executed": False,
        "nonclaims": NEUTRA_GPU_PREFLIGHT_NONCLAIMS,
    }


def _reverse_kl_style_loss(
    adapter: Any,
    z: tf.Tensor,
    *,
    shift: tf.Tensor,
    raw_scale: tf.Tensor,
) -> tf.Tensor:
    scale = tf.exp(raw_scale)
    theta = shift[tf.newaxis, :] + scale[tf.newaxis, :] * z
    value, _score = adapter.log_prob_and_grad(theta)
    logdet = tf.reduce_sum(raw_scale)
    loss = -tf.reduce_mean(value + logdet)
    tf.debugging.assert_all_finite(loss, "Phase 9 NeuTra preflight loss must be finite")
    return loss


def _make_fixture(spec: NeuTraGPUPreflightRouteSpec) -> Any:
    if spec.route_id == LGSSM_QR_ROUTE_ID:
        return make_lgssm_generic_target_fixture()
    if spec.route_id in {
        MODEL_B_SVD_UKF_ROUTE_ID,
        MODEL_B_SVD_CUBATURE_ROUTE_ID,
    }:
        return make_simple_nonlinear_generic_target_fixture(filter_id=spec.route_id)
    raise NeuTraGPUPreflightError(f"route is not implemented: {spec.route_id}")


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
        "logical_gpus": [str(device) for device in logical_gpus],
        "tensorflow_version": tf.__version__,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
        "tf_force_gpu_allow_growth": os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH", "unset"),
        "trusted_gpu_context_required": True,
    }
    if require_gpu and not physical_gpus:
        raise NeuTraGPUPreflightError("trusted TensorFlow GPU visibility is required")
    return manifest


def _validate_config(config: NeuTraGPUTrainingPreflightConfig) -> None:
    if int(config.seed) < 0:
        raise NeuTraGPUPreflightError("seed must be nonnegative")
    if int(config.batch_size) <= 0 or int(config.batch_size) > 4096:
        raise NeuTraGPUPreflightError("batch_size must be in 1..4096")
    seen: set[str] = set()
    for route_id in config.route_ids:
        route_key = str(route_id)
        if route_key in seen:
            raise NeuTraGPUPreflightError(f"duplicate route id: {route_key}")
        seen.add(route_key)
        _require_admitted_route(route_key)
    if str(config.device).upper().find("GPU") < 0:
        raise NeuTraGPUPreflightError("Phase 9 NeuTra preflight requires a GPU device")
    if bool(config.require_gpu) is not True:
        raise NeuTraGPUPreflightError("Phase 9 preflight forbids CPU training fallback")
    if os.environ.get("CUDA_VISIBLE_DEVICES") == "-1":
        raise NeuTraGPUPreflightError(
            "CUDA_VISIBLE_DEVICES=-1 is forbidden for GPU NeuTra training preflight"
        )


def _require_admitted_route(route_id: str) -> NeuTraGPUPreflightRouteSpec:
    key = str(route_id)
    try:
        spec = _ROUTE_SPECS[key]
    except KeyError as exc:
        raise NeuTraGPUPreflightError(f"unknown Phase 9 route id: {key}") from exc
    if spec.status != "admitted":
        raise NeuTraGPUPreflightError(
            f"Phase 9 route is not admitted: {key}; {spec.reason}"
        )
    return spec


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


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


def _stable_payload_sha256(payload: Mapping[str, Any]) -> str:
    normalized = dict(_json_safe(payload))
    normalized.pop("artifact_hash", None)
    normalized.pop("artifact_hash_semantics", None)
    blob = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _tensor_is_finite(tensor: tf.Tensor) -> bool:
    return bool(tf.reduce_all(tf.math.is_finite(tensor)).numpy())


def _is_finite_number(value: float) -> bool:
    return value == value and abs(value) < float("inf")


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for the trusted GPU Phase 9 preflight."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_PHASE9_PREFLIGHT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260707)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--initial-raw-scale", type=float, default=-1.3862943611198906)
    parser.add_argument("--device", default="/GPU:0")
    parser.add_argument("--route", action="append", dest="routes")
    parser.add_argument(
        "--jit-compile",
        action="store_true",
        help=(
            "Opt into an XLA/JIT probe. Phase 9 GPU preflight evidence does not "
            "require this; XLA readiness is a separate gate."
        ),
    )
    args = parser.parse_args(argv)
    config = NeuTraGPUTrainingPreflightConfig(
        route_ids=tuple(args.routes or ADMITTED_NEUTRA_GPU_PREFLIGHT_ROUTE_IDS),
        seed=int(args.seed),
        batch_size=int(args.batch_size),
        initial_raw_scale=float(args.initial_raw_scale),
        device=str(args.device),
        jit_compile=bool(args.jit_compile),
        require_gpu=True,
        output_path=args.output,
    )
    try:
        payload = run_neutra_gpu_training_preflight(config)
    except BaseException as exc:
        payload = phase9_error_payload(exc, config=config)
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
                "routes": list(config.route_ids),
                "nonclaims": NEUTRA_GPU_PREFLIGHT_NONCLAIMS,
            },
            sort_keys=True,
        )
    )
    return 0 if bool(payload["passed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
