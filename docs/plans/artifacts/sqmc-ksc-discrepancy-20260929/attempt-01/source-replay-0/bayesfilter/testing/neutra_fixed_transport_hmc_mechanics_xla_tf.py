"""Phase 18 fixed-transport HMC mechanics XLA compile diagnostic.

This helper loads the Phase 17 frozen affine payload, binds it to the current
LGSSM generic target adapter, and compiles a one-point mechanics value/score
function with ``jit_compile=True``.  It records timing and compilation-size
proxies.

It does not train NeuTra, run HMC sampling or tuning, generate external
samples, run a non-JIT fallback, or establish posterior/scientific/product
readiness.
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

from bayesfilter.inference import (
    FixedTransportValueScoreAdapter,
    load_frozen_neutra_artifact,
    stable_frozen_neutra_artifact_signature,
)
from bayesfilter.ssm import (
    stable_ssm_posterior_adapter_signature,
    stable_ssm_target_signature,
)
from bayesfilter.testing.lgssm_generic_target_adapter_tf import (
    make_lgssm_generic_target_fixture,
)
from bayesfilter.testing.neutra_gpu_affine_payload_tf import (
    EXPECTED_PHASE16_ADAPTER_SIGNATURE,
    EXPECTED_PHASE16_TARGET_SIGNATURE,
    PHASE17_PAYLOAD_FILENAME,
)
from bayesfilter.testing.neutra_gpu_bounded_training_tf import (
    DEFAULT_PHASE16_ARTIFACT_DIR,
    DEFAULT_SEED,
)


PHASE18_ROUTE = "phase18_fixed_transport_hmc_mechanics_xla_compile"
DEFAULT_PHASE17_PAYLOAD_PATH = DEFAULT_PHASE16_ARTIFACT_DIR / PHASE17_PAYLOAD_FILENAME
DEFAULT_PHASE18_OUTPUT_PATH = DEFAULT_PHASE16_ARTIFACT_DIR / (
    "lgssm_static_qr_exact_kalman_affine_neutra_"
    "phase18_fixed_transport_hmc_mechanics_xla_compile_seed20260707.json"
)

NEUTRA_FIXED_TRANSPORT_HMC_MECHANICS_XLA_NONCLAIMS = (
    "Phase 18 fixed-transport HMC mechanics XLA compile diagnostic only",
    "no HMC sampling or tuning claim",
    "no full-chain XLA diagnostic claim",
    "no NeuTra training claim",
    "no external sample generation claim",
    "no posterior convergence claim",
    "no production readiness claim",
    "no default execution readiness claim",
    "no scientific validity claim",
)


class NeuTraFixedTransportHMCMechanicsXLAError(RuntimeError):
    """Raised when the Phase 18 mechanics compile contract is violated."""


@dataclass(frozen=True)
class NeuTraFixedTransportHMCMechanicsXLAConfig:
    """Configuration for Phase 18 fixed-transport mechanics compile diagnostic."""

    payload_path: Path = DEFAULT_PHASE17_PAYLOAD_PATH
    output_path: Path = DEFAULT_PHASE18_OUTPUT_PATH
    seed: int = DEFAULT_SEED
    device: str = "/GPU:0"
    require_gpu: bool = True
    disallow_soft_placement: bool = True
    initial_position: tuple[float, ...] = (0.0, 0.0)

    def normalized(self) -> Mapping[str, Any]:
        return {
            "schema": "bayesfilter.neutra.fixed_transport_hmc_mechanics_xla_config.v1",
            "phase": PHASE18_ROUTE,
            "payload_path": str(self.payload_path),
            "output_path": str(self.output_path),
            "seed": int(self.seed),
            "device": str(self.device),
            "require_gpu": bool(self.require_gpu),
            "disallow_soft_placement": bool(self.disallow_soft_placement),
            "initial_position": tuple(float(item) for item in self.initial_position),
            "use_xla": True,
            "jit_compile": True,
            "jit_compile_false_runtime_allowed": False,
            "training_execution_target": "not_run_phase18_mechanics_compile_only",
            "hmc_policy": "mechanics_compile_only_no_sampling_no_tuning",
            "external_sample_generation_policy": "not_run",
            "nonclaims": NEUTRA_FIXED_TRANSPORT_HMC_MECHANICS_XLA_NONCLAIMS,
        }


def run_fixed_transport_hmc_mechanics_xla_diagnostic(
    config: NeuTraFixedTransportHMCMechanicsXLAConfig | None = None,
) -> Mapping[str, Any]:
    """Run the trusted GPU/XLA fixed-transport mechanics compile diagnostic."""

    cfg = NeuTraFixedTransportHMCMechanicsXLAConfig() if config is None else config
    start = time.monotonic()
    _validate_config(cfg)
    previous_soft_placement = tf.config.get_soft_device_placement()
    if cfg.disallow_soft_placement:
        tf.config.set_soft_device_placement(False)
    try:
        gpu_manifest = _gpu_manifest(require_gpu=bool(cfg.require_gpu))
        route_payload = _run_route(cfg)
    finally:
        tf.config.set_soft_device_placement(previous_soft_placement)

    passed = bool(route_payload["passed"])
    payload = {
        "schema": "bayesfilter.neutra.fixed_transport_hmc_mechanics_xla_result.v1",
        "phase": PHASE18_ROUTE,
        "passed": passed,
        "decision": (
            "PASS_PHASE18_FIXED_TRANSPORT_HMC_MECHANICS_XLA_COMPILE"
            if passed
            else "BLOCK_PHASE18_FIXED_TRANSPORT_HMC_MECHANICS_XLA_COMPILE"
        ),
        "config": cfg.normalized(),
        "gpu_manifest": gpu_manifest,
        **route_payload,
        "elapsed_seconds": time.monotonic() - start,
        "training_executed": False,
        "hmc_sampling_or_tuning_executed": False,
        "external_sample_generation_executed": False,
        "use_xla": True,
        "jit_compile": True,
        "jit_compile_false_runtime_executed": False,
        "nonclaims": NEUTRA_FIXED_TRANSPORT_HMC_MECHANICS_XLA_NONCLAIMS,
    }
    payload = {
        **payload,
        "artifact_hash": f"sha256:{_stable_payload_sha256(payload)}",
        "artifact_hash_semantics": "stable_json_sha256_excluding_artifact_hash_fields",
    }
    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(cfg.output_path, payload)
    return payload


def phase18_error_payload(
    error: BaseException,
    *,
    config: NeuTraFixedTransportHMCMechanicsXLAConfig,
) -> Mapping[str, Any]:
    """Build a blocker payload without converting failure into readiness."""

    return {
        "schema": "bayesfilter.neutra.fixed_transport_hmc_mechanics_xla_result.v1",
        "phase": PHASE18_ROUTE,
        "passed": False,
        "decision": "BLOCK_PHASE18_FIXED_TRANSPORT_HMC_MECHANICS_XLA_COMPILE",
        "config": config.normalized(),
        "error_type": type(error).__name__,
        "error_message": str(error),
        "training_executed": False,
        "hmc_sampling_or_tuning_executed": False,
        "external_sample_generation_executed": False,
        "use_xla": True,
        "jit_compile": True,
        "jit_compile_false_runtime_executed": False,
        "nonclaims": NEUTRA_FIXED_TRANSPORT_HMC_MECHANICS_XLA_NONCLAIMS,
    }


def _run_route(config: NeuTraFixedTransportHMCMechanicsXLAConfig) -> Mapping[str, Any]:
    fixture = make_lgssm_generic_target_fixture()
    target_signature = stable_ssm_target_signature(fixture.contract)
    adapter_signature = stable_ssm_posterior_adapter_signature(fixture.adapter)
    if target_signature != EXPECTED_PHASE16_TARGET_SIGNATURE:
        raise NeuTraFixedTransportHMCMechanicsXLAError("target signature mismatch")
    if adapter_signature != EXPECTED_PHASE16_ADAPTER_SIGNATURE:
        raise NeuTraFixedTransportHMCMechanicsXLAError("adapter signature mismatch")
    base_capability = fixture.adapter.value_score_capability()
    if not base_capability.is_accepted_xla_hmc_authority:
        raise NeuTraFixedTransportHMCMechanicsXLAError(
            "base adapter lacks accepted XLA-HMC value/score authority"
        )

    payload = _read_json_mapping(config.payload_path, "Phase 17 payload")
    payload_hash = _file_sha256(config.payload_path)
    loaded = load_frozen_neutra_artifact(
        payload,
        expected_target_signature=EXPECTED_PHASE16_TARGET_SIGNATURE,
    )
    artifact_signature = stable_frozen_neutra_artifact_signature(loaded)
    adapter = FixedTransportValueScoreAdapter(
        base_adapter=fixture.adapter,
        transport=loaded.transport,
        target_scope="lgssm-neutra-phase18-fixed-transport-hmc-mechanics",
        evidence_path="bayesfilter/testing/neutra_fixed_transport_hmc_mechanics_xla_tf.py",
        xla_hmc_ready=True,
        full_chain_xla_diagnostic_ready=False,
    )
    fixed_capability = adapter.value_score_capability()
    if not fixed_capability.is_accepted_xla_hmc_authority:
        raise NeuTraFixedTransportHMCMechanicsXLAError(
            "fixed-transport adapter lacks accepted XLA-HMC value/score authority"
        )
    z0 = tf.constant([tuple(float(item) for item in config.initial_position)], dtype=tf.float64)
    if z0.shape != (1, int(loaded.manifest.dimension)):
        raise NeuTraFixedTransportHMCMechanicsXLAError(
            "initial_position dimension must match payload dimension"
        )

    with tf.device(config.device):
        objective = tf.function(
            lambda z_arg: adapter.log_prob_and_grad(z_arg),
            jit_compile=True,
        )
        concrete_size = _concrete_graph_size(objective, z0)
        hlo_size = _compiler_ir_size(objective, z0)
        first_start = time.perf_counter()
        first_value, first_score = objective(z0)
        first_wall_seconds = time.perf_counter() - first_start
        second_start = time.perf_counter()
        second_value, second_score = objective(z0)
        second_wall_seconds = time.perf_counter() - second_start

    compile_time_proxy = max(0.0, float(first_wall_seconds - second_wall_seconds))
    finite_checks = {
        "mechanics_value_finite": _tensor_is_finite(first_value),
        "mechanics_score_finite": _tensor_is_finite(first_score),
        "second_mechanics_value_finite": _tensor_is_finite(second_value),
        "second_mechanics_score_finite": _tensor_is_finite(second_score),
    }
    device_checks = {
        "expected_device": str(config.device),
        "value_device": getattr(first_value, "device", ""),
        "score_device": getattr(first_score, "device", ""),
        "all_outputs_on_gpu": all(
            "GPU" in str(getattr(item, "device", "")).upper()
            for item in (first_value, first_score, second_value, second_score)
        ),
    }
    passed = bool(
        all(finite_checks.values())
        and bool(device_checks["all_outputs_on_gpu"])
        and base_capability.is_accepted_xla_hmc_authority
        and fixed_capability.is_accepted_xla_hmc_authority
    )
    return {
        "route": PHASE18_ROUTE,
        "passed": passed,
        "payload_path": str(config.payload_path),
        "payload_file_sha256": payload_hash,
        "payload_stable_hash": f"sha256:{_stable_payload_sha256(payload)}",
        "payload_phase": payload.get("phase"),
        "artifact_signature": artifact_signature,
        "transport_hash": loaded.manifest.transport_hash,
        "target_signature": target_signature,
        "adapter_signature": adapter_signature,
        "fixed_transport_adapter_signature": adapter.adapter_signature(),
        "base_value_score_capability": _capability_payload(base_capability),
        "fixed_transport_value_score_capability": _capability_payload(fixed_capability),
        "finite_checks": finite_checks,
        "device_checks": device_checks,
        "initial_position": _json_safe(z0),
        "first_mechanics_value": _json_safe(first_value),
        "first_mechanics_score": _json_safe(first_score),
        "second_mechanics_value": _json_safe(second_value),
        "second_mechanics_score": _json_safe(second_score),
        "first_call_wall_seconds": float(first_wall_seconds),
        "second_call_wall_seconds": float(second_wall_seconds),
        "compile_time_proxy_seconds": compile_time_proxy,
        "concrete_graph_serialized_bytes": concrete_size,
        "compiler_ir_hlo_text_bytes": hlo_size,
        "use_xla": True,
        "jit_compile": True,
        "jit_compile_false_runtime_executed": False,
        "training_executed": False,
        "hmc_sampling_or_tuning_executed": False,
        "external_sample_generation_executed": False,
    }


def _capability_payload(capability: Any) -> Mapping[str, Any]:
    return {
        "value_score_authority": capability.value_score_authority,
        "xla_hmc_ready": bool(capability.xla_hmc_ready),
        "accepted_xla_hmc_authority": bool(capability.is_accepted_xla_hmc_authority),
        "full_chain_xla_diagnostic_ready": bool(
            capability.full_chain_xla_diagnostic_ready
        ),
        "accepted_full_chain_xla_diagnostic_authority": bool(
            capability.is_accepted_full_chain_xla_diagnostic_authority
        ),
        "runtime_backend": capability.runtime_backend,
        "evidence_path": capability.evidence_path,
        "target_scope": capability.target_scope,
        "nonclaims": capability.nonclaims,
    }


def _validate_config(config: NeuTraFixedTransportHMCMechanicsXLAConfig) -> None:
    if int(config.seed) < 0:
        raise NeuTraFixedTransportHMCMechanicsXLAError("seed must be nonnegative")
    if str(config.device).upper().find("GPU") < 0:
        raise NeuTraFixedTransportHMCMechanicsXLAError(
            "Phase 18 trusted diagnostic requires GPU device"
        )
    if bool(config.require_gpu) is not True:
        raise NeuTraFixedTransportHMCMechanicsXLAError(
            "Phase 18 forbids CPU fallback for GPU/XLA evidence"
        )
    if os.environ.get("CUDA_VISIBLE_DEVICES") == "-1":
        raise NeuTraFixedTransportHMCMechanicsXLAError(
            "Phase 18 trusted GPU/XLA diagnostic cannot run with CUDA_VISIBLE_DEVICES=-1"
        )
    if len(tuple(config.initial_position)) <= 0:
        raise NeuTraFixedTransportHMCMechanicsXLAError(
            "initial_position must be nonempty"
        )


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
        raise NeuTraFixedTransportHMCMechanicsXLAError(
            "trusted TensorFlow GPU visibility is required"
        )
    return manifest


def _device_details(device: Any) -> Mapping[str, Any]:
    try:
        details = tf.config.experimental.get_device_details(device)
    except Exception:
        details = {}
    return {str(key): _json_safe(value) for key, value in details.items()}


def _concrete_graph_size(objective: Any, z: tf.Tensor) -> int | None:
    try:
        concrete = objective.get_concrete_function(z)
        graph_def = concrete.graph.as_graph_def()
        return int(graph_def.ByteSize())
    except Exception:
        return None


def _compiler_ir_size(objective: Any, z: tf.Tensor) -> int | None:
    try:
        compiler_ir = objective.experimental_get_compiler_ir(z)
        hlo_text = compiler_ir(stage="hlo")
        if isinstance(hlo_text, bytes):
            return len(hlo_text)
        return len(str(hlo_text).encode("utf-8"))
    except Exception:
        return None


def _read_json_mapping(path: Path, label: str) -> Mapping[str, Any]:
    if not path.exists():
        raise NeuTraFixedTransportHMCMechanicsXLAError(f"{label} is missing: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise NeuTraFixedTransportHMCMechanicsXLAError(
            f"{label} must be a JSON object"
        )
    return payload


def _tensor_is_finite(value: Any) -> bool:
    tensor = tf.convert_to_tensor(value, dtype=tf.float64)
    return bool(tf.reduce_all(tf.math.is_finite(tensor)).numpy())


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stable_payload_sha256(payload: Mapping[str, Any]) -> str:
    normalized = dict(payload)
    normalized.pop("artifact_hash", None)
    normalized.pop("artifact_hash_semantics", None)
    blob = json.dumps(_json_safe(normalized), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


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


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for Phase 18 mechanics compile diagnostic."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload-path", type=Path, default=DEFAULT_PHASE17_PAYLOAD_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_PHASE18_OUTPUT_PATH)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", type=str, default="/GPU:0")
    args = parser.parse_args(argv)
    config = NeuTraFixedTransportHMCMechanicsXLAConfig(
        payload_path=args.payload_path,
        output_path=args.output_path,
        seed=args.seed,
        device=args.device,
    )
    try:
        result = run_fixed_transport_hmc_mechanics_xla_diagnostic(config)
    except Exception as exc:
        result = phase18_error_payload(exc, config=config)
        config.output_path.parent.mkdir(parents=True, exist_ok=True)
        _write_json(config.output_path, result)
        print(json.dumps({"passed": False, "error_type": type(exc).__name__, "error": str(exc)}))
        return 1
    print(
        json.dumps(
            {
                "passed": bool(result["passed"]),
                "output_path": str(config.output_path),
                "compile_time_proxy_seconds": result["compile_time_proxy_seconds"],
                "concrete_graph_serialized_bytes": result[
                    "concrete_graph_serialized_bytes"
                ],
                "compiler_ir_hlo_text_bytes": result["compiler_ir_hlo_text_bytes"],
                "nonclaims": NEUTRA_FIXED_TRANSPORT_HMC_MECHANICS_XLA_NONCLAIMS,
            },
            sort_keys=True,
        )
    )
    return 0 if bool(result["passed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
