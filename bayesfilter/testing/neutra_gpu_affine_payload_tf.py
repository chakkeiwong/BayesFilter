"""Package a Phase 16 GPU/XLA-trained affine NeuTra state as a frozen payload.

This Phase 17 fixture consumes the bounded GPU/XLA training-state artifact
written by :mod:`bayesfilter.testing.neutra_gpu_bounded_training_tf`, packages
the learned affine parameters into the reviewed frozen affine-diagonal
transport schema, reloads the payload through the standard loader, and runs
bounded CPU-hidden loader/reference checks.

It does not train NeuTra, run HMC sampling or tuning, generate external
samples, run a new JIT compile gate, repair XLA, or establish posterior
correctness.
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
from bayesfilter.testing.neutra_gpu_bounded_training_tf import (
    DEFAULT_PHASE16_ARTIFACT_DIR,
    DEFAULT_SEED,
    LGSSM_QR_ROUTE_ID,
)


EXPECTED_PHASE16_TARGET_SIGNATURE = (
    "275bdd37a82d8c09d2c1b1935b6481f18224644ac659830a921c7958b6ed9038"
)
EXPECTED_PHASE16_ADAPTER_SIGNATURE = (
    "d89bdedcf759566f490ce5222ef753cc8c0c8ea39805d68c064c12d2bec07900"
)
PHASE16_TRAINING_STATE_PATH = DEFAULT_PHASE16_ARTIFACT_DIR / (
    "lgssm_static_qr_exact_kalman_affine_neutra_gpu_xla_training_state_seed20260707.json"
)
EXPECTED_PHASE16_TRAINING_STATE_FILE_SHA256 = (
    "727fea040502e4fcb1af2203b9a490d03ab00dca63fd756f501a5bc3c936af7b"
)
PHASE17_PAYLOAD_FILENAME = (
    "lgssm_static_qr_exact_kalman_affine_neutra_gpu_xla_frozen_payload_seed20260707.json"
)
PHASE17_VALIDATION_FILENAME = (
    "lgssm_static_qr_exact_kalman_affine_neutra_gpu_xla_payload_validation_seed20260707.json"
)

NEUTRA_GPU_AFFINE_PAYLOAD_NONCLAIMS = (
    "Phase 17 frozen GPU/XLA-trained affine payload packaging only",
    "no new NeuTra training claim",
    "no CPU NeuTra training claim",
    "no dense IAF training claim",
    "no HMC tuning or sampling claim",
    "no external sample generation claim",
    "no posterior convergence claim",
    "no new JIT compile or XLA readiness claim",
    "no route ranking claim",
    "no production readiness claim",
    "no default execution readiness claim",
    "no scientific validity claim",
)


class NeuTraGPUAffinePayloadError(RuntimeError):
    """Raised when the Phase 17 payload packaging contract is violated."""


@dataclass(frozen=True)
class NeuTraGPUAffinePayloadConfig:
    """Configuration for Phase 17 frozen affine payload packaging."""

    phase16_training_state_path: Path = PHASE16_TRAINING_STATE_PATH
    artifact_dir: Path = DEFAULT_PHASE16_ARTIFACT_DIR
    seed: int = DEFAULT_SEED
    require_cpu_hidden: bool = True
    target_signature: str = EXPECTED_PHASE16_TARGET_SIGNATURE
    adapter_signature: str = EXPECTED_PHASE16_ADAPTER_SIGNATURE
    training_state_file_sha256: str = EXPECTED_PHASE16_TRAINING_STATE_FILE_SHA256

    @property
    def payload_path(self) -> Path:
        return self.artifact_dir / PHASE17_PAYLOAD_FILENAME

    @property
    def validation_path(self) -> Path:
        return self.artifact_dir / PHASE17_VALIDATION_FILENAME

    def normalized(self) -> Mapping[str, Any]:
        return {
            "schema": "bayesfilter.neutra.gpu_affine_payload_config.v1",
            "phase": "phase17_frozen_gpu_xla_trained_affine_payload",
            "phase16_training_state_path": str(self.phase16_training_state_path),
            "artifact_dir": str(self.artifact_dir),
            "payload_path": str(self.payload_path),
            "validation_path": str(self.validation_path),
            "seed": int(self.seed),
            "require_cpu_hidden": bool(self.require_cpu_hidden),
            "expected_target_signature": str(self.target_signature),
            "expected_adapter_signature": str(self.adapter_signature),
            "expected_training_state_file_sha256": str(
                self.training_state_file_sha256
            ),
            "packaging_execution_target": "cpu_hidden_artifact_loader_reference",
            "training_execution_target": "not_run_phase16_gpu_xla_artifact_only",
            "hmc_policy": "not_run_mechanics_compile_deferred_to_phase18",
            "external_sample_generation_policy": "not_run_separate_cpu_multicore_phase",
            "xla_policy": (
                "source_artifact_jit_compile_true_no_phase17_compile_run_no_fallback"
            ),
            "nonclaims": NEUTRA_GPU_AFFINE_PAYLOAD_NONCLAIMS,
        }


@dataclass(frozen=True)
class NeuTraGPUAffinePayloadResult:
    """Materialized Phase 17 payload plus validation metadata."""

    config: NeuTraGPUAffinePayloadConfig
    phase16_training_state: Mapping[str, Any]
    frozen_payload: Mapping[str, Any]
    validation: Mapping[str, Any]
    payload_path: Path
    validation_path: Path

    @property
    def passed(self) -> bool:
        return bool(self.validation["passed"])


def package_and_validate_gpu_affine_payload(
    config: NeuTraGPUAffinePayloadConfig | None = None,
) -> NeuTraGPUAffinePayloadResult:
    """Package the Phase 16 GPU/XLA affine state and run loader/reference checks."""

    cfg = NeuTraGPUAffinePayloadConfig() if config is None else config
    start = time.monotonic()
    _validate_phase17_config(cfg)
    state = _read_json_mapping(cfg.phase16_training_state_path, "Phase 16 state")
    training_state_file_hash = _file_sha256(cfg.phase16_training_state_path)
    if training_state_file_hash != str(cfg.training_state_file_sha256):
        raise NeuTraGPUAffinePayloadError("Phase 16 state file sha256 mismatch")
    _validate_phase16_training_state(
        state,
        expected_target_signature=str(cfg.target_signature),
        expected_adapter_signature=str(cfg.adapter_signature),
    )

    fixture = make_lgssm_generic_target_fixture()
    observed_target_signature = stable_ssm_target_signature(fixture.contract)
    observed_adapter_signature = stable_ssm_posterior_adapter_signature(fixture.adapter)
    if observed_target_signature != str(cfg.target_signature):
        raise NeuTraGPUAffinePayloadError("BayesFilter target signature mismatch")
    if observed_adapter_signature != str(cfg.adapter_signature):
        raise NeuTraGPUAffinePayloadError("BayesFilter adapter signature mismatch")

    payload = _frozen_affine_payload(
        config=cfg,
        state=state,
        training_state_file_hash=training_state_file_hash,
    )
    loaded = load_frozen_neutra_artifact(
        payload,
        expected_target_signature=str(cfg.target_signature),
    )
    validation = _validation_payload(
        config=cfg,
        fixture=fixture,
        loaded_artifact=loaded,
        state=state,
        payload=payload,
        training_state_file_hash=training_state_file_hash,
        elapsed_seconds=time.monotonic() - start,
    )
    cfg.artifact_dir.mkdir(parents=True, exist_ok=True)
    _write_json(cfg.payload_path, payload)
    _write_json(cfg.validation_path, validation)
    return NeuTraGPUAffinePayloadResult(
        config=cfg,
        phase16_training_state=state,
        frozen_payload=payload,
        validation=validation,
        payload_path=cfg.payload_path,
        validation_path=cfg.validation_path,
    )


def load_phase17_payload(path: str | Path) -> Mapping[str, Any]:
    """Read a Phase 17 frozen affine payload JSON file."""

    return _read_json_mapping(Path(path), "Phase 17 payload")


def _validate_phase17_config(config: NeuTraGPUAffinePayloadConfig) -> None:
    if bool(config.require_cpu_hidden) and os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise NeuTraGPUAffinePayloadError(
            "Phase 17 packaging/loader checks must run with CUDA_VISIBLE_DEVICES=-1"
        )
    if int(config.seed) < 0:
        raise NeuTraGPUAffinePayloadError("seed must be nonnegative")
    _require_sha256_hex(str(config.target_signature), "target_signature")
    _require_sha256_hex(str(config.adapter_signature), "adapter_signature")
    _require_sha256_hex(
        str(config.training_state_file_sha256),
        "training_state_file_sha256",
    )


def _validate_phase16_training_state(
    state: Mapping[str, Any],
    *,
    expected_target_signature: str,
    expected_adapter_signature: str,
) -> None:
    required_pairs = {
        "schema": "bayesfilter.neutra.gpu_xla_bounded_training_state.v1",
        "phase": "phase16_bounded_gpu_xla_neutra_training",
        "decision": "PASS_PHASE16_BOUNDED_GPU_XLA_NEUTRA_TRAINING",
        "target_signature": expected_target_signature,
        "adapter_signature": expected_adapter_signature,
    }
    for key, expected in required_pairs.items():
        if state.get(key) != expected:
            raise NeuTraGPUAffinePayloadError(f"Phase 16 state {key} mismatch")
    route = _require_mapping(state.get("route"), "route")
    if route.get("route_id") != LGSSM_QR_ROUTE_ID:
        raise NeuTraGPUAffinePayloadError("Phase 16 state route mismatch")
    config = _require_mapping(state.get("config"), "config")
    if config.get("training_execution_target") != "gpu_required":
        raise NeuTraGPUAffinePayloadError("Phase 16 state must be GPU-trained")
    if config.get("cpu_training_fallback_policy") != "forbidden":
        raise NeuTraGPUAffinePayloadError("Phase 16 CPU fallback policy mismatch")
    if bool(config.get("jit_compile")) is not True:
        raise NeuTraGPUAffinePayloadError(
            "Phase 16 source state must preserve jit_compile=true"
        )

    boolean_expectations = {
        "passed": True,
        "bounded_optimizer_training_executed": True,
        "full_neutra_training_executed": False,
        "hmc_executed": False,
        "external_sample_generation_executed": False,
        "frozen_transport_payload_written": False,
    }
    for key, expected in boolean_expectations.items():
        if bool(state.get(key)) is not expected:
            raise NeuTraGPUAffinePayloadError(f"Phase 16 state {key} mismatch")

    dimension = int(state.get("parameter_dim", 0))
    if dimension <= 0:
        raise NeuTraGPUAffinePayloadError("Phase 16 parameter_dim must be positive")
    final_shift = _float_tuple(state.get("final_shift"), "final_shift")
    final_raw_scale = _float_tuple(state.get("final_raw_scale"), "final_raw_scale")
    if len(final_shift) != dimension or len(final_raw_scale) != dimension:
        raise NeuTraGPUAffinePayloadError("Phase 16 learned tensor dimension mismatch")


def _frozen_affine_payload(
    *,
    config: NeuTraGPUAffinePayloadConfig,
    state: Mapping[str, Any],
    training_state_file_hash: str,
) -> Mapping[str, Any]:
    final_shift = _float_tuple(state.get("final_shift"), "final_shift")
    final_raw_scale = _float_tuple(state.get("final_raw_scale"), "final_raw_scale")
    return {
        "schema": "bayesfilter.neutra.frozen_affine_diag.v1",
        "phase": "phase17_frozen_gpu_xla_trained_affine_payload",
        "transport_id": f"lgssm-gpu-trained-affine-diag-neutra-seed{int(config.seed)}",
        "dimension": len(final_shift),
        "target_signature": str(config.target_signature),
        "log_jacobian_available": True,
        "shift": [float(item) for item in final_shift],
        "raw_scale": [float(item) for item in final_raw_scale],
        "training_state_hash": f"sha256:{training_state_file_hash}",
        "expected_training_state_file_sha256": str(config.training_state_file_sha256),
        "source_training_state_path": str(config.phase16_training_state_path),
        "source_training_state_artifact_hash": str(state.get("artifact_hash", "")),
        "source_adapter_signature": str(config.adapter_signature),
        "source_route_id": LGSSM_QR_ROUTE_ID,
        "source_training_execution_target": "gpu_required",
        "packaging_execution_target": "cpu_safe_artifact_loader_reference",
        "bounded_optimizer_training_executed_in_source": True,
        "training_executed": False,
        "hmc_executed": False,
        "external_sample_generation_executed": False,
        "source_jit_compile": True,
        "fixed_transport_hmc_mechanics_executed": False,
        "fixed_transport_hmc_mechanics_deferred_to_phase18": True,
        "jit_compile_runtime_executed": False,
        "jit_compile_false_runtime_executed": False,
        "xla_readiness_claimed": False,
        "nonclaims": NEUTRA_GPU_AFFINE_PAYLOAD_NONCLAIMS,
    }


def _validation_payload(
    *,
    config: NeuTraGPUAffinePayloadConfig,
    fixture: Any,
    loaded_artifact: Any,
    state: Mapping[str, Any],
    payload: Mapping[str, Any],
    training_state_file_hash: str,
    elapsed_seconds: float,
) -> Mapping[str, Any]:
    z = tf.constant([[0.0, 0.0], [0.15, -0.20]], dtype=tf.float64)
    theta = loaded_artifact.transport.forward_batch(z)
    logdet = loaded_artifact.transport.log_abs_det_jacobian_batch(z)
    base_value, base_score = fixture.adapter.log_prob_and_grad(theta)
    source_value, source_score = fixture.source_target.target_log_prob_and_grad(
        fixture.initial_batch[0]
    )
    adapter_value, adapter_score = fixture.adapter.log_prob_and_grad(fixture.initial_batch)

    reference_value_residual = float(
        tf.reduce_max(tf.abs(adapter_value[0] - source_value)).numpy()
    )
    reference_score_residual = float(
        tf.reduce_max(tf.abs(adapter_score[0] - source_score)).numpy()
    )
    finite_checks = {
        "payload_shift_finite": _sequence_is_finite(payload["shift"]),
        "payload_raw_scale_finite": _sequence_is_finite(payload["raw_scale"]),
        "theta_finite": _tensor_is_finite(theta),
        "logdet_finite": _tensor_is_finite(logdet),
        "base_values_finite": _tensor_is_finite(base_value),
        "base_scores_finite": _tensor_is_finite(base_score),
    }
    boundary_checks = {
        "target_signature_match": (
            loaded_artifact.manifest.target_signature == str(config.target_signature)
        ),
        "adapter_signature_match": (
            stable_ssm_posterior_adapter_signature(fixture.adapter)
            == str(config.adapter_signature)
        ),
        "source_gpu_training_artifact": (
            state.get("config", {}).get("training_execution_target") == "gpu_required"
        ),
        "new_training_executed": False,
        "fixed_transport_hmc_mechanics_executed": False,
        "hmc_sampling_or_tuning_executed": False,
        "external_sample_generation_executed": False,
        "source_jit_compile_true": True,
        "fixed_transport_hmc_mechanics_deferred_to_phase18": True,
        "jit_compile_runtime_executed": False,
        "jit_compile_false_runtime_executed": False,
    }
    passed = (
        all(bool(value) for value in finite_checks.values())
        and all(bool(value) is False for key, value in boundary_checks.items() if key in {
            "new_training_executed",
            "fixed_transport_hmc_mechanics_executed",
            "hmc_sampling_or_tuning_executed",
            "external_sample_generation_executed",
            "jit_compile_runtime_executed",
            "jit_compile_false_runtime_executed",
        })
        and bool(boundary_checks["source_jit_compile_true"])
        and bool(boundary_checks["fixed_transport_hmc_mechanics_deferred_to_phase18"])
        and bool(boundary_checks["target_signature_match"])
        and bool(boundary_checks["adapter_signature_match"])
        and bool(boundary_checks["source_gpu_training_artifact"])
        and reference_value_residual <= 1.0e-10
        and reference_score_residual <= 1.0e-8
    )
    validation_without_hash = {
        "schema": "bayesfilter.neutra.gpu_affine_payload_validation.v1",
        "phase": "phase17_frozen_gpu_xla_trained_affine_payload",
        "passed": bool(passed),
        "decision": (
            "PASS_PHASE17_FROZEN_GPU_XLA_AFFINE_PAYLOAD"
            if passed
            else "BLOCK_PHASE17_FROZEN_GPU_XLA_AFFINE_PAYLOAD"
        ),
        "config": config.normalized(),
        "target_signature": str(config.target_signature),
        "adapter_signature": str(config.adapter_signature),
        "artifact_signature": stable_frozen_neutra_artifact_signature(loaded_artifact),
        "transport_hash": loaded_artifact.manifest.transport_hash,
        "payload_stable_hash": f"sha256:{_stable_json_hash(payload)}",
        "source_training_state_file_sha256": training_state_file_hash,
        "source_training_state_artifact_hash": str(state.get("artifact_hash", "")),
        "source_training_state_path": str(config.phase16_training_state_path),
        "payload_path": str(config.payload_path),
        "validation_path": str(config.validation_path),
        "finite_checks": finite_checks,
        "boundary_checks": boundary_checks,
        "reference_residuals": {
            "initial_batch_value_residual": reference_value_residual,
            "initial_batch_score_residual": reference_score_residual,
        },
        "forward_probe": {
            "z": _json_safe(z),
            "theta": _json_safe(theta),
            "logdet": _json_safe(logdet),
            "base_value": _json_safe(base_value),
            "base_score": _json_safe(base_score),
        },
        "mechanics_manifest": None,
        "mechanics_diagnostics": {
            "fixed_transport_hmc_mechanics_executed": False,
            "deferred_to_phase18": True,
        },
        "mechanics_value": None,
        "mechanics_score": None,
        "training_summary": {
            "source_initial_loss": float(state["initial_loss"]),
            "source_final_loss": float(state["final_loss"]),
            "source_steps": int(state["optimizer_steps_executed"]),
            "source_batch_size": int(state["config"]["batch_size"]),
            "source_learning_rate": float(state["config"]["learning_rate"]),
            "loss_is_explanatory_only": True,
        },
        "elapsed_seconds": float(elapsed_seconds),
        "device": "cpu_hidden_for_loader_reference",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
        "hmc_executed": False,
        "external_sample_generation_executed": False,
        "training_executed": False,
        "source_jit_compile": True,
        "fixed_transport_hmc_mechanics_executed": False,
        "fixed_transport_hmc_mechanics_deferred_to_phase18": True,
        "jit_compile_runtime_executed": False,
        "jit_compile_false_runtime_executed": False,
        "xla_readiness_claimed": False,
        "nonclaims": NEUTRA_GPU_AFFINE_PAYLOAD_NONCLAIMS,
    }
    return {
        **validation_without_hash,
        "validation_stable_hash": f"sha256:{_stable_json_hash(validation_without_hash)}",
        "artifact_hash_semantics": "stable_json_sha256_excluding_self_hash_field",
    }


def _read_json_mapping(path: Path, label: str) -> Mapping[str, Any]:
    if not path.exists():
        raise NeuTraGPUAffinePayloadError(f"{label} is missing: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return _require_mapping(payload, label)


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise NeuTraGPUAffinePayloadError(f"{name} must be a JSON object")
    return value


def _float_tuple(value: Any, name: str) -> tuple[float, ...]:
    if value is None:
        raise NeuTraGPUAffinePayloadError(f"{name} is required")
    try:
        result = tuple(float(item) for item in value)
    except TypeError as exc:
        raise NeuTraGPUAffinePayloadError(f"{name} must be a numeric sequence") from exc
    if not result:
        raise NeuTraGPUAffinePayloadError(f"{name} must be nonempty")
    if not _sequence_is_finite(result):
        raise NeuTraGPUAffinePayloadError(f"{name} must be finite")
    return result


def _sequence_is_finite(values: Sequence[float]) -> bool:
    return all(_is_finite_number(float(item)) for item in values)


def _tensor_is_finite(value: Any) -> bool:
    tensor = tf.convert_to_tensor(value, dtype=tf.float64)
    return bool(tf.reduce_all(tf.math.is_finite(tensor)).numpy())


def _is_finite_number(value: float) -> bool:
    return value == value and abs(value) < float("inf")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _stable_json_hash(payload: Any) -> str:
    blob = json.dumps(_json_safe(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_sha256_hex(value: str, name: str) -> None:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise NeuTraGPUAffinePayloadError(f"{name} must be a lowercase sha256 hex")


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
    """CLI entrypoint for Phase 17 payload packaging."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase16-training-state-path",
        type=Path,
        default=PHASE16_TRAINING_STATE_PATH,
    )
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_PHASE16_ARTIFACT_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)
    result = package_and_validate_gpu_affine_payload(
        NeuTraGPUAffinePayloadConfig(
            phase16_training_state_path=args.phase16_training_state_path,
            artifact_dir=args.artifact_dir,
            seed=args.seed,
        )
    )
    print(
        json.dumps(
            {
                "passed": bool(result.passed),
                "payload_path": str(result.payload_path),
                "validation_path": str(result.validation_path),
                "target_signature": result.validation["target_signature"],
                "adapter_signature": result.validation["adapter_signature"],
                "artifact_signature": result.validation["artifact_signature"],
                "transport_hash": result.validation["transport_hash"],
                "nonclaims": NEUTRA_GPU_AFFINE_PAYLOAD_NONCLAIMS,
            },
            sort_keys=True,
        )
    )
    if not result.passed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
