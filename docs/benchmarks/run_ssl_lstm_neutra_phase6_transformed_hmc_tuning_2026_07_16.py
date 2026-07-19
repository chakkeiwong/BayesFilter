#!/usr/bin/env python3
"""Bounded identity-mass HMC tuning for the frozen SSL-LSTM G/H charts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shlex
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import tensorflow as tf


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bayesfilter.inference.batched_value_score import (  # noqa: E402
    FixedTransportValueScoreAdapter,
)
from bayesfilter.inference.hmc import (  # noqa: E402
    FullChainHMCConfig,
    build_reusable_full_chain_tfp_hmc_runner,
)
from bayesfilter.inference.neutra_artifacts import (  # noqa: E402
    load_frozen_neutra_artifact,
)
from bayesfilter.inference.posterior_adapter import ValueScoreCapability  # noqa: E402
from bayesfilter.nonlinear.ssl_lstm_posterior_tf import (  # noqa: E402
    FREE_PARAMETER_NAMES,
    locked_ssl_lstm_posterior_target,
)


PLAN_PATH = Path(
    "docs/plans/bayesfilter-ssl-lstm-neutra-phase-6-transformed-hmc-tuning-"
    "plan-2026-07-16.md"
)
RESULT_PATH = Path(
    "docs/plans/bayesfilter-ssl-lstm-neutra-phase-6-transformed-hmc-tuning-"
    "result-2026-07-16.md"
)
SCRIPT_PATH = Path(__file__).resolve().relative_to(ROOT)
ARTIFACT_ROOT = Path(
    "docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/"
    "trial0-alternative-confirmation-2026-07-16"
)
PHASE5_RECEIPT_PATH = Path(
    "docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/phase-5-trial0-gh/"
    "gpu-xla-r2.json"
)
PHASE5_RECEIPT_SHA256 = (
    "f855fd3fc83260867582a79b024087efbc5ecd463321ddd98f5fae7c9056f55b"
)
LADDER_R2_RUNNER_SHA256 = (
    "ea903b2af5cdd8476aea1bc38841c6379c226ce51a0668999525f426015f97c3"
)
TARGET_SIGNATURE = (
    "549efdf2aa5d9534226cb29c3678489d92766f92e6140901355eac33618f719e"
)
TARGET_ADAPTER_SIGNATURE = (
    "004f86b5668939febb629c563ca02625998c878d1e74d88c463f93b029a5d556"
)
TARGET_SOURCE = Path("bayesfilter/nonlinear/ssl_lstm_posterior_tf.py")
ADAPTER_SOURCE = Path("bayesfilter/inference/batched_value_score.py")
LOADER_SOURCE = Path("bayesfilter/inference/neutra_artifacts.py")
HMC_SOURCE = Path("bayesfilter/inference/hmc.py")
A0_LOCK_PATH = Path(
    "docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a0/target-lock.json"
)
A0_LOCK_SHA256 = (
    "1f7fccbeafbaa344a80e77c73b4356f44258b78a65ea2499e8ebd194b79a4383"
)
PAYLOADS = {
    "fresh-g": (
        ARTIFACT_ROOT / "fresh-g/best-frozen-payload.json",
        "6e147d5b33d003e0c895f294fc6b33523dcf97dc24af794d26a677886dedc354",
    ),
    "fresh-h": (
        ARTIFACT_ROOT / "fresh-h/best-frozen-payload.json",
        "ed0e42602aa39788ca1ea8d3c881d8bf85e15b91a687ef9adbe00a7b2c9120fb",
    ),
}
ORIGINAL_STARTS = (
    (0.0, 0.0, 0.0, 0.0),
    (0.5, -0.5, 0.5, -0.5),
    (-0.5, 0.5, -0.5, 0.5),
    (0.5, 0.5, -0.5, -0.5),
)
TARGET_SCOPE = (
    "ssl_lstm_completion:a1:masked_svd_ukf_four_parameter:"
    "phase6_fixed_transport_identity_hmc"
)
ROOT_SEED = 20260716
CANARY_SEEDS = {"fresh-g": (6101, 6102), "fresh-h": (6201, 6202)}
SCALE_SEED_BASE = {"fresh-g": 6300, "fresh-h": 6400}
TRAJECTORY_SEED_BASE = {"fresh-g": 6500, "fresh-h": 6600}
CONFIRMATION_SEEDS = {"fresh-g": 6701, "fresh-h": 6801}
H_REPAIR_SEED = 6901
INITIAL_SCALE_GRID = (0.05, 0.10, 0.20, 0.40)
HIGH_SCALE_EXPANSION = (0.80, 1.60)
LOW_SCALE_EXPANSION = (0.025, 0.0125)
TRAJECTORY_GRID = (2, 4, 8, 16)
TRAJECTORY_PRIORITY = (8, 4, 16, 2)
PILOT_ACCEPTANCE_BAND = (0.50, 0.90)
CONFIRMATION_ACCEPTANCE_BAND = (0.55, 0.85)
TARGET_ACCEPTANCE = 0.70
PILOT_MIN_MOVEMENT = 0.25
CONFIRMATION_MIN_MOVEMENT = 0.50
MIN_RMS_JUMP_DISTANCE = 0.05
CANARY_RESULTS = 4
CANARY_BURNIN = 2
PILOT_RESULTS = 16
PILOT_BURNIN = 8
CONFIRMATION_RESULTS = 64
CONFIRMATION_BURNIN = 32


class Phase6Error(RuntimeError):
    """Raised when a Phase 6 binding, runtime, or evidence invariant fails."""


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _sha256(path: Path) -> str:
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def _canonical(payload: Any) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "NaN"
        return "Infinity" if value > 0.0 else "-Infinity"
    if hasattr(value, "numpy"):
        return _json_safe(value.numpy())
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    if hasattr(value, "item"):
        return _json_safe(value.item())
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    absolute = ROOT / path
    absolute.parent.mkdir(parents=True, exist_ok=True)
    if absolute.exists():
        raise Phase6Error(f"refusing to overwrite receipt: {path}")
    absolute.write_bytes(_canonical(_json_safe(payload)))


def _git(*arguments: str) -> str:
    environment = dict(os.environ)
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    return subprocess.run(
        ("git", *arguments),
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _strict_json(path: Path) -> dict[str, Any]:
    def pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in rows:
            if key in result:
                raise Phase6Error(f"duplicate JSON key {key!r}: {path}")
            result[key] = value
        return result

    def reject(value: str) -> None:
        raise Phase6Error(f"nonfinite JSON constant {value!r}: {path}")

    value = json.loads(
        (ROOT / path).read_text(encoding="utf-8"),
        object_pairs_hook=pairs,
        parse_constant=reject,
    )
    if not isinstance(value, dict):
        raise Phase6Error(f"expected JSON object: {path}")
    return value


def _source_bindings() -> dict[str, Any]:
    return {
        name: {"path": path.as_posix(), "sha256": _sha256(path)}
        for name, path in {
            "plan": PLAN_PATH,
            "runner": SCRIPT_PATH,
            "target": TARGET_SOURCE,
            "adapter": ADAPTER_SOURCE,
            "artifact_loader": LOADER_SOURCE,
            "hmc_runtime": HMC_SOURCE,
        }.items()
    }


def validate_phase5_receipt() -> dict[str, Any]:
    if _sha256(PHASE5_RECEIPT_PATH) != PHASE5_RECEIPT_SHA256:
        raise Phase6Error("Phase 5 trusted GPU/XLA receipt byte identity drift")
    receipt = _strict_json(PHASE5_RECEIPT_PATH)
    if receipt.get("status") != "PASSED":
        raise Phase6Error("Phase 5 receipt status is not PASSED")
    if receipt.get("decision") != "PHASE5_EXACT_TRANSFORMED_TARGET_PASSED":
        raise Phase6Error("Phase 5 decision drift")
    if receipt.get("mode") != "gpu-xla" or not receipt.get("no_hmc"):
        raise Phase6Error("Phase 5 route semantics drift")
    rows = {row.get("label"): row for row in receipt.get("rows", ())}
    for label, (path, payload_hash) in PAYLOADS.items():
        row = rows.get(label)
        if not isinstance(row, Mapping) or row.get("status") != "PASSED":
            raise Phase6Error(f"Phase 5 passing row missing: {label}")
        if row.get("target_signature") != TARGET_SIGNATURE:
            raise Phase6Error(f"Phase 5 target signature drift: {label}")
        if row.get("manifest", {}).get("payload_path") != path.as_posix():
            raise Phase6Error(f"Phase 5 payload path drift: {label}")
        if row.get("manifest", {}).get("payload_sha256") != payload_hash:
            raise Phase6Error(f"Phase 5 payload hash drift: {label}")
        if not row.get("compiled_transformed_program", {}).get("jit_compile"):
            raise Phase6Error(f"Phase 5 XLA evidence missing: {label}")
    for key, path in {
        "target": TARGET_SOURCE,
        "adapter": ADAPTER_SOURCE,
        "artifact_loader": LOADER_SOURCE,
    }.items():
        expected = receipt.get("source_bindings", {}).get(key, {}).get("sha256")
        if expected != _sha256(path):
            raise Phase6Error(f"source binding drift since Phase 5: {path}")
    return {
        "path": PHASE5_RECEIPT_PATH.as_posix(),
        "sha256": PHASE5_RECEIPT_SHA256,
        "decision": receipt["decision"],
        "source_bindings_revalidated": True,
    }


class TargetBatchBridge:
    """Phase-5-identical locked-target value/score bridge."""

    parameter_dim = 4

    def __init__(self, target: Any) -> None:
        self.target = target
        if int(target.parameter_dim) != 4:
            raise Phase6Error("locked target parameter dimension drift")
        if tuple(target.parameter_names) != tuple(FREE_PARAMETER_NAMES):
            raise Phase6Error("locked target parameter order drift")

    def adapter_signature(self) -> str:
        return self.target.adapter_signature()

    def target_signature(self) -> str:
        return self.target.target_signature()

    def value_score_capability(self) -> ValueScoreCapability:
        base = self.target.value_score_capability()
        if base.value_score_authority != "graph_native":
            raise Phase6Error("locked target is not graph-native")
        return ValueScoreCapability(
            value_score_authority=base.value_score_authority,
            xla_hmc_ready=True,
            full_chain_xla_diagnostic_ready=True,
            runtime_backend="phase6_receipt_bound_ssl_lstm_target_bridge",
            evidence_path=PHASE5_RECEIPT_PATH.as_posix(),
            target_scope=TARGET_SCOPE,
            nonclaims=(
                "Phase 6 identity-mass transformed-HMC diagnostics only",
                "authority is bound to the exact passing Phase 5 GPU/XLA receipt",
                "does not mutate or promote locked target capability metadata",
                "no convergence, posterior correctness, or default-readiness claim",
            ),
        )

    def log_prob_and_grad(self, values: Any) -> tuple[tf.Tensor, tf.Tensor]:
        tensor = tf.convert_to_tensor(values, tf.float64)
        if tensor.shape.rank == 1:
            return self.target.value_and_score(tensor)
        if tensor.shape.rank == 2:
            return self.target.batch_value_and_score(tensor)
        raise ValueError("locked target bridge requires rank 1 or 2")


def load_binding(label: str) -> tuple[Any, tf.Tensor, dict[str, Any]]:
    target = locked_ssl_lstm_posterior_target()
    if target.target_signature() != TARGET_SIGNATURE:
        raise Phase6Error("live target semantic signature drift")
    if target.adapter_signature() != TARGET_ADAPTER_SIGNATURE:
        raise Phase6Error("live target adapter signature drift")
    path, expected_hash = PAYLOADS[label]
    if _sha256(path) != expected_hash:
        raise Phase6Error(f"payload byte hash drift: {label}")
    payload = _strict_json(path)
    artifact = load_frozen_neutra_artifact(
        payload,
        expected_target_signature=TARGET_SIGNATURE,
    )
    bridge = TargetBatchBridge(target)
    adapter = FixedTransportValueScoreAdapter(
        base_adapter=bridge,
        transport=artifact.transport,
        target_scope=TARGET_SCOPE,
        runtime_backend="phase6_fixed_dense_iaf_identity_mass_hmc",
        evidence_path=PHASE5_RECEIPT_PATH.as_posix(),
        xla_hmc_ready=True,
        full_chain_xla_diagnostic_ready=True,
        require_batch_native=True,
        nonclaims=(
            "immutable fixed transport under Phase 6 HMC-scoped authority",
            "identity mass only; no learned or adapted HMC mass",
            "tuning samples are not retained posterior evidence",
        ),
    )
    base_capability = target.value_score_capability()
    scoped_capability = adapter.value_score_capability()
    if base_capability.xla_hmc_ready or base_capability.full_chain_xla_diagnostic_ready:
        raise Phase6Error("locked target capability was unexpectedly promoted")
    if not scoped_capability.is_accepted_full_chain_xla_diagnostic_authority:
        raise Phase6Error("Phase 6 scoped HMC authority did not bind")
    if _sha256(A0_LOCK_PATH) != A0_LOCK_SHA256:
        raise Phase6Error("A0 target-lock byte identity drift")
    lock = _strict_json(A0_LOCK_PATH)
    geometry = lock.get("sampler_geometry")
    if not isinstance(geometry, Mapping):
        raise Phase6Error("A0 sampler geometry missing")
    center = tf.constant(geometry["center_free"]["values"], tf.float64)
    scale = tf.constant(geometry["scale"]["values"], tf.float64)
    factor_z = tf.constant(geometry["factor_z"]["values"], tf.float64)
    factor = tf.linalg.diag(scale) @ factor_z
    original_affine_z = tf.constant(ORIGINAL_STARTS, tf.float64)
    original_theta = center + original_affine_z @ tf.transpose(factor)
    initial_z = artifact.transport.inverse_theta_to_z_batch(original_theta)
    replay = artifact.transport.forward_z_to_theta_batch(initial_z)
    roundtrip = float(tf.reduce_max(tf.abs(replay - original_theta)).numpy())
    if roundtrip > 1.0e-9 or not bool(tf.reduce_all(tf.math.is_finite(initial_z)).numpy()):
        raise Phase6Error(f"original-start inverse mapping failed: {label}")
    distinct = tf.reduce_all(
        tf.linalg.norm(initial_z[:, tf.newaxis, :] - initial_z[tf.newaxis, :, :], axis=-1)
        + tf.eye(4, dtype=tf.float64)
        > 0.0
    )
    if not bool(distinct.numpy()):
        raise Phase6Error(f"inverse-mapped starts are not distinct: {label}")
    binding = {
        "label": label,
        "payload_path": path.as_posix(),
        "payload_sha256": expected_hash,
        "transport_hash": artifact.manifest.transport_hash,
        "tensor_hash": artifact.manifest.tensor_hash,
        "topology_hash": artifact.manifest.topology_hash,
        "artifact_signature": artifact.artifact_signature,
        "target_signature": TARGET_SIGNATURE,
        "a0_target_lock_path": A0_LOCK_PATH.as_posix(),
        "a0_target_lock_sha256": A0_LOCK_SHA256,
        "a0_sampler_geometry_sha256": lock["signatures"]["sampler_geometry_sha256"],
        "original_start_coordinate_system": "A0_affine_latent_z",
        "original_affine_z": _json_safe(original_affine_z),
        "original_theta": _json_safe(original_theta),
        "base_target_capability_unchanged": {
            "xla_hmc_ready": base_capability.xla_hmc_ready,
            "full_chain_xla_diagnostic_ready": (
                base_capability.full_chain_xla_diagnostic_ready
            ),
        },
        "scoped_adapter_signature": adapter.adapter_signature(),
        "scoped_target_scope": TARGET_SCOPE,
        "initial_z": _json_safe(initial_z),
        "initial_z_radii": _json_safe(tf.linalg.norm(initial_z, axis=1)),
        "original_start_roundtrip_max_abs": roundtrip,
        "four_distinct_starts": True,
    }
    return adapter, initial_z, binding


def identity_mass_fixture() -> dict[str, Any]:
    momentum = tf.constant(
        [[1.0, -2.0, 0.5, 3.0], [-1.5, 0.25, 2.0, -0.75]],
        tf.float64,
    )
    identity = tf.eye(4, dtype=tf.float64)
    kinetic = 0.5 * tf.reduce_sum(tf.square(momentum), axis=1)
    explicit = 0.5 * tf.einsum("bi,ij,bj->b", momentum, identity, momentum)
    residual = float(tf.reduce_max(tf.abs(kinetic - explicit)).numpy())
    if residual != 0.0:
        raise Phase6Error("identity-mass kinetic-energy fixture failed")
    return {
        "mass_matrix": _json_safe(identity),
        "precision_matrix": _json_safe(identity),
        "momentum_covariance": _json_safe(identity),
        "cholesky_orientation": "identity_no_transform",
        "kinetic_formula": "0.5 * p^T p",
        "kinetic_energy": _json_safe(kinetic),
        "explicit_residual_max_abs": residual,
        "passed": True,
    }


def diagnose_run(
    *,
    samples: Any,
    initial_state: Any,
    trace: Mapping[str, Any],
    acceptance_band: tuple[float, float] | None,
    min_movement: float,
    min_rms_jump: float,
) -> dict[str, Any]:
    sample_tensor = tf.convert_to_tensor(samples, tf.float64)
    initial_tensor = tf.convert_to_tensor(initial_state, tf.float64)
    accepted = tf.convert_to_tensor(trace.get("is_accepted"), tf.bool)
    log_accept = tf.convert_to_tensor(trace.get("log_accept_ratio"), tf.float64)
    target = tf.convert_to_tensor(trace.get("target_log_prob"), tf.float64)
    if sample_tensor.shape.rank != 3 or tuple(sample_tensor.shape[1:]) != (4, 4):
        raise Phase6Error("HMC samples must have shape [draw,4,4]")
    if tuple(initial_tensor.shape) != (4, 4):
        raise Phase6Error("HMC initial state must have shape [4,4]")
    if tuple(accepted.shape) != tuple(sample_tensor.shape[:2]):
        raise Phase6Error("per-chain acceptance trace shape mismatch")
    if tuple(log_accept.shape) != tuple(sample_tensor.shape[:2]):
        raise Phase6Error("per-chain log-acceptance trace shape mismatch")
    if tuple(target.shape) != tuple(sample_tensor.shape[:2]):
        raise Phase6Error("per-chain target trace shape mismatch")
    previous = tf.concat((initial_tensor[tf.newaxis, ...], sample_tensor[:-1]), axis=0)
    jump = sample_tensor - previous
    jump_norm = tf.linalg.norm(jump, axis=-1)
    movement = tf.reduce_mean(tf.cast(jump_norm > 0.0, tf.float64), axis=0)
    any_moved = tf.reduce_any(jump_norm > 0.0, axis=0)
    rms_jump = tf.sqrt(tf.reduce_mean(tf.square(jump_norm), axis=0))
    acceptance = tf.reduce_mean(tf.cast(accepted, tf.float64), axis=0)
    finite = {
        "samples": bool(tf.reduce_all(tf.math.is_finite(sample_tensor)).numpy()),
        "log_accept_ratio": bool(tf.reduce_all(tf.math.is_finite(log_accept)).numpy()),
        "target_log_prob": bool(tf.reduce_all(tf.math.is_finite(target)).numpy()),
    }
    for key in ("proposed_target_log_prob", "log_acceptance_correction"):
        if key in trace:
            values = tf.convert_to_tensor(trace[key], tf.float64)
            if tuple(values.shape) != tuple(sample_tensor.shape[:2]):
                raise Phase6Error(f"per-chain {key} trace shape mismatch")
            finite[key] = bool(tf.reduce_all(tf.math.is_finite(values)).numpy())
    divergence_status = "unavailable_not_zero"
    divergence_count = None
    divergence_by_chain = None
    if "divergence" in trace:
        divergence = tf.convert_to_tensor(trace["divergence"], tf.bool)
        if tuple(divergence.shape) != tuple(sample_tensor.shape[:2]):
            raise Phase6Error("native divergence trace shape mismatch")
        divergence_status = "available"
        divergence_count = int(tf.reduce_sum(tf.cast(divergence, tf.int32)).numpy())
        divergence_by_chain = _json_safe(
            tf.reduce_sum(tf.cast(divergence, tf.int32), axis=0)
        )
    hard_vetoes: list[str] = []
    if not all(finite.values()):
        hard_vetoes.append("nonfinite_hmc_telemetry")
    if not bool(tf.reduce_all(any_moved).numpy()):
        hard_vetoes.append("unmoved_chain")
    if divergence_count is not None and divergence_count > 0:
        hard_vetoes.append("positive_native_divergence")
    if bool(tf.reduce_any(movement < min_movement).numpy()):
        hard_vetoes.append("per_chain_movement_below_threshold")
    if bool(tf.reduce_any(rms_jump < min_rms_jump).numpy()):
        hard_vetoes.append("per_chain_rms_jump_below_threshold")
    acceptance_vetoes: list[str] = []
    if acceptance_band is not None:
        low, high = acceptance_band
        if bool(tf.reduce_any(acceptance < low).numpy()):
            acceptance_vetoes.append("per_chain_acceptance_below_band")
        if bool(tf.reduce_any(acceptance > high).numpy()):
            acceptance_vetoes.append("per_chain_acceptance_above_band")
    viable = not hard_vetoes and not acceptance_vetoes
    return {
        "viable": viable,
        "finite": finite,
        "acceptance_rate": float(tf.reduce_mean(acceptance).numpy()),
        "acceptance_rate_by_chain": _json_safe(acceptance),
        "movement_rate_by_chain": _json_safe(movement),
        "chain_moved": _json_safe(any_moved),
        "rms_jump_distance_by_chain": _json_safe(rms_jump),
        "native_divergence_status": divergence_status,
        "native_divergence_count": divergence_count,
        "native_divergence_count_by_chain": divergence_by_chain,
        "hard_vetoes": hard_vetoes,
        "acceptance_vetoes": acceptance_vetoes,
        "thresholds": {
            "acceptance_band": acceptance_band,
            "minimum_movement_rate": min_movement,
            "minimum_rms_jump_distance": min_rms_jump,
        },
        "aggregate_acceptance_is_explanatory_only": True,
    }


def _build_runner(
    adapter: Any,
    initial_z: tf.Tensor,
    *,
    num_results: int,
    num_burnin_steps: int,
    leapfrog_steps: int,
) -> Any:
    config = FullChainHMCConfig(
        num_results=num_results,
        num_burnin_steps=num_burnin_steps,
        step_size=0.01,
        num_leapfrog_steps=leapfrog_steps,
        seed=(ROOT_SEED, 1),
        use_xla=True,
        trace_policy="standard",
        target_status_trace_policy="none",
        adaptation_policy="fixed_kernel_no_adaptation",
        target_scope=TARGET_SCOPE,
        chain_execution_mode="tf_function",
    )
    return build_reusable_full_chain_tfp_hmc_runner(
        adapter,
        initial_z,
        config,
        dynamic_num_leapfrog_steps=True,
    )


def _run_arm(
    runner: Any,
    *,
    initial_z: tf.Tensor,
    step_size: float,
    leapfrog_steps: int,
    seed_word: int,
    role: str,
    acceptance_band: tuple[float, float] | None,
    min_movement: float,
    min_rms_jump: float,
) -> dict[str, Any]:
    result = runner.run(
        current_state=initial_z,
        seed=(ROOT_SEED, seed_word),
        step_size=step_size,
        num_leapfrog_steps=leapfrog_steps,
    )
    diagnostics = diagnose_run(
        samples=result.samples,
        initial_state=initial_z,
        trace=result.trace,
        acceptance_band=acceptance_band,
        min_movement=min_movement,
        min_rms_jump=min_rms_jump,
    )
    return {
        "role": role,
        "seed": [ROOT_SEED, seed_word],
        "step_size": step_size,
        "num_leapfrog_steps": leapfrog_steps,
        "trajectory_length": step_size * leapfrog_steps,
        "diagnostics": diagnostics,
        "runner_diagnostics": _json_safe(result.diagnostics),
        "runner_metadata": _json_safe(result.metadata),
        "samples_retained_as_posterior_evidence": False,
    }


def _check_cap(started: float, wall_cap_seconds: float) -> None:
    if time.perf_counter() - started > wall_cap_seconds:
        raise Phase6Error("Phase 6 wall cap exhausted")


def run_canary(*, output: Path, wall_cap_seconds: float) -> dict[str, Any]:
    started_at = _now()
    started = time.perf_counter()
    phase5 = validate_phase5_receipt()
    fixture = identity_mass_fixture()
    rows = []
    for label in PAYLOADS:
        _check_cap(started, wall_cap_seconds)
        adapter, initial_z, binding = load_binding(label)
        runner = _build_runner(
            adapter,
            initial_z,
            num_results=CANARY_RESULTS,
            num_burnin_steps=CANARY_BURNIN,
            leapfrog_steps=2,
        )
        first_seed, warm_seed = CANARY_SEEDS[label]
        first = _run_arm(
            runner,
            initial_z=initial_z,
            step_size=0.01,
            leapfrog_steps=2,
            seed_word=first_seed,
            role="canary_first_compile_execute",
            acceptance_band=None,
            min_movement=0.0,
            min_rms_jump=0.0,
        )
        _check_cap(started, wall_cap_seconds)
        warm = _run_arm(
            runner,
            initial_z=initial_z,
            step_size=0.01,
            leapfrog_steps=2,
            seed_word=warm_seed,
            role="canary_warm_execute",
            acceptance_band=None,
            min_movement=0.0,
            min_rms_jump=0.0,
        )
        _check_cap(started, wall_cap_seconds)
        canary_vetoes = []
        for name, arm in (("first", first), ("warm", warm)):
            diagnostics = arm["diagnostics"]
            if not all(diagnostics["finite"].values()):
                canary_vetoes.append(f"{name}_nonfinite_telemetry")
            if not all(diagnostics["chain_moved"]):
                canary_vetoes.append(f"{name}_unmoved_chain")
            if (diagnostics["native_divergence_count"] or 0) > 0:
                canary_vetoes.append(f"{name}_positive_native_divergence")
        rows.append(
            {
                "label": label,
                "status": "PASSED" if not canary_vetoes else "VETOED",
                "binding": binding,
                "calls": [first, warm],
                "canary_vetoes": canary_vetoes,
            }
        )
    elapsed = time.perf_counter() - started
    passed = all(row["status"] == "PASSED" for row in rows)
    payload = {
        "schema": "bayesfilter.ssl_lstm_neutra.phase6_transformed_hmc_canary.v1",
        "status": "PASSED" if passed else "VETOED",
        "decision": (
            "PHASE6_STAGE_A_CANARY_PASSED_BUDGET_FREEZE_REQUIRED"
            if passed
            else "PHASE6_STAGE_A_CONTINUATION_VETO"
        ),
        "stage": "canary",
        "phase5_binding": phase5,
        "identity_mass_fixture": fixture,
        "rows": rows,
        "source_bindings": _source_bindings(),
        "run_manifest": _run_manifest(
            started_at=started_at,
            wall_time=elapsed,
            wall_cap_seconds=wall_cap_seconds,
            output=output,
            seeds=CANARY_SEEDS,
        ),
        "nonclaims": [
            "mechanics and timing canary only",
            "no HMC kernel nomination or freeze",
            "no retained posterior evidence, convergence, posterior correctness, ranking, predictive validity, or default readiness",
        ],
    }
    _write_json(output, payload)
    return payload


def _scale_expansion(rows_by_label: Mapping[str, Sequence[Mapping[str, Any]]]) -> tuple[float, ...]:
    relations: list[str] = []
    for rows in rows_by_label.values():
        rates = [
            value
            for row in rows
            for value in row["diagnostics"]["acceptance_rate_by_chain"]
        ]
        if all(value > PILOT_ACCEPTANCE_BAND[1] for value in rates):
            relations.append("all_high")
        elif all(value < PILOT_ACCEPTANCE_BAND[0] for value in rates):
            relations.append("all_low")
        else:
            relations.append("mixed")
    expansion: tuple[float, ...] = ()
    if "all_high" in relations:
        expansion += HIGH_SCALE_EXPANSION
    if "all_low" in relations:
        expansion += LOW_SCALE_EXPANSION
    return expansion


def _select_scale(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    viable = [row for row in rows if row["diagnostics"]["viable"]]
    if not viable:
        return None

    def key(row: Mapping[str, Any]) -> tuple[float, float]:
        rates = row["diagnostics"]["acceptance_rate_by_chain"]
        worst_deviation = max(abs(float(rate) - TARGET_ACCEPTANCE) for rate in rates)
        return round(worst_deviation, 12), -float(row["step_size"])

    return min(viable, key=key)


def _select_trajectory(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    by_leapfrog = {
        int(row["num_leapfrog_steps"]): row
        for row in rows
        if row["diagnostics"]["viable"]
    }
    return next((by_leapfrog[value] for value in TRAJECTORY_PRIORITY if value in by_leapfrog), None)


def run_ladder(
    *,
    output: Path,
    wall_cap_seconds: float,
    canary_receipt: Path,
    canary_sha256: str,
) -> dict[str, Any]:
    started_at = _now()
    started = time.perf_counter()
    phase5 = validate_phase5_receipt()
    if _sha256(canary_receipt) != canary_sha256:
        raise Phase6Error("authorized canary receipt hash mismatch")
    canary = _strict_json(canary_receipt)
    if canary.get("decision") != "PHASE6_STAGE_A_CANARY_PASSED_BUDGET_FREEZE_REQUIRED":
        raise Phase6Error("Stage A canary did not pass")
    for key, path in {
        "runner": SCRIPT_PATH,
        "hmc_runtime": HMC_SOURCE,
        "target": TARGET_SOURCE,
        "adapter": ADAPTER_SOURCE,
        "artifact_loader": LOADER_SOURCE,
    }.items():
        expected = canary.get("source_bindings", {}).get(key, {}).get("sha256")
        if expected != _sha256(path):
            raise Phase6Error(f"source binding drift since Stage A canary: {path}")
    rows_by_label: dict[str, list[dict[str, Any]]] = {}
    scale_runners: dict[str, Any] = {}
    bindings: dict[str, Any] = {}
    contexts: dict[str, tuple[Any, tf.Tensor]] = {}
    for label in PAYLOADS:
        adapter, initial_z, binding = load_binding(label)
        contexts[label] = (adapter, initial_z)
        bindings[label] = binding
        runner = _build_runner(
            adapter,
            initial_z,
            num_results=PILOT_RESULTS,
            num_burnin_steps=PILOT_BURNIN,
            leapfrog_steps=4,
        )
        scale_runners[label] = runner
        rows = []
        for index, step_size in enumerate(INITIAL_SCALE_GRID):
            _check_cap(started, wall_cap_seconds)
            rows.append(
                _run_arm(
                    runner,
                    initial_z=initial_z,
                    step_size=step_size,
                    leapfrog_steps=4,
                    seed_word=SCALE_SEED_BASE[label] + index,
                    role="scale_pilot",
                    acceptance_band=PILOT_ACCEPTANCE_BAND,
                    min_movement=PILOT_MIN_MOVEMENT,
                    min_rms_jump=MIN_RMS_JUMP_DISTANCE,
                )
            )
        rows_by_label[label] = rows
    expansion = _scale_expansion(rows_by_label)
    if expansion:
        for label, rows in rows_by_label.items():
            adapter, initial_z = contexts[label]
            runner = scale_runners[label]
            offset = len(INITIAL_SCALE_GRID)
            for local_index, step_size in enumerate(expansion):
                _check_cap(started, wall_cap_seconds)
                rows.append(
                    _run_arm(
                        runner,
                        initial_z=initial_z,
                        step_size=step_size,
                        leapfrog_steps=4,
                        seed_word=SCALE_SEED_BASE[label] + offset + local_index,
                        role="scale_expansion",
                        acceptance_band=PILOT_ACCEPTANCE_BAND,
                        min_movement=PILOT_MIN_MOVEMENT,
                        min_rms_jump=MIN_RMS_JUMP_DISTANCE,
                    )
                )
    selected_scale = {label: _select_scale(rows) for label, rows in rows_by_label.items()}
    trajectory_rows: dict[str, list[dict[str, Any]]] = {label: [] for label in PAYLOADS}
    selected_trajectory: dict[str, Mapping[str, Any] | None] = {
        label: None for label in PAYLOADS
    }
    if all(selected_scale.values()):
        for label in PAYLOADS:
            adapter, initial_z = contexts[label]
            step_size = float(selected_scale[label]["step_size"])  # type: ignore[index]
            runner = scale_runners[label]
            for index, leapfrog in enumerate(TRAJECTORY_GRID):
                _check_cap(started, wall_cap_seconds)
                trajectory_rows[label].append(
                    _run_arm(
                        runner,
                        initial_z=initial_z,
                        step_size=step_size,
                        leapfrog_steps=leapfrog,
                        seed_word=TRAJECTORY_SEED_BASE[label] + index,
                        role="trajectory_grid",
                        acceptance_band=PILOT_ACCEPTANCE_BAND,
                        min_movement=PILOT_MIN_MOVEMENT,
                        min_rms_jump=MIN_RMS_JUMP_DISTANCE,
                    )
                )
            selected_trajectory[label] = _select_trajectory(trajectory_rows[label])
    confirmation: dict[str, Any] = {}
    if all(selected_trajectory.values()):
        for label in PAYLOADS:
            adapter, initial_z = contexts[label]
            selected = selected_trajectory[label]
            assert selected is not None
            leapfrog = int(selected["num_leapfrog_steps"])
            runner = _build_runner(
                adapter,
                initial_z,
                num_results=CONFIRMATION_RESULTS,
                num_burnin_steps=CONFIRMATION_BURNIN,
                leapfrog_steps=leapfrog,
            )
            _check_cap(started, wall_cap_seconds)
            confirmation[label] = _run_arm(
                runner,
                initial_z=initial_z,
                step_size=float(selected["step_size"]),
                leapfrog_steps=leapfrog,
                seed_word=CONFIRMATION_SEEDS[label],
                role="fresh_longer_confirmation",
                acceptance_band=CONFIRMATION_ACCEPTANCE_BAND,
                min_movement=CONFIRMATION_MIN_MOVEMENT,
                min_rms_jump=MIN_RMS_JUMP_DISTANCE,
            )
    _check_cap(started, wall_cap_seconds)
    passed = bool(
        len(confirmation) == len(PAYLOADS)
        and all(row["diagnostics"]["viable"] for row in confirmation.values())
    )
    selected_kernels = (
        {
            label: {
                "mass_matrix": "identity",
                "step_size": row["step_size"],
                "num_leapfrog_steps": row["num_leapfrog_steps"],
                "trajectory_length": row["trajectory_length"],
                "transport_hash": bindings[label]["transport_hash"],
                "target_signature": TARGET_SIGNATURE,
                "selection_rule": "minimum_worst_chain_deviation_from_acceptance_0p70_then_fixed_trajectory_priority_8_4_16_2",
            }
            for label, row in confirmation.items()
        }
        if passed
        else {}
    )
    elapsed = time.perf_counter() - started
    if passed:
        status = "PASSED"
        decision = "PHASE6_IDENTITY_MASS_KERNELS_FROZEN"
    elif not all(selected_scale.values()):
        status = "IDENTITY_MASS_REPAIR_REQUIRED"
        decision = "PHASE6_IDENTITY_MASS_SCALE_SEARCH_EXHAUSTED"
    elif not all(selected_trajectory.values()):
        status = "IDENTITY_MASS_REPAIR_REQUIRED"
        decision = "PHASE6_IDENTITY_MASS_TRAJECTORY_SEARCH_EXHAUSTED"
    else:
        status = "IDENTITY_MASS_REPAIR_REQUIRED"
        decision = "PHASE6_IDENTITY_MASS_CONFIRMATION_FAILED"
    payload = {
        "schema": "bayesfilter.ssl_lstm_neutra.phase6_transformed_hmc_ladder.v1",
        "status": status,
        "decision": decision,
        "stage": "ladder",
        "phase5_binding": phase5,
        "canary_binding": {
            "path": canary_receipt.as_posix(),
            "sha256": canary_sha256,
        },
        "identity_mass_fixture": identity_mass_fixture(),
        "bindings": bindings,
        "scale_pilot": rows_by_label,
        "scale_expansion": list(expansion),
        "selected_scale": _json_safe(selected_scale),
        "trajectory_grid": trajectory_rows,
        "selected_trajectory": _json_safe(selected_trajectory),
        "confirmation": confirmation,
        "selected_kernels": selected_kernels,
        "source_bindings": _source_bindings(),
        "run_manifest": _run_manifest(
            started_at=started_at,
            wall_time=elapsed,
            wall_cap_seconds=wall_cap_seconds,
            output=output,
            seeds={
                "scale_bases": SCALE_SEED_BASE,
                "trajectory_bases": TRAJECTORY_SEED_BASE,
                "confirmation": CONFIRMATION_SEEDS,
            },
        ),
        "inference_status": {
            "hard_veto_screen": "passed" if passed else "see_per_arm_diagnostics",
            "viable_candidates": list(selected_kernels),
            "statistically_supported_ranking": "none",
            "descriptive_only_differences": [
                "acceptance rates",
                "movement rates",
                "RMS jump distances",
                "runtime",
                "G/H differences",
            ],
            "default_readiness": "not_assessed",
            "next_evidence_needed": (
                "independent retained-chain admission"
                if passed
                else "separately planned identity-mass failure diagnosis and repair"
            ),
        },
        "nonclaims": [
            "tuning viability only",
            "no G-versus-H superiority or ranking",
            "no retained posterior evidence, convergence, posterior correctness, predictive validity, or default readiness",
        ],
    }
    _write_json(output, payload)
    return payload


def run_h_confirmation_repair(
    *,
    output: Path,
    wall_cap_seconds: float,
    ladder_receipt: Path,
    ladder_sha256: str,
) -> dict[str, Any]:
    started_at = _now()
    started = time.perf_counter()
    phase5 = validate_phase5_receipt()
    if _sha256(ladder_receipt) != ladder_sha256:
        raise Phase6Error("authorized ladder receipt hash mismatch")
    ladder = _strict_json(ladder_receipt)
    if ladder.get("decision") != "PHASE6_IDENTITY_MASS_CONFIRMATION_FAILED":
        raise Phase6Error("repair requires the exact H confirmation failure")
    expected_sources = ladder.get("source_bindings", {})
    if expected_sources.get("runner", {}).get("sha256") != LADDER_R2_RUNNER_SHA256:
        raise Phase6Error("ladder predecessor runner binding mismatch")
    for key, path in {
        "hmc_runtime": HMC_SOURCE,
        "target": TARGET_SOURCE,
        "adapter": ADAPTER_SOURCE,
        "artifact_loader": LOADER_SOURCE,
    }.items():
        if expected_sources.get(key, {}).get("sha256") != _sha256(path):
            raise Phase6Error(f"source binding drift since ladder: {path}")
    g_confirmation = ladder.get("confirmation", {}).get("fresh-g")
    h_confirmation = ladder.get("confirmation", {}).get("fresh-h")
    if not isinstance(g_confirmation, Mapping) or not isinstance(h_confirmation, Mapping):
        raise Phase6Error("ladder confirmation rows missing")
    if not g_confirmation.get("diagnostics", {}).get("viable"):
        raise Phase6Error("G confirmation was not viable")
    if h_confirmation.get("diagnostics", {}).get("viable"):
        raise Phase6Error("H confirmation did not fail")
    if h_confirmation.get("diagnostics", {}).get("hard_vetoes"):
        raise Phase6Error("H confirmation repair cannot bypass a hard veto")
    if h_confirmation.get("diagnostics", {}).get("acceptance_vetoes") != [
        "per_chain_acceptance_below_band"
    ]:
        raise Phase6Error("H confirmation failure was not acceptance-low only")
    selected_scale = ladder.get("selected_scale", {}).get("fresh-h")
    trajectory_rows = ladder.get("trajectory_grid", {}).get("fresh-h", ())
    if not isinstance(selected_scale, Mapping) or float(selected_scale["step_size"]) != 0.8:
        raise Phase6Error("H selected scale is not the frozen 0.8")
    adjacent = next(
        (
            row
            for row in trajectory_rows
            if int(row["num_leapfrog_steps"]) == 4
        ),
        None,
    )
    if adjacent is None or not adjacent.get("diagnostics", {}).get("viable"):
        raise Phase6Error("prospective adjacent H L=4 rung was not viable")
    _check_cap(started, wall_cap_seconds)
    adapter, initial_z, h_binding = load_binding("fresh-h")
    runner = _build_runner(
        adapter,
        initial_z,
        num_results=CONFIRMATION_RESULTS,
        num_burnin_steps=CONFIRMATION_BURNIN,
        leapfrog_steps=4,
    )
    repair = _run_arm(
        runner,
        initial_z=initial_z,
        step_size=0.8,
        leapfrog_steps=4,
        seed_word=H_REPAIR_SEED,
        role="fresh_h_adjacent_trajectory_confirmation_repair",
        acceptance_band=CONFIRMATION_ACCEPTANCE_BAND,
        min_movement=CONFIRMATION_MIN_MOVEMENT,
        min_rms_jump=MIN_RMS_JUMP_DISTANCE,
    )
    _check_cap(started, wall_cap_seconds)
    passed = bool(repair["diagnostics"]["viable"])
    g_binding = ladder["bindings"]["fresh-g"]
    selected_kernels = (
        {
            "fresh-g": {
                "mass_matrix": "identity",
                "step_size": float(g_confirmation["step_size"]),
                "num_leapfrog_steps": int(g_confirmation["num_leapfrog_steps"]),
                "trajectory_length": float(g_confirmation["trajectory_length"]),
                "transport_hash": g_binding["transport_hash"],
                "target_signature": TARGET_SIGNATURE,
                "confirmation_source": ladder_receipt.as_posix(),
            },
            "fresh-h": {
                "mass_matrix": "identity",
                "step_size": 0.8,
                "num_leapfrog_steps": 4,
                "trajectory_length": 3.2,
                "transport_hash": h_binding["transport_hash"],
                "target_signature": TARGET_SIGNATURE,
                "confirmation_source": output.as_posix(),
            },
        }
        if passed
        else {}
    )
    elapsed = time.perf_counter() - started
    payload = {
        "schema": "bayesfilter.ssl_lstm_neutra.phase6_h_confirmation_repair.v1",
        "status": "PASSED" if passed else "IDENTITY_MASS_REPAIR_EXHAUSTED",
        "decision": (
            "PHASE6_IDENTITY_MASS_KERNELS_FROZEN_AFTER_H_REPAIR"
            if passed
            else "PHASE6_H_ADJACENT_CONFIRMATION_REPAIR_FAILED"
        ),
        "stage": "h-confirmation-repair",
        "phase5_binding": phase5,
        "ladder_binding": {
            "path": ladder_receipt.as_posix(),
            "sha256": ladder_sha256,
        },
        "repair_contract": {
            "label": "fresh-h",
            "step_size": 0.8,
            "num_leapfrog_steps": 4,
            "num_results": CONFIRMATION_RESULTS,
            "num_burnin_steps": CONFIRMATION_BURNIN,
            "seed": [ROOT_SEED, H_REPAIR_SEED],
            "trigger": "selected_L8_fresh_confirmation_acceptance_low_only",
            "justification": "prospectively_viable_adjacent_L4_shortens_trajectory_at_fixed_step",
            "new_candidate_search": False,
        },
        "g_confirmation_reused_without_rerun": g_confirmation,
        "h_binding": h_binding,
        "h_repair_confirmation": repair,
        "selected_kernels": selected_kernels,
        "source_bindings": _source_bindings(),
        "run_manifest": _run_manifest(
            started_at=started_at,
            wall_time=elapsed,
            wall_cap_seconds=wall_cap_seconds,
            output=output,
            seeds={"h_repair": [ROOT_SEED, H_REPAIR_SEED]},
        ),
        "inference_status": {
            "hard_veto_screen": (
                "passed" if passed else repair["diagnostics"]["hard_vetoes"]
            ),
            "viable_candidates": list(selected_kernels),
            "statistically_supported_ranking": "none",
            "descriptive_only_differences": [
                "G/H acceptance",
                "movement",
                "RMS jump distance",
                "runtime",
            ],
            "default_readiness": "not_assessed",
            "next_evidence_needed": (
                "independent retained-chain admission"
                if passed
                else "separate identity-mass diagnosis and repair plan"
            ),
        },
        "nonclaims": [
            "bounded adjacent-trajectory confirmation repair only",
            "no kernel superiority or G-versus-H ranking",
            "no retained posterior evidence, convergence, posterior correctness, predictive validity, or default readiness",
        ],
    }
    _write_json(output, payload)
    return payload


def _run_manifest(
    *,
    started_at: str,
    wall_time: float,
    wall_cap_seconds: float,
    output: Path,
    seeds: Any,
) -> dict[str, Any]:
    import tensorflow_probability as tfp

    physical = tf.config.list_physical_devices("GPU")
    logical = tf.config.list_logical_devices("GPU")
    return {
        "git_commit": _git("rev-parse", "HEAD").strip(),
        "git_dirty": bool(
            _git("status", "--porcelain=v1", "--untracked-files=all").strip()
        ),
        "command": shlex.join([sys.executable, *sys.argv]),
        "cwd": str(ROOT),
        "interpreter": sys.executable,
        "python_version": platform.python_version(),
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV", "unknown"),
        "tensorflow": tf.__version__,
        "tensorflow_probability": tfp.__version__,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "physical_devices": [item.name for item in physical],
        "logical_devices": [item.name for item in logical],
        "dtype": "float64",
        "jit_compile": True,
        "tf32_enabled": bool(
            tf.config.experimental.tensor_float_32_execution_enabled()
        ),
        "mass_matrix": "identity",
        "mass_representation": "momentum_covariance_equals_identity",
        "random_seeds": _json_safe(seeds),
        "started_at_utc": started_at,
        "completed_at_utc": _now(),
        "wall_time_seconds": wall_time,
        "wall_cap_seconds": wall_cap_seconds,
        "output_path": output.as_posix(),
        "plan_path": PLAN_PATH.as_posix(),
        "result_path": RESULT_PATH.as_posix(),
        "trust_basis": "owner_designated_managed_session_visible_gpu_trusted",
        "tuning_samples_retained": False,
    }


def _require_gpu() -> None:
    physical = tf.config.list_physical_devices("GPU")
    if not physical:
        raise Phase6Error("Phase 6 requires a visible trusted GPU")
    for gpu in physical:
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.experimental.enable_tensor_float_32_execution(True)
    if not tf.config.list_logical_devices("GPU"):
        raise Phase6Error("Phase 6 requires a logical GPU")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("canary", "ladder", "h-confirmation-repair"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wall-cap-seconds", type=float, required=True)
    parser.add_argument("--canary-receipt", type=Path)
    parser.add_argument("--canary-sha256")
    parser.add_argument("--ladder-receipt", type=Path)
    parser.add_argument("--ladder-sha256")
    args = parser.parse_args(argv)
    if not math.isfinite(args.wall_cap_seconds) or args.wall_cap_seconds <= 0.0:
        raise Phase6Error("wall cap must be positive and finite")
    _require_gpu()
    with tf.device("/GPU:0"):
        if args.stage == "canary":
            if any(
                value is not None
                for value in (
                    args.canary_receipt,
                    args.canary_sha256,
                    args.ladder_receipt,
                    args.ladder_sha256,
                )
            ):
                raise Phase6Error("canary stage does not accept canary binding arguments")
            payload = run_canary(
                output=args.output,
                wall_cap_seconds=float(args.wall_cap_seconds),
            )
        elif args.stage == "ladder":
            if args.canary_receipt is None or not args.canary_sha256:
                raise Phase6Error("ladder stage requires canary receipt and SHA-256")
            if args.ladder_receipt is not None or args.ladder_sha256 is not None:
                raise Phase6Error("ladder stage does not accept ladder binding arguments")
            payload = run_ladder(
                output=args.output,
                wall_cap_seconds=float(args.wall_cap_seconds),
                canary_receipt=args.canary_receipt,
                canary_sha256=str(args.canary_sha256),
            )
        else:
            if args.canary_receipt is not None or args.canary_sha256 is not None:
                raise Phase6Error("repair stage does not accept canary binding arguments")
            if args.ladder_receipt is None or not args.ladder_sha256:
                raise Phase6Error("repair stage requires ladder receipt and SHA-256")
            payload = run_h_confirmation_repair(
                output=args.output,
                wall_cap_seconds=float(args.wall_cap_seconds),
                ladder_receipt=args.ladder_receipt,
                ladder_sha256=str(args.ladder_sha256),
            )
    print(
        "JSON_SUMMARY "
        + json.dumps(
            {
                "decision": payload["decision"],
                "stage": payload["stage"],
                "wall_time_seconds": payload["run_manifest"]["wall_time_seconds"],
            },
            sort_keys=True,
        )
    )
    return 0 if payload["status"] == "PASSED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
