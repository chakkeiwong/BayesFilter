#!/usr/bin/env python3
"""Exact transformed-target preflight for the frozen SSL-LSTM G/H transports.

This runner is deliberately not an HMC harness.  It reloads the two immutable
trial-0 payloads, binds them to the locked target through the reviewed fixed
transport value/score adapter, and checks change-of-variables identities,
analytic scores, serialization, and original-start round trips.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import tensorflow as tf


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bayesfilter.inference.batched_value_score import FixedTransportValueScoreAdapter  # noqa: E402
from bayesfilter.inference.neutra_artifacts import (  # noqa: E402
    InvalidNeuTraArtifact,
    load_frozen_neutra_artifact,
)
from bayesfilter.inference.neutra_training import (  # noqa: E402
    NeuTraReverseKLTrainer,
    NeuTraTrainerConfig,
)
from bayesfilter.nonlinear.ssl_lstm_posterior_tf import (  # noqa: E402
    FREE_PARAMETER_NAMES,
    PRIOR_CENTER_VALUES,
    locked_ssl_lstm_posterior_target,
)


PLAN_PATH = Path(
    "docs/plans/bayesfilter-ssl-lstm-neutra-phase-5-exact-transformed-target-"
    "preflight-plan-2026-07-14.md"
)
SCRIPT_PATH = Path(__file__).resolve().relative_to(ROOT)
ARTIFACT_ROOT = Path(
    "docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/"
    "trial0-alternative-confirmation-2026-07-16"
)
OUTPUT_ROOT = Path(
    "docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/phase-5-trial0-gh"
)
RESULT_PATH = Path(
    "docs/plans/bayesfilter-ssl-lstm-neutra-phase-5-exact-transformed-target-"
    "preflight-result-2026-07-16.md"
)
TARGET_SIGNATURE = "549efdf2aa5d9534226cb29c3678489d92766f92e6140901355eac33618f719e"
TARGET_ADAPTER_SIGNATURE = "004f86b5668939febb629c563ca02625998c878d1e74d88c463f93b029a5d556"
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
BEST_STATES = {
    "fresh-g": (
        ARTIFACT_ROOT / "fresh-g/best-state.json",
        "54192a4f8eb67ecaf682324a055224c03be0bbf0a54ec0d4942b07d0b6e37abb",
    ),
    "fresh-h": (
        ARTIFACT_ROOT / "fresh-h/best-state.json",
        "ffb3a54889532022951b322964a43e286375c7367c4a43f075cbd1b4539dd71a",
    ),
}
TARGET_SEMANTIC_SHA256 = TARGET_SIGNATURE
A0_LOCK_PATH = Path(
    "docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a0/target-lock.json"
)
A0_LOCK_SHA256 = "1f7fccbeafbaa344a80e77c73b4356f44258b78a65ea2499e8ebd194b79a4383"
ORIGINAL_STARTS = (
    (0.0, 0.0, 0.0, 0.0),
    (0.5, -0.5, 0.5, -0.5),
    (-0.5, 0.5, -0.5, 0.5),
    (0.5, 0.5, -0.5, -0.5),
)
FD_STEP = 1.0e-5
VALUE_ATOL = 1.0e-10
PARITY_ATOL = 1.0e-10
SCORE_ATOL = 1.0e-6
SCORE_RTOL = 1.0e-5
ROUNDTRIP_ATOL = 1.0e-9


class PreflightError(RuntimeError):
    """Raised when an exact preflight gate fails."""


class TargetBatchBridge:
    """Expose the locked target through the reviewed scalar/batch boundary."""

    parameter_dim = 4

    def __init__(self, target: Any) -> None:
        self.target = target
        self.parameter_dim = int(target.parameter_dim)
        if tuple(target.parameter_names) != tuple(FREE_PARAMETER_NAMES):
            raise PreflightError("locked target parameter names drift")

    def adapter_signature(self) -> str:
        return self.target.adapter_signature()

    def target_signature(self) -> str:
        return self.target.target_signature()

    def target_scope(self) -> str:
        return self.target.target_scope

    def value_score_capability(self) -> Any:
        return self.target.value_score_capability()

    def log_prob_and_grad(self, values: Any) -> tuple[tf.Tensor, tf.Tensor]:
        tensor = tf.convert_to_tensor(values, dtype=tf.float64)
        if tensor.shape.rank == 1:
            return self.target.value_and_score(tensor)
        if tensor.shape.rank == 2:
            return self.target.batch_value_and_score(tensor)
        raise ValueError("locked target bridge requires rank 1 or 2")


def canonical(payload: Any) -> bytes:
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


def sha256(path: Path) -> str:
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(item) for item in value]
    if hasattr(value, "numpy"):
        return json_safe(value.numpy())
    if hasattr(value, "tolist"):
        return json_safe(value.tolist())
    if hasattr(value, "item"):
        return value.item()
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    absolute = ROOT / path
    absolute.parent.mkdir(parents=True, exist_ok=True)
    if absolute.exists():
        raise PreflightError(f"refusing to overwrite receipt: {path}")
    absolute.write_bytes(canonical(payload))


def load_target_and_bridge() -> tuple[Any, TargetBatchBridge]:
    target = locked_ssl_lstm_posterior_target()
    if target.target_signature() != TARGET_SIGNATURE:
        raise PreflightError("locked target semantic signature mismatch")
    if target.adapter_signature() != TARGET_ADAPTER_SIGNATURE:
        raise PreflightError("locked target adapter signature mismatch")
    return target, TargetBatchBridge(target)


def historical_geometry() -> tuple[tf.Tensor, tf.Tensor, str]:
    path = ROOT / A0_LOCK_PATH
    if hashlib.sha256(path.read_bytes()).hexdigest() != A0_LOCK_SHA256:
        raise PreflightError("A0 target-lock byte identity drift")
    lock = json.loads(path.read_text(encoding="utf-8"))
    geometry = lock["sampler_geometry"]
    center = tf.constant(geometry["center_free"]["values"], tf.float64)
    scale = tf.constant(geometry["scale"]["values"], tf.float64)
    factor_z = tf.constant(geometry["factor_z"]["values"], tf.float64)
    factor = tf.linalg.diag(scale) @ factor_z
    return center, factor, str(lock["signatures"]["sampler_geometry_sha256"])


def coordinate_shell(radius: float) -> tf.Tensor:
    rows: list[list[float]] = []
    for index in range(4):
        row = [0.0] * 4
        row[index] = radius
        rows.extend((row, [-value for value in row]))
    return tf.constant(rows, tf.float64)


def probe_bank() -> tuple[tf.Tensor, list[str], dict[str, Any]]:
    center, factor, geometry_sha256 = historical_geometry()
    shell2 = center + coordinate_shell(2.0) @ tf.transpose(factor)
    shell4 = center + coordinate_shell(4.0) @ tf.transpose(factor)
    original = center + tf.constant(ORIGINAL_STARTS, tf.float64) @ tf.transpose(factor)
    prior = tf.constant(PRIOR_CENTER_VALUES, tf.float64)[tf.newaxis, :]
    points = tf.concat((prior, shell2, shell4, original), axis=0)
    labels = ["prior_center"] + [f"shell_radius_2_{i}" for i in range(8)]
    labels += [f"shell_radius_4_{i}" for i in range(8)]
    labels += [f"original_start_{i}" for i in range(4)]
    metadata = {
        "point_count": len(labels),
        "prior_center": list(PRIOR_CENTER_VALUES),
        "shell_radii": [2.0, 4.0],
        "original_starts": [list(row) for row in ORIGINAL_STARTS],
        "sampler_geometry_sha256": geometry_sha256,
        "fd_step": FD_STEP,
    }
    return points, labels, metadata


def assert_no_mutable_training_state(transport: Any) -> dict[str, Any]:
    variables: list[str] = []
    state_surfaces: list[str] = []
    visited: set[int] = set()
    forbidden_names = frozenset(
        {"optimizer", "trainer", "trainable_variables", "variables"}
    )

    def walk(value: Any, path: str, depth: int) -> None:
        if depth > 5 or id(value) in visited:
            return
        visited.add(id(value))
        if isinstance(value, tf.Variable):
            variables.append(path)
            return
        if isinstance(value, Mapping):
            for key, item in value.items():
                walk(item, f"{path}[{key!r}]", depth + 1)
        elif isinstance(value, (tuple, list)):
            for index, item in enumerate(value):
                walk(item, f"{path}[{index}]", depth + 1)
        elif hasattr(value, "__dict__"):
            for key, item in vars(value).items():
                if key.lower() in forbidden_names and item not in (None, (), []):
                    state_surfaces.append(f"{path}.{key}")
                walk(item, f"{path}.{key}", depth + 1)

    walk(transport, "transport", 0)
    if variables or state_surfaces:
        details = variables + state_surfaces
        raise PreflightError("mutable training state reachable: " + ", ".join(details))
    return {
        "mutable_tf_variables": variables,
        "optimizer_trainer_surfaces": state_surfaces,
        "passed": True,
    }


def make_adapter(target_bridge: TargetBatchBridge, transport: Any) -> FixedTransportValueScoreAdapter:
    return FixedTransportValueScoreAdapter(
        base_adapter=target_bridge,
        transport=transport,
        target_scope="ssl_lstm_completion:a1:masked_svd_ukf_four_parameter:phase5_fixed_transport",
        require_batch_native=True,
    )


def load_transport(label: str) -> tuple[Any, dict[str, Any]]:
    path, expected_hash = PAYLOADS[label]
    payload_path = ROOT / path
    if hashlib.sha256(payload_path.read_bytes()).hexdigest() != expected_hash:
        raise PreflightError(f"{label} payload byte hash mismatch")
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    artifact = load_frozen_neutra_artifact(
        payload,
        expected_target_signature=TARGET_SIGNATURE,
    )
    if artifact.manifest.target_signature != TARGET_SIGNATURE:
        raise PreflightError(f"{label} manifest target signature mismatch")
    if artifact.manifest.tensor_hash != payload.get("tensor_hash"):
        raise PreflightError(f"{label} tensor hash mismatch after reload")
    return artifact.transport, {
        "payload_path": path.as_posix(),
        "payload_sha256": expected_hash,
        "transport_hash": artifact.manifest.transport_hash,
        "tensor_hash": artifact.manifest.tensor_hash,
        "topology_hash": artifact.manifest.topology_hash,
        "artifact_signature": artifact.artifact_signature,
    }


def trainer_config_from_state(state: Mapping[str, Any]) -> NeuTraTrainerConfig:
    payload = dict(state["config"])
    if payload.pop("schema", None) != "bayesfilter.neutra.trainer_config.v1":
        raise PreflightError("preserved trainer config schema mismatch")
    for key in (
        "hidden_layers",
        "initialization_seed",
        "fixed_translation",
        "target_parameter_names",
    ):
        payload[key] = tuple(payload[key])
    return NeuTraTrainerConfig(**payload)


def serialization_parity(
    label: str,
    target: Any,
    frozen: Any,
    z: tf.Tensor,
    persisted_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    path, expected_hash = BEST_STATES[label]
    absolute = ROOT / path
    if hashlib.sha256(absolute.read_bytes()).hexdigest() != expected_hash:
        raise PreflightError(f"{label} best-state byte hash mismatch")
    state = json.loads(absolute.read_text(encoding="utf-8"))
    trainer = NeuTraReverseKLTrainer(target, trainer_config_from_state(state))
    trainer.restore_state(state)
    trainable_theta, trainable_logdet = trainer.forward_and_logdet(z)
    frozen_theta = frozen.forward_z_to_theta_batch(z)
    frozen_logdet = frozen.log_abs_det_jacobian_batch(z)
    forward_residual = _max_abs(trainable_theta - frozen_theta)
    logdet_residual = _max_abs(trainable_logdet - frozen_logdet)
    payload_path, _ = PAYLOADS[label]
    persisted = json.loads((ROOT / payload_path).read_text(encoding="utf-8"))
    replayed = trainer.frozen_transport_payload(
        transport_id=str(persisted["transport_id"]),
        target_signature=TARGET_SIGNATURE,
    )
    hashes_match = all(
        replayed[key] == persisted[key]
        for key in ("topology_hash", "tensor_hash", "transport_hash")
    )
    training_state_hash_match = bool(
        replayed["training_state_hash"] == state["state_hash"]
        and replayed["training_state_hash"] == persisted["training_state_hash"]
    )
    passed = bool(
        forward_residual <= VALUE_ATOL
        and logdet_residual <= VALUE_ATOL
        and hashes_match
        and training_state_hash_match
        and replayed["transport_hash"] == persisted_manifest["transport_hash"]
    )
    if not passed:
        raise PreflightError(f"{label} pre-freeze/reload serialization parity failed")
    return {
        "passed": True,
        "best_state_path": path.as_posix(),
        "best_state_sha256": expected_hash,
        "forward_max_abs": forward_residual,
        "logdet_max_abs": logdet_residual,
        "topology_tensor_transport_hashes_match": hashes_match,
        "training_state_hash_match": training_state_hash_match,
        "trainer_used_for_diagnostic_only": True,
        "trainer_reachable_from_fixed_binding": False,
    }


def loader_negative_controls(label: str) -> dict[str, Any]:
    path, _expected_hash = PAYLOADS[label]
    payload = json.loads((ROOT / path).read_text(encoding="utf-8"))
    corrupted = copy.deepcopy(payload)
    dense = next(
        component
        for component in corrupted["components"]
        if component["kind"] == "dense_autoregressive_iaf"
    )
    dense["weights"][0][0][0] = float(dense["weights"][0][0][0]) + 1.0e-6
    corruption_rejected = False
    try:
        load_frozen_neutra_artifact(
            corrupted,
            expected_target_signature=TARGET_SIGNATURE,
        )
    except InvalidNeuTraArtifact:
        corruption_rejected = True
    target_mismatch_rejected = False
    try:
        load_frozen_neutra_artifact(
            payload,
            expected_target_signature="0" * 64,
        )
    except InvalidNeuTraArtifact:
        target_mismatch_rejected = True
    if not corruption_rejected or not target_mismatch_rejected:
        raise PreflightError(f"{label} loader negative control failed")
    return {
        "corrupted_tensor_rejected": corruption_rejected,
        "target_mismatch_rejected": target_mismatch_rejected,
        "passed": True,
    }


def mathematical_negative_controls(
    transport: Any,
    target_value: tf.Tensor,
    target_score: tf.Tensor,
    z: tf.Tensor,
    finite_difference: tf.Tensor,
) -> dict[str, Any]:
    logdet = transport.log_abs_det_jacobian_batch(z)
    logdet_score = transport.log_abs_det_jacobian_score_batch(z)
    pullback = transport.pullback_score_batch(z, target_score)
    wrong_sign_value = target_value - logdet
    correct_value = target_value + logdet
    wrong_sign_identity_residual = _max_abs(wrong_sign_value - correct_value)
    omitted_score = pullback
    omitted_error = tf.abs(omitted_score - finite_difference)
    omitted_tolerance = SCORE_ATOL + SCORE_RTOL * tf.abs(finite_difference)
    omitted_score_rejected = not bool(
        tf.reduce_all(omitted_error <= omitted_tolerance).numpy()
    )
    correct_score = pullback + logdet_score
    wrong_logdet_score = pullback - logdet_score
    wrong_score_rejected = not bool(
        tf.reduce_all(
            tf.abs(wrong_logdet_score - finite_difference) <= omitted_tolerance
        ).numpy()
    )
    if (
        wrong_sign_identity_residual <= VALUE_ATOL
        or not omitted_score_rejected
        or not wrong_score_rejected
    ):
        raise PreflightError("mathematical negative controls did not discriminate")
    return {
        "wrong_logdet_sign_identity_residual": wrong_sign_identity_residual,
        "wrong_logdet_sign_rejected": True,
        "omitted_logdet_score_max_abs": _max_abs(omitted_error),
        "omitted_logdet_score_rejected": omitted_score_rejected,
        "wrong_logdet_score_max_abs": _max_abs(
            wrong_logdet_score - finite_difference
        ),
        "wrong_logdet_score_rejected": wrong_score_rejected,
        "passed": True,
    }


def _max_abs(tensor: tf.Tensor) -> float:
    return float(tf.reduce_max(tf.abs(tensor)).numpy())


def run_candidate(
    label: str,
    target: Any,
    bridge: TargetBatchBridge,
    *,
    jit_compile: bool,
) -> dict[str, Any]:
    transport, manifest = load_transport(label)
    mutable = assert_no_mutable_training_state(transport)
    adapter = make_adapter(bridge, transport)
    points, labels, metadata = probe_bank()
    z = transport.inverse_theta_to_z_batch(points)

    @tf.function(
        input_signature=[tf.TensorSpec([21, 4], tf.float64)],
        jit_compile=jit_compile,
        reduce_retracing=True,
    )
    def transformed_program(values: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        return adapter.log_prob_and_grad_batch(values)

    transformed_value, transformed_score = transformed_program(z)
    hlo_sha256 = None
    if jit_compile:
        hlo = transformed_program.experimental_get_compiler_ir(z)(stage="hlo")
        hlo_sha256 = hashlib.sha256(str(hlo).encode("utf-8")).hexdigest()
    replay = transport.forward_z_to_theta_batch(z)
    roundtrip_residual = _max_abs(replay - points)
    original_start_roundtrip_residual = _max_abs(replay[17:21] - points[17:21])
    if not math.isfinite(roundtrip_residual) or roundtrip_residual > ROUNDTRIP_ATOL:
        raise PreflightError(f"{label} original/probe roundtrip gate failed")

    target_value, target_score = bridge.log_prob_and_grad(points)
    logdet = transport.log_abs_det_jacobian_batch(z)
    value_identity_residual = _max_abs(transformed_value - target_value - logdet)
    expected_score = transport.pullback_score_batch(z, target_score)
    expected_score += transport.log_abs_det_jacobian_score_batch(z)
    score_formula_residual = _max_abs(transformed_score - expected_score)

    scalar_values = []
    scalar_scores = []
    for row in z:
        value, score = adapter.log_prob_and_grad(row)
        scalar_values.append(value)
        scalar_scores.append(score)
    scalar_values_tensor = tf.stack(scalar_values)
    scalar_scores_tensor = tf.stack(scalar_scores)
    scalar_value_residual = _max_abs(transformed_value - scalar_values_tensor)
    scalar_score_residual = _max_abs(transformed_score - scalar_scores_tensor)

    permutation = tf.constant([20, 0, 11, 4, 17, 3, 9, 1, 15, 6, 8, 13, 2, 19, 5, 10, 7, 12, 14, 16, 18], tf.int32)
    perm_z = tf.gather(z, permutation)
    perm_values, perm_scores = transformed_program(perm_z)
    permutation_value_residual = _max_abs(perm_values - tf.gather(transformed_value, permutation))
    permutation_score_residual = _max_abs(perm_scores - tf.gather(transformed_score, permutation))

    plus = []
    minus = []
    for coordinate in range(4):
        direction = tf.one_hot(coordinate, 4, dtype=tf.float64)[tf.newaxis, :]
        plus.append(z + FD_STEP * direction)
        minus.append(z - FD_STEP * direction)
    plus_values = [transformed_program(value)[0] for value in plus]
    minus_values = [transformed_program(value)[0] for value in minus]
    finite_difference = tf.stack(
        [(up - down) / (2.0 * FD_STEP) for up, down in zip(plus_values, minus_values)],
        axis=1,
    )
    fd_error = tf.abs(transformed_score - finite_difference)
    fd_tolerance = SCORE_ATOL + SCORE_RTOL * tf.abs(finite_difference)
    finite_difference_passed = bool(tf.reduce_all(fd_error <= fd_tolerance).numpy())
    serialization = serialization_parity(
        label,
        target,
        transport,
        z,
        manifest,
    )
    loader_controls = loader_negative_controls(label)
    math_controls = mathematical_negative_controls(
        transport,
        target_value,
        target_score,
        z,
        finite_difference,
    )
    finite = all(
        bool(tf.reduce_all(tf.math.is_finite(value)).numpy())
        for value in (
            points,
            z,
            replay,
            target_value,
            target_score,
            logdet,
            transformed_value,
            transformed_score,
            finite_difference,
        )
    )
    passed = bool(
        finite
        and value_identity_residual <= VALUE_ATOL
        and score_formula_residual <= SCORE_ATOL
        and scalar_value_residual <= VALUE_ATOL
        and scalar_score_residual <= PARITY_ATOL
        and permutation_value_residual <= VALUE_ATOL
        and permutation_score_residual <= PARITY_ATOL
        and finite_difference_passed
    )
    if not passed:
        raise PreflightError(f"{label} exact transformed-target gate failed")
    output_devices = sorted(
        {
            str(transformed_value.device),
            str(transformed_score.device),
        }
    )
    if jit_compile and (
        not output_devices or not all("GPU:" in device for device in output_devices)
    ):
        raise PreflightError(f"{label} XLA outputs were not placed on GPU")
    return {
        "label": label,
        "status": "PASSED",
        "manifest": manifest,
        "mutable_state": mutable,
        "serialization_parity": serialization,
        "negative_controls": {
            "loader": loader_controls,
            "mathematical": math_controls,
        },
        "probe_metadata": metadata,
        "probe_labels": labels,
        "roundtrip_max_abs": roundtrip_residual,
        "original_start_roundtrip_max_abs": original_start_roundtrip_residual,
        "original_start_inverse_radii": [
            float(tf.linalg.norm(z[index]).numpy()) for index in range(17, 21)
        ],
        "value_identity_max_abs": value_identity_residual,
        "score_formula_max_abs": score_formula_residual,
        "scalar_batch_value_max_abs": scalar_value_residual,
        "scalar_batch_score_max_abs": scalar_score_residual,
        "permutation_value_max_abs": permutation_value_residual,
        "permutation_score_max_abs": permutation_score_residual,
        "finite_difference_max_abs": _max_abs(fd_error),
        "finite_difference_passed": finite_difference_passed,
        "finite": finite,
        "thresholds": {
            "roundtrip_atol": ROUNDTRIP_ATOL,
            "value_atol": VALUE_ATOL,
            "parity_atol": PARITY_ATOL,
            "score_atol": SCORE_ATOL,
            "score_rtol": SCORE_RTOL,
        },
        "adapter_signature": adapter.adapter_signature(),
        "target_signature": bridge.target_signature(),
        "compiled_transformed_program": {
            "jit_compile": jit_compile,
            "trace_count": int(transformed_program.experimental_get_tracing_count()),
            "hlo_sha256": hlo_sha256,
            "output_devices": output_devices,
        },
    }


def run(mode: str, output: Path, wall_cap_seconds: float) -> dict[str, Any]:
    started = time.perf_counter()
    if mode == "gpu-xla":
        gpus = tf.config.list_physical_devices("GPU")
        if not gpus:
            raise PreflightError("GPU/XLA mode requires a visible GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.experimental.enable_tensor_float_32_execution(True)
        device = "/GPU:0"
    else:
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            raise PreflightError(
                "cpu-reference mode requires CUDA_VISIBLE_DEVICES=-1 before import"
            )
        device = "/CPU:0"
    target, bridge = load_target_and_bridge()
    rows = []
    with tf.device(device):
        for label in PAYLOADS:
            if time.perf_counter() - started > wall_cap_seconds:
                raise PreflightError("Phase 5 wall cap exhausted")
            rows.append(
                run_candidate(
                    label,
                    target,
                    bridge,
                    jit_compile=mode == "gpu-xla",
                )
            )
    elapsed = time.perf_counter() - started
    payload = {
        "schema": "bayesfilter.ssl_lstm_neutra.phase5_exact_preflight.v1",
        "status": "PASSED",
        "decision": "PHASE5_EXACT_TRANSFORMED_TARGET_PASSED",
        "mode": mode,
        "no_hmc": True,
        "rows": rows,
        "source_bindings": {
            "plan": {"path": PLAN_PATH.as_posix(), "sha256": sha256(PLAN_PATH)},
            "runner": {"path": SCRIPT_PATH.as_posix(), "sha256": sha256(SCRIPT_PATH)},
            "target": {"path": "bayesfilter/nonlinear/ssl_lstm_posterior_tf.py", "sha256": sha256(Path("bayesfilter/nonlinear/ssl_lstm_posterior_tf.py"))},
            "adapter": {"path": "bayesfilter/inference/batched_value_score.py", "sha256": sha256(Path("bayesfilter/inference/batched_value_score.py"))},
            "artifact_loader": {"path": "bayesfilter/inference/neutra_artifacts.py", "sha256": sha256(Path("bayesfilter/inference/neutra_artifacts.py"))},
        },
        "run_manifest": {
            "command": " ".join(sys.argv),
            "git_commit": subprocess.check_output(
                ("git", "rev-parse", "HEAD"), cwd=ROOT, text=True
            ).strip(),
            "git_dirty": bool(
                subprocess.check_output(
                    ("git", "status", "--porcelain=v1", "--untracked-files=all"),
                    cwd=ROOT,
                    text=True,
                ).strip()
            ),
            "environment": "tfgpu",
            "python": platform.python_version(),
            "tensorflow": tf.__version__,
            "mode": mode,
            "wall_time_seconds": elapsed,
            "wall_cap_seconds": wall_cap_seconds,
            "dtype": "float64",
            "jit_compile": mode == "gpu-xla",
            "tf32": bool(tf.config.experimental.tensor_float_32_execution_enabled()),
            "visible_physical_gpus": [gpu.name for gpu in gpus],
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "data_version": "A0 target lock " + A0_LOCK_SHA256,
            "random_seeds": "N/A deterministic fixed probe bank",
            "output_path": output.as_posix(),
            "plan_path": PLAN_PATH.as_posix(),
            "result_path": RESULT_PATH.as_posix(),
            "trust_basis": "owner_designated_managed_session_visible_gpu_trusted" if mode == "gpu-xla" else "cpu_hidden_reference_only",
        },
        "nonclaims": [
            "no HMC transition, tuning, or retained sampling",
            "no posterior correctness, support completeness, predictive validity, superiority, or default readiness",
        ],
    }
    write_json(output, payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("cpu-reference", "gpu-xla"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wall-cap-seconds", type=float, default=1800.0)
    args = parser.parse_args(argv)
    payload = run(args.mode, args.output, float(args.wall_cap_seconds))
    print("JSON_SUMMARY " + json.dumps({"decision": payload["decision"], "mode": payload["mode"], "wall_time_seconds": payload["run_manifest"]["wall_time_seconds"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
