#!/usr/bin/env python3
"""Immutable retained-chain admission for the frozen SSL-LSTM G/H charts."""

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
    RetainedSampleHMCArchiveConfig,
    build_retained_sample_hmc_archive_runner,
)
from bayesfilter.inference.hmc_posterior_diagnostics import (  # noqa: E402
    compute_coordinate_diagnostics,
    posterior_mean_diagnostics,
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
    "docs/plans/bayesfilter-ssl-lstm-neutra-phase-7-retained-admission-"
    "plan-2026-07-16.md"
)
RESULT_PATH = Path(
    "docs/plans/bayesfilter-ssl-lstm-neutra-phase-7-retained-admission-"
    "result-2026-07-17.md"
)
SCRIPT_PATH = Path(__file__).resolve().relative_to(ROOT)
ARTIFACT_ROOT = Path(
    "docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/"
    "trial0-alternative-confirmation-2026-07-16"
)
PHASE7_ROOT = Path(
    "docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/"
    "phase-7-retained-admission"
)
PHASE5_RECEIPT_PATH = Path(
    "docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/phase-5-trial0-gh/"
    "gpu-xla-r2.json"
)
PHASE5_RECEIPT_SHA256 = (
    "f855fd3fc83260867582a79b024087efbc5ecd463321ddd98f5fae7c9056f55b"
)
PHASE6_RECEIPT_PATH = Path(
    "docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/phase-6-trial0-gh/"
    "h-confirmation-repair.json"
)
PHASE6_RECEIPT_SHA256 = (
    "dc340ab2570032a85062d0ec9cd8c9e020c41a133ec9d11b78982502ff08b9b2"
)
CANARY_RECEIPT_PATH = PHASE7_ROOT / "timing-canary.json"
CANARY_RECEIPT_SHA256 = (
    "647be960a5307d564d1777d9cee5488262f3345ac0fd46ae0a5aea05367841ef"
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
DIAGNOSTICS_SOURCE = Path("bayesfilter/inference/hmc_posterior_diagnostics.py")
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
TRANSPORT_HASHES = {
    "fresh-g": "5e485163a01f7f2a02d511fd40fa8d16f8249d528940a453df6386e1d68505aa",
    "fresh-h": "afa52cc59fba6e566649b085ae0367e3d91eb5a1cfd30fd9b7a5a15fcf4fd44a",
}
ORIGINAL_STARTS = (
    (0.0, 0.0, 0.0, 0.0),
    (0.5, -0.5, 0.5, -0.5),
    (-0.5, 0.5, -0.5, 0.5),
    (0.5, 0.5, -0.5, -0.5),
)
TARGET_SCOPE = (
    "ssl_lstm_completion:a1:masked_svd_ukf_four_parameter:"
    "phase7_fixed_transport_retained_admission"
)
STEP_SIZE = 0.8
NUM_LEAPFROG_STEPS = 4
CANARY_RESULTS = 4
CANARY_BURNIN = 2
CANARY_SEEDS = {
    "fresh-g": ((7101, 7102), (7111, 7112), (7121, 7122)),
    "fresh-h": ((7201, 7202), (7211, 7212), (7221, 7222)),
}
ACQUISITION_SEED_BASES = {"fresh-g": 8101, "fresh-h": 9101}
PHASE6_MAX_SEED_COMPONENT = 6901
R_HAT_MAX = 1.05
ESS_MIN = 100.0
MCSE_SD_RATIO_MAX = 0.10
ACCEPTANCE_BAND = (0.55, 0.85)
VALUE_MATCH_ABS = 1.0e-10
CROSS_REPLICATION_Z_MAX = 3.0
SECOND_MOMENT_INDICES = tuple(
    (left, right) for left in range(4) for right in range(left, 4)
)

ACQUISITION_SEGMENT_RESULTS = 256
ACQUISITION_BURNIN = 128
ACQUISITION_CHECKPOINT_SEGMENTS = (1, 2, 4, 8)
ACQUISITION_CHART_WALL_CAP_SECONDS = 1050.0
ACQUISITION_WALL_CAP_SECONDS = 2100.0


class Phase7Error(RuntimeError):
    """Raised when a Phase 7 binding, execution, or evidence invariant fails."""


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _absolute(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _sha256(path: Path) -> str:
    return hashlib.sha256(_absolute(path).read_bytes()).hexdigest()


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
    absolute = _absolute(path)
    absolute.parent.mkdir(parents=True, exist_ok=True)
    if absolute.exists():
        raise Phase7Error(f"refusing to overwrite receipt: {path}")
    absolute.write_bytes(_canonical(_json_safe(payload)))


def _strict_json(path: Path) -> dict[str, Any]:
    def pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in rows:
            if key in result:
                raise Phase7Error(f"duplicate JSON key {key!r}: {path}")
            result[key] = value
        return result

    def reject(value: str) -> None:
        raise Phase7Error(f"nonfinite JSON constant {value!r}: {path}")

    value = json.loads(
        _absolute(path).read_text(encoding="utf-8"),
        object_pairs_hook=pairs,
        parse_constant=reject,
    )
    if not isinstance(value, dict):
        raise Phase7Error(f"expected JSON object: {path}")
    return value


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
            "posterior_diagnostics": DIAGNOSTICS_SOURCE,
        }.items()
    }


def validate_upstream_receipts() -> dict[str, Any]:
    if _sha256(PHASE5_RECEIPT_PATH) != PHASE5_RECEIPT_SHA256:
        raise Phase7Error("Phase 5 receipt byte identity drift")
    phase5 = _strict_json(PHASE5_RECEIPT_PATH)
    if phase5.get("status") != "PASSED" or phase5.get("decision") != (
        "PHASE5_EXACT_TRANSFORMED_TARGET_PASSED"
    ):
        raise Phase7Error("Phase 5 passing decision drift")
    if _sha256(PHASE6_RECEIPT_PATH) != PHASE6_RECEIPT_SHA256:
        raise Phase7Error("Phase 6 receipt byte identity drift")
    phase6 = _strict_json(PHASE6_RECEIPT_PATH)
    if phase6.get("status") != "PASSED" or phase6.get("decision") != (
        "PHASE6_IDENTITY_MASS_KERNELS_FROZEN_AFTER_H_REPAIR"
    ):
        raise Phase7Error("Phase 6 frozen-kernel decision drift")
    for label, (path, payload_sha256) in PAYLOADS.items():
        if _sha256(path) != payload_sha256:
            raise Phase7Error(f"payload byte identity drift: {label}")
        kernel = phase6.get("selected_kernels", {}).get(label)
        if not isinstance(kernel, Mapping):
            raise Phase7Error(f"Phase 6 selected kernel missing: {label}")
        expected = {
            "mass_matrix": "identity",
            "step_size": STEP_SIZE,
            "num_leapfrog_steps": NUM_LEAPFROG_STEPS,
            "target_signature": TARGET_SIGNATURE,
            "transport_hash": TRANSPORT_HASHES[label],
        }
        for field, value in expected.items():
            if kernel.get(field) != value:
                raise Phase7Error(f"Phase 6 kernel drift: {label}.{field}")
    for key, path in {
        "target": TARGET_SOURCE,
        "adapter": ADAPTER_SOURCE,
        "artifact_loader": LOADER_SOURCE,
        "hmc_runtime": HMC_SOURCE,
    }.items():
        if phase6.get("source_bindings", {}).get(key, {}).get("sha256") != _sha256(path):
            raise Phase7Error(f"source binding drift since Phase 6: {path}")
    return {
        "phase5": {
            "path": PHASE5_RECEIPT_PATH.as_posix(),
            "sha256": PHASE5_RECEIPT_SHA256,
            "decision": phase5["decision"],
        },
        "phase6": {
            "path": PHASE6_RECEIPT_PATH.as_posix(),
            "sha256": PHASE6_RECEIPT_SHA256,
            "decision": phase6["decision"],
        },
        "kernel": {
            "mass_matrix": "identity",
            "step_size": STEP_SIZE,
            "num_leapfrog_steps": NUM_LEAPFROG_STEPS,
            "trajectory_length": STEP_SIZE * NUM_LEAPFROG_STEPS,
        },
    }


def validate_canary_receipt() -> dict[str, Any]:
    if _sha256(CANARY_RECEIPT_PATH) != CANARY_RECEIPT_SHA256:
        raise Phase7Error("Stage A canary receipt byte identity drift")
    receipt = _strict_json(CANARY_RECEIPT_PATH)
    if receipt.get("status") != "PASSED" or receipt.get("decision") != (
        "PHASE7_STAGE_A_TIMING_CANARY_PASSED_BUDGET_FREEZE_REQUIRED"
    ):
        raise Phase7Error("Stage A passing decision drift")
    if receipt.get("hard_vetoes") != []:
        raise Phase7Error("Stage A receipt contains a hard veto")
    if receipt.get("contract", {}).get("samples_excluded_from_retained_evidence") is not True:
        raise Phase7Error("Stage A canary/exclusion boundary drift")
    rows = receipt.get("charts", {})
    if set(rows) != set(PAYLOADS):
        raise Phase7Error("Stage A chart set drift")
    for label in PAYLOADS:
        row = rows[label]
        if row.get("excluded_from_retained_evidence") is not True:
            raise Phase7Error(f"Stage A sample exclusion drift: {label}")
        if row.get("cumulative_chain_moved") != [True, True, True, True]:
            raise Phase7Error(f"Stage A movement gate drift: {label}")
        segments = row.get("segments", ())
        if len(segments) != 3 or any(segment.get("passed") is not True for segment in segments):
            raise Phase7Error(f"Stage A segment gate drift: {label}")
    return {
        "path": CANARY_RECEIPT_PATH.as_posix(),
        "sha256": CANARY_RECEIPT_SHA256,
        "decision": receipt["decision"],
        "wall_time_seconds": receipt["run_manifest"]["wall_time_seconds"],
        "samples_excluded_from_retained_evidence": True,
    }


class TargetBatchBridge:
    """Receipt-bound Phase 7 value/score bridge for the locked target."""

    parameter_dim = 4

    def __init__(self, target: Any) -> None:
        self.target = target
        if int(target.parameter_dim) != 4:
            raise Phase7Error("locked target parameter dimension drift")
        if tuple(target.parameter_names) != tuple(FREE_PARAMETER_NAMES):
            raise Phase7Error("locked target parameter order drift")

    def adapter_signature(self) -> str:
        return self.target.adapter_signature()

    def target_signature(self) -> str:
        return self.target.target_signature()

    def value_score_capability(self) -> ValueScoreCapability:
        base = self.target.value_score_capability()
        if base.value_score_authority != "graph_native":
            raise Phase7Error("locked target is not graph-native")
        return ValueScoreCapability(
            value_score_authority=base.value_score_authority,
            xla_hmc_ready=True,
            full_chain_xla_diagnostic_ready=True,
            runtime_backend="phase7_receipt_bound_ssl_lstm_target_bridge",
            evidence_path=PHASE6_RECEIPT_PATH.as_posix(),
            target_scope=TARGET_SCOPE,
            nonclaims=(
                "Phase 7 exact fixed-transport retained HMC only",
                "authority is bound to passing Phase 5 and Phase 6 receipts",
                "does not mutate the locked target capability",
                "no posterior correctness or stationarity claim",
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
    if label not in PAYLOADS:
        raise Phase7Error(f"unknown chart label: {label}")
    target = locked_ssl_lstm_posterior_target()
    if target.target_signature() != TARGET_SIGNATURE:
        raise Phase7Error("live target semantic signature drift")
    if target.adapter_signature() != TARGET_ADAPTER_SIGNATURE:
        raise Phase7Error("live target adapter signature drift")
    path, expected_hash = PAYLOADS[label]
    if _sha256(path) != expected_hash:
        raise Phase7Error(f"payload byte hash drift: {label}")
    artifact = load_frozen_neutra_artifact(
        _strict_json(path),
        expected_target_signature=TARGET_SIGNATURE,
    )
    if artifact.manifest.transport_hash != TRANSPORT_HASHES[label]:
        raise Phase7Error(f"transport hash drift: {label}")
    bridge = TargetBatchBridge(target)
    adapter = FixedTransportValueScoreAdapter(
        base_adapter=bridge,
        transport=artifact.transport,
        target_scope=TARGET_SCOPE,
        runtime_backend="phase7_fixed_dense_iaf_retained_hmc",
        evidence_path=PHASE6_RECEIPT_PATH.as_posix(),
        xla_hmc_ready=True,
        full_chain_xla_diagnostic_ready=True,
        require_batch_native=True,
        nonclaims=(
            "immutable fixed transport under Phase 7 retained-HMC authority",
            "identity mass and frozen Phase 6 kernel only",
            "no posterior truth, stationarity, or predictive claim",
        ),
    )
    if not adapter.value_score_capability().is_accepted_full_chain_xla_diagnostic_authority:
        raise Phase7Error("Phase 7 scoped full-chain XLA authority did not bind")
    if _sha256(A0_LOCK_PATH) != A0_LOCK_SHA256:
        raise Phase7Error("A0 target-lock byte identity drift")
    lock = _strict_json(A0_LOCK_PATH)
    geometry = lock.get("sampler_geometry")
    if not isinstance(geometry, Mapping):
        raise Phase7Error("A0 sampler geometry missing")
    center = tf.constant(geometry["center_free"]["values"], tf.float64)
    scale = tf.constant(geometry["scale"]["values"], tf.float64)
    factor_z = tf.constant(geometry["factor_z"]["values"], tf.float64)
    factor = tf.linalg.diag(scale) @ factor_z
    original_theta = center + tf.constant(ORIGINAL_STARTS, tf.float64) @ tf.transpose(factor)
    initial_z = artifact.transport.inverse_theta_to_z_batch(original_theta)
    replay = artifact.transport.forward_z_to_theta_batch(initial_z)
    roundtrip = float(tf.reduce_max(tf.abs(replay - original_theta)).numpy())
    if roundtrip > 1.0e-9 or not bool(tf.reduce_all(tf.math.is_finite(initial_z)).numpy()):
        raise Phase7Error(f"original-start inverse mapping failed: {label}")
    pairwise = tf.linalg.norm(
        initial_z[:, tf.newaxis, :] - initial_z[tf.newaxis, :, :], axis=-1
    )
    off_diagonal = tf.boolean_mask(pairwise, tf.logical_not(tf.eye(4, dtype=tf.bool)))
    if not bool(tf.reduce_all(off_diagonal > 0.0).numpy()):
        raise Phase7Error(f"inverse-mapped starts are not distinct: {label}")
    return adapter, initial_z, {
        "label": label,
        "payload_sha256": expected_hash,
        "transport_hash": artifact.manifest.transport_hash,
        "tensor_hash": artifact.manifest.tensor_hash,
        "topology_hash": artifact.manifest.topology_hash,
        "artifact_signature": artifact.artifact_signature,
        "target_signature": TARGET_SIGNATURE,
        "a0_target_lock_sha256": A0_LOCK_SHA256,
        "a0_sampler_geometry_sha256": lock["signatures"]["sampler_geometry_sha256"],
        "original_start_roundtrip_max_abs": roundtrip,
        "four_distinct_starts": True,
        "initial_z_radii": _json_safe(tf.linalg.norm(initial_z, axis=1)),
        "scoped_adapter_signature": adapter.adapter_signature(),
        "scoped_target_scope": TARGET_SCOPE,
    }


def canary_seed_ledger() -> dict[str, tuple[tuple[int, int], ...]]:
    validate_seed_ledger(CANARY_SEEDS, disjoint_below=PHASE6_MAX_SEED_COMPONENT)
    return CANARY_SEEDS


def acquisition_seeds(label: str, segment_count: int) -> tuple[tuple[int, int], ...]:
    if label not in ACQUISITION_SEED_BASES:
        raise Phase7Error(f"unknown chart label: {label}")
    count = int(segment_count)
    if count <= 0:
        raise Phase7Error("segment_count must be positive")
    base = ACQUISITION_SEED_BASES[label]
    return tuple((base + 10 * index, base + 1 + 10 * index) for index in range(count))


def validate_seed_ledger(
    ledger: Mapping[str, Sequence[Sequence[int]]], *, disjoint_below: int
) -> None:
    pairs: list[tuple[int, int]] = []
    words: list[int] = []
    for rows in ledger.values():
        for row in rows:
            pair = tuple(int(item) for item in row)
            if len(pair) != 2:
                raise Phase7Error("each stateless seed must contain two integers")
            pairs.append(pair)
            words.extend(pair)
    if len(pairs) != len(set(pairs)) or len(words) != len(set(words)):
        raise Phase7Error("seed ledger contains reused pairs or components")
    if any(value <= int(disjoint_below) for value in words):
        raise Phase7Error("Phase 7 seed is not disjoint from Phase 6 seed range")


def _parse_tensor(path: Path, dtype: tf.DType = tf.float64) -> tf.Tensor:
    return tf.io.parse_tensor(_absolute(path).read_bytes(), out_type=dtype)


def _read_private_archive(archive_dir: Path, label: str) -> dict[str, Any]:
    manifest_path = archive_dir / f"{label}_private_manifest.json"
    manifest = _strict_json(manifest_path)
    if manifest.get("artifact_type") != "bayesfilter_private_retained_sample_hmc_archive":
        raise Phase7Error("unexpected retained archive manifest type")
    shards = manifest.get("sample_shards")
    if not isinstance(shards, list) or len(shards) != 1:
        raise Phase7Error("retained archive must contain exactly one sample shard")
    shard = shards[0]
    sidecars = manifest.get("sidecars", {})
    state = sidecars.get("final_state")
    target = sidecars.get("final_target_log_prob")
    if not isinstance(state, Mapping) or not isinstance(target, Mapping):
        raise Phase7Error("retained archive sidecars are incomplete")
    for item, role in ((shard, "sample"), (state, "final state"), (target, "final target")):
        path = Path(str(item["path"]))
        if _sha256(path) != item["sha256"]:
            raise Phase7Error(f"retained {role} hash mismatch")
    samples = _parse_tensor(Path(str(shard["path"])))
    final_state = _parse_tensor(Path(str(state["path"])))
    final_target = _parse_tensor(Path(str(target["path"])))
    if tuple(samples.shape[1:]) != (4, 4) or tuple(final_state.shape) != (4, 4):
        raise Phase7Error("retained archive state/sample shape mismatch")
    if tuple(final_target.shape) != (4,):
        raise Phase7Error("retained archive final target shape mismatch")
    return {
        "samples": samples,
        "final_state": final_state,
        "final_target_log_prob": final_target,
        "manifest": manifest,
        "manifest_sha256": _sha256(manifest_path),
        "manifest_path": manifest_path,
        "sample_sha256": str(shard["sha256"]),
        "final_state_sha256": str(state["sha256"]),
        "final_target_log_prob_sha256": str(target["sha256"]),
    }


def _build_post_archive_auditor(adapter: Any, num_results: int) -> Any:
    results = int(num_results)
    if results <= 0:
        raise Phase7Error("post-archive audit num_results must be positive")

    @tf.function(
        input_signature=(
            tf.TensorSpec(shape=(results, 4, 4), dtype=tf.float64),
            tf.TensorSpec(shape=(4,), dtype=tf.float64),
        ),
        jit_compile=True,
        reduce_retracing=True,
    )
    def audit(samples: tf.Tensor, archived_final: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        flat = tf.reshape(samples, (results * 4, 4))
        values, scores = adapter.log_prob_and_grad_batch(flat)
        values = tf.reshape(tf.convert_to_tensor(values, tf.float64), (results, 4))
        scores = tf.reshape(
            tf.convert_to_tensor(scores, tf.float64), (results, 4, 4)
        )
        all_finite = tf.reduce_all(tf.math.is_finite(samples))
        all_finite = tf.logical_and(
            all_finite, tf.reduce_all(tf.math.is_finite(values))
        )
        all_finite = tf.logical_and(
            all_finite, tf.reduce_all(tf.math.is_finite(scores))
        )
        all_finite = tf.logical_and(
            all_finite, tf.reduce_all(tf.math.is_finite(archived_final))
        )
        maximum_residual = tf.reduce_max(tf.abs(values[-1] - archived_final))
        return all_finite, maximum_residual

    return audit


def _build_theta_mapper(adapter: Any, num_results: int) -> Any:
    results = int(num_results)
    if results <= 0:
        raise Phase7Error("theta mapper num_results must be positive")

    @tf.function(
        input_signature=(
            tf.TensorSpec(shape=(results, 4, 4), dtype=tf.float64),
        ),
        jit_compile=True,
        reduce_retracing=True,
    )
    def map_theta(samples: tf.Tensor) -> tf.Tensor:
        flat = tf.reshape(samples, (results * 4, 4))
        theta = adapter.latent_to_position(flat)
        return tf.reshape(tf.convert_to_tensor(theta, tf.float64), (results, 4, 4))

    return map_theta


def _mapped_theta_audit(theta: tf.Tensor, theta_mapper: Any) -> dict[str, Any]:
    theta_shape_valid = tuple(theta.shape) == (
        ACQUISITION_SEGMENT_RESULTS,
        4,
        4,
    )
    theta_finite = bool(tf.reduce_all(tf.math.is_finite(theta)).numpy())
    theta_device = str(theta.device)
    trace_count_fn = getattr(theta_mapper, "experimental_get_tracing_count", None)
    theta_trace_count = None if trace_count_fn is None else int(trace_count_fn())
    return {
        "shape_valid": theta_shape_valid,
        "all_finite": theta_finite,
        "jit_compile": True,
        "compile_trace_count": theta_trace_count,
        "output_device": theta_device,
        "passed": (
            theta_shape_valid
            and theta_finite
            and theta_trace_count == 1
            and "GPU:" in theta_device
        ),
        "diagnostic_role": "engineering_validity_hard_gate",
    }


def _post_archive_audit(auditor: Any, archive: Mapping[str, Any]) -> dict[str, Any]:
    samples = tf.convert_to_tensor(archive["samples"], tf.float64)
    archived_final = tf.convert_to_tensor(archive["final_target_log_prob"], tf.float64)
    all_finite_tensor, maximum_tensor = auditor(samples, archived_final)
    all_finite = bool(all_finite_tensor.numpy())
    maximum = float(maximum_tensor.numpy())
    trace_count_fn = getattr(auditor, "experimental_get_tracing_count", None)
    trace_count = None if trace_count_fn is None else int(trace_count_fn())
    passed = all_finite and maximum <= VALUE_MATCH_ABS
    return {
        "passed": passed,
        "retained_point_count": int(samples.shape[0]) * 4,
        "all_samples_values_scores_finite": all_finite,
        "final_target_log_prob_max_abs_residual": maximum,
        "threshold": VALUE_MATCH_ABS,
        "jit_compile": True,
        "compile_trace_count": trace_count,
        "diagnostic_role": "engineering_validity_hard_gate",
        "nonclaims": [
            "post-archive transformed value/score audit only",
            "no convergence or posterior-correctness claim",
        ],
    }


def _lineage_metadata(
    *,
    chart: str,
    role: str,
    segment_index: int,
    seed: tuple[int, int],
    previous: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return {
        "phase": 7,
        "chart": chart,
        "role": role,
        "segment_index": int(segment_index),
        "segment_seed": list(seed),
        "phase5_receipt_sha256": PHASE5_RECEIPT_SHA256,
        "phase6_receipt_sha256": PHASE6_RECEIPT_SHA256,
        "target_signature": TARGET_SIGNATURE,
        "transport_hash": TRANSPORT_HASHES[chart],
        "previous_manifest_sha256": (
            None if previous is None else previous["manifest_sha256"]
        ),
        "previous_final_state_sha256": (
            None if previous is None else previous["final_state_sha256"]
        ),
        "canary_excluded_from_retained_evidence": role == "mechanics_timing_canary",
    }


def _run_segment(
    *,
    runner: Any,
    auditor: Any,
    archive_dir: Path,
    label: str,
    chart: str,
    role: str,
    segment_index: int,
    current_state: tf.Tensor,
    seed: tuple[int, int],
    previous: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    started = time.perf_counter()
    result = runner.run(
        archive_dir=_absolute(archive_dir),
        archive_label=label,
        current_state=current_state,
        seed=seed,
        step_size=STEP_SIZE,
        metadata=_lineage_metadata(
            chart=chart,
            role=role,
            segment_index=segment_index,
            seed=seed,
            previous=previous,
        ),
        overwrite=False,
    )
    elapsed = time.perf_counter() - started
    archive = _read_private_archive(archive_dir, label)
    caller = archive["manifest"].get("metadata", {}).get("caller_metadata", {})
    expected_previous = None if previous is None else previous["manifest_sha256"]
    if caller.get("previous_manifest_sha256") != expected_previous:
        raise Phase7Error("private manifest previous-manifest lineage mismatch")
    expected_state = None if previous is None else previous["final_state_sha256"]
    if caller.get("previous_final_state_sha256") != expected_state:
        raise Phase7Error("private manifest previous-state lineage mismatch")
    audit = _post_archive_audit(auditor, archive)
    diagnostics = _json_safe(result.diagnostics)
    hard_vetoes: list[str] = []
    if not audit["passed"]:
        hard_vetoes.append("post_archive_value_score_audit_failed")
    if not diagnostics.get("retained_samples_all_finite"):
        hard_vetoes.append("nonfinite_retained_samples")
    if diagnostics.get("sampler_health_diagnostics", {}).get(
        "log_accept_ratio", {}
    ).get("nonfinite_count"):
        hard_vetoes.append("nonfinite_log_accept_ratio")
    if diagnostics.get("sampler_health_diagnostics", {}).get(
        "target_log_prob", {}
    ).get("nonfinite_count"):
        hard_vetoes.append("nonfinite_target_log_prob")
    if diagnostics.get("divergence_count") not in (None, 0):
        hard_vetoes.append("positive_native_divergence")
    devices = sorted({str(result.final_state.device), str(result.final_target_log_prob.device)})
    if not devices or not all("GPU:" in device for device in devices):
        hard_vetoes.append("trusted_gpu_output_placement_missing")
    public = {
        "label": label,
        "chart": chart,
        "role": role,
        "segment_index": segment_index,
        "seed": list(seed),
        "elapsed_seconds": elapsed,
        "archive_hashes": {
            "private_manifest_sha256": archive["manifest_sha256"],
            "sample_sha256": archive["sample_sha256"],
            "final_state_sha256": archive["final_state_sha256"],
            "final_target_log_prob_sha256": archive["final_target_log_prob_sha256"],
        },
        "lineage": {
            "previous_manifest_sha256": expected_previous,
            "previous_final_state_sha256": expected_state,
        },
        "diagnostics": diagnostics,
        "post_archive_value_score_audit": audit,
        "runner_metadata": _json_safe(result.metadata),
        "evidence_output_devices": devices,
        "hard_vetoes": hard_vetoes,
        "passed": not hard_vetoes,
    }
    return archive, public


def _build_runner(
    *, adapter: Any, initial_state: tf.Tensor, num_results: int, burnin: int, seed: tuple[int, int]
) -> Any:
    config = RetainedSampleHMCArchiveConfig(
        num_results=num_results,
        num_burnin_steps=burnin,
        step_size=STEP_SIZE,
        num_leapfrog_steps=NUM_LEAPFROG_STEPS,
        seed=seed,
        use_xla=True,
        target_scope=TARGET_SCOPE,
        chain_execution_mode="tf_function",
    )
    return build_retained_sample_hmc_archive_runner(adapter, initial_state, config)


def _cumulative_movement(samples: tf.Tensor, initial_state: tf.Tensor) -> list[bool]:
    previous = tf.concat((initial_state[tf.newaxis, ...], samples[:-1]), axis=0)
    moved = tf.reduce_any(tf.not_equal(samples, previous), axis=(0, 2))
    return [bool(value) for value in moved.numpy().tolist()]


def run_canary(*, output: Path, wall_cap_seconds: float) -> dict[str, Any]:
    if wall_cap_seconds <= 0.0 or not math.isfinite(wall_cap_seconds):
        raise Phase7Error("wall cap must be positive and finite")
    started_at = _now()
    started = time.perf_counter()
    upstream = validate_upstream_receipts()
    canary_seed_ledger()
    rows: dict[str, Any] = {}
    hard_vetoes: list[str] = []
    for chart in PAYLOADS:
        adapter, initial_state, binding = load_binding(chart)
        initial_runner = _build_runner(
            adapter=adapter,
            initial_state=initial_state,
            num_results=CANARY_RESULTS,
            burnin=CANARY_BURNIN,
            seed=CANARY_SEEDS[chart][0],
        )
        continuation_runner = _build_runner(
            adapter=adapter,
            initial_state=initial_state,
            num_results=CANARY_RESULTS,
            burnin=0,
            seed=CANARY_SEEDS[chart][1],
        )
        auditor = _build_post_archive_auditor(adapter, CANARY_RESULTS)
        chart_archives: list[dict[str, Any]] = []
        chart_rows: list[dict[str, Any]] = []
        current = initial_state
        for index, seed in enumerate(CANARY_SEEDS[chart]):
            runner = initial_runner if index == 0 else continuation_runner
            archive, row = _run_segment(
                runner=runner,
                auditor=auditor,
                archive_dir=PHASE7_ROOT / "canary-private" / chart,
                label=f"{chart}-canary-segment-{index:03d}",
                chart=chart,
                role="mechanics_timing_canary",
                segment_index=index,
                current_state=current,
                seed=seed,
                previous=None if index == 0 else chart_archives[-1],
            )
            chart_archives.append(archive)
            chart_rows.append(row)
            current = archive["final_state"]
            hard_vetoes.extend(f"{chart}:{item}" for item in row["hard_vetoes"])
            if time.perf_counter() - started > wall_cap_seconds:
                raise Phase7Error("Stage A wall cap exceeded")
        samples = tf.concat([item["samples"] for item in chart_archives], axis=0)
        movement = _cumulative_movement(samples, initial_state)
        if not all(movement):
            hard_vetoes.append(f"{chart}:unmoved_chain")
        trace_counts = [row["runner_metadata"]["compile_trace_count"] for row in chart_rows]
        if trace_counts != [1, 1, 1]:
            hard_vetoes.append(f"{chart}:unexpected_xla_trace_count")
        audit_trace_counts = [
            row["post_archive_value_score_audit"]["compile_trace_count"]
            for row in chart_rows
        ]
        if audit_trace_counts != [1, 1, 1]:
            hard_vetoes.append(f"{chart}:unexpected_audit_xla_trace_count")
        rows[chart] = {
            "binding": binding,
            "segments": chart_rows,
            "cumulative_chain_moved": movement,
            "retained_draws_per_chain": CANARY_RESULTS * 3,
            "excluded_from_retained_evidence": True,
        }
    wall_time = time.perf_counter() - started
    decision = (
        "PHASE7_STAGE_A_TIMING_CANARY_PASSED_BUDGET_FREEZE_REQUIRED"
        if not hard_vetoes
        else "PHASE7_STAGE_A_TIMING_CANARY_FAILED"
    )
    payload = {
        "schema": "bayesfilter.ssl_lstm_neutra.phase7_timing_canary.v1",
        "status": "PASSED" if not hard_vetoes else "FAILED",
        "decision": decision,
        "stage": "timing-canary",
        "upstream_bindings": upstream,
        "contract": {
            "segment_results": CANARY_RESULTS,
            "initial_burnin": CANARY_BURNIN,
            "continuation_burnin": 0,
            "segments_per_chart": 3,
            "step_size": STEP_SIZE,
            "num_leapfrog_steps": NUM_LEAPFROG_STEPS,
            "seeds": _json_safe(CANARY_SEEDS),
            "samples_excluded_from_retained_evidence": True,
        },
        "charts": rows,
        "hard_vetoes": hard_vetoes,
        "source_bindings": _source_bindings(),
        "run_manifest": _run_manifest(
            stage="timing-canary",
            started_at=started_at,
            wall_time=wall_time,
            wall_cap_seconds=wall_cap_seconds,
            output=output,
            seeds=CANARY_SEEDS,
        ),
        "nonclaims": [
            "mechanics and timing canary only",
            "canary samples are never retained evidence",
            "acceptance and convergence diagnostics cannot promote this canary",
            "no posterior, stationarity, predictive, or ranking claim",
        ],
    }
    _write_json(output, payload)
    return payload


def _research_failure_classification(hard_vetoes: Sequence[str]) -> dict[str, Any]:
    invalidity_tokens = {
        "post_archive_value_score_audit_failed",
        "nonfinite_retained_samples",
        "nonfinite_log_accept_ratio",
        "nonfinite_target_log_prob",
        "nonfinite_or_invalid_mapped_theta",
        "trusted_gpu_output_placement_missing",
    }
    invalid = [item for item in hard_vetoes if item.split(":")[-1] in invalidity_tokens]
    resource = [
        item for item in hard_vetoes if item.split(":")[-1] == "resource_cap_exhausted"
    ]
    sampler = [item for item in hard_vetoes if item not in invalid and item not in resource]
    return {
        "evidence_or_implementation_invalidity": invalid,
        "resource_continuation_veto": resource,
        "sampler_hard_veto": sampler,
        "candidate_rejection_is_not_research_direction_rejection": True,
    }


def run_acquisition(*, output: Path, wall_cap_seconds: float) -> dict[str, Any]:
    if abs(float(wall_cap_seconds) - ACQUISITION_WALL_CAP_SECONDS) > 1.0e-9:
        raise Phase7Error("acquisition wall cap must equal the frozen 2100 seconds")
    started_at = _now()
    started = time.perf_counter()
    upstream = validate_upstream_receipts()
    canary = validate_canary_receipt()
    seed_ledger = {
        label: acquisition_seeds(label, ACQUISITION_CHECKPOINT_SEGMENTS[-1])
        for label in PAYLOADS
    }
    validate_seed_ledger(seed_ledger, disjoint_below=PHASE6_MAX_SEED_COMPONENT)
    chart_receipts: dict[str, Any] = {}
    admitted_theta: dict[str, tf.Tensor] = {}
    all_hard_vetoes: list[str] = []

    for chart in PAYLOADS:
        chart_started = time.perf_counter()
        adapter, initial_state, binding = load_binding(chart)
        initial_runner = _build_runner(
            adapter=adapter,
            initial_state=initial_state,
            num_results=ACQUISITION_SEGMENT_RESULTS,
            burnin=ACQUISITION_BURNIN,
            seed=seed_ledger[chart][0],
        )
        continuation_runner = _build_runner(
            adapter=adapter,
            initial_state=initial_state,
            num_results=ACQUISITION_SEGMENT_RESULTS,
            burnin=0,
            seed=seed_ledger[chart][1],
        )
        auditor = _build_post_archive_auditor(adapter, ACQUISITION_SEGMENT_RESULTS)
        theta_mapper = _build_theta_mapper(adapter, ACQUISITION_SEGMENT_RESULTS)
        archives: list[dict[str, Any]] = []
        public_segments: list[dict[str, Any]] = []
        theta_segments: list[tf.Tensor] = []
        checkpoints: list[dict[str, Any]] = []
        current = initial_state
        final_admission: dict[str, Any] = {
            "admitted": False,
            "decision": "NOT_YET_EVALUATED",
        }
        chart_hard_vetoes: list[str] = []

        for index, seed in enumerate(seed_ledger[chart]):
            elapsed_total = time.perf_counter() - started
            elapsed_chart = time.perf_counter() - chart_started
            if elapsed_total >= wall_cap_seconds or elapsed_chart >= ACQUISITION_CHART_WALL_CAP_SECONDS:
                chart_hard_vetoes.append("resource_cap_exhausted")
                break
            archive, public = _run_segment(
                runner=initial_runner if index == 0 else continuation_runner,
                auditor=auditor,
                archive_dir=PHASE7_ROOT / "retained-private" / chart,
                label=f"{chart}-retained-segment-{index:03d}",
                chart=chart,
                role="retained_admission",
                segment_index=index,
                current_state=current,
                seed=seed,
                previous=None if index == 0 else archives[-1],
            )
            archives.append(archive)
            public_segments.append(public)
            current = archive["final_state"]
            theta = theta_mapper(tf.convert_to_tensor(archive["samples"], tf.float64))
            public["mapped_theta_audit"] = _mapped_theta_audit(theta, theta_mapper)
            if not public["mapped_theta_audit"]["passed"]:
                public["hard_vetoes"].append("nonfinite_or_invalid_mapped_theta")
                public["passed"] = False
            theta_segments.append(theta)
            if public["hard_vetoes"]:
                chart_hard_vetoes.extend(public["hard_vetoes"])
                final_admission = {
                    "admitted": False,
                    "decision": "HARD_VETO_STOP",
                    "hard_vetoes": list(dict.fromkeys(chart_hard_vetoes)),
                    "promotion_vetoes": [],
                    "draw_count_per_chain": (index + 1) * ACQUISITION_SEGMENT_RESULTS,
                }
                break
            if (
                time.perf_counter() - chart_started
                > ACQUISITION_CHART_WALL_CAP_SECONDS
                or time.perf_counter() - started > wall_cap_seconds
            ):
                chart_hard_vetoes.append("resource_cap_exhausted")
                final_admission = {
                    "admitted": False,
                    "decision": "HARD_VETO_STOP",
                    "hard_vetoes": ["resource_cap_exhausted"],
                    "promotion_vetoes": [],
                    "draw_count_per_chain": (index + 1) * ACQUISITION_SEGMENT_RESULTS,
                }
                break
            if index + 1 in ACQUISITION_CHECKPOINT_SEGMENTS:
                z_cumulative = tf.concat([item["samples"] for item in archives], axis=0)
                theta_cumulative = tf.concat(theta_segments, axis=0)
                admission = cumulative_admission(
                    z_draw_major=z_cumulative,
                    theta_draw_major=theta_cumulative,
                    initial_state=initial_state,
                    segment_manifests=[item["manifest"] for item in archives],
                )
                checkpoints.append(admission)
                final_admission = admission
                if admission["hard_vetoes"]:
                    chart_hard_vetoes.extend(admission["hard_vetoes"])
                    break
                if admission["admitted"]:
                    admitted_theta[chart] = tf.transpose(theta_cumulative, (1, 0, 2))
                    break
        if final_admission.get("decision") == "NOT_YET_EVALUATED":
            final_admission = {
                "admitted": False,
                "decision": "HARD_VETO_STOP",
                "hard_vetoes": list(dict.fromkeys(chart_hard_vetoes)),
                "promotion_vetoes": [],
                "draw_count_per_chain": len(archives) * ACQUISITION_SEGMENT_RESULTS,
            }
        elif (
            final_admission.get("admitted") is not True
            and not final_admission.get("hard_vetoes")
            and len(archives) == ACQUISITION_CHECKPOINT_SEGMENTS[-1]
        ):
            final_admission = dict(final_admission)
            final_admission["decision"] = (
                "MAXIMUM_OPPORTUNITY_EXHAUSTED_NOT_ADMITTED"
            )
        all_hard_vetoes.extend(f"{chart}:{item}" for item in chart_hard_vetoes)
        chart_receipts[chart] = {
            "binding": binding,
            "segments": public_segments,
            "checkpoints": checkpoints,
            "final_admission": final_admission,
            "executed_segment_count": len(archives),
            "executed_draws_per_chain": len(archives) * ACQUISITION_SEGMENT_RESULTS,
            "chart_wall_time_seconds": time.perf_counter() - chart_started,
            "chart_wall_cap_seconds": ACQUISITION_CHART_WALL_CAP_SECONDS,
            "canary_samples_included": False,
        }

    admissions = {
        label: chart_receipts[label]["final_admission"] for label in PAYLOADS
    }
    both_admitted = all(item.get("admitted") is True for item in admissions.values())
    if both_admitted:
        stability: Mapping[str, Any] = admitted_cross_replication_stability(
            admissions=admissions,
            theta_chain_major=admitted_theta,
        )
    else:
        stability = {
            "decision": "NOT_REACHED_BOTH_CHARTS_NOT_ADMITTED",
            "passed": False,
            "nonclaims": ["no partial G/H comparison"],
        }
    classification = _research_failure_classification(all_hard_vetoes)
    if classification["evidence_or_implementation_invalidity"]:
        decision = "PHASE7_EVIDENCE_INVALID_BLOCKER"
        status = "FAILED"
    elif classification["sampler_hard_veto"]:
        decision = "PHASE7_SAMPLER_HARD_VETO_BLOCKER"
        status = "PASSED"
    elif classification["resource_continuation_veto"]:
        decision = "PHASE7_RESOURCE_CAP_EXHAUSTED_VALID_INCOMPLETE_EVIDENCE"
        status = "PASSED"
    elif not both_admitted:
        decision = "PHASE7_RETAINED_ADMISSION_NOT_MET"
        status = "PASSED"
    elif stability.get("passed") is not True:
        decision = "PHASE7_CROSS_REPLICATION_STABILITY_VETO"
        status = "PASSED"
    else:
        decision = "PHASE7_RETAINED_ADMISSION_PASSED_PHASE8_HANDOFF"
        status = "PASSED"
    wall_time = time.perf_counter() - started
    payload = {
        "schema": "bayesfilter.ssl_lstm_neutra.phase7_retained_acquisition.v1",
        "status": status,
        "decision": decision,
        "stage": "acquisition",
        "upstream_bindings": upstream,
        "canary_binding": canary,
        "contract": {
            "segment_results": ACQUISITION_SEGMENT_RESULTS,
            "initial_burnin": ACQUISITION_BURNIN,
            "continuation_burnin": 0,
            "checkpoint_segments": list(ACQUISITION_CHECKPOINT_SEGMENTS),
            "checkpoint_draws_per_chain": [
                value * ACQUISITION_SEGMENT_RESULTS
                for value in ACQUISITION_CHECKPOINT_SEGMENTS
            ],
            "chart_wall_cap_seconds": ACQUISITION_CHART_WALL_CAP_SECONDS,
            "cumulative_wall_cap_seconds": ACQUISITION_WALL_CAP_SECONDS,
            "step_size": STEP_SIZE,
            "num_leapfrog_steps": NUM_LEAPFROG_STEPS,
            "mass_matrix": "identity",
            "canary_samples_included": False,
        },
        "charts": chart_receipts,
        "both_charts_admitted": both_admitted,
        "cross_replication_stability": stability,
        "hard_vetoes": list(dict.fromkeys(all_hard_vetoes)),
        "failure_classification": classification,
        "source_bindings": _source_bindings(),
        "run_manifest": _run_manifest(
            stage="acquisition",
            started_at=started_at,
            wall_time=wall_time,
            wall_cap_seconds=wall_cap_seconds,
            output=output,
            seeds=seed_ledger,
        ),
        "inference_status": {
            "hard_veto_screen": (
                "failed"
                if (
                    classification["evidence_or_implementation_invalidity"]
                    or classification["sampler_hard_veto"]
                )
                else "passed"
            ),
            "statistically_supported_ranking": "none",
            "descriptive_only_differences": "all continuous G/H diagnostics",
            "default_readiness": "not_assessed",
            "next_evidence_needed": (
                "Phase 8 predictive moments" if decision.endswith("PHASE8_HANDOFF")
                else "repair or additional authority named by the Phase 7 result"
            ),
        },
        "nonclaims": [
            "finite-sample retained admission screen only",
            "no posterior oracle, posterior truth, or stationarity proof",
            "no sampler or transport ranking",
            "no predictive equivalence or broad scientific-validity claim",
        ],
    }
    _write_json(output, payload)
    return payload


def _coordinate_screen(samples: tf.Tensor) -> tuple[dict[str, Any], list[str]]:
    values = _json_safe(compute_coordinate_diagnostics(samples))
    failures: list[str] = []
    rhat = values["rank_normalized_split_rhat"]["maximum"]
    bulk = values["rank_normalized_ess"]["bulk"]
    tail = values["rank_normalized_ess"]["tail"]
    ratio = values["mean"]["mcse_sd_ratio"]
    arrays = (rhat, bulk, tail, ratio)
    if not all(math.isfinite(float(item)) for array in arrays for item in array):
        failures.append("nonfinite_rank_ess_or_mcse")
        return values, failures
    if max(float(item) for item in rhat) > R_HAT_MAX:
        failures.append("rank_normalized_split_rhat_above_threshold")
    if min(float(item) for item in bulk) < ESS_MIN:
        failures.append("bulk_ess_below_threshold")
    if min(float(item) for item in tail) < ESS_MIN:
        failures.append("tail_ess_below_threshold")
    if max(float(item) for item in ratio) > MCSE_SD_RATIO_MAX:
        failures.append("mcse_sd_ratio_above_threshold")
    return values, failures


def cumulative_admission(
    *,
    z_draw_major: tf.Tensor,
    theta_draw_major: tf.Tensor,
    initial_state: tf.Tensor,
    segment_manifests: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    z = tf.convert_to_tensor(z_draw_major, tf.float64)
    theta = tf.convert_to_tensor(theta_draw_major, tf.float64)
    if z.shape.rank != 3 or tuple(z.shape[1:]) != (4, 4) or theta.shape != z.shape:
        raise Phase7Error("cumulative z/theta samples must have shape [draw,4,4]")
    if int(z.shape[0]) < 4 or int(z.shape[0]) % 2:
        raise Phase7Error("diagnostic draw count must be even and at least four")
    hard_vetoes: list[str] = []
    promotion_vetoes: list[str] = []
    if not bool(tf.reduce_all(tf.math.is_finite(z)).numpy()) or not bool(
        tf.reduce_all(tf.math.is_finite(theta)).numpy()
    ):
        hard_vetoes.append("nonfinite_z_or_theta")
    moved = _cumulative_movement(z, initial_state)
    if not all(moved):
        hard_vetoes.append("unmoved_chain")
    accepted = tf.zeros((4,), tf.int64)
    total = tf.zeros((4,), tf.int64)
    divergence_statuses: list[str] = []
    for manifest in segment_manifests:
        diagnostics = manifest["diagnostics_private_metadata"]
        health = diagnostics["sampler_health_diagnostics"]
        rates = [float(item) for item in health["acceptance_rate_by_chain"]]
        draws = int(manifest["retained_sample_count"])
        counts = [int(round(draws * rate)) for rate in rates]
        if any(abs(draws * rate - count) > 1.0e-9 for rate, count in zip(rates, counts, strict=True)):
            raise Phase7Error("acceptance rates do not reconstruct exact counts")
        accepted += tf.constant(counts, tf.int64)
        total += tf.fill((4,), tf.cast(draws, tf.int64))
        if health["log_accept_ratio"]["nonfinite_count"]:
            hard_vetoes.append("nonfinite_log_accept_ratio")
        if health["target_log_prob"]["nonfinite_count"]:
            hard_vetoes.append("nonfinite_target_log_prob")
        divergence_statuses.append(str(diagnostics["native_divergence_status"]))
        if diagnostics.get("divergence_count") not in (None, 0):
            hard_vetoes.append("positive_native_divergence")
    rates = tf.cast(accepted, tf.float64) / tf.cast(total, tf.float64)
    aggregate = float(tf.reduce_sum(tf.cast(accepted, tf.float64)).numpy() / tf.reduce_sum(tf.cast(total, tf.float64)).numpy())
    rate_values = [float(item) for item in rates.numpy().tolist()]
    if not ACCEPTANCE_BAND[0] <= aggregate <= ACCEPTANCE_BAND[1]:
        promotion_vetoes.append("aggregate_acceptance_outside_threshold")
    if any(not ACCEPTANCE_BAND[0] <= item <= ACCEPTANCE_BAND[1] for item in rate_values):
        promotion_vetoes.append("per_chain_acceptance_outside_threshold")
    if hard_vetoes:
        coordinates: Mapping[str, Any] = {"status": "not_computed_hard_veto"}
    else:
        z_diag, z_failures = _coordinate_screen(tf.transpose(z, (1, 0, 2)))
        theta_diag, theta_failures = _coordinate_screen(tf.transpose(theta, (1, 0, 2)))
        promotion_vetoes.extend(f"z:{item}" for item in z_failures)
        promotion_vetoes.extend(f"theta:{item}" for item in theta_failures)
        coordinates = {"z": z_diag, "theta": theta_diag}
    admitted = not hard_vetoes and not promotion_vetoes
    return {
        "admitted": admitted,
        "decision": "ADMITTED" if admitted else (
            "HARD_VETO_STOP" if hard_vetoes else "EXTEND_TO_NEXT_FROZEN_CHECKPOINT"
        ),
        "draw_count_per_chain": int(z.shape[0]),
        "chain_moved": moved,
        "acceptance_rate": aggregate,
        "acceptance_rate_by_chain": rate_values,
        "native_divergence_statuses": divergence_statuses,
        "hard_vetoes": list(dict.fromkeys(hard_vetoes)),
        "promotion_vetoes": list(dict.fromkeys(promotion_vetoes)),
        "coordinate_diagnostics": coordinates,
        "thresholds": {
            "rhat_max": R_HAT_MAX,
            "bulk_ess_min": ESS_MIN,
            "tail_ess_min": ESS_MIN,
            "mcse_sd_ratio_max": MCSE_SD_RATIO_MAX,
            "acceptance_band": list(ACCEPTANCE_BAND),
        },
    }


def _functional_draws(theta_chain_major: tf.Tensor) -> tf.Tensor:
    theta = tf.convert_to_tensor(theta_chain_major, tf.float64)
    if theta.shape.rank != 3 or tuple(theta.shape[2:]) != (4,):
        raise Phase7Error("theta must have shape [chain,draw,4]")
    seconds = tf.stack(
        [theta[..., left] * theta[..., right] for left, right in SECOND_MOMENT_INDICES],
        axis=-1,
    )
    return tf.concat((theta, seconds), axis=-1)


def cross_replication_stability(
    theta_g_chain_major: tf.Tensor, theta_h_chain_major: tf.Tensor
) -> dict[str, Any]:
    functionals = {
        "fresh-g": _functional_draws(theta_g_chain_major),
        "fresh-h": _functional_draws(theta_h_chain_major),
    }
    diagnostics = {
        label: posterior_mean_diagnostics(values) for label, values in functionals.items()
    }
    mean_g = tf.convert_to_tensor(diagnostics["fresh-g"]["pooled_mean"], tf.float64)
    mean_h = tf.convert_to_tensor(diagnostics["fresh-h"]["pooled_mean"], tf.float64)
    mcse_g = tf.convert_to_tensor(diagnostics["fresh-g"]["mean_mcse"], tf.float64)
    mcse_h = tf.convert_to_tensor(diagnostics["fresh-h"]["mean_mcse"], tf.float64)
    denominator = tf.sqrt(tf.square(mcse_g) + tf.square(mcse_h))
    standardized = tf.where(
        denominator > 0.0,
        tf.abs(mean_g - mean_h) / denominator,
        tf.fill(tf.shape(denominator), tf.constant(float("nan"), tf.float64)),
    )
    finite = bool(tf.reduce_all(tf.math.is_finite(standardized)).numpy())
    maximum = float(tf.reduce_max(standardized).numpy()) if finite else float("nan")
    passed = finite and maximum <= CROSS_REPLICATION_Z_MAX
    names = [
        *(f"mean_theta_{index}" for index in range(4)),
        *(f"raw_second_theta_{left}_{right}" for left, right in SECOND_MOMENT_INDICES),
    ]
    return {
        "passed": passed,
        "decision": "STABILITY_SCREEN_PASSED" if passed else "STABILITY_PROMOTION_VETO",
        "functional_names": names,
        "standardized_absolute_difference": _json_safe(standardized),
        "maximum_standardized_absolute_difference": _json_safe(maximum),
        "threshold": CROSS_REPLICATION_Z_MAX,
        "means": {label: _json_safe(values["pooled_mean"]) for label, values in diagnostics.items()},
        "mean_mcse": {label: _json_safe(values["mean_mcse"]) for label, values in diagnostics.items()},
        "inference_status": {
            "statistically_supported_ranking": "none",
            "descriptive_only_differences": "all continuous G/H differences",
            "interpretation": "conservative MCSE-aware simultaneous stability screen",
        },
        "nonclaims": [
            "not a formal equivalence test",
            "neither chart is an oracle",
            "no sampler, transport, or method ranking",
        ],
    }


def admitted_cross_replication_stability(
    *,
    admissions: Mapping[str, Mapping[str, Any]],
    theta_chain_major: Mapping[str, tf.Tensor],
) -> dict[str, Any]:
    expected = ("fresh-g", "fresh-h")
    if set(admissions) != set(expected) or set(theta_chain_major) != set(expected):
        raise Phase7Error("cross-replication requires exact G/H inputs")
    not_admitted = [
        label for label in expected if admissions[label].get("admitted") is not True
    ]
    if not_admitted:
        raise Phase7Error(
            "cross-replication requires both independent admissions: "
            + ",".join(not_admitted)
        )
    return cross_replication_stability(
        theta_chain_major["fresh-g"], theta_chain_major["fresh-h"]
    )


def _run_manifest(
    *,
    stage: str,
    started_at: str,
    wall_time: float,
    wall_cap_seconds: float,
    output: Path,
    seeds: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "stage": stage,
        "command": " ".join(
            shlex.quote(item) for item in (sys.executable, *sys.argv)
        ),
        "cwd": str(ROOT),
        "interpreter": sys.executable,
        "python_version": platform.python_version(),
        "tensorflow": tf.__version__,
        "tensorflow_probability": __import__("tensorflow_probability").__version__,
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "physical_devices": [device.name for device in tf.config.list_physical_devices("GPU")],
        "logical_devices": [device.name for device in tf.config.list_logical_devices("GPU")],
        "jit_compile": True,
        "dtype": "float64",
        "tf32_enabled": bool(tf.config.experimental.tensor_float_32_execution_enabled()),
        "trust_basis": "owner_designated_managed_session_visible_gpu_trusted",
        "git_commit": _git("rev-parse", "HEAD").strip(),
        "git_dirty": bool(_git("status", "--porcelain").strip()),
        "random_seeds": _json_safe(seeds),
        "started_at_utc": started_at,
        "completed_at_utc": _now(),
        "wall_time_seconds": wall_time,
        "wall_cap_seconds": wall_cap_seconds,
        "output_path": output.as_posix(),
        "plan_path": PLAN_PATH.as_posix(),
        "result_path": RESULT_PATH.as_posix(),
        "canary_samples_retained_as_posterior_evidence": False,
    }


def _require_gpu() -> None:
    physical = tf.config.list_physical_devices("GPU")
    if not physical:
        raise Phase7Error("Phase 7 requires a visible trusted GPU")
    for gpu in physical:
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.experimental.enable_tensor_float_32_execution(True)
    if not tf.config.list_logical_devices("GPU"):
        raise Phase7Error("Phase 7 requires a logical GPU")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("timing-canary", "acquisition"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wall-cap-seconds", type=float, required=True)
    args = parser.parse_args(argv)
    _require_gpu()
    with tf.device("/GPU:0"):
        if args.stage == "timing-canary":
            payload = run_canary(
                output=args.output,
                wall_cap_seconds=float(args.wall_cap_seconds),
            )
        else:
            payload = run_acquisition(
                output=args.output,
                wall_cap_seconds=float(args.wall_cap_seconds),
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
