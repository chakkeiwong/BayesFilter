#!/usr/bin/env python3
"""Calibration-only target pilot for the Phase 8 predictive design."""

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
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import tensorflow as tf


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import bayesfilter.inference.predictive_equivalence as predictive_equivalence  # noqa: E402
from bayesfilter.inference.neutra_artifacts import load_frozen_neutra_artifact  # noqa: E402
from bayesfilter.inference.predictive_equivalence import (  # noqa: E402
    chain_batch_long_run_covariance,
    mean_log_variance_influence,
    pooled_pairwise_distance_scale,
)
from bayesfilter.nonlinear.ssl_lstm_predictive_tf import (  # noqa: E402
    SSLLSTMForecastConfig,
    forecast_ssl_lstm_paths,
    make_ssl_lstm_innovation_bank,
    ssl_lstm_forecast_compiled_program,
    ssl_lstm_terminal_covariance_audit_compiled_program,
    ssl_lstm_terminal_compiled_program,
)


PLAN_PATH = Path(
    "docs/plans/bayesfilter-ssl-lstm-neutra-phase-8-predictive-design-refresh-"
    "plan-2026-07-17.md"
)
RESULT_PATH = Path(
    "docs/plans/bayesfilter-ssl-lstm-neutra-phase-8-predictive-design-refresh-"
    "result-2026-07-17.md"
)
SCRIPT_PATH = Path(__file__).resolve().relative_to(ROOT)
PHASE7_ROOT = Path(
    "docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/"
    "phase-7-retained-admission"
)
PHASE7_RECEIPT_PATH = PHASE7_ROOT / "retained-acquisition.json"
PHASE7_RECEIPT_SHA256 = (
    "b79e5f6041e284de40bbd3834cc909fd12f45d012f172e570acccaa62dbe31a5"
)
CANARY_RECEIPT_PATH = Path(
    "docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/"
    "phase-8-predictive-design/engineering-canary-repair-01.json"
)
CANARY_RECEIPT_SHA256 = (
    "5924b550b1ca5b18d276bd8ea3a3a15cd27b28f95d25f4a7669bd3804f5a9127"
)
CHUNK_CANARY_RECEIPT_PATH = Path(
    "docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/"
    "phase-8-predictive-design/forecast-chunk-repair-canary.json"
)
CHUNK_CANARY_RECEIPT_SHA256 = (
    "e78e76203278548183f7974562249e3a292ae4f21e315cd137b955131e342587"
)
CHUNK_PREFIX_RECEIPT_PATH = Path(
    "docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/"
    "phase-8-predictive-design/forecast-chunk-exact-prefix-validation.json"
)
CHUNK_PREFIX_RECEIPT_SHA256 = (
    "f272d6eb407e8d3dbe11ebac2f1dcfac8a16cacd6ce5c92ca074c6bb050cb0d1"
)
DISTANCE_CANARY_RECEIPT_PATH = Path(
    "docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/"
    "phase-8-predictive-design/pairwise-distance-exact-shape-canary.json"
)
DISTANCE_CANARY_RECEIPT_SHA256 = (
    "fd8489ec557c49a0169af8a656d6619535ab41ca85c07b9ba167b593e29c0871"
)
ARTIFACT_ROOT = Path(
    "docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/"
    "trial0-alternative-confirmation-2026-07-16"
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
TARGET_SIGNATURE = (
    "549efdf2aa5d9534226cb29c3678489d92766f92e6140901355eac33618f719e"
)
PILOT_DRAWS_PER_CHAIN = 64
CONFIRMATION_DRAWS_PER_CHAIN = 448
TOTAL_DRAWS_PER_CHAIN = 512
BLOCK_LENGTH = 16
FORECAST_DRAW_CHUNK_SIZE = 16
BANDWIDTH_FACTORS = (0.25, 0.5, 1.0, 2.0, 4.0)
RIDGE_LADDER = (0.0, 1.0e-12, 1.0e-10, 1.0e-8, 1.0e-6)
CONDITION_NUMBER_MAX = 1.0e8
PILOT_SEED = (12001, 12002)
ARM_IDS = {"fresh-g": 1, "fresh-h": 2}


class Phase8PilotError(RuntimeError):
    """Raised when the Phase 8 target-pilot contract fails."""


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _absolute(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _sha256(path: Path) -> str:
    return hashlib.sha256(_absolute(path).read_bytes()).hexdigest()


def _strict_json(path: Path) -> dict[str, Any]:
    def pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in rows:
            if key in result:
                raise Phase8PilotError(f"duplicate JSON key {key!r}: {path}")
            result[key] = value
        return result

    def reject(value: str) -> None:
        raise Phase8PilotError(f"nonfinite JSON constant {value!r}: {path}")

    value = json.loads(
        _absolute(path).read_text(encoding="utf-8"),
        object_pairs_hook=pairs,
        parse_constant=reject,
    )
    if not isinstance(value, dict):
        raise Phase8PilotError(f"expected JSON object: {path}")
    return value


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, bytes):
        return value.decode("utf-8")
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


def _canonical(payload: Any) -> bytes:
    return (
        json.dumps(
            _json_safe(payload),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    absolute = _absolute(path)
    absolute.parent.mkdir(parents=True, exist_ok=True)
    if absolute.exists():
        raise Phase8PilotError(f"refusing to overwrite receipt: {path}")
    absolute.write_bytes(_canonical(payload))


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


def _trace_count(program: Any) -> int | None:
    method = getattr(program, "experimental_get_tracing_count", None)
    return None if method is None else int(method())


def validate_entry_receipts() -> dict[str, Any]:
    if _sha256(PHASE7_RECEIPT_PATH) != PHASE7_RECEIPT_SHA256:
        raise Phase8PilotError("Phase 7 receipt byte identity drift")
    phase7 = _strict_json(PHASE7_RECEIPT_PATH)
    if (
        phase7.get("status") != "PASSED"
        or phase7.get("decision")
        != "PHASE7_RETAINED_ADMISSION_PASSED_PHASE8_HANDOFF"
        or phase7.get("both_charts_admitted") is not True
        or phase7.get("hard_vetoes") != []
    ):
        raise Phase8PilotError("Phase 7 admission state drift")
    if _sha256(CANARY_RECEIPT_PATH) != CANARY_RECEIPT_SHA256:
        raise Phase8PilotError("Phase 8 engineering-canary byte identity drift")
    canary = _strict_json(CANARY_RECEIPT_PATH)
    if (
        canary.get("status") != "PASSED"
        or canary.get("decision")
        != "PHASE8_ENGINEERING_CANARY_PASSED_RESOURCE_FREEZE_REQUIRED"
        or canary.get("contract", {}).get("retained_phase7_samples_read") is not False
        or canary.get("contract", {}).get("confirmatory_forecast_bank_opened") is not False
    ):
        raise Phase8PilotError("Phase 8 engineering-canary state drift")
    if _sha256(CHUNK_CANARY_RECEIPT_PATH) != CHUNK_CANARY_RECEIPT_SHA256:
        raise Phase8PilotError("Phase 8 chunk-canary byte identity drift")
    chunk_canary = _strict_json(CHUNK_CANARY_RECEIPT_PATH)
    if (
        chunk_canary.get("status") != "PASSED"
        or chunk_canary.get("decision")
        != "PHASE8_FORECAST_CHUNK_REPAIR_PASSED_EXACT_PREFIX_VALIDATION_REQUIRED"
        or chunk_canary.get("contract", {}).get("draw_chunk_size")
        != FORECAST_DRAW_CHUNK_SIZE
    ):
        raise Phase8PilotError("Phase 8 chunk-canary state drift")
    if _sha256(CHUNK_PREFIX_RECEIPT_PATH) != CHUNK_PREFIX_RECEIPT_SHA256:
        raise Phase8PilotError("Phase 8 chunk-prefix receipt byte identity drift")
    chunk_prefix = _strict_json(CHUNK_PREFIX_RECEIPT_PATH)
    if (
        chunk_prefix.get("status") != "PASSED"
        or chunk_prefix.get("decision")
        != "PHASE8_FORECAST_CHUNK_EXACT_PREFIX_VALIDATED_PILOT_REPAIR_02_ELIGIBLE"
        or chunk_prefix.get("scope", {}).get("draw_chunk_size")
        != FORECAST_DRAW_CHUNK_SIZE
        or chunk_prefix.get("scope", {}).get("predictive_summary_computed") is not False
        or chunk_prefix.get("scope", {}).get("g_h_predictive_difference_computed")
        is not False
        or chunk_prefix.get("scope", {}).get("target_pilot_retried") is not False
    ):
        raise Phase8PilotError("Phase 8 chunk-prefix validation state drift")
    if _sha256(DISTANCE_CANARY_RECEIPT_PATH) != DISTANCE_CANARY_RECEIPT_SHA256:
        raise Phase8PilotError("Phase 8 distance-canary byte identity drift")
    distance_canary = _strict_json(DISTANCE_CANARY_RECEIPT_PATH)
    if (
        distance_canary.get("status") != "PASSED"
        or distance_canary.get("decision")
        != "PHASE8_PAIRWISE_DISTANCE_EXACT_SHAPE_REPAIR_PASSED_PILOT_REPAIR_03_ELIGIBLE"
        or distance_canary.get("contract", {}).get("path_shape") != [8, 64, 2, 10]
        or distance_canary.get("contract", {}).get("retained_samples_read") is not False
        or distance_canary.get("contract", {}).get("forecast_artifacts_read") is not False
        or distance_canary.get("contract", {}).get("g_h_difference_computed") is not False
        or distance_canary.get("result", {}).get("xla_trace_count") != 1
    ):
        raise Phase8PilotError("Phase 8 distance-canary state drift")
    return {
        "phase7": phase7,
        "canary": canary,
        "chunk_canary": chunk_canary,
        "chunk_prefix": chunk_prefix,
        "distance_canary": distance_canary,
    }


def _inside_phase7(path: Path) -> bool:
    try:
        _absolute(path).resolve().relative_to(_absolute(PHASE7_ROOT).resolve())
    except ValueError:
        return False
    return True


def _verify_manifest_files(
    manifest: Mapping[str, Any], public_hashes: Mapping[str, str]
) -> tuple[Path, dict[str, str]]:
    shards = manifest.get("sample_shards")
    if not isinstance(shards, list) or len(shards) != 1:
        raise Phase8PilotError("retained archive must contain one sample shard")
    sample = shards[0]
    sidecars = manifest.get("sidecars")
    if not isinstance(sidecars, Mapping):
        raise Phase8PilotError("retained archive sidecars missing")
    rows = {
        "sample": (sample, "sample_sha256"),
        "final_state": (sidecars.get("final_state"), "final_state_sha256"),
        "final_target": (
            sidecars.get("final_target_log_prob"),
            "final_target_log_prob_sha256",
        ),
    }
    verified: dict[str, str] = {}
    sample_path: Path | None = None
    for role, (row, public_key) in rows.items():
        if not isinstance(row, Mapping):
            raise Phase8PilotError(f"retained {role} metadata missing")
        path = Path(str(row.get("path")))
        if not _inside_phase7(path):
            raise Phase8PilotError(f"retained {role} path escapes Phase 7 root")
        observed = _sha256(path)
        if observed != row.get("sha256") or observed != public_hashes.get(public_key):
            raise Phase8PilotError(f"retained {role} hash mismatch")
        verified[role] = observed
        if role == "sample":
            sample_path = path
    assert sample_path is not None
    return sample_path, verified


def read_frozen_pilot_prefix(phase7: Mapping[str, Any]) -> tuple[dict[str, tf.Tensor], dict[str, Any]]:
    samples: dict[str, tf.Tensor] = {}
    audit: dict[str, Any] = {}
    charts = phase7.get("charts")
    if not isinstance(charts, Mapping) or set(charts) != set(PAYLOADS):
        raise Phase8PilotError("Phase 7 chart set drift")
    for chart in PAYLOADS:
        segments = charts[chart].get("segments")
        if not isinstance(segments, list) or len(segments) != 2:
            raise Phase8PilotError(f"Phase 7 segment set drift: {chart}")
        chart_audit = []
        for index, public in enumerate(segments):
            expected_label = f"{chart}-retained-segment-{index:03d}"
            if public.get("label") != expected_label or public.get("passed") is not True:
                raise Phase8PilotError(f"Phase 7 segment identity drift: {expected_label}")
            manifest_path = (
                PHASE7_ROOT
                / "retained-private"
                / chart
                / f"{expected_label}_private_manifest.json"
            )
            if _sha256(manifest_path) != public["archive_hashes"]["private_manifest_sha256"]:
                raise Phase8PilotError(f"private manifest hash drift: {expected_label}")
            manifest = _strict_json(manifest_path)
            if manifest.get("artifact_type") != "bayesfilter_private_retained_sample_hmc_archive":
                raise Phase8PilotError(f"private manifest type drift: {expected_label}")
            sample_path, verified = _verify_manifest_files(
                manifest, public["archive_hashes"]
            )
            parsed = index == 0
            if parsed:
                tensor = tf.io.parse_tensor(
                    _absolute(sample_path).read_bytes(), out_type=tf.float64
                )
                if tuple(tensor.shape) != (256, 4, 4):
                    raise Phase8PilotError(f"retained sample shape drift: {expected_label}")
                samples[chart] = tf.transpose(
                    tensor[:PILOT_DRAWS_PER_CHAIN], [1, 0, 2]
                )
            chart_audit.append(
                {
                    "segment_index": index,
                    "manifest_sha256": public["archive_hashes"]["private_manifest_sha256"],
                    "verified_file_hashes": verified,
                    "sample_values_parsed": parsed,
                    "sample_tensor_deserialization_scope": (
                        "full_256_draw_tensor_required_by_tftensor_format"
                        if parsed
                        else "none_hash_only"
                    ),
                    "parsed_draw_indices": (
                        [0, 255] if parsed else None
                    ),
                    "selected_for_pilot_draw_indices": (
                        [0, PILOT_DRAWS_PER_CHAIN - 1] if parsed else None
                    ),
                }
            )
        audit[chart] = chart_audit
    return samples, audit


def map_pilot_to_theta(chart: str, latent: tf.Tensor) -> tuple[tf.Tensor, dict[str, Any]]:
    path, expected_hash = PAYLOADS[chart]
    if _sha256(path) != expected_hash:
        raise Phase8PilotError(f"frozen payload byte drift: {chart}")
    artifact = load_frozen_neutra_artifact(
        _strict_json(path), expected_target_signature=TARGET_SIGNATURE
    )
    if artifact.manifest.transport_hash != TRANSPORT_HASHES[chart]:
        raise Phase8PilotError(f"frozen transport hash drift: {chart}")

    @tf.function(
        input_signature=[tf.TensorSpec([4 * PILOT_DRAWS_PER_CHAIN, 4], tf.float64)],
        jit_compile=True,
        reduce_retracing=True,
    )
    def mapper(values: tf.Tensor) -> tf.Tensor:
        return artifact.transport.forward_z_to_theta_batch(values)

    flat = tf.reshape(latent, [4 * PILOT_DRAWS_PER_CHAIN, 4])
    theta = tf.reshape(mapper(flat), [4, PILOT_DRAWS_PER_CHAIN, 4])
    trace_count = _trace_count(mapper)
    if (
        trace_count != 1
        or "GPU:" not in str(theta.device)
        or not bool(tf.reduce_all(tf.math.is_finite(theta)).numpy())
    ):
        raise Phase8PilotError(f"pilot transport mapping gate failed: {chart}")
    return theta, {
        "payload_sha256": expected_hash,
        "transport_hash": artifact.manifest.transport_hash,
        "tensor_hash": artifact.manifest.tensor_hash,
        "topology_hash": artifact.manifest.topology_hash,
        "mapped_theta_sha256": hashlib.sha256(
            bytes(tf.io.serialize_tensor(theta).numpy())
        ).hexdigest(),
        "output_device": str(theta.device),
        "compile_trace_count": trace_count,
    }


def _pilot_scale(paths: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    flat = tf.reshape(paths, [-1, 10])
    center = tf.reduce_mean(flat, axis=0)
    centered = flat - center
    variance = tf.reduce_sum(tf.square(centered), axis=0) / tf.cast(
        tf.shape(flat, out_type=tf.int32)[0] - 1, tf.float64
    )
    scale = tf.sqrt(variance)
    floor = tf.sqrt(tf.constant(2.220446049250313e-16, tf.float64)) * tf.maximum(
        tf.ones([10], tf.float64), tf.abs(center)
    )
    active = scale <= floor
    if bool(tf.reduce_any(active).numpy()) or not bool(
        tf.reduce_all(tf.math.is_finite(scale)).numpy()
    ):
        raise Phase8PilotError("pilot predictive scale is degenerate or nonfinite")
    return center, scale, floor, active


def _source_bindings() -> dict[str, Any]:
    return {
        name: {"path": path.as_posix(), "sha256": _sha256(path)}
        for name, path in {
            "plan": PLAN_PATH,
            "runner": SCRIPT_PATH,
            "statistics": Path("bayesfilter/inference/predictive_equivalence.py"),
            "forecast": Path("bayesfilter/nonlinear/ssl_lstm_predictive_tf.py"),
            "transport_loader": Path("bayesfilter/inference/neutra_artifacts.py"),
        }.items()
    }


def run_pilot(*, output: Path, wall_cap_seconds: float) -> dict[str, Any]:
    if not math.isfinite(wall_cap_seconds) or wall_cap_seconds <= 0.0:
        raise Phase8PilotError("wall cap must be positive and finite")
    if PILOT_DRAWS_PER_CHAIN + CONFIRMATION_DRAWS_PER_CHAIN != TOTAL_DRAWS_PER_CHAIN:
        raise Phase8PilotError("pilot/confirmation split does not exhaust retained draws")
    if PILOT_DRAWS_PER_CHAIN % BLOCK_LENGTH or CONFIRMATION_DRAWS_PER_CHAIN % BLOCK_LENGTH:
        raise Phase8PilotError("block length must divide pilot and confirmation draws")
    started_at = _now()
    started = time.perf_counter()
    entry = validate_entry_receipts()
    latent, archive_audit = read_frozen_pilot_prefix(entry["phase7"])
    config = SSLLSTMForecastConfig()
    config.assert_evidence_config()
    forecast_rows: dict[str, Any] = {}
    path_rows: dict[str, tf.Tensor] = {}
    influence_rows: dict[str, Any] = {}
    bank_hashes: list[str] = []
    for chart in PAYLOADS:
        theta, mapping = map_pilot_to_theta(chart, latent[chart])
        flat_theta = tf.reshape(theta, [4 * PILOT_DRAWS_PER_CHAIN, 4])
        bank = make_ssl_lstm_innovation_bank(
            config,
            4 * PILOT_DRAWS_PER_CHAIN,
            tf.constant(PILOT_SEED, tf.int32),
            "independent_arm",
            ARM_IDS[chart],
        )
        bank_hashes.extend(bank.tensor_hashes().values())
        call_started = time.perf_counter()
        forecast = forecast_ssl_lstm_paths(
            flat_theta,
            bank,
            config,
            draw_chunk_size=FORECAST_DRAW_CHUNK_SIZE,
            runtime_execution_role="trusted_gpu_xla_canary",
            trust_basis="owner_designated_managed_session_visible_gpu_trusted",
        )
        elapsed = time.perf_counter() - call_started
        paths = tf.reshape(
            tf.squeeze(forecast.observations, axis=-1),
            [4, PILOT_DRAWS_PER_CHAIN, config.replication_count, 10],
        )
        if not all("GPU:" in device for device in forecast.provenance.output_devices):
            raise Phase8PilotError(f"forecast output placement failed: {chart}")
        if any(status != 0 for status in forecast.provenance.terminal_covariance_statuses):
            raise Phase8PilotError(f"terminal covariance gate failed: {chart}")
        path_rows[chart] = paths
        forecast_rows[chart] = {
            "mapping": mapping,
            "bank": {
                "role": bank.role,
                "arm_id": bank.arm_id,
                "root_seed": list(PILOT_SEED),
                "content_signature": bank.content_signature,
                "tensor_hashes": bank.tensor_hashes(),
            },
            "elapsed_seconds": elapsed,
            "draw_chunk_size": forecast.provenance.draw_chunk_size,
            "path_tensor_sha256": hashlib.sha256(
                bytes(tf.io.serialize_tensor(paths).numpy())
            ).hexdigest(),
            "output_devices": forecast.provenance.output_devices,
            "terminal_covariance_statuses": forecast.provenance.terminal_covariance_statuses,
        }
        if time.perf_counter() - started > wall_cap_seconds:
            raise Phase8PilotError("Phase 8 target pilot wall cap exceeded")
    if len(bank_hashes) != len(set(bank_hashes)):
        raise Phase8PilotError("pilot innovation tensor families overlap")

    pooled = tf.concat(tuple(path_rows[chart] for chart in PAYLOADS), axis=0)
    center, scale, floor, floor_active = _pilot_scale(pooled)
    standardized = {
        chart: (path_rows[chart] - center) / scale for chart in PAYLOADS
    }
    pooled_standardized = tf.concat(tuple(standardized.values()), axis=0)
    distance = pooled_pairwise_distance_scale(pooled_standardized)
    bandwidths = distance.median_distance * tf.constant(BANDWIDTH_FACTORS, tf.float64)
    for chart in PAYLOADS:
        influence_rows[chart] = mean_log_variance_influence(standardized[chart])
    combined_influence = tf.concat(
        (
            2.0 * influence_rows["fresh-g"].influence_values,
            -2.0 * influence_rows["fresh-h"].influence_values,
        ),
        axis=0,
    )
    covariance = chain_batch_long_run_covariance(
        combined_influence,
        block_length=BLOCK_LENGTH,
        ridge_ladder=RIDGE_LADDER,
        condition_number_max=CONDITION_NUMBER_MAX,
    )
    if not covariance.inference_admissible:
        raise Phase8PilotError("pilot long-run covariance is inadmissible")
    gpu_tensors = {
        "center": center,
        "scale": scale,
        "median_distance": distance.median_distance,
        "bandwidths": bandwidths,
        "regularized_covariance": covariance.regularized_covariance,
    }
    if any("GPU:" not in str(value.device) for value in gpu_tensors.values()):
        raise Phase8PilotError("pilot calibration output placement failed")

    terminal_program = ssl_lstm_terminal_compiled_program(config, 256)
    terminal_covariance_program = ssl_lstm_terminal_covariance_audit_compiled_program(
        256
    )
    forecast_program = ssl_lstm_forecast_compiled_program(
        config, FORECAST_DRAW_CHUNK_SIZE
    )
    trace_counts = {
        "terminal_filter": _trace_count(terminal_program),
        "terminal_covariance_audit": _trace_count(terminal_covariance_program),
        "forecast": _trace_count(forecast_program),
        "influence": _trace_count(
            predictive_equivalence._mean_log_variance_influence_xla
        ),
        "pairwise_distance": _trace_count(
            predictive_equivalence._pairwise_distance_scale_xla
        ),
        "long_run_covariance": _trace_count(
            predictive_equivalence._long_run_covariance_xla
        ),
    }
    if any(count != 1 for count in trace_counts.values()):
        raise Phase8PilotError(f"pilot compiled surface trace gate failed: {trace_counts}")

    wall_time = time.perf_counter() - started
    payload = {
        "schema": "bayesfilter.ssl_lstm_neutra.phase8_target_pilot.v1",
        "status": "PASSED",
        "decision": "PHASE8_TARGET_PILOT_PASSED_CONTROL_CALIBRATION_REQUIRED",
        "entry_bindings": {
            "phase7_receipt": {
                "path": PHASE7_RECEIPT_PATH.as_posix(),
                "sha256": PHASE7_RECEIPT_SHA256,
                "decision": entry["phase7"]["decision"],
            },
            "engineering_canary": {
                "path": CANARY_RECEIPT_PATH.as_posix(),
                "sha256": CANARY_RECEIPT_SHA256,
                "decision": entry["canary"]["decision"],
            },
            "forecast_chunk_canary": {
                "path": CHUNK_CANARY_RECEIPT_PATH.as_posix(),
                "sha256": CHUNK_CANARY_RECEIPT_SHA256,
                "decision": entry["chunk_canary"]["decision"],
            },
            "forecast_chunk_exact_prefix": {
                "path": CHUNK_PREFIX_RECEIPT_PATH.as_posix(),
                "sha256": CHUNK_PREFIX_RECEIPT_SHA256,
                "decision": entry["chunk_prefix"]["decision"],
            },
            "pairwise_distance_exact_shape_canary": {
                "path": DISTANCE_CANARY_RECEIPT_PATH.as_posix(),
                "sha256": DISTANCE_CANARY_RECEIPT_SHA256,
                "decision": entry["distance_canary"]["decision"],
            },
        },
        "split_contract": {
            "pilot_draw_indices_per_chain": [0, PILOT_DRAWS_PER_CHAIN - 1],
            "pilot_draw_count_per_chain": PILOT_DRAWS_PER_CHAIN,
            "confirmation_draw_indices_per_chain": [
                PILOT_DRAWS_PER_CHAIN,
                TOTAL_DRAWS_PER_CHAIN - 1,
            ],
            "confirmation_draw_count_per_chain": CONFIRMATION_DRAWS_PER_CHAIN,
            "segment0_full_tensor_deserialized": True,
            "segment0_confirmation_suffix_selected_or_evaluated": False,
            "segment1_tensor_deserialized": False,
            "confirmation_values_used_in_any_computation": False,
            "confirmation_forecast_bank_opened": False,
            "pilot_permanently_excluded_from_phase9": True,
            "chart_labels_pooled_for_scale_and_bandwidth": True,
            "arm_specific_predictive_summaries_emitted": False,
            "g_h_predictive_difference_computed": False,
        },
        "archive_integrity": archive_audit,
        "forecast_provenance": forecast_rows,
        "pooled_calibration": {
            "center": center,
            "predictive_standard_deviation_scale": scale,
            "scale_floor": floor,
            "scale_floor_active": floor_active,
            "distance_convention": "median_positive_off_diagonal_euclidean_distance",
            "median_path_distance": distance.median_distance,
            "positive_pair_count": distance.positive_pair_count,
            "total_pair_count": distance.total_pair_count,
            "bandwidth_factors": BANDWIDTH_FACTORS,
            "bandwidth_candidates": bandwidths,
            "block_length": BLOCK_LENGTH,
            "forecast_draw_chunk_size": FORECAST_DRAW_CHUNK_SIZE,
            "confirmation_blocks_per_chain": CONFIRMATION_DRAWS_PER_CHAIN // BLOCK_LENGTH,
            "ridge_ladder": RIDGE_LADDER,
            "condition_number_max": CONDITION_NUMBER_MAX,
            "selected_pilot_ridge_index": covariance.selected_ridge_index,
            "selected_pilot_ridge_multiplier": covariance.selected_ridge_multiplier,
            "pilot_covariance_condition_number": covariance.condition_number,
            "pilot_covariance_role": "numerical_admissibility_not_final_precision_weight",
        },
        "compile_trace_counts": trace_counts,
        "source_bindings": _source_bindings(),
        "run_manifest": {
            "command": " ".join(shlex.quote(item) for item in (sys.executable, *sys.argv)),
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
            "random_seeds": {"pilot_root": PILOT_SEED, "arm_ids": ARM_IDS},
            "started_at_utc": started_at,
            "completed_at_utc": _now(),
            "wall_time_seconds": wall_time,
            "wall_cap_seconds": wall_cap_seconds,
            "output_path": output.as_posix(),
            "plan_path": PLAN_PATH.as_posix(),
            "result_path": RESULT_PATH.as_posix(),
        },
        "nonclaims": [
            "calibration-only excluded pilot; no Phase 9 confirmation opened",
            "no G/H predictive comparison, equivalence decision, or ranking",
            "no posterior correctness, truth, mode coverage, or model adequacy claim",
            "pilot covariance is not a frozen confirmatory precision weight",
            "bandwidth candidates still require controlled null/power calibration",
        ],
    }
    _write_json(output, payload)
    return payload


def _require_gpu() -> None:
    physical = tf.config.list_physical_devices("GPU")
    if not physical:
        raise Phase8PilotError("Phase 8 target pilot requires a visible trusted GPU")
    for gpu in physical:
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.experimental.enable_tensor_float_32_execution(True)
    if not tf.config.list_logical_devices("GPU"):
        raise Phase8PilotError("Phase 8 target pilot requires a logical GPU")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wall-cap-seconds", type=float, required=True)
    args = parser.parse_args(argv)
    _require_gpu()
    with tf.device("/GPU:0"):
        payload = run_pilot(
            output=args.output, wall_cap_seconds=float(args.wall_cap_seconds)
        )
    print(
        "JSON_SUMMARY "
        + json.dumps(
            {
                "decision": payload["decision"],
                "wall_time_seconds": payload["run_manifest"]["wall_time_seconds"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
