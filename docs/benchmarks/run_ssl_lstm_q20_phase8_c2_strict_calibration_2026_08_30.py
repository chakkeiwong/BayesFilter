#!/usr/bin/env python3
"""Run the bounded Phase 8 C2 strict-backend calibration screen.

The screen compares four target-specific capacity/learning-rate hypotheses on
two independent initialization roots.  It is a calibration experiment only:
it cannot establish whitening, mode discovery, posterior correctness, HMC
readiness, or a repository default.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
import traceback
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "docs/plans/bayesfilter-ssl-lstm-q20-phase8-c2-strict-calibration-subplan-2026-08-30.md"
MASTER_PLAN = ROOT / "docs/plans/bayesfilter-ssl-lstm-q20-tempered-rkl-transport-ensemble-implementation-plan-2026-08-28.md"
COMPATIBILITY_ROOT = ROOT / "docs/plans/artifacts/ssl-lstm-q20-tempered-rkl-transport-ensemble-2026-08-29/phase8-calibration/attempt-00-compatibility"
MAP_ARTIFACT = ROOT / "docs/plans/artifacts/ssl-lstm-q20-seed-b-posterior-reference-2026-08-07/r3/map-progress.json"
EXPECTED_TARGET_SIGNATURE = "9a86e60081f1b9cd288dbdb1dcbe1e9a5b5e23d9b5ef97afdb72ee95c23d7278"
COMPATIBILITY_FILES = (
    "complexity-target-junit.xml",
    "predictive-junit.xml",
    "focused-route-junit.xml",
)
BATCH_SIZE = 32
TRAIN_UPDATE_COUNT = 32
VALIDATION_SIZE = 256
STRESS_SIZE = 64
PILOT_MEMORY_CAP_BYTES = 4 * 1024**3
MATERIAL_CAP_SECONDS = 3600.0
SCHEMA = "bayesfilter.ssl_lstm_q20.tempered_rkl_phase8_c2_strict_calibration.v1"
DEFAULT_GPU_ID = "0"
PRINCIPAL_SQRT_BACKEND = "tensorflow_eigh_strict"
ARCHITECTURE_GRID = (
    {
        "name": "compact-high",
        "hidden_layers": (16, 16),
        "activation": "tanh",
        "learning_rate": 1.0e-3,
    },
    {
        "name": "compact-low",
        "hidden_layers": (16, 16),
        "activation": "tanh",
        "learning_rate": 5.0e-4,
    },
    {
        "name": "wide-high",
        "hidden_layers": (32, 32),
        "activation": "tanh",
        "learning_rate": 1.0e-3,
    },
    {
        "name": "wide-low",
        "hidden_layers": (32, 32),
        "activation": "tanh",
        "learning_rate": 5.0e-4,
    },
)
INITIALIZATION_ROOTS = ((20260830, 12001), (20260830, 12002))
TRAINING_ROOT = (20260830, 22001)
VALIDATION_ROOTS = ((20260830, 42001), (20260830, 42002))
STRESS_ROOT = (20260830, 52001)

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class Phase8PilotError(RuntimeError):
    """Raised when the frozen cost-pilot contract is violated."""


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise Phase8PilotError(f"refusing to overwrite artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _json_safe(value: Any, tf: Any) -> Any:
    if tf.is_tensor(value):
        return _json_safe(value.numpy(), tf)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item, tf) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item, tf) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist(), tf)
    if hasattr(value, "item"):
        return _json_safe(value.item(), tf)
    return str(value)


def _git(command: Sequence[str]) -> str:
    try:
        return subprocess.check_output(
            tuple(command), cwd=ROOT, text=True, stderr=subprocess.STDOUT
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"unavailable:{type(exc).__name__}"


def _nvidia_snapshot() -> Mapping[str, Any]:
    command = (
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    )
    try:
        output = subprocess.check_output(command, text=True, stderr=subprocess.STDOUT)
        return {"command": list(command), "rows": output.strip().splitlines()}
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"command": list(command), "error": type(exc).__name__}


def _static_scan(paths: Sequence[Path]) -> Mapping[str, Any]:
    forbidden = (
        "tf.map_fn",
        "tf.vectorized_map",
        "GradientTape.jacobian",
        "GradientTape.batch_jacobian",
        "pfor",
    )
    hits: dict[str, list[str]] = {item: [] for item in forbidden}
    for path in paths:
        source = path.read_text(encoding="utf-8")
        for token in forbidden:
            if token in source:
                hits[token].append(str(path.relative_to(ROOT)))
    return {
        "paths": [str(path.relative_to(ROOT)) for path in paths],
        "forbidden_tokens": list(forbidden),
        "hits": hits,
        "passed": not any(hits.values()),
    }


def _compatibility_receipts() -> Mapping[str, Any]:
    rows = []
    for name in COMPATIBILITY_FILES:
        path = COMPATIBILITY_ROOT / name
        if not path.is_file():
            raise Phase8PilotError(f"missing compatibility receipt: {path}")
        root = ET.parse(path).getroot()
        suites = tuple(root) if root.tag == "testsuites" else (root,)
        if not suites or any(suite.tag != "testsuite" for suite in suites):
            raise Phase8PilotError(f"unrecognized JUnit structure: {path}")
        tests = sum(int(suite.attrib.get("tests", 0)) for suite in suites)
        failures = sum(int(suite.attrib.get("failures", 0)) for suite in suites)
        errors = sum(int(suite.attrib.get("errors", 0)) for suite in suites)
        skipped = sum(int(suite.attrib.get("skipped", 0)) for suite in suites)
        elapsed = sum(float(suite.attrib.get("time", 0.0)) for suite in suites)
        if tests <= 0 or failures or errors:
            raise Phase8PilotError(f"compatibility receipt did not pass: {path}")
        rows.append(
            {
                "path": str(path.relative_to(ROOT)),
                "sha256": _sha256(path),
                "tests": tests,
                "failures": failures,
                "errors": errors,
                "skipped": skipped,
                "time_seconds": elapsed,
            }
        )
    return {"passed": True, "receipts": rows}


def _map_representatives(tf: Any, target_signature: str) -> tuple[Any, Mapping[str, Any]]:
    if target_signature != EXPECTED_TARGET_SIGNATURE:
        raise Phase8PilotError(
            "MAP representatives are stale for the current target signature"
        )
    payload = json.loads(MAP_ARTIFACT.read_text(encoding="utf-8"))
    starts = payload.get("starts")
    if not isinstance(starts, list):
        raise Phase8PilotError("MAP artifact does not contain starts")
    eligible: dict[str, list[Mapping[str, Any]]] = {"plus": [], "minus": []}
    for row in starts:
        if not isinstance(row, Mapping):
            continue
        try:
            position = [float(item) for item in row.get("position", ())]
            score = float(row.get("score_inf_norm"))
            log_prob = float(row.get("log_prob"))
        except (TypeError, ValueError):
            continue
        if (
            len(position) != 4
            or not all(math.isfinite(item) for item in position)
            or not math.isfinite(score)
            or score > 1.0e-5
            or not math.isfinite(log_prob)
            or position[2] == 0.0
        ):
            continue
        label = "plus" if position[2] > 0.0 else "minus"
        eligible[label].append(
            {
                "position": position,
                "score_inf_norm": score,
                "log_prob": log_prob,
                "start_index": int(row.get("start_index", -1)),
            }
        )
    if not eligible["plus"] or not eligible["minus"]:
        raise Phase8PilotError("MAP artifact lacks both declared sign regions")
    selected = {
        label: max(rows, key=lambda item: float(item["log_prob"]))
        for label, rows in eligible.items()
    }
    points = tf.stack(
        tuple(tf.convert_to_tensor(selected[label]["position"], tf.float64) for label in ("plus", "minus")),
        axis=0,
    )
    return points, {
        "path": str(MAP_ARTIFACT.relative_to(ROOT)),
        "sha256": _sha256(MAP_ARTIFACT),
        "bound_target_signature": target_signature,
        "selection": "highest_log_prob_finite_stationary_row_within_each_sign",
        "selected": selected,
    }


def _seed(tf: Any, root: tuple[int, int], *folds: int) -> tuple[int, int]:
    value = tf.constant(root, tf.int32)
    for fold in folds:
        value = tf.random.experimental.stateless_fold_in(value, int(fold))
    return tuple(int(item) for item in value.numpy())


def _memory_info(tf: Any, device_name: str) -> Mapping[str, Any]:
    try:
        return dict(tf.config.experimental.get_memory_info(device_name))
    except (AttributeError, RuntimeError, ValueError) as exc:
        return {"unavailable": type(exc).__name__}


def _reset_memory(tf: Any, device_name: str) -> None:
    try:
        tf.config.experimental.reset_memory_stats(device_name)
    except (AttributeError, RuntimeError, ValueError) as exc:
        raise Phase8PilotError("could not reset GPU allocator telemetry") from exc


def _checkpoint_replay_error(tf: Any, original: Any, restored: Any, latent: Any) -> Mapping[str, Any]:
    original_physical, original_logdet = original.forward_and_logdet(latent)
    restored_physical, restored_logdet = restored.forward_and_logdet(latent)
    recovered, recovered_logdet = restored.inverse_and_forward_logdet(original_physical)
    return {
        "forward_max_abs": tf.reduce_max(tf.abs(restored_physical - original_physical)),
        "forward_logdet_max_abs": tf.reduce_max(tf.abs(restored_logdet - original_logdet)),
        "roundtrip_max_abs": tf.reduce_max(tf.abs(recovered - latent)),
        "inverse_logdet_max_abs": tf.reduce_max(tf.abs(recovered_logdet - original_logdet)),
    }


def _diagnostic_payload(diagnostic: Any, tf: Any) -> Mapping[str, Any]:
    return {
        "finite": diagnostic.finite,
        "valid_row_count": diagnostic.valid_row_count,
        "batch_size": diagnostic.batch_size,
        "reverse_kl_mean": tf.reduce_mean(diagnostic.reverse_kl_per_sample),
        "centered_log_density_rms": diagnostic.centered_log_density_rms,
        "centered_log_density_median_abs": diagnostic.centered_log_density_median_abs,
        "centered_log_density_q90_abs": diagnostic.centered_log_density_q90_abs,
        "pullback_score_rms_per_coordinate": diagnostic.pullback_score_rms_per_coordinate,
        "pullback_score_maximum_row_norm": diagnostic.pullback_score_maximum_row_norm,
    }


def _checkpoint_scope(
    *,
    target_signature: str,
    principal_sqrt_backend: str,
    architecture_name: str,
    root_index: int,
    initialization_root: tuple[int, int],
    training_root: tuple[int, int],
    validation_root: tuple[int, int],
    stress_root: tuple[int, int],
    beta: float,
    validation_size: int,
) -> Mapping[str, Any]:
    beta_label = str(float(beta)).replace(".", "p")
    validation_ids = (
        [
            f"phase8-c2-endpoint-{architecture_name}-r{root_index}"
            f"-beta{beta_label}-n{int(validation_size)}"
        ]
        if float(beta) == 0.0
        else [
            f"phase8-c2-validation-{architecture_name}-r{root_index}"
            f"-beta{beta_label}-n{int(validation_size)}",
            f"phase8-c2-stress-{architecture_name}-r{root_index}"
            f"-beta{beta_label}-n{int(STRESS_SIZE)}",
        ]
    )
    return {
        "data_identity": f"ssl-lstm-q20:{target_signature}",
        "dtype": "float64",
        "backend": "tensorflow_tfp_gpu",
        "jit_compile": True,
        "principal_sqrt_backend": str(principal_sqrt_backend),
        "tf32_execution_enabled": True,
        "training_seed_derivation": {
            "initialization_root": list(initialization_root),
            "training_root": list(training_root),
            "validation_root": list(validation_root),
            "stress_root": list(stress_root),
            "folds": {
                "architecture_name": str(architecture_name),
                "root_index": int(root_index),
                "beta": float(beta),
            },
        },
        "validation_bank_ids": validation_ids,
    }


def _protocol(
    *,
    validation_size: int = VALIDATION_SIZE,
) -> Mapping[str, Any]:
    return {
        "schema": SCHEMA,
        "mode": "c2-strict-calibration",
        "principal_sqrt_backend": PRINCIPAL_SQRT_BACKEND,
        "batch_size": BATCH_SIZE,
        "architectures": [
            {
                "name": row["name"],
                "hidden_layers": list(row["hidden_layers"]),
                "activation": row["activation"],
                "learning_rate": row["learning_rate"],
            }
            for row in ARCHITECTURE_GRID
        ],
        "root_count": len(INITIALIZATION_ROOTS),
        "initialization_roots": [list(root) for root in INITIALIZATION_ROOTS],
        "training_root": list(TRAINING_ROOT),
        "validation_roots": [list(root) for root in VALIDATION_ROOTS],
        "stress_root": list(STRESS_ROOT),
        "betas": [0.0, 0.5],
        "updates_per_row": TRAIN_UPDATE_COUNT,
        "validation_size": int(validation_size),
        "stress_size": STRESS_SIZE,
        "memory_cap_bytes": PILOT_MEMORY_CAP_BYTES,
        "role": "target_specific_calibration_nomination_only",
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--print-protocol", action="store_true")
    parser.add_argument("--validation-size", type=int, choices=(VALIDATION_SIZE,), default=VALIDATION_SIZE)
    return parser.parse_args()


def _prepare_default_gpu_environment() -> Mapping[str, Any]:
    """Install the repository GPU defaults before importing TensorFlow.

    The Codex service may still impose an execution permission boundary, but
    that boundary is not part of the scientific runner.  This process always
    chooses one GPU and on-demand allocation unless the caller explicitly
    supplies a different valid value.
    """

    visible_before = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_before is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get(
            "BAYESFILTER_GPU_ID", DEFAULT_GPU_ID
        )
    growth_before = os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH")
    if growth_before is None:
        os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    return {
        "cuda_visible_devices_before": visible_before,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "cuda_visible_devices_defaulted": visible_before is None,
        "tf_force_gpu_allow_growth_before": growth_before,
        "tf_force_gpu_allow_growth": os.environ.get(
            "TF_FORCE_GPU_ALLOW_GROWTH", ""
        ),
        "memory_growth_defaulted": growth_before is None,
        "selection_policy": "repository_default_single_gpu_no_idle_probe",
    }


def _localization_stage_start(output_dir: Path, label: str, started: float) -> None:
    _write_json(
        output_dir / f"localization-{label}-start.json",
        {
            "schema": "bayesfilter.ssl_lstm_q20.tempered_rkl_phase8_localization_stage.v1",
            "status": "STARTED",
            "stage": label,
            "monotonic_started_seconds": started,
            "wall_started_unix_seconds": time.time(),
        },
    )


def _localization_stage_done(
    output_dir: Path,
    label: str,
    started: float,
    *,
    tf: Any,
    **payload: Any,
) -> Mapping[str, Any]:
    result = {
        "schema": "bayesfilter.ssl_lstm_q20.tempered_rkl_phase8_localization_stage.v1",
        "status": "COMPLETED",
        "stage": label,
        "elapsed_seconds": time.monotonic() - started,
        "wall_completed_unix_seconds": time.time(),
        **_json_safe(payload, tf),
    }
    _write_json(output_dir / f"localization-{label}-done.json", result)
    return result


def _run_target_localization(
    tf: Any,
    bridge: Any,
    *,
    output_dir: Path,
    center: Any,
    dimension: int,
    principal_sqrt_backend: str,
) -> Mapping[str, Any]:
    """Localize the first expensive q20 operation without training a candidate.

    A start marker is written before each call.  If an external timeout kills
    the process, the last start marker identifies the operation that was still
    running; no partial result is interpreted as a scientific failure.
    """

    from bayesfilter.inference.neutra_weighted_training import (
        WeightedDenseIAFTransport,
        WeightedNeuTraConfig,
    )
    from bayesfilter.inference.tempered_transport_ensemble_tf import (
        IndependentTemperedReverseKLTrainer,
        PreparedTransportInitialization,
        prepare_transport_initialization,
        pullback_gaussianization_diagnostic,
    )

    summaries: list[Mapping[str, Any]] = []

    def target_stage(label: str, latent: Any, beta: float) -> None:
        started = time.monotonic()
        _localization_stage_start(output_dir, label, started)
        value, score, status = bridge.value_score_status(
            latent, tf.constant(beta, tf.float64)
        )
        valid = tf.convert_to_tensor(status["bridge_valid"], tf.bool)
        summary = _localization_stage_done(
            output_dir,
            label,
            started,
            tf=tf,
            beta=float(beta),
            batch_size=int(latent.shape[0]),
            valid_rows=int(tf.reduce_sum(tf.cast(valid, tf.int32)).numpy()),
            finite_value=bool(tf.reduce_all(tf.math.is_finite(value)).numpy()),
            finite_score=bool(tf.reduce_all(tf.math.is_finite(score)).numpy()),
        )
        summaries.append(summary)

    latent8 = tf.random.stateless_normal(
        [8, dimension], tf.constant((20260830, 91001), tf.int32), dtype=tf.float64
    )
    latent256 = tf.random.stateless_normal(
        [256, dimension], tf.constant((20260830, 91002), tf.int32), dtype=tf.float64
    )
    target_stage("target-b8-beta0", latent8, 0.0)
    target_stage("target-b8-beta05", latent8, 0.5)
    target_stage("target-b256-beta0", latent256, 0.0)
    target_stage("target-b256-beta05", latent256, 0.5)

    config = WeightedNeuTraConfig(
        dimension=dimension,
        hidden_layers=(16, 16),
        stages=2,
        activation="tanh",
        initialization_scale=0.02,
        initialization_seed=(20260830, 92001),
        learning_rate=1.0e-3,
        jit_compile=True,
    )
    raw = WeightedDenseIAFTransport(config)
    started = time.monotonic()
    _localization_stage_start(output_dir, "transport-beta0-preflight", started)
    beta0 = prepare_transport_initialization(
        raw,
        bridge,
        component_id="localization-chart-0",
        seed=(20260830, 92002),
        batch_size=8,
        repair_scales=(1.0, 0.5, 0.25),
        beta=0.0,
        reference_center=center,
        reference_scale=math.sqrt(float(bridge.prior_variance)),
    )
    summaries.append(
        _localization_stage_done(
            output_dir,
            "transport-beta0-preflight",
            started,
            tf=tf,
            valid=bool(beta0.receipt.valid),
            repair_index=int(beta0.receipt.repair_index),
            finite_rows=int(beta0.receipt.finite_rows),
            transport_state_hash=beta0.receipt.transport_state_hash,
        )
    )

    started = time.monotonic()
    _localization_stage_start(output_dir, "transport-beta05-preflight", started)
    beta05 = prepare_transport_initialization(
        beta0.transport,
        bridge,
        component_id="localization-chart-0",
        seed=(20260830, 92003),
        batch_size=8,
        repair_scales=(1.0,),
        beta=0.5,
    )
    summaries.append(
        _localization_stage_done(
            output_dir,
            "transport-beta05-preflight",
            started,
            tf=tf,
            valid=bool(beta05.receipt.valid),
            repair_index=int(beta05.receipt.repair_index),
            finite_rows=int(beta05.receipt.finite_rows),
            transport_state_hash=beta05.receipt.transport_state_hash,
        )
    )

    validation_latent = tf.random.stateless_normal(
        [256, dimension], tf.constant((20260830, 92004), tf.int32), dtype=tf.float64
    )
    started = time.monotonic()
    _localization_stage_start(output_dir, "transport-beta05-diagnostic", started)
    diagnostic = pullback_gaussianization_diagnostic(
        beta05.transport, bridge, beta=0.5, latent=validation_latent
    )
    summaries.append(
        _localization_stage_done(
            output_dir,
            "transport-beta05-diagnostic",
            started,
            tf=tf,
            finite=bool(diagnostic.finite.numpy()),
            valid_rows=int(diagnostic.valid_row_count.numpy()),
            centered_log_density_rms=float(diagnostic.centered_log_density_rms.numpy()),
            pullback_score_rms_per_coordinate=[
                float(value)
                for value in diagnostic.pullback_score_rms_per_coordinate.numpy()
            ],
        )
    )

    started = time.monotonic()
    _localization_stage_start(output_dir, "transport-beta05-first-update", started)
    trainer = IndependentTemperedReverseKLTrainer(
        config,
        bridge,
        beta=0.5,
        component_id="localization-chart-0",
        batch_size=8,
        prepared_initialization=PreparedTransportInitialization(
            beta05.transport, beta05.receipt
        ),
    )
    update = trainer.train_step((20260830, 92005))
    summaries.append(
        _localization_stage_done(
            output_dir,
            "transport-beta05-first-update",
            started,
            tf=tf,
            valid=bool(update.valid.numpy()),
            loss=float(update.loss.numpy()),
            gradient_norm=float(update.gradient_norm.numpy()),
            target_call_count=int(update.target_call_count.numpy()),
        )
    )
    return {
        "status": "PASS_PHASE8_TARGET_LOCALIZATION",
        "mode": "target-localization",
        "principal_sqrt_backend": str(principal_sqrt_backend),
        "stages": summaries,
        "nonclaims": [
            "diagnostic-only operation timing",
            "no transport candidate selection",
            "no whitening, mode-discovery, posterior, HMC, or scaling claim",
        ],
    }


def _stage_marker(
    output_dir: Path, label: str, status: str, **payload: Any
) -> None:
    """Leave a unique marker so an external timeout is classifiable."""
    safe_label = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in str(label)
    )
    _write_json(
        output_dir / "stages" / f"{safe_label}-{status}.json",
        {
            "schema": f"{SCHEMA}.stage_marker.v1",
            "stage": str(label),
            "status": str(status),
            "wall_unix_seconds": time.time(),
            **payload,
        },
    )


def _budget_guard(started: float, *, reserve_seconds: float = 120.0) -> None:
    if time.monotonic() - started + float(reserve_seconds) >= MATERIAL_CAP_SECONDS:
        raise Phase8PilotError("C2 material cap exhausted before the next stage")


def _run_c2_row(
    tf: Any,
    bridge: Any,
    *,
    center: Any,
    dimension: int,
    architecture: Mapping[str, Any],
    architecture_index: int,
    root_index: int,
    output_dir: Path,
    device_name: str,
    started: float,
) -> Mapping[str, Any]:
    """Run one fixed architecture/root row and return its chart and evidence."""
    from bayesfilter.inference.neutra_weighted_training import (
        WeightedDenseIAFTransport,
        WeightedNeuTraConfig,
    )
    from bayesfilter.inference.tempered_transport_ensemble_tf import (
        IndependentTemperedReverseKLTrainer,
        PreparedTransportInitialization,
        capture_trainable_transport_checkpoint,
        paired_reverse_kl_improvement,
        prepare_transport_initialization,
        pullback_gaussianization_diagnostic,
        restore_trainable_transport_checkpoint,
    )
    from bayesfilter.inference.tempered_transitions_tf import (
        derive_reliability_tolerance,
    )

    architecture_name = str(architecture["name"])
    component_id = f"c2-{architecture_name}-root-{int(root_index)}"
    row_dir = output_dir / "rows" / component_id
    row_dir.mkdir(parents=True, exist_ok=True)
    _stage_marker(
        output_dir,
        component_id,
        "start",
        architecture=architecture_name,
        architecture_index=int(architecture_index),
        root_index=int(root_index),
    )
    _budget_guard(started)
    _reset_memory(tf, device_name)

    initialization_root = INITIALIZATION_ROOTS[root_index]
    validation_root = VALIDATION_ROOTS[root_index]
    initialization_seed = _seed(tf, initialization_root, architecture_index, 1)
    preflight_seed = _seed(tf, initialization_root, architecture_index, 2)
    beta05_seed = _seed(tf, initialization_root, architecture_index, 3)
    validation_seed = _seed(tf, validation_root, architecture_index, 1)
    stress_seed = _seed(tf, STRESS_ROOT, architecture_index, root_index)
    training_seed = _seed(tf, TRAINING_ROOT, architecture_index, root_index)
    config = WeightedNeuTraConfig(
        dimension=dimension,
        hidden_layers=tuple(int(value) for value in architecture["hidden_layers"]),
        stages=2,
        activation=str(architecture["activation"]),
        initialization_scale=0.02,
        initialization_seed=initialization_seed,
        learning_rate=float(architecture["learning_rate"]),
        jit_compile=True,
    )
    raw = WeightedDenseIAFTransport(config)
    endpoint_latent = tf.random.stateless_normal(
        [VALIDATION_SIZE, dimension],
        tf.constant(_seed(tf, initialization_root, architecture_index, 10), tf.int32),
        dtype=tf.float64,
    )
    beta0_prepared = prepare_transport_initialization(
        raw,
        bridge,
        component_id=component_id,
        seed=preflight_seed,
        batch_size=BATCH_SIZE,
        repair_scales=(1.0, 0.5, 0.25),
        beta=0.0,
        reference_center=center,
        reference_scale=math.sqrt(float(bridge.prior_variance)),
    )
    if not beta0_prepared.receipt.valid:
        raise Phase8PilotError(f"beta=0 preflight failed for {component_id}")
    endpoint_diagnostic = pullback_gaussianization_diagnostic(
        beta0_prepared.transport, bridge, beta=0.0, latent=endpoint_latent
    )
    endpoint_tolerance = derive_reliability_tolerance(dimension=dimension)
    if (
        not bool(endpoint_diagnostic.finite.numpy())
        or int(endpoint_diagnostic.valid_row_count.numpy()) != VALIDATION_SIZE
        or float(endpoint_diagnostic.centered_log_density_rms.numpy())
        > endpoint_tolerance
        or float(endpoint_diagnostic.pullback_score_maximum_row_norm.numpy())
        > endpoint_tolerance
    ):
        raise Phase8PilotError(f"beta=0 reference-affine check failed for {component_id}")

    beta0_scope = _checkpoint_scope(
        target_signature=str(bridge.target_signature),
        principal_sqrt_backend=PRINCIPAL_SQRT_BACKEND,
        architecture_name=architecture_name,
        root_index=root_index,
        initialization_root=initialization_root,
        training_root=TRAINING_ROOT,
        validation_root=validation_root,
        stress_root=STRESS_ROOT,
        beta=0.0,
        validation_size=VALIDATION_SIZE,
    )
    beta0_checkpoint = capture_trainable_transport_checkpoint(
        beta0_prepared.transport,
        component_id=component_id,
        beta=0.0,
        bridge_signature=str(bridge.signature),
        target_signature=str(bridge.target_signature),
        parent_checkpoint_hash=None,
        update_count=0,
        checkpoint_scope=beta0_scope,
    )
    _write_json(row_dir / "beta0-checkpoint.json", _json_safe(beta0_checkpoint, tf))
    beta0_restored = restore_trainable_transport_checkpoint(
        beta0_checkpoint,
        expected_context={
            "component_id": component_id,
            "beta": 0.0,
            "bridge_signature": str(bridge.signature),
            "target_signature": str(bridge.target_signature),
            "checkpoint_scope": beta0_checkpoint["checkpoint_scope"],
        },
    )
    beta0_replay = _checkpoint_replay_error(
        tf, beta0_prepared.transport, beta0_restored, endpoint_latent
    )
    if any(float(value.numpy()) > endpoint_tolerance for value in beta0_replay.values()):
        raise Phase8PilotError(f"beta=0 checkpoint replay failed for {component_id}")

    beta05_prepared = prepare_transport_initialization(
        beta0_restored,
        bridge,
        component_id=component_id,
        seed=beta05_seed,
        batch_size=BATCH_SIZE,
        repair_scales=(1.0,),
        beta=0.5,
    )
    if not beta05_prepared.receipt.valid:
        raise Phase8PilotError(f"beta=0.5 preflight failed for {component_id}")
    validation_latent = tf.random.stateless_normal(
        [VALIDATION_SIZE, dimension],
        tf.constant(validation_seed, tf.int32),
        dtype=tf.float64,
    )
    start_diagnostic = pullback_gaussianization_diagnostic(
        beta05_prepared.transport, bridge, beta=0.5, latent=validation_latent
    )
    start_scope = _checkpoint_scope(
        target_signature=str(bridge.target_signature),
        principal_sqrt_backend=PRINCIPAL_SQRT_BACKEND,
        architecture_name=architecture_name,
        root_index=root_index,
        initialization_root=initialization_root,
        training_root=TRAINING_ROOT,
        validation_root=validation_root,
        stress_root=STRESS_ROOT,
        beta=0.5,
        validation_size=VALIDATION_SIZE,
    )
    start_checkpoint = capture_trainable_transport_checkpoint(
        beta05_prepared.transport,
        component_id=component_id,
        beta=0.5,
        bridge_signature=str(bridge.signature),
        target_signature=str(bridge.target_signature),
        parent_checkpoint_hash=str(beta0_checkpoint["checkpoint_hash"]),
        update_count=0,
        checkpoint_scope=start_scope,
    )
    _write_json(row_dir / "beta05-start-checkpoint.json", _json_safe(start_checkpoint, tf))
    start_restored = restore_trainable_transport_checkpoint(
        start_checkpoint,
        expected_context={
            "component_id": component_id,
            "beta": 0.5,
            "bridge_signature": str(bridge.signature),
            "target_signature": str(bridge.target_signature),
            "checkpoint_scope": start_checkpoint["checkpoint_scope"],
        },
    )
    start_replay = _checkpoint_replay_error(
        tf, beta05_prepared.transport, start_restored, validation_latent
    )
    if any(float(value.numpy()) > endpoint_tolerance for value in start_replay.values()):
        raise Phase8PilotError(f"beta=0.5 start checkpoint replay failed for {component_id}")

    trainer = IndependentTemperedReverseKLTrainer(
        config,
        bridge,
        beta=0.5,
        component_id=component_id,
        batch_size=BATCH_SIZE,
        prepared_initialization=PreparedTransportInitialization(
            beta05_prepared.transport, beta05_prepared.receipt
        ),
    )
    updates = []
    update_times = []
    for update_index in range(TRAIN_UPDATE_COUNT):
        _budget_guard(started)
        update_started = time.monotonic()
        update = trainer.train_step(
            _seed(tf, training_seed, architecture_index, root_index, update_index)
        )
        elapsed = time.monotonic() - update_started
        update_times.append(elapsed)
        if not bool(update.valid.numpy()):
            raise Phase8PilotError(f"invalid optimizer update for {component_id}")
        updates.append(
            {
                "update": update_index + 1,
                "elapsed_seconds": elapsed,
                "loss": update.loss,
                "gradient_norm": update.gradient_norm,
                "clipped_gradient_norm": update.clipped_gradient_norm,
                "clipping_applied": update.clipping_applied,
                "step": update.step,
                "target_call_count": update.target_call_count,
                "cross_density_work": update.cross_density_work,
                "valid": update.valid,
            }
        )

    final_diagnostic = pullback_gaussianization_diagnostic(
        trainer.transport, bridge, beta=0.5, latent=validation_latent
    )
    improvement = paired_reverse_kl_improvement(start_diagnostic, final_diagnostic)
    final_checkpoint = capture_trainable_transport_checkpoint(
        trainer.transport,
        component_id=component_id,
        beta=0.5,
        bridge_signature=str(bridge.signature),
        target_signature=str(bridge.target_signature),
        parent_checkpoint_hash=str(start_checkpoint["checkpoint_hash"]),
        update_count=TRAIN_UPDATE_COUNT,
        checkpoint_scope=start_checkpoint["checkpoint_scope"],
    )
    _write_json(row_dir / "beta05-final-checkpoint.json", _json_safe(final_checkpoint, tf))
    final_restored = restore_trainable_transport_checkpoint(
        final_checkpoint,
        expected_context={
            "component_id": component_id,
            "beta": 0.5,
            "bridge_signature": str(bridge.signature),
            "target_signature": str(bridge.target_signature),
            "checkpoint_scope": final_checkpoint["checkpoint_scope"],
        },
    )
    final_replay = _checkpoint_replay_error(
        tf, trainer.transport, final_restored, validation_latent
    )
    if any(float(value.numpy()) > endpoint_tolerance for value in final_replay.values()):
        raise Phase8PilotError(f"final checkpoint replay failed for {component_id}")
    stress_latent = tf.random.stateless_normal(
        [STRESS_SIZE, dimension], tf.constant(stress_seed, tf.int32), dtype=tf.float64
    )
    stress_physical, _ = final_restored.forward_and_logdet(stress_latent)
    allocator = _memory_info(tf, device_name)
    peak = int(allocator.get("peak", PILOT_MEMORY_CAP_BYTES + 1))
    if peak > PILOT_MEMORY_CAP_BYTES:
        raise Phase8PilotError(f"allocator cap exceeded for {component_id}")
    row = {
        "status": "PASS_C2_ROW",
        "component_id": component_id,
        "architecture": {
            "name": architecture_name,
            "hidden_layers": list(config.hidden_layers),
            "activation": config.activation,
            "learning_rate": float(config.learning_rate),
            "stages": int(config.stages),
        },
        "architecture_index": int(architecture_index),
        "root_index": int(root_index),
        "batch_size": BATCH_SIZE,
        "training_updates": TRAIN_UPDATE_COUNT,
        "seeds": {
            "initialization_root": list(initialization_root),
            "initialization_seed": list(initialization_seed),
            "preflight_seed": list(preflight_seed),
            "beta05_seed": list(beta05_seed),
            "training_root": list(TRAINING_ROOT),
            "training_seed": list(training_seed),
            "validation_root": list(validation_root),
            "validation_seed": list(validation_seed),
            "stress_root": list(STRESS_ROOT),
            "stress_seed": list(stress_seed),
        },
        "beta0_preflight": beta0_prepared.receipt.payload(),
        "beta05_preflight": beta05_prepared.receipt.payload(),
        "endpoint_diagnostic": _diagnostic_payload(endpoint_diagnostic, tf),
        "start_diagnostic": _diagnostic_payload(start_diagnostic, tf),
        "final_diagnostic": _diagnostic_payload(final_diagnostic, tf),
        "paired_improvement": improvement,
        "beta0_checkpoint_hash": beta0_checkpoint["checkpoint_hash"],
        "start_checkpoint_hash": start_checkpoint["checkpoint_hash"],
        "final_checkpoint_hash": final_checkpoint["checkpoint_hash"],
        "beta0_replay": beta0_replay,
        "start_replay": start_replay,
        "final_replay": final_replay,
        "updates": updates,
        "median_update_seconds": statistics.median(update_times),
        "first_update_seconds": update_times[0],
        "allocator": allocator,
        "stress_positive_sign_fraction": tf.reduce_mean(
            tf.cast(stress_physical[:, 2] > 0.0, tf.float64)
        ),
        "target_signature": str(bridge.target_signature),
        "bridge_signature": str(bridge.signature),
        "principal_sqrt_backend": PRINCIPAL_SQRT_BACKEND,
        "jit_compile": True,
    }
    safe_row = _json_safe(row, tf)
    safe_row["row_hash"] = _stable_hash(safe_row)
    _write_json(row_dir / "row-result.json", safe_row)
    _stage_marker(output_dir, component_id, "done", row_status="PASS_C2_ROW")
    return {
        "record": safe_row,
        "transport": final_restored,
        "validation_latent": validation_latent,
        "stress_physical": stress_physical,
    }


def main() -> int:
    args = _parse_args()
    if args.print_protocol:
        print(json.dumps(_protocol(validation_size=args.validation_size), sort_keys=True, indent=2))
        return 0
    if args.output_dir is None:
        raise Phase8PilotError("--output-dir is required")
    gpu_environment = _prepare_default_gpu_environment()
    if not _truthy(os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH")):
        raise Phase8PilotError("C2 requires TF_FORCE_GPU_ALLOW_GROWTH=true before import")
    if os.environ.get("CUDA_VISIBLE_DEVICES", "").strip() in {"", "-1"}:
        raise Phase8PilotError("C2 requires one explicitly visible GPU")
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise Phase8PilotError(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    started = time.monotonic()

    import tensorflow as tf

    # Configure growth before any BayesFilter import or logical-device query.
    from bayesfilter.runtime.gpu_memory_policy import (
        configure_tensorflow_gpu_memory_growth,
    )

    memory_policy = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    tf.config.experimental.enable_tensor_float_32_execution(True)
    tf.config.set_soft_device_placement(False)
    logical_gpus = tuple(tf.config.list_logical_devices("GPU"))
    if len(logical_gpus) != 1:
        raise Phase8PilotError("C2 requires exactly one visible logical GPU")
    device_name = str(logical_gpus[0].name)

    parity_path = ROOT / "docs/plans/artifacts/ssl-lstm-q20-tempered-rkl-transport-ensemble-2026-08-30/c2-strict-calibration/backend-parity/attempt-02-b8-graph-custom/run_manifest.json"
    if not parity_path.is_file():
        raise Phase8PilotError("required B=8 parity receipt is missing")
    parity = json.loads(parity_path.read_text(encoding="utf-8"))
    if (
        parity.get("status") != "PASS_Q20_BACKEND_PARITY_BATCH"
        or int(parity.get("batch_size", -1)) != 8
        or parity.get("strict_backend") != PRINCIPAL_SQRT_BACKEND
        or not bool(parity.get("parity", {}).get("passed"))
    ):
        raise Phase8PilotError("required B=8 parity receipt does not pass")

    from bayesfilter.inference.tempered_target_tf import make_q20_tempered_bridge
    from bayesfilter.inference.tempered_transitions_tf import (
        screen_transport_reliability,
    )

    bridge = make_q20_tempered_bridge(
        20, jit_compile=True, principal_sqrt_backend=PRINCIPAL_SQRT_BACKEND
    )
    if str(bridge.target_signature) != EXPECTED_TARGET_SIGNATURE:
        raise Phase8PilotError("q=20 target signature changed after C2 freeze")
    dimension = int(bridge.parameter_dim)
    center = tf.convert_to_tensor(bridge.prior_center, tf.float64)
    declared_points, map_receipt = _map_representatives(
        tf, str(bridge.target_signature)
    )
    reference_points = tf.concat(
        (
            center[tf.newaxis, :],
            center[tf.newaxis, :] + 4.0 * tf.eye(dimension, dtype=tf.float64),
            center[tf.newaxis, :] - 4.0 * tf.eye(dimension, dtype=tf.float64),
        ),
        axis=0,
    )
    route_paths = (
        ROOT / "bayesfilter/inference/tempered_target_tf.py",
        ROOT / "bayesfilter/inference/tempered_transport_ensemble_tf.py",
        ROOT / "bayesfilter/inference/tempered_lineage_tf.py",
        ROOT / "bayesfilter/inference/tempered_transitions_tf.py",
    )
    route_scan = _static_scan(route_paths)
    if not route_scan["passed"]:
        raise Phase8PilotError(f"forbidden row mapping or pfor in route: {route_scan}")
    compatibility = _compatibility_receipts()
    initial_nvidia = _nvidia_snapshot()
    successes: list[Mapping[str, Any]] = []
    row_records: list[Mapping[str, Any]] = []
    row_failures: list[Mapping[str, Any]] = []
    for architecture_index, architecture in enumerate(ARCHITECTURE_GRID):
        for root_index in range(len(INITIALIZATION_ROOTS)):
            _budget_guard(started)
            try:
                result = _run_c2_row(
                    tf,
                    bridge,
                    center=center,
                    dimension=dimension,
                    architecture=architecture,
                    architecture_index=architecture_index,
                    root_index=root_index,
                    output_dir=output_dir,
                    device_name=device_name,
                    started=started,
                )
                successes.append(result)
                row_records.append(result["record"])
            except Exception as exc:
                failure = {
                    "status": "FAIL_C2_ROW",
                    "architecture": dict(architecture),
                    "architecture_index": int(architecture_index),
                    "root_index": int(root_index),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                row_failures.append(failure)
                _stage_marker(
                    output_dir,
                    f"{architecture['name']}-root-{root_index}",
                    "failure",
                    error_type=type(exc).__name__,
                    error=str(exc),
                )

    reliability_groups: list[Mapping[str, Any]] = []
    reliability_by_architecture: dict[str, bool] = {}
    for architecture_index, architecture in enumerate(ARCHITECTURE_GRID):
        group = [
            item
            for item in successes
            if int(item["record"]["architecture_index"]) == architecture_index
        ]
        if len(group) != len(INITIALIZATION_ROOTS):
            reliability_by_architecture[str(architecture["name"])] = False
            reliability_groups.append(
                {
                    "architecture": architecture["name"],
                    "status": "SKIPPED_MISSING_ROOT",
                    "passed": False,
                }
            )
            continue
        _budget_guard(started)
        _reset_memory(tf, device_name)
        charts = [item["transport"] for item in group]
        self_bank = tf.stack([item["validation_latent"] for item in group], axis=0)
        cross_bank = tf.stack([item["stress_physical"] for item in group], axis=0)

        def score_fn(physical: Any) -> Any:
            return bridge.value_score_status(
                physical, tf.constant(0.5, tf.float64)
            )[1]

        try:
            receipt = screen_transport_reliability(
                charts,
                component_ids=tuple(item["record"]["component_id"] for item in group),
                self_latent_bank=self_bank,
                cross_physical_bank=cross_bank,
                reference_points=reference_points,
                declared_points=declared_points,
                physical_score_fn=score_fn,
                maximum_condition_number=1.0e8,
            )
            receipt_payload = _json_safe(receipt.payload(), tf)
            reliability_groups.append(
                {
                    "architecture": architecture["name"],
                    "status": "PASS_RELIABILITY" if receipt.passed else "FAIL_RELIABILITY",
                    "passed": bool(receipt.passed),
                    "receipt": receipt_payload,
                }
            )
            reliability_by_architecture[str(architecture["name"])] = bool(
                receipt.passed
            )
        except Exception as exc:
            reliability_groups.append(
                {
                    "architecture": architecture["name"],
                    "status": "FAIL_RELIABILITY_EXCEPTION",
                    "passed": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            reliability_by_architecture[str(architecture["name"])] = False

    architecture_summary = []
    for architecture in ARCHITECTURE_GRID:
        rows = [
            row for row in row_records if row["architecture"]["name"] == architecture["name"]
        ]
        viable_rows = [
            row
            for row in rows
            if bool(row.get("paired_improvement", {}).get("training_viable", False))
        ]
        reliability_passed = bool(
            reliability_by_architecture.get(str(architecture["name"]), False)
        )
        architecture_summary.append(
            {
                "architecture": architecture["name"],
                "successful_roots": len(rows),
                "viable_roots": len(viable_rows) if reliability_passed else 0,
                "viable_on_both_roots": reliability_passed
                and len(viable_rows) == len(INITIALIZATION_ROOTS),
                "reliability_passed": reliability_passed,
                "nomination_rule": "paired held-out improvement upper endpoint < 0 plus reliability",
            }
        )
    all_rows_pass = len(row_records) == len(ARCHITECTURE_GRID) * len(INITIALIZATION_ROOTS)
    all_reliability_pass = all(
        bool(group.get("passed", False)) for group in reliability_groups
    ) and len(reliability_groups) == len(ARCHITECTURE_GRID)
    any_architecture_both_viable = any(
        bool(summary["viable_on_both_roots"]) for summary in architecture_summary
    )
    passed = all_rows_pass and all_reliability_pass and any_architecture_both_viable
    final_nvidia = _nvidia_snapshot()
    source_paths = (*route_paths, Path(__file__).resolve(), PLAN, MASTER_PLAN, parity_path)
    manifest: Mapping[str, Any] = {
        "schema": SCHEMA,
        "status": "PASS_PHASE8_C2_STRICT_CALIBRATION" if passed else "FAIL_PHASE8_C2_STRICT_CALIBRATION",
        "role": "target_specific_calibration_nomination_only",
        "protocol": _protocol(validation_size=args.validation_size),
        "command": sys.argv,
        "output_dir": str(output_dir),
        "git_commit": _git(("git", "rev-parse", "HEAD")),
        "git_status_porcelain": _git(("git", "status", "--porcelain")),
        "python": sys.executable,
        "python_version": platform.python_version(),
        "tensorflow": tf.__version__,
        "tensorflow_probability": __import__("tensorflow_probability").__version__,
        "q": 20,
        "parameter_dim": dimension,
        "target_signature": str(bridge.target_signature),
        "bridge_signature": str(bridge.signature),
        "properness_receipt": bridge.properness_receipt.payload(),
        "principal_sqrt_backend": PRINCIPAL_SQRT_BACKEND,
        "jit_compile": True,
        "tf32_execution_enabled": bool(
            tf.config.experimental.tensor_float_32_execution_enabled()
        ),
        "logical_gpus": [str(item.name) for item in logical_gpus],
        "memory_policy": memory_policy,
        "gpu_environment": gpu_environment,
        "gpu_snapshot_before": initial_nvidia,
        "gpu_snapshot_after": final_nvidia,
        "compatibility": compatibility,
        "backend_parity_prerequisite": {
            "path": str(parity_path.relative_to(ROOT)),
            "sha256": _sha256(parity_path),
            "status": parity.get("status"),
            "batch_size": parity.get("batch_size"),
            "custom_jit_compile": parity.get("custom_jit_compile"),
            "strict_jit_compile": parity.get("strict_jit_compile"),
            "parity": parity.get("parity"),
        },
        "map_representatives": map_receipt,
        "route_scan": route_scan,
        "rows": row_records,
        "row_failures": row_failures,
        "reliability_groups": reliability_groups,
        "architecture_summary": architecture_summary,
        "hard_screen": {
            "all_rows_pass": all_rows_pass,
            "all_reliability_pass": all_reliability_pass,
            "any_architecture_viable_on_both_roots": any_architecture_both_viable,
        },
        "budget": {
            "material_cap_seconds": MATERIAL_CAP_SECONDS,
            "wall_time_seconds": time.monotonic() - started,
        },
        "source_hashes": {
            str(path.relative_to(ROOT)): _sha256(path) for path in source_paths
        },
        "wall_time_seconds": time.monotonic() - started,
        "nonclaims": [
            "calibration screen only; no architecture superiority claim",
            "no IID Gaussian whitening or exhaustive mode-discovery claim",
            "no posterior correctness, HMC convergence, production, or scaling claim",
        ],
    }
    safe_manifest = _json_safe(manifest, tf)
    safe_manifest["manifest_hash"] = _stable_hash(safe_manifest)
    _write_json(output_dir / "run_manifest.json", safe_manifest)
    print(
        json.dumps(
            {
                "status": safe_manifest["status"],
                "successful_rows": len(row_records),
                "failed_rows": len(row_failures),
                "wall_time_seconds": safe_manifest["wall_time_seconds"],
                "output_dir": str(output_dir),
            },
            sort_keys=True,
        )
    )
    return 0 if passed else 2


def _legacy_cost_main() -> int:
    args = _parse_args()
    if args.print_protocol:
        print(
            json.dumps(
                _protocol(args.mode, validation_size=args.validation_size),
                sort_keys=True,
                indent=2,
            )
        )
        return 0
    if args.output_dir is None:
        raise Phase8PilotError("--output-dir is required")
    if args.mode == "target-localization" and args.validation_size != VALIDATION_SIZE:
        raise Phase8PilotError(
            "target-localization requires the frozen validation-size=256 probe"
        )
    gpu_environment = _prepare_default_gpu_environment()
    if not _truthy(os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH")):
        raise Phase8PilotError(
            "GPU pilot requires TF_FORCE_GPU_ALLOW_GROWTH=true before import"
        )
    if os.environ.get("CUDA_VISIBLE_DEVICES", "").strip() in {"", "-1"}:
        raise Phase8PilotError("GPU pilot requires one explicitly visible GPU")
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise Phase8PilotError(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    started = time.monotonic()

    import tensorflow as tf

    from bayesfilter.inference.neutra_weighted_training import (
        WeightedDenseIAFTransport,
        WeightedNeuTraConfig,
    )
    from bayesfilter.inference.tempered_target_tf import make_q20_tempered_bridge
    from bayesfilter.inference.tempered_transitions_tf import (
        derive_reliability_tolerance,
        screen_transport_reliability,
    )
    from bayesfilter.inference.tempered_transport_ensemble_tf import (
        IndependentTemperedReverseKLTrainer,
        PreparedTransportInitialization,
        TransportBank,
        capture_trainable_transport_checkpoint,
        paired_reverse_kl_improvement,
        prepare_transport_initialization,
        pullback_gaussianization_diagnostic,
        restore_trainable_transport_checkpoint,
    )
    from bayesfilter.runtime.gpu_memory_policy import (
        configure_tensorflow_gpu_memory_growth,
    )

    memory_policy = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    tf.config.experimental.enable_tensor_float_32_execution(True)
    logical_gpus = tuple(tf.config.list_logical_devices("GPU"))
    if len(logical_gpus) != 1:
        raise Phase8PilotError("cost pilot requires exactly one visible logical GPU")
    device_name = str(logical_gpus[0].name)
    compatibility = _compatibility_receipts()
    route_paths = (
        ROOT / "bayesfilter/inference/tempered_target_tf.py",
        ROOT / "bayesfilter/inference/tempered_transport_ensemble_tf.py",
        ROOT / "bayesfilter/inference/tempered_lineage_tf.py",
        ROOT / "bayesfilter/inference/tempered_transitions_tf.py",
    )
    route_scan = _static_scan(route_paths)
    if not route_scan["passed"]:
        raise Phase8PilotError(f"forbidden row mapping or pfor: {route_scan}")

    bridge = make_q20_tempered_bridge(
        20,
        jit_compile=True,
        principal_sqrt_backend=args.principal_sqrt_backend,
    )
    if str(bridge.target_signature) != EXPECTED_TARGET_SIGNATURE:
        raise Phase8PilotError("q=20 target signature changed after plan freeze")
    dimension = int(bridge.parameter_dim)
    center = tf.convert_to_tensor(bridge.prior_center, tf.float64)
    declared_points, map_receipt = _map_representatives(
        tf, str(bridge.target_signature)
    )
    reference_points = tf.concat(
        (
            center[tf.newaxis, :],
            center[tf.newaxis, :] + 4.0 * tf.eye(dimension, dtype=tf.float64),
            center[tf.newaxis, :] - 4.0 * tf.eye(dimension, dtype=tf.float64),
        ),
        axis=0,
    )
    initial_nvidia = _nvidia_snapshot()
    if args.mode == "target-localization":
        localization = _run_target_localization(
            tf,
            bridge,
            output_dir=output_dir,
            center=center,
            dimension=dimension,
            principal_sqrt_backend=args.principal_sqrt_backend,
        )
        manifest = {
            "schema": SCHEMA,
            "status": localization["status"],
            "role": "diagnostic_only_stage_localization",
            "protocol": _protocol(args.mode, validation_size=args.validation_size),
            "command": sys.argv,
            "output_dir": str(output_dir),
            "git_commit": _git(("git", "rev-parse", "HEAD")),
            "git_status_porcelain": _git(("git", "status", "--porcelain")),
            "python": sys.executable,
            "python_version": platform.python_version(),
            "tensorflow": tf.__version__,
            "tensorflow_probability": __import__("tensorflow_probability").__version__,
            "q": 20,
            "parameter_dim": dimension,
            "target_signature": str(bridge.target_signature),
            "bridge_signature": str(bridge.signature),
            "properness_receipt": bridge.properness_receipt.payload(),
            "principal_sqrt_backend": args.principal_sqrt_backend,
            "jit_compile": True,
            "tf32_execution_enabled": bool(
                tf.config.experimental.tensor_float_32_execution_enabled()
            ),
            "logical_gpus": [str(item.name) for item in logical_gpus],
            "memory_policy": memory_policy,
            "gpu_environment": gpu_environment,
            "gpu_launch_mode": os.environ.get(
                "BAYESFILTER_GPU_LAUNCH_MODE", "direct_repo_default"
            ),
            "gpu_trust_basis": os.environ.get(
                "BAYESFILTER_GPU_TRUST_BASIS",
                "repository_default_gpu_route_external_boundary_unclassified",
            ),
            "external_approval_is_runner_gate": False,
            "gpu_snapshot_before": initial_nvidia,
            "localization": localization,
            "source_hashes": {
                str(path.relative_to(ROOT)): _sha256(path)
                for path in (*route_paths, Path(__file__).resolve(), PLAN, MASTER_PLAN)
            },
            "wall_time_seconds": time.monotonic() - started,
            "nonclaims": list(localization["nonclaims"]),
        }
        manifest = _json_safe(manifest, tf)
        manifest["manifest_hash"] = _stable_hash(manifest)
        _write_json(output_dir / "run_manifest.json", manifest)
        print(
            json.dumps(
                {
                    "status": manifest["status"],
                    "mode": manifest["protocol"]["mode"],
                    "wall_time_seconds": manifest["wall_time_seconds"],
                    "output_dir": str(output_dir),
                },
                sort_keys=True,
            )
        )
        return 0
    batch_results = []

    for batch_index, batch_size in enumerate(BATCH_CANDIDATES):
        _reset_memory(tf, device_name)
        charts = []
        component_rows = []
        for component_index in range(COMPONENT_COUNT):
            component_id = f"cost-b{batch_size}-chart-{component_index}"
            config = WeightedNeuTraConfig(
                dimension=dimension,
                hidden_layers=(16, 16),
                stages=2,
                activation="tanh",
                initialization_scale=0.02,
                initialization_seed=_seed(
                    tf, (20260829, 11001), batch_index, component_index
                ),
                learning_rate=1.0e-3,
                jit_compile=True,
            )
            raw = WeightedDenseIAFTransport(config)
            beta0_prepared = prepare_transport_initialization(
                raw,
                bridge,
                component_id=component_id,
                seed=_seed(tf, (20260829, 11002), batch_index, component_index),
                batch_size=batch_size,
                repair_scales=(1.0, 0.5, 0.25),
                beta=0.0,
                reference_center=center,
                reference_scale=math.sqrt(float(bridge.prior_variance)),
            )
            if not beta0_prepared.receipt.valid:
                raise Phase8PilotError(f"beta=0 preflight failed for {component_id}")
            endpoint_latent = tf.random.stateless_normal(
                [args.validation_size, dimension],
                tf.constant(
                    _seed(tf, (20260829, 31001), batch_index, component_index, 0),
                    tf.int32,
                ),
                dtype=tf.float64,
            )
            endpoint_diagnostic = pullback_gaussianization_diagnostic(
                beta0_prepared.transport,
                bridge,
                beta=0.0,
                latent=endpoint_latent,
            )
            endpoint_tolerance = derive_reliability_tolerance(dimension=dimension)
            if (
                not bool(endpoint_diagnostic.finite.numpy())
                or float(endpoint_diagnostic.centered_log_density_rms.numpy())
                > endpoint_tolerance
                or float(endpoint_diagnostic.pullback_score_maximum_row_norm.numpy())
                > endpoint_tolerance
            ):
                raise Phase8PilotError(
                    f"reference-affine chart does not equal beta=0 law: {component_id}"
                )
            beta0_checkpoint = capture_trainable_transport_checkpoint(
                beta0_prepared.transport,
                component_id=component_id,
                beta=0.0,
                bridge_signature=str(bridge.signature),
                target_signature=str(bridge.target_signature),
                parent_checkpoint_hash=None,
                update_count=0,
                checkpoint_scope=_checkpoint_scope(
                    target_signature=str(bridge.target_signature),
                    principal_sqrt_backend=args.principal_sqrt_backend,
                    batch_index=batch_index,
                    component_index=component_index,
                    beta=0.0,
                    validation_size=args.validation_size,
                ),
            )
            _write_json(
                output_dir / f"batch-{batch_size}" / f"{component_id}-beta0.json",
                _json_safe(beta0_checkpoint, tf),
            )
            continued = restore_trainable_transport_checkpoint(
                beta0_checkpoint,
                expected_context={
                    "component_id": component_id,
                    "beta": 0.0,
                    "bridge_signature": str(bridge.signature),
                    "target_signature": str(bridge.target_signature),
                    "checkpoint_scope": beta0_checkpoint["checkpoint_scope"],
                },
            )
            replay = _checkpoint_replay_error(
                tf, beta0_prepared.transport, continued, endpoint_latent
            )
            if any(float(value.numpy()) > endpoint_tolerance for value in replay.values()):
                raise Phase8PilotError(f"beta=0 checkpoint replay failed: {component_id}")
            beta_half_prepared = prepare_transport_initialization(
                continued,
                bridge,
                component_id=component_id,
                seed=_seed(tf, (20260829, 11003), batch_index, component_index),
                batch_size=batch_size,
                repair_scales=(1.0,),
                beta=0.5,
            )
            validation_latent = tf.random.stateless_normal(
                [args.validation_size, dimension],
                tf.constant(
                    _seed(tf, (20260829, 41001), batch_index, component_index),
                    tf.int32,
                ),
                dtype=tf.float64,
            )
            start_diagnostic = pullback_gaussianization_diagnostic(
                beta_half_prepared.transport,
                bridge,
                beta=0.5,
                latent=validation_latent,
            )
            start_checkpoint = capture_trainable_transport_checkpoint(
                beta_half_prepared.transport,
                component_id=component_id,
                beta=0.5,
                bridge_signature=str(bridge.signature),
                target_signature=str(bridge.target_signature),
                parent_checkpoint_hash=str(beta0_checkpoint["checkpoint_hash"]),
                update_count=0,
                checkpoint_scope=_checkpoint_scope(
                    target_signature=str(bridge.target_signature),
                    principal_sqrt_backend=args.principal_sqrt_backend,
                    batch_index=batch_index,
                    component_index=component_index,
                    beta=0.5,
                    validation_size=args.validation_size,
                ),
            )
            _write_json(
                output_dir
                / f"batch-{batch_size}"
                / f"{component_id}-beta05-start.json",
                _json_safe(start_checkpoint, tf),
            )
            start_transport = restore_trainable_transport_checkpoint(start_checkpoint)
            trainer = IndependentTemperedReverseKLTrainer(
                config,
                bridge,
                beta=0.5,
                component_id=component_id,
                batch_size=batch_size,
                prepared_initialization=PreparedTransportInitialization(
                    beta_half_prepared.transport, beta_half_prepared.receipt
                ),
            )
            update_times = []
            update_rows = []
            for update_index in range(1 + STEADY_UPDATE_COUNT):
                update_started = time.monotonic()
                update = trainer.train_step(
                    _seed(
                        tf,
                        (20260829, 21001),
                        batch_index,
                        component_index,
                        update_index,
                    )
                )
                elapsed = time.monotonic() - update_started
                update_times.append(elapsed)
                update_rows.append(
                    {
                        "update": update_index + 1,
                        "elapsed_seconds": elapsed,
                        "loss": update.loss,
                        "gradient_norm": update.gradient_norm,
                        "clipped_gradient_norm": update.clipped_gradient_norm,
                        "clipping_applied": update.clipping_applied,
                        "target_call_count": update.target_call_count,
                        "valid": update.valid,
                        "loss_device": str(update.loss.device),
                    }
                )
            final_diagnostic = pullback_gaussianization_diagnostic(
                trainer.transport,
                bridge,
                beta=0.5,
                latent=validation_latent,
            )
            improvement = paired_reverse_kl_improvement(
                start_diagnostic, final_diagnostic
            )
            final_checkpoint = capture_trainable_transport_checkpoint(
                trainer.transport,
                component_id=component_id,
                beta=0.5,
                bridge_signature=str(bridge.signature),
                target_signature=str(bridge.target_signature),
                parent_checkpoint_hash=str(start_checkpoint["checkpoint_hash"]),
                update_count=1 + STEADY_UPDATE_COUNT,
                checkpoint_scope=start_checkpoint["checkpoint_scope"],
            )
            _write_json(
                output_dir / f"batch-{batch_size}" / f"{component_id}-beta05.json",
                _json_safe(final_checkpoint, tf),
            )
            restored_final = restore_trainable_transport_checkpoint(
                final_checkpoint,
                expected_context={
                    "component_id": component_id,
                    "beta": 0.5,
                    "bridge_signature": str(bridge.signature),
                    "target_signature": str(bridge.target_signature),
                    "checkpoint_scope": final_checkpoint["checkpoint_scope"],
                },
            )
            final_replay = _checkpoint_replay_error(
                tf, trainer.transport, restored_final, validation_latent
            )
            if any(float(value.numpy()) > endpoint_tolerance for value in final_replay.values()):
                raise Phase8PilotError(f"final checkpoint replay failed: {component_id}")
            occupancy_latent = tf.random.stateless_normal(
                [OCCUPANCY_SIZE, dimension],
                tf.constant(
                    _seed(tf, (20260829, 41002), batch_index, component_index),
                    tf.int32,
                ),
                dtype=tf.float64,
            )
            occupancy_physical, _ = restored_final.forward_and_logdet(occupancy_latent)
            positive_fraction = tf.reduce_mean(
                tf.cast(occupancy_physical[:, 2] > 0.0, tf.float64)
            )
            charts.append(restored_final)
            component_rows.append(
                {
                    "component_id": component_id,
                    "beta0_preflight": beta0_prepared.receipt.payload(),
                    "beta_half_preflight": beta_half_prepared.receipt.payload(),
                    "endpoint_diagnostic": _diagnostic_payload(endpoint_diagnostic, tf),
                    "beta0_checkpoint_hash": beta0_checkpoint["checkpoint_hash"],
                    "start_checkpoint_hash": start_checkpoint["checkpoint_hash"],
                    "final_checkpoint_hash": final_checkpoint["checkpoint_hash"],
                    "beta0_replay": replay,
                    "final_replay": final_replay,
                    "start_diagnostic": _diagnostic_payload(start_diagnostic, tf),
                    "final_diagnostic": _diagnostic_payload(final_diagnostic, tf),
                    "paired_improvement": improvement,
                    "compile_update_seconds": update_times[0],
                    "steady_update_seconds": update_times[1:],
                    "median_steady_update_seconds": statistics.median(update_times[1:]),
                    "updates": update_rows,
                    "forward_positive_sign_fraction": positive_fraction,
                    "start_transport_retained_for_pairing_only": start_transport is not trainer.transport,
                }
            )

        self_latent = tf.random.stateless_normal(
            [COMPONENT_COUNT, 4, dimension],
            tf.constant(_seed(tf, (20260829, 51001), batch_index, 1), tf.int32),
            dtype=tf.float64,
        )
        cross_rows = []
        for component_index, chart in enumerate(charts):
            values, _ = chart.forward_and_logdet(self_latent[component_index])
            cross_rows.append(values)
        cross_physical = tf.stack(cross_rows, axis=0)

        def score_fn(physical: Any) -> Any:
            return bridge.value_score_status(
                physical, tf.constant(0.5, tf.float64)
            )[1]

        reliability = screen_transport_reliability(
            charts,
            component_ids=tuple(row["component_id"] for row in component_rows),
            self_latent_bank=self_latent,
            cross_physical_bank=cross_physical,
            reference_points=reference_points,
            declared_points=declared_points,
            physical_score_fn=score_fn,
            maximum_condition_number=1.0e8,
        )
        if not reliability.passed:
            raise Phase8PilotError(
                f"learned-map reliability failed for batch {batch_size}: {reliability.payload()}"
            )
        bank = TransportBank(
            charts,
            component_ids=tuple(row["component_id"] for row in component_rows),
        )
        density_latent = tf.random.stateless_normal(
            [COMPONENT_COUNT, batch_size, dimension],
            tf.constant(_seed(tf, (20260829, 51001), batch_index, 2), tf.int32),
            dtype=tf.float64,
        )
        density_physical, _ = bank.forward_bank(density_latent)
        density_started = time.monotonic()
        cross_density = bank.cross_component_log_prob(density_physical)
        _ = cross_density.numpy()
        density_seconds = time.monotonic() - density_started
        memory = _memory_info(tf, device_name)
        peak = int(memory.get("peak", PILOT_MEMORY_CAP_BYTES + 1))
        if peak > PILOT_MEMORY_CAP_BYTES:
            raise Phase8PilotError(
                f"batch {batch_size} exceeded the 4 GiB pilot allocator cap"
            )
        batch_results.append(
            {
                "batch_size": batch_size,
                "components": component_rows,
                "reliability": reliability.payload(),
                "cross_density_shape": cross_density.shape.as_list(),
                "cross_density_work": COMPONENT_COUNT * COMPONENT_COUNT * batch_size,
                "cross_density_elapsed_seconds": density_seconds,
                "allocator": memory,
                "median_steady_update_seconds": statistics.median(
                    tuple(
                        float(row["median_steady_update_seconds"])
                        for row in component_rows
                    )
                ),
            }
        )

    small, large = batch_results
    selected_batch = (
        32
        if int(large["allocator"]["peak"]) <= PILOT_MEMORY_CAP_BYTES
        and float(large["median_steady_update_seconds"])
        <= 4.0 * float(small["median_steady_update_seconds"])
        else 8
    )
    final_nvidia = _nvidia_snapshot()
    source_paths = (*route_paths, Path(__file__).resolve(), PLAN, MASTER_PLAN)
    manifest = {
        "schema": SCHEMA,
        "status": "PASS_PHASE8_COST_PILOT",
        "role": "feasibility_only_no_candidate_selection",
        "protocol": _protocol(args.mode, validation_size=args.validation_size),
        "command": sys.argv,
        "output_dir": str(output_dir),
        "git_commit": _git(("git", "rev-parse", "HEAD")),
        "git_status_porcelain": _git(("git", "status", "--porcelain")),
        "python": sys.executable,
        "python_version": platform.python_version(),
        "tensorflow": tf.__version__,
        "tensorflow_probability": __import__("tensorflow_probability").__version__,
        "q": 20,
        "parameter_dim": dimension,
        "target_signature": str(bridge.target_signature),
        "bridge_signature": str(bridge.signature),
        "properness_receipt": bridge.properness_receipt.payload(),
        "principal_sqrt_backend": args.principal_sqrt_backend,
        "jit_compile": True,
        "tf32_execution_enabled": bool(
            tf.config.experimental.tensor_float_32_execution_enabled()
        ),
        "logical_gpus": [str(item.name) for item in logical_gpus],
        "memory_policy": memory_policy,
        "gpu_environment": gpu_environment,
        "gpu_launch_mode": os.environ.get(
            "BAYESFILTER_GPU_LAUNCH_MODE", "direct_repo_default"
        ),
        "gpu_trust_basis": os.environ.get(
            "BAYESFILTER_GPU_TRUST_BASIS",
            "repository_default_gpu_route_external_boundary_unclassified",
        ),
        "external_approval_is_runner_gate": False,
        "gpu_snapshot_before": initial_nvidia,
        "gpu_snapshot_after": final_nvidia,
        "compatibility": compatibility,
        "map_representatives": map_receipt,
        "route_scan": route_scan,
        "batch_results": batch_results,
        "batch_selection": {
            "selected_batch_size": selected_batch,
            "rule": "B32 iff finite, peak<=4GiB, and median step<=4x B8",
            "selection_role": "operational_cost_nomination_only",
        },
        "source_hashes": {
            str(path.relative_to(ROOT)): _sha256(path) for path in source_paths
        },
        "wall_time_seconds": time.monotonic() - started,
        "nonclaims": [
            "cost pilot does not select an architecture or sampler",
            "five updates do not establish training quality or Gaussianization",
            "base-bank diagnostics do not establish global mode coverage",
            "no HMC convergence, posterior correctness, superiority, or scaling claim",
        ],
    }
    manifest = _json_safe(manifest, tf)
    manifest["manifest_hash"] = _stable_hash(manifest)
    _write_json(output_dir / "run_manifest.json", manifest)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "selected_batch_size": selected_batch,
                "wall_time_seconds": manifest["wall_time_seconds"],
                "output_dir": str(output_dir),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    output: Path | None = None
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:
        try:
            arguments = _parse_args()
            output = arguments.output_dir
            if output is not None:
                failure_path = output.expanduser().resolve() / "failure.json"
                if failure_path.parent.exists() and not failure_path.exists():
                    _write_json(
                        failure_path,
                        {
                            "schema": SCHEMA,
                            "status": "FAIL_PHASE8_COST_PILOT",
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                            "traceback": traceback.format_exc(),
                        },
                    )
        finally:
            raise
