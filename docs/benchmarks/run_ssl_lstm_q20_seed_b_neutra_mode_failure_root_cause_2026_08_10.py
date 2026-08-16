#!/usr/bin/env python3
"""Diagnose seed-B NeuTra mode omission, pullback geometry, and HMC trapping."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "8")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")

ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = Path(os.environ.get("BAYESFILTER_CODE_ROOT", str(ROOT))).resolve()
BENCHMARKS = ROOT / "docs" / "benchmarks"
for directory in (BENCHMARKS, ROOT, CODE_ROOT):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

PLAN = Path(
    "docs/plans/"
    "bayesfilter-ssl-lstm-q20-seed-b-neutra-mode-failure-root-cause-plan-2026-08-10.md"
)
ORIGINAL_VALIDATION_PLAN = Path(
    "docs/plans/"
    "bayesfilter-ssl-lstm-q20-seed-b-terminal-neutra-validation-plan-2026-08-07.md"
)
RUNNER = Path(
    "docs/benchmarks/"
    "run_ssl_lstm_q20_seed_b_neutra_mode_failure_root_cause_2026_08_10.py"
)
MAP_ARTIFACT = Path(
    "docs/plans/artifacts/"
    "ssl-lstm-q20-seed-b-posterior-reference-2026-08-07/r3/map-progress.json"
)
TUNING_ARTIFACT = Path(
    "docs/plans/artifacts/"
    "ssl-lstm-q20-seed-b-terminal-neutra-validation-2026-08-07/"
    "r1/tuning/merged-tuning-result.json"
)
SEQUENTIAL_SUMMARY = Path(
    "docs/plans/artifacts/"
    "ssl-lstm-q20-seed-b-terminal-neutra-validation-2026-08-07/"
    "r2/sequential/summary.json"
)
ORIGINAL_LAUNCH = Path(
    "docs/plans/artifacts/"
    "ssl-lstm-q20-seed-b-terminal-neutra-validation-2026-08-07/"
    "r2/sequential/launch.json"
)
ARCHIVED_PARITY_RECEIPT = Path(
    "docs/plans/artifacts/"
    "ssl-lstm-q20-seed-b-terminal-neutra-validation-2026-08-07/"
    "r2/sequential/archive/retained/seed-b-terminal-retained-000-receipt.json"
)
OUTPUT_ROOT = Path(
    "docs/plans/artifacts/"
    "ssl-lstm-q20-seed-b-neutra-mode-failure-root-cause-2026-08-10/r1"
)
GEOMETRY_ARTIFACT = OUTPUT_ROOT / "geometry.json"

TARGET_SIGNATURE = "9a86e60081f1b9cd288dbdb1dcbe1e9a5b5e23d9b5ef97afdb72ee95c23d7278"
BASE_ADAPTER_SIGNATURE = "a8be6c212eb2a74ef926ccb9279871805949cae6e00c312a12525141638166f3"
PARAMETER_DIM = 4
OBSERVATION_WEIGHT_INDEX = 2
THREADS = 8
FLOW_SAMPLE_COUNT = 100_000
FLOW_SEED = (20260810, 1001)
PATH_POINT_COUNT = 65
CURVATURE_STEPS = (1.0e-3, 3.0e-4, 1.0e-4)
ROUNDTRIP_TOLERANCE = 1.0e-10
STATIONARY_SCORE_TOLERANCE = 1.0e-5
CURVATURE_STABILITY_TOLERANCE = 1.0e-3
CANARY_CHAINS_PER_REGION = 16
CANARY_TRANSITIONS = 8
MATERIAL_CHAINS_PER_REGION = 32
MATERIAL_TRANSITIONS = 64
STATIONARY_CHAINS_PER_REGION = 8
STATIONARY_TRANSITIONS = 4
STATIONARY_CONTROL_STEP_SIZE = 0.1
CANARY_SEED = (20260810, 2001)
MATERIAL_SEED = (20260810, 3001)
STATIONARY_SEED = (20260810, 4001)
CAMPAIGN_CAP_SECONDS = 12_000.0
ORIGINAL_TARGET_SCOPE = (
    "ssl_lstm_neutra_state_complexity_batch_native:q20:"
    "fixed_hmc_api:seed-b-terminal-step-6250:claim_tuning_grid6"
)
PARITY_VALUE_TOLERANCE = 5.0e-7
PARITY_SCORE_TOLERANCE = 5.0e-7
PARITY_DRAW_INDICES = (0, 31, 127, 255)


class RootCauseDiagnosticError(RuntimeError):
    """Raised when an evidence-contract invariant fails."""


def _abs(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _sha(path: Path) -> str:
    return hashlib.sha256(_abs(path).read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(_abs(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RootCauseDiagnosticError(f"expected JSON object: {path}")
    return payload


def _safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_safe(item) for item in value]
    if isinstance(value, bytes):
        return value.decode("ascii")
    if isinstance(value, Path):
        return value.as_posix()
    if hasattr(value, "numpy"):
        return _safe(value.numpy())
    if hasattr(value, "tolist"):
        return _safe(value.tolist())
    if hasattr(value, "item"):
        return _safe(value.item())
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    absolute = _abs(path)
    absolute.parent.mkdir(parents=True, exist_ok=True)
    if absolute.exists():
        raise RootCauseDiagnosticError(f"refusing to overwrite artifact: {path}")
    encoded = (
        json.dumps(_safe(payload), sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("ascii")
    temporary = absolute.with_suffix(absolute.suffix + ".tmp")
    temporary.write_bytes(encoded)
    temporary.replace(absolute)


def _write_tensor(path: Path, tensor: Any, tf: Any) -> dict[str, Any]:
    absolute = _abs(path)
    absolute.parent.mkdir(parents=True, exist_ok=True)
    if absolute.exists():
        raise RootCauseDiagnosticError(f"refusing to overwrite artifact: {path}")
    encoded = bytes(tf.io.serialize_tensor(tensor).numpy())
    absolute.write_bytes(encoded)
    return {
        "path": path.as_posix(),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "bytes": len(encoded),
        "dtype": tensor.dtype.name,
        "shape": list(tensor.shape),
    }


def zero_count_upper_bound(sample_count: int, confidence: float = 0.95) -> float:
    """Return the exact one-sided binomial bound after zero successes."""

    if sample_count <= 0 or not 0.0 < confidence < 1.0:
        raise ValueError("sample_count and confidence are invalid")
    return 1.0 - (1.0 - confidence) ** (1.0 / sample_count)


def identity_mass_kinetic_survival_4d(delta: float) -> float:
    """Return P[K > delta] for K=0.5*chi-square(4)."""

    if not math.isfinite(delta) or delta < 0.0:
        raise ValueError("delta must be finite and nonnegative")
    return math.exp(-delta) * (1.0 + delta)


def optimizer_result_item(tf: Any, value: Any, index: int) -> Any:
    """Return scalar TFP optimizer metadata or one row of batched metadata."""

    tensor = tf.convert_to_tensor(value)
    return tensor if tensor.shape.rank == 0 else tf.reshape(tensor, [-1])[index]


def select_sign_representatives(
    rows: list[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    eligible: dict[str, list[dict[str, Any]]] = {"plus": [], "minus": []}
    for row in rows:
        try:
            position = [float(value) for value in row["position"]]
            log_prob = float(row["log_prob"])
            score = float(row["score_inf_norm"])
        except (KeyError, TypeError, ValueError):
            continue
        if (
            len(position) != PARAMETER_DIM
            or not all(math.isfinite(value) for value in position)
            or not math.isfinite(log_prob)
            or not math.isfinite(score)
            or score > STATIONARY_SCORE_TOLERANCE
            or position[OBSERVATION_WEIGHT_INDEX] == 0.0
        ):
            continue
        label = "plus" if position[OBSERVATION_WEIGHT_INDEX] > 0.0 else "minus"
        eligible[label].append(
            {
                "position": position,
                "log_prob": log_prob,
                "score_inf_norm": score,
                "start_index": int(row.get("start_index", -1)),
            }
        )
    if not eligible["plus"] or not eligible["minus"]:
        raise RootCauseDiagnosticError("both sign representatives are required")
    return {
        label: max(candidates, key=lambda item: item["log_prob"])
        for label, candidates in eligible.items()
    }


def _stable_hash(payload: Any) -> str:
    encoded = json.dumps(
        _safe(payload), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _read_tensor_receipt(tf: Any, receipt: Mapping[str, Any], label: str) -> Any:
    path = Path(str(receipt["path"]))
    encoded = _abs(path).read_bytes()
    if hashlib.sha256(encoded).hexdigest() != str(receipt.get("sha256")):
        raise RootCauseDiagnosticError(f"{label} archived tensor hash mismatch")
    tensor = tf.io.parse_tensor(
        encoded, out_type=tf.dtypes.as_dtype(str(receipt["dtype"]))
    )
    if list(tensor.shape) != [int(value) for value in receipt["shape"]]:
        raise RootCauseDiagnosticError(f"{label} archived tensor shape mismatch")
    return tensor


def _archived_value_score_compatibility(tf: Any, adapter: Any) -> dict[str, Any]:
    receipt = _read_json(ARCHIVED_PARITY_RECEIPT)
    sample_receipt = receipt.get("sample_receipt")
    trace_receipts = receipt.get("trace_receipts")
    if not isinstance(sample_receipt, Mapping) or not isinstance(
        trace_receipts, Mapping
    ):
        raise RootCauseDiagnosticError("archived parity receipt is incomplete")
    archived_samples = _read_tensor_receipt(tf, sample_receipt, "samples")
    archived_values = _read_tensor_receipt(
        tf, trace_receipts["target_log_prob"], "target_log_prob"
    )
    archived_scores = _read_tensor_receipt(
        tf, trace_receipts["target_score"], "target_score"
    )
    selected_samples = tf.reshape(
        tf.gather(archived_samples, PARITY_DRAW_INDICES, axis=0),
        (-1, PARAMETER_DIM),
    )
    selected_values = tf.reshape(
        tf.gather(archived_values, PARITY_DRAW_INDICES, axis=0), (-1,)
    )
    selected_scores = tf.reshape(
        tf.gather(archived_scores, PARITY_DRAW_INDICES, axis=0),
        (-1, PARAMETER_DIM),
    )
    # The archive came from persistent one-chain workers, so parity must use the
    # original static target shape [1,4]; XLA reductions can be batch-shape sensitive.
    current_rows = [
        _evaluate(
            tf,
            adapter,
            selected_samples[index : index + 1],
            f"archived-value-score-parity-{index}",
        )
        for index in range(int(selected_samples.shape[0]))
    ]
    current_values = tf.concat([row[0] for row in current_rows], axis=0)
    current_scores = tf.concat([row[1] for row in current_rows], axis=0)
    value_residual = float(
        tf.reduce_max(tf.abs(current_values - selected_values)).numpy()
    )
    score_residual = float(
        tf.reduce_max(tf.abs(current_scores - selected_scores)).numpy()
    )
    passed = (
        value_residual <= PARITY_VALUE_TOLERANCE
        and score_residual <= PARITY_SCORE_TOLERANCE
    )
    result = {
        "status": "PASSED" if passed else "FAILED",
        "selected_draw_indices": PARITY_DRAW_INDICES,
        "selected_state_count": int(selected_samples.shape[0]),
        "evaluation_batch_size": 1,
        "evaluation_shape_provenance": "historical_persistent_one_chain_workers",
        "maximum_absolute_value_residual": value_residual,
        "maximum_absolute_score_residual": score_residual,
        "value_tolerance": PARITY_VALUE_TOLERANCE,
        "score_tolerance": PARITY_SCORE_TOLERANCE,
        "archived_receipt": {
            "path": ARCHIVED_PARITY_RECEIPT.as_posix(),
            "sha256": _sha(ARCHIVED_PARITY_RECEIPT),
            "sample": dict(sample_receipt),
            "target_log_prob": dict(trace_receipts["target_log_prob"]),
            "target_score": dict(trace_receipts["target_score"]),
        },
        "role": "numerical_compatibility_gate_after_manifest_schema_drift",
        "nonclaim": "finite point parity is not proof of global program identity",
    }
    if not passed:
        raise RootCauseDiagnosticError(
            "current transformed target fails archived value/score parity: "
            f"value={value_residual}, score={score_residual}"
        )
    return result


def _configure_and_build() -> tuple[Any, Any, Any, Any, dict[str, Any]]:
    import tensorflow as tf

    tf.config.threading.set_intra_op_parallelism_threads(THREADS)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.set_visible_devices([], "GPU")
    if tf.config.list_physical_devices("GPU"):
        raise RootCauseDiagnosticError("CPU-only diagnostic found a visible GPU")
    from bayesfilter.inference.neutra_artifacts import load_frozen_neutra_artifact
    from bayesfilter.inference.neutra_training import (
        NeuTraReverseKLTrainer,
        ssl_lstm_tuned_capacity_neutra_config,
    )
    from bayesfilter.inference.batched_value_score import FixedTransportValueScoreAdapter
    from bayesfilter.inference.posterior_adapter import ValueScoreCapability
    from bayesfilter.nonlinear.ssl_lstm_complexity_batched_target_tf import (
        batch_native_complexity_posterior_target,
    )
    from bayesfilter.nonlinear.ssl_lstm_complexity_target_tf import FREE_NAMES

    checkpoint = _read_json(
        Path(
            "docs/plans/artifacts/ssl-lstm-q20-neutra-budgeted-continuation-2026-08-06/"
            "r1/seed-b/checkpoint-4000.json"
        )
    )
    raw_checkpoint = dict(checkpoint)
    supplied_checkpoint_hash = str(raw_checkpoint.pop("checkpoint_hash", ""))
    if supplied_checkpoint_hash != hashlib.sha256(
        (json.dumps(raw_checkpoint, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("ascii")
    ).hexdigest():
        raise RootCauseDiagnosticError("seed-B checkpoint hash mismatch")
    state = checkpoint.get("trainer_state")
    if not isinstance(state, Mapping):
        raise RootCauseDiagnosticError("seed-B trainer state is missing")
    raw_state = dict(state)
    supplied_state_hash = str(raw_state.pop("state_hash", ""))
    if supplied_state_hash != _stable_hash(raw_state):
        raise RootCauseDiagnosticError("seed-B trainer state hash mismatch")
    config = state.get("config")
    if not isinstance(config, Mapping):
        raise RootCauseDiagnosticError("seed-B trainer config is missing")
    target = batch_native_complexity_posterior_target(
        20, jit_compile=True, principal_sqrt_backend="tensorflow_eigh"
    )
    trainer_config = ssl_lstm_tuned_capacity_neutra_config(
        dimension=int(config["dimension"]),
        fixed_translation=tuple(float(value) for value in config["fixed_translation"]),
        target_parameter_names=tuple(str(value) for value in config["target_parameter_names"]),
        target_signature=target.target_signature(),
        target_adapter_signature=target.adapter_signature(),
        learning_rate=float(config["learning_rate"]),
        initialization_scale=float(config["initialization_scale"]),
        gradient_clip_norm=float(config["gradient_clip_norm"]),
        initialization_seed=tuple(int(value) for value in config["initialization_seed"]),
        jit_compile=True,
    )
    trainer = NeuTraReverseKLTrainer(target, trainer_config)
    trainer.restore_state(state)
    if int(trainer.step.numpy()) != 6250:
        raise RootCauseDiagnosticError("restored optimizer step mismatch")
    frozen = trainer.frozen_transport_payload(
        transport_id="ssl-lstm-q20-seed-b-terminal-step-6250",
        target_signature=target.target_signature(),
    )
    loaded = load_frozen_neutra_artifact(
        frozen, expected_target_signature=target.target_signature()
    )
    transport = loaded.transport

    class Bridge:
        parameter_dim = PARAMETER_DIM
        parameter_names = tuple(FREE_NAMES)
        supports_retained_draw_batch = False
        supports_retained_flat_batch = True
        supports_retained_value_score_status = True
        target_status_invalid_rows_become_nonfinite = True
        target_scope = (
            "ssl_lstm_neutra_state_complexity_batch_native:q20:"
            "fixed_hmc_api:seed-b-terminal-step-6250"
        )

        def adapter_signature(self) -> str:
            return target.adapter_signature()

        def value_score_capability(self) -> ValueScoreCapability:
            return ValueScoreCapability(
                value_score_authority="graph_native",
                xla_hmc_ready=True,
                full_chain_xla_diagnostic_ready=True,
                runtime_backend="ssl_lstm_q20_seed_b_terminal_cpu_xla_bridge",
                evidence_path=ORIGINAL_VALIDATION_PLAN.as_posix(),
                target_scope=self.target_scope,
                nonclaims=("CPU/XLA diagnostic exception", "no posterior oracle"),
            )

        def log_prob_and_grad(self, values: Any) -> tuple[Any, Any]:
            return target.batch_value_and_score(tf.convert_to_tensor(values, tf.float64))

        def log_prob_and_grad_status(self, values: Any) -> tuple[Any, Any, Mapping[str, Any]]:
            return target.neutra_batch_log_prob_and_grad_status(values)

        def target_status_telemetry(self, values: Any) -> Mapping[str, Any]:
            return self.log_prob_and_grad_status(values)[2]

    bridge = Bridge()
    provenance = {
        "chart": "seed-b-terminal",
        "checkpoint_hash": supplied_checkpoint_hash,
        "checkpoint_sha256": _sha(
            Path(
                "docs/plans/artifacts/ssl-lstm-q20-neutra-budgeted-continuation-2026-08-06/"
                "r1/seed-b/checkpoint-4000.json"
            )
        ),
        "optimizer_step": 6250,
        "trainer_state_hash": supplied_state_hash,
        "target_signature": target.target_signature(),
        "target_adapter_signature": target.adapter_signature(),
        "transport_hash": loaded.manifest.transport_hash,
        "transport_artifact_signature": loaded.artifact_signature,
    }
    if provenance.get("target_signature") != TARGET_SIGNATURE:
        raise RootCauseDiagnosticError("target signature mismatch")
    if provenance.get("target_adapter_signature") != BASE_ADAPTER_SIGNATURE:
        raise RootCauseDiagnosticError("base adapter signature mismatch")
    adapter = FixedTransportValueScoreAdapter(
        base_adapter=bridge,
        transport=transport,
        target_scope=ORIGINAL_TARGET_SCOPE,
        runtime_backend="ssl_lstm_q20_seed_b_terminal_sequential_cpu_xla",
        evidence_path=ORIGINAL_VALIDATION_PLAN.as_posix(),
        xla_hmc_ready=True,
        full_chain_xla_diagnostic_ready=True,
        require_batch_native=True,
        nonclaims=(
            "CPU/XLA validation exception to GPU training default",
            "sequential sampler screen is not posterior correctness",
        ),
    )
    tuning = _read_json(TUNING_ARTIFACT)
    kernel = tuning.get("final_kernel_payload")
    if (
        tuning.get("passed") is not True
        or tuning.get("final_status") != "passed"
        or not isinstance(kernel, Mapping)
    ):
        raise RootCauseDiagnosticError("frozen tuning artifact is not passed")
    summary = _read_json(SEQUENTIAL_SUMMARY)
    if summary.get("status") != "SEQUENTIAL_SCREEN_PASSED":
        raise RootCauseDiagnosticError("source sequential run was not passed")
    historical_provenance = summary.get("provenance")
    if not isinstance(historical_provenance, Mapping):
        raise RootCauseDiagnosticError("historical sequential provenance is missing")
    for name in (
        "target_signature",
        "target_adapter_signature",
        "checkpoint_hash",
        "checkpoint_sha256",
        "optimizer_step",
        "trainer_state_hash",
    ):
        if str(provenance.get(name)) != str(historical_provenance.get(name)):
            raise RootCauseDiagnosticError(f"stable seed-B identity drift: {name}")
    if bridge.adapter_signature() != str(kernel.get("base_adapter_signature")):
        raise RootCauseDiagnosticError("base adapter signature drift")
    if (
        int(kernel.get("num_leapfrog_steps", -1)) != 3
        or float(kernel.get("step_size", math.nan)) != 0.8115211181271775
        or kernel.get("mass_policy") != "fixed_identity_z"
        or kernel.get("use_xla") is not True
    ):
        raise RootCauseDiagnosticError("frozen kernel mechanics mismatch")
    parity = _archived_value_score_compatibility(tf, adapter)
    current = {
        "base_adapter_signature": bridge.adapter_signature(),
        "transformed_adapter_signature": adapter.adapter_signature(),
        "fixed_transport_manifest_hash": adapter.transport_manifest_hash,
        "transformed_adapter_signature_payload": adapter.adapter_signature_payload(),
        "fixed_transport_manifest": adapter.manifest_payload()[
            "fixed_transport_manifest"
        ],
    }
    historical = {
        "base_adapter_signature": kernel.get("base_adapter_signature"),
        "transformed_adapter_signature": kernel.get("transformed_adapter_signature"),
        "fixed_transport_manifest_hash": kernel.get("fixed_transport_manifest_hash"),
    }
    historical_identity_exact = (
        current["transformed_adapter_signature"]
        == historical["transformed_adapter_signature"]
        and current["fixed_transport_manifest_hash"]
        == historical["fixed_transport_manifest_hash"]
    )
    bindings = {
        **dict(provenance),
        "current_identity": current,
        "historical_identity": historical,
        "historical_identity_exact": historical_identity_exact,
        "identity_drift_classification": (
            "historical_commit_checkpoint_and_archived_state_numerical_parity_"
            "with_unreproduced_dirty_run_manifest_hash"
            if CODE_ROOT != ROOT
            else "live_tree_schema_drift_with_archived_state_numerical_parity"
        ),
        "archived_value_score_compatibility": parity,
        "kernel": dict(kernel),
        "kernel_hash": _stable_hash(kernel),
        "source_sha256": {
            "plan": _sha(PLAN),
            "original_validation_plan": _sha(ORIGINAL_VALIDATION_PLAN),
            "runner": _sha(RUNNER),
            "map_artifact": _sha(MAP_ARTIFACT),
            "tuning_artifact": _sha(TUNING_ARTIFACT),
            "sequential_summary": _sha(SEQUENTIAL_SUMMARY),
            "original_launch": _sha(ORIGINAL_LAUNCH),
            "archived_parity_receipt": _sha(ARCHIVED_PARITY_RECEIPT),
            "code_root": CODE_ROOT.as_posix(),
            "code_commit": subprocess.run(
                ("git", "rev-parse", "HEAD"), cwd=CODE_ROOT, check=True,
                capture_output=True, text=True,
            ).stdout.strip(),
            "custom_symmetric_sylvester_op_sha256": _sha(
                Path("bayesfilter/ops/_symmetric_sylvester_ops.so")
            ),
        },
    }
    return tf, bridge, transport, adapter, bindings


def _require_valid_status(tf: Any, status: Mapping[str, Any], label: str) -> None:
    code = tf.convert_to_tensor(status["status_code"], tf.int32)
    valid = tf.convert_to_tensor(status["valid_pre_regularized_score"], tf.bool)
    if not bool(tf.reduce_all(tf.equal(code, 0))) or not bool(tf.reduce_all(valid)):
        raise RootCauseDiagnosticError(f"{label} target status is invalid")


def _evaluate(
    tf: Any, evaluator: Any, points: Any, label: str
) -> tuple[Any, Any]:
    value, score, status = evaluator.log_prob_and_grad_status(points)
    value = tf.convert_to_tensor(value, tf.float64)
    score = tf.convert_to_tensor(score, tf.float64)
    _require_valid_status(tf, status, label)
    if not bool(tf.reduce_all(tf.math.is_finite(value))) or not bool(
        tf.reduce_all(tf.math.is_finite(score))
    ):
        raise RootCauseDiagnosticError(f"{label} value/score is nonfinite")
    return value, score


def _curvature(
    tf: Any,
    evaluator: Any,
    center: Any,
    center_log_prob: float,
    label: str,
) -> dict[str, Any]:
    records = []
    precisions = []
    for step in CURVATURE_STEPS:
        offsets = tf.eye(PARAMETER_DIM, dtype=tf.float64) * tf.constant(step, tf.float64)
        points = tf.concat((center[tf.newaxis, :] + offsets, center[tf.newaxis, :] - offsets), axis=0)
        _value, score = _evaluate(tf, evaluator, points, f"{label}-hessian-{step}")
        hessian = tf.transpose((score[:PARAMETER_DIM] - score[PARAMETER_DIM:]) / (2.0 * step))
        precision = -0.5 * (hessian + tf.transpose(hessian))
        if not bool(tf.reduce_all(tf.math.is_finite(precision))):
            raise RootCauseDiagnosticError(f"{label} precision is nonfinite")
        eigenvalues = tf.linalg.eigvalsh(precision)
        records.append(
            {
                "step": step,
                "precision": precision,
                "eigenvalues": eigenvalues,
                "spd": bool(tf.reduce_all(eigenvalues > 0.0)),
            }
        )
        precisions.append(precision)
    denominator = tf.maximum(
        tf.linalg.norm(precisions[-1]), tf.constant(1.0e-300, tf.float64)
    )
    stability = float(
        (tf.linalg.norm(precisions[-1] - precisions[-2]) / denominator).numpy()
    )
    final_eigenvalues = tf.linalg.eigvalsh(precisions[-1])
    spd = bool(tf.reduce_all(final_eigenvalues > 0.0))
    stable = stability <= CURVATURE_STABILITY_TOLERANCE
    laplace_log_mass = None
    if spd and stable:
        laplace_log_mass = float(
            (
                tf.constant(center_log_prob, tf.float64)
                + 0.5 * PARAMETER_DIM * tf.math.log(tf.constant(2.0 * math.pi, tf.float64))
                - 0.5 * tf.reduce_sum(tf.math.log(final_eigenvalues))
            ).numpy()
        )
    return {
        "records": records,
        "last_two_relative_delta": stability,
        "stability_tolerance": CURVATURE_STABILITY_TOLERANCE,
        "stable": stable,
        "spd": spd,
        "laplace_log_mass": laplace_log_mass,
        "laplace_role": "explanatory_local_quadratic_approximation_only",
    }


def _optimize_transformed(tf: Any, adapter: Any, initial_z: Any) -> dict[str, Any]:
    import tensorflow_probability as tfp

    def objective(position: Any) -> tuple[Any, Any]:
        value, score = adapter.log_prob_and_grad(position)
        return -tf.convert_to_tensor(value, tf.float64), -tf.convert_to_tensor(score, tf.float64)

    compiled = tf.function(
        objective,
        input_signature=(tf.TensorSpec((2, PARAMETER_DIM), tf.float64),),
        jit_compile=True,
        reduce_retracing=False,
    )
    result = tfp.optimizer.lbfgs_minimize(
        compiled,
        initial_position=initial_z,
        max_iterations=200,
        tolerance=tf.constant(1.0e-10, tf.float64),
        parallel_iterations=1,
    )
    endpoints = tf.convert_to_tensor(result.position, tf.float64)
    values, scores = _evaluate(tf, adapter, endpoints, "transformed-optimizer-endpoints")
    rows = {}
    for index, label in enumerate(("plus", "minus")):
        score_inf = float(tf.reduce_max(tf.abs(scores[index])).numpy())
        rows[label] = {
            "z": endpoints[index],
            "log_prob_z": values[index],
            "score_z": scores[index],
            "score_inf_norm": score_inf,
            "stationary_usable": score_inf <= STATIONARY_SCORE_TOLERANCE,
            "converged": bool(optimizer_result_item(tf, result.converged, index)),
            "failed": bool(optimizer_result_item(tf, result.failed, index)),
            "iterations": int(
                optimizer_result_item(tf, result.num_iterations, index)
            ),
        }
    separation = float(tf.linalg.norm(endpoints[0] - endpoints[1]).numpy())
    return {
        "initial_z": initial_z,
        "endpoints": rows,
        "endpoint_distance_l2": separation,
        "distinct_at_1e_4": separation > 1.0e-4,
        "all_stationary_usable": all(row["stationary_usable"] for row in rows.values()),
        "note": "source-coordinate stationary points need not remain stationary after the log-Jacobian correction",
    }


def _path_profile(tf: Any, adapter: Any, transport: Any, plus_z: Any, minus_z: Any, plus_theta: Any, minus_theta: Any) -> dict[str, Any]:
    fraction = tf.linspace(tf.constant(0.0, tf.float64), tf.constant(1.0, tf.float64), PATH_POINT_COUNT)[:, tf.newaxis]
    latent_line = plus_z[tf.newaxis, :] + fraction * (minus_z - plus_z)[tf.newaxis, :]
    theta_line = plus_theta[tf.newaxis, :] + fraction * (minus_theta - plus_theta)[tf.newaxis, :]
    source_line_z = transport.inverse_theta_to_z_batch(theta_line)
    output = {}
    for label, points in (("latent_straight", latent_line), ("source_straight_inverse_mapped", source_line_z)):
        value, _score = _evaluate(tf, adapter, points, f"path-{label}")
        potential = -value
        peak = tf.reduce_max(potential)
        rise_plus = max(0.0, float((peak - potential[0]).numpy()))
        rise_minus = max(0.0, float((peak - potential[-1]).numpy()))
        output[label] = {
            "z": points,
            "log_prob_z": value,
            "sampled_potential": potential,
            "sampled_peak_index": int(tf.argmax(potential).numpy()),
            "sampled_rise_from_plus": rise_plus,
            "sampled_rise_from_minus": rise_minus,
            "kinetic_availability_4d_from_plus": identity_mass_kinetic_survival_4d(rise_plus),
            "kinetic_availability_4d_from_minus": identity_mass_kinetic_survival_4d(rise_minus),
        }
    return {
        "point_count": PATH_POINT_COUNT,
        "paths": output,
        "role": "sampled_path_profile_heuristic_only",
        "nonclaims": [
            "finite grid can miss a narrow peak",
            "neither path is an optimized minimum-energy path",
            "sampled rise is not a rigorous bound on the minimum continuous barrier",
        ],
    }


def _run_manifest(mode: str, started: float, tf: Any, bindings: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "git_commit": subprocess.run(
            ("git", "rev-parse", "HEAD"), cwd=ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip(),
        "git_dirty": bool(
            subprocess.run(
                ("git", "status", "--porcelain"), cwd=ROOT, check=True,
                capture_output=True, text=True,
            ).stdout.strip()
        ),
        "command": " ".join(sys.argv),
        "mode": mode,
        "environment": os.environ.get("CONDA_DEFAULT_ENV", "unknown"),
        "python": platform.python_version(),
        "tensorflow": tf.__version__,
        "cpu_gpu_status": "CPU_ONLY_GPU_HIDDEN",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cpu_threads": THREADS,
        "jit_compile": True,
        "wall_time_seconds": time.perf_counter() - started,
        "cap_seconds": CAMPAIGN_CAP_SECONDS,
        "artifact_root": OUTPUT_ROOT.as_posix(),
        "plan_file": PLAN.as_posix(),
        "result_file": (
            "docs/plans/"
            "bayesfilter-ssl-lstm-q20-seed-b-neutra-mode-failure-root-cause-result-2026-08-10.md"
        ),
        "random_seeds": {
            "flow": FLOW_SEED,
            "split_canary": CANARY_SEED,
            "split_material": MATERIAL_SEED,
        },
        "data_version": bindings["source_sha256"],
    }


def run_geometry() -> dict[str, Any]:
    started = time.perf_counter()
    tf, bridge, transport, adapter, bindings = _configure_and_build()
    map_payload = _read_json(MAP_ARTIFACT)
    representatives = select_sign_representatives(map_payload.get("starts", []))
    theta = tf.constant(
        [representatives[label]["position"] for label in ("plus", "minus")],
        tf.float64,
    )
    z = tf.convert_to_tensor(transport.inverse_theta_to_z_batch(theta), tf.float64)
    roundtrip = tf.convert_to_tensor(transport.forward_z_to_theta_batch(z), tf.float64)
    roundtrip_error = tf.reduce_max(tf.abs(roundtrip - theta), axis=1)
    if not bool(tf.reduce_all(roundtrip_error <= ROUNDTRIP_TOLERANCE)):
        raise RootCauseDiagnosticError("representative transform round trip failed")
    theta_value, theta_score = _evaluate(tf, bridge, theta, "source-representatives")
    z_value, z_score = _evaluate(tf, adapter, z, "inverse-mapped-representatives")
    logdet = tf.convert_to_tensor(transport.log_abs_det_jacobian_batch(z), tf.float64)
    base_log_prob = -0.5 * tf.reduce_sum(tf.square(z), axis=1) - 0.5 * PARAMETER_DIM * tf.math.log(tf.constant(2.0 * math.pi, tf.float64))
    proposal_log_prob_theta = base_log_prob - logdet

    @tf.function(
        input_signature=(tf.TensorSpec((FLOW_SAMPLE_COUNT, PARAMETER_DIM), tf.float64),),
        jit_compile=True,
        reduce_retracing=False,
    )
    def map_flow(base: Any) -> Any:
        return transport.forward_z_to_theta_batch(base)

    base_draws = tf.random.stateless_normal(
        (FLOW_SAMPLE_COUNT, PARAMETER_DIM), seed=tf.constant(FLOW_SEED, tf.int32), dtype=tf.float64
    )
    flow_theta = tf.convert_to_tensor(map_flow(base_draws), tf.float64)
    flow_negative = int(tf.reduce_sum(tf.cast(flow_theta[:, OBSERVATION_WEIGHT_INDEX] < 0.0, tf.int64)).numpy())
    flow_positive = FLOW_SAMPLE_COUNT - flow_negative
    flow_coverage = {
        "sample_count": FLOW_SAMPLE_COUNT,
        "seed": FLOW_SEED,
        "positive_count": flow_positive,
        "negative_count": flow_negative,
        "positive_fraction": flow_positive / FLOW_SAMPLE_COUNT,
        "negative_fraction": flow_negative / FLOW_SAMPLE_COUNT,
        "negative_zero_count_upper_95": (
            zero_count_upper_bound(FLOW_SAMPLE_COUNT) if flow_negative == 0 else None
        ),
        "observation_weight_minimum": tf.reduce_min(flow_theta[:, OBSERVATION_WEIGHT_INDEX]),
        "observation_weight_maximum": tf.reduce_max(flow_theta[:, OBSERVATION_WEIGHT_INDEX]),
        "role": "learned_reverse_kl_proposal_q_phi_only",
        "nonclaim": "not posterior basin mass",
    }
    transformed = _optimize_transformed(tf, adapter, z)
    source_curvature = {}
    transformed_curvature = {}
    for index, label in enumerate(("plus", "minus")):
        source_curvature[label] = _curvature(
            tf, bridge, theta[index], float(theta_value[index]), f"source-{label}"
        )
        endpoint = transformed["endpoints"][label]
        if endpoint["stationary_usable"]:
            transformed_curvature[label] = _curvature(
                tf,
                adapter,
                tf.convert_to_tensor(endpoint["z"], tf.float64),
                float(endpoint["log_prob_z"]),
                f"transformed-{label}",
            )
        else:
            transformed_curvature[label] = {
                "available": False,
                "reason": "transformed optimizer endpoint not stationary at declared tolerance",
            }
    laplace_logs = [source_curvature[label]["laplace_log_mass"] for label in ("plus", "minus")]
    source_laplace = {"available": all(value is not None for value in laplace_logs)}
    if source_laplace["available"]:
        logs = tf.constant(laplace_logs, tf.float64)
        weights = tf.nn.softmax(logs)
        source_laplace.update(
            {
                "log_local_masses": {"plus": logs[0], "minus": logs[1]},
                "two_mode_normalized_fractions": {"plus": weights[0], "minus": weights[1]},
                "role": "local_laplace_explanatory_only_assuming_these_two_modes",
            }
        )
    paths = _path_profile(
        tf, adapter, transport, z[0], z[1], theta[0], theta[1]
    )
    original_launch = _read_json(ORIGINAL_LAUNCH)
    original_z = tf.constant(original_launch["initial_z"], tf.float64)
    original_theta = transport.forward_z_to_theta_batch(original_z)
    original_signs = tf.where(original_theta[:, OBSERVATION_WEIGHT_INDEX] < 0.0, "minus", "plus")
    original_distances = tf.stack(
        (tf.linalg.norm(original_z - z[0], axis=1), tf.linalg.norm(original_z - z[1], axis=1)), axis=1
    )
    payload = {
        "schema": "bayesfilter.ssl_lstm.q20_seed_b_neutra_mode_failure_geometry.v1",
        "status": "GEOMETRY_DIAGNOSTIC_COMPLETED",
        "question": "why did seed-B NeuTra plus fixed HMC omit the negative observation-weight region",
        "representatives": representatives,
        "inverse_mapped_representatives": {
            label: {
                "theta": theta[index],
                "z": z[index],
                "roundtrip_max_abs_error": roundtrip_error[index],
                "log_prob_theta": theta_value[index],
                "score_theta_inf_norm": tf.reduce_max(tf.abs(theta_score[index])),
                "log_abs_det_jacobian": logdet[index],
                "log_prob_z": z_value[index],
                "score_z": z_score[index],
                "score_z_inf_norm": tf.reduce_max(tf.abs(z_score[index])),
                "standard_normal_log_prob_z": base_log_prob[index],
                "proposal_log_prob_theta": proposal_log_prob_theta[index],
                "unnormalized_log_p_minus_log_q": theta_value[index] - proposal_log_prob_theta[index],
            }
            for index, label in enumerate(("plus", "minus"))
        },
        "flow_proposal_coverage": flow_coverage,
        "transformed_optimization": transformed,
        "source_curvature": source_curvature,
        "transformed_curvature": transformed_curvature,
        "source_two_mode_laplace": source_laplace,
        "sampled_path_profiles": paths,
        "original_material_starts": {
            "z": original_z,
            "theta": original_theta,
            "observation_weight_sign": original_signs,
            "distance_to_inverse_mapped_representatives_l2_columns_plus_minus": original_distances,
        },
        "bindings": bindings,
        "run_manifest": _run_manifest("geometry", started, tf, bindings),
        "nonclaims": [
            "no exact posterior basin weights",
            "no exhaustive mode discovery",
            "flow proposal coverage is not posterior coverage",
            "Laplace masses are local approximations",
            "sampled paths are not minimum-barrier solutions or rigorous barrier bounds",
        ],
    }
    if time.perf_counter() - started > CAMPAIGN_CAP_SECONDS:
        raise RootCauseDiagnosticError("geometry wall-time cap breached")
    _write_json(GEOMETRY_ARTIFACT, payload)
    return payload


def transition_summary(initial_sign: int, signs: list[int]) -> dict[str, Any]:
    previous = int(initial_sign)
    plus_to_minus = 0
    minus_to_plus = 0
    for current in signs:
        current = int(current)
        if previous == 0 and current == 1:
            plus_to_minus += 1
        elif previous == 1 and current == 0:
            minus_to_plus += 1
        previous = current
    return {
        "plus_to_minus": plus_to_minus,
        "minus_to_plus": minus_to_plus,
        "any_opposite_region": any(int(value) != int(initial_sign) for value in signs),
        "terminal_sign": previous,
    }


def run_split(mode: str) -> dict[str, Any]:
    if mode not in {
        "split-canary",
        "split-material",
        "stationary-canary",
        "stationary-step-control",
    }:
        raise ValueError("invalid split mode")
    if not _abs(GEOMETRY_ARTIFACT).exists():
        raise RootCauseDiagnosticError("geometry artifact is required before split HMC")
    started = time.perf_counter()
    tf, _bridge, transport, adapter, bindings = _configure_and_build()
    geometry = _read_json(GEOMETRY_ARTIFACT)
    geometry_sha = _sha(GEOMETRY_ARTIFACT)
    if geometry.get("status") != "GEOMETRY_DIAGNOSTIC_COMPLETED":
        raise RootCauseDiagnosticError("geometry artifact is not complete")
    inverse = geometry["inverse_mapped_representatives"]
    if mode in {"stationary-canary", "stationary-step-control"}:
        endpoints = geometry["transformed_optimization"]["endpoints"]
        if not geometry["transformed_optimization"]["all_stationary_usable"]:
            raise RootCauseDiagnosticError("transformed stationary endpoints are unavailable")
        plus_z = tf.constant(endpoints["plus"]["z"], tf.float64)
        minus_z = tf.constant(endpoints["minus"]["z"], tf.float64)
        initialization = "exact_transformed_stationary_representatives"
    else:
        plus_z = tf.constant(inverse["plus"]["z"], tf.float64)
        minus_z = tf.constant(inverse["minus"]["z"], tf.float64)
        initialization = "inverse_mapped_sign_separated_source_representatives"
    if mode == "split-canary":
        per_region = CANARY_CHAINS_PER_REGION
        transitions = CANARY_TRANSITIONS
        seed = CANARY_SEED
        role = "mechanics_and_cost_canary_only"
    elif mode in {"stationary-canary", "stationary-step-control"}:
        per_region = STATIONARY_CHAINS_PER_REGION
        transitions = STATIONARY_TRANSITIONS
        seed = STATIONARY_SEED
        role = (
            "curvature_derived_step_size_causal_control"
            if mode == "stationary-step-control"
            else "transformed_stationary_fixed_kernel_localization_canary"
        )
    else:
        per_region = MATERIAL_CHAINS_PER_REGION
        transitions = MATERIAL_TRANSITIONS
        seed = MATERIAL_SEED
        role = "fixed_kernel_split_start_root_cause_diagnostic"
    initial = tf.concat(
        (
            tf.repeat(plus_z[tf.newaxis, :], per_region, axis=0),
            tf.repeat(minus_z[tf.newaxis, :], per_region, axis=0),
        ),
        axis=0,
    )
    kernel = bindings["kernel"]
    effective_step_size = (
        STATIONARY_CONTROL_STEP_SIZE
        if mode == "stationary-step-control"
        else float(kernel["step_size"])
    )
    import tensorflow_probability as tfp
    from bayesfilter.inference.batched_value_score import reviewed_value_score_target_fn

    target_fn = reviewed_value_score_target_fn(
        adapter, dtype=tf.float64, require_batched=True
    )
    hmc = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=target_fn,
        step_size=tf.constant(effective_step_size, tf.float64),
        num_leapfrog_steps=int(kernel["num_leapfrog_steps"]),
    )

    def trace_fn(_state: Any, results: Any) -> Mapping[str, Any]:
        return {
            "is_accepted": results.is_accepted,
            "log_accept_ratio": results.log_accept_ratio,
            "target_log_prob": results.accepted_results.target_log_prob,
            "proposed_target_log_prob": results.proposed_results.target_log_prob,
            "target_score": results.accepted_results.grads_target_log_prob[0],
        }

    @tf.function(jit_compile=True, reduce_retracing=False)
    def sample() -> Any:
        return tfp.mcmc.sample_chain(
            num_results=transitions,
            num_burnin_steps=0,
            current_state=initial,
            kernel=hmc,
            trace_fn=trace_fn,
            seed=tf.constant(seed, tf.int32),
        )

    samples, trace = sample()
    samples = tf.convert_to_tensor(samples, tf.float64)
    expected_shape = (transitions, 2 * per_region, PARAMETER_DIM)
    if tuple(samples.shape) != expected_shape or not bool(
        tf.reduce_all(tf.math.is_finite(samples))
    ):
        raise RootCauseDiagnosticError("split HMC sample tensor is invalid")
    flat_theta = transport.forward_z_to_theta_batch(tf.reshape(samples, (-1, PARAMETER_DIM)))
    theta = tf.reshape(flat_theta, expected_shape)
    signs = tf.cast(theta[:, :, OBSERVATION_WEIGHT_INDEX] < 0.0, tf.int32)
    chain_rows = []
    for chain in range(2 * per_region):
        initial_sign = 0 if chain < per_region else 1
        row = transition_summary(initial_sign, [int(value) for value in signs[:, chain]])
        row.update(
            {
                "chain": chain,
                "initial_region": "plus" if initial_sign == 0 else "minus",
                "negative_draw_count": int(tf.reduce_sum(signs[:, chain]).numpy()),
            }
        )
        chain_rows.append(row)
    plus_to_minus = sum(row["plus_to_minus"] for row in chain_rows)
    minus_to_plus = sum(row["minus_to_plus"] for row in chain_rows)
    plus_chains_reached_minus = sum(
        row["any_opposite_region"] for row in chain_rows[:per_region]
    )
    minus_chains_reached_plus = sum(
        row["any_opposite_region"] for row in chain_rows[per_region:]
    )
    label = {
        "split-canary": "canary",
        "split-material": "material",
        "stationary-canary": "stationary-canary",
        "stationary-step-control": "stationary-step-control",
    }[mode]
    sample_receipt = _write_tensor(OUTPUT_ROOT / f"split-{label}-samples.tftensor", samples, tf)
    sign_receipt = _write_tensor(OUTPUT_ROOT / f"split-{label}-signs.tftensor", signs, tf)
    accepted = tf.cast(trace["is_accepted"], tf.float64)
    log_accept = tf.convert_to_tensor(trace["log_accept_ratio"], tf.float64)
    accepted_receipt = _write_tensor(
        OUTPUT_ROOT / f"split-{label}-is-accepted.tftensor",
        tf.convert_to_tensor(trace["is_accepted"], tf.bool),
        tf,
    )
    log_accept_receipt = _write_tensor(
        OUTPUT_ROOT / f"split-{label}-log-accept-ratio.tftensor", log_accept, tf
    )
    proposed_receipt = _write_tensor(
        OUTPUT_ROOT / f"split-{label}-proposed-target-log-prob.tftensor",
        tf.convert_to_tensor(trace["proposed_target_log_prob"], tf.float64),
        tf,
    )
    trace_finite = bool(
        tf.reduce_all(tf.math.is_finite(log_accept))
        and tf.reduce_all(tf.math.is_finite(trace["target_log_prob"]))
        and tf.reduce_all(tf.math.is_finite(trace["target_score"]))
    )
    if not trace_finite:
        raise RootCauseDiagnosticError("split HMC trace contains nonfinite values")
    payload = {
        "schema": "bayesfilter.ssl_lstm.q20_seed_b_neutra_split_start_hmc.v1",
        "status": "SPLIT_START_HMC_COMPLETED",
        "role": role,
        "mode": mode,
        "initialization": initialization,
        "chains_per_region": per_region,
        "transitions_per_chain": transitions,
        "total_transition_opportunities": 2 * per_region * transitions,
        "seed": seed,
        "kernel": {
            "step_size": effective_step_size,
            "source_frozen_step_size": kernel["step_size"],
            "num_leapfrog_steps": kernel["num_leapfrog_steps"],
            "mass_policy": kernel["mass_policy"],
            "use_xla": True,
            "step_size_role": (
                "curvature_derived_explanatory_control"
                if mode == "stationary-step-control"
                else "frozen_selected_kernel"
            ),
        },
        "transition_counts": {
            "plus_to_minus": plus_to_minus,
            "minus_to_plus": minus_to_plus,
            "plus_initialized_chains_reaching_minus": plus_chains_reached_minus,
            "minus_initialized_chains_reaching_plus": minus_chains_reached_plus,
        },
        "chain_rows": chain_rows,
        "acceptance": {
            "overall_binary_rate": tf.reduce_mean(accepted),
            "plus_initialized_binary_rate": tf.reduce_mean(accepted[:, :per_region]),
            "minus_initialized_binary_rate": tf.reduce_mean(accepted[:, per_region:]),
            "mean_acceptance_probability": tf.reduce_mean(
                tf.exp(tf.minimum(log_accept, 0.0))
            ),
            "maximum_absolute_log_accept_ratio": tf.reduce_max(tf.abs(log_accept)),
            "trace_all_finite": trace_finite,
            "native_divergence_status": "not_exposed_by_tfp_hmc_kernel",
        },
        "samples": sample_receipt,
        "signs": sign_receipt,
        "trace_receipts": {
            "is_accepted": accepted_receipt,
            "log_accept_ratio": log_accept_receipt,
            "proposed_target_log_prob": proposed_receipt,
        },
        "geometry_artifact": {
            "path": GEOMETRY_ARTIFACT.as_posix(),
            "sha256": geometry_sha,
        },
        "bindings": bindings,
        "run_manifest": _run_manifest(mode, started, tf, bindings),
        "interpretation_boundaries": [
            "conditioned split starts do not estimate stationary mode weights",
            "finite zero transitions do not prove crossing is impossible",
            "result diagnoses this frozen kernel, not all HMC kernels",
            "canary is mechanics and localization evidence only",
        ],
    }
    if time.perf_counter() - started > CAMPAIGN_CAP_SECONDS:
        raise RootCauseDiagnosticError("split HMC wall-time cap breached")
    _write_json(OUTPUT_ROOT / f"split-{label}.json", payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", required=True,
        choices=("geometry", "split-canary", "split-material", "stationary-canary")
        + ("stationary-step-control",)
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = run_geometry() if args.mode == "geometry" else run_split(args.mode)
    print(json.dumps({
        "status": payload["status"],
        "mode": args.mode,
        "artifact_root": OUTPUT_ROOT.as_posix(),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
