#!/usr/bin/env python3
"""Sequential trusted GPU/XLA SSL-LSTM NeuTra training candidate runner."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import tensorflow as tf


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bayesfilter.inference.neutra_artifacts import (  # noqa: E402
    load_frozen_neutra_artifact,
)
from bayesfilter.inference.neutra_training import (  # noqa: E402
    NeuTraReverseKLTrainer,
    NeuTraTrainerConfig,
)
from bayesfilter.nonlinear.ssl_lstm_posterior_tf import (  # noqa: E402
    PRIOR_CENTER_VALUES,
    TARGET_SEMANTIC_SHA256,
    locked_ssl_lstm_posterior_target,
)


SCHEMA = "bayesfilter.ssl_lstm_neutra.phase4_bounded_training_candidate.v1"
PLAN_PATH = Path(
    "docs/plans/bayesfilter-ssl-lstm-neutra-phase-4-bounded-training-plan-2026-07-14.md"
)
RESULT_PATH = Path(
    "docs/plans/bayesfilter-ssl-lstm-neutra-phase-4-bounded-training-result-2026-07-14.md"
)
A0_LOCK_PATH = Path(
    "docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a0/target-lock.json"
)
A0_LOCK_SHA256 = "1f7fccbeafbaa344a80e77c73b4356f44258b78a65ea2499e8ebd194b79a4383"
GPU_BUDGET_SECONDS = 3600.0
FAILED_ATTEMPT_TRUST_BASIS = (
    "failure_receipt_only_gpu_provenance_not_established"
)
VALIDATION_BATCH_SIZE = 64
CHECKPOINT_EVERY = 100
VALIDATION_EVERY = 250
T_CRITICAL_ONE_SIDED_95_DF63 = 1.6694022215079607
INVERSE_RADIUS_MAX = 4.30
ROUNDTRIP_MAX_ABS = 1.0e-9
DENSE_SATURATION_FRACTION_MAX = 0.05
AFFINE_RAW_SCALE_MAX_ABS = math.log(10.0)
ORIGINAL_NEIGHBORHOOD_OFFSET = 0.10
MODERATE_SHELL_RADIUS = 2.0
FAR_TAIL_RADIUS = 4.0
PRIOR_PROBE_COUNT = 16
PRIOR_PROBE_SEED = (20260714, 3301)
A4_INITIAL_STATES = (
    (0.0, 0.0, 0.0, 0.0),
    (0.5, -0.5, 0.5, -0.5),
    (-0.5, 0.5, -0.5, 0.5),
    (0.5, 0.5, -0.5, -0.5),
)


class Phase4TrainingError(RuntimeError):
    """Raised when a Phase 4 candidate violates the prospective contract."""


@dataclass(frozen=True)
class CandidateSpec:
    label: str
    family: str
    role_code: int
    validation_role_code: int
    steps: int
    batch_size: int
    learning_rate: float
    hidden_layers: tuple[int, ...]
    activation: str = "tanh"
    s_max: float = 1.0
    repair: bool = False


CANDIDATES = {
    "affine_a": CandidateSpec(
        "affine_a", "affine_diag", 2101, 2201, 500, 64, 1.0e-3, (8, 8)
    ),
    "affine_b": CandidateSpec(
        "affine_b", "affine_diag", 2102, 2202, 500, 64, 1.0e-3, (8, 8)
    ),
    "dense_a": CandidateSpec(
        "dense_a", "dense_iaf", 2101, 2201, 2000, 64, 1.0e-3, (8, 8)
    ),
    "dense_b": CandidateSpec(
        "dense_b", "dense_iaf", 2102, 2202, 2000, 64, 1.0e-3, (8, 8)
    ),
    "repair_a": CandidateSpec(
        "repair_a", "dense_iaf", 2110, 2210, 2000, 64, 3.0e-4, (8, 8), repair=True
    ),
    "repair_b": CandidateSpec(
        "repair_b", "dense_iaf", 2111, 2211, 2000, 64, 3.0e-4, (8, 8), repair=True
    ),
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def _canonical_bytes(payload: Any) -> bytes:
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


def _write_json(path: Path, payload: Any) -> None:
    absolute = ROOT / path
    absolute.parent.mkdir(parents=True, exist_ok=True)
    if absolute.exists():
        raise Phase4TrainingError(f"output already exists: {path}")
    absolute.write_bytes(_canonical_bytes(payload))


def _strict_load(path: Path) -> dict[str, Any]:
    def reject(value: str) -> None:
        raise Phase4TrainingError(f"nonfinite JSON constant {value}: {path}")

    value = json.loads((ROOT / path).read_text(), parse_constant=reject)
    if not isinstance(value, dict):
        raise Phase4TrainingError(f"expected JSON object: {path}")
    return value


def _git(*args: str) -> str:
    return subprocess.check_output(("git", *args), cwd=ROOT, text=True).strip()


def _candidate_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "result": output_dir / "result.json",
        "failure": output_dir / "failure.json",
        "final_state": output_dir / "final-state.json",
        "frozen_payload": output_dir / "frozen-payload.json",
    }


def _require_fresh_directory(output_dir: Path) -> None:
    absolute = ROOT / output_dir
    if absolute.exists() and any(absolute.iterdir()):
        raise Phase4TrainingError(f"candidate output directory is not fresh: {output_dir}")
    absolute.mkdir(parents=True, exist_ok=True)


def _prior_budget(prior_results: tuple[Path, ...]) -> tuple[float, list[dict[str, Any]]]:
    total = 0.0
    rows = []
    for path in prior_results:
        payload = _strict_load(path)
        if payload.get("schema") != SCHEMA:
            raise Phase4TrainingError(f"prior result schema mismatch: {path}")
        manifest = payload.get("run_manifest")
        if not isinstance(manifest, dict):
            raise Phase4TrainingError(f"prior result manifest missing: {path}")
        seconds = float(manifest.get("charged_gpu_seconds", float("nan")))
        if not math.isfinite(seconds) or seconds < 0.0:
            raise Phase4TrainingError(f"invalid prior charged time: {path}")
        total += seconds
        rows.append(
            {
                "path": path.as_posix(),
                "sha256": _sha256(path),
                "candidate": payload.get("candidate", {}).get("label"),
                "charged_gpu_seconds": seconds,
                "decision": payload.get("decision"),
            }
        )
    if total >= GPU_BUDGET_SECONDS:
        raise Phase4TrainingError("Phase 4 GPU budget is already exhausted")
    return total, rows


def _trainer_config(spec: CandidateSpec) -> NeuTraTrainerConfig:
    return NeuTraTrainerConfig(
        dimension=4,
        family=spec.family,
        hidden_layers=spec.hidden_layers,
        activation=spec.activation,
        s_max=spec.s_max,
        initialization_scale=0.02,
        initialization_seed=(20260714, spec.role_code),
        learning_rate=spec.learning_rate,
        gradient_clip_norm=10.0,
        jit_compile=True,
    )


def _step_seed(role_code: int, step: int) -> tf.Tensor:
    root = tf.constant((20260714, int(role_code)), dtype=tf.int32)
    return tf.random.experimental.stateless_fold_in(root, int(step))


def _base_batch(spec: CandidateSpec, step: int) -> tf.Tensor:
    return tf.random.stateless_normal(
        (spec.batch_size, 4),
        seed=_step_seed(spec.role_code, step),
        dtype=tf.float64,
    )


def _validation_batch(spec: CandidateSpec) -> tf.Tensor:
    return tf.random.stateless_normal(
        (VALIDATION_BATCH_SIZE, 4),
        seed=tf.constant((20260714, spec.validation_role_code), dtype=tf.int32),
        dtype=tf.float64,
    )


def _host_step(result: Any) -> dict[str, Any]:
    row = {
        "step": int(result.step.numpy()),
        "loss": float(result.loss.numpy()),
        "surrogate": float(result.surrogate.numpy()),
        "target_value_mean": float(result.target_value_mean.numpy()),
        "logdet_mean": float(result.logdet_mean.numpy()),
        "gradient_norm": float(result.gradient_norm.numpy()),
        "clipped_gradient_norm": float(result.clipped_gradient_norm.numpy()),
        "clipping_applied": bool(result.clipping_applied.numpy()),
    }
    numeric = [value for key, value in row.items() if key not in {"step", "clipping_applied"}]
    if not all(math.isfinite(value) for value in numeric):
        raise Phase4TrainingError(f"nonfinite host-synchronized training row: {row}")
    return row


def _host_validation(validation: Any, *, family: str, s_max: float) -> dict[str, Any]:
    per_sample = [float(value) for value in validation.per_sample_loss.numpy().tolist()]
    target_value = [float(value) for value in validation.target_value.numpy().tolist()]
    theta = validation.theta.numpy().tolist()
    logdet = [float(value) for value in validation.logdet.numpy().tolist()]
    scale_log = validation.scale_log.numpy()
    flat = [*per_sample, *target_value, *logdet]
    flat.extend(float(value) for row in theta for value in row)
    flat.extend(float(value) for value in scale_log.reshape(-1).tolist())
    if not all(math.isfinite(value) for value in flat):
        raise Phase4TrainingError("nonfinite host-synchronized validation output")
    if family == "dense_iaf":
        saturation = float((abs(scale_log) >= 0.95 * float(s_max)).mean())
    else:
        saturation = 0.0
    return {
        "per_sample_loss": per_sample,
        "mean_loss": sum(per_sample) / len(per_sample),
        "target_value_mean": sum(target_value) / len(target_value),
        "logdet_mean": sum(logdet) / len(logdet),
        "scale_log_min": float(scale_log.min()),
        "scale_log_max": float(scale_log.max()),
        "saturation_fraction": saturation,
        "theta_min_by_coordinate": [min(row[index] for row in theta) for index in range(4)],
        "theta_max_by_coordinate": [max(row[index] for row in theta) for index in range(4)],
        "output_devices": sorted(
            {
                validation.per_sample_loss.device,
                validation.target_value.device,
                validation.theta.device,
                validation.logdet.device,
                validation.scale_log.device,
            }
        ),
    }


def paired_loss_upper_bound(initial: list[float], final: list[float]) -> dict[str, float]:
    if len(initial) != VALIDATION_BATCH_SIZE or len(final) != VALIDATION_BATCH_SIZE:
        raise Phase4TrainingError("heldout loss batches must have 64 paired rows")
    differences = [after - before for before, after in zip(initial, final)]
    mean = sum(differences) / len(differences)
    variance = sum((value - mean) ** 2 for value in differences) / (len(differences) - 1)
    standard_error = math.sqrt(variance / len(differences))
    upper = mean + T_CRITICAL_ONE_SIDED_95_DF63 * standard_error
    return {
        "mean_difference": mean,
        "standard_error": standard_error,
        "one_sided_95_upper": upper,
        "t_critical_df63": T_CRITICAL_ONE_SIDED_95_DF63,
    }


def _historical_geometry() -> tuple[tf.Tensor, tf.Tensor, str]:
    if _sha256(A0_LOCK_PATH) != A0_LOCK_SHA256:
        raise Phase4TrainingError("A0 target-lock byte identity drift")
    lock = _strict_load(A0_LOCK_PATH)
    if lock.get("signatures", {}).get("target_semantic_sha256") != TARGET_SEMANTIC_SHA256:
        raise Phase4TrainingError("A0 target semantic mismatch")
    geometry = lock.get("sampler_geometry")
    if not isinstance(geometry, dict):
        raise Phase4TrainingError("A0 sampler geometry missing")
    center = tf.constant(geometry["center_free"]["values"], tf.float64)
    scale = tf.constant(geometry["scale"]["values"], tf.float64)
    factor_z = tf.constant(geometry["factor_z"]["values"], tf.float64)
    factor = tf.linalg.diag(scale) @ factor_z
    return center, factor, str(lock["signatures"]["sampler_geometry_sha256"])


def _latent_rows(radius: float) -> tf.Tensor:
    rows = []
    for index in range(4):
        direction = [0.0] * 4
        direction[index] = float(radius)
        rows.extend((direction, [-value for value in direction]))
    return tf.constant(rows, tf.float64)


def _probe_bank() -> tuple[tf.Tensor, list[str], dict[str, Any]]:
    center, factor, geometry_signature = _historical_geometry()
    starts = tf.constant(A4_INITIAL_STATES, tf.float64)
    original_latent = []
    original_labels = []
    for start_index, start in enumerate(A4_INITIAL_STATES):
        original_latent.append(list(start))
        original_labels.append(f"original_start_{start_index}_center")
        for coordinate in range(4):
            for sign, suffix in ((1.0, "plus"), (-1.0, "minus")):
                row = list(start)
                row[coordinate] += sign * ORIGINAL_NEIGHBORHOOD_OFFSET
                original_latent.append(row)
                original_labels.append(
                    f"original_start_{start_index}_coordinate_{coordinate}_{suffix}"
                )
    original = center + tf.constant(original_latent, tf.float64) @ tf.transpose(factor)
    moderate_latent = _latent_rows(MODERATE_SHELL_RADIUS)
    moderate = center + moderate_latent @ tf.transpose(factor)
    moderate_labels = [f"moderate_shell_{index}" for index in range(8)]
    tail_latent = _latent_rows(FAR_TAIL_RADIUS)
    tail = center + tail_latent @ tf.transpose(factor)
    tail_labels = [f"far_tail_{index}" for index in range(8)]
    prior = tf.constant(PRIOR_CENTER_VALUES, tf.float64) + 4.0 * tf.random.stateless_normal(
        (PRIOR_PROBE_COUNT, 4),
        seed=tf.constant(PRIOR_PROBE_SEED, tf.int32),
        dtype=tf.float64,
    )
    prior_labels = [f"prior_probe_{index}" for index in range(PRIOR_PROBE_COUNT)]
    points = tf.concat((original, moderate, tail, prior), axis=0)
    labels = [*original_labels, *moderate_labels, *tail_labels, *prior_labels]
    metadata = {
        "historical_starts": [list(row) for row in A4_INITIAL_STATES],
        "historical_start_count": int(starts.shape[0]),
        "original_neighborhood_offset": ORIGINAL_NEIGHBORHOOD_OFFSET,
        "original_neighborhood_count": len(original_labels),
        "moderate_shell_radius": MODERATE_SHELL_RADIUS,
        "moderate_shell_count": len(moderate_labels),
        "far_tail_radius": FAR_TAIL_RADIUS,
        "far_tail_count": len(tail_labels),
        "prior_probe_count": PRIOR_PROBE_COUNT,
        "prior_probe_seed": list(PRIOR_PROBE_SEED),
        "prior_center": list(PRIOR_CENTER_VALUES),
        "prior_standard_deviation": 4.0,
        "sampler_geometry_sha256": geometry_signature,
        "point_count": len(labels),
    }
    return points, labels, metadata


def _probe_diagnostics(target: Any, frozen: Any) -> dict[str, Any]:
    points, labels, metadata = _probe_bank()
    z = frozen.inverse_theta_to_z_batch(points)
    replay = frozen.forward_batch(z)
    logdet = frozen.log_abs_det_jacobian_batch(z)
    target_value, target_score = target.batch_value_and_score(points)
    transformed_score = frozen.pullback_score_batch(z, target_score)
    transformed_score += frozen.log_abs_det_jacobian_score_batch(z)
    radii = tf.linalg.norm(z, axis=-1)
    roundtrip = tf.reduce_max(tf.abs(replay - points))
    tensors = (points, z, replay, logdet, target_value, target_score, transformed_score, radii)
    finite = all(bool(tf.reduce_all(tf.math.is_finite(value)).numpy()) for value in tensors)
    radius_values = [float(value) for value in radii.numpy().tolist()]
    original_end = metadata["original_neighborhood_count"]
    moderate_end = original_end + metadata["moderate_shell_count"]
    tail_end = moderate_end + metadata["far_tail_count"]
    rows = [
        {"label": label, "inverse_base_radius": radius}
        for label, radius in zip(labels, radius_values)
    ]
    return {
        "metadata": metadata,
        "all_finite": finite,
        "roundtrip_max_abs": float(roundtrip.numpy()),
        "original_neighborhood_max_inverse_radius": max(radius_values[:original_end]),
        "moderate_shell_max_inverse_radius": max(radius_values[original_end:moderate_end]),
        "far_tail_max_inverse_radius": max(radius_values[moderate_end:tail_end]),
        "prior_probe_max_inverse_radius": max(radius_values[tail_end:]),
        "rows": rows,
        "output_devices": sorted({value.device for value in tensors}),
    }


def _stable_state_equal(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return _canonical_bytes(left) == _canonical_bytes(right)


def _checkpoint_path(output_dir: Path, step: int) -> Path:
    return output_dir / f"checkpoint-{step:04d}.json"


def _source_bindings() -> list[dict[str, str]]:
    rows = (
        (Path(__file__).resolve().relative_to(ROOT), "phase4_candidate_runner"),
        (Path("bayesfilter/inference/neutra_training.py"), "trainer"),
        (Path("bayesfilter/inference/neutra_artifacts.py"), "frozen_transport"),
        (Path("bayesfilter/nonlinear/ssl_lstm_posterior_tf.py"), "locked_target"),
        (Path("tests/test_neutra_reverse_kl_training.py"), "trainer_tests"),
        (
            Path("tests/test_ssl_lstm_neutra_phase4_bounded_training.py"),
            "phase4_runner_tests",
        ),
        (PLAN_PATH, "prospective_plan"),
    )
    return [
        {"path": path.as_posix(), "role": role, "sha256": _sha256(path)}
        for path, role in rows
    ]


def _candidate_decision(
    *,
    spec: CandidateSpec,
    initial_validation: dict[str, Any],
    final_validation: dict[str, Any],
    loss_interval: dict[str, float],
    probes: dict[str, Any],
    affine_raw_scale_max_abs: float | None,
) -> tuple[str, list[str], list[str]]:
    hard_vetoes = []
    promotion_vetoes = []
    if not probes["all_finite"]:
        hard_vetoes.append("probe_nonfinite")
    if probes["roundtrip_max_abs"] > ROUNDTRIP_MAX_ABS:
        hard_vetoes.append("roundtrip_residual_above_threshold")
    if probes["original_neighborhood_max_inverse_radius"] > INVERSE_RADIUS_MAX:
        promotion_vetoes.append("original_neighborhood_missing_support")
    if probes["moderate_shell_max_inverse_radius"] > INVERSE_RADIUS_MAX:
        promotion_vetoes.append("moderate_shell_missing_support")
    if loss_interval["one_sided_95_upper"] >= 0.0:
        promotion_vetoes.append("heldout_loss_improvement_not_established")
    if spec.family == "dense_iaf":
        if final_validation["saturation_fraction"] > DENSE_SATURATION_FRACTION_MAX:
            promotion_vetoes.append("dense_scale_saturation_above_cap")
    elif affine_raw_scale_max_abs is None or affine_raw_scale_max_abs > AFFINE_RAW_SCALE_MAX_ABS:
        promotion_vetoes.append("affine_raw_scale_outside_cap")
    if hard_vetoes:
        return "INVALID_HARD_VETO", hard_vetoes, promotion_vetoes
    if promotion_vetoes:
        return "CANDIDATE_NOT_VIABLE", hard_vetoes, promotion_vetoes
    return "VIABLE_FROZEN_CANDIDATE", hard_vetoes, promotion_vetoes


def run_candidate(
    spec: CandidateSpec,
    *,
    output_dir: Path,
    prior_results: tuple[Path, ...],
) -> dict[str, Any]:
    _require_fresh_directory(output_dir)
    paths = _candidate_paths(output_dir)
    prior_seconds, prior_rows = _prior_budget(prior_results)
    started_at = _now()
    started = time.perf_counter()
    previous_soft_placement = tf.config.get_soft_device_placement()
    physical_gpus = tf.config.list_physical_devices("GPU")
    if not physical_gpus:
        raise Phase4TrainingError("trusted Phase 4 training requires a visible GPU")
    for gpu in physical_gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.experimental.enable_tensor_float_32_execution(True)
    tf.config.set_soft_device_placement(False)
    training_history = []
    validation_history = []
    checkpoint_rows = []
    resume_replay = None
    try:
        with tf.device("/GPU:0"):
            target = locked_ssl_lstm_posterior_target()
            config = _trainer_config(spec)
            trainer = NeuTraReverseKLTrainer(target, config)
            validation_z = _validation_batch(spec)
            initial_validation = _host_validation(
                trainer.validation_batch(validation_z),
                family=spec.family,
                s_max=spec.s_max,
            )
            validation_history.append({"step": 0, **initial_validation})

            for logical_step in range(1, spec.steps + 1):
                elapsed = time.perf_counter() - started
                if prior_seconds + elapsed >= GPU_BUDGET_SECONDS:
                    state = trainer.state_payload()
                    _write_json(_checkpoint_path(output_dir, int(trainer.step.numpy())), state)
                    raise Phase4TrainingError("Phase 4 GPU budget exhausted during candidate")
                z = _base_batch(spec, logical_step)
                if logical_step == CHECKPOINT_EVERY + 1:
                    before_replay = trainer.state_payload()
                    first_result = trainer.train_step(z)
                    first_row = _host_step(first_result)
                    expected_state = trainer.state_payload()
                    trainer.restore_state(before_replay)
                    replay_result = trainer.train_step(z)
                    replay_row = _host_step(replay_result)
                    replay_state = trainer.state_payload()
                    replay_passed = (
                        first_row == replay_row
                        and _stable_state_equal(expected_state, replay_state)
                    )
                    if not replay_passed:
                        raise Phase4TrainingError("material configuration resume replay mismatch")
                    resume_replay = {
                        "logical_step": logical_step,
                        "passed": True,
                        "pre_state_hash": before_replay["state_hash"],
                        "post_state_hash": replay_state["state_hash"],
                    }
                    row = replay_row
                else:
                    row = _host_step(trainer.train_step(z))
                training_history.append(row)
                if logical_step % CHECKPOINT_EVERY == 0:
                    state = trainer.state_payload()
                    path = _checkpoint_path(output_dir, logical_step)
                    _write_json(path, state)
                    checkpoint_rows.append(
                        {
                            "step": logical_step,
                            "path": path.as_posix(),
                            "sha256": _sha256(path),
                            "state_hash": state["state_hash"],
                        }
                    )
                if logical_step % VALIDATION_EVERY == 0:
                    validation = _host_validation(
                        trainer.validation_batch(validation_z),
                        family=spec.family,
                        s_max=spec.s_max,
                    )
                    validation_history.append({"step": logical_step, **validation})

            final_state = trainer.state_payload()
            _write_json(paths["final_state"], final_state)
            frozen_payload = trainer.frozen_transport_payload(
                transport_id=f"ssl-lstm-phase4-{spec.label}",
                target_signature=TARGET_SEMANTIC_SHA256,
            )
            _write_json(paths["frozen_payload"], frozen_payload)
            loaded = load_frozen_neutra_artifact(
                frozen_payload,
                expected_target_signature=TARGET_SEMANTIC_SHA256,
            )
            probes = _probe_diagnostics(target, loaded.transport)
            final_validation = validation_history[-1]
            loss_interval = paired_loss_upper_bound(
                initial_validation["per_sample_loss"],
                final_validation["per_sample_loss"],
            )
            affine_raw_scale_max_abs = None
            if spec.family == "affine_diag":
                affine_raw_scale_max_abs = max(
                    abs(float(value)) for value in final_state["variables"][1]
                )
            decision, hard_vetoes, promotion_vetoes = _candidate_decision(
                spec=spec,
                initial_validation=initial_validation,
                final_validation=final_validation,
                loss_interval=loss_interval,
                probes=probes,
                affine_raw_scale_max_abs=affine_raw_scale_max_abs,
            )
            output_devices = sorted(
                {
                    trainer.variables[0].device,
                    *initial_validation["output_devices"],
                    *final_validation["output_devices"],
                    *probes["output_devices"],
                }
            )
            if not output_devices or not all("GPU:" in device for device in output_devices):
                hard_vetoes.append("outputs_not_gpu_resident")
                decision = "INVALID_HARD_VETO"
    finally:
        tf.config.set_soft_device_placement(previous_soft_placement)

    wall_time = time.perf_counter() - started
    payload = {
        "schema": SCHEMA,
        "status": "COMPLETED",
        "decision": decision,
        "candidate": {
            "label": spec.label,
            "family": spec.family,
            "training_role_seed": [20260714, spec.role_code],
            "validation_role_seed": [20260714, spec.validation_role_code],
            "steps": spec.steps,
            "batch_size": spec.batch_size,
            "learning_rate": spec.learning_rate,
            "hidden_layers": list(spec.hidden_layers),
            "activation": spec.activation,
            "s_max": spec.s_max,
            "repair": spec.repair,
        },
        "hard_vetoes": hard_vetoes,
        "promotion_vetoes": promotion_vetoes,
        "training": {
            "history": training_history,
            "checkpoints": checkpoint_rows,
            "resume_replay": resume_replay,
            "final_state_path": paths["final_state"].as_posix(),
            "final_state_sha256": _sha256(paths["final_state"]),
            "final_state_hash": final_state["state_hash"],
        },
        "validation": {
            "history": validation_history,
            "paired_final_minus_initial": loss_interval,
        },
        "frozen_transport": {
            "path": paths["frozen_payload"].as_posix(),
            "sha256": _sha256(paths["frozen_payload"]),
            "artifact_signature": loaded.artifact_signature,
            "transport_hash": loaded.manifest.transport_hash,
            "topology_hash": loaded.manifest.topology_hash,
            "tensor_hash": loaded.manifest.tensor_hash,
            "training_state_hash": loaded.manifest.training_state_hash,
            "affine_raw_scale_max_abs": affine_raw_scale_max_abs,
        },
        "probe_diagnostics": probes,
        "thresholds": {
            "inverse_radius_max": INVERSE_RADIUS_MAX,
            "roundtrip_max_abs": ROUNDTRIP_MAX_ABS,
            "dense_saturation_fraction_max": DENSE_SATURATION_FRACTION_MAX,
            "affine_raw_scale_max_abs": AFFINE_RAW_SCALE_MAX_ABS,
            "heldout_loss_upper_bound_max": 0.0,
        },
        "source_files": _source_bindings(),
        "run_manifest": {
            "git_commit": _git("rev-parse", "HEAD"),
            "git_dirty": bool(_git("status", "--porcelain")),
            "command": " ".join(sys.argv),
            "environment": os.environ.get("CONDA_DEFAULT_ENV", "N/A"),
            "python": sys.version.split()[0],
            "tensorflow": tf.__version__,
            "physical_gpus": [device.name for device in physical_gpus],
            "output_devices": output_devices,
            "dtype": "float64",
            "tf32": bool(tf.config.experimental.tensor_float_32_execution_enabled()),
            "jit_compile": True,
            "soft_device_placement_during_run": False,
            "trust_basis": "owner_designated_managed_session_visible_gpu_trusted",
            "started_at_utc": started_at,
            "completed_at_utc": _now(),
            "wall_time_seconds": wall_time,
            "charged_gpu_seconds": wall_time,
            "prior_charged_gpu_seconds": prior_seconds,
            "cumulative_charged_gpu_seconds": prior_seconds + wall_time,
            "gpu_budget_seconds": GPU_BUDGET_SECONDS,
            "prior_results": prior_rows,
            "output_dir": output_dir.as_posix(),
            "plan_path": PLAN_PATH.as_posix(),
            "result_path": RESULT_PATH.as_posix(),
        },
        "evidence_contract": {
            "loss_role": "trainer_gate_not_transport_promotion_or_posterior_evidence",
            "prior_tail_role": "finiteness_and_repair_diagnostic_not_required_posterior_coverage",
            "candidate_pass_role": "nomination_for_exact_transformed_target_preflight_only",
        },
        "nonclaims": (
            "no posterior correctness or complete mode/tail coverage claim",
            "no HMC or sampler admission claim",
            "no predictive equivalence or scientific claim",
            "no candidate ranking or default-readiness claim",
        ),
    }
    _write_json(paths["result"], payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", choices=tuple(CANDIDATES), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prior-result", type=Path, action="append", default=[])
    args = parser.parse_args()
    spec = CANDIDATES[args.candidate]
    attempted_at = _now()
    attempted = time.perf_counter()
    try:
        payload = run_candidate(
            spec,
            output_dir=args.output_dir,
            prior_results=tuple(args.prior_result),
        )
    except Exception as error:
        failure_path = args.output_dir / "failure.json"
        if not (ROOT / failure_path).exists():
            try:
                prior_seconds, prior_rows = _prior_budget(tuple(args.prior_result))
            except Exception:
                prior_seconds, prior_rows = 0.0, []
            charged_seconds = time.perf_counter() - attempted
            _write_json(
                failure_path,
                {
                    "schema": SCHEMA,
                    "status": "FAILED",
                    "decision": "INVALID_HARD_VETO",
                    "candidate": {
                        "label": spec.label,
                        "family": spec.family,
                        "training_role_seed": [20260714, spec.role_code],
                        "validation_role_seed": [20260714, spec.validation_role_code],
                    },
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "created_at_utc": _now(),
                    "source_files": _source_bindings(),
                    "run_manifest": {
                        "git_commit": _git("rev-parse", "HEAD"),
                        "git_dirty": bool(_git("status", "--porcelain")),
                        "command": " ".join(sys.argv),
                        "started_at_utc": attempted_at,
                        "completed_at_utc": _now(),
                        "wall_time_seconds": charged_seconds,
                        "charged_gpu_seconds": charged_seconds,
                        "prior_charged_gpu_seconds": prior_seconds,
                        "cumulative_charged_gpu_seconds": prior_seconds
                        + charged_seconds,
                        "gpu_budget_seconds": GPU_BUDGET_SECONDS,
                        "prior_results": prior_rows,
                        "trust_basis": FAILED_ATTEMPT_TRUST_BASIS,
                        "output_dir": args.output_dir.as_posix(),
                        "plan_path": PLAN_PATH.as_posix(),
                        "result_path": RESULT_PATH.as_posix(),
                    },
                    "nonclaims": ["failed candidate artifact; no scientific conclusion"],
                },
            )
        raise
    print(f"{spec.label}: {payload['decision']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
