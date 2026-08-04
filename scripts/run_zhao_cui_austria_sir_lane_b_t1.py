#!/usr/bin/env python3
"""Run one bounded GPU/XLA Lane-B T1 pilot arm or untouched claim."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import tensorflow as tf  # noqa: E402

from bayesfilter.runtime.gpu_memory_policy import (  # noqa: E402
    configure_tensorflow_gpu_memory_growth,
)


MEMORY_POLICY = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)

from bayesfilter.highdim.stochastic_density_training import (  # noqa: E402
    TrainableFunctionalTT,
    make_adam_optimizer,
)
from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_artifact_compat import (  # noqa: E402
    COMPAT_DECODER_ID,
    load_lane_b_t1_artifact_v1_compat,
)
from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_target_tf import (  # noqa: E402
    generate_t1_proposal_cloud,
)
from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_tf import (  # noqa: E402
    BASELINE_ID,
    LaneBT1Settings,
    balanced_initial_cores,
    build_lane_b_frame,
    build_training_batch,
    calibrate_trainer_normalizer,
    estimate_shifted_log_normalizer,
    load_lane_b_t1_artifact,
    make_compiled_train_step,
    make_lane_b_t1_artifact,
    normalizer_estimates_agree,
    save_lane_b_t1_artifact,
    select_shift_constant,
    source_closure,
    trainer_config,
)


PLAN = Path(
    "docs/plans/"
    "bayesfilter-zhao-cui-austria-sir-lane-b-t1-execution-note-2026-07-30.md"
)
TRAIN_SEED = 73101
VALIDATION_SEED = 73201
CALIBRATION_SEED = 73301
UNTOUCHED_SEED = 73401
REFERENCE_SEED = 73501
TRAIN_COUNT = 8192
VALIDATION_COUNT = 16384
CALIBRATION_COUNT = 32768
UNTOUCHED_COUNT = 65536
MEMORY_CAP_BYTES = 6 * 1024**3

ARM_TABLE: Mapping[str, Mapping[str, Any]] = {
    "p01_r2_b3_lr3e4_l1_0": {
        "rank": 2, "basis_order": 1, "basis_num_elems": 2,
        "learning_rate": 3e-4, "l1_weight": 0.0,
    },
    "p02_r2_b3_lr3e4_l1_1e8": {
        "rank": 2, "basis_order": 1, "basis_num_elems": 2,
        "learning_rate": 3e-4, "l1_weight": 1e-8,
    },
    "p03_r4_b3_lr3e4_l1_1e9": {
        "rank": 4, "basis_order": 1, "basis_num_elems": 2,
        "learning_rate": 3e-4, "l1_weight": 1e-9,
    },
    "p04_r4_b3_lr3e4_l1_1e8": {
        "rank": 4, "basis_order": 1, "basis_num_elems": 2,
        "learning_rate": 3e-4, "l1_weight": 1e-8,
    },
    "p05_r4_b5_lr3e4_l1_1e9": {
        "rank": 4, "basis_order": 2, "basis_num_elems": 2,
        "learning_rate": 3e-4, "l1_weight": 1e-9,
    },
    "p06_r4_b5_lr1e4_l1_1e9": {
        "rank": 4, "basis_order": 2, "basis_num_elems": 2,
        "learning_rate": 1e-4, "l1_weight": 1e-9,
    },
}


def _settings(arm_id: str) -> LaneBT1Settings:
    if arm_id not in ARM_TABLE:
        raise ValueError(f"unknown pilot arm: {arm_id}")
    row = ARM_TABLE[arm_id]
    return LaneBT1Settings(
        arm_id=arm_id,
        rank=int(row["rank"]),
        basis_order=int(row["basis_order"]),
        basis_num_elems=int(row["basis_num_elems"]),
        learning_rate=float(row["learning_rate"]),
        l1_weight=float(row["l1_weight"]),
        l2_weight=1e-8,
        batch_size=512,
        train_steps=96,
        expansion_factor=4.0,
        covariance_jitter=1e-5,
        quantile_fraction=0.01,
        use_quantile_scale=True,
        tau=1e-8,
        gradient_clip_norm=100.0,
        cdf_grid_size=65,
        cdf_bisection_steps=24,
        cdf_max_working_bytes=512 * 1024 * 1024,
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, tf.Tensor):
        if value.shape.rank == 0:
            return _jsonable(value.numpy().item())
        return _jsonable(value.numpy().tolist())
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, float):
        if not (float("-inf") < value < float("inf")):
            return str(value)
        return value
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n")


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_manifest(command: list[str], started: float) -> Mapping[str, Any]:
    logical = tuple(tf.config.list_logical_devices("GPU"))
    if not logical:
        raise RuntimeError("Lane-B GPU run requires a logical GPU")
    return {
        "git_commit": _git_commit(),
        "command": command,
        "environment": sys.prefix,
        "host": platform.node(),
        "python": platform.python_version(),
        "tensorflow": tf.__version__,
        "device": tuple(device.name for device in logical),
        "dtype": "float64_reference_training",
        "tf32_enabled": bool(tf.config.experimental.tensor_float_32_execution_enabled()),
        "jit_compile": True,
        "gpu_memory_policy": dict(MEMORY_POLICY),
        "plan": PLAN.as_posix(),
        "random_seeds": {
            "training_frame": TRAIN_SEED,
            "validation": VALIDATION_SEED,
            "scale_calibration": CALIBRATION_SEED,
            "untouched_claim": UNTOUCHED_SEED,
            "frozen_references": REFERENCE_SEED,
            "trainer_initialization": 73001,
        },
        "trust_basis": "owner_designated_managed_session_visible_gpu_trusted",
        "source_sha256": {
            **dict(source_closure()),
            Path(__file__).resolve().relative_to(ROOT).as_posix(): _file_sha256(
                Path(__file__).resolve()
            ),
            PLAN.as_posix(): _file_sha256(ROOT / PLAN),
            "scripts/select_zhao_cui_austria_sir_lane_b_t1.py": _file_sha256(
                ROOT / "scripts/select_zhao_cui_austria_sir_lane_b_t1.py"
            ),
            "bayesfilter/highdim/zhao_cui_austria_sir_lane_b_artifact_compat.py": (
                _file_sha256(
                    ROOT
                    / "bayesfilter/highdim/"
                    "zhao_cui_austria_sir_lane_b_artifact_compat.py"
                )
            ),
        },
        "wall_time_seconds": time.monotonic() - started,
    }


def _compiled_validation_metric(trainer: TrainableFunctionalTT):
    @tf.function(jit_compile=True, reduce_retracing=True)
    def metric(
        points: tf.Tensor,
        target_sqrt: tf.Tensor,
        integration_weights: tf.Tensor,
        log_target: tf.Tensor,
        target_log_normalizer: tf.Tensor,
    ) -> tuple[tf.Tensor, ...]:
        del target_sqrt
        log_alpha = tf.math.log(integration_weights) + log_target
        alpha = tf.nn.softmax(log_alpha)
        candidate_log_density = tf.math.log(trainer.rho_theta(points)) - tf.math.log(
            trainer.normalizer()
        )
        target_log_density = log_target - target_log_normalizer
        candidate_rms = tf.sqrt(
            tf.reduce_sum(alpha * tf.square(candidate_log_density - target_log_density))
        )
        constant_rms = tf.sqrt(tf.reduce_sum(alpha * tf.square(target_log_density)))
        centered_candidate = candidate_log_density - tf.reduce_sum(
            alpha * candidate_log_density
        )
        centered_target = target_log_density - tf.reduce_sum(alpha * target_log_density)
        centered_rms = tf.sqrt(
            tf.reduce_sum(alpha * tf.square(centered_candidate - centered_target))
        )
        return candidate_rms, constant_rms, centered_rms, tf.reduce_min(alpha), tf.reduce_max(alpha)

    return metric


def _pilot(arm_id: str, output_dir: Path, max_seconds: float) -> Mapping[str, Any]:
    started = time.monotonic()
    settings = _settings(arm_id)
    training_cloud = generate_t1_proposal_cloud(
        sample_count=TRAIN_COUNT, seed=TRAIN_SEED, role="training_frame"
    )
    frame = build_lane_b_frame(training_cloud, settings)
    calibration_cloud = generate_t1_proposal_cloud(
        sample_count=CALIBRATION_COUNT, seed=CALIBRATION_SEED, role="calibration"
    )
    shift = select_shift_constant(calibration_cloud, frame)
    training = build_training_batch(training_cloud, frame, shift)
    validation_cloud = generate_t1_proposal_cloud(
        sample_count=VALIDATION_COUNT, seed=VALIDATION_SEED, role="validation"
    )
    validation = build_training_batch(validation_cloud, frame, shift)
    calibration_estimate = estimate_shifted_log_normalizer(calibration_cloud, shift)
    validation_estimate = estimate_shifted_log_normalizer(validation_cloud, shift)

    config = trainer_config(settings)
    initial_cores = balanced_initial_cores(settings, config.product_basis)
    trainer = TrainableFunctionalTT(config, initial_cores=initial_cores)
    initial_square_mass = trainer.sqrt_square_normalizer()
    initial_rho = trainer.rho_theta(training.points[: settings.batch_size])
    initial_mass_gate = bool(
        ((initial_square_mass >= 0.5) & (initial_square_mass <= 2.0)).numpy()
    )
    initial_variation_gate = bool(
        (tf.math.reduce_std(initial_rho) > tf.constant(1e-12, tf.float64)).numpy()
    )
    if not initial_mass_gate or not initial_variation_gate:
        raise RuntimeError("Lane-B balanced initialization gate failed")
    optimizer = make_adam_optimizer(config)
    compiled_step = make_compiled_train_step(trainer, optimizer)
    trace: list[Mapping[str, Any]] = []
    for step in range(settings.train_steps):
        if time.monotonic() - started > float(max_seconds):
            raise TimeoutError("pilot arm exceeded its predeclared wall-time cap")
        first = (step * settings.batch_size) % TRAIN_COUNT
        indices = tf.math.floormod(
            tf.range(first, first + settings.batch_size, dtype=tf.int32),
            tf.constant(TRAIN_COUNT, tf.int32),
        )
        values = compiled_step(
            tf.gather(training.points, indices),
            tf.gather(training.target_sqrt_values, indices),
            tf.gather(training.integration_weights, indices),
        )
        if step in {0, settings.train_steps - 1} or (step + 1) % 16 == 0:
            trace.append(
                {
                    "step": step + 1,
                    "total_loss": values[0],
                    "cross_entropy": values[1],
                    "log_normalizer_before_update": values[2],
                    "regularization": values[3],
                    "gradient_norm": values[4],
                    "rho_min": values[5],
                    "rho_max": values[6],
                }
            )

    pre_calibration_normalizer = trainer.normalizer()
    scale = calibrate_trainer_normalizer(
        trainer, calibration_estimate.log_shifted_normalizer
    )
    validation_metric = _compiled_validation_metric(trainer)
    metric_values = validation_metric(
        validation.points,
        validation.target_sqrt_values,
        validation.integration_weights,
        validation.log_target_reference,
        validation_estimate.log_shifted_normalizer,
    )
    generator = tf.random.Generator.from_seed(REFERENCE_SEED)
    frozen_references = generator.uniform([36, 16], dtype=tf.float64)
    normalizer_agreement = bool(
        normalizer_estimates_agree(calibration_estimate, validation_estimate).numpy()
    )
    shape_gate = bool((metric_values[0] <= 0.95 * metric_values[1]).numpy())
    memory = tf.config.experimental.get_memory_info("GPU:0")
    memory_gate = int(memory["peak"]) <= MEMORY_CAP_BYTES
    viable = normalizer_agreement and shape_gate and memory_gate
    artifact = None
    artifact_manifest = None
    if normalizer_agreement:
        artifact = make_lane_b_t1_artifact(
            settings=settings,
            frame=frame,
            trainer=trainer,
            shift_constant=shift,
            calibration_estimate=calibration_estimate,
            validation_estimate=validation_estimate,
            frozen_reference_points=frozen_references,
            training_cloud_manifest=training_cloud.manifest_payload(),
            validation_cloud_manifest=validation_cloud.manifest_payload(),
        )
        artifact_manifest = save_lane_b_t1_artifact(artifact, output_dir / "artifact")
    result = {
        "schema_version": "bayesfilter.zhao_cui_austria_sir_lane_b_t1_pilot.v1",
        "status": "VIABLE_T1_PILOT_ARM" if viable else "REJECTED_T1_PILOT_ARM",
        "baseline_id": BASELINE_ID,
        "arm": settings.manifest_payload(),
        "training_trace": trace,
        "pre_calibration_normalizer": pre_calibration_normalizer,
        "initialization": {
            "square_mass": initial_square_mass,
            "rho_min": tf.reduce_min(initial_rho),
            "rho_max": tf.reduce_max(initial_rho),
            "rho_stddev": tf.math.reduce_std(initial_rho),
            "mass_gate": initial_mass_gate,
            "variation_gate": initial_variation_gate,
        },
        "core_scale": scale,
        "calibration_estimate": calibration_estimate.manifest_payload(),
        "validation_estimate": validation_estimate.manifest_payload(),
        "validation_metrics": {
            "normalized_log_density_rms": metric_values[0],
            "constant_density_baseline_rms": metric_values[1],
            "centered_log_shape_rms": metric_values[2],
            "target_weight_min": metric_values[3],
            "target_weight_max": metric_values[4],
        },
        "gates": {
            "normalizer_agreement": normalizer_agreement,
            "shape_better_than_0p95_constant": shape_gate,
            "memory_under_6_gib": memory_gate,
            "viable": viable,
        },
        "artifact_manifest": (
            None if artifact_manifest is None else artifact_manifest.relative_to(ROOT).as_posix()
        ),
        "artifact_identity": None if artifact is None else artifact.identity.hash.value,
        "artifact_value": None if artifact is None else artifact.value(),
        "gpu_allocator": {key: int(value) for key, value in memory.items()},
        "run_manifest": _run_manifest(sys.argv, started),
        "inference_status": {
            "hard_veto_screen": "passed" if viable else "failed_current_arm",
            "statistically_supported_ranking": False,
            "descriptive_only_differences": True,
            "default_readiness": False,
            "next_evidence_needed": "all frozen pilot arms then untouched selected-arm claim",
        },
        "nonclaims": (
            "no statistically supported arm ranking",
            "no T1 admission from pilot data",
            "no score, T2, T20, HMC, or production claim",
        ),
    }
    return result


def _artifact_from_selection(selection_path: Path) -> tuple[Path, Mapping[str, Any]]:
    from scripts.select_zhao_cui_austria_sir_lane_b_t1 import build_selection

    selection = json.loads(selection_path.read_text())
    if selection.get("schema_version") != (
        "bayesfilter.zhao_cui_austria_sir_lane_b_t1_selection.v1"
    ):
        raise ValueError("Lane-B claim selection schema mismatch")
    if selection.get("status") != "SELECTED_VIABLE_T1_PILOT_ARM":
        raise ValueError("Lane-B claim requires a viable frozen selection")
    if selection.get("baseline_id") != BASELINE_ID:
        raise ValueError("Lane-B claim selection baseline mismatch")
    recomputed = build_selection(selection_path.parent)
    if json.dumps(selection, sort_keys=True) != json.dumps(recomputed, sort_keys=True):
        raise ValueError("Lane-B claim selection is not the deterministic recomputation")
    result_path = ROOT / str(selection["selected_result_path"])
    if _file_sha256(result_path) != selection.get("selected_result_sha256"):
        raise ValueError("Lane-B selected pilot result hash mismatch")
    result = json.loads(result_path.read_text())
    if result.get("status") != "VIABLE_T1_PILOT_ARM":
        raise ValueError("Lane-B selected pilot is not viable")
    if result.get("artifact_identity") != selection.get("selected_artifact_identity"):
        raise ValueError("Lane-B selected artifact identity mismatch")
    if result.get("artifact_manifest") != selection.get("selected_artifact_manifest"):
        raise ValueError("Lane-B selected artifact path mismatch")
    return (ROOT / str(result["artifact_manifest"])).parent, selection


def _claim(selection_path: Path, output_dir: Path) -> Mapping[str, Any]:
    started = time.monotonic()
    artifact_dir, selection = _artifact_from_selection(selection_path)
    artifact = load_lane_b_t1_artifact_v1_compat(artifact_dir)
    untouched_cloud = generate_t1_proposal_cloud(
        sample_count=UNTOUCHED_COUNT, seed=UNTOUCHED_SEED, role="untouched_claim"
    )
    estimate = estimate_shifted_log_normalizer(
        untouched_cloud, artifact.shift_constant
    )
    difference = tf.abs(
        tf.math.log(artifact.density().normalizer()) - estimate.log_shifted_normalizer
    )
    tolerance = 3.0 * tf.sqrt(
        tf.square(artifact.calibration_estimate.log_standard_error)
        + tf.square(estimate.log_standard_error)
    ) + tf.constant(1e-6, tf.float64)
    value_gate = bool((difference <= tolerance).numpy())
    memory = tf.config.experimental.get_memory_info("GPU:0")
    memory_gate = int(memory["peak"]) <= MEMORY_CAP_BYTES
    passed = value_gate and memory_gate
    return {
        "schema_version": "bayesfilter.zhao_cui_austria_sir_lane_b_t1_claim.v1",
        "status": "PASS_NEW_FIXED_VARIANT_T1_VALUE_BASELINE" if passed else "BLOCK_T1_UNTOUCHED_VALUE_GATE",
        "baseline_id": BASELINE_ID,
        "artifact_identity": artifact.identity.hash.value,
        "selection_sha256": _file_sha256(selection_path),
        "selected_arm_id": selection["selected_arm_id"],
        "artifact_reload_decoder_id": COMPAT_DECODER_ID,
        "artifact_value": artifact.value(),
        "untouched_log_evidence": estimate.log_evidence,
        "untouched_log_standard_error": estimate.log_standard_error,
        "shifted_normalizer_log_difference": difference,
        "shifted_normalizer_log_tolerance": tolerance,
        "gates": {
            "fresh_reload_identity": True,
            "untouched_value": value_gate,
            "memory_under_6_gib": memory_gate,
            "passed": passed,
        },
        "untouched_manifest": untouched_cloud.manifest_payload(),
        "gpu_allocator": {key: int(value) for key, value in memory.items()},
        "run_manifest": _run_manifest(sys.argv, started),
        "decision_table": {
            "decision": "admit_T1_and_open_T2_boundary" if passed else "repair_T1_without_reading_untouched_again",
            "primary_criterion_status": "passed" if value_gate else "failed",
            "veto_diagnostic_status": "passed" if memory_gate else "failed_memory",
            "main_uncertainty": "iid Monte Carlo standard error for untouched p(y1)",
            "next_justified_action": "B3 T2 previous-marginal boundary" if passed else "owner review of consumed untouched veto",
            "not_concluded": "no score, T2/T20, HMC, exact-likelihood theorem, or production readiness",
        },
        "inference_status": {
            "hard_veto_screen": "passed" if passed else "failed",
            "statistically_supported_ranking": False,
            "descriptive_only_differences": True,
            "default_readiness": False,
            "next_evidence_needed": "T2 boundary and separately planned T2 tuning",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pilot-arm", choices=tuple(ARM_TABLE))
    group.add_argument("--claim-selection", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-seconds", type=float, default=600.0)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    try:
        if args.pilot_arm is not None:
            result = _pilot(args.pilot_arm, output_dir, args.max_seconds)
        else:
            result = _claim(args.claim_selection.resolve(), output_dir)
        _write_json(output_dir / "result.json", result)
    except Exception as exc:
        failure = {
            "schema_version": "bayesfilter.zhao_cui_austria_sir_lane_b_failure.v1",
            "status": "INFRASTRUCTURE_OR_IMPLEMENTATION_FAILURE",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "command": sys.argv,
            "gpu_memory_policy": dict(MEMORY_POLICY),
        }
        _write_json(output_dir / "result.json", failure)
        raise


if __name__ == "__main__":
    main()
