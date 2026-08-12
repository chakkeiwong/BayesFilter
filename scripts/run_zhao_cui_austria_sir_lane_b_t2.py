#!/usr/bin/env python3
"""Run one sealed GPU/XLA Lane-B T2 pilot arm."""

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
    load_lane_b_t1_artifact_v1_compat,
)
from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_t2_prepared_tf import (  # noqa: E402
    load_t2_prepared_cloud,
    prepared_estimate,
    prepared_source_closure,
)
from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_t2_tf import (  # noqa: E402
    T2_BASELINE_ID,
    build_t2_frame,
    build_t2_training_batch,
    estimate_t2_shifted_log_normalizer,
    load_lane_b_t2_artifact,
    make_lane_b_t2_artifact,
    make_t2_compiled_train_step,
    make_t2_compiled_validation_metric,
    save_lane_b_t2_artifact,
    t2_source_closure,
    t2_trainer_config,
)
from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_tf import (  # noqa: E402
    LaneBT1Settings,
    balanced_initial_cores,
    calibrate_trainer_normalizer,
    normalizer_estimates_agree,
)


PLAN = Path("docs/plans/bayesfilter-zhao-cui-austria-sir-lane-b-t2-plan-2026-07-31.md")
EXECUTION_NOTE = Path(
    "docs/plans/"
    "bayesfilter-zhao-cui-austria-sir-lane-b-t2-pilot-execution-note-2026-07-31.md"
)
MEMORY_CAP_BYTES = 6 * 1024**3
MICROBATCH_SIZE = 256
ROLE_SPECS = {
    "training": (4096, 73801, 73811),
    "validation": (8192, 73802, 73812),
    "calibration": (12288, 73803, 73813),
}
ARM_TABLE: Mapping[str, Mapping[str, Any]] = {
    "t2_p01_r2_b3_lr3e4_l1_0": {
        "rank": 2, "basis_order": 1, "basis_num_elems": 2,
        "learning_rate": 3e-4, "l1_weight": 0.0,
    },
    "t2_p02_r2_b3_lr3e4_l1_1e8": {
        "rank": 2, "basis_order": 1, "basis_num_elems": 2,
        "learning_rate": 3e-4, "l1_weight": 1e-8,
    },
    "t2_p03_r4_b3_lr3e4_l1_1e9": {
        "rank": 4, "basis_order": 1, "basis_num_elems": 2,
        "learning_rate": 3e-4, "l1_weight": 1e-9,
    },
    "t2_p04_r4_b3_lr3e4_l1_1e8": {
        "rank": 4, "basis_order": 1, "basis_num_elems": 2,
        "learning_rate": 3e-4, "l1_weight": 1e-8,
    },
    "t2_p05_r4_b5_lr3e4_l1_1e9": {
        "rank": 4, "basis_order": 2, "basis_num_elems": 2,
        "learning_rate": 3e-4, "l1_weight": 1e-9,
    },
    "t2_p06_r4_b5_lr1e4_l1_1e9": {
        "rank": 4, "basis_order": 2, "basis_num_elems": 2,
        "learning_rate": 1e-4, "l1_weight": 1e-9,
    },
}


def _settings(arm_id: str) -> LaneBT1Settings:
    row = ARM_TABLE[arm_id]
    return LaneBT1Settings(
        arm_id=arm_id,
        rank=int(row["rank"]),
        basis_order=int(row["basis_order"]),
        basis_num_elems=int(row["basis_num_elems"]),
        learning_rate=float(row["learning_rate"]),
        l1_weight=float(row["l1_weight"]),
        l2_weight=1e-8,
        batch_size=MICROBATCH_SIZE,
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
        return _jsonable(value.numpy().item() if value.shape.rank == 0 else value.numpy().tolist())
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, float) and not (float("-inf") < value < float("inf")):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n")


def _load_role(directory: Path, role: str):
    if role not in ROLE_SPECS:
        raise ValueError(f"T2 pilot cannot consume role: {role}")
    cloud, payload = load_t2_prepared_cloud(directory)
    expected_count, expected_reference, expected_transition = ROLE_SPECS[role]
    if (
        cloud.role != role
        or cloud.sample_count != expected_count
        or cloud.reference_seed != expected_reference
        or cloud.transition_seed != expected_transition
    ):
        raise ValueError(f"T2 prepared {role} scope mismatch")
    return cloud, payload


def _run_manifest(
    *,
    started: float,
    prepared_paths: Mapping[str, Path],
) -> Mapping[str, Any]:
    logical = tuple(tf.config.list_logical_devices("GPU"))
    if not logical:
        raise RuntimeError("T2 pilot requires a logical GPU")
    extra_paths = (
        Path(__file__).resolve().relative_to(ROOT),
        Path("scripts/select_zhao_cui_austria_sir_lane_b_t2.py"),
        PLAN,
        EXECUTION_NOTE,
    )
    source_hashes = dict(t2_source_closure())
    for path in extra_paths:
        source_hashes[path.as_posix()] = _sha256(ROOT / path)
    return {
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "command": tuple(sys.argv),
        "environment": sys.prefix,
        "host": platform.node(),
        "python": platform.python_version(),
        "tensorflow": tf.__version__,
        "device": tuple(device.name for device in logical),
        "dtype": "float64_reference_training",
        "tf32_enabled": bool(tf.config.experimental.tensor_float_32_execution_enabled()),
        "jit_compile": True,
        "gpu_memory_policy": dict(MEMORY_POLICY),
        "trust_basis": "owner_designated_managed_session_visible_gpu_trusted",
        "full_cloud_update_semantics": "16_fixed_parameter_microbatches_then_one_clipped_adam_update",
        "prepared_source_sha256": dict(prepared_source_closure()),
        "prepared_result_sha256": {
            role: _sha256(path / "result.json") for role, path in prepared_paths.items()
        },
        "source_sha256": {path: source_hashes[path] for path in sorted(source_hashes)},
        "plan": PLAN.as_posix(),
        "execution_note": EXECUTION_NOTE.as_posix(),
        "wall_time_seconds": time.monotonic() - started,
    }


def run_pilot(
    *,
    arm_id: str,
    parent_dir: Path,
    prepared_paths: Mapping[str, Path],
    output_dir: Path,
    max_seconds: float,
) -> Mapping[str, Any]:
    started = time.monotonic()
    parent = load_lane_b_t1_artifact_v1_compat(parent_dir)
    training_cloud, training_payload = _load_role(prepared_paths["training"], "training")
    validation_cloud, validation_payload = _load_role(
        prepared_paths["validation"], "validation"
    )
    calibration_cloud, calibration_payload = _load_role(
        prepared_paths["calibration"], "calibration"
    )
    settings = _settings(arm_id)
    frame = build_t2_frame(training_cloud, settings)
    calibration_estimate = prepared_estimate(calibration_payload, calibrated=True)
    shift = calibration_estimate.shift_constant
    validation_estimate = estimate_t2_shifted_log_normalizer(validation_cloud, shift)
    training = build_t2_training_batch(training_cloud, frame, shift)
    validation = build_t2_training_batch(validation_cloud, frame, shift)
    normalizer_agreement = bool(
        normalizer_estimates_agree(calibration_estimate, validation_estimate).numpy()
    )
    if not normalizer_agreement:
        raise RuntimeError("T2 calibration-validation normalizer gate failed")

    config = t2_trainer_config(settings)
    trainer = TrainableFunctionalTT(
        config,
        initial_cores=balanced_initial_cores(settings, config.product_basis),
    )
    initial_square_mass = trainer.sqrt_square_normalizer()
    initial_rho = trainer.rho_theta(training.points[:MICROBATCH_SIZE])
    initial_mass_gate = bool(
        ((initial_square_mass >= 0.5) & (initial_square_mass <= 2.0)).numpy()
    )
    initial_variation_gate = bool(
        (tf.math.reduce_std(initial_rho) > tf.constant(1e-12, tf.float64)).numpy()
    )
    if not initial_mass_gate or not initial_variation_gate:
        raise RuntimeError("T2 balanced initialization gate failed")
    optimizer = make_adam_optimizer(config)
    step_fn = make_t2_compiled_train_step(
        trainer, optimizer, microbatch_size=MICROBATCH_SIZE
    )
    trace = []
    for step_index in range(settings.train_steps):
        if time.monotonic() - started > max_seconds:
            raise TimeoutError("T2 pilot exceeded its wall-time cap")
        values = step_fn(training.points, training.log_importance_weight)
        if step_index in {0, settings.train_steps - 1} or (step_index + 1) % 16 == 0:
            trace.append(
                {
                    "full_cloud_update": step_index + 1,
                    "total_loss_before_update": values[0],
                    "cross_entropy_before_update": values[1],
                    "log_normalizer_before_update": values[2],
                    "regularization_before_update": values[3],
                    "unclipped_full_cloud_gradient_norm": values[4],
                    "rho_min": values[5],
                    "rho_max": values[6],
                    "alpha_min": values[7],
                    "alpha_max": values[8],
                }
            )
        if time.monotonic() - started > max_seconds:
            raise TimeoutError("T2 pilot exceeded its wall-time cap")

    pre_calibration_normalizer = trainer.normalizer()
    scale = calibrate_trainer_normalizer(
        trainer, calibration_estimate.log_shifted_normalizer
    )
    metric_fn = make_t2_compiled_validation_metric(trainer)
    metrics = metric_fn(
        validation.points,
        validation.log_importance_weight,
        validation.log_target_reference,
        validation_estimate.log_shifted_normalizer,
    )
    shape_gate = bool((metrics[0] <= 0.95 * metrics[1]).numpy())

    artifact = None
    artifact_manifest = None
    reload_gate = False
    mass_residual = None
    if shape_gate:
        artifact = make_lane_b_t2_artifact(
            parent_artifact=parent,
            settings=settings,
            frame=frame,
            trainer=trainer,
            shift_constant=shift,
            calibration_estimate=calibration_estimate,
            validation_estimate=validation_estimate,
            training_cloud_manifest=training_cloud.manifest_payload(),
            validation_cloud_manifest=validation_cloud.manifest_payload(),
        )
        artifact_manifest = save_lane_b_t2_artifact(artifact, output_dir / "artifact")
        reloaded = load_lane_b_t2_artifact(
            artifact_manifest.parent, parent_artifact=parent
        )
        reload_gate = reloaded.identity == artifact.identity
        tf.debugging.assert_near(reloaded.value(), artifact.value(), atol=0.0)
        mass_residual = tf.abs(
            tf.math.log(reloaded.density().normalizer())
            - calibration_estimate.log_shifted_normalizer
        )

    memory = tf.config.experimental.get_memory_info("GPU:0")
    memory_gate = int(memory["peak"]) <= MEMORY_CAP_BYTES
    viable = normalizer_agreement and shape_gate and reload_gate and memory_gate
    run_manifest = _run_manifest(started=started, prepared_paths=prepared_paths)
    return {
        "schema_version": "bayesfilter.zhao_cui_austria_sir_lane_b_t2_pilot.v1",
        "status": "VIABLE_T2_PILOT_ARM" if viable else "REJECTED_T2_PILOT_ARM",
        "baseline_id": T2_BASELINE_ID,
        "arm": settings.manifest_payload(),
        "parent_t1_identity": parent.identity.hash.value,
        "training_trace": trace,
        "initialization": {
            "square_mass": initial_square_mass,
            "rho_min": tf.reduce_min(initial_rho),
            "rho_max": tf.reduce_max(initial_rho),
            "rho_stddev": tf.math.reduce_std(initial_rho),
            "mass_gate": initial_mass_gate,
            "variation_gate": initial_variation_gate,
        },
        "pre_calibration_normalizer": pre_calibration_normalizer,
        "core_scale": scale,
        "calibration_estimate": calibration_estimate.manifest_payload(),
        "validation_estimate": validation_estimate.manifest_payload(),
        "validation_metrics": {
            "normalized_log_density_rms": metrics[0],
            "constant_density_baseline_rms": metrics[1],
            "centered_log_shape_rms": metrics[2],
            "target_weight_min": metrics[3],
            "target_weight_max": metrics[4],
        },
        "gates": {
            "normalizer_agreement": normalizer_agreement,
            "shape_better_than_0p95_constant": shape_gate,
            "fresh_artifact_reload": reload_gate,
            "memory_under_6_gib": memory_gate,
            "viable": viable,
        },
        "artifact_manifest": (
            None if artifact_manifest is None else artifact_manifest.relative_to(ROOT).as_posix()
        ),
        "artifact_identity": None if artifact is None else artifact.identity.hash.value,
        "artifact_value": None if artifact is None else artifact.value(),
        "direct_mass_log_residual": mass_residual,
        "gpu_allocator": {key: int(value) for key, value in memory.items()},
        "prepared_inputs": {
            "training": training_payload["cloud_manifest"],
            "validation": validation_payload["cloud_manifest"],
            "calibration": calibration_payload["cloud_manifest"],
        },
        "run_manifest": run_manifest,
        "inference_status": {
            "hard_veto_screen": "passed" if viable else "failed_current_arm",
            "statistically_supported_ranking": False,
            "descriptive_only_differences": True,
            "default_readiness": False,
            "next_evidence_needed": "all six arms then untouched selected-arm value claim",
        },
        "nonclaims": (
            "no statistically supported arm ranking",
            "no T2 value admission before untouched claim",
            "no score, T20, HMC, production KR, or scientific claim",
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-arm", choices=tuple(ARM_TABLE), required=True)
    parser.add_argument("--parent-t1-dir", type=Path, required=True)
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--validation-dir", type=Path, required=True)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-seconds", type=float, default=600.0)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    prepared_paths = {
        "training": args.training_dir.resolve(),
        "validation": args.validation_dir.resolve(),
        "calibration": args.calibration_dir.resolve(),
    }
    try:
        result = run_pilot(
            arm_id=args.pilot_arm,
            parent_dir=args.parent_t1_dir.resolve(),
            prepared_paths=prepared_paths,
            output_dir=output,
            max_seconds=float(args.max_seconds),
        )
        _write_json(output / "result.json", result)
    except Exception as exc:
        _write_json(
            output / "result.json",
            {
                "schema_version": "bayesfilter.zhao_cui_austria_sir_lane_b_t2_failure.v1",
                "status": "INFRASTRUCTURE_OR_IMPLEMENTATION_FAILURE",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "command": tuple(sys.argv),
                "gpu_memory_policy": dict(MEMORY_POLICY),
            },
        )
        raise


if __name__ == "__main__":
    main()
