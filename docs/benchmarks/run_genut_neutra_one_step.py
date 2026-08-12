#!/usr/bin/env python3
"""One genuine GPU/XLA NeuTra optimizer update for one frozen GenUT target."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tensorflow as tf

from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth


PLAN = "docs/plans/bayesfilter-genut-four-model-neutra-readiness-plan-2026-08-04.md"
CENTERS = {
    "lgssm": (1.17, 0.805, 0.48, -1.02, -0.824),
    "ksc_sv": (0.31863936396437514, -0.31863936396437514),
    "predator_prey": (0.0, -0.8416212335729142, 0.0, -0.8416212335729142, 0.0, 0.0),
}


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=tuple(CENTERS), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--tuning-artifact", type=Path)
    parser.add_argument("--disable-tf32", action="store_true")
    return parser.parse_args()


def _write(path: Path, payload) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _controls(path: Path | None):
    if path is None:
        return None
    from bayesfilter.highdim.cubature_genut_neutra_targets import GenUTControls

    payload = json.loads(path.read_text(encoding="utf-8"))
    selected = payload["selected_controls"]
    return GenUTControls(
        epsilon=float(selected["epsilon"]),
        sinkhorn_steps=int(selected["sinkhorn_steps"]),
        balance_steps=int(selected["balance_steps"]),
        ridge=float(selected["ridge"]),
        higher_moment_correction_steps=int(selected["higher_moment_correction_steps"]),
        higher_moment_strength=float(selected["higher_moment_strength"]),
        higher_moment_floor=float(selected["higher_moment_floor"]),
        tuning_scope=str(payload["tuning_scope_id"]),
        tuning_artifact=str(path),
    )


def main() -> int:
    args = _args()
    if args.output_root.exists():
        raise RuntimeError(f"output root must be fresh: {args.output_root}")
    args.output_root.mkdir(parents=True)
    started = time.monotonic()
    memory_policy = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    tf.config.set_soft_device_placement(False)
    tf.config.experimental.enable_tensor_float_32_execution(not args.disable_tf32)
    tf.config.experimental.enable_op_determinism()
    from bayesfilter.highdim.cubature_genut_neutra_targets import make_genut_neutra_target
    from bayesfilter.inference.neutra_training import (
        PlainDenseIAFTrainingConfig,
        train_plain_dense_iaf,
    )

    target = make_genut_neutra_target(
        args.model,
        particle_count=1008,
        controls=_controls(args.tuning_artifact),
    )
    dimension = target.parameter_dim
    center = CENTERS[args.model]
    # This narrow factor is a mechanics-only warm-start hypothesis. It is not
    # a training protocol or a promoted geometry for a later serious run.
    factor = tuple(
        tuple(0.01 * float(row == column) for column in range(dimension))
        for row in range(dimension)
    )
    training_dir = args.output_root / "training"
    config = PlainDenseIAFTrainingConfig(
        target_signature=target.target_signature,
        dimension=dimension,
        affine_center=center,
        affine_factor=factor,
        output_dir=training_dir,
        seed=(20260804, 701),
        hidden_layers=(max(2, dimension),),
        stage_count=1,
        steps=1,
        batch_size=2,
        learning_rate=1.0e-3,
        checkpoint_every=1,
        heartbeat_every=1,
        jit_compile=True,
        device="/GPU:0",
        require_gpu=True,
    )
    result = train_plain_dense_iaf(adapter=target, config=config)
    allocator = tf.config.experimental.get_memory_info("GPU:0")
    record = dict(result.records[-1])
    binding = dict(result.runtime_metadata["batch_native_target"])
    passed = bool(
        result.completed_steps == 1
        and record["target_values_finite"]
        and record["target_status_all_valid"]
        and binding["scalar_fallback_used"] is False
        and binding["sample_axis_python_loop_used"] is False
        and binding["row_mapped_scalar_target_used"] is False
        and result.runtime_metadata["all_steps_numeric_finite"]
        and result.runtime_metadata["all_steps_target_values_finite"]
        and result.runtime_metadata["all_steps_target_status_valid"]
    )
    payload = {
        "schema": "bayesfilter.genut_neutra_one_step.v1",
        "model": args.model,
        "passed": passed,
        "completed_steps": result.completed_steps,
        "record": record,
        "runtime_metadata": result.runtime_metadata,
        "target_signature": target.target_signature,
        "adapter_signature": target.adapter_signature(),
        "controls": target.controls.payload(),
        "control_status": target.control_status,
        "training_batch_size": 2,
        "affine_factor_role": "narrow_mechanics_only_warm_start_hypothesis",
        "memory_policy": memory_policy,
        "gpu_allocator": {
            "current_bytes": int(allocator["current"]),
            "peak_bytes": int(allocator["peak"]),
        },
        "deterministic_ops_enabled": True,
        "jit_compile": True,
        "tf32_enabled": not args.disable_tf32,
        "wall_time_seconds": time.monotonic() - started,
        "training_state_path": str(result.state_path),
        "training_latest_path": str(result.latest_path),
        "plan": PLAN,
        "nonclaims": (
            "one optimizer update only",
            "no training quality, transport selection, HMC, or posterior claim",
        ),
    }
    _write(args.output_root / "result.json", payload)
    commit = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    _write(
        args.output_root / "run_manifest.json",
        {
            "schema": "bayesfilter.genut_neutra_one_step_manifest.v1",
            "git_commit": commit,
            "command": tuple(sys.argv),
            "environment": os.environ.get("CONDA_DEFAULT_ENV", "unknown"),
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "tensorflow_version": tf.__version__,
            "device": "/GPU:0",
            "memory_policy": memory_policy,
            "gpu_allocator": {
                "current_bytes": int(allocator["current"]),
                "peak_bytes": int(allocator["peak"]),
            },
            "deterministic_ops_enabled": True,
            "jit_compile": True,
            "tf32_enabled": not args.disable_tf32,
            "particle_count": 1008,
            "batch_size": 2,
            "target_signature": target.target_signature,
            "controls": target.controls.payload(),
            "seed": (20260804, 701),
            "wall_time_seconds": time.monotonic() - started,
            "output_root": str(args.output_root),
            "plan": PLAN,
            "result": str(args.output_root / "result.json"),
        },
    )
    print(json.dumps({"model": args.model, "passed": passed}, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        try:
            destination = _args().output_root
            destination.mkdir(parents=True, exist_ok=True)
            _write(
                destination / "failure.json",
                {
                    "schema": "bayesfilter.genut_neutra_one_step_failure.v1",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                    "plan": PLAN,
                },
            )
        finally:
            raise
