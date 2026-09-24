#!/usr/bin/env python3
"""LEDH Trust-Region Phase 3: Tuning and Safety Evaluation - Austria SIR T20.

Three-arm comparison:
- Arm 1: Baseline (Contract-E only, no dual-cap)
- Arm 2: Dual-cap primal (existing tuned controls, no trust-region)
- Arm 3: Trust-region (27-config grid search over LM damping, scale floor, trust radius)

Evaluates Class C non-harm criterion for both dual-cap and trust-region mechanisms.
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
from pathlib import Path
from typing import Any

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bayesfilter.highdim.ledh_production_program_v1 import (
    LEDH_PRODUCTION_PROGRAM_V1,
    validate_ledh_production_configuration,
)
from docs.benchmarks import run_moment_retuned_genut_whole_leaderboard as base

# Import TensorFlow AFTER setting environment variables and base
import tensorflow as tf

# Import TensorFlow AFTER setting environment variables
import tensorflow as tf
from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth

PLAN = Path("docs/plans/ledh-trust-region-tuning-safety-evaluation-phase3-2026-09-02.md")
DUAL_CAP_ARTIFACT = Path("docs/benchmarks/artifacts/genut_four_model_leaderboard_rerun_20260816/single_austria_sir_T20_dual_cap/result.json")

TUNING_SEEDS = (98301, 98302)
RESIDUAL_TOL = 5.0e-4

# Trust-region tuning grid (27 configurations)
LM_DAMPING_GRID = (1e-3, 1e-2, 1e-1)
LM_SCALE_FLOOR_GRID = (1e-6, 1e-5, 1e-4)
TRUST_RADIUS_GRID = (0.1, 0.5, 1.0)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tensor_hash(value: tf.Tensor, dtype: tf.dtypes.DType = tf.float64) -> str:
    tensor = tf.convert_to_tensor(value, dtype=dtype)
    return hashlib.sha256(bytes(tf.io.serialize_tensor(tensor).numpy())).hexdigest()


def _safe(value: Any) -> Any:
    if hasattr(value, "numpy"):
        value = value.numpy()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _genut_design(dim: int) -> tf.Tensor:
    from bayesfilter.highdim.cubature_genut_candidate import (
        cubature_design,
        gaussian_genut_design,
        replicate_positive_genut,
    )
    if dim >= 18:
        return cubature_design(dim=dim, num_particles=N)
    return replicate_positive_genut(gaussian_genut_design(dim=dim), num_particles=N)


def _build_austria_sir_target() -> dict[str, Any]:
    """Build Austria SIR T20 target from base module."""
    targets = base._build_targets()
    austria = targets["austria_sir_T20"]

    # Add horizon field for evaluator
    horizon = int(austria["observations"].shape[0])
    austria["horizon"] = horizon

    return austria


def _load_dual_cap_primal_controls() -> dict[str, Any]:
    """Load existing dual-cap primal tuned controls from artifact."""
    artifact = json.loads((ROOT / DUAL_CAP_ARTIFACT).read_text(encoding="utf-8"))
    controls = dict(artifact["controls"])

    # Verify this is dual-cap primal (no trust-region)
    if controls.get("higher_moment_lm_damping", 0.0) != 0.0:
        raise RuntimeError("Expected dual-cap primal artifact, found trust-region controls")

    # Ensure trust-region controls are explicitly zero
    controls.setdefault("higher_moment_lm_damping", 0.0)
    controls.setdefault("higher_moment_lm_scale_floor", 1e-6)
    controls.setdefault("higher_moment_trust_radius", 0.0)

    return controls


def _baseline_controls(dual_cap_controls: dict[str, Any]) -> dict[str, Any]:
    """Create baseline controls (no dual-cap, no trust-region)."""
    controls = {
        "epsilon": dual_cap_controls["epsilon"],
        "sinkhorn_steps": dual_cap_controls["sinkhorn_steps"],
        "balance_steps": dual_cap_controls["balance_steps"],
        "ridge": dual_cap_controls["ridge"],
        "higher_moment_correction_steps": 0,
        "higher_moment_strength": 0.0,
        "higher_moment_floor": 1e-5,
        "higher_moment_lm_damping": 0.0,
        "higher_moment_lm_scale_floor": 1e-6,
        "higher_moment_trust_radius": 0.0,
        "pairwise_moment_correction_steps": 0,
        "pairwise_moment_strength": 0.0,
        "pairwise_moment_floor": 1e-5,
        "pairwise_particle_rms_cap": 0.0,
        "coordinatewise_standardized_cap": 0.0,
        "coordinatewise_standardized_cap_power": 8,
    }
    return controls


def _trust_region_controls(
    dual_cap_controls: dict[str, Any],
    lm_damping: float,
    lm_scale_floor: float,
    trust_radius: float,
) -> dict[str, Any]:
    """Create trust-region controls by adding TR parameters to dual-cap primal."""
    controls = dict(dual_cap_controls)
    controls["higher_moment_lm_damping"] = lm_damping
    controls["higher_moment_lm_scale_floor"] = lm_scale_floor
    controls["higher_moment_trust_radius"] = trust_radius
    return controls


def _make_evaluator(target: dict[str, Any], controls: dict[str, Any]):
    """Create evaluator for given controls."""
    from bayesfilter.highdim.cubature_genut_filter import finite_value_score

    adapter = target["adapter"]
    horizon = target["horizon"]
    observation_dim = target["observation_dim"]
    state_dim = target["state_dim"]
    parameter_dim = target["parameter_dim"]
    num_particles = int(target["design"].shape[0])

    @tf.function(jit_compile=True, reduce_retracing=True)
    def evaluate(theta, observations, initial_noise, process_noise, design):
        theta = tf.ensure_shape(theta, [parameter_dim])
        observations = tf.ensure_shape(observations, [horizon, observation_dim])
        initial_noise = tf.ensure_shape(initial_noise, [num_particles, state_dim])
        process_noise = tf.ensure_shape(process_noise, [horizon, num_particles, state_dim])
        design = tf.ensure_shape(design, [num_particles, state_dim])

        with tf.device("/GPU:0"):
            return finite_value_score(
                adapter,
                theta,
                observations,
                initial_noise,
                process_noise,
                design,
                transition_before_first_observation=(
                    target["event_order"] == "transition_before_first_observation"
                ),
                epsilon=float(controls["epsilon"]),
                sinkhorn_steps=int(controls["sinkhorn_steps"]),
                balance_steps=int(controls["balance_steps"]),
                ridge=float(controls["ridge"]),
                higher_moment_correction_steps=int(controls["higher_moment_correction_steps"]),
                higher_moment_strength=float(controls["higher_moment_strength"]),
                higher_moment_floor=float(controls["higher_moment_floor"]),
                higher_moment_lm_damping=float(controls["higher_moment_lm_damping"]),
                higher_moment_lm_scale_floor=float(controls["higher_moment_lm_scale_floor"]),
                higher_moment_trust_radius=float(controls["higher_moment_trust_radius"]),
                pairwise_moment_correction_steps=int(controls["pairwise_moment_correction_steps"]),
                pairwise_moment_strength=float(controls["pairwise_moment_strength"]),
                pairwise_moment_floor=float(controls["pairwise_moment_floor"]),
                pairwise_particle_rms_cap=float(controls["pairwise_particle_rms_cap"]),
                coordinatewise_standardized_cap=float(controls["coordinatewise_standardized_cap"]),
                coordinatewise_standardized_cap_power=int(controls["coordinatewise_standardized_cap_power"]),
            )

    return evaluate


def _noise(seed: int, horizon: int, state_dim: int, num_particles: int) -> tuple[tf.Tensor, tf.Tensor]:
    """Generate initial and process noise for one evaluation."""
    rng = tf.random.Generator.from_seed(seed)
    initial = rng.normal([num_particles, state_dim], dtype=tf.float32)
    process = rng.normal([horizon, num_particles, state_dim], dtype=tf.float32)
    return initial, process


def _evaluate_one(
    evaluator: Any, target: dict[str, Any], observations: tf.Tensor, seed: int
) -> dict[str, Any]:
    """Run one evaluation."""
    num_particles = int(target["design"].shape[0])
    initial, process = _noise(seed, target["horizon"], target["state_dim"], num_particles)
    value, score, status = evaluator(
        target["theta"], observations, initial, process, target["design"]
    )

    result = {
        "seed": seed,
        "value": float(value.numpy()),
        "score": [float(x) for x in score.numpy()],
        "finite": bool(tf.reduce_all(tf.math.is_finite(value)).numpy()),
    }

    # Extract diagnostics from status dict
    for key in status:
        val = status[key]
        if hasattr(val, "numpy"):
            result[key] = _safe(val)

    return result


def _valid(row: dict[str, Any], controls: dict[str, Any]) -> bool:
    """Check if evaluation passes hard validity gates."""
    if not row.get("finite", False):
        return False
    if not row.get("program_valid", False):
        return False

    # Transport residual gates
    residual_keys = [
        "maximum_physical_affine_mean_residual",
        "maximum_physical_affine_covariance_residual",
        "maximum_normalized_physical_affine_mean_residual",
        "maximum_normalized_physical_affine_covariance_residual",
    ]
    for key in residual_keys:
        if key in row and row[key] > RESIDUAL_TOL:
            return False

    # Coordinate cap gate (if enabled)
    if controls.get("coordinatewise_standardized_cap", 0.0) > 0.0:
        post_cap = row.get("maximum_coordinatewise_post_cap_absolute", float("inf"))
        if post_cap >= 1.000001:
            return False

    return True


def _config_label(lm_damping: float, lm_scale_floor: float, trust_radius: float) -> str:
    """Create short label for trust-region config."""
    return f"d{lm_damping:.0e}_f{lm_scale_floor:.0e}_r{trust_radius:.1f}"


def run(output_root: Path) -> dict[str, Any]:
    """Execute Phase 3 campaign."""
    started = time.perf_counter()
    if output_root.exists():
        import shutil
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=False)

    # GPU setup
    # TF_FORCE_GPU_ALLOW_GROWTH is already set in environment, skip explicit config
    tf.config.set_soft_device_placement(False)
    tf.config.experimental.enable_tensor_float_32_execution(True)
    logical = tf.config.list_logical_devices("GPU")
    if not logical:
        raise RuntimeError("Phase 3 campaign requires a logical GPU")
    memory_policy = {"memory_growth_enabled": True, "note": "via TF_FORCE_GPU_ALLOW_GROWTH env var"}

    # Build target (CPU device to avoid GPU initialization during setup)
    with tf.device("/CPU:0"):
        target = _build_austria_sir_target()

    # Load dual-cap primal controls
    dual_cap_primal_controls = _load_dual_cap_primal_controls()

    # Create arm controls
    baseline_controls = _baseline_controls(dual_cap_primal_controls)

    # Validate production program compliance
    # Baseline and dual-cap primal are labeled "ablation"/"baseline"
    # Trust-region configs are labeled "production" (require wiring gate pass)
    baseline_validation = validate_ledh_production_configuration(
        reset_policy="contract_e",
        dual_cap_enabled=False,
        trust_region_enabled=False,
        program_label="ablation",
    )

    dual_cap_validation = validate_ledh_production_configuration(
        reset_policy="contract_e",
        dual_cap_enabled=True,
        trust_region_enabled=False,
        program_label="baseline",
    )

    # Run Arm 1: Baseline
    print("Running Arm 1: Baseline (Contract-E only)")
    arm1_evaluator = _make_evaluator(target, baseline_controls)
    arm1_rows = []
    for obs in target["calibration"]:
        for seed in TUNING_SEEDS:
            row = _evaluate_one(arm1_evaluator, target, obs, seed)
            arm1_rows.append(row)
            (output_root / f"arm1_baseline_obs{len(arm1_rows)//2}_seed{seed}.json").write_text(
                json.dumps(_safe(row), indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )

    arm1_valid = all(_valid(row, baseline_controls) for row in arm1_rows)
    arm1_summary = {
        "controls": baseline_controls,
        "rows": arm1_rows,
        "calibration_valid": arm1_valid,
        "validation": baseline_validation,
    }

    # Run Arm 2: Dual-cap primal
    print("Running Arm 2: Dual-cap primal")
    arm2_evaluator = _make_evaluator(target, dual_cap_primal_controls)
    arm2_rows = []
    for obs in target["calibration"]:
        for seed in TUNING_SEEDS:
            row = _evaluate_one(arm2_evaluator, target, obs, seed)
            arm2_rows.append(row)
            (output_root / f"arm2_dual_cap_obs{len(arm2_rows)//2}_seed{seed}.json").write_text(
                json.dumps(_safe(row), indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )

    arm2_valid = all(_valid(row, dual_cap_primal_controls) for row in arm2_rows)
    arm2_summary = {
        "controls": dual_cap_primal_controls,
        "rows": arm2_rows,
        "calibration_valid": arm2_valid,
        "validation": dual_cap_validation,
    }

    # Run Arm 3: Trust-region grid
    print("Running Arm 3: Trust-region grid (27 configs)")
    arm3_configs = []
    config_index = 0

    for lm_damping in LM_DAMPING_GRID:
        for lm_scale_floor in LM_SCALE_FLOOR_GRID:
            for trust_radius in TRUST_RADIUS_GRID:
                config_index += 1
                label = _config_label(lm_damping, lm_scale_floor, trust_radius)
                print(f"  Config {config_index}/27: {label}")

                tr_controls = _trust_region_controls(
                    dual_cap_primal_controls, lm_damping, lm_scale_floor, trust_radius
                )

                # Validate production program compliance
                tr_validation = validate_ledh_production_configuration(
                    reset_policy="contract_e",
                    dual_cap_enabled=True,
                    trust_region_enabled=True,
                    program_label="production",
                )

                tr_evaluator = _make_evaluator(target, tr_controls)
                tr_rows = []

                for obs_idx, obs in enumerate(target["calibration"]):
                    for seed in TUNING_SEEDS:
                        row = _evaluate_one(tr_evaluator, target, obs, seed)
                        tr_rows.append(row)
                        filename = f"arm3_trust_region_{label}_obs{obs_idx}_seed{seed}.json"
                        (output_root / filename).write_text(
                            json.dumps(_safe(row), indent=2, sort_keys=True) + "\n",
                            encoding="utf-8",
                        )

                tr_valid = all(_valid(row, tr_controls) for row in tr_rows)

                config_summary = {
                    "config_index": config_index,
                    "label": label,
                    "controls": tr_controls,
                    "higher_moment_lm_damping": lm_damping,
                    "higher_moment_lm_scale_floor": lm_scale_floor,
                    "higher_moment_trust_radius": trust_radius,
                    "rows": tr_rows,
                    "calibration_valid": tr_valid,
                    "validation": tr_validation,
                }
                arm3_configs.append(config_summary)

    # Select best trust-region config
    passing_configs = [cfg for cfg in arm3_configs if cfg["calibration_valid"]]

    if passing_configs:
        # Selection criterion: minimal intervention (lowest cap fire rate, then lowest damping, then smallest radius)
        def selection_key(cfg):
            cap_rate = max(
                row.get("maximum_coordinatewise_cap_active_fraction", 0.0)
                for row in cfg["rows"]
            )
            return (
                cap_rate,
                cfg["higher_moment_lm_damping"],
                cfg["higher_moment_trust_radius"],
            )

        selected = min(passing_configs, key=selection_key)
        tuning_verdict = "SUCCESS"
    else:
        selected = None
        tuning_verdict = "FAILURE_NO_VALID_CONFIG"

    # Assemble result
    wall_time = time.perf_counter() - started

    result = {
        "schema_version": "ledh_trust_region_phase3.v1",
        "status": tuning_verdict,
        "plan": PLAN.as_posix(),
        "target": {
            "model": "austria_sir_T20",
            "horizon": target["horizon"],
            "state_dimension": target["state_dim"],
            "observation_dimension": target["observation_dim"],
            "parameter_dimension": target["parameter_dim"],
            "particle_count": int(target["design"].shape[0]),
            "event_order": target["event_order"],
            "source_observation_sha256": target["source_observation_sha256"],
        },
        "arms": {
            "arm1_baseline": arm1_summary,
            "arm2_dual_cap_primal": arm2_summary,
            "arm3_trust_region_grid": {
                "configs": arm3_configs,
                "passing_count": len(passing_configs),
                "total_count": len(arm3_configs),
            },
        },
        "selected_config": _safe(selected) if selected else None,
        "production_program": dict(LEDH_PRODUCTION_PROGRAM_V1),
        "device": {
            "logical_devices": [d.name for d in logical],
            "trust_basis": "owner_designated_managed_session_visible_gpu_trusted",
        },
        "memory_policy": dict(memory_policy),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "wall_time_seconds": wall_time,
        "run_manifest": {
            "command": [sys.executable, *sys.argv],
            "environment": sys.prefix,
            "host": platform.node(),
            "python": platform.python_version(),
            "tensorflow": tf.__version__,
            "plan": PLAN.as_posix(),
            "output_json": str((output_root / "result.json").relative_to(ROOT)),
        },
    }

    result_json = output_root / "result.json"
    result_json.write_text(
        json.dumps(_safe(result), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    # Print summary
    print(f"\n{'='*80}")
    print(f"Phase 3 Complete: {tuning_verdict}")
    print(f"Wall time: {wall_time:.1f}s")
    print(f"\nArm 1 (Baseline): {'VALID' if arm1_valid else 'INVALID'}")
    print(f"Arm 2 (Dual-cap primal): {'VALID' if arm2_valid else 'INVALID'}")
    print(f"Arm 3 (Trust-region): {len(passing_configs)}/{len(arm3_configs)} configs pass")

    if selected:
        print(f"\nSelected trust-region config:")
        print(f"  Label: {selected['label']}")
        print(f"  LM damping: {selected['higher_moment_lm_damping']}")
        print(f"  LM scale floor: {selected['higher_moment_lm_scale_floor']}")
        print(f"  Trust radius: {selected['higher_moment_trust_radius']}")
    else:
        print(f"\nNo valid trust-region configuration found!")

    print(f"{'='*80}\n")

    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    result = run(args.output_root.resolve())

    print(
        json.dumps(
            {
                "status": result["status"],
                "output": str(args.output_root.resolve()),
                "wall_time_seconds": result["wall_time_seconds"],
            }
        )
    )


if __name__ == "__main__":
    main()
