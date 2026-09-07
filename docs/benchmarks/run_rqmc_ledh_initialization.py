#!/usr/bin/env python3
"""RQMC LEDH Initialization: Claim-Bearing Runner.

Executes one RQMC vs MC comparison run for LEDH-PFPF-OT with dual-cap and
trust-region stabilization. Supports 5 arms:
- mc: Standard Monte Carlo baseline
- sobol_matousek: Sobol with Matousek 1998 nested uniform scrambling
- sobol_owen: Sobol with Owen 1995 random digital shift + scramble
- halton_owen: Halton with Owen 2017 randomized drop-and-permute
- genut_guided: Ebeigbe et al. 2021 generalized unscented transformation

All arms use identical LEDH production program after initialization.

Usage:
    python docs/benchmarks/run_rqmc_ledh_initialization.py \
        --model lgssm_T50 \
        --arm mc \
        --seed 98301 \
        --tuning_artifact docs/benchmarks/artifacts/ledh_trust_region_lgssm_t50_20260903/result.json \
        --output docs/benchmarks/artifacts/rqmc_ledh_init_v1_20260903/runs/lgssm_T50_mc_seed98301/
"""

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
from typing import Any

# Configure TensorFlow GPU memory growth BEFORE any TF import
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import tensorflow as tf

# Must set memory growth immediately after TF import, before any device initialization
physical_gpus = tf.config.list_physical_devices("GPU")
if physical_gpus:
    for gpu in physical_gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

from bayesfilter.highdim.ledh_production_program_v1 import (
    LEDH_PRODUCTION_PROGRAM_V1,
    validate_ledh_production_configuration,
)
from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth

# Supported models and arms
SUPPORTED_MODELS = ("lgssm_T50", "ksc_sv_T10", "predator_prey_T20")
SUPPORTED_ARMS = ("mc", "sobol_matousek", "sobol_owen", "halton_owen", "genut_guided")

PLAN = Path("docs/plans/rqmc-ledh-initialization-master-program-2026-09-02.md")
N = 1008  # Particle count per production program


def replication_generator(seed: int) -> tf.random.Generator:
    """Create independent TensorFlow Generator from seed using SeedSequence hashing.

    Per tf-consecutive-from-seed-is-one-stream.md memory:
    from_seed(s) and from_seed(s+1) share the same Philox stream, causing
    pseudo-replications. Hash far apart via NumPy SeedSequence.
    """
    material = np.random.SeedSequence(int(seed)).generate_state(2, dtype=np.uint64)
    return tf.random.Generator.from_seed(int(material[0]))


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


def _genut_design(dim: int, rng: tf.random.Generator) -> tf.Tensor:
    """Generate GenUT guided initialization design."""
    from bayesfilter.highdim.cubature_genut_candidate import (
        cubature_design,
        gaussian_genut_design,
        replicate_positive_genut,
    )
    if dim >= 18:
        return cubature_design(dim=dim, num_particles=N)
    return replicate_positive_genut(gaussian_genut_design(dim=dim), num_particles=N)


def _generate_initial_noise(
    arm: str, state_dim: int, seed: int, rng: tf.random.Generator
) -> tf.Tensor:
    """Generate initial noise for given arm and seed.

    Returns [N, state_dim] tensor of standard normal particles.
    """
    if arm == "mc":
        # MC baseline: standard iid Gaussian
        return rng.normal([N, state_dim], dtype=tf.float32)

    elif arm == "sobol_matousek":
        # Sobol with Matousek 1998 nested uniform scrambling
        # Note: scipy.stats.qmc.Sobol(scramble=True) implements Owen digital scrambling,
        # which generalizes Matousek's nested uniform scrambles. Use optimization='lloyd'
        # for centroidal Voronoi point-set improvement.
        #
        # Lloyd (centroidal Voronoi tessellation) is only defined for d >= 2 and
        # raises for d == 1. Falling back to optimization=None at d == 1 makes this
        # arm BYTE-IDENTICAL to sobol_owen, so the two are not independent arms on
        # scalar-state models. Callers must treat them as one arm there; the run
        # artifact records this via `arm_degenerate_with`.
        from scipy.stats import qmc
        optimization = 'lloyd' if state_dim >= 2 else None
        sobol = qmc.Sobol(d=state_dim, scramble=True, optimization=optimization, seed=seed)
        uniforms = sobol.random(N)  # [N, state_dim] in [0, 1]
        # Convert to standard normal via inverse CDF
        zero_np = np.nextafter(0.0, 1.0, dtype=np.float32)
        one_np = np.nextafter(1.0, 0.0, dtype=np.float32)
        guarded = np.clip(uniforms, zero_np, one_np).astype(np.float32)
        from scipy.special import ndtri
        return tf.constant(ndtri(guarded), dtype=tf.float32)

    elif arm == "sobol_owen":
        # Sobol with Owen 1995 random digital shift + scramble
        # scipy.stats.qmc.Sobol(scramble=True, optimization=None) implements Owen scrambling
        from scipy.stats import qmc
        sobol = qmc.Sobol(d=state_dim, scramble=True, optimization=None, seed=seed)
        uniforms = sobol.random(N)  # [N, state_dim] in [0, 1]
        # Convert to standard normal via inverse CDF
        zero_np = np.nextafter(0.0, 1.0, dtype=np.float32)
        one_np = np.nextafter(1.0, 0.0, dtype=np.float32)
        guarded = np.clip(uniforms, zero_np, one_np).astype(np.float32)
        from scipy.special import ndtri
        return tf.constant(ndtri(guarded), dtype=tf.float32)

    elif arm == "halton_owen":
        # Halton with Owen randomized drop-and-permute
        import tensorflow_probability as tfp
        uniforms = tfp.mcmc.sample_halton_sequence(
            state_dim,
            num_results=N,
            dtype=tf.float32,
            randomized=True,
            seed=[int(seed) % 2147483647, (int(seed) + 1000) % 2147483647],
        )
        # Convert to standard normal via inverse CDF
        zero = tf.zeros([], tf.float32)
        one = tf.ones([], tf.float32)
        lower = tf.math.nextafter(zero, one)
        upper = tf.math.nextafter(one, zero)
        guarded = tf.clip_by_value(uniforms, lower, upper)
        return tf.math.ndtri(guarded)

    elif arm == "genut_guided":
        # GenUT guided initialization
        return _genut_design(state_dim, rng)

    else:
        raise ValueError(f"Unsupported arm: {arm}")


def _load_model_target(model: str) -> dict[str, Any]:
    """Load model target from base leaderboard module."""
    from docs.benchmarks import run_moment_retuned_genut_whole_leaderboard as base

    targets = base._build_targets()

    if model not in targets:
        raise ValueError(f"Model {model} not found in targets. Available: {list(targets.keys())}")

    target = targets[model]

    # Add horizon field for evaluator
    horizon = int(target["observations"].shape[0])
    target["horizon"] = horizon

    return target


def _load_tuning_artifact(artifact_path: Path) -> dict[str, Any]:
    """Load trust-region tuning artifact and extract controls."""
    if not artifact_path.exists():
        raise FileNotFoundError(f"Tuning artifact not found: {artifact_path}")

    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))

    # Extract controls from selected_config
    selected_config = artifact.get("selected_config")
    if not selected_config:
        raise ValueError(f"Tuning artifact missing 'selected_config' field")

    controls = dict(selected_config.get("controls", {}))

    # Ensure trust-region controls are present
    required_keys = [
        "higher_moment_lm_damping",
        "higher_moment_lm_scale_floor",
        "higher_moment_trust_radius",
    ]
    for key in required_keys:
        if key not in controls:
            raise ValueError(f"Tuning artifact missing required control: {key}")

    return controls


def _make_evaluator(target: dict[str, Any], controls: dict[str, Any]):
    """Create LEDH evaluator for given controls."""
    from bayesfilter.highdim.cubature_genut_filter import finite_value_score

    adapter = target["adapter"]
    horizon = target["horizon"]
    observation_dim = target["observation_dim"]
    state_dim = target["state_dim"]
    parameter_dim = target["parameter_dim"]
    num_particles = N

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


def _generate_process_noise(
    horizon: int, state_dim: int, seed: int, rng: tf.random.Generator
) -> tf.Tensor:
    """Generate process noise for transition steps.

    Returns [horizon, N, state_dim] tensor of standard normal noise.
    """
    return rng.normal([horizon, N, state_dim], dtype=tf.float32)


def _run_evaluation(
    model: str,
    arm: str,
    seed: int,
    tuning_artifact: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Run one RQMC evaluation and return result."""

    print(f"Loading model target: {model}")
    target = _load_model_target(model)

    print(f"Loading tuning artifact: {tuning_artifact}")
    controls = _load_tuning_artifact(tuning_artifact)

    print(f"Creating evaluator")
    evaluator = _make_evaluator(target, controls)

    print(f"Generating noise for arm={arm}, seed={seed}")
    rng = replication_generator(seed)

    initial_noise = _generate_initial_noise(arm, target["state_dim"], seed, rng)
    process_noise = _generate_process_noise(target["horizon"], target["state_dim"], seed, rng)

    # Use target's existing design (GenUT)
    design = target["design"]

    print(f"Running evaluation")
    start = time.time()
    value, score, status = evaluator(
        target["theta"],
        target["observations"],
        initial_noise,
        process_noise,
        design,
    )
    wall_time = time.time() - start

    # Extract result
    result = {
        "model": model,
        "arm": arm,
        "seed": seed,
        "value": float(value.numpy()),
        "score": [float(x) for x in score.numpy()],
        "finite": bool(tf.reduce_all(tf.math.is_finite(value)).numpy()),
        "wall_time_seconds": wall_time,
    }

    # Arm degeneracy: at state_dim == 1 the Lloyd point-set optimization that
    # distinguishes sobol_matousek from sobol_owen is undefined, so the two arms
    # compute the identical cloud. Record it so downstream analysis treats them
    # as one arm rather than two agreeing arms.
    if arm == "sobol_matousek" and int(target["state_dim"]) == 1:
        result["arm_degenerate_with"] = "sobol_owen"
        result["arm_degeneracy_reason"] = (
            "lloyd_point_set_optimization_undefined_at_state_dim_1"
        )

    # Extract diagnostics from status dict
    for key in status:
        val = status[key]
        if hasattr(val, "numpy"):
            result[key] = _safe(val)

    # Validate production program compliance
    validation = validate_ledh_production_configuration(
        reset_policy="contract_e",
        dual_cap_enabled=True,
        trust_region_enabled=True,
        transport_mode="chunked",
        chunk_policy="dpf_transport_exact_divisor_cap3000_v1",
        dtype="float32",
        program_label="production",
    )
    result["program_validation"] = validation

    return result


def _write_result(result: dict[str, Any], output_dir: Path):
    """Write result JSON and manifest."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get git info
    git_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    git_branch = subprocess.check_output(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=ROOT, text=True
    ).strip()

    # Build manifest
    manifest = {
        "plan": str(PLAN),
        "plan_sha256": _sha256(ROOT / PLAN),
        "git_commit": git_commit,
        "git_branch": git_branch,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "hostname": platform.node(),
        "program_id": LEDH_PRODUCTION_PROGRAM_V1["program_id"],
        "program_version": LEDH_PRODUCTION_PROGRAM_V1["version"],
    }

    # Write files
    result_path = output_dir / "result.json"
    manifest_path = output_dir / "manifest.json"

    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"Result written to: {result_path}")
    print(f"Manifest written to: {manifest_path}")


def main():
    parser = argparse.ArgumentParser(
        description="RQMC LEDH Initialization: Claim-Bearing Runner"
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=SUPPORTED_MODELS,
        help="Model to evaluate",
    )
    parser.add_argument(
        "--arm",
        required=True,
        choices=SUPPORTED_ARMS,
        help="Initialization method",
    )
    parser.add_argument(
        "--seed",
        required=True,
        type=int,
        help="Random seed",
    )
    parser.add_argument(
        "--tuning_artifact",
        required=True,
        type=Path,
        help="Path to trust-region tuning artifact JSON",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output directory for result files",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("RQMC LEDH Initialization: Claim-Bearing Run")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Arm: {args.arm}")
    print(f"Seed: {args.seed}")
    print(f"Tuning artifact: {args.tuning_artifact}")
    print(f"Output: {args.output}")
    print()

    # Verify GPU memory growth was configured
    print("Verifying TensorFlow GPU memory growth configuration")
    physical_gpus = tf.config.list_physical_devices("GPU")
    if not physical_gpus:
        print("ERROR: No GPU devices visible")
        sys.exit(1)
    for gpu in physical_gpus:
        growth = tf.config.experimental.get_memory_growth(gpu)
        print(f"  {gpu.name}: memory_growth={growth}")
        if not growth:
            print(f"ERROR: Memory growth not enabled for {gpu.name}")
            sys.exit(1)
    print()

    # Run evaluation
    result = _run_evaluation(
        args.model,
        args.arm,
        args.seed,
        args.tuning_artifact,
        args.output,
    )

    # Write result
    _write_result(result, args.output)

    # Print summary
    print()
    print("=" * 80)
    print("Run Complete")
    print("=" * 80)
    print(f"Terminal log-likelihood: {result['value']:.6f}")
    print(f"Finite: {result['finite']}")
    print(f"Program valid: {result.get('program_valid', 'N/A')}")
    print(f"Wall time: {result['wall_time_seconds']:.2f}s")
    print()

    if not result["finite"]:
        print("WARNING: Non-finite result - PROMOTION VETO")
        sys.exit(1)

    if not result.get("program_valid", False):
        print("WARNING: Program validation failed - PROMOTION VETO")
        sys.exit(1)

    print("Run passed validity gates")
    sys.exit(0)


if __name__ == "__main__":
    main()