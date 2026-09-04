#!/usr/bin/env python3
"""Phase 2B repair: complete 9 missing RQMC initialization runs.

Repairs:
1. KSC SV sobol_matousek (3 runs): skip Lloyd optimization for 1D case
2. Predator-Prey halton_owen (3 runs): use salvaged tuning artifact
3. Predator-Prey genut_guided (3 runs): use salvaged tuning artifact

Total: 9 runs to complete the 45-run Phase 2 campaign.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from scipy.stats import qmc

# Add repo root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from bayesfilter.highdim import ledh_production_program_v1


def _verify_gpu_memory_growth():
    """Verify memory growth is enabled on all GPUs."""
    print("\nVerifying TensorFlow GPU memory growth configuration")
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print("  No GPUs detected")
        return

    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
            config = tf.config.experimental.get_memory_growth(gpu)
            print(f"  {gpu.name}: memory_growth={config}")
        except Exception as e:
            print(f"  {gpu.name}: memory_growth configuration failed: {e}")
            raise


def _load_model_target(model: str) -> dict[str, Any]:
    """Load model target specification."""
    from bayesfilter.models import austria_sir_t20, ksc_sv_t10, lgssm_t50, predator_prey_t20

    targets = {
        "lgssm_T50": lgssm_t50.build_ledh_contract_e_tp_target,
        "ksc_sv_T10": ksc_sv_t10.build_ledh_contract_e_tp_target,
        "predator_prey_T20": predator_prey_t20.build_ledh_contract_e_tp_target,
        "austria_sir_T20": austria_sir_t20.build_ledh_contract_e_tp_target,
    }

    if model not in targets:
        raise ValueError(f"Unknown model: {model}")

    print(f"\nLoading model target: {model}")
    return targets[model]()


def _load_tuning_artifact(path: str) -> dict[str, Any]:
    """Load trust-region tuning artifact."""
    print(f"Loading tuning artifact: {path}")
    with open(path) as f:
        return json.load(f)


def _generate_initial_noise(
    arm: str,
    state_dim: int,
    particle_count: int,
    seed: int,
    rng: np.random.Generator,
) -> tf.Tensor:
    """Generate initial noise according to RQMC arm."""

    if arm == "mc":
        # Monte Carlo: pure random sampling
        uniforms = rng.uniform(size=(particle_count, state_dim)).astype(np.float32)

    elif arm == "sobol_matousek":
        # Sobol with Matousek scrambling + Lloyd optimization
        # Lloyd requires dim >= 2, so skip it for 1D
        sobol = qmc.Sobol(d=state_dim, scramble=True, seed=seed, optimization="lloyd-max-cd" if state_dim >= 2 else None)
        if state_dim == 1:
            print(f"  WARNING: Skipping Lloyd optimization for 1D Sobol (arm={arm}, dim={state_dim})")
        uniforms = sobol.random(particle_count).astype(np.float32)

    elif arm == "sobol_owen":
        # Sobol with Owen scrambling (no Lloyd)
        sobol = qmc.Sobol(d=state_dim, scramble=True, seed=seed)
        uniforms = sobol.random(particle_count).astype(np.float32)

    elif arm == "halton_owen":
        # Halton with Owen scrambling (no Lloyd)
        halton = qmc.Halton(d=state_dim, scramble=True, seed=seed)
        uniforms = halton.random(particle_count).astype(np.float32)

    elif arm == "genut_guided":
        # GENUT-guided: Gaussian-transformed Halton
        halton = qmc.Halton(d=state_dim, scramble=True, seed=seed)
        uniforms = halton.random(particle_count).astype(np.float32)
        # Transform to Gaussian via inverse CDF
        gaussian = tfp.distributions.Normal(0.0, 1.0).quantile(uniforms)
        return tf.constant(gaussian, dtype=tf.float32)

    else:
        raise ValueError(f"Unknown arm: {arm}")

    # Transform uniform to Gaussian via inverse CDF
    gaussian = tfp.distributions.Normal(0.0, 1.0).quantile(uniforms)
    return tf.constant(gaussian, dtype=tf.float32)


def _run_evaluation(
    model: str,
    arm: str,
    seed: int,
    tuning_artifact: dict[str, Any],
    target: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    """Run one RQMC initialization evaluation."""

    rng = np.random.default_rng(seed)
    particle_count = target["prepared_data"]["state_value"].shape[0]
    state_dim = target["state_dim"]

    print(f"\nGenerating noise for arm={arm}, seed={seed}")
    print(f"  Particle count: {particle_count}")
    print(f"  State dim: {state_dim}")

    initial_noise = _generate_initial_noise(arm, state_dim, particle_count, seed, rng)

    # Build evaluator with tuned trust-region controls
    print("Creating evaluator")
    config = tuning_artifact["selected_config"]

    def evaluator_fn(noise):
        return ledh_production_program_v1(
            target=target,
            initial_noise=noise,
            trust_region_damping=config["damping"],
            trust_region_scale_floor=config["scale_floor"],
            trust_region_radius=config["radius"],
        )

    # Wrap in tf.function for stability
    evaluator = tf.function(evaluator_fn)

    # Run evaluation
    print("Running evaluation")
    start = time.perf_counter()
    result = evaluator(initial_noise)
    wall_seconds = time.perf_counter() - start

    # Extract metrics
    value = float(result["value"].numpy())
    finite = bool(np.isfinite(value))

    program_diagnostics = result.get("program_diagnostics", {})
    fraction_coordinatewise_cap_active = float(
        program_diagnostics.get("fraction_coordinatewise_cap_active", 0.0)
    )

    # Build result artifact
    output_artifact = {
        "schema": "rqmc_ledh_init.phase2_run.v1",
        "model": model,
        "arm": arm,
        "seed": seed,
        "value": value,
        "finite": finite,
        "program_valid": finite,
        "fraction_coordinatewise_cap_active": fraction_coordinatewise_cap_active,
        "wall_seconds": wall_seconds,
        "tuning_artifact": str(Path(args.tuning_artifact).resolve()),
        "git_commit": None,  # Filled by campaign driver
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    # Write result
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "result.json"
    with open(result_path, "w") as f:
        json.dump(output_artifact, f, indent=2)

    print(f"\nResult:")
    print(f"  Value: {value:.4f}")
    print(f"  Finite: {finite}")
    print(f"  Cap fire rate: {fraction_coordinatewise_cap_active:.3f}")
    print(f"  Wall time: {wall_seconds:.1f}s")
    print(f"  Output: {result_path}")

    return output_artifact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--tuning_artifact", required=True)
    parser.add_argument("--output", required=True)

    global args
    args = parser.parse_args()

    print("=" * 80)
    print("RQMC LEDH Phase 2B Repair")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Arm: {args.arm}")
    print(f"Seed: {args.seed}")
    print(f"Tuning artifact: {args.tuning_artifact}")
    print(f"Output: {args.output}")

    _verify_gpu_memory_growth()
    target = _load_model_target(args.model)
    tuning_artifact = _load_tuning_artifact(args.tuning_artifact)

    output_dir = Path(args.output)
    _run_evaluation(
        model=args.model,
        arm=args.arm,
        seed=args.seed,
        tuning_artifact=tuning_artifact,
        target=target,
        output_dir=output_dir,
    )

    print("\n" + "=" * 80)
    print("Phase 2B repair run complete")
    print("=" * 80)


if __name__ == "__main__":
    main()
