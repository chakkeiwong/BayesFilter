"""Prepare device-independent fixed tensors for latent-SIR Contract E runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import tensorflow as tf


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bayesfilter.highdim.sir_latent_preclip_tf import (  # noqa: E402
    LATENT_PRECLIP_TARGET_ID,
    latent_preclip_zhao_cui_sir_austria_model,
)


DTYPE = tf.float64


def _record(value: tf.Tensor) -> dict[str, object]:
    tensor = tf.convert_to_tensor(value)
    serialized = tf.io.serialize_tensor(tensor).numpy()
    return {
        "dtype": tensor.dtype.name,
        "shape": tensor.shape.as_list(),
        "values": tensor.numpy().tolist(),
        "serialized_tensor_sha256": hashlib.sha256(serialized).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--time-steps", type=int, default=2)
    parser.add_argument("--particle-count", type=int, default=32)
    parser.add_argument("--seed", type=int, default=81103)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise ValueError("preparation must run CPU-only with CUDA_VISIBLE_DEVICES=-1")
    if args.time_steps < 1 or args.particle_count < 4:
        raise ValueError("time_steps and particle_count are too small")
    args.output.parent.mkdir(parents=True, exist_ok=False)

    model = latent_preclip_zhao_cui_sir_austria_model()
    generator = tf.random.Generator.from_seed(int(args.seed))
    state_dimension = model.state_dim()
    observation_dimension = model.observation_dim()
    theta = tf.zeros([3], DTYPE)
    initial_noise = generator.normal(
        [1, args.particle_count, state_dimension], dtype=DTYPE
    )
    transition_noise = generator.normal(
        [
            1,
            max(0, args.time_steps - 1),
            args.particle_count,
            state_dimension,
        ],
        dtype=DTYPE,
    )
    simulation_initial_noise = generator.normal([state_dimension], dtype=DTYPE)
    simulation_transition_noise = generator.normal(
        [max(0, args.time_steps - 1), state_dimension], dtype=DTYPE
    )
    observation_noise = generator.normal(
        [args.time_steps, observation_dimension], dtype=DTYPE
    )
    simulation = model.simulate_from_standard_normals(
        theta,
        simulation_initial_noise,
        simulation_transition_noise,
        observation_noise,
    )
    residual_design = generator.normal(
        [1, args.time_steps, args.particle_count, state_dimension], dtype=DTYPE
    )
    residual_design -= tf.reduce_mean(residual_design, axis=2, keepdims=True)
    prepared = {
        "observations": simulation["observations"],
        "initial_noise": initial_noise,
        "transition_noise": transition_noise,
        "fixed_reset_mask": tf.ones([1, args.time_steps], tf.bool),
        "residual_design": residual_design,
        "prepared_ridge": tf.fill(
            [1, args.time_steps], tf.constant(1.0e-6, DTYPE)
        ),
        "epsilon": tf.constant(0.25, DTYPE),
        "scaling": tf.constant(0.9, DTYPE),
    }
    payload = {
        "schema": "bayesfilter.latent_preclip_sir.prepared_inputs.v1",
        "status": "PASS_CPU_ONLY_PREPARATION",
        "target_id": LATENT_PRECLIP_TARGET_ID,
        "configuration": {
            "time_steps": args.time_steps,
            "particle_count": args.particle_count,
            "seed": args.seed,
            "cpu_only": True,
        },
        "theta": _record(theta),
        "prepared": {key: _record(value) for key, value in prepared.items()},
        "simulator": {
            "physical_path": _record(simulation["physical_path"]),
            "observations": _record(simulation["observations"]),
        },
        "command": " ".join(sys.argv),
        "nonclaims": [
            "preparation is not value or score evidence",
            "not canonical Contract E admission",
        ],
    }
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
