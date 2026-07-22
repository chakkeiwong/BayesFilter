"""Run bounded latent pre-clipping SIR Contract E--Chol diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bayesfilter.highdim import ledh_contract_e_latent_sir_tf as candidate
from bayesfilter.highdim.transport_chunk_policy import select_transport_chunks
from bayesfilter.highdim.sir_latent_preclip_tf import (
    LATENT_PRECLIP_TARGET_ID,
    latent_preclip_zhao_cui_sir_austria_model,
)


DTYPE = tf.float64


def _git_output(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=ROOT, check=False, capture_output=True, text=True
    )
    return result.stdout.strip()


def _tensor_sha256(value: tf.Tensor) -> str:
    encoded = tf.io.serialize_tensor(tf.convert_to_tensor(value)).numpy()
    return hashlib.sha256(encoded).hexdigest()


def _prepared(model, *, time_steps: int, particle_count: int, seed: int):
    state_dimension = model.state_dim()
    observation_dimension = model.observation_dim()
    generator = tf.random.Generator.from_seed(int(seed))
    initial_noise = generator.normal([1, particle_count, state_dimension], dtype=DTYPE)
    transition_noise = generator.normal(
        [1, max(0, time_steps - 1), particle_count, state_dimension], dtype=DTYPE
    )
    observation_noise = generator.normal([time_steps, observation_dimension], dtype=DTYPE)
    theta = tf.zeros([3], DTYPE)
    simulation = model.simulate_from_standard_normals(
        theta,
        generator.normal([state_dimension], dtype=DTYPE),
        generator.normal([time_steps - 1, state_dimension], dtype=DTYPE),
        observation_noise,
    )
    residual_design = generator.normal(
        [1, time_steps, particle_count, state_dimension], dtype=DTYPE
    )
    residual_design -= tf.reduce_mean(residual_design, axis=2, keepdims=True)
    return theta, simulation, {
        "observations": simulation["observations"],
        "initial_noise": initial_noise,
        "transition_noise": transition_noise,
        "fixed_reset_mask": tf.ones([1, time_steps], tf.bool),
        "residual_design": residual_design,
        "prepared_ridge": tf.fill([1, time_steps], tf.constant(1.0e-6, DTYPE)),
        "epsilon": tf.constant(0.25, DTYPE),
        "scaling": tf.constant(0.9, DTYPE),
    }


def _load_prepared(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "bayesfilter.latent_preclip_sir.prepared_inputs.v1":
        raise ValueError("prepared-input schema mismatch")
    if payload.get("target_id") != LATENT_PRECLIP_TARGET_ID:
        raise ValueError("prepared-input target mismatch")

    def tensor(record):
        dtype = tf.dtypes.as_dtype(record["dtype"])
        value = tf.convert_to_tensor(record["values"], dtype)
        value = tf.reshape(value, record["shape"])
        if _tensor_sha256(value) != record["serialized_tensor_sha256"]:
            raise ValueError("prepared tensor hash mismatch")
        return value

    theta = tensor(payload["theta"])
    prepared = {key: tensor(record) for key, record in payload["prepared"].items()}
    simulation = {
        key: tensor(record) for key, record in payload["simulator"].items()
    }
    return payload, theta, simulation, prepared


def _result(model, theta, prepared, args):
    spec = candidate.static_spec_from_model(model)
    tensors = candidate._as_prepared_tensors(prepared, spec)
    if args.canonical_registered_route:
        if not args.jit_compile:
            raise ValueError("canonical registered route requires XLA JIT")
        if args.sinkhorn_steps != candidate.CANONICAL_STEPS:
            raise ValueError("canonical registered route transport settings mismatch")

        def factory(parameter):
            return candidate.latent_sir_contract_e_canonical_value_and_score_tf(
                parameter,
                tensors["observations"],
                tensors["initial_noise"],
                tensors["transition_noise"],
                tensors["fixed_reset_mask"],
                tensors["residual_design"],
                tensors["prepared_ridge"],
                tensors["epsilon"],
                tensors["scaling"],
            )
    else:
        factory = candidate.make_latent_sir_contract_e_candidate(
            model,
            tensors,
            steps=args.sinkhorn_steps,
            row_chunk_size=args.row_chunk_size,
            col_chunk_size=args.col_chunk_size,
            jit_compile=args.jit_compile,
        )
    start = time.perf_counter()
    result = factory(theta)
    first_seconds = time.perf_counter() - start
    start = time.perf_counter()
    replay = factory(theta)
    replay_seconds = time.perf_counter() - start

    finite_difference = []
    endpoints = []
    for index in range(3):
        direction = tf.one_hot(index, 3, dtype=DTYPE)
        plus = factory(theta + args.fd_step * direction)
        minus = factory(theta - args.fd_step * direction)
        plus_value = float(plus["objective"].numpy())
        minus_value = float(minus["objective"].numpy())
        finite_difference.append((plus_value - minus_value) / (2.0 * args.fd_step))
        endpoints.append(
            {
                "parameter": candidate.PARAMETER_NAMES[index],
                "plus": plus_value,
                "minus": minus_value,
                "plus_valid": bool(tf.reduce_all(plus["valid_chart"]).numpy()),
                "minus_valid": bool(tf.reduce_all(minus["valid_chart"]).numpy()),
            }
        )
    score = np.asarray(result["score"].numpy())
    finite_difference_array = np.asarray(finite_difference)
    relative = np.abs(score - finite_difference_array) / np.maximum(
        np.maximum(np.abs(score), np.abs(finite_difference_array)), 1.0e-12
    )
    threshold = 0.05 * np.sqrt(3.0)
    devices = sorted({tensor.device for tensor in tf.nest.flatten(result) if tf.is_tensor(tensor)})
    return result, {
        "objective": float(result["objective"].numpy()),
        "score": score.tolist(),
        "finite_difference": finite_difference,
        "finite_difference_endpoints": endpoints,
        "finite_difference_policy": {
            "scope": "individual_coordinate_same_scalar_fd_only",
            "threshold": threshold,
            "relative_errors": relative.tolist(),
            "maximum_relative_error": float(np.max(relative)),
            "status": "pass" if float(np.max(relative)) <= threshold else "fail",
        },
        "valid_chart": result["valid_chart"].numpy().tolist(),
        "reset_valid_history": result["reset_valid_history"].numpy().tolist(),
        "minimum_mass_history": result["minimum_mass_history"].numpy().tolist(),
        "clip_boundary_away_history": result["clip_boundary_away_history"].numpy().tolist(),
        "increment_history": result["increment_history"].numpy().tolist(),
        "increment_score_history": result["increment_score_history"].numpy().tolist(),
        "first_call_seconds": first_seconds,
        "warm_replay_seconds": replay_seconds,
        "warm_replay_objective_equal": bool(
            tf.reduce_all(tf.equal(result["objective"], replay["objective"])).numpy()
        ),
        "output_devices": devices,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--time-steps", type=int, default=2)
    parser.add_argument("--particle-count", type=int, default=32)
    parser.add_argument("--seed", type=int, default=81103)
    parser.add_argument("--sinkhorn-steps", type=int, default=2)
    parser.add_argument("--fd-step", type=float, default=1.0e-5)
    parser.add_argument("--jit-compile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--canonical-registered-route", action="store_true")
    parser.add_argument("--prepared-input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.time_steps < 1 or args.particle_count < 4:
        raise ValueError("time_steps and particle_count are too small")
    chunks = select_transport_chunks(args.particle_count)
    args.row_chunk_size = chunks.row_chunk_size
    args.col_chunk_size = chunks.col_chunk_size
    args.output.parent.mkdir(parents=True, exist_ok=False)
    model = latent_preclip_zhao_cui_sir_austria_model()
    preparation_payload = None
    if args.prepared_input is None:
        theta, simulation, prepared = _prepared(
            model,
            time_steps=args.time_steps,
            particle_count=args.particle_count,
            seed=args.seed,
        )
    else:
        preparation_payload, theta, simulation, prepared = _load_prepared(
            args.prepared_input
        )
        prepared_config = preparation_payload["configuration"]
        if prepared_config["time_steps"] != args.time_steps:
            raise ValueError("prepared time_steps mismatch")
        if prepared_config["particle_count"] != args.particle_count:
            raise ValueError("prepared particle_count mismatch")
    result, diagnostic = _result(model, theta, prepared, args)
    gpu_devices = tf.config.list_physical_devices("GPU")
    payload = {
        "schema": "bayesfilter.latent_preclip_sir.contract_e_chol_diagnostic.v1",
        "status": (
            "PASS_BOUNDED_DIAGNOSTIC"
            if bool(tf.reduce_all(result["valid_chart"]).numpy())
            and diagnostic["finite_difference_policy"]["status"] == "pass"
            else "FAIL_BOUNDED_DIAGNOSTIC"
        ),
        "target_id": LATENT_PRECLIP_TARGET_ID,
        "route_id": candidate.CANDIDATE_ROUTE_ID,
        "route_status": candidate.CANDIDATE_STATUS,
        "route_execution_symbol": (
            "bayesfilter.highdim.ledh_contract_e_latent_sir_tf:"
            "latent_sir_contract_e_canonical_value_and_score_tf"
            if args.canonical_registered_route
            else "local_bound_candidate_factory"
        ),
        "configuration": {
            "time_steps": args.time_steps,
            "particle_count": args.particle_count,
            "seed": args.seed,
            "sinkhorn_steps": args.sinkhorn_steps,
            "row_chunk_size": args.row_chunk_size,
            "col_chunk_size": args.col_chunk_size,
            "jit_compile": args.jit_compile,
            "canonical_registered_route": args.canonical_registered_route,
            "dtype": "float64",
        },
        "diagnostic": diagnostic,
        "prepared_identity": {
            key: _tensor_sha256(value) for key, value in prepared.items()
        },
        "preparation": (
            {
                "path": str(args.prepared_input),
                "sha256": hashlib.sha256(args.prepared_input.read_bytes()).hexdigest(),
                "status": preparation_payload["status"],
            }
            if args.prepared_input is not None
            else {"status": "inline_device_specific_diagnostic_only"}
        ),
        "simulator": {
            "physical_path_sha256": _tensor_sha256(simulation["physical_path"]),
            "observations_sha256": _tensor_sha256(simulation["observations"]),
        },
        "environment": {
            "python": platform.python_version(),
            "tensorflow": tf.__version__,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
            "physical_gpus": [device.name for device in gpu_devices],
            "trust_basis": (
                "owner_designated_managed_session_visible_gpu_trusted"
                if gpu_devices
                else "deliberate_cpu_only_reference_preflight"
            ),
            "git_commit": _git_output("rev-parse", "HEAD"),
            "git_dirty": bool(_git_output("status", "--short")),
        },
        "command": " ".join(os.sys.argv),
        "nonclaims": [
            "not canonical Contract E admission",
            "not fixed-TTSIRT total-score closure",
            "not full-horizon correctness",
            "not HMC or leaderboard readiness",
            "same-scalar FD is not an oracle comparison",
        ],
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
