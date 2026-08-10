#!/usr/bin/env python3
"""Diagnostic Austria GenUT HMC smoke with a transferred NeuTra Gaussian force."""

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
from typing import Any, Mapping

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tensorflow as tf


PLAN = Path(
    "docs/plans/"
    "bayesfilter-austria-genut-neutra-value-surrogate-strategy-2026-08-03.md"
)
TUNING_ARTIFACT = Path(
    "docs/benchmarks/artifacts/genut_austria_antithetic_ensemble_20260803/"
    "tuning_attempt01/result.json"
)
TRANSPORT_RESULT = Path(
    "docs/plans/artifacts/multimodel-neutra-filter-posterior-20260715/"
    "phase-p6/SIR-SGQF/training/final/dim3_lr1e3/attempt-01/result.json"
)
EXPECTED_TRANSPORT_RESULT_SHA256 = (
    "c69b4e4e02b68d13be74f7a87ffc0ec9b1d6a47bc8438d56c048577a78531854"
)
EXPECTED_TRANSPORT_PAYLOAD_SHA256 = (
    "2ddff1ed2521ec674e64665bb8882a84ebc767e0850d677db72fa05a7e5ccdf4"
)
EXPECTED_TRANSPORT_HASH = (
    "dbd29efe786ec23c7b1098ba95ec6cad3a439b4889e04c67eeb2127965949c89"
)
MODEL_ID = "austria_sir_T20"
FIXED_NOISE_SEED = 140000
STEP_SIZES = (0.1, 0.2, 0.4)
NUM_LEAPFROG_STEPS = 10
CHAIN_COUNT = 4
TRANSITIONS = 32
REPEAT_COUNT = 8
INITIAL_OFFSET_SCALES = (1.0, 0.5, 0.25, 0.125, 0.0)
INITIAL_Z_OFFSETS = (
    (0.0, 0.0, 0.0),
    (0.10, -0.10, 0.08),
    (-0.10, 0.10, -0.08),
    (0.16, 0.08, -0.12),
)
POOLED_ACCEPTANCE_MIN = 0.5
PER_CHAIN_ACCEPTANCE_MIN = 0.25
PER_CHAIN_MOVE_MIN = 4
NORMALIZED_ESJD_MIN = 0.01


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _load_json(path: Path) -> Mapping[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _json_value(value: Any) -> Any:
    if isinstance(value, tf.Tensor):
        return _json_value(value.numpy().tolist())
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return value


def _load_transport() -> Any:
    from bayesfilter.inference.neutra_artifacts import load_frozen_neutra_artifact

    result_path = ROOT / TRANSPORT_RESULT
    if _sha256(result_path) != EXPECTED_TRANSPORT_RESULT_SHA256:
        raise RuntimeError("frozen SIR-SGQF transport result hash drift")
    result = _load_json(result_path)
    payload_path = ROOT / Path(result["payload"]["path"])
    if (
        _sha256(payload_path) != EXPECTED_TRANSPORT_PAYLOAD_SHA256
        or result["payload"]["file_sha256"] != EXPECTED_TRANSPORT_PAYLOAD_SHA256
    ):
        raise RuntimeError("frozen SIR-SGQF transport payload hash drift")
    payload = _load_json(payload_path)
    loaded = load_frozen_neutra_artifact(
        payload,
        expected_target_signature=str(payload["target_signature"]),
    )
    if (
        loaded.manifest.transport_hash != EXPECTED_TRANSPORT_HASH
        or result["transport_hash"] != EXPECTED_TRANSPORT_HASH
    ):
        raise RuntimeError("frozen SIR-SGQF transport identity drift")
    return loaded


def _load_genut_scope() -> tuple[Mapping[str, Any], Mapping[str, Any], Any]:
    from docs.benchmarks import run_genut_austria_antithetic_ensemble as genut
    from docs.benchmarks.run_moment_retuned_genut_whole_leaderboard import (
        _build_targets,
    )

    target = _build_targets()[MODEL_ID]
    if target["source_observation_sha256"] != genut.EXPECTED_OBSERVATION_SHA256:
        raise RuntimeError("Austria observation identity drift")
    tuning_path = ROOT / TUNING_ARTIFACT
    tuning = _load_json(tuning_path)
    controls = dict(genut._validate_tuning_payload(tuning))
    identity = genut._issue_current_identity(target, controls)
    genut._validate_tuning_payload(tuning, expected_identity=identity)
    evaluator = genut._make_current_evaluator(target, controls)
    return target, controls, evaluator


def _make_target(loaded: Any, target: Mapping[str, Any], evaluator: Any) -> Any:
    from docs.benchmarks import run_genut_austria_antithetic_ensemble as genut

    initial_noise, process_noise = genut._noise(FIXED_NOISE_SEED)
    observations = target["observations"]
    design = target["design"]

    def endpoint(position: tf.Tensor) -> tf.Tensor:
        z = tf.ensure_shape(tf.convert_to_tensor(position, tf.float64), [CHAIN_COUNT, 3])
        theta = loaded.transport.forward_batch(z)

        def scalar_log_posterior(theta_row: tf.Tensor) -> tf.Tensor:
            value, _unused_score, diagnostics = evaluator(
                tf.cast(theta_row, tf.float32),
                observations,
                initial_noise,
                process_noise,
                design,
            )
            prior, _unused_prior_score = sir_prior_value_score(theta_row[tf.newaxis, :])
            valid = tf.logical_and(
                diagnostics["program_valid"], tf.math.is_finite(value)
            )
            posterior = tf.cast(value, tf.float64) + prior[0]
            return tf.where(valid, posterior, tf.constant(-float("inf"), tf.float64))

        posterior = tf.map_fn(
            scalar_log_posterior,
            theta,
            fn_output_signature=tf.TensorSpec([], tf.float64),
            parallel_iterations=1,
        )
        logdet = loaded.transport.log_abs_det_jacobian_batch(z)
        transformed = posterior + logdet
        return tf.where(
            tf.math.is_finite(transformed),
            -transformed,
            tf.constant(float("inf"), tf.float64),
        )

    from bayesfilter.inference.neural_force_hmc import FrozenTargetPotential
    from bayesfilter.testing.sir_filter_neutra_target_design_tf import (
        sir_prior_value_score,
    )

    return FrozenTargetPotential(
        function=endpoint,
        identity=(
            "austria-genut-n1008-seed140000-current-tuned-value-score-discarded-"
            f"sir-sgqf-transport-{EXPECTED_TRANSPORT_HASH}"
        ),
        coordinate_system="transformed",
        includes_chart_log_jacobian=True,
        deterministic=True,
    )


def _repeatability(target: Any, initial: tf.Tensor) -> Mapping[str, Any]:
    rows = [tf.convert_to_tensor(target.function(initial), tf.float64) for _ in range(REPEAT_COUNT)]
    stack = tf.stack(rows)
    first = stack[0]
    exact = tf.reduce_all(tf.equal(stack, first[tf.newaxis, :]))
    finite = tf.reduce_all(tf.math.is_finite(stack))
    return {
        "repeat_count": REPEAT_COUNT,
        "values": stack,
        "all_finite": finite,
        "bitwise_equal_within_process": exact,
        "maximum_absolute_difference": tf.reduce_max(tf.abs(stack - first[tf.newaxis, :])),
    }


def _initialization_preflight(
    target: Any,
    loaded: Any,
    center: tf.Tensor,
) -> tuple[tf.Tensor, Mapping[str, Any]]:
    offsets = tf.constant(INITIAL_Z_OFFSETS, tf.float64)
    rows = []
    selected = None
    for scale in INITIAL_OFFSET_SCALES:
        positions = center[tf.newaxis, :] + tf.cast(scale, tf.float64) * offsets
        theta = loaded.transport.forward_batch(positions)
        potentials = tf.convert_to_tensor(target.function(positions), tf.float64)
        finite = tf.reduce_all(tf.math.is_finite(potentials))
        rows.append(
            {
                "offset_scale": scale,
                "positions": positions,
                "theta": theta,
                "potentials": potentials,
                "all_finite": finite,
            }
        )
        if selected is None and bool(finite.numpy()):
            selected = positions
    return selected, {
        "scale_ladder": INITIAL_OFFSET_SCALES,
        "selection_rule": "largest_predeclared_offset_scale_with_four_finite_endpoints",
        "rows": rows,
        "selected_scale": next(
            (
                row["offset_scale"]
                for row in rows
                if bool(row["all_finite"].numpy())
            ),
            None,
        ),
        "repair_classification": (
            "posthoc_localized_initialization_spread_repair_after_attempt02"
        ),
    }


def _reversibility(
    force: Any,
    initial: tf.Tensor,
) -> Mapping[str, Any]:
    from bayesfilter.inference.neural_force_hmc import (
        NeuralForceHMCConfig,
        neural_force_proposal,
    )

    momentum = tf.constant(
        (
            (0.20, -0.10, 0.05),
            (-0.15, 0.08, 0.12),
            (0.04, 0.18, -0.09),
            (-0.11, -0.06, 0.17),
        ),
        tf.float64,
    )
    config = NeuralForceHMCConfig(
        step_size=0.4,
        num_leapfrog_steps=NUM_LEAPFROG_STEPS,
        inverse_mass_diagonal=(1.0, 1.0, 1.0),
        dtype="float64",
    )
    forward = neural_force_proposal(initial, momentum, force, config)
    reverse = neural_force_proposal(forward.position, forward.momentum, force, config)
    position_error = tf.reduce_max(tf.abs(reverse.position - initial))
    momentum_error = tf.reduce_max(tf.abs(reverse.momentum - momentum))
    return {
        "maximum_position_error": position_error,
        "maximum_momentum_error": momentum_error,
        "tolerance": 1.0e-12,
        "passed": tf.logical_and(position_error <= 1.0e-12, momentum_error <= 1.0e-12),
    }


def _summarize_chain(
    chain: Any,
    initial: tf.Tensor,
    *,
    step_size: float,
    elapsed_seconds: float,
) -> Mapping[str, Any]:
    accepted = tf.convert_to_tensor(chain.accepted, tf.bool)
    positions = tf.convert_to_tensor(chain.positions, tf.float64)
    previous = tf.concat((initial[tf.newaxis, :, :], positions[:-1]), axis=0)
    squared_jump = tf.reduce_sum(tf.square(positions - previous), axis=-1)
    accepted_squared_jump = tf.where(accepted, squared_jump, tf.zeros_like(squared_jump))
    acceptance_by_chain = tf.reduce_mean(tf.cast(accepted, tf.float64), axis=0)
    moves_by_chain = tf.reduce_sum(
        tf.cast(tf.logical_and(accepted, squared_jump > 0.0), tf.int32), axis=0
    )
    pooled_acceptance = tf.reduce_mean(tf.cast(accepted, tf.float64))
    normalized_esjd = tf.reduce_mean(accepted_squared_jump) / 3.0
    reconstructed_delta_h = (
        chain.final_potential
        + chain.final_kinetic
        - chain.initial_potential
        - chain.initial_kinetic
    )
    finite_energy_rows = tf.logical_and(
        tf.math.is_finite(chain.delta_h), tf.math.is_finite(reconstructed_delta_h)
    )
    energy_residual = tf.where(
        finite_energy_rows,
        tf.abs(chain.delta_h - reconstructed_delta_h),
        tf.zeros_like(chain.delta_h),
    )
    energy_identity_error = tf.reduce_max(energy_residual)
    finite_absolute_delta_h = tf.where(
        tf.math.is_finite(chain.delta_h),
        tf.abs(chain.delta_h),
        tf.zeros_like(chain.delta_h),
    )
    endpoint_counts_valid = tf.reduce_all(tf.equal(chain.endpoint_call_count, 1))
    force_counts_valid = tf.reduce_all(
        tf.equal(chain.force_call_count, NUM_LEAPFROG_STEPS + 1)
    )
    occupied_finite = tf.reduce_all(tf.math.is_finite(chain.potentials))
    viability = tf.reduce_all(
        tf.stack(
            (
                pooled_acceptance >= POOLED_ACCEPTANCE_MIN,
                tf.reduce_all(acceptance_by_chain >= PER_CHAIN_ACCEPTANCE_MIN),
                tf.reduce_all(moves_by_chain >= PER_CHAIN_MOVE_MIN),
                normalized_esjd >= NORMALIZED_ESJD_MIN,
                energy_identity_error == 0.0,
                endpoint_counts_valid,
                force_counts_valid,
                occupied_finite,
            )
        )
    )
    return {
        "step_size": step_size,
        "num_leapfrog_steps": NUM_LEAPFROG_STEPS,
        "transitions_per_chain": TRANSITIONS,
        "pooled_acceptance": pooled_acceptance,
        "acceptance_by_chain": acceptance_by_chain,
        "nonzero_accepted_moves_by_chain": moves_by_chain,
        "normalized_accepted_esjd": normalized_esjd,
        "maximum_finite_absolute_delta_h": tf.reduce_max(finite_absolute_delta_h),
        "nonfinite_proposed_endpoint_count": tf.reduce_sum(
            tf.cast(tf.logical_not(tf.math.is_finite(chain.final_potential)), tf.int32)
        ),
        "full_energy_identity_max_error": energy_identity_error,
        "full_energy_identity_scope": "finite_proposed_endpoints_only",
        "finite_energy_reconstruction_count": tf.reduce_sum(
            tf.cast(finite_energy_rows, tf.int32)
        ),
        "occupied_potentials_all_finite": occupied_finite,
        "proposed_endpoint_finite_fraction": tf.reduce_mean(
            tf.cast(tf.math.is_finite(chain.final_potential), tf.float64)
        ),
        "one_endpoint_batch_call_per_transition": endpoint_counts_valid,
        "force_calls_equal_L_plus_one": force_counts_valid,
        "endpoint_batch_invocations": TRANSITIONS,
        "endpoint_scalar_evaluations": TRANSITIONS * CHAIN_COUNT,
        "elapsed_seconds": elapsed_seconds,
        "viability_passed": viability,
    }


def run(
    output_root: Path,
    *,
    step_sizes: tuple[float, ...] = STEP_SIZES,
) -> Mapping[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"output root must be fresh: {output_root}")
    output_root.mkdir(parents=True)
    started = time.perf_counter()
    started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    # Repository modules define TensorFlow constants at import time, so establish
    # allocator policy before importing any TensorFlow-bearing BayesFilter code.
    from bayesfilter.runtime.gpu_memory_policy import (
        configure_tensorflow_gpu_memory_growth,
    )

    memory_policy = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    tf.config.set_soft_device_placement(False)
    tf.config.experimental.enable_tensor_float_32_execution(True)
    logical_devices = tf.config.list_logical_devices("GPU")
    if not logical_devices:
        raise RuntimeError("behavioral smoke requires a logical GPU")

    from bayesfilter.inference.neural_force_hmc import (
        FrozenPositionOnlyForce,
        NeuralForceHMCConfig,
        sample_neural_force_hmc,
    )
    from bayesfilter.runtime import atomic_write_json
    from docs.benchmarks import run_genut_austria_antithetic_ensemble as genut

    loaded = _load_transport()
    genut_target, controls, evaluator = _load_genut_scope()
    target = _make_target(loaded, genut_target, evaluator)
    force = FrozenPositionOnlyForce(
        function=lambda position: position,
        identity="transferred-sir-sgqf-neutra-standard-normal-gradient-v1",
    )
    center = loaded.transport.inverse_theta_to_z_batch(tf.zeros((1, 3), tf.float64))[0]
    initial, initialization_preflight = _initialization_preflight(
        target, loaded, center
    )
    if initial is None:
        atomic_write_json(
            output_root / "initialization_preflight.json",
            _json_value(initialization_preflight),
        )
        raise RuntimeError("no initialization offset scale had four finite endpoints")

    repeatability = _repeatability(target, initial)
    reversibility = _reversibility(force, initial)
    if not bool(repeatability["all_finite"].numpy()):
        raise RuntimeError("initial endpoint preflight returned a nonfinite potential")
    if not bool(repeatability["bitwise_equal_within_process"].numpy()):
        raise RuntimeError("initial endpoint preflight was not bitwise repeatable")
    if not bool(reversibility["passed"].numpy()):
        raise RuntimeError("Gaussian-force proposal failed reversibility preflight")

    initial_potential = tf.convert_to_tensor(target.function(initial), tf.float64)
    rows = []
    for index, step_size in enumerate(step_sizes):
        config = NeuralForceHMCConfig(
            step_size=step_size,
            num_leapfrog_steps=NUM_LEAPFROG_STEPS,
            inverse_mass_diagonal=(1.0, 1.0, 1.0),
            dtype="float64",
        )

        @tf.function(jit_compile=True, reduce_retracing=True)
        def compiled(position: tf.Tensor, potential: tf.Tensor) -> Any:
            return sample_neural_force_hmc(
                position,
                potential,
                force,
                target,
                config,
                num_warmup=0,
                num_results=TRANSITIONS,
                seed=tf.constant((20260804, 41000 + index), tf.int32),
            )

        run_started = time.perf_counter()
        chain = compiled(initial, initial_potential)
        tf.reduce_sum(chain.positions).numpy()
        elapsed = time.perf_counter() - run_started
        rows.append(
            _summarize_chain(
                chain,
                initial,
                step_size=step_size,
                elapsed_seconds=elapsed,
            )
        )

    passed = tf.reduce_any(tf.stack([row["viability_passed"] for row in rows]))
    allocator = tf.config.experimental.get_memory_info("GPU:0")
    completed_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    source_paths = (
        Path(__file__).relative_to(ROOT),
        PLAN,
        Path("bayesfilter/inference/neural_force_hmc.py"),
        Path("bayesfilter/inference/neutra_artifacts.py"),
        Path("bayesfilter/highdim/cubature_genut_filter.py"),
        Path("bayesfilter/highdim/cubature_genut_adapters.py"),
        TUNING_ARTIFACT,
        TRANSPORT_RESULT,
    )
    payload = {
        "schema": "bayesfilter.genut_austria_neutra_zero_force_smoke.v1",
        "status": (
            "BEHAVIORAL_SMOKE_PASSED_AT_LEAST_ONE_CONFIGURATION"
            if bool(passed.numpy())
            else "BEHAVIORAL_SMOKE_NO_CONFIGURATION_PASSED"
        ),
        "passed": passed,
        "started_utc": started_utc,
        "completed_utc": completed_utc,
        "wall_time_seconds": time.perf_counter() - started,
        "git_commit": _git_commit(),
        "working_tree_note": "source hashes bind the dirty working-tree implementation",
        "host": platform.node(),
        "python_executable": sys.executable,
        "tensorflow_version": tf.__version__,
        "plan": PLAN.as_posix(),
        "question": (
            "Can the transferred SIR-SGQF NeuTra chart plus Gaussian force make "
            "useful MH-corrected proposals for the frozen Austria GenUT endpoint?"
        ),
        "target": {
            "model_id": MODEL_ID,
            "particle_count": genut.PARTICLE_COUNT,
            "horizon": genut.HORIZON,
            "fixed_noise_seed": FIXED_NOISE_SEED,
            "controls": controls,
            "prior": "independent Normal(0, 0.5^2) in three log-scale coordinates",
            "endpoint_computation": (
                "finite_value_score value used; returned score computed but discarded"
            ),
        },
        "proposal": {
            "force": "g(z)=z",
            "force_uses_genut_score": False,
            "transport_role": "different-filter diagnostic warm start only",
            "transport_hash": EXPECTED_TRANSPORT_HASH,
            "step_sizes": step_sizes,
            "num_leapfrog_steps": NUM_LEAPFROG_STEPS,
            "initial_z_center": center,
            "initial_z_offsets": INITIAL_Z_OFFSETS,
        },
        "criteria": {
            "pooled_acceptance_min": POOLED_ACCEPTANCE_MIN,
            "per_chain_acceptance_min": PER_CHAIN_ACCEPTANCE_MIN,
            "per_chain_nonzero_accepted_moves_min": PER_CHAIN_MOVE_MIN,
            "normalized_accepted_esjd_min": NORMALIZED_ESJD_MIN,
        },
        "initialization_preflight": initialization_preflight,
        "repeatability": repeatability,
        "reversibility": reversibility,
        "rows": rows,
        "device": {
            "logical_devices": [device.name for device in logical_devices],
            "dtype": {"hmc": "float64", "genut": "float32"},
            "tf32_enabled": tf.config.experimental.tensor_float_32_execution_enabled(),
            "jit_compile": True,
            "trust_basis": "owner_designated_managed_session_visible_gpu_trusted",
        },
        "memory_policy": memory_policy,
        "gpu_allocator": {key: int(value) for key, value in allocator.items()},
        "source_sha256": {
            path.as_posix(): _sha256(ROOT / path) for path in source_paths
        },
        "inference_status": {
            "hard_veto_screen": "passed" if bool(passed.numpy()) else "failed",
            "statistically_supported_ranking": False,
            "descriptive_only_differences": (
                "acceptance, movement, energy tails, and runtimes across step sizes"
            ),
            "default_readiness": False,
            "next_evidence_needed": (
                "true tangent-free batch-native GenUT value endpoint and fuller canary"
                if bool(passed.numpy())
                else "target-specific chart or value-residual diagnostic"
            ),
        },
        "nonclaims": (
            "no speedup claim because endpoint still computes the GenUT score",
            "no target-specific GenUT NeuTra training claim",
            "no convergence, posterior agreement, sampler ranking, or default claim",
            "no GenUT score correctness claim",
        ),
    }
    json_payload = _json_value(payload)
    atomic_write_json(output_root / "result.json", json_payload)
    manifest = {
        "schema": "bayesfilter.serious_run_manifest.v1",
        "git_commit": payload["git_commit"],
        "command": " ".join(sys.argv),
        "environment": sys.executable,
        "cpu_gpu_status": json_payload["device"],
        "memory_policy": json_payload["memory_policy"],
        "data_version": genut.EXPECTED_OBSERVATION_SHA256,
        "random_seeds": {
            "fixed_genut_noise": FIXED_NOISE_SEED,
            "hmc": [[20260804, 41000 + index] for index in range(len(step_sizes))],
        },
        "wall_time_seconds": json_payload["wall_time_seconds"],
        "output_artifact_paths": [str(output_root / "result.json")],
        "plan_file": PLAN.as_posix(),
        "result_file": str(output_root / "result.json"),
    }
    atomic_write_json(output_root / "run_manifest.json", manifest)
    return json_payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--step-sizes",
        type=float,
        nargs="+",
        default=STEP_SIZES,
    )
    args = parser.parse_args()
    step_sizes = tuple(float(value) for value in args.step_sizes)
    if not step_sizes or any(value not in STEP_SIZES for value in step_sizes):
        raise ValueError(f"step sizes must be selected from {STEP_SIZES}")
    result = run(args.output_root, step_sizes=step_sizes)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
