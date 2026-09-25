#!/usr/bin/env python3
"""Canonical SQMC ancestry characterization against the exact LGSSM oracle."""

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

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tensorflow as tf

from bayesfilter.runtime.gpu_memory_policy import (
    configure_tensorflow_gpu_memory_growth,
)

PLAN = Path("docs/plans/sqmc-oracle-comparison-master-program-v2-2026-09-09.md")
ARTIFACT_ROOT = Path(
    "docs/benchmarks/artifacts/sqmc-oracle-characterization-canonical-20260909"
)
SCHEMA = "bayesfilter.sqmc_oracle_characterization.canonical.v1"
PROGRAM = "canonical_score_sqmc_fp64_untuned_diagnostic"
TUNING = "UNTUNED: no exact-scope repository-issued tuning artifact"
DTYPE = tf.float64
THETA = (0.9, 0.8, 0.7, 0.6, 0.8)
PARAMETER_NAMES = ("phi1", "phi2", "phi3", "q_scale", "r_scale")
ROUTES = (
    "iid_dual_cap",
    "previous_inverse_cdf",
    "repaired_fixed_previous_controls",
    "repaired_permutation",
)
ANCESTRY = {
    "iid_dual_cap": "existing_one_to_one",
    "previous_inverse_cdf": "hilbert_inverse_cdf",
    "repaired_fixed_previous_controls": "hilbert_permutation_one_to_one",
    "repaired_permutation": "hilbert_permutation_one_to_one",
}
BASE_CONTROLS = {
    "correction_steps": 4,
    "correction_strength": 0.2,
    "pairwise_steps": 4,
    "pairwise_strength": 0.02,
    "pairwise_rms_cap": 2.0,
    "coordinate_cap": 0.98,
    "coordinate_cap_power": 8,
}
CONSERVATIVE_CONTROLS = {
    "correction_steps": 3,
    "correction_strength": 0.15,
    "pairwise_steps": 3,
    "pairwise_strength": 0.01,
    "pairwise_rms_cap": 1.5,
    "coordinate_cap": 0.97,
    "coordinate_cap_power": 6,
}


def _safe(value: Any) -> Any:
    if hasattr(value, "numpy"):
        value = value.numpy()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tensor_sha256(value: tf.Tensor) -> str:
    encoded = tf.io.serialize_tensor(tf.convert_to_tensor(value)).numpy()
    return hashlib.sha256(encoded).hexdigest()


def _attempt_directory(stage: str) -> Path:
    root = ROOT / ARTIFACT_ROOT
    root.mkdir(parents=True, exist_ok=True)
    for index in range(1, 100):
        candidate = root / f"{stage}_attempt{index:02d}"
        try:
            candidate.mkdir()
        except FileExistsError:
            continue
        return candidate
    raise RuntimeError("no unused attempt directory remains")


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _design(particle_count: int, dimension: int) -> tf.Tensor:
    if particle_count % (2 * dimension):
        raise ValueError("particle count must be divisible by 2 * state dimension")
    base = tf.concat([tf.eye(dimension, dtype=DTYPE), -tf.eye(dimension, dtype=DTYPE)], axis=0)
    return tf.tile(base, [particle_count // (2 * dimension), 1])


def _oracle(observations: tf.Tensor, theta: tf.Tensor) -> dict[str, tf.Tensor]:
    from bayesfilter.highdim.ledh_kalman_oracle_tf import (
        kalman_oracle_value_and_score,
    )

    observation_matrix = tf.constant(
        [[1.0, 0.25, -0.15], [0.2, 1.1, 0.3], [-0.1, 0.35, 0.9]],
        DTYPE,
    )

    def parameters(value: tf.Tensor) -> dict[str, tf.Tensor]:
        return {
            "transition_matrix": tf.linalg.diag(value[:3]),
            "process_covariance": tf.square(value[3]) * tf.eye(3, dtype=DTYPE),
            "observation_matrix": observation_matrix,
            "observation_covariance": tf.square(value[4]) * tf.eye(3, dtype=DTYPE),
            "initial_mean": tf.zeros([3], DTYPE),
            "initial_covariance": tf.eye(3, dtype=DTYPE),
        }

    return kalman_oracle_value_and_score(
        observations, theta, parameters, dtype=DTYPE
    )


def _inputs(seed: int, horizon: int, particle_count: int) -> dict[str, dict[str, tf.Tensor]]:
    from bayesfilter.highdim.sqmc_tf import (
        randomized_halton_gaussian,
        randomized_halton_joint,
    )

    iid_initial = tf.random.stateless_normal(
        [particle_count, 3], [seed, 101], dtype=DTYPE
    )
    iid_process = tf.stack(
        [
            tf.random.stateless_normal(
                [particle_count, 3], [seed, 1001 + time_index], dtype=DTYPE
            )
            for time_index in range(horizon)
        ]
    )
    rqmc_initial = randomized_halton_gaussian(
        num_particles=particle_count,
        dimension=3,
        seed=seed,
        salt=301,
        dtype=DTYPE,
    )
    process_rows = []
    ancestor_rows = []
    joint_hashes = []
    for time_index in range(horizon):
        raw, ancestors, innovations = randomized_halton_joint(
            num_particles=particle_count,
            state_dimension=3,
            seed=seed,
            salt=3001 + time_index,
            dtype=DTYPE,
        )
        process_rows.append(tf.math.ndtri(innovations))
        ancestor_rows.append(ancestors)
        joint_hashes.append(_tensor_sha256(raw))
    rqmc_process = tf.stack(process_rows)
    rqmc_ancestors = tf.stack(ancestor_rows)
    ignored = tf.zeros([horizon, particle_count], DTYPE)
    return {
        "iid": {
            "initial": iid_initial,
            "process": iid_process,
            "ancestors": ignored,
            "joint_hashes": [],
        },
        "sqmc": {
            "initial": rqmc_initial,
            "process": rqmc_process,
            "ancestors": rqmc_ancestors,
            "joint_hashes": joint_hashes,
        },
    }


def _route_controls(route: str) -> dict[str, Any]:
    from bayesfilter.highdim.ledh_alg1_contract import LEDH_PRODUCTION_PROGRAM_V1

    controls = dict(LEDH_PRODUCTION_PROGRAM_V1["score"])
    controls.update(
        CONSERVATIVE_CONTROLS if route == "repaired_permutation" else BASE_CONTROLS
    )
    controls.update(
        {
            "reset_epsilon": 2.0,
            "reset_ridge": 1.0e-5,
            "correction_lm_scale_floor": 1.0e-4,
            "coordinate_cap_power": (
                6 if route == "repaired_permutation" else 8
            ),
        }
    )
    return controls


def _make_evaluator(route: str, horizon: int, particle_count: int, xla: bool):
    from bayesfilter.highdim.ledh_canonical_models_tf import (
        diagonal_lgssm_canonical_model,
    )
    from bayesfilter.highdim.ledh_canonical_score_tf import (
        canonical_value_and_analytical_score,
    )

    theta = tf.constant(THETA, DTYPE)
    model, set_score_direction = diagonal_lgssm_canonical_model(theta)
    controls = _route_controls(route)
    design = _design(particle_count, 3)
    ancestry_policy = ANCESTRY[route]

    if xla:
        @tf.function(
            input_signature=(
                tf.TensorSpec([horizon, 3], DTYPE),
                tf.TensorSpec([particle_count, 3], DTYPE),
                tf.TensorSpec([horizon, particle_count, 3], DTYPE),
                tf.TensorSpec([horizon, particle_count], DTYPE),
            ),
            jit_compile=True,
            autograph=False,
        )
        def evaluate(observations, initial_states, process_noise, ancestor_uniforms):
            values = []
            scores = []
            with tf.device("/GPU:0"):
                initial_covariances = tf.eye(
                    3, batch_shape=[particle_count], dtype=DTYPE
                )
                for direction_index in range(len(THETA)):
                    set_score_direction(
                        tf.one_hot(direction_index, len(THETA), dtype=DTYPE)
                    )
                    value, score = canonical_value_and_analytical_score(
                        model,
                        theta,
                        initial_states,
                        initial_covariances,
                        process_noise,
                        observations,
                        flow_substeps=8,
                        with_score=True,
                        reset_policy="contract_e",
                        reset_design=design,
                        reset_epsilon=controls["reset_epsilon"],
                        reset_sinkhorn_steps=controls["reset_sinkhorn_steps"],
                        reset_balance_steps=controls["reset_balance_steps"],
                        reset_ridge=controls["reset_ridge"],
                        correction_steps=controls["correction_steps"],
                        correction_strength=controls["correction_strength"],
                        correction_lm_damping=controls["correction_lm_damping"],
                        correction_lm_scale_floor=controls[
                            "correction_lm_scale_floor"
                        ],
                        correction_trust_radius=controls[
                            "correction_trust_radius"
                        ],
                        pairwise_steps=controls["pairwise_steps"],
                        pairwise_strength=controls["pairwise_strength"],
                        pairwise_rms_cap=controls["pairwise_rms_cap"],
                        coordinate_cap=controls["coordinate_cap"],
                        coordinate_cap_power=controls["coordinate_cap_power"],
                        ancestry_policy=ancestry_policy,
                        process_ancestor_uniforms=ancestor_uniforms,
                        state_map_policy="adaptive_empirical",
                        hilbert_bits=12,
                    )
                    values.append(value)
                    scores.append(score[0])
            return tf.stack(values), tf.stack(scores)
    else:
        def evaluate(observations, initial_states, process_noise, ancestor_uniforms):
            values = []
            scores = []
            with tf.device("/GPU:0"):
                initial_covariances = tf.eye(
                    3, batch_shape=[particle_count], dtype=DTYPE
                )
                for direction_index in range(len(THETA)):
                    set_score_direction(
                        tf.one_hot(direction_index, len(THETA), dtype=DTYPE)
                    )
                    value, score = canonical_value_and_analytical_score(
                        model,
                        theta,
                        initial_states,
                        initial_covariances,
                        process_noise,
                        observations,
                        flow_substeps=8,
                        with_score=True,
                        reset_policy="contract_e",
                        reset_design=design,
                        reset_epsilon=controls["reset_epsilon"],
                        reset_sinkhorn_steps=controls["reset_sinkhorn_steps"],
                        reset_balance_steps=controls["reset_balance_steps"],
                        reset_ridge=controls["reset_ridge"],
                        correction_steps=controls["correction_steps"],
                        correction_strength=controls["correction_strength"],
                        correction_lm_damping=controls["correction_lm_damping"],
                        correction_lm_scale_floor=controls[
                            "correction_lm_scale_floor"
                        ],
                        correction_trust_radius=controls[
                            "correction_trust_radius"
                        ],
                        pairwise_steps=controls["pairwise_steps"],
                        pairwise_strength=controls["pairwise_strength"],
                        pairwise_rms_cap=controls["pairwise_rms_cap"],
                        coordinate_cap=controls["coordinate_cap"],
                        coordinate_cap_power=controls["coordinate_cap_power"],
                        ancestry_policy=ancestry_policy,
                        process_ancestor_uniforms=ancestor_uniforms,
                        state_map_policy="adaptive_empirical",
                        hilbert_bits=12,
                    )
                    values.append(value)
                    scores.append(score[0])
            return tf.stack(values), tf.stack(scores)

    return evaluate, controls


def _cell(
    route: str,
    seed: int,
    observations: tf.Tensor,
    oracle: dict[str, tf.Tensor],
    evaluator: Any,
    controls: dict[str, Any],
    inputs: dict[str, dict[str, tf.Tensor]],
) -> dict[str, Any]:
    selected = inputs["iid" if route == "iid_dual_cap" else "sqmc"]
    started = time.perf_counter()
    values, score = evaluator(
        observations,
        selected["initial"],
        selected["process"],
        selected["ancestors"],
    )
    elapsed = time.perf_counter() - started
    value_spread = tf.reduce_max(values) - tf.reduce_min(values)
    finite = bool(tf.reduce_all(tf.math.is_finite(values)).numpy()) and bool(
        tf.reduce_all(tf.math.is_finite(score)).numpy()
    )
    oracle_score = tf.cast(oracle["score"], DTYPE)
    return {
        "route": route,
        "seed": seed,
        "program": PROGRAM,
        "tuning": TUNING,
        "ancestry_policy": ANCESTRY[route],
        "point_set": "iid_gaussian" if route == "iid_dual_cap" else "randomized_halton_joint",
        "configuration_role": (
            "control_family_ablation"
            if route == "repaired_permutation"
            else "ancestry_mechanism"
        ),
        "controls": controls,
        "value": values[0],
        "directional_values": values,
        "maximum_directional_value_disagreement": value_spread,
        "score": score,
        "oracle_value": oracle["value"],
        "oracle_score": oracle_score,
        "absolute_value_error": tf.abs(values[0] - oracle["value"]),
        "score_l2_error": tf.linalg.norm(score - oracle_score),
        "absolute_score_errors": tf.abs(score - oracle_score),
        "finite": finite,
        "value_direction_invariant": bool((value_spread <= tf.cast(1.0e-10, DTYPE)).numpy()),
        "initial_sha256": _tensor_sha256(selected["initial"]),
        "process_sha256": _tensor_sha256(selected["process"]),
        "ancestor_sha256": (
            None if route == "iid_dual_cap" else _tensor_sha256(selected["ancestors"])
        ),
        "joint_sha256": selected["joint_hashes"],
        "elapsed_seconds": elapsed,
        "value_device": values.device,
    }


def _git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True
    ).strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("smoke", "diagnostic"), required=True)
    parser.add_argument(
        "--routes", default=",".join(ROUTES),
        help="comma-separated subset of the four historical configurations",
    )
    args = parser.parse_args()
    routes = tuple(item.strip() for item in args.routes.split(",") if item.strip())
    if not routes or any(route not in ROUTES for route in routes):
        raise ValueError(f"routes must be a nonempty subset of {ROUTES}")

    horizon = 2 if args.stage == "smoke" else 20
    particle_count = 24 if args.stage == "smoke" else 1008
    seeds = (97701,) if args.stage == "smoke" else (97701, 97702)
    output_directory = _attempt_directory(args.stage)
    started = time.perf_counter()

    memory_policy = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    tf.config.experimental.enable_tensor_float_32_execution(True)
    logical_gpus = tf.config.list_logical_devices("GPU")
    if len(logical_gpus) != 1:
        raise RuntimeError("the characterization requires exactly one logical GPU")

    from bayesfilter.highdim.ledh_canonical_neutra_targets_tf import (
        _lgssm_frozen_observations,
    )

    observations = tf.cast(_lgssm_frozen_observations()[:horizon], DTYPE)
    theta = tf.constant(THETA, DTYPE)
    oracle = _oracle(observations, theta)
    cells = []
    use_xla = args.stage == "smoke"
    evaluators = {
        route: _make_evaluator(route, horizon, particle_count, use_xla)
        for route in routes
    }

    for seed in seeds:
        with tf.device("/CPU:0"):
            inputs = _inputs(seed, horizon, particle_count)
        for route in routes:
            evaluator, controls = evaluators[route]
            print(f"running stage={args.stage} route={route} seed={seed}", flush=True)
            try:
                cell = _cell(
                    route, seed, observations, oracle, evaluator, controls, inputs
                )
            except Exception as error:
                cells.append({
                    "route": route, "seed": seed, "program": PROGRAM,
                    "tuning": TUNING, "finite": False,
                    "failure_type": type(error).__name__, "failure": str(error),
                })
                _write_json(output_directory / "partial_result.json", {"cells": cells})
                raise
            cells.append(cell)
            _write_json(output_directory / "partial_result.json", {"cells": cells})

    all_valid = all(
        cell.get("finite") and cell.get("value_direction_invariant")
        for cell in cells
    )
    elapsed = time.perf_counter() - started
    configuration_status = {
        "program": PROGRAM,
        "program_class": "canonical analytical-score diagnostic variant",
        "production_differences": (
            "float64 rather than float32/TF32; no exact-scope tuning artifact; "
            "directions evaluated separately; repaired_permutation changes "
            "correction controls"
        ),
        "tuning": TUNING,
        "claim_status": "UNTUNED_DIAGNOSTIC_ONLY",
        "must_not_conclude": (
            "no statistical route ranking, superiority, production readiness, "
            "HMC benefit, or KSC generalization"
        ),
    }
    result = {
        "configuration_status": configuration_status,
        "schema": SCHEMA,
        "status": "PASS" if all_valid else "FAIL_VALIDITY",
        "stage": args.stage,
        "research_question": (
            "Do historical SQMC configurations show descriptive value/score "
            "error differences on the frozen canonical LGSSM target?"
        ),
        "inference_status": {
            "hard_veto_screen": "PASS" if all_valid else "FAIL",
            "statistically_supported_ranking": "NONE_TWO_SEEDS_OR_FEWER",
            "descriptive_only_differences": True,
            "default_readiness": False,
            "next_evidence": (
                "exact-scope tuning followed by untouched multi-seed paired "
                "uncertainty analysis"
            ),
        },
        "scope": {
            "model_id": "canonical_lgssm_m3",
            "data_id": "benchmark_lgssm_m3_T50_seed81100_prefix",
            "horizon": horizon,
            "particle_count": particle_count,
            "state_dimension": 3,
            "parameter_names": PARAMETER_NAMES,
            "theta": THETA,
            "routes": routes,
            "seeds": seeds,
            "dtype": DTYPE.name,
            "tf32_enabled": True,
            "flow_substeps": 8,
            "chunk_policy": "dpf_transport_exact_divisor_cap3000_v1",
            "chunk_extent": particle_count,
        },
        "oracle": {
            "kind": "exact_same-target Kalman innovation likelihood and score",
            "value": oracle["value"],
            "score": oracle["score"],
            "observation_sha256": _tensor_sha256(observations),
        },
        "cells": cells,
        "all_valid": all_valid,
        "wall_seconds": elapsed,
    }
    result_path = output_directory / "result.json"
    _write_json(result_path, result)

    source_paths = (
        Path("docs/benchmarks/run_sqmc_oracle_characterization.py"),
        Path("bayesfilter/highdim/ledh_canonical_score_tf.py"),
        PLAN,
    )
    manifest = {
        "configuration_status": configuration_status,
        "schema": f"{SCHEMA}.manifest",
        "git_commit": _git_output("rev-parse", "HEAD"),
        "git_branch": _git_output("branch", "--show-current"),
        "git_status": _git_output("status", "--short"),
        "command": [sys.executable, *sys.argv],
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV", "unknown"),
        "python": sys.version,
        "tensorflow": tf.__version__,
        "host": platform.node(),
        "device": logical_gpus[0].name,
        "gpu_memory_policy": memory_policy,
        "source_sha256": {
            str(path): _sha256(ROOT / path) for path in source_paths
        },
        "plan": str(PLAN),
        "result": str(result_path.relative_to(ROOT)),
        "output_directory": str(output_directory.relative_to(ROOT)),
        "wall_seconds": elapsed,
    }
    _write_json(output_directory / "manifest.json", manifest)
    partial = output_directory / "partial_result.json"
    if partial.exists():
        partial.unlink()
    print(
        f"{result['status']}: {len(cells)} cells; artifact={result_path}",
        flush=True,
    )
    return 0 if all_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
