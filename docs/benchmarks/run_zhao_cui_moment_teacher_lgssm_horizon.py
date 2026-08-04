"""Scope-specific LGSSM tuning and untouched claim node for the TT teacher."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

PREPARE_TEACHER_ONLY = "--prepare-teacher-only" in sys.argv
if PREPARE_TEACHER_ONLY:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
else:
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tensorflow as tf

from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth


MEMORY_POLICY = (
    {
        "schema": "bayesfilter.tensorflow.cpu_only_setup.v1",
        "mode": "cpu_only_teacher_preparation",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    if PREPARE_TEACHER_ONLY
    else configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
)
tf.config.experimental.enable_tensor_float_32_execution(False)

from bayesfilter.highdim import ledh_contract_e_canonical_lgssm_tf as canonical
from bayesfilter.highdim import ledh_contract_e_lgssm_preparation_tf as particle_preparation
from bayesfilter.highdim.ledh_contract_e_identity import (
    issue_moment_teacher_lgssm_contract_e_route_identity,
)
from bayesfilter.highdim.ledh_contract_e_tp_lgssm_tf import exact_kalman_value
from bayesfilter.highdim.transport_chunk_policy import select_transport_chunks
from bayesfilter.highdim.zhao_cui_moment_teacher_lgssm_tf import (
    MomentTeacherControls,
    freeze_teacher_scale_shift_indices,
    issue_moment_teacher_tuning_artifact,
    make_lgssm_tuning_scope,
    make_moment_teacher_lgssm_prepared_particles_tf,
    prepare_lgssm_teacher_inputs,
    route_identity_prepared_inputs,
)
from scripts.filtering_value_gradient_benchmark_generate_p8_datasets import (
    _lgssm_dataset,
)


SCHEMA = "bayesfilter.zhao_cui_moment_teacher_lgssm_horizon.v1"
PLAN = "docs/plans/bayesfilter-zhao-cui-moment-teacher-integration-campaign-plan-2026-07-30.md"
DATASET_SEED = 81100
THETA = (0.72, 0.55, 0.35, 0.35, 0.45)
PARTICLE_COUNT = 1024
RIDGE = 7.301568984985351e-09
CALIBRATION_SEED = 81900
VALIDATION_SEED = 81901
CLAIM_SEEDS = tuple(range(81910, 81916))
LABELS = ("value", "phi1", "phi2", "phi3", "q_scale", "r_scale")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(value):
    if isinstance(value, tf.Tensor):
        return _json(value.numpy())
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json(item) for item in value]
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    return str(value)


def _write_teacher_cache(path: Path, teacher: dict[str, tf.Tensor]) -> dict[str, object]:
    path.mkdir(parents=True, exist_ok=False)
    records = {}
    for name, value in sorted(teacher.items()):
        tensor = tf.convert_to_tensor(value)
        target = path / f"{name}.tensor"
        tf.io.write_file(str(target), tf.io.serialize_tensor(tensor))
        records[name] = {
            "dtype": tensor.dtype.name,
            "shape": tensor.shape.as_list(),
            "sha256": _sha256(target),
        }
    manifest = {"schema": "bayesfilter.moment_teacher_prepared_cache.v1", "tensors": records}
    (path / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def _read_teacher_cache(path: Path) -> dict[str, tf.Tensor]:
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    result = {}
    for name, record in manifest["tensors"].items():
        source = path / f"{name}.tensor"
        if _sha256(source) != record["sha256"]:
            raise ValueError(f"teacher cache hash mismatch: {name}")
        tensor = tf.io.parse_tensor(
            tf.io.read_file(str(source)), out_type=tf.dtypes.as_dtype(record["dtype"])
        )
        result[name] = tf.ensure_shape(tensor, record["shape"])
    return result


def _controls(horizon: int) -> tuple[MomentTeacherControls, ...]:
    balance = {2: 5, 10: 5, 50: 8}[horizon]
    common = dict(
        sinkhorn_steps=20,
        balance_steps=balance,
        correction_floor=1.0e-6,
        pairwise_correction_steps=0,
        pairwise_strength=0.0,
        pairwise_floor=1.0e-6,
        column_scale_floor=1.0e-6,
        condition_number_veto=1.0e10,
        fit_residual_veto=2.0,
    )
    return (
        MomentTeacherControls(
            correction_steps=1,
            correction_strength=0.01,
            tt_ridge=1.0e-4,
            **common,
        ),
        MomentTeacherControls(
            correction_steps=1,
            correction_strength=0.025,
            tt_ridge=1.0e-5,
            **common,
        ),
    )


def _prepare_particle(observations, horizon: int, seed: int, controls):
    chunks = select_transport_chunks(PARTICLE_COUNT)
    return particle_preparation.prepare_contract_e_lgssm_inputs(
        observations=observations,
        estimator_seeds=(seed,),
        num_particles=PARTICLE_COUNT,
        fixed_reset_mask=[[True] * horizon],
        prepared_ridge=[[RIDGE] * horizon],
        epsilon=0.5,
        scaling=0.9,
        sinkhorn_steps=controls.sinkhorn_steps,
        balance_steps=controls.balance_steps,
        row_chunk_size=chunks.row_chunk_size,
        col_chunk_size=chunks.col_chunk_size,
        dtype=tf.float32,
    )


def _finite_result(result) -> bool:
    return bool(
        tf.math.is_finite(result["objective"]).numpy()
        and tf.reduce_all(tf.math.is_finite(result["score"])).numpy()
        and tf.reduce_all(result["valid_chart"]).numpy()
        and result["teacher_valid"].numpy()
        and tf.reduce_all(result["shape_valid_history"]).numpy()
    )


def _run_seed(evaluate, baseline, theta, prepared, seed: int) -> dict[str, object]:
    started = time.perf_counter()
    result = evaluate(theta, prepared["prepared"])
    baseline_result = baseline(theta, prepared["prepared"])
    if hasattr(tf.experimental, "sync_devices"):
        tf.experimental.sync_devices()
    return {
        "seed": seed,
        "objective": float(result["objective"].numpy()),
        "score": [float(item) for item in result["score"].numpy()],
        "empirical_contract_e_objective": float(
            baseline_result["objective"].numpy()
        ),
        "empirical_contract_e_score": [
            float(item) for item in baseline_result["score"].numpy()
        ],
        "paired_objective_difference": float(
            (result["objective"] - baseline_result["objective"]).numpy()
        ),
        "paired_score_difference": [
            float(item)
            for item in (result["score"] - baseline_result["score"]).numpy()
        ],
        "finite_valid": _finite_result(result),
        "particle_valid": bool(tf.reduce_all(result["valid_chart"]).numpy()),
        "teacher_valid": bool(result["teacher_valid"].numpy()),
        "shape_valid": bool(tf.reduce_all(result["shape_valid_history"]).numpy()),
        "baseline_valid": bool(
            tf.reduce_all(baseline_result["valid_chart"]).numpy()
        ),
        "maximum_mean_residual": float(tf.reduce_max(result["mean_residual_history"]).numpy()),
        "maximum_covariance_residual": float(
            tf.reduce_max(result["covariance_residual_history"]).numpy()
        ),
        "maximum_skew_residual": float(
            tf.reduce_max(tf.abs(result["skew_residual_history"])).numpy()
        ),
        "wall_seconds": time.perf_counter() - started,
        "preparation_identity": prepared["identity"],
    }


def _summary(rows, kalman_value, kalman_score) -> dict[str, object]:
    errors = [
        [row["objective"] - kalman_value]
        + [row["score"][index] - kalman_score[index] for index in range(5)]
        for row in rows
    ]
    members = {}
    for index, label in enumerate(LABELS):
        values = [row[index] for row in errors]
        mean = statistics.mean(values)
        standard_deviation = statistics.stdev(values)
        mcse = standard_deviation / math.sqrt(len(values))
        members[label] = {
            "mean_error": mean,
            "standard_deviation": standard_deviation,
            "mcse": mcse,
            "mean_error_over_mcse": abs(mean) / mcse if mcse > 0.0 else float("inf"),
            "signs": ["positive" if value > 0 else "negative" if value < 0 else "zero" for value in values],
        }
    return {
        "seed_count": len(rows),
        "all_valid": all(row["finite_valid"] for row in rows),
        "mean_objective": statistics.mean(row["objective"] for row in rows),
        "mean_score": [statistics.mean(row["score"][index] for row in rows) for index in range(5)],
        "errors_to_kalman": members,
        "paired_difference_to_empirical_contract_e": {
            "value": {
                "mean": statistics.mean(
                    row["paired_objective_difference"] for row in rows
                ),
                "mcse": statistics.stdev(
                    row["paired_objective_difference"] for row in rows
                )
                / math.sqrt(len(rows)),
            },
            "score": {
                LABELS[index + 1]: {
                    "mean": statistics.mean(
                        row["paired_score_difference"][index] for row in rows
                    ),
                    "mcse": statistics.stdev(
                        row["paired_score_difference"][index] for row in rows
                    )
                    / math.sqrt(len(rows)),
                }
                for index in range(5)
            },
        },
        "maximum_mean_residual": max(row["maximum_mean_residual"] for row in rows),
        "maximum_covariance_residual": max(row["maximum_covariance_residual"] for row in rows),
        "maximum_skew_residual": max(row["maximum_skew_residual"] for row in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, choices=(2, 10, 50), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-index", type=int, choices=(0, 1))
    parser.add_argument("--tuning-only", action="store_true")
    parser.add_argument("--calibration-only", action="store_true")
    parser.add_argument("--claim-only", action="store_true")
    parser.add_argument("--claim-seeds", default="")
    parser.add_argument("--teacher-cache", type=Path)
    parser.add_argument("--prepare-teacher-only", action="store_true")
    arguments = parser.parse_args()
    output = arguments.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    campaign_started = time.perf_counter()
    horizon = arguments.horizon
    theta = tf.constant(THETA, tf.float32)
    observations = tf.cast(
        _lgssm_dataset(DATASET_SEED)["observations"][:horizon], tf.float32
    )
    kalman_theta = tf.cast(theta, tf.float64)
    kalman_observations = tf.cast(observations, tf.float64)
    with tf.GradientTape() as tape:
        tape.watch(kalman_theta)
        kalman_value_tensor = exact_kalman_value(kalman_theta, kalman_observations)
    kalman_score_tensor = tape.gradient(kalman_value_tensor, kalman_theta)
    kalman_value = float(kalman_value_tensor.numpy())
    kalman_score = [float(item) for item in kalman_score_tensor.numpy()]

    scope = make_lgssm_tuning_scope(
        horizon=horizon,
        prepared_data_id=f"lgssm_dataset_seed_81100_float32_prefix_t{horizon}",
        particle_count=PARTICLE_COUNT,
        dtype=tf.float32,
        tf32_enabled=False,
        jit_compile=True,
    )
    calibration_rows = []
    selected = None
    candidate_rows = list(enumerate(_controls(horizon)))
    if arguments.candidate_index is not None:
        candidate_rows = [candidate_rows[arguments.candidate_index]]
    for candidate_index, controls in candidate_rows:
        if arguments.teacher_cache is None:
            raw_teacher = prepare_lgssm_teacher_inputs(
                observations=observations,
                time_steps=horizon,
                fit_rows=96,
                basis_size=2,
                rank=1,
                sweeps=1,
                chart_scale=2.5,
                defensive_weight=0.05,
                root_seed=82000 + horizon,
                dtype=tf.float32,
                center_theta=theta,
            )
            teacher = freeze_teacher_scale_shift_indices(raw_teacher, controls)
        else:
            teacher = _read_teacher_cache(arguments.teacher_cache.resolve())
        if arguments.prepare_teacher_only:
            teacher_cache_manifest = _write_teacher_cache(
                output / "teacher_prepared", teacher
            )
            payload = {
                "schema": SCHEMA,
                "status": "teacher_prepared",
                "horizon": horizon,
                "candidate_index": candidate_index,
                "controls": controls.as_dict(),
                "teacher_cache_manifest": teacher_cache_manifest,
                "claim_seeds_evaluated": False,
                "wall_time_seconds": time.perf_counter() - campaign_started,
            }
            (output / "result.json").write_text(
                json.dumps(_json(payload), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(json.dumps({"status": "teacher_prepared", "output": str(output)}))
            return
        artifact = issue_moment_teacher_tuning_artifact(
            scope=scope,
            controls=controls,
            calibration_data_id=f"lgssm_t{horizon}_seed_{CALIBRATION_SEED}",
            validation_data_id=f"lgssm_t{horizon}_seed_{VALIDATION_SEED}",
            selection_record_id=f"candidate_{candidate_index}_hard_valid_then_residual",
        )
        evaluate = make_moment_teacher_lgssm_prepared_particles_tf(
            teacher, artifact, expected_scope=scope, jit_compile=True
        )
        baseline = canonical.make_canonical_prepared_value_and_score_tf(
            batch_size=1,
            time_steps=horizon,
            num_particles=PARTICLE_COUNT,
            steps=controls.sinkhorn_steps,
            balance_steps=controls.balance_steps,
            row_chunk_size=scope.row_chunk_size,
            col_chunk_size=scope.col_chunk_size,
            jit_compile=True,
            dtype=tf.float32,
        )
        if arguments.claim_only:
            selected = (
                (0.0, 0.0, candidate_index),
                controls,
                teacher,
                artifact,
                evaluate,
                baseline,
                _prepare_particle(
                    observations, horizon, VALIDATION_SEED, controls
                ),
            )
            break
        calibration = _prepare_particle(observations, horizon, CALIBRATION_SEED, controls)
        validation = _prepare_particle(observations, horizon, VALIDATION_SEED, controls)
        print(
            json.dumps(
                {
                    "phase": "calibration_start",
                    "horizon": horizon,
                    "candidate_index": candidate_index,
                }
            ),
            flush=True,
        )
        calibration_result = _run_seed(
            evaluate, baseline, theta, calibration, CALIBRATION_SEED
        )
        print(
            json.dumps(
                {
                    "phase": "calibration_complete",
                    "horizon": horizon,
                    "candidate_index": candidate_index,
                    "finite_valid": calibration_result["finite_valid"],
                }
            ),
            flush=True,
        )
        if arguments.calibration_only:
            payload = {
                "schema": SCHEMA,
                "status": "calibration_complete",
                "horizon": horizon,
                "candidate_index": candidate_index,
                "calibration": calibration_result,
                "claim_seeds_evaluated": False,
            }
            result_path = output / "result.json"
            result_path.write_text(
                json.dumps(_json(payload), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            return
        validation_result = _run_seed(
            evaluate, baseline, theta, validation, VALIDATION_SEED
        )
        row = {
            "candidate_index": candidate_index,
            "controls": controls.as_dict(),
            "calibration": calibration_result,
            "validation": validation_result,
        }
        calibration_rows.append(row)
        if calibration_result["finite_valid"] and validation_result["finite_valid"]:
            metric = (
                validation_result["maximum_skew_residual"],
                validation_result["maximum_covariance_residual"],
                candidate_index,
            )
            if selected is None or metric < selected[0]:
                selected = (
                    metric,
                    controls,
                    teacher,
                    artifact,
                    evaluate,
                    baseline,
                    validation,
                )
    if selected is None:
        failure = {
            "schema": SCHEMA,
            "status": "tuning_hard_veto",
            "horizon": horizon,
            "calibration_rows": calibration_rows,
        }
        (output / "result.json").write_text(
            json.dumps(_json(failure), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        raise SystemExit(2)

    _, controls, teacher, artifact, evaluate, baseline, identity_prepared = selected
    tuning_route_identity = issue_moment_teacher_lgssm_contract_e_route_identity(
        prepared_inputs=route_identity_prepared_inputs(
            identity_prepared["prepared"], teacher, artifact
        )
    )
    if arguments.tuning_only:
        teacher_cache_manifest = _write_teacher_cache(
            output / "teacher_prepared", teacher
        )
        payload = {
            "schema": SCHEMA,
            "status": "tuning_pass",
            "horizon": horizon,
            "particle_count": PARTICLE_COUNT,
            "theta": THETA,
            "calibration_seed": CALIBRATION_SEED,
            "validation_seed": VALIDATION_SEED,
            "calibration_rows": calibration_rows,
            "selected_tuning_artifact": artifact.to_dict(),
            "route_identity": tuning_route_identity.to_dict(),
            "teacher_cache_manifest": teacher_cache_manifest,
            "device": {
                "memory_policy": MEMORY_POLICY,
                "physical_gpus": [
                    item.name for item in tf.config.list_physical_devices("GPU")
                ],
                "logical_gpus": [
                    item.name for item in tf.config.list_logical_devices("GPU")
                ],
                "tf32_enabled": tf.config.experimental.tensor_float_32_execution_enabled(),
                "jit_compile": True,
                "trust_basis": "owner_designated_managed_session_visible_gpu_trusted",
            },
            "plan": PLAN,
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "command": " ".join(sys.argv),
            "wall_time_seconds": time.perf_counter() - campaign_started,
            "nonclaims": [
                "tuning only; claim seeds were not evaluated",
                "not LGSSM accuracy evidence",
                "not HMC readiness",
            ],
        }
        result_path = output / "result.json"
        result_path.write_text(
            json.dumps(_json(payload), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "status": payload["status"],
                    "horizon": horizon,
                    "candidate_index": arguments.candidate_index,
                    "result": str(result_path),
                }
            )
        )
        return
    active_claim_seeds = CLAIM_SEEDS
    if arguments.claim_seeds:
        active_claim_seeds = tuple(
            int(token) for token in arguments.claim_seeds.split(",") if token.strip()
        )
        if not active_claim_seeds or len(set(active_claim_seeds)) != len(active_claim_seeds):
            raise ValueError("claim-seeds must be a nonempty unique list")
        if not set(active_claim_seeds).issubset(set(CLAIM_SEEDS)):
            raise ValueError("claim-seeds must be a subset of the frozen claim seeds")
    claim_rows = []
    for seed in active_claim_seeds:
        seed_prepared = _prepare_particle(observations, horizon, seed, controls)
        row = _run_seed(evaluate, baseline, theta, seed_prepared, seed)
        row["route_identity"] = (
            issue_moment_teacher_lgssm_contract_e_route_identity(
                prepared_inputs=route_identity_prepared_inputs(
                    seed_prepared["prepared"], teacher, artifact
                )
            ).to_dict()
        )
        claim_rows.append(row)
    summary = (
        _summary(claim_rows, kalman_value, kalman_score)
        if len(claim_rows) >= 2
        else None
    )
    graph = evaluate.get_concrete_function().graph.as_graph_def()
    operations = {node.op for node in graph.node}
    for function in graph.library.function:
        operations.update(node.op for node in function.node_def)
    allocator = tf.config.experimental.get_memory_info("GPU:0")
    hard_vetoes = {
        "tuning_selection_missing": selected is None,
        "claim_invalid": not all(row["finite_valid"] for row in claim_rows),
        "mean_covariance_restoration": not (
            max(row["maximum_mean_residual"] for row in claim_rows) <= 2.0e-5
            and max(row["maximum_covariance_residual"] for row in claim_rows)
            <= 2.0e-4
        ),
        "pyfunc": bool({"PyFunc", "EagerPyFunc"} & operations),
        "missing_while": not bool({"While", "StatelessWhile"} & operations),
        "tf32_enabled": bool(tf.config.experimental.tensor_float_32_execution_enabled()),
    }
    payload = {
        "schema": SCHEMA,
        "status": "pass" if not any(hard_vetoes.values()) else "hard_veto",
        "hard_vetoes": hard_vetoes,
        "horizon": horizon,
        "particle_count": PARTICLE_COUNT,
        "theta": THETA,
        "observations": observations,
        "kalman": {"value": kalman_value, "score": kalman_score},
        "calibration_seed": CALIBRATION_SEED,
        "validation_seed": VALIDATION_SEED,
        "claim_seeds": active_claim_seeds,
        "calibration_rows": calibration_rows,
        "selected_tuning_artifact": artifact.to_dict(),
        "tuning_route_identity": tuning_route_identity.to_dict(),
        "claim_rows": claim_rows,
        "claim_summary": summary,
        "device": {
            "memory_policy": MEMORY_POLICY,
            "allocator": {key: int(value) for key, value in allocator.items()},
            "physical_gpus": [item.name for item in tf.config.list_physical_devices("GPU")],
            "logical_gpus": [item.name for item in tf.config.list_logical_devices("GPU")],
            "tf32_enabled": tf.config.experimental.tensor_float_32_execution_enabled(),
            "jit_compile": True,
            "trust_basis": "owner_designated_managed_session_visible_gpu_trusted",
        },
        "graph": {
            "has_while": bool({"While", "StatelessWhile"} & operations),
            "pyfunc_count": sum(1 for op in operations if op in {"PyFunc", "EagerPyFunc"}),
        },
        "plan": PLAN,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "command": " ".join(sys.argv),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "tensorflow": tf.__version__,
            "conda_env": os.environ.get("CONDA_DEFAULT_ENV", "unset"),
        },
        "wall_time_seconds": time.perf_counter() - campaign_started,
        "nonclaims": [
            "six claim seeds give descriptive MCSE, not a statistically supported method ranking",
            "not HMC readiness",
            "not nonlinear validity",
        ],
    }
    result_path = output / "result.json"
    result_path.write_text(
        json.dumps(_json(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema": "bayesfilter.run_manifest.v1",
        "git_commit": payload["git_commit"],
        "command": payload["command"],
        "environment": payload["environment"],
        "device": payload["device"],
        "data_version": scope.prepared_data_id,
        "seeds": (
            list(active_claim_seeds)
            if arguments.claim_only
            else [CALIBRATION_SEED, VALIDATION_SEED, *active_claim_seeds]
        ),
        "wall_time_seconds": payload["wall_time_seconds"],
        "output_artifact": str(output.relative_to(ROOT)),
        "plan": PLAN,
        "result": str(result_path.relative_to(ROOT)),
        "result_sha256": _sha256(result_path),
    }
    (output / "run_manifest.json").write_text(
        json.dumps(_json(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": payload["status"], "horizon": horizon, "result": str(result_path)}))
    if payload["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
