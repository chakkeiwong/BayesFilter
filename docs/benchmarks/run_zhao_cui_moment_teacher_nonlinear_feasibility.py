"""Tune and run one nonlinear Zhao-Cui moment-teacher feasibility claim."""

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

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tensorflow as tf

from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth


MEMORY_POLICY = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
tf.config.experimental.enable_tensor_float_32_execution(False)

from bayesfilter.highdim.cubature_genut_adapters import (
    predator_prey_candidate_adapter,
)
from bayesfilter.highdim.ledh_contract_e_identity import (
    issue_moment_teacher_austria_sir_contract_e_route_identity,
    issue_moment_teacher_predator_prey_contract_e_route_identity,
)
from bayesfilter.highdim.models import zhao_cui_sir_austria_model
from bayesfilter.highdim.transport_chunk_policy import select_transport_chunks
from bayesfilter.highdim.zhao_cui_moment_teacher_lgssm_tf import MomentTeacherControls
from bayesfilter.highdim.zhao_cui_moment_teacher_nonlinear_tf import (
    AUSTRIA_SIR_ROUTE_ID,
    EVENT_ORDER,
    PREDATOR_PREY_ROUTE_ID,
    freeze_nonlinear_teacher_scale_shift_indices,
    issue_nonlinear_moment_teacher_tuning_artifact,
    latent_preclip_austria_sir_candidate_adapter,
    make_nonlinear_moment_teacher_value_and_score_tf,
    make_nonlinear_tuning_scope,
    prepare_nonlinear_teacher_inputs,
    route_identity_prepared_inputs,
)
from bayesfilter.testing.predator_prey_sgqf_neutra_target_tf import (
    PP_SOURCE_OBSERVATION_SHA256,
    generate_source_order_predator_prey_dataset_tf,
)
from bayesfilter.testing.sir_filter_neutra_target_design_tf import (
    SIR_OBSERVATION_SHA256,
    generate_frozen_sir_dataset_tf,
)


SCHEMA = "bayesfilter.zhao_cui_moment_teacher_nonlinear_feasibility.v1"
PLAN = "docs/plans/bayesfilter-zhao-cui-moment-teacher-integration-campaign-plan-2026-07-30.md"
PARTICLE_COUNT = 1024
CALIBRATION_SEED = 82900
VALIDATION_SEED = 82901
CLAIM_SEED = 82910


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


def _tensor_hash(value: tf.Tensor) -> str:
    return hashlib.sha256(bytes(tf.io.serialize_tensor(value).numpy())).hexdigest()


def _source_hash(path: str) -> str:
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def _controls(candidate_index: int) -> MomentTeacherControls:
    return MomentTeacherControls(
        sinkhorn_steps=20,
        balance_steps=8,
        correction_steps=1,
        correction_strength=(0.005, 0.01)[candidate_index],
        correction_floor=1.0e-6,
        pairwise_correction_steps=0,
        pairwise_strength=0.0,
        pairwise_floor=1.0e-6,
        tt_ridge=(1.0e-3, 1.0e-4)[candidate_index],
        column_scale_floor=1.0e-6,
        condition_number_veto=1.0e8,
        fit_residual_veto=2.0,
    )


def _model_spec(model_name: str):
    if model_name == "predator_prey":
        _states, observations64 = generate_source_order_predator_prey_dataset_tf()
        observations = tf.cast(observations64, tf.float32)
        adapter = predator_prey_candidate_adapter()
        theta = tf.constant([0.6, 114.0, 25.0, 0.3, 0.5, 0.5], tf.float32)
        observed_center = tf.reduce_mean(observations, axis=0)
        observed_radius = tf.reduce_max(tf.abs(observations - observed_center), axis=0)
        base_scale = observed_radius + tf.constant([18.0, 8.0], tf.float32)
        charts = (
            (observed_center, base_scale),
            (observed_center, 1.35 * base_scale),
        )
        return {
            "adapter": adapter,
            "theta": theta,
            "observations": observations,
            "route_id": PREDATOR_PREY_ROUTE_ID,
            "model_id": "zhao_cui_predator_prey_T20",
            "target_id": "source_order_seed81104_y1_y20",
            "source_observation_sha256": PP_SOURCE_OBSERVATION_SHA256,
            "initial_standard_deviation": 1.0,
            "process_standard_deviation": 2.0,
            "initial_variance": 1.0,
            "process_variance": 4.0,
            "pairs": tf.constant([[0, 1], [1, 0]], tf.int32),
            "charts": charts,
            "identity_issuer": issue_moment_teacher_predator_prey_contract_e_route_identity,
        }
    source = zhao_cui_sir_austria_model()
    _states, source_observations, _all_observations = generate_frozen_sir_dataset_tf()
    observations = tf.cast(source_observations, tf.float32)
    adapter = latent_preclip_austria_sir_candidate_adapter()
    initial_mean = tf.cast(source.initial_mean, tf.float32)
    infectious_observed = tf.reduce_mean(observations, axis=0)
    infectious_radius = tf.reduce_max(
        tf.abs(observations - infectious_observed), axis=0
    ) + 30.0
    susceptible_offset = tf.fill([9], tf.constant(260.0, tf.float32))
    susceptible_scale = tf.fill([9], tf.constant(260.0, tf.float32))
    offset = tf.reshape(
        tf.stack([susceptible_offset, infectious_observed], axis=1), [18]
    )
    scale = tf.reshape(
        tf.stack([susceptible_scale, infectious_radius], axis=1), [18]
    )
    pair_rows = []
    for node in range(9):
        pair_rows.extend(((2 * node, 2 * node + 1), (2 * node + 1, 2 * node)))
    return {
        "adapter": adapter,
        "theta": tf.zeros([3], tf.float32),
        "observations": observations,
        "route_id": AUSTRIA_SIR_ROUTE_ID,
        "model_id": "zhao_cui_austria_sir_latent_preclip_T20",
        "target_id": "seed81120_source_observations_y1_y20",
        "source_observation_sha256": SIR_OBSERVATION_SHA256,
        "initial_standard_deviation": 1.0,
        "process_standard_deviation": 1.0,
        "initial_variance": 1.0,
        "process_variance": 1.0,
        "pairs": tf.constant(pair_rows, tf.int32),
        "charts": ((offset, scale), (offset, 1.25 * scale)),
        "identity_issuer": issue_moment_teacher_austria_sir_contract_e_route_identity,
        "initial_mean": initial_mean,
    }


def _particle_prepared(spec, seed: int) -> dict[str, tf.Tensor]:
    horizon = int(spec["observations"].shape[0])
    dimension = spec["adapter"].state_dimension
    raw_design = tf.random.stateless_normal(
        [horizon, PARTICLE_COUNT, dimension], [seed, 2001], dtype=tf.float64
    )
    centered_design = raw_design - tf.reduce_mean(raw_design, axis=1, keepdims=True)
    centered_design *= tf.sqrt(
        tf.cast(PARTICLE_COUNT, tf.float64)
        / tf.cast(PARTICLE_COUNT - 1, tf.float64)
    )
    return {
        "observations": spec["observations"],
        "initial_noise": tf.cast(
            tf.random.stateless_normal(
                [PARTICLE_COUNT, dimension], [seed, 101], dtype=tf.float64
            ),
            tf.float32,
        ),
        "process_noise": tf.cast(
            tf.random.stateless_normal(
                [horizon, PARTICLE_COUNT, dimension], [seed, 1001], dtype=tf.float64
            ),
            tf.float32,
        ),
        "residual_design": tf.cast(centered_design, tf.float32),
        "prepared_ridge": tf.fill([horizon], tf.constant(1.0e-5, tf.float32)),
        "epsilon": tf.constant(0.5, tf.float32),
        "scaling": tf.constant(0.9, tf.float32),
    }


def _finite(result) -> bool:
    return bool(
        tf.math.is_finite(result["objective"]).numpy()
        and tf.reduce_all(tf.math.is_finite(result["score"])).numpy()
        and result["valid_chart"].numpy()
        and result["teacher_valid"].numpy()
        and tf.reduce_all(result["shape_valid_history"]).numpy()
    )


def _evaluate(evaluate, baseline, theta, prepared) -> dict[str, object]:
    started = time.perf_counter()
    candidate = evaluate(theta)
    empirical = baseline(theta)
    if hasattr(tf.experimental, "sync_devices"):
        tf.experimental.sync_devices()
    return {
        "objective": float(candidate["objective"].numpy()),
        "score": [float(item) for item in candidate["score"].numpy()],
        "empirical_contract_e_objective": float(empirical["objective"].numpy()),
        "empirical_contract_e_score": [float(item) for item in empirical["score"].numpy()],
        "paired_objective_difference": float(
            (candidate["objective"] - empirical["objective"]).numpy()
        ),
        "paired_score_difference": [
            float(item) for item in (candidate["score"] - empirical["score"]).numpy()
        ],
        "finite_valid": _finite(candidate),
        "baseline_valid": _finite(empirical),
        "maximum_mean_residual": float(
            tf.reduce_max(candidate["mean_residual_history"]).numpy()
        ),
        "maximum_covariance_residual": float(
            tf.reduce_max(candidate["covariance_residual_history"]).numpy()
        ),
        "maximum_skew_residual": float(
            tf.reduce_max(candidate["skew_residual_history"]).numpy()
        ),
        "wall_seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("predator_prey", "austria_sir"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    output = arguments.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    started = time.perf_counter()
    spec = _model_spec(arguments.model)
    horizon = int(spec["observations"].shape[0])
    if _tensor_hash(tf.cast(spec["observations"], tf.float64)) != spec["source_observation_sha256"]:
        # The canonical source hash is over the source FP64 tensor.
        source_tensor = (
            generate_source_order_predator_prey_dataset_tf()[1]
            if arguments.model == "predator_prey"
            else generate_frozen_sir_dataset_tf()[1]
        )
        if _tensor_hash(source_tensor) != spec["source_observation_sha256"]:
            raise ValueError("source observation identity mismatch")
    scope = make_nonlinear_tuning_scope(
        model_id=spec["model_id"],
        target_id=spec["target_id"],
        route_id=spec["route_id"],
        horizon=horizon,
        prepared_data_id=f"{spec['target_id']}_fp32_{EVENT_ORDER}",
        particle_count=PARTICLE_COUNT,
        state_dimension=spec["adapter"].state_dimension,
        parameter_count=spec["adapter"].parameter_count,
        dtype=tf.float32,
        tf32_enabled=False,
        jit_compile=True,
    )
    calibration = _particle_prepared(spec, CALIBRATION_SEED)
    validation = _particle_prepared(spec, VALIDATION_SEED)
    candidate_rows = []
    selected = None
    for candidate_index in range(2):
        controls = _controls(candidate_index)
        offset, scale = spec["charts"][candidate_index]
        try:
            teacher = prepare_nonlinear_teacher_inputs(
                adapter=spec["adapter"],
                observations=spec["observations"],
                state_offset=offset,
                state_scale=scale,
                center_theta=spec["theta"],
                initial_standard_deviation=spec["initial_standard_deviation"],
                process_standard_deviation=spec["process_standard_deviation"],
                fit_rows=96,
                basis_size=2,
                rank=1,
                sweeps=1,
                defensive_weight=0.0,
                pair_indices=spec["pairs"],
                root_seed=84000 + candidate_index,
            )
            teacher = freeze_nonlinear_teacher_scale_shift_indices(
                teacher,
                controls,
                spec["adapter"],
                initial_variance=spec["initial_variance"],
                process_variance=spec["process_variance"],
            )
            artifact = issue_nonlinear_moment_teacher_tuning_artifact(
                scope=scope,
                controls=controls,
                calibration_data_id=f"particle_seed_{CALIBRATION_SEED}",
                validation_data_id=f"particle_seed_{VALIDATION_SEED}",
                selection_record_id=f"candidate_{candidate_index}_valid_then_skew_residual",
                chart_id=f"candidate_{candidate_index}_{_tensor_hash(offset)}_{_tensor_hash(scale)}",
                pair_set_id=_tensor_hash(spec["pairs"]),
            )
            evaluate = make_nonlinear_moment_teacher_value_and_score_tf(
                adapter=spec["adapter"],
                particle_prepared=calibration,
                teacher_prepared=teacher,
                tuning_artifact=artifact,
                expected_scope=scope,
                initial_variance=spec["initial_variance"],
                process_variance=spec["process_variance"],
                jit_compile=True,
            )
            baseline = make_nonlinear_moment_teacher_value_and_score_tf(
                adapter=spec["adapter"],
                particle_prepared=calibration,
                teacher_prepared=None,
                tuning_artifact=artifact,
                expected_scope=scope,
                initial_variance=spec["initial_variance"],
                process_variance=spec["process_variance"],
                jit_compile=True,
            )
            calibration_row = _evaluate(evaluate, baseline, spec["theta"], calibration)
            validation_eval = make_nonlinear_moment_teacher_value_and_score_tf(
                adapter=spec["adapter"],
                particle_prepared=validation,
                teacher_prepared=teacher,
                tuning_artifact=artifact,
                expected_scope=scope,
                initial_variance=spec["initial_variance"],
                process_variance=spec["process_variance"],
                jit_compile=True,
            )
            validation_base = make_nonlinear_moment_teacher_value_and_score_tf(
                adapter=spec["adapter"],
                particle_prepared=validation,
                teacher_prepared=None,
                tuning_artifact=artifact,
                expected_scope=scope,
                initial_variance=spec["initial_variance"],
                process_variance=spec["process_variance"],
                jit_compile=True,
            )
            validation_row = _evaluate(
                validation_eval, validation_base, spec["theta"], validation
            )
            row = {
                "candidate_index": candidate_index,
                "artifact": artifact.to_dict(),
                "chart_offset": offset,
                "chart_scale": scale,
                "calibration": calibration_row,
                "validation": validation_row,
            }
            candidate_rows.append(row)
            valid = calibration_row["finite_valid"] and validation_row["finite_valid"]
            valid = valid and calibration_row["baseline_valid"] and validation_row["baseline_valid"]
            metric = (validation_row["maximum_skew_residual"], candidate_index)
            if valid and (selected is None or metric < selected[0]):
                selected = (metric, teacher, artifact, row)
        except Exception as error:
            candidate_rows.append(
                {
                    "candidate_index": candidate_index,
                    "status": "infrastructure_or_tuning_hard_veto",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
    if selected is None:
        payload = {
            "schema": SCHEMA,
            "status": "tuning_hard_veto",
            "model": arguments.model,
            "event_order": EVENT_ORDER,
            "particle_count": PARTICLE_COUNT,
            "horizon": horizon,
            "theta": spec["theta"],
            "source_observation_sha256": spec["source_observation_sha256"],
            "scope": scope.as_dict(),
            "calibration_seed": CALIBRATION_SEED,
            "validation_seed": VALIDATION_SEED,
            "claim_seed": CLAIM_SEED,
            "claim_seed_evaluated": False,
            "candidate_rows": candidate_rows,
            "device": {
                "memory_policy": MEMORY_POLICY,
                "physical_gpus": [
                    item.name for item in tf.config.list_physical_devices("GPU")
                ],
                "logical_gpus": [
                    item.name for item in tf.config.list_logical_devices("GPU")
                ],
                "tf32_enabled": (
                    tf.config.experimental.tensor_float_32_execution_enabled()
                ),
                "jit_compile": True,
                "trust_basis": (
                    "owner_designated_managed_session_visible_gpu_trusted"
                ),
            },
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "source_sha256": {
                path: _source_hash(path)
                for path in (
                    "bayesfilter/highdim/zhao_cui_moment_teacher_nonlinear_tf.py",
                    "bayesfilter/highdim/zhao_cui_moment_teacher_xla.py",
                    "bayesfilter/highdim/higher_moment_contract_e.py",
                    "bayesfilter/highdim/ledh_contract_e_streaming_tf.py",
                    "bayesfilter/highdim/ledh_contract_e_reset_tf.py",
                    "bayesfilter/highdim/cubature_genut_adapters.py",
                )
            },
            "command": " ".join(sys.argv),
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "tensorflow": tf.__version__,
                "conda_env": os.environ.get("CONDA_DEFAULT_ENV", "unset"),
            },
            "plan": PLAN,
            "wall_time_seconds": time.perf_counter() - started,
            "nonclaims": [
                "tuning failed before the claim seed was evaluated",
                "not evidence against the particle or Contract E implementation",
                "not HMC or posterior readiness",
                "not source-faithful Zhao-Cui filtering",
            ],
        }
        (output / "result.json").write_text(
            json.dumps(_json(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        raise SystemExit(2)
    _, teacher, artifact, selected_row = selected
    claim_prepared = _particle_prepared(spec, CLAIM_SEED)
    claim_eval = make_nonlinear_moment_teacher_value_and_score_tf(
        adapter=spec["adapter"],
        particle_prepared=claim_prepared,
        teacher_prepared=teacher,
        tuning_artifact=artifact,
        expected_scope=scope,
        initial_variance=spec["initial_variance"],
        process_variance=spec["process_variance"],
        jit_compile=True,
    )
    claim_base = make_nonlinear_moment_teacher_value_and_score_tf(
        adapter=spec["adapter"],
        particle_prepared=claim_prepared,
        teacher_prepared=None,
        tuning_artifact=artifact,
        expected_scope=scope,
        initial_variance=spec["initial_variance"],
        process_variance=spec["process_variance"],
        jit_compile=True,
    )
    claim = _evaluate(claim_eval, claim_base, spec["theta"], claim_prepared)
    identity = spec["identity_issuer"](
        prepared_inputs=route_identity_prepared_inputs(
            claim_prepared, teacher, artifact
        )
    ).to_dict()
    graph = claim_eval.get_concrete_function().graph.as_graph_def()
    operations = {node.op for node in graph.node}
    for function in graph.library.function:
        operations.update(node.op for node in function.node_def)
    allocator = tf.config.experimental.get_memory_info("GPU:0")
    hard_vetoes = {
        "claim_invalid": not claim["finite_valid"],
        "baseline_invalid": not claim["baseline_valid"],
        "mean_covariance_restoration": not (
            claim["maximum_mean_residual"] <= 2.0e-5
            and claim["maximum_covariance_residual"] <= 2.0e-4
        ),
        "tf32_enabled": bool(tf.config.experimental.tensor_float_32_execution_enabled()),
        "pyfunc": bool({"PyFunc", "EagerPyFunc"} & operations),
        "missing_while": not bool({"While", "StatelessWhile"} & operations),
    }
    payload = {
        "schema": SCHEMA,
        "status": "pass" if not any(hard_vetoes.values()) else "hard_veto",
        "model": arguments.model,
        "event_order": EVENT_ORDER,
        "particle_count": PARTICLE_COUNT,
        "horizon": horizon,
        "theta": spec["theta"],
        "source_observation_sha256": spec["source_observation_sha256"],
        "scope": scope.as_dict(),
        "candidate_rows": candidate_rows,
        "selected_candidate_index": selected_row["candidate_index"],
        "selected_tuning_artifact": artifact.to_dict(),
        "claim_seed": CLAIM_SEED,
        "claim": claim,
        "route_identity": identity,
        "hard_vetoes": hard_vetoes,
        "graph": {
            "has_while": bool({"While", "StatelessWhile"} & operations),
            "pyfunc_count": sum(op in {"PyFunc", "EagerPyFunc"} for op in operations),
        },
        "device": {
            "memory_policy": MEMORY_POLICY,
            "physical_gpus": [item.name for item in tf.config.list_physical_devices("GPU")],
            "logical_gpus": [item.name for item in tf.config.list_logical_devices("GPU")],
            "allocator": {name: int(value) for name, value in allocator.items()},
            "tf32_enabled": tf.config.experimental.tensor_float_32_execution_enabled(),
            "jit_compile": True,
            "trust_basis": "owner_designated_managed_session_visible_gpu_trusted",
        },
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "source_sha256": {
            path: _source_hash(path)
            for path in (
                "bayesfilter/highdim/zhao_cui_moment_teacher_nonlinear_tf.py",
                "bayesfilter/highdim/zhao_cui_moment_teacher_xla.py",
                "bayesfilter/highdim/higher_moment_contract_e.py",
                "bayesfilter/highdim/ledh_contract_e_streaming_tf.py",
                "bayesfilter/highdim/ledh_contract_e_reset_tf.py",
                "bayesfilter/highdim/cubature_genut_adapters.py",
            )
        },
        "command": " ".join(sys.argv),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "tensorflow": tf.__version__,
            "conda_env": os.environ.get("CONDA_DEFAULT_ENV", "unset"),
        },
        "plan": PLAN,
        "wall_time_seconds": time.perf_counter() - started,
        "nonclaims": [
            "one seed is descriptive feasibility evidence only",
            "no statistically supported method ranking",
            "not HMC or posterior readiness",
            "not source-faithful Zhao-Cui filtering",
        ],
    }
    result_path = output / "result.json"
    result_path.write_text(
        json.dumps(_json(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": payload["status"], "result": str(result_path)}))
    if payload["status"] != "pass":
        raise SystemExit(3)


if __name__ == "__main__":
    main()
