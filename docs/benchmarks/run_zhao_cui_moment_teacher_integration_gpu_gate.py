"""Trusted GPU/XLA gate for the integrated Zhao-Cui moment-teacher route."""

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

from bayesfilter.highdim import ledh_contract_e_lgssm_preparation_tf as particle_preparation
from bayesfilter.highdim.ledh_contract_e_identity import (
    issue_moment_teacher_lgssm_contract_e_route_identity,
)
from bayesfilter.highdim.transport_chunk_policy import select_transport_chunks
from bayesfilter.highdim.zhao_cui_moment_teacher_lgssm_tf import (
    MomentTeacherControls,
    issue_moment_teacher_tuning_artifact,
    make_lgssm_tuning_scope,
    make_moment_teacher_lgssm_value_and_score_tf,
    prepare_lgssm_teacher_inputs,
    route_identity_prepared_inputs,
)


SCHEMA = "bayesfilter.zhao_cui_moment_teacher_integration_gpu_gate.v1"
TRUST_BASIS = "owner_designated_managed_session_visible_gpu_trusted"
PLAN = "docs/plans/bayesfilter-zhao-cui-moment-teacher-integration-campaign-plan-2026-07-30.md"
SOURCE_PATHS = (
    "bayesfilter/highdim/zhao_cui_moment_teacher_lgssm_tf.py",
    "bayesfilter/highdim/zhao_cui_moment_teacher_xla.py",
    "bayesfilter/highdim/higher_moment_contract_e.py",
    "bayesfilter/highdim/ledh_contract_e_canonical_lgssm_tf.py",
    "bayesfilter/highdim/ledh_contract_e_streaming_tf.py",
    "bayesfilter/highdim/ledh_contract_e_reset_tf.py",
    "bayesfilter/highdim/ledh_contract_e_identity.py",
)


def _sha256(path: str) -> str:
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


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


def _graph_record(function) -> dict[str, object]:
    graph = function.get_concrete_function().graph.as_graph_def()
    counts: dict[str, int] = {}
    for node in graph.node:
        counts[node.op] = counts.get(node.op, 0) + 1
    for library_function in graph.library.function:
        for node in library_function.node_def:
            counts[node.op] = counts.get(node.op, 0) + 1
    return {
        "operation_counts": counts,
        "has_while": bool({"While", "StatelessWhile"} & set(counts)),
        "pyfunc_count": counts.get("PyFunc", 0) + counts.get("EagerPyFunc", 0),
    }


def _observations(dtype: tf.dtypes.DType) -> tf.Tensor:
    return tf.constant(
        [[0.15, -0.1, 0.05], [0.2, 0.0, -0.12]], dtype
    )


def _controls() -> MomentTeacherControls:
    return MomentTeacherControls(
        sinkhorn_steps=2,
        balance_steps=100,
        correction_steps=1,
        correction_strength=0.025,
        correction_floor=1.0e-6,
        pairwise_correction_steps=0,
        pairwise_strength=0.0,
        pairwise_floor=1.0e-6,
        tt_ridge=1.0e-5,
        column_scale_floor=1.0e-6,
        condition_number_veto=1.0e10,
        fit_residual_veto=2.0,
    )


def _build(dtype: tf.dtypes.DType):
    time_steps = 2
    particle_count = 32
    chunks = select_transport_chunks(particle_count)
    observations = _observations(dtype)
    particle_result = particle_preparation.prepare_contract_e_lgssm_inputs(
        observations=observations,
        estimator_seeds=(91731,),
        num_particles=particle_count,
        fixed_reset_mask=tf.ones([1, time_steps], tf.bool),
        prepared_ridge=[[1.0e-6] * time_steps],
        epsilon=0.5,
        scaling=0.9,
        sinkhorn_steps=2,
        balance_steps=100,
        row_chunk_size=chunks.row_chunk_size,
        col_chunk_size=chunks.col_chunk_size,
        dtype=dtype,
    )
    theta = tf.constant([0.55, 0.45, 0.35, 0.8, 0.6], dtype)
    teacher = prepare_lgssm_teacher_inputs(
        observations=observations,
        time_steps=time_steps,
        fit_rows=64,
        basis_size=2,
        rank=1,
        sweeps=1,
        chart_scale=2.5,
        defensive_weight=0.05,
        root_seed=91751,
        dtype=dtype,
        center_theta=theta,
    )
    scope = make_lgssm_tuning_scope(
        horizon=time_steps,
        prepared_data_id="lgssm_t2_gpu_integration_mechanics_v1",
        particle_count=particle_count,
        dtype=dtype,
        tf32_enabled=False,
        jit_compile=True,
    )
    artifact = issue_moment_teacher_tuning_artifact(
        scope=scope,
        controls=_controls(),
        calibration_data_id="lgssm_t2_gpu_mechanics_calibration_v1",
        validation_data_id="lgssm_t2_gpu_mechanics_validation_v1",
        selection_record_id="mechanics_fixture_not_claim_tuning_v1",
    )
    evaluate = make_moment_teacher_lgssm_value_and_score_tf(
        particle_result["prepared"],
        teacher,
        artifact,
        expected_scope=scope,
        jit_compile=True,
    )
    return theta, particle_result, teacher, scope, artifact, evaluate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    output = arguments.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    started_campaign = time.perf_counter()

    theta32, prepared32, teacher32, scope32, artifact32, evaluate32 = _build(tf.float32)
    identity = issue_moment_teacher_lgssm_contract_e_route_identity(
        prepared_inputs=route_identity_prepared_inputs(
            prepared32["prepared"], teacher32, artifact32
        )
    )
    graph = _graph_record(evaluate32)
    tf.config.experimental.reset_memory_stats("GPU:0")
    warm_started = time.perf_counter()
    warm = evaluate32(theta32)
    if hasattr(tf.experimental, "sync_devices"):
        tf.experimental.sync_devices()
    compile_and_warm_seconds = time.perf_counter() - warm_started
    run_started = time.perf_counter()
    result32 = evaluate32(theta32)
    if hasattr(tf.experimental, "sync_devices"):
        tf.experimental.sync_devices()
    warm_execution_seconds = time.perf_counter() - run_started
    allocator = tf.config.experimental.get_memory_info("GPU:0")

    theta64, _, _, _, _, evaluate64 = _build(tf.float64)
    reference64 = evaluate64(theta64)
    objective_error = float(
        tf.abs(tf.cast(result32["objective"], tf.float64) - reference64["objective"]).numpy()
    )
    score_difference = tf.abs(
        tf.cast(result32["score"], tf.float64) - reference64["score"]
    )
    score_relative = score_difference / tf.maximum(
        tf.abs(reference64["score"]), tf.constant(1.0e-8, tf.float64)
    )
    maximum_absolute_error = max(
        objective_error, float(tf.reduce_max(score_difference).numpy())
    )
    maximum_relative_error = max(
        objective_error
        / max(abs(float(reference64["objective"].numpy())), 1.0e-8),
        float(tf.reduce_max(score_relative).numpy()),
    )
    hard_vetoes = {
        "gpu_missing": not bool(tf.config.list_logical_devices("GPU")),
        "tf32_enabled": bool(tf.config.experimental.tensor_float_32_execution_enabled()),
        "memory_growth_unverified": not bool(
            MEMORY_POLICY.get("all_physical_devices_memory_growth", False)
            and MEMORY_POLICY.get("configured_before_logical_device_initialization", False)
        ),
        "pyfunc": graph["pyfunc_count"] != 0,
        "missing_while": not graph["has_while"],
        "invalid_fp32": not bool(tf.reduce_all(result32["valid_chart"]).numpy()),
        "invalid_fp64": not bool(tf.reduce_all(reference64["valid_chart"]).numpy()),
        "nonfinite_fp32": not bool(
            tf.math.is_finite(result32["objective"]).numpy()
            and tf.reduce_all(tf.math.is_finite(result32["score"])).numpy()
        ),
        "fp32_fp64_parity": not (
            maximum_absolute_error <= 2.0e-3
            and maximum_relative_error <= 5.0e-3
        ),
    }
    payload = {
        "schema": SCHEMA,
        "status": "pass" if not any(hard_vetoes.values()) else "hard_veto",
        "hard_vetoes": hard_vetoes,
        "plan": PLAN,
        "route_id": scope32.route_id,
        "classification": "extension_or_invention",
        "nonclaims": [
            "not claim tuning",
            "not LGSSM accuracy evidence",
            "not nonlinear evidence",
            "not HMC readiness",
        ],
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "command": " ".join(sys.argv),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "tensorflow": tf.__version__,
            "conda_env": os.environ.get("CONDA_DEFAULT_ENV", "unset"),
            "dtype": "float32",
            "tf32_enabled": tf.config.experimental.tensor_float_32_execution_enabled(),
            "jit_compile": True,
        },
        "device": {
            "physical_gpus": [item.name for item in tf.config.list_physical_devices("GPU")],
            "logical_gpus": [item.name for item in tf.config.list_logical_devices("GPU")],
            "memory_policy": MEMORY_POLICY,
            "allocator": {key: int(value) for key, value in allocator.items()},
            "trust_basis": TRUST_BASIS,
        },
        "scope": scope32.as_dict(),
        "scope_sha256": scope32.scope_sha256,
        "tuning_artifact": artifact32.to_dict(),
        "route_identity": identity.to_dict(),
        "particle_preparation_identity": prepared32["identity"],
        "graph": graph,
        "timing_seconds": {
            "compile_and_warm": compile_and_warm_seconds,
            "warm_execution": warm_execution_seconds,
            "campaign": time.perf_counter() - started_campaign,
        },
        "fp32": {
            "objective": result32["objective"],
            "score": result32["score"],
            "teacher_normalizers": result32["teacher_normalizers"],
            "mean_residual_history": result32["mean_residual_history"],
            "covariance_residual_history": result32["covariance_residual_history"],
        },
        "fp64_reference": {
            "objective": reference64["objective"],
            "score": reference64["score"],
        },
        "parity": {
            "objective_absolute_error": objective_error,
            "score_absolute_error": score_difference,
            "score_relative_error": score_relative,
            "maximum_absolute_error": maximum_absolute_error,
            "maximum_relative_error": maximum_relative_error,
            "absolute_veto": 2.0e-3,
            "relative_veto": 5.0e-3,
        },
        "source_sha256": {path: _sha256(path) for path in SOURCE_PATHS},
    }
    result_path = output / "result.json"
    result_path.write_text(
        json.dumps(_json(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": payload["status"], "result": str(result_path)}))
    if payload["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
