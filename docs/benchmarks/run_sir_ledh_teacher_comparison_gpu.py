"""Run paired LEDH/online-teacher SIR disagreement screens on trusted GPU/XLA."""

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
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from scipy.stats import t as student_t
import tensorflow as tf


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bayesfilter.runtime.gpu_memory_policy import (  # noqa: E402
    TensorFlowGPUMemoryPolicyError,
    configure_tensorflow_gpu_memory_limit,
)


GPU_MEMORY_LIMIT_ENV = "BAYESFILTER_TF_GPU_MEMORY_LIMIT_MIB"
REQUIRED_GPU_MEMORY_LIMIT_MIB = 8192


def _configure_import_time_gpu_memory_policy() -> Mapping[str, Any] | None:
    raw_limit = os.environ.get(GPU_MEMORY_LIMIT_ENV)
    if raw_limit is None:
        return None
    try:
        memory_limit_mib = int(raw_limit)
    except ValueError as exc:
        raise TensorFlowGPUMemoryPolicyError(
            f"{GPU_MEMORY_LIMIT_ENV} must be an integer MiB value"
        ) from exc
    if memory_limit_mib != REQUIRED_GPU_MEMORY_LIMIT_MIB:
        raise TensorFlowGPUMemoryPolicyError(
            f"this campaign requires {GPU_MEMORY_LIMIT_ENV}="
            f"{REQUIRED_GPU_MEMORY_LIMIT_MIB}"
        )
    return configure_tensorflow_gpu_memory_limit(
        tf,
        memory_limit_mib=memory_limit_mib,
        require_gpu=True,
    )


GPU_MEMORY_POLICY = _configure_import_time_gpu_memory_policy()

from bayesfilter.highdim import ledh_contract_e_latent_sir_tf as ledh  # noqa: E402
from bayesfilter.highdim.ledh_contract_e_identity import (  # noqa: E402
    issue_latent_sir_contract_e_route_identity,
    issue_latent_sir_two_node_contract_e_route_identity,
)
from bayesfilter.highdim.sir_latent_preclip_tf import (  # noqa: E402
    latent_preclip_two_node_spatial_sir_model,
    latent_preclip_zhao_cui_sir_austria_model,
)
from bayesfilter.highdim.sir_online_score_teacher_tf import (  # noqa: E402
    make_online_sir_teacher,
)


DTYPE = tf.float64
THETA = tf.constant([0.0, 0.0, 0.0], DTYPE)
REPLICATES = 16
TEACHER_PARTICLES = (128, 256)
LEDH_PARTICLES = 256
PARAMETER_COUNT = 3
FAMILY_SIZE = 4
MODEL_ROWS = (
    ("two_node", (1, 2, 5), 87100),
    ("austria_d18", (2, 5, 20), 87200),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _write_artifact_hashes(output_root: Path) -> None:
    _write_json(
        output_root / "artifact_hashes.json",
        {
            "schema": "bayesfilter.sir_ledh_teacher_comparison_hashes.v1",
            "artifacts": {
                str(path.relative_to(output_root)): _sha256(path)
                for path in sorted(output_root.iterdir())
                if path.is_file() and path.name != "artifact_hashes.json"
            },
        },
    )


def _gpu_memory_info() -> Mapping[str, Any]:
    try:
        info = tf.config.experimental.get_memory_info("GPU:0")
    except Exception as exc:
        return {
            "available": False,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }
    return {
        "available": True,
        "current_bytes": int(info["current"]),
        "peak_bytes": int(info["peak"]),
        "current_mib": float(info["current"]) / (1024.0**2),
        "peak_mib": float(info["peak"]) / (1024.0**2),
    }


def _failure_classification(exc: Exception) -> str:
    if isinstance(exc, tf.errors.ResourceExhaustedError):
        return "TENSORFLOW_RESOURCE_EXHAUSTED"
    upper_message = str(exc).upper()
    if any(token in upper_message for token in ("OUT OF MEMORY", "RESOURCE_EXHAUSTED", "OOM")):
        return "POSSIBLE_TENSORFLOW_OR_XLA_OOM"
    return "UNHANDLED_CAMPAIGN_EXCEPTION"


def _failure_record(
    exc: Exception,
    *,
    active_stage: str,
    completed_rows: Sequence[str],
    started_at: datetime,
    started: float,
) -> Mapping[str, Any]:
    classification = _failure_classification(exc)
    return {
        "schema": "bayesfilter.sir_ledh_teacher_comparison_failure.v1",
        "status": "CAMPAIGN_FAILED",
        "failure_classification": classification,
        "continuation_veto": True,
        "active_stage": active_stage,
        "completed_rows": list(completed_rows),
        "exception": {
            "type": f"{type(exc).__module__}.{type(exc).__name__}",
            "message": str(exc),
            "traceback": traceback.format_exc(),
        },
        "gpu_memory_policy": GPU_MEMORY_POLICY,
        "gpu_memory_info_at_failure": _gpu_memory_info(),
        "started_at": started_at.isoformat(),
        "failed_at": datetime.now(timezone.utc).isoformat(),
        "wall_time_seconds": time.perf_counter() - started,
        "interpretation": (
            "engineering continuation veto; no scientific inference about LEDH "
            "or the teacher is available from the incomplete stage"
        ),
        "nonclaims": [
            "not evidence that LEDH is inaccurate",
            "not evidence that the teacher is inaccurate",
            "not a completed paired comparison",
            "not HMC or leaderboard readiness evidence",
        ],
    }


def _tensor_from_record(record: Mapping[str, Any]) -> tf.Tensor:
    value = tf.reshape(
        tf.convert_to_tensor(
            record["values"], tf.dtypes.as_dtype(record["dtype"])
        ),
        record["shape"],
    )
    serialized = tf.io.serialize_tensor(value).numpy()
    if hashlib.sha256(serialized).hexdigest() != record["serialized_tensor_sha256"]:
        raise ValueError("Phase 3 prepared tensor hash mismatch")
    return value


def _phase3_replay(prepared_path: Path, cpu_result_path: Path) -> Mapping[str, Any]:
    prepared_payload = json.loads(prepared_path.read_text(encoding="utf-8"))
    cpu_payload = json.loads(cpu_result_path.read_text(encoding="utf-8"))
    prepared = {
        name: _tensor_from_record(record)
        for name, record in prepared_payload["prepared"].items()
    }
    result = _ledh_call("austria_d18", prepared)
    cpu_diagnostic = cpu_payload["diagnostic"]
    value_delta = abs(
        float(result["objective"].numpy()) - float(cpu_diagnostic["objective"])
    )
    score_delta = tf.abs(
        result["score"] - tf.constant(cpu_diagnostic["score"], DTYPE)
    )
    maximum_score_delta = float(tf.reduce_max(score_delta).numpy())
    prepared_hashes = {
        name: hashlib.sha256(tf.io.serialize_tensor(value).numpy()).hexdigest()
        for name, value in prepared.items()
    }
    if prepared_hashes != cpu_payload["prepared_identity"]:
        raise ValueError("Phase 3 CPU/GPU prepared identities differ")
    if value_delta > 1.0e-10 or maximum_score_delta > 1.0e-10:
        raise ValueError("Phase 3 final-source CPU/GPU numerical mismatch")
    if not bool(tf.reduce_all(result["valid_chart"]).numpy()):
        raise ValueError("Phase 3 final-source GPU chart is invalid")
    return {
        "status": "PASS_FINAL_SOURCE_CPU_GPU_REPLAY",
        "prepared_input": {"path": str(prepared_path), "sha256": _sha256(prepared_path)},
        "cpu_result": {"path": str(cpu_result_path), "sha256": _sha256(cpu_result_path)},
        "value_absolute_delta": value_delta,
        "score_absolute_deltas": score_delta.numpy().tolist(),
        "maximum_score_absolute_delta": maximum_score_delta,
        "prepared_identity": prepared_hashes,
        "valid_chart": result["valid_chart"].numpy().tolist(),
        "reset_valid_history": result["reset_valid_history"].numpy().tolist(),
        "output_devices": sorted(
            {tensor.device for tensor in tf.nest.flatten(result)}
        ),
    }


def _student_summary(samples: Sequence[float]) -> Mapping[str, Any]:
    values = tuple(float(value) for value in samples)
    if len(values) < 2 or not all(math.isfinite(value) for value in values):
        raise ValueError("paired samples must contain at least two finite values")
    mean = statistics.fmean(values)
    standard_deviation = statistics.stdev(values)
    standard_error = standard_deviation / math.sqrt(len(values))
    critical = float(
        student_t.ppf(1.0 - 0.05 / (2.0 * FAMILY_SIZE), len(values) - 1)
    )
    half_width = critical * standard_error
    return {
        "count": len(values),
        "mean": mean,
        "standard_deviation": standard_deviation,
        "standard_error": standard_error,
        "student_critical": critical,
        "bonferroni_family_size": FAMILY_SIZE,
        "half_width": half_width,
        "lower": mean - half_width,
        "upper": mean + half_width,
        "contains_zero": bool(mean - half_width <= 0.0 <= mean + half_width),
    }


def _comparison_summaries(
    left_value: tf.Tensor,
    left_score: tf.Tensor,
    right_value: tf.Tensor,
    right_score: tf.Tensor,
) -> Mapping[str, Any]:
    value_difference = left_value - right_value
    score_difference = left_score - right_score
    value_samples = value_difference.numpy().tolist()
    score_samples = score_difference.numpy().tolist()
    value_finite = [math.isfinite(float(value)) for value in value_samples]
    score_finite = [
        [math.isfinite(float(value)) for value in row] for row in score_samples
    ]
    available = all(value_finite) and all(all(row) for row in score_finite)
    result = {
        "available": available,
        "unavailable_reason": None if available else "NONFINITE_PAIRED_SAMPLES",
        "value_samples": value_samples,
        "score_samples": score_samples,
        "value_finite": value_finite,
        "score_finite": score_finite,
        "invalid_value_pair_count": value_finite.count(False),
        "invalid_score_pair_count": sum(
            not finite for row in score_finite for finite in row
        ),
        "value": None,
        "score": [None] * PARAMETER_COUNT,
    }
    if available:
        result["value"] = _student_summary(value_samples)
        result["score"] = [
            _student_summary(
                [score_samples[row][index] for row in range(len(score_samples))]
            )
            for index in range(PARAMETER_COUNT)
        ]
    return result


def _nan_aware_exact(left: tf.Tensor, right: tf.Tensor) -> bool:
    equal_or_both_nan = tf.equal(left, right) | (
        tf.math.is_nan(left) & tf.math.is_nan(right)
    )
    return bool(tf.reduce_all(equal_or_both_nan).numpy())


def _model(name: str):
    if name == "two_node":
        return latent_preclip_two_node_spatial_sir_model()
    if name == "austria_d18":
        return latent_preclip_zhao_cui_sir_austria_model()
    raise ValueError(f"unknown model row {name}")


def _observations(model, horizon: int, seed: int) -> tf.Tensor:
    state_dimension = model.state_dim()
    observation_dimension = model.observation_dim()
    simulation = model.simulate_from_standard_normals(
        THETA,
        tf.random.stateless_normal(
            [state_dimension], [seed, 10], dtype=DTYPE
        ),
        tf.random.stateless_normal(
            [max(0, horizon - 1), state_dimension], [seed, 11], dtype=DTYPE
        ),
        tf.random.stateless_normal(
            [horizon, observation_dimension], [seed, 12], dtype=DTYPE
        ),
    )
    return simulation["observations"]


def _replicated_normals(
    seeds: tf.Tensor, shape: tuple[int, int], domain: int
) -> tf.Tensor:
    return tf.map_fn(
        lambda seed: tf.random.stateless_normal(
            shape, seed=tf.stack([seed, tf.constant(domain, tf.int32)]), dtype=DTYPE
        ),
        seeds,
        fn_output_signature=tf.TensorSpec(shape, DTYPE),
    )


def _prepared(model, observations: tf.Tensor, seeds: tf.Tensor) -> Mapping[str, tf.Tensor]:
    horizon = int(observations.shape[0])
    state_dimension = model.state_dim()
    initial_noise = _replicated_normals(
        seeds, (LEDH_PARTICLES, state_dimension), 100
    )
    transition_ta = tf.TensorArray(
        DTYPE,
        size=max(0, horizon - 1),
        element_shape=[REPLICATES, LEDH_PARTICLES, state_dimension],
    )
    residual_ta = tf.TensorArray(
        DTYPE,
        size=horizon,
        element_shape=[REPLICATES, LEDH_PARTICLES, state_dimension],
    )
    for time_index in range(horizon):
        residual = _replicated_normals(
            seeds, (LEDH_PARTICLES, state_dimension), 3000 + time_index
        )
        residual -= tf.reduce_mean(residual, axis=1, keepdims=True)
        residual_ta = residual_ta.write(time_index, residual)
        if time_index > 0:
            transition_ta = transition_ta.write(
                time_index - 1,
                _replicated_normals(
                    seeds,
                    (LEDH_PARTICLES, state_dimension),
                    1000 + time_index,
                ),
            )
    transition_noise = (
        tf.transpose(transition_ta.stack(), [1, 0, 2, 3])
        if horizon > 1
        else tf.zeros(
            [REPLICATES, 1, LEDH_PARTICLES, state_dimension], DTYPE
        )
    )
    return {
        "observations": observations,
        "initial_noise": initial_noise,
        "transition_noise": transition_noise,
        "fixed_reset_mask": tf.ones([REPLICATES, horizon], tf.bool),
        "residual_design": tf.transpose(residual_ta.stack(), [1, 0, 2, 3]),
        "prepared_ridge": tf.fill(
            [REPLICATES, horizon], tf.constant(1.0e-6, DTYPE)
        ),
        "epsilon": tf.constant(0.25, DTYPE),
        "scaling": tf.constant(0.9, DTYPE),
    }


def _ledh_call(name: str, prepared: Mapping[str, tf.Tensor]) -> Mapping[str, tf.Tensor]:
    function = (
        ledh.latent_sir_two_node_contract_e_value_and_score_tf
        if name == "two_node"
        else ledh.latent_sir_contract_e_canonical_value_and_score_tf
    )
    return function(
        THETA,
        prepared["observations"],
        prepared["initial_noise"],
        prepared["transition_noise"],
        prepared["fixed_reset_mask"],
        prepared["residual_design"],
        prepared["prepared_ridge"],
        prepared["epsilon"],
        prepared["scaling"],
    )


def _identity(name: str, prepared: Mapping[str, tf.Tensor]):
    issuer = (
        issue_latent_sir_two_node_contract_e_route_identity
        if name == "two_node"
        else issue_latent_sir_contract_e_route_identity
    )
    return issuer(prepared_inputs=prepared)


def _row(name: str, horizon: int, seed_start: int) -> Mapping[str, Any]:
    model = _model(name)
    observations = _observations(model, horizon, seed_start - 1)
    seeds = tf.range(seed_start, seed_start + REPLICATES, dtype=tf.int32)
    prepared = _prepared(model, observations, seeds)

    started = time.perf_counter()
    ledh_result = _ledh_call(name, prepared)
    ledh_seconds = time.perf_counter() - started
    started = time.perf_counter()
    ledh_replay = _ledh_call(name, prepared)
    ledh_replay_seconds = time.perf_counter() - started

    teachers = {}
    teacher_seconds = {}
    for particle_count in TEACHER_PARTICLES:
        teacher = make_online_sir_teacher(
            model,
            observations,
            seeds,
            num_particles=particle_count,
            jit_compile=True,
        )
        started = time.perf_counter()
        teachers[particle_count] = teacher(THETA)
        teacher_seconds[particle_count] = time.perf_counter() - started

    refinement = _comparison_summaries(
        teachers[256]["log_likelihood"],
        teachers[256]["score"],
        teachers[128]["log_likelihood"],
        teachers[128]["score"],
    )
    refinement_shift = bool(
        refinement["available"]
        and not all(
            summary["contains_zero"]
            for summary in (refinement["value"], *refinement["score"])
        )
    )
    comparison = _comparison_summaries(
        ledh_result["per_batch_log_likelihood"],
        ledh_result["per_batch_score"],
        teachers[256]["log_likelihood"],
        teachers[256]["score"],
    )
    ledh_value_finite = tf.math.is_finite(
        ledh_result["per_batch_log_likelihood"]
    )
    ledh_score_finite = tf.reduce_all(
        tf.math.is_finite(ledh_result["per_batch_score"]), axis=1
    )
    ledh_computational_valid = bool(
        tf.reduce_all(
            ledh_result["valid_chart"] & ledh_value_finite & ledh_score_finite
        ).numpy()
    )
    teacher_computational_valid = bool(
        all(
            tf.reduce_all(teachers[particle_count]["finite"]).numpy()
            for particle_count in TEACHER_PARTICLES
        )
    )
    computational_valid = bool(
        ledh_computational_valid
        and teacher_computational_valid
        and refinement["available"]
        and comparison["available"]
    )
    disagreement = bool(
        comparison["available"]
        and not all(
            summary["contains_zero"]
            for summary in (comparison["value"], *comparison["score"])
        )
    )
    if not computational_valid:
        status = "COMPUTATIONAL_INVALID_NONFINITE_OR_INVALID_CHART"
    elif refinement_shift:
        status = "TEACHER_N_REFINEMENT_SHIFT_DETECTED_COMPARISON_DESCRIPTIVE_ONLY"
    elif disagreement:
        status = "LEDH_TEACHER_METHOD_DISAGREEMENT_DETECTED"
    else:
        status = "NO_LED_H_TEACHER_DISAGREEMENT_DETECTED_AT_CURRENT_PRECISION"

    identity = _identity(name, prepared)
    return {
        "model_row": name,
        "horizon_T": horizon,
        "status": status,
        "computational_valid": computational_valid,
        "continuation_valid": computational_valid,
        "target": {
            "theta": THETA.numpy().tolist(),
            "observations": observations.numpy().tolist(),
            "target_hypothesis": (
                "explicit_two_node_Austria_edge_not_exact_nine_node_marginal"
                if name == "two_node"
                else "Austria_d18_latent_preclip_target"
            ),
        },
        "configuration": {
            "replicates": REPLICATES,
            "teacher_particles": list(TEACHER_PARTICLES),
            "ledh_particles": LEDH_PARTICLES,
            "seeds": seeds.numpy().tolist(),
            "jit_compile": True,
            "dtype": "float64",
        },
        "identity": {
            "identity_sha256": identity.identity_sha256,
            "prepared_input_sha256": identity.prepared_input_sha256,
            "source_dependency_closure_sha256": identity.source_dependency_closure_sha256,
            "route_specification_id": identity.route_specification_id,
            "admitted": False,
        },
        "ledh": {
            "value_samples": ledh_result["per_batch_log_likelihood"].numpy().tolist(),
            "score_samples": ledh_result["per_batch_score"].numpy().tolist(),
            "valid_chart": ledh_result["valid_chart"].numpy().tolist(),
            "value_finite": ledh_value_finite.numpy().tolist(),
            "score_finite": ledh_score_finite.numpy().tolist(),
            "computational_valid": ledh_computational_valid,
            "reset_valid_history": ledh_result["reset_valid_history"].numpy().tolist(),
            "minimum_mass_history": ledh_result["minimum_mass_history"].numpy().tolist(),
            "flow_valid_history": ledh_result["flow_valid_history"].numpy().tolist(),
            "geometry_valid_history": ledh_result["geometry_valid_history"].numpy().tolist(),
            "quotient_valid_history": ledh_result["quotient_valid_history"].numpy().tolist(),
            "reset_finite_history": ledh_result["reset_finite_history"].numpy().tolist(),
            "reset_factor_positive_history": ledh_result[
                "reset_factor_positive_history"
            ].numpy().tolist(),
            "covariance_gap_eigenvalue_history": ledh_result[
                "covariance_gap_eigenvalue_history"
            ].numpy().tolist(),
            "quotient_row_residual_history": ledh_result[
                "quotient_row_residual_history"
            ].numpy().tolist(),
            "quotient_column_residual_history": ledh_result[
                "quotient_column_residual_history"
            ].numpy().tolist(),
            "quotient_column_residual_scale_history": ledh_result[
                "quotient_column_residual_scale_history"
            ].numpy().tolist(),
            "quotient_post_column_residual_history": ledh_result[
                "quotient_post_column_residual_history"
            ].numpy().tolist(),
            "clip_boundary_away_history": ledh_result["clip_boundary_away_history"].numpy().tolist(),
            "first_call_seconds": ledh_seconds,
            "replay_seconds": ledh_replay_seconds,
            "replay_exact": bool(
                _nan_aware_exact(
                    ledh_result["per_batch_log_likelihood"],
                    ledh_replay["per_batch_log_likelihood"],
                )
                and _nan_aware_exact(
                    ledh_result["per_batch_score"],
                    ledh_replay["per_batch_score"],
                )
            ),
            "output_devices": sorted(
                {tensor.device for tensor in tf.nest.flatten(ledh_result)}
            ),
        },
        "teacher": {
            str(particle_count): {
                "value_samples": teachers[particle_count]["log_likelihood"].numpy().tolist(),
                "score_samples": teachers[particle_count]["score"].numpy().tolist(),
                "finite": teachers[particle_count]["finite"].numpy().tolist(),
                "minimum_ess": teachers[particle_count]["minimum_ess"].numpy().tolist(),
                "maximum_backward_row_sum_error": float(
                    tf.reduce_max(
                        teachers[particle_count]["maximum_backward_row_sum_error"]
                    ).numpy()
                ),
                "first_call_seconds": teacher_seconds[particle_count],
                "output_devices": sorted(
                    {tensor.device for tensor in tf.nest.flatten(teachers[particle_count])}
                ),
            }
            for particle_count in TEACHER_PARTICLES
        },
        "teacher_computational_valid": teacher_computational_valid,
        "teacher_refinement_N256_minus_N128": refinement,
        "ledh_N256_minus_teacher_N256": comparison,
        "teacher_refinement_shift_detected": refinement_shift,
        "method_disagreement_detected": bool(disagreement and not refinement_shift),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--phase3-prepared", type=Path, required=True)
    parser.add_argument("--phase3-cpu-result", type=Path, required=True)
    args = parser.parse_args()
    if GPU_MEMORY_POLICY is None:
        raise RuntimeError(
            f"set {GPU_MEMORY_LIMIT_ENV}={REQUIRED_GPU_MEMORY_LIMIT_MIB} before launch"
        )
    if len(GPU_MEMORY_POLICY["physical_devices"]) != 1:
        raise RuntimeError("this campaign requires exactly one visible capped GPU")
    if not tf.config.list_physical_devices("GPU"):
        raise RuntimeError("trusted GPU is required")
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_root}")
    args.output_root.mkdir(parents=True, exist_ok=False)

    started_at = datetime.now(timezone.utc)
    started = time.perf_counter()
    manifest_path = args.output_root / "run_manifest.json"
    manifest = {
        "schema": "bayesfilter.sir_ledh_teacher_comparison_manifest.v1",
        "status": "RUNNING",
        "git_commit": subprocess.run(
            ("git", "rev-parse", "HEAD"), cwd=ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip(),
        "git_dirty": True,
        "command": " ".join(sys.argv),
        "environment": os.environ.get("CONDA_DEFAULT_ENV", "unknown"),
        "python": platform.python_version(),
        "tensorflow": tf.__version__,
        "physical_gpus": [device.name for device in tf.config.list_physical_devices("GPU")],
        "gpu_memory_policy": GPU_MEMORY_POLICY,
        "jit_compile": True,
        "dtype": "float64",
        "tf32_execution_enabled": bool(tf.config.experimental.tensor_float_32_execution_enabled()),
        "started_at": started_at.isoformat(),
        "finished_at": None,
        "wall_time_seconds": None,
        "output_root": str(args.output_root),
        "plan_file": "docs/plans/bayesfilter-sir-remaining-gap-closure-master-plan-2026-07-16.md",
        "result_file": None,
        "failure_file": None,
        "trust_basis": "owner_designated_managed_session_visible_gpu_trusted",
        "source_sha256": {
            str(path): _sha256(ROOT / path)
            for path in (
                Path("bayesfilter/highdim/ledh_contract_e_latent_sir_tf.py"),
                Path("bayesfilter/highdim/ledh_contract_e_identity.py"),
                Path("bayesfilter/highdim/ledh_contract_e_streaming_tf.py"),
                Path("bayesfilter/highdim/ledh_contract_e_reset_tf.py"),
                Path("bayesfilter/highdim/sir_online_score_teacher_tf.py"),
                Path("bayesfilter/highdim/sir_latent_preclip_tf.py"),
                Path("experiments/dpf_implementation/tf_tfp/resampling/annealed_transport_tf.py"),
                Path("docs/benchmarks/run_sir_ledh_teacher_comparison_gpu.py"),
            )
        },
    }
    _write_json(manifest_path, manifest)

    rows = []
    continuation_veto = None
    active_stage = "phase3_final_source_replay"
    try:
        phase3_replay = _phase3_replay(
            args.phase3_prepared, args.phase3_cpu_result
        )
        _write_json(
            args.output_root / "phase3_final_source_replay.json", phase3_replay
        )
        for name, horizons, seed_start in MODEL_ROWS:
            for offset, horizon in enumerate(horizons):
                active_stage = f"{name}:T{horizon}"
                row = _row(name, horizon, seed_start + 100 * offset)
                rows.append(row)
                _write_json(
                    args.output_root / f"row_{name}_t{horizon}.json", row
                )
                _write_json(
                    args.output_root / "progress.json",
                    {
                        "schema": "bayesfilter.sir_ledh_teacher_comparison_progress.v1",
                        "status": "RUNNING",
                        "completed_rows": [
                            f"{item['model_row']}:T{item['horizon_T']}" for item in rows
                        ],
                        "active_stage": None,
                        "last_status": row["status"],
                    },
                )
                if not row["continuation_valid"]:
                    continuation_veto = (
                        f"{row['model_row']}:T{row['horizon_T']}:{row['status']}"
                    )
                    break
            if continuation_veto is not None:
                break
    except Exception as exc:
        completed_rows = [
            f"{item['model_row']}:T{item['horizon_T']}" for item in rows
        ]
        failure_path = args.output_root / "failure.json"
        failure = _failure_record(
            exc,
            active_stage=active_stage,
            completed_rows=completed_rows,
            started_at=started_at,
            started=started,
        )
        _write_json(failure_path, failure)
        _write_json(
            args.output_root / "progress.json",
            {
                "schema": "bayesfilter.sir_ledh_teacher_comparison_progress.v1",
                "status": "FAILED",
                "completed_rows": completed_rows,
                "active_stage": active_stage,
                "last_status": failure["failure_classification"],
            },
        )
        manifest.update(
            {
                "status": "FAILED",
                "finished_at": failure["failed_at"],
                "wall_time_seconds": failure["wall_time_seconds"],
                "failure_file": str(failure_path),
                "gpu_memory_info_terminal": failure["gpu_memory_info_at_failure"],
            }
        )
        _write_json(manifest_path, manifest)
        _write_artifact_hashes(args.output_root)
        print(
            json.dumps(
                {
                    "status": "CAMPAIGN_FAILED",
                    "classification": failure["failure_classification"],
                    "active_stage": active_stage,
                    "failure_file": str(failure_path),
                }
            ),
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    _write_json(
        args.output_root / "progress.json",
        {
            "schema": "bayesfilter.sir_ledh_teacher_comparison_progress.v1",
            "status": "STOPPED" if continuation_veto is not None else "COMPLETE",
            "completed_rows": [
                f"{item['model_row']}:T{item['horizon_T']}" for item in rows
            ],
            "active_stage": None,
            "last_status": rows[-1]["status"] if rows else phase3_replay["status"],
            "continuation_veto": continuation_veto,
        },
    )
    wall_time = time.perf_counter() - started
    hard_vetoes = [
        f"{row['model_row']}:T{row['horizon_T']}:{row['status']}"
        for row in rows
        if row["status"] != "NO_LED_H_TEACHER_DISAGREEMENT_DETECTED_AT_CURRENT_PRECISION"
    ]
    payload = {
        "schema": "bayesfilter.sir_ledh_teacher_paired_comparison.v1",
        "status": (
            "COMPARISON_STOPPED_COMPUTATIONAL_INVALID"
            if continuation_veto is not None
            else (
                "COMPARISON_COMPLETE_WITH_VETOES"
                if hard_vetoes
                else "COMPARISON_COMPLETE_NO_DETECTED_DISAGREEMENT"
            )
        ),
        "phase3_final_source_replay": phase3_replay,
        "rows": rows,
        "hard_vetoes": hard_vetoes,
        "continuation_veto": continuation_veto,
        "evidence_contract": {
            "teacher_refinement": "paired N256 minus N128 Bonferroni Student intervals",
            "method_comparison": "paired LEDH N256 minus teacher N256 Bonferroni Student intervals",
            "refinement_shift_rule": "any interval excluding zero makes method comparison descriptive only",
            "equivalence_claim_available": False,
        },
        "decision_table": {
            "decision": (
                "stop ladder and diagnose computational invalidity"
                if continuation_veto is not None
                else (
                    "repair before promotion"
                    if hard_vetoes
                    else "continue without accuracy promotion"
                )
            ),
            "primary_criterion_status": hard_vetoes or "no disagreement detected",
            "veto_diagnostic_status": hard_vetoes or "none",
            "main_uncertainty": "J2 and d18 teacher have no external target oracle; intervals may be underpowered",
            "next_justified_action": (
                "diagnose the first invalid chart or nonfinite output before any later horizon"
                if continuation_veto is not None
                else "diagnose each refinement shift or method disagreement; preserve nonclaims"
            ),
            "not_concluded": "which method is closer to truth, practical equivalence, HMC readiness, leaderboard readiness",
        },
        "inference_status": {
            "hard_veto_screen": hard_vetoes or "none",
            "statistically_supported_ranking": "none",
            "descriptive_only_differences": "all rows with teacher refinement shift plus runtime/ESS trends",
            "default_readiness": "not established",
            "next_evidence_needed": "localized repair diagnostics and Zhao-Cui source-route derivative closure",
        },
        "post_run_red_team": {
            "strongest_alternative_explanation": "teacher particle bias or low power can mimic or hide method disagreement",
            "result_that_would_overturn_conclusion": "a repaired stable teacher or external oracle reverses the interval classification",
            "weakest_evidence": "no J2 or d18 external likelihood/score oracle",
        },
        "nonclaims": [
            "not evidence that either method is closer to truth",
            "not practical equivalence when intervals contain zero",
            "not HMC or leaderboard readiness",
            "not Zhao-Cui source-faithful comparator closure",
        ],
    }
    result_path = args.output_root / "result.json"
    _write_json(result_path, payload)
    manifest.update(
        {
            "status": (
                "COMPLETE_WITH_CONTINUATION_VETO"
                if continuation_veto is not None
                else "COMPLETE"
            ),
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "wall_time_seconds": wall_time,
            "result_file": str(result_path),
            "gpu_memory_info_terminal": _gpu_memory_info(),
        }
    )
    _write_json(manifest_path, manifest)
    _write_artifact_hashes(args.output_root)
    print(json.dumps({"status": payload["status"], "hard_vetoes": hard_vetoes}))


if __name__ == "__main__":
    main()
