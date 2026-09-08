"""Run the bounded Phase 8B exact-likelihood Laplace calibration/comparison.

The runner uses the same frozen C2 APF value/score evaluator for the Laplace
candidate and every comparator.  Schedule selection is performed on
independent calibration fixtures before ESS is inspected.  The original C2
observation path is a mechanism-regression path only; this script makes no
posterior, superiority, or production claim.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence


# This TensorFlow build can initialize logical devices while importing helper
# modules.  Defer the inherited growth variable until the repository policy is
# called explicitly.
_DEFERRED_TF_FORCE_GPU_ALLOW_GROWTH = os.environ.pop(
    "TF_FORCE_GPU_ALLOW_GROWTH", None
)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import tensorflow as tf


DTYPE = tf.float64
STATE_DIM = 4
HORIZON = 20
DEFAULT_PARTICLE_COUNT = 1024
CALIBRATION_PARTICLE_COUNT = 64
CALIBRATION_HORIZON = 8
PHASE_ID = "c2_exact_likelihood_laplace_phase8b_calibration_v1"
RESULT_SCHEMA = "c2_exact_likelihood_laplace_result_v2"
MANIFEST_SCHEMA = "c2_exact_likelihood_laplace_manifest_v2"
ROUTE_ID = "c2_exact_likelihood_laplace_apf_k1_frozen_branch_v1"
ROUTE_CLASSIFICATION = "extension_or_invention_candidate_diagnostic_only"

PLAN_PATH = ROOT / "docs/plans/c2-exact-likelihood-laplace-mixture-apf-phase8-20260904.md"
MASTER_PLAN_PATH = ROOT / "docs/plans/bayesfilter-c2-mixture-ukf-apf-master-program-2026-09-02.md"
FIXTURE_PATH = ROOT / "docs/benchmarks/fixtures/c2_sv_n4_seed52_obs42_t20_frozen_v1.json"
PHASE7_SNAPSHOT_ROOT = ROOT / "docs/benchmarks/artifacts/c2_mixture_ukf_apf_20260902/phase7-integrated-attempt02/snapshots"
DRIVER_PATH = ROOT / "docs/benchmarks/run_c2_exact_likelihood_laplace_phase8b_20260904.py"
LAPLACE_PATH = ROOT / "bayesfilter/highdim/exact_likelihood_laplace_apf_tf.py"
LAPLACE_ADAPTER_PATH = ROOT / "bayesfilter/highdim/c2_exact_likelihood_laplace_adapter.py"
MODEL_PATH = ROOT / "bayesfilter/highdim/c2_sv_frozen_proposal_apf_tf.py"
UKF_PATH = ROOT / "bayesfilter/highdim/c2_mixture_ukf_apf_c2_adapter.py"
EXACT_PATH = ROOT / "bayesfilter/highdim/zhao_cui_frozen_proposal_apf_tf.py"

# The ladder is fixed before any candidate ESS is evaluated.  The final
# configuration is selected by validity/residual/ascent/runtime only.
SCHEDULE_LADDER = (
    {
        "config_id": "one_full",
        "tempering_schedule": (1.0,),
        "step_fractions": (1.0,),
    },
    {
        "config_id": "half_full",
        "tempering_schedule": (0.5, 1.0),
        "step_fractions": (1.0, 1.0),
    },
    {
        "config_id": "quarter_full",
        "tempering_schedule": (0.25, 0.5, 0.75, 1.0),
        "step_fractions": (1.0, 1.0, 1.0, 1.0),
    },
    {
        "config_id": "quarter_half_steps",
        "tempering_schedule": (0.25, 0.5, 0.75, 1.0),
        "step_fractions": (0.5, 0.5, 0.5, 0.5),
    },
    {
        "config_id": "quarter_long",
        "tempering_schedule": (0.25, 0.5, 0.75, 1.0, 1.0, 1.0, 1.0, 1.0),
        "step_fractions": (1.0,) * 8,
    },
)
# These schedules are deliberately opt-in.  They are repair hypotheses
# nominated by the Phase 8D seed-424245 diagnostic and must not alter the
# original Phase 8B calibration unless the caller requests them explicitly.
REPAIR_SCHEDULE_LADDER = (
    {
        "config_id": "quarter_long_12",
        "tempering_schedule": (0.25, 0.5, 0.75, 1.0, 1.0, 1.0, 1.0, 1.0,
                                1.0, 1.0, 1.0, 1.0),
        "step_fractions": (1.0,) * 12,
    },
    {
        "config_id": "quarter_long_16",
        "tempering_schedule": (0.25, 0.5, 0.75, 1.0, 1.0, 1.0, 1.0, 1.0,
                                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        "step_fractions": (1.0,) * 16,
    },
    {
        "config_id": "fine_tempering_12",
        "tempering_schedule": (0.125, 0.25, 0.375, 0.5, 0.625, 0.75,
                                0.875, 1.0, 1.0, 1.0, 1.0, 1.0),
        "step_fractions": (1.0,) * 12,
    },
)
STATIONARITY_TOLERANCE = 2.0e-10
ASCENT_TOLERANCE = 2.0e-13
ABS_TOL = 2.0e-10
SCORE_TOL = 2.0e-7
# At time zero all families evaluate the same initial prior cloud; no
# transition proposal has yet used the observation.  The proposal-mechanism
# screen therefore starts at the first active transition while retaining t=0
# as an explanatory diagnostic.
PROPOSAL_ACTIVE_START_TIME = 1


def _phase_token(phase_id: str) -> str:
    """Map a phase identifier to the result-schema phase label."""
    lowered = str(phase_id).lower()
    if "phase8e" in lowered:
        return "PHASE8E"
    if "phase8d" in lowered:
        return "PHASE8D"
    if "phase8c" in lowered:
        return "PHASE8C"
    return "PHASE8B"


def _continuation_token(phase_token: str, valid: bool) -> str:
    if not valid:
        return f"CONTINUATION_VETO_REPAIR_{phase_token}"
    return {
        "PHASE8B": "CONTINUE_PHASE8C_AFTER_COST_REFRESH",
        "PHASE8C": "CONTINUE_PHASE8D_FROZEN_REPLICATION",
        "PHASE8D": "CONTINUE_PHASE8E_STATISTICAL_REPLICATION",
        "PHASE8E": "CONTINUE_PHASE8E_TERMINAL_REVIEW",
    }[phase_token]


CANDIDATE_LABEL = "exact_likelihood_laplace_k1"


def _row_has_accepted_program_failure(row: Mapping[str, object]) -> bool:
    """Identify failures that invalidate the accepted finite program.

    A failed fixed-Laplace stationarity check is a candidate repair trigger.
    A non-finite target/denominator or an invalid comparator is a continuation
    veto because the comparison no longer answers the declared question.
    """

    checks = row.get("checks")
    if isinstance(checks, Mapping):
        for key in (
            "finite_exact_value_and_score",
            "complete_denominator_and_law_finite",
            "apf_w_over_a_identity",
            "apf_recomputed_final_weight_parity",
        ):
            if key in checks and not bool(checks[key]):
                return True
    error = str(row.get("error", "")).lower()
    return any(
        token in error
        for token in (
            "non-finite",
            "nonfinite",
            "target/measure",
            "denominator mismatch",
        )
    )


def _record_validity_summary(
    records: Sequence[Mapping[str, object]],
    *,
    branch_count: int,
    family_count: int,
) -> Mapping[str, object]:
    """Separate candidate repair triggers from continuation-veto failures."""

    expected = int(branch_count) * int(family_count)
    candidate_rows = [
        row for row in records if str(row.get("label")) == CANDIDATE_LABEL
    ]
    comparator_rows = [
        row for row in records if str(row.get("label")) != CANDIDATE_LABEL
    ]
    records_complete = len(records) == expected
    candidate_complete = len(candidate_rows) == int(branch_count)
    comparator_complete = len(comparator_rows) == int(branch_count) * (int(family_count) - 1)
    candidate_valid = candidate_complete and all(
        bool(row.get("all_checks_pass", False)) for row in candidate_rows
    )
    comparator_valid = comparator_complete and all(
        bool(row.get("all_checks_pass", False)) for row in comparator_rows
    )
    candidate_failure = candidate_complete and not candidate_valid
    accepted_program_failure = any(
        _row_has_accepted_program_failure(row) for row in records
    )
    continuation_veto = (
        not records_complete
        or not comparator_valid
        or accepted_program_failure
    )
    return {
        "records_complete": records_complete,
        "candidate_branch_validity": candidate_valid,
        "comparator_branch_validity": comparator_valid,
        "all_branch_validity": records_complete and candidate_valid and comparator_valid,
        "candidate_failure": candidate_failure,
        "accepted_program_failure": accepted_program_failure,
        "continuation_veto": continuation_veto,
    }


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load helper module {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*args: str) -> str:
    try:
        completed = subprocess.run(
            ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
        )
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"
    return completed.stdout.strip()


def _jsonable(value: object) -> object:
    if isinstance(value, tf.Tensor):
        return _jsonable(value.numpy())
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite artifact value: {value!r}")
        return value
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    if hasattr(value, "item"):
        return _jsonable(value.item())
    raise TypeError(f"unsupported artifact value: {type(value).__name__}")


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_tensor(path: Path, value: tf.Tensor) -> Mapping[str, object]:
    tensor = tf.convert_to_tensor(value)
    path.write_bytes(tf.io.serialize_tensor(tensor).numpy())
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": _sha256_file(path),
        "shape": [int(x) for x in tensor.shape],
        "dtype": tensor.dtype.name,
    }


def _finite(value: object) -> bool:
    return bool(
        tf.reduce_all(tf.math.is_finite(tf.convert_to_tensor(value, DTYPE))).numpy()
    )


def _scalar(value: object) -> float:
    tensor = tf.reshape(tf.convert_to_tensor(value, DTYPE), [])
    if not _finite(tensor):
        raise ValueError("non-finite scalar")
    return float(tensor.numpy())


def _vector(value: object) -> list[float]:
    tensor = tf.reshape(tf.convert_to_tensor(value, DTYPE), [-1])
    if not _finite(tensor):
        raise ValueError("non-finite vector")
    return [float(x) for x in tensor.numpy().tolist()]


def _max_abs(value: object) -> float:
    return _scalar(tf.reduce_max(tf.abs(tf.convert_to_tensor(value, DTYPE))))


def _fresh_output(value: str) -> Path:
    candidate = Path(value)
    output = (ROOT / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True, exist_ok=False)
    return output


def _configure_gpu() -> Mapping[str, object]:
    if _DEFERRED_TF_FORCE_GPU_ALLOW_GROWTH is not None:
        os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = _DEFERRED_TF_FORCE_GPU_ALLOW_GROWTH
    from bayesfilter.runtime.gpu_memory_policy import (
        configure_tensorflow_gpu_memory_growth,
    )

    physical = tuple(tf.config.list_physical_devices("GPU"))
    policy = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    logical = tuple(tf.config.list_logical_devices("GPU"))
    if not logical:
        raise RuntimeError("Phase 8B requires a logical TensorFlow GPU")
    with tf.device("/GPU:0"):
        probe = tf.reduce_sum(tf.ones([32], DTYPE))
    if "GPU" not in str(probe.device).upper():
        raise RuntimeError(f"GPU placement probe ran on {probe.device}")
    return {
        "physical_devices": [str(item.name) for item in physical],
        "logical_devices": [str(item.name) for item in logical],
        "placement_probe_device": str(probe.device),
        "placement_probe_value": _scalar(probe),
        "memory_policy": policy,
    }


def _load_fixture(path: Path = FIXTURE_PATH) -> Mapping[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_id") != "bayesfilter.c2_sv_frozen_fixture.v1":
        raise ValueError("unexpected C2 fixture schema")
    if int(payload["state_dimension"]) != STATE_DIM or int(payload["horizon"]) != HORIZON:
        raise ValueError("unexpected C2 fixture dimensions")
    return payload


def _model_inputs(p2: Any, fixture: Mapping[str, object]):
    return p2._c2_model_and_inputs(fixture)


def _c2_calibration_observations(model: Any, theta: tf.Tensor, *, seed: tuple[int, int]) -> tuple[tf.Tensor, tf.Tensor]:
    """Return ordinary and near-zero independent calibration observations."""

    horizon = CALIBRATION_HORIZON
    dimension = int(model.state_dim())
    transition = model.transition_matrix(theta)
    covariance, _ = model.stationary_covariance_and_derivative(theta)
    chol = tf.linalg.cholesky(covariance)
    process_normals = tf.random.stateless_normal([horizon, dimension], seed, dtype=DTYPE)
    observation_normals = tf.random.stateless_normal(
        [horizon, dimension], [seed[0], seed[1] + 1], dtype=DTYPE
    )
    states = tf.TensorArray(DTYPE, size=horizon, clear_after_read=False,
                            element_shape=[dimension])
    states = states.write(0, tf.linalg.matvec(chol, process_normals[0]))

    def condition(index: tf.Tensor, _: tf.TensorArray) -> tf.Tensor:
        return index < horizon

    def body(index: tf.Tensor, values: tf.TensorArray):
        previous = values.read(index - 1)
        current = tf.linalg.matvec(transition, previous) + tf.constant(float(model.sigma), DTYPE) * process_normals[index]
        return index + 1, values.write(index, current)

    _, states = tf.while_loop(
        condition, body, (tf.constant(1, tf.int32), states), parallel_iterations=1
    )
    latent = states.stack()
    ordinary = tf.exp(0.5 * latent + theta[1]) * observation_normals
    near_zero = tf.tensor_scatter_nd_update(
        ordinary,
        tf.constant([[3]], tf.int32),
        tf.zeros([1, dimension], DTYPE) + tf.constant(1.0e-8, DTYPE),
    )
    return tf.ensure_shape(ordinary, [horizon, dimension]), tf.ensure_shape(
        near_zero, [horizon, dimension]
    )


def _linear_callbacks():
    matrix = tf.constant([[1.0, -0.4]], DTYPE)
    variance = tf.constant(0.6, DTYPE)

    def log_likelihood(states: tf.Tensor, observation: tf.Tensor) -> tf.Tensor:
        residual = observation[tf.newaxis, :] - tf.linalg.matmul(states, matrix, transpose_b=True)
        return -0.5 * (
            tf.cast(tf.shape(observation)[0], DTYPE)
            * tf.math.log(tf.constant(2.0 * math.pi, DTYPE) * variance)
            + tf.reduce_sum(tf.square(residual), axis=1) / variance
        )

    def score(states: tf.Tensor, observation: tf.Tensor) -> tf.Tensor:
        residual = observation[tf.newaxis, :] - tf.linalg.matmul(states, matrix, transpose_b=True)
        return tf.linalg.matmul(residual, matrix) / variance

    def information(states: tf.Tensor, observation: tf.Tensor) -> tf.Tensor:
        del observation
        value = tf.linalg.matmul(matrix, matrix, transpose_a=True) / variance
        return tf.broadcast_to(value[tf.newaxis, :, :], [tf.shape(states)[0], 2, 2])

    return log_likelihood, score, information


def _bimodal_callbacks():
    modes = tf.constant([-2.0, 2.0], DTYPE)
    variance = tf.constant(0.36, DTYPE)

    def component_log_density(states: tf.Tensor) -> tf.Tensor:
        x = states[:, 0:1]
        return -0.5 * tf.square(x - modes[tf.newaxis, :]) / variance

    def log_likelihood(states: tf.Tensor, observation: tf.Tensor) -> tf.Tensor:
        del observation
        return tf.reduce_logsumexp(component_log_density(states), axis=1) - tf.math.log(tf.constant(2.0, DTYPE))

    def score(states: tf.Tensor, observation: tf.Tensor) -> tf.Tensor:
        del observation
        logs = component_log_density(states)
        responsibilities = tf.nn.softmax(logs, axis=1)
        x = states[:, 0:1]
        component_scores = -(x - modes[tf.newaxis, :]) / variance
        return tf.reduce_sum(responsibilities * component_scores, axis=1, keepdims=True)

    def information(states: tf.Tensor, observation: tf.Tensor) -> tf.Tensor:
        del observation
        logs = component_log_density(states)
        responsibilities = tf.nn.softmax(logs, axis=1)
        x = states[:, 0:1]
        component_scores = -(x - modes[tf.newaxis, :]) / variance
        mean_score = tf.reduce_sum(responsibilities * component_scores, axis=1)
        second = tf.reduce_sum(responsibilities * tf.square(component_scores), axis=1)
        negative_hessian = 1.0 / variance - (second - tf.square(mean_score))
        return tf.reshape(negative_hessian, [-1, 1, 1])

    return log_likelihood, score, information


def _trace_metrics(result: Mapping[str, tf.Tensor]) -> Mapping[str, object]:
    before = tf.convert_to_tensor(result["iteration_objective_before_step"], DTYPE)
    after = tf.convert_to_tensor(result["iteration_objective_after_step"], DTYPE)
    improvement = after - before
    return {
        "valid": bool(result["valid"].numpy()),
        "finite": bool(result["finite"].numpy()),
        "maximum_relative_stationarity_residual": _max_abs(result["relative_stationarity_residual"]),
        "minimum_precision_eigenvalue": _scalar(tf.reduce_min(result["minimum_precision_eigenvalue"])),
        "minimum_covariance_eigenvalue": _scalar(tf.reduce_min(result["minimum_covariance_eigenvalue"])),
        "minimum_objective_improvement": _scalar(tf.reduce_min(improvement)),
        "maximum_objective_decrease_allowed": -ASCENT_TOLERANCE,
        "trace_count": int(result["iteration_count"].numpy()),
    }


def _run_generic_fixture(config: Any, *, fixture_name: str) -> Mapping[str, object]:
    if fixture_name == "linear_gaussian":
        callbacks = _linear_callbacks()
        prior_means = tf.constant([[0.3, -0.2], [-0.1, 0.5], [0.7, 0.4], [0.0, 0.1]], DTYPE)
        prior_covariance = tf.constant([[1.2, 0.2], [0.2, 0.8]], DTYPE)
        prior_covariances = tf.broadcast_to(prior_covariance[None, :, :], [4, 2, 2])
        observation = tf.constant([0.25], DTYPE)
        state_dim = 2
        observation_dim = 1
        means = prior_means
    else:
        callbacks = _bimodal_callbacks()
        means = tf.zeros([4, 1], DTYPE)
        prior_covariances = tf.ones([4, 1, 1], DTYPE)
        observation = tf.zeros([1], DTYPE)
        state_dim = 1
        observation_dim = 1
    from bayesfilter.highdim.exact_likelihood_laplace_apf_tf import (
        make_fixed_laplace_bank_kernel,
        single_start_offsets,
    )

    kernel = make_fixed_laplace_bank_kernel(
        batch_size=4,
        state_dim=state_dim,
        observation_dim=observation_dim,
        component_offsets=single_start_offsets(state_dim),
        log_likelihood_fn=callbacks[0],
        state_score_fn=callbacks[1],
        state_negative_hessian_fn=callbacks[2],
        config=config,
        jit_compile=True,
    )
    started = time.perf_counter()
    result = kernel(means, prior_covariances, observation)
    elapsed = time.perf_counter() - started
    metrics = dict(_trace_metrics(result))
    metrics["wall_seconds"] = elapsed
    metrics["kernel_trace_count"] = int(kernel.experimental_get_tracing_count())
    metrics["expected_invalid_curvature_fixture"] = fixture_name == "bimodal"
    metrics["invalid_curvature_fail_closed"] = (
        fixture_name != "bimodal" or (not metrics["valid"] and metrics["finite"])
    )
    metrics["selection_valid"] = (
        metrics["valid"]
        and metrics["finite"]
        and metrics["maximum_relative_stationarity_residual"] <= float(config.stationarity_relative_tolerance)
        and metrics["minimum_objective_improvement"] >= -ASCENT_TOLERANCE
    )
    return metrics


def _run_c2_calibration(
    *, model: Any, theta: tf.Tensor, ordinary: tf.Tensor, near_zero: tf.Tensor,
    config: Any, seed: int,
) -> Mapping[str, object]:
    from bayesfilter.highdim.c2_exact_likelihood_laplace_adapter import (
        compile_c2_exact_likelihood_laplace_apf_k1,
    )

    rows: dict[str, object] = {}
    for name, observations in (("c2_ordinary", ordinary), ("c2_near_zero", near_zero)):
        started = time.perf_counter()
        try:
            compilation = compile_c2_exact_likelihood_laplace_apf_k1(
                model=model,
                observations=observations,
                theta_reference=theta,
                particle_count=CALIBRATION_PARTICLE_COUNT,
                seed=seed,
                laplace_config=config,
                jit_compile=True,
            )
            diagnostics = [_trace_metrics_from_row(row) for row in compilation.proposal_diagnostics]
            max_residual = max(float(row["maximum_relative_stationarity_residual"]) for row in diagnostics)
            min_improvement = min(float(row["minimum_objective_improvement"]) for row in diagnostics)
            all_valid = all(bool(row["valid"]) and bool(row["finite"]) for row in diagnostics)
            rows[name] = {
                "valid": all_valid,
                "finite": all_valid,
                "maximum_relative_stationarity_residual": max_residual,
                "minimum_objective_improvement": min_improvement,
                "trace_count": int(compilation.manifest["laplace_kernel_trace_count"]),
                "sampler_trace_count": int(compilation.manifest["sampler_trace_count"]),
                "wall_seconds": time.perf_counter() - started,
                "selection_valid": bool(
                    all_valid
                    and max_residual <= float(config.stationarity_relative_tolerance)
                    and min_improvement >= -ASCENT_TOLERANCE
                ),
            }
        except Exception as exc:
            rows[name] = {
                "valid": False,
                "finite": False,
                "selection_valid": False,
                "wall_seconds": time.perf_counter() - started,
                "maximum_relative_stationarity_residual": None,
                "minimum_objective_improvement": None,
                "error": f"{type(exc).__name__}: {exc}",
            }
    return rows


def _trace_metrics_from_row(row: Mapping[str, object]) -> Mapping[str, object]:
    before = tf.convert_to_tensor(row["iteration_objective_before_step"], DTYPE)
    after = tf.convert_to_tensor(row["iteration_objective_after_step"], DTYPE)
    return {
        "valid": bool(tf.convert_to_tensor(row["laplace_valid"]).numpy()),
        "finite": bool(tf.convert_to_tensor(row["proposal_finite"]).numpy())
        and bool(tf.convert_to_tensor(row["exact_prefix_finite"]).numpy()),
        "maximum_relative_stationarity_residual": _max_abs(row["relative_stationarity_residual"]),
        "minimum_objective_improvement": _scalar(tf.reduce_min(after - before)),
    }


def _calibrate_schedule(
    *, model: Any, theta: tf.Tensor,
    schedule_ladder: Sequence[Mapping[str, object]] = SCHEDULE_LADDER,
) -> Mapping[str, object]:
    from bayesfilter.highdim.exact_likelihood_laplace_apf_tf import FixedLaplaceConfig

    ordinary, near_zero = _c2_calibration_observations(model, theta, seed=(20260904, 8811))
    records: list[Mapping[str, object]] = []
    for index, spec in enumerate(schedule_ladder):
        config = FixedLaplaceConfig(
            tuple(spec["tempering_schedule"]),
            tuple(spec["step_fractions"]),
            0.0,
            STATIONARITY_TOLERANCE,
        )
        started = time.perf_counter()
        linear = _run_generic_fixture(config, fixture_name="linear_gaussian")
        bimodal = _run_generic_fixture(config, fixture_name="bimodal")
        c2 = _run_c2_calibration(
            model=model, theta=theta, ordinary=ordinary, near_zero=near_zero,
            config=config, seed=99100 + index,
        )
        valid = bool(
            linear["selection_valid"]
            and c2["c2_ordinary"]["selection_valid"]
            and c2["c2_near_zero"]["selection_valid"]
        )
        residual_values = [
            float(linear["maximum_relative_stationarity_residual"]),
            *[
                float(c2[name]["maximum_relative_stationarity_residual"])
                for name in ("c2_ordinary", "c2_near_zero")
                if c2[name].get("maximum_relative_stationarity_residual") is not None
            ],
        ]
        worst_residual_value = max(residual_values) if len(residual_values) == 3 else float("inf")
        record = {
            "config_id": spec["config_id"],
            "tempering_schedule": list(spec["tempering_schedule"]),
            "step_fractions": list(spec["step_fractions"]),
            "iteration_count": len(spec["tempering_schedule"]),
            "linear_gaussian": linear,
            "bimodal": bimodal,
            "c2": c2,
            "selection_valid": valid,
            "worst_selection_residual": (
                worst_residual_value if math.isfinite(worst_residual_value) else None
            ),
            "wall_seconds": time.perf_counter() - started,
        }
        records.append(record)

    eligible = [row for row in records if bool(row["selection_valid"])]
    if not eligible:
        raise RuntimeError("no Phase 8B fixed schedule passed calibration")
    selected = sorted(
        eligible,
        key=lambda row: (
            float(row["worst_selection_residual"]),
            float(row["wall_seconds"]),
            int(row["iteration_count"]),
        ),
    )[0]
    return {
        "schema_version": "c2_phase8b_schedule_calibration_v1",
        "calibration_seed": [20260904, 8811],
        "particle_count": CALIBRATION_PARTICLE_COUNT,
        "horizon": CALIBRATION_HORIZON,
        "selection_rule": "valid linear and C2 ordinary/near-zero fixtures; minimize worst stationarity residual, then runtime, then iteration count",
        "claim_observations_not_used": True,
        "records": records,
        "eligible_config_ids": [row["config_id"] for row in eligible],
        "selected": selected,
        "ordinary_observations": ordinary,
        "near_zero_observations": near_zero,
    }


def _load_gaussian_hint_proposals(horizon: int):
    from bayesfilter.highdim.c2_sv_frozen_proposal_apf_tf import FrozenGaussianStateProposal

    proposals = []
    hashes: dict[str, str] = {}
    for time_index in range(1, int(horizon)):
        directory = PHASE7_SNAPSHOT_ROOT / f"t{time_index:02d}"
        metadata_path = directory / "metadata.json"
        mean_path = directory / "coordinate_offset.tensor"
        chol_path = directory / "coordinate_matrix.tensor"
        if not (metadata_path.is_file() and mean_path.is_file() and chol_path.is_file()):
            raise FileNotFoundError(f"missing Gaussian-hint snapshot at t={time_index}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("schema_id") != "gaussian_xla_retained_proposal_snapshot_v1":
            raise ValueError("unexpected Gaussian-hint snapshot schema")
        mean = tf.io.parse_tensor(tf.io.read_file(str(mean_path)), out_type=DTYPE)
        chol = tf.io.parse_tensor(tf.io.read_file(str(chol_path)), out_type=DTYPE)
        if mean.shape != (STATE_DIM,) or chol.shape != (STATE_DIM, STATE_DIM):
            raise ValueError("Gaussian-hint snapshot shape mismatch")
        proposals.append(FrozenGaussianStateProposal(
            mean=mean, chol=chol, time_index=time_index, family="gaussian_hint_marginal"
        ))
        hashes[f"t{time_index:02d}"] = _sha256_file(metadata_path)
    return tuple(proposals), hashes


def _exact_weight_path(p2: Any, model: Any, branch: Any, theta: tf.Tensor) -> tf.Tensor:
    initial = (
        model.initial_log_density(theta, branch.states[0])
        + model.observation_log_density(theta, branch.states[0], branch.observations[0], 0)
    )
    unnormalized = branch.initial_log_base_mass + initial - branch.initial_log_proposal_density
    weights = unnormalized - tf.reduce_logsumexp(unnormalized)
    paths = [weights]
    for time_index in range(1, branch.time_steps):
        ancestor = branch.ancestors[time_index - 1]
        previous = tf.gather(branch.states[time_index - 1], ancestor)
        current = branch.states[time_index]
        selected_weight = tf.gather(weights, ancestor)
        selected_auxiliary = tf.gather(
            branch.auxiliary_log_probabilities[time_index - 1], ancestor
        )
        transition_log = model.transition_log_density(theta, previous, current, time_index)
        observation_log = model.observation_log_density(
            theta, current, branch.observations[time_index], time_index
        )
        corrected = (
            selected_weight + transition_log + observation_log - selected_auxiliary
            - branch.transition_log_proposal_density[time_index - 1]
        )
        unnormalized = branch.transition_log_base_mass[time_index - 1] + corrected
        weights = unnormalized - tf.reduce_logsumexp(unnormalized)
        paths.append(weights)
    return tf.stack(paths)


def _save_branch_tensors(output: Path, label: str, compilation: Any, weight_path: tf.Tensor) -> Mapping[str, object]:
    branch = compilation.branch
    safe_label = label.replace("/", "_")
    sidecar: dict[str, object] = {}
    sidecar["states"] = _write_tensor(output / f"{safe_label}_states.tensor", branch.states)
    sidecar["ancestors"] = _write_tensor(output / f"{safe_label}_ancestors.tensor", branch.ancestors)
    sidecar["auxiliary_log_probabilities"] = _write_tensor(
        output / f"{safe_label}_auxiliary_log_probabilities.tensor", branch.auxiliary_log_probabilities
    )
    sidecar["transition_log_proposal_density"] = _write_tensor(
        output / f"{safe_label}_transition_log_proposal_density.tensor", branch.transition_log_proposal_density
    )
    sidecar["normalized_log_weights_by_time"] = _write_tensor(
        output / f"{safe_label}_normalized_log_weights_by_time.tensor", weight_path
    )
    if label == "exact_likelihood_laplace_k1":
        rows = compilation.proposal_diagnostics
        sidecar["posterior_means"] = _write_tensor(
            output / f"{safe_label}_posterior_means.tensor",
            tf.stack([tf.convert_to_tensor(row["posterior_mean"], DTYPE) for row in rows]),
        )
        sidecar["posterior_covariances"] = _write_tensor(
            output / f"{safe_label}_posterior_covariances.tensor",
            tf.stack([tf.convert_to_tensor(row["posterior_covariance"], DTYPE) for row in rows]),
        )
        for key in (
            "relative_stationarity_residual",
            "minimum_precision_eigenvalue",
            "minimum_covariance_eigenvalue",
            "iteration_step_max_abs",
            "iteration_objective_before_step",
            "iteration_objective_after_step",
            "lookahead_log_likelihood",
            "ancestor_log_probabilities",
        ):
            if key in rows[0]:
                sidecar[key] = _write_tensor(
                    output / f"{safe_label}_{key}.tensor",
                    tf.stack([tf.convert_to_tensor(row[key], DTYPE) for row in rows]),
                )
    return sidecar


def _central_difference(program: Any, theta: tf.Tensor, index: int) -> tf.Tensor:
    direction = tf.one_hot(index, int(theta.shape[0]), dtype=DTYPE)
    step = tf.constant(1.0e-5, DTYPE)
    return (
        program.evaluate(theta + step * direction)["log_likelihood"]
        - program.evaluate(theta - step * direction)["log_likelihood"]
    ) / (2.0 * step)


def _evaluate_branch(
    *, p2: Any, label: str, compilation: Any, model: Any, theta: tf.Tensor,
    output: Path, check_xla: bool,
) -> Mapping[str, object]:
    program = p2.prepare_frozen_proposal_apf_program(model, compilation.branch)
    result = program.evaluate(theta)
    finite = bool(result["finite"].numpy()) and _finite(result["log_likelihood"]) and _finite(result["score"])
    finite_difference = tf.stack([_central_difference(program, theta, i) for i in range(int(theta.shape[0]))]) if finite else tf.fill([int(theta.shape[0])], tf.constant(float("nan"), DTYPE))
    score_error = _max_abs(result["score"] - finite_difference) if finite else float("inf")
    nonjit_value_error = float("inf")
    nonjit_score_error = float("inf")
    nonjit_error = ""
    try:
        nonjit = program.compiled(jit_compile=False)(theta)
        nonjit_value_error = _max_abs(result["log_likelihood"] - nonjit["log_likelihood"])
        nonjit_score_error = _max_abs(result["score"] - nonjit["score"])
    except Exception as exc:
        nonjit_error = f"{type(exc).__name__}: {exc}"
    xla_value_error = None
    xla_score_error = None
    xla_error = ""
    if check_xla:
        try:
            xla = program.compiled(jit_compile=True)(theta)
            xla_value_error = _max_abs(result["log_likelihood"] - xla["log_likelihood"])
            xla_score_error = _max_abs(result["score"] - xla["score"])
        except Exception as exc:
            xla_error = f"{type(exc).__name__}: {exc}"
    apf = p2._apf_identity_and_law_diagnostics(
        model, compilation.branch, theta, expected_final_log_weights=result["final_log_weights"]
    )
    proposal = p2._branch_proposal_diagnostics(compilation)
    checks: dict[str, bool] = {
        "finite_exact_value_and_score": finite,
        "analytical_score_central_fd": score_error <= SCORE_TOL,
        "complete_denominator_and_law_finite": bool(proposal["finite"]),
        "apf_w_over_a_identity": float(apf["max_identity_backward_error"]) <= float(p2.APF_IDENTITY_BACKWARD_BOUND),
        "apf_recomputed_final_weight_parity": float(apf["final_log_weight_parity_backward_error"]) <= float(p2.APF_IDENTITY_BACKWARD_BOUND),
        "auxiliary_rows_normalized": float(apf["max_auxiliary_normalization_abs_error"]) <= ABS_TOL,
        "nonjit_exact_program_parity": nonjit_value_error <= 2.0e-10 and nonjit_score_error <= 2.0e-10,
    }
    if check_xla:
        checks["xla_exact_program_parity"] = (
            xla_value_error is not None and xla_score_error is not None
            and xla_value_error <= 2.0e-10 and xla_score_error <= 2.0e-10
        )
    if label == "exact_likelihood_laplace_k1":
        residual = max(_max_abs(row["relative_stationarity_residual"]) for row in compilation.proposal_diagnostics)
        improvement = min(
            _scalar(tf.reduce_min(tf.convert_to_tensor(row["iteration_objective_after_step"], DTYPE) - tf.convert_to_tensor(row["iteration_objective_before_step"], DTYPE)))
            for row in compilation.proposal_diagnostics
        )
        checks.update({
            "all_laplace_rows_valid": all(bool(tf.convert_to_tensor(row["laplace_valid"]).numpy()) for row in compilation.proposal_diagnostics),
            "laplace_stationarity": residual <= float(compilation.manifest["laplace_schedule"]["stationarity_relative_tolerance"]),
            "laplace_objective_ascent": improvement >= -ASCENT_TOLERANCE,
            "complete_k1_density_recomposition": max(_scalar(row["proposal_density_recomposition_max_abs"]) for row in compilation.proposal_diagnostics) <= ABS_TOL,
        })
    weight_path = _exact_weight_path(p2, model, compilation.branch, theta)
    sidecars = _save_branch_tensors(output, label, compilation, weight_path)
    return {
        "label": label,
        "compiler_id": compilation.compiler_id,
        "branch_id": compilation.branch.branch_id,
        "program_id": program.program_id,
        "particle_count": compilation.branch.particle_count,
        "time_steps": compilation.branch.time_steps,
        "output_device": str(result["log_likelihood"].device),
        "log_likelihood": _scalar(result["log_likelihood"]),
        "score": _vector(result["score"]),
        "finite_difference_score": _vector(finite_difference),
        "score_max_abs_error": score_error,
        "minimum_ess": _scalar(result["minimum_ess"]),
        "ess_by_time": _vector(result["ess_by_time"]),
        "maximum_log_weight_spread": _scalar(result["maximum_log_weight_spread"]),
        "maximum_normalized_weight_by_time": _vector(result["maximum_normalized_weight_by_time"]),
        "proposal": _jsonable(proposal),
        "apf": _jsonable(apf),
        "parity": {
            "nonjit_value_max_abs": nonjit_value_error,
            "nonjit_score_max_abs": nonjit_score_error,
            "nonjit_error": nonjit_error,
            "xla_value_max_abs": xla_value_error,
            "xla_score_max_abs": xla_score_error,
            "xla_error": xla_error,
        },
        "checks": checks,
        "all_checks_pass": all(checks.values()),
        "tensor_sidecars": sidecars,
        "interpretation": "calibration/mechanism diagnostic; ESS is not a correctness or superiority claim",
    }


def _heuristic_table(records: Sequence[Mapping[str, object]]) -> Mapping[str, object]:
    # Preserve branch identity.  A label-only dictionary silently discarded
    # all but the last realization when branch_count > 1, which could hide a
    # heuristic loss in the promotion screen.
    by_branch: dict[int, dict[str, Mapping[str, object]]] = {}
    for row in records:
        branch = int(row.get("branch_index", 0))
        by_branch.setdefault(branch, {})[str(row["label"])] = row
    heuristics = (
        "bootstrap_conditional",
        "transformed_student_nu8",
        "gaussian_hint_marginal",
        "stationary_independence",
        "ukf_apf_k1",
    )
    candidates = ("exact_likelihood_laplace_k1",)
    table: dict[str, object] = {}
    for candidate in candidates:
        comparisons = []
        for branch_index, by_label in sorted(by_branch.items()):
            row = by_label.get(candidate)
            if row is None or "ess_by_time" not in row:
                continue
            candidate_ess = [float(x) for x in row["ess_by_time"]]
            for heuristic in heuristics:
                other = by_label.get(heuristic)
                if other is None or "ess_by_time" not in other:
                    continue
                other_ess = [float(x) for x in other["ess_by_time"]]
                for time_index, (value, baseline) in enumerate(zip(candidate_ess, other_ess)):
                    candidate_loses = value < baseline
                    proposal_active = time_index >= PROPOSAL_ACTIVE_START_TIME
                    comparisons.append({
                        "branch_index": branch_index,
                        "heuristic": heuristic,
                        "time_index": time_index,
                        "candidate_ess": value,
                        "heuristic_ess": baseline,
                        "candidate_minus_heuristic": value - baseline,
                        "candidate_loses": candidate_loses,
                        "proposal_active": proposal_active,
                        "screen_candidate_loses": proposal_active and candidate_loses,
                        "decision_role": "promotion_veto" if proposal_active else "explanatory_only",
                    })
        table[candidate] = {
            "proposal_active_start_time": PROPOSAL_ACTIVE_START_TIME,
            "verdict": "PROMOTION_VETO_HEURISTIC_LOSS" if any(bool(x["screen_candidate_loses"]) for x in comparisons) else "PASSES_WEAK_HEURISTIC_SCREEN_ONLY",
            "comparisons": comparisons,
        }
    return table


def _result_markdown(payload: Mapping[str, object]) -> str:
    def metric(value: object, digits: int = 6) -> str:
        try:
            numeric = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return "n/a"
        return "n/a" if not math.isfinite(numeric) else f"{numeric:.{digits}g}"

    checks = payload["checks"]
    phase_name = str(payload.get("phase", PHASE_ID))
    data_label = str(payload.get("data_label", "already-seen C2 path"))
    lines = [
        f"# C2 {phase_name} exact-likelihood Laplace result",
        "",
        f"Status: `{payload['status']}`",
        f"Continuation: `{payload['continuation']}`",
        f"Failure class: `{payload['failure_class']}`",
        "",
        f"This is a calibration/mechanism comparison on the {data_label}.",
        "It uses the exact frozen APF evaluator and complete proposal denominator.",
        "",
        "## Decision",
        "",
        "| Decision | Status | Interpretation |",
        "| --- | --- | --- |",
        f"| Schedule calibration | {'PASS' if checks['schedule_calibration_valid'] else 'VETO'} | selected by validity/residual/ascent/runtime before ESS |",
        f"| Exact target and denominator | {'PASS' if checks['all_branch_validity'] else 'VETO'} | all evaluated branches passed finite-program checks |",
        f"| Candidate branch | {'PASS' if checks.get('candidate_branch_validity', False) else 'VETO'} | a candidate veto is a repair trigger, not automatically a continuation veto |",
        f"| Comparator branches | {'PASS' if checks.get('comparator_branch_validity', False) else 'VETO'} | comparator validity is required to interpret the comparison |",
        f"| Continuation veto | {'YES' if payload.get('continuation_veto', False) else 'NO'} | target/measure, nonfinite-program, missing-record, or infrastructure failure |",
        f"| Time-14 mechanism repair | {'PASS' if payload['time14_repair_nomination'] else 'NOT SHOWN'} | candidate ESS at time 14 versus transformed UKF |",
        "| Promotion/default | VETOED | diagnostic phase; no untouched promotion claim |",
        "",
        "## Inference status",
        "",
        "| Evidence class | Status |",
        "| --- | --- |",
        "| Hard veto screen | " + ("passed" if checks["all_branch_validity"] else "failed") + " |",
        "| Statistically supported ranking | none; one branch per family |",
        "| Descriptive-only differences | all ESS, log-weight, and time-index contrasts |",
        "| Default readiness | not assessed |",
        "| Next evidence | fresh paired run only after the recorded cost/seed refresh |",
        "",
        "## Family summary",
        "",
        "| Family | valid | minimum ESS | ESS at t=14 | branch seconds | failure |",
        "| --- | :---: | ---: | ---: | ---: | --- |",
    ]
    for row in payload["records"]:
        ess = row.get("ess_by_time", [])
        ess14 = ess[14] if isinstance(ess, Sequence) and len(ess) > 14 else None
        error = str(row.get("error", "")).replace("|", "\\|") or "-"
        lines.append(
            f"| {row.get('label', 'unknown')} | {bool(row.get('all_checks_pass', False))} | "
            f"{metric(row.get('minimum_ess'))} | {metric(ess14)} | "
            f"{metric(row.get('branch_wall_seconds'), 3)} | {error} |"
        )
    lines += [
        "",
        "ESS is descriptive here and is not used to select the Newton schedule.",
        f"The heuristic promotion screen starts at transition time t={PROPOSAL_ACTIVE_START_TIME}; t=0 remains an explanatory initial-cloud diagnostic.",
        "The full per-time weights and Laplace proposal tensors are listed in the JSON sidecars.",
        "",
        "## Calibration",
        "",
        f"Selected schedule: `{payload['calibration']['selected']['config_id']}`",
        "",
        "| Config | eligible | worst residual | runtime (s) |",
        "| --- | :---: | ---: | ---: |",
    ]
    for row in payload["calibration"]["records"]:
        residual = row["worst_selection_residual"]
        residual_text = metric(residual)
        lines.append(
            f"| {row['config_id']} | {row['selection_valid']} | {residual_text} | {metric(row.get('wall_seconds'), 3)} |"
        )
    lines += [
        "",
        "The bimodal fixture is a fail-closed explanatory diagnostic; it is not a K=1 multimodal success claim.",
        "",
    ]
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--fixture-path", default=str(FIXTURE_PATH))
    parser.add_argument("--plan-path", default=str(PLAN_PATH))
    parser.add_argument("--phase-id", default=PHASE_ID)
    parser.add_argument("--data-label", default="already-seen C2 path")
    parser.add_argument(
        "--replication-set-id",
        default=None,
        help="stable identifier for the independent fixture/branch replication set",
    )
    parser.add_argument(
        "--analysis-seed",
        type=int,
        default=None,
        help="seed recorded for the downstream fixture-cluster analysis",
    )
    parser.add_argument("--exclude-gaussian-hint", action="store_true")
    parser.add_argument("--fixed-schedule-config", default=None)
    parser.add_argument("--particle-count", type=int, default=DEFAULT_PARTICLE_COUNT)
    parser.add_argument("--branch-count", type=int, default=1)
    parser.add_argument("--calibration-only", action="store_true")
    parser.add_argument(
        "--include-repair-schedules",
        action="store_true",
        help="include the reviewed Phase 8D fixed-schedule repair hypotheses",
    )
    parser.add_argument("--jit-compile", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def run(args: argparse.Namespace) -> Path:
    if not bool(args.jit_compile):
        raise ValueError("Phase 8B requires XLA")
    if int(args.particle_count) < 2 or int(args.branch_count) < 1:
        raise ValueError("particle-count and branch-count must be positive")
    fixture_path = Path(args.fixture_path)
    if not fixture_path.is_absolute():
        fixture_path = (ROOT / fixture_path).resolve()
    plan_path = Path(args.plan_path)
    if not plan_path.is_absolute():
        plan_path = (ROOT / plan_path).resolve()
    phase_id = str(args.phase_id)
    data_label = str(args.data_label)
    phase_token = _phase_token(phase_id)
    replication_set_id = (
        None if args.replication_set_id is None else str(args.replication_set_id)
    )
    analysis_seed = None if args.analysis_seed is None else int(args.analysis_seed)
    if phase_token == "PHASE8E" and (
        not replication_set_id or analysis_seed is None
    ):
        raise ValueError(
            "Phase 8E requires --replication-set-id and --analysis-seed"
        )
    if phase_token == "PHASE8E" and not bool(args.exclude_gaussian_hint):
        raise ValueError(
            "Phase 8E requires --exclude-gaussian-hint for the declared five-family ladder"
        )
    if not fixture_path.is_file():
        raise FileNotFoundError(f"missing fixture: {fixture_path}")
    if not plan_path.is_file():
        raise FileNotFoundError(f"missing plan: {plan_path}")
    output = _fresh_output(args.output_root)
    (output / "plan-at-launch.md").write_text(plan_path.read_text(encoding="utf-8"), encoding="utf-8")
    started = time.perf_counter()
    runtime = _configure_gpu()
    p2 = _load_module("c2_phase8b_p2_helpers", ROOT / "docs/benchmarks/run_c2_mixture_ukf_apf_20260902.py")
    p2._load_algorithm_modules()
    fixture = _load_fixture(fixture_path)
    model, theta, observations = _model_inputs(p2, fixture)
    replication_metadata = {
        "set_id": replication_set_id,
        "analysis_seed": analysis_seed,
        "model_seed": fixture.get("model_seed"),
        "observation_seed": fixture.get("observation_seed"),
        "branch_seed_formula": {
            "base": 98200,
            "particle_count_multiplier": 1,
            "branch_index_stride": 1009,
            "expression": "98200 + particle_count + 1009 * branch_index",
        },
    }
    schedule_ladder = (
        (*SCHEDULE_LADDER, *REPAIR_SCHEDULE_LADDER)
        if bool(args.include_repair_schedules)
        else SCHEDULE_LADDER
    )
    calibration = _calibrate_schedule(
        model=model, theta=theta, schedule_ladder=schedule_ladder
    )
    schedule_ladder_id = (
        "baseline_plus_reviewed_repair_hypotheses"
        if bool(args.include_repair_schedules)
        else "baseline"
    )
    schedule_selection_mode = "calibrated"
    if args.fixed_schedule_config is not None:
        requested = str(args.fixed_schedule_config)
        fixed_rows = [
            row for row in calibration["records"]
            if str(row["config_id"]) == requested and bool(row["selection_valid"])
        ]
        if len(fixed_rows) != 1:
            raise RuntimeError(
                f"requested fixed schedule {requested!r} did not pass independent calibration"
            )
        calibration["selected"] = fixed_rows[0]
        calibration["selection_mode"] = "fixed_requested_config_after_independent_validity_check"
        schedule_selection_mode = "fixed_requested_config_after_independent_validity_check"
    source_paths = {
        "plan": plan_path,
        "master_plan": MASTER_PLAN_PATH,
        "driver": DRIVER_PATH,
        "fixture": fixture_path,
        "laplace": LAPLACE_PATH,
        "laplace_adapter": LAPLACE_ADAPTER_PATH,
        "model": MODEL_PATH,
        "ukf_adapter": UKF_PATH,
        "exact_evaluator": EXACT_PATH,
    }
    source_manifest = {
        name: {"path": str(path.relative_to(ROOT)), "sha256": _sha256_file(path)}
        for name, path in source_paths.items()
    }
    calibration_observation_files = {
        "ordinary": _write_tensor(output / "calibration_ordinary_observations.tensor", calibration.pop("ordinary_observations")),
        "near_zero": _write_tensor(output / "calibration_near_zero_observations.tensor", calibration.pop("near_zero_observations")),
    }
    _write_json(output / "calibration.json", {**calibration, "observation_files": calibration_observation_files})
    if bool(args.calibration_only):
        checks = {
            "schedule_calibration_valid": bool(calibration.get("selected")),
            "gpu_memory_growth_verified": bool(runtime["memory_policy"]["all_physical_devices_memory_growth"]),
        }
        payload = {
            "schema_version": RESULT_SCHEMA,
            "phase": phase_id,
            "data_label": data_label,
            "replication": replication_metadata,
            "schedule_selection_mode": schedule_selection_mode,
            "schedule_ladder_id": schedule_ladder_id,
            "include_repair_schedules": bool(args.include_repair_schedules),
            "status": f"PASS_{phase_token}_CALIBRATION_ONLY" if all(checks.values()) else f"VETO_{phase_token}_CALIBRATION_ONLY",
            "continuation": f"CONTINUE_{phase_token}_COMPARISON" if all(checks.values()) else f"CONTINUATION_VETO_REPAIR_{phase_token}",
            "failure_class": "none" if all(checks.values()) else "numerical_validity",
            "calibration": calibration,
            "calibration_observation_files": calibration_observation_files,
            "checks": checks,
            "runtime": runtime,
            "sources": source_manifest,
            "workspace": {"git_commit": _git("rev-parse", "HEAD"), "git_branch": _git("branch", "--show-current"), "git_status": _git("status", "--porcelain=v1")},
            "nonclaims": ["no particle comparison", "no ESS conclusion", "no promotion or default claim"],
        }
        _write_json(output / "result.json", payload)
        _write_json(output / "manifest.json", {"schema_version": MANIFEST_SCHEMA, "phase": phase_id, "command": " ".join(sys.argv), "plan_sha256": _sha256_file(plan_path), "checks": checks, "sources": payload["sources"], "workspace": payload["workspace"], "runtime": runtime, "data_label": data_label, "replication": replication_metadata, "schedule_selection_mode": schedule_selection_mode, "schedule_ladder_id": schedule_ladder_id, "include_repair_schedules": bool(args.include_repair_schedules)})
        (output / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
        (output / "result.md").write_text("# C2 " + phase_id + " calibration-only result\n\nStatus: `" + str(payload["status"]) + "`\n\nNo particle comparison was run.\n", encoding="utf-8")
        if not all(checks.values()):
            raise RuntimeError(str(payload["status"]))
        return output
    from bayesfilter.highdim.c2_exact_likelihood_laplace_adapter import compile_c2_exact_likelihood_laplace_apf_k1
    from bayesfilter.highdim.c2_sv_frozen_proposal_apf_tf import (
        compile_c2_bootstrap_proposal_branch,
        compile_c2_independent_proposal_branch,
        compile_c2_transformed_student_proposal_branch,
        stationary_gaussian_proposals,
    )
    from bayesfilter.highdim.c2_mixture_ukf_apf_c2_adapter import compile_c2_per_ancestor_ukf_apf_k1
    from bayesfilter.highdim.exact_likelihood_laplace_apf_tf import FixedLaplaceConfig

    selected = calibration["selected"]
    laplace_config = FixedLaplaceConfig(
        tuple(selected["tempering_schedule"]), tuple(selected["step_fractions"]),
        0.0, STATIONARITY_TOLERANCE,
    )
    if bool(args.exclude_gaussian_hint):
        gaussian_hints, snapshot_hashes = None, {}
    else:
        gaussian_hints, snapshot_hashes = _load_gaussian_hint_proposals(HORIZON)
    stationary = stationary_gaussian_proposals(model, theta, HORIZON)
    particle_count = int(args.particle_count)
    branch_count = int(args.branch_count)
    records: list[Mapping[str, object]] = []
    for branch_index in range(branch_count):
        seed = 98200 + particle_count + 1009 * branch_index
        specs = [
            ("exact_likelihood_laplace_k1", lambda: compile_c2_exact_likelihood_laplace_apf_k1(model=model, observations=observations, theta_reference=theta, particle_count=particle_count, seed=seed, laplace_config=laplace_config, jit_compile=True), True),
            ("ukf_apf_k1", lambda: compile_c2_per_ancestor_ukf_apf_k1(model=model, observations=observations, theta_reference=theta, particle_count=particle_count, seed=seed, jit_compile=True), True),
            ("bootstrap_conditional", lambda: compile_c2_bootstrap_proposal_branch(model=model, observations=observations, theta_reference=theta, particle_count=particle_count, seed=seed, jit_compile_sampler=True), False),
            ("transformed_student_nu8", lambda: compile_c2_transformed_student_proposal_branch(model=model, observations=observations, theta_reference=theta, nu=8.0, particle_count=particle_count, seed=seed, jit_compile_sampler=True), False),
            ("stationary_independence", lambda: compile_c2_independent_proposal_branch(model=model, observations=observations, theta_reference=theta, transition_proposals=stationary, particle_count=particle_count, seed=seed, family="stationary_independence", jit_compile_sampler=True), False),
        ]
        if gaussian_hints is not None:
            specs.insert(
                4,
                ("gaussian_hint_marginal", lambda: compile_c2_independent_proposal_branch(model=model, observations=observations, theta_reference=theta, transition_proposals=gaussian_hints, particle_count=particle_count, seed=seed, family="gaussian_hint_marginal", jit_compile_sampler=True), False),
            )
        for label, compiler, check_xla in specs:
            branch_started = time.perf_counter()
            record: dict[str, object] = {"label": label, "branch_index": branch_index, "seed": seed, "particle_count": particle_count}
            try:
                compilation = compiler()
                evaluated = dict(_evaluate_branch(p2=p2, label=label, compilation=compilation, model=model, theta=theta, output=output, check_xla=check_xla))
                evaluated["branch_wall_seconds"] = time.perf_counter() - branch_started
                evaluated["compiler_manifest"] = compilation.manifest
                record.update(_jsonable(evaluated))
            except Exception as exc:
                record.update({
                    "all_checks_pass": False,
                    "failure_class": "candidate_or_comparator_failure",
                    "error": f"{type(exc).__name__}: {exc}",
                    "branch_wall_seconds": time.perf_counter() - branch_started,
                })
            records.append(record)
            _write_json(output / "branch_records.partial.json", records)
    grouped: dict[str, list[Mapping[str, object]]] = {}
    for row in records:
        grouped.setdefault(str(row["label"]), []).append(row)
    summary = {
        label: {
            "record_count": len(rows),
            "valid_count": sum(bool(row.get("all_checks_pass", False)) for row in rows),
            "minimum_ess_mean": sum(float(row["minimum_ess"]) for row in rows if "minimum_ess" in row) / max(1, sum("minimum_ess" in row for row in rows)),
            "ess_by_time_mean": [
                sum(float(row["ess_by_time"][i]) for row in rows if "ess_by_time" in row) / max(1, sum("ess_by_time" in row for row in rows if "ess_by_time" in row))
                for i in range(HORIZON)
            ],
        }
        for label, rows in sorted(grouped.items())
    }
    heuristic = _heuristic_table(records)
    family_count = 6 if gaussian_hints is not None else 5
    record_summary = _record_validity_summary(
        records, branch_count=branch_count, family_count=family_count
    )
    all_branch_validity = bool(record_summary["all_branch_validity"])
    laplace_rows = [row for row in records if row.get("label") == "exact_likelihood_laplace_k1" and "ess_by_time" in row]
    ukf_rows = [row for row in records if row.get("label") == "ukf_apf_k1" and "ess_by_time" in row]
    time14_repair_nomination = bool(laplace_rows and ukf_rows and float(laplace_rows[0]["ess_by_time"][14]) > float(ukf_rows[0]["ess_by_time"][14]))
    checks = {
        "schedule_calibration_valid": bool(calibration["selected"]),
        "all_branch_validity": all_branch_validity,
        "records_complete": bool(record_summary["records_complete"]),
        "candidate_branch_validity": bool(record_summary["candidate_branch_validity"]),
        "comparator_branch_validity": bool(record_summary["comparator_branch_validity"]),
        "source_files_present": all(path.is_file() for path in (*source_paths.values(),)),
        "snapshot_bank_complete": bool(args.exclude_gaussian_hint) or len(snapshot_hashes) == HORIZON - 1,
        "gpu_memory_growth_verified": bool(runtime["memory_policy"]["all_physical_devices_memory_growth"]),
    }
    checks["hard_vetoes_pass"] = all(bool(value) for value in checks.values())
    continuation_veto = bool(
        record_summary["continuation_veto"]
        or not checks["schedule_calibration_valid"]
        or not checks["source_files_present"]
        or not checks["snapshot_bank_complete"]
        or not checks["gpu_memory_growth_verified"]
    )
    checks["continuation_veto_pass"] = not continuation_veto
    any_loss = heuristic["exact_likelihood_laplace_k1"]["verdict"] == "PROMOTION_VETO_HEURISTIC_LOSS"
    candidate_failure = bool(any_loss or record_summary["candidate_failure"])
    phase_token = _phase_token(phase_id)
    if continuation_veto:
        status = f"VETO_{phase_token}_VALIDITY"
        continuation = _continuation_token(phase_token, False)
    elif record_summary["candidate_failure"]:
        status = f"VETO_{phase_token}_CANDIDATE_VALIDITY"
        continuation = f"CONTINUE_{phase_token}_REPAIR_AFTER_CANDIDATE_FAILURE"
    elif checks["hard_vetoes_pass"] and any_loss:
        status = f"PASS_{phase_token}_VALIDITY_WITH_PROMOTION_VETO"
        continuation = _continuation_token(phase_token, True)
    else:
        status = f"PASS_{phase_token}_VALIDITY_DIAGNOSTIC"
        continuation = _continuation_token(phase_token, True)
    payload: Mapping[str, object] = {
        "schema_version": RESULT_SCHEMA,
        "phase": phase_id,
        "data_label": data_label,
        "replication": replication_metadata,
        "schedule_selection_mode": schedule_selection_mode,
        "schedule_ladder_id": schedule_ladder_id,
        "include_repair_schedules": bool(args.include_repair_schedules),
        "excluded_families": ["gaussian_hint_marginal"] if bool(args.exclude_gaussian_hint) else [],
        "status": status,
        "continuation": continuation,
        "failure_class": (
            "implementation_or_numerical_validity"
            if continuation_veto
            else ("candidate_failure" if candidate_failure else "none")
        ),
        "candidate_failure": candidate_failure,
        "continuation_veto": continuation_veto,
        "record_validity_summary": record_summary,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "route_id": ROUTE_ID,
        "route_classification": ROUTE_CLASSIFICATION,
        "particle_count": particle_count,
        "branch_count": branch_count,
        "horizon": HORIZON,
        "selected_schedule": selected,
        "calibration": {key: value for key, value in calibration.items() if key not in ("ordinary_observations", "near_zero_observations")},
        "calibration_observation_files": calibration_observation_files,
        "records": records,
        "family_summary": summary,
        "heuristic_dominance": heuristic,
        "proposal_active_start_time": PROPOSAL_ACTIVE_START_TIME,
        "time14_repair_nomination": time14_repair_nomination,
        "snapshot_metadata_sha256": snapshot_hashes,
        "checks": checks,
        "runtime": runtime,
        "environment": {"python": platform.python_version(), "tensorflow": tf.__version__, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unset"), "jit_compile": True},
        "sources": source_manifest,
        "workspace": {"git_commit": _git("rev-parse", "HEAD"), "git_branch": _git("branch", "--show-current"), "git_status": _git("status", "--porcelain=v1")},
        "evidence_contract": {"question": "whether exact likelihood local geometry repairs the transformation-tail mechanism on the declared data path", "primary": "validity plus calibration-selected fixed schedule; ESS is descriptive mechanism evidence", "promotion_veto": "candidate loss to a constructed cheap heuristic at an active transition time (t >= 1); t=0 is explanatory only because no transition proposal has been used", "continuation_veto": "target/denominator mismatch, nonfinite branch, missing records, or GPU provenance failure", "nonclaims": ["no posterior correctness", "no unbiased likelihood", "no statistical ranking", "no general-model or default claim"]},
    }
    manifest = {"schema_version": MANIFEST_SCHEMA, "phase": phase_id, "command": " ".join(sys.argv), "plan_sha256": _sha256_file(plan_path), "sources": payload["sources"], "workspace": payload["workspace"], "runtime": runtime, "checks": checks, "budget": {"gpu_hour_cap": 1, "branch_count": branch_count, "localized_repairs": 2}, "data_label": data_label, "replication": replication_metadata, "excluded_families": payload["excluded_families"], "schedule_selection_mode": schedule_selection_mode, "schedule_ladder_id": schedule_ladder_id, "include_repair_schedules": bool(args.include_repair_schedules), "proposal_active_start_time": PROPOSAL_ACTIVE_START_TIME}
    _write_json(output / "manifest.json", manifest)
    _write_json(output / "branch_records.json", records)
    _write_json(output / "result.json", payload)
    (output / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    (output / "result.md").write_text(_result_markdown(payload), encoding="utf-8")
    if not checks["hard_vetoes_pass"]:
        raise RuntimeError(status)
    return output


def main() -> int:
    args = _parse_args()
    output = run(args)
    print(json.dumps({"phase": str(args.phase_id), "output_root": str(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
