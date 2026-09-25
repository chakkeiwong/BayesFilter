"""Run the bounded Phase 5 recursive lagged-moment-map diagnostic.

This driver has two deliberately separate arms.  The two-dimensional arm is
an executable law-of-total-covariance/Kalman oracle.  The C2 arm binds the
same model-independent map to the exact C2 transition and observation
densities for a small, deterministic cloud.  Neither arm fits a TT density or
claims an importance-sampling or posterior result.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

# TensorFlow on this build can create logical GPUs when the growth environment
# variable is present at import time.  Defer and restore it around the
# repository-owned memory-policy helper, as the other C2 runners do.
_DEFERRED_TF_FORCE_GPU_ALLOW_GROWTH = os.environ.pop(
    "TF_FORCE_GPU_ALLOW_GROWTH", None
)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import tensorflow as tf

DTYPE = tf.float64
PHASE_ID = "c2_mixture_ukf_apf_phase5_recursive_map_v1"
RESULT_SCHEMA = "c2_mixture_ukf_apf_phase5_recursive_map_result_v1"
MANIFEST_SCHEMA = "c2_mixture_ukf_apf_phase5_recursive_map_manifest_v1"
ROUTE_ID = "recursive_lagged_transition_moment_map_v1"
MAP_CLASSIFICATION = "extension_or_invention_candidate_diagnostic_only"
SEED = (20260904, 501)
LINEAR_TOLERANCE = 2.0e-12
ROUNDTRIP_TOLERANCE = 2.0e-12
FINITE_CLOUD_ROWS = 128

LINEAR_FIXTURE_PATH = ROOT / "docs/benchmarks/fixtures/c2_mixture_ukf_lgssm_phase0_v1.json"
C2_FIXTURE_PATH = ROOT / "docs/benchmarks/fixtures/c2_sv_n4_seed52_obs42_t20_frozen_v1.json"
PLAN_PATH = ROOT / "docs/plans/c2-mixture-ukf-apf-phase5-recursive-map-20260904.md"
MOMENT_MAP_PATH = ROOT / "bayesfilter/highdim/recursive_moment_map_tf.py"
C2_MODEL_PATH = ROOT / "bayesfilter/highdim/c2_sv_frozen_proposal_apf_tf.py"

_MODULES_LOADED = False


def _load_modules() -> None:
    global _MODULES_LOADED
    global AffineCoordinateMap, C2StochasticVolatilityFrozenAPFModel
    global build_lagged_moment_map, make_weighted_transition_moment_kernel
    global affine_forward_inverse_residual
    if _MODULES_LOADED:
        return
    # Imports of bayesfilter.highdim are delayed until device policy setup.
    from bayesfilter.highdim.filtering import AffineCoordinateMap as _AffineCoordinateMap
    from bayesfilter.highdim.recursive_moment_map_tf import (
        affine_forward_inverse_residual as _affine_forward_inverse_residual,
        build_lagged_moment_map as _build_lagged_moment_map,
        make_weighted_transition_moment_kernel as _make_weighted_transition_moment_kernel,
    )
    from bayesfilter.highdim.c2_sv_frozen_proposal_apf_tf import (
        C2StochasticVolatilityFrozenAPFModel as _C2StochasticVolatilityFrozenAPFModel,
    )
    AffineCoordinateMap = _AffineCoordinateMap
    C2StochasticVolatilityFrozenAPFModel = _C2StochasticVolatilityFrozenAPFModel
    build_lagged_moment_map = _build_lagged_moment_map
    make_weighted_transition_moment_kernel = _make_weighted_transition_moment_kernel
    affine_forward_inverse_residual = _affine_forward_inverse_residual
    _MODULES_LOADED = True


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


def _workspace_manifest() -> Mapping[str, object]:
    status = _git("status", "--porcelain=v1")
    return {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_status": status,
        "git_status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
    }


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


def _tensor(value: object) -> tf.Tensor:
    return tf.convert_to_tensor(value, dtype=DTYPE)


def _finite(value: object) -> bool:
    return bool(tf.reduce_all(tf.math.is_finite(tf.convert_to_tensor(value, DTYPE))).numpy())


def _scalar(value: object) -> float:
    tensor = tf.convert_to_tensor(value, DTYPE)
    if not _finite(tensor):
        raise ValueError("attempted to serialize a non-finite scalar")
    return float(tensor.numpy())


def _max_abs(value: object) -> float:
    tensor = tf.convert_to_tensor(value, DTYPE)
    if not _finite(tensor):
        raise ValueError("non-finite diagnostic")
    return _scalar(tf.reduce_max(tf.abs(tensor)))


def _load_fixture(path: Path, schema: str) -> Mapping[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    actual = payload.get("schema_version", payload.get("schema_id"))
    if actual != schema:
        raise ValueError(f"unexpected fixture schema for {path}: {actual!r}")
    return payload


def _configure_runtime() -> Mapping[str, object]:
    if _DEFERRED_TF_FORCE_GPU_ALLOW_GROWTH is not None:
        os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = _DEFERRED_TF_FORCE_GPU_ALLOW_GROWTH
    physical = tuple(tf.config.list_physical_devices("GPU"))
    if physical:
        from bayesfilter.runtime.gpu_memory_policy import (
            configure_tensorflow_gpu_memory_growth,
        )

        policy = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
        logical = tuple(tf.config.list_logical_devices("GPU"))
        if not logical:
            raise RuntimeError("visible physical GPU produced no logical GPU")
        with tf.device("/GPU:0"):
            probe = tf.reduce_sum(tf.ones([32], DTYPE))
        if "GPU" not in str(probe.device).upper():
            raise RuntimeError(f"GPU placement probe ran on {probe.device}")
        return {
            "execution_lane": "trusted_gpu_xla_phase5_recursive_map",
            "memory_policy": policy,
            "physical_devices": [str(device.name) for device in physical],
            "logical_devices": [str(device.name) for device in logical],
            "placement_probe_device": str(probe.device),
            "placement_probe_value": _scalar(probe),
        }
    return {
        "execution_lane": "cpu_reference_or_debug",
        "memory_policy": {
            "schema": "bayesfilter.tensorflow.gpu_memory_policy.v1",
            "mode": "cpu_no_visible_gpu",
            "configured_before_logical_device_initialization": True,
        },
        "physical_devices": [str(device.name) for device in tf.config.list_physical_devices()],
        "logical_devices": [str(device.name) for device in tf.config.list_logical_devices()],
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--rows", type=int, default=FINITE_CLOUD_ROWS)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument(
        "--jit-compile", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


def _make_output_root(value: str) -> Path:
    candidate = Path(value)
    output = (ROOT / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {output}")
    output.mkdir(parents=True, exist_ok=False)
    return output


def _symmetric_sigma_cloud(mean: tf.Tensor, covariance: tf.Tensor) -> tf.Tensor:
    """Return four nonnegative-weight points with exactly the supplied 2-D moments."""

    chol = tf.linalg.cholesky(covariance)
    directions = tf.sqrt(tf.constant(2.0, DTYPE)) * tf.transpose(chol)
    return tf.concat(
        [mean[None, :] + directions, mean[None, :] - directions], axis=0
    )


def _weighted_moments(weights: tf.Tensor, points: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    mean = tf.einsum("n,nd->d", weights, points)
    centered = points - mean[None, :]
    covariance = tf.einsum("n,ni,nj->ij", weights, centered, centered)
    covariance = 0.5 * (covariance + tf.transpose(covariance))
    return mean, covariance


def _standardized_moments_clean(
    coordinate_map: Any, mean: tf.Tensor, covariance: tf.Tensor
) -> tuple[tf.Tensor, tf.Tensor]:
    """Compute moments after T(x)=L^{-1}(x-m) without row-orientation ambiguity."""

    dimension = int(mean.shape[0])
    matrix = coordinate_map.matrix
    standardized_mean = tf.linalg.solve(matrix, (mean - coordinate_map.offset)[:, None])[:, 0]
    left = tf.linalg.solve(matrix, covariance)
    inverse_transpose = tf.linalg.matrix_transpose(
        tf.linalg.solve(matrix, tf.eye(dimension, dtype=DTYPE))
    )
    standardized_covariance = tf.matmul(left, inverse_transpose)
    return standardized_mean, 0.5 * (
        standardized_covariance + tf.transpose(standardized_covariance)
    )


def _map_metrics(
    coordinate_map: Any,
    predicted_mean: tf.Tensor,
    predicted_covariance: tf.Tensor,
    reference_points: tf.Tensor,
) -> Mapping[str, object]:
    standardized_mean, standardized_covariance = _standardized_moments_clean(
        coordinate_map, predicted_mean, predicted_covariance
    )
    roundtrip = affine_forward_inverse_residual(coordinate_map, reference_points)
    identity = tf.eye(int(predicted_mean.shape[0]), dtype=DTYPE)
    singular_values = tf.linalg.svd(coordinate_map.matrix, compute_uv=False)
    return {
        "standardized_mean_max_abs": _max_abs(standardized_mean),
        "standardized_covariance_identity_max_abs": _max_abs(
            standardized_covariance - identity
        ),
        "roundtrip_max_abs": _scalar(roundtrip["forward_inverse_max_abs"]),
        "physical_finite": bool(roundtrip["physical_finite"].numpy()),
        "reference_finite": bool(roundtrip["reference_finite"].numpy()),
        "minimum_cholesky_diagonal": _scalar(
            tf.reduce_min(tf.linalg.diag_part(coordinate_map.matrix))
        ),
        "condition_number": _scalar(
            tf.reduce_max(singular_values) / tf.reduce_min(singular_values)
        ),
    }


def _map_shift(previous: Any | None, current: Any) -> Mapping[str, object]:
    if previous is None:
        return {"offset_max_abs": 0.0, "matrix_max_abs": 0.0, "max_abs": 0.0}
    offset = _max_abs(current.offset - previous.offset)
    matrix = _max_abs(current.matrix - previous.matrix)
    return {"offset_max_abs": offset, "matrix_max_abs": matrix, "max_abs": max(offset, matrix)}


def _kalman_update(
    mean: tf.Tensor,
    covariance: tf.Tensor,
    transition_matrix: tf.Tensor,
    transition_offset: tf.Tensor,
    process_covariance: tf.Tensor,
    observation_matrix: tf.Tensor,
    observation_offset: tf.Tensor,
    observation_covariance: tf.Tensor,
    observation: tf.Tensor,
) -> Mapping[str, tf.Tensor]:
    predicted_mean = tf.linalg.matvec(transition_matrix, mean) + transition_offset
    predicted_covariance = (
        transition_matrix @ covariance @ tf.transpose(transition_matrix)
        + process_covariance
    )
    innovation = observation - (
        tf.linalg.matvec(observation_matrix, predicted_mean) + observation_offset
    )
    innovation_covariance = (
        observation_matrix @ predicted_covariance @ tf.transpose(observation_matrix)
        + observation_covariance
    )
    cross = predicted_covariance @ tf.transpose(observation_matrix)
    gain = tf.transpose(
        tf.linalg.solve(innovation_covariance, tf.transpose(cross))
    )
    posterior_mean = predicted_mean + tf.linalg.matvec(gain, innovation)
    posterior_covariance = predicted_covariance - gain @ innovation_covariance @ tf.transpose(gain)
    posterior_covariance = 0.5 * (
        posterior_covariance + tf.transpose(posterior_covariance)
    )
    return {
        "predicted_mean": predicted_mean,
        "predicted_covariance": predicted_covariance,
        "posterior_mean": posterior_mean,
        "posterior_covariance": posterior_covariance,
        "innovation": innovation,
        "innovation_covariance": innovation_covariance,
    }


def _linear_oracle(
    fixture: Mapping[str, object], *, horizon: int, jit_compile: bool
) -> Mapping[str, object]:
    state_dim = int(fixture["state_dim"])
    if state_dim != 2:
        raise ValueError("Phase 5 linear oracle expects the 2-D fixture")
    transition = _tensor(fixture["transition_matrix"])
    transition_offset = _tensor(fixture["transition_offset"])
    process = _tensor(fixture["process_covariance"])
    observation_matrix = _tensor(fixture["observation_matrix"])
    observation_offset = _tensor(fixture["observation_offset"])
    observation_covariance = _tensor(fixture["observation_covariance"])
    mean = _tensor(fixture["initial_mean"])
    covariance = _tensor(fixture["initial_covariance"])
    weights = tf.fill([4], tf.constant(0.25, DTYPE))
    reference_points = _tensor([[0.0, 0.0], [1.0, -0.5], [-0.7, 0.2], [0.3, 0.8]])
    observations = _tensor(fixture["observation"])
    records: list[Mapping[str, object]] = []
    moment_kernel = make_weighted_transition_moment_kernel(
        particle_count=4, state_dim=2, jit_compile=bool(jit_compile)
    )
    first_map = None
    previous_map = None
    max_mean_error = 0.0
    max_covariance_error = 0.0
    max_roundtrip_error = 0.0
    for time_index in range(1, int(horizon) + 1):
        cloud = _symmetric_sigma_cloud(mean, covariance)
        conditional_means = tf.linalg.matmul(cloud, transition, transpose_b=True) + transition_offset
        conditional_covariances = tf.broadcast_to(process[None, :, :], [4, 2, 2])
        built = build_lagged_moment_map(
            weights,
            conditional_means,
            conditional_covariances,
            jit_compile=bool(jit_compile),
            kernel=moment_kernel,
        )
        coordinate_map = built["coordinate_map"]
        expected_mean = tf.linalg.matvec(transition, mean) + transition_offset
        expected_covariance = transition @ covariance @ tf.transpose(transition) + process
        mean_error = _max_abs(built["predicted_mean"] - expected_mean)
        covariance_error = _max_abs(built["predicted_covariance"] - expected_covariance)
        map_metrics = _map_metrics(coordinate_map, expected_mean, expected_covariance, reference_points)
        shift = _map_shift(previous_map, coordinate_map)
        if first_map is None:
            first_map = coordinate_map
        static_metrics = _map_metrics(first_map, expected_mean, expected_covariance, reference_points)
        roundtrip_error = float(map_metrics["roundtrip_max_abs"])
        max_mean_error = max(max_mean_error, mean_error)
        max_covariance_error = max(max_covariance_error, covariance_error)
        max_roundtrip_error = max(max_roundtrip_error, roundtrip_error)
        records.append(
            {
                "time_index": time_index,
                "conditional_count": 4,
                "mean_error_max_abs": mean_error,
                "covariance_error_max_abs": covariance_error,
                "lagged_map": map_metrics,
                "static_map": static_metrics,
                "map_shift": shift,
                "valid": bool(built["valid"].numpy()),
                "raw_minimum_eigenvalue": _scalar(built["raw_minimum_eigenvalue"]),
                "observation": observations + _tensor([0.1 * time_index, -0.05 * time_index]),
            }
        )
        update = _kalman_update(
            mean,
            covariance,
            transition,
            transition_offset,
            process,
            observation_matrix,
            observation_offset,
            observation_covariance,
            observations + _tensor([0.1 * time_index, -0.05 * time_index]),
        )
        mean = update["posterior_mean"]
        covariance = update["posterior_covariance"]
        previous_map = coordinate_map
    all_pass = (
        max_mean_error <= LINEAR_TOLERANCE
        and max_covariance_error <= LINEAR_TOLERANCE
        and max_roundtrip_error <= ROUNDTRIP_TOLERANCE
        and all(bool(row["valid"]) for row in records)
        and all(bool(row["lagged_map"]["physical_finite"]) for row in records)
    )
    return {
        "arm": "linear_kalman_moment_oracle",
        "horizon": int(horizon),
        "records": records,
        "max_mean_error": max_mean_error,
        "max_covariance_error": max_covariance_error,
        "max_roundtrip_error": max_roundtrip_error,
        "pass": all_pass,
        "observation_offset_rule": "fixture_observation + [0.1*t, -0.05*t]",
    }


def _effective_sample_size(weights: tf.Tensor) -> tf.Tensor:
    return tf.math.reciprocal(tf.reduce_sum(tf.square(weights)))


def _c2_probe(
    fixture: Mapping[str, object], *, rows: int, horizon: int, jit_compile: bool
) -> Mapping[str, object]:
    if rows < 2:
        raise ValueError("C2 probe rows must be at least two")
    dimension = int(fixture["state_dimension"])
    theta = _tensor([float(fixture["gamma"]), math.log(float(fixture["beta"]))])
    transition = _tensor(fixture["transition_matrix"])
    process = _tensor(fixture["process_covariance"])
    model = C2StochasticVolatilityFrozenAPFModel(
        coupling_matrix=transition - theta[0] * tf.eye(dimension, dtype=DTYPE),
        sigma=float(fixture["sigma"]),
    )
    model_transition = model.transition_matrix(theta)
    transition_error = _max_abs(model_transition - transition)
    model_process = tf.eye(dimension, dtype=DTYPE) * float(fixture["sigma"]) ** 2
    process_error = _max_abs(model_process - process)
    transition = model_transition
    observations = _tensor(fixture["observations"][: int(horizon)])
    stationary = _tensor(fixture["stationary_covariance"])
    stationary_from_model, _ = model.stationary_covariance_and_derivative(theta)
    stationary_error = _max_abs(stationary_from_model - stationary)
    initial_chol = tf.linalg.cholesky(stationary)
    normal_bank = tf.random.stateless_normal([rows, dimension], list(SEED), dtype=DTYPE)
    states = tf.linalg.matmul(normal_bank, initial_chol, transpose_b=True)
    weights = tf.fill([rows], tf.constant(1.0 / rows, DTYPE))
    moment_kernel = make_weighted_transition_moment_kernel(
        particle_count=rows, state_dim=dimension, jit_compile=bool(jit_compile)
    )
    # Consume the first observation before the first lagged transition map.
    initial_prior_mean, _ = _weighted_moments(weights, states)
    initial_log_likelihood = model.observation_log_density(theta, states, observations[0], 0)
    initial_log_weights = initial_log_likelihood - tf.reduce_logsumexp(initial_log_likelihood)
    weights = tf.exp(initial_log_weights)
    initial_mean, _ = _weighted_moments(weights, states)
    records: list[Mapping[str, object]] = [
        {
            "time_index": 0,
            "map_built": False,
            "observation_ess": _scalar(_effective_sample_size(weights)),
            "observation_mean_shift_max_abs": _max_abs(initial_mean - initial_prior_mean),
            "finite": _finite(states) and _finite(weights),
        }
    ]
    first_map = None
    previous_map = None
    all_valid = bool(records[0]["finite"])
    for time_index in range(1, int(horizon)):
        conditional_means = tf.linalg.matmul(states, transition, transpose_b=True)
        conditional_covariances = tf.broadcast_to(
            process[None, :, :], [rows, dimension, dimension]
        )
        built = build_lagged_moment_map(
            weights,
            conditional_means,
            conditional_covariances,
            jit_compile=bool(jit_compile),
            kernel=moment_kernel,
        )
        coordinate_map = built["coordinate_map"]
        if first_map is None:
            first_map = coordinate_map
        predicted_states, _ = coordinate_map.forward(normal_bank)
        uniform = tf.fill([rows], tf.constant(1.0 / rows, DTYPE))
        empirical_mean, empirical_covariance = _weighted_moments(uniform, predicted_states)
        lagged_metrics = _map_metrics(
            coordinate_map, empirical_mean, empirical_covariance, normal_bank[: min(rows, 8)]
        )
        static_metrics = _map_metrics(
            first_map, empirical_mean, empirical_covariance, normal_bank[: min(rows, 8)]
        )
        log_observation = model.observation_log_density(
            theta, predicted_states, observations[time_index], time_index
        )
        log_weights = log_observation - tf.reduce_logsumexp(log_observation)
        posterior_weights = tf.exp(log_weights)
        posterior_mean, posterior_covariance = _weighted_moments(
            posterior_weights, predicted_states
        )
        observation_shift = _max_abs(posterior_mean - empirical_mean)
        finite = (
            bool(built["valid"].numpy())
            and _finite(predicted_states)
            and _finite(log_observation)
            and _finite(posterior_weights)
            and _finite(posterior_covariance)
        )
        row = {
            "time_index": time_index,
            "map_built": True,
            "map_valid": bool(built["valid"].numpy()),
            "raw_minimum_eigenvalue": _scalar(built["raw_minimum_eigenvalue"]),
            "minimum_eigenvalue": _scalar(built["minimum_eigenvalue"]),
            "condition_number": _scalar(built["condition_number"]),
            "lagged_map": lagged_metrics,
            "static_map": static_metrics,
            "map_shift": _map_shift(previous_map, coordinate_map),
            "observation_ess": _scalar(_effective_sample_size(posterior_weights)),
            "observation_mean_shift_max_abs": observation_shift,
            "predicted_mean": empirical_mean,
            "posterior_mean": posterior_mean,
            "finite": finite,
        }
        records.append(row)
        all_valid = all_valid and finite
        states = predicted_states
        weights = posterior_weights
        previous_map = coordinate_map
    return {
        "arm": "c2_exact_observation_recursive_cloud_probe",
        "rows": int(rows),
        "horizon": int(horizon),
        "records": records,
        "stationary_covariance_fixture_error_max_abs": stationary_error,
        "transition_fixture_error_max_abs": transition_error,
        "process_covariance_fixture_error_max_abs": process_error,
        "model_contract_pass": transition_error <= 1.0e-12 and process_error <= 1.0e-12 and stationary_error <= 1.0e-12,
        "all_rows_valid": all_valid,
        "observation_density_route": "C2StochasticVolatilityFrozenAPFModel.observation_log_density",
        "transition_moment_route": "C2StochasticVolatilityFrozenAPFModel.transition_matrix + exact process covariance",
    }


def _environment(runtime: Mapping[str, object], args: argparse.Namespace) -> Mapping[str, object]:
    return {
        "python": platform.python_version(),
        "tensorflow": tf.__version__,
        "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER", "unset"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
        "tf_force_gpu_allow_growth": os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH", "unset"),
        "jit_compile": bool(args.jit_compile),
        "runtime": runtime,
    }


def _result_markdown(payload: Mapping[str, object]) -> str:
    linear = payload["linear_oracle"]
    c2 = payload["c2_probe"]
    lines = [
        "# C2 Mixture-UKF/APF Phase 5 Result",
        "",
        f"Status: `{payload['status']}`  ",
        f"Continuation: `{payload['continuation']}`  ",
        f"Failure class: `{payload['failure_class']}`",
        "",
        "## Decision",
        "",
        "| Decision | Primary criterion | Status | Interpretation |",
        "| --- | --- | --- | --- |",
        f"| Generic map mechanics | linear mean/covariance and round-trip tolerances | {'PASS' if linear['pass'] else 'VETO'} | {'callable matches the linear oracle' if linear['pass'] else 'repair generic map before continuation'} |",
        f"| C2 wiring | finite/SPD rows and exact observation call chain | {'PASS' if c2['all_rows_valid'] else 'VETO'} | {'descriptive integration evidence only' if c2['all_rows_valid'] else 'adapter/map validity failure'} |",
        "| Proposal/filter promotion | no criterion in this phase | NOT TESTED | no ESS or posterior claim |",
        "",
        "## Inference Status",
        "",
        "| Evidence class | Status |",
        "| --- | --- |",
        "| Hard veto screen | " + ("passed" if payload["checks"]["hard_vetoes_pass"] else "failed") + " |",
        "| Statistically supported ranking | not applicable |",
        "| Descriptive differences | C2 map shifts, standardized residuals, and ESS are descriptive |",
        "| Default readiness | not assessed |",
        "| Next evidence | fixed-map representation ladder on disjoint rows, after phase close |",
        "",
        "## Linear Oracle",
        "",
        f"Maximum mean error: `{linear['max_mean_error']:.6g}`; maximum covariance error: `{linear['max_covariance_error']:.6g}`; maximum round-trip error: `{linear['max_roundtrip_error']:.6g}`.",
        "",
        "| t | mean error | covariance error | lagged standardized mean | lagged covariance identity error | condition |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in linear["records"]:
        lines.append(
            f"| {row['time_index']} | {row['mean_error_max_abs']:.3g} | {row['covariance_error_max_abs']:.3g} | {row['lagged_map']['standardized_mean_max_abs']:.3g} | {row['lagged_map']['standardized_covariance_identity_max_abs']:.3g} | {row['lagged_map']['condition_number']:.3g} |"
        )
    lines += [
        "",
        "## C2 Probe",
        "",
        f"Rows: `{c2['rows']}`; horizon: `{c2['horizon']}`; model/fixture transition error: `{c2['transition_fixture_error_max_abs']:.6g}`; process covariance error: `{c2['process_covariance_fixture_error_max_abs']:.6g}`; stationary covariance parity: `{c2['stationary_covariance_fixture_error_max_abs']:.6g}`.",
        "",
        "| t | map valid | ESS after observation | observation mean shift | lagged standardized covariance error | static standardized covariance error | map shift |",
        "| ---: | :---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in c2["records"]:
        if not row.get("map_built"):
            lines.append(f"| 0 | n/a | {row['observation_ess']:.3g} | {row['observation_mean_shift_max_abs']:.3g} | n/a | n/a | n/a |")
        else:
            lines.append(
                f"| {row['time_index']} | {str(row['map_valid'])} | {row['observation_ess']:.3g} | {row['observation_mean_shift_max_abs']:.3g} | {row['lagged_map']['standardized_covariance_identity_max_abs']:.3g} | {row['static_map']['standardized_covariance_identity_max_abs']:.3g} | {row['map_shift']['max_abs']:.3g} |"
            )
    lines += [
        "",
        "The C2 rows use the exact model observation log density.  ESS and map residuals are explanatory diagnostics, not promotion evidence.",
        "",
        "## Red Team",
        "",
        f"Strongest alternative explanation: the fixed finite normal bank may make the C2 empirical cloud noisy even when the map is algebraically valid. The result would be overturned for mechanics if the independent linear oracle failed or if any accepted row were nonfinite/non-SPD. Weakest evidence: the small C2 cloud and three-step horizon do not test proposal efficiency or recursive posterior accuracy.",
        "",
    ]
    return "\n".join(lines)


def run(args: argparse.Namespace) -> Path:
    if int(args.rows) < 2:
        raise ValueError("--rows must be at least two")
    if int(args.horizon) < 1 or int(args.horizon) > 20:
        raise ValueError("--horizon must lie in [1, 20]")
    output = _make_output_root(args.output_root)
    started = time.perf_counter()
    runtime = _configure_runtime()
    _load_modules()
    linear_fixture = _load_fixture(LINEAR_FIXTURE_PATH, "c2_mixture_ukf_lgssm_phase0_v1")
    c2_fixture = _load_fixture(C2_FIXTURE_PATH, "bayesfilter.c2_sv_frozen_fixture.v1")
    linear = _linear_oracle(
        linear_fixture, horizon=int(args.horizon), jit_compile=bool(args.jit_compile)
    )
    c2 = _c2_probe(
        c2_fixture,
        rows=int(args.rows),
        horizon=int(args.horizon),
        jit_compile=bool(args.jit_compile),
    )
    checks = {
        "linear_oracle_pass": bool(linear["pass"]),
        "c2_rows_finite_spd": bool(c2["all_rows_valid"]),
        "c2_model_contract": bool(c2["model_contract_pass"]),
        "records_complete": len(c2["records"]) == int(args.horizon),
        "source_files_present": all(
            path.is_file()
            for path in (PLAN_PATH, MOMENT_MAP_PATH, C2_MODEL_PATH, LINEAR_FIXTURE_PATH, C2_FIXTURE_PATH)
        ),
    }
    checks["hard_vetoes_pass"] = all(bool(value) for value in checks.values())
    elapsed = time.perf_counter() - started
    sources = {
        "plan": {"path": str(PLAN_PATH.relative_to(ROOT)), "sha256": _sha256_file(PLAN_PATH)},
        "moment_map": {"path": str(MOMENT_MAP_PATH.relative_to(ROOT)), "sha256": _sha256_file(MOMENT_MAP_PATH)},
        "c2_model": {"path": str(C2_MODEL_PATH.relative_to(ROOT)), "sha256": _sha256_file(C2_MODEL_PATH)},
        "linear_fixture": {"path": str(LINEAR_FIXTURE_PATH.relative_to(ROOT)), "sha256": _sha256_file(LINEAR_FIXTURE_PATH)},
        "c2_fixture": {"path": str(C2_FIXTURE_PATH.relative_to(ROOT)), "sha256": _sha256_file(C2_FIXTURE_PATH)},
        "driver": {"path": str(Path(__file__).resolve().relative_to(ROOT)), "sha256": _sha256_file(Path(__file__).resolve())},
    }
    status = "PASS_PHASE5_MECHANICS_AND_C2_WIRING" if checks["hard_vetoes_pass"] else "VETO_PHASE5_MECHANICS_OR_WIRING"
    payload: Mapping[str, object] = {
        "schema_version": RESULT_SCHEMA,
        "phase": PHASE_ID,
        "status": status,
        "continuation": "CONTINUE_NO_REAL_BLOCKER" if checks["hard_vetoes_pass"] else "CONTINUATION_VETO_PHASE5_VALIDITY",
        "failure_class": "none" if checks["hard_vetoes_pass"] else "implementation_or_numerical_validity",
        "repair": "none; proceed to representation diagnostic" if checks["hard_vetoes_pass"] else "repair the failing mechanics or adapter check before continuation",
        "next_phase_refresh": "fixed-map Hermite/RBF representation ladder on disjoint rows" if checks["hard_vetoes_pass"] else "refresh Phase 5 with a focused generic repair",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": elapsed,
        "environment": _environment(runtime, args),
        "seed": SEED,
        "route_id": ROUTE_ID,
        "route_classification": MAP_CLASSIFICATION,
        "rows": int(args.rows),
        "horizon": int(args.horizon),
        "sources": sources,
        "workspace": _workspace_manifest(),
        "checks": checks,
        "linear_oracle": linear,
        "c2_probe": c2,
        "nonclaims": [
            "no posterior correctness claim",
            "no likelihood unbiasedness claim",
            "no ESS superiority claim",
            "no basis or proposal promotion claim",
            "no analytical total-gradient or HMC claim",
            "no production/default-readiness claim",
        ],
    }
    _write_json(output / "records.json", {"linear_oracle": linear["records"], "c2_probe": c2["records"]})
    _write_json(output / "result.json", payload)
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "phase": PHASE_ID,
        "command": " ".join(sys.argv),
        "plan_sha256": sources["plan"]["sha256"],
        "route_id": ROUTE_ID,
        "route_classification": MAP_CLASSIFICATION,
        "rows": int(args.rows),
        "horizon": int(args.horizon),
        "expected_record_count": int(args.horizon),
        "seed": SEED,
        "environment": _environment(runtime, args),
        "sources": sources,
        "workspace": payload["workspace"],
        "evidence_contract": {
            "primary": "linear oracle parity and finite/SPD C2 wiring",
            "promotion": "none in Phase 5",
            "vetoes": "nonfinite, non-SPD, failed linear parity, missing records, source mismatch",
            "nonclaims": payload["nonclaims"],
        },
    }
    _write_json(output / "manifest.json", manifest)
    (output / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    (output / "result.md").write_text(_result_markdown(payload), encoding="utf-8")
    return output


def main() -> int:
    args = _parse_args()
    output = run(args)
    print(json.dumps({"phase": PHASE_ID, "output_root": str(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
