"""Run the v3.4 fresh paired three-arm q=20 proposal diagnostic.

Six fresh pilot receipts provide distinct M0 seeds. For each seed one
defensive q cloud is shared by identity, isotropic-support, and mode-aware
geometry arms. q is the annealing bridge; each independent-MH arm evaluates
its own proposal density at both the current and candidate states. This is
finite replication evidence only and does not launch HMC or train NeuTra.
"""

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

if os.environ.get("CUDA_VISIBLE_DEVICES") == "-1":
    raise RuntimeError("Phase 52 q=20 runner requires a visible trusted GPU")
if os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH") != "true":
    raise RuntimeError("Phase 52 q=20 runner requires TF_FORCE_GPU_ALLOW_GROWTH=true")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tensorflow as tf

from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth


GPU_POLICY = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
PHYSICAL_GPUS = tuple(tf.config.list_physical_devices("GPU"))
LOGICAL_GPUS = tuple(tf.config.list_logical_devices("GPU"))
if not PHYSICAL_GPUS or not LOGICAL_GPUS:
    raise RuntimeError("Phase 52 GPU memory policy produced no logical GPU")
try:
    tf.config.experimental.enable_tensor_float_32_execution(True)
except (AttributeError, RuntimeError):
    pass

from bayesfilter.nonlinear.ssl_lstm_complexity_batched_target_tf import (
    batch_native_complexity_posterior_target,
)
from bayesfilter.testing.annealed_smc_tf import (
    normalized_weight_diagnostics,
    systematic_resample_indices,
)
from bayesfilter.testing.importance_sampling_tf import (
    gaussian_mixture_log_prob,
    sample_gaussian_mixture,
)


RUNNER = Path(__file__).resolve()
PLAN = ROOT / "docs/plans/bayesfilter-ssl-lstm-q20-parameter-authority-corrected-continuation-2026-08-25.md"
GEOMETRY = ROOT / "docs/plans/artifacts/ssl-lstm-q20-seed-b-neutra-mode-failure-root-cause-2026-08-10/r1/geometry.json"
TARGET_MODULE = ROOT / "bayesfilter/nonlinear/ssl_lstm_complexity_batched_target_tf.py"
SMC_MODULE = ROOT / "bayesfilter/testing/annealed_smc_tf.py"
IMPORTANCE_MODULE = ROOT / "bayesfilter/testing/importance_sampling_tf.py"
CORRECTED_PILOT_RUNNER = ROOT / "docs/benchmarks/run_ssl_lstm_q20_parameter_authority_corrected_phase28_2026_08_25.py"
PHASE50_REPORT = ROOT / "docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase50-defensive-proposal-support/report/result.json"
PHASE51_REPORT = ROOT / "docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase51-mode-aware-proposal-geometry/report/result.json"
PHASE52_ARTIFACT_ROOT = Path(
    "docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/"
    "phase52-fresh-paired-uncertainty-replication"
)
PHASE52_ATTEMPT_ROOT = PHASE52_ARTIFACT_ROOT / "attempt-02"

EXPECTED_VERSION = "v3.4-fresh-paired-uncertainty-replication"
EXPECTED_MEASURE = "theta_R4"
EXPECTED_TARGET = "9a86e60081f1b9cd288dbdb1dcbe1e9a5b5e23d9b5ef97afdb72ee95c23d7278"
EXPECTED_M0 = "a1f0f0493bb8bd594923b61ee9a92f3c8dcb72a612b64ad675b9ab7ff4723631"
EXPECTED_C0 = "270fc99b81d08e23670c62fcd02e69e7452f26b5e5641187c3083faecbac7067"
EXPECTED_PILOT_SCHEMA = "bayesfilter.ssl_lstm.q20.corrected_theta_authority_pilot.v1"
EXPECTED_ARM_SCHEMA = "bayesfilter.ssl_lstm.q20.corrected_theta_authority.arm.v1"
EXPECTED_PHASE50_REPORT_SCHEMA = "bayesfilter.ssl_lstm.q20.corrected_theta_defensive_support_report.v1"
EXPECTED_PHASE50_REPORT_STATUS = "PASS_V3_2_DEFENSIVE_SUPPORT_REPORT"
EXPECTED_PHASE51_REPORT_SCHEMA = "bayesfilter.ssl_lstm.q20.corrected_theta_mode_aware_geometry_report.v1"
EXPECTED_PHASE51_REPORT_STATUS = "PASS_V3_3_MODE_AWARE_GEOMETRY_REPORT"
EXPECTED_FIXTURE_SCHEMA = "bayesfilter.ssl_lstm.q20.corrected_theta_fresh_paired_fixture.v1"
EXPECTED_FIXTURE_STATUS = "PASS_V3_4_FRESH_PAIRED_FIXTURE"
EXPECTED_GEOMETRY_SHA256 = "dc3dd7b84566867bc49c11ad16f50778d21457adbb398a17c2a75f3c3b461eeb"
EXPECTED_PILOT_RUNNER_RECEIPT_SHA256 = "c0b793ab10bd8d69cec22347c3beba00b5dd15e77e129f61b25d8dc585b9b703"
EXPECTED_PILOT_RUNNER_CURRENT_SHA256 = "e06845ee3f16773f181380c35297beaa2c4a489561c4b7d642c89853bb8ace1b"
PILOT_RUNNER_EQUIVALENCE = "one_trailing_blank_line_only_verified_2026_08_28"

SCHEDULE = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
PARTICLES = 256
CALIBRATION_PARTICLES = 64
REPLICATE_COUNT = 6
EXPECTED_FRESH_SEEDS = tuple((20260826, 5201 + index) for index in range(REPLICATE_COUNT))
EXPECTED_PILOT_ROOT_SEEDS = tuple((20260826, 5101 + index) for index in range(REPLICATE_COUNT))
DEFENSIVE_EPSILON = 0.20
SAFE_STD = 2.0
SUPPORT_RHO = 0.50
SUPPORT_STD = 4.0
GEOMETRY_RHO = 0.50
GEOMETRY_SCALE = 2.0
MODE_AXIS = 2
MH_STEPS = 8
RESAMPLING_SEED_OFFSET = 1000
SUPPORT_SEED_OFFSET = 30000
GEOMETRY_SEED_OFFSET = 40000
LOG_TWO_PI = tf.constant(1.8378770664093453, tf.float64)


class Phase52Error(RuntimeError):
    """Raised when the fresh paired boundary cannot be audited."""


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tensor_sha(value: Any) -> str:
    encoded = bytes(tf.io.serialize_tensor(tf.convert_to_tensor(value)).numpy())
    return hashlib.sha256(encoded).hexdigest()


def _safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_safe(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, tf.TensorShape):
        return [_safe(item) for item in value.as_list()]
    if isinstance(value, tf.dtypes.DType):
        return value.name
    if hasattr(value, "numpy"):
        return _safe(value.numpy())
    if hasattr(value, "tolist"):
        return _safe(value.tolist())
    if hasattr(value, "item"):
        return _safe(value.item())
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise Phase52Error(f"refusing to overwrite artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(_safe(payload), sort_keys=True, indent=2, allow_nan=False) + "\n").encode("ascii")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(encoded)
    temporary.replace(path)


def _write_tensor(path: Path, value: Any) -> Mapping[str, Any]:
    if path.exists():
        raise Phase52Error(f"refusing to overwrite tensor: {path}")
    tensor = tf.convert_to_tensor(value)
    encoded = bytes(tf.io.serialize_tensor(tensor).numpy())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)
    return {
        "path": path.as_posix(),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "dtype": tensor.dtype.name,
        "shape": list(tensor.shape),
        "bytes": len(encoded),
    }


def _load_json(root: Path, name: str) -> tuple[Path, Mapping[str, Any]]:
    if root.is_absolute() or ".." in root.parts:
        raise Phase52Error(f"path must be repository-relative: {root}")
    path = ROOT / root / name
    if not path.is_file():
        raise Phase52Error(f"missing receipt: {path}")
    return path, json.loads(path.read_text(encoding="utf-8"))


def _load_frozen_report(
    path: Path,
    *,
    schema: str,
    status: str,
    version: str,
    branch: str,
    arm_key: str,
) -> tuple[Path, Mapping[str, Any]]:
    if not path.is_file():
        raise Phase52Error(f"missing frozen report: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema") != schema
        or payload.get("status") != status
        or payload.get("plan_version") != version
        or payload.get("target_signature") != EXPECTED_TARGET
        or payload.get("branch") != branch
    ):
        raise Phase52Error(f"frozen report is stale or has the wrong branch: {path}")
    rows = payload.get("replicate_rows")
    if not isinstance(rows, list) or len(rows) != 3:
        raise Phase52Error(f"frozen report must contain three rows: {path}")
    if any(not isinstance(row.get(arm_key), Mapping) for row in rows):
        raise Phase52Error(f"frozen report rows are incomplete for {arm_key}: {path}")
    return path, payload


def _load_phase50_report() -> tuple[Path, Mapping[str, Any]]:
    return _load_frozen_report(
        PHASE50_REPORT,
        schema=EXPECTED_PHASE50_REPORT_SCHEMA,
        status=EXPECTED_PHASE50_REPORT_STATUS,
        version="v3.2-defensive-proposal-support",
        branch="support_broadened_does_not_reduce_variability",
        arm_key="broadened_support_mh",
    )


def _load_phase51_report() -> tuple[Path, Mapping[str, Any]]:
    return _load_frozen_report(
        PHASE51_REPORT,
        schema=EXPECTED_PHASE51_REPORT_SCHEMA,
        status=EXPECTED_PHASE51_REPORT_STATUS,
        version="v3.3-mode-aware-proposal-geometry",
        branch="mode_aware_geometry_reduces_between_bank_variability_descriptive",
        arm_key="mode_aware_geometry_mh",
    )


def _load_geometry() -> Mapping[str, tf.Tensor]:
    if _sha(GEOMETRY) != EXPECTED_GEOMETRY_SHA256:
        raise Phase52Error("geometry artifact hash is stale")
    payload = json.loads(GEOMETRY.read_text(encoding="utf-8"))
    if payload.get("status") != "GEOMETRY_DIAGNOSTIC_COMPLETED":
        raise Phase52Error("geometry calibration artifact is incomplete")
    labels = ("minus", "plus")
    means = tf.constant([payload["representatives"][label]["position"] for label in labels], tf.float64)
    precisions = tf.constant([payload["source_curvature"][label]["records"][-1]["precision"] for label in labels], tf.float64)
    covariances = tf.linalg.inv(precisions)
    tf.debugging.assert_all_finite(covariances, "geometry covariance")
    tf.debugging.assert_positive(tf.linalg.eigvalsh(covariances), "geometry covariance eigenvalues")
    geometry_covariances = tf.square(tf.constant(GEOMETRY_SCALE, tf.float64)) * covariances
    tf.debugging.assert_all_finite(geometry_covariances, "inflated geometry covariance")
    tf.debugging.assert_positive(tf.linalg.eigvalsh(geometry_covariances), "inflated geometry covariance eigenvalues")
    return {
        "means": means,
        "covariances": covariances,
        "geometry_covariances": geometry_covariances,
        "probabilities": tf.constant((0.5, 0.5), tf.float64),
        "center": tf.reduce_mean(means, axis=0),
    }


def _normal_log_prob(theta: tf.Tensor, center: tf.Tensor, scale: float) -> tf.Tensor:
    scale_tensor = tf.constant(scale, tf.float64)
    standardized = (theta - center[tf.newaxis, :]) / scale_tensor
    return -0.5 * tf.reduce_sum(tf.square(standardized), axis=1) - 4.0 * (
        tf.math.log(scale_tensor) + 0.5 * LOG_TWO_PI
    )


def _safe_log_prob(theta: tf.Tensor, center: tf.Tensor) -> tf.Tensor:
    return _normal_log_prob(theta, center, SAFE_STD)


def _support_log_prob(theta: tf.Tensor, center: tf.Tensor) -> tf.Tensor:
    return _normal_log_prob(theta, center, SUPPORT_STD)


def _proposal_log_theta(theta: tf.Tensor, chart: Mapping[str, tf.Tensor]) -> tf.Tensor:
    local = gaussian_mixture_log_prob(theta, chart["probabilities"], chart["means"], chart["covariances"])
    safe = _safe_log_prob(theta, chart["center"])
    epsilon = tf.constant(DEFENSIVE_EPSILON, tf.float64)
    return tf.reduce_logsumexp(
        tf.stack((tf.math.log1p(-epsilon) + local, tf.math.log(epsilon) + safe), axis=1), axis=1
    )


def _support_proposal_log_theta(theta: tf.Tensor, chart: Mapping[str, tf.Tensor]) -> tf.Tensor:
    q_log = _proposal_log_theta(theta, chart)
    support_log = _support_log_prob(theta, chart["center"])
    rho = tf.constant(SUPPORT_RHO, tf.float64)
    return tf.reduce_logsumexp(
        tf.stack((tf.math.log1p(-rho) + q_log, tf.math.log(rho) + support_log), axis=1), axis=1
    )


def _geometry_log_theta(theta: tf.Tensor, chart: Mapping[str, tf.Tensor]) -> tf.Tensor:
    return gaussian_mixture_log_prob(
        theta,
        chart["probabilities"],
        chart["means"],
        chart["geometry_covariances"],
    )


def _geometry_proposal_log_theta(theta: tf.Tensor, chart: Mapping[str, tf.Tensor]) -> tf.Tensor:
    q_log = _proposal_log_theta(theta, chart)
    geometry_log = _geometry_log_theta(theta, chart)
    rho = tf.constant(GEOMETRY_RHO, tf.float64)
    return tf.reduce_logsumexp(
        tf.stack((tf.math.log1p(-rho) + q_log, tf.math.log(rho) + geometry_log), axis=1), axis=1
    )


def _sample_theta(seed: tuple[int, int], chart: Mapping[str, tf.Tensor]) -> tuple[tf.Tensor, tf.Tensor]:
    """Draw from the frozen defensive base proposal q."""
    split = tf.random.experimental.stateless_split(tf.constant(seed, tf.int32), 4)
    local, labels = sample_gaussian_mixture(
        PARTICLES,
        chart["probabilities"],
        chart["means"],
        chart["covariances"],
        seed=tuple(int(value) for value in split[0].numpy()),
    )
    safe_noise = tf.random.stateless_normal((PARTICLES, 4), seed=split[1], dtype=tf.float64)
    safe = chart["center"][tf.newaxis, :] + SAFE_STD * safe_noise
    choose_safe = tf.random.stateless_uniform((PARTICLES,), seed=split[2], dtype=tf.float64) < DEFENSIVE_EPSILON
    theta = tf.where(choose_safe[:, None], safe, local)
    component = tf.where(choose_safe, tf.fill((PARTICLES,), 2), labels)
    return tf.ensure_shape(theta, (PARTICLES, 4)), tf.ensure_shape(component, (PARTICLES,))


def _sample_support_theta(seed: tuple[int, int], chart: Mapping[str, tf.Tensor]) -> tuple[tf.Tensor, tf.Tensor]:
    """Draw exactly from r_support=(1-rho)q+rho*N(center,4^2 I)."""
    split = tf.random.experimental.stateless_split(tf.constant(seed, tf.int32), 4)
    q_theta, q_components = _sample_theta(tuple(int(value) for value in split[0].numpy()), chart)
    support_noise = tf.random.stateless_normal((PARTICLES, 4), seed=split[1], dtype=tf.float64)
    support_theta = chart["center"][tf.newaxis, :] + SUPPORT_STD * support_noise
    choose_support = tf.random.stateless_uniform((PARTICLES,), seed=split[2], dtype=tf.float64) < SUPPORT_RHO
    theta = tf.where(choose_support[:, None], support_theta, q_theta)
    component = tf.where(choose_support, tf.fill((PARTICLES,), 3), q_components)
    return tf.ensure_shape(theta, (PARTICLES, 4)), tf.ensure_shape(component, (PARTICLES,))


def _sample_geometry_theta(seed: tuple[int, int], chart: Mapping[str, tf.Tensor]) -> tuple[tf.Tensor, tf.Tensor]:
    """Draw exactly from r_geom=(1-rho)q+rho*s_geom."""
    split = tf.random.experimental.stateless_split(tf.constant(seed, tf.int32), 4)
    q_theta, q_components = _sample_theta(tuple(int(value) for value in split[0].numpy()), chart)
    geometry_theta, geometry_components = sample_gaussian_mixture(
        PARTICLES,
        chart["probabilities"],
        chart["means"],
        chart["geometry_covariances"],
        seed=tuple(int(value) for value in split[1].numpy()),
    )
    choose_geometry = tf.random.stateless_uniform((PARTICLES,), seed=split[2], dtype=tf.float64) < GEOMETRY_RHO
    theta = tf.where(choose_geometry[:, None], geometry_theta, q_theta)
    component = tf.where(choose_geometry, geometry_components + 3, q_components)
    return tf.ensure_shape(theta, (PARTICLES, 4)), tf.ensure_shape(component, (PARTICLES,))


def _evaluate(theta: tf.Tensor, target: Any, chart: Mapping[str, tf.Tensor]) -> Mapping[str, tf.Tensor]:
    theta = tf.ensure_shape(tf.convert_to_tensor(theta, tf.float64), (PARTICLES, 4))
    value, score, status = target.neutra_batch_log_prob_and_grad_status(theta)
    finite = tf.logical_and(tf.math.is_finite(value), tf.reduce_all(tf.math.is_finite(score), axis=1))
    valid = tf.logical_and(
        finite,
        tf.logical_and(
            tf.equal(tf.convert_to_tensor(status["status_code"], tf.int32), 0),
            tf.cast(status["valid_pre_regularized_score"], tf.bool),
        ),
    )
    proposal_q = _proposal_log_theta(theta, chart)
    proposal_support = _support_proposal_log_theta(theta, chart)
    proposal_geometry = _geometry_proposal_log_theta(theta, chart)
    finite = tf.logical_and(
        finite,
        tf.reduce_all(
            tf.stack(
                (
                    tf.math.is_finite(proposal_q),
                    tf.math.is_finite(proposal_support),
                    tf.math.is_finite(proposal_geometry),
                ),
                axis=1,
            ),
            axis=1,
        ),
    )
    valid = tf.logical_and(valid, finite)
    return {
        "theta": theta,
        "target": tf.ensure_shape(tf.convert_to_tensor(value, tf.float64), (PARTICLES,)),
        "proposal_q": tf.ensure_shape(proposal_q, (PARTICLES,)),
        "proposal_support": tf.ensure_shape(proposal_support, (PARTICLES,)),
        "proposal_geometry": tf.ensure_shape(proposal_geometry, (PARTICLES,)),
        "valid": tf.ensure_shape(valid, (PARTICLES,)),
        "status_code": tf.ensure_shape(tf.convert_to_tensor(status["status_code"], tf.int32), (PARTICLES,)),
    }


@tf.function(
    input_signature=(
        tf.TensorSpec((PARTICLES, 4), tf.float64),
        tf.TensorSpec((PARTICLES, 4), tf.float64),
        tf.TensorSpec((PARTICLES,), tf.float64),
        tf.TensorSpec((PARTICLES,), tf.bool),
        tf.TensorSpec((PARTICLES,), tf.float64),
    ),
    jit_compile=True,
    reduce_retracing=False,
)
def _independent_mh_accept(
    current: tf.Tensor,
    candidate: tf.Tensor,
    log_ratio: tf.Tensor,
    candidate_valid: tf.Tensor,
    uniforms: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    log_alpha = tf.minimum(tf.constant(0.0, tf.float64), log_ratio)
    accepted = tf.logical_and(candidate_valid, tf.math.log(uniforms) < log_alpha)
    return tf.where(accepted[:, None], candidate, current), accepted, log_alpha


def _summary(theta: tf.Tensor, weights: tf.Tensor, roots: tf.Tensor) -> Mapping[str, Any]:
    weights = tf.ensure_shape(weights / tf.reduce_sum(weights), (PARTICLES,))
    mean = tf.reduce_sum(weights[:, None] * theta, axis=0)
    centered = theta - mean[None, :]
    covariance = tf.einsum("n,ni,nj->ij", weights, centered, centered)
    offdiag = covariance - tf.linalg.diag(tf.linalg.diag_part(covariance))
    return {
        "theta_mean": mean,
        "theta_mean_0": mean[0],
        "covariance": covariance,
        "covariance_offdiag_max_abs": tf.reduce_max(tf.abs(offdiag)),
        "negative_mode_fraction": tf.reduce_sum(weights * tf.cast(theta[:, MODE_AXIS] < 0.0, tf.float64)),
        "root_count": tf.size(tf.unique(roots).y),
        "coordinate_min": tf.reduce_min(theta, axis=0),
        "coordinate_max": tf.reduce_max(theta, axis=0),
        "weighted_ess_fraction": tf.math.reciprocal(tf.reduce_sum(tf.square(weights))) / tf.cast(PARTICLES, tf.float64),
    }


def _empty_mutation(kernel: str = "identity", proposal_kind: str = "none") -> Mapping[str, Any]:
    return {
        "kernel": kernel,
        "proposal_kind": proposal_kind,
        "steps": 0,
        "accepted_count": 0,
        "invalid_candidate_count": 0,
        "accepted_invalid_count": 0,
        "candidate_component_fraction": 0.0,
        "candidate_safe_fraction": 0.0,
        "move_fraction": 0.0,
        "acceptance_rate": 0.0,
        "mean_displacement": 0.0,
        "log_alpha_min": 0.0,
        "log_alpha_max": 0.0,
    }


def _mutate(
    theta: tf.Tensor,
    values: Mapping[str, tf.Tensor],
    beta: float,
    target: Any,
    chart: Mapping[str, tf.Tensor],
    seed: tuple[int, int],
    proposal_kind: str | None,
) -> tuple[Mapping[str, tf.Tensor], Mapping[str, Any]]:
    if proposal_kind is None:
        return values, _empty_mutation()
    if proposal_kind == "support":
        sampler = _sample_support_theta
        proposal_key = "proposal_support"
    elif proposal_kind == "geometry":
        sampler = _sample_geometry_theta
        proposal_key = "proposal_geometry"
    else:
        raise Phase52Error(f"unknown proposal kind: {proposal_kind}")
    current_theta = theta
    current = values
    accepted_total = tf.constant(0, tf.int32)
    invalid_total = tf.constant(0, tf.int32)
    accepted_invalid_total = tf.constant(0, tf.int32)
    component_candidate_total = tf.constant(0, tf.int32)
    safe_candidate_total = tf.constant(0, tf.int32)
    displacement_total = tf.constant(0.0, tf.float64)
    log_alpha_values: list[tf.Tensor] = []
    for step in range(MH_STEPS):
        candidate_theta, candidate_components = sampler((seed[0], seed[1] + step), chart)
        uniform_seed = tf.constant((seed[0], seed[1] + 1000 + step), tf.int32)
        uniforms = tf.random.stateless_uniform(
            (PARTICLES,), seed=uniform_seed, minval=1.0e-12, maxval=1.0, dtype=tf.float64
        )
        candidate = _evaluate(candidate_theta, target, chart)
        current_bridge = (1.0 - beta) * current["proposal_q"] + beta * current["target"]
        candidate_bridge = (1.0 - beta) * candidate["proposal_q"] + beta * candidate["target"]
        log_ratio = candidate_bridge - current_bridge + current[proposal_key] - candidate[proposal_key]
        log_ratio = tf.where(
            candidate["valid"],
            log_ratio,
            tf.fill((PARTICLES,), tf.constant(float("-inf"), tf.float64)),
        )
        next_theta, accepted, log_alpha = _independent_mh_accept(
            current_theta, candidate_theta, log_ratio, candidate["valid"], uniforms
        )
        displacement = tf.where(accepted[:, None], candidate_theta - current_theta, tf.zeros_like(candidate_theta))
        current = {
            key: tf.where(accepted, candidate[key], current[key]) if key != "theta" else next_theta
            for key in (
                "theta",
                "target",
                "proposal_q",
                "proposal_support",
                "proposal_geometry",
                "valid",
                "status_code",
            )
        }
        current_theta = next_theta
        accepted_total += tf.reduce_sum(tf.cast(accepted, tf.int32))
        invalid_total += tf.reduce_sum(tf.cast(tf.logical_not(candidate["valid"]), tf.int32))
        accepted_invalid_total += tf.reduce_sum(
            tf.cast(tf.logical_and(accepted, tf.logical_not(candidate["valid"])), tf.int32)
        )
        component_candidate_total += tf.reduce_sum(tf.cast(candidate_components >= 3, tf.int32))
        safe_candidate_total += tf.reduce_sum(tf.cast(tf.equal(candidate_components, 2), tf.int32))
        displacement_total += tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.square(displacement), axis=1)))
        log_alpha_values.append(tf.boolean_mask(log_alpha, candidate["valid"]))
    accepted_float = tf.cast(accepted_total, tf.float64)
    log_values = tf.concat(log_alpha_values, axis=0)
    log_values = tf.cond(tf.size(log_values) > 0, lambda: log_values, lambda: tf.zeros((1,), tf.float64))
    denominator = tf.cast(PARTICLES * MH_STEPS, tf.float64)
    return current, {
        "kernel": "independent_mh",
        "proposal_kind": proposal_kind,
        "steps": MH_STEPS,
        "accepted_count": accepted_total,
        "invalid_candidate_count": invalid_total,
        "accepted_invalid_count": accepted_invalid_total,
        "candidate_component_fraction": tf.cast(component_candidate_total, tf.float64) / denominator,
        "candidate_safe_fraction": tf.cast(safe_candidate_total, tf.float64) / denominator,
        "move_fraction": accepted_float / denominator,
        "acceptance_rate": accepted_float / denominator,
        "mean_displacement": displacement_total / denominator,
        "log_alpha_min": tf.reduce_min(log_values),
        "log_alpha_max": tf.reduce_max(log_values),
    }


def _run_arm(
    initial: Mapping[str, tf.Tensor],
    arm: str,
    target: Any,
    chart: Mapping[str, tf.Tensor],
    replicate_seed: tuple[int, int],
    output: Path,
) -> Mapping[str, Any]:
    arm_proposal = {
        "identity": None,
        "isotropic_support_mh": "support",
        "mode_aware_geometry_mh": "geometry",
    }
    if arm not in arm_proposal:
        raise Phase52Error(f"unknown arm: {arm}")
    proposal_kind = arm_proposal[arm]
    proposal_seed_offset = {
        None: 0,
        "support": SUPPORT_SEED_OFFSET,
        "geometry": GEOMETRY_SEED_OFFSET,
    }[proposal_kind]
    theta = tf.identity(initial["theta"])
    values = {key: tf.identity(value) for key, value in initial["values"].items()}
    roots = tf.identity(initial["roots"])
    log_weights = tf.zeros((PARTICLES,), tf.float64)
    stage_rows: list[Mapping[str, Any]] = []
    all_valid = True
    all_invalid_candidates_rejected = True
    for stage_index, (left, right) in enumerate(zip(SCHEDULE[:-1], SCHEDULE[1:])):
        delta = tf.constant(right - left, tf.float64)
        # q is the shared annealing base; the arm density appears only in MH.
        log_weights = log_weights + delta * (values["target"] - values["proposal_q"])
        diagnostics = normalized_weight_diagnostics(log_weights)
        terminal = stage_index == len(SCHEDULE) - 2
        if not terminal:
            parents = systematic_resample_indices(
                diagnostics["normalized_log_weights"],
                seed=(replicate_seed[0], replicate_seed[1] + RESAMPLING_SEED_OFFSET + stage_index),
            )
            theta = tf.gather(theta, parents)
            values = {key: tf.gather(value, parents) for key, value in values.items()}
            roots = tf.gather(roots, parents)
            log_weights = tf.zeros((PARTICLES,), tf.float64)
            values, mutation = _mutate(
                theta,
                values,
                right,
                target,
                chart,
                (replicate_seed[0], replicate_seed[1] + proposal_seed_offset + stage_index * 100),
                proposal_kind,
            )
        else:
            mutation = _empty_mutation("terminal_identity", proposal_kind or "none")
        theta = values["theta"]
        valid_now = bool(tf.reduce_all(values["valid"]).numpy())
        all_valid = all_valid and valid_now
        all_invalid_candidates_rejected = (
            all_invalid_candidates_rejected
            and int(tf.convert_to_tensor(mutation.get("accepted_invalid_count", 0), tf.int32).numpy()) == 0
        )
        stage_rows.append(
            {
                "stage_index": stage_index,
                "previous_beta": left,
                "beta": right,
                "pre_resampling_ess_fraction": diagnostics["effective_sample_size_fraction"],
                "pre_resampling_maximum_weight": diagnostics["maximum_normalized_weight"],
                "resampled": not terminal,
                "unique_root_count_after_resampling": tf.size(tf.unique(roots).y),
                "mutation": mutation,
                "all_current_status_valid": valid_now,
            }
        )
    weights = normalized_weight_diagnostics(log_weights)["normalized_weights"]
    summary = _summary(theta, weights, roots)
    tensors = {
        "final_theta": _write_tensor(output / f"{arm}-final-theta.tftensor", theta),
        "final_roots": _write_tensor(output / f"{arm}-final-roots.tftensor", roots),
        "final_weights": _write_tensor(output / f"{arm}-final-weights.tftensor", weights),
    }
    gates = {
        "final_shape_N_by_4": theta.shape == (PARTICLES, 4),
        "all_status_valid": all_valid,
        "all_invalid_candidates_rejected": all_invalid_candidates_rejected,
        "finite_theta": bool(tf.reduce_all(tf.math.is_finite(theta)).numpy()),
        "finite_weights": bool(tf.reduce_all(tf.math.is_finite(weights)).numpy()),
        "finite_summary": bool(tf.reduce_all(tf.math.is_finite(summary["theta_mean"])).numpy()),
    }
    return {
        "status": "PASS_V3_4_MUTATION_ARM" if all(gates.values()) else "PHASE52_MUTATION_ARM_FAIL",
        "arm": arm,
        "initial_tensor_hash": _tensor_sha(initial["theta"]),
        "kernel": "identity" if arm == "identity" else "independent_mh",
        "proposal_kind": proposal_kind or "none",
        "replicate_seed": list(replicate_seed),
        "proposal_seed_offset": proposal_seed_offset,
        "resampling_seeds": [
            [replicate_seed[0], replicate_seed[1] + RESAMPLING_SEED_OFFSET + stage_index]
            for stage_index in range(len(SCHEDULE) - 2)
        ],
        "proposal_seeds": [
            [replicate_seed[0], replicate_seed[1] + proposal_seed_offset + stage_index * 100]
            for stage_index in range(len(SCHEDULE) - 2)
        ] if proposal_kind is not None else [],
        "mh_steps": MH_STEPS,
        "gates": gates,
        "stages": stage_rows,
        "final_summary": summary,
        "final_tensors": tensors,
        "nonclaims": [
            "Finite mutation clouds are not posterior or IID draws.",
            "Acceptance and proposal differences are descriptive only.",
            "No HMC, whitening, exhaustive mode discovery, canonical LEDH, superiority, or default claim.",
        ],
    }


def _pilot(path: Path) -> tuple[Path, Mapping[str, Any], Mapping[str, Any]]:
    pilot_path, payload = _load_json(path, "pilot.json")
    if (
        payload.get("schema") != EXPECTED_PILOT_SCHEMA
        or payload.get("status") != "PASS_THETA_MEASURE_PILOT"
        or payload.get("measure") != EXPECTED_MEASURE
    ):
        raise Phase52Error(f"pilot contract failed: {pilot_path}")
    m0 = payload.get("arms", {}).get("M0")
    c0 = payload.get("arms", {}).get("C0")
    if (
        not isinstance(m0, Mapping)
        or not isinstance(c0, Mapping)
        or m0.get("schema") != EXPECTED_ARM_SCHEMA
        or c0.get("schema") != EXPECTED_ARM_SCHEMA
    ):
        raise Phase52Error(f"pilot arm schema failed: {pilot_path}")
    if m0.get("status") != "PASS_THETA_MEASURE_PILOT" or c0.get("status") != "PASS_THETA_MEASURE_PILOT":
        raise Phase52Error(f"pilot arms are not passing: {pilot_path}")
    if m0.get("target_signature") != EXPECTED_TARGET or c0.get("target_signature") != EXPECTED_TARGET:
        raise Phase52Error(f"pilot target mismatch: {pilot_path}")
    m0_configuration = m0.get("configuration", {})
    c0_configuration = c0.get("configuration", {})
    if (
        m0_configuration.get("protocol_hash") != EXPECTED_M0
        or c0_configuration.get("protocol_hash") != EXPECTED_C0
    ):
        raise Phase52Error(f"pilot protocol mismatch: {pilot_path}")
    if (
        int(m0_configuration.get("particles", -1)) != PARTICLES
        or tuple(float(value) for value in m0_configuration.get("schedule", ())) != SCHEDULE
        or float(m0_configuration.get("defensive_epsilon", -1.0)) != DEFENSIVE_EPSILON
    ):
        raise Phase52Error(f"pilot M0 configuration mismatch: {pilot_path}")
    calibration = payload.get("calibration", {})
    if (
        not isinstance(calibration, Mapping)
        or calibration.get("status") != "CALIBRATION_COMPLETED"
        or calibration.get("measure") != EXPECTED_MEASURE
        or calibration.get("target_signature") != EXPECTED_TARGET
        or int(calibration.get("particle_count", -1)) != CALIBRATION_PARTICLES
    ):
        raise Phase52Error(f"pilot calibration mismatch: {pilot_path}")
    manifest = payload.get("run_manifest", {})
    root_seed = tuple(int(value) for value in manifest.get("seeds", {}).get("root", ()))
    if (
        manifest.get("runner") != CORRECTED_PILOT_RUNNER.as_posix()
        or manifest.get("cuda_visible_devices") != "-1"
        or manifest.get("tf_force_gpu_allow_growth") != "true"
        or manifest.get("physical_gpus") != []
        or manifest.get("logical_gpus") != []
        or manifest.get("jit_compile") is not True
        or manifest.get("source_sha256", {}).get("runner") != EXPECTED_PILOT_RUNNER_RECEIPT_SHA256
        or manifest.get("source_sha256", {}).get("plan") != _sha(PLAN)
    ):
        raise Phase52Error(f"pilot runner/device provenance mismatch: {pilot_path}")
    seed = tuple(int(value) for value in m0_configuration["seed"])
    return pilot_path, payload, {"m0": m0, "c0": c0, "seed": seed, "root_seed": root_seed}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-root-1", required=True, type=Path)
    parser.add_argument("--pilot-root-2", required=True, type=Path)
    parser.add_argument("--pilot-root-3", required=True, type=Path)
    parser.add_argument("--pilot-root-4", required=True, type=Path)
    parser.add_argument("--pilot-root-5", required=True, type=Path)
    parser.add_argument("--pilot-root-6", required=True, type=Path)
    parser.add_argument("--fixture-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    if _sha(CORRECTED_PILOT_RUNNER) != EXPECTED_PILOT_RUNNER_CURRENT_SHA256:
        raise Phase52Error("current corrected pilot runner does not match its audited equivalent source")
    roots = (
        args.pilot_root_1,
        args.pilot_root_2,
        args.pilot_root_3,
        args.pilot_root_4,
        args.pilot_root_5,
        args.pilot_root_6,
    )
    all_paths = roots + (args.fixture_root, args.output_root)
    if any(path.is_absolute() or ".." in path.parts for path in all_paths):
        raise Phase52Error("all paths must be repository-relative")
    expected_roots = tuple(PHASE52_ATTEMPT_ROOT / f"pilot-{index + 1:02d}" for index in range(REPLICATE_COUNT))
    if roots != expected_roots:
        raise Phase52Error("pilot roots do not match the predeclared fresh Phase 52 namespace")
    if args.fixture_root != PHASE52_ARTIFACT_ROOT / "fixture":
        raise Phase52Error("fixture root does not match the predeclared Phase 52 namespace")
    if args.output_root != PHASE52_ATTEMPT_ROOT / "q20-paired":
        raise Phase52Error("output root does not match the predeclared Phase 52 namespace")
    output = ROOT / args.output_root
    if output.exists():
        raise Phase52Error(f"refusing to overwrite output root: {output}")
    started = time.perf_counter()
    fixture_path, fixture = _load_json(args.fixture_root, "result.json")
    if (
        fixture.get("schema") != EXPECTED_FIXTURE_SCHEMA
        or fixture.get("status") != EXPECTED_FIXTURE_STATUS
        or fixture.get("plan_version") != EXPECTED_VERSION
        or fixture.get("depth_steps") != MH_STEPS
        or float(fixture.get("support_rho", -1.0)) != SUPPORT_RHO
        or float(fixture.get("support_std", -1.0)) != SUPPORT_STD
        or float(fixture.get("geometry_rho", -1.0)) != GEOMETRY_RHO
        or float(fixture.get("geometry_scale", -1.0)) != GEOMETRY_SCALE
        or not all(fixture.get("gates", {}).values())
    ):
        raise Phase52Error(f"fresh paired fixture is not passing: {fixture_path}")
    phase50_report_path, phase50_report = _load_phase50_report()
    phase51_report_path, phase51_report = _load_phase51_report()
    pilot_records = [_pilot(path) for path in roots]
    pilot_paths = [record[0] for record in pilot_records]
    if len({_sha(path) for path in pilot_paths}) != REPLICATE_COUNT:
        raise Phase52Error("pilot receipts are not distinct")
    seeds = [record[2]["seed"] for record in pilot_records]
    if len(set(seeds)) != REPLICATE_COUNT:
        raise Phase52Error("pilot M0 seeds are not distinct")
    if tuple(seeds) != EXPECTED_FRESH_SEEDS:
        raise Phase52Error("pilot M0 seeds do not match the predeclared fresh seed ledger")
    root_seeds = [record[2]["root_seed"] for record in pilot_records]
    if tuple(root_seeds) != EXPECTED_PILOT_ROOT_SEEDS:
        raise Phase52Error("pilot root seeds do not match the corrected generator ledger")
    chart = _load_geometry()
    target = batch_native_complexity_posterior_target(20, jit_compile=True, principal_sqrt_backend="tensorflow_eigh")
    if target.target_signature() != EXPECTED_TARGET:
        raise Phase52Error("target signature changed")
    replicates: list[Mapping[str, Any]] = []
    for index, ((pilot_path, _pilot_payload, _arm_info), root_seed) in enumerate(zip(pilot_records, seeds)):
        replicate_root = output / f"replicate-{index + 1:02d}"
        replicate_root.mkdir(parents=True)
        initial_theta, component = _sample_theta(root_seed, chart)
        initial_values = _evaluate(initial_theta, target, chart)
        if not bool(tf.reduce_all(initial_values["valid"]).numpy()):
            raise Phase52Error(f"initial q=20 proposal contains invalid rows in replicate {index + 1}")
        initial = {"theta": initial_theta, "values": initial_values, "roots": tf.range(PARTICLES, dtype=tf.int32)}
        initial_tensors = {
            "theta": _write_tensor(replicate_root / "initial-theta.tftensor", initial_theta),
            "proposal_q": _write_tensor(
                replicate_root / "initial-proposal-q-log-theta.tftensor", initial_values["proposal_q"]
            ),
            "proposal_support": _write_tensor(
                replicate_root / "initial-proposal-support-log-theta.tftensor",
                initial_values["proposal_support"],
            ),
            "proposal_geometry": _write_tensor(
                replicate_root / "initial-proposal-geometry-log-theta.tftensor",
                initial_values["proposal_geometry"],
            ),
            "target": _write_tensor(replicate_root / "initial-target-log-theta.tftensor", initial_values["target"]),
            "roots": _write_tensor(replicate_root / "initial-roots.tftensor", initial["roots"]),
            "components": _write_tensor(replicate_root / "initial-proposal-components.tftensor", component),
        }
        initial_hash = initial_tensors["theta"]["sha256"]
        identity = _run_arm(initial, "identity", target, chart, root_seed, replicate_root / "identity")
        support_arm = _run_arm(
            initial,
            "isotropic_support_mh",
            target,
            chart,
            root_seed,
            replicate_root / "isotropic-support-mh",
        )
        geometry_arm = _run_arm(
            initial,
            "mode_aware_geometry_mh",
            target,
            chart,
            root_seed,
            replicate_root / "mode-aware-geometry-mh",
        )
        resampling_seed_sets = (
            identity["resampling_seeds"],
            support_arm["resampling_seeds"],
            geometry_arm["resampling_seeds"],
        )
        paired = {
            "initial_tensor_hash": initial_hash,
            "identity_initial_tensor_hash": identity["initial_tensor_hash"],
            "isotropic_support_initial_tensor_hash": support_arm["initial_tensor_hash"],
            "mode_aware_geometry_initial_tensor_hash": geometry_arm["initial_tensor_hash"],
            "resampling_seed_offset": RESAMPLING_SEED_OFFSET,
            "support_seed_offset": SUPPORT_SEED_OFFSET,
            "geometry_seed_offset": GEOMETRY_SEED_OFFSET,
            "same_resampling_seeds": resampling_seed_sets[0] == resampling_seed_sets[1] == resampling_seed_sets[2],
            "same_initial_cloud": len(
                {
                    initial_hash,
                    identity["initial_tensor_hash"],
                    support_arm["initial_tensor_hash"],
                    geometry_arm["initial_tensor_hash"],
                }
            ) == 1,
        }
        replicates.append({
            "replicate": index + 1,
            "pilot_root": roots[index],
            "pilot_sha256": _sha(pilot_path),
            "pilot_m0_seed": list(root_seed),
            "pilot_root_seed": list(pilot_records[index][2]["root_seed"]),
            "initial_tensors": initial_tensors,
            "paired": paired,
            "identity": identity,
            "isotropic_support_mh": support_arm,
            "mode_aware_geometry_mh": geometry_arm,
        })
    hard_pass = all(
        rep["identity"]["status"] == "PASS_V3_4_MUTATION_ARM"
        and rep["isotropic_support_mh"]["status"] == "PASS_V3_4_MUTATION_ARM"
        and rep["mode_aware_geometry_mh"]["status"] == "PASS_V3_4_MUTATION_ARM"
        and rep["paired"]["same_initial_cloud"]
        and rep["paired"]["same_resampling_seeds"]
        for rep in replicates
    )
    result = {
        "schema": "bayesfilter.ssl_lstm.q20.corrected_theta_fresh_paired_boundary.v1",
        "status": "PASS_V3_4_FRESH_PAIRED_BOUNDARY" if hard_pass else "PHASE52_FRESH_PAIRED_BOUNDARY_FAIL",
        "plan_version": EXPECTED_VERSION,
        "role": "six_fresh_bank_paired_identity_support_geometry_finite_replication_diagnostic",
        "measure": EXPECTED_MEASURE,
        "target_signature": target.target_signature(),
        "target_adapter_signature": target.adapter_signature(),
        "schedule": list(SCHEDULE),
        "particles": PARTICLES,
        "calibration_particles": CALIBRATION_PARTICLES,
        "replicate_count": REPLICATE_COUNT,
        "defensive_epsilon": DEFENSIVE_EPSILON,
        "safe_std": SAFE_STD,
        "support_rho": SUPPORT_RHO,
        "support_std": SUPPORT_STD,
        "geometry_rho": GEOMETRY_RHO,
        "geometry_scale": GEOMETRY_SCALE,
        "annealing_base_law": "q(theta)",
        "support_proposal_law": "r_support(theta)=(1-rho)q(theta)+rho*N(center,4^2*I)",
        "geometry_proposal_law": "r_geom(theta)=(1-rho)q(theta)+rho*s_geom(theta)",
        "geometry_component_law": "s_geom(theta)=0.5*N(m_minus,kappa^2*C_minus)+0.5*N(m_plus,kappa^2*C_plus)",
        "mode_axis": MODE_AXIS,
        "mh_steps": MH_STEPS,
        "terminal_resampling": False,
        "fixture_required": True,
        "geometry_artifact_sha256": EXPECTED_GEOMETRY_SHA256,
        "pilot_receipts_distinct": True,
        "pilot_roots_match_fresh_namespace": True,
        "pilot_seeds_match_fresh_ledger": True,
        "pilot_runner_sha256": EXPECTED_PILOT_RUNNER_RECEIPT_SHA256,
        "pilot_runner_current_sha256": _sha(CORRECTED_PILOT_RUNNER),
        "pilot_runner_equivalence": PILOT_RUNNER_EQUIVALENCE,
        "phase50_report_sha256": _sha(phase50_report_path),
        "phase50_report_branch": phase50_report["branch"],
        "phase51_report_sha256": _sha(phase51_report_path),
        "phase51_report_branch": phase51_report["branch"],
        "replicates": replicates,
        "fresh_rows_used_for_training": False,
        "fresh_rows_used_for_selection": False,
        "hmc_launched": False,
        "device": {
            "trust_basis": "owner_designated_managed_session_visible_gpu_trusted",
            "gpu_memory_policy": GPU_POLICY,
            "physical_devices": [device.name for device in PHYSICAL_GPUS],
            "logical_devices": [device.name for device in LOGICAL_GPUS],
            "tf32_enabled": bool(tf.config.experimental.tensor_float_32_execution_enabled()),
            "jit_compile_target": True,
            "jit_compile_mutation": True,
        },
        "run_manifest": {
            "trust_basis": "owner_designated_managed_session_visible_gpu_trusted",
            "program": PLAN.as_posix(),
            "runner": RUNNER.as_posix(),
            "command": " ".join(sys.argv),
            "git_commit": subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=ROOT, text=True).strip(),
            "git_dirty": bool(subprocess.check_output(("git", "status", "--porcelain"), cwd=ROOT, text=True).strip()),
            "python": sys.executable,
            "python_version": platform.python_version(),
            "tensorflow": tf.__version__,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "tf_force_gpu_allow_growth": os.environ["TF_FORCE_GPU_ALLOW_GROWTH"],
            "gpu_memory_growth_verified": True,
            "jit_compile": True,
            "wall_seconds": time.perf_counter() - started,
            "source_sha256": {
                "plan": _sha(PLAN),
                "runner": _sha(RUNNER),
                "geometry": _sha(GEOMETRY),
                "target_module": _sha(TARGET_MODULE),
                "smc_module": _sha(SMC_MODULE),
                "importance_module": _sha(IMPORTANCE_MODULE),
                "corrected_pilot_runner": _sha(CORRECTED_PILOT_RUNNER),
                "corrected_pilot_runner_receipt": EXPECTED_PILOT_RUNNER_RECEIPT_SHA256,
                "phase50_report": _sha(phase50_report_path),
                "phase51_report": _sha(phase51_report_path),
                "fixture": _sha(fixture_path),
                **{f"pilot_{index + 1}": _sha(path) for index, path in enumerate(pilot_paths)},
            },
        },
        "nonclaims": [
            "Identity, isotropic-support, and mode-aware-geometry clouds are finite diagnostics, not IID or posterior proofs.",
            "Acceptance, ESS, mode mass, root count, and spread differences are descriptive only.",
            "Six-bank uncertainty diagnostics do not establish a population ranking or method superiority.",
            "No HMC, convergence, exhaustive mode discovery, canonical LEDH, whitening, superiority, or default claim.",
        ],
    }
    _write_json(output / "result.json", result)
    (output / "result.md").write_text(
        "# v3.4 Six-Bank Paired Proposal Boundary\n\nStatus: `"
        + result["status"]
        + "`\n\nFresh finite replication diagnostic; no ranking, whitening, or posterior claim.\n",
        encoding="ascii",
    )
    print(json.dumps({"status": result["status"], "output_root": args.output_root.as_posix()}, sort_keys=True))
    return 0 if hard_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
