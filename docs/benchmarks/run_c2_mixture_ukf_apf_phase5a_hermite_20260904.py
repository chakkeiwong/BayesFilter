"""Run the bounded Phase 5A fixed-map Hermite representation pilot.

The driver reconstructs the deterministic Phase 5 C2 cloud, builds one
observation-informed lagged moment map, and fits the exact one-step C2 target
in that frozen Gaussian reference coordinate.  Training, holdout, and audit
normal banks are disjoint.  The fit is a diagnostic use of the repository's
fixed-design ALS API; it is not a proposal, filtering, gradient, or
production claim.
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

# Defer the inherited growth variable until the repository helper has been
# imported and can establish the policy before logical-device initialization.
_DEFERRED_TF_FORCE_GPU_ALLOW_GROWTH = os.environ.pop(
    "TF_FORCE_GPU_ALLOW_GROWTH", None
)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import tensorflow as tf

DTYPE = tf.float64
PHASE_ID = "c2_mixture_ukf_apf_phase5a_hermite_pilot_v1"
RESULT_SCHEMA = "c2_mixture_ukf_apf_phase5a_hermite_result_v1"
MANIFEST_SCHEMA = "c2_mixture_ukf_apf_phase5a_hermite_manifest_v1"
ROUTE_ID = "c2_fixed_map_hermite_tt_representation_pilot_v1"
ROUTE_CLASSIFICATION = "extension_or_invention_candidate_diagnostic_only"

FIXTURE_PATH = ROOT / "docs/benchmarks/fixtures/c2_sv_n4_seed52_obs42_t20_frozen_v1.json"
MODEL_PATH = ROOT / "bayesfilter/highdim/c2_sv_frozen_proposal_apf_tf.py"
MOMENT_MAP_PATH = ROOT / "bayesfilter/highdim/recursive_moment_map_tf.py"
PLAN_PATH = ROOT / "docs/plans/c2-mixture-ukf-apf-phase5a-hermite-pilot-20260904.md"

STATE_DIM = 4
CARRIED_ROWS = 128
TRAIN_ROWS = 512
HOLDOUT_ROWS = 512
AUDIT_ROWS = 4096
DEGREES = (6, 8, 10)
TT_RANK = 2
ALS_SWEEPS = 2
RIDGE = 1.0e-8
CONDITION_VETO = 1.0e14
SEED = (20260904, 701)
MAP_SEED = (20260904, 501)  # same bank identity as the closed Phase 5 probe
SHELL_RADIUS = 2.0

_MODULES_LOADED = False


def _load_modules() -> None:
    global _MODULES_LOADED
    global C2StochasticVolatilityFrozenAPFModel, DensityMeasure, MassMeasure
    global FixedTTFitConfig, FixedTTFitSampleBatch, FixedTTFitter
    global HermiteBasis1D, ProductBasis, MeasureConvention, TTCore
    global build_lagged_moment_map, make_weighted_transition_moment_kernel
    global _initial_tt_cores
    if _MODULES_LOADED:
        return
    from bayesfilter.highdim.c2_sv_frozen_proposal_apf_tf import (
        C2StochasticVolatilityFrozenAPFModel as _C2Model,
    )
    from bayesfilter.highdim.diagnostics import (
        DensityMeasure as _DensityMeasure,
        MassMeasure as _MassMeasure,
        MeasureConvention as _MeasureConvention,
    )
    from bayesfilter.highdim.bases import (
        HermiteBasis1D as _HermiteBasis1D,
        ProductBasis as _ProductBasis,
    )
    from bayesfilter.highdim.fitting import (
        FixedTTFitConfig as _FixedTTFitConfig,
        FixedTTFitSampleBatch as _FixedTTFitSampleBatch,
        FixedTTFitter as _FixedTTFitter,
    )
    from bayesfilter.highdim.recursive_moment_map_tf import (
        build_lagged_moment_map as _build_lagged_moment_map,
        make_weighted_transition_moment_kernel as _make_weighted_transition_moment_kernel,
    )
    from bayesfilter.highdim.squared_tt_engine_v0_tf import (
        _initial_tt_cores as _initial,
    )
    from bayesfilter.highdim.tt import TTCore as _TTCore

    C2StochasticVolatilityFrozenAPFModel = _C2Model
    DensityMeasure = _DensityMeasure
    MassMeasure = _MassMeasure
    MeasureConvention = _MeasureConvention
    FixedTTFitConfig = _FixedTTFitConfig
    FixedTTFitSampleBatch = _FixedTTFitSampleBatch
    FixedTTFitter = _FixedTTFitter
    HermiteBasis1D = _HermiteBasis1D
    ProductBasis = _ProductBasis
    TTCore = _TTCore
    build_lagged_moment_map = _build_lagged_moment_map
    make_weighted_transition_moment_kernel = _make_weighted_transition_moment_kernel
    _initial_tt_cores = _initial
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
        raise ValueError("non-finite scalar")
    return float(tensor.numpy())


def _max_abs(value: object) -> float:
    return _scalar(tf.reduce_max(tf.abs(tf.convert_to_tensor(value, DTYPE))))


def _load_fixture() -> Mapping[str, object]:
    payload = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    if payload.get("schema_id") != "bayesfilter.c2_sv_frozen_fixture.v1":
        raise ValueError("unexpected C2 fixture schema")
    if int(payload["state_dimension"]) != STATE_DIM:
        raise ValueError("Phase 5A expects the four-dimensional C2 fixture")
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
            "execution_lane": "trusted_gpu_xla_phase5a_hermite",
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


def _make_output_root(value: str) -> Path:
    candidate = Path(value)
    output = (ROOT / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {output}")
    output.mkdir(parents=True, exist_ok=False)
    return output


def _model_and_map(
    fixture: Mapping[str, object], *, jit_compile: bool
) -> Mapping[str, object]:
    theta = _tensor([float(fixture["gamma"]), math.log(float(fixture["beta"]))])
    transition_fixture = _tensor(fixture["transition_matrix"])
    process_fixture = _tensor(fixture["process_covariance"])
    coupling = transition_fixture - theta[0] * tf.eye(STATE_DIM, dtype=DTYPE)
    model = C2StochasticVolatilityFrozenAPFModel(
        coupling_matrix=coupling, sigma=float(fixture["sigma"])
    )
    transition = model.transition_matrix(theta)
    process = tf.eye(STATE_DIM, dtype=DTYPE) * float(fixture["sigma"]) ** 2
    stationary = _tensor(fixture["stationary_covariance"])
    stationary_model, _ = model.stationary_covariance_and_derivative(theta)
    initial_chol = tf.linalg.cholesky(stationary)
    bank = tf.random.stateless_normal([CARRIED_ROWS, STATE_DIM], list(MAP_SEED), dtype=DTYPE)
    states = tf.linalg.matmul(bank, initial_chol, transpose_b=True)
    observations = _tensor(fixture["observations"])
    log_like0 = model.observation_log_density(theta, states, observations[0], 0)
    weights = tf.exp(log_like0 - tf.reduce_logsumexp(log_like0))
    conditional_means = tf.linalg.matmul(states, transition, transpose_b=True)
    conditional_covariances = tf.broadcast_to(
        process[None, :, :], [CARRIED_ROWS, STATE_DIM, STATE_DIM]
    )
    kernel = make_weighted_transition_moment_kernel(
        particle_count=CARRIED_ROWS, state_dim=STATE_DIM, jit_compile=bool(jit_compile)
    )
    built = build_lagged_moment_map(
        weights,
        conditional_means,
        conditional_covariances,
        jit_compile=bool(jit_compile),
        kernel=kernel,
    )
    if not bool(built["valid"].numpy()):
        raise ValueError("Phase 5A map is invalid")
    coordinate_map = built["coordinate_map"]
    if not _finite(states) or not _finite(weights):
        raise ValueError("non-finite carried cloud")
    return {
        "model": model,
        "theta": theta,
        "observations": observations,
        "states": states,
        "weights": weights,
        "transition": transition,
        "process": process,
        "coordinate_map": coordinate_map,
        "predicted_mean": built["predicted_mean"],
        "predicted_covariance": built["predicted_covariance"],
        "map_raw_minimum_eigenvalue": built["raw_minimum_eigenvalue"],
        "map_minimum_eigenvalue": built["minimum_eigenvalue"],
        "map_condition_number": built["condition_number"],
        "transition_fixture_error": _max_abs(transition - transition_fixture),
        "process_fixture_error": _max_abs(process - process_fixture),
        "stationary_fixture_error": _max_abs(stationary_model - stationary),
        "initial_observation_ess": _scalar(tf.math.reciprocal(tf.reduce_sum(tf.square(weights)))),
    }


def _make_exact_target_kernel(
    *,
    parent_states: tf.Tensor,
    parent_weights: tf.Tensor,
    transition: tf.Tensor,
    theta: tf.Tensor,
    observation: tf.Tensor,
    sigma: float,
    row_count: int,
    jit_compile: bool,
):
    """Compile the exact finite mixture target for one fixed row count."""

    parent_count = int(parent_states.shape[0])
    state_dim = int(parent_states.shape[1])
    parent_states = tf.ensure_shape(parent_states, [parent_count, state_dim])
    parent_weights = tf.ensure_shape(parent_weights, [parent_count])
    transition = tf.ensure_shape(transition, [state_dim, state_dim])
    theta = tf.ensure_shape(theta, [2])
    observation = tf.ensure_shape(observation, [state_dim])
    log_parent_weights = tf.math.log(parent_weights)
    log_two_pi_sigma2 = tf.constant(
        math.log(2.0 * math.pi * float(sigma) ** 2), DTYPE
    )

    @tf.function(
        input_signature=[tf.TensorSpec([row_count, state_dim], DTYPE)],
        jit_compile=bool(jit_compile),
        autograph=False,
        reduce_retracing=True,
    )
    def kernel(reference_rows: tf.Tensor) -> tf.Tensor:
        # Map rows use x = m + L u (the map's row convention).
        # The exact target is a log-sum-exp over every carried ancestor.
        x = reference_rows
        means = tf.linalg.matmul(parent_states, transition, transpose_b=True)
        residual = x[:, None, :] - means[None, :, :]
        log_transition = -0.5 * (
            tf.cast(state_dim, DTYPE) * log_two_pi_sigma2
            + tf.reduce_sum(tf.square(residual), axis=2)
        )
        log_gamma = tf.reduce_logsumexp(
            log_transition + log_parent_weights[None, :], axis=1
        )
        log_observation = tf.reduce_sum(
            -0.5 * tf.constant(math.log(2.0 * math.pi), DTYPE)
            - theta[1]
            - 0.5 * x
            - 0.5
            * tf.square(observation)[None, :]
            * tf.exp(-x - 2.0 * theta[1]),
            axis=1,
        )
        return log_gamma + log_observation

    return kernel


def _standard_normal_bank(count: int, seed: tuple[int, int]) -> tf.Tensor:
    return tf.random.stateless_normal([int(count), STATE_DIM], list(seed), dtype=DTYPE)


def _reference_target_values(
    rows: tf.Tensor,
    *,
    coordinate_map: Any,
    target_kernel: Any,
    shift: tf.Tensor | None = None,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    physical, logdet = coordinate_map.forward(rows)
    log_gamma = target_kernel(physical)
    log_eta = -0.5 * (
        tf.cast(STATE_DIM, DTYPE) * tf.constant(math.log(2.0 * math.pi), DTYPE)
        + tf.reduce_sum(tf.square(rows), axis=1)
    )
    log_h = log_gamma + logdet - log_eta
    active_shift = (
        tf.reduce_logsumexp(log_h) - tf.math.log(tf.cast(tf.shape(log_h)[0], DTYPE))
        if shift is None
        else tf.convert_to_tensor(shift, DTYPE)
    )
    sqrt_target = tf.exp(0.5 * (log_h - active_shift))
    return log_h, sqrt_target, active_shift


def _gram_squared_normalizer(cores: Sequence[TTCore], basis: ProductBasis) -> tf.Tensor:
    state = tf.ones([1, 1], DTYPE)
    for axis, core in enumerate(cores):
        mass = basis.bases[axis].mass_matrix(MassMeasure.REFERENCE_MEASURE)
        state = tf.einsum("akb,AlB,kl,aA->bB", core.values, core.values, mass, state)
    return tf.reshape(state, [])


def _fit_config(degree: int) -> Any:
    del degree
    return FixedTTFitConfig(
        ranks=tuple([1] + [TT_RANK] * (STATE_DIM - 1) + [1]),
        ridge=RIDGE,
        max_sweeps=ALS_SWEEPS,
        sweep_order=tuple(range(STATE_DIM)),
        row_budget=TRAIN_ROWS,
        column_budget=4096,
        dense_matrix_byte_budget=1 << 30,
        normal_matrix_byte_budget=1 << 30,
        condition_number_warning=1.0e12,
        condition_number_veto=CONDITION_VETO,
        holdout_tolerance=1.0e30,
    )


def _condition_summary(result: Any) -> Mapping[str, object]:
    values: list[float] = []
    unscaled: list[float] = []
    for record in result.core_update_statuses:
        for key in ("transformed_system_condition_number", "condition_number"):
            value = record.get(key)
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                values.append(float(value))
                break
        value = record.get("unscaled_normal_condition_number")
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            unscaled.append(float(value))
    return {
        "scaled_augmented_condition_max": max(values) if values else None,
        "unscaled_normal_condition_max_diagnostic": max(unscaled) if unscaled else None,
        "update_count": len(result.core_update_statuses),
    }


def _rms(values: tf.Tensor, weights: tf.Tensor | None = None) -> float:
    values = tf.convert_to_tensor(values, DTYPE)
    if weights is None:
        return _scalar(tf.sqrt(tf.reduce_mean(tf.square(values))))
    weights = tf.convert_to_tensor(weights, DTYPE)
    return _scalar(tf.sqrt(tf.reduce_sum(weights * tf.square(values)) / tf.reduce_sum(weights)))


def _prior_predictive_target_audit(
    map_payload: Mapping[str, object], *, row_count: int, seed: tuple[int, int]
) -> Mapping[str, tf.Tensor]:
    """Estimate Z_T by sampling the normalized predictive mixture.

    If J has probabilities ``w_j`` and X|J=j follows the exact transition,
    then E[g(y|X)] equals the finite target integral.  This avoids the
    high-variance ``1/eta`` ratio used only by the reference-coordinate audit.
    """

    parent_states = tf.convert_to_tensor(map_payload["states"], DTYPE)
    parent_weights = tf.convert_to_tensor(map_payload["weights"], DTYPE)
    transition = tf.convert_to_tensor(map_payload["transition"], DTYPE)
    process = tf.convert_to_tensor(map_payload["process"], DTYPE)
    theta = tf.convert_to_tensor(map_payload["theta"], DTYPE)
    observation = tf.convert_to_tensor(map_payload["observations"][1], DTYPE)
    model = map_payload["model"]
    parent_count = int(parent_states.shape[0])
    uniforms = tf.random.stateless_uniform([int(row_count)], list(seed), dtype=DTYPE)
    cdf = tf.cumsum(parent_weights)
    parent_indices = tf.searchsorted(cdf, uniforms, side="right")
    parent_indices = tf.minimum(parent_indices, tf.cast(parent_count - 1, tf.int32))
    noise = tf.random.stateless_normal(
        [int(row_count), STATE_DIM], [seed[0], seed[1] + 1], dtype=DTYPE
    )
    transition_means = tf.linalg.matmul(parent_states, transition, transpose_b=True)
    process_chol = tf.linalg.cholesky(process)
    physical = tf.gather(transition_means, parent_indices) + tf.linalg.matmul(
        noise, process_chol, transpose_b=True
    )
    log_likelihood = model.observation_log_density(
        theta, physical, observation, 1
    )
    likelihood = tf.exp(log_likelihood)
    mean = tf.reduce_mean(likelihood)
    standard_error = tf.math.reduce_std(likelihood) / tf.sqrt(tf.cast(row_count, DTYPE))
    if not _finite(likelihood) or not _finite(mean) or not _finite(standard_error):
        raise ValueError("non-finite prior-predictive target audit")
    return {
        "z_t": mean,
        "standard_error": standard_error,
        "relative_standard_error": standard_error / mean,
        "physical_rows": physical,
        "likelihood_values": likelihood,
        "parent_indices": parent_indices,
    }


def _run_degree(
    degree: int,
    *,
    map_payload: Mapping[str, object],
    banks: Mapping[str, tf.Tensor],
    target_values: Mapping[str, tf.Tensor],
    log_target_values: Mapping[str, tf.Tensor],
    direct_target: Mapping[str, tf.Tensor],
    shift: tf.Tensor,
) -> Mapping[str, object]:
    convention = MeasureConvention(
        density_measure=DensityMeasure.REFERENCE_MEASURE,
        mass_measure=MassMeasure.REFERENCE_MEASURE,
        reference_weight_name="standard_normal",
        physical_coordinate_name="x",
        reference_coordinate_name="u",
    )
    basis = ProductBasis(
        [HermiteBasis1D(max_degree=int(degree)) for _ in range(STATE_DIM)],
        convention,
    )
    fit_cfg = _fit_config(degree)
    train_rows = banks["train"]
    holdout_rows = banks["holdout"]
    train_target = target_values["train"]
    holdout_target = target_values["holdout"]
    sample_weights = tf.fill([TRAIN_ROWS], tf.constant(1.0 / TRAIN_ROWS, DTYPE))
    holdout_weights = tf.fill([HOLDOUT_ROWS], tf.constant(1.0 / HOLDOUT_ROWS, DTYPE))
    samples = FixedTTFitSampleBatch(
        points=train_rows,
        target_values=train_target,
        weights=sample_weights,
        holdout_points=holdout_rows,
        holdout_values=holdout_target,
        holdout_weights=holdout_weights,
    )
    basis_dim = int(degree) + 1
    initial = _initial_tt_cores(STATE_DIM, basis_dim, TT_RANK)
    fitter = FixedTTFitter()
    fit_result = fitter.fit(
        basis,
        samples,
        fit_cfg,
        initial,
        branch_seed=f"phase5a-degree-{degree}",
        measure_convention=convention,
        initialization_rule="constant_channel_identity_v1",
    )
    fitted = fit_result.fitted_tt
    holdout_prediction = fitted.evaluate(holdout_rows)
    audit_prediction = fitted.evaluate(banks["audit"])
    holdout_residual = holdout_prediction - holdout_target
    shell_mask = tf.reduce_max(tf.abs(holdout_rows), axis=1) >= SHELL_RADIUS
    central_mask = tf.logical_not(shell_mask)
    if not bool(tf.reduce_any(shell_mask).numpy()) or not bool(tf.reduce_any(central_mask).numpy()):
        raise ValueError("holdout bank does not contain both central and shell rows")
    gram_scaled = _gram_squared_normalizer(fitted.cores, basis)
    # Z_T is the integral of h_star.  The primary estimate comes from the
    # independent predictive-mixture bank; the reference-bank estimate below
    # is retained solely as a tail-variance control.  Keeping log_h separate
    # from the square-root fitting target prevents a factor-of-two error.
    z_t = tf.convert_to_tensor(direct_target["z_t"], DTYPE)
    direct_se = tf.convert_to_tensor(direct_target["standard_error"], DTYPE)
    reference_scaled_values = tf.exp(log_target_values["audit"] - shift)
    reference_z_t = tf.exp(shift) * tf.reduce_mean(reference_scaled_values)
    reference_se = tf.exp(shift) * tf.math.reduce_std(reference_scaled_values) / tf.sqrt(
        tf.cast(AUDIT_ROWS, DTYPE)
    )
    z_h = tf.exp(shift) * gram_scaled
    log_z_h = tf.math.log(z_h)
    log_z_t = tf.math.log(z_t)
    max_abs_core = max(_max_abs(core.values) for core in fitted.cores)
    finite = all(
        _finite(value)
        for value in (
            holdout_prediction,
            audit_prediction,
            gram_scaled,
            z_h,
            z_t,
            direct_se,
            reference_z_t,
            reference_se,
        )
    )
    condition = _condition_summary(fit_result)
    condition_max = condition["scaled_augmented_condition_max"]
    status_ok = str(fit_result.status.value) == "OK"
    if condition_max is None:
        status_ok = False
    return {
        "degree": int(degree),
        "basis_family": "normalized_probabilists_hermite",
        "basis_dim_per_axis": basis_dim,
        "rank_tuple": list(fit_cfg.ranks),
        "ridge": RIDGE,
        "sweeps": ALS_SWEEPS,
        "fit_status": fit_result.status.value,
        "termination_reason": fit_result.termination_reason,
        "fit_residual": _scalar(fit_result.fit_residual),
        "holdout_residual": _scalar(fit_result.holdout_residual),
        "holdout_rms_recomputed": _rms(holdout_residual),
        "holdout_central_rms": _rms(tf.boolean_mask(holdout_residual, central_mask)),
        "holdout_shell_rms": _rms(tf.boolean_mask(holdout_residual, shell_mask)),
        "holdout_central_count": int(tf.reduce_sum(tf.cast(central_mask, tf.int32)).numpy()),
        "holdout_shell_count": int(tf.reduce_sum(tf.cast(shell_mask, tf.int32)).numpy()),
        "audit_prediction_rms": _rms(audit_prediction),
        "gram_normalizer_scaled": _scalar(gram_scaled),
        "z_h": _scalar(z_h),
        "z_t_direct": _scalar(z_t),
        "z_t_direct_standard_error": _scalar(direct_se),
        "z_t_direct_relative_standard_error": _scalar(direct_se / z_t),
        "z_t_reference_normal_bank": _scalar(reference_z_t),
        "z_t_reference_standard_error": _scalar(reference_se),
        "z_t_reference_relative_standard_error": _scalar(reference_se / reference_z_t),
        "log_z_h_minus_log_z_t": _scalar(log_z_h - log_z_t),
        "condition": condition,
        "realized_rank_tuple": list(fitted.rank_tuple()),
        "max_abs_core_value": max_abs_core,
        "finite": bool(finite),
        "hard_valid": bool(finite and status_ok and float(condition_max) <= CONDITION_VETO),
        "map_condition_number": _scalar(map_payload["map_condition_number"]),
        "map_minimum_eigenvalue": _scalar(map_payload["map_minimum_eigenvalue"]),
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
    lines = [
        "# C2 Mixture-UKF/APF Phase 5A Hermite Pilot Result",
        "",
        f"Status: `{payload['status']}`  ",
        f"Continuation: `{payload['continuation']}`  ",
        f"Failure class: `{payload['failure_class']}`",
        "",
        "## Decision",
        "",
        "| Decision | Criterion | Status | Interpretation |",
        "| --- | --- | --- | --- |",
        f"| Map and exact target | fixture parity, finite/SPD map and target | {'PASS' if payload['checks']['map_target_valid'] else 'VETO'} | {'representation results are interpretable' if payload['checks']['map_target_valid'] else 'repair map/target before continuing'} |",
        f"| Fixed Hermite fitter | all three degrees finite, `OK`, condition below veto | {'PASS' if payload['checks']['all_fits_valid'] else 'VETO'} | {'pilot mechanics passed' if payload['checks']['all_fits_valid'] else 'repair fitter or numerical contract'} |",
        "| Representation promotion | no promoted threshold in this pilot | NOT TESTED | residuals and normalizer gaps are descriptive nomination evidence only |",
        "",
        "## Inference status",
        "",
        "| Evidence class | Status |",
        "| --- | --- |",
        f"| Hard veto screen | {'passed' if payload['checks']['hard_vetoes_pass'] else 'failed'} |",
        "| Statistically supported ranking | not available from one bank and three deterministic fits |",
        "| Descriptive differences | degree-wise holdout, shell, Gram, and normalizer diagnostics below |",
        "| Default readiness | not assessed |",
        f"| Next evidence | {payload['next_phase_refresh']} |",
        "",
        "## Fixed target and map",
        "",
        f"Carried rows: `{payload['carried_rows']}`; training/holdout/audit rows: `{payload['train_rows']}/{payload['holdout_rows']}/{payload['audit_rows']}`; map condition: `{payload['map']['condition_number']:.6g}`; minimum eigenvalue: `{payload['map']['minimum_eigenvalue']:.6g}`; initial observation ESS: `{payload['initial_observation_ess']:.6g}`.",
        "",
        f"Fixture transition error: `{payload['map']['transition_fixture_error']:.3g}`; process covariance error: `{payload['map']['process_fixture_error']:.3g}`; stationary covariance error: `{payload['map']['stationary_fixture_error']:.3g}`; common log shift: `{payload['target_log_shift']:.6g}`.",
        "",
        "## Degree ladder",
        "",
        "| degree | fit RMS | holdout RMS | central RMS | shell RMS | log ZH - log ZT | predictive rel. SE | reference rel. SE | scaled cond. max | hard-valid |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |",
    ]
    for row in payload["degree_records"]:
        lines.append(
            f"| {row['degree']} | {row['fit_residual']:.4g} | {row['holdout_rms_recomputed']:.4g} | {row['holdout_central_rms']:.4g} | {row['holdout_shell_rms']:.4g} | {row['log_z_h_minus_log_z_t']:.4g} | {row['z_t_direct_relative_standard_error']:.3g} | {row['z_t_reference_relative_standard_error']:.3g} | {row['condition']['scaled_augmented_condition_max']:.4g} | {row['hard_valid']} |"
        )
    lines += [
        "",
        "`Z_T` is the independent prior-predictive audit-bank estimate of the exact finite target integral; it is not an analytic oracle. The Gaussian-reference estimate and its relative standard error are shown only as a tail-variance diagnostic. The degree comparisons are descriptive and do not establish superiority or convergence.",
        "",
        "## Red team",
        "",
        "Strongest alternative explanation: the finite prior-predictive bank still has Monte Carlo error, while the Gaussian-reference bank can underresolve rare tails. Both standard errors are reported. The weakest evidence is the single fixed map and one C2 transition; no recursive filtering or proposal ESS claim follows.",
        "",
    ]
    return "\n".join(lines) + "\n"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--jit-compile", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def run(args: argparse.Namespace) -> Path:
    output = _make_output_root(args.output_root)
    started = time.perf_counter()
    runtime = _configure_runtime()
    _load_modules()
    fixture = _load_fixture()
    map_payload = _model_and_map(fixture, jit_compile=bool(args.jit_compile))
    coordinate_map = map_payload["coordinate_map"]
    parent_states = map_payload["states"]
    parent_weights = map_payload["weights"]
    transition = map_payload["transition"]
    theta = map_payload["theta"]
    observation = map_payload["observations"][1]

    banks = {
        "train": _standard_normal_bank(TRAIN_ROWS, (SEED[0], SEED[1] + 1)),
        "holdout": _standard_normal_bank(HOLDOUT_ROWS, (SEED[0], SEED[1] + 2)),
        "audit": _standard_normal_bank(AUDIT_ROWS, (SEED[0], SEED[1] + 3)),
    }
    kernels = {
        name: _make_exact_target_kernel(
            parent_states=parent_states,
            parent_weights=parent_weights,
            transition=transition,
            theta=theta,
            observation=observation,
            sigma=float(fixture["sigma"]),
            row_count=int(rows.shape[0]),
            jit_compile=bool(args.jit_compile),
        )
        for name, rows in banks.items()
    }
    log_h: dict[str, tf.Tensor] = {}
    common_shift: tf.Tensor | None = None
    # Determine the shift from the training bank only, then apply it to every
    # bank.  This preserves holdout/audit separation while keeping values finite.
    physical_train, logdet_train = coordinate_map.forward(banks["train"])
    raw_log_gamma = kernels["train"](physical_train)
    log_eta_train = -0.5 * (
        tf.cast(STATE_DIM, DTYPE) * tf.constant(math.log(2.0 * math.pi), DTYPE)
        + tf.reduce_sum(tf.square(banks["train"]), axis=1)
    )
    common_shift = tf.reduce_logsumexp(raw_log_gamma + logdet_train - log_eta_train) - tf.math.log(
        tf.cast(TRAIN_ROWS, DTYPE)
    )
    sqrt_targets: dict[str, tf.Tensor] = {}
    for name, rows in banks.items():
        log_h[name], sqrt_targets[name], _ = _reference_target_values(
            rows,
            coordinate_map=coordinate_map,
            target_kernel=kernels[name],
            shift=common_shift,
        )
        if not _finite(log_h[name]) or not _finite(sqrt_targets[name]):
            raise ValueError(f"non-finite exact target values in {name} bank")

    predictive_audit = _prior_predictive_target_audit(
        map_payload, row_count=AUDIT_ROWS, seed=(SEED[0], SEED[1] + 4)
    )
    direct_target = {
        "z_t": predictive_audit["z_t"],
        "standard_error": predictive_audit["standard_error"],
    }

    degree_records = [
        _run_degree(
            degree,
            map_payload=map_payload,
            banks=banks,
            target_values=sqrt_targets,
            log_target_values=log_h,
            direct_target=direct_target,
            shift=common_shift,
        )
        for degree in DEGREES
    ]
    checks = {
        "source_files_present": all(path.is_file() for path in (FIXTURE_PATH, MODEL_PATH, MOMENT_MAP_PATH, PLAN_PATH)),
        "map_target_valid": (
            map_payload["transition_fixture_error"] <= 1.0e-12
            and map_payload["process_fixture_error"] <= 1.0e-12
            and map_payload["stationary_fixture_error"] <= 1.0e-12
            and _finite(map_payload["predicted_mean"])
            and _finite(map_payload["predicted_covariance"])
            and _finite(map_payload["map_minimum_eigenvalue"])
            and float(map_payload["map_minimum_eigenvalue"].numpy()) > 0.0
        ),
        "banks_disjoint_by_seed": True,
        "predictive_target_audit_finite": _finite(predictive_audit["likelihood_values"]),
        "all_fits_valid": all(bool(row["hard_valid"]) for row in degree_records),
    }
    checks["hard_vetoes_pass"] = all(bool(value) for value in checks.values())
    elapsed = time.perf_counter() - started
    sources = {
        "plan": {"path": str(PLAN_PATH.relative_to(ROOT)), "sha256": _sha256_file(PLAN_PATH)},
        "driver": {"path": str(Path(__file__).resolve().relative_to(ROOT)), "sha256": _sha256_file(Path(__file__).resolve())},
        "fixture": {"path": str(FIXTURE_PATH.relative_to(ROOT)), "sha256": _sha256_file(FIXTURE_PATH)},
        "model": {"path": str(MODEL_PATH.relative_to(ROOT)), "sha256": _sha256_file(MODEL_PATH)},
        "moment_map": {"path": str(MOMENT_MAP_PATH.relative_to(ROOT)), "sha256": _sha256_file(MOMENT_MAP_PATH)},
    }
    status = "PASS_PHASE5A_HERMITE_MECHANICS" if checks["hard_vetoes_pass"] else "VETO_PHASE5A_HERMITE_MECHANICS"
    payload: Mapping[str, object] = {
        "schema_version": RESULT_SCHEMA,
        "phase": PHASE_ID,
        "status": status,
        "continuation": "CONTINUE_RBF_HYBRID_DIAGNOSTIC" if checks["hard_vetoes_pass"] else "CONTINUATION_VETO_PHASE5A_VALIDITY",
        "failure_class": "none" if checks["hard_vetoes_pass"] else "implementation_or_numerical_validity",
        "next_phase_refresh": "fixed-map RBF and Hermite-plus-RBF arms if Hermite residuals remain poor; otherwise replicated recursive validation",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": elapsed,
        "environment": _environment(runtime, args),
        "seed": SEED,
        "map_seed": MAP_SEED,
        "route_id": ROUTE_ID,
        "route_classification": ROUTE_CLASSIFICATION,
        "carried_rows": CARRIED_ROWS,
        "train_rows": TRAIN_ROWS,
        "holdout_rows": HOLDOUT_ROWS,
        "audit_rows": AUDIT_ROWS,
        "degrees": DEGREES,
        "tt_rank": TT_RANK,
        "als_sweeps": ALS_SWEEPS,
        "ridge": RIDGE,
        "target_log_shift": _scalar(common_shift),
        "sources": sources,
        "workspace": _workspace_manifest(),
        "checks": checks,
        "map": {
            "condition_number": _scalar(map_payload["map_condition_number"]),
            "minimum_eigenvalue": _scalar(map_payload["map_minimum_eigenvalue"]),
            "raw_minimum_eigenvalue": _scalar(map_payload["map_raw_minimum_eigenvalue"]),
            "transition_fixture_error": map_payload["transition_fixture_error"],
            "process_fixture_error": map_payload["process_fixture_error"],
            "stationary_fixture_error": map_payload["stationary_fixture_error"],
        },
        "initial_observation_ess": map_payload["initial_observation_ess"],
        "bank_seeds": {
            "train": (SEED[0], SEED[1] + 1),
            "holdout": (SEED[0], SEED[1] + 2),
            "audit": (SEED[0], SEED[1] + 3),
            "predictive_audit": (SEED[0], SEED[1] + 4),
        },
        "bank_hashes": {
            name: hashlib.sha256(tf.io.serialize_tensor(rows).numpy()).hexdigest()
            for name, rows in banks.items()
        },
        "predictive_audit_hash": hashlib.sha256(
            tf.io.serialize_tensor(predictive_audit["physical_rows"]).numpy()
        ).hexdigest(),
        "z_t_direct": predictive_audit["z_t"],
        "z_t_direct_standard_error": predictive_audit["standard_error"],
        "z_t_direct_relative_standard_error": predictive_audit["relative_standard_error"],
        "z_t_reference_normal_bank": tf.exp(common_shift) * tf.reduce_mean(
            tf.exp(log_h["audit"] - common_shift)
        ),
        "z_t_reference_relative_standard_error": (
            tf.math.reduce_std(tf.exp(log_h["audit"] - common_shift))
            / tf.sqrt(tf.cast(AUDIT_ROWS, DTYPE))
            / tf.reduce_mean(tf.exp(log_h["audit"] - common_shift))
        ),
        "degree_records": degree_records,
        "nonclaims": [
            "no proposal-efficiency or ESS claim",
            "no posterior-correctness or pseudo-marginal claim",
            "no degree ranking or basis promotion claim",
            "no recursive multi-step filtering claim",
            "no analytical total-gradient or HMC claim",
            "no production/default-readiness claim",
        ],
    }
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "phase": PHASE_ID,
        "command": " ".join(sys.argv),
        "plan_sha256": sources["plan"]["sha256"],
        "sources": sources,
        "route_id": ROUTE_ID,
        "route_classification": ROUTE_CLASSIFICATION,
        "seed": SEED,
        "map_seed": MAP_SEED,
        "bank_seeds": payload["bank_seeds"],
        "bank_hashes": payload["bank_hashes"],
        "degrees": DEGREES,
        "tt_rank": TT_RANK,
        "als_sweeps": ALS_SWEEPS,
        "ridge": RIDGE,
        "environment": payload["environment"],
        "workspace": payload["workspace"],
        "evidence_contract": {
            "primary": "fixed-map exact-target representation diagnostics on disjoint banks",
            "hard_vetoes": "nonfinite target/map/fit, fixture mismatch, missing records, condition above veto",
            "descriptive": "heldout and shell residuals, direct Z_T versus exact Gram Z_H, condition, rank",
            "nonclaims": payload["nonclaims"],
        },
    }
    _write_json(output / "manifest.json", manifest)
    _write_json(output / "result.json", payload)
    _write_json(output / "records.json", {"degrees": degree_records})
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
