"""Target-side conditional-moment adapter for the frozen SSL-LSTM forecast API.

This module is a target-shaped wiring and covariance preflight utility.  It does
not load posterior/HMC artifacts and does not perform a G/H comparison.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from dataclasses import dataclass
from typing import Any

import tensorflow as tf

from bayesfilter.inference.predictive_equivalence import (
    ConditionalMeanLogVarianceInfluenceResult,
    MeanLogVarianceInfluenceResult,
    conditional_mean_log_variance_influence,
    mean_log_variance_influence,
    standardize_forecast_paths,
)
from bayesfilter.nonlinear.ssl_lstm_predictive_tf import (
    SSLLSTMForecastConfig,
    SSLLSTMForecastPaths,
)
from bayesfilter.nonlinear.ssl_lstm_sgqf_ukf_adapters import (
    make_ssl_lstm_svd_ukf_components,
)


HORIZON = 10
OBSERVATION_DIM = 1
TARGET_INTEGRATION_SCHEMA = "bayesfilter.ssl_lstm.target_integration.v1"
SOURCE_PATH = "bayesfilter/inference/ssl_lstm_target_integration.py"
FORECAST_SOURCE_PATH = "bayesfilter/nonlinear/ssl_lstm_predictive_tf.py"
ADAPTER_SOURCE_PATH = "bayesfilter/nonlinear/ssl_lstm_sgqf_ukf_adapters.py"


class TargetIntegrationError(ValueError):
    """Raised when the target integration contract cannot be satisfied."""


@dataclass(frozen=True)
class TargetScaleCalibration:
    center: tf.Tensor
    scale: tf.Tensor
    pooled_path_count: int
    chain_count: int
    draw_count: int
    replication_count: int
    seed_roots: tuple[tuple[int, int], ...]
    forecast_config_signature: str
    innovation_bank_signatures: tuple[str, ...]
    observation_path_hashes: tuple[str, ...]
    calibration_signature: str


@dataclass(frozen=True)
class ConditionalObservationMoments:
    means: tf.Tensor
    variances: tf.Tensor
    observation_stds: tf.Tensor
    source_signature: str


@dataclass(frozen=True)
class TargetFeatureComparison:
    path: MeanLogVarianceInfluenceResult
    conditional: ConditionalMeanLogVarianceInfluenceResult
    paired_feature_difference: tf.Tensor
    paired_standard_error: tf.Tensor
    paired_mcse_multiplier: float
    paired_pass: bool
    independent_feature_difference: tf.Tensor
    standardized_path_values: tf.Tensor
    standardized_conditional_means: tf.Tensor
    standardized_conditional_variances: tf.Tensor


def _raw_hash(tensor: tf.Tensor) -> str:
    values = tf.unstack(tf.reshape(tf.convert_to_tensor(tensor, tf.float64), [-1]))
    return hashlib.sha256(
        b"".join(struct.pack("<d", float(value)) for value in values)
    ).hexdigest()


def _canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _finite(value: tf.Tensor, name: str) -> tf.Tensor:
    tensor = tf.convert_to_tensor(value)
    if tensor.dtype != tf.float64:
        raise TargetIntegrationError(f"{name} must have dtype float64")
    if tensor.shape.rank is None or not tensor.shape.is_fully_defined():
        raise TargetIntegrationError(f"{name} must have a fully defined shape")
    try:
        tf.debugging.assert_all_finite(tensor, f"{name} must be finite")
    except tf.errors.InvalidArgumentError as exc:
        raise TargetIntegrationError(f"{name} must be finite") from exc
    return tensor


def _validate_paths(paths: SSLLSTMForecastPaths) -> None:
    if not isinstance(paths, SSLLSTMForecastPaths):
        raise TypeError("paths must be SSLLSTMForecastPaths")
    observations = _finite(paths.observations, "observations")
    means = _finite(paths.observation_means, "observation_means")
    if observations.shape.rank != 4 or means.shape != observations.shape:
        raise TargetIntegrationError("observation tensors must share [draw,rep,10,1]")
    if observations.shape[-2:] != (HORIZON, OBSERVATION_DIM):
        raise TargetIntegrationError("the frozen target requires [draw,rep,10,1]")
    if paths.provenance.forecast_horizon != HORIZON:
        raise TargetIntegrationError("forecast horizon provenance is not ten")
    if paths.provenance.dtype != "float64":
        raise TargetIntegrationError("forecast provenance dtype is not float64")
    if paths.provenance.cluster_unit != "complete_ten_step_path_per_draw_replication":
        raise TargetIntegrationError("forecast cluster unit is not a complete path")
    if _raw_hash(paths.terminal.full_parameters) != paths.provenance.embedded_full_parameter_matrix_raw_sha256:
        raise TargetIntegrationError("embedded full-parameter provenance mismatch")
    reconstruction_scale = tf.maximum(
        tf.constant(1.0, tf.float64), tf.reduce_max(tf.abs(observations))
    )
    reconstruction_tolerance = (
        tf.constant(512.0 * 2.220446049250313e-16, tf.float64)
        * reconstruction_scale
    )
    if not bool(
        tf.reduce_all(
            tf.abs(means + paths.observation_innovations - observations)
            <= reconstruction_tolerance
        )
    ):
        raise TargetIntegrationError("observation mean/noise reconstruction mismatch")


def conditional_observation_moments(
    paths: SSLLSTMForecastPaths,
    config: SSLLSTMForecastConfig,
) -> ConditionalObservationMoments:
    """Return observation means and conditional observation variances.

    Process and terminal uncertainty remain in ``paths.observation_means``;
    only the additive observation-noise covariance is integrated analytically.
    """

    _validate_paths(paths)
    if not isinstance(config, SSLLSTMForecastConfig):
        raise TypeError("config must be SSLLSTMForecastConfig")
    if paths.provenance.forecast_config_signature != config.signature():
        raise TargetIntegrationError("forecast config provenance mismatch")
    full_parameters = _finite(paths.terminal.full_parameters, "full_parameters")
    if full_parameters.shape.rank != 2 or full_parameters.shape[0] != paths.observations.shape[0]:
        raise TargetIntegrationError("full parameter rows do not match draw count")
    variance_rows = []
    std_rows = []
    for index in range(int(full_parameters.shape[0])):
        components = make_ssl_lstm_svd_ukf_components(
            full_parameters[index],
            config.posterior_config.static_config,
            evidence_path="docs/plans/bayesfilter-ssl-lstm-completion-phase-a2-terminal-state-forecast-api-result-2026-07-11.md",
            std_floor=config.posterior_config.std_floor,
        )
        std = _finite(components.parameters.observation_std, "observation_std")
        if std.shape != (OBSERVATION_DIM,) or bool(tf.reduce_any(std <= 0.0)):
            raise TargetIntegrationError("observation standard deviation is invalid")
        std_rows.append(std)
        variance_rows.append(tf.square(std))
    stds = tf.stack(std_rows, axis=0)
    variances = tf.squeeze(tf.stack(variance_rows, axis=0), axis=-1)
    means = tf.squeeze(paths.observation_means, axis=-1)
    variances = tf.broadcast_to(variances[:, tf.newaxis, tf.newaxis], tf.shape(means))
    return ConditionalObservationMoments(
        means=_finite(means, "conditional means"),
        variances=_finite(variances, "conditional variances"),
        observation_stds=stds[:, 0],
        source_signature=_canonical_hash(
            {
                "schema": TARGET_INTEGRATION_SCHEMA,
                "adapter_source": ADAPTER_SOURCE_PATH,
                "forecast_source": FORECAST_SOURCE_PATH,
                "std_floor": float(config.posterior_config.std_floor).hex(),
                "variance_formula": "observation_std**2",
                "feature_order": "mean_h_then_log_variance_h",
            }
        ),
    )


def calibrate_horizon_scales(
    paths_by_chain: tuple[SSLLSTMForecastPaths, ...],
    *,
    seed_roots: tuple[tuple[int, int], ...],
) -> TargetScaleCalibration:
    """Freeze per-horizon path center and unbiased standard-deviation scales."""

    if not paths_by_chain:
        raise TargetIntegrationError("at least one calibration chain is required")
    if len(paths_by_chain) != len(seed_roots):
        raise TargetIntegrationError("one seed root is required per calibration chain")
    for paths in paths_by_chain:
        _validate_paths(paths)
    recorded_seed_roots = tuple(paths.provenance.innovation_root_seed for paths in paths_by_chain)
    if recorded_seed_roots != seed_roots:
        raise TargetIntegrationError("declared calibration seeds do not match forecast provenance")
    config_signatures = {
        paths.provenance.forecast_config_signature for paths in paths_by_chain
    }
    if len(config_signatures) != 1:
        raise TargetIntegrationError("calibration forecast config signatures differ")
    bank_signature_set = {
        paths.provenance.innovation_bank_signature for paths in paths_by_chain
    }
    if len(bank_signature_set) != len(paths_by_chain):
        raise TargetIntegrationError("calibration innovation banks must be unique")
    observations = tf.concat(
        [tf.squeeze(paths.observations, axis=-1) for paths in paths_by_chain], axis=0
    )
    count = int(tf.size(observations).numpy() // HORIZON)
    if count < 2:
        raise TargetIntegrationError("calibration bank must contain at least two paths")
    center = tf.reduce_mean(observations, axis=[0, 1])
    centered = observations - center
    scale = tf.sqrt(
        tf.reduce_sum(tf.square(centered), axis=[0, 1])
        / tf.cast(count - 1, tf.float64)
    )
    _finite(center, "calibration center")
    _finite(scale, "calibration scale")
    if bool(tf.reduce_any(scale <= 0.0)):
        raise TargetIntegrationError("calibration scales must be strictly positive")
    signatures = tuple(paths.provenance.innovation_bank_signature for paths in paths_by_chain)
    hashes = tuple(_raw_hash(tf.squeeze(paths.observations, axis=-1)) for paths in paths_by_chain)
    payload = {
        "schema": TARGET_INTEGRATION_SCHEMA,
        "center_hash": _raw_hash(center),
        "scale_hash": _raw_hash(scale),
        "seed_roots": [list(seed) for seed in seed_roots],
        "forecast_config_signature": paths_by_chain[0].provenance.forecast_config_signature,
        "innovation_bank_signatures": list(signatures),
        "observation_path_hashes": list(hashes),
        "pooled_path_count": count,
    }
    return TargetScaleCalibration(
        center=center,
        scale=scale,
        pooled_path_count=count,
        chain_count=len(paths_by_chain),
        draw_count=int(paths_by_chain[0].observations.shape[0]),
        replication_count=int(paths_by_chain[0].observations.shape[1]),
        seed_roots=seed_roots,
        forecast_config_signature=paths_by_chain[0].provenance.forecast_config_signature,
        innovation_bank_signatures=signatures,
        observation_path_hashes=hashes,
        calibration_signature=_canonical_hash(payload),
    )


def compare_path_and_conditional_moments(
    paths_by_chain: tuple[SSLLSTMForecastPaths, ...],
    calibration: TargetScaleCalibration,
    config: SSLLSTMForecastConfig,
    *,
    jit_compile: bool,
    paired_mcse_multiplier: float = 6.0,
    independent_paths_by_chain: tuple[SSLLSTMForecastPaths, ...] | None = None,
) -> TargetFeatureComparison:
    """Compare path and Rao features without making a method-ranking claim."""

    if type(paired_mcse_multiplier) is not float or not math.isfinite(paired_mcse_multiplier):
        raise TargetIntegrationError("paired_mcse_multiplier must be finite float")
    if paired_mcse_multiplier <= 0.0:
        raise TargetIntegrationError("paired_mcse_multiplier must be positive")
    if calibration.chain_count != len(paths_by_chain):
        raise TargetIntegrationError("calibration chain count mismatch")
    standardized_paths = []
    standardized_means = []
    standardized_variances = []
    for paths in paths_by_chain:
        moments = conditional_observation_moments(paths, config)
        values = tf.squeeze(paths.observations, axis=-1)
        standardized_paths.append(
            standardize_forecast_paths(
                values,
                calibration.center,
                calibration.scale,
                scale_floor=tf.constant(2.0**-40, tf.float64),
                jit_compile=jit_compile,
                allow_floor_use=False,
            )
        )
        standardized_means.append(
            (moments.means - calibration.center) / calibration.scale
        )
        standardized_variances.append(moments.variances / tf.square(calibration.scale))
    path_values = tf.stack(standardized_paths, axis=0)
    means = tf.stack(standardized_means, axis=0)
    variances = tf.stack(standardized_variances, axis=0)
    path_result = mean_log_variance_influence(path_values, jit_compile=jit_compile)
    conditional_result = conditional_mean_log_variance_influence(
        means, variances, jit_compile=jit_compile
    )
    path_cluster = tf.reshape(path_result.influence_values, [-1, 20])
    conditional_cluster = tf.reshape(conditional_result.influence_values, [-1, 20])
    differences = path_cluster - conditional_cluster
    count = int(differences.shape[0])
    if count < 2:
        raise TargetIntegrationError("paired diagnostic needs at least two clusters")
    feature_difference = path_result.feature_estimate - conditional_result.feature_estimate
    se = tf.sqrt(
        tf.reduce_sum(tf.square(differences - tf.reduce_mean(differences, axis=0)), axis=0)
        / tf.cast(count * (count - 1), tf.float64)
    )
    finite = bool(tf.reduce_all(tf.math.is_finite(feature_difference)))
    paired_pass = finite and bool(
        tf.reduce_all(
            tf.abs(feature_difference)
            <= tf.cast(paired_mcse_multiplier, tf.float64)
            * tf.maximum(se, tf.constant(1.0e-12, tf.float64))
        )
    )
    independent_difference = tf.zeros([20], tf.float64)
    if independent_paths_by_chain is not None:
        if len(independent_paths_by_chain) != len(paths_by_chain):
            raise TargetIntegrationError("independent robustness chain count mismatch")
        independent_means = []
        independent_variances = []
        for independent_paths in independent_paths_by_chain:
            moments = conditional_observation_moments(independent_paths, config)
            independent_means.append(
                (moments.means - calibration.center) / calibration.scale
            )
            independent_variances.append(
                moments.variances / tf.square(calibration.scale)
            )
        independent_result = conditional_mean_log_variance_influence(
            tf.stack(independent_means, axis=0),
            tf.stack(independent_variances, axis=0),
            jit_compile=jit_compile,
        )
        independent_difference = (
            conditional_result.feature_estimate - independent_result.feature_estimate
        )
    return TargetFeatureComparison(
        path=path_result,
        conditional=conditional_result,
        paired_feature_difference=feature_difference,
        paired_standard_error=se,
        paired_mcse_multiplier=paired_mcse_multiplier,
        paired_pass=paired_pass,
        independent_feature_difference=independent_difference,
        standardized_path_values=path_values,
        standardized_conditional_means=means,
        standardized_conditional_variances=variances,
    )
