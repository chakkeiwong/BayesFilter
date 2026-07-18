"""Dependence-aware predictive-law statistics for the SSL-LSTM validation lane.

The routines in this module validate statistical machinery.  A ``PASS`` returned
by :func:`classify_predictive_evidence` is only a branch of that machinery; it is
not, by itself, evidence that two fitted models are predictively equivalent.
"""

from __future__ import annotations

import hashlib
import math
import weakref
from dataclasses import dataclass, field
from typing import Any, Literal

import tensorflow as tf
import tensorflow_probability as tfp


DecisionStatus = Literal[
    "PASS",
    "MATERIAL_DIFFERENCE",
    "INCONCLUSIVE_UNDERPOWERED",
    "INVALID_HARD_VETO",
]
SamplingContract = Literal[
    "iid_oracle_fixture",
    "dependent_descriptive_only",
    "paired_diagnostic_shared",
]


class PredictiveContractError(ValueError):
    """Raised when an input cannot satisfy the declared statistical contract."""


@dataclass(frozen=True)
class PredictiveStatisticsConfig:
    horizon: int = 10
    quantile_probabilities: tuple[float, ...] = (0.05, 0.25, 0.50, 0.75, 0.95)
    central_moment_orders: tuple[int, ...] = (3, 4)
    jit_compile: bool = True


@dataclass(frozen=True)
class PredictiveSummary:
    means: tf.Tensor
    variances: tf.Tensor
    log_variances: tf.Tensor
    central_moments: tf.Tensor
    quantiles: tf.Tensor
    cross_horizon_covariance: tf.Tensor
    path_count: tf.Tensor
    status: tf.Tensor


@dataclass(frozen=True)
class MMDStatistics:
    squared_mmd_u: tf.Tensor
    squared_mmd_v_biased: tf.Tensor
    per_bandwidth_u: tf.Tensor
    per_bandwidth_v_biased: tf.Tensor
    bandwidths: tf.Tensor
    mixture_weights: tf.Tensor
    sampling_contract: SamplingContract
    iid_samples_verified: bool
    independent_arm_banks_verified: bool
    inference_admissible: bool
    status: tf.Tensor
    construction_signature: tf.Tensor = field(
        default_factory=lambda: tf.constant("", tf.string),
        repr=False,
        compare=False,
    )


@dataclass(frozen=True)
class CrossChainLinearMMD:
    squared_mmd_linear: tf.Tensor
    kernel_contrast_sequence: tf.Tensor
    chain_pair_schedule: tf.Tensor
    independent_arm_banks_verified: bool
    stationarity_required: bool
    stationarity_verified: bool
    mixing_verified: bool
    mechanics_only: bool
    inference_admissible: bool
    status: tf.Tensor
    construction_signature: tf.Tensor = field(
        default_factory=lambda: tf.constant("", tf.string),
        repr=False,
        compare=False,
    )


@dataclass(frozen=True)
class HierarchicalBootstrapIndices:
    chain_indices: tf.Tensor
    draw_indices: tf.Tensor
    forecast_replication_indices: tf.Tensor
    block_length: int
    block_mode: str
    chain_mode: str
    seed: tf.Tensor
    status: tf.Tensor


@dataclass(frozen=True)
class SimultaneousIntervals:
    estimate: tf.Tensor
    lower: tf.Tensor
    upper: tf.Tensor
    standard_error: tf.Tensor
    critical_value: tf.Tensor
    alpha: tf.Tensor
    method: Literal["bonferroni_studentized", "bootstrap_max_statistic"]
    inference_admissible: bool
    status: tf.Tensor
    construction_signature: tf.Tensor = field(
        default_factory=lambda: tf.constant("", tf.string),
        repr=False,
        compare=False,
    )


@dataclass(frozen=True)
class MMDInterval:
    estimate: tf.Tensor
    lower: tf.Tensor
    upper: tf.Tensor
    standard_error: tf.Tensor
    critical_value: tf.Tensor
    alpha: tf.Tensor
    block_count: tf.Tensor
    inference_admissible: bool
    status: tf.Tensor
    construction_signature: tf.Tensor = field(
        default_factory=lambda: tf.constant("", tf.string),
        repr=False,
        compare=False,
    )


@dataclass(frozen=True)
class PredictiveDecision:
    status: DecisionStatus
    primary_interval_status: DecisionStatus
    mmd_upper_bound_status: DecisionStatus
    hard_veto_codes: tuple[str, ...]
    explanatory_diagnostics: dict[str, tf.Tensor]


@dataclass(frozen=True)
class LongRunCovarianceResult:
    spectral_covariance: tf.Tensor
    pooled_mean_covariance: tf.Tensor
    regularized_covariance: tf.Tensor
    precision: tf.Tensor
    ridge_ladder: tf.Tensor
    selected_ridge_index: tf.Tensor
    selected_ridge_multiplier: tf.Tensor
    selected_diagonal_loading: tf.Tensor
    eigenvalues: tf.Tensor
    condition_number: tf.Tensor
    block_length: int
    batch_count: int
    chain_count: int
    draw_count: int
    inference_admissible: bool
    status: tf.Tensor
    construction_signature: tf.Tensor = field(
        default_factory=lambda: tf.constant("", tf.string),
        repr=False,
        compare=False,
    )


@dataclass(frozen=True)
class BartlettLongRunCovarianceResult:
    spectral_covariance: tf.Tensor
    pooled_mean_covariance: tf.Tensor
    regularized_covariance: tf.Tensor
    precision: tf.Tensor
    ridge_ladder: tf.Tensor
    selected_ridge_index: tf.Tensor
    selected_ridge_multiplier: tf.Tensor
    selected_diagonal_loading: tf.Tensor
    eigenvalues: tf.Tensor
    condition_number: tf.Tensor
    bandwidth: int
    bandwidth_multiplier: float
    chain_count: int
    draw_count: int
    numerically_admissible: bool
    inference_admissible: bool
    status: tf.Tensor
    construction_signature: tf.Tensor = field(
        default_factory=lambda: tf.constant("", tf.string),
        repr=False,
        compare=False,
    )


@dataclass(frozen=True)
class BatchedBartlettLongRunCovarianceResult:
    spectral_covariance: tf.Tensor
    pooled_mean_covariance: tf.Tensor
    regularized_covariance: tf.Tensor
    precision: tf.Tensor
    ridge_ladder: tf.Tensor
    selected_ridge_index: tf.Tensor
    selected_ridge_multiplier: tf.Tensor
    selected_diagonal_loading: tf.Tensor
    eigenvalues: tf.Tensor
    condition_number: tf.Tensor
    bandwidth: int
    bandwidth_multiplier: float
    batch_size: int
    chain_count: int
    draw_count: int
    numerically_admissible: tf.Tensor
    inference_admissible: tf.Tensor
    status: tf.Tensor
    construction_signature: tf.Tensor = field(
        default_factory=lambda: tf.constant("", tf.string),
        repr=False,
        compare=False,
    )


@dataclass(frozen=True)
class ProperScoreLoss:
    horizon_weights: tf.Tensor
    loss_matrix: tf.Tensor
    horizon_count: int
    status: tf.Tensor
    construction_signature: tf.Tensor = field(
        default_factory=lambda: tf.constant("", tf.string),
        repr=False,
        compare=False,
    )


@dataclass(frozen=True)
class QuadraticLossBounds:
    estimate: tf.Tensor
    covariance: tf.Tensor
    confidence_radius_squared: tf.Tensor
    point_loss: tf.Tensor
    lower_bound: tf.Tensor
    upper_bound: tf.Tensor
    lower_optimizer: tf.Tensor
    upper_optimizer: tf.Tensor
    lower_kkt_residual: tf.Tensor
    upper_kkt_residual: tf.Tensor
    covariance_eigenvalues: tf.Tensor
    inference_admissible: bool
    status: tf.Tensor
    construction_signature: tf.Tensor = field(
        default_factory=lambda: tf.constant("", tf.string),
        repr=False,
        compare=False,
    )


@dataclass(frozen=True)
class BatchedQuadraticLossBounds:
    estimate: tf.Tensor
    covariance: tf.Tensor
    loss_matrices: tf.Tensor
    confidence_radius_squared: tf.Tensor
    point_loss: tf.Tensor
    lower_bound: tf.Tensor
    upper_bound: tf.Tensor
    lower_kkt_residual: tf.Tensor
    upper_kkt_residual: tf.Tensor
    covariance_eigenvalues: tf.Tensor
    inference_admissible: tf.Tensor
    status: tf.Tensor
    construction_signature: tf.Tensor = field(
        default_factory=lambda: tf.constant("", tf.string),
        repr=False,
        compare=False,
    )


@dataclass(frozen=True)
class ProperScoreDecision:
    status: DecisionStatus
    loss_lower_bound: tf.Tensor
    loss_upper_bound: tf.Tensor
    acceptable_loss: tf.Tensor
    hard_veto_codes: tuple[str, ...]
    explanatory_diagnostics: dict[str, tf.Tensor]


@dataclass(frozen=True)
class DualProperScoreDecision:
    status: DecisionStatus
    average_loss_lower_bound: tf.Tensor
    average_loss_upper_bound: tf.Tensor
    horizon_loss_lower_bounds: tf.Tensor
    horizon_loss_upper_bounds: tf.Tensor
    acceptable_average_loss: tf.Tensor
    acceptable_horizon_loss: tf.Tensor
    hard_veto_codes: tuple[str, ...]
    explanatory_diagnostics: dict[str, tf.Tensor]


@dataclass(frozen=True)
class MeanLogVarianceInfluenceResult:
    feature_estimate: tf.Tensor
    standardized_means: tf.Tensor
    log_variances: tf.Tensor
    influence_values: tf.Tensor
    chain_count: int
    draw_count: int
    forecast_replication_count: int
    path_count: int
    status: tf.Tensor
    construction_signature: tf.Tensor = field(
        default_factory=lambda: tf.constant("", tf.string),
        repr=False,
        compare=False,
    )


@dataclass(frozen=True)
class ConditionalMeanLogVarianceInfluenceResult:
    feature_estimate: tf.Tensor
    standardized_means: tf.Tensor
    log_variances: tf.Tensor
    influence_values: tf.Tensor
    conditional_second_moments: tf.Tensor
    chain_count: int
    draw_count: int
    forecast_replication_count: int
    cluster_count: int
    status: tf.Tensor
    construction_signature: tf.Tensor = field(
        default_factory=lambda: tf.constant("", tf.string),
        repr=False,
        compare=False,
    )


@dataclass(frozen=True)
class SplitQuadraticLossBounds:
    estimate: tf.Tensor
    covariance: tf.Tensor
    average_point_loss: tf.Tensor
    average_lower_bound: tf.Tensor
    average_upper_bound: tf.Tensor
    horizon_point_losses: tf.Tensor
    horizon_lower_bounds: tf.Tensor
    horizon_upper_bounds: tf.Tensor
    average_confidence_radius_squared: tf.Tensor
    horizon_confidence_radii_squared: tf.Tensor
    average_lower_kkt_residual: tf.Tensor
    average_upper_kkt_residual: tf.Tensor
    horizon_lower_kkt_residuals: tf.Tensor
    horizon_upper_kkt_residuals: tf.Tensor
    average_alpha: tf.Tensor
    horizon_alphas: tf.Tensor
    allocated_familywise_alpha: tf.Tensor
    inference_admissible: bool
    status: tf.Tensor
    construction_signature: tf.Tensor = field(
        default_factory=lambda: tf.constant("", tf.string),
        repr=False,
        compare=False,
    )


@dataclass(frozen=True)
class BatchedSplitQuadraticLossBounds:
    estimate: tf.Tensor
    covariance: tf.Tensor
    average_point_loss: tf.Tensor
    average_lower_bound: tf.Tensor
    average_upper_bound: tf.Tensor
    horizon_point_losses: tf.Tensor
    horizon_lower_bounds: tf.Tensor
    horizon_upper_bounds: tf.Tensor
    average_confidence_radius_squared: tf.Tensor
    horizon_confidence_radius_squared: tf.Tensor
    average_lower_kkt_residual: tf.Tensor
    average_upper_kkt_residual: tf.Tensor
    horizon_lower_kkt_residuals: tf.Tensor
    horizon_upper_kkt_residuals: tf.Tensor
    average_alpha: tf.Tensor
    horizon_alpha: tf.Tensor
    allocated_familywise_alpha: tf.Tensor
    inference_admissible: tf.Tensor
    status: tf.Tensor
    construction_signature: tf.Tensor = field(
        default_factory=lambda: tf.constant("", tf.string),
        repr=False,
        compare=False,
    )


@dataclass(frozen=True)
class PairwiseDistanceScaleResult:
    median_distance: tf.Tensor
    positive_pair_count: tf.Tensor
    total_pair_count: tf.Tensor
    path_count: int
    status: tf.Tensor
    construction_signature: tf.Tensor = field(
        default_factory=lambda: tf.constant("", tf.string),
        repr=False,
        compare=False,
    )


_VALID = "VALID"
_INVALID = "INVALID_HARD_VETO"
_CONSTRUCTION_REGISTRY: dict[
    int, tuple[weakref.ReferenceType[object], str, str, str]
] = {}


def _update_fingerprint(digest: Any, value: Any) -> None:
    if tf.is_tensor(value):
        digest.update(b"tensor\0")
        digest.update(value.dtype.name.encode("ascii") + b"\0")
        digest.update(repr(value.shape.as_list()).encode("ascii") + b"\0")
        digest.update(bytes(tf.io.serialize_tensor(value).numpy()))
    elif value is None:
        digest.update(b"none\0")
    elif type(value) is bool:
        digest.update(b"bool\0" + (b"1" if value else b"0"))
    elif type(value) is int:
        digest.update(b"int\0" + str(value).encode("ascii") + b"\0")
    elif type(value) is float:
        digest.update(b"float\0" + value.hex().encode("ascii") + b"\0")
    elif type(value) is str:
        digest.update(b"str\0" + value.encode("utf-8") + b"\0")
    elif isinstance(value, (tuple, list)):
        digest.update(b"sequence\0" + str(len(value)).encode("ascii") + b"\0")
        for item in value:
            _update_fingerprint(digest, item)
    else:
        raise PredictiveContractError(
            f"unsupported construction fingerprint value: {type(value).__name__}"
        )


def _fingerprint(kind: str, values: tuple[tuple[str, Any], ...]) -> str:
    digest = hashlib.sha256()
    _update_fingerprint(digest, kind)
    for name, value in values:
        _update_fingerprint(digest, name)
        _update_fingerprint(digest, value)
    return digest.hexdigest()


def _result_fingerprint(kind: str, result: object) -> str:
    return _fingerprint(
        f"{kind}.result.v1",
        tuple(sorted(vars(result).items(), key=lambda item: item[0])),
    )


def _seal_result(
    result: Any,
    *,
    kind: str,
    provenance: tuple[tuple[str, Any], ...],
) -> Any:
    signature = _fingerprint(f"{kind}.construction.v1", provenance)
    object.__setattr__(result, "construction_signature", tf.constant(signature))
    output_fingerprint = _result_fingerprint(kind, result)
    key = id(result)

    def remove(reference: weakref.ReferenceType[object]) -> None:
        current = _CONSTRUCTION_REGISTRY.get(key)
        if current is not None and current[0] is reference:
            _CONSTRUCTION_REGISTRY.pop(key, None)

    reference = weakref.ref(result, remove)
    _CONSTRUCTION_REGISTRY[key] = (
        reference,
        kind,
        signature,
        output_fingerprint,
    )
    return result


def _authenticated_result(result: object, *, kind: str) -> bool:
    record = _CONSTRUCTION_REGISTRY.get(id(result))
    if record is None or record[0]() is not result or record[1] != kind:
        return False
    signature = getattr(result, "construction_signature", None)
    if not _has_exact_status(signature, record[2]):
        return False
    return record[3] == _result_fingerprint(kind, result)


def _require_tensor(value: tf.Tensor, name: str, *, rank: int | None = None) -> tf.Tensor:
    if not tf.is_tensor(value):
        raise PredictiveContractError(f"{name} must be a TensorFlow tensor")
    if value.dtype != tf.float64:
        raise PredictiveContractError(f"{name} must have dtype tf.float64")
    if rank is not None and value.shape.rank != rank:
        raise PredictiveContractError(f"{name} must have rank {rank}")
    if value.shape.rank is None or not value.shape.is_fully_defined():
        raise PredictiveContractError(f"{name} must have fully defined static shape")
    try:
        tf.debugging.assert_all_finite(value, f"{name} must be finite")
    except tf.errors.InvalidArgumentError as exc:
        raise PredictiveContractError(f"{name} must be finite") from exc
    return value


def _require_probability(value: tf.Tensor | float, name: str) -> tf.Tensor:
    if not tf.is_tensor(value) and type(value) is not float:
        raise PredictiveContractError(f"{name} must be a float or float64 tensor")
    tensor = value if tf.is_tensor(value) else tf.constant(value, tf.float64)
    tensor = _require_tensor(tensor, name, rank=0)
    scalar = tf.get_static_value(tensor)
    if scalar is None:
        raise PredictiveContractError(f"{name} must be statically known")
    if not 0.0 < float(scalar) < 1.0:
        raise PredictiveContractError(f"{name} must lie strictly between zero and one")
    return tensor


def _status_text(status: tf.Tensor) -> str:
    value = tf.get_static_value(status)
    if isinstance(value, bytes):
        return value.decode("ascii")
    return str(value)


def _has_exact_status(status: tf.Tensor, expected: str) -> bool:
    if (
        not tf.is_tensor(status)
        or status.dtype != tf.string
        or status.shape.rank != 0
    ):
        return False
    value = tf.get_static_value(status)
    return isinstance(value, bytes) and value == expected.encode("ascii")


def _has_declared_status(status: tf.Tensor) -> bool:
    return _has_exact_status(status, _VALID) or _has_exact_status(status, _INVALID)


def _scale_aware_equal(left: tf.Tensor, right: tf.Tensor) -> bool:
    scale = tf.maximum(
        tf.constant(1.0, tf.float64),
        tf.maximum(tf.reduce_max(tf.abs(left)), tf.reduce_max(tf.abs(right))),
    )
    tolerance = tf.constant(512.0 * 2.220446049250313e-16, tf.float64) * scale
    return bool(tf.reduce_all(tf.abs(left - right) <= tolerance))


def _require_bool(value: bool, name: str) -> bool:
    if type(value) is not bool:
        raise PredictiveContractError(f"{name} must be a Python bool")
    return value


def adapt_ssl_lstm_observations(
    observations: tf.Tensor,
    *,
    horizon: int = 10,
) -> tf.Tensor:
    """Adapt A2 ``[draw, replication, horizon, 1]`` observations to A3 shape."""

    if type(horizon) is not int:
        raise PredictiveContractError("horizon must be an integer")
    observations = _require_tensor(observations, "observations", rank=4)
    if horizon != 10:
        raise PredictiveContractError("the A3 horizon is fixed at 10")
    if observations.shape[-2] != horizon or observations.shape[-1] != 1:
        raise PredictiveContractError(
            "observations must have shape [draw, replication, 10, 1]"
        )
    return tf.expand_dims(tf.squeeze(observations, axis=-1), axis=0)


def _validate_summary_config(config: PredictiveStatisticsConfig) -> None:
    if type(config) is not PredictiveStatisticsConfig:
        raise PredictiveContractError("config must be PredictiveStatisticsConfig")
    _require_bool(config.jit_compile, "config.jit_compile")
    if type(config.horizon) is not int:
        raise PredictiveContractError("config.horizon must be an integer")
    if config.horizon != 10:
        raise PredictiveContractError("the A3 horizon is fixed at 10")
    if type(config.quantile_probabilities) is not tuple or any(
        type(probability) is not float or not math.isfinite(probability)
        for probability in config.quantile_probabilities
    ):
        raise PredictiveContractError(
            "quantile_probabilities must be a tuple of finite Python floats"
        )
    if not config.quantile_probabilities:
        raise PredictiveContractError("at least one quantile probability is required")
    if any(not 0.0 < probability < 1.0 for probability in config.quantile_probabilities):
        raise PredictiveContractError("quantile probabilities must be in (0, 1)")
    if tuple(sorted(set(config.quantile_probabilities))) != config.quantile_probabilities:
        raise PredictiveContractError("quantile probabilities must be unique and sorted")
    if type(config.central_moment_orders) is not tuple or any(
        type(order) is not int or order > 2**31 - 1
        for order in config.central_moment_orders
    ):
        raise PredictiveContractError(
            "central_moment_orders must be a tuple of int32-compatible Python integers"
        )
    if not config.central_moment_orders or any(order < 3 for order in config.central_moment_orders):
        raise PredictiveContractError("central moment orders must be at least three")
    if len(set(config.central_moment_orders)) != len(config.central_moment_orders):
        raise PredictiveContractError("central moment orders must be unique")


def _linear_quantiles(sorted_values: tf.Tensor, probabilities: tf.Tensor) -> tf.Tensor:
    count = tf.shape(sorted_values, out_type=tf.int32)[0]
    positions = probabilities * tf.cast(count - 1, tf.float64)
    lower_index = tf.cast(tf.floor(positions), tf.int32)
    upper_index = tf.cast(tf.math.ceil(positions), tf.int32)
    fraction = positions - tf.floor(positions)
    lower = tf.gather(sorted_values, lower_index, axis=0)
    upper = tf.gather(sorted_values, upper_index, axis=0)
    return lower + fraction[:, tf.newaxis] * (upper - lower)


def _summary_kernel(
    paths: tf.Tensor,
    probabilities: tf.Tensor,
    orders: tf.Tensor,
) -> tuple[tf.Tensor, ...]:
    horizon = tf.shape(paths, out_type=tf.int32)[-1]
    flat = tf.reshape(paths, [-1, horizon])
    count = tf.shape(flat, out_type=tf.int32)[0]
    means = tf.reduce_mean(flat, axis=0)
    centered = flat - means
    denominator = tf.cast(count - 1, tf.float64)
    variances = tf.reduce_sum(tf.square(centered), axis=0) / denominator
    covariance = tf.matmul(centered, centered, transpose_a=True) / denominator
    moments = tf.map_fn(
        lambda order: tf.reduce_mean(
            tf.pow(centered, tf.cast(order, tf.float64)), axis=0
        ),
        orders,
        fn_output_signature=tf.TensorSpec([None], tf.float64),
    )
    quantiles = _linear_quantiles(tf.sort(flat, axis=0), probabilities)
    return means, variances, tf.math.log(variances), moments, quantiles, covariance


_summary_xla = tf.function(_summary_kernel, autograph=False, jit_compile=True)
_summary_eager = tf.function(_summary_kernel, autograph=False, jit_compile=False)


def summarize_forecast_paths(
    paths: tf.Tensor,
    config: PredictiveStatisticsConfig = PredictiveStatisticsConfig(),
) -> PredictiveSummary:
    """Compute pooled descriptive summaries while preserving explicit input axes."""

    _validate_summary_config(config)
    paths = _require_tensor(paths, "paths", rank=4)
    if paths.shape[-1] != config.horizon:
        raise PredictiveContractError("paths must have horizon 10")
    path_count = paths.shape[0] * paths.shape[1] * paths.shape[2]
    if path_count < 2:
        raise PredictiveContractError("at least two complete forecast paths are required")
    probabilities = tf.constant(config.quantile_probabilities, tf.float64)
    orders = tf.constant(config.central_moment_orders, tf.int32)
    kernel = _summary_xla if config.jit_compile else _summary_eager
    means, variances, log_variances, moments, quantiles, covariance = kernel(
        paths, probabilities, orders
    )
    try:
        tf.debugging.assert_positive(variances, "sample variances must be positive")
        for value in (means, variances, log_variances, moments, quantiles, covariance):
            tf.debugging.assert_all_finite(value, "summary output must be finite")
    except tf.errors.InvalidArgumentError as exc:
        raise PredictiveContractError("summary produced invalid variance or nonfinite output") from exc
    return PredictiveSummary(
        means=means,
        variances=variances,
        log_variances=log_variances,
        central_moments=moments,
        quantiles=quantiles,
        cross_horizon_covariance=covariance,
        path_count=tf.constant(path_count, tf.int64),
        status=tf.constant(_VALID),
    )


def _mean_log_variance_influence_kernel(
    standardized_paths: tf.Tensor,
) -> tuple[tf.Tensor, ...]:
    chain_count = standardized_paths.shape[0]
    draw_count = standardized_paths.shape[1]
    replication_count = standardized_paths.shape[2]
    path_count = chain_count * draw_count * replication_count
    means = tf.reduce_mean(standardized_paths, axis=[0, 1, 2])
    centered = standardized_paths - means
    mean_influence = tf.reduce_mean(centered, axis=2)
    second_moment = tf.reduce_mean(tf.square(centered), axis=[0, 1, 2])
    variances = second_moment * tf.cast(path_count, tf.float64) / tf.cast(
        path_count - 1, tf.float64
    )
    log_variance_influence = (
        tf.reduce_mean(tf.square(centered), axis=2) / second_moment - 1.0
    )
    influence = tf.concat((mean_influence, log_variance_influence), axis=-1)
    return means, tf.math.log(variances), influence


_mean_log_variance_influence_xla = tf.function(
    _mean_log_variance_influence_kernel, autograph=False, jit_compile=True
)
_mean_log_variance_influence_eager = tf.function(
    _mean_log_variance_influence_kernel, autograph=False, jit_compile=False
)


def mean_log_variance_influence(
    standardized_paths: tf.Tensor,
    *,
    jit_compile: bool = True,
) -> MeanLogVarianceInfluenceResult:
    """Build co-primary estimates and draw-cluster influence sequences.

    Input paths must already be standardized with calibration-only horizon
    scales. Forecast replications are averaged inside each retained-draw
    cluster, while the draw axis remains available for chain-aware long-run
    covariance estimation. The log-variance influence is unchanged by the
    finite-sample correction applied to the reported sample variance.
    """

    _require_bool(jit_compile, "jit_compile")
    paths = _require_tensor(standardized_paths, "standardized_paths", rank=4)
    chain_count, draw_count, replication_count, horizon = paths.shape
    if chain_count < 2 or draw_count < 2 or replication_count < 1 or horizon != 10:
        raise PredictiveContractError(
            "standardized_paths must have static shape [chain>=2, draw>=2, replication>=1, 10]"
        )
    path_count = chain_count * draw_count * replication_count
    kernel = (
        _mean_log_variance_influence_xla
        if jit_compile
        else _mean_log_variance_influence_eager
    )
    means, log_variances, influence = kernel(paths)
    try:
        for value in (means, log_variances, influence):
            tf.debugging.assert_all_finite(value, "feature influence output must be finite")
        tf.debugging.assert_near(
            tf.reduce_mean(influence, axis=[0, 1]),
            tf.zeros([20], tf.float64),
            atol=tf.constant(4096.0 * 2.220446049250313e-16, tf.float64),
            rtol=tf.constant(0.0, tf.float64),
            message="empirical influence values must be centered",
        )
    except tf.errors.InvalidArgumentError as exc:
        raise PredictiveContractError(
            "feature influence produced zero variance, nonfinite output, or lost centering"
        ) from exc
    result = MeanLogVarianceInfluenceResult(
        feature_estimate=tf.concat((means, log_variances), axis=0),
        standardized_means=means,
        log_variances=log_variances,
        influence_values=influence,
        chain_count=chain_count,
        draw_count=draw_count,
        forecast_replication_count=replication_count,
        path_count=path_count,
        status=tf.constant(_VALID),
    )
    return _seal_result(
        result,
        kind="mean_log_variance_influence",
        provenance=(("standardized_paths", paths), ("jit_compile", jit_compile)),
    )


def _conditional_mean_log_variance_influence_kernel(
    conditional_means: tf.Tensor,
    conditional_variances: tf.Tensor,
) -> tuple[tf.Tensor, ...]:
    means = tf.reduce_mean(conditional_means, axis=[0, 1, 2])
    conditional_second = conditional_variances + tf.square(conditional_means)
    second_moments = tf.reduce_mean(conditional_second, axis=[0, 1, 2])
    variances = second_moments - tf.square(means)
    mean_influence = tf.reduce_mean(conditional_means - means, axis=2)
    variance_contribution = conditional_variances + tf.square(
        conditional_means - means
    )
    log_variance_influence = (
        tf.reduce_mean(variance_contribution, axis=2) / variances - 1.0
    )
    influence = tf.concat((mean_influence, log_variance_influence), axis=-1)
    return means, tf.math.log(variances), influence, second_moments


_conditional_mean_log_variance_influence_xla = tf.function(
    _conditional_mean_log_variance_influence_kernel,
    autograph=False,
    jit_compile=True,
)
_conditional_mean_log_variance_influence_eager = tf.function(
    _conditional_mean_log_variance_influence_kernel,
    autograph=False,
    jit_compile=False,
)


def conditional_mean_log_variance_influence(
    standardized_conditional_means: tf.Tensor,
    standardized_conditional_variances: tf.Tensor,
    *,
    jit_compile: bool = True,
) -> ConditionalMeanLogVarianceInfluenceResult:
    """Integrate conditional observation noise in mean/log-variance features.

    Inputs use shape ``[chain, draw, forecast replication, horizon]`` and must
    already use a scale frozen independently of confirmation. Conditional
    variances integrate only the noise represented by that conditional law.
    """

    _require_bool(jit_compile, "jit_compile")
    means = _require_tensor(
        standardized_conditional_means,
        "standardized_conditional_means",
        rank=4,
    )
    variances = _require_tensor(
        standardized_conditional_variances,
        "standardized_conditional_variances",
        rank=4,
    )
    if means.shape != variances.shape:
        raise PredictiveContractError(
            "conditional means and variances must have identical static shape"
        )
    chain_count, draw_count, replication_count, horizon = means.shape
    if chain_count < 2 or draw_count < 2 or replication_count < 1 or horizon != 10:
        raise PredictiveContractError(
            "conditional moments must have static shape "
            "[chain>=2, draw>=2, replication>=1, 10]"
        )
    if bool(tf.reduce_any(variances < 0.0)):
        raise PredictiveContractError("conditional variances must be nonnegative")
    kernel = (
        _conditional_mean_log_variance_influence_xla
        if jit_compile
        else _conditional_mean_log_variance_influence_eager
    )
    estimated_means, log_variances, influence, second_moments = kernel(
        means, variances
    )
    try:
        for value in (estimated_means, log_variances, influence, second_moments):
            tf.debugging.assert_all_finite(
                value, "conditional-moment influence output must be finite"
            )
        tf.debugging.assert_near(
            tf.reduce_mean(influence, axis=[0, 1]),
            tf.zeros([20], tf.float64),
            atol=tf.constant(4096.0 * 2.220446049250313e-16, tf.float64),
            rtol=tf.constant(0.0, tf.float64),
            message="conditional-moment influences must be centered",
        )
    except tf.errors.InvalidArgumentError as exc:
        raise PredictiveContractError(
            "conditional-moment influence produced nonpositive variance, "
            "nonfinite output, or lost centering"
        ) from exc
    result = ConditionalMeanLogVarianceInfluenceResult(
        feature_estimate=tf.concat((estimated_means, log_variances), axis=0),
        standardized_means=estimated_means,
        log_variances=log_variances,
        influence_values=influence,
        conditional_second_moments=second_moments,
        chain_count=chain_count,
        draw_count=draw_count,
        forecast_replication_count=replication_count,
        cluster_count=chain_count * draw_count,
        status=tf.constant(_VALID),
    )
    return _seal_result(
        result,
        kind="conditional_mean_log_variance_influence",
        provenance=(
            ("standardized_conditional_means", means),
            ("standardized_conditional_variances", variances),
            ("jit_compile", jit_compile),
        ),
    )


def _pairwise_distance_scale_kernel(paths: tf.Tensor) -> tuple[tf.Tensor, ...]:
    flat = tf.reshape(paths, [-1, 10])
    count = flat.shape[0]
    if count is None:
        raise PredictiveContractError("pairwise distance paths require a static path count")
    squared_norm = tf.reduce_sum(tf.square(flat), axis=1)
    squared_distance = (
        squared_norm[:, tf.newaxis]
        + squared_norm[tf.newaxis, :]
        - 2.0 * tf.matmul(flat, flat, transpose_b=True)
    )
    squared_distance = tf.maximum(squared_distance, tf.zeros_like(squared_distance))
    distances = tf.sqrt(squared_distance)
    upper = tf.range(count)[:, tf.newaxis] < tf.range(count)[tf.newaxis, :]
    positive = tf.logical_and(upper, distances > 0.0)
    positive_count = tf.reduce_sum(tf.cast(positive, tf.int32))
    fixed_candidates = tf.where(
        positive,
        distances,
        tf.fill(tf.shape(distances), tf.constant(float("inf"), tf.float64)),
    )
    sorted_values = tf.sort(tf.reshape(fixed_candidates, [count * count]))
    midpoint = tf.cast(tf.maximum(positive_count - 1, 0), tf.float64) / 2.0
    lower = tf.cast(tf.floor(midpoint), tf.int32)
    upper_index = tf.cast(tf.math.ceil(midpoint), tf.int32)
    median = 0.5 * (
        tf.gather(sorted_values, lower) + tf.gather(sorted_values, upper_index)
    )
    valid_count = positive_count > 0
    median = tf.where(valid_count, median, tf.constant(float("nan"), tf.float64))
    return median, positive_count, tf.constant(count * (count - 1) // 2, tf.int32)


_pairwise_distance_scale_xla = tf.function(
    _pairwise_distance_scale_kernel, autograph=False, jit_compile=True
)
_pairwise_distance_scale_eager = tf.function(
    _pairwise_distance_scale_kernel, autograph=False, jit_compile=False
)


def pooled_pairwise_distance_scale(
    paths: tf.Tensor,
    *,
    jit_compile: bool = True,
) -> PairwiseDistanceScaleResult:
    """Return the median positive Euclidean distance between complete paths."""

    _require_bool(jit_compile, "jit_compile")
    tensor = _require_tensor(paths, "paths", rank=4)
    if tensor.shape[-1] != 10:
        raise PredictiveContractError("paths must have horizon 10")
    path_count = int(tensor.shape[0] * tensor.shape[1] * tensor.shape[2])
    if path_count < 2:
        raise PredictiveContractError("at least two complete paths are required")
    kernel = _pairwise_distance_scale_xla if jit_compile else _pairwise_distance_scale_eager
    median, positive_count, total_count = kernel(tensor)
    try:
        tf.debugging.assert_positive(median, "median pairwise distance must be positive")
        tf.debugging.assert_all_finite(median, "median pairwise distance must be finite")
        tf.debugging.assert_equal(
            total_count,
            path_count * (path_count - 1) // 2,
            "pairwise distance count is inconsistent",
        )
    except tf.errors.InvalidArgumentError as exc:
        raise PredictiveContractError(
            "pairwise distance cloud is duplicate-degenerate or nonfinite"
        ) from exc
    result = PairwiseDistanceScaleResult(
        median_distance=median,
        positive_pair_count=positive_count,
        total_pair_count=total_count,
        path_count=path_count,
        status=tf.constant(_VALID),
    )
    return _seal_result(
        result,
        kind="pairwise_distance_scale",
        provenance=(("paths", tensor), ("jit_compile", jit_compile)),
    )


def _standardize_kernel(
    paths: tf.Tensor,
    center: tf.Tensor,
    scale: tf.Tensor,
    scale_floor: tf.Tensor,
) -> tf.Tensor:
    return (paths - center) / tf.maximum(scale, scale_floor)


_standardize_xla = tf.function(_standardize_kernel, autograph=False, jit_compile=True)
_standardize_eager = tf.function(_standardize_kernel, autograph=False, jit_compile=False)


def standardize_forecast_paths(
    paths: tf.Tensor,
    center: tf.Tensor,
    scale: tf.Tensor,
    *,
    scale_floor: tf.Tensor,
    jit_compile: bool = True,
    allow_floor_use: bool = False,
) -> tf.Tensor:
    _require_bool(jit_compile, "jit_compile")
    _require_bool(allow_floor_use, "allow_floor_use")
    paths = _require_tensor(paths, "paths")
    if paths.shape.rank is None or paths.shape.rank < 2:
        raise PredictiveContractError("paths must have at least two axes")
    center = _require_tensor(center, "center", rank=1)
    scale = _require_tensor(scale, "scale", rank=1)
    scale_floor = _require_tensor(scale_floor, "scale_floor", rank=0)
    if center.shape != scale.shape or center.shape[0] != paths.shape[-1]:
        raise PredictiveContractError("center and scale must match the path feature axis")
    try:
        tf.debugging.assert_positive(scale, "scale must be positive")
        tf.debugging.assert_positive(scale_floor, "scale_floor must be positive")
    except tf.errors.InvalidArgumentError as exc:
        raise PredictiveContractError("scale and scale_floor must be positive") from exc
    kernel = _standardize_xla if jit_compile else _standardize_eager
    try:
        floor_assertion = None
        if not allow_floor_use:
            floor_assertion = tf.debugging.assert_greater_equal(
                scale,
                scale_floor,
                "scale_floor use is forbidden for this evidentiary row",
            )
        dependencies = [] if floor_assertion is None else [floor_assertion]
        with tf.control_dependencies(dependencies):
            result = kernel(paths, center, scale, scale_floor)
        tf.debugging.assert_all_finite(result, "standardized paths must be finite")
    except tf.errors.InvalidArgumentError as exc:
        raise PredictiveContractError(
            "standardization violated the scale-floor use or finite-output contract"
        ) from exc
    return result


def _validate_kernel_config(
    bandwidths: tf.Tensor,
    mixture_weights: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    bandwidths = _require_tensor(bandwidths, "bandwidths", rank=1)
    mixture_weights = _require_tensor(mixture_weights, "mixture_weights", rank=1)
    if bandwidths.shape[0] == 0 or bandwidths.shape != mixture_weights.shape:
        raise PredictiveContractError("bandwidths and weights must have equal nonzero shape")
    try:
        tf.debugging.assert_positive(bandwidths, "bandwidths must be positive")
        tf.debugging.assert_greater_equal(
            mixture_weights, tf.constant(0.0, tf.float64), "weights must be nonnegative"
        )
    except tf.errors.InvalidArgumentError as exc:
        raise PredictiveContractError("invalid bandwidth or mixture weight") from exc
    bandwidth_values = [float(tf.get_static_value(value)) for value in tf.unstack(bandwidths)]
    if len(set(bandwidth_values)) != len(bandwidth_values):
        raise PredictiveContractError("bandwidths must be unique")
    weight_sum = float(tf.get_static_value(tf.reduce_sum(mixture_weights)))
    if abs(weight_sum - 1.0) > 64.0 * 2.220446049250313e-16:
        raise PredictiveContractError("mixture weights must sum to one")
    return bandwidths, mixture_weights


def _rbf_matrices(
    left: tf.Tensor,
    right: tf.Tensor,
    bandwidths: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    def squared_distances(first: tf.Tensor, second: tf.Tensor) -> tf.Tensor:
        differences = first[:, tf.newaxis, :] - second[tf.newaxis, :, :]
        return tf.reduce_sum(tf.square(differences), axis=-1)

    scale = 2.0 * tf.square(bandwidths)[:, tf.newaxis, tf.newaxis]
    return tuple(
        tf.exp(-squared_distances(first, second)[tf.newaxis, :, :] / scale)
        for first, second in ((left, left), (right, right), (left, right))
    )


def _mmd_kernel(
    left: tf.Tensor,
    right: tf.Tensor,
    bandwidths: tf.Tensor,
    weights: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    left_kernel, right_kernel, cross_kernel = _rbf_matrices(left, right, bandwidths)
    left_count = tf.cast(tf.shape(left)[0], tf.float64)
    right_count = tf.cast(tf.shape(right)[0], tf.float64)
    left_off_diagonal = (
        tf.reduce_sum(left_kernel, axis=[1, 2])
        - tf.reduce_sum(tf.linalg.diag_part(left_kernel), axis=1)
    ) / (left_count * (left_count - 1.0))
    right_off_diagonal = (
        tf.reduce_sum(right_kernel, axis=[1, 2])
        - tf.reduce_sum(tf.linalg.diag_part(right_kernel), axis=1)
    ) / (right_count * (right_count - 1.0))
    cross_mean = tf.reduce_mean(cross_kernel, axis=[1, 2])
    per_u = left_off_diagonal + right_off_diagonal - 2.0 * cross_mean
    per_v = (
        tf.reduce_mean(left_kernel, axis=[1, 2])
        + tf.reduce_mean(right_kernel, axis=[1, 2])
        - 2.0 * cross_mean
    )
    return tf.reduce_sum(weights * per_u), tf.reduce_sum(weights * per_v), per_u, per_v


_mmd_xla = tf.function(_mmd_kernel, autograph=False, jit_compile=True)
_mmd_eager = tf.function(_mmd_kernel, autograph=False, jit_compile=False)


def fixed_rbf_mmd(
    left_paths: tf.Tensor,
    right_paths: tf.Tensor,
    *,
    bandwidths: tf.Tensor,
    mixture_weights: tf.Tensor,
    sampling_contract: SamplingContract,
    iid_samples_verified: bool = False,
    independent_arm_banks_verified: bool = False,
    jit_compile: bool = True,
) -> MMDStatistics:
    """Return signed diagonal-excluded U and separately biased V MMD forms."""

    _require_bool(jit_compile, "jit_compile")
    _require_bool(iid_samples_verified, "iid_samples_verified")
    _require_bool(
        independent_arm_banks_verified, "independent_arm_banks_verified"
    )
    if type(sampling_contract) is not str:
        raise PredictiveContractError("sampling_contract must be a string")
    allowed = {
        "iid_oracle_fixture",
        "dependent_descriptive_only",
        "paired_diagnostic_shared",
    }
    if sampling_contract not in allowed:
        raise PredictiveContractError("unknown MMD sampling contract")
    left_paths = _require_tensor(left_paths, "left_paths")
    right_paths = _require_tensor(right_paths, "right_paths")
    if left_paths.shape.rank is None or left_paths.shape.rank < 2:
        raise PredictiveContractError("MMD paths must have at least two axes")
    if right_paths.shape.rank != left_paths.shape.rank:
        raise PredictiveContractError("MMD arms must have the same rank")
    if left_paths.shape[-1] != right_paths.shape[-1]:
        raise PredictiveContractError("MMD feature dimensions must agree")
    if sampling_contract == "iid_oracle_fixture" and left_paths.shape.rank != 2:
        raise PredictiveContractError("IID oracle MMD requires rank-two independent samples")
    left = tf.reshape(left_paths, [-1, left_paths.shape[-1]])
    right = tf.reshape(right_paths, [-1, right_paths.shape[-1]])
    if left.shape[0] < 2 or right.shape[0] < 2:
        raise PredictiveContractError("each MMD arm requires at least two paths")
    bandwidths, mixture_weights = _validate_kernel_config(
        bandwidths, mixture_weights
    )
    kernel = _mmd_xla if jit_compile else _mmd_eager
    squared_u, squared_v, per_u, per_v = kernel(
        left, right, bandwidths, mixture_weights
    )
    for value in (squared_u, squared_v, per_u, per_v):
        try:
            tf.debugging.assert_all_finite(value, "MMD output must be finite")
        except tf.errors.InvalidArgumentError as exc:
            raise PredictiveContractError("MMD produced nonfinite output") from exc
    result = MMDStatistics(
        squared_mmd_u=squared_u,
        squared_mmd_v_biased=squared_v,
        per_bandwidth_u=per_u,
        per_bandwidth_v_biased=per_v,
        bandwidths=bandwidths,
        mixture_weights=mixture_weights,
        sampling_contract=sampling_contract,
        iid_samples_verified=iid_samples_verified,
        independent_arm_banks_verified=independent_arm_banks_verified,
        # Quadratic U/V forms remain descriptive. Decision-bound MMD inference
        # is exclusively the authenticated cross-chain linear estimator.
        inference_admissible=False,
        status=tf.constant(_VALID),
    )
    return _seal_result(
        result,
        kind="quadratic_mmd",
        provenance=(
            ("left_paths", left_paths),
            ("right_paths", right_paths),
            ("bandwidths", bandwidths),
            ("mixture_weights", mixture_weights),
            ("sampling_contract", sampling_contract),
            ("iid_samples_verified", iid_samples_verified),
            ("independent_arm_banks_verified", independent_arm_banks_verified),
            ("jit_compile", jit_compile),
        ),
    )


def _cluster_kernel_sequence(
    first: tf.Tensor,
    second: tf.Tensor,
    bandwidths: tf.Tensor,
    weights: tf.Tensor,
) -> tf.Tensor:
    differences = (
        first[:, :, tf.newaxis, :] - second[:, tf.newaxis, :, :]
    )
    squared_distance = tf.reduce_sum(tf.square(differences), axis=-1)
    kernels = tf.exp(
        -squared_distance[tf.newaxis, :, :, :]
        / (2.0 * tf.square(bandwidths)[:, tf.newaxis, tf.newaxis, tf.newaxis])
    )
    per_bandwidth = tf.reduce_mean(kernels, axis=[2, 3])
    return tf.reduce_sum(weights[:, tf.newaxis] * per_bandwidth, axis=0)


_cluster_kernel_xla = tf.function(
    _cluster_kernel_sequence, autograph=False, jit_compile=True
)
_cluster_kernel_eager = tf.function(
    _cluster_kernel_sequence, autograph=False, jit_compile=False
)


def _schedule_rows(schedule: tf.Tensor, chain_count: int) -> list[tuple[int, int]]:
    if not tf.is_tensor(schedule) or schedule.dtype not in (tf.int32, tf.int64):
        raise PredictiveContractError("chain_pair_schedule must be an integer tensor")
    if schedule.shape.rank != 2 or schedule.shape[1] != 2 or not schedule.shape.is_fully_defined():
        raise PredictiveContractError("chain_pair_schedule must have static shape [pair, 2]")
    rows: list[tuple[int, int]] = []
    for row in tf.unstack(schedule):
        first, second = (
            int(tf.get_static_value(value)) for value in tf.unstack(row)
        )
        if first == second:
            raise PredictiveContractError("a chain cannot occupy both sides of a pair")
        if min(first, second) < 0 or max(first, second) >= chain_count:
            raise PredictiveContractError("chain pair index is out of bounds")
        rows.append((first, second))
    flattened = [index for row in rows for index in row]
    if len(set(flattened)) != len(flattened):
        raise PredictiveContractError("A3 chain pairs must be disjoint")
    if chain_count == 2:
        if len(rows) != 1 or set(flattened) != {0, 1}:
            raise PredictiveContractError(
                "two-chain mechanics requires exactly one pair containing both chains"
            )
    elif chain_count >= 4:
        if len(rows) < 2:
            raise PredictiveContractError(
                "inference requires at least two disjoint chain pairs"
            )
    else:
        raise PredictiveContractError(
            "cross-chain MMD supports two-chain mechanics or at least four chains"
        )
    return rows


def cross_chain_linear_mmd(
    left_paths: tf.Tensor,
    right_paths: tf.Tensor,
    *,
    bandwidths: tf.Tensor,
    mixture_weights: tf.Tensor,
    chain_pair_schedule: tf.Tensor,
    independent_arm_banks_verified: bool,
    stationarity_verified: bool,
    mixing_verified: bool,
    jit_compile: bool = True,
) -> CrossChainLinearMMD:
    """Construct distinct-chain linear MMD contrast sequences."""

    _require_bool(independent_arm_banks_verified, "independent_arm_banks_verified")
    _require_bool(stationarity_verified, "stationarity_verified")
    _require_bool(mixing_verified, "mixing_verified")
    _require_bool(jit_compile, "jit_compile")
    left_paths = _require_tensor(left_paths, "left_paths", rank=4)
    right_paths = _require_tensor(right_paths, "right_paths", rank=4)
    if left_paths.shape != right_paths.shape:
        raise PredictiveContractError("cross-chain MMD arms must have equal static shape")
    chain_count, draw_count, replication_count, horizon = left_paths.shape
    if chain_count not in {2} and chain_count < 4:
        raise PredictiveContractError(
            "cross-chain MMD supports two-chain mechanics or at least four chains"
        )
    if draw_count < 2 or replication_count < 1 or horizon != 10:
        raise PredictiveContractError(
            "cross-chain MMD requires >=2 draws, replications, and horizon 10"
        )
    rows = _schedule_rows(chain_pair_schedule, chain_count)
    bandwidths, mixture_weights = _validate_kernel_config(
        bandwidths, mixture_weights
    )
    kernel = _cluster_kernel_xla if jit_compile else _cluster_kernel_eager
    contrasts = []
    for first, second in rows:
        contrast = (
            kernel(left_paths[first], left_paths[second], bandwidths, mixture_weights)
            + kernel(right_paths[first], right_paths[second], bandwidths, mixture_weights)
            - kernel(left_paths[first], right_paths[second], bandwidths, mixture_weights)
            - kernel(left_paths[second], right_paths[first], bandwidths, mixture_weights)
        )
        contrasts.append(contrast)
    sequence = tf.stack(contrasts, axis=0)
    try:
        tf.debugging.assert_all_finite(sequence, "linear MMD contrasts must be finite")
    except tf.errors.InvalidArgumentError as exc:
        raise PredictiveContractError("linear MMD produced nonfinite output") from exc
    admissible = bool(
        chain_count >= 4
        and len(rows) >= 2
        and independent_arm_banks_verified
        and stationarity_verified
        and mixing_verified
    )
    mechanics_only = chain_count == 2
    result = CrossChainLinearMMD(
        squared_mmd_linear=tf.reduce_mean(sequence),
        kernel_contrast_sequence=sequence,
        chain_pair_schedule=chain_pair_schedule,
        independent_arm_banks_verified=bool(independent_arm_banks_verified),
        stationarity_required=True,
        stationarity_verified=stationarity_verified,
        mixing_verified=mixing_verified,
        mechanics_only=mechanics_only,
        inference_admissible=admissible,
        # A two-chain row is a valid mechanics computation but never admissible
        # for inference; failed scientific admission flags remain hard vetoes.
        status=tf.constant(_VALID if admissible or mechanics_only else _INVALID),
    )
    return _seal_result(
        result,
        kind="cross_chain_linear_mmd",
        provenance=(
            ("left_paths", left_paths),
            ("right_paths", right_paths),
            ("bandwidths", bandwidths),
            ("mixture_weights", mixture_weights),
            ("chain_pair_schedule", chain_pair_schedule),
            ("independent_arm_banks_verified", independent_arm_banks_verified),
            ("stationarity_verified", stationarity_verified),
            ("mixing_verified", mixing_verified),
            ("jit_compile", jit_compile),
        ),
    )


def _batch_geometry(values: tf.Tensor, block_length: int) -> tuple[int, int]:
    if values.shape.rank is None or values.shape.rank < 2:
        raise PredictiveContractError("values must begin with [chain, draw]")
    if type(block_length) is not int:
        raise PredictiveContractError("block_length must be an integer")
    draw_count = values.shape[1]
    if block_length < 1 or block_length > draw_count:
        raise PredictiveContractError("block_length lies outside [1, draw_count]")
    if draw_count % block_length:
        raise PredictiveContractError("remainder draws cannot be silently truncated")
    batch_count = draw_count // block_length
    if batch_count < 2:
        raise PredictiveContractError("at least two complete batches per chain are required")
    return draw_count, batch_count


def _batch_means_kernel(values: tf.Tensor, block_length: int) -> tf.Tensor:
    batch_count = values.shape[1] // block_length
    shape = [values.shape[0], batch_count, block_length, *values.shape[2:]]
    return tf.reduce_mean(tf.reshape(values, shape), axis=2)


_batch_means_xla = tf.function(_batch_means_kernel, autograph=False, jit_compile=True)
_batch_means_eager = tf.function(_batch_means_kernel, autograph=False, jit_compile=False)


def chain_batch_means(
    values: tf.Tensor,
    *,
    block_length: int,
    jit_compile: bool = True,
) -> tf.Tensor:
    """Partition only the retained-draw axis into contiguous complete blocks."""

    _require_bool(jit_compile, "jit_compile")
    values = _require_tensor(values, "values")
    _batch_geometry(values, block_length)
    kernel = _batch_means_xla if jit_compile else _batch_means_eager
    return kernel(values, block_length)


def _long_run_covariance_kernel(
    values: tf.Tensor,
    block_length: int,
    ridge_ladder: tf.Tensor,
    condition_number_max: tf.Tensor,
) -> tuple[tf.Tensor, ...]:
    chain_count = values.shape[0]
    draw_count = values.shape[1]
    feature_count = values.shape[2]
    batch_count = draw_count // block_length
    batches = tf.reduce_mean(
        tf.reshape(
            values,
            [chain_count, batch_count, block_length, feature_count],
        ),
        axis=2,
    )
    centered = batches - tf.reduce_mean(batches, axis=1, keepdims=True)
    per_chain = tf.einsum("cbf,cbg->cfg", centered, centered) / tf.cast(
        batch_count - 1, tf.float64
    )
    spectral = tf.cast(block_length, tf.float64) * tf.reduce_mean(per_chain, axis=0)
    spectral = 0.5 * (spectral + tf.transpose(spectral))
    pooled_mean = spectral / tf.cast(chain_count * draw_count, tf.float64)
    scale = tf.maximum(
        tf.reduce_mean(tf.linalg.diag_part(pooled_mean)),
        tf.constant(1.0, tf.float64),
    )
    identity = tf.eye(feature_count, dtype=tf.float64)
    candidates = (
        pooled_mean[tf.newaxis, :, :]
        + ridge_ladder[:, tf.newaxis, tf.newaxis] * scale * identity
    )
    eigenvalues = tf.linalg.eigvalsh(candidates)
    minimum = eigenvalues[:, 0]
    maximum = eigenvalues[:, -1]
    condition = tf.where(
        minimum > 0.0,
        maximum / minimum,
        tf.fill(tf.shape(minimum), tf.constant(float("inf"), tf.float64)),
    )
    eligible = tf.logical_and(
        tf.reduce_all(tf.math.is_finite(eigenvalues), axis=1),
        tf.logical_and(minimum > 0.0, condition <= condition_number_max),
    )
    any_eligible = tf.reduce_any(eligible)
    first = tf.argmax(tf.cast(eligible, tf.int32), output_type=tf.int32)
    selected_index = tf.where(any_eligible, first, tf.constant(-1, tf.int32))
    fallback_index = tf.shape(ridge_ladder, out_type=tf.int32)[0] - 1
    gather_index = tf.where(any_eligible, first, fallback_index)
    regularized = candidates[gather_index]
    selected_eigenvalues = eigenvalues[gather_index]
    selected_condition = condition[gather_index]
    selected_multiplier = tf.where(
        any_eligible,
        ridge_ladder[first],
        tf.constant(float("nan"), tf.float64),
    )
    selected_loading = tf.where(
        any_eligible,
        ridge_ladder[first] * scale,
        tf.constant(float("nan"), tf.float64),
    )
    precision = tf.cond(
        any_eligible,
        lambda: tf.linalg.inv(regularized),
        lambda: tf.fill(tf.shape(regularized), tf.constant(float("nan"), tf.float64)),
    )
    return (
        spectral,
        pooled_mean,
        regularized,
        precision,
        selected_index,
        selected_multiplier,
        selected_loading,
        selected_eigenvalues,
        selected_condition,
        any_eligible,
    )


_long_run_covariance_xla = tf.function(
    _long_run_covariance_kernel, autograph=False, jit_compile=True
)
_long_run_covariance_eager = tf.function(
    _long_run_covariance_kernel, autograph=False, jit_compile=False
)


def chain_batch_long_run_covariance(
    values: tf.Tensor,
    *,
    block_length: int,
    ridge_ladder: tuple[float, ...] = (0.0, 1.0e-12, 1.0e-10, 1.0e-8, 1.0e-6),
    condition_number_max: float = 1.0e8,
    jit_compile: bool = True,
) -> LongRunCovarianceResult:
    """Estimate pooled-mean long-run covariance and regularize it fail-closed.

    Contiguous within-chain batch means estimate the spectral covariance as
    ``block_length * mean(chain_sample_covariance(batch_means))``. Assuming
    independent chains, division by ``chain_count * draw_count`` gives the
    covariance of the pooled mean. The first ridge candidate with positive
    finite eigenvalues and condition number below the declared ceiling is
    selected; exhaustion returns an inadmissible result with no precision.
    """

    _require_bool(jit_compile, "jit_compile")
    tensor = _require_tensor(values, "values", rank=3)
    draw_count, batch_count = _batch_geometry(tensor, block_length)
    chain_count = int(tensor.shape[0])
    feature_count = int(tensor.shape[2])
    if chain_count < 2:
        raise PredictiveContractError("long-run covariance requires at least two chains")
    if feature_count < 1:
        raise PredictiveContractError("long-run covariance requires at least one feature")
    if type(ridge_ladder) is not tuple or not ridge_ladder:
        raise PredictiveContractError("ridge_ladder must be a nonempty tuple")
    if any(type(value) is not float or not math.isfinite(value) or value < 0.0 for value in ridge_ladder):
        raise PredictiveContractError("ridge_ladder must contain finite nonnegative floats")
    if tuple(sorted(set(ridge_ladder))) != ridge_ladder:
        raise PredictiveContractError("ridge_ladder must be unique and increasing")
    if type(condition_number_max) is not float or not math.isfinite(condition_number_max):
        raise PredictiveContractError("condition_number_max must be a finite float")
    if condition_number_max <= 1.0:
        raise PredictiveContractError("condition_number_max must exceed one")
    ladder = tf.constant(ridge_ladder, tf.float64)
    condition_limit = tf.constant(condition_number_max, tf.float64)
    kernel = _long_run_covariance_xla if jit_compile else _long_run_covariance_eager
    (
        spectral,
        pooled_mean,
        regularized,
        precision,
        selected_index,
        selected_multiplier,
        selected_loading,
        eigenvalues,
        condition_number,
        admissible,
    ) = kernel(tensor, block_length, ladder, condition_limit)
    admissible_value = bool(admissible.numpy())
    result = LongRunCovarianceResult(
        spectral_covariance=spectral,
        pooled_mean_covariance=pooled_mean,
        regularized_covariance=regularized,
        precision=precision,
        ridge_ladder=ladder,
        selected_ridge_index=selected_index,
        selected_ridge_multiplier=selected_multiplier,
        selected_diagonal_loading=selected_loading,
        eigenvalues=eigenvalues,
        condition_number=condition_number,
        block_length=block_length,
        batch_count=batch_count,
        chain_count=chain_count,
        draw_count=draw_count,
        inference_admissible=admissible_value,
        status=tf.constant(_VALID if admissible_value else _INVALID),
    )
    return _seal_result(
        result,
        kind="long_run_covariance",
        provenance=(
            ("values", tensor),
            ("block_length", block_length),
            ("ridge_ladder", ridge_ladder),
            ("condition_number_max", condition_number_max),
            ("jit_compile", jit_compile),
        ),
    )


def growing_hac_bandwidth(draw_count: int, *, multiplier: float = 1.0) -> int:
    """Return the prospective ``floor(multiplier * N**(1/3))`` bandwidth."""

    if type(draw_count) is not int or draw_count < 2:
        raise PredictiveContractError("draw_count must be an integer at least two")
    if type(multiplier) is not float or not math.isfinite(multiplier) or multiplier <= 0.0:
        raise PredictiveContractError("bandwidth multiplier must be a finite positive float")
    bandwidth = max(1, int(math.floor(multiplier * math.cbrt(draw_count))))
    if bandwidth >= draw_count:
        raise PredictiveContractError("growing HAC bandwidth must be smaller than draw_count")
    return bandwidth


def _bartlett_long_run_covariance_kernel(
    values: tf.Tensor,
    bandwidth: int,
    ridge_ladder: tf.Tensor,
    condition_number_max: tf.Tensor,
) -> tuple[tf.Tensor, ...]:
    chain_count = values.shape[0]
    draw_count = values.shape[1]
    feature_count = values.shape[2]
    centered = values - tf.reduce_mean(values, axis=1, keepdims=True)
    denominator = tf.cast(draw_count, tf.float64)
    spectral = tf.reduce_mean(
        tf.einsum("cnf,cng->cfg", centered, centered) / denominator,
        axis=0,
    )
    for lag in range(1, bandwidth + 1):
        gamma = tf.reduce_mean(
            tf.einsum(
                "cnf,cng->cfg",
                centered[:, lag:, :],
                centered[:, : draw_count - lag, :],
            )
            / denominator,
            axis=0,
        )
        weight = tf.constant(1.0 - lag / (bandwidth + 1.0), tf.float64)
        spectral = spectral + weight * (gamma + tf.transpose(gamma))
    spectral = 0.5 * (spectral + tf.transpose(spectral))
    pooled_mean = spectral / tf.cast(chain_count * draw_count, tf.float64)
    scale = tf.maximum(
        tf.reduce_mean(tf.abs(tf.linalg.diag_part(pooled_mean))),
        tf.constant(1.0, tf.float64),
    )
    identity = tf.eye(feature_count, dtype=tf.float64)
    candidates = (
        pooled_mean[tf.newaxis, :, :]
        + ridge_ladder[:, tf.newaxis, tf.newaxis] * scale * identity
    )
    eigenvalues = tf.linalg.eigvalsh(candidates)
    minimum = eigenvalues[:, 0]
    maximum = eigenvalues[:, -1]
    condition = tf.where(
        minimum > 0.0,
        maximum / minimum,
        tf.fill(tf.shape(minimum), tf.constant(float("inf"), tf.float64)),
    )
    eligible = tf.logical_and(
        tf.reduce_all(tf.math.is_finite(eigenvalues), axis=1),
        tf.logical_and(minimum > 0.0, condition <= condition_number_max),
    )
    any_eligible = tf.reduce_any(eligible)
    first = tf.argmax(tf.cast(eligible, tf.int32), output_type=tf.int32)
    selected_index = tf.where(any_eligible, first, tf.constant(-1, tf.int32))
    fallback_index = tf.shape(ridge_ladder, out_type=tf.int32)[0] - 1
    gather_index = tf.where(any_eligible, first, fallback_index)
    regularized = candidates[gather_index]
    selected_eigenvalues = eigenvalues[gather_index]
    selected_condition = condition[gather_index]
    selected_multiplier = tf.where(
        any_eligible,
        ridge_ladder[first],
        tf.constant(float("nan"), tf.float64),
    )
    selected_loading = tf.where(
        any_eligible,
        ridge_ladder[first] * scale,
        tf.constant(float("nan"), tf.float64),
    )
    precision = tf.cond(
        any_eligible,
        lambda: tf.linalg.inv(regularized),
        lambda: tf.fill(tf.shape(regularized), tf.constant(float("nan"), tf.float64)),
    )
    return (
        spectral,
        pooled_mean,
        regularized,
        precision,
        selected_index,
        selected_multiplier,
        selected_loading,
        selected_eigenvalues,
        selected_condition,
        any_eligible,
    )


_bartlett_long_run_covariance_xla = tf.function(
    _bartlett_long_run_covariance_kernel, autograph=False, jit_compile=True
)
_bartlett_long_run_covariance_eager = tf.function(
    _bartlett_long_run_covariance_kernel, autograph=False, jit_compile=False
)


def _batched_bartlett_long_run_covariance_kernel(
    values: tf.Tensor,
    bandwidth: int,
    ridge_ladder: tf.Tensor,
    condition_number_max: tf.Tensor,
) -> tuple[tf.Tensor, ...]:
    batch_size = values.shape[0]
    chain_count = values.shape[1]
    draw_count = values.shape[2]
    feature_count = values.shape[3]
    centered = values - tf.reduce_mean(values, axis=2, keepdims=True)
    denominator = tf.cast(draw_count, tf.float64)
    spectral = tf.reduce_mean(
        tf.einsum("bcnf,bcng->bcfg", centered, centered) / denominator,
        axis=1,
    )
    for lag in range(1, bandwidth + 1):
        gamma = tf.reduce_mean(
            tf.einsum(
                "bcnf,bcng->bcfg",
                centered[:, :, lag:, :],
                centered[:, :, : draw_count - lag, :],
            )
            / denominator,
            axis=1,
        )
        weight = tf.constant(1.0 - lag / (bandwidth + 1.0), tf.float64)
        spectral = spectral + weight * (gamma + tf.transpose(gamma, [0, 2, 1]))
    spectral = 0.5 * (spectral + tf.transpose(spectral, [0, 2, 1]))
    pooled_mean = spectral / tf.cast(chain_count * draw_count, tf.float64)
    scale = tf.maximum(
        tf.reduce_mean(tf.abs(tf.linalg.diag_part(pooled_mean)), axis=1),
        tf.constant(1.0, tf.float64),
    )
    identity = tf.eye(feature_count, dtype=tf.float64)
    candidates = (
        pooled_mean[:, tf.newaxis, :, :]
        + ridge_ladder[tf.newaxis, :, tf.newaxis, tf.newaxis]
        * scale[:, tf.newaxis, tf.newaxis, tf.newaxis]
        * identity[tf.newaxis, tf.newaxis, :, :]
    )
    eigenvalues = tf.linalg.eigvalsh(candidates)
    minimum = eigenvalues[:, :, 0]
    maximum = eigenvalues[:, :, -1]
    condition = tf.where(
        minimum > 0.0,
        maximum / minimum,
        tf.fill(tf.shape(minimum), tf.constant(float("inf"), tf.float64)),
    )
    eligible = tf.logical_and(
        tf.reduce_all(tf.math.is_finite(eigenvalues), axis=2),
        tf.logical_and(minimum > 0.0, condition <= condition_number_max),
    )
    any_eligible = tf.reduce_any(eligible, axis=1)
    first = tf.argmax(tf.cast(eligible, tf.int32), axis=1, output_type=tf.int32)
    selected_index = tf.where(any_eligible, first, tf.fill([batch_size], -1))
    fallback_index = tf.shape(ridge_ladder, out_type=tf.int32)[0] - 1
    gather_index = tf.where(any_eligible, first, tf.fill([batch_size], fallback_index))
    gather_rows = tf.stack((tf.range(batch_size, dtype=tf.int32), gather_index), axis=1)
    regularized = tf.gather_nd(candidates, gather_rows)
    selected_eigenvalues = tf.gather_nd(eigenvalues, gather_rows)
    selected_condition = tf.gather_nd(condition, gather_rows)
    selected_multiplier = tf.where(
        any_eligible,
        tf.gather(ridge_ladder, first),
        tf.fill([batch_size], tf.constant(float("nan"), tf.float64)),
    )
    selected_loading = tf.where(
        any_eligible,
        selected_multiplier * scale,
        tf.fill([batch_size], tf.constant(float("nan"), tf.float64)),
    )
    inverse = tf.linalg.inv(
        tf.where(
            any_eligible[:, tf.newaxis, tf.newaxis],
            regularized,
            tf.eye(feature_count, batch_shape=[batch_size], dtype=tf.float64),
        )
    )
    precision = tf.where(
        any_eligible[:, tf.newaxis, tf.newaxis],
        inverse,
        tf.fill(tf.shape(inverse), tf.constant(float("nan"), tf.float64)),
    )
    return (
        spectral,
        pooled_mean,
        regularized,
        precision,
        selected_index,
        selected_multiplier,
        selected_loading,
        selected_eigenvalues,
        selected_condition,
        any_eligible,
    )


_batched_bartlett_long_run_covariance_xla = tf.function(
    _batched_bartlett_long_run_covariance_kernel, autograph=False, jit_compile=True
)
_batched_bartlett_long_run_covariance_eager = tf.function(
    _batched_bartlett_long_run_covariance_kernel, autograph=False, jit_compile=False
)


def chain_bartlett_long_run_covariance(
    values: tf.Tensor,
    *,
    bandwidth_multiplier: float = 1.0,
    ridge_ladder: tuple[float, ...] = (0.0, 1.0e-12, 1.0e-10, 1.0e-8, 1.0e-6),
    condition_number_max: float = 1.0e8,
    jit_compile: bool = True,
) -> BartlettLongRunCovarianceResult:
    """Estimate pooled-mean covariance with a growing Bartlett HAC bandwidth."""

    _require_bool(jit_compile, "jit_compile")
    tensor = _require_tensor(values, "values", rank=3)
    chain_count, draw_count, feature_count = tensor.shape
    if chain_count < 2:
        raise PredictiveContractError("Bartlett HAC requires at least two chains")
    if feature_count < 1:
        raise PredictiveContractError("Bartlett HAC requires at least one feature")
    bandwidth = growing_hac_bandwidth(draw_count, multiplier=bandwidth_multiplier)
    if type(ridge_ladder) is not tuple or not ridge_ladder:
        raise PredictiveContractError("ridge_ladder must be a nonempty tuple")
    if any(
        type(value) is not float or not math.isfinite(value) or value < 0.0
        for value in ridge_ladder
    ):
        raise PredictiveContractError("ridge_ladder must contain finite nonnegative floats")
    if tuple(sorted(set(ridge_ladder))) != ridge_ladder:
        raise PredictiveContractError("ridge_ladder must be unique and increasing")
    if ridge_ladder[0] != 0.0:
        raise PredictiveContractError("Bartlett HAC ridge_ladder must begin at zero")
    if type(condition_number_max) is not float or not math.isfinite(condition_number_max):
        raise PredictiveContractError("condition_number_max must be a finite float")
    if condition_number_max <= 1.0:
        raise PredictiveContractError("condition_number_max must exceed one")
    ladder = tf.constant(ridge_ladder, tf.float64)
    condition_limit = tf.constant(condition_number_max, tf.float64)
    kernel = (
        _bartlett_long_run_covariance_xla
        if jit_compile
        else _bartlett_long_run_covariance_eager
    )
    (
        spectral,
        pooled_mean,
        regularized,
        precision,
        selected_index,
        selected_multiplier,
        selected_loading,
        eigenvalues,
        condition_number,
        admissible,
    ) = kernel(tensor, bandwidth, ladder, condition_limit)
    numerically_admissible = bool(admissible.numpy())
    selected_index_value = int(selected_index.numpy())
    inference_admissible = (
        numerically_admissible
        and selected_index_value == 0
        and float(selected_multiplier.numpy()) == 0.0
    )
    result = BartlettLongRunCovarianceResult(
        spectral_covariance=spectral,
        pooled_mean_covariance=pooled_mean,
        regularized_covariance=regularized,
        precision=precision,
        ridge_ladder=ladder,
        selected_ridge_index=selected_index,
        selected_ridge_multiplier=selected_multiplier,
        selected_diagonal_loading=selected_loading,
        eigenvalues=eigenvalues,
        condition_number=condition_number,
        bandwidth=bandwidth,
        bandwidth_multiplier=bandwidth_multiplier,
        chain_count=chain_count,
        draw_count=draw_count,
        numerically_admissible=numerically_admissible,
        inference_admissible=inference_admissible,
        status=tf.constant(_VALID if inference_admissible else _INVALID),
    )
    return _seal_result(
        result,
        kind="bartlett_long_run_covariance",
        provenance=(
            ("values", tensor),
            ("bandwidth", bandwidth),
            ("bandwidth_multiplier", bandwidth_multiplier),
            ("ridge_ladder", ridge_ladder),
            ("condition_number_max", condition_number_max),
            ("jit_compile", jit_compile),
        ),
    )


def batched_chain_bartlett_long_run_covariance(
    values: tf.Tensor,
    *,
    bandwidth_multiplier: float = 1.0,
    ridge_ladder: tuple[float, ...] = (0.0,),
    condition_number_max: float = 1.0e8,
    jit_compile: bool = True,
) -> BatchedBartlettLongRunCovarianceResult:
    """Vectorized growing Bartlett HAC for independent experiment replications."""

    _require_bool(jit_compile, "jit_compile")
    tensor = _require_tensor(values, "values", rank=4)
    batch_size, chain_count, draw_count, feature_count = tensor.shape
    if batch_size < 1 or chain_count < 2 or feature_count < 1:
        raise PredictiveContractError(
            "batched Bartlett HAC requires batch>=1, chain>=2, and feature>=1"
        )
    bandwidth = growing_hac_bandwidth(draw_count, multiplier=bandwidth_multiplier)
    if type(ridge_ladder) is not tuple or not ridge_ladder:
        raise PredictiveContractError("ridge_ladder must be a nonempty tuple")
    if any(
        type(value) is not float or not math.isfinite(value) or value < 0.0
        for value in ridge_ladder
    ):
        raise PredictiveContractError("ridge_ladder must contain finite nonnegative floats")
    if tuple(sorted(set(ridge_ladder))) != ridge_ladder or ridge_ladder[0] != 0.0:
        raise PredictiveContractError("ridge_ladder must be unique, increasing, and begin at zero")
    if type(condition_number_max) is not float or not math.isfinite(condition_number_max):
        raise PredictiveContractError("condition_number_max must be a finite float")
    if condition_number_max <= 1.0:
        raise PredictiveContractError("condition_number_max must exceed one")
    ladder = tf.constant(ridge_ladder, tf.float64)
    kernel = (
        _batched_bartlett_long_run_covariance_xla
        if jit_compile
        else _batched_bartlett_long_run_covariance_eager
    )
    (
        spectral,
        pooled_mean,
        regularized,
        precision,
        selected_index,
        selected_multiplier,
        selected_loading,
        eigenvalues,
        condition_number,
        numerical,
    ) = kernel(
        tensor,
        bandwidth,
        ladder,
        tf.constant(condition_number_max, tf.float64),
    )
    inference = tf.logical_and(numerical, selected_index == 0)
    status = tf.where(
        inference,
        tf.fill([batch_size], tf.constant(_VALID)),
        tf.fill([batch_size], tf.constant(_INVALID)),
    )
    result = BatchedBartlettLongRunCovarianceResult(
        spectral_covariance=spectral,
        pooled_mean_covariance=pooled_mean,
        regularized_covariance=regularized,
        precision=precision,
        ridge_ladder=ladder,
        selected_ridge_index=selected_index,
        selected_ridge_multiplier=selected_multiplier,
        selected_diagonal_loading=selected_loading,
        eigenvalues=eigenvalues,
        condition_number=condition_number,
        bandwidth=bandwidth,
        bandwidth_multiplier=bandwidth_multiplier,
        batch_size=batch_size,
        chain_count=chain_count,
        draw_count=draw_count,
        numerically_admissible=numerical,
        inference_admissible=inference,
        status=status,
    )
    return _seal_result(
        result,
        kind="batched_bartlett_long_run_covariance",
        provenance=(
            ("input_shape", tuple(tensor.shape.as_list())),
            ("bandwidth", bandwidth),
            ("bandwidth_multiplier", bandwidth_multiplier),
            ("ridge_ladder", ridge_ladder),
            ("condition_number_max", condition_number_max),
            ("jit_compile", jit_compile),
        ),
    )


def _bootstrap_index_kernel(
    chain_count: int,
    draw_count: int,
    replication_count: int,
    block_length: int,
    bootstrap_count: int,
    seed: tf.Tensor,
    circular: bool,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    seeds = tf.random.experimental.stateless_split(seed, 2)
    block_count = draw_count // block_length
    maximum_start = draw_count if circular else draw_count - block_length + 1
    starts = tf.random.stateless_uniform(
        [bootstrap_count, chain_count, block_count],
        seed=seeds[0],
        minval=0,
        maxval=maximum_start,
        dtype=tf.int32,
    )
    offsets = tf.range(block_length, dtype=tf.int32)
    blocks = starts[..., tf.newaxis] + offsets
    if circular:
        blocks = tf.math.floormod(blocks, draw_count)
    draw_indices = tf.reshape(blocks, [bootstrap_count, chain_count, draw_count])
    replication_indices = tf.random.stateless_uniform(
        [bootstrap_count, chain_count, draw_count, replication_count],
        seed=seeds[1],
        minval=0,
        maxval=replication_count,
        dtype=tf.int32,
    )
    chain_indices = tf.broadcast_to(
        tf.range(chain_count, dtype=tf.int32)[tf.newaxis, :],
        [bootstrap_count, chain_count],
    )
    return chain_indices, draw_indices, replication_indices


_bootstrap_indices_xla = tf.function(
    _bootstrap_index_kernel, autograph=False, jit_compile=True
)
_bootstrap_indices_eager = tf.function(
    _bootstrap_index_kernel, autograph=False, jit_compile=False
)


def hierarchical_resample_indices(
    *,
    chain_count: int,
    draw_count: int,
    forecast_replication_count: int,
    block_length: int,
    bootstrap_count: int,
    seed: tf.Tensor,
    chain_mode: Literal["stratified_fixed_chains"] = "stratified_fixed_chains",
    block_mode: Literal["moving", "circular"] = "moving",
    jit_compile: bool = True,
) -> HierarchicalBootstrapIndices:
    """Materialize fixed-chain, draw-block, forecast-cluster bootstrap indices."""

    _require_bool(jit_compile, "jit_compile")
    if type(chain_mode) is not str or type(block_mode) is not str:
        raise PredictiveContractError("bootstrap modes must be strings")
    counts = (
        chain_count,
        draw_count,
        forecast_replication_count,
        block_length,
        bootstrap_count,
    )
    if any(not isinstance(value, int) or isinstance(value, bool) for value in counts):
        raise PredictiveContractError("bootstrap counts must be integers")
    if chain_count < 1 or draw_count < 2 or forecast_replication_count < 1:
        raise PredictiveContractError("invalid chain/draw/replication count")
    if block_length < 1 or draw_count % block_length:
        raise PredictiveContractError("block_length must divide draw_count")
    if draw_count // block_length < 2:
        raise PredictiveContractError("at least two complete blocks are required")
    if bootstrap_count < 2:
        raise PredictiveContractError("at least two bootstrap replicates are required")
    if chain_mode != "stratified_fixed_chains":
        raise PredictiveContractError("A3 forbids resampling the tiny chain population")
    if block_mode not in {"moving", "circular"}:
        raise PredictiveContractError("unknown block bootstrap mode")
    if not tf.is_tensor(seed) or seed.dtype != tf.int32 or seed.shape != (2,):
        raise PredictiveContractError("seed must be a static int32[2] tensor")
    kernel = _bootstrap_indices_xla if jit_compile else _bootstrap_indices_eager
    chain_indices, draw_indices, replication_indices = kernel(
        chain_count,
        draw_count,
        forecast_replication_count,
        block_length,
        bootstrap_count,
        seed,
        block_mode == "circular",
    )
    return HierarchicalBootstrapIndices(
        chain_indices=chain_indices,
        draw_indices=draw_indices,
        forecast_replication_indices=replication_indices,
        block_length=block_length,
        block_mode=block_mode,
        chain_mode=chain_mode,
        seed=seed,
        status=tf.constant(_VALID),
    )


def _normal_quantile(probability: tf.Tensor) -> tf.Tensor:
    return tf.sqrt(tf.constant(2.0, tf.float64)) * tf.math.erfinv(
        2.0 * probability - 1.0
    )


def _vector_quantile(values: tf.Tensor, probability: tf.Tensor) -> tf.Tensor:
    sorted_values = tf.sort(values)
    count = tf.shape(values, out_type=tf.int32)[0]
    position = probability * tf.cast(count - 1, tf.float64)
    lower_index = tf.cast(tf.floor(position), tf.int32)
    upper_index = tf.cast(tf.math.ceil(position), tf.int32)
    fraction = position - tf.floor(position)
    return (
        tf.gather(sorted_values, lower_index)
        + fraction
        * (tf.gather(sorted_values, upper_index) - tf.gather(sorted_values, lower_index))
    )


def _bonferroni_interval_kernel(
    estimate: tf.Tensor,
    standard_error: tf.Tensor,
    alpha: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    probability = 1.0 - alpha / (
        2.0 * tf.cast(tf.shape(estimate, out_type=tf.int32)[0], tf.float64)
    )
    critical = _normal_quantile(probability)
    return estimate - critical * standard_error, estimate + critical * standard_error, critical


def _bootstrap_max_interval_kernel(
    estimate: tf.Tensor,
    standard_error: tf.Tensor,
    bootstrap_estimates: tf.Tensor,
    alpha: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    centered_bootstrap = bootstrap_estimates - tf.reduce_mean(
        bootstrap_estimates, axis=0
    )
    maxima = tf.reduce_max(tf.abs(centered_bootstrap) / standard_error, axis=1)
    critical = _vector_quantile(maxima, 1.0 - alpha)
    return estimate - critical * standard_error, estimate + critical * standard_error, critical


def _bootstrap_max_auto_interval_kernel(
    estimate: tf.Tensor,
    bootstrap_estimates: tf.Tensor,
    alpha: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    centered_bootstrap = bootstrap_estimates - tf.reduce_mean(
        bootstrap_estimates, axis=0
    )
    standard_error = tf.sqrt(
        tf.reduce_sum(tf.square(centered_bootstrap), axis=0)
        / tf.cast(tf.shape(bootstrap_estimates, out_type=tf.int32)[0] - 1, tf.float64)
    )
    lower, upper, critical = _bootstrap_max_interval_kernel(
        estimate, standard_error, bootstrap_estimates, alpha
    )
    return lower, upper, critical, standard_error


_bonferroni_interval_xla = tf.function(
    _bonferroni_interval_kernel, autograph=False, jit_compile=True
)
_bonferroni_interval_eager = tf.function(
    _bonferroni_interval_kernel, autograph=False, jit_compile=False
)
_bootstrap_max_interval_xla = tf.function(
    _bootstrap_max_interval_kernel, autograph=False, jit_compile=True
)
_bootstrap_max_interval_eager = tf.function(
    _bootstrap_max_interval_kernel, autograph=False, jit_compile=False
)
_bootstrap_max_auto_interval_xla = tf.function(
    _bootstrap_max_auto_interval_kernel, autograph=False, jit_compile=True
)
_bootstrap_max_auto_interval_eager = tf.function(
    _bootstrap_max_auto_interval_kernel, autograph=False, jit_compile=False
)


def _mmd_block_interval_kernel(
    estimate: tf.Tensor,
    sequence: tf.Tensor,
    alpha: tf.Tensor,
    block_length: int,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    batches = _batch_means_kernel(sequence, block_length)
    pair_count = tf.shape(batches, out_type=tf.int32)[0]
    batch_count = tf.shape(batches, out_type=tf.int32)[1]
    anchored = batches - batches[:, :1]
    centered = anchored - tf.reduce_mean(anchored, axis=1, keepdims=True)
    pair_variances = tf.reduce_sum(tf.square(centered), axis=1) / tf.cast(
        batch_count - 1, tf.float64
    )
    pair_mean_variance_terms = pair_variances / tf.cast(
        pair_count * pair_count * batch_count, tf.float64
    )
    variance = tf.reduce_sum(pair_mean_variance_terms)
    standard_error = tf.sqrt(variance)
    degrees_of_freedom = tf.square(variance) / tf.reduce_sum(
        tf.square(pair_mean_variance_terms) / tf.cast(batch_count - 1, tf.float64)
    )
    critical = tfp.distributions.StudentT(
        df=degrees_of_freedom,
        loc=tf.constant(0.0, tf.float64),
        scale=tf.constant(1.0, tf.float64),
    ).quantile(1.0 - alpha / 2.0)
    return (
        estimate - critical * standard_error,
        estimate + critical * standard_error,
        standard_error,
        critical,
    )


_mmd_block_interval_xla = tf.function(
    _mmd_block_interval_kernel, autograph=False, jit_compile=True
)
_mmd_block_interval_eager = tf.function(
    _mmd_block_interval_kernel, autograph=False, jit_compile=False
)


def simultaneous_feature_intervals(
    estimate: tf.Tensor,
    *,
    feature_alpha: tf.Tensor | float,
    method: Literal["bonferroni_studentized", "bootstrap_max_statistic"],
    standard_error: tf.Tensor | None = None,
    bootstrap_estimates: tf.Tensor | None = None,
    minimum_bootstrap_count: int = 20,
    jit_compile: bool = True,
) -> SimultaneousIntervals:
    """Construct intervals controlling the supplied feature family alpha."""

    estimate = _require_tensor(estimate, "estimate", rank=1)
    _require_bool(jit_compile, "jit_compile")
    if type(method) is not str:
        raise PredictiveContractError("interval method must be a string")
    if type(minimum_bootstrap_count) is not int or minimum_bootstrap_count < 2:
        raise PredictiveContractError(
            "minimum_bootstrap_count must be an integer of at least two"
        )
    alpha = _require_probability(feature_alpha, "feature_alpha")
    feature_count = estimate.shape[0]
    if feature_count < 1:
        raise PredictiveContractError("the feature family cannot be empty")
    if method == "bonferroni_studentized":
        if standard_error is None:
            raise PredictiveContractError("Bonferroni intervals require standard_error")
        standard_error = _require_tensor(standard_error, "standard_error", rank=1)
        if standard_error.shape != estimate.shape:
            raise PredictiveContractError("standard_error must match estimate")
        interval_kernel = (
            _bonferroni_interval_xla if jit_compile else _bonferroni_interval_eager
        )
    elif method == "bootstrap_max_statistic":
        if bootstrap_estimates is None:
            raise PredictiveContractError("max-statistic intervals require bootstrap estimates")
        bootstrap_estimates = _require_tensor(
            bootstrap_estimates, "bootstrap_estimates", rank=2
        )
        if bootstrap_estimates.shape[1] != feature_count:
            raise PredictiveContractError("bootstrap feature dimension must match estimate")
        if bootstrap_estimates.shape[0] < minimum_bootstrap_count:
            raise PredictiveContractError("too few bootstrap replicates")
        automatic_standard_error = standard_error is None
        if not automatic_standard_error:
            standard_error = _require_tensor(standard_error, "standard_error", rank=1)
            if standard_error.shape != estimate.shape:
                raise PredictiveContractError("standard_error must match estimate")
            interval_kernel = (
                _bootstrap_max_interval_xla
                if jit_compile
                else _bootstrap_max_interval_eager
            )
    else:
        raise PredictiveContractError("unknown simultaneous interval method")
    if method == "bonferroni_studentized":
        lower, upper, critical = interval_kernel(estimate, standard_error, alpha)
    elif automatic_standard_error:
        automatic_kernel = (
            _bootstrap_max_auto_interval_xla
            if jit_compile
            else _bootstrap_max_auto_interval_eager
        )
        lower, upper, critical, standard_error = automatic_kernel(
            estimate, bootstrap_estimates, alpha
        )
    else:
        lower, upper, critical = interval_kernel(
            estimate, standard_error, bootstrap_estimates, alpha
        )
    try:
        tf.debugging.assert_positive(standard_error, "standard errors must be positive")
        tf.debugging.assert_all_finite(critical, "critical value must be finite")
    except tf.errors.InvalidArgumentError as exc:
        raise PredictiveContractError("invalid interval standard error or critical value") from exc
    result = SimultaneousIntervals(
        estimate=estimate,
        lower=lower,
        upper=upper,
        standard_error=standard_error,
        critical_value=critical,
        alpha=alpha,
        method=method,
        inference_admissible=True,
        status=tf.constant(_VALID),
    )
    return _seal_result(
        result,
        kind="feature_interval",
        provenance=(
            ("estimate", estimate),
            ("feature_alpha", alpha),
            ("method", method),
            ("standard_error", standard_error),
            ("bootstrap_estimates", bootstrap_estimates),
            ("minimum_bootstrap_count", minimum_bootstrap_count),
            ("jit_compile", jit_compile),
        ),
    )


def cross_chain_mmd_upper_interval(
    statistic: CrossChainLinearMMD,
    *,
    mmd_alpha: tf.Tensor | float,
    block_length: int,
    jit_compile: bool = True,
) -> MMDInterval:
    """Build a two-sided block-mean interval for admitted linear MMD contrasts."""

    if type(statistic) is not CrossChainLinearMMD:
        raise PredictiveContractError("statistic must be CrossChainLinearMMD")
    if not _authenticated_result(statistic, kind="cross_chain_linear_mmd"):
        raise PredictiveContractError(
            "statistic lacks authenticated cross-chain construction evidence"
        )
    _require_bool(jit_compile, "jit_compile")
    alpha = _require_probability(mmd_alpha, "mmd_alpha")
    estimate = _require_tensor(statistic.squared_mmd_linear, "squared_mmd_linear", rank=0)
    sequence = _require_tensor(
        statistic.kernel_contrast_sequence, "kernel_contrast_sequence", rank=2
    )
    if not tf.is_tensor(statistic.chain_pair_schedule) or statistic.chain_pair_schedule.dtype not in (
        tf.int32,
        tf.int64,
    ):
        raise PredictiveContractError("chain_pair_schedule must be an integer tensor")
    schedule = statistic.chain_pair_schedule
    if (
        schedule.shape.rank != 2
        or schedule.shape[1] != 2
        or not schedule.shape.is_fully_defined()
        or schedule.shape[0] != sequence.shape[0]
    ):
        raise PredictiveContractError("chain schedule and contrast sequence disagree")
    schedule_values = [
        int(tf.get_static_value(value)) for value in tf.reshape(schedule, [-1])
    ]
    if (
        any(value < 0 for value in schedule_values)
        or any(first == second for first, second in zip(schedule_values[::2], schedule_values[1::2]))
        or len(set(schedule_values)) != len(schedule_values)
    ):
        raise PredictiveContractError("chain schedule must contain distinct nonnegative chains")
    _require_bool(
        statistic.independent_arm_banks_verified,
        "statistic.independent_arm_banks_verified",
    )
    _require_bool(statistic.stationarity_required, "statistic.stationarity_required")
    _require_bool(statistic.stationarity_verified, "statistic.stationarity_verified")
    _require_bool(statistic.mixing_verified, "statistic.mixing_verified")
    _require_bool(statistic.mechanics_only, "statistic.mechanics_only")
    _require_bool(statistic.inference_admissible, "statistic.inference_admissible")
    expected_estimate = tf.reduce_mean(sequence)
    estimate_scale = tf.maximum(
        tf.constant(1.0, tf.float64),
        tf.maximum(tf.abs(estimate), tf.abs(expected_estimate)),
    )
    if bool(
        tf.abs(estimate - expected_estimate)
        > tf.constant(512.0 * 2.220446049250313e-16, tf.float64) * estimate_scale
    ):
        raise PredictiveContractError("stored linear MMD estimate disagrees with contrasts")
    admitted_flags = (
        statistic.independent_arm_banks_verified
        and statistic.stationarity_required
        and statistic.stationarity_verified
        and statistic.mixing_verified
        and not statistic.mechanics_only
        and sequence.shape[0] >= 2
    )
    if statistic.inference_admissible and not admitted_flags:
        raise PredictiveContractError("linear MMD inference flags are internally inconsistent")
    if statistic.inference_admissible and not _has_exact_status(statistic.status, _VALID):
        raise PredictiveContractError("admissible linear MMD requires exact VALID status")
    if not statistic.inference_admissible or not _has_exact_status(statistic.status, _VALID):
        zero = tf.constant(0.0, tf.float64)
        result = MMDInterval(
            estimate=estimate,
            lower=zero,
            upper=zero,
            standard_error=zero,
            critical_value=zero,
            alpha=alpha,
            block_count=tf.constant(0, tf.int32),
            inference_admissible=False,
            status=tf.constant(_INVALID),
        )
        return _seal_result(
            result,
            kind="mmd_interval",
            provenance=(
                ("statistic_signature", statistic.construction_signature),
                ("mmd_alpha", alpha),
                ("block_length", block_length),
                ("jit_compile", jit_compile),
                ("invalid_reason", "source_not_admissible"),
            ),
        )
    _draw_count, batch_count = _batch_geometry(sequence, block_length)
    pair_count = sequence.shape[0]
    interval_kernel = _mmd_block_interval_xla if jit_compile else _mmd_block_interval_eager
    lower, upper, standard_error, critical = interval_kernel(
        estimate, sequence, alpha, block_length
    )
    contrast_scale = tf.maximum(
        tf.constant(1.0, tf.float64), tf.reduce_max(tf.abs(sequence))
    )
    degeneracy_floor = (
        tf.constant(4096.0 * 2.220446049250313e-16, tf.float64)
        * contrast_scale
        * tf.sqrt(tf.cast(tf.size(sequence), tf.float64))
    )
    try:
        tf.debugging.assert_greater(
            standard_error,
            degeneracy_floor,
            "long-run standard error is zero or roundoff-degenerate",
        )
        tf.debugging.assert_all_finite(critical, "critical value must be finite")
    except tf.errors.InvalidArgumentError:
        zero = tf.constant(0.0, tf.float64)
        result = MMDInterval(
            estimate=estimate,
            lower=zero,
            upper=zero,
            standard_error=standard_error,
            critical_value=critical,
            alpha=alpha,
            block_count=tf.constant(pair_count * batch_count, tf.int32),
            inference_admissible=False,
            status=tf.constant(_INVALID),
        )
        return _seal_result(
            result,
            kind="mmd_interval",
            provenance=(
                ("statistic_signature", statistic.construction_signature),
                ("mmd_alpha", alpha),
                ("block_length", block_length),
                ("jit_compile", jit_compile),
                ("invalid_reason", "nonfinite_or_degenerate_uncertainty"),
            ),
        )
    result = MMDInterval(
        estimate=estimate,
        lower=lower,
        upper=upper,
        standard_error=standard_error,
        critical_value=critical,
        alpha=alpha,
        block_count=tf.constant(pair_count * batch_count, tf.int32),
        inference_admissible=True,
        status=tf.constant(_VALID),
    )
    return _seal_result(
        result,
        kind="mmd_interval",
        provenance=(
            ("statistic_signature", statistic.construction_signature),
            ("statistic_result_fingerprint", _result_fingerprint("cross_chain_linear_mmd", statistic)),
            ("mmd_alpha", alpha),
            ("block_length", block_length),
            ("jit_compile", jit_compile),
        ),
    )


def _feature_status(
    intervals: SimultaneousIntervals,
    margins: tf.Tensor,
) -> DecisionStatus:
    material = tf.reduce_any(
        tf.logical_or(intervals.lower > margins, intervals.upper < -margins)
    )
    inside = tf.reduce_all(
        tf.logical_and(intervals.lower > -margins, intervals.upper < margins)
    )
    if bool(tf.get_static_value(material)):
        return "MATERIAL_DIFFERENCE"
    if bool(tf.get_static_value(inside)):
        return "PASS"
    return "INCONCLUSIVE_UNDERPOWERED"


def _mmd_status(interval: MMDInterval, tolerance: tf.Tensor) -> DecisionStatus:
    if bool(tf.get_static_value(interval.lower > tolerance)):
        return "MATERIAL_DIFFERENCE"
    if bool(tf.get_static_value(interval.upper < tolerance)):
        return "PASS"
    return "INCONCLUSIVE_UNDERPOWERED"


def _validated_feature_interval(
    intervals: SimultaneousIntervals,
    expected_shape: tf.TensorShape,
) -> tuple[bool, tf.Tensor]:
    try:
        if type(intervals) is not SimultaneousIntervals:
            raise PredictiveContractError("feature interval has the wrong result type")
        values = (
            _require_tensor(intervals.estimate, "feature estimate", rank=1),
            _require_tensor(intervals.lower, "feature lower", rank=1),
            _require_tensor(intervals.upper, "feature upper", rank=1),
            _require_tensor(intervals.standard_error, "feature standard_error", rank=1),
        )
        critical = _require_tensor(intervals.critical_value, "feature critical_value", rank=0)
        _require_probability(intervals.alpha, "feature interval alpha")
        _require_bool(intervals.inference_admissible, "feature inference_admissible")
        if type(intervals.method) is not str or intervals.method not in {
            "bonferroni_studentized",
            "bootstrap_max_statistic",
        }:
            raise PredictiveContractError("feature interval method is invalid")
        if not _has_declared_status(intervals.status):
            raise PredictiveContractError("feature status must be a declared scalar string")
        if any(value.shape != expected_shape for value in values):
            raise PredictiveContractError("feature interval tensors have inconsistent shapes")
        tf.debugging.assert_positive(values[3], "feature standard errors must be positive")
        tf.debugging.assert_positive(critical, "feature critical value must be positive")
        tf.debugging.assert_less_equal(values[1], values[0], "feature lower exceeds estimate")
        tf.debugging.assert_less_equal(values[0], values[2], "feature estimate exceeds upper")
        expected_half_width = critical * values[3]
        if not _scale_aware_equal(values[1], values[0] - expected_half_width) or not _scale_aware_equal(
            values[2], values[0] + expected_half_width
        ):
            raise PredictiveContractError("feature bounds disagree with interval algebra")
    except (AttributeError, PredictiveContractError, tf.errors.InvalidArgumentError):
        return False, tf.zeros(expected_shape, tf.float64)
    return True, values[2] - values[1]


def _validated_mmd_interval(interval: MMDInterval) -> tuple[bool, tf.Tensor]:
    try:
        if type(interval) is not MMDInterval:
            raise PredictiveContractError("MMD interval has the wrong result type")
        estimate = _require_tensor(interval.estimate, "MMD estimate", rank=0)
        lower = _require_tensor(interval.lower, "MMD lower", rank=0)
        upper = _require_tensor(interval.upper, "MMD upper", rank=0)
        standard_error = _require_tensor(
            interval.standard_error, "MMD standard_error", rank=0
        )
        critical = _require_tensor(interval.critical_value, "MMD critical_value", rank=0)
        _require_probability(interval.alpha, "MMD interval alpha")
        _require_bool(interval.inference_admissible, "MMD inference_admissible")
        if not _has_declared_status(interval.status):
            raise PredictiveContractError("MMD status must be a declared scalar string")
        if (
            not tf.is_tensor(interval.block_count)
            or interval.block_count.dtype not in (tf.int32, tf.int64)
            or interval.block_count.shape.rank != 0
        ):
            raise PredictiveContractError("MMD block_count must be an integer scalar tensor")
        tf.debugging.assert_positive(interval.block_count, "MMD block_count must be positive")
        tf.debugging.assert_positive(standard_error, "MMD standard error must be positive")
        tf.debugging.assert_positive(critical, "MMD critical value must be positive")
        tf.debugging.assert_less_equal(lower, estimate, "MMD lower exceeds estimate")
        tf.debugging.assert_less_equal(estimate, upper, "MMD estimate exceeds upper")
        expected_half_width = critical * standard_error
        if not _scale_aware_equal(lower, estimate - expected_half_width) or not _scale_aware_equal(
            upper, estimate + expected_half_width
        ):
            raise PredictiveContractError("MMD bounds disagree with interval algebra")
    except (AttributeError, PredictiveContractError, tf.errors.InvalidArgumentError):
        return False, tf.constant(0.0, tf.float64)
    return True, upper - lower


def classify_predictive_evidence(
    feature_intervals: SimultaneousIntervals,
    mmd_interval: MMDInterval,
    *,
    margins: tf.Tensor,
    mmd_tolerance: tf.Tensor,
    total_alpha: tf.Tensor | float,
    feature_alpha: tf.Tensor | float,
    mmd_alpha: tf.Tensor | float,
    mechanics_only: bool = False,
) -> PredictiveDecision:
    """Apply fail-closed practical-equivalence branch logic."""

    margins = _require_tensor(margins, "margins", rank=1)
    tolerance = _require_tensor(mmd_tolerance, "mmd_tolerance", rank=0)
    _require_bool(mechanics_only, "mechanics_only")
    total = _require_probability(total_alpha, "total_alpha")
    feature = _require_probability(feature_alpha, "feature_alpha")
    mmd = _require_probability(mmd_alpha, "mmd_alpha")
    hard_vetoes: list[str] = []
    feature_valid, feature_width = _validated_feature_interval(
        feature_intervals, margins.shape
    )
    mmd_valid, mmd_width = _validated_mmd_interval(mmd_interval)
    feature_authenticated = _authenticated_result(
        feature_intervals, kind="feature_interval"
    )
    mmd_authenticated = _authenticated_result(mmd_interval, kind="mmd_interval")
    if not feature_authenticated:
        hard_vetoes.append("FEATURE_INTERVAL_UNAUTHENTICATED")
    if not mmd_authenticated:
        hard_vetoes.append("MMD_INTERVAL_UNAUTHENTICATED")
    if not feature_valid:
        hard_vetoes.append("FEATURE_INTERVAL_MALFORMED")
    if not mmd_valid:
        hard_vetoes.append("MMD_INTERVAL_MALFORMED")
    try:
        tf.debugging.assert_positive(margins, "margins must be positive")
        tf.debugging.assert_positive(tolerance, "MMD tolerance must be positive")
    except tf.errors.InvalidArgumentError as exc:
        raise PredictiveContractError("margins and MMD tolerance must be positive") from exc
    if float(tf.get_static_value(feature + mmd)) > float(tf.get_static_value(total)):
        hard_vetoes.append("INVALID_JOINT_ALPHA_ALLOCATION")
    if feature_valid and abs(float(tf.get_static_value(feature_intervals.alpha - feature))) > 1e-15:
        hard_vetoes.append("FEATURE_ALPHA_BINDING_MISMATCH")
    if mmd_valid and abs(float(tf.get_static_value(mmd_interval.alpha - mmd))) > 1e-15:
        hard_vetoes.append("MMD_ALPHA_BINDING_MISMATCH")
    if (
        not feature_valid
        or not _has_exact_status(feature_intervals.status, _VALID)
        or not feature_intervals.inference_admissible
    ):
        hard_vetoes.append("FEATURE_INTERVAL_NOT_ADMISSIBLE")
    if (
        not mmd_valid
        or not _has_exact_status(mmd_interval.status, _VALID)
        or not mmd_interval.inference_admissible
    ):
        hard_vetoes.append("MMD_INTERVAL_NOT_ADMISSIBLE")
    if mechanics_only:
        hard_vetoes.append("MECHANICS_ONLY_CANNOT_PASS")
    if hard_vetoes:
        return PredictiveDecision(
            status="INVALID_HARD_VETO",
            primary_interval_status="INVALID_HARD_VETO",
            mmd_upper_bound_status="INVALID_HARD_VETO",
            hard_veto_codes=tuple(hard_vetoes),
            explanatory_diagnostics={
                "feature_interval_width": feature_width,
                "mmd_interval_width": mmd_width,
            },
        )
    feature_result = _feature_status(feature_intervals, margins)
    mmd_result = _mmd_status(mmd_interval, tolerance)
    if "MATERIAL_DIFFERENCE" in (feature_result, mmd_result):
        overall: DecisionStatus = "MATERIAL_DIFFERENCE"
    elif feature_result == "PASS" and mmd_result == "PASS":
        overall = "PASS"
    else:
        overall = "INCONCLUSIVE_UNDERPOWERED"
    return PredictiveDecision(
        status=overall,
        primary_interval_status=feature_result,
        mmd_upper_bound_status=mmd_result,
        hard_veto_codes=(),
        explanatory_diagnostics={
            "feature_interval_width": feature_intervals.upper - feature_intervals.lower,
            "mmd_interval_width": mmd_interval.upper - mmd_interval.lower,
        },
    )


def proper_score_loss(
    horizon_weights: tf.Tensor,
) -> ProperScoreLoss:
    """Build the local symmetric-log-score loss in mean/log-variance order."""

    weights = _require_tensor(horizon_weights, "horizon_weights", rank=1)
    horizon_count = int(weights.shape[0])
    if horizon_count < 1:
        raise PredictiveContractError("at least one horizon weight is required")
    try:
        tf.debugging.assert_positive(weights, "horizon weights must be positive")
        tf.debugging.assert_near(
            tf.reduce_sum(weights),
            tf.constant(1.0, tf.float64),
            atol=tf.constant(2048.0 * 2.220446049250313e-16, tf.float64),
            rtol=tf.constant(0.0, tf.float64),
            message="horizon weights must sum to one",
        )
    except tf.errors.InvalidArgumentError as exc:
        raise PredictiveContractError(
            "horizon weights must be positive and sum to one"
        ) from exc
    diagonal = tf.concat((0.5 * weights, 0.25 * weights), axis=0)
    result = ProperScoreLoss(
        horizon_weights=weights,
        loss_matrix=tf.linalg.diag(diagonal),
        horizon_count=horizon_count,
        status=tf.constant(_VALID),
    )
    return _seal_result(
        result,
        kind="proper_score_loss",
        provenance=(("horizon_weights", weights),),
    )


def horizon_proper_score_loss(
    horizon_count: int,
    horizon_index: int,
) -> ProperScoreLoss:
    """Build one horizon's regret inside the full joint feature space."""

    if type(horizon_count) is not int or horizon_count < 1:
        raise PredictiveContractError("horizon_count must be a positive integer")
    if (
        type(horizon_index) is not int
        or horizon_index < 0
        or horizon_index >= horizon_count
    ):
        raise PredictiveContractError("horizon_index is outside the horizon range")
    weights = tf.one_hot(horizon_index, horizon_count, dtype=tf.float64)
    diagonal = tf.concat((0.5 * weights, 0.25 * weights), axis=0)
    result = ProperScoreLoss(
        horizon_weights=weights,
        loss_matrix=tf.linalg.diag(diagonal),
        horizon_count=horizon_count,
        status=tf.constant(_VALID),
    )
    return _seal_result(
        result,
        kind="proper_score_loss",
        provenance=(
            ("horizon_count", horizon_count),
            ("horizon_index", horizon_index),
            ("horizon_weights", weights),
        ),
    )


def _quadratic_loss_bounds_kernel(
    estimate: tf.Tensor,
    covariance: tf.Tensor,
    loss_matrix: tf.Tensor,
    radius_squared: tf.Tensor,
    bisection_iterations: int,
) -> tuple[tf.Tensor, ...]:
    radius = tf.sqrt(radius_squared)
    factor = tf.linalg.cholesky(covariance)
    transformed_matrix = tf.matmul(
        factor,
        tf.matmul(loss_matrix, factor),
        transpose_a=True,
    )
    transformed_matrix = 0.5 * (
        transformed_matrix + tf.transpose(transformed_matrix)
    )
    transformed_gradient = tf.linalg.matvec(
        factor, tf.linalg.matvec(loss_matrix, estimate), transpose_a=True
    )
    eigenvalues, eigenvectors = tf.linalg.eigh(transformed_matrix)
    gradient_coordinates = tf.linalg.matvec(
        eigenvectors, transformed_gradient, transpose_a=True
    )
    dimension = estimate.shape[0]

    eigen_scale_minimum = tf.maximum(
        tf.constant(1.0, tf.float64), tf.reduce_max(tf.abs(eigenvalues))
    )
    null_tolerance = (
        tf.constant(4096.0 * 2.220446049250313e-16, tf.float64)
        * eigen_scale_minimum
    )
    positive_eigenvalue = eigenvalues > null_tolerance
    unconstrained_minimum = tf.where(
        positive_eigenvalue,
        -tf.math.divide_no_nan(gradient_coordinates, eigenvalues),
        tf.zeros_like(gradient_coordinates),
    )
    null_gradient_norm = tf.linalg.norm(
        tf.where(positive_eigenvalue, tf.zeros_like(gradient_coordinates), gradient_coordinates)
    )
    gradient_scale_minimum = tf.maximum(
        tf.constant(1.0, tf.float64), tf.linalg.norm(gradient_coordinates)
    )
    null_gradient_tolerance = (
        tf.constant(4096.0 * 2.220446049250313e-16, tf.float64)
        * gradient_scale_minimum
    )
    unconstrained_inside = tf.logical_and(
        null_gradient_norm <= null_gradient_tolerance,
        tf.linalg.norm(unconstrained_minimum) <= radius,
    )

    minimum_low = tf.constant(0.0, tf.float64)
    minimum_high = tf.maximum(
        tf.constant(1.0, tf.float64),
        tf.linalg.norm(gradient_coordinates) / radius,
    )
    for _ in range(64):
        norm_at_high = tf.linalg.norm(
            gradient_coordinates / (eigenvalues + minimum_high)
        )
        minimum_high = tf.where(
            norm_at_high > radius, 2.0 * minimum_high, minimum_high
        )
    for _ in range(bisection_iterations):
        midpoint = 0.5 * (minimum_low + minimum_high)
        norm_at_midpoint = tf.linalg.norm(
            gradient_coordinates / (eigenvalues + midpoint)
        )
        minimum_low = tf.where(norm_at_midpoint > radius, midpoint, minimum_low)
        minimum_high = tf.where(norm_at_midpoint > radius, minimum_high, midpoint)
    minimum_multiplier = tf.where(
        unconstrained_inside, tf.constant(0.0, tf.float64), minimum_high
    )
    minimum_coordinates = tf.where(
        unconstrained_inside,
        unconstrained_minimum,
        -gradient_coordinates / (eigenvalues + minimum_multiplier),
    )

    largest_eigenvalue = eigenvalues[-1]
    eigen_scale = tf.maximum(tf.constant(1.0, tf.float64), tf.abs(largest_eigenvalue))
    top_tolerance = tf.constant(4096.0 * 2.220446049250313e-16, tf.float64) * eigen_scale
    top_mask = largest_eigenvalue - eigenvalues <= top_tolerance
    top_gradient_norm = tf.linalg.norm(tf.where(top_mask, gradient_coordinates, 0.0))
    gradient_scale = tf.maximum(
        tf.constant(1.0, tf.float64), tf.linalg.norm(gradient_coordinates)
    )
    hard_tolerance = (
        tf.constant(4096.0 * 2.220446049250313e-16, tf.float64) * gradient_scale
    )
    denominator_at_top = largest_eigenvalue - eigenvalues
    particular_at_top = tf.where(
        top_mask,
        tf.zeros_like(gradient_coordinates),
        tf.math.divide_no_nan(gradient_coordinates, denominator_at_top),
    )
    particular_norm_squared = tf.reduce_sum(tf.square(particular_at_top))
    hard_case = tf.logical_and(
        top_gradient_norm <= hard_tolerance,
        particular_norm_squared <= radius_squared,
    )

    generic_gap = tf.maximum(
        top_gradient_norm / (2.0 * radius),
        tf.constant(16.0 * 2.220446049250313e-16, tf.float64) * eigen_scale,
    )
    maximum_low = tf.where(
        top_gradient_norm > hard_tolerance,
        largest_eigenvalue + generic_gap,
        largest_eigenvalue,
    )
    maximum_high = largest_eigenvalue + tf.maximum(
        tf.constant(1.0, tf.float64),
        tf.linalg.norm(gradient_coordinates) / radius,
    )

    def maximum_coordinates(multiplier: tf.Tensor) -> tf.Tensor:
        denominator = multiplier - eigenvalues
        return tf.where(
            tf.logical_and(top_mask, denominator <= top_tolerance),
            tf.zeros_like(gradient_coordinates),
            tf.math.divide_no_nan(gradient_coordinates, denominator),
        )

    for _ in range(64):
        norm_at_high = tf.linalg.norm(maximum_coordinates(maximum_high))
        maximum_high = tf.where(
            norm_at_high > radius,
            largest_eigenvalue + 2.0 * (maximum_high - largest_eigenvalue),
            maximum_high,
        )
    for _ in range(bisection_iterations):
        midpoint = 0.5 * (maximum_low + maximum_high)
        norm_at_midpoint = tf.linalg.norm(maximum_coordinates(midpoint))
        maximum_low = tf.where(norm_at_midpoint > radius, midpoint, maximum_low)
        maximum_high = tf.where(norm_at_midpoint > radius, maximum_high, midpoint)
    generic_maximum = maximum_coordinates(maximum_high)

    remaining_radius = tf.sqrt(tf.maximum(radius_squared - particular_norm_squared, 0.0))
    first_top_index = tf.argmax(tf.cast(top_mask, tf.int32), output_type=tf.int32)
    hard_direction = tf.one_hot(first_top_index, dimension, dtype=tf.float64)
    hard_maximum = particular_at_top + remaining_radius * hard_direction
    maximum_coordinates_result = tf.where(hard_case, hard_maximum, generic_maximum)
    maximum_multiplier = tf.where(hard_case, largest_eigenvalue, maximum_high)

    minimum_u = tf.linalg.matvec(eigenvectors, minimum_coordinates)
    maximum_u = tf.linalg.matvec(eigenvectors, maximum_coordinates_result)
    minimum_optimizer = estimate + tf.linalg.matvec(factor, minimum_u)
    maximum_optimizer = estimate + tf.linalg.matvec(factor, maximum_u)
    point_loss = tf.tensordot(estimate, tf.linalg.matvec(loss_matrix, estimate), 1)
    lower_bound = tf.tensordot(
        minimum_optimizer, tf.linalg.matvec(loss_matrix, minimum_optimizer), 1
    )
    upper_bound = tf.tensordot(
        maximum_optimizer, tf.linalg.matvec(loss_matrix, maximum_optimizer), 1
    )
    minimum_stationarity = tf.linalg.matvec(
        transformed_matrix, minimum_u
    ) + transformed_gradient + minimum_multiplier * minimum_u
    maximum_stationarity = tf.linalg.matvec(
        transformed_matrix, maximum_u
    ) + transformed_gradient - maximum_multiplier * maximum_u
    minimum_constraint_residual = tf.where(
        unconstrained_inside,
        tf.maximum(tf.linalg.norm(minimum_u) - radius, 0.0),
        tf.abs(tf.linalg.norm(minimum_u) - radius),
    )
    maximum_constraint_residual = tf.abs(tf.linalg.norm(maximum_u) - radius)
    lower_kkt = tf.maximum(
        tf.linalg.norm(minimum_stationarity), minimum_constraint_residual
    )
    upper_kkt = tf.maximum(
        tf.linalg.norm(maximum_stationarity), maximum_constraint_residual
    )
    return (
        point_loss,
        lower_bound,
        upper_bound,
        minimum_optimizer,
        maximum_optimizer,
        lower_kkt,
        upper_kkt,
        tf.linalg.eigvalsh(covariance),
    )


_quadratic_loss_bounds_xla = tf.function(
    _quadratic_loss_bounds_kernel, autograph=False, jit_compile=True
)
_quadratic_loss_bounds_eager = tf.function(
    _quadratic_loss_bounds_kernel, autograph=False, jit_compile=False
)


def _batched_quadratic_loss_bounds_kernel(
    estimate: tf.Tensor,
    covariance: tf.Tensor,
    loss_matrices: tf.Tensor,
    radius_squared: tf.Tensor,
    bisection_iterations: int,
) -> tuple[tf.Tensor, ...]:
    batch_size = estimate.shape[0]
    loss_count = loss_matrices.shape[0]
    dimension = estimate.shape[1]
    radius = tf.sqrt(radius_squared)
    factor = tf.linalg.cholesky(covariance)
    transformed = tf.einsum("bji,ljk,bkm->blim", factor, loss_matrices, factor)
    transformed = 0.5 * (transformed + tf.transpose(transformed, [0, 1, 3, 2]))
    loss_gradient = tf.einsum("lij,bj->bli", loss_matrices, estimate)
    gradient = tf.einsum("bji,blj->bli", factor, loss_gradient)
    eigenvalues, eigenvectors = tf.linalg.eigh(transformed)
    coordinates = tf.einsum("blji,blj->bli", eigenvectors, gradient)

    eigen_scale_minimum = tf.maximum(
        tf.constant(1.0, tf.float64), tf.reduce_max(tf.abs(eigenvalues), axis=2)
    )
    null_tolerance = (
        tf.constant(4096.0 * 2.220446049250313e-16, tf.float64)
        * eigen_scale_minimum
    )
    positive = eigenvalues > null_tolerance[:, :, tf.newaxis]
    unconstrained = tf.where(
        positive,
        -tf.math.divide_no_nan(coordinates, eigenvalues),
        tf.zeros_like(coordinates),
    )
    null_gradient_norm = tf.linalg.norm(
        tf.where(positive, tf.zeros_like(coordinates), coordinates), axis=2
    )
    gradient_scale = tf.maximum(
        tf.constant(1.0, tf.float64), tf.linalg.norm(coordinates, axis=2)
    )
    unconstrained_inside = tf.logical_and(
        null_gradient_norm
        <= tf.constant(4096.0 * 2.220446049250313e-16, tf.float64) * gradient_scale,
        tf.linalg.norm(unconstrained, axis=2) <= radius,
    )
    minimum_low = tf.zeros([batch_size, loss_count], tf.float64)
    minimum_high = tf.maximum(
        tf.ones([batch_size, loss_count], tf.float64),
        tf.linalg.norm(coordinates, axis=2) / radius,
    )

    def expand_minimum(
        iteration: tf.Tensor, high: tf.Tensor
    ) -> tuple[tf.Tensor, tf.Tensor]:
        norm_at_high = tf.linalg.norm(
            coordinates / (eigenvalues + high[:, :, tf.newaxis]), axis=2
        )
        return iteration + 1, tf.where(
            norm_at_high > radius, 2.0 * high, high
        )

    _, minimum_high = tf.while_loop(
        lambda iteration, high: iteration < 64,
        expand_minimum,
        (tf.constant(0), minimum_high),
        maximum_iterations=64,
    )

    def bisect_minimum(
        iteration: tf.Tensor, low: tf.Tensor, high: tf.Tensor
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        midpoint = 0.5 * (low + high)
        norm_at_midpoint = tf.linalg.norm(
            coordinates / (eigenvalues + midpoint[:, :, tf.newaxis]), axis=2
        )
        return (
            iteration + 1,
            tf.where(norm_at_midpoint > radius, midpoint, low),
            tf.where(norm_at_midpoint > radius, high, midpoint),
        )

    _, minimum_low, minimum_high = tf.while_loop(
        lambda iteration, low, high: iteration < bisection_iterations,
        bisect_minimum,
        (tf.constant(0), minimum_low, minimum_high),
        maximum_iterations=bisection_iterations,
    )
    minimum_multiplier = tf.where(
        unconstrained_inside, tf.constant(0.0, tf.float64), minimum_high
    )
    minimum_coordinates = tf.where(
        unconstrained_inside[:, :, tf.newaxis],
        unconstrained,
        -coordinates / (eigenvalues + minimum_multiplier[:, :, tf.newaxis]),
    )

    largest = eigenvalues[:, :, -1]
    eigen_scale = tf.maximum(tf.constant(1.0, tf.float64), tf.abs(largest))
    top_tolerance = (
        tf.constant(4096.0 * 2.220446049250313e-16, tf.float64) * eigen_scale
    )
    top_mask = largest[:, :, tf.newaxis] - eigenvalues <= top_tolerance[:, :, tf.newaxis]
    top_gradient_norm = tf.linalg.norm(
        tf.where(top_mask, coordinates, tf.zeros_like(coordinates)), axis=2
    )
    gradient_scale = tf.maximum(
        tf.constant(1.0, tf.float64), tf.linalg.norm(coordinates, axis=2)
    )
    hard_tolerance = (
        tf.constant(4096.0 * 2.220446049250313e-16, tf.float64) * gradient_scale
    )
    denominator_at_top = largest[:, :, tf.newaxis] - eigenvalues
    particular = tf.where(
        top_mask,
        tf.zeros_like(coordinates),
        tf.math.divide_no_nan(coordinates, denominator_at_top),
    )
    particular_norm_squared = tf.reduce_sum(tf.square(particular), axis=2)
    hard_case = tf.logical_and(
        top_gradient_norm <= hard_tolerance,
        particular_norm_squared <= radius_squared,
    )
    generic_gap = tf.maximum(
        top_gradient_norm / (2.0 * radius),
        tf.constant(16.0 * 2.220446049250313e-16, tf.float64) * eigen_scale,
    )
    maximum_low = tf.where(
        top_gradient_norm > hard_tolerance, largest + generic_gap, largest
    )
    maximum_high = largest + tf.maximum(
        tf.constant(1.0, tf.float64), tf.linalg.norm(coordinates, axis=2) / radius
    )

    def maximum_coordinates(multiplier: tf.Tensor) -> tf.Tensor:
        denominator = multiplier[:, :, tf.newaxis] - eigenvalues
        return tf.where(
            tf.logical_and(top_mask, denominator <= top_tolerance[:, :, tf.newaxis]),
            tf.zeros_like(coordinates),
            tf.math.divide_no_nan(coordinates, denominator),
        )

    def expand_maximum(
        iteration: tf.Tensor, high: tf.Tensor
    ) -> tuple[tf.Tensor, tf.Tensor]:
        norm_at_high = tf.linalg.norm(maximum_coordinates(high), axis=2)
        return iteration + 1, tf.where(
            norm_at_high > radius,
            largest + 2.0 * (high - largest),
            high,
        )

    _, maximum_high = tf.while_loop(
        lambda iteration, high: iteration < 64,
        expand_maximum,
        (tf.constant(0), maximum_high),
        maximum_iterations=64,
    )

    def bisect_maximum(
        iteration: tf.Tensor, low: tf.Tensor, high: tf.Tensor
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        midpoint = 0.5 * (low + high)
        norm_at_midpoint = tf.linalg.norm(maximum_coordinates(midpoint), axis=2)
        return (
            iteration + 1,
            tf.where(norm_at_midpoint > radius, midpoint, low),
            tf.where(norm_at_midpoint > radius, high, midpoint),
        )

    _, maximum_low, maximum_high = tf.while_loop(
        lambda iteration, low, high: iteration < bisection_iterations,
        bisect_maximum,
        (tf.constant(0), maximum_low, maximum_high),
        maximum_iterations=bisection_iterations,
    )
    generic_maximum = maximum_coordinates(maximum_high)
    remaining = tf.sqrt(tf.maximum(radius_squared - particular_norm_squared, 0.0))
    first_top = tf.argmax(tf.cast(top_mask, tf.int32), axis=2, output_type=tf.int32)
    hard_direction = tf.one_hot(first_top, dimension, dtype=tf.float64)
    maximum_coordinates_result = tf.where(
        hard_case[:, :, tf.newaxis],
        particular + remaining[:, :, tf.newaxis] * hard_direction,
        generic_maximum,
    )
    maximum_multiplier = tf.where(hard_case, largest, maximum_high)

    minimum_u = tf.einsum("blij,blj->bli", eigenvectors, minimum_coordinates)
    maximum_u = tf.einsum("blij,blj->bli", eigenvectors, maximum_coordinates_result)
    minimum_optimizer = estimate[:, tf.newaxis, :] + tf.einsum(
        "bij,blj->bli", factor, minimum_u
    )
    maximum_optimizer = estimate[:, tf.newaxis, :] + tf.einsum(
        "bij,blj->bli", factor, maximum_u
    )
    point_loss = tf.einsum("bi,lij,bj->bl", estimate, loss_matrices, estimate)
    lower_bound = tf.einsum(
        "bli,lij,blj->bl", minimum_optimizer, loss_matrices, minimum_optimizer
    )
    upper_bound = tf.einsum(
        "bli,lij,blj->bl", maximum_optimizer, loss_matrices, maximum_optimizer
    )
    minimum_stationarity = (
        tf.einsum("blij,blj->bli", transformed, minimum_u)
        + gradient
        + minimum_multiplier[:, :, tf.newaxis] * minimum_u
    )
    maximum_stationarity = (
        tf.einsum("blij,blj->bli", transformed, maximum_u)
        + gradient
        - maximum_multiplier[:, :, tf.newaxis] * maximum_u
    )
    minimum_constraint = tf.where(
        unconstrained_inside,
        tf.maximum(tf.linalg.norm(minimum_u, axis=2) - radius, 0.0),
        tf.abs(tf.linalg.norm(minimum_u, axis=2) - radius),
    )
    maximum_constraint = tf.abs(tf.linalg.norm(maximum_u, axis=2) - radius)
    lower_kkt = tf.maximum(
        tf.linalg.norm(minimum_stationarity, axis=2), minimum_constraint
    )
    upper_kkt = tf.maximum(
        tf.linalg.norm(maximum_stationarity, axis=2), maximum_constraint
    )
    return (
        point_loss,
        lower_bound,
        upper_bound,
        lower_kkt,
        upper_kkt,
        tf.linalg.eigvalsh(covariance),
    )


_batched_quadratic_loss_bounds_xla = tf.function(
    _batched_quadratic_loss_bounds_kernel, autograph=False, jit_compile=True
)
_batched_quadratic_loss_bounds_eager = tf.function(
    _batched_quadratic_loss_bounds_kernel, autograph=False, jit_compile=False
)


def quadratic_loss_confidence_bounds(
    estimate: tf.Tensor,
    covariance: tf.Tensor,
    loss: ProperScoreLoss,
    *,
    alpha: tf.Tensor | float = 0.05,
    bisection_iterations: int = 96,
    kkt_tolerance: float = 1.0e-8,
    jit_compile: bool = True,
) -> QuadraticLossBounds:
    """Return exact quadratic-loss extrema over a joint Wald ellipsoid."""

    estimate = _require_tensor(estimate, "estimate", rank=1)
    covariance = _require_tensor(covariance, "covariance", rank=2)
    _require_bool(jit_compile, "jit_compile")
    if type(loss) is not ProperScoreLoss or not _authenticated_result(
        loss, kind="proper_score_loss"
    ):
        raise PredictiveContractError("loss lacks authenticated proper-score construction")
    if covariance.shape != (estimate.shape[0], estimate.shape[0]):
        raise PredictiveContractError("covariance and estimate dimensions disagree")
    if estimate.shape[0] != 2 * loss.horizon_count:
        raise PredictiveContractError("estimate dimension disagrees with proper-score loss")
    if type(bisection_iterations) is not int or bisection_iterations < 32:
        raise PredictiveContractError("bisection_iterations must be an integer at least 32")
    if type(kkt_tolerance) is not float or not math.isfinite(kkt_tolerance):
        raise PredictiveContractError("kkt_tolerance must be a finite float")
    if kkt_tolerance <= 0.0:
        raise PredictiveContractError("kkt_tolerance must be positive")
    alpha_tensor = _require_probability(alpha, "alpha")
    if not _scale_aware_equal(covariance, tf.transpose(covariance)):
        raise PredictiveContractError("covariance must be symmetric")
    covariance_eigenvalues = tf.linalg.eigvalsh(covariance)
    positive_definite = bool(tf.reduce_all(covariance_eigenvalues > 0.0))
    radius_squared = tfp.distributions.Chi2(
        df=tf.cast(estimate.shape[0], tf.float64)
    ).quantile(1.0 - alpha_tensor)
    if not positive_definite:
        nan = tf.constant(float("nan"), tf.float64)
        result = QuadraticLossBounds(
            estimate=estimate,
            covariance=covariance,
            confidence_radius_squared=radius_squared,
            point_loss=nan,
            lower_bound=nan,
            upper_bound=nan,
            lower_optimizer=tf.fill(tf.shape(estimate), nan),
            upper_optimizer=tf.fill(tf.shape(estimate), nan),
            lower_kkt_residual=nan,
            upper_kkt_residual=nan,
            covariance_eigenvalues=covariance_eigenvalues,
            inference_admissible=False,
            status=tf.constant(_INVALID),
        )
        return _seal_result(
            result,
            kind="quadratic_loss_bounds",
            provenance=(
                ("estimate", estimate),
                ("covariance", covariance),
                ("loss_signature", loss.construction_signature),
                ("alpha", alpha_tensor),
                ("bisection_iterations", bisection_iterations),
                ("kkt_tolerance", kkt_tolerance),
                ("jit_compile", jit_compile),
                ("invalid_reason", "covariance_not_positive_definite"),
            ),
        )
    kernel = _quadratic_loss_bounds_xla if jit_compile else _quadratic_loss_bounds_eager
    (
        point_loss,
        lower_bound,
        upper_bound,
        lower_optimizer,
        upper_optimizer,
        lower_kkt,
        upper_kkt,
        covariance_eigenvalues,
    ) = kernel(
        estimate,
        covariance,
        loss.loss_matrix,
        radius_squared,
        bisection_iterations,
    )
    numerical_scale = tf.maximum(
        tf.constant(1.0, tf.float64),
        tf.maximum(tf.abs(lower_bound), tf.abs(upper_bound)),
    )
    admissible = bool(
        tf.reduce_all(
            tf.stack(
                (
                    tf.math.is_finite(point_loss),
                    tf.math.is_finite(lower_bound),
                    tf.math.is_finite(upper_bound),
                    lower_bound >= 0.0,
                    lower_bound <= point_loss,
                    point_loss <= upper_bound,
                    lower_kkt <= kkt_tolerance * numerical_scale,
                    upper_kkt <= kkt_tolerance * numerical_scale,
                )
            )
        )
    )
    result = QuadraticLossBounds(
        estimate=estimate,
        covariance=covariance,
        confidence_radius_squared=radius_squared,
        point_loss=point_loss,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        lower_optimizer=lower_optimizer,
        upper_optimizer=upper_optimizer,
        lower_kkt_residual=lower_kkt,
        upper_kkt_residual=upper_kkt,
        covariance_eigenvalues=covariance_eigenvalues,
        inference_admissible=admissible,
        status=tf.constant(_VALID if admissible else _INVALID),
    )
    return _seal_result(
        result,
        kind="quadratic_loss_bounds",
        provenance=(
            ("estimate", estimate),
            ("covariance", covariance),
            ("loss_signature", loss.construction_signature),
            ("loss_fingerprint", _result_fingerprint("proper_score_loss", loss)),
            ("alpha", alpha_tensor),
            ("bisection_iterations", bisection_iterations),
            ("kkt_tolerance", kkt_tolerance),
            ("jit_compile", jit_compile),
        ),
    )


def batched_quadratic_loss_confidence_bounds(
    estimate: tf.Tensor,
    covariance: tf.Tensor,
    loss_matrices: tf.Tensor,
    *,
    alpha: tf.Tensor | float = 0.05,
    bisection_iterations: int = 96,
    kkt_tolerance: float = 1.0e-8,
    jit_compile: bool = True,
) -> BatchedQuadraticLossBounds:
    """Vectorize exact loss extrema over experiment replications and losses."""

    estimate = _require_tensor(estimate, "estimate", rank=2)
    covariance = _require_tensor(covariance, "covariance", rank=3)
    loss_matrices = _require_tensor(loss_matrices, "loss_matrices", rank=3)
    _require_bool(jit_compile, "jit_compile")
    batch_size, dimension = estimate.shape
    if covariance.shape != (batch_size, dimension, dimension):
        raise PredictiveContractError("batched covariance and estimate dimensions disagree")
    covariance_scale = tf.maximum(
        tf.constant(1.0, tf.float64), tf.reduce_max(tf.abs(covariance))
    )
    if not bool(
        tf.reduce_all(
            tf.abs(covariance - tf.transpose(covariance, [0, 2, 1]))
            <= tf.constant(512.0 * 2.220446049250313e-16, tf.float64)
            * covariance_scale
        )
    ):
        raise PredictiveContractError("batched covariance must be symmetric")
    if loss_matrices.shape[1:] != (dimension, dimension) or loss_matrices.shape[0] < 1:
        raise PredictiveContractError("loss matrices must have shape [loss, feature, feature]")
    if not bool(
        tf.reduce_all(
            tf.abs(loss_matrices - tf.transpose(loss_matrices, [0, 2, 1]))
            <= tf.constant(512.0 * 2.220446049250313e-16, tf.float64)
        )
    ):
        raise PredictiveContractError("loss matrices must be symmetric")
    if bool(tf.reduce_any(tf.linalg.eigvalsh(loss_matrices) < -1.0e-14)):
        raise PredictiveContractError("loss matrices must be positive semidefinite")
    if type(bisection_iterations) is not int or bisection_iterations < 32:
        raise PredictiveContractError("bisection_iterations must be an integer at least 32")
    if type(kkt_tolerance) is not float or not math.isfinite(kkt_tolerance) or kkt_tolerance <= 0.0:
        raise PredictiveContractError("kkt_tolerance must be a finite positive float")
    alpha_tensor = _require_probability(alpha, "alpha")
    covariance_eigenvalues = tf.linalg.eigvalsh(covariance)
    covariance_valid = tf.reduce_all(covariance_eigenvalues > 0.0, axis=1)
    safe_covariance = tf.where(
        covariance_valid[:, tf.newaxis, tf.newaxis],
        covariance,
        tf.eye(dimension, batch_shape=[batch_size], dtype=tf.float64),
    )
    radius_squared = tfp.distributions.Chi2(df=tf.cast(dimension, tf.float64)).quantile(
        1.0 - alpha_tensor
    )
    kernel = (
        _batched_quadratic_loss_bounds_xla
        if jit_compile
        else _batched_quadratic_loss_bounds_eager
    )
    point, lower, upper, lower_kkt, upper_kkt, safe_eigenvalues = kernel(
        estimate,
        safe_covariance,
        loss_matrices,
        radius_squared,
        bisection_iterations,
    )
    numerical_scale = tf.maximum(
        tf.constant(1.0, tf.float64), tf.maximum(tf.abs(lower), tf.abs(upper))
    )
    loss_valid = tf.reduce_all(
        tf.stack(
            (
                tf.math.is_finite(point),
                tf.math.is_finite(lower),
                tf.math.is_finite(upper),
                lower >= -1.0e-12,
                lower <= point + 1.0e-12,
                point <= upper + 1.0e-12,
                lower_kkt <= kkt_tolerance * numerical_scale,
                upper_kkt <= kkt_tolerance * numerical_scale,
            ),
            axis=0,
        ),
        axis=0,
    )
    admissible = tf.logical_and(covariance_valid, tf.reduce_all(loss_valid, axis=1))
    status = tf.where(
        admissible,
        tf.fill([batch_size], tf.constant(_VALID)),
        tf.fill([batch_size], tf.constant(_INVALID)),
    )
    result = BatchedQuadraticLossBounds(
        estimate=estimate,
        covariance=covariance,
        loss_matrices=loss_matrices,
        confidence_radius_squared=radius_squared,
        point_loss=point,
        lower_bound=lower,
        upper_bound=upper,
        lower_kkt_residual=lower_kkt,
        upper_kkt_residual=upper_kkt,
        covariance_eigenvalues=tf.where(
            covariance_valid[:, tf.newaxis], covariance_eigenvalues, safe_eigenvalues
        ),
        inference_admissible=admissible,
        status=status,
    )
    return _seal_result(
        result,
        kind="batched_quadratic_loss_bounds",
        provenance=(
            ("estimate", estimate),
            ("covariance", covariance),
            ("loss_matrices", loss_matrices),
            ("alpha", alpha_tensor),
            ("bisection_iterations", bisection_iterations),
            ("kkt_tolerance", kkt_tolerance),
            ("jit_compile", jit_compile),
        ),
    )


def _validate_split_alpha_allocation(
    *,
    average_alpha: tf.Tensor | float,
    horizon_alpha: tf.Tensor | float,
    horizon_count: int,
    familywise_alpha: tf.Tensor | float,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    average = _require_probability(average_alpha, "average_alpha")
    horizon = _require_probability(horizon_alpha, "horizon_alpha")
    familywise = _require_probability(familywise_alpha, "familywise_alpha")
    allocated = average + tf.cast(horizon_count, tf.float64) * horizon
    if bool(allocated > familywise):
        raise PredictiveContractError(
            "split-region alpha allocation exceeds familywise_alpha"
        )
    return average, horizon, allocated


def split_quadratic_loss_confidence_bounds(
    estimate: tf.Tensor,
    covariance: tf.Tensor,
    average_loss: ProperScoreLoss,
    *,
    average_alpha: tf.Tensor | float = 0.025,
    horizon_alpha: tf.Tensor | float = 0.0025,
    familywise_alpha: tf.Tensor | float = 0.05,
    bisection_iterations: int = 96,
    kkt_tolerance: float = 1.0e-8,
    jit_compile: bool = True,
) -> SplitQuadraticLossBounds:
    """Bound average loss in 2H dimensions and each horizon in 2 dimensions."""

    estimate = _require_tensor(estimate, "estimate", rank=1)
    covariance = _require_tensor(covariance, "covariance", rank=2)
    if type(average_loss) is not ProperScoreLoss or not _authenticated_result(
        average_loss, kind="proper_score_loss"
    ):
        raise PredictiveContractError(
            "average_loss lacks authenticated proper-score construction"
        )
    horizon_count = average_loss.horizon_count
    if estimate.shape != (2 * horizon_count,) or covariance.shape != (
        2 * horizon_count,
        2 * horizon_count,
    ):
        raise PredictiveContractError(
            "estimate and covariance dimensions disagree with average_loss"
        )
    average_alpha_tensor, horizon_alpha_tensor, allocated = (
        _validate_split_alpha_allocation(
            average_alpha=average_alpha,
            horizon_alpha=horizon_alpha,
            horizon_count=horizon_count,
            familywise_alpha=familywise_alpha,
        )
    )
    average = quadratic_loss_confidence_bounds(
        estimate,
        covariance,
        average_loss,
        alpha=average_alpha_tensor,
        bisection_iterations=bisection_iterations,
        kkt_tolerance=kkt_tolerance,
        jit_compile=jit_compile,
    )
    local_loss = proper_score_loss(tf.ones([1], tf.float64))
    horizons: list[QuadraticLossBounds] = []
    for horizon in range(horizon_count):
        indices = tf.constant([horizon, horizon_count + horizon], tf.int32)
        local_estimate = tf.gather(estimate, indices)
        local_covariance = tf.gather(
            tf.gather(covariance, indices, axis=0), indices, axis=1
        )
        horizons.append(
            quadratic_loss_confidence_bounds(
                local_estimate,
                local_covariance,
                local_loss,
                alpha=horizon_alpha_tensor,
                bisection_iterations=bisection_iterations,
                kkt_tolerance=kkt_tolerance,
                jit_compile=jit_compile,
            )
        )
    admissible = average.inference_admissible and all(
        bounds.inference_admissible for bounds in horizons
    )
    result = SplitQuadraticLossBounds(
        estimate=estimate,
        covariance=covariance,
        average_point_loss=average.point_loss,
        average_lower_bound=average.lower_bound,
        average_upper_bound=average.upper_bound,
        horizon_point_losses=tf.stack([bounds.point_loss for bounds in horizons]),
        horizon_lower_bounds=tf.stack([bounds.lower_bound for bounds in horizons]),
        horizon_upper_bounds=tf.stack([bounds.upper_bound for bounds in horizons]),
        average_confidence_radius_squared=average.confidence_radius_squared,
        horizon_confidence_radii_squared=tf.stack(
            [bounds.confidence_radius_squared for bounds in horizons]
        ),
        average_lower_kkt_residual=average.lower_kkt_residual,
        average_upper_kkt_residual=average.upper_kkt_residual,
        horizon_lower_kkt_residuals=tf.stack(
            [bounds.lower_kkt_residual for bounds in horizons]
        ),
        horizon_upper_kkt_residuals=tf.stack(
            [bounds.upper_kkt_residual for bounds in horizons]
        ),
        average_alpha=average_alpha_tensor,
        horizon_alphas=tf.fill([horizon_count], horizon_alpha_tensor),
        allocated_familywise_alpha=allocated,
        inference_admissible=admissible,
        status=tf.constant(_VALID if admissible else _INVALID),
    )
    return _seal_result(
        result,
        kind="split_quadratic_loss_bounds",
        provenance=(
            ("estimate", estimate),
            ("covariance", covariance),
            ("average_loss_signature", average_loss.construction_signature),
            ("average_alpha", average_alpha_tensor),
            ("horizon_alpha", horizon_alpha_tensor),
            ("familywise_alpha", _require_probability(familywise_alpha, "familywise_alpha")),
            ("bisection_iterations", bisection_iterations),
            ("kkt_tolerance", kkt_tolerance),
            ("jit_compile", jit_compile),
        ),
    )


def batched_split_quadratic_loss_confidence_bounds(
    estimate: tf.Tensor,
    covariance: tf.Tensor,
    average_loss: ProperScoreLoss,
    *,
    average_alpha: tf.Tensor | float = 0.025,
    horizon_alpha: tf.Tensor | float = 0.0025,
    familywise_alpha: tf.Tensor | float = 0.05,
    bisection_iterations: int = 96,
    kkt_tolerance: float = 1.0e-8,
    jit_compile: bool = True,
) -> BatchedSplitQuadraticLossBounds:
    """Batched split-region bounds with authenticated horizon projections."""

    estimate = _require_tensor(estimate, "estimate", rank=2)
    covariance = _require_tensor(covariance, "covariance", rank=3)
    if type(average_loss) is not ProperScoreLoss or not _authenticated_result(
        average_loss, kind="proper_score_loss"
    ):
        raise PredictiveContractError(
            "average_loss lacks authenticated proper-score construction"
        )
    batch_size, dimension = estimate.shape
    horizon_count = average_loss.horizon_count
    if dimension != 2 * horizon_count or covariance.shape != (
        batch_size,
        dimension,
        dimension,
    ):
        raise PredictiveContractError(
            "batched estimate and covariance dimensions disagree with average_loss"
        )
    average_alpha_tensor, horizon_alpha_tensor, allocated = (
        _validate_split_alpha_allocation(
            average_alpha=average_alpha,
            horizon_alpha=horizon_alpha,
            horizon_count=horizon_count,
            familywise_alpha=familywise_alpha,
        )
    )
    average = batched_quadratic_loss_confidence_bounds(
        estimate,
        covariance,
        average_loss.loss_matrix[tf.newaxis, :, :],
        alpha=average_alpha_tensor,
        bisection_iterations=bisection_iterations,
        kkt_tolerance=kkt_tolerance,
        jit_compile=jit_compile,
    )
    horizon_indices = tf.stack(
        (
            tf.range(horizon_count, dtype=tf.int32),
            tf.range(horizon_count, dtype=tf.int32) + horizon_count,
        ),
        axis=1,
    )
    local_estimates = tf.gather(estimate, horizon_indices, axis=1)
    selectors = tf.one_hot(horizon_indices, dimension, dtype=tf.float64)
    local_covariances = tf.einsum(
        "hki,bij,hlj->bhkl", selectors, covariance, selectors
    )
    flat_estimates = tf.reshape(local_estimates, [batch_size * horizon_count, 2])
    flat_covariances = tf.reshape(
        local_covariances, [batch_size * horizon_count, 2, 2]
    )
    local_loss_matrix = tf.linalg.diag(tf.constant([0.5, 0.25], tf.float64))
    horizons = batched_quadratic_loss_confidence_bounds(
        flat_estimates,
        flat_covariances,
        local_loss_matrix[tf.newaxis, :, :],
        alpha=horizon_alpha_tensor,
        bisection_iterations=bisection_iterations,
        kkt_tolerance=kkt_tolerance,
        jit_compile=jit_compile,
    )
    horizon_admissible = tf.reshape(
        horizons.inference_admissible, [batch_size, horizon_count]
    )
    admissible = tf.logical_and(
        average.inference_admissible,
        tf.reduce_all(horizon_admissible, axis=1),
    )

    def horizon_values(value: tf.Tensor) -> tf.Tensor:
        return tf.reshape(value[:, 0], [batch_size, horizon_count])

    result = BatchedSplitQuadraticLossBounds(
        estimate=estimate,
        covariance=covariance,
        average_point_loss=average.point_loss[:, 0],
        average_lower_bound=average.lower_bound[:, 0],
        average_upper_bound=average.upper_bound[:, 0],
        horizon_point_losses=horizon_values(horizons.point_loss),
        horizon_lower_bounds=horizon_values(horizons.lower_bound),
        horizon_upper_bounds=horizon_values(horizons.upper_bound),
        average_confidence_radius_squared=average.confidence_radius_squared,
        horizon_confidence_radius_squared=horizons.confidence_radius_squared,
        average_lower_kkt_residual=average.lower_kkt_residual[:, 0],
        average_upper_kkt_residual=average.upper_kkt_residual[:, 0],
        horizon_lower_kkt_residuals=horizon_values(horizons.lower_kkt_residual),
        horizon_upper_kkt_residuals=horizon_values(horizons.upper_kkt_residual),
        average_alpha=average_alpha_tensor,
        horizon_alpha=horizon_alpha_tensor,
        allocated_familywise_alpha=allocated,
        inference_admissible=admissible,
        status=tf.where(
            admissible,
            tf.fill([batch_size], tf.constant(_VALID)),
            tf.fill([batch_size], tf.constant(_INVALID)),
        ),
    )
    return _seal_result(
        result,
        kind="batched_split_quadratic_loss_bounds",
        provenance=(
            ("estimate", estimate),
            ("covariance", covariance),
            ("average_loss_signature", average_loss.construction_signature),
            ("average_alpha", average_alpha_tensor),
            ("horizon_alpha", horizon_alpha_tensor),
            ("familywise_alpha", _require_probability(familywise_alpha, "familywise_alpha")),
            ("bisection_iterations", bisection_iterations),
            ("kkt_tolerance", kkt_tolerance),
            ("jit_compile", jit_compile),
        ),
    )


def classify_split_proper_score_equivalence(
    bounds: SplitQuadraticLossBounds,
    *,
    acceptable_average_loss: tf.Tensor,
    acceptable_horizon_loss: tf.Tensor,
    mechanics_only: bool = False,
) -> DualProperScoreDecision:
    """Classify one average region and its simultaneous horizon marginals."""

    average_tolerance = _require_tensor(
        acceptable_average_loss, "acceptable_average_loss", rank=0
    )
    horizon_tolerance = _require_tensor(
        acceptable_horizon_loss, "acceptable_horizon_loss", rank=0
    )
    _require_bool(mechanics_only, "mechanics_only")
    try:
        tf.debugging.assert_positive(average_tolerance)
        tf.debugging.assert_positive(horizon_tolerance)
    except tf.errors.InvalidArgumentError as exc:
        raise PredictiveContractError("dual acceptable losses must be positive") from exc
    hard_vetoes: list[str] = []
    if type(bounds) is not SplitQuadraticLossBounds or not _authenticated_result(
        bounds, kind="split_quadratic_loss_bounds"
    ):
        hard_vetoes.append("SPLIT_LOSS_BOUNDS_UNAUTHENTICATED")
    elif not bounds.inference_admissible or not _has_exact_status(bounds.status, _VALID):
        hard_vetoes.append("SPLIT_LOSS_BOUNDS_NOT_ADMISSIBLE")
    if mechanics_only:
        hard_vetoes.append("MECHANICS_ONLY_CANNOT_PASS")
    if hard_vetoes:
        nan = tf.constant(float("nan"), tf.float64)
        horizon_count = (
            int(bounds.horizon_lower_bounds.shape[0])
            if type(bounds) is SplitQuadraticLossBounds
            else 0
        )
        return DualProperScoreDecision(
            status="INVALID_HARD_VETO",
            average_loss_lower_bound=(
                bounds.average_lower_bound
                if type(bounds) is SplitQuadraticLossBounds
                else nan
            ),
            average_loss_upper_bound=(
                bounds.average_upper_bound
                if type(bounds) is SplitQuadraticLossBounds
                else nan
            ),
            horizon_loss_lower_bounds=(
                bounds.horizon_lower_bounds
                if type(bounds) is SplitQuadraticLossBounds
                else tf.fill([horizon_count], nan)
            ),
            horizon_loss_upper_bounds=(
                bounds.horizon_upper_bounds
                if type(bounds) is SplitQuadraticLossBounds
                else tf.fill([horizon_count], nan)
            ),
            acceptable_average_loss=average_tolerance,
            acceptable_horizon_loss=horizon_tolerance,
            hard_veto_codes=tuple(hard_vetoes),
            explanatory_diagnostics={},
        )
    if bool(
        tf.logical_and(
            bounds.average_upper_bound < average_tolerance,
            tf.reduce_all(bounds.horizon_upper_bounds < horizon_tolerance),
        )
    ):
        status: DecisionStatus = "PASS"
    elif bool(
        tf.logical_or(
            bounds.average_lower_bound > average_tolerance,
            tf.reduce_any(bounds.horizon_lower_bounds > horizon_tolerance),
        )
    ):
        status = "MATERIAL_DIFFERENCE"
    else:
        status = "INCONCLUSIVE_UNDERPOWERED"
    return DualProperScoreDecision(
        status=status,
        average_loss_lower_bound=bounds.average_lower_bound,
        average_loss_upper_bound=bounds.average_upper_bound,
        horizon_loss_lower_bounds=bounds.horizon_lower_bounds,
        horizon_loss_upper_bounds=bounds.horizon_upper_bounds,
        acceptable_average_loss=average_tolerance,
        acceptable_horizon_loss=horizon_tolerance,
        hard_veto_codes=(),
        explanatory_diagnostics={
            "average_point_loss": bounds.average_point_loss,
            "horizon_point_losses": bounds.horizon_point_losses,
            "average_confidence_radius_squared": (
                bounds.average_confidence_radius_squared
            ),
            "horizon_confidence_radii_squared": (
                bounds.horizon_confidence_radii_squared
            ),
            "allocated_familywise_alpha": bounds.allocated_familywise_alpha,
        },
    )


def classify_proper_score_equivalence(
    bounds: QuadraticLossBounds,
    *,
    acceptable_loss: tf.Tensor,
    mechanics_only: bool = False,
) -> ProperScoreDecision:
    """Classify a proper-score confidence region without an MMD promotion gate."""

    tolerance = _require_tensor(acceptable_loss, "acceptable_loss", rank=0)
    _require_bool(mechanics_only, "mechanics_only")
    try:
        tf.debugging.assert_positive(tolerance, "acceptable_loss must be positive")
    except tf.errors.InvalidArgumentError as exc:
        raise PredictiveContractError("acceptable_loss must be positive") from exc
    hard_vetoes: list[str] = []
    if type(bounds) is not QuadraticLossBounds or not _authenticated_result(
        bounds, kind="quadratic_loss_bounds"
    ):
        hard_vetoes.append("LOSS_BOUNDS_UNAUTHENTICATED")
    elif not bounds.inference_admissible or not _has_exact_status(bounds.status, _VALID):
        hard_vetoes.append("LOSS_BOUNDS_NOT_ADMISSIBLE")
    if mechanics_only:
        hard_vetoes.append("MECHANICS_ONLY_CANNOT_PASS")
    if hard_vetoes:
        nan = tf.constant(float("nan"), tf.float64)
        lower = bounds.lower_bound if type(bounds) is QuadraticLossBounds else nan
        upper = bounds.upper_bound if type(bounds) is QuadraticLossBounds else nan
        return ProperScoreDecision(
            status="INVALID_HARD_VETO",
            loss_lower_bound=lower,
            loss_upper_bound=upper,
            acceptable_loss=tolerance,
            hard_veto_codes=tuple(hard_vetoes),
            explanatory_diagnostics={},
        )
    if bool(bounds.upper_bound < tolerance):
        status: DecisionStatus = "PASS"
    elif bool(bounds.lower_bound > tolerance):
        status = "MATERIAL_DIFFERENCE"
    else:
        status = "INCONCLUSIVE_UNDERPOWERED"
    return ProperScoreDecision(
        status=status,
        loss_lower_bound=bounds.lower_bound,
        loss_upper_bound=bounds.upper_bound,
        acceptable_loss=tolerance,
        hard_veto_codes=(),
        explanatory_diagnostics={
            "point_loss": bounds.point_loss,
            "confidence_radius_squared": bounds.confidence_radius_squared,
            "lower_kkt_residual": bounds.lower_kkt_residual,
            "upper_kkt_residual": bounds.upper_kkt_residual,
        },
    )


def classify_dual_proper_score_equivalence(
    average_bounds: QuadraticLossBounds,
    horizon_bounds: tuple[QuadraticLossBounds, ...],
    *,
    acceptable_average_loss: tf.Tensor,
    acceptable_horizon_loss: tf.Tensor,
    mechanics_only: bool = False,
) -> DualProperScoreDecision:
    """Combine average and horizonwise losses over one joint confidence region."""

    average_tolerance = _require_tensor(
        acceptable_average_loss, "acceptable_average_loss", rank=0
    )
    horizon_tolerance = _require_tensor(
        acceptable_horizon_loss, "acceptable_horizon_loss", rank=0
    )
    _require_bool(mechanics_only, "mechanics_only")
    try:
        tf.debugging.assert_positive(
            average_tolerance, "acceptable_average_loss must be positive"
        )
        tf.debugging.assert_positive(
            horizon_tolerance, "acceptable_horizon_loss must be positive"
        )
    except tf.errors.InvalidArgumentError as exc:
        raise PredictiveContractError("dual acceptable losses must be positive") from exc
    if type(horizon_bounds) is not tuple or not horizon_bounds:
        raise PredictiveContractError("horizon_bounds must be a nonempty tuple")

    all_bounds = (average_bounds, *horizon_bounds)
    hard_vetoes: list[str] = []
    if any(
        type(bounds) is not QuadraticLossBounds
        or not _authenticated_result(bounds, kind="quadratic_loss_bounds")
        for bounds in all_bounds
    ):
        hard_vetoes.append("LOSS_BOUNDS_UNAUTHENTICATED")
    elif any(
        not bounds.inference_admissible or not _has_exact_status(bounds.status, _VALID)
        for bounds in all_bounds
    ):
        hard_vetoes.append("LOSS_BOUNDS_NOT_ADMISSIBLE")
    else:
        expected_dimension = int(average_bounds.estimate.shape[0])
        if len(horizon_bounds) * 2 != expected_dimension:
            hard_vetoes.append("HORIZON_BOUND_COUNT_MISMATCH")
        if any(
            bounds.estimate.shape != average_bounds.estimate.shape
            or bounds.covariance.shape != average_bounds.covariance.shape
            or not _scale_aware_equal(bounds.estimate, average_bounds.estimate)
            or not _scale_aware_equal(bounds.covariance, average_bounds.covariance)
            or not _scale_aware_equal(
                bounds.confidence_radius_squared,
                average_bounds.confidence_radius_squared,
            )
            for bounds in horizon_bounds
        ):
            hard_vetoes.append("LOSS_BOUNDS_NOT_ONE_JOINT_REGION")
    if mechanics_only:
        hard_vetoes.append("MECHANICS_ONLY_CANNOT_PASS")

    if type(average_bounds) is QuadraticLossBounds:
        average_lower = average_bounds.lower_bound
        average_upper = average_bounds.upper_bound
    else:
        average_lower = tf.constant(float("nan"), tf.float64)
        average_upper = tf.constant(float("nan"), tf.float64)
    if all(type(bounds) is QuadraticLossBounds for bounds in horizon_bounds):
        horizon_lowers = tf.stack([bounds.lower_bound for bounds in horizon_bounds])
        horizon_uppers = tf.stack([bounds.upper_bound for bounds in horizon_bounds])
    else:
        horizon_lowers = tf.fill([len(horizon_bounds)], tf.constant(float("nan"), tf.float64))
        horizon_uppers = tf.fill([len(horizon_bounds)], tf.constant(float("nan"), tf.float64))

    if hard_vetoes:
        status: DecisionStatus = "INVALID_HARD_VETO"
    elif bool(
        tf.logical_and(
            average_upper < average_tolerance,
            tf.reduce_all(horizon_uppers < horizon_tolerance),
        )
    ):
        status = "PASS"
    elif bool(
        tf.logical_or(
            average_lower > average_tolerance,
            tf.reduce_any(horizon_lowers > horizon_tolerance),
        )
    ):
        status = "MATERIAL_DIFFERENCE"
    else:
        status = "INCONCLUSIVE_UNDERPOWERED"

    diagnostics: dict[str, tf.Tensor] = {}
    if not hard_vetoes:
        diagnostics = {
            "average_point_loss": average_bounds.point_loss,
            "horizon_point_losses": tf.stack(
                [bounds.point_loss for bounds in horizon_bounds]
            ),
            "confidence_radius_squared": average_bounds.confidence_radius_squared,
        }
    return DualProperScoreDecision(
        status=status,
        average_loss_lower_bound=average_lower,
        average_loss_upper_bound=average_upper,
        horizon_loss_lower_bounds=horizon_lowers,
        horizon_loss_upper_bounds=horizon_uppers,
        acceptable_average_loss=average_tolerance,
        acceptable_horizon_loss=horizon_tolerance,
        hard_veto_codes=tuple(hard_vetoes),
        explanatory_diagnostics=diagnostics,
    )


__all__ = [
    "BatchedBartlettLongRunCovarianceResult",
    "BatchedQuadraticLossBounds",
    "BatchedSplitQuadraticLossBounds",
    "BartlettLongRunCovarianceResult",
    "ConditionalMeanLogVarianceInfluenceResult",
    "CrossChainLinearMMD",
    "DecisionStatus",
    "DualProperScoreDecision",
    "HierarchicalBootstrapIndices",
    "LongRunCovarianceResult",
    "MeanLogVarianceInfluenceResult",
    "PairwiseDistanceScaleResult",
    "MMDInterval",
    "MMDStatistics",
    "PredictiveContractError",
    "PredictiveDecision",
    "PredictiveStatisticsConfig",
    "PredictiveSummary",
    "ProperScoreDecision",
    "ProperScoreLoss",
    "QuadraticLossBounds",
    "SimultaneousIntervals",
    "SplitQuadraticLossBounds",
    "adapt_ssl_lstm_observations",
    "batched_chain_bartlett_long_run_covariance",
    "batched_quadratic_loss_confidence_bounds",
    "batched_split_quadratic_loss_confidence_bounds",
    "chain_batch_means",
    "chain_batch_long_run_covariance",
    "chain_bartlett_long_run_covariance",
    "classify_dual_proper_score_equivalence",
    "classify_proper_score_equivalence",
    "classify_split_proper_score_equivalence",
    "classify_predictive_evidence",
    "cross_chain_linear_mmd",
    "cross_chain_mmd_upper_interval",
    "conditional_mean_log_variance_influence",
    "fixed_rbf_mmd",
    "growing_hac_bandwidth",
    "hierarchical_resample_indices",
    "horizon_proper_score_loss",
    "mean_log_variance_influence",
    "pooled_pairwise_distance_scale",
    "proper_score_loss",
    "quadratic_loss_confidence_bounds",
    "split_quadratic_loss_confidence_bounds",
    "simultaneous_feature_intervals",
    "standardize_forecast_paths",
    "summarize_forecast_paths",
]
