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


__all__ = [
    "CrossChainLinearMMD",
    "DecisionStatus",
    "HierarchicalBootstrapIndices",
    "MMDInterval",
    "MMDStatistics",
    "PredictiveContractError",
    "PredictiveDecision",
    "PredictiveStatisticsConfig",
    "PredictiveSummary",
    "SimultaneousIntervals",
    "adapt_ssl_lstm_observations",
    "chain_batch_means",
    "classify_predictive_evidence",
    "cross_chain_linear_mmd",
    "cross_chain_mmd_upper_interval",
    "fixed_rbf_mmd",
    "hierarchical_resample_indices",
    "simultaneous_feature_intervals",
    "standardize_forecast_paths",
    "summarize_forecast_paths",
]
