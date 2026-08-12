"""Rank-normalized multi-chain diagnostics for preserved HMC transitions.

The public functions in this module consume chain-major samples
``[chain, draw, parameter]``.  They implement the Phase 29 operational screens
using TensorFlow/TFP and return host-side, JSON-safe reports only after the
TensorFlow calculations finish.  Passing these finite-sample screens is not a
proof of stationarity or posterior convergence.

References
----------
Vehtari et al. (2021), *Rank-normalization, folding, and localization: An
improved R-hat for assessing convergence of MCMC*, Bayesian Analysis 16(2).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Phase29DiagnosticThresholds:
    """Prospectively fixed thresholds for the Phase 29 diagnostic pilot."""

    rhat_max: float = 1.05
    bulk_ess_min: float = 100.0
    tail_ess_min: float = 100.0
    mcse_sd_ratio_max: float = 0.10
    acceptance_min: float = 0.20
    acceptance_max: float = 0.95
    ebfmi_min: float = 0.30
    initialization_memory_max: float = 3.70
    epoch_drift_z_max: float = 3.50
    epoch_sd_ratio_min: float = 0.80
    epoch_sd_ratio_max: float = 1.25
    delta_h_abs_max: float = 1000.0

    def __post_init__(self) -> None:
        positive = (
            "rhat_max",
            "bulk_ess_min",
            "tail_ess_min",
            "mcse_sd_ratio_max",
            "ebfmi_min",
            "initialization_memory_max",
            "epoch_drift_z_max",
            "epoch_sd_ratio_min",
            "epoch_sd_ratio_max",
            "delta_h_abs_max",
        )
        for name in positive:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
            object.__setattr__(self, name, value)
        for name in ("acceptance_min", "acceptance_max"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1]")
            object.__setattr__(self, name, value)
        if self.rhat_max <= 1.0:
            raise ValueError("rhat_max must be greater than one")
        if self.acceptance_min >= self.acceptance_max:
            raise ValueError("acceptance_min must be less than acceptance_max")
        if self.epoch_sd_ratio_min >= self.epoch_sd_ratio_max:
            raise ValueError("epoch SD ratio bounds are reversed")

    def payload(self) -> dict[str, float]:
        return {
            name: float(getattr(self, name))
            for name in self.__dataclass_fields__
        }


def _sample_tensor(samples: Any, *, label: str = "samples") -> Any:
    import tensorflow as tf

    tensor = tf.cast(tf.convert_to_tensor(samples), tf.float64)
    if tensor.shape.rank != 3:
        raise ValueError(f"{label} must have shape [chain, draw, parameter]")
    if any(dim is None for dim in tensor.shape):
        raise ValueError(f"{label} must have fully static shape")
    chains, draws, parameters = (int(dim) for dim in tensor.shape)
    if chains < 2:
        raise ValueError(f"{label} requires at least two chains")
    if draws < 4 or draws % 2:
        raise ValueError(f"{label} requires an even draw count of at least four")
    if parameters < 1:
        raise ValueError(f"{label} requires at least one parameter")
    return tensor


def _split_sample_major(samples: Any) -> Any:
    """Split each chain in half and return ``[draw, split_chain, parameter]``."""

    import tensorflow as tf

    draws = int(samples.shape[1])
    half = draws // 2
    split_chain_major = tf.concat(
        (samples[:, :half, :], samples[:, half:, :]), axis=0
    )
    return tf.transpose(split_chain_major, perm=(1, 0, 2))


def _average_rank_one(values: Any) -> Any:
    """Return one-based average ranks, including exact tie handling."""

    import tensorflow as tf

    values = tf.cast(values, tf.float64)
    count = tf.shape(values, out_type=tf.int32)[0]
    order = tf.argsort(values, axis=0, stable=True)
    ordered = tf.gather(values, order)
    group_start = tf.concat(
        (
            tf.constant([True]),
            tf.not_equal(ordered[1:], ordered[:-1]),
        ),
        axis=0,
    )
    group = tf.cumsum(tf.cast(group_start, tf.int32)) - 1
    group_count = tf.reduce_max(group) + 1
    positions = tf.cast(tf.range(1, count + 1), tf.float64)
    rank_sum = tf.math.unsorted_segment_sum(positions, group, group_count)
    rank_count = tf.math.unsorted_segment_sum(
        tf.ones_like(positions), group, group_count
    )
    ordered_rank = tf.gather(rank_sum / rank_count, group)
    inverse = tf.argsort(order, axis=0, stable=True)
    return tf.gather(ordered_rank, inverse)


def _rank_normalize(sample_major: Any) -> Any:
    """Apply pooled rank normalization independently to each parameter."""

    import tensorflow as tf
    import tensorflow_probability as tfp

    shape = tf.shape(sample_major, out_type=tf.int32)
    flat = tf.reshape(sample_major, (shape[0] * shape[1], shape[2]))
    ranks_by_parameter = tf.map_fn(
        _average_rank_one,
        tf.transpose(flat),
        fn_output_signature=tf.TensorSpec(shape=(None,), dtype=tf.float64),
        parallel_iterations=1,
    )
    ranks = tf.transpose(ranks_by_parameter)
    total = tf.cast(tf.shape(flat, out_type=tf.int32)[0], tf.float64)
    probability = (ranks - tf.constant(3.0 / 8.0, tf.float64)) / (
        total - tf.constant(1.0 / 4.0, tf.float64)
    )
    normal = tfp.distributions.Normal(
        loc=tf.constant(0.0, tf.float64),
        scale=tf.constant(1.0, tf.float64),
    )
    return tf.reshape(normal.quantile(probability), shape)


def _sample_sd(values: Any, *, axis: Any) -> Any:
    import tensorflow as tf

    values = tf.cast(values, tf.float64)
    mean = tf.reduce_mean(values, axis=axis, keepdims=True)
    squared = tf.reduce_sum(tf.square(values - mean), axis=axis)
    if isinstance(axis, int):
        count = tf.cast(tf.shape(values)[axis], tf.float64)
    else:
        count = tf.cast(
            tf.reduce_prod(tf.gather(tf.shape(values), list(axis))), tf.float64
        )
    return tf.sqrt(squared / (count - 1.0))


def _cross_chain_ess(sample_major: Any) -> Any:
    """Vehtari cross-chain ESS with a warning-free real FFT covariance.

    TFP 0.25 computes the same autocovariance through a complex FFT and then
    casts ``complex128`` to ``float64``.  The mathematical result is real, but
    that cast emits a lossy-conversion warning for every ESS call.  Using
    ``rfft``/``irfft`` preserves the real-valued contract directly.
    """

    import tensorflow as tf

    values = tf.cast(tf.convert_to_tensor(sample_major), tf.float64)
    if values.shape.rank != 3 or any(dim is None for dim in values.shape):
        raise ValueError("cross-chain ESS requires static [draw, chain, parameter]")
    draw_count, chain_count, _ = (int(dim) for dim in values.shape)
    rotated = tf.transpose(values, (1, 2, 0))
    centered = rotated - tf.reduce_mean(rotated, axis=-1, keepdims=True)
    fft_length = 1 << int(math.ceil(math.log2(2 * draw_count)))
    padded = tf.pad(centered, ((0, 0), (0, 0), (0, fft_length - draw_count)))
    spectrum = tf.signal.rfft(padded)
    autocov_rotated = tf.signal.irfft(
        spectrum * tf.math.conj(spectrum), fft_length=(fft_length,)
    )[..., :draw_count]
    denominators = tf.cast(
        tf.range(draw_count, 0, -1), tf.float64
    )
    autocov = tf.transpose(
        autocov_rotated / denominators[tf.newaxis, tf.newaxis, :],
        (2, 0, 1),
    )

    chain_means = tf.reduce_mean(values, axis=0)
    between_div_n = tf.math.reduce_variance(chain_means, axis=0) * (
        tf.cast(chain_count, tf.float64)
        / tf.cast(chain_count - 1, tf.float64)
    )
    biased_within = tf.reduce_mean(autocov[0], axis=0)
    variance_plus = biased_within + between_div_n
    mean_autocov = tf.reduce_mean(autocov, axis=1)
    autocorrelation = 1.0 - (
        biased_within[tf.newaxis, :] - mean_autocov
    ) / variance_plus[tf.newaxis, :]
    lag_weight = tf.cast(
        tf.range(draw_count, 0, -1), tf.float64
    ) / tf.cast(draw_count, tf.float64)
    weighted = autocorrelation * lag_weight[:, tf.newaxis]

    even_count = draw_count - draw_count % 2
    pair_shape = (even_count // 2, 2, int(values.shape[2]))
    pair_correlation = tf.reduce_sum(
        tf.reshape(autocorrelation[:even_count], pair_shape), axis=1
    )
    positive_mask = tf.maximum(
        1.0
        - tf.cumsum(tf.cast(pair_correlation < 0.0, tf.float64), axis=0),
        0.0,
    )
    weighted_pairs = tf.reduce_sum(
        tf.reshape(weighted[:even_count], pair_shape), axis=1
    ) * positive_mask
    return (
        tf.cast(chain_count * draw_count, tf.float64)
        / (-1.0 + 2.0 * tf.reduce_sum(weighted_pairs, axis=0))
    )


def _split_rhat_from_sample_major(sample_major: Any) -> Any:
    """Return the square-root between/within-chain variance diagnostic."""

    import tensorflow as tf

    values = tf.cast(sample_major, tf.float64)
    draws = tf.cast(tf.shape(values)[0], tf.float64)
    chain_means = tf.reduce_mean(values, axis=0)
    within_chain_variance = _sample_sd(values, axis=0) ** 2
    within = tf.reduce_mean(within_chain_variance, axis=0)
    between_over_draws = _sample_sd(chain_means, axis=0) ** 2
    variance_plus = ((draws - 1.0) / draws) * within + between_over_draws
    return tf.sqrt(variance_plus / within)


def rank_normalized_split_rhat(samples: Any) -> Mapping[str, Any]:
    """Compute bulk, folded, and maximum rank-normalized split R-hat."""

    import tensorflow as tf
    import tensorflow_probability as tfp

    tensor = _sample_tensor(samples)
    tf.debugging.assert_all_finite(tensor, "samples must be finite")
    split = _split_sample_major(tensor)
    bulk = _rank_normalize(split)
    pooled_median = tfp.stats.percentile(
        split, 50.0, axis=(0, 1), interpolation="midpoint"
    )
    folded = _rank_normalize(tf.abs(split - pooled_median))
    bulk_rhat = _split_rhat_from_sample_major(bulk)
    folded_rhat = _split_rhat_from_sample_major(folded)
    return {
        "bulk": bulk_rhat,
        "folded": folded_rhat,
        "maximum": tf.maximum(bulk_rhat, folded_rhat),
    }


def rank_normalized_bulk_tail_ess(samples: Any) -> Mapping[str, Any]:
    """Compute pooled rank-normalized bulk and 5%/95% tail ESS."""

    import tensorflow as tf
    import tensorflow_probability as tfp

    tensor = _sample_tensor(samples)
    tf.debugging.assert_all_finite(tensor, "samples must be finite")
    split = _split_sample_major(tensor)
    bulk = _cross_chain_ess(_rank_normalize(split))
    lower = tfp.stats.percentile(
        tensor, 5.0, axis=(0, 1), interpolation="linear"
    )
    upper = tfp.stats.percentile(
        tensor, 95.0, axis=(0, 1), interpolation="linear"
    )
    lower_ess = _cross_chain_ess(tf.cast(split <= lower, tf.float64))
    upper_ess = _cross_chain_ess(tf.cast(split >= upper, tf.float64))
    return {
        "bulk": bulk,
        "lower_5pct": lower_ess,
        "upper_95pct": upper_ess,
        "tail": tf.minimum(lower_ess, upper_ess),
    }


def posterior_mean_diagnostics(samples: Any) -> Mapping[str, Any]:
    """Return pooled/per-chain means, SDs, ESS, and mean MCSE."""

    import tensorflow as tf
    import tensorflow_probability as tfp

    tensor = _sample_tensor(samples)
    tf.debugging.assert_all_finite(tensor, "samples must be finite")
    chains = int(tensor.shape[0])
    draws = int(tensor.shape[1])
    parameters = int(tensor.shape[2])
    pooled = tf.reshape(tensor, (chains * draws, parameters))
    pooled_sd = _sample_sd(pooled, axis=0)
    sample_major = tf.transpose(tensor, perm=(1, 0, 2))
    pooled_mean_ess = _cross_chain_ess(sample_major)
    pooled_mcse = pooled_sd / tf.sqrt(pooled_mean_ess)
    per_chain_ess = tfp.mcmc.effective_sample_size(
        sample_major,
        filter_threshold=None,
        filter_beyond_positive_pairs=True,
        cross_chain_dims=None,
    )
    per_chain_sd = _sample_sd(tensor, axis=1)
    per_chain_mcse = per_chain_sd / tf.sqrt(per_chain_ess)
    ratio = tf.where(
        pooled_sd > 0.0,
        pooled_mcse / pooled_sd,
        tf.fill(tf.shape(pooled_sd), tf.constant(float("nan"), tf.float64)),
    )
    return {
        "pooled_mean": tf.reduce_mean(pooled, axis=0),
        "posterior_sd": pooled_sd,
        "pooled_mean_ess": pooled_mean_ess,
        "mean_mcse": pooled_mcse,
        "mcse_sd_ratio": ratio,
        "per_chain_mean": tf.reduce_mean(tensor, axis=1),
        "per_chain_sd": per_chain_sd,
        "per_chain_ess": per_chain_ess,
        "per_chain_mean_mcse": per_chain_mcse,
    }


def per_chain_ebfmi(initial_energy: Any) -> Any:
    """Compute E-BFMI from sequential initial total Hamiltonians.

    For each chain this is ``mean(diff(E)**2) / sample_variance(E)``.  The
    caller must pass exact total energy, not accepted-state potential energy.
    """

    import tensorflow as tf

    energy = tf.cast(tf.convert_to_tensor(initial_energy), tf.float64)
    if energy.shape.rank != 2 or any(dim is None for dim in energy.shape):
        raise ValueError("initial_energy must have static shape [chain, draw]")
    if int(energy.shape[0]) < 1 or int(energy.shape[1]) < 3:
        raise ValueError("initial_energy requires at least three draws")
    tf.debugging.assert_all_finite(energy, "initial_energy must be finite")
    numerator = tf.reduce_mean(tf.square(energy[:, 1:] - energy[:, :-1]), axis=1)
    denominator = _sample_sd(energy, axis=1) ** 2
    return tf.where(
        denominator > 0.0,
        numerator / denominator,
        tf.fill(tf.shape(denominator), tf.constant(float("nan"), tf.float64)),
    )


def initialization_memory_statistics(samples: Any) -> Mapping[str, Any]:
    """Compare each chain mean with its leave-one-chain-out mean.

    The leave-one-out MCSE combines independent per-chain mean MCSE estimates
    as ``sqrt(sum(mcse_i**2)) / number_of_chains``.
    """

    import tensorflow as tf

    tensor = _sample_tensor(samples)
    mean_diagnostics = posterior_mean_diagnostics(tensor)
    means = mean_diagnostics["per_chain_mean"]
    mcse = mean_diagnostics["per_chain_mean_mcse"]
    chains = tf.cast(tf.shape(means)[0], tf.float64)
    leave_one_out_mean = (tf.reduce_sum(means, axis=0) - means) / (chains - 1.0)
    mcse_square_sum = tf.reduce_sum(tf.square(mcse), axis=0)
    leave_one_out_mcse = tf.sqrt(
        tf.maximum(mcse_square_sum - tf.square(mcse), 0.0)
    ) / (chains - 1.0)
    combined = tf.sqrt(tf.square(mcse) + tf.square(leave_one_out_mcse))
    difference = means - leave_one_out_mean
    standardized = tf.where(
        combined > 0.0,
        difference / combined,
        tf.where(
            tf.equal(difference, 0.0),
            tf.zeros_like(difference),
            tf.sign(difference) * tf.constant(float("inf"), tf.float64),
        ),
    )
    return {
        "chain_mean": means,
        "leave_one_chain_out_mean": leave_one_out_mean,
        "chain_mean_mcse": mcse,
        "leave_one_chain_out_mean_mcse": leave_one_out_mcse,
        "combined_mcse": combined,
        "standardized_difference": standardized,
        "max_abs_standardized_difference": tf.reduce_max(
            tf.abs(standardized), axis=0
        ),
    }


def epoch_drift_statistics(current: Any, previous: Any) -> Mapping[str, Any]:
    """Compare adjacent non-overlapping epochs using MCSE and pooled SD."""

    import tensorflow as tf

    current_tensor = _sample_tensor(current, label="current epoch")
    previous_tensor = _sample_tensor(previous, label="previous epoch")
    if tuple(current_tensor.shape) != tuple(previous_tensor.shape):
        raise ValueError("adjacent epochs must have identical shapes")
    current_mean = posterior_mean_diagnostics(current_tensor)
    previous_mean = posterior_mean_diagnostics(previous_tensor)
    difference = current_mean["pooled_mean"] - previous_mean["pooled_mean"]
    combined_mcse = tf.sqrt(
        tf.square(current_mean["mean_mcse"])
        + tf.square(previous_mean["mean_mcse"])
    )
    standardized = tf.where(
        combined_mcse > 0.0,
        difference / combined_mcse,
        tf.where(
            tf.equal(difference, 0.0),
            tf.zeros_like(difference),
            tf.sign(difference) * tf.constant(float("inf"), tf.float64),
        ),
    )
    sd_ratio = current_mean["posterior_sd"] / previous_mean["posterior_sd"]
    return {
        "current_mean": current_mean["pooled_mean"],
        "previous_mean": previous_mean["pooled_mean"],
        "mean_difference": difference,
        "combined_mcse": combined_mcse,
        "standardized_mean_difference": standardized,
        "abs_standardized_mean_difference": tf.abs(standardized),
        "sd_ratio_current_over_previous": sd_ratio,
    }


def compute_coordinate_diagnostics(samples: Any) -> Mapping[str, Any]:
    """Compute the coordinate-level Phase 29 diagnostic bundle."""

    tensor = _sample_tensor(samples)
    return {
        "rank_normalized_split_rhat": rank_normalized_split_rhat(tensor),
        "rank_normalized_ess": rank_normalized_bulk_tail_ess(tensor),
        "mean": posterior_mean_diagnostics(tensor),
        "initialization_memory": initialization_memory_statistics(tensor),
    }


def _tensor_tree_to_python(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _tensor_tree_to_python(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_tensor_tree_to_python(item) for item in value]
    if isinstance(value, list):
        return [_tensor_tree_to_python(item) for item in value]
    if hasattr(value, "numpy"):
        return _tensor_tree_to_python(value.numpy())
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    return value


def _all_finite(value: Any) -> bool:
    import tensorflow as tf

    tensor = tf.cast(tf.convert_to_tensor(value), tf.float64)
    return bool(tf.reduce_all(tf.math.is_finite(tensor)).numpy())


def _coordinate_screen(
    samples: Any,
    *,
    previous: Any | None,
    thresholds: Phase29DiagnosticThresholds,
    include_transient_screens: bool,
) -> tuple[dict[str, Any], list[str]]:
    import tensorflow as tf

    diagnostics = dict(compute_coordinate_diagnostics(samples))
    rhat = diagnostics["rank_normalized_split_rhat"]["maximum"]
    ess = diagnostics["rank_normalized_ess"]
    mean = diagnostics["mean"]
    memory = diagnostics["initialization_memory"]
    failures: list[str] = []

    if not bool(tf.reduce_all(tf.math.is_finite(rhat)).numpy()):
        failures.append("rank_normalized_split_rhat_nonfinite")
    elif not bool(tf.reduce_all(rhat <= thresholds.rhat_max).numpy()):
        failures.append("rank_normalized_split_rhat_above_threshold")
    if not bool(tf.reduce_all(tf.math.is_finite(ess["bulk"])).numpy()):
        failures.append("bulk_ess_nonfinite")
    elif not bool(tf.reduce_all(ess["bulk"] >= thresholds.bulk_ess_min).numpy()):
        failures.append("bulk_ess_below_threshold")
    if not bool(tf.reduce_all(tf.math.is_finite(ess["tail"])).numpy()):
        failures.append("tail_ess_nonfinite")
    elif not bool(tf.reduce_all(ess["tail"] >= thresholds.tail_ess_min).numpy()):
        failures.append("tail_ess_below_threshold")
    ratio = mean["mcse_sd_ratio"]
    if not bool(tf.reduce_all(tf.math.is_finite(ratio)).numpy()):
        failures.append("mcse_sd_ratio_nonfinite")
    elif not bool(tf.reduce_all(ratio <= thresholds.mcse_sd_ratio_max).numpy()):
        failures.append("mcse_sd_ratio_above_threshold")

    if include_transient_screens:
        maximum = memory["max_abs_standardized_difference"]
        if not bool(tf.reduce_all(tf.math.is_finite(maximum)).numpy()):
            failures.append("initialization_memory_nonfinite")
        elif not bool(
            tf.reduce_all(maximum <= thresholds.initialization_memory_max).numpy()
        ):
            failures.append("initialization_memory_above_threshold")
        if previous is None:
            diagnostics["epoch_drift"] = {"status": "not_applicable_first_epoch"}
        else:
            drift = epoch_drift_statistics(samples, previous)
            diagnostics["epoch_drift"] = drift
            drift_z = drift["abs_standardized_mean_difference"]
            sd_ratio = drift["sd_ratio_current_over_previous"]
            if not bool(
                tf.reduce_all(tf.math.is_finite(drift_z)).numpy()
                and tf.reduce_all(tf.math.is_finite(sd_ratio)).numpy()
            ):
                failures.append("epoch_drift_nonfinite")
            else:
                if not bool(
                    tf.reduce_all(drift_z <= thresholds.epoch_drift_z_max).numpy()
                ):
                    failures.append("epoch_mean_drift_above_threshold")
                if not bool(
                    tf.reduce_all(
                        (sd_ratio >= thresholds.epoch_sd_ratio_min)
                        & (sd_ratio <= thresholds.epoch_sd_ratio_max)
                    ).numpy()
                ):
                    failures.append("epoch_sd_ratio_outside_threshold")
    else:
        diagnostics["epoch_drift"] = {"status": "not_part_of_pilot_screen"}
        diagnostics["initialization_memory_role"] = "explanatory_only_for_pilot"
    return _tensor_tree_to_python(diagnostics), failures


def _evaluate_phase29_window(
    coordinate_samples: Mapping[str, Any],
    *,
    initial_energy: Any,
    is_accepted: Any,
    delta_h: Any,
    previous_coordinate_samples: Mapping[str, Any] | None,
    thresholds: Phase29DiagnosticThresholds,
    include_transient_screens: bool,
    role: str,
) -> dict[str, Any]:
    import tensorflow as tf

    if not coordinate_samples:
        raise ValueError("coordinate_samples must be non-empty")
    if "final_latent" not in coordinate_samples:
        raise ValueError(
            "coordinate_samples must include the final_latent HMC coordinates"
        )
    tensors = {
        str(name): _sample_tensor(samples, label=f"{name} samples")
        for name, samples in coordinate_samples.items()
    }
    reference_shape = tuple(next(iter(tensors.values())).shape)
    if any(tuple(tensor.shape) != reference_shape for tensor in tensors.values()):
        raise ValueError("all coordinate systems must have identical sample shapes")
    chains, draws, _parameters = reference_shape
    energy = tf.cast(tf.convert_to_tensor(initial_energy), tf.float64)
    accepted = tf.cast(tf.convert_to_tensor(is_accepted), tf.bool)
    energy_error = tf.cast(tf.convert_to_tensor(delta_h), tf.float64)
    if tuple(energy.shape) != (chains, draws):
        raise ValueError("initial_energy must match [chain, draw]")
    if tuple(accepted.shape) != (chains, draws):
        raise ValueError("is_accepted must match [chain, draw]")
    if tuple(energy_error.shape) != (chains, draws):
        raise ValueError("delta_h must match [chain, draw]")

    hard_vetoes: list[str] = []
    for name, tensor in tensors.items():
        if not _all_finite(tensor):
            hard_vetoes.append(f"{name}_samples_nonfinite")
    if not _all_finite(energy):
        hard_vetoes.append("initial_energy_nonfinite")
    if not _all_finite(energy_error):
        hard_vetoes.append("delta_h_nonfinite")
    if hard_vetoes:
        return {
            "artifact_schema": "bayesfilter.phase29_posterior_diagnostics.v1",
            "role": role,
            "passed": False,
            "hard_vetoes": list(dict.fromkeys(hard_vetoes)),
            "promotion_vetoes": [],
            "thresholds": thresholds.payload(),
            "coordinate_diagnostics": {},
            "mechanics_diagnostics": {},
            "nonclaims": [
                "finite-sample operational screen only",
                "no convergence or stationarity proof",
            ],
        }

    if bool(tf.reduce_any(tf.abs(energy_error) > thresholds.delta_h_abs_max).numpy()):
        hard_vetoes.append("absolute_delta_h_above_hard_limit")
    latent = tensors["final_latent"]
    moved = tf.reduce_any(
        tf.not_equal(latent[:, 1:, :], latent[:, :-1, :]), axis=(1, 2)
    )
    if not bool(tf.reduce_all(moved).numpy()):
        hard_vetoes.append("unmoved_chain")

    previous = {} if previous_coordinate_samples is None else {
        str(name): _sample_tensor(samples, label=f"previous {name} samples")
        for name, samples in previous_coordinate_samples.items()
    }
    if previous and set(previous) != set(tensors):
        raise ValueError("previous coordinate systems must match current systems")

    coordinate_diagnostics: dict[str, Any] = {}
    promotion_vetoes: list[str] = []
    for name, tensor in tensors.items():
        report, failures = _coordinate_screen(
            tensor,
            previous=previous.get(name),
            thresholds=thresholds,
            include_transient_screens=include_transient_screens,
        )
        coordinate_diagnostics[name] = report
        promotion_vetoes.extend(f"{name}:{failure}" for failure in failures)

    acceptance_by_chain = tf.reduce_mean(tf.cast(accepted, tf.float64), axis=1)
    bfmi = per_chain_ebfmi(energy)
    if not bool(tf.reduce_all(tf.math.is_finite(acceptance_by_chain)).numpy()):
        promotion_vetoes.append("acceptance_nonfinite")
    elif not bool(
        tf.reduce_all(
            (acceptance_by_chain >= thresholds.acceptance_min)
            & (acceptance_by_chain <= thresholds.acceptance_max)
        ).numpy()
    ):
        promotion_vetoes.append("per_chain_acceptance_outside_threshold")
    if not bool(tf.reduce_all(tf.math.is_finite(bfmi)).numpy()):
        promotion_vetoes.append("ebfmi_nonfinite")
    elif not bool(tf.reduce_all(bfmi > thresholds.ebfmi_min).numpy()):
        promotion_vetoes.append("per_chain_ebfmi_below_threshold")

    hard_vetoes = list(dict.fromkeys(hard_vetoes))
    promotion_vetoes = list(dict.fromkeys(promotion_vetoes))
    return {
        "artifact_schema": "bayesfilter.phase29_posterior_diagnostics.v1",
        "role": role,
        "passed": not hard_vetoes and not promotion_vetoes,
        "hard_vetoes": hard_vetoes,
        "promotion_vetoes": promotion_vetoes,
        "thresholds": thresholds.payload(),
        "coordinate_diagnostics": coordinate_diagnostics,
        "mechanics_diagnostics": _tensor_tree_to_python(
            {
                "acceptance_rate_by_chain": acceptance_by_chain,
                "ebfmi_by_chain": bfmi,
                "chain_moved": moved,
                "max_abs_delta_h": tf.reduce_max(tf.abs(energy_error)),
            }
        ),
        "diagnostic_roles": {
            "hard_vetoes": "continuation_veto",
            "rhat_ess_mcse_acceptance_ebfmi": "promotion_veto",
            "initialization_memory_epoch_drift": (
                "warmup_qualification_promotion_veto"
                if include_transient_screens
                else "explanatory_only_for_pilot"
            ),
        },
        "nonclaims": [
            "finite-sample operational screen only",
            "no convergence or stationarity proof",
            "no sampler ranking, identification, or scientific-validity claim",
        ],
    }


def evaluate_phase29_warmup_epoch(
    coordinate_samples: Mapping[str, Any],
    *,
    initial_energy: Any,
    is_accepted: Any,
    delta_h: Any,
    previous_coordinate_samples: Mapping[str, Any] | None = None,
    thresholds: Phase29DiagnosticThresholds | None = None,
) -> dict[str, Any]:
    """Evaluate one non-overlapping Phase 29 warm-up epoch."""

    return _evaluate_phase29_window(
        coordinate_samples,
        initial_energy=initial_energy,
        is_accepted=is_accepted,
        delta_h=delta_h,
        previous_coordinate_samples=previous_coordinate_samples,
        thresholds=thresholds or Phase29DiagnosticThresholds(),
        include_transient_screens=True,
        role="warmup_diagnostic",
    )


def evaluate_phase29_posterior_pilot(
    coordinate_samples: Mapping[str, Any],
    *,
    initial_energy: Any,
    is_accepted: Any,
    delta_h: Any,
    thresholds: Phase29DiagnosticThresholds | None = None,
) -> dict[str, Any]:
    """Evaluate the disjoint Phase 29 held-out posterior pilot."""

    return _evaluate_phase29_window(
        coordinate_samples,
        initial_energy=initial_energy,
        is_accepted=is_accepted,
        delta_h=delta_h,
        previous_coordinate_samples=None,
        thresholds=thresholds or Phase29DiagnosticThresholds(),
        include_transient_screens=False,
        role="posterior_pilot",
    )


__all__ = [
    "Phase29DiagnosticThresholds",
    "compute_coordinate_diagnostics",
    "epoch_drift_statistics",
    "evaluate_phase29_posterior_pilot",
    "evaluate_phase29_warmup_epoch",
    "initialization_memory_statistics",
    "per_chain_ebfmi",
    "posterior_mean_diagnostics",
    "rank_normalized_bulk_tail_ess",
    "rank_normalized_split_rhat",
]
