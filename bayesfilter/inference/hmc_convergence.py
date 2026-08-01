"""Rank-normalized multi-chain diagnostics for deterministic HMC gates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import tensorflow as tf
import tensorflow_probability as tfp


RANK_NORMALIZED_SPLIT_RHAT_DEFINITION = (
    "max(rank-normalized split R-hat, folded rank-normalized split R-hat)"
)


@dataclass(frozen=True)
class RankNormalizedHMCThresholds:
    """All-parameter thresholds used by a fixed HMC convergence screen."""

    rhat_max: float = 1.01
    bulk_ess_min: float = 1000.0
    tail_ess_min: float = 400.0

    def __post_init__(self) -> None:
        rhat_max = float(self.rhat_max)
        bulk_ess_min = float(self.bulk_ess_min)
        tail_ess_min = float(self.tail_ess_min)
        if not 1.0 < rhat_max:
            raise ValueError("rhat_max must be greater than 1")
        if bulk_ess_min <= 0.0:
            raise ValueError("bulk_ess_min must be positive")
        if tail_ess_min <= 0.0:
            raise ValueError("tail_ess_min must be positive")
        object.__setattr__(self, "rhat_max", rhat_max)
        object.__setattr__(self, "bulk_ess_min", bulk_ess_min)
        object.__setattr__(self, "tail_ess_min", tail_ess_min)

    def payload(self) -> Mapping[str, float]:
        return {
            "rhat_max": self.rhat_max,
            "bulk_ess_min": self.bulk_ess_min,
            "tail_ess_min": self.tail_ess_min,
        }


def rank_normalized_split_rhat_summary(
    draws: Any,
    *,
    rhat_max: float = 1.01,
) -> Mapping[str, Any]:
    """Compute the shared rank-normalized split/folded R-hat screen."""

    threshold = float(rhat_max)
    if not 1.0 < threshold < float("inf"):
        raise ValueError("rhat_max must be finite and greater than 1")

    values = tf.convert_to_tensor(draws, dtype=tf.float64)
    if values.shape.rank != 3:
        raise ValueError("draws must have shape [draw, chain, parameter]")
    static_shape = values.shape.as_list()
    if any(dim is None for dim in static_shape):
        raise ValueError("draws must have a fully static shape")
    draw_count, chain_count, parameter_count = (int(dim) for dim in static_shape)
    if draw_count < 4:
        raise ValueError("at least four draws per chain are required")
    if chain_count < 2:
        raise ValueError("at least two chains are required")

    input_all_finite = bool(tf.reduce_all(tf.math.is_finite(values)).numpy())
    if not input_all_finite:
        return {
            "schema": "bayesfilter.rank_normalized_split_rhat_summary.v1",
            "rhat_definition": RANK_NORMALIZED_SPLIT_RHAT_DEFINITION,
            "rhat_threshold": threshold,
            "passed": False,
            "input_all_finite": False,
            "diagnostics_all_finite": False,
            "draw_count_per_chain": draw_count,
            "chain_count": chain_count,
            "parameter_count": parameter_count,
            "split_draw_count_per_chain": draw_count // 2,
            "split_chain_count": 2 * chain_count,
            "rank_normalized_split_rhat": (None,) * parameter_count,
            "folded_rank_normalized_split_rhat": (None,) * parameter_count,
            "rhat": (None,) * parameter_count,
            "max_rank_normalized_split_rhat": None,
            "max_folded_rank_normalized_split_rhat": None,
            "max_finite_rhat": None,
            "finite_rhat_count": 0,
            "nonfinite_rhat_count": parameter_count,
            "hard_vetoes": ("nonfinite_input_draws",),
        }

    rank_rhat, folded_rhat, rhat = _rank_normalized_split_rhat_components(values)
    finite = tf.logical_and(
        tf.math.is_finite(rhat),
        tf.logical_and(
            tf.math.is_finite(rank_rhat),
            tf.math.is_finite(folded_rhat),
        ),
    )
    finite_count = int(tf.reduce_sum(tf.cast(finite, tf.int32)).numpy())
    nonfinite_count = parameter_count - finite_count
    max_finite = _maximum_finite_tensor_value(rhat)
    passed = bool(
        finite_count == parameter_count
        and max_finite is not None
        and max_finite <= threshold
    )
    return {
        "schema": "bayesfilter.rank_normalized_split_rhat_summary.v1",
        "rhat_definition": RANK_NORMALIZED_SPLIT_RHAT_DEFINITION,
        "rhat_threshold": threshold,
        "passed": passed,
        "input_all_finite": True,
        "diagnostics_all_finite": nonfinite_count == 0,
        "draw_count_per_chain": draw_count,
        "chain_count": chain_count,
        "parameter_count": parameter_count,
        "split_draw_count_per_chain": draw_count // 2,
        "split_chain_count": 2 * chain_count,
        "rank_normalized_split_rhat": tuple(
            float(item) for item in rank_rhat.numpy().reshape(-1)
        ),
        "folded_rank_normalized_split_rhat": tuple(
            float(item) for item in folded_rhat.numpy().reshape(-1)
        ),
        "rhat": tuple(float(item) for item in rhat.numpy().reshape(-1)),
        "max_rank_normalized_split_rhat": _maximum_finite_tensor_value(rank_rhat),
        "max_folded_rank_normalized_split_rhat": _maximum_finite_tensor_value(
            folded_rhat
        ),
        "max_finite_rhat": max_finite,
        "finite_rhat_count": finite_count,
        "nonfinite_rhat_count": nonfinite_count,
        "hard_vetoes": () if nonfinite_count == 0 else (
            "nonfinite_rank_normalized_rhat",
        ),
    }


def rank_normalized_hmc_diagnostics(
    draws: Any,
    *,
    parameter_names: Sequence[str],
    thresholds: RankNormalizedHMCThresholds,
) -> Mapping[str, Any]:
    """Compute rank-normalized split/folded R-hat and bulk/tail ESS.

    ``draws`` must have shape ``[draw, chain, parameter]`` and already be in
    the model coordinates named by ``parameter_names``.
    """

    values = tf.convert_to_tensor(draws, dtype=tf.float64)
    if values.shape.rank != 3:
        raise ValueError("draws must have shape [draw, chain, parameter]")
    static_shape = values.shape.as_list()
    if any(dim is None for dim in static_shape):
        raise ValueError("draws must have a fully static shape")
    draw_count, chain_count, parameter_count = (int(dim) for dim in static_shape)
    names = tuple(str(name) for name in parameter_names)
    if draw_count < 4:
        raise ValueError("at least four draws per chain are required")
    if chain_count < 2:
        raise ValueError("at least two chains are required")
    if len(names) != parameter_count:
        raise ValueError("parameter_names must match the trailing draw dimension")

    all_finite = bool(tf.reduce_all(tf.math.is_finite(values)).numpy())
    if not all_finite:
        return _failed_nonfinite_payload(
            draw_count=draw_count,
            chain_count=chain_count,
            parameter_names=names,
            thresholds=thresholds,
        )

    rhat_summary = rank_normalized_split_rhat_summary(
        values,
        rhat_max=thresholds.rhat_max,
    )
    rank_rhat = tf.constant(
        rhat_summary["rank_normalized_split_rhat"],
        dtype=tf.float64,
    )
    folded_rhat = tf.constant(
        rhat_summary["folded_rank_normalized_split_rhat"],
        dtype=tf.float64,
    )
    rhat = tf.constant(rhat_summary["rhat"], dtype=tf.float64)
    rank_values = _rank_normalize(values)
    split_rank = _split_chains(rank_values)
    bulk_ess = tfp.mcmc.effective_sample_size(
        split_rank,
        filter_beyond_positive_pairs=True,
        cross_chain_dims=1,
    )

    q05 = tfp.stats.percentile(
        values,
        5.0,
        axis=(0, 1),
        interpolation="linear",
    )
    q95 = tfp.stats.percentile(
        values,
        95.0,
        axis=(0, 1),
        interpolation="linear",
    )
    lower_indicator = tf.cast(values <= q05[tf.newaxis, tf.newaxis, :], tf.float64)
    upper_indicator = tf.cast(values >= q95[tf.newaxis, tf.newaxis, :], tf.float64)
    lower_ess = tfp.mcmc.effective_sample_size(
        _split_chains(lower_indicator),
        filter_beyond_positive_pairs=True,
        cross_chain_dims=1,
    )
    upper_ess = tfp.mcmc.effective_sample_size(
        _split_chains(upper_indicator),
        filter_beyond_positive_pairs=True,
        cross_chain_dims=1,
    )
    tail_ess = tf.minimum(lower_ess, upper_ess)

    finite_diagnostics = tf.logical_and(
        tf.math.is_finite(rhat),
        tf.logical_and(tf.math.is_finite(bulk_ess), tf.math.is_finite(tail_ess)),
    )
    parameter_pass = tf.logical_and(
        finite_diagnostics,
        tf.logical_and(
            rhat <= tf.constant(thresholds.rhat_max, dtype=tf.float64),
            tf.logical_and(
                bulk_ess >= tf.constant(thresholds.bulk_ess_min, dtype=tf.float64),
                tail_ess >= tf.constant(thresholds.tail_ess_min, dtype=tf.float64),
            ),
        ),
    )
    parameter_rows = tuple(
        {
            "parameter": name,
            "rank_normalized_split_rhat": float(rank_rhat[index].numpy()),
            "folded_rank_normalized_split_rhat": float(folded_rhat[index].numpy()),
            "rhat": float(rhat[index].numpy()),
            "bulk_ess": float(bulk_ess[index].numpy()),
            "tail_ess": float(tail_ess[index].numpy()),
            "lower_tail_ess": float(lower_ess[index].numpy()),
            "upper_tail_ess": float(upper_ess[index].numpy()),
            "passed": bool(parameter_pass[index].numpy()),
        }
        for index, name in enumerate(names)
    )
    return {
        "schema": "bayesfilter.rank_normalized_hmc_diagnostics.v1",
        "passed": bool(tf.reduce_all(parameter_pass).numpy()),
        "input_all_finite": True,
        "diagnostics_all_finite": bool(tf.reduce_all(finite_diagnostics).numpy()),
        "draw_count_per_chain": draw_count,
        "chain_count": chain_count,
        "parameter_count": parameter_count,
        "split_draw_count_per_chain": draw_count // 2,
        "split_chain_count": 2 * chain_count,
        "thresholds": thresholds.payload(),
        "definitions": {
            "rank_transform": "Blom average-rank normal score",
            "rhat": RANK_NORMALIZED_SPLIT_RHAT_DEFINITION,
            "bulk_ess": "split-chain cross-chain ESS of rank-normalized draws",
            "tail_ess": "minimum split-chain cross-chain ESS of pooled q05/q95 indicators",
            "autocorrelation_truncation": "TFP initial positive pairs",
            "quantile_interpolation": "linear",
        },
        "max_rhat": float(tf.reduce_max(rhat).numpy()),
        "min_bulk_ess": float(tf.reduce_min(bulk_ess).numpy()),
        "min_tail_ess": float(tf.reduce_min(tail_ess).numpy()),
        "parameter_diagnostics": parameter_rows,
        "hard_vetoes": () if bool(tf.reduce_all(finite_diagnostics).numpy()) else (
            "nonfinite_convergence_diagnostic",
        ),
        "nonclaims": (
            "all-parameter HMC convergence screen only",
            "no posterior recovery claim",
            "no sampler superiority claim",
            "no production or default readiness claim",
        ),
    }


def _rank_normalize(values: tf.Tensor) -> tf.Tensor:
    shape = tf.shape(values)
    flat = tf.reshape(values, [-1, shape[-1]])
    count = tf.shape(flat)[0]

    def rank_column(column: tf.Tensor) -> tf.Tensor:
        order = tf.argsort(column, stable=True)
        sorted_column = tf.gather(column, order)
        new_group = tf.concat(
            [
                tf.constant([True]),
                tf.not_equal(sorted_column[1:], sorted_column[:-1]),
            ],
            axis=0,
        )
        group = tf.cumsum(tf.cast(new_group, tf.int32)) - 1
        ranks = tf.cast(tf.range(1, count + 1), tf.float64)
        rank_sum = tf.math.segment_sum(ranks, group)
        group_count = tf.math.segment_sum(tf.ones_like(ranks), group)
        sorted_ranks = tf.gather(rank_sum / group_count, group)
        return tf.gather(sorted_ranks, tf.argsort(order, stable=True))

    ranks = tf.map_fn(
        rank_column,
        tf.transpose(flat, [1, 0]),
        fn_output_signature=tf.TensorSpec(shape=(None,), dtype=tf.float64),
    )
    ranks = tf.transpose(ranks, [1, 0])
    probability = (ranks - 3.0 / 8.0) / (tf.cast(count, tf.float64) + 1.0 / 4.0)
    normal = tfp.distributions.Normal(
        loc=tf.constant(0.0, tf.float64),
        scale=tf.constant(1.0, tf.float64),
    )
    return tf.reshape(normal.quantile(probability), shape)


def _rank_normalized_split_rhat_components(
    values: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    rank_values = _rank_normalize(values)
    pooled_median = tfp.stats.percentile(
        values,
        50.0,
        axis=(0, 1),
        interpolation="linear",
    )
    folded_values = tf.abs(values - pooled_median[tf.newaxis, tf.newaxis, :])
    folded_rank_values = _rank_normalize(folded_values)
    rank_rhat = tfp.mcmc.potential_scale_reduction(
        _split_chains(rank_values),
        independent_chain_ndims=1,
        split_chains=False,
    )
    folded_rhat = tfp.mcmc.potential_scale_reduction(
        _split_chains(folded_rank_values),
        independent_chain_ndims=1,
        split_chains=False,
    )
    return rank_rhat, folded_rhat, tf.maximum(rank_rhat, folded_rhat)


def _maximum_finite_tensor_value(values: tf.Tensor) -> float | None:
    finite_values = tf.boolean_mask(values, tf.math.is_finite(values))
    if int(tf.size(finite_values).numpy()) == 0:
        return None
    return float(tf.reduce_max(finite_values).numpy())


def _split_chains(values: tf.Tensor) -> tf.Tensor:
    draw_count = int(values.shape[0])
    chain_count = int(values.shape[1])
    half = draw_count // 2
    first = values[:half]
    last = values[-half:]
    return tf.reshape(
        tf.stack([first, last], axis=2),
        [half, 2 * chain_count, int(values.shape[2])],
    )


def _failed_nonfinite_payload(
    *,
    draw_count: int,
    chain_count: int,
    parameter_names: tuple[str, ...],
    thresholds: RankNormalizedHMCThresholds,
) -> Mapping[str, Any]:
    rows = tuple(
        {
            "parameter": name,
            "rank_normalized_split_rhat": None,
            "folded_rank_normalized_split_rhat": None,
            "rhat": None,
            "bulk_ess": None,
            "tail_ess": None,
            "lower_tail_ess": None,
            "upper_tail_ess": None,
            "passed": False,
        }
        for name in parameter_names
    )
    return {
        "schema": "bayesfilter.rank_normalized_hmc_diagnostics.v1",
        "passed": False,
        "input_all_finite": False,
        "diagnostics_all_finite": False,
        "draw_count_per_chain": draw_count,
        "chain_count": chain_count,
        "parameter_count": len(parameter_names),
        "split_draw_count_per_chain": draw_count // 2,
        "split_chain_count": 2 * chain_count,
        "thresholds": thresholds.payload(),
        "max_rhat": None,
        "min_bulk_ess": None,
        "min_tail_ess": None,
        "parameter_diagnostics": rows,
        "hard_vetoes": ("nonfinite_input_draws",),
        "nonclaims": (
            "all-parameter HMC convergence screen only",
            "no posterior recovery claim",
            "no sampler superiority claim",
            "no production or default readiness claim",
        ),
    }
