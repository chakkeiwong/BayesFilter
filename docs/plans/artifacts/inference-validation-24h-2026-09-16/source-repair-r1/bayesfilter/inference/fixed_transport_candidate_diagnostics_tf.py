"""TensorFlow post-chain diagnostics for the fixed-transport candidate screen.

This is an eager diagnostic boundary, not a sampling or transition kernel.
It reuses BayesFilter's convergence estimators and materializes only reports.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import tensorflow as tf

from .hmc_convergence import (
    RankNormalizedHMCThresholds,
    _real_fft_cross_chain_ess,
    _split_chains,
    rank_normalized_hmc_diagnostics,
)
from .fixed_transport_candidate_selection import FixedTransportCandidateSelectionConfig


def fixed_transport_candidate_diagnostics(
    samples: Any,
    *,
    initial_state: Any,
    mechanics: Mapping[str, Any],
    config: FixedTransportCandidateSelectionConfig,
    parameter_names: Sequence[str],
) -> Mapping[str, Any]:
    """Return observations; the candidate-set protocol owns pass/fail decisions.

    Samples and starts must share coordinates. Mean MCSE/SD is estimated using
    the ESS of the unranked draws, not rank-normalized bulk ESS. Both are
    reported so consumers cannot confuse the two estimands.
    """

    values = tf.convert_to_tensor(samples, dtype=tf.float64)
    starts = tf.convert_to_tensor(initial_state, dtype=tf.float64)
    if values.shape.rank != 3 or starts.shape != values.shape[1:]:
        raise ValueError("samples [draw, chain, parameter] and initial_state shapes must agree")
    rank = rank_normalized_hmc_diagnostics(
        values,
        parameter_names=parameter_names,
        thresholds=RankNormalizedHMCThresholds(
            rhat_max=config.rhat_max,
            bulk_ess_min=config.min_bulk_ess,
            tail_ess_min=config.min_tail_ess,
        ),
    )
    movement = tf.reduce_max(tf.abs(values - starts[tf.newaxis, :, :]), axis=(0, 2))
    mean_ess = _real_fft_cross_chain_ess(_split_chains(values))
    sample_count = tf.cast(tf.shape(values)[0] * tf.shape(values)[1], tf.float64)
    standard_deviation = tf.math.reduce_std(values, axis=(0, 1)) * tf.sqrt(
        sample_count / (sample_count - 1.0)
    )
    mean_mcse_sd = tf.math.rsqrt(mean_ess)
    valid_precision = tf.logical_and(
        tf.logical_and(tf.math.is_finite(mean_ess), mean_ess > 0.0),
        tf.logical_and(
            tf.math.is_finite(standard_deviation),
            tf.reduce_all(tf.math.reduce_variance(values, axis=0) > 0.0, axis=0),
        ),
    )
    mean_mcse_sd = tf.where(valid_precision, mean_mcse_sd, tf.constant(float("inf"), tf.float64))
    status = mechanics.get("target_status_telemetry") or {}
    finite = bool(tf.reduce_all(tf.math.is_finite(values)).numpy()) and all(
        mechanics.get(name) is True
        for name in ("log_accept_ratio_finite", "target_log_prob_finite")
    ) and all(
        mechanics.get(name) is not False
        for name in ("proposed_target_log_prob_finite", "target_score_finite")
    )
    return {
        "all_finite": finite,
        "target_status_valid": status.get("all_status_valid") is True,
        "all_chain_movement": bool(tf.reduce_all(movement > 0.0).numpy()),
        "native_divergence_status": mechanics.get("divergence_status", "unavailable"),
        "native_divergence_count": mechanics.get("divergence_count"),
        "max_rhat": rank.get("max_rhat"),
        "min_bulk_ess": rank.get("min_bulk_ess"),
        "min_tail_ess": rank.get("min_tail_ess"),
        "max_mcse_sd_ratio": float(tf.reduce_max(mean_mcse_sd).numpy()),
        "mean_ess": [float(value) for value in tf.unstack(mean_ess)],
        "mean_mcse_sd_ratio": [float(value) for value in tf.unstack(mean_mcse_sd)],
        "mean_mcse": [float(value) for value in tf.unstack(standard_deviation * mean_mcse_sd)],
        "sample_standard_deviation": [float(value) for value in tf.unstack(standard_deviation)],
        "mcse_method": "unranked_split_chain_ess",
        "movement_max_abs_by_chain": [float(value) for value in tf.unstack(movement)],
        "rank_normalized_diagnostics": rank,
        "hard_vetoes": [],
    }
