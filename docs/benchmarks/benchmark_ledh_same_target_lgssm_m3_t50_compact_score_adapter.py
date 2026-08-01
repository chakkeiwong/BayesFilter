"""Unified-harness adapter for the compact LGSSM value/score route."""

from __future__ import annotations

import argparse
from typing import Any, Mapping, Sequence

import tensorflow as tf

from bayesfilter.highdim.ledh_historical_raw_policy import (
    require_historical_raw_diagnostic_opt_in,
)
from docs.benchmarks import benchmark_ledh_same_target_lgssm_m3_t50_value as value_mod


ROW_ID = value_mod.ROW_ID
TRUTH_THETA = tuple(value_mod.TRUTH_THETA)
PARAMETER_NAMES = tuple(value_mod.PARAMETER_NAMES)
COMPACT_SCORE_ROUTE_ID = value_mod.COMPACT_SCORE_ROUTE_ID
FULL_ROW_SINKHORN_EPSILON = value_mod.FULL_ROW_SINKHORN_EPSILON


def _configure_precision(args: argparse.Namespace) -> dict[str, Any]:
    return value_mod._configure_precision(args)  # noqa: SLF001


def _require_compact_args(args: argparse.Namespace) -> None:
    require_historical_raw_diagnostic_opt_in(
        args, route_name="LGSSM compact raw-barycentric adapter"
    )
    if args.transport_gradient_mode != (
        value_mod.core_tf.MANUAL_STREAMING_FINITE_TRANSPORT_GRADIENT_MODE
    ):
        raise ValueError("compact LGSSM score requires manual streaming finite transport")
    if args.transport_ad_mode != "full":
        raise ValueError("compact LGSSM score requires transport_ad_mode='full'")


def _candidate_tensors(
    prepared_tensors: Mapping[str, tf.Tensor],
    theta: tf.Tensor,
) -> dict[str, tf.Tensor]:
    tensors = dict(prepared_tensors)
    batch_size = int(tf.convert_to_tensor(tensors["initial_noise"]).shape[0])
    components = value_mod._lgssm_components(theta, batch_size)  # noqa: SLF001
    tensors["initial_particles"] = (
        tf.convert_to_tensor(tensors["initial_noise"], dtype=value_mod.DTYPE)
        * components["initial_std"][tf.newaxis, tf.newaxis, :]
    )
    return tensors


def _prepare_compact_xla_inputs(args: argparse.Namespace) -> dict[str, Any]:
    _configure_precision(args)
    _require_compact_args(args)
    if len(args.batch_seeds) != 1:
        raise ValueError("LGSSM XLA score shards require exactly one seed")
    theta = tf.constant(TRUTH_THETA, dtype=value_mod.DTYPE)
    tensors = value_mod._build_lgssm_manual_tensors(args, theta)  # noqa: SLF001
    return {
        "tensors": tensors,
        "semantics": {
            "row_id": ROW_ID,
            "target_observation_policy": "lgssm_gaussian_observation_density",
            "theta_coordinate_system": "physical_benchmark_exact_oracle",
            "candidate_dependent_initial_particles": True,
        },
    }


def _compact_value_and_score_from_components(
    args: argparse.Namespace,
    theta_values: tf.Tensor | Sequence[float],
    *,
    prepared_tensors: Mapping[str, tf.Tensor] | None = None,
) -> dict[str, tf.Tensor]:
    _configure_precision(args)
    _require_compact_args(args)
    theta = tf.reshape(
        tf.convert_to_tensor(theta_values, dtype=value_mod.DTYPE),
        [len(PARAMETER_NAMES)],
    )
    tensors = (
        value_mod._build_lgssm_manual_tensors(args, theta)  # noqa: SLF001
        if prepared_tensors is None
        else dict(prepared_tensors)
    )
    return value_mod._compact_value_and_score_from_components(  # noqa: SLF001
        tensors,
        args,
        theta,
    )


def _manual_value_only_from_components(
    args: argparse.Namespace,
    theta_values: tf.Tensor | Sequence[float],
    *,
    prepared_tensors: Mapping[str, tf.Tensor] | None = None,
) -> dict[str, tf.Tensor]:
    _configure_precision(args)
    _require_compact_args(args)
    theta = tf.reshape(
        tf.convert_to_tensor(theta_values, dtype=value_mod.DTYPE),
        [len(PARAMETER_NAMES)],
    )
    if prepared_tensors is None:
        tensors = value_mod._build_lgssm_manual_tensors(args, theta)  # noqa: SLF001
    else:
        tensors = _candidate_tensors(prepared_tensors, theta)
    return value_mod._same_target_value_from_components(  # noqa: SLF001
        tensors,
        args,
        theta,
    )


def _compact_score_tensor_outputs(
    args: argparse.Namespace,
    theta: tf.Tensor,
    prepared: Mapping[str, Any],
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    result = _compact_value_and_score_from_components(
        args,
        theta,
        prepared_tensors=prepared["tensors"],
    )
    return (
        result["objective"],
        result["log_likelihood"],
        result["gradient_tensor"],
        result["per_seed_gradient"],
    )


def _value_tensor_outputs(
    args: argparse.Namespace,
    theta: tf.Tensor,
    prepared: Mapping[str, Any],
) -> tuple[tf.Tensor, tf.Tensor]:
    result = _manual_value_only_from_components(
        args,
        theta,
        prepared_tensors=prepared["tensors"],
    )
    return result["objective"], result["log_likelihood"]
