#!/usr/bin/env python3
"""Fit and evaluate the coupled nonlinear Zhao-Cui-inspired rung-2 APF.

The assembled route is an extension/invention.  Zhao-Cui squared-TT and
paired-core marginal operations are used as source-grounded primitives; the
nonlinear block target, reordered compiler, numerical inverse, and fixed-branch
APF score are local BayesFilter mechanisms.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import itertools
import json
import math
import os
from pathlib import Path
import platform
import string
import subprocess
import sys
import time
import traceback
from typing import Mapping, Sequence


if "--cpu-reference" in sys.argv:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
else:
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/bayesfilter-mpl")
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

import tensorflow as tf

from bayesfilter.runtime.gpu_memory_policy import (  # noqa: E402
    configure_tensorflow_gpu_memory_growth,
)


def _configure_device_early(cpu_reference: bool) -> Mapping[str, object]:
    if cpu_reference:
        if tf.config.list_physical_devices("GPU"):
            raise RuntimeError("CPU reference mode must hide all GPU devices")
        tf.config.experimental.enable_tensor_float_32_execution(False)
        return {
            "execution_class": "explicit_cpu_reference",
            "physical_gpus": [],
            "logical_gpus": [],
            "online_device": "/CPU:0",
            "gpu_memory_policy": "N/A: CUDA_VISIBLE_DEVICES=-1 before TensorFlow import",
            "tf32_enabled": False,
        }
    memory_policy = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    tf.config.experimental.enable_tensor_float_32_execution(True)
    logical = tf.config.list_logical_devices("GPU")
    if not logical:
        raise RuntimeError("trusted rung 2 requires a logical GPU")
    return {
        "execution_class": "trusted_visible_gpu",
        "physical_gpus": [row["device"] for row in memory_policy["physical_devices"]],
        "logical_gpus": [device.name for device in logical],
        "online_device": "/GPU:0",
        "gpu_memory_policy": memory_policy,
        "tf32_enabled": bool(tf.config.experimental.tensor_float_32_execution_enabled()),
    }


DEVICE = _configure_device_early("--cpu-reference" in sys.argv)

from bayesfilter.highdim.bases import (  # noqa: E402
    BoundedInterval,
    LegendreBasis1D,
    ProductBasis,
)
from bayesfilter.highdim.diagnostics import (  # noqa: E402
    DensityMeasure,
    HighDimStatus,
    MassMeasure,
    MeasureConvention,
)
from bayesfilter.highdim.filtering import legendre_gauss_nodes_weights  # noqa: E402
from bayesfilter.highdim.fitting import FixedTTFitConfig  # noqa: E402
from bayesfilter.highdim.squared_tt import SquaredTTDensity  # noqa: E402
from bayesfilter.highdim.stochastic_density_training import (  # noqa: E402
    P75ObjectiveBatch,
    P75TrainableTTConfig,
    TrainableFunctionalTT,
    make_adam_optimizer,
)
from bayesfilter.highdim.transport import FixedTTSIRTTransport, KRCDFConfig  # noqa: E402
from bayesfilter.highdim.zhao_cui_coupled_nonlinear import (  # noqa: E402
    CoupledNonlinearGaussianModel,
    GaussianQuantileCoordinateMap,
)
from bayesfilter.highdim.zhao_cui_frozen_proposal_apf_tf import (  # noqa: E402
    PreparedFrozenProposalBranch,
    combine_fixed_ttsirt_block_compilations,
    compile_fixed_ttsirt_proposal_branch,
    prepare_frozen_proposal_apf_program,
)


FIT_DTYPE = tf.float64
ONLINE_DTYPE = tf.float32
PLAN_PATH = "docs/plans/bayesfilter-zhao-cui-coupled-nonlinear-blockwise-ttsirt-apf-rung2-plan-2026-07-22.md"
ARTIFACT_SCHEMA = "bayesfilter.zhao_cui_coupled_nonlinear_ttsirt_apf.rung2.v3"
CANONICAL_SCOPE = (24, 3, 512)
DEBUG_SCOPE = (2, 2, 16)
TARGET_LOG_NORMALIZER_TOLERANCE = 0.05
CONDITIONAL_LOG_RMS_TOLERANCE = 0.75
SCORE_FD_TOLERANCE = 0.05
ESS_FRACTION_TOLERANCE = 0.20
ROUNDTRIP_TOLERANCE = 1e-4


@dataclass(frozen=True)
class ProposalRandomness:
    initial_uniforms: tf.Tensor
    ancestor_uniforms: tf.Tensor
    transition_uniforms: tf.Tensor


@dataclass(frozen=True)
class FitSpec:
    degree: int
    rank: int
    scale: float
    l1_weight: float

    def payload(self) -> Mapping[str, object]:
        return {
            "degree": self.degree,
            "rank": self.rank,
            "scale": self.scale,
            "l1_weight": self.l1_weight,
        }


@dataclass(frozen=True)
class FittedBlockProposal:
    spec: FitSpec
    coordinate_map: GaussianQuantileCoordinateMap
    transports: tuple[FixedTTSIRTTransport, ...]
    diagnostics: Mapping[str, object]


def _convention() -> MeasureConvention:
    return MeasureConvention(
        density_measure=DensityMeasure.REFERENCE_MEASURE,
        mass_measure=MassMeasure.REFERENCE_MEASURE,
        reference_weight_name="uniform_probability_on_minus_one_one",
    )


def _basis(dimension: int, degree: int) -> ProductBasis:
    return ProductBasis(
        [LegendreBasis1D(BoundedInterval(-1.0, 1.0), degree) for _ in range(dimension)],
        _convention(),
    )


def _quadrature(dimension: int, order: int) -> tuple[tf.Tensor, tf.Tensor]:
    nodes, weights = legendre_gauss_nodes_weights(order)
    axis_nodes = [nodes for _ in range(dimension)]
    axis_weights = [0.5 * weights for _ in range(dimension)]
    mesh = tf.meshgrid(*axis_nodes, indexing="ij")
    weight_mesh = tf.meshgrid(*axis_weights, indexing="ij")
    points = tf.stack([tf.reshape(value, [-1]) for value in mesh], axis=1)
    product_weights = tf.ones([tf.shape(points)[0]], FIT_DTYPE)
    for value in weight_mesh:
        product_weights = product_weights * tf.reshape(value, [-1])
    return points, product_weights


def _reference_log_density(dimension: int) -> tf.Tensor:
    return -tf.cast(dimension, FIT_DTYPE) * tf.math.log(tf.constant(2.0, FIT_DTYPE))


def _map_for(model: CoupledNonlinearGaussianModel, scale: float, dimension: int) -> GaussianQuantileCoordinateMap:
    locations = tf.tile(
        tf.constant([model.initial_mean_s, model.initial_mean_i], FIT_DTYPE),
        [dimension // 2],
    )
    return GaussianQuantileCoordinateMap(
        locations=locations,
        scales=tf.fill([dimension], tf.constant(scale, FIT_DTYPE)),
    )


def _observations(time_steps: int) -> tf.Tensor:
    base = tf.constant([0.19, 0.23, 0.21], FIT_DTYPE)
    if time_steps > int(base.shape[0]):
        raise ValueError("rung 2 supports at most three predeclared observations")
    return base[:time_steps]


def _reference_target_log(
    model: CoupledNonlinearGaussianModel,
    coordinate_map: GaussianQuantileCoordinateMap,
    points: tf.Tensor,
    observation: tf.Tensor,
    time_index: int,
    previous_density: SquaredTTDensity | None,
) -> tf.Tensor:
    physical, log_det = coordinate_map.forward(points)
    log_det_components = coordinate_map.forward_log_det_components(points)
    theta = tf.zeros([3], FIT_DTYPE)
    if time_index == 0:
        log_target = model.initial_log_density(theta, physical)
        log_target = log_target + model.observation_log_density(
            theta, physical, observation, 0
        )
        return log_target + log_det - _reference_log_density(2)
    if previous_density is None:
        raise ValueError("adjacent target requires a previous fitted density")
    previous_local = points[:, :2]
    previous_physical = physical[:, :2]
    current_physical = physical[:, 2:]
    retained_axes = (
        (0, 1)
        if previous_density.sqrt_tt.product_basis.dimension == 2
        else (2, 3)
    )
    previous_log = tf.math.log(
        previous_density.normalized_marginal_density_values(
            retained_axes, previous_local
        )
    )
    transition_log = model.transition_log_density(
        theta, previous_physical, current_physical, time_index
    )
    observation_log = model.observation_log_density(
        theta, current_physical, observation, time_index
    )
    return (
        previous_log
        + transition_log
        + observation_log
        + tf.reduce_sum(log_det_components[:, 2:4], axis=1)
        - _reference_log_density(2)
    )


def _quadrature_log_normalizer(log_target: tf.Tensor, weights: tf.Tensor) -> tf.Tensor:
    target = tf.convert_to_tensor(log_target, FIT_DTYPE)
    quadrature_weights = tf.convert_to_tensor(weights, FIT_DTYPE)
    if target.shape.rank != 1 or quadrature_weights.shape != target.shape:
        raise ValueError("target and quadrature weights must be matching vectors")
    if not bool(
        tf.reduce_all(tf.math.is_finite(target)).numpy()
        and tf.reduce_all(tf.math.is_finite(quadrature_weights)).numpy()
        and tf.reduce_all(quadrature_weights > 0.0).numpy()
    ):
        raise ValueError("target and quadrature weights must be finite and positive")
    log_z = tf.reduce_logsumexp(tf.math.log(quadrature_weights) + target)
    if not bool(tf.math.is_finite(log_z).numpy()):
        raise ValueError("target quadrature log normalizer must be finite")
    return log_z


def _normalize_log_target(log_target: tf.Tensor, weights: tf.Tensor) -> tuple[tf.Tensor, float]:
    log_z = _quadrature_log_normalizer(log_target, weights)
    return log_target - log_z, float(log_z.numpy())


def _validation_kl(
    density: SquaredTTDensity,
    points: tf.Tensor,
    weights: tf.Tensor,
    log_target: tf.Tensor,
) -> Mapping[str, float]:
    normalized_target, log_normalizer = _normalize_log_target(log_target, weights)
    target_density = tf.exp(normalized_target)
    fit_log = density.log_density(points)
    pointwise = normalized_target - fit_log
    kl = tf.reduce_sum(weights * target_density * pointwise)
    centered = pointwise - tf.reduce_sum(weights * target_density * pointwise)
    values = tf.sort(tf.abs(pointwise))
    q95 = values[tf.cast(tf.floor(0.95 * tf.cast(tf.shape(values)[0] - 1, FIT_DTYPE)), tf.int32)]
    q99 = values[tf.cast(tf.floor(0.99 * tf.cast(tf.shape(values)[0] - 1, FIT_DTYPE)), tf.int32)]
    finite = (
        bool(tf.math.is_finite(kl).numpy())
        and bool(tf.reduce_all(tf.math.is_finite(centered)).numpy())
        and bool(tf.reduce_all(tf.math.is_finite(values)).numpy())
        and bool(tf.math.is_finite(q95).numpy())
        and bool(tf.math.is_finite(q99).numpy())
    )
    if not finite:
        raise ValueError("non-finite validation KL diagnostics")
    return {
        "kl": float(kl.numpy()),
        "log_normalizer": log_normalizer,
        "log_density_abs_q95": float(q95.numpy()),
        "log_density_abs_q99": float(q99.numpy()),
        "log_density_abs_max": float(tf.reduce_max(values).numpy()),
    }


def _fit_density(
    model: CoupledNonlinearGaussianModel,
    spec: FitSpec,
    points: tf.Tensor,
    weights: tf.Tensor,
    log_target: tf.Tensor,
    *,
    seed: int,
    prefit_steps: int,
    train_steps: int,
) -> tuple[SquaredTTDensity, Mapping[str, object]]:
    normalized_target, log_normalizer = _normalize_log_target(log_target, weights)
    dimension = int(points.shape[1])
    product_basis = _basis(dimension, spec.degree)
    initial_cores, ranks, initializer = _quadrature_tt_svd_initial_cores(
        product_basis,
        points,
        weights,
        tf.exp(0.5 * normalized_target),
        rank_cap=spec.rank,
    )
    config = P75TrainableTTConfig(
        product_basis=product_basis,
        ranks=ranks,
        tau=tf.constant(1e-6, FIT_DTYPE),
        normalizer_floor=tf.constant(1e-12, FIT_DTYPE),
        denominator_floor=tf.constant(1e-12, FIT_DTYPE),
        l1_weight=tf.constant(spec.l1_weight, FIT_DTYPE),
        l2_weight=tf.constant(1e-8, FIT_DTYPE),
        learning_rate=3e-4,
        gradient_clip_norm=10.0,
        seed=seed,
        metadata={
            "scope": "zhao_cui_coupled_nonlinear_rung2",
            "training_role": "calibration_only",
            "route_classification": "extension_or_invention",
            "initializer": "fixed_quadrature_legendre_projection_tt_svd",
            "initializer_classification": "extension_or_invention",
        },
    )
    trainer = TrainableFunctionalTT(config, initial_cores=initial_cores)
    optimizer = make_adam_optimizer(config)
    batch = P75ObjectiveBatch(
        points=points,
        target_values=tf.exp(0.5 * normalized_target),
        weights=weights,
        provenance_label="rung2_calibration_target_bridge",
    )
    trace = []
    for step in range(prefit_steps):
        terms = trainer.square_root_prefit_step(batch, optimizer)
        trace.append({"phase": "sqrt_prefit", "step": step + 1, "loss": float(terms.total_loss.numpy())})
    for step in range(train_steps):
        terms = trainer.train_step(batch, optimizer)
        trace.append({"phase": "density", "step": step + 1, "loss": float(terms.total_loss.numpy())})
    density = trainer.snapshot_density()
    _ = density.normalizer()
    return density, {
        "config": dict(config_payload(config)),
        "calibration_log_normalizer": log_normalizer,
        "initializer": initializer,
        "trace": trace,
        "fit_status": "FIT_OK",
    }


def _quadrature_tt_svd_initial_cores(
    product_basis: ProductBasis,
    points: tf.Tensor,
    weights: tf.Tensor,
    target_sqrt: tf.Tensor,
    *,
    rank_cap: int,
) -> tuple[tuple[tf.Tensor, ...], tuple[int, ...], Mapping[str, object]]:
    """Project a fixed square-root target and compress it by TensorFlow TT-SVD."""

    dimension = product_basis.dimension
    if dimension > len(string.ascii_lowercase):
        raise ValueError("TT-SVD projection dimension exceeds einsum label budget")
    if int(rank_cap) < 1:
        raise ValueError("rank_cap must be positive")
    labels = string.ascii_lowercase[:dimension]
    basis_values = tuple(
        product_basis.evaluate_axis(axis, points[:, axis])
        for axis in range(dimension)
    )
    equation = "n," + ",".join(f"n{label}" for label in labels) + "->" + labels
    coefficients = tf.einsum(
        equation,
        tf.convert_to_tensor(weights, FIT_DTYPE)
        * tf.convert_to_tensor(target_sqrt, FIT_DTYPE),
        *basis_values,
    )
    if not bool(tf.reduce_all(tf.math.is_finite(coefficients)).numpy()):
        raise ValueError("TT-SVD projection coefficients must be finite")

    cores = []
    realized_ranks = [1]
    remainder = coefficients
    left_rank = 1
    discarded_square = tf.constant(0.0, FIT_DTYPE)
    for axis in range(dimension - 1):
        basis_dim = product_basis.bases[axis].basis_dim
        matrix = tf.reshape(remainder, [left_rank * basis_dim, -1])
        singular_values, left_vectors, right_vectors = tf.linalg.svd(
            matrix, full_matrices=False
        )
        realized_rank = min(int(rank_cap), int(singular_values.shape[0]))
        if realized_rank < int(singular_values.shape[0]):
            discarded_square += tf.reduce_sum(
                tf.square(singular_values[realized_rank:])
            )
        cores.append(
            tf.reshape(
                left_vectors[:, :realized_rank],
                [left_rank, basis_dim, realized_rank],
            )
        )
        remainder = (
            singular_values[:realized_rank, tf.newaxis]
            * tf.transpose(right_vectors[:, :realized_rank])
        )
        left_rank = realized_rank
        realized_ranks.append(realized_rank)
    cores.append(
        tf.reshape(
            remainder,
            [left_rank, product_basis.bases[-1].basis_dim, 1],
        )
    )
    realized_ranks.append(1)
    for core in cores:
        if not bool(tf.reduce_all(tf.math.is_finite(core)).numpy()):
            raise ValueError("TT-SVD initializer cores must be finite")
    return tuple(cores), tuple(realized_ranks), {
        "family": "fixed_quadrature_legendre_projection_tt_svd",
        "route_classification": "extension_or_invention",
        "rank_cap": int(rank_cap),
        "realized_ranks": tuple(realized_ranks),
        "coefficient_tensor_shape": tuple(int(value) for value in coefficients.shape),
        "discarded_singular_value_square_sum": float(discarded_square.numpy()),
        "optimizer_role": "l1_aware_density_refinement_after_fixed_projection",
    }


def config_payload(config: P75TrainableTTConfig) -> Mapping[str, object]:
    return {
        "dimension": config.product_basis.dimension,
        "basis_dim_tuple": config.product_basis.basis_dim_tuple(),
        "ranks": config.ranks,
        "tau": float(config.tau.numpy()),
        "l1_weight": float(config.l1_weight.numpy()),
        "l2_weight": float(config.l2_weight.numpy()),
        "learning_rate": config.learning_rate,
        "gradient_clip_norm": config.gradient_clip_norm,
        "seed": config.seed,
        "metadata": dict(config.metadata),
    }


def _fit_candidate(
    model: CoupledNonlinearGaussianModel,
    observations: tf.Tensor,
    spec: FitSpec,
    *,
    calibration_order: int,
    validation_order: int,
    seed: int,
    prefit_steps: int,
    train_steps: int,
) -> FittedBlockProposal:
    transports: list[FixedTTSIRTTransport] = []
    target_rows = []
    previous_density: SquaredTTDensity | None = None
    coordinate_map_2 = _map_for(model, spec.scale, 2)
    for time_index in range(int(observations.shape[0])):
        active_dim = 2 if time_index == 0 else 4
        coordinate_map = _map_for(model, spec.scale, active_dim)
        calibration_points, calibration_weights = _quadrature(active_dim, calibration_order)
        validation_points, validation_weights = _quadrature(active_dim, validation_order)
        calibration_log_target = _reference_target_log(
            model,
            coordinate_map,
            calibration_points,
            observations[time_index],
            time_index,
            previous_density,
        )
        validation_log_target = _reference_target_log(
            model,
            coordinate_map,
            validation_points,
            observations[time_index],
            time_index,
            previous_density,
        )
        density, fit_diag = _fit_density(
            model,
            spec,
            calibration_points,
            calibration_weights,
            calibration_log_target,
            seed=seed + time_index,
            prefit_steps=prefit_steps,
            train_steps=train_steps,
        )
        validation = _validation_kl(density, validation_points, validation_weights, validation_log_target)
        log_normalizer_delta = abs(
            float(fit_diag["calibration_log_normalizer"])
            - float(validation["log_normalizer"])
        )
        target_rows.append(
            {
                "time_index": time_index,
                "target_dimension": active_dim,
                "retained_previous_axes": None
                if time_index == 0
                else (0, 1)
                if previous_density.sqrt_tt.product_basis.dimension == 2
                else (2, 3),
                "validation": validation,
                "calibration_validation_log_normalizer_abs_delta": log_normalizer_delta,
                "fit": fit_diag,
            }
        )
        transports.append(
            FixedTTSIRTTransport(
                density=density,
                cdf_config=KRCDFConfig(
                    grid_size=65,
                    bisection_steps=22,
                    monotonicity_tolerance=1e-12,
                    bracket_tolerance=1e-12,
                    denominator_floor=1e-12,
                    max_floor_count=0,
                ),
            )
        )
        previous_density = density
    return FittedBlockProposal(
        spec=spec,
        coordinate_map=coordinate_map_2,
        transports=tuple(transports),
        diagnostics={
            "target_rows": target_rows,
            "maximum_validation_kl": max(row["validation"]["kl"] for row in target_rows),
            "maximum_calibration_validation_log_normalizer_abs_delta": max(
                row["calibration_validation_log_normalizer_abs_delta"]
                for row in target_rows
            ),
            "audit_data_evaluated": False,
        },
    )


def _audit_candidate(
    model: CoupledNonlinearGaussianModel,
    observations: tf.Tensor,
    candidate: FittedBlockProposal,
    *,
    audit_order: int,
) -> FittedBlockProposal:
    audited_rows = []
    for time_index, (transport, selection_row) in enumerate(
        zip(candidate.transports, candidate.diagnostics["target_rows"])
    ):
        active_dim = 2 if time_index == 0 else 4
        coordinate_map = _map_for(model, candidate.spec.scale, active_dim)
        audit_points, audit_weights = _quadrature(active_dim, audit_order)
        previous_density = (
            None if time_index == 0 else candidate.transports[time_index - 1].density
        )
        audit_log_target = _reference_target_log(
            model,
            coordinate_map,
            audit_points,
            observations[time_index],
            time_index,
            previous_density,
        )
        audit = _validation_kl(
            transport.density, audit_points, audit_weights, audit_log_target
        )
        log_normalizers = (
            float(selection_row["fit"]["calibration_log_normalizer"]),
            float(selection_row["validation"]["log_normalizer"]),
            float(audit["log_normalizer"]),
        )
        audited_rows.append(
            {
                **dict(selection_row),
                "audit": audit,
                "log_normalizer_cross_order_max_abs_delta": (
                    max(log_normalizers) - min(log_normalizers)
                ),
            }
        )
    diagnostics = {
        **dict(candidate.diagnostics),
        "target_rows": audited_rows,
        "maximum_audit_kl": max(row["audit"]["kl"] for row in audited_rows),
        "maximum_log_normalizer_cross_order_abs_delta": max(
            row["log_normalizer_cross_order_max_abs_delta"]
            for row in audited_rows
        ),
        "audit_data_evaluated": True,
        "audit_evaluation_role": "final_frozen_candidate_only",
    }
    return replace(candidate, diagnostics=diagnostics)


def _proposal_randomness(block_count: int, time_steps: int, particle_count: int, seed: int) -> ProposalRandomness:
    return ProposalRandomness(
        initial_uniforms=tf.random.stateless_uniform([block_count, 2, particle_count], [seed, 101], dtype=FIT_DTYPE),
        ancestor_uniforms=tf.random.stateless_uniform([time_steps - 1, particle_count], [seed, 202], dtype=FIT_DTYPE),
        transition_uniforms=tf.random.stateless_uniform([time_steps - 1, block_count, 2, particle_count], [seed, 303], dtype=FIT_DTYPE),
    )


def _standard_normal(uniforms: tf.Tensor, dtype: tf.dtypes.DType) -> tf.Tensor:
    eps = tf.constant(1e-7 if dtype == tf.float32 else 1e-15, dtype)
    clipped = tf.clip_by_value(tf.cast(uniforms, dtype), eps, 1.0 - eps)
    return tf.sqrt(tf.constant(2.0, dtype)) * tf.math.erfinv(2.0 * clipped - 1.0)


def _compile_fitted_block(
    candidate: FittedBlockProposal,
    observations: tf.Tensor,
    randomness: ProposalRandomness,
    block_index: int,
    auxiliary_log: tf.Tensor,
) -> object:
    return compile_fixed_ttsirt_proposal_branch(
        observations=tf.reshape(observations, [-1, 1]),
        initial_transport=candidate.transports[0],
        transition_transports=candidate.transports[1:],
        coordinate_map=candidate.coordinate_map,
        initial_reference_points=randomness.initial_uniforms[block_index],
        ancestor_uniforms=randomness.ancestor_uniforms,
        auxiliary_log_probabilities=auxiliary_log,
        transition_reference_points=randomness.transition_uniforms[:, block_index],
    )


def _compile_fitted_predictive_full(
    candidate: FittedBlockProposal,
    observations: tf.Tensor,
    randomness: ProposalRandomness,
    block_count: int,
) -> PreparedFrozenProposalBranch:
    """Compile a frozen weight-aware predictive auxiliary genealogy."""

    model = CoupledNonlinearGaussianModel(block_count, dtype=FIT_DTYPE)
    theta = tf.zeros([3], FIT_DTYPE)
    particle_count = int(randomness.initial_uniforms.shape[-1])
    observation_matrix = tf.tile(observations[:, tf.newaxis], [1, block_count])

    initial_blocks = []
    initial_log_q_blocks = []
    for block_index in range(block_count):
        initial_local = candidate.transports[0].inverse_transport(
            randomness.initial_uniforms[block_index]
        )
        initial_physical, forward_log_det = candidate.coordinate_map.forward(
            tf.transpose(initial_local)
        )
        initial_blocks.append(initial_physical)
        initial_log_q_blocks.append(
            tf.math.log(candidate.transports[0].eval_pdf(initial_local))
            - forward_log_det
        )
    previous_state = tf.concat(initial_blocks, axis=1)
    initial_log_q = tf.add_n(initial_log_q_blocks)
    initial_log_weights = (
        model.initial_log_density(theta, previous_state)
        + model.observation_log_density(
            theta, previous_state, observation_matrix[0], 0
        )
        - initial_log_q
    )
    previous_log_weights = initial_log_weights - tf.reduce_logsumexp(
        initial_log_weights
    )

    states = [previous_state]
    ancestor_rows = []
    auxiliary_rows = []
    transition_log_q_rows = []
    for time_index, transport in enumerate(candidate.transports[1:], start=1):
        predictive_mean = model.predictive_observation_mean(theta, previous_state)
        predictive_variance = model.predictive_observation_variance()
        residual = observation_matrix[time_index][tf.newaxis, :] - predictive_mean
        predictive_log_likelihood = -0.5 * tf.reduce_sum(
            tf.constant(1.8378770664093453, FIT_DTYPE)
            + tf.math.log(predictive_variance)[tf.newaxis, :]
            + tf.square(residual) / predictive_variance[tf.newaxis, :],
            axis=1,
        )
        auxiliary_log = previous_log_weights + predictive_log_likelihood
        auxiliary_log = auxiliary_log - tf.reduce_logsumexp(auxiliary_log)
        cdf = tf.concat(
            [
                tf.math.cumsum(tf.exp(auxiliary_log))[:-1],
                tf.ones([1], FIT_DTYPE),
            ],
            axis=0,
        )
        ancestor = tf.searchsorted(
            cdf,
            randomness.ancestor_uniforms[time_index - 1],
            side="right",
            out_type=tf.int32,
        )
        parent_state = tf.gather(previous_state, ancestor)

        current_blocks = []
        conditional_log_q_blocks = []
        for block_index in range(block_count):
            parent_block = parent_state[
                :, 2 * block_index : 2 * block_index + 2
            ]
            parent_local, _ = candidate.coordinate_map.inverse(parent_block)
            current_local = transport.conditional_inverse_transport(
                tf.transpose(parent_local),
                randomness.transition_uniforms[
                    time_index - 1, block_index
                ],
            )
            current_physical, current_forward_log_det = (
                candidate.coordinate_map.forward(tf.transpose(current_local))
            )
            current_blocks.append(current_physical)
            conditional_log_q_blocks.append(
                transport.conditional_proposal_log_density(
                    conditioning_points=tf.transpose(parent_local),
                    generated_points=current_local,
                )
                - current_forward_log_det
            )
        current_state = tf.concat(current_blocks, axis=1)
        transition_log_q = tf.add_n(conditional_log_q_blocks)
        reference_log_weights = (
            tf.gather(previous_log_weights, ancestor)
            + model.transition_log_density(
                theta, parent_state, current_state, time_index
            )
            + model.observation_log_density(
                theta,
                current_state,
                observation_matrix[time_index],
                time_index,
            )
            - tf.gather(auxiliary_log, ancestor)
            - transition_log_q
        )
        previous_log_weights = reference_log_weights - tf.reduce_logsumexp(
            reference_log_weights
        )
        previous_state = current_state
        states.append(current_state)
        ancestor_rows.append(ancestor)
        auxiliary_rows.append(auxiliary_log)
        transition_log_q_rows.append(transition_log_q)

    return PreparedFrozenProposalBranch(
        observations=observation_matrix,
        states=tf.stack(states),
        initial_log_proposal_density=initial_log_q,
        ancestors=tf.stack(ancestor_rows),
        auxiliary_log_probabilities=tf.stack(auxiliary_rows),
        transition_log_proposal_density=tf.stack(transition_log_q_rows),
    )


def _inverse_roundtrip_diagnostics(
    candidate: FittedBlockProposal,
    compilations: Sequence[object],
    randomness: ProposalRandomness,
    *,
    maximum_samples_per_block: int,
) -> Mapping[str, object]:
    rows = []
    maximum_error = 0.0
    for block_index, compilation in enumerate(compilations):
        branch = compilation.branch
        sample_count = min(maximum_samples_per_block, branch.particle_count)
        initial_physical = branch.states[0, :sample_count]
        initial_local, _ = candidate.coordinate_map.inverse(initial_physical)
        recovered_initial = candidate.transports[0].forward_transport(
            tf.transpose(initial_local)
        )
        initial_error = float(
            tf.reduce_max(
                tf.abs(
                    recovered_initial
                    - randomness.initial_uniforms[
                        block_index, :, :sample_count
                    ]
                )
            ).numpy()
        )
        transition_errors = []
        for time_index, transport in enumerate(
            candidate.transports[1:], start=1
        ):
            ancestor = branch.ancestors[time_index - 1, :sample_count]
            previous_physical = tf.gather(
                branch.states[time_index - 1], ancestor
            )
            current_physical = branch.states[time_index, :sample_count]
            previous_local, _ = candidate.coordinate_map.inverse(
                previous_physical
            )
            current_local, _ = candidate.coordinate_map.inverse(current_physical)
            joint_local = tf.concat([previous_local, current_local], axis=1)
            recovered_joint = transport.forward_transport(tf.transpose(joint_local))
            error = float(
                tf.reduce_max(
                    tf.abs(
                        recovered_joint[2:]
                        - randomness.transition_uniforms[
                            time_index - 1,
                            block_index,
                            :,
                            :sample_count,
                        ]
                    )
                ).numpy()
            )
            transition_errors.append(error)
        block_maximum = max((initial_error, *transition_errors))
        maximum_error = max(maximum_error, block_maximum)
        rows.append(
            {
                "block_index": block_index,
                "sample_count": sample_count,
                "initial_max_abs_error": initial_error,
                "transition_max_abs_errors": transition_errors,
                "maximum_abs_error": block_maximum,
            }
        )
    return {
        "maximum_abs_error": maximum_error,
        "tolerance": ROUNDTRIP_TOLERANCE,
        "blocks": rows,
    }


def _uniform_auxiliary(time_steps: int, particle_count: int, dtype: tf.dtypes.DType) -> tf.Tensor:
    return tf.fill([time_steps - 1, particle_count], -tf.math.log(tf.cast(particle_count, dtype)))


def _compile_exact_full(
    model: CoupledNonlinearGaussianModel,
    theta: tf.Tensor,
    observations: tf.Tensor,
    randomness: ProposalRandomness,
    auxiliary_mode: str,
) -> PreparedFrozenProposalBranch:
    dtype = model.dtype
    block_model = CoupledNonlinearGaussianModel(1, dtype=dtype)
    block_count = model.block_count
    particle_count = int(randomness.initial_uniforms.shape[-1])
    log_n = tf.math.log(tf.cast(particle_count, dtype))
    initial_states = []
    initial_q = []
    for block in range(block_count):
        previous_mean = tf.constant([block_model.initial_mean_s, block_model.initial_mean_i], dtype)
        previous_var = tf.constant([block_model.initial_variance_s, block_model.initial_variance_i], dtype)
        y0 = tf.cast(observations[0], dtype)
        offset = model.physical_parameters(theta)["observation_offset"]
        pred_var = tf.constant(block_model.initial_variance_i + block_model.observation_variance, dtype)
        gain = tf.constant(block_model.initial_variance_i, dtype) / pred_var
        mean = tf.concat(
            [
                tf.fill([particle_count, 1], previous_mean[0]),
                tf.fill([particle_count, 1], previous_mean[1] + gain * (y0 - previous_mean[1] - offset)),
            ],
            axis=1,
        )
        variance = tf.stack(
            [
                tf.fill([particle_count], previous_var[0]),
                tf.fill([particle_count], previous_var[1] * block_model.observation_variance / pred_var),
            ],
            axis=1,
        )
        noise = tf.transpose(
            _standard_normal(randomness.initial_uniforms[block], dtype)
        )
        state = mean + noise * tf.sqrt(variance)
        initial_states.append(state)
        initial_q.append(_diag_log_density(state, mean, variance))
    states_by_time = [tf.concat(initial_states, axis=1)]
    initial_log_q = tf.add_n(initial_q)
    ancestors = []
    auxiliary_rows = []
    transition_q_rows = []
    current_blocks = initial_states
    for time_index in range(1, int(observations.shape[0])):
        predictions = []
        for block in range(block_count):
            prediction = block_model.transition_mean(theta, current_blocks[block])
            predictions.append(prediction)
        if auxiliary_mode == "predictive":
            log_aux = tf.zeros([particle_count], dtype)
            for prediction in predictions:
                pred_y = prediction[:, 1] + model.physical_parameters(theta)["observation_offset"]
                pred_var = tf.constant(block_model.process_variance_i + block_model.observation_variance, dtype)
                log_aux = log_aux + _diag_log_density(
                    tf.fill([particle_count, 1], tf.cast(observations[time_index], dtype)),
                    pred_y[:, None],
                    tf.fill([particle_count, 1], pred_var),
                )
            log_aux = log_aux - tf.reduce_logsumexp(log_aux)
        else:
            log_aux = tf.fill([particle_count], -log_n)
        cdf = tf.concat([tf.math.cumsum(tf.exp(log_aux))[:-1], tf.ones([1], dtype)], axis=0)
        ancestor = tf.searchsorted(cdf, tf.cast(randomness.ancestor_uniforms[time_index - 1], dtype), side="right", out_type=tf.int32)
        next_blocks = []
        q_blocks = []
        for block in range(block_count):
            selected = tf.gather(predictions[block], ancestor)
            y = tf.cast(observations[time_index], dtype)
            pred_var = tf.constant(block_model.process_variance_i + block_model.observation_variance, dtype)
            gain = tf.constant(block_model.process_variance_i, dtype) / pred_var
            mean = tf.concat(
                [
                    selected[:, 0:1],
                    selected[:, 1:2] + gain * (y - selected[:, 1:2] - model.physical_parameters(theta)["observation_offset"]),
                ],
                axis=1,
            )
            variance = tf.concat(
                [
                    tf.fill([particle_count, 1], tf.constant(block_model.process_variance_s, dtype)),
                    tf.fill([particle_count, 1], tf.constant(block_model.process_variance_i * block_model.observation_variance, dtype) / pred_var),
                ],
                axis=1,
            )
            noise = tf.transpose(
                _standard_normal(
                    randomness.transition_uniforms[time_index - 1, block], dtype
                )
            )
            state = mean + noise * tf.sqrt(variance)
            next_blocks.append(state)
            q_blocks.append(_diag_log_density(state, mean, variance))
        current_blocks = next_blocks
        states_by_time.append(tf.concat(next_blocks, axis=1))
        ancestors.append(ancestor)
        auxiliary_rows.append(log_aux)
        transition_q_rows.append(tf.add_n(q_blocks))
    return PreparedFrozenProposalBranch(
        observations=tf.tile(tf.cast(observations[:, None], dtype), [1, block_count]),
        states=tf.stack(states_by_time),
        initial_log_proposal_density=initial_log_q,
        ancestors=tf.stack(ancestors),
        auxiliary_log_probabilities=tf.stack(auxiliary_rows),
        transition_log_proposal_density=tf.stack(transition_q_rows),
    )


def _diag_log_density(values: tf.Tensor, means: tf.Tensor, variances: tf.Tensor) -> tf.Tensor:
    return -0.5 * tf.reduce_sum(
        tf.constant(1.8378770664093453, values.dtype)
        + tf.math.log(variances)
        + tf.square(values - means) / variances,
        axis=1,
    )


def _cast_fitted_branch(
    candidate: FittedBlockProposal,
    source: PreparedFrozenProposalBranch,
    block_count: int,
) -> tuple[PreparedFrozenProposalBranch, Mapping[str, object]]:
    states = tf.cast(source.states, ONLINE_DTYPE)
    evaluation = tf.cast(states, FIT_DTYPE)
    initial_terms = []
    for block in range(block_count):
        local, log_dz_dx = candidate.coordinate_map.inverse(evaluation[0, :, 2 * block : 2 * block + 2])
        initial_terms.append(tf.math.log(candidate.transports[0].eval_pdf(tf.transpose(local))) + log_dz_dx)
    transition_rows = []
    for time_index, transport in enumerate(candidate.transports[1:], start=1):
        ancestor = source.ancestors[time_index - 1]
        terms = []
        for block in range(block_count):
            previous = tf.gather(evaluation[time_index - 1, :, 2 * block : 2 * block + 2], ancestor)
            current = evaluation[time_index, :, 2 * block : 2 * block + 2]
            previous_local, _ = candidate.coordinate_map.inverse(previous)
            current_local, current_log_dz_dx = candidate.coordinate_map.inverse(current)
            terms.append(
                transport.conditional_proposal_log_density(
                    conditioning_points=tf.transpose(previous_local),
                    generated_points=tf.transpose(current_local),
                )
                + current_log_dz_dx
            )
        transition_rows.append(tf.add_n(terms))
    branch = PreparedFrozenProposalBranch(
        observations=tf.cast(source.observations, ONLINE_DTYPE),
        states=states,
        initial_log_proposal_density=tf.cast(tf.add_n(initial_terms), ONLINE_DTYPE),
        ancestors=source.ancestors,
        auxiliary_log_probabilities=tf.cast(source.auxiliary_log_probabilities, ONLINE_DTYPE),
        transition_log_proposal_density=tf.cast(tf.stack(transition_rows), ONLINE_DTYPE),
    )
    return branch, {"proposal_density_matches_online_state": True}


def _evaluate_arm(model: CoupledNonlinearGaussianModel, branch: PreparedFrozenProposalBranch) -> Mapping[str, object]:
    program = prepare_frozen_proposal_apf_program(model, branch)
    compiled = program.compiled()
    started = time.monotonic()
    result = compiled(tf.constant([0.0, 0.0, 0.0], ONLINE_DTYPE))
    compile_seconds = time.monotonic() - started
    warmed = compiled(tf.constant([0.0, 0.0, 0.0], ONLINE_DTYPE))
    step = tf.constant(1e-3, ONLINE_DTYPE)
    fd = []
    theta = tf.constant([0.0, 0.0, 0.0], ONLINE_DTYPE)
    for index in range(3):
        direction = tf.one_hot(index, 3, dtype=ONLINE_DTYPE)
        fd.append((compiled(theta + step * direction)["log_likelihood"] - compiled(theta - step * direction)["log_likelihood"]) / (2.0 * step))
    finite_difference = tf.stack(fd)
    return {
        "program_id": program.program_id,
        "branch_id": branch.branch_id,
        "log_likelihood": float(result["log_likelihood"].numpy()),
        "score": [float(value) for value in result["score"].numpy()],
        "same_scalar_fd_score": [float(value) for value in finite_difference.numpy()],
        "same_scalar_fd_max_abs_error": float(tf.reduce_max(tf.abs(result["score"] - finite_difference)).numpy()),
        "warmed_repeatability_abs_error": float(tf.abs(result["log_likelihood"] - warmed["log_likelihood"]).numpy()),
        "minimum_ess": float(result["minimum_ess"].numpy()),
        "minimum_ess_fraction": float((result["minimum_ess"] / tf.cast(branch.particle_count, ONLINE_DTYPE)).numpy()),
        "maximum_log_weight_spread": float(result["maximum_log_weight_spread"].numpy()),
        "finite": bool(result["finite"].numpy()),
        "compile_inclusive_seconds": compile_seconds,
        "output_device": result["log_likelihood"].device,
    }


def _conditional_audit(
    model: CoupledNonlinearGaussianModel,
    candidate: FittedBlockProposal,
    branch: PreparedFrozenProposalBranch,
    observations: tf.Tensor,
    block_count: int,
) -> Mapping[str, float]:
    errors = []
    for time_index, transport in enumerate(candidate.transports[1:], start=1):
        ancestor = branch.ancestors[time_index - 1]
        for block in range(block_count):
            previous = tf.gather(branch.states[time_index - 1, :, 2 * block : 2 * block + 2], ancestor)
            current = branch.states[time_index, :, 2 * block : 2 * block + 2]
            model_block = CoupledNonlinearGaussianModel(1, dtype=ONLINE_DTYPE)
            theta = tf.zeros([3], ONLINE_DTYPE)
            exact_log_q = tf.cast(
                model_block.conditional_log_density(
                    theta,
                    previous,
                    current,
                    tf.reshape(observations[time_index], [1]),
                ),
                FIT_DTYPE,
            )
            # Isolate this block's proposal term by recomputing it at the rounded state.
            previous_local, _ = candidate.coordinate_map.inverse(tf.cast(previous, FIT_DTYPE))
            current_local, current_log_dz_dx = candidate.coordinate_map.inverse(tf.cast(current, FIT_DTYPE))
            local_log_q = transport.conditional_proposal_log_density(
                conditioning_points=tf.transpose(previous_local),
                generated_points=tf.transpose(current_local),
            ) + current_log_dz_dx
            errors.append(tf.reshape(local_log_q, [-1]) - exact_log_q)
    values = tf.concat(errors, axis=0)
    abs_values = tf.sort(tf.abs(values))
    q95 = abs_values[tf.cast(tf.floor(0.95 * tf.cast(tf.shape(abs_values)[0] - 1, FIT_DTYPE)), tf.int32)]
    q99 = abs_values[tf.cast(tf.floor(0.99 * tf.cast(tf.shape(abs_values)[0] - 1, FIT_DTYPE)), tf.int32)]
    return {
        "log_density_error_rms": float(tf.sqrt(tf.reduce_mean(tf.square(values))).numpy()),
        "log_density_error_q95": float(q95.numpy()),
        "log_density_error_q99": float(q99.numpy()),
        "log_density_error_max": float(tf.reduce_max(abs_values).numpy()),
    }


def _specs(debug: bool) -> tuple[FitSpec, ...]:
    if debug:
        return (FitSpec(2, 2, 0.2, 0.0),)
    return tuple(
        FitSpec(degree, rank, scale, 0.0)
        for degree, rank, scale in itertools.product((4, 6), (8, 12), (0.2, 0.22))
    )


def _l1_specs(structure: FitSpec, debug: bool) -> tuple[FitSpec, ...]:
    if debug:
        return (replace(structure, l1_weight=0.0),)
    return tuple(
        replace(structure, l1_weight=l1_weight)
        for l1_weight in (0.0, 1e-6, 1e-5)
    )


def _select_l1_candidate(
    candidates: Sequence[FittedBlockProposal],
) -> tuple[FittedBlockProposal, Mapping[str, object]]:
    if not candidates:
        raise ValueError("L1 selection requires at least one candidate")
    zero = [item for item in candidates if item.spec.l1_weight == 0.0]
    positive = [item for item in candidates if item.spec.l1_weight > 0.0]
    if len(zero) != 1:
        raise ValueError("L1 selection requires exactly one zero-L1 comparator")
    zero_candidate = zero[0]
    zero_kl = float(zero_candidate.diagnostics["maximum_validation_kl"])
    if not positive:
        return zero_candidate, {
            "selected_l1_weight": 0.0,
            "zero_l1_validation_kl": zero_kl,
            "best_positive_l1_validation_kl": None,
            "positive_l1_margin": None,
            "selection_reason": "debug_zero_l1_comparator_only",
        }
    best_positive = min(
        positive,
        key=lambda item: (
            float(item.diagnostics["maximum_validation_kl"]),
            item.spec.l1_weight,
        ),
    )
    positive_kl = float(best_positive.diagnostics["maximum_validation_kl"])
    margin = max(0.005, 0.02 * abs(zero_kl))
    selected = (
        best_positive if positive_kl <= zero_kl - margin else zero_candidate
    )
    return selected, {
        "selected_l1_weight": selected.spec.l1_weight,
        "zero_l1_validation_kl": zero_kl,
        "best_positive_l1_validation_kl": positive_kl,
        "positive_l1_margin": margin,
        "positive_l1_required_improvement": margin,
        "positive_l1_observed_improvement": zero_kl - positive_kl,
        "selection_reason": (
            "positive_l1_margin_met"
            if selected.spec.l1_weight > 0.0
            else "zero_l1_retained_margin_not_met"
        ),
    }


def _candidate_summary(candidate: FittedBlockProposal) -> Mapping[str, object]:
    diagnostics = candidate.diagnostics
    return {
        "spec": candidate.spec.payload(),
        "maximum_validation_kl": diagnostics["maximum_validation_kl"],
        "maximum_calibration_validation_log_normalizer_abs_delta": diagnostics[
            "maximum_calibration_validation_log_normalizer_abs_delta"
        ],
        "audit_data_evaluated": diagnostics.get("audit_data_evaluated", False),
    }


def _git_payload() -> Mapping[str, object]:
    commit = subprocess.run(("git", "rev-parse", "HEAD"), check=True, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(("git", "status", "--short"), check=True, capture_output=True, text=True).stdout.splitlines()
    return {"commit": commit, "dirty": bool(dirty), "dirty_line_count": len(dirty)}


def _jsonable(value: object) -> object:
    if isinstance(value, tf.Tensor):
        materialized = value.numpy()
        return materialized.item() if value.shape.rank == 0 else materialized.tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _write_failure_receipt(error: Exception) -> None:
    try:
        output_root = Path(sys.argv[sys.argv.index("--output-root") + 1])
    except (ValueError, IndexError):
        return
    if not output_root.is_dir() or (output_root / "result.json").exists():
        return
    payload = {
        "schema": ARTIFACT_SCHEMA + ".failure",
        "status": "FAIL_INFRASTRUCTURE_OR_HARNESS",
        "failure_classification": "not_candidate_evidence",
        "exception_type": type(error).__name__,
        "exception_message": str(error),
        "traceback": traceback.format_exc(),
        "command": " ".join(sys.argv),
        "git": _git_payload(),
        "nonclaims": ["no candidate rejection", "no research-direction rejection"],
    }
    (output_root / "failure_result.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cpu-reference", action="store_true")
    parser.add_argument("--debug-smoke", action="store_true")
    args = parser.parse_args()
    scope = DEBUG_SCOPE if args.debug_smoke else CANONICAL_SCOPE
    args.output_root.mkdir(parents=True, exist_ok=False)
    started_at = datetime.now(timezone.utc)
    started = time.monotonic()
    block_count = scope[0] // 2
    time_steps = scope[1]
    particle_count = scope[2]
    observations = _observations(time_steps)
    fit_model = CoupledNonlinearGaussianModel(1, dtype=FIT_DTYPE)
    online_model = CoupledNonlinearGaussianModel(block_count, dtype=ONLINE_DTYPE)
    randomness = _proposal_randomness(block_count, time_steps, particle_count, 220724)
    stage_a_specs = _specs(args.debug_smoke)
    stage_a_results = []
    with tf.device("/CPU:0"):
        for spec in stage_a_specs:
            stage_a_results.append(
                _fit_candidate(
                    fit_model,
                    observations,
                    spec,
                    calibration_order=3 if args.debug_smoke else 8,
                    validation_order=4 if args.debug_smoke else 9,
                    seed=220724,
                    prefit_steps=0,
                    train_steps=2 if args.debug_smoke else 32,
                )
            )
    selected_structure = min(
        stage_a_results,
        key=lambda item: float(item.diagnostics["maximum_validation_kl"]),
    )
    stage_b_specs = _l1_specs(selected_structure.spec, args.debug_smoke)
    stage_b_results = []
    with tf.device("/CPU:0"):
        for spec in stage_b_specs:
            stage_b_results.append(
                _fit_candidate(
                    fit_model,
                    observations,
                    spec,
                    calibration_order=3 if args.debug_smoke else 8,
                    validation_order=4 if args.debug_smoke else 9,
                    seed=220724,
                    prefit_steps=0,
                    train_steps=2 if args.debug_smoke else 32,
                )
            )
    selected_pre_audit, l1_selection = _select_l1_candidate(stage_b_results)
    selected = _audit_candidate(
        fit_model,
        observations,
        selected_pre_audit,
        audit_order=5 if args.debug_smoke else 10,
    )
    auxiliary = _uniform_auxiliary(time_steps, particle_count, FIT_DTYPE)
    with tf.device("/CPU:0"):
        block_compilations = [
            _compile_fitted_block(selected, observations, randomness, block, auxiliary)
            for block in range(block_count)
        ]
        inverse_roundtrip = _inverse_roundtrip_diagnostics(
            selected,
            block_compilations,
            randomness,
            maximum_samples_per_block=4 if args.debug_smoke else 8,
        )
        combined_compilation = combine_fixed_ttsirt_block_compilations(
            block_compilations, observation_mode="concatenate"
        )
        fitted_uniform_branch, cast_diag = _cast_fitted_branch(
            selected, combined_compilation.branch, block_count
        )
        fitted_predictive_source = _compile_fitted_predictive_full(
            selected, observations, randomness, block_count
        )
        fitted_predictive_branch, predictive_cast_diag = _cast_fitted_branch(
            selected, fitted_predictive_source, block_count
        )
        exact_uniform = _compile_exact_full(online_model, tf.zeros([3], ONLINE_DTYPE), tf.cast(observations, ONLINE_DTYPE), randomness, "uniform")
        exact_predictive = _compile_exact_full(online_model, tf.zeros([3], ONLINE_DTYPE), tf.cast(observations, ONLINE_DTYPE), randomness, "predictive")
    with tf.device(str(DEVICE["online_device"])):
        arms = {
            "exact_predictive_auxiliary": _evaluate_arm(online_model, exact_predictive),
            "exact_uniform_auxiliary": _evaluate_arm(online_model, exact_uniform),
            "fitted_ttsirt_uniform_auxiliary": _evaluate_arm(
                online_model, fitted_uniform_branch
            ),
            "fitted_ttsirt_predictive_auxiliary": _evaluate_arm(
                online_model, fitted_predictive_branch
            ),
        }
    conditional_audit = _conditional_audit(
        online_model,
        selected,
        fitted_predictive_branch,
        tf.cast(observations, ONLINE_DTYPE),
        block_count,
    )
    all_fit_results = tuple(stage_a_results) + tuple(stage_b_results)
    harness_gates = {
        "all_fit_candidates_finite": all(
            math.isfinite(float(row.diagnostics["maximum_validation_kl"]))
            and math.isfinite(
                float(
                    row.diagnostics[
                        "maximum_calibration_validation_log_normalizer_abs_delta"
                    ]
                )
            )
            for row in all_fit_results
        ),
        "selected_audit_data_frozen_only": selected.diagnostics["audit_data_evaluated"]
        and selected.diagnostics.get("audit_evaluation_role")
        == "final_frozen_candidate_only",
        "l1_selection_recorded": bool(l1_selection["selection_reason"]),
        "selected_validation_kl_finite": math.isfinite(float(selected.diagnostics["maximum_validation_kl"])),
        "selected_audit_kl_finite": math.isfinite(float(selected.diagnostics["maximum_audit_kl"])),
        "candidate_finite": arms["fitted_ttsirt_predictive_auxiliary"]["finite"],
        "candidate_score_fd_le_0p05": arms["fitted_ttsirt_predictive_auxiliary"]["same_scalar_fd_max_abs_error"] <= SCORE_FD_TOLERANCE,
        "candidate_warm_repeatability_le_1e_5": arms["fitted_ttsirt_predictive_auxiliary"]["warmed_repeatability_abs_error"] <= 1e-5,
        "inverse_roundtrip_max_abs_error_le_1e_4": inverse_roundtrip[
            "maximum_abs_error"
        ] <= ROUNDTRIP_TOLERANCE,
        "expected_gpu_or_cpu_reference_device": ("CPU" in arms["fitted_ttsirt_predictive_auxiliary"]["output_device"] if args.cpu_reference else "GPU" in arms["fitted_ttsirt_predictive_auxiliary"]["output_device"]),
        "memory_growth_verified": args.cpu_reference or DEVICE["gpu_memory_policy"]["all_physical_devices_memory_growth"],
    }
    candidate_gates = {
        "selected_log_normalizer_cross_order_delta_le_0p05": selected.diagnostics[
            "maximum_log_normalizer_cross_order_abs_delta"
        ] <= TARGET_LOG_NORMALIZER_TOLERANCE,
        "conditional_log_density_rms_le_0p75": conditional_audit[
            "log_density_error_rms"
        ] <= CONDITIONAL_LOG_RMS_TOLERANCE,
        "candidate_ess_fraction_ge_0p20": arms[
            "fitted_ttsirt_predictive_auxiliary"
        ]["minimum_ess_fraction"] >= ESS_FRACTION_TOLERANCE,
    }
    gates = {**harness_gates, **candidate_gates}
    harness_passed = all(harness_gates.values())
    candidate_screen_passed = all(candidate_gates.values())
    passed = harness_passed and (args.debug_smoke or candidate_screen_passed)
    status = "PASS_DEBUG_SMOKE" if args.debug_smoke and passed else "PASS_CPU_REFERENCE_PRECHECK" if args.cpu_reference and passed else "PASS_ENGINEERING_RUNG2" if passed else "BLOCK_CANDIDATE_CONTINUE_RESEARCH_RUNG2"
    memory = {"current": 0, "peak": 0} if args.cpu_reference else tf.config.experimental.get_memory_info("GPU:0")
    payload = _jsonable({
        "schema": ARTIFACT_SCHEMA,
        "status": status,
        "execution_role": "debug_smoke" if args.debug_smoke else "cpu_reference_precheck" if args.cpu_reference else "trusted_gpu_claim",
        "scope": {"dimension": scope[0], "time_steps": scope[1], "particle_count": scope[2], "block_count": block_count},
        "route_classification": "extension_or_invention",
        "model": online_model.manifest_payload(),
        "selected_candidate": {"spec": selected.spec.payload(), "diagnostics": selected.diagnostics},
        "stage_a_candidates": [_candidate_summary(item) for item in stage_a_results],
        "stage_b_candidates": [_candidate_summary(item) for item in stage_b_results],
        "l1_selection": l1_selection,
        "candidate_count": len(all_fit_results),
        "arms": arms,
        "conditional_density_audit": conditional_audit,
        "proposal_cast_diagnostics": {
            "uniform": cast_diag,
            "predictive": predictive_cast_diag,
        },
        "inverse_roundtrip": inverse_roundtrip,
        "gates": gates,
        "gate_roles": {
            "harness_and_correctness": tuple(harness_gates),
            "candidate_screen": tuple(candidate_gates),
            "debug_exit_uses_harness_only": True,
        },
        "harness_passed": harness_passed,
        "candidate_screen_passed": candidate_screen_passed,
        "decision": {
            "candidate_rejected": (not passed) if not args.debug_smoke and not args.cpu_reference else None,
            "research_direction_rejected": False,
            "primary_criterion_status": "passed" if not args.debug_smoke and not args.cpu_reference and passed else "not_assessed_nonclaiming_execution",
            "next_justified_action": (
                "run canonical tuning ladder"
                if args.debug_smoke and harness_passed
                else "repair candidate or tune a fresh scope"
                if not candidate_screen_passed
                else "coupled nonlinear multi-seed/rank rung"
            ),
            "not_concluded": "no source-faithful Zhao-Cui, Austria SIR, NAWM, HMC, production, or default-readiness claim",
        },
        "device": DEVICE,
        "run_manifest": {
            "git": _git_payload(),
            "command": " ".join(sys.argv),
            "environment": sys.executable,
            "python_version": platform.python_version(),
            "tensorflow_version": tf.__version__,
            "fit_dtype": FIT_DTYPE.name,
            "online_dtype": ONLINE_DTYPE.name,
            "tf32_enabled": DEVICE["tf32_enabled"],
            "jit_compile": True,
            "gpu_memory_policy": DEVICE["gpu_memory_policy"],
            "gpu_allocator_current_bytes": int(memory["current"]),
            "gpu_allocator_peak_bytes": int(memory["peak"]),
            "random_seeds": [220724],
            "wall_time_seconds": time.monotonic() - started,
            "started_at_utc": started_at.isoformat(),
            "plan_file": PLAN_PATH,
            "trust_basis": "explicit_cpu_reference" if args.cpu_reference else "owner_designated_managed_session_visible_gpu_trusted",
        },
        "nonclaims": [
            "no source-faithful Zhao-Cui claim",
            "no exact randomized-estimator or pseudo-marginal claim",
            "no Austria SIR or NAWM claim",
            "no posterior correctness or HMC convergence claim",
            "no production KR closure or default-readiness claim",
            "no statistical ranking or superiority claim",
        ],
    })
    (args.output_root / "result.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_root / "result.md").write_text(
        "# Zhao-Cui Coupled Nonlinear TTSIRT-APF Rung-2 Result\n\n"
        f"Status: `{status}`\n\n"
        f"Selected candidate: `{selected.spec.payload()}`\n\n"
        f"Candidate ESS fraction: `{arms['fitted_ttsirt_predictive_auxiliary']['minimum_ess_fraction']}`\n\n"
        f"Candidate score/FD error: `{arms['fitted_ttsirt_predictive_auxiliary']['same_scalar_fd_max_abs_error']}`\n\n"
        "This is an extension/invention diagnostic and carries no source-faithful, HMC, NAWM, or production claim.\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, sort_keys=True))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        _write_failure_receipt(error)
        raise
