#!/usr/bin/env python3
"""Fit and evaluate the synthetic 24D blockwise TTSIRT-APF rung."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import itertools
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
from typing import Mapping


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
        physical = tf.config.list_physical_devices("GPU")
        if physical:
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
        raise RuntimeError("trusted rung 1 requires a logical GPU")
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
from bayesfilter.highdim.fitting import (  # noqa: E402
    FixedTTFitConfig,
    FixedTTFitSampleBatch,
    FixedTTFitter,
)
from bayesfilter.highdim.squared_tt import (  # noqa: E402
    SquaredTTDensity,
    TensorProductReferenceDensity,
)
from bayesfilter.highdim.transport import FixedTTSIRTTransport, KRCDFConfig  # noqa: E402
from bayesfilter.highdim.zhao_cui_fixed_adjacent_tt_tf import (  # noqa: E402
    norm_balanced_initial_cores,
)
from bayesfilter.highdim.zhao_cui_frozen_proposal_apf_tf import (  # noqa: E402
    AlgebraicCoordinateMap,
    MEASURE_ID,
    ROUTE_CLASSIFICATION,
    SCORE_BACKEND_ID,
    PreparedFrozenProposalBranch,
    combine_fixed_ttsirt_block_compilations,
    compile_fixed_ttsirt_proposal_branch,
    prepare_frozen_proposal_apf_program,
    prepare_frozen_proposal_branch,
)
FIT_DTYPE = tf.float64
ONLINE_DTYPE = tf.float32
LOG_TWO_PI = math.log(2.0 * math.pi)
PLAN_PATH = (
    "docs/plans/"
    "bayesfilter-zhao-cui-blockwise-ttsirt-apf-rung1-plan-2026-07-22.md"
)
SCHEMA = "bayesfilter.zhao_cui_blockwise_ttsirt_apf.rung1.v1"
CALIBRATION_ORDER = 25
HOLDOUT_ORDER = 32
CDF_GRID_SIZE = 129
CDF_BISECTION_STEPS = 24
RIDGE = 1e-10
FIT_SWEEPS = 2
TARGET_MASS_TOLERANCE = 3e-2
CONDITIONAL_TIEOUT_TOLERANCE = 1e-10
ROUNDOFF_TOLERANCE = 1e-4
SCORE_FD_TOLERANCE = 3e-2
ESS_FRACTION_TOLERANCE = 0.5
CANONICAL_SCOPE = (24, 3, 256)


@dataclass(frozen=True)
class ScalarGaussianTargets:
    """Exact normalized scalar filtering and adjacent Gaussian targets."""

    observations: tf.Tensor
    theta: tf.Tensor
    prior_variance: float
    transition_variance: float
    observation_variance: float
    transition_scale: float
    posterior_means: tuple[tf.Tensor, ...]
    posterior_variances: tuple[tf.Tensor, ...]
    predictive_log_increments: tuple[tf.Tensor, ...]

    @classmethod
    def build(
        cls,
        observations: tf.Tensor,
        theta: tf.Tensor,
        *,
        prior_variance: float,
        transition_variance: float,
        observation_variance: float,
        transition_scale: float,
    ) -> "ScalarGaussianTargets":
        values = tf.reshape(tf.convert_to_tensor(observations, FIT_DTYPE), [-1])
        parameters = tf.reshape(tf.convert_to_tensor(theta, FIT_DTYPE), [2])
        means = []
        variances = []
        increments = []
        mean = parameters[0]
        variance = tf.cast(prior_variance, FIT_DTYPE)
        for time_index in range(int(values.shape[0])):
            if time_index > 0:
                mean = transition_scale * mean + parameters[0]
                variance = transition_scale**2 * variance + transition_variance
            innovation_variance = variance + observation_variance
            increments.append(
                _scalar_normal_log_density(
                    values[time_index], mean + parameters[1], innovation_variance
                )
            )
            gain = variance / innovation_variance
            mean = mean + gain * (values[time_index] - parameters[1] - mean)
            variance = (1.0 - gain) * variance
            means.append(mean)
            variances.append(variance)
        return cls(
            observations=values,
            theta=parameters,
            prior_variance=float(prior_variance),
            transition_variance=float(transition_variance),
            observation_variance=float(observation_variance),
            transition_scale=float(transition_scale),
            posterior_means=tuple(means),
            posterior_variances=tuple(variances),
            predictive_log_increments=tuple(increments),
        )

    @property
    def time_steps(self) -> int:
        return int(self.observations.shape[0])

    def initial_log_density(self, physical_points: tf.Tensor) -> tf.Tensor:
        current = tf.reshape(tf.convert_to_tensor(physical_points, FIT_DTYPE), [-1])
        return _normal_log_density_1d(
            current, self.posterior_means[0], self.posterior_variances[0]
        )

    def adjacent_log_density(
        self, time_index: int, physical_points: tf.Tensor
    ) -> tf.Tensor:
        if time_index < 1 or time_index >= self.time_steps:
            raise IndexError("adjacent target time index is out of range")
        points = tf.convert_to_tensor(physical_points, FIT_DTYPE)
        previous = points[:, 0]
        current = points[:, 1]
        transition_mean = self.transition_scale * previous + self.theta[0]
        return (
            _normal_log_density_1d(
                previous,
                self.posterior_means[time_index - 1],
                self.posterior_variances[time_index - 1],
            )
            + _normal_log_density_1d(
                current, transition_mean, self.transition_variance
            )
            + _normal_log_density_1d(
                self.observations[time_index],
                current + self.theta[1],
                self.observation_variance,
            )
            - self.predictive_log_increments[time_index]
        )


@dataclass(frozen=True)
class CandidateSpec:
    degree: int
    rank: int
    scale: float
    defensive_mass: float

    def selection_key(self, maximum_holdout_residual: float) -> tuple[float, ...]:
        return (
            float(maximum_holdout_residual),
            float(self.degree),
            float(self.rank),
            float(self.defensive_mass),
            float(self.scale),
        )

    def payload(self) -> Mapping[str, object]:
        return {
            "degree": self.degree,
            "adjacent_rank": self.rank,
            "algebraic_scale": self.scale,
            "defensive_mass": self.defensive_mass,
        }


@dataclass(frozen=True)
class FittedCandidate:
    spec: CandidateSpec
    transports: tuple[FixedTTSIRTTransport, ...]
    diagnostics: Mapping[str, object]


@dataclass(frozen=True)
class ProposalRandomness:
    initial_uniforms: tf.Tensor
    ancestor_uniforms: tf.Tensor
    transition_uniforms: tf.Tensor


def _proposal_randomness(
    *,
    dimension: int,
    time_steps: int,
    particle_count: int,
    seed: int,
) -> ProposalRandomness:
    return ProposalRandomness(
        initial_uniforms=tf.random.stateless_uniform(
            [dimension, particle_count], [seed, 101], dtype=FIT_DTYPE
        ),
        ancestor_uniforms=tf.random.stateless_uniform(
            [time_steps - 1, particle_count], [seed, 202], dtype=FIT_DTYPE
        ),
        transition_uniforms=tf.random.stateless_uniform(
            [time_steps - 1, dimension, particle_count],
            [seed, 303],
            dtype=FIT_DTYPE,
        ),
    )


@dataclass(frozen=True)
class DiagonalGaussianModel:
    """Diagonal LGSSM with analytical parameter-only density scores."""

    dimension: int
    dtype: tf.dtypes.DType = ONLINE_DTYPE
    prior_variance: float = 1.25
    transition_variance: float = 0.7
    observation_variance: float = 0.8
    transition_scale: float = 0.65

    def parameter_dim(self) -> int:
        return 2

    def state_dim(self) -> int:
        return self.dimension

    def observation_dim(self) -> int:
        return self.dimension

    def frozen_apf_measure_id(self) -> str:
        return MEASURE_ID

    def frozen_apf_score_backend_id(self) -> str:
        return SCORE_BACKEND_ID

    def initial_log_density(self, theta: tf.Tensor, state: tf.Tensor) -> tf.Tensor:
        return _normal_log_density_nd(state, theta[0], self.prior_variance, self.dtype)

    def transition_log_density(
        self,
        theta: tf.Tensor,
        previous: tf.Tensor,
        current: tf.Tensor,
        time_index: int,
    ) -> tf.Tensor:
        del time_index
        mean = self.transition_scale * previous + theta[0]
        return _normal_log_density_nd(current, mean, self.transition_variance, self.dtype)

    def observation_log_density(
        self,
        theta: tf.Tensor,
        state: tf.Tensor,
        observation: tf.Tensor,
        time_index: int,
    ) -> tf.Tensor:
        del time_index
        return _normal_log_density_nd(
            observation[None, :], state + theta[1], self.observation_variance, self.dtype
        )

    def initial_log_density_parameter_score(
        self, theta: tf.Tensor, state: tf.Tensor
    ) -> tf.Tensor:
        component = tf.reduce_sum(state - theta[0], axis=1) / self.prior_variance
        return tf.stack([component, tf.zeros_like(component)], axis=1)

    def transition_log_density_parameter_score(
        self,
        theta: tf.Tensor,
        previous: tf.Tensor,
        current: tf.Tensor,
        time_index: int,
    ) -> tf.Tensor:
        del time_index
        residual = current - (self.transition_scale * previous + theta[0])
        component = tf.reduce_sum(residual, axis=1) / self.transition_variance
        return tf.stack([component, tf.zeros_like(component)], axis=1)

    def observation_log_density_parameter_score(
        self,
        theta: tf.Tensor,
        state: tf.Tensor,
        observation: tf.Tensor,
        time_index: int,
    ) -> tf.Tensor:
        del time_index
        residual = observation[None, :] - (state + theta[1])
        component = tf.reduce_sum(residual, axis=1) / self.observation_variance
        return tf.stack([tf.zeros_like(component), component], axis=1)

    def manifest_payload(self) -> Mapping[str, object]:
        return {
            "family": "synthetic_independent_diagonal_gaussian_lgssm",
            "dimension": self.dimension,
            "prior_variance": self.prior_variance,
            "transition_variance": self.transition_variance,
            "observation_variance": self.observation_variance,
            "transition_scale": self.transition_scale,
            "dtype": self.dtype.name,
        }


def _scalar_observations(time_steps: int) -> tf.Tensor:
    base = tf.constant([-0.24, 0.31, -0.08], FIT_DTYPE)
    if time_steps > int(base.shape[0]):
        raise ValueError("rung 1 supports at most three predeclared observations")
    return base[:time_steps]


def _convention() -> MeasureConvention:
    return MeasureConvention(
        density_measure=DensityMeasure.REFERENCE_MEASURE,
        mass_measure=MassMeasure.REFERENCE_MEASURE,
        reference_weight_name="uniform_probability_on_minus_one_one",
    )


def _basis(dimension: int, degree: int) -> ProductBasis:
    return ProductBasis(
        [
            LegendreBasis1D(BoundedInterval(-1.0, 1.0), degree)
            for _ in range(dimension)
        ],
        _convention(),
    )


def _quadrature(dimension: int, order: int) -> tuple[tf.Tensor, tf.Tensor]:
    nodes, weights = legendre_gauss_nodes_weights(order)
    axis_nodes = [nodes for _ in range(dimension)]
    axis_weights = [0.5 * weights for _ in range(dimension)]
    node_mesh = tf.meshgrid(*axis_nodes, indexing="ij")
    weight_mesh = tf.meshgrid(*axis_weights, indexing="ij")
    points = tf.stack([tf.reshape(value, [-1]) for value in node_mesh], axis=1)
    product_weights = tf.ones([tf.shape(points)[0]], FIT_DTYPE)
    for value in weight_mesh:
        product_weights = product_weights * tf.reshape(value, [-1])
    return points, product_weights


def _midpoint_holdout(dimension: int, order: int) -> tuple[tf.Tensor, tf.Tensor]:
    indices = tf.cast(tf.range(order), FIT_DTYPE)
    nodes = -1.0 + 2.0 * (indices + 0.5) / tf.cast(order, FIT_DTYPE)
    weights = tf.fill([order], tf.math.reciprocal(tf.cast(order, FIT_DTYPE)))
    node_mesh = tf.meshgrid(*[nodes for _ in range(dimension)], indexing="ij")
    weight_mesh = tf.meshgrid(*[weights for _ in range(dimension)], indexing="ij")
    points = tf.stack([tf.reshape(value, [-1]) for value in node_mesh], axis=1)
    product_weights = tf.ones([tf.shape(points)[0]], FIT_DTYPE)
    for value in weight_mesh:
        product_weights = product_weights * tf.reshape(value, [-1])
    return points, product_weights


def _reference_log_target(
    targets: ScalarGaussianTargets,
    coordinate_map: AlgebraicCoordinateMap,
    points: tf.Tensor,
    *,
    time_index: int,
) -> tf.Tensor:
    physical, log_abs_det = coordinate_map.forward(points)
    physical_log_density = (
        targets.initial_log_density(physical)
        if time_index == 0
        else targets.adjacent_log_density(time_index, physical)
    )
    log_reference_density = -tf.cast(int(points.shape[1]), FIT_DTYPE) * tf.math.log(
        tf.constant(2.0, FIT_DTYPE)
    )
    return physical_log_density + log_abs_det - log_reference_density


def _fit_candidate(
    spec: CandidateSpec,
    targets: ScalarGaussianTargets,
) -> FittedCandidate:
    transports = []
    target_rows = []
    for time_index in range(targets.time_steps):
        active_dimension = 1 if time_index == 0 else 2
        product_basis = _basis(active_dimension, spec.degree)
        coordinate_map = AlgebraicCoordinateMap(
            tf.fill([active_dimension], tf.cast(spec.scale, FIT_DTYPE))
        )
        fit_points, fit_weights = _quadrature(active_dimension, CALIBRATION_ORDER)
        holdout_points, holdout_weights = _midpoint_holdout(
            active_dimension, HOLDOUT_ORDER
        )
        fit_log_target = _reference_log_target(
            targets, coordinate_map, fit_points, time_index=time_index
        )
        holdout_log_target = _reference_log_target(
            targets, coordinate_map, holdout_points, time_index=time_index
        )
        fit_config = FixedTTFitConfig(
            ranks=(1, 1) if active_dimension == 1 else (1, spec.rank, 1),
            ridge=RIDGE,
            max_sweeps=FIT_SWEEPS,
            sweep_order=(0,) if active_dimension == 1 else (0, 1, 1, 0),
            row_budget=int(fit_points.shape[0]),
            column_budget=max((spec.degree + 1) * spec.rank**2, spec.degree + 1),
            dense_matrix_byte_budget=128_000_000,
            normal_matrix_byte_budget=16_000_000,
            condition_number_warning=1e12,
            condition_number_veto=1e16,
            holdout_tolerance=1e6,
        )
        ranks = fit_config.ranks
        fit_result = FixedTTFitter().fit(
            product_basis=product_basis,
            samples=FixedTTFitSampleBatch(
                points=fit_points,
                target_values=tf.exp(0.5 * fit_log_target),
                weights=fit_weights,
                holdout_points=holdout_points,
                holdout_values=tf.exp(0.5 * holdout_log_target),
                holdout_weights=holdout_weights,
            ),
            config=fit_config,
            initial_cores=norm_balanced_initial_cores(product_basis, ranks),
            branch_seed=(
                f"zhao-cui-rung1-d{spec.degree}-r{spec.rank}-s{spec.scale}-"
                f"tau{spec.defensive_mass}-t{time_index}"
            ),
            measure_convention=_convention(),
            initialization_rule="orthonormal_mode_diagonal_norm_balanced_v1",
        )
        if fit_result.status is not HighDimStatus.OK:
            raise ValueError(f"fit_t{time_index}_{fit_result.status.value}")
        density = _density_from_fit(fit_result, product_basis, spec.defensive_mass)
        fitted_sqrt = tf.sqrt(tf.exp(density.log_density(holdout_points)))
        target_sqrt = tf.exp(0.5 * holdout_log_target)
        relative_holdout = _relative_rms(
            fitted_sqrt, target_sqrt, holdout_weights
        )
        target_mass = tf.reduce_sum(fit_weights * tf.exp(fit_log_target))
        maximum_condition = max(
            float(update["condition_number"])
            for update in fit_result.core_update_statuses
        )
        target_rows.append(
            {
                "time_index": time_index,
                "target_dimension": active_dimension,
                "axis_order": ("x_0",)
                if time_index == 0
                else ("x_previous", "x_current"),
                "fit_status": fit_result.status.value,
                "fit_branch_hash": fit_result.branch_hash.value,
                "raw_fit_sqrt_rms": float(fit_result.fit_residual.numpy()),
                "raw_holdout_sqrt_rms": float(fit_result.holdout_residual.numpy()),
                "full_proposal_relative_holdout_sqrt_rms": float(
                    relative_holdout.numpy()
                ),
                "reference_target_quadrature_mass": float(target_mass.numpy()),
                "reference_target_quadrature_mass_abs_error": float(
                    tf.abs(target_mass - 1.0).numpy()
                ),
                "maximum_scaled_augmented_condition_number": maximum_condition,
                "rank_tuple": ranks,
            }
        )
        transports.append(
            FixedTTSIRTTransport(
                density=density,
                cdf_config=KRCDFConfig(
                    grid_size=CDF_GRID_SIZE,
                    bisection_steps=CDF_BISECTION_STEPS,
                    monotonicity_tolerance=1e-12,
                    bracket_tolerance=1e-12,
                    denominator_floor=1e-12,
                    max_floor_count=0,
                ),
            )
        )
    return FittedCandidate(
        spec=spec,
        transports=tuple(transports),
        diagnostics={
            "targets": target_rows,
            "maximum_full_proposal_relative_holdout_sqrt_rms": max(
                row["full_proposal_relative_holdout_sqrt_rms"] for row in target_rows
            ),
            "maximum_reference_target_quadrature_mass_abs_error": max(
                row["reference_target_quadrature_mass_abs_error"] for row in target_rows
            ),
        },
    )


def _density_from_fit(
    fit_result: object,
    product_basis: ProductBasis,
    defensive_mass: float,
) -> SquaredTTDensity:
    defensive = TensorProductReferenceDensity(product_basis, _convention())
    tau = tf.constant(defensive_mass, FIT_DTYPE)
    floor = tf.constant(1e-12, FIT_DTYPE)
    identity = SquaredTTDensity.expected_branch_identity(
        sqrt_tt=fit_result.fitted_tt,
        defensive_density=defensive,
        tau=tau,
        normalizer_floor=floor,
        denominator_floor=floor,
        measure_convention=_convention(),
    )
    return SquaredTTDensity(
        sqrt_tt=fit_result.fitted_tt,
        defensive_density=defensive,
        tau=tau,
        normalizer_floor=floor,
        denominator_floor=floor,
        measure_convention=_convention(),
        branch_identity=identity,
    )


def _select_candidate(
    targets: ScalarGaussianTargets,
) -> tuple[FittedCandidate, list[Mapping[str, object]]]:
    records = []
    successful = []
    for degree, rank, scale, defensive_mass in itertools.product(
        (6, 10), (2, 4), (1.5, 2.5), (1e-6, 1e-4)
    ):
        spec = CandidateSpec(degree, rank, scale, defensive_mass)
        started = time.monotonic()
        try:
            candidate = _fit_candidate(spec, targets)
            maximum_residual = float(
                candidate.diagnostics[
                    "maximum_full_proposal_relative_holdout_sqrt_rms"
                ]
            )
            record = {
                **spec.payload(),
                "status": "FIT_OK",
                "selection_metric": maximum_residual,
                "fit_seconds": time.monotonic() - started,
                "diagnostics": candidate.diagnostics,
            }
            successful.append((spec.selection_key(maximum_residual), candidate))
        except (ValueError, tf.errors.OpError) as error:
            record = {
                **spec.payload(),
                "status": "FIT_REJECTED",
                "reason": f"{type(error).__name__}: {error}",
                "fit_seconds": time.monotonic() - started,
            }
        records.append(record)
    if not successful:
        raise RuntimeError("all 16 TTSIRT fit candidates were rejected")
    minimum_metric = min(item[0][0] for item in successful)
    tied = [item for item in successful if item[0][0] <= minimum_metric + 1e-6]
    tied.sort(key=lambda item: item[0][1:])
    return tied[0][1], records


def _compile_ttsirt_candidate(
    candidate: FittedCandidate,
    scalar_observations: tf.Tensor,
    *,
    dimension: int,
    particle_count: int,
    randomness: ProposalRandomness,
):
    time_steps = int(scalar_observations.shape[0])
    if randomness.initial_uniforms.shape != (dimension, particle_count):
        raise ValueError("initial proposal randomness has the wrong shape")
    if randomness.ancestor_uniforms.shape != (time_steps - 1, particle_count):
        raise ValueError("ancestor proposal randomness has the wrong shape")
    if randomness.transition_uniforms.shape != (
        time_steps - 1,
        dimension,
        particle_count,
    ):
        raise ValueError("transition proposal randomness has the wrong shape")
    log_particle_count = tf.math.log(tf.cast(particle_count, FIT_DTYPE))
    auxiliary = tf.fill(
        [time_steps - 1, particle_count], -log_particle_count
    )
    coordinate_map = AlgebraicCoordinateMap(
        tf.constant([candidate.spec.scale], FIT_DTYPE)
    )
    blocks = []
    for block_index in range(dimension):
        blocks.append(
            compile_fixed_ttsirt_proposal_branch(
                observations=scalar_observations[:, None],
                initial_transport=candidate.transports[0],
                transition_transports=candidate.transports[1:],
                coordinate_map=coordinate_map,
                initial_reference_points=randomness.initial_uniforms[
                    block_index : block_index + 1
                ],
                ancestor_uniforms=randomness.ancestor_uniforms,
                auxiliary_log_probabilities=auxiliary,
                transition_reference_points=randomness.transition_uniforms[
                    :, block_index : block_index + 1, :
                ],
            )
        )
    return combine_fixed_ttsirt_block_compilations(
        tuple(blocks), observation_mode="concatenate"
    )


def _candidate_mechanics_diagnostics(
    candidate: FittedCandidate,
    compilation: object,
) -> Mapping[str, object]:
    initial_local, _ = AlgebraicCoordinateMap(
        tf.constant([candidate.spec.scale], FIT_DTYPE)
    ).inverse(compilation.branch.states[0, :, :1])
    initial_transport = candidate.transports[0]
    initial_reference = initial_transport.forward_transport(tf.transpose(initial_local))
    initial_reconstructed = initial_transport.inverse_transport(initial_reference)
    roundtrip_errors = [
        float(
            tf.reduce_max(
                tf.abs(initial_reconstructed - tf.transpose(initial_local))
            ).numpy()
        )
    ]
    conditional_tieout_errors = []
    for time_index, transport in enumerate(candidate.transports[1:], start=1):
        ancestor = compilation.branch.ancestors[time_index - 1]
        previous = tf.gather(compilation.branch.states[time_index - 1, :, :1], ancestor)
        current = compilation.branch.states[time_index, :, :1]
        coordinate_map = AlgebraicCoordinateMap(
            tf.constant([candidate.spec.scale], FIT_DTYPE)
        )
        previous_local, _ = coordinate_map.inverse(previous)
        current_local, _ = coordinate_map.inverse(current)
        conditioning = tf.transpose(previous_local)
        generated = tf.transpose(current_local)
        conditional = transport.conditional_proposal_log_density(
            conditioning_points=conditioning,
            generated_points=generated,
        )
        joint = tf.math.log(transport.eval_pdf(tf.concat([conditioning, generated], axis=0)))
        prefix_relative = transport.density.normalized_marginal_density_values(
            (0,), previous_local
        )
        prefix_log = tf.math.log(0.5 * prefix_relative)
        conditional_tieout_errors.append(
            float(tf.reduce_max(tf.abs(conditional - (joint - prefix_log))).numpy())
        )
        reference = transport.forward_transport(
            tf.concat([conditioning, generated], axis=0)
        )[1:]
        reconstructed = transport.conditional_inverse_transport(conditioning, reference)
        roundtrip_errors.append(
            float(tf.reduce_max(tf.abs(reconstructed - generated)).numpy())
        )
    return {
        "maximum_conditional_formula_tieout_abs_error": max(
            conditional_tieout_errors, default=0.0
        ),
        "maximum_inverse_forward_roundtrip_abs_error": max(roundtrip_errors),
        "per_transport_roundtrip_abs_error": roundtrip_errors,
    }


def _cast_ttsirt_branch(
    candidate: FittedCandidate,
    compilation: object,
) -> tuple[PreparedFrozenProposalBranch, Mapping[str, object]]:
    """Re-evaluate proposal densities at the rounded online states."""

    source = compilation.branch
    states = tf.cast(source.states, ONLINE_DTYPE)
    evaluation_states = tf.cast(states, FIT_DTYPE)
    coordinate_map = AlgebraicCoordinateMap(
        tf.constant([candidate.spec.scale], FIT_DTYPE)
    )
    initial_terms = []
    for block_index in range(source.state_dimension):
        local, log_dz_dx = coordinate_map.inverse(
            evaluation_states[0, :, block_index : block_index + 1]
        )
        initial_terms.append(
            tf.math.log(candidate.transports[0].eval_pdf(tf.transpose(local)))
            + log_dz_dx
        )
    initial_log_q = tf.add_n(initial_terms)

    transition_rows = []
    for time_index, transport in enumerate(candidate.transports[1:], start=1):
        ancestor = source.ancestors[time_index - 1]
        block_terms = []
        for block_index in range(source.state_dimension):
            previous = tf.gather(
                evaluation_states[
                    time_index - 1, :, block_index : block_index + 1
                ],
                ancestor,
            )
            current = evaluation_states[
                time_index, :, block_index : block_index + 1
            ]
            previous_local, _ = coordinate_map.inverse(previous)
            current_local, current_log_dz_dx = coordinate_map.inverse(current)
            block_terms.append(
                transport.conditional_proposal_log_density(
                    conditioning_points=tf.transpose(previous_local),
                    generated_points=tf.transpose(current_local),
                )
                + current_log_dz_dx
            )
        transition_rows.append(tf.add_n(block_terms))
    transition_log_q = tf.stack(transition_rows)
    initial_delta = tf.reduce_max(
        tf.abs(initial_log_q - source.initial_log_proposal_density)
    )
    transition_delta = tf.reduce_max(
        tf.abs(transition_log_q - source.transition_log_proposal_density)
    )
    branch = prepare_frozen_proposal_branch(
        observations=tf.cast(source.observations, ONLINE_DTYPE),
        states=states,
        initial_log_proposal_density=tf.cast(initial_log_q, ONLINE_DTYPE),
        ancestors=source.ancestors,
        auxiliary_log_probabilities=tf.cast(
            source.auxiliary_log_probabilities, ONLINE_DTYPE
        ),
        transition_log_proposal_density=tf.cast(
            transition_log_q, ONLINE_DTYPE
        ),
    )
    return branch, {
        "policy": "round_states_to_float32_then_recompute_pointwise_log_q_v1",
        "source_fit_dtype": FIT_DTYPE.name,
        "online_dtype": ONLINE_DTYPE.name,
        "maximum_initial_log_q_rounding_correction": float(initial_delta.numpy()),
        "maximum_transition_log_q_rounding_correction": float(
            transition_delta.numpy()
        ),
        "proposal_density_matches_online_state": True,
    }


def _compile_exact_branch(
    model: DiagonalGaussianModel,
    theta: tf.Tensor,
    observations: tf.Tensor,
    *,
    particle_count: int,
    randomness: ProposalRandomness,
    auxiliary_mode: str,
) -> PreparedFrozenProposalBranch:
    if auxiliary_mode not in {"predictive", "uniform"}:
        raise ValueError("unsupported auxiliary mode")
    dtype = model.dtype
    log_particle_count = tf.math.log(tf.cast(particle_count, dtype))
    initial_predictive_variance = model.prior_variance + model.observation_variance
    initial_gain = model.prior_variance / initial_predictive_variance
    initial_variance = (
        model.prior_variance * model.observation_variance / initial_predictive_variance
    )
    initial_mean = theta[0] + initial_gain * (
        observations[0] - theta[1] - theta[0]
    )
    initial_uniforms = tf.cast(tf.transpose(randomness.initial_uniforms), dtype)
    noise = _standard_normal_from_uniform(
        initial_uniforms, dtype
    )
    current = initial_mean + tf.sqrt(tf.cast(initial_variance, dtype)) * noise
    states = [current]
    initial_log_q = _normal_log_density_nd(
        current, initial_mean, initial_variance, dtype
    )
    ancestors = []
    auxiliary_rows = []
    transition_log_q_rows = []
    predictive_variance = model.transition_variance + model.observation_variance
    gain = model.transition_variance / predictive_variance
    proposal_variance = (
        model.transition_variance * model.observation_variance / predictive_variance
    )
    for time_index in range(1, int(observations.shape[0])):
        predicted = model.transition_scale * current + theta[0]
        if auxiliary_mode == "predictive":
            predictive_log_density = _normal_log_density_nd(
                observations[time_index][None, :],
                predicted + theta[1],
                predictive_variance,
                dtype,
            )
            log_auxiliary = (
                -log_particle_count
                + predictive_log_density
                - tf.reduce_logsumexp(-log_particle_count + predictive_log_density)
            )
        else:
            log_auxiliary = tf.fill([particle_count], -log_particle_count)
        cdf = tf.math.cumsum(tf.exp(log_auxiliary))
        cdf = tf.concat([cdf[:-1], tf.ones([1], dtype)], axis=0)
        ancestor = tf.searchsorted(
            cdf,
            tf.cast(randomness.ancestor_uniforms[time_index - 1], dtype),
            side="right",
            out_type=tf.int32,
        )
        selected_prediction = tf.gather(predicted, ancestor)
        proposal_mean = selected_prediction + gain * (
            observations[time_index][None, :] - theta[1] - selected_prediction
        )
        noise = _standard_normal_from_uniform(
            tf.cast(
                tf.transpose(randomness.transition_uniforms[time_index - 1]),
                dtype,
            ),
            dtype,
        )
        current = proposal_mean + tf.sqrt(tf.cast(proposal_variance, dtype)) * noise
        states.append(current)
        ancestors.append(ancestor)
        auxiliary_rows.append(log_auxiliary)
        transition_log_q_rows.append(
            _normal_log_density_nd(
                current, proposal_mean, proposal_variance, dtype
            )
        )
    return prepare_frozen_proposal_branch(
        observations=observations,
        states=tf.stack(states),
        initial_log_proposal_density=initial_log_q,
        ancestors=tf.stack(ancestors),
        auxiliary_log_probabilities=tf.stack(auxiliary_rows),
        transition_log_proposal_density=tf.stack(transition_log_q_rows),
    )


def _evaluate_arm(
    model: DiagonalGaussianModel,
    branch: PreparedFrozenProposalBranch,
    theta: tf.Tensor,
) -> tuple[Mapping[str, object], object]:
    program = prepare_frozen_proposal_apf_program(model, branch)
    compiled = program.compiled()
    compile_started = time.monotonic()
    result = compiled(theta)
    compile_seconds = time.monotonic() - compile_started
    warm_started = time.monotonic()
    warmed = compiled(theta)
    warmed_seconds = time.monotonic() - warm_started
    fd_step = tf.cast(1e-3, model.dtype)
    fd_rows = []
    for parameter_index in range(model.parameter_dim()):
        direction = tf.one_hot(parameter_index, model.parameter_dim(), dtype=model.dtype)
        plus = compiled(theta + fd_step * direction)["log_likelihood"]
        minus = compiled(theta - fd_step * direction)["log_likelihood"]
        fd_rows.append((plus - minus) / (2.0 * fd_step))
    finite_difference = tf.stack(fd_rows)
    score_error = tf.reduce_max(tf.abs(result["score"] - finite_difference))
    repeatability_error = tf.abs(
        result["log_likelihood"] - warmed["log_likelihood"]
    )
    minimum_ess_fraction = result["minimum_ess"] / tf.cast(
        branch.particle_count, model.dtype
    )
    payload = {
        "program_id": program.program_id,
        "branch_id": branch.branch_id,
        "log_likelihood": float(result["log_likelihood"].numpy()),
        "score": _float_list(result["score"]),
        "same_scalar_fd_score": _float_list(finite_difference),
        "same_scalar_fd_max_abs_error": float(score_error.numpy()),
        "warmed_repeatability_abs_error": float(repeatability_error.numpy()),
        "minimum_ess": float(result["minimum_ess"].numpy()),
        "minimum_ess_fraction": float(minimum_ess_fraction.numpy()),
        "maximum_log_weight_spread": float(
            result["maximum_log_weight_spread"].numpy()
        ),
        "finite": bool(result["finite"].numpy()),
        "compile_inclusive_seconds": compile_seconds,
        "warmed_seconds": warmed_seconds,
        "output_device": result["log_likelihood"].device,
    }
    return payload, program


def _exact_kalman_value_and_score(
    model: DiagonalGaussianModel,
    theta: tf.Tensor,
    observations: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    dtype = model.dtype
    mean = tf.fill([model.dimension], theta[0])
    derivative = tf.stack(
        [tf.ones([model.dimension], dtype), tf.zeros([model.dimension], dtype)],
        axis=1,
    )
    variance = tf.cast(model.prior_variance, dtype)
    value = tf.zeros([], dtype)
    score = tf.zeros([2], dtype)
    for time_index in range(int(observations.shape[0])):
        if time_index > 0:
            mean = model.transition_scale * mean + theta[0]
            derivative = model.transition_scale * derivative + tf.constant(
                [1.0, 0.0], dtype
            )[None, :]
            variance = model.transition_scale**2 * variance + model.transition_variance
        innovation_variance = variance + model.observation_variance
        predictive_mean = mean + theta[1]
        predictive_derivative = derivative + tf.constant([0.0, 1.0], dtype)[None, :]
        innovation = observations[time_index] - predictive_mean
        value = value + _normal_log_density_nd(
            observations[time_index][None, :],
            predictive_mean,
            innovation_variance,
            dtype,
        )[0]
        score = score + tf.reduce_sum(
            innovation[:, None] * predictive_derivative / innovation_variance,
            axis=0,
        )
        gain = variance / innovation_variance
        mean = mean + gain * innovation
        derivative = derivative - gain * predictive_derivative
        variance = (1.0 - gain) * variance
    return value, score


def _git_payload() -> Mapping[str, object]:
    commit = subprocess.run(
        ("git", "rev-parse", "HEAD"), check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = subprocess.run(
        ("git", "status", "--short"), check=True, capture_output=True, text=True
    ).stdout.splitlines()
    return {"commit": commit, "dirty": bool(dirty), "dirty_line_count": len(dirty)}


def _write_markdown(path: Path, payload: Mapping[str, object]) -> None:
    lines = [
        "# Zhao-Cui Blockwise TTSIRT-APF Rung-1 Result",
        "",
        f"Status: `{payload['status']}`",
        "",
        "This is a synthetic independent-block mechanics result for a BayesFilter extension. It is not source-faithful Zhao-Cui, nonlinear-model evidence, Austria SIR evidence, or NAWM evidence.",
        "",
        "## Decision",
        "",
        "| Field | Result |",
        "| --- | --- |",
    ]
    decision = payload["decision"]
    lines.extend(f"| {key} | `{value}` |" for key, value in decision.items())
    lines.extend(["", "## Gates", "", "| Gate | Status |", "| --- | --- |"])
    lines.extend(f"| {key} | `{value}` |" for key, value in payload["gates"].items())
    lines.extend(["", "## Arms", "", "| Arm | ESS fraction | Score/FD max error | Log likelihood |", "| --- | ---: | ---: | ---: |"])
    for name, row in payload["arms"].items():
        lines.append(
            f"| {name} | {row['minimum_ess_fraction']:.6g} | "
            f"{row['same_scalar_fd_max_abs_error']:.6g} | {row['log_likelihood']:.9g} |"
        )
    lines.extend(
        [
            "",
            "## Inference Status",
            "",
            "No stochastic ranking is supported. The exact arms are mechanism references; observed cross-arm differences are descriptive only. The candidate either passes or fails the predeclared hard screen.",
            "",
            "## Nonclaims",
            "",
        ]
    )
    lines.extend(f"- {claim}" for claim in payload["nonclaims"])
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _relative_rms(
    approximation: tf.Tensor, target: tf.Tensor, weights: tf.Tensor
) -> tf.Tensor:
    numerator = tf.sqrt(tf.reduce_sum(weights * tf.square(approximation - target)))
    denominator = tf.sqrt(tf.reduce_sum(weights * tf.square(target)))
    return numerator / tf.maximum(denominator, tf.constant(1e-15, FIT_DTYPE))


def _normal_log_density_1d(
    value: tf.Tensor, mean: tf.Tensor, variance: tf.Tensor | float
) -> tf.Tensor:
    variance_tensor = tf.cast(variance, FIT_DTYPE)
    residual = tf.cast(value, FIT_DTYPE) - tf.cast(mean, FIT_DTYPE)
    return -0.5 * (
        tf.cast(LOG_TWO_PI, FIT_DTYPE)
        + tf.math.log(variance_tensor)
        + tf.square(residual) / variance_tensor
    )


def _scalar_normal_log_density(
    value: tf.Tensor, mean: tf.Tensor, variance: tf.Tensor
) -> tf.Tensor:
    return tf.reshape(_normal_log_density_1d(value, mean, variance), [])


def _normal_log_density_nd(
    value: tf.Tensor,
    mean: tf.Tensor,
    variance: tf.Tensor | float,
    dtype: tf.dtypes.DType,
) -> tf.Tensor:
    residual = tf.cast(value, dtype) - tf.cast(mean, dtype)
    variance_tensor = tf.cast(variance, dtype)
    dimension = tf.cast(tf.shape(residual)[-1], dtype)
    return -0.5 * (
        dimension * (tf.cast(LOG_TWO_PI, dtype) + tf.math.log(variance_tensor))
        + tf.reduce_sum(tf.square(residual), axis=-1) / variance_tensor
    )


def _standard_normal_from_uniform(
    uniforms: tf.Tensor, dtype: tf.dtypes.DType
) -> tf.Tensor:
    values = tf.convert_to_tensor(uniforms, dtype=dtype)
    epsilon = tf.cast(1e-7 if dtype == tf.float32 else 1e-15, dtype)
    clipped = tf.clip_by_value(values, epsilon, 1.0 - epsilon)
    return tf.sqrt(tf.cast(2.0, dtype)) * tf.math.erfinv(2.0 * clipped - 1.0)


def _float_list(value: tf.Tensor) -> list[float]:
    return [float(item) for item in tf.reshape(value, [-1]).numpy()]


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
    """Preserve infrastructure failures without claiming a scientific result."""

    try:
        output_index = sys.argv.index("--output-root") + 1
        output_root = Path(sys.argv[output_index])
    except (ValueError, IndexError):
        return
    if not output_root.is_dir() or (output_root / "result.json").exists():
        return
    payload = {
        "schema": "bayesfilter.zhao_cui_blockwise_ttsirt_apf.rung1.failure.v1",
        "status": "FAIL_INFRASTRUCTURE_OR_HARNESS",
        "failure_classification": "not_candidate_evidence",
        "exception_type": type(error).__name__,
        "exception_message": str(error),
        "traceback": traceback.format_exc(),
        "command": " ".join(sys.argv),
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": _git_payload(),
        "nonclaims": [
            "no candidate rejection",
            "no research-direction rejection",
            "no scientific or performance interpretation",
        ],
    }
    (output_root / "failure_result.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_root / "failure_result.md").write_text(
        "\n".join(
            [
                "# Zhao-Cui Blockwise TTSIRT-APF Rung-1 Failed Attempt",
                "",
                "Status: `FAIL_INFRASTRUCTURE_OR_HARNESS`",
                "",
                f"Failure: `{type(error).__name__}: {error}`",
                "",
                "This attempt is not candidate evidence and does not reject the research direction.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dimension", type=int, default=24)
    parser.add_argument("--time-steps", type=int, default=3)
    parser.add_argument("--particle-count", type=int, default=256)
    parser.add_argument("--seed", type=int, default=220723)
    parser.add_argument("--cpu-reference", action="store_true")
    parser.add_argument("--debug-smoke", action="store_true")
    args = parser.parse_args()
    if args.dimension < 1 or args.time_steps < 2 or args.particle_count < 2:
        raise ValueError("invalid rung dimensions")
    realized_scope = (args.dimension, args.time_steps, args.particle_count)
    if not args.debug_smoke and realized_scope != CANONICAL_SCOPE:
        raise ValueError(
            "noncanonical dimensions require --debug-smoke; canonical scope is "
            "d=24,T=3,N=256"
        )
    execution_role = (
        "debug_smoke"
        if args.debug_smoke
        else "cpu_reference_precheck"
        if args.cpu_reference
        else "trusted_gpu_claim"
    )
    args.output_root.mkdir(parents=True, exist_ok=False)
    started_at = datetime.now(timezone.utc)
    started = time.monotonic()
    device = DEVICE

    theta64 = tf.constant([0.18, -0.11], FIT_DTYPE)
    scalar_observations = _scalar_observations(args.time_steps)
    targets = ScalarGaussianTargets.build(
        scalar_observations,
        theta64,
        prior_variance=1.25,
        transition_variance=0.7,
        observation_variance=0.8,
        transition_scale=0.65,
    )
    randomness = _proposal_randomness(
        dimension=args.dimension,
        time_steps=args.time_steps,
        particle_count=args.particle_count,
        seed=args.seed,
    )
    with tf.device("/CPU:0"):
        fit_started = time.monotonic()
        candidate, tuning_records = _select_candidate(targets)
        fit_seconds = time.monotonic() - fit_started
        compile_started = time.monotonic()
        compilation = _compile_ttsirt_candidate(
            candidate,
            scalar_observations,
            dimension=args.dimension,
            particle_count=args.particle_count,
            randomness=randomness,
        )
        mechanics = _candidate_mechanics_diagnostics(candidate, compilation)
        candidate_branch, online_cast_diagnostics = _cast_ttsirt_branch(
            candidate, compilation
        )
        proposal_compile_seconds = time.monotonic() - compile_started

    model = DiagonalGaussianModel(args.dimension)
    theta = tf.cast(theta64, ONLINE_DTYPE)
    observations = tf.tile(
        tf.cast(scalar_observations[:, None], ONLINE_DTYPE), [1, args.dimension]
    )
    with tf.device("/CPU:0"):
        exact_predictive_branch = _compile_exact_branch(
            model,
            theta,
            observations,
            particle_count=args.particle_count,
            randomness=randomness,
            auxiliary_mode="predictive",
        )
        exact_uniform_branch = _compile_exact_branch(
            model,
            theta,
            observations,
            particle_count=args.particle_count,
            randomness=randomness,
            auxiliary_mode="uniform",
        )

    arms = {}
    programs = {}
    with tf.device(str(device["online_device"])):
        for name, branch in (
            ("exact_predictive_auxiliary", exact_predictive_branch),
            ("exact_uniform_auxiliary", exact_uniform_branch),
            ("fitted_ttsirt_uniform_auxiliary", candidate_branch),
        ):
            arms[name], programs[name] = _evaluate_arm(model, branch, theta)

    exact_value, exact_score = _exact_kalman_value_and_score(
        model, theta, observations
    )
    for arm in arms.values():
        arm["descriptive_kalman_value_error"] = arm["log_likelihood"] - float(
            exact_value.numpy()
        )
        arm["descriptive_kalman_score_error"] = [
            observed - exact
            for observed, exact in zip(arm["score"], _float_list(exact_score))
        ]

    candidate_arm = arms["fitted_ttsirt_uniform_auxiliary"]
    fit_mass_error = float(
        candidate.diagnostics[
            "maximum_reference_target_quadrature_mass_abs_error"
        ]
    )
    transport_manifests = [
        transport.manifest_payload() for transport in candidate.transports
    ]
    gates = {
        "all_candidate_fits_finite": all(
            row["fit_status"] == HighDimStatus.OK.value
            for row in candidate.diagnostics["targets"]
        ),
        "reference_target_mass_error_le_0p03": fit_mass_error
        <= TARGET_MASS_TOLERANCE,
        "positive_defensive_mass": candidate.spec.defensive_mass > 0.0,
        "paired_core_conditional_backend": all(
            manifest["proposition2_marginal_backend"]
            == "paired_core_mass_contraction_prefix_suffix"
            for manifest in transport_manifests
        ),
        "diagnostic_nonproduction_kr_classification": all(
            manifest["production_kr_closure"] is False
            for manifest in transport_manifests
        ),
        "independent_block_proposal_randomness": len(
            set(compilation.manifest["block_compiler_ids"])
        )
        == args.dimension,
        "conditional_formula_tieout_le_1e_10": mechanics[
            "maximum_conditional_formula_tieout_abs_error"
        ]
        <= CONDITIONAL_TIEOUT_TOLERANCE,
        "inverse_forward_roundtrip_le_1e_4": mechanics[
            "maximum_inverse_forward_roundtrip_abs_error"
        ]
        <= ROUNDOFF_TOLERANCE,
        "candidate_finite": candidate_arm["finite"],
        "proposal_density_recomputed_at_online_state": online_cast_diagnostics[
            "proposal_density_matches_online_state"
        ],
        "all_arms_warmed_repeatability_error_le_1e_5": all(
            arm["warmed_repeatability_abs_error"] <= 1e-5
            for arm in arms.values()
        ),
        "same_scalar_score_fd_error_le_0p03": candidate_arm[
            "same_scalar_fd_max_abs_error"
        ]
        <= SCORE_FD_TOLERANCE,
        "candidate_minimum_ess_fraction_ge_0p5": candidate_arm[
            "minimum_ess_fraction"
        ]
        >= ESS_FRACTION_TOLERANCE,
        "xla_enabled": True,
        "expected_device": (
            "CPU" in candidate_arm["output_device"]
            if args.cpu_reference
            else "GPU" in candidate_arm["output_device"]
        ),
        "memory_growth_verified": args.cpu_reference
        or device["gpu_memory_policy"]["all_physical_devices_memory_growth"],
    }
    numerical_screen_passed = all(gates.values())
    claim_evaluated = execution_role == "trusted_gpu_claim"
    candidate_rejected = (not numerical_screen_passed) if claim_evaluated else None
    memory = (
        {"current": 0, "peak": 0}
        if args.cpu_reference
        else tf.config.experimental.get_memory_info("GPU:0")
    )
    if numerical_screen_passed:
        status = {
            "debug_smoke": "PASS_DEBUG_SMOKE",
            "cpu_reference_precheck": "PASS_CPU_REFERENCE_PRECHECK",
            "trusted_gpu_claim": "PASS_ENGINEERING_RUNG1",
        }[execution_role]
    else:
        status = (
            "BLOCK_CANDIDATE_CONTINUE_RESEARCH_RUNG1"
            if claim_evaluated
            else "FAIL_DEBUG_OR_PRECHECK"
        )
    payload = {
        "schema": SCHEMA,
        "status": status,
        "execution_role": execution_role,
        "canonical_scope": {
            "dimension": CANONICAL_SCOPE[0],
            "time_steps": CANONICAL_SCOPE[1],
            "particle_count": CANONICAL_SCOPE[2],
        },
        "route_classification": ROUTE_CLASSIFICATION,
        "candidate": {
            **candidate.spec.payload(),
            "selection_metric": candidate.diagnostics[
                "maximum_full_proposal_relative_holdout_sqrt_rms"
            ],
            "fit_diagnostics": candidate.diagnostics,
            "mechanics_diagnostics": mechanics,
            "online_cast_diagnostics": online_cast_diagnostics,
            "compiler_id": compilation.compiler_id,
            "compiler_manifest": compilation.manifest,
        },
        "tuning": {
            "candidate_count": len(tuning_records),
            "selection_rule": "finite_then_minimum_maximum_full_proposal_relative_holdout_sqrt_rms",
            "calibration_order_per_axis": CALIBRATION_ORDER,
            "holdout_midpoints_per_axis": HOLDOUT_ORDER,
            "records": tuning_records,
        },
        "arms": arms,
        "exact_kalman": {
            "log_likelihood": float(exact_value.numpy()),
            "score": _float_list(exact_score),
        },
        "gates": gates,
        "decision": {
            "candidate_rejected": candidate_rejected,
            "research_direction_rejected": False,
            "primary_criterion_status": (
                "passed"
                if claim_evaluated and numerical_screen_passed
                else "failed"
                if claim_evaluated
                else "not_assessed_nonclaiming_execution"
            ),
            "veto_diagnostic_status": (
                "passed"
                if numerical_screen_passed
                else "candidate veto fired"
                if claim_evaluated
                else "debug_or_precheck_failure"
            ),
            "main_uncertainty": "synthetic independent blocks and diagnostic finite-grid KR only",
            "next_justified_action": (
                "coupled nonlinear block/rank rung"
                if claim_evaluated and numerical_screen_passed
                else "fresh authorized canonical trusted-GPU claim run"
                if numerical_screen_passed
                else "fresh scope-specific proposal or auxiliary-law repair"
            ),
            "not_concluded": "no nonlinear, HMC, source-faithful, NAWM, or default-readiness claim",
        },
        "inference_status": {
            "hard_veto_screen": (
                "passed"
                if claim_evaluated and numerical_screen_passed
                else "failed"
                if claim_evaluated
                else "not assessed by nonclaiming execution"
            ),
            "statistically_supported_ranking": "none",
            "descriptive_only_differences": "ESS, value, score, and timing differences across one frozen branch per arm",
            "default_readiness": "not assessed",
            "next_evidence_needed": (
                "coupled nonlinear multi-seed block/rank ladder"
                if claim_evaluated and numerical_screen_passed
                else "canonical trusted-GPU claim run"
                if numerical_screen_passed
                else "repair and fresh untouched claim branch"
            ),
        },
        "post_run_red_team": {
            "strongest_alternative_explanation": "the independent product target is unusually favorable and the finite-grid inverse may not generate the exact fitted density",
            "result_that_would_overturn": "coupled nonlinear blocks collapse ESS or violate same-scalar/measure gates after scope-specific tuning",
            "weakest_evidence": "one frozen branch, short T=3 horizon, and diagnostic grid-CDF inversion",
        },
        "device": device,
        "run_manifest": {
            "git": _git_payload(),
            "command": " ".join(sys.argv),
            "environment": sys.executable,
            "python_version": platform.python_version(),
            "tensorflow_version": tf.__version__,
            "fit_dtype": FIT_DTYPE.name,
            "online_dtype": ONLINE_DTYPE.name,
            "tf32_enabled": device["tf32_enabled"],
            "jit_compile": True,
            "gpu_status": device["execution_class"],
            "gpu_memory_policy": device["gpu_memory_policy"],
            "random_seeds": [args.seed],
            "common_random_numbers_across_arms": True,
            "dimension": args.dimension,
            "time_steps": args.time_steps,
            "particle_count": args.particle_count,
            "started_at_utc": started_at.isoformat(),
            "wall_time_seconds": time.monotonic() - started,
            "offline_fit_seconds": fit_seconds,
            "proposal_compile_seconds": proposal_compile_seconds,
            "gpu_allocator_current_bytes": int(memory["current"]),
            "gpu_allocator_peak_bytes": int(memory["peak"]),
            "output_artifacts": [
                str(args.output_root / "result.json"),
                str(args.output_root / "result.md"),
            ],
            "plan_file": PLAN_PATH,
            "result_file": str(args.output_root / "result.md"),
            "trust_basis": (
                "explicit_cpu_reference"
                if args.cpu_reference
                else "owner_designated_managed_session_visible_gpu_trusted"
            ),
            "data_version": "synthetic_scalar_observations_v1",
            "execution_role": execution_role,
            "debug_smoke": args.debug_smoke,
        },
        "nonclaims": [
            "no source-faithful Zhao-Cui claim",
            "no exact randomized-estimator or pseudo-marginal claim",
            "no nonlinear or cross-coordinate scalability claim",
            "no Austria SIR or NAWM claim",
            "no posterior correctness or HMC convergence claim",
            "no statistically supported ranking or superiority claim",
            "no production KR closure or default-readiness claim",
        ],
    }
    payload = _jsonable(payload)
    json_path = args.output_root / "result.json"
    markdown_path = args.output_root / "result.md"
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_markdown(markdown_path, payload)
    print(json.dumps(payload, sort_keys=True))
    if not numerical_screen_passed:
        raise SystemExit(2)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        _write_failure_receipt(error)
        raise
