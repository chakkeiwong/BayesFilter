"""Batch-native T1 target and absolute-scale training for centered Lane-B TTs."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Callable, Sequence

import tensorflow as tf

from bayesfilter.highdim.models import parameterized_zhao_cui_sir_austria_model
from bayesfilter.highdim.zhao_cui_austria_sir_centered_density_tf import (
    CenteredThetaFeatures,
    LaneBCenteredResidualChild,
    _cross_mass,
    _cross_prefix_values,
    _evaluate_component,
    centered_lane_b_product_basis,
)
from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_target_tf import (
    generate_sealed_lane_b_dataset,
)
from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_tf import (
    LaneBT1Artifact,
    balanced_initial_cores,
    lane_b_measure_convention,
)


DTYPE = tf.float64
PARAMETER_DIM = 3
STATE_DIM = 18
OBSERVATION_DIM = 9
JOINT_DIM = 36
LOG_TWO_PI = tf.constant(math.log(2.0 * math.pi), DTYPE)
REFERENCE_LOG_DENSITY = tf.constant(-JOINT_DIM * math.log(2.0), DTYPE)
PARAMETER_BATCH_SCHEMA = (
    "bayesfilter.zhao_cui_austria_sir_parameter_density_training_batch.v1"
)
ABSOLUTE_LOSS_ID = "absolute_i_divergence_common_random_number_v1"
ORIGIN_SCORE_PREFIT_ID = "exact_normalized_origin_score_training_only_v1"
TARGET_INFORMED_ADDITIVE_INITIALIZATION_ID = (
    "exact_origin_point_and_global_score_additive_rank2_weighted_ridge_v1"
)
TARGET_INFORMED_WITHIN_REGION_PAIR_INITIALIZATION_ID = (
    "exact_origin_scores_additive_plus_within_region_si_pairs_rank7_ridge_v1"
)
WITHIN_REGION_PAIR_AXES = tuple((axis, axis + 1) for axis in range(0, JOINT_DIM, 2))
_PARAMETERIZED_MODEL = parameterized_zhao_cui_sir_austria_model()
_BASE_MODEL = _PARAMETERIZED_MODEL.base_model
_ADJACENCY = tf.convert_to_tensor(_BASE_MODEL._adjacency_matrix, DTYPE)  # noqa: SLF001
_DEGREE = tf.convert_to_tensor(_BASE_MODEL._neighbor_degree, DTYPE)  # noqa: SLF001
_INITIAL_MEAN = tf.convert_to_tensor(_BASE_MODEL.initial_mean, DTYPE)


def _batch_native_transition_mean_and_jacobian(
    theta: tf.Tensor, z0: tf.Tensor
) -> tuple[tf.Tensor, tf.Tensor]:
    parameters = tf.convert_to_tensor(theta, DTYPE)
    state = tf.convert_to_tensor(z0, DTYPE)
    kappa = tf.constant(0.1, DTYPE) * tf.exp(parameters[:, 0])
    nu = tf.constant(18.0, DTYPE) * tf.exp(parameters[:, 1])
    tangent = tf.zeros(tf.concat([tf.shape(state), [PARAMETER_DIM]], axis=0), DTYPE)
    parameter_eye = tf.eye(PARAMETER_DIM, dtype=DTYPE)

    def rhs(values: tf.Tensor, derivatives: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        susceptible = values[:, :, 0::2]
        infectious = values[:, :, 1::2]
        d_susceptible = derivatives[:, :, 0::2, :]
        d_infectious = derivatives[:, :, 1::2, :]
        susceptible_neighbor = (
            tf.einsum("tnj,kj->tnk", susceptible, _ADJACENCY)
            - susceptible * _DEGREE[tf.newaxis, tf.newaxis, :]
        )
        infectious_neighbor = (
            tf.einsum("tnj,kj->tnk", infectious, _ADJACENCY)
            - infectious * _DEGREE[tf.newaxis, tf.newaxis, :]
        )
        d_susceptible_neighbor = (
            tf.einsum("tnjp,kj->tnkp", d_susceptible, _ADJACENCY)
            - d_susceptible * _DEGREE[tf.newaxis, tf.newaxis, :, tf.newaxis]
        )
        d_infectious_neighbor = (
            tf.einsum("tnjp,kj->tnkp", d_infectious, _ADJACENCY)
            - d_infectious * _DEGREE[tf.newaxis, tf.newaxis, :, tf.newaxis]
        )
        infection = (
            kappa[:, tf.newaxis, tf.newaxis] * susceptible * infectious
        )
        d_infection = kappa[:, tf.newaxis, tf.newaxis, tf.newaxis] * (
            infectious[:, :, :, tf.newaxis] * d_susceptible
            + susceptible[:, :, :, tf.newaxis] * d_infectious
        ) + infection[:, :, :, tf.newaxis] * parameter_eye[
            0, tf.newaxis, tf.newaxis, tf.newaxis, :
        ]
        rhs_susceptible = -infection + 0.5 * susceptible_neighbor
        rhs_infectious = (
            infection
            - nu[:, tf.newaxis, tf.newaxis] * infectious
            + 0.5 * infectious_neighbor
        )
        d_rhs_susceptible = -d_infection + 0.5 * d_susceptible_neighbor
        d_rhs_infectious = (
            d_infection
            - nu[:, tf.newaxis, tf.newaxis, tf.newaxis] * d_infectious
            - (nu[:, tf.newaxis, tf.newaxis] * infectious)[:, :, :, tf.newaxis]
            * parameter_eye[1, tf.newaxis, tf.newaxis, tf.newaxis, :]
            + 0.5 * d_infectious_neighbor
        )
        return (
            tf.reshape(
                tf.stack([rhs_susceptible, rhs_infectious], axis=3),
                tf.shape(values),
            ),
            tf.reshape(
                tf.stack([d_rhs_susceptible, d_rhs_infectious], axis=3),
                tf.shape(derivatives),
            ),
        )

    step = tf.constant(0.005, DTYPE)
    for _ in range(4):
        k1, d1 = rhs(state, tangent)
        k2, d2 = rhs(state + 0.5 * step * k1, tangent + 0.5 * step * d1)
        k3, d3 = rhs(state + 0.5 * step * k2, tangent + 0.5 * step * d2)
        # Preserve the Zhao-Cui half-step fourth-stage variant.
        k4, d4 = rhs(state + 0.5 * step * k3, tangent + 0.5 * step * d3)
        state = state + (step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        tangent = tangent + (step / 6.0) * (d1 + 2.0 * d2 + 2.0 * d3 + d4)
    return state, tangent


def batch_native_t1_from_common_noise(
    theta: tf.Tensor,
    initial_noise: tf.Tensor,
    transition_noise: tf.Tensor,
    observation: tf.Tensor,
) -> dict[str, tf.Tensor]:
    """Evaluate the theta-by-sample proposal and target without row mapping."""

    parameters = tf.convert_to_tensor(theta, DTYPE)
    initial = tf.convert_to_tensor(initial_noise, DTYPE)
    transition = tf.convert_to_tensor(transition_noise, DTYPE)
    y1 = tf.reshape(tf.convert_to_tensor(observation, DTYPE), [OBSERVATION_DIM])
    if parameters.shape.rank != 2 or parameters.shape[1] != PARAMETER_DIM:
        raise ValueError("theta must have shape [theta_count,3]")
    if initial.shape.rank != 2 or initial.shape[1] != STATE_DIM:
        raise ValueError("initial_noise must have shape [sample_count,18]")
    if transition.shape != initial.shape:
        raise ValueError("transition_noise must match initial_noise")
    theta_count = tf.shape(parameters)[0]
    sample_count = tf.shape(initial)[0]
    z0_base = _INITIAL_MEAN[tf.newaxis, :] + initial
    z0 = tf.broadcast_to(
        z0_base[tf.newaxis, :, :], [theta_count, sample_count, STATE_DIM]
    )
    transition_mean, transition_mean_jacobian = (
        _batch_native_transition_mean_and_jacobian(parameters, z0)
    )
    z1 = transition_mean + transition[tf.newaxis, :, :]

    initial_log = -0.5 * (
        tf.cast(STATE_DIM, DTYPE) * LOG_TWO_PI + tf.reduce_sum(tf.square(initial), axis=1)
    )
    initial_log = tf.broadcast_to(initial_log[tf.newaxis, :], [theta_count, sample_count])
    transition_log = -0.5 * (
        tf.cast(STATE_DIM, DTYPE) * LOG_TWO_PI
        + tf.reduce_sum(tf.square(transition), axis=1)
    )
    transition_log = tf.broadcast_to(
        transition_log[tf.newaxis, :], [theta_count, sample_count]
    )
    variance = tf.constant(100.0, DTYPE) * tf.exp(2.0 * parameters[:, 2])
    observation_residual = y1[tf.newaxis, tf.newaxis, :] - z1[:, :, 1::2]
    observation_log = -0.5 * tf.reduce_sum(
        LOG_TWO_PI
        + tf.math.log(variance)[:, tf.newaxis, tf.newaxis]
        + tf.square(observation_residual) / variance[:, tf.newaxis, tf.newaxis],
        axis=2,
    )
    joint_points = tf.concat([z1, z0], axis=2)
    complete_log = initial_log + transition_log + observation_log
    transition_score = tf.reduce_sum(
        (z1 - transition_mean)[:, :, :, tf.newaxis]
        * transition_mean_jacobian,
        axis=2,
    )
    observation_quad = tf.reduce_sum(
        tf.square(observation_residual) / variance[:, tf.newaxis, tf.newaxis],
        axis=2,
    )
    observation_score = tf.zeros(
        [theta_count, sample_count, PARAMETER_DIM], DTYPE
    )
    observation_score = tf.concat(
        [
            observation_score[:, :, :2],
            (observation_quad - tf.cast(OBSERVATION_DIM, DTYPE))[:, :, tf.newaxis],
        ],
        axis=2,
    )
    complete_score = transition_score + observation_score
    return {
        "joint_points": joint_points,
        "z0": z0,
        "z1": z1,
        "proposal_log_density": initial_log + transition_log,
        "observation_log_density": observation_log,
        "complete_log_density": complete_log,
        "complete_data_score": complete_score,
    }


def _physical_to_local_and_reference_batch(
    joint_points: tf.Tensor, parent: LaneBT1Artifact
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    values = tf.convert_to_tensor(joint_points, DTYPE)
    theta_count = tf.shape(values)[0]
    sample_count = tf.shape(values)[1]
    flat = tf.reshape(values, [-1, JOINT_DIM])
    local_columns = tf.linalg.triangular_solve(
        parent.frame.matrix,
        tf.transpose(flat) - parent.frame.mu[:, tf.newaxis],
        lower=True,
    )
    local = tf.transpose(local_columns)
    scaled = local
    reference = scaled * tf.math.rsqrt(1.0 + tf.square(scaled))
    reference_to_local_log_jacobian = -1.5 * tf.math.log(
        1.0 - tf.square(tf.clip_by_value(reference, -1.0 + 1e-12, 1.0 - 1e-12))
    )
    log_coordinate_jacobian = (
        parent.frame.log_abs_det()
        + tf.reduce_sum(reference_to_local_log_jacobian, axis=1)
    )
    return (
        tf.reshape(local, [theta_count, sample_count, JOINT_DIM]),
        tf.reshape(reference, [theta_count, sample_count, JOINT_DIM]),
        tf.reshape(log_coordinate_jacobian, [theta_count, sample_count]),
    )


@dataclass(frozen=True)
class T1ParameterDensityBatch:
    theta: tf.Tensor
    physical_points: tf.Tensor
    local_points: tf.Tensor
    reference_points: tf.Tensor
    target_log_density_reference: tf.Tensor
    proposal_log_density: tf.Tensor
    coordinate_log_jacobian: tf.Tensor
    observation_log_density: tf.Tensor
    complete_data_score: tf.Tensor
    role: str

    def __post_init__(self) -> None:
        theta = tf.convert_to_tensor(self.theta, DTYPE)
        physical = tf.convert_to_tensor(self.physical_points, DTYPE)
        local = tf.convert_to_tensor(self.local_points, DTYPE)
        reference = tf.convert_to_tensor(self.reference_points, DTYPE)
        log_target = tf.convert_to_tensor(self.target_log_density_reference, DTYPE)
        proposal = tf.convert_to_tensor(self.proposal_log_density, DTYPE)
        coordinate_jacobian = tf.convert_to_tensor(self.coordinate_log_jacobian, DTYPE)
        observation = tf.convert_to_tensor(self.observation_log_density, DTYPE)
        score = tf.convert_to_tensor(self.complete_data_score, DTYPE)
        if theta.shape.rank != 2 or theta.shape[1] != PARAMETER_DIM:
            raise ValueError("batch theta must have shape [theta_count,3]")
        if physical.shape.rank != 3 or physical.shape[0] != theta.shape[0] or physical.shape[2] != JOINT_DIM:
            raise ValueError("batch physical points have the wrong shape")
        if local.shape != physical.shape:
            raise ValueError("batch local points have the wrong shape")
        if reference.shape != local.shape:
            raise ValueError("batch reference points must match local points")
        expected = local.shape[:2]
        if (
            log_target.shape != expected
            or proposal.shape != expected
            or coordinate_jacobian.shape != expected
            or observation.shape != expected
        ):
            raise ValueError("batch density arrays must have shape [theta_count,sample_count]")
        if score.shape != (*expected, PARAMETER_DIM):
            raise ValueError("batch score has the wrong shape")
        for name, value in (
            ("theta", theta),
            ("physical", physical),
            ("local", local),
            ("reference", reference),
            ("log_target", log_target),
            ("proposal", proposal),
            ("coordinate_jacobian", coordinate_jacobian),
            ("observation", observation),
            ("score", score),
        ):
            tf.debugging.assert_all_finite(value, f"{name} must be finite")
        object.__setattr__(self, "theta", theta)
        object.__setattr__(self, "physical_points", physical)
        object.__setattr__(self, "local_points", local)
        object.__setattr__(self, "reference_points", reference)
        object.__setattr__(self, "target_log_density_reference", log_target)
        object.__setattr__(self, "proposal_log_density", proposal)
        object.__setattr__(self, "coordinate_log_jacobian", coordinate_jacobian)
        object.__setattr__(self, "observation_log_density", observation)
        object.__setattr__(self, "complete_data_score", score)


def build_t1_parameter_density_batch(
    *,
    parent: LaneBT1Artifact,
    theta: tf.Tensor,
    initial_noise: tf.Tensor,
    transition_noise: tf.Tensor,
    role: str,
) -> T1ParameterDensityBatch:
    _states, observations, _all = generate_sealed_lane_b_dataset()
    evaluated = batch_native_t1_from_common_noise(
        theta, initial_noise, transition_noise, observations[0]
    )
    local, reference, coordinate_log_jacobian = _physical_to_local_and_reference_batch(
        evaluated["joint_points"], parent
    )
    log_target = (
        evaluated["complete_log_density"]
        + coordinate_log_jacobian
        - REFERENCE_LOG_DENSITY
        + parent.shift_constant
    )
    return T1ParameterDensityBatch(
        theta=theta,
        physical_points=evaluated["joint_points"],
        local_points=local,
        reference_points=reference,
        target_log_density_reference=log_target,
        proposal_log_density=evaluated["proposal_log_density"],
        coordinate_log_jacobian=coordinate_log_jacobian,
        observation_log_density=evaluated["observation_log_density"],
        complete_data_score=evaluated["complete_data_score"],
        role=str(role),
    )


@dataclass(frozen=True)
class AbsoluteDensityLossTerms:
    total_loss: tf.Tensor
    absolute_density_loss: tf.Tensor
    derivative_matching_loss: tf.Tensor
    exact_child_mass: tf.Tensor
    target_log_density_term: tf.Tensor
    target_mass_estimate: tf.Tensor
    target_mass_standard_error: tf.Tensor
    minimum_rho: tf.Tensor


@dataclass(frozen=True)
class RatioScoreEstimate:
    value: tf.Tensor
    score: tf.Tensor
    score_standard_error: tf.Tensor
    effective_sample_size: tf.Tensor

    def __post_init__(self) -> None:
        value = tf.reshape(tf.convert_to_tensor(self.value, DTYPE), [])
        score = tf.reshape(tf.convert_to_tensor(self.score, DTYPE), [PARAMETER_DIM])
        standard_error = tf.reshape(
            tf.convert_to_tensor(self.score_standard_error, DTYPE), [PARAMETER_DIM]
        )
        ess = tf.reshape(tf.convert_to_tensor(self.effective_sample_size, DTYPE), [])
        tf.debugging.assert_all_finite(value, "ratio estimate value")
        tf.debugging.assert_all_finite(score, "ratio estimate score")
        tf.debugging.assert_all_finite(standard_error, "ratio estimate MCSE")
        tf.debugging.assert_positive(ess, "ratio estimate ESS")
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "score", score)
        object.__setattr__(self, "score_standard_error", standard_error)
        object.__setattr__(self, "effective_sample_size", ess)


@dataclass(frozen=True)
class TargetInformedAdditiveInitialization:
    residual_components: tuple[tuple[tf.Tensor, ...], ...]
    ridge_fraction: float
    global_score_weight: float
    realized_ridge: tf.Tensor
    target_likelihood_score: tf.Tensor
    target_point_score_rms: tf.Tensor
    design_column_rms: tf.Tensor
    coefficient_rms: tf.Tensor
    prefix_weight: float
    prefix_score_tolerance: tf.Tensor
    training_prefix_score_standardized_residual: tf.Tensor

    def __post_init__(self) -> None:
        for name in (
            "realized_ridge",
            "target_likelihood_score",
            "target_point_score_rms",
            "design_column_rms",
            "coefficient_rms",
            "prefix_score_tolerance",
            "training_prefix_score_standardized_residual",
        ):
            value = tf.convert_to_tensor(getattr(self, name), DTYPE)
            tf.debugging.assert_all_finite(value, f"{name} must be finite")
            object.__setattr__(self, name, value)
        if not math.isfinite(float(self.ridge_fraction)) or float(
            self.ridge_fraction
        ) <= 0.0:
            raise ValueError("ridge_fraction must be positive and finite")
        if not math.isfinite(float(self.global_score_weight)) or float(
            self.global_score_weight
        ) <= 0.0:
            raise ValueError("global_score_weight must be positive and finite")
        if not math.isfinite(float(self.prefix_weight)) or float(
            self.prefix_weight
        ) < 0.0:
            raise ValueError("prefix_weight must be finite and nonnegative")


@dataclass(frozen=True)
class TargetInformedPairInitialization:
    residual_components: tuple[tuple[tf.Tensor, ...], ...]
    ridge_fraction: float
    global_score_weight: float
    prefix_weight: float
    realized_ridge: tf.Tensor
    target_likelihood_score: tf.Tensor
    target_point_score_rms: tf.Tensor
    coefficient_rms: tf.Tensor
    training_prefix_score_standardized_residual: tf.Tensor

    def __post_init__(self) -> None:
        for name in (
            "realized_ridge",
            "target_likelihood_score",
            "target_point_score_rms",
            "coefficient_rms",
            "training_prefix_score_standardized_residual",
        ):
            value = tf.convert_to_tensor(getattr(self, name), DTYPE)
            tf.debugging.assert_all_finite(value, f"{name} must be finite")
            object.__setattr__(self, name, value)


def _parent_additive_cross_features(
    parent: LaneBT1Artifact, basis: object
) -> tf.Tensor:
    """Return exact <h0, phi_axis,index> features under the reference measure."""

    active_measure = lane_b_measure_convention().mass_measure
    transfer = []
    for axis, core in enumerate(parent.cores):
        integral = basis.bases[axis].integral_vector(active_measure)
        transfer.append(tf.einsum("lir,i->lr", core, integral))
    left = [tf.ones([1], DTYPE)]
    for matrix in transfer:
        left.append(tf.einsum("l,lr->r", left[-1], matrix))
    right: list[tf.Tensor] = [tf.zeros([1], DTYPE)] * (len(parent.cores) + 1)
    right[-1] = tf.ones([1], DTYPE)
    for axis in range(len(parent.cores) - 1, -1, -1):
        right[axis] = tf.einsum("lr,r->l", transfer[axis], right[axis + 1])
    features = []
    for axis, core in enumerate(parent.cores):
        mass = basis.bases[axis].mass_matrix(active_measure)
        features.append(
            tf.einsum(
                "l,lir,ij,r->j",
                left[axis],
                core,
                mass,
                right[axis + 1],
            )
        )
    return tf.concat(features, axis=0)


def additive_prefix_score_operator(
    *,
    parent: LaneBT1Artifact,
    local_prefix_points: tf.Tensor,
) -> tf.Tensor:
    """Return the exact origin prefix-score operator for additive coefficients."""

    points = tf.convert_to_tensor(local_prefix_points, DTYPE)
    if points.shape.rank != 2 or points.shape[1] is None:
        raise ValueError("local_prefix_points must have shape [point,prefix_dim]")
    prefix_dim = int(points.shape[1])
    if prefix_dim <= 0 or prefix_dim > JOINT_DIM:
        raise ValueError("prefix dimension is out of range")
    basis = centered_lane_b_product_basis(
        order=parent.settings.basis_order,
        num_elems=parent.settings.basis_num_elems,
    )
    point_count = tf.shape(points)[0]
    active_measure = lane_b_measure_convention().mass_measure
    transfers = []
    evaluated_basis = []
    for axis, core in enumerate(parent.cores):
        if axis < prefix_dim:
            evaluated = basis.evaluate_axis(axis, points[:, axis])
            evaluated_basis.append(evaluated)
            transfers.append(tf.einsum("ni,lir->nlr", evaluated, core))
        else:
            integral = basis.bases[axis].integral_vector(active_measure)
            static = tf.einsum("lir,i->lr", core, integral)
            transfers.append(
                tf.broadcast_to(
                    static[tf.newaxis, :, :],
                    [point_count, int(static.shape[0]), int(static.shape[1])],
                )
            )
    left = [tf.ones([point_count, 1], DTYPE)]
    for matrix in transfers:
        left.append(tf.einsum("nl,nlr->nr", left[-1], matrix))
    right: list[tf.Tensor] = [tf.zeros([point_count, 1], DTYPE)] * (
        JOINT_DIM + 1
    )
    right[-1] = tf.ones([point_count, 1], DTYPE)
    for axis in range(JOINT_DIM - 1, -1, -1):
        right[axis] = tf.einsum(
            "nlr,nr->nl", transfers[axis], right[axis + 1]
        )
    parent_prefix_integral = tf.reshape(left[-1], [point_count])
    cross_features = []
    for axis, core in enumerate(parent.cores):
        if axis < prefix_dim:
            cross_features.append(
                parent_prefix_integral[:, tf.newaxis] * evaluated_basis[axis]
            )
        else:
            mass = basis.bases[axis].mass_matrix(active_measure)
            replaced = tf.einsum("lir,ij->ljr", core, mass)
            cross_features.append(
                tf.einsum(
                    "nl,ljr,nr->nj", left[axis], replaced, right[axis + 1]
                )
            )
    prefix_cross = tf.concat(cross_features, axis=1)
    prefix_normalizer = _cross_prefix_values(
        parent.cores, parent.cores, basis, points
    ) + tf.constant(parent.settings.tau, DTYPE)
    global_features = _parent_additive_cross_features(parent, basis)
    parent_mass = _cross_mass(parent.cores, parent.cores, basis)
    global_operator = 2.0 * global_features / (
        parent_mass + tf.constant(parent.settings.tau, DTYPE)
    )
    operator = (
        2.0 * prefix_cross / prefix_normalizer[:, tf.newaxis]
        - global_operator[tf.newaxis, :]
    )
    tf.debugging.assert_all_finite(operator, "additive prefix score operator")
    return operator


def _within_region_feature_values(basis: object, points: tf.Tensor) -> tf.Tensor:
    values = tf.convert_to_tensor(points, DTYPE)
    axis_values = [
        basis.evaluate_axis(axis, values[:, axis]) for axis in range(JOINT_DIM)
    ]
    additive = tf.concat(axis_values, axis=1)
    pairs = [
        tf.reshape(
            axis_values[left][:, :, tf.newaxis]
            * axis_values[right][:, tf.newaxis, :],
            [tf.shape(values)[0], -1],
        )
        for left, right in WITHIN_REGION_PAIR_AXES
    ]
    return tf.concat([additive, *pairs], axis=1)


def _basis_feature_component(
    *,
    axis_indices: Sequence[int],
    basis_indices: Sequence[int],
    basis_dims: Sequence[int],
) -> tuple[tf.Tensor, ...]:
    selected = dict(zip((int(axis) for axis in axis_indices), (int(index) for index in basis_indices)))
    cores = []
    for axis, width in enumerate(basis_dims):
        values = tf.ones([1, int(width), 1], DTYPE)
        if axis in selected:
            values = tf.reshape(
                tf.one_hot(selected[axis], int(width), dtype=DTYPE),
                [1, int(width), 1],
            )
        cores.append(values)
    return tuple(cores)


def _within_region_cross_features(
    *,
    parent: LaneBT1Artifact,
    basis: object,
    local_prefix_points: tf.Tensor | None,
) -> tf.Tensor:
    """Cross <h0,feature> globally or conditionally at prefix points."""

    basis_dims = basis.basis_dim_tuple()
    components = []
    for axis, width in enumerate(basis_dims):
        for basis_index in range(int(width)):
            components.append(
                _basis_feature_component(
                    axis_indices=(axis,),
                    basis_indices=(basis_index,),
                    basis_dims=basis_dims,
                )
            )
    width = int(basis_dims[0])
    for left, right in WITHIN_REGION_PAIR_AXES:
        for left_index in range(width):
            for right_index in range(width):
                components.append(
                    _basis_feature_component(
                        axis_indices=(left, right),
                        basis_indices=(left_index, right_index),
                        basis_dims=basis_dims,
                    )
                )
    if local_prefix_points is None:
        return tf.stack(
            [_cross_mass(parent.cores, component, basis) for component in components]
        )
    points = tf.convert_to_tensor(local_prefix_points, DTYPE)
    return tf.stack(
        [
            _cross_prefix_values(parent.cores, component, basis, points)
            for component in components
        ],
        axis=1,
    )


def within_region_pair_prefix_score_operator(
    *, parent: LaneBT1Artifact, local_prefix_points: tf.Tensor
) -> tf.Tensor:
    points = tf.convert_to_tensor(local_prefix_points, DTYPE)
    basis = centered_lane_b_product_basis(
        order=parent.settings.basis_order,
        num_elems=parent.settings.basis_num_elems,
    )
    prefix_cross = _within_region_cross_features(
        parent=parent, basis=basis, local_prefix_points=points
    )
    global_cross = _within_region_cross_features(
        parent=parent, basis=basis, local_prefix_points=None
    )
    prefix_normalizer = _cross_prefix_values(
        parent.cores, parent.cores, basis, points
    ) + tf.constant(parent.settings.tau, DTYPE)
    parent_mass = _cross_mass(parent.cores, parent.cores, basis)
    return (
        2.0 * prefix_cross / prefix_normalizer[:, tf.newaxis]
        - 2.0
        * global_cross[tf.newaxis, :]
        / (parent_mass + tf.constant(parent.settings.tau, DTYPE))
    )


def _additive_pair_rank_seven_component(
    coefficients: tf.Tensor, *, basis_dim: int
) -> tuple[tf.Tensor, ...]:
    """Encode additive plus disjoint adjacent-pair features at TT rank seven."""

    values = tf.reshape(tf.convert_to_tensor(coefficients, DTYPE), [-1])
    additive_count = JOINT_DIM * int(basis_dim)
    additive = tf.reshape(values[:additive_count], [JOINT_DIM, int(basis_dim)])
    pair = tf.reshape(
        values[additive_count:],
        [len(WITHIN_REGION_PAIR_AXES), int(basis_dim), int(basis_dim)],
    )
    rank = int(basis_dim) + 2
    cores = []
    pair_lookup = {left: index for index, (left, _right) in enumerate(WITHIN_REGION_PAIR_AXES)}
    for axis in range(JOINT_DIM):
        width = int(basis_dim)
        if axis == 0:
            core = tf.zeros([1, width, rank], DTYPE)
            core = tf.tensor_scatter_nd_update(
                core,
                tf.constant([[0, j, 0] for j in range(width)], tf.int32),
                tf.ones([width], DTYPE),
            )
            core = tf.tensor_scatter_nd_update(
                core,
                tf.constant([[0, j, 1] for j in range(width)], tf.int32),
                additive[axis],
            )
            for j in range(width):
                core = tf.tensor_scatter_nd_update(
                    core, tf.constant([[0, j, 2 + j]], tf.int32), tf.ones([1], DTYPE)
                )
        elif axis == JOINT_DIM - 1:
            core = tf.zeros([rank, width, 1], DTYPE)
            core = tf.tensor_scatter_nd_update(
                core,
                tf.constant([[0, j, 0] for j in range(width)], tf.int32),
                additive[axis],
            )
            core = tf.tensor_scatter_nd_update(
                core,
                tf.constant([[1, j, 0] for j in range(width)], tf.int32),
                tf.ones([width], DTYPE),
            )
            pair_values = pair[pair_lookup[axis - 1]]
            for left_index in range(width):
                core = tf.tensor_scatter_nd_update(
                    core,
                    tf.constant([[2 + left_index, j, 0] for j in range(width)], tf.int32),
                    pair_values[left_index],
                )
        elif axis % 2 == 0:
            core = tf.zeros([rank, width, rank], DTYPE)
            for state in (0, 1):
                core = tf.tensor_scatter_nd_update(
                    core,
                    tf.constant([[state, j, state] for j in range(width)], tf.int32),
                    tf.ones([width], DTYPE),
                )
            core = tf.tensor_scatter_nd_update(
                core,
                tf.constant([[0, j, 1] for j in range(width)], tf.int32),
                additive[axis],
            )
            for j in range(width):
                core = tf.tensor_scatter_nd_update(
                    core, tf.constant([[0, j, 2 + j]], tf.int32), tf.ones([1], DTYPE)
                )
        else:
            core = tf.zeros([rank, width, rank], DTYPE)
            for state in (0, 1):
                core = tf.tensor_scatter_nd_update(
                    core,
                    tf.constant([[state, j, state] for j in range(width)], tf.int32),
                    tf.ones([width], DTYPE),
                )
            core = tf.tensor_scatter_nd_update(
                core,
                tf.constant([[0, j, 1] for j in range(width)], tf.int32),
                additive[axis],
            )
            pair_values = pair[pair_lookup[axis - 1]]
            for left_index in range(width):
                core = tf.tensor_scatter_nd_update(
                    core,
                    tf.constant([[2 + left_index, j, 1] for j in range(width)], tf.int32),
                    pair_values[left_index],
                )
        cores.append(core)
    return tuple(cores)


def _additive_rank_two_component(
    coefficients: tf.Tensor, basis_dims: Sequence[int]
) -> tuple[tf.Tensor, ...]:
    values = tf.convert_to_tensor(coefficients, DTYPE)
    if values.shape.rank != 2 or values.shape[0] != len(basis_dims):
        raise ValueError("additive coefficients must have shape [dimension,basis_dim]")
    if any(int(width) != int(values.shape[1]) for width in basis_dims):
        raise ValueError("additive coefficients require equal axis basis dimensions")
    cores = []
    for axis, width in enumerate(basis_dims):
        one = tf.ones([int(width)], DTYPE)
        zero = tf.zeros([int(width)], DTYPE)
        coefficient = values[axis]
        if axis == 0:
            core = tf.stack([one, coefficient], axis=1)[tf.newaxis, :, :]
        elif axis == len(basis_dims) - 1:
            core = tf.stack([coefficient, one], axis=0)[:, :, tf.newaxis]
        else:
            core = tf.stack(
                [
                    tf.stack([one, coefficient], axis=1),
                    tf.stack([zero, one], axis=1),
                ],
                axis=0,
            )
        cores.append(core)
    return tuple(cores)


def embed_residual_component_at_rank(
    component: Sequence[tf.Tensor], *, target_rank: int
) -> tuple[tf.Tensor, ...]:
    """Zero-pad TT bonds without changing the represented finite function."""

    if int(target_rank) < 2:
        raise ValueError("target_rank must be at least two")
    cores = tuple(tf.convert_to_tensor(core, DTYPE) for core in component)
    if not cores:
        raise ValueError("component must be nonempty")
    if int(cores[0].shape[0]) != 1 or int(cores[-1].shape[2]) != 1:
        raise ValueError("component must have boundary rank one")
    output = []
    for axis, core in enumerate(cores):
        left_rank = 1 if axis == 0 else int(target_rank)
        right_rank = 1 if axis == len(cores) - 1 else int(target_rank)
        old_left = int(core.shape[0])
        old_right = int(core.shape[2])
        if old_left > left_rank or old_right > right_rank:
            raise ValueError("target_rank cannot truncate an existing TT bond")
        paddings = tf.constant(
            [
                [0, left_rank - old_left],
                [0, 0],
                [0, right_rank - old_right],
            ],
            tf.int32,
        )
        output.append(tf.pad(core, paddings))
    return tuple(output)


def embed_residual_component_with_connected_channels(
    component: Sequence[tf.Tensor],
    *,
    target_rank: int,
    seed: int,
    seeded_channel_epsilon: float,
) -> tuple[tf.Tensor, ...]:
    """Expand a rank-2 component with small complete paths in every new channel."""

    if not math.isfinite(float(seeded_channel_epsilon)) or float(
        seeded_channel_epsilon
    ) <= 0.0:
        raise ValueError("seeded_channel_epsilon must be positive and finite")
    embedded = list(
        embed_residual_component_at_rank(component, target_rank=int(target_rank))
    )
    old_rank = max(int(core.shape[0]) for core in component)
    if int(target_rank) <= old_rank:
        return tuple(embedded)
    epsilon = tf.constant(float(seeded_channel_epsilon), DTYPE)
    for channel in range(old_rank, int(target_rank)):
        first_noise = tf.random.stateless_normal(
            [int(embedded[0].shape[1])],
            seed=tf.constant([int(seed) + channel, 1], tf.int32),
            dtype=DTYPE,
        )
        last_noise = tf.random.stateless_normal(
            [int(embedded[-1].shape[1])],
            seed=tf.constant([int(seed) + channel, 2], tf.int32),
            dtype=DTYPE,
        )
        embedded[0] = tf.tensor_scatter_nd_update(
            embedded[0],
            tf.constant(
                [[0, basis_index, channel] for basis_index in range(int(embedded[0].shape[1]))],
                tf.int32,
            ),
            epsilon * first_noise,
        )
        for axis in range(1, len(embedded) - 1):
            embedded[axis] = tf.tensor_scatter_nd_update(
                embedded[axis],
                tf.constant(
                    [
                        [channel, basis_index, channel]
                        for basis_index in range(int(embedded[axis].shape[1]))
                    ],
                    tf.int32,
                ),
                tf.ones([int(embedded[axis].shape[1])], DTYPE),
            )
        embedded[-1] = tf.tensor_scatter_nd_update(
            embedded[-1],
            tf.constant(
                [[channel, basis_index, 0] for basis_index in range(int(embedded[-1].shape[1]))],
                tf.int32,
            ),
            epsilon * last_noise,
        )
    return tuple(embedded)


def core_tangent_to_residual_component(
    *,
    parent_cores: Sequence[tf.Tensor],
    tangent_cores: Sequence[tf.Tensor],
) -> tuple[tf.Tensor, ...]:
    """Encode a corewise product-rule tangent as one exact block TT."""

    parents = tuple(tf.convert_to_tensor(core, DTYPE) for core in parent_cores)
    tangents = tuple(tf.convert_to_tensor(core, DTYPE) for core in tangent_cores)
    if len(parents) < 2 or len(tangents) != len(parents):
        raise ValueError("parent and tangent core sequences must have equal length at least two")
    for axis, (parent, tangent) in enumerate(zip(parents, tangents)):
        if parent.shape.rank != 3 or tangent.shape != parent.shape:
            raise ValueError(f"core tangent shape mismatch at axis {axis}")
    output = []
    for axis, (parent, tangent) in enumerate(zip(parents, tangents)):
        left_rank = int(parent.shape[0])
        width = int(parent.shape[1])
        right_rank = int(parent.shape[2])
        if axis == 0:
            output.append(tf.concat([parent, tangent], axis=2))
        elif axis == len(parents) - 1:
            output.append(tf.concat([tangent, parent], axis=0))
        else:
            zero = tf.zeros([left_rank, width, right_rank], DTYPE)
            upper = tf.concat([parent, tangent], axis=2)
            lower = tf.concat([zero, parent], axis=2)
            output.append(tf.concat([upper, lower], axis=0))
    return tuple(output)


def core_tangent_banks_from_residual_components(
    *,
    parent_cores: Sequence[tf.Tensor],
    residual_components: Sequence[Sequence[tf.Tensor]],
) -> tuple[tuple[tf.Tensor, ...], ...]:
    """Invert the exact product-rule block encoding into axis-major banks."""

    parents = tuple(tf.convert_to_tensor(core, DTYPE) for core in parent_cores)
    components = tuple(
        tuple(tf.convert_to_tensor(core, DTYPE) for core in component)
        for component in residual_components
    )
    if len(parents) < 2 or len(components) != PARAMETER_DIM:
        raise ValueError("three residual components and at least two parent cores are required")
    parameter_tangents = []
    for parameter, component in enumerate(components):
        if len(component) != len(parents):
            raise ValueError(
                f"residual component {parameter} does not match the parent dimension"
            )
        tangents = []
        for axis, (parent, block) in enumerate(zip(parents, component)):
            left_rank = int(parent.shape[0])
            width = int(parent.shape[1])
            right_rank = int(parent.shape[2])
            if axis == 0:
                if block.shape != (left_rank, width, 2 * right_rank):
                    raise ValueError("first product-rule block has the wrong shape")
                tf.debugging.assert_equal(
                    block[:, :, :right_rank],
                    parent,
                    message="first product-rule parent block mismatch",
                )
                tangents.append(block[:, :, right_rank:])
            elif axis == len(parents) - 1:
                if block.shape != (2 * left_rank, width, right_rank):
                    raise ValueError("last product-rule block has the wrong shape")
                tf.debugging.assert_equal(
                    block[left_rank:, :, :],
                    parent,
                    message="last product-rule parent block mismatch",
                )
                tangents.append(block[:left_rank, :, :])
            else:
                if block.shape != (2 * left_rank, width, 2 * right_rank):
                    raise ValueError("middle product-rule block has the wrong shape")
                tf.debugging.assert_equal(
                    block[:left_rank, :, :right_rank],
                    parent,
                    message="upper-left product-rule parent block mismatch",
                )
                tf.debugging.assert_equal(
                    block[left_rank:, :, :right_rank],
                    tf.zeros_like(parent),
                    message="lower-left product-rule zero block mismatch",
                )
                tf.debugging.assert_equal(
                    block[left_rank:, :, right_rank:],
                    parent,
                    message="lower-right product-rule parent block mismatch",
                )
                tangents.append(block[:left_rank, :, right_rank:])
        parameter_tangents.append(tuple(tangents))
    return tuple(
        tuple(parameter_tangents[parameter][axis] for parameter in range(PARAMETER_DIM))
        for axis in range(len(parents))
    )


def target_informed_additive_score_initialization(
    *,
    parent: LaneBT1Artifact,
    local_points: tf.Tensor,
    target_complete_data_score: tf.Tensor,
    importance_log_weight: tf.Tensor,
    ridge_fraction: float,
    global_score_weight: float = 1.0,
    prefix_local_points: tf.Tensor | None = None,
    prefix_target_score: tf.Tensor | None = None,
    prefix_score_standard_error: tf.Tensor | None = None,
    prefix_weight: float = 0.0,
) -> TargetInformedAdditiveInitialization:
    """Fit additive rank-2 residuals through the exact origin-score operator."""

    if not math.isfinite(float(ridge_fraction)) or float(ridge_fraction) <= 0.0:
        raise ValueError("ridge_fraction must be positive and finite")
    if not math.isfinite(float(global_score_weight)) or float(
        global_score_weight
    ) <= 0.0:
        raise ValueError("global_score_weight must be positive and finite")
    if not math.isfinite(float(prefix_weight)) or float(prefix_weight) < 0.0:
        raise ValueError("prefix_weight must be finite and nonnegative")
    points = tf.convert_to_tensor(local_points, DTYPE)
    target_score = tf.convert_to_tensor(target_complete_data_score, DTYPE)
    log_weight = tf.convert_to_tensor(importance_log_weight, DTYPE)
    if points.shape.rank != 2 or points.shape[1] != JOINT_DIM:
        raise ValueError("local_points must have shape [sample,36]")
    if target_score.shape != (points.shape[0], PARAMETER_DIM):
        raise ValueError("target score must have shape [sample,3]")
    if log_weight.shape != (points.shape[0],):
        raise ValueError("importance log weight must have shape [sample]")
    basis = centered_lane_b_product_basis(
        order=parent.settings.basis_order,
        num_elems=parent.settings.basis_num_elems,
    )
    basis_dims = basis.basis_dim_tuple()
    if len(set(int(width) for width in basis_dims)) != 1:
        raise ValueError("additive initialization requires equal axis basis dimensions")
    feature_values = tf.concat(
        [basis.evaluate_axis(axis, points[:, axis]) for axis in range(JOINT_DIM)],
        axis=1,
    )
    parent_amplitude = _evaluate_component(parent.cores, basis, points)
    rho = tf.square(parent_amplitude) + tf.constant(parent.settings.tau, DTYPE)
    point_factor = 2.0 * parent_amplitude / rho
    cross_features = _parent_additive_cross_features(parent, basis)
    parent_mass = _cross_mass(parent.cores, parent.cores, basis)
    normalizer_features = 2.0 * cross_features / (
        parent_mass + tf.constant(parent.settings.tau, DTYPE)
    )
    design = (
        point_factor[:, tf.newaxis] * feature_values
        - normalizer_features[tf.newaxis, :]
    )
    maximum = tf.reduce_max(log_weight)
    scaled = tf.exp(log_weight - maximum)
    normalized = scaled / tf.reduce_sum(scaled)
    target_likelihood_score = tf.reduce_sum(
        normalized[:, tf.newaxis] * target_score, axis=0
    )
    target_point_score = target_score - target_likelihood_score[tf.newaxis, :]
    target_point_score_rms = tf.sqrt(
        tf.reduce_sum(
            normalized[:, tf.newaxis] * tf.square(target_point_score), axis=0
        )
    )
    target_point_score_rms = tf.maximum(
        target_point_score_rms, tf.constant(1e-6, DTYPE)
    )
    point_design = (
        tf.sqrt(normalized)[tf.newaxis, :, tf.newaxis]
        * design[tf.newaxis, :, :]
        / target_point_score_rms[:, tf.newaxis, tf.newaxis]
    )
    point_target = (
        tf.sqrt(normalized)[tf.newaxis, :, tf.newaxis]
        * tf.transpose(target_point_score)[:, :, tf.newaxis]
        / target_point_score_rms[:, tf.newaxis, tf.newaxis]
    )
    global_scale = tf.maximum(tf.abs(target_likelihood_score), tf.ones([3], DTYPE))
    global_design = (
        tf.sqrt(tf.constant(float(global_score_weight), DTYPE))
        * normalizer_features[tf.newaxis, tf.newaxis, :]
        / global_scale[:, tf.newaxis, tf.newaxis]
    )
    global_target = (
        tf.sqrt(tf.constant(float(global_score_weight), DTYPE))
        * target_likelihood_score[:, tf.newaxis, tf.newaxis]
        / global_scale[:, tf.newaxis, tf.newaxis]
    )
    prefix_inputs = (
        prefix_local_points,
        prefix_target_score,
        prefix_score_standard_error,
    )
    if all(value is None for value in prefix_inputs):
        if float(prefix_weight) != 0.0:
            raise ValueError("nonzero prefix_weight requires all prefix arrays")
        prefix_tolerance = tf.zeros([0, PARAMETER_DIM], DTYPE)
        prefix_design = tf.zeros([PARAMETER_DIM, 0, tf.shape(design)[1]], DTYPE)
        prefix_target = tf.zeros([PARAMETER_DIM, 0, 1], DTYPE)
        prefix_operator = tf.zeros([0, tf.shape(design)[1]], DTYPE)
        prefix_target_tensor = tf.zeros([0, PARAMETER_DIM], DTYPE)
    elif any(value is None for value in prefix_inputs):
        raise ValueError("prefix fitting requires all three prefix arrays")
    else:
        assert prefix_local_points is not None
        assert prefix_target_score is not None
        assert prefix_score_standard_error is not None
        prefix_operator = additive_prefix_score_operator(
            parent=parent, local_prefix_points=prefix_local_points
        )
        prefix_target_tensor = tf.convert_to_tensor(prefix_target_score, DTYPE)
        prefix_se = tf.convert_to_tensor(prefix_score_standard_error, DTYPE)
        expected_prefix_shape = (prefix_operator.shape[0], PARAMETER_DIM)
        if (
            prefix_target_tensor.shape != expected_prefix_shape
            or prefix_se.shape != expected_prefix_shape
        ):
            raise ValueError("prefix target and MCSE arrays have the wrong shape")
        tf.debugging.assert_non_negative(prefix_se, "prefix score MCSE")
        prefix_tolerance = 3.0 * prefix_se + tf.constant(1e-5, DTYPE)
        root_weight = tf.sqrt(tf.constant(float(prefix_weight), DTYPE))
        prefix_design = (
            root_weight
            * prefix_operator[tf.newaxis, :, :]
            / tf.transpose(prefix_tolerance)[:, :, tf.newaxis]
        )
        prefix_target = (
            root_weight
            * tf.transpose(prefix_target_tensor)[:, :, tf.newaxis]
            / tf.transpose(prefix_tolerance)[:, :, tf.newaxis]
        )
    augmented_design = tf.concat(
        [point_design, global_design, prefix_design], axis=1
    )
    augmented_target = tf.concat(
        [point_target, global_target, prefix_target], axis=1
    )
    gram = tf.linalg.matmul(
        augmented_design, augmented_design, transpose_a=True
    )
    mean_diagonal = tf.reduce_mean(tf.linalg.diag_part(gram), axis=1)
    realized_ridge = tf.maximum(
        tf.constant(float(ridge_fraction), DTYPE) * mean_diagonal,
        tf.fill([PARAMETER_DIM], tf.constant(1e-12, DTYPE)),
    )
    feature_count = tf.shape(gram)[1]
    regularized = gram + realized_ridge[:, tf.newaxis, tf.newaxis] * tf.eye(
        feature_count, batch_shape=[PARAMETER_DIM], dtype=DTYPE
    )
    rhs = tf.linalg.matmul(
        augmented_design, augmented_target, transpose_a=True
    )
    coefficients = tf.transpose(
        tf.squeeze(
            tf.linalg.cholesky_solve(tf.linalg.cholesky(regularized), rhs),
            axis=2,
        )
    )
    width = int(basis_dims[0])
    coefficient_tensor = tf.reshape(coefficients, [JOINT_DIM, width, PARAMETER_DIM])
    components = tuple(
        _additive_rank_two_component(coefficient_tensor[:, :, index], basis_dims)
        for index in range(PARAMETER_DIM)
    )
    fitted_prefix_score = tf.linalg.matmul(prefix_operator, coefficients)
    prefix_standardized_residual = tf.math.divide_no_nan(
        tf.abs(fitted_prefix_score - prefix_target_tensor), prefix_tolerance
    )
    return TargetInformedAdditiveInitialization(
        residual_components=components,
        ridge_fraction=float(ridge_fraction),
        global_score_weight=float(global_score_weight),
        realized_ridge=realized_ridge,
        target_likelihood_score=target_likelihood_score,
        target_point_score_rms=target_point_score_rms,
        design_column_rms=tf.sqrt(tf.reduce_mean(tf.square(design), axis=0)),
        coefficient_rms=tf.sqrt(tf.reduce_mean(tf.square(coefficients), axis=0)),
        prefix_weight=float(prefix_weight),
        prefix_score_tolerance=prefix_tolerance,
        training_prefix_score_standardized_residual=(
            prefix_standardized_residual
        ),
    )


def target_informed_within_region_pair_score_initialization(
    *,
    parent: LaneBT1Artifact,
    local_points: tf.Tensor,
    target_complete_data_score: tf.Tensor,
    importance_log_weight: tf.Tensor,
    ridge_fraction: float,
    global_score_weight: float,
    prefix_local_points: tf.Tensor,
    prefix_target_score: tf.Tensor,
    prefix_score_standard_error: tf.Tensor,
    prefix_weight: float,
) -> TargetInformedPairInitialization:
    """Fit additive and within-region S-I pair terms to all origin score operators."""

    for name, value in (
        ("ridge_fraction", ridge_fraction),
        ("global_score_weight", global_score_weight),
        ("prefix_weight", prefix_weight),
    ):
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{name} must be positive and finite")
    points = tf.convert_to_tensor(local_points, DTYPE)
    target_score = tf.convert_to_tensor(target_complete_data_score, DTYPE)
    log_weight = tf.convert_to_tensor(importance_log_weight, DTYPE)
    prefix_points = tf.convert_to_tensor(prefix_local_points, DTYPE)
    prefix_target = tf.convert_to_tensor(prefix_target_score, DTYPE)
    prefix_se = tf.convert_to_tensor(prefix_score_standard_error, DTYPE)
    if points.shape.rank != 2 or points.shape[1] != JOINT_DIM:
        raise ValueError("local_points must have shape [sample,36]")
    if target_score.shape != (points.shape[0], PARAMETER_DIM):
        raise ValueError("target score must have shape [sample,3]")
    if log_weight.shape != (points.shape[0],):
        raise ValueError("importance log weight must have shape [sample]")
    if prefix_points.shape.rank != 2 or prefix_points.shape[1] is None:
        raise ValueError("prefix points must have shape [point,prefix_dim]")
    expected_prefix = (prefix_points.shape[0], PARAMETER_DIM)
    if prefix_target.shape != expected_prefix or prefix_se.shape != expected_prefix:
        raise ValueError("prefix target and MCSE arrays have the wrong shape")
    basis = centered_lane_b_product_basis(
        order=parent.settings.basis_order,
        num_elems=parent.settings.basis_num_elems,
    )
    basis_dim = int(basis.basis_dim_tuple()[0])
    feature_values = _within_region_feature_values(basis, points)
    parent_amplitude = _evaluate_component(parent.cores, basis, points)
    rho = tf.square(parent_amplitude) + tf.constant(parent.settings.tau, DTYPE)
    global_cross = _within_region_cross_features(
        parent=parent, basis=basis, local_prefix_points=None
    )
    parent_mass = _cross_mass(parent.cores, parent.cores, basis)
    global_operator = 2.0 * global_cross / (
        parent_mass + tf.constant(parent.settings.tau, DTYPE)
    )
    point_operator = (
        (2.0 * parent_amplitude / rho)[:, tf.newaxis] * feature_values
        - global_operator[tf.newaxis, :]
    )
    maximum = tf.reduce_max(log_weight)
    scaled = tf.exp(log_weight - maximum)
    normalized = scaled / tf.reduce_sum(scaled)
    target_likelihood_score = tf.reduce_sum(
        normalized[:, tf.newaxis] * target_score, axis=0
    )
    target_point_score = target_score - target_likelihood_score[tf.newaxis, :]
    target_point_rms = tf.sqrt(
        tf.reduce_sum(
            normalized[:, tf.newaxis] * tf.square(target_point_score), axis=0
        )
    )
    target_point_rms = tf.maximum(target_point_rms, tf.constant(1e-6, DTYPE))
    point_design = (
        tf.sqrt(normalized)[tf.newaxis, :, tf.newaxis]
        * point_operator[tf.newaxis, :, :]
        / target_point_rms[:, tf.newaxis, tf.newaxis]
    )
    point_target = (
        tf.sqrt(normalized)[tf.newaxis, :, tf.newaxis]
        * tf.transpose(target_point_score)[:, :, tf.newaxis]
        / target_point_rms[:, tf.newaxis, tf.newaxis]
    )
    global_scale = tf.maximum(tf.abs(target_likelihood_score), tf.ones([3], DTYPE))
    global_design = (
        tf.sqrt(tf.constant(float(global_score_weight), DTYPE))
        * global_operator[tf.newaxis, tf.newaxis, :]
        / global_scale[:, tf.newaxis, tf.newaxis]
    )
    global_target = (
        tf.sqrt(tf.constant(float(global_score_weight), DTYPE))
        * target_likelihood_score[:, tf.newaxis, tf.newaxis]
        / global_scale[:, tf.newaxis, tf.newaxis]
    )
    prefix_operator = within_region_pair_prefix_score_operator(
        parent=parent, local_prefix_points=prefix_points
    )
    prefix_tolerance = 3.0 * prefix_se + tf.constant(1e-5, DTYPE)
    root_prefix_weight = tf.sqrt(tf.constant(float(prefix_weight), DTYPE))
    prefix_design = (
        root_prefix_weight
        * prefix_operator[tf.newaxis, :, :]
        / tf.transpose(prefix_tolerance)[:, :, tf.newaxis]
    )
    prefix_target_scaled = (
        root_prefix_weight
        * tf.transpose(prefix_target)[:, :, tf.newaxis]
        / tf.transpose(prefix_tolerance)[:, :, tf.newaxis]
    )
    design = tf.concat([point_design, global_design, prefix_design], axis=1)
    target = tf.concat([point_target, global_target, prefix_target_scaled], axis=1)
    gram = tf.linalg.matmul(design, design, transpose_a=True)
    mean_diagonal = tf.reduce_mean(tf.linalg.diag_part(gram), axis=1)
    realized_ridge = tf.maximum(
        tf.constant(float(ridge_fraction), DTYPE) * mean_diagonal,
        tf.fill([PARAMETER_DIM], tf.constant(1e-12, DTYPE)),
    )
    feature_count = tf.shape(gram)[1]
    regularized = gram + realized_ridge[:, tf.newaxis, tf.newaxis] * tf.eye(
        feature_count, batch_shape=[PARAMETER_DIM], dtype=DTYPE
    )
    rhs = tf.linalg.matmul(design, target, transpose_a=True)
    coefficients = tf.transpose(
        tf.squeeze(
            tf.linalg.cholesky_solve(tf.linalg.cholesky(regularized), rhs),
            axis=2,
        )
    )
    components = tuple(
        _additive_pair_rank_seven_component(
            coefficients[:, parameter_index], basis_dim=basis_dim
        )
        for parameter_index in range(PARAMETER_DIM)
    )
    fitted_prefix = tf.linalg.matmul(prefix_operator, coefficients)
    return TargetInformedPairInitialization(
        residual_components=components,
        ridge_fraction=float(ridge_fraction),
        global_score_weight=float(global_score_weight),
        prefix_weight=float(prefix_weight),
        realized_ridge=realized_ridge,
        target_likelihood_score=target_likelihood_score,
        target_point_score_rms=target_point_rms,
        coefficient_rms=tf.sqrt(tf.reduce_mean(tf.square(coefficients), axis=0)),
        training_prefix_score_standardized_residual=(
            tf.abs(fitted_prefix - prefix_target) / prefix_tolerance
        ),
    )


def fixed_rank_initial_residual_components(
    *,
    parent: LaneBT1Artifact,
    features: CenteredThetaFeatures,
    rank: int,
    seed: int,
    amplitude_scale: float = 1e-3,
    perturbation_scale: float = 1e-3,
) -> tuple[tuple[tf.Tensor, ...], ...]:
    """Build distinct, nondegenerate residual TTs at one declared fixed rank."""

    if int(rank) <= 0:
        raise ValueError("residual rank must be positive")
    for name, value in (
        ("amplitude_scale", amplitude_scale),
        ("perturbation_scale", perturbation_scale),
    ):
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{name} must be positive and finite")
    settings = replace(
        parent.settings,
        arm_id=f"centered_residual_rank_{int(rank)}_initialization",
        rank=int(rank),
    )
    basis = centered_lane_b_product_basis(
        order=settings.basis_order,
        num_elems=settings.basis_num_elems,
    )
    balanced = balanced_initial_cores(settings, basis)
    components = []
    for component_index in range(features.feature_count):
        cores = []
        for axis, core in enumerate(balanced):
            noise = tf.random.stateless_normal(
                tf.shape(core),
                seed=tf.constant(
                    [int(seed) + 104729 * component_index, axis + 1], tf.int32
                ),
                dtype=DTYPE,
            )
            value = core + tf.constant(perturbation_scale, DTYPE) * noise
            if axis == 0:
                value = tf.constant(amplitude_scale, DTYPE) * value
            cores.append(value)
        components.append(tuple(cores))
    return tuple(components)


class CenteredResidualTrainer:
    """Train only residual TTs while the admitted parent remains immutable."""

    def __init__(
        self,
        parent: LaneBT1Artifact,
        *,
        features: CenteredThetaFeatures | None = None,
        initial_residual_components: Sequence[Sequence[tf.Tensor]] | None = None,
    ) -> None:
        self.parent = parent
        self.features = features or CenteredThetaFeatures()
        self.basis = centered_lane_b_product_basis(
            order=parent.settings.basis_order,
            num_elems=parent.settings.basis_num_elems,
        )
        if initial_residual_components is None:
            initial_residual_components = tuple(
                tuple(
                    (
                        tf.constant(1e-3 * (component_index + 1), DTYPE) * core
                        if axis == 0
                        else tf.identity(core)
                    )
                    for axis, core in enumerate(parent.cores)
                )
                for component_index in range(self.features.feature_count)
            )
        if len(initial_residual_components) != self.features.feature_count:
            raise ValueError("one initial residual component is required per feature")
        variables = []
        for component_index, component in enumerate(initial_residual_components):
            if len(component) != len(parent.cores):
                raise ValueError("residual component axis count mismatch")
            variables.append(
                tuple(
                    tf.Variable(
                        tf.convert_to_tensor(core, DTYPE),
                        trainable=True,
                        name=f"residual_{component_index:02d}_core_{axis:02d}",
                    )
                    for axis, core in enumerate(component)
                )
            )
        self.residual_variables = tuple(variables)

    @property
    def trainable_variables(self) -> tuple[tf.Variable, ...]:
        return tuple(core for component in self.residual_variables for core in component)

    def position(self) -> tf.Tensor:
        return residual_components_position(self.residual_variables)

    def assign_position(self, position: tf.Tensor) -> None:
        components = residual_components_from_position(
            template_components=self.residual_variables, position=position
        )
        for variable_component, value_component in zip(
            self.residual_variables, components
        ):
            for variable, value in zip(variable_component, value_component):
                variable.assign(value)

    def _component_values(self, flat_points: tf.Tensor) -> tf.Tensor:
        parent_values = _evaluate_component(self.parent.cores, self.basis, flat_points)
        residual_values = [
            _evaluate_component(component, self.basis, flat_points)
            for component in self.residual_variables
        ]
        return tf.stack([parent_values, *residual_values], axis=1)

    def _gram(self) -> tf.Tensor:
        components = (self.parent.cores,) + self.residual_variables
        rows = [
            tf.stack([_cross_mass(left, right, self.basis) for right in components])
            for left in components
        ]
        gram = tf.stack(rows)
        return 0.5 * (gram + tf.transpose(gram))

    def absolute_density_loss(
        self,
        batch: T1ParameterDensityBatch,
        *,
        l1_weight: tf.Tensor | float = 0.0,
        l2_weight: tf.Tensor | float = 0.0,
        derivative_weight: tf.Tensor | float = 0.0,
    ) -> AbsoluteDensityLossTerms:
        tf.debugging.assert_equal(
            batch.theta[0],
            tf.zeros([PARAMETER_DIM], DTYPE),
            message="the first batch row must be the exact theta origin",
        )
        return self.absolute_density_loss_arrays(
            batch.theta,
            batch.local_points,
            tf.ones_like(batch.observation_log_density) * self.parent.shift_constant
            + batch.observation_log_density,
            l1_weight=l1_weight,
            l2_weight=l2_weight,
            derivative_points=batch.local_points[0],
            derivative_target_score=batch.complete_data_score[0],
            derivative_importance_log_weight=batch.observation_log_density[0],
            derivative_weight=derivative_weight,
        )

    def absolute_density_loss_arrays(
        self,
        theta: tf.Tensor,
        local_points: tf.Tensor,
        log_absolute_importance_weight: tf.Tensor,
        *,
        l1_weight: tf.Tensor | float = 0.0,
        l2_weight: tf.Tensor | float = 0.0,
        derivative_points: tf.Tensor | None = None,
        derivative_target_score: tf.Tensor | None = None,
        derivative_importance_log_weight: tf.Tensor | None = None,
        derivative_weight: tf.Tensor | float = 0.0,
    ) -> AbsoluteDensityLossTerms:
        parameters = tf.convert_to_tensor(theta, DTYPE)
        points = tf.convert_to_tensor(local_points, DTYPE)
        log_weight = tf.convert_to_tensor(log_absolute_importance_weight, DTYPE)
        theta_count = tf.shape(parameters)[0]
        sample_count = tf.shape(points)[1]
        flat_points = tf.reshape(points, [-1, JOINT_DIM])
        component_values = tf.reshape(
            self._component_values(flat_points),
            [theta_count, sample_count, self.features.feature_count + 1],
        )
        feature_values, _jacobian = self.features.batch_values_and_jacobian(parameters)
        weights = tf.concat(
            [tf.ones([theta_count, 1], DTYPE), feature_values], axis=1
        )
        amplitude = tf.einsum("tnc,tc->tn", component_values, weights)
        rho = tf.square(amplitude) + tf.constant(self.parent.settings.tau, DTYPE)
        gram = self._gram()
        child_mass = tf.einsum("ti,ij,tj->t", weights, gram, weights) + tf.constant(
            self.parent.settings.tau, DTYPE
        )
        absolute_weights = tf.exp(log_weight)
        target_log_term = tf.reduce_mean(
            absolute_weights * tf.math.log(rho), axis=1
        )
        mass_estimate = tf.reduce_mean(absolute_weights, axis=1)
        centered_mass = absolute_weights - mass_estimate[:, tf.newaxis]
        mass_se = tf.sqrt(
            tf.reduce_sum(tf.square(centered_mass), axis=1)
            / tf.cast(sample_count * (sample_count - 1), DTYPE)
        )
        data_loss = tf.reduce_mean(child_mass - target_log_term)
        derivative_weight_tensor = tf.cast(derivative_weight, DTYPE)
        derivative_inputs = (
            derivative_points,
            derivative_target_score,
            derivative_importance_log_weight,
        )
        if all(value is None for value in derivative_inputs):
            tf.debugging.assert_equal(
                derivative_weight_tensor,
                tf.zeros([], DTYPE),
                message="nonzero derivative weight requires explicit origin score arrays",
            )
            derivative_loss = tf.zeros([], DTYPE)
        elif any(value is None for value in derivative_inputs):
            raise ValueError("derivative matching requires all three origin arrays")
        else:
            assert derivative_points is not None
            assert derivative_target_score is not None
            assert derivative_importance_log_weight is not None
            derivative_loss = self.origin_point_score_loss_arrays(
                derivative_points,
                derivative_target_score,
                derivative_importance_log_weight,
            )
        l1 = tf.add_n([tf.reduce_sum(tf.abs(value)) for value in self.trainable_variables])
        l2 = tf.add_n([tf.reduce_sum(tf.square(value)) for value in self.trainable_variables])
        total = (
            data_loss
            + derivative_weight_tensor * derivative_loss
            + tf.cast(l1_weight, DTYPE) * l1
            + tf.cast(l2_weight, DTYPE) * l2
        )
        return AbsoluteDensityLossTerms(
            total_loss=total,
            absolute_density_loss=data_loss,
            derivative_matching_loss=derivative_loss,
            exact_child_mass=child_mass,
            target_log_density_term=target_log_term,
            target_mass_estimate=mass_estimate,
            target_mass_standard_error=mass_se,
            minimum_rho=tf.reduce_min(rho),
        )

    def origin_point_score_loss_arrays(
        self,
        local_points: tf.Tensor,
        target_complete_data_score: tf.Tensor,
        importance_log_weight: tf.Tensor,
    ) -> tf.Tensor:
        metrics = self.origin_point_score_metrics_arrays(
            local_points,
            target_complete_data_score,
            importance_log_weight,
        )
        return tf.reduce_sum(tf.square(metrics["normalized_score_residual_rms"]))

    def origin_global_score_metrics_arrays(
        self,
        target_score: tf.Tensor,
        score_standard_error: tf.Tensor,
    ) -> dict[str, tf.Tensor]:
        target = tf.reshape(tf.convert_to_tensor(target_score, DTYPE), [PARAMETER_DIM])
        standard_error = tf.reshape(
            tf.convert_to_tensor(score_standard_error, DTYPE), [PARAMETER_DIM]
        )
        tf.debugging.assert_non_negative(standard_error, "global score MCSE")
        components = (self.parent.cores,) + self.residual_variables
        cross = tf.stack(
            [_cross_mass(components[0], component, self.basis) for component in components[1:]]
        )
        parent_mass = _cross_mass(components[0], components[0], self.basis)
        child_score = 2.0 * cross / (
            parent_mass + tf.constant(self.parent.settings.tau, DTYPE)
        )
        tolerance = 3.0 * standard_error + tf.constant(1e-5, DTYPE)
        standardized = (child_score - target) / tolerance
        return {
            "loss": tf.reduce_mean(tf.square(standardized)),
            "child_score": child_score,
            "target_score": target,
            "score_standard_error": standard_error,
            "standardized_residual": tf.abs(standardized),
        }

    def origin_prefix_score_metrics_arrays(
        self,
        local_prefix_points: tf.Tensor,
        target_score: tf.Tensor,
        score_standard_error: tf.Tensor,
    ) -> dict[str, tf.Tensor]:
        points = tf.convert_to_tensor(local_prefix_points, DTYPE)
        target = tf.convert_to_tensor(target_score, DTYPE)
        standard_error = tf.convert_to_tensor(score_standard_error, DTYPE)
        if points.shape.rank != 2 or points.shape[1] is None:
            raise ValueError("prefix points must have shape [point,prefix_dim]")
        expected = (points.shape[0], PARAMETER_DIM)
        if target.shape != expected or standard_error.shape != expected:
            raise ValueError("prefix target and MCSE arrays have the wrong shape")
        tf.debugging.assert_non_negative(standard_error, "prefix score MCSE")
        components = (self.parent.cores,) + self.residual_variables
        parent_prefix = _cross_prefix_values(
            components[0], components[0], self.basis, points
        ) + tf.constant(self.parent.settings.tau, DTYPE)
        prefix_cross = tf.stack(
            [
                _cross_prefix_values(components[0], component, self.basis, points)
                for component in components[1:]
            ],
            axis=1,
        )
        parent_mass = _cross_mass(components[0], components[0], self.basis)
        global_cross = tf.stack(
            [_cross_mass(components[0], component, self.basis) for component in components[1:]]
        )
        child_score = (
            2.0 * prefix_cross / parent_prefix[:, tf.newaxis]
            - 2.0
            * global_cross[tf.newaxis, :]
            / (parent_mass + tf.constant(self.parent.settings.tau, DTYPE))
        )
        tolerance = 3.0 * standard_error + tf.constant(1e-5, DTYPE)
        standardized = (child_score - target) / tolerance
        return {
            "loss": tf.reduce_mean(tf.square(standardized)),
            "child_score": child_score,
            "target_score": target,
            "score_standard_error": standard_error,
            "standardized_residual": tf.abs(standardized),
        }

    def origin_point_score_metrics_arrays(
        self,
        local_points: tf.Tensor,
        target_complete_data_score: tf.Tensor,
        importance_log_weight: tf.Tensor,
    ) -> dict[str, tf.Tensor]:
        points = tf.convert_to_tensor(local_points, DTYPE)
        target_score = tf.convert_to_tensor(target_complete_data_score, DTYPE)
        log_weight = tf.convert_to_tensor(importance_log_weight, DTYPE)
        component_values = self._component_values(points)
        parent_amplitude = component_values[:, 0]
        residual_values = component_values[:, 1:]
        rho = tf.square(parent_amplitude) + tf.constant(
            self.parent.settings.tau, DTYPE
        )
        child_unnormalized_score = (
            2.0
            * parent_amplitude[:, tf.newaxis]
            * residual_values
            / rho[:, tf.newaxis]
        )
        maximum = tf.reduce_max(log_weight)
        scaled = tf.exp(log_weight - maximum)
        normalized = scaled / tf.reduce_sum(scaled)
        target_normalizer_score = tf.reduce_sum(
            normalized[:, tf.newaxis] * target_score, axis=0
        )
        target_point_score = target_score - target_normalizer_score[tf.newaxis, :]
        # The exact child normalizer score follows from cross-component masses.
        components = (self.parent.cores,) + self.residual_variables
        cross = tf.stack(
            [_cross_mass(components[0], component, self.basis) for component in components[1:]]
        )
        parent_mass = _cross_mass(components[0], components[0], self.basis)
        child_normalizer_score = 2.0 * cross / (
            parent_mass + tf.constant(self.parent.settings.tau, DTYPE)
        )
        child_point_score = (
            child_unnormalized_score - child_normalizer_score[tf.newaxis, :]
        )
        child_point_score_mean = tf.reduce_sum(
            normalized[:, tf.newaxis] * child_point_score, axis=0
        )
        child_point_score_standard_deviation = tf.sqrt(
            tf.reduce_sum(
                normalized[:, tf.newaxis]
                * tf.square(
                    child_point_score - child_point_score_mean[tf.newaxis, :]
                ),
                axis=0,
            )
        )
        residual = child_point_score - target_point_score
        scale = tf.sqrt(
            tf.reduce_sum(
                normalized[:, tf.newaxis] * tf.square(target_point_score), axis=0
            )
        )
        scale = tf.maximum(scale, tf.constant(1e-6, DTYPE))
        residual_rms = tf.sqrt(
            tf.reduce_sum(
                normalized[:, tf.newaxis] * tf.square(residual), axis=0
            )
        )
        normalized_rms = residual_rms / scale
        loss = tf.reduce_sum(
            normalized[:, tf.newaxis]
            * tf.square(residual / scale[tf.newaxis, :])
        )
        return {
            "loss": loss,
            "target_likelihood_score": target_normalizer_score,
            "child_likelihood_score": child_normalizer_score,
            "target_point_score_rms": scale,
            "score_residual_rms": residual_rms,
            "normalized_score_residual_rms": normalized_rms,
            "child_point_score_standard_deviation": (
                child_point_score_standard_deviation
            ),
            "importance_effective_sample_size": tf.math.reciprocal(
                tf.reduce_sum(tf.square(normalized))
            ),
        }

    def heldout_metrics(self, batch: T1ParameterDensityBatch) -> dict[str, tf.Tensor]:
        parameters = batch.theta
        theta_count = tf.shape(parameters)[0]
        sample_count = tf.shape(batch.local_points)[1]
        components = tf.reshape(
            self._component_values(tf.reshape(batch.local_points, [-1, JOINT_DIM])),
            [theta_count, sample_count, self.features.feature_count + 1],
        )
        feature_values, _jacobian = self.features.batch_values_and_jacobian(parameters)
        weights = tf.concat([tf.ones([theta_count, 1], DTYPE), feature_values], axis=1)
        amplitude = tf.einsum("tnc,tc->tn", components, weights)
        rho = tf.square(amplitude) + tf.constant(self.parent.settings.tau, DTYPE)
        gram = self._gram()
        child_log_mass = tf.math.log(
            tf.einsum("ti,ij,tj->t", weights, gram, weights)
            + tf.constant(self.parent.settings.tau, DTYPE)
        )
        log_absolute_weight = self.parent.shift_constant + batch.observation_log_density
        maximum = tf.reduce_max(log_absolute_weight, axis=1, keepdims=True)
        scaled = tf.exp(log_absolute_weight - maximum)
        normalized = scaled / tf.reduce_sum(scaled, axis=1, keepdims=True)
        mean_scaled = tf.reduce_mean(scaled, axis=1)
        target_log_mass = tf.squeeze(maximum, axis=1) + tf.math.log(
            mean_scaled
        )
        centered_scaled = scaled - mean_scaled[:, tf.newaxis]
        scaled_standard_error = tf.sqrt(
            tf.reduce_sum(tf.square(centered_scaled), axis=1)
            / tf.cast(sample_count * (sample_count - 1), DTYPE)
        )
        log_density_residual = (
            tf.math.log(rho)
            - child_log_mass[:, tf.newaxis]
            - batch.target_log_density_reference
            + target_log_mass[:, tf.newaxis]
        )
        centered_log_rms = tf.sqrt(
            tf.reduce_sum(normalized * tf.square(log_density_residual), axis=1)
        )
        return {
            "child_log_mass": child_log_mass,
            "target_log_mass": target_log_mass,
            "target_log_mass_standard_error": scaled_standard_error / mean_scaled,
            "absolute_log_mass_error": tf.abs(child_log_mass - target_log_mass),
            "normalized_log_density_rms": centered_log_rms,
            "importance_effective_sample_size": tf.math.reciprocal(
                tf.reduce_sum(tf.square(normalized), axis=1)
            ),
            "minimum_rho": tf.reduce_min(rho, axis=1),
        }

    def freeze_child(self) -> LaneBCenteredResidualChild:
        residuals = tuple(
            tuple(tf.identity(core) for core in component)
            for component in self.residual_variables
        )
        return LaneBCenteredResidualChild(
            parent=self.parent,
            residual_components=residuals,
            features=self.features,
        )


class CoreAffineTangentTrainer:
    """Current-basis product-rule tangent with parent-shaped trainable cores."""

    def __init__(
        self,
        parent: LaneBT1Artifact,
        *,
        initial_tangent_banks: Sequence[Sequence[tf.Tensor]] | None = None,
    ) -> None:
        if not isinstance(parent, LaneBT1Artifact):
            raise TypeError("core-affine tangent training requires a T1 parent")
        self.parent = parent
        self.features = CenteredThetaFeatures()
        self.basis = centered_lane_b_product_basis(
            order=parent.settings.basis_order,
            num_elems=parent.settings.basis_num_elems,
        )
        if initial_tangent_banks is None:
            initial = tuple(
                tuple(tf.zeros_like(core) for _ in range(PARAMETER_DIM))
                for core in parent.cores
            )
        else:
            initial = tuple(
                tuple(tf.convert_to_tensor(value, DTYPE) for value in bank)
                for bank in initial_tangent_banks
            )
        if len(initial) != len(parent.cores):
            raise ValueError("core-affine tangent bank must match the parent dimension")
        variables = []
        for axis, (parent_core, bank) in enumerate(zip(parent.cores, initial)):
            if len(bank) != PARAMETER_DIM:
                raise ValueError("each core-affine bank must contain three tangents")
            row = []
            for parameter, value in enumerate(bank):
                if value.shape != parent_core.shape:
                    raise ValueError(
                        f"core-affine tangent shape mismatch at axis {axis}, parameter {parameter}"
                    )
                row.append(
                    tf.Variable(
                        value,
                        trainable=True,
                        name=f"core_affine_axis_{axis:02d}_parameter_{parameter}",
                    )
                )
            variables.append(tuple(row))
        self.tangent_variables = tuple(variables)

    @property
    def trainable_variables(self) -> tuple[tf.Variable, ...]:
        return tuple(
            variable for bank in self.tangent_variables for variable in bank
        )

    def position(self) -> tf.Tensor:
        return tf.concat(
            [tf.reshape(variable, [-1]) for variable in self.trainable_variables],
            axis=0,
        )

    def assign_position(self, position: tf.Tensor) -> None:
        banks = core_affine_tangent_banks_from_position(
            parent_cores=self.parent.cores, position=position
        )
        for variable_bank, value_bank in zip(self.tangent_variables, banks):
            for variable, value in zip(variable_bank, value_bank):
                variable.assign(value)

    @property
    def residual_variables(self) -> tuple[tuple[tf.Tensor, ...], ...]:
        return tuple(
            core_tangent_to_residual_component(
                parent_cores=self.parent.cores,
                tangent_cores=tuple(
                    bank[parameter] for bank in self.tangent_variables
                ),
            )
            for parameter in range(PARAMETER_DIM)
        )

    def _delegate(self) -> CenteredResidualTrainer:
        # Operator methods only inspect these fields; gradients flow through
        # the component tensors built from the structured variables.
        delegate = object.__new__(CenteredResidualTrainer)
        delegate.parent = self.parent
        delegate.features = self.features
        delegate.basis = self.basis
        delegate.residual_variables = self.residual_variables
        return delegate

    def origin_point_score_metrics_arrays(
        self,
        local_points: tf.Tensor,
        target_complete_data_score: tf.Tensor,
        importance_log_weight: tf.Tensor,
    ) -> dict[str, tf.Tensor]:
        return self._delegate().origin_point_score_metrics_arrays(
            local_points, target_complete_data_score, importance_log_weight
        )

    def origin_global_score_metrics_arrays(
        self, target_score: tf.Tensor, score_standard_error: tf.Tensor
    ) -> dict[str, tf.Tensor]:
        return self._delegate().origin_global_score_metrics_arrays(
            target_score, score_standard_error
        )

    def origin_prefix_score_metrics_arrays(
        self,
        local_prefix_points: tf.Tensor,
        target_score: tf.Tensor,
        score_standard_error: tf.Tensor,
    ) -> dict[str, tf.Tensor]:
        return self._delegate().origin_prefix_score_metrics_arrays(
            local_prefix_points, target_score, score_standard_error
        )

    def heldout_metrics(self, batch: T1ParameterDensityBatch) -> dict[str, tf.Tensor]:
        return self._delegate().heldout_metrics(batch)

    def freeze_child(self) -> LaneBCenteredResidualChild:
        return LaneBCenteredResidualChild(
            parent=self.parent,
            residual_components=tuple(
                tuple(tf.identity(core) for core in component)
                for component in self.residual_variables
            ),
            features=self.features,
        )


@dataclass(frozen=True)
class QuadraticConjugateGradientResult:
    """Finite matrix-free solve state for an exactly quadratic callback."""

    position: tf.Tensor
    converged: bool
    failed: bool
    num_iterations: int
    initial_residual_norm: tf.Tensor
    residual_norm: tf.Tensor
    relative_residual_norm: tf.Tensor
    minimum_curvature: tf.Tensor
    trace: tuple[tuple[int, tf.Tensor, tf.Tensor], ...]


def solve_quadratic_value_gradient_with_conjugate_gradient(
    value_and_gradient: Callable[[tf.Tensor], tuple[tf.Tensor, tf.Tensor]],
    *,
    initial_position: tf.Tensor,
    tolerance: float,
    max_iterations: int,
    trace_interval: int = 16,
) -> QuadraticConjugateGradientResult:
    """Solve the normal equation defined by an affine quadratic gradient."""

    if not math.isfinite(float(tolerance)) or float(tolerance) <= 0.0:
        raise ValueError("conjugate-gradient tolerance must be positive and finite")
    if int(max_iterations) <= 0:
        raise ValueError("conjugate-gradient max_iterations must be positive")
    if int(trace_interval) <= 0:
        raise ValueError("conjugate-gradient trace_interval must be positive")
    zero = tf.zeros_like(tf.convert_to_tensor(initial_position, DTYPE))
    _zero_value, affine_gradient = value_and_gradient(zero)
    rhs = -affine_gradient

    def hessian_action(vector: tf.Tensor) -> tf.Tensor:
        _value, gradient = value_and_gradient(tf.convert_to_tensor(vector, DTYPE))
        action = gradient - affine_gradient
        tf.debugging.assert_all_finite(action, "quadratic Hessian action")
        return action

    position = tf.identity(tf.convert_to_tensor(initial_position, DTYPE))
    residual = rhs - hessian_action(position)
    direction = tf.identity(residual)
    residual_squared = tf.reduce_sum(tf.square(residual))
    initial_residual_norm = tf.sqrt(residual_squared)
    rhs_norm = tf.linalg.norm(rhs)
    residual_scale = tf.maximum(rhs_norm, tf.constant(1.0, DTYPE))
    threshold = tf.constant(float(tolerance), DTYPE) * residual_scale
    minimum_curvature = tf.constant(float("inf"), DTYPE)
    trace: list[tuple[int, tf.Tensor, tf.Tensor]] = [
        (0, initial_residual_norm, initial_residual_norm / residual_scale)
    ]
    converged = bool((initial_residual_norm <= threshold).numpy())
    failed = False
    completed = 0
    for iteration in range(1, int(max_iterations) + 1):
        if converged or failed:
            break
        action = hessian_action(direction)
        curvature = tf.tensordot(direction, action, axes=1)
        minimum_curvature = tf.minimum(minimum_curvature, curvature)
        if not bool(tf.math.is_finite(curvature).numpy()) or float(curvature) <= 0.0:
            failed = True
            break
        step = residual_squared / curvature
        position = position + step * direction
        residual = residual - step * action
        next_residual_squared = tf.reduce_sum(tf.square(residual))
        residual_norm = tf.sqrt(next_residual_squared)
        relative = residual_norm / residual_scale
        completed = iteration
        if iteration % int(trace_interval) == 0 or iteration == int(max_iterations):
            trace.append((iteration, residual_norm, relative))
        converged = bool((residual_norm <= threshold).numpy())
        if converged:
            if trace[-1][0] != iteration:
                trace.append((iteration, residual_norm, relative))
            residual_squared = next_residual_squared
            break
        beta = next_residual_squared / residual_squared
        direction = residual + beta * direction
        residual_squared = next_residual_squared
    final_residual_norm = tf.sqrt(residual_squared)
    tf.debugging.assert_all_finite(position, "conjugate-gradient position")
    tf.debugging.assert_all_finite(final_residual_norm, "conjugate-gradient residual")
    return QuadraticConjugateGradientResult(
        position=position,
        converged=converged,
        failed=failed,
        num_iterations=completed,
        initial_residual_norm=initial_residual_norm,
        residual_norm=final_residual_norm,
        relative_residual_norm=final_residual_norm / residual_scale,
        minimum_curvature=minimum_curvature,
        trace=tuple(trace),
    )


def core_affine_tangent_banks_from_position(
    *, parent_cores: Sequence[tf.Tensor], position: tf.Tensor
) -> tuple[tuple[tf.Tensor, ...], ...]:
    """Unflatten axis-major, parameter-minor core-affine tangent coefficients."""

    parents = tuple(tf.convert_to_tensor(core, DTYPE) for core in parent_cores)
    if not parents:
        raise ValueError("core-affine parent cores must be nonempty")
    sizes = [int(tf.TensorShape(core.shape).num_elements()) for core in parents]
    expected = PARAMETER_DIM * sum(sizes)
    flat = tf.reshape(tf.convert_to_tensor(position, DTYPE), [-1])
    if flat.shape[0] is not None and int(flat.shape[0]) != expected:
        raise ValueError("core-affine position has the wrong static size")
    tf.debugging.assert_equal(
        tf.size(flat), expected, message="core-affine position has the wrong size"
    )
    split_sizes = [size for size in sizes for _ in range(PARAMETER_DIM)]
    chunks = tf.split(flat, split_sizes, axis=0)
    output = []
    cursor = 0
    for parent in parents:
        bank = []
        for _ in range(PARAMETER_DIM):
            bank.append(tf.reshape(chunks[cursor], parent.shape))
            cursor += 1
        output.append(tuple(bank))
    return tuple(output)


def residual_components_position(
    residual_components: Sequence[Sequence[tf.Tensor]],
) -> tf.Tensor:
    """Flatten component-major, axis-minor residual-TT cores."""

    components = tuple(tuple(component) for component in residual_components)
    if not components or any(not component for component in components):
        raise ValueError("residual components must be nonempty")
    return tf.concat(
        [
            tf.reshape(tf.convert_to_tensor(core, DTYPE), [-1])
            for component in components
            for core in component
        ],
        axis=0,
    )


def residual_components_from_position(
    *,
    template_components: Sequence[Sequence[tf.Tensor]],
    position: tf.Tensor,
) -> tuple[tuple[tf.Tensor, ...], ...]:
    """Reconstruct residual-TT cores from a flattened position."""

    templates = tuple(
        tuple(tf.convert_to_tensor(core, DTYPE) for core in component)
        for component in template_components
    )
    if not templates or any(not component for component in templates):
        raise ValueError("template residual components must be nonempty")
    sizes = [
        int(tf.TensorShape(core.shape).num_elements())
        for component in templates
        for core in component
    ]
    flat = tf.reshape(tf.convert_to_tensor(position, DTYPE), [-1])
    expected = sum(sizes)
    if flat.shape[0] is not None and int(flat.shape[0]) != expected:
        raise ValueError("residual-TT position has the wrong static size")
    tf.debugging.assert_equal(
        tf.size(flat), expected, message="residual-TT position has the wrong size"
    )
    chunks = tf.split(flat, sizes, axis=0)
    output = []
    cursor = 0
    for component in templates:
        cores = []
        for core in component:
            cores.append(tf.reshape(chunks[cursor], core.shape))
            cursor += 1
        output.append(tuple(cores))
    return tuple(output)


def core_affine_released_coordinate_mask(
    parent_cores: Sequence[tf.Tensor],
) -> tf.Tensor:
    """Mark full-TT coordinates fixed by the product-rule block manifold."""

    parents = tuple(tf.convert_to_tensor(core, DTYPE) for core in parent_cores)
    if len(parents) < 2:
        raise ValueError("at least two parent cores are required")
    component_mask = []
    for axis, parent in enumerate(parents):
        left_rank = int(parent.shape[0])
        width = int(parent.shape[1])
        right_rank = int(parent.shape[2])
        fixed = tf.ones_like(parent, dtype=tf.bool)
        free = tf.zeros_like(parent, dtype=tf.bool)
        if axis == 0:
            mask = tf.concat([fixed, free], axis=2)
        elif axis == len(parents) - 1:
            mask = tf.concat([free, fixed], axis=0)
        else:
            upper = tf.concat([fixed, free], axis=2)
            lower = tf.concat([fixed, fixed], axis=2)
            mask = tf.concat([upper, lower], axis=0)
        expected = (
            (left_rank, width, 2 * right_rank)
            if axis == 0
            else (2 * left_rank, width, right_rank)
            if axis == len(parents) - 1
            else (2 * left_rank, width, 2 * right_rank)
        )
        if mask.shape != expected:
            raise ValueError("released-coordinate mask shape mismatch")
        component_mask.append(mask)
    return tf.concat(
        [
            tf.reshape(mask, [-1])
            for _parameter in range(PARAMETER_DIM)
            for mask in component_mask
        ],
        axis=0,
    )


def core_affine_origin_total_score_loss_arrays(
    *,
    parent: LaneBT1Artifact,
    position: tf.Tensor,
    point_local_points: tf.Tensor,
    point_target_score: tf.Tensor,
    point_importance_log_weight: tf.Tensor,
    global_target_score: tf.Tensor,
    global_score_standard_error: tf.Tensor,
    prefix_local_points: tf.Tensor,
    prefix_target_score: tf.Tensor,
    prefix_score_standard_error: tf.Tensor,
    point_weight: float,
    global_weight: float,
    prefix_weight: float,
    l2_weight: float,
    basis: ProductBasis | None = None,
) -> tuple[tf.Tensor, dict[str, tf.Tensor], dict[str, tf.Tensor], dict[str, tf.Tensor]]:
    """Evaluate the full convex origin-score objective from a flat position."""

    banks = core_affine_tangent_banks_from_position(
        parent_cores=parent.cores, position=position
    )
    components = tuple(
        core_tangent_to_residual_component(
            parent_cores=parent.cores,
            tangent_cores=tuple(bank[parameter] for bank in banks),
        )
        for parameter in range(PARAMETER_DIM)
    )
    delegate = object.__new__(CenteredResidualTrainer)
    delegate.parent = parent
    delegate.features = CenteredThetaFeatures()
    delegate.basis = basis or centered_lane_b_product_basis(
        order=parent.settings.basis_order, num_elems=parent.settings.basis_num_elems
    )
    delegate.residual_variables = components
    point = delegate.origin_point_score_metrics_arrays(
        point_local_points, point_target_score, point_importance_log_weight
    )
    global_metrics = delegate.origin_global_score_metrics_arrays(
        global_target_score, global_score_standard_error
    )
    prefix = delegate.origin_prefix_score_metrics_arrays(
        prefix_local_points, prefix_target_score, prefix_score_standard_error
    )
    total = (
        tf.constant(float(point_weight), DTYPE) * point["loss"]
        + tf.constant(float(global_weight), DTYPE) * global_metrics["loss"]
        + tf.constant(float(prefix_weight), DTYPE) * prefix["loss"]
        + tf.constant(float(l2_weight), DTYPE)
        * tf.reduce_sum(tf.square(tf.convert_to_tensor(position, DTYPE)))
    )
    return total, point, global_metrics, prefix


def make_compiled_core_affine_total_score_value_and_gradient(
    *,
    parent: LaneBT1Artifact,
    point_local_points: tf.Tensor,
    point_target_score: tf.Tensor,
    point_importance_log_weight: tf.Tensor,
    global_target_score: tf.Tensor,
    global_score_standard_error: tf.Tensor,
    prefix_local_points: tf.Tensor,
    prefix_target_score: tf.Tensor,
    prefix_score_standard_error: tf.Tensor,
    point_weight: float,
    global_weight: float,
    prefix_weight: float,
    l2_weight: float,
):
    """Compile a functional L-BFGS callback for the full score pool."""

    for name, value in (
        ("point_weight", point_weight),
        ("global_weight", global_weight),
        ("prefix_weight", prefix_weight),
    ):
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{name} must be positive and finite")
    if not math.isfinite(float(l2_weight)) or float(l2_weight) < 0.0:
        raise ValueError("l2_weight must be finite and nonnegative")
    frozen_basis = centered_lane_b_product_basis(
        order=parent.settings.basis_order,
        num_elems=parent.settings.basis_num_elems,
    )

    @tf.function(jit_compile=True, reduce_retracing=True)
    def value_and_gradient(position: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        values = tf.convert_to_tensor(position, DTYPE)
        with tf.GradientTape() as tape:
            tape.watch(values)
            total, _point, _global, _prefix = (
                core_affine_origin_total_score_loss_arrays(
                    parent=parent,
                    position=values,
                    point_local_points=point_local_points,
                    point_target_score=point_target_score,
                    point_importance_log_weight=point_importance_log_weight,
                    global_target_score=global_target_score,
                    global_score_standard_error=global_score_standard_error,
                    prefix_local_points=prefix_local_points,
                    prefix_target_score=prefix_target_score,
                    prefix_score_standard_error=prefix_score_standard_error,
                    point_weight=point_weight,
                    global_weight=global_weight,
                    prefix_weight=prefix_weight,
                    l2_weight=l2_weight,
                    basis=frozen_basis,
                )
            )
        gradient = tape.gradient(total, values)
        if gradient is None:
            raise ValueError("core-affine functional objective has a missing gradient")
        tf.debugging.assert_all_finite(total, "core-affine functional objective")
        tf.debugging.assert_all_finite(gradient, "core-affine functional gradient")
        return total, gradient

    return value_and_gradient


def make_compiled_core_affine_gate_minimax_value_and_gradient(
    *,
    parent: LaneBT1Artifact,
    point_local_points: tf.Tensor,
    point_target_score: tf.Tensor,
    point_importance_log_weight: tf.Tensor,
    global_target_score: tf.Tensor,
    global_score_standard_error: tf.Tensor,
    prefix_local_points: tf.Tensor,
    prefix_target_score: tf.Tensor,
    prefix_score_standard_error: tf.Tensor,
    temperature: float,
    l2_weight: float,
):
    """Compile the smooth maximum of the existing normalized score gates."""

    if not math.isfinite(float(temperature)) or float(temperature) <= 0.0:
        raise ValueError("smooth-minimax temperature must be positive and finite")
    if not math.isfinite(float(l2_weight)) or float(l2_weight) < 0.0:
        raise ValueError("l2_weight must be finite and nonnegative")
    frozen_basis = centered_lane_b_product_basis(
        order=parent.settings.basis_order,
        num_elems=parent.settings.basis_num_elems,
    )
    inverse_temperature = tf.constant(1.0 / float(temperature), DTYPE)
    temperature_tensor = tf.constant(float(temperature), DTYPE)

    @tf.function(jit_compile=True, reduce_retracing=True)
    def value_and_gradient(position: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        values = tf.convert_to_tensor(position, DTYPE)
        with tf.GradientTape() as tape:
            tape.watch(values)
            _total, point, global_metrics, prefix = (
                core_affine_origin_total_score_loss_arrays(
                    parent=parent,
                    position=values,
                    point_local_points=point_local_points,
                    point_target_score=point_target_score,
                    point_importance_log_weight=point_importance_log_weight,
                    global_target_score=global_target_score,
                    global_score_standard_error=global_score_standard_error,
                    prefix_local_points=prefix_local_points,
                    prefix_target_score=prefix_target_score,
                    prefix_score_standard_error=prefix_score_standard_error,
                    point_weight=1.0,
                    global_weight=1.0,
                    prefix_weight=1.0,
                    l2_weight=0.0,
                    basis=frozen_basis,
                )
            )
            gate_squared = tf.concat(
                [
                    tf.square(point["normalized_score_residual_rms"] / 0.90),
                    tf.square(global_metrics["standardized_residual"]),
                    tf.reshape(tf.square(prefix["standardized_residual"]), [-1]),
                ],
                axis=0,
            )
            smooth_maximum = inverse_temperature * tf.reduce_logsumexp(
                temperature_tensor * gate_squared
            )
            total = smooth_maximum + tf.constant(float(l2_weight), DTYPE) * tf.reduce_sum(
                tf.square(values)
            )
        gradient = tape.gradient(total, values)
        if gradient is None:
            raise ValueError("core-affine smooth-minimax objective has a missing gradient")
        tf.debugging.assert_all_finite(total, "core-affine smooth-minimax objective")
        tf.debugging.assert_all_finite(gradient, "core-affine smooth-minimax gradient")
        return total, gradient

    return value_and_gradient


def make_compiled_full_tt_gate_minimax_value_and_gradient(
    *,
    parent: LaneBT1Artifact,
    template_components: Sequence[Sequence[tf.Tensor]],
    reference_position: tf.Tensor,
    point_local_points: tf.Tensor,
    point_target_score: tf.Tensor,
    point_importance_log_weight: tf.Tensor,
    global_target_score: tf.Tensor,
    global_score_standard_error: tf.Tensor,
    prefix_local_points: tf.Tensor,
    prefix_target_score: tf.Tensor,
    prefix_score_standard_error: tf.Tensor,
    temperature: float,
    l2_displacement_weight: float,
):
    """Compile gate-scaled minimax fitting over unrestricted residual TT cores."""

    if not math.isfinite(float(temperature)) or float(temperature) <= 0.0:
        raise ValueError("smooth-minimax temperature must be positive and finite")
    if (
        not math.isfinite(float(l2_displacement_weight))
        or float(l2_displacement_weight) < 0.0
    ):
        raise ValueError("l2 displacement weight must be finite and nonnegative")
    templates = tuple(
        tuple(tf.convert_to_tensor(core, DTYPE) for core in component)
        for component in template_components
    )
    reference = tf.reshape(tf.convert_to_tensor(reference_position, DTYPE), [-1])
    reconstructed = residual_components_from_position(
        template_components=templates, position=reference
    )
    if any(
        actual.shape != expected.shape
        for actual_component, expected_component in zip(reconstructed, templates)
        for actual, expected in zip(actual_component, expected_component)
    ):
        raise ValueError("full-TT reference position does not match the template")
    frozen_basis = centered_lane_b_product_basis(
        order=parent.settings.basis_order,
        num_elems=parent.settings.basis_num_elems,
    )
    inverse_temperature = tf.constant(1.0 / float(temperature), DTYPE)
    temperature_tensor = tf.constant(float(temperature), DTYPE)

    @tf.function(jit_compile=True, reduce_retracing=True)
    def value_and_gradient(position: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        values = tf.reshape(tf.convert_to_tensor(position, DTYPE), [-1])
        with tf.GradientTape() as tape:
            tape.watch(values)
            components = residual_components_from_position(
                template_components=templates, position=values
            )
            delegate = object.__new__(CenteredResidualTrainer)
            delegate.parent = parent
            delegate.features = CenteredThetaFeatures()
            delegate.basis = frozen_basis
            delegate.residual_variables = components
            point = delegate.origin_point_score_metrics_arrays(
                point_local_points,
                point_target_score,
                point_importance_log_weight,
            )
            global_metrics = delegate.origin_global_score_metrics_arrays(
                global_target_score, global_score_standard_error
            )
            prefix = delegate.origin_prefix_score_metrics_arrays(
                prefix_local_points,
                prefix_target_score,
                prefix_score_standard_error,
            )
            gate_squared = tf.concat(
                [
                    tf.square(point["normalized_score_residual_rms"] / 0.90),
                    tf.square(global_metrics["standardized_residual"]),
                    tf.reshape(tf.square(prefix["standardized_residual"]), [-1]),
                ],
                axis=0,
            )
            smooth_maximum = inverse_temperature * tf.reduce_logsumexp(
                temperature_tensor * gate_squared
            )
            displacement = values - reference
            total = smooth_maximum + tf.constant(
                float(l2_displacement_weight), DTYPE
            ) * tf.reduce_sum(tf.square(displacement))
        gradient = tape.gradient(total, values)
        if gradient is None:
            raise ValueError("full-TT smooth-minimax objective has a missing gradient")
        tf.debugging.assert_all_finite(total, "full-TT smooth-minimax objective")
        tf.debugging.assert_all_finite(gradient, "full-TT smooth-minimax gradient")
        return total, gradient

    return value_and_gradient


def parent_shift(batch: T1ParameterDensityBatch, parent: LaneBT1Artifact) -> tf.Tensor:
    return tf.ones_like(batch.observation_log_density) * parent.shift_constant


def make_compiled_absolute_train_step(
    trainer: CenteredResidualTrainer,
    optimizer: tf.keras.optimizers.Optimizer,
    *,
    l1_weight: float,
    l2_weight: float,
    derivative_weight: float = 0.0,
    gradient_clip_norm: float,
):
    if hasattr(optimizer, "build"):
        optimizer.build(trainer.trainable_variables)

    @tf.function(jit_compile=True, reduce_retracing=True)
    def train_step(
        theta: tf.Tensor,
        local_points: tf.Tensor,
        log_absolute_importance_weight: tf.Tensor,
        origin_local_points: tf.Tensor,
        origin_complete_data_score: tf.Tensor,
        origin_likelihood_log_weight: tf.Tensor,
    ) -> tuple[tf.Tensor, ...]:
        with tf.GradientTape() as tape:
            terms = trainer.absolute_density_loss_arrays(
                theta,
                local_points,
                log_absolute_importance_weight,
                l1_weight=tf.constant(l1_weight, DTYPE),
                l2_weight=tf.constant(l2_weight, DTYPE),
                derivative_points=origin_local_points,
                derivative_target_score=origin_complete_data_score,
                derivative_importance_log_weight=origin_likelihood_log_weight,
                derivative_weight=tf.constant(derivative_weight, DTYPE),
            )
        gradients = tape.gradient(terms.total_loss, trainer.trainable_variables)
        if any(gradient is None for gradient in gradients):
            raise ValueError("absolute density training has a missing gradient")
        for name, value in (
            ("total_loss", terms.total_loss),
            ("absolute_density_loss", terms.absolute_density_loss),
            ("derivative_matching_loss", terms.derivative_matching_loss),
            ("exact_child_mass", terms.exact_child_mass),
            ("target_log_density_term", terms.target_log_density_term),
            ("target_mass_estimate", terms.target_mass_estimate),
            ("target_mass_standard_error", terms.target_mass_standard_error),
            ("minimum_rho", terms.minimum_rho),
        ):
            tf.debugging.assert_all_finite(value, f"nonfinite training term: {name}")
        for index, gradient in enumerate(gradients):
            tf.debugging.assert_all_finite(
                gradient, f"nonfinite residual gradient: {index}"
            )
        clipped, gradient_norm = tf.clip_by_global_norm(
            gradients, tf.constant(gradient_clip_norm, DTYPE)
        )
        tf.debugging.assert_all_finite(gradient_norm, "nonfinite gradient norm")
        for index, gradient in enumerate(clipped):
            tf.debugging.assert_all_finite(
                gradient, f"nonfinite clipped residual gradient: {index}"
            )
        optimizer.apply_gradients(zip(clipped, trainer.trainable_variables))
        for index, variable in enumerate(trainer.trainable_variables):
            tf.debugging.assert_all_finite(
                variable, f"nonfinite updated residual core: {index}"
            )
        maximum_core_magnitude = tf.reduce_max(
            tf.stack(
                [tf.reduce_max(tf.abs(value)) for value in trainer.trainable_variables]
            )
        )
        return (
            terms.total_loss,
            terms.absolute_density_loss,
            terms.derivative_matching_loss,
            tf.reduce_mean(terms.exact_child_mass),
            tf.reduce_mean(terms.target_log_density_term),
            tf.reduce_mean(terms.target_mass_estimate),
            tf.reduce_max(terms.target_mass_standard_error),
            terms.minimum_rho,
            gradient_norm,
            maximum_core_magnitude,
        )

    return train_step


def make_compiled_origin_score_prefit_step(
    trainer: CenteredResidualTrainer,
    optimizer: tf.keras.optimizers.Optimizer,
    *,
    gradient_clip_norm: float,
):
    """Compile a training-only prefit of the exact finite-child origin score."""

    if not math.isfinite(float(gradient_clip_norm)) or float(gradient_clip_norm) <= 0.0:
        raise ValueError("gradient_clip_norm must be positive and finite")
    if hasattr(optimizer, "build"):
        optimizer.build(trainer.trainable_variables)

    @tf.function(jit_compile=True, reduce_retracing=True)
    def prefit_step(
        origin_local_points: tf.Tensor,
        origin_complete_data_score: tf.Tensor,
        origin_likelihood_log_weight: tf.Tensor,
    ) -> tuple[tf.Tensor, ...]:
        with tf.GradientTape() as tape:
            metrics = trainer.origin_point_score_metrics_arrays(
                origin_local_points,
                origin_complete_data_score,
                origin_likelihood_log_weight,
            )
            loss = metrics["loss"]
        gradients = tape.gradient(loss, trainer.trainable_variables)
        if any(gradient is None for gradient in gradients):
            raise ValueError("origin score prefit has a missing gradient")
        tf.debugging.assert_all_finite(loss, "nonfinite origin score prefit loss")
        for index, gradient in enumerate(gradients):
            tf.debugging.assert_all_finite(
                gradient, f"nonfinite origin score prefit gradient: {index}"
            )
        clipped, gradient_norm = tf.clip_by_global_norm(
            gradients, tf.constant(gradient_clip_norm, DTYPE)
        )
        tf.debugging.assert_all_finite(
            gradient_norm, "nonfinite origin score prefit gradient norm"
        )
        optimizer.apply_gradients(zip(clipped, trainer.trainable_variables))
        for index, variable in enumerate(trainer.trainable_variables):
            tf.debugging.assert_all_finite(
                variable, f"nonfinite origin score prefit core: {index}"
            )
        maximum_core_magnitude = tf.reduce_max(
            tf.stack(
                [tf.reduce_max(tf.abs(value)) for value in trainer.trainable_variables]
            )
        )
        return (
            loss,
            metrics["target_likelihood_score"],
            metrics["child_likelihood_score"],
            metrics["target_point_score_rms"],
            metrics["score_residual_rms"],
            metrics["normalized_score_residual_rms"],
            metrics["child_point_score_standard_deviation"],
            metrics["importance_effective_sample_size"],
            gradient_norm,
            maximum_core_magnitude,
        )

    return prefit_step


def make_compiled_origin_total_score_train_step(
    trainer: CenteredResidualTrainer,
    optimizer: tf.keras.optimizers.Optimizer,
    *,
    point_weight: float,
    global_weight: float,
    prefix_weight: float,
    l2_weight: float,
    gradient_clip_norm: float,
):
    """Compile direct residual-TT training against all exact origin score operators."""

    for name, value in (
        ("point_weight", point_weight),
        ("global_weight", global_weight),
        ("prefix_weight", prefix_weight),
        ("gradient_clip_norm", gradient_clip_norm),
    ):
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{name} must be positive and finite")
    if not math.isfinite(float(l2_weight)) or float(l2_weight) < 0.0:
        raise ValueError("l2_weight must be finite and nonnegative")
    if hasattr(optimizer, "build"):
        optimizer.build(trainer.trainable_variables)

    @tf.function(jit_compile=True, reduce_retracing=True)
    def train_step(
        point_local_points: tf.Tensor,
        point_target_score: tf.Tensor,
        point_importance_log_weight: tf.Tensor,
        global_target_score: tf.Tensor,
        global_score_standard_error: tf.Tensor,
        prefix_local_points: tf.Tensor,
        prefix_target_score: tf.Tensor,
        prefix_score_standard_error: tf.Tensor,
    ) -> tuple[tf.Tensor, ...]:
        with tf.GradientTape() as tape:
            point = trainer.origin_point_score_metrics_arrays(
                point_local_points, point_target_score, point_importance_log_weight
            )
            global_metrics = trainer.origin_global_score_metrics_arrays(
                global_target_score, global_score_standard_error
            )
            prefix = trainer.origin_prefix_score_metrics_arrays(
                prefix_local_points, prefix_target_score, prefix_score_standard_error
            )
            l2 = tf.add_n(
                [tf.reduce_sum(tf.square(value)) for value in trainer.trainable_variables]
            )
            total = (
                tf.constant(float(point_weight), DTYPE) * point["loss"]
                + tf.constant(float(global_weight), DTYPE) * global_metrics["loss"]
                + tf.constant(float(prefix_weight), DTYPE) * prefix["loss"]
                + tf.constant(float(l2_weight), DTYPE) * l2
            )
        gradients = tape.gradient(total, trainer.trainable_variables)
        if any(gradient is None for gradient in gradients):
            raise ValueError("origin total-score training has a missing gradient")
        for value in (
            total,
            point["loss"],
            global_metrics["loss"],
            prefix["loss"],
            global_metrics["standardized_residual"],
            prefix["standardized_residual"],
        ):
            tf.debugging.assert_all_finite(value, "nonfinite total-score training term")
        for index, gradient in enumerate(gradients):
            tf.debugging.assert_all_finite(
                gradient, f"nonfinite total-score gradient: {index}"
            )
        clipped, gradient_norm = tf.clip_by_global_norm(
            gradients, tf.constant(float(gradient_clip_norm), DTYPE)
        )
        optimizer.apply_gradients(zip(clipped, trainer.trainable_variables))
        maximum_core_magnitude = tf.reduce_max(
            tf.stack(
                [tf.reduce_max(tf.abs(value)) for value in trainer.trainable_variables]
            )
        )
        return (
            total,
            point["loss"],
            global_metrics["loss"],
            prefix["loss"],
            point["normalized_score_residual_rms"],
            global_metrics["standardized_residual"],
            tf.reduce_max(prefix["standardized_residual"]),
            gradient_norm,
            maximum_core_magnitude,
        )

    return train_step


def rotating_prefix_minibatch_indices(
    *, pool_size: int, batch_size: int, update: int, seed: int
) -> tf.Tensor:
    """Return one deterministic shuffled batch from an exact pool epoch."""

    pool_count = int(pool_size)
    batch_count = int(batch_size)
    update_index = int(update)
    if pool_count <= 0 or batch_count <= 0:
        raise ValueError("pool_size and batch_size must be positive")
    if pool_count % batch_count != 0:
        raise ValueError("batch_size must divide pool_size")
    if update_index < 0:
        raise ValueError("update must be nonnegative")
    batches_per_epoch = pool_count // batch_count
    epoch = update_index // batches_per_epoch
    batch_in_epoch = update_index % batches_per_epoch
    permutation = tf.random.experimental.stateless_shuffle(
        tf.range(pool_count, dtype=tf.int32),
        seed=tf.constant([int(seed), epoch], tf.int32),
    )
    first = batch_in_epoch * batch_count
    return permutation[first : first + batch_count]


def estimate_t1_ratio_score(
    batch: T1ParameterDensityBatch, *, theta_index: int
) -> RatioScoreEstimate:
    index = int(theta_index)
    if index < 0 or index >= int(batch.theta.shape[0]):
        raise IndexError("theta_index out of range")
    log_weight = batch.observation_log_density[index]
    maximum = tf.reduce_max(log_weight)
    scaled = tf.exp(log_weight - maximum)
    weights = scaled / tf.reduce_sum(scaled)
    local_score = batch.complete_data_score[index]
    score = tf.reduce_sum(weights[:, tf.newaxis] * local_score, axis=0)
    mean_scaled = tf.reduce_mean(scaled)
    influence = scaled[:, tf.newaxis] * (
        local_score - score[tf.newaxis, :]
    ) / mean_scaled
    sample_count = tf.shape(local_score)[0]
    variance = tf.reduce_sum(tf.square(influence), axis=0) / tf.cast(
        sample_count - 1, DTYPE
    )
    standard_error = tf.sqrt(variance / tf.cast(sample_count, DTYPE))
    value = tf.exp(maximum) * mean_scaled
    return RatioScoreEstimate(
        value=value,
        score=score,
        score_standard_error=standard_error,
        effective_sample_size=tf.math.reciprocal(tf.reduce_sum(tf.square(weights))),
    )


def estimate_t1_prefix_scores(
    *,
    prefix_points: tf.Tensor,
    global_score: RatioScoreEstimate,
    sample_count: int,
    seed: int,
) -> tuple[RatioScoreEstimate, ...]:
    """Independent conditional-ratio estimates of origin retained-prefix scores."""

    points = tf.convert_to_tensor(prefix_points, DTYPE)
    if points.shape.rank != 2 or points.shape[1] != STATE_DIM:
        raise ValueError("prefix_points must have shape [point_count,18]")
    if int(sample_count) < 2:
        raise ValueError("prefix score estimation requires at least two samples")
    model = parameterized_zhao_cui_sir_austria_model()
    latent_model = __import__(
        "bayesfilter.highdim.sir_latent_preclip_tf",
        fromlist=["latent_preclip_zhao_cui_sir_austria_model"],
    ).latent_preclip_zhao_cui_sir_austria_model()
    _states, observations, _all = generate_sealed_lane_b_dataset()
    theta = tf.zeros([PARAMETER_DIM], DTYPE)
    prior_mean = model.base_model.initial_mean
    with tf.GradientTape() as tape:
        mean_variable = tf.Variable(prior_mean[tf.newaxis, :])
        transition_at_mean = latent_model.transition_mean(
            theta, mean_variable, time_index=1
        )
    jacobian = tf.reshape(
        tape.jacobian(transition_at_mean, mean_variable), [STATE_DIM, STATE_DIM]
    )
    precision = tf.eye(STATE_DIM, dtype=DTYPE) + tf.linalg.matmul(
        jacobian, jacobian, transpose_a=True
    )
    precision_chol = tf.linalg.cholesky(precision)
    covariance = tf.linalg.cholesky_solve(
        precision_chol, tf.eye(STATE_DIM, dtype=DTYPE)
    )
    noise = tf.random.stateless_normal(
        [int(sample_count), STATE_DIM],
        seed=tf.constant([int(seed), 991], tf.int32),
        dtype=DTYPE,
    )
    output = []
    for point_index, z1 in enumerate(tf.unstack(points, axis=0)):
        innovation = z1 - transition_at_mean[0]
        conditional_mean = prior_mean + tf.linalg.matvec(
            covariance, tf.linalg.matvec(jacobian, innovation, transpose_a=True)
        )
        centered = tf.linalg.triangular_solve(
            precision_chol,
            tf.transpose(noise),
            lower=True,
            adjoint=True,
        )
        z0 = conditional_mean[tf.newaxis, :] + tf.transpose(centered)
        delta = z0 - conditional_mean[tf.newaxis, :]
        q_log = (
            -0.5 * tf.cast(STATE_DIM, DTYPE) * LOG_TWO_PI
            + tf.reduce_sum(tf.math.log(tf.linalg.diag_part(precision_chol)))
            - 0.5 * tf.einsum("ni,ij,nj->n", delta, precision, delta)
        )
        tiled_z1 = tf.broadcast_to(z1[tf.newaxis, :], [int(sample_count), STATE_DIM])
        log_numerator = model.initial_log_density(theta, z0) + model.transition_log_density(
            theta, z0, tiled_z1, t=1
        )
        conditional_local_score = (
            model.initial_log_density_parameter_score(theta, z0)
            + model.transition_log_density_parameter_score(theta, z0, tiled_z1, t=1)
        )
        log_weight = log_numerator - q_log
        maximum = tf.reduce_max(log_weight)
        scaled = tf.exp(log_weight - maximum)
        weights = scaled / tf.reduce_sum(scaled)
        conditional_score = tf.reduce_sum(
            weights[:, tf.newaxis] * conditional_local_score, axis=0
        )
        mean_scaled = tf.reduce_mean(scaled)
        influence = scaled[:, tf.newaxis] * (
            conditional_local_score - conditional_score[tf.newaxis, :]
        ) / mean_scaled
        variance = tf.reduce_sum(tf.square(influence), axis=0) / tf.cast(
            sample_count - 1, DTYPE
        )
        conditional_se = tf.sqrt(variance / tf.cast(sample_count, DTYPE))
        observation_score = model.observation_log_density_parameter_score(
            theta, z1[tf.newaxis, :], observations[0], t=1
        )[0]
        prefix_score = conditional_score + observation_score - global_score.score
        prefix_se = tf.sqrt(
            tf.square(conditional_se) + tf.square(global_score.score_standard_error)
        )
        conditional_mass = tf.exp(maximum) * mean_scaled
        output.append(
            RatioScoreEstimate(
                value=conditional_mass,
                score=prefix_score,
                score_standard_error=prefix_se,
                effective_sample_size=tf.math.reciprocal(
                    tf.reduce_sum(tf.square(weights))
                ),
            )
        )
    return tuple(output)


__all__ = [
    "ABSOLUTE_LOSS_ID",
    "AbsoluteDensityLossTerms",
    "CenteredResidualTrainer",
    "CoreAffineTangentTrainer",
    "QuadraticConjugateGradientResult",
    "core_affine_origin_total_score_loss_arrays",
    "core_tangent_banks_from_residual_components",
    "core_affine_tangent_banks_from_position",
    "fixed_rank_initial_residual_components",
    "RatioScoreEstimate",
    "T1ParameterDensityBatch",
    "batch_native_t1_from_common_noise",
    "build_t1_parameter_density_batch",
    "core_tangent_to_residual_component",
    "estimate_t1_prefix_scores",
    "estimate_t1_ratio_score",
    "make_compiled_absolute_train_step",
    "make_compiled_core_affine_total_score_value_and_gradient",
    "make_compiled_core_affine_gate_minimax_value_and_gradient",
    "make_compiled_full_tt_gate_minimax_value_and_gradient",
    "make_compiled_origin_total_score_train_step",
    "rotating_prefix_minibatch_indices",
    "residual_components_from_position",
    "residual_components_position",
    "solve_quadratic_value_gradient_with_conjugate_gradient",
]
