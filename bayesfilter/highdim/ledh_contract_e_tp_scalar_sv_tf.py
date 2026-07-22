"""Experimental scalar-SV Contract E--TP recursion.

The LEDH observation surface is used only to construct a finite proposal.  All
importance corrections, continuation features, and dense references use the
row's target transition and target observation density.
"""

from __future__ import annotations

from dataclasses import dataclass

import tensorflow as tf

from bayesfilter.highdim import ledh_contract_e_tp_tf as tp
from bayesfilter.highdim.ledh_forward_contract import (
    ACTUAL_SV_ROW_ID,
    GENERALIZED_SV_ROW_ID,
    KSC_SV_ROW_ID,
)
from bayesfilter.highdim.models import GeneralizedSVPriorMeanSSM
from bayesfilter.highdim.sv_mixture_cut4 import (
    ExactTransformedSVSSM,
    KSCMixtureTransformedSVSSM,
)


DTYPE = tf.float64
ALGORITHM_ID = tp.ALGORITHM_ID
FEATURE_NAMES = (
    "mass",
    "state",
    "state_square",
    "stabilized_target_continuation_likelihood",
)
FEATURE_COUNT = len(FEATURE_NAMES)


@dataclass(frozen=True)
class ScalarSVContractETPSpec:
    """Bind one scalar target to its distinct affine LEDH proposal surface."""

    row_id: str
    model: object
    target_observation_policy: str
    flow_observation_policy: str
    flow_observation_variance: float
    target_transform_offset: float | None
    flow_transform_offset: float | None
    transition_before_first_observation: bool

    @property
    def parameter_dimension(self) -> int:
        return int(self.model.parameter_dim())


def make_scalar_sv_spec(row_id: str) -> ScalarSVContractETPSpec:
    """Return the frozen scalar target/proposal boundary for ``row_id``."""

    if row_id == ACTUAL_SV_ROW_ID:
        return ScalarSVContractETPSpec(
            row_id=row_id,
            model=ExactTransformedSVSSM(sigma=1.0),
            target_observation_policy="exact_log_y_square_log_chi_square",
            flow_observation_policy="gaussianized_log_square_affine_surface",
            flow_observation_variance=4.934802200544679,
            target_transform_offset=0.0,
            flow_transform_offset=0.0,
            transition_before_first_observation=False,
        )
    if row_id == KSC_SV_ROW_ID:
        return ScalarSVContractETPSpec(
            row_id=row_id,
            model=KSCMixtureTransformedSVSSM(sigma=1.0, transform_offset=1.0e-8),
            target_observation_policy="offset_log_y_square_ksc_mixture",
            flow_observation_policy="gaussianized_log_square_affine_surface",
            flow_observation_variance=4.934802200544679,
            target_transform_offset=1.0e-8,
            flow_transform_offset=1.0e-8,
            transition_before_first_observation=False,
        )
    if row_id == GENERALIZED_SV_ROW_ID:
        return ScalarSVContractETPSpec(
            row_id=row_id,
            model=GeneralizedSVPriorMeanSSM(process_scale=1.0),
            target_observation_policy="raw_zero_mean_normal_generalized_sv",
            flow_observation_policy="log_square_affine_proposal_only",
            flow_observation_variance=2.0,
            target_transform_offset=None,
            flow_transform_offset=1.0e-6,
            transition_before_first_observation=True,
        )
    raise ValueError(f"unsupported scalar Contract E--TP row: {row_id}")


def target_and_flow_observations(
    spec: ScalarSVContractETPSpec, raw_observations: tf.Tensor
) -> tuple[tf.Tensor, tf.Tensor]:
    """Return target and proposal observations without conflating their roles."""

    raw = tf.reshape(tf.convert_to_tensor(raw_observations, DTYPE), [-1, 1])
    if spec.target_transform_offset is None:
        target = raw
    else:
        target = tf.math.log(
            tf.square(raw) + tf.cast(spec.target_transform_offset, DTYPE)
        )
    if spec.flow_transform_offset is None:
        flow = raw
    else:
        flow = tf.math.log(
            tf.square(raw) + tf.cast(spec.flow_transform_offset, DTYPE)
        )
    return target, flow


def _normal_log_density(
    value: tf.Tensor, mean: tf.Tensor, variance: tf.Tensor
) -> tf.Tensor:
    variance = tf.convert_to_tensor(variance, DTYPE)
    return -0.5 * (
        tf.math.log(tf.constant(2.0 * 3.141592653589793, DTYPE) * variance)
        + tf.square(value - mean) / variance
    )


def _dynamics(
    spec: ScalarSVContractETPSpec, theta: tf.Tensor
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    parameters = spec.model.physical_parameters(theta)
    gamma = parameters["gamma"]
    if spec.row_id == GENERALIZED_SV_ROW_ID:
        stationary_mean = parameters["mu"]
        process_scale = tf.convert_to_tensor(spec.model.process_scale, DTYPE)
    else:
        stationary_mean = tf.constant(0.0, DTYPE)
        process_scale = tf.convert_to_tensor(spec.model.sigma, DTYPE)
    stationary_scale = process_scale / tf.sqrt(1.0 - tf.square(gamma))
    return stationary_mean, stationary_scale, gamma, process_scale


def _flow_surface(
    spec: ScalarSVContractETPSpec, theta: tf.Tensor
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    parameters = spec.model.physical_parameters(theta)
    if spec.row_id == GENERALIZED_SV_ROW_ID:
        slope = parameters["tau"]
        offset = tf.constant(0.0, DTYPE)
    else:
        slope = tf.constant(1.0, DTYPE)
        offset = tf.constant(2.0, DTYPE) * tf.math.log(parameters["beta"])
    return slope, offset, tf.constant(spec.flow_observation_variance, DTYPE)


def _affine_ledh_flow(
    pre_flow: tf.Tensor,
    prior_mean: tf.Tensor,
    prior_variance: tf.Tensor,
    flow_observation: tf.Tensor,
    slope: tf.Tensor,
    offset: tf.Tensor,
    observation_variance: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Apply the one-dimensional affine LEDH map and return proposal accounting."""

    posterior_variance = tf.math.reciprocal(
        tf.math.reciprocal(prior_variance)
        + tf.square(slope) / observation_variance
    )
    posterior_mean = posterior_variance * (
        prior_mean / prior_variance
        + slope * (flow_observation - offset) / observation_variance
    )
    scale = tf.sqrt(posterior_variance / prior_variance)
    post_flow = posterior_mean + scale * (pre_flow - prior_mean)
    proposal_log_density = _normal_log_density(
        pre_flow, prior_mean, prior_variance
    )
    forward_log_det = tf.math.log(scale)
    return post_flow, proposal_log_density, forward_log_det


def initial_rule(
    spec: ScalarSVContractETPSpec,
    theta: tf.Tensor,
    standard_nodes: tf.Tensor,
    standard_weights: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Return the stationary initial quadrature and transition innovations."""

    nodes = tf.reshape(tf.convert_to_tensor(standard_nodes, DTYPE), [-1])
    weights = tf.reshape(tf.convert_to_tensor(standard_weights, DTYPE), [-1])
    stationary_mean, stationary_scale, _gamma, _process_scale = _dynamics(
        spec, theta
    )
    return (
        stationary_mean + stationary_scale * nodes,
        tf.math.log(weights),
        nodes,
        tf.math.log(weights),
    )


def _teacher_step(
    spec: ScalarSVContractETPSpec,
    theta: tf.Tensor,
    parents: tf.Tensor,
    parent_log_weights: tf.Tensor,
    standard_nodes: tf.Tensor,
    standard_log_weights: tf.Tensor,
    target_observation: tf.Tensor,
    flow_observation: tf.Tensor,
    time_index: int,
) -> dict[str, tf.Tensor]:
    """Execute one fixed quadrature/LEDH/target-correction teacher step."""

    stationary_mean, stationary_scale, gamma, process_scale = _dynamics(spec, theta)
    slope, offset, observation_variance = _flow_surface(spec, theta)
    if time_index == 0 and not spec.transition_before_first_observation:
        prior_mean = tf.fill(tf.shape(standard_nodes), stationary_mean)
        prior_variance = tf.square(stationary_scale)
        pre_flow = stationary_mean + stationary_scale * standard_nodes
        base_log_weights = standard_log_weights
        previous = None
    else:
        parent_count = tf.shape(parents)[0]
        innovation_count = tf.shape(standard_nodes)[0]
        previous = tf.repeat(parents, innovation_count)
        if spec.row_id == GENERALIZED_SV_ROW_ID:
            prior_mean = stationary_mean + gamma * (previous - stationary_mean)
        else:
            prior_mean = gamma * previous
        prior_variance = tf.square(process_scale)
        pre_flow = prior_mean + process_scale * tf.tile(
            standard_nodes, [parent_count]
        )
        base_log_weights = (
            tf.repeat(parent_log_weights, innovation_count)
            + tf.tile(standard_log_weights, [parent_count])
        )
    particles, proposal_log_density, forward_log_det = _affine_ledh_flow(
        pre_flow,
        prior_mean,
        prior_variance,
        tf.reshape(flow_observation, [1])[0],
        slope,
        offset,
        observation_variance,
    )
    points = particles[:, None]
    if time_index == 0 and not spec.transition_before_first_observation:
        target_transition = spec.model.initial_log_density(theta, points)
    else:
        target_transition = spec.model.transition_log_density(
            theta, previous[:, None], points, t=time_index
        )
    target_observation_log = spec.model.observation_log_density(
        theta, points, target_observation, t=time_index
    )
    log_weights = (
        base_log_weights
        + target_transition
        + target_observation_log
        - proposal_log_density
        + forward_log_det
    )
    return {
        "particles": points,
        "log_unnormalized_weights": log_weights,
        "increment": tf.reduce_logsumexp(log_weights),
    }


def _pairwise_transition_log_density(
    spec: ScalarSVContractETPSpec,
    theta: tf.Tensor,
    previous_points: tf.Tensor,
    next_points: tf.Tensor,
    time_index: int,
) -> tf.Tensor:
    previous = tf.reshape(tf.convert_to_tensor(previous_points, DTYPE), [-1, 1])
    following = tf.reshape(tf.convert_to_tensor(next_points, DTYPE), [-1, 1])
    previous_count = tf.shape(previous)[0]
    next_count = tf.shape(following)[0]
    values = spec.model.transition_log_density(
        theta,
        tf.repeat(previous, next_count, axis=0),
        tf.tile(following, [previous_count, 1]),
        t=time_index,
    )
    return tf.reshape(values, [previous_count, next_count])


def target_continuation_log_likelihood(
    spec: ScalarSVContractETPSpec,
    theta: tf.Tensor,
    points: tf.Tensor,
    future_target_observations: tf.Tensor,
    grid_points: tf.Tensor,
    grid_weights: tf.Tensor,
    *,
    first_future_time_index: int,
) -> tf.Tensor:
    """Evaluate a fixed-grid target continuation by backward recursion."""

    points = tf.reshape(tf.convert_to_tensor(points, DTYPE), [-1, 1])
    observations = tf.reshape(
        tf.convert_to_tensor(future_target_observations, DTYPE), [-1, 1]
    )
    grid = tf.reshape(tf.convert_to_tensor(grid_points, DTYPE), [-1, 1])
    log_grid_weights = tf.math.log(
        tf.reshape(tf.convert_to_tensor(grid_weights, DTYPE), [-1])
    )
    future_count = observations.shape[0]
    if future_count is None or future_count < 1:
        return tf.zeros([tf.shape(points)[0]], DTYPE)
    child_log = tf.zeros([tf.shape(grid)[0]], DTYPE)
    for local_index in range(future_count - 1, 0, -1):
        absolute_time = first_future_time_index + local_index
        transition = _pairwise_transition_log_density(
            spec, theta, grid, grid, absolute_time
        )
        observation_log = spec.model.observation_log_density(
            theta, grid, observations[local_index], t=absolute_time
        )
        child_log = tf.reduce_logsumexp(
            transition
            + log_grid_weights[None, :]
            + observation_log[None, :]
            + child_log[None, :],
            axis=1,
        )
    transition = _pairwise_transition_log_density(
        spec, theta, points, grid, first_future_time_index
    )
    observation_log = spec.model.observation_log_density(
        theta,
        grid,
        observations[0],
        t=first_future_time_index,
    )
    return tf.reduce_logsumexp(
        transition
        + log_grid_weights[None, :]
        + observation_log[None, :]
        + child_log[None, :],
        axis=1,
    )


def _pairwise_transition_log_density_static(
    spec: ScalarSVContractETPSpec,
    theta: tf.Tensor,
    previous_points: tf.Tensor,
    next_points: tf.Tensor,
    time_index: tf.Tensor,
) -> tf.Tensor:
    """Pairwise transition matrix with compile-time Cartesian shapes."""

    previous = tf.reshape(tf.convert_to_tensor(previous_points, DTYPE), [-1, 1])
    following = tf.reshape(tf.convert_to_tensor(next_points, DTYPE), [-1, 1])
    previous_count = previous.shape[0]
    next_count = following.shape[0]
    if previous_count is None or next_count is None:
        raise ValueError("loop-native continuation requires static point counts")
    previous_pairs = tf.reshape(
        tf.broadcast_to(previous[:, None, :], [previous_count, next_count, 1]),
        [previous_count * next_count, 1],
    )
    following_pairs = tf.reshape(
        tf.broadcast_to(following[None, :, :], [previous_count, next_count, 1]),
        [previous_count * next_count, 1],
    )
    values = spec.model.transition_log_density(
        theta, previous_pairs, following_pairs, t=time_index
    )
    return tf.reshape(values, [previous_count, next_count])


def target_continuation_log_likelihood_loop(
    spec: ScalarSVContractETPSpec,
    theta: tf.Tensor,
    points: tf.Tensor,
    future_target_observations: tf.Tensor,
    future_count: tf.Tensor,
    grid_points: tf.Tensor,
    grid_weights: tf.Tensor,
    *,
    first_future_time_index: tf.Tensor,
) -> tf.Tensor:
    """Loop-native fixed-window target continuation with a valid prefix count."""

    points = tf.reshape(tf.convert_to_tensor(points, DTYPE), [-1, 1])
    observations = tf.reshape(
        tf.convert_to_tensor(future_target_observations, DTYPE), [-1, 1]
    )
    grid = tf.reshape(tf.convert_to_tensor(grid_points, DTYPE), [-1, 1])
    grid_count = grid.shape[0]
    if observations.shape[0] is None or grid_count is None:
        raise ValueError("loop-native continuation requires static window/grid shapes")
    future_count = tf.reshape(tf.convert_to_tensor(future_count, tf.int32), [])
    first_future_time_index = tf.reshape(
        tf.convert_to_tensor(first_future_time_index, tf.int32), []
    )
    log_grid_weights = tf.math.log(
        tf.reshape(tf.convert_to_tensor(grid_weights, DTYPE), [grid_count])
    )

    def cond(index: tf.Tensor, _child_log: tf.Tensor) -> tf.Tensor:
        del _child_log
        return index >= 0

    def body(index: tf.Tensor, child_log: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        absolute_time = first_future_time_index + index
        transition = _pairwise_transition_log_density_static(
            spec, theta, grid, grid, absolute_time
        )
        observation_log = spec.model.observation_log_density(
            theta, grid, observations[index], t=absolute_time
        )
        recursed_child = tf.reduce_logsumexp(
            transition
            + log_grid_weights[None, :]
            + observation_log[None, :]
            + child_log[None, :],
            axis=1,
        )
        next_child = tf.where(index > 0, recursed_child, child_log)
        return index - 1, next_child

    _, child_log = tf.while_loop(
        cond,
        body,
        (future_count - 1, tf.zeros([grid_count], DTYPE)),
        parallel_iterations=1,
        maximum_iterations=observations.shape[0],
    )
    transition = _pairwise_transition_log_density_static(
        spec, theta, points, grid, first_future_time_index
    )
    observation_log = spec.model.observation_log_density(
        theta, grid, observations[0], t=first_future_time_index
    )
    return tf.reduce_logsumexp(
        transition
        + log_grid_weights[None, :]
        + observation_log[None, :]
        + child_log[None, :],
        axis=1,
    )


def _features_loop(
    spec: ScalarSVContractETPSpec,
    theta: tf.Tensor,
    points: tf.Tensor,
    future_target_observations: tf.Tensor,
    future_count: tf.Tensor,
    grid_points: tf.Tensor,
    grid_weights: tf.Tensor,
    *,
    first_future_time_index: tf.Tensor,
) -> tf.Tensor:
    """Evaluate the fixed four-row feature map through one continuation loop."""

    values = tf.reshape(tf.convert_to_tensor(points, DTYPE), [-1])
    reference, _stationary_scale, _gamma, _process_scale = _dynamics(spec, theta)
    combined = tf.concat([values, reference[None]], axis=0)
    continuation = target_continuation_log_likelihood_loop(
        spec,
        theta,
        combined[:, None],
        future_target_observations,
        future_count,
        grid_points,
        grid_weights,
        first_future_time_index=first_future_time_index,
    )
    continuation_log = continuation[:-1]
    reference_log = continuation[-1]
    return tf.stack(
        [
            tf.ones_like(values),
            values,
            tf.square(values),
            tf.exp(continuation_log - reference_log),
        ],
        axis=0,
    )


def _teacher_transition_step_loop(
    spec: ScalarSVContractETPSpec,
    theta: tf.Tensor,
    parents: tf.Tensor,
    parent_log_weights: tf.Tensor,
    standard_nodes: tf.Tensor,
    standard_log_weights: tf.Tensor,
    target_observation: tf.Tensor,
    flow_observation: tf.Tensor,
    time_index: tf.Tensor,
) -> dict[str, tf.Tensor]:
    """Execute a positive-time teacher step with static Cartesian shapes."""

    if spec.transition_before_first_observation:
        raise ValueError("Phase 3 loop core excludes transition-before-first rows")
    parents = tf.reshape(tf.convert_to_tensor(parents, DTYPE), [-1])
    parent_log_weights = tf.reshape(
        tf.convert_to_tensor(parent_log_weights, DTYPE), [-1]
    )
    standard_nodes = tf.reshape(tf.convert_to_tensor(standard_nodes, DTYPE), [-1])
    standard_log_weights = tf.reshape(
        tf.convert_to_tensor(standard_log_weights, DTYPE), [-1]
    )
    parent_count = parents.shape[0]
    innovation_count = standard_nodes.shape[0]
    if parent_count is None or innovation_count is None:
        raise ValueError("loop-native teacher requires static parent/innovation counts")
    teacher_count = parent_count * innovation_count
    previous = tf.reshape(
        tf.broadcast_to(parents[:, None], [parent_count, innovation_count]),
        [teacher_count],
    )
    innovations = tf.reshape(
        tf.broadcast_to(
            standard_nodes[None, :], [parent_count, innovation_count]
        ),
        [teacher_count],
    )
    base_log_weights = tf.reshape(
        tf.broadcast_to(
            parent_log_weights[:, None], [parent_count, innovation_count]
        )
        + tf.broadcast_to(
            standard_log_weights[None, :], [parent_count, innovation_count]
        ),
        [teacher_count],
    )
    _stationary_mean, _stationary_scale, gamma, process_scale = _dynamics(
        spec, theta
    )
    slope, offset, observation_variance = _flow_surface(spec, theta)
    prior_mean = gamma * previous
    prior_variance = tf.square(process_scale)
    pre_flow = prior_mean + process_scale * innovations
    particles, proposal_log_density, forward_log_det = _affine_ledh_flow(
        pre_flow,
        prior_mean,
        prior_variance,
        tf.reshape(flow_observation, [1])[0],
        slope,
        offset,
        observation_variance,
    )
    points = particles[:, None]
    target_transition = spec.model.transition_log_density(
        theta, previous[:, None], points, t=time_index
    )
    target_observation_log = spec.model.observation_log_density(
        theta, points, target_observation, t=time_index
    )
    log_weights = (
        base_log_weights
        + target_transition
        + target_observation_log
        - proposal_log_density
        + forward_log_det
    )
    return {
        "particles": points,
        "log_unnormalized_weights": log_weights,
        "increment": tf.reduce_logsumexp(log_weights),
    }


def _future_window(
    observations: tf.Tensor,
    first_future_index: tf.Tensor,
    lookahead_steps: int,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Gather a fixed-size future window and return its valid prefix count."""

    time_steps = observations.shape[0]
    if time_steps is None or time_steps < 2:
        raise ValueError("future windows require a static horizon of at least two")
    count = tf.minimum(
        tf.cast(lookahead_steps, tf.int32),
        tf.cast(time_steps, tf.int32) - first_future_index,
    )
    offsets = tf.range(lookahead_steps, dtype=tf.int32)
    indices = tf.minimum(first_future_index + offsets, time_steps - 1)
    return tf.gather(observations, indices), count


def _actual_sv_parameter_jacobian(theta: tf.Tensor) -> dict[str, tf.Tensor]:
    """Return Actual-SV physical parameters and their two-direction tangents."""

    theta = tf.reshape(tf.convert_to_tensor(theta, DTYPE), [2])
    sqrt_two_pi = tf.sqrt(tf.constant(2.0 * 3.141592653589793, DTYPE))
    gamma = 0.5 * (
        1.0
        + tf.math.erf(theta[0] / tf.sqrt(tf.constant(2.0, DTYPE)))
    )
    gamma_derivative = tf.exp(-0.5 * tf.square(theta[0])) / sqrt_two_pi
    gamma_tangent = tf.stack([gamma_derivative, tf.constant(0.0, DTYPE)])
    stationary_variance = tf.math.reciprocal(1.0 - tf.square(gamma))
    stationary_variance_tangent = (
        2.0
        * gamma
        * tf.square(stationary_variance)
        * gamma_tangent
    )
    stationary_scale = tf.sqrt(stationary_variance)
    stationary_scale_tangent = (
        0.5
        * stationary_scale
        * stationary_variance_tangent
        / stationary_variance
    )
    offset = 2.0 * theta[1]
    offset_tangent = tf.constant([0.0, 2.0], DTYPE)
    return {
        "gamma": gamma,
        "gamma_tangent": gamma_tangent,
        "stationary_variance": stationary_variance,
        "stationary_variance_tangent": stationary_variance_tangent,
        "stationary_scale": stationary_scale,
        "stationary_scale_tangent": stationary_scale_tangent,
        "offset": offset,
        "offset_tangent": offset_tangent,
    }


def _normal_log_density_multi_jvp(
    value: tf.Tensor,
    mean: tf.Tensor,
    variance: tf.Tensor,
    value_tangent: tf.Tensor,
    mean_tangent: tf.Tensor,
    variance_tangent: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Evaluate a scalar Normal log density and two-direction total JVP."""

    variance = tf.convert_to_tensor(variance, DTYPE)
    residual = value - mean
    primal = _normal_log_density(value, mean, variance)
    tangent = -0.5 * (
        variance_tangent / variance[..., None]
        + 2.0
        * residual[..., None]
        * (value_tangent - mean_tangent)
        / variance[..., None]
        - tf.square(residual)[..., None]
        * variance_tangent
        / tf.square(variance)[..., None]
    )
    return primal, tangent


def _actual_sv_affine_ledh_multi_jvp(
    pre_flow: tf.Tensor,
    prior_mean: tf.Tensor,
    prior_variance: tf.Tensor,
    flow_observation: tf.Tensor,
    pre_flow_tangent: tf.Tensor,
    prior_mean_tangent: tf.Tensor,
    prior_variance_tangent: tf.Tensor,
    offset: tf.Tensor,
    offset_tangent: tf.Tensor,
    observation_variance: tf.Tensor,
) -> dict[str, tf.Tensor]:
    """Manual two-direction JVP of the Actual-SV affine LEDH map."""

    posterior_variance = tf.math.reciprocal(
        tf.math.reciprocal(prior_variance)
        + tf.math.reciprocal(observation_variance)
    )
    posterior_variance_tangent = (
        tf.square(posterior_variance)[..., None]
        * prior_variance_tangent
        / tf.square(prior_variance)[..., None]
    )
    information_mean = (
        prior_mean / prior_variance
        + (flow_observation - offset) / observation_variance
    )
    information_mean_tangent = (
        prior_mean_tangent / prior_variance[..., None]
        - prior_mean[..., None]
        * prior_variance_tangent
        / tf.square(prior_variance)[..., None]
        - offset_tangent / observation_variance
    )
    posterior_mean = posterior_variance * information_mean
    posterior_mean_tangent = (
        posterior_variance_tangent * information_mean[..., None]
        + posterior_variance[..., None] * information_mean_tangent
    )
    scale = tf.sqrt(posterior_variance / prior_variance)
    scale_tangent = 0.5 * scale[..., None] * (
        posterior_variance_tangent / posterior_variance[..., None]
        - prior_variance_tangent / prior_variance[..., None]
    )
    particles = posterior_mean + scale * (pre_flow - prior_mean)
    particles_tangent = (
        posterior_mean_tangent
        + scale_tangent * (pre_flow - prior_mean)[..., None]
        + scale[..., None] * (pre_flow_tangent - prior_mean_tangent)
    )
    proposal_log_density, proposal_tangent = _normal_log_density_multi_jvp(
        pre_flow,
        prior_mean,
        prior_variance,
        pre_flow_tangent,
        prior_mean_tangent,
        prior_variance_tangent,
    )
    return {
        "particles": particles,
        "particles_tangent": particles_tangent,
        "proposal_log_density": proposal_log_density,
        "proposal_log_density_tangent": proposal_tangent,
        "forward_log_det": tf.math.log(scale),
        "forward_log_det_tangent": scale_tangent / scale[..., None],
    }


def _actual_sv_observation_multi_jvp(
    theta: tf.Tensor,
    points: tf.Tensor,
    points_tangent: tf.Tensor,
    observation: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Exact transformed-SV observation log density and total JVP."""

    parameters = _actual_sv_parameter_jacobian(theta)
    values = tf.reshape(points, [-1])
    value_tangent = tf.reshape(points_tangent, [-1, 2])
    residual = (
        tf.reshape(observation, [1])[0] - parameters["offset"] - values
    )
    residual_tangent = -parameters["offset_tangent"][None, :] - value_tangent
    primal = (
        0.5 * residual
        - 0.5 * tf.exp(residual)
        - 0.5 * tf.math.log(tf.constant(2.0 * 3.141592653589793, DTYPE))
    )
    tangent = 0.5 * (1.0 - tf.exp(residual))[:, None] * residual_tangent
    return primal, tangent


def _logsumexp_multi_jvp(
    values: tf.Tensor, tangents: tf.Tensor, *, axis: int
) -> tuple[tf.Tensor, tf.Tensor]:
    primal = tf.reduce_logsumexp(values, axis=axis)
    weights = tf.nn.softmax(values, axis=axis)
    tangent = tf.reduce_sum(weights[..., None] * tangents, axis=axis)
    return primal, tangent


def _actual_sv_initial_teacher_multi_jvp(
    spec: ScalarSVContractETPSpec,
    theta: tf.Tensor,
    standard_nodes: tf.Tensor,
    standard_log_weights: tf.Tensor,
    target_observation: tf.Tensor,
    flow_observation: tf.Tensor,
) -> dict[str, tf.Tensor]:
    """Initial Actual-SV teacher and its explicit two-direction tangent."""

    parameters = _actual_sv_parameter_jacobian(theta)
    nodes = tf.reshape(standard_nodes, [-1])
    count = tf.shape(nodes)[0]
    zeros = tf.zeros([count, 2], DTYPE)
    prior_mean = tf.zeros([count], DTYPE)
    prior_mean_tangent = zeros
    prior_variance = tf.fill([count], parameters["stationary_variance"])
    prior_variance_tangent = tf.broadcast_to(
        parameters["stationary_variance_tangent"][None, :], [count, 2]
    )
    pre_flow = parameters["stationary_scale"] * nodes
    pre_flow_tangent = (
        nodes[:, None] * parameters["stationary_scale_tangent"][None, :]
    )
    ledh = _actual_sv_affine_ledh_multi_jvp(
        pre_flow,
        prior_mean,
        prior_variance,
        tf.reshape(flow_observation, [1])[0],
        pre_flow_tangent,
        prior_mean_tangent,
        prior_variance_tangent,
        parameters["offset"],
        parameters["offset_tangent"],
        tf.constant(spec.flow_observation_variance, DTYPE),
    )
    initial_log, initial_tangent = _normal_log_density_multi_jvp(
        ledh["particles"],
        prior_mean,
        prior_variance,
        ledh["particles_tangent"],
        prior_mean_tangent,
        prior_variance_tangent,
    )
    observation_log, observation_tangent = _actual_sv_observation_multi_jvp(
        theta,
        ledh["particles"][:, None],
        ledh["particles_tangent"][:, None, :],
        target_observation,
    )
    log_weights = (
        standard_log_weights
        + initial_log
        + observation_log
        - ledh["proposal_log_density"]
        + ledh["forward_log_det"]
    )
    log_weights_tangent = (
        initial_tangent
        + observation_tangent
        - ledh["proposal_log_density_tangent"]
        + ledh["forward_log_det_tangent"]
    )
    increment, increment_tangent = _logsumexp_multi_jvp(
        log_weights, log_weights_tangent, axis=0
    )
    return {
        "particles": ledh["particles"][:, None],
        "particles_tangent": ledh["particles_tangent"][:, None, :],
        "log_unnormalized_weights": log_weights,
        "log_unnormalized_weights_tangent": log_weights_tangent,
        "increment": increment,
        "increment_tangent": increment_tangent,
    }


def _actual_sv_transition_teacher_multi_jvp(
    spec: ScalarSVContractETPSpec,
    theta: tf.Tensor,
    parents: tf.Tensor,
    parent_tangents: tf.Tensor,
    parent_log_weights: tf.Tensor,
    parent_log_weight_tangents: tf.Tensor,
    standard_nodes: tf.Tensor,
    standard_log_weights: tf.Tensor,
    target_observation: tf.Tensor,
    flow_observation: tf.Tensor,
) -> dict[str, tf.Tensor]:
    """Positive-time Actual-SV teacher and explicit total JVP."""

    parameters = _actual_sv_parameter_jacobian(theta)
    parents = tf.reshape(parents, [-1])
    parent_tangents = tf.reshape(parent_tangents, [-1, 2])
    parent_count = parents.shape[0]
    innovation_count = standard_nodes.shape[0]
    if parent_count is None or innovation_count is None:
        raise ValueError("manual teacher JVP requires static parent and node counts")
    teacher_count = parent_count * innovation_count
    previous = tf.reshape(
        tf.broadcast_to(parents[:, None], [parent_count, innovation_count]),
        [teacher_count],
    )
    previous_tangent = tf.reshape(
        tf.broadcast_to(
            parent_tangents[:, None, :], [parent_count, innovation_count, 2]
        ),
        [teacher_count, 2],
    )
    innovations = tf.reshape(
        tf.broadcast_to(
            tf.reshape(standard_nodes, [1, innovation_count]),
            [parent_count, innovation_count],
        ),
        [teacher_count],
    )
    base_log_weights = tf.reshape(
        tf.broadcast_to(
            parent_log_weights[:, None], [parent_count, innovation_count]
        )
        + tf.broadcast_to(
            standard_log_weights[None, :], [parent_count, innovation_count]
        ),
        [teacher_count],
    )
    base_log_weight_tangent = tf.reshape(
        tf.broadcast_to(
            parent_log_weight_tangents[:, None, :],
            [parent_count, innovation_count, 2],
        ),
        [teacher_count, 2],
    )
    prior_mean = parameters["gamma"] * previous
    prior_mean_tangent = (
        parameters["gamma_tangent"][None, :] * previous[:, None]
        + parameters["gamma"] * previous_tangent
    )
    prior_variance = tf.ones([teacher_count], DTYPE)
    prior_variance_tangent = tf.zeros([teacher_count, 2], DTYPE)
    pre_flow = prior_mean + innovations
    ledh = _actual_sv_affine_ledh_multi_jvp(
        pre_flow,
        prior_mean,
        prior_variance,
        tf.reshape(flow_observation, [1])[0],
        prior_mean_tangent,
        prior_mean_tangent,
        prior_variance_tangent,
        parameters["offset"],
        parameters["offset_tangent"],
        tf.constant(spec.flow_observation_variance, DTYPE),
    )
    transition_log, transition_tangent = _normal_log_density_multi_jvp(
        ledh["particles"],
        prior_mean,
        prior_variance,
        ledh["particles_tangent"],
        prior_mean_tangent,
        prior_variance_tangent,
    )
    observation_log, observation_tangent = _actual_sv_observation_multi_jvp(
        theta,
        ledh["particles"][:, None],
        ledh["particles_tangent"][:, None, :],
        target_observation,
    )
    log_weights = (
        base_log_weights
        + transition_log
        + observation_log
        - ledh["proposal_log_density"]
        + ledh["forward_log_det"]
    )
    log_weights_tangent = (
        base_log_weight_tangent
        + transition_tangent
        + observation_tangent
        - ledh["proposal_log_density_tangent"]
        + ledh["forward_log_det_tangent"]
    )
    increment, increment_tangent = _logsumexp_multi_jvp(
        log_weights, log_weights_tangent, axis=0
    )
    return {
        "particles": ledh["particles"][:, None],
        "particles_tangent": ledh["particles_tangent"][:, None, :],
        "log_unnormalized_weights": log_weights,
        "log_unnormalized_weights_tangent": log_weights_tangent,
        "increment": increment,
        "increment_tangent": increment_tangent,
    }


def _actual_sv_pairwise_transition_multi_jvp(
    theta: tf.Tensor,
    previous_points: tf.Tensor,
    previous_tangents: tf.Tensor,
    next_points: tf.Tensor,
    next_tangents: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Actual-SV pairwise AR(1) log density and two-direction JVP."""

    parameters = _actual_sv_parameter_jacobian(theta)
    previous = tf.reshape(previous_points, [-1])
    following = tf.reshape(next_points, [-1])
    previous_dot = tf.reshape(previous_tangents, [-1, 2])
    following_dot = tf.reshape(next_tangents, [-1, 2])
    previous_count = previous.shape[0]
    next_count = following.shape[0]
    if previous_count is None or next_count is None:
        raise ValueError("pairwise manual JVP requires static point counts")
    previous_pairs = tf.broadcast_to(
        previous[:, None], [previous_count, next_count]
    )
    following_pairs = tf.broadcast_to(
        following[None, :], [previous_count, next_count]
    )
    previous_dot_pairs = tf.broadcast_to(
        previous_dot[:, None, :], [previous_count, next_count, 2]
    )
    following_dot_pairs = tf.broadcast_to(
        following_dot[None, :, :], [previous_count, next_count, 2]
    )
    mean = parameters["gamma"] * previous_pairs
    mean_dot = (
        parameters["gamma_tangent"][None, None, :]
        * previous_pairs[..., None]
        + parameters["gamma"] * previous_dot_pairs
    )
    residual = following_pairs - mean
    residual_dot = following_dot_pairs - mean_dot
    primal = -0.5 * (
        tf.math.log(tf.constant(2.0 * 3.141592653589793, DTYPE))
        + tf.square(residual)
    )
    tangent = -residual[..., None] * residual_dot
    return primal, tangent


def _actual_sv_continuation_multi_jvp(
    spec: ScalarSVContractETPSpec,
    theta: tf.Tensor,
    points: tf.Tensor,
    point_tangents: tf.Tensor,
    future_target_observations: tf.Tensor,
    future_count: tf.Tensor,
    grid_points: tf.Tensor,
    grid_weights: tf.Tensor,
    *,
    first_future_time_index: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Fixed-window target continuation and explicit two-direction JVP."""

    points = tf.reshape(tf.convert_to_tensor(points, DTYPE), [-1, 1])
    point_tangents = tf.reshape(point_tangents, [-1, 1, 2])
    observations = tf.reshape(
        tf.convert_to_tensor(future_target_observations, DTYPE), [-1, 1]
    )
    grid = tf.reshape(tf.convert_to_tensor(grid_points, DTYPE), [-1, 1])
    grid_count = grid.shape[0]
    window_count = observations.shape[0]
    if grid_count is None or window_count is None:
        raise ValueError("manual continuation requires static grid/window shapes")
    grid_tangents = tf.zeros([grid_count, 1, 2], DTYPE)
    log_grid_weights = tf.math.log(
        tf.reshape(tf.convert_to_tensor(grid_weights, DTYPE), [grid_count])
    )
    future_count = tf.reshape(tf.convert_to_tensor(future_count, tf.int32), [])

    def cond(
        index: tf.Tensor, _child: tf.Tensor, _child_dot: tf.Tensor
    ) -> tf.Tensor:
        return index >= 0

    def body(
        index: tf.Tensor, child: tf.Tensor, child_dot: tf.Tensor
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        transition, transition_dot = _actual_sv_pairwise_transition_multi_jvp(
            theta, grid, grid_tangents, grid, grid_tangents
        )
        observation_log, observation_dot = _actual_sv_observation_multi_jvp(
            theta,
            grid,
            grid_tangents,
            observations[index],
        )
        values = (
            transition
            + log_grid_weights[None, :]
            + observation_log[None, :]
            + child[None, :]
        )
        tangents = (
            transition_dot
            + observation_dot[None, :, :]
            + child_dot[None, :, :]
        )
        recursed_child, recursed_child_dot = _logsumexp_multi_jvp(
            values, tangents, axis=1
        )
        next_child = tf.where(index > 0, recursed_child, child)
        next_child_dot = tf.where(index > 0, recursed_child_dot, child_dot)
        return index - 1, next_child, next_child_dot

    _, child, child_dot = tf.while_loop(
        cond,
        body,
        (
            future_count - 1,
            tf.zeros([grid_count], DTYPE),
            tf.zeros([grid_count, 2], DTYPE),
        ),
        parallel_iterations=1,
        maximum_iterations=window_count,
    )
    transition, transition_dot = _actual_sv_pairwise_transition_multi_jvp(
        theta, points, point_tangents, grid, grid_tangents
    )
    observation_log, observation_dot = _actual_sv_observation_multi_jvp(
        theta, grid, grid_tangents, observations[0]
    )
    values = (
        transition
        + log_grid_weights[None, :]
        + observation_log[None, :]
        + child[None, :]
    )
    tangents = (
        transition_dot
        + observation_dot[None, :, :]
        + child_dot[None, :, :]
    )
    return _logsumexp_multi_jvp(values, tangents, axis=1)


def _actual_sv_features_multi_jvp(
    spec: ScalarSVContractETPSpec,
    theta: tf.Tensor,
    points: tf.Tensor,
    point_tangents: tf.Tensor,
    future_target_observations: tf.Tensor,
    future_count: tf.Tensor,
    grid_points: tf.Tensor,
    grid_weights: tf.Tensor,
    *,
    first_future_time_index: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Four Actual-SV features and their explicit total two-direction JVP."""

    values = tf.reshape(points, [-1])
    value_dot = tf.reshape(point_tangents, [-1, 2])
    combined = tf.concat([values, tf.zeros([1], DTYPE)], axis=0)
    combined_dot = tf.concat([value_dot, tf.zeros([1, 2], DTYPE)], axis=0)
    continuation, continuation_dot = _actual_sv_continuation_multi_jvp(
        spec,
        theta,
        combined[:, None],
        combined_dot[:, None, :],
        future_target_observations,
        future_count,
        grid_points,
        grid_weights,
        first_future_time_index=first_future_time_index,
    )
    log_ratio = continuation[:-1] - continuation[-1]
    log_ratio_dot = continuation_dot[:-1] - continuation_dot[-1]
    continuation_feature = tf.exp(log_ratio)
    features = tf.stack(
        [
            tf.ones_like(values),
            values,
            tf.square(values),
            continuation_feature,
        ],
        axis=0,
    )
    feature_tangents = tf.stack(
        [
            tf.zeros_like(value_dot),
            value_dot,
            2.0 * values[:, None] * value_dot,
            continuation_feature[:, None] * log_ratio_dot,
        ],
        axis=0,
    )
    return features, feature_tangents


def contract_e_tp_scalar_sv_loop_core(
    spec: ScalarSVContractETPSpec,
    theta: tf.Tensor,
    target_observations: tf.Tensor,
    flow_observations: tf.Tensor,
    standard_nodes: tf.Tensor,
    standard_weights: tf.Tensor,
    active_indices: tf.Tensor,
    row_scales: tf.Tensor,
    continuation_grid_points: tf.Tensor,
    continuation_grid_weights: tf.Tensor,
    *,
    lookahead_steps: int,
) -> dict[str, tf.Tensor]:
    """Execute actual/KSC fixed-square recursion with functional time loops."""

    if spec.row_id not in (ACTUAL_SV_ROW_ID, KSC_SV_ROW_ID):
        raise ValueError("Phase 3 loop core supports actual SV and KSC-SV only")
    if spec.transition_before_first_observation:
        raise ValueError("Phase 3 loop core requires observation at the initial state")
    if lookahead_steps < 1:
        raise ValueError("lookahead_steps must be positive")
    theta = tf.reshape(tf.convert_to_tensor(theta, DTYPE), [spec.parameter_dimension])
    target_observations = tf.reshape(
        tf.convert_to_tensor(target_observations, DTYPE), [-1, 1]
    )
    flow_observations = tf.reshape(
        tf.convert_to_tensor(flow_observations, DTYPE), [-1, 1]
    )
    standard_nodes = tf.reshape(tf.convert_to_tensor(standard_nodes, DTYPE), [-1])
    standard_weights = tf.reshape(
        tf.convert_to_tensor(standard_weights, DTYPE), [-1]
    )
    active_indices = tf.convert_to_tensor(active_indices, tf.int32)
    row_scales = tf.convert_to_tensor(row_scales, DTYPE)
    time_steps = target_observations.shape[0]
    innovation_count = standard_nodes.shape[0]
    if time_steps is None or time_steps < 1 or innovation_count is None:
        raise ValueError("loop-native scalar SV requires static positive shapes")
    if active_indices.shape != (time_steps - 1, FEATURE_COUNT):
        raise ValueError("active_indices has the wrong loop-native shape")
    if row_scales.shape != (time_steps - 1, FEATURE_COUNT):
        raise ValueError("row_scales has the wrong loop-native shape")

    parents0, parent_log_weights0, nodes, node_log_weights = initial_rule(
        spec, theta, standard_nodes, standard_weights
    )
    first = _teacher_step(
        spec,
        theta,
        parents0,
        parent_log_weights0,
        nodes,
        node_log_weights,
        target_observations[0],
        flow_observations[0],
        0,
    )
    if time_steps == 1:
        return {
            "objective": first["increment"],
            "increment_history": first["increment"][None],
            "minimum_weight_history": tf.zeros([0], DTYPE),
            "condition_number_history": tf.zeros([0], DTYPE),
            "feature_residual_history": tf.zeros([0, FEATURE_COUNT], DTYPE),
            "valid_history": tf.ones([1], tf.bool),
            "final_particles": first["particles"],
            "final_log_unnormalized_weights": first["log_unnormalized_weights"],
        }

    first_window, first_count = _future_window(
        target_observations, tf.constant(1, tf.int32), lookahead_steps
    )
    first_features = _features_loop(
        spec,
        theta,
        first["particles"],
        first_window,
        first_count,
        continuation_grid_points,
        continuation_grid_weights,
        first_future_time_index=tf.constant(1, tf.int32),
    )
    first_projection = tp._contract_e_tp_dense_square_forward_core(
        first["particles"],
        first["log_unnormalized_weights"],
        first_features,
        active_indices[0],
        row_scales[0],
    )
    parents = tf.reshape(first_projection["student_points"], [FEATURE_COUNT])
    parent_log_weights = tf.math.log(first_projection["student_weights"])
    valid = first_projection["valid_chart"]
    increments0 = tf.tensor_scatter_nd_update(
        tf.zeros([time_steps], DTYPE), [[0]], [first["increment"]]
    )
    minimum0 = tf.tensor_scatter_nd_update(
        tf.zeros([time_steps - 1], DTYPE),
        [[0]],
        [first_projection["minimum_weight"]],
    )
    condition0 = tf.tensor_scatter_nd_update(
        tf.zeros([time_steps - 1], DTYPE),
        [[0]],
        [first_projection["condition_number"]],
    )
    residual0 = tf.tensor_scatter_nd_update(
        tf.zeros([time_steps - 1, FEATURE_COUNT], DTYPE),
        [[0]],
        [first_projection["feature_residual"]],
    )
    valid0 = tf.tensor_scatter_nd_update(
        tf.zeros([time_steps], tf.bool), [[0]], [valid]
    )

    def cond(
        index: tf.Tensor,
        _parents: tf.Tensor,
        _log_weights: tf.Tensor,
        _total: tf.Tensor,
        _valid: tf.Tensor,
        *_history: tf.Tensor,
    ) -> tf.Tensor:
        del _parents, _log_weights, _total, _valid, _history
        return index < time_steps - 1

    def body(
        index: tf.Tensor,
        current_parents: tf.Tensor,
        current_log_weights: tf.Tensor,
        total: tf.Tensor,
        prior_valid: tf.Tensor,
        increments: tf.Tensor,
        minimum_weights: tf.Tensor,
        condition_numbers: tf.Tensor,
        feature_residuals: tf.Tensor,
        valid_history: tf.Tensor,
    ) -> tuple[tf.Tensor, ...]:
        teacher = _teacher_transition_step_loop(
            spec,
            theta,
            current_parents,
            current_log_weights,
            nodes,
            node_log_weights,
            target_observations[index],
            flow_observations[index],
            index,
        )
        window, future_count = _future_window(
            target_observations, index + 1, lookahead_steps
        )
        features = _features_loop(
            spec,
            theta,
            teacher["particles"],
            window,
            future_count,
            continuation_grid_points,
            continuation_grid_weights,
            first_future_time_index=index + 1,
        )
        projection = tp._contract_e_tp_dense_square_forward_core(
            teacher["particles"],
            teacher["log_unnormalized_weights"],
            features,
            active_indices[index],
            row_scales[index],
        )
        step_valid = prior_valid & projection["valid_chart"]
        next_parents = tf.reshape(projection["student_points"], [FEATURE_COUNT])
        next_log_weights = tf.math.log(projection["student_weights"])
        return (
            index + 1,
            next_parents,
            next_log_weights,
            total + teacher["increment"],
            step_valid,
            tf.tensor_scatter_nd_update(
                increments, [[index]], [teacher["increment"]]
            ),
            tf.tensor_scatter_nd_update(
                minimum_weights, [[index]], [projection["minimum_weight"]]
            ),
            tf.tensor_scatter_nd_update(
                condition_numbers, [[index]], [projection["condition_number"]]
            ),
            tf.tensor_scatter_nd_update(
                feature_residuals, [[index]], [projection["feature_residual"]]
            ),
            tf.tensor_scatter_nd_update(valid_history, [[index]], [step_valid]),
        )

    if time_steps > 2:
        loop_result = tf.while_loop(
            cond,
            body,
            (
                tf.constant(1, tf.int32),
                parents,
                parent_log_weights,
                first["increment"],
                valid,
                increments0,
                minimum0,
                condition0,
                residual0,
                valid0,
            ),
            parallel_iterations=1,
            maximum_iterations=time_steps - 2,
        )
        parents = loop_result[1]
        parent_log_weights = loop_result[2]
        total = loop_result[3]
        valid = loop_result[4]
        increments = loop_result[5]
        minimum_weights = loop_result[6]
        condition_numbers = loop_result[7]
        feature_residuals = loop_result[8]
        valid_history = loop_result[9]
    else:
        total = first["increment"]
        increments = increments0
        minimum_weights = minimum0
        condition_numbers = condition0
        feature_residuals = residual0
        valid_history = valid0

    terminal_index = tf.constant(time_steps - 1, tf.int32)
    terminal = _teacher_transition_step_loop(
        spec,
        theta,
        parents,
        parent_log_weights,
        nodes,
        node_log_weights,
        target_observations[-1],
        flow_observations[-1],
        terminal_index,
    )
    total = total + terminal["increment"]
    increments = tf.tensor_scatter_nd_update(
        increments, [[time_steps - 1]], [terminal["increment"]]
    )
    valid_history = tf.tensor_scatter_nd_update(
        valid_history, [[time_steps - 1]], [valid]
    )
    return {
        "objective": total,
        "increment_history": increments,
        "minimum_weight_history": minimum_weights,
        "condition_number_history": condition_numbers,
        "feature_residual_history": feature_residuals,
        "valid_history": valid_history,
        "final_particles": terminal["particles"],
        "final_log_unnormalized_weights": terminal["log_unnormalized_weights"],
    }


def make_contract_e_tp_scalar_sv_loop_tf(
    spec: ScalarSVContractETPSpec,
    target_observations: tf.Tensor,
    flow_observations: tf.Tensor,
    standard_nodes: tf.Tensor,
    standard_weights: tf.Tensor,
    active_indices: tf.Tensor,
    row_scales: tf.Tensor,
    continuation_grid_points: tf.Tensor,
    continuation_grid_weights: tf.Tensor,
    *,
    lookahead_steps: int,
    jit_compile: bool = True,
):
    """Bind one prepared actual/KSC loop-native scalar and total score."""

    @tf.function(
        input_signature=[tf.TensorSpec([spec.parameter_dimension], DTYPE)],
        jit_compile=jit_compile,
        reduce_retracing=True,
    )
    def evaluate(theta: tf.Tensor) -> dict[str, tf.Tensor]:
        with tf.GradientTape() as tape:
            tape.watch(theta)
            result = contract_e_tp_scalar_sv_loop_core(
                spec,
                theta,
                target_observations,
                flow_observations,
                standard_nodes,
                standard_weights,
                active_indices,
                row_scales,
                continuation_grid_points,
                continuation_grid_weights,
                lookahead_steps=lookahead_steps,
            )
        score = tape.gradient(result["objective"], theta)
        valid = tf.reduce_all(result["valid_history"])
        def poison(value: tf.Tensor) -> tf.Tensor:
            return tf.where(
                valid,
                value,
                tf.fill(tf.shape(value), tf.cast(float("nan"), value.dtype)),
            )
        return {
            **result,
            "valid": valid,
            "objective": poison(result["objective"]),
            "score": poison(score),
            "increment_history": poison(result["increment_history"]),
            "final_particles": poison(result["final_particles"]),
            "final_log_unnormalized_weights": poison(
                result["final_log_unnormalized_weights"]
            ),
        }

    return evaluate


def contract_e_tp_actual_sv_overcomplete_loop_core(
    spec: ScalarSVContractETPSpec,
    theta: tf.Tensor,
    target_observations: tf.Tensor,
    flow_observations: tf.Tensor,
    standard_nodes: tf.Tensor,
    standard_weights: tf.Tensor,
    active_indices: tf.Tensor,
    row_scales: tf.Tensor,
    reference_weights: tf.Tensor,
    continuation_grid_points: tf.Tensor,
    continuation_grid_weights: tf.Tensor,
    *,
    lookahead_steps: int,
) -> dict[str, tf.Tensor]:
    """Execute the fixed-capacity Actual-SV Pearson recursion."""

    if spec.row_id != ACTUAL_SV_ROW_ID:
        raise ValueError("the overcomplete loop is scoped to Actual SV")
    if spec.transition_before_first_observation:
        raise ValueError("Actual-SV overcomplete loop uses observation at initialization")
    if lookahead_steps < 1:
        raise ValueError("lookahead_steps must be positive")
    theta = tf.reshape(tf.convert_to_tensor(theta, DTYPE), [2])
    target_observations = tf.reshape(
        tf.convert_to_tensor(target_observations, DTYPE), [-1, 1]
    )
    flow_observations = tf.reshape(
        tf.convert_to_tensor(flow_observations, DTYPE), [-1, 1]
    )
    standard_nodes = tf.reshape(tf.convert_to_tensor(standard_nodes, DTYPE), [-1])
    standard_weights = tf.reshape(
        tf.convert_to_tensor(standard_weights, DTYPE), [-1]
    )
    active_indices = tf.convert_to_tensor(active_indices, tf.int32)
    row_scales = tf.convert_to_tensor(row_scales, DTYPE)
    reference_weights = tf.convert_to_tensor(reference_weights, DTYPE)
    time_steps = target_observations.shape[0]
    innovation_count = standard_nodes.shape[0]
    anchor_count = active_indices.shape[1]
    if (
        time_steps is None
        or time_steps < 1
        or innovation_count is None
        or anchor_count is None
    ):
        raise ValueError("overcomplete loop requires static positive shapes")
    if anchor_count < FEATURE_COUNT:
        raise ValueError("overcomplete capacity must be at least the feature count")
    if active_indices.shape != (time_steps - 1, anchor_count):
        raise ValueError("active_indices has the wrong overcomplete loop shape")
    if row_scales.shape != (time_steps - 1, FEATURE_COUNT):
        raise ValueError("row_scales has the wrong overcomplete loop shape")
    if reference_weights.shape != (time_steps - 1, anchor_count):
        raise ValueError("reference_weights has the wrong overcomplete loop shape")

    parents0, parent_log_weights0, nodes, node_log_weights = initial_rule(
        spec, theta, standard_nodes, standard_weights
    )
    first = _teacher_step(
        spec,
        theta,
        parents0,
        parent_log_weights0,
        nodes,
        node_log_weights,
        target_observations[0],
        flow_observations[0],
        0,
    )
    if time_steps == 1:
        return {
            "objective": first["increment"],
            "increment_history": first["increment"][None],
            "minimum_weight_history": tf.zeros([0], DTYPE),
            "matrix_condition_number_history": tf.zeros([0], DTYPE),
            "gram_condition_number_history": tf.zeros([0], DTYPE),
            "scaled_relative_residual_history": tf.zeros([0], DTYPE),
            "feature_residual_history": tf.zeros([0, FEATURE_COUNT], DTYPE),
            "valid_history": tf.ones([1], tf.bool),
            "final_particles": first["particles"],
            "final_log_unnormalized_weights": first["log_unnormalized_weights"],
        }

    first_window, first_count = _future_window(
        target_observations, tf.constant(1, tf.int32), lookahead_steps
    )
    first_features = _features_loop(
        spec,
        theta,
        first["particles"],
        first_window,
        first_count,
        continuation_grid_points,
        continuation_grid_weights,
        first_future_time_index=tf.constant(1, tf.int32),
    )
    first_projection = tp._contract_e_tp_diagonal_kkt_forward_core(
        first["particles"],
        first["log_unnormalized_weights"],
        first_features,
        active_indices[0],
        row_scales[0],
        reference_weights[0],
    )
    valid = first_projection["valid_chart"]
    parents = tf.reshape(
        tf.where(
            valid,
            first_projection["student_points"],
            tf.zeros_like(first_projection["student_points"]),
        ),
        [anchor_count],
    )
    safe_first_weights = tf.where(
        valid,
        first_projection["student_weights"],
        tf.fill([anchor_count], tf.math.reciprocal(tf.cast(anchor_count, DTYPE))),
    )
    parent_log_weights = tf.math.log(safe_first_weights)
    increments0 = tf.tensor_scatter_nd_update(
        tf.zeros([time_steps], DTYPE), [[0]], [first["increment"]]
    )
    minimum0 = tf.tensor_scatter_nd_update(
        tf.zeros([time_steps - 1], DTYPE),
        [[0]],
        [first_projection["minimum_weight"]],
    )
    matrix_condition0 = tf.tensor_scatter_nd_update(
        tf.zeros([time_steps - 1], DTYPE),
        [[0]],
        [first_projection["matrix_condition_number"]],
    )
    gram_condition0 = tf.tensor_scatter_nd_update(
        tf.zeros([time_steps - 1], DTYPE),
        [[0]],
        [first_projection["gram_condition_number"]],
    )
    scaled_residual0 = tf.tensor_scatter_nd_update(
        tf.zeros([time_steps - 1], DTYPE),
        [[0]],
        [first_projection["scaled_relative_residual"]],
    )
    feature_residual0 = tf.tensor_scatter_nd_update(
        tf.zeros([time_steps - 1, FEATURE_COUNT], DTYPE),
        [[0]],
        [first_projection["feature_residual"]],
    )
    valid0 = tf.tensor_scatter_nd_update(
        tf.zeros([time_steps], tf.bool), [[0]], [valid]
    )

    def cond(index: tf.Tensor, *_state: tf.Tensor) -> tf.Tensor:
        return index < time_steps - 1

    def body(
        index: tf.Tensor,
        current_parents: tf.Tensor,
        current_log_weights: tf.Tensor,
        total: tf.Tensor,
        prior_valid: tf.Tensor,
        increments: tf.Tensor,
        minimum_weights: tf.Tensor,
        matrix_conditions: tf.Tensor,
        gram_conditions: tf.Tensor,
        scaled_residuals: tf.Tensor,
        feature_residuals: tf.Tensor,
        valid_history: tf.Tensor,
    ) -> tuple[tf.Tensor, ...]:
        teacher = _teacher_transition_step_loop(
            spec,
            theta,
            current_parents,
            current_log_weights,
            nodes,
            node_log_weights,
            target_observations[index],
            flow_observations[index],
            index,
        )
        window, future_count = _future_window(
            target_observations, index + 1, lookahead_steps
        )
        features = _features_loop(
            spec,
            theta,
            teacher["particles"],
            window,
            future_count,
            continuation_grid_points,
            continuation_grid_weights,
            first_future_time_index=index + 1,
        )
        projection = tp._contract_e_tp_diagonal_kkt_forward_core(
            teacher["particles"],
            teacher["log_unnormalized_weights"],
            features,
            active_indices[index],
            row_scales[index],
            reference_weights[index],
        )
        step_valid = prior_valid & projection["valid_chart"]
        next_parents = tf.reshape(
            tf.where(
                step_valid,
                projection["student_points"],
                tf.zeros_like(projection["student_points"]),
            ),
            [anchor_count],
        )
        safe_weights = tf.where(
            step_valid,
            projection["student_weights"],
            tf.fill(
                [anchor_count], tf.math.reciprocal(tf.cast(anchor_count, DTYPE))
            ),
        )
        return (
            index + 1,
            next_parents,
            tf.math.log(safe_weights),
            total + teacher["increment"],
            step_valid,
            tf.tensor_scatter_nd_update(increments, [[index]], [teacher["increment"]]),
            tf.tensor_scatter_nd_update(
                minimum_weights, [[index]], [projection["minimum_weight"]]
            ),
            tf.tensor_scatter_nd_update(
                matrix_conditions,
                [[index]],
                [projection["matrix_condition_number"]],
            ),
            tf.tensor_scatter_nd_update(
                gram_conditions, [[index]], [projection["gram_condition_number"]]
            ),
            tf.tensor_scatter_nd_update(
                scaled_residuals,
                [[index]],
                [projection["scaled_relative_residual"]],
            ),
            tf.tensor_scatter_nd_update(
                feature_residuals, [[index]], [projection["feature_residual"]]
            ),
            tf.tensor_scatter_nd_update(valid_history, [[index]], [step_valid]),
        )

    if time_steps > 2:
        loop_result = tf.while_loop(
            cond,
            body,
            (
                tf.constant(1, tf.int32),
                parents,
                parent_log_weights,
                first["increment"],
                valid,
                increments0,
                minimum0,
                matrix_condition0,
                gram_condition0,
                scaled_residual0,
                feature_residual0,
                valid0,
            ),
            parallel_iterations=1,
            maximum_iterations=time_steps - 2,
        )
        parents = loop_result[1]
        parent_log_weights = loop_result[2]
        total = loop_result[3]
        valid = loop_result[4]
        increments = loop_result[5]
        minimum_weights = loop_result[6]
        matrix_conditions = loop_result[7]
        gram_conditions = loop_result[8]
        scaled_residuals = loop_result[9]
        feature_residuals = loop_result[10]
        valid_history = loop_result[11]
    else:
        total = first["increment"]
        increments = increments0
        minimum_weights = minimum0
        matrix_conditions = matrix_condition0
        gram_conditions = gram_condition0
        scaled_residuals = scaled_residual0
        feature_residuals = feature_residual0
        valid_history = valid0

    terminal_index = tf.constant(time_steps - 1, tf.int32)
    terminal = _teacher_transition_step_loop(
        spec,
        theta,
        parents,
        parent_log_weights,
        nodes,
        node_log_weights,
        target_observations[-1],
        flow_observations[-1],
        terminal_index,
    )
    total += terminal["increment"]
    increments = tf.tensor_scatter_nd_update(
        increments, [[time_steps - 1]], [terminal["increment"]]
    )
    valid_history = tf.tensor_scatter_nd_update(
        valid_history, [[time_steps - 1]], [valid]
    )
    return {
        "objective": total,
        "increment_history": increments,
        "minimum_weight_history": minimum_weights,
        "matrix_condition_number_history": matrix_conditions,
        "gram_condition_number_history": gram_conditions,
        "scaled_relative_residual_history": scaled_residuals,
        "feature_residual_history": feature_residuals,
        "valid_history": valid_history,
        "final_particles": terminal["particles"],
        "final_log_unnormalized_weights": terminal["log_unnormalized_weights"],
    }


def make_contract_e_tp_actual_sv_overcomplete_tf(
    spec: ScalarSVContractETPSpec,
    target_observations: tf.Tensor,
    flow_observations: tf.Tensor,
    standard_nodes: tf.Tensor,
    standard_weights: tf.Tensor,
    active_indices: tf.Tensor,
    row_scales: tf.Tensor,
    reference_weights: tf.Tensor,
    continuation_grid_points: tf.Tensor,
    continuation_grid_weights: tf.Tensor,
    *,
    lookahead_steps: int,
    jit_compile: bool = True,
):
    """Bind one fixed overcomplete Actual-SV scalar and autodiff oracle score."""

    @tf.function(
        input_signature=[tf.TensorSpec([2], DTYPE)],
        jit_compile=jit_compile,
        reduce_retracing=True,
    )
    def evaluate(theta: tf.Tensor) -> dict[str, tf.Tensor]:
        with tf.GradientTape() as tape:
            tape.watch(theta)
            result = contract_e_tp_actual_sv_overcomplete_loop_core(
                spec,
                theta,
                target_observations,
                flow_observations,
                standard_nodes,
                standard_weights,
                active_indices,
                row_scales,
                reference_weights,
                continuation_grid_points,
                continuation_grid_weights,
                lookahead_steps=lookahead_steps,
            )
        score = tape.gradient(result["objective"], theta)
        valid = tf.reduce_all(result["valid_history"])
        return {
            **result,
            "valid": valid,
            "objective": tp._poison_invalid(result["objective"], valid),
            "score_autodiff_oracle": tp._poison_invalid(score, valid),
            "increment_history": tp._poison_invalid(
                result["increment_history"], valid
            ),
            "final_particles": tp._poison_invalid(result["final_particles"], valid),
            "final_log_unnormalized_weights": tp._poison_invalid(
                result["final_log_unnormalized_weights"], valid
            ),
        }

    return evaluate


def make_contract_e_tp_actual_sv_overcomplete_forward_ad_tf(
    spec: ScalarSVContractETPSpec,
    target_observations: tf.Tensor,
    flow_observations: tf.Tensor,
    standard_nodes: tf.Tensor,
    standard_weights: tf.Tensor,
    active_indices: tf.Tensor,
    row_scales: tf.Tensor,
    reference_weights: tf.Tensor,
    continuation_grid_points: tf.Tensor,
    continuation_grid_weights: tf.Tensor,
    *,
    lookahead_steps: int,
    jit_compile: bool = True,
):
    """Bind an independent automatic forward-mode directional derivative."""

    @tf.function(
        input_signature=[tf.TensorSpec([2], DTYPE), tf.TensorSpec([2], DTYPE)],
        jit_compile=jit_compile,
        reduce_retracing=True,
    )
    def evaluate(theta: tf.Tensor, direction: tf.Tensor) -> dict[str, tf.Tensor]:
        with tf.autodiff.ForwardAccumulator(theta, direction) as accumulator:
            result = contract_e_tp_actual_sv_overcomplete_loop_core(
                spec,
                theta,
                target_observations,
                flow_observations,
                standard_nodes,
                standard_weights,
                active_indices,
                row_scales,
                reference_weights,
                continuation_grid_points,
                continuation_grid_weights,
                lookahead_steps=lookahead_steps,
            )
        directional_derivative = accumulator.jvp(result["objective"])
        valid = tf.reduce_all(result["valid_history"])
        return {
            "valid": valid,
            "objective": tp._poison_invalid(result["objective"], valid),
            "directional_derivative": tp._poison_invalid(
                directional_derivative, valid
            ),
        }

    return evaluate


def make_contract_e_tp_actual_sv_overcomplete_forward_tf(
    spec: ScalarSVContractETPSpec,
    target_observations: tf.Tensor,
    flow_observations: tf.Tensor,
    standard_nodes: tf.Tensor,
    standard_weights: tf.Tensor,
    active_indices: tf.Tensor,
    row_scales: tf.Tensor,
    reference_weights: tf.Tensor,
    continuation_grid_points: tf.Tensor,
    continuation_grid_weights: tf.Tensor,
    *,
    lookahead_steps: int,
    jit_compile: bool = True,
):
    """Bind the fixed overcomplete Actual-SV value and chart diagnostics."""

    @tf.function(
        input_signature=[tf.TensorSpec([2], DTYPE)],
        jit_compile=jit_compile,
        reduce_retracing=True,
    )
    def evaluate(theta: tf.Tensor) -> dict[str, tf.Tensor]:
        result = contract_e_tp_actual_sv_overcomplete_loop_core(
            spec,
            theta,
            target_observations,
            flow_observations,
            standard_nodes,
            standard_weights,
            active_indices,
            row_scales,
            reference_weights,
            continuation_grid_points,
            continuation_grid_weights,
            lookahead_steps=lookahead_steps,
        )
        valid = tf.reduce_all(result["valid_history"])
        return {
            **result,
            "valid": valid,
            "objective": tp._poison_invalid(result["objective"], valid),
            "increment_history": tp._poison_invalid(
                result["increment_history"], valid
            ),
            "final_particles": tp._poison_invalid(result["final_particles"], valid),
            "final_log_unnormalized_weights": tp._poison_invalid(
                result["final_log_unnormalized_weights"], valid
            ),
        }

    return evaluate


def contract_e_tp_actual_sv_overcomplete_manual_jvp_loop_core(
    spec: ScalarSVContractETPSpec,
    theta: tf.Tensor,
    target_observations: tf.Tensor,
    flow_observations: tf.Tensor,
    standard_nodes: tf.Tensor,
    standard_weights: tf.Tensor,
    active_indices: tf.Tensor,
    row_scales: tf.Tensor,
    reference_weights: tf.Tensor,
    continuation_grid_points: tf.Tensor,
    continuation_grid_weights: tf.Tensor,
    *,
    lookahead_steps: int,
) -> dict[str, tf.Tensor]:
    """Execute the Actual-SV value and explicit two-direction total JVP."""

    if spec.row_id != ACTUAL_SV_ROW_ID:
        raise ValueError("manual overcomplete JVP is scoped to Actual SV")
    theta = tf.reshape(tf.convert_to_tensor(theta, DTYPE), [2])
    target_observations = tf.reshape(
        tf.convert_to_tensor(target_observations, DTYPE), [-1, 1]
    )
    flow_observations = tf.reshape(
        tf.convert_to_tensor(flow_observations, DTYPE), [-1, 1]
    )
    standard_nodes = tf.reshape(tf.convert_to_tensor(standard_nodes, DTYPE), [-1])
    standard_weights = tf.reshape(
        tf.convert_to_tensor(standard_weights, DTYPE), [-1]
    )
    standard_log_weights = tf.math.log(standard_weights)
    active_indices = tf.convert_to_tensor(active_indices, tf.int32)
    row_scales = tf.convert_to_tensor(row_scales, DTYPE)
    reference_weights = tf.convert_to_tensor(reference_weights, DTYPE)
    time_steps = target_observations.shape[0]
    anchor_count = active_indices.shape[1]
    if time_steps is None or time_steps < 1 or anchor_count is None:
        raise ValueError("manual overcomplete loop requires static positive shapes")
    if active_indices.shape != (time_steps - 1, anchor_count):
        raise ValueError("manual active_indices shape mismatch")
    if row_scales.shape != (time_steps - 1, FEATURE_COUNT):
        raise ValueError("manual row_scales shape mismatch")
    if reference_weights.shape != (time_steps - 1, anchor_count):
        raise ValueError("manual reference_weights shape mismatch")

    first = _actual_sv_initial_teacher_multi_jvp(
        spec,
        theta,
        standard_nodes,
        standard_log_weights,
        target_observations[0],
        flow_observations[0],
    )
    if time_steps == 1:
        return {
            "objective": first["increment"],
            "score_manual": first["increment_tangent"],
            "increment_history": first["increment"][None],
            "score_increment_history": first["increment_tangent"][None, :],
            "minimum_weight_history": tf.zeros([0], DTYPE),
            "gram_condition_number_history": tf.zeros([0], DTYPE),
            "scaled_relative_residual_history": tf.zeros([0], DTYPE),
            "feature_residual_history": tf.zeros([0, FEATURE_COUNT], DTYPE),
            "valid_history": tf.ones([1], tf.bool),
            "final_particles": first["particles"],
            "final_particle_tangents": first["particles_tangent"],
            "final_log_unnormalized_weights": first["log_unnormalized_weights"],
            "final_log_unnormalized_weight_tangents": first[
                "log_unnormalized_weights_tangent"
            ],
        }

    first_window, first_count = _future_window(
        target_observations, tf.constant(1, tf.int32), lookahead_steps
    )
    first_features, first_feature_tangents = _actual_sv_features_multi_jvp(
        spec,
        theta,
        first["particles"],
        first["particles_tangent"],
        first_window,
        first_count,
        continuation_grid_points,
        continuation_grid_weights,
        first_future_time_index=tf.constant(1, tf.int32),
    )
    first_projection = tp._contract_e_tp_diagonal_kkt_multi_jvp_core(
        first["particles"],
        first["log_unnormalized_weights"],
        first_features,
        active_indices[0],
        row_scales[0],
        reference_weights[0],
        first["particles_tangent"],
        first["log_unnormalized_weights_tangent"],
        first_feature_tangents,
    )
    valid = first_projection["valid_chart"]
    parents = tf.reshape(
        tf.where(
            valid,
            first_projection["student_points"],
            tf.zeros_like(first_projection["student_points"]),
        ),
        [anchor_count],
    )
    parent_tangents = tf.reshape(
        tf.where(
            valid,
            first_projection["student_points_tangent"],
            tf.zeros_like(first_projection["student_points_tangent"]),
        ),
        [anchor_count, 2],
    )
    safe_first_weights = tf.where(
        valid,
        first_projection["student_weights"],
        tf.fill([anchor_count], tf.math.reciprocal(tf.cast(anchor_count, DTYPE))),
    )
    safe_first_weight_tangents = tf.where(
        valid,
        first_projection["student_weights_tangent"],
        tf.zeros([anchor_count, 2], DTYPE),
    )
    parent_log_weights = tf.math.log(safe_first_weights)
    parent_log_weight_tangents = (
        safe_first_weight_tangents / safe_first_weights[:, None]
    )
    increments0 = tf.tensor_scatter_nd_update(
        tf.zeros([time_steps], DTYPE), [[0]], [first["increment"]]
    )
    score_increments0 = tf.tensor_scatter_nd_update(
        tf.zeros([time_steps, 2], DTYPE),
        [[0]],
        [first["increment_tangent"]],
    )
    minimum0 = tf.tensor_scatter_nd_update(
        tf.zeros([time_steps - 1], DTYPE),
        [[0]],
        [first_projection["minimum_weight"]],
    )
    gram_condition0 = tf.tensor_scatter_nd_update(
        tf.zeros([time_steps - 1], DTYPE),
        [[0]],
        [first_projection["gram_condition_number"]],
    )
    scaled_residual0 = tf.tensor_scatter_nd_update(
        tf.zeros([time_steps - 1], DTYPE),
        [[0]],
        [first_projection["scaled_relative_residual"]],
    )
    feature_residual0 = tf.tensor_scatter_nd_update(
        tf.zeros([time_steps - 1, FEATURE_COUNT], DTYPE),
        [[0]],
        [first_projection["feature_residual"]],
    )
    valid0 = tf.tensor_scatter_nd_update(
        tf.zeros([time_steps], tf.bool), [[0]], [valid]
    )

    def cond(index: tf.Tensor, *_state: tf.Tensor) -> tf.Tensor:
        return index < time_steps - 1

    def body(
        index: tf.Tensor,
        current_parents: tf.Tensor,
        current_parent_tangents: tf.Tensor,
        current_log_weights: tf.Tensor,
        current_log_weight_tangents: tf.Tensor,
        total: tf.Tensor,
        total_tangent: tf.Tensor,
        prior_valid: tf.Tensor,
        increments: tf.Tensor,
        score_increments: tf.Tensor,
        minimum_weights: tf.Tensor,
        gram_conditions: tf.Tensor,
        scaled_residuals: tf.Tensor,
        feature_residuals: tf.Tensor,
        valid_history: tf.Tensor,
    ) -> tuple[tf.Tensor, ...]:
        teacher = _actual_sv_transition_teacher_multi_jvp(
            spec,
            theta,
            current_parents,
            current_parent_tangents,
            current_log_weights,
            current_log_weight_tangents,
            standard_nodes,
            standard_log_weights,
            target_observations[index],
            flow_observations[index],
        )
        window, future_count = _future_window(
            target_observations, index + 1, lookahead_steps
        )
        features, feature_tangents = _actual_sv_features_multi_jvp(
            spec,
            theta,
            teacher["particles"],
            teacher["particles_tangent"],
            window,
            future_count,
            continuation_grid_points,
            continuation_grid_weights,
            first_future_time_index=index + 1,
        )
        projection = tp._contract_e_tp_diagonal_kkt_multi_jvp_core(
            teacher["particles"],
            teacher["log_unnormalized_weights"],
            features,
            active_indices[index],
            row_scales[index],
            reference_weights[index],
            teacher["particles_tangent"],
            teacher["log_unnormalized_weights_tangent"],
            feature_tangents,
        )
        step_valid = prior_valid & projection["valid_chart"]
        next_parents = tf.reshape(
            tf.where(
                step_valid,
                projection["student_points"],
                tf.zeros_like(projection["student_points"]),
            ),
            [anchor_count],
        )
        next_parent_tangents = tf.reshape(
            tf.where(
                step_valid,
                projection["student_points_tangent"],
                tf.zeros_like(projection["student_points_tangent"]),
            ),
            [anchor_count, 2],
        )
        safe_weights = tf.where(
            step_valid,
            projection["student_weights"],
            tf.fill(
                [anchor_count], tf.math.reciprocal(tf.cast(anchor_count, DTYPE))
            ),
        )
        safe_weight_tangents = tf.where(
            step_valid,
            projection["student_weights_tangent"],
            tf.zeros([anchor_count, 2], DTYPE),
        )
        return (
            index + 1,
            next_parents,
            next_parent_tangents,
            tf.math.log(safe_weights),
            safe_weight_tangents / safe_weights[:, None],
            total + teacher["increment"],
            total_tangent + teacher["increment_tangent"],
            step_valid,
            tf.tensor_scatter_nd_update(increments, [[index]], [teacher["increment"]]),
            tf.tensor_scatter_nd_update(
                score_increments, [[index]], [teacher["increment_tangent"]]
            ),
            tf.tensor_scatter_nd_update(
                minimum_weights, [[index]], [projection["minimum_weight"]]
            ),
            tf.tensor_scatter_nd_update(
                gram_conditions, [[index]], [projection["gram_condition_number"]]
            ),
            tf.tensor_scatter_nd_update(
                scaled_residuals,
                [[index]],
                [projection["scaled_relative_residual"]],
            ),
            tf.tensor_scatter_nd_update(
                feature_residuals, [[index]], [projection["feature_residual"]]
            ),
            tf.tensor_scatter_nd_update(valid_history, [[index]], [step_valid]),
        )

    if time_steps > 2:
        loop_result = tf.while_loop(
            cond,
            body,
            (
                tf.constant(1, tf.int32),
                parents,
                parent_tangents,
                parent_log_weights,
                parent_log_weight_tangents,
                first["increment"],
                first["increment_tangent"],
                valid,
                increments0,
                score_increments0,
                minimum0,
                gram_condition0,
                scaled_residual0,
                feature_residual0,
                valid0,
            ),
            parallel_iterations=1,
            maximum_iterations=time_steps - 2,
        )
        parents = loop_result[1]
        parent_tangents = loop_result[2]
        parent_log_weights = loop_result[3]
        parent_log_weight_tangents = loop_result[4]
        total = loop_result[5]
        total_tangent = loop_result[6]
        valid = loop_result[7]
        increments = loop_result[8]
        score_increments = loop_result[9]
        minimum_weights = loop_result[10]
        gram_conditions = loop_result[11]
        scaled_residuals = loop_result[12]
        feature_residuals = loop_result[13]
        valid_history = loop_result[14]
    else:
        total = first["increment"]
        total_tangent = first["increment_tangent"]
        increments = increments0
        score_increments = score_increments0
        minimum_weights = minimum0
        gram_conditions = gram_condition0
        scaled_residuals = scaled_residual0
        feature_residuals = feature_residual0
        valid_history = valid0

    terminal = _actual_sv_transition_teacher_multi_jvp(
        spec,
        theta,
        parents,
        parent_tangents,
        parent_log_weights,
        parent_log_weight_tangents,
        standard_nodes,
        standard_log_weights,
        target_observations[-1],
        flow_observations[-1],
    )
    total += terminal["increment"]
    total_tangent += terminal["increment_tangent"]
    increments = tf.tensor_scatter_nd_update(
        increments, [[time_steps - 1]], [terminal["increment"]]
    )
    score_increments = tf.tensor_scatter_nd_update(
        score_increments,
        [[time_steps - 1]],
        [terminal["increment_tangent"]],
    )
    valid_history = tf.tensor_scatter_nd_update(
        valid_history, [[time_steps - 1]], [valid]
    )
    return {
        "objective": total,
        "score_manual": total_tangent,
        "increment_history": increments,
        "score_increment_history": score_increments,
        "minimum_weight_history": minimum_weights,
        "gram_condition_number_history": gram_conditions,
        "scaled_relative_residual_history": scaled_residuals,
        "feature_residual_history": feature_residuals,
        "valid_history": valid_history,
        "final_particles": terminal["particles"],
        "final_particle_tangents": terminal["particles_tangent"],
        "final_log_unnormalized_weights": terminal["log_unnormalized_weights"],
        "final_log_unnormalized_weight_tangents": terminal[
            "log_unnormalized_weights_tangent"
        ],
    }


def make_contract_e_tp_actual_sv_overcomplete_manual_jvp_tf(
    spec: ScalarSVContractETPSpec,
    target_observations: tf.Tensor,
    flow_observations: tf.Tensor,
    standard_nodes: tf.Tensor,
    standard_weights: tf.Tensor,
    active_indices: tf.Tensor,
    row_scales: tf.Tensor,
    reference_weights: tf.Tensor,
    continuation_grid_points: tf.Tensor,
    continuation_grid_weights: tf.Tensor,
    *,
    lookahead_steps: int,
    jit_compile: bool = True,
):
    """Bind the fixed Actual-SV scalar and explicit total manual score."""

    @tf.function(
        input_signature=[tf.TensorSpec([2], DTYPE)],
        jit_compile=jit_compile,
        reduce_retracing=True,
    )
    def evaluate(theta: tf.Tensor) -> dict[str, tf.Tensor]:
        result = contract_e_tp_actual_sv_overcomplete_manual_jvp_loop_core(
            spec,
            theta,
            target_observations,
            flow_observations,
            standard_nodes,
            standard_weights,
            active_indices,
            row_scales,
            reference_weights,
            continuation_grid_points,
            continuation_grid_weights,
            lookahead_steps=lookahead_steps,
        )
        valid = tf.reduce_all(result["valid_history"])
        return {
            **result,
            "valid": valid,
            "objective": tp._poison_invalid(result["objective"], valid),
            "score_manual": tp._poison_invalid(result["score_manual"], valid),
            "increment_history": tp._poison_invalid(
                result["increment_history"], valid
            ),
            "score_increment_history": tp._poison_invalid(
                result["score_increment_history"], valid
            ),
            "final_particles": tp._poison_invalid(result["final_particles"], valid),
            "final_log_unnormalized_weights": tp._poison_invalid(
                result["final_log_unnormalized_weights"], valid
            ),
        }

    return evaluate


def prepare_actual_sv_overcomplete_chart_step(
    teacher_points: tf.Tensor,
    log_unnormalized_weights: tf.Tensor,
    teacher_features: tf.Tensor,
    anchor_count: int,
) -> dict[str, tf.Tensor]:
    """Construct one deterministic center Pearson chart using TensorFlow only."""

    if anchor_count < FEATURE_COUNT:
        raise ValueError("anchor_count must be at least the feature count")
    points = tf.reshape(tf.convert_to_tensor(teacher_points, DTYPE), [-1, 1])
    log_weights = tf.reshape(
        tf.convert_to_tensor(log_unnormalized_weights, DTYPE), [-1]
    )
    features = tf.convert_to_tensor(teacher_features, DTYPE)
    teacher_count = points.shape[0]
    if teacher_count is None or teacher_count < anchor_count:
        raise ValueError("teacher must contain at least anchor_count points")
    teacher = tp._dense_teacher_reduce_core(log_weights, features)
    normalized = teacher["normalized_weights"]
    values = points[:, 0]
    ordering = tf.argsort(values, stable=True)
    cumulative = tf.cumsum(tf.gather(normalized, ordering))
    probabilities = (
        tf.cast(tf.range(anchor_count), DTYPE) + tf.constant(0.5, DTYPE)
    ) / tf.cast(anchor_count, DTYPE)
    sorted_positions = tf.searchsorted(cumulative, probabilities, side="left")
    sorted_positions = tf.minimum(sorted_positions, teacher_count - 1)
    quantile_indices = tf.gather(ordering, sorted_positions)
    distinct_quantiles, _ = tf.unique(quantile_indices)
    mass_order = tf.argsort(normalized, direction="DESCENDING", stable=True)
    already_selected = tf.reduce_any(
        mass_order[:, None] == distinct_quantiles[None, :], axis=1
    )
    fill_indices = tf.boolean_mask(mass_order, ~already_selected)
    needed = anchor_count - tf.size(distinct_quantiles)
    active_indices = tf.concat(
        [distinct_quantiles, fill_indices[:needed]], axis=0
    )
    active_indices = tf.ensure_shape(active_indices, [anchor_count])
    anchor_values = tf.gather(values, active_indices)
    assignments = tf.argmin(
        tf.abs(values[:, None] - anchor_values[None, :]),
        axis=1,
        output_type=tf.int32,
    )
    voronoi_weights = tf.math.unsorted_segment_sum(
        normalized, assignments, anchor_count
    )
    target = teacher["target"]
    row_scale = tf.maximum(
        tf.reduce_max(tf.abs(features), axis=1), tf.abs(target)
    )
    active_features = tf.gather(features, active_indices, axis=1)
    scaled_matrix = active_features / row_scale[:, None]
    scaled_target = target / row_scale
    inverse_precision_features = voronoi_weights[:, None] * tf.transpose(
        scaled_matrix
    )
    gram = tf.linalg.matmul(scaled_matrix, inverse_precision_features)
    reference_weights = voronoi_weights + tf.linalg.matvec(
        inverse_precision_features,
        tf.linalg.solve(
            gram,
            (
                scaled_target
                - tf.linalg.matvec(scaled_matrix, voronoi_weights)
            )[:, None],
        )[:, 0],
    )
    projection = tp._contract_e_tp_diagonal_kkt_forward_core(
        points,
        log_weights,
        features,
        active_indices,
        row_scale,
        reference_weights,
    )
    preparation_valid = (
        tf.equal(tf.size(tf.unique(active_indices)[0]), anchor_count)
        & tf.reduce_all(voronoi_weights > 0.0)
        & tf.reduce_all(reference_weights > 0.0)
        & projection["valid_chart"]
    )
    return {
        **projection,
        "preparation_valid": preparation_valid,
        "quantile_probabilities": probabilities,
        "quantile_indices_before_deduplication": quantile_indices,
        "active_indices": active_indices,
        "voronoi_weights": voronoi_weights,
        "reference_weights": reference_weights,
        "row_scale": row_scale,
    }


def _features(
    spec: ScalarSVContractETPSpec,
    theta: tf.Tensor,
    points: tf.Tensor,
    future_target_observations: tf.Tensor,
    grid_points: tf.Tensor,
    grid_weights: tf.Tensor,
    *,
    first_future_time_index: int,
) -> tf.Tensor:
    values = tf.reshape(tf.convert_to_tensor(points, DTYPE), [-1])
    continuation_log = target_continuation_log_likelihood(
        spec,
        theta,
        values[:, None],
        future_target_observations,
        grid_points,
        grid_weights,
        first_future_time_index=first_future_time_index,
    )
    reference, _stationary_scale, _gamma, _process_scale = _dynamics(spec, theta)
    reference_log = target_continuation_log_likelihood(
        spec,
        theta,
        reference[None, None],
        future_target_observations,
        grid_points,
        grid_weights,
        first_future_time_index=first_future_time_index,
    )[0]
    return tf.stack(
        [
            tf.ones_like(values),
            values,
            tf.square(values),
            tf.exp(continuation_log - reference_log),
        ],
        axis=0,
    )


def effective_progressive_lookaheads(
    requested_lookaheads: tuple[int, ...], future_count: int
) -> tuple[int, ...]:
    """Return distinct available prefix lengths in deterministic order.

    Near the end of a record, two requested lookaheads can describe the same
    remaining future.  Keeping both would duplicate a feature row and make the
    equality chart rank deficient, so the finite program removes duplicates
    before chart preparation rather than switching charts at runtime.
    """

    if future_count < 1:
        return ()
    if not requested_lookaheads or any(int(value) < 1 for value in requested_lookaheads):
        raise ValueError("progressive lookaheads must be positive")
    if tuple(sorted(set(requested_lookaheads))) != tuple(requested_lookaheads):
        raise ValueError("progressive lookaheads must be strictly increasing")
    return tuple(dict.fromkeys(min(int(value), int(future_count)) for value in requested_lookaheads))


def progressive_features(
    spec: ScalarSVContractETPSpec,
    theta: tf.Tensor,
    points: tf.Tensor,
    future_target_observations: tf.Tensor,
    grid_points: tf.Tensor,
    grid_weights: tf.Tensor,
    *,
    first_future_time_index: int,
    requested_lookaheads: tuple[int, ...],
) -> tf.Tensor:
    """Evaluate moments and distinct progressive target-continuation marks."""

    values = tf.reshape(tf.convert_to_tensor(points, DTYPE), [-1])
    future = tf.reshape(
        tf.convert_to_tensor(future_target_observations, DTYPE), [-1, 1]
    )
    future_count = future.shape[0]
    if future_count is None:
        raise ValueError("progressive features require a static future length")
    realized = effective_progressive_lookaheads(
        requested_lookaheads, int(future_count)
    )
    reference, _stationary_scale, _gamma, _process_scale = _dynamics(spec, theta)
    rows = [tf.ones_like(values), values, tf.square(values)]
    for lookahead in realized:
        observations = future[:lookahead]
        continuation_log = target_continuation_log_likelihood(
            spec,
            theta,
            values[:, None],
            observations,
            grid_points,
            grid_weights,
            first_future_time_index=first_future_time_index,
        )
        reference_log = target_continuation_log_likelihood(
            spec,
            theta,
            reference[None, None],
            observations,
            grid_points,
            grid_weights,
            first_future_time_index=first_future_time_index,
        )[0]
        rows.append(tf.exp(continuation_log - reference_log))
    return tf.stack(rows, axis=0)


def contract_e_tp_scalar_sv_recursive_kkt_core(
    spec: ScalarSVContractETPSpec,
    theta: tf.Tensor,
    target_observations: tf.Tensor,
    flow_observations: tf.Tensor,
    standard_nodes: tf.Tensor,
    standard_weights: tf.Tensor,
    active_indices: tuple[tf.Tensor, ...],
    row_scales: tuple[tf.Tensor, ...],
    reference_weights: tuple[tf.Tensor, ...],
    precisions: tuple[tf.Tensor, ...],
    continuation_grid_points: tf.Tensor,
    continuation_grid_weights: tf.Tensor,
    *,
    requested_lookaheads: tuple[int, ...],
) -> dict[str, object]:
    """Execute a fixed overcomplete progressive-feature scalar-SV program.

    Every time-index chart, reference vector, and precision matrix is prepared
    outside this function and remains fixed.  Feature values and all target
    continuation marks retain their total parameter dependence.
    """

    theta = tf.reshape(tf.convert_to_tensor(theta, DTYPE), [spec.parameter_dimension])
    target_observations = tf.reshape(
        tf.convert_to_tensor(target_observations, DTYPE), [-1, 1]
    )
    flow_observations = tf.reshape(
        tf.convert_to_tensor(flow_observations, DTYPE), [-1, 1]
    )
    time_steps = target_observations.shape[0]
    if time_steps is None or time_steps < 1:
        raise ValueError("scalar-SV Contract E--TP requires a static positive horizon")
    expected = time_steps - 1
    if not (
        len(active_indices)
        == len(row_scales)
        == len(reference_weights)
        == len(precisions)
        == expected
    ):
        raise ValueError("prepared KKT chart sequences must have time_steps - 1 entries")
    effective_progressive_lookaheads(requested_lookaheads, max(time_steps - 1, 1))

    parents, parent_log_weights, nodes, node_log_weights = initial_rule(
        spec, theta, standard_nodes, standard_weights
    )
    total = tf.constant(0.0, DTYPE)
    increments = []
    minimum_weights = []
    condition_numbers = []
    feature_residuals = []
    valid_history = []
    realized_lookaheads = []
    incoming_weight_history = []
    for time_index in range(time_steps):
        incoming_weight_history.append(tf.exp(parent_log_weights))
        teacher = _teacher_step(
            spec,
            theta,
            parents,
            parent_log_weights,
            nodes,
            node_log_weights,
            target_observations[time_index],
            flow_observations[time_index],
            time_index,
        )
        total += teacher["increment"]
        increments.append(teacher["increment"])
        if time_index + 1 == time_steps:
            valid_history.append(tf.constant(True))
            continue
        future = target_observations[time_index + 1 :]
        realized = effective_progressive_lookaheads(
            requested_lookaheads, int(future.shape[0])
        )
        features = progressive_features(
            spec,
            theta,
            teacher["particles"],
            future,
            continuation_grid_points,
            continuation_grid_weights,
            first_future_time_index=time_index + 1,
            requested_lookaheads=requested_lookaheads,
        )
        if row_scales[time_index].shape != (3 + len(realized),):
            raise ValueError("prepared row scale does not match realized feature count")
        projection = tp._contract_e_tp_dense_kkt_forward_core(
            teacher["particles"],
            teacher["log_unnormalized_weights"],
            features,
            active_indices[time_index],
            row_scales[time_index],
            reference_weights[time_index],
            precisions[time_index],
        )
        parents = tf.reshape(projection["student_points"], [-1])
        parent_log_weights = tf.math.log(projection["student_weights"])
        minimum_weights.append(projection["minimum_weight"])
        condition_numbers.append(projection["condition_number"])
        feature_residuals.append(projection["feature_residual"])
        valid_history.append(projection["valid_chart"])
        realized_lookaheads.append(realized)
    return {
        "objective": total,
        "increment_history": tf.stack(increments),
        "minimum_weight_history": tf.stack(minimum_weights)
        if minimum_weights
        else tf.zeros([0], DTYPE),
        "condition_number_history": tf.stack(condition_numbers)
        if condition_numbers
        else tf.zeros([0], DTYPE),
        "feature_residual_history": tuple(feature_residuals),
        "valid_history": tf.stack(valid_history),
        "realized_lookaheads": tuple(realized_lookaheads),
        "final_particles": teacher["particles"],
        "final_log_unnormalized_weights": teacher["log_unnormalized_weights"],
        "incoming_weight_history": incoming_weight_history,
    }


def contract_e_tp_scalar_sv_recursive_core(
    spec: ScalarSVContractETPSpec,
    theta: tf.Tensor,
    target_observations: tf.Tensor,
    flow_observations: tf.Tensor,
    standard_nodes: tf.Tensor,
    standard_weights: tf.Tensor,
    active_indices: tf.Tensor,
    row_scales: tf.Tensor,
    continuation_grid_points: tf.Tensor,
    continuation_grid_weights: tf.Tensor,
    *,
    lookahead_steps: int,
) -> dict[str, tf.Tensor]:
    """Execute the fixed-chart scalar-SV Contract E--TP finite program."""

    if lookahead_steps < 1:
        raise ValueError("lookahead_steps must be positive")
    theta = tf.reshape(tf.convert_to_tensor(theta, DTYPE), [spec.parameter_dimension])
    target_observations = tf.reshape(
        tf.convert_to_tensor(target_observations, DTYPE), [-1, 1]
    )
    flow_observations = tf.reshape(
        tf.convert_to_tensor(flow_observations, DTYPE), [-1, 1]
    )
    time_steps = target_observations.shape[0]
    if time_steps is None or time_steps < 1:
        raise ValueError("scalar-SV Contract E--TP requires a static positive horizon")
    if active_indices.shape != (time_steps - 1, FEATURE_COUNT):
        raise ValueError(
            f"active_indices must have shape {(time_steps - 1, FEATURE_COUNT)}"
        )
    if row_scales.shape != (time_steps - 1, FEATURE_COUNT):
        raise ValueError(
            f"row_scales must have shape {(time_steps - 1, FEATURE_COUNT)}"
        )
    parents, parent_log_weights, nodes, node_log_weights = initial_rule(
        spec, theta, standard_nodes, standard_weights
    )
    total = tf.constant(0.0, DTYPE)
    increments = []
    minimum_weights = []
    condition_numbers = []
    feature_residuals = []
    valid_history = []
    incoming_weight_history = []
    for time_index in range(time_steps):
        incoming_weight_history.append(tf.exp(parent_log_weights))
        teacher = _teacher_step(
            spec,
            theta,
            parents,
            parent_log_weights,
            nodes,
            node_log_weights,
            target_observations[time_index],
            flow_observations[time_index],
            time_index,
        )
        total += teacher["increment"]
        increments.append(teacher["increment"])
        if time_index + 1 == time_steps:
            valid_history.append(tf.constant(True))
            continue
        stop = min(time_steps, time_index + 1 + lookahead_steps)
        features = _features(
            spec,
            theta,
            teacher["particles"],
            target_observations[time_index + 1 : stop],
            continuation_grid_points,
            continuation_grid_weights,
            first_future_time_index=time_index + 1,
        )
        projection = tp._contract_e_tp_dense_square_forward_core(
            teacher["particles"],
            teacher["log_unnormalized_weights"],
            features,
            active_indices[time_index],
            row_scales[time_index],
        )
        parents = tf.reshape(projection["student_points"], [-1])
        parent_log_weights = tf.math.log(projection["student_weights"])
        minimum_weights.append(projection["minimum_weight"])
        condition_numbers.append(projection["condition_number"])
        feature_residuals.append(projection["feature_residual"])
        valid_history.append(projection["valid_chart"])
    return {
        "objective": total,
        "increment_history": tf.stack(increments),
        "minimum_weight_history": tf.stack(minimum_weights)
        if minimum_weights
        else tf.zeros([0], DTYPE),
        "condition_number_history": tf.stack(condition_numbers)
        if condition_numbers
        else tf.zeros([0], DTYPE),
        "feature_residual_history": tf.stack(feature_residuals)
        if feature_residuals
        else tf.zeros([0, FEATURE_COUNT], DTYPE),
        "valid_history": tf.stack(valid_history),
        "final_particles": teacher["particles"],
        "final_log_unnormalized_weights": teacher["log_unnormalized_weights"],
        "incoming_weight_history": incoming_weight_history,
    }


def scalar_sv_dense_reference_value(
    spec: ScalarSVContractETPSpec,
    theta: tf.Tensor,
    target_observations: tf.Tensor,
    grid_points: tf.Tensor,
    grid_weights: tf.Tensor,
) -> dict[str, tf.Tensor]:
    """Return a deterministic fixed-grid same-target scalar filtering value."""

    theta = tf.reshape(tf.convert_to_tensor(theta, DTYPE), [spec.parameter_dimension])
    observations = tf.reshape(
        tf.convert_to_tensor(target_observations, DTYPE), [-1, 1]
    )
    grid = tf.reshape(tf.convert_to_tensor(grid_points, DTYPE), [-1, 1])
    log_weights = tf.math.log(
        tf.reshape(tf.convert_to_tensor(grid_weights, DTYPE), [-1])
    )
    log_posterior = None
    increments = []
    for time_index, observation in enumerate(tf.unstack(observations, axis=0)):
        if time_index == 0:
            if spec.transition_before_first_observation:
                previous_log_density = spec.model.initial_log_density(theta, grid)
                transition = _pairwise_transition_log_density(
                    spec, theta, grid, grid, time_index
                )
                log_predictive = tf.reduce_logsumexp(
                    log_weights[:, None]
                    + previous_log_density[:, None]
                    + transition,
                    axis=0,
                )
            else:
                log_predictive = spec.model.initial_log_density(theta, grid)
        else:
            transition = _pairwise_transition_log_density(
                spec, theta, grid, grid, time_index
            )
            log_predictive = tf.reduce_logsumexp(
                log_weights[:, None] + log_posterior[:, None] + transition,
                axis=0,
            )
        log_unnormalized = log_predictive + spec.model.observation_log_density(
            theta, grid, observation, t=time_index
        )
        increment = tf.reduce_logsumexp(log_weights + log_unnormalized)
        log_posterior = log_unnormalized - increment
        increments.append(increment)
    increment_history = tf.stack(increments)
    return {
        "objective": tf.reduce_sum(increment_history),
        "increment_history": increment_history,
        "final_log_posterior": log_posterior,
    }


__all__ = [
    "ALGORITHM_ID",
    "DTYPE",
    "FEATURE_COUNT",
    "FEATURE_NAMES",
    "ScalarSVContractETPSpec",
    "contract_e_tp_scalar_sv_loop_core",
    "contract_e_tp_actual_sv_overcomplete_loop_core",
    "contract_e_tp_actual_sv_overcomplete_manual_jvp_loop_core",
    "contract_e_tp_scalar_sv_recursive_core",
    "contract_e_tp_scalar_sv_recursive_kkt_core",
    "effective_progressive_lookaheads",
    "initial_rule",
    "make_scalar_sv_spec",
    "make_contract_e_tp_scalar_sv_loop_tf",
    "make_contract_e_tp_actual_sv_overcomplete_tf",
    "make_contract_e_tp_actual_sv_overcomplete_forward_ad_tf",
    "make_contract_e_tp_actual_sv_overcomplete_forward_tf",
    "make_contract_e_tp_actual_sv_overcomplete_manual_jvp_tf",
    "prepare_actual_sv_overcomplete_chart_step",
    "progressive_features",
    "scalar_sv_dense_reference_value",
    "target_and_flow_observations",
    "target_continuation_log_likelihood",
    "target_continuation_log_likelihood_loop",
]
