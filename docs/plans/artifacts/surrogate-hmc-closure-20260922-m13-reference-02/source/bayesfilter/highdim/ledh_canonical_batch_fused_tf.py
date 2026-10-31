"""Batch-native adapter over the canonical LEDH analytical score engine.

The algorithmic finite program lives only in ``ledh_canonical_score_tf``.
This module adapts per-point callbacks and explicit ``[B,K,P]`` directions to
that authority with TensorFlow ``map_fn`` control flow. There is no Python row
or direction loop in the traced graph, no pfor/vectorized-map, and no autodiff.

Primal cloud reductions therefore inherit the canonical single-cloud semantics;
the explicit B and K axes are retained by the adapter. Contract-E reset,
higher-moment correction, trust/cap controls, and annealed execution all pass
through unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import tensorflow as tf

from bayesfilter.highdim.ledh_canonical_score_tf import (
    NonlinearScoreModel,
    canonical_value_and_analytical_score,
)

from bayesfilter.highdim.ledh_failure_status_tf import (
    initial_status, record_failure, FailureCode, FailureStage,
)


def _map_evaluations(fn, elements, signature, *, use_whileloop=False):
    """Two loop implementations, one evaluator and one numerical contract."""
    if not use_whileloop:
        return tf.map_fn(fn, elements, fn_output_signature=signature, parallel_iterations=1)
    count = tf.shape(tf.nest.flatten(elements)[0])[0]
    outputs = tf.nest.map_structure(
        lambda spec: tf.zeros(tf.concat([[count], tf.constant(spec.shape.as_list(),tf.int32)],0), spec.dtype),
        signature)
    def body(index, outputs):
        row = tf.nest.map_structure(lambda value: value[index],elements)
        result = fn(row)
        outputs = tf.nest.map_structure(
            lambda target,value: tf.tensor_scatter_nd_update(target,tf.reshape(index,[1,1]),value[None]),
            outputs,result)
        return index+1,outputs
    return tf.while_loop(lambda index,*_: index<count,body,(tf.constant(0),outputs),
                         parallel_iterations=1)[1]


Tensor = tf.Tensor


@dataclass(frozen=True)
class PerPointScoreModel:
    """Model callbacks aligned with flattened per-particle theta rows."""

    transition_mean_fn: Callable[[Tensor, Tensor], Tensor]
    transition_mean_tangent_fn: Callable[[Tensor, Tensor, Tensor, Tensor], Tensor]
    observation_fn: Callable[[Tensor], Tensor]
    observation_jacobian_fn: Callable[[Tensor], Tensor]
    observation_tangent_fn: Callable[[Tensor, Tensor], Tensor]
    process_covariance: Tensor
    observation_covariance: Tensor
    observation_log_density_fn: Callable[[Tensor, Tensor, Tensor], Tensor] | None = None
    observation_log_density_tangent_fn: (
        Callable[[Tensor, Tensor, Tensor, Tensor, Tensor], Tensor] | None
    ) = None


    # Optional parameter-dependent callbacks. Values take theta [P]; covariance
    # tangents also take direction [P]. Observation callbacks receive rows
    # aligned with particles. Missing tangent partners are configuration errors.
    process_covariance_fn: Callable | None = None
    process_covariance_tangent_fn: Callable | None = None
    observation_covariance_fn: Callable | None = None
    observation_covariance_tangent_fn: Callable | None = None
    parameter_observation_fn: Callable | None = None
    parameter_observation_tangent_fn: Callable | None = None
    parameter_observation_jacobian_fn: Callable | None = None
    parameter_observation_jacobian_tangent_fn: Callable | None = None
    observation_jacobian_tangent_fn: Callable | None = None
    initial_distribution_fn: Callable | None = None
    initial_distribution_tangent_fn: Callable | None = None
    # Additive structural transition x=f(ancestor)+B e. If supplied, the
    # process covariance and frozen noises are r-dimensional innovations.
    innovation_matrix: Tensor | None = None
    innovation_matrix_fn: Callable | None = None
    innovation_matrix_tangent_fn: Callable | None = None

    def __post_init__(self):
        for value_name,tangent_name in (
            ('process_covariance_fn','process_covariance_tangent_fn'),
            ('observation_covariance_fn','observation_covariance_tangent_fn'),
            ('parameter_observation_fn','parameter_observation_tangent_fn'),
            ('parameter_observation_jacobian_fn','parameter_observation_jacobian_tangent_fn'),
            ('initial_distribution_fn','initial_distribution_tangent_fn'),
            ('innovation_matrix_fn','innovation_matrix_tangent_fn')):
            if (getattr(self,value_name) is None) != (getattr(self,tangent_name) is None):
                raise ValueError(f'{value_name} and {tangent_name} must be supplied together')
        if self.innovation_matrix is not None and self.innovation_matrix_fn is not None:
            raise ValueError('supply constant or parameter-dependent innovation_matrix, not both')


def _single_cloud_model(
    model: PerPointScoreModel,
    theta: Tensor,
    direction: Tensor,
) -> NonlinearScoreModel:
    """Bind one theta row and direction to the single-cloud callback contract."""

    def theta_rows(points: Tensor) -> Tensor:
        return tf.broadcast_to(theta[None, :], [tf.shape(points)[0], tf.shape(theta)[0]])

    def direction_rows(points: Tensor) -> Tensor:
        return tf.broadcast_to(
            direction[None, :], [tf.shape(points)[0], tf.shape(direction)[0]]
        )

    def transition_mean_fn(_theta: Tensor, points: Tensor) -> Tensor:
        return model.transition_mean_fn(theta_rows(points), points)

    def transition_mean_tangent_fn(
        _theta: Tensor, points: Tensor, d_points: Tensor
    ) -> Tensor:
        return model.transition_mean_tangent_fn(
            theta_rows(points), points, d_points, direction_rows(points)
        )

    observation_log_density_fn = None
    observation_log_density_tangent_fn = None
    if model.observation_log_density_fn is not None:

        def observation_log_density_fn(
            _theta: Tensor, points: Tensor, observation: Tensor
        ) -> Tensor:
            return model.observation_log_density_fn(
                theta_rows(points), points, observation
            )

        if model.observation_log_density_tangent_fn is None:
            raise ValueError(
                "observation_log_density_tangent_fn is required when "
                "observation_log_density_fn is configured"
            )

        def observation_log_density_tangent_fn(
            _theta: Tensor,
            points: Tensor,
            observation: Tensor,
            d_points: Tensor,
        ) -> Tensor:
            return model.observation_log_density_tangent_fn(
                theta_rows(points),
                points,
                observation,
                d_points,
                direction_rows(points),
            )

    q = model.process_covariance if model.process_covariance_fn is None else model.process_covariance_fn(theta)
    r = model.observation_covariance if model.observation_covariance_fn is None else model.observation_covariance_fn(theta)
    observation_fn = model.observation_fn if model.parameter_observation_fn is None else (
        lambda points: model.parameter_observation_fn(theta_rows(points),points))
    observation_tangent_fn = model.observation_tangent_fn if model.parameter_observation_tangent_fn is None else (
        lambda points,dx: model.parameter_observation_tangent_fn(theta_rows(points),points,dx,direction_rows(points)))
    jacobian_fn = model.observation_jacobian_fn if model.parameter_observation_jacobian_fn is None else (
        lambda points: model.parameter_observation_jacobian_fn(theta_rows(points),points))
    jacobian_tangent_fn = model.observation_jacobian_tangent_fn if model.parameter_observation_jacobian_tangent_fn is None else (
        lambda points,dx: model.parameter_observation_jacobian_tangent_fn(theta_rows(points),points,dx,direction_rows(points)))
    return NonlinearScoreModel(
        transition_mean_fn=transition_mean_fn,
        transition_mean_tangent_fn=transition_mean_tangent_fn,
        observation_fn=observation_fn,
        observation_jacobian_fn=jacobian_fn,
        observation_tangent_fn=observation_tangent_fn,
        observation_jacobian_tangent_fn=jacobian_tangent_fn,
        process_covariance=q,
        observation_covariance=r,
        process_covariance_tangent_fn=None if model.process_covariance_tangent_fn is None else (
            lambda _theta: model.process_covariance_tangent_fn(theta,direction)),
        observation_covariance_tangent_fn=None if model.observation_covariance_tangent_fn is None else (
            lambda _theta: model.observation_covariance_tangent_fn(theta,direction)),
        observation_log_density_fn=observation_log_density_fn,
        observation_log_density_tangent_fn=observation_log_density_tangent_fn,
        innovation_matrix=model.innovation_matrix if model.innovation_matrix_fn is None else model.innovation_matrix_fn(theta),
        innovation_matrix_tangent_fn=None if model.innovation_matrix_tangent_fn is None else (
            lambda _theta: model.innovation_matrix_tangent_fn(theta, direction)),
    )


def canonical_batch_fused_value_score(
    model: PerPointScoreModel,
    theta: Tensor,
    theta_directions: Tensor,
    initial_states: Tensor,
    initial_covariances: Tensor,
    noises: Tensor,
    observations: Tensor,
    *,
    substeps: int,
    jitter: float = 1.0e-12,
    reset_policy: str = "none",
    reset_design: Tensor | None = None,
    reset_epsilon: float = 2.0,
    reset_sinkhorn_steps: int = 8,
    reset_balance_steps: int = 8,
    reset_ridge: float = 1.0e-5,
    correction_steps: int = 0,
    correction_strength: float = 0.2,
    correction_lm_damping: float = 1.0e-2,
    correction_lm_scale_floor: float = 1.0e-4,
    correction_trust_radius: float = 0.5,
    pairwise_steps: int = 0,
    pairwise_strength: float = 0.02,
    pairwise_rms_cap: float = 2.0,
    coordinate_cap: float = 0.0,
    coordinate_cap_power: int = 8,
    annealed_stages: int = 1,
    annealed_seed: int = 0,
    k_batch_mode: str = "sequential",
    _use_whileloop: bool = False,
    return_filter_moments: bool = False,
) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
    """Evaluate value ``[B]`` and analytical directional scores ``[B,K]``.

    Rank-2 directions ``[B,P]`` are promoted to K=1 and the returned score is
    squeezed to ``[B]`` for backward compatibility. Frozen particles, noises,
    observations, and reset design are shared across rows.

    ``jitter`` remains in the public signature. The shared canonical engine's
    governed jitter is 1e-12, so non-default values fail loudly rather than
    silently selecting a second numerical program.
    """

    if jitter != 1.0e-12:
        raise ValueError("the unified canonical engine requires jitter=1e-12")
    if k_batch_mode not in ("sequential", "pfor"):
        raise ValueError("k_batch_mode must be 'sequential' or 'pfor'")
    if reset_policy not in ("none", "contract_e"):
        raise ValueError("reset_policy must be 'none' or 'contract_e'")
    if reset_policy == "contract_e" and reset_design is None:
        raise ValueError("reset_design is required for reset_policy='contract_e'")
    if annealed_stages < 1:
        raise ValueError("annealed_stages must be positive")

    initial_states = tf.convert_to_tensor(initial_states)
    dtype = initial_states.dtype
    theta = tf.cast(tf.convert_to_tensor(theta), dtype)
    directions_input = tf.cast(tf.convert_to_tensor(theta_directions), dtype)
    initial_covariances = tf.cast(initial_covariances, dtype)
    noises = tf.cast(noises, dtype)
    observations = tf.cast(observations, dtype)
    if reset_design is not None:
        reset_design = tf.cast(reset_design, dtype)

    if directions_input.shape.rank == 2:
        directions = directions_input[:, None, :]
        squeeze_output = True
    elif directions_input.shape.rank == 3:
        directions = directions_input
        squeeze_output = False
    else:
        raise ValueError("theta_directions must have rank 2 or 3")

    if theta.shape.rank != 2:
        raise ValueError("theta must have rank 2")
    if theta.shape[0] is not None and directions.shape[0] is not None:
        if theta.shape[0] != directions.shape[0]:
            raise ValueError("theta and theta_directions batch dimensions must agree")
    if theta.shape[1] is not None and directions.shape[2] is not None:
        if theta.shape[1] != directions.shape[2]:
            raise ValueError("theta and theta_directions parameter dimensions must agree")

    k_count = directions.shape[1]
    if k_count is None or k_count < 1:
        raise ValueError("K dimension must be statically known and positive")
    if theta.shape[0] == 0:
        raise ValueError("theta must contain at least one row")

    common_kwargs = dict(
        flow_substeps=substeps,
        with_score=True,
        return_status=True,
        return_trace=return_filter_moments,
        reset_policy=reset_policy,
        reset_design=reset_design,
        reset_epsilon=reset_epsilon,
        reset_sinkhorn_steps=reset_sinkhorn_steps,
        reset_balance_steps=reset_balance_steps,
        reset_ridge=reset_ridge,
        correction_steps=correction_steps,
        correction_strength=correction_strength,
        correction_lm_damping=correction_lm_damping,
        correction_lm_scale_floor=correction_lm_scale_floor,
        correction_trust_radius=correction_trust_radius,
        pairwise_steps=pairwise_steps,
        pairwise_strength=pairwise_strength,
        pairwise_rms_cap=pairwise_rms_cap,
        coordinate_cap=coordinate_cap,
        coordinate_cap_power=coordinate_cap_power,
        annealed_stages=annealed_stages,
        annealed_seed=annealed_seed,
    )

    status_signature = tf.nest.map_structure(lambda value: tf.TensorSpec([],value.dtype), initial_status())
    if return_filter_moments:
        horizon, dimension = int(observations.shape[0]), int(initial_states.shape[1])
        status_signature.update(
            filtering_mean=tf.TensorSpec([horizon,dimension],dtype),
            filtering_covariance=tf.TensorSpec([horizon,dimension,dimension],dtype),
            filtering_ess=tf.TensorSpec([horizon],dtype),
            post_reset_mean=tf.TensorSpec([horizon,dimension],dtype),
            post_reset_covariance=tf.TensorSpec([horizon,dimension,dimension],dtype))
        for key in ("higher_moment_maximum_coordinatewise_pre_cap_absolute",
                    "higher_moment_maximum_coordinatewise_post_cap_absolute",
                    "higher_moment_mean_coordinatewise_cap_displacement",
                    "higher_moment_fraction_coordinatewise_cap_active",
                    "higher_moment_minimum_coordinatewise_cap_derivative"):
            status_signature[key]=tf.TensorSpec([horizon],dtype)

    def evaluate_row(inputs):
        theta_row, row_directions = inputs

        def evaluate_direction(direction):
            states,covariances = initial_states,initial_covariances
            d_states,d_covariances = None,None
            if model.initial_distribution_fn is not None:
                from bayesfilter.highdim.ledh_numerical_safety_tf import safe_cholesky,cholesky_differential
                mean,covariance=model.initial_distribution_fn(theta_row)
                d_mean,d_covariance=model.initial_distribution_tangent_fn(theta_row,direction)
                valid,chol=safe_cholesky(covariance,"initial_distribution")
                d_chol=cholesky_differential(chol,d_covariance,valid)
                # initial_states are fixed standard-normal base draws in this
                # explicit mode. P0 is separately the local UKF design covariance,
                # following canonical_filter_tf's initialization contract.
                states=mean[None]+tf.linalg.matmul(initial_states,chol,transpose_b=True)
                d_states=d_mean[None]+tf.linalg.matmul(initial_states,d_chol,transpose_b=True)
                covariances=tf.broadcast_to(covariance,tf.shape(initial_covariances))
                d_covariances=tf.broadcast_to(d_covariance,tf.shape(initial_covariances))
            result = canonical_value_and_analytical_score(
                _single_cloud_model(model, theta_row, direction), theta_row,
                states, covariances, noises, observations,
                initial_state_tangent=d_states,initial_covariance_tangent=d_covariances,
                **common_kwargs)
            if return_filter_moments:
                value, score, trace, status = result
                means, covs, esses, reset_means, reset_covs = [], [], [], [], []
                for step in trace:
                    cloud, weights = step['children'], step['posterior_weights']
                    mean = tf.reduce_sum(weights[:,None]*cloud,axis=0)
                    centered = cloud-mean
                    covariance = tf.einsum('n,ni,nj->ij',weights,centered,centered)
                    means.append(mean); covs.append(covariance)
                    esses.append(1/tf.reduce_sum(tf.square(weights)))
                    reset_cloud=step['states_after_reset']
                    reset_mean=tf.reduce_mean(reset_cloud,axis=0)
                    reset_centered=reset_cloud-reset_mean
                    reset_means.append(reset_mean)
                    reset_covs.append(tf.einsum('ni,nj->ij',reset_centered,reset_centered)/tf.cast(tf.shape(reset_cloud)[0],dtype))
                status.update(filtering_mean=tf.stack(means),
                              filtering_covariance=tf.stack(covs),filtering_ess=tf.stack(esses),
                              post_reset_mean=tf.stack(reset_means),post_reset_covariance=tf.stack(reset_covs))
                for key in ("higher_moment_maximum_coordinatewise_pre_cap_absolute",
                            "higher_moment_maximum_coordinatewise_post_cap_absolute",
                            "higher_moment_mean_coordinatewise_cap_displacement",
                            "higher_moment_fraction_coordinatewise_cap_active",
                            "higher_moment_minimum_coordinatewise_cap_derivative"):
                    status[key]=tf.stack([step[key] for step in trace])
            else:
                value, score, status = result
            return value, score[0], status

        signature = (tf.TensorSpec([],dtype),tf.TensorSpec([],dtype),status_signature)
        if k_batch_mode == 'pfor':
            values,scores,direction_status = tf.vectorized_map(evaluate_direction,row_directions)
        else:
            values,scores,direction_status = _map_evaluations(
                evaluate_direction,row_directions,signature,use_whileloop=_use_whileloop)
        value_valid = direction_status['value_valid'] & tf.math.is_finite(values)
        force_valid = direction_status['force_valid'] & tf.math.is_finite(scores) & value_valid
        # Never compare infinities arithmetically. The predicate also executes
        # under XLA, where TensorFlow debugging assertions may be elided.
        finite_values = tf.where(value_valid,values,tf.zeros_like(values))
        consistent = tf.reduce_all(value_valid == value_valid[0])
        consistent &= tf.reduce_all(tf.abs(finite_values-finite_values[0]) <=
            tf.constant(1e-12,dtype)*(1+tf.abs(finite_values[0])))
        failed = ~(value_valid & force_valid)
        first = tf.argmax(tf.cast(failed,tf.int32),output_type=tf.int32)
        status = {key:value[first] for key,value in direction_status.items()}
        status['value_valid'] = tf.reduce_all(value_valid)
        status['force_valid'] = tf.reduce_all(force_valid)
        value_first = tf.argmax(tf.cast(~value_valid,tf.int32),output_type=tf.int32)
        status['value_failure_code'] = direction_status['value_failure_code'][value_first]
        status = record_failure(status, consistent, consistent,
            time=-1, stage=FailureStage.DIRECTIONS, value_code=FailureCode.DIRECTION_MISMATCH)
        status['direction_value_valid'] = value_valid
        status['direction_force_valid'] = force_valid
        status['direction_force_failure_code'] = direction_status['force_failure_code']
        value = tf.where(status['value_valid'],values[0],tf.constant(float('-inf'),dtype))
        # Individual directional evidence remains available, but the force
        # supplied to an integrator is a single, deterministic all-or-zero field.
        score = tf.where(status['force_valid'],scores,tf.zeros_like(scores))
        return value,score,status

    row_signature = dict(status_signature)
    row_signature.update(direction_value_valid=tf.TensorSpec([k_count],tf.bool),
                         direction_force_valid=tf.TensorSpec([k_count],tf.bool),
                         direction_force_failure_code=tf.TensorSpec([k_count],tf.int32))
    values,scores,diagnostics = _map_evaluations(
        evaluate_row,(theta,directions),
        (tf.TensorSpec([],dtype),tf.TensorSpec([k_count],dtype),row_signature),
        use_whileloop=_use_whileloop)
    diagnostics['program_valid'] = diagnostics['value_valid'] & diagnostics['force_valid']
    if squeeze_output:
        scores = scores[:,0]
    return values,scores,diagnostics


def canonical_batch_fused_value_score_whileloop(*args, **kwargs):
    """Dense while-loop evaluator with the same settings and authority as map mode.

    In particular, this entry point has no separate reset family or correction
    defaults. Both entry points execute the same scalar canonical calculation
    and expose identical value, force and per-direction status.
    """
    kwargs['_use_whileloop'] = True
    return canonical_batch_fused_value_score(*args, **kwargs)
