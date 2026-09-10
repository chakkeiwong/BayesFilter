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

    return NonlinearScoreModel(
        transition_mean_fn=transition_mean_fn,
        transition_mean_tangent_fn=transition_mean_tangent_fn,
        observation_fn=model.observation_fn,
        observation_jacobian_fn=model.observation_jacobian_fn,
        observation_tangent_fn=model.observation_tangent_fn,
        process_covariance=model.process_covariance,
        observation_covariance=model.observation_covariance,
        observation_log_density_fn=observation_log_density_fn,
        observation_log_density_tangent_fn=observation_log_density_tangent_fn,
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
    if k_count is None:
        raise ValueError("K dimension must be statically known")

    common_kwargs = dict(
        flow_substeps=substeps,
        with_score=True,
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

    def evaluate_row(inputs: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
        theta_row, row_directions = inputs

        def evaluate_direction(direction: Tensor) -> tuple[Tensor, Tensor]:
            bound_model = _single_cloud_model(model, theta_row, direction)
            value, score = canonical_value_and_analytical_score(
                bound_model,
                theta_row,
                initial_states,
                initial_covariances,
                noises,
                observations,
                **common_kwargs,
            )
            return value, score[0]

        values, scores = tf.map_fn(
            evaluate_direction,
            row_directions,
            fn_output_signature=(
                tf.TensorSpec([], dtype),
                tf.TensorSpec([], dtype),
            ),
            parallel_iterations=1,
        )
        tf.debugging.assert_near(
            values,
            tf.broadcast_to(values[0], tf.shape(values)),
            rtol=1.0e-12,
            atol=1.0e-12,
            message="direction changed the primal canonical value",
        )
        return values[0], scores

    values, scores = tf.map_fn(
        evaluate_row,
        (theta, directions),
        fn_output_signature=(
            tf.TensorSpec([], dtype),
            tf.TensorSpec([k_count], dtype),
        ),
        parallel_iterations=1,
    )

    valid = tf.math.is_finite(values) & tf.reduce_all(
        tf.math.is_finite(scores), axis=1
    )
    nan = tf.cast(float("nan"), dtype)
    values = tf.where(valid, values, nan)
    scores = tf.where(valid[:, None], scores, nan)
    if squeeze_output:
        scores = scores[:, 0]
    return values, scores, {"program_valid": valid}


__all__ = ["PerPointScoreModel", "canonical_batch_fused_value_score"]
