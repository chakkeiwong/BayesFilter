"""Resampling policies for experimental structural TF particle filters."""

from __future__ import annotations

import tensorflow as tf

from experiments.dpf_implementation.tf_tfp.resampling.sinkhorn_tf import sinkhorn_resample_tf
from experiments.dpf_implementation.tf_tfp.structural.contracts_tf import DTYPE, StructuralSSMModelTF
from experiments.dpf_implementation.tf_tfp.structural.particle_state_tf import (
    StructuralParticleStateTF,
    StructuralResamplingResultTF,
)


NO_RESAMPLING = "none"
CATEGORICAL_ANCESTOR = "categorical_ancestor"
SINKHORN_CURRENT_Z = "sinkhorn_current_z"
SINKHORN_FULL_CONTEXT = "sinkhorn_full_context"
SUPPORTED_POLICIES = (NO_RESAMPLING, CATEGORICAL_ANCESTOR, SINKHORN_CURRENT_Z, SINKHORN_FULL_CONTEXT)


def apply_structural_resampling_policy_tf(
    *,
    model: StructuralSSMModelTF,
    state: StructuralParticleStateTF,
    weights: tf.Tensor,
    policy_id: str,
    seed: int,
    time_index: int,
    sinkhorn_epsilon: float = 0.45,
    sinkhorn_iterations: int = 70,
    sinkhorn_tolerance: float = 1e-7,
) -> StructuralResamplingResultTF:
    weights = tf.reshape(tf.cast(weights, DTYPE), [-1])
    count = int(weights.shape[0])
    if policy_id not in SUPPORTED_POLICIES:
        raise ValueError(f"unsupported structural resampling policy: {policy_id}")
    if policy_id == NO_RESAMPLING:
        residual = model.completion_residual(
            state.previous_s,
            state.previous_z,
            state.current_z,
            state.current_s,
        )
        return StructuralResamplingResultTF(
            next_z=state.current_z,
            next_s=state.current_s,
            next_log_weights=tf.math.log(tf.maximum(weights, tf.constant(1e-300, DTYPE))),
            diagnostics={
                "resampling_policy_id": policy_id,
                "resampled": False,
                "context_semantics": "weighted_particles_carried_forward_without_resampling",
                "max_completion_residual_after_policy": _float(tf.reduce_max(tf.abs(residual))),
            },
        )
    if policy_id == CATEGORICAL_ANCESTOR:
        indices = tf.random.stateless_categorical(
            tf.reshape(tf.math.log(tf.maximum(weights, tf.constant(1e-300, DTYPE))), [1, -1]),
            count,
            seed=_seed_pair(seed, 8000 + time_index),
            dtype=tf.int32,
        )[0]
        previous_z = tf.gather(state.previous_z, indices)
        previous_s = tf.gather(state.previous_s, indices)
        current_z = tf.gather(state.current_z, indices)
        current_s = tf.gather(state.current_s, indices)
        residual = model.completion_residual(previous_s, previous_z, current_z, current_s)
        return StructuralResamplingResultTF(
            next_z=current_z,
            next_s=current_s,
            next_log_weights=_uniform_log_weights(count),
            diagnostics={
                "resampling_policy_id": policy_id,
                "resampled": True,
                "context_semantics": "categorical_ancestor_rows_gathered_without_relaxing_deterministic_block",
                "max_completion_residual_after_policy": _float(tf.reduce_max(tf.abs(residual))),
                "ancestor_index_min": int(tf.reduce_min(indices).numpy()),
                "ancestor_index_max": int(tf.reduce_max(indices).numpy()),
            },
        )
    if policy_id == SINKHORN_CURRENT_Z:
        resampled = sinkhorn_resample_tf(
            state.current_z,
            weights,
            epsilon=sinkhorn_epsilon,
            max_iterations=sinkhorn_iterations,
            tolerance=sinkhorn_tolerance,
        )
        relaxed_current_z = resampled.particles
        completed_s = model.complete_s(state.previous_s, state.previous_z, relaxed_current_z)
        residual = model.completion_residual(state.previous_s, state.previous_z, relaxed_current_z, completed_s)
        return StructuralResamplingResultTF(
            next_z=relaxed_current_z,
            next_s=completed_s,
            next_log_weights=_uniform_log_weights(count),
            diagnostics={
                "resampling_policy_id": policy_id,
                "resampled": True,
                "context_semantics": "finite_sinkhorn_relaxes_current_stochastic_block_only_destination_context_preserved",
                "max_completion_residual_after_policy": _float(tf.reduce_max(tf.abs(residual))),
                **_sinkhorn_diagnostics(resampled.diagnostics),
            },
        )
    structural_context = tf.concat([state.previous_z, state.previous_s, state.current_z], axis=1)
    resampled = sinkhorn_resample_tf(
        structural_context,
        weights,
        epsilon=sinkhorn_epsilon,
        max_iterations=sinkhorn_iterations,
        tolerance=sinkhorn_tolerance,
    )
    z_dim = int(state.previous_z.shape[1])
    s_dim = int(state.previous_s.shape[1])
    relaxed_previous_z = resampled.particles[:, :z_dim]
    relaxed_previous_s = resampled.particles[:, z_dim : z_dim + s_dim]
    relaxed_current_z = resampled.particles[:, z_dim + s_dim :]
    completed_s = model.complete_s(relaxed_previous_s, relaxed_previous_z, relaxed_current_z)
    residual = model.completion_residual(relaxed_previous_s, relaxed_previous_z, relaxed_current_z, completed_s)
    return StructuralResamplingResultTF(
        next_z=relaxed_current_z,
        next_s=completed_s,
        next_log_weights=_uniform_log_weights(count),
        diagnostics={
            "resampling_policy_id": policy_id,
            "resampled": True,
            "context_semantics": "old_ad_hoc_comparator_finite_sinkhorn_relaxes_previous_context_and_current_z",
            "max_completion_residual_after_policy": _float(tf.reduce_max(tf.abs(residual))),
            **_sinkhorn_diagnostics(resampled.diagnostics),
        },
    )


def _sinkhorn_diagnostics(diagnostics: dict) -> dict:
    return {
        "resampling_method": diagnostics["component_id"],
        "sinkhorn_epsilon": diagnostics["epsilon"],
        "sinkhorn_iterations_used": diagnostics["iterations_used"],
        "sinkhorn_tolerance": diagnostics["tolerance"],
        "max_row_residual": diagnostics["max_row_residual"],
        "max_column_residual": diagnostics["max_column_residual"],
        "total_mass_residual": diagnostics["total_mass_residual"],
        "min_coupling": diagnostics["min_coupling"],
        "finite_coupling": diagnostics["finite_coupling"],
        "finite_particles": diagnostics["finite_particles"],
    }


def _uniform_log_weights(count: int) -> tf.Tensor:
    return tf.fill([count], -tf.math.log(tf.cast(count, DTYPE)))


def _seed_pair(seed: int, salt: int) -> tf.Tensor:
    return tf.constant([int(seed) % 2147483647, int(salt) % 2147483647], dtype=tf.int32)


def _float(value: tf.Tensor) -> float:
    return float(tf.cast(value, DTYPE).numpy())
