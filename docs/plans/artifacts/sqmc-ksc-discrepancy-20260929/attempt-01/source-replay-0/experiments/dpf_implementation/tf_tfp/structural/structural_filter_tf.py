"""Structural PF-PF filter with explicit stochastic/completion blocks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import tensorflow as tf

from experiments.dpf_implementation.tf_tfp.filters.bootstrap_pf_tf import (
    normalize_log_weights_tf,
    weighted_mean_and_variance_tf,
)
from experiments.dpf_implementation.tf_tfp.structural.contracts_tf import DTYPE, StructuralSSMModelTF
from experiments.dpf_implementation.tf_tfp.structural.particle_state_tf import StructuralParticleStateTF
from experiments.dpf_implementation.tf_tfp.structural.resampling_policies_tf import (
    NO_RESAMPLING,
    apply_structural_resampling_policy_tf,
)


@dataclass(frozen=True)
class StructuralFilterTFResult:
    method_id: str
    seed: int
    num_particles: int
    resampling_policy_id: str
    negative_log_likelihood: tf.Tensor
    log_likelihood_estimate: tf.Tensor
    filtered_means: tf.Tensor
    filtered_variances: tf.Tensor
    ess_by_time: tf.Tensor
    resampling_count: int
    diagnostics: list[dict[str, Any]]
    finite: bool


def run_structural_ledh_pfpf_tf(
    *,
    model: StructuralSSMModelTF,
    observations: tf.Tensor,
    seed: int,
    num_particles: int,
    resampling_policy_id: str,
    sinkhorn_epsilon: float = 0.45,
    sinkhorn_iterations: int = 70,
    sinkhorn_tolerance: float = 1e-7,
    method_id: str | None = None,
) -> StructuralFilterTFResult:
    previous_z = tf.cast(model.initial_z_sample(num_particles, seed), DTYPE)
    previous_s = tf.cast(model.initial_s_from_z(previous_z), DTYPE)
    log_weights = tf.fill([num_particles], -tf.math.log(tf.cast(num_particles, DTYPE)))
    log_likelihood = tf.constant(0.0, DTYPE)
    filtered_means = []
    filtered_variances = []
    ess_values = []
    diagnostics: list[dict[str, Any]] = []
    resampling_count = 0

    for time_index, observation in enumerate(tf.unstack(tf.cast(observations, DTYPE), axis=0)):
        pre_z = model.transition_z_sample(previous_z, seed, time_index)
        flow = model.local_flow_proposal(pre_z, previous_z, previous_s, observation, time_index)
        current_z = flow.post_z
        current_s = model.complete_s(previous_s, previous_z, current_z)
        target_transition = model.transition_z_log_prob(current_z, previous_z, time_index)
        target_observation = model.observation_log_prob(current_z, current_s, observation, time_index)
        corrected_log_weights = (
            log_weights
            + target_transition
            + target_observation
            - flow.pre_flow_log_density
            + flow.forward_log_det
        )
        weights, increment = normalize_log_weights_tf(corrected_log_weights)
        log_likelihood = log_likelihood + increment
        ess = 1.0 / tf.reduce_sum(weights * weights)
        state = StructuralParticleStateTF(
            previous_z=previous_z,
            previous_s=previous_s,
            current_z=current_z,
            current_s=current_s,
        )
        full_state = state.completed_state()
        mean, variance = weighted_mean_and_variance_tf(full_state, weights)
        filtered_means.append(mean)
        filtered_variances.append(variance)
        ess_values.append(ess)
        completion_residual = model.completion_residual(previous_s, previous_z, current_z, current_s)
        policy_result = apply_structural_resampling_policy_tf(
            model=model,
            state=state,
            weights=weights,
            policy_id=resampling_policy_id,
            seed=seed,
            time_index=time_index,
            sinkhorn_epsilon=sinkhorn_epsilon,
            sinkhorn_iterations=sinkhorn_iterations,
            sinkhorn_tolerance=sinkhorn_tolerance,
        )
        if policy_result.diagnostics["resampled"]:
            resampling_count += 1
        previous_z = policy_result.next_z
        previous_s = policy_result.next_s
        log_weights = policy_result.next_log_weights
        step_diag = {
            "time_index": int(time_index),
            "ess": _float(ess),
            "ess_ratio": _float(ess / tf.cast(num_particles, DTYPE)),
            "finite_corrected_log_weights": _finite_bool(corrected_log_weights),
            "min_corrected_log_weight": _float(tf.reduce_min(corrected_log_weights)),
            "max_corrected_log_weight": _float(tf.reduce_max(corrected_log_weights)),
            "max_abs_corrected_log_weight": _float(tf.reduce_max(tf.abs(corrected_log_weights))),
            "finite_target_transition_z": _finite_bool(target_transition),
            "finite_observation_log_prob_completed_state": _finite_bool(target_observation),
            "max_completion_residual_before_policy": _float(tf.reduce_max(tf.abs(completion_residual))),
            "density_contract": "transition_and_proposal_density_on_stochastic_z_only_no_density_on_deterministic_s",
            "flow_logdet_contract": "forward_log_det_on_stochastic_z_block_only",
            **flow.diagnostics,
            **policy_result.diagnostics,
        }
        if not step_diag["finite_corrected_log_weights"]:
            raise FloatingPointError("structural corrected log weights are non-finite")
        diagnostics.append(step_diag)

    means = tf.stack(filtered_means, axis=0)
    variances = tf.stack(filtered_variances, axis=0)
    ess_tensor = tf.stack(ess_values, axis=0)
    negative_log_likelihood = -log_likelihood
    finite = bool(
        tf.math.is_finite(negative_log_likelihood).numpy()
        and tf.reduce_all(tf.math.is_finite(means)).numpy()
        and tf.reduce_all(tf.math.is_finite(variances)).numpy()
        and tf.reduce_all(tf.math.is_finite(ess_tensor)).numpy()
    )
    return StructuralFilterTFResult(
        method_id=method_id or f"structural_ledh_pfpf_{resampling_policy_id}_tf",
        seed=int(seed),
        num_particles=int(num_particles),
        resampling_policy_id=resampling_policy_id,
        negative_log_likelihood=negative_log_likelihood,
        log_likelihood_estimate=log_likelihood,
        filtered_means=means,
        filtered_variances=variances,
        ess_by_time=ess_tensor,
        resampling_count=int(resampling_count),
        diagnostics=diagnostics,
        finite=finite,
    )


def summarize_structural_diagnostics(diagnostics: list[dict[str, Any]]) -> dict[str, float | int | bool]:
    sinkhorn_residuals = []
    max_completion = 0.0
    finite_weights = True
    min_ess = float("inf")
    for diag in diagnostics:
        max_completion = max(
            max_completion,
            float(diag.get("max_completion_residual_before_policy", 0.0)),
            float(diag.get("max_completion_residual_after_policy", 0.0)),
        )
        finite_weights = finite_weights and bool(diag.get("finite_corrected_log_weights", False))
        min_ess = min(min_ess, float(diag.get("ess", 0.0)))
        if "max_row_residual" in diag:
            sinkhorn_residuals.append(float(diag["max_row_residual"]))
            sinkhorn_residuals.append(float(diag["max_column_residual"]))
            sinkhorn_residuals.append(float(diag["total_mass_residual"]))
    return {
        "max_completion_residual": max_completion,
        "max_sinkhorn_residual": max(sinkhorn_residuals) if sinkhorn_residuals else 0.0,
        "finite_corrected_log_weights": finite_weights,
        "min_ess": min_ess,
        "resampling_steps": sum(1 for diag in diagnostics if diag.get("resampled")),
    }


def _finite_bool(value: tf.Tensor) -> bool:
    return bool(tf.reduce_all(tf.math.is_finite(tf.cast(value, DTYPE))).numpy())


def _float(value: tf.Tensor) -> float:
    return float(tf.cast(value, DTYPE).numpy())
