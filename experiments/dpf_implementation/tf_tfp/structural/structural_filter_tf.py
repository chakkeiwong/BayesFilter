"""Structural PF-PF filter with explicit stochastic/completion blocks."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from types import SimpleNamespace
from typing import Any

import tensorflow as tf

from experiments.dpf_implementation.tf_tfp.filters.bootstrap_pf_tf import (
    normalize_log_weights_tf,
    weighted_mean_and_variance_tf,
)
from experiments.dpf_implementation.tf_tfp.structural.contracts_tf import DTYPE, StructuralSSMModelTF
from experiments.dpf_implementation.tf_tfp.structural.particle_state_tf import StructuralParticleStateTF
from experiments.dpf_implementation.tf_tfp.structural.resampling_policies_tf import (
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


@lru_cache(maxsize=8)
def make_structural_filter_tf(observation_spec,z_spec,s_spec,callbacks,*,seed,
    resampling_policy_id,sinkhorn_epsilon=.45,sinkhorn_iterations=70,sinkhorn_tolerance=1e-7,jit_compile=True):
    """Bind static callback and tensor schemas once for repeated target calls."""
    from experiments.dpf_implementation.tf_tfp.filters.native_execution_tf import (
        compile_recomputed_diagnostic,scan_steps,tensor_diagnostics,
    )
    model = SimpleNamespace(**dict(zip(_CALLBACK_NAMES,callbacks)))
    horizon,num_particles = observation_spec.shape[0],z_spec.shape[0]
    if horizon is None or horizon < 1:
        raise ValueError("Structural filter requires a fixed positive observation horizon")
    labels = {}
    def evaluate(observations, initial_z, initial_s):
        log_weights = tf.fill([num_particles], -tf.math.log(tf.cast(num_particles, DTYPE)))
        def step(time_index, values):
            previous_z, previous_s, log_weights, log_likelihood = values
            observation = observations[time_index]
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
            ess = 1.0 / tf.reduce_sum(weights * weights)
            state = StructuralParticleStateTF(
                previous_z=previous_z,
                previous_s=previous_s,
                current_z=current_z,
                current_s=current_s,
            )
            full_state = state.completed_state()
            mean, variance = weighted_mean_and_variance_tf(full_state, weights)
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
            previous_z = policy_result.next_z
            previous_s = policy_result.next_s
            log_weights = policy_result.next_log_weights
            step_diag = {
                "time_index": time_index,
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
            return (previous_z, previous_s, log_weights, log_likelihood + increment), (
                mean, variance, ess, tensor_diagnostics(step_diag, labels))
        final, history = scan_steps(step, (initial_z, initial_s, log_weights, tf.constant(0., DTYPE)), horizon)
        return final[3], history

    call = compile_recomputed_diagnostic(evaluate, [observation_spec,z_spec,s_spec], jit_compile=jit_compile)
    call.report_labels = labels
    return call


_CALLBACK_NAMES = ("transition_z_sample","local_flow_proposal","complete_s",
    "transition_z_log_prob","observation_log_prob","completion_residual")


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
    jit_compile: bool = True,
) -> StructuralFilterTFResult:
    from experiments.dpf_implementation.tf_tfp.filters.native_execution_tf import (
        host_reports,
    )

    observations = tf.convert_to_tensor(observations, DTYPE)
    previous_z = tf.cast(model.initial_z_sample(num_particles, seed), DTYPE)
    previous_s = tf.cast(model.initial_s_from_z(previous_z), DTYPE)
    call = make_structural_filter_tf(tf.TensorSpec(observations.shape,DTYPE),
        tf.TensorSpec(previous_z.shape,DTYPE),tf.TensorSpec(previous_s.shape,DTYPE),
        tuple(getattr(model,name) for name in _CALLBACK_NAMES),seed=seed,
        resampling_policy_id=resampling_policy_id,sinkhorn_epsilon=sinkhorn_epsilon,
        sinkhorn_iterations=sinkhorn_iterations,sinkhorn_tolerance=sinkhorn_tolerance,jit_compile=jit_compile)
    labels = call.report_labels
    log_likelihood, history = call(observations, previous_z, previous_s)
    means, variances, ess_tensor, numeric_diagnostics = history
    diagnostics = host_reports(numeric_diagnostics, labels)
    for row in diagnostics:
        row.update(jit_compile=jit_compile, time_loop_route="tf_while_loop", canonical_algorithm_admitted=False)
        if not (row["finite_corrected_log_weights"] and row.get("valid", True) and row.get("valid_flow", True)):
            raise FloatingPointError("Structural filter violated a numerical or resampling veto")
    finite = bool(tf.reduce_all(tf.stack([tf.reduce_all(tf.math.is_finite(value))
        for value in (log_likelihood, means, variances, ess_tensor)])).numpy())
    return StructuralFilterTFResult(
        method_id=method_id or f"structural_ledh_pfpf_{resampling_policy_id}_tf",
        seed=int(seed), num_particles=int(num_particles), resampling_policy_id=resampling_policy_id,
        negative_log_likelihood=-log_likelihood, log_likelihood_estimate=log_likelihood,
        filtered_means=means, filtered_variances=variances, ess_by_time=ess_tensor,
        resampling_count=sum(int(row["resampled"]) for row in diagnostics), diagnostics=diagnostics, finite=finite)


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
    result = tf.reduce_all(tf.math.is_finite(tf.cast(value, DTYPE)))
    return bool(result.numpy()) if tf.executing_eagerly() else result


def _float(value: tf.Tensor) -> float:
    result = tf.cast(value, DTYPE)
    return float(result.numpy()) if tf.executing_eagerly() else result
