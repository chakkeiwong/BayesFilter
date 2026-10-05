"""TensorFlow LEDH-PF-PF with finite-Sinkhorn relaxed OT resampling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import tensorflow as tf

from experiments.dpf_implementation.tf_tfp.filters.bootstrap_pf_tf import (
    DTYPE,
    normalize_log_weights_tf,
    weighted_mean_and_variance_tf,
)
from experiments.dpf_implementation.tf_tfp.flows.ledh_tf import (
    LedhFlowBatchResult,
)
from experiments.dpf_implementation.tf_tfp.resampling.annealed_transport_tf import (
    annealed_transport_resample_tf,
)
from experiments.dpf_implementation.tf_tfp.resampling.sinkhorn_tf import (
    sinkhorn_resample_tf,
)


@dataclass(frozen=True)
class LedhPFPFOTTFResult:
    method_id: str
    seed: int
    num_particles: int
    log_likelihood_estimate: tf.Tensor
    filtered_means: tf.Tensor
    filtered_variances: tf.Tensor
    ess_by_time: tf.Tensor
    resampling_count: int
    resampling_diagnostics: list[dict[str, Any]]
    finite: bool


def run_ledh_pfpf_ot_tf(
    *,
    observations: tf.Tensor,
    initial_sample: Callable[[int, int], tf.Tensor],
    transition_sample: Callable[[tf.Tensor, int, int], tf.Tensor],
    ledh_flow: Callable[[tf.Tensor, tf.Tensor, tf.Tensor, int], LedhFlowBatchResult],
    transition_log_density: Callable[[tf.Tensor, tf.Tensor, int], tf.Tensor],
    observation_log_density: Callable[[tf.Tensor, tf.Tensor, int], tf.Tensor],
    seed: int,
    num_particles: int,
    ess_threshold_ratio: float = 0.98,
    sinkhorn_epsilon: float = 0.5,
    sinkhorn_iterations: int = 80,
    sinkhorn_tolerance: float = 1e-7,
    transport_method: str = "annealed_transport",
    annealed_scaling: float = 0.9,
    annealed_convergence_threshold: float = 1e-3,
    method_id: str = "ledh_pfpf_ot_annealed_transport_tf",
) -> LedhPFPFOTTFResult:
    particles = tf.cast(initial_sample(num_particles, seed), DTYPE)
    log_weights = tf.fill([num_particles], -tf.math.log(tf.cast(num_particles, DTYPE)))
    filtered_means = []
    filtered_variances = []
    ess_by_time = []
    diagnostics: list[dict[str, Any]] = []
    resampling_count = 0
    log_likelihood = tf.constant(0.0, dtype=DTYPE)

    for t, observation in enumerate(tf.unstack(tf.cast(observations, DTYPE), axis=0)):
        ancestors = tf.identity(particles)
        pre_flow = tf.cast(transition_sample(ancestors, seed, t), DTYPE)
        flow = ledh_flow(pre_flow, ancestors, observation, t)
        post_flow = flow.post_flow_particles
        target_transition = tf.cast(transition_log_density(post_flow, ancestors, t), DTYPE)
        target_observation = tf.cast(observation_log_density(post_flow, observation, t), DTYPE)
        corrected_log_weights = (
            log_weights
            + target_transition
            + target_observation
            - flow.pre_flow_log_density
            + flow.forward_log_det
        )
        weights, incremental = normalize_log_weights_tf(corrected_log_weights)
        log_likelihood = log_likelihood + incremental
        ess = 1.0 / tf.reduce_sum(weights * weights)
        mean, variance = weighted_mean_and_variance_tf(post_flow, weights)
        filtered_means.append(mean)
        filtered_variances.append(variance)
        ess_by_time.append(ess)
        step_diag: dict[str, Any] = {
            "time_index": int(t),
            "ess": _float(ess),
            "ess_ratio": _float(ess / tf.cast(num_particles, DTYPE)),
            "pfpf_correction": "log_target_transition_plus_observation_minus_q0_plus_forward_logdet",
            "finite_corrected_log_weights": _finite_bool(corrected_log_weights),
            "min_corrected_log_weight": _float(tf.reduce_min(corrected_log_weights)),
            "max_corrected_log_weight": _float(tf.reduce_max(corrected_log_weights)),
            "max_abs_corrected_log_weight": _float(tf.reduce_max(tf.abs(corrected_log_weights))),
            "finite_target_transition": _finite_bool(target_transition),
            "finite_target_observation": _finite_bool(target_observation),
            **flow.diagnostics,
        }
        if not step_diag["finite_corrected_log_weights"]:
            raise FloatingPointError("PF-PF corrected log weights are non-finite")
        do_resample = bool((ess < ess_threshold_ratio * num_particles).numpy())
        if do_resample:
            resampled, resampling_method = _resample_particles(
                particles=post_flow,
                weights=weights,
                log_weights=tf.math.log(tf.maximum(weights, tf.constant(1e-300, dtype=DTYPE))),
                transport_method=transport_method,
                epsilon=sinkhorn_epsilon,
                sinkhorn_iterations=sinkhorn_iterations,
                sinkhorn_tolerance=sinkhorn_tolerance,
                annealed_scaling=annealed_scaling,
                annealed_convergence_threshold=annealed_convergence_threshold,
            )
            particles = resampled.particles
            log_weights = tf.fill([num_particles], -tf.math.log(tf.cast(num_particles, DTYPE)))
            resampling_count += 1
            step_diag.update(
                {
                    "resampled": True,
                    "resampling_method": resampling_method,
                    **resampled.diagnostics,
                }
            )
        else:
            particles = post_flow
            log_weights = tf.math.log(tf.maximum(weights, tf.constant(1e-300, dtype=DTYPE)))
            step_diag.update(
                {
                    "resampled": False,
                    "resampling_method": "none",
                    "resampling_status": "no_resampling_ess_above_threshold",
                }
            )
        diagnostics.append(step_diag)

    filtered_means_tensor = tf.stack(filtered_means, axis=0)
    filtered_variances_tensor = tf.stack(filtered_variances, axis=0)
    ess_tensor = tf.stack(ess_by_time, axis=0)
    finite = bool(
        tf.math.is_finite(log_likelihood).numpy()
        and tf.reduce_all(tf.math.is_finite(filtered_means_tensor)).numpy()
        and tf.reduce_all(tf.math.is_finite(filtered_variances_tensor)).numpy()
        and tf.reduce_all(tf.math.is_finite(ess_tensor)).numpy()
    )
    return LedhPFPFOTTFResult(
        method_id=method_id,
        seed=int(seed),
        num_particles=int(num_particles),
        log_likelihood_estimate=log_likelihood,
        filtered_means=filtered_means_tensor,
        filtered_variances=filtered_variances_tensor,
        ess_by_time=ess_tensor,
        resampling_count=int(resampling_count),
        resampling_diagnostics=diagnostics,
        finite=finite,
    )


def _resample_particles(
    *,
    particles: tf.Tensor,
    weights: tf.Tensor,
    log_weights: tf.Tensor,
    transport_method: str,
    epsilon: float,
    sinkhorn_iterations: int,
    sinkhorn_tolerance: float,
    annealed_scaling: float,
    annealed_convergence_threshold: float,
):
    if transport_method == "annealed_transport":
        return (
            annealed_transport_resample_tf(
                particles,
                log_weights,
                epsilon=epsilon,
                scaling=annealed_scaling,
                convergence_threshold=annealed_convergence_threshold,
                max_iterations=sinkhorn_iterations,
            ),
            "filterflow_style_annealed_transport_tf",
        )
    if transport_method == "fixed_target_sinkhorn":
        return (
            sinkhorn_resample_tf(
                particles,
                weights,
                epsilon=epsilon,
                max_iterations=sinkhorn_iterations,
                tolerance=sinkhorn_tolerance,
            ),
            "fixed_target_sinkhorn_local_comparator_tf",
        )
    raise ValueError(f"unknown transport_method: {transport_method}")


def _finite_bool(value: tf.Tensor) -> bool:
    return bool(tf.reduce_all(tf.math.is_finite(tf.cast(value, DTYPE))).numpy())


def _float(value: tf.Tensor) -> float:
    return float(tf.cast(value, DTYPE).numpy())
