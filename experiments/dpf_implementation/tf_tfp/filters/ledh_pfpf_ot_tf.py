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
    jit_compile: bool = True,
) -> LedhPFPFOTTFResult:
    from experiments.dpf_implementation.tf_tfp.filters.particle_ot_execution_tf import (
        make_particle_ot_filter, particle_ot_reports,
    )

    observations = tf.convert_to_tensor(observations, DTYPE)
    particles = tf.cast(initial_sample(num_particles, seed), DTYPE)
    call = make_particle_ot_filter(tf.TensorSpec(observations.shape, DTYPE),
        tf.TensorSpec(particles.shape, DTYPE), transition_sample=transition_sample,
        observation_log_density=observation_log_density, seed=seed,
        ess_threshold_ratio=ess_threshold_ratio, jit_compile=jit_compile,
        transport_method=transport_method, epsilon=sinkhorn_epsilon,
        sinkhorn_iterations=sinkhorn_iterations, sinkhorn_tolerance=sinkhorn_tolerance,
        annealed_scaling=annealed_scaling, annealed_convergence_threshold=annealed_convergence_threshold,
        retained_teacher_warmstart_fn=None, ledh_flow=ledh_flow, transition_log_density=transition_log_density)
    total, history = call(observations, particles)
    means, variances, ess, _, _ = history
    diagnostics = particle_ot_reports(history, call, jit_compile=jit_compile)
    finite = bool(tf.reduce_all(tf.stack([tf.reduce_all(tf.math.is_finite(value))
        for value in (total, means, variances, ess)])).numpy())
    return LedhPFPFOTTFResult(method_id, int(seed), int(num_particles), total,
        means, variances, ess, sum(int(row["resampled"]) for row in diagnostics), diagnostics, finite)


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
