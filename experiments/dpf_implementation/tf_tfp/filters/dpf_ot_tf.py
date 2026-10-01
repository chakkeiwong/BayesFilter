"""TensorFlow finite-Sinkhorn relaxed OT-DPF core."""

from __future__ import annotations

from typing import Any, Callable

import tensorflow as tf

from experiments.dpf_implementation.tf_tfp.filters.bootstrap_pf_tf import (
    DTYPE,
    ParticleFilterTFResult,
    normalize_log_weights_tf,
    weighted_mean_and_variance_tf,
)
from experiments.dpf_implementation.tf_tfp.resampling.annealed_transport_tf import (
    annealed_transport_resample_tf,
)
from experiments.dpf_implementation.tf_tfp.resampling.sinkhorn_tf import (
    SinkhornLogStateTF,
    sinkhorn_resample_tf,
)


def run_ot_dpf_tf(
    *,
    observations: tf.Tensor,
    initial_sample: Callable[[int, int], tf.Tensor],
    transition_sample: Callable[[tf.Tensor, int, int], tf.Tensor],
    observation_log_density: Callable[[tf.Tensor, tf.Tensor, int], tf.Tensor],
    seed: int,
    num_particles: int,
    ess_threshold_ratio: float = 0.5,
    sinkhorn_epsilon: float = 0.5,
    sinkhorn_iterations: int = 80,
    sinkhorn_tolerance: float = 1e-7,
    transport_method: str = "annealed_transport",
    annealed_scaling: float = 0.9,
    annealed_convergence_threshold: float = 1e-3,
    retained_teacher_warmstart_fn: Callable[[tf.Tensor, tf.Tensor, float, int], SinkhornLogStateTF] | None = None,
    jit_compile: bool = True,
) -> ParticleFilterTFResult:
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
        retained_teacher_warmstart_fn=retained_teacher_warmstart_fn)
    total, history = call(observations, particles)
    means, variances, ess, _, _ = history
    diagnostics = particle_ot_reports(history, call, jit_compile=jit_compile)
    finite = bool(tf.reduce_all(tf.stack([tf.reduce_all(tf.math.is_finite(value))
        for value in (total, means, variances, ess)])).numpy())
    method_suffix = {"annealed_transport": "annealed_transport_tf",
        "fixed_target_sinkhorn": "fixed_target_sinkhorn_local_comparator_tf",
        "retained_teacher_sinkhorn_warmstart": "retained_teacher_sinkhorn_warmstart_tf"}.get(transport_method, transport_method)
    return ParticleFilterTFResult(f"ot_dpf_{method_suffix}", int(seed), int(num_particles), total,
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
    retained_teacher_warmstart_fn: Callable[[tf.Tensor, tf.Tensor, float, int], SinkhornLogStateTF] | None,
    time_index: int,
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
    if transport_method == "retained_teacher_sinkhorn_warmstart":
        if retained_teacher_warmstart_fn is None:
            raise ValueError(
                "retained_teacher_warmstart_fn is required for transport_method='retained_teacher_sinkhorn_warmstart'"
            )
        initial_state = retained_teacher_warmstart_fn(
            tf.cast(particles, DTYPE),
            tf.cast(weights, DTYPE),
            float(epsilon),
            time_index,
        )
        result = sinkhorn_resample_tf(
            particles,
            weights,
            epsilon=epsilon,
            max_iterations=sinkhorn_iterations,
            tolerance=sinkhorn_tolerance,
            initial_state=initial_state,
        )
        diagnostics = dict(result.diagnostics)
        diagnostics.update(
            {
                "warmstart_provider": getattr(retained_teacher_warmstart_fn, "__name__", type(retained_teacher_warmstart_fn).__name__),
                "warmstart_callback_used": True,
                "corrective_refinement_retained": True,
                "transport_method": "retained_teacher_sinkhorn_warmstart",
            }
        )
        return (
            type(result)(
                particles=result.particles,
                coupling=result.coupling,
                source_weights=result.source_weights,
                target_weights=result.target_weights,
                final_state=result.final_state,
                canonicalized_final_state=result.canonicalized_final_state,
                diagnostics=diagnostics,
            ),
            "retained_teacher_sinkhorn_warmstart_tf",
        )
    raise ValueError(f"unknown transport_method: {transport_method}")


def _finite_bool(value: tf.Tensor) -> bool:
    return bool(tf.reduce_all(tf.math.is_finite(tf.cast(value, DTYPE))).numpy())


def _float(value: tf.Tensor) -> float:
    return float(tf.cast(value, DTYPE).numpy())
