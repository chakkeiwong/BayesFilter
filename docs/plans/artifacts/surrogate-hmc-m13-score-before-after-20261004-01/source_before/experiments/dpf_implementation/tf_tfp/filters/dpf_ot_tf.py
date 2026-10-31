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
) -> ParticleFilterTFResult:
    particles = tf.cast(initial_sample(num_particles, seed), DTYPE)
    log_weights = tf.fill([num_particles], -tf.math.log(tf.cast(num_particles, DTYPE)))
    filtered_means = []
    filtered_variances = []
    ess_by_time = []
    diagnostics: list[dict[str, Any]] = []
    resampling_count = 0
    log_likelihood = tf.constant(0.0, dtype=DTYPE)

    for t, observation in enumerate(tf.unstack(tf.cast(observations, DTYPE), axis=0)):
        particles = tf.cast(transition_sample(particles, seed, t), DTYPE)
        obs_log_weights = tf.cast(observation_log_density(particles, observation, t), DTYPE)
        weights, incremental = normalize_log_weights_tf(log_weights + obs_log_weights)
        log_likelihood = log_likelihood + incremental
        ess = 1.0 / tf.reduce_sum(weights * weights)
        mean, variance = weighted_mean_and_variance_tf(particles, weights)
        filtered_means.append(mean)
        filtered_variances.append(variance)
        ess_by_time.append(ess)
        do_resample = bool((ess < ess_threshold_ratio * num_particles).numpy())
        if do_resample:
            resampled, resampling_method = _resample_particles(
                particles=particles,
                weights=weights,
                log_weights=tf.math.log(tf.maximum(weights, tf.constant(1e-300, dtype=DTYPE))),
                transport_method=transport_method,
                epsilon=sinkhorn_epsilon,
                sinkhorn_iterations=sinkhorn_iterations,
                sinkhorn_tolerance=sinkhorn_tolerance,
                annealed_scaling=annealed_scaling,
                annealed_convergence_threshold=annealed_convergence_threshold,
                retained_teacher_warmstart_fn=retained_teacher_warmstart_fn,
                time_index=t,
            )
            particles = resampled.particles
            log_weights = tf.fill([num_particles], -tf.math.log(tf.cast(num_particles, DTYPE)))
            resampling_count += 1
            diag: dict[str, Any] = {
                "time_index": int(t),
                "ess": float(ess.numpy()),
                "ess_ratio": float((ess / tf.cast(num_particles, DTYPE)).numpy()),
                "resampled": True,
                "resampling_method": resampling_method,
                **resampled.diagnostics,
            }
        else:
            log_weights = tf.math.log(tf.maximum(weights, tf.constant(1e-300, dtype=DTYPE)))
            diag = {
                "time_index": int(t),
                "ess": float(ess.numpy()),
                "ess_ratio": float((ess / tf.cast(num_particles, DTYPE)).numpy()),
                "resampled": False,
                "resampling_method": "none",
                "backend": "tensorflow",
            }
        diagnostics.append(diag)

    filtered_means_tensor = tf.stack(filtered_means, axis=0)
    filtered_variances_tensor = tf.stack(filtered_variances, axis=0)
    ess_tensor = tf.stack(ess_by_time, axis=0)
    finite = bool(
        tf.math.is_finite(log_likelihood).numpy()
        and tf.reduce_all(tf.math.is_finite(filtered_means_tensor)).numpy()
        and tf.reduce_all(tf.math.is_finite(filtered_variances_tensor)).numpy()
        and tf.reduce_all(tf.math.is_finite(ess_tensor)).numpy()
    )
    if transport_method == "annealed_transport":
        method_suffix = "annealed_transport_tf"
    elif transport_method == "fixed_target_sinkhorn":
        method_suffix = "fixed_target_sinkhorn_local_comparator_tf"
    elif transport_method == "retained_teacher_sinkhorn_warmstart":
        method_suffix = "retained_teacher_sinkhorn_warmstart_tf"
    else:
        method_suffix = transport_method
    return ParticleFilterTFResult(
        method_id=f"ot_dpf_{method_suffix}",
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
            int(time_index),
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
