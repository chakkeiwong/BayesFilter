"""Streaming GPU-oriented LEDH-PFPF-OT value path.

This module is the streaming companion to
``experimental_batched_ledh_pfpf_ot_tf``.  It keeps the same deterministic
fixed-branch accounting, but shapes the computation for XLA/GPU execution:

* time recursion uses ``tf.while_loop``;
* per-particle LEDH maps are processed in chunks;
* streaming OT avoids returning a dense ``[B, N, N]`` transport matrix;
* likelihood-only runs can avoid storing filtered history.

Repository governance now treats the GPU-oriented LEDH-PFPF-OT TF32 route as
the default production target for DPF transport work.  Public API exposure and
HMC/posterior claims remain separately gated.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import tensorflow as tf

from bayesfilter.highdim.transport_chunk_policy import resolve_transport_chunks

from experiments.dpf_implementation.tf_tfp.filters.experimental_batched_ledh_pfpf_ot_tf import (
    DTYPE,
    AnnealedTransportWarmstartStateTF,
    BatchedLEDHFlowTensors,
    BatchedLEDHPFPFOTValueScoreTensors,
    batched_annealed_transport_core_tf,
    batched_ledh_flow_core_tf,
    uniform_log_weights,
    _log_weight_floor,
    _normalize_log_weights,
    _require_equal,
    _require_static_rank,
    _static_shape,
    _to_float_tensor,
    _weighted_mean_and_variance,
)
from experiments.dpf_implementation.tf_tfp.resampling.annealed_transport_tf import (
    DEFAULT_STREAMING_CHUNK_SIZE,
)


@dataclass(frozen=True)
class StreamingLEDHFlowTensors:
    """Chunked LEDH flow tensors needed by the value recursion."""

    post_flow_particles: tf.Tensor
    pre_flow_log_density: tf.Tensor
    forward_log_det: tf.Tensor

    def __post_init__(self) -> None:
        post_flow_particles = _to_float_tensor(
            self.post_flow_particles,
            "post_flow_particles",
        )
        pre_flow_log_density = _to_float_tensor(
            self.pre_flow_log_density,
            "pre_flow_log_density",
        )
        forward_log_det = _to_float_tensor(self.forward_log_det, "forward_log_det")
        _require_static_rank(post_flow_particles, 3, "post_flow_particles")
        _require_static_rank(pre_flow_log_density, 2, "pre_flow_log_density")
        _require_static_rank(forward_log_det, 2, "forward_log_det")
        batch_size, num_particles, _state_dim = _static_shape(
            post_flow_particles,
            "post_flow_particles",
        )
        expected = (batch_size, num_particles)
        if _static_shape(pre_flow_log_density, "pre_flow_log_density") != expected:
            raise ValueError("pre_flow_log_density shape mismatch")
        if _static_shape(forward_log_det, "forward_log_det") != expected:
            raise ValueError("forward_log_det shape mismatch")
        object.__setattr__(self, "post_flow_particles", post_flow_particles)
        object.__setattr__(self, "pre_flow_log_density", pre_flow_log_density)
        object.__setattr__(self, "forward_log_det", forward_log_det)


@dataclass(frozen=True)
class StreamingLEDHPFPFOTValueTensors:
    """Streaming LEDH-PFPF-OT value-recursion outputs."""

    log_likelihood: tf.Tensor
    filtered_means: tf.Tensor
    filtered_variances: tf.Tensor
    ess_by_time: tf.Tensor
    final_particles: tf.Tensor
    max_row_residual: tf.Tensor
    max_column_residual: tf.Tensor

    def __post_init__(self) -> None:
        log_likelihood = _to_float_tensor(self.log_likelihood, "log_likelihood")
        filtered_means = _to_float_tensor(self.filtered_means, "filtered_means")
        filtered_variances = _to_float_tensor(
            self.filtered_variances,
            "filtered_variances",
        )
        ess_by_time = _to_float_tensor(self.ess_by_time, "ess_by_time")
        final_particles = _to_float_tensor(self.final_particles, "final_particles")
        max_row_residual = _to_float_tensor(self.max_row_residual, "max_row_residual")
        max_column_residual = _to_float_tensor(self.max_column_residual, "max_column_residual")
        _require_static_rank(log_likelihood, 1, "log_likelihood")
        _require_static_rank(filtered_means, 3, "filtered_means")
        _require_static_rank(filtered_variances, 3, "filtered_variances")
        _require_static_rank(ess_by_time, 2, "ess_by_time")
        _require_static_rank(final_particles, 3, "final_particles")
        _require_static_rank(max_row_residual, 0, "max_row_residual")
        _require_static_rank(max_column_residual, 0, "max_column_residual")
        mean_shape = _static_shape(filtered_means, "filtered_means")
        variance_shape = _static_shape(filtered_variances, "filtered_variances")
        if variance_shape != mean_shape:
            raise ValueError(
                "filtered_variances shape mismatch: "
                f"got {variance_shape}, expected {mean_shape}"
            )
        time_steps, batch_size, state_dim = mean_shape
        _require_equal(
            _static_shape(log_likelihood, "log_likelihood")[0],
            batch_size,
            "log_likelihood batch size",
        )
        expected_ess_shape = (time_steps, batch_size)
        if _static_shape(ess_by_time, "ess_by_time") != expected_ess_shape:
            raise ValueError(
                "ess_by_time shape mismatch: "
                f"got {_static_shape(ess_by_time, 'ess_by_time')}, "
                f"expected {expected_ess_shape}"
            )
        final_particle_shape = _static_shape(final_particles, "final_particles")
        if final_particle_shape[0] != batch_size or final_particle_shape[2] != state_dim:
            raise ValueError("final_particles shape mismatch")
        object.__setattr__(self, "log_likelihood", log_likelihood)
        object.__setattr__(self, "filtered_means", filtered_means)
        object.__setattr__(self, "filtered_variances", filtered_variances)
        object.__setattr__(self, "ess_by_time", ess_by_time)
        object.__setattr__(self, "final_particles", final_particles)
        object.__setattr__(self, "max_row_residual", max_row_residual)
        object.__setattr__(self, "max_column_residual", max_column_residual)


def batched_ledh_flow_streaming_particles_tf(
    *,
    pre_flow_particles: tf.Tensor,
    ancestors: tf.Tensor,
    observation: tf.Tensor,
    transition_matrix: tf.Tensor,
    transition_covariance: tf.Tensor,
    observation_covariance: tf.Tensor,
    observation_fn: Callable[[tf.Tensor], tf.Tensor],
    observation_jacobian_fn: Callable[[tf.Tensor], tf.Tensor],
    observation_residual_fn: Callable[[tf.Tensor, tf.Tensor], tf.Tensor],
    prior_mean_fn: Callable[[tf.Tensor], tf.Tensor] | None = None,
    jitter: float | tf.Tensor = 1.0e-9,
    particle_chunk_size: int = DEFAULT_STREAMING_CHUNK_SIZE,
) -> StreamingLEDHFlowTensors:
    """Apply LEDH flow in particle chunks without storing ``[B,N,D,D]`` state."""

    if particle_chunk_size <= 0:
        raise ValueError("particle_chunk_size must be positive")

    x0 = _to_float_tensor(pre_flow_particles, "pre_flow_particles")
    ancestors = _to_float_tensor(ancestors, "ancestors")
    _require_static_rank(x0, 3, "pre_flow_particles")
    _require_static_rank(ancestors, 3, "ancestors")
    batch_size, num_particles, state_dim = _static_shape(x0, "pre_flow_particles")
    if _static_shape(ancestors, "ancestors") != (batch_size, num_particles, state_dim):
        raise ValueError("ancestors shape mismatch")

    chunk_size = int(particle_chunk_size)
    chunk_tensor = tf.constant(chunk_size, dtype=tf.int32)
    num_particles_tensor = tf.constant(int(num_particles), dtype=tf.int32)
    num_blocks = (int(num_particles) + chunk_size - 1) // chunk_size

    post_blocks = tf.TensorArray(
        dtype=DTYPE,
        size=int(num_blocks),
        element_shape=tf.TensorShape([batch_size, chunk_size, state_dim]),
    )
    pre_log_blocks = tf.TensorArray(
        dtype=DTYPE,
        size=int(num_blocks),
        element_shape=tf.TensorShape([batch_size, chunk_size]),
    )
    logdet_blocks = tf.TensorArray(
        dtype=DTYPE,
        size=int(num_blocks),
        element_shape=tf.TensorShape([batch_size, chunk_size]),
    )

    def cond(
        particle_start: tf.Tensor,
        _post_blocks: tf.TensorArray,
        _pre_log_blocks: tf.TensorArray,
        _logdet_blocks: tf.TensorArray,
    ) -> tf.Tensor:
        return particle_start < num_particles_tensor

    def body(
        particle_start: tf.Tensor,
        post_ta: tf.TensorArray,
        pre_log_ta: tf.TensorArray,
        logdet_ta: tf.TensorArray,
    ):
        pre_chunk = _slice_axis1_padded_3d(x0, particle_start, chunk_size)
        ancestor_chunk = _slice_axis1_padded_3d(ancestors, particle_start, chunk_size)
        flow: BatchedLEDHFlowTensors = batched_ledh_flow_core_tf(
            pre_flow_particles=pre_chunk,
            ancestors=ancestor_chunk,
            observation=observation,
            transition_matrix=transition_matrix,
            transition_covariance=transition_covariance,
            observation_covariance=observation_covariance,
            observation_fn=observation_fn,
            observation_jacobian_fn=observation_jacobian_fn,
            observation_residual_fn=observation_residual_fn,
            prior_mean_fn=prior_mean_fn,
            jitter=jitter,
        )
        block_index = particle_start // chunk_tensor
        post_ta = post_ta.write(block_index, flow.post_flow_particles)
        pre_log_ta = pre_log_ta.write(block_index, flow.pre_flow_log_density)
        logdet_ta = logdet_ta.write(block_index, flow.forward_log_det)
        return particle_start + chunk_tensor, post_ta, pre_log_ta, logdet_ta

    _, post_blocks, pre_log_blocks, logdet_blocks = tf.while_loop(
        cond,
        body,
        loop_vars=(
            tf.constant(0, dtype=tf.int32),
            post_blocks,
            pre_log_blocks,
            logdet_blocks,
        ),
        parallel_iterations=1,
        maximum_iterations=int(num_blocks),
    )

    post_flow = _flatten_particle_blocks_3d(
        post_blocks,
        batch_size=batch_size,
        num_blocks=int(num_blocks),
        chunk_size=chunk_size,
        num_particles=num_particles_tensor,
        state_dim=state_dim,
    )
    pre_log = _flatten_particle_blocks_2d(
        pre_log_blocks,
        batch_size=batch_size,
        num_blocks=int(num_blocks),
        chunk_size=chunk_size,
        num_particles=num_particles_tensor,
    )
    logdet = _flatten_particle_blocks_2d(
        logdet_blocks,
        batch_size=batch_size,
        num_blocks=int(num_blocks),
        chunk_size=chunk_size,
        num_particles=num_particles_tensor,
    )
    post_flow.set_shape([batch_size, num_particles, state_dim])
    pre_log.set_shape([batch_size, num_particles])
    logdet.set_shape([batch_size, num_particles])
    return StreamingLEDHFlowTensors(
        post_flow_particles=post_flow,
        pre_flow_log_density=pre_log,
        forward_log_det=logdet,
    )


def streaming_batched_ledh_pfpf_ot_value_core_tf(
    *,
    observations: tf.Tensor,
    initial_particles: tf.Tensor,
    fixed_resampling_mask: tf.Tensor,
    transition_matrix: tf.Tensor,
    transition_covariance: tf.Tensor,
    observation_covariance: tf.Tensor,
    observation_fn: Callable[[tf.Tensor], tf.Tensor],
    observation_jacobian_fn: Callable[[tf.Tensor], tf.Tensor],
    observation_residual_fn: Callable[[tf.Tensor, tf.Tensor], tf.Tensor],
    transition_log_density_fn: Callable[[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor],
    observation_log_density_fn: Callable[[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor],
    prior_mean_fn: Callable[[tf.Tensor, tf.Tensor], tf.Tensor] | None = None,
    pre_flow_particles: tf.Tensor | None = None,
    pre_flow_step_fn: Callable[[tf.Tensor, tf.Tensor], tf.Tensor] | None = None,
    initial_log_weights: tf.Tensor | None = None,
    sinkhorn_epsilon: float | tf.Tensor = 0.5,
    annealed_scaling: float | tf.Tensor = 0.9,
    annealed_convergence_threshold: float | tf.Tensor = 1.0e-3,
    sinkhorn_iterations: int | tf.Tensor = 80,
    ledh_jitter: float | tf.Tensor = 1.0e-9,
    transport_gradient_mode: str = "raw",
    transport_plan_mode: str = "streaming",
    transport_ad_mode: str = "stabilized",
    row_chunk_size: int | None = None,
    col_chunk_size: int | None = None,
    particle_chunk_size: int = DEFAULT_STREAMING_CHUNK_SIZE,
    skip_transport_when_no_active: bool = True,
    return_history: bool = False,
    retained_teacher_warmstart_fn: Callable[[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor], AnnealedTransportWarmstartStateTF] | None = None,
) -> StreamingLEDHPFPFOTValueTensors:
    """Run a streaming fixed-branch batched LEDH-PFPF-OT value recursion."""

    if transport_plan_mode not in {"streaming", "dense"}:
        raise ValueError("transport_plan_mode must be 'streaming' or 'dense'")
    if particle_chunk_size <= 0:
        raise ValueError("particle_chunk_size must be positive")
    if (pre_flow_particles is None) == (pre_flow_step_fn is None):
        raise ValueError("provide exactly one of pre_flow_particles or pre_flow_step_fn")

    observations = _to_float_tensor(observations, "observations")
    particles = _to_float_tensor(initial_particles, "initial_particles")
    fixed_resampling_mask = tf.convert_to_tensor(fixed_resampling_mask, dtype=tf.bool)
    _require_static_rank(observations, 2, "observations")
    _require_static_rank(particles, 3, "initial_particles")
    _require_static_rank(fixed_resampling_mask, 2, "fixed_resampling_mask")

    batch_size, num_particles, state_dim = _static_shape(particles, "initial_particles")
    chunks = resolve_transport_chunks(
        num_particles,
        row_chunk_size=row_chunk_size,
        col_chunk_size=col_chunk_size,
    )
    row_chunk_size = chunks.row_chunk_size
    col_chunk_size = chunks.col_chunk_size
    time_steps, _observation_dim = _static_shape(observations, "observations")
    expected_mask_shape = (batch_size, time_steps)
    if _static_shape(fixed_resampling_mask, "fixed_resampling_mask") != expected_mask_shape:
        raise ValueError(
            "fixed_resampling_mask shape mismatch: "
            f"got {_static_shape(fixed_resampling_mask, 'fixed_resampling_mask')}, "
            f"expected {expected_mask_shape}"
        )
    static_no_resampling = _is_static_all_false_bool_tensor(fixed_resampling_mask)
    if pre_flow_particles is not None:
        pre_flow_particles = _to_float_tensor(pre_flow_particles, "pre_flow_particles")
        _require_static_rank(pre_flow_particles, 4, "pre_flow_particles")
        expected_pre_flow_shape = (batch_size, time_steps, num_particles, state_dim)
        if _static_shape(pre_flow_particles, "pre_flow_particles") != expected_pre_flow_shape:
            raise ValueError(
                "pre_flow_particles shape mismatch: "
                f"got {_static_shape(pre_flow_particles, 'pre_flow_particles')}, "
                f"expected {expected_pre_flow_shape}"
            )

    if initial_log_weights is None:
        log_weights = uniform_log_weights(batch_size, num_particles)
    else:
        log_weights = _to_float_tensor(initial_log_weights, "initial_log_weights")
        _require_static_rank(log_weights, 2, "initial_log_weights")
        if _static_shape(log_weights, "initial_log_weights") != (batch_size, num_particles):
            raise ValueError("initial_log_weights shape mismatch")

    log_likelihood = tf.zeros([batch_size], dtype=DTYPE)
    time_steps_tensor = tf.shape(observations)[0]
    def likelihood_cond(
        time_index: tf.Tensor,
        _particles: tf.Tensor,
        _log_weights: tf.Tensor,
        _log_likelihood: tf.Tensor,
    ) -> tf.Tensor:
        return time_index < time_steps_tensor

    def step_body(
        time_index: tf.Tensor,
        current_particles: tf.Tensor,
        current_log_weights: tf.Tensor,
        current_log_likelihood: tf.Tensor,
        *,
        compute_diagnostics: bool,
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        observation = observations[time_index]
        if pre_flow_step_fn is None:
            pre_flow = pre_flow_particles[:, time_index, :, :]
        else:
            pre_flow = _to_float_tensor(
                pre_flow_step_fn(current_particles, time_index),
                "pre_flow_step_fn output",
            )
            _require_static_rank(pre_flow, 3, "pre_flow_step_fn output")
            if _static_shape(pre_flow, "pre_flow_step_fn output") != (
                batch_size,
                num_particles,
                state_dim,
            ):
                raise ValueError("pre_flow_step_fn output shape mismatch")

        step_prior_mean_fn = None
        if prior_mean_fn is not None:
            step_prior_mean_fn = lambda points: prior_mean_fn(points, time_index)

        flow = batched_ledh_flow_streaming_particles_tf(
            pre_flow_particles=pre_flow,
            ancestors=current_particles,
            observation=observation,
            transition_matrix=transition_matrix,
            transition_covariance=transition_covariance,
            observation_covariance=observation_covariance,
            observation_fn=observation_fn,
            observation_jacobian_fn=observation_jacobian_fn,
            observation_residual_fn=observation_residual_fn,
            prior_mean_fn=step_prior_mean_fn,
            jitter=ledh_jitter,
            particle_chunk_size=particle_chunk_size,
        )
        post_flow = flow.post_flow_particles
        target_transition = _to_float_tensor(
            transition_log_density_fn(post_flow, current_particles, time_index),
            "transition_log_density_fn output",
        )
        target_observation = _to_float_tensor(
            observation_log_density_fn(post_flow, observation, time_index),
            "observation_log_density_fn output",
        )
        _require_static_rank(target_transition, 2, "transition_log_density_fn output")
        _require_static_rank(target_observation, 2, "observation_log_density_fn output")
        if _static_shape(target_transition, "transition_log_density_fn output") != (
            batch_size,
            num_particles,
        ):
            raise ValueError("transition_log_density_fn output shape mismatch")
        if _static_shape(target_observation, "observation_log_density_fn output") != (
            batch_size,
            num_particles,
        ):
            raise ValueError("observation_log_density_fn output shape mismatch")

        corrected_log_weights = (
            current_log_weights
            + target_transition
            + target_observation
            - flow.pre_flow_log_density
            + flow.forward_log_det
        )
        weights, incremental = _normalize_log_weights(corrected_log_weights)
        next_log_likelihood = current_log_likelihood + incremental
        if compute_diagnostics:
            ess = 1.0 / tf.reduce_sum(weights * weights, axis=1)
            mean, variance = _weighted_mean_and_variance(post_flow, weights)
        else:
            ess = tf.zeros([batch_size], dtype=DTYPE)
            mean = tf.zeros([batch_size, state_dim], dtype=DTYPE)
            variance = tf.zeros([batch_size, state_dim], dtype=DTYPE)
        normalized_log_weights = tf.math.log(
            tf.maximum(weights, _log_weight_floor())
        )
        mask = fixed_resampling_mask[:, time_index]

        def do_transport() -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
            warmstart_state = None
            if retained_teacher_warmstart_fn is not None:
                normalized_center = tf.reduce_mean(post_flow, axis=1, keepdims=True)
                normalized_scale = tf.maximum(tf.math.reduce_std(post_flow, axis=1), tf.constant(1e-12, dtype=DTYPE))
                scaled_post_flow = (post_flow - normalized_center) / normalized_scale[:, None, :]
                warmstart_state = retained_teacher_warmstart_fn(
                    scaled_post_flow,
                    normalized_log_weights,
                    mask,
                    tf.cast(sinkhorn_epsilon, DTYPE),
                )
            transported = batched_annealed_transport_core_tf(
                post_flow,
                normalized_log_weights,
                mask,
                epsilon=sinkhorn_epsilon,
                scaling=annealed_scaling,
                convergence_threshold=annealed_convergence_threshold,
                max_iterations=sinkhorn_iterations,
                transport_gradient_mode=transport_gradient_mode,
                transport_plan_mode=transport_plan_mode,
                transport_ad_mode=transport_ad_mode,
                row_chunk_size=row_chunk_size,
                col_chunk_size=col_chunk_size,
                warmstart_state=warmstart_state,
            )
            return (
                transported.particles,
                transported.log_weights,
                transported.max_row_residual,
                transported.max_column_residual,
            )

        def skip_transport() -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
            zero = tf.constant(0.0, dtype=DTYPE)
            return post_flow, normalized_log_weights, zero, zero

        dynamic_no_resampling = tf.logical_not(tf.reduce_any(mask))
        if static_no_resampling:
            next_particles, next_log_weights, step_row_residual, step_column_residual = skip_transport()
        elif skip_transport_when_no_active:
            next_particles, next_log_weights, step_row_residual, step_column_residual = tf.cond(
                dynamic_no_resampling,
                skip_transport,
                do_transport,
            )
        else:
            next_particles, next_log_weights, step_row_residual, step_column_residual = do_transport()

        return (
            next_particles,
            next_log_weights,
            next_log_likelihood,
            mean,
            variance,
            ess,
            step_row_residual,
            step_column_residual,
        )

    if return_history:
        means_ta = tf.TensorArray(
            dtype=DTYPE,
            size=time_steps_tensor,
            element_shape=tf.TensorShape([batch_size, state_dim]),
        )
        variances_ta = tf.TensorArray(
            dtype=DTYPE,
            size=time_steps_tensor,
            element_shape=tf.TensorShape([batch_size, state_dim]),
        )
        ess_ta = tf.TensorArray(
            dtype=DTYPE,
            size=time_steps_tensor,
            element_shape=tf.TensorShape([batch_size]),
        )

        def history_cond(
            time_index: tf.Tensor,
            _particles: tf.Tensor,
            _log_weights: tf.Tensor,
            _log_likelihood: tf.Tensor,
            _means_ta: tf.TensorArray,
            _variances_ta: tf.TensorArray,
            _ess_ta: tf.TensorArray,
            _max_row_residual: tf.Tensor,
            _max_column_residual: tf.Tensor,
        ) -> tf.Tensor:
            return time_index < time_steps_tensor

        def history_body(
            time_index: tf.Tensor,
            current_particles: tf.Tensor,
            current_log_weights: tf.Tensor,
            current_log_likelihood: tf.Tensor,
            means_acc: tf.TensorArray,
            variances_acc: tf.TensorArray,
            ess_acc: tf.TensorArray,
            current_max_row_residual: tf.Tensor,
            current_max_column_residual: tf.Tensor,
        ):
            (
                next_particles,
                next_log_weights,
                next_log_likelihood,
                mean,
                variance,
                ess,
                row_residual,
                column_residual,
            ) = step_body(
                time_index,
                current_particles,
                current_log_weights,
                current_log_likelihood,
                compute_diagnostics=True,
            )
            means_acc = means_acc.write(time_index, mean)
            variances_acc = variances_acc.write(time_index, variance)
            ess_acc = ess_acc.write(time_index, ess)
            return (
                time_index + 1,
                next_particles,
                next_log_weights,
                next_log_likelihood,
                means_acc,
                variances_acc,
                ess_acc,
                tf.maximum(current_max_row_residual, row_residual),
                tf.maximum(current_max_column_residual, column_residual),
            )

        (
            _,
            _final_particles,
            _final_log_weights,
            log_likelihood,
            means_ta,
            variances_ta,
            ess_ta,
            max_row_residual_overall,
            max_column_residual_overall,
        ) = tf.while_loop(
            history_cond,
            history_body,
            loop_vars=(
                tf.constant(0, dtype=tf.int32),
                particles,
                log_weights,
                log_likelihood,
                means_ta,
                variances_ta,
                ess_ta,
                tf.constant(0.0, dtype=DTYPE),
                tf.constant(0.0, dtype=DTYPE),
            ),
            parallel_iterations=1,
            maximum_iterations=time_steps_tensor,
        )
        filtered_means = means_ta.stack()
        filtered_variances = variances_ta.stack()
        ess_by_time = ess_ta.stack()
        filtered_means.set_shape([time_steps, batch_size, state_dim])
        filtered_variances.set_shape([time_steps, batch_size, state_dim])
        ess_by_time.set_shape([time_steps, batch_size])
    else:
        def likelihood_cond(
            time_index: tf.Tensor,
            _particles: tf.Tensor,
            _log_weights: tf.Tensor,
            _log_likelihood: tf.Tensor,
            _max_row_residual: tf.Tensor,
            _max_column_residual: tf.Tensor,
        ) -> tf.Tensor:
            return time_index < time_steps_tensor

        def likelihood_body(
            time_index: tf.Tensor,
            current_particles: tf.Tensor,
            current_log_weights: tf.Tensor,
            current_log_likelihood: tf.Tensor,
            current_max_row_residual: tf.Tensor,
            current_max_column_residual: tf.Tensor,
        ):
            next_particles, next_log_weights, next_log_likelihood, _, _, _, row_residual, column_residual = step_body(
                time_index,
                current_particles,
                current_log_weights,
                current_log_likelihood,
                compute_diagnostics=False,
            )
            return (
                time_index + 1,
                next_particles,
                next_log_weights,
                next_log_likelihood,
                tf.maximum(current_max_row_residual, row_residual),
                tf.maximum(current_max_column_residual, column_residual),
            )

        (
            _,
            _final_particles,
            _final_log_weights,
            log_likelihood,
            max_row_residual_overall,
            max_column_residual_overall,
        ) = tf.while_loop(
            likelihood_cond,
            likelihood_body,
            loop_vars=(
                tf.constant(0, dtype=tf.int32),
                particles,
                log_weights,
                log_likelihood,
                tf.constant(0.0, dtype=DTYPE),
                tf.constant(0.0, dtype=DTYPE),
            ),
            parallel_iterations=1,
            maximum_iterations=time_steps_tensor,
        )
        filtered_means = tf.zeros([0, batch_size, state_dim], dtype=DTYPE)
        filtered_variances = tf.zeros([0, batch_size, state_dim], dtype=DTYPE)
        ess_by_time = tf.zeros([0, batch_size], dtype=DTYPE)

    _final_particles.set_shape([batch_size, num_particles, state_dim])
    return StreamingLEDHPFPFOTValueTensors(
        log_likelihood=log_likelihood,
        filtered_means=filtered_means,
        filtered_variances=filtered_variances,
        ess_by_time=ess_by_time,
        final_particles=_final_particles,
        max_row_residual=max_row_residual_overall,
        max_column_residual=max_column_residual_overall,
    )


def streaming_batched_ledh_pfpf_ot_value_and_score_tf(
    theta_batch: tf.Tensor,
    value_fn: Callable[[tf.Tensor], StreamingLEDHPFPFOTValueTensors | tf.Tensor],
) -> BatchedLEDHPFPFOTValueScoreTensors:
    """Differentiate a fixed relaxed streaming LEDH-PFPF-OT objective."""

    theta = _to_float_tensor(theta_batch, "theta_batch")
    _require_static_rank(theta, 2, "theta_batch")
    batch_size = _static_shape(theta, "theta_batch")[0]

    with tf.GradientTape() as tape:
        tape.watch(theta)
        value_result = value_fn(theta)
        if isinstance(value_result, StreamingLEDHPFPFOTValueTensors):
            value = value_result.log_likelihood
        else:
            value = _to_float_tensor(value_result, "value_fn output")
        _require_static_rank(value, 1, "value_fn output")
        _require_equal(_static_shape(value, "value_fn output")[0], batch_size, "value batch size")
        objective = tf.reduce_sum(value)

    score = tape.gradient(
        objective,
        theta,
        unconnected_gradients=tf.UnconnectedGradients.ZERO,
    )
    if score is None:
        score = tf.zeros_like(theta)
    return BatchedLEDHPFPFOTValueScoreTensors(log_likelihood=value, score=score)


def _slice_axis1_padded_3d(
    tensor: tf.Tensor,
    start: tf.Tensor,
    chunk_size: int,
) -> tf.Tensor:
    total = tf.shape(tensor)[1]
    raw_indices = start + tf.range(tf.cast(chunk_size, tf.int32), dtype=tf.int32)
    safe_indices = tf.minimum(raw_indices, total - 1)
    gathered = tf.gather(tensor, safe_indices, axis=1)
    gathered.set_shape([tensor.shape[0], chunk_size, tensor.shape[2]])
    return gathered


def _flatten_particle_blocks_3d(
    blocks: tf.TensorArray,
    *,
    batch_size: int,
    num_blocks: tf.Tensor,
    chunk_size: int,
    num_particles: tf.Tensor,
    state_dim: int,
) -> tf.Tensor:
    stacked = blocks.stack()
    transposed = tf.transpose(stacked, [1, 0, 2, 3])
    flat = tf.reshape(transposed, [batch_size, num_blocks * chunk_size, state_dim])
    return flat[:, :num_particles, :]


def _flatten_particle_blocks_2d(
    blocks: tf.TensorArray,
    *,
    batch_size: int,
    num_blocks: tf.Tensor,
    chunk_size: int,
    num_particles: tf.Tensor,
) -> tf.Tensor:
    stacked = blocks.stack()
    transposed = tf.transpose(stacked, [1, 0, 2])
    flat = tf.reshape(transposed, [batch_size, num_blocks * chunk_size])
    return flat[:, :num_particles]


def _is_static_all_false_bool_tensor(tensor: tf.Tensor) -> bool:
    try:
        value = tf.get_static_value(tensor)
    except (TypeError, ValueError):
        return False
    if value is None:
        return False
    return not bool(value.any())
