"""Diagnostic TensorFlow/TFP replica-exchange fixed-HMC mechanics.

This module is intentionally in the testing namespace.  It validates and exposes
the mechanics needed to assess a multimodal sampler candidate; it does not define
a BayesFilter inference default or estimate posterior mode weights.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import tensorflow as tf
import tensorflow_probability as tfp


TensorTarget = Callable[[tf.Tensor], tf.Tensor]
ReplicaSampler = Callable[[tf.Tensor, tf.Tensor], Mapping[str, tf.Tensor]]


def validate_replica_exchange_configuration(
    inverse_temperatures: Sequence[float],
    step_sizes: Sequence[float],
    *,
    num_leapfrog_steps: int,
    num_steps: int,
) -> None:
    """Validate static configuration before tracing a replica-exchange graph."""

    betas = tuple(float(value) for value in inverse_temperatures)
    steps = tuple(float(value) for value in step_sizes)
    if len(betas) < 2:
        raise ValueError("replica exchange requires at least two temperatures")
    if len(steps) != len(betas):
        raise ValueError("one step size is required per inverse temperature")
    if betas[0] != 1.0:
        raise ValueError("the first inverse temperature must equal one")
    if any(not 0.0 < beta <= 1.0 for beta in betas):
        raise ValueError("power-tempering inverse temperatures must be in (0, 1]")
    if any(left <= right for left, right in zip(betas, betas[1:])):
        raise ValueError("inverse temperatures must be strictly decreasing")
    if any(not step > 0.0 for step in steps):
        raise ValueError("step sizes must be positive")
    if isinstance(num_leapfrog_steps, bool) or int(num_leapfrog_steps) < 2:
        raise ValueError("fixed HMC requires at least two leapfrog steps")
    if isinstance(num_steps, bool) or int(num_steps) <= 0:
        raise ValueError("num_steps must be positive")


def replica_identities_at_temperatures(
    accepted_swap_matrices: tf.Tensor,
) -> tf.Tensor:
    """Reconstruct the initial replica identity occupying each temperature.

    TFP records `accepted[old_temperature, new_temperature, chain]`.  The
    returned tensor has shape `[step, temperature, chain]` and stores the
    initial temperature identity at every later temperature.
    """

    accepted = tf.convert_to_tensor(accepted_swap_matrices, tf.bool)
    if accepted.shape.rank != 4:
        raise ValueError("accepted swap matrices must have rank four")
    tf.debugging.assert_equal(
        tf.reduce_sum(tf.cast(accepted, tf.int32), axis=1),
        tf.ones(tf.gather(tf.shape(accepted), (0, 2, 3)), tf.int32),
        message="each accepted-swap row must select one new temperature",
    )
    tf.debugging.assert_equal(
        tf.reduce_sum(tf.cast(accepted, tf.int32), axis=2),
        tf.ones(tf.gather(tf.shape(accepted), (0, 1, 3)), tf.int32),
        message="each accepted-swap column must select one old temperature",
    )
    replica_count = tf.shape(accepted)[1]
    chain_count = tf.shape(accepted)[3]
    initial = tf.broadcast_to(
        tf.range(replica_count, dtype=tf.int32)[:, tf.newaxis],
        (replica_count, chain_count),
    )

    def apply_swap(identities: tf.Tensor, matrix: tf.Tensor) -> tf.Tensor:
        # `[old, new, chain] -> [chain, new, old]` gives the source position
        # for every destination temperature.
        old_position_for_new = tf.argmax(
            tf.transpose(tf.cast(matrix, tf.int32), (2, 1, 0)),
            axis=2,
            output_type=tf.int32,
        )
        new_by_chain = tf.gather(
            tf.transpose(identities, (1, 0)),
            old_position_for_new,
            axis=1,
            batch_dims=1,
        )
        return tf.transpose(new_by_chain, (1, 0))

    return tf.scan(apply_swap, accepted, initializer=initial)


def pre_swap_states_from_accepted_swaps(
    post_swap_states: tf.Tensor,
    accepted_swap_matrices: tf.Tensor,
) -> tf.Tensor:
    """Invert accepted pair swaps for rank-1 event states.

    TFP records `accepted[old_temperature, new_temperature, chain]`.  Exact
    accepted pair swaps are permutations, so this recovers the locally updated
    state at every old temperature before the exchange is applied.
    """

    post = tf.convert_to_tensor(post_swap_states)
    accepted = tf.convert_to_tensor(accepted_swap_matrices, tf.bool)
    if post.shape.rank != 4:
        raise ValueError("post-swap states must have [step, temperature, chain, event]")
    if accepted.shape.rank != 4:
        raise ValueError("accepted swap matrices must have rank four")
    tf.debugging.assert_equal(
        tf.shape(post)[:3],
        tf.gather(tf.shape(accepted), (0, 1, 3)),
        message="post-state and accepted-swap shapes are incompatible",
    )
    return tf.einsum(
        "sonc,sncd->socd",
        tf.cast(accepted, post.dtype),
        post,
    )


def replica_travel_diagnostics(
    identities_at_temperature: tf.Tensor,
) -> Mapping[str, tf.Tensor]:
    """Summarize cold/hot visits and completed hot-to-cold round trips."""

    identities = tf.convert_to_tensor(identities_at_temperature, tf.int32)
    if identities.shape.rank != 3:
        raise ValueError("identity trace must have shape [step, temperature, chain]")
    replica_count = tf.shape(identities)[1]
    chain_count = tf.shape(identities)[2]
    identity_ids = tf.range(replica_count, dtype=tf.int32)
    positions = tf.argmax(
        tf.cast(
            identities[:, :, :, tf.newaxis]
            == identity_ids[tf.newaxis, tf.newaxis, tf.newaxis, :],
            tf.int32,
        ),
        axis=1,
        output_type=tf.int32,
    )
    # positions: [step, chain, identity].  Phase 0 waits for cold, phase 1 has
    # seen cold and waits for hot, and phase 2 has seen hot and waits to return.
    at_hot = positions == replica_count - 1
    at_cold = positions == 0
    initial_phase = tf.broadcast_to(
        tf.cast(identity_ids == 0, tf.int32)[tf.newaxis, :],
        (chain_count, replica_count),
    )

    def update_round_trip(
        state: tuple[tf.Tensor, tf.Tensor],
        endpoint_flags: tuple[tf.Tensor, tf.Tensor],
    ) -> tuple[tf.Tensor, tf.Tensor]:
        phase, count = state
        cold, hot = endpoint_flags
        completed = tf.logical_and(phase == 2, cold)
        count = count + tf.cast(completed, tf.int32)
        phase = tf.where(completed, tf.ones_like(phase), phase)
        phase = tf.where(tf.logical_and(phase == 0, cold), tf.ones_like(phase), phase)
        phase = tf.where(tf.logical_and(phase == 1, hot), tf.fill(tf.shape(phase), 2), phase)
        return phase, count

    _phases, cumulative_round_trips = tf.scan(
        update_round_trip,
        (at_cold, at_hot),
        initializer=(initial_phase, tf.zeros_like(initial_phase)),
    )
    return {
        "temperature_position_by_chain_identity": positions,
        "visited_cold": tf.reduce_any(at_cold, axis=0),
        "visited_hot": tf.reduce_any(at_hot, axis=0),
        "round_trip_returns": cumulative_round_trips[-1],
    }


def make_replica_exchange_fixed_hmc_sampler(
    target_log_prob_fn: TensorTarget,
    initial_state: tf.Tensor,
    *,
    inverse_temperatures: Sequence[float],
    step_sizes: Sequence[float],
    num_leapfrog_steps: int,
    num_steps: int,
    jit_compile: bool = True,
) -> ReplicaSampler:
    """Build one reusable, fully traced replica-exchange sampler.

    Reusing the returned callable separates first-call compilation from cached
    transition timing.  Each call may continue from the preceding terminal
    state with a fresh stateless seed.
    """

    validate_replica_exchange_configuration(
        inverse_temperatures,
        step_sizes,
        num_leapfrog_steps=num_leapfrog_steps,
        num_steps=num_steps,
    )
    state = tf.convert_to_tensor(initial_state)
    if state.shape.rank is None or state.shape.rank < 3:
        raise ValueError("initial_state must have shape [replica, chain, event...]")
    if state.shape[0] != len(tuple(inverse_temperatures)):
        raise ValueError("initial_state replica dimension does not match temperatures")
    if not state.dtype.is_floating:
        raise ValueError("initial_state must have a floating dtype")
    if not state.shape.is_fully_defined():
        raise ValueError("initial_state must have a fully static shape")

    betas = tf.constant(tuple(inverse_temperatures), state.dtype)
    step_vector = tf.constant(tuple(step_sizes), state.dtype)
    step_shape = (len(tuple(step_sizes)),) + (1,) * (state.shape.rank - 1)
    hmc_steps = tf.reshape(step_vector, step_shape)

    def make_kernel(replica_target_log_prob_fn: TensorTarget) -> Any:
        return tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=replica_target_log_prob_fn,
            step_size=hmc_steps,
            num_leapfrog_steps=int(num_leapfrog_steps),
        )

    kernel = tfp.mcmc.ReplicaExchangeMC(
        target_log_prob_fn=target_log_prob_fn,
        inverse_temperatures=betas,
        make_kernel_fn=make_kernel,
        swap_proposal_fn=tfp.mcmc.even_odd_swap_proposal_fn(swap_frequency=1.0),
        state_includes_replicas=True,
        validate_args=True,
    )

    def trace_fn(_state: tf.Tensor, results: Any) -> Mapping[str, tf.Tensor]:
        return {
            "hmc_is_accepted": results.pre_swap_replica_results.is_accepted,
            "hmc_log_accept_ratio": results.pre_swap_replica_results.log_accept_ratio,
            "swap_is_proposed_adjacent": results.is_swap_proposed_adjacent,
            "swap_is_accepted_adjacent": results.is_swap_accepted_adjacent,
            "swap_is_accepted_matrix": results.is_swap_accepted,
            "potential_energy": results.potential_energy,
        }

    @tf.function(
        input_signature=(
            tf.TensorSpec(state.shape, state.dtype),
            tf.TensorSpec([2], tf.int32),
        ),
        jit_compile=jit_compile,
        reduce_retracing=False,
    )
    def sample(current_state: tf.Tensor, seed: tf.Tensor) -> Mapping[str, tf.Tensor]:
        samples, trace = tfp.mcmc.sample_chain(
            num_results=int(num_steps),
            num_burnin_steps=0,
            current_state=current_state,
            kernel=kernel,
            trace_fn=trace_fn,
            seed=seed,
        )
        pre_swap_states = pre_swap_states_from_accepted_swaps(
            samples, trace["swap_is_accepted_matrix"]
        )
        identities = replica_identities_at_temperatures(
            trace["swap_is_accepted_matrix"]
        )
        travel = replica_travel_diagnostics(identities)
        return {
            "replica_states": tf.convert_to_tensor(samples, state.dtype),
            "pre_swap_replica_states": tf.convert_to_tensor(
                pre_swap_states, state.dtype
            ),
            "cold_states": tf.convert_to_tensor(samples[:, 0], state.dtype),
            "hmc_is_accepted": tf.convert_to_tensor(
                trace["hmc_is_accepted"], tf.bool
            ),
            "hmc_log_accept_ratio": tf.convert_to_tensor(
                trace["hmc_log_accept_ratio"], state.dtype
            ),
            "swap_is_proposed_adjacent": tf.convert_to_tensor(
                trace["swap_is_proposed_adjacent"], tf.bool
            ),
            "swap_is_accepted_adjacent": tf.convert_to_tensor(
                trace["swap_is_accepted_adjacent"], tf.bool
            ),
            "swap_is_accepted_matrix": tf.convert_to_tensor(
                trace["swap_is_accepted_matrix"], tf.bool
            ),
            "potential_energy": tf.convert_to_tensor(
                trace["potential_energy"], state.dtype
            ),
            "replica_identities_at_temperature": identities,
            **travel,
        }

    return sample


def run_replica_exchange_fixed_hmc(
    target_log_prob_fn: TensorTarget,
    initial_state: tf.Tensor,
    *,
    inverse_temperatures: Sequence[float],
    step_sizes: Sequence[float],
    num_leapfrog_steps: int,
    num_steps: int,
    seed: tuple[int, int],
    jit_compile: bool = True,
) -> Mapping[str, tf.Tensor]:
    """Run fully traced replica exchange and return mechanics diagnostics.

    `initial_state` must have shape `[replica, chain, ...event]`.  All steps are
    returned; callers choose a warm-up prefix without losing replica identities.
    """

    state = tf.convert_to_tensor(initial_state)
    sampler = make_replica_exchange_fixed_hmc_sampler(
        target_log_prob_fn,
        state,
        inverse_temperatures=inverse_temperatures,
        step_sizes=step_sizes,
        num_leapfrog_steps=num_leapfrog_steps,
        num_steps=num_steps,
        jit_compile=jit_compile,
    )
    return sampler(state, tf.constant(seed, tf.int32))


def replica_exchange_finite(trace: Mapping[str, tf.Tensor]) -> tf.Tensor:
    """Return whether every scientific floating trace field is finite."""

    fields = (
        "replica_states",
        "hmc_log_accept_ratio",
        "potential_energy",
    )
    return tf.reduce_all(
        tf.stack(
            [tf.reduce_all(tf.math.is_finite(trace[name])) for name in fields]
        )
    )
