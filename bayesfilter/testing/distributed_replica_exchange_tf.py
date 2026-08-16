"""TensorFlow-only distributed value/score replica-exchange mechanics.

The coordinator owns exact fixed-HMC and adjacent-swap algebra.  An injected
batch evaluator may shard target value/score calls across persistent XLA CPU
workers.  Proposal-path invalidity forces a Metropolis self-loop for that row;
it does not terminate unrelated chains.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import tensorflow as tf


BatchEvaluator = Callable[
    [tf.Tensor, str],
    tuple[tf.Tensor, tf.Tensor, Mapping[str, tf.Tensor], Mapping[str, Any]],
]


def _mass_cholesky(mass_matrix: Any | None, dimension: int) -> tf.Tensor | None:
    if mass_matrix is None:
        return None
    mass = tf.convert_to_tensor(mass_matrix, tf.float64)
    if mass.shape != (dimension, dimension):
        raise ValueError("mass_matrix must have shape [dimension, dimension]")
    tf.debugging.assert_all_finite(mass, "mass_matrix must be finite")
    tf.debugging.assert_near(
        mass,
        tf.transpose(mass),
        atol=tf.constant(1.0e-12, tf.float64),
        rtol=tf.constant(1.0e-12, tf.float64),
        message="mass_matrix must be symmetric",
    )
    tf.debugging.assert_positive(
        tf.linalg.eigvalsh(mass),
        message="mass_matrix must be positive definite",
    )
    return tf.linalg.cholesky(mass)


def _mass_inverse_matvec(cholesky: tf.Tensor | None, momentum: tf.Tensor) -> tf.Tensor:
    if cholesky is None:
        return momentum
    return tf.linalg.cholesky_solve(cholesky, momentum[..., tf.newaxis])[..., 0]


def _kinetic_energy(cholesky: tf.Tensor | None, momentum: tf.Tensor) -> tf.Tensor:
    velocity = _mass_inverse_matvec(cholesky, momentum)
    return 0.5 * tf.reduce_sum(momentum * velocity, axis=2)


def _valid_evaluation(
    value: Any,
    score: Any,
    status: Mapping[str, Any],
) -> tf.Tensor:
    values = tf.convert_to_tensor(value, tf.float64)
    scores = tf.convert_to_tensor(score, tf.float64)
    if values.shape.rank != 1 or scores.shape.rank != 2:
        raise ValueError("value and score must have shapes [row] and [row, dimension]")
    if scores.shape[0] != values.shape[0]:
        raise ValueError("value and score row counts must match")
    if "status_code" not in status or "valid_pre_regularized_score" not in status:
        raise ValueError("target status is missing required validity fields")
    return tf.logical_and(
        tf.convert_to_tensor(status["status_code"], tf.int32) == 0,
        tf.logical_and(
            tf.convert_to_tensor(status["valid_pre_regularized_score"], tf.bool),
            tf.logical_and(
                tf.math.is_finite(values),
                tf.reduce_all(tf.math.is_finite(scores), axis=1),
            ),
        ),
    )


def _fold_seed(master_seed: Any, transition_index: int, domain: int) -> tf.Tensor:
    seed = tf.ensure_shape(tf.convert_to_tensor(master_seed, tf.int32), [2])
    seed = tf.random.experimental.stateless_fold_in(seed, int(transition_index))
    return tf.random.experimental.stateless_fold_in(seed, int(domain))


def leapfrog_proposal(
    state: Any,
    base_target_log_prob: Any,
    base_score: Any,
    momentum: Any,
    *,
    inverse_temperatures: Sequence[float],
    step_sizes: Sequence[float],
    num_leapfrog_steps: int,
    evaluator: BatchEvaluator,
    request_prefix: str,
    mass_matrix: Any | None = None,
) -> Mapping[str, Any]:
    """Integrate one fixed-HMC proposal using distributed value/score waves."""

    position = tf.convert_to_tensor(state, tf.float64)
    target = tf.convert_to_tensor(base_target_log_prob, tf.float64)
    score = tf.convert_to_tensor(base_score, tf.float64)
    initial_momentum = tf.convert_to_tensor(momentum, tf.float64)
    if position.shape.rank != 3:
        raise ValueError("state must have shape [temperature, chain, dimension]")
    if not position.shape.is_fully_defined():
        raise ValueError("state must have a fully static shape")
    replicas, chains, dimension = (int(value) for value in position.shape)
    if target.shape != (replicas, chains) or score.shape != position.shape:
        raise ValueError("cached target/score shapes do not match state")
    if initial_momentum.shape != position.shape:
        raise ValueError("momentum shape does not match state")
    mass_cholesky = _mass_cholesky(mass_matrix, dimension)
    betas = tf.constant(tuple(float(value) for value in inverse_temperatures), tf.float64)
    steps = tf.constant(tuple(float(value) for value in step_sizes), tf.float64)
    if betas.shape != (replicas,) or steps.shape != (replicas,):
        raise ValueError("temperature and step vectors must match replica count")
    if isinstance(num_leapfrog_steps, bool) or int(num_leapfrog_steps) < 2:
        raise ValueError("num_leapfrog_steps must be at least two")
    epsilon = steps[:, tf.newaxis, tf.newaxis]
    beta = betas[:, tf.newaxis, tf.newaxis]
    proposal = tf.identity(position)
    proposal_target = tf.identity(target)
    proposal_score = tf.identity(score)
    path_valid = tf.ones((replicas, chains), tf.bool)
    p = initial_momentum + 0.5 * epsilon * beta * proposal_score
    wave_metadata = []
    wave_valid_counts = []

    for leapfrog_index in range(int(num_leapfrog_steps)):
        candidate = proposal + epsilon * _mass_inverse_matvec(mass_cholesky, p)
        flat_candidate = tf.reshape(candidate, (replicas * chains, dimension))
        value, candidate_score, status, metadata = evaluator(
            flat_candidate,
            f"{request_prefix}-leapfrog-{leapfrog_index:02d}",
        )
        value = tf.reshape(tf.convert_to_tensor(value, tf.float64), (replicas, chains))
        candidate_score = tf.reshape(
            tf.convert_to_tensor(candidate_score, tf.float64),
            (replicas, chains, dimension),
        )
        valid = tf.reshape(
            _valid_evaluation(
                tf.reshape(value, (-1,)),
                tf.reshape(candidate_score, (-1, dimension)),
                status,
            ),
            (replicas, chains),
        )
        path_valid = tf.logical_and(path_valid, valid)
        active = path_valid[..., tf.newaxis]
        proposal = tf.where(active, candidate, proposal)
        proposal_target = tf.where(path_valid, value, proposal_target)
        proposal_score = tf.where(active, candidate_score, tf.zeros_like(candidate_score))
        coefficient = 0.5 if leapfrog_index == int(num_leapfrog_steps) - 1 else 1.0
        p = p + coefficient * epsilon * beta * proposal_score
        wave_metadata.append(metadata)
        wave_valid_counts.append(tf.reduce_sum(tf.cast(valid, tf.int32)))

    final_momentum = p
    return {
        "proposed_state": proposal,
        "proposed_base_target_log_prob": proposal_target,
        "proposed_base_score": proposal_score,
        "initial_momentum": initial_momentum,
        "final_momentum": final_momentum,
        "path_valid": path_valid,
        "wave_valid_counts": tf.stack(wave_valid_counts),
        "wave_metadata": tuple(wave_metadata),
    }


def _gather_temperature_sources(values: tf.Tensor, sources: tf.Tensor) -> tf.Tensor:
    rank = values.shape.rank
    if rank not in (2, 3):
        raise ValueError("temperature values must have rank two or three")
    by_chain = tf.transpose(values, (1, 0) if rank == 2 else (1, 0, 2))
    source_by_chain = tf.transpose(sources, (1, 0))
    gathered = tf.gather(by_chain, source_by_chain, axis=1, batch_dims=1)
    return tf.transpose(gathered, (1, 0) if rank == 2 else (1, 0, 2))


def apply_alternating_adjacent_swaps(
    state: Any,
    base_target_log_prob: Any,
    base_score: Any,
    identities_at_temperature: Any,
    *,
    inverse_temperatures: Sequence[float],
    transition_index: int,
    seed: Any,
) -> Mapping[str, tf.Tensor]:
    """Apply one deterministic even/odd set of exact adjacent exchanges."""

    values = tf.convert_to_tensor(state, tf.float64)
    target = tf.convert_to_tensor(base_target_log_prob, tf.float64)
    score = tf.convert_to_tensor(base_score, tf.float64)
    identities = tf.convert_to_tensor(identities_at_temperature, tf.int32)
    if values.shape.rank != 3 or not values.shape.is_fully_defined():
        raise ValueError("state must have static [temperature, chain, dimension] shape")
    replicas, chains, _dimension = (int(item) for item in values.shape)
    if target.shape != (replicas, chains) or score.shape != values.shape:
        raise ValueError("target/score shapes do not match swap state")
    if identities.shape != (replicas, chains):
        raise ValueError("identity shape does not match swap state")
    betas = tf.constant(tuple(float(value) for value in inverse_temperatures), tf.float64)
    if betas.shape != (replicas,):
        raise ValueError("inverse temperatures do not match replica count")
    parity = int(transition_index) % 2
    pairs = tuple((left, left + 1) for left in range(parity, replicas - 1, 2))
    sources = [tf.fill((chains,), index) for index in range(replicas)]
    proposed_adjacent = tf.zeros((replicas - 1, chains), tf.bool)
    accepted_adjacent = tf.zeros((replicas - 1, chains), tf.bool)
    log_accept_adjacent = tf.fill(
        (replicas - 1, chains), tf.constant(float("-inf"), tf.float64)
    )
    uniform = tf.random.stateless_uniform(
        (replicas - 1, chains), seed=seed, dtype=tf.float64
    )
    for left, right in pairs:
        log_ratio = (betas[left] - betas[right]) * (target[right] - target[left])
        accepted = tf.math.log(uniform[left]) < tf.minimum(
            tf.constant(0.0, tf.float64), log_ratio
        )
        sources[left] = tf.where(accepted, tf.fill((chains,), right), sources[left])
        sources[right] = tf.where(accepted, tf.fill((chains,), left), sources[right])
        proposed_adjacent = tf.tensor_scatter_nd_update(
            proposed_adjacent, [[left]], [tf.ones((chains,), tf.bool)]
        )
        accepted_adjacent = tf.tensor_scatter_nd_update(
            accepted_adjacent, [[left]], [accepted]
        )
        log_accept_adjacent = tf.tensor_scatter_nd_update(
            log_accept_adjacent, [[left]], [log_ratio]
        )
    source_tensor = tf.stack(sources)
    source_by_chain = tf.transpose(source_tensor, (1, 0))
    swap_matrix = tf.transpose(
        tf.one_hot(source_by_chain, depth=replicas, dtype=tf.int32),
        (2, 1, 0),
    ) > 0
    return {
        "state": _gather_temperature_sources(values, source_tensor),
        "base_target_log_prob": _gather_temperature_sources(target, source_tensor),
        "base_score": _gather_temperature_sources(score, source_tensor),
        "identities_at_temperature": _gather_temperature_sources(
            identities, source_tensor
        ),
        "source_temperature_for_destination": source_tensor,
        "swap_is_proposed_adjacent": proposed_adjacent,
        "swap_is_accepted_adjacent": accepted_adjacent,
        "swap_log_accept_ratio_adjacent": log_accept_adjacent,
        "swap_is_accepted_matrix": swap_matrix,
    }


def distributed_replica_exchange_transition(
    state: Any,
    base_target_log_prob: Any,
    base_score: Any,
    identities_at_temperature: Any,
    *,
    inverse_temperatures: Sequence[float],
    step_sizes: Sequence[float],
    num_leapfrog_steps: int,
    transition_index: int,
    master_seed: tuple[int, int],
    evaluator: BatchEvaluator,
    mass_matrix: Any | None = None,
) -> Mapping[str, Any]:
    """Run one distributed fixed-HMC mutation and adjacent exchange."""

    current = tf.convert_to_tensor(state, tf.float64)
    current_target = tf.convert_to_tensor(base_target_log_prob, tf.float64)
    current_score = tf.convert_to_tensor(base_score, tf.float64)
    replicas, chains, dimension = (int(value) for value in current.shape)
    standard_momentum = tf.random.stateless_normal(
        current.shape,
        seed=_fold_seed(master_seed, transition_index, 101),
        dtype=tf.float64,
    )
    mass_cholesky = _mass_cholesky(mass_matrix, dimension)
    momentum = (
        standard_momentum
        if mass_cholesky is None
        else tf.linalg.matvec(mass_cholesky, standard_momentum)
    )
    proposal = leapfrog_proposal(
        current,
        current_target,
        current_score,
        momentum,
        inverse_temperatures=inverse_temperatures,
        step_sizes=step_sizes,
        num_leapfrog_steps=num_leapfrog_steps,
        evaluator=evaluator,
        request_prefix=f"transition-{int(transition_index):06d}",
        mass_matrix=mass_matrix,
    )
    betas = tf.constant(tuple(float(value) for value in inverse_temperatures), tf.float64)
    initial_kinetic = _kinetic_energy(mass_cholesky, momentum)
    final_kinetic = _kinetic_energy(mass_cholesky, proposal["final_momentum"])
    log_accept_ratio = (
        betas[:, tf.newaxis]
        * (proposal["proposed_base_target_log_prob"] - current_target)
        + initial_kinetic
        - final_kinetic
    )
    log_accept_ratio = tf.where(
        proposal["path_valid"],
        log_accept_ratio,
        tf.fill((replicas, chains), tf.constant(float("-inf"), tf.float64)),
    )
    uniform = tf.random.stateless_uniform(
        (replicas, chains),
        seed=_fold_seed(master_seed, transition_index, 202),
        dtype=tf.float64,
    )
    accepted = tf.logical_and(
        proposal["path_valid"],
        tf.math.log(uniform)
        < tf.minimum(tf.constant(0.0, tf.float64), log_accept_ratio),
    )
    accepted_event = accepted[..., tf.newaxis]
    pre_swap_state = tf.where(accepted_event, proposal["proposed_state"], current)
    pre_swap_target = tf.where(
        accepted,
        proposal["proposed_base_target_log_prob"],
        current_target,
    )
    pre_swap_score = tf.where(
        accepted_event,
        proposal["proposed_base_score"],
        current_score,
    )
    swapped = apply_alternating_adjacent_swaps(
        pre_swap_state,
        pre_swap_target,
        pre_swap_score,
        identities_at_temperature,
        inverse_temperatures=inverse_temperatures,
        transition_index=transition_index,
        seed=_fold_seed(master_seed, transition_index, 303),
    )
    return {
        "state": swapped["state"],
        "base_target_log_prob": swapped["base_target_log_prob"],
        "base_score": swapped["base_score"],
        "identities_at_temperature": swapped["identities_at_temperature"],
        "pre_swap_state": pre_swap_state,
        "proposed_state": proposal["proposed_state"],
        "hmc_is_accepted": accepted,
        "hmc_log_accept_ratio": log_accept_ratio,
        "hmc_path_valid": proposal["path_valid"],
        "initial_momentum": proposal["initial_momentum"],
        "final_momentum": proposal["final_momentum"],
        "initial_kinetic_energy": initial_kinetic,
        "final_kinetic_energy": final_kinetic,
        "wave_valid_counts": proposal["wave_valid_counts"],
        "wave_metadata": proposal["wave_metadata"],
        "swap_is_proposed_adjacent": swapped["swap_is_proposed_adjacent"],
        "swap_is_accepted_adjacent": swapped["swap_is_accepted_adjacent"],
        "swap_log_accept_ratio_adjacent": swapped[
            "swap_log_accept_ratio_adjacent"
        ],
        "swap_is_accepted_matrix": swapped["swap_is_accepted_matrix"],
        "source_temperature_for_destination": swapped[
            "source_temperature_for_destination"
        ],
        "configuration": {
            "inverse_temperatures": tuple(float(value) for value in inverse_temperatures),
            "step_sizes": tuple(float(value) for value in step_sizes),
            "num_leapfrog_steps": int(num_leapfrog_steps),
            "transition_index": int(transition_index),
            "master_seed": tuple(int(value) for value in master_seed),
            "replicas": replicas,
            "chains": chains,
            "dimension": dimension,
            "mass_matrix": (
                None
                if mass_matrix is None
                else tuple(
                    tuple(float(item) for item in row)
                    for row in tf.convert_to_tensor(mass_matrix, tf.float64).numpy()
                )
            ),
        },
    }


def initialize_distributed_replica_state(
    state: Any,
    *,
    evaluator: BatchEvaluator,
    request_id: str = "initial-state",
) -> Mapping[str, Any]:
    """Evaluate and validate the initial state before any transition."""

    values = tf.convert_to_tensor(state, tf.float64)
    if values.shape.rank != 3 or not values.shape.is_fully_defined():
        raise ValueError("state must have static [temperature, chain, dimension] shape")
    replicas, chains, dimension = (int(item) for item in values.shape)
    target, score, status, metadata = evaluator(
        tf.reshape(values, (replicas * chains, dimension)), request_id
    )
    valid = _valid_evaluation(target, score, status)
    if not bool(tf.reduce_all(valid).numpy()):
        raise ValueError("initial distributed replica state is target-invalid")
    identities = tf.broadcast_to(
        tf.range(replicas, dtype=tf.int32)[:, tf.newaxis],
        (replicas, chains),
    )
    return {
        "state": values,
        "base_target_log_prob": tf.reshape(target, (replicas, chains)),
        "base_score": tf.reshape(score, (replicas, chains, dimension)),
        "identities_at_temperature": identities,
        "evaluation_metadata": metadata,
    }
