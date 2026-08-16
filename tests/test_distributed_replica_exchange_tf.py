"""Known-law tests for distributed TensorFlow replica-exchange mechanics."""

from __future__ import annotations

import math

import tensorflow as tf
import tensorflow_probability as tfp

from bayesfilter.testing.distributed_replica_exchange_tf import (
    apply_alternating_adjacent_swaps,
    distributed_replica_exchange_transition,
    initialize_distributed_replica_state,
    leapfrog_proposal,
)


DTYPE = tf.float64


def _gaussian_evaluator(rows: tf.Tensor, request_id: str):
    values = -0.5 * tf.reduce_sum(tf.square(rows), axis=1)
    status = {
        "status_code": tf.zeros(tf.shape(rows)[0], tf.int32),
        "valid_pre_regularized_score": tf.ones(tf.shape(rows)[0], tf.bool),
    }
    return values, -rows, status, {"request_id": request_id}


def test_one_step_harmonic_leapfrog_matches_closed_form() -> None:
    state = tf.constant([[[1.0]]], DTYPE)
    target = tf.constant([[-0.5]], DTYPE)
    score = tf.constant([[[-1.0]]], DTYPE)
    momentum = tf.constant([[[0.25]]], DTYPE)
    epsilon = 0.1
    result = leapfrog_proposal(
        state,
        target,
        score,
        momentum,
        inverse_temperatures=(1.0,),
        step_sizes=(epsilon,),
        num_leapfrog_steps=2,
        evaluator=_gaussian_evaluator,
        request_prefix="closed-form",
    )
    p_half_0 = 0.25 - 0.5 * epsilon
    q_1 = 1.0 + epsilon * p_half_0
    p_1 = p_half_0 - epsilon * q_1
    q_2 = q_1 + epsilon * p_1
    p_2 = p_1 - 0.5 * epsilon * q_2
    tf.debugging.assert_near(result["proposed_state"], [[[q_2]]])
    tf.debugging.assert_near(result["final_momentum"], [[[p_2]]])
    tf.debugging.assert_equal(result["path_valid"], [[True]])


def test_dense_mass_harmonic_leapfrog_matches_closed_form() -> None:
    state = tf.constant([[[1.0, -2.0]]], DTYPE)
    target = tf.constant([[-2.5]], DTYPE)
    score = tf.constant([[[-1.0, 2.0]]], DTYPE)
    momentum = tf.constant([[[0.5, -0.25]]], DTYPE)
    mass = tf.constant(((4.0, 0.0), (0.0, 0.25)), DTYPE)
    epsilon = 0.1
    result = leapfrog_proposal(
        state,
        target,
        score,
        momentum,
        inverse_temperatures=(1.0,),
        step_sizes=(epsilon,),
        num_leapfrog_steps=2,
        evaluator=_gaussian_evaluator,
        request_prefix="dense-mass-closed-form",
        mass_matrix=mass,
    )
    p_half_0 = momentum + 0.5 * epsilon * score
    q_1 = state + epsilon * tf.linalg.matvec(tf.linalg.inv(mass), p_half_0)
    p_1 = p_half_0 - epsilon * q_1
    q_2 = q_1 + epsilon * tf.linalg.matvec(tf.linalg.inv(mass), p_1)
    p_2 = p_1 - 0.5 * epsilon * q_2
    tf.debugging.assert_near(result["proposed_state"], q_2, atol=1.0e-12)
    tf.debugging.assert_near(result["final_momentum"], p_2, atol=1.0e-12)


def test_explicit_identity_mass_preserves_default_transition() -> None:
    state = tf.constant(
        [[[0.25, -0.5], [0.75, 0.125]], [[-0.25, 0.5], [-0.75, -0.125]]],
        DTYPE,
    )
    initial = initialize_distributed_replica_state(state, evaluator=_gaussian_evaluator)
    kwargs = dict(
        state=initial["state"],
        base_target_log_prob=initial["base_target_log_prob"],
        base_score=initial["base_score"],
        identities_at_temperature=initial["identities_at_temperature"],
        inverse_temperatures=(1.0, 0.5),
        step_sizes=(0.1, 0.2),
        num_leapfrog_steps=3,
        transition_index=7,
        master_seed=(20260811, 9101),
        evaluator=_gaussian_evaluator,
    )
    default = distributed_replica_exchange_transition(**kwargs)
    explicit = distributed_replica_exchange_transition(
        **kwargs, mass_matrix=tf.eye(2, dtype=DTYPE)
    )
    for key in (
        "state",
        "base_target_log_prob",
        "base_score",
        "hmc_is_accepted",
        "hmc_log_accept_ratio",
        "swap_is_accepted_matrix",
    ):
        tf.debugging.assert_equal(default[key], explicit[key])


def test_dense_mass_transition_uses_matching_momentum_and_kinetic_energy() -> None:
    state = tf.zeros((1, 2, 2), DTYPE)
    initial = initialize_distributed_replica_state(state, evaluator=_gaussian_evaluator)
    mass = tf.constant(((4.0, 1.0), (1.0, 2.0)), DTYPE)
    result = distributed_replica_exchange_transition(
        state=initial["state"],
        base_target_log_prob=initial["base_target_log_prob"],
        base_score=initial["base_score"],
        identities_at_temperature=initial["identities_at_temperature"],
        inverse_temperatures=(1.0,),
        step_sizes=(0.1,),
        num_leapfrog_steps=2,
        transition_index=3,
        master_seed=(20260811, 9201),
        evaluator=_gaussian_evaluator,
        mass_matrix=mass,
    )
    momentum = result["initial_momentum"]
    expected = 0.5 * tf.reduce_sum(
        momentum * tf.linalg.matvec(tf.linalg.inv(mass), momentum),
        axis=2,
    )
    tf.debugging.assert_near(result["initial_kinetic_energy"], expected, atol=1.0e-12)
    assert result["configuration"]["mass_matrix"] == (
        (4.0, 1.0),
        (1.0, 2.0),
    )


def test_dense_mass_rejects_nonsymmetric_or_non_spd_input() -> None:
    state = tf.zeros((1, 1, 2), DTYPE)
    target = tf.zeros((1, 1), DTYPE)
    score = tf.zeros_like(state)
    momentum = tf.zeros_like(state)
    for mass in (
        tf.constant(((1.0, 1.0), (0.0, 1.0)), DTYPE),
        tf.constant(((1.0, 0.0), (0.0, -1.0)), DTYPE),
    ):
        try:
            leapfrog_proposal(
                state,
                target,
                score,
                momentum,
                inverse_temperatures=(1.0,),
                step_sizes=(0.1,),
                num_leapfrog_steps=2,
                evaluator=_gaussian_evaluator,
                request_prefix="invalid-mass",
                mass_matrix=mass,
            )
        except (tf.errors.InvalidArgumentError, ValueError):
            pass
        else:
            raise AssertionError("invalid mass matrix was accepted")


def test_leapfrog_proposal_matches_tfp_for_tfp_generated_momentum() -> None:
    state = tf.constant([[[1.0, 2.0], [3.0, 4.0]]], DTYPE)

    def target(rows: tf.Tensor) -> tf.Tensor:
        return -0.5 * tf.reduce_sum(tf.square(rows), axis=1)

    flat_state = state[0]
    kernel = tfp.mcmc.UncalibratedHamiltonianMonteCarlo(
        target_log_prob_fn=target,
        step_size=tf.constant(0.1, DTYPE),
        num_leapfrog_steps=3,
    )
    tfp_state, tfp_result = kernel.one_step(
        flat_state,
        kernel.bootstrap_results(flat_state),
        seed=tf.constant((20260810, 6001), tf.int32),
    )
    initial_momentum = tfp_result.initial_momentum[0][tf.newaxis, ...]
    candidate = leapfrog_proposal(
        state,
        target(flat_state)[tf.newaxis, ...],
        (-flat_state)[tf.newaxis, ...],
        initial_momentum,
        inverse_temperatures=(1.0,),
        step_sizes=(0.1,),
        num_leapfrog_steps=3,
        evaluator=_gaussian_evaluator,
        request_prefix="tfp-parity",
    )
    tf.debugging.assert_near(
        candidate["proposed_state"][0], tfp_state, atol=1.0e-12
    )
    tf.debugging.assert_near(
        candidate["proposed_base_target_log_prob"][0],
        tfp_result.target_log_prob,
        atol=1.0e-12,
    )
    tf.debugging.assert_near(
        candidate["final_momentum"][0],
        tfp_result.final_momentum[0],
        atol=1.0e-12,
    )


def test_invalid_proposal_path_is_rejected_without_killing_other_row() -> None:
    def bounded(rows: tf.Tensor, request_id: str):
        values, score, status, metadata = _gaussian_evaluator(rows, request_id)
        valid = tf.abs(rows[:, 0]) < 2.0
        return values, score, {
            "status_code": tf.where(valid, 0, 1),
            "valid_pre_regularized_score": valid,
        }, metadata

    state = tf.constant([[[0.0], [1.9]], [[0.0], [1.9]]], DTYPE)
    initial = initialize_distributed_replica_state(state, evaluator=bounded)
    result = distributed_replica_exchange_transition(
        state=initial["state"],
        base_target_log_prob=initial["base_target_log_prob"],
        base_score=initial["base_score"],
        identities_at_temperature=initial["identities_at_temperature"],
        inverse_temperatures=(1.0, 0.5),
        step_sizes=(0.1, 5.0),
        num_leapfrog_steps=2,
        transition_index=0,
        master_seed=(20260810, 6101),
        evaluator=bounded,
    )
    invalid = tf.logical_not(result["hmc_path_valid"])
    assert bool(tf.reduce_any(invalid).numpy())
    tf.debugging.assert_equal(
        tf.boolean_mask(result["hmc_is_accepted"], invalid),
        tf.zeros(tf.reduce_sum(tf.cast(invalid, tf.int32)), tf.bool),
    )
    tf.debugging.assert_all_finite(result["state"], "retained state")


def test_adjacent_swap_uses_exact_power_tempering_ratio_and_permutation() -> None:
    state = tf.reshape(tf.range(6, dtype=DTYPE), (3, 2, 1))
    target = tf.constant(((0.0, 0.0), (100.0, 100.0), (0.0, 0.0)), DTYPE)
    score = tf.zeros_like(state)
    identities = tf.broadcast_to(tf.range(3, dtype=tf.int32)[:, None], (3, 2))
    result = apply_alternating_adjacent_swaps(
        state,
        target,
        score,
        identities,
        inverse_temperatures=(1.0, 0.5, 0.25),
        transition_index=0,
        seed=(20260810, 6201),
    )
    # Swapping target-100 states into beta=1 is certain under this construction.
    tf.debugging.assert_equal(result["swap_is_accepted_adjacent"][0], [True, True])
    tf.debugging.assert_equal(result["state"][0], state[1])
    tf.debugging.assert_equal(result["state"][1], state[0])
    tf.debugging.assert_equal(result["identities_at_temperature"][0], [1, 1])
    tf.debugging.assert_equal(result["identities_at_temperature"][1], [0, 0])
    matrix = tf.cast(result["swap_is_accepted_matrix"], tf.int32)
    tf.debugging.assert_equal(tf.reduce_sum(matrix, axis=0), tf.ones((3, 2), tf.int32))
    tf.debugging.assert_equal(tf.reduce_sum(matrix, axis=1), tf.ones((3, 2), tf.int32))


def test_distributed_transition_replays_and_preserves_hamiltonian_identity() -> None:
    state = tf.zeros((4, 2, 1), DTYPE)
    initial = initialize_distributed_replica_state(state, evaluator=_gaussian_evaluator)
    kwargs = dict(
        state=initial["state"],
        base_target_log_prob=initial["base_target_log_prob"],
        base_score=initial["base_score"],
        identities_at_temperature=initial["identities_at_temperature"],
        inverse_temperatures=(1.0, 0.5, 0.25, 0.125),
        step_sizes=(0.1, 0.1 / math.sqrt(0.5), 0.2, 0.1 / math.sqrt(0.125)),
        num_leapfrog_steps=3,
        transition_index=7,
        master_seed=(20260810, 6301),
        evaluator=_gaussian_evaluator,
    )
    first = distributed_replica_exchange_transition(**kwargs)
    replay = distributed_replica_exchange_transition(**kwargs)
    for name in (
        "state",
        "hmc_is_accepted",
        "hmc_log_accept_ratio",
        "swap_is_accepted_matrix",
    ):
        tf.debugging.assert_equal(first[name], replay[name])
    initial_kinetic = first["initial_kinetic_energy"]
    final_kinetic = first["final_kinetic_energy"]
    proposed_target = -0.5 * tf.reduce_sum(tf.square(first["proposed_state"]), axis=2)
    expected = (
        tf.constant((1.0, 0.5, 0.25, 0.125), DTYPE)[:, None]
        * (proposed_target - initial["base_target_log_prob"])
        + initial_kinetic
        - final_kinetic
    )
    tf.debugging.assert_near(first["hmc_log_accept_ratio"], expected, atol=1.0e-12)
