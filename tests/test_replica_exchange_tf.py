"""Focused diagnostic tests for exact TFP replica-exchange mechanics."""

from __future__ import annotations

import math

import pytest
import tensorflow as tf
import tensorflow_probability as tfp

from bayesfilter.testing.replica_exchange_tf import (
    make_replica_exchange_fixed_hmc_sampler,
    replica_exchange_finite,
    replica_identities_at_temperatures,
    replica_travel_diagnostics,
    pre_swap_states_from_accepted_swaps,
    run_replica_exchange_fixed_hmc,
    validate_replica_exchange_configuration,
)


DTYPE = tf.float64
BETAS = (1.0, 0.3, 0.09, 0.027)
STEPS = tuple(0.25 / math.sqrt(beta) for beta in BETAS)
MEANS = tf.constant((-4.0, 4.0), DTYPE)
SCALES = tf.constant((0.5, 0.5), DTYPE)


def _mixture_target(weights: tuple[float, float]):
    log_weights = tf.math.log(tf.constant(weights, DTYPE))
    log_normalizer = tf.math.log(SCALES) + 0.5 * tf.math.log(
        tf.constant(2.0 * math.pi, DTYPE)
    )

    def target(state: tf.Tensor) -> tf.Tensor:
        standardized = (state[..., 0, tf.newaxis] - MEANS) / SCALES
        component_log_prob = -0.5 * tf.square(standardized) - log_normalizer
        return tf.reduce_logsumexp(log_weights + component_log_prob, axis=-1)

    return target


def _one_sided_initial_state(chain_count: int = 8) -> tf.Tensor:
    positive = tf.linspace(
        tf.constant(3.5, DTYPE), tf.constant(4.5, DTYPE), chain_count
    )
    return tf.repeat(positive[tf.newaxis, :, tf.newaxis], len(BETAS), axis=0)


def _swap_matrix(permutation: tuple[int, ...]) -> tf.Tensor:
    replica_count = len(permutation)
    matrix = [
        [old == permutation[new] for new in range(replica_count)]
        for old in range(replica_count)
    ]
    return tf.constant(matrix, tf.bool)[:, :, tf.newaxis]


def test_configuration_rejects_improper_or_untuned_inputs() -> None:
    with pytest.raises(ValueError, match="strictly decreasing"):
        validate_replica_exchange_configuration(
            (1.0, 0.5, 0.5), (0.1, 0.2, 0.3),
            num_leapfrog_steps=3, num_steps=4,
        )
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        validate_replica_exchange_configuration(
            (1.0, 0.5, 0.0), (0.1, 0.2, 0.3),
            num_leapfrog_steps=3, num_steps=4,
        )
    with pytest.raises(ValueError, match="one step size"):
        validate_replica_exchange_configuration(
            (1.0, 0.5), (0.1,), num_leapfrog_steps=3, num_steps=4
        )
    with pytest.raises(ValueError, match="at least two leapfrog"):
        validate_replica_exchange_configuration(
            (1.0, 0.5), (0.1, 0.2), num_leapfrog_steps=1, num_steps=4
        )


def test_identity_reconstruction_and_round_trip_state_machine() -> None:
    # Initial identity 0 travels cold -> middle -> hot -> middle -> cold.
    accepted = tf.stack(
        (
            _swap_matrix((1, 0, 2)),
            _swap_matrix((0, 2, 1)),
            _swap_matrix((1, 0, 2)),
            _swap_matrix((0, 2, 1)),
            _swap_matrix((1, 0, 2)),
        ),
        axis=0,
    )
    identities = replica_identities_at_temperatures(accepted)
    expected = tf.constant(
        (
            ((1,), (0,), (2,)),
            ((1,), (2,), (0,)),
            ((2,), (1,), (0,)),
            ((2,), (0,), (1,)),
            ((0,), (2,), (1,)),
        ),
        tf.int32,
    )
    tf.debugging.assert_equal(identities, expected)
    travel = replica_travel_diagnostics(identities)
    assert int(travel["round_trip_returns"][0, 0].numpy()) == 1
    assert bool(travel["visited_hot"][0, 0].numpy())


def test_pre_swap_state_reconstruction_inverts_accepted_permutation() -> None:
    accepted = tf.stack((_swap_matrix((1, 0, 2)),), axis=0)
    pre = tf.constant(((((10.0,),), ((20.0,),), ((30.0,),)),), DTYPE)
    post = tf.constant(((((20.0,),), ((10.0,),), ((30.0,),)),), DTYPE)
    recovered = pre_swap_states_from_accepted_swaps(post, accepted)
    tf.debugging.assert_equal(recovered, pre)


def test_plain_hmc_remains_in_one_mode_on_separated_fixture() -> None:
    target = _mixture_target((0.5, 0.5))
    initial = _one_sided_initial_state()[0]
    kernel = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=target,
        step_size=tf.constant(0.25, DTYPE),
        num_leapfrog_steps=4,
    )

    @tf.function(jit_compile=True)
    def sample() -> tf.Tensor:
        return tfp.mcmc.sample_chain(
            num_results=400,
            num_burnin_steps=100,
            current_state=initial,
            kernel=kernel,
            trace_fn=None,
            seed=tf.constant((20260810, 1001), tf.int32),
        )

    states = sample()
    assert bool(tf.reduce_all(states[..., 0] > 0.0).numpy())


def test_reusable_xla_sampler_continues_state_without_retracing() -> None:
    initial = _one_sided_initial_state(chain_count=2)
    sampler = make_replica_exchange_fixed_hmc_sampler(
        _mixture_target((0.5, 0.5)),
        initial,
        inverse_temperatures=BETAS,
        step_sizes=STEPS,
        num_leapfrog_steps=2,
        num_steps=2,
        jit_compile=True,
    )
    first = sampler(initial, tf.constant((20260810, 1051), tf.int32))
    continued = sampler(
        first["replica_states"][-1],
        tf.constant((20260810, 1052), tf.int32),
    )
    assert sampler.experimental_get_tracing_count() == 1
    assert first["replica_states"].shape == (2, 4, 2, 1)
    assert continued["replica_states"].shape == (2, 4, 2, 1)
    assert bool(replica_exchange_finite(first).numpy())
    assert bool(replica_exchange_finite(continued).numpy())


@pytest.mark.parametrize(
    ("weights", "seed", "lower", "upper"),
    (
        ((0.5, 0.5), (20260810, 1101), 0.40, 0.60),
        ((0.8, 0.2), (20260810, 1201), 0.68, 0.90),
    ),
)
def test_xla_replica_exchange_recovers_known_mixture_from_one_sided_starts(
    weights: tuple[float, float],
    seed: tuple[int, int],
    lower: float,
    upper: float,
) -> None:
    trace = run_replica_exchange_fixed_hmc(
        _mixture_target(weights),
        _one_sided_initial_state(),
        inverse_temperatures=BETAS,
        step_sizes=STEPS,
        num_leapfrog_steps=4,
        num_steps=1000,
        seed=seed,
        jit_compile=True,
    )
    cold = trace["cold_states"][200:, ..., 0]
    negative = cold < 0.0
    fraction = float(tf.reduce_mean(tf.cast(negative, DTYPE)).numpy())
    transitions = int(
        tf.reduce_sum(tf.cast(negative[1:] != negative[:-1], tf.int32)).numpy()
    )
    assert bool(replica_exchange_finite(trace).numpy())
    assert trace["pre_swap_replica_states"].shape == trace["replica_states"].shape
    assert lower <= fraction <= upper
    assert transitions > 100
    assert int(tf.reduce_sum(trace["round_trip_returns"]).numpy()) > 0
    proposed = tf.reduce_sum(
        tf.cast(trace["swap_is_proposed_adjacent"][200:], tf.int32), axis=(0, 2)
    )
    accepted = tf.reduce_sum(
        tf.cast(trace["swap_is_accepted_adjacent"][200:], tf.int32), axis=(0, 2)
    )
    assert bool(tf.reduce_all(proposed > 0).numpy())
    assert bool(tf.reduce_all(accepted > 0).numpy())
