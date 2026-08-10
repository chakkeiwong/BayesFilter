from __future__ import annotations

from types import SimpleNamespace

import tensorflow as tf

from docs.benchmarks.run_genut_austria_neutra_zero_force_smoke import (
    _summarize_chain,
)


def test_nonfinite_proposal_is_excluded_from_energy_identity() -> None:
    transitions = 8
    chains = 4
    dimension = 3
    initial = tf.zeros((chains, dimension), tf.float64)
    positions = tf.stack(
        [
            tf.fill((chains, dimension), tf.cast(index + 1, tf.float64))
            for index in range(transitions)
        ]
    )
    positions = tf.tensor_scatter_nd_update(
        positions,
        indices=[[3, 0]],
        updates=positions[2, 0][tf.newaxis, :],
    )
    accepted = tf.ones((transitions, chains), tf.bool)
    accepted = tf.tensor_scatter_nd_update(accepted, [[3, 0]], [False])
    final_potential = tf.zeros((transitions, chains), tf.float64)
    final_potential = tf.tensor_scatter_nd_update(
        final_potential, [[3, 0]], [float("inf")]
    )
    delta_h = tf.identity(final_potential)
    chain = SimpleNamespace(
        accepted=accepted,
        positions=positions,
        delta_h=delta_h,
        initial_potential=tf.zeros_like(final_potential),
        final_potential=final_potential,
        initial_kinetic=tf.zeros_like(final_potential),
        final_kinetic=tf.zeros_like(final_potential),
        potentials=tf.zeros_like(final_potential),
        endpoint_call_count=tf.ones((transitions,), tf.int32),
        force_call_count=tf.fill((transitions,), 11),
    )

    summary = _summarize_chain(
        chain,
        initial,
        step_size=0.1,
        elapsed_seconds=1.0,
    )

    assert int(summary["nonfinite_proposed_endpoint_count"].numpy()) == 1
    assert int(summary["finite_energy_reconstruction_count"].numpy()) == 31
    assert float(summary["full_energy_identity_max_error"].numpy()) == 0.0
    assert bool(summary["viability_passed"].numpy()) is True
