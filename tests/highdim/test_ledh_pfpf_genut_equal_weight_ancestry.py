from __future__ import annotations

import inspect
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import tensorflow as tf

from bayesfilter.highdim import ledh_pfpf_genut_initial_rqmc_tf as core


def _select(
    particles: tf.Tensor,
    weights: tf.Tensor,
    *,
    policy: str,
    uniforms: tf.Tensor | None = None,
    bits: int = 12,
) -> dict[str, tf.Tensor]:
    count = int(particles.shape[0])
    if uniforms is None:
        uniforms = tf.linspace(0.001, 0.999, count)
    return core._transition_ancestors(  # noqa: SLF001
        particles,
        weights,
        uniforms,
        ancestry_policy=policy,
        state_map_location=tf.zeros([particles.shape[1]], particles.dtype),
        state_map_scale=tf.ones([particles.shape[1]], particles.dtype),
        hilbert_bits=bits,
        state_map_policy="fixed_supplied",
    )


def test_equal_weight_policies_are_exact_same_n1008_permutation() -> None:
    count = 1008
    particles = tf.random.stateless_normal([count, 4], [810, 811])
    weights = tf.fill([count], tf.constant(1.0 / count, tf.float32))
    supplied = tf.sort(tf.random.stateless_uniform([count], [812, 813]))

    systematic = _select(
        particles,
        weights,
        policy="hilbert_systematic_equal_weight",
        uniforms=supplied,
    )
    permutation = _select(
        particles,
        weights,
        policy="hilbert_permutation_one_to_one",
        uniforms=supplied,
    )

    tf.debugging.assert_equal(
        systematic["selected_row_identities"],
        permutation["selected_row_identities"],
    )
    tf.debugging.assert_equal(
        systematic["flow_ancestors"], permutation["flow_ancestors"]
    )
    tf.debugging.assert_equal(
        tf.sort(systematic["selected_row_identities"]), tf.range(count)
    )
    assert int(systematic["ancestry_unique_count"].numpy()) == count
    assert bool(systematic["ancestry_permutation_valid"].numpy())
    assert bool(systematic["equal_weight_valid"].numpy())


def test_equal_weight_route_fails_closed_for_nonuniform_weights() -> None:
    particles = tf.constant(
        ((-1.2, 0.4), (0.5, 1.1), (1.6, -0.8), (-0.3, -1.5)),
        tf.float32,
    )
    weights = tf.constant((0.10, 0.20, 0.30, 0.40), tf.float32)
    for policy in (
        "hilbert_systematic_equal_weight",
        "hilbert_permutation_one_to_one",
    ):
        selected = _select(particles, weights, policy=policy)
        assert not bool(selected["equal_weight_valid"].numpy())
        assert not bool(selected["ancestry_permutation_valid"].numpy())
        assert float(selected["equal_weight_error"].numpy()) > 0.0


def test_unique_hilbert_keys_are_input_record_permutation_equivariant() -> None:
    particles = tf.constant(
        ((-3.0, -2.0), (-1.0, 2.5), (0.5, -0.5), (2.0, 1.5)), tf.float32
    )
    weights = tf.fill([4], 0.25)
    first = _select(
        particles, weights, policy="hilbert_permutation_one_to_one", bits=16
    )
    input_permutation = tf.constant((2, 0, 3, 1), tf.int32)
    second = _select(
        tf.gather(particles, input_permutation),
        tf.gather(weights, input_permutation),
        policy="hilbert_permutation_one_to_one",
        bits=16,
    )
    assert int(first["hilbert_ties"].numpy()) == 0
    assert int(second["hilbert_ties"].numpy()) == 0
    tf.debugging.assert_equal(first["flow_ancestors"], second["flow_ancestors"])


def test_hilbert_ties_are_reported_and_stably_inherit_input_order() -> None:
    particles = tf.constant(
        ((0.0, 0.0), (0.0, 0.0), (2.0, 2.0), (2.0, 2.0)), tf.float32
    )
    weights = tf.fill([4], 0.25)
    selected = _select(
        particles, weights, policy="hilbert_permutation_one_to_one", bits=2
    )
    assert int(selected["hilbert_ties"].numpy()) == 2
    tf.debugging.assert_equal(
        tf.sort(selected["selected_row_identities"]), tf.range(4)
    )
    assert bool(selected["ancestry_permutation_valid"].numpy())


def test_active_core_keeps_proposal_ancestry_out_of_all_parent_score_alignment() -> None:
    source = inspect.getsource(core)
    assert "parents = particles_value" in source
    assert "parent_marks = score_marks_value" in source
    assert "flow_ancestors = ancestor_records" in source
    assert "standard_pairwise_backward_marks(" in source
    assert 'ancestor_records["ancestry_permutation_valid"]' in source
    for forbidden in (
        "GradientTape",
        "ForwardAccumulator",
        "finite_difference",
        "ancestor_proposal_probability",
    ):
        assert forbidden not in source
