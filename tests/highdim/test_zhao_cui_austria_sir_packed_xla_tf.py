from __future__ import annotations

from pathlib import Path

import tensorflow as tf

from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_artifact_compat import (
    load_lane_b_t1_artifact_v1_compat,
)
from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_tf import lane_b_product_basis
from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_training_jvp_tf import (
    ADAM_BETA_1,
    ADAM_BETA_2,
    ADAM_EPSILON,
    ROOT,
    _evaluate_cores,
    _square_mass,
    functional_adam_apply_gradients,
)
from bayesfilter.highdim.zhao_cui_austria_sir_packed_xla_tf import (
    MATERIAL_REPLAY_ATOL,
    MATERIAL_REPLAY_RTOL,
    material_replay_metrics,
    pack_cores,
    packed_adam_apply_gradients,
    packed_amplitude,
    packed_core_mask,
    packed_normalized_density,
    packed_normalized_prefix_density,
    packed_per_core_regularizers,
    packed_square_mass,
    packed_tuple_global_norm,
    precompute_basis_values,
    precompute_mass_matrices,
)


DTYPE = tf.float64
PARENT_DIR = ROOT / (
    "docs/plans/artifacts/zhao-cui-austria-sir-lane-b-t1-20260730/"
    "pilot-final-02/p05_r4_b5_lr3e4_l1_1e9/artifact"
)


def _parent():
    return load_lane_b_t1_artifact_v1_compat(Path(PARENT_DIR))


def test_packed_amplitude_and_mass_match_tuple_authority_under_xla() -> None:
    parent = _parent()
    basis = lane_b_product_basis(
        order=parent.settings.basis_order,
        num_elems=parent.settings.basis_num_elems,
    )
    points = tf.random.stateless_uniform(
        [7, 36], [82001, 1], minval=-0.8, maxval=0.8, dtype=DTYPE
    )
    packed = pack_cores(parent.cores)
    tables = precompute_basis_values(basis, points)
    masses = precompute_mass_matrices(basis)

    @tf.function(jit_compile=True)
    def compiled(active, active_tables, active_masses):
        return (
            packed_amplitude(active, active_tables),
            packed_square_mass(active, active_masses),
        )

    amplitude, mass = compiled(packed, tables, masses)
    tf.debugging.assert_near(
        amplitude, _evaluate_cores(parent.cores, basis, points), atol=2e-12, rtol=2e-12
    )
    tf.debugging.assert_near(
        mass, _square_mass(parent.cores, basis), atol=2e-12, rtol=2e-12
    )


def test_packed_adam_matches_tuple_authority_and_masks_padding() -> None:
    parent = _parent()
    packed = pack_cores(parent.cores)
    mask = packed_core_mask(tuple(core.shape for core in parent.cores))
    gradients = tf.random.stateless_normal(packed.shape, [82002, 1], dtype=DTYPE)
    momentums = tf.random.stateless_normal(packed.shape, [82002, 2], dtype=DTYPE)
    velocities = tf.square(
        tf.random.stateless_normal(packed.shape, [82002, 3], dtype=DTYPE)
    )
    gradients *= mask
    momentums *= mask
    velocities *= mask
    tuple_gradients = tuple(
        gradients[axis, : core.shape[0], :, : core.shape[2]]
        for axis, core in enumerate(parent.cores)
    )
    tuple_momentums = tuple(
        momentums[axis, : core.shape[0], :, : core.shape[2]]
        for axis, core in enumerate(parent.cores)
    )
    tuple_velocities = tuple(
        velocities[axis, : core.shape[0], :, : core.shape[2]]
        for axis, core in enumerate(parent.cores)
    )
    expected = functional_adam_apply_gradients(
        parent.cores,
        tuple_momentums,
        tuple_velocities,
        tuple_gradients,
        step=tf.constant(7, tf.int32),
        learning_rate=tf.cast(tf.constant(3e-4, tf.float32), DTYPE),
        gradient_clip_norm=tf.constant(100.0, DTYPE),
    )

    @tf.function(jit_compile=True)
    def compiled():
        return packed_adam_apply_gradients(
            packed,
            momentums,
            velocities,
            gradients,
            mask,
            step=tf.constant(7, tf.int32),
            learning_rate=tf.cast(tf.constant(3e-4, tf.float32), DTYPE),
            gradient_clip_norm=tf.constant(100.0, DTYPE),
            beta_1=ADAM_BETA_1,
            beta_2=ADAM_BETA_2,
            epsilon=ADAM_EPSILON,
        )

    observed = compiled()
    for packed_value, expected_values in zip(observed, expected):
        for axis, core in enumerate(parent.cores):
            tf.debugging.assert_near(
                packed_value[axis, : core.shape[0], :, : core.shape[2]],
                expected_values[axis],
                atol=5e-15,
                rtol=5e-15,
            )
        tf.debugging.assert_equal(packed_value * (1.0 - mask), tf.zeros_like(packed))


def test_packed_reductions_match_true_shape_tuple_authority_under_xla() -> None:
    parent = _parent()
    mask = packed_core_mask(tuple(core.shape for core in parent.cores))
    packed = pack_cores(parent.cores)
    gradients = (
        tf.random.stateless_normal(packed.shape, [82004, 1], dtype=DTYPE) * mask
    )
    expected_l1 = tf.add_n(
        [tf.reduce_sum(tf.abs(core)) for core in parent.cores]
    )
    expected_l2 = tf.add_n(
        [tf.reduce_sum(tf.square(core)) for core in parent.cores]
    )
    tuple_gradients = tuple(
        gradients[axis, : core.shape[0], :, : core.shape[2]]
        for axis, core in enumerate(parent.cores)
    )
    expected_norm = tf.linalg.global_norm(tuple_gradients)

    @tf.function(jit_compile=True)
    def compiled():
        return (*packed_per_core_regularizers(packed, mask), packed_tuple_global_norm(gradients, mask))

    observed_l1, observed_l2, observed_norm = compiled()
    tf.debugging.assert_near(observed_l1, expected_l1, atol=5e-15, rtol=5e-15)
    tf.debugging.assert_near(observed_l2, expected_l2, atol=5e-15, rtol=5e-15)
    tf.debugging.assert_near(observed_norm, expected_norm, atol=5e-15, rtol=5e-15)


def test_packed_adam_matches_tuple_authority_when_clipping_is_active() -> None:
    parent = _parent()
    packed = pack_cores(parent.cores)
    mask = packed_core_mask(tuple(core.shape for core in parent.cores))
    gradients = (
        tf.constant(20.0, DTYPE)
        * tf.random.stateless_normal(packed.shape, [82005, 1], dtype=DTYPE)
        * mask
    )
    momentums = tf.zeros_like(packed)
    velocities = tf.zeros_like(packed)
    tuple_gradients = tuple(
        gradients[axis, : core.shape[0], :, : core.shape[2]]
        for axis, core in enumerate(parent.cores)
    )
    expected = functional_adam_apply_gradients(
        parent.cores,
        tuple(tf.zeros_like(core) for core in parent.cores),
        tuple(tf.zeros_like(core) for core in parent.cores),
        tuple_gradients,
        step=tf.constant(1, tf.int32),
        learning_rate=tf.cast(tf.constant(3e-4, tf.float32), DTYPE),
        gradient_clip_norm=tf.constant(1.0, DTYPE),
    )

    @tf.function(jit_compile=True)
    def compiled():
        return packed_adam_apply_gradients(
            packed,
            momentums,
            velocities,
            gradients,
            mask,
            step=tf.constant(1, tf.int32),
            learning_rate=tf.cast(tf.constant(3e-4, tf.float32), DTYPE),
            gradient_clip_norm=tf.constant(1.0, DTYPE),
            beta_1=ADAM_BETA_1,
            beta_2=ADAM_BETA_2,
            epsilon=ADAM_EPSILON,
        )

    observed = compiled()
    for packed_value, expected_values in zip(observed, expected):
        for axis, core in enumerate(parent.cores):
            tf.debugging.assert_near(
                packed_value[axis, : core.shape[0], :, : core.shape[2]],
                expected_values[axis],
                atol=5e-15,
                rtol=5e-15,
            )


def test_packed_nested_contraction_reverse_gradient_compiles_with_xla() -> None:
    parent = _parent()
    basis = lane_b_product_basis(
        order=parent.settings.basis_order,
        num_elems=parent.settings.basis_num_elems,
    )
    points = tf.zeros([2, 36], DTYPE)
    tables = precompute_basis_values(basis, points)
    masses = precompute_mass_matrices(basis)
    packed = pack_cores(parent.cores)

    @tf.function(jit_compile=True)
    def compiled(active):
        with tf.GradientTape() as tape:
            tape.watch(active)
            amplitude = packed_amplitude(active, tables)
            loss = tf.reduce_sum(tf.square(amplitude)) + packed_square_mass(
                active, masses
            )
        return tape.gradient(loss, active)

    gradient = compiled(packed)
    tf.debugging.assert_all_finite(gradient, "packed reverse gradient")


def test_packed_full_and_prefix_density_match_tuple_authority() -> None:
    parent = _parent()
    basis = lane_b_product_basis(
        order=parent.settings.basis_order,
        num_elems=parent.settings.basis_num_elems,
    )
    points = tf.random.stateless_uniform(
        [5, 36], [82003, 1], minval=-0.8, maxval=0.8, dtype=DTYPE
    )
    tables = precompute_basis_values(basis, points)
    masses = precompute_mass_matrices(basis)
    packed = pack_cores(parent.cores)
    tau = tf.constant(parent.settings.tau, DTYPE)

    @tf.function(jit_compile=True)
    def compiled():
        return (
            packed_normalized_density(packed, tables, masses, tau),
            packed_normalized_prefix_density(packed, tables, masses, tau),
        )

    full, prefix = compiled()
    authority = parent.density()
    tf.debugging.assert_near(
        full,
        authority.normalized_retained_density_values(tuple(range(36)), points),
        atol=3e-12,
        rtol=3e-12,
    )
    tf.debugging.assert_near(
        prefix,
        authority.normalized_marginal_density_values(tuple(range(18)), points[:, :18]),
        atol=3e-12,
        rtol=3e-12,
    )


def test_material_replay_gate_has_mixed_five_digit_boundary() -> None:
    parent = _parent()
    packed = pack_cores(parent.cores)
    mask = packed_core_mask(tuple(core.shape for core in parent.cores))
    threshold = MATERIAL_REPLAY_ATOL + MATERIAL_REPLAY_RTOL * tf.abs(packed)
    passed, _absolute, normalized = material_replay_metrics(
        packed + tf.constant(0.99, DTYPE) * threshold * mask,
        packed,
        mask,
    )
    assert bool(passed)
    tf.debugging.assert_less_equal(normalized, tf.constant(1.0, DTYPE))
    failed, _absolute, normalized = material_replay_metrics(
        packed + tf.constant(1.01, DTYPE) * threshold * mask,
        packed,
        mask,
    )
    assert not bool(failed)
    tf.debugging.assert_greater(normalized, tf.constant(1.0, DTYPE))
