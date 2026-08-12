from __future__ import annotations

import tensorflow as tf
import pytest

import bayesfilter.highdim as highdim
from bayesfilter.highdim.zhao_cui_moment_teacher_als import (
    fixed_als_value_jvp,
    fixed_tt_teacher_recursion_jvp,
)
from bayesfilter.highdim.zhao_cui_moment_teacher_xla import (
    PaddedALSSetup,
    pad_tt_cores,
    padded_fixed_als_value_jvp_xla,
    padded_fixed_teacher_recursion_marginal_xla,
    padded_fixed_teacher_recursion_shape_xla,
    padded_squared_tt_normalized_marginal_jvp_xla,
    padded_squared_tt_shape_targets_jvp_xla,
)
from bayesfilter.highdim.zhao_cui_moment_teacher import (
    legendre_monomial_operator_matrix,
    squared_tt_normalized_marginal_jvp,
    squared_tt_shape_targets_jvp,
    tensor_product_reference_monomial_moment,
)


def _convention():
    return highdim.MeasureConvention(
        density_measure=highdim.DensityMeasure.REFERENCE_MEASURE,
        mass_measure=highdim.MassMeasure.REFERENCE_MEASURE,
        reference_weight_name="omega",
    )


def _problem():
    convention = _convention()
    product = highdim.ProductBasis(
        [
            highdim.LegendreBasis1D(highdim.BoundedInterval(-1.0, 1.0), 2),
            highdim.LegendreBasis1D(highdim.BoundedInterval(-1.0, 1.0), 2),
        ],
        convention,
    )
    points = tf.random.stateless_uniform([32, 2], [101, 102], -0.9, 0.9, dtype=tf.float64)
    basis_values = tf.stack(
        [product.evaluate_axis(axis, points[:, axis]) for axis in range(2)]
    )
    target = tf.exp(-0.4 * tf.reduce_sum(tf.square(points - [0.2, -0.1]), axis=1))
    dot_target = 0.2 * target * (points[:, 0] - 0.3)
    weights = tf.ones([32], tf.float64) / 32.0
    dot_weights = tf.zeros_like(weights)
    cores = (
        highdim.TTCore(
            tf.constant([[[1.0, 0.1], [0.0, 0.3], [0.2, -0.1]]], tf.float64)
        ),
        highdim.TTCore(
            tf.constant(
                [
                    [[0.9], [0.2], [-0.1]],
                    [[0.1], [0.7], [0.2]],
                ],
                tf.float64,
            )
        ),
    )
    dot_cores = tuple(
        highdim.TTCore(tf.fill(tf.shape(core.values), tf.constant(0.01, tf.float64)))
        for core in cores
    )
    mask = tf.constant(
        [
            [[[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]], [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]],
            [[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]],
        ],
        tf.float64,
    )
    initial = pad_tt_cores(
        tf.stack(
            [
                tf.pad(cores[0].values, [[0, 1], [0, 0], [0, 0]]),
                tf.pad(cores[1].values, [[0, 0], [0, 0], [0, 1]]),
            ]
        ),
        mask,
        padded_rank=2,
        padded_basis=3,
    )
    initial_dot = pad_tt_cores(
        tf.stack(
            [
                tf.pad(dot_cores[0].values, [[0, 1], [0, 0], [0, 0]]),
                tf.pad(dot_cores[1].values, [[0, 0], [0, 0], [0, 1]]),
            ]
        ),
        mask,
        padded_rank=2,
        padded_basis=3,
    )
    return (
        product,
        points,
        basis_values,
        mask,
        target,
        dot_target,
        weights,
        dot_weights,
        cores,
        dot_cores,
        initial,
        initial_dot,
        convention,
    )


def _config():
    return highdim.FixedTTFitConfig(
        ranks=(1, 2, 1),
        ridge=1e-8,
        max_sweeps=2,
        sweep_order=(0, 1),
        row_budget=64,
        column_budget=16,
        dense_matrix_byte_budget=100_000,
        normal_matrix_byte_budget=20_000,
        condition_number_warning=1e10,
        condition_number_veto=1e14,
        holdout_tolerance=1e6,
    )


def test_padded_xla_value_and_jvp_match_reference():
    (
        product,
        points,
        basis_values,
        mask,
        target,
        dot_target,
        weights,
        dot_weights,
        cores,
        dot_cores,
        initial,
        initial_dot,
        convention,
    ) = _problem()
    config = _config()
    reference = fixed_als_value_jvp(
        product, points, target, weights, dot_target, config, cores, dot_cores,
        convention, dot_weights=dot_weights,
    )
    graph_cores, graph_dot_cores, diagnostics, valid = padded_fixed_als_value_jvp_xla(
        basis_values,
        mask,
        tf.constant([0, 1, 0, 1], tf.int32),
        target,
        dot_target,
        weights,
        dot_weights,
        initial,
        initial_dot,
        tf.constant(config.ridge, tf.float64),
        tf.constant(config.column_scale_floor, tf.float64),
        tf.constant(config.condition_number_veto, tf.float64),
        tf.constant(1e-7, tf.float64),
    )
    expected_cores = tf.stack(
        [
            tf.pad(reference.cores[0].values, [[0, 1], [0, 0], [0, 0]]),
            tf.pad(reference.cores[1].values, [[0, 0], [0, 0], [0, 1]]),
        ]
    ) * mask
    expected_dot_cores = tf.stack(
        [
            tf.pad(reference.dot_cores[0].values, [[0, 1], [0, 0], [0, 0]]),
            tf.pad(reference.dot_cores[1].values, [[0, 0], [0, 0], [0, 1]]),
        ]
    ) * mask
    tf.debugging.assert_near(graph_cores, expected_cores, atol=2e-6, rtol=2e-6)
    tf.debugging.assert_near(graph_dot_cores, expected_dot_cores, atol=3e-5, rtol=3e-5)
    assert bool(valid.numpy())
    assert diagnostics.shape == (4, 8)


def test_padded_setup_rejects_zero_ridge_and_bad_schedule():
    with tf.device("/CPU:0"):
        basis = tf.zeros([2, 4, 3], tf.float64)
        mask = tf.ones([2, 2, 3, 2], tf.float64)
    with pytest.raises(ValueError, match="strictly positive ridge"):
        PaddedALSSetup(basis, mask, tf.constant([0], tf.int32), 0.0)
    with pytest.raises(ValueError, match="schedule axis is out of range"):
        PaddedALSSetup(basis, mask, tf.constant([2], tf.int32), 1e-8)


def test_padded_xla_jvp_matches_centered_finite_difference():
    (
        _,
        _,
        basis_values,
        mask,
        target,
        dot_target,
        weights,
        dot_weights,
        _,
        _,
        initial,
        initial_dot,
        _,
    ) = _problem()
    config = _config()
    schedule = tf.constant([0, 1, 0, 1], tf.int32)

    def run(active_target, active_cores, target_tangent, core_tangent):
        return padded_fixed_als_value_jvp_xla(
            basis_values,
            mask,
            schedule,
            active_target,
            target_tangent,
            weights,
            dot_weights,
            active_cores,
            core_tangent,
            tf.constant(config.ridge, tf.float64),
            tf.constant(config.column_scale_floor, tf.float64),
            tf.constant(config.condition_number_veto, tf.float64),
            tf.constant(1e-7, tf.float64),
        )

    _, tangent, _, valid = run(target, initial, dot_target, initial_dot)
    h = tf.constant(1e-5, tf.float64)
    plus, _, _, plus_valid = run(
        target + h * dot_target,
        initial + h * initial_dot,
        tf.zeros_like(dot_target),
        tf.zeros_like(initial_dot),
    )
    minus, _, _, minus_valid = run(
        target - h * dot_target,
        initial - h * initial_dot,
        tf.zeros_like(dot_target),
        tf.zeros_like(initial_dot),
    )
    assert bool((valid & plus_valid & minus_valid).numpy())
    tf.debugging.assert_near(tangent, (plus - minus) / (2.0 * h), atol=4e-5, rtol=4e-5)


def test_padded_xla_graph_has_control_flow_and_no_host_callbacks():
    (
        _,
        _,
        basis_values,
        mask,
        target,
        dot_target,
        weights,
        dot_weights,
        _,
        _,
        initial,
        initial_dot,
        _,
    ) = _problem()
    config = _config()
    concrete = padded_fixed_als_value_jvp_xla.get_concrete_function(
        basis_values,
        mask,
        tf.constant([0, 1, 0, 1], tf.int32),
        target,
        dot_target,
        weights,
        dot_weights,
        initial,
        initial_dot,
        tf.constant(config.ridge, tf.float64),
        tf.constant(config.column_scale_floor, tf.float64),
        tf.constant(config.condition_number_veto, tf.float64),
        tf.constant(1e-7, tf.float64),
    )
    graph_def = concrete.graph.as_graph_def()
    op_types = {node.op for node in graph_def.node}
    for function in graph_def.library.function:
        op_types.update(node.op for node in function.node_def)
    assert op_types.isdisjoint({"PyFunc", "EagerPyFunc"})
    assert op_types.intersection({"While", "StatelessWhile"})


def test_padded_xla_failed_gate_returns_nonfinite_cores():
    (
        _,
        _,
        basis_values,
        mask,
        target,
        dot_target,
        weights,
        dot_weights,
        _,
        _,
        initial,
        initial_dot,
        _,
    ) = _problem()
    cores, dot_cores, _, valid = padded_fixed_als_value_jvp_xla(
        basis_values,
        mask,
        tf.constant([0], tf.int32),
        target,
        dot_target,
        weights,
        dot_weights,
        initial,
        initial_dot,
        tf.constant(1e-8, tf.float64),
        tf.constant(2.220446049250313e-16, tf.float64),
        tf.constant(1.0, tf.float64),
        tf.constant(1e-7, tf.float64),
    )
    assert not bool(valid.numpy())
    assert not bool(tf.reduce_any(tf.math.is_finite(cores)).numpy())
    assert not bool(tf.reduce_any(tf.math.is_finite(dot_cores)).numpy())


def test_padded_xla_normalized_marginal_matches_reference():
    product = highdim.ProductBasis(
        [
            highdim.LegendreBasis1D(highdim.BoundedInterval(-1.0, 1.0), 2),
            highdim.LegendreBasis1D(highdim.BoundedInterval(-1.0, 1.0), 2),
        ],
        _convention(),
    )
    sqrt_tt = highdim.FunctionalTT(
        (
            highdim.TTCore(
                tf.constant([[[1.0, 0.1], [0.0, 0.3], [0.2, -0.1]]], tf.float64)
            ),
            highdim.TTCore(
                tf.constant(
                    [[[0.9], [0.2], [-0.1]], [[0.1], [0.7], [0.2]]], tf.float64
                )
            ),
        ),
        product,
        _convention(),
    )
    defensive = highdim.TensorProductReferenceDensity(product, _convention())
    identity = highdim.SquaredTTDensity.expected_branch_identity(
        sqrt_tt=sqrt_tt,
        defensive_density=defensive,
        tau=tf.constant(0.2, tf.float64),
        normalizer_floor=tf.constant(1e-12, tf.float64),
        denominator_floor=tf.constant(1e-12, tf.float64),
        measure_convention=_convention(),
    )
    density = highdim.SquaredTTDensity(
        sqrt_tt=sqrt_tt,
        defensive_density=defensive,
        tau=tf.constant(0.2, tf.float64),
        normalizer_floor=tf.constant(1e-12, tf.float64),
        denominator_floor=tf.constant(1e-12, tf.float64),
        measure_convention=_convention(),
        branch_identity=identity,
    )
    dot_cores = tuple(
        highdim.TTCore(tf.fill(tf.shape(core.values), tf.constant(0.01, tf.float64)))
        for core in density.sqrt_tt.cores
    )
    points = tf.constant([[-0.7], [0.1], [0.55]], tf.float64)
    reference = squared_tt_normalized_marginal_jvp(
        density, (0,), points, dot_cores, dot_tau=tf.constant(-0.03, tf.float64)
    )
    mask = tf.constant(
        [
            [[[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]], [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]],
            [[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]],
        ],
        tf.float64,
    )
    cores = tf.stack(
        [
            tf.pad(density.sqrt_tt.cores[0].values, [[0, 1], [0, 0], [0, 0]]),
            tf.pad(density.sqrt_tt.cores[1].values, [[0, 0], [0, 0], [0, 1]]),
        ]
    ) * mask
    dots = tf.stack(
        [
            tf.pad(dot_cores[0].values, [[0, 1], [0, 0], [0, 0]]),
            tf.pad(dot_cores[1].values, [[0, 0], [0, 0], [0, 1]]),
        ]
    ) * mask
    query_basis = tf.stack(
        [
            density.sqrt_tt.product_basis.evaluate_axis(0, points[:, 0]),
            tf.ones([3, 3], tf.float64),
        ]
    )
    mass = tf.stack([tf.eye(3, dtype=tf.float64), tf.eye(3, dtype=tf.float64)])
    values, tangent, normalizer, normalizer_tangent = padded_squared_tt_normalized_marginal_jvp_xla(
        cores,
        dots,
        query_basis,
        tf.constant([True, False]),
        mass,
        density.tau,
        tf.constant(-0.03, tf.float64),
        tf.ones([3], tf.float64),
        tf.zeros([3], tf.float64),
        tf.constant(1.0, tf.float64),
        tf.constant(0.0, tf.float64),
    )
    tf.debugging.assert_near(values, reference.values, atol=2e-10, rtol=2e-10)
    tf.debugging.assert_near(tangent, reference.tangent, atol=2e-10, rtol=2e-10)
    tf.debugging.assert_near(normalizer, reference.normalizer, atol=2e-10, rtol=2e-10)
    tf.debugging.assert_near(
        normalizer_tangent, reference.normalizer_tangent, atol=2e-10, rtol=2e-10
    )


def test_padded_xla_two_step_recursion_matches_reference_marginal():
    (
        product,
        points,
        basis_values,
        mask,
        _,
        _,
        weights,
        dot_weights,
        cores,
        _,
        initial,
        _,
        convention,
    ) = _problem()
    config = _config()
    direction = tf.constant(0.35, tf.float64)
    theta = tf.constant(0.1, tf.float64)
    first_base = -0.4 * tf.reduce_sum(
        tf.square(points - [0.2, -0.1]), axis=1
    ) + theta * points[:, 0]
    second_base = -0.2 * tf.square(points[:, 1] - points[:, 0]) + theta * points[:, 1]
    base_targets = tf.stack([first_base, second_base])
    dot_base_targets = tf.stack(
        [direction * points[:, 0], direction * points[:, 1]]
    )
    zero_dot_cores = tuple(
        highdim.TTCore(tf.zeros_like(core.values)) for core in cores
    )
    reference = fixed_tt_teacher_recursion_jvp(
        product,
        points,
        base_targets,
        dot_base_targets,
        weights,
        config,
        cores,
        zero_dot_cores,
        convention,
        carried_keep_axes=(0,),
        previous_query_points=points[:, :1],
        state_offset=tf.zeros([2], tf.float64),
        state_matrix=tf.eye(2, dtype=tf.float64),
        unscaled_defensive_weight=tf.constant(0.05, tf.float64),
    )
    shift_indices = tf.constant(
        [step.square_root_target.scale_shift_index for step in reference.steps],
        tf.int32,
    )
    initial_dot = tf.zeros_like(initial)
    query_basis = tf.stack(
        [
            product.evaluate_axis(0, points[:, 0]),
            tf.ones_like(product.evaluate_axis(1, points[:, 1])),
        ]
    )
    mass = tf.stack([tf.eye(3, dtype=tf.float64), tf.eye(3, dtype=tf.float64)])
    graph_cores, graph_dot_cores, values, tangents, normalizers, valid = (
        padded_fixed_teacher_recursion_marginal_xla(
            basis_values,
            mask,
            tf.constant([0, 1, 0, 1], tf.int32),
            base_targets,
            dot_base_targets,
            weights,
            dot_weights,
            initial,
            initial_dot,
            shift_indices,
            tf.fill([2], tf.constant(0.05, tf.float64)),
            tf.zeros([2], tf.float64),
            query_basis,
            tf.constant([True, False]),
            mass,
            tf.ones([2, 32], tf.float64),
            tf.zeros([2, 32], tf.float64),
            tf.constant(1.0, tf.float64),
            tf.constant(0.0, tf.float64),
            tf.constant(config.ridge, tf.float64),
            tf.constant(config.column_scale_floor, tf.float64),
            tf.constant(config.condition_number_veto, tf.float64),
            tf.constant(1e-7, tf.float64),
        )
    )
    expected_cores = tf.stack(
        [
            tf.pad(reference.steps[-1].density.sqrt_tt.cores[0].values, [[0, 1], [0, 0], [0, 0]]),
            tf.pad(reference.steps[-1].density.sqrt_tt.cores[1].values, [[0, 0], [0, 0], [0, 1]]),
        ]
    ) * mask
    expected_dot_cores = tf.stack(
        [
            tf.pad(reference.steps[-1].dot_cores[0].values, [[0, 1], [0, 0], [0, 0]]),
            tf.pad(reference.steps[-1].dot_cores[1].values, [[0, 0], [0, 0], [0, 1]]),
        ]
    ) * mask
    expected_values = tf.stack([item.values for item in reference.carried_marginals])
    expected_tangents = tf.stack([item.tangent for item in reference.carried_marginals])
    expected_normalizers = tf.stack(
        [item.normalizer for item in reference.carried_marginals]
    )
    assert bool(valid.numpy())
    tf.debugging.assert_near(graph_cores, expected_cores, atol=3e-6, rtol=3e-6)
    tf.debugging.assert_near(
        graph_dot_cores, expected_dot_cores, atol=4e-5, rtol=4e-5
    )
    tf.debugging.assert_near(values, expected_values, atol=3e-6, rtol=3e-6)
    tf.debugging.assert_near(tangents, expected_tangents, atol=4e-5, rtol=4e-5)
    tf.debugging.assert_near(normalizers, expected_normalizers, atol=3e-6, rtol=3e-6)


def test_padded_xla_shape_targets_and_jvp_match_reference():
    product = highdim.ProductBasis(
        [
            highdim.LegendreBasis1D(highdim.BoundedInterval(-1.0, 1.0), 2),
            highdim.LegendreBasis1D(highdim.BoundedInterval(-1.0, 1.0), 2),
        ],
        _convention(),
    )
    sqrt_tt = highdim.FunctionalTT(
        (
            highdim.TTCore(
                tf.constant(
                    [[[1.0, 0.20, -0.10], [0.15, 0.60, 0.05], [-0.10, 0.05, 0.30]]],
                    tf.float64,
                )
            ),
            highdim.TTCore(
                tf.constant(
                    [
                        [[0.90], [0.20], [-0.05]],
                        [[0.10], [0.55], [0.15]],
                        [[-0.05], [0.10], [0.35]],
                    ],
                    tf.float64,
                )
            ),
        ),
        product,
        _convention(),
    )
    defensive = highdim.TensorProductReferenceDensity(product, _convention())
    identity = highdim.SquaredTTDensity.expected_branch_identity(
        sqrt_tt=sqrt_tt,
        defensive_density=defensive,
        tau=tf.constant(0.2, tf.float64),
        normalizer_floor=tf.constant(1e-12, tf.float64),
        denominator_floor=tf.constant(1e-12, tf.float64),
        measure_convention=_convention(),
    )
    density = highdim.SquaredTTDensity(
        sqrt_tt=sqrt_tt,
        defensive_density=defensive,
        tau=tf.constant(0.2, tf.float64),
        normalizer_floor=tf.constant(1e-12, tf.float64),
        denominator_floor=tf.constant(1e-12, tf.float64),
        measure_convention=_convention(),
        branch_identity=identity,
    )
    dot_cores = tuple(
        highdim.TTCore(tf.fill(tf.shape(core.values), tf.constant(0.01, tf.float64)))
        for core in sqrt_tt.cores
    )
    dot_tau = tf.constant(-0.03, tf.float64)
    reference = squared_tt_shape_targets_jvp(
        density,
        tf.zeros([2], tf.float64),
        tf.eye(2, dtype=tf.float64),
        dot_cores,
        pair_indices=((0, 1),),
        dot_tau=dot_tau,
    )
    mask = tf.constant(
        [
            [
                [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            ],
            [
                [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            ],
        ],
        tf.float64,
    )
    cores = tf.stack(
        [
            tf.pad(sqrt_tt.cores[0].values, [[0, 2], [0, 0], [0, 0]]),
            tf.pad(sqrt_tt.cores[1].values, [[0, 0], [0, 0], [0, 2]]),
        ]
    ) * mask
    dots = tf.stack(
        [
            tf.pad(dot_cores[0].values, [[0, 2], [0, 0], [0, 0]]),
            tf.pad(dot_cores[1].values, [[0, 0], [0, 0], [0, 2]]),
        ]
    ) * mask
    operator_powers = tf.stack(
        [
            tf.stack(
                [
                    legendre_monomial_operator_matrix(
                        product.bases[axis], power, highdim.MassMeasure.REFERENCE_MEASURE
                    )
                    for power in range(5)
                ]
            )
            for axis in range(2)
        ]
    )
    defensive_moments = []
    for axis in range(2):
        rows = []
        for power in range(5):
            powers = [0, 0]
            powers[axis] = power
            rows.append(tensor_product_reference_monomial_moment(density, powers))
        defensive_moments.append(tf.stack(rows))
    defensive_moments = tf.stack(defensive_moments)
    actual = padded_squared_tt_shape_targets_jvp_xla(
        cores,
        dots,
        operator_powers,
        defensive_moments,
        tf.zeros([2], tf.float64),
        tf.zeros([2], tf.float64),
        tf.eye(2, dtype=tf.float64),
        tf.zeros([2, 2], tf.float64),
        tf.constant([[0, 1]], tf.int32),
        density.tau,
        dot_tau,
        tf.constant(1.0, tf.float64),
    )
    expected = (
        reference.skew,
        reference.kurtosis,
        tf.reshape(reference.pairwise_co_skew[0, 1], [1]),
        tf.reshape(reference.pairwise_co_kurtosis[0, 1], [1]),
        reference.skew_tangent[:, 0],
        reference.kurtosis_tangent[:, 0],
        tf.reshape(reference.pairwise_co_skew_tangent[0, 1, 0], [1]),
        tf.reshape(reference.pairwise_co_kurtosis_tangent[0, 1, 0], [1]),
    )
    for graph_value, reference_value in zip(actual, expected):
        tf.debugging.assert_near(graph_value, reference_value, atol=3e-6, rtol=3e-6)

    h = tf.constant(1e-6, tf.float64)
    zero_dots = tf.zeros_like(dots)
    plus = padded_squared_tt_shape_targets_jvp_xla(
        cores + h * dots,
        zero_dots,
        operator_powers,
        defensive_moments,
        tf.zeros([2], tf.float64),
        tf.zeros([2], tf.float64),
        tf.eye(2, dtype=tf.float64),
        tf.zeros([2, 2], tf.float64),
        tf.constant([[0, 1]], tf.int32),
        density.tau + h * dot_tau,
        tf.constant(0.0, tf.float64),
        tf.constant(1.0, tf.float64),
    )
    minus = padded_squared_tt_shape_targets_jvp_xla(
        cores - h * dots,
        zero_dots,
        operator_powers,
        defensive_moments,
        tf.zeros([2], tf.float64),
        tf.zeros([2], tf.float64),
        tf.eye(2, dtype=tf.float64),
        tf.zeros([2, 2], tf.float64),
        tf.constant([[0, 1]], tf.int32),
        density.tau - h * dot_tau,
        tf.constant(0.0, tf.float64),
        tf.constant(1.0, tf.float64),
    )
    for tangent, plus_value, minus_value in zip(actual[4:], plus[:4], minus[:4]):
        tf.debugging.assert_near(
            tangent,
            (plus_value - minus_value) / (2.0 * h),
            atol=4e-6,
            rtol=4e-6,
        )


def test_padded_xla_fused_recursion_emits_reference_shape_targets():
    (
        product,
        points,
        basis_values,
        mask,
        _,
        _,
        weights,
        dot_weights,
        cores,
        _,
        initial,
        _,
        convention,
    ) = _problem()
    config = _config()
    direction = tf.constant(0.35, tf.float64)
    theta = tf.constant(0.1, tf.float64)
    base_targets = tf.stack(
        [
            -0.4 * tf.reduce_sum(tf.square(points - [0.2, -0.1]), axis=1)
            + theta * points[:, 0],
            -0.2 * tf.square(points[:, 1] - points[:, 0]) + theta * points[:, 1],
        ]
    )
    dot_base_targets = tf.stack(
        [direction * points[:, 0], direction * points[:, 1]]
    )
    zero_dot_cores = tuple(
        highdim.TTCore(tf.zeros_like(core.values)) for core in cores
    )
    reference = fixed_tt_teacher_recursion_jvp(
        product,
        points,
        base_targets,
        dot_base_targets,
        weights,
        config,
        cores,
        zero_dot_cores,
        convention,
        carried_keep_axes=(0,),
        previous_query_points=points[:, :1],
        state_offset=tf.zeros([2], tf.float64),
        state_matrix=tf.eye(2, dtype=tf.float64),
        pair_indices=((0, 1),),
        unscaled_defensive_weight=tf.constant(0.05, tf.float64),
    )
    shift_indices = tf.constant(
        [step.square_root_target.scale_shift_index for step in reference.steps],
        tf.int32,
    )
    operator_powers = tf.stack(
        [
            tf.stack(
                [
                    legendre_monomial_operator_matrix(
                        product.bases[axis], power, highdim.MassMeasure.REFERENCE_MEASURE
                    )
                    for power in range(5)
                ]
            )
            for axis in range(2)
        ]
    )
    defensive_density = reference.steps[0].density
    defensive_moments = []
    for axis in range(2):
        axis_moments = []
        for power in range(5):
            powers = [0, 0]
            powers[axis] = power
            axis_moments.append(
                tensor_product_reference_monomial_moment(defensive_density, powers)
            )
        defensive_moments.append(tf.stack(axis_moments))
    query_basis = tf.stack(
        [
            product.evaluate_axis(0, points[:, 0]),
            tf.ones_like(product.evaluate_axis(1, points[:, 1])),
        ]
    )
    fused_args = (
        basis_values,
        mask,
        tf.constant([0, 1, 0, 1], tf.int32),
        base_targets,
        dot_base_targets,
        weights,
        dot_weights,
        initial,
        tf.zeros_like(initial),
        shift_indices,
        tf.fill([2], tf.constant(0.05, tf.float64)),
        tf.zeros([2], tf.float64),
        query_basis,
        tf.constant([True, False]),
        tf.stack([tf.eye(3, dtype=tf.float64), tf.eye(3, dtype=tf.float64)]),
        tf.ones([2, 32], tf.float64),
        tf.zeros([2, 32], tf.float64),
        tf.constant(1.0, tf.float64),
        tf.constant(0.0, tf.float64),
        operator_powers,
        tf.stack(defensive_moments),
        tf.zeros([2], tf.float64),
        tf.zeros([2], tf.float64),
        tf.eye(2, dtype=tf.float64),
        tf.zeros([2, 2], tf.float64),
        tf.constant([[0, 1]], tf.int32),
        tf.constant(config.ridge, tf.float64),
        tf.constant(config.column_scale_floor, tf.float64),
        tf.constant(config.condition_number_veto, tf.float64),
        tf.constant(1e-7, tf.float64),
    )
    outputs = padded_fixed_teacher_recursion_shape_xla(*fused_args)
    assert bool(outputs[-1].numpy())
    expected_skew = tf.stack([shape.skew for shape in reference.shape_targets])
    expected_kurtosis = tf.stack(
        [shape.kurtosis for shape in reference.shape_targets]
    )
    expected_co_skew = tf.stack(
        [tf.reshape(shape.pairwise_co_skew[0, 1], [1]) for shape in reference.shape_targets]
    )
    expected_co_kurtosis = tf.stack(
        [tf.reshape(shape.pairwise_co_kurtosis[0, 1], [1]) for shape in reference.shape_targets]
    )
    expected_dot_skew = tf.stack(
        [shape.skew_tangent[:, 0] for shape in reference.shape_targets]
    )
    expected_dot_kurtosis = tf.stack(
        [shape.kurtosis_tangent[:, 0] for shape in reference.shape_targets]
    )
    expected_dot_co_skew = tf.stack(
        [tf.reshape(shape.pairwise_co_skew_tangent[0, 1, 0], [1]) for shape in reference.shape_targets]
    )
    expected_dot_co_kurtosis = tf.stack(
        [tf.reshape(shape.pairwise_co_kurtosis_tangent[0, 1, 0], [1]) for shape in reference.shape_targets]
    )
    for actual, expected in zip(
        outputs[5:13],
        (
            expected_skew,
            expected_kurtosis,
            expected_co_skew,
            expected_co_kurtosis,
            expected_dot_skew,
            expected_dot_kurtosis,
            expected_dot_co_skew,
            expected_dot_co_kurtosis,
        ),
    ):
        tf.debugging.assert_near(actual, expected, atol=6e-5, rtol=6e-5)
    graph_def = padded_fixed_teacher_recursion_shape_xla.get_concrete_function(
        *fused_args
    ).graph.as_graph_def()
    op_types = {node.op for node in graph_def.node}
    for function in graph_def.library.function:
        op_types.update(node.op for node in function.node_def)
    assert op_types.isdisjoint({"PyFunc", "EagerPyFunc"})
    assert op_types.intersection({"While", "StatelessWhile"})
