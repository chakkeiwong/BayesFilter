from __future__ import annotations

import tensorflow as tf

import bayesfilter.highdim as highdim
from bayesfilter.highdim.zhao_cui_moment_teacher import (
    frozen_squared_tt_shape_targets,
    apply_tt_shape_targets_reference_jvp,
    monomial_operator_matrices,
    padded_squared_tt_observable_jvp_xla,
    squared_tt_affine_form_moment,
    squared_tt_affine_form_moment_jvp,
    squared_tt_raw_moment,
    squared_tt_normalized_marginal_jvp,
    squared_tt_reference_moments,
    squared_tt_shape_targets_jvp,
    tt_particle_contract_e_step_reference_jvp,
)
from bayesfilter.highdim.higher_moment_contract_e import higher_moment_shape_jvp


def _convention() -> highdim.MeasureConvention:
    return highdim.MeasureConvention(
        density_measure=highdim.DensityMeasure.REFERENCE_MEASURE,
        mass_measure=highdim.MassMeasure.REFERENCE_MEASURE,
        reference_weight_name="omega",
    )


def _density(epsilon: float = 0.0) -> highdim.SquaredTTDensity:
    product = highdim.ProductBasis(
        [
            highdim.LegendreBasis1D(highdim.BoundedInterval(-1.0, 1.0), 2),
            highdim.LegendreBasis1D(highdim.BoundedInterval(-1.0, 1.0), 2),
        ],
        _convention(),
    )
    cores = [
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
    ]
    if epsilon:
        direction = [
            highdim.TTCore(tf.fill(tf.shape(core.values), tf.constant(epsilon, tf.float64)))
            for core in cores
        ]
        cores = [
            highdim.TTCore(core.values + direction.values)
            for core, direction in zip(cores, direction)
        ]
    ftt = highdim.FunctionalTT(cores, product, _convention())
    defensive = highdim.TensorProductReferenceDensity(product, _convention())
    tau = tf.constant(0.2, tf.float64)
    return highdim.SquaredTTDensity(
        sqrt_tt=ftt,
        defensive_density=defensive,
        tau=tau,
        normalizer_floor=tf.constant(1e-12, tf.float64),
        denominator_floor=tf.constant(1e-12, tf.float64),
        measure_convention=_convention(),
        branch_identity=highdim.SquaredTTDensity.expected_branch_identity(
            sqrt_tt=ftt,
            defensive_density=defensive,
            tau=tau,
            normalizer_floor=tf.constant(1e-12, tf.float64),
            denominator_floor=tf.constant(1e-12, tf.float64),
            measure_convention=_convention(),
        ),
    )


def _quadrature_moment(density, first, offset, power, second=None, second_offset=0.0, second_power=0):
    grid, axis_weights = highdim.legendre_gauss_nodes_weights(10)
    axis_weights = 0.5 * axis_weights
    x, y = tf.meshgrid(grid, grid, indexing="ij")
    wx, wy = tf.meshgrid(axis_weights, axis_weights, indexing="ij")
    points = tf.stack([tf.reshape(x, [-1]), tf.reshape(y, [-1])], axis=1)
    values = tf.exp(density.log_density(points))
    first_form = offset + tf.linalg.matvec(points, first)
    observable = tf.pow(first_form, power)
    if second is not None and second_power:
        observable = observable * tf.pow(second_offset + tf.linalg.matvec(points, second), second_power)
    return tf.reduce_sum(values * observable * tf.reshape(wx * wy, [-1]))


def test_squared_tt_raw_moment_matches_dense_quadrature_with_defensive_mass():
    density = _density()
    actual = squared_tt_raw_moment(density, (2, 1))
    expected = _quadrature_moment(
        density,
        tf.constant([1.0, 0.0], tf.float64),
        tf.constant(0.0, tf.float64),
        2,
        tf.constant([0.0, 1.0], tf.float64),
        tf.constant(0.0, tf.float64),
        1,
    )
    tf.debugging.assert_near(actual, expected, atol=2e-12)


def test_squared_tt_affine_form_moment_matches_dense_quadrature():
    density = _density()
    first = tf.constant([0.7, -0.35], tf.float64)
    second = tf.constant([-0.2, 0.45], tf.float64)
    actual = squared_tt_affine_form_moment(
        density,
        first,
        tf.constant(0.15, tf.float64),
        2,
        second_coefficients=second,
        second_offset=tf.constant(-0.1, tf.float64),
        second_power=2,
    )
    expected = _quadrature_moment(
        density,
        first,
        tf.constant(0.15, tf.float64),
        2,
        second,
        tf.constant(-0.1, tf.float64),
        2,
    )
    tf.debugging.assert_near(actual, expected, atol=2e-12)


def test_squared_tt_affine_form_jvp_matches_finite_difference():
    density = _density()
    dot_cores = tuple(
        highdim.TTCore(tf.fill(tf.shape(core.values), tf.constant(0.03, tf.float64)))
        for core in density.sqrt_tt.cores
    )
    analytic = squared_tt_affine_form_moment_jvp(
        density,
        tf.constant([0.7, -0.35], tf.float64),
        tf.constant(0.15, tf.float64),
        2,
        dot_cores,
        second_coefficients=tf.constant([-0.2, 0.45], tf.float64),
        second_offset=tf.constant(-0.1, tf.float64),
        second_power=2,
    ).tangent
    h = tf.constant(1e-5, tf.float64)
    plus_cores = tuple(highdim.TTCore(core.values + h * dot.values) for core, dot in zip(density.sqrt_tt.cores, dot_cores))
    minus_cores = tuple(highdim.TTCore(core.values - h * dot.values) for core, dot in zip(density.sqrt_tt.cores, dot_cores))
    def replace(cores):
        defensive = highdim.TensorProductReferenceDensity(density.sqrt_tt.product_basis, _convention())
        return highdim.SquaredTTDensity(
            sqrt_tt=highdim.FunctionalTT(cores, density.sqrt_tt.product_basis, _convention()),
            defensive_density=defensive,
            tau=density.tau,
            normalizer_floor=density.normalizer_floor,
            denominator_floor=density.denominator_floor,
            measure_convention=_convention(),
            branch_identity=highdim.SquaredTTDensity.expected_branch_identity(
                sqrt_tt=highdim.FunctionalTT(cores, density.sqrt_tt.product_basis, _convention()),
                defensive_density=defensive,
                tau=density.tau,
                normalizer_floor=density.normalizer_floor,
                denominator_floor=density.denominator_floor,
                measure_convention=_convention(),
            ),
        )
    plus = squared_tt_affine_form_moment(replace(plus_cores), tf.constant([0.7, -0.35], tf.float64), tf.constant(0.15, tf.float64), 2, second_coefficients=tf.constant([-0.2, 0.45], tf.float64), second_offset=tf.constant(-0.1, tf.float64), second_power=2)
    minus = squared_tt_affine_form_moment(replace(minus_cores), tf.constant([0.7, -0.35], tf.float64), tf.constant(0.15, tf.float64), 2, second_coefficients=tf.constant([-0.2, 0.45], tf.float64), second_offset=tf.constant(-0.1, tf.float64), second_power=2)
    tf.debugging.assert_near(analytic, (plus - minus) / (2.0 * h), atol=2e-7)


def test_normalized_marginal_jvp_matches_finite_difference():
    density = _density()
    dot_cores = tuple(
        highdim.TTCore(tf.fill(tf.shape(core.values), tf.constant(0.03, tf.float64)))
        for core in density.sqrt_tt.cores
    )
    points = tf.constant([[-0.8], [-0.1], [0.45]], tf.float64)
    dot_tau = tf.constant(-0.04, tf.float64)
    analytic = squared_tt_normalized_marginal_jvp(
        density,
        (0,),
        points,
        dot_cores,
        dot_tau=dot_tau,
    )

    def replace(sign):
        h = tf.constant(1e-6, tf.float64)
        cores = tuple(
            highdim.TTCore(core.values + sign * h * dot.values)
            for core, dot in zip(density.sqrt_tt.cores, dot_cores)
        )
        sqrt_tt = highdim.FunctionalTT(
            cores, density.sqrt_tt.product_basis, _convention()
        )
        tau = density.tau + sign * h * dot_tau
        identity = highdim.SquaredTTDensity.expected_branch_identity(
            sqrt_tt=sqrt_tt,
            defensive_density=density.defensive_density,
            tau=tau,
            normalizer_floor=density.normalizer_floor,
            denominator_floor=density.denominator_floor,
            measure_convention=_convention(),
        )
        return highdim.SquaredTTDensity(
            sqrt_tt=sqrt_tt,
            defensive_density=density.defensive_density,
            tau=tau,
            normalizer_floor=density.normalizer_floor,
            denominator_floor=density.denominator_floor,
            measure_convention=_convention(),
            branch_identity=identity,
        )

    h = tf.constant(1e-6, tf.float64)
    plus = replace(tf.constant(1.0, tf.float64)).normalized_marginal_density_values(
        (0,), points
    )
    minus = replace(tf.constant(-1.0, tf.float64)).normalized_marginal_density_values(
        (0,), points
    )
    tf.debugging.assert_near(analytic.values, density.normalized_marginal_density_values((0,), points), atol=2e-12)
    tf.debugging.assert_near(
        analytic.tangent,
        (plus - minus) / (2.0 * h),
        atol=2e-8,
        rtol=2e-8,
    )


def test_lebesgue_defensive_marginal_includes_integrated_domain_volume():
    convention = highdim.MeasureConvention(
        density_measure=highdim.DensityMeasure.REFERENCE_LEBESGUE,
        mass_measure=highdim.MassMeasure.REFERENCE_LEBESGUE,
        reference_weight_name="lebesgue",
    )
    product = highdim.ProductBasis(
        [
            highdim.LegendreBasis1D(highdim.BoundedInterval(-1.0, 1.0), 0),
            highdim.LegendreBasis1D(highdim.BoundedInterval(-2.0, 2.0), 0),
        ],
        convention,
    )
    cores = (
        highdim.TTCore(tf.zeros([1, 1, 1], tf.float64)),
        highdim.TTCore(tf.zeros([1, 1, 1], tf.float64)),
    )
    sqrt_tt = highdim.FunctionalTT(cores, product, convention)
    defensive = highdim.TensorProductReferenceDensity(product, convention)
    tau = tf.constant(1.0, tf.float64)
    identity = highdim.SquaredTTDensity.expected_branch_identity(
        sqrt_tt=sqrt_tt,
        defensive_density=defensive,
        tau=tau,
        normalizer_floor=tf.constant(1e-12, tf.float64),
        denominator_floor=tf.constant(1e-12, tf.float64),
        measure_convention=convention,
    )
    density = highdim.SquaredTTDensity(
        sqrt_tt=sqrt_tt,
        defensive_density=defensive,
        tau=tau,
        normalizer_floor=tf.constant(1e-12, tf.float64),
        denominator_floor=tf.constant(1e-12, tf.float64),
        measure_convention=convention,
        branch_identity=identity,
    )
    points = tf.constant([[-0.5], [0.5]], tf.float64)
    expected = tf.fill([2], tf.constant(1.0 / 2.0, tf.float64))
    tf.debugging.assert_near(
        density.normalized_marginal_density_values((0,), points), expected
    )


def test_frozen_tt_targets_feed_shape_correction_with_ordered_pair_masks():
    targets = frozen_squared_tt_shape_targets(
        _density(),
        tf.zeros([2], tf.float64),
        tf.eye(2, dtype=tf.float64),
        pair_indices=((0, 1),),
        parameter_count=1,
    )
    tf.debugging.assert_equal(
        targets.pairwise_co_skew_mask,
        tf.constant([[0.0, 1.0], [0.0, 0.0]], tf.float64),
    )
    tf.debugging.assert_equal(
        targets.pairwise_co_kurtosis_mask,
        tf.constant([[0.0, 1.0], [1.0, 0.0]], tf.float64),
    )
    tf.debugging.assert_near(
        targets.pairwise_co_kurtosis,
        tf.transpose(targets.pairwise_co_kurtosis),
        atol=2e-12,
    )

    count = 48
    source = tf.random.stateless_normal([count, 2], [41, 42], dtype=tf.float64)
    weights = tf.nn.softmax(
        tf.random.stateless_normal([count], [43, 44], dtype=tf.float64)
    )
    points = tf.random.stateless_normal([count, 2], [45, 46], dtype=tf.float64)
    result = higher_moment_shape_jvp(
        source,
        weights,
        tf.zeros([count, 2, 1], tf.float64),
        tf.zeros([count, 1], tf.float64),
        points,
        tf.zeros([count, 2, 1], tf.float64),
        correction_steps=1,
        strength=0.01,
        floor=1e-6,
        pairwise_correction_steps=1,
        pairwise_strength=0.01,
        pairwise_floor=1e-6,
        **targets.explicit_target_kwargs(),
    )
    tf.debugging.assert_equal(result["target_source_id"], 1)
    tf.debugging.assert_equal(
        result["pairwise_co_skew_target_mask"], targets.pairwise_co_skew_mask
    )
    tf.debugging.assert_equal(
        result["pairwise_co_kurtosis_target_mask"],
        targets.pairwise_co_kurtosis_mask,
    )
    tf.debugging.assert_equal(
        result["pairwise_co_skew_residual"][1, 0],
        tf.constant(0.0, tf.float64),
    )
    assert bool(result["valid"].numpy())


def test_recursive_shape_target_jvp_matches_finite_difference():
    density = _density()
    dot_cores = tuple(
        highdim.TTCore(tf.fill(tf.shape(core.values), tf.constant(0.03, tf.float64)))
        for core in density.sqrt_tt.cores
    )
    offset = tf.constant([0.1, -0.2], tf.float64)
    matrix = tf.constant([[1.0, 0.15], [-0.2, 0.9]], tf.float64)
    dot_tau = tf.constant(-0.04, tf.float64)
    analytic = squared_tt_shape_targets_jvp(
        density,
        offset,
        matrix,
        dot_cores,
        pair_indices=((0, 1),),
        dot_tau=dot_tau,
    )
    h = tf.constant(1e-6, tf.float64)

    def replace(sign):
        cores = tuple(
            highdim.TTCore(core.values + sign * h * dot.values)
            for core, dot in zip(density.sqrt_tt.cores, dot_cores)
        )
        sqrt_tt = highdim.FunctionalTT(cores, density.sqrt_tt.product_basis, _convention())
        tau = density.tau + sign * h * dot_tau
        identity = highdim.SquaredTTDensity.expected_branch_identity(
            sqrt_tt=sqrt_tt,
            defensive_density=density.defensive_density,
            tau=tau,
            normalizer_floor=density.normalizer_floor,
            denominator_floor=density.denominator_floor,
            measure_convention=_convention(),
        )
        return highdim.SquaredTTDensity(
            sqrt_tt=sqrt_tt,
            defensive_density=density.defensive_density,
            tau=tau,
            normalizer_floor=density.normalizer_floor,
            denominator_floor=density.denominator_floor,
            measure_convention=_convention(),
            branch_identity=identity,
        )

    plus = frozen_squared_tt_shape_targets(
        replace(tf.constant(1.0, tf.float64)),
        offset,
        matrix,
        pair_indices=((0, 1),),
        parameter_count=1,
    )
    minus = frozen_squared_tt_shape_targets(
        replace(tf.constant(-1.0, tf.float64)),
        offset,
        matrix,
        pair_indices=((0, 1),),
        parameter_count=1,
    )
    tf.debugging.assert_near(analytic.skew_tangent[:, 0], (plus.skew - minus.skew) / (2.0 * h), atol=2e-6, rtol=2e-6)
    tf.debugging.assert_near(analytic.kurtosis_tangent[:, 0], (plus.kurtosis - minus.kurtosis) / (2.0 * h), atol=2e-6, rtol=2e-6)
    tf.debugging.assert_near(analytic.pairwise_co_skew_tangent[:, :, 0], (plus.pairwise_co_skew - minus.pairwise_co_skew) / (2.0 * h), atol=2e-6, rtol=2e-6)
    tf.debugging.assert_near(analytic.pairwise_co_kurtosis_tangent[:, :, 0], (plus.pairwise_co_kurtosis - minus.pairwise_co_kurtosis) / (2.0 * h), atol=2e-6, rtol=2e-6)

    count = 48
    source = tf.random.stateless_normal([count, 2], [151, 152], dtype=tf.float64)
    weights = tf.nn.softmax(
        tf.random.stateless_normal([count], [153, 154], dtype=tf.float64)
    )
    points = tf.random.stateless_normal([count, 2], [155, 156], dtype=tf.float64)

    def repair(targets):
        return apply_tt_shape_targets_reference_jvp(
            source,
            weights,
            tf.zeros([count, 2, 1], tf.float64),
            tf.zeros([count, 1], tf.float64),
            points,
            tf.zeros([count, 2, 1], tf.float64),
            targets,
            correction_steps=1,
            strength=0.01,
            floor=1e-6,
            pairwise_correction_steps=1,
            pairwise_strength=0.01,
            pairwise_floor=1e-6,
        )

    repaired = repair(analytic)
    repaired_plus = repair(plus)
    repaired_minus = repair(minus)
    assert bool(repaired["valid"].numpy())
    tf.debugging.assert_near(
        repaired["particles_tangent"][:, :, 0],
        (repaired_plus["particles"] - repaired_minus["particles"]) / (2.0 * h),
        atol=2e-5,
        rtol=2e-5,
    )


def test_particle_ot_contract_e_tt_composition_preserves_particle_increment():
    density = _density()
    dot_cores = tuple(
        highdim.TTCore(tf.fill(tf.shape(core.values), tf.constant(0.01, tf.float64)))
        for core in density.sqrt_tt.cores
    )
    targets = squared_tt_shape_targets_jvp(
        density,
        tf.zeros([2], tf.float64),
        tf.eye(2, dtype=tf.float64),
        dot_cores,
        pair_indices=((0, 1),),
    )
    count = 48
    source = tf.random.stateless_normal([count, 2], [161, 162], dtype=tf.float64)
    source_tangent = 0.02 * tf.random.stateless_normal(
        [count, 2, 1], [165, 166], dtype=tf.float64
    )
    weights = tf.nn.softmax(
        tf.random.stateless_normal([count], [163, 164], dtype=tf.float64)
    )
    raw_weight_direction = tf.random.stateless_normal(
        [count], [167, 168], dtype=tf.float64
    )
    weight_tangent = weights * (
        raw_weight_direction
        - tf.reduce_sum(weights * raw_weight_direction)
    )
    weight_tangent = weight_tangent[:, None]
    design = tf.tile(
        tf.constant([[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]], tf.float64),
        [count // 4, 1],
    )
    increment = tf.constant(-3.25, tf.float64)
    increment_tangent = tf.constant([0.7], tf.float64)
    result = tt_particle_contract_e_step_reference_jvp(
        source,
        weights,
        source_tangent,
        weight_tangent,
        increment,
        increment_tangent,
        design,
        targets,
        epsilon=2.0,
        sinkhorn_steps=8,
        balance_steps=8,
        ridge=1e-3,
        correction_steps=1,
        strength=0.01,
        floor=1e-6,
        pairwise_correction_steps=1,
        pairwise_strength=0.01,
        pairwise_floor=1e-6,
    )
    tf.debugging.assert_equal(result.particle_log_increment, increment)
    tf.debugging.assert_equal(
        result.particle_log_increment_tangent, increment_tangent
    )
    tf.debugging.assert_equal(result.diagnostics["reset_valid"], True)
    tf.debugging.assert_equal(result.diagnostics["shape_valid"], True)
    assert result.particles.shape == source.shape
    assert result.particles_tangent.shape == (count, 2, 1)

    h = tf.constant(1e-6, tf.float64)

    def perturbed_density(sign):
        cores = tuple(
            highdim.TTCore(core.values + sign * h * dot.values)
            for core, dot in zip(density.sqrt_tt.cores, dot_cores)
        )
        sqrt_tt = highdim.FunctionalTT(
            cores, density.sqrt_tt.product_basis, _convention()
        )
        identity = highdim.SquaredTTDensity.expected_branch_identity(
            sqrt_tt=sqrt_tt,
            defensive_density=density.defensive_density,
            tau=density.tau,
            normalizer_floor=density.normalizer_floor,
            denominator_floor=density.denominator_floor,
            measure_convention=_convention(),
        )
        return highdim.SquaredTTDensity(
            sqrt_tt=sqrt_tt,
            defensive_density=density.defensive_density,
            tau=density.tau,
            normalizer_floor=density.normalizer_floor,
            denominator_floor=density.denominator_floor,
            measure_convention=_convention(),
            branch_identity=identity,
        )

    def perturbed(sign):
        local_targets = frozen_squared_tt_shape_targets(
            perturbed_density(sign),
            tf.zeros([2], tf.float64),
            tf.eye(2, dtype=tf.float64),
            pair_indices=((0, 1),),
            parameter_count=1,
        )
        return tt_particle_contract_e_step_reference_jvp(
            source + sign * h * source_tangent[:, :, 0],
            weights + sign * h * weight_tangent[:, 0],
            tf.zeros_like(source_tangent),
            tf.zeros_like(weight_tangent),
            increment + sign * h * increment_tangent[0],
            tf.zeros_like(increment_tangent),
            design,
            local_targets,
            epsilon=2.0,
            sinkhorn_steps=8,
            balance_steps=8,
            ridge=1e-3,
            correction_steps=1,
            strength=0.01,
            floor=1e-6,
            pairwise_correction_steps=1,
            pairwise_strength=0.01,
            pairwise_floor=1e-6,
        )

    plus = perturbed(tf.constant(1.0, tf.float64))
    minus = perturbed(tf.constant(-1.0, tf.float64))
    tf.debugging.assert_near(
        result.particles_tangent[:, :, 0],
        (plus.particles - minus.particles) / (2.0 * h),
        atol=3e-5,
        rtol=3e-5,
    )


def test_padded_xla_observable_kernel_matches_variable_rank_reference():
    density = _density()
    dot_cores = tuple(
        highdim.TTCore(tf.fill(tf.shape(core.values), tf.constant(0.03, tf.float64)))
        for core in density.sqrt_tt.cores
    )
    operators = monomial_operator_matrices(density, (2, 1))
    reference = squared_tt_raw_moment(density, (2, 1))
    reference_jvp = squared_tt_affine_form_moment_jvp(
        density,
        tf.constant([1.0, 0.0], tf.float64),
        tf.constant(0.0, tf.float64),
        2,
        dot_cores,
        second_coefficients=tf.constant([0.0, 1.0], tf.float64),
        second_offset=tf.constant(0.0, tf.float64),
        second_power=1,
    )
    rank = 3
    first_core = tf.pad(density.sqrt_tt.cores[0].values, [[0, rank - 1], [0, 0], [0, 0]])
    second_core = tf.pad(density.sqrt_tt.cores[1].values, [[0, 0], [0, 0], [0, rank - 1]])
    first_dot = tf.pad(dot_cores[0].values, [[0, rank - 1], [0, 0], [0, 0]])
    second_dot = tf.pad(dot_cores[1].values, [[0, 0], [0, 0], [0, rank - 1]])
    cores = tf.stack([first_core, second_core])
    directions = tf.stack([first_dot, second_dot])
    observable_operators = tf.stack(operators)
    mass_operators = tf.stack(
        [basis.mass_matrix(_convention().mass_measure) for basis in density.sqrt_tt.product_basis.bases]
    )
    zeros = tf.zeros_like(observable_operators)
    value, tangent, normalizer, normalizer_tangent = padded_squared_tt_observable_jvp_xla(
        cores,
        directions,
        observable_operators,
        zeros,
        mass_operators,
        tf.zeros_like(mass_operators),
        density.tau,
        tf.constant(0.0, tf.float64),
        tf.constant(0.0, tf.float64),
        tf.constant(0.0, tf.float64),
        tf.constant(1.0, tf.float64),
        tf.constant(0.0, tf.float64),
    )
    tf.debugging.assert_near(value, reference, atol=2e-12)
    tf.debugging.assert_near(tangent, reference_jvp.tangent, atol=2e-12)
    tf.debugging.assert_near(normalizer, density.normalizer(), atol=2e-12)
    tf.debugging.assert_near(
        normalizer_tangent, reference_jvp.normalizer_tangent, atol=2e-12
    )

    concrete = padded_squared_tt_observable_jvp_xla.get_concrete_function(
        cores,
        directions,
        observable_operators,
        zeros,
        mass_operators,
        tf.zeros_like(mass_operators),
        density.tau,
        tf.constant(0.0, tf.float64),
        tf.constant(0.0, tf.float64),
        tf.constant(0.0, tf.float64),
        tf.constant(1.0, tf.float64),
        tf.constant(0.0, tf.float64),
    )
    operations = {node.op for node in concrete.graph.as_graph_def().node}
    assert "StatelessWhile" in operations or "While" in operations
    assert "PyFunc" not in operations
    assert "EagerPyFunc" not in operations


def test_scalar_lgssm_fitted_tt_moments_match_kalman_and_wick_targets():
    product = highdim.ProductBasis(
        [highdim.LegendreBasis1D(highdim.BoundedInterval(-1.0, 1.0), 28)],
        _convention(),
    )
    config = highdim.FixedBranchFilterConfig(
        fit_config=highdim.FixedTTFitConfig(
            ranks=(1, 1),
            ridge=1e-12,
            max_sweeps=1,
            sweep_order=(0,),
            row_budget=128,
            column_budget=64,
            dense_matrix_byte_budget=300_000,
            normal_matrix_byte_budget=10_000,
            condition_number_warning=1e10,
            condition_number_veto=1e14,
            holdout_tolerance=1e6,
        ),
        density_tau=0.0,
        normalizer_floor=1e-14,
        denominator_floor=1e-14,
        retained_storage_byte_budget=10_000_000,
        coordinate_maps=(
            highdim.AffineCoordinateMap(
                offset=tf.constant([0.0], tf.float64),
                matrix=tf.constant([[2.0]], tf.float64),
            ),
        ),
        measure_convention=_convention(),
        deterministic_seed="tt-moment-teacher-lgssm-t1",
        product_basis=product,
        fit_quadrature_order=64,
    )
    model = highdim.LinearGaussianSSM(
        initial_mean=tf.constant([0.0], tf.float64),
        initial_covariance=tf.constant([[1.0]], tf.float64),
        transition_matrix=tf.constant([[0.7]], tf.float64),
        transition_covariance=tf.constant([[0.25]], tf.float64),
        observation_matrix=tf.constant([[1.0]], tf.float64),
        observation_covariance=tf.constant([[0.09]], tf.float64),
    )
    result = highdim.FixedBranchSquaredTTFilter(config).log_likelihood(
        model,
        tf.zeros([0], tf.float64),
        tf.constant([0.2], tf.float64),
    )
    density = result.steps[0].density
    assert density is not None
    reference = squared_tt_reference_moments(density, (0,))
    physical_mean = 2.0 * reference.mean[0]
    physical_variance = 4.0 * reference.covariance[0, 0]
    kalman_mean = result.retained_filter.diagnostics["mean"][0]
    kalman_variance = result.retained_filter.diagnostics["covariance"][0, 0]
    standardized_coefficient = tf.math.rsqrt(reference.covariance[0, 0])
    standardized_offset = -reference.mean[0] * standardized_coefficient
    skew = squared_tt_affine_form_moment(
        density,
        tf.reshape(standardized_coefficient, [1]),
        standardized_offset,
        3,
    )
    kurtosis = squared_tt_affine_form_moment(
        density,
        tf.reshape(standardized_coefficient, [1]),
        standardized_offset,
        4,
    )
    # A bounded degree-28 square-root polynomial is not an exact Gaussian.
    # These tolerances gate the fitted teacher error, after contraction parity
    # has already been established independently above.
    tf.debugging.assert_near(physical_mean, kalman_mean, atol=2e-8)
    tf.debugging.assert_near(physical_variance, kalman_variance, atol=2e-7)
    tf.debugging.assert_near(skew, tf.constant(0.0, tf.float64), atol=2e-6)
    tf.debugging.assert_near(kurtosis, tf.constant(3.0, tf.float64), atol=2e-6)
