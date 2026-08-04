from __future__ import annotations

import tensorflow as tf

import bayesfilter.highdim as highdim
from bayesfilter.highdim.zhao_cui_moment_teacher_als import (
    fixed_als_value_jvp,
    fixed_tt_teacher_recursion_jvp,
    fixed_tt_teacher_step_jvp,
    scaled_defensive_weight_jvp,
    square_root_target_jvp,
)


def _convention():
    return highdim.MeasureConvention(
        density_measure=highdim.DensityMeasure.REFERENCE_MEASURE,
        mass_measure=highdim.MassMeasure.REFERENCE_MEASURE,
        reference_weight_name="omega",
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
    return product, points, target, dot_target, weights, dot_weights, cores, dot_cores


def test_fixed_als_replay_matches_fitter_value_and_reports_residuals():
    product, points, target, dot_target, weights, dot_weights, cores, dot_cores = _problem()
    config = _config()
    convention = _convention()
    replay = fixed_als_value_jvp(
        product,
        points,
        target,
        weights,
        dot_target,
        config,
        cores,
        dot_cores,
        convention,
        dot_weights=dot_weights,
    )
    fitted = highdim.FixedTTFitter().fit(
        product,
        highdim.FixedTTFitSampleBatch(points, target, weights),
        config,
        cores,
        branch_seed="als-replay-test",
        measure_convention=convention,
    )
    for actual, expected in zip(replay.cores, fitted.fitted_tt.cores):
        tf.debugging.assert_near(actual.values, expected.values, atol=2e-10, rtol=2e-10)
    assert len(replay.update_diagnostics) == config.max_sweeps * len(config.sweep_order)
    for record in replay.update_diagnostics:
        assert float(record["value_solve_residual"].numpy()) < 1e-8
        assert float(record["jvp_solve_residual"].numpy()) < 1e-8
        assert record["design_tangent_included"] is True


def test_square_root_target_jvp_includes_log_scale_shift_tangent():
    log_target = tf.constant([-2.0, 0.5, -0.25], tf.float64)
    dot_log_target = tf.constant([0.3, -0.2, 0.7], tf.float64)
    result = square_root_target_jvp(log_target, dot_log_target)
    assert result.scale_shift_index == 1
    tf.debugging.assert_equal(result.log_scale_shift, log_target[1])
    tf.debugging.assert_equal(result.dot_log_scale_shift, dot_log_target[1])
    tf.debugging.assert_equal(result.values[1], tf.constant(1.0, tf.float64))
    tf.debugging.assert_equal(result.tangent[1], tf.constant(0.0, tf.float64))

    h = tf.constant(1e-6, tf.float64)
    plus = square_root_target_jvp(
        log_target + h * dot_log_target,
        tf.zeros_like(dot_log_target),
        scale_shift_index=result.scale_shift_index,
    )
    minus = square_root_target_jvp(
        log_target - h * dot_log_target,
        tf.zeros_like(dot_log_target),
        scale_shift_index=result.scale_shift_index,
    )
    tf.debugging.assert_near(
        result.tangent,
        (plus.values - minus.values) / (2.0 * h),
        atol=2e-10,
        rtol=2e-10,
    )


def test_square_root_scaling_cancels_only_without_defensive_density():
    log_target = tf.constant([-1.0, 0.4, -0.2], tf.float64)
    target = tf.exp(log_target)
    fit = square_root_target_jvp(log_target, tf.zeros_like(log_target))
    represented = tf.square(fit.values)
    tf.debugging.assert_near(
        represented / tf.reduce_sum(represented),
        target / tf.reduce_sum(target),
        atol=2e-15,
    )

    tau = tf.constant(0.2, tf.float64)
    defensive = tf.ones_like(target)
    with_shift = (represented + tau * defensive) / tf.reduce_sum(
        represented + tau * defensive
    )
    unshifted = (target + tau * defensive) / tf.reduce_sum(target + tau * defensive)
    assert bool(tf.reduce_any(tf.abs(with_shift - unshifted) > 1e-4).numpy())

    scaled = scaled_defensive_weight_jvp(
        tau,
        tf.constant(0.0, tf.float64),
        fit.log_scale_shift,
        tf.constant(0.0, tf.float64),
    )
    scale_consistent = (represented + scaled.tau * defensive) / tf.reduce_sum(
        represented + scaled.tau * defensive
    )
    tf.debugging.assert_near(scale_consistent, unshifted, atol=2e-15)


def test_scaled_defensive_weight_jvp_includes_shift_derivative():
    weight = tf.constant(0.3, tf.float64)
    dot_weight = tf.constant(-0.04, tf.float64)
    shift = tf.constant(1.2, tf.float64)
    dot_shift = tf.constant(0.25, tf.float64)
    analytic = scaled_defensive_weight_jvp(
        weight, dot_weight, shift, dot_shift
    )
    h = tf.constant(1e-6, tf.float64)
    plus = scaled_defensive_weight_jvp(
        weight + h * dot_weight,
        tf.constant(0.0, tf.float64),
        shift + h * dot_shift,
        tf.constant(0.0, tf.float64),
    )
    minus = scaled_defensive_weight_jvp(
        weight - h * dot_weight,
        tf.constant(0.0, tf.float64),
        shift - h * dot_shift,
        tf.constant(0.0, tf.float64),
    )
    tf.debugging.assert_near(
        analytic.dot_tau,
        (plus.tau - minus.tau) / (2.0 * h),
        atol=2e-11,
        rtol=2e-11,
    )


def test_fixed_als_replay_jvp_matches_centered_finite_difference():
    product, points, target, dot_target, weights, dot_weights, cores, dot_cores = _problem()
    config = _config()
    convention = _convention()
    analytic = fixed_als_value_jvp(
        product,
        points,
        target,
        weights,
        dot_target,
        config,
        cores,
        dot_cores,
        convention,
        dot_weights=dot_weights,
    )
    h = tf.constant(1e-5, tf.float64)
    plus = fixed_als_value_jvp(
        product,
        points,
        target + h * dot_target,
        weights + h * dot_weights,
        tf.zeros_like(dot_target),
        config,
        tuple(highdim.TTCore(core.values + h * dot.values) for core, dot in zip(cores, dot_cores)),
        tuple(highdim.TTCore(tf.zeros_like(core.values)) for core in cores),
        convention,
    )
    minus = fixed_als_value_jvp(
        product,
        points,
        target - h * dot_target,
        weights - h * dot_weights,
        tf.zeros_like(dot_target),
        config,
        tuple(highdim.TTCore(core.values - h * dot.values) for core, dot in zip(cores, dot_cores)),
        tuple(highdim.TTCore(tf.zeros_like(core.values)) for core in cores),
        convention,
    )
    for actual, p_core, m_core in zip(analytic.dot_cores, plus.cores, minus.cores):
        finite_difference = (p_core.values - m_core.values) / (2.0 * h)
        tf.debugging.assert_near(actual.values, finite_difference, atol=3e-6, rtol=3e-6)


def test_two_step_teacher_recursion_jvp_matches_centered_finite_difference():
    product, points, _, _, weights, _, cores, dot_cores = _problem()
    config = _config()
    convention = _convention()
    direction = tf.constant(0.35, tf.float64)

    def run(theta, dot_theta):
        first_base = -0.4 * tf.reduce_sum(
            tf.square(points - [0.2, -0.1]), axis=1
        ) + theta * points[:, 0]
        second_base = (
            -0.2 * tf.square(points[:, 1] - points[:, 0])
            + theta * points[:, 1]
        )
        recursion = fixed_tt_teacher_recursion_jvp(
            product,
            points,
            tf.stack([first_base, second_base]),
            tf.stack([dot_theta * points[:, 0], dot_theta * points[:, 1]]),
            weights,
            config,
            cores,
            tuple(
                highdim.TTCore(tf.zeros_like(core.values)) for core in cores
            ),
            convention,
            carried_keep_axes=(0,),
            previous_query_points=points[:, :1],
            state_offset=tf.zeros([2], tf.float64),
            state_matrix=tf.eye(2, dtype=tf.float64),
            pair_indices=((0, 1),),
            unscaled_defensive_weight=tf.constant(0.05, tf.float64),
        )
        assert len(recursion.steps) == 2
        assert len(recursion.carried_marginals) == 2
        assert len(recursion.shape_targets) == 2
        return recursion.steps[-1]

    base = run(tf.constant(0.1, tf.float64), direction)
    h = tf.constant(1e-5, tf.float64)
    plus = run(tf.constant(0.1, tf.float64) + h * direction, tf.constant(0.0, tf.float64))
    minus = run(tf.constant(0.1, tf.float64) - h * direction, tf.constant(0.0, tf.float64))
    for actual, p_core, m_core in zip(
        base.dot_cores, plus.density.sqrt_tt.cores, minus.density.sqrt_tt.cores
    ):
        tf.debugging.assert_near(
            actual.values,
            (p_core.values - m_core.values) / (2.0 * h),
            atol=8e-6,
            rtol=8e-6,
        )
