from __future__ import annotations

import tensorflow as tf

import bayesfilter.highdim as highdim


def _convention() -> highdim.MeasureConvention:
    return highdim.MeasureConvention(
        density_measure=highdim.DensityMeasure.REFERENCE_MEASURE,
        mass_measure=highdim.MassMeasure.REFERENCE_MEASURE,
        reference_weight_name="omega",
    )


def _coordinate_map() -> highdim.AffineCoordinateMap:
    return highdim.AffineCoordinateMap(
        offset=tf.constant([0.0], dtype=tf.float64),
        matrix=tf.constant([[6.0]], dtype=tf.float64),
    )


def _basis(dimension: int, degree: int = 6) -> highdim.ProductBasis:
    return highdim.ProductBasis(
        [
            highdim.LegendreBasis1D(
                highdim.BoundedInterval(-1.0, 1.0),
                degree,
            )
            for _ in range(dimension)
        ],
        _convention(),
    )


def _filter_config(
    dimension: int,
    *,
    order: int,
    ranks: tuple[int, ...],
    sweep_order: tuple[int, ...],
    seed: str,
) -> highdim.FixedBranchFilterConfig:
    product = _basis(dimension)
    cores = highdim.norm_balanced_initial_cores(product, ranks)
    return highdim.FixedBranchFilterConfig(
        fit_config=highdim.FixedTTFitConfig(
            ranks=ranks,
            ridge=1e-10,
            max_sweeps=2,
            sweep_order=sweep_order,
            row_budget=max(512, order**dimension),
            column_budget=128,
            dense_matrix_byte_budget=2_000_000,
            normal_matrix_byte_budget=200_000,
            condition_number_warning=1e12,
            condition_number_veto=1e16,
            holdout_tolerance=1.0,
        ),
        density_tau=0.0,
        normalizer_floor=1e-14,
        denominator_floor=1e-14,
        retained_storage_byte_budget=10_000_000,
        coordinate_maps=(_coordinate_map(),),
        measure_convention=_convention(),
        deterministic_seed=seed,
        product_basis=product,
        initial_cores=cores,
        fit_quadrature_order=order,
    )


def _adjacent_config(
    order: int = 13,
    *,
    transition_before_first_observation: bool = False,
) -> highdim.ScalarAdjacentTTConfig:
    return highdim.ScalarAdjacentTTConfig(
        initial=_filter_config(
            1,
            order=order,
            ranks=(1, 1),
            sweep_order=(0,),
            seed="phase6-test-initial",
        ),
        adjacent=_filter_config(
            2,
            order=order,
            ranks=(1, 2, 1),
            sweep_order=(0, 1, 1, 0),
            seed="phase6-test-adjacent",
        ),
        scalar_coordinate_map=_coordinate_map(),
        transition_before_first_observation=transition_before_first_observation,
    )


def _legacy_config(order: int = 13) -> highdim.FixedBranchFilterConfig:
    return _filter_config(
        1,
        order=order,
        ranks=(1, 1),
        sweep_order=(0,),
        seed="phase6-test-legacy",
    )


def _model_and_theta() -> tuple[highdim.ExactTransformedSVSSM, tf.Tensor]:
    model = highdim.ExactTransformedSVSSM(sigma=1.0)
    theta = model.unconstrained_from_physical(gamma=0.6, beta=0.4)
    return model, theta


def _transformed_observations() -> tf.Tensor:
    raw = tf.constant([[0.35], [-0.22], [0.41]], dtype=tf.float64)
    return highdim.exact_transformed_sv_observations(raw)


def test_t1_matches_legacy_one_axis_finite_program() -> None:
    model, theta = _model_and_theta()
    observations = _transformed_observations()[:1]
    adjacent = highdim.scalar_adjacent_state_fixed_tt_value(
        model,
        theta,
        observations,
        _adjacent_config(),
        fixture_id="phase6.test.t1.adjacent",
        branch_seed_prefix="phase6-test-t1",
    )
    legacy = highdim.scalar_nonlinear_fixed_design_tt_value_path(
        model,
        theta,
        observations,
        _legacy_config(),
        fixture_id="phase6.test.t1.legacy",
        initial_target_id="phase6.test.t1.initial",
        transition_target_id="phase6.test.t1.transition",
        branch_seed_prefix="phase6-test-t1",
        retained_moment_order=65,
        retained_propagation_order=65,
    )

    tf.debugging.assert_near(
        adjacent.log_likelihood,
        legacy.log_likelihood,
        atol=2e-12,
        rtol=2e-12,
    )
    assert adjacent.steps[0].diagnostics["axis_order"] == ("x_0",)
    assert adjacent.steps[0].diagnostics["integrated_axes"] == ()


def test_t2_uses_adjacent_state_fit_and_previous_axis_marginal() -> None:
    model, theta = _model_and_theta()
    result = highdim.scalar_adjacent_state_fixed_tt_value(
        model,
        theta,
        _transformed_observations()[:2],
        _adjacent_config(),
        fixture_id="phase6.test.t2.structure",
        branch_seed_prefix="phase6-test-t2-structure",
    )

    assert len(result.steps) == 2
    step = result.steps[1]
    assert step.target_kind == "adjacent_state_update"
    assert step.density.sqrt_tt.product_basis.dimension == 2
    assert step.diagnostics["axis_order"] == ("x_t", "x_t_minus_1")
    assert step.diagnostics["integrated_axes"] == (1,)
    tf.debugging.assert_near(step.marginal_mass, 1.0, atol=2e-10, rtol=2e-10)


def test_t1_transitioned_initial_uses_joint_previous_current_target() -> None:
    model = highdim.GeneralizedSVPriorMeanSSM()
    theta = tf.constant([1.0, -2.0, 0.0], dtype=tf.float64)
    observations = tf.constant([[0.25]], dtype=tf.float64)
    result = highdim.scalar_adjacent_state_fixed_tt_value(
        model,
        theta,
        observations,
        _adjacent_config(transition_before_first_observation=True),
        fixture_id="phase6.test.t1.transitioned-initial",
        branch_seed_prefix="phase6-test-t1-transitioned-initial",
    )

    assert len(result.steps) == 1
    step = result.steps[0]
    assert step.target_kind == "transitioned_initial_adjacent_state_update"
    assert step.density.sqrt_tt.product_basis.dimension == 2
    assert step.diagnostics["axis_order"] == ("x_t", "x_t_minus_1")
    assert step.diagnostics["integrated_axes"] == (1,)
    assert result.diagnostics["transition_before_first_observation"] is True
    tf.debugging.assert_near(step.marginal_mass, 1.0, atol=2e-10, rtol=2e-10)


def test_t2_total_autodiff_matches_independent_same_scalar_fd() -> None:
    model, theta = _model_and_theta()
    result = highdim.scalar_adjacent_state_fixed_tt_score(
        model,
        theta,
        _transformed_observations()[:2],
        _adjacent_config(),
        finite_difference_h=(3e-3, 1e-3),
        fixture_id="phase6.test.t2.score",
        branch_seed_prefix="phase6-test-t2-score",
    )

    assert result.score.shape == (2,)
    assert len(result.finite_difference_table.valid_rows()) == 4
    assert result.diagnostics["previous_marginal_derivative_included"] is True
    for row in result.finite_difference_table.valid_rows():
        denominator = tf.maximum(
            tf.maximum(tf.abs(row.analytic_gradient), tf.abs(row.centered_difference)),
            tf.constant(1e-12, dtype=tf.float64),
        )
        tf.debugging.assert_less(
            row.abs_error / denominator,
            tf.constant(2e-2, dtype=tf.float64),
        )


def test_compatibility_hash_rejects_realized_shift_branch_change() -> None:
    model, theta = _model_and_theta()
    observations = _transformed_observations()[:2]
    base = highdim.scalar_adjacent_state_fixed_tt_value(
        model,
        theta,
        observations,
        _adjacent_config(),
        fixture_id="phase6.test.compat.base",
        branch_seed_prefix="phase6-test-compat",
    )
    changed_observations = tf.tensor_scatter_nd_add(
        observations,
        [[1, 0]],
        [tf.constant(20.0, dtype=tf.float64)],
    )
    changed = highdim.scalar_adjacent_state_fixed_tt_value(
        model,
        theta,
        changed_observations,
        _adjacent_config(),
        fixture_id="phase6.test.compat.changed",
        branch_seed_prefix="phase6-test-compat",
    )

    assert (
        tuple(step.log_scale_shift_index for step in base.steps)
        != tuple(step.log_scale_shift_index for step in changed.steps)
    )
    assert base.compatibility_hash != changed.compatibility_hash
