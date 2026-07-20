from __future__ import annotations

import pytest
import tensorflow as tf

import bayesfilter.highdim as highdim
import bayesfilter.highdim.fitting as fitting
import bayesfilter.highdim.source_route as source_route


def _convention() -> highdim.MeasureConvention:
    return highdim.MeasureConvention(
        density_measure=highdim.DensityMeasure.REFERENCE_MEASURE,
        mass_measure=highdim.MassMeasure.REFERENCE_MEASURE,
        reference_weight_name="omega",
    )


def _product_basis(degrees: tuple[int, ...]) -> highdim.ProductBasis:
    return highdim.ProductBasis(
        [
            highdim.LegendreBasis1D(highdim.BoundedInterval(-1.0, 1.0), degree)
            for degree in degrees
        ],
        _convention(),
    )


def _grid2() -> tf.Tensor:
    return tf.constant(
        [
            [-0.75, -0.50],
            [-0.75, 0.25],
            [-0.25, -0.50],
            [-0.25, 0.25],
            [0.25, -0.50],
            [0.25, 0.25],
            [0.75, -0.50],
            [0.75, 0.25],
        ],
        dtype=tf.float64,
    )


def _config(
    ranks: tuple[int, ...],
    sweep_order: tuple[int, ...] | None = None,
    ridge: float = 1e-12,
    max_sweeps: int = 1,
    holdout_tolerance: float = 1e-10,
    condition_number_veto: float = 1e14,
    row_budget: int = 10_000,
    column_budget: int = 1_000,
    dense_matrix_byte_budget: int = 10_000_000,
    normal_matrix_byte_budget: int = 10_000_000,
    stabilization_policy_id: str = fitting._STABILIZATION_POLICY_ID,
    solver_backend: str = "tensorflow.linalg.lstsq(fast=False)",
    column_scale_floor: float = fitting._DEFAULT_COLUMN_SCALE_FLOOR,
) -> highdim.FixedTTFitConfig:
    dimension = len(ranks) - 1
    return highdim.FixedTTFitConfig(
        ranks=ranks,
        ridge=ridge,
        max_sweeps=max_sweeps,
        sweep_order=tuple(range(dimension)) if sweep_order is None else sweep_order,
        row_budget=row_budget,
        column_budget=column_budget,
        dense_matrix_byte_budget=dense_matrix_byte_budget,
        normal_matrix_byte_budget=normal_matrix_byte_budget,
        condition_number_warning=1e10,
        condition_number_veto=condition_number_veto,
        holdout_tolerance=holdout_tolerance,
        stabilization_policy_id=stabilization_policy_id,
        solver_backend=solver_backend,
        column_scale_floor=column_scale_floor,
    )


def _rank_one_truth(product: highdim.ProductBasis) -> highdim.FunctionalTT:
    return highdim.FunctionalTT(
        [
            highdim.TTCore(tf.constant([[[0.75], [0.25]]], dtype=tf.float64)),
            highdim.TTCore(tf.constant([[[1.25], [-0.125]]], dtype=tf.float64)),
        ],
        product,
        _convention(),
    )


def _rank_two_truth(product: highdim.ProductBasis) -> highdim.FunctionalTT:
    return highdim.FunctionalTT(
        [
            highdim.TTCore(
                tf.constant(
                    [
                        [[1.0, 0.0], [0.0, 1.0]],
                    ],
                    dtype=tf.float64,
                )
            ),
            highdim.TTCore(
                tf.constant(
                    [
                        [[0.75], [0.20]],
                        [[-0.35], [0.60]],
                    ],
                    dtype=tf.float64,
                )
            ),
        ],
        product,
        _convention(),
    )


def _samples_from_truth(
    truth: highdim.FunctionalTT,
    points: tf.Tensor,
) -> highdim.FixedTTFitSampleBatch:
    return highdim.FixedTTFitSampleBatch(
        points=points,
        target_values=truth.evaluate(points),
        weights=tf.ones([int(points.shape[0])], dtype=tf.float64),
    )


def test_fixed_fit_rank_one_separable_target_exact():
    product = _product_basis((1, 1))
    truth = _rank_one_truth(product)
    points = _grid2()
    samples = _samples_from_truth(truth, points)
    initial_cores = (
        highdim.TTCore(tf.constant([[[0.1], [0.2]]], dtype=tf.float64)),
        truth.cores[1],
    )

    result = highdim.FixedTTFitter().fit(
        product,
        samples,
        _config((1, 1, 1), max_sweeps=2),
        initial_cores,
        branch_seed="rank-one",
        measure_convention=_convention(),
    )

    assert result.status is highdim.HighDimStatus.OK
    tf.debugging.assert_near(result.fitted_tt.evaluate(points), samples.target_values, atol=1e-10)
    tf.debugging.assert_near(result.fit_residual, tf.constant(0.0, dtype=tf.float64), atol=1e-10)


def test_fixed_fit_known_rank_two_bivariate_target():
    product = _product_basis((1, 1))
    truth = _rank_two_truth(product)
    points = _grid2()
    samples = _samples_from_truth(truth, points)
    initial_cores = (
        highdim.TTCore(
            tf.constant(
                [
                    [[0.5, 0.1], [0.1, 0.5]],
                ],
                dtype=tf.float64,
            )
        ),
        truth.cores[1],
    )

    result = highdim.FixedTTFitter().fit(
        product,
        samples,
        _config((1, 2, 1), max_sweeps=2),
        initial_cores,
        branch_seed="rank-two",
        measure_convention=_convention(),
    )

    assert result.status is highdim.HighDimStatus.OK
    tf.debugging.assert_near(result.fitted_tt.evaluate(points), samples.target_values, atol=1e-10)
    tf.debugging.assert_near(result.fit_residual, tf.constant(0.0, dtype=tf.float64), atol=1e-10)


def test_environment_recomputed_after_each_core_update():
    product = _product_basis((1, 1))
    truth = _rank_one_truth(product)
    samples = _samples_from_truth(truth, _grid2())
    result = highdim.FixedTTFitter().fit(
        product,
        samples,
        _config((1, 1, 1), max_sweeps=1),
        [
            highdim.TTCore(tf.constant([[[0.1], [0.2]]], dtype=tf.float64)),
            truth.cores[1],
        ],
        branch_seed="environment",
        measure_convention=_convention(),
    )

    hashes = result.diagnostics["environment_rebuild_hashes"]
    assert result.diagnostics["environment_rebuild_count"] == 2
    assert hashes[0] != hashes[1]


def test_design_matrix_shape_and_vectorization_order():
    product = _product_basis((1, 1))
    points = tf.constant([[-0.5, 0.25], [0.75, -0.25]], dtype=tf.float64)
    cores = (
        highdim.TTCore(tf.ones([1, 2, 2], dtype=tf.float64)),
        highdim.TTCore(
            tf.constant(
                [
                    [[2.0], [3.0]],
                    [[5.0], [7.0]],
                ],
                dtype=tf.float64,
            )
        ),
    )
    samples = highdim.FixedTTFitSampleBatch(
        points=points,
        target_values=tf.ones([2], dtype=tf.float64),
        weights=tf.ones([2], dtype=tf.float64),
    )
    system = highdim.FixedTTFitter().build_core_update_system(
        product,
        points,
        samples.target_values,
        samples.weights,
        cores,
        core_index=0,
        config=_config((1, 2, 1)),
    )

    psi0 = product.evaluate_axis(0, points[:, 0])
    psi1 = product.evaluate_axis(1, points[:, 1])
    right0 = 2.0 * psi1[:, 0] + 3.0 * psi1[:, 1]
    right1 = 5.0 * psi1[:, 0] + 7.0 * psi1[:, 1]
    expected = tf.stack(
        [
            psi0[:, 0] * right0,
            psi0[:, 0] * right1,
            psi0[:, 1] * right0,
            psi0[:, 1] * right1,
        ],
        axis=1,
    )

    assert system.design_matrix.shape == (2, 4)
    tf.debugging.assert_near(system.design_matrix, expected, atol=1e-12)


def test_normal_equation_solution_matches_direct_dense_reference():
    product = _product_basis((1, 1))
    truth = _rank_two_truth(product)
    points = _grid2()
    samples = _samples_from_truth(truth, points)
    initial_cores = (truth.cores[0], truth.cores[1])
    config = _config((1, 2, 1), max_sweeps=1)
    system = highdim.FixedTTFitter().build_core_update_system(
        product,
        points,
        samples.target_values,
        samples.weights,
        initial_cores,
        core_index=1,
        config=config,
    )
    direct_solution = tf.linalg.solve(
        system.normal_matrix,
        tf.reshape(system.rhs, [-1, 1]),
    )[:, 0]
    result = highdim.FixedTTFitter().fit(
        product,
        samples,
        config,
        initial_cores,
        branch_seed="normal-equation",
        measure_convention=_convention(),
    )

    fitted_core1 = tf.reshape(result.fitted_tt.cores[1].values, [-1])
    tf.debugging.assert_near(fitted_core1, direct_solution, atol=1e-12)


def test_deterministic_replay_same_manifest_same_values():
    product = _product_basis((1, 1))
    truth = _rank_one_truth(product)
    samples = _samples_from_truth(truth, _grid2())
    config = _config((1, 1, 1), max_sweeps=2)
    initial_cores = (
        highdim.TTCore(tf.constant([[[0.1], [0.2]]], dtype=tf.float64)),
        truth.cores[1],
    )

    first = highdim.FixedTTFitter().fit(
        product,
        samples,
        config,
        initial_cores,
        branch_seed=17,
        measure_convention=_convention(),
    )
    second = highdim.FixedTTFitter().fit(
        product,
        samples,
        config,
        initial_cores,
        branch_seed=17,
        measure_convention=_convention(),
    )

    assert first.branch_hash == second.branch_hash
    tf.debugging.assert_near(
        first.fitted_tt.evaluate(samples.points),
        second.fitted_tt.evaluate(samples.points),
        atol=0.0,
    )
    reloaded = highdim.FunctionalTT(
        second.fitted_tt.cores,
        product,
        _convention(),
        branch_identity=second.branch_identity,
    )
    tf.debugging.assert_near(reloaded.evaluate(samples.points), second.fitted_tt.evaluate(samples.points))


def test_branch_manifest_contains_required_fixed_fit_fields():
    product = _product_basis((1, 1))
    truth = _rank_one_truth(product)
    samples = _samples_from_truth(truth, _grid2())
    result = highdim.FixedTTFitter().fit(
        product,
        samples,
        _config((1, 1, 1), max_sweeps=1),
        [
            highdim.TTCore(tf.constant([[[0.1], [0.2]]], dtype=tf.float64)),
            truth.cores[1],
        ],
        branch_seed="manifest",
        measure_convention=_convention(),
    )

    payload = result.branch_identity.manifest.payload
    expected_top_level = {
        "product_basis",
        "measure_convention",
        "samples",
        "sample_hashes",
        "ranks",
        "ridge",
        "dtype",
        "sweep_order",
        "max_sweeps",
        "initial_core_hash",
        "initialization_rule",
        "per_core_update_statuses",
        "environment_rebuild_hashes",
        "stabilization_choices",
        "complexity_budgets",
        "solver_backend",
        "deterministic_seed",
        "termination_reason",
        "stop_condition_triggered",
    }
    assert expected_top_level.issubset(set(payload))
    assert {
        "sample_points_hash",
        "target_values_hash",
        "weights_hash",
        "holdout_hash",
    }.issubset(set(payload["sample_hashes"]))
    assert payload["per_core_update_statuses"][0]["condition_number"] is not None


def test_branch_hash_changes_for_samples_ridge_ranks_sweep_order():
    product = _product_basis((1, 1))
    truth = _rank_one_truth(product)
    samples = _samples_from_truth(truth, _grid2())
    changed_samples = highdim.FixedTTFitSampleBatch(
        points=samples.points + tf.constant([[0.01, 0.0]], dtype=tf.float64),
        target_values=truth.evaluate(samples.points + tf.constant([[0.01, 0.0]], dtype=tf.float64)),
        weights=samples.weights,
    )
    base_initial = (
        highdim.TTCore(tf.constant([[[0.1], [0.2]]], dtype=tf.float64)),
        truth.cores[1],
    )
    base = highdim.FixedTTFitter().fit(
        product,
        samples,
        _config((1, 1, 1), max_sweeps=1),
        base_initial,
        branch_seed="hash",
        measure_convention=_convention(),
    )
    changed_sample_result = highdim.FixedTTFitter().fit(
        product,
        changed_samples,
        _config((1, 1, 1), max_sweeps=1),
        base_initial,
        branch_seed="hash",
        measure_convention=_convention(),
    )
    changed_ridge_result = highdim.FixedTTFitter().fit(
        product,
        samples,
        _config((1, 1, 1), ridge=1e-8, max_sweeps=1),
        base_initial,
        branch_seed="hash",
        measure_convention=_convention(),
    )
    changed_order_result = highdim.FixedTTFitter().fit(
        product,
        samples,
        _config((1, 1, 1), sweep_order=(1, 0), max_sweeps=1),
        base_initial,
        branch_seed="hash",
        measure_convention=_convention(),
    )
    rank_two_truth = _rank_two_truth(product)
    rank_changed_result = highdim.FixedTTFitter().fit(
        product,
        _samples_from_truth(rank_two_truth, _grid2()),
        _config((1, 2, 1), max_sweeps=1),
        rank_two_truth.cores,
        branch_seed="hash",
        measure_convention=_convention(),
    )

    hashes = {
        base.branch_hash.value,
        changed_sample_result.branch_hash.value,
        changed_ridge_result.branch_hash.value,
        changed_order_result.branch_hash.value,
        rank_changed_result.branch_hash.value,
    }
    assert len(hashes) == 5


def test_trivariate_coupled_example_reports_coordinate_order_sensitivity():
    product = _product_basis((1, 1, 1))
    cores = [
        highdim.TTCore(tf.constant([[[1.0, 0.0], [0.0, 1.0]]], dtype=tf.float64)),
        highdim.TTCore(
            tf.constant(
                [
                    [[1.0, 0.2], [0.3, 0.4]],
                    [[-0.2, 0.7], [0.5, -0.1]],
                ],
                dtype=tf.float64,
            )
        ),
        highdim.TTCore(tf.constant([[[0.8], [0.1]], [[-0.4], [0.6]]], dtype=tf.float64)),
    ]
    truth = highdim.FunctionalTT(cores, product, _convention())
    points = tf.constant(
        [
            [-0.75, -0.50, -0.25],
            [-0.75, 0.25, 0.75],
            [-0.25, -0.50, 0.75],
            [-0.25, 0.25, -0.25],
            [0.25, -0.50, -0.25],
            [0.25, 0.25, 0.75],
            [0.75, -0.50, 0.75],
            [0.75, 0.25, -0.25],
        ],
        dtype=tf.float64,
    )
    samples = _samples_from_truth(truth, points)
    forward = highdim.FixedTTFitter().fit(
        product,
        samples,
        _config((1, 2, 2, 1), sweep_order=(0, 1, 2), max_sweeps=1),
        cores,
        branch_seed="coord-order",
        measure_convention=_convention(),
    )
    reverse = highdim.FixedTTFitter().fit(
        product,
        samples,
        _config((1, 2, 2, 1), sweep_order=(2, 1, 0), max_sweeps=1),
        cores,
        branch_seed="coord-order",
        measure_convention=_convention(),
    )

    assert forward.diagnostics["coordinate_order"] == (0, 1, 2)
    assert reverse.diagnostics["coordinate_order"] == (0, 1, 2)
    assert forward.diagnostics["sweep_order"] == (0, 1, 2)
    assert reverse.diagnostics["sweep_order"] == (2, 1, 0)
    assert forward.diagnostics["coordinate_order_sensitivity_reported"] is True
    assert forward.branch_hash != reverse.branch_hash


def test_p70_seeded_channel_initializer_creates_nonzero_extra_paths():
    cores = source_route._source_route_seeded_channel_initial_cores(
        ranks=(1, 2, 2, 1),
        basis_dim=2,
        constant_value=tf.constant(3.0, dtype=tf.float64),
    )

    tf.debugging.assert_near(cores[0].values[0, 0, 0], tf.constant(3.0, dtype=tf.float64))
    tf.debugging.assert_near(cores[1].values[0, 0, 0], tf.constant(1.0, dtype=tf.float64))
    tf.debugging.assert_near(cores[2].values[0, 0, 0], tf.constant(1.0, dtype=tf.float64))
    assert float(cores[0].values[0, 1, 1].numpy()) > 0.0
    assert float(cores[1].values[1, 1, 1].numpy()) > 0.0
    assert float(cores[2].values[1, 1, 0].numpy()) > 0.0

    activity = source_route._p70_channel_activity_diagnostics(
        cores=cores,
        target_dim=3,
        fit_rank=2,
    )

    assert activity["status"] == "ok"
    assert activity["extra_channel_active_bond_counts"][1] == 2
    assert activity["inactive_extra_channels"] == ()


def test_p70_constant_path_rank_two_fails_channel_activity_predicate():
    cores = source_route._source_route_constant_path_initial_cores(
        ranks=(1, 2, 2, 1),
        basis_dim=2,
        constant_value=tf.constant(3.0, dtype=tf.float64),
    )

    activity = source_route._p70_channel_activity_diagnostics(
        cores=cores,
        target_dim=3,
        fit_rank=2,
    )

    assert activity["status"] == "rank_channel_activity_failed"
    assert activity["extra_channel_active_bond_counts"][1] == 0
    assert activity["inactive_extra_channels"] == (1,)


def test_p70_row_adequacy_has_hard_and_preferred_tiers():
    blocked = source_route._p70_row_adequacy_diagnostics(
        row_count=7,
        target_dim=36,
        fit_degree=1,
        fit_rank=2,
    )
    diagnostic = source_route._p70_row_adequacy_diagnostics(
        row_count=9,
        target_dim=36,
        fit_degree=1,
        fit_rank=2,
    )
    passed = source_route._p70_row_adequacy_diagnostics(
        row_count=36,
        target_dim=36,
        fit_degree=1,
        fit_rank=2,
    )

    assert blocked["status"] == "branch_fit_row_adequacy_failed"
    assert diagnostic["status"] == "diagnostic_only_below_preferred_rows"
    assert passed["status"] == "ok"
    assert blocked["n_hard"] == 9
    assert blocked["n_preferred"] == 36


def test_p70_fixed_fitting_policy_payload_records_thresholds_and_nonclaims():
    row = source_route._p70_row_adequacy_diagnostics(
        row_count=36,
        target_dim=36,
        fit_degree=1,
        fit_rank=2,
    )
    cores = source_route._source_route_seeded_channel_initial_cores(
        ranks=(1, 2, 2, 1),
        basis_dim=2,
        constant_value=tf.constant(3.0, dtype=tf.float64),
    )
    activity = source_route._p70_channel_activity_diagnostics(
        cores=cores,
        target_dim=3,
        fit_rank=2,
    )

    payload = source_route._p70_fixed_fitting_policy_payload(
        target_dim=36,
        fit_degree=1,
        fit_rank=2,
        row_adequacy=row,
        channel_activity=activity,
    )

    assert payload["initialization_rule"] == source_route.P70_FIXED_BRANCH_INITIALIZATION_RULE
    assert payload["sweep_order"] == source_route._p70_canonical_alternating_sweep_order(36)
    assert payload["max_sweeps"] == source_route.P70_FIXED_BRANCH_MAX_SWEEPS
    assert payload["row_adequacy"]["status"] == "ok"
    assert payload["channel_activity"]["status"] == "ok"
    assert payload["threshold_role"] == "bayesfilter_fixed_hmc_engineering_safeguards"
    assert "not Zhao-Cui source-faithful theory" in payload["nonclaims"]


def test_canonical_repeated_axis_schedule_is_accepted_and_recorded():
    product = _product_basis((1, 1, 1))
    truth = highdim.FunctionalTT(
        [
            highdim.TTCore(tf.constant([[[1.0], [0.1]]], dtype=tf.float64)),
            highdim.TTCore(tf.constant([[[0.9], [0.2]]], dtype=tf.float64)),
            highdim.TTCore(tf.constant([[[1.1], [-0.1]]], dtype=tf.float64)),
        ],
        product,
        _convention(),
    )
    points = tf.constant(
        [
            [-0.75, -0.75, -0.75],
            [-0.75, 0.25, 0.25],
            [-0.25, -0.25, 0.75],
            [0.25, 0.75, -0.25],
            [0.75, -0.75, 0.75],
            [0.75, 0.25, -0.75],
        ],
        dtype=tf.float64,
    )
    order = source_route._p70_canonical_alternating_sweep_order(3)

    result = highdim.FixedTTFitter().fit(
        product,
        _samples_from_truth(truth, points),
        _config((1, 1, 1, 1), sweep_order=order, max_sweeps=1),
        truth.cores,
        branch_seed="p70-canonical-schedule",
        measure_convention=_convention(),
    )

    assert result.status is highdim.HighDimStatus.OK
    assert result.diagnostics["sweep_order"] == (0, 1, 2, 2, 1, 0)
    assert result.branch_identity.manifest.payload["sweep_order"] == (0, 1, 2, 2, 1, 0)


def test_legacy_permutation_schedules_remain_valid():
    product = _product_basis((1, 1))
    truth = _rank_one_truth(product)
    points = _grid2()

    result = highdim.FixedTTFitter().fit(
        product,
        _samples_from_truth(truth, points),
        _config((1, 1, 1), sweep_order=(1, 0), max_sweeps=1),
        truth.cores,
        branch_seed="legacy-permutation",
        measure_convention=_convention(),
    )

    assert result.status is highdim.HighDimStatus.OK
    assert result.diagnostics["sweep_order"] == (1, 0)


@pytest.mark.parametrize(
    "order",
    [
        (),
        (0, 1),
        (0, 1, 2, 2, 1),
        (0, 1, 2, 2, 2, 1, 0),
        (0, 1, 2, 3, 2, 1, 0),
        (0, 1, 2, 1, 0, 2),
    ],
)
def test_malformed_repeated_axis_schedules_are_rejected(order):
    product = _product_basis((1, 1, 1))
    points = tf.constant(
        [
            [-0.75, -0.75, -0.75],
            [0.0, 0.0, 0.0],
            [0.75, 0.75, 0.75],
        ],
        dtype=tf.float64,
    )
    samples = highdim.FixedTTFitSampleBatch(
        points=points,
        target_values=tf.ones([3], dtype=tf.float64),
        weights=tf.ones([3], dtype=tf.float64),
    )
    cores = [
        highdim.TTCore(tf.ones([1, 2, 1], dtype=tf.float64)),
        highdim.TTCore(tf.ones([1, 2, 1], dtype=tf.float64)),
        highdim.TTCore(tf.ones([1, 2, 1], dtype=tf.float64)),
    ]

    with pytest.raises(ValueError, match="sweep_order"):
        highdim.FixedTTFitter().fit(
            product,
            samples,
            _config((1, 1, 1, 1), sweep_order=order, max_sweeps=1),
            cores,
            branch_seed="bad-schedule",
            measure_convention=_convention(),
        )


def test_holdout_residual_veto_for_under_ranked_target():
    product = _product_basis((1, 1))
    rank_two_truth = _rank_two_truth(product)
    training = tf.constant([[-0.75, -0.75], [-0.25, 0.25], [0.25, -0.25], [0.75, 0.75]], dtype=tf.float64)
    holdout = tf.constant([[-0.75, 0.75], [-0.25, -0.25], [0.25, 0.25], [0.75, -0.75]], dtype=tf.float64)
    samples = highdim.FixedTTFitSampleBatch(
        points=training,
        target_values=rank_two_truth.evaluate(training),
        weights=tf.ones([4], dtype=tf.float64),
        holdout_points=holdout,
        holdout_values=rank_two_truth.evaluate(holdout),
        holdout_weights=tf.ones([4], dtype=tf.float64),
    )

    result = highdim.FixedTTFitter().fit(
        product,
        samples,
        _config((1, 1, 1), max_sweeps=2, holdout_tolerance=1e-14),
        [
            highdim.TTCore(tf.constant([[[1.0], [0.1]]], dtype=tf.float64)),
            highdim.TTCore(tf.constant([[[1.0], [0.1]]], dtype=tf.float64)),
        ],
        branch_seed="holdout",
        measure_convention=_convention(),
    )

    assert result.status is highdim.HighDimStatus.HOLDOUT_RESIDUAL_VETO
    assert result.stop_condition_triggered == highdim.HighDimStatus.HOLDOUT_RESIDUAL_VETO.value
    assert result.holdout_residual > 1e-14


def test_fixed_fit_complexity_gate_is_covered_in_fit_file():
    product = _product_basis((1, 1))
    samples = highdim.FixedTTFitSampleBatch(
        points=tf.constant([[-0.5, -0.5], [0.5, 0.5]], dtype=tf.float64),
        target_values=tf.ones([2], dtype=tf.float64),
        weights=tf.ones([2], dtype=tf.float64),
    )

    result = highdim.FixedTTFitter().fit(
        product,
        samples,
        _config((1, 2, 1), row_budget=1),
        [
            highdim.TTCore(tf.ones([1, 2, 2], dtype=tf.float64)),
            highdim.TTCore(tf.ones([2, 2, 1], dtype=tf.float64)),
        ],
        branch_seed="fit-file-complexity",
        measure_convention=_convention(),
    )

    assert result.status is highdim.HighDimStatus.COMPLEXITY_GATE
    assert result.stop_condition_triggered == highdim.HighDimStatus.COMPLEXITY_GATE.value


def test_fixed_fit_condition_number_veto_is_covered_in_fit_file():
    product = _product_basis((1,))
    samples = highdim.FixedTTFitSampleBatch(
        points=tf.constant([[0.0], [0.0]], dtype=tf.float64),
        target_values=tf.ones([2], dtype=tf.float64),
        weights=tf.ones([2], dtype=tf.float64),
    )

    result = highdim.FixedTTFitter().fit(
        product,
        samples,
        _config((1, 1), ridge=0.0, condition_number_veto=10.0),
        [highdim.TTCore(tf.ones([1, 2, 1], dtype=tf.float64))],
        branch_seed="fit-file-condition",
        measure_convention=_convention(),
    )

    assert result.status is highdim.HighDimStatus.CONDITION_NUMBER_VETO
    assert result.stop_condition_triggered == highdim.HighDimStatus.CONDITION_NUMBER_VETO.value


def test_scaled_augmented_ridge_matches_direct_solution_on_imbalanced_weighted_design():
    design = tf.constant(
        [
            [1.0, -5.0e-3, 2.5e3],
            [1.0, 0.0, -2.5e3],
            [1.0, 5.0e-3, 7.5e3],
            [1.0, 1.0e-2, -5.0e3],
            [1.0, -1.0e-2, 5.0e3],
        ],
        dtype=tf.float64,
    )
    target = tf.constant([0.7, 1.1, 1.5, 1.9, 0.2], dtype=tf.float64)
    weights = tf.constant([1.0, 0.125, 2.0, 0.5, 3.0], dtype=tf.float64)
    ridge = 2.5e-3
    normal, rhs = fitting._normal_equations(design, target, weights, ridge)
    scales, _, _ = fitting._weighted_column_scales(
        design,
        weights,
        fitting._DEFAULT_COLUMN_SCALE_FLOOR,
    )
    scaled_design = design / scales[tf.newaxis, :]
    transformed_normal = tf.matmul(
        scaled_design,
        scaled_design * weights[:, tf.newaxis],
        transpose_a=True,
    ) + ridge * tf.linalg.diag(1.0 / tf.square(scales))
    transformed_rhs = tf.linalg.matvec(scaled_design, weights * target, transpose_a=True)
    transformed_solution = tf.linalg.solve(
        transformed_normal,
        tf.reshape(transformed_rhs, [-1, 1]),
    )[:, 0] / scales
    scaled_normal = tf.matmul(
        scaled_design,
        scaled_design * weights[:, tf.newaxis],
        transpose_a=True,
    ) + ridge * tf.eye(int(design.shape[1]), dtype=tf.float64)
    scaled_rhs = tf.linalg.matvec(scaled_design, weights * target, transpose_a=True)
    isotropic_scaled_solution = tf.linalg.solve(
        scaled_normal,
        tf.reshape(scaled_rhs, [-1, 1]),
    )[:, 0] / scales

    stable = fitting._solve_scaled_augmented_ridge(
        design=design,
        target_values=target,
        weights=weights,
        ridge=ridge,
    )
    stable_objective = tf.reduce_sum(
        weights * tf.square(tf.linalg.matvec(design, stable.solution) - target)
    ) + ridge * tf.reduce_sum(tf.square(stable.solution))
    transformed_objective = tf.reduce_sum(
        weights * tf.square(tf.linalg.matvec(design, transformed_solution) - target)
    ) + ridge * tf.reduce_sum(tf.square(transformed_solution))

    tf.debugging.assert_near(stable.solution, transformed_solution, atol=1e-10)
    tf.debugging.assert_near(stable_objective, transformed_objective, atol=1e-12)
    isotropic_difference = float(
        tf.reduce_max(tf.abs(stable.solution - isotropic_scaled_solution)).numpy()
    )
    assert isotropic_difference > 1e-7
    optimality_residual = tf.linalg.matvec(normal, stable.solution) - rhs
    assert float(tf.norm(optimality_residual).numpy()) < 1e-5
    tf.debugging.assert_near(
        tf.linalg.matvec(design, stable.solution),
        tf.linalg.matvec(design, transformed_solution),
        atol=1e-8,
    )
    assert stable.diagnostics["solver_mode"] == (
        "objective_preserving_column_scaled_augmented_ridge"
    )
    assert stable.diagnostics["objective_preserving_column_scaling"] is True
    assert stable.diagnostics["transformed_ridge_rule"] == "rho_times_S_inverse_squared"
    assert stable.diagnostics["ridge_metric_summary"]["max_diagonal"] > (
        stable.diagnostics["ridge_metric_summary"]["min_diagonal"]
    )


def test_scaled_augmented_ridge_predictions_are_column_rescaling_invariant():
    design = tf.constant(
        [
            [1.0, -0.5, 0.25],
            [1.0, 0.0, -0.25],
            [1.0, 0.5, 0.75],
            [1.0, 1.0, -0.5],
            [1.0, -1.0, 0.5],
        ],
        dtype=tf.float64,
    )
    target = tf.constant([0.7, 1.1, 1.5, 1.9, 0.2], dtype=tf.float64)
    weights = tf.ones([5], dtype=tf.float64)
    ridge = 0.0
    scale = tf.constant([1e-3, 10.0, 1e4], dtype=tf.float64)

    base = fitting._solve_scaled_augmented_ridge(
        design=design,
        target_values=target,
        weights=weights,
        ridge=ridge,
    )
    rescaled = fitting._solve_scaled_augmented_ridge(
        design=design * scale[tf.newaxis, :],
        target_values=target,
        weights=weights,
        ridge=ridge,
    )

    tf.debugging.assert_near(
        tf.linalg.matvec(design, base.solution),
        tf.linalg.matvec(design * scale[tf.newaxis, :], rescaled.solution),
        atol=1e-10,
    )


def test_stable_overdetermined_lstsq_custom_gradient_matches_directional_fd():
    matrix = tf.constant(
        [
            [1.0, -0.5],
            [1.0, 0.0],
            [1.0, 0.75],
            [1.0, 1.25],
        ],
        dtype=tf.float64,
    )
    rhs = tf.constant([[0.2], [0.8], [1.1], [1.9]], dtype=tf.float64)
    dot_matrix = tf.constant(
        [[0.03, -0.02], [-0.01, 0.04], [0.02, 0.01], [-0.04, 0.02]],
        dtype=tf.float64,
    )
    dot_rhs = tf.constant([[0.01], [-0.02], [0.03], [0.02]], dtype=tf.float64)
    output_weight = tf.constant([[0.7], [-0.4]], dtype=tf.float64)

    with tf.GradientTape() as tape:
        tape.watch((matrix, rhs))
        solution = fitting._stable_overdetermined_lstsq(matrix, rhs)
        scalar = tf.reduce_sum(output_weight * solution)
    grad_matrix, grad_rhs = tape.gradient(scalar, (matrix, rhs))
    analytic = tf.reduce_sum(grad_matrix * dot_matrix) + tf.reduce_sum(
        grad_rhs * dot_rhs
    )

    h = tf.constant(1e-5, dtype=tf.float64)
    plus = fitting._stable_overdetermined_lstsq(
        matrix + h * dot_matrix,
        rhs + h * dot_rhs,
    )
    minus = fitting._stable_overdetermined_lstsq(
        matrix - h * dot_matrix,
        rhs - h * dot_rhs,
    )
    finite_difference = tf.reduce_sum(output_weight * (plus - minus)) / (2.0 * h)

    tf.debugging.assert_near(analytic, finite_difference, atol=2e-9, rtol=2e-8)


@pytest.mark.parametrize(
    ("design", "target", "weights", "ridge", "match"),
    [
        (
            tf.constant([[1.0, float("nan")], [0.0, 1.0]], dtype=tf.float64),
            tf.constant([1.0, 2.0], dtype=tf.float64),
            tf.ones([2], dtype=tf.float64),
            1e-6,
            "nonfinite input",
        ),
        (
            tf.constant([[1.0, 0.0], [0.0, 1.0]], dtype=tf.float64),
            tf.constant([1.0, 2.0], dtype=tf.float64),
            tf.constant([1.0, -1.0], dtype=tf.float64),
            1e-6,
            "negative weights",
        ),
        (
            tf.constant([[1.0, 0.0], [0.0, 1.0]], dtype=tf.float64),
            tf.constant([1.0, 2.0], dtype=tf.float64),
            tf.ones([2], dtype=tf.float64),
            -1e-6,
            "ridge must be finite nonnegative",
        ),
    ],
)
def test_scaled_augmented_ridge_rejects_invalid_system_inputs(
    design,
    target,
    weights,
    ridge,
    match,
):
    with pytest.raises(ValueError, match=match):
        fitting._solve_scaled_augmented_ridge(
            design=design,
            target_values=target,
            weights=weights,
            ridge=ridge,
        )


def test_fixed_fit_records_scaled_augmented_diagnostics_for_imbalanced_design():
    product = _product_basis((1, 1))
    points = _grid2()
    target = tf.sin(points[:, 0]) + 0.1 * points[:, 1] + 1.0
    samples = highdim.FixedTTFitSampleBatch(
        points=points,
        target_values=target,
        weights=tf.ones([int(points.shape[0])], dtype=tf.float64),
    )
    cores = (
        highdim.TTCore(tf.ones([1, 2, 2], dtype=tf.float64)),
        highdim.TTCore(
            tf.constant(
                [
                    [[1e-6], [1e-6]],
                    [[1e6], [1e6]],
                ],
                dtype=tf.float64,
            )
        ),
    )

    result = highdim.FixedTTFitter().fit(
        product,
        samples,
        _config((1, 2, 1), ridge=1e-10, sweep_order=(0, 1), max_sweeps=1),
        cores,
        branch_seed="scaled-augmented-diagnostics",
        measure_convention=_convention(),
    )

    record = result.core_update_statuses[0]
    assert result.status is highdim.HighDimStatus.OK
    assert record["solver_mode"] == "objective_preserving_column_scaled_augmented_ridge"
    assert record["solver_backend"] == "tensorflow.linalg.lstsq(fast=False)"
    assert record["condition_number_semantics"] == "scaled_augmented_solve_condition"
    assert record["stabilization_policy_id"] == fitting._STABILIZATION_POLICY_ID
    assert record["objective_preserving_column_scaling"] is True
    assert record["transformed_ridge_rule"] == "rho_times_S_inverse_squared"
    assert record["condition_number_gate_target"] == "scaled_augmented_solved_system"
    assert record["original_unscaled_normal_condition_role"] == "diagnostic_only"
    assert record["column_scale_hash"] is not None
    assert record["transformed_system_condition_number"] == record["condition_number"]
    assert record["unscaled_normal_condition_number"] > record["condition_number"]
    assert record["column_scale_spread"] > 1e6
    assert record["unscaled_normal_condition_veto"] is True
    assert record["status"] == highdim.HighDimStatus.OK.value
    summary = result.diagnostics["stabilization_diagnostics_summary"]
    numeric_record_conditions = [
        item["condition_number"]
        for item in result.core_update_statuses
        if isinstance(item.get("condition_number"), (int, float))
    ]
    numeric_unscaled_conditions = [
        item["unscaled_normal_condition_number"]
        for item in result.core_update_statuses
        if isinstance(item.get("unscaled_normal_condition_number"), (int, float))
    ]
    assert summary["stabilized_record_count"] >= 1
    assert summary["transformed_system_condition_number_max"] == max(numeric_record_conditions)
    assert summary["original_unscaled_normal_condition_number_max"] == max(
        numeric_unscaled_conditions
    )
    assert summary["original_unscaled_normal_condition_role"] == "diagnostic_only"
    assert summary["condition_number_gate_target"] == "scaled_augmented_solved_system"
    assert summary["column_scale_spread_max"] == record["column_scale_spread"]
    assert record["column_scale_hash"] in summary["column_scale_hashes"]
    assert summary["ridge_metric_summary"]["transformed_ridge_rule"] == (
        "rho_times_S_inverse_squared"
    )
    assert summary["ridge_metric_summary"]["max_diagonal"] >= (
        summary["ridge_metric_summary"]["min_diagonal"]
    )


def test_fixed_fit_manifest_records_stable_solver_policy():
    product = _product_basis((1, 1))
    truth = _rank_one_truth(product)
    samples = _samples_from_truth(truth, _grid2())

    result = highdim.FixedTTFitter().fit(
        product,
        samples,
        _config((1, 1, 1), max_sweeps=1),
        [
            highdim.TTCore(tf.constant([[[0.1], [0.2]]], dtype=tf.float64)),
            truth.cores[1],
        ],
        branch_seed="stable-solver-manifest",
        measure_convention=_convention(),
    )

    payload = result.branch_identity.manifest.payload
    record = payload["per_core_update_statuses"][0]
    assert payload["solver_backend"] == "tensorflow.linalg.lstsq(fast=False)"
    assert payload["stabilization_policy_id"] == fitting._STABILIZATION_POLICY_ID
    assert payload["objective_preserving_column_scaling"] is True
    assert payload["column_scale_floor"] == fitting._DEFAULT_COLUMN_SCALE_FLOOR
    assert payload["transformed_ridge_rule"] == "rho_times_S_inverse_squared"
    assert payload["condition_number_gate_target"] == "scaled_augmented_solved_system"
    assert payload["original_unscaled_normal_condition_role"] == "diagnostic_only"
    assert payload["stabilization_policy"]["stabilization_policy_id"] == (
        fitting._STABILIZATION_POLICY_ID
    )
    assert result.diagnostics["stabilization_diagnostics_summary"]["available"] is True
    assert payload["stabilization_choices"]["solver_mode"] == (
        "objective_preserving_column_scaled_augmented_ridge"
    )
    assert payload["stabilization_choices"]["scale_floor_rule"] == fitting._SCALE_FLOOR_RULE
    assert record["scale_floor_rule"] == fitting._SCALE_FLOOR_RULE
    assert record["scaled_augmented_condition_number"] == record["condition_number"]
    assert "stable solve is not a Phase 6 diagnostic pass" in record["nonclaims"]


def test_fixed_fit_branch_hash_changes_for_stabilization_policy_fields():
    product = _product_basis((1, 1))
    truth = _rank_one_truth(product)
    samples = _samples_from_truth(truth, _grid2())
    initial_cores = [
        highdim.TTCore(tf.constant([[[0.1], [0.2]]], dtype=tf.float64)),
        truth.cores[1],
    ]
    base_config = _config((1, 1, 1), max_sweeps=1)
    policy_config = _config(
        (1, 1, 1),
        max_sweeps=1,
        column_scale_floor=2.0 * fitting._DEFAULT_COLUMN_SCALE_FLOOR,
    )
    identity_config = _config(
        (1, 1, 1),
        max_sweeps=1,
        stabilization_policy_id="p71_test_policy_hash_probe_v1",
    )

    base = highdim.FixedTTFitter().fit(
        product,
        samples,
        base_config,
        initial_cores,
        branch_seed="stable-policy-hash",
        measure_convention=_convention(),
    )
    floor_changed = highdim.FixedTTFitter().fit(
        product,
        samples,
        policy_config,
        initial_cores,
        branch_seed="stable-policy-hash",
        measure_convention=_convention(),
    )
    id_changed = highdim.FixedTTFitter().fit(
        product,
        samples,
        identity_config,
        initial_cores,
        branch_seed="stable-policy-hash",
        measure_convention=_convention(),
    )

    assert base.branch_hash != floor_changed.branch_hash
    assert base.branch_hash != id_changed.branch_hash
    assert floor_changed.branch_identity.manifest.payload["column_scale_floor"] == (
        2.0 * fitting._DEFAULT_COLUMN_SCALE_FLOOR
    )
    assert id_changed.branch_identity.manifest.payload["stabilization_policy_id"] == (
        "p71_test_policy_hash_probe_v1"
    )


def test_p70_policy_payload_records_nondefault_ridge_and_stabilization_policy():
    row = source_route._p70_row_adequacy_diagnostics(
        row_count=8,
        target_dim=2,
        fit_degree=1,
        fit_rank=1,
    )
    cores = source_route._source_route_seeded_channel_initial_cores(
        ranks=(1, 1, 1),
        basis_dim=2,
        constant_value=tf.constant(1.0, dtype=tf.float64),
    )
    activity = source_route._p70_channel_activity_diagnostics(
        cores=cores,
        target_dim=2,
        fit_rank=1,
    )
    stabilization_policy = {
        "stabilization_policy_id": fitting._STABILIZATION_POLICY_ID,
        "solver_backend": "tensorflow.linalg.lstsq(fast=False)",
        "objective_preserving_column_scaling": True,
        "column_scale_floor": fitting._DEFAULT_COLUMN_SCALE_FLOOR,
        "transformed_ridge_rule": "rho_times_S_inverse_squared",
        "condition_number_gate_target": "scaled_augmented_solved_system",
        "original_unscaled_normal_condition_role": "diagnostic_only",
    }
    stabilization_summary = {
        "available": True,
        "transformed_system_condition_number_max": 3.0,
        "original_unscaled_normal_condition_number_max": 30.0,
        "original_unscaled_normal_condition_role": "diagnostic_only",
        "condition_number_gate_target": "scaled_augmented_solved_system",
        "column_scale_hashes": ("abc",),
        "ridge_metric_summary": {
            "transformed_ridge_rule": "rho_times_S_inverse_squared",
            "coordinate_system": "scaled_z_coordinates",
            "min_diagonal": 1.0e-8,
            "max_diagonal": 1.0e-4,
        },
    }

    payload = source_route._p70_fixed_fitting_policy_payload(
        target_dim=2,
        fit_degree=1,
        fit_rank=1,
        ridge=7.0e-7,
        row_adequacy=row,
        channel_activity=activity,
        stabilization_policy=stabilization_policy,
        stabilization_diagnostics_summary=stabilization_summary,
    )

    assert payload["ridge"] == 7.0e-7
    assert payload["stabilization_policy_id"] == fitting._STABILIZATION_POLICY_ID
    assert payload["solver_backend"] == "tensorflow.linalg.lstsq(fast=False)"
    assert payload["objective_preserving_column_scaling"] is True
    assert payload["column_scale_floor"] == fitting._DEFAULT_COLUMN_SCALE_FLOOR
    assert payload["transformed_ridge_rule"] == "rho_times_S_inverse_squared"
    assert payload["condition_number_gate_target"] == "scaled_augmented_solved_system"
    assert payload["original_unscaled_normal_condition_role"] == "diagnostic_only"
    assert payload["stabilization_diagnostics_summary"]["column_scale_hashes"] == ("abc",)
    assert payload["stabilization_diagnostics_summary"]["ridge_metric_summary"][
        "transformed_ridge_rule"
    ] == "rho_times_S_inverse_squared"


def test_p59_fixed_ttsirt_failed_fit_payload_honors_nondefault_ridge_and_stability():
    local_fit_points = tf.zeros([1, 4], dtype=tf.float64)
    target_values = tf.ones([4], dtype=tf.float64)
    weights = tf.ones([4], dtype=tf.float64)

    with pytest.raises(source_route.P70FixedFitDiagnosticError) as exc_info:
        source_route._p59_fixed_ttsirt_transport_from_values(
            local_fit_points=local_fit_points,
            target_values=target_values,
            fit_weights=weights,
            target_dim=1,
            fit_degree=1,
            fit_rank=1,
            ridge=0.0,
            branch_seed="p71-failed-fit-ridge-payload",
            convention=_convention(),
        )

    payload = exc_info.value.payload
    policy = payload["p70_fixed_fitting_policy"]
    first_record = payload["core_update_statuses"][0]
    assert payload["ridge"] == 0.0
    assert policy["ridge"] == 0.0
    assert payload["stabilization_policy_id"] == fitting._STABILIZATION_POLICY_ID
    assert payload["objective_preserving_column_scaling"] is True
    assert payload["condition_number_gate_target"] == "scaled_augmented_solved_system"
    assert payload["original_unscaled_normal_condition_role"] == "diagnostic_only"
    assert payload["stabilization_diagnostics_summary"]["available"] is True
    assert payload["stabilization_diagnostics_summary"][
        "transformed_system_condition_number_max"
    ] == "inf"
    assert payload["stabilization_diagnostics_summary"][
        "original_unscaled_normal_condition_number_max"
    ] == first_record["unscaled_normal_condition_number"]
    assert payload["stabilization_diagnostics_summary"]["column_scale_hashes"] == (
        first_record["column_scale_hash"],
    )
    assert payload["stabilization_diagnostics_summary"]["ridge_metric_summary"][
        "transformed_ridge_rule"
    ] == "rho_times_S_inverse_squared"
    assert policy["stabilization_policy_id"] == fitting._STABILIZATION_POLICY_ID
    assert policy["stabilization_diagnostics_summary"]["column_scale_hashes"] == (
        first_record["column_scale_hash"],
    )
    assert first_record["stabilization_policy_id"] == fitting._STABILIZATION_POLICY_ID
    assert first_record["transformed_ridge_rule"] == "rho_times_S_inverse_squared"
    assert first_record["condition_number_gate_target"] == "scaled_augmented_solved_system"
    assert first_record["status"] == highdim.HighDimStatus.CONDITION_NUMBER_VETO.value
