from __future__ import annotations

from dataclasses import fields
import json

import numpy as np
import pytest
import tensorflow as tf

import bayesfilter.inference.sequential_map_covariance as sequential
from bayesfilter.inference import (
    SEQUENTIAL_MAP_COVARIANCE_NONCLAIMS,
    SequentialMapCovarianceConfig,
    estimate_sequential_map_covariance,
)
from bayesfilter.inference.sequential_map_covariance import (
    _fit_score_curvature,
    _proposal_is_accepted,
    _proposal_score_gate,
    _score_fit_partition_indices,
)


def _quadratic_target(precision: np.ndarray, mode: np.ndarray):
    precision_tf = tf.constant(precision, dtype=tf.float64)
    mode_tf = tf.constant(mode, dtype=tf.float64)

    def value_and_score(theta: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        theta = tf.convert_to_tensor(theta, dtype=tf.float64)
        delta = theta - mode_tf
        score = -tf.linalg.matvec(precision_tf, delta)
        value = 0.5 * tf.reduce_sum(delta * score)
        return value, score

    return value_and_score


def _batched_quadratic_target(precision: np.ndarray, mode: np.ndarray):
    precision_tf = tf.constant(precision, dtype=tf.float64)
    mode_tf = tf.constant(mode, dtype=tf.float64)

    def value_and_score(theta: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        delta = tf.convert_to_tensor(theta, tf.float64) - mode_tf[None, :]
        score = -tf.einsum("ij,bj->bi", precision_tf, delta)
        return 0.5 * tf.reduce_sum(delta * score, axis=1), score

    return value_and_score


def test_proposal_score_policy_default_and_positional_prefix_are_compatible() -> None:
    prior_names = [
        "terminal_score_max_abs",
        "initial_radius",
        "search_sample_count",
        "regression_sample_count",
        "terminal_sample_count",
        "max_attempts",
        "max_exact_evaluations",
        "locator_max_iterations",
        "locator_max_line_search_iterations",
        "locator_standardized_box_radius",
        "locator_gradient_tolerance",
        "locator_stopping_condition",
        "locator_policy",
        "ridge",
        "eigenvalue_floor",
        "max_condition_number",
        "minimum_radius",
        "maximum_radius",
        "acceptance_ratio",
        "shrink_threshold",
        "expansion_threshold",
        "shrink_factor",
        "expansion_factor",
        "score_reduction_factor",
        "holdout_fraction",
        "score_holdout_relative_rmse",
        "terminal_projection_relative_frobenius_cap",
        "max_stalled_attempts",
        "refinement_geometry_policy",
        "dimension_scaled_search",
        "orthogonal_antithetic_search",
        "reuse_search_scores",
        "structured_fresh_sample_multiplier",
        "structured_max_factors",
        "structured_holdout_score_relative_rmse",
        "max_terminal_fit_attempts",
        "require_proposal_score_reduction",
        "stop_on_stalled_attempts",
        "seed",
    ]
    config = SequentialMapCovarianceConfig()
    assert [field.name for field in fields(config)][:-3] == prior_names
    assert fields(config)[-3].name == "proposal_score_acceptance_policy"
    assert fields(config)[-2].name == "record_refinement_movement_diagnostics"
    assert fields(config)[-1].name == "pair_disjoint_score_holdout"
    assert config.proposal_score_acceptance_policy == "fractional"
    assert config.record_refinement_movement_diagnostics is False
    assert config.pair_disjoint_score_holdout is False

    prior_values = [getattr(config, name) for name in prior_names]
    positional = SequentialMapCovarianceConfig(*prior_values)
    assert [getattr(positional, name) for name in prior_names] == prior_values
    assert positional.proposal_score_acceptance_policy == "fractional"
    assert positional.record_refinement_movement_diagnostics is False
    assert positional.pair_disjoint_score_holdout is False

    complete_preexisting = SequentialMapCovarianceConfig(
        *(prior_values + ["resolvable_decrease"])
    )
    assert complete_preexisting.proposal_score_acceptance_policy == (
        "resolvable_decrease"
    )
    assert complete_preexisting.record_refinement_movement_diagnostics is False
    assert complete_preexisting.pair_disjoint_score_holdout is False


def test_pair_disjoint_holdout_is_opt_in_and_preserves_whole_antithetic_pairs() -> None:
    legacy_train, legacy_holdout = _score_fit_partition_indices(
        40, 0.25, pair_disjoint=False
    )
    np.testing.assert_array_equal(legacy_train, np.arange(30))
    np.testing.assert_array_equal(legacy_holdout, np.arange(30, 40))

    train, holdout = _score_fit_partition_indices(40, 0.25, pair_disjoint=True)
    assert len(train) == 30
    assert len(holdout) == 10
    pair_count = 20
    train_pairs = {int(index % pair_count) for index in train}
    holdout_pairs = {int(index % pair_count) for index in holdout}
    assert train_pairs.isdisjoint(holdout_pairs)
    assert train_pairs | holdout_pairs == set(range(pair_count))


def test_pair_disjoint_holdout_rejects_odd_fit_counts_before_target_use() -> None:
    with pytest.raises(ValueError, match="pair-disjoint"):
        SequentialMapCovarianceConfig(
            regression_sample_count=39,
            pair_disjoint_score_holdout=True,
        )
    with pytest.raises(ValueError, match="pair-disjoint"):
        SequentialMapCovarianceConfig(
            terminal_sample_count=39,
            pair_disjoint_score_holdout=True,
        )


def test_pair_disjoint_score_fit_recovers_quadratic_with_honest_holdout() -> None:
    precision = np.array(
        [[3.0, 0.4, -0.2], [0.4, 2.0, 0.3], [-0.2, 0.3, 1.5]]
    )
    config = SequentialMapCovarianceConfig(
        regression_sample_count=40,
        terminal_sample_count=40,
        holdout_fraction=0.25,
        pair_disjoint_score_holdout=True,
    )
    fit, evaluations = _fit_score_curvature(
        _quadratic_target(precision, np.zeros(3)),
        tf.zeros(3, tf.float64),
        tf.zeros(3, tf.float64),
        tf.ones(3, tf.float64),
        dimension=3,
        radius=0.05,
        sample_count=40,
        seed=(2026, 727),
        config=config,
        evaluations=0,
        batched_value_and_score_fn=_batched_quadratic_target(
            precision, np.zeros(3)
        ),
    )

    assert evaluations == 40
    assert fit["status"] == "usable"
    assert fit["pair_disjoint_score_holdout"] is True
    assert fit["training_sample_count"] == 30
    assert fit["holdout_sample_count"] == 10
    assert fit["holdout_score_relative_rmse"] < 1.0e-6
    np.testing.assert_allclose(fit["projected_precision_z"], precision, atol=1.0e-7)


def test_proposal_score_policy_validates_before_target_evaluation() -> None:
    with pytest.raises(ValueError, match="proposal_score_acceptance_policy"):
        SequentialMapCovarianceConfig(
            proposal_score_acceptance_policy="automatic"
        )


def test_resolvable_score_gate_repairs_completed_ccma_false_rejections() -> None:
    observed = [
        (6.090779448987021, 5.869613655287451),
        (6.043467377055098, 5.933511954581472),
        (6.034981458583831, 5.979424427646117),
    ]
    for old_norm, new_norm in observed:
        fractional = _proposal_score_gate(
            old_norm,
            new_norm,
            policy="fractional",
            fractional_factor=0.95,
            active=True,
        )
        resolvable = _proposal_score_gate(
            old_norm,
            new_norm,
            policy="resolvable_decrease",
            fractional_factor=0.95,
            active=True,
        )
        assert fractional["passed"] is False
        assert resolvable["passed"] is True
        assert resolvable["legacy_fractional_passed"] is False


def test_resolvable_score_gate_rejects_equal_worse_and_subfloor_changes() -> None:
    old_norm = 6.0
    floor = np.sqrt(np.finfo(np.float64).eps) * old_norm
    for new_norm in (old_norm, old_norm + 1.0, old_norm - 0.99 * floor):
        gate = _proposal_score_gate(
            old_norm,
            new_norm,
            policy="resolvable_decrease",
            fractional_factor=0.95,
            active=True,
        )
        assert gate["passed"] is False
    gate = _proposal_score_gate(
        old_norm,
        old_norm - 1.01 * floor,
        policy="resolvable_decrease",
        fractional_factor=0.95,
        active=True,
    )
    assert gate["passed"] is True


@pytest.mark.parametrize(
    ("finite", "actual", "predicted", "rho", "score_gate_passed"),
    [
        (False, 1.0, 1.0, 1.0, True),
        (True, 0.0, 1.0, 1.0, True),
        (True, 1.0, 0.0, 1.0, True),
        (True, 1.0, 1.0, 0.09, True),
        (True, 1.0, 1.0, 1.0, False),
    ],
)
def test_proposal_acceptance_conjunction_remains_fail_closed(
    finite: bool,
    actual: float,
    predicted: float,
    rho: float,
    score_gate_passed: bool,
) -> None:
    assert _proposal_is_accepted(
        finite,
        actual=actual,
        predicted=predicted,
        rho=rho,
        acceptance_ratio=0.10,
        score_gate_passed=score_gate_passed,
    ) is False
    assert _proposal_is_accepted(
        True,
        actual=1.0,
        predicted=1.0,
        rho=0.10,
        acceptance_ratio=0.10,
        score_gate_passed=True,
    ) is True


def test_nonfinite_score_norm_fails_both_active_policies() -> None:
    for policy in ("fractional", "resolvable_decrease"):
        gate = _proposal_score_gate(
            6.0,
            float("nan"),
            policy=policy,
            fractional_factor=0.95,
            active=True,
        )
        assert gate["passed"] is False


def test_disabled_score_gate_is_policy_inert() -> None:
    outputs = [
        _proposal_score_gate(
            6.0,
            7.0,
            policy=policy,
            fractional_factor=0.95,
            active=False,
        )
        for policy in ("fractional", "resolvable_decrease")
    ]
    assert all(output["passed"] is True for output in outputs)
    assert all(output["active"] is False for output in outputs)
    assert outputs[0]["required_score_norm_max"] is None
    assert outputs[1]["required_score_norm_max"] is None


def test_disabled_score_gate_has_identical_integrated_behavior() -> None:
    precision = np.array([[2.5, 0.3], [0.3, 1.5]])
    mode = np.array([0.15, -0.08])

    def run(policy: str):
        return estimate_sequential_map_covariance(
            _quadratic_target(precision, mode),
            [np.zeros(2)],
            batched_value_and_score_fn=_batched_quadratic_target(precision, mode),
            config=SequentialMapCovarianceConfig(
                locator_policy="center_first",
                terminal_score_max_abs=1.0e-8,
                search_sample_count=8,
                regression_sample_count=18,
                terminal_sample_count=18,
                max_attempts=4,
                max_exact_evaluations=256,
                require_proposal_score_reduction=False,
                proposal_score_acceptance_policy=policy,
                seed=(2026, 724),
            ),
        )

    fractional = run("fractional")
    resolvable = run("resolvable_decrease")
    assert fractional.accepted == resolvable.accepted
    assert fractional.status == resolvable.status
    np.testing.assert_array_equal(fractional.map_candidate, resolvable.map_candidate)
    for left, right in zip(
        fractional.diagnostics["history"],
        resolvable.diagnostics["history"],
        strict=True,
    ):
        assert left["action"] == right["action"]
        assert left["radius_action"] == right["radius_action"]
        assert left["radius_after"] == right["radius_after"]
        assert left["proposal_score_gate"]["active"] is False
        assert right["proposal_score_gate"]["active"] is False


def test_omitted_policy_matches_explicit_fractional_behavior() -> None:
    precision = np.array([[2.5, 0.3], [0.3, 1.5]])
    mode = np.array([0.15, -0.08])
    shared = dict(
        locator_policy="center_first",
        terminal_score_max_abs=1.0e-8,
        search_sample_count=8,
        regression_sample_count=18,
        terminal_sample_count=18,
        max_attempts=4,
        max_exact_evaluations=256,
        seed=(2026, 725),
    )
    omitted = estimate_sequential_map_covariance(
        _quadratic_target(precision, mode),
        [np.zeros(2)],
        batched_value_and_score_fn=_batched_quadratic_target(precision, mode),
        config=SequentialMapCovarianceConfig(**shared),
    )
    explicit = estimate_sequential_map_covariance(
        _quadratic_target(precision, mode),
        [np.zeros(2)],
        batched_value_and_score_fn=_batched_quadratic_target(precision, mode),
        config=SequentialMapCovarianceConfig(
            **shared, proposal_score_acceptance_policy="fractional"
        ),
    )
    assert omitted.payload() == explicit.payload()


def test_policy_switch_preserves_transactional_center_and_radius(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def scalar(theta: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        value = tf.reshape(tf.convert_to_tensor(theta, tf.float64), [])
        delta = value - 1.0
        return -0.5 * delta**2, tf.reshape(-delta, [1])

    def batched(theta: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        values = tf.reshape(tf.convert_to_tensor(theta, tf.float64), [-1])
        delta = values - 1.0
        return -0.5 * delta**2, (-delta)[:, None]

    monkeypatch.setattr(
        sequential,
        "_antithetic_cloud",
        lambda sample_count, dimension, radius, seed: tf.zeros(
            [sample_count, dimension], tf.float64
        ),
    )
    monkeypatch.setattr(
        sequential,
        "_solve_trust_region_tf",
        lambda precision, linear, radius: {
            "step": np.array([0.01]),
            "boundary_active": False,
            "predicted_improvement": 0.00995,
        },
    )
    monkeypatch.setattr(
        sequential,
        "_fit_score_curvature",
        lambda *args, **kwargs: (
            {
                "status": "usable",
                "projected_precision_z": np.eye(1),
                "projection_relative_frobenius": 0.0,
            },
            kwargs["evaluations"],
        ),
    )

    def run(policy: str):
        return estimate_sequential_map_covariance(
            scalar,
            [np.zeros(1)],
            batched_value_and_score_fn=batched,
            config=SequentialMapCovarianceConfig(
                locator_policy="center_first",
                terminal_score_max_abs=1.0e-12,
                initial_radius=0.25,
                search_sample_count=2,
                regression_sample_count=4,
                terminal_sample_count=4,
                max_attempts=1,
                max_exact_evaluations=32,
                proposal_score_acceptance_policy=policy,
                record_refinement_movement_diagnostics=True,
                seed=(2026, 726),
            ),
        )

    fractional = run("fractional")
    resolvable = run("resolvable_decrease")
    fractional_row = fractional.diagnostics["history"][0]
    resolvable_row = resolvable.diagnostics["history"][0]

    assert fractional_row["recentered"] is False
    assert fractional_row["action"] == "proposal_rejected"
    assert fractional_row["radius_action"] == "contract"
    assert fractional_row["radius_after"] == 0.125
    np.testing.assert_array_equal(fractional.map_candidate, np.array([0.0]))
    fractional_movement = fractional_row["refinement_movement"]
    assert fractional_movement["proposal_accepted"] is False
    np.testing.assert_allclose(
        fractional_movement["evaluated_proposal_position_z"], [0.01]
    )
    np.testing.assert_allclose(fractional_movement["terminal_position_z"], [0.0])
    assert fractional_movement["terminal_value"] == -0.5
    assert fractional_movement["radius_before"] == 0.25
    assert fractional_movement["radius_after"] == 0.125

    assert resolvable_row["action"] == "proposal_accepted"
    assert resolvable_row["radius_action"] == "retain"
    assert resolvable_row["radius_after"] == 0.25
    assert resolvable_row["proposal_score_gate"]["legacy_fractional_passed"] is False
    np.testing.assert_allclose(resolvable.map_candidate, np.array([0.01]))
    resolvable_movement = resolvable_row["refinement_movement"]
    assert resolvable_movement["proposal_accepted"] is True
    np.testing.assert_allclose(resolvable_movement["terminal_position_z"], [0.01])
    assert resolvable_movement["terminal_value"] == pytest.approx(-0.5 * 0.99**2)
    assert resolvable_movement["radius_before"] == 0.25
    assert resolvable_movement["radius_after"] == 0.25
    assert resolvable.diagnostics["refinement_movement_initial"] == {
        "position_z": [0.0],
        "value": -0.5,
        "score_norm": 1.0,
    }


def test_refinement_movement_diagnostics_are_default_off() -> None:
    def run(**overrides: object):
        return estimate_sequential_map_covariance(
            _quadratic_target(np.eye(2), np.array([0.2, -0.1])),
            [np.zeros(2)],
            config=SequentialMapCovarianceConfig(
                locator_policy="center_first",
                terminal_score_max_abs=1.0e-12,
                search_sample_count=8,
                regression_sample_count=18,
                terminal_sample_count=18,
                max_attempts=1,
                max_exact_evaluations=64,
                proposal_score_acceptance_policy="resolvable_decrease",
                **overrides,
            ),
        )

    result = run()
    explicit_false = run(record_refinement_movement_diagnostics=False)
    assert result.payload() == explicit_false.payload()
    assert "refinement_movement_initial" not in result.diagnostics
    assert all(
        "refinement_movement" not in row
        for row in result.diagnostics.get("history", [])
    )


def test_rotated_quadratic_recovers_mode_and_fresh_covariance() -> None:
    rotation = np.array([[0.8, -0.6], [0.6, 0.8]])
    precision = rotation @ np.diag([2.0, 7.0]) @ rotation.T
    mode = np.array([0.35, -0.22])
    scale = np.array([0.5, 2.0])
    config = SequentialMapCovarianceConfig(
        terminal_score_max_abs=1.0e-8,
        initial_radius=0.2,
        search_sample_count=8,
        regression_sample_count=18,
        terminal_sample_count=18,
        max_attempts=4,
        max_exact_evaluations=256,
        seed=(2026, 715),
    )

    first = estimate_sequential_map_covariance(
        _quadratic_target(precision, mode),
        [np.zeros(2)],
        scale=scale,
        config=config,
    )
    second = estimate_sequential_map_covariance(
        _quadratic_target(precision, mode),
        [np.zeros(2)],
        scale=scale,
        config=config,
    )

    assert first.accepted is True
    assert first.status == "usable"
    np.testing.assert_allclose(first.map_candidate, mode, atol=1.0e-8)
    np.testing.assert_allclose(first.precision, precision, atol=1.0e-7)
    np.testing.assert_allclose(first.covariance, np.linalg.inv(precision), atol=1.0e-7)
    assert abs(first.precision[0, 1]) > 1.0
    assert first.diagnostics["terminal_fit_fresh"] is True
    assert first.diagnostics["terminal_seed"] != first.diagnostics["search_seed"]
    assert first.diagnostics["exact_evaluations"] <= config.max_exact_evaluations
    assert first.diagnostics["precision_coordinate_system"] == "theta"
    assert first.diagnostics["regression_coordinate_system"] == "z"
    assert first.payload() == second.payload()
    assert tuple(first.payload()["nonclaims"]) == SEQUENTIAL_MAP_COVARIANCE_NONCLAIMS


def test_nonstationary_locator_fails_closed_at_evaluation_budget() -> None:
    mode = np.array([0.4, -0.3])

    def quartic(theta: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        delta = tf.convert_to_tensor(theta, tf.float64) - tf.constant(mode, tf.float64)
        return -tf.reduce_sum(delta**4), -4.0 * delta**3

    result = estimate_sequential_map_covariance(
        quartic,
        [np.zeros(2)],
        config=SequentialMapCovarianceConfig(
            terminal_score_max_abs=1.0e-12,
            locator_max_iterations=1,
            max_exact_evaluations=64,
        ),
    )

    assert result.accepted is False
    assert result.status == "maximum_exact_evaluations"
    assert result.map_candidate is not None
    assert result.precision is None
    assert result.covariance is None


def test_malformed_score_fails_closed() -> None:
    def malformed(theta: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        del theta
        return tf.constant(0.0, tf.float64), tf.zeros([3], tf.float64)

    with pytest.raises(ValueError, match="one entry per parameter"):
        estimate_sequential_map_covariance(malformed, [np.zeros(2)])


def test_nonlinear_canary_recovers_after_truncated_locator() -> None:
    def rosenbrock(theta: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        x, y = tf.unstack(tf.convert_to_tensor(theta, tf.float64))
        objective = (1.0 - x) ** 2 + 20.0 * (y - x**2) ** 2
        gradient_x = -2.0 * (1.0 - x) - 80.0 * x * (y - x**2)
        gradient_y = 40.0 * (y - x**2)
        return -objective, -tf.stack([gradient_x, gradient_y])

    result = estimate_sequential_map_covariance(
        rosenbrock,
        [np.zeros(2)],
        config=SequentialMapCovarianceConfig(
            terminal_score_max_abs=2.0e-2,
            initial_radius=0.25,
            search_sample_count=16,
            regression_sample_count=48,
            terminal_sample_count=48,
            locator_max_iterations=1,
            max_attempts=8,
            max_exact_evaluations=800,
            score_holdout_relative_rmse=0.8,
            max_stalled_attempts=5,
            seed=(2026, 716),
        ),
    )

    assert result.accepted is True
    assert result.diagnostics["terminal_max_abs_scaled_score"] <= 2.0e-2
    assert result.diagnostics["history"]
    assert any(
        row.get("recentered")
        or row.get("radius_action") == "contract"
        or row.get("action") == "proposal_rejected"
        for row in result.diagnostics["history"]
    )
    assert result.diagnostics["terminal_fit"]["status"] == "usable"


def test_rank_deficient_terminal_fit_fails_closed() -> None:
    result = estimate_sequential_map_covariance(
        _quadratic_target(np.eye(3), np.zeros(3)),
        [np.zeros(3)],
        config=SequentialMapCovarianceConfig(
            terminal_score_max_abs=1.0e-8,
            terminal_sample_count=2,
            max_attempts=1,
            max_exact_evaluations=64,
        ),
    )
    assert result.accepted is False
    assert result.status == "sequential_refinement_without_terminal_geometry"


def test_terminal_projection_veto_fails_closed() -> None:
    def saddle(theta: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        x, y = tf.unstack(tf.convert_to_tensor(theta, tf.float64))
        return -x**2 + y**2, tf.stack([-2.0 * x, 2.0 * y])

    result = estimate_sequential_map_covariance(
        saddle,
        [np.zeros(2)],
        config=SequentialMapCovarianceConfig(
            terminal_score_max_abs=1.0e-8,
            terminal_sample_count=24,
            terminal_projection_relative_frobenius_cap=1.0e-6,
            max_exact_evaluations=128,
        ),
    )
    assert result.accepted is False
    assert result.status == "terminal_projection_exceeds_cap"


def test_no_finite_start_fails_closed() -> None:
    def nonfinite(theta: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        theta = tf.convert_to_tensor(theta, tf.float64)
        return tf.constant(float("nan"), tf.float64), tf.fill(
            tf.shape(theta), tf.constant(float("nan"), tf.float64)
        )

    result = estimate_sequential_map_covariance(nonfinite, [np.zeros(2)])
    assert result.accepted is False
    assert result.status == "no_finite_locator_candidate"


def test_locator_uses_start_centered_standardized_coordinates() -> None:
    visited = []

    def target(theta: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        theta = tf.convert_to_tensor(theta, tf.float64)
        visited.append(np.asarray(theta.numpy(), dtype=float))
        mode = tf.constant([1000.0, 1.0e-3], tf.float64)
        scale = tf.constant([100.0, 1.0e-4], tf.float64)
        delta = (theta - mode) / scale
        return -0.5 * tf.reduce_sum(delta**2), -delta / scale

    result = estimate_sequential_map_covariance(
        target,
        [np.array([900.0, 0.9e-3])],
        scale=np.array([100.0, 1.0e-4]),
        config=SequentialMapCovarianceConfig(
            terminal_score_max_abs=1.0e-8,
            terminal_sample_count=18,
            max_exact_evaluations=128,
        ),
    )
    assert result.accepted is True
    assert result.diagnostics["locator"][0]["coordinate_system"] == (
        "start_centered_prior_standardized_smooth_box"
    )
    start = np.array([900.0, 0.9e-3])
    scale = np.array([100.0, 1.0e-4])
    standardized = np.asarray([(row - start) / scale for row in visited])
    assert np.max(np.abs(standardized)) <= 4.0 + 1.0e-12


def test_batched_cloud_route_matches_scalar_result() -> None:
    precision = np.array([[3.0, 0.7], [0.7, 2.0]])
    mode = np.array([0.2, -0.1])
    config = SequentialMapCovarianceConfig(
        terminal_score_max_abs=1.0e-8,
        terminal_sample_count=18,
        max_exact_evaluations=128,
        seed=(2026, 718),
    )
    scalar = estimate_sequential_map_covariance(
        _quadratic_target(precision, mode), [np.zeros(2)], config=config
    )
    batched = estimate_sequential_map_covariance(
        _quadratic_target(precision, mode),
        [np.zeros(2)],
        batched_value_and_score_fn=_batched_quadratic_target(precision, mode),
        config=config,
    )
    assert scalar.accepted is True and batched.accepted is True
    np.testing.assert_allclose(batched.map_candidate, scalar.map_candidate, atol=1.0e-12)
    np.testing.assert_allclose(batched.precision, scalar.precision, atol=1.0e-10)
    np.testing.assert_allclose(batched.covariance, scalar.covariance, atol=1.0e-12)
    assert batched.diagnostics["exact_evaluations"] == scalar.diagnostics["exact_evaluations"]


def test_native_batched_locator_recovers_same_quadratic_mode() -> None:
    precision = np.array([[2.5, 0.4], [0.4, 1.5]])
    mode = np.array([0.15, -0.08])
    starts = [np.zeros(2), np.array([-0.3, 0.2])]
    config = SequentialMapCovarianceConfig(
        terminal_score_max_abs=1.0e-8,
        terminal_sample_count=18,
        max_exact_evaluations=256,
        seed=(2026, 719),
    )
    result = estimate_sequential_map_covariance(
        _quadratic_target(precision, mode),
        starts,
        batched_value_and_score_fn=_batched_quadratic_target(precision, mode),
        batched_locator_value_and_score_fn=_batched_quadratic_target(
            precision, mode
        ),
        config=config,
    )
    assert result.accepted is True
    np.testing.assert_allclose(result.map_candidate, mode, atol=1.0e-8)
    assert all(row["native_batched_locator"] for row in result.diagnostics["locator"])


def test_locator_gradient_tolerance_is_explicit_and_recorded() -> None:
    precision = np.array([[2.5, 0.4], [0.4, 1.5]])
    mode = np.array([0.15, -0.08])
    config = SequentialMapCovarianceConfig(
        terminal_score_max_abs=1.0e-8,
        locator_gradient_tolerance=0.05,
        terminal_sample_count=18,
        max_exact_evaluations=256,
        seed=(2026, 720),
    )
    result = estimate_sequential_map_covariance(
        _quadratic_target(precision, mode),
        [np.zeros(2), np.array([-0.3, 0.2])],
        batched_value_and_score_fn=_batched_quadratic_target(precision, mode),
        batched_locator_value_and_score_fn=_batched_quadratic_target(
            precision, mode
        ),
        config=config,
    )

    assert result.accepted is True
    assert result.diagnostics["terminal_max_abs_scaled_score"] <= 1.0e-8
    assert all(
        row["gradient_tolerance"] == 0.05
        for row in result.diagnostics["locator"]
    )


def test_locator_gradient_tolerance_must_be_positive_finite() -> None:
    with pytest.raises(ValueError, match="locator_gradient_tolerance"):
        SequentialMapCovarianceConfig(locator_gradient_tolerance=0.0)


def test_progress_callback_records_locator_and_terminal_stages() -> None:
    events = []
    precision = np.array([[2.5, 0.4], [0.4, 1.5]])
    mode = np.array([0.15, -0.08])
    result = estimate_sequential_map_covariance(
        _quadratic_target(precision, mode),
        [np.zeros(2), np.array([-0.3, 0.2])],
        batched_value_and_score_fn=_batched_quadratic_target(precision, mode),
        batched_locator_value_and_score_fn=_batched_quadratic_target(
            precision, mode
        ),
        config=SequentialMapCovarianceConfig(
            terminal_score_max_abs=1.0e-8,
            terminal_sample_count=18,
            max_exact_evaluations=256,
            seed=(2026, 721),
        ),
        progress_callback=events.append,
    )

    assert result.accepted is True
    stages = [event["stage"] for event in events]
    assert stages[0] == "initializer_started"
    assert "locator_objective_completed" in stages
    assert "locator_completed" in stages
    assert "candidate_selected" in stages
    assert "terminal_fit_started" in stages
    assert "terminal_fit_completed" in stages
    assert stages[-1] == "initializer_completed"
    json.dumps(events)


def test_locator_stopping_condition_is_explicit_and_validated() -> None:
    config = SequentialMapCovarianceConfig(
        locator_stopping_condition="converged_any"
    )
    assert config.locator_stopping_condition == "converged_any"
    with pytest.raises(ValueError, match="locator_stopping_condition"):
        SequentialMapCovarianceConfig(locator_stopping_condition="automatic")


def test_center_first_stationary_center_skips_locator_and_fits_geometry() -> None:
    events = []
    precision = np.array([[3.0, 0.4], [0.4, 2.0]])
    mode = np.array([0.2, -0.1])
    result = estimate_sequential_map_covariance(
        _quadratic_target(precision, mode),
        [mode],
        batched_value_and_score_fn=_batched_quadratic_target(precision, mode),
        batched_locator_value_and_score_fn=lambda theta: (_ for _ in ()).throw(
            AssertionError("center-first must not call the locator")
        ),
        config=SequentialMapCovarianceConfig(
            locator_policy="center_first",
            terminal_score_max_abs=1.0e-8,
            terminal_sample_count=18,
            max_exact_evaluations=128,
            seed=(2026, 722),
        ),
        progress_callback=events.append,
    )

    assert result.accepted is True
    np.testing.assert_allclose(result.map_candidate, mode, atol=1.0e-12)
    np.testing.assert_allclose(result.precision, precision, atol=1.0e-10)
    assert result.diagnostics["locator"] == [
        {
            "finite": True,
            "coordinate_system": "reviewed_exact_center",
            "locator_policy": "center_first",
            "locator_skipped": True,
            "skip_reason": "exact_center_admission",
        }
    ]
    assert "locator_skipped_center_first" in [event["stage"] for event in events]
    assert "locator_objective_completed" not in [event["stage"] for event in events]


def test_center_first_nonstationary_center_uses_local_refinement() -> None:
    precision = np.array([[2.5, 0.3], [0.3, 1.5]])
    mode = np.array([0.15, -0.08])
    result = estimate_sequential_map_covariance(
        _quadratic_target(precision, mode),
        [np.zeros(2)],
        batched_value_and_score_fn=_batched_quadratic_target(precision, mode),
        config=SequentialMapCovarianceConfig(
            locator_policy="center_first",
            terminal_score_max_abs=1.0e-8,
            search_sample_count=8,
            regression_sample_count=18,
            terminal_sample_count=18,
            max_attempts=4,
            max_exact_evaluations=256,
            seed=(2026, 723),
        ),
    )

    assert result.accepted is True
    np.testing.assert_allclose(result.map_candidate, mode, atol=1.0e-8)
    assert result.diagnostics["history"]


def test_center_first_requires_one_center_and_policy_is_validated() -> None:
    with pytest.raises(ValueError, match="exactly one center"):
        estimate_sequential_map_covariance(
            _quadratic_target(np.eye(2), np.zeros(2)),
            [np.zeros(2), np.ones(2)],
            config=SequentialMapCovarianceConfig(locator_policy="center_first"),
        )
    with pytest.raises(ValueError, match="locator_policy"):
        SequentialMapCovarianceConfig(locator_policy="automatic")


def test_default_locator_policy_remains_multistart() -> None:
    config = SequentialMapCovarianceConfig()
    assert config.locator_policy == "multistart"


def test_terminal_fit_attempt_cap_is_enforced() -> None:
    calls = 0

    def flat_quartic(theta: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        nonlocal calls
        calls += 1
        value = tf.convert_to_tensor(theta, tf.float64)
        # The center is stationary, but a quartic score has no usable local
        # linear curvature at that center, so every terminal fit is rejected.
        return -tf.reduce_sum(value**4), -4.0 * value**3

    result = estimate_sequential_map_covariance(
        flat_quartic,
        [np.zeros(2)],
        config=SequentialMapCovarianceConfig(
            locator_policy="center_first",
            terminal_score_max_abs=1.0e-8,
            terminal_sample_count=18,
            max_attempts=8,
            max_exact_evaluations=512,
            score_holdout_relative_rmse=1.0e-12,
            max_terminal_fit_attempts=2,
        ),
    )

    assert result.accepted is False
    terminal_rows = [
        row
        for row in result.diagnostics["history"]
        if row.get("action") == "terminal_fit_rejected"
    ]
    assert len(terminal_rows) == 2
    assert calls > 0


def test_terminal_fit_attempt_cap_must_be_positive() -> None:
    with pytest.raises(ValueError, match="max_terminal_fit_attempts"):
        SequentialMapCovarianceConfig(max_terminal_fit_attempts=0)
