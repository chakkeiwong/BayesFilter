"""Synthetic verification; NumPy is used only for fixtures and independent oracles."""

from __future__ import annotations

import json

import numpy as np
import pytest
import tensorflow as tf
import bayesfilter.inference.posterior_curvature_refinement as refinement
from bayesfilter.inference.score_curvature_tf import (
    _relative_response_rmse,
    fit_dense_score_precision_tf,
)

from bayesfilter.inference import (
    PosteriorCurvatureRefinementConfig,
    refine_posterior_local_curvature,
)


def _gaussian(mean: np.ndarray, precision: np.ndarray):
    mean_tf = tf.constant(mean, tf.float64)
    precision_tf = tf.constant(precision, tf.float64)

    def value_score(theta: tf.Tensor):
        delta = theta - mean_tf
        score = -tf.matmul(delta, precision_tf)
        value = -0.5 * tf.reduce_sum(delta * tf.matmul(delta, precision_tf), axis=1)
        return value, score

    return value_score


def _all_eligible(theta: tf.Tensor) -> tf.Tensor:
    return tf.ones(tf.shape(theta)[0], tf.bool)


def _config(**overrides) -> PosteriorCurvatureRefinementConfig:
    values = {
        "rows_per_partition": 48,
        "batch_size": 8,
        "max_physical_rows": 1000,
        "selection_relative_rmse_cap": 1.0e-8,
        "audit_relative_rmse_cap": 1.0e-8,
        "proposal_relative_rmse_cap": 1.0e-8,
        "replicate_generalized_eigenvalue_spread_cap": 1.01,
    }
    values.update(overrides)
    return PosteriorCurvatureRefinementConfig(**values)


def test_correlated_gaussian_recovers_position_covariance_and_orientation() -> None:
    mean = np.array([0.7, -0.4])
    precision = np.array([[3.0, 0.6], [0.6, 1.5]])
    covariance = np.linalg.inv(precision)
    pilot_factor = np.array([[2.0, 0.0], [0.35, 0.8]])
    result = refine_posterior_local_curvature(
        _gaussian(mean, precision),
        np.array([0.1, -0.2], dtype=np.float64),
        pilot_factor,
        batched_eligibility_fn=_all_eligible,
        config=_config(),
    )

    assert result.accepted is True
    assert result.status == "eligible_for_local_position_factor"
    np.testing.assert_allclose(result.center.numpy(), [0.1, -0.2])
    np.testing.assert_allclose(result.refined_covariance.numpy(), covariance, atol=1.0e-9)
    np.testing.assert_allclose(
        result.refined_factor.numpy() @ result.refined_factor.numpy().T,
        covariance,
        atol=1.0e-9,
    )
    assert result.diagnostics["center_held_fixed"] is True
    assert result.diagnostics["proposal_relative_rmse"] < 1.0e-8
    np.testing.assert_allclose(result.precision_z, pilot_factor.T @ precision @ pilot_factor, atol=1.0e-9)
    center_score = -precision @ (np.array([0.1, -0.2]) - mean)
    assert result.diagnostics["center_score_refined_l2"] == pytest.approx(
        np.linalg.norm(result.refined_factor.numpy().T @ center_score)
    )
    json.dumps(result.payload(), allow_nan=False)


def test_pilot_width_does_not_shrink_exact_gaussian_result() -> None:
    precision = np.diag([0.25, 4.0])
    result_small = refine_posterior_local_curvature(
        _gaussian(np.zeros(2), precision),
        np.zeros(2, np.float64),
        np.diag([0.03, 0.04]),
        batched_eligibility_fn=_all_eligible,
        config=_config(seed=17, coordinate_half_width=0.2),
    )
    result_large = refine_posterior_local_curvature(
        _gaussian(np.zeros(2), precision),
        np.zeros(2, np.float64),
        np.diag([4.0, 0.5]),
        batched_eligibility_fn=_all_eligible,
        config=_config(seed=17, coordinate_half_width=1.0),
    )
    assert result_small.accepted and result_large.accepted
    np.testing.assert_allclose(
        result_small.refined_covariance.numpy(),
        result_large.refined_covariance.numpy(),
        atol=1.0e-8,
    )


@pytest.mark.parametrize("dimension", [1, 7, 142])
def test_uniform_ball_design_is_bounded_and_reproducible(dimension) -> None:
    seed = tf.constant([918, 17], tf.int32)
    first = refinement._uniform_ball_offsets(512, dimension, seed, 1.0)
    second = refinement._uniform_ball_offsets(512, dimension, seed, 1.0)
    np.testing.assert_array_equal(first, second)
    assert bool(tf.reduce_all(tf.linalg.norm(first, axis=1) <= 1.0).numpy())
    mean_squared_radius = float(tf.reduce_mean(tf.reduce_sum(first * first, axis=1)).numpy())
    assert mean_squared_radius == pytest.approx(dimension / (dimension + 2.0), abs=0.06)


def test_uniform_ball_design_recovers_correlated_gaussian_locally() -> None:
    precision = np.array([[3.0, 0.6], [0.6, 1.5]])
    covariance = np.linalg.inv(precision)
    result = refine_posterior_local_curvature(
        _gaussian(np.array([0.4, -0.2]), precision),
        np.array([0.1, -0.3], np.float64),
        np.array([[1.2, 0.0], [0.25, 0.9]], np.float64),
        batched_eligibility_fn=_all_eligible,
        config=_config(
            fit_design="uniform_ball",
            coordinate_half_width=1.0, seed=71,
        ),
    )
    assert result.accepted
    assert result.diagnostics["fit_design"] == "uniform_ball"
    assert result.diagnostics["proposal_distribution"] == "standard_normal_in_refined_factor_coordinates"
    np.testing.assert_allclose(result.refined_covariance, covariance, atol=1.0e-9)


def test_unknown_local_geometry_design_is_rejected() -> None:
    with pytest.raises(ValueError, match="fit_design"):
        PosteriorCurvatureRefinementConfig(fit_design="unknown")


def test_ball_option_preserves_positional_arguments_and_box_default() -> None:
    config = PosteriorCurvatureRefinementConfig(0.75, 3, 48, 8)
    assert (config.coordinate_half_width, config.replicate_count,
            config.rows_per_partition, config.batch_size) == (0.75, 3, 48, 8)
    assert config.fit_design == "uniform_box"


@pytest.mark.parametrize("radius", [0.1, 1.0, 2.0])
def test_ball_radius_does_not_scale_gaussian_covariance(radius) -> None:
    precision = np.array([[4.0, 0.7], [0.7, 1.0]])
    result = refine_posterior_local_curvature(
        _gaussian(np.zeros(2), precision), np.zeros(2),
        np.array([[0.03, 0.0], [0.02, 4.0]]),
        batched_eligibility_fn=_all_eligible,
        config=_config(fit_design="uniform_ball", coordinate_half_width=radius),
    )
    assert result.accepted
    np.testing.assert_allclose(result.refined_covariance, np.linalg.inv(precision), rtol=1e-9)


def test_good_local_ball_fit_still_rejects_bad_gaussian_proposal() -> None:
    def radial_quartic(theta):
        squared_radius = tf.reduce_sum(theta**2, axis=1)
        return (-0.5 * squared_radius - 0.025 * squared_radius**2,
                -(1.0 + 0.1 * squared_radius[:, None]) * theta)

    result = refine_posterior_local_curvature(
        radial_quartic, np.zeros(24), np.eye(24),
        batched_eligibility_fn=_all_eligible,
        config=PosteriorCurvatureRefinementConfig(fit_design="uniform_ball", seed=20260912),
    )
    assert result.status == "refined_proposal_rejected"
    assert all(item["accepted"] for item in result.diagnostics["replicates"])
    assert result.diagnostics["audit_relative_rmse"] < 0.20
    assert result.diagnostics["proposal_relative_rmse"] > 0.35
    assert result.refined_factor is None and result.refined_covariance is None


def test_ball_fit_cannot_bypass_gaussian_support_veto() -> None:
    result = refine_posterior_local_curvature(
        _gaussian(np.zeros(2), np.eye(2) / 4.0), np.zeros(2), np.eye(2) * 0.01,
        batched_eligibility_fn=lambda theta: tf.reduce_all(tf.abs(theta) <= 0.05, axis=1),
        config=_config(fit_design="uniform_ball"),
    )
    assert result.status == "ineligible_target_row"
    assert result.diagnostics["failure_partition"] == "proposal"
    assert result.refined_factor is None


def test_center_is_not_reselected_when_probe_has_higher_value() -> None:
    result = refine_posterior_local_curvature(
        _gaussian(np.array([1.0]), np.array([[2.0]])),
        np.array([0.0], np.float64),
        np.array([[1.0]], np.float64),
        batched_eligibility_fn=_all_eligible,
        config=_config(rows_per_partition=32, batch_size=8),
    )
    assert result.accepted
    np.testing.assert_allclose(result.center.numpy(), [0.0])
    assert result.diagnostics["center_held_fixed"] is True


@pytest.mark.parametrize(
    ("factor", "message"),
    [
        (np.array([[1.0, 1.0], [0.0, 1.0]]), "lower triangular"),
        (np.array([[0.0, 0.0], [0.0, 1.0]]), "positive diagonal"),
        (np.array([[1.0, 0.0], [0.0, np.nan]]), "finite"),
    ],
)
def test_invalid_pilot_factor_fails_before_target_call(factor: np.ndarray, message: str) -> None:
    calls = []

    def target(theta: tf.Tensor):
        calls.append(tuple(theta.shape))
        return -tf.reduce_sum(theta * theta, axis=1), -2.0 * theta

    with pytest.raises(ValueError, match=message):
        refine_posterior_local_curvature(
            target,
            np.zeros(2, np.float64),
            factor,
            batched_eligibility_fn=_all_eligible,
            config=_config(),
        )
    assert calls == []


def test_ineligible_and_nonfinite_rows_are_hard_rejections() -> None:
    def eligible(theta: tf.Tensor) -> tf.Tensor:
        return theta[:, 0] <= 0.2

    result = refine_posterior_local_curvature(
        _gaussian(np.zeros(2), np.eye(2)),
        np.zeros(2, np.float64),
        np.eye(2, dtype=np.float64),
        batched_eligibility_fn=eligible,
        config=_config(seed=19),
    )
    assert result.accepted is False
    assert result.status == "ineligible_target_row"
    assert result.refined_factor is None

    def nonfinite(theta: tf.Tensor):
        value, score = _gaussian(np.zeros(2), np.eye(2))(theta)
        return tf.tensor_scatter_nd_update(value, [[0]], [tf.constant(np.inf, tf.float64)]), score

    result = refine_posterior_local_curvature(
        nonfinite,
        np.zeros(2, np.float64),
        np.eye(2, dtype=np.float64),
        batched_eligibility_fn=_all_eligible,
        config=_config(seed=23),
    )
    assert result.accepted is False
    assert result.status == "nonfinite_target_value"


def test_flat_target_rejects_non_spd_curvature() -> None:
    def flat(theta: tf.Tensor):
        return tf.zeros(tf.shape(theta)[0], tf.float64), tf.zeros_like(theta)

    result = refine_posterior_local_curvature(
        flat,
        np.zeros(2, np.float64),
        np.eye(2, dtype=np.float64),
        batched_eligibility_fn=_all_eligible,
        config=_config(seed=29),
    )
    assert result.accepted is False
    assert result.status == "curvature_fit_rejected"


def test_physical_row_budget_is_checked_before_target_calls() -> None:
    calls = []

    def target(theta: tf.Tensor):
        calls.append(1)
        return -tf.reduce_sum(theta * theta, axis=1), -2.0 * theta

    with pytest.raises(ValueError, match="max_physical_rows"):
        refine_posterior_local_curvature(
            target,
            np.zeros(2, np.float64),
            np.eye(2, dtype=np.float64),
            batched_eligibility_fn=_all_eligible,
            config=_config(max_physical_rows=10),
        )
    assert calls == []


def test_selection_failure_does_not_evaluate_audit_or_proposal() -> None:
    calls: list[tuple[int, int]] = []

    def target(theta: tf.Tensor):
        calls.append((int(theta.shape[0]), len(calls)))
        value = -tf.reduce_sum(theta * theta, axis=1)
        score = -2.0 * theta
        if len(calls) == 14:
            score = tf.tensor_scatter_nd_update(
                score, [[0, 0]], [tf.constant(np.nan, tf.float64)]
            )
        return value, score

    result = refine_posterior_local_curvature(
        target,
        np.zeros(2, np.float64),
        np.eye(2, dtype=np.float64),
        batched_eligibility_fn=_all_eligible,
        config=_config(seed=31),
    )
    assert result.accepted is False
    assert result.status == "nonfinite_target_score"
    assert result.diagnostics["logical_rows"] == 1 + 2 * 48 + 8
    assert result.diagnostics["completed_logical_rows"] == 97
    assert result.diagnostics["physical_rows"] == 112
    assert result.diagnostics["target_rows"] == 112
    assert result.diagnostics["failure_partition"] == "selection_0"
    assert result.diagnostics["logical_rows_per_partition"] == 48


def _run(target=None, *, eligibility=_all_eligible, **config):
    return refine_posterior_local_curvature(
        _gaussian(np.zeros(2), np.eye(2)) if target is None else target,
        np.zeros(2, np.float64), np.eye(2, dtype=np.float64),
        batched_eligibility_fn=eligibility,
        config=_config(rows_per_partition=16, batch_size=16, **config),
    )


def test_dense_kernel_matches_independent_lstsq_and_dynamic_graph() -> None:
    generator = np.random.default_rng(918)
    offsets = generator.normal(size=(48, 3))
    center_score = np.array([0.4, -0.8, 1.2])
    precision = np.array([[3.0, 0.2, 0.5], [0.2, 2.0, -0.1], [0.5, -0.1, 1.0]])
    scores = center_score - offsets @ precision + 0.01 * offsets**3
    coefficient = np.linalg.lstsq(offsets, center_score - scores, rcond=None)[0]
    oracle = 0.5 * (coefficient + coefficient.T)
    eager = fit_dense_score_precision_tf(center_score, offsets, scores)
    graph = tf.function(fit_dense_score_precision_tf, input_signature=[
        tf.TensorSpec([3], tf.float64), tf.TensorSpec([None, 3], tf.float64),
        tf.TensorSpec([None, 3], tf.float64),
    ])
    compiled = graph(center_score, offsets, scores)
    graph(center_score, offsets[:24], scores[:24])
    assert graph.experimental_get_tracing_count() == 1
    assert compiled["design_rank"].dtype == tf.int32
    assert compiled["raw_spd"].dtype == tf.bool
    np.testing.assert_allclose(eager["raw_precision"], oracle, atol=2e-13)
    np.testing.assert_allclose(compiled["raw_precision"], oracle, atol=2e-13)


@pytest.mark.parametrize("scale", [1e-200, 1.0, 1e200])
def test_relative_residual_is_scale_invariant_without_squaring_overflow(scale) -> None:
    offsets = tf.constant([[1.0, 0.0], [0.0, 1.0]], tf.float64)
    actual = _relative_response_rmse(
        tf.eye(2, dtype=tf.float64) * scale, tf.zeros(2, tf.float64),
        offsets, -2.0 * scale * offsets,
    )
    assert float(actual) == pytest.approx(0.5)


@pytest.mark.parametrize("design", [
    np.ones((16, 2)),
    np.column_stack((np.linspace(-1, 1, 16), np.linspace(-1, 1, 16) ** 2 * 1e-8)),
])
def test_rank_deficient_or_ill_conditioned_design_rejects_before_audit(monkeypatch, design) -> None:
    monkeypatch.setattr(refinement, "_uniform_offsets", lambda *args: tf.constant(design, tf.float64))
    result = _run()
    assert not result.accepted and result.status == "curvature_fit_rejected"
    assert result.refined_factor is None
    assert result.diagnostics["failure_partition"] == "fit"
    assert [part["role"] for part in result.diagnostics["partitions"]] == [
        "center", "training_0", "training_1", "selection_0", "selection_1",
    ]


@pytest.mark.parametrize("precision", [np.diag([1.0, -1.0]), np.diag([1.0, 1e-12])])
def test_raw_non_spd_or_excessive_precision_condition_is_not_projected(precision) -> None:
    result = _run(_gaussian(np.zeros(2), precision))
    assert result.status == "curvature_fit_rejected"
    assert result.precision_z is None and result.refined_covariance is None
    json.dumps(result.payload(), allow_nan=False)


def test_all_pairs_replicate_spread_detects_scale_changes() -> None:
    identity = tf.eye(2, dtype=tf.float64)
    spread = refinement._precision_spread((identity, 1.4 * identity, identity / 1.4))
    assert spread == pytest.approx(1.96)


def test_replicate_instability_rejects_without_audit(monkeypatch) -> None:
    monkeypatch.setattr(refinement, "_precision_spread", lambda _: 2.0)
    result = _run()
    assert result.status == "replicate_instability"
    assert result.diagnostics["callback_batches"] == 5


@pytest.mark.parametrize("bad_call,status,calls", [
    (6, "audit_rejected", 6), (7, "refined_proposal_rejected", 7),
])
@pytest.mark.parametrize("fit_design", ["uniform_box", "uniform_ball"])
def test_holdouts_veto_frozen_candidate_without_refit(monkeypatch, bad_call, status, calls, fit_design) -> None:
    events = []
    callback_count = 0
    original = refinement.fit_dense_score_precision_tf

    def fit(*args, **kwargs):
        events.append("fit")
        return original(*args, **kwargs)

    def target(theta):
        nonlocal callback_count
        callback_count += 1
        events.append(f"target_{callback_count}")
        values = -0.5 * tf.reduce_sum(theta**2, axis=1)
        return values, -theta * (2.0 if callback_count == bad_call else 1.0)

    monkeypatch.setattr(refinement, "fit_dense_score_precision_tf", fit)
    result = _run(target, fit_design=fit_design)
    assert result.status == status and result.refined_factor is None
    assert callback_count == calls
    assert events[5:8] == ["fit", "fit", "target_6"]
    assert events.count("fit") == 2


def test_proposal_is_generated_with_refined_factor_and_distinct_seed() -> None:
    batches = []
    target = _gaussian(np.zeros(2), np.array([[4.0, 1.0], [1.0, 2.0]]))

    def record(theta):
        batches.append(theta.numpy())
        return target(theta)

    result = _run(record)
    assert result.accepted
    seed = result.diagnostics["partitions"][-1]["seed"]
    latent = tf.random.stateless_normal([16, 2], seed, dtype=tf.float64).numpy()
    np.testing.assert_allclose(batches[-1], latent @ result.refined_factor.numpy().T, atol=1e-14)
    assert not np.allclose(batches[-1], latent)
    seeds = [tuple(part["seed"]) for part in result.diagnostics["partitions"][1:]]
    assert len(set(seeds)) == 6


def test_expansion_into_invalid_support_rejects_without_shrinking() -> None:
    result = refine_posterior_local_curvature(
        _gaussian(np.zeros(2), np.eye(2) / 4.0), np.zeros(2, np.float64),
        np.eye(2, dtype=np.float64) * 0.01,
        batched_eligibility_fn=lambda theta: tf.reduce_all(tf.abs(theta) <= 0.05, axis=1),
        config=_config(),
    )
    assert result.status == "ineligible_target_row"
    assert result.diagnostics["failure_partition"] == "proposal"
    assert result.refined_factor is None
    assert result.diagnostics["physical_rows"] - result.diagnostics["target_rows"] == 8


@pytest.mark.parametrize("fault,expected", [
    ("eligibility_dtype", TypeError), ("eligibility_shape", ValueError),
    ("values_dtype", TypeError), ("score_dtype", TypeError),
    ("values_shape", ValueError), ("score_shape", ValueError),
    ("callback_exception", RuntimeError),
])
def test_callback_contract_errors_propagate(fault, expected) -> None:
    def eligible(theta):
        if fault == "eligibility_dtype":
            return tf.ones(16, tf.int32)
        if fault == "eligibility_shape":
            return tf.ones((16, 1), tf.bool)
        return _all_eligible(theta)

    def target(theta):
        if fault == "callback_exception":
            raise RuntimeError("callback failed")
        values = -tf.reduce_sum(theta**2, axis=1)
        scores = -2.0 * theta
        if fault == "values_dtype":
            values = tf.cast(values, tf.float32)
        if fault == "score_dtype":
            scores = tf.cast(scores, tf.float32)
        if fault == "values_shape":
            values = values[:, None]
        if fault == "score_shape":
            scores = scores[:, 0]
        return values, scores

    with pytest.raises(expected):
        _run(target, eligibility=eligible)


@pytest.mark.parametrize("which", ["value", "score", "eligibility"])
def test_failed_center_attempts_are_accounted_and_payload_is_strict_json(which) -> None:
    def target(theta):
        values = tf.zeros(16, tf.float64)
        scores = tf.zeros_like(theta)
        return (values + float("nan"), scores) if which == "value" else (values, scores + float("inf"))

    result = _run(target, eligibility=lambda theta: tf.fill([16], which != "eligibility"))
    diagnostics = result.diagnostics
    assert diagnostics["physical_rows"] == 16
    assert diagnostics["logical_rows"] == 1
    assert diagnostics["padded_rows"] == 15
    assert diagnostics["completed_logical_rows"] == 0
    assert diagnostics["eligibility_batches"] == 1
    assert diagnostics["callback_batches"] == (0 if which == "eligibility" else 1)
    assert diagnostics["target_rows"] == (0 if which == "eligibility" else 16)
    assert diagnostics["failure_partition"] == "center"
    assert diagnostics["logical_rows_per_partition"] == 16
    assert result.refined_factor is None
    json.dumps(result.payload(), allow_nan=False)


@pytest.mark.parametrize("fault,status", [
    ("nan", "factorization_failed"), ("reconstruction", "factor_reconstruction_failed"),
])
def test_factorization_cannot_pass_nan_or_bad_reconstruction(monkeypatch, fault, status) -> None:
    monkeypatch.setattr(refinement, "_precision_spread", lambda _: 1.0)
    original = tf.linalg.cholesky
    calls = 0

    def cholesky(matrix):
        nonlocal calls
        calls += 1
        actual = original(matrix)
        if calls == 2:
            return actual * (float("nan") if fault == "nan" else 1.01)
        return actual

    monkeypatch.setattr(tf.linalg, "cholesky", cholesky)
    result = _run()
    assert result.status == status and result.refined_factor is None
    assert result.diagnostics["callback_batches"] == 6
    json.dumps(result.payload(), allow_nan=False)


def test_nonfinite_transformed_score_is_rejected() -> None:
    def target(theta):
        return tf.zeros(tf.shape(theta)[0], tf.float64), tf.ones_like(theta) * 1e308

    result = refine_posterior_local_curvature(
        target, np.zeros(2, np.float64), 10.0 * np.eye(2, dtype=np.float64),
        batched_eligibility_fn=_all_eligible, config=_config(),
    )
    assert result.status == "nonfinite_transformed_score" and not result.accepted


def test_nonfinite_position_is_not_sent_to_callbacks() -> None:
    def target(theta):
        assert bool(tf.reduce_all(tf.math.is_finite(theta)))
        return tf.zeros(tf.shape(theta)[0], tf.float64), tf.zeros_like(theta)

    result = refine_posterior_local_curvature(
        target, np.array([1e308, 0.0]), np.diag([1e308, 1.0]),
        batched_eligibility_fn=_all_eligible, config=_config(),
    )
    assert result.status == "nonfinite_position"
    assert result.diagnostics["physical_rows"] > result.diagnostics["target_rows"]


@pytest.mark.parametrize("name", ["seed", "replicate_count", "batch_size", "rows_per_partition", "max_physical_rows"])
def test_integer_config_does_not_silently_truncate(name) -> None:
    with pytest.raises(TypeError):
        PosteriorCurvatureRefinementConfig(**{name: 2.5})


@pytest.mark.parametrize("options", [{"seed": 2**31}, {"seed": -(2**31)-1},
                                        {"coordinate_half_width": float("nan")},
                                        {"lineage": {"bad": object()}},
                                        {"lineage": {"bad": float("inf")}}])
def test_invalid_seed_scale_or_lineage_is_not_silently_coerced(options) -> None:
    with pytest.raises((ValueError, TypeError)):
        PosteriorCurvatureRefinementConfig(**options)


def test_inputs_are_snapshots_not_mutated_and_center_metric_uses_refined_factor() -> None:
    center = tf.Variable([0.2, -0.1], dtype=tf.float64)
    pilot = tf.Variable([[0.01, 0.0], [0.0, 0.01]], dtype=tf.float64)
    result = refine_posterior_local_curvature(
        _gaussian(np.array([1.0, 0.0]), np.eye(2)), center, pilot,
        batched_eligibility_fn=_all_eligible, config=_config(),
    )
    assert result.accepted
    np.testing.assert_array_equal(center.numpy(), [0.2, -0.1])
    np.testing.assert_array_equal(pilot.numpy(), np.eye(2) * 0.01)
    center.assign([4.0, 5.0])
    pilot.assign(np.eye(2))
    np.testing.assert_array_equal(result.center, [0.2, -0.1])
    np.testing.assert_array_equal(result.pilot_factor, np.eye(2) * 0.01)
    assert result.diagnostics["center_score_refined_l2"] == pytest.approx(np.sqrt(0.65))


def test_tracing_public_orchestration_is_explicitly_unsupported() -> None:
    with pytest.raises(RuntimeError, match="eager execution"):
        tf.function(lambda: _run().refined_factor)()
