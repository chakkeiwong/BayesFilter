from __future__ import annotations

import json

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.inference.fixed_center_curvature import (
    FIXED_CENTER_CURVATURE_NONCLAIMS,
    FixedCenterCurvatureThresholds,
    compare_precision_geometry,
    consensus_shrunk_precision,
    consensus_shrunk_precision_tf,
    fit_fixed_center_curvature,
)


def _clouds(
    precisions: tuple[np.ndarray, ...],
    *,
    center_score: np.ndarray,
    rows: int = 8,
    seed_base: int = 20260716,
) -> tuple[np.ndarray, np.ndarray]:
    offsets = []
    scores = []
    for index, precision in enumerate(precisions):
        rng = np.random.default_rng(seed_base + index)
        cloud = rng.normal(size=(rows, center_score.size)) * 0.1
        offsets.append(cloud)
        scores.append(center_score[None, :] - cloud @ precision.T)
    return np.stack(offsets), np.stack(scores)


def _thresholds(**overrides) -> FixedCenterCurvatureThresholds:
    values = {
        "selection_holdout_relative_rmse_cap": 1.0e-8,
        "audit_relative_rmse_cap": 1.0e-8,
        "projection_relative_frobenius_cap": 1.0e-10,
        "generalized_eigenvalue_spread_cap": 1.05,
        "trace_normalized_frobenius_cap": 1.0e-6,
        "trace_normalized_operator_cap": 1.0e-6,
        "principal_angle_degrees_cap": 1.0e-6,
        "principal_subspace_rank": 1,
    }
    values.update(overrides)
    return FixedCenterCurvatureThresholds(**values)


def test_stable_fixed_center_curvature_accepts_nonzero_score() -> None:
    precision = np.array([[3.0, 0.6], [0.6, 1.5]])
    center_score = np.array([4.0, -2.5])
    train_z, train_scores = _clouds(
        (precision, precision), center_score=center_score
    )
    select_z, select_scores = _clouds(
        (precision, precision), center_score=center_score, seed_base=20260816
    )
    audit_z, audit_scores = _clouds(
        (precision,), center_score=center_score, seed_base=20260916
    )

    result = fit_fixed_center_curvature(
        np.array([0.2, -0.1]),
        center_score,
        train_z,
        train_scores,
        select_z,
        select_scores,
        audit_z[0],
        audit_scores[0],
        thresholds=_thresholds(),
        factor_max=1,
        lineage={"fit_seeds": [20260716, 20260717], "audit_seed": 20260718},
    )

    assert result.accepted is True
    assert result.status == "eligible_for_exact_hmc_canary"
    assert result.diagnostics["center_stationarity_required"] is False
    assert result.diagnostics["center_score_role"] == "explanatory_only"
    assert result.diagnostics["selection"]["audit_used_for_selection"] is False
    assert result.diagnostics["selection"]["audit_changed_selection"] is False
    np.testing.assert_allclose(result.selected_precision_z, precision, atol=1.0e-10)
    assert tuple(result.payload()["nonclaims"]) == FIXED_CENTER_CURVATURE_NONCLAIMS
    json.dumps(result.payload())


def test_low_score_does_not_override_cross_replicate_instability() -> None:
    first = np.diag([1.0, 2.0])
    second = np.diag([1.0, 20.0])
    center_score = np.array([0.01, -0.01])
    train_z, train_scores = _clouds((first, second), center_score=center_score)
    select_z, select_scores = _clouds(
        (first, second), center_score=center_score, seed_base=20260816
    )
    audit_z, audit_scores = _clouds(
        (first,), center_score=center_score, seed_base=20260916
    )

    result = fit_fixed_center_curvature(
        np.zeros(2),
        center_score,
        train_z,
        train_scores,
        select_z,
        select_scores,
        audit_z[0],
        audit_scores[0],
        thresholds=_thresholds(generalized_eigenvalue_spread_cap=2.0),
        factor_max=1,
    )

    assert result.accepted is False
    assert result.status == "geometry_readiness_blocked"
    dense_stability = result.diagnostics["selection"]["stability"]["dense"]
    assert dense_stability["passed"] is False
    assert (
        dense_stability["comparisons"][0]["metrics"]["generalized_eigenvalues"][
            "spread"
        ]
        > 2.0
    )
    dense_fits = [fit for fit in result.fits if fit.family == "dense"]
    assert all(fit.raw_precision_z is not None for fit in dense_fits)
    assert all(fit.raw_eigenvalues is not None for fit in dense_fits)


def test_fresh_audit_cloud_can_veto_without_changing_selection() -> None:
    precision = np.diag([2.0, 4.0])
    wrong_audit_precision = np.diag([8.0, 0.5])
    center_score = np.array([0.5, -0.3])
    train_z, train_scores = _clouds(
        (precision, precision), center_score=center_score
    )
    select_z, select_scores = _clouds(
        (precision, precision), center_score=center_score, seed_base=20260816
    )
    audit_z, audit_scores = _clouds(
        (wrong_audit_precision,), center_score=center_score, seed_base=20260916
    )

    result = fit_fixed_center_curvature(
        np.zeros(2),
        center_score,
        train_z,
        train_scores,
        select_z,
        select_scores,
        audit_z[0],
        audit_scores[0],
        thresholds=_thresholds(audit_relative_rmse_cap=0.1),
        factor_max=1,
    )

    assert result.accepted is False
    assert result.status == "audit_holdout_rejected"
    assert result.selected_precision_z is not None
    assert result.diagnostics["selection"]["audit_changed_selection"] is False
    with pytest.raises(ValueError, match="not eligible"):
        result.build_mass_artifact(scale=np.ones(2), adapter_signature="a" * 64)


def test_incomplete_stability_thresholds_remain_diagnostic_only() -> None:
    precision = np.diag([2.0, 3.0])
    center_score = np.array([1.0, -1.0])
    train_z, train_scores = _clouds(
        (precision, precision), center_score=center_score
    )
    select_z, select_scores = _clouds(
        (precision, precision), center_score=center_score, seed_base=20260816
    )
    audit_z, audit_scores = _clouds(
        (precision,), center_score=center_score, seed_base=20260916
    )
    thresholds = FixedCenterCurvatureThresholds(
        selection_holdout_relative_rmse_cap=1.0e-8,
        audit_relative_rmse_cap=1.0e-8,
        projection_relative_frobenius_cap=1.0e-10,
    )

    result = fit_fixed_center_curvature(
        np.zeros(2),
        center_score,
        train_z,
        train_scores,
        select_z,
        select_scores,
        audit_z[0],
        audit_scores[0],
        thresholds=thresholds,
        factor_max=1,
    )

    assert result.accepted is False
    assert result.status == "diagnostic_only"
    assert result.diagnostics["thresholds_complete"] is False


def test_diagnostic_center_artifact_uses_covariance_scaling_and_role() -> None:
    precision_z = np.diag([2.0, 8.0])
    center_score = np.array([2.0, -3.0])
    train_z, train_scores = _clouds(
        (precision_z, precision_z), center_score=center_score
    )
    select_z, select_scores = _clouds(
        (precision_z, precision_z), center_score=center_score, seed_base=20260816
    )
    audit_z, audit_scores = _clouds(
        (precision_z,), center_score=center_score, seed_base=20260916
    )
    result = fit_fixed_center_curvature(
        np.array([0.4, -0.2]),
        center_score,
        train_z,
        train_scores,
        select_z,
        select_scores,
        audit_z[0],
        audit_scores[0],
        thresholds=_thresholds(),
        factor_max=1,
    )

    scale = np.array([0.5, 2.0])
    artifact = result.build_mass_artifact(
        scale=scale,
        adapter_signature="b" * 64,
    )
    expected_covariance_theta = (
        np.diag(scale) @ np.linalg.inv(precision_z) @ np.diag(scale)
    )
    assert artifact.position_role == "diagnostic_center"
    np.testing.assert_allclose(artifact.covariance, expected_covariance_theta)
    assert artifact.regularization_report["position_role"] == "diagnostic_center"
    payload = artifact.to_payload(include_arrays=True)
    assert payload["position_role"] == "diagnostic_center"


def test_consensus_shrinkage_and_geometry_metrics_are_spd_and_oriented() -> None:
    first = np.diag([1.0, 4.0, 8.0])
    second = np.diag([1.1, 3.8, 7.5])
    target = np.diag([1.0, 2.0, 4.0])
    candidate = consensus_shrunk_precision(
        (first, second), target=target, weight=0.25
    )
    expected = 0.75 * 0.5 * (first + second) + 0.25 * target
    np.testing.assert_allclose(candidate, expected)
    assert np.min(np.linalg.eigvalsh(candidate)) > 0.0

    metrics = compare_precision_geometry(first, second, subspace_rank=2)
    assert metrics["generalized_eigenvalues"]["spread"] >= 1.0
    assert metrics["positive_subspace_rank"] == 2
    assert metrics["maximum_principal_angle_degrees"] == pytest.approx(0.0)


def test_consensus_tensorflow_kernel_has_xla_value_and_gradient_parity() -> None:
    precisions = tf.constant(
        [np.diag([1.0, 4.0]), np.diag([1.2, 3.5])], tf.float64
    )
    target = tf.constant(np.diag([2.0, 2.0]), tf.float64)

    @tf.function(jit_compile=True)
    def compiled(weight: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        with tf.GradientTape() as tape:
            tape.watch(weight)
            value = consensus_shrunk_precision_tf(precisions, target, weight)
            objective = tf.reduce_sum(tf.square(value))
        return value, tape.gradient(objective, weight)

    weight = tf.constant(0.3, tf.float64)
    with tf.GradientTape() as tape:
        tape.watch(weight)
        eager = consensus_shrunk_precision_tf(precisions, target, weight)
        objective = tf.reduce_sum(tf.square(eager))
    eager_gradient = tape.gradient(objective, weight)
    xla, xla_gradient = compiled(weight)
    np.testing.assert_allclose(xla.numpy(), eager.numpy(), atol=1.0e-12)
    np.testing.assert_allclose(xla_gradient.numpy(), eager_gradient.numpy(), atol=1.0e-12)


def test_factor_covariance_is_invariant_to_column_rotation_and_permutation() -> None:
    loadings = np.array(
        [[0.4, 0.0], [0.2, 0.3], [-0.15, 0.25], [0.1, -0.2]]
    )
    angle = 0.7
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    rotated = loadings @ rotation
    permuted = rotated[:, ::-1]
    diagonal = 1.0 - np.sum(loadings**2, axis=1)
    covariance = np.diag(diagonal) + loadings @ loadings.T
    rotated_covariance = np.diag(diagonal) + rotated @ rotated.T
    permuted_covariance = np.diag(diagonal) + permuted @ permuted.T
    np.testing.assert_allclose(rotated_covariance, covariance, atol=1.0e-12)
    np.testing.assert_allclose(permuted_covariance, covariance, atol=1.0e-12)


def test_partition_views_fail_closed_and_audit_budget_is_enforced() -> None:
    precision = np.eye(2)
    center_score = np.zeros(2)
    shared_z, shared_scores = _clouds(
        (precision, precision), center_score=center_score
    )
    selection_z, selection_scores = _clouds(
        (precision, precision), center_score=center_score, seed_base=20260816
    )
    audit_z, audit_scores = _clouds(
        (precision,), center_score=center_score, rows=3, seed_base=20260916
    )

    with pytest.raises(ValueError, match="disjoint arrays"):
        fit_fixed_center_curvature(
            np.zeros(2),
            center_score,
            shared_z,
            shared_scores,
            shared_z,
            shared_scores,
            audit_z[0],
            audit_scores[0],
            thresholds=_thresholds(),
            factor_max=1,
        )

    with pytest.raises(ValueError, match="copied rows"):
        fit_fixed_center_curvature(
            np.zeros(2),
            center_score,
            shared_z,
            shared_scores,
            shared_z.copy(),
            shared_scores.copy(),
            audit_z[0],
            audit_scores[0],
            thresholds=_thresholds(),
            factor_max=1,
        )

    with pytest.raises(ValueError, match="audit rows"):
        fit_fixed_center_curvature(
            np.zeros(2),
            center_score,
            shared_z,
            shared_scores,
            selection_z,
            selection_scores,
            audit_z[0],
            audit_scores[0],
            thresholds=_thresholds(),
            factor_max=1,
        )


def test_diagonal_only_selector_marks_nonpromotable_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import bayesfilter.inference.fixed_center_curvature as fixed

    fit = fixed.FixedCenterCurvatureFit(
        family="dense",
        replicate_index=0,
        factor_count=None,
        accepted=True,
        status="usable",
        raw_precision_z=np.array([[2.0, 0.5], [0.5, 4.0]]),
        precision_z=np.array([[2.0, 0.5], [0.5, 4.0]]),
        covariance_z=np.linalg.inv(np.array([[2.0, 0.5], [0.5, 4.0]])),
        raw_eigenvalues=np.linalg.eigvalsh(np.array([[2.0, 0.5], [0.5, 4.0]])),
        raw_nonpositive_count=0,
        projection_relative_frobenius=0.0,
        selection_holdout_relative_rmse=0.0,
        diagnostics={"geometry_admissible": True},
    )
    fits = (fit, fixed.FixedCenterCurvatureFit(**{**fit.__dict__, "replicate_index": 1}))
    monkeypatch.setattr(
        fixed,
        "_mean_selection_error",
        lambda candidate, *_args: 0.0 if np.allclose(candidate, np.diag(np.diag(candidate))) else 1.0,
    )
    selected, diagnostics = fixed._select_candidate(
        fits,
        np.zeros(2),
        ((np.eye(2), -np.eye(2)),),
        thresholds=_thresholds(selection_holdout_relative_rmse_cap=2.0),
        shrinkage_weights=(0.0, 1.0),
        structured_target_family=None,
    )

    assert selected is not None
    assert diagnostics["selected_weight"] == 1.0
    assert diagnostics["selected_target"] == "diagonal_consensus"
    assert diagnostics["diagonal_only"] is True


def test_explicit_structured_target_uses_shrinkage_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import bayesfilter.inference.fixed_center_curvature as fixed

    dense_precision = np.array([[2.0, 0.5], [0.5, 4.0]])
    factor_precision = np.diag([1.5, 3.0])

    def make_fit(family: str, replicate: int, precision: np.ndarray):
        return fixed.FixedCenterCurvatureFit(
            family=family,
            replicate_index=replicate,
            factor_count=1 if family == "factor_1" else None,
            accepted=True,
            status="usable",
            raw_precision_z=precision,
            precision_z=precision,
            covariance_z=np.linalg.inv(precision),
            raw_eigenvalues=np.linalg.eigvalsh(precision),
            raw_nonpositive_count=0,
            projection_relative_frobenius=0.0,
            selection_holdout_relative_rmse=0.0,
            diagnostics={"geometry_admissible": True},
        )

    fits = tuple(
        make_fit(family, replicate, precision)
        for family, precision in (
            ("dense", dense_precision),
            ("factor_1", factor_precision),
        )
        for replicate in (0, 1)
    )
    monkeypatch.setattr(
        fixed,
        "_mean_selection_error",
        lambda candidate, *_args: abs(float(candidate[0, 0]) - 1.75),
    )
    selected, diagnostics = fixed._select_candidate(
        fits,
        np.zeros(2),
        ((np.eye(2), -np.eye(2)),),
        thresholds=_thresholds(selection_holdout_relative_rmse_cap=2.0),
        shrinkage_weights=(0.0, 0.5, 1.0),
        structured_target_family="factor_1",
    )

    assert selected is not None
    assert selected["family"] == "consensus_factor_1"
    assert diagnostics["selected_weight"] == 0.5
    assert diagnostics["selected_target"] == "factor_1"


def test_two_factor_fit_is_skipped_when_one_factor_is_adequate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import bayesfilter.inference.fixed_center_curvature as fixed

    calls = []
    precision = np.diag([2.0, 3.0])
    center_score = np.zeros(2)
    train_z, train_scores = _clouds((precision, precision), center_score=center_score)
    select_z, select_scores = _clouds(
        (precision, precision), center_score=center_score, seed_base=20260816
    )
    audit_z, audit_scores = _clouds(
        (precision,), center_score=center_score, seed_base=20260916
    )

    def controlled_fit(*_args, replicate_index, factor_count, **_kwargs):
        calls.append(factor_count)
        return fixed.FixedCenterCurvatureFit(
            family=f"factor_{factor_count}",
            replicate_index=replicate_index,
            factor_count=factor_count,
            accepted=True,
            status="usable",
            raw_precision_z=precision,
            precision_z=precision,
            covariance_z=np.linalg.inv(precision),
            raw_eigenvalues=np.linalg.eigvalsh(precision),
            raw_nonpositive_count=0,
            projection_relative_frobenius=0.0,
            selection_holdout_relative_rmse=0.0,
            diagnostics={"geometry_admissible": True},
        )

    monkeypatch.setattr(fixed, "_fit_structured_precision", controlled_fit)
    result = fit_fixed_center_curvature(
        np.zeros(2),
        center_score,
        train_z,
        train_scores,
        select_z,
        select_scores,
        audit_z[0],
        audit_scores[0],
        thresholds=_thresholds(),
        factor_max=2,
    )

    assert calls == [1, 1]
    assert result.diagnostics["selection"]["factor_escalation"] == {
        "one_factor_passed_all_replicates": True,
        "one_factor_stability_passed": True,
        "two_factor_attempted": False,
        "reason": "one_factor_fit_holdout_and_stability_passed",
    }


def test_two_factor_fit_is_attempted_after_one_factor_rejection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import bayesfilter.inference.fixed_center_curvature as fixed

    calls = []
    precision = np.diag([2.0, 3.0])
    center_score = np.zeros(2)
    train_z, train_scores = _clouds((precision, precision), center_score=center_score)
    select_z, select_scores = _clouds(
        (precision, precision), center_score=center_score, seed_base=20260816
    )
    audit_z, audit_scores = _clouds(
        (precision,), center_score=center_score, seed_base=20260916
    )

    def controlled_fit(*_args, replicate_index, factor_count, **_kwargs):
        calls.append(factor_count)
        accepted = factor_count == 2
        return fixed.FixedCenterCurvatureFit(
            family=f"factor_{factor_count}",
            replicate_index=replicate_index,
            factor_count=factor_count,
            accepted=accepted,
            status="usable" if accepted else "holdout_score_fit_rejected",
            raw_precision_z=precision,
            precision_z=precision,
            covariance_z=np.linalg.inv(precision),
            raw_eigenvalues=np.linalg.eigvalsh(precision),
            raw_nonpositive_count=0,
            projection_relative_frobenius=0.0,
            selection_holdout_relative_rmse=0.0 if accepted else 1.0,
            diagnostics={"geometry_admissible": True},
        )

    monkeypatch.setattr(fixed, "_fit_structured_precision", controlled_fit)
    result = fit_fixed_center_curvature(
        np.zeros(2),
        center_score,
        train_z,
        train_scores,
        select_z,
        select_scores,
        audit_z[0],
        audit_scores[0],
        thresholds=_thresholds(),
        factor_max=2,
    )

    assert calls == [1, 1, 2, 2]
    assert result.selected_family == "factor_2"
    assert result.diagnostics["selection"]["factor_escalation"] == {
        "one_factor_passed_all_replicates": False,
        "one_factor_stability_passed": True,
        "two_factor_attempted": True,
        "reason": "one_factor_fit_holdout_or_stability_rejected",
    }
