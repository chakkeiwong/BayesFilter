"""Frozen-input numerical comparisons against the pinned pre-repair source."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.inference import quadratic_geometry as current
from tests.filter_repair_geometry_reference import FrozenLegacyGeometryStream
from tests.test_quadratic_geometry import _batched_quadratic_target, _quadratic_target


@pytest.fixture(scope="module")
def baseline(tmp_path_factory):
    path = tmp_path_factory.mktemp("quadratic_baseline") / "quadratic_reference.py"
    path.write_bytes(
        subprocess.check_output(
            [
                "git",
                "show",
                "3582b4ac5fea67fb5da7fa60a1cbfaf19df35adf:bayesfilter/inference/quadratic_geometry.py",
            ],
            cwd=Path(__file__).resolve().parents[1],
        )
    )
    spec = importlib.util.spec_from_file_location("quadratic_baseline_diagnostic", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("batched", [False, True])
def test_complete_geometry_matches_baseline_on_identical_clouds(
    baseline, monkeypatch, batched
):
    monkeypatch.setattr(current, "GeometryTensorStream", FrozenLegacyGeometryStream)
    precision = np.array([[3.0, 0.2, 0.0], [0.2, 1.6, 0.1], [0.0, 0.1, 0.9]])
    mode = np.array([0.08, -0.04, 0.03])
    scalar = _quadratic_target(precision, mode=mode)
    batch = _batched_quadratic_target(precision, mode=mode) if batched else None
    settings = dict(
        rank=2,
        sample_count=120,
        pilot_direction_count=64,
        eigenvalue_floor=0.1,
        max_condition_number=100.0,
        holdout_rmse_abs_tolerance=0.1,
        seed=(2026, 714),
    )
    expected = baseline.fit_low_rank_spd_quadratic_geometry(
        scalar,
        np.zeros(3),
        batched_value_and_score_fn=batch,
        config=baseline.LowRankSPDQuadraticGeometryConfig(**settings),
    )
    actual = current.fit_low_rank_spd_quadratic_geometry(
        scalar,
        np.zeros(3),
        batched_value_and_score_fn=batch,
        config=current.LowRankSPDQuadraticGeometryConfig(**settings),
    )
    assert actual.accepted == expected.accepted is True
    assert actual.status == expected.status
    assert actual.exact_evaluation_count == expected.exact_evaluation_count
    assert actual.center_refinement_accepted == expected.center_refinement_accepted
    for field in (
        "precision",
        "covariance",
        "linear_term",
        "mu",
        "intercept",
        "lambda0",
        "refined_center",
        "best_evaluated_position",
        "best_evaluated_score",
    ):
        np.testing.assert_allclose(
            getattr(actual, field), getattr(expected, field), rtol=1e-10, atol=1e-10
        )
    for field in ("train_rmse", "holdout_rmse", "holdout_threshold"):
        np.testing.assert_allclose(
            actual.diagnostics[field],
            expected.diagnostics[field],
            rtol=1e-10,
            atol=1e-10,
        )
    for field in ("score_design_rank", "mu_clipped_count"):
        assert actual.diagnostics["fit"][field] == expected.diagnostics["fit"][field]


@pytest.mark.parametrize("rank_deficient", [False, True])
def test_score_fit_rank_cutoff_matches_numpy_authority(baseline, rank_deficient):
    rng = np.random.default_rng(421)
    z = rng.normal(size=(50, 3))
    if rank_deficient:
        z[:, 2] = 0.0
    q = np.eye(3)[:, 1:]
    score_center = np.array([0.3, -0.1, 0.2])
    precision = np.diag([1.4, 2.1, 3.2])
    score = score_center - z @ precision
    values = 0.7 + z @ score_center - 0.5 * np.sum(z * (z @ precision), axis=1)
    config = current.LowRankSPDQuadraticGeometryConfig(rank=2, eigenvalue_floor=0.1)
    args = dict(q_basis=q, cfg=config, dim=3, rank=2, center_score_z=score_center)
    expected = baseline._fit_constrained_quadratic(z, values, score, **args)
    actual = current._fit_constrained_quadratic(z, values, score, **args)
    assert expected["status"] == actual["status"] == "usable"
    assert int(actual["score_design_rank"]) == expected["score_design_rank"]
    for field in ("precision", "mu", "raw_mu", "raw_lambda0", "intercept"):
        np.testing.assert_allclose(
            actual[field], expected[field], rtol=1e-10, atol=1e-10
        )


@pytest.mark.parametrize("batch", [False, True])
def test_xla_evaluator_enforces_budget_and_eligibility(batch):
    from bayesfilter.inference.posterior_local_initializer import (
        _EligibilityTrackingEvaluator,
        _evaluate_rows,
    )

    scalar = lambda x: (-tf.reduce_sum(x * x), -2.0 * x)
    batched = lambda x: (-tf.reduce_sum(x * x, axis=1), -2.0 * x)
    evaluator = _EligibilityTrackingEvaluator(
        scalar,
        dimension=2,
        max_rows=5,
        batched_fn=batched if batch else None,
        eligibility_fn=lambda x: x[0] >= 0.0,
        batched_eligibility_fn=(lambda x: x[:, 0] >= 0.0) if batch else None,
    )
    values, _, valid = _evaluate_rows(
        evaluator, tf.constant([[1.0, 2.0], [-1.0, 2.0], [2.0, 1.0]], tf.float64)
    )
    assert not valid
    assert np.isfinite(values[0]) and not np.isfinite(values[1])
    report = evaluator.diagnostics()
    assert report["evaluated_rows"] == 3
    assert report["mismatch_rows"] == report["invalid_rows"] == 1
    _, _, valid = _evaluate_rows(evaluator, tf.ones([3, 2], tf.float64))
    assert not valid
    assert evaluator.diagnostics()["budget_exhausted"] is True
    assert evaluator.diagnostics()["evaluated_rows"] == (3 if batch else 5)
