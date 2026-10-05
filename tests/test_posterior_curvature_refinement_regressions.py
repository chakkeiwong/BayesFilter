"""Regression oracles only; legacy NumPy/HMC code is never a new runtime dependency."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.inference import (
    PosteriorCurvatureRefinementConfig,
    refine_posterior_local_curvature,
)
from bayesfilter.inference import posterior_curvature_tf as runtime


def _target(theta: tf.Tensor):
    mean = tf.constant([0.2, -0.5, 0.7], tf.float64)
    precision = tf.constant([[4.0, 0.4, 0.0], [0.4, 2.0, 0.2], [0.0, 0.2, 1.0]], tf.float64)
    delta = theta - mean
    return -0.5 * tf.reduce_sum(delta * tf.matmul(delta, precision), axis=1), -tf.matmul(delta, precision)


def _eligible(theta: tf.Tensor) -> tf.Tensor:
    return tf.ones(tf.shape(theta)[0], tf.bool)


def test_public_call_path_has_no_legacy_hmc_import() -> None:
    path = Path(__file__).parents[1] / "bayesfilter/inference/posterior_curvature_refinement.py"
    tree = ast.parse(path.read_text())
    imports = [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert all("hmc" not in module for module in imports)
    assert all("numpy" not in module for module in imports)


def test_stateless_seed_and_row_accounting_are_reproducible() -> None:
    config = PosteriorCurvatureRefinementConfig(
        rows_per_partition=33,
        batch_size=8,
        max_physical_rows=1000,
        seed=20260908,
        selection_relative_rmse_cap=1.0e-8,
        audit_relative_rmse_cap=1.0e-8,
        proposal_relative_rmse_cap=1.0e-8,
        replicate_generalized_eigenvalue_spread_cap=1.01,
    )
    first = refine_posterior_local_curvature(
        _target,
        np.zeros(3, np.float64),
        np.array([[1.2, 0.0, 0.0], [0.1, 0.9, 0.0], [0.2, -0.1, 0.7]]),
        batched_eligibility_fn=_eligible,
        config=config,
    )
    second = refine_posterior_local_curvature(
        _target,
        np.zeros(3, np.float64),
        np.array([[1.2, 0.0, 0.0], [0.1, 0.9, 0.0], [0.2, -0.1, 0.7]]),
        batched_eligibility_fn=_eligible,
        config=config,
    )
    assert first.accepted and second.accepted
    np.testing.assert_array_equal(first.refined_factor.numpy(), second.refined_factor.numpy())
    diagnostics = first.diagnostics
    assert diagnostics["logical_rows"] == 1 + 4 * 33 + 33 + 33
    assert diagnostics["physical_rows"] == 8 + 6 * 40
    assert diagnostics["padded_rows"] == diagnostics["physical_rows"] - diagnostics["logical_rows"]


def test_public_result_does_not_construct_mass_artifact() -> None:
    result = refine_posterior_local_curvature(
        _target,
        np.zeros(3, np.float64),
        np.eye(3, dtype=np.float64),
        batched_eligibility_fn=_eligible,
        config=PosteriorCurvatureRefinementConfig(
            rows_per_partition=24,
            batch_size=8,
            max_physical_rows=1000,
            selection_relative_rmse_cap=1.0e-8,
            audit_relative_rmse_cap=1.0e-8,
            proposal_relative_rmse_cap=1.0e-8,
            replicate_generalized_eigenvalue_spread_cap=1.01,
        ),
    )
    assert result.accepted
    assert result.status == "eligible_for_local_position_factor"
    assert "mass" not in result.payload()["status"]


def test_fresh_public_import_does_not_load_hmc_or_localizer() -> None:
    code = """
import sys
from bayesfilter.inference import (
    PosteriorCurvatureRefinementConfig, PosteriorCurvatureRefinementResult,
    refine_posterior_local_curvature, fit_dense_score_precision_tf,
)
for name in sys.modules:
    if name.startswith('bayesfilter.inference.'):
        assert name.rsplit('.', 1)[-1] in {
            'score_curvature_tf', 'posterior_curvature_refinement',
            'posterior_curvature_tf', 'posterior_curvature_report', 'mass_matrix_tf',
        }, name
assert not any(name.startswith(('MacroFinance', 'filters.', 'inference.hmc')) for name in sys.modules)
print('fresh lazy closure passed')
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "fresh lazy closure passed" in result.stdout


@pytest.mark.parametrize("condition", [3.0, 1e6])
def test_new_helper_and_legacy_dense_fit_share_identical_raw_precision(monkeypatch, condition) -> None:
    from bayesfilter.inference.fixed_center_curvature import _fit_dense_precision

    center_buffer = tf.Variable(tf.zeros([2, 2], tf.float64))
    with tf.device(center_buffer.device):
        fit_count = tf.Variable(0, dtype=tf.int64)
        offset_buffer = tf.Variable(tf.zeros([2, 48, 2], tf.float64))
        score_buffer = tf.Variable(tf.zeros([2, 48, 2], tf.float64))
        selection_offset_buffer = tf.Variable(tf.zeros([2, 48, 2], tf.float64))
        selection_score_buffer = tf.Variable(tf.zeros([2, 48, 2], tf.float64))
        precision_buffer = tf.Variable(tf.zeros([2, 2, 2], tf.float64))
    original = runtime.fit_dense_score_precision_tf

    def capture(center, offsets, scores, **selection):
        fit = original(center, offsets, scores, **selection)
        slot = tf.reshape(fit_count.assign_add(1) - 1, [1, 1])
        writes = [center_buffer.scatter_nd_update(slot, center[None]),
            offset_buffer.scatter_nd_update(slot, offsets[None]),
            score_buffer.scatter_nd_update(slot, scores[None]),
            selection_offset_buffer.scatter_nd_update(slot, selection['selection_offsets'][None]),
            selection_score_buffer.scatter_nd_update(slot, selection['selection_scores'][None]),
            precision_buffer.scatter_nd_update(slot, fit['raw_precision'][None])]
        with tf.control_dependencies(writes):
            return {name: tf.identity(value) for name, value in fit.items()}

    def target(theta):
        precision = tf.constant([[2.0, 0.3], [0.3, condition]], tf.float64)
        shifted = theta - tf.constant([0.3, -0.1], tf.float64)
        return -0.5 * tf.reduce_sum(shifted * (shifted @ precision), axis=1), -shifted @ precision

    monkeypatch.setattr(runtime, "fit_dense_score_precision_tf", capture)
    result = refine_posterior_local_curvature(
        target, np.zeros(2), np.array([[0.8, 0.0], [0.2, 1.1]]),
        batched_eligibility_fn=_eligible,
        config=PosteriorCurvatureRefinementConfig(rows_per_partition=48, batch_size=16),
    )
    assert result.accepted
    assert int(fit_count) == len(result.diagnostics['replicates']) == 2
    for index in range(int(fit_count)):
        legacy = _fit_dense_precision(
            center_buffer[index].numpy(), offset_buffer[index].numpy(), score_buffer[index].numpy(),
            selection_offset_buffer[index].numpy(), selection_score_buffer[index].numpy(),
            replicate_index=0, eigenvalue_floor=1e-12, max_condition_number=1e10,
            holdout_cap=1e-7, projection_cap=1e-7, require_raw_spd=True,
        )
        assert legacy.accepted
        np.testing.assert_allclose(precision_buffer[index], legacy.raw_precision_z, atol=1e-10, rtol=1e-12)


def test_additive_log_density_constants_do_not_change_geometry() -> None:
    def shifted(theta):
        values, scores = _target(theta)
        return values + 1e12, scores

    cfg = PosteriorCurvatureRefinementConfig(rows_per_partition=32, batch_size=16)
    original = refine_posterior_local_curvature(
        _target, np.zeros(3), np.eye(3), batched_eligibility_fn=_eligible, config=cfg,
    )
    translated = refine_posterior_local_curvature(
        shifted, np.zeros(3), np.eye(3), batched_eligibility_fn=_eligible, config=cfg,
    )
    assert original.accepted and translated.accepted
    np.testing.assert_array_equal(original.refined_factor, translated.refined_factor)
    assert original.diagnostics["proposal_relative_rmse"] == translated.diagnostics["proposal_relative_rmse"]


def test_affine_unit_changes_transform_position_covariance_not_pilot_precision() -> None:
    transformation = np.array([[10.0, 0.0, 0.0], [0.5, 0.1, 0.0], [-0.3, 0.2, 2.0]])
    translation = np.array([10.0, -0.5, 7.0])
    inverse = tf.constant(np.linalg.inv(transformation), tf.float64)
    pilot = np.array([[1.0, 0.0, 0.0], [0.2, 0.5, 0.0], [0.1, -0.3, 2.0]])
    center = np.array([0.1, -0.2, 0.3])

    def transformed(theta):
        raw = tf.matmul(theta - translation, inverse, transpose_b=True)
        values, scores = _target(raw)
        return values, scores @ inverse

    cfg = PosteriorCurvatureRefinementConfig(rows_per_partition=32, batch_size=16)
    raw = refine_posterior_local_curvature(
        _target, center, pilot, batched_eligibility_fn=_eligible, config=cfg,
    )
    changed = refine_posterior_local_curvature(
        transformed, transformation @ center + translation, transformation @ pilot,
        batched_eligibility_fn=_eligible, config=cfg,
    )
    assert raw.accepted and changed.accepted
    np.testing.assert_allclose(changed.refined_covariance,
                               transformation @ raw.refined_covariance.numpy() @ transformation.T,
                               atol=1e-9, rtol=1e-10)
    np.testing.assert_allclose(raw.precision_z, changed.precision_z, atol=1e-10)
    assert changed.diagnostics["center_score_refined_l2"] == pytest.approx(raw.diagnostics["center_score_refined_l2"])
