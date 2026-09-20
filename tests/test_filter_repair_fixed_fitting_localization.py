"""Bounded diagnostic localization of baseline/XLA geometry differences."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

from bayesfilter.inference import factor_correlation_geometry as factor
from bayesfilter.inference import fixed_center_curvature as fixed
from bayesfilter.inference.mass_matrix_tf import _eigenpairs
from bayesfilter.ops.qr_lstsq_tf import complete_orthogonal_lstsq

D = tf.float64


def _baseline():
    source = subprocess.check_output(["git", "show",
        "3582b4ac:bayesfilter/inference/factor_correlation_geometry.py"], text=True)
    spec = importlib.util.spec_from_loader("fixed_fitting_original_factor_reference", loader=None)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    exec(compile(source, spec.name, "exec"), module.__dict__)  # noqa: S102
    return module


def test_frozen_initialization_and_loss_breakdown():
    baseline = _baseline()
    dimension = 3
    center = tf.constant([-.2, .05, .3], D)
    precision = tf.linalg.diag(tf.constant([1., 2., 3.], D)) + .13
    training = .2 * tf.sin(.17 * tf.cast(tf.reshape(tf.range(27), [9, 3]), D) ** 2 + .1)
    scores = center - training @ precision
    response = center - scores
    weights = tf.fill([9], tf.constant(1 / 9, D))
    weights /= tf.reduce_sum(weights)
    cfg = factor.FactorCorrelationGeometryConfig()
    before_dense = baseline._weighted_dense_precision(training, response, weights, max_condition_number=1e8)
    before_cov = tf.linalg.inv(before_dense)
    before_std, before_load, before_anchor = baseline._initial_factor_state(before_cov, factor_count=1, loading_margin=1e-6)
    before_raw = baseline._encode_state(before_std, before_load, before_anchor, cfg)

    def factory(refined):
        eigenpairs = (lambda matrix: _eigenpairs(matrix, True)) if refined else tf.linalg.eigh

        @tf.function(input_signature=[tf.TensorSpec([9, 3], D), tf.TensorSpec([9, 3], D), tf.TensorSpec([9], D)],
            jit_compile=True, autograph=False)
        def initialize(offsets, responses, weights):
            raw = complete_orthogonal_lstsq(offsets * tf.sqrt(weights[:, None]), responses * tf.sqrt(weights[:, None]))
            square = .5 * (raw + tf.transpose(raw))
            values, vectors = eigenpairs(square)
            residual = tf.reduce_max(tf.abs(square @ vectors - vectors * values))
            projected = vectors * tf.maximum(values, tf.maximum(tf.reduce_max(tf.abs(values)), 1.) / 1e8)
            dense = projected @ tf.transpose(vectors)
            covariance = tf.linalg.inv(dense)
            deviations = tf.sqrt(tf.linalg.diag_part(covariance))
            correlation = covariance / (deviations[:, None] * deviations[None, :])
            corr_values, corr_vectors = eigenpairs(correlation)
            corr_residual = tf.reduce_max(tf.abs(correlation @ corr_vectors - corr_vectors * corr_values))
            loadings = corr_vectors[:, -1:] * tf.sqrt(tf.maximum(corr_values[-1:] - 1., 1e-6))
            row_norm = tf.linalg.norm(loadings, axis=1, keepdims=True)
            loadings *= tf.minimum(tf.ones_like(row_norm), tf.constant(.8 * np.sqrt(1. - 1e-6), D) / tf.maximum(row_norm, 1e-15))
            anchor = tf.argmax(tf.abs(loadings[:, 0]), output_type=tf.int32)
            loadings *= tf.where(loadings[anchor, 0] >= 0., tf.constant(1., D), tf.constant(-1., D))
            encoded = factor._encode_state(deviations, loadings, (anchor,), cfg)
            return dense, encoded, residual, corr_residual
        return initialize

    report = {}
    for refined in (False, True):
        dense, raw, residual, corr_residual = factory(refined)(training, response, weights)
        report[str(refined)] = {"dense_error": float(tf.reduce_max(tf.abs(dense - before_dense))),
            "encoded_error": float(tf.reduce_max(tf.abs(raw - before_raw))),
            "dense_eigen_residual": float(residual), "correlation_eigen_residual": float(corr_residual)}

    def objective(module, raw):
        covariance, _, _ = module._decode_covariance(raw, dimension=dimension, anchors=before_anchor, config=cfg)
        precision = tf.linalg.cholesky_solve(tf.linalg.cholesky(covariance), tf.eye(dimension, dtype=D))
        return tf.reduce_sum(weights * tf.reduce_mean(tf.square(tf.einsum("ij,bj->bi", precision, training) - response), 1))

    with tf.GradientTape() as tape:
        tape.watch(before_raw)
        loss = objective(baseline, before_raw)
    gradient = tape.gradient(loss, before_raw)

    @tf.function(input_signature=[tf.TensorSpec([6], D)], jit_compile=True, autograph=False)
    def compiled(raw):
        with tf.GradientTape() as tape:
            tape.watch(raw)
            value = objective(factor, raw)
        return value, tape.gradient(value, raw)

    value, score = compiled(before_raw)
    report["identical_state"] = {"loss_error": float(tf.abs(value - loss)),
        "gradient_error": float(tf.reduce_max(tf.abs(score - gradient)))}
    report["dense_projection"] = float(fixed._run_kernel(fixed._dense_fit_kernel, center, training, scores,
        training, scores, tf.constant(1e-8, D), tf.constant(1e8, D))[4])
    print("FIXED_FITTING_LOCALIZATION " + json.dumps(report, sort_keys=True))
    assert report["True"]["dense_eigen_residual"] < 1e-12
    assert report["True"]["correlation_eigen_residual"] < 1e-12
    assert report["True"]["encoded_error"] < 1e-12


def test_original_fit_field_breakdown(monkeypatch):
    from tests.test_filter_repair_fixed_fitting import _inputs

    original = _baseline()
    inputs = tuple(tf.constant(value, D) for value in _inputs(3))
    center, train, scores, holdout, holdout_scores = inputs[1:6]
    cfg = factor.FactorCorrelationGeometryConfig(max_condition_number=1e8,
        holdout_score_relative_rmse=.1)
    captured = []
    initial = []
    minimize = factor.tfp.optimizer.lbfgs_minimize

    def capture(*args, **kwargs):
        initial.append(kwargs['initial_position'])
        result = minimize(*args, **kwargs)
        captured.append(result)
        return result

    with monkeypatch.context() as context:
        context.setattr(original.tfp.optimizer, 'lbfgs_minimize', capture)
        before = original.fit_factor_correlation_score_geometry(center, train[0], scores[0],
            holdout[0], holdout_scores[0], config=original.FactorCorrelationGeometryConfig(
                max_condition_number=1e8, holdout_score_relative_rmse=.1))
    baseline = captured[0]
    arguments = (center, train[0], scores[0], holdout[0], holdout_scores[0], tf.fill([9], tf.constant(1 / 9, D)))
    report = {}
    source = Path(factor.__file__).read_text()
    decorator = ('        @tf.function(input_signature=[tf.TensorSpec([parameter_count], tf.float64)],\n'
        '            jit_compile=jit_compile, autograph=False)\n')
    assert source.count(decorator) == 1
    spec = importlib.util.spec_from_loader('factor_unshared_diagnostic_reference', loader=None)
    unshared = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = unshared
    exec(compile(source.replace(decorator, ''), spec.name, 'exec'), unshared.__dict__)  # noqa: S102
    spec = importlib.util.spec_from_loader('factor_initialization_boundaries_diagnostic', loader=None)
    bounded = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = bounded
    boundary_source = source.replace('import tensorflow_probability as tfp',
        'import tensorflow_probability as tfp\n'
        'from tensorflow.compiler.tf2xla.ops.gen_xla_ops import xla_optimization_barrier')
    boundary_source = boundary_source.replace('        dense_covariance = tf.linalg.inv(dense_precision)',
        '        dense_precision = xla_optimization_barrier(input=[dense_precision])[0]\n'
        '        dense_covariance = xla_optimization_barrier(input=[tf.linalg.inv(dense_precision)])[0]')
    boundary_source = boundary_source.replace('        initial_raw = _encode_state(',
        '        initial_standard_deviations, initial_loadings = xla_optimization_barrier(\n'
        '            input=[initial_standard_deviations, initial_loadings])\n'
        '        initial_raw = _encode_state(')
    exec(compile(boundary_source, spec.name, 'exec'), bounded.__dict__)  # noqa: S102
    for label, module, jit, frozen_initial in (
            ('graph', factor, False, False), ('xla', factor, True, False),
            ('xla_original_init', factor, True, True),
            ('unshared', unshared, True, False), ('unshared_original_init', unshared, True, True),
            ('explicit_initialization_boundaries', bounded, True, False)):
        with monkeypatch.context() as context:
            if frozen_initial:
                context.setattr(module, '_encode_state', lambda *_: initial[0])

            def diagnosis(*args, module=module, **kwargs):
                return module._prediction_jacobian_diagnostics(*args, **kwargs)

            after = module._make_factor_program(3, 9, 6, cfg, jit, diagnosis)(*arguments)
        same_rank, same_condition = original._prediction_jacobian_diagnostics(
            after['optimizer'].position, train[0], dimension=3,
            anchors=before.anchor_indices, config=cfg)
        report[label] = {
            'precision_error': float(tf.reduce_max(tf.abs(after['precision'] - before.precision_z))),
            'raw_error': float(tf.reduce_max(tf.abs(after['optimizer'].position - baseline.position))),
            'optimizer_iterations': [int(baseline.num_iterations), int(after['optimizer'].num_iterations)],
            'optimizer_evaluations': [int(baseline.num_objective_evaluations), int(after['optimizer'].num_objective_evaluations)],
            'condition_before': before.diagnostics['prediction_jacobian_condition_number'],
            'condition_after': float(after['jacobian_condition']),
            'condition_reference_at_current_raw': float(same_condition),
            'rank_reference_at_current_raw': int(same_rank),
        }
    print('FIXED_FITTING_ORIGINAL_FIELD_BREAKDOWN ' + json.dumps(report, sort_keys=True))
    assert all(np.isfinite(row['condition_after']) for row in report.values())
