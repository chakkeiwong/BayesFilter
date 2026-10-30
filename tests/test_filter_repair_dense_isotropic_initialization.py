"""Explain the original isotropic factor initialization; no admission waiver."""

import hashlib
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from tests.test_filter_repair_fixed_fitting_localization import _baseline
from tests.test_filter_repair_geometry_control import clean, save


def test_isotropic_initial_state_and_stopping_condition(request):
    artifact = Path('/home/ubuntu/workspace/BayesFilter/docs/plans/artifacts/'
        'filter-gradient-repair-20260917/run-03537/dense-seeded-isotropic-attribution.json')
    record = json.loads(artifact.read_text())
    original = _baseline()
    cfg = original.FactorCorrelationGeometryConfig(max_condition_number=1e8,
        holdout_score_relative_rmse=.1)
    results = []
    for case in record['cases']:
        dimension = case['dimension']
        for index in range(2):
            offsets = tf.constant(case['original_clouds'][index], tf.float64)
            count = offsets.shape[0]
            weights = tf.fill([count], tf.constant(1. / count, tf.float64))
            weights /= tf.reduce_sum(weights)
            precision = original._weighted_dense_precision(offsets, offsets, weights,
                max_condition_number=cfg.max_condition_number)
            covariance = tf.linalg.inv(precision)
            deviations, loadings, anchors = original._initial_factor_state(covariance,
                factor_count=1, loading_margin=cfg.loading_margin)
            raw = original._encode_state(deviations, loadings, anchors, cfg)
            diagonal = tf.sqrt(tf.linalg.diag_part(covariance))
            correlation = covariance / (diagonal[:, None] * diagonal[None, :])
            eigenvalues = tf.linalg.eigvalsh(correlation)

            def loss(raw, dimension=dimension, anchors=anchors, offsets=offsets, weights=weights):
                candidate, _, _ = original._decode_covariance(raw,
                    dimension=dimension, anchors=anchors, config=cfg)
                inverse = tf.linalg.cholesky_solve(tf.linalg.cholesky(candidate), tf.eye(dimension, dtype=tf.float64))
                prediction = tf.einsum('ij,bj->bi', inverse, offsets)
                return tf.reduce_sum(weights * tf.reduce_mean(tf.square(prediction - offsets), axis=1)), inverse

            with tf.GradientTape() as tape:
                tape.watch(raw)
                value, decoded_precision = loss(raw)
            gradient = tape.gradient(value, raw)
            full = next(fit for fit in case['outputs']['original_original_cloud']['record']['fits']
                if fit['family'] == 'factor_1' and fit['replicate_index'] == index)
            difference = float(np.max(np.abs(decoded_precision.numpy() - np.asarray(full['precision_z']))))
            results.append(clean({'dimension': dimension, 'replicate': index,
                'weighted_precision': precision, 'correlation_eigenvalues': eigenvalues,
                'maximum_eigenvalue_gap': tf.reduce_max(eigenvalues) - tf.reduce_min(eigenvalues),
                'loading_floor_in_sqrt': 1e-6, 'initial_loadings': loadings, 'anchors': anchors,
                'initial_loss': value, 'maximum_absolute_gradient': tf.reduce_max(tf.abs(gradient)),
                'optimizer_tolerance': cfg.tolerance, 'optimizer_iterations': full['diagnostics']['optimizer_iterations'],
                'initial_precision': decoded_precision, 'full_original_precision': full['precision_z'],
                'maximum_initial_vs_final_error': difference}))
    save(request, 'dense-isotropic-initial-state.json', {
        'role': 'explanatory_original_source_mechanism', 'input_artifact': str(artifact),
        'input_sha256': hashlib.sha256(artifact.read_bytes()).hexdigest(), 'records': results,
        'nonclaims': ['No rank/anchor/precision comparison waiver; no new runtime threshold or clipping.'],
    })
    assert all(row['optimizer_iterations'] == 0 for row in results)
    assert all(row['maximum_absolute_gradient'] <= row['optimizer_tolerance'] for row in results)
    assert all(row['maximum_initial_vs_final_error'] < 1e-13 for row in results)
