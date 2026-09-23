"""Explanatory isotropic fit attribution; execution is not qualification."""

import dataclasses

import numpy as np
import tensorflow as tf

from bayesfilter.inference import fixed_center_curvature as current
from bayesfilter.inference.dense_initializer_random_tf import (
    make_dense_initializer_cloud_design,
)
from tests.test_filter_repair_dense_rng import original_cloud
from tests.test_filter_repair_dense_validated_fit import (
    _thresholds,
    normalized,
    original_fitter,
)
from tests.test_filter_repair_geometry_control import clean, save


def test_identical_cloud_isotropic_fitter_attribution(request):
    original, hashes = original_fitter()
    draw, excerpt = original_cloud()
    cases = []
    for dimension, rows in ((3, [9, 9, 6, 6, 7]), (23, [68, 68, 46, 46, 46])):
        configuration = type('CloudSettings', (), {'seed': (215, 91), 'curvature_radius': .13})()
        expected_clouds = [draw(configuration, 0, index, count, dimension)[2].numpy()
            for index, count in enumerate(rows)]
        generate = make_dense_initializer_cloud_design(dimension, 2, rows[0], rows[2], rows[-1], 1)
        bank = generate(tf.constant(configuration.seed, tf.int32), tf.constant(.13, tf.float64)).numpy()[0]
        actual_clouds = [bank[index, :count].copy() for index, count in enumerate(rows)]
        outputs = {}
        for label, clouds in (('original_cloud', expected_clouds), ('generated_cloud', actual_clouds)):
            train, select, audit = np.stack(clouds[:2]), np.stack(clouds[2:4]), clouds[4]
            inputs = (np.zeros(dimension), np.zeros(dimension), train, -train, select, -select, audit, -audit)
            for name, module in (('original', original), ('current', current)):
                thresholds = module.FixedCenterCurvatureThresholds(**dataclasses.asdict(_thresholds(current)))
                record = module.fit_fixed_center_curvature(*inputs, thresholds=thresholds).payload()
                precision = record['selected_precision_z']
                outputs[name + '_' + label] = {
                    'record': normalized({'result': record})['result'],
                    'max_error_from_identity': None if precision is None else float(np.max(np.abs(
                        np.asarray(precision) - np.eye(dimension)))),
                }
        cases.append({'dimension': dimension, 'rows': rows,
            'original_clouds': clean(expected_clouds), 'generated_clouds': clean(actual_clouds),
            'maximum_cloud_error': max(float(np.max(np.abs(a - b))) for a, b in
                zip(expected_clouds, actual_clouds, strict=True)), 'outputs': outputs})
    save(request, 'dense-seeded-isotropic-attribution.json', {
        'role': 'explanatory_only_no_equality_or_selection_waiver', 'cases': cases,
        'original_fitter_sources': hashes, 'original_rng_excerpt': excerpt,
        'nonclaims': ['This run does not qualify isotropic seeded composition, actual DZ5, or costs.'],
    })
    assert len(cases) == 2 and all(len(case['outputs']) == 4 for case in cases)
