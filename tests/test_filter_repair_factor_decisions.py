"""Frozen complete-record authority for tensor-native factor rejection decisions."""

import importlib.util
import math
import re
import subprocess
import sys
from collections import namedtuple

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.inference import factor_correlation_geometry as factor
from bayesfilter.inference.factor_decisions_tf import factor_decisions_program
from tests.test_filter_repair_fixed_stability import _compare

D = tf.float64
Summary = namedtuple('Summary', 'failed converged num_iterations num_objective_evaluations objective_value')


@pytest.fixture(scope='module')
def frozen():
    source = subprocess.check_output(['git', 'show',
        'f3f47f76:bayesfilter/inference/factor_correlation_geometry.py'], text=True)
    spec = importlib.util.spec_from_loader('factor_decisions_frozen', loader=None)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    exec(compile(source, spec.name, 'exec'), module.__dict__)  # noqa: S102 - pinned complete record authority
    return module


def _healthy(factors):
    dimension = 3 if factors == 1 else 5
    rank = 2 * dimension if factors == 1 else 3 * dimension - 1
    return {'invalid_covariance_evaluations': tf.constant(0, tf.int64),
        'anchors': tf.range(factors), 'covariance': tf.eye(dimension, dtype=D),
        'precision': tf.eye(dimension, dtype=D), 'deviations': tf.ones([dimension], D),
        'loadings': tf.ones([dimension, factors], D) * .1, 'eigenvalues': tf.ones([dimension], D),
        'finite': tf.constant(True), 'condition_number': tf.constant(1., D),
        'train_rmse': tf.constant(.01, D), 'holdout_error': tf.constant(.02, D),
        'holdout_relative': tf.constant(.1, D), 'jacobian_rank': tf.constant(rank),
        'jacobian_condition': tf.constant(2., D),
        'optimizer': Summary(tf.constant(False), tf.constant(True), tf.constant(5),
            tf.constant(11), tf.constant(.001, D))}


def _arguments(record):
    return (record['invalid_covariance_evaluations'], record['finite'], record['eigenvalues'],
        record['condition_number'], record['jacobian_rank'], record['holdout_relative'],
        record['optimizer'].failed, record['loadings'])


@pytest.mark.parametrize('factors', [1, 2])
@pytest.mark.parametrize('fault', ['healthy', 'invalid', 'finite', 'zero_eigenvalue', 'negative_eigenvalue',
    'nan_eigenvalues', 'mixed_nan_eigenvalues', 'condition_edge', 'condition_over', 'nan_condition', 'rank', 'holdout_edge',
    'holdout_over', 'nan_holdout', 'optimizer', 'all', 'condition_rank', 'rank_holdout', 'holdout_optimizer'])
def test_complete_record_precedence_and_boundaries(frozen, factors, fault):
    dimension = 3 if factors == 1 else 5
    config = factor.FactorCorrelationGeometryConfig(factor_count=factors)
    computed = _healthy(factors)
    if fault in ('invalid', 'all'):
        computed['invalid_covariance_evaluations'] = tf.constant(2, tf.int64)
    if fault in ('finite', 'all'):
        computed['finite'] = tf.constant(False)
    if fault.endswith('eigenvalue'):
        computed['eigenvalues'] = tf.constant([0. if fault == 'zero_eigenvalue' else -.1] + [1.] * (dimension - 1), D)
    if fault == 'nan_eigenvalues':
        computed['eigenvalues'] = tf.fill([dimension], tf.constant(float('nan'), D))
    if fault == 'mixed_nan_eigenvalues':
        computed['eigenvalues'] = tf.constant([float('nan')] + [1.] * (dimension - 1), D)
    cap = config.max_condition_number * (1. + 1e-8)
    if fault in ('condition_edge', 'condition_over', 'all', 'condition_rank'):
        computed['condition_number'] = tf.constant(cap if fault == 'condition_edge' else np.nextafter(cap, math.inf), D)
    if fault == 'nan_condition':
        computed['condition_number'] = tf.constant(float('nan'), D)
    if fault in ('rank', 'all', 'condition_rank', 'rank_holdout'):
        computed['jacobian_rank'] = tf.constant(0)
        computed['jacobian_condition'] = tf.constant(float('inf'), D)
    if fault in ('holdout_edge', 'holdout_over', 'all', 'rank_holdout', 'holdout_optimizer'):
        computed['holdout_relative'] = tf.constant(config.holdout_score_relative_rmse if fault == 'holdout_edge'
            else np.nextafter(config.holdout_score_relative_rmse, math.inf), D)
    if fault == 'nan_holdout':
        computed['holdout_relative'] = tf.constant(float('nan'), D)
    if fault in ('optimizer', 'all', 'holdout_optimizer'):
        computed['optimizer'] = computed['optimizer']._replace(failed=tf.constant(True), converged=tf.constant(False))
    expected = frozen._factor_result_from_computed(computed, config, dimension, 2 * dimension, 2 * dimension, True).payload()
    actual = factor._factor_result_from_computed(computed, config, dimension, 2 * dimension, 2 * dimension, True).payload()
    _compare(actual, expected)


@pytest.mark.parametrize('factors', [1, 2])
def test_enclosing_decisions_retain_runtime_operands_and_reference(factors):
    dimension = 3 if factors == 1 else 5
    cfg = factor.FactorCorrelationGeometryConfig(factor_count=factors)
    candidate = factor_decisions_program(dimension, cfg)
    reference = factor_decisions_program(dimension, cfg, jit_compile=False)

    @tf.function(input_signature=candidate.input_signature, jit_compile=True, autograph=False)
    def enclosing(*args):
        return candidate.python_function(*args)

    inputs = _arguments(_healthy(factors))
    result = enclosing(*inputs)
    _compare({k: v.numpy() for k, v in result.items()}, {k: v.numpy() for k, v in reference(*inputs).items()})
    hlo = enclosing.experimental_get_compiler_ir(*inputs)(stage='hlo')
    entry = hlo[hlo.rfind('\nENTRY '):]
    assert sorted(int(i) for i in re.findall(r'\bparameter\((\d+)\)', entry)) == list(range(8))
    changed = list(inputs)
    changed[0] = tf.constant(1, tf.int64)
    changed[1] = tf.constant(False)
    changed[2] = changed[2] * -.1
    changed[3] = tf.constant(1e100, D)
    changed[4] = tf.constant(0)
    changed[5] = tf.constant(1., D)
    changed[6] = tf.constant(True)
    changed[7] = changed[7] * .9
    result = enclosing(*changed)
    assert int(result['status_code']) == 1
    _compare({k: v.numpy() for k, v in result.items()}, {k: v.numpy() for k, v in reference(*changed).items()})
    assert enclosing.experimental_get_compiler_ir(*changed)(stage='hlo') == hlo
    assert enclosing.experimental_get_tracing_count() == 1
