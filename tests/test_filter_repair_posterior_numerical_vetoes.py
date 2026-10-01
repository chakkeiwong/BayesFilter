"""Original-record checks for finite analytical scores whose differences overflow."""

import dataclasses
import json
from pathlib import Path

import pytest
import tensorflow as tf

from bayesfilter.inference.posterior_curvature_report import posterior_curvature_result
from bayesfilter.inference.posterior_curvature_tf import (
    make_posterior_curvature_controller,
)
from tests.test_filter_repair_posterior_curvature import reference
from tests.test_filter_repair_posterior_curvature_extras import pure_fixture
from tests.test_filter_repair_quadratic_batches import _equal_records


@pytest.mark.parametrize('amplitude', [1e4, 1.4e308])
def test_finite_scores_preserve_original_numerical_rejection(amplitude, request):
    _, eligibility, cfg, arguments = pure_fixture(1)
    arguments = (tf.constant([-.01], tf.float64), arguments[1], arguments[2])
    scale = tf.constant(amplitude, tf.float64)

    def callback(points):
        # These are the value and analytical score of a Laplace target away
        # from its cusp. Every evaluated point/value/score is finite here;
        # subtracting opposite large scores can nevertheless overflow.
        return -scale * tf.reduce_sum(tf.abs(points), axis=1), -scale * tf.sign(points)

    expected, hashes = reference(callback, eligibility, cfg, arguments)
    records = {}
    for jit in (False, True):
        program = make_posterior_curvature_controller(callback, eligibility, 1, cfg, jit_compile=jit)
        try:
            raw = program(*arguments)
            records[str(jit)] = posterior_curvature_result(raw, arguments[0], arguments[1],
                dataclasses.replace(cfg, seed=int(arguments[2]))).payload()
        except tf.errors.InvalidArgumentError as error:
            records[str(jit)] = {'unexpected_exception': type(error).__name__, 'message': str(error)}
    directory = Path(request.config.getoption('xmlpath')).parent
    with (directory / f'posterior-finite-score-overflow-{amplitude}.json').open('x') as output:
        json.dump({'amplitude': amplitude, 'original_source_sha256': hashes,
            'original': expected, 'candidate': records,
            'role': 'complete original numerical-rejection comparison; no tolerance allowance'},
            output, indent=2, allow_nan=False)
        output.write('\n')
    for result in records.values():
        _equal_records(result, expected)
