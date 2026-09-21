"""Additional uniform fixed-batch configuration and enclosed-program checks."""

import dataclasses

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.inference.quadratic_rounds_tf import make_quadratic_controller
from tests.test_filter_repair_quadratic_batches import _equal_records
from tests.test_filter_repair_quadratic_rounds import materialize, reference
from tests.test_filter_repair_uniform_rounds import fixture


@pytest.mark.parametrize("dimension,batch,rows", [(3, 2, 3), (5, 8, 7)])
def test_nondefault_batches_square_design_and_enclosure(dimension, batch, rows):
    base, config, args = fixture(dimension, 'nonquadratic', rows=rows, seed=-19, width=.03)

    def callback(points):
        values, scores, _ = base(points)
        return values, scores, tf.ones([batch], tf.bool)

    config = dataclasses.replace(config, batch_size=batch)
    kernel = make_quadratic_controller(callback, dimension, config)
    expected, _ = reference(callback, config, args)
    computed = kernel(*args)
    _equal_records(materialize(computed, config, dimension, int(args[2])), expected)
    enclosing = tf.function(lambda *values: kernel(*values), input_signature=kernel.input_signature,
        autograph=False, jit_compile=True)
    enclosed = enclosing(*args)
    for left, right in zip(tf.nest.flatten(enclosed), tf.nest.flatten(computed), strict=True):
        np.testing.assert_allclose(left, right, atol=1e-10, rtol=1e-10)
    assert enclosing.experimental_get_tracing_count() == kernel.experimental_get_tracing_count() == 1
