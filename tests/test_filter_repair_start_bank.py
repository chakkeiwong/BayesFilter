"""Small XLA diagnostic for the unchanged chronological start-bank rule."""

import numpy as np
import tensorflow as tf

from bayesfilter.inference.hmc_warmup import _start_bank_geometry_and_eligibility


def test_start_bank_selector_has_complete_xla_and_stable_signature():
    call = _start_bank_geometry_and_eligibility()
    tolerance = tf.constant(1e-4, tf.float64)
    states = tf.constant([[0., 0.], [1., 0.], [1., 0.], [2., 0.],
                          [3., 0.], [4., 0.]], tf.float64)
    result = call(states, tolerance)
    np.testing.assert_array_equal(result[4], [0, 1, 3, 4])
    assert int(result[5]) == 0 and int(result[6]) == 1
    assert call.get_concrete_function().function_def.attr["_XlaMustCompile"].b
    assert "HloModule" in call.experimental_get_compiler_ir(states, tolerance)(stage="hlo")
    changed = tf.tensor_scatter_nd_update(states, [[2, 0]], [1.5])
    np.testing.assert_array_equal(call(changed, tolerance)[4], [0, 1, 2, 3, 4])
    assert call.experimental_get_tracing_count() == 1
