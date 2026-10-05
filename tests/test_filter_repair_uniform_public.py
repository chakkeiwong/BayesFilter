"""Public uniform wiring keeps all original records and callback semantics."""

import dataclasses

import pytest
import tensorflow as tf

from bayesfilter.inference.batched_quadratic_center import (
    refine_batched_quadratic_center,
)
from tests.test_filter_repair_quadratic_batches import _equal_records
from tests.test_filter_repair_quadratic_rounds import public_record, reference
from tests.test_filter_repair_uniform_rounds import fixture


@pytest.mark.parametrize("dimension", [3, 5])
@pytest.mark.parametrize("jit", [False, True])
@pytest.mark.parametrize("case", ["nonquadratic", "initial_evidence", "bad_evidence", "nonspd"])
def test_uniform_public_original(dimension, jit, case):
    callback, config, args = fixture(dimension, case)
    expected, traces = reference(callback, config, args)
    actual = refine_batched_quadratic_center(callback, args[0], args[1],
        config=dataclasses.replace(config, jit_compile_trust=jit),
        _initial_evidence=(args[4], args[5]) if bool(args[3]) else None).payload()
    _equal_records(public_record(actual, jit=jit, original_traces=traces), expected)


@pytest.mark.parametrize("jit", [False, True])
def test_public_uniform_reuses_controller_with_dynamic_target_and_inputs(jit):
    from bayesfilter.inference.quadratic_rounds_tf import (
        clear_paired_quadratic_controller_cache,
        quadratic_controller,
    )

    clear_paired_quadratic_controller_cache()
    base, config, args = fixture(3, "nonquadratic", rows=33, seed=-1729, width=.03)
    strength = tf.Variable(1., dtype=tf.float64)

    def callback(points):
        values, scores, valid = base(points)
        return strength * values, strength * scores, valid

    first = None
    for index, multiplier in enumerate((1., 2.)):
        strength.assign(multiplier)
        changed = (args[0] + .01 * index, args[1] * (1 + .1 * index), args[2] + index, *args[3:])
        config = dataclasses.replace(config, seed=int(changed[2]), jit_compile_trust=jit)
        expected, traces = reference(callback, config, changed)
        result = refine_batched_quadratic_center(callback, changed[0], changed[1], config=config)
        program = quadratic_controller(callback, 3, config, jit_compile=jit)
        assert program.experimental_get_tracing_count() == 1
        if first is None:
            first = program
        assert first is program
        for value in (result.center, result.center_value, result.center_score):
            assert isinstance(value, tf.Tensor)
        if result.accepted:
            assert isinstance(result.pilot_factor, tf.Tensor)
            assert isinstance(result.precision_z, tf.Tensor)
        assert isinstance(result.diagnostics["rounds"][0]["anchor"], list)
        assert isinstance(result.diagnostics["candidate_batches"][0]["positions"], list)
        _equal_records(public_record(result.payload(), jit=jit, original_traces=traces), expected)
    clear_paired_quadratic_controller_cache()
