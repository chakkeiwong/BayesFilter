"""Uniform-controller resource ownership and changing-target reference checks."""

import gc
import json
import weakref
from pathlib import Path

import tensorflow as tf

from bayesfilter.inference import quadratic_rounds_tf as runtime
from tests.test_filter_repair_quadratic_batches import _equal_records
from tests.test_filter_repair_quadratic_round_cache import EqualCallable
from tests.test_filter_repair_quadratic_rounds import materialize, reference
from tests.test_filter_repair_uniform_rounds import fixture


def scaled_callback(base, strength):
    def callback(points):
        values, scores, valid = base(points)
        return strength * values, strength * scores, valid
    return callback


def test_uniform_target_state_and_resources(request):
    runtime.clear_paired_quadratic_controller_cache()
    counter = tf.Variable(0, dtype=tf.int64)
    strength = tf.Variable(1., dtype=tf.float64)
    base, config, args = fixture(3, 'move', counter=counter)

    callback = scaled_callback(base, strength)
    target = EqualCallable(callback)
    kernel = runtime.quadratic_controller(target, 3, config)
    records = []
    for multiplier in (1., 2.):
        strength.assign(multiplier)
        counter.assign(0)
        expected, _ = reference(target, config, args)
        expected_calls = int(counter)
        counter.assign(0)
        computed = kernel(*args)
        assert int(counter) == expected_calls == int(computed['callback_batches'])
        actual = materialize(computed, config, 3, int(args[2]))
        _equal_records(actual, expected)
        records.append(actual)
        assert runtime.quadratic_controller(target, 3, config) is kernel
    assert records[0] != records[1]
    assert kernel.experimental_get_tracing_count() == 1
    concrete = kernel.get_concrete_function()
    watched = {name: weakref.ref(value) for name, value in
        (('graph', concrete.graph), ('counter', counter), ('strength', strength), ('callback', target))}
    replacement, _, _ = fixture(3, 'centered')
    replacement = EqualCallable(replacement, offset=10.)
    assert target == replacement
    assert runtime.quadratic_controller(replacement, 3, config) is not kernel
    runtime.clear_paired_quadratic_controller_cache()
    del kernel, concrete, target, callback, base, counter, strength, computed
    gc.collect()
    released = {name: reference() is None for name, reference in watched.items()}
    directory = Path(request.config.getoption('xmlpath')).parent
    with (directory / 'uniform-resource-ownership.json').open('x') as handle:
        json.dump({'released': released, 'changed_target_original_records': records,
            'trace_count': 1, 'cache_capacity': 1,
            'nonclaim': 'Python resource collection does not establish native executable eviction.'},
            handle, indent=2, allow_nan=False)
        handle.write('\n')
    assert all(released.values())
