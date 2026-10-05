"""Diagnostic exported initializer failure, cache and wall-clock boundaries."""

import gc
import weakref
from dataclasses import replace

import pytest
import tensorflow as tf

from bayesfilter import inference
from bayesfilter.inference import posterior_initializer_controller_tf as native
from bayesfilter.inference import posterior_local_initializer as public
from tests.test_filter_repair_geometry_control import clean, save
from tests.test_filter_repair_posterior_initializer_controller import original_module
from tests.test_filter_repair_posterior_initializer_public import configurations
from tests.test_filter_repair_quadratic_batches import _equal_records

D = tf.float64


@pytest.fixture(autouse=True)
def release_owner():
    native.clear_posterior_initializer_cache()
    yield
    native.clear_posterior_initializer_cache()


def test_exported_compiler_failure_cache_and_collection(monkeypatch, request):
    class UnsupportedTarget:
        def __init__(self):
            self.count = tf.Variable(0, dtype=tf.int64)
            self.eager_calls = []

        def scalar(self, point):
            if tf.executing_eagerly():
                self.eager_calls.append(True)
            increment = self.count.assign_add(1)
            with tf.control_dependencies([increment]):
                value = tf.strings.to_number(tf.strings.as_string(point[0], precision=17), out_type=D)
                return -.5 * value ** 2, -point

    def forbidden(*args, **kwargs):
        raise AssertionError("compiler errors must never select a host fallback")

    monkeypatch.setattr(public, "_initialize_posterior_wall_diagnostic", forbidden)
    config, movement, thresholds = configurations(1, "stationary")
    target = UnsupportedTarget()
    references, errors = [], []

    def invoke(current, cfg):
        with pytest.raises(tf.errors.OpError, match="AsString|StringToNumber") as caught:
            inference.initialize_posterior_local_location_scale(current.scalar, [.13], scale=[.8],
                config=cfg, movement_config=movement, curvature_thresholds=thresholds)
        errors.append(type(caught.value).__name__)
        assert not current.eager_calls and int(current.count) == 0
        owner = native._LAST_OWNER[2]
        references.append((weakref.ref(owner), weakref.ref(owner.compiled),
            weakref.ref(owner.dependency_scope)))
        return owner

    first = invoke(target, config)
    assert invoke(target, config) is first  # Newly materialized bound method, same receiver.
    assert first.compiled.experimental_get_tracing_count() == 1
    other = UnsupportedTarget()
    second = invoke(other, config)
    assert second is not first  # Same method definition, different receiver.
    third = invoke(other, replace(config, seed=(32, 43)))
    assert third is not second
    del first, second, third, target, other
    native.clear_posterior_initializer_cache()
    gc.collect()
    assert all(ref() is None for group in references for ref in group)
    save(request, "posterior-initializer-public-compiler-cache.json", {
        "error_types": errors, "physical_target_rows": 0, "eager_calls": 0,
        "bound_method_reused": True, "receiver_change_rebuilt": True,
        "configuration_change_rebuilt": True, "python_owners_programs_scopes_collected": True,
        "native_executable_eviction_proved": False})


def test_explicit_wall_diagnostic_is_labeled_and_separate(monkeypatch, request):
    reference, hashes = original_module()
    config, movement, thresholds = configurations(1, "stationary")
    config = replace(config, locator_config=replace(config.locator_config,
        jit_compile=False, max_wall_seconds=1.))
    trace_modes = []
    count = tf.Variable(0, dtype=tf.int64)

    def invalid(point):
        trace_modes.append(tf.executing_eagerly())
        increment = count.assign_add(1)
        with tf.control_dependencies([increment]):
            return tf.constant(float("nan"), D), tf.zeros_like(point)

    def forbidden(*args, **kwargs):
        raise AssertionError("explicit wall diagnostic must not construct the native owner")

    monkeypatch.setattr(native, "posterior_initializer_owner", forbidden)
    options = {"config": config, "movement_config": movement, "curvature_thresholds": thresholds}
    expected = reference.initialize_posterior_local_location_scale(invalid, [.13], **options).payload(include_arrays=True)
    actual = inference.initialize_posterior_local_location_scale(invalid, [.13], **options).payload(include_arrays=True)
    assert actual["diagnostics"].pop("execution") == "explicit_nondefault_host_wall_clock_diagnostic"
    _equal_records(clean(actual), clean(expected))
    assert actual["status"] == "initial_target_invalid" and int(count) == 2
    assert trace_modes == [False, False]  # The host controller still uses compiled callback helpers.
    with pytest.raises(ValueError, match="max_wall_seconds requires jit_compile=False"):
        replace(config.locator_config, jit_compile=True)
    save(request, "posterior-initializer-public-wall-diagnostic.json", {
        "reference_sha256": hashes, "actual": clean(actual), "expected": clean(expected),
        "explicit_nondefault_diagnostic": True, "physical_calls": int(count), "callback_trace_modes": trace_modes})
