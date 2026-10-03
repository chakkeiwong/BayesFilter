"""Public single-locator execution and actual consumer diagnostics."""

import dataclasses
import gc
import weakref
from threading import Thread

import pytest
import tensorflow as tf

from bayesfilter.inference import joint_center as public
from bayesfilter.inference import joint_center_tf as native
from bayesfilter.inference.program_cache_scope import _CURRENT_SCOPE, ProgramCacheScope
from tests import test_filter_repair_joint_center_native as existing
from tests.test_filter_repair_block_capture import stable_hlo
from tests.test_filter_repair_geometry_control import clean, save

D = tf.float64


@pytest.fixture(autouse=True)
def release_public_owner():
    native.clear_joint_center_cache()
    yield
    native.clear_joint_center_cache()


@pytest.mark.parametrize("dimension", [1, 3])
@pytest.mark.parametrize("case", ["quadratic", "quartic", "constant", "nonfinite", "cap", "iterations"])
def test_public_original_records(dimension, case, request):
    existing.test_native_joint_locator_original_records(dimension, case, request, public=True)


@pytest.mark.parametrize("case", ["interior", "failed", "accounting", "cap", "endpoint_invalid",
                                   "sentinel", "invalid_position"])
def test_public_synthetic_outcomes(case, monkeypatch, request):
    existing.test_native_joint_locator_synthetic_failures(case, monkeypatch, request, public=True)


def test_public_ownership_operands_and_release(request):
    class Target:
        def __init__(self, center):
            self.center = tf.constant(center, D)

        def __eq__(self, other):
            return isinstance(other, Target)

        def score(self, point):
            delta = point - self.center
            return -.5 * tf.reduce_sum(delta ** 2), -delta

    target = Target([.2, -.1, .4])
    point, scale = tf.constant([.7, .5, -.3], D), tf.constant([1., .8, 1.2], D)
    config = public.JointCenterLocatorConfig(max_iterations=10)
    outside = ProgramCacheScope()
    with outside.activate():
        first = public.locate_joint_center(target.score, point, scale=scale, config=config)
        owner = native.joint_center_program(target.score, 3, config, device=point.device)
        assert _CURRENT_SCOPE.get() is outside
        before = stable_hlo(owner.experimental_get_compiler_ir(point, scale)(stage="hlo"))
        changed = public.locate_joint_center(target.score, point + .11, scale=scale * .9, config=config)
        replay = public.locate_joint_center(target.score, point, scale=scale, config=config)
    assert clean(dataclasses.asdict(first)) == clean(dataclasses.asdict(replay))
    assert changed.initial_position.numpy().tolist() != first.initial_position.numpy().tolist()
    assert owner is native.joint_center_program(target.score, 3, config, device=point.device)
    assert owner.experimental_get_tracing_count() == 1
    assert before == stable_hlo(owner.experimental_get_compiler_ir(point + .11, scale * .9)(stage="hlo"))
    graph = owner.get_concrete_function().graph
    owner_ref, graph_ref = weakref.ref(owner), weakref.ref(graph)
    capped_cfg = dataclasses.replace(config, max_objective_evaluations=1)
    capped = public.locate_joint_center(target.score, point, scale=scale, config=capped_cfg)
    assert capped.cap_exhausted and not capped.endpoint_accepted
    assert native.joint_center_program(target.score, 3, capped_cfg, device=point.device) is not owner
    del owner, graph
    gc.collect()
    assert owner_ref() is None and graph_ref() is None
    target_ref = weakref.ref(target)
    replacement = Target([.1, .3, -.2])
    owner = native.joint_center_program(replacement.score, 3, config, device=point.device)
    del target
    gc.collect()
    assert target_ref() is None
    assert owner.dependency_scope is not outside
    save(request, "joint-public-owner.json", {"changed_operands_same_hlo": True,
        "one_trace": True, "method_and_receiver_identity": True, "configuration_isolation": True,
        "python_owner_graph_callback_collected": True, "native_eviction_proved": False})


def test_public_validation_graph_label_and_frozen_derivatives(monkeypatch):
    from bayesfilter import inference

    assert inference.locate_joint_center is public.locate_joint_center
    calls = []

    def target(point):
        calls.append("traced")
        return -.5 * tf.reduce_sum((point - .2) ** 2), .2 - point

    def forbidden(*args, **kwargs):
        raise AssertionError("Ordinary public call reached wall diagnostic")

    monkeypatch.setattr(public, "_locate_joint_center_wall_diagnostic", forbidden)
    for point, scale in (([], None), ([[1.]], None), ([1.], [0.]), ([1.], [1., 2.]),
                         ([1.], [float("nan")]), ([float("inf")], None)):
        with pytest.raises(ValueError):
            public.locate_joint_center(target, point, scale=scale)
    assert not calls
    point, scale = tf.Variable([.7], dtype=D), tf.Variable([1.2], dtype=D)
    with tf.GradientTape() as tape:
        result = public.locate_joint_center(target, point, scale=scale)
        returned = result.endpoint_position + result.best_evaluated_position + result.endpoint_score
    assert result.jit_compile and result.endpoint_accepted
    assert tape.gradient(returned, (point, scale)) == (None, None)
    graph_cfg = public.JointCenterLocatorConfig(jit_compile=False)
    graph_result = public.locate_joint_center(target, point, scale=scale, config=graph_cfg)
    owner = native.joint_center_program(target, 1, graph_cfg, device=point.device)
    assert not graph_result.jit_compile and not owner.function_spec.jit_compile


def test_public_wall_diagnostic_and_host_callback():
    from tests import test_joint_center as api

    api.test_non_xla_wall_guard_returns_typed_timeout_without_target_overrun()
    executed = []

    def host(point):
        value = tf.py_function(lambda: executed.append(True) or tf.constant(0., D), [], D)
        return value, tf.ones_like(point)

    with pytest.raises(ValueError, match="unsupported callback"):
        public.locate_joint_center(host, [.4])
    assert executed == []


@pytest.mark.parametrize("name", ["test_joint_center_api_is_exported_from_inference_package",
    "test_config_defaults_to_xla_and_validates_positive_fields",
    "test_default_xla_and_non_xla_endpoints_match",
    "test_hard_cap_guards_attempt_602_before_target_evaluation",
    "test_nonfinite_initial_target_fails_closed_before_optimizer",
    "test_optimizer_exception_falls_back_without_endpoint_promotion",
    "test_result_payload_is_array_free_and_has_no_geometry_authority",
    "test_same_state_continuation_matches_one_shot_optimizer",
    "test_staged_global_cap_can_fire_only_after_checkpoint"])
def test_existing_public_api(name, monkeypatch):
    from tests import test_joint_center as api

    function = getattr(api, name)
    if "monkeypatch" in function.__annotations__:
        function(monkeypatch)
    else:
        function()


def test_actual_posterior_consumer(monkeypatch, request):
    from bayesfilter.inference import posterior_local_initializer as consumer
    from tests import test_posterior_local_initializer as assertions

    records = []

    def observe(*args, **kwargs):
        result = public.locate_joint_center(*args, **kwargs)
        records.append(clean(dataclasses.asdict(result)))
        return result

    assert consumer.locate_joint_center is public.locate_joint_center
    monkeypatch.setattr(consumer, "locate_joint_center", observe)
    assertions.test_gaussian_recovers_location_and_physical_covariance_under_scaling(True)
    assert len(records) == 1 and records[0]["jit_compile"]
    save(request, "joint-public-posterior-consumer.json", {"records": records,
        "outer_initializer_host_migration_complete": False})


def test_actual_quadratic_consumer(monkeypatch, request):
    from bayesfilter.inference import quadratic_map_covariance as consumer
    from tests import test_quadratic_map_covariance as assertions

    records = []

    def observe(*args, **kwargs):
        result = public.locate_joint_center(*args, **kwargs)
        records.append(clean(dataclasses.asdict(result)))
        return result

    assert consumer.locate_joint_center is public.locate_joint_center
    monkeypatch.setattr(consumer, "locate_joint_center", observe)
    assertions.test_enabled_locator_is_finite_locator_only_not_covariance_authority()
    assertions.test_locator_exception_falls_back_to_initial_position_for_geometry(monkeypatch)
    assert len(records) == 2 and all(record["jit_compile"] for record in records)
    assert records[1]["status"] == "optimizer_exception"
    save(request, "joint-public-quadratic-consumer.json", {"records": records})


def test_real_compiler_failure_and_consumer_fallback(monkeypatch, request):
    from bayesfilter.inference import quadratic_map_covariance as consumer

    optimizer = native.tfp.optimizer.lbfgs_minimize
    callbacks = tf.Variable(0, dtype=tf.int64)
    eager_calls = []

    def injected(*args, **kwargs):
        result = optimizer(*args, **kwargs)
        incompatible = tf.ensure_shape(tf.py_function(
            lambda: eager_calls.append(True) or tf.constant(0., D), [], D), [])
        return result._replace(objective_value=result.objective_value + incompatible)

    def callback(point):
        count = callbacks.assign_add(1)
        with tf.control_dependencies([count]):
            return -.5 * tf.reduce_sum((point - .2) ** 2), .2 - point

    monkeypatch.setattr(native.tfp.optimizer, "lbfgs_minimize", injected)
    point, scale = tf.constant([.7], D), tf.constant([1.], D)
    cfg = public.JointCenterLocatorConfig(max_iterations=3)
    with pytest.raises(tf.errors.OpError, match="EagerPyFunc") as caught:
        public.locate_joint_center(callback, point, scale=scale, config=cfg)
    owner = native.joint_center_program(callback, 1, cfg, device=point.device)
    assert not owner.construction_error and not eager_calls and int(callbacks) == 0
    released = []

    def inspect_lock():
        acquired = owner.invocation_lock.acquire(timeout=1.)
        released.append(acquired)
        if acquired:
            owner.invocation_lock.release()

    thread = Thread(target=inspect_lock, daemon=True)
    thread.start()
    thread.join(timeout=2.)
    assert released == [True]
    fallback, diagnostic = consumer._run_locator(value_and_score_fn=callback,
        initial_position=point, initial_value=-.125, initial_score=tf.constant([-.5], D),
        config=consumer.QuadraticMapCovarianceLocatorConfig(enabled=True, max_iterations=3))
    assert diagnostic["status"] == "tfp_lbfgs_locator_exception_initial_fallback"
    assert diagnostic["exception_type"] == type(caught.value).__name__
    assert fallback.numpy().tolist() == point.numpy().tolist()
    assert not eager_calls and int(callbacks) == 0
    monkeypatch.setattr(native.tfp.optimizer, "lbfgs_minimize", optimizer)
    native.clear_joint_center_cache()
    recovered = public.locate_joint_center(callback, point, scale=scale, config=cfg)
    assert recovered.endpoint_accepted and int(callbacks) == recovered.physical_target_rows
    save(request, "joint-public-compiler-error.json", {"error": type(caught.value).__name__,
        "eager_fallback_calls": eager_calls, "lock_released": True,
        "consumer_diagnostic": diagnostic, "fresh_owner_recovers": True})
