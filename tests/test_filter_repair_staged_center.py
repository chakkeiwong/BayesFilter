"""Complete frozen staged-locator results, validator boundaries and operands."""

import ast
import dataclasses
import gc
import hashlib
import weakref
from threading import Thread

import pytest
import tensorflow as tf
import tensorflow_probability as tfp

from bayesfilter.inference.joint_center import JointCenterStagedConfig
from bayesfilter.inference.joint_center_staged_tf import (
    StagedJointCenterProgram,
    run_staged_program,
)
from tests.filter_repair_frozen_checkpoint import FrozenCheckpoint
from tests.test_filter_repair_block_capture import stable_hlo
from tests.test_filter_repair_geometry_control import clean, save
from tests.test_filter_repair_quadratic_batches import _equal_records

D = tf.float64


def original_staged_source(label):
    """Keep CPU authority untouched; record the minimal GPU resource adaptation."""
    original = FrozenCheckpoint("3582b4ac", label)
    previous = original.load("bayesfilter.inference.joint_center")
    compatibility = None
    if tf.config.list_logical_devices("GPU"):
        source = original.sources["bayesfilter/inference/joint_center.py"]
        function = next(node for node in ast.parse(source).body
            if isinstance(node, ast.FunctionDef) and node.name == "locate_joint_center_staged")
        text = ast.get_source_segment(source, function)
        # Original GPU failure03338 is preserved. Only attempts, target rows,
        # best callback index and the matching cap change integer width.
        assert text.count("tf.int32") == 4
        adapted = text.replace("tf.int32", "tf.int64")
        exec(compile(adapted, "3582b4ac:staged_int64_accounting_only", "exec"), previous.__dict__)  # noqa: S102
        compatibility = {"classification": "original_with_int64_accounting_only",
            "original_failure_run": "03338", "replacement": {"tf.int32": "tf.int64"},
            "occurrences": 4, "original_function_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "adapted_function_sha256": hashlib.sha256(adapted.encode()).hexdigest(),
            "float_arithmetic_changed": False}
    return original, previous, compatibility


@pytest.mark.parametrize("dimension,case", [(1, "quadratic"), (3, "quadratic"), (1, "quartic"),
    (3, "quartic"), (3, "constant"), (3, "invalid"), (3, "cap"), (3, "cap_after"), (3, "reject"), (3, "validator_error")])
def test_staged_original_records(dimension, case, request):
    original, previous, compatibility = original_staged_source("staged_center_original")
    counter = tf.Variable(0, dtype=tf.int64)
    positions = tf.Variable(tf.zeros([1200, dimension], D))
    precision = tf.linalg.diag(tf.cast(tf.range(dimension), D) + 1.3) + .07
    mode = .14 + tf.cast(tf.range(dimension), D) * .03

    def target_body(point):
        index = counter.assign_add(1) - 1
        update = positions.scatter_nd_update(index[None, None], point[None])
        with tf.control_dependencies([update]):
            delta = point - mode
            score = -tf.linalg.matvec(precision, delta)
            value = .5 * tf.reduce_sum(delta * score)
        if case == "quartic":
            value -= .03 * tf.reduce_sum(delta ** 4)
            score -= .12 * delta ** 3
        if case == "constant":
            value, score = tf.constant(1., D), tf.zeros_like(point)
        if case == "invalid":
            value = tf.constant(float("nan"), D)
        return value, score

    # Isolate candidate lifetime from the original's traced functions and logs.
    # A shared callback cannot distinguish which arm keeps a reference alive.
    def target(point):
        return target_body(point)

    def before_target(point):
        return target_body(point)

    config = JointCenterStagedConfig(checkpoint_iterations=1, total_iterations=10,
        gradient_tolerance=1e-8, max_objective_evaluations={"cap": 1, "cap_after": 3}.get(case, 120))
    before_config = previous.JointCenterStagedConfig(**dataclasses.asdict(config))
    owner = StagedJointCenterProgram(target, dimension, config)
    reports, hlos, continuing_hlos = [], [], []
    validator_calls = []

    def validator(checkpoint):
        validator_calls.append({"target_calls": int(counter), "checkpoint": clean(dataclasses.asdict(checkpoint))})
        if case == "validator_error":
            raise RuntimeError("checkpoint veto")
        return case != "reject"

    for shift in (0., .11, 0.):
        start = .6 + shift - tf.cast(tf.range(dimension), D) * .14
        scale = .8 + shift + tf.cast(tf.range(dimension), D) * .09
        counter.assign(0)
        validator_calls.clear()
        expected = previous.locate_joint_center_staged(before_target, start, scale=scale,
            checkpoint_validator=validator, config=before_config)
        expected_calls = {"calls": int(counter), "positions": positions[:int(counter)].numpy().tolist(),
            "validators": list(validator_calls)}
        counter.assign(0)
        validator_calls.clear()
        actual = run_staged_program(owner, start, scale, validator)
        actual_calls = {"calls": int(counter), "positions": positions[:int(counter)].numpy().tolist(),
            "validators": list(validator_calls)}
        reports.append({"actual": clean(dataclasses.asdict(actual)), "original": clean(dataclasses.asdict(expected)),
            "actual_calls": clean(actual_calls), "expected_calls": clean(expected_calls)})
        hlos.append(owner.checkpoint.experimental_get_compiler_ir(start, scale)(stage="hlo"))
        if actual.continuation_started:
            # Compiler IR takes runtime state operands. This additional checkpoint
            # is outside the recorded paired invocation and does not enter parity.
            counter.assign(0)
            state = owner.checkpoint(start, scale)
            continuing_hlos.append(owner.continuation.experimental_get_compiler_ir(start, scale, state)(stage="hlo"))
    first_graph = owner.checkpoint.get_concrete_function().graph
    references = {"owner": weakref.ref(owner), "checkpoint_graph": weakref.ref(first_graph), "callback": weakref.ref(target)}
    if owner.continuation.experimental_get_tracing_count():
        last_graph = owner.continuation.get_concrete_function().graph
        references["continuation_graph"] = weakref.ref(last_graph)
        del last_graph
    trace_counts = (owner.checkpoint.experimental_get_tracing_count(), owner.continuation.experimental_get_tracing_count())
    del first_graph, owner, target
    gc.collect()
    released = {key: ref() is None for key, ref in references.items()}
    report = {"records": reports, "original_sources": original.hashes(), "original_compatibility": compatibility,
        "trace_counts": trace_counts,
        "checkpoint_hlo_unchanged": len({stable_hlo(hlo) for hlo in hlos}) == 1,
        "continuation_hlo_unchanged": len({stable_hlo(hlo) for hlo in continuing_hlos}) <= 1,
        "checkpoint_hlo_sha256": hashlib.sha256(hlos[0].encode()).hexdigest(), "python_released": released}
    save(request, f"staged-center-{case}-{dimension}.json", report)
    for row in reports:
        _equal_records(row["actual"], row["original"])
        _equal_records(row["actual_calls"], row["expected_calls"])
        assert row["actual_calls"]["calls"] == row["actual"]["physical_target_rows"]
        if case == "cap_after":
            assert row["actual"]["checkpoint_validated"] and row["actual"]["continuation_started"]
            assert row["actual"]["status"] == "evaluation_cap_exhausted"
    assert trace_counts[0] == 1 and trace_counts[1] <= 1
    assert report["checkpoint_hlo_unchanged"] and report["continuation_hlo_unchanged"]
    assert all(released.values()), released


def test_staged_native_rejects_host_wall_guard():
    with pytest.raises(ValueError, match="parent wall deadline"):
        StagedJointCenterProgram(lambda point: (tf.reduce_sum(point), point), 1,
            JointCenterStagedConfig(jit_compile=False, max_wall_seconds=1.))


@pytest.mark.parametrize("jit", [False, True])
def test_staged_execution_label_comes_from_actual_program(jit):
    """Non-JIT is an explicit tiny diagnostic, never default GPU evidence."""
    def callback(point):
        return -.5 * tf.reduce_sum(point ** 2), -point

    owner = StagedJointCenterProgram(callback, 1,
        JointCenterStagedConfig(checkpoint_iterations=1, total_iterations=2, jit_compile=jit))
    result = run_staged_program(owner, tf.constant([.5], D), tf.constant([1.], D), lambda _: True)
    assert result.jit_compile is jit
    assert bool(owner.continuation.function_spec.jit_compile) is jit


@pytest.mark.parametrize("name", [
    "test_staged_checkpoint_is_private_immutable_and_validated_once",
    "test_staged_locator_retains_best_internal_callback_across_continuation",
    "test_checkpoint_rejection_prevents_continuation_target_calls",
    "test_checkpoint_validator_exception_prevents_continuation",
    "test_same_state_continuation_matches_one_shot_optimizer",
    "test_staged_global_cap_can_fire_only_after_checkpoint",
    "test_staged_finite_sentinel_endpoint_is_not_promoted",
])
def test_existing_staged_consumer_assertions_in_xla(name, monkeypatch, request):
    from tests import test_joint_center as consumers

    results = []

    def adapter(callback, initial_position, *, checkpoint_validator, scale=None, config=None):
        config = dataclasses.replace(config, jit_compile=True)
        initial = tf.convert_to_tensor(initial_position, D)
        scale = tf.ones_like(initial) if scale is None else tf.convert_to_tensor(scale, D)
        owner = StagedJointCenterProgram(callback, int(initial.shape[0]), config)
        result = run_staged_program(owner, initial, scale, checkpoint_validator)
        results.append(clean(dataclasses.asdict(result)))
        return result

    monkeypatch.setattr(consumers, "locate_joint_center_staged", adapter)
    function = getattr(consumers, name)
    if "monkeypatch" in function.__annotations__:
        function(monkeypatch)
    else:
        function()
    assert results
    save(request, name + ".json", {"consumer": name, "jit_compile": True, "results": results})


@pytest.mark.parametrize("stage", ["checkpoint", "continuation"])
def test_staged_optimizer_construction_failure_preserves_records(stage, monkeypatch, request):
    original, previous, compatibility = original_staged_source("staged_construction_error")
    optimizer = tfp.optimizer.lbfgs_minimize
    counter = tf.Variable(0, dtype=tf.int64)

    def injected(*args, **kwargs):
        is_continuation = kwargs.get("previous_optimizer_results") is not None
        if is_continuation == (stage == "continuation"):
            raise RuntimeError("injected optimizer construction error")
        return optimizer(*args, **kwargs)

    monkeypatch.setattr(tfp.optimizer, "lbfgs_minimize", injected)

    def callback(point):
        count = counter.assign_add(1)
        with tf.control_dependencies([count]):
            score = -(point - .25) * tf.constant([1., 2., 3.], D)
            return .5 * tf.reduce_sum((point - .25) * score), score

    start, scale = tf.constant([.5, -.4, .7], D), tf.constant([1., .7, 1.1], D)
    config = JointCenterStagedConfig(checkpoint_iterations=1, total_iterations=10, gradient_tolerance=1e-8)
    expected = previous.locate_joint_center_staged(callback, start, scale=scale,
        config=previous.JointCenterStagedConfig(**dataclasses.asdict(config)), checkpoint_validator=lambda _: True)
    expected_count = int(counter)
    counter.assign(0)
    owner = StagedJointCenterProgram(callback, 3, config)
    actual = run_staged_program(owner, start, scale, lambda _: True)
    report = {"original": clean(dataclasses.asdict(expected)), "actual": clean(dataclasses.asdict(actual)),
        "original_count": expected_count, "actual_count": int(counter), "original_sources": original.hashes(),
        "original_compatibility": compatibility}
    save(request, f"staged-construction-{stage}.json", report)
    _equal_records(report["actual"], report["original"])
    assert int(counter) == expected_count == actual.physical_target_rows
    assert actual.status == "optimizer_exception" and actual.exception_type == "RuntimeError"


def test_continuation_uses_supplied_state_after_unrelated_invocation(request):
    from bayesfilter.inference.joint_center_staged_tf import (
        checkpoint_record,
        staged_result,
    )

    original, previous, compatibility = original_staged_source("staged_restored_state")
    counter = tf.Variable(0, dtype=tf.int64)
    positions = tf.Variable(tf.zeros([1200, 3], D))

    def callback(point):
        index = counter.assign_add(1) - 1
        update = positions.scatter_nd_update(index[None, None], point[None])
        with tf.control_dependencies([update]):
            delta = point - tf.constant([.3, -.1, .5], D)
            score = -delta * tf.constant([1., 2., 3.], D) - .04 * delta ** 3
            value = -.5 * tf.reduce_sum(delta ** 2 * tf.constant([1., 2., 3.], D)) - .01 * tf.reduce_sum(delta ** 4)
        return value, score

    config = JointCenterStagedConfig(checkpoint_iterations=1, total_iterations=10, gradient_tolerance=1e-8)
    initial, scale = tf.constant([1., -.8, 1.2], D), tf.constant([.5, 1.1, 2.], D)
    expected = previous.locate_joint_center_staged(callback, initial, scale=scale,
        config=previous.JointCenterStagedConfig(**dataclasses.asdict(config)), checkpoint_validator=lambda _: True)
    expected_rows = positions[:int(counter)].numpy().tolist()
    counter.assign(0)
    first_owner = StagedJointCenterProgram(callback, 3, config)
    other_owner = StagedJointCenterProgram(callback, 3, config)
    first = first_owner.checkpoint(initial, scale)
    checkpoint_rows = positions[:int(counter)].numpy().tolist()
    # Populate a different owner's resources with a different start/scale, then
    # resume the supplied checkpoint on it. No resource value may be implicit.
    other_owner.checkpoint(initial + .2, scale * .7)
    counter.assign(0)
    final = other_owner.continuation(initial, scale, first)
    actual_rows = checkpoint_rows + positions[:int(counter)].numpy().tolist()
    actual = staged_result(final, checkpoint_record(first[2]), validated=True, validator_calls=1,
        continuation_started=True, jit_compile=True)
    report = {"actual": clean(dataclasses.asdict(actual)), "original": clean(dataclasses.asdict(expected)),
        "actual_calls": actual_rows, "expected_calls": expected_rows, "original_sources": original.hashes(),
        "original_compatibility": compatibility}
    save(request, "staged-restored-state.json", report)
    _equal_records(report["actual"], report["original"])
    _equal_records(actual_rows, expected_rows)
    assert other_owner.continuation.experimental_get_tracing_count() == 1


@pytest.mark.parametrize("stage", ["checkpoint", "continuation"])
def test_native_compilation_failure_propagates_without_fallback(stage, monkeypatch, request):
    """Real unsupported XLA op: never execute its Python body or retry eagerly."""
    optimizer = tfp.optimizer.lbfgs_minimize
    fallback_calls, validator_calls = [], []
    counter = tf.Variable(0, dtype=tf.int64)

    def forbidden_fallback():
        fallback_calls.append(True)
        return tf.constant(0., D)

    def injected(*args, **kwargs):
        result = optimizer(*args, **kwargs)
        if (kwargs.get("previous_optimizer_results") is not None) == (stage == "continuation"):
            incompatible = tf.ensure_shape(tf.py_function(forbidden_fallback, [], D), [])
            result = result._replace(objective_value=result.objective_value + incompatible)
        return result

    def callback(point):
        count = counter.assign_add(1)
        with tf.control_dependencies([count]):
            delta = point - tf.constant([.2, -.1, .4], D)
            score = -delta * tf.constant([1., 2., 3.], D)
            return .5 * tf.reduce_sum(delta * score), score

    def validator(checkpoint):
        validator_calls.append(clean(dataclasses.asdict(checkpoint)))
        return True

    monkeypatch.setattr(tfp.optimizer, "lbfgs_minimize", injected)
    config = JointCenterStagedConfig(checkpoint_iterations=1, total_iterations=10, gradient_tolerance=1e-8)
    owner = StagedJointCenterProgram(callback, 3, config)
    start, scale = tf.constant([.8, -.4, 1.2], D), tf.constant([.7, 1.1, 1.3], D)
    with pytest.raises(tf.errors.OpError) as caught:
        run_staged_program(owner, start, scale, validator)
    failed_rows = int(counter)
    assert "EagerPyFunc" in str(caught.value)
    assert fallback_calls == []
    assert len(validator_calls) == int(stage == "continuation")
    assert not owner.construction_errors
    if stage == "checkpoint":
        assert failed_rows == 0
        assert owner.continuation.experimental_get_tracing_count() == 0
    else:
        assert failed_rows == validator_calls[0]["physical_target_rows"]

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
    monkeypatch.setattr(tfp.optimizer, "lbfgs_minimize", optimizer)
    fresh = StagedJointCenterProgram(callback, 3, config)
    counter.assign(0)
    recovered = run_staged_program(fresh, start, scale, lambda _: True)
    assert recovered.endpoint_accepted and recovered.continuation_started
    assert int(counter) == recovered.physical_target_rows
    save(request, f"staged-native-failure-{stage}.json", {
        "stage": stage, "exception_type": type(caught.value).__name__,
        "message": str(caught.value), "fallback_calls": fallback_calls,
        "validator_calls": validator_calls, "failed_target_rows": failed_rows,
        "lock_released": released[0], "fresh_result": clean(dataclasses.asdict(recovered))})


def test_nested_validator_restores_outer_state_and_exact_calls(request):
    """A reentrant validator must not contaminate the outer optimizer state."""
    original, previous, compatibility = original_staged_source("staged_nested_validator")
    counter = tf.Variable(0, dtype=tf.int64)
    positions = tf.Variable(tf.zeros([1200, 3], D))

    def callback(point):
        index = counter.assign_add(1) - 1
        update = positions.scatter_nd_update(index[None, None], point[None])
        with tf.control_dependencies([update]):
            delta = point - tf.constant([.3, -.1, .5], D)
            score = -delta * tf.constant([1., 2., 3.], D) - .04 * delta ** 3
            value = -.5 * tf.reduce_sum(delta ** 2 * tf.constant([1., 2., 3.], D)) - .01 * tf.reduce_sum(delta ** 4)
        return value, score

    config = JointCenterStagedConfig(checkpoint_iterations=1, total_iterations=10, gradient_tolerance=1e-8)
    start, scale = tf.constant([1., -.8, 1.2], D), tf.constant([.5, 1.1, 2.], D)
    owner = StagedJointCenterProgram(callback, 3, config)
    before_config = previous.JointCenterStagedConfig(**dataclasses.asdict(config))

    def prior(point, units, validator):
        return previous.locate_joint_center_staged(callback, point, scale=units,
            config=before_config, checkpoint_validator=validator)

    def candidate(point, units, validator):
        return run_staged_program(owner, point, units, validator)

    records = []
    for execute in (prior, candidate):
        counter.assign(0)
        nested, calls = [], []

        def validator(checkpoint, execute=execute, nested=nested, calls=calls):
            calls.append({"checkpoint": clean(dataclasses.asdict(checkpoint)), "target_calls": int(counter)})
            inner = execute(start + .2, scale * .7, lambda _: True)
            nested.append(clean(dataclasses.asdict(inner)))
            return True

        outer = execute(start, scale, validator)
        records.append({"outer": clean(dataclasses.asdict(outer)), "inner": nested,
            "validator_calls": calls, "positions": positions[:int(counter)].numpy().tolist(),
            "target_rows": int(counter)})
    report = {"original": records[0], "actual": records[1], "original_sources": original.hashes(),
        "original_compatibility": compatibility}
    save(request, "staged-nested-validator.json", report)
    _equal_records(report["actual"], report["original"])
    assert len(records[1]["validator_calls"]) == len(records[1]["inner"]) == 1
    assert records[1]["target_rows"] == records[1]["outer"]["physical_target_rows"] + records[1]["inner"][0]["physical_target_rows"]
    assert owner.checkpoint.experimental_get_tracing_count() == owner.continuation.experimental_get_tracing_count() == 1
