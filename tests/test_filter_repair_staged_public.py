"""Public staged execution, exact original records and ownership diagnostics."""

import dataclasses
import gc
import weakref

import pytest
import tensorflow as tf

from bayesfilter.inference import joint_center as public
from bayesfilter.inference import joint_center_staged_tf as native
from bayesfilter.inference.program_cache_scope import _CURRENT_SCOPE, ProgramCacheScope
from tests import test_filter_repair_staged_center as existing
from tests import test_filter_repair_staged_center_cost as cost
from tests.test_filter_repair_block_capture import stable_hlo
from tests.test_filter_repair_geometry_control import save

D = tf.float64


@pytest.fixture(autouse=True)
def release_public_owner():
    native.clear_staged_joint_center_cache()
    yield
    native.clear_staged_joint_center_cache()


@pytest.mark.parametrize("dimension,case", [
    (1, "quadratic"), (3, "quadratic"), (1, "quartic"), (3, "quartic"),
    (3, "constant"), (3, "invalid"), (3, "cap"), (3, "cap_after"),
    (3, "reject"), (3, "validator_error"),
])
def test_public_original_records(dimension, case, request):
    existing.test_staged_original_records(dimension, case, request, public=True)


@pytest.mark.parametrize("dimension", [1, 3])
@pytest.mark.parametrize("arm", ["prior", "graph", "xla"])
def test_public_cost(arm, dimension, request):
    cost.test_staged_center_complete_costs(arm, dimension, request, public=True)


@pytest.mark.parametrize("name", [
    "test_staged_checkpoint_is_private_immutable_and_validated_once",
    "test_staged_locator_retains_best_internal_callback_across_continuation",
    "test_checkpoint_rejection_prevents_continuation_target_calls",
    "test_checkpoint_validator_exception_prevents_continuation",
    "test_same_state_continuation_matches_one_shot_optimizer",
    "test_staged_global_cap_can_fire_only_after_checkpoint",
    "test_staged_finite_sentinel_endpoint_is_not_promoted",
])
def test_existing_consumers_use_public_xla(name, monkeypatch, request):
    existing.test_existing_staged_consumer_assertions_in_xla(name, monkeypatch, request, public=True)


@pytest.mark.parametrize("stage", ["checkpoint", "continuation"])
def test_public_native_error_is_not_a_result_or_fallback(stage, monkeypatch, request):
    existing.test_native_compilation_failure_propagates_without_fallback(stage, monkeypatch, request, public=True)


def test_public_nested_validator_restores_outer_state(request):
    existing.test_nested_validator_restores_outer_state_and_exact_calls(request, public=True)


def test_cache_identity_changed_operands_and_release(request):
    class Target:
        # Equal/unhashable callable objects must not share a compiled target.
        def __eq__(self, other):
            return isinstance(other, Target)

        def __init__(self, center):
            self.center = tf.constant(center, D)

        def __call__(self, point):
            delta = point - self.center
            return -.5 * tf.reduce_sum(delta ** 2), -delta

    target = Target([.2, -.1, .4])
    initial, scale = tf.constant([.7, .5, -.3], D), tf.constant([1., .8, 1.2], D)
    config = public.JointCenterStagedConfig(checkpoint_iterations=1, total_iterations=5)
    validator_scopes = []
    outside = ProgramCacheScope()

    def validator(_):
        validator_scopes.append(_CURRENT_SCOPE.get() is outside)
        return True

    with outside.activate():
        first = public.locate_joint_center_staged(target, initial, scale=scale, config=config,
                                                 checkpoint_validator=validator)
        owner = native.staged_joint_center_program(target, 3, config, device=initial.device)
        before_hlo = stable_hlo(owner.checkpoint.experimental_get_compiler_ir(initial, scale)(stage="hlo"))
        changed = public.locate_joint_center_staged(target, initial + .11, scale=scale * .9,
                                                   config=config, checkpoint_validator=validator)
        replay = public.locate_joint_center_staged(target, initial, scale=scale, config=config,
                                                  checkpoint_validator=validator)
    assert first.endpoint_accepted and changed.endpoint_accepted
    existing._equal_records(existing.clean(dataclasses.asdict(first)), existing.clean(dataclasses.asdict(replay)))
    assert validator_scopes == [True, True, True] and outside.program_count == 0
    assert owner is native.staged_joint_center_program(target, 3, config, device=initial.device)
    assert owner.checkpoint.experimental_get_tracing_count() == owner.continuation.experimental_get_tracing_count() == 1
    assert before_hlo == stable_hlo(owner.checkpoint.experimental_get_compiler_ir(initial + .11, scale * .9)(stage="hlo"))
    graph = owner.checkpoint.get_concrete_function().graph
    refs = {"callback": weakref.ref(target), "owner": weakref.ref(owner), "graph": weakref.ref(graph)}
    replacement = Target([.1, .3, -.2])
    replacement_owner = native.staged_joint_center_program(replacement, 3, config, device=initial.device)
    assert replacement_owner is not owner
    del target, owner, graph
    gc.collect()
    collected = {key: reference() is None for key, reference in refs.items()}
    save(request, "staged-public-ownership.json", {"one_trace_per_stage": True,
        "changed_operands_same_hlo": True, "validator_outside_owner_scope": validator_scopes,
        "replaced_python_owners_collected": collected})
    assert all(collected.values()), collected


def test_public_boundary_validation_and_frozen_derivatives(monkeypatch):
    from bayesfilter import inference

    assert inference.locate_joint_center_staged is public.locate_joint_center_staged
    def target(point):
        delta = point - .2
        return -.5 * tf.reduce_sum(delta ** 2), -delta

    def forbidden_legacy(*args, **kwargs):
        raise AssertionError("Ordinary public call reached host-clock diagnostic")

    monkeypatch.setattr(public, "_locate_joint_center_staged_wall_diagnostic", forbidden_legacy)
    for initial, scale in (([], None), ([[1.]], None), ([1.], [0.]), ([1.], [1., 2.]), ([1.], [float("nan")])):
        with pytest.raises(ValueError):
            public.locate_joint_center_staged(target, initial, scale=scale, checkpoint_validator=lambda _: True)
    initial, scale = tf.Variable([.7], dtype=D), tf.Variable([1.2], dtype=D)
    with tf.GradientTape() as tape:
        result = public.locate_joint_center_staged(target, initial, scale=scale,
            config=public.JointCenterStagedConfig(checkpoint_iterations=1, total_iterations=5),
            checkpoint_validator=lambda _: True)
        returned = result.endpoint_position + result.best_evaluated_position + result.endpoint_score
    assert result.jit_compile and result.endpoint_accepted
    assert tape.gradient(returned, (initial, scale)) == (None, None)


def test_bound_method_cache_identity_and_configuration(request):
    class Target:
        def score(self, point):
            delta = point - .2
            return -.5 * tf.reduce_sum(delta ** 2), -delta

    target = Target()
    point, scale = tf.constant([.5], D), tf.constant([1.2], D)
    config = public.JointCenterStagedConfig(checkpoint_iterations=1, total_iterations=5)
    first = public.locate_joint_center_staged(target.score, point, scale=scale, config=config,
                                            checkpoint_validator=lambda _: True)
    owner = native.staged_joint_center_program(target.score, 1, config, device=point.device)
    second = public.locate_joint_center_staged(target.score, point, scale=scale, config=config,
                                             checkpoint_validator=lambda _: True)
    existing._equal_records(existing.clean(dataclasses.asdict(first)), existing.clean(dataclasses.asdict(second)))
    assert owner is native.staged_joint_center_program(target.score, 1, config, device=point.device)
    assert owner.checkpoint.experimental_get_tracing_count() == owner.continuation.experimental_get_tracing_count() == 1
    changed = dataclasses.replace(config, max_objective_evaluations=1)
    capped = public.locate_joint_center_staged(target.score, point, scale=scale, config=changed,
                                             checkpoint_validator=lambda _: True)
    assert capped.cap_exhausted and not capped.endpoint_accepted
    replacement = native.staged_joint_center_program(target.score, 1, changed, device=point.device)
    assert replacement is not owner
    other = Target()
    assert native.staged_joint_center_program(other.score, 1, changed, device=point.device) is not replacement
    refs = [weakref.ref(target), weakref.ref(owner), weakref.ref(replacement)]
    del target, owner, replacement
    gc.collect()
    collected = [ref() is None for ref in refs]
    save(request, "staged-public-bound-method.json", {"one_trace_per_stage": True,
        "changed_config_rejected_at_declared_cap": capped.cap_exhausted,
        "distinct_receiver_not_reused": True, "python_owners_collected": collected})
    assert all(collected)


def test_explicit_graph_and_wall_diagnostics_remain_nondefault(monkeypatch):
    from tests import test_joint_center

    def target(point):
        return -.5 * tf.reduce_sum(point ** 2), -point

    config = public.JointCenterStagedConfig(checkpoint_iterations=1, total_iterations=2, jit_compile=False)
    result = public.locate_joint_center_staged(target, tf.constant([.5], D), config=config,
                                             checkpoint_validator=lambda _: True)
    assert not result.jit_compile
    with pytest.raises(ValueError, match="explicit non-JIT"):
        public._locate_joint_center_staged_wall_diagnostic(target, [.5], config=config,
                                                         checkpoint_validator=lambda _: True)
    test_joint_center.test_staged_global_wall_guard_can_fire_only_after_checkpoint(monkeypatch)
