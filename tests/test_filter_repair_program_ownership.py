"""Configuration scope isolation and actual sequential callback lifetime checks."""

import gc
import json
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from bayesfilter.inference import sequential_controller_tf as native
from bayesfilter.inference.program_cache_scope import (
    ProgramCacheScope,
    scoped_program_cache,
)
from bayesfilter.inference.sequential_controller_report import sequential_result
from tests.test_filter_repair_gap_followup import inference_caches
from tests.test_filter_repair_sequential_controller import fixture


def test_factory_scope_reuse_isolation_and_standalone_compatibility():
    built = []

    @scoped_program_cache(maxsize=2)
    def factory(callback, signature):
        result = (callback, signature, object())
        built.append(weakref.ref(callback))
        return result

    target = lambda x: x
    first, second = ProgramCacheScope(), ProgramCacheScope()
    with first.activate():
        one = factory(target, 1)
        assert factory(target, 1) is one
        with second.activate():
            two = factory(target, 1)
            assert two is not one
        assert factory(target, 1) is one
    assert first.program_count == second.program_count == 1
    assert factory.cache_info().currsize == 0
    standalone = factory(target, 1)
    assert standalone is not one and standalone is not two
    assert factory(target, 1) is standalone
    assert factory.cache_info().currsize == 1
    assert factory.cache_parameters() == {'maxsize': 2, 'typed': False}
    factory.cache_clear()
    assert factory.cache_info().currsize == 0


def test_factory_scope_restores_context_after_failure_and_across_threads():
    @scoped_program_cache(maxsize=2)
    def factory(value):
        if value == 'failure':
            raise RuntimeError('controlled factory failure')
        return object()

    broken = ProgramCacheScope()
    with pytest.raises(RuntimeError, match='controlled'), broken.activate():
        factory('failure')
    assert broken.program_count == 0
    factory('standalone')
    assert factory.cache_info().currsize == 1
    barrier = threading.Barrier(2, timeout=20)

    def construct():
        scope = ProgramCacheScope()
        with scope.activate():
            value = factory('same signature')
            barrier.wait()
            assert factory('same signature') is value
        return value, scope.program_count

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(construct), pool.submit(construct)]
        first, second = [future.result(timeout=30) for future in futures]
    assert first[0] is not second[0]
    assert first[1] == second[1] == 1
    assert factory.cache_info().currsize == 1


@pytest.mark.parametrize('case', ['terminal', 'factor_one', 'batched_locator'])
def test_actual_owner_releases_nested_callbacks_and_keeps_retained_handle(case, request):
    scalar, batched, locator, cfg, starts, scale = fixture(case)
    count, dimension = int(starts.shape[0]), int(starts.shape[1])
    owner = native.sequential_controller(scalar, batched, locator, count, dimension,
        cfg, cfg.search_sample_count, progress=True, device=starts.device)
    first = sequential_result(owner(starts, scale), cfg, count, dimension).payload()
    assert native.sequential_controller(scalar, batched, locator, count, dimension,
        cfg, cfg.search_sample_count, progress=True, device=starts.device) is owner
    references = {'scalar': weakref.ref(scalar), 'batched': weakref.ref(batched),
        'owner': weakref.ref(owner), 'root': weakref.ref(owner.compiled),
        'scope': weakref.ref(owner.dependency_scope)}
    dependencies = {str(index): weakref.ref(program) for index, program
        in enumerate(owner.dependency_scope._programs.values())}
    assert len(dependencies) >= 8
    held = owner.compiled
    cache_sizes = {name: cached.cache_info().currsize for name, cached in inference_caches().items()}

    def observe():
        gc.collect()
        return {'root_refs': {name: ref() is not None for name, ref in references.items()},
            'dependency_refs': {name: ref() is not None for name, ref in dependencies.items()}}

    other_scalar, other_batched, other_locator, _, _, _ = fixture(case)
    successor = native.sequential_controller(other_scalar, other_batched, other_locator,
        count, dimension, cfg, cfg.search_sample_count, progress=True, device=starts.device)
    assert successor is not owner
    del owner, scalar, batched, locator
    stages = {'root_replaced_handle_retained': observe()}
    # Retaining a callable is a legitimate owner. Replacing the current public
    # owner cannot invalidate that compiled handle or mix target identities.
    retained_result = sequential_result(held(starts, scale), cfg, count, dimension).payload()
    retained_trace_count = held.experimental_get_tracing_count()
    del held
    stages['old_handle_released'] = observe()
    successor_result = sequential_result(successor(starts, scale), cfg, count, dimension).payload()
    successor_trace_count = successor.compiled.experimental_get_tracing_count()
    after_sizes = {name: cached.cache_info().currsize for name, cached in inference_caches().items()}
    successor_refs = [weakref.ref(other_scalar), weakref.ref(other_batched), weakref.ref(successor)]
    native.clear_sequential_controller_cache()
    del successor, other_scalar, other_batched, other_locator
    gc.collect()
    successor_alive = [ref() is not None for ref in successor_refs]
    report = {'case': case, 'stages': stages, 'standalone_cache_sizes_before': cache_sizes,
        'standalone_cache_sizes_after': after_sizes, 'result': first,
        'retained_result': retained_result, 'successor_result': successor_result,
        'retained_trace_count': retained_trace_count, 'successor_trace_count': successor_trace_count,
        'successor_refs_alive_after_clear': successor_alive,
        'nonclaims': ['Python callback/dependency collection does not prove native executable eviction.']}
    path = Path(request.config.getoption('xmlpath')).parent / f'sequential-owner-{case}.json'
    with path.open('x') as out:
        json.dump(report, out, indent=2)
        out.write('\n')
    assert stages['root_replaced_handle_retained']['root_refs']['scalar']
    assert retained_result == successor_result == first
    assert retained_trace_count == successor_trace_count == 1
    assert not any(stages['old_handle_released']['root_refs'].values())
    assert not any(stages['old_handle_released']['dependency_refs'].values())
    assert after_sizes == cache_sizes
    assert not any(successor_alive)


def test_distinct_public_target_identities_do_not_reuse_stale_graphs():
    import tensorflow as tf

    from bayesfilter.inference import sequential_map_covariance as public
    from tests.test_filter_repair_lifecycle_original import _compare_original

    _, _, _, cfg, starts, scale = fixture('terminal')

    def target(multiplier):
        def batched(rows):
            return -.5 * multiplier * tf.reduce_sum(rows ** 2, axis=1), -multiplier * rows

        def scalar(row):
            values, scores = batched(row[None])
            return values[0], scores[0]

        return scalar, batched

    first_scalar, first_batch = target(1.)
    first = public.estimate_sequential_map_covariance(first_scalar, starts,
        batched_value_and_score_fn=first_batch, scale=scale, config=cfg)
    first_owner = native._LAST_CONTROLLER[4]
    second_scalar, second_batch = target(2.)
    second = public.estimate_sequential_map_covariance(second_scalar, starts,
        batched_value_and_score_fn=second_batch, scale=scale, config=cfg)
    second_owner = native._LAST_CONTROLLER[4]
    assert first_owner is not second_owner
    assert first.accepted and second.accepted
    tf.debugging.assert_near(second.precision, 2. * first.precision, atol=1e-9, rtol=1e-9)
    _compare_original(sequential_result(first_owner(starts, scale), cfg, 1, 3).payload(), first.payload())
    _compare_original(sequential_result(second_owner(starts, scale), cfg, 1, 3).payload(), second.payload())
