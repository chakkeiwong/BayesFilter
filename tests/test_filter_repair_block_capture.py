"""Diagnostic qualification of dynamic complete-center operands in a block graph."""

import gc
import json
import re
import weakref
from difflib import unified_diff
from pathlib import Path

import pytest
import tensorflow as tf

from bayesfilter.inference.block_conditional_tf import ConditionalSequentialProgram
from bayesfilter.inference.block_coordinate_center import BlockCoordinateCenterBlock
from bayesfilter.inference.sequential_controller_report import sequential_result
from tests.filter_repair_frozen_checkpoint import FrozenCheckpoint
from tests.test_block_coordinate_center import _sequential_config
from tests.test_filter_repair_lifecycle_original import _compare_original

D = tf.float64


def stable_hlo(text):
    # Only dummy-source zero-node uniquifiers differ in03006. Preserve the
    # complete operation path, all operands/instructions/constants and other metadata.
    return re.sub(r'op_name="((?:StatefulPartitionedCall/)*zeros(?:_\d+)?)(?:/_\d+)+"(?= source_file="dummy_file_name" source_line=10)',
        r'op_name="\1/UNIQUE"', text)


@pytest.mark.parametrize('batched', [False, True])
def test_conditional_full_center_is_dynamic_and_matches_original(batched, request):
    precision = tf.constant([[4., 1., 2.], [1., 5., .5], [2., .5, 6.]], D)
    mode = tf.constant([1., -.5, .2], D)
    calls = tf.Variable(0, dtype=tf.int64)

    def scalar(point):
        calls.assign_add(1)
        delta = point - mode
        score = -tf.linalg.matvec(precision, delta)
        return .5 * tf.reduce_sum(delta * score), score

    def batch(points):
        calls.assign_add(tf.shape(points, out_type=tf.int64)[0])
        delta = points - mode[None]
        scores = -tf.einsum('ij,bj->bi', precision, delta)
        return .5 * tf.reduce_sum(delta * scores, axis=1), scores

    cfg = _sequential_config()
    block = BlockCoordinateCenterBlock('conditional', 1, 3, cfg)
    owner = ConditionalSequentialProgram(scalar, batch if batched else None, 3, block)
    assert int(calls) == 0
    checkpoint = FrozenCheckpoint('3582b4ac', 'block_capture_original')
    original = checkpoint.load('bayesfilter.inference.sequential_map_covariance')
    rows, hlo = [], []
    for center, scale in (([0., .1, .1], [1., 1., 1.]), ([.5, .15, .05], [1., .9, 1.1])):
        center, scale = tf.constant(center, D), tf.constant(scale, D)
        calls.assign(0)
        computed = owner.compiled(center, scale)
        actual = sequential_result(computed, cfg, 1, 2).payload()
        actual_calls = int(calls)
        calls.assign(0)

        def conditional(point, center=center):
            values, scores = scalar(tf.concat([center[:1], point], axis=0))
            return values, scores[1:]

        def conditional_batch(points, center=center):
            values, scores = batch(tf.concat([tf.broadcast_to(center[None, :1], [points.shape[0], 1]), points], axis=1))
            return values, scores[:, 1:]

        expected = original.estimate_sequential_map_covariance(conditional, center[None, 1:],
            batched_value_and_score_fn=conditional_batch if batched else None, scale=scale[1:], config=cfg).payload()
        rows.append({'center': center.numpy().tolist(), 'scale': scale.numpy().tolist(),
            'actual': actual, 'expected': expected, 'actual_calls': actual_calls, 'expected_calls': int(calls)})
        hlo.append(owner.compiled.experimental_get_compiler_ir(center, scale)(stage='hlo'))
    report = {'role': 'dynamic capture mechanism; not complete ordered-block qualification',
        'batched': batched, 'records': rows, 'original_source_sha256': checkpoint.hashes(),
        'trace_count': owner.compiled.experimental_get_tracing_count(),
        'hlo_raw_unchanged': hlo[0] == hlo[1],
        'hlo_unchanged': stable_hlo(hlo[0]) == stable_hlo(hlo[1])}
    root = Path(request.config.getoption('xmlpath')).parent
    with (root / f'block-capture-{batched}.json').open('x') as out:
        json.dump(report, out, indent=2)
        out.write('\n')
    (root / f'block-capture-{batched}.hlo.txt').write_text(hlo[0])
    (root / f'block-capture-{batched}-changed.hlo.txt').write_text(hlo[1])
    (root / f'block-capture-{batched}.hlo.diff').write_text('\n'.join(
        unified_diff(hlo[0].splitlines(), hlo[1].splitlines(), n=0)))
    for row in rows:
        _compare_original(row['actual'], row['expected'])
        assert row['actual_calls'] == row['expected_calls']
    assert rows[0]['actual']['map_candidate'] != rows[1]['actual']['map_candidate']
    assert report['trace_count'] == 1 and report['hlo_unchanged']
    graph = owner.compiled.get_concrete_function().graph.as_graph_def()
    operations = {node.op for nodes in (graph.node, *(fn.node_def for fn in graph.library.function)) for node in nodes}
    assert not operations & {'PyFunc', 'EagerPyFunc', 'PyFuncStateless'}
    assert owner.compiled.get_concrete_function().function_def.attr['_XlaMustCompile'].b


def test_hlo_metadata_normalization_preserves_executable_and_source_changes():
    source = 'constant(0), metadata={op_type="Const" op_name="StatefulPartitionedCall/zeros_24/_69" source_file="dummy_file_name" source_line=10}'
    assert stable_hlo(source) == stable_hlo(source.replace('/_69', '/_54'))
    for changed in (source.replace('constant(0)', 'constant(1)'), source.replace('zeros_24', 'zeros_25'),
                    source.replace('dummy_file_name', 'real.py'), source.replace('source_line=10', 'source_line=11')):
        assert stable_hlo(source) != stable_hlo(changed)


@pytest.mark.parametrize('batched', [False, True])
def test_conditional_owner_releases_callback_dependencies(batched, request):
    from tensorflow.python.framework import ops

    from bayesfilter.inference import block_coordinate_center as public

    def targets(multiplier):
        def batch(rows):
            return -.5 * multiplier * tf.reduce_sum(rows ** 2, axis=1), -multiplier * rows

        def scalar(row):
            values, scores = batch(row[None])
            return values[0], scores[0]

        return scalar, batch

    block = BlockCoordinateCenterBlock('owned', 1, 3, _sequential_config())
    center, scale = tf.zeros([3], D), tf.ones([3], D)
    scalar, batch = targets(1.)
    caches = (public._block_target_program, public._target_program)
    before = [factory.cache_info().currsize for factory in caches]
    owner = ConditionalSequentialProgram(scalar, batch if batched else None, 3, block)
    actual = sequential_result(owner.compiled(center, scale), block.sequential_config, 1, 2).payload()
    first_registry = set(ops._gradient_registry.list())
    refs = {'scalar': weakref.ref(scalar), 'batch': weakref.ref(batch), 'owner': weakref.ref(owner),
        'root': weakref.ref(owner.compiled), 'graph': weakref.ref(owner.compiled.get_concrete_function().graph),
        'scope': weakref.ref(owner.dependency_scope), 'controller': weakref.ref(owner.controller),
        'dependencies': weakref.ref(owner.controller.dependency_scope)}
    held = owner.compiled
    second_scalar, second_batch = targets(2.)
    successor = ConditionalSequentialProgram(second_scalar, second_batch if batched else None, 3, block)
    second = sequential_result(successor.compiled(center, scale), block.sequential_config, 1, 2).payload()
    new_registry_entries = sorted(set(ops._gradient_registry.list()) - first_registry)
    del owner, scalar, batch
    gc.collect()
    retained = {name: ref() is not None for name, ref in refs.items()}
    retained_result = sequential_result(held(center, scale), block.sequential_config, 1, 2).payload()
    del held
    gc.collect()
    released = {name: ref() is None for name, ref in refs.items()}
    after = [factory.cache_info().currsize for factory in caches]
    retained_graph = refs['graph']()
    retaining_graphs = []
    if retained_graph is not None:
        from tests.test_filter_repair_gap_followup import inference_caches

        # Diagnostic reachability through graph ancestry; no cache is cleared
        # to manufacture successful lifetime evidence.
        for obj in gc.get_objects():
            if isinstance(obj, tf.types.experimental.ConcreteFunction):
                graph, names, seen = obj.graph, [], set()
                while graph is not None and id(graph) not in seen:
                    seen.add(id(graph))
                    names.append(getattr(graph, 'name', type(graph).__name__))
                    if graph is retained_graph:
                        retaining_graphs.append(names)
                        break
                    graph = getattr(graph, 'outer_graph', None)
        for obj in gc.get_objects():
            if isinstance(obj, tf.Graph) and getattr(obj, 'outer_graph', None) is retained_graph:
                retaining_graphs.append(['graph_without_required_concrete', getattr(obj, 'name', type(obj).__name__)])
        cache_state = {name: factory.cache_info().currsize for name, factory in inference_caches().items()}
    else:
        cache_state = {}
    report = {'schema': 'filter_block_capture_ownership.v1', 'batched': batched,
        'retained': retained, 'released': released, 'cache_sizes_before': before,
        'cache_sizes_after': after, 'original_target': actual, 'retained_result': retained_result,
        'successor': second, 'retaining_graph_ancestry': retaining_graphs,
        'diagnostic_cache_state': cache_state,
        'same_signature_successor_new_gradient_entries': new_registry_entries,
        'nonclaims': ['Python collection does not establish native eviction.']}
    root = Path(request.config.getoption('xmlpath')).parent
    with (root / f'block-capture-ownership-{batched}.json').open('x') as out:
        json.dump(report, out, indent=2, allow_nan=False)
        out.write('\n')
    assert retained['scalar']
    assert all(released.values()), released
    assert actual == retained_result
    assert actual['accepted'] and second['accepted']
    tf.debugging.assert_near(tf.constant(second['precision'], D), 2. * tf.constant(actual['precision'], D),
        rtol=1e-10, atol=1e-10)
    assert before == after
    assert new_registry_entries == []
