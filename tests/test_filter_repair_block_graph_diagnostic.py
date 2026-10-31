"""Diagnostic-only attribution of TensorFlow registry retention after tracing."""

import gc
import json
import types
import weakref
from collections import deque
from pathlib import Path

import tensorflow as tf
from tensorflow.python.framework import ops

from bayesfilter.inference.block_conditional_tf import ConditionalSequentialProgram
from bayesfilter.inference.block_coordinate_center import BlockCoordinateCenterBlock
from tests.test_block_coordinate_center import _sequential_config


def _ancestry(graph, target):
    names, seen = [], set()
    while graph is not None and id(graph) not in seen:
        seen.add(id(graph))
        names.append(getattr(graph, 'name', type(graph).__name__))
        if graph is target:
            return names
        graph = getattr(graph, 'outer_graph', None)
    return None


def _closure_paths(function, target):
    queue = deque([(function, [], 0)])
    seen, paths, visited = set(), [], 0
    depth_truncated = False
    while queue and visited < 10000:
        value, path, depth = queue.popleft()
        if id(value) in seen:
            continue
        seen.add(id(value))
        visited += 1
        graph = value if isinstance(value, tf.Graph) else getattr(value, 'graph', None)
        if isinstance(graph, tf.Graph):
            ancestry = _ancestry(graph, target)
            if ancestry is not None:
                paths.append({'closure_path': path, 'type': type(value).__name__,
                    'name': getattr(value, 'name', None), 'graph_ancestry': ancestry})
            continue
        children = []
        if isinstance(value, types.FunctionType):
            label = f'{value.__qualname__}@{value.__code__.co_filename}:{value.__code__.co_firstlineno}'
            for name, cell in zip(value.__code__.co_freevars, value.__closure__ or (), strict=True):
                try:
                    children.append((cell.cell_contents, f'{label}.closure[{name}]'))
                except ValueError:
                    pass
            children.extend((item, f'{label}.default[{index}]')
                for index, item in enumerate(value.__defaults__ or ()))
        elif isinstance(value, (tuple, list)):
            children = [(item, f'[{index}]') for index, item in enumerate(value)]
        elif isinstance(value, dict):
            children = [(item, f'[{key}]') for key, item in value.items()]
        if children and depth >= 12:
            depth_truncated = True
            continue
        queue.extend((item, [*path, label], depth + 1) for item, label in children)
    return {'paths': paths, 'visited': visited,
        'truncated': bool(queue) or depth_truncated}


def test_trace_only_registry_ownership_attribution(request):
    registry = ops._gradient_registry
    before = set(registry.list())

    def scalar(row):
        return -.5 * tf.reduce_sum(row ** 2), -row

    owner = ConditionalSequentialProgram(scalar, None, 3,
        BlockCoordinateCenterBlock('trace_only', 1, 3, _sequential_config()))
    refs = {'owner': weakref.ref(owner), 'callback': weakref.ref(scalar),
        'graph': weakref.ref(owner.compiled.get_concrete_function().graph)}
    del owner, scalar
    gc.collect()
    released = {name: ref() is None for name, ref in refs.items()}
    target = refs['graph']()
    paths = {}
    if target is not None:
        for key in sorted(set(registry.list()) - before):
            result = _closure_paths(registry.lookup(key), target)
            if result['paths'] or result['truncated']:
                paths[key] = result
    report = {'role': 'trace-only explanatory attribution; no numerical execution',
        'released_before_inspection': released, 'registry_paths': paths,
        'new_registry_entries': len(set(registry.list()) - before),
        'nonclaims': ['No correctness, native memory eviction or cost qualification.']}
    root = Path(request.config.getoption('xmlpath')).parent
    with (root / 'block-graph-registry-attribution.json').open('x') as output:
        json.dump(report, output, indent=2)
        output.write('\n')
    assert released['callback'] and released['owner']
    assert not any(result['truncated'] for result in paths.values())
