"""Changed-data XLA inputs and actual enclosing resource ownership."""

import gc
import json
import re
import sys
import weakref
from pathlib import Path

import pytest
import tensorflow as tf

from bayesfilter.inference import sequential_lifecycle_tf as native
from bayesfilter.inference import sequential_map_covariance as current
from bayesfilter.inference.sequential_refinement_tf import refinement_program
from bayesfilter.inference.sequential_terminal_tf import terminal_program
from tests.filter_repair_frozen_checkpoint import FrozenCheckpoint
from tests.test_filter_repair_fixed_stability import _compare
from tests.test_filter_repair_lifecycle_actual import _excerpts, _materialize
from tests.test_filter_repair_structured_memory import _inputs

D = tf.float64


@pytest.mark.parametrize('dimension', [3, 5])
def test_actual_lifecycle_runtime_inputs_and_resource_lifetime(dimension, request):
    checkpoint = FrozenCheckpoint('cfbc32d2', 'lifecycle_runtime')
    original, finish = _excerpts(checkpoint)
    scalar, batched, _ = _inputs(dimension, 4)
    cfg = current.SequentialMapCovarianceConfig(locator_policy='center_first',
        refinement_geometry_policy='factor_correlation', structured_max_factors=1 if dimension == 3 else 2,
        structured_holdout_score_relative_rmse=.001,
        max_attempts=2, search_sample_count=4, terminal_sample_count=24,
        max_exact_evaluations=256, initial_radius=.25, terminal_score_max_abs=1e-5,
        record_refinement_movement_diagnostics=True)
    refine = refinement_program(scalar, batched, dimension, cfg, cfg.search_sample_count)
    terminal = terminal_program(scalar, batched, dimension, cfg)
    program = native.lifecycle_program(refine, terminal, dimension, cfg, cfg.search_sample_count)
    observations, hlos = [], []
    directory = Path(request.config.getoption('xmlpath')).parent
    for index, (multiplier, scale_multiplier, evaluations) in enumerate(((1., 1., 1), (.8, 1.1, 7))):
        center = tf.linspace(tf.constant(.002, D), tf.constant(.004, D), dimension) * multiplier
        value, score = scalar(center)
        scale = tf.linspace(tf.constant(.8, D), tf.constant(1.2, D), dimension) * scale_multiplier
        arguments = (center, value, score, scale, tf.constant(evaluations, tf.int64))
        before_progress, after_progress = [], []
        before = original(scalar, batched, center, float(value), score, scale, cfg,
            evaluations, before_progress.append).payload()
        result = program(*arguments)
        after = _materialize(result, center, value, score, scale, cfg, finish, after_progress).payload()
        _compare(after, before)
        _compare(after_progress, before_progress)
        hlo = program.experimental_get_compiler_ir(*arguments)(stage='hlo')
        hlos.append(hlo)
        with (directory / f'lifecycle-runtime-{dimension}-{index}.hlo').open('x') as handle:
            handle.write(hlo)
        observations.append({'before': before, 'after': after,
            'progress': after_progress, 'input_evaluations': evaluations})
    concrete = program.get_concrete_function()
    graph = concrete.graph
    definition = graph.as_graph_def()
    nodes = [*definition.node, *(node for function in definition.library.function for node in function.node_def)]
    assert any(node.op in ('While', 'StatelessWhile') for node in nodes)
    assert not any(node.op in ('PyFunc', 'EagerPyFunc') for node in nodes)
    arity = 5 + len(concrete.captured_inputs)
    with (directory / f'lifecycle-runtime-observations-{dimension}.json').open('x') as handle:
        json.dump({'observations': observations, 'expected_operands': arity,
            'frozen_source_sha256': checkpoint.hashes()}, handle, indent=2)
        handle.write('\n')
    for hlo in hlos:
        entry = hlo[hlo.rfind('\nENTRY '):]
        assert sorted(map(int, re.findall(r'\bparameter\((\d+)\)', entry))) == list(range(arity))

    def normalize(hlo):
        # Grappler's generated zero metadata can change across IR inspection.
        return re.sub(r'op_name="(zeros(?:_\d+)?)(?:/_\d+)+"(?= source_file="dummy_file_name" source_line=10)',
            r'op_name="\1/__generated"', hlo)

    assert normalize(hlos[0]) == normalize(hlos[1])
    assert program.experimental_get_tracing_count() == 1
    resources = [weakref.ref(variable) for variable in graph.variables]
    assert resources, 'This fixture must enclose actual factor validity resources'
    graph_ref = weakref.ref(graph)
    cleared = set()
    for name, module in tuple(sys.modules.items()):
        if name.startswith('bayesfilter.inference.'):
            for value in vars(module).values():
                if hasattr(value, 'cache_clear') and id(value) not in cleared:
                    value.cache_clear()
                    cleared.add(id(value))
    del value, program, refine, terminal
    gc.collect()
    repeated = concrete(*arguments)
    _compare(current._json_ready(repeated), current._json_ready(result))
    del graph, concrete
    gc.collect()
    living = []
    for resource in resources:
        variable = resource()
        if variable is None:
            continue
        owners = []
        for owner in gc.get_referrers(variable):
            description = {'type': type(owner).__name__}
            if isinstance(owner, dict):
                description['keys'] = [str(key) for key in owner if isinstance(key, (str, int))][:12]
                description['owners'] = [{'type': type(parent).__name__,
                    'name': getattr(parent, 'name', None)} for parent in gc.get_referrers(owner)
                    if type(parent).__name__.endswith('Graph')]
                description['attribute_owners'] = [
                    {'attributes': [str(key) for key, item in parent.items() if item is owner],
                     'objects': [{'type': type(ancestor).__name__,
                         'name': getattr(ancestor, 'name', None)}
                         for ancestor in gc.get_referrers(parent)
                         if type(ancestor).__name__.endswith('Graph')]}
                    for parent in gc.get_referrers(owner) if isinstance(parent, dict)]
            owners.append(description)
        living.append({'name': variable.name, 'owners': owners})
    del variable, resource
    with (directory / f'lifecycle-resource-release-{dimension}.json').open('x') as handle:
        json.dump({'graph_released': graph_ref() is None, 'resource_count': len(resources),
            'living_resources': living}, handle, indent=2)
        handle.write('\n')
    assert graph_ref() is None
    assert all(ref() is None for ref in resources)
    with (directory / f'lifecycle-runtime-{dimension}.json').open('x') as handle:
        json.dump({'observations': observations, 'runtime_operands': arity,
            'graph_nodes': len(nodes), 'traces': 1, 'resource_count': len(resources),
            'resources_released': True, 'execution_after_python_cache_eviction': True,
            'frozen_source_sha256': checkpoint.hashes(),
            'nonclaims': ['Python factory eviction is not native XLA executable eviction.']}, handle, indent=2)
        handle.write('\n')
