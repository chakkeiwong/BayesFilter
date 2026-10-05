"""Diagnostic compiler-input localization; no abbreviated runtime candidate."""

import json
import re
from pathlib import Path

import tensorflow as tf

from bayesfilter.inference.sequential_map_covariance import (
    SequentialMapCovarianceConfig,
)
from bayesfilter.inference.sequential_refinement_tf import refinement_program
from bayesfilter.inference.sequential_terminal_tf import terminal_program
from tests.test_filter_repair_structured_memory import _inputs

D = tf.float64


def test_lifecycle_dependency_compiler_operands(request):
    scalar, batched, _ = _inputs(3, 4)
    cfg = SequentialMapCovarianceConfig(refinement_geometry_policy='factor_correlation',
        structured_max_factors=1, max_attempts=2, search_sample_count=4, terminal_sample_count=24)
    center, scale = tf.constant([.002, .003, .004], D), tf.constant([.8, 1., 1.2], D)
    value, score = scalar(center)
    terminal = terminal_program(scalar, batched, 3, cfg)
    refine = refinement_program(scalar, batched, 3, cfg, 4)
    directory = Path(request.config.getoption('xmlpath')).parent
    records = {}
    for name, program, args in (
            ('terminal', terminal, (center, score, scale, tf.constant(.25, D), tf.constant([2026, 100718]))),
            ('refinement', refine, (center, value, score, scale, tf.constant(.25, D), tf.constant(0), tf.constant(0)))):
        concrete = program.get_concrete_function()
        hlo = program.experimental_get_compiler_ir(*args)(stage='hlo')
        entry = hlo[hlo.rfind('\nENTRY '):]
        parameters = [line.strip() for line in entry.splitlines() if re.search(r'\bparameter\(\d+\)', line)]
        records[name] = {'expected_operands': len(args) + len(concrete.captured_inputs),
            'observed_operands': len(parameters), 'parameters': parameters,
            'graph_inputs': [{'name': v.name, 'shape': v.shape.as_list(), 'dtype': v.dtype.name}
                for v in concrete.inputs], 'traces': program.experimental_get_tracing_count()}
        with (directory / f'lifecycle-{name}.hlo').open('x') as handle:
            handle.write(hlo)
        with (directory / f'lifecycle-{name}.graph.pb').open('xb') as handle:
            handle.write(concrete.graph.as_graph_def().SerializeToString())
    with (directory / 'lifecycle-specialization.json').open('x') as handle:
        json.dump(records, handle, indent=2)
        handle.write('\n')
    print(json.dumps(records))
