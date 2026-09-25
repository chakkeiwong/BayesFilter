"""Diagnostic post-run attribution, never an executable benchmark kernel.

Compare the declared graph-reference and XLA geometry-fit implementations.
Their solver graphs differ; this is not an identical-graph JIT flag ablation.
"""

import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(actual, expected, path='$', *, exact=False, differences=None):
    if differences is None:
        differences = []
    if isinstance(expected, dict):
        assert isinstance(actual, dict) and actual.keys() == expected.keys(), path
        for key in expected:
            # Same predeclared exception as geometry-fit cost qualification:
            # independently content-derived hashes are retained per arm.
            if path == '$.diagnostics' and key == 'artifact_hash':
                continue
            compare(actual[key], expected[key], f'{path}.{key}', exact=exact, differences=differences)
    elif isinstance(expected, list):
        assert isinstance(actual, list) and len(actual) == len(expected), path
        for index, (left, right) in enumerate(zip(actual, expected, strict=True)):
            compare(left, right, f'{path}[{index}]', exact=exact, differences=differences)
    elif isinstance(expected, (bool, int, str)) or expected is None:
        assert type(actual) is type(expected) and actual == expected, (path, actual, expected)
    elif isinstance(expected, float):
        assert isinstance(actual, float), (path, type(actual))
        if math.isnan(expected):
            assert math.isnan(actual), path
        elif math.isinf(expected) or exact:
            assert actual == expected, (path, actual, expected)
        else:
            error = abs(actual - expected)
            assert error <= 1e-10 + 1e-10 * abs(expected), (path, error)
            differences.append({'path': path, 'absolute_error': error})
    else:
        raise TypeError((path, type(expected)))
    return differences


def load(number):
    directory = ROOT / f'run-{number:05d}'
    manifest = json.loads((directory / 'run.json').read_text())
    data = json.loads((directory / 'gap-fit-memory.json').read_text())
    environment = [json.loads(line) for line in (directory / 'process.log').read_text().splitlines()
        if line.startswith('{"tensorflow_version"')]
    assert len(environment) == 1
    assert manifest['state'] == 'passed' and manifest['test_evidence']['passed']
    assert manifest['test_evidence']['skipped'] == 0
    assert data['python_graph_released'] and data['graph']['traces'] == 1
    return manifest, data, environment[0], {name: digest(directory / name)
        for name in ('run.json', 'gap-fit-memory.json', 'process.log')}


def pair(graph_number, xla_number):
    graph_manifest, graph, graph_env, graph_hashes = load(graph_number)
    xla_manifest, xla, xla_env, xla_hashes = load(xla_number)
    assert graph_manifest['source_sha256'] == xla_manifest['source_sha256']
    assert graph_manifest['device'] == xla_manifest['device']
    assert graph_manifest['environment'] == xla_manifest['environment']
    assert graph_env == xla_env
    assert not graph['jit_compile'] and xla['jit_compile']
    compare(xla['input_records'], graph['input_records'], exact=True)
    differences = compare(xla['result'], graph['result'])
    gpu = graph_manifest['device'] == 'GPU'
    uuid = None
    if gpu:
        uuid = graph_manifest['gpu_uuid']
        assert uuid and uuid == xla_manifest['gpu_uuid'] == graph_env['cuda_visible_devices']
        assert graph_manifest['gpu_performance_preflight_uncontended']
        assert xla_manifest['gpu_performance_preflight_uncontended']
        assert graph_env['gpu_memory_policy']['all_physical_devices_memory_growth']
        assert graph_env['gpu_memory_policy']['configured_before_logical_device_initialization']
    else:
        assert graph_env['cuda_visible_devices'] == '-1'
    rss = {arm: {stage: record['rollup']['Rss'] for stage, record in data['stages'].items()}
        for arm, data in (('graph', graph), ('xla', xla))}
    return {'runs': [graph_number, xla_number], 'device': graph_manifest['device'], 'gpu_uuid': uuid,
        'environment': graph_env, 'source_sha256': graph_manifest['source_sha256'],
        'artifact_sha256': {'graph': graph_hashes, 'xla': xla_hashes},
        'inputs_exactly_equal': True, 'complete_results_equal_at_unchanged_tolerance': True,
        'atol': 1e-10, 'rtol': 1e-10, 'discrete_fields_exact': True,
        'numeric_fields_compared': len(differences),
        'largest_difference': max(differences, key=lambda row: row['absolute_error']),
        'arm_specific_payload_hashes': [graph['result']['diagnostics']['artifact_hash'],
            xla['result']['diagnostics']['artifact_hash']],
        'smaps_rss_bytes': rss,
        'xla_minus_graph_first_execution_rss_bytes': rss['xla']['first_execution'] - rss['graph']['first_execution'],
        'xla_minus_graph_increment_since_prepared_bytes': (
            rss['xla']['first_execution'] - rss['xla']['prepared']
            - rss['graph']['first_execution'] + rss['graph']['prepared']),
        'timing_seconds': {'graph': graph['timing_seconds'], 'xla': xla['timing_seconds']},
        'allocator_bytes': {arm: {stage: row['gpu'] for stage, row in data['stages'].items()}
            for arm, data in (('graph', graph), ('xla', xla))},
        'graphs': {'graph': graph['graph'], 'xla': xla['graph']}}


def main():
    report = {'schema': 'filter_repair_gap_memory_diagnostic_v1',
        'role': 'explanatory_diagnostic_not_terminal_performance',
        'analyzer_sha256': digest(Path(__file__)), 'pairs': [pair(2629, 2630), pair(2631, 2632)],
        'limitations': ['One process per arm and one D3 extent; no timing ranking or general leak claim.',
            'Python graph release does not identify native executable or allocator retention.',
            'Graph reference uses its declared SVD; XLA uses its compatible solver graph.',
            'ru_maxrss exceeds /proc high water already at preparation; attribute stages with smaps RSS.',
            'Idle preflight does not guarantee exclusive GPU use throughout execution.']}
    output = ROOT / 'gap-memory-attribution-02632.json'
    with output.open('x') as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps({'artifact': str(output), 'pairs': [{k: row[k] for k in
        ('runs', 'device', 'gpu_uuid', 'inputs_exactly_equal', 'complete_results_equal_at_unchanged_tolerance',
         'largest_difference', 'xla_minus_graph_first_execution_rss_bytes')} for row in report['pairs']]}))


if __name__ == '__main__':
    main()
