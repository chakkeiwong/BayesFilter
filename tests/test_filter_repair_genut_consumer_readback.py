"""Diagnostic source applicability and live-wiring evidence verification."""

import json

from tests.test_filter_repair_genut_current_consumers import (
    DEPENDENCIES,
    RAW,
    ROOT,
    digest,
)
from tests.test_filter_repair_geometry_control import save


def test_current_genut_consumer_evidence(request):
    rows = []
    for number, device in ((4936, 'CPU'), (4938, 'GPU')):
        directory = RAW/f'run-{number:05d}'
        manifest = json.loads((directory/'run.json').read_text())
        assert manifest['state'] == 'passed' and manifest['device'] == device
        value = json.loads((directory/'genut-consumer-value.json').read_text())
        score = json.loads((directory/'genut-consumer-analytical.json').read_text())
        expected_sources = {p: digest(ROOT/p) for p in DEPENDENCIES}
        assert value['sources'] == score['sources'] == expected_sources
        assert all(manifest['source_sha256'][p] == h for p, h in expected_sources.items())
        assert len(value['records']) == 3
        for record in value['records']:
            expected = {'disabled': [0, 0], 'reduced': [1, 0], 'trust': [0, 1]}[record['branch']]
            assert record['runtime_counts'] == expected
            assert record['maximum_output_error'] == 0.
            assert record['ordinary']['program_valid']
        assert score['runtime_batched_jvp_calls'] == 2 and score['reduced_primal_blocked']
        assert score['maximum_output_error'] == 0. and score['finite']
        assert score['jit_compile'] and score['trace_count'] == 1
        provenance = next(json.loads(line) for line in (directory/'process.log').read_text().splitlines()
            if line.startswith('{"tensorflow_version":'))
        assert provenance['gpu_memory_policy']['all_physical_devices_memory_growth']
        assert provenance['gpu_memory_policy']['configured_before_logical_device_initialization']
        if device == 'GPU':
            assert provenance['gpu_memory_policy']['physical_devices']
        rows.append({'device': device, 'run': directory.name,
            'manifest_sha256': digest(directory/'run.json'),
            'value_sha256': digest(directory/'genut-consumer-value.json'),
            'analytical_sha256': digest(directory/'genut-consumer-analytical.json'),
            'memory_policy': provenance['gpu_memory_policy']})
    source = 'bayesfilter/highdim/dual_cap_genut_primal_tf.py'
    applicability = []
    for number in (4248, 4253, 4259, 4260):
        manifest = json.loads((RAW/f'run-{number:05d}/run.json').read_text())
        assert manifest['source_sha256'][source] == digest(ROOT/source)
        applicability.append({'run': number, 'runtime_sha256': manifest['source_sha256'][source],
            'same_runtime_bytes': True})
    save(request, 'genut-consumer-terminal.json', {'schema': 'filter_genut_current_consumer_review.v1',
        'evidence': rows, 'saved_precision_runtime_applicability': applicability,
        'sources': {p: digest(ROOT/p) for p in DEPENDENCIES},
        'remaining': ['Optional reduced-primal precision and cap-report findings remain unresolved.',
            'Its diagnostic reverse gradients do not validate or invalidate the distinct analytical correction.',
            'Other public caller/dtype/control scopes and canonical conformance are not covered.']})
