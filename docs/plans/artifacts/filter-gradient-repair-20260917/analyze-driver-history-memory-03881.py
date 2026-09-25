"""Independent analysis of matched history-loader records and provenance."""

import hashlib
import json
import statistics
import sys
from pathlib import Path
from xml.etree import ElementTree as ET


def analyze(root):
    rows = []
    closure = None
    identity = None
    for number in range(3876, 3882):
        directory = root / f'run-{number:05d}'
        run = json.loads((directory / 'run.json').read_text())
        value = json.loads((directory / 'driver-history-memory.json').read_text())
        cases = list(ET.parse(directory / 'junit.xml').getroot().iter('testcase'))
        assert len(cases) == 1 and not list(cases[0])
        assert run['state'] == 'passed' and run['device'] == 'CPU' and run['exit_code'] == 0
        assert run['environment']['CUDA_VISIBLE_DEVICES'] == '-1'
        if closure is None:
            closure = run['source_sha256']
        assert closure == run['source_sha256']
        current_identity = {key: value[key] for key in ('history_sha256', 'run_count',
            'next_run_number', 'attempt_key', 'exact_source_attempts', 'charged_seconds')}
        if identity is None:
            identity = current_identity
        assert identity == current_identity
        assert value['run_count'] == 3870 and value['next_run_number'] == 3871
        assert value['exact_source_attempts'] == 1
        assert value['rss_sample_count'] > 2 and value['seconds'] > 0.
        assert value['sampled_peak_rss_bytes'] >= value['rss_retained_bytes'] > value['rss_before_bytes']
        rows.append({'run': number, 'repeat': run['key'][6], **value,
            'result_sha256': hashlib.sha256((directory / 'driver-history-memory.json').read_bytes()).hexdigest()})
    medians = {}
    for arm in ('prior', 'streamed'):
        selected = [row for row in rows if row['arm'] == arm]
        assert sorted(row['repeat'] for row in selected) == [0, 1, 2]
        assert len({row['driver_source_sha256'] for row in selected}) == 1
        medians[arm] = {key: statistics.median(row[key] for row in selected) for key in
            ('seconds', 'rss_before_bytes', 'rss_retained_bytes', 'sampled_peak_rss_bytes',
             'lifetime_peak_before_bytes', 'peak_rss_bytes')}
    return {'schema': 'filter_repair_driver_history_analysis.v1', 'verified': True,
        'shared_identity': identity, 'rows': rows, 'medians': medians,
        'retained_rss_reduction_fraction': 1. - medians['streamed']['rss_retained_bytes'] / medians['prior']['rss_retained_bytes'],
        'wall_time_ratio': medians['streamed']['seconds'] / medians['prior']['seconds'],
        'preserved_initial_cohort': [3873, 3874, 3875],
        'scope': 'Host history bookkeeping; no target compiler, tensor memory, or numerical algorithm conclusion.'}


if __name__ == '__main__':
    print(json.dumps(analyze(Path(sys.argv[1])), indent=2, allow_nan=False))
