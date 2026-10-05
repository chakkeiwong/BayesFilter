"""Saved-record qualification of complete core numerical-owner costs."""

import json
from pathlib import Path

from scripts import run_filter_repair_campaign as runner
from scripts.compare_filter_repair_campaign import compare_pair
from scripts.filter_repair_core_resources import CASES, cost_rows, summarize, validate
from scripts.filter_repair_cost_provenance import require_same_physical_gpu


def test_core_resource_readback(request):
    baseline = json.loads((runner.BASELINE_ROOT/'source-manifest.json').read_text())['files']
    rows = cost_rows(runner)
    good = [row for row in rows if row['state'] == 'passed']
    for run in good:
        result = json.loads(Path(run['result']).read_text())
        validate(run, result, runner, baseline)
    require_same_physical_gpu([row['gpu_uuid'] for row in good])
    comparisons = summarize(rows)
    assert len(comparisons) == 2*len(CASES)
    for entry in comparisons:
        for pair in entry['pairs']:
            a, b = [json.loads((Path(pair[name])/'result.json').read_text())
                    for name in ('before_run', 'after_run')]
            pair['maximum_absolute_error'] = compare_pair(a, b)
    result = {'schema': 'filter_repair.core_resources_readback.v1',
        'comparisons': comparisons,
        'failed_workers_preserved': [{'result': row['result'], 'state': row['state']}
                                     for row in rows if row['state'] != 'passed'],
        'nonclaims': ['Fixed finite fixtures and measured numerical-owner boundaries only.',
                     'No arbitrary-shape capacity, universal speed or canonical LEDH admission.']}
    (Path(request.config.getoption('xmlpath')).parent/'core-resource-readback.json').write_text(
        json.dumps(result, indent=2)+'\n')
