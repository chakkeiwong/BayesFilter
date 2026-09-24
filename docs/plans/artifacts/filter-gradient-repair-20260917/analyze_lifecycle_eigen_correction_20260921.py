"""Diagnostic attribution of original/intermediate/repaired lifecycle records."""

import hashlib
import json
import math
import sys
from pathlib import Path

root = Path(sys.argv[1])


def differences(actual, expected, path='result'):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys(), path
        return [row for key in expected for row in differences(actual[key], expected[key], path + '.' + key)]
    if isinstance(expected, list):
        return [row for index, (a, b) in enumerate(zip(actual, expected, strict=True))
            for row in differences(a, b, path + f'[{index}]')]
    if isinstance(expected, (str, bool, int)) or expected is None:
        agrees = actual == expected
    else:
        agrees = actual == expected or (math.isnan(actual) and math.isnan(expected)) or (
            math.isfinite(actual) and math.isfinite(expected)
            and abs(actual - expected) <= 1e-10 + 1e-10 * abs(expected))
    return [] if agrees else [{'path': path, 'intermediate': expected, 'repaired': actual}]


rows = []
for old, new in zip(range(1777, 1785), range(1810, 1818), strict=True):
    before_dir, after_dir = root / f'run-{old:05d}', root / f'run-{new:05d}'
    before_manifest = json.loads((before_dir / 'run.json').read_text())
    after_manifest = json.loads((after_dir / 'run.json').read_text())
    assert before_manifest['state'] == after_manifest['state'] == 'passed'
    assert before_manifest['device'] == after_manifest['device']
    for path in sorted(after_dir.glob('lifecycle-actual-*.json')):
        before_path = before_dir / path.name
        before, after = json.loads(before_path.read_text()), json.loads(path.read_text())
        assert before['checkpoint'] == 'cfbc32d2' and after['checkpoint'] == '3582b4ac'
        assert before['before_calls'] == before['after_calls'] == after['before_calls'] == after['after_calls']
        assert len(before['after_points']) == len(after['after_points'])
        rows.append({'case': after['case'], 'device': after_manifest['device'],
            'intermediate_run': old, 'original_comparison_run': new,
            'artifact_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (before_path, path)},
            'endpoint_differences': differences(after['after'], before['after']),
            'progress_differences': differences(after['after_progress'], before['after_progress']),
            'target_row_differences': differences(after['after_points'], before['after_points'], 'target_rows'),
            'target_rows': after['after_calls']})
assert len(rows) == 16
report = {'schema': 'filter_repair_lifecycle_eigen_correction.v1', 'comparisons': rows,
    'original_precision_authority': '3582b4ac', 'intermediate_with_demonstrated_defect': 'cfbc32d2',
    'tolerances': {'atol': 1e-10, 'rtol': 1e-10},
    'interpretation': 'All new artifacts pass complete original-record and target-order checks. Intermediate differences are retained and attributed to correction of the exposed eigensystem residuals.',
    'nonclaims': ['Public outer integration, resource/cost refresh, factor mode gap and terminal admission remain separate.',
        'Execution-mode metadata absent from the original schema is handled explicitly by the original-record test; numerical fields are not excluded.'],
    'analysis_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
with (root / 'lifecycle-eigen-correction-01818.json').open('x') as handle:
    json.dump(report, handle, indent=2)
    handle.write('\n')
for row in rows:
    print(row['device'], row['case'], 'endpoint_fields', len(row['endpoint_differences']),
        'progress_fields', len(row['progress_differences']), 'target_rows', row['target_rows'],
        'changed_target_fields', len(row['target_row_differences']))
