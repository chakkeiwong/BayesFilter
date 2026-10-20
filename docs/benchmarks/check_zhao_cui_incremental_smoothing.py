#!/usr/bin/env python3
"""Independent NumPy/SciPy diagnostic: compare incremental/end-only source runs."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from scipy.io import loadmat


def compare(left, right):
    results = [json.loads((p / 'result.json').read_text()) for p in (left, right)]
    assert all(r['status'] == 'complete' and r['source_tree_unchanged'] for r in results)
    assert results[0]['data_sha256'] == results[1]['data_sha256']
    checks = {}
    for filename in ['fit-diagnostics.mat', *sorted(p.name for p in left.glob('smoothing-t*.mat'))]:
        a, b = [loadmat(p / filename) for p in (left, right)]
        keys = ['fit_ess'] if filename == 'fit-diagnostics.mat' else ['thetas','sams','w','raw_log_weight','proposal_history','path_quantiles']
        for key in keys:
            checks[f'{filename}:{key}'] = bool(np.array_equal(a[key], b[key], equal_nan=True))
    progress = []
    for p in (left, right):
        with (p / 'fit-progress.csv').open() as stream:
            progress.append([(row[0], row[1], row[3]) for row in csv.reader(stream)])
    checks['fit_progress_ess_and_rank'] = progress[0] == progress[1]
    assert len(results[0]['smoothing']) == len(results[1]['smoothing'])
    for a, b in zip(results[0]['smoothing'], results[1]['smoothing']):
        checks[f"summary_t{int(a['time'])}"] = all(a[k] == b[k] for k in a if k != 'seconds')
    return dict(left=str(left), right=str(right), exact_equal=all(checks.values()), checks=checks,
                role='engineering_parity_only', excludes=['timings','scientific_replication','score_oracle'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('left', type=Path)
    parser.add_argument('right', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.left, args.right)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(exact_equal=result['exact_equal'], checks=len(result['checks']), output=str(args.output))))
    raise SystemExit(0 if result['exact_equal'] else 1)
