"""Diagnostic reanalysis of saved means, never new HMC or estimator promotion."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.stats import norm
from bayesfilter.inference.hmc_precision import mean_precision
from bayesfilter.testing.inference_validation.storage import read_tensor, read_json, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('fresh output required')
    start = time.monotonic()
    rows, summaries, hashes = [], {}, {}
    for fit in read_json(args.inventory)['rows']:
        if fit['target'] not in ('gaussian', 'beta_binomial'):
            continue
        pipe = read_json(fit['pipeline'])
        member = next((m for m in pipe['members'] if m['candidate_id'] in pipe['selection']['candidate_ids']), {})
        row = {'target': fit['target'], 'replication': fit['replication'], 'delivered': fit['delivered'], 'arms': {}}
        for arm in ('stopped', 'fixed'):
            row['arms'][arm] = {}
            location = member.get('draws_path') if arm == 'stopped' else member.get('fixed_comparator', {}).get('draws_path')
            if not location or not Path(location).exists():
                continue
            values = read_tensor(location)
            hashes[location] = read_json(location+'.json')['sha256']
            if values.shape[0] < 4:
                continue
            old = fit['stopping_pair'].get(arm, {})
            names = [name for name in old if name.endswith(':mean')]
            for method in ('lugsail', 'autocorrelation'):
                report = mean_precision(values, method=method, jit_compile=False)
                for i, name in enumerate(names):
                    valid = bool(report['valid'][i])
                    estimate = float(report['estimate'][i]); truth = old[name]['reference']
                    se = float(report['mcse'][i]) if valid else None
                    entry = {'estimate': estimate, 'reference': truth, 'error': estimate-truth,
                             'mcse': se, 'available': valid,
                             'covered': bool(valid and abs(estimate-truth) <= norm.ppf(.975)*se),
                             'original_mcse': old[name].get('mcse')}
                    row['arms'][arm][name+':'+method] = entry
                    summaries.setdefault(fit['target']+':'+arm+':'+name+':'+method, []).append(entry)
        rows.append(row)
    assert len(rows) == 256
    aggregate = {}
    for key, entries in summaries.items():
        errors = [e['error'] for e in entries]
        pairs = [e for e in entries if e['available']]
        rms_error = math.sqrt(float(np.mean(np.square([e['error'] for e in pairs])))) if pairs else None
        rms_se = math.sqrt(float(np.mean(np.square([e['mcse'] for e in pairs])))) if pairs else None
        aggregate[key] = {'planned': 128, 'arrays': len(entries), 'available': len(pairs),
            'covered': sum(e['covered'] for e in entries),
            'empirical_error_sd': float(np.std(errors, ddof=1)) if len(errors)>1 else None,
            'signed_error_mean': float(np.mean(errors)), 'rms_error_available': rms_error,
            'rms_mcse_available': rms_se, 'se_to_error_ratio_descriptive': rms_se/rms_error if rms_error else None,
            'missing_arrays': 128-len(entries), 'unavailable_se': 128-len(pairs)}
    result = {'role':'saved-array development; fixed original stopping decisions',
              'rows':rows, 'summary':aggregate, 'tensor_sha256':hashes,
              'inventory_sha256':hashlib.sha256(args.inventory.read_bytes()).hexdigest(),
              'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'ranking_supported':False, 'new_samples':False, 'default_promoted':False,
              'elapsed_seconds':time.monotonic()-start}
    write_json(args.output, result)
    print(json.dumps({'output':str(args.output), 'summary':aggregate}))


if __name__ == '__main__':
    main()
