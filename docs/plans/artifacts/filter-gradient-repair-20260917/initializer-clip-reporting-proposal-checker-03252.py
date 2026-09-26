"""Diagnostic-only review of a proposed zero-boundary reporting criterion."""
import copy
import hashlib
import json
import math
from pathlib import Path

ROOT = Path('/home/ubuntu/workspace/BayesFilter/docs/plans/artifacts/filter-gradient-repair-20260917')


def review_count(actual, expected):
    a, b = actual['fit'], expected['fit']
    bounds = []
    for geometry, fit in ((actual, a), (expected, b)):
        assert fit['score_design_rank'] == geometry['rank'] + 1
        rows = (geometry['finite_sample_count'] - geometry['holdout_count']) * geometry['dimension']
        assert rows > 0 and fit['lambda0'] > 0
        condition = fit['score_design_condition_number']
        assert math.isfinite(condition) and condition >= 1
        indicator = math.ulp(1.) * max(rows, geometry['rank'] + 1) * condition
        assert indicator <= math.sqrt(math.ulp(1.))
        magnitude = math.hypot(fit['raw_lambda0'], *fit['raw_mu'])
        assert math.isfinite(magnitude) and magnitude > 0
        band = indicator * magnitude
        upper = (geometry['config']['max_condition_number'] - 1) * fit['lambda0']
        assert fit['mu_clipped_count'] == sum(mu < 0 or mu > upper for mu in fit['raw_mu'])
        bounds.append((band, upper))
    assert len(a['raw_mu']) == len(b['raw_mu'])
    changes = []
    for index, (mu_a, mu_b) in enumerate(zip(a['raw_mu'], b['raw_mu'], strict=True)):
        assert math.isfinite(mu_a) and math.isfinite(mu_b)
        assert (mu_a > bounds[0][1]) == (mu_b > bounds[1][1]), 'Upper-clipping changes remain exact'
        if (mu_a < 0) != (mu_b < 0):
            assert abs(mu_a) <= bounds[0][0] and abs(mu_b) <= bounds[1][0], 'Resolved sign change'
            changes.append({'index': index, 'actual_mu': mu_a, 'original_mu': mu_b,
                'actual_resolution_scale': bounds[0][0], 'original_resolution_scale': bounds[1][0]})
    assert changes
    return {'actual_count': a['mu_clipped_count'], 'original_count': b['mu_clipped_count'],
        'changes': changes, 'role': 'roundoff-scale reporting comparison; not certified coefficient interval'}


def compare(actual, expected, path='', reviews=None):
    reviews = [] if reviews is None else reviews
    if isinstance(expected, dict):
        omitted = {'random_stream', 'artifact_hash'}
        if expected.get('schema') == 'bayesfilter.quadratic_map_covariance.locator.v1':
            assert actual.get('jit_compile') is True
            omitted.add('jit_compile')
        assert set(actual) - omitted == set(expected) - omitted, path
        if {'fit', 'dimension', 'rank', 'finite_sample_count', 'holdout_count'} <= set(expected):
            if actual['fit']['mu_clipped_count'] != expected['fit']['mu_clipped_count']:
                proof = review_count(actual, expected)
                reviews.append({'path': path + '.fit.mu_clipped_count', **proof})
                actual = copy.deepcopy(actual)
                actual['fit']['mu_clipped_count'] = expected['fit']['mu_clipped_count']
        for key in expected.keys() - omitted:
            compare(actual[key], expected[key], path + '.' + key, reviews)
    elif isinstance(expected, list):
        assert isinstance(actual, list) and len(actual) == len(expected), path
        for index, (a,b) in enumerate(zip(actual, expected, strict=True)):
            compare(a,b,path + f'[{index}]',reviews)
    elif isinstance(expected, float):
        assert isinstance(actual, (int,float)) and math.isclose(actual,expected,rel_tol=1e-10,abs_tol=1e-10), path
    elif expected == 'scalar_value_and_score_loop':
        assert actual in (expected, 'tensorflow_scalar_row_loop'), path
    else:
        assert actual == expected, path
    return reviews


summaries = []
for number in (2775, 3252):
    path = ROOT / f'run-{number:05d}' / 'initializer-residual-candidate.json'
    data = json.loads(path.read_text())
    for record in data['records']:
        review = []
        for left,right in (('actual','expected'),('actual_events','expected_events'),('actual_calls','expected_calls')):
            compare(record[left], record[right], left, review)
        summaries.append({'run':number,'dimension':record['dimension'],'batched':record['batched'],
            'proposed_comparison_passes':True,'reviewed_diagnostic_counts':review,
            'source_result_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})

# Adverse checks prove that this cannot excuse resolved sign/rank/count changes.
record = json.loads((ROOT/'run-03252/initializer-residual-candidate.json').read_text())['records'][2]
a = record['actual']['iterations'][2]['geometry_diagnostics']
b = record['expected']['iterations'][2]['geometry_diagnostics']
adverse = []
for fault in ('resolved', 'rank', 'forged_count', 'upper', 'nonfinite'):
    left,right = copy.deepcopy(a),copy.deepcopy(b)
    if fault == 'resolved':
        left['fit']['raw_mu'][0] = 1e-11
    elif fault == 'rank':
        left['fit']['score_design_rank'] = 1
    elif fault == 'forged_count':
        left['fit']['mu_clipped_count'] = 2
    elif fault == 'upper':
        left['fit']['raw_mu'][0] = 1000.
        left['fit']['mu_clipped_count'] = 1
    else:
        left['fit']['raw_mu'][0] = float('nan')
    try:
        review_count(left,right)
    except AssertionError:
        adverse.append(fault)
    else:
        raise AssertionError(f'Proposal accepted invalid evidence: {fault}')
assert len(adverse) == 5
output = {'schema':'filter_initializer_clip_reporting_proposal_review.v1',
    'status':'PROPOSAL_ONLY_OWNER_AGREEMENT_PENDING', 'records':summaries,
    'adverse_checks_rejected':adverse, 'checker_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'nonclaims':['No runtime arithmetic or comparator has changed.',
        'The scale is the existing design-roundoff indicator times coefficient norm, not a certified error bound.',
        'All original numerical tolerances and actual decision/count/order gates remain unchanged.']}
with (ROOT/'initializer-clip-reporting-proposal-review-03252.json').open('x') as stream:
    json.dump(output,stream,indent=2,allow_nan=False)
    stream.write('\n')
print(json.dumps({'records_reviewed':len(summaries),'diagnostic_counts':sum(len(r['reviewed_diagnostic_counts']) for r in summaries),'adverse_checks':adverse}))
