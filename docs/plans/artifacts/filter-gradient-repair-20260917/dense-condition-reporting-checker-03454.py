"""Post-run diagnostic inspection of the proposed condition-report disposition."""
import copy
import hashlib
import json
import math
from pathlib import Path

root = Path('/home/ubuntu/workspace/BayesFilter/docs/plans/artifacts/filter-gradient-repair-20260917')
source = root / 'run-03450/dense-attempt-3-centered.json'
attribution = root / 'run-03454/dense-attempt-condition-attribution.json'
record = json.loads(source.read_text())


def proposed_check(actual, original):
    # Proposed fixture-specific reporting allowance, never a runtime decision.
    observed = []
    def walk(a, b, path=()):
        if isinstance(b, dict):
            assert a.keys() == b.keys(), path
            for key in b:
                walk(a[key], b[key], (*path, key))
        elif isinstance(b, list):
            assert len(a) == len(b), path
            for index, (x, y) in enumerate(zip(a, b, strict=True)):
                walk(x, y, (*path, index))
        elif isinstance(b, float):
            if path in [('result', 'fits', index, 'diagnostics', 'prediction_jacobian_condition_number') for index in (1, 3)] and not math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-10):
                assert isinstance(a, float) and math.isfinite(a) and math.isfinite(b)
                # This is the archived two-replicate one-factor reporting fixture only.
                # Preserve range and an explanatory error measure; do not claim
                # either scalar is an accurately resolved condition number.
                assert min(a, b) > 1e12 and abs(a-b) <= 1e-3 * max(a,b)
                index = path[2]
                left, right = actual['result']['fits'][index], original['result']['fits'][index]
                assert left['family'] == right['family'] == 'factor_1'
                assert left['diagnostics']['prediction_jacobian_rank'] == right['diagnostics']['prediction_jacobian_rank'] == 6
                observed.append({'path':list(path),'candidate':a,'original':b,
                    'status':'ill_conditioned_diagnostic_unresolved',
                    'relative_difference':abs(a-b)/abs(b),
                    'epsilon_dimension_condition':27 * math.ulp(1.) * max(a,b)})
            else:
                assert math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-10), (path,a,b)
        else:
            assert a == b, (path,a,b)
    walk(actual, original)
    assert len(observed) in (1, 2)
    return observed

accepted = [proposed_check(row['fit']['actual'], row['fit']['original']) for row in record['records']]
mutations = {}
base = record['records'][0]['fit']
for name in ('changed_rank','changed_status','changed_covariance','missing_condition','well_conditioned','nonfinite','unexplained_large_difference'):
    a=copy.deepcopy(base['actual'])
    diagnostic=a['result']['fits'][1]['diagnostics']
    if name=='changed_rank':diagnostic['prediction_jacobian_rank']-=1
    if name=='changed_status':a['result']['status']='changed'
    if name=='changed_covariance':a['result']['selected_covariance_z'][0][0]+=.01
    if name=='missing_condition':diagnostic.pop('prediction_jacobian_condition_number')
    if name=='well_conditioned':diagnostic['prediction_jacobian_condition_number']=100.
    if name=='nonfinite':diagnostic['prediction_jacobian_condition_number']=math.inf
    if name=='unexplained_large_difference':diagnostic['prediction_jacobian_condition_number']*=1.1
    try:proposed_check(a,base['original'])
    except (AssertionError,KeyError):mutations[name]='rejected'
    else:raise AssertionError(name)
report={'status':'proposal_only_pending_owner_decision','source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
 'attribution':str(attribution),'attribution_sha256':hashlib.sha256(attribution.read_bytes()).hexdigest(),
 'observations':accepted,'adverse_checks':mutations,
 'nonclaims':['No runtime or comparator change is installed.','The range limit is a fixture evidence boundary, not a general numerical safety threshold.']}
with (root/'dense-condition-reporting-proposal-03454.json').open('x') as handle:json.dump(report,handle,indent=2)
print(json.dumps({'compared_records':len(accepted),'adverse_rejections':mutations}))
