"""Post-run diagnostic reporting only; never selects a runtime or a tolerance."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


def percent(delta):
    try:
        return 100. * math.expm1(delta)
    except OverflowError:
        return None


def distribution(values):
    if not values:
        return dict(n=0, median=None, minimum=None, maximum=None)
    return dict(n=len(values), median=statistics.median(values), minimum=min(values), maximum=max(values))


def summarize(source_rows, tf_rows, variants):
    source = {(r['case'],r['variant']):r for r in source_rows}
    port = {(r['case'],r['solver'],r['variant']):r for r in tf_rows}
    assert len(source)==len(source_rows) and len(port)==len(tf_rows)
    assert all((r['case'],r['variant']) in source for r in tf_rows)
    cases=[]
    for cid in sorted({r['case'] for r in source_rows}):
        reference=source[cid,'zero']
        assert reference['valid'] and reference['zero_control_exact']
        for solver in sorted({r['solver'] for r in tf_rows}):
            base=port[cid,solver,'zero']
            delta=base['terminal_logZ']-reference['terminal_logZ'] if base['valid'] else None
            row=dict(case=cid,forcing=reference['forcing'],seed=reference['seed'],solver=solver,
                source_base_logZ=reference['terminal_logZ'],source_base_stages=reference['stage_count'],
                tf_base_valid=base['valid'],tf_base_error=base['error'],tf_base_stages=base['stage_count'],
                tf_base_logZ=base['terminal_logZ'],baseline_delta_logZ=delta,
                baseline_likelihood_percent=percent(delta) if delta is not None else None,arms=[])
            for epsilon in (1e-10,1e-8,1e-6):
                for arm in (('normal','uniform','combined') if epsilon==1e-8 else ('combined',)):
                    trials=[v for v in variants if v['epsilon']==epsilon and v['arm']==arm]
                    pairs=[(source[cid,v['id']],port[cid,solver,v['id']]) for v in trials]
                    source_delta=[a['terminal_logZ']-reference['terminal_logZ'] for a,t in pairs if a['valid']]
                    tf_delta=[t['terminal_logZ']-base['terminal_logZ'] for a,t in pairs if t['valid'] and base['valid']]
                    sa=distribution([abs(x) for x in source_delta])
                    st=distribution([abs(x) for x in tf_delta])
                    additional=sum(a['valid'] and not t['valid'] for a,t in pairs)
                    complete=sa['n']==len(trials) and st['n']==len(trials)
                    within=abs(delta)<=sa['median'] if delta is not None and sa['n']==len(trials) else None
                    row['arms'].append(dict(arm=arm,epsilon=epsilon,trials=len(trials),
                        source_failures=sum(not a['valid'] for a,t in pairs),
                        tf_failures=sum(not t['valid'] for a,t in pairs),additional_tf_failures=additional,
                        source_abs_delta_logZ=sa,tf_abs_delta_logZ=st,
                        source_signed_likelihood_percent=distribution([percent(x) for x in source_delta if percent(x) is not None]),
                        source_median_absolute_likelihood_percent=statistics.median(abs(percent(x)) for x in source_delta) if source_delta and all(percent(x) is not None for x in source_delta) else None,
                        tf_signed_likelihood_percent=distribution([percent(x) for x in tf_delta if percent(x) is not None]),
                        paired_abs_delta_logZ=distribution([abs(t['terminal_logZ']-a['terminal_logZ']) for a,t in pairs if a['valid'] and t['valid']]),
                        source_stopping_changes=sum(a['valid'] and a['stage_count']!=reference['stage_count'] for a,t in pairs),
                        tf_stopping_changes=sum(t['valid'] and t['stage_count']!=base['stage_count'] for a,t in pairs),
                        source_any_ancestry_change=sum(any((s.get('ancestry_changed') or 0)>0 for s in a['stages']) for a,t in pairs),
                        tf_any_ancestry_change=sum(any((s.get('ancestry_changed') or 0)>0 for s in t['stages']) for a,t in pairs),
                        source_fraction_exceeding_baseline_port_difference=sum(abs(x)>=abs(delta) for x in source_delta)/len(source_delta) if delta is not None and source_delta else None,
                        baseline_difference_within_source_median=within,
                        tf_to_source_median_sensitivity=st['median']/sa['median'] if complete and sa['median']>0 else None,
                        disposition='ADDITIONAL_TF_FAILURES' if additional or not base['valid'] else
                            'INCOMPLETE_PAIRED_VALIDITY' if not complete else
                            'BASELINE_GAP_WITHIN_TYPICAL_SOURCE_SENSITIVITY' if within else 'BASELINE_GAP_EXCEEDS_TYPICAL_SOURCE_SENSITIVITY'))
            cases.append(row)
    total_additional=sum(source[r['case'],r['variant']]['valid'] and not r['valid'] for r in tf_rows)
    return dict(scope='SIX_CASE_AUTONOMOUS_RANDOM_INPUT_SENSITIVITY',cases=cases,
        source_runs=len(source_rows),source_valid=sum(r['valid'] for r in source_rows),
        tf_runs=len(tf_rows),tf_valid=sum(r['valid'] for r in tf_rows),additional_tf_failures=total_additional,
        current_port_acceptable=False if total_additional else None,
        verdict='CURRENT_PORT_FAILS_NO_ADDITIONAL_FAILURE_REQUIREMENT' if total_additional else 'SENSITIVITY_MEASURED_FINAL_NONINFERIORITY_NOT_ESTABLISHED',
        inference='Four antithetic direction pairs per arm/case. Descriptive sensitivity; no independent-seed population ranking or non-inferiority claim.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source','tf','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    assert not args.output.exists()
    source_path=args.source/'source-rows.json'
    tf_path=args.tf/'tf-rows.json'
    fixture_path=args.source/'fixture.json'
    source=json.loads(source_path.read_text())
    port=json.loads(tf_path.read_text())
    fixture=json.loads(fixture_path.read_text())
    assert len(source)==174 and len(port)==348
    for directory,path in ((args.source,source_path),(args.tf,tf_path)):
        manifest=json.loads((directory/'manifest.json').read_text())
        summary=json.loads((directory/'summary.json').read_text())
        assert manifest['status']=='COMPLETE' and summary['status']=='PASS'
        assert hashlib.sha256(path.read_bytes()).hexdigest()==summary['rows_sha256']
    result=summarize(source,port,fixture['variants'])
    result['input_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (source_path,tf_path,fixture_path)}
    args.output.mkdir()
    (args.output/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('cases','input_sha256')},indent=2))


if __name__=='__main__':
    main()
