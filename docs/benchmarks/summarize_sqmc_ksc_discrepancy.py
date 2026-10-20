"""Standard-library post-run diagnostics; no runtime selection or admission."""
from __future__ import annotations
import csv
import json
import math
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'docs/plans/artifacts/sqmc-ksc-discrepancy-20260929/attempt-01'
ROUTES=('iid_dual_cap','previous_inverse_cdf','repaired_permutation','repaired_permutation_ablation')
LABELS=dict(zip(ROUTES,('IID','Inverse CDF','Permutation .98','Permutation .97')))


def read(p):return json.loads(p.read_text())


def latest(unit):
    paths=sorted(BASE.glob(unit+'-attempt-*/result.json'))
    if not paths:raise RuntimeError('Missing '+unit)
    value=read(paths[-1])
    if value['status']!='complete':raise RuntimeError('Incomplete '+unit)
    return value


def stats(xs):
    avg=statistics.mean(xs);sd=statistics.stdev(xs) if len(xs)>1 else None
    se=sd/math.sqrt(len(xs)) if sd is not None else None
    return dict(n=len(xs),mean=avg,sd=sd,se=se,exploratory_t95=[avg-2.364624251*se,avg+2.364624251*se] if len(xs)==8 else None)


def dump(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')


def mechanism_summary(row):
    trace=row['mechanism']['direction_traces']
    base=trace[0]
    before=base['before_moments'];after=base['after_moments']
    value_shifts=[b-a for a,b in zip(base['source_log_prediction'],base['reset_log_prediction'])]
    score_shifts=[[b-a for a,b in zip(t['source_predictive_score'],t['reset_predictive_score'])] for t in trace]
    integration_value=[a-b for a,b in zip(base['actual_log_increment'][1:],base['reset_log_prediction'])]
    integration_score=[[a-b for a,b in zip(t['actual_score_increment'][1:],t['reset_predictive_score'])] for t in trace]
    def magnitude(xs):
        return dict(mean_absolute=statistics.mean(map(abs,xs)),max_absolute=max(map(abs,xs)),
                    rms=math.sqrt(statistics.mean(x*x for x in xs)))
    return dict(data_seed=row['data_seed'],route=row['route'],n=row['n'],
                maximum_mean_change=max(abs(b[0]-a[0]) for a,b in zip(before,after)),
                maximum_relative_variance_change=max(abs(b[1]-a[1])/a[1] for a,b in zip(before,after)),
                before_mean_skew=statistics.mean(x[2] for x in before),
                before_mean_kurtosis=statistics.mean(x[3] for x in before),
                after_mean_skew=statistics.mean(x[2] for x in after),
                after_mean_kurtosis=statistics.mean(x[3] for x in after),
                within_group_variance_ratio_mean=statistics.mean(base['within_group_variance_ratio']),
                within_group_variance_ratio_max=max(base['within_group_variance_ratio']),
                reset_next_log_prediction_change=magnitude(value_shifts),
                reset_next_predictive_score_change=[magnitude(x) for x in score_shifts],
                subsequent_integration_log_error=magnitude(integration_value),
                subsequent_integration_score_error=[magnitude(x) for x in integration_score],
                parity_max_error=row['mechanism']['parity_max_error'],
                interpretation='Local effects along realized trajectory; not an additive decomposition of total oracle error.')


def main():
    out=BASE/'analysis-01';out.mkdir(exist_ok=False)
    prepared=read(BASE/'prepared.json');ledger=read(BASE.parent/'budget.json')
    refs=latest('reference')['rows']
    fd=[];rows=[]
    for route in ROUTES:
        fd.extend(latest('derivative__'+route)['fd_checks'])
        for phase in ('replay','derivative','particles','numerics','mechanism'):
            for row in latest(phase+'__'+route)['rows']:
                rows.append(dict(row,phase=phase))
    summary=[]
    for case in prepared['cases']:
        for route in ROUTES:
            groups={}
            for n in (1008,2016,4032):
                group=[r for r in rows if r['phase']=='particles' and r['data_seed']==case['data_seed'] and r['route']==route and r['n']==n]
                if len(group)!=8:raise RuntimeError('Missing replication cells')
                valid=[r for r in group if r['valid']]
                item=dict(data_seed=case['data_seed'],route=route,n=n,valid_count=len(valid),total_count=len(group))
                if valid:
                    item.update(value=stats([r['value'] for r in valid]),value_error=stats([r['value_error'] for r in valid]),
                                absolute_value_error=stats([abs(r['value_error']) for r in valid]),
                                score=[stats([r['score'][j] for r in valid]) for j in range(2)],
                                score_error=[stats([r['score_error'][j] for r in valid]) for j in range(2)],
                                absolute_score_error=[stats([abs(r['score_error'][j]) for r in valid]) for j in range(2)],
                                mean_norm_error=stats([r['score_l2_error'] for r in valid]))
                    item['norm_mean_error']=math.sqrt(sum(a['mean']**2 for a in item['score_error']))
                    count=len(valid)
                    covariance=[[sum((r['score_error'][i]-item['score_error'][i]['mean'])*
                                     (r['score_error'][j]-item['score_error'][j]['mean']) for r in valid)/(count-1)
                                 for j in range(2)] for i in range(2)]
                    item['score_error_sample_covariance']=covariance
                    item['mean_score_error_estimated_covariance']=[[x/count for x in row] for row in covariance]
                    item['empirical_squared_error_decomposition']=dict(
                        mean_squared_norm_error=statistics.mean(r['score_l2_error']**2 for r in valid),
                        squared_norm_mean_error=item['norm_mean_error']**2,
                        centered_design_variation=(count-1)/count*sum(covariance[j][j] for j in range(2)),
                        interpretation='Exact sample identity; squared sample mean error is not an unbiased estimate of squared population bias.')
                summary.append(item);groups[n]={r['design_seed']:r for r in valid}
            for item in summary[-3:]:
                if item['n']==1008:continue
                pairs=[(r,groups[1008][s]) for s,r in groups[item['n']].items() if s in groups[1008]]
                if pairs:item['paired_norm_error_change_vs_N1008']=stats([r['score_l2_error']-b['score_l2_error'] for r,b in pairs])
    numeric=[]
    for case in prepared['cases']:
        for route in ROUTES:
            group=[r for r in rows if r['phase']=='numerics' and r['data_seed']==case['data_seed'] and r['route']==route]
            baseline={r['design_seed']:r for r in group if r['arm']=='baseline'}
            for arm in dict.fromkeys(r['arm'] for r in group):
                selected=[r for r in group if r['arm']==arm and r['valid']]
                item=dict(data_seed=case['data_seed'],route=route,arm=arm,valid_count=len(selected))
                if selected:
                    item.update(mean_value_error=statistics.mean(r['value_error'] for r in selected),
                                mean_l2_error=statistics.mean(r['score_l2_error'] for r in selected),
                                mean_signed_score_error=[statistics.mean(r['score_error'][j] for r in selected) for j in range(2)],
                                paired_value_changes=[r['value']-baseline[r['design_seed']]['value'] for r in selected],
                                paired_score_changes=[[a-b for a,b in zip(r['score'],baseline[r['design_seed']]['score'])] for r in selected])
                numeric.append(item)
    verdict=dict(fd_counts={status:sum(r['status']==status for r in fd) for status in sorted(set(r['status'] for r in fd))},
                 invalid_particle_cells=sum(not r['valid'] for r in rows),
                 heuristic_losses={key:sum(r.get('heuristic_dominance',{}).get(key)=='observed_loss' for r in rows) for key in ('zero_score','first_only','gaussian_kalman')},
                 ranking='not assessed; explanatory diagnostics only',default_readiness=False,
                 derivative_interpretation='inspect branch-matched checks; unresolved cases cannot certify analytical correctness',
                 plan='docs/plans/sqmc-ksc-discrepancy-analysis-20260929.md')
    mechanism=[mechanism_summary(r) for r in rows if r['phase']=='mechanism']
    dump(out/'summary.json',dict(configuration=prepared['program'],tuning=prepared['tuning'],replications=summary,numerics=numeric,mechanism=mechanism,finite_differences=fd,decision=verdict,budget=ledger))
    dump(out/'all-evaluations.json',rows)
    with (out/'values-and-scores.csv').open('w',newline='') as f:
        names=['phase','data_seed','route','horizon','n','design_seed','arm','valid','value','reference_value','value_error','absolute_value_error','score_gamma_raw','score_log_beta','reference_gamma_raw','reference_log_beta','error_gamma_raw','error_log_beta','absolute_error_gamma_raw','absolute_error_log_beta','score_l2_error']
        writer=csv.DictWriter(f,fieldnames=names);writer.writeheader()
        for r in rows:
            flat={k:r.get(k) for k in names}
            flat['absolute_value_error']=abs(r['value_error']) if r.get('value_error') is not None else None
            for j,label in enumerate(('gamma_raw','log_beta')):
                flat['score_'+label]=r['score'][j];flat['reference_'+label]=r['reference_score'][j]
                flat['error_'+label]=r.get('score_error',[None,None])[j]
                flat['absolute_error_'+label]=r.get('absolute_score_error',[None,None])[j]
            writer.writerow(flat)
    lines=['# KSC discrepancy diagnostic tables','',prepared['program']+'. '+prepared['tuning']+'. No production/HMC/default or method-ranking inference.','',
           'All intervals below are exploratory conditional intervals over eight designs on a fixed selected dataset. Retrospective dataset selection prevents population inference.','',
           '## Reference values and scores','', '| Dataset | T | log likelihood | gamma_raw score | log_beta score |','|---|---:|---:|---:|---:|']
    for r in refs:lines.append(f"| {r['data_seed']} | {r['horizon']} | {r['value']:.9f} | {r['score'][0]:.9f} | {r['score'][1]:.9f} |")
    lines+=['','## Derivative checks','', '| Dataset | Route | T | Coordinate | Analytical | Status | Error on accepted FD steps |','|---|---|---:|---:|---:|---|---:|']
    for r in fd:lines.append(f"| {r['data_seed']} | {LABELS[r['route']]} | {r['horizon']} | {r['coordinate']} | {r['analytical_score']:.7g} | {r['status']} | {r.get('error','N/A')} |")
    lines+=['','## Conditional particle replication','', '| Dataset | Route | N | Mean logL error | Mean gamma error ± SE | Mean beta error ± SE | Mean norm error ± SE | Norm of mean error |','|---|---|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        if r['valid_count']!=8:raise RuntimeError('Invalid cells require explicit manual report disposition')
        a,b=r['score_error'];m=r['mean_norm_error']
        lines.append(f"| {r['data_seed']} | {LABELS[r['route']]} | {r['n']} | {r['value_error']['mean']:.6g} | {a['mean']:.6g} ± {a['se']:.3g} | {b['mean']:.6g} ± {b['se']:.3g} | {m['mean']:.6g} ± {m['se']:.3g} | {r['norm_mean_error']:.6g} |")
    lines+=['','## Numerical-control sensitivity','', '| Dataset | Route | Arm | Mean logL error | Mean norm score error |','|---|---|---|---:|---:|']
    for r in numeric:lines.append(f"| {r['data_seed']} | {LABELS[r['route']]} | {r['arm']} | {r.get('mean_value_error',float('nan')):.6g} | {r.get('mean_l2_error',float('nan')):.6g} |")
    lines+=['','## Decision and inference status','', '| Item | Status |','|---|---|',
            f"| Hard validity vetoes | {verdict['invalid_particle_cells']} invalid evaluated particle cells |",
            f"| Derivative classification | {verdict['fd_counts']} |",
            '| Statistically supported method ranking | Not tested; all four retained |',
            '| Descriptive differences | Numerical control arms have two designs; report individually |',
            '| Default readiness | Not established |',
            '| Next evidence | Causal localization depends on the recorded FD and convergence findings |',
            '', 'The reader-facing result note supplies causal interpretation and terminal review. Full numerical rows, signed coordinate errors, SD/SE and exploratory intervals remain in the JSON/CSV files.', '']
    (out/'report.md').write_text('\n'.join(lines))
    print(json.dumps(dict(output=str(out),rows=len(rows),fd_checks=len(fd),decision=verdict)))


if __name__=='__main__':main()
