#!/usr/bin/env python3
"""Post-run reporting only; stdlib bootstrap of paired design replications."""
import csv
import json
import math
from pathlib import Path
import random
import statistics as stats
import sys

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'docs/plans/artifacts/sqmc-ksc-reset-repair-20260929/attempt-01'
ROUTES=('iid_dual_cap','previous_inverse_cdf','repaired_permutation','repaired_permutation_ablation')
LABELS=dict(zip(ROUTES,('IID','Inverse CDF','Permutation','Permutation ablation')))

def read(p): return json.loads(p.read_text())
def mean(xs): return stats.mean(xs)
def interval(xs):
    ys=sorted(xs);return [ys[int(.025*(len(ys)-1))],ys[int(.975*(len(ys)-1))]]
def moments(rows):
    n=len(rows);errors=[r['score_error'] for r in rows]
    vmean=[mean(x[k] for x in errors) for k in range(2)]
    sd=[stats.stdev(x[k] for x in errors) if n>1 else None for k in range(2)]
    se=[x/math.sqrt(n) if x is not None else None for x in sd]
    covariance=[[sum((x[j]-vmean[j])*(x[k]-vmean[k]) for x in errors)/(n-1)
                 for k in range(2)] for j in range(2)] if n>1 else None
    rng=random.Random(61917)
    norm_samples=[]
    for _ in range(10000):
        sample=rng.choices(errors,k=n)
        norm_samples.append(math.sqrt(sum(mean(x[k] for x in sample)**2 for k in range(2))))
    return dict(n=n,mean_value=mean(r['value'] for r in rows),
      mean_score=[mean(r['score'][k] for r in rows) for k in range(2)],
      mean_score_error=vmean,replicate_score_error_sd=sd,mean_score_error_component_se=se,
      replicate_score_error_covariance=covariance,
      mean_score_error_covariance=[[x/n for x in row] for row in covariance] if covariance else None,
      mean_score_error_vector_rms_se=math.sqrt(sum(x*x for x in se)) if n>1 else None,
      norm_mean_score_error=math.sqrt(sum(x*x for x in vmean)),
      norm_mean_score_error_bootstrap_sd=stats.stdev(norm_samples),
      norm_mean_score_error_bootstrap_95ci=interval(norm_samples),
      mean_absolute_value_error=mean(r['absolute_value_error'] for r in rows),
      mean_absolute_score_error=[mean(r['absolute_score_error'][k] for r in rows) for k in range(2)],
      mean_replicate_score_l2_error=mean(r['score_l2_error'] for r in rows))

def run():
    out=BASE/'analysis-01';out.mkdir(exist_ok=False)
    results=[(p,read(p)) for p in sorted(BASE.glob('*-attempt-*/result.json'))]
    completed={r['unit']:r for _,r in results if r['status']=='finished'}
    refs={(r['data_seed'],r['horizon']):r for r in completed['reference']['rows']}
    all_rows=[]
    for _,r in results:
        for row in r['rows']:
            if 'arm' in row: all_rows.append(dict(unit=r['unit'],**row))
    (out/'all-evaluations.json').write_text(json.dumps(all_rows,indent=2)+'\n')
    fields=('unit','route','arm','horizon','data_seed','design_seed','n','valid','value','score',
            'reference_value','reference_score','value_error','absolute_value_error','score_error',
            'absolute_score_error','score_l2_error','wall_seconds')
    with (out/'all-evaluations.csv').open('w') as file:
        writer=csv.DictWriter(file,fieldnames=fields);writer.writeheader()
        for row in all_rows:
            writer.writerow({k:json.dumps(row[k]) if isinstance(row.get(k),(list,dict)) else row.get(k) for k in fields})
    cells=[];checks=[];design_sets=[];validation=[]
    for route in ROUTES:
        fd=completed['checks__'+route]['fd_checks']
        checks.append(dict(check='GPU derivatives '+route,pass_check=len(fd)==4 and all(r['status']=='pass' for r in fd)))
        for horizon in (10,120):
            selection=completed[f'calibration__{route}__{horizon}']['selection']['arm']
            validrows=completed[f'validation__{route}__{horizon}']['rows']
            vs={arm:[r for r in validrows if r['arm']==arm] for arm in ('baseline',selection)}
            validation_pass=all(r['valid'] for r in validrows)
            if validation_pass:
                bs=moments(vs['baseline']);cs=moments(vs[selection])
                validation_pass=(cs['mean_replicate_score_l2_error']<=bs['mean_replicate_score_l2_error']
                    and cs['mean_absolute_value_error']<=bs['mean_absolute_value_error']+max(.1*bs['mean_absolute_value_error'],.01))
            validation.append(dict(route=route,horizon=horizon,selected_arm=selection,
              pass_check=validation_pass,all_valid=all(r['valid'] for r in validrows),
              baseline=moments(vs['baseline']) if all(r['valid'] for r in vs['baseline']) else None,
              candidate=moments(vs[selection]) if all(r['valid'] for r in vs[selection]) else None))
            holdout=completed[f'evaluation__{route}__{horizon}']['rows'][:]
            extension=completed.get(f'extension__{route}__{horizon}')
            if extension: holdout+=extension['rows']
            for seed in (243001,243002):
                group={arm:{r['design_seed']:r for r in holdout if r['arm']==arm and r['data_seed']==seed}
                       for arm in ('baseline',selection)}
                b=group['baseline'];c=group[selection];paired=sorted(set(b)&set(c))
                design_sets.append(tuple(paired))
                valid=len(paired) in (4,8) and set(b)==set(c) and all(b[s]['valid'] and c[s]['valid'] for s in paired)
                checks.append(dict(check=f'balanced valid holdout {route}/{horizon}/{seed}',pass_check=valid and set(b)==set(c)))
                row=dict(route=route,horizon=horizon,data_seed=seed,selected_arm=selection,
                         validation_pass=validation_pass,valid=valid)
                if valid:
                    bs=moments([b[s] for s in paired]);cs=moments([c[s] for s in paired])
                    diffs=[c[s]['score_l2_error']-b[s]['score_l2_error'] for s in paired]
                    rng=random.Random(61923)
                    ci=interval([mean(rng.choices(diffs,k=len(diffs))) for _ in range(10000)])
                    ref=refs[seed,horizon]
                    heuristics={k:math.sqrt(sum((a-z)**2 for a,z in zip(v['score'],ref['score'])))
                                for k,v in ref['heuristics'].items()}
                    loss=[k for k,v in heuristics.items() if cs['mean_replicate_score_l2_error']>v]
                    value_pass=cs['mean_absolute_value_error']<=bs['mean_absolute_value_error']+max(.1*bs['mean_absolute_value_error'],.01)
                    row.update(baseline=bs,candidate=cs,paired_mean_l2_difference=mean(diffs),
                      paired_l2_difference_95ci=ci,heuristic_score_l2_errors=heuristics,heuristics_lost_to=loss,
                      heuristic_dominance_verdict='veto' if loss else 'screen_pass',
                      value_non_harm_pass=value_pass,
                      repair_nomination=value_pass and mean(diffs)<0 and validation_pass,
                      statistical_repair_support=value_pass and ci[1]<0 and validation_pass)
                    checks.append(dict(check=f'paired inputs {route}/{horizon}/{seed}',pass_check=all(b[s]['input_sha256']==c[s]['input_sha256'] for s in paired)))
                cells.append(row)
    checks.append(dict(check='all methods, horizons and datasets have identical design replication sets',
                       pass_check=len(set(design_sets))==1))
    checks.append(dict(check='full mixture reference refinement/grid',pass_check=all(r['max_agreement_error']<=1e-7 for r in refs.values())))
    route_verdicts=[]
    for route in ROUTES:
        group=[c for c in cells if c['route']==route]
        route_verdicts.append(dict(route=route,
          repair_nomination_all_cells=all(c.get('repair_nomination',False) for c in group),
          statistical_repair_support_all_cells=all(c.get('statistical_repair_support',False) for c in group),
          heuristic_promotion_veto=any(c.get('heuristics_lost_to') for c in group),
          validation_pass_all_horizons=all(c['validation_pass'] for c in group)))
    report=dict(cells=cells,validation=validation,route_verdicts=route_verdicts,checks=checks,all_evidence_checks_pass=all(x['pass_check'] for x in checks),
                failed_units=[r['unit'] for _,r in results if r['status']!='finished'],
                invalid_evaluations=sum(not r['valid'] for r in all_rows),evaluations=len(all_rows))
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=['# KSC reset repair: untouched comparisons','',
      'Full seven-mixture reference. Means average fixed-design replications; both analytical score coordinates are included.',
      'FP64 GPU/XLA diagnostic variants, N=1008. No canonical/default admission.',
      '', '| Method | T | Data | Selected arm | Mean absolute value error: old → new | Mean score L2 error: old → new | Paired L2 difference 95% interval |',
      '|---|---:|---:|---|---:|---:|---|']
    for r in cells:
        if not r['valid']: lines.append(f"| {LABELS[r['route']]} | {r['horizon']} | {r['data_seed']} | {r['selected_arm']} | invalid | invalid | unavailable |");continue
        b,c=r['baseline'],r['candidate'];ci=r['paired_l2_difference_95ci']
        lines.append(f"| {LABELS[r['route']]} | {r['horizon']} | {r['data_seed']} | {r['selected_arm']} | {b['mean_absolute_value_error']:.6g} → {c['mean_absolute_value_error']:.6g} | {b['mean_replicate_score_l2_error']:.6g} → {c['mean_replicate_score_l2_error']:.6g} | [{ci[0]:.6g}, {ci[1]:.6g}] |")
    for seed in (243001,243002):
        for horizon in (10,120):
            ref=refs[seed,horizon];g=ref['heuristics']['gaussian']
            lines+=['',f'## Actual values and scores: data {seed}, T={horizon}','',
              '| Method/arm | Mean log likelihood | Mean gamma score | Mean log-beta score | Mean-error SE: gamma, log-beta |',
              '|---|---:|---:|---:|---|',
              f"| Full seven-mixture reference | {ref['value']:.9f} | {ref['score'][0]:.9f} | {ref['score'][1]:.9f} | quadrature checked |",
              f"| Gaussian Kalman heuristic | {g['value']:.9f} | {g['score'][0]:.9f} | {g['score'][1]:.9f} | deterministic approximation |"]
            for r in cells:
                if (r['data_seed'],r['horizon'])!=(seed,horizon) or not r['valid']:continue
                for arm in ('baseline','candidate'):
                    x=r[arm];se=x['mean_score_error_component_se']
                    lines.append(f"| {LABELS[r['route']]} / {arm} | {x['mean_value']:.9f} | {x['mean_score'][0]:.9f} | {x['mean_score'][1]:.9f} | {se[0]:.5g}, {se[1]:.5g} |")
    lines+=['','Intervals are conditional on each fixed synthetic dataset and on four/eight design replications. They are exploratory, unadjusted for multiple cells, and do not establish a population-wide method ranking.',
      'Component SE is replicate error SD divided by sqrt(n). The JSON also contains the full covariance of the mean error vector, its RMS vector SE (square root of covariance trace), and bootstrap SD/interval for the norm of the mean error vector. These differ from the SD of replicate error norms.','']
    (out/'comparisons.md').write_text('\n'.join(lines))
    print(json.dumps(dict(analysis=str(out),evaluations=len(all_rows),invalid=report['invalid_evaluations'],
      checks=len(checks),checks_pass=report['all_evidence_checks_pass'],
      nominations=sum(r.get('repair_nomination',False) for r in cells),
      statistical_support=sum(r.get('statistical_repair_support',False) for r in cells),
      heuristic_vetoes=sum(bool(r.get('heuristics_lost_to')) for r in cells))))

if __name__=='__main__': run()
