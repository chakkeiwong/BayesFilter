#!/usr/bin/env python3
"""Post-run diagnostic reporting; never selects runtime controls or an oracle."""
from __future__ import annotations
import csv
import json
import math
from pathlib import Path
import statistics
from collections import defaultdict
from scipy.io import loadmat
from bayesfilter.testing.zhao_cui_checkpoint_reference import checkpoint_metadata
from docs.benchmarks.run_ledh_zhao_horizons import ROOT, OUT, HORIZONS, MODELS, dump, read, budget_accounting, all_attempts

NAMES={'predator_prey':['r','K','a','s','u','v'],
       'sir_d18':['log_kappa_scale','log_nu_scale','log_observation_noise_scale']}

def author_method(model,rank):
    return 'zhao_cui' if rank==({'predator_prey':20,'sir_d18':40}[model]) else f'zhao_cui_rank{rank}'

def stats(values):
    return dict(mean=statistics.mean(values),se=statistics.stdev(values)/math.sqrt(len(values)) if len(values)>1 else None,n=len(values))

def selected_author_scores(rows):
    """One prespecified-radius score per actual fit, including timeout recovery."""
    selected={}
    for row in rows:
        required={'predator_prey':.0003125,'sir_d18':.00015625}[row['model']]
        key=(row['model'],row['horizon'],row['method'],row['proposal']['directory'])
        for estimate in row['estimates']:
            if not math.isclose(estimate['radius'],required,rel_tol=0.,abs_tol=1e-15): continue
            candidate=dict(model=row['model'],horizon=row['horizon'],method=row['method'],estimate=estimate,source=row['source'])
            if key in selected and selected[key]['estimate']['score']!=estimate['score']:
                raise ValueError('conflicting repeated score for the same proposal and radius')
            selected[key]=candidate
    return list(selected.values())


def report():
    attempts=all_attempts()
    number=1
    while (OUT/f'report-{number:03d}').exists(): number+=1
    destination=OUT/f'report-{number:03d}'
    destination.mkdir()
    groups=defaultdict(list); raw_rows=[]; cells=[]; zhao_rows=[]; zhao_scores=[]; invalid=[]; paired=[]
    def cell(model,horizon,method,quantity,mean,se,n,uncertainty):
        cells.append(dict(model=model,horizon=horizon,method=method,quantity=quantity,
                          mean=mean,se=se,n=n,uncertainty=uncertainty))
    for attempt in attempts:
        directory=Path(attempt['output']); label=attempt['label']
        if label.startswith(('ledh-','covariance-')):
            files=[directory/'rows.json'] if (directory/'rows.json').exists() else list(directory.glob('worker-*/rows.json'))
            for file in files:
                for row in read(file):
                    expected=read(OUT/'inputs'/f"{row['model']}-T{row['horizon']}"/'dataset.json')
                    if row['observation_sha256']!=expected['observation_sha256']:
                        raise ValueError('LEDH dataset mismatch')
                    method=('covariance_only' if row['arm']=='covariance_only' else row['importance_weight_policy'])
                    row=dict(row,method=method,source=str(file)); raw_rows.append(row)
                    if not row['valid']: invalid.append(row); continue
                    groups[(row['model'],row['horizon'],method)].append(row)
        if label=='bootstrap-references' and (directory/'summary.json').exists():
            for row in read(directory/'summary.json'):
                model,horizon=row['model'],row['horizon']
                expected=read(OUT/'inputs'/f'{model}-T{horizon}'/'dataset.json')
                if row['observation_sha256']!=expected['observation_sha256']: raise ValueError('bootstrap dataset mismatch')
                method=f"bootstrap_N{row['particles']}"
                cell(model,horizon,method,'log_likelihood',row['log_likelihood'],row['jackknife_mcse_log_likelihood'],row['replications'],'replication jackknife; excludes particle bias')
                for k,name in enumerate(NAMES[model]):
                    cell(model,horizon,method,name,row['score'][k],row['jackknife_mcse_score'][k],row['replications'],'replication jackknife; excludes particle bias')
        if label.startswith('zhao-') and (directory/'manifest.json').exists():
            settings=read(directory/'manifest.json')['settings']
            model='predator_prey' if settings['model']=='pp' else 'sir_d18'
            source_dataset=read(directory/(settings['model']+'-input-dataset.json'))
            if source_dataset['observation_sha256']!=read(OUT/'inputs'/f'{model}-T50'/'dataset.json')['observation_sha256']:
                raise ValueError('author reference dataset mismatch')
            for horizon in HORIZONS:
                saved=directory/f'smoothing-t{horizon:02d}.mat'
                if not saved.exists(): continue
                try: _,parent=checkpoint_metadata(directory,horizon)
                except LookupError: continue
                mat=loadmat(saved,variable_names=['raw_log_weight','lml','legacy_mean_log_weight'])
                weights=mat['raw_log_weight'].reshape(-1).tolist()
                if any(math.isnan(w) or w==math.inf for w in weights): raise ValueError('invalid author weights')
                maximum=max(weights); scaled=[math.exp(w-maximum) for w in weights]
                total=sum(scaled); ess=total*total/sum(w*w for w in scaled)
                ell=maximum+math.log(total/len(weights))
                if abs(ell-float(mat['lml'].item()))>1e-7: raise ValueError('source corrected likelihood mismatch')
                legacy=float(mat['legacy_mean_log_weight'].item()) if 'legacy_mean_log_weight' in mat else None
                zhao_rows.append(dict(model=model,horizon=horizon,method=author_method(model,settings['rank']),parent_status=parent['parent_status'],fit_seed=settings['fit_seed'],rank=settings['rank'],
                    log_likelihood=ell,raw_author_lml=legacy,paths=len(weights),ess=ess,
                    conditional_log_likelihood_se=math.sqrt(max(0.,(len(weights)/ess-1)/(len(weights)-1))),source=str(saved)))
        if label.startswith('quadratic-') and (directory/'result.json').exists():
            result=read(directory/'result.json')
            command=attempt['command']; model=command[command.index('--model')+1]; horizon=int(command[command.index('--horizon')+1])
            if result['status']!='complete' and not (result['status']=='failed' and result.get('failure_type')=='TimeoutError'): continue
            for proposal_index,proposal in enumerate(result['proposals']):
                if proposal['observation_sha256']!=read(OUT/'inputs'/f'{model}-T{horizon}'/'dataset.json')['observation_sha256']:
                    raise ValueError('quadratic prefix mismatch')
                estimates=[e for e in result['estimates'] if e['proposal']==proposal_index]
                method=author_method(model,proposal['rank'])
                zhao_scores.append(dict(model=model,horizon=horizon,method=method,proposal=proposal,estimates=estimates,parent_status=result['status'],source=str(directory/'result.json')))
    for row in selected_author_scores(zhao_scores):
        model,horizon,method=row['model'],row['horizon'],row['method']
        smallest=row['estimate']
        for k,name in enumerate(NAMES[model]):
            groups[(model,horizon,'zhao_score:'+method+':'+name)].append(dict(value=smallest['score'][k],
                conditional_se=smallest['conditional_path_jackknife'].get('standard_error',[None]*len(NAMES[model]))[k],
                radius=smallest['radius'],source=row['source']))
    for (model,horizon,method),rows in groups.items():
        if method.startswith('zhao_score:'):
            _,author,quantity=method.split(':',2)
            st=stats([r['value'] for r in rows])
            uncertainty='between-fit SE; conditional SE retained separately'
            if len(rows)==1:
                st['se']=rows[0]['conditional_se']; uncertainty='one fit: conditional path jackknife SE only; excludes fit/support/radius bias'
            cell(model,horizon,author,quantity,**st,uncertainty=uncertainty)
            continue
        for quantity,values in [('log_likelihood',[r['log_likelihood'] for r in rows]),
                *[(name,[r['score'][k] for r in rows]) for k,name in enumerate(NAMES[model])]]:
            cell(model,horizon,method,quantity,**stats(values),uncertainty='between-design SE, four paired seeds at one dataset')
        if method=='marginal_mixture':
            original={r['design_seed']:r for r in groups.get((model,horizon,'ancestor'),[])}
            for row in rows:
                if row['design_seed'] in original:
                    other=original[row['design_seed']]
                    paired.append(dict(model=model,horizon=horizon,seed=row['design_seed'],
                        log_likelihood_change=row['log_likelihood']-other['log_likelihood'],
                        score_change=[a-b for a,b in zip(row['score'],other['score'])]))
    for model in MODELS:
        for horizon in HORIZONS:
            methods={r['method'] for r in zhao_rows if r['model']==model and r['horizon']==horizon}
            for method in sorted(methods):
                rows=[r for r in zhao_rows if r['model']==model and r['horizon']==horizon and r['method']==method]
                st=stats([r['log_likelihood'] for r in rows])
                uncertainty='between-fit SE; conditional path SE in original-author-values.json'
                if len(rows)==1:
                    st['se']=rows[0]['conditional_log_likelihood_se']; uncertainty='one fit: conditional path delta-method SE only; excludes proposal bias'
                cell(model,horizon,method,'log_likelihood',**st,uncertainty=uncertainty)
    lookup={(c['model'],c['horizon'],c['method'],c['quantity']):c for c in cells}
    comparisons=[]
    for c in cells:
        key=(c['model'],c['horizon'])
        for reference_method in ('zhao_cui','zhao_cui_rank20','bootstrap_N131072'):
            reference=lookup.get((*key,reference_method,c['quantity']))
            if reference and c['method']!=reference_method:
                comparisons.append(dict(**c,reference_method=reference_method,reference=reference['mean'],reference_se=reference['se'],
                    signed_error=c['mean']-reference['mean'],absolute_error=abs(c['mean']-reference['mean'])))
    heuristics=[]
    errors={(c['model'],c['horizon'],c['method'],c['quantity'],c['reference_method']):c for c in comparisons}
    for c in comparisons:
        if c['method']!='marginal_mixture': continue
        for method in ('ancestor','covariance_only','bootstrap_N1008'):
            other=errors.get((c['model'],c['horizon'],method,c['quantity'],c['reference_method']))
            if other:
                heuristics.append(dict(model=c['model'],horizon=c['horizon'],quantity=c['quantity'],heuristic=method,reference_method=c['reference_method'],
                    marginal_absolute_error=c['absolute_error'],heuristic_absolute_error=other['absolute_error'],
                    difference=c['absolute_error']-other['absolute_error'],
                    descriptive_underperformance=c['absolute_error']>other['absolute_error']))
    dump(destination/'cells.json',cells); dump(destination/'original-author-values.json',zhao_rows)
    dump(destination/'original-author-scores.json',zhao_scores)
    dump(destination/'ledh-replicates.json',raw_rows); dump(destination/'paired-changes.json',paired)
    dump(destination/'reference-errors.json',comparisons); dump(destination/'heuristic-comparisons.json',heuristics)
    derivative_checks=[]
    for result_file in sorted(OUT.glob('score-check-*/result.json')):
        result=read(result_file)
        derivative_checks.append(dict(source=str(result_file),status=result['status'],passes=result['passes'],
            rows=[dict(model=r['model'],policy=r['policy'],passes=r['passes'],
                max_normalized_error=max(max(e) for e in r['normalized_errors']),
                trace_parity=r['trace_parity'],branch_changes=r.get('trace_branch_changes')) for r in result['rows']]))
    dump(destination/'derivative-checks.json',derivative_checks)
    dump(destination/'decision.json',dict(invalid_evaluations=len(invalid),
        derivative_checks=derivative_checks,
        parent_reference_status=[dict(model=r['model'],horizon=r['horizon'],rank=r['rank'],fit_seed=r['fit_seed'],status=r['parent_status']) for r in zhao_rows],statistical_ranking='unsupported',
        default_readiness='not assessed; untuned diagnostic scopes',
        heuristic_dominance='descriptive comparison only; promotion withheld',
        incomplete_attempts=[r['label'] for r in attempts if r['status']!='complete'],
        budget=budget_accounting(attempts)))
    with (destination/'values-and-scores.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['model','horizon','method','quantity','mean','se','n','uncertainty'])
        writer.writeheader(); writer.writerows(cells)
    def number(c):
        if not c: return 'pending'
        return f"{c['mean']:.8g}"+(f" ± {c['se']:.3g}" if c['se'] is not None else ' (one fit)')
    methods=['ancestor','marginal_mixture','zhao_cui','bootstrap_N131072']
    lines=['# Matched likelihoods and scores','','Values are means ± one Monte Carlo standard error where available. Each model uses one frozen dataset and exact time prefixes. Four paired LEDH designs quantify seed variation. References are approximate; their SE excludes bias.','','| Model | T | Ancestor LEDH | Mixture LEDH | Original-author TT | Bootstrap N=131072 |','|---|---:|---:|---:|---:|---:|']
    for model in MODELS:
        for horizon in HORIZONS:
            lines.append('| '+model+' | '+str(horizon)+' | '+' | '.join(number(lookup.get((model,horizon,m,'log_likelihood'))) for m in methods)+' |')
    lines+=['','Scores are derivatives in the displayed physical parameter coordinates for predator–prey and log-scale coordinates for SIR.','','| Model | T | Parameter | Ancestor LEDH | Mixture LEDH | Original-author quadratic | Bootstrap N=131072 |','|---|---:|---|---:|---:|---:|---:|']
    for model in MODELS:
        for horizon in HORIZONS:
            for name in NAMES[model]:
                lines.append('| '+model+' | '+str(horizon)+' | '+name+' | '+' | '.join(number(lookup.get((model,horizon,m,name))) for m in methods)+' |')
    lines+=['','The separate SIR rank-20 calculation is a rank diagnostic. Its uncertainty is conditional on one fitted proposal. It is not pooled with rank40.',
        '','| T | Quantity | Original-author rank20 | Original-author rank40 |','|---:|---|---:|---:|']
    for horizon in HORIZONS:
        for quantity in ('log_likelihood',*NAMES['sir_d18']):
            lines.append('| '+str(horizon)+' | '+quantity+' | '+' | '.join(number(lookup.get(('sir_d18',horizon,method,quantity))) for method in ('zhao_cui_rank20','zhao_cui'))+' |')
    lines+=['','Finite-program derivative diagnostics (implementation checks, not statistical score accuracy):',
        '','| Check | Model | Policy | Largest normalized error over recorded steps | Trace parity | Screen |','|---|---|---|---:|---|---|']
    for check in derivative_checks:
        for row in check['rows']:
            lines.append(f"| {Path(check['source']).parent.name} | {row['model']} | {row['policy']} | {row['max_normalized_error']:.5g} | {row['trace_parity']} | {row['passes']} |")
    lines+=['','The author TT column uses logmeanexp of the original raw importance weights. The raw author lml diagnostic (mean log weight), conditional path SE, ESS and fit seeds are in original-author-values.json. TT score uncertainties use between-fit SE when independent fits exist, otherwise conditional path jackknife SE from one fit; the CSV labels this distinction. Rank20 SIR checks are retained separately from rank40 and never pooled. TT scores use the smallest predeclared regression radius; all radii and heldout checks remain in their run directories.','','| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |','|---|---|---|---|---|---|','| Hold promotion | Numerical comparisons reported continuously | See invalid and unfinished attempts in decision.json | Reference bias, one dataset, limited independent fits | Complete pending runs; assess particle/rank/radius stability | Superiority, oracle certification, HMC/default readiness |','','| Inference status | Finding |','|---|---|',f'| Hard veto screen | {len(invalid)} invalid returned LEDH evaluations; incomplete jobs are separately listed |','| Statistically supported ranking | None established |','| Descriptive differences | Per-component errors and paired changes retained |','| Default readiness | Not assessed; controls are untuned in these scopes |','| Next evidence | Stable independent references and fresh scope-specific tuning/holdout data |','', 'Post-run alternative explanation: reference bias and untuned moment/reset controls can dominate a weighting difference. Agreement of two approximations alone is insufficient; the weakest evidence is long-horizon score accuracy with limited independent fits.']
    (destination/'results.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(report=str(destination),cells=len(cells),invalid=len(invalid))))
    return destination

if __name__=='__main__': report()
