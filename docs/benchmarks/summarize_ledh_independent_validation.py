#!/usr/bin/env python3
"""Post-run diagnostic reporting only; no tuning, selection or runtime effects."""
from __future__ import annotations
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'docs/plans/artifacts/ledh-independent-validation-20261008-01'
RESULT = ROOT / 'docs/benchmarks/ledh-independent-validation-results-20261008.md'
METHODS = ('old', 'old_covariance_only', 'new')
T975 = {1:12.7062047364, 2:4.30265272975, 3:3.18244630528, 7:2.36462425101}
NAMES = {'lgssm':['phi_coordinate','log_q_scale','log_r_scale','initial_mean_scale'],
         'ksc':['Phi_inverse_gamma','log_beta'],
         'predator_prey':['r','carrying_capacity','half_saturation','s','u','v'],
         'sir_d18':['log_kappa_scale','log_nu_scale','log_observation_noise_scale']}
read = lambda p: json.loads(Path(p).read_text())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


def stats(values):
    values = list(values)
    if not values or not all(math.isfinite(v) for v in values):
        raise ValueError('empty or nonfinite diagnostic sample')
    n = len(values)
    mean = statistics.mean(values)
    sd = statistics.stdev(values) if n>1 else None
    se = sd/math.sqrt(n) if sd is not None else None
    half = se*T975[n-1] if n-1 in T975 else None
    return dict(n=n,mean=mean,sd=sd,se=se,
                interval95=None if half is None else [mean-half,mean+half])


def combine(rows):
    """Equal-size independent importance estimates; derivative of mean likelihood."""
    maximum=max(r['value'] for r in rows)
    weights=[math.exp(r['value']-maximum) for r in rows]
    total=sum(weights)
    return dict(value=maximum+math.log(total/len(rows)),
        score=[sum(w*r['score'][j] for w,r in zip(weights,rows))/total
               for j in range(len(rows[0]['score']))])


def pooled_reference(rows):
    result=combine(rows)
    n=len(rows)
    result['replications']=n
    if n>1:
        leave=[combine(rows[:i]+rows[i+1:]) for i in range(n)]
        components=[[r['value'],*r['score']] for r in leave]
        se=[math.sqrt((n-1)/n*sum((v[j]-statistics.mean(x[j] for x in components))**2
                    for v in components)) for j in range(len(components[0]))]
        result.update(value_se=se[0],score_se=se[1:])
    else:
        result.update(value_se=None,score_se=None)
    return result


def errors(row, ref):
    return dict(absolute_value_error=abs(row['value']-ref['value']),
        score_l2_error=math.sqrt(sum((v-r)**2 for v,r in zip(row['score'],ref['score']))))


def arm(rows, ref=None):
    if not rows: return dict(complete=False,n=0,invalid=[])
    invalid=[r['seed'] for r in rows if not r['valid'] or
             not all(math.isfinite(v) for v in [r['value'],*r['score']])]
    result=dict(complete=len(rows)==8 and not invalid,n=len(rows),invalid=invalid)
    if not result['complete']: return result
    result.update(value=stats(r['value'] for r in rows),
        score=[stats(r['score'][j] for r in rows) for j in range(len(rows[0]['score']))])
    if ref is not None:
        es=[errors(r,ref) for r in rows]
        result.update({k:stats(e[k] for e in es) for k in es[0]})
        result['score_error_of_mean']=math.sqrt(sum((s['mean']-r)**2
                                                  for s,r in zip(result['score'],ref['score'])))
    return result


def paired(rows, baseline, ref):
    old={r['seed']:r for r in rows if r['method']==baseline}
    new={r['seed']:r for r in rows if r['method']=='new'}
    if set(old)!=set(new) or not arm(list(old.values()))['complete'] or not arm(list(new.values()))['complete']:
        return dict(complete=False)
    differences={key:[] for key in ('absolute_value_error','score_l2_error')}
    for seed in sorted(old):
        a,b=errors(old[seed],ref),errors(new[seed],ref)
        for key in differences: differences[key].append(b[key]-a[key])
    return dict(complete=True,**{k:stats(v) for k,v in differences.items()},
                interpretation='paired design t interval conditional on these observations and this reference')


def reference_rows(attempts, model, seed, expected_hash, sources):
    refs={}
    for job in attempts:
        folder=Path(job['output'])
        if job['kind']=='bootstrap' and job['status']=='complete' and (folder/'summary.json').exists():
            for name in ('summary.json','replications.json'):
                sources[str(folder/name)]=sha(folder/name)
            for row in read(folder/'summary.json'):
                if row['model']!=model or row['data_seed']!=seed: continue
                if row['observation_sha256']!=expected_hash: raise ValueError('bootstrap observation mismatch')
                individual=[r for r in read(folder/'replications.json')
                    if r['model']==model and r['data_seed']==seed and r['particles']==row['particles']]
                if len(individual)!=4 or {r['seed'] for r in individual}!=set(range(261008501,261008505)):
                    raise ValueError('bootstrap replication contract mismatch')
                if any(r['observation_sha256']!=expected_hash or r['horizon']!=50 for r in individual):
                    raise ValueError('bootstrap replication target mismatch')
                refs['bootstrap_N'+str(row['particles'])]=dict(value=row['log_likelihood'],score=row['score'],
                    value_se=row['jackknife_mcse_log_likelihood'],score_se=row['jackknife_mcse_score'],
                    replications=row['replications'],complete=row['replications']==4,
                    kind='bootstrap Fisher reference; finite-particle bias not bounded',source=str(folder/'summary.json'),
                    diagnostics={k:row[k] for k in ('minimum_particle_ess','maximum_particle_weight',
                        'minimum_final_distinct_initial_ancestors','between_replication_likelihood_ess')},
                    individual_runs=[dict(value=r['log_likelihood'],score=r['score'],seed=r['seed']) for r in individual])
    fit_rows=[]
    for job in attempts:
        if job['kind']!='zhao_score' or job['status']!='complete' or job['model']!=model or job['data_seed']!=seed: continue
        folder=Path(job['output'])
        if not (folder/'result.json').exists(): continue
        result=read(folder/'result.json')
        if result['status']!='complete': continue
        sources[str(folder/'result.json')]=sha(folder/'result.json')
        for index,proposal in enumerate(result['proposals']):
            if proposal['observation_sha256']!=expected_hash: raise ValueError('Zhao observation mismatch')
            if proposal['joint_density_parity_max']>1e-7 or proposal['evaluated_paths']!=100000:
                raise ValueError('Zhao density/path contract mismatch')
            if (proposal['rank']!=20 or proposal['horizon']!=50 or proposal['fit_seed'] not in (41,53)
                    or proposal['smooth_seed']!=proposal['fit_seed']+3000):
                raise ValueError('Zhao fit scope mismatch')
            estimates=[e for e in result['estimates'] if e['proposal']==index]
            target_radius=.0003125 if model=='predator_prey' else .00015625
            selected=[e for e in estimates if e['radius']==target_radius]
            if len(selected)!=1 or len(estimates)!=2: raise ValueError('missing predeclared score radius')
            estimate=selected[0]
            fit_rows.append(dict(value=proposal['log_likelihood'],score=estimate['score'],
                value_se=proposal.get('conditional_log_likelihood_jackknife_se'),
                score_se=estimate['conditional_path_jackknife'].get('standard_error'),
                fit_seed=proposal['fit_seed'],rank=proposal['rank'],radius=target_radius,
                source=str(folder/'result.json'),provenance=proposal,estimates=estimates))
    if fit_rows:
        if len({r['fit_seed'] for r in fit_rows})!=len(fit_rows): raise ValueError('duplicate reference fit')
        refs['zhao_rank20']=dict(**pooled_reference(fit_rows),complete=len(fit_rows)==2,
            kind='author-path importance/quadratic numerical reference; rank/support/radius bias not bounded',
            fits=fit_rows,uncertainty='between-fit delete-one; conditional path uncertainty also preserved per fit')
    return refs


def report():
    campaign=read(OUT/'campaign.json');attempts=read(OUT/'attempts.json');run=read(OUT/'run.json')
    folder=OUT/('report-'+time.strftime('%Y%m%d-%H%M%S',time.gmtime()))
    folder.mkdir(exist_ok=False)
    failures=[dict(label=r['label'],status=r['status'],exit_code=r.get('exit_code'),log=r['log'])
              for r in attempts if r['status'] not in ('complete','running')]
    running=[dict(label=r['label'],pid=r.get('pid'),log=r['log']) for r in attempts if r['status']=='running']
    datasets=[];raw=[];sources={};problems=[]
    for data in campaign['data']:
        model,seed=data['model'],data['data_seed']
        expected=campaign['designs'][str(seed)]
        matches=[a for a in attempts if a['kind']=='filter' and a['model']==model and a['data_seed']==seed]
        if len(matches)!=1: problems.append(f'{model}/{seed}: missing or duplicate filter attempt');continue
        job=matches[0];worker=Path(job['output'])
        if not (worker/'rows.json').exists() or not (worker/'manifest.json').exists():
            problems.append(f'{model}/{seed}: missing filter results');continue
        manifest=read(worker/'manifest.json');rows=read(worker/'rows.json')
        if len(NAMES[model])!=len(manifest['theta']): raise ValueError('parameter label mismatch')
        sources[str(worker/'manifest.json')]=sha(worker/'manifest.json')
        sources[str(worker/'rows.json')]=sha(worker/'rows.json')
        if manifest['data']['observation_sha256']!=data['observation_sha256']:
            raise ValueError('filter observation mismatch')
        if manifest['tuning_sha256']!=campaign['frozen_tuning'][model]['sha256']:
            raise ValueError('filter tuning mismatch')
        for key,value in manifest['source_sha256'].items():
            if key in campaign['source_sha256'] and value!=campaign['source_sha256'][key]:
                raise ValueError('filter numerical source changed: '+key)
        for design in expected:
            group=[r for r in rows if r['seed']==design]
            if sorted(r['method'] for r in group)!=sorted(METHODS):
                problems.append(f'{model}/{seed}/{design}: incomplete method triplet');continue
            for key in ('initial','process'):
                if len({r['input_sha256'][key] for r in group})!=1: raise ValueError('unpaired random inputs')
        if len(rows)!=24 or {r['seed'] for r in rows}!=set(expected):
            problems.append(f'{model}/{seed}: wrong design count')
        refs=reference_rows(attempts,model,seed,data['observation_sha256'],sources)
        if manifest['reference'].get('value') is not None: refs['model_reference']=manifest['reference']
        for estimate in [*rows,*refs.values()]:
            if len(estimate['score'])!=len(NAMES[model]): raise ValueError('score dimension mismatch')
        for name,ref in refs.items():
            if not all(math.isfinite(v) for v in [ref['value'],*ref['score']]):
                raise ValueError('nonfinite reference: '+name)
        if model in ('predator_prey','sir_d18'):
            for name in ('bootstrap_N1008','bootstrap_N32768','bootstrap_N131072','zhao_rank20'):
                if name not in refs or not refs[name]['complete']: problems.append(f'{model}/{seed}: incomplete {name}')
        arms={method:arm([r for r in rows if r['method']==method]) for method in METHODS}
        comparisons={}
        for name,ref in refs.items():
            comparisons[name]=dict(arms={method:arm([r for r in rows if r['method']==method],ref) for method in METHODS},
                paired={base:paired(rows,base,ref) for base in METHODS if base!='new'})
        heuristic=[]
        for refname in ('model_reference','bootstrap_N131072','zhao_rank20'):
            if refname not in comparisons: continue
            comparison=comparisons[refname]
            for base in ('old','old_covariance_only'):
                p=comparison['paired'][base]
                if p['complete']:
                    for metric in ('absolute_value_error','score_l2_error'):
                        heuristic.append(dict(reference=refname,baseline=base,metric=metric,
                            observed_new_minus_baseline=p[metric]['mean'],
                            observed_loss=p[metric]['mean']>0,
                            interval95=p[metric]['interval95'],evidence='conditional paired design comparison'))
            if 'bootstrap_N1008' in refs and arms['new']['complete']:
                be={key:statistics.mean(errors(r,refs[refname])[key] for r in refs['bootstrap_N1008']['individual_runs'])
                    for key in ('absolute_value_error','score_l2_error')}
                for metric in be:
                    heuristic.append(dict(reference=refname,baseline='bootstrap_N1008_individual_runs',metric=metric,
                        observed_new_minus_baseline=comparison['arms']['new'][metric]['mean']-be[metric],
                        observed_loss=comparison['arms']['new'][metric]['mean']>be[metric],
                        evidence='descriptive unpaired comparison of eight proposal runs and four same-N bootstrap runs'))
        datasets.append(dict(model=model,data_seed=seed,parameter_names=NAMES[model],theta=manifest['theta'],
            reference=refs,arms=arms,comparisons=comparisons,heuristic_comparisons=heuristic,
            beta=manifest['tuning']['beta'],worker=str(worker),complete=all(a['complete'] for a in arms.values()),
            reference_difference=None if not all(k in refs for k in ('bootstrap_N131072','zhao_rank20')) else
                dict(zhao_minus_bootstrap_value=refs['zhao_rank20']['value']-refs['bootstrap_N131072']['value'],
                     zhao_minus_bootstrap_score=[a-b for a,b in zip(refs['zhao_rank20']['score'],refs['bootstrap_N131072']['score'])])))
        for row in rows:
            raw.append(dict(model=model,data_seed=seed,**{k:row[k] for k in ('method','seed','value','score','valid','wall_seconds','input_sha256')}))
    across=[]
    for model in campaign['models']:
        selected=[d for d in datasets if d['model']==model]
        if len(selected)!=2: continue
        for refname in set.intersection(*(set(d['comparisons']) for d in selected)):
            for metric in ('absolute_value_error','score_l2_error'):
                pairs=[d['comparisons'][refname]['paired']['old'] for d in selected]
                if all(p['complete'] for p in pairs):
                    across.append(dict(model=model,reference=refname,metric=metric,
                        dataset_level=stats(p[metric]['mean'] for p in pairs),
                        interpretation='two datasets, df=1; weak regime inference, reference uncertainty excluded'))
    invalid=[dict(model=r['model'],data_seed=r['data_seed'],method=r['method'],seed=r['seed']) for r in raw if not r['valid'] or not all(math.isfinite(x) for x in [r['value'],*r['score']])]
    decision=dict(campaign_complete=run['status']=='finished' and not failures and not running and not problems and len(datasets)==8,
        hard_veto_screen=dict(invalid_rows=invalid,failed_attempts=failures,incomplete_evidence=problems),
        running_attempts=running,
        statistically_supported_ranking='none claimed across data-generating regimes; reference bias not bounded for nonlinear models',
        descriptive_only='raw differences, per-design means, two-dataset trends, nonlinear reference differences',
        default_readiness='not evaluated or promoted',
        next_evidence_needed='more independent datasets and nonlinear reference rank/path convergence before a broad ranking',
        heuristic_dominance_verdict='descriptive_losses_present_promotion_veto' if any(h['observed_loss'] for d in datasets for h in d['heuristic_comparisons']) else ('incomplete_comparisons' if problems or running or failures else 'no_observed_loss_in_constructed_comparisons_not_certification'))
    result=dict(campaign=campaign,run=run,decision=decision,datasets=datasets,across_datasets=across,rows=raw,
        input_sha256=sources,worker_seconds=sum(r.get('wall_seconds',0) for r in attempts))
    dump(folder/'comparison.json',result)
    with (folder/'values-and-scores.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['model','data_seed','method','design_seed','valid','log_likelihood','score_json'])
        for r in raw: writer.writerow([r['model'],r['data_seed'],r['method'],r['seed'],r['valid'],r['value'],json.dumps(r['score'])])
        for d in datasets:
            for name,ref in d['reference'].items():
                writer.writerow([d['model'],d['data_seed'],name,'reference','numerical_reference',ref['value'],json.dumps(ref['score'])])
    text=markdown(result,folder)
    (folder/'comparison.md').write_text(text)
    RESULT.write_text(text)
    print(json.dumps(dict(report=str(folder),decision=decision,worker_seconds=result['worker_seconds']),indent=2))
    return 0 if decision['campaign_complete'] and not invalid else 1


def markdown(result,folder):
    decision=result['decision']
    lines=['# Independent T=50 validation of the frozen covariance-guided mixture','',
        'The settings from the matched comparison were frozen before these new observations and particle designs. '
        'Each dataset has eight paired designs at N=1008, T=50, FP64 GPU/XLA. '
        'The raw likelihood and every score coordinate are in the linked CSV; reference fits and uncertainty are preserved in the JSON.','',
        f"Campaign complete: **{decision['campaign_complete']}**. Default promotion: **not evaluated**.",
        f"Heuristic comparison: `{decision['heuristic_dominance_verdict']}`.", '',
        'The new method has larger observed errors than simpler comparators in some model/dataset/metric combinations. '
        'These observations do not support universal non-deterioration. '
        'These losses block promotion; they do not establish that the method is worse in every setting.' if decision['heuristic_dominance_verdict'].startswith('descriptive_losses') else
        'Completion and conditional comparisons must be checked before interpretation.', '',
        '| Model | Data seed | Reference | Method | Mean log likelihood | Mean absolute likelihood error | Mean score L2 error | Error of mean score |',
        '|---|---:|---|---|---:|---:|---:|---:|']
    for d in result['datasets']:
        for refname,c in d['comparisons'].items():
            if refname not in ('model_reference','bootstrap_N131072','zhao_rank20'): continue
            for method,a in c['arms'].items():
                if a['complete']:
                    lines.append(f"| {d['model']} | {d['data_seed']} | {refname} | {method} | {a['value']['mean']:.6f} | {a['absolute_value_error']['mean']:.6f} | {a['score_l2_error']['mean']:.6f} | {a['score_error_of_mean']:.6f} |")
                else: lines.append(f"| {d['model']} | {d['data_seed']} | {refname} | {method} | incomplete | | | |")
    lines+=['','Negative paired differences favor the new method. These intervals vary particle designs conditional on one dataset and a fixed numerical reference. They exclude reference bias.','',
        '| Model | Data seed | Reference | Error | Paired new minus old | Conditional 95% interval |',
        '|---|---:|---|---|---:|---|']
    for d in result['datasets']:
        for refname,c in d['comparisons'].items():
            if refname not in ('model_reference','bootstrap_N131072','zhao_rank20'): continue
            p=c['paired']['old']
            if not p['complete']: continue
            for metric in ('absolute_value_error','score_l2_error'):
                s=p[metric];bounds=s['interval95']
                lines.append(f"| {d['model']} | {d['data_seed']} | {refname} | {metric} | {s['mean']:.6f} | [{bounds[0]:.6f}, {bounds[1]:.6f}] |")
    lines+=actual_values_markdown(result)
    lines+=reference_diagnostics_markdown(result)
    lines+=['','## Decision and uncertainty','',
        '| Item | Finding |','|---|---|',
        f"| Completion | {decision['campaign_complete']} |",
        f"| Primary criterion | Continuous likelihood and score errors reported per model and dataset; {decision['heuristic_dominance_verdict']} |",
        f"| Veto diagnostics | {len(decision['hard_veto_screen']['invalid_rows'])} invalid rows; {len(decision['hard_veto_screen']['failed_attempts'])} failed attempts; {len(decision['running_attempts'])} running attempts; {len(decision['hard_veto_screen']['incomplete_evidence'])} missing-evidence findings |",
        '| Main uncertainty | Two datasets; finite particle, TT rank/support, and regression-radius bias |',
        '| Next justified action | Inspect conditional failures and reference stability; preserve frozen settings |',
        '| Not concluded | Universal improvement, exact nonlinear oracle, default readiness or HMC validity |','',
        '| Inference item | Status |','|---|---|',
        '| Hard veto screen | See explicit invalid/failed/missing records in comparison.json |',
        f"| Statistically supported ranking | {decision['statistically_supported_ranking']} |",
        f"| Descriptive-only differences | {decision['descriptive_only']} |",
        f"| Default readiness | {decision['default_readiness']} |",
        f"| Next evidence needed | {decision['next_evidence_needed']} |",'',
        'The strongest alternative explanation is favorable or unfavorable data-specific geometry combined with shared reference approximation error. '
        'Agreement of two fits alone cannot exclude shared rank or support bias. '
        'Independent-data reversals or a stable, more accurate reference would overturn a favorable interpretation. '
        'The weakest inferential element is the small number of datasets.','',
        f"Artifacts: `{folder.relative_to(ROOT)}/comparison.json` and `values-and-scores.csv`.",
        f"Aggregate worker time: {result['worker_seconds']/3600:.3f} hours.",
        'Plan: `docs/plans/ledh-independent-validation-20261008.md`.']
    return '\n'.join(lines)+'\n'


def vector(values):
    return 'not estimated' if values is None else '['+', '.join(f'{x:.6f}' for x in values)+']'


def scalar(value):
    return 'not estimated' if value is None else f'{value:.6f}'


def actual_values_markdown(result):
    lines=['','## Actual likelihood and score values','',
        'Filter entries are arithmetic means over eight designs; their standard errors measure design variability. '
        'Reference entries pool likelihood estimates before taking logs and weight scores by likelihood. '
        'Reference standard errors exclude approximation bias. The exact/refined reference has no Monte Carlo standard error.', '',
        '| Model | Data seed | Method/reference | Log likelihood | SE | Score vector | Score SE |',
        '|---|---:|---|---:|---:|---|---|']
    for d in result['datasets']:
        lines.append(f"| {d['model']} | {d['data_seed']} | coordinate order | | | {', '.join(d['parameter_names'])} | |")
        for method,a in d['arms'].items():
            if a['complete']:
                lines.append(f"| {d['model']} | {d['data_seed']} | {method} | {a['value']['mean']:.6f} | {a['value']['se']:.6f} | {vector([x['mean'] for x in a['score']])} | {vector([x['se'] for x in a['score']])} |")
        for name,r in d['reference'].items():
            lines.append(f"| {d['model']} | {d['data_seed']} | {name} | {r['value']:.6f} | {scalar(r.get('value_se'))} | {vector(r['score'])} | {vector(r.get('score_se'))} |")
    return lines


def reference_diagnostics_markdown(result):
    lines=['','## Reference stability','',
        'Two author fits cannot bound shared TT rank or support bias. Within-fit path jackknife uncertainty and between-fit uncertainty answer different questions. '
        'The table displays both regression radii rather than selecting the radius after seeing agreement. '
        'The smaller radius supplies the displayed pooled reference. ESS and regression residuals are explanatory diagnostics, not evidence of score correctness.', '',
        '| Model | Data seed | Fit seed | Radius | Likelihood | Path ESS | Maximum weight | Heldout max residual | Score vector | Conditional score SE |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---|---|']
    for d in result['datasets']:
        for fit in d['reference'].get('zhao_rank20',{}).get('fits',[]):
            for e in fit['estimates']:
                p=fit['provenance']
                lines.append(f"| {d['model']} | {d['data_seed']} | {fit['fit_seed']} | {e['radius']} | {fit['value']:.6f} | {p['ess']:.1f} | {p['max_weight']:.6g} | {e['heldout_residual_max']:.6g} | {vector(e['score'])} | {vector(e['conditional_path_jackknife'].get('standard_error'))} |")
    lines+=['','| Model | Data seed | Zhao minus bootstrap log likelihood | Zhao minus bootstrap score vector |',
            '|---|---:|---:|---|']
    for d in result['datasets']:
        r=d['reference_difference']
        if r is not None:
            lines.append(f"| {d['model']} | {d['data_seed']} | {r['zhao_minus_bootstrap_value']:.6f} | {vector(r['zhao_minus_bootstrap_score'])} |")
    lines+=['','## Variation across the two datasets','',
        'These intervals use only two independent dataset-level error differences (df=1). '
        'They exclude reference bias and should not be read as broad cross-regime evidence.', '',
        '| Model | Reference | Error | Mean new minus old | Dataset-level 95% interval |',
        '|---|---|---|---:|---|']
    for r in result['across_datasets']:
        if r['reference'] not in ('model_reference','bootstrap_N131072','zhao_rank20'): continue
        s=r['dataset_level'];bounds=s['interval95']
        lines.append(f"| {r['model']} | {r['reference']} | {r['metric']} | {s['mean']:.6f} | [{bounds[0]:.6f}, {bounds[1]:.6f}] |")
    return lines


if __name__=='__main__': raise SystemExit(report())
