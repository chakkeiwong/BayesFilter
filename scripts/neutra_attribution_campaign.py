"""Bounded attribution queue using the existing locked campaign accountant."""
from __future__ import annotations

import json
import hashlib
import math
from pathlib import Path
import time

from bayesfilter.testing.neutra_scientific_design import target_catalog
from bayesfilter.testing.neutra_target_specifications import random_mixture_specification

PILOT_SEEDS=(409,419)
CONFIRM_SEEDS=(431,433,439,443,449,457,461,463)
CONDITIONERS=('author_cmade','diagnostic_hoffman_made')
TARGETS=('calibration_random_two','calibration_random_three')


def read(path):return json.loads(Path(path).read_text())


def write(path,value):
    path=Path(path)
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def sign_summary(results):
    positive=sum(r['paired_kl_benefit']['normal_approximation_99_lower']>0 for r in results)
    n=len(results)
    p=sum(math.comb(n,k) for k in range(positive,n+1))/2**n if n else 1.
    effects=[r['paired_kl_benefit']['mean'] for r in results]
    screens={arm:sum(bool(r.get('endpoints',{}).get(arm,{}).get('passed')) for r in results)
             for arm in ('affine','nonlinear')}
    return dict(pairs=n,positive_conservative_signs=positive,one_sided_sign_p=p,
        bonferroni_four_p=min(1.,4*p),median_effect=(sorted(effects)[(n-1)//2]+sorted(effects)[n//2])/2 if n else None,
        descriptive_mean_effect=sum(effects)/n if n else None,
        observed_effect_range=[min(effects),max(effects)] if effects else None,
        distribution_screen_pass_counts=screens,
        passed=n==8 and 4*p<.05,
        interpretation='replicated median log-density benefit only; reference noise screened; '
                       'shape-screen failures veto map promotion independently')


def checked_result(row):
    output=Path(row['output'])
    manifest=read(output/'manifest.json')
    if manifest['status']!='complete' or manifest['source']!=row['source']+'/source.json':
        raise RuntimeError('completed attribution manifest mismatch')
    for name,digest in manifest['artifact_sha256'].items():
        if hashlib.sha256((output/name).read_bytes()).hexdigest()!=digest:
            raise RuntimeError('attribution artifact differs: '+str(output/name))
    source=Path(row['source'])
    for name,digest in read(source/'source.json')['sha256'].items():
        if hashlib.sha256((source/name).read_bytes()).hexdigest()!=digest:
            raise RuntimeError('attribution source differs: '+str(source/name))
    result=read(output/'result.json')
    for endpoint in result.get('endpoints',{}).values():
        for block in endpoint.get('training',[]):
            if (not block['finite'] or 'GPU' not in block['device'] or
                    not block['jit_compile'] or block['samplewise_loop'] or block['batch_size']<=1):
                raise RuntimeError('invalid attribution execution path')
        if endpoint.get('probe_path'):
            probe=read(endpoint['probe_path'])
            if not probe['complete'] or not probe['finite'] or probe['valid_rows']!=1000:
                raise RuntimeError('invalid saved attribution score probe')
    return result


def write_report(root,state,comparisons,selection):
    """Operational result note, preserving scientific scope and failed fits."""
    records={r['job']:checked_result(r) for r in state['completed']}
    lines=['# Scalar nonlinearity attribution results','',
        'Status: '+('numerical queue complete; terminal interpretation below.' if
                   (root/'attribution-result.json').exists() else 'provisional; the numerical queue is still running.'),'',
        'This is an exact-teacher, FP64 GPU/XLA density-fitting experiment. '
        'It does not change the canonical IAF or establish HMC/q20 readiness.','',
        '## Mathematical and implementation controls','']
    for label in ('gaussian','theorem'):
        key=f'attribution-control-{label}-author_cmade-s401'
        if key not in records:continue
        r=records[key]
        lines.append(f"{label}: "+'; '.join(f"{a} KL={e['forward_kl_reference']['mean']:.6g}, "
            f"99% conditional MC interval [{e['forward_kl_reference']['normal_approximation_99_lower']:.6g}, "
            f"{e['forward_kl_reference']['normal_approximation_99_upper']:.6g}], shape screen {e['passed']}"
            for a,e in r['endpoints'].items())+'.')
        if label=='theorem':
            lines.append(f"The analytic Gaussian lower bound is {r['affine_kl_lower_bound']:.10f} nats; "
                f"the learned nonlinear fit's upper MC limit is below it: {r['nonlinear_below_proven_bound']}.")
        lines.append('')
    lines+=['## Development pilots (descriptive only)','',
        '| Pair | Nonlinear KL benefit | Affine / nonlinear shape screens |',
        '|---|---:|---|']
    for name,r in records.items():
        if name.startswith('attribution-pilot-'):
            lines.append(f"| {name} | {r['paired_kl_benefit']['mean']:.6g} | "
                         f"{r['endpoints']['affine']['passed']} / {r['endpoints']['nonlinear']['passed']} |")
    lines.append('')
    lines+=['## Independent fixed-recipe confirmation','',
        '| Target / conditioner | Positive pairs | Mean KL benefit (descriptive) | Shape passes A / N | Adjusted sign p | Density effect supported |',
        '|---|---:|---:|---|---:|---|']
    for name,r in comparisons.items():
        lines.append(f"| {name} | {r['positive_conservative_signs']}/{r['pairs']} | "
            f"{r['descriptive_mean_effect']:.6g} | {r.get('distribution_screen_pass_counts')} | "
            f"{r['bonferroni_four_p']:.6g} | {r['passed']} |")
    lines+=['','The primary sign test concerns repeated paired density differences under '
        'the declared recipe. Reference-row precision is distinct from variation across '
        'training seeds. Four target/conditioner contrasts share familywise alpha .05.','',
        '## Capacity and optimizer controls','',
        '| Job | Arm | Validation-selected rate | Reference KL | Shape screen |',
        '|---|---|---:|---:|---|']
    for name,r in records.items():
        if not name.startswith(('attribution-capacity-','attribution-rate-')):continue
        for arm,e in r['endpoints'].items():
            lines.append(f"| {name} | {arm} | see selection JSON | "
                f"{e['forward_kl_reference']['mean']:.6g} | {e['passed']} |")
    lines+=['','The two pilot seeds and reused validation data make these sensitivity '
        'results descriptive. Rate selection uses validation cross entropy only; '
        'it does not add confirmation replications.','',
        '## Longer affine training and transfer','']
    for name,r in records.items():
        if name.startswith('attribution-time-'):
            lines.append(f"- {name}: {r['total_updates']} total updates, training-time ratio "
                f"{r['achieved_time_ratio']:.3f}, nonlinear-minus-affine log-density benefit "
                f"{r['paired_kl_benefit']['mean']:.6g}; stop={r['stopped']}.")
        elif name.startswith('attribution-transfer-'):
            lines.append(f"- {name}: paired benefit {r['paired_kl_benefit']['mean']:.6g}; "
                f"affine/nonlinear shape screens {r['endpoints']['affine']['passed']}/"
                f"{r['endpoints']['nonlinear']['passed']}.")
    lines+=['','## Decisions and inference status','',
        '| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |',
        '|---|---|---|---|---|---|',
        '| Scalar mechanism | Paired confirmation table above | Preserve all shape/numerical rejections | Conditioner and optimization interactions | Interpret capacity/longer-training controls | Universal IAF impossibility in multiple dimensions |',
        '| Transfer | Four new target geometries if primary cMADE contrast passed | Existing shape checks | Two seeds per target | Independent downstream validation if justified | Generic teacher or q20 success |','',
        '| Inference status | Result |','|---|---|',
        '| Hard veto screen | See each endpoint; failed fits are retained |',
        '| Statistically supported ranking | Only the specified log-density contrasts; no overall map or sampler ranking |',
        '| Descriptive-only differences | KL means, tails, runtime, calibration and transfer controls |',
        '| Default readiness | Not established |',
        '| Next evidence needed | Native-teacher and downstream validation, with any observed interactions retained |','',
        'Post-run red team: exact teacher samples remove mode-discovery error but limit '
        'transfer to realistic teachers. The affine diagnostic has redundant aggregate '
        'parameterization, so canonical capacity/optimizer controls remain essential. '
        'A positive finite-recipe effect is not a theorem about the best multivariate '
        'affine autoregressive map. Narrow seed/target scope is the weakest evidence.','',
        'Artifacts: `attribution-result.json`, `attribution-confirmation-design.json`, '
        '`attribution-optimizer-selection.json` and every completed worker directory.']
    (root/'attribution-results.md').write_text('\n'.join(lines)+'\n')


def run_campaign(controller):
    root=Path(controller.state['attempts'][0]['output']).parent
    path=root/'attribution-state.json'
    state=read(path) if path.exists() else dict(status='prepared',completed=[],created_unix=time.time())
    catalog=target_catalog()
    def sync(status,next_action):
        state.update(status=status,next_action=next_action,remaining=controller.remaining(),updated_unix=time.time())
        write(path,state)
        controller.sync('attribution_'+status,next_action)
        write(root/'next-phase.json',dict(phase='attribution_'+status,next_action=next_action,
            remaining=controller.remaining(),resume_command='bash /home/ubuntu/python/BayesFilter/scripts/run_neutra_scientific_campaign.sh attribution'))
    def execute(job,spec,seed,profile):
        profile={**profile,'gpu_index':2}
        prior=next((r for r in state['completed'] if r['job']==job),None)
        if prior:
            # Prior frozen evidence remains valid across orchestration edits.
            return checked_result(prior)
        available=controller.remaining()
        requested_gpu=profile['worker_wall_limit']
        requested_cpu=profile.get('worker_cpu_limit',2*requested_gpu)
        if requested_gpu>available['gpu_process_seconds'] or requested_cpu>available['cpu_core_seconds']:
            sync('allocation_review','full worker reservation unavailable: '+job)
            raise RuntimeError('insufficient full attribution worker reservation')
        row=controller.execute(job,phase='nonlinearity_attribution',target=job,seed=seed,
            method='scalar_nonlinearity_attribution',role=profile.get('role','diagnostic'),
            specification=spec,profile=profile)
        if row['status']!='complete':
            sync('repair_required','inspect preserved failed worker '+job)
            raise RuntimeError('attribution worker failed: '+row['output'])
        state['completed'].append({k:row[k] for k in ('job','output','source','wall_seconds','cpu_core_seconds')})
        sync('running','completed '+job)
        return checked_result(row)
    if state['status']=='complete' and state.get('controls_complete'):
        for row in state['completed']:checked_result(row)
        print(json.dumps(read(root/'attribution-result.json'),indent=2))
        return 0
    if not state.get('checks_passed'):
        check=controller.execute('attribution-check',phase='preflight',device='cpu',profile={'worker_wall_limit':240.})
        if check['status']!='complete':
            sync('repair_required','repair focused checks before experiments')
            return 1
        state['checks_passed']=True
        sync('checked','price paired scalar intervention')
    if not state.get('pricing'):
        price=execute('attribution-price',catalog[TARGETS[1]],397,
            dict(conditioner='author_cmade',updates=2048,pricing_only=True,worker_wall_limit=600.,role='pricing'))
        # The first block includes tracing. Use both setup and warmed block cost.
        row=state['completed'][-1]
        endpoints=price['endpoints']
        steady=sum(e['training'][-1]['wall_seconds']/e['training'][-1]['updates'] for e in endpoints.values())
        limit=math.ceil(2*row['wall_seconds']+16384*steady)
        state['pricing']=dict(pair_wall_limit=limit,price_wall=row['wall_seconds'],
            warmed_seconds_per_paired_update=steady,forecast='two pricing processes plus full warmed update allocation')
        sync('priced','run full one-dimensional controls and independent pilots')
    limit=state['pricing']['pair_wall_limit']
    def pair(stage,label,spec,conditioner,seed):
        return execute(f'attribution-{stage}-{label}-{conditioner}-s{seed}',spec,seed,
            dict(conditioner=conditioner,updates=16384,worker_wall_limit=limit,role=stage))
    controls={
        'gaussian':dict(kind='gaussian',mean=[0.],covariance=[[1.]]),
        'theorem':dict(kind='isotropic_mixture',dimension=1,centers=[[-3.],[3.]],
            variances=[1.,1.],weights=[.5,.5],theorem_control=True)}
    for label,spec in controls.items():
        result=pair('control',label,spec,'author_cmade',401)
        if label=='gaussian' and not all(e['passed'] for e in result['endpoints'].values()):
            sync('control_failed','inspect Gaussian control before architectural interpretation')
            return 1
    for target in TARGETS:
        for conditioner in CONDITIONERS:
            for seed in PILOT_SEEDS:pair('pilot',target,catalog[target],conditioner,seed)
    if not state.get('confirmation_design'):
        # Replace the short pricing extrapolation with full pilot measurements.
        # Keep hard limits distinct from the expected queue cost.
        forecasts={}
        for conditioner in CONDITIONERS:
            rows=[r for r in state['completed'] if r['job'].startswith('attribution-pilot-')
                  and conditioner in r['job']]
            forecasts[conditioner]=dict(worker_wall_limit=math.ceil(1.25*max(r['wall_seconds'] for r in rows)),
                worker_cpu_limit=math.ceil(1.25*max(r['cpu_core_seconds'] for r in rows)))
        remaining=controller.remaining()
        reserved=len(CONFIRM_SEEDS)*len(TARGETS)*sum(p['worker_wall_limit'] for p in forecasts.values())
        cpu_reserved=len(CONFIRM_SEEDS)*len(TARGETS)*sum(p['worker_cpu_limit'] for p in forecasts.values())
        # Full confirmation must fit both resource ledgers before launch.
        if reserved>remaining['gpu_process_seconds'] or cpu_reserved>remaining['cpu_core_seconds']:
            state['proposed_reservation']=dict(gpu=reserved,cpu=cpu_reserved,forecasts=forecasts)
            sync('allocation_review','measured confirmation reservation exceeds remaining allocation')
            return 2
        state['confirmation_design']=dict(seeds=CONFIRM_SEEDS,targets=TARGETS,
            conditioners=CONDITIONERS,familywise_alpha=.05,primary_test='exact_one_sided_sign_bonferroni4',
            gpu_reservation=reserved,cpu_reservation=cpu_reserved,forecasts=forecasts,created_unix=time.time())
        write(root/'attribution-confirmation-design.json',state['confirmation_design'])
        sync('confirmation_frozen','run the predeclared paired confirmation')
    comparisons={}
    confirmation_rows={}
    for target in TARGETS:
        for conditioner in CONDITIONERS:
            forecast=state['confirmation_design']['forecasts'][conditioner]
            rows=[execute(f'attribution-confirm-{target}-{conditioner}-s{seed}',catalog[target],seed,
                dict(conditioner=conditioner,updates=16384,role='confirm',**forecast)) for seed in CONFIRM_SEEDS]
            comparisons[target+'/'+conditioner]=sign_summary(rows)
            confirmation_rows[target+'/'+conditioner]=rows
            state['comparisons']=comparisons
            sync('confirming','completed contrast '+target+'/'+conditioner)
    # Preserve both rates on all paired conditioner arms before interpreting
    # conditioner-independent superiority. Confirmation remains the fixed recipe.
    for target in TARGETS:
        for conditioner in CONDITIONERS:
            for seed in PILOT_SEEDS:
                execute(f'attribution-rate-{target}-{conditioner}-lr0.001-s{seed}',catalog[target],seed,
                    dict(conditioner=conditioner,final_rate=.001,updates=16384,
                        worker_wall_limit=limit,role='optimizer_control'))
    # Canonical capacity/rate controls use independent pilot streams.
    for target in TARGETS:
        for kind,width in (('iaf',64),('iaf',134),('iaf_uncapped',134)):
            for rate in (.001,.0003):
                for seed in PILOT_SEEDS:
                    execute(f'attribution-capacity-{target}-{kind}{width}-lr{rate}-s{seed}',catalog[target],seed,
                        dict(conditioner=kind,width=width,final_rate=rate,updates=16384,
                            worker_wall_limit=limit,role='capacity_optimizer_control'))
    for target in TARGETS:
        for seed in PILOT_SEEDS:
            name=f'attribution-pilot-{target}-author_cmade-s{seed}'
            parent=next(r for r in state['completed'] if r['job']==name)
            execute(f'attribution-time-{target}-s{seed}',catalog[target],seed,
                dict(action='equal_compute',parent=parent['output'],worker_wall_limit=limit,
                    role='equal_training_time_control'))
    selection={}
    for target in TARGETS:
        for conditioner in CONDITIONERS:
            for arm in ('affine','nonlinear'):
                scores={}
                for rate,stage in ((.0003,'pilot'),(.001,'rate')):
                    entries=[]
                    for seed in PILOT_SEEDS:
                        name=(f'attribution-pilot-{target}-{conditioner}-s{seed}' if stage=='pilot' else
                              f'attribution-rate-{target}-{conditioner}-lr0.001-s{seed}')
                        saved=next(r for r in state['completed'] if r['job']==name)
                        entries.append(checked_result(saved)['endpoints'][arm]['heldout']['heldout_cross_entropy'])
                    scores[str(rate)]=sum(entries)/len(entries)
                selection[target+'/'+conditioner+'/'+arm]=dict(
                    validation_scores=scores,selected_rate=float(min(scores,key=scores.get)),
                    rule='minimum mean pilot validation cross entropy; descriptive calibration only')
        for kind,width in (('iaf',64),('iaf',134),('iaf_uncapped',134)):
            scores={}
            for rate in (.001,.0003):
                entries=[]
                for seed in PILOT_SEEDS:
                    name=f'attribution-capacity-{target}-{kind}{width}-lr{rate}-s{seed}'
                    saved=next(r for r in state['completed'] if r['job']==name)
                    entries.append(checked_result(saved)['endpoints']['canonical']['heldout']['heldout_cross_entropy'])
                scores[str(rate)]=sum(entries)/len(entries)
            selection[target+'/'+kind+str(width)]=dict(validation_scores=scores,
                selected_rate=float(min(scores,key=scores.get)),
                rule='minimum mean pilot validation cross entropy; descriptive calibration only')
    write(root/'attribution-optimizer-selection.json',selection)
    interactions={}
    for target in TARGETS:
        effects=[a['paired_kl_benefit']['mean']-b['paired_kl_benefit']['mean']
            for a,b in zip(confirmation_rows[target+'/author_cmade'],
                           confirmation_rows[target+'/diagnostic_hoffman_made'])]
        interactions[target]=dict(paired_effect_differences=effects,
            descriptive_mean=sum(effects)/len(effects),
            interpretation='descriptive conditioner interaction; not an additional confirmatory test')
    if all(comparisons[target+'/author_cmade']['passed'] for target in TARGETS):
        for modes,target_seed in ((2,3101),(2,3102),(3,3201),(3,3202)):
            spec=random_mixture_specification(modes,target_seed)
            for seed in (479,487):
                pair('transfer',f'random{modes}-{target_seed}',spec,'author_cmade',seed)
    for row in state['completed']:checked_result(row)
    result=dict(status='attribution_complete',comparisons=comparisons,
        optimizer_calibration=selection,
        conditioner_interactions=interactions,
        completed=state['completed'],remaining=controller.remaining(),scientific_promotion=False,
        scope='exact-teacher diagnostic density attribution; no HMC, default or q20 claim',
        limitations=['finite optimizer grid','capacity control nominal counts differ',
                    'optimizer/capacity/transfer controls have two fitting seeds and are descriptive',
                    'no distribution-wide necessity theorem'])
    write(root/'attribution-result.json',result)
    write_report(root,state,comparisons,selection)
    state['controls_complete']=True
    sync('complete','review results and unresolved attribution limits')
    print(json.dumps(result,indent=2))
    return 0
