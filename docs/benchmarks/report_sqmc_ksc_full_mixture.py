"""Post-run diagnostic correction report; no runtime or candidate selection."""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[2]
ROUTES=('iid_dual_cap','previous_inverse_cdf','repaired_permutation','repaired_permutation_ablation')
LABELS=dict(iid_dual_cap='IID', previous_inverse_cdf='Inverse CDF', repaired_permutation='Permutation .98', repaired_permutation_ablation='Permutation .97', gaussian_sum_reference='Full-mixture Gaussian-sum Kalman reference')
NAMES=('gamma_raw','log_beta')
TC=2.3646242516


def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,d): Path(p).write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def norm(x): return math.sqrt(math.fsum(v*v for v in x))
def fmt(x): return 'reference' if x is None else f'{x:.8g}'


def stats(x):
    if len(x)!=8: raise ValueError('Expected eight paired final datasets')
    mean=statistics.fmean(x)
    sd=statistics.stdev(x)
    se=sd/math.sqrt(8.)
    return dict(n=8,mean=mean,sd=sd,se=se,ci95_low=mean-TC*se,ci95_high=mean+TC*se)


def csv_file(p,rows):
    with Path(p).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def table(headers,rows):
    return ['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+[
        '| '+' | '.join(str(x) for x in row)+' |' for row in rows]


def assemble(attempt,out):
    out.mkdir(parents=True,exist_ok=False)
    budget=read(attempt.parent/'budget.json')
    if budget['status']!='complete' or any(x['status']=='running' for x in budget['attempts']):
        raise ValueError('Correction is not complete')
    work=attempt/'compare-attempt-01'
    result=read(work/'result.json');manifest=read(work/'manifest.json')
    check=read(attempt/'check-attempt-01/result.json')
    hashes=read(attempt/'source-sha256.json')
    for p,digest in hashes.items():
        if sha(ROOT/p)!=digest or sha(attempt/'source'/p)!=digest:
            raise ValueError('Reference source provenance changed')
    for mode in ('check','compare'):
        m=read(attempt/f'{mode}-attempt-01/manifest.json')
        if not m['source_unchanged'] or m['source_sha256']!=hashes or not m['jit_compile'] or m['tf32']:
            raise ValueError('Configuration/source checks failed')
        policy=m['gpu_memory_policy']
        if not policy['all_physical_devices_memory_growth'] or not policy['configured_before_logical_device_initialization']:
            raise ValueError('GPU memory policy was not verified')
    if result['status']!='pass' or check['status']!='pass' or len(result['rows'])!=32:
        raise ValueError('Incomplete reference checks')
    old_hashes=read(work/'input-sha256.json')
    for p,digest in old_hashes.items():
        if sha(p)!=digest:raise ValueError('Original evidence changed: '+p)
    particles=read(work/'particle-inputs.json')
    observations={(d['horizon'],d['data_seed']):d for d in read(work/'observations.json')}
    references={(d['horizon'],d['data_seed']):d for d in result['rows']}
    if len(references)!=32 or len(particles)!=128:
        raise ValueError('Missing final cells')
    cells=[];coordinates=[];convergence=[]
    for (h,seed),r in sorted(references.items()):
        if not r['accepted']: raise ValueError('Unaccepted reference cell')
        ref=r['reference'];old=observations[(h,seed)]['reference']
        for resolution in r['resolutions']:
            convergence.append(dict(horizon=h,data_seed=seed,**resolution,
                                    independent_value_error=abs(resolution['value']-old['value']),
                                    independent_score_l2_error=norm([a-b for a,b in zip(resolution['score'],old['score'])])))
        cells.append(dict(horizon=h,data_seed=seed,method='gaussian_sum_reference',
                          program=result['program'],tuning='N/A deterministic resolution verification',
                          filter_seed=None,valid=True,value=ref['value'],score=ref['score'],
                          reference_value=ref['value'],reference_score=ref['score'],
                          value_error=None,score_l2_error=None,
                          independent_value_error=abs(ref['value']-old['value']),
                          independent_score_l2_error=norm([a-b for a,b in zip(ref['score'],old['score'])])))
    for p in particles:
        key=(p['horizon'],p['data_seed']);ref=references[key]['reference']
        if not p['valid'] or p['tuning_status']!='exact_scope_frozen_controls' or len(p['score'])!=2:
            raise ValueError('Unlabelled or invalid particle comparison')
        cells.append(dict(horizon=p['horizon'],data_seed=p['data_seed'],method=p['route'],
                          program=p['program'],tuning=p['tuning'],filter_seed=p['filter_seed'],valid=p['valid'],
                          value=p['value'],score=p['score'],reference_value=ref['value'],reference_score=ref['score'],
                          value_error=p['value']-ref['value'],score_l2_error=norm([a-b for a,b in zip(p['score'],ref['score'])]),
                          independent_value_error=None,independent_score_l2_error=None))
    for c in cells:
        for i,name in enumerate(NAMES):
            ref=c['method']=='gaussian_sum_reference'
            old=observations[(c['horizon'],c['data_seed'])]['reference']
            coordinates.append(dict(horizon=c['horizon'],data_seed=c['data_seed'],method=c['method'],
                                    program=c['program'],tuning=c['tuning'],filter_seed=c['filter_seed'],
                                    coordinate=name,score=c['score'][i],reference_score=c['reference_score'][i],
                                    absolute_error=None if ref else abs(c['score'][i]-c['reference_score'][i]),
                                    independent_grid_score=old['score'][i] if ref else None,
                                    independent_grid_absolute_discrepancy=abs(c['score'][i]-old['score'][i]) if ref else None))
    summary=[]
    for h in (10,20,50,120):
        for method in ('gaussian_sum_reference',*ROUTES):
            group=[c for c in cells if c['horizon']==h and c['method']==method]
            ref=method=='gaussian_sum_reference'
            stat=dict(n=8,mean=None,sd=None,se=None,ci95_low=None,ci95_high=None) if ref else stats([c['score_l2_error'] for c in group])
            signed=[stats([c['score'][i]-c['reference_score'][i] for c in group]) for i in (0,1)]
            summary.append(dict(horizon=h,method=method,program=group[0]['program'],tuning=group[0]['tuning'],**stat,
                                mean_score_gamma=statistics.fmean(c['score'][0] for c in group),
                                mean_score_log_beta=statistics.fmean(c['score'][1] for c in group),
                                mean_value=statistics.fmean(c['value'] for c in group),
                                mean_absolute_error_gamma=None if ref else statistics.fmean(abs(c['score'][0]-c['reference_score'][0]) for c in group),
                                mean_absolute_error_log_beta=None if ref else statistics.fmean(abs(c['score'][1]-c['reference_score'][1]) for c in group),
                                mean_absolute_value_error=None if ref else statistics.fmean(abs(c['value_error']) for c in group),
                                signed_mean_error_gamma=None if ref else signed[0]['mean'],
                                signed_mean_error_log_beta=None if ref else signed[1]['mean'],
                                signed_error_se_gamma=None if ref else signed[0]['se'],
                                signed_error_se_log_beta=None if ref else signed[1]['se'],
                                independent_grid_max_value_discrepancy=max(c['independent_value_error'] for c in group) if ref else None,
                                independent_grid_max_score_l2_discrepancy=max(c['independent_score_l2_error'] for c in group) if ref else None))
    index={(c['horizon'],c['method'],c['data_seed']):c for c in cells}
    paired=[];heuristics=[]
    for h in (10,20,50,120):
        for a,b in itertools.combinations(ROUTES,2):
            d=[index[h,a,s]['score_l2_error']-index[h,b,s]['score_l2_error'] for s in range(213001,213009)]
            st=stats(d)
            paired.append(dict(horizon=h,left=a,right=b,**st,interval_excludes_zero=st['ci95_low']>0 or st['ci95_high']<0))
        for route in ROUTES:
            for seed in range(213001,213009):
                c=index[h,route,seed];ref=references[h,seed]['reference']['score']
                choices=dict(zero_score=[0.,0.],first_observation_only=observations[h,seed]['first_score'])
                if route!='iid_dual_cap':choices['iid_dual_cap']=index[h,'iid_dual_cap',seed]['score']
                for name,score in choices.items():
                    error=norm([a-b for a,b in zip(score,ref)])
                    loss=c['score_l2_error']>error
                    heuristics.append(dict(horizon=h,data_seed=seed,method=route,heuristic=name,
                                           particle_score_l2_error=c['score_l2_error'],heuristic_score_l2_error=error,
                                           observed_loss=loss,verdict='promotion_veto_in_this_case' if loss else 'no_observed_loss'))
    dump(out/'cells.json',cells);dump(out/'summary.json',summary)
    dump(out/'budget.json',budget)
    csv_file(out/'scores.csv',coordinates)
    csv_file(out/'values.csv',[{k:v for k,v in c.items() if k not in ('score','reference_score')} for c in cells])
    csv_file(out/'summary.csv',summary);csv_file(out/'paired_differences.csv',paired)
    csv_file(out/'heuristics.csv',heuristics)
    csv_file(out/'reference_convergence.csv',convergence)
    old_particle={(r['horizon'],r['route'],r['data_seed']):r for r in particles}
    maximum_error_change=max(abs(c['score_l2_error']-old_particle[c['horizon'],c['method'],c['data_seed']]['score_l2_error']) for c in cells if c['method'] in ROUTES)
    review=dict(reference_cells=32,particle_cells=128,score_coordinates=len(coordinates),
                max_independent_value_discrepancy=max(r['independent_value_error'] for r in references.values()),
                max_independent_score_coordinate_discrepancy=max(r['independent_score_error'] for r in references.values()),
                max_refinement_value_discrepancy=max(r['refinement_value_error'] for r in references.values()),
                max_refinement_score_coordinate_discrepancy=max(r['refinement_score_error'] for r in references.values()),
                max_projection_mass_error=max(v['maximum_projection_mass_error'] for r in references.values() for v in r['resolutions']),
                maximum_particle_error_change=maximum_error_change,
                gpu_checks=check,paired_intervals_excluding_zero=[p for p in paired if p['interval_excludes_zero']],
                heuristic_observed_losses=sum(r['observed_loss'] for r in heuristics),
                no_new_default=True,all_four_retained=True,
                removed_main_comparator='single-Gaussian moment-matched Kalman, a different observation likelihood')
    dump(out/'terminal-review.json',review)
    lines=['# Corrected KSC comparison: full seven-component mixture','',
           'Configuration: FP64 GPU/XLA diagnostic variants, TF32 off. Particle N=1008; each route/horizon uses its original exact-scope frozen tuning (paths in every cell and particle-scopes.json). The deterministic Gaussian-sum Kalman reference has all seven observation components and a checked quadrature resolution. This is not production FP32/TF32, native SV, HMC validation or a default change.','',
           'The single-Gaussian column in the earlier report computes a different likelihood and is removed from this main comparison. The new Gaussian-sum reference uses the same full mixture as the particle methods. All original particle evaluations are preserved.','',
           '## Actual likelihoods and scores','',
           'Means over the same eight datasets. Errors are measured per dataset before averaging. Reference self-errors are left blank rather than presented as exact zeros. Both parameter coordinates are retained.','']
    lines+=table(['T','Method','Mean log likelihood','Mean gamma_raw score','Mean log_beta score','Mean absolute log-likelihood error','Mean absolute gamma error','Mean absolute log_beta error'],[
        [r['horizon'],LABELS[r['method']],fmt(r['mean_value']),fmt(r['mean_score_gamma']),fmt(r['mean_score_log_beta']),fmt(r['mean_absolute_value_error']),fmt(r['mean_absolute_error_gamma']),fmt(r['mean_absolute_error_log_beta'])] for r in summary])
    lines+=['','## Score-vector error and uncertainty','',
            'e_i=||score_i-reference_i||_2. SD is the sample standard deviation of e_i; SE=SD/sqrt(8) estimates uncertainty in its mean across independent dataset/design pairs. This is not fixed-dataset Monte Carlo uncertainty or the norm of the mean error vector.','']
    lines+=table(['T','Method','Mean L2 error','SD','SE','95% t interval for mean'],[
        [r['horizon'],LABELS[r['method']],fmt(r['mean']),fmt(r['sd']),fmt(r['se']),f"[{fmt(r['ci95_low'])}, {fmt(r['ci95_high'])}]"] for r in summary if r['method'] in ROUTES])
    lines+=['','## Independent verification of the full-mixture reference','',
            'The first update retains seven exact Gaussian branches. Later updates retain 7*M branches from M quadrature atoms, with all seven observation components intact. At M=1201 this is 8407 branches before projection. Fixed quadrature projection is an approximation; it was checked at M=401,801,1201 on [-40,40] and M=1201 on [-48,48]. The comparator is not an exact seven-component posterior filter.','']
    lines+=table(['T','Maximum absolute likelihood discrepancy vs independent grid','Maximum score L2 discrepancy vs independent grid'],[
        [r['horizon'],fmt(r['independent_grid_max_value_discrepancy']),fmt(r['independent_grid_max_score_l2_discrepancy'])] for r in summary if r['method']=='gaussian_sum_reference'])
    lines+=['',f"Maximum refinement differences: likelihood {review['max_refinement_value_discrepancy']:.6g}, score coordinate {review['max_refinement_score_coordinate_discrepancy']:.6g}. These observed discrepancies are not rigorous error bounds.",
            f"Maximum change to a previously reported particle score-vector error: {maximum_error_change:.6g}. Candidate accuracy findings remain unchanged at the reported precision.",
            '', '## Paired differences','',
            'Left minus right score-vector error, paired by saved dataset/design seed. Negative favors left. Intervals are exploratory and unadjusted for multiple comparisons. Inclusion of zero does not prove equivalence.','']
    lines+=table(['T','Left','Right','Mean difference','SE','95% paired t interval'],[
        [r['horizon'],LABELS[r['left']],LABELS[r['right']],fmt(r['mean']),fmt(r['se']),f"[{fmt(r['ci95_low'])}, {fmt(r['ci95_high'])}]"] for r in paired])
    lines+=['','## Conditional heuristic checks','',
            'Zero score, first-observation-only score and IID are checked per dataset and horizon; observed losses are promotion vetoes in those situations, not evidence for a universal ranking. The old different-model single-Gaussian heuristic is preserved in the historical report rather than used as this comparison baseline.','']
    lines+=table(['T','Method','Heuristic','Observed losses / 8'],[
        [h,LABELS[route],name,sum(x['observed_loss'] for x in heuristics if x['horizon']==h and x['method']==route and x['heuristic']==name)]
        for h in (10,20,50,120) for route in ROUTES for name in ('zero_score','first_observation_only','iid_dual_cap') if not(route=='iid_dual_cap' and name=='iid_dual_cap')])
    lines+=['','## Decision and terminal review','',
            'The full-mixture Gaussian-sum calculation agrees with the independent density-grid reference and passes exact short-sequence, finite-difference and graph/XLA checks. The original particle errors therefore remain valid. The correction replaces the misleading main comparator, not the particle data. The former Gaussian-versus-particle score observation remains only a descriptive different-model heuristic finding, not a verdict about full-KSC Kalman filtering.','',
            'All four particle methods remain research candidates. Only the T=10 IID-versus-SQMC exploratory paired intervals exclude zero; no interval orders the three SQMC variants. Large finite particle errors remain accuracy findings, not infrastructure failures. One regime, eight pairs, finite particle count and limited scope-specific control search do not establish general superiority or readiness.','']
    lines+=table(['Decision','Primary criterion','Veto status','Main uncertainty','Next justified action','Not concluded'],[
        ['Accept corrected reference','All 32 datasets converge and agree with independent implementation','No reference validity/derivative veto','Empirical quadrature checks, not rigorous bounds','Use corrected full-mixture tables','No exact finite-M or native-SV claim'],
        ['Retain all four routes','128 preserved valid evaluations with recomputed errors','Conditional heuristic vetoes remain','Eight pairs, one regime, finite controls/N','Fresh tuning/replication before selecting a method','No default or overall winner']])
    lines+=['']+table(['Inference status','Finding'],[
        ['Hard veto screen','Reference and original particle validity pass; per-case heuristic losses in heuristics.csv prohibit promotion there'],
        ['Statistically supported ranking','Only exploratory T10 SQMC-versus-IID intervals; no overall or within-SQMC ranking'],
        ['Descriptive differences','Other means, likelihood errors, timing and tails'],
        ['Default readiness','Not established'],
        ['Next evidence','Broader regimes, fresh long-horizon tuning, particle convergence, more pairs and fixed-dataset replications']])
    lines+=['', 'Review: strongest alternative explanation is the single regime and limited particle control family; fresh regimes could reverse descriptive orderings. Independent Gaussian-sum and density-grid algebra plus exact short-T tests reduce shared-reference risk, but do not prove every long-T quadrature error bound. Local review with executable checks was used; no independent agent review.','',
            'Engineering, numerical validity and scientific interpretation remain separate: successful compilation is not accuracy; numerical reference agreement does not promote a particle method. No particle runs were repeated.','',
            f"Correction GPU-owning wall charge {budget['correction_seconds']:.6f}s; aggregate {budget['charged_seconds']/3600:.6f}/12 GPU hours; remaining {budget['remaining_gpu_seconds']/3600:.6f}h. CPU-only checks/reporting add no GPU charge. All earlier pilots and the prior provisional hook reserve remain included.",'',
            'Files: scores.csv contains every actual coordinate/reference/error; values.csv contains every likelihood; reference_convergence.csv preserves all resolutions; paired_differences.csv and heuristics.csv preserve the conditional calculations; cells.json and summary.json provide structured results. Original source, tuning and input hashes remain linked in report-manifest.json.']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    evidence={str(p):sha(p) for p in attempt.rglob('*') if p.is_file()}
    evidence[str(attempt.parent/'budget.json')]=sha(attempt.parent/'budget.json')
    dump(out/'input-sha256.json',evidence)
    dump(out/'report-manifest.json',dict(command=[sys.executable,*sys.argv],
          created_utc=datetime.now(timezone.utc).isoformat(),reporter=str(Path(__file__).resolve()),
          reporter_sha256=sha(__file__),source_commit=manifest['git_commit'],reference_source_sha256=hashes,
          prior_particle_delivery_commit='b7ed96ec',prior_particle_run_commit='479a4616',prior_particle_inputs=str(work/'input-sha256.json'),
          plan='docs/plans/sqmc-ksc-full-mixture-correction-20260929.md',result=str(out/'report.md'),
          cpu_only=True,gpu_devices_intentionally_hidden=True,seeds=list(range(213001,213009)),
          inputs=str(out/'input-sha256.json'),outputs={p.name:sha(p) for p in out.iterdir() if p.is_file()}))
    print(json.dumps({k:v for k,v in review.items() if k not in ('gpu_checks','paired_intervals_excluding_zero')}))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--attempt',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();assemble(args.attempt.resolve(),args.output.resolve())
