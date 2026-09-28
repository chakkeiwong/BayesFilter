"""Post-run diagnostic reporting only; never a candidate or selection path."""
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

ROUTES=('iid_dual_cap','previous_inverse_cdf','repaired_permutation','repaired_permutation_ablation')
LABELS=dict(iid_dual_cap='IID',previous_inverse_cdf='Inverse CDF',repaired_permutation='Permutation .98',repaired_permutation_ablation='Permutation .97',mixture_reference='Mixture reference',gaussian_kalman='Gaussian Kalman approximation')
FINAL=set(range(213001,213009))
T_CRITICAL={1:12.7062047364,2:4.3026527297,3:3.1824463053,4:2.7764451052,5:2.5705818356,6:2.4469118511,7:2.3646242516}


def read(path):return json.loads(path.read_text())
def dump(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def norm(v):return math.sqrt(sum(x*x for x in v))
def close(a,b):return math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-10)
def fmt(x):return '—' if x is None else f'{x:.6g}'
def csv_file(path,rows):
    if not rows:return
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
def stats(values):
    n=len(values)
    if not n:return dict(n=0,mean=None,sd=None,se=None,ci95_low=None,ci95_high=None)
    mean=statistics.fmean(values)
    sd=statistics.stdev(values) if n>1 else None
    se=sd/math.sqrt(n) if sd is not None else None
    radius=T_CRITICAL[n-1]*se if se is not None else None
    return dict(n=n,mean=mean,sd=sd,se=se,ci95_low=mean-radius if radius is not None else None,ci95_high=mean+radius if radius is not None else None)


def assemble(campaign,out):
    out.mkdir(parents=True,exist_ok=False)
    ledger=read(campaign/'budget.json')
    if any(a['status']=='running' for a in ledger['attempts']):raise ValueError('Do not publish a terminal report while a worker is running')
    entries={}
    failures=[]
    evidence={}
    for attempt in ledger['attempts']:
        if attempt['unit']=='check':continue
        if attempt['returncode']!=0:failures.append(attempt);continue
        key=attempt['unit']
        if key in entries:raise ValueError('Duplicate completed scope; requires explicit evidence resolution')
        entries[key]=attempt
    observations={}
    final_rows={}
    coordinates=[]
    values=[]
    scopes=[]
    safety=[]
    convergence=[]
    snapshots=set()
    for unit,attempt in entries.items():
        directory=Path(attempt['output']);result=read(directory/'result.json');manifest=read(directory/'manifest.json')
        if not result.get('program') or not result.get('tuning_status'):raise ValueError('Unlabelled result quarantined')
        if not manifest.get('source_unchanged'):raise ValueError('Worker source provenance incomplete')
        assert result['program']=='fp64_gpu_xla_ksc_seven_mixture'
        assert result['particle_count']==1008
        assert manifest['jit_compile'] is True and manifest['tf32'] is False
        policy=manifest['gpu_memory_policy']
        assert policy['all_physical_devices_memory_growth'] and policy['configured_before_logical_device_initialization']
        assert 'GPU:0' in manifest['framework_gpu_probe']['device']
        for filename,key in (('data.json','data_sha256'),('designs.json','design_sha256')):
            assert hashlib.sha256((directory/filename).read_bytes()).hexdigest()==manifest[key]
        snapshots.add(Path(attempt['source_snapshot']))
        for path in directory.rglob('*'):
            if path.is_file():evidence[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        h=result['horizon'];route=result['route'];data=read(directory/'data.json')
        for seed,datum in data.items():
            reference=datum['reference']
            assert reference['maximum_value_difference']<=1e-6 and reference['maximum_score_difference']<=1e-5
            convergence.append(dict(horizon=h,route=route,seed=int(seed),value_difference=reference['maximum_value_difference'],score_difference=reference['maximum_score_difference']))
            key=(h,int(seed))
            if key in observations:
                previous=observations[key]
                assert datum['observations']==previous['observations']
                assert close(reference['value'],previous['reference']['value'])
                assert all(close(a,b) for a,b in zip(reference['score'],previous['reference']['score']))
            else:observations[key]=datum
        scope=dict(horizon=h,route=route,program=result['program'],tuning_status=result['tuning_status'],tuning_path=result['tuning_path'],status=result['status'],final_count=len(result['rows']),valid_count=sum(r['valid'] for r in result['rows']),selected_controls=result.get('selected_controls'),path=str(directory))
        scopes.append(scope)
        if result['rows']:
            assert {r['data_seed'] for r in result['rows']}==FINAL
            assert result['tuning_status']=='exact_scope_frozen_controls'
            tuning=read(directory/'tuning.json')['tuning_artifact']
            ts=tuning['scope']
            assert (ts['horizon'],ts['route'],ts['particle_count'],ts['parameter_count'])==(h,route,1008,2)
            assert ts['chunk_size']==1008 and ts['dtype']=='float64' and ts['tf32'] is False
            assert ts['theta']==[1.5,0.] and ts['state_dimension']==1
            assert ts['reset_contract']=='contract_e_chol_v1'
            assert ts['prepared_data_regime']=='ksc_seven_mixture_predict_first_independent_pairs_v1'
            assert tuning['controls']==result['selected_controls']
            assert set(tuning['claim_seeds'])==FINAL
            assert set(tuning['calibration_seeds']).isdisjoint(tuning['validation_seeds'])
            for group in ('calibration_seeds','validation_seeds'):
                assert set(tuning[group]).isdisjoint(FINAL)
        for row in result['rows']:
            seed=row['data_seed'];key=(h,route,seed)
            assert key not in final_rows
            assert row['filter_seed']==seed+1000
            final_rows[key]=row
            reference=observations[(h,seed)]['reference']
            assert all(close(a,b) for a,b in zip(row['oracle_score'],reference['score']))
            assert row['score_method']=='analytical_recursive' and row['jit_compile'] is True
            assert row['tf32'] is False and 'GPU:0' in row['compute_device']
            if row['valid']:
                assert len(row['score'])==len(row['oracle_score'])==len(row['component_absolute_errors'])==2
                assert all(math.isfinite(x) for x in [row['value'],*row['score'],row['oracle_value'],*row['oracle_score']])
                error=norm([a-b for a,b in zip(row['score'],row['oracle_score'])])
                assert close(error,row['score_l2_error'])
                assert all(close(abs(a-b),e) for a,b,e in zip(row['score'],row['oracle_score'],row['component_absolute_errors']))
            values.append(dict(horizon=h,data_seed=seed,filter_seed=row['filter_seed'],method=route,program=result['program'],tuning=result['tuning_path'],valid=row['valid'],value=row.get('value'),reference_value=reference['value'],absolute_error=abs(row['value']-reference['value']) if row['valid'] else None))
            for i,name in enumerate(('gamma_raw','log_beta')):
                coordinates.append(dict(horizon=h,data_seed=seed,filter_seed=row['filter_seed'],method=route,program=result['program'],tuning=result['tuning_path'],valid=row['valid'],coordinate=name,score=row['score'][i] if row['valid'] else None,reference_score=reference['score'][i],absolute_error=row['component_absolute_errors'][i] if row['valid'] else None))
        safety_path=directory/'safety_sensitivity.json'
        if safety_path.exists():
            for variant in read(safety_path):
                safety.append(dict(horizon=h,route=route,parameter=variant['parameter'],factor=variant['factor'],valid=variant['row']['valid'],score_change_max_abs=variant['score_change_max_abs']))
    for snapshot in snapshots:
        hashes=read(snapshot/'sha256.json')
        for path,expected in hashes.items():
            assert hashlib.sha256((snapshot/path).read_bytes()).hexdigest()==expected
        evidence[str(snapshot/'sha256.json')]=hashlib.sha256((snapshot/'sha256.json').read_bytes()).hexdigest()
    for (h,seed),datum in sorted(observations.items()):
        if seed not in FINAL:continue
        reference=datum['reference']
        for method,score,value in [('mixture_reference',reference['score'],reference['value']),('gaussian_kalman',datum['gaussian_score'],datum['gaussian_value'])]:
            program='independent_converged_mixture_grid' if method=='mixture_reference' else 'moment_matched_gaussian_kalman'
            values.append(dict(horizon=h,data_seed=seed,filter_seed=None,method=method,program=program,tuning='N/A deterministic reference/comparator',valid=True,value=value,reference_value=reference['value'],absolute_error=abs(value-reference['value'])))
            for i,name in enumerate(('gamma_raw','log_beta')):
                coordinates.append(dict(horizon=h,data_seed=seed,filter_seed=None,method=method,program=program,tuning='N/A deterministic reference/comparator',valid=True,coordinate=name,score=score[i],reference_score=reference['score'][i],absolute_error=abs(score[i]-reference['score'][i])))
    summary=[]
    for h in (10,20,50,120):
        for route in ('mixture_reference','gaussian_kalman',*ROUTES):
            rows=[r for (hh,rr,_),r in final_rows.items() if hh==h and rr==route and r['valid']]
            if route in ('mixture_reference','gaussian_kalman'):
                rows=[]
                for (hh,seed),datum in observations.items():
                    if hh!=h or seed not in FINAL:continue
                    ref=datum['reference'];score=ref['score'] if route=='mixture_reference' else datum['gaussian_score'];value=ref['value'] if route=='mixture_reference' else datum['gaussian_value']
                    rows.append(dict(score=score,oracle_score=ref['score'],score_l2_error=norm([a-b for a,b in zip(score,ref['score'])]),value=value,oracle_value=ref['value']))
            if not rows:continue
            error=stats([r['score_l2_error'] for r in rows])
            mean_score=[statistics.fmean(r['score'][i] for r in rows) for i in (0,1)]
            signed=[stats([r['score'][i]-r['oracle_score'][i] for r in rows]) for i in (0,1)]
            summary.append(dict(horizon=h,method=route,**error,mean_score_gamma=mean_score[0],mean_score_log_beta=mean_score[1],mean_absolute_error_gamma=statistics.fmean(abs(r['score'][0]-r['oracle_score'][0]) for r in rows),mean_absolute_error_log_beta=statistics.fmean(abs(r['score'][1]-r['oracle_score'][1]) for r in rows),signed_mean_error_gamma=signed[0]['mean'],signed_mean_error_log_beta=signed[1]['mean'],signed_error_se_gamma=signed[0]['se'],signed_error_se_log_beta=signed[1]['se'],norm_mean_error=norm([x['mean'] for x in signed]),mean_value=statistics.fmean(r['value'] for r in rows),mean_absolute_value_error=statistics.fmean(abs(r['value']-r['oracle_value']) for r in rows)))
    paired=[];heuristics=[]
    for h in (10,20,50,120):
        for left,right in itertools.combinations(ROUTES,2):
            pairs=[(final_rows.get((h,left,seed)),final_rows.get((h,right,seed))) for seed in sorted(FINAL)]
            diffs=[a['score_l2_error']-b['score_l2_error'] for a,b in pairs if a and b and a['valid'] and b['valid']]
            stat=stats(diffs)
            evidence_status='incomplete_pairs' if len(diffs)!=8 else 'exploratory_paired_t_interval; no_familywise_or_broad_superiority_claim'
            paired.append(dict(horizon=h,left=left,right=right,**stat,status=evidence_status))
        for route in ROUTES:
            for seed in sorted(FINAL):
                row=final_rows.get((h,route,seed))
                if not row or not row['valid']:continue
                datum=observations[(h,seed)];ref=datum['reference']['score']
                opponents=dict(zero_score=norm(ref),first_only=norm([a-b for a,b in zip(datum['first_score'],ref)]),gaussian_kalman=norm([a-b for a,b in zip(datum['gaussian_score'],ref)]))
                iid=final_rows.get((h,'iid_dual_cap',seed))
                if route!='iid_dual_cap' and iid and iid['valid']:opponents['iid']=iid['score_l2_error']
                for name,error in opponents.items():
                    heuristics.append(dict(horizon=h,method=route,data_seed=seed,heuristic=name,error=row['score_l2_error'],heuristic_error=error,observed_loss=row['score_l2_error']>error,promotion_veto=row['score_l2_error']>error,continuation_veto=False))
    verdict=dict(completed_scopes=len(scopes),expected_scopes=16,valid_cells=sum(r['valid'] for r in final_rows.values()),total_cells=len(final_rows),expected_cells=128,score_coordinates=len([r for r in coordinates if r['method'] in ROUTES]),candidate_failures=[s for s in scopes if s['status']!='complete'],infrastructure_or_reference_attempts=failures,source_snapshot_count=len(snapshots),reference_maximum_value_difference=max((r['value_difference'] for r in convergence),default=None),reference_maximum_score_difference=max((r['score_difference'] for r in convergence),default=None),heuristic_observed_losses=sum(r['observed_loss'] for r in heuristics),scientific_promotion=False,ranking='Exploratory paired intervals only; no broad or familywise superiority claim',terminal_complete=len(scopes)==16 and len(final_rows)==128)
    csv_file(out/'scores.csv',coordinates);csv_file(out/'values.csv',values);csv_file(out/'summary.csv',summary);csv_file(out/'paired_differences.csv',paired);csv_file(out/'heuristics.csv',heuristics);csv_file(out/'safety_sensitivity.csv',safety);csv_file(out/'reference_convergence.csv',convergence)
    dump(out/'scopes.json',scopes);dump(out/'terminal-review.json',verdict);dump(out/'budget.json',ledger);dump(out/'summary.json',dict(summary=summary,paired=paired,verdict=verdict))
    rows=['# KSC mixture comparison','',
        'Program: FP64 TensorFlow/GPU/XLA diagnostic variant, TF32 off, Contract E and dual-cap safeguards on. This differs from the production FP32/TF32 target. Particle cells use independently selected and validated controls for their exact route/horizon; `scopes.json` links every tuning artifact. Gaussian Kalman is a moment-matched approximation. The mixture grid is a converged independent reference, not a mathematically exact long-horizon oracle. These results do not establish native-SV accuracy, HMC readiness, default readiness or broad method superiority.','',
        'Each final scope uses 1,008 particles and eight independent dataset/design pairs at theta=(1.5,0), with both score coordinates. The table reports mean score-vector L2 error, its sample standard deviation, and the standard error of its mean. Reference error is zero by definition.','',
        '| T | Method | Pairs | Mean L2 error | SD(error) | SE(mean error) | Mean absolute log-likelihood error |','| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for s in summary:rows.append('| '+ ' | '.join([str(s['horizon']),LABELS[s['method']],str(s['n']),fmt(s['mean']),fmt(s['sd']),fmt(s['se']),fmt(s['mean_absolute_value_error'])])+' |')
    rows+=['','`scores.csv` contains every actual score coordinate, reference score and absolute error. `values.csv` contains every actual log likelihood and error. `summary.csv` also reports mean signed coordinate errors, coordinate SEs and the norm of the mean error vector; that norm differs from the mean of error norms.','',
        'The eight pairs vary both data and random design. Their SEs therefore are not estimates of conditional Monte Carlo uncertainty for one fixed dataset. Paired t intervals in `paired_differences.csv` are exploratory, use shared pairs, and have no adjustment for the many comparisons. Small-sample shape assumptions and limited model coverage constrain their interpretation.','',
        '## Actual mean scores and log likelihoods','',
        'Means below average the same eight datasets. Individual reference scores vary by dataset; use scores.csv for exact pair comparisons.','',
        '| T | Method | Mean gamma_raw score | Mean log_beta score | Mean absolute gamma error | Mean absolute log_beta error | Mean log likelihood |','| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for s in summary:rows.append('| '+ ' | '.join([str(s['horizon']),LABELS[s['method']],fmt(s['mean_score_gamma']),fmt(s['mean_score_log_beta']),fmt(s['mean_absolute_error_gamma']),fmt(s['mean_absolute_error_log_beta']),fmt(s['mean_value'])])+' |')
    rows+=['','## Exploratory paired error differences','',
        'Negative values favor the left method on these pairs. Each interval is a paired 95% t interval for the mean L2-error difference, without multiplicity correction. Intervals excluding zero support only that limited comparison under the t-interval assumptions; they do not establish a general ranking.','',
        '| T | Left | Right | Pairs | Mean difference | 95% interval |','| --- | --- | --- | ---: | ---: | --- |']
    for pair in paired:
        rows.append('| '+ ' | '.join([str(pair['horizon']),LABELS[pair['left']],LABELS[pair['right']],str(pair['n']),fmt(pair['mean']),f"[{fmt(pair['ci95_low'])}, {fmt(pair['ci95_high'])}]"] )+' |')
    rows+=['','## Conditional heuristic checks','',
        'Every observed loss below is a promotion veto for that case; it does not discard a finite candidate or stop the remaining comparisons. The IID method is an additional comparator for each SQMC route.','',
        '| T | Method | Heuristic | Observed losses / comparisons |','| --- | --- | --- | ---: |']
    for key,group in itertools.groupby(sorted(heuristics,key=lambda r:(r['horizon'],r['method'],r['heuristic'])),lambda r:(r['horizon'],r['method'],r['heuristic'])):
        group=list(group);rows.append(f'| {key[0]} | {LABELS[key[1]]} | {key[2]} | {sum(r["observed_loss"] for r in group)}/{len(group)} |')
    rows+=['','## Terminal review','',
        f'Completed scopes: {len(scopes)}/16. Valid final cells: {verdict["valid_cells"]}/{verdict["total_cells"]}; expected final cells: 128. Failed attempts and candidate rejections are preserved separately in terminal-review.json. Source snapshots, shared observations/reference scores, seed separation, actual-coordinate errors and quadrature convergence were checked during assembly.','',
        f'Largest observed quadrature discrepancies: value {fmt(verdict["reference_maximum_value_difference"])}, score coordinate {fmt(verdict["reference_maximum_score_difference"])}. Refinement agreement is not a rigorous bound on quadrature error.','',
        '| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | Not concluded |','| --- | --- | --- | --- | --- | --- |',
        '| Retain all four for research | Owner decision; report actual errors | Apply per-case heuristic and validity vetoes | Single regime, eight pairs, limited controls | Broader regimes and replication before ranking | No default promotion |',
        '| Accept completed comparisons as bounded evidence | Shared target and converged references checked | Invalid cells and reference failures remain explicit | Finite quadrature and particle/flow bias | Inspect paired intervals and sensitivity records | No native-SV equivalence |','',
        '| Inference status | Finding |','| --- | --- |',
        f'| Hard veto screen | {len(verdict["candidate_failures"])} scope failures; {len(failures)} nonzero attempts; {verdict["heuristic_observed_losses"]} conditional heuristic losses |',
        '| Statistically supported ranking | Only the predeclared exploratory pairwise intervals are available; no overall ranking or familywise claim |',
        '| Descriptive-only differences | Observed mean errors, likelihood errors, sensitivity changes and runtimes |',
        '| Default readiness | Not established |',
        '| Next evidence | Independent replication, broader parameter regimes, N/resolution convergence and scope-specific safeguard calibration |','',
        'Strongest alternative explanation: a favorable result can arise from the single persistent regime, limited per-scope control search, or a few dataset/design draws. A reversal under broader regimes or fresh pairs would overturn a general ranking. The mixture approximates native SV, so matching this reference does not establish exact native-SV scores. Sensitivity diagnostics record changes without selecting settings by final accuracy.','',
        '## Compute accounting','',
        f'Total conservative charge: {ledger["charged_seconds"]/3600:.6f} of 12 GPU-hours, including prior work and the 300-second provisional commit-hook reserve. Remaining: {ledger["remaining_gpu_seconds"]/3600:.6f} GPU-hours. Renewed elapsed deadline: {ledger["deadline_utc"]}. No package change, HMC, push, publication or scientific/default promotion was performed.','']
    (out/'report.md').write_text('\n'.join(rows))
    evidence[str(campaign/'budget.json')]=hashlib.sha256((campaign/'budget.json').read_bytes()).hexdigest()
    dump(out/'input-sha256.json',evidence)
    dump(out/'report-manifest.json',dict(command=__import__('sys').argv,generated_utc=datetime.now(timezone.utc).isoformat(),role='CPU standard-library post-run diagnostic reporting; no framework/GPU initialization',plan='docs/plans/sqmc-ksc-sv-comparison-20260928.md',script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),output=str(out)))
    print(json.dumps(verdict))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--campaign',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();assemble(args.campaign.resolve(),args.output.resolve())
