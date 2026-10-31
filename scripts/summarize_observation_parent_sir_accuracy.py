"""Post-run diagnostic reporting only; never imported by filtering runtimes."""
import argparse,json,math,statistics,itertools
from pathlib import Path
import numpy as np
from scipy.stats import t,binomtest
ROOT=Path(__file__).resolve().parents[1]
ARMS=['original','conditional_uniform','conditional_weighted','identity_only','uniform_only','weighted_only']
QUANTITIES=['log_likelihood','infection_score','recovery_score','noise_score']
def save(path,data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def load(path):return json.loads(path.read_text())
def vector(row):return [row['value'],*row['score']]
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    measurements={};references={};sources=[];manifests=[]
    for path in sorted(a.root.glob('*/rows.json')):
        if path.parent==a.out:continue
        rows=load(path);sources.append(str(path))
        for row in rows:
            if 'fit' in row:
                key=(row['case'],row['horizon'],row['fit'])
                if key in references and references[key]!=row:raise ValueError(f'conflicting reference duplicate {key}')
                references[key]=row
            else:
                key=(row['case'],row['horizon'],row['arm'],row['seed'])
                if key in measurements and measurements[key]!=row:raise ValueError(f'conflicting duplicate {key}')
                measurements[key]=row
        mp=path.parent/'manifest.json'
        if mp.exists():manifests.append(load(mp))
    reference_cells=[];stats=[];comparisons=[];heuristic=[]
    cells=sorted(set((r['case'],r['horizon']) for r in measurements.values())|set((r['case'],r['horizon']) for r in references.values()))
    for case,horizon in cells:
        refs=sorted([r for r in references.values() if r['case']==case and r['horizon']==horizon],key=lambda r:r['fit'])
        have=len(refs)==2
        overlap=have and all(r['reference_overlap_pass'] for r in refs)
        z=[abs(x-y)/math.hypot(sx,sy) if math.hypot(sx,sy)>0 else (0. if x==y else None) for x,y,sx,sy in zip(vector(refs[0]),vector(refs[1]),refs[0]['conditional_jackknife_se'],refs[1]['conditional_jackknife_se'])] if have else [None]*4
        agree=have and all(x is not None and x<=3 for x in z)
        usable=overlap and agree
        reference_cells.append(dict(case=case,horizon=horizon,usable=usable,overlap_pass=overlap,agreement_pass=agree,fit_difference_combined_se=z,minimum_ess=min((r['importance_ess'] for r in refs),default=None),references=refs))
        byarm={arm:{r['seed']:r for r in measurements.values() if r['case']==case and r['horizon']==horizon and r['arm']==arm} for arm in ARMS}
        for arm,rows in byarm.items():
            valid=[r for r in rows.values() if r['valid']]
            stat=dict(case=case,horizon=horizon,arm=arm,observations=len(rows),valid=len(valid),failed_seeds=[s for s,r in rows.items() if not r['valid']],reference_usable=usable)
            if valid:
                values=np.asarray([vector(r) for r in valid]);stat.update(mean=values.mean(0).tolist(),ess_mean=statistics.mean(r['ess_mean'] for r in valid),ess_min=min(r['ess_min'] for r in valid),covariance_error_max=max(r['covariance_error_max'] for r in valid))
                if usable:stat['rmse_by_reference']={r['fit']:np.sqrt(np.mean((values-vector(r))**2,0)).tolist() for r in refs}
            stats.append(stat)
        if not usable:continue
        for left,right in itertools.combinations(ARMS,2):
            pairs=sorted(s for s in byarm[left].keys() & byarm[right].keys() if byarm[left][s]['valid'] and byarm[right][s]['valid'])
            if len(pairs)<2:continue
            checks=[]
            for ref in refs:
                baseline=np.asarray(vector(ref));lv=np.asarray([vector(byarm[left][s]) for s in pairs]);rv=np.asarray([vector(byarm[right][s]) for s in pairs]);diff=(lv-baseline)**2-(rv-baseline)**2
                mean=diff.mean(0);half=t.ppf(.975,len(pairs)-1)*diff.std(0,ddof=1)/math.sqrt(len(pairs))
                ps=[]
                for k in range(4):
                    nz=diff[:,k][diff[:,k]!=0];ps.append(float(binomtest(int(np.sum(nz>0)),len(nz)).pvalue) if len(nz) else 1.)
                checks.append(dict(fit=ref['fit'],mean_squared_error_difference=mean.tolist(),ci95_lower=(mean-half).tolist(),ci95_upper=(mean+half).tolist(),sign_p_two_sided=ps))
            supported=[]
            for k in range(4):
                direction=[]
                for q in checks:
                    direction.append(-1 if q['ci95_upper'][k]<0 and q['sign_p_two_sided'][k]<=.05 else 1 if q['ci95_lower'][k]>0 and q['sign_p_two_sided'][k]<=.05 else 0)
                supported.append(left if len(pairs)>=8 and direction==[-1,-1] else right if len(pairs)>=8 and direction==[1,1] else None)
            comparisons.append(dict(case=case,horizon=horizon,left=left,right=right,paired_seeds=sorted(pairs),reference_checks=checks,exploratory_supported_lower_mse=supported))
        for method in ARMS[:3]:
            for control in ARMS[3:]:
                ms=next(s for s in stats if s['case']==case and s['horizon']==horizon and s['arm']==method);cs=next(s for s in stats if s['case']==case and s['horizon']==horizon and s['arm']==control)
                if 'rmse_by_reference' not in ms or 'rmse_by_reference' not in cs:continue
                for k,q in enumerate(QUANTITIES):
                    worse=[ms['rmse_by_reference'][r['fit']][k]>cs['rmse_by_reference'][r['fit']][k] for r in refs]
                    heuristic.append(dict(case=case,horizon=horizon,method=method,control=control,quantity=q,descriptive_disadvantage_both_fits=all(worse),promotion_veto=any(worse)))
    summary=dict(scope='fixed-data SIR diagnostic; approximate reference, branchwise finite-program scores; no canonical/HMC admission',sources=sources,quantities=QUANTITIES,rows=len(measurements),invalid_rows=sum(not r['valid'] for r in measurements.values()),expected_rows=11*3*6*8,usable_reference_cells=sum(r['usable'] for r in reference_cells),reference_cells=reference_cells,statistics=stats,paired_comparisons=comparisons,heuristic_dominance=dict(verdict='promotion_veto' if any(r['promotion_veto'] for r in heuristic) else 'no_observed_veto' if heuristic else 'unavailable',conditional_checks=heuristic),measured_run_seconds=sum(m.get('wall_seconds',0) for m in manifests),manifests=manifests)
    expected_cases=['nominal','infection09','infection11','recovery09','recovery11','noise08','noise125','infection0999','infection1001','recovery0999','recovery1001']
    expected={(c,h,a,s) for c in expected_cases for h in (10,20,50) for a in ARMS for s in range(281201,281209)}
    missing=sorted(expected-set(measurements));unexpected=sorted(set(measurements)-expected)
    summary['completeness']={'complete':not missing and not unexpected,'missing_keys':missing,'unexpected_keys':unexpected,'expected_unique_measurements':len(expected),'observed_unique_measurements':len(measurements)}
    save(a.out/'summary.json',summary)
    lines=['# SIR likelihood and score comparison','',f"Rows: {summary['rows']}/{summary['expected_rows']}; invalid rows: {summary['invalid_rows']}; usable reference cells: {summary['usable_reference_cells']}/{len(cells)}.",'','## Reference validity','','| Case | T | Min path ESS | Fit agreement | Usable |','|---|---:|---:|---|---|']
    for r in reference_cells:lines.append(f"| {r['case']} | {r['horizon']} | {r['minimum_ess']:.1f} | {r['agreement_pass']} | {r['usable']} |" if r['minimum_ess'] is not None else f"| {r['case']} | {r['horizon']} | unavailable | unavailable | False |")
    lines+=['','## Conditional accuracy','','RMSE ranges span the two references. Failed reference cells have no accuracy claims.','','| Case | T | Arm | Valid seeds | Log likelihood RMSE | Infection RMSE | Recovery RMSE | Noise RMSE | Mean ESS |','|---|---:|---|---:|---:|---:|---:|---:|---:|']
    for s in stats:
        if 'rmse_by_reference' not in s:continue
        rs=list(s['rmse_by_reference'].values());cells1=['%.3g–%.3g'%(min(r[k] for r in rs),max(r[k] for r in rs)) for k in range(4)]
        lines.append(f"| {s['case']} | {s['horizon']} | {s['arm']} | {s['valid']} | "+' | '.join(cells1)+f" | {s['ess_mean']:.1f} |")
    lines+=['','## Exploratory paired findings','','Requires eight valid pairs, two-reference agreement, a95% paired t interval excluding zero, and exact sign p<=0.05. No familywise claim.','']
    for c in comparisons:
        for q,winner in zip(QUANTITIES,c['exploratory_supported_lower_mse']):
            if winner:lines.append(f"- {c['case']}, T{c['horizon']}, {q}: {winner} has supported lower squared error than the other arm in {c['left']} vs {c['right']}.")
    lines+=['','Heuristic-dominance verdict: '+summary['heuristic_dominance']['verdict']+'. Every conditional check is retained in summary.json.']
    (a.out/'tables.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:summary[k] for k in ('rows','expected_rows','invalid_rows','usable_reference_cells','measured_run_seconds')}))
if __name__=='__main__':main()
