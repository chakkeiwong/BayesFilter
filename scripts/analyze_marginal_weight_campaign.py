"""Post-run diagnostic aggregation only; no tuning/admission decisions."""
import itertools,json,math,statistics
from pathlib import Path
from scipy.stats import t
root=Path(__file__).resolve().parents[1]
base=root/'docs/plans/artifacts';prefix='surrogate-hmc-marginal-weight-20261005-'
out=base/(prefix+'supplement');aggregate={'lg_ksc':[],'m13':[]}
def read(path):return json.loads(path.read_text())
for name in ('lg1-t2','lg3-t2','lg3-t10','ksc-t10','ksc-t20'):
 r=read(base/(prefix+'01')/name/'result.json')
 aggregate['lg_ksc'].append(dict(name=name,n=r['scope']['n'],points=len(r['summaries']),
  seeds=[s['n'] for s in r['summaries']],all_noninferior=r['noninferiority_established'],
  significant_deterioration=any(v['significant_deterioration'] for s in r['summaries'] for v in s['metrics'].values()),
  maximum_value_upper_bound=max(s['metrics']['value']['simultaneous_interval'][1] for s in r['summaries']),
  maximum_score_upper_bound=max(s['metrics']['score']['simultaneous_interval'][1] for s in r['summaries']),
  reference_checks=r['reference_checks']))
for tag in ('02','03'):
 for name in ('m13-t1','m13-t20'):
  r=read(base/(prefix+tag)/name/'result.json')
  for point,(rows,ref,summary) in enumerate(zip(r['claim_rows'],r['references'],r['summaries'])):
   methods={}
   for policy in ('ancestor','marginal_mixture'):
    good=all(row[policy]['valid'] for row in rows)
    methods[policy]=dict(valid_count=sum(row[policy]['valid'] for row in rows),n=len(rows),
     mean_absolute_log_error=statistics.mean(abs(row[policy]['value']-ref['value']) for row in rows) if good else None,
     mean_scaled_score_error=statistics.mean(math.sqrt(sum((.1*(x-y))**2 for x,y in zip(row[policy]['score'],ref['score']))/2) for row in rows) if good else None,
     median_seconds=statistics.median(row[policy]['seconds'] for row in rows))
   row=dict(name=name,particles=r['scope']['n'],point=point,parameters=[0.,0.] if point==0 else [-.73272085,-.01465508],
    methods=methods,reference=ref,summary=summary,
    bootstrap_mean_absolute_value_error=statistics.mean(abs(x['bootstrap_value']-ref['value']) for x in rows),
    bootstrap_comparison_scope='equal particle count, different cost; reference approximate at T20')
   if name=='m13-t20':
    low=r['reference_checks']['refinements'][0]['rows'][point]
    row['reference_refinement_z']=[(a-b)/math.sqrt(sa*sa+sb*sb) for a,b,sa,sb in zip(
      [ref['value']]+ref['score'],[low['value']]+low['score'],ref['mcse'],low['mcse'])]
    if summary['valid']:
     # Sensitivity to a +/-3 MCSE reference box. This does not bound particle bias.
     worst={}
     for metric in ('value','score'):
      bounds=[]
      perturbations=([(-1,),(0,),(1,)] if metric=='value' else list(itertools.product((-1,0,1),repeat=2)))
      for shift in perturbations:
       target=(ref['value']+3*shift[0]*ref['mcse'][0] if metric=='value' else [x+3*s*se for x,s,se in zip(ref['score'],shift,ref['mcse'][1:])])
       def error(x):
        return abs(x['value']-target) if metric=='value' else math.sqrt(sum((.1*(a-b))**2 for a,b in zip(x['score'],target))/2)
       differences=[error(x['marginal_mixture'])-error(x['ancestor']) for x in rows]
       bounds.append(statistics.mean(differences)+t.ppf(1-.05/180,len(rows)-1)*statistics.stdev(differences)/math.sqrt(len(rows)))
      worst[metric]=float(max(bounds))
     row['worst_paired_upper_bound_under_reference_3mcse_box']=worst
   aggregate['m13'].append(row)
heuristic_rows=read(out/'heuristics.json')['rows']
aggregate['heuristic_dominance']=dict(
 status='PROMOTION_VETO_OBSERVED_HEURISTIC_UNDERPERFORMANCE',
 statistically_supported_ranking=False,
 rationale='Conditional observed error losses veto promotion; no uncertainty-based ranking against these comparators was tested.',
 n256=[dict(name=r['name'],point=r['point'],candidate_mean_absolute_value_error=r['candidate_mean_absolute_value_error'],
  competitor_mean_absolute_value_errors=r['competitor_mean_absolute_value_errors'],
  observed_lower_error_comparators=r['descriptive_underperformance']) for r in heuristic_rows],
 n4096_bootstrap=[dict(name=r['name'],point=r['point'],
  candidate_mean_absolute_value_error=r['methods']['marginal_mixture']['mean_absolute_log_error'],
  bootstrap_mean_absolute_value_error=r['bootstrap_mean_absolute_value_error'],
  observed_lower_bootstrap_error=r['bootstrap_mean_absolute_value_error']<r['methods']['marginal_mixture']['mean_absolute_log_error'])
  for r in aggregate['m13'] if r['particles']==4096])
p=out/'precision.json'
if p.exists():
 precision=read(p);rows=precision['rows']
 aggregate['precision']=dict(rows=rows,
  max_lg_ksc_value_difference=max(r['absolute_value_difference'] for r in rows if not r['name'].startswith('m13')),
  max_lg_ksc_scaled_score_difference=max(r['scaled_score_difference'] for r in rows if not r['name'].startswith('m13')))
totals={tag:read(base/(prefix+tag)/'campaign.json')['used_seconds'] for tag in ('01','02','03')}
for phase in ('tile','precision','heuristics'):
 p=out/(phase+'.json')
 if p.exists():totals[phase]=read(p)['manifest']['wall_seconds']
totals['failed_tile_upper_bound']=10.
aggregate['elapsed_gpu_budget']=dict(entries=totals,used=sum(totals.values()),cap=21600.,remaining=21600.-sum(totals.values()),
 qualification='diagnostic phase times exclude startup; reserve an extra 60 seconds for all three diagnostic process startups')
aggregate['elapsed_gpu_budget']['used']+=60.;aggregate['elapsed_gpu_budget']['remaining']-=60.
(out/'aggregate.json').write_text(json.dumps(aggregate,indent=2,allow_nan=False))
for r in aggregate['lg_ksc']:print(r['name'],r['all_noninferior'],r['maximum_value_upper_bound'],r['maximum_score_upper_bound'])
for r in aggregate['m13']:
 print(r['particles'],r['name'],r['point'],'bootstrap',r['bootstrap_mean_absolute_value_error'],'marginal',r['methods']['marginal_mixture']['mean_absolute_log_error'],'uncertainty bound',r.get('worst_paired_upper_bound_under_reference_3mcse_box'))
print('budget',aggregate['elapsed_gpu_budget']['used'])
