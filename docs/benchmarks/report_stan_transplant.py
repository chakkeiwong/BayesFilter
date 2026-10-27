"""Post-run diagnostic reporting only; no sampler or tuning authority."""
import argparse
import json
import math
from pathlib import Path
import statistics

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    r=json.loads((args.run/'manifest.json').read_text())
    if r['status']!='complete':raise ValueError('complete comparison required')
    groups={}
    for cell in r['results']:groups.setdefault(cell['target'],[]).append(cell)
    records=[]
    for target,cells in groups.items():
        item={'target':target,'replications':len(cells)}
        if len(cells)!=3:raise ValueError('predeclared interval expects three replications')
        for engine in ['tfp','stan']:
            rows=[c[engine] for c in cells]
            item[engine]=dict(median_tuned_seconds=statistics.median(x['tuned_total_seconds'] for x in rows),
                median_total_with_serialization_seconds=statistics.median(x['tuned_total_seconds']+x.get('serialization_seconds',0) for x in rows),
                seconds=[x['tuned_total_seconds'] for x in rows],
                passes=sum(x['diagnostics']['posterior_screen_pass'] for x in rows),
                max_rhat=max(x['diagnostics']['max_rhat'] for x in rows),
                min_bulk_ess=min(x['diagnostics']['minimum_bulk_ess'] for x in rows),
                min_tail_ess=min(x['diagnostics']['minimum_tail_ess'] for x in rows),
                max_standardized_mcse=max(x['diagnostics']['maximum_standardized_mcse'] for x in rows),
                median_leapfrogs=statistics.median(x['diagnostics']['leapfrogs'] for x in rows),
                median_diagnostics_seconds=statistics.median(x['diagnostics']['diagnostics_seconds'] for x in rows),
                max_native_divergence_fraction=max(x['diagnostics']['native_divergence_fraction'] for x in rows))
        item['tfp']['median_warmup_seconds']=statistics.median(c['tfp']['warmup_seconds'] for c in cells)
        item['tfp']['median_sampling_seconds']=statistics.median(c['tfp']['sample_seconds'] for c in cells)
        item['stan']['median_max_published_warmup_seconds']=statistics.median(max(v['warmup'] for v in c['stan']['published_chain_seconds']) for c in cells)
        item['stan']['median_max_published_sampling_seconds']=statistics.median(max(v['sampling'] for v in c['stan']['published_chain_seconds']) for c in cells)
        item['tfp']['first_calls_including_discarded_execution_seconds']=sum(r['compiled'][target][phase]['first_call_including_execution_seconds'] for phase in ['warmup','sampling'])
        logs=[math.log(c['tfp']['tuned_total_seconds']/c['stan']['tuned_total_seconds']) for c in cells]
        # Exact df=2 Student-t .975 quantile, from its closed-form CDF.
        quantile=.95*math.sqrt(2/(1-.95**2))
        middle=statistics.mean(logs);radius=quantile*statistics.stdev(logs)/math.sqrt(3)
        item['paired_geometric_time_ratio']=math.exp(middle)
        item['model_dependent_95pct_ratio_interval']=[math.exp(middle-radius),math.exp(middle+radius)]
        item['valid_efficiency_comparison']=item['tfp']['passes']==item['stan']['passes']==3
        item['ranking_scope']=('Measured fixed-budget elapsed-time difference, conditional on three posterior screens; not optimal time to precision'
                               if item['valid_efficiency_comparison'] else 'No valid efficiency ranking; posterior screens failed')
        records.append(item)
    output={'source_manifest':str(args.run/'manifest.json'),'records':records,
            'wall_seconds':r['wall_seconds'],'host_max_rss_kib':r['host_max_rss_kib'],
            'interval_assumption':'Independent paired log time ratios are approximately Gaussian; only three repetitions, no simultaneous correction across targets',
            'timing_boundary':'TF synchronized numerical warmup+sampling; native blocking process wall includes startup/init/warmup/sampling/CSV; TF serialization separately added',
            'native_cache_limitation':'Only finite native retained draws and native successful execution checked; internal gradient cache not exported'}
    args.output.write_text(json.dumps(output,indent=2)+chr(10))
    for item in records:
        t=item['tfp'];s=item['stan']
        print(f"| {item['target']} | {t['median_tuned_seconds']:.3f} | {s['median_tuned_seconds']:.3f} | {t['passes']}/3, {s['passes']}/3 | {item['paired_geometric_time_ratio']:.2f} |")

if __name__=='__main__':main()
