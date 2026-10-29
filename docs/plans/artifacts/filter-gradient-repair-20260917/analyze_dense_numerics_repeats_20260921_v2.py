"""Post-run diagnostic report; no runtime or admission use."""
import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

ROOT = Path('/home/ubuntu/workspace/BayesFilter/docs/plans/artifacts/filter-gradient-repair-20260917')


def compare(actual, expected, path='result'):
    if isinstance(expected, dict):
        assert isinstance(actual,dict) and actual.keys()==expected.keys(),path
        for key in expected:
            compare(actual[key],expected[key],path+'.'+key)
    elif isinstance(expected,list):
        assert isinstance(actual,list) and len(actual)==len(expected),path
        for i,(left,right) in enumerate(zip(actual,expected,strict=True)):
            compare(left,right,f'{path}[{i}]')
    elif isinstance(expected,float):
        assert math.isfinite(expected) and abs(actual-expected)<=1e-10+1e-10*abs(expected),(path,actual,expected)
    else:
        assert type(actual) is type(expected) and actual==expected,(path,actual,expected)


parser=argparse.ArgumentParser()
parser.add_argument('--first-run',type=int,required=True)
parser.add_argument('--last-run',type=int,required=True)
args=parser.parse_args()
groups={}
source=None
for number in range(args.first_run,args.last_run+1):
    directory=ROOT/f'run-{number:05d}'
    manifest=json.loads((directory/'run.json').read_text())
    assert manifest['state']=='passed' and manifest['key'][1].startswith('dense_numerics_memory_')
    source=manifest['source_sha256'] if source is None else source
    assert source==manifest['source_sha256']
    row=json.loads((directory/'quadratic-numerics-memory.json').read_text())
    assert row['kind']=='dense'
    if manifest['device']=='GPU':
        assert manifest['environment']['CUDA_VISIBLE_DEVICES']=='3'
    groups.setdefault((manifest['device'],row['dimension']),{}).setdefault(row['arm'],[]).append((number,manifest['key'][6],row))
assert len(groups)==4
results=[]
for (device,dimension),arms in sorted(groups.items()):
    assert set(arms)=={'before','graph','xla'}
    reference=arms['before'][0][2]
    metrics={}
    for arm,rows in arms.items():
        assert len(rows)==3 and {row[1] for row in rows}=={0,1,2}
        values=[]
        for number,repeat,row in rows:
            assert row['input_sha256']==reference['input_sha256']
            assert row['original_source_sha256']==reference['original_source_sha256']
            compare(row['result'],reference['result'])
            compare(row['changed_result'],reference['changed_result'])
            obs=[*row['stages'].values(),*(v['memory'] for v in row['samples'])]
            values.append({'run':number,'repeat':repeat,
                'warm_ms':1000*statistics.median(v['seconds'] for v in row['samples'][1:]),
                'cold_ms':1000*row['samples'][0]['seconds'],
                'cold_total_ms':1000*(row['samples'][0]['seconds']+row['build_seconds']+row['trace_seconds']),
                'build_ms':1000*row['build_seconds'],'trace_ms':1000*row['trace_seconds'],
                'host_rss_peak':max(v['host']['VmRSS'] for v in obs),
                'reported_vmhwm':max(v['host']['VmHWM'] for v in obs),
                'warm_rss_change':row['samples'][-1]['memory']['host']['VmRSS']-row['samples'][1]['memory']['host']['VmRSS'],
                'gpu_peak':max(v['gpu']['peak'] for v in obs) if device=='GPU' else None,
                'trace_count':row['trace_count'],'graph_nodes':row['graph_nodes'],
                'gpu_warm_current':sorted({v['memory']['gpu']['current'] for v in row['samples'][1:]}) if device=='GPU' else None})
        warm=[v['warm_ms'] for v in values]
        metrics[arm]={'runs':values,'warm_process_median_ms':statistics.median(warm),'warm_process_range_ms':[min(warm),max(warm)],
            'cold_total_median_ms':statistics.median(v['cold_total_ms'] for v in values),
            'cold_total_range_ms':[min(v['cold_total_ms'] for v in values),max(v['cold_total_ms'] for v in values)],
            'host_peak_median':statistics.median(v['host_rss_peak'] for v in values),
            'gpu_peak_median':statistics.median(v['gpu_peak'] for v in values) if device=='GPU' else None}
    triggers=[]
    for baseline in ('before','graph'):
        old,new=metrics[baseline],metrics['xla']
        if new['cold_total_median_ms']>2*old['cold_total_median_ms']:
            triggers.append('cold_total_over_2x_vs_'+baseline)
        if new['warm_process_median_ms']>1.2*old['warm_process_median_ms']:
            triggers.append('warm_over_20_percent_vs_'+baseline)
        if new['host_peak_median']-old['host_peak_median']>256*2**20:
            triggers.append('host_over_256MiB_vs_'+baseline)
        if device=='GPU' and new['gpu_peak_median']>2*old['gpu_peak_median']:
            triggers.append('gpu_peak_over_2x_vs_'+baseline)
    result={'device':device,'dimension':dimension,'complete_records_equal':True,'metrics':metrics,'triggers':triggers}
    results.append(result)
    print(json.dumps({'device':device,'dimension':dimension,'warm_ms':{k:v['warm_process_median_ms'] for k,v in metrics.items()},'ranges':{k:v['warm_process_range_ms'] for k,v in metrics.items()},'triggers':triggers}))
report={'role':'descriptive_dense_dependency_fresh_process_repeats','runs':[args.first_run,args.last_run],
    'comparisons':results,'atol':1e-10,'rtol':1e-10,
    'analysis_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'nonclaims':['No statistical ranking; three process repeats only.','RSS/VmHWM are observed/reported, not exact allocation attribution.','Warm allocation observations do not establish general leak freedom.','Complete public initializer costs remain separate.']}
with (ROOT/f'dense-numerics-repeats-{args.last_run:05d}.json').open('x') as handle:
    json.dump(report,handle,indent=2,allow_nan=False)
    handle.write('\n')
