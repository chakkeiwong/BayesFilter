"""Post-run timing summaries; no sampler selection or runtime decisions."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def summarize(root):
    out=dict(artifacts=str(root),attempts=[],layout=[],batch=[],native=[],qualification=[])
    for p in sorted(root.glob('*/manifest.json')):
        m=json.loads(p.read_text());out['attempts'].append(dict(path=str(p),status=m['status'],
            workers=len(m['attempts']),seconds=sum(x['wall_seconds'] for x in m['attempts']),
            sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
        if m['status']!='completed':continue
        result=p.parent/'results/result.json'
        if result.exists():
            r=json.loads(result.read_text())
            # The original unbalanced screen is preserved but excluded from promotion.
            if p.parent.name=='05-layout-screen':continue
            for row in r['results']:
                e=dict(target=row['target'],source=str(result))
                if r['mode']=='layout':
                    e['comparisons']=row['comparisons'];e['medians']={};e['by_order']={}
                    for c in row['comparisons']:
                        rr=[x for x in row['records'] if x['candidate']==c]
                        e['medians'][c]={k:statistics.median(x['seconds'][k] for x in rr) for k in ('fieldwise',c)}
                        e['by_order'][c]={k:statistics.median(x['seconds']['fieldwise']/x['seconds'][c] for x in rr if x['order'][0]==k) for k in ('fieldwise',c)}
                    e['graphs']={k:{f:v[f] for f in ('while_count','fusion_count','trace_count','model_gradient_inside_loop')} for k,v in row['graphs'].items()}
                    e['first_call_seconds']=row['first_call_seconds']
                elif r['mode']=='batch':
                    e.update(chains=row['config']['chains'],seconds=row['median_seconds'],
                        throughput=row['chain_transitions_per_second'],prefix_parity=row['retained_prefix_parity'],
                        first_call_seconds=row['first_call_seconds'])
                else:
                    e['medians']={k:statistics.median(x['seconds'][k] for x in row['records']) for k in ('tf','generic','native','generic_serial','native_serial')}
                    e['comparisons']=row['comparisons'];e['tf_preprocess']=row['tf_preprocess']
                    e['native_preprocess']={k:statistics.median(x['preprocess_seconds'][k] for x in row['records']) for k in ('generic','native')}
                    e['max_native_draw_difference']=max(x['native_draw_max_difference'] for x in row['records'])
                    e['work_ranges']={k:[min(x['evaluations'][k] for x in row['records']),max(x['evaluations'][k] for x in row['records'])] for k in ('tf','generic','native')}
                out[r['mode']].append(e)
        val=p.parent/'validation/result.json'
        if val.exists():
            r=json.loads(val.read_text());out['qualification'].append(dict(source=str(val),status=r['status'],
                layout=r.get('execution_layout',m['command'][m['command'].index('--layout')+1]),
                device=r['device'],checks=r['checks'],api_checks=r['api_checks']))
    out['worker_seconds']=sum(x['seconds'] for x in out['attempts'])
    out['worker_launches']=sum(x['workers'] for x in out['attempts'])
    return out


def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();result=summarize(a.root)
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps({k:result[k] for k in ('worker_seconds','worker_launches','layout','batch','native')},indent=2))
if __name__=='__main__':main()
