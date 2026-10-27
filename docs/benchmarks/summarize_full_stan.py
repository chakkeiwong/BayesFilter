"""Post-run diagnostic reporting; descriptive summaries, no winner selection."""
import argparse
import json
from pathlib import Path
import statistics

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('manifests',nargs='+',type=Path)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    rows=[]
    for path in args.manifests:
        manifest=json.loads(path.read_text())
        for row in manifest['results']:
            for engine in ('direct','tfp','stan','fixed'):
                if engine not in row:continue
                r=row[engine]
                if engine=='fixed':
                    rows.append(dict(target=row['target'],seed=row['seed'],init_scale=row.get('init_scale',1.),
                        engine=engine,source=str(path),
                        status=r['status'],completion_status=r.get('completion_status'),
                        seconds=r['tuned_total_seconds'],tuning_seconds=r.get('tuning_seconds'),
                        timing_scope=r.get('timing_scope'),
                        verified_count=len(r.get('verified_candidate_ids',[])),members=r['candidate_results']))
                    continue
                diag=r.get('diagnostics',{})
                rows.append(dict(target=row['target'],seed=row['seed'],init_scale=row.get('init_scale',1.),
                    engine=engine,source=str(path),
                    seconds=r['tuned_total_seconds'],serialization_seconds=r.get('serialization_seconds',0.),
                    passed=diag.get('posterior_screen_pass'),rhat=diag.get('max_rhat'),
                    bulk_ess=diag.get('minimum_bulk_ess'),tail_ess=diag.get('minimum_tail_ess'),
                    mcse=diag.get('maximum_standardized_mcse'),reference=diag.get('reference_agreement'),
                    leaps=diag.get('leapfrogs'),divergence=diag.get('native_divergence_fraction')))
    keys=[(r['target'],r['engine'],r['init_scale'],r['seed']) for r in rows]
    if len(keys)!=len(set(keys)):
        raise ValueError('Duplicate target/engine/start-scale/seed rows would count a replication twice')
    grouped=[]
    for target in dict.fromkeys(r['target'] for r in rows):
        for engine,scale in dict.fromkeys((r['engine'],r['init_scale']) for r in rows if r['target']==target):
            subset=[r for r in rows if r['target']==target and r['engine']==engine and r['init_scale']==scale]
            times=[r['seconds'] for r in subset]
            record=dict(target=target,engine=engine,init_scale=scale,n=len(subset),seconds_median=statistics.median(times),
                        seconds_min=min(times),seconds_max=max(times))
            if engine!='fixed':
                record['passes']=sum(r['passed'] is True for r in subset)
                for name in ('rhat','bulk_ess','tail_ess','mcse','leaps','divergence'):
                    vals=[r[name] for r in subset if r.get(name) is not None]
                    record[name+'_median']=statistics.median(vals) if vals else None
            else:
                record['verified_members']=sum(r['verified_count'] for r in subset)
                record['assessed_members']=sum(len(r['members']) for r in subset)
                record['passing_members']=sum(x['diagnostics']['posterior_screen_pass'] for r in subset for x in r['members'])
                record['fits_with_passing_member']=sum(any(x['diagnostics']['posterior_screen_pass'] for x in r['members']) for r in subset)
                record['no_verified_fits']=sum(r['status']=='no_verified_candidate' for r in subset)
                record['failed_fits']=sum(r['status']=='preparation_or_tuning_failed' for r in subset)
                record['unfinished_searches']=sum(r['completion_status'] not in ('complete',None) for r in subset)
            grouped.append(record)
    result=dict(inference='descriptive only; three seeds and no predeclared inferential comparison do not establish ranking',
                manifests=list(map(str,args.manifests)),rows=rows,summary=grouped)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    for r in grouped:print(json.dumps(r))

if __name__=='__main__':main()
