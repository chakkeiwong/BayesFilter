"""Saved-array development diagnostic; never confirmation or new stopped draws."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.stats import beta, norm
import tensorflow as tf

from bayesfilter.inference.hmc_precision import quantile_precision, QUANTILE_PRECISION_METHOD
from bayesfilter.inference.hmc_convergence import rank_normalized_split_rhat_summary
from bayesfilter.inference.hmc_posterior_diagnostics import rank_normalized_bulk_tail_ess
from bayesfilter.testing.inference_validation.storage import read_tensor, read_json, write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    started=time.monotonic();inventory=read_json(args.inventory)
    summaries={};results=[];warmup=[];hashes={}
    for row in inventory['rows']:
        if row['target'] not in ('gaussian','beta_binomial'):
            continue
        target=row['target'];pipe=read_json(row['pipeline'])
        member=next((m for m in pipe['members'] if m['candidate_id'] in pipe['selection']['candidate_ids']),{})
        output={'target':target,'replication':row['replication'],'delivered':row['delivered'],'arms':{}}
        for arm in ('stopped','fixed'):
            location=member.get('draws_path') if arm=='stopped' else member.get('fixed_comparator',{}).get('draws_path')
            old=row['stopping_pair'].get(arm,{})
            output['arms'][arm]={}
            if not location or not Path(location).exists():
                continue
            values=read_tensor(location)
            hashes[location]=read_json(location+'.json')['sha256']
            if int(values.shape[0])<4:
                continue
            report=quantile_precision(values,.5)
            ordered=np.sort(np.asarray(values).reshape(-1,values.shape[-1]),axis=0)
            names=[name for name in old if name.endswith(':quantile')]
            for i,name in enumerate(names):
                original=old[name];truth=original['reference'];estimate=float(report['estimate'][i])
                se=float(report['mcse'][i]);ess=float(report['indicator_ess'][i])
                valid=bool(report['valid'][i]);direct=None
                if math.isfinite(ess) and ess>0:
                    probability=beta.ppf([.025,.975],ess*.5+1,ess*.5+1)
                    a=max(int(math.floor(probability[0]*len(ordered))),1)-1
                    b=min(int(math.ceil(probability[1]*len(ordered))),len(ordered))-1
                    if ordered[b,i]>ordered[a,i]:
                        direct=[float(ordered[a,i]),float(ordered[b,i])]
                entry={'original':original,'estimate':estimate,'mcse':se if valid else None,
                       'indicator_ess':ess if math.isfinite(ess) else None,
                       'corrected_symmetric_covered':bool(valid and abs(estimate-truth)<=norm.ppf(.975)*se),
                       'corrected_available':valid,'direct_interval':direct,
                       'direct_covered':bool(direct is not None and direct[0]<=truth<=direct[1])}
                output['arms'][arm][name]=entry
                counts=summaries.setdefault(target+':'+arm+':'+name,Counter())
                counts['available']+=valid;counts['original_covered']+=bool(original['covered'])
                counts['corrected_symmetric_covered']+=entry['corrected_symmetric_covered']
                counts['direct_covered']+=entry['direct_covered']
                counts['changed_mcse']+=bool(valid and original.get('mcse') is not None
                    and not math.isclose(se,original['mcse'],rel_tol=1e-9,abs_tol=1e-14))
        if 'warmup_cap' in row['categories']:
            location=member['warmup_path'];values=read_tensor(location)
            hashes[location]=read_json(location+'.json')['sha256']
            rhat=rank_normalized_split_rhat_summary(values,rhat_max=1.05)
            ess=rank_normalized_bulk_tail_ess(tf.transpose(values,(1,0,2)))
            previous=row['last_warmup_check']['modern_rhat']
            warmup.append({'replication':row['replication'],'draws':int(values.shape[0]),
                           'recent_window':previous,'whole_window':rhat,
                           'whole_bulk_ess':ess['bulk'],'whole_tail_ess':ess['tail']})
        results.append(output)
    assert len(results)==256 and len(warmup)==7
    report={'role':'development saved-array analysis; original stopping times unchanged',
            'quantile_method':QUANTILE_PRECISION_METHOD,'planned_per_target':128,'rows':results,
            'summary':summaries,'warmup_caps':warmup,'tensor_sha256':hashes,
            'inventory_sha256':hashlib.sha256(args.inventory.read_bytes()).hexdigest(),
            'default_promoted':False,'ranking_supported':False,'new_hmc_fits':0,
            'elapsed_seconds':time.monotonic()-started}
    if args.output.exists():raise ValueError('output already exists')
    write_json(args.output,report)
    print(json.dumps({'result':str(args.output),'summary':summaries,'elapsed_seconds':report['elapsed_seconds']}))


if __name__=='__main__':
    main()
