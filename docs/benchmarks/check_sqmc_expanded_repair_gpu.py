"""Bounded dynamic-diagnostic parity and revised compilation-cost check."""
from pathlib import Path
import argparse
import json
import subprocess
import time
import traceback
import run_sqmc_expanded_comparison as campaign
import run_sqmc_expanded_repair as repair

def main(out):
    repair.configure_profile()
    started=time.perf_counter();out.mkdir(parents=True,exist_ok=False)
    meta=campaign.manifest();result=dict(status='running',records=[],nonclaims='mechanics/cost only')
    tf=None
    try:
        prior=campaign.ROOT/'docs/plans/artifacts/sqmc-expanded-20260928/renewal-01/repair-check-04'
        previous=json.loads((prior/'result.json').read_text())
        prior_meta=json.loads((prior/'manifest.json').read_text())
        if previous['status']!='passed' or campaign.numerical_sources(meta['source_sha256'])!=campaign.numerical_sources(prior_meta['source_sha256']):
            raise RuntimeError('previous numerical cost evidence cannot be reused')
        result['reused_cost_source']=str(prior);result['reused_cost_sha256']=campaign.digest(prior/'result.json')
        inventory=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader'],text=True)
        meta['compute_apps_before']=inventory
        if campaign.GPU_UUID in inventory: raise RuntimeError('selected GPU is occupied')
        tf,gpu_meta=campaign.configure_gpu();meta.update(gpu_meta)
        from bayesfilter.highdim.sqmc_full_lgssm_tf import FullLGSSMSpec
        from bayesfilter.highdim.sqmc_campaign_tf import random_inputs
        for d,horizon,route,seed,epsilons in [(3,2,'iid_dual_cap',195002,(.4,102.4)),(10,120,'repaired_permutation',179003,(102.4,.4))]:
            spec=FullLGSSMSpec('full_matrix',d);theta=spec.default_theta(tf.float64)
            obs=spec.simulate(theta,horizon,seed,jit_compile=True)
            inputs=random_inputs(route,seed,1020,d,horizon,tf.float64)
            kernel=repair.trace_summary_kernel(spec,route,campaign.CONTROLS[-1],1020,horizon,dynamic_epsilon=True)
            for index,epsilon in enumerate(epsilons):
                for phase in (('cold','warm') if index==0 else ('warm',)):
                    tick=time.perf_counter()
                    row=repair.summarize_trace(kernel(theta,*inputs,obs,tf.constant(epsilon,tf.float32)))
                    row.update(d=d,horizon=horizon,route=route,epsilon=epsilon,phase=phase,wall_seconds=time.perf_counter()-tick)
                    result['records'].append(row);campaign.dump(out/'result.json',result)
                    print(json.dumps({k:v for k,v in row.items() if k!='per_time'}),flush=True)
                    references=[r for r in previous['records'] if r['stage']=='trace' and r['d']==d and r['horizon']==horizon and r['epsilon']==epsilon]
                    if references:
                        ref=references[0]
                        errors=[abs(row['maximum_column_tv']-ref['maximum_column_tv']),abs(row['maximum_row_error']-ref['maximum_row_error'])]
                        if row['valid']:
                            errors += [abs(row['raw_value']-ref['raw_value']),max(abs(a-b) for a,b in zip(row['raw_score'],ref['raw_score']))]
                        if row['valid']!=ref['valid'] or max(errors)>1e-10: raise RuntimeError('dynamic diagnostic differs from static fixture')
                    if epsilon==102.4 and not row['nomination_pass']:raise RuntimeError('previous accepted fixture became invalid')
            if kernel.experimental_get_tracing_count()!=1: raise RuntimeError('dynamic calibration retraced')
        costs=[r for r in previous['records'] if r['stage']=='cost']
        warm={s:max(r['wall_seconds'] for r in costs if r['phase']=='warm' and r['flow_substeps']==s) for s in (2,8)}
        cold=max(r['wall_seconds'] for r in costs if r['phase']=='cold')
        trace_cold=max(r['wall_seconds'] for r in result['records'] if r['phase']=='cold')
        trace_warm=max(r['wall_seconds'] for r in result['records'] if r['phase']=='warm')
        estimates=[]
        for scope,family,d,t,n in campaign.SCOPES:
            parameters=d*d+d*(d+1)//2+2
            # Two calibration calls per flow candidate, then validation plus four final calls at worst flow cost.
            full_score=(2*warm[2]+7*warm[8])*(t/2)*(parameters/157)
            seconds=4*(full_score+2*cold+15+trace_cold+9*trace_warm)
            estimates.append(dict(scope=scope,estimated_seconds=seconds))
        base=sum(r['estimated_seconds'] for r in estimates)
        result.update(status='passed',projection=dict(scope_estimates=estimates,base_seconds=base,conservative_seconds=1.5*base,
            assumptions='One scalar compilation and nine warm calls per unit; measured full-T120 trace bound for every scope; exact 2 flow2 plus 7 flow8 worst-case evaluation counts; two cold full-score charges; linear horizon/direction scaling; 50% margin.'))
        if campaign.source_hashes()!=meta['source_sha256']: raise RuntimeError('source changed during check')
        meta['status']='passed'
    except BaseException as error:
        result.update(status='check_failed',error=repr(error));meta.update(status='check_failed',error=repr(error))
        traceback.print_exc();raise
    finally:
        meta.update(finished_utc=campaign.now(),wall_seconds=time.perf_counter()-started)
        if tf is not None: meta['gpu_allocator_bytes']=tf.config.experimental.get_memory_info('GPU:0')
        campaign.dump(out/'result.json',result);campaign.dump(out/'manifest.json',meta)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    main(Path(parser.parse_args().output).resolve())
