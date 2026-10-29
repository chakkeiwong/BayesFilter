"""Standalone bounded author-funnel comparison and post-run reporting."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import json
import math
import multiprocessing
from pathlib import Path
import resource
import time


def write(path,data):path.write_text(json.dumps(data,indent=2,allow_nan=False))


def sample_reference(a,tuning):
    from .funnel_reference import chain
    jobs=[(s,a.dimension,a.draws,a.settle,tuning,a.engine=='author-corrected',
           a.phase=='transient') for s in range(4)]
    begin=time.perf_counter()
    with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn')) as pool:
        chains=list(pool.map(chain,jobs))
    write(a.output/'chains.json',chains)
    (a.output/'author-correction.diff').write_text(chains[0]['patch'])
    extra={}
    if a.phase=='transient':
        extra['transient']=dict(initial_v=-6.,final_v=[c['draws'][-1][0] for c in chains],
            ranges=[[min(q[0] for q in c['draws']),max(q[0] for q in c['draws'])] for c in chains],
            all_finite=all(math.isfinite(x) for c in chains for q in c['draws'] for x in q),
            interpretation='Affordable neck-start mechanics; not exact paper v=-30 reproduction')
    return dict(sampling_group_seconds=time.perf_counter()-begin,
        worker_seconds=[x['seconds'] for x in chains],cpu_workers=4,
        initial_states=[x['q0'] for x in chains],independent_reference=True,
        jit_compile=False,reference_exception='Unmodified/corrected author Python reference',**extra)


def sample_tf(a,tuning):
    from .worker import setup,graph_report
    from .funnel_xla import functions
    tf,report=setup(a.device)
    def timed(tf,fn,*args):
        begin=time.perf_counter();result=fn(*args)
        # Fetch a result tensor: context.async_wait alone can leave GPU work
        # outstanding in this environment. Include the actual device barrier.
        result_q=result.state.q if hasattr(result,'state') else result.q
        result_q.numpy()
        return result,time.perf_counter()-begin
    chunk=64
    if (a.draws+a.settle)%chunk:raise ValueError('total draws must divide chunk=64')
    with tf.device('/GPU:0' if a.device=='gpu' else '/CPU:0'):
        init,sample,cfg=functions(tf,a.dimension,tuning['step'],tuning['delta'],chunk)
        starts=tuning['starts']
        if a.phase=='transient':starts=[[-6.]+[0.]*(a.dimension-1) for _ in range(4)]
        q=tf.constant(starts,tf.float64);seed=tf.constant([[104729+s,0] for s in range(4)],tf.int32)
        state,init_time=timed(tf,init,q,seed)
        preview,compile_time=timed(tf,sample,state)
        report['graph']=graph_report(sample,(state,),a.output/'sample.hlo.txt')
        draws=[];depth=[];ev=[];macro=[];failed=[];reverse=[];times=[]
        for i in range((a.draws+a.settle)//chunk):
            r,elapsed=timed(tf,sample,state);state=r.state
            if not bool(tf.reduce_all(state.valid).numpy()):raise ValueError('invalid retained cache/state')
            draws.extend(r.draws.numpy().tolist());depth.extend(r.depth.numpy().tolist())
            ev.extend(r.evaluations.numpy().tolist());macro.extend(r.macros.numpy().tolist())
            failed.extend(r.failed.numpy().tolist());reverse.extend(r.reverse_failed.numpy().tolist());times.append(elapsed)
            if i%16==0:print(json.dumps(dict(chunks=i+1,seconds=sum(times))),flush=True)
        report.update(config=asdict(cfg),initialization_seconds=init_time,
            first_call_seconds=compile_time,compile_includes_preview_draws=chunk,
            sampling_group_seconds=sum(times),chunk_seconds=times,chunk_size=chunk,
            preview_replayed_exact=preview.draws.numpy().tolist()==draws[:chunk],
            initial_states=starts,seeds=[104729+s for s in range(4)],
            timing_barrier='materialize_result_state_q')
        write(a.output/'draws.json',dict(draws=draws,depth=depth,evaluations=ev,
             macros=macro,failed=failed,reverse_zero=reverse))
        if a.phase=='transient':
            report['transient']=dict(initial_v=-6.,final_v=[q[0] for q in draws[-1]],
                ranges=[[min(row[c][0] for row in draws),max(row[c][0] for row in draws)] for c in range(4)],
                all_finite=all(math.isfinite(x) for row in draws for q in row for x in q),
                interpretation='Affordable neck-start mechanics; not exact paper v=-30 reproduction')
    if a.device=='gpu':report['allocator']=tf.config.experimental.get_memory_info('GPU:0')
    return report


def summarize(tf,x):
    from .worker import diagnostics
    d=diagnostics(tf,x)
    z=tf.concat([x[:,:,:1]/3.,x[:,:,1:]*tf.exp(-x[:,:,:1]/2.)],-1)
    zd=diagnostics(tf,z)
    v=x[:,:,:1]
    vd=diagnostics(tf,v)
    d.update(v_rhat=vd['max_rhat'],v_bulk_ess=vd['min_bulk_ess'],v_tail_ess=vd['min_tail_ess'],
             transformed=zd,finite=bool(tf.reduce_all(tf.math.is_finite(x)).numpy()),
             movement=bool(tf.reduce_all(tf.reduce_any(x[1:]!=x[:-1],axis=0)).numpy()),
             v_mean=float(tf.reduce_mean(v).numpy()),v_sd=float(tf.math.reduce_std(v).numpy()),
             v_range=[float(tf.reduce_min(v).numpy()),float(tf.reduce_max(v).numpy())])
    means=tf.reduce_mean(z,[0,1]);variances=tf.math.reduce_variance(z,[0,1])
    d['transformed_mean']=means.numpy().tolist();d['transformed_variance']=variances.numpy().tolist()
    # Batch means: 16 blocks per chain, explanatory when stationarity is unsupported.
    n=int(x.shape[0]);blocks=16;size=n//blocks
    b=tf.reshape(z[:blocks*size],[blocks,size,4,z.shape[-1]])
    bm=tf.reduce_mean(b,1)
    se=tf.math.reduce_std(tf.reshape(bm,[-1,z.shape[-1]]),axis=0)/tf.sqrt(tf.constant(4*blocks-1.,tf.float64))
    d['transformed_mean_batch_se']=se.numpy().tolist()
    for threshold in (-3.,-6.):
        y=tf.cast(v<threshold,tf.float64)
        batches=tf.reduce_mean(tf.reshape(y[:blocks*size],[blocks,size,4]),1)
        p=float(tf.reduce_mean(y).numpy());err=float((tf.math.reduce_std(batches)/tf.sqrt(tf.constant(4*blocks-1.,tf.float64))).numpy())
        d['tail_'+str(int(threshold))]=dict(observed=p,analytic=.5*math.erfc(-threshold/(3*math.sqrt(2))),
            batch_se=err,chain_rates=tf.reduce_mean(y,0).numpy().reshape(-1).tolist())
    d['posterior_screen']=bool(d['finite'] and d['movement'] and max(d['max_rhat'],zd['max_rhat'])<=1.01
        and min(d['min_bulk_ess'],d['min_tail_ess'],zd['min_bulk_ess'],zd['min_tail_ess'])>=400)
    d['uncertainty_status']='Batch SE is descriptive unless chains are stationary; no equivalence or ranking test'
    return d


def analyze(a):
    from .worker import setup
    tf,report=setup('cpu');report['cases']=[];tf_runs=[]
    for directory in a.runs:
        meta=json.loads((directory/'result.json').read_text())
        if (directory/'draws.json').exists():
            raw=json.loads((directory/'draws.json').read_text());x=tf.constant(raw['draws'],tf.float64)
            work=float(tf.reduce_sum(raw['evaluations']).numpy())
            failures=int(tf.reduce_sum(tf.cast(raw['failed'],tf.int32)).numpy())
            tf_runs.append((str(directory),meta,raw))
        else:
            chains=json.loads((directory/'chains.json').read_text())
            x=tf.transpose(tf.constant([c['draws'] for c in chains],tf.float64),[1,0,2])
            work=sum(sum(d[6]+d[7] for d in c['diagnostics']) for c in chains);failures=None
        retained=x[meta['settle']:]
        case=dict(run=str(directory),engine=meta['engine'],device=meta['device'],dimension=meta['dimension'],
            step=meta['step'],delta=meta['delta'],draws=meta['draws'],settle=meta['settle'],
            sampling_group_seconds=(meta['sampling_group_seconds'] if meta['engine']!='tf' or
                meta.get('timing_barrier')=='materialize_result_state_q' else None),gradient_evaluations=work,
            nonfinite_trajectory_flags=failures,diagnostics=summarize(tf,retained),ladder=[])
        if meta['engine']!='tf':
            ds=[d for c in chains for d in c['diagnostics']]
            case['author_telemetry']=dict(max_forward_exponent=max(d[9] for d in ds),
                cap_reached_iterations=sum(d[9]>=10 for d in ds),max_sampled_doublings=max(d[1] for d in ds),
                max_computed_doublings=max(d[20] for d in ds))
        for n in (512,2048,8192):
            if n<=retained.shape[0]:case['ladder'].append(dict(draws=n,diagnostics=summarize(tf,retained[:n])))
        report['cases'].append(case)
        print(json.dumps({k:v for k,v in case.items() if k in ('engine','device','dimension','draws','sampling_group_seconds')}),flush=True)
    report['matched_comparisons']=[]
    for candidate in report['cases']:
        if candidate['engine']!='tf':continue
        for baseline in report['cases']:
            if (baseline['engine']!='author-corrected' or baseline['dimension']!=candidate['dimension']
                or baseline['draws']!=candidate['draws']):continue
            a0,b0=candidate['diagnostics'],baseline['diagnostics']
            comp=dict(candidate=candidate['run'],baseline=baseline['run'],dimension=candidate['dimension'],
                interpretation='Descriptive difference; neither equivalence nor ranking established')
            for label in ('tail_-3','tail_-6'):
                diff=a0[label]['observed']-b0[label]['observed']
                se=math.hypot(a0[label]['batch_se'],b0[label]['batch_se'])
                comp[label]=dict(difference=diff,combined_batch_se=se,
                    descriptive_95_interval=[diff-1.96*se,diff+1.96*se])
            report['matched_comparisons'].append(comp)
    report['cross_device']=[]
    for path,meta,raw in tf_runs:
        if meta['device']!='cpu':continue
        for gpath,gmeta,graw in tf_runs:
            if gmeta['device']!='gpu' or any(meta[k]!=gmeta[k] for k in ('dimension','draws','step','delta')):continue
            x=tf.constant(raw['draws'],tf.float64);y=tf.constant(graw['draws'],tf.float64)
            report['cross_device'].append(dict(cpu=path,gpu=gpath,
                max_absolute_difference=float(tf.reduce_max(tf.abs(x-y)).numpy()),
                max_scaled_difference=float(tf.reduce_max(tf.abs(x-y)/(1+tf.abs(x))).numpy()),
                exact_depth=raw['depth']==graw['depth'],exact_work=raw['evaluations']==graw['evaluations'],
                exact_macros=raw['macros']==graw['macros'],exact_reverse_zero=raw['reverse_zero']==graw['reverse_zero'],
                interpretation='Long chaotic trajectories need not be bitwise identical; matched-input local parity is the correctness check'))
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',required=True)
    p.add_argument('--engine',default='tf');p.add_argument('--device',default='cpu')
    p.add_argument('--dimension',type=int,default=11);p.add_argument('--draws',type=int,default=512)
    p.add_argument('--settle',type=int,default=512);p.add_argument('--tuning',type=Path)
    p.add_argument('--runs',type=Path,nargs='*',default=[]);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();begin=time.monotonic()
    report=dict(phase=a.phase,engine=a.engine,device=a.device,dimension=a.dimension,draws=a.draws,settle=a.settle)
    if a.phase=='tune':
        from .funnel_reference import tune
        report.update(tune(a.dimension,a.output))
    elif a.phase=='extreme':
        from .funnel_reference import extreme_probe
        report.update(extreme_probe(a.output))
    elif a.phase=='analyze':report.update(analyze(a))
    else:
        tuning=json.loads(a.tuning.read_text()) if a.tuning else dict(step=.3,delta=.3,starts=[])
        if a.tuning and tuning['dimension']!=a.dimension:raise ValueError('tuning dimension mismatch')
        report.update(step=tuning['step'],delta=tuning['delta'],tuning_file=str(a.tuning))
        report.update(sample_tf(a,tuning) if a.engine=='tf' else sample_reference(a,tuning))
    report.update(wall_seconds=time.monotonic()-begin,peak_host_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    write(a.output/'result.json',report);print(json.dumps(dict(output=str(a.output),seconds=report['wall_seconds'])),flush=True)

if __name__=='__main__':main()
