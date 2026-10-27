"""Owner-authorized tuning-inclusive TFP/XLA versus CmdStan diagnostic."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import resource
import shutil
import subprocess
import sys
import time
import traceback
import benchmark_nuts_xla as base
from nuts_benchmark_targets import make_target, NAMES
from stan_nuts_warmup import Config, make_functions

ROOT=Path(__file__).resolve().parents[2]
PLAN='docs/plans/bayesfilter-stan-nuts-transplant-2026-10-27.md'
RESULT='docs/plans/bayesfilter-stan-nuts-transplant-result-2026-10-27.md'

def write(path,data):
    def clean(x):
        if isinstance(x,float) and not math.isfinite(x): return None
        if isinstance(x,dict): return {str(k):clean(v) for k,v in x.items()}
        if isinstance(x,(list,tuple)): return [clean(v) for v in x]
        return x
    path.write_text(json.dumps(clean(data),indent=2,allow_nan=False)+chr(10))

def inputs(tf,target,seed):
    d=target.dimension
    z=tf.random.stateless_normal([4,d],[seed,321],dtype=tf.float64)
    starts=target.from_reference(z)
    kind=0
    if target.name=='rotated_gaussian32': kind=1
    elif target.name.startswith('banana'): kind=2
    elif target.name=='funnel10': kind=3
    elif target.name=='regression32': kind=4
    design=(tf.random.stateless_normal([512,d],[20261027,61],dtype=tf.float64)
            if kind==4 else tf.zeros([0,d],tf.float64))
    inverse=tf.transpose(target.to_reference(tf.eye(d,dtype=tf.float64))) if kind==1 else tf.eye(d,dtype=tf.float64)
    data=dict(d=d,kind=kind,n=512 if kind==4 else 0,design=design.numpy().tolist(),
              inverse_transform=inverse.numpy().tolist(),bend=.5 if d==2 else 1.)
    return starts,data

def assess(tf,target,samples,stats):
    from bayesfilter.inference.hmc_ess import stan_cross_chain_ess
    from bayesfilter.inference.hmc_posterior_diagnostics import (
        rank_normalized_split_rhat,rank_normalized_bulk_tail_ess)
    started=time.perf_counter();d=target.dimension
    z=tf.reshape(target.to_reference(tf.reshape(samples,[-1,d])),samples.shape)
    obs=tf.concat((z,tf.square(z)),axis=-1)
    ess=stan_cross_chain_ess(obs)
    rhat=rank_normalized_split_rhat(tf.transpose(tf.concat((samples,obs),axis=-1),[1,0,2]))
    tail=rank_normalized_bulk_tail_ess(tf.transpose(z,[1,0,2]))
    variance=tf.math.reduce_variance(obs,axis=[0,1]);mcse=tf.sqrt(variance/ess)
    reference_sd=tf.concat((tf.ones([d],tf.float64),tf.fill([d],tf.sqrt(tf.constant(2.,tf.float64)))),axis=0)
    expected=tf.concat((tf.zeros([d],tf.float64),tf.ones([d],tf.float64)),axis=0)
    error=tf.abs(tf.reduce_mean(obs,axis=[0,1])-expected)
    screen=tf.reduce_all(error<=tf.maximum(4.*mcse,.05*reference_sd))
    move=tf.reduce_sum(tf.cast(tf.reduce_any(samples[1:]!=samples[:-1],axis=-1),tf.int32),axis=0)
    out=base.json_tensor(tf,dict(max_rhat=tf.reduce_max(rhat['maximum']),
        minimum_mean_ess=tf.reduce_min(ess),minimum_bulk_ess=tf.reduce_min(tail['bulk']),
        minimum_tail_ess=tf.reduce_min(tail['tail']),maximum_standardized_mcse=tf.reduce_max(mcse/reference_sd),
        reference_agreement=screen,reference_means=tf.reduce_mean(z,axis=[0,1]),
        reference_second_moments=tf.reduce_mean(tf.square(z),axis=[0,1]),
        reference_mcse=mcse,movements_per_chain=move,
        finite_retained=tf.reduce_all(tf.math.is_finite(samples)),
        acceptance=tf.reduce_mean(stats[...,0]),leapfrogs=tf.reduce_mean(stats[...,2]),
        tree_cap_fraction=tf.reduce_mean(stats[...,3]),native_divergence_fraction=tf.reduce_mean(stats[...,4]),
        finite_cache=tf.reduce_all(stats[...,5]==1.)))
    out['posterior_screen_pass']=bool(out['finite_retained'] and out['finite_cache']
        and all(x>0 for x in out['movements_per_chain']) and out['reference_agreement']
        and out['max_rhat']<=1.01 and out['minimum_bulk_ess']>=400
        and out['minimum_tail_ess']>=400 and out['maximum_standardized_mcse']<=.05)
    out['diagnostics_seconds']=time.perf_counter()-started
    return out

def tensor_save(tf,path,value):
    with tf.device('/CPU:0'):
        path.write_bytes(tf.io.serialize_tensor(tf.identity(value)).numpy())

def compile_fn(tf,fn,args,folder,label,device):
    out,seconds=base.timed(fn,*args)
    hlo=fn.experimental_get_compiler_ir(*args)(stage='optimized_hlo')
    (folder/(label+'.hlo.txt')).write_text(hlo)
    graph=base.verify_graph(fn)
    if fn.experimental_get_tracing_count()!=1: raise RuntimeError('Unexpected retrace')
    if any('/device:'+device.upper()+':0' not in x.device for x in out):
        raise RuntimeError('Wrong device placement')
    return out,dict(first_call_including_execution_seconds=seconds,
        hlo_sha256=hashlib.sha256(hlo.encode()).hexdigest(),traces=1,graph=graph)

def tf_functions(tf,tfp,target,args,folder):
    count=4 if args.layout=='batch' else 1
    cfg=Config(target.dimension,chains=count,warmup=args.warmup,draws=args.draws,
               metric='dense' if target.name=='rotated_gaussian32' else 'diag',depth=args.depth)
    warm,sample=make_functions(tf,tfp,target.log_prob,cfg)
    q,_=inputs(tf,target,20260901)
    if count==1:q=q[:1]
    seed=tf.constant([20260901,13],tf.int32)
    warmed,wreport=compile_fn(tf,warm,(q,seed),folder,'warmup',args.device)
    if not bool(tf.reduce_all(warmed[6])): raise RuntimeError('Invalid warmup compile execution')
    sampled,sreport=compile_fn(tf,sample,warmed[:4],folder,'sample',args.device)
    if not bool(tf.reduce_all(sampled.valid)): raise RuntimeError('Invalid sampling compile execution')
    return warm,sample,dict(config=asdict(cfg),warmup=wreport,sampling=sreport)

def tf_run(tf,warm,sample,starts,seed,args,folder):
    if args.layout=='batch': initial=[(starts,tf.constant([seed,17],tf.int32))]
    else: initial=[(starts[i:i+1],tf.constant([seed,17+i],tf.int32)) for i in range(4)]
    def call_w(pair):
        with tf.device('/'+args.device.upper()+':0'):return warm(*pair)
    def call_s(w):
        with tf.device('/'+args.device.upper()+':0'):return sample(*w[:4])
    with ThreadPoolExecutor(max_workers=4 if args.layout=='scalar' else 1) as pool:
        t=time.perf_counter();ws=list(pool.map(call_w,initial))
        for w in ws:w[-1].numpy().item()
        warm_seconds=time.perf_counter()-t
        if not all(bool(tf.reduce_all(w[6])) for w in ws):raise RuntimeError('Invalid warmup state')
        t=time.perf_counter();ss=list(pool.map(call_s,ws))
        for s in ss:s[-1].numpy().item()
        sample_seconds=time.perf_counter()-t
        if not all(bool(tf.reduce_all(s.valid)) for s in ss):raise RuntimeError('Invalid retained state')
    samples=tf.concat([s[2] for s in ss],axis=1);stats=tf.concat([s[3] for s in ss],axis=1)
    warm_draws=tf.concat([w[4] for w in ws],axis=1);warm_stats=tf.concat([w[5] for w in ws],axis=1)
    io=time.perf_counter()
    for label,value in [('samples',samples),('telemetry',stats),('warmup',warm_draws),('warmup-telemetry',warm_stats)]:
        tensor_save(tf,folder/(label+'.tensor'),value)
    report=dict(warmup_seconds=warm_seconds,sample_seconds=sample_seconds,
        tuned_total_seconds=warm_seconds+sample_seconds,serialization_seconds=time.perf_counter()-io,
        epsilon=tf.concat([w[1] for w in ws],axis=0).numpy().tolist(),
        metric_factors=tf.concat([w[2] for w in ws],axis=0).numpy().tolist(),
        initial_and_boundary_probes=warm_stats[-1,:,8].numpy().tolist())
    return samples,stats,report

def stan_run(tf,starts,data,seed,args,folder):
    write(folder/'data.json',data);cmds=[]
    for chain,row in enumerate(starts.numpy().tolist()):
        init=folder/f'init-{chain}.json';write(init,dict(q=row))
        output=folder/f'chain-{chain}.csv';datafile=folder/'data.json'
        cmd=[str(args.stan.resolve()),'sample',f'num_warmup={args.warmup}',
            f'num_samples={args.draws}','save_warmup=1','thin=1','adapt','engaged=1',
            'delta=0.8','gamma=0.05','kappa=0.75','t0=10','init_buffer=75','term_buffer=50','window=25',
            'algorithm=hmc','engine=nuts',f'max_depth={args.depth}',
            'metric='+('dense_e' if data['kind']==1 else 'diag_e'),'stepsize=1','stepsize_jitter=0',
            'data',f'file={datafile}',f'init={init}',
            'random',f'seed={seed}',f'id={chain+1}','output',f'file={output}',
            'sig_figs=17','refresh=0']
        cmds.append(cmd)
    def run(item):
        chain,cmd=item;t=time.perf_counter()
        with (folder/f'chain-{chain}.log').open('w') as log:
            # Python wait(timeout=...) polls with increasing sleep intervals.
            # Blocking wait on a bounded external watchdog avoids quantizing
            # subsecond native timings in increments up to 50 milliseconds.
            subprocess.run(['timeout','300',*cmd],stdout=log,stderr=subprocess.STDOUT,check=True,
                           env={**os.environ,'STAN_NUM_THREADS':'1'})
        return time.perf_counter()-t
    t=time.perf_counter()
    with ThreadPoolExecutor(max_workers=4) as pool: chain_wall=list(pool.map(run,enumerate(cmds)))
    wall=time.perf_counter()-t;draws=[];traces=[];published=[];eps=[]
    for chain in range(4):
        text=(folder/f'chain-{chain}.csv').read_text()
        rows=list(csv.DictReader(line for line in text.splitlines() if line and not line.startswith('#')))
        if len(rows)!=args.warmup+args.draws:raise RuntimeError('Stan draw count mismatch')
        rows=rows[args.warmup:]
        draws.append([[float(r[f'q.{j+1}']) for j in range(data['d'])] for r in rows])
        traces.append([[float(r['accept_stat__']),0.,float(r['n_leapfrog__']),
            float(float(r['treedepth__'])>=args.depth),float(r['divergent__']),1.] for r in rows])
        published.append(dict(warmup=float(re.search(r'([0-9.eE+-]+) seconds [(]Warm-up[)]',text).group(1)),
                              sampling=float(re.search(r'([0-9.eE+-]+) seconds [(]Sampling[)]',text).group(1))))
        eps.append(float(rows[0]['stepsize__']))
    samples=tf.transpose(tf.constant(draws,tf.float64),[1,0,2]);stats=tf.transpose(tf.constant(traces,tf.float64),[1,0,2])
    report=dict(process_wall_seconds=wall,chain_wall_seconds=chain_wall,
        published_chain_seconds=published,epsilon=eps,commands=cmds,
        subprocess_watchdog=['timeout','300'],wait_mode='blocking waitpid',
        native_cache_check='Stan successful exit; endpoint score cache not exported',
        tuned_total_seconds=wall,wall_includes='process/init/adaptation/retained/CSV; common diagnostic time separate')
    return samples,stats,report

def density_check(tf,target,starts,data,args,folder):
    datafile=folder/'data.json';write(datafile,data);numeric=[];commands=[]
    # Stan 2.37 get_uparams_r flattens multi-point JSON column-major and then
    # reshapes row-major. A single vector avoids that external input mismatch.
    for i,point in enumerate(starts.numpy().tolist()):
        pointfile=folder/f'point-{i}.json';output=folder/f'log-prob-{i}.csv'
        write(pointfile,dict(params_r=point))
        cmd=[str(args.stan.resolve()),'log_prob',f'unconstrained_params={pointfile}',
             'jacobian=0','data',f'file={datafile}','output',f'file={output}','sig_figs=17']
        commands.append(cmd)
        with (folder/f'log-prob-{i}.log').open('w') as log:
            subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=60)
        rows=list(csv.reader(line for line in output.read_text().splitlines() if line and not line.startswith('#')))
        numeric.append([float(x) for x in rows[1]])
    @tf.function(input_signature=[tf.TensorSpec([4,target.dimension],tf.float64)],autograph=False,jit_compile=True)
    def score(q):
        with tf.GradientTape() as tape:tape.watch(q);v=target.log_prob(q)
        return v,tape.gradient(v,q)
    value,gradient=score(starts);native=tf.constant(numeric,tf.float64)
    ve=float(tf.reduce_max(tf.abs(native[:,0]-value)/(1.+tf.abs(value))))
    ge=float(tf.reduce_max(tf.abs(native[:,1:]-gradient)/(1.+tf.abs(gradient))))
    if ve>1e-10 or ge>1e-9:raise RuntimeError(f'Stan target mismatch {ve} {ge}')
    return dict(value_relative_error=ve,gradient_relative_error=ge,commands=commands)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',choices=['cpu','gpu'],default='cpu')
    p.add_argument('--layout',choices=['scalar','batch'],default='scalar')
    p.add_argument('--mode',choices=['calibrate','compare'],default='compare')
    p.add_argument('--warmup',type=int,default=1000);p.add_argument('--draws',type=int,default=2000)
    p.add_argument('--depth',type=int,default=10);p.add_argument('--seeds',default='20261027,20261028,20261029')
    p.add_argument('--targets',default=','.join(NAMES))
    p.add_argument('--stan',type=Path,default=ROOT/'docs/benchmarks/stan_targets')
    args=p.parse_args();args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=False)
    began=time.perf_counter()
    threads='1' if args.layout=='scalar' else '4'
    conf=base.parser().parse_args(['--output',str(args.output),'--device',args.device,'--gpu','0','--threads',threads])
    tf,tfp,memory=base.setup(conf)
    manifest=dict(schema='bayesfilter.stan_transplant.v1',command=sys.argv,plan=PLAN,result_note=RESULT,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        dirty_worktree=True,environment=sys.executable,tensorflow=tf.__version__,tfp=tfp.__version__,
        python=platform.python_version(),device=args.device,gpu_intentionally_hidden=args.device=='cpu',
        memory_policy=memory,tf32=False,jit_compile=True,layout=args.layout,intra_op_threads=int(threads),
        stan_version='2.37.0',seeds=args.seeds,compiled={},results=[],
        data_version='analytic target definitions plus stateless regression seed 20261027,61',
        device_trust_basis='escalated_gpu_worker' if args.device=='gpu' else 'explicit_cpu_reference',
        startup_framework_seconds=time.perf_counter()-began,source_hashes={})
    archive=args.output/'source';archive.mkdir()
    for path in [Path(__file__),Path(base.__file__),ROOT/'docs/benchmarks/stan_nuts_warmup.py',
                 ROOT/'docs/benchmarks/nuts_benchmark_targets.py',ROOT/'docs/benchmarks/stan_targets.stan',
                 ROOT/'bayesfilter/inference/hmc_ess.py',ROOT/'bayesfilter/inference/hmc_posterior_diagnostics.py']:
        manifest['source_hashes'][str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
        shutil.copyfile(path,archive/(path.name+'.txt'))
    write(args.output/'manifest.json',manifest)
    try:
        with tf.device('/'+args.device.upper()+':0'):
            for name in args.targets.split(','):
                target=make_target(tf,name);folder=args.output/name;folder.mkdir()
                warm,sample,compiled=tf_functions(tf,tfp,target,args,folder)
                manifest['compiled'][name]=compiled;q,data=inputs(tf,target,20261020)
                if args.mode=='compare':
                    manifest.setdefault('density_checks',{})[name]=density_check(tf,target,q,data,args,folder)
                for index,seed in enumerate(map(int,args.seeds.split(','))):
                    starts,data=inputs(tf,target,seed);record=dict(target=name,seed=seed)
                    order=['tfp','stan'] if index%2==0 else ['stan','tfp']
                    if args.mode=='calibrate':order=['tfp']
                    for engine in order:
                        out=folder/f'{engine}-{seed}';out.mkdir()
                        if engine=='tfp':draws,stats,report=tf_run(tf,warm,sample,starts,seed,args,out)
                        else:draws,stats,report=stan_run(tf,starts,data,seed,args,out)
                        if args.mode=='compare':
                            with tf.device('/CPU:0'):report['diagnostics']=assess(tf,target,tf.identity(draws),tf.identity(stats))
                            if engine=='stan':
                                report['diagnostics']['finite_cache']=None
                                report['diagnostics']['cache_note']='Native internal cache is not exported; finite draws and successful native execution checked.'
                        record[engine]=report;write(out/'result.json',report)
                        print(json.dumps(dict(target=name,seed=seed,engine=engine,seconds=report['tuned_total_seconds'],
                            passed=report.get('diagnostics',{}).get('posterior_screen_pass'))),flush=True)
                    manifest['results'].append(record);write(args.output/'manifest.json',manifest)
        manifest['status']='complete'
    except Exception as exc:
        manifest['status']='failed';manifest['error']=str(exc);manifest['traceback']=traceback.format_exc()
        raise
    finally:
        manifest['wall_seconds']=time.perf_counter()-began
        manifest['host_max_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if args.device=='gpu':manifest['allocator']=tf.config.experimental.get_memory_info('GPU:0')
        write(args.output/'manifest.json',manifest)

if __name__=='__main__': main()
