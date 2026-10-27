"""Direct Stan-port / TFP / native comparison; kernels are FP64 and XLA.

Research reporting only. Fixed-budget timings are descriptive; validity screens
must pass before an efficiency comparison. Compilation and host costs are
reported separately, and all warmup output is preserved and discarded.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import subprocess
import sys
import time
import traceback

import benchmark_nuts_xla as base
import benchmark_stan_transplant as old
from nuts_benchmark_targets import make_target,NAMES
import stan_nuts as port

ROOT=Path(__file__).resolve().parents[2]
PLAN='docs/plans/bayesfilter-full-stan-nuts-port-execution-2026-10-27.md'
RESULT='docs/plans/bayesfilter-full-stan-nuts-port-result-2026-10-27.md'

def inputs(tf,target,seed,scale=1.):
    if target.name!='regression64':
        q,data=old.inputs(tf,target,seed)
    else:
        q=target.from_reference(tf.random.stateless_normal([4,64],[seed,321],dtype=tf.float64))
        data=dict(d=64,kind=4,n=4096,bend=1.,inverse_transform=tf.eye(64,dtype=tf.float64).numpy().tolist(),
                  design=tf.random.stateless_normal([4096,64],[20261027,61],dtype=tf.float64).numpy().tolist())
    if scale!=1.:q=target.from_reference(scale*target.to_reference(q))
    return q,data

def direct_functions(tf,target,args,folder):
    c=4 if args.layout=='batch' else 1;d=target.dimension
    cfg=port.Config(d,chains=c,warmup=args.warmup,draws=args.draws,
                    metric='dense' if target.name=='rotated_gaussian32' else 'diag',depth=args.depth)
    init,warm,sample=port.make_functions(tf,target.log_prob,cfg)
    @tf.function(input_signature=init.input_signature,autograph=False,jit_compile=True)
    def initialize_timed(q,present,seed):
        result=init(q,present,seed)
        return (*result,tf.reduce_sum(result[0]))
    q,_=inputs(tf,target,20260901);q=q[:c]
    seeds=tf.constant([[4*20260901+i,0] for i in range(c)],tf.int32)
    metric=tf.eye(d,batch_shape=[c],dtype=tf.float64) if cfg.metric=='dense' else tf.ones([c,d],tf.float64)
    epsilon=tf.ones([c],tf.float64)
    initialized,ireport=old.compile_fn(tf,initialize_timed,(q,tf.ones_like(q,tf.bool),seeds),folder,'initialize',args.device)
    warmed,wreport=old.compile_fn(tf,warm,(initialized[0],epsilon,metric,initialized[1]),folder,'warmup',args.device)
    port.require_valid(warmed)
    sampled,sreport=old.compile_fn(tf,sample,warmed[:4],folder,'sample',args.device)
    port.require_valid(sampled)
    return (init,warm,sample,epsilon,metric),dict(config=asdict(cfg),initialize=ireport,warmup=wreport,sample=sreport)

def direct_run(tf,functions,starts,seed,args,folder):
    init,warm,sample,e,m=functions
    bank=[(starts,tf.constant([[4*seed+i,0] for i in range(4)],tf.int32))] if args.layout=='batch' else [
        (starts[i:i+1],tf.constant([[4*seed+i,0]],tf.int32)) for i in range(4)]
    def call_w(pair):
        with tf.device('/'+args.device.upper()+':0'):
            q,r,ok,_=init(pair[0],tf.ones_like(pair[0],tf.bool),pair[1])
            if not bool(tf.reduce_all(ok)):raise RuntimeError('Invalid initialization')
            return port.require_valid(warm(q,e,m,r))
    def call_s(w):
        with tf.device('/'+args.device.upper()+':0'):return port.require_valid(sample(*w[:4]))
    with ThreadPoolExecutor(max_workers=4 if args.layout=='scalar' else 1) as pool:
        t=time.perf_counter();ws=list(pool.map(call_w,bank))
        for w in ws:w.completion_token.numpy()
        warming=time.perf_counter()-t
        t=time.perf_counter();ss=list(pool.map(call_s,ws))
        for s in ss:s.completion_token.numpy()
        sampling=time.perf_counter()-t
    samples=tf.concat([s.draws for s in ss],axis=1);stats=tf.concat([s.telemetry for s in ss],axis=1)
    warm_draws=tf.concat([w.draws for w in ws],axis=1);warm_stats=tf.concat([w.telemetry for w in ws],axis=1)
    t=time.perf_counter()
    for name,value in [('samples',samples),('telemetry',stats),('warmup',warm_draws),('warmup-telemetry',warm_stats)]:
        old.tensor_save(tf,folder/(name+'.tensor'),value)
    report=dict(warmup_seconds=warming,sample_seconds=sampling,tuned_total_seconds=warming+sampling,
        serialization_seconds=time.perf_counter()-t,epsilon=tf.concat([w.step_size for w in ws],0).numpy().tolist(),
        inverse_metric=tf.concat([w.inverse_metric for w in ws],0).numpy().tolist(),
        initial_and_boundary_probes=warm_stats[-1,:,11].numpy().tolist(),
        numerical_timing_includes='initialization validation, step search, adaptation, retained sampling; host serialization separate')
    return samples,stats,report

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',choices=['cpu','gpu'],default='gpu')
    p.add_argument('--layout',choices=['scalar','batch'],default='batch')
    p.add_argument('--mode',choices=['calibrate','compare'],default='compare')
    p.add_argument('--engines',default='direct,tfp,stan')
    p.add_argument('--targets',default=','.join((*NAMES,'regression64')))
    p.add_argument('--seeds',default='20261027,20261028,20261029')
    p.add_argument('--warmup',type=int,default=1000);p.add_argument('--draws',type=int,default=2000)
    p.add_argument('--depth',type=int,default=10)
    p.add_argument('--init-scale',type=float,default=1.)
    p.add_argument('--stan',type=Path,default=ROOT/'docs/benchmarks/stan_targets')
    args=p.parse_args();args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=False)
    began=time.perf_counter();threads=1 if args.layout=='scalar' else 4
    conf=base.parser().parse_args(['--output',str(args.output),'--device',args.device,'--gpu','0','--threads',str(threads)])
    tf,tfp,memory=base.setup(conf)
    engines=args.engines.split(',')
    if not set(engines)<={'direct','tfp','stan','fixed'}:raise ValueError('Unknown engine')
    manifest=dict(schema='bayesfilter.stan_full_port.v1',plan=PLAN,result_file=RESULT,command=sys.argv,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        environment=sys.executable,python=platform.python_version(),tensorflow=tf.__version__,tfp=tfp.__version__,
        stan_version='2.37.0',device=args.device,device_trust_basis='escalated_gpu_worker' if args.device=='gpu' else 'explicit_cpu_reference',
        gpu_intentionally_hidden=args.device=='cpu',memory_policy=memory,jit_compile=True,tf32=False,
        layout=args.layout,intra_op_threads=threads,config=vars(args)|{'output':str(args.output),'stan':str(args.stan)},
        seeds=args.seeds,data_version='analytic fixtures; regression stateless seed 20261027,61',
        compile={},results=[],source_hashes={},startup_framework_seconds=time.perf_counter()-began)
    archive=args.output/'source';archive.mkdir()
    sources=[Path(__file__),Path(port.__file__),Path(old.__file__),Path(base.__file__),ROOT/'docs/benchmarks/nuts_benchmark_targets.py',
             ROOT/'docs/benchmarks/stan_nuts_warmup.py',ROOT/'docs/benchmarks/stan_targets.stan',ROOT/'docs/benchmarks/stan_targets',
             ROOT/'bayesfilter/inference/hmc_ess.py',ROOT/'bayesfilter/inference/hmc_posterior_diagnostics.py']
    for source in sources:
        manifest['source_hashes'][str(source.relative_to(ROOT))]=hashlib.sha256(source.read_bytes()).hexdigest()
        if source.suffix:shutil.copyfile(source,archive/(source.name+'.txt'))
    old.write(args.output/'manifest.json',manifest)
    try:
        with tf.device('/'+args.device.upper()+':0'):
            for name in args.targets.split(','):
                target=make_target(tf,name);folder=args.output/name;folder.mkdir()
                functions={};manifest['compile'][name]={}
                if 'direct' in engines:
                    f=folder/'direct-compile';f.mkdir()
                    functions['direct'],manifest['compile'][name]['direct']=direct_functions(tf,target,args,f)
                if 'tfp' in engines:
                    f=folder/'tfp-compile';f.mkdir()
                    w,s,report=old.tf_functions(tf,tfp,target,args,f)
                    functions['tfp']=(w,s);manifest['compile'][name]['tfp']=report
                q,data=inputs(tf,target,20261020,args.init_scale)
                if args.mode=='compare':
                    manifest.setdefault('density_checks',{})[name]=old.density_check(tf,target,q,data,args,folder)
                for index,seed in enumerate(map(int,args.seeds.split(','))):
                    starts,data=inputs(tf,target,seed,args.init_scale)
                    record=dict(target=name,seed=seed,init_scale=args.init_scale)
                    order=engines[index%len(engines):]+engines[:index%len(engines)]
                    for engine in order:
                        out=folder/f'{engine}-{seed}';out.mkdir()
                        if engine=='direct':draws,stats,report=direct_run(tf,functions['direct'],starts,seed,args,out)
                        elif engine=='tfp':draws,stats,report=old.tf_run(tf,*functions['tfp'],starts,seed,args,out)
                        elif engine=='stan':draws,stats,report=old.stan_run(tf,starts,data,seed,args,out)
                        else:
                            from benchmark_full_stan_hmc import run_fixed
                            draws,stats,report=run_fixed(tf,target,starts,seed,args,out)
                        if args.mode=='compare' and draws is not None:
                            with tf.device('/CPU:0'):report['diagnostics']=old.assess(tf,target,tf.identity(draws),tf.identity(stats))
                            if engine=='stan':report['diagnostics']['finite_cache']=None
                        record[engine]=report;old.write(out/'result.json',report)
                        print(json.dumps(dict(target=name,seed=seed,engine=engine,seconds=report['tuned_total_seconds'],
                            passed=report.get('diagnostics',{}).get('posterior_screen_pass'))),flush=True)
                    manifest['results'].append(record);old.write(args.output/'manifest.json',manifest)
        manifest['status']='complete'
    except Exception as exc:
        manifest['status']='failed';manifest['error']=str(exc);manifest['traceback']=traceback.format_exc()
        raise
    finally:
        manifest['wall_seconds']=time.perf_counter()-began
        manifest['host_max_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if args.device=='gpu':manifest['allocator']=tf.config.experimental.get_memory_info('GPU:0')
        old.write(args.output/'manifest.json',manifest)

if __name__=='__main__':main()
