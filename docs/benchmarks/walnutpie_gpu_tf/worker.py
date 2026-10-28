"""Engineering validation/benchmark worker. NumPy only in oracle diagnostics."""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import resource
import statistics
import subprocess
import time
from .core import Config, Kernel
from .api import make_functions, initial_state, run, state_spec


def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False))

def setup(device):
    if os.environ.get('TF_FORCE_GPU_ALLOW_GROWTH')!='true':raise RuntimeError('memory growth required before import')
    if device=='cpu' and os.environ.get('CUDA_VISIBLE_DEVICES')!='-1':raise RuntimeError('CPU reference must hide GPUs')
    import tensorflow as tf
    tf.config.set_soft_device_placement(False)
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.experimental.enable_tensor_float_32_execution(False)
    physical=tf.config.list_physical_devices('GPU');growth=[]
    for gpu in physical:
        tf.config.experimental.set_memory_growth(gpu,True)
        flag=tf.config.experimental.get_memory_growth(gpu)
        if flag is not True:raise RuntimeError('growth verification failed')
        growth.append(dict(name=gpu.name,growth=flag,details=tf.config.experimental.get_device_details(gpu)))
    if device=='gpu' and not physical:raise RuntimeError('trusted GPU missing')
    return tf,dict(python=platform.python_version(),tensorflow=tf.__version__,device=device,gpus=growth,
        gpu_hidden=device=='cpu',jit_compile=True,tf32=False,xla_flags=os.environ.get('XLA_FLAGS',''),
        trust_basis='trusted_escalated_execution',memory_policy='memory_growth_before_initialization')


def model_for(tf,kind):
    if kind==0:return lambda q:-.5*tf.reduce_sum(q*q,-1)
    if kind==1:
        def model(q):
            x,y=q[:,::2],q[:,1::2];z=y-.3*(x*x-1.)
            return -.5*tf.reduce_sum(x*x+z*z,-1)
        return model
    if kind==2:
        def model(q):
            v=q[:,0];return -v*v/18.-.5*(q.shape[-1]-1)*v-.5*tf.exp(-v)*tf.reduce_sum(q[:,1:]**2,-1)
        return model
    def regression(q,x,y):
        r=tf.matmul(q,x,transpose_b=True)-y
        return -.5*tf.reduce_sum(q*q,-1)-.5*tf.reduce_sum(r*r,-1)/tf.cast(tf.shape(x)[0],tf.float64)
    return regression


def synchronize(tf,result):
    from tensorflow.python.eager import context
    context.async_wait()
    return result


def timed(tf,fn,*args):
    start=time.perf_counter();r=fn(*args);synchronize(tf,r)
    return r,time.perf_counter()-start


def graph_report(fn,args,path):
    concrete=fn.get_concrete_function()
    graph=concrete.graph.as_graph_def()
    nodes=list(graph.node)+[n for f in graph.library.function for n in f.node_def]
    forbidden=[n.op for n in nodes if any(s in n.op for s in ('PyFunc','BfNative','PyCall'))]
    if forbidden:raise AssertionError(forbidden)
    hlo=fn.experimental_get_compiler_ir(*args)(stage='optimized_hlo');path.write_text(hlo)
    calls=sorted(set(re.findall(r'custom_call_target="([^"]+)"',hlo)))
    if any(any(s in x.lower() for s in ('callback','native_sampler','host_compute')) for x in calls):raise AssertionError(calls)
    if fn.experimental_get_tracing_count()!=1:raise AssertionError('retracing')
    return dict(sha256=hashlib.sha256(hlo.encode()).hexdigest(),bytes=len(hlo),custom_calls=calls,
        trace_count=fn.experimental_get_tracing_count(),nodes=len(nodes),hlo=str(path),
        while_count=len(re.findall(r'\bwhile\(',hlo)),fusion_count=len(re.findall(r'\bfusion\(',hlo)),
        model_gradient_inside_loop='gradient_tape' in hlo and 'while' in hlo)


def validate(tf,args,report):
    import numpy as np  # Explicit independent-oracle diagnostics only.
    output=subprocess.check_output([str(args.oracle)],text=True)
    (args.output/'oracle.jsonl').write_text(output)
    rows=[json.loads(line) for line in output.splitlines()]
    checks=[];coverage=dict(macro_accept=0,macro_reject=0,refinement=0,reverse_reject=0,transition_depths=[])
    def close(actual,expected):np.testing.assert_allclose(actual,expected,rtol=2e-11,atol=2e-11)
    for kind in range(4):
        rr=[r for r in rows if r['kind']==kind and r['type']=='transition']
        c,d=len(rr),len(rr[0]['q']);cfg=Config(c,d)
        target=model_for(tf,kind)
        if kind==3:
            x=tf.constant([[.1*((i*7+j*3)%11-5) for j in range(d)] for i in range(8)],tf.float64)
            y=tf.constant([.1*(i-3) for i in range(8)],tf.float64)
            target=lambda q: model_for(tf,3)(q,x,y)
        k=Kernel(tf,target,cfg,tape=True)
        q=tf.constant([r['q'] for r in rr],tf.float64);inv=tf.constant([r['inverse_mass'] for r in rr],tf.float64)
        eps=tf.constant([r['epsilon'] for r in rr],tf.float64);minimum=tf.constant([r['minimum'] for r in rr],tf.int32)
        seeds=tf.constant([[r['key'],0] for r in rr],tf.int32)
        sig=[tf.TensorSpec(x.shape,x.dtype) for x in (q,inv,eps,minimum,seeds)]
        @tf.function(input_signature=sig,jit_compile=True,autograph=False)
        def transition(q,inv,eps,minimum,seeds):
            lp,g=k.value_gradient(q)
            return k.transition(q,lp,g,eps,inv,minimum,seeds,k.adam_init(eps),adapt=True)
        t=transition(q,inv,eps,minimum,seeds)
        for field,ref in [('q','out'),('g','gradient'),('lp','lp')]:close(getattr(t,field).numpy(),[r[ref] for r in rr])
        for field,ref in [('depth','depth'),('evaluations','evaluations'),('macros','macros')]:
            np.testing.assert_array_equal(getattr(t,field).numpy(),[r[ref] for r in rr])
        np.testing.assert_array_equal(t.seed.numpy()[:,1],[r['counter'] for r in rr])
        close(tf.exp(t.adam.theta).numpy(),[r['adapted_epsilon'] for r in rr])
        coverage['transition_depths']+=t.depth.numpy().tolist()
        mr=[r for r in rows if r['kind']==kind and r['type']=='macro']
        @tf.function(input_signature=sig,jit_compile=True,autograph=False)
        def macro(q,inv,eps,minimum,seeds):
            lp,g=k.value_gradient(q);z,_=k.random(seeds,k.active(),normal=True)
            p=k.point(q,z/tf.sqrt(inv),g,lp,inv)
            return k.macro(p,inv,eps,minimum,k.active(),k.adam_init(eps),True)
        point,ok,rf,ev,adam=macro(q,inv,eps,minimum,seeds)
        np.testing.assert_array_equal(ok.numpy(),[r['ok'] for r in mr]);np.testing.assert_array_equal(ev.numpy(),[r['evaluations'] for r in mr])
        close(tf.exp(adam.theta).numpy(),[r['adapted_epsilon'] for r in mr])
        for j,r in enumerate(mr):
            if all(v is not None for v in r['out']):close(point.q.numpy()[j],r['out'])
        coverage['macro_accept']+=int(tf.reduce_sum(tf.cast(ok,tf.int32)).numpy())
        coverage['macro_reject']+=int(tf.reduce_sum(tf.cast(~ok,tf.int32)).numpy())
        coverage['refinement']+=int(tf.reduce_sum(tf.cast(ev>minimum,tf.int32)).numpy())
        coverage['reverse_reject']+=int(tf.reduce_sum(tf.cast(rf,tf.int32)).numpy())
        @tf.function(input_signature=[sig[0],sig[-1]],jit_compile=True,autograph=False)
        def initialize(q,seeds):return initial_state(k,q,seeds)
        init=initialize(q,seeds)
        ir=[r for r in rows if r['kind']==kind and r['type']=='initialize']
        close(init.epsilon.numpy(),[r['adapted_epsilon'] for r in ir]);assert bool(tf.reduce_all(init.valid).numpy())
        checks.append(dict(kind=kind,transitions=len(rr),macros=len(mr),initializations=len(ir),
            graph=graph_report(transition,(q,inv,eps,minimum,seeds),args.output/f'transition-{kind}.hlo.txt')))
    # Full upstream class adaptation against one-step compiled continuation.
    for kind,target_macro in [(kind,target) for kind in range(3) for target in (15.,1.)]:
        cfg=Config(1,4,warmup=1,draws=1,target_macro_steps=target_macro);k=Kernel(tf,model_for(tf,kind),cfg,tape=True)
        @tf.function(input_signature=[tf.TensorSpec([1,4],tf.float64),tf.TensorSpec([1,2],tf.int32)],jit_compile=True,autograph=False)
        def init(q,seed):return initial_state(k,q,seed)
        @tf.function(input_signature=[state_spec(tf,cfg)],jit_compile=True,autograph=False)
        def warm(s):return run(k,s,1,True)
        @tf.function(input_signature=[state_spec(tf,cfg)],jit_compile=True,autograph=False)
        def sample(s):return run(k,s,1,False)
        state=init(tf.constant([[-.4,-.1,.2,.5]],tf.float64),tf.constant([[7,0]],tf.int32))
        for r in [r for r in rows if r['kind']==kind and r['type']=='lifecycle' and r['target_macro_steps']==target_macro]:
            result=(warm if r['warmup'] else sample)(state);state=result.state
            close(state.q.numpy()[0],r['q']);close(state.epsilon.numpy()[0],r['epsilon'])
            close(state.inverse_mass.numpy()[0],r['inverse_mass'])
            assert int(state.minimum.numpy()[0])==r['minimum'];assert int(state.seed.numpy()[0,1])==r['counter']
            if target_macro==1.:
                # Controlled source replay: remove accumulated nonlinear
                # trajectory amplification without relaxing local tolerances.
                native_q=tf.constant([r['q']],tf.float64)
                native_lp,native_g=k.value_gradient(native_q)
                native_epsilon=tf.constant([r['epsilon']],tf.float64)
                state=state._replace(q=native_q,lp=native_lp,g=native_g,
                    epsilon=native_epsilon,inverse_mass=tf.constant([r['inverse_mass']],tf.float64),
                    adam=state.adam._replace(theta=tf.math.log(native_epsilon)))
        checks.append(dict(kind=kind,target_macro_steps=target_macro,lifecycle_transitions=40,
            controlled_state_replay=target_macro==1.,final_minimum=int(state.minimum.numpy()[0])))
    # Public runtime-data composition, frozen continuation, chain isolation.
    cfg=Config(4,4,warmup=8,draws=6);model=lambda q,shift:-.5*tf.reduce_sum((q-shift)**2,-1)
    funcs=make_functions(tf,model,cfg,[tf.TensorSpec([4],tf.float64)])
    q=tf.constant([[.1*(j+1)+i*.03 for j in range(4)] for i in range(4)],tf.float64)
    seeds=tf.constant([[100+i,0] for i in range(4)],tf.int32);shift=tf.zeros([4],tf.float64)
    fit=funcs.fit(q,seeds,shift);assert bool(tf.reduce_all(fit.valid).numpy())
    init=funcs.initialize(q,seeds,shift);warm=funcs.warmup(init,shift);sample=funcs.sample(warm.state,shift)
    close(fit.sampling.draws.numpy(),sample.draws.numpy())
    long=make_functions(tf,model,replace(cfg,draws=12),[tf.TensorSpec([4],tf.float64)])
    full=long.sample(warm.state,shift);second=funcs.sample(sample.state,shift)
    close(tf.concat([sample.draws,second.draws],0).numpy(),full.draws.numpy())
    changed=funcs.fit(q,seeds,shift+1.)
    assert not np.allclose(changed.sampling.draws.numpy(),fit.sampling.draws.numpy())
    isolated=funcs.fit(tf.tensor_scatter_nd_add(q,[[1,0]],[1.]),seeds,shift)
    close(tf.gather(isolated.sampling.draws,[0,2,3],axis=1).numpy(),tf.gather(fit.sampling.draws,[0,2,3],axis=1).numpy())
    # Retained cached log density and gradient agree with a fresh target call.
    lp,g=funcs.value_gradient(sample.state.q,shift)
    close(lp.numpy(),sample.state.lp.numpy());close(g.numpy(),sample.state.g.numpy())
    bad=funcs.initialize(q*tf.constant(float('nan'),tf.float64),seeds,shift)
    assert not bool(tf.reduce_any(bad.valid).numpy())
    duplicate=funcs.initialize(q,tf.constant([[1,0]]*4,tf.int32),shift)
    assert not bool(tf.reduce_any(duplicate.valid).numpy())
    # Warmup continuation must preserve Adam cadence and discount iteration.
    half=make_functions(tf,model,replace(cfg,warmup=4),[tf.TensorSpec([4],tf.float64)])
    first_half=half.warmup(init,shift)
    continued=half.warmup(first_half.state,shift)
    for a,b in zip(tf.nest.flatten(continued.state),tf.nest.flatten(warm.state)):
        close(a.numpy(),b.numpy())
    # Compilation rejects CPU/Python callback models rather than falling back.
    def callback_target(x):
        value=tf.py_function(lambda y:-.5*tf.reduce_sum(y*y,-1),[x],tf.float64)
        return tf.ensure_shape(value,[4])
    callback=make_functions(tf,callback_target,cfg)
    try:
        callback.value_gradient(q)
    except (tf.errors.InvalidArgumentError,tf.errors.UnimplementedError,ValueError):
        pass
    else:
        raise AssertionError('Python callback unexpectedly admitted to XLA')
    report.update(checks=checks,coverage=coverage,api_checks=['full_fit_composition','runtime_model_data',
        'sample_continuation','warmup_continuation','chain_isolation','cache_agreement',
        'invalid_initial_state','duplicate_rng_rejected','callback_rejected'],
        full_fit_graph=graph_report(funcs.fit,(q,seeds,shift),args.output/'full-fit.hlo.txt'),
        output_device=fit.sampling.draws.device)
    assert coverage['macro_reject'] and coverage['reverse_reject'] and coverage['refinement']


def fixture(tf,name,args):
    kind={'gaussian':0,'banana':1,'funnel':2,'regression':3}[name]
    d=args.regression_dimension if kind==3 else 10
    q=tf.reshape(tf.cast(tf.range(4*d),tf.float64),[4,d])
    q=.3*tf.sin(q+1.)
    if kind==3:
        n=args.regression_size
        i=tf.cast(tf.range(n),tf.float64)[:,None]+1.;j=tf.cast(tf.range(d),tf.float64)[None,:]+1.
        x=tf.sin(i*j*.013)+tf.cos(i*(j+1.)*.007);y=tf.zeros([n],tf.float64)
        data=(x,y);specs=tuple(tf.TensorSpec(v.shape,v.dtype) for v in data)
    else:n=0;data=();specs=()
    return kind,d,n,q,model_for(tf,kind),data,specs


def diagnostics(tf,draws):
    # Reporting-only post-run diagnostics from the repository TF authority.
    from bayesfilter.inference.hmc_posterior_diagnostics import rank_normalized_split_rhat, rank_normalized_bulk_tail_ess
    # Diagnostic boundary: these summaries are not execution/admission controls.
    with tf.device('/CPU:0'):
        x=tf.transpose(draws,[1,0,2])
        r=rank_normalized_split_rhat(x);e=rank_normalized_bulk_tail_ess(x)
        return dict(max_rhat=float(tf.reduce_max(r['maximum']).numpy()),
            min_bulk_ess=float(tf.reduce_min(e['bulk']).numpy()),
            min_tail_ess=float(tf.reduce_min(e['tail']).numpy()),
            mean=tf.reduce_mean(x,[0,1]).numpy().tolist())


def benchmark(tf,args,report):
    report['targets']=[]
    for name in args.targets.split(','):
        kind,d,n,q,model,data,specs=fixture(tf,name,args)
        warm=min(args.warmup,100) if kind==3 else args.warmup
        draws=min(args.draws,100) if kind==3 else args.draws
        cfg=Config(4,d,warmup=warm,draws=draws)
        f=make_functions(tf,model,cfg,specs)
        seeds=tf.constant([[20261028+i,0] for i in range(4)],tf.int32)
        r,compile_time=timed(tf,f.fit,q,seeds,*data)
        assert bool(tf.reduce_all(r.valid).numpy())
        times=[]
        for _ in range(7):r,t=timed(tf,f.fit,q,seeds,*data);times.append(t)
        graph=graph_report(f.fit,(q,seeds,*data),args.output/(name+'.hlo.txt'))
        path=args.output/(name+'-input.txt')
        with path.open('w') as handle:
            handle.write(f'{kind} {d} {n} 4 {warm} {draws} {cfg.max_depth}\n')
            for row in q.numpy().tolist():handle.write(' '.join(map(repr,row))+'\n')
            for value in data:
                for row in value.numpy().tolist():handle.write((' '.join(map(repr,row)) if isinstance(row,list) else repr(row))+'\n')
        native=json.loads(subprocess.check_output([str(args.oracle),str(path)],text=True))
        write(args.output/(name+'-native.json'),native)
        write(args.output/(name+'-draws.json'),r.sampling.draws.numpy().tolist())
        native_draws=tf.transpose(tf.constant([c['draws'] for c in native['chains']],tf.float64),[1,0,2])
        tf_diagnostic=diagnostics(tf,r.sampling.draws)
        native_diagnostic=diagnostics(tf,native_draws)
        if kind==2:
            tf_diagnostic['fraction_v_below_minus3']=float(tf.reduce_mean(tf.cast(r.sampling.draws[:,:,0]<-3.,tf.float64)).numpy())
            native_diagnostic['fraction_v_below_minus3']=float(tf.reduce_mean(tf.cast(native_draws[:,:,0]<-3.,tf.float64)).numpy())
        entry=dict(name=name,config=asdict(cfg),data_shape=[n,d],first_call_seconds=compile_time,
            steady_seconds=times,median_seconds=statistics.median(times),graph=graph,
            native_serial_seconds=native['serial_seconds'],native_parallel_seconds=native['parallel_seconds'],
            native_serial_median=statistics.median(native['serial_seconds']),
            native_parallel_median=statistics.median(native['parallel_seconds']),
            native_evaluations=[x['evaluations'] for x in native['chains']],
            tf_transition_evaluations=(tf.reduce_sum(r.warmup.evaluations,axis=0)+
                tf.reduce_sum(r.sampling.evaluations,axis=0)).numpy().tolist(),
            tf_macro_failures=int(tf.reduce_sum(tf.cast(r.sampling.failed,tf.int32)).numpy()),
            tf_reverse_failures=int(tf.reduce_sum(tf.cast(r.sampling.reverse_failed,tf.int32)).numpy()),
            epsilon=r.sampling.state.epsilon.numpy().tolist(),device=r.sampling.draws.device,
            tf_diagnostics=tf_diagnostic,native_diagnostics=native_diagnostic,
            retained_valid=True,inference='descriptive engineering timings; no posterior or ranking promotion')
        report['targets'].append(entry);write(args.output/'result.json',report)
        print(json.dumps({k:v for k,v in entry.items() if k in ['name','first_call_seconds','median_seconds','native_parallel_median']}),flush=True)
    if args.device=='gpu':report['allocator']=tf.config.experimental.get_memory_info('GPU:0')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device',choices=['cpu','gpu'],required=True);p.add_argument('--mode',choices=['validate','benchmark'],required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--oracle',type=Path,required=True)
    p.add_argument('--targets',default='gaussian,banana,funnel,regression');p.add_argument('--warmup',type=int,default=500);p.add_argument('--draws',type=int,default=500)
    p.add_argument('--regression-size',type=int,default=4096);p.add_argument('--regression-dimension',type=int,default=128)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    tf,report=setup(args.device);report['mode']=args.mode
    begin=time.monotonic()
    with tf.device('/GPU:0' if args.device=='gpu' else '/CPU:0'):
        (validate if args.mode=='validate' else benchmark)(tf,args,report)
    report['wall_seconds']=time.monotonic()-begin;report['status']='passed'
    report['peak_host_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if args.device=='gpu':report['allocator']=tf.config.experimental.get_memory_info('GPU:0')
    write(args.output/'result.json',report);print(json.dumps({'status':'passed','mode':args.mode,'wall_seconds':report['wall_seconds']}))

if __name__=='__main__':main()
