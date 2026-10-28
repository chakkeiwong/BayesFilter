"""Paired execution measurements; no posterior or default promotion.

All executable numerical kernels use TF/XLA or the independent C++ reference.
Host standard-library calculations summarize timings only.
"""
import argparse
from dataclasses import replace,asdict
import hashlib
import json
import math
from pathlib import Path
import resource
import statistics
import subprocess
import time
from .api import make_functions
from .core import Config
from .worker import setup,timed,write,graph_report,model_for

SEEDS=[20261029+1000*i for i in range(7)]


def ratios_summary(blocks):
    logs=[statistics.median(math.log(a/b) for a,b in block) for block in blocks]
    mean=statistics.mean(logs)
    half=2.446912*statistics.stdev(logs)/math.sqrt(7)
    return dict(geometric_speed_ratio=math.exp(mean),interval95=[math.exp(mean-half),math.exp(mean+half)],
                seed_log_ratios=logs,scope='paired seed-block runtime; no sampler-efficiency ranking')


def sufficient_model(tf,q,gram,linear,constant):
    return -.5*tf.reduce_sum(q*q,-1)-.5*tf.reduce_sum(q*tf.matmul(q,gram),-1)+tf.linalg.matvec(q,linear)-.5*constant


def fixture(tf,name,chains):
    d=128 if name in ('regression','sufficient','logistic') else 10
    q=.3*tf.sin(tf.reshape(tf.cast(tf.range(chains*d),tf.float64),[chains,d])+1.)
    kind={'gaussian':0,'banana':1,'funnel':2,'regression':3,'logistic':4,'sufficient':5}[name]
    if kind<3:return q,model_for(tf,kind),(),(),kind
    i=tf.cast(tf.range(4096),tf.float64)[:,None]+1.;j=tf.cast(tf.range(d),tf.float64)[None,:]+1.
    x=tf.sin(i*j*.013)+tf.cos(i*(j+1.)*.007)
    y=tf.cast(tf.range(4096)%2,tf.float64) if kind==4 else tf.zeros([4096],tf.float64)
    raw=(x,y)
    if kind==3:model=model_for(tf,3)
    elif kind==4:
        def model(q,x,y):
            z=tf.matmul(q,x,transpose_b=True)
            return -.5*tf.reduce_sum(q*q,-1)-tf.reduce_mean(tf.nn.softplus(z)-y*z,-1)
    else:
        def model(q,gram,linear,constant):
            return sufficient_model(tf,q,gram,linear,constant)
        @tf.function(input_signature=[tf.TensorSpec(x.shape,tf.float64),tf.TensorSpec(y.shape,tf.float64)],jit_compile=True,autograph=False)
        def prepare(x,y):
            n=tf.cast(tf.shape(x)[0],tf.float64)
            return tf.matmul(x,x,transpose_a=True)/n,tf.linalg.matvec(x,y,transpose_a=True)/n,tf.reduce_sum(y*y)/n
        data,first=timed(tf,prepare,x,y);data,steady=timed(tf,prepare,x,y)
        return q,model,data,raw,kind,dict(first_call_seconds=first,steady_seconds=steady)
    return q,model,raw,raw,kind


def unpack_fixture(tf,name,chains):
    f=fixture(tf,name,chains)
    return (*f[:5],f[5] if len(f)==6 else None)


def assert_equal_fit(tf,a,b):
    # Diagnostic host boundary, deliberately outside synchronized intervals.
    for x,y in zip(tf.nest.flatten(a),tf.nest.flatten(b)):
        if x.dtype.is_floating:tf.debugging.assert_near(x,y,rtol=2e-11,atol=2e-11)
        else:tf.debugging.assert_equal(x,y)
    tf.debugging.assert_equal(a.valid,tf.ones_like(a.valid))
    tf.debugging.assert_equal(b.valid,tf.ones_like(b.valid))


def write_fixture(path,kind,q,raw,warmup,draws):
    c,d=q.shape;n=int(raw[0].shape[0]) if raw else 0
    with path.open('w') as f:
        f.write(f'{kind} {d} {n} {c} {warmup} {draws} 5\n')
        for row in q.numpy().tolist():f.write(' '.join(map(repr,row))+'\n')
        for v in raw:
            for row in v.numpy().tolist():f.write((' '.join(map(repr,row)) if isinstance(row,list) else repr(row))+'\n')


def layout(tf,a,report):
    for name in a.targets.split(','):
        q,model,data,raw,kind,pre=unpack_fixture(tf,name,4)
        cfg=Config(4,q.shape[-1],warmup=a.warmup,draws=a.draws)
        fs={x:make_functions(tf,model,replace(cfg,execution_layout=x),[tf.TensorSpec(v.shape,v.dtype) for v in data]) for x in a.layouts.split(',')}
        first={};results={};graphs={}
        seed=tf.constant([[SEEDS[0]+i,0] for i in range(4)],tf.int32)
        for label,f in fs.items():
            results[label],first[label]=timed(tf,f.fit,q,seed,*data)
            graphs[label]=graph_report(f.fit,(q,seed,*data),a.output/(name+'-'+label+'.hlo.txt'))
        for r in results.values():assert_equal_fit(tf,results['fieldwise'],r)
        device=results['fieldwise'].sampling.draws.device
        work=int(tf.reduce_sum(results['fieldwise'].warmup.evaluations).numpy()+tf.reduce_sum(results['fieldwise'].sampling.evaluations).numpy())
        del r,results
        pairs={x:[] for x in fs if x!='fieldwise'};records=[]
        for block,key in enumerate(SEEDS):
            seed=tf.constant([[key+i,0] for i in range(4)],tf.int32)
            for candidate in pairs:
                checked={label:fs[label].fit(q,seed,*data) for label in ('fieldwise',candidate)}
                assert_equal_fit(tf,checked['fieldwise'],checked[candidate])
                del checked
                observations=[]
                for rep in range(4):
                    order=['fieldwise',candidate] if (block+rep)%2==0 else [candidate,'fieldwise']
                    times={}
                    for label in order:
                        measured,elapsed=timed(tf,fs[label].fit,q,seed,*data)
                        times[label]=elapsed
                        del measured
                    observations.append((times['fieldwise'],times[candidate]))
                    records.append(dict(seed=key,rep=rep,candidate=candidate,order=order,seconds=times))
                pairs[candidate].append(observations)
        entry=dict(target=name,config=asdict(cfg),first_call_seconds=first,graphs=graphs,records=records,
            comparisons={x:ratios_summary(v) for x,v in pairs.items()},retained_parity=True,
            output_device=device,transition_evaluations=work)
        entry['by_order']={candidate:{order:statistics.median(
            r['seconds']['fieldwise']/r['seconds'][candidate] for r in records
            if r['candidate']==candidate and r['order'][0]==order)
            for order in ('fieldwise',candidate)} for candidate in pairs}
        report['results'].append(entry);write(a.output/'result.json',report)
        print(json.dumps(dict(target=name,comparisons=entry['comparisons'])),flush=True)


def batch(tf,a,report):
    for name in a.targets.split(','):
        saved={}
        for c in (1,4,16):
            q,model,data,raw,kind,pre=unpack_fixture(tf,name,c)
            cfg=Config(c,q.shape[-1],warmup=a.warmup,draws=a.draws,execution_layout=a.layouts)
            f=make_functions(tf,model,cfg,[tf.TensorSpec(v.shape,v.dtype) for v in data])
            seeds=tf.constant([[SEEDS[0]+i,0] for i in range(c)],tf.int32)
            r,first=timed(tf,f.fit,q,seeds,*data)
            tf.debugging.assert_equal(r.valid,tf.ones_like(r.valid))
            logical=int(tf.reduce_sum(r.warmup.evaluations).numpy()+tf.reduce_sum(r.sampling.evaluations).numpy())
            current=r.sampling.draws.numpy().tolist()
            del r
            times=[]
            for _ in range(7):
                measured,t=timed(tf,f.fit,q,seeds,*data);times.append(t);del measured
            for old_c,old in saved.items():
                tf.debugging.assert_near(tf.constant(old,tf.float64),tf.constant(current,tf.float64)[:,:old_c],rtol=2e-11,atol=2e-11)
            saved[c]=current
            e=dict(target=name,config=asdict(cfg),first_call_seconds=first,steady_seconds=times,
                median_seconds=statistics.median(times),chain_transitions_per_second=c*(a.warmup+a.draws)/statistics.median(times),
                retained_prefix_parity=True,logical_transition_evaluations=logical,
                graph=graph_report(f.fit,(q,seeds,*data),a.output/f'{name}-{c}.hlo.txt'))
            report['results'].append(e);write(a.output/'result.json',report)
            print(json.dumps({k:e[k] for k in ('target','median_seconds','chain_transitions_per_second')}|{'chains':c}),flush=True)


def target_checks(tf,a):
    """Independent analytic checks with nonzero responses and extreme logits."""
    q=tf.constant([[.2,-.3,.4,-.5],[-2.,1.,.5,-.7],[30.,-20.,10.,-40.]],tf.float64)
    x=tf.constant([[.1*((i*7+j*3)%11-5) for j in range(4)] for i in range(11)],tf.float64)
    y=tf.constant([.1*(i-3) for i in range(11)],tf.float64)
    labels=tf.constant([i%2 for i in range(11)],tf.float64)
    for kind in (3,4,5):
        response=labels if kind==4 else y
        if kind==4:
            def model(q,x,y):
                z=tf.matmul(q,x,transpose_b=True)
                return -.5*tf.reduce_sum(q*q,-1)-tf.reduce_mean(tf.nn.softplus(z)-y*z,-1)
        else:model=model_for(tf,3)
        f=make_functions(tf,model,Config(3,4),[tf.TensorSpec(x.shape,tf.float64),tf.TensorSpec(response.shape,tf.float64)])
        lp,g=f.value_gradient(q,x,response)
        if kind==5:
            n=tf.cast(tf.shape(x)[0],tf.float64)
            values=(tf.matmul(x,x,transpose_a=True)/n,tf.linalg.matvec(x,response,transpose_a=True)/n,tf.reduce_sum(response*response)/n)
            sf=make_functions(tf,lambda q,*data:sufficient_model(tf,q,*data),Config(3,4),
                [tf.TensorSpec(v.shape,v.dtype) for v in values])
            slp,sg=sf.value_gradient(q,*values)
            tf.debugging.assert_near(lp,slp,rtol=2e-11,atol=2e-11)
            tf.debugging.assert_near(g,sg,rtol=2e-11,atol=2e-11)
        path=a.output/f'gradient-check-{kind}.txt';write_fixture(path,kind,q,(x,response),0,0)
        for exe in (a.oracle,a.native_oracle):
            values=json.loads(subprocess.check_output([str(exe),'--evaluate',str(path)],text=True))['values']
            tf.debugging.assert_near(lp,tf.constant([v['lp'] for v in values],tf.float64),rtol=2e-11,atol=2e-11)
            tf.debugging.assert_near(g,tf.constant([v['g'] for v in values],tf.float64),rtol=2e-11,atol=2e-11)
    return dict(native_target_value_score_cases=18,nonzero_regression_responses=True,extreme_logits=True)


def native(tf,a,report):
    report['target_checks']=target_checks(tf,a)
    for name in a.targets.split(','):
        q,model,data,raw,kind,pre=unpack_fixture(tf,name,4)
        cfg=Config(4,q.shape[-1],warmup=a.warmup,draws=a.draws,execution_layout=a.layouts)
        f=make_functions(tf,model,cfg,[tf.TensorSpec(v.shape,v.dtype) for v in data])
        path=a.output/(name+'-input.txt');write_fixture(path,kind,q,raw,a.warmup,a.draws)
        lp,g=f.value_gradient(q,*data)
        for exe in (a.oracle,a.native_oracle):
            evaluation=json.loads(subprocess.check_output([str(exe),'--evaluate',str(path)],text=True))['values']
            tf.debugging.assert_near(lp,tf.constant([x['lp'] for x in evaluation],tf.float64),rtol=2e-11,atol=2e-11)
            tf.debugging.assert_near(g,tf.constant([x['g'] for x in evaluation],tf.float64),rtol=2e-11,atol=2e-11)
        if name=='sufficient':
            full=make_functions(tf,model_for(tf,3),cfg,[tf.TensorSpec(v.shape,v.dtype) for v in raw])
            raw_lp,raw_g=full.value_gradient(q,*raw)
            tf.debugging.assert_near(lp,raw_lp,rtol=2e-11,atol=2e-11);tf.debugging.assert_near(g,raw_g,rtol=2e-11,atol=2e-11)
        seed=tf.constant([[SEEDS[0]+i,0] for i in range(4)],tf.int32)
        initial,first=timed(tf,f.fit,q,seed,*data)
        tf.debugging.assert_equal(initial.valid,tf.ones_like(initial.valid));del initial
        graph=graph_report(f.fit,(q,seed,*data),a.output/(name+'-fit.hlo.txt'))
        records=[]
        for block,key in enumerate(SEEDS):
            seed=tf.constant([[key+i,0] for i in range(4)],tf.int32)
            for rep in range(3):
                order=['tf','generic','native']
                offset=(block+rep)%3;order=order[offset:]+order[:offset]
                times={};draws={};work={};preprocess={}
                for label in order:
                    if label=='tf':
                        r,t=timed(tf,f.fit,q,seed,*data);times[label]=t
                        tf.debugging.assert_equal(r.valid,tf.ones_like(r.valid))
                        work[label]=int(tf.reduce_sum(r.warmup.evaluations).numpy()+tf.reduce_sum(r.sampling.evaluations).numpy())
                        del r
                    else:
                        exe=a.oracle if label=='generic' else a.native_oracle
                        nr=json.loads(subprocess.check_output([str(exe),str(path),str(key),'1'],text=True))
                        times[label]=nr['parallel_seconds'][0];times[label+'_serial']=nr['serial_seconds'][0]
                        draws[label]=tf.constant([x['draws'] for x in nr['chains']],tf.float64)
                        tf.debugging.assert_all_finite(draws[label],'native retained draws')
                        if any(not math.isfinite(x['epsilon']) or x['epsilon']<=0 for x in nr['chains']):
                            raise AssertionError('invalid native tuning')
                        work[label]=sum(x['evaluations'] for x in nr['chains']);preprocess[label]=nr['preprocess_seconds']
                # Different architecture reductions can alter trajectories; report, do not silently equate.
                native_max=float(tf.reduce_max(tf.abs(draws['generic']-draws['native'])).numpy())
                records.append(dict(seed=key,rep=rep,order=order,seconds=times,evaluations=work,
                    preprocess_seconds=preprocess,native_draw_max_difference=native_max))
        comparisons={}
        for label in ('native','tf'):
            blocks=[[(x['seconds']['generic'],x['seconds'][label]) for x in records if x['seed']==key] for key in SEEDS]
            comparisons['generic_over_'+label]=ratios_summary(blocks)
        entry=dict(target=name,config=asdict(cfg),records=records,comparisons=comparisons,
            tf_first_call_seconds=first,tf_preprocess=pre,analytic_gradient_check=True,graph=graph,
            input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            interpretation='native/compiler differences may change trajectories; TF/native timing is descriptive')
        report['results'].append(entry);write(a.output/'result.json',report)
        print(json.dumps(dict(target=name,comparisons=comparisons)),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['layout','batch','native'],required=True)
    p.add_argument('--device',choices=['cpu','gpu'],default='gpu');p.add_argument('--output',type=Path,required=True)
    p.add_argument('--layouts',default='fieldwise,packed,flat');p.add_argument('--targets',default='gaussian')
    p.add_argument('--warmup',type=int,default=100);p.add_argument('--draws',type=int,default=100)
    p.add_argument('--oracle',type=Path);p.add_argument('--native-oracle',type=Path)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    tf,report=setup(a.device);report.update(mode=a.mode,results=[],seeds=SEEDS)
    begin=time.monotonic()
    with tf.device('/GPU:0' if a.device=='gpu' else '/CPU:0'):
        {'layout':layout,'batch':batch,'native':native}[a.mode](tf,a,report)
    report.update(status='passed',wall_seconds=time.monotonic()-begin,peak_host_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if a.device=='gpu':report['allocator']=tf.config.experimental.get_memory_info('GPU:0')
    write(a.output/'result.json',report)
if __name__=='__main__':main()
