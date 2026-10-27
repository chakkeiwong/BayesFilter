"""CPU/GPU XLA audit against saved unmodified-native transition oracles."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import resource
import sys
import time
import benchmark_nuts_xla as base
import benchmark_stan_transplant as common
import stan_nuts as port

def target(tf,kind):
    if kind==0:return lambda q: -.5*tf.reduce_sum(q*q,-1)
    if kind==1:return lambda q: -.5*(q[:,0]**2+(q[:,1]-(q[:,0]**2-1.))**2)
    if kind==2:return lambda q: -.5*q[:,0]**2/9.-.5*q[:,0]-.5*q[:,1]**2*tf.exp(-q[:,0])
    if kind==3:return lambda q: tf.where(tf.abs(q[:,0])<1.,-.5*tf.reduce_sum(q*q,-1),tf.constant(-float('inf'),tf.float64))
    return lambda q: 0.*tf.reduce_sum(q,-1)

def main():
    p=argparse.ArgumentParser();p.add_argument('--device',choices=['cpu','gpu'],required=True)
    p.add_argument('--oracle',type=Path,required=True);a=p.parse_args()
    folder=Path(os.environ['BAYESFILTER_STAN_PORT_OUTPUT'])
    conf=base.parser().parse_args(['--output',str(folder),'--device',a.device,'--gpu','0','--threads','1'])
    began=time.perf_counter();tf,tfp,memory=base.setup(conf)
    report=dict(command=sys.argv,device=a.device,memory_policy=memory,tensorflow=tf.__version__,
        gpu_intentionally_hidden=a.device=='cpu',tf32=False,jit_compile=True,cases=0,maximum_scaled_error=0.,lifecycle={})
    cache={}
    with tf.device('/'+a.device.upper()+':0'):
        for line in a.oracle.read_text().splitlines():
            row=json.loads(line);v=list(map(float,row['stdin'].split()))
            d,kind,depth=map(int,v[:3]);epsilon,jitter,threshold=v[3:6]
            q,pv=v[6:8],v[8:10];matrix=[v[10:12],v[12:14]];tape=v[15:]
            metric_kind='unit' if matrix==[[1.,0.],[0.,1.]] else ('diag' if matrix[0][1]==0. else 'dense')
            key=kind,metric_kind,depth,jitter,threshold
            if key not in cache:
                cfg=port.Config(2,depth=depth,metric=metric_kind,jitter=jitter,max_delta=threshold)
                def make(cfg,kind):
                    shape=[1,2,2] if cfg.metric=='dense' else [1,2]
                    @tf.function(input_signature=[tf.TensorSpec([1,2],tf.float64),tf.TensorSpec([1,2],tf.float64),
                        tf.TensorSpec([1],tf.float64),tf.TensorSpec(shape,tf.float64),tf.TensorSpec([1,256],tf.float64)],
                        autograph=False,jit_compile=True)
                    def fn(q,pv,e,m,u):
                        k=port.Kernel(tf,target(tf,kind),cfg,uniform_tape=u,momentum=pv)
                        return k.transition(q,e,m,k.factor(m),tf.zeros([1,2],tf.int32))
                    return fn
                cache[key]=make(cfg,kind)
            metric=matrix if metric_kind=='dense' else [matrix[0][0],matrix[1][1]]
            args=[tf.constant(x,tf.float64) for x in ([q],[pv],[epsilon],[metric],[tape])]
            out=cache[key](*args)
            expected=[[float(x) for x in l.split()] for l in row['stdout'].splitlines()]
            top=expected[0]
            discrete=[int(out.leaps[0]),int(out.depth[0]),int(out.divergent[0]),int(out.seed[0,1])]
            if discrete!=[int(top[j]) for j in (1,2,3,5)]:raise RuntimeError(f'Native branch mismatch {key}: {discrete}, {top}')
            got=[float(out.acceptance[0]),float(out.energy[0]),float(out.epsilon[0]),
                 *out.phase.q.numpy().tolist()[0],*out.phase.p.numpy().tolist()[0],
                 *out.phase.g.numpy().tolist()[0],float(out.phase.v[0])]
            want=[top[0],top[4],top[6],*expected[1],*expected[2],*expected[3],expected[4][0]]
            error=max(abs(x-y)/(1.+abs(y)) for x,y in zip(got,want))
            if error>2e-12:raise RuntimeError(f'Native numerical mismatch {key}: {error}')
            report['maximum_scaled_error']=max(error,report['maximum_scaled_error']);report['cases']+=1
        for metric in ('unit','diag','dense'):
            cfg=port.Config(2,chains=2,warmup=160,draws=16,metric=metric,depth=6)
            init,warm,sample=port.make_functions(tf,target(tf,1),cfg)
            q=tf.constant([[.2,.1],[-.1,.4]],tf.float64);e=tf.ones([2],tf.float64);seed=tf.constant([[4*20261027,0],[4*20261027+1,0]],tf.int32)
            m=tf.eye(2,batch_shape=[2],dtype=tf.float64) if metric=='dense' else tf.ones([2,2],tf.float64)
            w,wr=common.compile_fn(tf,warm,(q,e,m,seed),folder,metric+'-warm',a.device);port.require_valid(w)
            s,sr=common.compile_fn(tf,sample,w[:4],folder,metric+'-sample',a.device);port.require_valid(s)
            _,ws=base.timed(warm,q,e,m,seed);_,ss=base.timed(sample,*w[:4])
            report['lifecycle'][metric]=dict(config=asdict(cfg),warmup=wr,sample=sr,warmup_seconds=ws,sample_seconds=ss)
            common.write(folder/'audit.json',report)
    report['wall_seconds']=time.perf_counter()-began
    report['host_max_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if a.device=='gpu':report['allocator']=tf.config.experimental.get_memory_info('GPU:0')
    report['status']='passed';common.write(folder/'audit.json',report)
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
