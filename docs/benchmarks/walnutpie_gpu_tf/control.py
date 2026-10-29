"""Synchronized XLA equivalent of pinned Walnutpie stop statistics.

Legacy log-density R-hat is a run-control statistic, not posterior certification.
Python only schedules compiled chunks, wall-clock checks and user callbacks.
"""
from dataclasses import dataclass
from typing import NamedTuple
import math
import time
from .api import make_kernel,run,state_spec

@dataclass(frozen=True)
class RunControl:
    warmup_min: int = 50
    warmup_max: int = 1000
    sample_min: int = 50
    sample_max: int = 1000
    warmup_stride: int = 5
    chunk_size: int = 5
    step_tolerance: float = .1
    mass_tolerance: float = 1.
    logp_rhat_tolerance: float = 1.01
    def __post_init__(self):
        for low,high in ((self.warmup_min,self.warmup_max),(self.sample_min,self.sample_max)):
            if type(low) is not int or type(high) is not int or low<0 or high<low:raise ValueError('invalid control limits')
        if self.sample_min<2:raise ValueError('at least two draws for logp variance')
        for n in (self.warmup_stride,self.chunk_size):
            if type(n) is not int or n<1:raise ValueError('stride/chunk must be positive')
        for x in (self.step_tolerance,self.mass_tolerance,self.logp_rhat_tolerance):
            if not math.isfinite(x) or x<0:raise ValueError('invalid tolerance')

class ControlState(NamedTuple):
    sampler: object
    count: object
    mean: object
    m2: object
    stopped: object
    reason: object  # 0 running, 1 criterion, 2 max count, 3 invalid state
    statistic: object

class ControlledRun(NamedTuple):
    state: object
    draws: object
    log_density: object
    mask: object

class Controllers(NamedTuple):
    begin: object
    warmup: object
    sample: object
    warmup_chunk: object
    sample_chunk: object
    warmup_statistics: object
    sampling_statistic: object


def warmup_statistics(tf,state):
    logmass=-tf.math.log(state.inverse_mass)
    geometric_mass=tf.exp(tf.reduce_mean(logmass,0))
    geometric_step=tf.exp(tf.reduce_mean(tf.math.log(state.epsilon)))
    mass=tf.exp(logmass)
    md=tf.reduce_max(tf.sqrt(tf.reduce_sum(((mass-geometric_mass)/geometric_mass)**2,-1)))
    sd=tf.reduce_max(tf.abs(state.epsilon-geometric_step)/geometric_step)
    return sd,md


def sampling_statistic(tf,mean,m2,count):
    n=tf.cast(count,tf.float64)
    within=tf.reduce_mean(m2/(n-1.))
    between=tf.reduce_sum((mean-tf.reduce_mean(mean))**2)/tf.cast(tf.size(mean)-1,tf.float64)
    return tf.sqrt(1.+between/within)


def make_controllers(tf,model,config,control=RunControl(),model_signature=()):
    c=config;p=control
    if c.chains<2:raise ValueError('cross-chain run control requires at least two chains')
    st=state_spec(tf,c);v=tf.TensorSpec([c.chains],tf.float64)
    cs=ControlState(st,tf.TensorSpec([],tf.int32),v,v,tf.TensorSpec([],tf.bool),
        tf.TensorSpec([],tf.int32),tf.TensorSpec([2],tf.float64))
    @tf.function(input_signature=[st],jit_compile=True,autograph=False)
    def begin(s):
        valid=tf.reduce_all(s.valid)
        return ControlState(s,tf.constant(0),tf.zeros([c.chains],tf.float64),tf.zeros([c.chains],tf.float64),
            ~valid,tf.where(valid,0,3),tf.fill([2],tf.constant(float('nan'),tf.float64)))
    def build(adapt,capacity):
        maximum=p.warmup_max if adapt else p.sample_max
        @tf.function(input_signature=[cs,*model_signature],jit_compile=True,autograph=False)
        def controlled(state,*data):
            k=make_kernel(tf,lambda q:model(q,*data),c)
            invalid=(state.count<0) | ~tf.reduce_all(state.sampler.valid)
            reason=tf.where(invalid,3,tf.where(~state.stopped & (state.count>=maximum),2,state.reason))
            state=state._replace(stopped=state.stopped | (reason!=0),reason=reason)
            draws=tf.zeros([capacity,c.chains,c.dimension],tf.float64)
            logp=tf.zeros([capacity,c.chains],tf.float64)
            mask=tf.zeros([capacity],tf.bool)
            if adapt and p.warmup_min==0:
                a,b=warmup_statistics(tf,state.sampler)
                met=(state.count==0)&~state.stopped&tf.math.is_finite(a)&tf.math.is_finite(b)&(a<=p.step_tolerance)&(b<=p.mass_tolerance)
                state=state._replace(stopped=state.stopped|met,reason=tf.where(met,1,state.reason),
                    statistic=tf.where(met,tf.stack([a,b]),state.statistic))
            if capacity==0:return ControlledRun(state,draws,logp,mask)
            def body(i,s,q,lp,mask):
                r=run(k,s.sampler,1,adapt);n=s.count+1
                delta=r.state.lp-s.mean;mean=s.mean+delta/tf.cast(n,tf.float64)
                m2=s.m2+delta*(r.state.lp-mean)
                if adapt:
                    a,b=warmup_statistics(tf,r.state)
                    ready=(n>=p.warmup_min)&((n%p.warmup_stride==0)|(n>=maximum))
                    met=ready & tf.math.is_finite(a)&tf.math.is_finite(b)&(a<=p.step_tolerance)&(b<=p.mass_tolerance)
                    stat=tf.stack([a,b])
                else:
                    a=sampling_statistic(tf,mean,m2,n)
                    met=(n>=p.sample_min)&tf.math.is_finite(a)&(a<=p.logp_rhat_tolerance)
                    stat=tf.stack([a,tf.constant(float('nan'),tf.float64)])
                invalid=~tf.reduce_all(r.state.valid)
                reason=tf.where(invalid,3,tf.where(met,1,tf.where(n>=maximum,2,0)))
                s=ControlState(r.state,n,mean,m2,reason!=0,reason,stat)
                idx=tf.reshape(i,[1,1])
                return i+1,s,tf.tensor_scatter_nd_update(q,idx,r.state.q[None]),tf.tensor_scatter_nd_update(lp,idx,r.state.lp[None]),tf.tensor_scatter_nd_update(mask,idx,[True])
            _,s,q,lp,mask=tf.while_loop(lambda i,s,*_: (i<capacity)&~s.stopped&(s.count<maximum),body,
                (tf.constant(0),state,draws,logp,mask),parallel_iterations=1)
            return ControlledRun(s,q,lp,mask)
        return controlled
    ws=tf.function(lambda s:warmup_statistics(tf,s),input_signature=[st],jit_compile=True,autograph=False)
    rs=tf.function(lambda mean,m2,n:sampling_statistic(tf,mean,m2,n),
        input_signature=[v,v,tf.TensorSpec([],tf.int32)],jit_compile=True,autograph=False)
    return Controllers(begin,build(True,p.warmup_max),build(False,p.sample_max),
        build(True,p.chunk_size),build(False,p.chunk_size),ws,rs)


def drive_chunks(function,state,*data,deadline_seconds=None,cancelled=None,on_chunk=None):
    """Host-only scheduling; deadline is checked between synchronized chunks.

    Returned chunks preserve all warmup/draws and their masks. Resuming with the
    returned state retains online moments, RNG and adaptation. Compilation time
    counts toward the deadline; a running compiled chunk cannot be interrupted.
    """
    if deadline_seconds is not None and (not math.isfinite(deadline_seconds) or deadline_seconds<0):raise ValueError('invalid deadline')
    start=time.monotonic();chunks=[];why='running'
    previous_count=int(state.count.numpy())
    while not bool(state.stopped.numpy()):
        if cancelled is not None and cancelled():why='cancelled';break
        if deadline_seconds is not None and time.monotonic()-start>=deadline_seconds:why='deadline';break
        result=function(state,*data)
        # Materialization synchronizes only run-control metadata, not model work.
        count=int(result.state.count.numpy())
        if count<=previous_count and not bool(result.state.stopped.numpy()):
            raise RuntimeError('compiled controller made no progress')
        previous_count=count
        state=result.state;chunks.append(result)
        if on_chunk is not None:on_chunk(result,count)
    if bool(state.stopped.numpy()):why={1:'criterion',2:'maximum',3:'invalid'}[int(state.reason.numpy())]
    return dict(state=state,chunks=chunks,reason=why,wall_seconds=time.monotonic()-start)
