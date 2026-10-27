"""Standalone Stan 2.37 warm-up transplant for stock TFP NUTS.

This is the owner-authorized diagnostic benchmark, not a public tuner. All
numerical adaptation and sampling lives in enclosing XLA functions. The
affine metric coordinates q=A z satisfy A A^T = Stan inverse metric.

No CmdStan dependency is needed at runtime. Targets must map independent
float64 rows [chains, dimension] to differentiable float64 values [chains].
Use warmup_checked, freeze and sample_checked at host boundaries; raw compiled
functions return validity flags, because XLA does not enforce TF assertions.

Stan-derived adaptation arithmetic: Copyright (c) 2011--2020, Stan Developers
and their Assignees. BSD 3-Clause terms are retained in stan_nuts_LICENSE.txt.
"""
from dataclasses import dataclass
import math
from numbers import Real
from typing import NamedTuple


class FrozenState(NamedTuple):
    position: object
    step_size: object
    metric_factor: object
    seed: object


class WarmupResult(NamedTuple):
    position: object
    step_size: object
    metric_factor: object
    seed: object
    draws: object
    telemetry: object
    valid: object
    completion_token: object


class SampleResult(NamedTuple):
    position: object
    seed: object
    draws: object
    telemetry: object
    valid: object
    completion_token: object


def require_valid(result):
    """Host validation only; never include this materialization in an XLA graph."""
    if not all(result.valid.numpy().tolist()):
        raise RuntimeError('Invalid adaptation/sampling state; inspect saved telemetry')
    return result


def warmup_checked(warmup, position, seed):
    return require_valid(warmup(position, seed))


def freeze(warmed):
    """Freeze a successful warm-up; warm-up draws are not posterior draws."""
    require_valid(warmed)
    return FrozenState(*warmed[:4])


def sample_checked(sample, state):
    """Validate a frozen sampling chunk. Resume with its position and seed."""
    return require_valid(sample(*state))


@dataclass(frozen=True)
class Config:
    dimension: int
    chains: int = 1
    warmup: int = 1000
    draws: int = 2000
    metric: str = 'diag'
    depth: int = 10
    delta: float = .8
    initial_epsilon: float = 1.
    initial_buffer: int = 75
    terminal_buffer: int = 50
    window: int = 25
    gamma: float = .05
    kappa: float = .75
    t0: float = 10.

    def __post_init__(self):
        for name in ('dimension', 'chains', 'warmup', 'draws', 'depth',
                     'initial_buffer', 'terminal_buffer', 'window'):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= 2**31-1:
                raise ValueError(f'{name} must be a positive int32-sized Python integer')
        # TFP trajectory counters are int32, with at most 2**depth-1 leapfrogs.
        if self.depth > 30:
            raise ValueError('depth must fit TFP int32 trajectory counters (<=30)')
        for name in ('delta', 'gamma', 'kappa', 't0', 'initial_epsilon'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError(f'{name} must be a finite positive real')
            try:
                finite = math.isfinite(value)
            except OverflowError:
                finite = False
            if not finite or value <= 0:
                raise ValueError(f'{name} must be a finite positive real')
        if self.metric not in ('diag', 'dense') or not 0 < self.delta < 1:
            raise ValueError('invalid metric or acceptance target')
        if self.warmup >= 20 and self.initial_buffer+self.terminal_buffer+self.window > self.warmup:
            raise ValueError('supply fitting windows; short-budget Stan fallback is not supported')


def stan_schedule(config):
    """Zero-based active/end flags, transcribed from windowed_adaptation.hpp."""
    n=config.warmup
    if n < 20:
        return [False]*n, [False]*n
    initial, terminal, window=config.initial_buffer, config.terminal_buffer, config.window
    end=initial+window-1
    active=[]; ends=[]
    for i in range(n):
        active.append(initial <= i < n-terminal)
        ends.append(i == end)
        if i == end and end != n-terminal-1:
            window *= 2
            end=i+window
            if end != n-terminal-1 and end+2*window >= n-terminal:
                end=n-terminal-1
    return active, ends


def dual_update(tf, count, hbar, xbar, mu, acceptance, config):
    count=count+1
    n=tf.cast(count,tf.float64)
    eta=1./(n+config.t0)
    hbar=(1.-eta)*hbar+eta*(config.delta-tf.minimum(acceptance,1.))
    x=mu-hbar*tf.sqrt(n)/config.gamma
    weight=tf.pow(n,-config.kappa)
    xbar=(1.-weight)*xbar+weight*x
    return count,hbar,xbar,tf.exp(x)


def welford_update(tf, count, mean, m2, position):
    """Window-local streaming covariance, in Stan's floating-point order."""
    count=count+1
    diff=position-mean
    mean=mean+diff/tf.cast(count,mean.dtype)
    m2=m2+(position-mean)[:,:,None]*diff[:,None,:]
    return count,mean,m2


def regularized_metric(tf, m2, count, dense, previous_metric=None):
    """Stan leaves the previous estimate unchanged when count <= 1."""
    n=tf.cast(count,tf.float64)
    eye=tf.eye(m2.shape[-1],batch_shape=[m2.shape[0]],dtype=tf.float64)
    if previous_metric is None:
        previous_metric=eye
    covariance=tf.where(count>1,m2/tf.maximum(n-1.,1.),previous_metric)
    covariance=n/(n+5.)*covariance+1e-3*5./(n+5.)*eye
    if not dense:
        covariance=tf.linalg.diag(tf.linalg.diag_part(covariance))
    return covariance


def reasonable_step_size(tf, epsilon, rng, probe, max_trials=2048):
    """Stan base_hmc::init_stepsize control flow with a supplied probe.

    probe(epsilon, rng) returns (H0-H1, next_rng) using a fresh momentum and
    the same initial position on every call. The extra bound exceeds the
    doubling/halving range of positive FP64 epsilon below Stan's 1e7 cap.
    """
    valid=tf.math.is_finite(epsilon)&(epsilon>0.)
    eligible=valid&(epsilon<=1e7)
    threshold=tf.math.log(tf.constant(.8,tf.float64))
    def clean(value):
        return tf.where(tf.math.is_nan(value),tf.constant(-float('inf'),tf.float64),value)
    first,rng=tf.cond(tf.reduce_any(eligible),lambda:probe(epsilon,rng),
                     lambda:(tf.zeros_like(epsilon),rng))
    direction=clean(first)>threshold
    def body(i,e,r,pending,counts,ok):
        loga,r=probe(e,r)
        loga=clean(loga)
        go=tf.where(direction,loga>threshold,loga<threshold)&pending
        updated=tf.where(go,e*tf.where(direction,tf.constant(2.,e.dtype),
                                       tf.constant(.5,e.dtype)),e)
        in_bounds=(updated>0.)&(updated<=1e7)&tf.math.is_finite(updated)
        ok &= ~(go&~in_bounds)
        return i+1,updated,r,go&in_bounds,counts+tf.cast(pending,tf.int32),ok
    _,epsilon,rng,pending,counts,valid=tf.while_loop(
        lambda i,e,r,p,n,ok:(i<max_trials)&tf.reduce_any(p),body,
        (tf.constant(0),epsilon,rng,eligible,tf.cast(eligible,tf.int32),valid),
        parallel_iterations=1)
    return epsilon,rng,counts,valid&~pending


def leapfrog_probe(tf, value_score, position, momentum, epsilon, value, grad):
    """One unit-mass velocity-Verlet probe; gradients are log-density scores."""
    half=momentum+.5*epsilon[:,None]*grad
    q=position+epsilon[:,None]*half
    v,g=value_score(q)
    end_p=half+.5*epsilon[:,None]*g
    log_accept=v-value+.5*tf.reduce_sum(
        tf.square(momentum)-tf.square(end_p),axis=-1)
    log_accept=tf.where(tf.math.is_nan(log_accept),
                        tf.constant(-float('inf'),position.dtype),log_accept)
    return q,end_p,log_accept


def make_functions(tf, tfp, log_prob, config):
    """Build target-generic, FP64, stable-signature XLA warm-up and sampling.

    The caller owns parameter transforms/Jacobians and ensures that log_prob
    does not couple chains. Config changes or a different target need a new
    factory call. Input contract violations fail at tracing; numerical failures
    are reported in result.valid and rejected by the checked host helpers.
    """
    c,d=config.chains,config.dimension
    dtype=tf.float64
    active,ends=stan_schedule(config)
    def physical(z,a):
        return tf.linalg.matvec(a,z)
    def latent(q,a):
        return tf.linalg.triangular_solve(a,q[...,None])[...,0]
    def target(z,a):
        value=log_prob(physical(z,a))
        if not tf.is_tensor(value) or value.dtype!=dtype:
            raise TypeError('log_prob must return a float64 Tensor of shape [chains]')
        return tf.ensure_shape(value,[c])
    def kernel(a,epsilon):
        return tfp.mcmc.NoUTurnSampler(
            lambda z: target(z,a),epsilon[:,None],
            max_tree_depth=config.depth,max_energy_diff=1000.,
            unrolled_leapfrog_steps=1,parallel_iterations=1)
    def value_score(z,a):
        with tf.GradientTape() as tape:
            tape.watch(z)
            value=target(z,a)
        grad=tape.gradient(value,z)
        if grad is None:
            raise ValueError('log_prob must be differentiable with respect to position')
        return value,grad
    def find_epsilon(z,a,epsilon,rng):
        value,grad=value_score(z,a)
        def probe(e,r):
            keys=tf.random.experimental.stateless_split(r,2)
            p=tf.random.stateless_normal([c,d],keys[1],dtype=dtype)
            _,_,log_accept=leapfrog_probe(tf,lambda q:value_score(q,a),z,p,e,value,grad)
            return log_accept,keys[0]
        e,r,counts,success=reasonable_step_size(tf,epsilon,rng,probe)
        success &= tf.math.is_finite(value)&tf.reduce_all(tf.math.is_finite(grad),axis=-1)
        success &= tf.reduce_all(tf.math.is_finite(z),axis=-1)
        return e,r,counts,success
    def valid_parameters(e,a):
        ok=tf.math.is_finite(e)&(e>0.)
        ok &= tf.reduce_all(tf.math.is_finite(a),axis=[-2,-1])
        ok &= tf.reduce_all(tf.linalg.diag_part(a)>0.,axis=-1)
        ok &= tf.reduce_all(a==tf.linalg.band_part(a,-1,0),axis=[-2,-1])
        if config.metric=='diag':
            ok &= tf.reduce_all(a==tf.linalg.diag(tf.linalg.diag_part(a)),axis=[-2,-1])
        return ok
    def health(q,results):
        ok=tf.reduce_all(tf.math.is_finite(q),axis=-1)
        ok &= tf.math.is_finite(results.target_log_prob)
        ok &= tf.reduce_all(tf.math.is_finite(results.grads_target_log_prob[0]),axis=-1)
        ok &= ~tf.math.is_nan(results.log_accept_ratio) & (results.log_accept_ratio<float('inf'))
        return ok
    def stats(q,kr):
        return tf.stack((tf.exp(tf.minimum(kr.log_accept_ratio,0.)),
            tf.cast(kr.is_accepted,dtype),tf.cast(kr.leapfrogs_taken,dtype),
            tf.cast(kr.reach_max_depth,dtype),tf.cast(kr.has_divergence,dtype),
            tf.cast(health(q,kr),dtype)),axis=-1)

    @tf.function(input_signature=[tf.TensorSpec([c,d],dtype),tf.TensorSpec([2],tf.int32)],
                 autograph=False,jit_compile=True)
    def warmup(q,rng):
        a=tf.eye(d,batch_shape=[c],dtype=dtype)
        epsilon=tf.fill([c],tf.constant(config.initial_epsilon,dtype))
        mu=tf.math.log(10.*epsilon)  # Stan service sets this before init_stepsize.
        epsilon,rng,probes,success=find_epsilon(q,a,epsilon,rng)
        z=q
        kr=kernel(a,epsilon).bootstrap_results(z)
        success &= health(q,kr)&valid_parameters(epsilon,a)
        saved=tf.TensorArray(dtype,size=config.warmup,element_shape=[c,d])
        trace=tf.TensorArray(dtype,size=config.warmup,element_shape=[c,10])
        active_tf=tf.constant(active);end_tf=tf.constant(ends)
        def body(i,z,a,e,r,kr,n,h,x,mu,wn,mean,m2,probes,ok,saved,trace):
            keys=tf.random.experimental.stateless_split(r,2);r=keys[0]
            k=kernel(a,e)
            z,kr=k.one_step(z,kr._replace(step_size=e[:,None]),seed=keys[1])
            q=physical(z,a)
            row=stats(q,kr)
            ok &= health(q,kr)
            n,h,x,e=dual_update(tf,n,h,x,mu,row[:,0],config)
            is_active=active_tf[i]
            wn,mean,m2=tf.cond(is_active,
                lambda:welford_update(tf,wn,mean,m2,q),lambda:(wn,mean,m2))
            def boundary():
                covariance=regularized_metric(tf,m2,wn,config.metric=='dense',
                                              tf.matmul(a,a,transpose_b=True))
                new_a=tf.linalg.cholesky(covariance)
                new_z=latent(q,new_a)
                new_e,new_r,count,good=find_epsilon(new_z,new_a,e,r)
                new_kr=kernel(new_a,new_e).bootstrap_results(new_z)
                return (new_z,new_a,new_e,new_r,new_kr,tf.constant(0),
                    tf.zeros_like(h),tf.zeros_like(x),tf.math.log(10.*new_e),
                    tf.constant(0),tf.zeros_like(mean),tf.zeros_like(m2),
                    probes+count,ok&good&health(q,new_kr)&valid_parameters(new_e,new_a))
            z,a,e,r,kr,n,h,x,mu,wn,mean,m2,probes,ok=tf.cond(end_tf[i],boundary,
                lambda:(z,a,e,r,kr,n,h,x,mu,wn,mean,m2,probes,ok))
            ok &= tf.math.is_finite(e)&(e>0.)
            detail=tf.stack((e,tf.fill([c],tf.cast(end_tf[i],dtype)),
                             tf.cast(probes,dtype),tf.cast(ok,dtype)),axis=-1)
            return (i+1,z,a,e,r,kr,n,h,x,mu,wn,mean,m2,probes,ok,
                    saved.write(i,q),trace.write(i,tf.concat((row,detail),axis=-1)))
        result=tf.while_loop(lambda i,*_:i<config.warmup,body,
            (tf.constant(0),z,a,epsilon,rng,kr,tf.constant(0),tf.zeros([c],dtype),
             tf.zeros([c],dtype),mu,tf.constant(0),tf.zeros([c,d],dtype),
             tf.zeros([c,d,d],dtype),probes,success,saved,trace),parallel_iterations=1)
        _,z,a,e,r,kr,n,h,x,mu,wn,mean,m2,probes,ok,saved,trace=result
        e=tf.where(n>0,tf.exp(x),e)
        q=physical(z,a)
        ok &= health(q,kr)&valid_parameters(e,a)
        draws,telemetry=saved.stack(),trace.stack()
        token=tf.reduce_sum(q)+tf.reduce_sum(e)+tf.reduce_sum(a)+tf.reduce_sum(telemetry)
        return WarmupResult(q,e,a,r,draws,telemetry,ok,token)

    @tf.function(input_signature=[tf.TensorSpec([c,d],dtype),tf.TensorSpec([c],dtype),
        tf.TensorSpec([c,d,d],dtype),tf.TensorSpec([2],tf.int32)],autograph=False,jit_compile=True)
    def sample(q,e,a,rng):
        z=latent(q,a);k=kernel(a,e);kr=k.bootstrap_results(z)
        ok=valid_parameters(e,a)&health(q,kr)
        saved=tf.TensorArray(dtype,size=config.draws,element_shape=[c,d])
        trace=tf.TensorArray(dtype,size=config.draws,element_shape=[c,6])
        def body(i,z,kr,r,ok,saved,trace):
            keys=tf.random.experimental.stateless_split(r,2)
            z,kr=k.one_step(z,kr,seed=keys[1])
            q=physical(z,a)
            ok &= health(q,kr)
            row=stats(q,kr)
            row=tf.concat((row[:,:5],tf.cast(ok[:,None],dtype)),axis=-1)
            return i+1,z,kr,keys[0],ok,saved.write(i,q),trace.write(i,row)
        _,z,kr,rng,ok,saved,trace=tf.while_loop(lambda i,*_:i<config.draws,body,
            (tf.constant(0),z,kr,rng,ok,saved,trace),parallel_iterations=1)
        draws,telemetry=saved.stack(),trace.stack()
        token=tf.reduce_sum(draws)+tf.reduce_sum(telemetry)
        return SampleResult(physical(z,a),rng,draws,telemetry,ok,token)
    return warmup,sample
