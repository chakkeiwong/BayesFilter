"""Paper-derived RBF controlled Gaussian filter; forward consumer only.

Heng et al., published equation11/Algorithm1 and author supplement9.1 define
one parent-dependent Gaussian mixture. Gather the resampled parent before
computing BOTH mixture probabilities and means. This intentionally repairs
the pinned ToyMultimodal source's inconsistent component-probability row.
RBF fitting is separate and is not implemented by this module.
"""
from __future__ import annotations
import math
import tensorflow as tf
from bayesfilter.replication.controlled_smc_tf import systematic_ancestors


def rbf_gaussian_mixture(mean,variance,beta,weights,knots,constant):
    """Exact Gaussian product; mean[B,N], policy[B,M], beta/constant[B]."""
    variance=tf.convert_to_tensor(variance,mean.dtype)
    denominator=1+2*beta*variance
    component_variance=variance/denominator
    component_mean=(mean[:,:,None]+2*beta[:,None,None]*variance*knots[:,None,:])/denominator[:,None,None]
    component_log_mass=(tf.math.log(weights)[:,None,:]
        -beta[:,None,None]/denominator[:,None,None]*(mean[:,:,None]-knots[:,None,:])**2)
    logsum=tf.reduce_logsumexp(component_log_mass,-1)
    probabilities=tf.nn.softmax(component_log_mass,-1)
    log_integral=constant[:,None]-.5*tf.math.log(denominator)[:,None]+logsum
    valid=(tf.reduce_all(tf.math.is_finite(mean)) & tf.math.is_finite(variance) & (variance>0)
        & tf.reduce_all(tf.math.is_finite(beta)&(beta>=0))
        & tf.reduce_all(tf.math.is_finite(weights)&(weights>=0))
        & tf.reduce_all(tf.reduce_sum(weights,-1)>0)
        & tf.reduce_all(tf.math.is_finite(knots)) & tf.reduce_all(tf.math.is_finite(constant))
        & tf.reduce_all(tf.math.is_finite(denominator)&(denominator>0))
        & tf.reduce_all(tf.math.is_finite(log_integral)))
    return dict(mean=component_mean,variance=component_variance,probabilities=probabilities,
                log_integral=log_integral,valid=valid)


def rbf_log_policy(x,beta,weights,knots,constant):
    return constant[:,None]+tf.reduce_logsumexp(tf.math.log(weights)[:,None,:]
        -beta[:,None,None]*(x[:,:,None]-knots[:,None,:])**2,-1)


def multimodal_transition(t,x):
    return .5*x+25*x/(1+x*x)+8*tf.cos(1.2*tf.cast(t,x.dtype))


class RBFControlledSMC:
    """Fixed-signature, batch-native TF/XLA forward filter with explicit policy."""
    def __init__(self,*,observations,particles,replicas,components,
                 observation_variance,initial_variance=5.,transition_variance=10.,jit_compile=True):
        if min(observations,particles,replicas,components)<1:
            raise ValueError('Positive filter sizes required')
        if min(observation_variance,initial_variance,transition_variance)<=0:
            raise ValueError('Positive model variances required')
        self.h,self.n,self.b,self.m=observations,particles,replicas,components
        self.r=tf.constant(observation_variance,tf.float64)
        self.q0=tf.constant(initial_variance,tf.float64);self.q=tf.constant(transition_variance,tf.float64)
        scalar=tf.TensorSpec([observations,replicas],tf.float64)
        policy=tf.TensorSpec([observations,replicas,components],tf.float64)
        cloud=tf.TensorSpec([observations,replicas,particles],tf.float64)
        self.forward=tf.function(self._forward,input_signature=[scalar,policy,policy,scalar,scalar,cloud,cloud,scalar],
                                 jit_compile=jit_compile,autograph=False)

    def _forward(self,beta,weights,knots,constant,y,normal,component_uniform,uniform):
        dtype=tf.float64
        initial=rbf_gaussian_mixture(tf.zeros([self.b,self.n],dtype),self.q0,beta[0],weights[0],knots[0],constant[0])
        initial_log_normalizer=initial['log_integral'][:,0]
        arrays=tuple(tf.TensorArray(dtype,self.h) for _ in range(4))
        ancestry=tf.TensorArray(tf.int32,self.h);components=tf.TensorArray(tf.int32,self.h)
        def body(t,parent,logz,valid,arrays,ancestry,components):
            # parent already contains the selected ancestors, not pre-resampling rows.
            mean=tf.cond(t==0,lambda:tf.zeros_like(parent),lambda:multimodal_transition(t,parent))
            variance=tf.where(t==0,self.q0,self.q)
            proposal=rbf_gaussian_mixture(mean,variance,beta[t],weights[t],knots[t],constant[t])
            cdf=tf.cumsum(proposal['probabilities'],-1)
            cdf=tf.concat([cdf[:,:,:-1],tf.ones_like(cdf[:,:,-1:])],-1)
            component=tf.reduce_sum(tf.cast(component_uniform[t,:,:,None]>=cdf,tf.int32),-1)
            selected_mean=tf.gather(proposal['mean'],component,axis=2,batch_dims=2)
            x=selected_mean+tf.sqrt(proposal['variance'])[:,None]*normal[t]
            def next_integral():
                result=rbf_gaussian_mixture(multimodal_transition(t+1,x),self.q,beta[t+1],weights[t+1],knots[t+1],constant[t+1])
                return result['log_integral'],result['valid']
            lookahead,next_valid=tf.cond(t+1<self.h,next_integral,lambda:(tf.zeros_like(x),tf.constant(True)))
            logg=-.5*(tf.math.log(tf.constant(2*math.pi,dtype)*self.r)+(y[t,:,None]-x*x/20)**2/self.r)
            lw=logg+lookahead-rbf_log_policy(x,beta[t],weights[t],knots[t],constant[t])
            normalized=tf.nn.softmax(lw,-1);ess=1/tf.reduce_sum(normalized**2,-1)
            logz+=tf.reduce_logsumexp(lw,-1)-tf.math.log(tf.cast(self.n,dtype))
            logz+=tf.where(t==0,initial_log_normalizer,tf.zeros_like(logz))
            ancestor=systematic_ancestors(normalized,uniform[t])
            parent=tf.gather(x,ancestor,axis=1,batch_dims=1)
            valid=valid&proposal['valid']&next_valid&tf.reduce_all(tf.math.is_finite(x))&tf.reduce_all(tf.math.is_finite(lw))
            valid=valid&tf.reduce_all((component_uniform[t]>=0)&(component_uniform[t]<1))
            valid=valid&tf.reduce_all((uniform[t]>=0)&(uniform[t]<1))
            arrays=tuple(a.write(t,v) for a,v in zip(arrays,(x,lw,logz,ess)))
            return t+1,parent,logz,valid,arrays,ancestry.write(t,ancestor),components.write(t,component)
        state=tf.while_loop(lambda t,*_:t<self.h,body,
            (tf.constant(0),tf.zeros([self.b,self.n],dtype),tf.zeros([self.b],dtype),initial['valid'],arrays,ancestry,components),
            parallel_iterations=1)
        result={k:a.stack() for k,a in zip(('x','log_incremental_weight','logZ','ess'),state[4])}
        result.update(ancestors=state[5].stack(),components=state[6].stack(),valid=state[3])
        return result
