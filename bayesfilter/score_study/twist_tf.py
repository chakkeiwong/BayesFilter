"""Gaussian positive-power psi-APF with the full normalizer telescope.

Local implementation of Guarniero--Johansen--Lee (2017), equations (5)--(6),
Proposition 1. Includes x0 with g0=psi0=1. This is a comparator, not LEDH.
The analytical score holds ancestor labels locally fixed and is not claimed
to be an unbiased gradient of the expected log particle likelihood.
"""
from functools import lru_cache
import tensorflow as tf
from .gaussian_tf import (parameterized_model, chol_tangent, condition, symmetric,
                          gaussian_log_density_and_tangent)


@lru_cache(maxsize=16)
def make_twist_kernel(d, o, N, T, power, dtype_name="float64", jit_compile=True):
    if not isinstance(power, (int, float)) or not 0 <= power < float("inf"):
        raise ValueError("fixed finite nonnegative twist power required")
    dtype = tf.as_dtype(dtype_name)
    @tf.function(input_signature=[tf.TensorSpec([6], dtype), tf.TensorSpec([T,o],dtype),
        tf.TensorSpec([N,d],dtype), tf.TensorSpec([T,N,d],dtype),
        tf.TensorSpec([T+1,N],dtype)], jit_compile=jit_compile)
    def kernel(theta, observations, initial_noise, noise, uniforms):
        A,dA,H,dH,m,dm,P,dP,Q,dQ,R,dR = parameterized_model(theta,d,o)
        L = tf.linalg.cholesky(P)
        x = m + tf.einsum("ij,nj->ni",L,initial_noise)
        dx = dm[:,None,:] + tf.einsum("pij,nj->pni",chol_tangent(L,dP),initial_noise)
        LQ = tf.linalg.cholesky(Q)
        if power > 0:
            effective_R, effective_dR = R/power,dR/power
            S = symmetric(H@Q@tf.transpose(H)+effective_R)
            dS = symmetric(dH@Q@tf.transpose(H)+H@dQ@tf.transpose(H)+H@Q@tf.linalg.matrix_transpose(dH)+effective_dR)
            LS = tf.linalg.cholesky(S)
            K = tf.transpose(tf.linalg.cholesky_solve(LS,H@Q))
            dK = tf.linalg.matrix_transpose(tf.linalg.cholesky_solve(LS,
                tf.linalg.matrix_transpose(dQ@tf.transpose(H)+Q@tf.linalg.matrix_transpose(dH)-K@dS)))
            zeros,dzeros = tf.zeros([d],dtype),tf.zeros([6,d],dtype)
            _,_,CP,dCP,_,_ = condition(zeros,dzeros,Q,dQ,tf.zeros([o],dtype),H,dH,effective_R,effective_dR)
            LC,dLC = tf.linalg.cholesky(CP),chol_tangent(tf.linalg.cholesky(CP),dCP)
            logdet_R = 2*tf.reduce_sum(tf.math.log(tf.linalg.diag_part(tf.linalg.cholesky(R))))
            log2pi = tf.math.log(2*tf.acos(tf.cast(-1.,dtype)))
            c = -.5*(power-1)*(tf.cast(o,dtype)*log2pi+logdet_R)-.5*tf.cast(o,dtype)*tf.math.log(tf.cast(power,dtype))
            dc = -.5*(power-1)*tf.linalg.trace(tf.linalg.cholesky_solve(tf.linalg.cholesky(R),dR))
        def future_factor(x,dx,y):
            if power == 0: return tf.zeros([N],dtype),tf.zeros([6,N],dtype)
            predicted = tf.einsum("ij,nj->ni",A,x)
            dpredicted = tf.einsum("pij,nj->pni",dA,x)+tf.einsum("ij,pnj->pni",A,dx)
            residual = y-tf.einsum("ij,nj->ni",H,predicted)
            dresidual = -tf.einsum("pij,nj->pni",dH,predicted)-tf.einsum("ij,pnj->pni",H,dpredicted)
            lf,dlf = gaussian_log_density_and_tangent(residual,dresidual,S,dS[:,None,:,:])
            return lf+c,dlf+dc[:,None]
        def normalize_and_resample(x,dx,terms,dterms,u):
            w = tf.nn.softmax(terms)
            increment = tf.reduce_logsumexp(terms)-tf.math.log(tf.cast(N,dtype))
            derivative = tf.reduce_sum(w*dterms,axis=1)
            cdf = tf.concat([tf.cumsum(w)[:-1],tf.ones([1],dtype)],axis=0)
            index = tf.searchsorted(cdf,u,side="right")
            return tf.gather(x,index),tf.gather(dx,index,axis=1),increment,derivative,1/tf.reduce_sum(w*w)
        # g0^psi = f(x0,psi1); omitting this initial normalizer is wrong.
        if power > 0:
            lf,dlf = future_factor(x,dx,observations[0])
            x,dx,logZ,dlogZ,ess = normalize_and_resample(x,dx,lf,dlf,uniforms[0])
        else:
            logZ,dlogZ,ess = tf.zeros([],dtype),tf.zeros([6],dtype),tf.cast(N,dtype)
        def step(t,x,dx,logZ,dlogZ,ess):
            pred = tf.einsum("ij,nj->ni",A,x)
            dpred = tf.einsum("pij,nj->pni",dA,x)+tf.einsum("ij,pnj->pni",A,dx)
            if power > 0:
                residual = observations[t]-tf.einsum("ij,nj->ni",H,pred)
                dr = -tf.einsum("pij,nj->pni",dH,pred)-tf.einsum("ij,pnj->pni",H,dpred)
                x = pred+tf.einsum("ij,nj->ni",K,residual)+tf.einsum("ij,nj->ni",LC,noise[t])
                dx = dpred+tf.einsum("pij,nj->pni",dK,residual)+tf.einsum("ij,pnj->pni",K,dr)+tf.einsum("pij,nj->pni",dLC,noise[t])
            else:
                x = pred+tf.einsum("ij,nj->ni",LQ,noise[t])
                dx = dpred+tf.einsum("pij,nj->pni",chol_tangent(LQ,dQ),noise[t])
            residual = observations[t]-tf.einsum("ij,nj->ni",H,x)
            dr = -tf.einsum("pij,nj->pni",dH,x)-tf.einsum("ij,pnj->pni",H,dx)
            lg,dlg = gaussian_log_density_and_tangent(residual,dr,R,dR[:,None,:,:])
            # psi_{T+1} tilde is one: the terminal future factor is zero in log.
            lf,dlf = tf.cond(t+1<T,lambda:future_factor(x,dx,observations[t+1]),
                            lambda:(tf.zeros([N],dtype),tf.zeros([6,N],dtype)))
            terms,dterms = (1-power)*lg+lf,(1-power)*dlg+dlf
            x,dx,increment,derivative,current_ess = normalize_and_resample(x,dx,terms,dterms,uniforms[t+1])
            return t+1,x,dx,logZ+increment,dlogZ+derivative,tf.minimum(ess,current_ess)
        result = tf.while_loop(lambda t,*_:t<T,step,(0,x,dx,logZ,dlogZ,ess),parallel_iterations=1)
        return result[3],result[4],result[5]
    return kernel
