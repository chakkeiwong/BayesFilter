"""Batch-native unregularized NNLS and one-refinement RBF backward fitting.

The active-set method solves min ||G w-y||^2, w>=0 using QR, with explicit
roundoff, rank, iteration and capacity checks. Static capacity limits execution,
not the accepted objective: reaching it rejects the fit. The RBF recursion is
the particle-knot branch of Heng et al.'s ToyMultimodal author implementation.
Only its first policy refinement is implemented here; no general I>1 claim.
"""
from __future__ import annotations
import math
import tensorflow as tf
from bayesfilter.replication.rbf_controlled_smc_tf import rbf_gaussian_mixture, multimodal_transition

D=tf.float64
EPS=2.220446049250313e-16


class NonnegativeLeastSquares:
    def __init__(self,*,rows,columns,replicas,capacity=128,stopping_factor=10.,jit_compile=True):
        if min(rows,columns,replicas,capacity)<1:raise ValueError('Positive dimensions required')
        self.rows,self.n,self.b=rows,columns,replicas
        self.c=min(capacity,columns);self.max_iterations=5*columns
        factors=list(stopping_factor) if isinstance(stopping_factor,(tuple,list)) else [stopping_factor]*replicas
        if len(factors)!=replicas or any(not math.isfinite(v) or v<=0 for v in factors):
            raise ValueError('One finite positive stopping factor per replica required')
        self.stopping_factor=tf.constant(factors,D)
        self.solve=tf.function(self._solve,input_signature=[tf.TensorSpec([replicas,rows,columns],D),
            tf.TensorSpec([replicas,rows],D)],jit_compile=jit_compile,autograph=False)

    def _solve(self,design,target):
        c,n,b=self.c,self.n,self.b
        roundoff_scale=EPS*tf.reduce_max(tf.reduce_sum(tf.abs(design),axis=1),axis=-1)*max(self.rows,n)
        tolerance=self.stopping_factor*roundoff_scale
        source_tolerance=10*roundoff_scale
        ids=tf.fill([b,c],tf.constant(-1,tf.int32));x=tf.zeros([b,n],D)
        valid=tf.reduce_all(tf.math.is_finite(design),[1,2])&tf.reduce_all(tf.math.is_finite(target),1)
        def mask(index):
            return tf.one_hot(index,n,dtype=D)
        def dual(value):
            residual=target-tf.linalg.matvec(design,value)
            return tf.linalg.matvec(design,residual,transpose_a=True)
        def eligible(value,index):
            return tf.where(tf.reduce_sum(mask(index),1)>0,tf.constant(float('-inf'),D),dual(value))
        def qr_solve(index):
            occupied=index>=0
            block=tf.gather(design,tf.maximum(index,0),axis=2,batch_dims=1)*tf.cast(occupied[:,None,:],D)
            # Inactive columns have orthogonal unit vectors in extra rows.
            # These add no penalty or ridge to any active coefficient.
            extra=tf.linalg.diag(tf.cast(~occupied,D))
            augmented=tf.concat([block,extra],axis=1)
            rhs=tf.concat([target,tf.zeros([b,c],D)],axis=1)
            q,r=tf.linalg.qr(augmented,full_matrices=False)
            norm=tf.sqrt(tf.reduce_sum(block**2,axis=1))
            ratio=tf.where(occupied,tf.math.divide_no_nan(tf.abs(tf.linalg.diag_part(r)),norm),tf.ones([b,c],D))
            minimum=tf.reduce_min(ratio,-1)
            rank_ok=(minimum>8*EPS*(self.rows+c))&tf.reduce_all(tf.math.is_finite(r),[1,2])
            safe_r=tf.where(rank_ok[:,None,None],r,tf.eye(c,batch_shape=[b],dtype=D))
            zslots=tf.linalg.triangular_solve(safe_r,tf.linalg.matvec(q,rhs,transpose_a=True)[:,:,None],lower=False)[:,:,0]
            zslots*=tf.cast(occupied,D)
            z=tf.einsum('bc,bcn->bn',zslots,mask(index))
            return z,zslots,rank_ok,minimum
        def condition(i,value,index,ok,peak,removed,minimum):
            return (i<self.max_iterations)&tf.reduce_any(ok&(tf.reduce_max(eligible(value,index),-1)>tolerance))
        def body(i,value,index,ok,peak,removed,minimum):
            scores=eligible(value,index);advance=ok&(tf.reduce_max(scores,-1)>tolerance)
            room=tf.reduce_any(index<0,-1);ok=ok&(~advance|room)
            advance=advance&room
            slot=tf.argmax(tf.cast(index<0,tf.int32),-1,output_type=tf.int32)
            column=tf.argmax(scores,-1,output_type=tf.int32)
            index=tf.where(advance[:,None]&(tf.range(c)[None,:]==slot[:,None]),column[:,None],index)
            peak=tf.maximum(peak,tf.reduce_sum(tf.cast(index>=0,tf.int32),-1))
            z,zslots,rank_ok,margin=qr_solve(index)
            ok=ok&(~advance|rank_ok);minimum=tf.minimum(minimum,margin)
            i+=1
            def negative(zs,idx,good):return (zs<0)&(idx>=0)&advance[:,None]&good[:,None]
            def inner_cond(j,v,idx,zfull,zs,good,removed,minimum):
                return (j<self.max_iterations)&tf.reduce_any(negative(zs,idx,good))
            def inner(j,v,idx,zfull,zs,good,removed,minimum):
                neg=negative(zs,idx,good);moving=tf.reduce_any(neg,-1)
                oldslots=tf.gather(v,tf.maximum(idx,0),axis=1,batch_dims=1)
                fraction=tf.where(neg,oldslots/tf.where(neg,oldslots-zs,tf.ones_like(zs)),tf.fill([b,c],tf.constant(float('inf'),D)))
                alpha=tf.where(moving,tf.reduce_min(fraction,-1),tf.ones([b],D))
                drop=neg&(fraction==alpha[:,None])
                newidx=tf.where(drop,tf.fill([b,c],-1),idx)
                moved=(v+alpha[:,None]*(zfull-v))*tf.reduce_sum(mask(newidx),1)
                v=tf.where(moving[:,None],moved,v)
                zfull,zs,rank_ok,margin=qr_solve(newidx)
                good=good&(~advance|rank_ok)
                return j+1,v,newidx,zfull,zs,good,removed+tf.reduce_sum(tf.cast(drop,tf.int32),-1),tf.minimum(minimum,margin)
            i,value,index,z,zslots,ok,removed,minimum=tf.while_loop(inner_cond,inner,
                (i,value,index,z,zslots,ok,removed,minimum),parallel_iterations=1)
            ok=ok&~tf.reduce_any(negative(zslots,index,ok),-1)
            value=tf.where((advance&ok)[:,None],z,value)
            ok=ok&tf.reduce_all(tf.math.is_finite(value)&(value>=0),-1)
            return i,value,index,ok,peak,removed,minimum
        iterations,x,ids,valid,peak,removed,minimum=tf.while_loop(condition,body,
            (tf.constant(0),x,ids,valid,tf.zeros([b],tf.int32),tf.zeros([b],tf.int32),tf.ones([b],D)),parallel_iterations=1)
        gradient=-dual(x);residual=tf.linalg.matvec(design,x)-target
        dual_error=tf.maximum(tf.zeros_like(tolerance),-tf.reduce_min(gradient,-1))
        complementarity=tf.reduce_max(tf.abs(x*gradient),-1)
        threshold=20*source_tolerance
        converged=tf.reduce_max(eligible(x,ids),-1)<=tolerance
        valid=valid&converged&(dual_error<=threshold)&(complementarity<=threshold*tf.maximum(tf.ones_like(tolerance),tf.reduce_max(x,-1)))
        return dict(coefficients=x,valid=valid,rss=tf.reduce_sum(residual**2,-1),gradient_tolerance=tolerance,
            dual_error=dual_error,complementarity=complementarity,iterations=iterations,
            peak_active=peak,removed=removed,minimum_qr_margin=minimum)


class RBFBackwardFit:
    """First refinement of the zero policy; all particle knots precede top-M."""
    def __init__(self,*,observations,particles,replicas,components,observation_variance,
                 bandwidth_factor,transition_variance=10.,capacity=128,stopping_factor=10.,jit_compile=True):
        if not 1<=components<=particles:raise ValueError('Invalid retained basis size')
        if particles<2 or min(observation_variance,transition_variance,bandwidth_factor)<=0:
            raise ValueError('Positive variances/bandwidth and at least two particles required')
        self.h,self.n,self.b,self.m=observations,particles,replicas,components
        self.r=tf.constant(observation_variance,D);self.q=tf.constant(transition_variance,D)
        self.factor=tf.constant(bandwidth_factor,D)
        self.nnls=NonnegativeLeastSquares(rows=particles,columns=particles,replicas=replicas,capacity=capacity,stopping_factor=stopping_factor,jit_compile=jit_compile)
        self.fit=tf.function(self._fit,input_signature=[tf.TensorSpec([observations,replicas,particles],D),
            tf.TensorSpec([observations,replicas],D)],jit_compile=jit_compile,autograph=False)

    def _fit(self,cloud,y):
        arrays=tuple(tf.TensorArray(D,self.h) for _ in range(10))
        counts=tuple(tf.TensorArray(tf.int32,self.h) for _ in range(3))
        def body(t,lookahead,valid,arrays,counts):
            x=cloud[t];centered=x-tf.reduce_mean(x,-1,keepdims=True)
            beta=self.factor*tf.sqrt(tf.reduce_sum(centered**2,-1)/(self.n-1))
            logg=-.5*(tf.math.log(tf.constant(2*math.pi,D)*self.r)+(y[t,:,None]-x*x/20)**2/self.r)
            logtarget=logg+lookahead;constant=tf.reduce_max(logtarget,-1)
            target=tf.exp(logtarget-constant[:,None])
            design=tf.exp(-beta[:,None,None]*(x[:,:,None]-x[:,None,:])**2)
            solved=self.nnls._solve(design,target)
            selected=tf.math.top_k(solved['coefficients'],k=self.m,sorted=True)
            weights=selected.values;knots=tf.gather(x,selected.indices,axis=1,batch_dims=1)
            def previous():
                integral=rbf_gaussian_mixture(multimodal_transition(t,cloud[t-1]),self.q,beta,weights,knots,constant)
                return integral['log_integral'],integral['valid']
            lookahead,integral_ok=tf.cond(t>0,previous,lambda:(tf.zeros_like(x),tf.constant(True)))
            valid=valid&tf.reduce_all(solved['valid'])&integral_ok&tf.reduce_all(tf.math.is_finite(constant))
            valid=valid&tf.reduce_all(tf.reduce_sum(weights,-1)>0)
            values=(beta,weights,knots,constant,solved['coefficients'],target,solved['rss'],
                solved['dual_error'],solved['complementarity'],solved['minimum_qr_margin'])
            arrays=tuple(a.write(t,v) for a,v in zip(arrays,values))
            counts=tuple(a.write(t,v) for a,v in zip(counts,(solved['iterations'],solved['peak_active'],solved['removed'])))
            return t-1,lookahead,valid,arrays,counts
        state=tf.while_loop(lambda t,*_:t>=0,body,
            (tf.constant(self.h-1),tf.zeros([self.b,self.n],D),tf.constant(True),arrays,counts),parallel_iterations=1)
        result={k:a.stack() for k,a in zip(('beta','weights','knots','constant','coefficients','target','rss',
            'dual_error','complementarity','minimum_qr_margin'),state[3])}
        result.update({k:a.stack() for k,a in zip(('iterations','peak_active','removed'),state[4])})
        result['valid']=state[2]
        return result
