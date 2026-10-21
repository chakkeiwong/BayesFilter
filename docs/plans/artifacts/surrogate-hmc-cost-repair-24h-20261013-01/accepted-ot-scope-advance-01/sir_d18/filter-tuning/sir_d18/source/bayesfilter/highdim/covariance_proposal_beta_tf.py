"""Algorithm 0: independent-pilot convex beta calibration, TensorFlow only."""
import tensorflow as tf

def project_simplex(value,floor):
    lower=tf.stack([floor,tf.zeros_like(floor),tf.zeros_like(floor)])
    z=value-lower; radius=1-floor
    u=tf.sort(z,direction="DESCENDING")
    thresholds=(tf.cumsum(u)-radius)/tf.cast(tf.range(1,4),value.dtype)
    rho=tf.reduce_sum(tf.cast(u>thresholds,tf.int32))-1
    tau=tf.gather(thresholds,tf.maximum(rho,0))
    return lower+tf.maximum(z-tau,0)

def pilot_objective(beta,ratios,lambdas):
    den=tf.einsum('cmb,b->cm',ratios,beta)
    value=tf.reduce_sum(lambdas/den)
    gradient=-tf.einsum('cm,cmb->b',lambdas/tf.square(den),ratios)
    hessian=2*tf.einsum('cm,cmi,cmj->ij',lambdas/(den**3),ratios,ratios)
    return value,gradient,hessian

def gap_certificate(beta,gradient,floor):
    vertex=tf.stack([floor,tf.zeros_like(floor),tf.zeros_like(floor)])+(1-floor)*tf.one_hot(tf.argmin(gradient),3,dtype=beta.dtype)
    return tf.reduce_sum(gradient*(beta-vertex))

def pilot_ratios(log_components,log_target):
    logh=tf.reduce_logsumexp(log_components,axis=-1)-tf.math.log(tf.cast(3,log_components.dtype))
    logw=log_target-logh
    scale=tf.reduce_logsumexp(logw,axis=-1,keepdims=True)-tf.math.log(tf.cast(tf.shape(logw)[-1],logw.dtype))
    return tf.exp(log_components-logh[...,None]),tf.square(tf.exp(logw-scale))/tf.cast(tf.size(logw),logw.dtype)

def make_beta_solver(contexts,samples,dtype=tf.float64,jit_compile=True,max_steps=2000,max_backtracks=40):
    dtype=tf.as_dtype(dtype)
    @tf.function(input_signature=[tf.TensorSpec([contexts,samples,3],dtype),tf.TensorSpec([contexts,samples],dtype),tf.TensorSpec([],dtype),tf.TensorSpec([],dtype)],jit_compile=jit_compile,autograph=False)
    def fit(log_components,log_target,floor,tolerance):
        ratios,lambdas=pilot_ratios(log_components,log_target)
        beta=tf.stack([floor,tf.zeros_like(floor),tf.zeros_like(floor)])+(1-floor)*tf.fill([3],tf.constant(1/3,dtype))
        f,g,_=pilot_objective(beta,ratios,lambdas)
        gap=gap_certificate(beta,g,floor)
        valid=(floor>0)&(floor<1)&(tolerance>0)&tf.math.is_finite(tolerance)&tf.reduce_all(tf.math.is_finite(ratios))&tf.reduce_all(tf.math.is_finite(lambdas))
        def body(k,beta,f,g,gap,valid,last_eta):
            def trial(eta):
                b=project_simplex(beta-eta*g,floor)
                nf,ng,_=pilot_objective(b,ratios,lambdas)
                delta=b-beta
                bound=f+tf.reduce_sum(g*delta)+tf.reduce_sum(delta*delta)/(2*eta)
                ok=tf.math.is_finite(nf)&(nf<=bound+tf.constant(1e-14 if dtype==tf.float64 else 1e-6,dtype)*tf.maximum(tf.constant(1.,dtype),tf.abs(f)))
                return b,nf,ng,ok
            eta=tf.constant(1.,dtype)
            nb,nf,ng,ok=trial(eta)
            def back(k,eta,nb,nf,ng,ok):
                eta=eta/2; nb,nf,ng,ok=trial(eta)
                return k+1,eta,nb,nf,ng,ok
            _,eta,nb,nf,ng,ok=tf.while_loop(lambda k,*a:(k<max_backtracks)&~a[-1],back,(0,eta,nb,nf,ng,ok),parallel_iterations=1)
            ngap=gap_certificate(nb,ng,floor)
            return k+1,nb,nf,ng,ngap,valid&ok,eta
        k,beta,f,g,gap,valid,eta=tf.while_loop(lambda k,b,f,g,gap,v,e:(k<max_steps)&v&(gap>tolerance*tf.maximum(tf.constant(1.,dtype),f)),body,(0,beta,f,g,gap,valid,tf.constant(1.,dtype)),parallel_iterations=1)
        return dict(beta=beta,objective=f,gap=gap,iterations=k,converged=valid&(gap<=tolerance*tf.maximum(tf.constant(1.,dtype),f)),ratios=ratios,lambdas=lambdas)
    return fit
