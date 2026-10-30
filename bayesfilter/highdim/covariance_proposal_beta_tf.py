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


def pilot_log_terms(log_components, log_target):
    log_mixture = tf.reduce_logsumexp(log_components, axis=-1) - tf.math.log(tf.cast(3, log_components.dtype))
    log_weights = log_target - log_mixture
    log_scale = tf.reduce_logsumexp(log_weights, axis=-1, keepdims=True) - tf.math.log(tf.cast(tf.shape(log_weights)[-1], log_weights.dtype))
    return log_components - log_mixture[..., None], 2 * (log_weights - log_scale) - tf.math.log(tf.cast(tf.size(log_weights), log_weights.dtype))


def pilot_log_objective(beta, log_ratios, log_lambdas):
    log_denominator = tf.reduce_logsumexp(log_ratios + tf.math.log(beta), axis=-1)
    value = tf.exp(tf.reduce_logsumexp(log_lambdas - log_denominator))
    gradient = -tf.exp(tf.reduce_logsumexp(log_lambdas[..., None] + log_ratios - 2 * log_denominator[..., None], axis=[0, 1]))
    return value, gradient


def make_pairwise_beta_solver(contexts, samples, dtype=tf.float64, jit_compile=True, max_steps=2000, max_line_steps=512):
    """Optional exact-objective calibration with pairwise simplex line searches."""
    dtype = tf.as_dtype(dtype)
    signature = [tf.TensorSpec([contexts, samples, 3], dtype), tf.TensorSpec([contexts, samples], dtype), tf.TensorSpec([], dtype), tf.TensorSpec([], dtype)]

    @tf.function(input_signature=signature, jit_compile=jit_compile, autograph=False)
    def fit(log_components, log_target, floor, tolerance):
        log_ratios, log_lambdas = pilot_log_terms(log_components, log_target)
        lower = tf.stack([floor, tf.zeros_like(floor), tf.zeros_like(floor)])
        beta = lower + (1 - floor) * tf.fill([3], tf.constant(1 / 3, dtype))
        objective, gradient = pilot_log_objective(beta, log_ratios, log_lambdas)
        gap = gap_certificate(beta, gradient, floor)
        valid = (floor > 0) & (floor < 1) & (tolerance > 0) & tf.math.is_finite(tolerance) & tf.reduce_all(tf.math.is_finite(log_ratios)) & tf.reduce_all(tf.math.is_finite(log_lambdas))

        def body(iteration, beta, objective, gradient, gap, valid):
            excess = beta - lower
            pair = tf.gather(tf.constant([[0, 1], [0, 2], [1, 2]], tf.int32), iteration % 3)
            first_receives = tf.gather(gradient, pair[0]) < tf.gather(gradient, pair[1])
            receiver = tf.where(first_receives, pair[0], pair[1])
            donor = tf.where(first_receives, pair[1], pair[0])
            direction = tf.one_hot(receiver, 3, dtype=dtype) - tf.one_hot(donor, 3, dtype=dtype)
            maximum = tf.gather(excess, donor)
            endpoint = lower + tf.where(tf.one_hot(donor, 3, on_value=True, off_value=False), tf.zeros_like(excess), excess + maximum * direction)
            endpoint_objective, endpoint_gradient = pilot_log_objective(endpoint, log_ratios, log_lambdas)
            endpoint_slope = tf.gather(endpoint_gradient, receiver) - tf.gather(endpoint_gradient, donor)

            def interior():
                def search_condition(count, left, right, candidate, value, slope):
                    return (count < max_line_steps) & (tf.abs(slope) > tolerance * tf.maximum(tf.constant(1., dtype), objective) / 4) & (right > left)

                def search_body(count, left, right, candidate, value, slope):
                    midpoint = left + (right - left) / 2
                    candidate = lower + (excess + midpoint * direction)
                    value, candidate_gradient = pilot_log_objective(candidate, log_ratios, log_lambdas)
                    slope = tf.gather(candidate_gradient, receiver) - tf.gather(candidate_gradient, donor)
                    stalled = (midpoint == left) | (midpoint == right)
                    next_left = tf.where(slope < 0, midpoint, left)
                    next_right = tf.where(slope > 0, midpoint, right)
                    next_left = tf.where(stalled, next_right, next_left)
                    return count + 1, next_left, next_right, candidate, value, slope

                result = tf.while_loop(search_condition, search_body, (0, tf.zeros_like(maximum), maximum, beta, objective, tf.gather(gradient, receiver) - tf.gather(gradient, donor)), parallel_iterations=1)
                return result[3], result[4]

            next_beta, next_objective = tf.cond(endpoint_slope <= 0, lambda: (endpoint, endpoint_objective), interior)
            next_objective, next_gradient = pilot_log_objective(next_beta, log_ratios, log_lambdas)
            next_gap = gap_certificate(next_beta, next_gradient, floor)
            roundoff = tf.constant(1e-13 if dtype == tf.float64 else 1e-5, dtype) * tf.maximum(tf.constant(1., dtype), tf.abs(objective))
            next_valid = valid & tf.math.is_finite(next_objective) & tf.reduce_all(tf.math.is_finite(next_gradient)) & (next_objective <= objective + roundoff) & tf.reduce_all(next_beta >= lower)
            return iteration + 1, next_beta, next_objective, next_gradient, next_gap, next_valid

        result = tf.while_loop(lambda iteration, beta, objective, gradient, gap, valid: (iteration < max_steps) & valid & (gap > tolerance * tf.maximum(tf.constant(1., dtype), objective)), body, (0, beta, objective, gradient, gap, valid), parallel_iterations=1)
        iteration, beta, objective, gradient, gap, valid = result
        return dict(beta=beta, objective=objective, gap=gap, iterations=iteration, converged=valid & (gap <= tolerance * tf.maximum(tf.constant(1., dtype), objective)), ratios=tf.exp(log_ratios), lambdas=tf.exp(log_lambdas))

    return fit
