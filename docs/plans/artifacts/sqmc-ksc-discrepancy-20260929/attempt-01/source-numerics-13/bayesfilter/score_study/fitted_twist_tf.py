"""Frozen Gaussian-plus-floor psi-APF and offline recursive fitting.

Uses Guarniero--Johansen--Lee equations (5)--(6), Algorithm 3 and the
independent final run of Algorithm 4. Log-quadratic least squares and fixed
iteration count are explicit local adaptations, not their equation (15) fit
or adaptive stopping rule. Fitted psi is frozen when differentiating theta.
"""
from functools import lru_cache
import tensorflow as tf
from .gaussian_tf import (parameterized_model,chol_tangent,gaussian_log_density_and_tangent,symmetric)
from .conditional_means_tf import conditional_mean, conditional_mean_and_tangent, validate_curves
from .complete_data_score_tf import initial_score, increment_score


def gaussian_floor_log(x,dx,center,covariance,log_floor):
    lg,dlg=gaussian_log_density_and_tangent(x-center,dx,covariance,
        tf.zeros([6,1,x.shape[-1],x.shape[-1]],x.dtype))
    total=tf.reduce_logsumexp(tf.stack([lg,tf.fill(tf.shape(lg),log_floor)],-1),-1)
    return total,tf.exp(lg-total)[None,:]*dlg


def normalizer(mean,dmean,Q,dQ,center,V,log_floor):
    lg,dlg=gaussian_log_density_and_tangent(center-mean,-dmean,Q+V,dQ[:,None,:,:])
    total=tf.reduce_logsumexp(tf.stack([lg,tf.fill(tf.shape(lg),log_floor)],-1),-1)
    probability=tf.exp(lg-total)
    return total,probability[None,:]*dlg,probability


def twisted_transition(mean,dmean,Q,dQ,center,V,probability,noise,uniform):
    S=Q+V; L=tf.linalg.cholesky(S)
    K=tf.transpose(tf.linalg.cholesky_solve(L,tf.transpose(Q)))
    dK=tf.linalg.matrix_transpose(tf.linalg.cholesky_solve(L,
        tf.linalg.matrix_transpose(dQ-K@dQ)))
    C=symmetric(Q-K@Q)
    dC=symmetric(dQ-dK@Q-K@dQ)
    LC=tf.linalg.cholesky(C); LQ=tf.linalg.cholesky(Q)
    # Keep the time-dependent center out of the broadcast GEMM operand.
    # GPU/XLA/TF32 can otherwise reuse the first loop iteration's center;
    # the time-varying-center diagnostic reproduces that wrong-result case.
    # K(center-mean) = K center - K mean, including its total tangent.
    adapted=mean-tf.einsum("ij,nj->ni",K,mean)+tf.linalg.matvec(K,center)[None,:]
    adapted+=tf.einsum("ij,nj->ni",LC,noise)
    dadapted=dmean-tf.einsum("ij,pnj->pni",K,dmean)
    dadapted+=tf.einsum("pij,j->pi",dK,center)[:,None,:]-tf.einsum("pij,nj->pni",dK,mean)
    dadapted+=tf.einsum("pij,nj->pni",chol_tangent(LC,dC),noise)
    prior=mean+tf.einsum("ij,nj->ni",LQ,noise)
    dprior=dmean+tf.einsum("pij,nj->pni",chol_tangent(LQ,dQ),noise)
    choose=uniform<probability
    return tf.where(choose[:,None],adapted,prior),tf.where(choose[None,:,None],dadapted,dprior)


def resampling_score_control(additive, indices, cumulative_weights, uniform_bits):
    """Diagnostic martingale control for uniforms k / 2**uniform_bits.

    Count the lattice points selected by the actual right-sided CDF search,
    including the last-index clamp. FP64 accumulation limits cancellation;
    it does not alter the particles or the underlying FP32 score.
    """
    cumulative = tf.cast(cumulative_weights[:-1], tf.float64)
    boundaries = tf.concat([tf.zeros([1], tf.float64),
        tf.clip_by_value(cumulative, 0., 1.), tf.ones([1], tf.float64)], 0)
    lattice = tf.constant(float(2**uniform_bits), tf.float64)
    counts = tf.math.ceil(lattice * boundaries)
    probabilities = (counts[1:] - counts[:-1]) / lattice
    scores = tf.cast(additive, tf.float64)
    selected_mean = tf.reduce_mean(tf.gather(scores, indices, axis=1), axis=1)
    expected_mean = tf.reduce_sum(scores * probabilities[None, :], axis=1)
    return tf.sqrt(tf.cast(tf.shape(indices)[0], tf.float64)) * (selected_mean - expected_mean)


@lru_cache(maxsize=12)
def make_fitted_twist_kernel(d,o,N,T,dtype_name="float64",jit_compile=True,constant_twist=False,
                             transition_curve=0.,observation_curve=0.,include_fisher_score=False,
                             include_resampling_controls=False,resampling_uniform_bits=None,
                             include_numerical_trace=False):
    """Optionally append a terminal Fisher estimate on the same genealogy.

    The existing score differentiates the fixed-label finite value program.
    The extra score estimates the physical-model score by Fisher's identity;
    it is not a gradient of that finite program or finite-N unbiased.
    Optional FP64 controls cover the T nonterminal ancestor draws. Callers
    must supply independent uniforms on the explicitly declared binary grid.
    Numerical traces expose actual ancestor CDFs/indices and Gaussian mixture
    probabilities for precision diagnostics; they do not change decisions.
    """
    validate_curves(d,o,transition_curve,observation_curve)
    dtype=tf.as_dtype(dtype_name)
    if include_resampling_controls:
        if not include_fisher_score:
            raise ValueError("resampling controls require the Fisher score")
        max_bits = 23 if dtype == tf.float32 else 52
        if type(resampling_uniform_bits) is not int or not 1 <= resampling_uniform_bits <= max_bits:
            raise ValueError("resampling controls require a representable uniform lattice")
    elif resampling_uniform_bits is not None:
        raise ValueError("uniform lattice is only used by resampling controls")
    @tf.function(input_signature=[tf.TensorSpec([6],dtype),tf.TensorSpec([T,o],dtype),
        tf.TensorSpec([N,d],dtype),tf.TensorSpec([T,N,d],dtype),tf.TensorSpec([T+1,N],dtype),
        tf.TensorSpec([T,N],dtype),tf.TensorSpec([T,d],dtype),tf.TensorSpec([T,d,d],dtype),
        tf.TensorSpec([T],dtype)],jit_compile=jit_compile)
    def kernel(theta,observations,initial_noise,noise,uniforms,mixture_uniforms,centers,covariances,log_floors):
        model=parameterized_model(theta,d,o)
        A,dA,H,dH,m,dm,P,dP,Q,dQ,R,dR=model
        L=tf.linalg.cholesky(P)
        x=m+tf.einsum("ij,nj->ni",L,initial_noise)
        dx=dm[:,None,:]+tf.einsum("pij,nj->pni",chol_tangent(L,dP),initial_noise)
        def predict(x,dx):
            return conditional_mean_and_tangent(x,dx,A,dA,transition_curve)
        def future(x,dx,t):
            if constant_twist:
                return tf.zeros([N],dtype),tf.zeros([6,N],dtype),tf.zeros([N],dtype)
            return normalizer(*predict(x,dx),Q,dQ,centers[t],covariances[t],log_floors[t])
        def resample(x,dx,logw,dlogw,u):
            w=tf.nn.softmax(logw)
            cumulative=tf.cumsum(w)
            indices=tf.searchsorted(cumulative,u,side="right")
            indices=tf.minimum(indices,N-1)
            return tf.gather(x,indices),tf.gather(dx,indices,axis=1),tf.reduce_logsumexp(logw)-tf.math.log(tf.cast(N,dtype)),tf.reduce_sum(w[None,:]*dlogw,-1),indices,cumulative
        initial_log,initial_dlog,_=future(x,dx,0)
        additive=initial_score(x,model) if include_fisher_score else tf.zeros([6,N],dtype)
        x,dx,total,dtotal,indices,cumulative=resample(x,dx,initial_log,initial_dlog,uniforms[0])
        ancestor_trace=tf.TensorArray(tf.int32,size=T+1,element_shape=[N])
        cdf_trace=tf.TensorArray(dtype,size=T+1,element_shape=[N])
        mixture_trace=tf.TensorArray(dtype,size=T,element_shape=[N])
        if include_numerical_trace:
            ancestor_trace=ancestor_trace.write(0,indices)
            cdf_trace=cdf_trace.write(0,cumulative)
        controls=tf.TensorArray(tf.float64,size=T,element_shape=[6])
        if include_resampling_controls:
            controls=controls.write(0,resampling_score_control(additive,indices,cumulative,resampling_uniform_bits))
        if include_fisher_score:
            additive=tf.gather(additive,indices,axis=1)
        arrays=tf.TensorArray(dtype,size=T,element_shape=[N,d])
        def step(t,x,dx,total,dtotal,clouds,additive,fisher,controls,ancestor_trace,cdf_trace,mixture_trace):
            previous=x
            mean,dmean=predict(x,dx)
            if constant_twist:
                probability=tf.zeros([N],dtype)
                LQ=tf.linalg.cholesky(Q)
                x=mean+tf.einsum("ij,nj->ni",LQ,noise[t])
                dx=dmean+tf.einsum("pij,nj->pni",chol_tangent(LQ,dQ),noise[t])
            else:
                _,_,probability=normalizer(mean,dmean,Q,dQ,centers[t],covariances[t],log_floors[t])
                x,dx=twisted_transition(mean,dmean,Q,dQ,centers[t],covariances[t],probability,noise[t],mixture_uniforms[t])
            clouds=clouds.write(t,x)
            observed,dobserved=conditional_mean_and_tangent(x,dx,H,dH,observation_curve,quadratic=True)
            residual,dr=observations[t]-observed,-dobserved
            lg,dlg=gaussian_log_density_and_tangent(residual,dr,R,dR[:,None,:,:])
            if constant_twist:
                lp,dlp=tf.zeros([N],dtype),tf.zeros([6,N],dtype)
            else:
                lp,dlp=gaussian_floor_log(x,dx,centers[t],covariances[t],log_floors[t])
            lf,dlf=tf.cond(t+1<T,lambda:future(x,dx,t+1)[:2],
                lambda:(tf.zeros([N],dtype),tf.zeros([6,N],dtype)))
            logw=lg+lf-lp
            if include_fisher_score:
                additive+=increment_score(previous,x,observations[t],model,transition_curve,observation_curve)
                # Only the terminal weighted genealogy targets the physical posterior.
                fisher=tf.cond(t+1==T,lambda:tf.reduce_sum(tf.nn.softmax(logw)[None,:]*additive,-1),lambda:fisher)
            x,dx,term,dterm,indices,cumulative=resample(x,dx,logw,dlg+dlf-dlp,uniforms[t+1])
            if include_numerical_trace:
                ancestor_trace=ancestor_trace.write(t+1,indices)
                cdf_trace=cdf_trace.write(t+1,cumulative)
                mixture_trace=mixture_trace.write(t,probability)
            if include_resampling_controls:
                controls=tf.cond(t+1<T,
                    lambda:controls.write(t+1,resampling_score_control(additive,indices,cumulative,resampling_uniform_bits)),
                    lambda:controls)
            if include_fisher_score:
                additive=tf.gather(additive,indices,axis=1)
            return t+1,x,dx,total+term,dtotal+dterm,clouds,additive,fisher,controls,ancestor_trace,cdf_trace,mixture_trace
        result=tf.while_loop(lambda t,*_:t<T,step,
            (0,x,dx,total,dtotal,arrays,additive,tf.zeros([6],dtype),controls,
             ancestor_trace,cdf_trace,mixture_trace),parallel_iterations=1)
        outputs=(result[3],result[4],result[5].stack())
        if include_resampling_controls:
            outputs=(*outputs,result[7],result[8].stack())
        elif include_fisher_score:
            outputs=(*outputs,result[7])
        if include_numerical_trace:
            outputs=(*outputs,{"ancestor_indices":result[9].stack(),
                "ancestor_cdf":result[10].stack(),"gaussian_probability":result[11].stack()})
        return outputs
    return kernel


def quadratic_features(points):
    d=points.shape[-1]
    columns=[tf.ones_like(points[:,0])]+list(tf.unstack(points,axis=1))
    columns += [-.5*points[:,i]**2 for i in range(d)]
    columns += [-points[:,i]*points[:,j] for i in range(d) for j in range(i+1,d)]
    return tf.stack(columns,-1)


def fit_log_quadratic(points,targets,floor_ratio):
    """Full-rank QR fit; reject nonpositive precision instead of clipping it."""
    d=points.shape[-1];dtype=points.dtype
    features=quadratic_features(points)
    scale=tf.maximum(tf.sqrt(tf.reduce_mean(features**2,0)),tf.cast(1.,dtype))
    scaled=features/scale
    singular=tf.linalg.svd(scaled,compute_uv=False)
    eps=tf.cast(2**-23 if dtype==tf.float32 else 2**-52,dtype)
    rank_valid=tf.reduce_min(singular)>eps*tf.cast(max(scaled.shape),dtype)*singular[0]
    q,r=tf.linalg.qr(scaled,full_matrices=False)
    coefficient=tf.linalg.triangular_solve(r,tf.transpose(q)@targets[:,None],lower=False)[:,0]/scale
    precision=tf.linalg.diag(coefficient[1+d:1+2*d])
    pairs=[(i,j) for i in range(d) for j in range(i+1,d)]
    if pairs:
        indices=pairs+[(j,i) for i,j in pairs]
        values=tf.concat([coefficient[1+2*d:],coefficient[1+2*d:]],0)
        precision=tf.tensor_scatter_nd_add(precision,indices,values)
    eigenvalues=tf.linalg.eigvalsh(precision)
    valid=rank_valid & (tf.reduce_min(eigenvalues)>eps*tf.cast(d,dtype)*tf.reduce_max(tf.abs(eigenvalues)))
    factor=tf.linalg.cholesky(precision)
    covariance=tf.linalg.cholesky_solve(factor,tf.eye(d,dtype=dtype))
    center=tf.linalg.cholesky_solve(factor,coefficient[1:1+d,None])[:,0]
    log_peak=tf.reduce_sum(tf.math.log(tf.linalg.diag_part(factor)))-.5*tf.cast(d,dtype)*tf.math.log(2*tf.acos(tf.cast(-1.,dtype)))
    log_floor=tf.math.log(tf.cast(floor_ratio,dtype))+log_peak
    rmse=tf.sqrt(tf.reduce_mean((features@coefficient[:,None]-targets[:,None])**2))
    return tuple(tf.where(valid,x,tf.cast(float("nan"),dtype)) for x in (center,covariance,log_floor)),(valid,rmse,singular)


@lru_cache(maxsize=12)
def make_recursive_fit_kernel(d,o,N,T,floor_ratio,dtype_name="float64",jit_compile=True,
                              transition_curve=0.,observation_curve=0.):
    validate_curves(d,o,transition_curve,observation_curve)
    if not 0<floor_ratio<1: raise ValueError("declare a positive floor ratio below one")
    feature_count=1+2*d+d*(d-1)//2
    if N<feature_count: raise ValueError("too few fit points for full quadratic")
    dtype=tf.as_dtype(dtype_name)
    @tf.function(input_signature=[tf.TensorSpec([6],dtype),tf.TensorSpec([T,o],dtype),
        tf.TensorSpec([T,N,d],dtype)],jit_compile=jit_compile)
    def fit(theta,observations,clouds):
        A,_,H,_,_,_,_,_,Q,_,R,_=parameterized_model(theta,d,o)
        centers=tf.TensorArray(dtype,size=T); covariances=tf.TensorArray(dtype,size=T)
        floors=tf.TensorArray(dtype,size=T); errors=tf.TensorArray(dtype,size=T)
        def step(t,center,V,log_floor,valid,centers,covariances,floors,errors):
            x=clouds[t]
            residual=observations[t]-conditional_mean(x,H,observation_curve,quadratic=True)
            lg,_=gaussian_log_density_and_tangent(residual,tf.zeros([6,N,o],dtype),R,tf.zeros([6,1,o,o],dtype))
            lf=tf.cond(t+1<T,lambda:normalizer(conditional_mean(x,A,transition_curve),tf.zeros([6,N,d],dtype),
                Q,tf.zeros([6,d,d],dtype),center,V,log_floor)[0],lambda:tf.zeros([N],dtype))
            (center,V,log_floor),(ok,error,_)=fit_log_quadratic(x,lg+lf,floor_ratio)
            return t-1,center,V,log_floor,valid&ok,centers.write(t,center),covariances.write(t,V),floors.write(t,log_floor),errors.write(t,error)
        out=tf.while_loop(lambda t,*_:t>=0,step,(T-1,tf.zeros([d],dtype),tf.eye(d,dtype=dtype),
            tf.zeros([],dtype),tf.constant(True),centers,covariances,floors,errors),parallel_iterations=1)
        return out[5].stack(),out[6].stack(),out[7].stack(),out[4],out[8].stack()
    return fit
